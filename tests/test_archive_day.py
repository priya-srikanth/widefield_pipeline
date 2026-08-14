"""Tests for the end-of-day archival tool's DAQ discovery (E: -> N: destinations).

The canonical server layout is ``DAQ_recorder_output/<date>/<animal>_<date>_<time>.h5``; the E:
layout is not canonical (files sit loose, or under a date dir, sometimes a typo'd one), so the
destination must be derived from ``--date``, never from the E: parent folder name.
"""
from pathlib import Path

from wfield_local import archive_day


def _cfg(tmp_path):
    return dict(archive_day.DEFAULTS,
                e_daq=str(tmp_path / "E" / "DAQ_recorder_output"),
                n_daq=str(tmp_path / "N" / "DAQ_recorder_output"))


def _touch(path: Path, size: int = 16):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\0" * size)


def test_daq_dst_is_date_folder_for_every_e_layout(tmp_path):
    cfg = _cfg(tmp_path)
    e, n = Path(cfg["e_daq"]), Path(cfg["n_daq"])
    _touch(e / "PS92_20260808_124733.h5")                    # loose at the DAQ root
    _touch(e / "20260808" / "PS93_20260808_172316.h5")       # already under its date dir
    _touch(e / "20250808" / "PS94_20260808_074558.h5")       # typo'd date dir, filename correct
    dsts = sorted(Path(j["dst"]) for j in archive_day.discover_daq(cfg, "20260808"))
    assert dsts == sorted([n / "20260808" / f"{a}_20260808_{t}.h5"
                           for a, t in (("PS92", "124733"), ("PS93", "172316"), ("PS94", "074558"))])
    # the regression this guards: a loose E: file must NOT nest DAQ_recorder_output under itself
    assert not any("DAQ_recorder_output" in Path(d).parent.name for d in dsts)


def test_daq_discovery_filters_by_date_and_extension(tmp_path):
    cfg = _cfg(tmp_path)
    e = Path(cfg["e_daq"])
    _touch(e / "PS92_20260808_124733.h5")
    _touch(e / "PS92_20260807_151146.h5")     # different date -> not this night's job
    _touch(e / "PS92_20260808_124733.txt")    # not an .h5
    (names,) = [[Path(j["src"]).name for j in archive_day.discover_daq(cfg, "20260808")]]
    assert names == ["PS92_20260808_124733.h5"]


def test_daq_discovery_empty_when_e_daq_absent(tmp_path):
    cfg = dict(_cfg(tmp_path), e_daq=str(tmp_path / "nope"))
    assert archive_day.discover_daq(cfg, "20260808") == []


def _repaired_session(tmp_path, orig_frames=100, h=4, w=5,
                      orig_name="pco_edge_run000_00000000_1_4_5_uint16.dat"):
    """E: tree holding a REPAIRED .dat + its manifest, and an M: tree holding the ORIGINAL."""
    import json

    e = tmp_path / "E" / "20260813" / "PS95_x" / "raw_widefield_data"
    e.mkdir(parents=True)
    (e / "pco_2_4_5_uint16.dat").write_bytes(b"\0" * 64)
    # Keys mirror the REAL repair_manifest.json. It records no frame geometry -- that comes from the
    # labcams filename -- and an earlier version of this fixture invented an output_shape key, so the
    # test passed while the live run raised KeyError and silently refused every delete.
    (e / "repair_manifest.json").write_text(json.dumps({
        "source_dat": f"N:/whatever/{orig_name}",
        "source_h5": "N:/whatever/x.h5",
        "n_frames_in": orig_frames,
        "n_pairs_out": 10,
        "split_frame": 3,
    }))
    m = tmp_path / "M" / "20260813" / "PS95_x" / "raw_widefield_data"
    m.mkdir(parents=True)
    return e, m, orig_name, orig_frames * h * w * 2


def test_repaired_dat_is_an_intermediate_not_a_raw(tmp_path):
    """It is a rebuild of an acquisition, not one. Archiving it would file a derivative in the raw
    archive under a name indistinguishable from an original."""
    from wfield_local import archive_day as ad

    e, m, _name, _sz = _repaired_session(tmp_path)
    cfg = dict(e_lab=str(tmp_path / "E"), m_raw=str(tmp_path / "M"), n_lab=str(tmp_path / "N"),
               e_daq=str(tmp_path / "Ed"), n_daq=str(tmp_path / "Nd"))
    jobs, inter, _daq = ad.discover(cfg, "20260813")
    kinds = {j["kind"] for j in inter}
    assert "repaired_raw" in kinds, "a .dat beside repair_manifest.json must be an intermediate"
    assert not [j for j in jobs if j["kind"] == "raw"], "and must NOT be copied to the raw archive"


def test_repaired_dat_is_only_deletable_once_the_ORIGINAL_is_on_standby(tmp_path):
    """The normal 'a raw for this session was copied from E:' test cannot apply -- the original was
    never staged locally. Deleting on that basis would drop the only local copy of a session whose
    source might not be archived at all."""
    from wfield_local import archive_day as ad

    e, m, name, size = _repaired_session(tmp_path)
    cfg = dict(e_lab=str(tmp_path / "E"), m_raw=str(tmp_path / "M"), n_lab=str(tmp_path / "N"),
               e_daq=str(tmp_path / "Ed"), n_daq=str(tmp_path / "Nd"))
    _jobs, inter, _daq = ad.discover(cfg, "20260813")
    j = next(x for x in inter if x["kind"] == "repaired_raw")

    assert not ad._original_raw_archived(cfg, "20260813", j), "no original on M: -> refuse"
    (m / name).write_bytes(b"\0" * (size - 1))
    assert not ad._original_raw_archived(cfg, "20260813", j), "wrong size -> refuse"
    (m / name).write_bytes(b"\0" * size)
    assert ad._original_raw_archived(cfg, "20260813", j), "correct original present -> allow"


def test_clean_refuses_to_delete_raw_that_is_STAGED_but_not_yet_preprocessed(tmp_path, monkeypatch):
    """Re-staging raw from the archive to reprocess it makes it look deletable: it IS archived, by
    definition. Deleting it throws away a 40-minute network copy of data not yet used. Only a session
    with SVD output has actually spent its raw."""
    from wfield_local import archive_day as ad

    e = tmp_path / "E" / "20260813" / "PS92_x" / "raw_widefield_data"
    e.mkdir(parents=True)
    raw = e / "pco_edge_run000_00000000_2_4_5_uint16.dat"
    raw.write_bytes(b"\0" * 128)
    m = tmp_path / "M" / "20260813" / "PS92_x" / "raw_widefield_data"
    m.mkdir(parents=True)
    (m / raw.name).write_bytes(b"\0" * 128)              # archived, so "copied" is satisfied
    n = tmp_path / "N"
    cfg = dict(e_lab=str(tmp_path / "E"), m_raw=str(tmp_path / "M"), n_lab=str(n),
               e_daq=str(tmp_path / "Ed"), n_daq=str(tmp_path / "Nd"))

    assert not ad._session_processed(cfg, "20260813", "PS92_x"), "no SVTcorr yet -> staged"
    res = (n / "20260813" / "PS92_x" / "motion_corrected" / "wfield_local_results")
    res.mkdir(parents=True)
    (res / "SVTcorr.npy").write_bytes(b"\0" * 16)
    assert ad._session_processed(cfg, "20260813", "PS92_x"), "SVD output present -> spent"
