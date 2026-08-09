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
