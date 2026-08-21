"""Tests for the config-driven preprocessing orchestrator (discovery + reference derivation).

Discovery is exercised against a synthetic raw tree mimicking the messy real E: layouts
(raw_widefield_data subdir vs not; DAQ under a date-dir, flat, or a typo'd date-dir).
"""
from pathlib import Path

from wfield_local import config, preprocess
from wfield_local.paths import PathResolver


def _touch(path: Path, size: int = 16):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\0" * size)


def _build_raw_tree(tmp_path):
    raw = tmp_path / "labcams_data"
    daq = tmp_path / "DAQ_recorder_output"
    d = "20260808"
    # PS94: standard raw_widefield_data/ layout
    _touch(raw / d / "PS94_20260808_104125" / "raw_widefield_data"
           / "pco_edge_run000_00000000_2_460_480_uint16.dat", 999)
    # PS95: extra nesting (animal/session/raw_widefield_data)
    _touch(raw / d / "PS95" / "PS95_20260808_124637" / "raw_widefield_data"
           / "pco_edge_run000_00000000_2_462_464_uint16.dat", 500)
    # PS92: .dat directly under the session dir (no raw_widefield_data)
    _touch(raw / d / "PS92_20260808" / "pco_edge_run000_00000000_2_540_640_uint16.dat", 700)
    # DAQ: under matching date-dir / flat baseline name / typo'd date-dir
    _touch(daq / d / "PS94_20260808_105410.h5")
    _touch(daq / "PS95_baseline_20260808_125106.h5")
    _touch(daq / "20250808" / "PS92_20260808_151146.h5")   # date-DIR typo, filename correct
    return str(raw), str(daq)


def test_discover_various_layouts_and_daq_matching(tmp_path):
    raw, daq = _build_raw_tree(tmp_path)
    found = {s["animal"]: s for s in preprocess._discover("20260808", raw, daq)}
    assert set(found) == {"PS92", "PS94", "PS95"}
    assert found["PS94"]["dims"] == "2_460_480"
    assert found["PS95"]["dims"] == "2_462_464"
    assert found["PS92"]["dims"] == "2_540_640"
    assert found["PS94"]["sess"] == "PS94_20260808_104125"
    assert found["PS95"]["sess"] == "PS95_20260808_124637"
    assert found["PS92"]["sess"] == "PS92_20260808"
    # every session matched a DAQ .h5 (incl. flat baseline + typo'd date-dir)
    for a in ("PS92", "PS94", "PS95"):
        assert found[a]["daq_h5"] and found[a]["daq_h5"].endswith(".h5")
    assert found["PS92"]["daq_h5"].endswith("PS92_20260808_151146.h5")
    assert found["PS95"]["daq_h5"].endswith("PS95_baseline_20260808_125106.h5")


def test_discover_picks_largest_dat_per_session(tmp_path):
    raw = tmp_path / "labcams_data"
    daq = tmp_path / "DAQ_recorder_output"
    sess = raw / "20260808" / "PS94_20260808_104125" / "raw_widefield_data"
    _touch(sess / "pco_edge_run000_00000000_2_460_480_uint16.dat", 100)
    _touch(sess / "pco_edge_run001_00000000_2_460_480_uint16.dat", 9000)  # bigger -> chosen
    _touch(daq / "20260808" / "PS94_20260808_105410.h5")
    (s,) = preprocess._discover("20260808", str(raw), str(daq))
    assert s["raw_dat"].endswith("run001_00000000_2_460_480_uint16.dat")
    assert s["n_dats"] == 2


def test_discover_empty_when_date_absent(tmp_path):
    assert preprocess._discover("20991231", str(tmp_path), str(tmp_path)) == []


def test_discover_processed_from_microscope(tmp_path, monkeypatch):
    """Processed-session discovery (for --skip-preprocess when raw is archived off E:)."""
    lab = tmp_path / "labcams"
    daq = tmp_path / "DAQ_recorder_output"
    d = "20260807"
    # a processed session = motion_corrected/ holding a *cleanpairs_frame_map.npz
    _touch(lab / d / "PS92_20260807_150924" / "motion_corrected"
           / "pco_edge_run000_00000000_2_460_480_uint16_daq_led_cleanpairs_frame_map.npz")
    # an un-processed session dir (no frame_map) must be skipped
    (lab / d / "PS93_20260807_174403" / "motion_corrected").mkdir(parents=True)
    _touch(daq / d / "PS92_20260807_151146.h5")
    rv = PathResolver(machine="imaging")
    monkeypatch.setattr(rv, "root", lambda name: {
        "labcams": str(lab), "daq_recorder_output": str(daq)}.get(name, PathResolver.root(rv, name)))
    found = preprocess.discover_processed_sessions(d, rv)
    assert [s["animal"] for s in found] == ["PS92"]          # PS93 skipped (no frame_map)
    (s,) = found
    assert s["sess"] == "PS92_20260807_150924"
    assert s["dims"] == "2_460_480"                          # parsed from the frame_map name
    assert s["raw_dat"] is None                              # raw archived -> photobleach skips it
    assert s["daq_h5"].endswith("PS92_20260807_151146.h5")   # matched from the N: DAQ root


def test_reference_for_derived_from_config():
    rv = PathResolver(machine="imaging")
    params = config.defaults()["preprocess"]
    ref_date, results, landmarks = preprocess.reference_for("PS92", params, rv)
    assert ref_date == "0606"
    assert results.startswith("N:/") and results.endswith(
        "20260606/PS92_20260606_122451/motion_corrected/wfield_local_results")
    assert landmarks.endswith("dorsal_cortex_landmarks_v2.json")  # PS92 -> v2
    _, _, lm95 = preprocess.reference_for("PS95", params, rv)
    assert lm95.endswith("dorsal_cortex_landmarks_v1.json")       # PS95 -> v1


def test_photobleach_importable_and_parameterized():
    from wfield_local import photobleach as pb
    assert callable(pb.analyze) and callable(pb.summary) and callable(pb.run)
    # no module-level OUT/SESSIONS constants leaked back in
    assert not hasattr(pb, "OUT") and not hasattr(pb, "SESSIONS")


# --------------------------------------------------------------------------- activity maps
def _maps_session():
    return dict(animal="PS92", mmdd="0808", sess="PS92_20260808_120000",
                sess_dir="ignored", raw_dat="ignored", daq_h5="E:/DAQ/PS92_20260808_121000.h5",
                dims="2_460_480")


def test_maps_commands_order_and_windows():
    rv = PathResolver(machine="imaging")
    params = config.defaults()["preprocess"]
    cmds = preprocess._maps_commands(_maps_session(), params, rv, allow_missing=True)
    # the cue/lick/quiet chain, then canonical events -> quiet/running SVD maps
    assert [c[0] for c in cmds] == [
        "wfield_local.framemap_event_maps",
        "wfield_local.plot_spout_trial_averages_shared_scale",
        "wfield_local.plot_spout_position_contrasts",
        "wfield_local.framemap_event_maps",
        "wfield_local.plot_lick_position_contrasts",
        "wfield_local.plot_lick_vs_cue_spout_maps",
        "wfield_local.quiet_periods",
        "wfield_local.framemap_event_maps",
        "wfield_local.behavior_events",
        "wfield_local.plot_running_activity_maps",
    ]
    # behavior_events (per animal+date) precedes the running/quiet maps that consume it
    assert cmds[8][1:] == ["20260808", "--only", "PS92"]
    assert "--events" in cmds[9] and cmds[9][cmds[9].index("--events") + 1].endswith("PS92/20260808.npz")
    # cue map (cmd 1) vs lick maps (cmds 4 & 8) selected via --what
    assert cmds[0][cmds[0].index("--what") + 1] == "cue"
    assert cmds[3][cmds[3].index("--what") + 1] == "lick"
    assert cmds[7][cmds[7].index("--what") + 1] == "lick"
    # windows come from defaults.yaml preprocess.maps
    assert cmds[0][cmds[0].index("--pre-s") + 1] == "2.0"
    assert cmds[0][cmds[0].index("--post-s") + 1] == "2.0"
    assert cmds[3][cmds[3].index("--post-s") + 1] == "0.15"
    assert cmds[7][cmds[7].index("--post-s") + 1] == "0.15"
    # cmd 8 is the quiet-gated lick pass
    assert "--quiet-frame" in cmds[7]
    # labels + N: figure I/O (every command except behavior_events, which is keyed by animal+date)
    for c in cmds:
        if c[0] != "wfield_local.behavior_events":
            assert "PS92_0808_affine8v1" in c
    assert any(a.startswith("N:/") and a.endswith("motion_corrected/wfield_local_results")
               for a in cmds[0])
    # dry-run placeholder frame_map appears (no N: access on this box)
    fm = cmds[0][cmds[0].index("--frame-map") + 1]
    assert fm.endswith("<cleanpairs_frame_map.npz>")


def test_maps_commands_bt_discovered_by_glob(tmp_path, monkeypatch):
    rv = PathResolver(machine="imaging")
    params = config.defaults()["preprocess"]
    blogs = tmp_path / "Behavior_logs"
    _touch(blogs / "PS92_20260808_121500" / "trials.csv")
    monkeypatch.setattr(rv, "root", lambda name: str(blogs) if name == "behavior_logs"
                        else PathResolver.root(rv, name))
    cmds = preprocess._maps_commands(_maps_session(), params, rv, allow_missing=True)
    for i in (0, 3, 7):                      # cmds 1/4/8 carry the discovered trials.csv
        assert "--behavior-trials" in cmds[i]
        assert cmds[i][cmds[i].index("--behavior-trials") + 1].endswith("trials.csv")


def test_maps_commands_bt_absent_when_not_discoverable(tmp_path, monkeypatch):
    rv = PathResolver(machine="imaging")
    params = config.defaults()["preprocess"]
    blogs = tmp_path / "Behavior_logs"      # empty tree -> no trials.csv
    blogs.mkdir()
    monkeypatch.setattr(rv, "root", lambda name: str(blogs) if name == "behavior_logs"
                        else PathResolver.root(rv, name))
    cmds = preprocess._maps_commands(_maps_session(), params, rv, allow_missing=True)
    for i in (0, 3, 7):
        assert "--behavior-trials" not in cmds[i]


def test_maps_commands_bt_prefers_explicit_session_key(tmp_path, monkeypatch):
    rv = PathResolver(machine="imaging")
    params = config.defaults()["preprocess"]
    blogs = tmp_path / "Behavior_logs"      # a glob-discoverable CSV also exists...
    _touch(blogs / "PS92_20260808_121500" / "trials.csv")
    monkeypatch.setattr(rv, "root", lambda name: str(blogs) if name == "behavior_logs"
                        else PathResolver.root(rv, name))
    sess = _maps_session()
    sess["behavior_trials"] = "M:/recovered/PS92_20260808/trials.csv"   # ...but explicit wins
    cmds = preprocess._maps_commands(sess, params, rv, allow_missing=True)
    for i in (0, 3, 7):
        assert cmds[i][cmds[i].index("--behavior-trials") + 1] == \
            "M:/recovered/PS92_20260808/trials.csv"


def test_maps_commands_raises_without_frame_map_when_not_dry_run():
    import pytest
    rv = PathResolver(machine="imaging")
    params = config.defaults()["preprocess"]
    with pytest.raises(SystemExit):        # no N: frame_map + allow_missing=False -> hard stop
        preprocess._maps_commands(_maps_session(), params, rv, allow_missing=False)


# ------------------------------------------------------------------------------------------------
# The push step must never destroy the results it is meant to publish.
#
# `push` is rmtree(destination) then copytree(source -> destination). That is right when raw was
# staged locally and outputs live on E:. It is catastrophic when raw is discovered ON MICROSCOPE
# (--raw-root, added 2026-08-20 because that night's raw arrived there with nothing on E:), because
# source and destination are then the SAME directory: the rmtree deletes hours of computation and
# the copytree then fails with its source gone. Caught by dry-running before the first real run.
# ------------------------------------------------------------------------------------------------

def test_push_is_skipped_when_source_and_destination_are_the_same(tmp_path, capsys):
    import inspect

    from wfield_local import preprocess

    src = inspect.getsource(preprocess)
    assert "_same = (Path(results).resolve() == Path(nres).resolve())" in src, (
        "the push step must compare resolved source and destination before rmtree")
    i = src.index("_same = (Path(results)")
    guarded = src[i:i + 900]
    assert "if not dry_run and not _same:" in guarded, (
        "rmtree/copytree must be gated on the two paths differing")


def test_discovery_roots_can_be_overridden(tmp_path):
    """--raw-root exists so a night whose raw landed on the share can still be processed, with the
    outputs still written beside the raw."""
    import inspect

    from wfield_local import preprocess

    sig = inspect.signature(preprocess.discover_raw_sessions)
    assert "raw_root" in sig.parameters and "daq_root" in sig.parameters


def test_discovery_finds_nothing_rather_than_guessing(tmp_path):
    """An empty root returns [] -- the 2026-08-20 symptom was discovery correctly reporting zero
    sessions, not discovery silently inventing them somewhere else."""
    from wfield_local.preprocess import _discover

    assert _discover("20260820", str(tmp_path / "nope"), str(tmp_path)) == []


# ------------------------------------------------------------------------------------------------
# raw_integrity: measure against the right block, and only after the file stops moving.
#
# On 2026-08-20 I called PS94's upload truncated. Twice wrong: the .dat is a sequence of SINGLE
# h x w frames (the "2" in 2_460_480 is the channel pairing, not the file's atomic unit), and a
# recording that stops on an odd frame leaves a half pair, which is normal. The file was also still
# being written, so the size I read was a partial flush 256 B past a frame boundary.
# ------------------------------------------------------------------------------------------------

def _fake_raw(tmp_path, frames, extra=0, camlog_rows=None, h=4, w=5):
    d = tmp_path / "raw_widefield_data"
    d.mkdir(parents=True, exist_ok=True)
    dat = d / "pco_edge_run000_00000000_2_4_5_uint16.dat"
    dat.write_bytes(b"\0" * (frames * h * w * 2 + extra))
    (d / "pco_edge_run000_00000000.camlog").write_text(
        "".join(f"{i},0,0\n" for i in range(frames if camlog_rows is None else camlog_rows)))
    return dat


def test_an_odd_frame_count_is_normal_not_truncation(tmp_path):
    """The exact case I got wrong: a complete file whose last 415/470 pair is incomplete."""
    from wfield_local.preprocess import raw_integrity

    r = raw_integrity(_fake_raw(tmp_path, frames=373833 % 1000 * 2 + 1), "2_4_5", settle_s=0)
    assert r["ok"], "an odd frame count is where the recording stopped, not corruption"
    assert r["odd_last_pair"] and r["remainder"] == 0


def test_a_file_cut_mid_frame_is_refused(tmp_path):
    from wfield_local.preprocess import raw_integrity

    r = raw_integrity(_fake_raw(tmp_path, frames=100, extra=17), "2_4_5", settle_s=0)
    assert not r["ok"] and "partial upload" in r["reason"]


def test_a_file_still_growing_is_refused(tmp_path, monkeypatch):
    """The other half of my mistake: judging a size that was still being written."""
    from wfield_local import preprocess

    dat = _fake_raw(tmp_path, frames=100)

    real_sleep = preprocess.time.sleep if hasattr(preprocess, "time") else None
    import time as _t

    def grow(_s):
        dat.write_bytes(dat.read_bytes() + b"\0" * (4 * 5 * 2))

    monkeypatch.setattr(_t, "sleep", grow)
    r = preprocess.raw_integrity(dat, "2_4_5", settle_s=0.01)
    assert not r["ok"] and "still uploading" in r["reason"]
    assert real_sleep is None or True


def test_camlog_disagreement_warns_but_does_not_block(tmp_path):
    """Held to a WARNING deliberately: the invariant holds on 4 of 4 sessions still on the share,
    and older raw is archived off, so there is not enough evidence to block a night's processing."""
    from wfield_local.preprocess import raw_integrity

    r = raw_integrity(_fake_raw(tmp_path, frames=100, camlog_rows=97), "2_4_5", settle_s=0)
    assert r["ok"], "a camlog mismatch must not make the file unusable"
    assert r["camlog_rows"] == 97 and r["frames"] == 100
