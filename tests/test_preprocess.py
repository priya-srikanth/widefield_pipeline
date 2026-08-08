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
    # exactly the 8-command chain, in order, with the right module names
    assert [c[0] for c in cmds] == [
        "wfield_local.framemap_event_maps",
        "wfield_local.plot_spout_trial_averages_shared_scale",
        "wfield_local.plot_spout_position_contrasts",
        "wfield_local.framemap_event_maps",
        "wfield_local.plot_lick_position_contrasts",
        "wfield_local.plot_lick_vs_cue_spout_maps",
        "wfield_local.quiet_periods",
        "wfield_local.framemap_event_maps",
    ]
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
    # labels + N: figure I/O
    for c in cmds:
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
