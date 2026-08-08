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
