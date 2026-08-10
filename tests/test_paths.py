"""Tests for the PathResolver + machine-aware session resolution (reads real configs/)."""
import pytest

from wfield_local import config
from wfield_local.paths import PathResolver, is_absolute


def test_machine_routing_analysis_vs_imaging():
    a = PathResolver(machine="analysis")
    i = PathResolver(machine="imaging")
    assert a.root("labcams").startswith("M:/")
    assert i.root("labcams").startswith("N:/")
    # same relative tail resolves onto each machine's mount
    rel = "20260602/PS92_x/motion_corrected"
    assert a.resolve("labcams", rel) == "M:/MICROSCOPE/Priya/Widefield/labcams/" + rel
    assert i.resolve("labcams", rel) == "N:/MICROSCOPE/Priya/Widefield/labcams/" + rel


def test_raw_roots_imaging_only():
    i = PathResolver(machine="imaging")
    assert i.root("raw_labcams") == "E:/labcams_data"
    assert i.root("raw_daq") == "E:/DAQ_recorder_output"
    # unavailable on the analysis box -> raises rather than returning a wrong mount
    a = PathResolver(machine="analysis")
    with pytest.raises(RuntimeError):
        a.root("raw_labcams")


def test_unknown_root_and_bad_machine():
    a = PathResolver(machine="analysis")
    with pytest.raises(KeyError):
        a.root("does_not_exist")
    with pytest.raises(ValueError):
        PathResolver(machine="laptop")


def test_absolute_passthrough():
    a = PathResolver(machine="analysis")
    for p in ("M:/x/y.h5", r"\\host\share\z", "/mnt/data", "N:/already/there"):
        assert a.resolve("labcams", p) == p.replace("\\", "/")
    assert is_absolute("M:/x") and is_absolute("//host/s") and is_absolute("/a")
    assert not is_absolute("20260602/rel/path")


def test_sessions_resolve_absolute_on_both_machines():
    sa = {s["label"]: s for s in config.load_sessions(machine="analysis")}
    si = {s["label"]: s for s in config.load_sessions(machine="imaging")}
    assert len(sa) == len(si) and len(sa) >= 37   # both machines resolve the same set (grows as days register)
    a, i = sa["PS92_0602"], si["PS92_0602"]
    assert a["mc"].startswith("M:/") and i["mc"].startswith("N:/")
    # identical relative tail on both boxes
    tail = "20260602/PS92_20260602_151820/illuminated_rescue/motion_corrected"
    assert a["mc"].endswith(tail) and i["mc"].endswith(tail)
    assert a["h5"].endswith("20260602/PS92_20260602_152607.h5")


def test_sessions_yaml_is_root_relative_not_absolute():
    """The stored config must be relative (foreign-absolute behavior_trials excepted)."""
    raw = config._load("sessions.yaml")["sessions"]
    for animal in raw:
        for date, e in raw[animal].items():
            for field in ("mc", "h5", "fmdir"):
                v = e.get(field)
                if v is not None:
                    assert not is_absolute(v), (animal, date, field, v)


# --------------------------------------------------------------------------- mac/SMB client
def test_mac_machine_resolves_the_smb_mount():
    """A mac/Linux client with MICROSCOPE at /Volumes/Priya reads/writes the same shared tree."""
    m = PathResolver(machine="mac")
    assert m.root("microscope") == "/Volumes/Priya"
    assert m.root("behavior_logs") == "/Volumes/Priya/Behavior_logs/Widefield"
    assert m.root("daq_recorder_output") == "/Volumes/Priya/Widefield/DAQ_recorder_output"
    # same root-relative tail as the other machines -> sessions.yaml resolves identically
    tail = "20260602/PS92_20260602_151820/illuminated_rescue/motion_corrected"
    assert m.resolve("labcams", tail) == f"/Volumes/Priya/Widefield/labcams/{tail}"


def test_mac_has_no_local_raw_roots():
    m = PathResolver(machine="mac")
    for name in ("raw_labcams", "raw_daq", "figures_working"):
        with pytest.raises(RuntimeError):        # null mount -> unavailable on this machine
            m.root(name)


def test_every_logical_root_declares_every_machine():
    """A new logical root must not silently omit a machine (it would fail only at runtime)."""
    import yaml

    from wfield_local.paths import CONFIG_DIR, MACHINES
    cfg = yaml.safe_load((CONFIG_DIR / "paths.yaml").read_text(encoding="utf-8"))
    for name, spec in cfg["logical_roots"].items():
        assert set(spec["mounts"]) == set(MACHINES), name
