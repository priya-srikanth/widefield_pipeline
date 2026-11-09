"""Per-process loader memos from the 11-09 grant-render profile: hit counting, independent copies, key changes reload,
memo thresholds."""
from __future__ import annotations

import os

import numpy as np

from wfield_local import config, gate_source, joint_locanmf, paths, session_memo


# ---------------------------------------------------------------- session_memo.per_path
def _counting_loader():
    calls = []

    @session_memo.per_path
    def load(path, scale=1, *, extra=0):
        calls.append((str(path), scale, extra))
        return {"a": np.arange(3) * scale + extra, "nested": {"b": [1, 2]}, "sr": 5000.0}
    return load, calls


def test_per_path_hits_and_copies(tmp_path):
    f = tmp_path / "x.h5"
    f.write_bytes(b"1")
    load, calls = _counting_loader()
    a = load(f)
    a["a"][0] = 99
    a["nested"]["b"].append(3)
    b = load(str(f))                       # same file given as str: a hit
    assert len(calls) == 1
    assert b["a"][0] == 0 and b["nested"]["b"] == [1, 2]
    assert b["a"] is not load(f)["a"]


def test_per_path_args_and_file_change_reload(tmp_path):
    f = tmp_path / "x.h5"
    f.write_bytes(b"1")
    load, calls = _counting_loader()
    load(f)
    load(f, 2)
    load(f, extra=1)
    assert len(calls) == 3
    load(f, 2)
    assert len(calls) == 3
    st = f.stat()
    f.write_bytes(b"22")                   # rewritten file: size + mtime change
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    load(f)
    assert len(calls) == 4
    load.cache_clear()
    load(f)
    assert len(calls) == 5


def test_per_path_missing_file_passes_through(tmp_path):
    load, calls = _counting_loader()
    load(tmp_path / "none.h5")
    load(tmp_path / "none.h5")
    assert len(calls) == 2


def test_per_path_store_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(session_memo, "PER_PATH_MAX", 3)
    load, calls = _counting_loader()
    fs = []
    for i in range(4):
        f = tmp_path / f"{i}.h5"
        f.write_bytes(b"1")
        fs.append(f)
        load(f)
    assert load.cache_info() <= 3


def test_spout_loader_is_memoized(monkeypatch, tmp_path):
    from wfield_local import plot_spout_trial_averages as psta
    assert hasattr(psta._load_daq_events, "cache_clear")
    f = tmp_path / "s.h5"
    f.write_bytes(b"0")
    n = []

    class _F:
        def __enter__(self):
            n.append(1)
            return self

        def __exit__(self, *a):
            return False
    bits = np.zeros((10, 4), np.uint8)
    bits[3:5, 0] = 1
    monkeypatch.setattr(psta.daq_io, "open_daq", lambda p: _F())
    monkeypatch.setattr(psta.daq_io, "session_attrs", lambda fh: (5000.0, "t"))
    monkeypatch.setattr(psta.daq_io, "digital_bits",
                        lambda fh: (["cue", "spout_strobe", "pco_exposure", "x"], bits))
    monkeypatch.setattr(psta.daq_io, "strobe_codes", lambda b, nm, s: np.zeros(len(s), int))
    a = psta._load_daq_events(f)
    a["cue_samples"][:] = -1
    b = psta._load_daq_events(f)
    assert len(n) == 1 and (b["cue_samples"] >= 0).all()


# ---------------------------------------------------------------- config.load_sessions
def test_load_sessions_memo_copies_env_and_reread(monkeypatch):
    calls = []
    real = config._load_sessions_uncached

    def spy(*a, **k):
        calls.append(a)
        return real(*a, **k)
    monkeypatch.setattr(config, "_load_sessions_uncached", spy)
    monkeypatch.delenv("WIDEFIELD_ONLY_ANIMALS", raising=False)
    a = config.load_sessions()
    a[0]["label"] = "CHANGED"
    a.append({"label": "extra"})
    b = config.load_sessions()
    assert len(calls) == 1 and b[0]["label"] != "CHANGED" and len(b) == len(a) - 1
    first = b[0]["label"].split("_")[0]
    monkeypatch.setenv("WIDEFIELD_ONLY_ANIMALS", first)
    c = config.load_sessions()
    assert len(calls) == 2 and {s["label"].split("_")[0] for s in c} == {first}
    config.load_sessions(animals=[first])
    config.load_sessions(animals=(first,))
    assert len(calls) == 3
    config._load.cache_clear()             # sessions.yaml re-read (await_locanmf) -> fresh
    config.load_sessions(animals=[first])
    assert len(calls) == 4
    config.resolver.cache_clear()          # new resolver -> fresh
    config.load_sessions(animals=[first])
    assert len(calls) == 5


def test_load_sessions_memo_matches_uncached():
    assert config.load_sessions() == config._load_sessions_uncached()
    assert config.load_sessions() == config._load_sessions_uncached()


# ---------------------------------------------------------------- gate_source
def test_gate_memo_threshold_covers_a_cohort():
    assert gate_source._MEMO_MAX >= 256


def test_gate_extra_checks_memo_before_cue_load(monkeypatch):
    from wfield_local import lp_licks
    from wfield_local import plot_spout_trial_averages as psta
    lab = config.load_sessions()[0]["label"]
    cue_loads = []

    def cue(h5):
        cue_loads.append(h5)
        return {"cue_samples": np.array([100, 5000, 9000]), "sample_rate_hz": 1000.0}
    monkeypatch.setattr(psta, "_load_daq_events", cue)
    monkeypatch.setattr(gate_source, "_daq_contact_samples", lambda h5: (np.array([150.0, 9900.0]), 1000.0))
    monkeypatch.setattr(lp_licks, "spec_token", lambda rv=None: "tok")

    def no_lp(*a, **k):
        raise lp_licks.LPUnavailable("none")
    monkeypatch.setattr(lp_licks, "session_lp", no_lp)
    monkeypatch.setattr(gate_source, "_EXTRA", {})
    r1 = gate_source.extra(lab, 1.0)
    r2 = gate_source.extra(lab, 1.0)
    assert len(cue_loads) == 1 and r2 is r1
    assert r1["daq"].tolist() == [True, False, True]
    # an explicit train equal to the DAQ one hits the hashed entry without a load
    r3 = gate_source.extra(lab, 1.0, cue_samples=np.array([100, 5000, 9000]), sr=1000.0)
    assert r3 is r1 and len(cue_loads) == 1
    gate_source.extra(lab, 2.0)            # new window -> new entry
    assert len(cue_loads) == 2
    monkeypatch.setattr(lp_licks, "spec_token", lambda rv=None: "tok2")   # LP inputs rewritten -> new entry
    gate_source.extra(lab, 1.0)
    assert len(cue_loads) == 3


def test_daq_train_memo_keeps_a_cohort(monkeypatch):
    from wfield_local import plot_lick_aligned_averages as pla
    n = []
    monkeypatch.setattr(pla, "_load_daq_events",
                        lambda h5, *a, **k: n.append(h5) or {"lick_samples": [3, 1], "sample_rate_hz": 1000.0})
    monkeypatch.setattr(gate_source, "_DAQ_TRAIN", {})
    for _ in range(2):
        for i in range(120):
            lk, sr = gate_source._daq_contact_samples(f"s{i}.h5")
    assert len(n) == 120 and lk.tolist() == [1.0, 3.0] and sr == 1000.0


# ---------------------------------------------------------------- joint_locanmf._cache_glob
def test_cache_glob_lists_once_and_sees_new_files(monkeypatch, tmp_path):
    for nm in ("PS1_0801__vc__a.pkl", "PS1_0801__vc__b.txt", "PS2_0801__vc__c.pkl"):
        (tmp_path / nm).write_bytes(b"0")
    scans = []
    real = os.scandir
    monkeypatch.setattr(joint_locanmf.os, "scandir", lambda p: scans.append(p) or real(p))
    got = joint_locanmf._cache_glob(tmp_path, "PS1_0801__vc__*.pkl")
    assert sorted(got) == sorted(tmp_path.glob("PS1_0801__vc__*.pkl"))
    joint_locanmf._cache_glob(tmp_path, "PS2_0801__vc__*.pkl")
    assert len(scans) == 1
    mt = tmp_path.stat().st_mtime_ns
    (tmp_path / "PS1_0801__vc__d.pkl").write_bytes(b"0")
    os.utime(tmp_path, ns=(mt + 10**9, mt + 10**9))
    got = joint_locanmf._cache_glob(tmp_path, "PS1_0801__vc__*.pkl")
    assert len(scans) == 2 and {p.name for p in got} == {"PS1_0801__vc__a.pkl", "PS1_0801__vc__d.pkl"}
    assert joint_locanmf._cache_glob(tmp_path / "missing", "*.pkl") == []


# ---------------------------------------------------------------- paths._parsed_yaml
def test_path_resolver_yaml_parsed_once_and_copied(monkeypatch, tmp_path):
    n = []
    real = paths.yaml.safe_load
    monkeypatch.setattr(paths.yaml, "safe_load", lambda fh: n.append(1) or real(fh))
    a, b = paths.PathResolver(), paths.PathResolver()
    assert len(n) == 1 and a._cfg == b._cfg and a._cfg is not b._cfg
    f = tmp_path / "p.yaml"
    f.write_text("logical_roots: {}\n", encoding="utf-8")
    paths.PathResolver(config_path=f)
    f.write_text("logical_roots: {x: {mounts: {}}}\n", encoding="utf-8")
    st = f.stat()
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    r = paths.PathResolver(config_path=f)
    assert len(n) == 3 and "x" in r._cfg["logical_roots"]
