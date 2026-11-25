"""Efficiency audit 11-24 (Priya approved items 1, 2, 4): keys of the new run-independent caches. Temp dirs only."""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------------------------- item 1: FaceRhythm own_series
@pytest.fixture
def qfr(monkeypatch):
    for p in (ROOT / "scripts" / "cohort2_qc", ROOT / "scripts"):
        monkeypatch.syspath_prepend(str(p))
    import quiet_spikes_fr as q
    return q


class _FakeTiming:
    @staticmethod
    def path(bundle, key):
        return Path(bundle) / "timing" / f"{key}.npz"

    @staticmethod
    def load(bundle, key):
        return None

    @staticmethod
    def lossy(drop):
        return drop


def _fake_inputs(tmp_path):
    sd, od = tmp_path / "set", tmp_path / "bundle" / "out"
    for f in (sd / "own" / "k1.npz", od / "k1_session.npz", od / "k1_points.npz", od.parent / "timing" / "k1.npz"):
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"x")
    return sd, ["k1"], 1.5, np.arange(4.0), 0.95, 0.5, od, _FakeTiming


def test_own_series_store_is_run_independent(qfr, monkeypatch, tmp_path):
    monkeypatch.delenv(qfr.OWN_SERIES_ENV, raising=False)
    monkeypatch.setenv("WFIELD_INPUT_CACHE", str(tmp_path / "ic"))
    assert qfr.own_series_store() == tmp_path / "ic" / "fr_own_series"
    assert "fr_regressed" not in str(qfr.own_series_store())               # never the run-tag folder
    monkeypatch.setenv(qfr.OWN_SERIES_ENV, str(tmp_path / "explicit"))
    assert qfr.own_series_store() == tmp_path / "explicit"


def test_own_series_key_tracks_inputs_not_the_run(qfr, monkeypatch, tmp_path):
    import os
    inp = _fake_inputs(tmp_path)
    k0 = qfr.own_series_key("PS92_0806", inp)
    monkeypatch.setenv("WIDEFIELD_REBUILD_ROOT", str(tmp_path / "another_root"))   # run identity: not in the key
    monkeypatch.setenv("QS_V2_DIRNAME", "another_run")
    assert qfr.own_series_key("PS92_0806", inp) == k0
    tf = inp[6].parent / "timing" / "k1.npz"                                  # a timing-sidecar fix: new key
    tf.write_bytes(b"xy")
    os.utime(tf, (1, 1))
    k1 = qfr.own_series_key("PS92_0806", inp)
    assert k1 != k0
    inp2 = (*inp[:2], 1.6, *inp[3:])                                         # another master scale: new key
    assert qfr.own_series_key("PS92_0806", inp2) != k1


def test_own_series_reads_the_keyed_store(qfr, monkeypatch, tmp_path):
    inp = _fake_inputs(tmp_path)
    monkeypatch.setenv(qfr.OWN_SERIES_ENV, str(tmp_path / "store"))
    monkeypatch.setattr(qfr, "_own_inputs", lambda label: inp)
    calls = []
    out = dict(t5=np.arange(3.0), W5=np.ones((3, 2)), t25=np.arange(5.0), P25=np.zeros((5, 20), np.float32))
    monkeypatch.setattr(qfr, "own_series_build", lambda label, inputs=None: calls.append(1) or out)
    a = qfr.own_series("PS92_0806")
    b = qfr.own_series("PS92_0806")
    assert calls == [1]
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    files = list((tmp_path / "store").glob("PS92_0806_*.npz"))
    assert len(files) == 1 and qfr.own_series_key("PS92_0806", inp) in files[0].name


# ------------------------------------------------------------------------------------- item 2: csig / read-through
@pytest.fixture
def sc(monkeypatch, tmp_path):
    from wfield_local import run_mode, session_cache, trial_classes
    monkeypatch.setattr(session_cache, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.delenv(session_cache.READ_FROM_ENV, raising=False)
    monkeypatch.delenv("WIDEFIELD_NO_CACHE", raising=False)
    monkeypatch.setattr(trial_classes, "camera_gate_on", lambda: True)
    monkeypatch.setattr(trial_classes, "class_source_signature", lambda *a, **k: "COHORT")
    sigs = {"PS92_0806": "aaaa", "PS93_0806": "bbbb"}
    monkeypatch.setattr(trial_classes, "session_class_signature", lambda lab: sigs[lab])
    monkeypatch.setattr(run_mode, "session_kind_tag",
                        lambda exclude=(): "_".join(x for x in ("hv-x", "daqlk", "" if "csig" in exclude else
                                                                "csig-COHORT") if x))
    monkeypatch.setattr(session_cache, "session_signature", lambda s, p=None: "SIG")
    monkeypatch.setattr(session_cache, "_attempts_supplement_token", lambda lab: "")
    from wfield_local import gate_source, lp_licks
    monkeypatch.setattr(lp_licks, "is_lp", lambda: False)
    monkeypatch.setattr(gate_source, "is_lp", lambda: False)
    return session_cache, sigs


def test_tf_kinds_carry_no_class_signature(sc):
    session_cache, _ = sc
    assert session_cache.effective_kind("tf-cue-abc", "PS92_0806") == "tf-cue-abc-hv-x_daqlk"
    assert session_cache.effective_kind("tf-cue-abc") == "tf-cue-abc-hv-x_daqlk"


def test_other_kinds_carry_their_own_session_signature(sc):
    session_cache, sigs = sc
    a = session_cache.effective_kind("prm-cue", "PS92_0806")
    b = session_cache.effective_kind("prm-cue", "PS93_0806")
    assert a == "prm-cue-hv-x_daqlk_csigs-aaaa" and b == "prm-cue-hv-x_daqlk_csigs-bbbb"
    sigs["PS93_0806"] = "cccc"                                     # PS93 relabelled: PS92's key does not move
    assert session_cache.effective_kind("prm-cue", "PS92_0806") == a
    assert session_cache.effective_kind("prm-cue") == "prm-cue-hv-x_daqlk_csig-COHORT"   # no label: cohort csig


def test_unbuildable_classes_fall_back_to_the_cohort_csig(sc, monkeypatch):
    session_cache, _ = sc
    from wfield_local import trial_classes

    def boom(lab):
        raise FileNotFoundError("unseeded")
    monkeypatch.setattr(trial_classes, "session_class_signature", boom)
    assert session_cache.effective_kind("prm-cue", "PS92_0806") == "prm-cue-hv-x_daqlk_csig-COHORT"


def test_read_through_copies_an_identical_entry(sc, monkeypatch, tmp_path):
    session_cache, _ = sc
    prev = tmp_path / "prev_root_cache"
    prev.mkdir()
    kind = session_cache.effective_kind("prm-cue", "PS92_0806")
    (prev / f"PS92_0806__{kind}__SIG.pkl").write_bytes(pickle.dumps({"v": 1}))
    monkeypatch.setenv(session_cache.READ_FROM_ENV, str(prev))
    got = session_cache.cached({"label": "PS92_0806"}, "prm-cue", lambda: pytest.fail("must not compute"), verbose=False)
    assert got == {"v": 1}
    assert (tmp_path / "cache" / f"PS92_0806__{kind}__SIG.pkl").exists()        # now in this run's own store
    assert (prev / f"PS92_0806__{kind}__SIG.pkl").exists()                       # the earlier store untouched


def test_read_through_never_serves_another_key(sc, monkeypatch, tmp_path):
    session_cache, _ = sc
    prev = tmp_path / "prev"
    prev.mkdir()
    (prev / "PS92_0806__prm-cue-hv-x_daqlk_csigs-zzzz__SIG.pkl").write_bytes(pickle.dumps({"v": "stale"}))
    (prev / "PS92_0806__prm-cue-hv-x_daqlk_csigs-aaaa__OTHERSIG.pkl").write_bytes(pickle.dumps({"v": "stale"}))
    monkeypatch.setenv(session_cache.READ_FROM_ENV, str(prev))
    assert session_cache.cached({"label": "PS92_0806"}, "prm-cue", lambda: {"v": "fresh"}, verbose=False) == {"v": "fresh"}


def test_legacy_csig_tf_entry_is_adopted_but_not_for_other_kinds(sc, tmp_path):
    session_cache, _ = sc
    c = tmp_path / "cache"
    c.mkdir()
    (c / "PS92_0806__tf-cue-abc-hv-x_daqlk_csig-3d7bbe921e__SIG.pkl").write_bytes(pickle.dumps({"tf": 1}))
    (c / "PS92_0806__prm-cue-hv-x_daqlk_csig-3d7bbe921e__SIG.pkl").write_bytes(pickle.dumps({"prm": "old"}))
    assert session_cache.cached({"label": "PS92_0806"}, "tf-cue-abc", lambda: pytest.fail("x"), verbose=False) == {"tf": 1}
    assert (c / "PS92_0806__tf-cue-abc-hv-x_daqlk__SIG.pkl").exists()
    assert session_cache.cached({"label": "PS92_0806"}, "prm-cue", lambda: {"prm": "new"}, verbose=False) == {"prm": "new"}


def test_session_class_signature_hashes_only_that_session(monkeypatch):
    import pandas as pd

    from wfield_local import session_licks as SL
    from wfield_local import trial_classes as tc
    rows = {"PS92_0806": pd.DataFrame({"trial_id": [1, 2], "cls": ["contact", "nolick"]}),
            "PS93_0806": pd.DataFrame({"trial_id": [1, 2], "cls": ["contact", "contact"]})}
    monkeypatch.setattr(tc, "session_trials", lambda lab, window=None: rows[lab])
    monkeypatch.setattr(tc, "_class_code_digest", lambda: "code")
    monkeypatch.setattr(SL, "fused_on", lambda: False)
    monkeypatch.setattr(SL, "signature", lambda rv=None: "")
    monkeypatch.setattr(tc, "_SESSION_SIG", {})
    a = tc.session_class_signature("PS92_0806")
    rows["PS93_0806"] = pd.DataFrame({"trial_id": [1, 2], "cls": ["nolick", "nolick"]})
    monkeypatch.setattr(tc, "_SESSION_SIG", {})
    assert tc.session_class_signature("PS92_0806") == a
    rows["PS92_0806"] = pd.DataFrame({"trial_id": [1, 2], "cls": ["contact", "contact"]})
    monkeypatch.setattr(tc, "_SESSION_SIG", {})
    assert tc.session_class_signature("PS92_0806") != a


def test_no_sys_path_leak():
    assert all("cohort2_qc" not in p for p in sys.path[:1])


# ------------------------------------------------------------------------------------- item 4: design-raw sidecar
@pytest.fixture
def fe_raw(monkeypatch, tmp_path):
    from wfield_local import frozen_encoder as fe
    calls = []

    def build(B, ft, animal, date, rv):
        calls.append(1)
        return {"running": np.arange(6.0).reshape(3, 2), "lick_onset": np.ones((3, 1))}
    monkeypatch.setattr(fe, "_design_raw_build", build)
    monkeypatch.setattr(fe, "design_raw_key", lambda label, B, ft, a, d, rv: f"K{B}")
    return fe, calls


def test_design_raw_built_once_then_loaded(fe_raw, monkeypatch, tmp_path):
    fe, calls = fe_raw
    monkeypatch.setenv(fe.DESIGN_RAW_ENV, str(tmp_path / "dr"))
    a = fe.design_raw("PS92_0806", 1, np.arange(3.0), "PS92", "20260806", None)
    a["running"] = None                                            # callers mutate the dict: the store must not care
    b = fe.design_raw("PS92_0806", 1, np.arange(3.0), "PS92", "20260806", None)
    assert calls == [1] and np.array_equal(b["running"], np.arange(6.0).reshape(3, 2))
    fe.design_raw("PS92_0806", 2, np.arange(3.0), "PS92", "20260806", None)       # other behaviour content: new key
    assert calls == [1, 1] and len(list((tmp_path / "dr").glob("PS92_0806_*.pkl"))) == 2


def test_design_raw_is_opt_in(fe_raw, monkeypatch, tmp_path):
    fe, calls = fe_raw
    monkeypatch.delenv(fe.DESIGN_RAW_ENV, raising=False)
    monkeypatch.setenv(fe._ce().CACHE_ENV, str(tmp_path / "ic"))          # an input cache alone does not switch it on
    assert fe.design_raw_dir() is None
    fe.design_raw("PS92_0806", 1, np.arange(3.0), "PS92", "20260806", None)
    fe.design_raw("PS92_0806", 1, np.arange(3.0), "PS92", "20260806", None)
    assert calls == [1, 1] and not (tmp_path / "ic").exists()
    monkeypatch.setenv(fe.DESIGN_RAW_ENV, "0")
    assert fe.design_raw_dir() is None


def test_design_raw_key_ignores_scheduling_env_but_not_design_env(monkeypatch):
    from wfield_local import frozen_encoder as fe
    monkeypatch.setattr(fe, "FR_VARS", [])
    ft = np.arange(10.0)
    k0 = fe.design_raw_key("PS92_0806", {"b": 1}, ft, "PS92", "20260806", None)
    monkeypatch.setenv("WIDEFIELD_MAX_JOBS", "7")
    monkeypatch.setenv("FROZEN_FIG_OUT", "C:/somewhere")
    assert fe.design_raw_key("PS92_0806", {"b": 1}, ft, "PS92", "20260806", None) == k0
    monkeypatch.setenv("FROZEN_LICKS_ALL", "none")
    k1 = fe.design_raw_key("PS92_0806", {"b": 1}, ft, "PS92", "20260806", None)
    assert k1 != k0
    assert fe.design_raw_key("PS92_0806", {"b": 2}, ft, "PS92", "20260806", None) != k1
    assert fe.design_raw_key("PS92_0806", {"b": 1}, ft + 1e-9, "PS92", "20260806", None) != k1


def test_relabelled_designs_bypass_the_sidecar():
    src = (ROOT / "wfield_local" / "frozen_encoder.py").read_text(encoding="utf-8")
    i = src.index("    if relabel:\n        lab_r")
    block = src[i:src.index("    xc = {v:", i)]
    assert "raw_blocks(B, ft, pos_perm=" in block and "design_raw(label" in block.split("else:")[1]
