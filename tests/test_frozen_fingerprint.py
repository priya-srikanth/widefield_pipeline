"""FREEZE_VERSION 2 (cohort-1 rebuild 2026-11-03, Priya: yes): every frozen spec carries a fingerprint of the actual
training data, the run's trial-selection tag and the hemo variant; fingerprinted models never go into a store holding
older models; the LP arm's DAQ-trained cross-check still finds its DAQ model. All stores here are temp dirs."""
from __future__ import annotations

import hashlib
import inspect
import json

import numpy as np
import pytest

from wfield_local import frozen_models as fm

TR = ["PS92_0810", "PS92_0811"]


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    for k in ("WIDEFIELD_HEMO_VARIANT", "WIDEFIELD_TRIAL_GATE", "WIDEFIELD_LICK_SOURCE", "WIDEFIELD_LP_FROZEN",
              "WIDEFIELD_ENL_WINDOW_S", "WIDEFIELD_GATE_SOURCE"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("WIDEFIELD_FROZEN_MODEL_DIR", str(tmp_path / "store"))
    monkeypatch.setattr(fm, "server_dir", lambda: None)


def _spec(X, **kw):
    return fm.make_spec("PS92", "decoder", align="cue", source="locanmf", train_labels=TR, train_data=X, **kw)


def test_train_data_is_required():
    with pytest.raises(TypeError, match="train_data"):
        fm.make_spec("PS92", "decoder", align="cue", source="locanmf", train_labels=TR)


def test_spec_carries_fingerprint_selection_and_variant():
    X = (np.arange(12.).reshape(4, 3), np.array([0, 1, 0, 1]))
    s = _spec(X)
    assert s["freeze_version"] == 2
    assert s["hemo_variant"] == "meegkit_hpfit"
    assert s["selection"] == "sel-daq-quittail"
    assert s["train_fp"] == fm.data_fingerprint(X)
    assert fm.spec_id(_spec(X)) == fm.spec_id(s)                           # deterministic


def test_changed_trial_set_under_same_labels_mints_a_new_id():
    """The PS92 section G case: same labels / SVTcorr, one training row different -> different id."""
    X = np.arange(12.).reshape(4, 3)
    y = np.array([0, 1, 0, 1])
    a = _spec((X, y))
    X2 = X.copy()
    X2[2, 1] += 1e-9
    assert fm.spec_id(_spec((X2, y))) != fm.spec_id(a)
    assert fm.spec_id(_spec((X[:3], y[:3]))) != fm.spec_id(a)
    assert fm.spec_id(_spec((X.astype(np.float32), y))) != fm.spec_id(a)    # dtype is identity too


def test_selection_and_hemo_variant_change_the_id(monkeypatch):
    X = np.ones((2, 2))
    a = fm.spec_id(_spec(X))
    monkeypatch.setenv("WIDEFIELD_TRIAL_GATE", "camera")
    b = fm.spec_id(_spec(X))
    monkeypatch.delenv("WIDEFIELD_TRIAL_GATE")
    monkeypatch.setenv("WIDEFIELD_HEMO_VARIANT", "meegkit_hpfit_lag_refitT_swapmask")
    c = fm.spec_id(_spec(X))
    assert len({a, b, c}) == 3
    assert fm.spec_id(_spec(X, selection="sel-x")) != c


def test_fingerprint_handles_nested_containers_deterministically():
    obj = {"b": [np.arange(3), None, 1.5], "a": (np.zeros((2, 2), np.float32), "x")}
    assert fm.data_fingerprint(obj) == fm.data_fingerprint(dict(reversed(list(obj.items()))))
    assert fm.data_fingerprint(obj) != fm.data_fingerprint({**obj, "a": (np.ones((2, 2), np.float32), "x")})


def test_refuses_to_write_into_a_store_with_older_models(tmp_path):
    d = fm.local_dir() / "PS92" / "decoder_cue_locanmf_old000000000"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({"spec": {"freeze_version": 1, "animal": "PS92"}}))
    with pytest.raises(fm.LegacyStoreError, match="fresh store"):
        fm.load_or_fit(_spec(np.ones((2, 2))), lambda: {"m": 1}, log=lambda *_: None)
    assert [p.name for p in (fm.local_dir() / "PS92").iterdir()] == ["decoder_cue_locanmf_old000000000"]


def test_fresh_store_freezes_then_hits():
    s = _spec(np.ones((2, 2)))
    p1, st1 = fm.load_or_fit(s, lambda: {"m": 1}, log=lambda *_: None)
    p2, st2 = fm.load_or_fit(_spec(np.ones((2, 2))), lambda: {"m": 2}, log=lambda *_: None)
    assert (st1, st2) == (fm.STATUS_NEW, fm.STATUS_HIT) and p2 == {"m": 1}


def test_lp_daq_cross_check_finds_the_daq_trained_model(monkeypatch):
    daq = _spec((np.ones((3, 2)), np.array([0, 1, 1])))
    fm.load_or_fit(daq, lambda: {"daq": True}, log=lambda *_: None)
    monkeypatch.setenv("WIDEFIELD_LICK_SOURCE", "lp")
    monkeypatch.setenv("WIDEFIELD_LP_FROZEN", "daq")
    lp_side = _spec((np.zeros((5, 2)), np.array([0, 1, 1, 0, 1])))       # the LP run's own (different) pool
    assert lp_side["train_fp"] == fm.DAQ_RUN_FP and "lick_source" not in lp_side
    payload, status = fm.load_or_fit(lp_side, lambda: pytest.fail("must not fit"), log=lambda *_: None)
    assert payload == {"daq": True} and status == fm.STATUS_HIT


def test_lp_daq_cross_check_is_ambiguous_with_two_daq_models(monkeypatch):
    fm.load_or_fit(_spec(np.ones((2, 2))), lambda: {"a": 1}, log=lambda *_: None)
    fm.load_or_fit(_spec(np.zeros((2, 2))), lambda: {"b": 1}, log=lambda *_: None)
    monkeypatch.setenv("WIDEFIELD_LICK_SOURCE", "lp")
    monkeypatch.setenv("WIDEFIELD_LP_FROZEN", "daq")
    with pytest.raises(RuntimeError, match="DAQ-trained models"):
        fm.load_or_fit(_spec(np.ones((2, 2))), lambda: None, log=lambda *_: None)


# --------------------------------------------------------------------- lick-landing read-outs: scaling bit-identical
#: Coordinator 11-03 (Priya-approved check, DECISIONS fa70b651): the lick-landing frozen decoders use per-session
#: FRAME-level SD scaling over ALL frames (in the read-out scripts' feature building), no centring over lick samples,
#: then this helper's StandardScaler + RidgeCV. The fingerprint change touched only the SPEC of `pre_loso_ridge`; the
#: fitting functions must stay byte-identical. Any edit to them fails here -- change the pin only with Priya's sign-off.
_PINNED = {"ridge_fit": None, "fit_all": None, "data_sig": None}


def _src_sha(fn) -> str:
    return hashlib.sha1(inspect.getsource(fn).encode()).hexdigest()


def test_frozen_readouts_fit_functions_are_unchanged():
    from wfield_local import frozen_readouts as fr
    got = {k: _src_sha(getattr(fr, k)) for k in _PINNED}
    assert got == PINNED_SHA, got


def test_stored_readout_payload_equals_the_in_memory_fit(monkeypatch):
    from wfield_local import frozen_readouts as fr
    rng = np.random.default_rng(3)
    pre = TR + ["PS92_0812"]
    X = {q: rng.standard_normal((20, 4)) for q in pre}
    Y = {q: rng.standard_normal(20) for q in pre}
    al = np.logspace(-2, 2, 5)
    mem = fr.fit_all(pre, X, Y, al, None, True)
    sto = fr.pre_loso_ridge("PS92", "test-tag", pre, X, Y, al, log=lambda *_: None)
    for k in mem:
        np.testing.assert_array_equal(mem[k][0].scale_, sto[k][0].scale_)
        np.testing.assert_array_equal(mem[k][0].mean_, sto[k][0].mean_)
        np.testing.assert_array_equal(mem[k][1].coef_, sto[k][1].coef_)


PINNED_SHA = {
    "ridge_fit": "c4a0f5eb1ab50ea460c9eddf10c7ff5a055dbdce",       # = the pre-rebuild source (verified 11-03)
    "fit_all": "7a8117fea5913bc3ea91bc405b9a585a81be83d6",
    "data_sig": "3b2a840eff6dc69521f2fa33c3264dc624432bbe",
}
