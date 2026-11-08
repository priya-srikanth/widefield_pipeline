"""bal_em: the PRODUCTION frozen behavioral-state decoder since 2026-11-03 (Priya: "ok bal_em it is").

Train on class-balanced pre-stroke sessions (center = equal-weight mean of the TRUE class means; training labels are
allowed); score a session by the label-free EM center (Saerens et al. 2002 priors -> soft class means of the raw
features -> equal-weight center, production SD, 3 passes). DECISIONS 2026-11-03.

Pins: (a) the spec id changes only for the new scaling; (b) no test-session label reaches a scored session's
center -- shuffling them leaves the scaled features (so the predictions) identical; (c) on a synthetic 3-class case
whose class mix shifts toward running, bal_em recovers the running recall plain session_z loses.
"""
import numpy as np
import pytest

from wfield_local import frozen_models as fm
from wfield_local import locomotor_decoder as ld
from wfield_local import locomotor_state as ls

CL = list(ls.THREE_WAY)
D = 6


def _sess(rng, mix, n=300, a=4.0):
    k = rng.choice(3, size=n, p=mix)
    mu = np.zeros((3, D))
    mu[CL.index("running"), 0] = a
    mu[CL.index("licking"), 1] = a
    X = rng.normal(size=(n, D)) + mu[k] + rng.normal(scale=3.0, size=D)     # per-session offset
    return X * rng.uniform(0.7, 1.4), np.array(CL)[k]                        # per-session gain


def _cohort(seed=1):
    """Five pre sessions at a quiet-heavy mix, three post sessions where running is 70 % (PS94's shape)."""
    rng = np.random.default_rng(seed)
    got = {f"PSX_08{i:02d}": (*_sess(rng, [0.6, 0.1, 0.3]), None, -10 + i) for i in range(1, 6)}
    for i in range(3):
        got[f"PSX_09{i:02d}"] = (*_sess(rng, [0.15, 0.7, 0.15]), None, 5 + i)
    return got


@pytest.fixture
def no_store(monkeypatch):
    """Route frozen_state_models through an in-memory fit (no store); record the specs it builds."""
    seen = []

    def fake_load_or_fit(spec, fit, *, meta=None, refreeze=None, log=print):
        seen.append(spec)
        return fit(), fm.STATUS_NEW
    monkeypatch.setattr(fm, "load_or_fit", fake_load_or_fit)
    monkeypatch.setattr(fm, "_session_sig", lambda lab: "sig")
    monkeypatch.setattr(fm.config, "animal_of", lambda lab: lab.split("_")[0])
    return seen


# ------------------------------------------------------------------ (a) spec identity
def test_spec_id_changes_only_for_the_new_scaling():
    """Three distinct ids, one per scaling; scaling absent = the same id as scaling=None. COHORT-1 REBUILD (11-03,
    FREEZE_VERSION 2): every id changed by design (training-data fingerprint, selection, hemo variant), so the pre-11-03
    pinned values (e14c78c2c8e8 / 85aa81b36cb3) no longer apply on this branch; the rebuild fits a fresh store."""
    kw = dict(align="segments", source="joint_x", train_labels=[], basis_id="x", n_features=3, post_s=1.0,
              train_data=None)
    ids = {sc: fm.spec_id(fm.make_spec("PS92", "state", **kw, scaling=sc))
           for sc in (None, "session_z", "session_bal_em")}
    assert ids[None] == fm.spec_id(fm.make_spec("PS92", "state", **kw))
    assert len(set(ids.values())) == 3


def test_default_scaling_is_bal_em_and_session_z_stays_selectable(monkeypatch):
    monkeypatch.delenv(ld.STATE_SCALING_ENV, raising=False)
    monkeypatch.delenv(ld.STATE_ZSCORE_ENV, raising=False)
    assert ld.state_scaling() == "session_bal_em"
    monkeypatch.setenv(ld.STATE_SCALING_ENV, "session_z")
    assert ld.state_scaling() == "session_z"
    monkeypatch.setenv(ld.STATE_ZSCORE_ENV, "0")                 # the pre-11-02 unscaled run, as before
    assert ld.state_scaling() is None
    monkeypatch.delenv(ld.STATE_ZSCORE_ENV)
    monkeypatch.setenv(ld.STATE_SCALING_ENV, "bogus")
    with pytest.raises(ValueError):
        ld.state_scaling()


def test_frozen_state_models_spec_carries_the_scaling(no_store):
    got = _cohort()
    ld.frozen_state_models("PSX", "abc", got, scaling="session_z")
    ld.frozen_state_models("PSX", "abc", got, scaling="session_bal_em")
    ld.frozen_state_models("PSX", "abc", got, scaling="session_bal_em", min_mass=0.05)
    ld.frozen_state_models("PSX", "abc", got, scaling=None)
    sc = [(s.get("scaling"), s.get("scaling_params")) for s in no_store]
    assert sc == [("session_z", None),
                  ("session_bal_em", {"min_mass": None, "n_outer": 3}),        # min_mass recorded even when OFF
                  ("session_bal_em", {"min_mass": 0.05, "n_outer": 3}),
                  (None, None)]
    ids = [fm.spec_id(s) for s in no_store]
    assert len(set(ids)) == 4
    # everything but the scaling fields is the same identity
    strip = [{k: v for k, v in s.items() if k not in ("scaling", "scaling_params")} for s in no_store]
    assert all(x == strip[0] for x in strip)
    # session_z / unscaled specs carry no scaling_params key at all (their stored ids are unchanged)
    assert "scaling_params" not in no_store[0] and "scaling_params" not in no_store[3]


def test_bal_em_payload_carries_training_priors(no_store):
    got = _cohort()
    models, _st = ld.frozen_state_models("PSX", "abc", got, scaling="session_bal_em")
    pre = sorted(lab for lab, v in got.items() if v[3] <= 0)
    assert np.allclose(models["pi_train"]["full"], ld.class_prior([got[lab][1] for lab in pre]))
    held = pre[0]
    assert np.allclose(models["pi_train"]["loso"][held],
                       ld.class_prior([got[lab][1] for lab in pre if lab != held]))
    zmodels, _st = ld.frozen_state_models("PSX", "abc", got, scaling="session_z")
    assert "pi_train" not in zmodels                              # session_z payload unchanged in shape


# ------------------------------------------------------------------ (b) no test labels
def test_bal_em_scoring_uses_no_test_labels(no_store, monkeypatch):
    """Shuffle every scored session's labels at SCORE time (post sessions, and the pre sessions scored by their LOSO
    models; the frozen models -- trained on pre labels, which is allowed -- are fixed): the scaled features handed to
    the model, hence every prediction, are bit-identical."""
    got = _cohort()
    rng = np.random.default_rng(7)
    shuf = {lab: (X, rng.permutation(y), g, d) for lab, (X, y, g, d) in got.items()}
    captured = {}

    def capture(tag):
        def _score(model, Z, y, classes=ls.THREE_WAY):
            captured.setdefault(tag, []).append(model.predict(np.asarray(Z)))
            return None
        return _score
    models, _s = ld.frozen_state_models("PSX", "abc", got, scaling="session_bal_em")
    monkeypatch.setattr(ld, "score", capture("true"))
    ld.score_animal("PSX", "abc", got, scaling="session_bal_em", models=models)
    monkeypatch.setattr(ld, "score", capture("shuffled"))
    ld.score_animal("PSX", "abc", shuf, scaling="session_bal_em", models=models)
    assert len(captured["true"]) == len(got)
    for a, b in zip(captured["true"], captured["shuffled"]):
        assert np.array_equal(a, b)


def test_bal_em_centre_takes_no_labels_by_signature():
    import inspect
    for fn in (ld.bal_em_centre, ld.score_transform):
        params = set(inspect.signature(fn).parameters)
        assert not params & {"y", "labels", "y_true"}, fn.__name__


# ------------------------------------------------------------------ (c) the point of it
def test_bal_em_recovers_minority_recall_session_z_loses(no_store):
    """Post sessions run 70 % of the time against 10 % pre. Plain session_z centers on the session mean, which moves
    toward running, and reads running as quiet; bal_em's label-free equal-weight center recovers it."""
    got = _cohort()
    run = {}
    for sc in ("session_z", "session_bal_em"):
        rec, _st, _m = ld.score_animal("PSX", "abc", got, scaling=sc)
        run[sc] = np.mean([r["per_class"]["running"] for d, r in rec.items() if d != "PRE"])
        bal = np.mean([r["balacc"] for d, r in rec.items() if d != "PRE"])
        run[sc + "_bal"] = bal
    assert run["session_z"] < 0.2
    assert run["session_bal_em"] > 0.7
    assert run["session_bal_em_bal"] > run["session_z_bal"] + 0.15


def test_min_mass_guard_is_off_by_default_and_changes_only_when_set():
    assert ld.BAL_EM_MIN_MASS is None
    assert ld.bal_em_params() == {"min_mass": None, "n_outer": 3}
    assert ld.bal_em_params(0.05) == {"min_mass": 0.05, "n_outer": 3}
    rng = np.random.default_rng(3)
    X, y = _sess(rng, [0.6, 0.1, 0.3])
    m = ld.fit_frozen(ld.balanced_zscore(X, y), y)
    pi = ld.class_prior([y])
    Z0, i0 = ld.bal_em_centre(m, X, pi, return_info=True)
    Z1, i1 = ld.bal_em_centre(m, X, pi, min_mass=None, return_info=True)
    assert np.array_equal(Z0, Z1) and not i0["guard"]
    _Z2, i2 = ld.bal_em_centre(m, X, pi, min_mass=0.99, return_info=True)   # forces the guard
    assert i2["guard"]
