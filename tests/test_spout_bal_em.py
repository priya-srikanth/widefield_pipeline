"""bal_em: the PRODUCTION frozen SPOUT-POSITION decoder scaling since 2026-11-03 (Priya: "I would adopt B everywhere").

Train on pre-stroke sessions each centred on the equal-weight mean of its TRUE class means; score every session with
the label-free EM centre of its ENGAGED trials (Saerens priors -> soft class means -> 1/6 each, production SD, 3
passes, no guard), applied to its no-lick trials too. DECISIONS 2026-11-03.

Pins: (a) spec ids: session_z (= the pre-11-03 production) is byte-identical to the old spec, bal_em is a new id that
differs only in ``scaling`` / ``scaling_params``; (b) no scored-session label reaches its centre -- shuffling them leaves
the inputs to the model bit-identical, and the scoring functions take no labels by signature; (c) on a synthetic
6-spout case whose engaged set loses one spout, the old engaged-trial z-score biases the per-spout accuracies and
bal_em removes most of that bias; (d) the state decoder still reads the shared primitives (moved, not copied).
"""
import inspect

import numpy as np
import pytest

from wfield_local import frozen_models as fm
from wfield_local import grant_kit
from wfield_local import label_shift
from wfield_local import locanmf_frozen_decoder as fd
from wfield_local import locomotor_decoder as ld
from wfield_local import spout_scaling as ss

K, D = 6, 12


@pytest.fixture
def fake_store(monkeypatch):
    seen = []

    def fake_load_or_fit(spec, fit, *, meta=None, refreeze=None, log=print):
        seen.append(spec)
        return fit(), fm.STATUS_NEW
    monkeypatch.setattr(fm, "load_or_fit", fake_load_or_fit)
    monkeypatch.setattr(fm, "_session_sig", lambda lab: "sig")
    monkeypatch.setattr(fm.config, "animal_of", lambda lab: lab.split("_")[0])
    return seen


def _session(rng, mix, n=360, a=2.5, n_nolick=0):
    """Six spout patterns + a per-session offset and gain (what the per-session normalisation is for). ``n_nolick``
    extra trials of spout 5 share the session's offset and gain: the no-lick arm of the acute far-contra cell."""
    k = rng.choice(K, size=n, p=mix)
    k = np.concatenate([k, np.full(n_nolick, K - 1)])
    P = np.zeros((K, D))
    for c in range(K):
        P[c, c] = a
        P[c, (c + 1) % K + K // 2] = -0.5 * a
    X = rng.normal(size=(len(k), D)) + P[k]
    X = (X + rng.normal(scale=2.0, size=D)) * rng.uniform(0.7, 1.4)
    return X[:n], k[:n], X[n:], k[n:]


def _pool(seed=0, post_mix=None, n_pre=5, n_post=3):
    """Production-frame pool: engaged rows z-scored per session (as `pool_sessions`), plus no-lick rows of the missing
    spout normalised with the ENGAGED statistics (the shape of the acute far-contra cell)."""
    rng = np.random.default_rng(seed)
    post_mix = post_mix or [0.24, 0.24, 0.24, 0.14, 0.12, 0.02]
    XE, YE, GE, XU, YU, GU, kept = [], [], [], [], [], [], []
    for i in range(n_pre + n_post):
        is_post = i >= n_pre
        X, y, Xn, yn = _session(rng, post_mix if is_post else [1 / K] * K, n_nolick=60 if is_post else 0)
        mu, sd = X.mean(0), X.std(0)
        XE.append((X - mu) / sd)
        YE.append(y)
        GE.append(np.full(len(y), i))
        XU.append((Xn - mu) / sd)
        YU.append(yn)
        GU.append(np.full(len(yn), i))
        kept.append(f"PSX_09{i:02d}" if is_post else f"PSX_08{i:02d}")
    return (np.vstack(XE), np.concatenate(YE), np.concatenate(GE), np.vstack(XU), np.concatenate(YU),
            np.concatenate(GU), kept, list(range(n_pre)))


# ------------------------------------------------------------------ (a) spec identity
def test_session_z_spec_is_the_pre_1103_spec(fake_store, monkeypatch):
    """Under WIDEFIELD_SPOUT_SCALING=session_z the decoder spec has NO scaling keys and the same id as a spec built the
    way `decoder_spec` built it before 11-03; bal_em is a new id differing only in the scaling fields."""
    XE, YE, GE, *_r, kept, pre_i = _pool()
    kw = dict(align="cue", source="roi", post_s=2.0, YE=YE, GE=GE)
    # COHORT-1 REBUILD (merge 11-03): FREEZE_VERSION 2 specs carry the training-data fingerprint, so the pinned HEAD
    # literal (d19c950c2139) no longer applies on this branch; the contract -- session_z adds no key, bal_em differs
    # only in the scaling fields -- is what is checked
    m = np.isin(GE, sorted(pre_i))
    old = fm.make_spec("PSX", "decoder", align="cue", source="roi", train_labels=[kept[i] for i in pre_i],
                       post_s=2.0, zscore=True, basis_id=None, n_features=int(XE.shape[1]),
                       train_data=(XE[m], YE[m], GE[m]))
    monkeypatch.setenv(ss.SPOUT_SCALING_ENV, "session_z")
    z = fd.decoder_spec(XE, kept, pre_i, **kw)
    assert z == old and "scaling" not in z and "scaling_params" not in z
    assert fm.spec_id(z) == fm.spec_id(old)
    monkeypatch.delenv(ss.SPOUT_SCALING_ENV)
    b = fd.decoder_spec(XE, kept, pre_i, **kw)                           # production default
    assert b["scaling"] == "session_bal_em"
    assert b["scaling_params"] == {"centre_set": "engaged", "guard": None, "min_mass": None, "n_outer": 3,
                                   "train_centre": "true_class_means_equal_weight"}
    assert fm.spec_id(b) != fm.spec_id(old)
    assert {k: v for k, v in b.items() if k not in ("scaling", "scaling_params")} == old
    # the models' own spec follows the same switch
    fd.frozen_decoder_models(XE, YE, GE, kept, pre_i, align="cue", source="roi", log=lambda *a: None)
    assert fake_store[-1] == b
    fd.frozen_decoder_models(XE, YE, GE, kept, pre_i, align="cue", source="roi", log=lambda *a: None,
                             scaling=None)
    assert fake_store[-1] == old


def test_other_spec_ids_are_untouched():
    """Every non-decoder spec is built without the spout scaling: the state ids pinned in test_state_bal_em, and an
    encoder spec, are unchanged by this module existing."""
    # COHORT-1 REBUILD (merge 11-03): FREEZE_VERSION 2 ids differ from main's pinned literals (e14c78c2c8e8 /
    # 85aa81b36cb3, see test_state_bal_em); the spout scaling still adds nothing to a non-decoder spec
    kw = dict(align="segments", source="joint_x", train_labels=[], basis_id="x", n_features=3, post_s=1.0,
              train_data=None)
    a = fm.make_spec("PS92", "state", **kw)
    assert "scaling" not in a and fm.spec_id(fm.make_spec("PS92", "state", **kw, scaling="session_z")) != fm.spec_id(a)
    assert ss.spec_kw(None) == {}


def test_default_is_bal_em_and_session_z_stays_selectable(monkeypatch):
    monkeypatch.delenv(ss.SPOUT_SCALING_ENV, raising=False)
    assert ss.spout_scaling() == "session_bal_em"
    monkeypatch.setenv(ss.SPOUT_SCALING_ENV, "session_z")
    assert ss.spout_scaling() is None
    monkeypatch.setenv(ss.SPOUT_SCALING_ENV, "bogus")
    with pytest.raises(ValueError):
        ss.spout_scaling()


def test_bal_em_payload_carries_training_priors(fake_store, monkeypatch):
    monkeypatch.delenv(ss.SPOUT_SCALING_ENV, raising=False)
    XE, YE, GE, *_r, kept, pre_i = _pool()
    m, _st = fd.frozen_decoder_models(XE, YE, GE, kept, pre_i, align="cue", source="roi", log=lambda *a: None)
    assert m["scaling"] == "session_bal_em"
    assert np.allclose(m["pi_train"]["full"], ss.class_prior(YE[np.isin(GE, pre_i)]))
    assert np.allclose(m["pi_train"]["loso"][kept[0]], ss.class_prior(YE[np.isin(GE, pre_i[1:])]))
    mz, _st = fd.frozen_decoder_models(XE, YE, GE, kept, pre_i, align="cue", source="roi", log=lambda *a: None,
                                       scaling=None)
    assert "pi_train" not in mz and set(mz) == {"full", "loso"}           # session_z payload unchanged in shape


def test_session_z_scored_frame_is_the_production_frame(fake_store):
    XE, YE, GE, XU, YU, GU, kept, pre_i = _pool()
    mz, _st = fd.frozen_decoder_models(XE, YE, GE, kept, pre_i, align="cue", source="roi", log=lambda *a: None,
                                       scaling=None)
    A, B = fd.frozen_scored_frame(mz, XE, GE, XU, GU, kept, pre_i)
    assert A is XE and B is XU


# ------------------------------------------------------------------ (b) no scored-session labels
def test_scoring_takes_no_labels_by_signature():
    for fn in (ss.score_centre, ss.score_session, ss.scored_frame, fd.frozen_scored_frame,
               label_shift.bal_em_centre, grant_kit._frozen_score_rows):
        assert not set(inspect.signature(fn).parameters) & {"y", "YE", "YU", "labels", "y_true"}, fn.__name__


def test_shuffling_scored_labels_leaves_inputs_bit_identical(fake_store, monkeypatch):
    """Post sessions' labels shuffled (engaged AND no-lick), and one PRE session's labels shuffled: the rows each
    session is scored on are bit-identical (the pre session is scored by a LOSO model that never saw it)."""
    monkeypatch.delenv(ss.SPOUT_SCALING_ENV, raising=False)
    XE, YE, GE, XU, YU, GU, kept, pre_i = _pool()
    rng = np.random.default_rng(3)
    post = [i for i in range(len(kept)) if i not in pre_i]
    m, _ = fd.frozen_decoder_models(XE, YE, GE, kept, pre_i, align="cue", source="roi", log=lambda *a: None)
    A0, B0 = fd.frozen_scored_frame(m, XE, GE, XU, GU, kept, pre_i)
    # post labels never enter the models or the frame
    YEs, YUs = YE.copy(), YU.copy()
    me = np.isin(GE, post)
    YEs[me] = rng.permutation(YEs[me])
    YUs[:] = rng.permutation(YUs)
    m1, _ = fd.frozen_decoder_models(XE, YEs, GE, kept, pre_i, align="cue", source="roi", log=lambda *a: None)
    A1, B1 = fd.frozen_scored_frame(m1, XE, GE, XU, GU, kept, pre_i)
    assert np.array_equal(A0, A1) and np.array_equal(B0, B1)
    # a held-out PRE session's own labels do not reach ITS scored rows
    YEp = YE.copy()
    mp = GE == pre_i[0]
    YEp[mp] = rng.permutation(YEp[mp])
    m2, _ = fd.frozen_decoder_models(XE, YEp, GE, kept, pre_i, align="cue", source="roi", log=lambda *a: None)
    A2, _B2 = fd.frozen_scored_frame(m2, XE, GE, XU, GU, kept, pre_i)
    assert np.array_equal(A0[mp], A2[mp])


def test_bundle_frame_ignores_scored_labels(monkeypatch):
    """The same property through `grant_kit._frozen_frame`, the path behind families 5* / acc_by_position."""
    monkeypatch.delenv(ss.SPOUT_SCALING_ENV, raising=False)
    XE, YE, GE, XU, YU, GU, kept, pre_i = _pool()

    def bundle(ye):
        return {"XE": XE, "YE": ye, "GE": GE, "XU": XU, "GU": GU, "kept": kept, "pre_i": set(pre_i),
                "e_pre": np.isin(GE, pre_i)}
    a = grant_kit._frozen_frame(bundle(YE))
    ys = YE.copy()
    me = ~np.isin(GE, pre_i)
    ys[me] = np.random.default_rng(5).permutation(ys[me])
    b = grant_kit._frozen_frame(bundle(ys))
    assert np.array_equal(a["XEs"], b["XEs"]) and np.array_equal(a["XUs"], b["XUs"])


# ------------------------------------------------------------------ (c) the point of it
def _per_class(y, p):
    return np.array([np.mean(p[y == c] == c) for c in range(K)])


def test_bal_em_removes_the_mix_bias_session_z_has(fake_store, monkeypatch):
    """Post sessions' ENGAGED set has spout 5 at 2 % (pre: 1/6); spout-5 trials are scored mostly from the no-lick arm,
    in the engaged frame (the acute far-contra cell). The engaged-trial z-score pulls every trial toward spout 5's
    pattern; bal_em's label-free centre lands near the ORACLE centre (true class means, the same bal_em-trained model)
    and removes most of the per-spout bias."""
    XE, YE, GE, XU, YU, GU, kept, pre_i = _pool(seed=11)
    post = [i for i in range(len(kept)) if i not in pre_i]
    res = {}
    for sc in (None, "session_bal_em"):
        m, _ = fd.frozen_decoder_models(XE, YE, GE, kept, pre_i, align="cue", source="roi", log=lambda *a: None,
                                        scaling=sc)
        A, B = fd.frozen_scored_frame(m, XE, GE, XU, GU, kept, pre_i)
        y = np.concatenate([np.concatenate([YE[GE == i], YU[GU == i]]) for i in post])
        X = np.vstack([np.vstack([A[GE == i], B[GU == i]]) for i in post])
        res[sc] = _per_class(y, m["full"].predict(X))
        if sc:
            # the oracle: the same bal_em-trained model, each post session centred on its TRUE class means
            Xo = []
            for i in post:
                c = label_shift.balanced_centre(XE[GE == i], YE[GE == i], range(K))
                Xo.append(np.vstack([XE[GE == i] - c, XU[GU == i] - c]))
            res["oracle"] = _per_class(y, m["full"].predict(np.vstack(Xo)))
    bias_z = np.abs(res[None] - res["oracle"]).sum()
    bias_b = np.abs(res["session_bal_em"] - res["oracle"]).sum()
    # the z-score centre leaves spout 5, so the PRESENT spouts are pulled toward it and under-read (the near-position
    # over-statement of the real check), while spout 5 itself is not under-read
    assert res[None][:5].mean() < res["oracle"][:5].mean() - 0.03
    assert res[None][5] >= res["oracle"][5]
    assert bias_b < 0.5 * bias_z                              # bal_em removes most of the bias (measured ~.25 -> ~.06)


# ------------------------------------------------------------------ (d) one definition
def test_state_decoder_reads_the_shared_primitives():
    assert ld.bal_em_centre is label_shift.bal_em_centre
    assert ld.saerens_em is label_shift.saerens_em
    assert ld._session_sd is label_shift.session_sd
    assert ld._proba_full is label_shift.proba_full


# ------------------------------------------------------------------ (e) the extension to the other cross-session decoders
def _pipe_factory():
    return fd._pipe()


def test_frozen_frame_session_z_is_the_old_inline_fit(monkeypatch):
    """Under session_z `FrozenFrame` hands back the production arrays and the inline fit the callers always made."""
    monkeypatch.setenv(ss.SPOUT_SCALING_ENV, "session_z")
    XE, YE, GE, XU, _YU, GU, kept, pre_i = _pool()
    ff = ss.FrozenFrame(XE, YE, GE, pre_i, _pipe_factory, loso=False)
    assert ff.XT is ff.XE and ff.apply(XE, 6) is XE and ff.apply_rows(XU, GU) is XU
    tr = np.isin(GE, pre_i)
    assert np.array_equal(ff.full.predict(XE), fd._pipe().fit(XE[tr], YE[tr]).predict(XE))


def test_frozen_frame_scored_rows_ignore_scored_labels(monkeypatch):
    """Post labels shuffled -> the scored rows are bit-identical (six-way and a class-filtered four-way model)."""
    monkeypatch.delenv(ss.SPOUT_SCALING_ENV, raising=False)
    XE, YE, GE, XU, _YU, GU, kept, pre_i = _pool()
    post = ~np.isin(GE, pre_i)
    ys = YE.copy()
    ys[post] = np.random.default_rng(9).permutation(ys[post])
    for classes in (None, [0, 1, 2, 4]):
        tm = None if classes is None else np.isin(YE, classes)     # training rows only read on pre sessions
        a = ss.FrozenFrame(XE, YE, GE, pre_i, _pipe_factory, classes=classes, train_mask=tm)
        b = ss.FrozenFrame(XE, ys, GE, pre_i, _pipe_factory, classes=classes, train_mask=tm)
        for i in (5, 6, 7):
            assert np.array_equal(a.apply(XU[GU == i], i), b.apply(XU[GU == i], i))
            assert np.array_equal(a.apply(XE[GE == i], i), b.apply(XE[GE == i], i))
        assert a.scaling == "session_bal_em" and not np.array_equal(a.XT, XE)


def test_state_frames_are_off_by_default_and_label_free(monkeypatch):
    """The state-class bal_em for the engagement discriminators is OPT-IN (it fails their own pre-stroke check,
    DECISIONS 2026-11-03): by default, even with the spout decoder on bal_em, the frames are the input arrays."""
    monkeypatch.delenv(ss.SPOUT_SCALING_ENV, raising=False)
    monkeypatch.delenv(ss.STATE_DISC_SCALING_ENV, raising=False)
    assert ss.spout_scaling() == "session_bal_em" and ss.state_disc_scaling() is None
    XE, YE, GE, XU, YU, GU, kept, pre_i = _pool()
    se, su = np.ones(len(XE), bool), np.ones(len(XU), bool)
    A, B = ss.state_train_frame(XE, GE, XU, GU, se, su, pre_i)
    assert A is XE and B is XU
    monkeypatch.setenv(ss.STATE_DISC_SCALING_ENV, "state_bal_em")
    from sklearn.linear_model import LogisticRegression
    yd = np.r_[np.ones(len(XE)), np.zeros(len(XU))]
    m = LogisticRegression(max_iter=2000).fit(np.vstack([XE, XU]), yd)
    pi = ss.model_prior(m, yd)
    A1, B1 = ss.state_scored(m, pi, XE, GE, XU, GU, se, su, [5, 6, 7])
    # the centre comes from the session's rows alone: position labels never enter, and other sessions are untouched
    assert np.array_equal(A1[GE < 5], XE[GE < 5])
    assert not np.array_equal(B1[GU == 5], XU[GU == 5])
    for fn in (ss.state_scored, ss.state_centre, ss.FrozenFrame.apply, ss.FrozenFrame.centre):
        assert not set(inspect.signature(fn).parameters) & {"y", "YE", "YU", "labels", "y_true"}, fn.__name__


def test_lick_helpers_are_identity_by_default(monkeypatch):
    from wfield_local import poststroke_compare as pc
    monkeypatch.delenv(ss.STATE_DISC_SCALING_ENV, raising=False)
    XE, YE, GE, XU, YU, GU, kept, pre_i = _pool()
    d = {"XE": XE, "YE": YE, "GE": GE, "XU": XU, "YU": YU, "GU": GU, "kept": kept, "pre_i": set(pre_i),
         "post_i": {5, 6, 7}}
    assert pc._lick_train(d, [0, 1, 2, 3, 4, 5]) is d
    A, B = pc._lick_scored(d, [0, 1], None, None, [5], d)
    assert A is XE and B is XU


# ------------------------------------------------------------------ (f) items B / C (Priya 11-03)
def test_spout_frame_mode_needs_the_spout_decoder_on_bal_em(monkeypatch):
    monkeypatch.setenv(ss.STATE_DISC_SCALING_ENV, "spout_frame")
    monkeypatch.delenv(ss.SPOUT_SCALING_ENV, raising=False)
    assert ss.state_disc_scaling() == ss.SCALING_SPOUT_FRAME
    monkeypatch.setenv(ss.SPOUT_SCALING_ENV, "session_z")
    assert ss.state_disc_scaling() is None


def test_projection_arrays_identity_when_off_and_label_free_when_on(monkeypatch):
    XE, YE, GE, XU, YU, GU, kept, pre_i = _pool()
    monkeypatch.setenv(ss.PROJECTION_FRAME_ENV, "session_z")
    A, B = ss.projection_arrays(XE, YE, GE, XU, GU, pre_i, _pipe_factory)
    assert A is XE and B is XU
    monkeypatch.setenv(ss.PROJECTION_FRAME_ENV, "spout_frame")
    monkeypatch.delenv(ss.SPOUT_SCALING_ENV, raising=False)
    A1, B1 = ss.projection_arrays(XE, YE, GE, XU, GU, pre_i, _pipe_factory)
    ys = YE.copy()
    post = ~np.isin(GE, pre_i)
    ys[post] = np.random.default_rng(4).permutation(ys[post])
    A2, B2 = ss.projection_arrays(XE, ys, GE, XU, GU, pre_i, _pipe_factory)
    assert np.array_equal(A1, A2) and np.array_equal(B1, B2)
    assert not np.array_equal(A1, XE)


def test_frozen_frame_train_apply_matches_the_training_rows(monkeypatch):
    monkeypatch.delenv(ss.SPOUT_SCALING_ENV, raising=False)
    XE, YE, GE, XU, YU, GU, kept, pre_i = _pool()
    ff = ss.FrozenFrame(XE, YE, GE, pre_i, _pipe_factory, loso=False)
    for i in pre_i:
        assert np.allclose(ff.train_apply(XE[GE == i], i), ff.XT[GE == i], atol=0, rtol=0)
    assert ff.train_apply(XE[GE == 6], 6) is not None
