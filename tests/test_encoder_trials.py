"""Per-trial outputs of the combined encoder (Priya 11-04, `<label>_trials.npz`; wfield_local/encoder_trials.py): every
cue trial is scored (fitted trials held out, the others by the full-data model, never in training), the session outputs
do not change, and the reader rebuilds them per class. Synthetic session, no share access."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wfield_local import combined_encoding as ce
from wfield_local import encoder_trials as et

POS = ("close_contra", "close_ipsi", "close_mid", "far_contra", "far_ipsi", "far_mid")


def _inputs(n_trials: int = 72, seed: int = 0) -> dict:
    """Trials every 8 s; trials 4, 11, 18, ... and a block at the end are NOT engaged (scored only); every third
    engaged trial has no lick (alignment falls back to its DAQ contact on half of them)."""
    rng = np.random.default_rng(seed)
    cue = 10.0 + 8.0 * np.arange(n_trials) + rng.uniform(-0.3, 0.3, n_trials)
    pos = np.array([POS[i % 6] for i in rng.permutation(n_trials)])
    eng = np.ones(n_trials, bool)
    eng[3::7] = False
    eng[-6:] = False
    T = pd.DataFrame({"trial_id": np.arange(1, n_trials + 1), "pos": pos, "strobe_s": cue - 3.0, "cue_s": cue,
                      "stop_s": cue + 3.0, "engaged": eng})
    ft = np.arange(0.0, cue[-1] + 10.0, 0.1)
    lick_tr = np.flatnonzero(np.arange(n_trials) % 3 != 2)
    lick = np.sort(cue[lick_tr] + 0.4 + rng.uniform(0, 0.3, len(lick_tr)))
    # contacts: one per lick; on trials without a lick (some); and EARLY ones (cue + 0.1 s, > 50 ms before the LP onset)
    contact = np.sort(np.r_[lick + 0.02, cue[np.arange(n_trials) % 6 == 2] + 0.5, cue[np.arange(n_trials) % 6 == 0] + 0.1])
    an = {k: (ft, rng.normal(size=len(ft))) for k in ("tongue_x", "tongue_y", "tongue_prot", "tongue_speed", "jaw_x",
                                                     "jaw_y", "jaw_speed", "nose_x", "nose_y", "running")}
    B = ce.Behaviour(T, lick, lick + 0.05, np.array([pos[np.searchsorted(cue, x) - 1] for x in lick]),
                     np.zeros(len(lick)), contact, np.zeros(0), an, "test")
    K, npx = 6, 16
    V = np.zeros((len(ft), K))
    for c_, p_ in zip(cue, pos):
        m = (ft >= c_) & (ft < c_ + 2.0)
        V[m, POS.index(p_) % K] += np.exp(-(ft[m] - c_))
    for x in lick:
        m = (ft >= x) & (ft < x + 1.0)
        V[m, 1] += 0.5
    V += 0.1 * rng.normal(size=V.shape) + 0.002 * ft[:, None]
    Ub = rng.normal(size=(npx, K))
    areas = {"a1": np.arange(0, 6), "a2": np.arange(6, 12), "a3": np.arange(10, 16)}
    am = {"areas": areas, "cortex": np.arange(npx), "tag": "test", "version": "t", "split": ["a1", "a2"],
          "plain": ["a3"]}
    pstat = np.arange(2, npx)
    imaging = (V.astype(np.float32), ft, Ub, np.arange(npx), np.zeros(npx), ["a1", "a2", "a3"], (4, 4), pstat)
    return {"label": "PS93_0814", "imaging": imaging, "behaviour": B, "area_masks": am, "epoch": "pre", "day": -4,
            "video": None, "fr": None}


@pytest.fixture(scope="module")
def runs():
    import os
    old = {k: os.environ.get(k) for k in (ce.TRIALS_ENV, ce.SMOKE_ENV, ce.ONLY_ENV)}
    os.environ[ce.SMOKE_ENV] = "1"
    os.environ.pop(ce.ONLY_ENV, None)
    out = {}
    try:
        for mode in ("off", "full32", "full"):
            os.environ[ce.TRIALS_ENV] = mode
            out[mode] = ce.run_session("PS93_0814", inputs=_inputs())
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return out


SESSION_KEYS = ("peth", "peth_lick", "sse", "sst", "sse_trial", "sst_trial", "sse_lick", "sst_lick", "sse_trial_lick",
                "sst_trial_lick")


def test_session_outputs_identical_with_trials_on(runs):
    a, b = runs["off"], runs["full32"]
    for k in SESSION_KEYS:
        np.testing.assert_array_equal(a[k], b[k], err_msg=k)
    assert sorted(a["maps"]) == sorted(b["maps"])
    for k in a["maps"]:
        np.testing.assert_array_equal(a["maps"][k], b["maps"][k], err_msg=str(k))
    pd.testing.assert_frame_equal(a["region"], b["region"])
    assert "trials" not in a and "trials" in b


def _z(out, tmp_path):
    out = dict(out, res=dict(out["res"], label="PS93_0814"))
    ce.save(out, tmp_path)
    return et.load(tmp_path / "sessions" / "PS93_0814_trials.npz")


def test_every_cue_trial_scored_and_never_trained(runs, tmp_path):
    z = _z(runs["full32"], tmp_path)
    t = z["table"]
    T = _inputs()["behaviour"].trials
    assert len(t) == len(T) and (t.trial_id.to_numpy() == T.trial_id.to_numpy()).all()
    assert (t.fitted == T.engaged).all()
    assert (t.fold[~t.fitted] == -1).all() and (t.fold[t.fitted] >= 0).all()
    assert runs["full32"]["res"]["n_trials"] == int(T.engaged.sum())          # the fit saw only engaged trials
    mi = list(z["tr_models"]).index("full")
    nf = ~t.fitted.to_numpy()
    assert np.isfinite(z["sse_bins_cue"][mi][nf]).all()                        # scored-only trials are scored ...
    for m, ok in zip(z["tr_models"], z["tr_scorable"]):
        if str(m).startswith("shuf"):                                          # ... except by shuffle models
            assert not ok and np.isnan(z["sse_bins_cue"][list(z["tr_models"]).index(m)][nf]).all()
    # the never-fitted trials are predicted about as well as the held-out fitted ones (same generative model)
    r2 = {k: 1 - (s := et.class_sse(z, m, "cue"))["sse"][mi].sum() / s["sst"].sum() for k, m in (("x", nf), ("f", ~nf))}
    assert r2["f"] > 0.2 and r2["x"] > 0.5 * r2["f"], r2


def test_reader_rebuilds_session_outputs_from_engaged_classes(runs, tmp_path):
    """Two engaged 'classes' summed = the session's peth / sse (exact up to float32 storage)."""
    out = runs["full32"]
    z = _z(out, tmp_path)
    t = z["table"]
    fit = t.fitted.to_numpy()
    c1 = fit & (np.arange(len(t)) % 2 == 0)
    c2 = fit & ~c1
    pe, parts, positions = et.class_peth(z, fit, "cue")
    assert parts[:-1] == out["res"]["peth_parts"] and parts[-1] == "cv" and positions == out["res"]["pos_order"]
    np.testing.assert_array_equal(pe[:-1], out["peth"])
    lp = fit & (t.lplick_row >= 0).to_numpy()
    pl, _, _ = et.class_peth(z, lp, "lplick")
    np.testing.assert_array_equal(pl[:-1], out["peth_lick"])                  # the LP onset = the npz's peth_lick
    b = z["bin_of_off"]
    nb = len(z["bin_edges_s"]) - 1
    same = (t.lick_row == t.lplick_row).to_numpy()                            # binned lick SSE: this file's timing
    assert (lp & ~same).any()
    for align, key, m in (("cue", "sse", fit),):
        s1, s2 = et.class_sse(z, c1 & m, align), et.class_sse(z, c2 & m, align)
        want = np.stack([out[key][:, :, b == q].sum(2) for q in range(nb)], 1)        # (models, bins, areas)
        np.testing.assert_allclose(s1["sse"] + s2["sse"], want, rtol=1e-5)
        wst = np.stack([out["sst" + key[3:]][:, b == q].sum(1) for q in range(nb)], 0)
        np.testing.assert_allclose(s1["sst"] + s2["sst"], wst, rtol=1e-5)
    f = et.class_full_fine(z, fit, "cue")
    mi = out["res"]["tr_models"].index("full")
    np.testing.assert_allclose(f["sse"], out["sse"][mi], rtol=1e-5)
    np.testing.assert_allclose(f["sse_trial"], out["sse_trial"][mi], rtol=1e-5)
    np.testing.assert_allclose(f["sst_trial"], out["sst_trial"], rtol=1e-5)
    fl = et.class_full_fine(z, lp, "lplick")                                    # lick-aligned, the npz's timing
    np.testing.assert_allclose(fl["sse"], out["sse_lick"][mi], rtol=1e-5)
    np.testing.assert_allclose(fl["sse_trial"], out["sse_trial_lick"][mi], rtol=1e-5)
    np.testing.assert_allclose(fl["sst"], out["sst_lick"], rtol=1e-5)


def test_lick_alignment_falls_back_to_daq_contact(runs, tmp_path):
    z = _z(runs["full32"], tmp_path)
    t = z["table"]
    src = t.timing_source.to_numpy()
    assert set(src) == {"lp", "daq", "none"}
    d = src == "daq"
    assert (t.first_lick_s[d] == t.daq_first_contact_s[d]).all() and (t.daq_first_contact_s[d] >= t.cue_s[d] + 0.07).all()
    close = t.pos.str.startswith("close").to_numpy()
    late = (t.lp_first_lick_s - t.daq_first_contact_s > ce.LP_DAQ_TOL_S).to_numpy()
    assert (src[close & late] == "daq").all() and (close & late).any()               # close + late LP -> the contact
    assert (src[~close & late] == "lp").all() and t.grooming_suspect[~close & late].all() and (~close & late).any()
    assert not t.grooming_suspect[~late].any()
    nolp = np.isnan(t.lp_first_lick_s.to_numpy()) & np.isfinite(t.daq_first_contact_s.to_numpy())
    assert (src[nolp] == "daq").all() and nolp.any()                                  # no LP onset -> the contact
    assert ((src == "lp") == (t.first_lick_s == t.lp_first_lick_s)).all()
    assert (t.lick_row[src == "none"] == -1).all() and (t.lick_row[src != "none"] >= 0).all()


def test_float16_frames_close(runs, tmp_path):
    z = _z(runs["full"], tmp_path / "h")
    assert z["frames"].dtype == np.float16
    pe, _, _ = et.class_peth(z, z["table"].fitted.to_numpy(), "cue")
    ref = runs["full32"]["peth"]
    assert np.nanmax(np.abs(pe[:-1] - ref)) < 2e-3 * np.nanmax(np.abs(ref))


def test_class_summary_and_join(runs, tmp_path):
    z = _z(runs["full32"], tmp_path)
    t = z["table"]
    A = pd.DataFrame({"label": "PS93_0814", "trial_id": t.trial_id,
                      "cls": np.where(t.fitted, np.where(t.has_lick if "has_lick" in t else t.timing_source == "lp",
                                                         "hit", "incomplete"), "stopped"),
                      "cls_raw": "x", "attempted": t.fitted, "canonical": True})
    zj = et.join_classes(z, "PS93_0814", attempts=A)
    S = et.class_summary(zj)
    assert list(S.cls) == ["hit", "incomplete", "stopped"]
    assert S.n.sum() == len(t) and (S.cvR2 < 1).all() and np.isfinite(S.trial_R2).all()


def test_helpers_extend_exactly():
    rng = np.random.default_rng(3)
    t = np.sort(rng.uniform(0, 3000, 500))
    S = ce._slow_basis(t, 600.0)
    np.testing.assert_array_equal(ce._slow_basis(t, 600.0, at=t), S)
    X, E = rng.normal(size=(500, 4)).astype(np.float32), rng.normal(size=(500, 3)).astype(np.float32)
    Z, Zx = ce._residualise(X, E, slice(None), extra=(X[:50], E[:50]))
    np.testing.assert_array_equal(Z, ce._residualise(X, E, slice(None)))
    np.testing.assert_allclose(Zx, Z[:50], atol=1e-5)
    np.testing.assert_array_equal(ce._z(X, slice(None)), ce._zapply(X, *ce._zstats(X, slice(None))))


def test_trials_mode_default(monkeypatch):
    monkeypatch.delenv(ce.TRIALS_ENV, raising=False)
    monkeypatch.delenv(ce.ONLY_ENV, raising=False)
    assert ce.trials_mode() == "auto"
    assert ce.auto_trials_mode("movement_first", ce.MAIN_TR_MODELS) == "full"            # main
    monkeypatch.setenv(ce.ONLY_ENV, "drop:")
    assert ce.auto_trials_mode("movement_first", ("full", "drop:task", "drop:position")) == "off"   # main has them
    assert ce.auto_trials_mode("movement_first", ("full",)) == "off"                     # droppos / shufpos / nodir
    assert ce.auto_trials_mode("musall", ("full", "shuf:task")) == "sse"                 # another order
    assert ce.auto_trials_mode("movement_first", ("full", "nokin:none")) == "sse"        # bins_*
    assert ce.auto_trials_mode("movement_first", ("full", "fr:full")) == "sse"           # fr_*
    monkeypatch.setenv(ce.TRIALS_ENV, "off")
    assert ce.trials_mode() == "off"
    monkeypatch.setenv(ce.TRIALS_ENV, "bogus")
    with pytest.raises(ValueError):
        ce.trials_mode()


# ------------------------------------------------------------------------------------ engagement from a shipped table
def _attempts_of(B, label="PS93_0814", canonical=True):
    T = B.trials
    return pd.DataFrame({"label": label, "trial_id": T.trial_id, "cls": np.where(T.engaged, "hit", "stopped"),
                         "attempted": T.engaged.to_numpy(), "canonical": canonical})


def test_rederive_engaged_reproduces_baked_flags(tmp_path, monkeypatch):
    from wfield_local import orofacial_cohort as oc
    monkeypatch.setattr(oc, "ENGAGEMENT_GATE", "attempted")
    B = _inputs()["behaviour"]
    f = tmp_path / "trial_attempts.csv"
    A = _attempts_of(B)
    pd.concat([A, A.assign(label="PS92_0806")]).sample(frac=1, random_state=0).to_csv(f, index=False)   # other rows
    B2 = ce.rederive_engaged(B, "PS93_0814", f)
    assert B2.trials.engaged.dtype == bool and (B2.trials.engaged.to_numpy() == B.trials.engaged.to_numpy()).all()
    assert B2.trials is not B.trials and B.trials.engaged.sum() < len(B.trials)                # B itself untouched
    # a changed rule shows up; canonical_attempted also needs canonical
    A2 = A.copy()
    A2.loc[0, "attempted"] = not A2.loc[0, "attempted"]
    A2.to_csv(f, index=False)
    assert ce.rederive_engaged(B, "PS93_0814", f).trials.engaged.iloc[0] != B.trials.engaged.iloc[0]
    _attempts_of(B, canonical=False).to_csv(f, index=False)
    assert not ce.rederive_engaged(B, "PS93_0814", f, gate="canonical_attempted").trials.engaged.any()


def test_rederive_engaged_refuses(tmp_path, monkeypatch):
    from wfield_local import orofacial_cohort as oc
    monkeypatch.setattr(oc, "ENGAGEMENT_GATE", "attempted")
    B = _inputs()["behaviour"]
    f = tmp_path / "t.csv"
    _attempts_of(B).iloc[:-2].to_csv(f, index=False)                         # two trials without a row
    with pytest.raises(ValueError, match="2 of 72 trials"):
        ce.rederive_engaged(B, "PS93_0814", f)
    _attempts_of(B).to_csv(f, index=False)
    with pytest.raises(ValueError, match="trials of PS93_0815"):
        ce.rederive_engaged(B, "PS93_0815", f)
    with pytest.raises(ValueError, match="reference"):
        ce.rederive_engaged(B, "PS93_0814", f, gate="reference")


def test_run_session_with_shipped_table_is_identical(tmp_path, monkeypatch):
    """ATTEMPTS_ENV with the table the flags were baked from: the same session outputs; the table is recorded."""
    from wfield_local import orofacial_cohort as oc
    monkeypatch.setattr(oc, "ENGAGEMENT_GATE", "attempted")
    monkeypatch.setenv(ce.SMOKE_ENV, "1")
    monkeypatch.setenv(ce.TRIALS_ENV, "off")
    monkeypatch.setenv(ce.ONLY_ENV, "trmodels")
    x = _inputs(n_trials=48)
    a = ce.run_session("PS93_0814", inputs=x)
    f = tmp_path / "trial_attempts.csv"
    _attempts_of(x["behaviour"]).to_csv(f, index=False)
    monkeypatch.setenv(ce.ATTEMPTS_ENV, str(f))
    b = ce.run_session("PS93_0814", inputs=x)
    for k in SESSION_KEYS:
        np.testing.assert_array_equal(a[k], b[k], err_msg=k)
    pd.testing.assert_frame_equal(a["region"], b["region"])
    assert b["res"]["attempts_table"] == {"path": str(f), "sha256": ce.file_sha256(f)}
    assert "attempts_table" not in a["res"]
