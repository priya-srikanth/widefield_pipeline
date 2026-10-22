"""frozen_encoder.fit_animal: the Gram-matrix ridge (pooled-pre scaling, per-group penalties, leave-one-session-out)
must equal a direct ridge on the stacked, scaled pre sessions."""
import numpy as np

from wfield_local import frozen_encoder as fe

PEN = {"a": "task", "b": "lick"}


def _sessions(rng, n_sess=3, n=400, p=6, r=3):
    col_var = np.array(["a"] * 3 + ["b"] * (p - 3))
    W = rng.normal(size=(p, r))
    out = []
    for k in range(n_sess):
        X = rng.normal(size=(n, p)) * (1 + k)            # session-dependent scale: the pooled SD must be used
        X -= X.mean(0)
        Y = X @ W + 0.5 * rng.normal(size=(n, r))
        Y -= Y.mean(0)
        out.append((X, Y, {"label": f"S{k}", "animal": "A", "epoch": "pre", "col_var": col_var,
                           "area_names": ["x", "y", "cortex"], "n": n, "XtX": X.T @ X, "XtY": X.T @ Y,
                           "YtY": (Y ** 2).sum(0), "sxx": (X ** 2).sum(0),
                           "pos_order": ["p"], "pos_counts": {"p": n}}))
    return out


def _ridge(X, Y, lam):
    return np.linalg.solve(X.T @ X + np.diag(lam) + 1e-8 * np.eye(X.shape[1]), X.T @ Y)


def test_beta_all_and_loso_match_direct_ridge():
    rng = np.random.default_rng(0)
    ss = _sessions(rng)
    f = fe.fit_animal([s for *_, s in ss], alpha_grid=(1.0, 100.0), pen_of=PEN)
    Xs = np.vstack([x for x, _, _ in ss])
    Ys = np.vstack([y for _, y, _ in ss])
    sd = np.sqrt((Xs ** 2).sum(0) / len(Xs))
    np.testing.assert_allclose(f["scale"], sd)
    lam = np.array([f["alphas"][PEN[v]] for v in ss[0][2]["col_var"]])
    np.testing.assert_allclose(f["beta_all"], _ridge(Xs / sd, Ys, lam), rtol=1e-8, atol=1e-10)
    Xo = np.vstack([x for x, _, s in ss if s["label"] != "S1"]) / sd
    Yo = np.vstack([y for _, y, s in ss if s["label"] != "S1"])
    np.testing.assert_allclose(f["beta_loso"]["S1"], _ridge(Xo, Yo, lam), rtol=1e-8, atol=1e-10)


def test_loso_r2_matches_direct_prediction():
    rng = np.random.default_rng(1)
    ss = _sessions(rng)
    f = fe.fit_animal([s for *_, s in ss], alpha_grid=(10.0,), pen_of=PEN)
    err = tot = 0.0
    for x, y, s in ss:
        yh = (x / f["scale"]) @ f["beta_loso"][s["label"]]
        err += ((y - yh) ** 2).sum()
        tot += (y ** 2).sum()
    assert abs(f["loso_r2"] - (1 - err / tot)) < 1e-9


def test_layout_mismatch_refused():
    rng = np.random.default_rng(2)
    ss = [s for *_, s in _sessions(rng)]
    ss[1] = {**ss[1], "col_var": ss[1]["col_var"][::-1].copy()}
    try:
        fe.fit_animal(ss, alpha_grid=(1.0,), pen_of=PEN)
    except ValueError:
        return
    raise AssertionError("a session with a different column layout was accepted")


def test_ordered_variants_same_span_as_joint():
    """movement_first / task_first re-express the same columns: with ~no penalty the full prediction equals joint's;
    no_movement / movement_only keep only their groups' columns."""
    rng = np.random.default_rng(3)
    ss = _sessions(rng)
    for _x, _y, s in ss:
        s["col_var"] = np.array(["cue"] * 3 + ["lick_onset"] * 3)
    st = [s for *_, s in ss]
    X0 = ss[0][0] / fe.fit_animal(st, alpha_grid=(1e-6,))["scale"]
    pred = {}
    for v in fe.VARIANTS:
        f = fe.fit_animal(st, alpha_grid=(1e-6,), variant=v)
        pred[v] = (X0 @ f["C"]) @ f["beta_all"]
        if v == "no_movement":
            assert list(f["vcol"]) == ["cue"] * 3
        if v == "movement_only":
            assert list(f["vcol"]) == ["lick_onset"] * 3
    np.testing.assert_allclose(pred["movement_first"], pred["joint"], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(pred["task_first"], pred["joint"], rtol=1e-6, atol=1e-8)


def test_split_blocks_layout():
    """The encoder's licks / tongue / jaw / nose blocks are split by their fixed column order."""
    from wfield_local import combined_encoding as ce
    nl, T = len(ce.CLAGS), 7
    raw = {v: np.full((T, 4), i, float) for i, v in enumerate(("cue", "arrival", "pos_enl", "pos_cue"))}
    raw["licks"] = np.hstack([np.zeros((T, 5)), np.ones((T, 5))])          # [onset, contact]
    raw["exec_angle"] = np.zeros((T, 5))
    raw["running"] = np.zeros((T, nl))
    raw["tongue"] = np.repeat(np.arange(4.0), nl)[None, :].repeat(T, 0)    # x, y, prot, speed
    raw["jaw"] = np.repeat(np.arange(3.0), nl)[None, :].repeat(T, 0)       # x, y, speed
    raw["nose"] = np.repeat(np.arange(2.0), nl)[None, :].repeat(T, 0)      # x, y
    out = fe.split_blocks(raw)
    assert set(out) == set(fe.FGROUP)
    assert (out["lick_onset"] == 0).all() and (out["contact"] == 1).all()
    assert set(out["tongue_dir"][0]) == {0.0, 1.0, 2.0} and set(out["tongue_mag"][0]) == {3.0}   # prot is spatial
    assert set(out["jaw_dir"][0]) == {0.0} and set(out["jaw_mag"][0]) == {1.0, 2.0}
    assert set(out["nose_dir"][0]) == {0.0} and set(out["nose_mag"][0]) == {1.0}


def test_drop_transform_reproduces_encoder_drop_columns():
    """D @ C (drop p) must equal combined_encoding's drop_pos columns X_q - m_q/(1-m_p) (shared - X_p), q != p."""
    rng = np.random.default_rng(4)
    order = ["a", "b", "c"]
    n, w = 50, 2
    frac = {"a": 0.2, "b": 0.3, "c": 0.5}
    Xq = {q: rng.normal(size=(n, w)) for q in order}           # per-position kernels (one var)
    shared = sum(Xq.values())
    D = np.hstack([Xq[q] - frac[q] * shared for q in order])   # deviation coding (raw_blocks)
    extra = rng.normal(size=(n, 1))
    for var in ("pos_enl", "pos_cue"):
        col_var = np.array(["cue"] + [var] * (w * len(order)) + ["pos_enl" if var == "pos_cue" else "pos_cue"] * 0)
        Xfull = np.hstack([extra, D])
        C, keep = fe._drop_transform(col_var, order, frac, "b")
        got = Xfull @ C
        want = np.hstack([extra] + [Xq[q] - frac[q] / (1 - frac["b"]) * (shared - Xq["b"]) for q in order if q != "b"])
        np.testing.assert_allclose(got, want, atol=1e-12)
        assert list(col_var[keep]) == ["cue"] + [var] * (w * 2)


def test_musall_r2_splits_condition_and_single_trial():
    """A prediction equal to each spout's mean response: avg R2 = 1, trial R2 = 0 (nothing single-trial)."""
    rng = np.random.default_rng(5)
    pos = np.repeat(["a", "b", "c"], 20)
    mean = {q: rng.normal(size=(8, 2)) for q in "abc"}
    D = np.stack([mean[q] + 0.3 * rng.normal(size=(8, 2)) for q in pos])
    P = np.stack([mean[q] for q in pos])
    D_mean = {q: D[pos == q].mean(0) for q in "abc"}
    P = np.stack([D_mean[q] for q in pos])                       # the data's own condition means
    r = fe.musall_r2(D, P, pos)
    np.testing.assert_allclose(r["avg"], 1.0, atol=1e-12)
    np.testing.assert_allclose(r["trial"], 0.0, atol=1e-12)
    assert (r["frame"] > 0).all() and (r["frame"] < 1).all()
    r2 = fe.musall_r2(D, D.copy(), pos)                          # perfect prediction
    for k in ("frame", "trial", "avg"):
        np.testing.assert_allclose(r2[k], 1.0, atol=1e-12)
