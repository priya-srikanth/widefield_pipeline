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
                           "YtY": (Y ** 2).sum(0), "sxx": (X ** 2).sum(0)}))
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
        s["col_var"] = np.array(["cue"] * 3 + ["licks"] * 3)
    st = [s for *_, s in ss]
    X0 = ss[0][0] / fe.fit_animal(st, alpha_grid=(1e-6,))["scale"]
    pred = {}
    for v in fe.VARIANTS:
        f = fe.fit_animal(st, alpha_grid=(1e-6,), variant=v)
        pred[v] = (X0 @ f["C"]) @ f["beta_all"]
        if v == "no_movement":
            assert list(f["vcol"]) == ["cue"] * 3
        if v == "movement_only":
            assert list(f["vcol"]) == ["licks"] * 3
    np.testing.assert_allclose(pred["movement_first"], pred["joint"], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(pred["task_first"], pred["joint"], rtol=1e-6, atol=1e-8)
