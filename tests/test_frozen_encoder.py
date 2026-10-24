"""frozen_encoder.fit_animal: the Gram-matrix ridge (pooled-pre scaling, per-group penalties, leave-one-session-out)
must equal a direct ridge on the stacked, scaled pre sessions."""
from pathlib import Path

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
        s["col_var"] = np.array(["cue"] * 3 + ["lick_onset:dev"] * 3)
    st = [s for *_, s in ss]
    X0 = ss[0][0] / fe.fit_animal(st, alpha_grid=(1e-6,))["scale"]
    pred = {}
    for v in ("joint", "movement_first", "task_first", "target_first", "exec_first", "no_movement", "movement_only"):
        f = fe.fit_animal(st, alpha_grid=(1e-6,), variant=v)
        pred[v] = (X0 @ f["C"]) @ f["beta_all"]
        if v == "no_movement":
            assert list(f["vcol"]) == ["cue"] * 3
        if v == "movement_only":
            assert list(f["vcol"]) == ["lick_onset:dev"] * 3
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
    # base variables (split later into s / d); "push" is derived later from the signed running columns
    assert set(out) == {v.split(":")[0] for v in fe.FGROUP} - {"push"}
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


def test_fit_animal_job_equals_direct_fits(tmp_path):
    """The per-animal fit job (run in a worker, statistics read from the saved pickle) gives the same fits as calling
    fit_animal in the main process; sessions of other animals / not listed are ignored."""
    import pickle
    rng = np.random.default_rng(5)
    ss = _sessions(rng, n_sess=4)
    for _x, _y, s in ss:
        s["col_var"] = np.array(["cue"] * 3 + ["lick_onset:dev"] * 3)
    st = [s for *_, s in ss]
    other = dict(st[0], label="S9", animal="B")
    sp = tmp_path / "pre_stats.pkl"
    sp.write_bytes(pickle.dumps(st + [other]))
    labs = ("S0", "S1", "S3")
    got = fe.fit_animal_job(("A", str(sp), labs, ("joint", "task_first"), False))
    for v in ("joint", "task_first"):
        ref = fe.fit_animal([s for s in st if s["label"] in labs], variant=v)
        np.testing.assert_array_equal(got[v]["beta_all"], ref["beta_all"])
        assert got[v]["alphas"] == ref["alphas"]


def test_spout_split_recovers_templates_and_sums_back():
    """Spout-locked templates = per spout x cue-relative bin mean of the centred columns; s + d = x exactly; in the
    sessions the template came from, d has zero mean per spout and bin (where the bin has enough trials)."""
    rng = np.random.default_rng(7)
    k0, K = fe._template_bins()
    order = ["a", "b"]
    prof = {"a": np.sin(np.arange(K) / 9.0), "b": 2.0 * np.cos(np.arange(K) / 7.0)}
    rows_p, rows_k, rows_x = [], [], []
    for _tr in range(60):
        for p in order:
            kk = np.arange(k0 + 10, k0 + 200)
            rows_p.append(np.full(len(kk), p, dtype=object))
            rows_k.append(kk)
            rows_x.append((prof[p][kk - k0] + 0.3 * rng.normal(size=len(kk)))[:, None] * np.array([[1.0, -2.0]]))
    pos_of, k_of, X = np.concatenate(rows_p), np.concatenate(rows_k), np.vstack(rows_x)
    S, N = fe.template_sums(X, pos_of, k_of, order)
    M = fe.template_means(S, N)
    for pi, p in enumerate(order):
        got = M[pi, 10:200, 0]
        assert np.abs(got - prof[p][10:200]).max() < 0.2          # noise 0.3 / sqrt(60)
        assert (M[pi, :10] == 0).all()                             # no trials there -> no template
    s, d = fe.spout_split(X, pos_of, k_of, order, M)
    np.testing.assert_allclose(s + d, X)
    S2, N2 = fe.template_sums(d, pos_of, k_of, order)
    ok = N2 >= fe.TEMPLATE_MIN_N
    assert np.abs(S2[ok] / N2[ok][:, None]).max() < 1e-9


def test_build_templates_pools_sessions_by_trial_count():
    """Pooled template = sums / counts over sessions (each trial weighs the same), per animal."""
    k0, K = fe._template_bins()
    def part(label, val, n):
        S = np.zeros((1, K, 1))
        Nn = np.zeros((1, K))
        S[0, 5] = val * n
        Nn[0, 5] = n
        return {"label": label, "animal": "A", "pos_order": ["a"], "sums": {v: (S, Nn) for v in fe.SPLIT_VARS}}
    t = fe.build_templates([part("A_1", 1.0, 30), part("A_2", 4.0, 10)])["A"]
    assert abs(t["mean"]["jaw_mag"][0, 5, 0] - (30 * 1.0 + 10 * 4.0) / 40) < 1e-12
    assert t["n_sessions"] == 2
    assert t["running_cap"] is None                     # no running samples -> no cap (old parts stay valid)


def test_running_cap_pooled_over_pre_frames_and_clips_both_signs():
    """Cap = RUN_CAP_Q percentile of |speed| over the RUNNING pre frames (above the threshold) pooled across sessions;
    stationary frames do not pull it down. Clipping keeps sign, leaves values inside the cap and NaN untouched."""
    k0, K = fe._template_bins()
    def part(label, run):
        S, Nn = np.zeros((1, K, 1)), np.zeros((1, K))
        return {"label": label, "animal": "A", "pos_order": ["a"], "sums": {v: (S, Nn) for v in fe.SPLIT_VARS},
                "run_abs": np.asarray(run, np.float32)}
    r1, r2 = np.r_[np.zeros(100000), np.arange(1000.0)], 5000.0 + np.arange(100.0)    # mostly stationary
    t = fe.build_templates([part("A_1", r1), part("A_2", r2)], run_thresh=3.0)["A"]
    mov = np.concatenate([r1, r2])
    assert abs(t["running_cap"] - np.percentile(mov[mov > 3.0], fe.RUN_CAP_Q)) < 1e-2
    assert t["running_cap"] > 1000                      # the all-frame p99 would be ~0 here
    x = np.array([[-50.0, 3.0], [np.nan, 50.0], [9.9, -2.0]])
    y = fe.cap_running(x, 10.0)
    assert np.array_equal(y[[0, 1, 2], [0, 1, 0]], [-10.0, 10.0, 9.9]) and y[2, 1] == -2.0 and np.isnan(y[1, 0])
    assert fe.cap_running(x, None) is x


def test_ordered_projection_ignores_null_direction():
    """The ordered transform must not blow up along an exactly collinear direction of the A block (the position
    deviation blocks sum to ~0): applied to new data where that direction is no longer null, the task_first
    prediction stays close to joint's (the v3 near-zero-ridge solve gave R^2 -1.06 on a post session)."""
    rng = np.random.default_rng(11)
    n, r = 3000, 2
    a1, a2 = rng.normal(size=n), rng.normal(size=n)
    b = 0.5 * a1 + rng.normal(size=n)
    X = np.column_stack([a1, a2, a1 + a2 + 3e-4 * rng.normal(size=n), b])   # A block with a NEAR-null direction
    Y = X @ rng.normal(size=(4, r)) + 0.1 * rng.normal(size=(n, r))
    col_var = np.array(["cue"] * 3 + ["lick_onset:dev"])
    st = [{"label": "S0", "animal": "A", "epoch": "pre", "col_var": col_var, "area_names": ["x", "cortex"], "n": n,
           "XtX": X.T @ X, "XtY": X.T @ Y, "YtY": (Y ** 2).sum(0), "sxx": (X ** 2).sum(0), "pos_order": ["p"],
           "pos_counts": {"p": n}}]
    fj = fe.fit_animal(st, alpha_grid=(1e-3,), variant="joint")
    ft = fe.fit_animal(st, alpha_grid=(1e-3,), variant="task_first")
    Xn = X.copy()
    Xn[:, 2] += 0.3 * rng.normal(size=n)                            # "post": the null direction is no longer null
    pj = (Xn / fj["scale"]) @ (fj["C"] @ fj["beta_all"])
    pt = (Xn / ft["scale"]) @ (ft["C"] @ ft["beta_all"])
    assert np.abs(ft["C"]).max() < 10
    assert np.linalg.norm(pt - pj) / np.linalg.norm(pj) < 0.2


def test_template_index_switches_from_cue_to_lick_locked():
    """Cue-locked bins before the first post-cue lick (and without one), lick-locked from it; out of span -> k0 - 1."""
    k0, K = fe._template_bins()
    kc = int(np.ceil(fe.TEMPLATE_S[1] / fe.TEMPLATE_BIN_S)) - k0 + 1
    t_cue = np.array([0.32, 1.0, 1.0, -10.0, 0.5])
    t_lick = np.array([np.nan, -0.2, 0.064, np.nan, 99.0])
    k = fe.template_index(t_cue, t_lick)
    assert k[0] == 10 and k[1] == round(1.0 / fe.TEMPLATE_BIN_S)            # cue-locked (no lick / before it)
    assert k[2] == k0 + kc + 2                                               # lick-locked, 64 ms after the lick
    assert k[3] == k0 - 1 and k[4] == k0 - 1                                 # outside both spans
    assert (k[[0, 1, 2]] - k0 < K).all()


def test_build_basis_orthonormal_and_captures_variance():
    """The shared basis: orthonormal columns spanning the pooled sessions' dominant spatial patterns, K chosen by
    BASIS_VAR; a session's own patterns are captured."""
    rng = np.random.default_rng(13)
    npix, k = 500, 10
    common = np.linalg.qr(rng.normal(size=(npix, 3)))[0]
    parts = []
    for s in range(3):
        Ub = np.hstack([common, 0.05 * rng.normal(size=(npix, k - 3))])
        w = np.r_[100.0, 50.0, 20.0, np.full(k - 3, 0.1)]
        parts.append({"label": f"A_{s}", "animal": "A", "n": 1000, "UbL": (Ub * np.sqrt(w)).astype(np.float32)})
    out = fe.build_basis(parts)["A"]
    W = out["P"].astype(float)
    np.testing.assert_allclose(W.T @ W, np.eye(W.shape[1]), atol=1e-4)
    assert out["evr"][-1] >= fe.BASIS_VAR and W.shape[1] <= fe.BASIS_KMAX
    resid = common - W @ (W.T @ common)                    # the shared patterns lie in the basis
    assert np.linalg.norm(resid) / np.linalg.norm(common) < 0.05


def test_running_spline_basis_continuous_flat_above_cap_and_curve_matches():
    """Spline running basis: blocks of lag columns, identical to the capped column below the first knot, flat above the
    cap; `running_curve` = the basis times the betas (summed over lags); linear mode = the capped columns."""
    x = np.column_stack([np.linspace(-2, 30, 200)] * 2)                  # two "lag" columns
    knots, cap = [3.0, 6.0, 9.0], 15.0
    B = fe.running_basis(x, cap, knots, "spline")
    assert B.shape == (200, 2 * 4)
    assert np.array_equal(B[:, :2], np.clip(x, -cap, cap))
    hi = x[:, 0] >= cap
    assert np.allclose(B[hi], B[hi][0])                                   # flat beyond the cap
    assert np.array_equal(fe.running_basis(x, cap, knots, "linear"), np.clip(x, -cap, cap))
    beta = np.array([1.0, 0.5, -0.4, 0.0, -0.3, -0.2, 0.1, 0.0])          # blocks [s | k1 | k2 | k3] x 2 lags
    s = x[:, 0]
    y = fe.running_curve(beta, knots, s, cap)
    assert np.allclose(y, B @ beta)


def test_pixel_sse_from_target_moments_matches_direct_residuals():
    """Pixel R2 statistics: diag(W E W') with E from the Gram terms (`residual_moment`) = the per-pixel SSE of the
    residual reconstructed in the basis; grid kind = each block's own diagonal; cortex-mean target dropped."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("frm", Path(__file__).resolve().parents[1] / "scripts" / "frozen_r2_maps.py")
    frm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frm)
    rng = np.random.default_rng(3)
    T, p, K, npx = 400, 6, 5, 30
    Z = rng.standard_normal((T, p))
    Yt = np.column_stack([rng.standard_normal((T, K)), rng.standard_normal(T)])      # K basis targets + cortex
    Wf = rng.standard_normal((p, K + 1))
    E = fe.residual_moment(Z.T @ Z, Z.T @ Yt, Yt.T @ Yt, Wf)
    Rz = Yt - Z @ Wf
    assert np.allclose(E, Rz.T @ Rz)
    Wb = np.linalg.qr(rng.standard_normal((npx, K)))[0]
    keep = np.r_[np.ones(K, bool), False]
    sse = frm.pixel_sse({"kind": "basis", "P": Wb}, E, keep)
    assert np.allclose(sse, ((Rz[:, :K] @ Wb.T) ** 2).sum(0))
    assert np.allclose(frm.pixel_sse({"kind": "grid", "P": None}, E, keep), (Rz[:, :K] ** 2).sum(0))


def test_running_threshold_spline_zero_below_threshold():
    """spline_thr: no response below the first knot (the running threshold), hinge blocks only, flat above the cap;
    running_curve matches basis @ beta."""
    x = np.column_stack([np.linspace(-2, 30, 300)] * 2)
    knots, cap = [3.0, 6.0, 9.0], 15.0
    B = fe.running_basis(x, cap, knots, "spline_thr")
    assert B.shape == (300, 2 * 3)
    assert np.all(B[x[:, 0] <= 3.0] == 0)
    hi = x[:, 0] >= cap
    assert np.allclose(B[hi], B[hi][0])
    beta = np.array([1.0, 0.5, -0.4, 0.0, -0.3, -0.2])
    assert np.allclose(fe.running_curve(beta, knots, x[:, 0], cap, mode="spline_thr"), B @ beta)


def test_knot0_override(monkeypatch):
    """FROZEN_RUN_KNOT0 replaces the first knot (and drops higher knots below it); unset = the v7 default 1.5 mm/s."""
    monkeypatch.delenv("FROZEN_RUN_KNOT0", raising=False)
    assert fe._knots([3.0, 4.7, 6.5, 8.7]) == [1.5, 4.7, 6.5, 8.7]
    monkeypatch.setenv("FROZEN_RUN_KNOT0", "3")
    assert fe._knots([3.0, 4.7, 6.5, 8.7]) == [3.0, 4.7, 6.5, 8.7]
    monkeypatch.setenv("FROZEN_RUN_KNOT0", "1")
    assert fe._knots([3.0, 4.7, 6.5, 8.7]) == [1.0, 4.7, 6.5, 8.7]
    monkeypatch.setenv("FROZEN_RUN_KNOT0", "5")
    assert fe._knots([3.0, 4.7, 6.5, 8.7]) == [5.0, 6.5, 8.7]


def test_swap_template_pools_other_spouts_by_count():
    """swap_template: spout q's template = the other spouts' templates weighted by their frame counts per bin; bins
    where no other spout has data -> 0; the other spouts unchanged."""
    M = np.zeros((3, 4, 1))
    M[0, :, 0], M[1, :, 0], M[2, :, 0] = 1.0, 2.0, 5.0
    N = np.array([[10, 10, 10, 10], [30, 0, 30, 0], [10, 10, 0, 0]], float)
    S = fe.swap_template(M, N, 0)
    assert np.allclose(S[0, :, 0], [(30 * 2 + 10 * 5) / 40, 5.0, 2.0, 0.0])
    assert np.array_equal(S[1:], M[1:])


def test_push_columns_backward_only_floor_and_cap():
    """push: only backward speed beyond PUSH_FLOOR counts, forward speed -> 0, capped, NaN kept."""
    x = np.array([[-5.0], [-0.2], [0.0], [3.0], [np.nan], [-1.3]])
    y = fe.push_columns(x, cap=2.3)
    assert np.allclose(y[[0, 1, 2, 3, 5], 0], [2.0, 0.0, 0.0, 0.0, 1.0]) and np.isnan(y[4, 0])
    assert np.allclose(fe._push_abs(x[:, 0]), [5.0, 1.3])


def test_v7_running_defaults(monkeypatch):
    """STATS_VERSION 7: forward-only running from 1.5 mm/s, no push term, unless overridden; the run tag records all
    three so saved statistics are never reused across running models."""
    for v in ("FROZEN_RUNNING", "FROZEN_RUN_KNOT0", "FROZEN_PUSH"):
        monkeypatch.delenv(v, raising=False)
    assert fe.STATS_VERSION == 7
    assert fe._run_mode() == "spline_thr" and not fe._push_on()
    assert fe._run_tag() == "spline_thr|k0=1.5"
    monkeypatch.setenv("FROZEN_RUNNING", "linear")
    monkeypatch.setenv("FROZEN_PUSH", "1")
    assert fe._run_tag() == "linear|k0=1.5|push"


def test_free_push_is_cross_validated_ridge_on_the_frozen_residual(monkeypatch):
    """FROZEN_PUSH=free: base = frozen non-push prediction; push part = per-fold ridge of the residual on the push
    columns, each block predicted from the other blocks only; no push columns -> zeros."""
    rng = np.random.default_rng(3)
    n, p, r = 200, 6, 3
    Zv = rng.normal(size=(n, p))
    qm = np.array([False, False, False, False, True, True])
    beta = rng.normal(size=(p, r))
    R = Zv @ beta + rng.normal(size=(n, r))
    base, pf = fe.free_push(Zv, R, beta, qm, alpha=2.0, folds=4)
    assert np.allclose(base, Zv[:, ~qm] @ beta[~qm])
    res, Zp = R - base, Zv[:, qm]
    for a, b in ((0, 50), (50, 100), (100, 150), (150, 200)):
        tr = np.r_[0:a, b:n]
        coef = np.linalg.solve(Zp[tr].T @ Zp[tr] + 2.0 * np.eye(2), Zp[tr].T @ res[tr])
        assert np.allclose(pf[a:b], Zp[a:b] @ coef)
    b0, p0 = fe.free_push(Zv, R, beta, np.zeros(p, bool), alpha=2.0)
    assert np.allclose(b0, Zv @ beta) and not p0.any()
    monkeypatch.setenv("FROZEN_PUSH", "free")
    assert fe._push_on() and fe._push_free()
    monkeypatch.setenv("FROZEN_PUSH", "1")
    assert fe._push_on() and not fe._push_free()


def test_nose_is_other():
    """Grouping v2 (Priya 10-23): both parts of nose x and nose y are uninstructed "other", not lick direction."""
    assert fe.FGROUP["nose_dir:spl"] == fe.FGROUP["nose_dir:dev"] == fe.FGROUP["nose_mag"] == "other"
    assert fe.FGROUP["tongue_dir:dev"] == "exec_within" and fe.FGROUP["tongue_dir:spl"] == "exec_locked"
    assert fe.GROUPING_VERSION >= 2


def test_contact_own_category():
    """Grouping v3: contact is in neither task nor movement / nontask; the shared-variance drop variant exists."""
    assert "contact" not in fe.SUPER["nontask"] and "contact" not in fe.SUPER["task"]
    assert "contact" not in fe.SUPER["movement"]
    assert fe._DROP["no_contact_exec_within"] == ("contact", "exec_within")
    assert "no_contact_exec_within" in fe.VARIANTS and fe.GROUPING_VERSION == 3


def test_tongue_3d_traces_fill_rules():
    """3-D tongue on cam4 pose rows: triangulated -> mm; in the mouth (cam4 0, no 3-D) -> 0; out but untriangulated -> NaN."""
    import numpy as np
    import pandas as pd
    from wfield_local import combined_encoding as ce
    ix = pd.DataFrame({"trial_id": [1, 1, 1], "t_ms": [0.0, 4.0, 8.0]})
    F = pd.DataFrame({"trial_id": [1], "t_ms": [4.0], "tongue_ml": [1.0], "tongue_ap": [2.0], "tongue_dv": [-2.0]})
    x, y, prot = ce.tongue_3d_traces(ix, np.array([0.0, 5.0, 7.0]), F)
    assert x[0] == 0 and y[0] == 0 and prot[0] == 0
    assert x[1] == 1.0 and y[1] == 2.0 and np.isclose(prot[1], 3.0)
    assert np.isnan(x[2]) and np.isnan(prot[2])


def test_run_tag_marks_tongue3d(monkeypatch):
    monkeypatch.setenv("COMBINED_ENCODING_TONGUE", "3d")
    assert fe._run_tag().endswith("|tongue3d")
    monkeypatch.delenv("COMBINED_ENCODING_TONGUE")
    assert "tongue3d" not in fe._run_tag()
