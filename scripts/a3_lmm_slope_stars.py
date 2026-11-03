"""Priya's A3 rule (11-03, via a8): per contrast (one post epoch vs pre, session rows, that animal's pre + that epoch),
    y ~ post + (1 + post | animal), REML; fixed-effect t test with Satterthwaite df; Holm within a family.
Singular fit pre-specified as lme4's isSingular: a diagonal of the RELATIVE Cholesky factor (L / sigma) < 1e-4 (slope
variance 0 or |corr| = 1). Singular fits keep their estimate and are flagged. No model choice after seeing p.

Own REML (no R on this box): direct likelihood, Cholesky-parameterised G, several starts, scipy L-BFGS-B with l00, l11 >= 0.
Satterthwaite: df = 2 Var(b)^2 / (grad' A grad), grad = dVar(b)/d(g00, g01, g11, s2), A = inverse Hessian of the
negative REML log-likelihood in that same parameterisation (lmerTest's 'varpar' route), both by central differences.
Cross-checked against statsmodels MixedLM (beta, se) on the fits where it converged non-singular.
The random-INTERCEPT-only columns (ri_beta, ri_p) in the committed CSV came from a statsmodels side run (same rows)."""
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import optimize, stats

REPO = "C:/Users/SabatiniLab/Github/widefield_pipeline"
sys.path.insert(0, REPO)
OUT = sys.argv[1]
warnings.filterwarnings("ignore")
LONG = OUT + "_long.csv"

try:
    L = pd.read_csv(LONG)
except FileNotFoundError:
    from wfield_local import epoch_behaviour as eb
    from wfield_local import epoch_figures as ef
    cap = {}
    ef.bar_row = lambda values, out, *, name, marks=None, points=None, **k: cap.setdefault(name, {}).update(
        marks=marks, points=points) or name
    ef.contrast_panel = lambda rows, out, *, name, **k: cap.setdefault(name, {}).update(rows=rows) or name
    eb.fig_behaviour(OUT + "_nofig")
    sys.path.insert(0, REPO + "/scripts")
    import engagement_by_epoch as en
    en.timecourse = lambda out: "skipped"
    en.main(["--out", OUT + "_nofig"])
    recs = []
    for fam, short in (("epoch_1b_behaviour_by_position", "1b"), ("epoch_1f_engagement_by_epoch", "1f")):
        marks, pts = cap[fam]["marks"] or {}, cap[fam]["points"] or {}
        for e in marks:
            for key in marks[e]:
                for ep in ("pre", e):
                    for a, v in ((pts.get(ep) or {}).get(key) or []):
                        recs.append((short, e, key, marks[e][key], ep, a, v))
    L = pd.DataFrame(recs, columns=["family", "epoch", "cell", "star_boot_animal", "arm", "animal", "y"])
    L.to_csv(LONG, index=False)


def unpack(th):
    """varpar (g00, g01, g11, s2) -> G, s2."""
    g00, g01, g11, s2 = th
    return np.array([[g00, g01], [g01, g11]]), s2


def neg_reml(th, groups, want_beta=False):
    G, s2 = unpack(th)
    ld, XtVX, XtVy, yVy = 0.0, np.zeros((2, 2)), np.zeros(2), 0.0
    for X, y in groups:
        V = X @ G @ X.T + s2 * np.eye(len(y))
        try:
            c = np.linalg.cholesky(V)
        except np.linalg.LinAlgError:
            return np.inf if not want_beta else None
        ld += 2 * np.log(np.diag(c)).sum()
        Vi = np.linalg.inv(V)
        XtVX += X.T @ Vi @ X
        XtVy += X.T @ Vi @ y
        yVy += y @ Vi @ y
    try:
        cov_b = np.linalg.inv(XtVX)
    except np.linalg.LinAlgError:
        return np.inf if not want_beta else None
    b = cov_b @ XtVy
    quad = yVy - XtVy @ b
    val = 0.5 * (ld + np.linalg.slogdet(XtVX)[1] + quad)
    return (b, cov_b) if want_beta else val


def from_chol(p):
    l00, l10, l11, ls = p
    s2 = np.exp(ls)
    Lc = np.array([[l00, 0], [l10, l11]]) * np.sqrt(s2)        # relative Cholesky (lme4 theta) x sigma
    G = Lc @ Lc.T
    return np.array([G[0, 0], G[0, 1], G[1, 1], s2])


def fit(D):
    groups = [(np.c_[np.ones(len(g)), g.post.values], g.y.values) for _, g in D.groupby("animal")]
    s2_0 = np.log(max(D.y.var(), 1e-6))
    best = None
    for start in ([0.5, 0, 0.5, s2_0], [1, 0, 0.1, s2_0], [0.1, 0, 1, s2_0], [0.01, 0, 0.01, s2_0], [2, 0.5, 2, s2_0]):
        r = optimize.minimize(lambda p: neg_reml(from_chol(p), groups), start, method="L-BFGS-B",
                              bounds=[(0, None), (None, None), (0, None), (None, None)])
        if np.isfinite(r.fun) and (best is None or r.fun < best.fun):
            best = r
    p = best.x
    th = from_chol(p)
    b, cov_b = neg_reml(th, groups, want_beta=True)
    var_b = cov_b[1, 1]
    singular = bool(min(p[0], p[2]) < 1e-4)        # lme4 isSingular(tol = 1e-4) on the relative Cholesky diagonal

    def f(t):
        out = neg_reml(t, groups, want_beta=True)
        return np.nan if out is None else out[1][1, 1]

    h = np.maximum(np.abs(th) * 1e-4, 1e-7)
    grad = np.array([(f(th + h[i] * np.eye(4)[i]) - f(th - h[i] * np.eye(4)[i])) / (2 * h[i]) for i in range(4)])
    H = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            ei, ej = h[i] * np.eye(4)[i], h[j] * np.eye(4)[j]
            H[i, j] = (neg_reml(th + ei + ej, groups) - neg_reml(th + ei - ej, groups)
                       - neg_reml(th - ei + ej, groups) + neg_reml(th - ei - ej, groups)) / (4 * h[i] * h[j])
    try:
        A = np.linalg.inv(H)
        den = grad @ A @ grad
        df = 2 * var_b ** 2 / den if den > 0 else np.nan
    except np.linalg.LinAlgError:
        df = np.nan
    t = b[1] / np.sqrt(var_b)
    G, s2 = unpack(th)
    corr = G[0, 1] / np.sqrt(G[0, 0] * G[1, 1]) if G[0, 0] > 0 and G[1, 1] > 0 else np.nan
    return dict(beta=b[1], se=np.sqrt(var_b), t=t, df_satt=df,
                p_satt=2 * stats.t.sf(abs(t), df) if np.isfinite(df) else np.nan,
                p_df3=2 * stats.t.sf(abs(t), D.animal.nunique() - 1),
                converged=bool(best.success), singular=singular, var_int=G[0, 0], var_slope=G[1, 1],
                corr=corr, var_resid=s2, rel_chol_diag=f"{p[0]:.3g};{p[2]:.3g}", negREML=best.fun)


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for k, i in enumerate(o):
        run = max(run, (len(p) - k) * p[i])
        adj[i] = min(1.0, run)
    return adj


rows = []
for (fam, e, cell), D in L.groupby(["family", "epoch", "cell"], sort=False):
    D = D.assign(post=(D.arm == e).astype(int))
    r = fit(D)
    rows.append(dict(family=fam, epoch=e, cell=cell, star_boot_animal=D.star_boot_animal.iloc[0],
                     n_animals=D.animal.nunique(), n_sess_pre=int((D.post == 0).sum()), n_sess_post=int(D.post.sum()),
                     **r))
T = pd.DataFrame(rows)
T["star_boot_animal"] = T.star_boot_animal.fillna("")
for _fam, idx in T.groupby("family").groups.items():
    T.loc[idx, "p_holm"] = holm(T.loc[idx, "p_satt"].fillna(1).values)
    T.loc[idx, "p_bonf"] = np.minimum(1, T.loc[idx, "p_satt"].fillna(1).values * len(idx))
    T.loc[idx, "p_df3_holm"] = holm(T.loc[idx, "p_df3"].values)
T["star_lmm_df3"] = np.where(T.p_df3_holm < 0.05, "**", np.where(T.p_df3 < 0.05, "*", ""))
T["star_lmm"] = np.where(T.p_holm < 0.05, "**", np.where(T.p_satt < 0.05, "*", ""))
T.loc[T.p_satt.isna(), "star_lmm"] = "NA"
old = pd.read_csv(OUT.rsplit("/", 1)[0] + "/a3ba.csv")
old["family"] = np.where(old.family.str.contains("1b"), "1b", "1f")
T = T.merge(old[["family", "epoch", "cell", "star_old"]].rename(columns={"star_old": "star_boot_session_old"}),
            on=["family", "epoch", "cell"], how="left")
T["star_boot_session_old"] = T.star_boot_session_old.fillna("")
T.to_csv(OUT + "_slope.csv", index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
print(T.round(4).to_string())
