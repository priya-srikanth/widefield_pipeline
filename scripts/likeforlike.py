"""LIKE-FOR-LIKE reference with LOCAL-LINEAR weighting and edge extrapolation (Priya 10-27: "can we use the same linear
strategy for the edges?", "why can't we decode outside pre reach? isn't that what the local linear is for" -> "sounds good").

For one coordinate (lateral, angle, reach, ...), per lick set x animal, the reference at a lick's ACTUAL value x is what
the frozen decoder read for the animal's PRE licks that landed at x:
  local-linear  a Gaussian-weighted straight line (decoded vs actual) through the pre licks near x, read off at x. Unlike
                the weighted MEAN (Nadaraya-Watson, `lick_matched_readout --ref gauss`), it is not pulled inward at the
                edges of the pre range, where every neighbour sits on one side.
  edges         beyond the pre range [lo, hi] (0.5th / 99.5th percentile of the pre actual values) the edge line is
                carried out: ref(x) = ref(hi) + slope(hi) * (x - hi). EVERY lick gets a reference (none dropped);
                `beyond` = signed distance outside the range (0 inside), so results are reported on all licks (primary)
                and on inside-range licks (sensitivity).
  uncertainty   the reference is refit on pre SESSIONS resampled with replacement (B draws); the spread of the epoch effect
                over draws shows how much the (edge) slope estimate moves the answer.
Pre licks are scored leave-one-pre-session-out (their own session never enters their reference), as in the Gaussian
version. Width sigma: smallest held-out pre prediction error, chosen on pre-stroke data only.
The reference is evaluated on a grid (step sigma / 5) inside [lo, hi] and interpolated; it is smooth on that scale.

THE SHARED OFFSET (Priya 10-27 late: "yes replace beyond pull"): `offset(P, x, y, sigmas)` is the like-for-like same-place
offset every former 'offset beyond the pull' column now uses (lick order, per spout, first lick, overcorrection, recovery
link, 3-axis, ...): resid = decoded - reference(actual), the reference from the animal's PRE licks of the same set (pre
licks leave-one-pre-session-out), sigma chosen once per call by held-out pre error (table printed). Reference method:
env WFIELD_LFL_REF = "gauss" (DEFAULT; Priya 10-27 late, "ok gaussian": Gaussian-weighted MEAN, likeforlike method "nw";
NaN beyond the pre range and where the weight is < 3, as lick_matched_readout --ref gauss) | "loclin" (local line, edge
line carried out: every lick gets a reference).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

Q_EDGE = (0.5, 99.5)       # pre range = these percentiles of the pre actual values
SLOPE_RIDGE = 0.1          # slope shrinkage: local x-variance floor = (SLOPE_RIDGE * sigma)^2 (a lone lick -> flat line)
REF = os.environ.get("WFIELD_LFL_REF", "gauss")
if REF not in ("gauss", "loclin"):
    raise ValueError(f"WFIELD_LFL_REF={REF!r}: expected gauss or loclin")
METHOD = {"gauss": "nw", "loclin": "loclin"}[REF]
REF_LAB = {"gauss": "Gaussian-weighted pre reference", "loclin": "local-linear pre reference"}[REF]
SIGMA_MM = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4)         # widths tried, mm (lateral / sagittal / reach / x-y-z)
SIGMA_DEG = (2.0, 3.0, 5.0, 7.5, 10.0, 15.0)        # widths tried, degrees (angle / elevation)


def _ll_grid(xr: np.ndarray, yr: np.ndarray, g: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Local-linear value and slope at grid points g (weights normalised per point, so sparse places never underflow)."""
    lw = -0.5 * ((g[:, None] - xr[None, :]) / sigma) ** 2
    w = np.exp(lw - lw.max(1, keepdims=True))
    sw = w.sum(1)
    xb, yb = (w @ xr) / sw, (w @ yr) / sw
    dx = xr[None, :] - xb[:, None]
    sxx = (w * dx ** 2).sum(1) / sw + (SLOPE_RIDGE * sigma) ** 2
    sxy = (w * dx * (yr[None, :] - yb[:, None])).sum(1) / sw
    b = sxy / sxx
    return yb + b * (g - xb), b


def _nw_grid(xr: np.ndarray, yr: np.ndarray, g: np.ndarray, sigma: float) -> np.ndarray:
    """Weighted MEAN (Nadaraya-Watson) at g; NaN where the raw weight is < 3 (the `--ref gauss` rule)."""
    w = np.exp(-0.5 * ((g[:, None] - xr[None, :]) / sigma) ** 2)
    sw = w.sum(1)
    return np.where(sw >= 3, (w @ yr) / np.maximum(sw, 1e-12), np.nan)


def reference(xr: np.ndarray, yr: np.ndarray, x: np.ndarray, sigma: float, method: str = "loclin"
              ) -> tuple[np.ndarray, np.ndarray]:
    """Reference at x from pre licks (xr actual, yr decoded). Returns (ref, beyond)."""
    ok = np.isfinite(xr) & np.isfinite(yr)
    xr, yr = xr[ok], yr[ok]
    lo, hi = np.percentile(xr, Q_EDGE)
    beyond = np.where(x > hi, x - hi, np.where(x < lo, x - lo, 0.0))
    g = np.arange(lo, hi + sigma / 5, sigma / 5)
    if method == "nw":
        v = _nw_grid(xr, yr, g, sigma)
        ref = np.interp(np.clip(x, lo, hi), g, v)
        return np.where(beyond == 0, ref, np.nan), beyond
    v, b = _ll_grid(xr, yr, g, sigma)
    ref = np.interp(np.clip(x, lo, hi), g, v)
    ref = np.where(x > hi, v[-1] + b[-1] * (x - g[-1]), ref)
    ref = np.where(x < lo, v[0] + b[0] * (x - g[0]), ref)
    return ref, beyond


def ref_table(P: pd.DataFrame, x: str, y: str, sigma: float, method: str = "loclin") -> pd.DataFrame:
    """Per lick: ref, beyond (pre licks leave-one-pre-session-out). P needs set, animal, label, epoch, x, y."""
    ref, bey = np.full(len(P), np.nan), np.full(len(P), np.nan)
    for (_st, _a), g in P.groupby(["set", "animal"]):
        pre = g[g.epoch == "pre"]
        if len(pre) < 20:
            continue
        post = g[g.epoch != "pre"]
        i = P.index.get_indexer(post.index)
        ref[i], bey[i] = reference(pre[x].to_numpy(float), pre[y].to_numpy(float), post[x].to_numpy(float), sigma, method)
        for lab, h in pre.groupby("label"):
            o = pre[pre.label != lab]
            i = P.index.get_indexer(h.index)
            ref[i], bey[i] = reference(o[x].to_numpy(float), o[y].to_numpy(float), h[x].to_numpy(float), sigma, method)
    return pd.DataFrame({"ref": ref, "beyond": bey}, index=P.index)


def choose(P: pd.DataFrame, x: str, y: str, sigmas, method: str = "loclin") -> tuple[float, pd.DataFrame]:
    """Held-out PRE prediction error per method x sigma, on the pre licks BOTH methods cover (inside the pre range, NW
    weight >= 3). Returns (sigma of `method` with the smallest error, table)."""
    Q = P[P.epoch == "pre"]
    R = {}
    for sg in sigmas:
        for m in ("nw", "loclin"):
            R[(m, sg)] = ref_table(Q, x, y, sg, m).ref.to_numpy()
    rows = []
    for sg in sigmas:
        com = np.isfinite(R[("nw", sg)]) & np.isfinite(R[("loclin", sg)])
        for m in ("nw", "loclin"):
            e = (Q[y].to_numpy(float) - R[(m, sg)]) ** 2
            rows.append(dict(method=m, sigma=sg, mse_common=float(np.nanmean(e[com])), mse_all=float(np.nanmean(e)),
                             coverage=float(np.isfinite(R[(m, sg)]).mean())))
    T = pd.DataFrame(rows)
    L = T[T.method == method]
    return float(L.loc[L.mse_common.idxmin(), "sigma"]), T


def offset(P: pd.DataFrame, x: str, y: str, sigmas, what: str = "", method: str | None = None
           ) -> tuple[np.ndarray, np.ndarray]:
    """THE like-for-like offset per lick (module docstring): (resid = y - reference(x), outside = signed distance beyond
    the animal's pre range, 0 inside). P needs set, animal, label, epoch, x, y (any index). Sigma by held-out pre error
    for the method (env WFIELD_LFL_REF), table printed; + = read more toward + of the coordinate than same-place pre."""
    method = method or METHOD
    Q = P.reset_index(drop=True)
    ok = np.isfinite(Q[x].to_numpy(float)) & np.isfinite(Q[y].to_numpy(float))
    Qo = Q[ok].reset_index(drop=True)
    sg, cv = choose(Qo, x, y, sigmas, method)
    print(f"[like-for-like {what or y}] reference {method} ({REF}), held-out pre error per method x sigma:\n"
          + cv.round(4).to_string(index=False) + f"\n-> sigma = {sg}", flush=True)
    R = ref_table(Qo, x, y, sg, method)
    res, out = np.full(len(Q), np.nan), np.full(len(Q), np.nan)
    res[ok] = Qo[y].to_numpy(float) - R.ref.to_numpy()
    out[ok] = R.beyond.to_numpy()
    return res, out


def _effect(P: pd.DataFrame, r: np.ndarray, sel: np.ndarray) -> dict:
    """Mean over animals of (post-epoch mean of session means - pre mean of session means), per set x epoch."""
    Q = P.assign(r=r)[sel].dropna(subset=["r"])
    S = Q.groupby(["set", "animal", "label", "epoch"]).r.mean().reset_index()
    A = S.groupby(["set", "animal", "epoch"]).r.mean().unstack("epoch")
    out = {}
    for st, g in A.groupby(level="set"):
        for ep in [c for c in g.columns if c != "pre"]:
            out[(st, ep)] = float((g[ep] - g["pre"]).mean())
    return out


def boot_reference(P: pd.DataFrame, x: str, y: str, sigma: float, B: int = 100, seed: int = 0) -> pd.DataFrame:
    """Reference uncertainty: refit the post-lick reference on pre sessions resampled with replacement (per set x animal),
    B times; per set x epoch x scope (all / inside the pre range) the 2.5 / 97.5 % of the epoch effect over draws (pre
    licks keep their leave-one-session-out reference)."""
    rng = np.random.default_rng(seed)
    base = ref_table(P, x, y, sigma)
    resid0 = P[y].to_numpy(float) - base.ref.to_numpy()
    inside = (base.beyond.to_numpy() == 0) | (P.epoch.to_numpy() == "pre")
    post_mask = P.epoch.to_numpy() != "pre"
    draws = {"all": [], "inside pre range": []}
    groups = [(g, g[g.epoch == "pre"]) for _k, g in P.groupby(["set", "animal"])]
    for _b in range(B):
        r = resid0.copy()
        for g, pre in groups:
            labs = pre.label.unique()
            if len(labs) < 2:
                continue
            pick = rng.choice(labs, len(labs), replace=True)
            o = pd.concat([pre[pre.label == lab] for lab in pick])
            post = g[g.epoch != "pre"]
            i = P.index.get_indexer(post.index)
            r[i] = post[y].to_numpy(float) - reference(o[x].to_numpy(float), o[y].to_numpy(float),
                                                       post[x].to_numpy(float), sigma)[0]
        draws["all"].append(_effect(P, r, np.ones(len(P), bool)))
        draws["inside pre range"].append(_effect(P, r, inside))
    rows = []
    for scope, D in draws.items():
        pt = _effect(P, resid0, np.ones(len(P), bool) if scope == "all" else inside)
        for k in pt:
            v = np.array([d.get(k, np.nan) for d in D])
            rows.append(dict(scope=scope, set=k[0], epoch=k[1], point=pt[k], ref_lo=float(np.nanpercentile(v, 2.5)),
                             ref_hi=float(np.nanpercentile(v, 97.5)), B=B))
    _ = post_mask
    return pd.DataFrame(rows)


def coverage(P: pd.DataFrame, beyond: np.ndarray) -> pd.DataFrame:
    """Share of licks beyond the pre range per set x epoch (and the median distance beyond, among those)."""
    Q = P.assign(b=beyond)
    return Q.groupby(["set", "epoch"]).b.agg(
        share_beyond=lambda s: float((s != 0).mean()),
        median_beyond=lambda s: float(s[s != 0].abs().median()) if (s != 0).any() else 0.0).reset_index()
