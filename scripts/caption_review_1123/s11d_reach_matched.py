"""S11D caption review (Priya 11-23, item 21): is the speed-for-reach result robust to the shape of the curve?

S11D: residual peak 3-D tongue speed = observed minus the speed expected at the lick's reach from the animal's
pre-stroke fit for the same spout side, v = a + c (1 - exp(-r / k)) (nonlinear least squares); post-stroke licks
outside the animal's pre-stroke reach range for that side are excluded; residuals averaged per session and side, then
per animal; 95 % interval from a hierarchical bootstrap over animals, then sessions; held-out pre-stroke sessions are
predicted from fits to the animal's other pre-stroke sessions (manuscript Methods "Speed adjusted for reach").

The S11D producing script (`s11_velocity_reach_residual.py`, CC-A) is not on this machine, so the saturating fit is
re-implemented here from the Methods and run beside the alternatives on the same licks with the same aggregation:
  sat     v = a + c (1 - exp(-r / k))                       (the S11D model)
  hyp     v = a + c r / (r + k)                             (alternative saturating form)
  bin     reach-matched: mean pre-stroke speed of the same animal x side in the lick's 0.25 mm reach bin
          (bins with >= MIN_BIN pre-stroke licks; a post lick in a thinner or empty bin is dropped)
  knn     reach-matched: mean speed of the K nearest pre-stroke licks in reach (same animal x side)
Pre-stroke sessions are always predicted leave-one-session-out (every model refit / re-binned without the session).
Licks: the four-camera fused post-cue table (kinematics_fused_postcue/licks.csv.gz), 3-D support (kin_ok), August
pre-stroke sessions only (config.early_pre dropped), finite reach and peak speed. Side: L = ipsilesional, center =
middle, R = contralesional (all lesions left).

Output: per model x side x epoch the mean over animals and the hierarchical bootstrap interval (4,000 draws,
seed 0), the share of post licks kept, per-animal epoch means (rule 8), session table, and a comparison figure.

    python s11d_reach_matched.py <licks.csv.gz> <figdir (vls_style)> <out_dir>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

BIN_MM, MIN_BIN, K_NN = 0.25, 10, 50
N_BOOT, SEED = 4000, 0
EPS = ["pre", "acute", "subacute", "chronic"]
SIDE = {"close_L": "ipsi", "far_L": "ipsi", "close_center": "middle", "far_center": "middle",
        "close_R": "contra", "far_R": "contra"}
MODELS = ["sat", "hyp", "bin", "knn"]


def f_sat(r, a, c, k):
    return a + c * (1 - np.exp(-r / k))


def f_hyp(r, a, c, k):
    return a + c * r / (r + k)


def _fit(f, r, v):
    p0 = (np.percentile(v, 5), np.percentile(v, 95) - np.percentile(v, 5), max(np.median(r), 0.5))
    try:
        p, _ = curve_fit(f, r, v, p0=p0, maxfev=20000, bounds=([-np.inf, -np.inf, 1e-3], [np.inf, np.inf, 1e3]))
        return lambda x, p=p: f(x, *p)
    except RuntimeError:
        return None


def predictor(model: str, r: np.ndarray, v: np.ndarray):
    """-> callable reach -> expected speed (NaN where the model has no support)."""
    lo, hi = r.min(), r.max()
    if model in ("sat", "hyp"):
        g = _fit(f_sat if model == "sat" else f_hyp, r, v)
        if g is None:
            return None
        return lambda x: np.where((x >= lo) & (x <= hi), g(x), np.nan)
    if model == "bin":
        b = np.floor(r / BIN_MM).astype(int)
        s = pd.DataFrame({"b": b, "v": v}).groupby("b").v.agg(["mean", "size"])
        s = s[s["size"] >= MIN_BIN]["mean"]

        def pb(x):
            bx = np.floor(x / BIN_MM).astype(int)
            return np.where((x >= lo) & (x <= hi), s.reindex(bx).to_numpy(float), np.nan)
        return pb
    from sklearn.neighbors import KNeighborsRegressor
    knn = KNeighborsRegressor(n_neighbors=min(K_NN, len(r))).fit(r[:, None], v)

    def pk(x):                                         # mean speed of the k nearest pre-stroke licks in reach
        out = np.full(len(x), np.nan)
        ok = (x >= lo) & (x <= hi)
        if ok.any():
            out[ok] = knn.predict(x[ok, None])
        return out
    return pk


def hier_boot(d: dict, rng) -> tuple[float, float, float]:
    """{animal: session means}: mean over animals of the animal means; animals then sessions resampled."""
    an = sorted(d)
    est = float(np.mean([np.mean(d[a]) for a in an]))
    out = np.empty(N_BOOT)
    for i in range(N_BOOT):
        aa = rng.choice(an, len(an))
        out[i] = np.mean([rng.choice(d[a], len(d[a])).mean() for a in aa])
    return est, float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main(src: str, figdir: str, out: str) -> None:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from wfield_local import config
    L = pd.read_csv(src, usecols=["position", "context", "reach_mm", "speed_peak", "animal", "date", "session",
                                  "epoch", "kin_ok", "cams"], dtype={"date": str})
    L = L[(L.context == "post_cue") & L.kin_ok.astype(bool) & np.isfinite(L.reach_mm) & np.isfinite(L.speed_peak)]
    L = L[L.cams.astype(str) != "cam1"]
    L = L[L.epoch.isin(EPS) & ~((L.epoch == "pre") & L.date.map(config.early_pre))].copy()
    L["side"] = L.position.map(SIDE)
    L = L.dropna(subset=["side"])
    print(f"licks {len(L):,}; sessions {L.session.nunique()}; pre sessions "
          f"{L[L.epoch == 'pre'].session.nunique()}")
    rows = []
    for (a, sd), g in L.groupby(["animal", "side"]):
        pre = g[g.epoch == "pre"]
        post = g[g.epoch != "pre"]
        r0, v0 = pre.reach_mm.to_numpy(), pre.speed_peak.to_numpy()
        for m in MODELS:
            pr = predictor(m, r0, v0)
            if pr is not None and len(post):
                e = pr(post.reach_mm.to_numpy())
                rows.append(post.assign(model=m, pred=e)[["animal", "side", "session", "epoch", "model", "pred",
                                                            "speed_peak"]])
            for s, gs in pre.groupby("session"):                # leave-one-session-out for every pre session
                rest = pre[pre.session != s]
                pr2 = predictor(m, rest.reach_mm.to_numpy(), rest.speed_peak.to_numpy())
                if pr2 is None:
                    continue
                rows.append(gs.assign(model=m, pred=pr2(gs.reach_mm.to_numpy()))[
                    ["animal", "side", "session", "epoch", "model", "pred", "speed_peak"]])
        print(f"  {a} {sd}: {len(pre):,} pre / {len(post):,} post licks", flush=True)
    R = pd.concat(rows, ignore_index=True)
    R["kept"] = np.isfinite(R.pred)
    R["res"] = R.speed_peak - R.pred
    keep = R.groupby(["model", "epoch"]).kept.mean().unstack()
    print("\nshare of licks with a prediction (kept):\n", keep.round(3).to_string())
    keep.to_csv(out / "s11d_share_kept.csv", float_format="%.4g")
    S = R[R.kept].groupby(["model", "animal", "side", "session", "epoch"]).agg(res=("res", "mean"),
                                                                              n=("res", "size")).reset_index()
    S.to_csv(out / "s11d_sessions.csv", index=False, float_format="%.4g")
    pa = S.groupby(["model", "side", "epoch", "animal"]).res.mean().unstack()
    pa.to_csv(out / "s11d_per_animal.csv", float_format="%.4g")
    st = []
    for (m, sd, ep), g in S.groupby(["model", "side", "epoch"]):
        d = {a: x.res.to_numpy() for a, x in g.groupby("animal")}
        est, lo, hi = hier_boot(d, np.random.default_rng(SEED))
        st.append({"model": m, "side": sd, "epoch": ep, "est": est, "lo": lo, "hi": hi,
                   "star": bool(lo > 0 or hi < 0), "n_sessions": len(g), "n_animals": len(d),
                   "animals_below0": int(sum(np.mean(v) < 0 for v in d.values()))})
    # pooled over sides (the held-out pre-stroke check quoted in Methods: 0.1 mm/s, -2.3 to 2.4)
    for (m, ep), g in S.groupby(["model", "epoch"]):
        gg = g.groupby(["animal", "session"]).res.mean().reset_index()
        d = {a: x.res.to_numpy() for a, x in gg.groupby("animal")}
        est, lo, hi = hier_boot(d, np.random.default_rng(SEED))
        st.append({"model": m, "side": "all", "epoch": ep, "est": est, "lo": lo, "hi": hi,
                   "star": bool(lo > 0 or hi < 0), "n_sessions": len(gg), "n_animals": len(d),
                   "animals_below0": int(sum(np.mean(v) < 0 for v in d.values()))})
    ST = pd.DataFrame(st)
    ST["epoch"] = pd.Categorical(ST.epoch, EPS, ordered=True)
    ST["side"] = pd.Categorical(ST.side, ["ipsi", "middle", "contra", "all"], ordered=True)
    ST["model"] = pd.Categorical(ST.model, MODELS, ordered=True)
    ST = ST.sort_values(["side", "epoch", "model"])
    ST.to_csv(out / "s11d_stats_by_model.csv", index=False, float_format="%.4g")
    pd.set_option("display.width", 250)
    print("\n", ST.round(2).to_string(index=False))
    W = ST.pivot_table(index=["side", "epoch"], columns="model", values="star", observed=True)
    print("\nstars (interval excludes 0) by model:\n", W.to_string())
    # figure: estimate +- interval per model, side x epoch
    sys.path.insert(0, figdir)
    import matplotlib.pyplot as plt
    import vls_style as VS
    VS.apply()
    col = {"sat": "k", "hyp": "0.45", "bin": "#1f78b4", "knn": "#33a02c"}
    lab = {"sat": "saturating fit (S11D)", "hyp": "hyperbolic fit", "bin": "0.25 mm reach bins", "knn": f"{K_NN} nearest in reach"}
    fig, axs = plt.subplots(1, 3, figsize=(6.5, 1.9), sharey=True)
    for ax, sd in zip(axs, ["ipsi", "middle", "contra"]):
        for j, ep in enumerate(EPS):
            for i, m in enumerate(MODELS):
                r = ST[(ST.model == m) & (ST.side == sd) & (ST.epoch == ep)]
                if r.empty:
                    continue
                r = r.iloc[0]
                x = j + (i - 1.5) * 0.17
                ax.errorbar([x], [r.est], yerr=[[r.est - r.lo], [r.hi - r.est]], fmt="o", ms=2.6, color=col[m],
                            elinewidth=0.8, capsize=0, label=lab[m] if (j == 0 and sd == "ipsi") else None)
                if r.star and ep != "pre":
                    ax.text(x, 1.0, "*", transform=ax.get_xaxis_transform(), ha="center", va="top", color=col[m],
                            fontsize=7)
        ax.axhline(0, color="0.6", lw=0.5, ls="--")
        ax.set_xticks(range(4))
        ax.set_xticklabels(["pre\n(held out)", "acute", "subacute", "chronic"], fontsize=6)
        ax.set_title({"ipsi": "ipsilesional", "middle": "middle", "contra": "contralesional"}[sd] + " spouts",
                     fontsize=7, pad=8)
        ax.spines["top"].set_visible(False)
    axs[0].set_ylabel("residual peak speed\n(mm/s)", fontsize=6.5)
    h, lb = axs[0].get_legend_handles_labels()
    fig.legend(h, lb, fontsize=5.8, frameon=False, loc="upper center", ncol=4, handlelength=1,
               bbox_to_anchor=(0.55, 1.0))
    fig.subplots_adjust(left=0.09, right=0.995, bottom=0.22, top=0.74, wspace=0.12)
    for x in ("pdf", "png"):
        fig.savefig(out.parent / f"S11D_speed_reach_shapefree_ccB1123.{x}", dpi=300 if x == "png" else None)


if __name__ == "__main__":
    main(*sys.argv[1:4])
