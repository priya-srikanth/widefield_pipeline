"""TREND MODEL for the frozen-encoder residual (Priya 10-24: "let's do random splits and trend model").

Question: is the chronic residual (cortex responding more than the pre-stroke mapping predicts) above the trend the
residual was ALREADY on before the stroke? August pre sessions only (June excluded from encoder analyses: kinematics
off, no FaceRhythm), each scored leave-one-session-out by the frozen run.

Per animal x region x window, far-contra spout (and the mean over the six spouts):
  pre trend      residual ~ a + b * day over the August pre sessions (OLS), with its 95 % PREDICTION interval at each
                 post day (t-quantile, parameter + residual variance). The pre span is ~8 days, so extrapolating a
                 month or two ahead is very uncertain -- the interval says so, and that is the honest answer.
  deviation      each post session's residual minus the pre-trend prediction; per epoch: mean deviation, the share of
                 sessions above the 95 % PI, and a z of the epoch mean against the mean prediction SE.
  flat baseline  the same against the pre MEAN (no slope): how the answer depends on assuming a trend at all.
  chronic slope  residual ~ day within the chronic sessions (behaviour stable): does it keep rising after recovery?
The residual per session = evoked (lick: 0 .. 1 s vs -0.3 .. -0.1 s around the first lick; cue: 0.5 .. 1.5 s vs
-0.5 .. 0 s) of data minus full, from the saved PETHs (`<variant>/sessions/<label>.npz`), x1e3.

Usage: python scripts/frozen_residual_trend.py [--tag gate-attempted_v7g3_t3d] [--variant joint]
Output: <E>/frozen_encoder_<tag>/residual_trend.csv, residual_trend_sessions.csv, figures/residual_trend_<window>.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REGIONS = ("SSp-m_left", "SSp-m_right", "cortex")
WINDOWS = {"lick": ((0.0, 1.0), (-0.3, -0.1)), "cue": ((0.5, 1.5), (-0.5, 0.0))}


def session_table(run: Path, variant: str) -> pd.DataFrame:
    from wfield_local import epochs
    rows = []
    for js in sorted((run / variant / "sessions").glob("*.json")):
        m = json.loads(js.read_text(encoding="utf-8"))
        if m["label"][-4:] < "0801":
            continue
        z = np.load(js.with_suffix(".npz"))
        off = z["peth_off_s"]
        parts = m["peth_parts"]
        for w, ((r0, r1), (b0, b1)) in WINDOWS.items():
            P = z["peth_lick" if w == "lick" else "peth"]
            rs, bs = (off >= r0) & (off <= r1), (off >= b0) & (off <= b1)
            ev = np.nanmean(P[..., rs], -1) - np.nanmean(P[..., bs], -1)          # (parts, spouts, areas)
            res = ev[parts.index("data")] - ev[parts.index("full")]
            for ar in REGIONS:
                if ar not in m["areas"]:
                    continue
                ai = m["areas"].index(ar)
                vals = {p: res[pi, ai] for pi, p in enumerate(m["pos_order"])}
                rows.append({"animal": m["animal"], "label": m["label"], "epoch": m["epoch"],
                             "day": epochs.days_since_stroke(m["label"]), "window": w, "region": ar,
                             "far_contra": vals.get("far_contra", np.nan) * 1e3,
                             "all_spouts": np.nanmean(list(vals.values())) * 1e3})
    return pd.DataFrame(rows)


def trend_test(pre_d, pre_y, post_d, post_y):
    """OLS pre trend -> per-post-session prediction, SE of prediction, 95 % PI, deviation."""
    n = len(pre_d)
    X = np.column_stack([np.ones(n), pre_d])
    beta, *_ = np.linalg.lstsq(X, pre_y, rcond=None)
    res = pre_y - X @ beta
    s2 = (res ** 2).sum() / max(n - 2, 1)
    XtXi = np.linalg.inv(X.T @ X)
    Xp = np.column_stack([np.ones(len(post_d)), post_d])
    pred = Xp @ beta
    se_mean = np.sqrt(np.einsum("ij,jk,ik->i", Xp, XtXi, Xp) * s2)
    se_pred = np.sqrt(se_mean ** 2 + s2)
    t = stats.t.ppf(0.975, max(n - 2, 1))
    return beta, pred, se_mean, se_pred, t, np.sqrt(s2)


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    from wfield_local.writeguard import assert_writable
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="gate-attempted_v7g3_t3d")
    ap.add_argument("--variant", default="joint")
    a = ap.parse_args(argv)
    run = oo.root(PathResolver()) / oc.RUN / "imaging_encoding" / f"frozen_encoder_{a.tag}"
    assert_writable(run)
    T = session_table(run, a.variant)
    T.to_csv(run / "residual_trend_sessions.csv", index=False)
    rows = []
    for (an, w, ar), g in T.groupby(["animal", "window", "region"]):
        pre, post = g[g.epoch == "pre"], g[g.epoch != "pre"]
        for col in ("far_contra", "all_spouts"):
            pd_, py = pre.day.to_numpy(float), pre[col].to_numpy(float)
            ok = np.isfinite(py)
            if ok.sum() < 4:
                continue
            beta, pred, se_m, se_p, t, sres = trend_test(pd_[ok], py[ok], post.day.to_numpy(float), post[col].to_numpy(float))
            dev = post[col].to_numpy(float) - pred
            flat_dev = post[col].to_numpy(float) - py[ok].mean()
            flat_se = py[ok].std(ddof=1)
            for ep in ("acute", "subacute", "chronic"):
                m = (post.epoch == ep).to_numpy()
                if not m.any():
                    continue
                r = {"animal": an, "window": w, "region": ar, "spout": col, "epoch": ep, "n_post": int(m.sum()),
                     "n_pre": int(ok.sum()), "pre_slope_per_day": beta[1], "pre_resid_sd": sres,
                     "mean_residual": float(np.nanmean(post[col].to_numpy(float)[m])),
                     "trend_pred": float(np.mean(pred[m])), "dev_vs_trend": float(np.nanmean(dev[m])),
                     "z_vs_trend": float(np.nanmean(dev[m]) / np.sqrt(np.mean(se_m[m] ** 2) + sres ** 2 / m.sum())),
                     "frac_above_PI": float(np.mean(dev[m] > t * se_p[m])),
                     "pi_halfwidth_mean": float(np.mean(t * se_p[m])),
                     "dev_vs_flat": float(np.nanmean(flat_dev[m])),
                     "z_vs_flat": float(np.nanmean(flat_dev[m]) / (flat_se * np.sqrt(1 / m.sum() + 1 / ok.sum())))}
                if ep == "chronic" and m.sum() >= 4:
                    sl = stats.linregress(post.day.to_numpy(float)[m], post[col].to_numpy(float)[m])
                    r["chronic_slope_per_day"], r["chronic_slope_p"] = sl.slope, sl.pvalue
                rows.append(r)
    R = pd.DataFrame(rows)
    R.to_csv(run / "residual_trend.csv", index=False)
    pd.set_option("display.width", 250)
    show = R[(R.spout == "far_contra") & (R.region.isin(["SSp-m_left", "SSp-m_right"])) & (R.window == "lick")]
    print(show[["animal", "region", "epoch", "n_post", "pre_slope_per_day", "mean_residual", "trend_pred",
                "dev_vs_trend", "z_vs_trend", "pi_halfwidth_mean", "dev_vs_flat", "z_vs_flat",
                "chronic_slope_per_day", "chronic_slope_p"]].round(2).to_string(index=False))
    fd = run / "figures"
    fd.mkdir(exist_ok=True)
    for w in WINDOWS:
        fig, axes = plt.subplots(2, 4, figsize=(15, 6.5), sharex=True)
        for r, ar in enumerate(("SSp-m_left", "SSp-m_right")):
            for k, an in enumerate(sorted(T.animal.unique())):
                ax = axes[r, k]
                g = T[(T.animal == an) & (T.window == w) & (T.region == ar)].sort_values("day")
                pre = g[g.epoch == "pre"]
                ok = np.isfinite(pre.far_contra)
                dd = np.linspace(g.day.min(), g.day.max(), 200)
                beta, pred, se_m, se_p, t, _s = trend_test(pre.day[ok].to_numpy(float), pre.far_contra[ok].to_numpy(float),
                                                           dd, dd)
                ax.fill_between(dd, pred - t * se_p, pred + t * se_p, color="0.85", label="pre trend 95% PI")
                ax.plot(dd, pred, color="0.4", lw=1, label="pre trend")
                for ep, col in (("pre", "0.2"), ("acute", "#d95f02"), ("subacute", "#e6ab02"), ("chronic", "#1b9e77")):
                    x = g[g.epoch == ep]
                    ax.scatter(x.day, x.far_contra, s=16, color=col, label=ep, zorder=3)
                ax.axvline(0, color="r", lw=0.8, ls=":")
                ax.axhline(0, color="k", lw=0.5)
                lo, hi = np.nanpercentile(g.far_contra, [1, 99])
                pad = (hi - lo) * 0.6 + 1
                ax.set_ylim(lo - pad, hi + pad)
                if r == 0:
                    ax.set_title(an)
                if k == 0:
                    ax.set_ylabel(f"{ar} residual (data - full)\nfar contra, {w} window, x1e3")
                if r == 1:
                    ax.set_xlabel("days since stroke")
        axes[0, 0].legend(fontsize=7, frameon=False)
        fig.suptitle(f"Frozen-encoder residual per session vs day: straight-line trend through the August pre-stroke "
                     f"sessions (LOSO) with 95 % prediction interval, {w} window. Pre span ~8 days -> wide interval when "
                     f"extrapolated.", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(fd / f"residual_trend_{w}.png", dpi=130)
        plt.close(fig)
    print(f"-> {run / 'residual_trend.csv'}, {fd}/residual_trend_*.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
