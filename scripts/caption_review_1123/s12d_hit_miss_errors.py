"""S12D caption review (Priya 11-23, item 2, six-position half): how can a hit and a miss share the same angle error?

Reads the cached lick-pair table of `scripts/lick_correction.py` (four-camera cam3-on run, corr_pairs.csv.gz) and the
session table behind S12D (corr_beh_sessions.csv; group all, order any = the step_b cell). Reports, for the pairs
step_b is fitted on (lick k with a previous lick in the trial, so e_prev exists; all spouts; August pre + post epochs):
  * the landing-error distributions of lick k for hits (DAQ contact) and misses: median, IQR, 5-95 %, per epoch
    (pooled over licks, and the mean over animals of the per-animal medians);
  * overlap: share of misses whose error lies inside the central 90 % of the same session's hit errors;
  * reach and lateral miss of misses vs hits (why a well-aimed lick can still miss);
  * the binned contrast (xcorr: 5-deg error bins, miss-weighted) beside step_b, tested with the same paired
    animals -> sessions bootstrap (analysis_kit.boot_delta, 4,000 draws, seed 0), as a no-regression check.
Writes CSVs and one histogram figure.

    python s12d_hit_miss_errors.py <lick_correction dir> <figdir (vls_style)> <out_dir>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

EPS = ["pre", "acute", "subacute", "chronic"]


def main(lc: str, figdir: str, out: str) -> None:
    lc, out = Path(lc), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    P = pd.read_csv(lc / "corr_pairs.csv.gz")
    P = P[(P.epoch != "pre") | P.aug_pre.astype(bool)]
    P = P[np.isfinite(P.e_prev) & np.isfinite(P.e)].copy()           # the step_b pairs (k >= 1)
    P["outcome"] = np.where(P.miss == 1, "miss", "hit")
    rows = []
    for (ep, oc), g in P.groupby(["epoch", "outcome"]):
        am = g.groupby("animal").e.median()
        rows.append({"epoch": ep, "outcome": oc, "n_licks": len(g), "n_sessions": g.label.nunique(),
                     "median": g.e.median(), "q25": g.e.quantile(.25), "q75": g.e.quantile(.75),
                     "p05": g.e.quantile(.05), "p95": g.e.quantile(.95), "abs_median": g.e.abs().median(),
                     "animal_median_mean": am.mean(), "reach_median_mm": g.reach_mm.median(),
                     "miss_mm_median": g.miss_mm.median() if "miss_mm" in g else np.nan})
    D = pd.DataFrame(rows)
    D["epoch"] = pd.Categorical(D.epoch, EPS, ordered=True)
    D = D.sort_values(["epoch", "outcome"])
    D.to_csv(out / "s12d_error_distributions.csv", index=False, float_format="%.4g")
    pd.set_option("display.width", 250)
    print(D.round(2).to_string(index=False))
    # overlap per session: misses inside the central 90 % of that session's hits
    ov = []
    for lab, g in P.groupby("label"):
        h, m = g.e[g.miss == 0], g.e[g.miss == 1]
        if len(h) >= 10 and len(m) >= 10:
            lo, hi = h.quantile(.05), h.quantile(.95)
            ov.append({"label": lab, "animal": g.animal.iloc[0], "epoch": g.epoch.iloc[0],
                       "share_miss_in_hit90": float(((m >= lo) & (m <= hi)).mean()),
                       "share_miss_within_5deg_of_target": float((m.abs() <= 5).mean()),
                       "share_hit_within_5deg_of_target": float((h.abs() <= 5).mean())})
    O = pd.DataFrame(ov)
    O.to_csv(out / "s12d_overlap_sessions.csv", index=False, float_format="%.4g")
    Oe = O.groupby(["epoch", "animal"])[["share_miss_in_hit90", "share_miss_within_5deg_of_target",
                                         "share_hit_within_5deg_of_target"]].mean().groupby("epoch").mean()
    print("\nper-session overlap, mean over animals:\n", Oe.round(3).to_string())
    Oe.to_csv(out / "s12d_overlap_by_epoch.csv", float_format="%.4g")
    # miss within +-5 deg: reach and lateral offset vs hits in the same bin
    near = P[P.e.abs() <= 5]
    rr = near.groupby(["epoch", "outcome"]).agg(n=("e", "size"), reach_mm=("reach_mm", "median"),
                                                miss_mm=("miss_mm", "median")).reset_index()
    print("\nlicks within +-5 deg of the target line: reach / 3-D miss distance (median):\n", rr.round(2).to_string())
    rr.to_csv(out / "s12d_within5deg_reach.csv", index=False, float_format="%.4g")
    # binned contrast vs step_b, same bootstrap
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from wfield_local import analysis_kit as ak
    S = pd.read_csv(lc / "corr_beh_sessions.csv")
    S = S[(S.group == "all") & (S.order == "any")]
    tr = []
    for m in ("step_b", "xcorr", "step"):
        d = S[np.isfinite(S[m])]
        pre = {a: g[m].to_numpy() for a, g in d[d.epoch == "pre"].groupby("animal")}
        for ep in EPS[1:]:
            post = {a: g[m].to_numpy() for a, g in d[d.epoch == ep].groupby("animal")}
            iv = ak.boot_delta(post, pre, np.random.default_rng(0), 4000)
            tr.append({"metric": m, "epoch": ep, "est": iv.point if iv else np.nan, "lo": iv.lo if iv else np.nan,
                       "hi": iv.hi if iv else np.nan, "n_sessions": sum(len(v) for v in post.values()),
                       "excl0": bool(iv and (iv.lo > 0 or iv.hi < 0))})
    TR = pd.DataFrame(tr)
    TR.to_csv(out / "s12d_stepb_vs_binned.csv", index=False, float_format="%.4g")
    print("\nstep_b vs binned (xcorr) vs no-trial-bias step, boot_delta vs pre:\n", TR.round(2).to_string(index=False))
    # figure: error histograms, hits vs misses, pre and acute / chronic
    sys.path.insert(0, figdir)
    import matplotlib.pyplot as plt
    import vls_style as VS
    VS.apply()
    fig, axs = plt.subplots(1, 4, figsize=(6.5, 1.45), sharex=True)
    bins = np.arange(-60, 61, 2.5)
    for ax, ep in zip(axs, EPS):
        g = P[P.epoch == ep]
        for oc, col in (("hit", "0.35"), ("miss", "#c0392b")):
            v = g.e[g.outcome == oc]
            ax.hist(v, bins=bins, density=True, histtype="step", color=col, lw=0.9,
                    label=f"{oc} (n = {len(v):,})")
        ax.axvline(0, color="0.6", lw=0.5, ls="--")
        ax.set_title(ep, fontsize=7, pad=2)
        ax.legend(fontsize=5.5, frameon=False, loc="upper left", handlelength=1.2)
        ax.set_xlabel("landing error of lick k (deg)", fontsize=6.5)
        ax.tick_params(labelsize=6)
    axs[0].set_ylabel("density", fontsize=6.5)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.27, top=0.86, wspace=0.3)
    for x in ("pdf", "png"):
        fig.savefig(out.parent / f"S12D_hit_miss_error_hist_ccB1123.{x}", dpi=300 if x == "png" else None)


if __name__ == "__main__":
    main(*sys.argv[1:4])
