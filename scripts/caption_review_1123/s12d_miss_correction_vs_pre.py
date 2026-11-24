"""S12D alternative (Priya 11-23): correction after a MISS, post-stroke vs pre-stroke, with a hit control.

Lick pairs = the step_b set of `scripts/lick_correction.py` (cached corr_pairs.csv.gz, four-camera cam3-on run):
consecutive cued licks k -> k+1 of a trial, peak-to-peak <= 350 ms, lick k with a previous lick in the trial (e_prev
exists); all spouts; August pre-stroke sessions only. miss = lick k without DAQ contact.

Per session (each >= MIN_N licks k of the class):
  gain_miss   OLS  dang = a + g (-e_k) + spout terms  over misses; g = fraction of lick k's angle error corrected by
              lick k+1 (1 = back on the target line, 0 = the next lick repeats lick k)
  gain_hit    the same over hits (control: regression to the mean / lick variability, no error feedback)
  gain_diff   gain_miss - gain_hit (same session)
  binned      sign(-e_k) dang after misses in 5-deg |e_k| bins x spout side, minus the same animal's pre-stroke misses
              in the same bin and side (pre sessions: reference without the session); miss-weighted mean over cells
  ac1_hit     lag-1 correlation of the angle error within trials, e_k vs e_k+1, over hit-k pairs (1 = drift / repeat,
              0 = independent noise around the target)
Robustness: the miss measures on misses with reach >= REACH_CUT, and on the LATERAL error in mm (lick k's tip
lateral position at peak minus the animal's pre-stroke contact-lick mean at that spout, + = contralesional; lick k+1's
lateral position recovered from its angle and reach: side = sgn sin(angle) reach).
Descriptives: pre-stroke angle and lateral error of misses and hits by spout and by reach bin; reach of misses by epoch.
Stats: change from each animal's own pre-stroke sessions, `analysis_kit.boot_delta` (4,000 draws, seed 0); star =
95 % interval excludes 0; per-animal changes written (rule 8).

    python s12d_miss_correction_vs_pre.py <lick_correction dir> <figdir (vls_style)> <out_dir> [reach_cut]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

EPS = ["pre", "acute", "subacute", "chronic"]
POS = ["close_ipsi", "close_mid", "close_contra", "far_ipsi", "far_mid", "far_contra"]
SIDE = {p: p.split("_")[1] for p in POS}
MIN_N, BIN, MIN_CELL, MIN_REF = 10, 5.0, 2, 5
RBINS = [0, 2, 3, 4, 99]
RLAB = ["<2", "2-3", "3-4", ">=4"]


def gain(g: pd.DataFrame, e: str = "e", d: str = "dang") -> float:
    if len(g) < MIN_N:
        return np.nan
    ne = -g[e].to_numpy(float)
    y = g[d].to_numpy(float)
    lv = [p for p in POS if (g.pos == p).any()]
    F = np.column_stack([(g.pos == p).astype(float) for p in lv[1:]]) if len(lv) > 1 else np.zeros((len(g), 0))
    b, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(g)), ne, F]), y, rcond=None)
    return float(b[1])


def binned(g: pd.DataFrame, ref: pd.DataFrame, tw: str = "toward", bcol: str = "bin") -> float:
    gm = g.groupby(["side", bcol])[tw].agg(["mean", "size"])
    rm = ref.groupby(["side", bcol])[tw].agg(["mean", "size"])
    j = gm.join(rm, rsuffix="_ref", how="inner")
    j = j[(j["size"] >= MIN_CELL) & (j["size_ref"] >= MIN_REF)]
    return float(np.average(j["mean"] - j["mean_ref"], weights=j["size"])) if len(j) else np.nan


def ac1(g: pd.DataFrame) -> float:
    return float(np.corrcoef(g.e, g.e1)[0, 1]) if len(g) >= MIN_N else np.nan


def main(lc: str, figdir: str, out: str, reach_cut: str = "3.5") -> None:
    rc = float(reach_cut)
    lc, out = Path(lc), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from wfield_local import analysis_kit as ak
    P = pd.read_csv(lc / "corr_pairs.csv.gz")
    P = P[(P.epoch != "pre") | P.aug_pre.astype(bool)]
    P = P[np.isfinite(P.e_prev) & np.isfinite(P.e)].copy()
    raw = np.degrees(np.arcsin(np.clip(P.side_ml / P.reach_mm, -1, 1)))
    sgn = float(np.sign(np.nanmedian(np.sign(P.angle) * np.sign(raw))))
    P["side"] = P.pos.map(SIDE)
    P["bin"] = np.floor(P.e.abs() / BIN).astype(int)
    P["lat"] = sgn * P.side_ml                                      # + = contralesional, mm
    P["lat1"] = np.sin(np.radians(P.angle1)) * P.reach1             # angle1 is already signed (+ = contra)
    TL = P[P.aug_pre.astype(bool) & (P.miss == 0)].groupby(["animal", "pos"]).lat.mean().rename("tgt_lat")
    P = P.join(TL, on=["animal", "pos"])
    P["e_lat"] = P.lat - P.tgt_lat
    P["dlat"] = P.lat1 - P.lat
    P["toward_lat"] = np.sign(-P.e_lat) * P.dlat
    P["bin_lat"] = np.floor(P.e_lat.abs() / 0.25).astype(int)
    P["rbin"] = pd.cut(P.reach_mm, RBINS, labels=RLAB, right=False)
    # ---------------- descriptives (a, b, d)
    pre = P[P.epoch == "pre"].assign(outcome=np.where(P[P.epoch == "pre"].miss == 1, "miss", "hit"))
    da = pre.groupby(["pos", "outcome"]).agg(n=("e", "size"), angle_err_med=("e", "median"),
                                           lat_err_med_mm=("e_lat", "median"), reach_med=("reach_mm", "median"))
    db = pre.groupby(["rbin", "outcome"], observed=True).agg(n=("e", "size"), angle_err_med=("e", "median"),
                                                          lat_err_med_mm=("e_lat", "median"))
    dd = P.assign(outcome=np.where(P.miss == 1, "miss", "hit")).groupby(["epoch", "outcome"]).agg(
        n=("e", "size"), reach_med=("reach_mm", "median"), angle_err_med=("e", "median"),
        lat_err_med_mm=("e_lat", "median"), share_reach_ge_cut=("reach_mm", lambda x: float((x >= rc).mean())))
    pd.set_option("display.width", 250)
    for nm, D in (("pre_by_spout", da), ("pre_by_reach", db), ("by_epoch", dd)):
        D.to_csv(out / f"s12d_alt_desc_{nm}.csv", float_format="%.4g")
        print(f"\n== {nm}\n", D.round(2).to_string())
    # ---------------- per-session measures
    M = P[P.miss == 1]
    H = P[P.miss == 0]
    Mr = M[M.reach_mm >= rc]
    rows = []
    for lab in sorted(P.label.unique()):
        g, h, gr = M[M.label == lab], H[H.label == lab], Mr[Mr.label == lab]
        any_ = P[P.label == lab].iloc[0]
        an = any_.animal
        refm = M[(M.animal == an) & (M.epoch == "pre") & (M.label != lab)]
        refr = Mr[(Mr.animal == an) & (Mr.epoch == "pre") & (Mr.label != lab)]
        r = {"label": lab, "animal": an, "epoch": any_.epoch, "n_miss": len(g), "n_hit": len(h),
             "n_miss_reach": len(gr), "gain_miss": gain(g), "gain_hit": gain(h),
             "binned": binned(g, refm) if len(g) >= MIN_N else np.nan, "ac1_hit": ac1(h),
             "gain_miss_reach": gain(gr), "binned_reach": binned(gr, refr) if len(gr) >= MIN_N else np.nan,
             "gain_miss_lat": gain(g, "e_lat", "dlat"), "gain_hit_lat": gain(h, "e_lat", "dlat"),
             "binned_lat": binned(g, refm, "toward_lat", "bin_lat") if len(g) >= MIN_N else np.nan,
             "reach_miss_med": g.reach_mm.median()}
        r["gain_diff"] = r["gain_miss"] - r["gain_hit"]
        r["gain_diff_lat"] = r["gain_miss_lat"] - r["gain_hit_lat"]
        rows.append(r)
    S = pd.DataFrame(rows)
    S.to_csv(out / "s12d_alt_sessions.csv", index=False, float_format="%.4g")
    METS = ["gain_miss", "gain_hit", "gain_diff", "binned", "ac1_hit", "gain_miss_reach", "binned_reach",
            "gain_miss_lat", "gain_hit_lat", "gain_diff_lat", "binned_lat", "reach_miss_med"]
    st = []
    for m in METS:
        d = S[np.isfinite(S[m])]
        pre_d = {a: g[m].to_numpy() for a, g in d[d.epoch == "pre"].groupby("animal")}
        pv = float(np.mean([v.mean() for v in pre_d.values()])) if pre_d else np.nan
        ncol = "n_miss_reach" if m.endswith("_reach") else ("n_hit" if "hit" in m else "n_miss")
        st.append({"metric": m, "epoch": "pre", "n_sessions": int((d.epoch == "pre").sum()),
                   "n_licks": int(d[d.epoch == "pre"][ncol].sum()), "pre_value": pv})
        for ep in EPS[1:]:
            q = d[d.epoch == ep]
            post = {a: g[m].to_numpy() for a, g in q.groupby("animal")}
            iv = ak.boot_delta(post, pre_d, np.random.default_rng(0), 4000)
            st.append({"metric": m, "epoch": ep, "n_sessions": len(q), "n_licks": int(q[ncol].sum()),
                       "pre_value": pv, "est": iv.point if iv else np.nan, "lo": iv.lo if iv else np.nan,
                       "hi": iv.hi if iv else np.nan, "star": bool(iv and (iv.lo > 0 or iv.hi < 0)),
                       **{f"chg_{a}": (post[a].mean() - pre_d[a].mean()) if a in post and a in pre_d else np.nan
                          for a in sorted(pre_d)}})
    ST = pd.DataFrame(st)
    ST.to_csv(out / "s12d_alt_stats.csv", index=False, float_format="%.4g")
    print("\n", ST.round(3).to_string(index=False))
    # ---------------- figure: main = miss gain change (hit gain muted alongside); supplementary = binned (deg)
    sys.path.insert(0, figdir)
    import matplotlib.pyplot as plt
    import vls_style as VS
    VS.apply()
    plt.rcParams.update({'font.size': 6.8, 'axes.labelsize': 6.8, 'axes.titlesize': 6.8, 'xtick.labelsize': 6.5,
                         'ytick.labelsize': 6.5, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6,
                         'ytick.major.width': 0.6, 'xtick.major.size': 2, 'ytick.major.size': 2})
    C = VS.EP
    rng = np.random.RandomState(3)

    def chg(m):
        d = S[np.isfinite(S[m])].copy()
        d["chg"] = d[m] - d.animal.map(d[d.epoch == "pre"].groupby("animal")[m].mean())
        return d

    def draw(ax, m, dx, muted):
        d = chg(m)
        for j, ep in enumerate(EPS):
            x = j + dx
            v = d[d.epoch == ep].chg.to_numpy()
            col = "0.6" if muted else C[ep]
            ax.plot(x + (rng.rand(len(v)) - .5) * 0.22, v, "o", ms=1.4, color=col, alpha=0.3, mec="none")
            if ep == "pre":
                ax.plot([x], [0], "o", ms=3.2, color=col, mec="k" if not muted else "0.45", mew=0.4)
                continue
            r = ST[(ST.metric == m) & (ST.epoch == ep)].iloc[0]
            ax.errorbar([x], [r.est], yerr=[[r.est - r.lo], [r.hi - r.est]], fmt="o", ms=3.2, color=col,
                        mec="k" if not muted else "0.45", mew=0.4, elinewidth=0.9, capsize=0)
            if r.star:
                ax.text(x, 1.0, "*", transform=ax.get_xaxis_transform(), ha="center", va="top",
                        color=col, fontsize=8)
        return d

    fig, axs = plt.subplots(1, 2, figsize=(3.4, 1.35), gridspec_kw={"width_ratios": [1.5, 1]})
    a0 = axs[0]
    d1 = draw(a0, "gain_miss", -0.17, False)
    d2 = draw(a0, "gain_hit", 0.17, True)
    v = np.r_[d1.chg, d2.chg]
    lo_, hi_ = np.nanpercentile(v, 1), np.nanpercentile(v, 99)
    a0.set_ylim(lo_ - 0.1 * (hi_ - lo_), hi_ + 0.3 * (hi_ - lo_))
    a0.set_ylabel("Δ correction gain\n(fraction of error)")
    a0.text(0.02, 0.02, "after a miss", color="k", fontsize=5.8, transform=a0.transAxes)
    a0.text(0.02, 0.11, "after a hit", color="0.55", fontsize=5.8, transform=a0.transAxes)
    a1 = axs[1]
    d3 = draw(a1, "binned", 0.0, False)
    lo_, hi_ = np.nanpercentile(d3.chg, 1), np.nanpercentile(d3.chg, 99)
    a1.set_ylim(lo_ - 0.1 * (hi_ - lo_), hi_ + 0.3 * (hi_ - lo_))
    a1.set_ylabel("Δ correction after a\nmiss, error-matched (deg)")
    for ax in axs:
        ax.axhline(0, color="0.6", lw=0.5, ls="--", zorder=0)
        ax.set_xticks(range(4))
        ax.set_xticklabels(["pre", "ac", "sub", "chr"])
        ax.set_xlim(-0.6, 3.6)
        ax.spines["top"].set_visible(False)
    fig.subplots_adjust(left=0.15, right=0.99, bottom=0.17, top=0.95, wspace=0.7)
    for x in ("pdf", "png", "svg"):
        fig.savefig(out.parent / f"S12D_miss_correction_vs_pre_ccB1123.{x}", dpi=300 if x == "png" else None)


if __name__ == "__main__":
    main(*sys.argv[1:5])
