"""S12D, millimeter version (Priya 11-23): correction gain after a miss and after a hit, change from own pre-stroke.

Plot only. Reads the tables written by `s12d_miss_correction_vs_pre.py` (commit 12e055c3); nothing is recomputed.
  left (main)  gain_miss_lat (filled) and gain_hit_lat (open), both in the epoch color, side by side per epoch: fraction of lick k's lateral
               error (mm, vs the animal's pre-stroke contact-lick mean lateral position at that spout) corrected by
               lick k+1; session change from the animal's own pre-stroke session mean; markers and bars =
               boot_delta estimate and 95 % interval (4,000 draws, seed 0); * = interval excludes 0 (top band).
  right (check) binned_lat: lateral correction toward the target after a miss (mm), 0.25-mm |error| bins x spout side,
               minus the same animal's pre-stroke misses in the same cell; change from own pre.
Canvas: height of the live S12D panel (C2R_G_after_miss_r3_lp.pdf, 84.96 x 93.6 pt), C2R renderer style.

    python s12d_mm_render.py <data_item2b_alt dir> <figures dir with vls_style.py> <out stem>
Refuses to overwrite an existing output.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

EPS = ["pre", "acute", "subacute", "chronic"]
XL = ["pre", "ac", "sub", "chr"]
LEG = "0.2"          # neutral dark legend markers (v2: hit = open, miss = filled, both in the epoch color)
DX = 0.19


def main(data: str, figdir: str, stem: str) -> None:
    data, stem = Path(data), Path(stem)
    for x in ("pdf", "svg", "png"):
        if stem.with_suffix(f".{x}").exists():
            sys.exit(f"refusing to overwrite {stem.with_suffix('.' + x)}")
    S = pd.read_csv(data / "s12d_alt_sessions.csv")
    ST = pd.read_csv(data / "s12d_alt_stats.csv")
    ST["star"] = ST["star"].astype(str).str.lower().eq("true")
    sys.path.insert(0, figdir)
    import matplotlib.pyplot as plt
    import vls_style as VS
    VS.apply()
    plt.rcParams.update({"font.size": 6.8, "axes.labelsize": 6.8, "axes.titlesize": 6.8, "xtick.labelsize": 6.5,
                         "ytick.labelsize": 6.5, "legend.fontsize": 6.0, "axes.linewidth": 0.6,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2,
                         "ytick.major.size": 2})
    C = VS.EP
    rng = np.random.RandomState(3)

    def chg(m):
        d = S[np.isfinite(S[m])].copy()
        d["chg"] = d[m] - d.animal.map(d[d.epoch == "pre"].groupby("animal")[m].mean())
        return d

    def draw(ax, m, dx, hollow, stars):
        """Epoch colors for both classes: miss = filled circles, hit (hollow=True) = open circles (Priya 11-23 v2)."""
        d = chg(m)
        for j, ep in enumerate(EPS):
            x = j + dx
            v = d[d.epoch == ep].chg.to_numpy()
            col = C[ep]
            pt = dict(mfc="none", mec=col, mew=0.45) if hollow else dict(mfc=col, mec="none")
            mk = dict(mfc="white", mec=col, mew=0.9) if hollow else dict(mfc=col, mec="k", mew=0.4)
            ax.plot(x + (rng.rand(len(v)) - .5) * 0.13, v, "o", ms=1.8 if hollow else 1.6, alpha=0.4, zorder=1, **pt)
            if ep == "pre":
                ax.plot([x], [0], "o", ms=3.6, zorder=3, **mk)
                continue
            r = ST[(ST.metric == m) & (ST.epoch == ep)].iloc[0]
            ax.errorbar([x], [r.est], yerr=[[r.est - r.lo], [r.hi - r.est]], fmt="o", ms=3.6, color=col,
                        elinewidth=0.9, capsize=0, zorder=3, **mk)
            if r.star:
                stars.append((x, col))
        return d

    def finish(ax, vals, stars, ylab):
        lo, hi = np.nanmin(vals), np.nanmax(vals)
        rg = hi - lo
        ax.set_ylim(lo - 0.06 * rg, hi + 0.30 * rg)                 # top band reserved for stars
        for x, col in stars:
            ax.text(x, 0.99, "*", transform=ax.get_xaxis_transform(), ha="center", va="top", color=col,
                    fontsize=8, fontweight="bold")
        VS.zero_line(ax)
        ax.set_xticks(range(4))
        ax.set_xticklabels(XL)
        ax.set_xlim(-0.55, 3.55)
        ax.set_ylabel(ylab)
        ax.spines[["top", "right"]].set_visible(False)

    h_in, left_in, gap_in, right_in = 1.3, 0.5, 0.62, 0.06
    w0, w1 = 1.30, 0.80
    w_in = left_in + w0 + gap_in + w1 + right_in
    fig = plt.figure(figsize=(w_in, h_in))
    b, t = 0.2, 0.86
    a0 = fig.add_axes([left_in / w_in, b, w0 / w_in, t - b])
    a1 = fig.add_axes([(left_in + w0 + gap_in) / w_in, b, w1 / w_in, t - b])
    st0, st1 = [], []
    d1 = draw(a0, "gain_miss_lat", -DX / 1.0 * 0.95, False, st0)
    d2 = draw(a0, "gain_hit_lat", DX * 0.95, True, st0)
    finish(a0, np.r_[d1.chg, d2.chg], st0, "Δ correction gain\n(fraction of lateral error)")
    hm = plt.Line2D([], [], marker="o", ls="none", ms=3.6, mfc=LEG, mec=LEG, mew=0.4)
    hh = plt.Line2D([], [], marker="o", ls="none", ms=3.6, mfc="white", mec=LEG, mew=0.9)
    a0.legend([hm, hh], ["after a miss", "after a hit"], loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2,
              frameon=False, borderaxespad=0.1, handlelength=1.0, handletextpad=0.3, columnspacing=1.2)
    d3 = draw(a1, "binned_lat", 0.0, False, st1)
    finish(a1, d3.chg.to_numpy(), st1, "Δ correction after a miss,\nerror-matched (mm)")
    for x in ("pdf", "svg", "png"):
        fig.savefig(stem.with_suffix(f".{x}"), dpi=300 if x == "png" else None)
    print(f"canvas {w_in * 72:.2f} x {h_in * 72:.2f} pt -> {stem}")
    for m in ("gain_miss_lat", "gain_hit_lat", "gain_diff_lat", "binned_lat"):
        print(ST[ST.metric == m][["metric", "epoch", "n_sessions", "n_licks", "pre_value", "est", "lo", "hi",
                                  "star"]].round(3).to_string(index=False, header=m == "gain_miss_lat"))


if __name__ == "__main__":
    main(*sys.argv[1:4])
