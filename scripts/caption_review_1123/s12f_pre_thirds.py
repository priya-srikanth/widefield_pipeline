"""S12F caption review (Priya 11-23, item 9): show the pre-stroke sessions as session thirds, as the two-tone S8E does,
instead of a gray band.

Same data and per-session drawing as the S12F renderer (`review/c2_recovery_mechanics_2026-11-21/
c2_recovery_mechanics_r3.py`, section "E day course (T1)": learned_compensation T1, far-contralesional first-lick
launch error e_L by session third, post-stroke days 1-14, solid within a session, dashed to the next day, dotted over
skipped days). Changes, all taken from the two-tone S8E renderer (stroke_orofacial_pipeline
scripts/manuscript/recovery_mechanics/c1_recovery_mechanics_summary.py `daycourse`):
  * y = change from the animal's own pre-stroke mean (mean of all its pre-stroke session thirds), dashed line at 0;
  * the animal's last three pre-stroke sessions are drawn first, as thirds, in dark gray ('0.3'), joined to each other
    and to the first post-stroke session by the same dashed / dotted connectors;
  * no gray band.
x: post-stroke sessions keep the post-stroke day ticks (every other session, as before); the pre-stroke block is
labeled "pre". Page size as the live panel (5.6 x 1.35 in).

    python s12f_pre_thirds.py <manuscript_code/figures> <learned_compensation dir> <out_stem>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

N_PRE = 3


def main(figdir: str, lc: str, out_stem: str) -> None:
    sys.path.insert(0, figdir)
    import matplotlib.pyplot as plt
    import vls_style as S
    for x in ("pdf", "svg", "png"):
        if Path(f"{out_stem}.{x}").exists():
            sys.exit(f"refusing to overwrite {out_stem}.{x}")
    S.apply()
    plt.rcParams.update({'font.size': 6.8, 'axes.labelsize': 6.8, 'axes.titlesize': 6.8, 'xtick.labelsize': 6.5,
                         'ytick.labelsize': 6.5, 'legend.fontsize': 6.3, 'axes.linewidth': 0.6,
                         'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2,
                         'ytick.major.size': 2})
    C = S.EP
    T1 = pd.read_csv(Path(lc) / "t1_sessions.csv")
    an = ["PS92", "PS93", "PS94", "PS95"]
    num = dict(zip(an, ["#9", "#10", "#11", "#12"]))
    cols = ["early_eL", "mid_eL", "late_eL"]
    fig, axs = plt.subplots(1, 4, figsize=(5.6, 1.35), sharey=False)
    for ax, a in zip(axs, an):
        d = T1[T1.animal == a].sort_values(["day", "label"])
        pre_all = d[d.epoch == "pre"]
        v = pre_all[cols].to_numpy(float).ravel()
        b0 = np.nanmean(v)                                       # = the old band's centre line
        print(f"{num[a]}: pre mean {b0:.2f} deg, SD of thirds {np.nanstd(v):.2f} (n sessions {len(pre_all)})")
        pre = pre_all.tail(N_PRE)
        post = d[(d.epoch != "pre") & (d.day >= 1) & (d.day <= 14)]
        seq = pd.concat([pre, post]).reset_index(drop=True)
        prev, ticks, labels = None, [], []
        x0 = 0.0
        for _k, r in seq.iterrows():
            xs = x0 + np.array([-0.25, 0, 0.25])
            ys = r[cols].to_numpy(float) - b0
            col = "0.3" if r.epoch == "pre" else C[r.epoch]
            ax.plot(xs, ys, "-", color=col, lw=0.9, zorder=2)
            ax.plot(xs, ys, "o", ms=1.8, color=col, zorder=3)
            if prev is not None:
                ax.plot([prev[0], xs[0]], [prev[1], ys[0]], "--" if r.day - prev[2] == 1 else ":", color="0.55",
                        lw=0.6, zorder=1)
            prev = (xs[-1], ys[-1], r.day)
            if r.epoch != "pre":
                ticks.append(x0)
                labels.append(str(int(r.day)))
            x0 += 1.0
        ax.axhline(0, color="0.6", lw=0.6, ls="--", zorder=0)
        ax.set_xticks(ticks[::2])
        ax.set_xticklabels(labels[::2])
        ax.text((N_PRE - 1) / 2, -0.02, "pre", transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=6.5, color="0.3")
        ax.set_xlim(-0.6, x0 - 0.4)
        ax.set_title(num[a], pad=2)
        ax.set_xlabel("post-stroke day", labelpad=1)
    axs[0].set_ylabel("Δ first-lick launch\nerror (deg)")
    fig.subplots_adjust(left=0.1, right=0.995, bottom=0.27, top=0.86, wspace=0.32)
    for x in ("pdf", "svg", "png"):
        fig.savefig(f"{out_stem}.{x}", dpi=300 if x == "png" else None)


if __name__ == "__main__":
    main(*sys.argv[1:4])
