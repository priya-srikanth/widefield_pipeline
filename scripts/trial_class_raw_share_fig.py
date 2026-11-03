"""RAW share of trials per class (jaw moves / no lick; licked / no spout contact), per session x spout, grouped by epoch.

Priya 11-03: "is there an epoch-grouped plot of fraction of trials with jaw movement but no tongue visible, and with lick
but not spout contact?" -- the kinematics page's Fig 6.1 (jaw_only_incomplete_figs.py) shows the CHANGE vs each animal's
own August pre; this is the same data as plain fractions, so the pre level is visible. Reads only
session_poses/cam1_licks/trial_classes/{sessions.csv, tests.csv} (nothing re-run).

Plot: one open marker per session (shape = animal, colour = epoch); black bar = mean of animal means +/- SEM over
animals. Stars = the Fig 6.1 tests (paired nested bootstrap of the change vs own pre; * CI excludes 0, ** Bonferroni
over 18), drawn in the top band.

    python scripts/trial_class_raw_share_fig.py [--out <png>]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.jaw_only_incomplete_figs import EP_COL, EPS, LAB, POS, SHOW  # noqa: E402

ROOT = Path("M:/MICROSCOPE/Priya/DeepLabCut/Widefield/session_poses/cam1_licks/trial_classes")
MARK = {"PS92": "o", "PS93": "s", "PS94": "^", "PS95": "D"}
ROWS = ("jaw_only", "incomplete")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "figures" / "trial_class_share_raw.png"))
    a = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local.sig_marks import reserve_headroom, star_top
    S = pd.read_csv(ROOT / "sessions.csv")
    R = pd.read_csv(ROOT / "tests.csv")
    rng = np.random.default_rng(1)
    fig, axs = plt.subplots(len(ROWS), len(POS), figsize=(17, 7.6), sharey="row", squeeze=False)
    rows_out = []
    for i, c in enumerate(ROWS):
        col = f"share_{c}"
        for j, p in enumerate(POS):
            ax = axs[i, j]
            for k, ep in enumerate(EPS):
                g = S[(S.position == p) & (S.epoch == ep)]
                for an, ga in g.groupby("animal"):
                    ax.scatter(k + rng.uniform(-0.22, 0.22, len(ga)), ga[col], s=22, marker=MARK[an],
                               facecolors="none", edgecolors=EP_COL[ep], lw=0.9, alpha=0.8)
                am = g.groupby("animal")[col].mean()
                if len(am):
                    m, se = am.mean(), (am.std(ddof=1) / np.sqrt(len(am)) if len(am) > 1 else 0.0)
                    ax.errorbar(k, m, yerr=se, fmt="_", ms=22, mew=2.4, color="k", capsize=4, lw=1.6, zorder=5)
                    rows_out.append(dict(cls=c, position=p, epoch=ep, mean_of_animals=m, sem=se, n_animals=len(am),
                                         n_sessions=len(g), n_trials=int(g[f"n_{c}"].sum()),
                                         **{f"{x}": am.get(x, np.nan) for x in MARK}))
                r = R[(R.cls == c) & (R.position == p) & (R.epoch == ep)]
                if len(r) and bool(r.iloc[0].sig):
                    star_top(ax, k, "**" if bool(r.iloc[0].sig_bonf) else "*", color=EP_COL[ep], fontsize=13)
            ax.set_xticks(range(4))
            ax.set_xticklabels(["pre", "acute", "subac", "chron"], fontsize=8.5)
            if i == 0:
                ax.set_title(LAB[p], fontsize=11)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
        axs[i, 0].set_ylabel(f"fraction of trials:\n{SHOW[c]}", fontsize=9)
    hs = [plt.Line2D([], [], marker=MARK[x], ls="", mfc="none", mec="0.3", label=x) for x in MARK]
    hs += [plt.Line2D([], [], marker="o", ls="", mfc="none", mec=EP_COL[e], label=e) for e in EPS]
    fig.legend(handles=hs, loc="lower center", ncol=8, frameon=False, fontsize=9)
    fig.suptitle("Fraction of trials with the jaw moving but no lick (cam1-corrected) and with a lick but no spout contact, "
                 "per session x spout, by epoch (pre = August); engaged, jaw-QC'd trials;\nopen markers = sessions "
                 "(shape = animal, colour = epoch); black bar = mean of animals +/- SEM; stars = test of the change vs own "
                 "pre (Fig 6.1): * CI excludes 0, ** Bonferroni over 18", fontsize=9.5)
    reserve_headroom(fig)
    fig.tight_layout(rect=(0, 0.04, 1, 0.9))
    out = Path(a.out)
    fig.savefig(out, dpi=100)
    pd.DataFrame(rows_out).to_csv(out.with_suffix(".csv"), index=False, float_format="%.4g")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
