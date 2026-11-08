"""RAW share of trials per class (jaw moves / no lick; licked / no spout contact), per session x spout, grouped by epoch.

Priya 11-03: "is there an epoch-grouped plot of fraction of trials with jaw movement but no tongue visible, and with lick
but not spout contact?" -- the kinematics page's Fig 6.1 (jaw_only_incomplete_figs.py) shows the CHANGE vs each animal's
own August pre; this is the same data as plain fractions, so the pre level is visible. Reads only
session_poses/cam1_licks/trial_classes/{sessions.csv, tests.csv} (nothing re-run).

Plot: one open marker per session (shape = animal, color = epoch); black bar = mean of animal means +/- SEM over
animals. Stars = the Fig 6.1 tests (paired nested bootstrap of the change vs own pre; * CI excludes 0, ** Bonferroni
over 18), drawn in the top band.

    python scripts/trial_class_raw_share_fig.py [--out <png>]
"""
from __future__ import annotations
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import jaw_only_incomplete_figs as jf  # noqa: E402
from scripts.jaw_only_incomplete_figs import EP_COL, EPS, LAB, POS, SHOW  # noqa: E402

ROOT = Path("M:/MICROSCOPE/Priya/DeepLabCut/Widefield/session_poses/cam1_licks/trial_classes")
MARK = {"PS92": "o", "PS93": "s", "PS94": "^", "PS95": "D"}
ROWS = ("jaw_only", "incomplete", "none")
NAME = {**SHOW, "none": "no response\n(no lick, no jaw movement)", "contact": "contact lick",
        "no_response": "no response\n(no lick, no jaw movement)"}
# --all-engaged (11-03): jaw_only_incomplete_figs' table since 07385b37 -- denominator = EVERY engaged trial (before it
# table keeps only jaw_pass_qc trials, i.e. trials on which the jaw MOVED, so its `none` is empty by construction)
ROWS_ALL = ("contact", "incomplete", "jaw_only", "no_response")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "figures" / "trial_class_share_raw.png"))
    ap.add_argument("--all-engaged", default=None, help="dir with the all-engaged class table (session_poses/cam1_licks/trial_classes; jaw_only_incomplete_figs.classify, 11-03)")
    ap.add_argument("--change", action="store_true", help="plot change vs each animal's own pre mean (same tests)")
    a = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local.sig_marks import reserve_headroom, star_top
    global ROWS
    if a.all_engaged:
        ROWS = ROWS_ALL
        S = pd.read_csv(Path(a.all_engaged) / "sessions.csv")
        R = pd.DataFrame(columns=["cls"])       # every class tested here, same test as the source
    else:
        S = pd.read_csv(ROOT / "sessions.csv")
        R = pd.read_csv(ROOT / "tests.csv")
    # classes the source run did not test (Priya 11-03: "no response") get the SAME test (jaw_only_incomplete_figs.tests:
    # paired nested bootstrap of the change vs own pre, seed 0), written next to the figure, not into the source folder
    miss = [c for c in ROWS if c not in set(R.cls)]
    if miss:
        keep = jf.SHOW
        jf.SHOW = {c: NAME[c] for c in miss}
        try:
            Rn = jf.tests(S)
        finally:
            jf.SHOW = keep
        Rn.to_csv(Path(a.out).with_name(Path(a.out).stem + "_tests_added.csv"), index=False, float_format="%.4g")
        R = pd.concat([R, Rn], ignore_index=True)
    if a.change:     # the pre-normalised twin (Priya 11-03: raw and pre-normalised side by side): share minus own pre mean
        for c in ROWS:
            base = S[S.epoch == "pre"].groupby(["animal", "position"])[f"share_{c}"].mean()
            S[f"share_{c}"] = S[f"share_{c}"] - base.reindex(pd.MultiIndex.from_arrays([S.animal, S.position])).to_numpy()
    rng = np.random.default_rng(1)
    fig, axs = plt.subplots(len(ROWS), len(POS), figsize=(17, 3.6 * len(ROWS)), sharey="row", squeeze=False)
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
            if a.change:
                ax.axhline(0, color="0.6", lw=0.8, zorder=0)
            ax.set_xticks(range(4))
            ax.set_xticklabels(["pre", "acute", "subac", "chron"], fontsize=8.5)
            if i == 0:
                ax.set_title(LAB[p], fontsize=11)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
        axs[i, 0].set_ylabel(f"fraction of trials{', change vs own pre' if a.change else ''}:\n{NAME[c]}", fontsize=9)
    hs = [plt.Line2D([], [], marker=MARK[x], ls="", mfc="none", mec="0.3", label=x) for x in MARK]
    hs += [plt.Line2D([], [], marker="o", ls="", mfc="none", mec=EP_COL[e], label=e) for e in EPS]
    fig.legend(handles=hs, loc="lower center", ncol=8, frameon=False, fontsize=9)
    den = ("ALL engaged trials (one class per trial, shares sum to 1)" if a.all_engaged
           else "engaged trials on which the jaw moved (jaw_pass_qc)")
    if a.all_engaged and (Path(a.all_engaged) / "denominator.txt").exists():   # a --trials run of jaw_only_incomplete_figs
        den = (Path(a.all_engaged) / "denominator.txt").read_text(encoding="utf-8").strip()
    what = ("CHANGE vs each animal's own August pre mean in the fraction of trials per class" if a.change
            else "Fraction of trials per class")
    fig.suptitle(what + ", per session x spout, by epoch (pre = August); denominator = " + den +
                 ";\nopen markers = sessions (shape = animal, color = epoch); black bar = mean of animals +/- SEM; stars = "
                 "paired nested bootstrap of the change vs own pre: * CI excludes 0, ** Bonferroni over 18 (per row)",
                 fontsize=9.5)
    reserve_headroom(fig)
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    out = Path(a.out)
    save_png_svg(fig, out, dpi=100)
    pd.DataFrame(rows_out).to_csv(out.with_suffix(".csv"), index=False, float_format="%.4g")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
