"""ALL FOUR trial classes per session x spout x epoch (Priya 11-03: "tc share all should go in the deck (fraction of trials
with contact lick, no contact lick, jaw but no tongue, and no response)").

    python -m scripts.trial_class_share_all

Reads the session table of `scripts.jaw_only_incomplete_figs` (session_poses/cam1_licks/trial_classes/sessions.csv; same
classes, same engaged + jaw-QC'd denominator, so the four shares sum to 1 per session x spout) -- nothing is re-classified.
FIG 1 (trial_class_share_all.png): RAW share per class (rows: contact, incomplete, jaw-only, none), per session (open
marker, shape = animal, colour = epoch); black dot + bar = mean of animals +- SEM over animals; stars in the top band =
change vs own August pre (analysis_kit.boot_delta, 2,000 draws): * CI excludes 0, ** p < 0.05 / 18 (6 spouts x 3 epochs,
per class). Contact / incomplete / jaw-only stars are read from that script's tests.csv (identical draws to Fig 6.1);
'none' is tested here (tests_none.csv; it was not tested there).
FIG 2 (trial_class_composition.png): stacked composition, mean of animals (each animal = mean of its sessions), per spout
x epoch -- the same numbers as FIG 1's black dots.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.jaw_only_incomplete_figs import EPS, LAB, NB, POS

OUT = Path("M:/MICROSCOPE/Priya/DeepLabCut/Widefield/session_poses/cam1_licks/trial_classes")
ROWS = ("contact", "incomplete", "jaw_only", "none")
NAME = {"contact": "contact lick", "incomplete": "incomplete lick\n(licked, no contact)",
        "jaw_only": "jaw moves, no lick\n(no tongue)", "none": "no response"}
CLS_COL = {"contact": "#2b8cbe", "incomplete": "#e6550d", "jaw_only": "#fdae6b", "none": "0.8"}
EP_COL = {"pre": "0.2", "acute": "#d01c8b", "subacute": "#b8860b", "chronic": "#01665e"}  # = EPOCH_LINE
MARK = {"PS92": "o", "PS93": "s", "PS94": "^", "PS95": "D"}


def test_none(S: pd.DataFrame) -> pd.DataFrame:
    from wfield_local import analysis_kit as ak
    rng = np.random.default_rng(0)
    rows = []
    for p in POS:
        g = S[S.position == p]
        pre = {a: x.share_none.tolist() for a, x in g[g.epoch == "pre"].groupby("animal")}
        for ep in EPS[1:]:
            post = {a: x.share_none.tolist() for a, x in g[g.epoch == ep].groupby("animal")}
            iv, o = ak.boot_delta(post, pre, rng, n_boot=NB, draws=True)
            if iv is None:
                continue
            pv = ak.boot_p(o)
            rows.append(dict(cls="none", position=p, epoch=ep, change=iv.point, lo=iv.lo, hi=iv.hi, p=pv,
                             n_animals=iv.n_animals, sig=iv.lo > 0 or iv.hi < 0, sig_bonf=pv < 0.05 / 18))
    return pd.DataFrame(rows)


def animal_means(S: pd.DataFrame) -> pd.DataFrame:
    cols = [f"share_{c}" for c in ROWS]
    return S.groupby(["position", "epoch", "animal"])[cols].mean().reset_index()


def fig_share(S, R, A, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local import sig_marks
    rng = np.random.default_rng(1)
    fig, axs = plt.subplots(len(ROWS), len(POS), figsize=(17, 12), sharey="row", squeeze=False)
    for i, c in enumerate(ROWS):
        col = f"share_{c}"
        for j, p in enumerate(POS):
            ax = axs[i, j]
            for k, ep in enumerate(EPS):
                g = S[(S.position == p) & (S.epoch == ep)]
                for a, ga in g.groupby("animal"):
                    ax.scatter(k - 0.08 + rng.uniform(-0.16, 0.16, len(ga)), ga[col], s=16, marker=MARK[a],
                               facecolors="none", edgecolors=EP_COL[ep], lw=0.8, alpha=0.8)
                am = A[(A.position == p) & (A.epoch == ep)][col]
                if len(am):
                    sem = am.std(ddof=1) / np.sqrt(len(am)) if len(am) > 1 else 0
                    ax.errorbar(k + 0.3, am.mean(), yerr=sem, fmt="o", ms=6, color="k", capsize=3, lw=1.6)
                r = R[(R.cls == c) & (R.position == p) & (R.epoch == ep)]
                if len(r) and bool(r.iloc[0].sig):
                    sig_marks.star_top(ax, k + 0.1, "**" if bool(r.iloc[0].sig_bonf) else "*", color=EP_COL[ep],
                                       fontsize=13)
            ax.set_xticks(range(4))
            ax.set_xticklabels(["pre", "acute", "subac", "chron"], fontsize=8.5)
            ax.set_ylim(-0.03, 1.03)
            if i == 0:
                ax.set_title(LAB[p], fontsize=11)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
        axs[i, 0].set_ylabel(f"fraction of trials:\n{NAME[c]}", fontsize=10)
    hs = [plt.Line2D([], [], marker=MARK[a], ls="", mfc="none", mec="0.3", label=a) for a in MARK]
    hs += [plt.Line2D([], [], marker="o", ls="", mfc="none", mec=EP_COL[e], label=e) for e in EPS]
    hs += [plt.Line2D([], [], marker="o", ls="", color="k", label="mean of animals +- SEM")]
    fig.legend(handles=hs, loc="lower center", ncol=9, frameon=False, fontsize=9.5)
    sig_marks.reserve_headroom(fig)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def fig_composition(A, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    M = A.groupby(["position", "epoch"])[[f"share_{c}" for c in ROWS]].mean()
    fig, axs = plt.subplots(1, len(POS), figsize=(17, 4.6), sharey=True)
    for j, p in enumerate(POS):
        ax = axs[j]
        bot = np.zeros(len(EPS))
        for c in ROWS:
            v = np.array([M.loc[(p, ep), f"share_{c}"] if (p, ep) in M.index else np.nan for ep in EPS])
            ax.bar(range(4), v, bottom=bot, color=CLS_COL[c], width=0.75, edgecolor="white", lw=0.5,
                   label=NAME[c].replace("\n", " "))
            bot = bot + np.nan_to_num(v)
        ax.set_xticks(range(4))
        ax.set_xticklabels(["pre", "acute", "subac", "chron"], fontsize=9)
        ax.set_title(LAB[p], fontsize=11)
        ax.set_ylim(0, 1)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    axs[0].set_ylabel("fraction of trials\n(mean of animals)", fontsize=10)
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> int:
    S = pd.read_csv(OUT / "sessions.csv")
    assert np.allclose(S[[f"share_{c}" for c in ROWS]].sum(1), 1, atol=1e-3)  # shares are rounded in the CSV; counts sum exactly
    R0 = pd.read_csv(OUT / "tests.csv")
    Rn = test_none(S)
    Rn.to_csv(OUT / "tests_none.csv", index=False)
    R = pd.concat([R0, Rn], ignore_index=True)
    A = animal_means(S)
    A.to_csv(OUT / "animal_means_all_classes.csv", index=False)
    print(A[A.position == "far_R"].round(2).to_string(index=False))
    print(Rn.round(3).to_string(index=False))
    fig_share(S, R, A, OUT / "figures" / "trial_class_share_all.png")
    fig_composition(A, OUT / "figures" / "trial_class_composition.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
