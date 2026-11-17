"""DECODED and ACTUAL lateral lick position per spout, per epoch (Priya 10-28: "do we have a figure of decoded position per
spout? (and actual position per spout)"). Absolute positions, no offset model, no reference.

Per lick: actual = 3-D lateral landing (`act`), decoded = frozen KEY-row lateral decoder (`dec`; z-scored Allen areas with the
ALM / tjM1 split, lags 0..0.2 s; pre sessions leave-one-session-out) -- the per-lick cache of `iti_lick_order`
(lick_order_preds_areas_splitr8). Cued licks grouped by target spout, ITI licks by the previous trial's spout.
Per session mean per spout (>= MIN licks), per animal mean of its sessions, then mean +/- SEM over animals (rule 8: the
per-animal table is printed and saved first). Decoded positions are on the decoder's scale, which is compressed relative
to landing even pre-stroke (ridge shrinkage), so compare decoded with decoded across epochs and the pre line in each panel.
Outputs <E>/iti_licks/position_per_spout_animals.csv, figures/lick_position_per_spout.png;
figdata/lick_position_per_spout.pkl (--plot-only redraws from it).
Usage: python scripts/lick_position_per_spout.py [--plot-only]
"""
from __future__ import annotations
from wfield_local.figure_layout import save_png_svg  # PNG + svg/<name>.svg twin (2026-11-06)

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
import iti_lick_order as lo  # noqa: E402
from figdata import load_fd, plot_only, save_fd  # noqa: E402

EP = ["pre", "acute", "subacute", "chronic"]
MIN = 5
SETS = {"trial": "CUED (by target spout)", "iti": "ITI (by previous trial's spout)"}


def compute() -> dict:
    P = il.subset_preds(pd.read_csv(lo.preds_path(), usecols=["set", "animal", "label", "epoch", "pos", "act", "dec"]))
    P = P[P.pos.isin(lo.POS)]
    S = P.groupby(["set", "animal", "label", "epoch", "pos"]).agg(act=("act", "mean"), dec=("dec", "mean"),
                                                                  n=("act", "size")).reset_index()
    S = S[S.n >= MIN]
    A = S.groupby(["set", "animal", "epoch", "pos"])[["act", "dec"]].mean().reset_index()
    A.to_csv(il.OUT_W / "position_per_spout_animals.csv", index=False)
    pd.set_option("display.width", 220)
    for st in SETS:
        print(f"== {st}: per animal decoded | actual (mm)")
        t = A[A.set == st].pivot_table(index=["animal", "pos"], columns="epoch", values=["dec", "act"]).round(2)
        print(t.reindex(columns=EP, level=1).to_string())
    return dict(A=A)


def plot(F: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local.orofacial_cohort import ANIMAL_MARKER, EPOCH_LINE
    col = {"pre": "k", **EPOCH_LINE}
    A = F["A"]
    POS = lo.POS
    x = np.arange(len(POS))
    fig, axs = plt.subplots(2, 4, figsize=(19, 8.6), sharey="row", squeeze=False)
    for i, st in enumerate(SETS):
        g = A[A.set == st]
        M = g.groupby(["epoch", "pos"])[["act", "dec"]].agg(["mean", "sem"])
        for j, ep in enumerate(EP):
            ax = axs[i, j]
            if ep != "pre":   # pre reference in every post panel: grey markers, no connecting line (spouts are separate)
                for v, mk, dx in (("dec", "o", -0.08), ("act", "s", 0.08)):
                    ax.plot(x + dx, M.loc["pre"].reindex(POS)[(v, "mean")], mk, color="0.75", ms=9, mfc="none",
                            mew=1.5, zorder=1)
            if ep not in M.index.get_level_values(0):
                continue
            m = M.loc[ep].reindex(POS)
            ax.errorbar(x - 0.08, m[("dec", "mean")], yerr=m[("dec", "sem")], fmt="o", color=col[ep], ms=7, lw=2,
                        capsize=2, label="DECODED", zorder=3)
            ax.errorbar(x + 0.08, m[("act", "mean")], yerr=m[("act", "sem")], fmt="s", color=col[ep], mfc="white",
                        ms=7, lw=1.4, capsize=2, label="ACTUAL landing", zorder=3)
            for a, h in g[g.epoch == ep].groupby("animal"):
                h = h.set_index("pos").reindex(POS)
                mk = ANIMAL_MARKER.get(a, "o")
                ax.plot(x - 0.2, h.dec, mk, color=col[ep], ms=3, alpha=0.4, mew=0)
                ax.plot(x + 0.2, h.act, mk, color=col[ep], ms=3, alpha=0.4, mfc="none")
            ax.axhline(0, color="0.8", lw=0.6)
            ax.set_xticks(x, [p.replace("_", "\n") for p in POS], fontsize=8)
            ax.set_title(f"{SETS[st]}: {ep.upper()}" + ("" if ep == "pre" else "\n(gray open = pre: circle decoded, square actual)"),
                         fontsize=9, color=col[ep])
            if j == 0:
                ax.set_ylabel("lateral position (mm, + = contralesional)", fontsize=9)
                ax.legend(fontsize=8, frameon=False, loc="upper left")
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
    fig.suptitle("DECODED (filled circles) and ACTUAL landing (open squares) lateral lick position PER SPOUT, per epoch; mean +/- "
                 "SEM over animals, small = animals.\nFrozen pre-stroke KEY-row lateral decoder (z-scored Allen areas with the "
                 "ALM / tjM1 split, lags 0..0.2 s; pre leave-one-session-out). Decoded values are on the decoder's (shrunken) "
                 "scale: compare with the gray pre line. PROVISIONAL", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_png_svg(fig, il.OUT_W / "figures" / "lick_position_per_spout.png", dpi=100)
    plt.close(fig)


def plot_by_epoch(F: dict) -> None:
    """Second layout (Priya 10-28): epochs together in one panel; DECODED in one column, ACTUAL in the other."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local.orofacial_cohort import EPOCH_LINE
    col = {"pre": "k", **EPOCH_LINE}
    A = F["A"]
    POS = lo.POS
    x = np.arange(len(POS))
    fig, axs = plt.subplots(2, 2, figsize=(15, 9), sharey="row", squeeze=False)
    for i, st in enumerate(SETS):
        g = A[A.set == st]
        M = g.groupby(["epoch", "pos"])[["act", "dec"]].agg(["mean", "sem"])
        for j, (v, lab, mk) in enumerate((("dec", "DECODED", "o"), ("act", "ACTUAL landing", "s"))):
            ax = axs[i, j]
            for k, ep in enumerate(EP):
                if ep not in M.index.get_level_values(0):
                    continue
                m = M.loc[ep].reindex(POS)
                ax.errorbar(x + (k - 1.5) * 0.14, m[(v, "mean")], yerr=m[(v, "sem")], fmt=mk, color=col[ep], ms=7,
                            capsize=2, lw=1.4, label=ep, mfc=col[ep] if v == "dec" else "white", mew=1.5)
            ax.axhline(0, color="0.8", lw=0.6)
            for b in x[:-1] + 0.5:                         # spouts are separate: thin dividers, no connecting lines
                ax.axvline(b, color="0.93", lw=0.8, zorder=0)
            ax.set_xticks(x, [p.replace("_", "\n") for p in POS], fontsize=8.5)
            ax.set_title(f"{SETS[st]}: {lab} lateral position", fontsize=10)
            if j == 0:
                ax.set_ylabel("lateral position (mm, + = contralesional)", fontsize=9)
            if i == 0 and j == 0:
                ax.legend(fontsize=8, frameon=False, loc="upper left", ncol=4)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
    fig.suptitle("Lateral lick position PER SPOUT, epochs side by side: DECODED (left; frozen pre-stroke KEY-row decoder, its "
                 "shrunken scale) and ACTUAL landing (right); mean +/- SEM over animals. PROVISIONAL", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_png_svg(fig, il.OUT_W / "figures" / "lick_position_per_spout_byepoch.png", dpi=100)
    plt.close(fig)


def trial_set_sentence() -> str:
    """THE lick set of the figures (Priya 11-09): KEY per-lick cache rows (run's ITI filter), session x spout cells with
    >= MIN licks."""
    import lick_refit_decoders as rd
    return rd.key_sets(filt=None, extra=[f"session x spout cells with >= {MIN} licks"])


def main() -> int:
    from wfield_local import trial_set_text as tst
    tst.declare(text=trial_set_sentence())
    if plot_only():
        F = load_fd("lick_position_per_spout")
        plot(F)
        plot_by_epoch(F)
        return 0
    F = compute()
    save_fd("lick_position_per_spout", F)
    plot(F)
    plot_by_epoch(F)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
