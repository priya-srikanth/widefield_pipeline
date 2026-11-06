"""ANIMAL-AVERAGED version of the ITI lick PETH (Priya 10-28: "make 14.7 animal-averaged, as a supplement after 14.1").

Same licks, window and rows as the 14.1 figure (`lick_cued_ramp`, ITI bout starts with >= 2 s without licking, re-sampled
-2..+3 s, baseline -1.5..-1.0 s) but averaged over ANIMALS: each animal's session PETHs are averaged first, then mean +-
SEM over the four animals (so an animal with many chronic sessions does not weigh more), on the wider ROI set of the old
14.7 (`iti_licks.RAMP_ROIS`). Rows: raw; / the animal's pre response AUC (0..0.5 s); difference from the animal's own pre
curve / its pre peak. Per-bin tests as 14.1 (paired nested bootstrap animals > sessions, BH-FDR across bins).
Output figures/lick_iti_ramp_animals.png; figdata/lick_iti_ramp_animals.pkl (--plot-only redraws from it).
Usage: python scripts/lick_iti_ramp_animals.py [--plot-only]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402
import lick_cued_ramp as cr  # noqa: E402
from figdata import load_fd, plot_only, save_fd  # noqa: E402

ROIS = list(il.RAMP_ROIS)


def compute() -> dict:
    D = cr._sessions()
    PE = cr._pe(D, "iti", cr.BASE_L, ROIS)
    pc = cr._pre_curve(PE)
    pk = {k: np.nanmax(v[(cr.LAGS_W >= 0) & (cr.LAGS_W <= 1.5)]) for k, v in pc.items()}
    au = {k: cr._auc(v, cr.LAGS_W, 0.0, 0.5) for k, v in pc.items()}
    panels = {}
    for j, roi in enumerate(ROIS):
        for i in range(3):
            long, curves = [], []
            for ep in il.EP4:
                t = PE[(PE.roi == roi) & (PE.epoch == ep)]
                if t.empty:
                    continue
                M = np.stack(t.peth.to_numpy())
                if i == 1:
                    M = M / np.array([au.get((roi, a), np.nan) for a in t.animal])[:, None]
                elif i == 2:
                    M = np.stack([(m_ - pc.get((roi, a), np.nan)) / pk.get((roi, a), np.nan) for m_, a in zip(M, t.animal)])
                A = np.stack([np.nanmean(M[(t.animal == a).to_numpy()], 0) for a in sorted(t.animal.unique())])
                se = np.nanstd(A, 0, ddof=1) / np.sqrt(len(A)) if len(A) > 1 else np.full(A.shape[1], np.nan)
                curves.append((ep, np.nanmean(A, 0), se, len(A)))
                long += [pd.DataFrame({"label": x, "animal": a, "epoch": ep, "lag": cr.LAGS_W, "v": m_})
                         for x, a, m_ in zip(t.label, t.animal, M)]
            S = il.sig_compute(pd.concat(long, ignore_index=True).dropna(), value="v", tag={"roi": roi, "row": i}) \
                if (long and i < 2) else None
            panels[(i, j)] = {"roi": roi, "curves": curves, "sig": S}
        panels[(2, j)]["sig"] = panels[(1, j)]["sig"]
    return {"panels": panels}


def plot(F: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local.orofacial_cohort import EPOCH_LINE as colors
    fig, axs = plt.subplots(3, len(ROIS), figsize=(3 * len(ROIS), 10.5), sharex=True, squeeze=False)
    lab = {0: "dF/F x1e3", 1: "/ animal's pre response AUC", 2: "minus the animal's PRE curve\n(/ its pre peak)"}
    for j, roi in enumerate(ROIS):
        for i in range(3):
            ax = axs[i, j]
            p = F["panels"][(i, j)]
            for ep, mu, se, n in p["curves"]:
                if i == 2 and ep == "pre":
                    continue
                ax.plot(cr.LAGS_W, mu, color=colors[ep], lw=2, label=f"{ep} ({n} animals)")
                ax.fill_between(cr.LAGS_W, mu - se, mu + se, color=colors[ep], alpha=0.12, lw=0)
            if p["sig"] is not None:
                il.sig_draw(ax, p["sig"], colors)
                il._headroom([ax])
            ax.axvline(0, color="0.6", lw=0.6)
            ax.axhline(0, color="0.5" if i == 2 else "0.8", lw=0.7, ls="--" if i == 2 else "-")
            ax.set_xlim(-2, 3)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
            if i == 0:
                ax.set_title(roi, fontsize=9)
            if j == 0:
                ax.set_ylabel(lab[i], fontsize=8.5)
            if i == 2:
                ax.set_xlabel("time from ITI lick onset (s)", fontsize=8)
    axs[0, 0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.suptitle("SUPPLEMENT to 14.1, ANIMAL-AVERAGED: dF/F around self-initiated ITI licks (bout start, >= 2 s without "
                 "licking), -2..+3 s, baseline -1.5..-1.0 s; each animal's sessions averaged first, line = mean of the four "
                 "animals, band = SEM over animals.\nRows: raw; / the animal's pre response AUC; difference from its own pre "
                 "curve / its pre peak. " + il.SIG_NOTE.replace("\n", " "), fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(il.OUT_W / "figures" / "lick_iti_ramp_animals.png", dpi=95)
    plt.close(fig)


def main() -> int:
    if plot_only():
        plot(load_fd("lick_iti_ramp_animals"))
        return 0
    F = compute()
    save_fd("lick_iti_ramp_animals", F)
    plot(F)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
