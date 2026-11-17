"""tjM1 ITI traces for Priya (11-17): session-mean baseline-subtracted contra (tjM1_right) / ipsi (tjM1_left) curves,
set iti, per animal x epoch (spout-pooled rows), plus docked PS94 pre per spout. Reads the neg_auc_diag extract pickles.
usage: python neg_auc_tjm1_traces.py <all.pkl> <dock.pkl> <fig_dir>"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

AN = ["PS92", "PS93", "PS94", "PS95"]
EP4 = ["pre", "acute", "subacute", "chronic"]
POS = ["close_ipsi", "close_mid", "close_contra", "far_ipsi", "far_mid", "far_contra"]
COL = {"c": "tab:red", "i": "tab:blue"}


def auc(v, L, w1):
    m = (L >= -1e-9) & (L <= w1 + 1e-9)
    return float(np.trapezoid(v[m], L[m]))


def undefined(g, L, w1):
    return int(sum((auc(r.curve_c, L, w1) <= 0) or (auc(r.curve_i, L, w1) <= 0) for r in g.itertuples()))


def deco(ax, L):
    ax.axvspan(-1.0, -0.5, color="0.88", lw=0, zorder=0)
    ax.axvspan(0.0, 0.5, color="#fff2cc", lw=0, zorder=0)        # 0..0.5 s window
    ax.axvspan(0.5, 1.0, color="#fffaeb", lw=0, zorder=0)        # 0.5..1.0 s (the 0..1 s window = both)
    ax.axhline(0, color="0.4", lw=0.8, ls="--")
    ax.axvline(0, color="0.6", lw=0.6)
    ax.set_xlim(L[0], L[-1])
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def grid(C, L, title, path, st="iti", pair="tjM1", xl="time from ITI bout-start lick (s)"):
    C = C[(C.set == st) & (C.pair == pair)]
    fig, axs = plt.subplots(4, 4, figsize=(18, 13), sharex=True, sharey=True)
    for r, an in enumerate(AN):
        for k, ep in enumerate(EP4):
            ax = axs[r, k]
            deco(ax, L)
            g = C[(C.animal == an) & (C.epoch == ep)]
            pooled, sp = g[g.spout == "pooled"], g[g.spout.isin(POS)]
            for s in ("c", "i"):
                for v in pooled[f"curve_{s}"]:
                    ax.plot(L, v, color=COL[s], lw=0.6, alpha=0.35)
                if len(pooled):
                    ax.plot(L, np.nanmean(np.stack(pooled[f"curve_{s}"].to_numpy()), 0), color=COL[s], lw=2.2)
            txt = (f"{len(pooled)} sessions\nundefined spout rows:\n0..1 s {undefined(sp, L, 1.0)}/{len(sp)}, "
                   f"0..0.5 s {undefined(sp, L, 0.5)}/{len(sp)}")
            ax.text(0.98, 0.97, txt, transform=ax.transAxes, ha="right", va="top", fontsize=7.5)
            if r == 0:
                ax.set_title(ep, fontsize=11)
            if k == 0:
                ax.set_ylabel(f"{an}\ndF/F x1e3 (baseline -1.0..-0.5 s)", fontsize=9)
            if r == 3:
                ax.set_xlabel(xl, fontsize=9)
    h = [plt.Line2D([], [], color=COL["c"], lw=2.2, label=f"contralesional {pair} (right)"),
         plt.Line2D([], [], color=COL["i"], lw=2.2, label=f"ipsilesional {pair} (left)"),
         plt.Line2D([], [], color="0.5", lw=0.6, label="thin = one session (spouts pooled); thick = animal x epoch mean")]
    fig.legend(handles=h, loc="lower center", ncol=3, fontsize=9.5, frameon=False)
    fig.suptitle(title + "\nGray = baseline -1.0..-0.5 s; yellow = 0..0.5 s AUC window, pale yellow = 0.5..1.0 s (0..1.0 s "
                 "window = both). Corner: undefined = session x spout rows with contra or ipsi AUC <= 0. Diagnostic "
                 "(neg_auc_diag 11-17)", fontsize=10)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    fig.savefig(path, dpi=100)
    plt.close(fig)


def per_spout(C, L, an, ep, title, path):
    g = C[(C.set == "iti") & (C.pair == "tjM1") & (C.animal == an) & (C.epoch == ep)]
    fig, axs = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    for ax, p in zip(axs.ravel(), POS):
        deco(ax, L)
        h = g[g.spout == p]
        nu = 0
        for r in h.itertuples():
            bad = (auc(r.curve_c, L, 1.0) <= 0) or (auc(r.curve_i, L, 1.0) <= 0)
            nu += bad
            for s in ("c", "i"):
                ax.plot(L, getattr(r, f"curve_{s}"), color=COL[s], lw=1.4 if bad else 0.6, alpha=0.9 if bad else 0.35,
                        ls="-" if not bad else (0, (4, 1.5)))
        for s in ("c", "i"):
            if len(h):
                ax.plot(L, np.nanmean(np.stack(h[f"curve_{s}"].to_numpy()), 0), color=COL[s], lw=2.4)
        ax.set_title(p.replace("_", " "), fontsize=10)
        ax.text(0.98, 0.97, f"{len(h)} sessions (licks/session {int(h.n.min()) if len(h) else 0}-"
                f"{int(h.n.max()) if len(h) else 0})\nundefined 0..1 s: {nu} (dashed)\n0..0.5 s: {undefined(h, L, 0.5)}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8)
    for ax in axs[:, 0]:
        ax.set_ylabel("dF/F x1e3 (baseline -1.0..-0.5 s)", fontsize=9)
    for ax in axs[1]:
        ax.set_xlabel("time from ITI bout-start lick (s)", fontsize=9)
    fig.suptitle(title + "\nred = contralesional tjM1, blue = ipsilesional; thin = session, dashed = session whose 0..1 s "
                 "ratio is undefined, thick = mean. Gray = baseline; yellow = 0..0.5 s, pale = 0.5..1.0 s. Diagnostic",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    fd = Path(sys.argv[3])
    fd.mkdir(parents=True, exist_ok=True)
    A = pickle.load(open(sys.argv[1], "rb"))
    D = pickle.load(open(sys.argv[2], "rb"))
    L = A["L"]
    grid(A["C"], L, "tjM1, ITI bout starts (ALL ITI, no lick for >= 2 s): session-mean contra / ipsi dF/F",
         fd / "tjm1_iti_traces_allITI.png")
    grid(D["C"], L, "tjM1, ITI bout starts (DOCKED spout, no lick for >= 2 s): session-mean contra / ipsi dF/F",
         fd / "tjm1_iti_traces_docked.png")
    per_spout(D["C"], L, "PS94", "pre", "tjM1, docked-spout ITI bout starts, PS94 PRE, per previous-trial spout",
              fd / "tjm1_iti_traces_docked_PS94_pre_perspout.png")
    for st, xl, what in (("cued", "time from first cued lick (s)", "FIRST CUED LICK (lick-aligned)"),
                         ("cue", "time from cue (s)", "CUE-ALIGNED (same trials)")):
        for pair in ("tjM1", "orofacial"):
            grid(A["C"], L, f"{pair}, {what}: session-mean contra / ipsi dF/F (rebuild, all-ITI run)",
                 fd / f"{pair.lower()}_{st}_traces.png", st=st, pair=pair, xl=xl)
    print("ok")
