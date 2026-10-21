"""CD figures on the AUGUST baseline with leave-one-out pre sessions (`scripts/cd_loo_sessions.py`; Priya 10-21: "please
do the CD mean and trajectories").

    python -m scripts.cd_loo_figures [--out <labcams>/grant_figures/epoch/cd_loo]

Per alignment (pre-cue / cue / lick), rows = animals, columns = spout positions, each panel the position's OWN coding
direction (diagonal; pole-normalised: 0 = pre-stroke not-P, 1 = pre-stroke lick at P):
  cd_loo_window_<align>.png   every session's window mean (`CD_WINDOWS`) against days from stroke; August pre sessions
                              scored leave-one-out (grey; band = their range, dashed = median), post coloured by epoch
  cd_loo_traj_<align>.png     the time course per epoch: mean +- SEM over SESSIONS (sessions are the unit, as
                              `cd_trajectories`), pre = held-out August pre sessions; shaded = the window-mean window
Reads <labcams>/grant_figures/epoch/data/{cd_window_projection_loo_sessions_long.csv, cd_loo_traces.npz}. Remember the
haemodynamic caveat (`cd_trajectories`): read amplitude and gross time course, not onset timing.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

POS = ("close_L", "close_center", "close_R", "far_L", "far_center", "far_R")
EP_COL = {"pre": "0.45", "acute": "#d01c8b", "subacute": "#b8860b", "chronic": "#01665e"}
ALIGN_LABEL = {"precue": "pre-cue (s from cue)", "cue": "s from cue", "lick": "s from first lick"}


def fig_window(C: pd.DataFrame, align: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    D = C[C["align"] == align]
    animals = sorted(D.animal.unique())
    fig, axes = plt.subplots(len(animals), len(POS), figsize=(2.6 * len(POS), 2.1 * len(animals)), squeeze=False,
                             sharex=True)
    for i, an in enumerate(animals):
        for j, p in enumerate(POS):
            ax = axes[i, j]
            g = D[(D.animal == an) & (D.position == p)]
            pre = g[g.epoch == "pre"]
            if len(pre):
                ax.axhspan(pre.value.min(), pre.value.max(), color="0.9", zorder=0)
                ax.axhline(pre.value.median(), color="0.4", ls="--", lw=0.8)
            for ep, gg in g.groupby("epoch"):
                ax.scatter(gg.day, gg.value, s=12, color=EP_COL.get(ep, "C0"), zorder=3, label=ep)
            ax.axhline(0, color="0.7", lw=0.5)
            ax.axhline(1, color="0.7", lw=0.5, ls=":")
            ax.axvline(0.5, color="k", lw=0.5, ls=":")
            ax.tick_params(labelsize=6)
            if i == 0:
                ax.set_title(p, fontsize=8)
            if j == 0:
                ax.set_ylabel(f"{an}\nCD projection", fontsize=7)
            if i == len(animals) - 1:
                ax.set_xlabel("days from stroke", fontsize=7)
    axes[0, 0].legend(fontsize=6, loc="best")
    meas = D.measure.iloc[0] if len(D) else align
    fig.suptitle(f"CD window mean, own direction ({meas}); August pre held out (grey band = pre range), "
                 "1 = pre-stroke lick signature", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def fig_traj(Z, C: pd.DataFrame, align: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from scripts.epoch_sessions_long import CD_WINDOWS
    if f"trace_{align}" not in Z:
        return
    t, X = Z[f"t_{align}"], Z[f"trace_{align}"].astype(float)
    lab, ep, pos = Z[f"label_{align}"].astype(str), Z[f"epoch_{align}"].astype(str), Z[f"position_{align}"].astype(str)
    an = np.array([x[:4] for x in lab])
    animals = sorted(set(an))
    _, lo, hi = CD_WINDOWS[align]
    fig, axes = plt.subplots(len(animals), len(POS), figsize=(2.6 * len(POS), 2.1 * len(animals)), squeeze=False,
                             sharex=True)
    for i, a in enumerate(animals):
        for j, p in enumerate(POS):
            ax = axes[i, j]
            ax.axvspan(lo, hi, color="0.93", zorder=0)
            for e in ("pre", "acute", "subacute", "chronic"):
                m = (an == a) & (pos == p) & (ep == e)
                if m.sum() < 2:
                    continue
                with np.errstate(all="ignore"):
                    mu = np.nanmean(X[m], 0)
                    se = np.nanstd(X[m], 0) / np.sqrt(np.isfinite(X[m]).sum(0).clip(1))
                ax.plot(t, mu, color=EP_COL[e], lw=1.1 if e != "pre" else 1.4, label=f"{e} ({m.sum()})")
                ax.fill_between(t, mu - se, mu + se, color=EP_COL[e], alpha=0.15, lw=0)
            ax.axhline(0, color="0.7", lw=0.5)
            ax.axhline(1, color="0.7", lw=0.5, ls=":")
            ax.axvline(0, color="k", lw=0.5)
            ax.tick_params(labelsize=6)
            if i == 0:
                ax.set_title(p, fontsize=8)
            if j == 0:
                ax.set_ylabel(f"{a}\nCD projection", fontsize=7)
            if i == len(animals) - 1:
                ax.set_xlabel(ALIGN_LABEL.get(align, "s"), fontsize=7)
            if j == len(POS) - 1:
                ax.legend(fontsize=5, loc="upper left", frameon=False)
    fig.suptitle(f"CD trajectory on each position's own direction, {align}-aligned; mean +- SEM over sessions; pre = "
                 "August sessions held out of the fit; shaded = window-mean window", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv=None) -> int:
    from wfield_local import cd_trajectories as cdt
    from wfield_local import figure_layout as FL
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    fig_dir = Path(cdt.default_out())
    C = pd.read_csv(FL.sidecar_for(fig_dir, "cd_window_projection_loo", "_sessions_long.csv", mkdir=False))
    Z = np.load(FL.sidecar_for(fig_dir, "cd_loo_traces", ".npz", mkdir=False))
    out = a.out or fig_dir / "cd_loo"
    out.mkdir(parents=True, exist_ok=True)
    for align in ("precue", "cue", "lick"):
        fig_window(C, align, out / f"cd_loo_window_{align}.png")
        fig_traj(Z, C, align, out / f"cd_loo_traj_{align}.png")
        print("->", out / f"cd_loo_window_{align}.png", "+ traj")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
