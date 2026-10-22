"""Musall et al. 2019-style figures from the FROZEN pre-stroke encoder (Priya 10-22: "did we regenerate the musall
style PETH vs trial-by-trial prediction, task-aligned and task-independent, and task-aligned vs unique contribution?").

Reads `<E>/frozen_encoder_<tag>/frozen_musall_r2_by_animal.csv` (`wfield_local.frozen_encoder.summarise_musall`):
held-out R^2 per variant x animal x epoch x align (cue / first lick) x measure x area, where measure = FRAME (every
frame, condition-locked + single-trial), TRIAL (each spout's mean removed: single-trial only) and AVG (per-spout mean
responses only). The frozen variants are separate fits, so Musall's quantities are differences of variants:

  ALONE        task = no_movement; movement = movement_only; all non-task = no_task
  UNIQUE       task = joint - no_task; movement = task_movement - no_movement; direction = task_movement_dir -
               task_movement; contact = joint - task_movement_dir   (sequential for movement / direction / contact:
               each beyond task and the groups added before it)
  TASK-INDEP.  non-task = joint - no_movement; movement = task_movement - no_movement
  TASK-ALIGNED non-task = no_task - task-independent; movement = movement_only - task-independent

Figures (per area, one per align): (1) alone vs unique (Musall Fig. 4a), (2) task-aligned vs task-independent
(Fig. 5), (3) trial-averaged vs single-trial vs frame R^2 of the full model and each alone model (PETH vs
trial-by-trial). Bars = mean over animals, dots = animals; epochs as columns. Held out throughout (pre sessions
leave-one-session-out, post sessions the full pre fit).
Usage: python scripts/frozen_musall_figures.py [--tag gate-attempted] [--areas cortex SSp-m_left]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

EPOCHS = ("pre", "acute", "subacute", "chronic")
MEASURES = ("avg", "trial", "frame")
MLABEL = {"avg": "trial-averaged", "trial": "single-trial", "frame": "frame (all)"}
#: what "unique" means here (Priya 10-22): differences of REFITS (order-free, drop-style), not Musall's shuffle-in-the-
#: ordered-model; movement / direction are gains when ADDED in sequence, so they are not beyond the later groups
#: a task-independent part below this is flagged (pre-stroke LOSO noise is ~ +-0.02)
NEG_TOL = -0.02
UNIQUE_NOTE = ("unique = gain of a refit with vs without the group (no shuffle, no orthogonalisation): task = joint - "
               "no_task (beyond all); movement = beyond task; direction = beyond task + movement; contact = beyond all")


def quantities(A: pd.DataFrame) -> pd.DataFrame:
    """Musall quantities per animal x epoch x align x measure x area, from the variant table."""
    W = A.pivot_table(index=["animal", "epoch", "align", "measure", "area"], columns="variant", values="r2")
    q = pd.DataFrame(index=W.index)

    def g(v):
        return W[v] if v in W else np.nan
    q["alone:task"] = g("no_movement")
    q["alone:movement"] = g("movement_only")
    q["alone:non-task"] = g("no_task")
    q["unique:task"] = g("joint") - g("no_task")
    q["unique:movement"] = g("task_movement") - g("no_movement")
    q["unique:direction"] = g("task_movement_dir") - g("task_movement")
    q["unique:contact"] = g("joint") - g("task_movement_dir")
    q["indep:non-task"] = g("joint") - g("no_movement")
    q["aligned:non-task"] = q["alone:non-task"] - q["indep:non-task"]
    q["indep:movement"] = q["unique:movement"]
    q["aligned:movement"] = q["alone:movement"] - q["indep:movement"]
    q["full"] = g("joint")
    return q.reset_index()


def _bars(ax, Q, cols, colors, labels=None, stacked=None):
    x = np.arange(len(cols))
    for i, c in enumerate(cols):
        v = Q.groupby("animal")[c].mean()
        ax.bar(x[i], v.mean(), 0.7, color=colors[i], alpha=0.85)
        ax.scatter(np.full(len(v), x[i]) + np.linspace(-0.15, 0.15, len(v)), v.values, s=9, color="k", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels or cols, rotation=40, ha="right", fontsize=7)
    ax.axhline(0, color="0.5", lw=0.6)


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    from wfield_local.writeguard import assert_writable
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="gate-attempted")
    ap.add_argument("--areas", nargs="+", default=["cortex", "SSp-m_left", "MOs_left"])
    a = ap.parse_args(argv)
    out = oo.root(PathResolver()) / oc.RUN / "imaging_encoding" / f"frozen_encoder_{a.tag}"
    A = pd.read_csv(out / "frozen_musall_r2_by_animal.csv")
    Q = quantities(A)
    fd = out / "figures"
    assert_writable(fd)
    fd.mkdir(parents=True, exist_ok=True)
    Q.to_csv(out / "frozen_musall_quantities_by_animal.csv", index=False, float_format="%.4f")
    epochs = [e for e in EPOCHS if e in set(Q.epoch)]
    stamp = ("FROZEN pre-stroke encoder (per animal, August pre; pre sessions leave-one-session-out). Attempted-trial "
             "gate, cam4 + cam1 licks (cam4 only where cam1 is missing). PROVISIONAL.")
    for area in a.areas:
        for align in ("cue", "lick"):
            S = Q[(Q["area"] == area) & (Q["align"] == align)]
            if S.empty:
                continue
            # (1) alone vs unique
            fig, axs = plt.subplots(len(MEASURES), len(epochs), figsize=(3.0 * len(epochs), 2.6 * len(MEASURES)),
                                    squeeze=False, sharey="row")
            cols = ["alone:task", "unique:task", "alone:movement", "unique:movement", "unique:direction",
                    "unique:contact"]
            colors = ["#9ecae1", "#3182bd", "#fdae6b", "#e6550d", "#a1d99b", "#756bb1"]
            for i, meas in enumerate(MEASURES):
                for j, ep in enumerate(epochs):
                    ax = axs[i][j]
                    _bars(ax, S[(S.measure == meas) & (S.epoch == ep)], cols, colors)
                    if i == 0:
                        ax.set_title(ep, fontsize=9)
                    if j == 0:
                        ax.set_ylabel(f"{MLABEL[meas]} R$^2$", fontsize=8)
            fig.suptitle(f"Alone (light) vs unique (dark), {area}, {align}-aligned -- Musall Fig. 4a style\n"
                         f"{UNIQUE_NOTE}\n{stamp}", fontsize=8)
            fig.tight_layout()
            fig.savefig(fd / f"musall_alone_vs_unique_{area}_{align}.png", dpi=130)
            plt.close(fig)
            # (2) task-aligned vs task-independent
            fig, axs = plt.subplots(len(MEASURES), len(epochs), figsize=(2.6 * len(epochs), 2.4 * len(MEASURES)),
                                    squeeze=False, sharey="row")
            for i, meas in enumerate(MEASURES):
                for j, ep in enumerate(epochs):
                    ax = axs[i][j]
                    T = S[(S.measure == meas) & (S.epoch == ep)].groupby("animal").mean(numeric_only=True)
                    broken = []
                    for k, grp in enumerate(("movement", "non-task")):
                        alone, ind = T[f"alone:{grp}"], T[f"indep:{grp}"]
                        col = "#e6550d" if grp == "movement" else "#636363"
                        # whole bar = ALONE (grey = task-aligned part); task-independent drawn from 0 on top of it, so
                        # a NEGATIVE task-independent part (frozen model scored on new data: adding the group costs
                        # R^2) is drawn below zero, hatched, instead of hidden under a grey bar stacked on it
                        ax.bar(k, alone.mean(), 0.6, color="0.82", label="task-aligned (alone - indep.)" if k == 0 else None)
                        neg = ind.mean() < 0
                        ax.bar(k, ind.mean(), 0.42, color="white" if neg else col, edgecolor=col,
                               hatch="////" if neg else None, lw=0.8,
                               label="task-independent" if k == 0 else None)
                        xs = k + np.linspace(-0.2, 0.2, len(T))
                        ax.scatter(xs, alone.values, s=10, facecolor="none", edgecolor="k", lw=0.6, zorder=3)
                        ax.scatter(xs, ind.values, s=10, color=col, edgecolor="k", lw=0.4, zorder=4)
                        if (ind < NEG_TOL).any():
                            broken.append(f"{grp}: {', '.join(ind.index[ind < NEG_TOL])}")
                    ax.axhline(0, color="0.4", lw=0.6)
                    ax.set_xticks([0, 1])
                    ax.set_xticklabels(["movement\n(directionless)", "all non-task"], fontsize=7)
                    if broken:                   # Musall's split assumes task-independent >= 0
                        ax.text(0.02, 0.98, f"indep. < {NEG_TOL} (split not valid):\n" +"\n".join(broken), transform=ax.transAxes,
                                va="top", fontsize=5.5, color="#b30000")
                    if i == 0:
                        ax.set_title(ep, fontsize=9)
                    if j == 0:
                        ax.set_ylabel(f"{MLABEL[meas]} R$^2$", fontsize=8)
            axs[0][0].legend(fontsize=6, frameon=False, loc="lower right")
            fig.suptitle(f"Task-aligned vs task-independent, {area}, {align}-aligned -- Musall Fig. 5 style.\nBar = alone "
                         f"(grey = task-aligned), inner bar = task-independent (hatched if < 0); open dots = alone, filled = "
                         f"task-independent, per animal\n{stamp}", fontsize=8)
            fig.tight_layout()
            fig.savefig(fd / f"musall_task_aligned_{area}_{align}.png", dpi=130)
            plt.close(fig)
            # (3) trial-averaged vs single-trial vs frame, full + alone models
            fig, axs = plt.subplots(1, len(epochs), figsize=(3.0 * len(epochs), 2.8), squeeze=False, sharey=True)
            mods = ["full", "alone:task", "alone:movement", "alone:non-task"]
            for j, ep in enumerate(epochs):
                ax = axs[0][j]
                for mi, meas in enumerate(MEASURES):
                    T = S[(S.measure == meas) & (S.epoch == ep)].groupby("animal").mean(numeric_only=True)
                    x = np.arange(len(mods)) + (mi - 1) * 0.26
                    ax.bar(x, [T[m].mean() for m in mods], 0.25, label=MLABEL[meas],
                           color=("#2c7fb8", "#f03b20", "0.6")[mi])
                    for xi, m in zip(x, mods):                       # animals
                        ax.scatter(np.full(len(T), xi), T[m].values, s=5, color="k", zorder=3)
                ax.set_xticks(np.arange(len(mods)))
                ax.set_xticklabels(["full", "task\nalone", "movement\nalone", "non-task\nalone"], fontsize=7)
                ax.axhline(0, color="0.5", lw=0.6)
                ax.set_title(ep, fontsize=9)
            axs[0][0].set_ylabel("held-out R$^2$", fontsize=8)
            axs[0][0].legend(fontsize=6, frameon=False)
            fig.suptitle(f"Trial-averaged (PETH) vs single-trial vs frame R$^2$, {area}, {align}-aligned\n{stamp}",
                         fontsize=8)
            fig.tight_layout()
            fig.savefig(fd / f"musall_peth_vs_trial_{area}_{align}.png", dpi=130)
            plt.close(fig)
    print(f"-> {fd} (musall_*.png); quantities {out / 'frozen_musall_quantities_by_animal.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
