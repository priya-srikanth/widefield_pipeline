"""TARGET vs EXECUTION readout of the FROZEN_DESIGN=te run (DECISIONS 2026-10-26; Priya: "target kernel vs execution
kernel"). Groups (`frozen_encoder` TE_BLOCK): target = spout identity (pos_enl, pos_cue) + lick-locked kernels labelled
by the trial's target spout; execution = 3-D tongue / jaw, exec angle, FaceRhythm, lick-locked kernels labelled by
3-D landing (ITI licks included); contact = every DAQ lick; timing / other nuisance.

Per session x window (cue / lick) x area, from `<run>/frozen_musall_r2.csv` (held out: pre sessions leave-one-session-
out, post sessions the full pre fit): UNIQUE(g) = joint - no_<g> (a refit without the group: order-free, shared variance
credited to nobody), ALONE(g) = <g>_only; contrast execution - target (unique). Change from pre per epoch, nested
bootstrap (animals > sessions; `frozen_fr_compare.boot`); per-animal table first (rule 8).
Figure: <run>/figures/te_unique_lines.png (epoch lines, one row per window, columns = areas).
Usage: python scripts/te_readout.py [--tag gate-attempted_v7g3_t3d_frA_te] [--measure frame]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import frozen_fr_compare as ffc  # noqa: E402
from fr_musall import E  # noqa: E402

AREAS = ["SSp-m_right", "SSp-m_left", "MOp_right", "MOp_left", "cortex"]
UNIQUE = {"target": "no_target", "execution": "no_execution", "contact": "no_contact", "task": "no_task",
          "movement": "no_movement"}
ALONE = {"target alone": "target_only", "execution alone": "execution_only"}
SERIES = [("target", "target (unique)", "#b2182b"), ("execution", "execution (unique)", "#2166ac"),
          ("contact", "contact (unique)", "#999999"), ("execution - target", "execution minus target", "#01665e")]


def table(run: Path, measure: str) -> pd.DataFrame:
    M = pd.read_csv(run / "frozen_musall_r2.csv")
    M = M[(M.measure == measure) & M.area.isin(AREAS)]
    W = M.pivot_table(index=["label", "animal", "epoch", "align", "area"], columns="variant", values="r2")
    out = {"joint": W["joint"]}
    for g, v in UNIQUE.items():
        if v in W:
            out[g] = W["joint"] - W[v]
    for g, v in ALONE.items():
        if v in W:
            out[g] = W[v]
    T = pd.DataFrame(out)
    T["execution - target"] = T["execution"] - T["target"]
    return T.reset_index().rename(columns={"align": "window"})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="gate-attempted_v7g3_t3d_frA_te")
    ap.add_argument("--measure", default="frame", choices=["frame", "trial", "avg"])
    a = ap.parse_args(argv)
    run = E / f"frozen_encoder_{a.tag}"
    T = table(run, a.measure)
    vals = [c for c in T.columns if c not in ("label", "animal", "epoch", "window", "area")]
    L = T.melt(id_vars=["label", "animal", "epoch", "window", "area"], value_vars=vals, var_name="q", value_name="v")
    pd.set_option("display.width", 220)
    print(f"== {a.measure}: per animal mean (rule 8), lick window")
    per = L[(L.window == "lick") & L.q.isin(["target", "execution", "contact", "execution - target"])] \
        .groupby(["q", "area", "animal", "epoch"]).v.mean().unstack("epoch").reindex(columns=ffc.EP4)
    print(per.round(4).to_string())
    B = pd.concat([ffc.boot(L[L.q == q], "v").assign(q=q) for q in vals], ignore_index=True)
    B.to_csv(run / f"te_unique_boot_{a.measure}.csv", index=False)
    T.to_csv(run / f"te_unique_sessions_{a.measure}.csv", index=False)
    print(B[B.q.isin(["target", "execution", "contact", "execution - target", "target alone", "execution alone"])]
          [["q", "window", "area", "epoch", "ref", "point", "lo", "hi", "n_up", "n"]].round(4).to_string(index=False))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(len(ffc.WINDOWS), len(AREAS), figsize=(14, 3.2 * len(ffc.WINDOWS) + 1.2), sharey=True,
                            squeeze=False)
    for r, w in enumerate(ffc.WINDOWS):
        for c, ar in enumerate(AREAS):
            ax = axs[r, c]
            ffc.epoch_lines(ax, B[B.window == w], SERIES, ar, key="q")
            if r == 0:
                ax.set_title(ffc.ALAB.get(ar, ar), fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{w} window\nunique R2, change from pre", fontsize=8.5)
    ffc.lines_legend(fig, SERIES, split=False)
    fig.suptitle(f"NEURAL: TARGET vs EXECUTION kernels in the frozen pre-stroke encoder ({a.measure} R2; "
                 f"{run.name.replace('frozen_encoder_', '')}).\nUnique = full model minus a refit without the group; change "
                 "from pre within animal (line = mean of animals, band = 95% nested CI animals > sessions).\nTarget = spout "
                 "identity + target-labelled lick kernels; execution = 3-D tongue / jaw, exec angle, FaceRhythm, landing-"
                 "labelled lick kernels; contact = every DAQ lick.", fontsize=9.5)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.8, bottom=0.15, wspace=0.08, hspace=0.25)
    (run / "figures").mkdir(exist_ok=True)
    fig.savefig(run / "figures" / f"te_unique_lines_{a.measure}.png", dpi=110)
    plt.close(fig)
    print("->", run / "figures" / f"te_unique_lines_{a.measure}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
