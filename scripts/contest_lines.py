"""Epoch-line figures for the 10-26 contest results that had only tables (ENCODING_FINDINGS 8.19; Priya 10-26: "which
figure is this?"). Unique R2 = joint - no_<g> (frozen, frame measure); remapping index = self-fit unique minus frozen
unique (`session_vs_frozen`); each a change from pre within animal, nested bootstrap (`frozen_fr_compare.boot`), drawn
with `frozen_fr_compare.epoch_lines` (line = mean of animals, band = 95 % CI, filled = CI excludes 0, grey band = pre
split-half range).
  contest_landing_vs_target.png  target vs per-spout lick kernels, with the lick kernels split by LANDING (..._frA_lsland)
                                 or by TARGET spout (..._frA_ls)
  contest_fr_lags.png            lagged FaceRhythm: cortex-leads (future face) vs face-leads (past face) unique (..._frA_frlag)
  contest_selffit_remap.png      self-fit remapping: target without / with per-spout lick kernels in the design, lick
                                 kernels, whole spout group (..._frA, ..._frA_ls)
Usage: python scripts/contest_lines.py
Output: <E>/contest_lines/
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import frozen_fr_compare as ffc  # noqa: E402
from fr_musall import E  # noqa: E402

AREAS = ["SSp-m_right", "SSp-m_left", "MOp_right", "MOp_left", "cortex"]
OUT = E / "contest_lines"


def unique_table(tag: str, qs: dict) -> pd.DataFrame:
    """Per session x window x area unique R2 of each group: {series label: group}."""
    D = pd.read_csv(E / f"frozen_encoder_{tag}" / "frozen_musall_r2.csv")
    D = D[(D.measure == "frame") & (D.label.str[-4:] >= "0801") & D.area.isin(AREAS)]
    W = D.pivot_table(index=["label", "animal", "epoch", "align", "area"], columns="variant", values="r2").reset_index()
    rows = []
    for lab, g in qs.items():
        rows.append(W[["label", "animal", "epoch", "align", "area"]].assign(frame=lab, u=W["joint"] - W[f"no_{g}"]))
    return pd.concat(rows).rename(columns={"align": "window"})


def remap_table(tag: str, qs: dict) -> pd.DataFrame:
    """Per session remapping index (self-fit unique - frozen unique) of each group, from session_vs_frozen/gap.csv."""
    G = pd.read_csv(E / f"frozen_encoder_{tag}" / "session_vs_frozen" / "gap.csv")
    G = G[G.area.isin(AREAS) & (G.label.str[-4:] >= "0801")]
    W = G.pivot_table(index=["label", "animal", "epoch", "window", "area"], columns="variant", values=["self", "frozen"])
    rows = []
    for lab, g in qs.items():
        u = (W[("self", "joint")] - W[("self", f"no_{g}")]) - (W[("frozen", "joint")] - W[("frozen", f"no_{g}")])
        rows.append(u.rename("u").reset_index().assign(frame=lab))
    return pd.concat(rows)


def figure(plt, T: pd.DataFrame, series, title: str, ylab: str, path) -> None:
    B = pd.concat([ffc.boot(T[T.frame == lab], "u").assign(frame=lab) for lab, _l, _c in series])
    B.to_csv(path.with_suffix(".csv"), index=False)
    fig, axs = plt.subplots(len(ffc.WINDOWS), len(AREAS), figsize=(15, 3.3 * len(ffc.WINDOWS) + 1.3), sharey=True,
                            squeeze=False)
    for r, w in enumerate(ffc.WINDOWS):
        for c, a in enumerate(AREAS):
            ax = axs[r, c]
            ffc.epoch_lines(ax, B[B.window == w], series, a)
            if r == 0:
                ax.set_title(ffc.ALAB.get(a, a), fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{w} window\n{ylab}", fontsize=8.5)
    ffc.lines_legend(fig, series)
    fig.suptitle(title, fontsize=10)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.85, bottom=0.16, wspace=0.08, hspace=0.25)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print("->", path)


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    OUT.mkdir(parents=True, exist_ok=True)
    T = pd.concat([unique_table("gate-attempted_v7g3_t3d_frA_lsland", {"target | landing split": "target",
                                                                      "lick kernels by LANDING": "lickspout"}),
                   unique_table("gate-attempted_v7g3_t3d_frA_ls", {"target | target split": "target",
                                                                  "lick kernels by TARGET": "lickspout"})])
    figure(plt, T, [("target | landing split", "target (lick kernels split by landing)", "#b2182b"),
                    ("lick kernels by LANDING", "lick kernels split by 3-D landing", "#2166ac"),
                    ("target | target split", "target (lick kernels split by target spout)", "#f4a582"),
                    ("lick kernels by TARGET", "lick kernels split by target spout", "#92c5de")],
           "NEURAL: target vs per-spout lick kernels in the FROZEN pre-stroke model -- lick kernels assigned by where each lick "
           "LANDED (dark) or by the trial's TARGET spout (light).\nUnique R2 (joint minus the model without the group), change "
           "from pre within animal; line = mean of animals, band = 95% nested CI.", "unique R2, change from pre",
           OUT / "contest_landing_vs_target.png")
    T = unique_table("gate-attempted_v7g3_t3d_frA_frlag", {"cortex leads": "frcortexlead", "face leads": "frfacelead"})
    figure(plt, T, [("cortex leads", "cortex leads face (future-face FaceRhythm columns)", "#7b3294"),
                    ("face leads", "face leads cortex (past-face FaceRhythm columns)", "#008837")],
           "NEURAL: lagged FaceRhythm in the FROZEN model -- unique R2 of the lead (+0.1..0.3 s) vs lag (-0.1..-0.3 s) face "
           "columns, change from pre.\nPre-stroke 'face leads' dominates (calcium lag), so only the CHANGE is read.",
           "unique R2, change from pre", OUT / "contest_fr_lags.png")
    T = pd.concat([remap_table("gate-attempted_v7g3_t3d_frA", {"target | no lick kernels": "target"}),
                   remap_table("gate-attempted_v7g3_t3d_frA_ls", {"target | with lick kernels": "target",
                                                                  "lick kernels": "lickspout",
                                                                  "spout group (target + executed + lick kernels)": "spout"})])
    figure(plt, T, [("target | no lick kernels", "target, design without per-spout lick kernels", "#f4a582"),
                    ("target | with lick kernels", "target, design WITH per-spout lick kernels", "#b2182b"),
                    ("lick kernels", "per-spout lick kernels", "#2166ac"),
                    ("spout group (target + executed + lick kernels)", "whole spout group", "#4d4d4d")],
           "NEURAL: REMAPPING index from per-session self-fits (self-fit unique minus frozen unique), change from pre.\n"
           "With per-spout lick kernels in the design the target's remapping roughly halves and the lick kernels show none, "
           "while the whole spout group stays remapped.", "remapping index, change from pre",
           OUT / "contest_selffit_remap.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
