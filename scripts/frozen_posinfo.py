"""Movement WITH vs WITHOUT lick-position information, across pose AND FaceRhythm, in the FROZEN frame (Priya 10-25: "can we
group by lick position information vs no lick position information across FR and kinematics?").

posinfo   = pose spout-locked parts + pose within-spout direction (+ executed angle) + FR side / distance (+ FR spout-locked
            parts when the run used FROZEN_FR_SPLIT=1)
noposinfo = pose lick timing / speed / opening deviations, nose, running + FR lick sum / other (their deviations when split)
Per session x area x window: unique = R2[joint] - R2[no_<g>], alone = R2[<g>_only] (frozen pre-stroke weights; pre held
out leave-one-session-out). Epoch - pre, nested bootstrap; per-animal table first (rule 8).
Usage: python scripts/frozen_posinfo.py --tag gate-attempted_v7g3_t3d_frA_pos   (quick: FR lick sum unsplit = noposinfo)
       python scripts/frozen_posinfo.py --tag gate-attempted_v7g3_t3d_frAsplit  (FR split like pose)
Output: <E>/frozen_fr_compare/posinfo_<tag>.csv, figures/frozen_posinfo_<tag>_{cue,lick}.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fr_musall import E  # noqa: E402
from frozen_fr_compare import ALAB, WINDOWS, bars, boot, session_r2  # noqa: E402

KEY = ["label", "animal", "epoch", "window", "area"]
AREAS = ["cortex", "SSp-m_right", "SSp-m_left", "MOp_right", "MOp_left"]
QS = [("posinfo unique", "no_posinfo", "unique"), ("noposinfo unique", "no_noposinfo", "unique"),
      ("posinfo alone", "posinfo_only", "alone"), ("noposinfo alone", "noposinfo_only", "alone")]
COLS = {"posinfo unique": "#d95f02", "noposinfo unique": "#7570b3", "posinfo alone": "#fdb863", "noposinfo alone": "#b2abd2"}


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import frozen_fr_compare as ffc
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="gate-attempted_v7g3_t3d_frA_pos")
    a = ap.parse_args(argv)
    folder = f"frozen_encoder_{a.tag}"
    out = E / "frozen_fr_compare"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    J = session_r2(folder, "joint").set_index(KEY).r2
    rows = []
    for q, v, kind in QS:
        if not (E / folder / v / "sessions").is_dir():
            continue
        R = session_r2(folder, v).set_index(KEY).r2
        rows.append(((J - R) if kind == "unique" else R).rename("u").reset_index().assign(q=q))
    U = pd.concat(rows)
    pd.set_option("display.width", 220)
    per = U[U.area.isin(["SSp-m_right", "SSp-m_left"])].groupby(["q", "window", "area", "animal", "epoch"]).u.mean() \
        .unstack("epoch")[["pre", "acute", "subacute", "chronic"]]
    print("== per animal (rule 8)\n" + per.round(3).to_string())
    B = pd.concat([boot(U[U.q == q], "u").assign(frame=q, q=q) for q in U.q.unique()])
    B.to_csv(out / f"posinfo_{a.tag}.csv", index=False)
    ffc.COL.update(COLS)
    split = "split" in a.tag
    for w in WINDOWS:
        fig, axs = plt.subplots(2, 1, figsize=(15, 8.2))
        for ax, kind in zip(axs, ("unique", "alone")):
            qs = [q for q, _v, k in QS if k == kind and (B.q == q).any()]
            bars(ax, B[(B.window == w) & B.q.isin(qs)], qs, AREAS, "change from pre (R2)",
                 f"{kind.upper()} R2, epoch - pre: orange = movement WITH lick-position information, purple = WITHOUT")
            pre = B[(B.window == w) & B.q.isin(qs) & (B.epoch == "chronic")].set_index(["q", "area"]).ref
            ax.text(0.005, 0.03, "pre level: " + "; ".join(f"{q.split()[0]} {ALAB[ar]} {pre.get((q, ar), float('nan')):.3f}"
                                                         for q in qs for ar in ("SSp-m_right", "cortex")),
                    transform=ax.transAxes, fontsize=7)
            ax.legend(fontsize=8, frameon=False, loc="upper right")
        note = ("FR split into spout-locked template + deviation like pose" if split else
                "QUICK version: FR lick sum unsplit, counted as no-position (approximate)")
        fig.suptitle(f"NEURAL: movement WITH vs WITHOUT lick-position information (pose + FaceRhythm), FROZEN frame, {w} window "
                     f"({note}).\nBars = mean of animals' epoch - pre; whisker 95 % nested CI; * excludes 0; dots = animals. "
                     "contra = right (lesion left).", fontsize=9.5)
        fig.subplots_adjust(left=0.05, right=0.995, top=0.88, bottom=0.07, hspace=0.45)
        fig.savefig(out / "figures" / f"frozen_posinfo_{a.tag}_{w}.png", dpi=105)
        plt.close(fig)
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
