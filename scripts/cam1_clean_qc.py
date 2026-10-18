"""QC figures for tuning cam1 clean traces (`docs/POSE_MODELS.md` D2; Priya 2026-10-18: "build tongue/jaw clean traces for
cam1"). Model-INDEPENDENT geometry first, settled on any decent model (DLC c1r2 for now); model-specific rules wait for the
chosen cam1 model. Each figure is for Priya's approval before a value goes into `orofacial_clean.cameras.cam1`.

    python -m scripts.cam1_clean_qc mouth PS93:20260814 --model "{d}/dlc_r2/*.csv"

``mouth``: is the cam1 spout-line intersection the MOUTH? One no-lick ENL frame per spout position (spout parked, tongue not
detected; the frame nearest that position's median spout tip), with all six position medians, the three close->far lines
extended, and their least-squares intersection (`spout_frame`, as on cam4). Also prints how well the lines meet (rms) and
each line's distance from the point. Writes ``<cam1 clip folder>/eval/qc_mouth_origin.png``.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np

POS_ORDER = ["far_L", "close_L", "close_center", "far_center", "close_R", "far_R"]
SIDE_COL = {"L": (0.95, 0.45, 0.1), "center": (0.2, 0.7, 0.95), "R": (0.75, 0.3, 0.9)}


def cmd_mouth(session: str, model: str) -> None:
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from scripts.session_poses import read_index, session_dir
    from wfield_local import orofacial_clean as oc
    from wfield_local import spout_frame as SF
    from wfield_local.paths import PathResolver
    rv = PathResolver()
    animal, date, d = session_dir(rv, f"{session}:cam1")
    idx = read_index(d)
    hits = glob.glob(model.replace("{d}", str(d)))
    if len(hits) != 1:
        raise SystemExit(f"{model!r}: {len(hits)} matches")
    P = oc.read_pose(Path(hits[0])).iloc[:len(idx)]
    spans = [(int(g.index.min()), int(g.index.max()) + 1, str(g.position.iloc[0])) for _, g in idx.groupby("trial_k")]
    fr = SF.from_medians(SF.position_medians(P[["spout_x", "spout_y", "spout_likelihood"]].to_numpy(), spans))
    print(f"{animal} {date} cam1: origin ({fr.origin[0]:.1f}, {fr.origin[1]:.1f}), lines meet rms {fr.rms_px:.2f} px, "
          f"distance per line {', '.join(f'{k} {v:.2f}' for k, v in fr.line_dist_px.items())} px")
    # one parked, no-lick frame per position: ENL (strobe + 300 ms .. cue), tongue not detected, spout confident, nearest the median
    t, sb = idx.t_ms.to_numpy(float), idx.strobe_ms.to_numpy(float)
    enl = (t >= sb + 300) & (t < 0) & (P.tongue_likelihood.to_numpy() < 0.1) & (P.spout_likelihood.to_numpy() >= 0.9)
    cap = cv2.VideoCapture(str(d / "windows.mp4"))                    # READ-ONLY
    fig, axs = plt.subplots(2, 3, figsize=(15, 10.5))
    for ax, pos in zip(axs.ravel(), POS_ORDER):
        m = enl & (idx.position.to_numpy() == pos)
        if not m.any() or pos not in fr.positions:
            ax.set_title(f"{pos}: no parked frame"); ax.axis("off"); continue
        px, py = fr.positions[pos][:2]
        rows = np.flatnonzero(m)
        r = rows[np.argmin(np.hypot(P.spout_x.to_numpy()[rows] - px, P.spout_y.to_numpy()[rows] - py))]
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(r))
        ok, im = cap.read()
        ax.imshow(im[:, :, ::-1])
        for side, (c, f) in SF.SIDES.items():
            if c in fr.positions and f in fr.positions:
                a, b = np.array(fr.positions[f][:2]), np.array(fr.origin)
                ext = a + 1.25 * (b - a)
                ax.plot([a[0], ext[0]], [a[1], ext[1]], "-", color=SIDE_COL[side], lw=1.2, alpha=0.9)
        for q, (x, y, _) in fr.positions.items():
            ax.plot(x, y, "o", ms=6, mfc="none", mec="w" if q != pos else "yellow", mew=1.5)
        ax.plot(*fr.origin, "x", color="red", ms=14, mew=2.5)
        jx, jy, jl = (P[f"jaw_{c}"].to_numpy() for c in ("x", "y", "likelihood"))
        ax.plot(jx[r], jy[r], "+", color="lime", ms=12, mew=2) if jl[r] >= 0.6 else None
        ax.set_title(f"{pos} (trial {int(idx.trial_id[r])}, {t[r]:+.0f} ms from cue)", fontsize=10)
        ax.set_xlim(150, 470); ax.set_ylim(560, 240); ax.axis("off")
    cap.release()
    fig.suptitle(f"{animal} {date} cam1 -- is the spout-line intersection (red X) the MOUTH?  Lines: far -> close per side "
                 f"(orange L, blue centre, purple R), extended; rings = the six median spout tips (yellow = this panel's); "
                 f"green + = jaw (p >= 0.6)\nlines meet to rms {fr.rms_px:.2f} px; origin ({fr.origin[0]:.0f}, {fr.origin[1]:.0f})",
                 fontsize=11)
    fig.tight_layout()
    out = d / "eval" / "qc_mouth_origin.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"-> {out}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["mouth"])
    ap.add_argument("session", metavar="ANIMAL:YYYYMMDD")
    ap.add_argument("--model", required=True, metavar="PATH (with {d})")
    a = ap.parse_args(argv)
    {"mouth": lambda: cmd_mouth(a.session, a.model)}[a.cmd]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
