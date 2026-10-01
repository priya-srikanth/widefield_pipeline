"""Test the spout-position reference frame: do the close->far lines for L, centre and R converge on the mouth?

    python -m scripts.spout_reference_frame                 # the six round-4 sessions (cached DLC poses)

WHY (Priya, 2026-10-01): "use the median spout tip position during the trial time (from position strobe to
trial end) at all 6 positions to determine the 'positions' of the spout at each. then draw lines connect
close/far center, close/far L, close/far R, which should project back to the assigned 'mouth'." If they do, the
six spout positions define a per-session task frame (AP = close_center -> far_center, LR = far_L -> far_R)
for tongue angles, with no extra labels.

DATA. The DLC round-3 poses cached by `wfield_local.dlc_hard_frames scan` (`round4_scan/<animal>_<date>_dlc.npz`):
12 trials per session, 2 per position, each cue - 1.0 s .. cue + 3.5 s. That span lies inside position-strobe ..
trial-end (the spout arrives ~3 s before the cue; the response window is 3.5 s), so the spout is parked at its
position throughout -- a subset of the requested span, not all of it. Spout frames with likelihood <= 0.6 are
excluded; the median is per position over every remaining frame.

GEOMETRY. Each side's line runs from far_<side> through close_<side> and is extended past close toward the animal.
The "projected mouth" is the least-squares point closest to all three lines (perpendicular distances); the
residual is the RMS of those distances (0 = the three lines meet exactly). Compared against the nose and the
resting jaw (median jaw in the pre-cue second, i.e. mouth closed), both labelled points.
"""
from __future__ import annotations


import numpy as np
import pandas as pd

from wfield_local import dlc_frames as DF
from wfield_local.dlc_hard_frames import PARTS, _cache, _session_io, load_poses
from wfield_local.paths import PathResolver

PCUT = 0.6
SIDES = {"L": ("close_L", "far_L"), "center": ("close_center", "far_center"), "R": ("close_R", "far_R")}
POS_COLORS = {"close_L": "#1f77b4", "far_L": "#aec7e8", "close_center": "#2ca02c", "far_center": "#98df8a",
              "close_R": "#d62728", "far_R": "#ff9896"}


def spout_medians(wins) -> dict[str, tuple[float, float, int]]:
    """{position: (x, y, n_frames)} -- median confident spout tip per position."""
    k = PARTS.index("spout")
    acc: dict[str, list[np.ndarray]] = {}
    for _f0, tr, pose in wins:
        ok = pose[:, k, 2] > PCUT
        acc.setdefault(tr["pos_name"], []).append(pose[ok, k, :2])
    out = {}
    for pos, arrs in acc.items():
        a = np.concatenate(arrs)
        if len(a):
            out[pos] = (float(np.median(a[:, 0])), float(np.median(a[:, 1])), len(a))
    return out


def rest_point(wins, part: str, pre_frames: int = 250) -> tuple[float, float]:
    """Median confident `part` over the first ``pre_frames`` of each window (the pre-cue second: at rest)."""
    k = PARTS.index(part)
    a = np.concatenate([pose[:pre_frames][pose[:pre_frames, k, 2] > PCUT, k, :2] for _, _, pose in wins])
    return float(np.median(a[:, 0])), float(np.median(a[:, 1]))


def ls_intersection(lines) -> tuple[np.ndarray, float, list[float]]:
    """Least-squares point nearest to lines given as (point, direction); returns (point, rms, per-line distance)."""
    A, b = np.zeros((2, 2)), np.zeros(2)
    for p, d in lines:
        d = d / np.linalg.norm(d)
        M = np.eye(2) - np.outer(d, d)                 # projector onto the line's normal
        A += M
        b += M @ p
    x = np.linalg.solve(A, b)
    dist = [float(np.linalg.norm((np.eye(2) - np.outer(d / np.linalg.norm(d), d / np.linalg.norm(d))) @ (x - p)))
            for p, d in lines]
    return x, float(np.sqrt(np.mean(np.square(dist)))), dist


def angle_deg(u, v) -> float:
    """Unsigned angle between two direction vectors, degrees."""
    c = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    return float(np.degrees(np.arccos(np.clip(abs(c), -1, 1))))


def main() -> int:
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rv = PathResolver()
    sess = pd.read_csv(_cache(rv) / "sessions.csv", dtype=str)
    out_dir = DF.staging_root(rv).parent / "reference_frame"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    fig, axs = plt.subplots(2, 3, figsize=(15, 10.5))
    for ax, s in zip(axs.ravel(), sess.itertuples()):
        tpl, vid, _t = _session_io(s.animal, s.date, s.sid, rv)
        wins = load_poses(_cache(rv) / f"{s.animal}_{s.date}_dlc.npz")
        med = spout_medians(wins)
        nose, jaw = rest_point(wins, "nose"), rest_point(wins, "jaw")
        lines = {side: (np.array(med[c][:2]), np.array(med[c][:2]) - np.array(med[f][:2]))
                 for side, (c, f) in SIDES.items() if c in med and f in med}
        cap = cv2.VideoCapture(str(vid))               # READ-ONLY
        cap.set(cv2.CAP_PROP_POS_FRAMES, wins[0][0] + 10)
        ok, im = cap.read()
        cap.release()
        if ok:
            ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        row = {"animal": s.animal, "date": s.date, "epoch": s.epoch,
               **{f"{p}_xy": f"{v[0]:.0f},{v[1]:.0f} (n={v[2]})" for p, v in med.items()},
               "nose": f"{nose[0]:.0f},{nose[1]:.0f}", "jaw_rest": f"{jaw[0]:.0f},{jaw[1]:.0f}"}
        if len(lines) >= 2:
            X, rms, dist = ls_intersection(list(lines.values()))
            for side, (_p, d) in lines.items():
                f_pt = np.array(med[SIDES[side][1]][:2])
                t_end = np.dot(X - f_pt, d) / np.dot(d, d)            # extend from far, through close, to X
                seg = np.array([f_pt, f_pt + d * max(t_end, 1.0) * 1.15])
                ax.plot(seg[:, 0], seg[:, 1], "-", lw=1.2, color=POS_COLORS[SIDES[side][0]])
            ax.plot(*X, "*", ms=16, mfc="yellow", mec="k", label=f"lines' meeting point (rms {rms:.0f} px)")
            mid = np.array(jaw) - np.array(nose)
            row.update({"meet_xy": f"{X[0]:.0f},{X[1]:.0f}", "line_rms_px": round(rms, 1),
                        **{f"dist_{k}": round(v, 1) for k, v in zip(lines, dist)},
                        "meet_to_jaw_px": round(float(np.linalg.norm(X - np.array(jaw))), 1),
                        "meet_to_nose_px": round(float(np.linalg.norm(X - np.array(nose))), 1),
                        "center_axis_vs_midline_deg": round(angle_deg(lines["center"][1], mid), 1) if "center" in lines else np.nan})
        for p, (x, y, _n) in med.items():
            ax.plot(x, y, "o", ms=8, mfc=POS_COLORS[p], mec="k", label=p)
        ax.plot(*nose, "s", ms=8, mfc="cyan", mec="k", label="nose (rest)")
        ax.plot(*jaw, "D", ms=7, mfc="blue", mec="k", label="jaw (rest)")
        ax.set_title(f"{s.animal} {s.date} {s.epoch} | meet-jaw {row.get('meet_to_jaw_px', np.nan)} px, "
                     f"rms {row.get('line_rms_px', np.nan)}", fontsize=9)
        ax.set_xlim(150, 530)
        ax.set_ylim(640, 230)
        ax.axis("off")
        rows.append(row)
    axs.ravel()[0].legend(fontsize=6, loc="lower left")
    fig.suptitle("Spout reference frame (cam4, DLC round 3): median spout tip per position, close<-far lines extended "
                 "toward the animal; star = least-squares meeting point", fontsize=11)
    fig.tight_layout()
    p = out_dir / "spout_reference_frame_round4_sessions.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    t = pd.DataFrame(rows)
    t.to_csv(out_dir / "spout_reference_frame_round4_sessions.csv", index=False)
    cols = ["animal", "date", "epoch", "meet_xy", "line_rms_px", "meet_to_jaw_px", "meet_to_nose_px",
            "center_axis_vs_midline_deg", "nose", "jaw_rest"]
    print(t[[c for c in cols if c in t]].to_string(index=False))
    print(f"-> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
