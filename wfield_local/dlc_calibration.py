"""ChArUco visibility survey for the 4-camera behavior rig — the go/no-go gate for 3D DLC.

Triangulating DLC keypoints (anipose / aniposelib) needs a calibration recording in which a ChArUco
board is seen by every camera and — crucially — seen by camera PAIRS at the same instant. The bundle
adjustment builds a graph whose nodes are cameras and whose edges are pairs with enough simultaneous
board detections; if that graph is not connected, the cameras cannot be placed in one coordinate
frame and no amount of downstream code fixes it. This module measures that graph BEFORE anyone
labels a frame or books GPU time, because the only remedy is to re-record the calibration with the
animal off the rig.

It answers three questions, in order:

1. **Can each camera see the board at all?** (intrinsics: focal length + distortion, per camera)
2. **Which camera PAIRS see it simultaneously?** (extrinsics: relative pose)
3. **Is the pair graph connected?** (can all four cameras land in one frame)

**IT COUNTS POSES, NOT FRAMES**, and the first version did not — which made it wrong in both
directions on the same recording. Priya, 2026-09-08: *"do cam4 and cam2/3 really not have more
co-visible frames? I thought I recorded for a fair amount of time at many angles."* She had, and the
frame-based verdict was an artefact. Measured on ``camera_calibration_20260805`` at 50 Hz:

    cam    usable  poses   px/bit    pair        frames  poses
    cam1     6350     20     5.2     cam1-cam4     6163     22   OK
    cam2       70     16     2.1     cam1-cam2       69     16   OK
    cam3        8      7     ~2      cam2-cam4       70     16   OK
    cam4     8969      4     5.2     cam1-cam3        8      7   thin
                                     cam3-cam4        8      7   thin
                                     cam2-cam3        0      0   none

Three things only the pose count shows. **cam2 IS connected** to the snout pair (16 shared poses) —
the earlier "11 frames" was a step-25 sampling artefact, and the same recording read 69 at step 5.
**cam3 is the real hole** at 7 shared poses. And **cam4, which looks like the best camera on every
frame-based measure, has FOUR distinct poses** out of 8,969 usable frames: the board was held in
front of it rather than swept, so its intrinsics are the least constrained in the rig.

**Why the side views resolve so few markers.** On cam2 the board's squares ARE found — 58 rejected
candidates in one frame — at 12.6 px per marker side, i.e. **2.1 px per code cell** against cam4's
5.1. A DICT_4X4 marker is 6 cells across and stops decoding below ~3, so this is a board that is
present and too small to READ. Detector tuning does not touch it (minMarkerPerimeterRate, corner
refinement and threshold ladders all return the same 6 markers); a physically larger board, or one
held closer, is the only fix. Reporting "never presented" instead of "too small" would send someone
to re-sweep a board that cannot work.

**The board.** ``DICT_4X4_50``; marker ids 0-37 observed, i.e. a 38-marker board. Its physical
geometry (squaresX/Y, square mm, marker mm) is NOT recoverable from the video and is not needed
here — this module counts detections, it does not solve for camera matrices. Metric 3D
reconstruction does need it, so record it in ``configs/defaults.yaml`` before running a calibration.

Requires OpenCV >= 4.7 (``cv2.aruco`` moved into the main distribution then; no contrib build
needed).

CLI::

    python -m wfield_local.dlc_calibration                  # newest camera_calibration_* dir
    python -m wfield_local.dlc_calibration --dir <path>     # a specific recording
    python -m wfield_local.dlc_calibration --step 10        # denser sampling (slower)
"""
from __future__ import annotations

import argparse
import itertools
import re
from pathlib import Path

import pandas as pd

from wfield_local import writeguard

#: Measured from ``camera_calibration_20260805`` by trying every predefined dictionary against a
#: frame in which the board was large and in focus: only the 4X4 family matched.
ARUCO_DICT = "DICT_4X4_50"

#: A ChArUco corner needs its four surrounding markers, so 4 is the smallest count that can pin a
#: board pose at all. It is the floor for "this frame is usable", not a recommendation.
MIN_MARKERS = 4

#: POSES, NOT FRAMES — and this is the correction that matters (Priya, 2026-09-08: "I thought I
#: recorded for a fair amount of time at many angles"). She had. Counting FRAMES made this module
#: report a harsher verdict than the data supported, in both directions:
#:
#:   * it undercounted, because a threshold on frames SAMPLED at ``step`` is a different quantity at
#:     every step. cam1-cam2 read 11 at step 25 and 69 at step 5 — same recording, opposite verdict.
#:   * it OVERcounted, because at 250 fps a board held still for two seconds is 500 frames and ONE
#:     pose, and a calibration is constrained by distinct views of the board, not by frame count.
#:     cam4 has 44,845 usable frames and only FOUR distinct poses.
#:
#: A pose here is a run of co-visible samples separated from the next by more than ``POSE_GAP_S``.
POSE_GAP_S = 0.5

#: Distinct board poses per camera for usable intrinsics. OpenCV's own guidance is ~20 well-spread
#: views; below that focal length and distortion trade off against each other.
MIN_CAM_POSES = 20

#: Distinct poses seen simultaneously by a PAIR before its relative pose is worth estimating.
MIN_PAIR_POSES = 15

CAM_RE = re.compile(r"^(cam\d+)_", re.IGNORECASE)
#: Calibration folders are named by hand and the convention has already drifted --
#: `camera_calibration_20260805` became `Widefield_camera_calibration_20260805`, and the next one
#: was `Widefield_calibration_20260910_4cm_6x6_Charuco`. Matching the exact old name meant the
#: newest recording was invisible to `find_calibration_dir` the day it appeared. So: anything with
#: "calibration" and an 8-digit date in it, and the DATE is what orders them, not the rest of the
#: name -- a trailing description of the board is useful to a human and must not affect the pick.
CAL_DIR_RE = re.compile(r"calibration[_-](\d{8})", re.IGNORECASE)


def find_calibration_dir(root=None, machine=None) -> Path:
    """Newest ``camera_calibration_<YYYYMMDD>/`` under the behavior-camera tree.

    Newest by the DATE IN THE NAME, not by mtime: these directories get touched whenever the
    dropped-frame QC is re-run or a file is copied, so mtime would silently prefer whichever one
    was last poked. Re-record a calibration and it is picked up with no config edit -- which is the
    point, since the verdict this module reports moves with the recording.
    """
    if root is None:
        from wfield_local.paths import PathResolver
        root = PathResolver(machine=machine).root("camera_calibration")
    root = Path(root)
    dirs = []
    for p in sorted(root.iterdir()):
        m = CAL_DIR_RE.search(p.name)
        if m and p.is_dir():
            dirs.append((m.group(1), p.name, p))
    if not dirs:
        raise FileNotFoundError(f"No camera_calibration_<YYYYMMDD>/ directory under {root}")
    return max(dirs)[2]


def _detector():
    from cv2 import aruco
    params = aruco.DetectorParameters()
    # The board is often small and motion-blurred in the wide side views; widening the adaptive
    # threshold window is what lets those frames contribute at all.
    params.adaptiveThreshWinSizeMax = 45
    return aruco.ArucoDetector(aruco.getPredefinedDictionary(getattr(aruco, ARUCO_DICT)), params)


#: A DICT_4X4 marker is 6 bits across including its border, so one marker side spans 6 code cells.
#: Below ~3 px per cell the pattern stops being readable however good the optics: the square is
#: still found as a candidate and then REJECTED, which is why a too-small board looks like a board
#: that was never shown unless the rejections are counted.
BITS_ACROSS = 6
MIN_PX_PER_BIT = 3.0


def _px_per_bit(corners) -> float:
    """Median code-cell size in pixels across the decoded markers of one frame, or 0 if none.

    THE MOST USEFUL NUMBER IN THIS MODULE, and it was missing from the first version. It separates
    "the board was never presented to this camera" from "the board was presented and is too small to
    read", which have completely different fixes -- re-sweep vs print a bigger board.
    """
    import cv2
    import numpy as np

    if corners is None or len(corners) == 0:
        return 0.0
    sides = [cv2.arcLength(np.asarray(c, dtype=np.float32).reshape(-1, 2), True) / 4.0
             for c in corners]
    return round(float(np.median(sides)) / BITS_ACROSS, 2)


def count_poses(frames, fps: float = 250.0, gap_s: float = POSE_GAP_S) -> int:
    """Distinct board poses among sampled frame indices: runs separated by more than ``gap_s``.

    Robust to the sampling step in a way a frame count is not, and it is the quantity a calibration
    actually consumes -- 500 consecutive frames of a board held still are one view of it, however
    densely they are sampled.
    """
    import numpy as np
    f = np.sort(np.asarray(list(frames), dtype=float))
    if f.size == 0:
        return 0
    return 1 + int((np.diff(f) / fps > gap_s).sum())


def survey_video(path, step: int = 25) -> pd.DataFrame:
    """Per-sampled-frame marker counts for one video: columns ``cam, frame, n_markers, ids``.

    Decodes SEQUENTIALLY (``grab()`` and skip) rather than seeking. These are 250 fps recordings --
    a calibration video is ~48k frames and a session video ~1.5M -- and random seeking in the FMP4
    stream costs a keyframe re-decode per call, which is far slower over the share than walking the
    file once.
    """
    import cv2
    path = Path(path)
    m = CAM_RE.match(path.name)
    cam = m.group(1).lower() if m else path.stem
    det = _detector()
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {path}")
    rows, i = [], 0
    try:
        while cap.grab():
            if i % step == 0:
                ok, frame = cap.retrieve()
                if ok:
                    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    corners, ids, rejected = det.detectMarkers(grey)
                    n = 0 if ids is None else len(ids)
                    id_str = "" if ids is None else " ".join(str(int(x)) for x in sorted(ids.ravel()))
                    rows.append({"cam": cam, "frame": i, "n_markers": n, "ids": id_str,
                                 # WHY a frame fails, not just that it did. `rejected` are
                                 # quadrilaterals that look like markers but whose bit pattern could
                                 # not be read -- a board present but too small reads as many
                                 # rejections and few decodes, which is a different problem from a
                                 # board that was never in view.
                                 "n_rejected": len(rejected) if rejected is not None else 0,
                                 "px_per_bit": _px_per_bit(corners)})
            i += 1
    finally:
        cap.release()
    return pd.DataFrame(rows, columns=["cam", "frame", "n_markers", "ids",
                                       "n_rejected", "px_per_bit"])


def survey(cal_dir, step: int = 25, pattern: str = "cam*.avi") -> pd.DataFrame:
    """Survey every camera video in a calibration directory."""
    vids = sorted(Path(cal_dir).glob(pattern))
    if not vids:
        raise FileNotFoundError(f"No {pattern} under {cal_dir}")
    return pd.concat([survey_video(v, step) for v in vids], ignore_index=True)


def per_camera(df: pd.DataFrame) -> pd.DataFrame:
    """One row per camera: sampled frames, detection rates, best frame."""
    out = []
    for cam, g in df.groupby("cam", sort=True):
        n = g["n_markers"]
        usable = int((n >= MIN_MARKERS).sum())
        # Frames where squares were FOUND but not read: the signature of a board that is present
        # and too small. Measured on frames with at least one rejection and no usable decode.
        seen_ppb = g.loc[g.get("px_per_bit", pd.Series(dtype=float)) > 0, "px_per_bit"]             if "px_per_bit" in g else pd.Series(dtype=float)
        rej = g["n_rejected"] if "n_rejected" in g else pd.Series(dtype=float)
        out.append({
            "cam": cam,
            "sampled": len(g),
            "any_pct": round(float((n > 0).mean() * 100), 1),
            "usable": usable,
            "usable_pct": round(float((n >= MIN_MARKERS).mean() * 100), 1),
            "max_markers": int(n.max()) if len(n) else 0,
            "px_per_bit": round(float(seen_ppb.median()), 2) if len(seen_ppb) else 0.0,
            "rejected_pct": round(float((rej > 0).mean() * 100), 1) if len(rej) else 0.0,
            "poses": count_poses(g.loc[n >= MIN_MARKERS, "frame"]),
            "ok": bool(count_poses(g.loc[n >= MIN_MARKERS, "frame"]) >= MIN_CAM_POSES),
        })
    return pd.DataFrame(out)


def per_pair(df: pd.DataFrame) -> pd.DataFrame:
    """One row per camera pair: frames in which BOTH cameras cleared ``MIN_MARKERS``.

    Pairs on the sampled frame INDEX, which is sound here only because the rig's four cameras are
    hardware-synchronised off one Arduino heartbeat and no calibration recording has dropped a frame
    (``dropframe_qc``). If a future recording drops frames, frame index stops meaning time and this
    has to pair through the Bonsai timestamps instead.
    """
    piv = df.pivot_table(index="frame", columns="cam", values="n_markers", aggfunc="max").fillna(0)
    out = []
    for a, b in itertools.combinations(sorted(piv.columns), 2):
        shared = piv.index[(piv[a] >= MIN_MARKERS) & (piv[b] >= MIN_MARKERS)]
        poses = count_poses(shared)
        out.append({"cam_a": a, "cam_b": b, "both": len(shared), "poses": poses,
                    "ok": bool(poses >= MIN_PAIR_POSES)})
    return pd.DataFrame(out, columns=["cam_a", "cam_b", "both", "poses", "ok"])


def connected(pairs: pd.DataFrame, cams) -> tuple[bool, list[list[str]]]:
    """Is the pair graph connected over ``cams``? Returns ``(connected, components)``."""
    parent = {c: c for c in cams}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for _, r in pairs[pairs["ok"]].iterrows():
        a, b = find(r["cam_a"]), find(r["cam_b"])
        if a != b:
            parent[a] = b
    groups: dict[str, list[str]] = {}
    for c in cams:
        groups.setdefault(find(c), []).append(c)
    comps = [sorted(v) for v in groups.values()]
    return len(comps) == 1, sorted(comps)


def report_lines(df: pd.DataFrame, cal_dir, step: int) -> list[str]:
    cams = per_camera(df)
    pairs = per_pair(df)
    ok, comps = connected(pairs, list(cams["cam"]))
    lines = [f"ChArUco calibration survey - {Path(cal_dir).name}",
             f"dictionary {ARUCO_DICT}; every {step}th frame; >={MIN_MARKERS} markers = usable", "",
             (f"{'cam':6s} {'sampled':>8s} {'any':>7s} {'usable':>8s} {'poses':>6s} "
              f"{'max':>5s} {'px/bit':>7s}  verdict")]
    for _, r in cams.iterrows():
        ppb = float(r.get("px_per_bit", 0.0) or 0.0)
        if r["ok"]:
            verdict = "OK"
        elif 0 < ppb < MIN_PX_PER_BIT:
            # The distinction that matters: the board WAS there and could not be read.
            verdict = (f"BOARD TOO SMALL ({ppb:.1f} px/bit, need >={MIN_PX_PER_BIT:.0f}); "
                       f"{MIN_PX_PER_BIT / ppb:.1f}x larger or closer")
        elif r["any_pct"] < 1.0:
            verdict = "BOARD NEVER PRESENTED to this camera"
        else:
            verdict = f"ONLY {int(r['poses'])} DISTINCT POSES (need {MIN_CAM_POSES})"
        lines.append(f"{r['cam']:6s} {r['sampled']:8d} {r['any_pct']:6.1f}% {r['usable']:8d} "
                     f"{int(r['poses']):6d} {r['max_markers']:5d} {ppb:7.2f}  {verdict}")
    lines += ["", f"{'pair':14s} {'frames':>8s} {'poses':>6s}  verdict"]
    for _, r in pairs.iterrows():
        verdict = "OK" if r["ok"] else f"only {int(r['poses'])} pose(s), need {MIN_PAIR_POSES}"
        lines.append(f"{r['cam_a']}-{r['cam_b']:9s} {r['both']:8d} {int(r['poses']):6d}  {verdict}")
    lines.append("")
    if ok:
        lines.append("RESULT: pair graph CONNECTED - all cameras can be placed in one frame.")
    else:
        lines.append("RESULT: pair graph NOT CONNECTED - 3D reconstruction is impossible from this "
                     "recording.")
        for comp in comps:
            lines.append(f"  component: {', '.join(comp)}")
        small = [r for _, r in cams.iterrows()
                 if not r["ok"] and 0 < float(r.get("px_per_bit", 0.0) or 0.0) < MIN_PX_PER_BIT]
        if small:
            need = max(MIN_PX_PER_BIT / float(r["px_per_bit"]) for r in small)
            lines.append(f"  Fix: the board IS reaching {', '.join(r['cam'] for r in small)} and is "
                         f"too small to DECODE there. Print it ~{need:.1f}x larger, or hold it that "
                         f"much closer to them; re-sweeping the same board will not help.")
        else:
            lines.append("  Fix: re-record with the board presented to EVERY camera, and to camera "
                         "PAIRS at the same instant.")
    return lines


def write_report(df: pd.DataFrame, cal_dir, step: int, out_dir=None) -> tuple[Path, Path]:
    out = Path(out_dir or cal_dir)
    writeguard.assert_writable(out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"charuco_survey_{Path(cal_dir).name}"
    csv_path, txt_path = out / f"{stem}.csv", out / f"{stem}.txt"
    df.to_csv(csv_path, index=False)
    txt_path.write_text("\n".join(report_lines(df, cal_dir, step)) + "\n", encoding="utf-8")
    return csv_path, txt_path


def run(cal_dir=None, step: int = 25, out_dir=None, machine=None) -> pd.DataFrame:
    cal_dir = Path(cal_dir) if cal_dir else find_calibration_dir(machine=machine)
    print(f"[dlc_calibration] surveying {cal_dir} (every {step}th frame)", flush=True)
    df = survey(cal_dir, step)
    csv_path, _ = write_report(df, cal_dir, step, out_dir)
    print("\n".join(report_lines(df, cal_dir, step)), flush=True)
    print(f"[dlc_calibration] -> {csv_path}", flush=True)
    return df


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=None, help="calibration recording dir (default: newest by name)")
    ap.add_argument("--step", type=int, default=25, help="sample every Nth frame (default 25 = 10 Hz)")
    ap.add_argument("--output", default=None, help="report output dir (default: the recording dir)")
    ap.add_argument("--machine", default=None)
    args = ap.parse_args(argv)
    run(args.dir, args.step, args.output, args.machine)
    # Always 0: a disconnected graph is a REPORTED finding about the recording, not a failure of
    # this step, and the nightly routes nonzero exits into FAILURES (which blocks deck publication).
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
