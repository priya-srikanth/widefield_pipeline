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

Measured on ``camera_calibration_20260805`` (2026-09-08), sampling every 25th frame — 10 Hz over
192 s, 1926 samples per camera:

    cam1  any=88.5%  >=4 markers=65.5%      cam1-cam4 both>=4: 1225 frames   OK
    cam2  any=20.1%  >=4 markers= 0.6%      cam1-cam2 both>=4:   11 frames   too few
    cam3  any=12.3%  >=4 markers= 0.2%      cam1-cam3 both>=4:    3 frames   too few
    cam4  any=95.0%  >=4 markers=93.0%      cam2-cam3 both>=4:    0 frames   disconnected

That recording therefore calibrates the cam1-cam4 snout pair and nothing else: the board was waved
in front of the two snout cameras and never presented systematically to the side views, which
additionally see it small and oblique (600x450 across a much wider FOV, so the 4x4 markers fall
below reliable detection size). The side cameras are the ones that see the eye, so this is not a
corner worth cutting.

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

#: Per-camera usable frames needed for stable intrinsics. Well below what anipose suggests for a
#: careful job (~200) because this is a SCREENING threshold: under it the recording is certainly
#: unusable, over it is worth attempting.
MIN_CAM_FRAMES = 100

#: Simultaneous detections needed on a camera PAIR before its relative pose is worth estimating.
#: Deliberately lenient for the same reason.
MIN_PAIR_FRAMES = 50

CAM_RE = re.compile(r"^(cam\d+)_", re.IGNORECASE)
CAL_DIR_RE = re.compile(r"^camera_calibration_(\d{8})$", re.IGNORECASE)


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
        m = CAL_DIR_RE.match(p.name)
        if m and p.is_dir():
            dirs.append((m.group(1), p))
    if not dirs:
        raise FileNotFoundError(f"No camera_calibration_<YYYYMMDD>/ directory under {root}")
    return max(dirs)[1]


def _detector():
    from cv2 import aruco
    params = aruco.DetectorParameters()
    # The board is often small and motion-blurred in the wide side views; widening the adaptive
    # threshold window is what lets those frames contribute at all.
    params.adaptiveThreshWinSizeMax = 45
    return aruco.ArucoDetector(aruco.getPredefinedDictionary(getattr(aruco, ARUCO_DICT)), params)


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
                    _, ids, _ = det.detectMarkers(grey)
                    n = 0 if ids is None else len(ids)
                    id_str = "" if ids is None else " ".join(str(int(x)) for x in sorted(ids.ravel()))
                    rows.append({"cam": cam, "frame": i, "n_markers": n, "ids": id_str})
            i += 1
    finally:
        cap.release()
    return pd.DataFrame(rows, columns=["cam", "frame", "n_markers", "ids"])


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
        out.append({
            "cam": cam,
            "sampled": len(g),
            "any_pct": round(float((n > 0).mean() * 100), 1),
            "usable": usable,
            "usable_pct": round(float((n >= MIN_MARKERS).mean() * 100), 1),
            "max_markers": int(n.max()) if len(n) else 0,
            "ok": bool(usable >= MIN_CAM_FRAMES),
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
        both = int(((piv[a] >= MIN_MARKERS) & (piv[b] >= MIN_MARKERS)).sum())
        out.append({"cam_a": a, "cam_b": b, "both": both, "ok": bool(both >= MIN_PAIR_FRAMES)})
    return pd.DataFrame(out, columns=["cam_a", "cam_b", "both", "ok"])


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
             f"{'cam':6s} {'sampled':>8s} {'any':>7s} {'usable':>8s} {'usable%':>8s} {'max':>5s}  verdict"]
    for _, r in cams.iterrows():
        verdict = "OK" if r["ok"] else f"TOO FEW (<{MIN_CAM_FRAMES})"
        lines.append(f"{r['cam']:6s} {r['sampled']:8d} {r['any_pct']:6.1f}% {r['usable']:8d} "
                     f"{r['usable_pct']:7.1f}% {r['max_markers']:5d}  {verdict}")
    lines += ["", f"{'pair':14s} {'both':>7s}  verdict"]
    for _, r in pairs.iterrows():
        verdict = "OK" if r["ok"] else f"TOO FEW (<{MIN_PAIR_FRAMES})"
        lines.append(f"{r['cam_a']}-{r['cam_b']:9s} {r['both']:7d}  {verdict}")
    lines.append("")
    if ok:
        lines.append("RESULT: pair graph CONNECTED - all cameras can be placed in one frame.")
    else:
        lines.append("RESULT: pair graph NOT CONNECTED - 3D reconstruction is impossible from this "
                     "recording.")
        for comp in comps:
            lines.append(f"  component: {', '.join(comp)}")
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
