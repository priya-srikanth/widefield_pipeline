"""Solve the four-camera calibration from a ChArUco recording: intrinsics, extrinsics, error.

``dlc_calibration`` answers "is this recording usable?". This answers "what are the cameras?" --
per-camera intrinsics (focal length, principal point, distortion) and a rigid transform placing
every camera in ONE coordinate frame, which is what triangulating DLC keypoints needs.

**CORRECTION (2026-09-10): aniposelib DOES work here, and this module was written on a claim that
was wrong.** The claim was that aniposelib calls ``cv2.aruco.interpolateCornersCharuco`` and
``calibrateCameraCharuco``, removed in OpenCV 4.8+ — true of aniposelib 0.4-0.5, and asserted here
without checking the current release. **aniposelib 0.8.0 uses the modern ``CharucoDetector`` /
``ArucoDetector`` API**; tested against this environment's cv2 4.11 it builds a board, renders it and
detects 25/25 corners.

What that means for this module. Its DIAGNOSTICS earned their place: reporting corners per camera is
what exposed that the 09-10 recording cannot calibrate the snout cameras, and the marker-based gate
in ``dlc_calibration`` was passing it. Its SOLVE is the weaker half — it chains pairwise extrinsics
over a spanning tree, where aniposelib does joint bundle adjustment over all cameras at once, and
aniposelib also carries the triangulation this is all for. Prefer aniposelib for the solve; keep
this for the diagnostics and as an independent cross-check, which is worth having on a quantity no
one can eyeball.

Installing it needs the labelling GUI closed (it holds ``cv2.pyd`` open) and pulls
``opencv-contrib-python`` over ``opencv-python``.

**The reported number that matters is REPROJECTION ERROR**, in pixels. The pose counts in
``dlc_calibration`` are a screening gate -- they say a solve is worth attempting, not that it is
good. A calibration with plenty of poses and 3 px of reprojection error is a bad calibration, and
only this step can tell you.

**Extrinsics are chained over a SPANNING TREE, not solved pairwise and averaged.** Every pair that
clears the gate gives a relative pose, but they are not mutually consistent in the presence of
noise; composing them around a loop does not return the identity. Taking a tree rooted at the
best-observed camera makes the choice explicit and reproducible instead of letting an averaging
scheme hide it. The loop-closure residual is REPORTED, because it is the honest measure of how much
the pairwise estimates disagree.

**The board geometry comes from ``configs/defaults.yaml dlc.board``** and is the metric scale of
everything downstream: get ``square_mm`` wrong and every reconstructed distance is wrong by that
factor, with nothing in the output to reveal it.

CLI (from the ``dlc`` env, or any env with OpenCV >= 4.7)::

    python -m wfield_local.dlc_calibrate                    # newest calibration recording
    python -m wfield_local.dlc_calibrate --dir <path>
    python -m wfield_local.dlc_calibrate --step 25          # frame sampling for detection
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from wfield_local import config
from wfield_local.dlc_calibration import (
    ARUCO_DICT,
    CAM_RE,
    POSE_GAP_S,
    count_poses,
    find_calibration_dir,
)
from wfield_local.writeguard import assert_writable

#: ChArUco corners needed in a frame before its board pose is used. Four is the algebraic minimum
#: for a homography; six leaves the pose over-determined, which is what keeps a noisy detection from
#: dragging the intrinsics.
MIN_CORNERS = 6

#: Frames per camera fed to `calibrateCamera`. Beyond this the solution stops moving and the runtime
#: does not; the constraint is distinct POSES, and a recording rarely holds more than ~100.
MAX_CALIB_FRAMES = 120


def _make_board(spec):
    from cv2 import aruco
    d = aruco.getPredefinedDictionary(getattr(aruco, spec["dictionary"]))
    return aruco.CharucoBoard((spec["squares_x"], spec["squares_y"]),
                              spec["square_mm"], spec["marker_mm"], d)


def board_from_config():
    """The ChArUco board described by ``dlc.board`` -- the fallback when a recording has no sidecar."""
    b = (config.defaults().get("dlc") or {}).get("board") or {}
    spec = {
        "squares_x": int(b.get("squares_x", 6)), "squares_y": int(b.get("squares_y", 6)),
        "square_mm": float(b.get("square_mm", 6.67)), "marker_mm": float(b.get("marker_mm", 5.07)),
        "dictionary": str(b.get("dictionary", ARUCO_DICT)),
    }
    return _make_board(spec), spec


def board_for(cal_dir):
    """The board used for ONE recording: ``board.yaml`` beside it, else ``dlc.board``.

    PER RECORDING, not global, and this is what makes combining recordings possible at all. Two
    boards can share a dictionary -- ours both use DICT_4X4_50 with overlapping marker ids -- so
    detecting a recording with the WRONG geometry does not error: it interpolates corners at
    positions that belong to a different board and returns them confidently. The only defence is
    that each recording carries its own spec, which is also why `dlc_board` prints the spec onto the
    board itself.
    """
    import yaml

    side = Path(cal_dir) / "board.yaml"
    if side.exists():
        b = yaml.safe_load(side.read_text(encoding="utf-8")) or {}
        spec = {"squares_x": int(b["squares_x"]), "squares_y": int(b["squares_y"]),
                "square_mm": float(b["square_mm"]), "marker_mm": float(b["marker_mm"]),
                "dictionary": str(b.get("dictionary", ARUCO_DICT))}
        return _make_board(spec), spec
    return board_from_config()


def write_board_sidecar(cal_dir, spec) -> Path:
    """Record which board a recording used, beside the recording."""
    import yaml

    side = Path(cal_dir) / "board.yaml"
    assert_writable(side.parent)
    side.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    return side


def detect(cal_dir, step: int = 25, pattern: str = "cam*.avi"):
    """ChArUco corners per camera per sampled frame.

    Returns ``(cams, per_cam, size)`` where ``per_cam[cam][frame] = (corners, ids)``.

    All cameras are sampled on the SAME frame grid, because the pairs are formed by frame index --
    sound only because the rig's four cameras are hardware-synced off one Arduino heartbeat and no
    calibration recording has dropped a frame (``dropframe_qc``).
    """
    import cv2
    from cv2 import aruco

    board, _ = board_for(cal_dir)
    det = aruco.CharucoDetector(board)
    vids = sorted(Path(cal_dir).glob(pattern))
    if not vids:
        raise FileNotFoundError(f"No {pattern} under {cal_dir}")

    per_cam: dict[str, dict[int, tuple]] = {}
    size: dict[str, tuple[int, int]] = {}
    for v in vids:
        m = CAM_RE.match(v.name)
        cam = m.group(1).lower() if m else v.stem
        cap = cv2.VideoCapture(str(v))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open {v}")
        found, i = {}, 0
        try:
            while cap.grab():
                if i % step == 0:
                    ok, frame = cap.retrieve()
                    if ok:
                        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        size[cam] = (grey.shape[1], grey.shape[0])
                        cc, ci, _, _ = det.detectBoard(grey)
                        if cc is not None and len(cc) >= MIN_CORNERS:
                            found[i] = (cc, ci)
                i += 1
        finally:
            cap.release()
        per_cam[cam] = found
        print(f"[dlc_calibrate] {cam}: {len(found)} frames with >={MIN_CORNERS} charuco corners "
              f"({count_poses(found)} poses)", flush=True)
    return sorted(per_cam), per_cam, size


def _spread(frames, k: int, fps: float = 250.0):
    """Up to ``k`` frames chosen one per pose, so the selection is DIVERSE rather than dense.

    Taking the first k, or k at random, over-weights whichever moment the board lingered in; a
    calibration is constrained by distinct views and duplicates of one view add runtime, not
    information.
    """
    f = sorted(frames)
    if not f:
        return []
    picks, last = [f[0]], f[0]
    for x in f[1:]:
        if (x - last) / fps > POSE_GAP_S:
            picks.append(x)
            last = x
    if len(picks) > k:
        picks = [picks[i] for i in np.linspace(0, len(picks) - 1, k).round().astype(int)]
    return picks


def intrinsics(cam, per_cam, size, board):
    """``(K, dist, rms, n)`` for one camera from its ChArUco detections."""
    import cv2

    frames = _spread(per_cam[cam], MAX_CALIB_FRAMES)
    obj, img = [], []
    for f in frames:
        cc, ci = per_cam[cam][f]
        op, ip = board.matchImagePoints(cc, ci)
        if op is not None and len(op) >= MIN_CORNERS:
            obj.append(op)
            img.append(ip)
    if len(obj) < 5:
        raise RuntimeError(f"{cam}: only {len(obj)} usable views; cannot solve intrinsics")
    rms, K, dist, _, _ = cv2.calibrateCamera(obj, img, size[cam], None, None)
    return K, dist, float(rms), len(obj)


def stereo(a, b, per_cam, size, board, K, dist):
    """``(R, T, rms, n)`` placing camera ``b`` relative to ``a``, with intrinsics held FIXED.

    Fixed because they were already solved from far more views than this pair shares; letting
    `stereoCalibrate` refit them here would let a thin pair's noise rewrite a well-constrained
    focal length.
    """
    import cv2

    shared = sorted(set(per_cam[a]) & set(per_cam[b]))
    obj, ia, ib = [], [], []
    for f in _spread(shared, MAX_CALIB_FRAMES):
        ca, ida = per_cam[a][f]
        cb, idb = per_cam[b][f]
        common = np.intersect1d(ida.ravel(), idb.ravel())
        if len(common) < MIN_CORNERS:
            continue
        sel_a = np.isin(ida.ravel(), common)
        sel_b = np.isin(idb.ravel(), common)
        op, pa = board.matchImagePoints(ca[sel_a], ida[sel_a])
        _, pb = board.matchImagePoints(cb[sel_b], idb[sel_b])
        if op is None or pa is None or pb is None or len(op) < MIN_CORNERS:
            continue
        obj.append(op)
        ia.append(pa)
        ib.append(pb)
    if len(obj) < 4:
        return None, None, float("nan"), len(obj)
    rms, *_, R, T, _, _ = cv2.stereoCalibrate(
        obj, ia, ib, K[a], dist[a], K[b], dist[b], size[a],
        flags=cv2.CALIB_FIX_INTRINSIC)
    return R, T, float(rms), len(obj)


def spanning_tree(pairs: dict, root: str, cams):
    """Camera -> (parent, R, T) over the best-observed edges. Returns the tree and its edge order.

    Edges are taken in DESCENDING shared-view count, so every camera is attached through the
    strongest path available to it rather than whichever edge happened to be visited first.
    """
    order = sorted(pairs, key=lambda k: -pairs[k][3])
    placed = {root}
    tree: dict[str, tuple] = {}
    used = []
    progress = True
    while progress and len(placed) < len(cams):
        progress = False
        for (a, b) in order:
            R, T, _rms, n = pairs[(a, b)]
            if R is None or (a in placed) == (b in placed):
                continue
            if a in placed:
                tree[b] = (a, R, T)
            else:                                   # invert: we have b relative to a
                tree[a] = (b, R.T, -R.T @ T)
            placed.add(b if a in placed else a)
            used.append((a, b, n))
            progress = True
    return tree, used


def to_world(tree: dict, root: str, cams):
    """Compose each camera's transform to the root frame."""
    RT = {root: (np.eye(3), np.zeros((3, 1)))}
    for _ in range(len(cams)):
        for cam, (parent, R, T) in tree.items():
            if cam in RT or parent not in RT:
                continue
            Rp, Tp = RT[parent]
            RT[cam] = (R @ Rp, R @ Tp + T)
    return RT


def loop_residuals(pairs: dict, RT: dict) -> list[tuple[str, str, float, float]]:
    """For every pair NOT in the tree, how far the chained transform is from the measured one.

    The honest measure of how much the pairwise estimates disagree. A tree uses n-1 of the edges and
    silently discards the rest; if a discarded edge says something very different, the calibration
    is internally inconsistent no matter how good each pair's own reprojection error looked.
    """
    out = []
    for (a, b), (R, T, _rms, _n) in pairs.items():
        if R is None or a not in RT or b not in RT:
            continue
        Ra, Ta = RT[a]
        Rb, Tb = RT[b]
        R_chain = Rb @ Ra.T
        T_chain = Tb - R_chain @ Ta
        ang = float(np.degrees(np.arccos(np.clip((np.trace(R_chain @ R.T) - 1) / 2, -1, 1))))
        dt = float(np.linalg.norm(T_chain - T))
        out.append((a, b, round(ang, 2), round(dt, 2)))
    return sorted(out, key=lambda r: -r[2])


def calibrate(cal_dir=None, step: int = 25, out_dir=None, machine=None) -> dict:
    cal_dir = Path(cal_dir) if cal_dir else find_calibration_dir(machine=machine)
    board, spec = board_from_config()
    print(f"[dlc_calibrate] {cal_dir}", flush=True)
    print(f"[dlc_calibrate] board {spec['squares_x']}x{spec['squares_y']}, square "
          f"{spec['square_mm']} mm, marker {spec['marker_mm']} mm, {spec['dictionary']}", flush=True)

    cams, per_cam, size = detect(cal_dir, step)

    K, dist, cam_rms = {}, {}, {}
    print("\n[dlc_calibrate] intrinsics", flush=True)
    for cam in cams:
        K[cam], dist[cam], rms, n = intrinsics(cam, per_cam, size, board)
        cam_rms[cam] = rms
        f = (K[cam][0, 0] + K[cam][1, 1]) / 2
        print(f"    {cam}: {n:3d} views, f={f:7.1f} px, reprojection RMS {rms:.3f} px", flush=True)

    print("\n[dlc_calibrate] pairwise extrinsics", flush=True)
    pairs = {}
    for i, a in enumerate(cams):
        for b in cams[i + 1:]:
            R, T, rms, n = stereo(a, b, per_cam, size, board, K, dist)
            pairs[(a, b)] = (R, T, rms, n)
            base = f"    {a}-{b}: {n:3d} shared views"
            print(f"{base}, RMS {rms:.3f} px" if R is not None else f"{base} -- TOO FEW, skipped",
                  flush=True)

    root = max(cams, key=lambda c: len(per_cam[c]))
    tree, used = spanning_tree(pairs, root, cams)
    RT = to_world(tree, root, cams)
    if len(RT) < len(cams):
        missing = [c for c in cams if c not in RT]
        raise RuntimeError(f"cameras not reachable from {root}: {missing}")
    print(f"\n[dlc_calibrate] root {root}; tree edges "
          f"{', '.join(f'{a}-{b}({n})' for a, b, n in used)}", flush=True)

    resid = loop_residuals(pairs, RT)
    print("[dlc_calibrate] loop-closure residual on the edges the tree did NOT use:", flush=True)
    for a, b, ang, dt in resid:
        used_edge = any({a, b} == {x, y} for x, y, _ in used)
        print(f"    {a}-{b}: {ang:6.2f} deg, {dt:7.2f} mm{'   (tree edge)' if used_edge else ''}",
              flush=True)

    out = {"recording": Path(cal_dir).name, "board": spec, "root": root,
           "cameras": {c: {"size": list(size[c]), "matrix": K[c].tolist(),
                           "distortions": dist[c].ravel().tolist(),
                           "rotation": RT[c][0].tolist(), "translation": RT[c][1].ravel().tolist(),
                           "intrinsic_rms_px": cam_rms[c], "n_views": len(per_cam[c])}
                       for c in cams},
           "pairs": {f"{a}-{b}": {"rms_px": pairs[(a, b)][2], "n": pairs[(a, b)][3]}
                     for (a, b) in pairs},
           "loop_residuals": [{"a": a, "b": b, "deg": d, "mm": t} for a, b, d, t in resid]}
    write(out, out_dir or cal_dir)
    return out


def write(out: dict, dest) -> list[Path]:
    """Write the calibration as JSON and as anipose-style TOML."""
    dest = Path(dest)
    assert_writable(dest)
    dest.mkdir(parents=True, exist_ok=True)
    j = dest / "calibration.json"
    j.write_text(json.dumps(out, indent=2), encoding="utf-8")

    b = out["board"]
    lines = [f"# {out['recording']}  root={out['root']}",
             (f"# board {b['squares_x']}x{b['squares_y']} square={b['square_mm']}mm "
              f"marker={b['marker_mm']}mm"), ""]
    for cam, c in out["cameras"].items():
        import cv2
        rvec = cv2.Rodrigues(np.asarray(c["rotation"]))[0].ravel().tolist()
        lines += [f"[cam_{cam[-1]}]", f'name = "{cam}"', f"size = {c['size']}",
                  f"matrix = {c['matrix']}", f"distortions = {c['distortions']}",
                  f"rotation = {[round(v, 8) for v in rvec]}",
                  f"translation = {[round(v, 6) for v in c['translation']]}", ""]
    t = dest / "calibration.toml"
    t.write_text("\n".join(lines), encoding="utf-8")
    print(f"[dlc_calibrate] -> {j}\n[dlc_calibrate] -> {t}", flush=True)
    return [j, t]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=None, help="calibration recording (default: newest)")
    ap.add_argument("--step", type=int, default=25, help="sample every Nth frame for detection")
    ap.add_argument("--output", default=None, help="where to write (default: the recording dir)")
    ap.add_argument("--machine", default=None)
    args = ap.parse_args(argv)
    calibrate(args.dir, args.step, args.output, args.machine)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
