"""Pooled multi-recording camera calibration via aniposelib's joint bundle adjustment.

``dlc_calibration`` asks "is this recording usable?"; ``dlc_calibrate`` solves ONE recording by
chaining pairwise extrinsics over a spanning tree. This solves SEVERAL recordings at once with
aniposelib, which optimises every camera together instead of composing pairs, and which carries the
triangulation the whole exercise is for.

Why pooling is needed at all: **no single board size works for all four cameras on this rig.** cam1
and cam4 are snout views at ~25 px/mm; cam2 and cam3 are whole-animal views at ~11 px/mm. A board
big enough for the side views to DECODE overflows the snout views, and a board small enough to fit
the snout views falls below the ~3 px-per-code-cell floor on the side views. Two recordings with two
boards is the obvious way out -- hence this module.

**THE TWO HALVES POOL ON COMPLETELY DIFFERENT TERMS, and conflating them is the trap.**

*Intrinsics pool freely.* ``cv2.calibrateCamera`` takes ``objectPoints`` PER VIEW, so views of a 40 mm
board and views of a 26 mm board sit side by side in one solve with no special handling. The only
assumption is that the optics did not change between recordings -- nobody refocused or swapped a
lens. Each camera is solved independently, so a recording that is useless for one camera is simply
absent from that camera's view list rather than harmful to it.

*Extrinsics do not pool without an extra assumption: THE RIG DID NOT MOVE.* A relative pose between
two cameras is a fact about where the cameras were on the day, and pooling two recordings asserts
they were in the same place on both days. That is not checkable from within a single recording, so
this module **measures it** (`cross_recording_agreement`) using any pair observed well in more than
one recording: solve that pair separately per recording and report the disagreement in degrees and
mm. **When no pair is observed twice, the assumption is UNTESTABLE and pooled extrinsics are refused
rather than silently taken** -- an untested rigid-rig assumption is the kind of error that produces a
confident, wrong 3D reconstruction with nothing in the output to reveal it.

**aniposelib's thresholds are stricter than ours and they are what actually decide.**
``CameraGroup.calibrate_rows`` keeps a view for the intrinsics initialisation only at
``min_corners_intrinsic=9`` ChArUco corners, and admits a row to the bundle adjustment only at
``ids.size >= 8``. ``dlc_calibrate`` used 6. A camera clearing our gate and failing aniposelib's is
not a contradiction -- it is why `gate()` below reports aniposelib's numbers, not ours.

**aniposelib's ChArUco detector parameters are overridden here.** It ships
``adaptiveThreshWinSizeMin=50 .. Max=700, step 50`` (``boards.py``), tuned for a board filling a
large frame. Our frames are 600x600 and 600x450 and the markers are tens of pixels, so a 50 px
minimum threshold window starts above the feature size. `_tuned_board` restores a window ladder that
fits this rig. The board GEOMETRY is untouched; only the detector is.

**Detections are cached beside the recording** (``anipose_rows_<digest>.pkl``). Decoding four
~650 MB videos off the share takes minutes and the solve takes milliseconds, so a re-solve with
different pooling options should not re-decode. The digest covers the board spec, the frame step and
the detector settings, so changing any of them invalidates the cache rather than silently reusing it.

CLI (from the ``dlc`` env)::

    python -m wfield_local.dlc_anipose --gate                 # what aniposelib will accept, no solve
    python -m wfield_local.dlc_anipose                        # solve the newest recording
    python -m wfield_local.dlc_anipose --dir A --dir B        # pool intrinsics across two recordings
    python -m wfield_local.dlc_anipose --dir A --dir B --assume-rig-unmoved   # ... and extrinsics
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

from wfield_local.dlc_calibration import (
    CAM_RE,
    POSE_GAP_S,
    count_poses,
    find_calibration_dir,
)
from wfield_local.writeguard import assert_writable

#: ChArUco corners per view for aniposelib's intrinsics initialisation
#: (``CameraGroup.calibrate_rows(min_corners_intrinsic=...)``). Mirrored here so the gate can report
#: the threshold that will actually be applied instead of one of our own choosing.
ANIPOSE_MIN_CORNERS_INTRINSIC = 9

#: ChArUco corners for a row to enter the bundle adjustment (``calibrate_rows``: ``ids.size >= 8``).
ANIPOSE_MIN_CORNERS_BUNDLE = 8

#: Distinct board POSES per camera before intrinsics are worth solving -- see ``dlc_calibration``:
#: frames are not poses, and a board held still for two seconds at 250 fps is 500 frames and one view.
MIN_POSES = 20

#: Views handed to one ``calibrateCamera`` call. Past this the solution stops moving and the runtime
#: does not.
MAX_VIEWS = 150

#: Rotation / translation disagreement between two recordings' independent estimates of the SAME
#: camera pair, past which the rig cannot be treated as unmoved. 1 degree at a 100 mm working
#: distance is ~1.7 mm of transverse error -- already larger than the keypoint precision that 3D
#: tracking is trying to achieve, so there is no point pooling beyond it.
MAX_RIG_DRIFT_DEG = 1.0
MAX_RIG_DRIFT_MM = 2.0

#: THE CHECK REPROJECTION ERROR CANNOT MAKE, and the one that caught the 2026-09-10 recording.
#: Solved with a free principal point, THREE of the four cameras put it outside their own sensor --
#: cam1 at (1354, 402) on a 600x600 frame, 1.8 image-widths off centre -- while reporting 0.9-1.9 px
#: reprojection error. The model fits; it is simply not identifiable, because every board pose in the
#: recording sits at a similar depth and tilt, and focal length, principal point and distortion then
#: trade off against one another along a valley the residual cannot see. A calibration that passes
#: on RMS and fails here is worse than no calibration: it triangulates confidently and wrongly.
MAX_PP_OFFSET = 0.2

#: Radial distortion past which the "calibration" is absorbing the degeneracy above rather than
#: describing a lens. These are machine-vision C-mount lenses; |k1| is a few tenths. The marker-corner
#: fallback on cam1 returns -4.05 with the principal point pinned, which is the same non-identifiability
#: reappearing in the only free parameter left to it.
MAX_ABS_K1 = 1.0

#: THE SHARPEST CHECK OF THE THREE, because it tests against a fact about the SENSOR rather than a
#: plausible range. Blackfly pixels are square, so fx and fy are the same focal length measured in the
#: same units and must agree to a fraction of a percent. Pinning the principal point on the 09-10
#: recording still left cam2 at fx=1458, fy=1617 -- a 10% non-square pixel, which no camera on this
#: rig has. The degeneracy simply moved again: constrain the principal point and it reappears in the
#: aspect ratio, constrain that too and it goes to distortion (cam3's k2 came back at -13.4).
#: The only real fix is at the rig -- sweep the board through depth.
MAX_ASPECT_ERROR = 0.05


def _tuned_board(spec):
    """An aniposelib ``CharucoBoard`` with detector parameters that fit a 600 px frame.

    aniposelib's own ladder starts at a 50 px adaptive-threshold window, which is larger than a
    marker in every view on this rig. Geometry is untouched -- only the detector.
    """
    import cv2
    from aniposelib.boards import CharucoBoard

    b = CharucoBoard(spec["squares_x"], spec["squares_y"],
                     square_length=spec["square_mm"], marker_length=spec["marker_mm"],
                     marker_bits=int(str(spec["dictionary"]).split("_")[1][0]),
                     dict_size=int(str(spec["dictionary"]).split("_")[-1]))
    p = b.detector_params
    p.adaptiveThreshWinSizeMin = 3
    p.adaptiveThreshWinSizeMax = 45
    p.adaptiveThreshWinSizeStep = 6
    p.adaptiveThreshConstant = 7
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    b.detector = cv2.aruco.ArucoDetector(b.dictionary, p)
    b.charuco_detector = cv2.aruco.CharucoDetector(b.board)
    b.charuco_detector.setDetectorParameters(p)
    return b


def _detect_video(board, path, step: int):
    """aniposelib's DETECTION, walked with ``grab()`` so sampled frames are the only ones decoded.

    ``CharucoBoard.detect_video`` does ``ret, frame = cap.read()`` for every frame and then
    ``continue``s on the ones it is skipping -- a full decode of ~83,000 frames per video to detect
    on a tenth of them. Over the share that is the difference between minutes and an hour per
    recording, and it buys nothing: ``grab()`` advances the decoder without producing a frame.

    Its ``go`` behaviour is deliberately NOT kept. aniposelib re-arms a ``step/2`` run of dense
    detection after every hit, so on a recording where most frames carry a board -- 80% of cam1's
    here -- the run never expires and it detects on essentially every frame regardless of ``skip``.
    Everything downstream selects one row per POSE, so those dense runs are decoded, detected and
    then discarded. Sampling strictly every ``step``th frame gives the same poses.

    Rows come out in aniposelib's format (``framenum``/``corners``/``ids``/``filled``), so nothing
    downstream can tell the difference.
    """
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {path}")
    rows, i = [], 0
    try:
        while cap.grab():
            if i % step == 0:
                ok, frame = cap.retrieve()
                if ok:
                    corners, ids = board.detect_image(frame)
                    if corners is not None and len(corners) > 0:
                        rows.append({"framenum": i, "corners": corners, "ids": ids})
            i += 1
    finally:
        cap.release()
    return board.fill_points_rows(rows)


def _digest(spec, step) -> str:
    """Cache key over everything that changes what detection returns."""
    payload = json.dumps({"spec": spec, "step": step, "detector": "tuned-v1"}, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:10]


def detect_rows(cal_dir, step: int = 10, pattern: str = "cam*.avi", refresh: bool = False):
    """``(cams, rows_by_cam, sizes, spec)`` for one recording, cached beside it.

    Rows are aniposelib's own format (``framenum``/``corners``/``ids``/``filled``), produced by its
    own detector, so nothing is lost in translation on the way into ``calibrate_rows``.
    """
    import cv2

    from wfield_local.dlc_calibrate import board_for

    cal_dir = Path(cal_dir)
    _, spec = board_for(cal_dir)
    if not (cal_dir / "board.yaml").exists():
        # `board_for` falls back to `dlc.board` in config, which is the right default and the wrong
        # thing to do SILENTLY: record with the 26 mm board, forget the sidecar, and the config still
        # says 40 mm -- same layout, same dictionary, same corner ids, every reconstructed distance
        # wrong by 1.54x with nothing in the output to reveal it. This is the failure that cost us
        # the 2026-08-05 recording outright.
        print(f"[dlc_anipose] WARNING: {cal_dir.name} has no board.yaml; assuming the board in "
              f"configs/defaults.yaml dlc.board ({spec['squares_x']}x{spec['squares_y']}, square "
              f"{spec['square_mm']} mm). If that is not the board in these videos, every metric "
              f"result is wrong by the ratio and nothing downstream can tell.", flush=True)
    cache = cal_dir / f"anipose_rows_{_digest(spec, step)}.pkl"
    if cache.exists() and not refresh:
        d = pickle.loads(cache.read_bytes())
        print(f"[dlc_anipose] {cal_dir.name}: cached detections ({cache.name})", flush=True)
        return d["cams"], d["rows"], d["sizes"], spec

    if not spec.get("squares_x") or not spec.get("square_mm"):
        raise ValueError(
            f"{cal_dir.name}: board.yaml has no usable geometry ({spec}). A ChArUco corner is "
            f"defined only relative to a known square layout, so nothing can be detected without "
            f"it, and square_mm is the metric scale of every 3D distance downstream.")

    vids = sorted(cal_dir.glob(pattern))
    if not vids:
        # An empty result is a worse outcome than an error: it reads downstream as "every camera is
        # unusable", which is the same shape as a genuine finding about a real recording.
        raise FileNotFoundError(f"No {pattern} under {cal_dir}")

    board = _tuned_board(spec)
    rows, sizes = {}, {}
    for vid in vids:
        m = CAM_RE.match(vid.name)
        cam = m.group(1).lower() if m else vid.stem
        cap = cv2.VideoCapture(str(vid))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open {vid}")
        try:
            sizes[cam] = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                          int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        finally:
            cap.release()
        rows[cam] = _detect_video(board, vid, step)
        mx = max((len(r["ids"]) for r in rows[cam] if r["ids"] is not None), default=0)
        print(f"[dlc_anipose] {cal_dir.name} {cam}: {len(rows[cam])} rows with any corner, "
              f"max {mx} corners", flush=True)

    assert_writable(cal_dir)
    cache.write_bytes(pickle.dumps({"cams": sorted(rows), "rows": rows, "sizes": sizes,
                                    "spec": spec, "step": step}))
    return sorted(rows), rows, sizes, spec


def _framenum(row) -> int:
    """Frame index out of aniposelib's ``framenum``, which is ``(prefix, n)`` when prefixed."""
    f = row["framenum"]
    return int(f[1]) if isinstance(f, (tuple, list)) else int(f)


def pose_counts(rows, min_corners: int, fps: float = 250.0) -> int:
    """Distinct board poses among rows clearing ``min_corners`` -- the quantity a solve consumes."""
    keep = [_framenum(r) for r in rows if r["ids"] is not None and len(r["ids"]) >= min_corners]
    return count_poses(keep, fps=fps)


def gate(recordings) -> dict:
    """What aniposelib will accept, per camera, per recording -- reported BEFORE any solve.

    Separates the two ways a camera fails, which have opposite fixes: too few CORNERS per view means
    the board is too large in that view (a fragment carries markers but no interpolated corners);
    enough corners but too few POSES means the board was held rather than swept.
    """
    out = {}
    for cal_dir, (cams, rows, _sizes, spec) in recordings.items():
        per = {}
        for cam in cams:
            r = rows[cam]
            mx = max((len(x["ids"]) for x in r if x["ids"] is not None), default=0)
            per[cam] = {
                "rows": len(r),
                "max_corners": int(mx),
                "poses_bundle": pose_counts(r, ANIPOSE_MIN_CORNERS_BUNDLE),
                "poses_intrinsic": pose_counts(r, ANIPOSE_MIN_CORNERS_INTRINSIC),
            }
            p = per[cam]
            if p["max_corners"] < ANIPOSE_MIN_CORNERS_BUNDLE:
                p["verdict"] = (f"UNUSABLE: never more than {p['max_corners']} ChArUco corners, "
                                f"aniposelib needs {ANIPOSE_MIN_CORNERS_BUNDLE}. The board is too "
                                f"LARGE in this view -- a fragment of it carries markers but no "
                                f"interpolated corners.")
            elif p["poses_intrinsic"] < MIN_POSES:
                p["verdict"] = (f"WEAK: {p['poses_intrinsic']} poses with "
                                f">={ANIPOSE_MIN_CORNERS_INTRINSIC} corners, want {MIN_POSES}")
            else:
                p["verdict"] = "OK"
        out[str(cal_dir)] = {"board": spec, "cameras": per}
    return out


def intrinsics_quality(K, dist, size) -> dict:
    """Is this K IDENTIFIABLE, as distinct from well-fitting? See ``MAX_PP_OFFSET``.

    Reported alongside every solve because reprojection error cannot distinguish a calibration from
    a curve that happens to pass through the points.
    """
    cx, cy = float(K[0, 2]), float(K[1, 2])
    fx, fy = float(K[0, 0]), float(K[1, 1])
    off = float(np.hypot(cx - size[0] / 2, cy - size[1] / 2) / max(size))
    k1 = float(np.ravel(dist)[0])
    aspect = abs(fx - fy) / max(fx, fy)
    inside = 0 <= cx < size[0] and 0 <= cy < size[1]
    problems = []
    if not inside:
        problems.append(f"principal point ({cx:.0f},{cy:.0f}) is OUTSIDE the {size[0]}x{size[1]} frame")
    elif off > MAX_PP_OFFSET:
        problems.append(f"principal point is {off:.2f} image-widths off centre (max {MAX_PP_OFFSET})")
    if aspect > MAX_ASPECT_ERROR:
        problems.append(f"fx={fx:.0f} and fy={fy:.0f} differ by {aspect * 100:.1f}%, i.e. a "
                        f"non-square pixel -- these sensors have square pixels")
    if abs(k1) > MAX_ABS_K1:
        problems.append(f"k1={k1:+.2f} is not a lens (max |k1| {MAX_ABS_K1})")
    return {"pp": [cx, cy], "pp_offset": round(off, 3), "k1": round(k1, 4),
            "focal": [fx, fy], "aspect_error": round(aspect, 4),
            "identifiable": not problems, "problems": problems}


def pooled_intrinsics(recordings, cam, flags: int = 0):
    """``(K, dist, rms, n, sources)`` for one camera from EVERY recording that shows it the board.

    This is the half that pools without an extra assumption. ``cv2.calibrateCamera`` takes object
    points per view, so a 40 mm board and a 26 mm board contribute side by side; all that is assumed
    is that nobody refocused or swapped the lens between recordings.
    """
    import cv2

    obj, img, size, sources = [], [], None, {}
    # Frame sizes are checked across ALL recordings before any detection work, so an impossible
    # pooling is refused on the cheap fact rather than after building boards for each.
    seen = {tuple(s[cam]) for _c, _r, s, _sp in recordings.values() if cam in s}
    if len(seen) > 1:
        raise ValueError(f"{cam}: frame size differs between recordings ({sorted(seen)}) -- these "
                         f"are not the same camera configuration and their intrinsics must not be "
                         f"pooled.")
    for cal_dir, (cams, rows, sizes, spec) in recordings.items():
        if cam not in cams:
            continue
        size = sizes[cam] if size is None else size
        board = _tuned_board(spec)
        o, i = board.get_all_calibration_points(_one_per_pose(rows[cam]),
                                                min_points=ANIPOSE_MIN_CORNERS_INTRINSIC)
        pairs = [(a, b) for a, b in zip(o, i) if len(a) >= ANIPOSE_MIN_CORNERS_INTRINSIC]
        sources[Path(cal_dir).name] = len(pairs)
        obj += [a for a, _ in pairs]
        img += [b for _, b in pairs]
    if len(obj) < 5:
        raise RuntimeError(f"{cam}: {len(obj)} usable views across {len(recordings)} recording(s); "
                           f"cannot solve intrinsics")
    if len(obj) > MAX_VIEWS:
        k = np.linspace(0, len(obj) - 1, MAX_VIEWS).round().astype(int)
        obj, img = [obj[j] for j in k], [img[j] for j in k]
    rms, K, dist, _, _ = cv2.calibrateCamera(obj, img, tuple(size), None, None, flags=flags)
    return K, dist, float(rms), len(obj), sources


def _one_per_pose(rows, fps: float = 250.0, gap_s: float = POSE_GAP_S):
    """One row per distinct board pose -- duplicates of a held board add runtime, not information."""
    out, last = [], -1e18
    for r in sorted(rows, key=_framenum):
        f = _framenum(r)
        if (f - last) / fps > gap_s:
            out.append(r)
            last = f
    return out


def pair_extrinsics(rows, sizes, spec, a, b, K, dist):
    """``(R, T, rms, n)`` for one camera pair within ONE recording, intrinsics held fixed.

    Used for the rig-unmoved test: the same pair solved independently per recording, then compared.
    """
    import cv2

    board = _tuned_board(spec)
    objp = board.get_object_points().reshape(-1, 3)
    by_frame = {}
    for cam in (a, b):
        for r in rows[cam]:
            by_frame.setdefault(_framenum(r), {})[cam] = r
    obj, ia, ib = [], [], []
    last = -1e18
    for f in sorted(by_frame):
        if (f - last) / 250.0 <= POSE_GAP_S or len(by_frame[f]) < 2:
            continue
        ra, rb = by_frame[f][a], by_frame[f][b]
        ids = np.intersect1d(np.asarray(ra["ids"]).ravel(), np.asarray(rb["ids"]).ravel())
        if len(ids) < ANIPOSE_MIN_CORNERS_BUNDLE:
            continue
        pa = ra["filled"].reshape(-1, 2)[ids]
        pb = rb["filled"].reshape(-1, 2)[ids]
        if np.isnan(pa).any() or np.isnan(pb).any():
            continue
        obj.append(np.float32(objp[ids]))
        ia.append(np.float32(pa))
        ib.append(np.float32(pb))
        last = f
    if len(obj) < 4:
        return None, None, float("nan"), len(obj)
    rms, *_, R, T, _, _ = cv2.stereoCalibrate(obj, ia, ib, K[a], dist[a], K[b], dist[b],
                                              tuple(sizes[a]), flags=cv2.CALIB_FIX_INTRINSIC)
    return R, T, float(rms), len(obj)


def cross_recording_agreement(recordings, K, dist) -> list[dict]:
    """Does a camera PAIR sit in the same place in two recordings? The rig-unmoved test.

    For every pair solvable in more than one recording, the two independent relative poses are
    compared. An empty result means the assumption is UNTESTABLE with these recordings, which is not
    the same as passing it -- ``solve`` treats it as a refusal.
    """
    import itertools

    per_pair: dict[tuple, list] = {}
    for cal_dir, (cams, rows, sizes, spec) in recordings.items():
        for a, b in itertools.combinations(sorted(cams), 2):
            if a not in K or b not in K:
                continue
            R, T, rms, n = pair_extrinsics(rows, sizes, spec, a, b, K, dist)
            if R is not None:
                per_pair.setdefault((a, b), []).append((Path(cal_dir).name, R, T, rms, n))

    out = []
    for (a, b), got in sorted(per_pair.items()):
        if len(got) < 2:
            continue
        for (n1, R1, T1, _r1, c1), (n2, R2, T2, _r2, c2) in itertools.combinations(got, 2):
            dR = R2 @ R1.T
            ang = float(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))))
            mm = float(np.linalg.norm(T2.ravel() - T1.ravel()))
            out.append({"pair": f"{a}-{b}", "a": n1, "b": n2, "deg": round(ang, 3),
                        "mm": round(mm, 3), "n_a": c1, "n_b": c2,
                        "ok": bool(ang <= MAX_RIG_DRIFT_DEG and mm <= MAX_RIG_DRIFT_MM)})
    return out


def solve(cal_dirs, step: int = 10, assume_rig_unmoved: bool = False, refresh: bool = False,
          out_dir=None, machine=None, fix_principal_point: bool = False) -> dict:
    """Pool the given recordings and solve. Intrinsics always; extrinsics from one recording unless
    ``assume_rig_unmoved`` AND the rig-unmoved test passes.

    ``fix_principal_point`` pins the principal point to the FRAME centre, which rescues an
    unidentifiable solve (see ``MAX_PP_OFFSET``) at the cost of an assumption that is only
    approximately true: these frames are ROIs read off a larger sensor, so the optical axis sits at
    the ROI centre only if the ROI is centred on the sensor. It is a fallback for a recording whose
    board poses are not diverse enough, never the first choice.
    """
    import cv2
    from aniposelib.cameras import CameraGroup

    flags = (cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3
             if fix_principal_point else 0)

    dirs = [Path(d) for d in (cal_dirs or [find_calibration_dir(machine=machine)])]
    recordings = {d: detect_rows(d, step, refresh=refresh) for d in dirs}

    print("\n[dlc_anipose] what aniposelib will accept", flush=True)
    g = gate(recordings)
    for d, info in g.items():
        print(f"  {Path(d).name}  (board {info['board']['squares_x']}x{info['board']['squares_y']}, "
              f"square {info['board']['square_mm']} mm)", flush=True)
        for cam, p in sorted(info["cameras"].items()):
            print(f"    {cam}: max {p['max_corners']:2d} corners, {p['poses_intrinsic']:3d} "
                  f"intrinsic poses, {p['poses_bundle']:3d} bundle poses -- {p['verdict']}",
                  flush=True)

    cams = sorted({c for (cs, _r, _s, _sp) in recordings.values() for c in cs})
    K, dist, rms, nviews, src, qual = {}, {}, {}, {}, {}, {}
    print("\n[dlc_anipose] pooled intrinsics"
          + ("  (principal point PINNED to the frame centre)" if fix_principal_point else ""),
          flush=True)
    usable = []
    for cam in cams:
        try:
            K[cam], dist[cam], rms[cam], nviews[cam], src[cam] = pooled_intrinsics(
                recordings, cam, flags)
        except RuntimeError as exc:
            print(f"    {cam}: {exc}", flush=True)
            continue
        usable.append(cam)
        qual[cam] = intrinsics_quality(K[cam], dist[cam], _any_size(recordings, cam))
        f = (K[cam][0, 0] + K[cam][1, 1]) / 2
        where = ", ".join(f"{k}:{v}" for k, v in src[cam].items())
        print(f"    {cam}: {nviews[cam]:3d} views ({where}), f={f:7.1f} px, RMS {rms[cam]:.3f} px",
              flush=True)
        for p in qual[cam]["problems"]:
            print(f"          NOT IDENTIFIABLE: {p}", flush=True)

    bad = [c for c in usable if not qual[c]["identifiable"]]
    if bad:
        remedy = ("The principal point is ALREADY pinned, so there is nothing further to constrain "
                  "here -- the degeneracy has moved into the aspect ratio or the distortion, and "
                  "the fix is at the rig."
                  if fix_principal_point else
                  "Or re-run with --fix-principal-point to pin it to the frame centre -- an "
                  "assumption, not a measurement: these frames are ROIs and the optical axis is at "
                  "the ROI centre only if the ROI is centred on the sensor.")
        raise RuntimeError(
            f"intrinsics are not identifiable for {', '.join(bad)} despite a low reprojection error "
            f"-- every board pose in these recordings sits at a similar depth and tilt, so focal "
            f"length, principal point, aspect ratio and distortion trade off against one another. "
            f"RE-RECORD with the board SWEPT through depth and tilted well off parallel. " + remedy)

    agreement = cross_recording_agreement(recordings, K, dist) if len(recordings) > 1 else []
    if len(recordings) > 1:
        print("\n[dlc_anipose] rig-unmoved test (same pair, solved separately per recording)",
              flush=True)
        if not agreement:
            print("    NO PAIR is solvable in more than one recording -- the assumption that the "
                  "rig did not move CANNOT BE TESTED from these recordings.", flush=True)
        for r in agreement:
            print(f"    {r['pair']}: {r['deg']:6.3f} deg, {r['mm']:7.3f} mm between "
                  f"{r['a']} ({r['n_a']} poses) and {r['b']} ({r['n_b']} poses) "
                  f"-- {'consistent' if r['ok'] else 'MOVED'}", flush=True)

    pooled_ok = bool(agreement) and all(r["ok"] for r in agreement)
    if assume_rig_unmoved and not pooled_ok:
        raise RuntimeError(
            "--assume-rig-unmoved refused: " +
            ("no pair is observed in more than one recording, so the assumption is untestable"
             if not agreement else
             "the rig-unmoved test FAILED on " +
             ", ".join(r["pair"] for r in agreement if not r["ok"])) +
            ". Pooled extrinsics would place the cameras somewhere neither recording supports.")

    extr_dirs = (dirs if (assume_rig_unmoved and pooled_ok)
                 else [_best_extrinsic_dir(recordings, usable)])
    print(f"\n[dlc_anipose] extrinsics from {', '.join(d.name for d in extr_dirs)}", flush=True)

    cgroup = CameraGroup.from_names(usable)
    for cam, c in zip(usable, cgroup.cameras):
        c.set_size(recordings[dirs[0]][2].get(cam) or _any_size(recordings, cam))
        c.set_camera_matrix(K[cam].copy())
        c.set_distortions(dist[cam].ravel().copy())
    all_rows, board = _rows_for(recordings, extr_dirs, usable)
    err = cgroup.calibrate_rows(all_rows, board, init_intrinsics=False, init_extrinsics=True,
                                verbose=True, only_extrinsics=True)
    print(f"[dlc_anipose] bundle-adjusted reprojection error {err:.3f} px", flush=True)

    out = {
        "recordings": [d.name for d in dirs],
        "extrinsics_from": [d.name for d in extr_dirs],
        "assume_rig_unmoved": bool(assume_rig_unmoved and pooled_ok),
        "rig_unmoved_test": agreement,
        "reprojection_error_px": float(err),
        "fix_principal_point": bool(fix_principal_point),
        "gate": g,
        "cameras": {cam: {**c.get_dict(), "intrinsic_rms_px": rms[cam],
                          "n_views": nviews[cam], "views_by_recording": src[cam],
                          "identifiability": qual[cam]}
                    for cam, c in zip(usable, cgroup.cameras)},
    }
    write(out, cgroup, out_dir or dirs[-1])
    return out


def _any_size(recordings, cam):
    for _cs, _r, sizes, _sp in recordings.values():
        if cam in sizes:
            return sizes[cam]
    raise KeyError(cam)


def _best_extrinsic_dir(recordings, cams=None) -> Path:
    """The recording whose cameras jointly clear aniposelib's bundle threshold most often.

    Extrinsics come from ONE recording by default because a relative pose is a fact about one day.

    Scored over the cameras that SURVIVED the intrinsics solve, not over every camera in the
    recording. Scoring over all of them takes the minimum across a camera that has already been
    dropped, which is zero for every recording -- and then `max` picks whichever one happened to be
    first rather than the best.
    """
    def score(item):
        _d, (rec_cams, rows, _s, _sp) = item
        use = [c for c in rec_cams if cams is None or c in cams]
        return min((pose_counts(rows[c], ANIPOSE_MIN_CORNERS_BUNDLE) for c in use), default=0)
    return max(recordings.items(), key=score)[0]


def _rows_for(recordings, extr_dirs, cams):
    """Rows per camera for the bundle adjustment, and the board they were detected with.

    Rows from several recordings are concatenated only when they share a board: aniposelib's
    ``calibrate_rows`` takes ONE board and indexes object points by ChArUco corner id, so rows from a
    different geometry would be read against the wrong 3D structure -- silently, because both our
    boards are DICT_4X4_50 with overlapping ids.
    """
    specs = {json.dumps(recordings[d][3], sort_keys=True) for d in extr_dirs}
    if len(specs) > 1:
        raise ValueError(
            "extrinsics cannot pool recordings that used DIFFERENT boards: aniposelib indexes object "
            "points by ChArUco corner id against one board geometry. Pool their INTRINSICS (which "
            "this module already does) and take extrinsics from a single recording.")
    spec = recordings[extr_dirs[0]][3]
    per_cam = {c: [] for c in cams}
    for i, d in enumerate(extr_dirs):
        for c in cams:
            for r in recordings[d][1].get(c, []):
                q = dict(r)
                q["framenum"] = (i, _framenum(r))
                per_cam[c].append(q)
    return [per_cam[c] for c in cams], _tuned_board(spec)


def write(out: dict, cgroup, dest) -> list[Path]:
    """JSON (with the provenance) plus the anipose TOML that triangulation actually loads."""
    dest = Path(dest)
    assert_writable(dest)
    dest.mkdir(parents=True, exist_ok=True)
    j = dest / "calibration_anipose.json"
    j.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    t = dest / "calibration_anipose.toml"
    cgroup.dump(str(t))
    print(f"[dlc_anipose] -> {j}\n[dlc_anipose] -> {t}", flush=True)
    return [j, t]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", action="append", default=None,
                    help="calibration recording (repeatable; default: the newest)")
    ap.add_argument("--step", type=int, default=10, help="sample every Nth frame for detection")
    ap.add_argument("--assume-rig-unmoved", action="store_true",
                    help="pool EXTRINSICS too; refused unless the rig-unmoved test passes")
    ap.add_argument("--gate", action="store_true", help="report what aniposelib accepts, no solve")
    ap.add_argument("--fix-principal-point", action="store_true",
                    help="pin the principal point to the frame centre (an assumption; see --help)")
    ap.add_argument("--refresh", action="store_true", help="re-detect, ignoring the cache")
    ap.add_argument("--output", default=None)
    ap.add_argument("--machine", default=None)
    args = ap.parse_args(argv)

    if args.gate:
        dirs = [Path(d) for d in (args.dir or [find_calibration_dir(machine=args.machine)])]
        recordings = {d: detect_rows(d, args.step, refresh=args.refresh) for d in dirs}
        for d, info in gate(recordings).items():
            print(f"\n{Path(d).name}  board {info['board']['squares_x']}x"
                  f"{info['board']['squares_y']} square {info['board']['square_mm']} mm")
            for cam, p in sorted(info["cameras"].items()):
                print(f"  {cam}: max {p['max_corners']:2d} corners, "
                      f"{p['poses_intrinsic']:3d} intrinsic poses, "
                      f"{p['poses_bundle']:3d} bundle poses -- {p['verdict']}")
        return 0

    solve(args.dir, args.step, args.assume_rig_unmoved, args.refresh, args.output, args.machine,
          args.fix_principal_point)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
