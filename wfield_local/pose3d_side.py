"""Side cameras (cam2 + cam3) in the 3-D pose: per-session PnP, N-camera triangulation, and the CROSS-CAMERA OFFSET check.

    from wfield_local import pose3d_side as PS
    side = PS.side_session(rv, animal, date, preds_dir, clips_dir, ix4)    # {cam: (ix, flat raw px)}
    cg4, err = PS.pose_all(cg, meds, rig["geometry"])                      # all four cameras posed from the spouts
    R = PS.offsets_vs_reference(cg4, S, side)                              # side point vs the cam1 + cam4 3-D point

WHY (Priya 2026-11-04: "prep and test the code for using cam 2/3 for full 3d keypoints ... I'll want to know if we need
to adjust for any offset between cam 2 vs cam 3 vs cam 4 nose / spout / jaw"). The side views add the jaw in profile (no
spout rod over the chin) and the tongue's forward / downward reach -- exactly the depth cam1 + cam4 measure least well.
Before they enter the triangulation, each camera's keypoint has to be the SAME physical point: a labeller places "nose"
on the snout tip as each view shows it, and the front (cam4), bottom (cam1) and profile (cam2 / cam3) silhouettes need
not share a tip. The spout already has such an offset (`pose3d.EDGE_OFFSET_MM`: cam1 labels the rod's inferior edge,
cam4 the superior edge).

HOW.
  * POSES: `pose3d.pnp_group` unchanged -- it poses EVERY camera in the group from its own spout medians (cv2 SQPnP + LM
    against the commanded positions), intrinsics from the 09-11 ChArUco solve, which calibrated all four cameras (cam2
    83 / cam3 54 board poses; 600 x 450, the AUGUST side configuration). So cam1 / cam4 get exactly the poses the
    production 3-D uses. JUNE side video is 800 x 600 with a 1.24x larger face (`lp_sidecam`): another camera set-up,
    which 09-11 does not describe -- June side sessions are flagged, not trusted, until they get their own intrinsics.
  * SIDE POINTS: the shared side model's MODEL-space predictions (cam3 flipped, cropped, June rescaled) mapped back to
    the raw frame with `lp_sidecam.to_raw`; confident at `lp_sidecam.PRELABEL_CUT` (tongue 0.4, others 0.6).
  * REFERENCE: the production cam1 + cam4 frames (`pose3d.session_3d`: cleaned tongue / jaw, nose / spout raw >= 0.9),
    paired with the side frames on (trial_id, t_ms) -- the side review clips are cut on each camera's own template on
    the same 4 ms grid from the cue.
  * OFFSET per side camera x part: the side keypoint minus the projection of the cam1 + cam4 3-D point, in px (median
    dx, dy: a SYSTEMATIC shift is a labelling offset; the spread is noise), and in mm as the perpendicular vector from
    that 3-D point to the side camera's ray (world ml / ap / dv; the component along the ray is invisible to one camera).
  * SIDE STEREO: cam2 + cam3 triangulated alone, against cam1 + cam4: the full 3-D offset vector per part.

RESULT (11-04, PS93 0814 / 0821 / 0908 + PS95 0606; POSE_MODELS Part E) and the CORRECTIONS `fuse` applies:
  * SPOUT -- cam4 is the truth (its superior-edge keypoint IS the commanded upper tip) and cam1 carries
    `pose3d.EDGE_OFFSET_MM` (one rod diameter below). The side spout keypoints sit ON the commanded tip (ray vs
    commanded 0.00-0.02 mm; and the side jaw / tongue agree with cam1 + cam4 within ~0.3 mm, which a side spout offset
    would break): NO side spout offset. The 0.8 mm "side spout offset" first measured was against `session_3d`'s
    per-frame spout, which triangulates cam1's inferior and cam4's superior edge as one point (a blend 1.04 mm behind the
    commanded tip in ap, identically at every position); `fuse` triangulates the spout with cam1's pose shifted by its
    edge offset instead (the `_identity_world` rule), which lands on the commanded tip (0.00 mm).
  * NOSE -- each view places "nose" on its own silhouette: the side nose sits HIGHER than the cam1 / cam4 nose
    (dv +0.29 to +1.18 mm by session). `SIDE_NOSE_OFFSET_MM` (per-camera medians of the four sessions) moves the side
    keypoints onto the cam1 / cam4 definition before fusing.
  * JAW, TONGUE -- fused as measured (offsets <= 0.4 mm, about the label noise).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SIDE = ("cam2", "cam3")
PARTS = ("nose", "jaw", "tongue", "spout")
#: Where each SIDE camera's nose keypoint sits relative to the cam1 / cam4 nose (world ml, ap, dv mm): the per-camera
#: median of `offsets_vs_reference` over PS93 0814 / 0821 / 0908 + PS95 0606 (dv range 0.29-0.90 cam2, 0.70-1.18 cam3).
SIDE_NOSE_OFFSET_MM = {"cam2": (0.02, 0.05, 0.68), "cam3": (-0.06, 0.13, 0.90)}


def part_offsets(part: str) -> dict:
    """{camera: offset mm} that `fuse` applies for ``part`` (module docstring: spout = cam1's edge, nose = the side
    definitions, jaw / tongue none)."""
    from wfield_local import pose3d as P3
    if part == "spout":
        return {c: o for c, o in P3.EDGE_OFFSET_MM.items() if any(o)}
    if part == "nose":
        return dict(SIDE_NOSE_OFFSET_MM)
    return {}


def side_width(rv, animal: str, date: str, cam: str) -> int:
    """Raw frame width of the session's side video (600 August onward, 800 June)."""
    import cv2
    vids = sorted((Path(rv.root("behavior_cameras")) / date / animal).glob(f"{cam}_*.avi"))
    cap = cv2.VideoCapture(str(vids[0]))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()
    return w


def side_session(rv, animal: str, date: str, preds_dir: Path, clips_dir: Path, ix4: pd.DataFrame) -> dict:
    """{cam: (ix, flat)} for the side review clip of one session. ``ix``: trial_id, position, t_ms, strobe_ms (from the
    cam4 windows, per trial); ``flat``: RAW px ``{part}_x/_y/_likelihood``, rows aligned with ``ix``."""
    from wfield_local import lp_sidecam as LS
    idx = pd.read_csv(Path(clips_dir) / f"{animal}_{date}" / "index.csv")
    strobe = ix4.groupby("trial_id").strobe_ms.first()
    out = {}
    for cam in SIDE:
        pr = pd.read_csv(Path(preds_dir) / f"{animal}_{date}_{cam}.csv", header=[0, 1, 2], index_col=0)
        pr.columns = pr.columns.droplevel(0)
        pr = pr.iloc[:len(idx)].reset_index(drop=True)
        w = side_width(rv, animal, date, cam)
        flat = pd.DataFrame(index=pr.index)
        for p in PARTS:
            x, y = LS.to_raw(pr[(p, "x")].to_numpy(float), pr[(p, "y")].to_numpy(float), cam, w)
            flat[f"{p}_x"], flat[f"{p}_y"], flat[f"{p}_likelihood"] = x, y, pr[(p, "likelihood")].to_numpy(float)
        ix = idx[["trial_id", "position", "t_ms"]].copy()
        ix["src_frame"] = idx[f"src_frame_{cam}"].to_numpy()          # raw video frame (for drawing on the raw view)
        ix["strobe_ms"] = ix.trial_id.map(strobe).to_numpy(float)
        ix["width"] = w
        out[cam] = (ix, flat)
    return out


def confident_xy(flat: pd.DataFrame, part: str, cut: float | None = None) -> np.ndarray:
    from wfield_local import lp_sidecam as LS
    xy = flat[[f"{part}_x", f"{part}_y"]].to_numpy(float)
    lk = flat[f"{part}_likelihood"].to_numpy(float)
    return np.where((lk >= (LS.PRELABEL_CUT[part] if cut is None else cut))[:, None], xy, np.nan)


def pose_all(cg, meds: dict, geometry: dict, side_edge_dv: float = 0.0):
    """Every camera in ``meds`` posed per session from the spouts (`pose3d.pnp_group`). ``side_edge_dv``: where the
    side spout keypoint sits below the commanded (upper-tip) point, mm (0 = at it; -rod radius = the rod's centre line)."""
    from wfield_local import pose3d as P3
    g = cg.subset_cameras_names(list(meds))
    off = {**P3.EDGE_OFFSET_MM, **{c: (0.0, 0.0, side_edge_dv) for c in SIDE}}
    return P3.pnp_group(g, meds, geometry, off)


def cam_of(g, name: str):
    return next(c for c in g.cameras if c.get_name() == name)


def project(cam, X: np.ndarray) -> np.ndarray:
    out = np.full((len(X), 2), np.nan)
    ok = np.isfinite(X).all(1)
    if ok.any():
        out[ok] = cam.project(X[ok]).reshape(-1, 2)
    return out


def ray_offset(cam, xy: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Perpendicular vector (n, 3) world mm from 3-D point X to the ray through image point xy (NaN where missing)."""
    from wfield_local import pose3d as P3
    out = np.full((len(X), 3), np.nan)
    ok = np.isfinite(xy).all(1) & np.isfinite(X).all(1)
    if ok.any():
        c, d = P3.camera_rays(cam, xy[ok])
        v = X[ok] - c
        out[ok] = (v - (v * d).sum(1, keepdims=True) * d) * -1.0       # point -> ray
    return out


def triangulate_n(g, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(X (n, 3), reprojection px (n_cams, n)) from ``pts`` (n_cams, n, 2) in ``g``'s camera order; a point needs >= 2
    finite views (aniposelib drops NaN views per point)."""
    n = pts.shape[1]
    X = np.full((n, 3), np.nan)
    rep = np.full((pts.shape[0], n), np.nan)
    ok = np.isfinite(pts).all(2).sum(0) >= 2
    if ok.any():
        X[ok] = g.triangulate(pts[:, ok], undistort=True)
        pr = g.project(X[ok])
        rep[:, ok] = np.linalg.norm(pr - pts[:, ok], axis=2)
    return X, rep


def shifted(g, offsets: dict):
    """Copy of ``g`` in which camera c projects X + offsets[c] (its keypoint sits there): t' = t + R o."""
    import copy

    import cv2
    gs = copy.deepcopy(g)
    for cam in gs.cameras:
        o = np.asarray(offsets.get(cam.get_name(), (0.0, 0.0, 0.0)), float)
        if o.any():
            M = cam.get_extrinsics_mat()
            cam.set_translation(M[:3, 3] + M[:3, :3] @ o)
            cam.set_rotation(cv2.Rodrigues(M[:3, :3])[0].ravel())
    return gs


def fuse(g, pts_by_part: dict) -> dict:
    """{part: (X (n, 3) world mm, reprojection px (n_cams, n), n_views (n,))} from ``pts_by_part[part]`` = (n_cams, n, 2)
    in ``g``'s camera order, NaN = not seen, with each part's `part_offsets` applied."""
    out = {}
    for part, pts in pts_by_part.items():
        gs = shifted(g, part_offsets(part))
        X, rep = triangulate_n(gs, pts)
        out[part] = (X, rep, np.isfinite(pts).all(2).sum(0))
    return out


def four_camera_points(S: pd.DataFrame, info: dict, c1, c4, side: dict, order: list[str]) -> tuple[pd.DataFrame, dict]:
    """The frames shared by cam1 + cam4 (`session_3d`'s pairing) and both side review clips, and per part the
    (n_cams, n, 2) image points in ``order``: cam1 / cam4 as `session_3d` uses them (tongue / jaw after each camera's
    cleaning, MEASURED frames only; nose / spout raw >= `pose3d.STATIC_LK`), side points confident at
    `lp_sidecam.PRELABEL_CUT`. Returns (frames: S rows + side clip row per camera, {part: pts})."""
    from wfield_local import pose3d as P3
    from wfield_local import spout_frame as SF
    ix1, flat1 = c1
    ix4, flat4 = c4
    i1, i4 = P3.pair(ix1, ix4)
    mouth1, mouth4 = SF.from_medians_lr(info["med1"]).origin, SF.from_medians(info["med4"]).origin
    base = {}
    for part in PARTS:
        if part in P3.CLEANED:
            a = P3.cleaned_xy(flat1, ix1, part, "cam1", mouth1)[0][i1]
            b = P3.cleaned_xy(flat4, ix4, part, "cam4", mouth4)[0][i4]
        else:
            a, b = P3._raw_xy(flat1, part, i1), P3._raw_xy(flat4, part, i4)
        base[part] = {"cam1": a, "cam4": b}
    F = S.reset_index(drop=True).reset_index().rename(columns={"index": "iS"})
    for cam in SIDE:
        iS, iX = pair_side(S, side[cam][0])
        F = F.merge(pd.DataFrame({"iS": iS, f"i_{cam}": iX}), on="iS")
    pts = {}
    for part in PARTS:
        cols = []
        for cam in order:
            if cam in SIDE:
                cols.append(confident_xy(side[cam][1], part)[F[f"i_{cam}"].to_numpy()])
            else:
                cols.append(base[part][cam][F.iS.to_numpy()])
        pts[part] = np.stack(cols)
    return F, pts


def pair_side(S: pd.DataFrame, ix: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Row numbers (iS, iside) on exact (trial_id, t_ms)."""
    a = S[["trial_id", "t_ms"]].reset_index(drop=True).reset_index().rename(columns={"index": "iS"})
    b = ix[["trial_id", "t_ms"]].reset_index(drop=True).reset_index().rename(columns={"index": "iX"})
    a["t_ms"], b["t_ms"] = a.t_ms.round(1), b.t_ms.round(1)
    m = a.merge(b, on=["trial_id", "t_ms"], how="inner")
    return m.iS.to_numpy(), m.iX.to_numpy()


def _summ(v: np.ndarray) -> dict:
    v = v[np.isfinite(v).all(1)] if v.ndim == 2 else v[np.isfinite(v)]
    return {"n": len(v)} | ({} if not len(v) else
                            {"med": np.round(np.median(v, 0), 2).tolist(), "iqr": np.round(np.subtract(
                                *np.percentile(v, [75, 25], 0)), 2).tolist()})


def offsets_vs_reference(g, S: pd.DataFrame, side: dict) -> pd.DataFrame:
    """One row per side camera x part: the side keypoint vs the cam1 + cam4 3-D point (module docstring)."""
    rows = []
    for cam in SIDE:
        ix, flat = side[cam]
        iS, iX = pair_side(S, ix)
        c = cam_of(g, cam)
        for p in PARTS:
            X = S[[f"{p}_ml", f"{p}_ap", f"{p}_dv"]].to_numpy(float)[iS]
            xy = confident_xy(flat, p)[iX]
            d = xy - project(c, X)
            r = ray_offset(c, xy, X)
            ok = np.isfinite(d).all(1)
            rows.append({"cam": cam, "part": p, "n": int(ok.sum()),
                         "dx_px": float(np.median(d[ok, 0])) if ok.any() else np.nan,
                         "dy_px": float(np.median(d[ok, 1])) if ok.any() else np.nan,
                         "abs_px": float(np.median(np.linalg.norm(d[ok], axis=1))) if ok.any() else np.nan,
                         "abs_px_iqr": float(np.subtract(*np.percentile(np.linalg.norm(d[ok], axis=1), [75, 25])))
                         if ok.any() else np.nan,
                         **{f"ray_{ax}_mm": float(np.median(r[ok, k])) if ok.any() else np.nan
                            for k, ax in enumerate(("ml", "ap", "dv"))},
                         "ray_abs_mm": float(np.median(np.linalg.norm(r[ok], axis=1))) if ok.any() else np.nan})
    return pd.DataFrame(rows)


def side_stereo_vs_reference(g, S: pd.DataFrame, side: dict) -> pd.DataFrame:
    """cam2 + cam3 triangulated alone (both confident) minus the cam1 + cam4 point, per part (median ml / ap / dv mm)."""
    ix2, f2 = side["cam2"]
    ix3, f3 = side["cam3"]
    a = ix2[["trial_id", "t_ms"]].reset_index(drop=True).reset_index().rename(columns={"index": "i2"})
    b = ix3[["trial_id", "t_ms"]].reset_index(drop=True).reset_index().rename(columns={"index": "i3"})
    m = a.merge(b, on=["trial_id", "t_ms"])
    gs = g.subset_cameras_names(list(SIDE))
    iS, iM = pair_side(S, m)
    rows = []
    for p in PARTS:
        pts = np.stack([confident_xy(f2, p)[m.i2.to_numpy()][iM], confident_xy(f3, p)[m.i3.to_numpy()][iM]])
        X, rep = triangulate_n(gs, pts)
        ref = S[[f"{p}_ml", f"{p}_ap", f"{p}_dv"]].to_numpy(float)[iS]
        d = X - ref
        ok = np.isfinite(d).all(1)
        rows.append({"part": p, "n_side3d": int(np.isfinite(X).all(1).sum()), "n_both": int(ok.sum()),
                     **{f"d_{ax}_mm": float(np.median(d[ok, k])) if ok.any() else np.nan
                        for k, ax in enumerate(("ml", "ap", "dv"))},
                     "d_abs_mm": float(np.median(np.linalg.norm(d[ok], axis=1))) if ok.any() else np.nan,
                     "rep_cam2_px": float(np.nanmedian(rep[0])) if np.isfinite(rep[0]).any() else np.nan,
                     "rep_cam3_px": float(np.nanmedian(rep[1])) if np.isfinite(rep[1]).any() else np.nan})
    return pd.DataFrame(rows)
