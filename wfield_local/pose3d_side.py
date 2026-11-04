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
        ix["trial_k"] = idx.trial_k.to_numpy()                       # window id, for `orofacial_clean.clean_windows`
        out[cam] = (ix, flat)
    return out


def side_windows(rv, bundle: str, animal: str, date: str, cam: str, model_space: bool = True):
    """``(ix, flat)`` for one side camera's WHOLE session from an O2 bundle (`o2_inference.session_windows`: the same
    cue windows / template timing as cam1 / cam4). ``model_space``: the bundle ran the shared side model on the
    transformed video (cam3 flipped, cropped, June rescaled, `lp_sidecam`), so its x / y go back to the raw frame with
    `lp_sidecam.to_raw` here. None when the session is missing or incomplete."""
    from wfield_local import lp_sidecam as LS
    from wfield_local import o2_inference as oi
    got = oi.session_windows(rv, bundle, animal, date, cam)
    if got is None:
        return None
    ix, flat = got
    ix, flat = ix.copy(), flat.copy()
    w = side_width(rv, animal, date, cam)
    if model_space:
        for part in PARTS:
            flat[f"{part}_x"], flat[f"{part}_y"] = LS.to_raw(flat[f"{part}_x"].to_numpy(float),
                                                             flat[f"{part}_y"].to_numpy(float), cam, w)
    ix["width"] = w
    return ix, flat


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


def four_camera_points(S: pd.DataFrame, info: dict, c1, c4, side: dict, order: list[str],
                       keep_all: bool = False) -> tuple[pd.DataFrame, dict]:
    """The frames shared by cam1 + cam4 (`session_3d`'s pairing) and both side review clips, and per part the
    (n_cams, n, 2) image points in ``order``: cam1 / cam4 as `session_3d` uses them (tongue / jaw after each camera's
    cleaning, MEASURED frames only; nose / spout raw >= `pose3d.STATIC_LK`), side points confident at
    `lp_sidecam.PRELABEL_CUT`. Returns (frames: S rows + side clip row per camera, {part: pts}). ``keep_all``
    (whole sessions): every S row is kept and a side camera without that frame counts as not seeing it (``i_<cam>``
    -1); False (review clips): only frames all four cameras have."""
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
        F = F.merge(pd.DataFrame({"iS": iS, f"i_{cam}": iX}), on="iS", how="left" if keep_all else "inner")
        F[f"i_{cam}"] = F[f"i_{cam}"].fillna(-1).astype(int)
    pts = {}
    for part in PARTS:
        cols = []
        for cam in order:
            if cam in SIDE:
                k = F[f"i_{cam}"].to_numpy()
                xy = confident_xy(side[cam][1], part)
                cols.append(np.where((k >= 0)[:, None], xy[np.clip(k, 0, None)], np.nan))
            else:
                cols.append(base[part][cam][F.iS.to_numpy()])
        pts[part] = np.stack(cols)
    return F, pts


def pair_side(S: pd.DataFrame, ix: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Row numbers (iS, iside). Both with ``t_daq_ms`` (whole-session windows): `pose3d.pair` (nearest true time per
    trial within half a frame, one-to-one -- dropped frames leave gaps). Otherwise (review clips): exact (trial_id, t_ms)."""
    if "t_daq_ms" in S and "t_daq_ms" in ix:
        from wfield_local import pose3d as P3
        return P3.pair(S[["trial_id", "t_ms", "t_daq_ms"]].reset_index(drop=True), ix.reset_index(drop=True))
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



# --------------------------------------------------------------------------- full four-camera pipeline (11-04)
# Priya 2026-11-04 ("is the current video using cleaned traces for cam 1 and 4, with cross-cam interpolation? can we
# build in the same types of processing for cam 2/3 and then use all 4 cameras for better interpolation of occluded
# points?", after a water drop moved keypoints at the start of the PS93 0821 clip). Four stages, each the 4-camera
# counterpart of a `pose3d.session_3d` stage:
#   1. SIDE CLEANING  `side_clean`: `orofacial_clean` (v5p3 + jaw v3.4) on cam2 / cam3 with cam4's thresholds scaled by
#      the side camera's px / mm at the mouth (`px_per_mm`, from the session's PnP poses), tongue max gap 24 ms (cam1's
#      rule: longer gaps bridge licks) without the between-lick veto, jaw x-flip rule OFF (a profile view has no
#      left / right ambiguity; the flip rule exists for cam4's side-ambiguous jaw).
#   2. ROBUST FUSION  `fuse_robust`: triangulate every view, then repeatedly drop the view whose ray passes furthest from
#      the point while that distance exceeds `REJECT_MM` (mm, so independent of each camera's magnification), down to
#      two views; two views that still disagree by more than it give no point.
#   3. DROPLET RULE  (`droplet_mask`, the idea of `cam1_licks.DROPLET_*`): a tongue run seen ONLY by side cameras (no
#      cam1 / cam4 measurement) that is near-static (spread < 3 px, range < 12 px, at cam4's scale) is a water drop.
#   4. FILL + QC  `fill_n`: frames left with ONE view take that camera's ray at the depth interpolated from neighbouring
#      multi-view frames (`FILL_MS`; Akima), dropped beyond 3 mm from the estimate, as `pose3d.fill_single_view`; then
#      `pose3d.qc_3d` (far / behind / single-frame spikes) on tongue and jaw.
#   TONGUE: likewise a per-session side-tongue offset onto the cam1 + cam4 tip (`session_tongue_offsets`; ~+0.1-0.2 mm ap).
#   NOSE: the side-nose offset is fitted PER SESSION (`session_nose_offsets`: median side-vs-cam1+cam4 ray offset over
#   the session's frames) -- it varies 0.3-1.2 mm by session, so the four-session constant left +-0.3 mm.
#   SPOUT: poses and the mouth come from the ENL no-lick spout medians (`spout_frame.ruler_medians`); cam4 = truth,
#   cam1's edge offset built in (`part_offsets`).

REJECT_MM = {"nose": 0.8, "jaw": 1.0, "tongue": 1.2, "spout": 0.5}
FILL_MS = {"tongue": 24.0, "jaw": 200.0, "nose": 200.0, "spout": 200.0}
DROPLET_SPREAD_PX, DROPLET_RANGE_PX, DROPLET_MIN_FRAMES = 3.0, 12.0, 5
#: Side jaw Y CEILING = cam4's x px/mm ratio x this (11-04, PS93 0908 cleaning QC): the v3.4 ceiling is measured from ONE
#: session-wide resting jaw, and in a profile view the full jaw drop is visible, so the plainly scaled value (~45-47 px)
#: wiped 192-193 real wide-open frames per side camera on 0908 (y - session rest p99 51-53, max 57-60 px; within a trial
#: openings stay < 45 px; cam2 and cam3 lost the same trials). cam1 needed the same raise (125 -> 160 px, x 1.28).
#: x 1.4 -> ~62-66 px: above every real opening seen, far below a jaw jumping to the tongue (`jaw_off_tongue` catches that).
SIDE_Y_CEILING_FACTOR = 1.4
#: cam4 px thresholds in `orofacial_clean` that scale with image magnification (side = cam4 value x px/mm ratio)
PX_KEYS_TONGUE = ("x_range_pm", "interlick_speed_px", "interlick_near_px")
PX_KEYS_JAW = ("x_range_pm",)
PX_KEYS_V34 = ("y_ceiling_px", "x_ceiling_px", "frame_jump_thr_px", "frame_jump_thr_x_px", "x_flip_flank_agree_px",
               "x_flip_flank_rest_px", "tongue_ride_px", "tongue_out_px")


def px_per_mm(cam) -> float:
    """Image px per mm at the mouth zero: mean projected length of the two world unit steps most perpendicular to the
    camera's view (a posed camera, world mm)."""
    O = np.zeros((1, 3))
    lens = sorted(float(np.linalg.norm(project(cam, np.eye(3)[k][None]) - project(cam, O))) for k in range(3))
    return float(np.mean(lens[1:]))


def side_params(scale: float) -> tuple[dict, dict, dict]:
    """(tongue p, jaw p, jaw v34 q) for a side camera: cam4's `orofacial_clean` values, px keys x ``scale``, plus the
    side rules (module comment, stage 1)."""
    import copy

    from wfield_local import orofacial_clean as OC
    pt, pj = copy.deepcopy(OC.params("tongue", "cam4")), copy.deepcopy(OC.params("jaw", "cam4"))
    for k in PX_KEYS_TONGUE:
        if pt.get(k) is not None:
            pt[k] = float(pt[k]) * scale
    pt["max_gap_ms"], pt["interp_veto_interlick"] = 24.0, False
    for k in PX_KEYS_JAW:
        pj[k] = float(pj[k]) * scale
    q = pj["v34"]
    for k in PX_KEYS_V34:
        if q.get(k) is not None:
            q[k] = float(q[k]) * scale
    q["x_flip_step_px"] = None
    if q.get("y_ceiling_px") is not None:
        q["y_ceiling_px"] = float(q["y_ceiling_px"]) * SIDE_Y_CEILING_FACTOR
    return pt, pj, q


def side_clean(flat: pd.DataFrame, ix: pd.DataFrame, mouth_px, scale: float) -> dict:
    """{"tongue" / "jaw": (n, 2) MEASURED raw px after cleaning (NaN = not measured)} for one side camera: the
    `orofacial_clean.clean_windows` steps with the side parameters (`side_params`)."""
    from wfield_local import orofacial_clean as OC
    pt, pj, q = side_params(scale)
    cols = [c for c in flat.columns if c.startswith(("tongue_", "jaw_"))]
    df = flat[cols].reset_index(drop=True)
    index = ix.reset_index(drop=True)
    parts, pos, cues, k, sep = [], [], [], 0, 300
    for _tk, g in index.groupby("trial_k", sort=True):
        rows = g.index.to_numpy()
        parts.append(df.iloc[rows].reset_index(drop=True))
        pos.append(np.arange(k, k + len(rows)))
        cues.append(k + int(np.argmin(np.abs(g.t_ms.to_numpy()))))
        k += len(rows)
        parts.append(pd.DataFrame(np.nan, index=range(sep), columns=df.columns))
        k += sep
    big = pd.concat(parts, ignore_index=True)
    pos = np.concatenate(pos)
    out = {}
    for part, p in (("tongue", pt), ("jaw", pj)):
        src = OC.jaw_off_tongue(big, q=q) if part == "jaw" else big
        c = OC.clean_bodypart(src, part, cues, 250.0, p=p, x0y0=tuple(mouth_px))
        if part == "jaw" and q.get("enabled", True):
            c = OC.jaw_v34(c, cues, q=q)
        x, y, fm = c.x_final[pos] + c.X0, c.y_final[pos] + c.Y0, c.fill_method[pos]
        ok = (fm == OC.FILL_NONE) & np.isfinite(x) & np.isfinite(y)
        res = np.full((len(index), 2), np.nan)
        res[ok] = np.c_[x, y][ok]
        out[part] = res
    return out


def session_nose_offsets(g, S: pd.DataFrame, side: dict) -> dict:
    """Per-session side-nose offsets (median ray offset of each side camera's nose vs the cam1 + cam4 nose)."""
    off = offsets_vs_reference(g, S, side)
    n = off[off.part == "nose"].set_index("cam")
    return {c: tuple(float(n.loc[c, f"ray_{a}_mm"]) for a in ("ml", "ap", "dv")) if c in n.index and n.loc[c, "n"] > 50
            else SIDE_NOSE_OFFSET_MM[c] for c in SIDE}


def session_tongue_offsets(g, S: pd.DataFrame, side: dict, min_n: int = 50) -> dict:
    """Per-session side-TONGUE offsets onto the cam1 + cam4 tip definition (Priya 2026-11-04 "yes please do"): the
    median ray offset of each side camera's tongue tip vs the cam1 + cam4 tip, on frames BOTH cam1 and cam4 measured
    (``tongue_src`` == 1; no one-camera fills in the reference). Without it the four-camera tip sat +0.12 mm forward
    (ap; 0.07-0.19 by session, ml / dv 0.00) of the two-camera tip on every shared frame -- a definition difference
    (the side views mark the tip further out), not accuracy: at DAQ spout contacts the paired difference to the
    commanded tip was -0.01 mm (112 contacts, PS93 0814 / 0821 / 0908). Fewer than ``min_n`` frames -> no offset."""
    S2 = S.copy()
    if "tongue_src" in S2:
        S2.loc[S2.tongue_src.to_numpy() != 1, ["tongue_ml", "tongue_ap", "tongue_dv"]] = np.nan
    off = offsets_vs_reference(g, S2, side)
    t = off[off.part == "tongue"].set_index("cam")
    return {c: tuple(float(t.loc[c, f"ray_{a}_mm"]) for a in ("ml", "ap", "dv")) if c in t.index and t.loc[c, "n"] >= min_n
            else (0.0, 0.0, 0.0) for c in SIDE}


def ray_dist(g, pts: np.ndarray, X: np.ndarray) -> np.ndarray:
    """(n_cams, n) mm from X to each view's ray (NaN where the view or X is missing)."""
    return np.stack([np.linalg.norm(ray_offset(c, pts[j], X), axis=1) for j, c in enumerate(g.cameras)])


def fuse_robust(g, pts: np.ndarray, reject_mm: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(X (n, 3), used (n_cams, n) bool, ray distance mm (n_cams, n)): triangulate, drop the worst view while it is
    beyond ``reject_mm`` and > 2 views remain; two views still beyond it give no point."""
    P = np.array(pts, float)
    used = np.isfinite(P).all(2)
    for _ in range(P.shape[0] - 1):
        X, _rep = triangulate_n(g, np.where(used[:, :, None], P, np.nan))
        d = np.where(used, ray_dist(g, P, X), np.nan)
        dd = np.where(np.isfinite(d), d, -1.0)
        worst = np.argmax(dd, axis=0)
        dw = dd[worst, np.arange(dd.shape[1])]
        bad = (dw > reject_mm) & (used.sum(0) > 2)
        if not bad.any():
            break
        used[worst[bad], np.flatnonzero(bad)] = False
    X, _rep = triangulate_n(g, np.where(used[:, :, None], P, np.nan))
    d = np.where(used, ray_dist(g, P, X), np.nan)
    two_bad = (used.sum(0) == 2) & (np.where(np.isfinite(d), d, -1.0).max(0) > reject_mm)
    X[two_bad] = np.nan
    used[:, two_bad] = False
    return X, used, d


def droplet_mask(tongue_pts: np.ndarray, order: list[str], trial: np.ndarray, scale_cam4: dict) -> np.ndarray:
    """(n_cams, n) True = a side-camera tongue point that is a water drop (module comment, stage 3)."""
    has14 = np.isfinite(tongue_pts[[order.index("cam1"), order.index("cam4")]]).all(2).any(0)
    out = np.zeros(tongue_pts.shape[:2], bool)
    for c in SIDE:
        j = order.index(c)
        seen = np.isfinite(tongue_pts[j]).all(1) & ~has14
        s = scale_cam4[c]
        k = 0
        while k < len(seen):
            if not seen[k]:
                k += 1
                continue
            e = k
            while e + 1 < len(seen) and seen[e + 1] and trial[e + 1] == trial[k]:
                e += 1
            if e - k + 1 >= DROPLET_MIN_FRAMES:
                R = tongue_pts[j, k:e + 1]
                med = np.median(R, axis=0)
                spread = np.median(np.linalg.norm(R - med, axis=1))
                rng = float(np.max(np.linalg.norm(R[:, None] - R[None], axis=2)))
                if spread < DROPLET_SPREAD_PX * s and rng < DROPLET_RANGE_PX * s:
                    out[j, k:e + 1] = True
            k = e + 1
    return out


def fill_n(g, X: np.ndarray, pts: np.ndarray, used: np.ndarray, trial: np.ndarray, t: np.ndarray, max_gap_ms: float,
           max_ray_miss_mm: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    """Frames with ONE used view: that camera's ray at the depth of the neighbouring multi-view frames (Akima through
    them inside two-sided gaps, nearest within ``max_gap_ms`` one-sided), as `pose3d.fill_single_view`. Returns
    (X, views: number of views behind each point, -1 for a one-view fill, 0 none)."""
    from scipy.interpolate import Akima1DInterpolator

    from wfield_local import pose3d as P3
    X = np.array(X, float)
    multi = np.isfinite(X).all(1)
    nv = np.where(multi, used.sum(0), 0)
    one = (used.sum(0) == 1) & ~multi
    tt = pd.Series(np.where(multi, t, np.nan))
    tp, tn = tt.groupby(trial).ffill().to_numpy(), tt.groupby(trial).bfill().to_numpy()
    Xb = pd.DataFrame(np.where(multi[:, None], X, np.nan))
    Xp, Xn = Xb.groupby(trial).ffill().to_numpy(), Xb.groupby(trial).bfill().to_numpy()
    dp, dn = t - tp, tn - t
    okp, okn = dp <= max_gap_ms, dn <= max_gap_ms
    with np.errstate(invalid="ignore", divide="ignore"):
        w = (dp / (dp + dn))[:, None]
    est = np.where((okp & okn)[:, None], Xp + w * (Xn - Xp),
                   np.where(okp[:, None], Xp, np.where(okn[:, None], Xn, np.nan)))
    two = okp & okn & one
    for tr in np.unique(trial[two]):
        kb = multi & (trial == tr)
        if kb.sum() >= 3:
            kt = two & (trial == tr)
            est[kt] = Akima1DInterpolator(t[kb], X[kb], axis=0)(t[kt])
    for j, cam in enumerate(g.cameras):
        k = one & used[j] & np.isfinite(est).all(1)
        if k.any():
            C, U = P3.camera_rays(cam, pts[j][k])
            Pk = C + ((est[k] - C) * U).sum(1)[:, None] * U
            Pk[np.linalg.norm(Pk - est[k], axis=1) > max_ray_miss_mm] = np.nan
            X[k] = Pk
            nv[k] = np.where(np.isfinite(Pk).all(1), -1, 0)
    return X, nv


def fuse_session(rv, animal: str, date: str, preds_dir: Path | None = None, clips_dir: Path | None = None, cg=None,
                 side_bundle: str | None = None, c1=None, c4=None) -> dict:
    """The whole four-camera pipeline for one session. Side input: the review clips (``preds_dir`` + ``clips_dir``;
    only the frames all four cameras have) or, with ``side_bundle``, the WHOLE-SESSION O2 output of the side model
    (`side_windows`; every cam1 + cam4 frame kept). ``c1`` / ``c4``: already-loaded production windows (else loaded).
    Returns {"F": frames (S rows + side rows), "order", "g" (posed group), "pts" (cleaned image points per part),
    "parts": {part: {"X", "views", "used", "dist", "g" (the part's offset group)}}, "stats", "rig", "side", "S"}."""
    from scripts.apparatus_mouth import _rig
    from scripts.pose3d import PROD, _cgroup
    from wfield_local import o2_inference as oi
    from wfield_local import pose3d as P3
    from wfield_local import spout_frame as SF
    c1 = c1 or oi.session_windows(rv, PROD["cam1"], animal, date, "cam1")
    c4 = c4 or oi.session_windows(rv, PROD["cam4"], animal, date, "cam4")
    if c1 is None or c4 is None:
        raise FileNotFoundError(f"{animal} {date}: cam1 / cam4 production output missing")
    rig = _rig(rv, animal, date)
    cg = cg or _cgroup(rv)
    S, info = P3.session_3d(cg, rig, c1[0], c1[1], c4[0], c4[1])
    if side_bundle:
        side = {c: side_windows(rv, side_bundle, animal, date, c) for c in SIDE}
        if any(v is None for v in side.values()):
            raise FileNotFoundError(f"{animal} {date}: side bundle {side_bundle} incomplete")
    else:
        side = side_session(rv, animal, date, preds_dir, clips_dir, c4[0])
    meds = {"cam1": info["med1"], "cam4": info["med4"], **{c: SF.ruler_medians(side[c][1], side[c][0]) for c in SIDE}}
    g, _ = pose_all(cg, {c: meds[c] for c in ("cam1", "cam4", "cam2", "cam3")}, rig["geometry"])
    order = [c.get_name() for c in g.cameras]
    cams = {c.get_name(): c for c in g.cameras}
    scale = {c: px_per_mm(cams[c]) / px_per_mm(cams["cam4"]) for c in order}
    F, pts = four_camera_points(S, info, c1, c4, side, order, keep_all=bool(side_bundle))
    for c in SIDE:                                    # stage 1: the side tongue / jaw become their CLEANED measured points
        mouth = project(cams[c], np.zeros((1, 3)))[0]
        cl = side_clean(side[c][1], side[c][0], mouth, scale[c])
        k = F[f"i_{c}"].to_numpy()
        for part in ("tongue", "jaw"):
            pts[part][order.index(c)] = np.where((k >= 0)[:, None], cl[part][np.clip(k, 0, None)], np.nan)
    trial, t = F.trial_id.to_numpy(), F.t_ms.to_numpy(float)
    drop = droplet_mask(pts["tongue"], order, trial, scale)
    pts["tongue"][drop] = np.nan
    offsets = {"spout": part_offsets("spout"), "nose": session_nose_offsets(g, S, side), "jaw": {},
               "tongue": session_tongue_offsets(g, S, side)}
    parts = {}
    stats = {"scale": scale, "nose_offsets": offsets["nose"], "tongue_offsets": offsets["tongue"],
             "droplet_frames": int(drop.any(0).sum())}
    k14 = [order.index("cam1"), order.index("cam4")]
    for part in PARTS:
        gs = shifted(g, offsets[part])
        X, used, d = fuse_robust(gs, pts[part], REJECT_MM[part])
        rejected = np.isfinite(pts[part]).all(2) & ~used
        X, nv = fill_n(gs, X, pts[part], used, trial, t, FILL_MS[part])
        if part in ("tongue", "jaw"):
            X, qc = P3.qc_3d(X, part, trial, t)
            nv[~np.isfinite(X).all(1)] = 0
            stats.update(qc)
        parts[part] = {"X": X, "views": nv, "used": used, "dist": d, "g": gs}
        stats[part] = {"cam1+cam4": float(np.isfinite(pts[part][k14]).all(2).all(0).mean()),
                       "fused": float(np.isfinite(X).all(1).mean()), "one_view_fill": float((nv == -1).mean()),
                       **{f"rejected_{c}": int(rejected[j].sum()) for j, c in enumerate(order)}}
    return {"F": F, "order": order, "g": g, "pts": pts, "parts": parts, "stats": stats, "rig": rig, "side": side, "S": S}


def frames_table(Z: dict) -> pd.DataFrame:
    """The four-camera result as a `pose3d.session_3d`-style per-frame table, so `pose3d.lick_peaks` and the kinematics
    read it unchanged: trial_id, position, t_ms, stop_ms (+ t_daq_ms, src1, src4), per part ``{part}_ml/_ap/_dv`` mm,
    ``{part}_src`` (1 = >= 2 cameras, 2 = one camera + neighbours, 0 none), ``{part}_views`` (cameras behind the point;
    -1 = one-view fill), ``{part}_rep1`` / ``_rep4`` (px, cam1 / cam4, where used) and ``{part}_rep_<cam>`` for all four."""
    F, order = Z["F"], Z["order"]
    keep = [c for c in ("trial_id", "position", "t_ms", "stop_ms", "t_daq_ms", "src1", "src4") if c in F]
    T = F[keep].reset_index(drop=True).copy()
    for part, P in Z["parts"].items():
        X, nv = P["X"], P["views"]
        for k, ax in enumerate(("ml", "ap", "dv")):
            T[f"{part}_{ax}"] = X[:, k]
        T[f"{part}_views"] = nv
        T[f"{part}_src"] = np.where(nv >= 2, 1, np.where(nv == -1, 2, 0))
        pts = Z["pts"][part]
        for j, cam in enumerate(P["g"].cameras):
            r = np.linalg.norm(project(cam, X) - pts[j], axis=1)
            r[~P["used"][j]] = np.nan
            T[f"{part}_rep_{order[j]}"] = r
        T[f"{part}_rep1"], T[f"{part}_rep4"] = T[f"{part}_rep_cam1"], T[f"{part}_rep_cam4"]
    return T
