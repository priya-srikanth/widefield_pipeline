"""Per-frame 3-D pose from cam1 (bottom) + cam4 (front): triangulated tongue / jaw / nose / spout in mouth-centred mm.

    from wfield_local import pose3d as P3
    S = P3.session_3d(cg, rig, ix1, flat1, ix4, flat4)       # one row per paired frame
    L = P3.lick_peaks(S, rig)                                # one row per 3-D lick: reach, angle vs the target spout

HOW (Priya 2026-10-19: "build the per-frame 3-D step"). aniposelib's ``CameraGroup.triangulate`` (undistort + DLT) with
the 09-11 rig calibration -- the call `spout_world.apparatus_mouth` already makes for the spouts, NOT anipose's filter /
optimisation pipeline (two cameras, no rigid limbs: its constraints add little; multi-camera EKS is the next step if
the tips prove noisy). Then `spout_world`'s per-session similarity frame (spout medians fitted to the commanded layout)
maps every point to (ml, ap, dv) mm: + mouse right, + out of the mouth toward the spouts, + up; origin = the stage's
mouth set point.

1. PAIRING. Both cameras' windows are cut on the template-aligned DAQ clock on the same 4 ms grid (99.99 % of frames
   match exactly, 10-18), so frames pair on (trial_id, t_ms) with an exact join -- no interpolation.
2. CLEANING FIRST, per camera, with that camera's rules (`orofacial_clean.clean_windows`, cam1 / cam4 params): tongue
   and jaw are triangulated only where BOTH cameras have a MEASURED (FILL_NONE) point, so no interpolated point enters.
   Nose and spout (slow) use the raw point at likelihood >= ``STATIC_LK`` in both.
3. JUNE cam4 is scaled by `orofacial_camera.scale_factor` about cam4's PRINCIPAL POINT before cleaning (POSE_MODELS D5:
   the origin centre v36 uses moves the cam1 mouth ~20 px; the centre does not matter for 2-D measures, only here).
4. REPROJECTION ERROR per camera is kept on every point (``rep1`` / ``rep4`` px). It is the test of whether the two
   cameras' "tongue tip" is the same physical point: cam1 sees the tip from below, cam4 from the front.
5. KNOWN CONSTANT OFFSET. The frame is fitted to the LABELLED spout point (cam1 bottom edge, cam4 top front edge), a
   fixed vector from the commanded tip; the fit puts it into the translation, so every body point carries the same
   small constant shift. Differences, angles between licks and pre vs post comparisons are unaffected.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STATIC_LK = 0.9            # nose / spout: raw points, both cameras at least this confident
CLEANED = ("tongue", "jaw")
STATIC = ("nose", "spout")
KEY = ["trial_id", "t_ms"]


def pair(ix1: pd.DataFrame, ix4: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Row numbers (i1, i4) of frames present in both cameras, matched exactly on (trial_id, t_ms)."""
    a = ix1[KEY].reset_index().rename(columns={"index": "i1"})
    b = ix4[KEY].reset_index().rename(columns={"index": "i4"})
    m = a.merge(b, on=KEY, how="inner", validate="one_to_one")
    return m.i1.to_numpy(), m.i4.to_numpy()


def scale_june_cam4(flat: pd.DataFrame, factor: float, centre) -> pd.DataFrame:
    """x, y of every part x ``factor`` about ``centre`` (cam4 principal point), likelihoods untouched."""
    out = flat.copy()
    for c in [c for c in flat.columns if c.endswith("_x")]:
        out[c] = centre[0] + factor * (flat[c].to_numpy(float) - centre[0])
    for c in [c for c in flat.columns if c.endswith("_y")]:
        out[c] = centre[1] + factor * (flat[c].to_numpy(float) - centre[1])
    return out


def cleaned_xy(flat: pd.DataFrame, ix: pd.DataFrame, part: str, cam: str, mouth) -> tuple[np.ndarray, np.ndarray]:
    """(xy (n, 2) absolute image px, NaN where not MEASURED) for ``part`` after that camera's cleaning."""
    from wfield_local import orofacial_clean as oc
    cols = [c for c in flat.columns if c.startswith(("tongue_", "jaw_"))]
    c, pos, _ = oc.clean_windows(flat[cols].reset_index(drop=True), ix.reset_index(drop=True), part,
                                 x0y0=tuple(mouth), cam=cam)
    x, y, fm = c.x_final[pos] + c.X0, c.y_final[pos] + c.Y0, c.fill_method[pos]
    ok = (fm == oc.FILL_NONE) & np.isfinite(x) & np.isfinite(y)
    xy = np.where(ok[:, None], np.c_[x, y], np.nan)
    return xy, ok


def _raw_xy(flat: pd.DataFrame, part: str, rows: np.ndarray) -> np.ndarray:
    """Raw (x, y) at ``rows``, NaN below ``STATIC_LK`` (nose / spout)."""
    xy = flat[[f"{part}_x", f"{part}_y"]].to_numpy(float)[rows]
    return np.where((flat[f"{part}_likelihood"].to_numpy(float)[rows] >= STATIC_LK)[:, None], xy, np.nan)


def triangulate(cg,xy1: np.ndarray, xy4: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(X (n, 3) calibration units, reprojection px (n, 2) per camera); NaN where either view is missing."""
    n = len(xy1)
    X, rep = np.full((n, 3), np.nan), np.full((n, 2), np.nan)
    ok = np.isfinite(xy1).all(1) & np.isfinite(xy4).all(1)
    if ok.any():
        pts = np.stack([xy1[ok], xy4[ok]])                       # (2 cams, m, 2), order = cg cameras
        Xo = cg.triangulate(pts, undistort=True)
        X[ok] = Xo
        rep[ok] = np.linalg.norm(cg.project(Xo) - pts, axis=2).T
    return X, rep


def session_3d(cg, rig: dict, ix1: pd.DataFrame, flat1: pd.DataFrame, ix4: pd.DataFrame, flat4: pd.DataFrame,
               june_factor: float | None = None) -> tuple[pd.DataFrame, dict]:
    """Per paired frame: trial_id, position, t_ms, stop_ms and per part ``{part}_ml/_ap/_dv`` mm + ``{part}_rep1/_rep4``
    px. ``flat1`` / ``flat4`` RAW image px (``{part}_x/_y/_likelihood``), rows aligned with ``ix1`` / ``ix4``.
    ``june_factor``: scale cam4 about its principal point first. Returns (frames, info) -- info carries the world fit,
    the apparatus mouth in each camera and the spout medians used."""
    from wfield_local import spout_frame as SF
    from wfield_local import spout_world as SW
    cg = cg.subset_cameras_names(["cam1", "cam4"])
    if june_factor:
        K = cg.cameras[1].get_camera_matrix()
        flat4 = scale_june_cam4(flat4, june_factor, (K[0, 2], K[1, 2]))
    m1, m4 = SF.ruler_medians(flat1, ix1), SF.ruler_medians(flat4, ix4)
    am = SW.apparatus_mouth({"cam1": m1, "cam4": m4}, rig["geometry"], cg)
    world = am["world"]
    # each camera cleans around its own decided mouth: cam1 = L/R spout-line intersection (D6), cam4 = spout-line origin
    mouth1, mouth4 = SF.from_medians_lr(m1).origin, SF.from_medians(m4).origin
    i1, i4 = pair(ix1, ix4)
    out = ix1.iloc[i1][["trial_id", "position", "t_ms", "stop_ms"]].reset_index(drop=True)
    out["src1"], out["src4"] = ix1.src_frame.to_numpy()[i1], ix4.src_frame.to_numpy()[i4]     # video frame per camera
    for name, ix, ii in (("video1", ix1, i1), ("video4", ix4, i4)):                             # split days only
        if "video" in ix:
            out[name] = ix.video.to_numpy()[ii]
    for part in CLEANED + STATIC:
        if part in CLEANED:
            a, _ = cleaned_xy(flat1, ix1, part, "cam1", mouth1)
            b, _ = cleaned_xy(flat4, ix4, part, "cam4", mouth4)
            a, b = a[i1], b[i4]
        else:
            a, b = _raw_xy(flat1, part, i1), _raw_xy(flat4, part, i4)
        X, rep = triangulate(cg, a, b)
        W = world.to_world(X)
        for k, ax in enumerate(("ml", "ap", "dv")):
            out[f"{part}_{ax}"] = W[:, k]
        out[f"{part}_rep1"], out[f"{part}_rep4"] = rep[:, 0], rep[:, 1]
    info = {"world": world, "mouth_px": am["mouth_px"], "reproj_spout_px": am["reproj_px"], "med1": m1, "med4": m4,
            "n_cam1": len(ix1), "n_cam4": len(ix4), "n_paired": len(i1)}
    return out, info


def lick_peaks(S: pd.DataFrame, rig: dict, min_frames: int = 3, max_rep_px: float | None = None) -> pd.DataFrame:
    """One row per 3-D lick: a run of >= ``min_frames`` consecutive frames (4 ms apart, one trial, cue .. trial stop)
    with a triangulated tongue. ``max_rep_px`` drops frames whose tongue reprojection exceeds it in either camera.

    AIM IS MEASURED AGAINST THE TARGET SPOUT, NOT FROM THE ORIGIN (10-19, PS93 0814). The origin is the stage's mouth SET
    POINT, not where the tongue leaves the mouth (the resting jaw sits ~2 mm forward and down, at the close_center
    spout), so angles from it came out 11-48 deg off target with the wrong structure. Origin-free measures instead:
      * ``miss_mm`` -- closest 3-D approach of the tip to the commanded target spout tip during the run (+ its frame);
      * ``lat_err_mm`` at that frame -- (tongue ml - spout ml), + = OUTWARD of the spout (mouse left at the centre),
        the sign of the cam1 2-D readout (`scripts.cam1_lick_aim`);
      * ``reach_mm`` / ``ml, ap, dv`` at the frame farthest from the origin (the extension peak; the tongue reaches
        1.5-3 mm BELOW the spout tip there -- the hanging drop -- so dv is part of every lick).
    The commanded tip = the triangulated spout median to ~0.05 mm (the frame is fitted to it)."""
    from wfield_local import spout_world as SW
    t, tid = S.t_ms.to_numpy(float), S.trial_id.to_numpy()
    P = S[["tongue_ml", "tongue_ap", "tongue_dv"]].to_numpy(float)
    have = np.isfinite(P).all(1) & (t >= 0) & (t <= S.stop_ms.to_numpy(float))
    if max_rep_px is not None:
        have &= (S.tongue_rep1.to_numpy(float) <= max_rep_px) & (S.tongue_rep4.to_numpy(float) <= max_rep_px)
    tgt = {p: SW.commanded_world(rig["geometry"], p) for p in SW.POSITIONS}
    rows, i, n = [], 0, len(S)
    while i < n:
        if not have[i]:
            i += 1
            continue
        j = i + 1
        while j < n and have[j] and tid[j] == tid[i] and abs(t[j] - t[j - 1] - 4.0) < 0.5:
            j += 1
        pos = str(S.position.iloc[i])
        if j - i >= min_frames and pos in tgt:
            r = np.linalg.norm(P[i:j], axis=1)
            k = i + int(np.argmax(r))
            g = tgt[pos]
            dm = np.linalg.norm(P[i:j] - g, axis=1)
            m = i + int(np.argmin(dm))
            out_sign = np.sign(g[0]) if abs(g[0]) > 1e-6 else -1.0        # centre: + = mouse left
            rows.append({"trial_id": int(tid[i]), "position": pos, "t_onset_ms": t[i], "t_peak_ms": t[k],
                         "n_frames": j - i, "reach_mm": float(r[k - i]),
                         "ml": P[k, 0], "ap": P[k, 1], "dv": P[k, 2],
                         "miss_mm": float(dm[m - i]), "t_closest_ms": t[m],
                         "lat_err_mm": float(out_sign * (P[m, 0] - g[0])),
                         "ap_err_mm": float(P[m, 1] - g[1]), "dv_err_mm": float(P[m, 2] - g[2]),
                         "rep1_px": float(S.tongue_rep1.iloc[k]), "rep4_px": float(S.tongue_rep4.iloc[k])})
        i = j
    return pd.DataFrame(rows)
