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
4. CAMERA POSES PER SESSION FROM THE COMMANDED SPOUTS (`pnp_group`, the default since 10-19): the 09-11 extrinsics put
   every part 6-10 px apart between cameras (a sideways error + day-to-day drift); PnP on the six known spout positions
   brings that to ~1 px and gives stage mm directly. The old route (09-11 extrinsics + `disparity_shift` + the
   `spout_world` similarity fit) stays as ``method="calib"``.
   REPROJECTION ERROR per camera is kept on every point (``rep1`` / ``rep4`` px). It is the test of whether the two
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


PAIR_TOL_MS = 2.0      # half a frame


def pair(ix1: pd.DataFrame, ix4: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Row numbers (i1, i4) of frames present in both cameras.

    With ``t_daq_ms`` in both indexes (each frame's camera-timestamp time from the cue, `o2_inference.build_windows`,
    10-20): nearest frame within ``PAIR_TOL_MS``, one-to-one, per trial -- so DROPPED frames (PS92 0908) leave gaps
    instead of sliding every later frame of the trial onto the wrong partner (Priya: align on the sync pulse, the session
    is good for its first ~104 min). Otherwise (trial clips): exact match on (trial_id, t_ms). On drop-free recordings
    both give the same pairs."""
    if "t_daq_ms" not in ix1 or "t_daq_ms" not in ix4:
        a = ix1[KEY].reset_index().rename(columns={"index": "i1"})
        b = ix4[KEY].reset_index().rename(columns={"index": "i4"})
        m = a.merge(b, on=KEY, how="inner", validate="one_to_one")
        return m.i1.to_numpy(), m.i4.to_numpy()
    a = pd.DataFrame({"trial_id": ix1.trial_id.to_numpy(), "t": ix1.t_daq_ms.to_numpy(float), "i1": np.arange(len(ix1))})
    b = pd.DataFrame({"trial_id": ix4.trial_id.to_numpy(), "t": ix4.t_daq_ms.to_numpy(float), "i4": np.arange(len(ix4)),
                      "t4": ix4.t_daq_ms.to_numpy(float)})
    m = pd.merge_asof(a.sort_values("t"), b.sort_values("t"), on="t", by="trial_id", tolerance=PAIR_TOL_MS,
                      direction="nearest").dropna(subset=["i4"])
    m["d"] = (m.t - m.t4).abs()
    m = m.sort_values("d").drop_duplicates("i4").sort_values("i1")          # one-to-one: the closest cam1 frame wins
    return m.i1.to_numpy(int), m.i4.to_numpy(int)


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


def disparity_shift(cg, m1: dict, m4: dict) -> tuple[float, float]:
    """Per-session horizontal (x) correction per camera, px, from the parked spout medians: the median of (reprojected -
    observed) x over positions. Measured 10-19: EVERY part (spout, nose, jaw, tongue) shows the same cam1 +6..+10 / cam4
    -6..-10 px, horizontal only, varying by session -- a small ml error in the 09-11 extrinsics plus day-to-day camera
    drift. NOT the spout's edge labels (cam1 inferior, cam4 superior edge): they differ in dv / ap, which lies along the
    epipolar direction (cameras below and in front) and is absorbed into depth with no residual. Adding these shifts to
    each camera's x before triangulating puts both views on the same ray pair; the 3-D point stays (to first order) where
    it was. Rigid with no tongue involved: apparatus only."""
    from wfield_local import spout_world as SW
    pos = [p for p in SW.POSITIONS if p in m1 and p in m4]
    pts = np.stack([[np.asarray(m1[p][:2], float) for p in pos], [np.asarray(m4[p][:2], float) for p in pos]])
    r = cg.project(cg.triangulate(pts, undistort=True)) - pts
    return float(np.median(r[0, :, 0])), float(np.median(r[1, :, 0]))


def shift_x(flat: pd.DataFrame, dx: float) -> pd.DataFrame:
    out = flat.copy()
    for c in [c for c in flat.columns if c.endswith("_x")]:
        out[c] = flat[c].to_numpy(float) + dx
    return out


#: The spout ROD's diameter, measured 10-19 on parked frames (horizontal profile just behind the rounded end, edges = the
#: strongest opposite-sign gradients, px -> mm from each camera's PnP pose): cam1 1.23-1.24 mm (IQR +-0.03), cam4 1.27-1.31
#: (low contrast), PS92 0806 / 0807 / 0818. cam1's value is used.
ROD_DIAMETER_MM = 1.24
#: Where each camera's spout KEYPOINT sits relative to the commanded point (world ml, ap, dv mm). The mouth zero was set with
#: the UPPER spout tip at the mouth (Priya 10-19: "mouth zero is relative to the upper spout tip, i.e. cam4"), and the
#: commanded positions move that point; cam4 labels the SUPERIOR edge (= the commanded point) and cam1 the INFERIOR edge,
#: one rod diameter below (Priya: "place cam1 and cam4 that much apart in vertical space").
EDGE_OFFSET_MM = {"cam4": (0.0, 0.0, 0.0), "cam1": (0.0, 0.0, -ROD_DIAMETER_MM)}


#: One-camera fill limit per part (ms): the nearest both-camera frame must be this close (10-19). Tongue 24 ms (the cam1 gap
#: rule; held-out error 0.13-0.23 mm). JAW 200 ms (Priya 10-19 "gap test"): the jaw moves slowly -- held-out error at 200 ms
#: gaps median 0.04-0.07 mm, p90 0.20-0.34 (24 ms: 0.03-0.05 / 0.10), PS92 0807 + 0818; 24 ms filled only 1-3 % of the
#: one-camera jaw frames, 200 ms 13-23 % (the rest are long stretches -- e.g. the rod over the chin in cam4 -- not filled).
FILL_GAP_MS = {"tongue": 24.0, "jaw": 200.0}


def pnp_group(cg, med_by_cam: dict, geometry: dict, edge_offset: dict | None = EDGE_OFFSET_MM):
    """A copy of ``cg`` whose cameras are POSED PER SESSION from the commanded spout positions (Priya 10-19: "use the
    ground-truth spout positions"): for each camera, cv2 SQPnP + Levenberg-Marquardt on the six parked spout medians
    (image px) against `spout_world.commanded_world` (mm, mouth set point = origin). Intrinsics + distortion stay the
    09-11 solve's. Triangulating with it gives (ml, ap, dv) mm directly -- no similarity fit, no 09-11 extrinsics, so the
    09-11 sideways error and day-to-day camera drift (`disparity_shift`) are gone by construction.
    Measured 10-19 (PS92 0807 / 0818): spout reprojection 0.2-1.8 px per camera (09-11 extrinsics: 6-10), nose 0.7-3.0 px
    and jaw 1.0 px between cameras (8-10), tongue 3.5 (~10). Camera centres: cam4 ~160 mm in front, slightly above; cam1
    ~110 mm below and ~105 mm forward.
    THE EDGE LABELS (10-19): cam1 labels the rod's INFERIOR edge, cam4 its SUPERIOR edge, and the commanded point (the
    mouth zero) is the UPPER tip. ``edge_offset`` (default `EDGE_OFFSET_MM`) puts each camera's keypoint where it really
    is: cam4 at the commanded point, cam1 one rod diameter below. Without it (None) each camera absorbs its own offset
    into its pose and every body point carries a constant shift of up to ~one rod radius (mostly depth; reprojection
    cannot see it, the cameras being below and in front) -- harmless for changes and comparisons, wrong for absolute
    distances to the spout.
    Returns (group, {cam: per-position reprojection px})."""
    import copy

    import cv2

    from wfield_local import spout_world as SW
    pos = [p for p in SW.POSITIONS if all(p in med_by_cam[c] for c in med_by_cam)]
    if len(pos) < 5:
        raise ValueError(f"only {len(pos)} spout positions in both cameras ({pos}); PnP needs 5")
    obj = np.array([SW.commanded_world(geometry, p) for p in pos], float)
    g = copy.deepcopy(cg)
    err = {}
    for cam in g.cameras:
        img = np.array([med_by_cam[cam.get_name()][p][:2] for p in pos], float)
        o = obj + np.asarray((edge_offset or {}).get(cam.get_name(), (0.0, 0.0, 0.0)), float)
        K, D = cam.get_camera_matrix(), np.asarray(cam.get_distortions(), float)
        _, rvec, tvec = cv2.solvePnP(o, img, K, D, flags=cv2.SOLVEPNP_SQPNP)
        rvec, tvec = cv2.solvePnPRefineLM(o, img, K, D, rvec, tvec)
        cam.set_rotation(rvec.ravel())
        cam.set_translation(tvec.ravel())
        pr, _ = cv2.projectPoints(o, rvec, tvec, K, D)
        err[cam.get_name()] = dict(zip(pos, np.linalg.norm(pr[:, 0] - img, axis=1)))
    return g, err


def _identity_world(g, med_by_cam: dict, geometry: dict, edge_offset: dict | None = EDGE_OFFSET_MM):
    """`spout_world.SpoutWorld` for a PnP group (scale 1, identity): residuals / distance check of the triangulated spouts.

    With ``edge_offset`` the two keypoints are DIFFERENT physical points (cam1 the inferior edge, one rod diameter below
    cam4's superior edge = the commanded point), so they are triangulated with each camera's pose shifted by its offset
    (projection of X + o == projection of X with t' = t + R o): the result is the commanded point the keypoints imply.
    Without the shift the triangulated 'spout' is a blend of the two edges and the check fails (10-19: every session read
    > 0.2 mm and was skipped downstream)."""
    import copy

    import cv2

    from wfield_local import spout_world as SW
    pos = [p for p in SW.POSITIONS if all(p in med_by_cam[c] for c in med_by_cam)]
    pts = np.stack([[np.asarray(med_by_cam[c.get_name()][p][:2], float) for p in pos] for c in g.cameras])
    gs = copy.deepcopy(g)
    for cam in gs.cameras:
        o = np.asarray((edge_offset or {}).get(cam.get_name(), (0.0, 0.0, 0.0)), float)
        if o.any():
            M = cam.get_extrinsics_mat()
            cam.set_translation(M[:3, 3] + M[:3, :3] @ o)
            cam.set_rotation(cv2.Rodrigues(M[:3, :3])[0].ravel())
    X = gs.triangulate(pts, undistort=True)
    med = dict(zip(pos, X))
    res = {p: float(np.linalg.norm(med[p] - SW.commanded_world(geometry, p))) for p in pos}
    return SW.SpoutWorld(1.0, np.eye(3), np.zeros(3), res, SW.distance_check(med, geometry), {}, {})


def camera_rays(cam, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(centre (3,), unit directions (n, 3)) of the rays through image points ``xy`` (n, 2), in ``cam``'s world
    (calibration space; stage mm for a `pnp_group` camera). Undistorted with the camera's own model."""
    import cv2
    K, D = cam.get_camera_matrix(), np.asarray(cam.get_distortions(), float)
    nrm = cv2.undistortPoints(np.asarray(xy, float).reshape(-1, 1, 2), K, D).reshape(-1, 2)
    M = cam.get_extrinsics_mat()
    R, t = M[:3, :3], M[:3, 3]
    U = (R.T @ np.c_[nrm, np.ones(len(nrm))].T).T
    return -R.T @ t, U / np.linalg.norm(U, axis=1, keepdims=True)


#: parts whose LONG one-camera stretches (beyond the fill limit) are placed with the rest-depth anchor. EMPTY since
#: 10-20 evening (Priya: "should we just give up on filling in occluded jaw like this?" -- yes): the anchor put OPEN
#: one-camera jaws 0.5-1.4 mm off (opening underestimated, cam4-only -0.55 mm; check sheet C, cam1 + anchor too high),
#: and a depth model fitted on both-camera frames (`scripts/pose3d_jaw_depth_model_check.py`: open-jaw error 0.25-0.31
#: mm held out) still missed at close_center, where both cameras almost never see an open jaw (extrapolation).
#: One-camera jaw frames beyond the 200 ms fill stay NaN: centre-spout 3-D jaw ~84 % coverage, missing mostly OPEN
#: close_center jaws -> close_center jaw OPENING is biased low (leave it out of opening comparisons, as cam4 does).
#: FAR_DIR / CLOSE_DIR (side spouts, ~0 % anchored) unaffected. `anchor_single_view` kept for the record / tests.
ANCHOR_PARTS: tuple = ()


def _cam_depth(cam, X: np.ndarray) -> np.ndarray:
    M = cam.get_extrinsics_mat()
    return (X @ M[:3, :3].T + M[:3, 3])[:, 2]


def anchor_single_view(X: np.ndarray, a: np.ndarray, b: np.ndarray, cg, trial: np.ndarray, t: np.ndarray,
                       src: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One-camera frames still without 3-D after `fill_single_view` -> the seen camera's ray at the CAMERA-FRAME DEPTH
    of the trial's resting point (median both-camera point before the cue; the session's if the trial has none).
    ``src`` 4 = cam1 + anchor, 5 = cam4 + anchor. For the JAW (Priya 10-20: recover the centre-spout jaw -- the rod is
    end-on in cam1's view at both centre positions, and at close_center it also covers the chin in cam4). Held-out
    (`scripts/pose3d_jaw_anchor_check.py`, PS92 0807 / 0818 / 0915, PS93 0807; truth = both-camera jaw on side-spout
    trials, one camera hidden): ml error median 0.03-0.05 mm either camera; dv 0.02-0.04 (cam4) / 0.07-0.15 (cam1); ap
    0.15-0.39 (cam4 alone cannot see it) / 0.06-0.14 (cam1); per-trial lr_wide r 0.95-0.99, FAR_DIR within 0.02 mm."""
    X, src = X.copy(), src.copy()
    pre = (src == 1) & (t < 0)
    if not pre.any():
        return X, src
    Xr = np.tile(np.nanmedian(X[pre], axis=0), (len(X), 1))
    for k in np.unique(trial[pre]):
        Xr[trial == k] = np.median(X[pre & (trial == k)], axis=0)
    fa, fb = np.isfinite(a).all(1), np.isfinite(b).all(1)
    left = ~np.isfinite(X).all(1)
    for ci, xy, m, code in ((0, a, left & fa & ~fb, 4), (1, b, left & fb & ~fa, 5)):
        if not m.any():
            continue
        cam = cg.cameras[ci]
        C, U = camera_rays(cam, xy[m])
        R = cam.get_extrinsics_mat()[:3, :3]
        sc = _cam_depth(cam, Xr[m]) / (U @ R.T)[:, 2]
        X[m] = C + sc[:, None] * U
        src[m] = code
    return X, src


def fill_single_view(X: np.ndarray, a: np.ndarray, b: np.ndarray, cg, trial: np.ndarray, t: np.ndarray,
                     max_gap_ms: float = 24.0, interp: str = "linear",
                     max_ray_miss_mm: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    """Fill frames that ONE camera measured (Priya 10-19: "use the known geometry to resolve occlusions in one view").

    About 40 % of tongue frames are seen by one camera only (PS92 0807: 18 % cam1-only, 25 % cam4-only). That camera
    fixes the point to its RAY exactly (cam4 alone: ml and dv; cam1 alone: ml and ap -- so ml, the aim coordinate, is
    known either way); the depth along the ray comes from the nearest BOTH-camera frames of the same trial within
    ``max_gap_ms`` (default 24 ms, the cam1 gap rule, D8): their 3-D points interpolated in time (or the one side
    within reach), then moved to the closest point on the ray. ``X`` (n, 3) triangulated (NaN where not both),
    ``a`` / ``b`` the cam1 / cam4 measured 2-D (NaN = not measured), rows in time order within each trial.
    ``interp``: how the neighbours' 3-D points are interpolated when BOTH sides are within reach -- "linear" (the two
    nearest), "pchip" / "akima" (a shape-preserving spline through every both-camera frame of the trial, evaluated only
    inside two-sided gaps). One-sided gaps always hold the nearest point (no extrapolation). A filled point is DROPPED
    when the seen camera's ray passes more than ``max_ray_miss_mm`` (3 mm) from the neighbours' estimate: that camera's
    2-D point then disagrees with the 3-D track (paw-in-face frames, 10-20 mm out). NOT 1 mm (first try, 10-19): the
    interpolated estimate is routinely > 1 mm off the ray during a fast lick (~1 mm per frame) IN THE COORDINATES THE SEEN
    CAMERA MEASURES -- exactly what snapping onto the ray corrects -- and 1 mm dropped 41-51 % of cam4-only fills
    (PS92 0807 / 0607); 3 mm keeps 97.6-99.8 %.
    Returns (X filled, source code per frame: 0 none, 1 both cameras, 2 cam1 + neighbours, 3 cam4 + neighbours)."""
    X = np.array(X, float)
    both = np.isfinite(X).all(1)
    has1, has4 = np.isfinite(a).all(1), np.isfinite(b).all(1)
    src = np.where(both, 1, np.where(has1 & ~has4, 2, np.where(has4 & ~has1, 3, 0)))
    tt = pd.Series(np.where(both, t, np.nan))
    tp, tn = tt.groupby(trial).ffill().to_numpy(), tt.groupby(trial).bfill().to_numpy()
    Xb = pd.DataFrame(np.where(both[:, None], X, np.nan))
    Xp, Xn = Xb.groupby(trial).ffill().to_numpy(), Xb.groupby(trial).bfill().to_numpy()
    dp, dn = t - tp, tn - t
    okp, okn = dp <= max_gap_ms, dn <= max_gap_ms                     # NaN compares False
    with np.errstate(invalid="ignore", divide="ignore"):
        w = (dp / (dp + dn))[:, None]
    est = np.where((okp & okn)[:, None], Xp + w * (Xn - Xp),
                   np.where(okp[:, None], Xp, np.where(okn[:, None], Xn, np.nan)))
    if interp != "linear":
        from scipy.interpolate import Akima1DInterpolator, PchipInterpolator
        kind = {"pchip": PchipInterpolator, "akima": Akima1DInterpolator}[interp]
        two = okp & okn & ~both
        for tr in np.unique(trial[two]):
            kb = both & (trial == tr)
            if kb.sum() < 3:
                continue                                             # too few knots: keep the linear estimate
            kt = two & (trial == tr)
            est[kt] = kind(t[kb], X[kb], axis=0)(t[kt])
    for code, xy, cam in ((2, a, cg.cameras[0]), (3, b, cg.cameras[1])):
        k = (src == code) & np.isfinite(est).all(1)
        if k.any():
            C, U = camera_rays(cam, xy[k])
            Pk = C + ((est[k] - C) * U).sum(1)[:, None] * U
            far = np.linalg.norm(Pk - est[k], axis=1) > max_ray_miss_mm          # the camera disagrees with its neighbours
            Pk[far] = np.nan
            X[k] = Pk
    src[(src > 1) & ~np.isfinite(X).all(1)] = 0
    return X, src


#: Plausibility gates on the 3-D tongue / jaw (10-19, after a lick path 8-36 mm out on PS92 0607). Implausible points are
#: rare (0.16-0.26 % of both-camera tongue points) and reprojection does NOT reliably flag them (0607: 22 % above 15 px,
#: median 4.4 px): an error along the depth direction (the cameras sit below and in front) still triangulates cleanly.
#: Real 3-D steps between 4 ms frames: median 0.32 mm, p99 1.6, p99.9 2.3 (PS92 0806).
QC_MAX_MM = {"tongue": 8.0, "jaw": 5.0}      # tongue: from the mouth zero (normal p99 6.6 mm); jaw: from its session median
QC_SPIKE_MM = 1.5                            # a single frame > 1.5 mm out AND back, its neighbours within 0.75 mm
#: Tongue BEHIND the mouth zero (Priya 10-19: "impossible (negative from mouth)"). ap < 0 is common and smooth to ~-1 mm
#: (13-20 % of points; p1 -0.9..-1.1: a tip emerging under the upper lip, which the mouth zero's spout touched); beyond
#: -2 mm is a tail (0.04-0.09 % on 0806 / 0807 / 0818, 0.21 % on 0607 whose p0.1 is -4.4 mm) -> dropped.
QC_TONGUE_AP_MIN_MM = -2.0
#: SESSION gates (from `scripts/pose3d.py`'s summary_*.csv): the spout fit (PS92 0605: 1.4 mm, cam4 template failure) and
#: the median TONGUE reprojection -- 3.1-4.0 px on every good session; a camera-timing failure pairs frames from different
#: moments, so rigid parts still fit but moving ones do not (PS92 0908, the frame-drop cliff in EXPERIMENT_ERRORS:
#: 8.5 px, 35k 3-D tongue frames vs ~200k).
SESSION_MAX_FIT_RESID_MM = 0.2
SESSION_MAX_TONGUE_REPROJ_PX = 6.0


def failed_sessions(d3) -> dict:
    """{"<animal>_<date>": reason} for sessions whose 3-D fails a session gate (read from ``d3 / summary_*.csv``)."""
    from pathlib import Path
    S = pd.concat([pd.read_csv(f) for f in Path(d3).glob("summary_*.csv")], ignore_index=True)
    out = {}
    for r in S.itertuples():
        fit, rep = getattr(r, "fit_resid_mm_max", np.nan), getattr(r, "tongue_rep1_med_px", np.nan)
        if np.isfinite(fit) and fit > SESSION_MAX_FIT_RESID_MM:
            out[r.session] = f"spout fit residual {fit:.2f} mm > {SESSION_MAX_FIT_RESID_MM}"
        elif np.isfinite(rep) and rep > SESSION_MAX_TONGUE_REPROJ_PX:
            out[r.session] = f"tongue reprojection {rep:.1f} px > {SESSION_MAX_TONGUE_REPROJ_PX} (camera timing)"
    return out


def qc_3d(X: np.ndarray, part: str, trial: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, dict]:
    """Drop implausible 3-D points (NaN): beyond `QC_MAX_MM` and single-frame out-and-back spikes. Returns (X, counts)."""
    X = np.array(X, float)
    ok = np.isfinite(X).all(1)
    ref = np.zeros(3) if part == "tongue" else np.nanmedian(X[ok], axis=0) if ok.any() else np.zeros(3)
    far = ok & (np.linalg.norm(X - ref, axis=1) > QC_MAX_MM.get(part, np.inf))
    behind = ok & (X[:, 1] < QC_TONGUE_AP_MIN_MM) & ~far if part == "tongue" else np.zeros(len(X), bool)
    X[behind] = np.nan
    X[far] = np.nan
    ok = np.isfinite(X).all(1)
    i = np.flatnonzero(ok[1:-1] & ok[:-2] & ok[2:] & (trial[2:] == trial[:-2])
                       & (np.abs(t[2:] - t[:-2] - 8.0) < 0.5)) + 1
    out_ = np.linalg.norm(X[i] - X[i - 1], axis=1)
    back = np.linalg.norm(X[i + 1] - X[i], axis=1)
    span = np.linalg.norm(X[i + 1] - X[i - 1], axis=1)
    spike = i[(out_ > QC_SPIKE_MM) & (back > QC_SPIKE_MM) & (span < QC_SPIKE_MM / 2)]
    X[spike] = np.nan
    return X, {f"{part}_qc_far": int(far.sum()), f"{part}_qc_behind": int(behind.sum()), f"{part}_qc_spike": int(len(spike))}


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
               june_factor: float | None = None, method: str = "pnp",
               fill_gap_ms: dict | float | None = None, fill_interp: str = "akima") -> tuple[pd.DataFrame, dict]:
    """Per paired frame: trial_id, position, t_ms, stop_ms and per part ``{part}_ml/_ap/_dv`` mm + ``{part}_rep1/_rep4``
    px. ``flat1`` / ``flat4`` RAW image px (``{part}_x/_y/_likelihood``), rows aligned with ``ix1`` / ``ix4``.
    ``june_factor``: scale cam4 about its principal point first (cleaning thresholds are in Aug+ px).
    ``method``: "pnp" (default, 10-19) = cameras posed per session from the commanded spouts (`pnp_group`); "calib" =
    the 09-11 extrinsics + the per-session x shifts of `disparity_shift` (info ``shift_px``; 2-D points drawn on the raw
    video must subtract them) + the similarity fit -- kept for comparison. ``fill_gap_ms`` (None = `FILL_GAP_MS`, per part; a number = both parts): fill single-camera
    tongue / jaw frames (`fill_single_view`; ``{part}_src`` 1 both, 2 cam1, 3 cam4, 0 none; their rep columns stay NaN;
    jaw also 4 / 5 = cam1 / cam4 + rest-depth anchor, `anchor_single_view`);
    None = both cameras only. ``fill_interp`` (default "akima", 10-19): held-out 3-D error, PS92 0807 / 0818, linear ->
    Akima: 8 ms gaps 0.136-0.144 -> 0.127-0.137 mm, 16 ms 0.164-0.176 -> 0.145-0.162, 24 ms 0.220-0.272 -> 0.203-0.231
    (PCHIP in between); largest gain on long cam1-only gaps. Returns (frames, info) -- info carries the world fit,
    the apparatus mouth in each camera and the spout medians used."""
    from wfield_local import spout_frame as SF
    from wfield_local import spout_world as SW
    cg = cg.subset_cameras_names(["cam1", "cam4"])
    if june_factor:
        K = cg.cameras[1].get_camera_matrix()
        flat4 = scale_june_cam4(flat4, june_factor, (K[0, 2], K[1, 2]))
    m1, m4 = SF.ruler_medians(flat1, ix1), SF.ruler_medians(flat4, ix4)
    dx1 = dx4 = 0.0
    if method == "pnp":
        cg, err = pnp_group(cg, {"cam1": m1, "cam4": m4}, rig["geometry"])
        world = _identity_world(cg, {"cam1": m1, "cam4": m4}, rig["geometry"])
        mpx = cg.project(np.zeros((1, 3)))
        am = {"mouth_px": {"cam1": mpx[0, 0], "cam4": mpx[1, 0]},
              "reproj_px": {c: float(np.mean(list(e.values()))) for c, e in err.items()}}
    else:
        dx1, dx4 = disparity_shift(cg, m1, m4)
        flat1, flat4 = shift_x(flat1, dx1), shift_x(flat4, dx4)
        m1, m4 = SF.ruler_medians(flat1, ix1), SF.ruler_medians(flat4, ix4)
        am = SW.apparatus_mouth({"cam1": m1, "cam4": m4}, rig["geometry"], cg)
        world = am["world"]
    # each camera cleans around its own decided mouth: cam1 = L/R spout-line intersection (D6), cam4 = spout-line origin
    mouth1, mouth4 = SF.from_medians_lr(m1).origin, SF.from_medians(m4).origin
    i1, i4 = pair(ix1, ix4)
    # frame time = cam4's ``t_ms`` (the frame-index time the v36 cam4 lick windows use; identical to cam1's when nothing
    # dropped) + ``t_daq_ms`` (true time from the cue, camera timestamp) where the index carries it
    out = ix4.iloc[i4][["trial_id", "position", "t_ms", "stop_ms"]].reset_index(drop=True)
    if "t_daq_ms" in ix4:
        out["t_daq_ms"] = ix4.t_daq_ms.to_numpy(float)[i4]
    out["src1"], out["src4"] = ix1.src_frame.to_numpy()[i1], ix4.src_frame.to_numpy()[i4]     # video frame per camera
    for name, ix, ii in (("video1", ix1, i1), ("video4", ix4, i4)):                             # split days only
        if "video" in ix:
            out[name] = ix.video.to_numpy()[ii]
    qc = {}
    for part in CLEANED + STATIC:
        if part in CLEANED:
            a, _ = cleaned_xy(flat1, ix1, part, "cam1", mouth1)
            b, _ = cleaned_xy(flat4, ix4, part, "cam4", mouth4)
            a, b = a[i1], b[i4]
        else:
            a, b = _raw_xy(flat1, part, i1), _raw_xy(flat4, part, i4)
        X, rep = triangulate(cg, a, b)
        gap = (FILL_GAP_MS if fill_gap_ms is None else fill_gap_ms if isinstance(fill_gap_ms, dict)
               else {p_: fill_gap_ms for p_ in CLEANED}).get(part)
        if part in CLEANED and gap:
            X, src = fill_single_view(X, a, b, cg, out.trial_id.to_numpy(), out.t_ms.to_numpy(float), gap,
                                      fill_interp)
            if part in ANCHOR_PARTS:                       # long one-camera stretches: rest-depth anchor (src 4 / 5)
                X, src = anchor_single_view(X, a, b, cg, out.trial_id.to_numpy(), out.t_ms.to_numpy(float), src)
            out[f"{part}_src"] = src
        W = world.to_world(X)
        if part in CLEANED:
            W, cnt = qc_3d(W, part, out.trial_id.to_numpy(), out.t_ms.to_numpy(float))
            qc.update(cnt)
            if f"{part}_src" in out:
                out.loc[~np.isfinite(W).all(1), f"{part}_src"] = 0
        for k, ax in enumerate(("ml", "ap", "dv")):
            out[f"{part}_{ax}"] = W[:, k]
        out[f"{part}_rep1"], out[f"{part}_rep4"] = rep[:, 0], rep[:, 1]
    info = {"qc": qc, "method": method, "cg": cg, "shift_px": (dx1, dx4), "world": world, "mouth_px": am["mouth_px"], "reproj_spout_px": am["reproj_px"], "med1": m1, "med4": m4,
            "n_cam1": len(ix1), "n_cam4": len(ix4), "n_paired": len(i1)}
    return out, info


def lick_events(S: pd.DataFrame, max_gap_ms: float = 12.0, min_frames: int = 3) -> pd.DataFrame:
    """Lick EVENTS from the combined 3-D track (Priya 10-19: "define licks by the combined track to add in cam1's small
    licks"): runs of 3-D tongue frames (both cameras or one camera + neighbours, after QC) inside the response window
    (cue .. trial stop), gaps up to ``max_gap_ms`` bridged, >= ``min_frames`` frames with a point. Per event: on / off
    (first / last frame with a point), the 3-D peak (frame farthest from the mouth zero), ``lick_idx`` (order in the
    trial), ``n_3d``. Licks seen by ONE camera with no both-camera frame within 24 ms have no 3-D point and are not here
    (depth prior: to do)."""
    t, tid = S.t_ms.to_numpy(float), S.trial_id.to_numpy()
    P = S[["tongue_ml", "tongue_ap", "tongue_dv"]].to_numpy(float)
    have = np.isfinite(P).all(1) & (t >= 0) & (t <= S.stop_ms.to_numpy(float))
    k = np.flatnonzero(have)
    rows = []
    if not len(k):
        return pd.DataFrame(columns=["trial_id", "position", "on_ms", "off_ms", "t_ms", "lick_idx", "n_3d"])
    brk = np.r_[True, (tid[k[1:]] != tid[k[:-1]]) | (t[k[1:]] - t[k[:-1]] > max_gap_ms + 0.5)]
    starts = np.flatnonzero(brk)
    ends = np.r_[starts[1:], len(k)]
    for a, b in zip(starts, ends):
        kk = k[a:b]
        if len(kk) < min_frames:
            continue
        pk = kk[int(np.argmax(np.linalg.norm(P[kk], axis=1)))]
        rows.append({"trial_id": int(tid[kk[0]]), "position": str(S.position.iloc[kk[0]]), "on_ms": t[kk[0]],
                     "off_ms": t[kk[-1]], "t_ms": t[pk], "n_3d": len(kk)})
    E = pd.DataFrame(rows)
    E["lick_idx"] = E.groupby("trial_id").cumcount()
    return E


def lick_peaks(S: pd.DataFrame, rig: dict, min_frames: int = 3, max_rep_px: float | None = None) -> pd.DataFrame:
    """One row per 3-D lick: a run of >= ``min_frames`` consecutive frames (4 ms apart, one trial, cue .. trial stop)
    with a 3-D tongue (both cameras, or one camera + neighbours: `fill_single_view`; ``share_single_view`` per lick).
    ``max_rep_px`` drops BOTH-camera frames whose tongue reprojection exceeds it in either camera.

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
    src = S.tongue_src.to_numpy() if "tongue_src" in S else np.where(have, 1, 0)
    if max_rep_px is not None:                                  # single-camera frames have no reprojection error
        have &= (src != 1) | ((S.tongue_rep1.to_numpy(float) <= max_rep_px) & (S.tongue_rep4.to_numpy(float) <= max_rep_px))
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
                         "n_frames": j - i, "share_single_view": float(np.mean(src[i:j] > 1)),
                         "reach_mm": float(r[k - i]),
                         "ml": P[k, 0], "ap": P[k, 1], "dv": P[k, 2],
                         "miss_mm": float(dm[m - i]), "t_closest_ms": t[m],
                         "lat_err_mm": float(out_sign * (P[m, 0] - g[0])),
                         "ap_err_mm": float(P[m, 1] - g[1]), "dv_err_mm": float(P[m, 2] - g[2]),
                         "rep1_px": float(S.tongue_rep1.iloc[k]), "rep4_px": float(S.tongue_rep4.iloc[k])})
        i = j
    return pd.DataFrame(rows)
