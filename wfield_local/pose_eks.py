"""Ensemble Kalman Smoother (EKS) plumbing for two-camera orofacial pose (cam1 below, cam4 frontal).

WHY (Priya, 2026-10-05). DLC and Lightning Pose each miss or mis-place the tongue on different frames, and
cam1 cannot see what cam4 sees (the rod hides the nose from below at the centre spout positions). The
Lightning Pose ecosystem's EKS (`pip install ensemble-kalman-smoother`, import ``eks``) fuses an ENSEMBLE
of per-camera predictions -- and, in its multi-camera form, the two views -- into one smoothed track with
a posterior variance per frame. The variance is the useful new quantity: it is large where ensemble
members disagree, which is exactly where neither network alone should be trusted.

This module is the reusable part, kept free of an ``eks`` import at module level so the unit tests (and
the ``locanmf`` env, which does not carry eks/jax) can import it:

  * frame matching between cameras on the DAQ clock (`cam_frame_to_daq_s`, `match_frames`) -- the same
    affine alignment templates `dlc_frames.frame_of` / `dlc_hard_frames.matched_frames` use;
  * converting our pose arrays ((n, K, 3) = [x, y, likelihood]) and LP csvs to the DLC-style DataFrames
    EKS reads (`pose_from_lp_csv`, `pose_to_frame`);
  * calling EKS (`smooth_multicam`, `smooth_singlecam`; lazy import) and returning smoothed poses plus
    posterior variance as plain arrays (`EKSResult`).

The env with eks lives apart from ``dlc``/``locanmf``/``lp`` (rule: never modify those): on the analysis
box it is the Windows conda env ``eks`` (python 3.10, ``pip install ensemble-kalman-smoother`` -> 4.6.2,
jax 0.4.36 CPU), created 2026-10-05.

FIRST FINDINGS (2026-10-05, six 3-s windows, 2-member ensembles -- a plumbing test, not an evaluation):
the auto-fitted smoothing parameter OVER-SMOOTHS the tongue (single-camera cam1 kept ~1/3 of each lick's
excursion; multicam ~0.8), so pass ``smooth_param`` explicitly (~10 kept 0.86-0.98 and still removed every
out-and-back spike). Posterior variance is NOT a presence signal on its own and is over-confident for a
view filled from the other camera (held-out cam1 tongue: ~17 px median error at a reported sd of 1.5-4 px).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

#: Our orofacial keypoints, in the order every (n, K, 3) pose array in this repo uses.
PARTS = ("nose", "jaw", "tongue", "spout")


# ---------------------------------------------------------------- frame matching on the DAQ clock
def cam_frame_to_daq_s(tpl: dict, frames) -> np.ndarray:
    """DAQ time (s) of camera frame indices, from the camera's alignment template (affine, drop-proof).

    The inverse of `dlc_frames.frame_of` at zero offset. Fractional frames are allowed, so a residual can
    be measured rather than rounded away."""
    f = np.asarray(frames, dtype=float)
    return (f * float(tpl["slope_daqSample_per_camFrame"]) + float(tpl["intercept_daqSample"])) / float(tpl["fs_daq"])


def daq_s_to_cam_frame(tpl: dict, t_daq_s) -> np.ndarray:
    """FRACTIONAL camera frame at DAQ time(s) ``t_daq_s`` (round it yourself; the fraction is the residual)."""
    t = np.asarray(t_daq_s, dtype=float)
    return (t * float(tpl["fs_daq"]) - float(tpl["intercept_daqSample"])) / float(tpl["slope_daqSample_per_camFrame"])


def match_frames(tpl_src: dict, tpl_dst: dict, frames_src) -> tuple[np.ndarray, np.ndarray]:
    """For every ``src`` frame, the NEAREST ``dst`` frame at the same DAQ instant, and the residual (s).

    Returns ``(dst_frames int, residual_s)`` with residual = t(dst frame) - t(src frame). The two 250-fps
    cameras run off different clocks (fps differ in the 5th digit), so the offset is not a whole frame
    and drifts slowly; nearest-frame matching leaves at most half a frame (2 ms) of timing error, which
    is reported, not hidden -- a tongue moving 1-2 px/ms at full protrusion speed makes it visible.
    Frames are clipped to the destination recording, so the last frames of a recording never index past
    the video."""
    fs = np.asarray(frames_src, dtype=float)
    t = cam_frame_to_daq_s(tpl_src, fs)
    g = np.rint(daq_s_to_cam_frame(tpl_dst, t)).astype(np.int64)
    n_dst = int(tpl_dst.get("n_cam_frames", np.iinfo(np.int64).max))
    g = np.clip(g, 0, n_dst - 1)
    return g, cam_frame_to_daq_s(tpl_dst, g) - t


# ---------------------------------------------------------------- pose formats
def pose_from_lp_csv(path: str | Path, parts=PARTS) -> np.ndarray:
    """(n, K, 3) [x, y, likelihood] from a Lightning Pose / DLC-style csv (3-row header scorer/bodypart/coord)."""
    df = pd.read_csv(path, header=[0, 1, 2], index_col=0)
    df.columns = df.columns.droplevel(0)
    return np.stack([np.stack([df[p]["x"], df[p]["y"], df[p]["likelihood"]], 1) for p in parts], 1).astype(float)


def mask_low(pose: np.ndarray, cut: float | dict, parts=PARTS) -> np.ndarray:
    """Copy of ``pose`` with x, y set to NaN where likelihood < ``cut`` (a float, or {part: cut}).

    WHY: in our labelling convention a blank means OCCLUDED (DECISIONS 2026-09-26), and both networks
    report an occluded part as a low-likelihood guess somewhere plausible. Handing that guess to a
    smoother as data pulls the track toward it; NaN tells EKS there is no observation at that frame."""
    out = np.array(pose, dtype=float, copy=True)
    for k, p in enumerate(parts):
        c = cut[p] if isinstance(cut, dict) else cut
        bad = ~(out[:, k, 2] >= c)
        out[bad, k, :2] = np.nan
    return out


def pose_to_frame(pose: np.ndarray, parts=PARTS, scorer: str = "model") -> pd.DataFrame:
    """DLC-style DataFrame (MultiIndex scorer / bodyparts / coords = x, y, likelihood) from (n, K, 3).

    This is the format EKS's own loaders produce from DLC/LP csvs, so arrays from either network (or from
    the cam4 CPU DLC run) enter EKS identically."""
    n, K, _ = pose.shape
    assert K == len(parts), (K, parts)
    cols = pd.MultiIndex.from_tuples([(scorer, p, c) for p in parts for c in ("x", "y", "likelihood")],
                                     names=["scorer", "bodyparts", "coords"])
    return pd.DataFrame(pose.reshape(n, K * 3), columns=cols)


def flat_frame(pose: np.ndarray, parts=PARTS) -> pd.DataFrame:
    """Single-level columns ``<part>_x``, ``<part>_y``, ``<part>_likelihood`` (the layout EKS's older
    ensemble functions index by name)."""
    n, K, _ = pose.shape
    return pd.DataFrame(pose.reshape(n, K * 3), columns=[f"{p}_{c}" for p in parts for c in ("x", "y", "likelihood")])


@dataclass
class EKSResult:
    """Smoothed track for one camera: ``pose`` (n, K, 3) [x, y, eks likelihood-like score] and the
    posterior variance ``var`` (n, K, 2) of x and y (px^2). ``raw`` keeps the package's own output
    DataFrame for anything not unpacked here (ensemble medians/variances, z-scores)."""
    pose: np.ndarray
    var: np.ndarray
    raw: pd.DataFrame | None = None


def unpack_eks_frame(df: pd.DataFrame, parts=PARTS, includes_obs_var: bool = False) -> EKSResult:
    """`EKSResult` from an EKS output DataFrame (DLC-style MultiIndex, coords incl. x/y/likelihood,
    x_ens_var/y_ens_var and x_posterior_var/y_posterior_var -- eks 4.x names).

    ``includes_obs_var``: eks 4.x's MULTI-camera linear path reports posterior_var = C V C' + the
    frame's ensemble (observation) variance, i.e. the spread of a new OBSERVATION, while the
    single-camera path reports C V C' alone. Set it for multicam output to subtract the ensemble
    variance back out, so ``var`` is the uncertainty of the smoothed TRACK in both cases (without
    this, an occluded frame filled from the other view still reads as >= the occlusion floor)."""
    sub = df.droplevel(0, axis=1) if df.columns.nlevels == 3 else df
    pose = np.stack([np.stack([sub[p]["x"], sub[p]["y"], sub[p]["likelihood"]], 1) for p in parts], 1)
    var = np.stack([np.stack([sub[p]["x_posterior_var"], sub[p]["y_posterior_var"]], 1) for p in parts], 1)
    if includes_obs_var:
        ens = np.stack([np.stack([sub[p]["x_ens_var"], sub[p]["y_ens_var"]], 1) for p in parts], 1)
        var = np.maximum(var - ens, 0.0)
    return EKSResult(pose.astype(float), var.astype(float), df)


# ---------------------------------------------------------------- calling EKS (lazy import)
def marker_stack(poses_by_cam: dict, parts=PARTS, min_likelihood: float = 1e-3) -> tuple[np.ndarray, list[str]]:
    """(M, V, T, K, 3) array for EKS's ``MarkerArray`` from ``{cam: [pose (T, K, 3), ...one per member]}``.

    Every camera must carry the same number of members and frames (frames matched on the DAQ clock
    first: `match_frames`). Likelihoods are floored at ``min_likelihood`` because EKS's
    ``confidence_weighted_var`` divides the ensemble variance by the members' mean likelihood, and an
    exact 0 from either network would make that 0/0 or x/0.

    NaN coordinates are REFUSED: EKS 4.x has no missing-observation path (dynamax's Kalman filter
    propagates a NaN through every later frame, and a NaN in ONE member gives ``nanvar`` = 0, i.e. a
    perfectly trusted observation). Occlusion enters only through likelihood -> variance; see
    `smooth_multicam` ``occluded_cut``."""
    cams = list(poses_by_cam)
    M = {len(v) for v in poses_by_cam.values()}
    T = {p.shape[0] for v in poses_by_cam.values() for p in v}
    if len(M) != 1 or len(T) != 1:
        raise ValueError(f"members per camera {M} and frames {T} must each be one value")
    arr = np.stack([np.stack([np.asarray(p, float) for p in poses_by_cam[c]], 0) for c in cams], 1)
    assert arr.shape[3] == len(parts), (arr.shape, parts)
    if not np.isfinite(arr[..., :2]).all():
        raise ValueError("NaN/inf coordinates: EKS needs every keypoint in every camera on every frame")
    arr[..., 2] = np.clip(np.nan_to_num(arr[..., 2], nan=0.0), min_likelihood, 1.0)
    return arr, cams


def floor_occluded_var(ens: np.ndarray, cut, var: float) -> np.ndarray:
    """Raise the ensemble variance to at least ``var`` (px^2) where the members' MEAN likelihood < ``cut``.

    ``ens`` is EKS's ensemble array (1, V, T, K, 5) with fields [x, y, var_x, var_y, likelihood];
    ``cut`` is a float or one value per keypoint (length K, e.g. tongue 0.4, others 0.6). WHY:
    with only two members, two networks that put an occluded tongue at the SAME wrong spot have a tiny
    spread, and dividing by a mean likelihood of ~0.1 inflates it only 10x -- still a confident
    observation of a part that is not there. A floor makes 'occluded' mean 'no information here' (the
    package itself uses 1000 px^2 for an undefined variance), so the track is carried by the dynamics
    and, in the multi-camera fit, by the other view."""
    out = np.array(ens, copy=True)
    low = out[..., 4] < np.asarray(cut, dtype=float)
    for f in (2, 3):
        out[..., f] = np.where(low, np.maximum(out[..., f], var), out[..., f])
    return out


def _with_ensemble_hook(module, cut, var):
    """Context manager: wrap ``module.ensemble`` so its output passes through `floor_occluded_var`.

    The multi-camera entry point computes the ensemble internally and exposes no variance hook, so
    the floor is applied by wrapping that one name for the duration of the call (no package edit)."""
    import contextlib

    @contextlib.contextmanager
    def cm():
        if cut is None:
            yield
            return
        from eks.marker_array import MarkerArray
        orig = module.ensemble

        def hooked(marker_array, *a, **kw):
            e = orig(marker_array, *a, **kw)
            return MarkerArray(floor_occluded_var(e.array, cut, var), data_fields=list(e.data_fields))

        module.ensemble = hooked
        try:
            yield
        finally:
            module.ensemble = orig
    return cm()


def smooth_multicam(poses_by_cam: dict, parts=PARTS, smooth_param=None, quantile_keep_pca: float = 50.0,
                    n_latent: int = 3, inflate_vars: bool = False, occluded_cut=None,
                    occluded_var: float = 1000.0, calibration=None,
                    min_likelihood: float = 1e-3, return_3d: bool = False):
    """Multi-camera EKS (``eks.multicam_smoother.ensemble_kalman_smoother_multicam``, eks 4.x).

    ``poses_by_cam`` = ``{cam: [member pose (T, K, 3), ...]}``, frames already matched across cameras.
    Returns ``({cam: EKSResult}, s_finals (K,))``.

    Without ``calibration`` this is EKS's LINEAR path: per keypoint, the 2V image coordinates of all
    views are stacked and a PCA (fit on the ``quantile_keep_pca`` % of frames with the lowest ensemble
    variance) gives an ``n_latent``-dim state; with 2 views of one 3-D point, 3 latents are the
    affine-camera case. A view with huge variance on a frame is then predicted from the other view
    through that PCA -- that is how cam4 can fill a cam1 tongue. No geometry is assumed.

    ``calibration`` (an anipose toml path, or an aniposelib ``CameraGroup``) switches to EKS's
    NONLINEAR path: a 3-D state per keypoint, observed through each camera's calibrated projection
    (an extended Kalman smoother). The group is SUBSET AND REORDERED to ``poses_by_cam``'s keys,
    because EKS pairs the k-th camera of the group with the k-th camera of the marker array by
    position, not by name, and the rig toml carries four cameras (cam1..cam4). The 2026-09-11 solve
    reprojects the paired cam1+cam4 labels at ~7-9 px for every part, the nose included (Priya,
    2026-10-05: the two views' nose is close to one point after all), and its spout-position
    distances match the stage to 1-2 % in Aug-Sep -- June sessions need another calibration.
    MEASURED 2026-10-05 (six Aug/Sep windows, s 3/10/30): the calibrated path was WORSE than the
    calibration-free one on every cam1 measure. Hidden-cam1-tongue hold-out error was 42-45 px against
    17-18 px. The cam1 nose at the centre spout landed 95-155 px from the same session's visible far_L
    nose (calibration-free: 85-98 px, about where the networks' raw guess sits). The 3-D estimate itself
    reprojects at 6-9 px in cam1 (the labelled-pair level). Cause: a view that loses a part leaves depth
    along the other camera's ray unobserved, and EKS seeds the 3-D state from a likelihood-BLIND
    triangulation of the raw (occluded) guesses, which the random walk then holds. A larger occlusion
    variance (1e5, 1e6) did not fix it. Do not use it for occluded parts without a likelihood-weighted
    initialisation.
    NB the smoothing parameter is NOT comparable between the two paths: the linear path normalises
    its process covariance to max 1 in PCA units, the nonlinear one takes it from the triangulated
    track's frame-to-frame MAD in calibration units (mm).

    ``return_3d`` adds EKS's 3-D latent DataFrame (x, y, z and their posterior variances per
    keypoint; meaningful in calibration units only on the nonlinear path) as a third element."""
    from eks import multicam_smoother as MS
    from eks.marker_array import MarkerArray

    arr, cams = marker_stack(poses_by_cam, parts, min_likelihood)
    camgroup = None
    if calibration is not None:
        camgroup = calibration
        if isinstance(calibration, (str, Path)):
            from aniposelib.cameras import CameraGroup
            camgroup = CameraGroup.load(str(calibration))
        camgroup = camgroup.subset_cameras_names(cams)
        assert [c.get_name() for c in camgroup.cameras] == cams, (camgroup.get_names(), cams)
    with _with_ensemble_hook(MS, occluded_cut, occluded_var):
        dfs, s_finals, df3d = MS.ensemble_kalman_smoother_multicam(
            marker_array=MarkerArray(arr, data_fields=["x", "y", "likelihood"]), keypoint_names=list(parts),
            camera_names=cams, smooth_param=smooth_param, quantile_keep_pca=quantile_keep_pca,
            inflate_vars=inflate_vars, n_latent=n_latent, camgroup=camgroup)
    # both multicam paths (linear PCA and calibrated projection) add the ensemble variance in
    out = ({c: unpack_eks_frame(d, parts, includes_obs_var=True) for c, d in zip(cams, dfs)}, np.asarray(s_finals))
    return (*out, df3d) if return_3d else out


def smooth_singlecam(members: list, parts=PARTS, smooth_param=None, occluded_cut=None,
                     occluded_var: float = 1000.0, min_likelihood: float = 1e-3) -> tuple[EKSResult, np.ndarray]:
    """Single-camera EKS (``eks.singlecam_smoother.ensemble_kalman_smoother_singlecam``) over ensemble
    ``members`` (list of (T, K, 3)); each keypoint smoothed in 2-D on its own. Same occlusion floor as
    `smooth_multicam`, for a like-for-like comparison."""
    from eks import singlecam_smoother as SS
    from eks.marker_array import MarkerArray

    arr, _ = marker_stack({"cam": members}, parts, min_likelihood)
    with _with_ensemble_hook(SS, occluded_cut, occluded_var):
        df, s_finals = SS.ensemble_kalman_smoother_singlecam(
            marker_array=MarkerArray(arr, data_fields=["x", "y", "likelihood"]), keypoint_names=list(parts),
            smooth_param=smooth_param)
    return unpack_eks_frame(df, parts), np.asarray(s_finals)


def pca_keep_quantile(mean_lik: np.ndarray, cut, cap: float = 50.0, lo: float = 5.0, min_seen: float = 0.05) -> float:
    """``quantile_keep_pca`` for a multicam fit with the occlusion floor on.

    ``mean_lik`` is the members' mean likelihood, (V, T, K). EKS fits each keypoint's cross-view PCA on
    the frames whose MAX-over-cameras ensemble variance is at or below that percentile -- and then
    truncates every keypoint to the smallest such count. With the floor on, any frame where a part is
    occluded in either view sits at exactly the floor, so once the percentile reaches the floor the
    PCA is fit on occluded guesses. The tongue is out on maybe a fifth of a 3-s window, so the
    package default of 50 % does exactly that. This returns 90 % of the both-views-visible fraction of
    the scarcest keypoint, clipped to [lo, cap]; keypoints seen in both views on < ``min_seen`` of
    frames (cam1's nose at the centre spout, hidden by the rod) are left out of the minimum -- their
    PCA is unidentifiable from this window whatever the setting, and they would starve the others."""
    seen = (np.asarray(mean_lik) >= np.asarray(cut, dtype=float)).all(axis=0).mean(axis=0)   # (K,)
    usable = seen[seen >= min_seen]
    if usable.size == 0:
        return lo
    return float(np.clip(90.0 * usable.min(), lo, cap))
