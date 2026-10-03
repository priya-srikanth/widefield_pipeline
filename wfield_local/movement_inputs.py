"""Build `movement_encoding.MovementInputs` for one session from behaviour-camera pose data + DAQ events.

Pairs with `movement_encoding` (the model). Everything is put on the DAQ clock here; the encoder bins it onto imaging
frames. Real sessions need whole-session pose predictions (O2, after the next DLC / LP iteration -- Priya 2026-10-02);
the functions take plain arrays so they are testable now and wire to `orofacial_clean` / `tongue_kinematics` outputs
later without change.

  cam_frames_to_daq_s   video frame index -> DAQ seconds (inverse of `dlc_frames.frame_of`, same affine template)
  pose_signals          per-video-frame tongue protrusion (tongue-in = lip level), protrusion speed, sideways (LR)
                        position, jaw position and speed, from cleaned traces + the spout frame
  build_inputs          cue kernels per spout position, lick / jaw events, a direction-modulated tongue-onset kernel,
                        continuous tongue / jaw / state signals -> MovementInputs, with lags and groups from
                        `configs/defaults.yaml movement_encoding`
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from wfield_local import config
from wfield_local.movement_encoding import ContinuousRegressor, EventRegressor, MovementInputs

DEFAULTS = {
    "lags_s": {"cue": [-0.5, 1.5], "reward": [-0.2, 1.5], "tongue_onset": [-0.5, 1.0], "contact": [-0.3, 1.0],
               "jaw_onset": [-0.5, 1.0], "tongue_onset_x_angle": [-0.5, 1.0],
               "tongue_onset_x_deviation": [-0.5, 1.0]},
    "continuous_lags_s": [0.0, 0.1, 0.2],
    "groups": {"cue": "task", "reward": "task", "tongue_onset": "lick_events", "contact": "lick_events",
               "jaw_onset": "lick_events", "tongue_onset_x_angle": "direction",
               "tongue_onset_x_deviation": "direction", "tongue_protrusion": "tongue",
               "tongue_speed": "tongue", "tongue_lr": "tongue", "jaw_y": "jaw", "jaw_speed": "jaw",
               "running": "state"},
    "n_folds": 5,
    "alpha_grid": [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0],
    "min_reach_px": 60.0,
}


def params(overrides: dict | None = None) -> dict:
    p = config._deep_merge(DEFAULTS, config.defaults().get("movement_encoding", {}) or {})
    return config._deep_merge(p, overrides) if overrides else p


def cam_frames_to_daq_s(tpl: dict, frames) -> np.ndarray:
    """Video frame index -> DAQ seconds: the inverse of `dlc_frames.frame_of` (daq_sample = intercept + slope *
    frame), the alignment template's affine (~1.2 ms residual)."""
    f = np.asarray(frames, dtype=np.float64)
    return (float(tpl["intercept_daqSample"]) + float(tpl["slope_daqSample_per_camFrame"]) * f) / float(tpl["fs_daq"])


def pose_signals(tongue_x, tongue_y, tongue_in, jaw_y, *, origin, ap_axis, fps: float, lip_px: float | None = None,
                 jaw_unknown=None) -> dict:
    """Per-video-frame movement signals.

    ``tongue_x/y`` absolute image px (cleaned; e.g. `orofacial_clean` x_final + X0); ``tongue_in`` True where the tongue
    is not visible (baseline fill). ``origin`` / ``ap_axis`` = the session's spout frame. Protrusion = distance from
    the mouth, tongue-in frames set to the LIP LEVEL (``lip_px``; default = 5th percentile of visible protrusion), the
    convention of the centered lick phase. Speed = d protrusion / dt (px/s; + = outward). LR = sideways position in the
    spout frame (+ = image-right), NaN when the tongue is in (no direction without a tongue). ``jaw_y`` cleaned jaw y
    (baseline-subtracted); ``jaw_unknown`` True -> NaN (contributes nothing after z-scoring)."""
    x, y = np.asarray(tongue_x, float), np.asarray(tongue_y, float)
    tin = np.asarray(tongue_in, bool) | ~np.isfinite(x) | ~np.isfinite(y)
    rx, ry = -float(ap_axis[0]), -float(ap_axis[1])
    vx, vy = x - float(origin[0]), y - float(origin[1])
    ap, lr = rx * vx + ry * vy, ry * vx - rx * vy
    prot = np.hypot(ap, lr)
    if lip_px is None:
        vis = prot[~tin]
        lip_px = float(np.nanpercentile(vis, 5)) if vis.size else 0.0
    prot_f = np.where(tin, lip_px, prot)
    speed = np.gradient(prot_f) * float(fps)
    jy = np.asarray(jaw_y, float).copy()
    if jaw_unknown is not None:
        jy[np.asarray(jaw_unknown, bool)] = np.nan
    jspeed = np.gradient(np.where(np.isfinite(jy), jy, np.nan)) * float(fps)
    return {"tongue_protrusion": prot_f, "tongue_speed": speed, "tongue_lr": np.where(tin, np.nan, lr),
            "jaw_y": jy, "jaw_speed": jspeed, "lip_px": lip_px}


def build_inputs(frame_times_s, *, cues: pd.DataFrame | None = None, events: dict | None = None,
                 modulated: dict | None = None, video_signals: dict | None = None, video_t_s=None,
                 state_signals: dict | None = None, trial_starts_s=None, include: list[str] | None = None,
                 overrides: dict | None = None) -> MovementInputs:
    """MovementInputs for one session.

    ``cues``          DataFrame with `cue_s` (DAQ s) and `pos_name` -> one cue kernel PER POSITION ("cue_<pos>"), the
                      target regressors.
    ``events``        {name: DAQ seconds}, e.g. tongue_onset, contact, jaw_onset, reward.
    ``modulated``     {name: (DAQ seconds, amplitude)}, e.g. tongue_onset_x_angle = (onsets, peak angle per lick).
    ``video_signals`` {name: per-video-frame array} with ``video_t_s`` (DAQ s per video frame, `cam_frames_to_daq_s`).
    ``state_signals`` {name: (t_s, values)}, e.g. running speed.
    ``include``       optional list of regressor base names to keep (others dropped) -- for model comparison.
    Lags and groups from `params()` (configs/defaults.yaml `movement_encoding`)."""
    p = params(overrides)
    lags, groups, clags = p["lags_s"], p["groups"], tuple(float(v) for v in p["continuous_lags_s"])

    def keep(base):
        return include is None or base in include

    regs = []
    if cues is not None and keep("cue"):
        for pos, g in cues.groupby("pos_name"):
            regs.append(EventRegressor(f"cue_{pos}", g.cue_s.to_numpy(float), tuple(lags["cue"]), groups["cue"]))
    for name, t in (events or {}).items():
        if keep(name):
            regs.append(EventRegressor(name, np.asarray(t, float), tuple(lags[name]), groups[name]))
    for name, (t, a) in (modulated or {}).items():
        if keep(name):
            regs.append(EventRegressor(name, np.asarray(t, float), tuple(lags[name]), groups[name],
                                       amplitude=np.asarray(a, float)))
    for name, v in (video_signals or {}).items():
        if name == "lip_px" or not keep(name):
            continue
        if video_t_s is None:
            raise ValueError("video_signals need video_t_s (cam_frames_to_daq_s)")
        regs.append(ContinuousRegressor(name, np.asarray(video_t_s, float), np.asarray(v, float), groups[name], clags))
    for name, (t, v) in (state_signals or {}).items():
        if keep(name):
            regs.append(ContinuousRegressor(name, np.asarray(t, float), np.asarray(v, float), groups[name], clags))
    return MovementInputs(frame_times_s=np.asarray(frame_times_s, float), regressors=regs,
                          trial_starts_s=None if trial_starts_s is None else np.asarray(trial_starts_s, float))
