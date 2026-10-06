"""Per-trial tongue kinematics: v7 lick detection per trial, then lick / velocity / count / bout / angle features.

    from wfield_local import orofacial_clean as oc, tongue_kinematics as tk
    c = oc.clean_bodypart(pose_df, "tongue", centers, fps)
    trials = [tk.Trial(trial_id=k, cue_frame=cf, position=pos) for k, (cf, pos) in ...]
    res = tk.from_clean(c, trials, spout_frame=tk.SpoutFrame(origin=(ox, oy), ap_axis=(ax, ay)))
    res.per_trial, res.per_lick, res.bouts, res.licking_dynamics_traces

PORTED 2026-10-01 from `../stroke_orofacial_pipeline/src/stroke_orofacial/dlc_kinematics/` (read-only), the
path their production default `use_v7_pipeline: true` takes through `_build_tongue_pertrial_core`. Detection
(pre-clean, detectors, gates) is in `wfield_local.tongue_detect`; this module transcribes:

  tongue_pertrial.py
    `_run_v7_pipeline_per_trial`        -> `run_v7_pipeline_per_trial` (pre-clean -> 3 detectors -> merge ->
                                           F/D/E -> M -> kept-lick records with angle + max dy/dt)
    `_extract_trial_slice`              -> `extract_trial_slice`
    `_build_cleaned_trace_df` / `_v7_cleaned_bundle_arrays` -> `_overlay_cleaned` (session arrays)
    `_extract_lick1_lick2`              (v7 branch) -> `extract_lick1_lick2`
    `_extract_peak_velocity_first5`     (v7 branch) -> `extract_peak_velocity_first5`
    `_extract_n_licks_family`           (as called by v7: slope detector on the CLEANED slice)
    `_extract_licking_dynamics_reductions` (kept_licks branch)
    `_extract_licking_dynamics_traces`, `_moving_avg_window`
    `_bout_time_bounds_ms`, `_kept_licks_to_bundle_peaks`, and the bout-summary rows of
    `_build_visual_bundle_per_trial` (kept_licks branch)  -> `bout_rows_for_trial`
    the per-trial loop of `_build_tongue_pertrial_core`    -> `compute_tongue_kinematics`
  _detector.py  `_build_bouts`
  _angle.py     `_compute_signed_tongue_angle_deg` (REFERENCE FRAME CHANGED, below), `_gaussian_smooth_nan`,
                `_extract_per_frame_angle_smoothed`, `_extract_angle_at_lick_timestamps`,
                `_extract_angle_max_signed_per_lick`, `_extract_cell50_own_bout_scalars` (kept_licks branch)

NOT PORTED, and why:
  * Every `use_v7_pipeline=False` branch (raw slope re-detection for lick1/2, velocity, own-bout, dynamics,
    bundle peaks) and the `detector_per_cell` / `bout_per_cell` legacy-parity overrides (all empty in their
    production config; `_per_cell_detector` therefore returns the base `detector` block, which is what is
    used here).
  * `_evaluate_session_inclusion` -- a SESSION-level keep/drop gate with old-cohort trial-count thresholds,
    not a per-trial feature. Trivial to add once thresholds exist for this cohort.
  * The visual bundle's array snippets, angle masks, phase rows and first-peak-aligned snippet, and
    `tongue_phase_aligned_endpoints` (Q13b endpoints) -- plotting / phase-plot substrate, not in scope.
  * Loaders, eye-midpoint resolution (`_resolve_eye_midpoint_session_median`: replaced by `SpoutFrame`),
    writers, parquet, JSON packing, day-binned aggregation, wavesurfer trial derivation.

WHAT CHANGED, ON PURPOSE (each also noted at its site):
  * TRIALS: their `TrialEvent(trial_index, tone_frame_idx: int, side: L|R)` -> `Trial(trial_id, cue_frame:
    float, position)`, position one of `POSITIONS` (one spout moving over 6 places, not two fixed spouts).
    "side" is "position" throughout; the licking-dynamics traces are grouped per POSITION (6 groups) where
    theirs were per side (2). The cue frame is a FLOAT (camera frame of a DAQ time); t_ms is measured from
    it exactly, and the slice bounds are floor/ceil of cue + window, which equals theirs for an integer cue.
  * ANGLE REFERENCE: theirs = angle at the eye midpoint between (eye -> spout midpoint) and (eye -> tongue),
    in spout-midpoint-centred coordinates. Ours = angle at `SpoutFrame.origin` (the mouth: where the
    close->far spout lines meet) between the reversed AP axis (out of the mouth, toward the spouts) and
    (origin -> tongue tip), in ABSOLUTE image px: tip = (x_final + X0, y_final + Y0), because
    `orofacial_clean` output is baseline-subtracted. Positive = toward image-right. No SpoutFrame = no angles
    (all angle columns NaN, `angle_per_frame` None).
  * n_licks direction tally: theirs counted x > +5 px as "left" and x < -5 px as "right" in THEIR camera's
    spout frame. Here x_final is image x relative to the retracted tongue (X0), and which image side is the
    mouse's left depends on our camera, so the counts are named by IMAGE side: `n_img_right` (x > +thr),
    `n_img_left` (x < -thr).
  * Constants: module `DEFAULTS` mirroring their YAML (`tongue_pertrial.*`), deep-merged with
    `configs/defaults.yaml: orofacial_kinematics.tongue`. Their hard-coded widest trial window
    `(70.0, 8000.0)` is `trial_slice_win_ms`; the visual-bundle windows the bout table uses are `bout_table.*`.
  * Outputs are DataFrames (per-lick, per-trial, per-bout, per-position traces) + session arrays, not their
    JSON/parquet. The per-bout table carries only the bout-summary rows of their mixed-row `df_bouts`; the
    per-peak rows are the per-lick table (which gains `bout_idx`, `multi_bout_rank`,
    `angle_smoothed_at_peak_deg`).
  * EXTENSION (not in theirs): per-lick `vy_peak_px_per_s` / `vxy_peak_px_per_s` for EVERY kept lick, by the
    exact rule their first-5 velocity extractor applies; the per-trial first-5 columns are those values for
    the first five kept licks inside `peak_velocity_detect_win_ms`, i.e. identical to theirs.

FIXED ON REQUEST (Priya 2026-10-01; see the NOTE in `tongue_detect.merge_detector_outputs`): peak FRAMES out
of the detectors are detect-window-local but theirs uses them as trial-slice frames, a one-frame (4 ms) early
shift in x-at-peak, rise/fall ms, gate windows and velocity windows (peak TIMES correct). `fix_detect_offset`
(True in configs/defaults.yaml, False in DEFAULTS so the parity tests still mirror theirs) shifts them.
Also carried over: kept licks exist only inside `lick12_detect_win_ms` (70-3000 ms), so the dynamics and
own-bout features, whose apply windows run to 5000 ms, see no lick after 3000 ms -- while `n_licks` (slope
detector on the cleaned slice, 70-8000 ms) counts to 5000 ms. And trial slices run to +8000 ms: if trials
are closer than that, a later trial's cleaned slice overwrites the overlap in the session arrays.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from wfield_local import tongue_detect as td
from wfield_local import trial_windows as TW

POSITIONS: tuple[str, ...] = ("far_L", "close_L", "far_center", "close_center", "close_R", "far_R")

#: Their YAML `dlc_kinematics.tongue_pertrial` (+ `tongue_visual.bundle` for the bout table), restricted to
#: what the v7 path reads, plus the detection blocks from `tongue_detect.DEFAULTS`.
DEFAULTS: dict[str, Any] = td._deep_merge(td.DEFAULTS, {
    # Per-metric windows, ms post-CUE (theirs: post-tone/reward; same role). Their 70 ms start skips the
    # delivery transient; our cue has no such transient at +0 ms, but it is kept so the features mean the same.
    "lick12_detect_win_ms": [70.0, 3000.0],         # ALSO the v7 detect window -> kept licks live only here
    "peak_velocity_detect_win_ms": [70.0, 3000.0],
    "lick_count_detect_win_ms": [70.0, 8000.0],
    "lick_count_apply_win_ms": [70.0, 5000.0],
    "lick_count_bin_win_ms": [70.0, 5070.0],        # 5 x 1-s bins
    "licking_dyn_apply_win_ms": [70.0, 5000.0],
    "angle_apply_win_ms": [70.0, 5000.0],
    # Theirs: `widest_detect = (70.0, 8000.0)` hard-coded in `_build_tongue_pertrial_core`. The v7 pre-clean
    # runs on THIS slice, so its start must equal lick12_detect_win_ms[0] for the frame bookkeeping to stay
    # what theirs is (see the module docstring's off-by-one note).
    "trial_slice_win_ms": [70.0, 8000.0],
    # OURS: True shifts detector frames onto the trial slice (the off-by-one fix). False here = theirs (parity
    # tests); configs/defaults.yaml turns it on.
    "fix_detect_offset": False,
    "bout": {
        "max_ili_ms": 300.0,
        "min_ili_ms": 80.0,
        "single_bout_win_ms": [-120.0, 200.0],
        "min_licks_in_bout_for_ili": 3,
    },
    "direction_thresholds": {
        # Theirs: x_left_thr_px: 5.0 / x_right_thr_px: -5.0 (named for their camera's spout sides).
        "img_right_thr_px": 5.0,                    # px, OLD RIG -- retune
        "img_left_thr_px": -5.0,                    # px, OLD RIG -- retune
    },
    "n_licks_categorical": {"lick_ge10_thr": 10},
    "n_licks_binning": {"bin_count": 5, "bin_ms": 1000.0},
    "max_licks_per_trial": 40,
    "licking_dynamics": {
        "bin_ms": 20.0,
        "lickrate_smooth_sigma_ms": 50.0,
        "ili_smooth_sigma_ms": 50.0,
        "prob_smooth_movavg_ms": 200.0,
        "min_trials_per_bin": 3,
    },
    "angle": {
        # -1 with image coords (y DOWN) makes a tongue that turns toward image-RIGHT read positive when the
        # reference points down the image (mouth above the spouts). Equivalently: positive = counter-
        # clockwise as displayed. Changing it inverts the convention, not just a label.
        "sign_flip": -1.0,
        "smoothing_gaussian_sigma_ms": 8.0,
        # OURS (2026-10-01): angle_max_signed only over frames whose tongue-mouth distance is >= this fraction of
        # the distance at the lick's peak. With the origin AT the mouth, frames near the lips give short,
        # unstable vectors and won the max (sign flips vs the peak angle). None = theirs (whole +-peak_pad).
        "max_signed_min_frac_of_peak": None,
        # OURS: also clip the +-peak_pad window to the lick's own visible extent (on_frame..off_frame), so a
        # short lick's window cannot reach into the next lick (PS93 0908 trial 288). Needs lick_geometry or
        # velocity_window: visible_rise (which compute the extent). False = theirs.
        "max_signed_within_lick": False,
    },
    # OURS (Priya 2026-10-01): "v7" = theirs, max dy/dt over [rise_start, fall_end] (for lmax licks that is the
    # peak +-16 ms, so the value depended on which detector found the lick); "visible_rise" = over the lick's
    # whole visible rise (`lick_extent`), and retraction speed over its visible fall.
    "velocity_window": "v7",
    # OURS: per-lick geometry -- lick extent (on/off frames), xy protrusion from the mouth in the spout frame
    # (protrusion_px, ap_px, lr_px), and the lick-phase table (angle / protrusion resampled on phase 0..1).
    # Use with fix_detect_offset: the extent is grown from the peak FRAME, which theirs puts one frame early.
    "lick_geometry": False,
    # OURS: lick-phase table. mode "centered" = stroke_orofacial's centered-on-peak phase (their
    # `build_first_lick_centered_phase_rows`, ili_source "synthetic_only", the mode that won their visual gate):
    # window = peak +- cycle_ms / 2 in REAL time, phase = (t - (peak - cycle/2)) / cycle, cycle_ms = the animal's
    # pre-stroke median within-bout ILI (None -> this session's median within-bout ILI). Every lick on one fixed
    # time scale, so rise / fall durations (and peak sharpness) stay real. mode "extent" = ours of 2026-10-01: each
    # lick's visible rise and fall stretched to 0-0.5 / 0.5-1 -- peak sharpness then depends on when the tongue
    # becomes visible (Priya 2026-10-02 worried shape differences were an artefact of this; they can be).
    "phase": {"n_points": 21, "extent_slack_px": 2.0, "mode": "extent", "cycle_ms": None, "min_coverage": 0.2},
    # OURS (Priya 2026-10-01): "y" = theirs, detect licks on image y; "protrusion" = on the tongue-mouth
    # DISTANCE (spout frame), so lateral licks (far_L / far_R) are not under-read. The pipeline's "y" trace is
    # then the distance (per-lick `y`, velocities = protrusion speed); geometry and angles keep image coords.
    # Needs a SpoutFrame and X0/Y0.
    "detect_on": "y",
    # OURS: DAQ spout contact per lick (`contacts_ms`): match "peak" = a contact onset within match_ms of the
    # lick's peak (first version); "span" = a contact onset inside the lick's own rise start .. fall end (+-
    # span_pad_ms) -- Priya 2026-10-06: 65 / 83 "unmatched" PS93 0814 contacts were real licks touching the
    # spout > 60 ms from the peak (retraction, second touch).
    "contact": {"match_ms": 60.0, "match": "peak", "span_pad_ms": 8.0},
    # OURS: per-trial spout tip (median of confident frames, cue -> trial stop) for reach accuracy.
    "spout_ref": {"lk_thr": 0.6, "fallback_win_ms": [0.0, 3500.0]},
    "bout_table": {                                  # theirs: tongue_visual.bundle
        "plot_win_ms": [-180.0, 8000.0],             # only the grid that maps a lick time to a frame
        "detect_win_ms": [70.0, 3000.0],
    },
})


def params(overrides: dict | None = None) -> dict:
    """DEFAULTS <- `configs/defaults.yaml: orofacial_kinematics.tongue` <- ``overrides`` (deep merge)."""
    return td._deep_merge(td._deep_merge(copy.deepcopy(DEFAULTS), td.config_overrides()), overrides or {})


# --------------------------------------------------------------------------- inputs

@dataclass(frozen=True)
class Trial:
    """One trial: cue camera frame (float), spout position, id. Replaces their `TrialEvent` (int tone frame,
    L/R side): this rig has one spout that moves among six positions."""
    trial_id: Any
    cue_frame: float
    position: str
    #: OURS (2026-10-01): the trial's own bounds, camera frames (float). With `stop_frame` set, every post-cue
    #: response window ends at the trial stop instead of the ported fixed 3000/5000/8000 ms
    #: (`trial_windows.end_at_stop`). None = the ported fixed windows.
    stop_frame: float | None = None
    strobe_frame: float | None = None

    def __post_init__(self):
        if self.position not in POSITIONS:
            raise ValueError(f"position {self.position!r} not in {POSITIONS}")


@dataclass(frozen=True)
class SpoutFrame:
    """Angle reference, image px. ``origin`` = the mouth (where the close->far spout lines meet); ``ap_axis`` =
    direction from the far_center spout TOWARD the origin (normalised here). Angle 0 = tongue pointing straight
    out along -ap_axis, i.e. at far_center."""
    origin: tuple[float, float]
    ap_axis: tuple[float, float]

    def __post_init__(self):
        a = np.asarray(self.ap_axis, float)
        nrm = float(np.hypot(*a))
        if not (np.isfinite(nrm) and nrm > 0):
            raise ValueError(f"ap_axis must be a nonzero finite vector, got {self.ap_axis}")
        object.__setattr__(self, "ap_axis", (float(a[0] / nrm), float(a[1] / nrm)))
        object.__setattr__(self, "origin", (float(self.origin[0]), float(self.origin[1])))


def trials_from_frame(df: pd.DataFrame) -> list[Trial]:
    """Columns ``trial_id, cue_frame, position`` -> list of `Trial` (rows with a NaN cue are dropped)."""
    d = df.dropna(subset=["cue_frame"])
    opt = lambda r, c: (float(getattr(r, c)) if c in d.columns and pd.notna(getattr(r, c)) else None)  # noqa: E731
    return [Trial(r.trial_id, float(r.cue_frame), str(r.position), opt(r, "stop_frame"), opt(r, "strobe_frame"))
            for r in d.itertuples(index=False)]


# --------------------------------------------------------------------------- trial slice

def extract_trial_slice(arr, cue_frame: float, win_ms, fps: float):
    """``arr`` cut to ``win_ms`` around the cue, plus its t axis (ms from cue). Their `_extract_trial_slice`.

    Returns (slice, t_ms, lo, hi_incl) -- lo/hi are the session frames spanned (theirs recomputed them in the
    caller with the same formula). Adapted for a FLOAT cue frame: bounds are floor(cue + w0) / ceil(cue + w1)
    in frames, which is theirs ``int(tone + floor(w0))`` exactly when the cue is an integer, and t_ms is from
    the true (fractional) cue, so a lick time is not quantised to the frame the cue happened to round to.
    """
    n = len(arr)
    lo_frame = int(np.floor(cue_frame + win_ms[0] / 1000.0 * fps))
    hi_frame = int(np.ceil(cue_frame + win_ms[1] / 1000.0 * fps))
    lo = max(0, lo_frame)
    hi = min(n, hi_frame + 1)
    if lo >= hi:
        return arr[lo:lo], np.zeros(0, dtype=np.float64), lo, lo - 1
    t_ms = (np.arange(lo, hi, dtype=np.float64) - cue_frame) / fps * 1000.0
    return arr[lo:hi], t_ms, lo, hi - 1


# --------------------------------------------------------------------------- angle (_angle.py)

def compute_signed_tongue_angle_deg(x_img, y_img, frame: SpoutFrame, *, sign_flip: float = -1.0):
    """Signed angle (deg) at ``frame.origin`` from the reversed AP axis to (tip - origin), image px.

    Same atan2(cross, dot) * sign_flip construction as their `_compute_signed_tongue_angle_deg`; only the
    reference changed (theirs: eye midpoint -> spout midpoint). ``x_img, y_img`` must be ABSOLUTE image px
    (`orofacial_clean` x_final + X0, y_final + Y0).
    """
    ox, oy = frame.origin
    ref_x, ref_y = -frame.ap_axis[0], -frame.ap_axis[1]      # out of the mouth, toward the spouts
    vx = np.asarray(x_img, dtype=np.float64) - ox
    vy = np.asarray(y_img, dtype=np.float64) - oy
    dot = ref_x * vx + ref_y * vy
    cross = ref_x * vy - ref_y * vx
    return float(sign_flip) * np.degrees(np.arctan2(cross, dot))


def spout_coords(x_img, y_img, frame: SpoutFrame | None):
    """(ap, lr) px of absolute image points in the spout frame: ap = along -ap_axis (out of the mouth, toward
    the spouts), lr = perpendicular, + = image-right (= the sign of `compute_signed_tongue_angle_deg`, whose angle
    is atan2(lr, ap)). Without a frame: ap = image y, lr = image x (whatever origin the inputs carry)."""
    x_img, y_img = np.asarray(x_img, dtype=np.float64), np.asarray(y_img, dtype=np.float64)
    if frame is None:
        return y_img, x_img
    rx, ry = -frame.ap_axis[0], -frame.ap_axis[1]
    vx, vy = x_img - frame.origin[0], y_img - frame.origin[1]
    return rx * vx + ry * vy, ry * vx - rx * vy


def lick_extent(y, is_base, peak_frame: int, slack_px: float = 2.0) -> tuple[int, int]:
    """(on, off) frames of one lick's VISIBLE excursion around ``peak_frame``: back while the frame is visible
    (not baseline fill, finite) and y keeps falling (``slack_px`` tolerance), forward likewise. OURS."""
    yv = np.where(is_base, np.nan, np.asarray(y, dtype=np.float64))
    n = len(yv)
    on = off = int(peak_frame)
    if not (0 <= on < n) or not np.isfinite(yv[on]):
        return on, off
    while on > 0 and np.isfinite(yv[on - 1]) and yv[on - 1] <= yv[on] + slack_px:
        on -= 1
    while off < n - 1 and np.isfinite(yv[off + 1]) and yv[off + 1] <= yv[off] + slack_px:
        off += 1
    return on, off


def _min_dydt(y_seg, fps) -> float:
    y_seg = np.asarray(y_seg, dtype=np.float64)
    if int(np.count_nonzero(np.isfinite(y_seg))) < 2:
        return float("nan")
    d = np.gradient(y_seg) * fps
    return float(np.nanmin(d)) if np.isfinite(d).any() else float("nan")


def _max_dydt(y_seg, fps) -> float:
    y_seg = np.asarray(y_seg, dtype=np.float64)
    if int(np.count_nonzero(np.isfinite(y_seg))) < 2:
        return float("nan")
    d = np.gradient(y_seg) * fps
    return float(np.nanmax(d)) if np.isfinite(d).any() else float("nan")


def gaussian_smooth_nan(arr, sigma_units: float, *, mode: str = "nearest"):
    """NaN-aware gaussian: smooth(values with NaN->0) / smooth(finite mask); NaN where mask weight <= 1e-8."""
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return arr
    finite = np.isfinite(arr).astype(np.float64)
    safe = np.where(np.isfinite(arr), arr, 0.0)
    if sigma_units <= 0:
        return np.where(finite > 0, safe, np.nan)
    num = gaussian_filter1d(safe, sigma_units, mode=mode)
    den = gaussian_filter1d(finite, sigma_units, mode=mode)
    out = np.full_like(arr, np.nan)
    nz = den > 1e-8
    out[nz] = num[nz] / den[nz]
    return out


def per_frame_angle_smoothed(x_img, y_img, frame: SpoutFrame, fps: float, *, p: dict):
    """Session-length signed angle, THEN gaussian smoothing at sigma = smoothing_gaussian_sigma_ms (theirs)."""
    a = p["angle"]
    raw = compute_signed_tongue_angle_deg(x_img, y_img, frame, sign_flip=float(a["sign_flip"]))
    return gaussian_smooth_nan(raw, float(a["smoothing_gaussian_sigma_ms"]) / 1000.0 * fps, mode="nearest")


def angle_at_lick_timestamp(angle_per_frame, fps, cue_frame, lick_t_ms):
    """Linear interpolation of the per-frame angle at cue + t (their `_extract_angle_at_lick_timestamps`)."""
    if angle_per_frame is None or not np.isfinite(lick_t_ms):
        return float("nan")
    target = cue_frame + lick_t_ms / 1000.0 * fps
    n = len(angle_per_frame)
    if target < 0 or target > n - 1:
        return float("nan")
    return float(np.interp(target, np.arange(n, dtype=np.float64), angle_per_frame))


def angle_max_signed_per_lick(angle_per_frame, fps, cue_frame, lick_t_peaks_ms, peak_pad_ms, *,
                              dist_per_frame=None, min_frac_of_peak=None, extents=None):
    """Per lick, the largest-|angle| value (sign kept) within +-peak_pad_ms of its time.

    Theirs centred on ``tone_frame_idx + int(round(t * fps / 1000))``; with a float cue the centre is
    ``round(cue + t * fps / 1000)`` -- the same frame for an integer cue, and the nearest frame otherwise.
    """
    out = np.full(len(lick_t_peaks_ms), np.nan, dtype=np.float64)
    if angle_per_frame is None:
        return out
    n_frames = len(angle_per_frame)
    pad = int(round(peak_pad_ms / 1000.0 * fps))
    for j, t in enumerate(lick_t_peaks_ms):
        if not np.isfinite(t):
            continue
        center = int(round(cue_frame + t / 1000.0 * fps))
        w0, w1 = max(0, center - pad), min(n_frames, center + pad + 1)
        if extents is not None and j < len(extents) and extents[j] is not None:
            w0, w1 = max(w0, int(extents[j][0])), min(w1, int(extents[j][1]) + 1)   # OURS: within the lick
        win = angle_per_frame[w0:w1]
        if min_frac_of_peak is not None and dist_per_frame is not None and 0 <= center < n_frames:
            # OURS: only the outer part of the lick (see DEFAULTS angle.max_signed_min_frac_of_peak)
            dwin = dist_per_frame[w0:w1]
            win = np.where(dwin >= float(min_frac_of_peak) * dist_per_frame[center], win, np.nan)
        if not np.any(np.isfinite(win)):
            continue
        out[j] = float(win[int(np.nanargmax(np.abs(win)))])
    return out


# --------------------------------------------------------------------------- bouts

def build_bouts(peak_times_ms, max_ili_ms: float) -> list[np.ndarray]:
    """Group peaks with ILI <= max_ili_ms (their `_build_bouts`; single-lick bouts always kept, as theirs --
    their `allow_single_lick_bouts` knob has no consumer)."""
    peak_times_ms = np.asarray(peak_times_ms, dtype=np.float64)
    n = len(peak_times_ms)
    if n == 0:
        return []
    order = np.argsort(peak_times_ms)
    st = peak_times_ms[order]
    bouts: list[list[int]] = [[int(order[0])]]
    for k in range(1, n):
        if st[k] - st[k - 1] <= max_ili_ms:
            bouts[-1].append(int(order[k]))
        else:
            bouts.append([int(order[k])])
    return [np.asarray(b, dtype=np.int64) for b in bouts]


def _bout_time_bounds_ms(peak_times_ms, single_bout_win_ms):
    """Multi-lick bout: first/last peak extended by half the first/last ILI; single: fixed window."""
    if len(peak_times_ms) >= 2:
        lead = 0.5 * (peak_times_ms[1] - peak_times_ms[0])
        tail = 0.5 * (peak_times_ms[-1] - peak_times_ms[-2])
        return float(peak_times_ms[0] - lead), float(peak_times_ms[-1] + tail)
    return (float(peak_times_ms[0]) + float(single_bout_win_ms[0]),
            float(peak_times_ms[0]) + float(single_bout_win_ms[1]))


# --------------------------------------------------------------------------- v7 per trial

@dataclass
class V7TrialResult:
    """One trial through the v7 pipeline (their `_V7TrialResult`; ``side`` -> ``position``). Frame indices
    in ``kept_licks`` / ``decisions`` are trial-slice-local (one frame early unless ``fix_detect_offset``)."""
    trial_id: Any
    position: str
    cue_frame: float
    y_clean: np.ndarray
    x_clean: np.ndarray
    is_baseline_fill_clean: np.ndarray
    t_ms: np.ndarray
    kept_licks: list[dict[str, Any]]
    decisions: list[dict[str, Any]]           # every merged peak with keep/gate/reason (QC; not in theirs)
    preclean_diag: td.PrecleanDiagnostics
    session_frame_lo: int = -1
    session_frame_hi: int = -1
    y_geom: np.ndarray | None = None          # OURS: image y (rel Y0) when detection ran on another trace


def run_v7_pipeline_per_trial(y_slice, x_slice, is_interp_fill_slice, is_baseline_fill_slice, likelihood_slice,
                              t_ms, fps, *, p, trial_id, position, cue_frame, session_frame_lo, session_frame_hi,
                              X0=None, Y0=None, spout_frame: SpoutFrame | None = None,
                              y_geom_slice=None) -> V7TrialResult:
    """Pre-clean -> lmax + bounded + legacy detectors -> merge -> F, D, E(+impute) -> M -> kept-lick records.

    Their `_run_v7_pipeline_per_trial`, statement for statement, except the per-lick angle (reference frame,
    absolute coords; NaN without a SpoutFrame). ``p`` is the full `params()` dict.
    """
    preclean_params, detector_params = p["preclean"], p["detector"]
    detector_v2_params, gates_params = p["detector_v2"], p["gates"]
    detect_win_ms = tuple(p["lick12_detect_win_ms"])
    n_frames = len(y_slice)

    # Stage 1: pre-clean
    y_clean, x_clean, is_base_clean, diag = td.preclean_trace(
        y_slice, x_slice, is_interp_fill_slice, is_baseline_fill_slice, n_frames, fps, params=preclean_params)

    # Stage 2: three detectors, all on the CLEANED trace
    lmax_pl = td.detect_local_maxima(y_clean, x_clean, t_ms, fps, is_baseline_fill=is_base_clean,
                                     detect_win_ms=detect_win_ms, detector_params=detector_params,
                                     params=detector_v2_params)
    bounded_params = dict(detector_params)
    bounded_params["max_fall_search_ms"] = float(detector_v2_params["max_fall_search_ms"])
    bounded_params["accept_implicit_baseline_fall"] = True
    bounded_pl = td.legacy_peaks_to_peak_list(
        td.slope_detect_lick_peaks(y_clean, x_clean, t_ms, fps, detect_win_ms=detect_win_ms,
                                   detector_params=bounded_params, n_max=None, is_baseline_fill=is_base_clean),
        detector_params, fps)
    legacy_pl = td.legacy_peaks_to_peak_list(
        td.slope_detect_lick_peaks(y_clean, x_clean, t_ms, fps, detect_win_ms=detect_win_ms,
                                   detector_params=detector_params, n_max=None),
        detector_params, fps)

    # OURS (Priya 2026-10-01, `fix_detect_offset`): move the detector frames onto the trial slice before they
    # index anything (theirs used them as-is, one frame early -- module docstring). Off in DEFAULTS = parity.
    if p.get("fix_detect_offset", False):
        b = td._window_bounds(t_ms, detect_win_ms)
        if b is not None:
            lmax_pl, bounded_pl, legacy_pl = (td.shift_peak_list(pl, b[0]) for pl in (lmax_pl, bounded_pl, legacy_pl))

    # Stage 2.5: dedupe, lmax wins on overlap
    merged = td.merge_detector_outputs([("lmax", lmax_pl), ("bounded", bounded_pl), ("legacy", legacy_pl)],
                                       likelihood=likelihood_slice, fps=fps, params=detector_v2_params)
    decisions: list[dict[str, Any]] = [{
        "rank": rank, "src": pk.source, "peak_frame_local": pk.frame, "t_ms": pk.t_ms, "y": pk.y,
        "lik": pk.likelihood, "rise_start_frame": pk.rise_start_frame, "fall_end_frame": pk.fall_end_frame,
        "keep": True, "imputed": False, "gate": "", "reason": ""} for rank, pk in enumerate(merged)]

    # Stage 3: F -> D -> E per peak; E only until the first peak is promoted (their Decision 17).
    promoted = None
    min_peak_y_abs = float(detector_params["min_peak_y_abs"])
    for r in decisions:
        pf = r["peak_frame_local"]
        # OURS (2026-10-01, Priya: "let's get rid of gate f"): gate F (raw y std < 1 px within +-20 ms = "flat
        # plateau") rejected REAL licks on our data -- the tongue held against the spout (flat at full extension)
        # and small incomplete licks at the lips (PS93 0908 QC, 7/7 F rejections). `gates.f_enabled: false` in
        # configs/defaults.yaml turns it off; DEFAULTS keep it on so the parity tests still mirror theirs.
        F_rej, F_reason, F_std = (td.gate_F(y_clean, is_interp_fill_slice, is_base_clean, pf, n_frames, fps,
                                            params=gates_params)
                                  if gates_params.get("f_enabled", True) else (False, "", np.nan))
        if F_rej:
            r.update(keep=False, reason=F_reason, gate="F", wp_raw_std=F_std)
            continue
        keep_D, reason_D = td.gate_D(y_clean, is_interp_fill_slice, is_base_clean, pf, n_frames, fps,
                                     params=gates_params)
        if not keep_D:
            r.update(keep=False, reason=reason_D, gate="D", wp_raw_std=F_std)
            continue
        if promoted is None:
            peer_ys = [pr["y"] for pr in decisions if pr["rank"] != r["rank"]]
            E_rej, _E_reason, z, _peer_med = td.gate_E(r["y"], r["lik"], peer_ys, params=gates_params)
            if E_rej:
                # x passed is the UN-pre-cleaned slice (theirs, deliberately): the excise rebuilds x at the new
                # peak from anchors that exclude the outlier window, which needs the original fill values.
                spl = td.parabolic_excise(y_clean, x_slice, is_interp_fill_slice, is_base_clean, pf, n_frames, fps,
                                          params=gates_params)
                if spl is None:
                    spl = td.spline_excise(y_clean, x_slice, is_interp_fill_slice, is_base_clean, pf, n_frames,
                                           fps, params=gates_params)
                if spl is None:
                    r.update(keep=False, reason="E_spline_no_anchors", gate="E_spline_drop", z=z)
                    continue
                if spl["new_peak_y"] < min_peak_y_abs:
                    r.update(keep=False, reason=f"E_spline_low_y new={spl['new_peak_y']:.1f}",
                             gate="E_spline_drop", z=z)
                    continue
                # Impute y + frame. t_ms is NOT updated -- theirs keeps the original peak time.
                r["y_original"] = r["y"]
                r["y"] = float(spl["new_peak_y"])
                r["peak_frame_local"] = int(spl["new_peak_frame"])
                r["new_peak_x"] = float(spl.get("new_peak_x", float("nan")))
                r["imputed"] = True
                r["gate"] = "E_spline_impute"
                r["reason"] = f"E_spline_imputed (orig y={r['y_original']:.0f} -> {r['y']:.0f})"
                r["z"] = z
                promoted = r
                continue
        r["wp_raw_std"] = F_std if "wp_raw_std" not in r else r["wp_raw_std"]
        if promoted is None:
            promoted = r

    # Stage 4: M gate
    td.run_m_gate(decisions, y_clean, likelihood_slice, is_interp_fill_slice, is_base_clean, n_frames, fps,
                  promoted, params=gates_params)

    # Stage 5: kept-lick records
    conf_radius = max(1, int(round(float(gates_params["m_conf_radius_ms"]) / 1000.0 * fps)))
    sign_flip = float(p["angle"]["sign_flip"])
    kept: list[dict[str, Any]] = []
    lick_idx = 0
    for r in decisions:
        if not r["keep"]:
            continue
        pf = int(r["peak_frame_local"])
        lo, hi = max(0, pf - conf_radius), min(n_frames, pf + conf_radius + 1)
        raw_in_win = (~is_interp_fill_slice[lo:hi]) & (~is_base_clean[lo:hi])
        n_raw = int(raw_in_win.sum())
        confidence = 0.0 if n_raw == 0 else n_raw * float(np.mean(likelihood_slice[lo:hi][raw_in_win]))
        rsf, fef = r.get("rise_start_frame", -1), r.get("fall_end_frame", -1)
        rise_start_ms = float(t_ms[rsf]) if 0 <= rsf < n_frames else float("nan")
        fall_end_ms = float(t_ms[fef]) if 0 <= fef < n_frames else float("nan")
        is_imputed = bool(r.get("imputed", False))
        if is_imputed and np.isfinite(float(r.get("new_peak_x", float("nan")))):
            x_at_peak = float(r["new_peak_x"])
        else:
            x_at_peak = float(x_clean[pf]) if 0 <= pf < n_frames else float("nan")

        # Per-lick angle at the (possibly imputed) peak, UNSMOOTHED, from (x_at_peak, y) -- theirs. ADAPTED:
        # our reference frame, and absolute px (+X0, +Y0) since x/y here are baseline-subtracted.
        # OURS: with detection on another trace (detect_on: protrusion), the angle needs the IMAGE y.
        y_at_peak = (float(y_geom_slice[pf]) if y_geom_slice is not None and 0 <= pf < n_frames
                     else float(r["y"]))
        if spout_frame is not None and np.isfinite(x_at_peak) and np.isfinite(y_at_peak):
            peak_angle_deg = float(compute_signed_tongue_angle_deg(
                x_at_peak + X0, y_at_peak + Y0, spout_frame, sign_flip=sign_flip))
        else:
            peak_angle_deg = float("nan")

        # max dy/dt in [rise_start, fall_end] of the CLEANED y with baseline-fill masked (their Decision 5).
        max_velocity_y = float("nan")
        geo: dict[str, Any] = {}
        if p.get("lick_geometry", False) or p.get("velocity_window", "v7") == "visible_rise":
            on, off = lick_extent(y_clean, is_base_clean, pf, float(p["phase"]["extent_slack_px"]))
            yv = np.where(is_base_clean, np.nan, y_clean.astype(np.float64))
            geo = {"on_frame": on, "off_frame": off, "on_ms": float(t_ms[on]), "off_ms": float(t_ms[off]),
                   "max_retract_velocity_y_px_per_s": -_min_dydt(yv[pf:off + 1], fps) if off > pf else float("nan")}
            if p.get("lick_geometry", False):
                xo = 0.0 if X0 is None else X0
                yo = 0.0 if Y0 is None else Y0
                yg = yv if y_geom_slice is None else np.where(is_base_clean, np.nan, y_geom_slice)
                ap, lr = spout_coords(x_clean[on:off + 1] + xo, yg[on:off + 1] + yo, spout_frame)
                dist = np.hypot(ap, lr)
                k = pf - on
                geo.update({"protrusion_px": float(dist[k]), "ap_px": float(ap[k]), "lr_px": float(lr[k]),
                            "protrusion_max_px": float(np.nanmax(dist)) if np.isfinite(dist).any() else float("nan")})
        if p.get("velocity_window", "v7") == "visible_rise":
            max_velocity_y = _max_dydt(yv[geo["on_frame"]:pf + 1], fps) if pf > geo["on_frame"] else float("nan")
        elif 0 <= rsf <= fef < n_frames:
            y_win = np.where(is_base_clean[rsf:fef + 1], np.nan, y_clean[rsf:fef + 1].astype(np.float64))
            if int(np.count_nonzero(np.isfinite(y_win))) >= 2:
                dy_win = np.gradient(y_win) * fps
                if np.isfinite(dy_win).any():
                    max_velocity_y = float(np.nanmax(dy_win))

        kept.append({
            "lick_idx": lick_idx, "t_ms": float(r["t_ms"]), "y": float(r["y"]), "x": x_at_peak,
            "y_original": float(r.get("y_original", float("nan"))),
            "rise_start_ms": rise_start_ms, "fall_end_ms": fall_end_ms,
            "rise_start_frame": int(rsf), "fall_end_frame": int(fef),
            "imputed": is_imputed, "source": str(r["src"]), "confidence": confidence,
            "likelihood": float(r["lik"]), "peak_angle_deg": peak_angle_deg,
            "max_velocity_y_px_per_s": max_velocity_y, **geo,
            **({"y_image": y_at_peak} if y_geom_slice is not None else {}),
        })
        lick_idx += 1

    return V7TrialResult(trial_id=trial_id, position=position, cue_frame=float(cue_frame), y_clean=y_clean,
                         x_clean=x_clean, is_baseline_fill_clean=is_base_clean, t_ms=t_ms, kept_licks=kept,
                         decisions=decisions, preclean_diag=diag, session_frame_lo=session_frame_lo,
                         session_frame_hi=session_frame_hi,
                         y_geom=None if y_geom_slice is None else np.asarray(y_geom_slice, dtype=np.float64))


def _overlay_cleaned(x_final, y_final, is_base, results: list[V7TrialResult]):
    """Session arrays with each trial's cleaned slice written over the input (their `_build_cleaned_trace_df`
    via `_v7_cleaned_bundle_arrays`). In trial order, so a later trial wins an overlap, as theirs."""
    x, y, b = x_final.astype(np.float64).copy(), y_final.astype(np.float64).copy(), is_base.copy()
    for r in results:
        lo, hi = r.session_frame_lo, r.session_frame_hi
        if lo < 0 or hi < 0 or lo > hi or len(r.y_clean) != hi - lo + 1:
            continue
        y[lo:hi + 1] = r.y_clean
        x[lo:hi + 1] = r.x_clean
        b[lo:hi + 1] = r.is_baseline_fill_clean
    return x, y, b


# --------------------------------------------------------------------------- per-trial features

def extract_lick1_lick2(res: V7TrialResult) -> dict[str, float]:
    """lick1/lick2 y, t, x from the first two KEPT licks (their v7 branch of `_extract_lick1_lick2`)."""
    out = {f"lick{k}_{f}": np.nan for k in (1, 2) for f in ("y_peak", "t_peak_ms", "x_at_ypeak")}
    for k, lk in enumerate(res.kept_licks[:2], start=1):
        out[f"lick{k}_y_peak"] = float(lk["y"])
        out[f"lick{k}_t_peak_ms"] = float(lk["t_ms"])
        out[f"lick{k}_x_at_ypeak"] = float(lk["x"])
    return out


def _lick_velocity(res: V7TrialResult, lk: dict, fps: float, min_finite: int,
                   window: str = "v7") -> tuple[float, float, bool]:
    """(vy_peak, vxy_peak, window_ok) for one kept lick, by the v7 branch of `_extract_peak_velocity_first5`:
    max dy/dt and max |(dx, dy)|/dt over [rise_start, fall_end] of the cleaned trace, baseline-fill masked,
    requiring >= min_finite frames with both x and y finite. No smoothing (the v7 branch does none)."""
    n = len(res.y_clean)
    rsf, fef = int(lk.get("rise_start_frame", -1)), int(lk.get("fall_end_frame", -1))
    if window == "visible_rise" and "on_frame" in lk:
        # OURS: the whole visible rise, onset -> peak frame (see DEFAULTS velocity_window)
        rsf, fef = int(lk["on_frame"]), int(np.argmin(np.abs(res.t_ms - float(lk["t_ms"]))))
    if not (0 <= rsf <= fef < n):
        return float("nan"), float("nan"), False
    is_b = res.is_baseline_fill_clean[rsf:fef + 1]
    y_w = np.where(is_b, np.nan, res.y_clean[rsf:fef + 1].astype(np.float64))
    x_w = np.where(is_b, np.nan, res.x_clean[rsf:fef + 1].astype(np.float64))
    vy = vxy = float("nan")
    if int(np.count_nonzero(np.isfinite(y_w) & np.isfinite(x_w))) >= min_finite:
        dy = np.gradient(y_w) * fps
        dx = np.gradient(x_w) * fps
        v = np.sqrt(dx ** 2 + dy ** 2)
        if np.isfinite(dy).any():
            vy = float(np.nanmax(dy))
        if np.isfinite(v).any():
            vxy = float(np.nanmax(v))
    return vy, vxy, True


def extract_peak_velocity_first5(res: V7TrialResult, fps: float, *, p: dict) -> dict[str, np.ndarray]:
    """First five kept licks inside peak_velocity_detect_win_ms: vy / vxy peaks and times, NaN-padded to 5.
    A lick with no valid rise/fall window still takes a slot (time set, velocity NaN), as theirs."""
    lo, hi = p["peak_velocity_detect_win_ms"]
    min_finite = int(p["detector"]["min_finite_samples_per_lick"])
    vy_out, vxy_out, t_out = (np.full(5, np.nan) for _ in range(3))
    j = 0
    for lk in res.kept_licks:
        if j >= 5:
            break
        t = float(lk["t_ms"])
        if t < lo or t > hi:
            continue
        vy, vxy, _ = _lick_velocity(res, lk, fps, min_finite, p.get("velocity_window", "v7"))
        vy_out[j], vxy_out[j], t_out[j] = vy, vxy, t
        j += 1
    return {"vy_peaks_bylick": vy_out, "vxy_peaks_bylick": vxy_out, "lick_t_peaks_bylick": t_out}


def extract_n_licks_family(y_t, x_t, t_ms, fps, *, p: dict) -> dict[str, Any]:
    """Slope-detector lick count on the CLEANED slice (their v7 routing of `_extract_n_licks_family`).

    NOT the kept-lick count: theirs re-runs the slope detector on the v7-cleaned trace over 70-8000 ms, so
    v7-rejected protrusions (wiped to 0) cannot count, but the gates F/D/E/M are not applied either.
    """
    detect_win = tuple(p["lick_count_detect_win_ms"])
    apply_win = tuple(p["lick_count_apply_win_ms"])
    bin_win = tuple(p["lick_count_bin_win_ms"])
    thr_r = float(p["direction_thresholds"]["img_right_thr_px"])
    thr_l = float(p["direction_thresholds"]["img_left_thr_px"])
    ge10 = int(p["n_licks_categorical"]["lick_ge10_thr"])
    n_bins = int(p["n_licks_binning"]["bin_count"])
    bin_ms = float(p["n_licks_binning"]["bin_ms"])
    peaks = td.slope_detect_lick_peaks(y_t, x_t, t_ms, fps, detect_win_ms=detect_win,
                                       detector_params=p["detector"], n_max=int(p["max_licks_per_trial"]))
    if len(peaks) == 0:
        return {"n_licks": 0, "n_img_right": 0, "n_img_left": 0, "n_other": 0, "lick_ge10": False,
                "n_licks_per_bin": np.zeros(n_bins, dtype=np.int64)}
    in_apply = (peaks.peak_times_ms >= apply_win[0]) & (peaks.peak_times_ms <= apply_win[1])
    n_licks = int(np.count_nonzero(in_apply))
    x_apply = peaks.peak_x[in_apply]
    n_r = int(np.count_nonzero(x_apply > thr_r))      # theirs: n_left  (x > x_left_thr_px)
    n_l = int(np.count_nonzero(x_apply < thr_l))      # theirs: n_right (x < x_right_thr_px)
    edges = np.array([bin_win[0] + i * bin_ms for i in range(n_bins + 1)], dtype=np.float64)
    in_bin = (peaks.peak_times_ms >= edges[0]) & (peaks.peak_times_ms < edges[-1])
    counts = (np.histogram(peaks.peak_times_ms[in_bin], bins=edges)[0] if np.any(in_bin)
              else np.zeros(n_bins, dtype=np.int64))
    return {"n_licks": n_licks, "n_img_right": n_r, "n_img_left": n_l, "n_other": int(n_licks - n_r - n_l),
            "lick_ge10": bool(n_licks >= ge10), "n_licks_per_bin": counts.astype(np.int64)}


def _kept_times_in(kept_licks, win):
    """Sorted finite kept-lick times inside [win0, win1] (the shared kept_licks branch of theirs)."""
    return np.asarray(sorted(float(lk["t_ms"]) for lk in kept_licks
                             if np.isfinite(float(lk["t_ms"])) and win[0] <= float(lk["t_ms"]) <= win[1]),
                      dtype=np.float64)


def licking_dynamics_reductions(kept_licks, *, p: dict) -> dict[str, float | int]:
    """bout_count (multi-lick bouts), mean valid ILI in the first multi-lick bout, lick rate over the apply
    window -- kept_licks branch of their `_extract_licking_dynamics_reductions`."""
    apply_win = tuple(p["licking_dyn_apply_win_ms"])
    b = p["bout"]
    max_ili, min_ili, min_in_bout = float(b["max_ili_ms"]), float(b["min_ili_ms"]), int(b["min_licks_in_bout_for_ili"])
    t = _kept_times_in(kept_licks, apply_win)
    dur_s = (apply_win[1] - apply_win[0]) / 1000.0
    rate = float(len(t) / dur_s) if dur_s > 0 else float("nan")
    multi = [bb for bb in build_bouts(t, max_ili) if len(bb) >= 2]
    out = {"bout_count": int(len(multi)), "mean_within_bout_ili_ms": float("nan"), "lick_rate_hz": rate}
    if not multi or len(multi[0]) < min_in_bout:
        return out
    ilis = np.diff(np.sort(t[multi[0]]))
    valid = (ilis >= min_ili) & (ilis <= max_ili)
    if np.any(valid):
        out["mean_within_bout_ili_ms"] = float(np.mean(ilis[valid]))
    return out


def own_bout_scalars(kept_licks, *, p: dict) -> dict[str, float | int]:
    """n_peaks, n_bouts, n_multi_lick_bouts, first_peak_t_ms, n_licks_in_first_multi_bout over
    angle_apply_win_ms -- kept_licks branch of their `_extract_cell50_own_bout_scalars`."""
    t = _kept_times_in(kept_licks, tuple(p["angle_apply_win_ms"]))
    bouts = build_bouts(t, float(p["bout"]["max_ili_ms"]))
    multi = [bb for bb in bouts if len(bb) >= 2]
    return {"n_peaks": int(len(t)), "n_bouts": len(bouts), "n_multi_lick_bouts": len(multi),
            "first_peak_t_ms": float(t.min()) if len(t) else float("nan"),
            "n_licks_in_first_multi_bout": int(len(multi[0])) if multi else 0}


def _moving_avg_window(arr, win_n):
    """Centred NaN-tolerant moving average over ``win_n`` bins."""
    win_n = max(1, int(win_n))
    if win_n <= 1:
        return arr.astype(np.float64, copy=True)
    finite = np.isfinite(arr).astype(np.float64)
    safe = np.where(np.isfinite(arr), arr, 0.0)
    k = np.ones(win_n, dtype=np.float64)
    num, den = np.convolve(safe, k, mode="same"), np.convolve(finite, k, mode="same")
    out = np.full_like(arr, np.nan, dtype=np.float64)
    nz = den > 0
    out[nz] = num[nz] / den[nz]
    return out


def licking_dynamics_traces(per_trial_peaks_by_position: dict[str, list[np.ndarray]], *, p: dict) -> pd.DataFrame:
    """Per-POSITION time-binned traces over the apply window (their `_extract_licking_dynamics_traces`, which
    grouped by L/R side): smoothed lick rate, within-bout ILI rate, P(lick), P(within-bout lick)."""
    apply_win = tuple(p["licking_dyn_apply_win_ms"])
    ld = p["licking_dynamics"]
    bin_ms = float(ld["bin_ms"])
    min_trials_per_bin = int(ld["min_trials_per_bin"])
    max_ili_ms = float(p["bout"]["max_ili_ms"])
    min_in_bout = int(p["bout"]["min_licks_in_bout_for_ili"])
    edges = np.arange(apply_win[0], apply_win[1] + bin_ms, bin_ms)
    centers = 0.5 * (edges[:-1] + edges[1:])
    n_bins = len(centers)
    sig_l, sig_i = float(ld["lickrate_smooth_sigma_ms"]) / bin_ms, float(ld["ili_smooth_sigma_ms"]) / bin_ms
    mov_n = max(1, int(round(float(ld["prob_smooth_movavg_ms"]) / bin_ms)))

    rows = []
    for pos, lists in per_trial_peaks_by_position.items():
        n_trials = len(lists)
        if n_trials == 0:
            rows += [{"timebin_ms": float(c), "position": pos, "lick_rate_per_t": np.nan,
                      "ili_within_bout_rate_per_t": np.nan, "p_lick_per_t": np.nan, "p_within_bout_per_t": np.nan,
                      "n_trials": 0} for c in centers]
            continue
        lick_count = np.zeros(n_bins)
        ili_sum = np.zeros(n_bins)
        ili_n = np.zeros(n_bins, dtype=np.int64)
        n_with_lick = np.zeros(n_bins, dtype=np.int64)
        n_with_wb = np.zeros(n_bins, dtype=np.int64)
        for tp in lists:
            if len(tp) == 0:
                continue
            counts, _ = np.histogram(tp, bins=edges)
            lick_count += counts
            n_with_lick += (counts > 0).astype(np.int64)
            st = np.sort(tp)
            wb: list[float] = []
            for bb in build_bouts(st, max_ili_ms):
                if len(bb) < min_in_bout:
                    continue
                bt = st[bb]
                ilis = np.diff(bt)
                for bi, dt in zip(np.digitize(bt[1:], edges) - 1, ilis):
                    if 0 <= bi < n_bins and dt > 0:
                        ili_sum[bi] += 1000.0 / float(dt)
                        ili_n[bi] += 1
                wb.extend(bt.tolist())
            if wb:
                n_with_wb += (np.histogram(np.asarray(wb), bins=edges)[0] > 0).astype(np.int64)
        rate = lick_count / n_trials / (bin_ms / 1000.0)
        # NB (theirs): `min_trials_per_bin` is compared against the number of ILIs in the bin, not trials.
        ili_mean = np.where(ili_n >= min_trials_per_bin, ili_sum / np.maximum(ili_n, 1), np.nan)
        rate_s = gaussian_smooth_nan(rate, sig_l)
        ili_s = gaussian_smooth_nan(ili_mean, sig_i)
        pl_s = _moving_avg_window(n_with_lick / n_trials, mov_n)
        pw_s = _moving_avg_window(n_with_wb / n_trials, mov_n)
        rows += [{"timebin_ms": float(c), "position": pos, "lick_rate_per_t": float(rate_s[i]),
                  "ili_within_bout_rate_per_t": float(ili_s[i]), "p_lick_per_t": float(pl_s[i]),
                  "p_within_bout_per_t": float(pw_s[i]), "n_trials": n_trials} for i, c in enumerate(centers)]
    return pd.DataFrame(rows)


def bout_rows_for_trial(res: V7TrialResult, angle_per_frame, fps, n_session: int, *, p: dict):
    """Bout-summary rows + per-lick bout labels, from kept licks (kept_licks branch of their
    `_build_visual_bundle_per_trial` / `_kept_licks_to_bundle_peaks`).

    Returns (rows, lick_labels) where lick_labels maps lick_idx -> (bout_idx, multi_bout_rank,
    angle_smoothed_at_peak_deg). The peak angle is the SMOOTHED per-frame angle at the frame nearest the lick
    time on the bout-table plot grid, as theirs (unlike the kept lick's own `peak_angle_deg`).
    """
    bt = p["bout_table"]
    lo_w, hi_w = float(bt["detect_win_ms"][0]), float(bt["detect_win_ms"][1])
    max_n = int(p["max_licks_per_trial"])
    # Lick time -> nearest frame on the plot-window grid, clipped to the session, exactly as theirs (the grid
    # is all this slice is used for, so a dummy array of the session length stands in for the data).
    _, t_grid, lo_plot, _ = extract_trial_slice(np.empty(n_session), res.cue_frame, tuple(bt["plot_win_ms"]), fps)
    if angle_per_frame is not None and t_grid.size:
        angle_snip = angle_per_frame[lo_plot:lo_plot + t_grid.size]
    else:
        angle_snip = np.full(t_grid.size, np.nan)
    times, ys, idxs, lick_ids = [], [], [], []
    if res.kept_licks and t_grid.size:
        for lk in res.kept_licks:
            t = float(lk["t_ms"])
            if not np.isfinite(t) or t < lo_w or t > hi_w:
                continue
            fi = int(np.clip(np.searchsorted(t_grid, t), 0, t_grid.size - 1))
            if fi > 0 and abs(t_grid[fi - 1] - t) < abs(t_grid[fi] - t):
                fi -= 1
            times.append(t)
            ys.append(float(lk.get("y", np.nan)))
            idxs.append(fi)
            lick_ids.append(int(lk["lick_idx"]))
            if max_n and len(times) >= max_n:
                break
    times, ys, idxs = np.asarray(times, float), np.asarray(ys, float), np.asarray(idxs, np.int64)

    def ang(li):
        if li < 0 or li >= len(angle_snip):
            return float("nan")
        v = angle_snip[li]
        return float(v) if np.isfinite(v) else float("nan")

    rows, labels, multi_seen = [], {}, 0
    for bout_idx, ix in enumerate(build_bouts(times, float(p["bout"]["max_ili_ms"]))):
        is_multi = len(ix) >= 2
        mrank = float(multi_seen) if is_multi else float("nan")
        multi_seen += int(is_multi)
        pt, py, pi = times[ix], ys[ix], idxs[ix]
        t0, t1 = _bout_time_bounds_ms(pt, tuple(p["bout"]["single_bout_win_ms"]))
        rows.append({
            "trial_id": res.trial_id, "position": res.position, "bout_idx": bout_idx, "multi_bout_rank": mrank,
            "n_peaks": int(len(ix)), "bout_t0_ms": t0, "bout_t1_ms": t1,
            "peak_t0_ms": float(pt[0]), "peak_t1_ms": float(pt[1]) if len(pt) >= 2 else float("nan"),
            "peak_y0": float(py[0]), "peak_angle0": ang(int(pi[0])),
            "peak_y1": float(py[1]) if len(pt) >= 2 else float("nan"),
            "peak_angle1": ang(int(pi[1])) if len(pt) >= 2 else float("nan")})
        for j, k in enumerate(ix):
            labels[lick_ids[k]] = (bout_idx, mrank, ang(int(pi[j])))
    return rows, labels


def within_bout_ili_ms(results: list[V7TrialResult], max_ili_ms: float = 300.0) -> np.ndarray:
    """Intervals between ADJACENT kept-lick peaks of a trial that belong to the same bout (<= max_ili_ms) --
    the ILIs stroke_orofacial pools for `pre_stroke_median_ili_ms` (adjacent peaks of a multi-lick bout)."""
    out = []
    for r in results:
        t = np.sort([float(k["t_ms"]) for k in r.kept_licks])
        d = np.diff(t)
        out += d[(d > 0) & (d <= max_ili_ms)].tolist()
    return np.asarray(out, dtype=np.float64)


def lick_phase_table(results: list[V7TrialResult], X0, Y0, spout_frame: SpoutFrame | None, fps: float, *,
                     n_points: int = 21, spout_angle_by_trial: dict | None = None, mode: str = "extent",
                     cycle_ms: float | None = None) -> pd.DataFrame:
    """OURS (Priya 2026-10-01: angle-over-lick-phase plots are among the most informative): every kept lick
    resampled on a phase axis -- 0 = the frame the tongue appears (`on_frame`), 0.5 = the peak, 1 = the last
    visible frame (`off_frame`); rise and fall are each stretched to half the axis, so licks of different
    length line up.

    Per (lick, phase): ap_px / lr_px (spout frame, mouth origin), protrusion_px = hypot, angle_deg =
    atan2(lr, ap) (unsmoothed; + = image-right, as `compute_signed_tongue_angle_deg`), plus the lick's rise /
    fall ms. Linear interpolation between visible frames; NaN where the lick has no rise or no fall frame.
    """
    if mode == "centered":
        return _centered_phase_table(results, X0, Y0, spout_frame, fps, n_points=n_points,
                                     spout_angle_by_trial=spout_angle_by_trial, cycle_ms=cycle_ms)
    grid = np.linspace(0.0, 1.0, n_points)
    rows = []
    xo = 0.0 if X0 is None else X0
    yo = 0.0 if Y0 is None else Y0
    for r in results:
        yv = np.where(r.is_baseline_fill_clean, np.nan,
                      (r.y_clean if r.y_geom is None else r.y_geom).astype(np.float64))
        sa = (spout_angle_by_trial or {}).get(r.trial_id, np.nan)
        for lk in r.kept_licks:
            if "on_frame" not in lk:
                continue
            on, off = int(lk["on_frame"]), int(lk["off_frame"])
            pf = int(np.argmin(np.abs(r.t_ms - float(lk["t_ms"]))))
            ap, lr = spout_coords(r.x_clean[on:off + 1] + xo, yv[on:off + 1] + yo, spout_frame)
            f = np.arange(on, off + 1, dtype=np.float64)
            # frame -> phase: rise [on, pf] -> [0, .5], fall [pf, off] -> [.5, 1]
            ph = np.where(f <= pf, 0.5 * (f - on) / max(pf - on, 1), 0.5 + 0.5 * (f - pf) / max(off - pf, 1))
            ok = np.isfinite(ap) & np.isfinite(lr)
            if pf - on < 1 or off - pf < 1 or ok.sum() < 3:
                a_g = l_g = np.full(n_points, np.nan)
            else:
                a_g, l_g = np.interp(grid, ph[ok], ap[ok]), np.interp(grid, ph[ok], lr[ok])
            for g, a_, l_ in zip(grid, a_g, l_g):
                ang_ = float(np.degrees(np.arctan2(l_, a_)))
                rows.append((r.trial_id, r.position, int(lk["lick_idx"]), float(lk["t_ms"]), float(g), a_, l_,
                             float(np.hypot(a_, l_)), ang_, ang_ - sa, (pf - on) * 1000.0 / fps,
                             (off - pf) * 1000.0 / fps))
    return pd.DataFrame(rows, columns=["trial_id", "position", "lick_idx", "t_ms", "phase", "ap_px", "lr_px",
                                       "protrusion_px", "angle_deg", "delta_angle_deg", "rise_ms", "fall_ms"])


def _centered_phase_table(results, X0, Y0, spout_frame, fps, *, n_points, spout_angle_by_trial, cycle_ms):
    """Centered-on-peak phase (stroke_orofacial): for each kept lick, the frames at peak_t + (phase - 0.5) *
    cycle_ms (nearest frame), tongue position where visible, NaN where the tongue is in (baseline fill) or the
    window leaves the trial slice. The window spans half an ILI each side, so it can reach into a neighbouring
    lick in a fast bout -- as theirs. rise_ms / fall_ms carry the lick's own visible extent for reference."""
    if cycle_ms is None:
        ili = within_bout_ili_ms(results)
        cycle_ms = float(np.median(ili)) if len(ili) else float("nan")
    grid = np.linspace(0.0, 1.0, n_points)
    rows = []
    xo = 0.0 if X0 is None else X0
    yo = 0.0 if Y0 is None else Y0
    cols = ["trial_id", "position", "lick_idx", "t_ms", "phase", "ap_px", "lr_px", "protrusion_px", "angle_deg",
            "delta_angle_deg", "rise_ms", "fall_ms", "cycle_ms", "visible", "in_slice", "protrusion_filled_px"]
    if not np.isfinite(cycle_ms):
        return pd.DataFrame(columns=cols)
    # OURS (Priya 2026-10-02): a frame with the tongue IN (baseline fill) is not "missing" for protrusion -- the
    # tongue is at / behind the lips. `protrusion_filled_px` puts it at this session's LIP LEVEL (median distance
    # at which a lick's tongue first appears), so means over phase are not biased toward the licks still out;
    # angles stay NaN there (no direction without a tongue). `visible` / `in_slice` let a plot mask phase points
    # where too few licks show the tongue (`phase.min_coverage`).
    firsts = []
    for r in results:
        yv0 = np.where(r.is_baseline_fill_clean, np.nan, (r.y_clean if r.y_geom is None else r.y_geom).astype(float))
        a0, l0 = spout_coords(r.x_clean + xo, yv0 + yo, spout_frame)
        d0 = np.hypot(a0, l0)
        firsts += [d0[int(k["on_frame"])] for k in r.kept_licks if "on_frame" in k and 0 <= int(k["on_frame"]) < len(d0)]
    firsts = np.asarray(firsts, dtype=float)
    lip_px = float(np.nanmedian(firsts)) if np.isfinite(firsts).any() else 0.0
    for r in results:
        yv = np.where(r.is_baseline_fill_clean, np.nan,
                      (r.y_clean if r.y_geom is None else r.y_geom).astype(np.float64))
        ap_all, lr_all = spout_coords(r.x_clean + xo, yv + yo, spout_frame)
        sa = (spout_angle_by_trial or {}).get(r.trial_id, np.nan)
        dt = float(r.t_ms[1] - r.t_ms[0]) if len(r.t_ms) > 1 else 1000.0 / fps
        for lk in r.kept_licks:
            tq = float(lk["t_ms"]) + (grid - 0.5) * cycle_ms
            f = np.rint((tq - r.t_ms[0]) / dt).astype(int)
            ok = (f >= 0) & (f < len(r.t_ms))
            a_ = np.where(ok, ap_all[np.clip(f, 0, len(r.t_ms) - 1)], np.nan)
            l_ = np.where(ok, lr_all[np.clip(f, 0, len(r.t_ms) - 1)], np.nan)
            ang = np.degrees(np.arctan2(l_, a_))
            pf = int(np.argmin(np.abs(r.t_ms - float(lk["t_ms"]))))
            rise = (pf - lk["on_frame"]) * 1000.0 / fps if "on_frame" in lk else np.nan
            fall = (lk["off_frame"] - pf) * 1000.0 / fps if "off_frame" in lk else np.nan
            for g, aa, ll, an, ins in zip(grid, a_, l_, ang, ok):
                vis = bool(ins and np.isfinite(aa) and np.isfinite(ll))
                prot = float(np.hypot(aa, ll))
                rows.append((r.trial_id, r.position, int(lk["lick_idx"]), float(lk["t_ms"]), float(g), aa, ll,
                             prot, float(an), float(an) - sa, rise, fall, cycle_ms, vis, bool(ins),
                             (prot if vis else (lip_px if ins else np.nan))))
    return pd.DataFrame(rows, columns=cols)


def phase_mean(ph: pd.DataFrame, var: str, *, by=("phase",), min_coverage: float = 0.2):
    """Mean and SEM of ``var`` per phase (or ``by``). For direction variables, phase points where fewer than
    ``min_coverage`` of the licks show a VISIBLE tongue are set to NaN (protrusion_filled_px is never masked) (OURS, Priya 2026-10-02: hide the ends where only the few licks still
    out set the mean; generous by default). Coverage = visible / in-slice rows; tables without `visible` (extent
    mode) are not masked."""
    g = ph.groupby(list(by))
    mu, se = g[var].mean(), g[var].std() / np.sqrt(g[var].count())
    # protrusion with tongue-in frames filled at the lip level is DEFINED everywhere in the slice -> not masked;
    # the coverage rule is for direction (angle, deviation, ap / lr), which needs a visible tongue
    if "visible" in ph and "in_slice" in ph and var != "protrusion_filled_px":
        cov = g.visible.sum() / g.in_slice.sum().clip(lower=1)
        bad = cov < min_coverage
        mu, se = mu.where(~bad), se.where(~bad)
    return mu, se


def trial_spout_reference(trials: list, spout_xy, fps: float, frame: SpoutFrame | None, *, p: dict) -> pd.DataFrame:
    """OURS: per trial, the spout tip (median of frames with spout likelihood >= spout_ref.lk_thr from the cue to
    the trial stop, else the fallback window) in image px and in the spout frame: spout_angle_deg =
    atan2(lr, ap) (same convention as the tongue angle), spout_dist_px from the mouth. ``spout_xy`` = (x, y,
    likelihood) session arrays aligned with the tongue arrays."""
    sx, sy, sl = (np.asarray(a, dtype=np.float64) for a in spout_xy)
    thr = float(p["spout_ref"]["lk_thr"])
    fb = p["spout_ref"]["fallback_win_ms"]
    n = len(sx)
    rows = []
    for tr in trials:
        c = float(tr.cue_frame)
        stop = TW.stop_ms_of(tr.cue_frame, tr.stop_frame, fps)
        lo_ms, hi_ms = (0.0, stop) if stop is not None else (float(fb[0]), float(fb[1]))
        lo, hi = int(max(0, np.floor(c + lo_ms / 1000.0 * fps))), int(min(n, np.ceil(c + hi_ms / 1000.0 * fps) + 1))
        ok = np.isfinite(sx[lo:hi]) & np.isfinite(sy[lo:hi]) & (sl[lo:hi] >= thr) if hi > lo else np.zeros(0, bool)
        if ok.sum() == 0:
            rows.append((tr.trial_id, np.nan, np.nan, np.nan, np.nan, np.nan, 0))
            continue
        x_, y_ = float(np.median(sx[lo:hi][ok])), float(np.median(sy[lo:hi][ok]))
        ap, lr = spout_coords(x_, y_, frame)
        rows.append((tr.trial_id, x_, y_, float(ap), float(lr), float(np.degrees(np.arctan2(lr, ap))), int(ok.sum())))
    t = pd.DataFrame(rows, columns=["trial_id", "spout_x_img", "spout_y_img", "spout_ap_px", "spout_lr_px",
                                    "spout_angle_deg", "spout_n_frames"])
    t["spout_dist_px"] = np.hypot(t.spout_ap_px, t.spout_lr_px)
    return t


# --------------------------------------------------------------------------- driver

@dataclass
class TongueKinematics:
    """Everything `compute_tongue_kinematics` produces."""
    per_trial: pd.DataFrame
    per_lick: pd.DataFrame
    bouts: pd.DataFrame
    licking_dynamics_traces: pd.DataFrame
    x_clean: np.ndarray                      # session arrays, v7-cleaned inside trial slices
    y_clean: np.ndarray
    is_baseline_fill_clean: np.ndarray
    angle_per_frame: np.ndarray | None       # smoothed signed angle (deg) per session frame; None w/o SpoutFrame
    trial_results: list[V7TrialResult]
    params: dict = field(default_factory=dict)
    lick_phase: pd.DataFrame | None = None   # OURS: `lick_phase_table` (with lick_geometry)
    X0: float | None = None                  # OURS: what `lick_phase_table` needs to rebuild the phase table
    Y0: float | None = None
    spout_frame: SpoutFrame | None = None
    fps: float = 250.0
    spout_angle_by_trial: dict | None = None


def compute_tongue_kinematics(x_final, y_final, fill_method, likelihood, trials: Iterable[Trial] | pd.DataFrame, *,
                              fps: float = 250.0, X0: float | None = None, Y0: float | None = None,
                              spout_frame: SpoutFrame | None = None, params_override: dict | None = None,
                              spout_xy=None, contacts_ms: dict | None = None) -> TongueKinematics:
    """v7 per-trial pipeline + features over a session (the v7 path of their `_build_tongue_pertrial_core`).

    ``x_final, y_final, fill_method, likelihood`` are `orofacial_clean.Clean` x_final / y_final / fill_method /
    lk (baseline-subtracted, protrusion = +y). ``X0, Y0`` (`Clean.X0/Y0`) are needed only for angles.
    OURS: ``spout_xy`` = (x, y, likelihood) image-px spout arrays aligned with the tongue arrays -> per-trial
    spout reference + per-lick reach accuracy (needs lick_geometry); ``contacts_ms`` = {trial_id: DAQ spout-
    contact onsets, ms from the cue} -> per-lick `contact`.
    """
    p = params(params_override)
    if isinstance(trials, pd.DataFrame):
        trials = trials_from_frame(trials)
    trials = list(trials)
    if spout_frame is not None and (X0 is None or Y0 is None):
        raise ValueError("angles need absolute coordinates: pass X0, Y0 (orofacial_clean.Clean.X0/.Y0)")
    x_final = np.asarray(x_final, dtype=np.float64)
    y_final = np.asarray(y_final, dtype=np.float64)
    lik = np.asarray(likelihood, dtype=np.float64)
    is_intp, is_base = td.interp_masks(fill_method)
    y_img = None
    if p.get("detect_on", "y") == "protrusion":
        # OURS: detect on the tongue-mouth distance; keep the image y for geometry / angles.
        if spout_frame is None or X0 is None or Y0 is None:
            raise ValueError("detect_on: protrusion needs a SpoutFrame and X0, Y0")
        y_img = y_final
        d = np.hypot(*spout_coords(x_final + X0, y_final + Y0, spout_frame))
        y_final = np.where(is_base, 0.0, d)
    # OURS: per-trial params -- post-cue windows end at each trial's own stop when it is known (trial_windows).
    p_tr = [TW.end_at_stop(p, TW.stop_ms_of(tr.cue_frame, tr.stop_frame, fps)) for tr in trials]

    # Pre-pass: v7 per trial (theirs computes these once, up front, so the cleaned session exists before the
    # angle is taken).
    results: list[V7TrialResult] = []
    for tr, pt in zip(trials, p_tr):
        ys, t_ms, lo, hi = extract_trial_slice(y_final, tr.cue_frame, tuple(pt["trial_slice_win_ms"]), fps)
        sl = slice(lo, hi + 1)
        results.append(run_v7_pipeline_per_trial(
            ys, x_final[sl], is_intp[sl], is_base[sl], lik[sl], t_ms, fps, p=pt, trial_id=tr.trial_id,
            position=tr.position, cue_frame=tr.cue_frame, session_frame_lo=lo, session_frame_hi=hi,
            X0=X0, Y0=Y0, spout_frame=spout_frame, y_geom_slice=None if y_img is None else y_img[sl]))

    x_c, y_c, b_c = _overlay_cleaned(x_final, y_final, is_base, results)
    y_g = y_c if y_img is None else np.where(b_c, 0.0, y_img)      # image y for angles (OURS: protrusion mode)
    angle = (per_frame_angle_smoothed(x_c + X0, y_g + Y0, spout_frame, fps, p=p)
             if spout_frame is not None else None)
    sref = (trial_spout_reference(trials, spout_xy, fps, spout_frame, p=p).set_index("trial_id")
            if spout_xy is not None else None)
    match_ms = float(p["contact"]["match_ms"])
    match_mode = p["contact"].get("match", "peak")
    span_pad = float(p["contact"].get("span_pad_ms", 8.0))

    peak_pad_ms = float(p["detector"]["peak_pad_ms"])
    min_finite = int(p["detector"]["min_finite_samples_per_lick"])
    min_frac = p["angle"].get("max_signed_min_frac_of_peak")
    dist = None
    if angle is not None and min_frac is not None:
        dist = np.hypot(*spout_coords(x_c + X0, np.where(b_c, np.nan, y_g) + Y0, spout_frame))
    by_pos: dict[str, list[np.ndarray]] = {pos: [] for pos in POSITIONS}
    trial_rows, lick_rows, bout_rows = [], [], []
    for tr, res, pt in zip(trials, results, p_tr):
        l12 = extract_lick1_lick2(res)
        vel = extract_peak_velocity_first5(res, fps, p=pt)
        nl = extract_n_licks_family(res.y_clean, res.x_clean, res.t_ms, fps, p=pt)
        ldr = licking_dynamics_reductions(res.kept_licks, p=pt)
        own = own_bout_scalars(res.kept_licks, p=pt)
        a_v = np.array([angle_at_lick_timestamp(angle, fps, tr.cue_frame, t) for t in vel["lick_t_peaks_bylick"]])
        ext = None
        if p["angle"].get("max_signed_within_lick", False):
            by_t = {float(k["t_ms"]): (res.session_frame_lo + k["on_frame"], res.session_frame_lo + k["off_frame"])
                    for k in res.kept_licks if "on_frame" in k}
            ext = [by_t.get(float(t)) for t in vel["lick_t_peaks_bylick"]]
        a_max = angle_max_signed_per_lick(angle, fps, tr.cue_frame, vel["lick_t_peaks_bylick"], peak_pad_ms,
                                          dist_per_frame=dist, min_frac_of_peak=min_frac, extents=ext)
        by_pos[tr.position].append(_kept_times_in(res.kept_licks, tuple(pt["licking_dyn_apply_win_ms"])))
        brows, labels = bout_rows_for_trial(res, angle, fps, len(y_final), p=pt)
        bout_rows += brows

        d = res.preclean_diag
        row = {"trial_id": tr.trial_id, "position": tr.position, "cue_frame": tr.cue_frame,
               "response_end_ms": TW.stop_ms_of(tr.cue_frame, tr.stop_frame, fps),
               "n_kept_licks": len(res.kept_licks), "n_detected_peaks": len(res.decisions), **l12,
               "lick1_angle_at_ypeak": angle_at_lick_timestamp(angle, fps, tr.cue_frame, l12["lick1_t_peak_ms"]),
               "lick2_angle_at_ypeak": angle_at_lick_timestamp(angle, fps, tr.cue_frame, l12["lick2_t_peak_ms"])}
        for j in range(5):
            row[f"vy_peak_lick{j + 1}"] = vel["vy_peaks_bylick"][j]
            row[f"vxy_peak_lick{j + 1}"] = vel["vxy_peaks_bylick"][j]
            row[f"t_peak_lick{j + 1}_ms"] = vel["lick_t_peaks_bylick"][j]
            row[f"angle_at_vpeak_lick{j + 1}"] = a_v[j]
            row[f"angle_max_signed_lick{j + 1}"] = a_max[j]
        row.update({k: v for k, v in nl.items() if k != "n_licks_per_bin"})
        for j, c in enumerate(nl["n_licks_per_bin"]):
            row[f"n_licks_bin{j + 1}"] = int(c)
        row.update(ldr)
        row.update(own)
        row.update({"pc_n_artifact_clusters": d.n_artifact_clusters, "pc_n_artifact_frames": d.n_artifact_frames,
                    "pc_n_frame_outliers": d.n_frame_outliers, "pc_n_pchip_extended": d.n_pchip_extended,
                    "pc_n_interp_wiped": d.n_interp_wiped, "pc_n_x_hard_outliers": d.n_x_hard_outliers,
                    "pc_n_x_joint_outliers": d.n_x_joint_outliers})
        cts = None if contacts_ms is None else np.asarray(contacts_ms.get(tr.trial_id, []), dtype=np.float64)
        if sref is not None and tr.trial_id in sref.index:
            row.update(sref.loc[tr.trial_id].to_dict())
        trial_rows.append(row)
        n_lick_contact = 0

        for lk in res.kept_licks:
            vy, vxy, _ = _lick_velocity(res, lk, fps, min_finite, pt.get("velocity_window", "v7"))
            bidx, mrank, a_sm = labels.get(lk["lick_idx"], (np.nan, np.nan, np.nan))
            lick_rows.append({
                "trial_id": tr.trial_id, "position": tr.position, "cue_frame": tr.cue_frame, **lk,
                "x_img": lk["x"] + X0 if X0 is not None else np.nan,
                "y_img": lk.get("y_image", lk["y"]) + Y0 if Y0 is not None else np.nan,
                "vy_peak_px_per_s": vy, "vxy_peak_px_per_s": vxy,
                "bout_idx": bidx, "multi_bout_rank": mrank, "angle_smoothed_at_peak_deg": a_sm})
            lr_ = lick_rows[-1]
            if cts is not None:                                      # OURS: DAQ spout contact
                dmin = float(np.min(np.abs(cts - lk["t_ms"]))) if len(cts) else np.inf
                if match_mode == "span" and np.isfinite(lk.get("rise_start_ms", np.nan))                         and np.isfinite(lk.get("fall_end_ms", np.nan)):
                    # the detector's rise start / fall end are narrower than the physical lick: contacts land
                    # ~28 ms before the peak, while the tongue still extends (PS93 0814) -> widen to peak -60 ..
                    # +120 ms (extension + one retraction of a ~156 ms lick cycle)
                    lo = min(lk["rise_start_ms"], lk["t_ms"] - 60.0) - span_pad
                    hi = max(lk["fall_end_ms"], lk["t_ms"] + 120.0) + span_pad
                    lr_["contact"] = bool(np.any((cts >= lo) & (cts <= hi)))
                else:
                    lr_["contact"] = bool(dmin <= match_ms)
                lr_["contact_dist_ms"] = dmin
                n_lick_contact += int(lr_["contact"])
            if sref is not None and "ap_px" in lk and tr.trial_id in sref.index:   # OURS: reach accuracy
                s_ = sref.loc[tr.trial_id]
                ang_peak = float(np.degrees(np.arctan2(lk["lr_px"], lk["ap_px"])))
                lr_.update({"spout_angle_deg": s_.spout_angle_deg, "spout_dist_px": s_.spout_dist_px,
                            "delta_angle_deg": ang_peak - s_.spout_angle_deg,
                            "tip_to_spout_px": float(np.hypot(lk["ap_px"] - s_.spout_ap_px, lk["lr_px"] - s_.spout_lr_px)),
                            "reach_frac": lk["protrusion_px"] / s_.spout_dist_px if s_.spout_dist_px > 0 else np.nan})
        if cts is not None:
            trial_rows[-1].update({"n_daq_contacts": int(len(cts)), "n_licks_contact": n_lick_contact,
                                   "n_licks_no_contact": len(res.kept_licks) - n_lick_contact})

    lick_cols = ["trial_id", "position", "cue_frame", "lick_idx", "t_ms", "y", "x", "y_original", "rise_start_ms",
                 "fall_end_ms", "rise_start_frame", "fall_end_frame", "imputed", "source", "confidence", "likelihood",
                 "peak_angle_deg", "max_velocity_y_px_per_s", "x_img", "y_img", "vy_peak_px_per_s",
                 "vxy_peak_px_per_s", "bout_idx", "multi_bout_rank", "angle_smoothed_at_peak_deg"]
    extra = ["on_frame", "off_frame", "on_ms", "off_ms", "max_retract_velocity_y_px_per_s", "protrusion_px",
             "ap_px", "lr_px", "protrusion_max_px", "y_image", "contact", "contact_dist_ms", "spout_angle_deg",
             "spout_dist_px", "delta_angle_deg", "tip_to_spout_px", "reach_frac"]
    lick_cols += [c for c in extra if any(c in r for r in lick_rows)]          # OURS, only when computed
    bout_cols = ["trial_id", "position", "bout_idx", "multi_bout_rank", "n_peaks", "bout_t0_ms", "bout_t1_ms",
                 "peak_t0_ms", "peak_t1_ms", "peak_y0", "peak_angle0", "peak_y1", "peak_angle1"]
    return TongueKinematics(
        per_trial=pd.DataFrame(trial_rows),
        per_lick=pd.DataFrame(lick_rows, columns=lick_cols),
        bouts=pd.DataFrame(bout_rows, columns=bout_cols),
        licking_dynamics_traces=licking_dynamics_traces(by_pos, p=p),
        x_clean=x_c, y_clean=y_c, is_baseline_fill_clean=b_c, angle_per_frame=angle,
        trial_results=results, params=p,
        lick_phase=(lick_phase_table(results, X0, Y0, spout_frame, fps, n_points=int(p["phase"]["n_points"]),
                                     spout_angle_by_trial=None if sref is None else sref.spout_angle_deg.to_dict(),
                                     mode=p["phase"].get("mode", "extent"), cycle_ms=p["phase"].get("cycle_ms"))
                    if p.get("lick_geometry", False) else None),
        X0=X0, Y0=Y0, spout_frame=spout_frame, fps=fps,
        spout_angle_by_trial=None if sref is None else sref.spout_angle_deg.to_dict())


def from_clean(clean, trials, *, spout_frame: SpoutFrame | None = None, params_override: dict | None = None,
               spout_xy=None, contacts_ms: dict | None = None) -> TongueKinematics:
    """`compute_tongue_kinematics` on an `orofacial_clean.Clean` (tongue)."""
    return compute_tongue_kinematics(clean.x_final, clean.y_final, clean.fill_method, clean.lk, trials,
                                     fps=float(clean.fps), X0=float(clean.X0), Y0=float(clean.Y0),
                                     spout_frame=spout_frame, params_override=params_override,
                                     spout_xy=spout_xy, contacts_ms=contacts_ms)
