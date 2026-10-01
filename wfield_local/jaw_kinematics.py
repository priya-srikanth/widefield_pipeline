"""Per-trial jaw features around each cue: baseline, peak deflections, threshold, sustained-movement QC.

    from wfield_local import jaw_kinematics as jk
    table, meta = jk.jaw_pertrial(jaw.y_final, jaw.fps, trials)     # jaw = orofacial_clean.Clean (v3.4)

PORTED 2026-10-01 from `../stroke_orofacial_pipeline/src/stroke_orofacial/dlc_kinematics/jaw_pertrial.py`
(Phase 7a.5, schema "v7a.5"), which itself extracted the inline jaw logic of the legacy tongue notebook's
Cell 53 mismatch classifier (vendored at `docs/legacy_snippets/tongue_jaw_mismatch.py`). Transcribed function
for function, same numerics:

  _compute_per_trial          -> `_compute_per_trial`   (the 10 scalars for one trial, degenerate-baseline branch)
  _ms_to_frames               -> `_ms_to_frames`
  _snippet                    -> `_snippet`             (NaN-padded cue window)
  _window_mask                -> `_window_mask`         (inclusive ms window)
  _compute_jaw_baseline       -> `_compute_jaw_baseline` (mean + POPULATION sd, >= 2 finite frames)
  _compute_jaw_deflections    -> `_compute_jaw_deflections` ("above_or_below": signed value at argmax |defl|)
  _compute_jaw_threshold      -> `_compute_jaw_threshold` (max(sd_mult * sd, min_px) + which term won)
  _check_sustained_density    -> `_check_sustained_density` (>= ceil(0.625 * 8) = 5 supra frames in any 8)
  _build_jaw_pertrial_core    -> `jaw_pertrial`         (loop over trials + metadata), minus the parquet load

The 10 per-trial scalars (`JAW_COLUMNS`), as in theirs:
  jaw_baseline_mean / jaw_baseline_sd   mean and np.nanstd of y_final in baseline_win_ms (-500, 0) ms
  jaw_peak_pos / neg / abs_deflection   max, min, max|.| of (y_final - baseline_mean) in detect_win_ms (0, 5000)
  jaw_peak_selected_deflection          signed (y - baseline) at argmax |.|  (cohort direction mode)
  jaw_thresh_abs                        the APPLIED threshold, max(sd_mult * sd, min_px)
  jaw_thresh_px_used                    "sd_scaled" | "px_floor" | "none" (degenerate baseline)
  jaw_n_frames_over_thresh              frames in detect_win_ms with |defl| > jaw_thresh_abs
  jaw_pass_qc                           sustained-density predicate -- the "jaw moved" call the mismatch
                                        classifier (`tongue_jaw_mismatch`) consumes

INPUT: the jaw y_final of `orofacial_clean` after `jaw_v34` -- the equivalent of their `use_cleaned_traces: true`
(Phase 7f v3.4 cleaned_traces), which is what every one of their jaw analyses read. Only y is used: jaw
deflection is a vertical quantity, theirs never read x_final either.

WHAT CHANGED, ON PURPOSE (each also noted at its site):
  * Trials. Theirs: `TrialEvent(trial_index, side, tone_frame_idx:int)`, two fixed L/R spouts, trial_index
    counted WITHIN a side (so (trial_index, side) was the key). Ours: one moving spout, a `trials` table with
    `trial_id` (session-unique, so it alone is the key), `cue_frame` (FLOAT camera frame from the alignment
    template, rounded here to the nearest frame) and `position` (far_L ... far_R). No rule in this module
    needs left/right; `position` is carried through for per-position grouping downstream.
  * No I/O. Theirs read a parquet (pyarrow) and wrote an atomic JSON per session; this returns a DataFrame +
    metadata dict and leaves writing to the caller (pyarrow is not installed here).
  * Non-finite scalars are NaN, not None. Theirs emitted None because JSON has no NaN; a DataFrame does.
  * A NaN cue frame (cue outside the camera template, say) yields the degenerate-baseline row. Theirs could
    not get one -- tone_frame_idx was always an int.
  * Config: numeric constants in `DEFAULTS` below (their `dlc_kinematics.jaw_pertrial` YAML block), overridable
    from `configs/defaults.yaml` `orofacial_kinematics.jaw` (deep-merged) or a `params=` dict.

KNOWN DIVERGENCE FROM THE LEGACY NOTEBOOK, INHERITED (ported as their module has it, not "fixed"): Cell 53
counted a frame suprathreshold when |defl| > sd_mult * sd AND |defl| >= min_px; their module (and so this one)
uses |defl| > max(sd_mult * sd, min_px). The two differ only at |defl| == min_px exactly, when the floor wins.

Every px value here is the OLD rig's (Flea3 view) and must be re-measured on our cam4 view; ms values transfer
because both rigs record at 250 fps. `sustained_window_n_frames` is in FRAMES (8 = 32 ms at 250 fps), and so
transfers only while fps stays 250.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from wfield_local import config

#: Their `jaw_pertrial.JAW_PERTRIAL_VERSION`; kept so a table can be traced to the algorithm it came from.
JAW_PERTRIAL_VERSION = "v7a.5"

#: Mirrors their `configs/defaults.yaml` `dlc_kinematics.jaw_pertrial`. Overridden by
#: `orofacial_kinematics.jaw` in OUR defaults.yaml (see `params`).
DEFAULTS: dict = {
    "detect_win_ms": [0.0, 5000.0],             # post-cue window for the jaw deflection
    "baseline_win_ms": [-500.0, 0.0],           # pre-cue baseline window for mean + sd
    "deflection_threshold_sd_multiplier": 4.0,  # |y - baseline| > 4 * baseline_sd ...
    "deflection_threshold_min_px": 4.0,         # px, OLD RIG -- retune.  ... or > 4 px, whichever is larger
    "sustained_density_threshold": 0.625,       # >= 5/8 frames over threshold ...
    "sustained_window_n_frames": 8,             # ... in some rolling 8-FRAME window (32 ms at 250 fps)
}

#: The 10 per-trial scalars, in their emit order.
JAW_COLUMNS = (
    "jaw_baseline_mean", "jaw_baseline_sd",
    "jaw_peak_pos_deflection", "jaw_peak_neg_deflection", "jaw_peak_abs_deflection",
    "jaw_peak_selected_deflection",
    "jaw_thresh_abs", "jaw_thresh_px_used", "jaw_n_frames_over_thresh", "jaw_pass_qc",
)


def params(overrides: dict | None = None) -> dict:
    """`DEFAULTS` <- `configs/defaults.yaml` `orofacial_kinematics.jaw` <- ``overrides`` (each deep-merged)."""
    p = config._deep_merge(DEFAULTS, config.defaults().get("orofacial_kinematics", {}).get("jaw", {}) or {})
    return config._deep_merge(p, overrides) if overrides else p


def cue_frame_index(cue_frame) -> int | None:
    """Nearest camera frame to a (float) cue frame, or None when it is not finite.

    Theirs took an int `tone_frame_idx` straight from the wavesurfer<->video alignment. Ours comes from the
    affine alignment template (`orofacial_clean.cue_frames`) and is fractional; np.round (half-to-even) is the
    same rounding `orofacial_clean.pool_mask` / `stack` use, so a trial lands on the same frame everywhere.
    """
    c = float(cue_frame)
    return int(np.round(c)) if math.isfinite(c) else None


# --------------------------------------------------------------------------- public API

def jaw_pertrial(jaw_y, fps: float, trials: pd.DataFrame,
                 overrides: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Per-trial jaw scalars for one session -- their `_build_jaw_pertrial_core`, minus the parquet load.

    ``jaw_y``  per-frame jaw y_final (baseline-subtracted px) on the camera clock -- `Clean.y_final` after
               `jaw_v34`. Passed UNMASKED, as theirs was: frames v3.4 could not refill sit at the baseline (0)
               and count as data in both the baseline and the deflection. Pass `Clean.y_masked` to change that.
    ``trials`` one row per cue: ``trial_id``, ``cue_frame`` (float camera frame), ``position``.
    ``overrides`` deep-merged on top of `params()` (their per-call ``params`` dict).

    Returns ``(table, metadata)``: one row per trial in input order with ``trial_id``, ``position``,
    ``cue_frame``, ``cue_frame_idx`` (-1 when the cue frame is NaN) and `JAW_COLUMNS`; metadata = the knobs used (their JSON ``metadata``).
    """
    p = _resolve(overrides)
    jaw_y = np.asarray(jaw_y, dtype=np.float64)
    n_frames = len(jaw_y)
    rows = [_compute_per_trial(r.trial_id, r.position, r.cue_frame, jaw_y, float(fps), n_frames, **p)
            for r in _iter_trials(trials)]
    cols = ["trial_id", "position", "cue_frame", "cue_frame_idx", *JAW_COLUMNS]
    table = pd.DataFrame(rows, columns=cols)
    metadata = {
        "fps": float(fps),
        "n_frames": int(n_frames),
        "feature_extraction_version": JAW_PERTRIAL_VERSION,
        "detect_win_ms": list(p["detect_win_ms"]),
        "baseline_win_ms": list(p["baseline_win_ms"]),
        "deflection_threshold_sd_multiplier": p["sd_mult"],
        "deflection_threshold_min_px": p["min_px"],
        "sustained_density_threshold": p["density_thr"],
        "sustained_window_n_frames": p["window_n"],
    }
    return table, metadata


def _resolve(overrides: dict | None) -> dict:
    """`params()` flattened to `_compute_per_trial`'s keyword names (theirs did this in the core function)."""
    q = params(overrides)
    return dict(
        detect_win_ms=tuple(float(v) for v in q["detect_win_ms"]),
        baseline_win_ms=tuple(float(v) for v in q["baseline_win_ms"]),
        sd_mult=float(q["deflection_threshold_sd_multiplier"]),
        min_px=float(q["deflection_threshold_min_px"]),
        density_thr=float(q["sustained_density_threshold"]),
        window_n=int(q["sustained_window_n_frames"]),
    )


def _iter_trials(trials: pd.DataFrame):
    missing = {"trial_id", "cue_frame", "position"} - set(trials.columns)
    if missing:
        raise KeyError(f"trials table needs columns trial_id, cue_frame, position; missing {sorted(missing)}")
    return trials[["trial_id", "cue_frame", "position"]].itertuples(index=False)


# --------------------------------------------------------------------------- per-trial compute (transcribed)

def _compute_per_trial(trial_id, position, cue_frame, jaw_y: np.ndarray, fps: float, n_frames: int, *,
                       detect_win_ms: tuple[float, float], baseline_win_ms: tuple[float, float],
                       sd_mult: float, min_px: float, density_thr: float, window_n: int) -> dict:
    """The 10 scalars for one trial (their `_compute_per_trial`; ``ev`` replaced by id / position / cue frame)."""
    # Snippet = union of the baseline and detect windows. Window masks are evaluated on t_ms, so the snippet's
    # extent changes nothing as long as it covers both (legacy Cell 53 used a fixed (-1500, 5500) ms).
    snippet_lo_ms = min(baseline_win_ms[0], detect_win_ms[0])
    snippet_hi_ms = max(baseline_win_ms[1], detect_win_ms[1])
    pre_f = abs(_ms_to_frames(snippet_lo_ms, fps))
    post_f = abs(_ms_to_frames(snippet_hi_ms, fps))
    t_ms = np.arange(-pre_f, post_f + 1) * (1000.0 / fps)

    # OURS: the cue frame is a float from the alignment template; NaN (no template coverage) -> an all-NaN
    # snippet, which falls into the degenerate-baseline branch below exactly like a trial off the recording.
    c = cue_frame_index(cue_frame)
    if c is None:
        jaw_yw = np.full(pre_f + post_f + 1, np.nan)
    else:
        jaw_yw = _snippet(jaw_y, c, pre_f, post_f, n_frames)

    base_mean, base_sd = _compute_jaw_baseline(jaw_yw, t_ms, baseline_win_ms=baseline_win_ms)
    thresh_abs, thresh_label = _compute_jaw_threshold(base_sd, sd_mult=sd_mult, min_px=min_px)

    head = dict(trial_id=trial_id, position=position, cue_frame=float(cue_frame),
                cue_frame_idx=(-1 if c is None else c))
    if not np.isfinite(base_mean) or not np.isfinite(base_sd) or base_sd <= 0:
        # Legacy semantics: a degenerate baseline (< 2 finite frames, or a perfectly flat one) gives NaN
        # peaks, NaN threshold, count 0 and QC FAIL -- Cell 53's `if jaw_baseline_sd > 0: ... else:
        # jaw_pass_qc = False`. A flat baseline is a real possibility on a v3.4 jaw (baseline fill = exact 0),
        # and it means "cannot judge", not "moved". Theirs emitted None here (JSON); we emit NaN.
        return {**head,
                "jaw_baseline_mean": float(base_mean), "jaw_baseline_sd": float(base_sd),
                "jaw_peak_pos_deflection": np.nan, "jaw_peak_neg_deflection": np.nan,
                "jaw_peak_abs_deflection": np.nan, "jaw_peak_selected_deflection": np.nan,
                "jaw_thresh_abs": np.nan, "jaw_thresh_px_used": "none",
                "jaw_n_frames_over_thresh": 0, "jaw_pass_qc": False}

    peaks = _compute_jaw_deflections(jaw_yw, t_ms, base_mean, detect_win_ms=detect_win_ms)
    pass_qc, n_over = _check_sustained_density(jaw_yw, t_ms, base_mean, thresh_abs, detect_win_ms=detect_win_ms,
                                               window_n=window_n, density_thr=density_thr)
    return {**head,
            "jaw_baseline_mean": float(base_mean), "jaw_baseline_sd": float(base_sd),
            "jaw_peak_pos_deflection": peaks["pos"], "jaw_peak_neg_deflection": peaks["neg"],
            "jaw_peak_abs_deflection": peaks["abs"], "jaw_peak_selected_deflection": peaks["selected"],
            "jaw_thresh_abs": float(thresh_abs), "jaw_thresh_px_used": str(thresh_label),
            "jaw_n_frames_over_thresh": int(n_over), "jaw_pass_qc": bool(pass_qc)}


# --------------------------------------------------------------------------- numeric helpers (transcribed)

def _ms_to_frames(ms: float, fps: float) -> int:
    return int(np.round((ms / 1000.0) * fps))


def _snippet(arr: np.ndarray, center_fr: int, pre_f: int, post_f: int, n_frames: int) -> np.ndarray:
    """``arr[center-pre_f : center+post_f+1]``, NaN-padded where it runs off the recording (Cell 53 `_snippet`)."""
    T = pre_f + post_f + 1
    lo = center_fr - pre_f
    hi = center_fr + post_f
    out = np.full(T, np.nan, dtype=float)
    src_lo = max(lo, 0)
    src_hi = min(hi, n_frames - 1)
    if src_hi < src_lo:        # their port's guard; Cell 53 skipped such trials before getting here
        return out
    dst_lo = src_lo - lo
    dst_hi = dst_lo + (src_hi - src_lo)
    out[dst_lo:dst_hi + 1] = arr[src_lo:src_hi + 1]
    return out


def _window_mask(t_ms: np.ndarray, win_ms: tuple[float, float]) -> np.ndarray:
    """Inclusive at BOTH ends -- so frame 0 (the cue frame) is in the baseline AND the detect window, as in theirs."""
    lo, hi = float(win_ms[0]), float(win_ms[1])
    t = np.asarray(t_ms, float)
    return np.isfinite(t) & (t >= lo) & (t <= hi)


def _compute_jaw_baseline(jaw_y: np.ndarray, t_ms: np.ndarray, *,
                          baseline_win_ms: tuple[float, float]) -> tuple[float, float]:
    """Mean and POPULATION sd (np.nanstd, ddof=0) of the finite baseline frames; (NaN, NaN) if fewer than 2."""
    base_mask = _window_mask(t_ms, baseline_win_ms)
    base_vals = np.asarray(jaw_y, float)[base_mask]
    base_vals = base_vals[np.isfinite(base_vals)]
    if base_vals.size < 2:
        return float("nan"), float("nan")
    return float(np.nanmean(base_vals)), float(np.nanstd(base_vals))


def _compute_jaw_deflections(jaw_y: np.ndarray, t_ms: np.ndarray, baseline_mean: float, *,
                             detect_win_ms: tuple[float, float]) -> dict:
    """Peak deflections in the detect window. ``selected`` = the signed value at argmax |defl| -- the cohort
    "above_or_below" direction mode (theirs dropped the per-mouse "above" mode; so does this)."""
    test_mask = _window_mask(t_ms, detect_win_ms)
    test_vals = np.asarray(jaw_y, float)[test_mask] - baseline_mean
    test_vals = test_vals[np.isfinite(test_vals)]
    if test_vals.size == 0:
        return dict(pos=float("nan"), neg=float("nan"), abs=float("nan"), selected=float("nan"))
    pos = float(np.nanmax(test_vals))
    neg = float(np.nanmin(test_vals))
    abs_ = float(np.nanmax(np.abs(test_vals)))
    idx_sel = int(np.nanargmax(np.abs(test_vals)))
    return dict(pos=pos, neg=neg, abs=abs_, selected=float(test_vals[idx_sel]))


def _compute_jaw_threshold(baseline_sd: float, *, sd_mult: float, min_px: float) -> tuple[float, str]:
    """``max(sd_mult * sd, min_px)`` and which term won. A tie goes to "px_floor" (theirs: strict ``>``)."""
    sd_term = sd_mult * baseline_sd
    if not np.isfinite(sd_term):
        return float(min_px), "px_floor"
    if sd_term > min_px:
        return float(sd_term), "sd_scaled"
    return float(min_px), "px_floor"


def _check_sustained_density(jaw_y: np.ndarray, t_ms: np.ndarray, baseline_mean: float, threshold: float, *,
                             detect_win_ms: tuple[float, float], window_n: int,
                             density_thr: float) -> tuple[bool, int]:
    """(pass_qc, n_frames_over_thresh). pass_qc: some ``window_n``-frame run of the detect window holds at least
    ceil(density_thr * window_n) suprathreshold frames -- Cell 53's `_count_true_runs_in_window`. A density,
    not a contiguous run: it tolerates a dropped frame or two inside a real movement, and the window is short
    enough (32 ms) that scattered single-frame noise rarely fills 5 of 8."""
    test_mask = _window_mask(t_ms, detect_win_ms)
    defl = np.asarray(jaw_y, float) - baseline_mean
    supra_full = np.isfinite(defl) & test_mask & (np.abs(defl) > threshold)
    n_over = int(np.sum(supra_full))

    supra_test = supra_full[test_mask].astype(int)
    W = max(1, int(window_n))
    K = max(1, int(np.ceil(density_thr * W)))
    if supra_test.size == 0:
        return False, n_over
    if supra_test.size < W:      # shorter than one window: judge the whole thing (Cell 53's fallback)
        return bool(supra_test.sum() >= K), n_over
    counts = np.convolve(supra_test, np.ones(W, dtype=int), mode="valid")
    return bool(np.any(counts >= K)), n_over
