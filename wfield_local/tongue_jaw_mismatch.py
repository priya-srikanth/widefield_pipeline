"""Tongue-jaw MISMATCH trials: post-cue jaw movement WITHOUT a detected lick.

    from wfield_local import tongue_jaw_mismatch as tjm
    res = tjm.classify_from_clean(jaw, tongue, trials)        # jaw, tongue = orofacial_clean.Clean
    res.trials                                                # one row per cue
    tjm.summarize_by_position(res.trials)                     # candidates / trials per spout position

WHY IT MATTERS (their docstring, Priya 2026-04-30): after the stroke a missing lick is NOT a perceptual or
motivational failure if the jaw still moves to the cue. The dissociation -- jaw motor output spared, tongue
absent -- localises the deficit to tongue motor control. Mismatch trials are positive evidence for that
reading, not a QC gate.

THE RULE (their Cell 53, unchanged):

    candidate_no_lick_with_jaw_move = quiet_tongue_pass AND jaw_pass_qc

  quiet_tongue_pass  ZERO tongue peaks from the dy/dt slope detector in (70, 5000) ms after the cue
  jaw_pass_qc        `jaw_kinematics` sustained-density predicate: >= 5 of some 8 frames in (0, 5000) ms with
                     |jaw_y - baseline| > max(4 * baseline_sd, 4 px)

PORTED 2026-10-01, CLASSIFIER ONLY, from `../stroke_orofacial_pipeline/src/stroke_orofacial/dlc_kinematics/`:

  tongue_jaw_mismatch._build_classifier_result     -> `classify`        (per-trial loop, predicate, metadata)
  tongue_jaw_mismatch._build_tongue_detector_params -> `_detector_params` (mismatch-only invariants pinned)
  tongue_jaw_mismatch._snippet / _ms_to_frames     -> reused from `jaw_kinematics` (theirs were byte-identical
                                                      duplicates of the jaw_pertrial helpers)
  _detector._slope_detect_lick_peaks               -> `slope_detect_lick_peaks` (the paths the mismatch knobs reach)
  _detector._first_run_from / _moving_avg_nan      -> `_first_run_from` / `_moving_avg_nan`
  jaw side                                         -> `jaw_kinematics.jaw_pertrial` (their jaw_pertrial module)

WHAT TONGUE INFORMATION IT CONSUMES: the tongue x_final / y_final TRACES, nothing from a per-trial lick table.
Theirs deliberately re-ran the slope detector locally with its own knobs (no smoothing, no min-finite filter,
peaks accepted without a fall run) and did NOT read the tongue per-trial endpoints' n_licks, which use a
different window and stricter detector settings. So the detector is transcribed here rather than taken from
the tongue per-trial port, and the two lick counts are not expected to agree. x enters only as a finiteness
mask (a frame with no x cannot carry a rise); the decision is made on y.

WHAT CHANGED, ON PURPOSE (each also noted at its site):
  * Trials: `trial_id` + float `cue_frame` + `position` (one moving spout, 6 positions) instead of
    `TrialEvent(trial_index, side, tone_frame_idx)` with two fixed L/R spouts. The jaw lookup is keyed by
    trial_id alone (theirs: (trial_index, side), because trial_index restarted per side). Their Cell 55
    per-SIDE candidate counts become `summarize_by_position` (the numbers only; plotting not ported).
    No classification rule needs left/right.
  * Tongue input = `orofacial_clean` v5p3 tongue with baseline-filled frames set to NaN (`Clean.x_masked` /
    `y_masked`). Theirs (use_v7_cleaned=True) read the v7 PRE-CLEANED tongue with is_baseline_fill masked the
    same way; the v7 pre-clean is not ported yet (see orofacial_clean's docstring), so stray tracking that v7
    would have removed can still produce a peak here -- which can only turn a candidate INTO a non-candidate,
    never the reverse.
  * Jaw and tongue share one cue frame: both come from the same camera (cam4). Theirs looked up separate
    tongue/jaw reward frames from the per-bodypart npz files.
  * No I/O, no plots, no cohort figures, no rel_day: returns a DataFrame + metadata (pyarrow is not installed;
    stroke-relative day belongs to whoever aggregates sessions).
  * Config: `DEFAULTS` mirrors their `dlc_kinematics.tongue_jaw_mismatch.classifier` YAML; overridable from
    `configs/defaults.yaml` `orofacial_kinematics.mismatch` (deep-merged). Jaw knobs come from
    `jaw_kinematics.params()`.

INHERITED DIFFERENCES FROM THE LEGACY NOTEBOOK (their module's behaviour, kept): the detector has no
max_licks=30 cap (only matters for the reported count, never for zero-vs-nonzero); after a peak rejected by
min_peak_y_abs (or an all-NaN peak segment) the search resumes at turnover+1, where Cell 53 resumed after that
peak's fall run (or the peak segment). So a too-small rise followed by a big one BEFORE the small one's fall
run is a lick here and was skipped there -- this port can only find more licks than Cell 53, i.e. turn a
legacy candidate into a non-candidate, never the reverse.

GEOMETRY ASSUMPTION TO CHECK ON THIS RIG: a lick is a RISE in tongue y_final (dy/dt > +slope_thr), i.e.
protrusion moves the tongue toward larger image y, and min_peak_y_abs is measured from the retracted baseline.
That held on their Flea3 view; confirm the sign on cam4 before trusting a quiet_tongue_pass.

px-valued knobs (slope_thr_px_per_s, min_peak_y_abs, and the jaw floor) are the OLD rig's -- retune.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from wfield_local import config, jaw_kinematics
from wfield_local.jaw_kinematics import _ms_to_frames, _snippet, cue_frame_index

#: Their `TONGUE_JAW_MISMATCH_VERSION`.
TONGUE_JAW_MISMATCH_VERSION = "v7a.5"

#: Cohort direction mode; theirs dropped the per-mouse "above" override and recorded this for provenance.
JAW_DIRECTION_MODE = "above_or_below"

#: Mirrors their `dlc_kinematics.tongue_jaw_mismatch.classifier`. Overridden by `orofacial_kinematics.mismatch`.
DEFAULTS: dict = {
    "quiet_tongue_win_ms": [70.0, 5000.0],   # window in which ZERO detected licks = "quiet tongue"
    "tongue_detector": {                     # LOCAL to the mismatch classifier (Cell 53's knobs, = Cell 33's)
        "slope_thr_px_per_s": 250.0,         # px, OLD RIG -- retune (px/s)
        "rise_min_ms": 16.0,
        "fall_min_ms": 16.0,
        "min_peak_y_abs": 20.0,              # px, OLD RIG -- retune
        "peak_pad_ms": 52.0,
        "do_smooth_y_for_detection": False,
    },
}

#: Six spout positions in display order, for `summarize_by_position`.
POSITIONS = ("far_L", "close_L", "far_center", "close_center", "close_R", "far_R")


def params(overrides: dict | None = None) -> dict:
    """`DEFAULTS` <- `configs/defaults.yaml` `orofacial_kinematics.mismatch` <- ``overrides`` (deep-merged)."""
    p = config._deep_merge(DEFAULTS, config.defaults().get("orofacial_kinematics", {}).get("mismatch", {}) or {})
    return config._deep_merge(p, overrides) if overrides else p


@dataclass(frozen=True)
class MismatchResult:
    """Their `TongueJawMismatchResult`: per-trial table + the session's knobs (their parquet file metadata)."""
    trials: pd.DataFrame
    metadata: dict


# --------------------------------------------------------------------------- classifier

def classify_from_clean(jaw, tongue, trials: pd.DataFrame, *, overrides: dict | None = None,
                        jaw_overrides: dict | None = None, jaw_table: pd.DataFrame | None = None) -> MismatchResult:
    """`classify` on two `orofacial_clean.Clean` objects (jaw after `jaw_v34`).

    Jaw: `y_final` UNMASKED, as their jaw_pertrial read it. Tongue: `x_masked` / `y_masked` (baseline fill ->
    NaN), as their use_v7_cleaned loader did -- a baseline-filled 0 is "no data", not a tongue at rest.
    """
    if abs(float(jaw.fps) - float(tongue.fps)) > 1e-6:
        raise ValueError(f"tongue/jaw fps mismatch ({tongue.fps} vs {jaw.fps})")      # Cell 53's check
    return classify(jaw.y_final, tongue.x_masked, tongue.y_masked, float(tongue.fps), trials,
                    overrides=overrides, jaw_overrides=jaw_overrides, jaw_table=jaw_table)


def classify(jaw_y, tongue_x, tongue_y, fps: float, trials: pd.DataFrame, *, overrides: dict | None = None,
             jaw_overrides: dict | None = None, jaw_table: pd.DataFrame | None = None) -> MismatchResult:
    """Per-trial mismatch classification for one session (their `_build_classifier_result`).

    ``jaw_y``              per-frame jaw y_final (see `jaw_kinematics.jaw_pertrial`).
    ``tongue_x/tongue_y``  per-frame tongue x_final / y_final, baseline-filled frames already NaN.
    ``trials``             ``trial_id``, ``cue_frame`` (float camera frame), ``position``.
    ``jaw_table``          an existing `jaw_kinematics.jaw_pertrial` table for the SAME trials; computed here
                           when omitted (theirs built-or-reused the jaw JSON the same way).
    """
    cfg = params(overrides)
    quiet_win_ms = tuple(float(v) for v in cfg["quiet_tongue_win_ms"])
    detector_params = _detector_params(cfg["tongue_detector"])
    jp = jaw_kinematics.params(jaw_overrides)

    tongue_y = np.asarray(tongue_y, dtype=np.float64)
    tongue_x = np.asarray(tongue_x, dtype=np.float64)
    n_frames = len(tongue_y)
    fps = float(fps)

    if jaw_table is None:
        jaw_table, _ = jaw_kinematics.jaw_pertrial(jaw_y, fps, trials, jaw_overrides)
    # OURS: keyed by trial_id alone -- session-unique on this rig. Theirs keyed (trial_index, side) because
    # their trial_index restarted within each side.
    jaw_lookup = {tid: k for k, tid in enumerate(jaw_table["trial_id"].tolist())}

    # Tongue snippet spans the quiet window and the cue: [min(0, lo), max(0, hi)] ms.
    pre_f = abs(_ms_to_frames(min(0.0, quiet_win_ms[0]), fps))
    post_f = abs(_ms_to_frames(max(0.0, quiet_win_ms[1]), fps))
    t_ms_local = np.arange(-pre_f, post_f + 1) * (1000.0 / fps)

    rows = []
    for r in jaw_kinematics._iter_trials(trials):
        c = cue_frame_index(r.cue_frame)
        if c is None:
            # OURS: no cue frame -> no tongue data -> zero peaks. The jaw side is then degenerate (QC fail), so
            # the trial can never be a candidate; it is kept as a row so the table stays one-per-trial.
            y_w = x_w = np.full(pre_f + post_f + 1, np.nan)
        else:
            y_w = _snippet(tongue_y, c, pre_f, post_f, n_frames)
            x_w = _snippet(tongue_x, c, pre_f, post_f, n_frames)

        peaks = slope_detect_lick_peaks(y_w, x_w, t_ms_local, fps, detect_win_ms=quiet_win_ms,
                                        detector_params=detector_params)
        n_licks = int(len(peaks.peak_times_ms))
        quiet_tongue_pass = bool(n_licks == 0)          # their max_licks_for_quiet is fixed at 0

        k = jaw_lookup.get(r.trial_id)
        if k is None:
            raise KeyError(f"jaw table has no trial_id {r.trial_id!r} -- trials drifted between the jaw table "
                           f"and this call; rebuild it with the same trials")
        j = jaw_table.iloc[k]
        jaw_pass_qc = bool(j["jaw_pass_qc"])

        rows.append({
            "trial_id": r.trial_id,
            "position": r.position,
            "cue_frame": float(r.cue_frame),
            "cue_frame_idx": -1 if c is None else c,
            "fps": fps,
            "n_licks_in_quiet_win": n_licks,
            "tongue_peak_times_ms": [float(t) for t in peaks.peak_times_ms],
            "quiet_tongue_pass": quiet_tongue_pass,
            **{col: j[col] for col in jaw_kinematics.JAW_COLUMNS if col != "jaw_pass_qc"},
            "jaw_pass_qc": jaw_pass_qc,
            "candidate_no_lick_with_jaw_move": bool(quiet_tongue_pass and jaw_pass_qc),
        })

    cols = ["trial_id", "position", "cue_frame", "cue_frame_idx", "fps", "n_licks_in_quiet_win",
            "tongue_peak_times_ms", "quiet_tongue_pass", *jaw_kinematics.JAW_COLUMNS,
            "candidate_no_lick_with_jaw_move"]
    df = pd.DataFrame(rows, columns=cols)
    td = cfg["tongue_detector"]
    metadata = {
        "schema_version": TONGUE_JAW_MISMATCH_VERSION,
        "jaw_pertrial_version": jaw_kinematics.JAW_PERTRIAL_VERSION,
        "fps": fps,
        "n_frames": int(n_frames),
        "n_trials": int(len(df)),
        "quiet_tongue_win_ms_lo": quiet_win_ms[0],
        "quiet_tongue_win_ms_hi": quiet_win_ms[1],
        "max_licks_for_quiet": 0,
        "jaw_baseline_win_ms_lo": float(jp["baseline_win_ms"][0]),
        "jaw_baseline_win_ms_hi": float(jp["baseline_win_ms"][1]),
        "jaw_detect_win_ms_lo": float(jp["detect_win_ms"][0]),
        "jaw_detect_win_ms_hi": float(jp["detect_win_ms"][1]),
        "jaw_thresh_sd_multiplier": float(jp["deflection_threshold_sd_multiplier"]),
        "jaw_thresh_min_px": float(jp["deflection_threshold_min_px"]),
        "jaw_sustained_density_threshold": float(jp["sustained_density_threshold"]),
        "jaw_sustained_window_n_frames": int(jp["sustained_window_n_frames"]),
        "jaw_direction_mode": JAW_DIRECTION_MODE,
        "tongue_detector_slope_thr_px_per_s": float(td["slope_thr_px_per_s"]),
        "tongue_detector_rise_min_ms": float(td["rise_min_ms"]),
        "tongue_detector_fall_min_ms": float(td["fall_min_ms"]),
        "tongue_detector_min_peak_y_abs": float(td["min_peak_y_abs"]),
        "tongue_detector_peak_pad_ms": float(td["peak_pad_ms"]),
        "tongue_detector_do_smooth_y_for_detection": bool(td["do_smooth_y_for_detection"]),
    }
    return MismatchResult(trials=df, metadata=metadata)


def summarize_by_position(table: pd.DataFrame) -> pd.DataFrame:
    """Per spout position: n_trials, n_quiet_tongue, n_jaw_moved, n_candidates, frac_candidates.

    OURS, replacing their Cell 55 per-SIDE count text ("L candidate trials: n/N") -- the same numbers, grouped
    by the six positions of the one moving spout. Positions outside `POSITIONS` (an unresolved label) are
    appended after them rather than dropped, so the counts always add up to the table.
    """
    order = [p for p in POSITIONS if p in set(table["position"])]
    order += sorted({p for p in table["position"] if p not in POSITIONS}, key=str)
    out = []
    for pos in order:
        g = table[table["position"] == pos]
        n = len(g)
        nc = int(g["candidate_no_lick_with_jaw_move"].sum())
        out.append({"position": pos, "n_trials": n, "n_quiet_tongue": int(g["quiet_tongue_pass"].sum()),
                    "n_jaw_moved": int(g["jaw_pass_qc"].sum()), "n_candidates": nc,
                    "frac_candidates": nc / n if n else np.nan})
    return pd.DataFrame(out, columns=["position", "n_trials", "n_quiet_tongue", "n_jaw_moved", "n_candidates",
                                      "frac_candidates"])


def _detector_params(tongue_detector_cfg: dict) -> dict:
    """Their `_build_tongue_detector_params`: the YAML knobs plus the invariants Cell 53 had baked in.

    min_finite_samples_per_lick = 0       Cell 53 had no min-finite filter.
    smooth_ms_detect_y = 0.0              so do_smooth_y_for_detection=True is a no-op (1-frame window) --
                                          inherited as-is; Cell 53's SMOOTH_MS=12 never reached their module.
    strict_no_fallback = True             no turnover before the window ends -> stop searching.
    accept_peaks_without_fall_run = True  Cell 53 looked for the fall only to advance the search, never to
                                          reject a peak (Cells 36/39/42 behaviour, not Cell 33's strict one).
    """
    return {**tongue_detector_cfg, "min_finite_samples_per_lick": 0, "smooth_ms_detect_y": 0.0,
            "strict_no_fallback": True, "accept_peaks_without_fall_run": True}


# --------------------------------------------------------------------------- slope detector (transcribed)

@dataclass
class LickPeaks:
    """Their `_LickPeaks`, minus `fall_ends`. Indices are into the DETECT-WINDOW slice (add its first index to
    get snippet frames); `peak_times_ms` are already cue-relative."""
    peak_times_ms: np.ndarray
    peak_y: np.ndarray
    peak_x: np.ndarray
    rise_starts: np.ndarray
    turnover: np.ndarray
    peak_indices: np.ndarray

    @classmethod
    def empty(cls) -> LickPeaks:
        f, i = np.zeros(0, np.float64), np.zeros(0, np.int64)
        return cls(f, f.copy(), f.copy(), i, i.copy(), i.copy())


def _moving_avg_nan(arr: np.ndarray, window_ms: float, fps: float) -> np.ndarray:
    """NaN-tolerant centred moving average (their `_detector._moving_avg_nan`)."""
    win_n = max(1, int(round(window_ms / 1000.0 * fps)))
    if win_n <= 1:
        return arr.astype(np.float64, copy=True)
    finite = np.isfinite(arr).astype(np.float64)
    safe = np.where(np.isfinite(arr), arr, 0.0)
    kernel = np.ones(win_n, dtype=np.float64)
    num = np.convolve(safe, kernel, mode="same")
    den = np.convolve(finite, kernel, mode="same")
    out = np.full_like(arr, np.nan, dtype=np.float64)
    nz = den > 0
    out[nz] = num[nz] / den[nz]
    return out


def _first_run_from(mask: np.ndarray, min_len: int, start: int) -> tuple[int | None, int | None]:
    """First True run of length >= ``min_len`` at index >= ``start``: (start, end INCLUSIVE) or (None, None).
    (Cell 53's version returned an exclusive end; their module's inclusive end shifts `end_idx` one frame
    earlier, onto the last FALL frame, which can never start a rise run -- so no detection changes.)"""
    n = len(mask)
    i = start
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if j - i >= min_len:
                return i, j - 1
            i = j
        else:
            i += 1
    return None, None


def slope_detect_lick_peaks(y_t: np.ndarray, x_t: np.ndarray, t_ms: np.ndarray, fps: float, *,
                            detect_win_ms: tuple[float, float], detector_params: dict) -> LickPeaks:
    """Rise/turnover/peak lick detector on one trial snippet (their `_detector._slope_detect_lick_peaks`).

    Per lick: a rise (dy/dt > slope_thr for >= rise_min_ms) -> turnover (first frame after it where dy/dt is
    finite and <= 0) -> peak = argmax y in [rise start, turnover + peak_pad] -> accept if |y_peak| >=
    min_peak_y_abs and enough finite frames around it -> optional fall run (dy/dt < -slope_thr for >=
    fall_min_ms) that only sets where the next search starts unless strict mode rejects fall-less peaks.

    NOT TRANSCRIBED: their Phase 7e bounded-fall mode (`max_fall_search_ms`, `accept_implicit_baseline_fall`,
    `is_baseline_fill`), `n_max`, and `fall_ends` -- the mismatch classifier never sets them (the knobs default
    off), so those branches were unreachable from it.
    """
    slope_thr = float(detector_params["slope_thr_px_per_s"])
    rise_min_ms = float(detector_params["rise_min_ms"])
    fall_min_ms = float(detector_params["fall_min_ms"])
    peak_pad_ms = float(detector_params["peak_pad_ms"])
    min_peak_y_abs = float(detector_params["min_peak_y_abs"])
    min_finite = int(detector_params["min_finite_samples_per_lick"])
    do_smooth_y = bool(detector_params["do_smooth_y_for_detection"])
    smooth_ms_y = float(detector_params["smooth_ms_detect_y"])
    strict_no_fallback = bool(detector_params.get("strict_no_fallback", True))
    accept_no_fall = bool(detector_params.get("accept_peaks_without_fall_run", False))

    if y_t.size == 0:
        return LickPeaks.empty()
    in_win = (t_ms >= detect_win_ms[0]) & (t_ms <= detect_win_ms[1])
    if not np.any(in_win):
        return LickPeaks.empty()
    i_lo = int(np.argmax(in_win))
    i_hi = int(len(in_win) - 1 - np.argmax(in_win[::-1]))
    yseg = y_t[i_lo:i_hi + 1].astype(np.float64, copy=True)
    xseg = x_t[i_lo:i_hi + 1].astype(np.float64, copy=True)
    tseg = t_ms[i_lo:i_hi + 1]

    y_for_slope = _moving_avg_nan(yseg, smooth_ms_y, fps) if do_smooth_y else yseg.copy()
    finite = np.isfinite(y_for_slope)
    dy = np.full_like(y_for_slope, np.nan, dtype=np.float64)
    if np.count_nonzero(finite) >= 2:
        dy = np.gradient(y_for_slope) * fps                     # px/s; NaN propagates to both neighbours
    # A frame counts only where dy, y AND x are finite: x is a tracking-validity mask, not a feature.
    finite_dy = np.isfinite(dy) & np.isfinite(yseg) & np.isfinite(xseg)

    rise_mask = finite_dy & (dy > slope_thr)
    fall_mask = finite_dy & (dy < -slope_thr)
    rise_min_n = max(1, int(round(rise_min_ms / 1000.0 * fps)))
    fall_min_n = max(1, int(round(fall_min_ms / 1000.0 * fps)))
    peak_pad_n = max(1, int(round(peak_pad_ms / 1000.0 * fps)))

    L = len(yseg)
    rise_starts, turnovers, peaks, times, ys, xs = [], [], [], [], [], []
    start = 0
    while True:
        r0, r1 = _first_run_from(rise_mask, rise_min_n, start)
        if r0 is None:
            break
        # Turnover: walk past dy > 0 AND past non-finite dy -- a tracking dropout mid-rise is not a turnover
        # (their port fix #1; treating NaN as turnover produced an extra early peak).
        k = r1
        while k < L and (not np.isfinite(dy[k]) or dy[k] > 0):
            k += 1
        if k >= L:
            if strict_no_fallback:          # Cell 53: no turnover inside the window -> stop
                break
            start = r1 + 1
            continue

        k_end = min(L - 1, k + peak_pad_n)
        seg = yseg[r0:k_end + 1]
        if seg.size == 0 or not np.any(np.isfinite(seg)):
            start = k + 1
            continue
        i_peak = r0 + int(np.nanargmax(seg))

        y_peak_val = yseg[i_peak]
        if not np.isfinite(y_peak_val) or abs(y_peak_val) < min_peak_y_abs:
            start = k + 1                   # see "INHERITED DIFFERENCES" in the module docstring
            continue

        win_lo = max(0, i_peak - peak_pad_n)
        win_hi = min(L - 1, i_peak + peak_pad_n)
        if int(np.count_nonzero(np.isfinite(yseg[win_lo:win_hi + 1]))) < min_finite:
            start = k + 1
            continue

        # The fall run only moves the next search past this lick's retraction (their fix #2/#3): restarting
        # at i_peak + 1 could re-detect a shoulder of the same lick.
        end_idx = i_peak
        if fall_min_n > 1:
            f0, f1 = _first_run_from(fall_mask, fall_min_n, i_peak)
            if f0 is None:
                if strict_no_fallback and not accept_no_fall:
                    start = i_peak + 1
                    continue
            else:
                end_idx = max(end_idx, f1 - 1)

        rise_starts.append(r0)
        turnovers.append(k)
        peaks.append(i_peak)
        ys.append(float(y_peak_val))
        xs.append(float(xseg[i_peak]))
        times.append(float(tseg[i_peak]))
        start = end_idx + 1

    return LickPeaks(peak_times_ms=np.asarray(times, np.float64), peak_y=np.asarray(ys, np.float64),
                     peak_x=np.asarray(xs, np.float64), rise_starts=np.asarray(rise_starts, np.int64),
                     turnover=np.asarray(turnovers, np.int64), peak_indices=np.asarray(peaks, np.int64))
