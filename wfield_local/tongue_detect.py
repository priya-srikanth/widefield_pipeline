"""Tongue lick detection on cleaned traces: v7 pre-clean, three lick detectors + merge, gates F/D/E/M.

Input is the v5p3-cleaned tongue from `wfield_local.orofacial_clean` (`Clean.x_final / y_final / fill_method`,
`Clean.lk`). Per-trial orchestration and the per-trial / per-lick features built on this live in
`wfield_local.tongue_kinematics`.

PORTED 2026-10-01 from `../stroke_orofacial_pipeline/src/stroke_orofacial/dlc_kinematics/` (read-only), the
path their production default `dlc_kinematics.tongue_pertrial.use_v7_pipeline: true` takes. Transcribed
function for function, same numerics, same order of operations:

  _preclean.py    `preclean_trace`, `_classify_raw_clusters`, `_find_y_frame_outliers`,
                  `_extend_pchip_around_range`, `_wipe_bracketed_interp`, `_derive_x0_baseline_fill`,
                  `_find_x_isolated_jumps`, `_clean_x_trace`, `_RawCluster`, `_PrecleanDiagnostics`
  _detector.py    `_moving_avg_nan`, `_first_run_from`, `_slope_detect_lick_peaks` (both the "bounded" and the
                  "legacy" configurations the v7 orchestrator calls), `_LickPeaks`
  _detector_v2.py `detect_local_maxima`, `merge_detector_outputs`, `_PeakList`, `_DetectedPeak`
  tongue_pertrial.py `_legacy_peaks_to_peak_list` (the adapter between the two peak shapes)
  _gates.py       `gate_F`, `gate_D`, `gate_E`, `_compute_imputed_x`, `parabolic_excise`, `spline_excise`,
                  `_peak_confidence`, `run_m_gate`

NOT PORTED (dead on the v7 path, or not part of detection):
  * `_detector._filter_peaks_by_bout_mode` -- only the legacy (use_v7_pipeline=False) extractors call it.
  * `_detector._build_bouts` -- a FEATURE, not detection; it lives in `tongue_kinematics.build_bouts`.
  * `_gates.LickEventRecord` -- a parquet row schema; the per-lick DataFrame in `tongue_kinematics` replaces it.

WHAT CHANGED, ON PURPOSE (each also noted at its site):
  * Constants come from the module-level `DEFAULTS` (mirroring their YAML blocks `preclean`, `detector`,
    `detector_v2`, `gates`), deep-merged with `configs/defaults.yaml: orofacial_kinematics.tongue` if present.
    Every px-valued one is marked "px, OLD RIG -- retune": their camera view is not ours. ms values transfer
    because both rigs run at 250 fps; the few FRAME-COUNT constants (isolated_max_n, flat_min_len,
    x_flank_max_distance, min-anchor counts) transfer for the same reason and only for it.
  * `gate_E`'s robust-std floor (0.5 px, inline in theirs) is now `gates.e_min_robust_std_px`, because it is a
    px threshold like the others and has to be retunable with them.
  * Fill flags: theirs read `is_interp_fill` / `is_baseline_fill` columns; ours derive them from
    `orofacial_clean.fill_method` (`interp_masks`), which is the same information (v5p3 stage 4 / stage 5).
  * Sign convention: everything here assumes PROTRUSION = +y_final, as theirs did. That holds for
    `orofacial_clean` output, whose baseline is the lowest image-y (tongue in) and image y grows downward,
    so a tongue reaching down toward the spouts is positive. A view where the tongue protrudes UPWARD in the
    image would need y negated before this module.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.signal import find_peaks

# --------------------------------------------------------------------------- params

#: Their YAML `dlc_kinematics.tongue_pertrial.{preclean, detector, detector_v2, gates}` (values as of the port).
#: px constants are the OLD rig's (a different camera, distance and magnification); they must be re-measured
#: on our cam4 view before any number this module produces is trusted.
DEFAULTS: dict[str, Any] = {
    "preclean": {
        # Y-side cluster classifier (Step 1a)
        "high_thr_px": 130.0,             # px, OLD RIG -- retune  (also gates.d_y_thr_px, by design)
        "shape_low_thr_px": 80.0,         # px, OLD RIG -- retune
        "shape_range_thr_px": 50.0,       # px, OLD RIG -- retune
        "nbrhd_radius_ms": 100.0,
        # Isolated-short cluster override
        "isolated_max_n": 3,              # frames (transfers only because both rigs are 250 fps)
        "isolated_radius_ms": 20.0,
        # Frame-level y outlier (Step 1b)
        "frame_jump_thr_px": 100.0,       # px, OLD RIG -- retune
        "frame_radius_ms": 20.0,
        # PCHIP-extend wipe (Step 1d)
        "pchip_extend_ms": 40.0,
        # Long flat plateau
        "flat_min_len": 20,               # frames (250 fps)
        "flat_max_range_px": 3.0,         # px, OLD RIG -- retune
        "flat_min_y_px": 30.0,            # px, OLD RIG -- retune
        # x-side rules (their Addendum #1)
        "x_abs_max_px": 110.0,            # px, OLD RIG -- retune   (Rule 3 hard bound on |x|)
        "x_pchip_extend_ms": 40.0,
        "x_jump_thr_px": 30.0,            # px, OLD RIG -- retune   (Rule 4)
        "x_flank_max_distance": 5,        # frames (250 fps)
        # Rule 4 mode: "snapback" = theirs (single frame jumping >thr toward the rest x -> x AND y wiped);
        # "swap" = OURS, retuned 2026-10-06 (direction-agnostic burst of <= x_swap_max_frames, x-only re-PCHIP)
        "x_jump_mode": "snapback",
        "x_swap_max_frames": 3,
    },
    "detector": {
        "slope_thr_px_per_s": 250.0,      # px, OLD RIG -- retune   (px/s)
        "rise_min_ms": 16.0,
        "fall_min_ms": 16.0,
        "peak_pad_ms": 52.0,
        "min_peak_y_abs": 20.0,           # px, OLD RIG -- retune
        "min_finite_samples_per_lick": 5,  # frames (250 fps)
        "strict_no_fallback": True,
        "accept_peaks_without_fall_run": False,
        "do_smooth_y_for_detection": True,
        "smooth_ms_detect_y": 12.0,
        # Read by their per-trial velocity extractor's LEGACY branch only; kept so the block mirrors theirs.
        "do_smooth_vel": True,
        "smooth_ms_vel": 12.0,
    },
    "detector_v2": {
        "lm_min_distance_ms": 50.0,
        "dedupe_radius_ms": 50.0,
        "max_fall_search_ms": 52.0,
    },
    "gates": {
        "f_wp_win_ms": 20.0,
        "f_wp_std_thr": 1.0,              # px, OLD RIG -- retune
        "f_k_min_raw": 3,                 # frames (250 fps)
        "d_win_ms": 150.0,
        "d_y_thr_px": 130.0,              # px, OLD RIG -- retune   (== preclean.high_thr_px by design)
        "e_z_thr": 3.0,
        "e_lik_override": 0.85,           # DLC likelihood; LP's confidence is scaled differently -- recheck
        "e_k_min_peers": 4,
        # Inline `if robust_std < 0.5` in their gate_E; lifted here because it is a px threshold.
        "e_min_robust_std_px": 0.5,       # px, OLD RIG -- retune
        "parabolic_tight_radius_ms": 60.0,
        "parabolic_min_anchors": 5,       # frames (250 fps)
        "spline_excise_radius_ms": 8.0,
        "spline_anchor_win_ms": 160.0,
        "spline_local_max_win_ms": 20.0,
        "spline_min_anchors": 4,          # frames (250 fps)
        "x_outlier_excise_ms": 8.0,
        "m_min_interval_ms": 80.0,
        "m_conf_radius_ms": 20.0,
        "m_early_interval_ms": 60.0,
        "m_early_win_ms": 250.0,          # ms post-CUE here (post-tone in theirs; same event)
    },
}


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def config_overrides() -> dict:
    """`configs/defaults.yaml: orofacial_kinematics.tongue` ({} when the block is absent)."""
    from wfield_local import config
    return (config.defaults().get("orofacial_kinematics") or {}).get("tongue") or {}


def params(overrides: dict | None = None) -> dict:
    """DEFAULTS <- YAML block <- ``overrides`` (deep merge; later wins). Returns a fresh dict."""
    return _deep_merge(_deep_merge(copy.deepcopy(DEFAULTS), config_overrides()), overrides or {})


def interp_masks(fill_method: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(is_interp_fill, is_baseline_fill) from `orofacial_clean.fill_method`.

    Their v5p3 parquet stored these two booleans; `orofacial_clean` stores the code they were derived from
    (1 PCHIP / 2 linear = stage-4 gap fill, 3 = stage-5 baseline fill), so this is a relabel, not a change.
    """
    fm = np.asarray(fill_method)
    return np.isin(fm, (1, 2)), fm == 3


# =========================================================================== pre-clean (_preclean.py)

@dataclass
class RawCluster:
    """One contiguous run of raw DLC frames (their `_RawCluster`)."""
    lo: int                 # first frame index (inclusive)
    hi: int                 # last frame index (inclusive)
    length: int             # hi - lo + 1
    y_min: float
    y_max: float
    y_median: float
    y_range: float
    label: str              # "keep_low" / "keep_real ..." / "artifact_*"


@dataclass
class PrecleanDiagnostics:
    """Per-trace cleaning counters (their `_PrecleanDiagnostics`)."""
    clusters: list[RawCluster] = field(default_factory=list)
    n_artifact_clusters: int = 0
    n_artifact_frames: int = 0
    n_interp_wiped: int = 0          # bracketed-interp wipes (Step 1e)
    n_frame_outliers: int = 0        # Step 1b single-frame y outliers
    n_pchip_extended: int = 0        # Step 1d PCHIP-extend frames
    n_x_hard_outliers: int = 0       # Rule 3
    n_x_joint_outliers: int = 0      # Rule 4 (snapback mode)
    n_x_swaps: int = 0               # Rule 4 (swap mode): frames whose x was re-interpolated, y kept
    n_x_pchip_extended: int = 0      # Rule 3 PCHIP-extend
    x0_baseline_fill: float = 0.0    # Rule 1


def preclean_trace(y, x, is_interp_fill, is_baseline_fill, n_frames, fps, *, params):
    """v7 pre-clean: y-side cluster + frame rules, then x-side Rules 1-4. Their `preclean_trace`.

    Returns ``(y_clean, x_clean, is_baseline_fill_clean, diagnostics)``. Wiped y frames are set to 0 (the
    baseline-fill value of a baseline-subtracted trace) and flagged baseline-fill, so the detector sees them
    as "tongue in".
    """
    pchip_extend = max(2, int(round(float(params["pchip_extend_ms"]) / 1000.0 * fps)))

    # Step 1a: classify clusters
    clusters = _classify_raw_clusters(y, is_interp_fill, is_baseline_fill, n_frames, fps, params=params)

    y_clean = y.copy()
    is_base_clean = is_baseline_fill.copy()
    n_pchip_extended = 0

    # Step 1c: wipe artifact clusters + Step 1d PCHIP-extend around each
    for c in clusters:
        if not c.label.startswith("artifact"):
            continue
        y_clean[c.lo:c.hi + 1] = 0.0
        is_base_clean[c.lo:c.hi + 1] = True
        n_pchip_extended += _extend_pchip_around_range(c.lo, c.hi, y_clean, is_base_clean, is_interp_fill,
                                                       n_frames, pchip_extend)

    # Step 1b: frame-level y outliers within kept clusters -- run AFTER 1c, as theirs does, so already-wiped
    # clusters are not revisited. The local median reads y_clean, i.e. after the 1c wipes.
    n_frame_outliers = 0
    for k in _find_y_frame_outliers(y_clean, clusters, n_frames, fps, params=params):
        if is_base_clean[k]:
            continue
        y_clean[k] = 0.0
        is_base_clean[k] = True
        n_frame_outliers += 1
        n_pchip_extended += _extend_pchip_around_range(k, k, y_clean, is_base_clean, is_interp_fill,
                                                       n_frames, pchip_extend)

    # Step 1e: bracketed-interp wipe
    n_interp_wiped = _wipe_bracketed_interp(y_clean, is_interp_fill, is_base_clean, n_frames)

    # X-side (Rules 1-4)
    x_clean, x_hard, x_joint, x_extend, is_base_aug, x0, x_swap = _clean_x_trace(
        x, is_interp_fill, is_base_clean, is_baseline_fill, n_frames, fps, params=params)

    # Rule 4: at joint x+y outlier frames ALSO wipe y, so the detector runs on the joint-cleaned trace.
    if x_joint.any():
        y_clean[x_joint] = 0.0
        is_base_clean = is_base_aug

    arts = [c for c in clusters if c.label.startswith("artifact")]
    diag = PrecleanDiagnostics(
        clusters=clusters, n_artifact_clusters=len(arts), n_artifact_frames=sum(c.length for c in arts),
        n_interp_wiped=n_interp_wiped, n_frame_outliers=n_frame_outliers, n_pchip_extended=n_pchip_extended,
        n_x_hard_outliers=int(x_hard.sum()), n_x_joint_outliers=int(x_joint.sum()),
        n_x_pchip_extended=int(x_extend.sum()), x0_baseline_fill=x0, n_x_swaps=int(x_swap.sum()))
    return y_clean, x_clean, is_base_clean, diag


def _classify_raw_clusters(y, is_interp_fill, is_baseline_fill, n_frames, fps, *, params):
    """Step 1a: label each raw cluster keep_low / keep_real / artifact_{narrow,isolated_short,flat}.

    A cluster that reaches high (>= high_thr) is REAL if it has lick shape on its own (Rule A: dips low, or
    spans a wide range) or if it is attached to the trace and has a low non-baseline neighbour nearby
    (Rule B). Otherwise it is a narrow high plateau -- the DLC jump-to-a-wrong-landmark signature.
    """
    high_thr = float(params["high_thr_px"])
    shape_low_thr = float(params["shape_low_thr_px"])
    shape_range_thr = float(params["shape_range_thr_px"])
    isolated_max_n = int(params["isolated_max_n"])
    flat_min_len = int(params["flat_min_len"])
    flat_max_range = float(params["flat_max_range_px"])
    flat_min_y = float(params["flat_min_y_px"])
    nbrhd_radius = max(1, int(round(float(params["nbrhd_radius_ms"]) / 1000.0 * fps)))
    isolated_radius = max(1, int(round(float(params["isolated_radius_ms"]) / 1000.0 * fps)))

    raw_mask = (~is_interp_fill) & (~is_baseline_fill) & np.isfinite(y)
    non_base_mask = (~is_baseline_fill) & np.isfinite(y)   # raw + PCHIP-interp count as non-base

    clusters: list[RawCluster] = []
    i = 0
    while i < n_frames:
        if not raw_mask[i]:
            i += 1
            continue
        j = i
        while j < n_frames and raw_mask[j]:
            j += 1
        cluster_y = y[i:j]
        n = j - i
        y_max = float(np.max(cluster_y))
        y_min = float(np.min(cluster_y))
        y_median = float(np.median(cluster_y))
        y_range = y_max - y_min

        label = "keep_low"
        if y_max >= high_thr:
            rule_A = (y_min < shape_low_thr) or (y_range > shape_range_thr)
            left_connected = (i > 0) and bool(non_base_mask[i - 1])
            right_connected = (j < n_frames) and bool(non_base_mask[j])
            rule_B1 = left_connected or right_connected
            nlo = max(0, i - nbrhd_radius)
            nhi = min(n_frames, j + nbrhd_radius)
            nbrhd_nb = non_base_mask[nlo:nhi].copy()
            nbrhd_nb[(i - nlo):(j - nlo)] = False  # exclude self
            rule_B2 = bool((nbrhd_nb & (y[nlo:nhi] < high_thr)).any())
            rule_B = rule_B1 and rule_B2
            if rule_A or rule_B:
                label = f"keep_real (A={int(rule_A)} B1={int(rule_B1)} B2={int(rule_B2)})"
            else:
                label = (f"artifact_narrow max={y_max:.0f} min={y_min:.0f} range={y_range:.0f} n={n} "
                         f"B1={int(rule_B1)} B2={int(rule_B2)}")
            # Isolated-short override: short high cluster with no other raw cluster within +-isolated_radius
            if n <= isolated_max_n:
                iso_lo = max(0, i - isolated_radius)
                iso_hi = min(n_frames, j + isolated_radius)
                other_raw = raw_mask[iso_lo:iso_hi].copy()
                other_raw[(i - iso_lo):(j - iso_lo)] = False
                if not bool(other_raw.any()):
                    label = f"artifact_isolated_short max={y_max:.0f} n={n}"

        # Long flat plateau (any height above flat_min_y) -- a frozen tracker, not a tongue.
        if n >= flat_min_len and y_range < flat_max_range and y_median > flat_min_y:
            label = f"artifact_flat n={n} range={y_range:.1f}"

        clusters.append(RawCluster(lo=i, hi=j - 1, length=n, y_min=y_min, y_max=y_max, y_median=y_median,
                                   y_range=y_range, label=label))
        i = j
    return clusters


def _find_y_frame_outliers(y_clean, clusters, n_frames, fps, *, params):
    """Step 1b: frames in kept clusters with |y - median(other frames within +-frame_radius, same cluster)| >
    frame_jump_thr. Needs >= 3 other finite frames, as theirs."""
    frame_jump_thr = float(params["frame_jump_thr_px"])
    frame_radius = max(2, int(round(float(params["frame_radius_ms"]) / 1000.0 * fps)))
    seeds: list[int] = []
    for c in clusters:
        if c.label.startswith("artifact"):
            continue
        lo, hi = c.lo, c.hi
        for k in range(lo, hi + 1):
            klo = max(lo, k - frame_radius)
            khi = min(hi + 1, k + frame_radius + 1)
            others = [y_clean[kk] for kk in range(klo, khi) if kk != k and np.isfinite(y_clean[kk])]
            if len(others) < 3:
                continue
            if abs(float(y_clean[k]) - float(np.median(others))) > frame_jump_thr:
                seeds.append(k)
    return seeds


def _extend_pchip_around_range(lo, hi, y_clean, is_base_clean, is_interp_fill, n_frames, pchip_extend):
    """Step 1d: wipe up to ``pchip_extend`` interp frames on each side of [lo, hi] (in place).

    Why: a PCHIP fill next to a wiped artifact was ANCHORED on that artifact, so it carries the artifact's
    height into frames that look clean. Returns the number of newly wiped frames.
    """
    n_extended = 0
    k, cnt = lo - 1, 0
    while k >= 0 and is_interp_fill[k] and not is_base_clean[k] and cnt < pchip_extend:
        y_clean[k] = 0.0
        is_base_clean[k] = True
        n_extended += 1
        k -= 1
        cnt += 1
    k, cnt = hi + 1, 0
    while k < n_frames and is_interp_fill[k] and not is_base_clean[k] and cnt < pchip_extend:
        y_clean[k] = 0.0
        is_base_clean[k] = True
        n_extended += 1
        k += 1
        cnt += 1
    return n_extended


def _wipe_bracketed_interp(y_clean, is_interp_fill, is_base_clean, n_frames):
    """Step 1e: wipe any interp run bracketed on BOTH sides by baseline/wiped frames (in place) -- such a run
    was interpolated between anchors that no longer exist."""
    n_wiped = 0
    i = 0
    while i < n_frames:
        if is_interp_fill[i] and not is_base_clean[i]:
            j = i
            while j < n_frames and is_interp_fill[j] and not is_base_clean[j]:
                j += 1
            left_wiped = (i == 0) or bool(is_base_clean[i - 1])
            right_wiped = (j >= n_frames) or bool(is_base_clean[j])
            if left_wiped and right_wiped:
                y_clean[i:j] = 0.0
                is_base_clean[i:j] = True
                n_wiped += (j - i)
            i = j
        else:
            i += 1
    return n_wiped


def _derive_x0_baseline_fill(x, is_baseline_fill_orig):
    """Rule 1 helper: median x over the ORIGINAL baseline-fill frames (0.0 if none).

    On `orofacial_clean` output this is 0 by construction (baseline frames are X0 - X0); kept so the rule is
    the same rule if a caller ever feeds an un-subtracted trace.
    """
    if not is_baseline_fill_orig.any():
        return 0.0
    vals = x[is_baseline_fill_orig]
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return 0.0
    return float(np.median(vals))


def _find_x_isolated_jumps(x, is_interp_fill, is_baseline_fill_clean, n_frames, x0, *, params):
    """Rule 4: raw frames deviating > x_jump_thr from BOTH nearest raw flanks AND sitting closer to x0 than
    both -- the DLC snap-back-to-rest pattern, which a real lateral lick (moving AWAY from x0) never shows."""
    jump_thr = float(params["x_jump_thr_px"])
    max_flank_distance = int(params["x_flank_max_distance"])
    raw_mask = (~is_interp_fill) & (~is_baseline_fill_clean) & np.isfinite(x)
    flag = np.zeros(n_frames, dtype=bool)
    for i in range(n_frames):
        if not raw_mask[i]:
            continue
        left_idx = -1
        for d in range(1, max_flank_distance + 1):
            k = i - d
            if k < 0:
                break
            if raw_mask[k]:
                left_idx = k
                break
        if left_idx < 0:
            continue
        right_idx = -1
        for d in range(1, max_flank_distance + 1):
            k = i + d
            if k >= n_frames:
                break
            if raw_mask[k]:
                right_idx = k
                break
        if right_idx < 0:
            continue
        x_left, x_right, x_curr = float(x[left_idx]), float(x[right_idx]), float(x[i])
        if abs(x_curr - x_left) <= jump_thr:
            continue
        if abs(x_curr - x_right) <= jump_thr:
            continue
        if abs(x_curr - x0) >= min(abs(x_left - x0), abs(x_right - x0)):
            continue
        flag[i] = True
    return flag


def _find_x_swaps(x, raw_mask, n_frames, *, params):
    """Rule 4, retuned for the mobile-spout rig (OURS, Priya 2026-10-06): a burst of 1..x_swap_max_frames raw
    frames whose x sits more than x_jump_thr_px from BOTH nearest raw flanks, while the flanks agree with each
    other (within the same threshold) -- in EITHER direction. On this rig the tongue straddles the spout on
    licks, and DLC's tip can hop to the lobe on the other side of the spout for a frame or two (2026-10-06, PS93
    0814 trial 30: 29 px for one frame, at the lick's peak). Theirs flagged only single frames jumping TOWARD the
    rest x and wiped x AND y; a swap's distance from the mouth is real, so only x is re-interpolated. Threshold
    from the rig's data: among frames whose flanks agree, the both-flank deviation is p99 14 px, p99.9 21 px."""
    thr = float(params["x_jump_thr_px"])
    max_flank = int(params["x_flank_max_distance"])
    max_burst = int(params.get("x_swap_max_frames", 3))
    raw_idx = np.flatnonzero(raw_mask)
    flag = np.zeros(n_frames, dtype=bool)
    for k0, i in enumerate(raw_idx):
        if flag[i] or k0 == 0:
            continue
        left = raw_idx[k0 - 1]
        if i - left > max_flank:
            continue
        for L in range(1, max_burst + 1):
            k1 = k0 + L
            if k1 >= len(raw_idx):
                break
            right = raw_idx[k1]
            burst = raw_idx[k0:k1]
            if right - burst[-1] > max_flank or np.any(np.diff(burst) > max_flank):
                break
            xl, xr = float(x[left]), float(x[right])
            if abs(xl - xr) > thr:
                continue
            xb = x[burst]
            if np.all(np.abs(xb - xl) > thr) and np.all(np.abs(xb - xr) > thr):
                flag[burst] = True
                break
    return flag


def _clean_x_trace(x, is_interp_fill, is_baseline_fill_clean, is_baseline_fill_orig, n_frames, fps, *, params):
    """x-side Rules 1-4. Returns (x_clean, hard_mask, joint_mask, extend_mask, is_base_clean_augmented, x0,
    swap_mask); Rule 4 per ``x_jump_mode`` (joint_mask in "snapback", swap_mask in "swap" -- the other empty)."""
    x_abs_max = float(params["x_abs_max_px"])
    pchip_extend = max(2, int(round(float(params["x_pchip_extend_ms"]) / 1000.0 * fps)))

    x0 = _derive_x0_baseline_fill(x, is_baseline_fill_orig)
    # Built on the POST-y-clean baseline mask, so y-wiped frames are not x anchors (Rule 2) for free.
    raw_mask = (~is_interp_fill) & (~is_baseline_fill_clean) & np.isfinite(x)

    x_hard = raw_mask & (np.abs(x) > x_abs_max)                       # Rule 3
    if params.get("x_jump_mode", "snapback") == "swap":
        x_joint = np.zeros(n_frames, dtype=bool)
        x_swap = _find_x_swaps(x, raw_mask & ~x_hard, n_frames, params=params)
    else:
        x_joint = _find_x_isolated_jumps(x, is_interp_fill, is_baseline_fill_clean, n_frames, x0, params=params)
        x_joint &= ~x_hard
        x_swap = np.zeros(n_frames, dtype=bool)

    # PCHIP-extend for HARD outliers only (a joint outlier is a brief burst; its neighbours are not tainted).
    # NB as in theirs this mask is only COUNTED -- the extended frames are re-PCHIP'd below because they are
    # interp frames anyway, not because they are in this mask.
    x_extend = np.zeros(n_frames, dtype=bool)
    for i in np.where(x_hard)[0]:
        k, cnt = i - 1, 0
        while k >= 0 and is_interp_fill[k] and not is_baseline_fill_clean[k] and cnt < pchip_extend:
            x_extend[k] = True
            k -= 1
            cnt += 1
        k, cnt = i + 1, 0
        while k < n_frames and is_interp_fill[k] and not is_baseline_fill_clean[k] and cnt < pchip_extend:
            x_extend[k] = True
            k += 1
            cnt += 1

    anchor_mask = raw_mask & (~x_hard) & (~x_joint) & (~x_swap)
    anchor_idx = np.where(anchor_mask)[0]
    x_clean = np.full(n_frames, np.nan, dtype=np.float64)
    x_clean[anchor_idx] = x[anchor_idx]

    is_base_aug = is_baseline_fill_clean | x_joint
    # Rule 1: x = x0 at y-side baseline frames only, NOT at joint outliers (those sit inside a real lick).
    x_clean[is_baseline_fill_clean] = x0

    if len(anchor_idx) >= 4:
        pchip = PchipInterpolator(anchor_idx.astype(float), x[anchor_idx], extrapolate=False)
        interp_mask = (is_interp_fill & (~is_baseline_fill_clean)) | x_hard | x_joint | x_swap
        idx = np.where(interp_mask)[0]
        if len(idx):
            vals = pchip(idx.astype(float))
            fin = np.isfinite(vals)
            x_clean[idx[fin]] = vals[fin]
    # Frames that end up NaN here (hard/joint outliers outside the anchor span, or < 4 anchors) stay NaN, as
    # in theirs.
    return x_clean, x_hard, x_joint, x_extend, is_base_aug, x0, x_swap


# =========================================================================== slope detector (_detector.py)

@dataclass
class LickPeaks:
    """Output of `slope_detect_lick_peaks` (their `_LickPeaks`). Indices are local to the detect-window
    segment ``yseg = y[i_lo:i_hi+1]``; ``peak_times_ms`` are already trial-relative."""
    peak_times_ms: np.ndarray
    peak_y: np.ndarray
    peak_x: np.ndarray
    rise_starts: np.ndarray
    turnover: np.ndarray
    peak_indices: np.ndarray
    fall_ends: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))

    @classmethod
    def empty(cls) -> LickPeaks:
        z = np.zeros(0, dtype=np.float64)
        zi = np.zeros(0, dtype=np.int64)
        return cls(z, z.copy(), z.copy(), zi, zi.copy(), zi.copy(), zi.copy())

    def __len__(self) -> int:
        return len(self.peak_times_ms)


def _moving_avg_nan(arr, window_ms, fps):
    """NaN-tolerant centred moving average over ``window_ms`` (their `_moving_avg_nan`)."""
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


def _first_run_from(mask, min_len, start):
    """First True run of length >= ``min_len`` starting at index >= ``start`` -> (start, end_incl) or (None, None)."""
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


def _window_bounds(t_ms, detect_win_ms):
    """(i_lo, i_hi) of the first/last sample inside ``detect_win_ms``, or None -- shared clip of all detectors."""
    in_win = (t_ms >= detect_win_ms[0]) & (t_ms <= detect_win_ms[1])
    if not np.any(in_win):
        return None
    return int(np.argmax(in_win)), int(len(in_win) - 1 - np.argmax(in_win[::-1]))


def slope_detect_lick_peaks(y_t, x_t, t_ms, fps, *, detect_win_ms, detector_params, n_max=None,
                            is_baseline_fill=None) -> LickPeaks:
    """Rise/turnover/fall slope detector (their `_slope_detect_lick_peaks`).

    Per lick: dy/dt > slope_thr for >= rise_min_ms; turnover where dy/dt stops being positive (NaN counts as
    still rising, their port-bug fix #1); peak = argmax y in [rise start, turnover + peak_pad]; |y_peak| >=
    min_peak_y_abs; >= min_finite_samples_per_lick finite y in +-peak_pad; then a fall run of >= fall_min_ms.

    The v7 orchestrator calls this twice: once as the "legacy" detector (no extra keys), once as the
    "bounded" detector with ``max_fall_search_ms`` set (fall searched only within that span after the peak)
    and ``accept_implicit_baseline_fall`` True (no fall run is fine if the very next frame is baseline-fill,
    i.e. the tongue vanished on retraction).
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
    mfs = detector_params.get("max_fall_search_ms")
    max_fall_search_n = max(1, int(round(float(mfs) / 1000.0 * fps))) if mfs is not None else None
    accept_implicit_baseline_fall = bool(detector_params.get("accept_implicit_baseline_fall", False))

    if y_t.size == 0:
        return LickPeaks.empty()
    b = _window_bounds(t_ms, detect_win_ms)
    if b is None:
        return LickPeaks.empty()
    i_lo, i_hi = b
    yseg = y_t[i_lo:i_hi + 1].astype(np.float64, copy=True)
    xseg = x_t[i_lo:i_hi + 1].astype(np.float64, copy=True)
    tseg = t_ms[i_lo:i_hi + 1]
    is_base_seg = is_baseline_fill[i_lo:i_hi + 1] if is_baseline_fill is not None else None

    y_for_slope = _moving_avg_nan(yseg, smooth_ms_y, fps) if do_smooth_y else yseg.copy()
    finite = np.isfinite(y_for_slope)
    dy = np.full_like(y_for_slope, np.nan, dtype=np.float64)
    if np.count_nonzero(finite) >= 2:
        dy = np.gradient(y_for_slope) * fps  # px/s
    # x enters here: a frame with NaN x (an x-side outlier pre-clean could not re-interpolate) cannot start or
    # carry a rise/fall run, exactly as in theirs.
    finite_dy = np.isfinite(dy) & np.isfinite(yseg) & np.isfinite(xseg)
    rise_mask = finite_dy & (dy > slope_thr)
    fall_mask = finite_dy & (dy < -slope_thr)
    rise_min_n = max(1, int(round(rise_min_ms / 1000.0 * fps)))
    fall_min_n = max(1, int(round(fall_min_ms / 1000.0 * fps)))
    peak_pad_n = max(1, int(round(peak_pad_ms / 1000.0 * fps)))

    L = len(yseg)
    rise_starts, turnovers, peaks, times, ys, xs, fall_ends = [], [], [], [], [], [], []
    start = 0
    while True:
        if n_max is not None and len(peaks) >= n_max:
            break
        r0, r1 = _first_run_from(rise_mask, rise_min_n, start)
        if r0 is None:
            break
        k = r1
        while k < L and (not np.isfinite(dy[k]) or dy[k] > 0):
            k += 1
        if k >= L:
            if strict_no_fallback:
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
            start = k + 1
            continue
        win_lo = max(0, i_peak - peak_pad_n)
        win_hi = min(L - 1, i_peak + peak_pad_n)
        if int(np.count_nonzero(np.isfinite(yseg[win_lo:win_hi + 1]))) < min_finite:
            start = k + 1
            continue
        # Fall validation; the next search starts after the fall run's end (their port-bug fixes #2/#3).
        end_idx = i_peak
        if fall_min_n > 1:
            if max_fall_search_n is not None:
                search_hi = min(L, i_peak + max_fall_search_n + 1)
                effective_fall_mask = fall_mask.copy()
                effective_fall_mask[search_hi:] = False
            else:
                effective_fall_mask = fall_mask
            f0, f1 = _first_run_from(effective_fall_mask, fall_min_n, i_peak)
            if f0 is None:
                implicit = (accept_implicit_baseline_fall and is_base_seg is not None and (i_peak + 1) < L
                            and bool(is_base_seg[i_peak + 1]))
                if implicit:
                    end_idx = i_peak + 1
                elif strict_no_fallback and not accept_no_fall:
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
        fall_ends.append(end_idx)
        start = end_idx + 1

    return LickPeaks(np.asarray(times, dtype=np.float64), np.asarray(ys, dtype=np.float64),
                     np.asarray(xs, dtype=np.float64), np.asarray(rise_starts, dtype=np.int64),
                     np.asarray(turnovers, dtype=np.int64), np.asarray(peaks, dtype=np.int64),
                     np.asarray(fall_ends, dtype=np.int64))


# =========================================================================== v7 detector (_detector_v2.py)

@dataclass
class DetectedPeak:
    """One merged peak with provenance (their `_DetectedPeak`). Frames are detect-segment-local -- see the
    NOTE in `merge_detector_outputs`."""
    frame: int
    t_ms: float
    y: float
    x: float
    likelihood: float
    source: str  # "lmax" | "bounded" | "legacy"
    rise_start_frame: int = -1
    fall_end_frame: int = -1


@dataclass
class PeakList:
    """Parallel-array peak list (their `_PeakList`); indices detect-segment-local."""
    peak_times_ms: np.ndarray
    peak_y: np.ndarray
    peak_x: np.ndarray
    peak_indices: np.ndarray
    rise_starts: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    fall_ends: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))

    @classmethod
    def empty(cls) -> PeakList:
        z = np.zeros(0, dtype=np.float64)
        zi = np.zeros(0, dtype=np.int64)
        return cls(z, z.copy(), z.copy(), zi, zi.copy(), zi.copy())

    def __len__(self) -> int:
        return len(self.peak_times_ms)


def detect_local_maxima(y, x, t_ms, fps, *, is_baseline_fill, detect_win_ms, n_max=None, detector_params,
                        params) -> PeakList:
    """find_peaks local maxima, each validated by a rise run before and a fall run (or an immediate
    baseline retraction) after. Their `detect_local_maxima`; tracks rise_start = p - consec_rise and
    fall_end = p + consec_fall (p + 1 on implicit baseline fall)."""
    slope_thr = float(detector_params["slope_thr_px_per_s"])
    rise_min_ms = float(detector_params["rise_min_ms"])
    fall_min_ms = float(detector_params["fall_min_ms"])
    min_peak_y_abs = float(detector_params["min_peak_y_abs"])
    do_smooth_y = bool(detector_params["do_smooth_y_for_detection"])
    smooth_ms_y = float(detector_params["smooth_ms_detect_y"])
    lm_min_distance_ms = float(params["lm_min_distance_ms"])

    if y.size == 0:
        return PeakList.empty()
    b = _window_bounds(t_ms, detect_win_ms)
    if b is None:
        return PeakList.empty()
    i_lo, i_hi = b
    yseg = y[i_lo:i_hi + 1].astype(np.float64, copy=True)
    xseg = x[i_lo:i_hi + 1].astype(np.float64, copy=True)
    tseg = t_ms[i_lo:i_hi + 1]
    is_base_seg = is_baseline_fill[i_lo:i_hi + 1]

    y_for_slope = _moving_avg_nan(yseg, smooth_ms_y, fps) if do_smooth_y else yseg.copy()
    finite = np.isfinite(y_for_slope)
    dy = np.full_like(y_for_slope, np.nan, dtype=np.float64)
    if np.count_nonzero(finite) >= 2:
        dy = np.gradient(y_for_slope) * fps
    rise_min_n = max(1, int(round(rise_min_ms / 1000.0 * fps)))
    fall_min_n = max(1, int(round(fall_min_ms / 1000.0 * fps)))
    min_distance = max(1, int(round(lm_min_distance_ms / 1000.0 * fps)))

    # find_peaks on the SMOOTHED y (theirs); NaN -> -inf so a gap is never a peak.
    y_search = y_for_slope.copy()
    y_search[~np.isfinite(y_search)] = -np.inf
    peaks_idx, _ = find_peaks(y_search, height=min_peak_y_abs, distance=min_distance)

    L = len(yseg)
    times, ys, xs, pidx, rss, fes = [], [], [], [], [], []
    for p in peaks_idx:
        consec_rise = 0
        for k in range(p - 1, max(-1, p - rise_min_n * 3), -1):
            if k < 0 or not np.isfinite(dy[k]):
                break
            if dy[k] > slope_thr:
                consec_rise += 1
                if consec_rise >= rise_min_n:
                    break
            else:
                break
        if consec_rise < rise_min_n:
            continue
        consec_fall = 0
        for k in range(p + 1, min(L, p + fall_min_n * 3)):
            if not np.isfinite(dy[k]):
                break
            if dy[k] < -slope_thr:
                consec_fall += 1
                if consec_fall >= fall_min_n:
                    break
            else:
                break
        fall_ok = consec_fall >= fall_min_n
        implicit_baseline_fall = False
        if not fall_ok and p + 1 < L and is_base_seg[p + 1]:
            fall_ok = True
            implicit_baseline_fall = True
        if not fall_ok:
            continue
        # Height test on the UNSMOOTHED y (find_peaks' height was on the smoothed one) -- theirs, both.
        y_peak = yseg[p]
        if not np.isfinite(y_peak) or abs(y_peak) < min_peak_y_abs:
            continue
        times.append(float(tseg[p]))
        ys.append(float(y_peak))
        xs.append(float(xseg[p]))
        pidx.append(int(p))
        rss.append(int(p - consec_rise))
        fes.append(int(p + 1) if implicit_baseline_fall else int(p + consec_fall))
        if n_max is not None and len(times) >= n_max:
            break
    return PeakList(np.asarray(times, dtype=np.float64), np.asarray(ys, dtype=np.float64),
                    np.asarray(xs, dtype=np.float64), np.asarray(pidx, dtype=np.int64),
                    np.asarray(rss, dtype=np.int64), np.asarray(fes, dtype=np.int64))


def legacy_peaks_to_peak_list(legacy: LickPeaks, detector_params, fps) -> PeakList:
    """`LickPeaks` -> `PeakList` (their `tongue_pertrial._legacy_peaks_to_peak_list`). ``turnover`` is dropped;
    a missing ``fall_ends`` falls back to peak + peak_pad (never happens with this module's detector)."""
    if len(legacy) == 0:
        return PeakList.empty()
    if len(legacy.fall_ends) == len(legacy.peak_indices):
        fall_ends = legacy.fall_ends
    else:
        peak_pad_n = max(1, int(round(float(detector_params["peak_pad_ms"]) / 1000.0 * fps)))
        fall_ends = legacy.peak_indices + peak_pad_n
    return PeakList(legacy.peak_times_ms, legacy.peak_y, legacy.peak_x, legacy.peak_indices,
                    legacy.rise_starts, fall_ends)


def shift_peak_list(pl: PeakList, offset: int) -> PeakList:
    """OURS (the fix for the NOTE in `merge_detector_outputs`): detect-segment-local frames -> trial-slice
    frames by adding the window's first index. Times / y / x are already right and pass through."""
    if pl is None or len(pl) == 0 or offset == 0:
        return pl
    return PeakList(pl.peak_times_ms, pl.peak_y, pl.peak_x, pl.peak_indices + offset,
                    pl.rise_starts + offset, pl.fall_ends + offset)


def merge_detector_outputs(peak_lists, likelihood, fps, *, params) -> list[DetectedPeak]:
    """Cross-detector dedupe (their `merge_detector_outputs`): in the order given (lmax, bounded, legacy), drop
    a peak within dedupe_radius frames of one already added. Sorted by frame.

    NOTE -- A SOURCE OFF-BY-ONE, TRANSCRIBED AS-IS: the peak indices are DETECT-SEGMENT-local (index 0 = the
    first frame with t >= 70 ms), but their v7 orchestrator indexes the TRIAL-SLICE likelihood with them here
    and then uses them as trial-slice frames in the gates, for x at the peak, for rise/fall ms and for the
    velocity windows. The trial slice starts at floor(70 ms * 250 fps) = 17 frames = 68 ms, so index 0 of the
    slice is OUTSIDE the window and the segment is shifted by exactly one frame (4 ms): every frame-indexed
    quantity is read one frame EARLY. Peak TIMES are unaffected (they come from the segment's own t axis).
    Kept so results match theirs; the fix is to add the window's i_lo to every index before it leaves the
    detectors. (With a float cue frame the shift is still one frame unless the cue's fractional part is
    exactly .5, where the slice's first frame lands on 70.0 ms and the shift is zero.)
    """
    dedupe_radius = max(1, int(round(float(params["dedupe_radius_ms"]) / 1000.0 * fps)))
    added: list[DetectedPeak] = []
    for source, pl in peak_lists:
        if pl is None or len(pl) == 0:
            continue
        for k in range(len(pl)):
            pf = int(pl.peak_indices[k])
            if any(abs(pf - p.frame) <= dedupe_radius for p in added):
                continue
            lik = float(likelihood[pf]) if 0 <= pf < len(likelihood) else float("nan")
            added.append(DetectedPeak(
                frame=pf, t_ms=float(pl.peak_times_ms[k]), y=float(pl.peak_y[k]), x=float(pl.peak_x[k]),
                likelihood=lik, source=source,
                rise_start_frame=int(pl.rise_starts[k]) if k < len(pl.rise_starts) else -1,
                fall_end_frame=int(pl.fall_ends[k]) if k < len(pl.fall_ends) else -1))
    added.sort(key=lambda p: p.frame)
    return added


# =========================================================================== gates (_gates.py)

def gate_F(y, is_interp_fill, is_baseline_fill, peak_frame, n_frames, fps, *, params):
    """Within-peak raw std: REJECT if std(raw y in +-f_wp_win) < f_wp_std_thr (a flat plateau, not a lick).
    Returns (reject, reason, std_raw)."""
    win_frames = int(round(float(params["f_wp_win_ms"]) / 1000.0 * fps))
    std_thr = float(params["f_wp_std_thr"])
    k_min_raw = int(params["f_k_min_raw"])
    lo = max(0, peak_frame - win_frames)
    hi = min(n_frames, peak_frame + win_frames + 1)
    y_w = y[lo:hi]
    raw_y = y_w[(~is_interp_fill[lo:hi]) & (~is_baseline_fill[lo:hi]) & np.isfinite(y_w)]
    if len(raw_y) < k_min_raw:
        return True, f"F_too_few_raw n={len(raw_y)}", float("nan")
    std_raw = float(np.std(raw_y))
    if std_raw < std_thr:
        return True, f"F_raw_flat std={std_raw:.2f}", std_raw
    return False, "", std_raw


def gate_D(y, is_interp_fill, is_baseline_fill, peak_frame, n_frames, fps, *, params):
    """Left/right low context: KEEP only if both +-d_win sides hold a low raw frame (< d_y_thr) or a baseline
    frame -- a lick goes out AND comes back. Returns (keep, reason); note True = keep, unlike gate_F."""
    win_frames = int(round(float(params["d_win_ms"]) / 1000.0 * fps))
    y_thr = float(params["d_y_thr_px"])
    lo = max(0, peak_frame - win_frames)
    hi = min(n_frames, peak_frame + win_frames + 1)
    y_w = y[lo:hi]
    is_base_w = is_baseline_fill[lo:hi]
    raw_mask = (~is_interp_fill[lo:hi]) & (~is_base_w) & np.isfinite(y_w)
    pi = peak_frame - lo
    left_ok = bool((raw_mask[:pi] & (y_w[:pi] < y_thr)).any()) or bool(is_base_w[:pi].any())
    right_ok = bool((raw_mask[pi + 1:] & (y_w[pi + 1:] < y_thr)).any()) or bool(is_base_w[pi + 1:].any())
    if left_ok and right_ok:
        return True, ""
    if not left_ok:
        return False, "D_left"
    return False, "D_right"


def gate_E(candidate_y, candidate_lik, peer_ys, *, params):
    """Outlier vs peers: REJECT if robust z > e_z_thr AND likelihood < e_lik_override.
    Returns (reject, reason, z, peer_median). Abstains with < e_k_min_peers peers or a near-zero MAD."""
    z_thr = float(params["e_z_thr"])
    lik_override = float(params["e_lik_override"])
    k_min = int(params["e_k_min_peers"])
    min_rstd = float(params["e_min_robust_std_px"])   # theirs: inline 0.5 (lifted to config; see DEFAULTS)
    peers = [yp for yp in peer_ys if np.isfinite(yp)]
    if len(peers) < k_min:
        return False, "abstain_few_peers", float("nan"), float("nan")
    median = float(np.median(peers))
    mad = float(np.median(np.abs(np.array(peers) - median)))
    robust_std = 1.4826 * mad                         # MAD -> sigma for a normal; a statistical constant
    if robust_std < min_rstd:
        return False, "abstain_stationary", float("nan"), median
    z = (candidate_y - median) / robust_std
    if z > z_thr and candidate_lik < lik_override:
        return True, f"E_outlier z={z:.2f}", z, median
    return False, "", z, median


def _anchor_idx(mask_src, is_interp_fill, is_baseline_fill, lo_out, hi_out, a_lo, a_hi, n_frames):
    """Raw frames of ``mask_src`` inside [a_lo, a_hi) with [lo_out, hi_out) excised -- the anchor rule shared
    by `_compute_imputed_x`, `parabolic_excise`, `spline_excise` (three identical copies in theirs)."""
    raw_mask = (~is_interp_fill) & (~is_baseline_fill) & np.isfinite(mask_src)
    cand = raw_mask.copy()
    cand[lo_out:hi_out] = False
    win = np.zeros(n_frames, dtype=bool)
    win[a_lo:a_hi] = True
    return np.where(cand & win)[0], raw_mask


def _compute_imputed_x(x, is_interp_fill, is_baseline_fill, outlier_peak_frame, new_peak_frame, n_frames, fps,
                       *, params):
    """x at the imputed peak via PCHIP through raw x anchors in +-spline_anchor_win, with +-x_outlier_excise
    around the outlier removed. NaN if anchors are too few / one-sided / do not span ``new_peak_frame``."""
    excise_radius = max(1, int(round(float(params["x_outlier_excise_ms"]) / 1000.0 * fps)))
    anchor_radius = max(4, int(round(float(params["spline_anchor_win_ms"]) / 1000.0 * fps)))
    min_anchors = int(params["spline_min_anchors"])
    lo_out = max(0, outlier_peak_frame - excise_radius)
    hi_out = min(n_frames, outlier_peak_frame + excise_radius + 1)
    a_lo = max(0, outlier_peak_frame - anchor_radius)
    a_hi = min(n_frames, outlier_peak_frame + anchor_radius + 1)
    anchor_idx, _ = _anchor_idx(x, is_interp_fill, is_baseline_fill, lo_out, hi_out, a_lo, a_hi, n_frames)
    if len(anchor_idx) < min_anchors:
        return float("nan")
    if not ((anchor_idx < outlier_peak_frame).any() and (anchor_idx > outlier_peak_frame).any()):
        return float("nan")
    if new_peak_frame < anchor_idx.min() or new_peak_frame > anchor_idx.max():
        return float("nan")
    try:
        pchip = PchipInterpolator(anchor_idx.astype(float), x[anchor_idx], extrapolate=False)
    except Exception:  # noqa: BLE001 -- theirs: NaN, never fail
        return float("nan")
    val = float(pchip(float(new_peak_frame)))
    return val if np.isfinite(val) else float("nan")


def parabolic_excise(y, x, is_interp_fill, is_baseline_fill, outlier_peak_frame, n_frames, fps, *, params):
    """Quadratic fit to raw anchors in +-parabolic_tight_radius (outlier window excised); if concave-down with
    the vertex in the window, the vertex replaces the peak. Else None (caller falls back to spline)."""
    tight_radius = max(2, int(round(float(params["parabolic_tight_radius_ms"]) / 1000.0 * fps)))
    min_anchors = int(params["parabolic_min_anchors"])
    excise_radius = max(1, int(round(float(params["spline_excise_radius_ms"]) / 1000.0 * fps)))
    lo_out = max(0, outlier_peak_frame - excise_radius)
    hi_out = min(n_frames, outlier_peak_frame + excise_radius + 1)
    a_lo = max(0, outlier_peak_frame - tight_radius)
    a_hi = min(n_frames, outlier_peak_frame + tight_radius + 1)
    anchor_idx, _ = _anchor_idx(y, is_interp_fill, is_baseline_fill, lo_out, hi_out, a_lo, a_hi, n_frames)
    if len(anchor_idx) < min_anchors:
        return None
    if not ((anchor_idx < outlier_peak_frame).any() and (anchor_idx > outlier_peak_frame).any()):
        return None
    try:
        coef = np.polyfit(anchor_idx.astype(float), y[anchor_idx], 2)
    except Exception:  # noqa: BLE001
        return None
    a, b, c = float(coef[0]), float(coef[1]), float(coef[2])
    if a >= 0:
        return None
    vx = -b / (2.0 * a)
    if not (a_lo <= vx <= a_hi - 1):
        return None
    new_peak_frame = int(round(vx))
    eval_x = np.arange(lo_out, hi_out, dtype=float)
    return dict(
        new_peak_y=float(a * vx * vx + b * vx + c),
        new_peak_x=_compute_imputed_x(x, is_interp_fill, is_baseline_fill, outlier_peak_frame=outlier_peak_frame,
                                      new_peak_frame=new_peak_frame, n_frames=n_frames, fps=fps, params=params),
        new_peak_frame=new_peak_frame, interp_y=a * eval_x ** 2 + b * eval_x + c,
        outlier_lo=lo_out, outlier_hi=hi_out, method="parabolic")


def spline_excise(y, x, is_interp_fill, is_baseline_fill, outlier_peak_frame, n_frames, fps, *, params):
    """PCHIP through raw anchors in +-spline_anchor_win (outlier excised); new peak = max over
    +-spline_local_max_win of the PCHIP and of surviving raw y. None if too few anchors."""
    excise_radius = max(1, int(round(float(params["spline_excise_radius_ms"]) / 1000.0 * fps)))
    anchor_radius = max(4, int(round(float(params["spline_anchor_win_ms"]) / 1000.0 * fps)))
    local_max_radius = max(2, int(round(float(params["spline_local_max_win_ms"]) / 1000.0 * fps)))
    min_anchors = int(params["spline_min_anchors"])
    lo_out = max(0, outlier_peak_frame - excise_radius)
    hi_out = min(n_frames, outlier_peak_frame + excise_radius + 1)
    a_lo = max(0, outlier_peak_frame - anchor_radius)
    a_hi = min(n_frames, outlier_peak_frame + anchor_radius + 1)
    anchor_idx, raw_mask = _anchor_idx(y, is_interp_fill, is_baseline_fill, lo_out, hi_out, a_lo, a_hi, n_frames)
    if len(anchor_idx) < min_anchors:
        return None
    if not ((anchor_idx < outlier_peak_frame).any() and (anchor_idx > outlier_peak_frame).any()):
        return None
    try:
        pchip = PchipInterpolator(anchor_idx.astype(float), y[anchor_idx], extrapolate=False)
    except Exception:  # noqa: BLE001
        return None
    interp_y = pchip(np.arange(lo_out, hi_out, dtype=float))
    lm_lo = max(0, outlier_peak_frame - local_max_radius)
    lm_hi = min(n_frames, outlier_peak_frame + local_max_radius + 1)
    cleaned_lm = pchip(np.arange(lm_lo, lm_hi, dtype=float))
    valid = np.isfinite(cleaned_lm)
    in_kept = np.array([raw_mask[idx] and not (lo_out <= idx < hi_out) for idx in range(lm_lo, lm_hi)])
    all_vals = np.full(lm_hi - lm_lo, -np.inf)
    all_vals[valid] = cleaned_lm[valid]
    for j_local, idx in enumerate(range(lm_lo, lm_hi)):
        if in_kept[j_local]:
            all_vals[j_local] = max(all_vals[j_local], y[idx])
    if not np.any(np.isfinite(all_vals)):
        return None
    new_peak_frame = lm_lo + int(np.argmax(all_vals))
    return dict(
        new_peak_y=float(np.max(all_vals)),
        new_peak_x=_compute_imputed_x(x, is_interp_fill, is_baseline_fill, outlier_peak_frame=outlier_peak_frame,
                                      new_peak_frame=new_peak_frame, n_frames=n_frames, fps=fps, params=params),
        new_peak_frame=new_peak_frame, interp_y=interp_y, outlier_lo=lo_out, outlier_hi=hi_out)


def _peak_confidence(peak_frame, likelihood, is_interp_fill, is_baseline_fill, n_frames, conf_radius):
    """#raw frames in +-conf_radius x their mean likelihood (0.0 if none)."""
    lo = max(0, peak_frame - conf_radius)
    hi = min(n_frames, peak_frame + conf_radius + 1)
    raw_in_win = (~is_interp_fill[lo:hi]) & (~is_baseline_fill[lo:hi])
    n_raw = int(raw_in_win.sum())
    if n_raw == 0:
        return 0.0
    return n_raw * float(np.mean(likelihood[lo:hi][raw_in_win]))


def run_m_gate(decisions, y, likelihood, is_interp_fill, is_baseline_fill, n_frames, fps, promoted, *, params):
    """Min inter-lick interval: of two consecutive KEPT peaks closer than the threshold, drop the lower-
    confidence one (higher y breaks ties); repeat until none. Threshold relaxes to m_early_interval_ms when
    both peaks are inside the first m_early_win_ms (their Rule 5: keep close-paired first licks). Mutates
    ``decisions``; returns the (possibly re-chosen) promoted peak. ``y`` is unused, as in theirs."""
    m_min = float(params["m_min_interval_ms"])
    m_early_int = float(params["m_early_interval_ms"])
    m_early_win = float(params["m_early_win_ms"])
    conf_radius = max(1, int(round(float(params["m_conf_radius_ms"]) / 1000.0 * fps)))

    def _thr(t1, t2):
        return m_early_int if (t1 < m_early_win and t2 < m_early_win) else m_min

    while True:
        kept = sorted([d for d in decisions if d["keep"]], key=lambda d: d["t_ms"])
        merged = False
        for i in range(len(kept) - 1):
            d1, d2 = kept[i], kept[i + 1]
            threshold = _thr(d1["t_ms"], d2["t_ms"])
            if d2["t_ms"] - d1["t_ms"] < threshold:
                c1 = _peak_confidence(d1["peak_frame_local"], likelihood, is_interp_fill, is_baseline_fill,
                                      n_frames, conf_radius)
                c2 = _peak_confidence(d2["peak_frame_local"], likelihood, is_interp_fill, is_baseline_fill,
                                      n_frames, conf_radius)
                if c1 < c2:
                    loser = d1
                elif c2 < c1:
                    loser = d2
                else:
                    loser = d1 if d1["y"] < d2["y"] else d2
                loser["keep"] = False
                loser["reason"] = (f"M_close interval={d2['t_ms'] - d1['t_ms']:.0f}ms thr={threshold:.0f}ms "
                                   f"c1={c1:.2f} c2={c2:.2f} drop_{'d1' if loser is d1 else 'd2'}")
                loser["gate"] = "M"
                merged = True
                if promoted is not None and loser["t_ms"] == promoted.get("t_ms"):
                    surviving = sorted([d for d in decisions if d["keep"]], key=lambda d: d["t_ms"])
                    promoted = surviving[0] if surviving else None
                break
        if not merged:
            break
    return promoted
