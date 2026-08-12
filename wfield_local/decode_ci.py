"""Bootstrap confidence intervals for per-position decoding accuracy, per session.

Purpose: make a single session's accuracy a MEASUREMENT rather than a point, so a post-stroke session
can be judged against the pre-stroke reference band instead of against one number. Works identically
for a within-session decoder and for a frozen/leave-one-session-out one -- it resamples the SESSION's
trials with the model held fixed.

WHY A CLUSTER BOOTSTRAP. Trials are not independent: the task presents positions in runs of ~6
(``blocks``), and the whole pipeline already respects that with GroupKFold. Resampling individual
trials would treat ~6 correlated trials as 6 independent observations and produce intervals that are
far too narrow -- the classic consequence of ignoring clustering. So we resample BLOCKS with
replacement, which preserves the within-block dependence and gives honest coverage. `by="trial"` is
available for comparison and will visibly disagree.

TWO DIFFERENT UNCERTAINTIES, do not conflate them:
  * this function -> "how precisely do we know THIS session's accuracy", model fixed;
  * the spread of LOSO accuracies ACROSS sessions -> "how much does an unseen day vary".
The post-stroke reference band needs the second; the first says whether one session is measured
well enough to place against it.

    from wfield_local.decode_ci import bootstrap_recall
    ci = bootstrap_recall(y_true, y_pred, blocks)      # -> overall + per-position CIs
"""
from __future__ import annotations

import numpy as np

from wfield_local.plot_lick_aligned_averages import DISPLAY_ORDER, POSITION_NAMES

N_BOOT = 2000
ALPHA = 0.05


def _recall(y_true, y_pred, positions):
    """Per-position recall + overall accuracy for one (possibly resampled) trial set."""
    out = np.full(len(positions), np.nan)
    for i, p in enumerate(positions):
        m = y_true == p
        if m.any():
            out[i] = float((y_pred[m] == p).mean())
    acc = float((y_pred == y_true).mean()) if len(y_true) else np.nan
    return out, acc


def bootstrap_recall(y_true, y_pred, blocks=None, n_boot=N_BOOT, alpha=ALPHA, seed=0,
                     by="block", positions=None):
    """Percentile bootstrap CIs for overall accuracy and per-position recall.

    ``blocks`` = per-trial block id (required for the default cluster bootstrap). Returns a dict with
    point estimates, CI bounds, and ``n_effective`` -- the number of resampling units, which is the
    honest sample size and is much smaller than the trial count.
    """
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    positions = np.asarray(positions if positions is not None else DISPLAY_ORDER)
    rng = np.random.default_rng(seed)
    point, acc0 = _recall(y_true, y_pred, positions)

    if by == "block":
        if blocks is None:
            raise ValueError("blocks are required for the cluster bootstrap (by='block')")
        blocks = np.asarray(blocks)
        units = np.unique(blocks)
        idx_of = {u: np.flatnonzero(blocks == u) for u in units}
    else:
        units = np.arange(len(y_true))
        idx_of = None

    boot_r = np.full((n_boot, len(positions)), np.nan)
    boot_a = np.full(n_boot, np.nan)
    for b in range(n_boot):
        pick = rng.choice(units, size=len(units), replace=True)
        sel = (np.concatenate([idx_of[u] for u in pick]) if idx_of is not None
               else pick)
        if sel.size == 0:
            continue
        boot_r[b], boot_a[b] = _recall(y_true[sel], y_pred[sel], positions)

    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    with np.errstate(invalid="ignore"):
        r_lo = np.nanpercentile(boot_r, lo, axis=0)
        r_hi = np.nanpercentile(boot_r, hi, axis=0)
    return {
        "by": by, "n_boot": int(n_boot), "alpha": alpha,
        "n_trials": len(y_true), "n_effective": len(units),
        "positions": [POSITION_NAMES[int(p)] for p in positions],
        "accuracy": acc0,
        "accuracy_ci": [float(np.nanpercentile(boot_a, lo)), float(np.nanpercentile(boot_a, hi))],
        "recall": [float(v) for v in point],
        "recall_ci_lo": [float(v) for v in r_lo],
        "recall_ci_hi": [float(v) for v in r_hi],
        "chance": 1.0 / len(positions),
    }


def reference_band(per_session_ci, key="accuracy"):
    """Pre-stroke reference band from several sessions' bootstrap results.

    Combines the two uncertainties: the ACROSS-session spread (min/max and 10-90th percentile of the
    point estimates -- how much an unseen day varies) and the widest within-session CI (how precisely
    any one day is measured). A post-stroke session is evidence of impairment when its own CI falls
    ENTIRELY BELOW the band, which requires both that it is low and that it is measured well enough
    to say so.
    """
    pts = np.array([c[key] for c in per_session_ci], float)
    los = np.array([c[f"{key}_ci"][0] for c in per_session_ci], float)
    his = np.array([c[f"{key}_ci"][1] for c in per_session_ci], float)
    return {"n_sessions": len(pts), "median": float(np.median(pts)),
            "min": float(pts.min()), "max": float(pts.max()),
            "p10": float(np.percentile(pts, 10)), "p90": float(np.percentile(pts, 90)),
            "widest_ci": [float(los.min()), float(his.max())]}
