"""ROLLING masked-median detrend -- the smooth version of `filter_acausality_test.detrend_masked`.

THE DEFECT IT FIXES. `detrend_masked` walks NON-OVERLAPPING windows, takes one median per window at
its centre, and joins those centres with `np.interp` -- LINEAR interpolation. The resulting trend is
piecewise-linear with a knot every `win_s`, so at 300 s over a 100 min session it has ~20 knots and
can only bend at them. That is visible as kinks in the drift-estimator overlay
(`plot_drift_estimators`), and a trend with corners injects structure of its own into whatever is
detrended with it.

THE FIX is to evaluate the SAME masked median on a DENSE grid of window centres (default every 10 s)
instead of one per window. The window still sets the kinetics -- this changes the realisation, not the
timescale -- but with centres ~30x denser the interpolation artefact disappears.

WHAT IS DELIBERATELY UNCHANGED: it is still a MEDIAN over masked samples, not a convolution. So it
keeps the property that made the windowed family safe in the first place -- no impulse response, so it
cannot displace signal backwards in time the way the zero-phase Butterworth does. Making this smooth
must not quietly make it a filter.

EDGES: windows are truncated at the record ends, exactly as the non-overlapping version's first and
last windows are, and `np.interp` clamps beyond the outermost usable centre.
"""
from __future__ import annotations

import numpy as np

from wfield_local.hemo_variants import FS

DEFAULT_STRIDE_S = 10.0


def rolling_masked_trend(X, mask, win_s, stride_s=DEFAULT_STRIDE_S, min_frac=0.05, min_n=20):
    """``(trend, coverage)`` for ``X`` (K, T) -- the rolling masked-median trend.

    ``win_s`` is the window that sets the kinetics (unchanged from `detrend_masked`); ``stride_s`` is
    only how finely it is evaluated, and does NOT change the timescale.
    """
    X = np.asarray(X, dtype=np.float64)
    K, T = X.shape
    m = np.asarray(mask, bool)
    m = m[:T] if m.size >= T else np.pad(m, (0, T - m.size))
    w = max(1, int(round(win_s * FS)))
    half = w // 2
    stride = max(1, int(round(stride_s * FS)))

    cs, vs = [], []
    for c in range(0, T, stride):
        a, b = max(0, c - half), min(T, c + half)
        mm = m[a:b]
        need = max(min_n, min_frac * (b - a))
        if mm.sum() >= need:
            cs.append(float(c))
            vs.append(np.median(X[:, a:b][:, mm], axis=1))
    if len(cs) < 2:
        return np.zeros_like(X), 0.0
    C = np.asarray(cs, float)
    V = np.stack(vs, 1)
    t = np.arange(T, dtype=float)
    trend = np.stack([np.interp(t, C, V[k]) for k in range(K)])
    return trend, len(cs) / max(1, int(np.ceil(T / stride)))


def rolling_detrend(X, mask, win_s, stride_s=DEFAULT_STRIDE_S):
    """``X`` minus its rolling masked-median trend."""
    trend, _cov = rolling_masked_trend(X, mask, win_s, stride_s)
    return np.asarray(X, dtype=np.float64) - trend
