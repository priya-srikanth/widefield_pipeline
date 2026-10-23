"""Stray-lick test: overlapping lick kernels are recovered by the regression, and licks are categorised by context,
contact and nearest calibrated spout centre."""
import numpy as np
import pandas as pd
from scipy import sparse

from wfield_local import stray_licks as sl


def test_overlapping_kernels_recovered():
    rng = np.random.default_rng(0)
    n, lags = 6000, np.arange(-2, 10)
    kA, kB = np.sin(np.linspace(0, 3, len(lags)))[:, None], -np.cos(np.linspace(0, 2, len(lags)))[:, None]
    fa = np.sort(rng.choice(np.arange(20, n - 20), 300, replace=False))
    fb = fa + rng.integers(2, 6, len(fa))                    # B events follow A within a kernel's span (overlap)
    XA, XB = sl._lagged(fa, n, lags), sl._lagged(fb, n, lags)
    D = np.column_stack([np.ones(n), np.linspace(-1, 1, n)])
    Y = XA @ kA + XB @ kB + 0.3 * rng.normal(size=(n, 1)) + D @ np.array([[0.5], [0.2]])
    beta = sl.fit_kernels(sparse.hstack([XA, XB]).tocsr(), D, Y, ridge_frac=1e-4)
    assert np.allclose(beta[:len(lags)], kA, atol=0.08) and np.allclose(beta[len(lags):], kB, atol=0.08)


def test_classify_categories():
    cen = {"far_contra": np.array([1.0, 0.0, 0.0]), "far_mid": np.array([0.0, 1.0, 0.0])}
    L = pd.DataFrame({"side_ml": [1.0, 0.1, 0.9, 0.0, 0.05, np.nan], "out_ap": [0.0, 0.9, 0.1, 1.0, 0.95, 0.0],
                      "down_dv": [0.0] * 6,
                      "context": ["post_cue", "post_cue", "post_cue", "iti", "pre_cue", "post_cue"],
                      "spout": ["far_contra", "far_contra", "far_contra", "far_contra", "far_mid", "far_mid"],
                      "contact": [True, False, False, False, False, False]})
    cat = sl.classify(L, cen)
    assert list(cat) == ["contact|far_contra", "stray|far_contra|far_mid", "near|far_contra", "iti|far_mid", "near|far_mid", ""]
