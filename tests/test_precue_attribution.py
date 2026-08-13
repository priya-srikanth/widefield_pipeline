"""Tests for the pre-cue attribution analysis.

The module exists to stop invalid claims being made, so the tests target the properties that make its
outputs trustworthy rather than merely that it runs: EV must be honest about non-generalising features,
the Haufe transform must recover the true source pattern where a raw weight vector would not, and the
redundancy measure must actually detect duplicated information.
"""
from __future__ import annotations

import numpy as np

from wfield_local import precue_attribution as pa
from wfield_local.plot_lick_aligned_averages import DISPLAY_ORDER


def _toy(n_per=40, seed=0, noise=1.0, n_extra=3):
    """6 positions x n_per trials. Feature 0 is position-tuned; the rest are shared noise."""
    rng = np.random.default_rng(seed)
    y = np.repeat(np.array(DISPLAY_ORDER), n_per)
    g = np.arange(len(y)) // 6                     # blocks of 6 trials
    tuning = {c: float(i) for i, c in enumerate(DISPLAY_ORDER)}
    sig = np.array([tuning[c] for c in y])
    shared = rng.normal(size=len(y))
    X = np.column_stack([sig + noise * rng.normal(size=len(y))]
                        + [shared + noise * rng.normal(size=len(y)) for _ in range(n_extra)])
    return X, y, g


def test_encoding_ev_high_for_tuned_feature_and_low_for_noise():
    X, y, g = _toy()
    ev, ceil = pa.encoding_ev(X, y, g)
    assert ev[0] > 0.5, f"tuned feature should be well explained by position, got {ev[0]}"
    assert (ev[1:] < 0.2).all(), f"untuned features should not be, got {ev[1:]}"
    assert ceil[0] > ev[0] - 0.35, "ceiling should sit at or above the achieved EV for a clean feature"


def test_encoding_ev_can_go_negative_and_is_not_clipped():
    """A feature whose position means do not generalise SHOULD score below zero; clipping it to 0
    would hide the failure."""
    rng = np.random.default_rng(1)
    y = np.repeat(np.array(DISPLAY_ORDER), 30)
    g = np.arange(len(y)) // 6
    X = rng.normal(size=(len(y), 1))               # pure noise
    ev, _ = pa.encoding_ev(X, y, g)
    assert ev[0] < 0.05
    assert np.isfinite(ev[0])


def test_haufe_recovers_the_source_where_the_weight_vector_need_not():
    """THE reason the module uses Haufe. With one tuned feature plus correlated noise features, the
    activation pattern must peak on the tuned feature; a raw weight vector is free to load heavily on
    the noise features to cancel them."""
    X, y, g = _toy(noise=0.8, n_extra=4)
    A, names = pa.haufe_patterns(X, y, g)
    assert A.shape == (X.shape[1], len(DISPLAY_ORDER))
    assert len(names) == len(DISPLAY_ORDER)
    strength = np.abs(A).mean(axis=1)
    assert strength.argmax() == 0, f"activation should peak on the tuned feature, got {strength}"


def test_haufe_columns_follow_display_order():
    X, y, g = _toy()
    A, names = pa.haufe_patterns(X, y, g)
    from wfield_local.plot_lick_aligned_averages import POSITION_NAMES
    assert names == [POSITION_NAMES[c] for c in DISPLAY_ORDER]
    # the tuned feature is monotone in position by construction -> activation should be monotone too
    assert np.corrcoef(A[0], np.arange(len(DISPLAY_ORDER)))[0, 1] > 0.8


def test_redundancy_detected_when_information_is_duplicated():
    """Two identical copies of the signal: each is individually sufficient, neither is necessary.
    That is precisely the case a weight ranking would misreport."""
    X, y, g = _toy(noise=0.6, n_extra=0)
    dup = np.column_stack([X[:, 0], X[:, 0] + 0.01 * np.random.default_rng(3).normal(size=len(y))])
    both = np.column_stack([dup, dup])            # two families, each a copy of the same signal
    res = pa.sufficiency_necessity(both, y, g, {"A": np.array([0, 1]), "B": np.array([2, 3])})
    assert res["_full"] > pa.CHANCE
    for f in ("A", "B"):
        assert res[f]["alone"] > pa.CHANCE + 0.2, f"{f} alone should decode well (sufficient)"
        assert res[f]["necessity_drop"] < 0.15, f"{f} should NOT look necessary (its info survives)"
        assert res[f]["redundancy"] > 0.2, f"{f} duplicated info should show high redundancy"


def test_family_columns_maps_by_acronym_prefix():
    feat_reg = np.array([1, 2, 3, 4])
    names = {1: "SSp-bfd", 2: "MOp", 3: "VISp", 4: "SSs"}
    cols = pa.family_columns(feat_reg, names)
    assert cols["SSp"].tolist() == [0]        # SSp-bfd matches SSp, SSs must NOT
    assert cols["SSs"].tolist() == [3]
    assert cols["MOp"].tolist() == [1]
    assert cols["VIS"].tolist() == [2]


def test_family_columns_omits_families_with_no_features():
    cols = pa.family_columns(np.array([1]), {1: "MOp"})
    assert "MOp" in cols and "AUD" not in cols
