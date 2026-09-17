"""Guards for the across-epoch rest-baseline drift measurement.

The three things that would make this measurement wrong without failing loudly:

  1. the per-session normalisation stops removing cross-day GAIN, so the result reports window
     brightness instead of baseline drift;
  2. the null stops being exchangeable, so an epoch drawn from pre comes out significant;
  3. the cosine gets read against zero, where its construction bias lives.

Each has a test here. They are pure -- no share mount, no session data -- which is why the
statistic and its null were split out of `run`.
"""
from __future__ import annotations

import numpy as np
import pytest

from scripts.rest_migration.rest_baseline_epoch_drift import (
    _stat, epoch_row, normalise,
)


# ----------------------------------------------------------------- the scale-invariance claim
def test_normalise_is_invariant_to_a_per_session_gain():
    """THE DESIGN'S CENTRAL CLAIM. Expression, bleaching and window clarity act as a multiplicative
    constant on a session, and NO SUBTRACTION REMOVES A GAIN TERM. If this fails the measurement is
    reporting how bright the window was."""
    rng = np.random.default_rng(0)
    b, e = rng.normal(size=50), rng.normal(size=50)
    ref = normalise(b, e)
    for gain in (0.01, 0.5, 3.0, 250.0):
        got = normalise(b * gain, e * gain)
        assert np.allclose(got[0], ref[0])
        assert np.allclose(got[1], ref[1])


def test_normalise_refuses_a_session_with_no_signal():
    """A zero evoked norm is a divide-by-zero, and a session that reaches the statistic with an
    undefined baseline is worse than a session that drops out."""
    assert normalise(np.ones(5), np.zeros(5)) is None
    assert normalise(np.ones(5), np.array([np.nan] * 5)) is None


def test_normalise_puts_the_evoked_map_on_the_unit_sphere():
    """`b/k` is then literally 'the baseline in units of this session's signal', which is what lets
    ||d|| be read as a fraction without a further division."""
    rng = np.random.default_rng(1)
    _b, e = normalise(rng.normal(size=30), rng.normal(size=30))
    assert np.linalg.norm(e) == pytest.approx(1.0)


# ------------------------------------------------------------------------ the statistic itself
def test_identical_groups_give_exactly_zero_shift():
    bn = np.tile(np.arange(8.0), (6, 1))
    nd, _cos = _stat(bn, np.arange(3), np.arange(3, 6), np.ones(8))
    assert nd == pytest.approx(0.0)


def test_the_shift_is_the_norm_of_the_difference_of_group_means():
    """Closed form, so a refactor that quietly changes the estimand fails here rather than in a
    figure."""
    bn = np.array([[1.0, 0.0], [3.0, 0.0], [0.0, 4.0], [0.0, 8.0]])
    nd, _ = _stat(bn, [0, 1], [2, 3], np.array([1.0, 0.0]))
    # mean(a) = (2, 0); mean(b) = (0, 6); d = (2, -6)
    assert nd == pytest.approx(np.hypot(2.0, 6.0))


def test_the_shift_does_not_depend_on_group_order():
    """It is a norm, so swapping the arms must not move it -- the property that keeps this out of
    the trap `retained` fell into, where the two directions had different denominators."""
    rng = np.random.default_rng(3)
    bn = rng.normal(size=(9, 12))
    e = rng.normal(size=12)
    assert _stat(bn, [0, 1, 2], [3, 4, 5], e)[0] == pytest.approx(
        _stat(bn, [3, 4, 5], [0, 1, 2], e)[0])


# ---------------------------------------------------------------------------- the null is sound
def test_an_epoch_drawn_from_pre_is_not_significant():
    """THE POSITIVE CONTROL. Exchangeable data must not produce a small p -- if it does, the null
    is not rebuilding the observed statistic the same way and every number this script prints is
    an artefact of the construction rather than a measurement."""
    rng = np.random.default_rng(4)
    bn = rng.normal(size=(20, 40))
    e_ref = rng.normal(size=40)
    # The "epoch" is just four more exchangeable sessions.
    ps = [epoch_row(bn, e_ref, np.arange(4), np.arange(4, 20), 400,
                    np.random.default_rng(100 + i))["p"] for i in range(12)]
    assert np.mean([p < 0.05 for p in ps]) < 0.3, ps
    assert np.median(ps) > 0.15, ps


def test_a_planted_shift_is_detected():
    """The negative control the positive one is worthless without: move the epoch's baselines along
    a fixed direction and the test must fire."""
    rng = np.random.default_rng(5)
    bn = rng.normal(size=(20, 40)) * 0.2
    bump = np.zeros(40)
    bump[:10] = 3.0
    bn[:4] += bump
    r = epoch_row(bn, rng.normal(size=40), np.arange(4), np.arange(4, 20), 400,
                  np.random.default_rng(7))
    assert r["p"] < 0.01
    assert r["shift_frac_of_evoked"] > r["null_p95"]


def test_the_cosine_is_biased_positive_on_exchangeable_data():
    """THE REASON THE COSINE CARRIES NO p. `d = mean_E(bn) - mean_pre(bn)` and
    `e_ref = mean_pre(post/k) - mean_pre(bn)` share `-mean_pre(bn)` with the same sign, so the raw
    cosine runs positive on data with NO drift at all. Reproduce the construction exactly and the
    bias must show up -- this is the measurement that says the number is not readable against
    zero."""
    rng = np.random.default_rng(6)
    n, k = 24, 60
    cos = []
    for t in range(200):
        r_ = np.random.default_rng(500 + t)
        bn = r_.normal(size=(n, k)) * 0.5
        pre = np.arange(4, n)
        e_ref = r_.normal(size=k) * 0.5 - bn[pre].mean(axis=0)   # the construction `run` uses
        cos.append(epoch_row(bn, e_ref, np.arange(4), pre, 5, r_)["cos_with_evoked_biased"])
    # Exchangeable data, no drift planted: an UNBIASED cosine would centre on zero. The MEAN is the
    # test -- the sign fraction is a weaker statement and would need far more draws to pin tightly.
    assert np.mean(cos) > 0.05, np.mean(cos)
    assert np.mean([c > 0 for c in cos]) > 0.55, np.mean([c > 0 for c in cos])


def test_no_cosine_p_is_reported():
    """A GUARD AGAINST IT COMING BACK. Two nulls were tried and neither reproduces the bias (any
    group resampled within pre has the shared term cancel), so a p here would be
    anti-conservative. If a future change reinstates one, this fails and sends the reader to the
    algebra in `epoch_row` before they trust it."""
    rng = np.random.default_rng(12)
    r = epoch_row(rng.normal(size=(20, 30)), rng.normal(size=30),
                  np.arange(4), np.arange(4, 20), 100, np.random.default_rng(13))
    assert "cos_p" not in r
    assert "cos_null_median" not in r
    assert "cos_with_evoked_biased" in r


def test_a_degenerate_pre_group_does_not_crash():
    """One pre session left over after the split leaves `gb` tiny but non-empty; two epochs the
    size of pre leave it empty and those draws are skipped rather than dividing by zero."""
    rng = np.random.default_rng(8)
    bn = rng.normal(size=(6, 10))
    r = epoch_row(bn, rng.normal(size=10), np.arange(2), np.arange(2, 6), 50,
                  np.random.default_rng(9))
    assert r["p"] is not None
    assert 0.0 <= r["p"] <= 1.0
