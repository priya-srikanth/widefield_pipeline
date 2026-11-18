"""lick_balance_angle_lmm.bounded_index (Priya 11-17): THE contra / ipsi index (c - i) / (|c| + |i|) behind dec_idx and
resp_idx (docs/status/deck_swap_20261117/ITI_MATCHED_ANGLE_MARKERS.md section 5). Kept in the balance module (not
analysis_kit): it is the hemispheric-balance definition, used by lick_balance_angle_lmm and lick_iti_matched_markers."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from lick_balance_angle_lmm import bounded_index  # noqa: E402


def test_both_positive_equals_ratio_form():
    c = np.array([1.0, 2.0, 3.0, 0.5, 10.0])
    i = np.array([1.0, 1.0, 0.5, 2.0, 0.1])
    r = c / i
    np.testing.assert_allclose(bounded_index(c, i), (r - 1) / (r + 1), rtol=0, atol=1e-15)


def test_sign_cases():
    assert bounded_index(2.0, -1.0) == 1.0                    # c > 0 > i -> +1
    assert bounded_index(-1.0, 3.0) == -1.0                   # c < 0 < i -> -1
    assert bounded_index(-1.0, -3.0) == 0.5                   # both negative: (c - i) / (|c| + |i|) = 2 / 4
    assert bounded_index(-3.0, -1.0) == -0.5
    assert bounded_index(0.0, 2.0) == -1.0 and bounded_index(2.0, 0.0) == 1.0
    assert bounded_index(1.5, 1.5) == 0.0


def test_zero_zero_and_nan_give_nan_without_error():
    with np.errstate(all="raise"):                            # no warning / FloatingPointError escapes
        assert np.isnan(bounded_index(0.0, 0.0))
        assert np.isnan(bounded_index(np.nan, 1.0))
        assert np.isnan(bounded_index(1.0, np.nan))


def test_array_input_shape_and_bounds():
    rng = np.random.default_rng(0)
    c, i = rng.normal(size=(50, 3)), rng.normal(size=(50, 3))
    out = bounded_index(c, i)
    assert out.shape == (50, 3)
    assert np.all(np.abs(out) <= 1.0)
    np.testing.assert_array_equal(bounded_index(list(c[:, 0]), list(i[:, 0])), out[:, 0])
    np.testing.assert_array_equal(bounded_index(c, i), -bounded_index(i, c))   # antisymmetric
