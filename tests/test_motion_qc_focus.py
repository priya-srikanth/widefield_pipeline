"""Motion-QC sharpness: separate the alignment gain from the interpolation cost.

Priya, 2026-08-23: "why is the sharpness so often WORSE after motion correction?"

Measured over 80 sessions: 78% scored below 1 on focus_ratio (corrected/raw), median 0.786. It is
not damage. Correction does two things to a high-frequency metric -- removes motion blur (gain) and
resamples every frame at non-integer offsets (cost, because bilinear interpolation is a low-pass
filter and variance-of-Laplacian counts exactly what it removes). The ratio measures gain MINUS
cost, so at sub-pixel motion -- most sessions -- only the cost is left.

The evidence for that reading: focus_ratio correlates +0.70 with MEDIAN shift and -0.01 with MAX
shift, and the top motion quartile averages 1.13 against 0.69-0.84 for the rest. PS93_0606, with
8.69 px of median motion, comes out 2.52x sharper.

focus_gain divides the cost out by passing the raw mean through the same interpolator once, at the
session's own typical FRACTIONAL offset.
"""
import numpy as np
import pytest
from scipy import ndimage

from wfield_local.qc_motion_correction import _focus, _interp_cost


@pytest.fixture
def img():
    rng = np.random.default_rng(0)
    return ndimage.gaussian_filter(rng.random((160, 160)) * 1000, 2)


def test_a_zero_shift_costs_nothing(img):
    """The bug in the first attempt at this baseline. Bilinear interpolation at an integer offset
    has weights 1 and 0 -- it is the identity -- so a zero-shift baseline measures NOTHING and the
    'gain' it produces is just the old ratio wearing a different name."""
    assert _interp_cost(img, 0.0, 0.0) == pytest.approx(_focus(img), rel=1e-9)


def test_a_fractional_shift_really_does_cost_sharpness(img):
    base = _focus(img)
    assert _interp_cost(img, 0.5, 0.5) < 0.95 * base
    assert _interp_cost(img, 0.25, 0.25) < base


def test_the_cost_grows_with_the_fractional_offset(img):
    """Monotone up to the half-pixel worst case, which is where interpolation averages two pixels
    equally."""
    costs = [_interp_cost(img, d, d) for d in (0.0, 0.1, 0.25, 0.5)]
    assert costs == sorted(costs, reverse=True), costs


def test_dividing_the_cost_out_lifts_a_perfectly_aligned_session_to_one(img):
    """A session with NO motion: corrected is the raw put through the interpolator, nothing more.
    The raw ratio calls that damage; the gain calls it what it is."""
    corrected = ndimage.shift(img, (0.3, 0.3), order=1, mode="nearest")
    raw_ratio = _focus(corrected) / _focus(img)
    gain = _focus(corrected) / _interp_cost(img, 0.3, 0.3)
    assert raw_ratio < 0.95, "the un-corrected comparison reads as a loss"
    assert gain == pytest.approx(1.0, rel=1e-9), "with the cost divided out it is exactly neutral"


def test_real_motion_still_shows_a_gain(img):
    """The metric must not become blind: blur the 'raw' as motion would, and the gain must exceed 1
    once the blur is removed."""
    raw_blurred = ndimage.gaussian_filter(img, 1.5)          # what motion does to a mean image
    corrected = ndimage.shift(img, (0.3, 0.3), order=1, mode="nearest")
    gain = _focus(corrected) / _interp_cost(raw_blurred, 0.3, 0.3)
    assert gain > 1.5, gain
