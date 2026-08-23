"""The lateral-axis measures, and the reading each of the three numbers licenses.

WHY THESE EXIST. The first version of this analysis reported "the lateral axis re-forms" on the
strength of a low cosine with the pre-stroke axis alone. That does not support it: a fitted direction
through trials carrying NO lateral information scores near zero too. The verdict logic below is what
keeps the three questions apart -- is it different, does it still exist, is it the lesion -- so it is
the thing worth pinning.
"""
from __future__ import annotations

import numpy as np
import pytest

from wfield_local import lateral_axis as la


def _band(p5, p95):
    return {"mean": (p5 + p95) / 2, "p5": p5, "p95": p95}


def test_an_axis_that_does_not_reproduce_itself_is_not_a_finding():
    """PS92's close MISS cell: cos +0.318 against a floor of 0.47-0.79 looks like a change, but the
    axis reproduces itself at +0.078 -- there is no lateral code there to have changed."""
    v = la.verdict({"n": [42, 53], "cos_with_prestroke": 0.318,
                    "noise_floor": _band(0.474, 0.792),
                    "own_reproducibility": _band(-0.261, 0.416)})
    assert "DOES NOT REPRODUCE" in v and "no lateral code" in v


def test_below_the_floor_with_a_reproducible_axis_is_the_strong_result():
    """PS92's far MISS cell: near-orthogonal to the pre-stroke axis (+0.151) AND reproducing at
    +0.781. A stable different axis -- the only case where "re-formed" is earned."""
    v = la.verdict({"n": [132, 355], "cos_with_prestroke": 0.151,
                    "noise_floor": _band(0.585, 0.814),
                    "own_reproducibility": _band(0.654, 0.849)})
    assert "BELOW the floor" in v and "stable DIFFERENT axis" in v


def test_within_the_floor_is_reported_as_no_detected_change():
    """A cosine has to beat what two halves of clean pre-stroke data reach at the same n."""
    v = la.verdict({"n": [300, 300], "cos_with_prestroke": 0.90,
                    "noise_floor": _band(0.88, 0.95),
                    "own_reproducibility": _band(0.82, 0.91)})
    assert "WITHIN the floor" in v and "no detected change" in v


def test_too_few_trials_says_so_rather_than_returning_a_number():
    assert "not fitted" in la.verdict({"n": [5, 5], "too_few": True})


def test_split_half_refuses_when_the_halves_would_overlap():
    """Disjoint halves or nothing. Overlapping halves inflate the floor toward 1, which would make
    every real comparison look like a change."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 12))
    assert la.split_half(X, X, None, 20, 20, rng) is None          # 2*20 > 30
    assert la.split_half(X, X, None, 10, 10, rng) is not None


def test_split_half_separates_a_real_axis_from_noise():
    """Sanity on the instrument itself: if reproducibility did not separate structure from noise,
    none of the verdicts would mean anything."""
    rng = np.random.default_rng(1)
    n, d = 200, 40
    w = rng.normal(size=d)
    w /= np.linalg.norm(w)
    L = rng.normal(size=(n, d)) + 3.0 * w
    R = rng.normal(size=(n, d)) - 3.0 * w
    strong = la.split_half(L, R, None, 80, 80, rng)
    noise = la.split_half(rng.normal(size=(n, d)), rng.normal(size=(n, d)), None, 80, 80, rng)
    assert strong.mean() > 0.8, strong.mean()
    assert abs(noise.mean()) < 0.25, noise.mean()


@pytest.mark.parametrize("ring,pair", sorted(la.RINGS.items()))
def test_every_ring_is_a_pure_lateral_contrast(ring, pair):
    """Within a ring, so no close-vs-far component can enter. The pooled version was imbalanced by
    up to -0.17 post-stroke, because misses concentrate at the far positions."""
    a, b = pair
    assert a.startswith(ring) and b.startswith(ring)
    assert a.endswith("_L") and b.endswith("_R")
