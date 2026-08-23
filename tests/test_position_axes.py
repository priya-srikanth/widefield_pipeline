"""The pairwise position-axis measures, and the reading each number licenses.

WHY THESE EXIST. This analysis reported "the lateral axis re-forms" on the strength of a low cosine
with the pre-stroke axis alone, which does not support it -- a direction fitted through trials
carrying NO position information scores near zero too. It then ordered the cohort by comparing each
cosine against the PRE-STROKE reliability alone, ignoring that the post-stroke axis is itself noisy,
which inflated PS94 relative to PS95. `verdict()` and the disattenuation are what keep those two
mistakes from recurring, so they are what is pinned here.
"""
from __future__ import annotations

import numpy as np
import pytest

from wfield_local import position_axes as pa


def _cell(cos, r_pre, r_post, n=(200, 200)):
    d = None
    if r_pre and r_post and r_pre > 0 and r_post > 0:
        d = cos / float(np.sqrt(r_pre * r_post))
    return {"n": list(n), "cos": cos, "r_pre": r_pre, "r_post": r_post, "disattenuated": d}


def test_an_axis_that_does_not_reproduce_itself_is_not_a_finding():
    """PS92's close MISS cell: cos +0.318 looks like a change until you notice the axis reproduces
    itself at +0.078 -- there is no position code there to have changed."""
    assert "NO AXIS" in pa.verdict(_cell(0.318, 0.64, 0.078))


def test_the_ceiling_accounts_for_BOTH_reliabilities():
    """An axis cannot resemble another more than it resembles itself. Judging a cosine against the
    pre-stroke reliability alone made PS94 (0.756 against a 0.952 floor) look clearly worse than
    PS95; disattenuated they are 0.87 and 0.84, i.e. indistinguishable."""
    ps94 = _cell(0.756, 0.952, 0.800)
    ps95 = _cell(0.684, 0.813, 0.810)
    assert ps94["disattenuated"] == pytest.approx(0.866, abs=0.01)
    assert ps95["disattenuated"] == pytest.approx(0.843, abs=0.01)
    assert abs(ps94["disattenuated"] - ps95["disattenuated"]) < 0.05, "must not order these apart"


def test_a_large_real_change_still_reads_as_one():
    """PS93 close: disattenuated 0.37 with a reliability of 0.83 -- a stable, genuinely different
    axis, and the one cell that survives the correction as an outlier."""
    v = pa.verdict(_cell(0.313, 0.845, 0.831))
    assert "CHANGED" in v and "stable new axis" in v


def test_an_unchanged_axis_is_not_reported_as_a_change():
    assert "UNCHANGED" in pa.verdict(_cell(0.88, 0.95, 0.95))


def test_too_few_trials_says_so_rather_than_returning_a_number():
    assert "not fitted" in pa.verdict({"n": [5, 5], "too_few": True})


def test_split_half_refuses_overlapping_halves():
    """Overlapping halves share trials, dragging reliability toward 1 and making every comparison
    look like a change."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 12))
    assert pa.split_half(X, X, None, 20, 20, rng) is None
    assert pa.split_half(X, X, None, 10, 10, rng) is not None


def test_split_half_separates_a_real_axis_from_noise():
    """Sanity on the instrument: if reliability did not separate structure from noise, none of the
    verdicts would mean anything."""
    rng = np.random.default_rng(1)
    n, d = 200, 40
    w = rng.normal(size=d)
    w /= np.linalg.norm(w)
    strong = pa.split_half(rng.normal(size=(n, d)) + 3.0 * w,
                           rng.normal(size=(n, d)) - 3.0 * w, None, 80, 80, rng)
    noise = pa.split_half(rng.normal(size=(n, d)), rng.normal(size=(n, d)), None, 80, 80, rng)
    assert strong > 0.8, strong
    assert abs(noise) < 0.25, noise


def test_every_position_pair_is_covered_and_typed():
    """15 unordered pairs, each classified by WHAT IT VARIES -- the thing six position labels hide."""
    from collections import Counter
    assert len(pa.PAIRS) == 15
    counts = Counter(pa.pair_type(a, b) for a, b in pa.PAIRS)
    assert counts == {"diagonal": 6, "lateral-centre": 4, "distance": 3, "lateral": 2}


@pytest.mark.parametrize("a,b,want", [
    ("close_L", "close_R", "lateral"),        # same ring, L vs R
    ("far_L", "far_R", "lateral"),
    ("close_center", "close_L", "lateral-centre"),
    ("far_L", "close_L", "distance"),         # same side, different ring
    ("far_R", "close_L", "diagonal"),         # differs in both
])
def test_pair_types_match_the_task_geometry(a, b, want):
    assert pa.pair_type(a, b) == want


def test_decompose_recovers_a_reference_axis_it_was_built_from():
    """Sanity: an axis that IS one of the references must show cos 1 with it and no residual."""
    rng = np.random.default_rng(3)
    refs = {}
    for k in ("a", "b", "c"):
        v = rng.normal(size=50)
        refs[k] = v / np.linalg.norm(v)
    got = pa.decompose(refs["b"], refs)
    assert got["cos"]["b"] == pytest.approx(1.0, abs=1e-6)
    assert got["residual_outside_span"] == pytest.approx(0.0, abs=1e-6)


def test_decompose_reports_a_full_residual_for_genuinely_new_structure():
    """THE POINT OF THE MEASURE. An axis orthogonal to every pre-stroke reference has moved into
    structure that did not previously exist -- a far stronger claim than "it changed", and one the
    cosines alone cannot express."""
    d = 60
    eye = np.eye(d)
    refs = {f"r{i}": eye[i] for i in range(5)}
    got = pa.decompose(eye[40], refs)                 # orthogonal to every reference
    assert max(abs(v) for v in got["cos"].values()) < 1e-9
    assert got["residual_outside_span"] == pytest.approx(1.0, abs=1e-6)


def test_decompose_does_not_let_overlapping_references_sum_past_one():
    """References are NOT orthogonal -- position axes share structure -- so the cosines overlap and
    must never be added up. The residual is the well-defined quantity and stays in [0, 1]."""
    rng = np.random.default_rng(4)
    base = rng.normal(size=40)
    base /= np.linalg.norm(base)
    tilt = rng.normal(size=40) * 0.1
    refs = {"one": base, "two": (base + tilt) / np.linalg.norm(base + tilt)}
    got = pa.decompose(base, refs)
    assert sum(abs(v) for v in got["cos"].values()) > 1.0      # cosines DO oversum
    assert 0.0 <= got["residual_outside_span"] <= 1.0          # the residual does not
