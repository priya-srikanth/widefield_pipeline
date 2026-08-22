"""The would-be-lick reference for no-lick trials at lick alignment.

WHY THIS FILE EXISTS. The `nolick_ref` parameter was added and then silently did nothing: the
signature patch landed in the module that IMPORTS `features_with_indices` rather than the one that
defines it, so the keyword was never threaded through. The module imported, ruff passed, and all 464
tests passed -- the failure only appeared as a TypeError at runtime, because no test exercised the
new parameter. These tests pin the behaviour rather than the plumbing.
"""
from __future__ import annotations

import numpy as np

from wfield_local.locanmf_position_decoder import would_be_lick_offsets


def test_offset_is_per_position_not_a_single_number():
    """Latency differs by position -- far positions are slower, and post-stroke far_R is 10x."""
    codes = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    rt = np.array([10, 10, 10, 12, 12, 12, 60, 60, 60, 64, 64, 64], float)
    per, overall, _n = would_be_lick_offsets(codes, rt, np.ones(12, bool))
    assert per[0] == 11.0 and per[1] == 62.0
    assert overall == 36.0                      # session median, reported but no longer USED
    assert per[0] != per[1], "a single session-wide offset would misplace every far trial"


def test_a_thin_position_keeps_its_own_median_and_is_flagged():
    """Three trials is not a median -- but it is the right ORDER OF MAGNITUDE, and the session
    median is not.

    This test used to assert the opposite. The old rule sent a thin position to the SESSION median,
    and that fallback fires precisely where the animal has stopped licking while the session median
    is set by the positions that still work: PS94's far_R no-lick windows were placed at 0.17-0.23 s
    when its own successful licks there took 1.80-2.25 s, i.e. outside the 2 s window entirely.
    Being off by a factor of two beats being off by 2 s.
    """
    codes = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1])
    rt = np.array([10, 10, 10, 10, 10, 10, 99, 99, 99], float)
    per, overall, n = would_be_lick_offsets(codes, rt, np.ones(9, bool), min_trials=5)
    assert per[1] == 99.0, "a thin position must not inherit the session median"
    assert n[1] == 3, "the count is returned so a cell resting on 3 trials can say so"
    assert overall is not None


def test_only_engaged_trials_define_the_offset():
    """No-lick trials have no RT to contribute; including them would drag the estimate."""
    codes = np.array([0, 0, 0, 0, 0, 0])
    rt = np.array([10, 10, 10, 10, 10, 10], float)
    engaged = np.array([True] * 5 + [False])
    rt_with_junk = rt.copy()
    rt_with_junk[-1] = -1                        # a no-lick trial's rt is meaningless
    per, _, _n = would_be_lick_offsets(codes, rt_with_junk, engaged)
    assert per[0] == 10.0


def test_no_engaged_trials_yields_nothing_rather_than_a_guess():
    """The caller drops those trials; inventing an offset would place windows arbitrarily."""
    per, overall, n = would_be_lick_offsets(np.array([0, 0]), np.array([5.0, 5.0]),
                                            np.zeros(2, bool))
    assert per == {} and overall is None and n == {}


def test_a_position_with_no_engaged_trial_gets_no_offset_at_all():
    """The case that motivated the change: the animal never licked there, so nothing in this
    session says when it would have. The caller drops those trials rather than borrowing a latency
    from the positions that still work."""
    codes = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    rt = np.array([5, 5, 5, 5, 5, -1, -1, -1, -1, -1], float)
    engaged = np.array([True] * 5 + [False] * 5)
    per, overall, n = would_be_lick_offsets(codes, rt, engaged)
    assert 1 not in per, "position 1 has no engaged trial -- it must not inherit `overall`"
    assert n[1] == 0
    assert overall == 5.0


def test_the_parameter_is_actually_threaded_through():
    """The regression that motivated this file: nolick_ref must reach _trial_features.

    Checked by signature rather than by running the pipeline, because the failure mode was a keyword
    that existed in one module and not in the one that forwards to it.
    """
    import inspect

    from wfield_local.locanmf_position_decoder import _trial_features
    from wfield_local.precue_engagement_states import features_with_indices

    assert "nolick_ref" in inspect.signature(_trial_features).parameters
    assert "nolick_ref" in inspect.signature(features_with_indices).parameters
    src = inspect.getsource(features_with_indices)
    assert "nolick_ref=nolick_ref" in src, "the wrapper must FORWARD it, not just accept it"
