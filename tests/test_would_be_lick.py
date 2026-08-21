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
    per, overall = would_be_lick_offsets(codes, rt, np.ones(12, bool))
    assert per[0] == 11.0 and per[1] == 62.0
    assert overall == 36.0                      # session median, used only as a fallback
    assert per[0] != per[1], "a single session-wide offset would misplace every far trial"


def test_a_position_with_too_few_engaged_trials_falls_back():
    """A median over three trials is not a median."""
    codes = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1])
    rt = np.array([10, 10, 10, 10, 10, 10, 99, 99, 99], float)
    per, overall = would_be_lick_offsets(codes, rt, np.ones(9, bool), min_trials=5)
    assert 0 in per and 1 not in per            # position 1 has 3 -> falls back to `overall`
    assert overall is not None


def test_only_engaged_trials_define_the_offset():
    """No-lick trials have no RT to contribute; including them would drag the estimate."""
    codes = np.array([0, 0, 0, 0, 0, 0])
    rt = np.array([10, 10, 10, 10, 10, 10], float)
    engaged = np.array([True] * 5 + [False])
    rt_with_junk = rt.copy()
    rt_with_junk[-1] = -1                        # a no-lick trial's rt is meaningless
    per, _ = would_be_lick_offsets(codes, rt_with_junk, engaged)
    assert per[0] == 10.0


def test_no_engaged_trials_yields_nothing_rather_than_a_guess():
    """The caller drops those trials; inventing an offset would place windows arbitrarily."""
    per, overall = would_be_lick_offsets(np.array([0, 0]), np.array([5.0, 5.0]),
                                         np.zeros(2, bool))
    assert per == {} and overall is None


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
