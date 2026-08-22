"""The ITI gate on the post-lick position maps.

WHY. Without it every lick in a session carries the position of the most recent CUE, however long
after it falls, and cue-to-cue is at least 8 s. PS94 8/17 far_center and far_R had ZERO responses yet
contributed 93 and 83 licks to their maps at a median of 7.2-7.6 s after the cue -- by which time the
next spout had already moved into place. The contamination is graded by severity (18% pre-stroke at
every position against 28/47/50/100/100% post-stroke), so it tracks the deficit it would be read as
evidence for.
"""
from __future__ import annotations

import numpy as np
import pytest

from wfield_local.framemap_event_maps import in_trial_mask


def test_an_iti_lick_is_dropped_and_a_response_is_kept():
    cues = np.array([0.0, 1000.0, 2000.0])
    ends = np.array([350.0, 1350.0, 2350.0])          # trial_end ~ end of the response window
    licks = np.array([50.0,        # response to cue 0            -> keep
                      340.0,       # consumption, still in trial 0 -> keep
                      700.0,       # ITI after trial 0             -> drop
                      1100.0,      # response to cue 1             -> keep
                      1900.0])     # ITI after trial 1             -> drop
    assert in_trial_mask(licks, cues, ends).tolist() == [True, True, False, True, False]


def test_reward_consumption_is_kept_not_dropped():
    """trial_end lands AFTER the response window, so a fast hit still contributes its bout.

    Gating at the response time itself would throw away the consumption licking, which is part of
    the trial and is most of the post-lick signal.
    """
    cues = np.array([0.0])
    ends = np.array([3500.0])
    licks = np.array([200.0, 900.0, 1800.0, 3400.0, 3600.0])
    assert in_trial_mask(licks, cues, ends).tolist() == [True, True, True, True, False]


def test_a_lick_before_the_first_cue_is_dropped():
    """It belongs to no trial, so it cannot be attributed to any position."""
    assert in_trial_mask(np.array([10.0]), np.array([100.0]), np.array([400.0])).tolist() == [False]


def test_the_last_trial_is_bounded_by_its_own_end_not_left_open():
    """A session ends with a trial whose end exists; nothing after it may leak in."""
    got = in_trial_mask(np.array([50.0, 500.0]), np.array([0.0]), np.array([400.0]))
    assert got.tolist() == [True, False]


def test_a_trial_with_no_end_pulse_keeps_nothing_after_the_next_cue():
    """If trial_end is missing for the LAST trial, that trial is unbounded (inf) by construction --
    which is the safe direction only because a later cue would re-assign the lick anyway."""
    got = in_trial_mask(np.array([50.0, 5000.0]), np.array([0.0]), np.array([]))
    assert got.all(), "with no end pulses at all the gate must be a no-op, not drop everything"


@pytest.mark.parametrize("n_end", [0, 1])
def test_an_unusable_trial_end_line_leaves_the_licks_alone(n_end):
    """Refusing to gate is right when the signal is absent; silently dropping every lick is not."""
    licks = np.arange(0.0, 500.0, 50.0)
    got = in_trial_mask(licks, np.array([0.0]), np.array([300.0])[:n_end])
    assert got.all() if n_end == 0 else got.sum() == 6
