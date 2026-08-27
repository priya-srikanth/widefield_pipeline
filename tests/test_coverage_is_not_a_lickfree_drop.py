"""A cue outside the imaging coverage is not a licking failure, and must not be counted as one.

MEASURED on PS95_0813 (Priya, 2026-08-26), the session that exposed it:

    total cues                                        871
    outside the imaging coverage (`_frames` marks -1)  197
    genuine lick-free drops                             1
    kept (587 engaged + 86 no-lick)                   673

`_frames` already excluded the 197 by setting their frame to -1. But `_trial_features` then let them
fall through to `precue_window_start`, which computes `fixed = -1 - win_n < 0`, returns None, and
lands the trial on `n_dropped_dirty`. So the log announced "dropped 198 trial(s) with no lick-free 2s
window" -- 23% of the session -- when the real lick-free exclusion rate was 1/674 = 0.15%, in line
with PS95_0812 and PS95_0814, which report none at all.

THE KEPT SET NEVER CHANGED, and that is why this went unnoticed for two weeks: both paths drop the
same trial, so every number computed from the features was correct. What was wrong was the reason
given, on a PRE-STROKE session that feeds the reference band -- a label asserting a cause that
nothing checks, pointing an exclusion at the animal's behaviour when the cause was a gap in the
imaging. The same shape as the frozen models trained on post-stroke data and the "curated" dates that
meant pre-stroke only by historical accident.
"""
import inspect

import numpy as np

from wfield_local import locanmf_position_decoder as D


def test_a_negative_cue_frame_makes_precue_window_start_return_none():
    """The MECHANISM. This is what routed coverage drops into the lick-free counter."""
    licks = np.array([10.0, 200.0, 900.0])
    assert D.precue_window_start(-1, np.nan, licks, 60, lickfree=True) is None
    assert D.precue_window_start(-1, np.nan, licks, 60, lickfree=False) is None
    # ...and a normal trial well clear of any lick still keeps its fixed window, so the guard above
    # is not simply rejecting everything.
    assert D.precue_window_start(500, np.nan, licks, 60, lickfree=True) == 440


def test_coverage_is_dropped_before_the_lickfree_accounting():
    """ORDER IS THE FIX. The guard has to precede the `precue_window_start` call; behind it, the
    trial is still dropped but still misattributed, and the test would pass on a broken build."""
    src = inspect.getsource(D._trial_features)
    guard = src.index("n_dropped_coverage += 1")
    lickfree = src.index("n_dropped_dirty += 1")
    assert guard < lickfree, "the coverage guard must come first, or the drop is misattributed again"


def test_the_two_counters_are_separate():
    """One `continue` per trial, two counters. Sharing one would restore the ambiguity the fix
    removes -- the point is being able to answer "how many of these are behavioural?"."""
    src = inspect.getsource(D._trial_features)
    assert src.count("n_dropped_coverage = 0") == 1
    assert src.count("n_dropped_dirty = 0") == 1
    assert "n_dropped_coverage" in src and "n_dropped_dirty" in src


def test_the_coverage_message_says_it_is_excluded_from_the_lickfree_count():
    """A reader sees both lines together; without this the natural reading is that they add up."""
    src = inspect.getsource(D._trial_features)
    i = src.index("n_dropped_coverage:")
    assert "NOT counted below" in src[i:i + 500]
