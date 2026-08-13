"""Tests for the searched lick-free pre-cue window.

The search recovers trials that a fixed cue-ending window would discard, so the tests target the ways
that recovery could go wrong and quietly produce a better-looking but invalid result: straying before
the spout strobe (where prior-trial activity predicts the upcoming position), overlapping a lick, or
silently returning a window of the wrong length.
"""
from __future__ import annotations

import numpy as np

from wfield_local.precue_lickfree import lickfree_window

WN = 62          # ~2 s at 31.23 Hz


def test_uses_the_window_ending_at_the_cue_when_it_is_already_clean():
    w0 = lickfree_window(cue_f=1000, strobe_f=500, licks=np.array([600.0]), win_n=WN)
    assert w0 == 1000 - WN


def test_recovers_a_trial_with_a_late_lick_by_stepping_back():
    """THE case Priya raised: a lick 200 ms before the cue must not cost the whole trial when 2 s of
    clean data sits just earlier."""
    late = 1000 - 6                                   # ~200 ms before the cue at 31.23 Hz
    w0 = lickfree_window(cue_f=1000, strobe_f=500, licks=np.array([float(late)]), win_n=WN)
    assert w0 is not None
    assert w0 + WN <= late, "window must END at or before the lick"
    assert w0 >= 500, "window must not start before the spout strobe"


def test_never_returns_a_window_starting_before_the_strobe():
    """Before the strobe the position does not exist yet, and prior-trial activity predicts it
    (the task avoids recent repeats), so such a window would manufacture a pre-cue code."""
    for strobe in (900, 950, 970):
        w0 = lickfree_window(cue_f=1000, strobe_f=strobe, licks=np.array([]), win_n=WN)
        if w0 is not None:
            assert w0 >= strobe


def test_returns_none_when_the_interval_is_shorter_than_the_window():
    assert lickfree_window(cue_f=1000, strobe_f=980, licks=np.array([]), win_n=WN) is None


def test_returns_none_when_licking_leaves_no_clean_gap():
    licks = np.arange(500, 1000, 10, dtype=float)     # a lick every ~0.3 s, no 2 s gap anywhere
    assert lickfree_window(cue_f=1000, strobe_f=500, licks=licks, win_n=WN) is None


def test_chosen_window_contains_no_licks_and_is_full_length():
    rng = np.random.default_rng(0)
    for _ in range(200):
        strobe = 0
        cue = int(rng.integers(WN + 5, 900))
        licks = np.sort(rng.choice(np.arange(strobe, cue), size=int(rng.integers(0, 12)),
                                   replace=False).astype(float)) if cue > strobe + 1 else np.array([])
        w0 = lickfree_window(cue, strobe, licks, WN)
        if w0 is None:
            continue
        assert w0 >= strobe and w0 + WN <= cue
        assert not ((licks >= w0) & (licks < w0 + WN)).any(), "chosen window contains a lick"


def test_prefers_the_latest_eligible_gap():
    """Closest to the cue is the most informative about the upcoming action, so an early gap must not
    win when a later one also fits."""
    licks = np.array([300.0, 700.0])                  # gaps: [0,300), (300,700), (700,1000)
    w0 = lickfree_window(cue_f=1000, strobe_f=0, licks=licks, win_n=WN)
    assert w0 == 1000 - WN                            # the last gap is wide enough -> use it
