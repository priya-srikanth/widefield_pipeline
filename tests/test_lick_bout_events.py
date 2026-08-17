"""Tests for bout-onset extraction and, above all, for the position ATTRIBUTION.

The bug being prevented is subtle and was measured on real data before this module existed: a lick
after the spout has moved but before the next cue was being given the PREVIOUS trial's position.
Blocks of ~6 same-position trials hide it most of the time, which is exactly why it needs a test
built on a BLOCK TRANSITION rather than on random positions.
"""
from __future__ import annotations

import numpy as np
import pytest

from wfield_local import lick_bout_events as lb

FS = 1000.0
RW = 3.5


def _session(strobe_t, cue_t, lick_t, codes):
    """Everything in seconds; converted to samples at FS."""
    return dict(cue_samples=np.array(cue_t) * FS, strobe_samples=np.array(strobe_t) * FS,
                lick_samples=np.array(lick_t) * FS, codes=np.array(codes),
                sample_rate_hz=FS, response_window_s=RW)


def _run(**kw):
    return lb.bout_onsets_with_position(kw["cue_samples"], kw["strobe_samples"], kw["lick_samples"],
                                        kw["codes"], kw["sample_rate_hz"], kw["response_window_s"],
                                        max_ili_s=0.3, min_bout_licks=2)


def test_lick_after_spout_move_gets_the_NEW_position_not_the_previous_one():
    """THE bug. Trial 0 is position 3, trial 1 is position 5; the spout moves between them.

    A bout starting 1 s after the second strobe -- i.e. before the second cue -- is at position 5.
    Attributing it by the most recent CUE would call it position 3.
    """
    s = _session(strobe_t=[0.0, 10.0], cue_t=[3.0, 13.0],
                 lick_t=[11.0, 11.1, 11.2], codes=[3, 5])
    ev = _run(**s)
    assert len(ev["code"]) == 1
    assert ev["code"][0] == 5, "must take the position the spout had MOVED to"
    assert ev["phase"][0] == "approach"
    assert ev["t_from_cue_s"][0] == pytest.approx(-2.0, abs=1e-6)


def test_post_cue_bout_is_labelled_response_and_keeps_its_trial():
    s = _session(strobe_t=[0.0], cue_t=[3.0], lick_t=[3.4, 3.5, 3.6], codes=[2])
    ev = _run(**s)
    assert ev["code"][0] == 2 and ev["phase"][0] == "response"


def test_nothing_past_the_response_window():
    """The spout is already moving there, so the bout belongs to no trial cleanly."""
    s = _session(strobe_t=[0.0], cue_t=[3.0], lick_t=[7.0, 7.1, 7.2], codes=[2])
    assert len(_run(**s)["code"]) == 0


def test_bout_exactly_at_the_window_edge_is_kept():
    s = _session(strobe_t=[0.0], cue_t=[3.0], lick_t=[6.5, 6.6], codes=[2])
    ev = _run(**s)
    assert len(ev["code"]) == 1 and ev["t_from_cue_s"][0] == pytest.approx(RW, abs=1e-6)


def test_licks_before_any_strobe_are_dropped():
    """Before the first spout move there is no position, so there is nothing to label."""
    s = _session(strobe_t=[5.0], cue_t=[8.0], lick_t=[1.0, 1.1, 1.2], codes=[4])
    assert len(_run(**s)["code"]) == 0


def test_only_bout_ONSETS_are_returned_not_every_lick():
    """Ten licks at 100 ms spacing are ONE bout, not ten observations."""
    s = _session(strobe_t=[0.0], cue_t=[3.0],
                 lick_t=list(np.arange(3.2, 4.2, 0.1)), codes=[1])
    ev = _run(**s)
    assert len(ev["code"]) == 1
    assert ev["n_licks"][0] >= 9


def test_a_gap_longer_than_max_ili_splits_into_two_bouts():
    s = _session(strobe_t=[0.0], cue_t=[3.0],
                 lick_t=[3.1, 3.2, 4.0, 4.1], codes=[1])
    ev = _run(**s)
    assert len(ev["code"]) == 2 and set(ev["phase"]) == {"response"}


def test_singleton_licks_are_dropped_by_min_bout_licks():
    s = _session(strobe_t=[0.0], cue_t=[3.0], lick_t=[3.2], codes=[1])
    assert len(_run(**s)["code"]) == 0


def test_unlabelled_position_is_excluded():
    s = _session(strobe_t=[0.0], cue_t=[3.0], lick_t=[3.2, 3.3], codes=[-1])
    assert len(_run(**s)["code"]) == 0


def test_within_a_block_the_old_cue_rule_and_the_strobe_rule_agree():
    """KNOWN-GOOD case: same position either side, so attribution cannot differ.

    Without this the test suite would only ever exercise the failing case, and a rule that broke the
    common path would pass.
    """
    s = _session(strobe_t=[0.0, 10.0], cue_t=[3.0, 13.0],
                 lick_t=[11.0, 11.1], codes=[4, 4])
    ev = _run(**s)
    assert ev["code"][0] == 4


def test_summarize_reports_what_the_extra_events_buy():
    s = _session(strobe_t=[0.0, 10.0], cue_t=[3.0, 13.0],
                 lick_t=[3.2, 3.3, 11.0, 11.1, 13.2, 13.3], codes=[1, 2])
    out = lb.summarize(_run(**s))
    assert out["n_bouts"] == 3 and out["n_trials_represented"] == 2
    assert out["by_phase"]["approach"] == 1 and out["by_phase"]["response"] == 2
    assert out["bouts_per_trial"] == pytest.approx(1.5)


def test_empty_input_does_not_raise():
    s = _session(strobe_t=[0.0], cue_t=[3.0], lick_t=[], codes=[1])
    assert lb.summarize(_run(**s))["n_bouts"] == 0
