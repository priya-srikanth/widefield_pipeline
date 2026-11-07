"""11-06: under the camera gate the working period is a SPAN (first .. last camera lick), so `engaged_frame_mask` must
put the quit boundary after the LAST engaged cue, not at the first non-engaged one (which was cue 0 whenever the first
cue was not a camera lick: 8 / 107 sessions kept only their pre-first-cue frames as "engaged" rest)."""
from __future__ import annotations

import numpy as np

from wfield_local.rest_engagement import frame_gate_bounds

CS = np.array([100, 200, 300, 400, 500])
LO, HI = np.iinfo(np.int64).min, np.iinfo(np.int64).max


def test_daq_suffix_is_unchanged():
    # DAQ gate: non-engaged cues are a terminal suffix -> quit at the first of them, start open
    assert frame_gate_bounds(CS, [True, True, True, False, False], camera=False) == (LO, 400)


def test_camera_span_starting_at_cue0_is_unchanged():
    assert frame_gate_bounds(CS, [True, True, True, False, False], camera=True) == (LO, 400)


def test_camera_span_with_early_non_lick_cue():
    # the bug: first cue not a camera lick -> the old rule returned quit = 100 (everything after cue 0 quit)
    assert frame_gate_bounds(CS, [False, True, True, True, False], camera=True) == (200, 500)


def test_camera_span_to_the_last_cue_has_no_quit():
    assert frame_gate_bounds(CS, [False, True, True, True, True], camera=True) == (200, HI)


def test_mid_session_gap_does_not_cut_the_span():
    # camera_engaged_period is a contiguous span, but a gap must never move the quit boundary forward
    assert frame_gate_bounds(CS, [True, False, True, True, False], camera=True) == (LO, 500)
