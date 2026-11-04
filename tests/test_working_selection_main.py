"""The rebuild's camera working set follows main's setting (48f2cc6d, `working_trials`): WIDEFIELD_WORKING_TRIALS =
in_working (production: contact | incomplete, jaw_only its own class) | in_working_jaw (the jaw-inclusive robustness
arm) | quit_tail (= the DAQ gate here); main's tag part "-cls-contact-incomplete"; the static "working" gate and
`in_working` stay jaw-inclusive exactly as on main."""
from __future__ import annotations

import numpy as np
import pytest

from wfield_local import trial_classes as TC
from wfield_local import working_trials as WT

NR, C, J, I, S = "no_response", "contact", "jaw_only", "incomplete", "stopped"


@pytest.fixture
def camera(monkeypatch):
    monkeypatch.setenv(TC.GATE_ENV, "camera")
    monkeypatch.setenv(TC.WINDOW_ENV, "cue_stop")
    for k in ("WIDEFIELD_TRIAL_END", "WIDEFIELD_TRIAL_STOPPED", "WIDEFIELD_STOP_MIN_START_S", "WIDEFIELD_REST_START",
              WT.WORKING_TRIALS_ENV):
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def test_static_gates_match_main():
    assert TC.GATES["working"] == frozenset({C, I, J}) == WT.WORKING_CLASSES[WT.IN_WORKING_JAW]
    assert TC.GATES["camera_lick"] == frozenset({C, I}) == WT.WORKING_CLASSES[WT.IN_WORKING]


def test_production_selection_is_contact_incomplete(camera):
    cls = [C, I, J, NR]
    assert WT.selection() == WT.IN_WORKING
    assert TC.in_sel(cls, "working").tolist() == [True, True, False, False]
    assert TC.in_sel(cls, "miss_working").tolist() == [False, True, False, False]
    assert TC.camera_select("working", np.array(cls, object), np.array(cls, object),
                            np.ones(4, bool), np.ones(4, bool))[1].tolist() == [True, True, False, False]


def test_jaw_arm(camera):
    camera.setenv(WT.WORKING_TRIALS_ENV, "in_working_jaw")
    assert TC.in_sel([C, I, J, NR], "working").tolist() == [True, True, True, False]
    assert TC.in_sel([C, I, J, NR], "miss_working").tolist() == [False, True, True, False]


def test_tags_carry_mains_class_part(camera):
    assert TC.run_selection_tag() == "sel-camera-cue_stop-endlick-cls-contact-incomplete-stopRun6j"
    assert TC.run_selection() == TC.run_selection_tag() == TC.nolick_selection()
    # main's working_trials.tag on this branch gives main's strings exactly
    assert WT.tag() == "sel-camera-cue_stop-endlick-cls-contact-incomplete"
    camera.setenv(WT.WORKING_TRIALS_ENV, "in_working_jaw")
    assert TC.run_selection_tag() == "sel-camera-cue_stop-endlick-stopRun6"      # the 11-03 afternoon rebuild tag
    assert WT.tag() == "sel-camera-cue_stop-endlick"


def test_quit_tail_under_the_camera_gate_raises(camera):
    camera.setenv(WT.WORKING_TRIALS_ENV, "quit_tail")
    with pytest.raises(TC.DaqGateRefused):
        TC.in_sel([C], "working")
    camera.setenv(TC.GATE_ENV, "daq")
    assert TC.run_selection_tag() is None                                         # the DAQ rule: untagged


def test_jaw_counting_follows_the_selection(camera):
    raw = np.array([C, NR, NR, NR, J, NR, NR, NR, C], object)
    # production (Priya 11-04): jaw_only is a run member anywhere -- it counts toward the 6 and is stopped
    assert TC.jaw_counts_in_stop() and TC.stopped_rule_name() == "stopRun6j"
    assert TC.classify_session(raw, rule="run", min_run=6, end="lick").tolist() == [C] + [S] * 7 + [C]
    # in_working_jaw: the afternoon rule -- a jaw_only trial before the last lick is working and ENDS the run
    camera.setenv(WT.WORKING_TRIALS_ENV, "in_working_jaw")
    assert not TC.jaw_counts_in_stop() and TC.stopped_rule_name() == "stopRun6"
    assert S not in TC.classify_session(raw, rule="run", min_run=6, end="lick").tolist()
    # ... and after the last lick it still bridges without counting (T3)
    tail = np.array([C, NR, NR, NR, J, NR, NR, NR], object)
    assert TC.classify_session(tail, rule="run", min_run=6, end="lick").tolist() == [C, S, S, S, "after_end", S, S, S]
    tail5 = np.array([C, NR, NR, NR, J, NR, NR], object)
    assert S not in TC.classify_session(tail5, rule="run", min_run=6, end="lick").tolist()


def test_5c_cache_key_carries_the_selection(camera, monkeypatch):
    from wfield_local import grant_confusion as gc
    seen = []
    monkeypatch.setattr(gc, "_collect_5c_cached", lambda a, v, m, run: seen.append(run) or ({}, []))
    gc._collect_5c("cue", "working", "frozen")
    camera.setenv(WT.WORKING_TRIALS_ENV, "in_working_jaw")
    gc._collect_5c("cue", "working", "frozen")
    assert seen[0] != seen[1] and seen[0][-1] == WT.IN_WORKING and seen[1][-1] == WT.IN_WORKING_JAW
