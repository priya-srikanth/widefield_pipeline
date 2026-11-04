"""Priya 11-04 stop rule A2 "both sides" (PRIMARY) and the robustness arms A0 ("run") / A3 ("long")
(trial_classes._classify_guarded; reproduces the census script srg/r2.py exactly, REBUILD_CODE 8.43)."""
import numpy as np
import pytest

from wfield_local import trial_classes as TC

C, I, J, NR, U, S = "contact", "incomplete", "jaw_only", "no_response", "unknown", "stopped"
REF, FAR = "close_L", "far_R"


def a2(cls, pos, t=None, rule="both", end="lick", jaw=True):
    t = np.arange(len(cls)) * 10.0 + 400.0 if t is None else np.asarray(t, float)
    t[0] = 0.0
    return TC.classify_session(np.array(cls, object), np.ones(len(cls), bool), np.array(pos, object), rule=rule,
                               min_run=6, end=end, cue_s=t, jaw_counts=jaw).tolist()


def test_reference_free_run_between_two_reference_contacts_stays():
    cls = [C] + [NR, J, NR, NR, NR, NR] + [C, I]
    pos = [REF] + [FAR] * 6 + [REF, FAR]
    assert a2(cls, pos) == cls                                   # far-spout deficit, not a stop
    assert a2(cls, pos, rule="run")[1:7] == [S] * 6              # A0 stops it


@pytest.mark.parametrize("before,after", [(I, C), (C, I), (NR, C)])
def test_reference_free_run_is_stopped_unless_both_neighbours_are_contacts(before, after):
    cls = [C, before] + [NR] * 6 + [after, C, I]
    pos = [FAR, REF] + [FAR] * 6 + [REF, FAR, FAR]
    assert a2(cls, pos)[2:8] == [S] * 6


def test_no_reference_neighbour_on_one_side_is_stopped():
    cls = [C] + [NR] * 6 + [C]
    pos = [FAR] + [FAR] * 6 + [REF]                              # nothing at a reference spout before the run
    assert a2(cls, pos)[1:7] == [S] * 6


def test_run_with_a_reference_trial_is_stopped():
    cls = [C] + [NR] * 6 + [C]
    pos = [REF] + [FAR, FAR, REF, FAR, FAR, FAR] + [REF]
    assert a2(cls, pos)[1:7] == [S] * 6


def test_everything_after_the_last_lick_is_stopped_and_unknown_stays():
    cls = [C, I, NR, J, U, NR]
    pos = [REF, FAR, FAR, REF, FAR, FAR]
    assert a2(cls, pos) == [C, I, S, S, U, S]
    assert a2([NR, NR, U], [REF, REF, REF]) == [S, S, U]         # no lick at all: everything quit


def test_short_runs_and_unknown_gaps():
    cls = [C] + [NR] * 5 + [C]
    pos = [FAR] * 7
    assert a2(cls, pos) == cls                                   # 5 < 6: isolated lapses
    cls = [C, NR, NR, U, U, NR, NR, NR, NR, C]                   # unknown neither breaks nor extends
    pos = [FAR] * 10
    assert a2(cls, pos) == [C, S, S, U, U, S, S, S, S, C]


def test_early_run_folds_into_its_raw_class():
    cls = [C] + [NR, J, NR, NR, NR, NR] + [C]
    pos = [FAR] * 8
    t = np.r_[0.0, 100.0 + np.arange(6) * 10.0, 400.0]           # the run starts 100 s in
    assert a2(cls, pos, t=t) == cls
    t = np.r_[0.0, 300.0 + np.arange(6) * 10.0, 400.0]           # at exactly 300 s: a stop
    assert a2(cls, pos, t=t)[1:7] == [S] * 6
    assert TC.EARLY_RUN_CLASS not in a2(cls, pos, t=np.r_[0.0, 1.0 + np.arange(7)])


def test_a3_needs_more_than_ten_and_has_no_guard():
    pos10 = [REF] + [FAR] * 10 + [REF]
    assert a2([C] + [NR] * 10 + [C], pos10, rule="long")[1:11] == [NR] * 10
    pos11 = [REF] + [FAR] * 11 + [REF]
    assert a2([C] + [NR] * 11 + [C], pos11, rule="long")[1:12] == [S] * 11     # both neighbours contacts: no guard
    assert a2([C] + [NR] * 11 + [C, NR], pos11 + [FAR], rule="long")[-1] == S


def test_in_working_jaw_jaw_only_ends_a_run():
    cls = [C, NR, NR, NR, J, NR, NR, NR, C]
    pos = [FAR] * 9
    assert a2(cls, pos, jaw=False) == cls
    assert a2(cls, pos, jaw=True)[1:8] == [S] * 7


def test_rule_names_and_default(monkeypatch):
    monkeypatch.delenv(TC.STOPPED_ENV, raising=False)
    monkeypatch.delenv("WIDEFIELD_STOP_MIN_START_S", raising=False)
    monkeypatch.delenv("WIDEFIELD_WORKING_TRIALS", raising=False)
    assert TC.stopped_rule() == "both" and TC.stopped_rule_name() == "stopA2jrt3500c70"
    for rule, name in (("run", "stopRun6jrt3500c70"), ("long", "stopA3n11jrt3500c70"), ("any", "stopAnyrt3500c70")):
        monkeypatch.setenv(TC.STOPPED_ENV, rule)
        assert TC.stopped_rule_name() == name
    monkeypatch.setenv(TC.STOPPED_ENV, "both")
    monkeypatch.setenv("WIDEFIELD_STOP_MIN_START_S", "600")
    assert TC.stopped_rule_name() == "stopA2jt600rt3500c70"


def test_profile_arms_name_the_robustness_rules():
    from wfield_local import render_profile as rp
    d = rp.load("cohort1_rebuild_20261103")
    assert d["env"]["WIDEFIELD_TRIAL_STOPPED"] == "both"
    assert d["notes"]["expected_run_tag"].endswith("-stopA2jrt3500c70")
    assert rp.load("cohort1_rebuild_20261103", arm="stopA0")["env"]["WIDEFIELD_TRIAL_STOPPED"] == "run"
    assert rp.load("cohort1_rebuild_20261103", arm="stopA3")["env"]["WIDEFIELD_TRIAL_STOPPED"] == "long"


def test_cue_classes_for_passes_positions_to_the_guard(monkeypatch):
    import pandas as pd
    cls = [C] + [NR] * 6 + [C, I]
    pos = [REF] + [FAR] * 6 + [REF, FAR]
    cue = np.r_[0.0, 400.0 + np.arange(8) * 10.0]
    T = pd.DataFrame({"trial_id": np.arange(9), "cue_daq": cue, "cls": cls, "position": pos})
    monkeypatch.setattr(TC, "session_trials", lambda lab, window=None: T.copy())
    monkeypatch.delenv(TC.STOPPED_ENV, raising=False)
    monkeypatch.delenv("WIDEFIELD_TRIAL_END", raising=False)
    TC._CUE_CLS.clear()
    assert TC.cue_classes_for("PS92_0901", cue).tolist() == cls
    TC._CUE_CLS.clear()


def test_breaks_hook_ends_a_run_and_is_off_by_default():
    # the hook for the pending "no licks in the stopped time" rule: a trial whose span holds a lick ends a run
    cls = [C] + [NR] * 7 + [C]
    pos = [FAR] * 9
    assert a2(cls, pos)[1:8] == [S] * 7
    b = np.zeros(9, bool)
    b[4] = True                                                  # 3 | broken | 3 -> two runs < 6
    t = np.arange(9) * 10.0 + 400.0
    t[0] = 0.0
    for rule in ("both", "run", "long"):
        out = TC.classify_session(np.array(cls, object), np.ones(9, bool), np.array(pos, object), rule=rule, min_run=6,
                                  end="lick", cue_s=t, breaks=b).tolist()
        assert S not in out[1:8], rule
    with pytest.raises(ValueError):
        TC.classify_session(np.array(cls, object), rule="both", end="lick", breaks=np.zeros(3, bool))


def test_licks_in_spans():
    got = TC.licks_in_spans([0.0, 10.0, 20.0, np.nan], [10.0, 20.0, 30.0, 5.0], [9.99, 20.0, 35.0, np.nan])
    assert got.tolist() == [True, False, True, False]
