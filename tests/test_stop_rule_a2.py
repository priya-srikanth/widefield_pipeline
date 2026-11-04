"""Priya 11-04 stop rule A2 "both sides" (PRIMARY) and the robustness arms A0 ("run") / A3 ("long")
(trial_classes._classify_guarded; reproduces the census script srg/r2.py exactly, REBUILD_CODE 8.43)."""
import numpy as np
import pytest

from wfield_local import trial_classes as TC

C, I, J, NR, U, S = "contact", "incomplete", "jaw_only", "no_response", "unknown", "stopped"
REF, FAR = "close_L", "far_R"


def a2(cls, pos, t=None, rule="both", end="lick", jaw=True, span=None, lick_rule="none", tail="lick_trial"):
    t = np.arange(len(cls)) * 10.0 + 400.0 if t is None else np.asarray(t, float)
    t[0] = 0.0
    return TC.classify_session(np.array(cls, object), np.ones(len(cls), bool), np.array(pos, object), rule=rule,
                               min_run=6, end=end, cue_s=t, jaw_counts=jaw, span_licks=span, lick_rule=lick_rule,
                               tail_start=tail).tolist()


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
    assert TC.stopped_rule() == "both" and TC.stopped_rule_name() == "stopA2jexrt3500c70"
    for rule, name in (("run", "stopRun6jrt3500c70"), ("long", "stopA3n11jexrt3500c70"), ("any", "stopAnyrt3500c70")):
        monkeypatch.setenv(TC.STOPPED_ENV, rule)
        assert TC.stopped_rule_name() == name
    monkeypatch.setenv(TC.STOPPED_ENV, "both")
    monkeypatch.setenv("WIDEFIELD_STOP_MIN_START_S", "600")
    assert TC.stopped_rule_name() == "stopA2jt600exrt3500c70"


def test_profile_arms_name_the_robustness_rules():
    from wfield_local import render_profile as rp
    d = rp.load("cohort1_rebuild_20261103")
    assert d["env"]["WIDEFIELD_TRIAL_STOPPED"] == "both"
    assert d["notes"]["expected_run_tag"].endswith("-stopA2jexrt3500c70")
    assert rp.load("cohort1_rebuild_20261103", arm="stopA0")["env"]["WIDEFIELD_TRIAL_STOPPED"] == "run"
    assert rp.load("cohort1_rebuild_20261103", arm="stopA3")["env"]["WIDEFIELD_TRIAL_STOPPED"] == "long"


def test_cue_classes_for_passes_positions_to_the_guard(monkeypatch):
    import pandas as pd
    cls = [C] + [NR] * 6 + [C, I]
    pos = [REF] + [FAR] * 6 + [REF, FAR]
    cue = np.r_[0.0, 400.0 + np.arange(8) * 10.0]
    T = pd.DataFrame({"trial_id": np.arange(9), "cue_daq": cue, "cls": cls, "position": pos, "strobe_ms": -2000.0,
                      "stop_ms": 3000.0})
    monkeypatch.setattr(TC, "session_trials", lambda lab, window=None: T.copy())
    monkeypatch.setattr(TC, "_daq_contact_onsets", lambda a, d: np.zeros(0))       # no licks in any span
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
                                  end="lick", cue_s=t, breaks=b, lick_rule="none").tolist()
        assert S not in out[1:8], rule
    with pytest.raises(ValueError):
        TC.classify_session(np.array(cls, object), rule="both", end="lick", breaks=np.zeros(3, bool), lick_rule="none")


def test_licks_in_spans():
    got = TC.licks_in_spans([0.0, 10.0, 20.0, np.nan], [10.0, 20.0, 30.0, 5.0], [9.99, 20.0, 35.0, np.nan])
    assert got.tolist() == [True, False, True, False]


# ------------------------------------------------------------------- stopped time holds no licks (Priya 11-04)
SL = "stopped_licking"


def _span(n, **cells):
    """span lick matrix with counts at (trial, category) = n_licks; categories as TC.SPAN_CATS."""
    M = np.zeros((n, len(TC.SPAN_CATS)), int)
    for key, v in cells.items():
        k, c = key[1:].split("_", 1)
        M[int(k), TC.SPAN_CATS.index(c)] = v
    return M


def test_span_lick_matrix_categories():
    cue = np.array([10.0, 20.0, 30.0])
    strobe, stop = cue - 2.0, cue + np.array([4.0, 3.0, 4.0])
    licks = [7.5, 8.5, 10.03, 10.5, 13.9, 14.5, 18.1, 30.5, 35.0]
    M = TC.span_lick_matrix(cue, strobe, stop, licks)
    assert M[0].tolist() == [1, 1, 1, 1, 1]             # precue 8.5, 0-70 10.03, resp 10.5, >max_rt 13.9, ITI 14.5
    assert M[1].tolist() == [1, 0, 0, 0, 0]             # 18.1
    assert M[2].tolist() == [0, 0, 1, 0, 0]             # 30.5; 35.0 after the LAST stop is unassigned; 7.5 too early
    M = TC.span_lick_matrix(cue, [np.nan] * 3, stop, [7.5, 17.5])
    assert M[0, 0] == 1 and M[1, 0] == 1                # a missing strobe = cue - 3 s


def test_touched_is_own_span_or_either_iti():
    M = _span(5, k1_precue=2, k3_iti_after=1)
    assert TC.touched_by_licks(M).tolist() == [False, True, False, True, True]


def test_excise_marks_touched_stopped_trials_and_keeps_the_rest():
    cls = [C] + [NR] * 7 + [C, NR, NR]
    pos = [FAR] * 11
    M = _span(11, k3_precue=4, k9_iti_after=1)
    out = a2(cls, pos, span=M, lick_rule="excise")
    assert out[1:8] == [S, S, SL, S, S, S, S]           # the run is NOT re-formed; only the touched trial moves
    assert out[9:] == [SL, SL]                          # quit tail: 9 (its own ITI), 10 (the ITI before it)
    assert out[0] == C and out[8] == C                  # lick trials untouched by the rule
    assert SL not in TC.STOPPED_CLASSES and not TC.in_gate([SL], "working").any()


def test_strict_break_reforms_the_runs():
    cls = [C] + [NR] * 7 + [C]
    pos = [FAR] * 9
    M = _span(9, k4_precue=1)                           # breaks before 4, and 4 leaves the run: 3 | 4 left -> < 6
    assert a2(cls, pos, span=M, lick_rule="strict")[1:8] == [NR] * 7
    cls = [C] + [NR] * 14 + [C]
    M = _span(16, k8_precue=1)
    out = a2(cls, [FAR] * 16, span=M, lick_rule="strict")
    assert out[1:8] == [S] * 7 and out[8] == NR and out[9:15] == [S] * 6


def test_tail_after_the_last_lick_event():
    cls = [C, NR, NR, NR, NR]
    pos = [FAR] * 5
    M = _span(5, k2_iti_after=1)                         # a lick in the ITI after trial 2: the tail starts at 3
    assert a2(cls, pos, span=M, lick_rule="none", tail="lick_event") == [C, NR, NR, S, S]
    assert a2(cls, pos, span=M, lick_rule="none") == [C, S, S, S, S]


def test_lick_rule_without_spans_raises_and_names_enter_the_tag(monkeypatch):
    with pytest.raises(ValueError, match="span_licks"):
        a2([C, NR], [FAR, FAR], lick_rule="excise")
    for k in (TC.STOPPED_ENV, TC.STOP_LICKS_ENV, TC.STOP_TAIL_ENV, "WIDEFIELD_STOP_MIN_START_S", "WIDEFIELD_WORKING_TRIALS"):
        monkeypatch.delenv(k, raising=False)
    assert TC.stop_lick_rule() == "excise" and TC.stop_tail_start() == "lick_trial"
    assert TC.stopped_rule_name() == "stopA2jexrt3500c70"
    monkeypatch.setenv(TC.STOP_LICKS_ENV, "strict")
    assert TC.stopped_rule_name() == "stopA2jsbrt3500c70"
    monkeypatch.setenv(TC.STOP_LICKS_ENV, "none")
    monkeypatch.setenv(TC.STOP_TAIL_ENV, "lick_event")
    assert TC.stopped_rule_name() == "stopA2jtert3500c70"
    monkeypatch.setenv(TC.STOPPED_ENV, "run")
    assert TC.stopped_rule_name() == "stopRun6jrt3500c70"         # A0 does not use the lick rule
    monkeypatch.setenv(TC.STOP_LICKS_ENV, "bogus")
    with pytest.raises(ValueError):
        TC.stop_lick_rule()


def test_session_span_licks_reads_licks_all_peaks_and_daq(tmp_path, monkeypatch):
    import pandas as pd
    sd = tmp_path / "PS92" / "20260806"
    sd.mkdir(parents=True)
    pd.DataFrame({"daq_peak_s": [8.5, 14.5]}).to_csv(sd / "licks_all.csv", index=False)
    monkeypatch.setattr(TC, "_daq_contact_onsets", lambda a, d: np.array([18.1]))
    M = TC.session_span_licks("PS92_0806", [10.0, 20.0], [8.0, 18.0], [14.0, 23.0], v36=tmp_path)
    assert M[0].tolist() == [1, 0, 0, 0, 1] and M[1].tolist() == [1, 0, 0, 0, 0]


def test_trial_attempts_excludes_stopped_licking():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import scripts.trial_attempts as ta
    assert "stopped_licking" in ta.RUN_EXCLUDED


# ---------------------------------------------------------------- per-cue stop kind (bcdg port's g_group_final)
def _kind_session():
    # contact, run of 7 (trial 3 lick-touched), contact, tail of 3 (trial 11 lick-touched)
    cls = [C] + [NR] * 7 + [C, NR, J, NR]
    pos = [FAR] * 12
    M = _span(12, k3_precue=1, k11_after_max_rt=1)
    t = np.arange(12) * 10.0 + 400.0
    t[0] = 0.0
    return np.array(cls, object), np.array(pos, object), M, t


def test_stop_kind_marks_run_and_tail_and_keeps_it_for_stopped_licking():
    cls, pos, M, t = _kind_session()
    for rule in ("both", "long"):
        out, kind = TC.classify_session(cls, np.ones(12, bool), pos, rule=rule, min_run=6, end="lick", cue_s=t,
                                        span_licks=M, lick_rule="excise", return_kind=True)
        if rule == "long":                                  # 7 < 11: no run under A3; the tail is the same
            assert kind.tolist()[1:8] == [""] * 7
            continue
        assert out.tolist() == [C, S, S, SL, S, S, S, S, C, S, S, SL]
        assert kind.tolist() == ["", "run", "run", "run", "run", "run", "run", "run", "", "tail", "tail", "tail"]
        assert set(kind) <= set(TC.STOP_KINDS)


def test_stop_kind_default_return_and_a0():
    cls, pos, M, t = _kind_session()
    assert isinstance(TC.classify_session(cls, None, pos, rule="both", end="lick", cue_s=t, span_licks=M), np.ndarray)
    out, kind = TC.classify_session(cls, None, pos, rule="run", min_run=6, end="lick", cue_s=t, return_kind=True)
    assert kind.tolist() == ["run" if c == S else "" for c in out]   # A0: no stopped tail (after_end), runs only


def test_stop_kind_for_matches_cue_classes_for(monkeypatch):
    import pandas as pd
    cls, pos, M, t = _kind_session()
    T = pd.DataFrame({"trial_id": np.arange(12), "cue_daq": t, "cls": cls, "position": pos, "strobe_ms": -2000.0,
                      "stop_ms": 3000.0})
    monkeypatch.setattr(TC, "session_trials", lambda lab, window=None: T.copy())
    monkeypatch.setattr(TC, "cue_span_licks", lambda lab, T_, cs: M)
    for k in (TC.STOPPED_ENV, TC.STOP_LICKS_ENV, TC.STOP_TAIL_ENV, "WIDEFIELD_TRIAL_END"):
        monkeypatch.delenv(k, raising=False)
    TC._CUE_CLS.clear()
    TC._CUE_KIND.clear()
    assert TC.cue_classes_for("PS92_0901", t).tolist()[3] == SL
    assert TC.stop_kind_for("PS92_0901", t).tolist() == ["", "run", "run", "run", "run", "run", "run", "run", "",
                                                         "tail", "tail", "tail"]
    TC._CUE_CLS.clear()
    TC._CUE_KIND.clear()
    assert TC.stop_kind_for("PS92_0901", t)[11] == "tail"           # works without a prior cue_classes_for call
    TC._CUE_CLS.clear()
    TC._CUE_KIND.clear()
