"""Priya's 11-03 decisions on the camera trial gate (coordinator relay, fix round 3), one test each."""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wfield_local import trial_classes as TC

REPO = Path(__file__).resolve().parents[1]
NR, U, C, J, I = "no_response", "unknown", "contact", "jaw_only", "incomplete"


def test_1_contact_nolick_in_neither_and_counted():
    """Decision 1: a no-lick (DAQ) row with camera class contact is in neither miss-while-working nor stopped, and is
    COUNTED per animal x epoch."""
    assert not TC.in_gate([C], "miss_working").any()
    assert not np.isin([C], list(TC.STOPPED_CLASSES)).any()
    t = TC.contact_nolick_counts(cls=[C, NR, C, I, C], nolick=[True, True, True, True, False],
                                 animal=["PS92"] * 5, epoch=["acute", "acute", "acute", "acute", "acute"])
    r = t.iloc[0]
    assert (r.n_nolick, r.n_contact_nolick) == (4, 2) and r.share == 0.5
    assert (REPO / "scripts" / "camera_coverage_tables.py").exists()


def _fake_session(monkeypatch, cls, cue_s, matched=None):
    """Route `cue_classes_for` through its REAL path (session_trials -> cue_classes -> classify_session) for a fake
    session whose video trials are ``cls`` at ``cue_s`` (``matched`` False = no video trial for that cue)."""
    cue_s = np.asarray(cue_s, float)
    keep = np.ones(len(cue_s), bool) if matched is None else np.asarray(matched, bool)
    t = pd.DataFrame({"cue_daq": cue_s[keep], "cls": np.asarray(cls, object)[keep], "position": ["far_R"] * keep.sum()})
    monkeypatch.setattr(TC, "session_trials", lambda label, window=None: t)
    monkeypatch.setenv(TC.STOPPED_ENV, "run")
    monkeypatch.setenv(TC.END_ENV, "lick")
    TC._CUE_CLS.clear()


def test_2_rest_period_is_first_to_last_camera_lick(monkeypatch):
    """Decision 2 + T8 + fourth review LOW: the rest-baseline working period runs from the session's FIRST camera lick
    to its LAST camera lick (warm-up before, jaw-only after are outside) -- through the real cue_classes_for path."""
    cls = [NR, NR, C, J, I, J, J, NR]
    _fake_session(monkeypatch, cls, np.arange(8.0))
    assert TC.camera_engaged_period("X_0001", np.arange(8.0)).tolist() == [False, False, True, True, True,
                                                                           False, False, False]


def test_2_t8_uncovered_tail_still_ends_at_the_last_camera_lick(monkeypatch):
    """T8: with an uncovered video tail the TRIAL classes keep no end cutoff, but the rest period still ends at the
    last camera lick (never at the last covered cue)."""
    cls = [C, I, J, J] + [U] * 6
    matched = [True] * 4 + [False] * 6                                  # 6 unmatched = a real uncovered tail
    _fake_session(monkeypatch, cls, np.arange(10.0), matched)
    out = TC.cue_classes_for("X_0002", np.arange(10.0))
    assert out.tolist() == [C, I, J, J] + [U] * 6                       # no after_end (tail uncovered)
    assert TC.camera_engaged_period("X_0002", np.arange(10.0)).tolist() == [True, True] + [False] * 8
    TC._CUE_CLS.clear()


def test_2_robustness_anchor_ends_the_rest_period_at_the_anchor(monkeypatch):
    """Under end anchor "contact" the incomplete licks after the last contact are after_end, so the rest period ends at
    the last contact (DECISIONS: under a robustness anchor the rest period ends at that anchor)."""
    cls = [C, I, C, I, I]
    _fake_session(monkeypatch, cls, np.arange(5.0))
    monkeypatch.setenv(TC.END_ENV, "contact")
    TC._CUE_CLS.clear()
    assert TC.camera_engaged_period("X_0003", np.arange(5.0)).tolist() == [True, True, True, False, False]
    TC._CUE_CLS.clear()


def test_3_noattempt_is_an_isolated_lapse_inside_the_working_period():
    """Decision 3: no_response NOT in a sustained stop stays "no_response" (an isolated lapse) only inside the working
    period; after the last lick it leaves both classes. attempt_coding / attempt_maps call it "noattempt"."""
    from wfield_local import attempt_coding, attempt_maps
    raw = np.array([C, NR, C] + [NR] * 6 + [NR], object)       # a lapse, then a sustained stop at the end
    out = TC.classify_session(raw, rule="run", min_run=6, end="lick")
    assert out[1] == NR and (out[3:] == "stopped").all()
    raw2 = np.array([C, NR, C, NR], object)                      # an isolated no_response AFTER the last lick
    assert TC.classify_session(raw2, rule="run", min_run=6, end="lick").tolist() == [C, NR, C, "after_end"]
    cam = np.array([NR, "stopped", "after_end"], object)
    cls_u = np.array(["no_try"] * 3)
    _ce, cu = attempt_coding.classify(np.array([], str), cls_u, np.array([], bool), np.zeros(3, bool),
                                      TC.RefusedMask("x"), camera_u=cam)
    assert cu.tolist() == ["noattempt", "stopped", ""]
    _ce, cu = attempt_maps.classes_for_rows(np.array([], str), cls_u, None, False, camera_u=cam)
    assert cu.tolist() == ["noattempt", "stopped", ""]


def test_4_parallel_ordering_rule_passes_for_the_seven_scripts():
    """Decision 4: SFX resolved at import (workers see it), JOBS in a dict, fan_out results sorted."""
    for f in ("arrival_to_cue_traces", "lick_cue_to_lick", "revup_neural", "stopped_trial_traces", "lick_iti_prelick"):
        tree = ast.parse((REPO / "scripts" / f"{f}.py").read_text(encoding="utf-8"))
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        assert not [g for g in ast.walk(main) if isinstance(g, ast.Global)], f


def test_5_three_readouts_are_stored():
    """Decision 5: the three remaining inline pre-LOSO ridges go through the store, with fingerprints."""
    for f, needle in (("face_offset_partial", "frozen_readouts.fit_all"),
                      ("lick_balance_blind_decoder", "frozen_readouts.pre_loso_ridge"),
                      ("lick_nonlinear_decoder", "frozen_models.load_or_fit")):
        src = (REPO / "scripts" / f"{f}.py").read_text(encoding="utf-8")
        assert needle in src and "for held in [*pre, None]" not in src.split("def _fit_reference")[0], f
    src = (REPO / "scripts" / "lick_nonlinear_decoder.py").read_text(encoding="utf-8")
    assert "train_data=" in src


def test_6_partial_video_coverage_rules_stay_inside_the_covered_span():
    """Decision 6 (N6): uncovered trials stay unknown; an uncovered TAIL disables the end cutoff; a stop run never
    stitches across an uncovered gap."""
    raw = np.array([C, J, J] + [U] * 5, object)                  # a REAL uncovered tail (>= MIN_UNCOVERED_RUN cues)
    m = np.array([True] * 3 + [False] * 5)
    out = TC.classify_session(raw, m, rule="run", min_run=6, end="lick")
    assert out.tolist() == [C, J, J] + [U] * 5                   # no after_end: the real last lick is unknown
    assert TC.classify_session(raw, None, rule="run", min_run=6, end="lick")[1] == "after_end"
    gap = np.array([C] + [NR] * 3 + [U] * 5 + [NR] * 3 + [C], object)       # a REAL gap (5 unmatched cues)
    mg = np.array([True] * 4 + [False] * 5 + [True] * 4)
    assert "stopped" not in TC.classify_session(gap, mg, rule="run", min_run=6, end="lick").tolist()
    ing = np.array([C] + [NR] * 3 + [U] * 2 + [NR] * 3 + [C], object)       # in-video unknowns (no jaw call)
    assert (TC.classify_session(ing, None, rule="run", min_run=6, end="lick") == "stopped").sum() == 6


def test_6_coverage_note(tmp_path):
    p = tmp_path / "camera_coverage.csv"
    pd.DataFrame([{"label": "PS92_0908", "frac": 0.825, "partial": True},
                  {"label": "PS93_0818", "frac": 1.0, "partial": False}]).to_csv(p, index=False)
    note = TC.coverage_note(p)
    assert "PS92_0908 82.5%" in note and "PS93_0818" not in note
    assert "missing" in TC.coverage_note(tmp_path / "nope.csv")


def test_7_jaw_only_after_the_last_lick_bridges_but_is_not_counted():
    """Decision 7 as revised by T3 (Priya): the review probe [contact, NR x3, jaw, NR x3, jaw, NR x2] is ONE bridged run
    of 8 no_response; the jaw-only trials are after_end, not stopped."""
    # T3 (Priya, as approved): the jaw-only trials BRIDGE the run but are not stopped and do not count toward the 6
    raw = np.array([C, NR, NR, NR, J, NR, NR, NR, J, NR, NR], object)
    out = TC.classify_session(raw, rule="run", min_run=6, end="lick")
    S = "stopped"
    assert out.tolist() == [C, S, S, S, "after_end", S, S, S, "after_end", S, S]
    early = np.array([NR, NR, NR, J, C, NR, NR, NR], object)            # a jaw BEFORE the last lick still breaks
    assert "stopped" not in TC.classify_session(early, rule="run", min_run=6, end="lick").tolist()


def test_sensitivity_rule_any_and_cue_classes_for_uses_the_session_rules(monkeypatch):
    raw = np.array([C, NR, C, NR], object)
    assert TC.classify_session(raw, rule="any", min_run=6, end="lick").tolist() == [C, "stopped", C, "stopped"]
    t = pd.DataFrame({"cue_daq": [1.0, 2.0, 3.0], "cls": [C, J, J], "position": ["far_R"] * 3})
    monkeypatch.setattr(TC, "session_trials", lambda label, window=None: t)
    monkeypatch.setenv(TC.STOPPED_ENV, "run")
    monkeypatch.setenv(TC.END_ENV, "lick")
    TC._CUE_CLS.clear()
    # T7: ONE trailing unmatched cue is not an uncovered tail -> the end cutoff applies
    assert TC.cue_classes_for("Y_0001", [1.0, 2.0, 3.0, 4.0]).tolist() == [C, "after_end", "after_end", U]
    TC._CUE_CLS.clear()
    assert TC.cue_classes_for("Y_0002", [1.0, 2.0, 3.0]).tolist() == [C, "after_end", "after_end"]
    TC._CUE_CLS.clear()
    with pytest.raises(ValueError):
        TC.classify_session(raw, rule="run", min_run=6, end="bogus")


def test_nolick_window_is_cue_stop_and_named_in_the_tag(monkeypatch):
    """FINAL no-lick window (Priya 11-03): the no-lick analysis uses trial_classes' cue_stop window and table; the tag
    names cue_stop even when trial_selection.window says otherwise; a DAQ-undetected cue whose cue_stop class is
    contact is a contact trial (late_rewarded), never no-lick."""
    from wfield_local import nolick_decoder
    monkeypatch.setenv(TC.GATE_ENV, "camera")
    monkeypatch.setenv(TC.WINDOW_ENV, "trial")
    assert TC.nolick_selection().startswith("sel-camera-cue_stop-")
    monkeypatch.setenv(TC.GATE_ENV, "daq")
    assert TC.nolick_selection() == TC.run_selection()
    src = (REPO / "wfield_local" / "nolick_decoder.py").read_text(encoding="utf-8")
    assert 'window="cue_stop"' in src and '(ccls == "contact"), "late_rewarded"' in src
    assert callable(nolick_decoder.categorize)


def test_check_table_causes_and_stop_lengths():
    import sys
    sys.path.insert(0, str(REPO / "scripts"))
    import camera_coverage_tables as cct
    cat = np.array(["undetected"] * 4 + ["engaged"], object)
    cls = np.array([C] * 4 + [C], object)
    cue_f = np.array([100, 200, 300, 400, 500])
    ls = np.array([200, 297, 900])                   # a lick IN cue 2's frame; one 3 frames before cue 3
    contact_ms = np.array([4000.0, 500.0, 500.0, 500.0, 300.0])
    why = cct.disagreement_causes(cat, cls, cue_f, ls, 31.25, contact_ms, 3.5)
    assert why.tolist() == ["late_after_response_window", "cue_frame", "refractory_merge", "definition_mismatch", ""]
    T = pd.DataFrame({"animal": ["PS92"] * 3, "epoch": ["pre"] * 3, "stop_ms": [3525.0, 3700.0, 5000.0]})
    s = cct.stop_lengths(T).iloc[0]
    assert (s.n_trials, s.n_stop_gt_3600, s.median_stop_gt_3600_ms) == (3, 2, 4350.0)


# ------------------------------------------------------------- third review T3 / T7 / T11 (rule edges)
def test_runs_of_exactly_five_and_six_no_response():
    five = np.array([C] + [NR] * 5 + [C], object)
    six = np.array([C] + [NR] * 6 + [C], object)
    assert "stopped" not in TC.classify_session(five, rule="run", min_run=6, end="lick").tolist()
    assert TC.classify_session(six, rule="run", min_run=6, end="lick").tolist() == [C] + ["stopped"] * 6 + [C]


def test_all_jaw_run_is_never_a_stop():
    jaws = np.array([C] + [J] * 8, object)
    out = TC.classify_session(jaws, rule="run", min_run=6, end="lick").tolist()
    assert out == [C] + ["after_end"] * 8 and "stopped" not in out
    mixed = np.array([C] + [NR, J] * 5 + [NR], object)                      # 6 NR bridged by 5 jaw-only
    out = TC.classify_session(mixed, rule="run", min_run=6, end="lick").tolist()
    assert out.count("stopped") == 6 and out.count("after_end") == 5


def test_empty_session_and_no_camera_lick():
    assert TC.classify_session(np.array([], object), rule="run", min_run=6, end="lick").tolist() == []
    nolick = np.array([NR] * 7 + [J], object)                               # no camera lick at all
    out = TC.classify_session(nolick, rule="run", min_run=6, end="lick").tolist()
    assert out == ["stopped"] * 7 + ["after_end"]      # no time given: a run of 7 is a stop; no working period
    t = np.arange(8) * 10.0                                                 # all within 5 min of the first cue
    out = TC.classify_session(nolick, rule="run", min_run=6, end="lick", cue_s=t).tolist()
    assert out == ["early_run"] * 7 + ["after_end"]


def test_time_rule_a_run_must_start_5_min_after_the_first_cue():
    """Priya 11-03: a stop counts only if its run STARTS >= 5 min (300 s) after the session's first cue; an earlier run
    is in neither class (the warm-up class is gone: isolated no_response before the first lick is a lapse)."""
    raw = np.array([NR] * 7 + [C, J, C] + [NR] * 6 + [C], object)
    t = np.r_[np.arange(7) * 10.0, 100.0, 110.0, 120.0, 400.0 + np.arange(6) * 10.0, 470.0]
    out = TC.classify_session(raw, rule="run", min_run=6, end="lick", cue_s=t).tolist()
    assert out[:7] == ["early_run"] * 7 and out[10:16] == ["stopped"] * 6
    assert not TC.in_gate(["early_run"], "working").any() and "early_run" not in TC.STOPPED_CLASSES
    straddle = np.array([C] + [NR] * 6 + [C], object)                       # starts at 290 s, ends after 300 s
    ts = np.r_[0.0, 290.0 + np.arange(6) * 5.0, 330.0]
    assert TC.classify_session(straddle, rule="run", min_run=6, end="lick", cue_s=ts).tolist()[1:7] == ["early_run"] * 6
    lapse = np.array([NR, NR, C, C], object)                                 # isolated no_response before the lick
    assert TC.classify_session(lapse, rule="run", min_run=6, end="lick", cue_s=[0, 1, 2, 3.0]).tolist()[:2] == [NR, NR]
    assert TC.stop_min_start_s() == 300.0 and TC.stopped_rule_name() == "stopRun6"
    with pytest.raises(ValueError):
        TC.classify_session(raw, rule="run", min_run=6, end="lick", cue_s=t[:3])


def test_short_unmatched_stretches_are_unknown_not_uncovered():
    raw = np.array([C] + [NR] * 3 + [U] + [NR] * 3 + [C, J], object)       # one isolated unmatched cue mid-run
    m = np.array([True] * 4 + [False] + [True] * 5)
    out = TC.classify_session(raw, m, rule="run", min_run=6, end="lick").tolist()
    assert out.count("stopped") == 6 and out[-1] == "after_end"
    head = np.array([U] * 5 + [C, J], object)                                 # a missing HEAD
    mh = np.array([False] * 5 + [True] * 2)
    assert TC.classify_session(head, mh, rule="run", min_run=6, end="lick").tolist() == [U] * 5 + [C, "after_end"]


def test_7_bridge_needs_six_no_response_nr3_j_nr2():
    """T3: a jaw-only bridge does not count, so NR3 J NR2 after the last lick (5 no_response) is NOT a stop: all after_end;
    NR3 J NR3 (6) is."""
    A, S = "after_end", "stopped"
    five = np.array([C, NR, NR, NR, J, NR, NR], object)
    assert TC.classify_session(five, rule="run", min_run=6, end="lick").tolist() == [C] + [A] * 6
    six = np.array([C, NR, NR, NR, J, NR, NR, NR], object)
    assert TC.classify_session(six, rule="run", min_run=6, end="lick").tolist() == [C, S, S, S, A, S, S, S]


def test_7_no_bridge_with_an_uncovered_tail():
    """No end cutoff with an uncovered tail, so jaw-only trials are working and do not bridge: no stop."""
    raw = np.array([C, NR, NR, NR, J, NR, NR, NR] + [U] * 5, object)
    m = np.array([True] * 8 + [False] * 5)
    out = TC.classify_session(raw, m, rule="run", min_run=6, end="lick")
    assert "stopped" not in out.tolist() and out[4] == J


def test_stop_run_ends_where_licking_resumes_and_unknown_is_skipped():
    """Fourth review LOW: an in-video unknown is skipped (neither breaks nor extends a run), so [C,N,N,N,U,N,N,N,C] is
    one run of 6 no_response = stopped; a contact or an incomplete inside the run ENDS it (licking resumed): 3 + 3 is
    not a stop."""
    S = "stopped"
    raw = np.array([C, NR, NR, NR, U, NR, NR, NR, C], object)
    assert TC.classify_session(raw, rule="run", min_run=6, end="lick").tolist() == [C, S, S, S, U, S, S, S, C]
    for lick in (C, I):
        split = np.array([C, NR, NR, NR, lick, NR, NR, NR, C], object)
        assert "stopped" not in TC.classify_session(split, rule="run", min_run=6, end="lick").tolist(), lick
    jaw_early = np.array([C, NR, NR, NR, J, NR, NR, NR, C], object)       # jaw-only before the last lick ends it too
    assert "stopped" not in TC.classify_session(jaw_early, rule="run", min_run=6, end="lick").tolist()


@pytest.mark.parametrize("where", ["head", "middle", "tail"])
def test_min_uncovered_run_is_exactly_five(where):
    """T7: 4 consecutive unmatched cues are NOT uncovered (filled), 5 are -- at the head, mid-session and the tail. Any
    MIN_UNCOVERED_RUN other than 5 fails one of the two."""
    assert TC.MIN_UNCOVERED_RUN == 5
    for k, uncovered in ((4, False), (5, True)):
        n = 20
        m = np.ones(n, bool)
        lo = {"head": 0, "middle": 8, "tail": n - k}[where]
        m[lo: lo + k] = False
        eff = TC._effective_coverage(m)
        assert (not eff.all()) == uncovered, (where, k)
    # the end cutoff depends on it at the tail: 4 unmatched -> cutoff applies, 5 -> dropped
    for k, cut in ((4, True), (5, False)):
        raw = np.array([C, J] + [U] * k, object)
        m = np.array([True, True] + [False] * k)
        assert (TC.classify_session(raw, m, rule="run", min_run=6, end="lick")[1] == "after_end") == cut, k


def test_rule_any_applies_the_time_rule():
    raw = np.array([NR, NR, C, NR, C], object)
    t = np.array([0.0, 10.0, 20.0, 400.0, 410.0])
    assert TC.classify_session(raw, rule="any", min_run=6, end="lick", cue_s=t).tolist() == ["early_run", "early_run", C,
                                                                                          "stopped", C]


def test_rest_start_is_a_named_setting(monkeypatch):
    cls = [NR, NR, C, J, I, J, J, NR]
    _fake_session(monkeypatch, cls, np.arange(8.0))
    assert TC.rest_start() == "first_lick"
    assert TC.camera_engaged_period("X_0009", np.arange(8.0)).tolist()[:5] == [False, False, True, True, True]
    monkeypatch.setattr(TC, "_selection_cfg", lambda: {"rest_start": "session"})
    TC._CUE_CLS.clear()
    assert TC.camera_engaged_period("X_0009", np.arange(8.0)).tolist()[:5] == [True] * 5
    TC._CUE_CLS.clear()
