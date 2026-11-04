"""Camera trial gate: every trial-SELECTION gate is wired to `trial_classes` or REFUSES (rebuild 2026-11-03, task 2).

Under ``WIDEFIELD_TRIAL_GATE=camera`` an output is tagged sel-camera, so a gate still on the DAQ quit-tail rule would
mislabel it. The DAQ rule (`precue_engagement_states.engagement_gate`) refuses unless the caller opened
`trial_classes.daq_quit_rule(reason)`; the quit-tail MASK handed out by `_gate_all` / the grant bundle is a
`RefusedMask`. Measures OF engagement (quit, engagement decline, attempt rate, the behaviour reporting gate) keep the DAQ
rule. The REGISTER below lists every call site; a new one fails `test_every_daq_rule_call_site_is_registered` until
someone decides which kind it is.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from wfield_local import trial_classes as TC

REPO = Path(__file__).resolve().parents[1]

#: file -> how its DAQ quit-tail call sites behave under gate=camera
#:   wired    a camera branch takes the classes from trial_classes; the DAQ call is on the daq branch only
#:   exempt   a measure of engagement / a diagnostic of the gate rules / a labelled DAQ row, inside daq_quit_rule
#:   refuses  not wired: under gate=camera it raises DaqGateRefused (engagement_gate) or uses a RefusedMask
REGISTER = {
    "wfield_local/precue_engagement_states.py": "wired",      # states 5 / 6 (+ engagement_gate itself)
    "wfield_local/nolick_decoder.py": "wired",                # categorize (-> cd_trajectories, enl_decode)
    "wfield_local/poststroke_compare.py": "wired",            # poststroke_engagement / nolick_state_vector
    "wfield_local/rest_engagement.py": "wired",               # engaged_by_cue / engaged_frame_mask
    "wfield_local/rest_by_position.py": "wired",              # _engaged_trials
    "wfield_local/position_coding_directions.py": "wired",    # _gate_all -> RefusedMask under camera
    "wfield_local/grant_kit.py": "wired",                     # _pooled_bundle / _class_select
    "wfield_local/attempt_coding.py": "wired",
    "wfield_local/attempt_maps.py": "wired",
    "wfield_local/position_axes.py": "wired",
    "wfield_local/beta_maps.py": "wired",                     # _quit_mask only on the daq branch of session_maps
    "wfield_local/hemo_map_control.py": "wired",
    "wfield_local/position_reference_maps.py": "wired",
    "wfield_local/spout_behavior.py": "exempt",               # behaviour reporting gate
    "scripts/disengagement_tests.py": "exempt",               # quit onset
    "scripts/rest_migration/quit_point.py": "exempt",         # the S28 quit
    "scripts/gate_lp_check.py": "exempt",                     # diagnostic of the DAQ / LP gate rules
    "scripts/lp_gate_option_counts.py": "exempt",             # same
    "scripts/s1_refit_basis_table.py": "exempt",              # labelled quit-tail robustness row
    "wfield_local/enl_states.py": "wired",                    # states_for(cam_cls=...) under camera (11-03)
    "scripts/enl_state_counts.py": "refuses",
    "scripts/stopped_trial_traces.py": "refuses",
    "scripts/rest_migration/channel_position_maps.py": "refuses",
    "scripts/rest_migration/archive/restw_reliability.py": "refuses",
    "scripts/engagement_axis_balance.py": "refuses",          # _gate_all mask (RefusedMask)
    "scripts/nocontact_census.py": "refuses",                 # _gate_all mask (RefusedMask)
    # callers of the two shared MEASURE functions that open daq_quit_rule themselves (second pass 11-03): every new
    # caller must be classified here too, so an imaging selection cannot reach the DAQ rule through them unregistered
    "wfield_local/analysis_kit.py": "measure",               # session_behavior -> session_quit (the S28 quit)
    "wfield_local/behavior_clips.py": "measure",             # reference_engagement (behaviour clip labels)
}
CALL = re.compile(r"(?<![\w.])(?:engagement_gate|_gate_all|_quit_mask|reference_engagement|session_quit)\(")


@pytest.fixture
def camera(monkeypatch):
    monkeypatch.setenv(TC.GATE_ENV, "camera")


@pytest.fixture
def daq(monkeypatch):
    monkeypatch.setenv(TC.GATE_ENV, "daq")


def _seq():
    """30 trials at the reference spouts, responding, then a non-recovering collapse (the quit)."""
    order = np.arange(60)
    responded = np.r_[np.ones(40, bool), np.zeros(20, bool)]
    pos = np.array(["close_L", "close_center"] * 30)
    return order, responded, pos


def test_every_daq_rule_call_site_is_registered():
    found = set()
    for base in ("wfield_local", "scripts"):
        for p in (REPO / base).rglob("*.py"):
            rel = p.relative_to(REPO).as_posix()
            for ln in p.read_text(encoding="utf-8").splitlines():
                code = ln.split("#", 1)[0]
                if CALL.search(code) and not re.match(r"\s*def ", code):
                    found.add(rel)
                    break
    found -= {"scripts/archive/axis_composition_null.py",          # archived, never run (uses the RefusedMask anyway)
              "scripts/archive/gate_bias_probe.py"}                 # archived behaviour probe
    assert found == set(REGISTER), {"unregistered": sorted(found - set(REGISTER)),
                                    "stale": sorted(set(REGISTER) - found)}


def test_exempt_sites_open_the_daq_rule():
    for rel, kind in REGISTER.items():
        if kind == "exempt":
            assert "daq_quit_rule(" in (REPO / rel).read_text(encoding="utf-8"), rel


def test_engagement_gate_default_unchanged_and_refuses_under_camera(daq, monkeypatch):
    from wfield_local.precue_engagement_states import engagement_gate
    ref = np.asarray(engagement_gate(*_seq()), bool)
    assert ref[-20:].all() and not ref[:30].any()
    monkeypatch.setenv(TC.GATE_ENV, "camera")
    with pytest.raises(TC.DaqGateRefused):
        engagement_gate(*_seq())
    with TC.daq_quit_rule("test: a measure of engagement"):
        assert np.array_equal(np.asarray(engagement_gate(*_seq()), bool), ref)
    with pytest.raises(TC.DaqGateRefused):                         # the block closed again
        engagement_gate(*_seq())


def test_refusal_is_not_swallowed_by_fail_open_handlers(camera):
    from wfield_local.precue_engagement_states import engagement_gate
    with pytest.raises(SystemExit):
        try:
            engagement_gate(*_seq())
        except Exception:                                          # noqa: BLE001 - what the rest gates do
            pytest.fail("DaqGateRefused was caught by `except Exception`")


def test_refused_mask_refuses_every_use():
    m = TC.RefusedMask("test")
    a = np.ones(3, bool)
    for use in (lambda: ~m, lambda: a & m, lambda: m & a, lambda: a | m, lambda: np.asarray(m), lambda: len(m),
                lambda: m[0], lambda: bool(m), lambda: list(m)):
        with pytest.raises(TC.DaqGateRefused):
            use()


def test_camera_engaged_period(monkeypatch):
    # DECIDED (Priya 11-03, decision 2): the period ends at the session LAST CAMERA LICK
    cls = np.array(["contact", "no_response", "incomplete", "jaw_only", "unknown", "no_response"], object)
    monkeypatch.setattr(TC, "cue_classes_for", lambda label, cue_s, window=None: cls[: len(cue_s)])
    monkeypatch.setattr(TC, "_matched_cues", lambda label, cue_s, window=None: np.ones(len(cue_s), bool))
    assert TC.camera_engaged_period("X_0000", np.arange(6.0)).tolist() == [True, True, True, False, False, False]
    cls = np.array(["unknown"] * 4, object)
    assert not TC.camera_engaged_period("X_0000", np.arange(4.0)).any()        # no video: no working period


def test_rest_gates_camera_branch_does_not_fail_open(camera, monkeypatch):
    from wfield_local import rest_by_position, rest_engagement
    import wfield_local.plot_spout_trial_averages as psta
    monkeypatch.setattr(psta, "_load_daq_events", lambda h5, *a, **k: {"sample_rate_hz": 5000.0})

    def boom(*_a, **_k):
        raise RuntimeError("class table unreadable")
    monkeypatch.setattr(TC, "cue_classes_for", boom)
    s = {"label": "X_0000", "h5": "nowhere.h5"}
    with pytest.raises(RuntimeError, match="unreadable"):
        rest_engagement.engaged_by_cue(s, np.array([5000, 10000]), np.array([0, 1]))
    with pytest.raises(RuntimeError, match="unreadable"):
        rest_by_position._engaged_trials(s, np.array([5000, 10000]), np.array([0, 1]))
    monkeypatch.setattr(TC, "cue_classes_for",
                        lambda label, cue_s, window=None: np.array(["contact", "incomplete", "no_response"], object))
    monkeypatch.setattr(TC, "_matched_cues", lambda label, cue_s, window=None: np.ones(len(cue_s), bool))
    eng, note = rest_engagement.engaged_by_cue(s, np.array([5000, 10000, 15000]), np.array([0, 1, 2]))
    assert eng.tolist() == [True, True, False]
    assert "sel-camera" in note
    assert rest_by_position._engaged_trials(s, np.array([5000, 10000, 15000]), np.array([0, 1, 2])).tolist() == \
        [True, True, False]


def _toy_pool(n_sess=2):
    """A tiny pooled no-lick bookkeeping set for `_gate_all`: per session 4 engaged + 3 no-lick rows."""
    class F:
        indices = {}
    kept = [f"PS92_08{10 + i}" for i in range(n_sess)]
    YE, GE, YU, GU = [], [], [], []
    for i, lab in enumerate(kept):
        F.indices[lab] = (np.array([0, 1, 2, 3]), np.array([4, 5, 6]))
        YE += [1, 3, 1, 3]; GE += [i] * 4; YU += [1, 3, 1]; GU += [i] * 3
    return F, kept, np.array(YE), np.array(GE), np.array(YU), np.array(GU)


def test_gate_all_refuses_its_mask_but_keeps_bookkeeping(monkeypatch):
    from wfield_local import position_coding_directions as pcd
    F, kept, YE, GE, YU, GU = _toy_pool()
    XE, XU = np.zeros((len(YE), 2)), np.zeros((len(YU), 2))
    monkeypatch.setenv(TC.GATE_ENV, "daq")
    ne, fnl, fe = pcd._gate_all(F, kept, XE, YE, GE, XU, YU, GU)
    assert isinstance(ne, np.ndarray) and ne.dtype == bool and len(ne) == len(YU)
    monkeypatch.setenv(TC.GATE_ENV, "camera")
    ne_c, fnl_c, fe_c = pcd._gate_all(F, kept, XE, YE, GE, XU, YU, GU)
    assert isinstance(ne_c, TC.RefusedMask)
    assert np.array_equal(fnl, fnl_c) and np.array_equal(fe, fe_c)
    ne_d, _, _ = pcd._gate_all(F, kept, XE, YE, GE, XU, YU, GU, rule="daq")      # labelled DAQ row
    assert np.array_equal(ne_d, ne)


def test_attempt_classes_under_camera():
    from wfield_local import attempt_coding, attempt_maps
    cls_u = np.array(["no_try", "jaw_only", "hit", "no_try", "unknown"])
    cam_u = np.array(["stopped", "jaw_only", "contact", "unknown", "unknown"], object)
    post = np.zeros(5, bool)
    _ce, cu = attempt_coding.classify(np.array([], str), cls_u, np.array([], bool), post, TC.RefusedMask("x"),
                                      camera_u=cam_u)
    assert cu.tolist() == ["stopped", "reaction", "late", "", ""]
    _ce, cu = attempt_maps.classes_for_rows(np.array([], str), cls_u, None, False, camera_u=cam_u)
    assert cu.tolist() == ["stopped", "reaction", "", "", ""]
    # daq rule unchanged
    q = np.array([True, False, False, False, True])
    _ce, cu = attempt_maps.classes_for_rows(np.array([], str), cls_u, q, False)
    assert cu.tolist() == ["stopped", "reaction", "", "noattempt", ""]


# ============================================================================ review fixes (11-03, second pass)
def test_camera_miss_working_excludes_contact(monkeypatch):
    """Review M6: miss-while-working on the no-lick arm = responded, NO contact class."""
    cls = np.array(["contact", "incomplete", "jaw_only", "no_response", "unknown"], object)
    monkeypatch.setattr(TC, "row_classes", lambda label, idx: cls[np.asarray(idx)])
    ke, kn = TC.camera_rows("X_0000", np.array([], int), np.arange(5), "miss_working")
    assert kn.tolist() == [False, True, False, False, False]       # 11-03 late: jaw_only is not working
    ke, kn = TC.camera_rows("X_0000", np.array([], int), np.arange(5), "working")
    assert kn.tolist() == [True, True, False, False, False]


def test_require_gate(monkeypatch):
    monkeypatch.setenv(TC.GATE_ENV, "daq")
    TC.require_gate("daq", "x")
    monkeypatch.setenv(TC.GATE_ENV, "camera")
    with pytest.raises(TC.DaqGateRefused):
        TC.require_gate("daq", "x")


def test_class_signature_moves_with_the_sources(tmp_path, monkeypatch):
    """Review H4: a regenerated jaw_onsets / v36 / cam1 table changes the camera cache signature."""
    v36 = tmp_path / "v36" / "sessions"
    (v36 / "PS92" / "20260818").mkdir(parents=True)
    (v36 / "PS92" / "20260818" / "trials.csv").write_text("a\n1\n")
    c1 = tmp_path / "cam1"
    c1.mkdir()
    qc = tmp_path / "v36" / "imaging_encoding" / "qc"
    qc.mkdir(parents=True)
    (qc / "jaw_onsets.csv").write_text("x\n")
    TC._CLASS_SIG.clear()
    s0 = TC.class_source_signature(v36, c1)
    for f in (qc / "jaw_onsets.csv", v36 / "PS92" / "20260818" / "trials.csv"):
        TC._CLASS_SIG.clear()
        f.write_text(f.read_text() + "2\n")
        s1 = TC.class_source_signature(v36, c1)
        assert s1 != s0
        s0 = s1
    TC._CLASS_SIG.clear()
    (c1 / "PS92_20260818_events.csv").write_text("e\n")
    assert TC.class_source_signature(v36, c1) != s0


def test_selection_matched_in_daq_lookup_and_carries_gate_source(monkeypatch):
    """Review M7: find_daq_trained matches selection; the gate-LP source enters the frozen selection."""
    from wfield_local import frozen_models as fm
    assert "selection" not in fm._DAQ_LOOKUP_IGNORES
    monkeypatch.setenv(TC.GATE_ENV, "daq")
    monkeypatch.delenv("WIDEFIELD_GATE_SOURCE", raising=False)
    assert TC.run_selection() == "sel-daq-quittail" and TC.run_selection_tag() is None
    monkeypatch.setenv("WIDEFIELD_GATE_SOURCE", "lp")
    assert TC.run_selection() == "sel-daq-quittail-gateLP" and TC.run_selection_tag() is None


def test_lp_only_kinds_fingerprint_real_rows_under_daq_crosscheck(monkeypatch):
    """Review H1: an LP-only-by-nature kind never gets the constant DAQ_RUN_FP."""
    from wfield_local import frozen_models as fm
    monkeypatch.setattr(fm, "daq_trained_in_lp", lambda: True)
    monkeypatch.setattr(fm, "_session_sig", lambda lab: "sig")
    kw = dict(align="cue-dom", source="locanmf", train_labels=["PS92_0806"])
    a = fm.make_spec("PS92", "attempt_axes", train_data=np.arange(3), lp_only=True, **kw)
    b = fm.make_spec("PS92", "attempt_axes", train_data=np.arange(4), lp_only=True, **kw)
    assert a["train_fp"] != fm.DAQ_RUN_FP and a["train_fp"] != b["train_fp"] and a["lick_source"] == "lp"
    assert fm.make_spec("PS92", "decoder", train_data=np.arange(3), **kw)["train_fp"] == fm.DAQ_RUN_FP


def test_projall_part_and_full_output_tag(monkeypatch):
    """Review M1 / M3: the forced projection control is its own part; keyed names follow LP / gate-LP too."""
    from wfield_local import run_mode
    for k in ("WIDEFIELD_HEMO_VARIANT", "WIDEFIELD_TRIAL_GATE", "WIDEFIELD_LICK_SOURCE", "WIDEFIELD_GATE_SOURCE",
              "WIDEFIELD_JOINT_PROJECT_ALL", "WIDEFIELD_ENL_WINDOW_S"):
        monkeypatch.delenv(k, raising=False)
    assert run_mode.keyed_name("section_g.json") == "section_g.json"
    monkeypatch.setenv("WIDEFIELD_JOINT_PROJECT_ALL", "1")
    assert "projall" in run_mode.parts() and "projall" in run_mode.cache_parts()
    monkeypatch.delenv("WIDEFIELD_JOINT_PROJECT_ALL")
    monkeypatch.setenv("WIDEFIELD_GATE_SOURCE", "lp")
    assert run_mode.keyed_name("coding_direction.json") == "coding_direction__DAQ_gateLP.json"


def test_figures_working_is_tag_checked(monkeypatch, tmp_path):
    """Review H2: a tagged run may not write a default-named file under the local figures_working root."""
    from wfield_local import config, writeguard

    class R:
        def root(self, name):
            return str(tmp_path / "cue_lick")
    monkeypatch.setattr(config, "resolver", lambda: R())
    monkeypatch.delenv("WIDEFIELD_TRIAL_GATE", raising=False)
    assert writeguard.is_writable(tmp_path / "cue_lick" / "coding_direction_cue.png")
    monkeypatch.setenv("WIDEFIELD_TRIAL_GATE", "camera")
    assert not writeguard.is_writable(tmp_path / "cue_lick" / "coding_direction_cue.png")
    assert writeguard.is_writable(tmp_path / "cue_lick" / "sel-camera-cue_stop-endlick-cls-contact-incomplete-stopA2jexrt3500c70" / "coding_direction_cue.png")


def test_cue_classes_all_nan_table_is_unknown():
    """Review L4: an all-NaN cue_daq table gives "unknown" for every cue (origin/main raised IndexError); pinned."""
    import pandas as pd
    t = pd.DataFrame({"cue_daq": [np.nan, np.nan], "cls": ["contact", "no_response"]})
    assert TC.cue_classes(t, [1.0, 2.0]).tolist() == ["unknown", "unknown"]


def test_sustained_stop_rule(monkeypatch):
    """Priya 11-03: stopped = no_response inside a run of >= tail_min_misses (6) consecutive camera no_response trials;
    unknown trials neither break nor extend a run and are never stopped; isolated no_response = neither class.
    Sensitivity rule "any": every no_response is stopped. The rule name is in the selection tag."""
    NR, U, C = "no_response", "unknown", "contact"
    cls = [C, NR, NR, C] + [NR, NR, U, NR, NR, NR, NR] + [C, NR] + [U, U]
    run = lambda c, rule="run": TC.classify_session(np.array(c, object), rule=rule, min_run=6, end="none")  # noqa: E731
    out = run(cls).tolist()
    assert out[:4] == [C, NR, NR, C]                                   # a run of 2: isolated, unchanged
    assert out[4:11] == ["stopped"] * 2 + [U] + ["stopped"] * 4       # 6 no_response across one unknown
    assert out[11:] == [C, NR, U, U]
    assert run([C] + [NR] * 5 + [U] * 3).tolist() == [C] + [NR] * 5 + [U] * 3   # unknown never extends
    assert run(cls, "any").tolist().count("stopped") == cls.count(NR)
    assert not TC.in_gate(["no_response", "stopped"], "working").any()
    monkeypatch.setenv(TC.GATE_ENV, "camera")
    monkeypatch.delenv(TC.STOPPED_ENV, raising=False)
    assert TC.run_selection_tag() == "sel-camera-cue_stop-endlick-cls-contact-incomplete-stopA2jexrt3500c70"
    monkeypatch.setenv(TC.STOPPED_ENV, "any")
    assert TC.run_selection_tag() == "sel-camera-cue_stop-endlick-cls-contact-incomplete-stopAnyrt3500c70" and TC.run_selection() == TC.run_selection_tag()
    monkeypatch.setenv(TC.STOPPED_ENV, "bogus")
    with pytest.raises(ValueError):
        TC.stopped_rule()


def test_enl_states_camera_branch(monkeypatch):
    from wfield_local import enl_states
    monkeypatch.setenv(TC.GATE_ENV, "camera")
    cam = np.array(["contact", "incomplete", "stopped", "no_response", "unknown"], object)
    order = np.arange(5)
    fl = np.array([0.3, np.nan, np.nan, np.nan, np.nan])
    st = enl_states.states_for(order, fl > 0, np.array(["far_R"] * 5), fl, fl, 2.0, cam_cls=cam)
    assert st.tolist()[1:] == ["working", "stopped", "", ""]
    with pytest.raises(TC.DaqGateRefused):
        enl_states.states_for(order, fl > 0, np.array(["far_R"] * 5), fl, fl, 2.0)


def test_session_end_relabel_equals_in_working(monkeypatch):
    """Priya 11-03 (main b506c1e3): camera working stops at the session LAST camera lick. `cue_classes_for` applies it
    once (`classify_session`), so in_gate(cls, "working") on its output == in_working(raw cls, cue order)."""
    raw = np.array(["jaw_only", "contact", "jaw_only", "no_response", "incomplete", "jaw_only", "unknown", "jaw_only"],
                   object)
    for end in ("lick", "contact", "none"):
        rel = TC.classify_session(raw, rule="run", min_run=6, end=end)
        assert np.array_equal(TC.in_gate(rel, "working"), TC.in_working(raw, np.arange(len(raw)), end=end)), end
    rel = TC.classify_session(raw, rule="run", min_run=6, end="lick")
    assert rel.tolist()[5:] == ["after_end", "unknown", "after_end"]          # jaw-only after the last lick leave
    assert TC.classify_session(raw, rule="run", min_run=6, end="contact").tolist()[4] == "after_end"   # incomplete after the last contact
    pos = np.array(["close_L", "close_L", "far_R", "far_R", "far_R", "far_R", "far_R", "far_R"], object)
    assert not TC.in_gate(TC.classify_session(raw, None, pos, rule="run", min_run=6, end="close"), "working")[2:].any()
    monkeypatch.setenv(TC.GATE_ENV, "camera")
    monkeypatch.setenv(TC.END_ENV, "contact")
    assert TC.run_selection_tag().startswith("sel-camera-cue_stop-endcontact-cls-contact-incomplete-")
    monkeypatch.setenv(TC.END_ENV, "bogus")
    with pytest.raises(ValueError):
        TC.end_anchor()
