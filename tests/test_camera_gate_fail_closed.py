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
    "wfield_local/enl_states.py": "refuses",
    "scripts/enl_state_counts.py": "refuses",
    "scripts/stopped_trial_traces.py": "refuses",
    "scripts/rest_migration/channel_position_maps.py": "refuses",
    "scripts/rest_migration/archive/restw_reliability.py": "refuses",
    "scripts/engagement_axis_balance.py": "refuses",          # _gate_all mask (RefusedMask)
    "scripts/nocontact_census.py": "refuses",                 # _gate_all mask (RefusedMask)
}
CALL = re.compile(r"(?<![\w.])(?:engagement_gate|_gate_all|_quit_mask)\(")


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
    found -= {"scripts/archive/axis_composition_null.py"}         # archived, never run (uses the RefusedMask anyway)
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
    cls = np.array(["contact", "no_response", "jaw_only", "unknown", "no_response"], object)
    monkeypatch.setattr(TC, "cue_classes_for", lambda label, cue_s, window=None: cls[: len(cue_s)])
    assert TC.camera_engaged_period("X_0000", np.arange(5.0)).tolist() == [True, True, True, False, False]
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
                        lambda label, cue_s, window=None: np.array(["contact", "jaw_only", "no_response"], object))
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
    cam_u = np.array(["no_response", "jaw_only", "contact", "unknown", "unknown"], object)
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
