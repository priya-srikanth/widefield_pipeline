"""Section G's preserved positions count camera WORKING rows, not successful licks (Priya 11-07: "we should NOT gate on
successful licks"). A position with 50 incomplete working rows but only 5 DAQ-engaged rows is kept under the camera
in_working selection, dropped under in_working_contact and under the DAQ rule; the post-lick arm (DAQ-lick aligned)
never scores a position without >= MIN_POST engaged rows."""
import numpy as np
import pytest

from wfield_local import poststroke_compare as pc
from wfield_local import poststroke_section_g as sg
from wfield_local import trial_classes as tc
from wfield_local import working_trials as wt

FAR_R = 5          # the "far contra" position of the synthetic pool
OTHERS = (1, 0, 2, 4, 3)


def _pool():
    """Session 0 = pre, 1 = post. Post: every other position 30 engaged contact rows; FAR_R 5 engaged contact rows plus
    45 no-lick incomplete rows (DAQ: no lick)."""
    YE, GE, CE, YU, GU, CU = [], [], [], [], [], []
    for g in (0, 1):
        for c in OTHERS:
            YE += [c] * 30
            GE += [g] * 30
            CE += ["contact"] * 30
        ne = 30 if g == 0 else 5
        YE += [FAR_R] * ne
        GE += [g] * ne
        CE += ["contact"] * ne
        if g == 1:
            YU += [FAR_R] * 45
            GU += [1] * 45
            CU += ["incomplete"] * 45
    d = {"YE": np.array(YE), "GE": np.array(GE), "YU": np.array(YU), "GU": np.array(GU),
         "pre_i": {0}, "post_i": {1}, "kept": ["PSX_0806", "PSX_0820"]}
    return d, np.array(CE, dtype=object), np.array(CU, dtype=object)


@pytest.fixture
def camera(monkeypatch):
    monkeypatch.setenv(tc.GATE_ENV, "camera")
    monkeypatch.delenv(wt.WORKING_TRIALS_ENV, raising=False)
    d, ce, cu = _pool()

    def _cg(dd):          # what `_camera_groups` returns: the run selection's working rows (`in_sel`)
        return {"have": {0, 1}, "work_e": tc.in_sel(ce, "working"), "work_u": tc.in_sel(cu, "working")}

    monkeypatch.setattr(pc, "_camera_groups", _cg)
    return monkeypatch, d


def test_camera_in_working_keeps_the_incomplete_position(camera):
    _mp, d = camera
    det = pc.preserved_detail(d, 1)
    assert det["rule"] == "camera_working" and FAR_R in det["keep"] and len(det["keep"]) == 6
    assert det["counts"]["far_R"] == 50 and det["min_post"] == pc.MIN_POST
    assert pc.preserved_positions(d, session=1) == det["keep"]
    assert FAR_R in pc.preserved_positions(d)                       # intersection mode: the same rule


def test_contact_only_drops_it(camera):
    mp, d = camera
    mp.setenv(wt.WORKING_TRIALS_ENV, "in_working_contact")
    det = pc.preserved_detail(d, 1)
    assert det["rule"] == "camera_working" and FAR_R not in det["keep"] and det["counts"]["far_R"] == 5
    mp.setenv(wt.WORKING_TRIALS_ENV, "in_working_jaw")              # jaw-inclusive: still kept (incomplete is working)
    assert FAR_R in pc.preserved_detail(d, 1)["keep"]


def test_daq_rule_drops_it_and_is_unchanged(monkeypatch):
    monkeypatch.setattr(pc, "_camera_groups", lambda dd: None)    # DAQ gate / lick window / no classes
    d, _ce, _cu = _pool()
    det = pc.preserved_detail(d, 1)
    assert det["rule"] == "daq_engaged" and FAR_R not in det["keep"] and det["counts"]["far_R"] == 5
    assert pc.preserved_positions(d, session=1) == [c for c in pc.DISPLAY_ORDER if c != FAR_R]


def test_session_without_camera_classes_falls_back(camera):
    mp, d = camera
    mp.setattr(pc, "_camera_groups", lambda dd: {"have": {0}, "work_e": np.ones(len(d["YE"]), bool),
                                                 "work_u": np.ones(len(d["YU"]), bool)})
    assert pc.preserved_detail(d, 1)["rule"] == "daq_engaged"


def test_post_lick_keep_needs_engaged_rows(camera):
    _mp, d = camera
    pres = pc.preserved_detail(d, 1)
    assert FAR_R in pres["keep"]
    kl = sg.lick_keep({"cue": d, "lick": d}, 1, pres)
    assert FAR_R not in kl and len(kl) == 5
    daq = {"keep": [1, 0], "rule": "daq_engaged"}
    assert sg.lick_keep({"cue": d}, 1, daq) == [1, 0]              # DAQ rule: keep itself


def test_one_arm_scores_post_lick_on_the_lick_keep(monkeypatch):
    seen = {}

    def _rec(name):
        def f(dd, keep, post_all_trials=True):
            seen[(dd["align"], name)] = list(keep)
            return {"ok": 1}
        return f

    monkeypatch.setattr(pc, "decode_matched", _rec("dm"))
    monkeypatch.setattr(pc, "recoding_test", _rec("rt"))
    monkeypatch.setattr(pc, "pattern_similarity", lambda dd, keep, post_all_trials=True: {})
    monkeypatch.setattr(pc, "crossed_confusion", lambda dd, **k: {})
    dba = {a: {"align": a} for a in ("cue", "lick", "precue")}
    keep, kl = [1, 0, 2, 4, 3, 5], [1, 0, 2, 4, 3]
    rec = sg._one_arm(dba, 1, keep, arm_all=False, keep_lick=kl)
    assert seen[("cue", "dm")] == keep and seen[("precue", "dm")] == keep and seen[("lick", "dm")] == kl
    seen.clear()
    rec = sg._one_arm(dba, 1, keep, arm_all=False, keep_lick=[1])
    assert ("lick", "dm") not in seen and "post-lick note" in rec and ("cue", "dm") in seen
