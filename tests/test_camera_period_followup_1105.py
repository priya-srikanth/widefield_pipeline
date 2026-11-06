"""Priya 11-05 ~22:10 (via coordinator), follow-ups of (f) 2:
  (2) epoch family 1b on the CAMERA working period, the DAQ version a labelled robustness row (boundaries stay pinned);
  (3) iti_lick_rate and within_lick_correction cut to the camera working period (no licks after the camera quit / in
      stopped runs; sessions without camera classes left out and listed).

Pins: the one working-period definition (`trial_classes.OUT_OF_WORKING_PERIOD`, shared with trial_attempts'
RUN_EXCLUDED), its cue / trial-id mappings, and each consumer's use of it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wfield_local import trial_classes as tc

REPO = Path(__file__).resolve().parents[1]
CLS = np.array(["contact", "incomplete", "jaw_only", "no_response", "stopped", "stopped_licking", "after_end",
                "unknown"], dtype=object)


def test_working_period_is_one_definition(monkeypatch):
    assert tc.in_working_period(CLS).tolist() == [True] * 4 + [False] * 4
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    import trial_attempts as ta
    assert ta.RUN_EXCLUDED is tc.OUT_OF_WORKING_PERIOD


@pytest.fixture
def session(monkeypatch):
    full = np.arange(len(CLS)) * 10.0
    monkeypatch.setattr(tc, "daq_cue_seconds", lambda lab: full)
    monkeypatch.setattr(tc, "cue_classes_for", lambda lab, cs, w=None: CLS.copy())
    T = pd.DataFrame({"trial_id": np.arange(len(CLS)) + 1, "cue_daq": full + 0.005})
    monkeypatch.setattr(tc, "session_trials", lambda lab, *a, **k: T)
    return full


def test_camera_period_trials_and_trial_ids(session):
    w, c, has = tc.camera_period_trials("PS92_0810", session[[0, 3, 4, 7]] + 0.01)
    assert has and w.tolist() == [True, True, False, False] and c.tolist() == [True, False, False, False]
    w, _c, _h = tc.camera_period_trials("PS92_0810", [999.0])         # no matching cue: outside the period
    assert w.tolist() == [False]
    ids, has = tc.working_trial_ids("PS92_0810")
    assert has and [k for k, v in ids.items() if v] == [1, 2, 3, 4]


def test_no_video_session_has_no_period(monkeypatch):
    monkeypatch.setattr(tc, "daq_cue_seconds", lambda lab: np.arange(3.0))
    monkeypatch.setattr(tc, "cue_classes_for", lambda lab, cs, w=None: np.array(["unknown"] * 3, dtype=object))
    w, _c, has = tc.camera_period_trials("PS93_0813", np.arange(3.0))
    assert not has and not w.any()


# ------------------------------------------------------------------------------------------------ (2) epoch 1b
def test_epoch_1b_camera_metrics_and_records(monkeypatch):
    from wfield_local import epoch_behaviour as eb
    t = pd.DataFrame({"pos_name": ["far_R"] * 4 + ["close_L"] * 2, "pos_idx": [5, 5, 5, 5, 0, 0],
                      "cue_s": np.arange(6.0), "hit": [1, 1, 1, 1, 1, 1], "engaged": True, "block": [0] * 6})
    monkeypatch.setattr(eb, "_trials_table", lambda an, mmdd: t)
    # trials 0-2 working (contact, no_response, contact), 3 stopped (DAQ hit, camera outside), 4-5 working contact
    monkeypatch.setattr(tc, "camera_period_trials", lambda lab, cs: (
        np.array([1, 1, 1, 0, 1, 1], bool), np.array([1, 0, 1, 0, 1, 1], bool), True))
    m = eb._camera_position_metrics("PS92", "0810")
    assert m["far_R"][0] == pytest.approx(2 / 3) and m["far_R"][3] == 3 and m["close_L"][3] == 2
    tt = eb._camera_trials("PS92", "0810")
    assert tt.hit.tolist() == [True, False, True, False, True, True]          # hit = camera contact, not the DAQ lick
    monkeypatch.setattr(tc, "camera_period_trials", lambda lab, cs: (np.zeros(6, bool), np.zeros(6, bool), False))
    eb._NO_CAMERA.clear()
    assert eb._camera_position_metrics("PS93", "0813") == {} and "PS93_0813" in eb._NO_CAMERA


def test_epoch_1b_names_and_dispatch():
    from wfield_local import epoch_behaviour as eb
    assert eb.PERIODS["camera"][0] == "" and eb.PERIODS["daq"][0] == "_daq_quittail"
    assert "ROBUSTNESS" in eb.PERIODS["daq"][1]
    src = (REPO / "wfield_local" / "epoch_grant_figures.py").read_text(encoding="utf-8")
    assert src.count('fig_behaviour(out, period="daq")') == 2
    import inspect
    fb = inspect.getsource(eb.fig_behaviour)
    assert 'name=f"epoch_1b_behaviour_by_position{sfx}"' in fb and "_behaviour_records(period)" in fb
    # the 1c time course (the boundaries' own series) stays DAQ
    assert "_camera" not in inspect.getsource(eb.fig_behaviour_timecourse)


# ------------------------------------------------------------------------------------------------ (3) lick families
def test_iti_lick_rate_counts_the_working_period_only(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    import iti_lick_rate as ilr

    from wfield_local import orofacial_outputs as oo
    T = pd.DataFrame({"trial_id": [1, 2, 3, 4], "cue_s": [10.0, 20.0, 30.0, 40.0], "strobe_s": [7.0, 17.0, 27.0, 37.0],
                      "stop_s": [13.0, 23.0, 33.0, 43.0]})
    monkeypatch.setattr(ilr.il, "_behaviour", lambda lab: type("B", (), {"trials": T})())
    (tmp_path / "licks_all.csv").write_text(
        "trial_id,context,contact\n1,iti,False\n1,iti,True\n2,iti,False\n3,iti,False\n3,iti,False\n4,iti,False\n",
        encoding="utf-8")
    monkeypatch.setattr(oo, "session_dir", lambda *a, **k: tmp_path)
    # trials 1-2 working, 3-4 after the camera quit
    monkeypatch.setattr(tc, "camera_period_trials", lambda lab, cs: (np.array([1, 1, 0, 0], bool), None, True))
    monkeypatch.setattr(tc, "working_trial_ids", lambda lab: ({1: True, 2: True, 3: False, 4: False}, True))
    r = ilr.session_row("PS92_0810")
    assert r["n_iti"] == 3 and r["n_trials"] == 2                      # licks after trials 3 / 4 are gone
    assert r["iti_min"] == pytest.approx((4.0 + 4.0) / 60.0)          # gaps after trials 1 and 2 only
    monkeypatch.setattr(tc, "camera_period_trials", lambda lab, cs: (np.zeros(4, bool), None, False))
    assert ilr.session_row("PS93_0813") is None                        # no video: left out


def test_within_lick_correction_drops_licks_outside_the_period(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    import within_lick_correction as wlc
    monkeypatch.setattr(wlc, "LICKS", tmp_path)
    pd.DataFrame({"sid": "PS92_20260810", "trial_id": [1, 1, 2, 3], "context": ["post_cue", "iti", "post_cue", "iti"],
                  "x": [1, 2, 3, 4]}).to_pickle(tmp_path / "PS92_20260810.pkl")
    pd.DataFrame({"sid": "PS93_20260813", "trial_id": [1], "context": ["post_cue"], "x": [9]}).to_pickle(
        tmp_path / "PS93_20260813.pkl")
    ids = {"PS92_0810": ({1: True, 2: False, 3: True}, True), "PS93_0813": ({}, False)}
    monkeypatch.setattr(tc, "working_trial_ids", lambda lab: ids[lab])
    wlc._NO_CAMERA.clear()
    L = wlc.load_licks(["PS92_20260810", "PS93_20260813"], ["sid", "x"])
    assert L.x.tolist() == [1, 2, 4] and list(L.columns) == ["sid", "x"]
    assert wlc._NO_CAMERA == {"PS93_20260813"}
    sys.modules.pop("within_lick_correction", None)
