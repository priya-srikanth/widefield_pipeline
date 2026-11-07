"""scripts/stopped_trial_traces.py on the CAMERA trial gate (Priya 11-07, option a).

Until 11-07 the script refused under ``WIDEFIELD_TRIAL_GATE=camera`` (`trial_classes.require_gate("daq", ...)`), because
its working / stopped states came from the DAQ quit-tail rule (`engagement_gate`). Now, under camera, `_states` takes the
camera class of every DAQ cue from `trial_classes.cue_classes_for` (full cue train; the stop-run rule) and the states from
the wired `enl_states.states_for(cam_cls=...)`; under daq nothing changes. Synthetic sessions only (no data read).
"""
from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from wfield_local import trial_classes as TC  # noqa: E402

N = 60
SR = 1000.0


def _raw():
    """Raw camera classes along a 60-cue session (10 s apart, all at a reference spout): working, a mid-session run of
    6 no_response (starts 300 s after the first cue: a stop run), working, one isolated jaw_only (neither), working,
    then the quit tail after the last camera lick."""
    raw = np.array(["contact", "incomplete"] * 15 + ["no_response"] * 6 + ["contact"] * 10 + ["jaw_only"]
                   + ["contact", "incomplete", "contact", "contact"] + ["no_response"] * 9, dtype=object)
    assert raw.size == N
    return raw


@pytest.fixture
def stt(monkeypatch):
    for k in list(os.environ):
        if k.startswith(("WIDEFIELD_", "WFIELD_")):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv(TC.STOP_LICKS_ENV, "none")      # no span-lick table (needs data); the run / tail rule itself
    import stopped_trial_traces as m
    return m


def _fake_session(monkeypatch, raw, *, not_eng=None, label="SYN1_0801"):
    """Patch every data reader `_states` uses with a synthetic session; returns the cue times (s)."""
    from scripts.rest_migration import engagement_decomposition as ed
    from wfield_local import config, daq_io, locanmf_cue_lick_analysis, nolick_decoder, precue_engagement_states
    cs = np.arange(N) * 10.0 + 5.0
    hit = raw == "contact"
    tr = [{"order": i, "hit": bool(hit[i]), "latency_s": 0.4 if hit[i] else None, "pos": 0} for i in range(N)]
    monkeypatch.setattr(config, "load_sessions", lambda *a, **k: [{"label": label, "h5": "syn.h5"}])
    monkeypatch.setattr(nolick_decoder, "response_window_for", lambda s: (0.0, 3.5))
    monkeypatch.setattr(ed, "session_trials", lambda s, rw: tr)
    monkeypatch.setattr(locanmf_cue_lick_analysis, "_load_cue_events", lambda h5: {"cue_samples": cs * SR})
    monkeypatch.setattr(daq_io, "open_daq", lambda h5: contextlib.nullcontext(None))
    monkeypatch.setattr(daq_io, "session_attrs", lambda f: (SR, None))
    if not_eng is None:
        def _refuse(*a, **k):
            raise AssertionError("engagement_gate called on the camera path")
        monkeypatch.setattr(precue_engagement_states, "engagement_gate", _refuse)
    else:
        monkeypatch.setattr(precue_engagement_states, "engagement_gate", lambda o, h, p: not_eng)
    table = pd.DataFrame({"cue_daq": cs, "cls": raw, "position": ["close_L"] * N, "trial_id": np.arange(N)})
    monkeypatch.setattr(TC, "session_trials", lambda lab, window=None, **k: table)
    TC._CUE_CLS.clear()
    TC._CUE_KIND.clear()
    return cs


def test_camera_path_selects_the_trial_classes_sets(stt, monkeypatch):
    monkeypatch.setenv(TC.GATE_ENV, "camera")
    raw = _raw()
    cs = _fake_session(monkeypatch, raw)
    S = stt._states("SYN1_0801")
    cls = np.asarray(TC.cue_classes_for("SYN1_0801", cs), dtype=object)          # THE classes (independent call)
    assert S.camcls.tolist() == list(cls)
    hit = raw == "contact"
    stopped = ~hit & np.isin(cls, list(TC.STOPPED_CLASSES))
    working = ~hit & TC.in_sel(cls, "miss_working")
    assert np.array_equal(S.state.to_numpy() == "stopped", stopped)
    assert np.array_equal(S.state.to_numpy() == "working", working)
    assert np.array_equal(S.state.to_numpy() == "success", hit)
    # the synthetic session's expected sets: the mid-session run AND the quit tail are stopped; the isolated jaw_only is
    # in neither (its own class); incomplete trials before the last camera lick are miss_working
    assert np.flatnonzero(stopped).tolist() == list(range(30, 36)) + list(range(51, 60))
    assert S.state[46] == "other" and S.camcls[46] == "jaw_only"
    assert np.flatnonzero(working).tolist() == [i for i in range(N) if raw[i] == "incomplete"]
    assert TC.in_working_period(cls[working]).all()


def test_camera_states_unit(stt, monkeypatch):
    monkeypatch.setenv(TC.GATE_ENV, "camera")
    cls = np.array(["contact", "incomplete", "jaw_only", "no_response", "stopped", "stopped_licking", "after_end",
                    "unknown", "stopped"], dtype=object)
    hit = np.array([1, 0, 0, 0, 0, 0, 0, 0, 1], bool)              # a DAQ hit is never stopped (one definition)
    st = stt.camera_states(hit, cls)
    assert st.tolist() == ["success", "working", "other", "other", "stopped", "other", "other", "other", "success"]
    assert stt._nolick(st).tolist() == [False, True, True, True, True, True, True, True, False]


def test_daq_path_unchanged(stt, monkeypatch):
    monkeypatch.setenv(TC.GATE_ENV, "daq")
    raw = _raw()
    not_eng = np.r_[np.zeros(50, bool), np.ones(10, bool)]
    _fake_session(monkeypatch, raw, not_eng=not_eng)

    def _no_camera(*a, **k):
        raise AssertionError("camera classes read on the daq path")
    monkeypatch.setattr(TC, "cue_classes_for", _no_camera)
    S = stt._states("SYN1_0801")
    hit = raw == "contact"
    exp = np.where(hit, "success", np.where(not_eng, "stopped", "working"))
    assert S.state.tolist() == exp.tolist()
    assert list(S.columns) == ["order", "frac", "cue_daq", "state"]
    # the class masks / no-lick set read exactly as before on DAQ states
    D = pd.DataFrame({"state": exp, "cls": np.where(raw == "incomplete", "incomplete", "no_try")})
    M = stt.class_masks(D)
    assert np.array_equal(M["jaw_nolick"].to_numpy(), (D.state != "success").to_numpy() & (D.cls == "incomplete").to_numpy())
    assert stt.class_def() == stt.CLASS_DEF


def test_refusal_removed_and_registered_wired():
    src = (REPO / "scripts" / "stopped_trial_traces.py").read_text(encoding="utf-8")
    assert "require_gate(" not in src
    reg = (REPO / "tests" / "test_camera_gate_fail_closed.py").read_text(encoding="utf-8")
    assert '"scripts/stopped_trial_traces.py": "wired"' in reg


def test_camera_caption_names_the_gate(stt, monkeypatch):
    monkeypatch.setenv(TC.GATE_ENV, "camera")
    txt = stt.class_def()
    assert "camera trial gate" in txt and TC.stopped_rule_name() in txt and "engagement_gate" not in txt
    D = pd.DataFrame({"animal": ["PS92"] * 4, "epoch": ["pre"] * 4, "state": ["success", "working", "stopped", "other"],
                      "camcls": ["contact", "incomplete", "stopped", "jaw_only"]})
    T = stt.state_counts(D)
    assert T.loc[("PS92", "pre")].to_dict() == {"other": 1, "stopped": 1, "success": 1, "working": 1,
                                               "other:jaw_only": 1}
