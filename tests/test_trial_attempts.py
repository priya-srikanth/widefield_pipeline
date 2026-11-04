"""scripts/trial_attempts.session_summary: the attempt-rate denominator is the canonical period, the execution
composite (time to contact) and components use attempted trials only, and the lateral error is referenced to the
animal's own August pre."""
import numpy as np
import pandas as pd
import pytest

from scripts import trial_attempts as ta


def _trials(label, pos, rows):
    """rows: (cls, canonical, contact_ms, angle)"""
    out = []
    for i, (cls, can, con, ang) in enumerate(rows):
        att = cls in ta.ATTEMPT_CLASSES
        out.append({"label": label, "trial_id": i, "pos": pos, "cls": cls, "attempted": att, "canonical": can,
                    "contact": con, "n_licks": 1 if cls in ("hit", "incomplete") else 0,
                    "n_nocontact": 1 if cls == "incomplete" else 0, "cam4": 100.0 if att else np.nan,
                    "combined": 90.0 if att else np.nan, "has_cam1": True, "angle_first": ang,
                    "jaw_onset_ms": 60.0 if att else np.nan})
    return out


def test_denominators_and_composite():
    rows = (_trials("PS92_0806", "far_contra", [("hit", True, 200.0, 0.0), ("hit", True, 400.0, 2.0)])
            + _trials("PS92_0901", "far_contra", [("hit", True, 500.0, 10.0), ("incomplete", True, np.nan, 12.0),
                                                  ("no_try", True, np.nan, np.nan),
                                                  ("no_try", False, np.nan, np.nan),     # quit tail: not counted
                                                  ("hit", False, 5000.0, 14.0)]))        # attempted, outside canonical
    W = ta.session_summary(pd.DataFrame(rows)).set_index("label")
    r = W.loc["PS92_0901"]
    assert r.attempt_rate_far_contra == pytest.approx(2 / 3)               # canonical: 2 attempted of 3
    assert r.n_attempted_far_contra == 3                                   # attempted counts outside canonical too
    assert r.ttc_ms_far_contra == pytest.approx((500 + 3500 + 3500) / 3)   # no contact and late contact -> 3.5 s
    assert r.completion_far_contra == pytest.approx(2 / 3)
    assert r.lateral_bias_far_contra == pytest.approx(12.0 - 1.0)          # vs August-pre mean angle 1.0
    assert W.loc["PS92_0806"].lateral_bias_far_contra == pytest.approx(0.0)
    assert r.latency_move_ms_far_contra == pytest.approx(60.0)            # jaw onset precedes the first lick


def test_camera_gated_run_uses_the_runs_camera_classes(monkeypatch, tmp_path):
    """Coordinator 11-04: under the camera gate the classes are the run's (classify_session: end at the last camera
    lick, sustained stop with jaw_only counting (11-04), the 5-min early_run rule) and the attempt-rate denominator is the working
    period (stopped / early_run / after_end / unknown out; jaw_only and no_try in, as non-attempts)."""
    import sys
    import types

    for k in list(__import__("os").environ):
        if k.startswith(("WIDEFIELD_", "WFIELD_")):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("WIDEFIELD_TRIAL_GATE", "camera")
    cls = (["contact"] + ["no_response"] * 6 + ["contact", "jaw_only", "incomplete"]   # a 6-run 400 s in: stopped
           + ["no_response"] * 3)                                                  # after the last lick
    n = len(cls)
    cue = np.array([0.0] + [400.0 + 10 * i for i in range(1, n)])                 # the run starts >= 300 s in
    C = pd.DataFrame({"trial_id": np.arange(n), "cls": cls, "position": "far_contra", "cue_daq": cue,
                      "n_licks": 0, "n_nocontact": 0, "jaw_cam4": np.nan, "jaw_cam1": np.nan, "jaw_at_cue": False,
                      "jaw_onset_ms": np.nan})
    monkeypatch.setattr(ta.TC, "session_trials", lambda lab, window=None: C.copy())
    T0 = pd.DataFrame({"trial_id": np.arange(n), "stop_ms": 3000.0, "pos": "far_contra", "contact": np.nan,
                       "cam4": np.nan, "combined": np.nan, "has_cam1": True, "label": "PS92_0901"})
    monkeypatch.setitem(sys.modules, "scripts.first_lick_latency",
                        types.SimpleNamespace(session_latency=lambda lab: T0.copy()))
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    (tmp_path / "licks.csv").write_text("trial_id,on_ms,peak_angle_deg\n0,100,1.0\n")
    monkeypatch.setattr(oo, "session_dir", lambda *a, **k: tmp_path)
    monkeypatch.setattr(oc, "engagement_keep", lambda *a, **k: (None, "reference"))
    T = ta.session_trials("PS92_0901")
    assert T.cls.tolist() == (["hit"] + ["stopped"] * 6 + ["hit", "jaw_only", "incomplete"] + ["after_end"] * 3)
    assert T.canonical.tolist() == [True] + [False] * 6 + [True] * 3 + [False] * 3
    assert T.gate.iloc[0].startswith("camera:sel-camera") and "cls_raw" in T
    # the 5-min rule: the same run starting < 300 s after the first cue is early_run, not stopped
    C["cue_daq"] = np.arange(n) * 10.0
    T = ta.session_trials("PS92_0901")
    assert T.cls.tolist()[1:7] == ["early_run"] * 6 and not T.canonical.iloc[1:7].any()


def test_camera_attempts_jaw_only_in_a_stop_run_leaves_the_denominator(monkeypatch, tmp_path):
    """Priya 11-04: jaw_only counts toward the 6 and is stopped, so J,NR,J,NR,NR,NR >= 300 s in leaves the attempt-rate
    denominator (canonical False) while cls_raw keeps the trial-level class; a lone jaw_only stays in it as a
    non-attempt. The gate names the stopRun6j selection."""
    import sys
    import types

    for k in list(__import__("os").environ):
        if k.startswith(("WIDEFIELD_", "WFIELD_")):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("WIDEFIELD_TRIAL_GATE", "camera")
    cls = ["contact", "jaw_only", "no_response", "jaw_only", "no_response", "no_response", "no_response",
           "contact", "jaw_only", "contact"]
    n = len(cls)
    cue = np.array([0.0] + [300.0 + 10 * i for i in range(n - 1)])               # the run starts at exactly 300 s
    C = pd.DataFrame({"trial_id": np.arange(n), "cls": cls, "position": "far_contra", "cue_daq": cue,
                      "n_licks": 0, "n_nocontact": 0, "jaw_cam4": np.nan, "jaw_cam1": np.nan, "jaw_at_cue": False,
                      "jaw_onset_ms": np.nan})
    monkeypatch.setattr(ta.TC, "session_trials", lambda lab, window=None: C.copy())
    T0 = pd.DataFrame({"trial_id": np.arange(n), "stop_ms": 3000.0, "pos": "far_contra", "contact": np.nan,
                       "cam4": np.nan, "combined": np.nan, "has_cam1": True, "label": "PS92_0901"})
    monkeypatch.setitem(sys.modules, "scripts.first_lick_latency",
                        types.SimpleNamespace(session_latency=lambda lab: T0.copy()))
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    (tmp_path / "licks.csv").write_text("trial_id,on_ms,peak_angle_deg\n0,100,1.0\n")
    monkeypatch.setattr(oo, "session_dir", lambda *a, **k: tmp_path)
    monkeypatch.setattr(oc, "engagement_keep", lambda *a, **k: (None, "reference"))
    T = ta.session_trials("PS92_0901")
    assert T.cls.tolist() == ["hit"] + ["stopped"] * 6 + ["hit", "jaw_only", "hit"]
    assert T.canonical.tolist() == [True] + [False] * 6 + [True] * 3             # denominator = 4 trials
    assert T.cls_raw.tolist()[1:7] == [ta.TA_NAMES.get(c, c) for c in cls[1:7]]     # the raw jaw_only is kept
    assert not T.attempted.iloc[1:7].any() and not T.attempted.iloc[8]
    assert T.gate.iloc[0].endswith("-stopRun6j")
    # 5 run members (one fewer no_response): not a stop -- every jaw_only / no_response stays in the denominator
    C.loc[6, "cls"] = "contact"
    T = ta.session_trials("PS92_0901")
    assert "stopped" not in T.cls.tolist() and T.canonical.all()
