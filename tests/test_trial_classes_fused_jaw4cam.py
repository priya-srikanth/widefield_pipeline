"""trial_classes on the FUSED four-camera lick train (Priya 11-21 Q1 = (b)) and the SOURCE-AWARE close-spout jaw rule
(Q5): at close_center / close_mid the four-camera 3-D jaw call judges the jaw; a two-camera table keeps the 11-03
default."""
import numpy as np
import pandas as pd

from tests.test_trial_classes import CASES, _session
from wfield_local import session_licks as SL
from wfield_local import trial_classes as TC


def _fused_table(rows):
    A = pd.DataFrame(rows, columns=["trial_id", "on_ms", "off_ms", "t_ms"])
    A["lick_key"] = SL.lick_key(A.trial_id, A.t_ms)
    return A


def test_fused_train_replaces_cam4_plus_cam1(tmp_path, monkeypatch):
    v36, c1 = _session(tmp_path)
    # the fused train: trial 2's cam4 lick, trial 6 gains a side-only lick, trial 3's cam1-only lick is gone (not in
    # the train); trial 1 touches the spout (DAQ contact 10 ms inside its span)
    F = _fused_table([(1, 150.0, 220.0, 190.0), (2, 200.0, 260.0, 240.0), (6, 400.0, 450.0, 430.0)])
    monkeypatch.setattr(SL, "session_licks", lambda label, rv=None: F.copy())
    monkeypatch.setattr(TC, "_daq_contact_onsets", lambda a, d: np.array([10.0 + 0.16]))   # trial 1: cue 10 s
    monkeypatch.setattr(TC, "_paths", lambda v, c: (v36, c1))       # the synthetic tree for trials / jaw / cam1 files
    T = TC.session_trials("PS92_0818", licks="fused").set_index("trial_id")
    assert T.loc[6, "cls"] == "incomplete" and T.loc[3, "cls"] == "no_response"   # cam1-only events not added on top
    assert T.loc[4, "cls"] == "incomplete"                     # cam1 "saw a tongue" evidence unchanged
    assert T.loc[2, "n_nocontact"] == 1 and T.loc[1, "n_nocontact"] == 0
    assert (T.licks_source == "fused").all()
    L = TC.session_trials("PS92_0818", v36=v36, c1=c1).set_index("trial_id")
    assert (L.licks_source == "cam4+cam1").all() and L.loc[3, "cls"] == "incomplete"   # legacy unchanged


def _jaw3d(tmp_path, source, moved, at_cue=False):
    v36, c1 = _session(tmp_path)
    qc = tmp_path / "v36" / "imaging_encoding" / "qc"
    J = pd.read_csv(qc / "jaw_onsets.csv")
    J["jaw3d_moved"] = np.where(J.trial_id == 8, moved, np.nan)
    J["jaw3d_onset_ms"] = np.where(J.trial_id == 8, 250.0, np.nan)
    J["jaw3d_at_cue"] = (J.trial_id == 8) & at_cue
    J["jaw3d_source"] = source
    J.to_csv(qc / "jaw_onsets.csv", index=False)
    TC.jaw_table.cache_clear()
    return TC.session_trials("PS92_0818", v36=v36, c1=c1).set_index("trial_id")


def test_close_center_judged_by_four_camera_3d(tmp_path):
    T = _jaw3d(tmp_path, "4cam", 1.0)
    assert T.loc[8, "cls"] == "jaw_only" and T.loc[8, "jaw_judged"] and T.loc[8, "jaw_3d"] == 1
    assert T.loc[8, "jaw_onset_ms"] == 250.0
    still = _jaw3d(tmp_path / "b", "4cam", 0.0)
    assert still.loc[8, "cls"] == "no_response" and still.loc[8, "jaw_judged"]
    at = _jaw3d(tmp_path / "c", "4cam", 1.0, at_cue=True)
    assert at.loc[8, "cls"] == "unknown"                       # moving at the cue, no 3-D re-onset
    others = {t: c[-1] for t, c in CASES.items() if t != 8}
    assert {t: T.loc[t, "cls"] for t in others} == others      # every other spout: the per-camera rule, unchanged


def test_two_camera_3d_call_is_not_used_at_close_center(tmp_path):
    T = _jaw3d(tmp_path, "2cam", 1.0)
    assert T.loc[8, "cls"] == "no_response" and not T.loc[8, "jaw_judged"] and np.isnan(T.loc[8, "jaw_3d"])
