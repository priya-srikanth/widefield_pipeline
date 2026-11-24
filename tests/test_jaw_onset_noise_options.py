"""`jaw_kinematics.jaw_onsets` pre-cue noise options (CC-B 11-23, Priya "let's try"): defaults = the production rule;
"mad" = median + 1.4826 MAD; ``noise_cap`` caps the 4 x noise term (the floor still applies)."""
import numpy as np
import pandas as pd

from wfield_local import jaw_kinematics as jk

FPS = 250.0


def _trace(spike_pre: bool, amp: float):
    """One trial: 0.5 s pre-cue noise (sd 1), optional brief pre-cue movement, then a step of ``amp`` at +200 ms."""
    rng = np.random.default_rng(0)
    pre, post = 250, 500
    y = rng.normal(0, 1.0, pre + post)
    if spike_pre:
        y[150:170] += 40.0                     # a brief movement in the baseline window inflates the sd
    y[pre + 50:] += amp
    trials = pd.DataFrame({"trial_id": [1], "position": ["far_L"], "cue_frame": [float(pre)],
                           "stop_frame": [float(pre + post - 1)]})
    return y, trials


def test_default_unchanged_by_new_keys():
    y, t = _trace(False, 12.0)
    a = jk.jaw_onsets(y, FPS, t, {"deflection_threshold_min_px": 4.0})
    b = jk.jaw_onsets(y, FPS, t, {"deflection_threshold_min_px": 4.0, "baseline_noise": "sd", "noise_cap": None})
    pd.testing.assert_frame_equal(a, b)
    assert a.jaw_pass_qc.iloc[0] == 1


def test_mad_and_cap_recover_a_movement_after_a_noisy_baseline():
    y, t = _trace(True, 20.0)          # 20: above cap 10 + the spike-shifted mean (~6), below 4 sd (~59)
    sd = jk.jaw_onsets(y, FPS, t, {"deflection_threshold_min_px": 4.0})
    mad = jk.jaw_onsets(y, FPS, t, {"deflection_threshold_min_px": 4.0, "baseline_noise": "mad"})
    cap = jk.jaw_onsets(y, FPS, t, {"deflection_threshold_min_px": 4.0, "noise_cap": 10.0})
    assert sd.jaw_pass_qc.iloc[0] == 0               # the pre-cue spike sets 4 sd above the step
    assert mad.jaw_pass_qc.iloc[0] == 1 and cap.jaw_pass_qc.iloc[0] == 1


def test_onset_threshold():
    p = {"sd_mult": 4.0, "min_px": 4.0, "noise_cap": None}
    assert jk.onset_threshold(0.5, p) == 4.0 and jk.onset_threshold(3.0, p) == 12.0
    assert jk.onset_threshold(3.0, {**p, "noise_cap": 10.0}) == 10.0
    assert jk.onset_threshold(0.5, {**p, "noise_cap": 2.0}) == 4.0   # the floor still applies
