"""Tests for wfield_local.lick_detection, focused on the min-ILI physiological floor."""
import numpy as np

from wfield_local.lick_detection import detect_licks


def _signal_with_licks(times_s, fs=5000.0, dur_s=3.0, dip_ms=2.0):
    """High-rest (5 V) trace with brief dips to 0 V at each lick time (spout contact)."""
    sig = np.full(int(fs * dur_s), 5.0)
    w = int(fs * dip_ms / 1000.0)
    for t in times_s:
        s = int(t * fs)
        sig[s:s + w] = 0.0
    return sig, fs


def test_min_ili_floor_removes_sub_floor_onsets():
    # licks 100 ms apart (real) plus one 25 ms after a lick (double-detection artifact)
    sig, fs = _signal_with_licks([0.30, 0.325, 0.50, 0.70, 0.90])
    base = detect_licks(sig, fs, 2.5, 0.5, lockout_s=(0.0, 0.0))          # no lockout, no floor
    floored = detect_licks(sig, fs, 2.5, 0.5, lockout_s=(0.0, 0.0), min_ili_s=0.040)
    assert base["lick_onsets"].size == 5
    assert floored["lick_onsets"].size == 4                               # the 25 ms double is dropped
    assert np.min(np.diff(floored["lick_onsets"])) / fs >= 0.040


def test_effective_refractory_is_max_of_floor_and_refractory():
    sig, fs = _signal_with_licks([0.2, 0.26, 0.5, 0.9])   # 60 ms pair
    # floor 40 ms keeps the 60 ms pair; refractory 100 ms would drop it. max() must win.
    d_floor = detect_licks(sig, fs, 2.5, 0.5, lockout_s=(0.0, 0.0), min_ili_s=0.040)
    d_refr = detect_licks(sig, fs, 2.5, 0.5, lockout_s=(0.0, 0.0), refractory_s=0.100)
    d_both = detect_licks(sig, fs, 2.5, 0.5, lockout_s=(0.0, 0.0), refractory_s=0.100, min_ili_s=0.040)
    assert d_floor["lick_onsets"].size == 4
    assert d_both["eff_refractory_s"] == 0.100                            # max(0.04, 0.10)
    assert np.array_equal(d_both["lick_onsets"], d_refr["lick_onsets"])   # coarser refractory subsumes floor


def test_no_floor_is_backward_compatible():
    sig, fs = _signal_with_licks([0.2, 0.5, 0.9])
    d = detect_licks(sig, fs, 2.5, 0.5)
    assert d["min_ili_s"] == 0.0 and d["eff_refractory_s"] == 0.0
    assert d["lick_onsets"].size == 3
