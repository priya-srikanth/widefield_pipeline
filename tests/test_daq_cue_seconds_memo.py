"""daq_cue_seconds is memoized per (label, input environment) and hands every caller its own copy (profile 11-08)."""
import numpy as np

from wfield_local import config, trial_classes as tc
import wfield_local.plot_spout_trial_averages as psta


def _fake(monkeypatch):
    calls = []
    monkeypatch.setattr(config, "load_sessions", lambda: [{"label": "PSX_0101", "h5": "x.h5"}])

    def load(h5):
        calls.append(h5)
        return {"cue_samples": np.array([1000, 3000]), "sample_rate_hz": 1000.0}
    monkeypatch.setattr(psta, "_load_daq_events", load)
    tc._daq_cue_seconds_cached.cache_clear()
    return calls


def test_loaded_once_and_equal(monkeypatch):
    calls = _fake(monkeypatch)
    a = tc.daq_cue_seconds("PSX_0101")
    b = tc.daq_cue_seconds("PSX_0101")
    assert len(calls) == 1
    np.testing.assert_array_equal(a, [1.0, 3.0])
    np.testing.assert_array_equal(a, b)
    tc._daq_cue_seconds_cached.cache_clear()


def test_caller_edits_do_not_leak(monkeypatch):
    _fake(monkeypatch)
    a = tc.daq_cue_seconds("PSX_0101")
    a[:] = -1
    np.testing.assert_array_equal(tc.daq_cue_seconds("PSX_0101"), [1.0, 3.0])
    tc._daq_cue_seconds_cached.cache_clear()


def test_env_change_reloads(monkeypatch):
    calls = _fake(monkeypatch)
    tc.daq_cue_seconds("PSX_0101")
    monkeypatch.setenv("WIDEFIELD_COHORT", "cohort2")
    tc.daq_cue_seconds("PSX_0101")
    assert len(calls) == 2
    tc._daq_cue_seconds_cached.cache_clear()
