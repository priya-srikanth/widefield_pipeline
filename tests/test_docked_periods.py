"""The docked window: dock -> next trial_start, measured from the log or reconstructed per position.

The rules pinned here are the ones the module's docstrings say would manufacture a position effect
if broken: a trial with an unknown position gets NO window (never a mean offset), the window opens
at the PER-POSITION p95 offset, a dock with no later trial_start is dropped, and fewer than 20
windows returns None rather than a thin mask.
"""
import numpy as np
import pandas as pd

from wfield_local import docked_periods as dp

FS = 5000.0


def _daq_trials(n=30, period=8.0):
    cue = (np.arange(n) * period + 2.0) * FS
    ts = (np.arange(n + 1) * period + 7.5) * FS          # next trial_start 5.5 s after each cue
    return cue.astype(np.int64), ts.astype(np.int64), int((n + 2) * period * FS)


def test_reconstructed_uses_per_position_offset():
    cue, ts, ns = _daq_trials()
    codes = np.arange(cue.size) % 6
    m = dp.docked_mask_reconstructed(cue, codes, ts, ns, fs=FS)
    assert m is not None
    for k in range(6):
        t0 = cue[k] + round(dp._DOCK_AFTER_CUE_P95[k] * FS)
        assert not m[t0 - 1] and m[t0], f"position {k}: window must open at its own p95 offset"
        assert m[ts[k] - 1] and not m[ts[k]], "window must close at the next trial_start"


def test_unknown_position_gets_no_window():
    cue, ts, ns = _daq_trials()
    codes = np.arange(cue.size) % 6
    codes[0] = -1
    m = dp.docked_mask_reconstructed(cue, codes, ts, ns, fs=FS)
    assert not m[cue[0]:ts[0]].any(), "an unknown position must be skipped, not given a mean offset"
    assert m[cue[1]:ts[1]].any()


def test_too_few_windows_returns_none():
    cue, ts, ns = _daq_trials(n=19)
    assert dp.docked_mask_reconstructed(cue, np.zeros(19, int), ts, ns, fs=FS) is None


def test_any_without_codes_has_no_fallback():
    cue, ts, ns = _daq_trials()
    assert dp.docked_mask_any(None, [], cue, None, ts, ns, fs=FS) == (None, None)


def _events_csv(d, n=30, period=8.0, lag=3.0):
    """A GUI log on a device clock offset `lag` s from the DAQ, with sync pulses every 1 s."""
    rows = []
    for k in range(n):
        rows.append((k, "dock", (k * period + 6.7 + lag) * 1000))
        rows.append((k, "dock", (k * period + 7.0 + lag) * 1000))      # retried move: first one wins
        rows.append((k, "trial_start", (k * period + 7.5 + lag) * 1000))
    sync_daq = np.arange(0, n * period, 1.0)
    for t in sync_daq:
        rows.append((np.nan, "sync", (t + lag) * 1000))
    pd.DataFrame(rows, columns=["trial_id", "event_name", "device_t_ms"]).to_csv(d / "events.csv", index=False)
    return (sync_daq * FS).astype(np.int64), int((n + 2) * period * FS)


def test_log_mask_maps_device_clock_to_daq(tmp_path):
    sync, ns = _events_csv(tmp_path)
    m = dp.docked_mask(tmp_path, sync, ns, fs=FS)
    assert m is not None
    # trial 0: first dock at DAQ 6.7 s; next trial_start AFTER it is trial 0's at 7.5 s
    a, b = round(6.7 * FS), round(7.5 * FS)
    assert not m[a - 2] and m[a + 2] and m[b - 2] and not m[b + 2]


def test_any_prefers_log_then_reconstructs(tmp_path):
    sync, ns = _events_csv(tmp_path)
    cue, ts, _ = _daq_trials()
    m, src = dp.docked_mask_any(tmp_path, sync, cue, np.arange(cue.size) % 6, ts, ns, fs=FS)
    assert src == "log" and m is not None
    m, src = dp.docked_mask_any(tmp_path / "missing", sync, cue, np.arange(cue.size) % 6, ts, ns, fs=FS)
    assert src == "reconstructed" and m is not None


def test_missing_events_csv_is_none(tmp_path):
    assert dp.dock_events(tmp_path) is None
