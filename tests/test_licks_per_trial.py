"""Licks per trial (Priya 11-03): TOTAL licks from cue + 70 ms to the TRIAL STOP; lick rate from the inter-lick intervals.
The trial stop is the DAQ trial_end TTL, else the task rule: first contact + the response window (miss: cue + window)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from wfield_local import daq_trials, trial_classes

REPO = Path(__file__).resolve().parents[1]


def test_contact_offset_matches_the_camera_classes():
    assert daq_trials.CONTACT_FROM_CUE_S == trial_classes.CONTACT_FROM_CUE_S


def test_trial_stops_prefers_the_ttl_and_falls_back_to_the_task_rule():
    cue = np.array([0.0, 10.0, 20.0, 30.0])
    lick = np.array([0.03, 0.5, 1.0, 12.0, 12.2, 31.2])          # 0.03 s is before cue + 70 ms: not the first contact
    ttl = np.array([3.9, 33.0])                                   # trials 1 and 4 have a trial_end
    stop, src = daq_trials.trial_stops(cue, lick, 3.5, ttl)
    assert src.tolist() == ["daq_trial_end", "rule", "rule", "daq_trial_end"]
    assert np.allclose(stop, [3.9, 12.0 + 3.5, 20.0 + 3.5, 33.0])  # hit: first contact + 3.5; miss: cue + 3.5
    stop2, _ = daq_trials.trial_stops(cue, lick, 3.5, None)
    assert np.isclose(stop2[0], 0.5 + 3.5)                        # first contact from cue + 70 ms


def test_licks_to_stop_counts_from_cue_plus_70ms_to_the_stop():
    cue = np.array([0.0, 10.0])
    lick = np.array([0.03, 0.2, 1.0, 3.9, 4.1, 10.5])
    n = daq_trials.licks_to_stop(cue, np.array([4.0, 13.5]), lick)
    assert n.tolist() == [3, 1]                                   # 0.03 (< 70 ms) and 4.1 (after the stop) excluded


def test_readouts_use_the_trial_window_and_ili_rate():
    sb = (REPO / "wfield_local" / "spout_behavior.py").read_text(encoding="utf-8")
    assert "c + daq_trials.CONTACT_FROM_CUE_S) & (all_s <= stop)" in sb
    assert "1.0 / np.median(np.diff(post))" in sb                 # rate from ILIs, not count / duration
    for f in ("quit_prodrome", "lick_bout_structure"):
        src = (REPO / "scripts" / "rest_migration" / f"{f}.py").read_text(encoding="utf-8")
        assert 'r["stop_sample"]' in src or 'n_licks_trial' in src, f
    qp = (REPO / "scripts" / "rest_migration" / "quit_prodrome.py").read_text(encoding="utf-8")
    assert 'float(tt[i]["n_licks_trial"])' in qp and 'tt[i + 1]["cum_licks"]' not in qp


def test_gui_fallback_session_counts_to_the_sessions_own_window(tmp_path):
    """Fifth review (a): a GUI-log (no DAQ) session ends each trial at first contact + ITS OWN response window
    (gui_config.json 3500 ms), not the yaml default 2.0 s. A lick 3.0 s after a 0.1 s first lick counts."""
    import json

    import pandas as pd
    import sys
    sys.path.insert(0, str(REPO / "tests"))
    from test_spout_behavior import _events, _write_session

    from wfield_local import config
    from wfield_local import spout_behavior as sb
    ev = _events({1: {"cue_ms": 1000, "licks_ms": [1100, 1200, 4100]}, 2: {"cue_ms": 9000, "licks_ms": [9100]}})
    d = _write_session(tmp_path, "PS92_20260806_120000",
                       [dict(tid=1, pos_idx=1, hit=True), dict(tid=2, pos_idx=1, hit=True)], ev)
    (d / "gui_config.json").write_text(json.dumps({"timing": {"response_window": "3500"}}))
    m = sb.lick_microstructure(d, sb.load_trials(d), config.defaults()["behavior"])
    per = m["per_position"].set_index("pos_name")
    assert per.loc["close_L", "licks_per_trial"] == pd.Series([3, 1]).mean()   # the 4.1 s lick counts (old: 2 + 1)
