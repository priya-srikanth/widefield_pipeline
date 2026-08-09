"""Tests for the DAQ-sourced trial table (wfield_local.daq_trials).

These exercise the decode -> quality-gate -> score path on synthetic event streams (no h5 needed):
the pairing rule, the guards that send a degraded session back to the behavior log, and the
per-session response window that the scoring depends on.
"""
import json

import numpy as np
import pytest

from wfield_local import daq_trials as dt


def _dec(cue_s, strobe_s, codes, lick_s=(), reward_s=(), trial_start_s=None, fs=5000.0):
    return {"fs": fs, "cue_s": np.asarray(cue_s, float), "strobe_s": np.asarray(strobe_s, float),
            "codes": np.asarray(codes, int),
            "trial_start_s": np.asarray(cue_s if trial_start_s is None else trial_start_s, float),
            "lick_s": np.asarray(lick_s, float), "reward_s": np.asarray(reward_s, float), "h5": "x"}


# --------------------------------------------------------------------------- pairing
def test_positions_pair_with_last_strobe_before_cue():
    # strobe precedes its cue by ~3 s (firmware: move, emit code, then the ENL, then the cue)
    d = _dec(cue_s=[10.0, 20.0, 30.0], strobe_s=[7.0, 17.0, 27.0], codes=[4, 1, 5])
    assert dt.positions_for_cues(d).tolist() == [4, 1, 5]


def test_extra_non_trial_strobe_does_not_shift_positions():
    """A manual/commanded move (moveToNamedPosition) also emits a code — index pairing would
    shift every trial by one; time pairing must not."""
    d = _dec(cue_s=[10.0, 20.0], strobe_s=[1.0, 7.0, 17.0], codes=[3, 4, 1])   # 1.0 = pre-session move
    assert dt.positions_for_cues(d).tolist() == [4, 1]
    ok, reason = dt.quality(d, min_distinct_positions=2)
    assert ok and "1 non-trial strobe" in reason


def test_cue_with_no_preceding_strobe_is_unpaired():
    d = _dec(cue_s=[5.0, 20.0], strobe_s=[17.0], codes=[2])
    assert dt.positions_for_cues(d).tolist() == [-1, 2]
    ok, reason = dt.quality(d)
    assert not ok and "no preceding strobe" in reason


# --------------------------------------------------------------------------- quality gate
def test_quality_rejects_dead_strobe_bit():
    """Aug-2026 dead spout_bit1: codes collapse onto {0,1,4,5} -> must fall back to the log."""
    d = _dec(cue_s=np.arange(10) * 10.0 + 10, strobe_s=np.arange(10) * 10.0 + 7,
             codes=[0, 1, 4, 5, 0, 1, 4, 5, 0, 1])
    ok, reason = dt.quality(d)
    assert not ok and "4 distinct positions" in reason


def test_quality_accepts_all_six_positions():
    d = _dec(cue_s=np.arange(6) * 10.0 + 10, strobe_s=np.arange(6) * 10.0 + 7, codes=[0, 1, 2, 3, 4, 5])
    ok, reason = dt.quality(d)
    assert ok and "6 positions" in reason


def test_quality_rejects_empty_streams():
    assert dt.quality(_dec([], [], []))[0] is False
    assert dt.quality(_dec([10.0], [], []))[0] is False


# --------------------------------------------------------------------------- scoring
def test_build_trials_scores_hit_miss_latency_and_anticipatory():
    d = _dec(cue_s=[10.0, 20.0], strobe_s=[7.0, 17.0], codes=[1, 4],
             trial_start_s=[8.0, 18.0],
             #      pre-cue (ENL) licks: 8.5, 9.0 -> trial 1 anticipatory=2
             #      trial 1 response at +0.25 s (hit);  trial 2: none -> miss
             lick_s=[8.5, 9.0, 10.25, 10.4], reward_s=[10.5])
    t = dt.build_trials(d, response_window=3.5)
    assert t["pos_idx"].tolist() == [1, 4]
    assert t["pos_name"].tolist() == ["close_L", "far_L"]
    assert t["hit"].tolist() == [1, 0] and t["miss"].tolist() == [0, 1]
    assert t["latency_s"].iloc[0] == pytest.approx(0.25)
    assert np.isnan(t["latency_s"].iloc[1])
    assert t["n_licks_post"].tolist() == [2, 0]
    assert t["n_licks_pre"].tolist() == [2, 0]
    assert t["reward_delivered"].tolist() == [1, 0]
    assert t["responded"].tolist() == [True, False]
    assert (t["source"] == "DAQ").all()


def test_response_window_capped_at_next_cue():
    """A lick after the next cue can never be credited to the previous trial, even inside the window."""
    d = _dec(cue_s=[10.0, 11.0], strobe_s=[7.0, 8.0], codes=[1, 2], lick_s=[11.5])
    t = dt.build_trials(d, response_window=3.5)     # 11.5 is within 10+3.5 but past the next cue
    assert t["hit"].tolist() == [0, 1]


def test_lick_train_rides_along_for_the_figures():
    d = _dec(cue_s=[10.0], strobe_s=[7.0], codes=[1], lick_s=[10.1, 10.2])
    t = dt.build_trials(d, response_window=3.5)
    assert t.attrs["lick_s"].tolist() == [10.1, 10.2]      # no events.csv / sync fit needed
    assert t.attrs["fs"] == 5000.0


# --------------------------------------------------------------------------- response window
def _cfg(tmp_path, payload):
    d = tmp_path / "PS92_20260806_120000"
    d.mkdir(exist_ok=True)
    (d / "gui_config.json").write_text(json.dumps(payload))
    return d


def test_response_window_read_from_gui_config_timing(tmp_path):
    """The GUI writes it in ms, as a STRING, under `timing` — sessions to date ran 3500 ms."""
    d = _cfg(tmp_path, {"timing": {"response_window": "3500", "precue_min": "2000"}})
    assert dt.response_window_s(d, default=2.0) == (3.5, "gui_config.json:timing")


def test_response_window_accepts_cue_section(tmp_path):
    d = _cfg(tmp_path, {"cue": {"response_window": 1500}})
    assert dt.response_window_s(d, default=2.0) == (1.5, "gui_config.json:cue")


def test_response_window_falls_back_to_default(tmp_path):
    d = _cfg(tmp_path, {"timing": {"precue_min": "2000"}})      # no response_window
    assert dt.response_window_s(d, default=2.0) == (2.0, "default")
    assert dt.response_window_s(tmp_path / "nonexistent", default=2.0) == (2.0, "default")
