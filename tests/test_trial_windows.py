"""trial_windows: post-cue windows end at the trial's own stop; nothing changes without one."""
import numpy as np
import pandas as pd

from wfield_local import jaw_kinematics as jk
from wfield_local import tongue_kinematics as tk
from wfield_local import trial_windows as TW


def test_end_at_stop_moves_response_and_slack_ends_only():
    p = tk.DEFAULTS
    q = TW.end_at_stop(p, 3700.0)
    for k in ("lick12_detect_win_ms", "peak_velocity_detect_win_ms", "lick_count_apply_win_ms",
              "licking_dyn_apply_win_ms", "angle_apply_win_ms"):
        assert q[k] == [p[k][0], 3700.0]
    assert q["bout_table"]["detect_win_ms"] == [p["bout_table"]["detect_win_ms"][0], 3700.0]
    assert q["trial_slice_win_ms"] == [p["trial_slice_win_ms"][0], 6700.0]
    assert q["lick_count_detect_win_ms"][1] == 6700.0
    assert q["lick_count_bin_win_ms"] == p["lick_count_bin_win_ms"]        # bins stay fixed
    assert p["lick12_detect_win_ms"] == [70.0, 3000.0]                      # original untouched


def test_no_stop_returns_same_dict():
    assert TW.end_at_stop(tk.DEFAULTS, None) is tk.DEFAULTS
    assert TW.end_at_stop(tk.DEFAULTS, float("nan")) is tk.DEFAULTS
    assert TW.stop_ms_of(100.0, None, 250.0) is None
    assert TW.stop_ms_of(100.0, 1025.0, 250.0) == 3700.0


def test_jaw_detect_window_ends_at_stop():
    fps, n = 250.0, 4000
    y = np.zeros(n)
    y[1000 - 125:1000] = np.random.default_rng(0).normal(0, 0.5, 125)    # baseline noise
    y[1000 + 1000:1000 + 1100] = 30.0                                    # jaw opens 4.0-4.4 s after the cue
    base = pd.DataFrame({"trial_id": [1], "cue_frame": [1000.0], "position": ["far_L"]})
    t_fixed, _ = jk.jaw_pertrial(y, fps, base)                            # ported 0-5000 ms: sees it
    t_stop, _ = jk.jaw_pertrial(y, fps, base.assign(stop_frame=1000.0 + 3.7 * fps))   # stop at 3.7 s: does not
    assert bool(t_fixed.jaw_pass_qc.iloc[0]) and not bool(t_stop.jaw_pass_qc.iloc[0])


def test_trial_carries_optional_bounds():
    df = pd.DataFrame({"trial_id": [1, 2], "cue_frame": [10.0, 20.0], "position": ["far_L", "far_R"],
                       "stop_frame": [935.0, np.nan]})
    tr = tk.trials_from_frame(df)
    assert tr[0].stop_frame == 935.0 and tr[1].stop_frame is None and tr[0].strobe_frame is None
