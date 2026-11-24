"""`frame_coverage`: per-trial camera frame coverage on the UNION of the contributing cameras (coordinator 11-22): one
camera missing frames does not gate a trial; frames missing in every camera (PS92 0908) do."""
import numpy as np
import pandas as pd

from wfield_local import frame_coverage as FC


def _frames(t0, t1, fps=249.81, drop=None, phase=0.0):
    t = np.arange(t0 + phase / fps, t1, 1.0 / fps)
    if drop is not None:
        t = t[~drop(t)]
    return t


def test_threshold_is_a_named_constant():
    assert FC.MIN_FRAME_COVERAGE == 0.90 and FC.GATE_REASON == "frame_coverage<0.90"


def test_trial_windows_match_the_camera_window_rule():
    b = pd.DataFrame({"trial_id": [2, 1], "cue_s": [20.0, 10.0], "strobe_s": [17.0, np.nan], "stop_s": [23.5, 13.5]})
    W = FC.trial_windows_s(b)
    assert W.trial_id.tolist() == [1, 2]
    np.testing.assert_allclose(W.t0_s, [10.0 - 1.0, 17.0 - 0.5])          # cue - 0.5 - 0.5; strobe - 0.5
    np.testing.assert_allclose(W.t1_s, [16.5, 26.5])                       # stop + 3 s, cut at the next start


def test_union_coverage_tolerates_one_camera_but_gates_simultaneous_drops():
    win = pd.DataFrame({"trial_id": [1, 2, 3], "t0_s": [0.0, 10.0, 20.0], "t1_s": [8.0, 18.0, 28.0]})
    lost_all = lambda t: (t > 22.0)                                        # noqa: E731  trial 3: every camera drops
    ft = {"cam4": _frames(0, 30, drop=lambda t: ((t > 2) & (t < 7)) | lost_all(t)),   # trial 1: cam4 alone drops 60 %
          "cam1": _frames(0, 30, phase=0.3, drop=lost_all),
          "cam2": _frames(0, 30, phase=0.6, drop=lambda t: (t > 12) & (t < 17) | lost_all(t)),
          "cam3": _frames(0, 30, phase=0.9, drop=lost_all)}
    T = FC.coverage_table(win, ft)
    assert T.n_cams.iloc[0] == 4 and T.n_expected.tolist() == [2000, 2000, 2000]
    assert T.cov_cam4.iloc[0] < 0.45 and T.cov_cam2.iloc[1] < 0.45          # one camera's own coverage is recorded
    assert (T.cov_union.iloc[:2] > 0.99).all()                             # ... but the union still covers the trial
    assert T.gated.tolist() == [False, False, True] and T.cov_union.iloc[2] < 0.3
    assert T.gate_reason.tolist() == ["", "", FC.GATE_REASON]


def test_no_camera_means_gated():
    win = pd.DataFrame({"trial_id": [1], "t0_s": [0.0], "t1_s": [1.0]})
    T = FC.coverage_table(win, {"cam4": np.array([])})
    assert bool(T.gated.iloc[0]) and T.cov_union.iloc[0] == 0.0
