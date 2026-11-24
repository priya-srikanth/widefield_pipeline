"""`trial_classes.jaw_call`: one jaw definition, three modes (11-23). classes = 2-D per camera, 4-cam 3-D at the
hidden spout; 3d = 4-cam 3-D everywhere (at-cue 3-D = unknown); 2d = per-camera calls only."""
import numpy as np
import pandas as pd
import pytest

from wfield_local import trial_classes as TC


def _j():
    # rows: far 2-D moved; close_center (2-D masked upstream); far with 3-D already moving at the cue; far 2-cam 3-D
    return pd.DataFrame({
        "cam4_moved": [1, np.nan, 1, 1], "cam1_moved": [np.nan, np.nan, np.nan, 0],
        "jaw_onset_ms": [100.0, np.nan, 50.0, 70.0], "onset_at_cue": [False, False, False, False],
        "jaw_reonset_ms": [np.nan] * 4,
        "jaw3d_moved": [1, 1, 1, 1], "jaw3d_onset_ms": [110.0, 200.0, 0.0, 90.0],
        "jaw3d_at_cue": [False, False, True, False], "jaw3d_source": ["4cam", "4cam", "4cam", "2cam"]})


POS = np.array(["far_L", "close_center", "far_R", "far_center"])


def test_classes_mode():
    c = TC.jaw_call(_j(), POS, "classes")
    assert np.allclose(c["jaw_onset_ms"], [100.0, 200.0, 50.0, 70.0], equal_nan=True)
    assert c["judged"].tolist() == [True, True, True, True]
    assert c["use3"].tolist() == [False, True, False, False]


def test_3d_mode():
    c = TC.jaw_call(_j(), POS, "3d")
    assert np.allclose(c["jaw_onset_ms"], [110.0, 200.0, np.nan, np.nan], equal_nan=True)
    assert c["judged"].tolist() == [True, True, False, False]   # at-cue 3-D -> unknown; 2-cam 3-D never used


def test_2d_mode():
    c = TC.jaw_call(_j(), POS, "2d")
    assert np.allclose(c["jaw_onset_ms"], [100.0, np.nan, 50.0, 70.0], equal_nan=True)
    assert c["judged"].tolist() == [True, False, True, True]


def test_bad_mode():
    with pytest.raises(ValueError):
        TC.jaw_call(_j(), POS, "x")


def test_reonset3d_option():
    """11-23 (Priya "let's try"): an at-cue 3-D movement takes its 3-D re-onset when reonset3d is on; default off."""
    j = _j().assign(jaw3d_reonset_ms=[np.nan, np.nan, 340.0, np.nan])
    off = TC.jaw_call(j, POS, "3d")
    on = TC.jaw_call(j, POS, "3d", reonset3d=True)
    assert np.isnan(off["jaw_onset_ms"][2]) and not off["judged"][2]
    assert on["jaw_onset_ms"][2] == 340.0 and on["judged"][2]
    assert np.allclose(on["jaw_onset_ms"][[0, 1]], [110.0, 200.0])
