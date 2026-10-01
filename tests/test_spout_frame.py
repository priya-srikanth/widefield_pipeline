"""spout_frame: three lines through a known point recover it; frame coordinates and angles have the stated signs."""
import numpy as np
import pandas as pd

from wfield_local import spout_frame as SF

MOUTH = np.array([340.0, 350.0])


def _medians():
    """Spouts on three rays from MOUTH, downward in the image: centre straight down, 'L' to image-right (cam4)."""
    out = {}
    for side, ang in (("center", 0.0), ("L", 40.0), ("R", -40.0)):
        d = np.array([np.sin(np.radians(ang)), np.cos(np.radians(ang))])        # image coords, y down
        for kind, r in (("close", 60.0), ("far", 140.0)):
            x, y = MOUTH + r * d
            out[f"{kind}_{side}"] = (x, y, 100)
    return out


def test_origin_recovered_and_axes():
    f = SF.from_medians(_medians())
    assert np.allclose(f.origin, MOUTH, atol=1e-6) and f.rms_px < 1e-6
    assert np.allclose(f.ap_axis, [0, -1], atol=1e-9)          # far_center -> mouth = up the image
    assert f.lr_axis[0] > 0 and f.mouse_left_sign == 1


def test_to_frame_and_angle_signs():
    f = SF.from_medians(_medians())
    ap, lr = SF.to_frame([340.0, 380.0], [450.0, 450.0], f)
    assert np.isclose(ap[0], 100) and np.isclose(lr[0], 0)     # straight out of the mouth
    assert lr[1] > 0                                              # image-right
    ang = SF.angle_deg([340.0, 440.0, 240.0], [450.0, 450.0, 450.0], f)
    assert np.isclose(ang[0], 0) and np.isclose(ang[1], 45) and np.isclose(ang[2], -45)


def test_trial_spans_strobe_to_next_trial_start():
    tpl = {"fs_daq": 5000.0, "slope_daqSample_per_camFrame": 20.0, "intercept_daqSample": 0.0}
    trials = pd.DataFrame({"cue_s": [5.0, 15.0], "trial_start_s": [1.0, 11.0], "pos_name": ["far_L", "close_R"]})
    spans = SF.trial_spans(trials, [2.0, 12.0], tpl, n_frames=10_000)
    assert spans == [(500, 2750, "far_L"), (3000, 10_000, "close_R")]


def test_position_medians_skip_low_confidence():
    xyp = np.zeros((10, 3))
    xyp[:, 0], xyp[:, 1], xyp[:, 2] = 100, 200, 1.0
    xyp[3, :] = (999, 999, 0.1)
    assert SF.position_medians(xyp, [(0, 10, "far_L")]) == {"far_L": (100.0, 200.0, 9)}
