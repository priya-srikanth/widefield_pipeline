"""The context-frame rules: bursts, contact pairing, and that existing targets are never re-labelled context."""
import numpy as np
import pandas as pd

from wfield_local.dlc_context_frames import CONTEXT, burst, contact_context, pair_contacts

# identity-ish template: DAQ sample = camera frame * 20 (5000 Hz DAQ, 250 fps camera), no intercept
TPL = {"fs_daq": 5000.0, "fps_cam": 250.0, "slope_daqSample_per_camFrame": 20.0,
       "intercept_daqSample": 0.0, "n_cam_frames": 10_000}


def test_burst_clips_at_edges():
    assert burst(10, 2) == [8, 9, 10, 11, 12]
    assert burst(1, 3) == [0, 1, 2, 3, 4]
    assert burst(9998, 3, n_frames=10_000) == [9995, 9996, 9997, 9998, 9999]


def test_pair_contacts_takes_first_offset_before_next_onset():
    got = pair_contacts([1.0, 2.0, 3.0], [1.08, 1.09, 2.5, 3.5])
    assert np.allclose(got[:, 1], [1.08, 2.5, 3.5])


def test_pair_contacts_missing_end_is_nan_not_the_next_licks():
    got = pair_contacts([1.0, 2.0], [2.1])      # onset 1.0 has no end before 2.0
    assert np.isnan(got[0, 1]) and got[1, 1] == 2.1


def _targets():
    # a lick at frame 1000 (t = 4.0 s), cue 1 s earlier; targets at -4, 0, +8, +16 frames
    base = {"animal": "PS92", "date": "20260820", "cam": "cam1", "epoch": "acute",
            "video_stem": "cam1_x", "trial_id": 7, "position": "far_L", "category": "success"}
    rows = [{**base, "frame": 1000 + d, "phase": ph, "t_from_cue_s": 1.0 + d / 250}
            for d, ph in ((-4, "lick-16"), (0, "lick+0"), (8, "lick+32"), (16, "lick+64"))]
    return pd.DataFrame(rows)


def test_context_around_onset_and_end_excludes_targets():
    contacts = np.array([[4.0, 4.0 + 20 / 250]])          # contact ends 20 frames after onset
    rows = contact_context(_targets(), TPL, contacts, half=4)
    frames = sorted(r["frame"] for r in rows)
    assert all(r["category"] == CONTEXT for r in rows)
    assert 1000 not in frames and 996 not in frames and 1016 not in frames      # targets stay targets
    assert set(range(997, 1005)) - {1000} <= set(frames)                          # onset burst
    assert set(range(1016 + 1, 1025)) <= set(frames)                               # end burst (1020 +-4)
    ph = {r["frame"]: r["phase"] for r in rows}
    assert ph[1020] == "ctx_off+0" and ph[997] == "ctx_on-3"
    assert abs(next(r for r in rows if r["frame"] == 1020)["t_from_cue_s"] - (1.0 + 20 / 250)) < 1e-6


def test_lick_without_matching_daq_onset_is_skipped():
    rows = contact_context(_targets(), TPL, np.array([[9.0, 9.08]]), half=4)
    assert rows == []
