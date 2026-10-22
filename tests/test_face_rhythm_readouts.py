"""wfield_local.face_rhythm_readouts: window amplitudes, the contra / ipsi convention (cam4: ipsilesional = IMAGE
RIGHT), and joining a two-video session. Synthetic proj files only."""
import numpy as np

from wfield_local import face_rhythm_readouts as frr

P, F, R = 4, 3, 2
HOME = np.array([[10.0, 5], [20, 5], [80, 5], [90, 5]])        # x < 50: image left = CONTRALESIONAL


def _proj(path, date, left_power=1.0, right_power=1.0, resp_amp=2.0):
    t_ms = np.arange(-2000.0, 4001, 40.0)
    tr = np.ones((2, len(t_ms), R), np.float32)
    tr[:, t_ms >= 0] = resp_amp
    pw = np.zeros((2, P, F), np.float32)
    pw[:, :2], pw[:, 2:] = left_power, right_power
    sp = np.array([4.0, 4.0, 1.0, 1.0], np.float32) if left_power > right_power else np.ones(P, np.float32)
    n5 = 200
    np.savez(path, trials=tr, trials_err=np.full((2, len(t_ms)), 0.1, np.float32), session=np.full((n5, R), 0.5),
             session_err=np.full(n5, 0.2), session_t_daq_s=np.arange(n5) / 5.0, session_rate_hz=5.0, spec_hash="x",
             t_ms=t_ms, cue_daq_s=np.array([10.0, 20.0]), trial_id=np.arange(2), position=np.array(["a", "b"]),
             animal=path.stem[:4], epoch="pre", date=date, power_between=pw, power_precue=np.stack([pw, pw]),
             power_response=np.stack([pw, pw]), power_positions=np.array(["a", "b"]),
             power_n_bins=np.array([50, 10, 20, 10, 20]),
             speed_between=sp[None][0], speed_precue=np.stack([sp, sp]), speed_response=np.stack([sp, sp]),
             speed_n_bins=np.array([50, 10, 20, 10, 20]))


def test_amplitude_windows_and_split_session_join(tmp_path):
    (tmp_path / "proj").mkdir()
    _proj(tmp_path / "proj" / "PS92_20260812_cam4_v1.npz", "20260812")
    _proj(tmp_path / "proj" / "PS92_20260812_cam4_v2.npz", "20260812")
    S, Pos = frr.amplitudes(tmp_path, ["fr_0", "fr_1"])
    assert len(S) == 1 and S.label.iloc[0] == "PS92_0812"
    assert np.isclose(S["amp_fr_0_precue"].iloc[0], 1.0) and np.isclose(S["amp_fr_0_response"].iloc[0], 2.0)
    assert np.isclose(S["amp_fr_0_between"].iloc[0], 0.5)
    assert set(Pos.position) == {"a", "b"}


def test_asymmetry_sign_contra_is_image_left(tmp_path):
    (tmp_path / "proj").mkdir()
    _proj(tmp_path / "proj" / "PS93_20260810_cam4.npz", "20260810", left_power=3.0, right_power=1.0)
    S, maps = frr.movement(tmp_path, HOME, np.array([1.0, 5.0, 20.0]), midline_x=50.0, measure="share")
    assert maps["ipsi_mask"].tolist() == [False, False, True, True]
    assert np.isclose(S["asym_share_response"].iloc[0], (3 - 1) / (3 + 1))    # more CONTRAlesional -> positive
    S2, _ = frr.movement(tmp_path, HOME, np.array([1.0, 5.0, 20.0]), midline_x=50.0, measure="speed")
    assert np.isclose(S2["asym_speed_response"].iloc[0], (4 - 1) / (4 + 1))      # RMS: sqrt(16) vs sqrt(1)


def test_regions_around_the_master_nose():
    x0, y0 = frr.MASTER_MIDLINE_X, frr.NOSE_Y
    pts = np.array([[x0, y0], [x0, y0 + 100], [x0 + 100, y0 + 40], [x0, y0 - 150], [x0 + 250, y0 + 250]])
    assert list(frr.region_of(pts)) == ["nose", "mouth_jaw", "whisker_pad", "upper_face", "other"]
