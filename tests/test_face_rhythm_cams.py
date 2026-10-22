"""FaceRhythm on cameras other than cam4 (wfield_local.face_rhythm_cams, 10-22): per-camera settings, frame lookup on a
camera's own clock, cam4's trial grid re-expressed in another camera's frames, set references across bundles, and
camera-agnostic session keys. Synthetic arrays only (no share access)."""
import numpy as np
import pytest

from wfield_local import face_rhythm_cams as frc
from wfield_local import face_rhythm_outputs as fro


def test_every_camera_has_the_settings_the_chain_reads():
    keys = {"view", "shape", "lp_landmarks", "click_landmarks", "rest_exclude", "face_side", "ipsi_image_side"}
    assert set(frc.CAMS) == {"cam1", "cam2", "cam3", "cam4"}
    for c in frc.CAMS.values():
        assert keys <= set(c)
        assert c["lp_landmarks"] or len(c["click_landmarks"]) >= 3       # >= 3 points for a similarity start
    assert [n for n, c in frc.CAMS.items() if c["lp_landmarks"]] == ["cam4"]
    with pytest.raises(ValueError):
        frc.cam("cam5")


def test_video_for_keeps_the_recording():
    assert frc.video_for("cam4_2026-08-18T17_23_46.avi", "cam2") == "cam2_2026-08-18T17_23_46.avi"


def test_frames_at_follows_the_camera_clock_and_skips_gaps():
    fd = np.arange(100) / 250.0 + 3.0                    # 250 fps, first frame at DAQ 3 s
    fd[40:50] = np.nan                                   # a segment gap (no template covers it)
    x = frc.frames_at(fd, np.array([3.0, 3.1, 2.0, 3.0 + 99 / 250 + 1]))
    assert np.allclose(x[:2], [0, 25]) and np.isnan(x[2:]).all()
    n = frc.frames_at(fd, np.array([3.0 + 10.4 / 250, 9.0]), nearest=True)
    assert n[0] == 10 and n[1] == -1


def test_other_cam_grid_same_trials_shifted_frames():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import o2_face_rhythm as o
    t_ms = o.T_MS
    cues = np.array([20.0, 40.0])
    src4 = np.stack([(c + t_ms / 1000.0) * 250.0 for c in cues])
    src4[1, -5:] = np.nan                                 # cam4's window stops short on trial 2
    tr = {"src_frame": src4, "trial_id": np.array([3, 4]), "cue_daq_s": cues, "position": np.array(["far_L"] * 2),
          "t_ms": t_ms, "daq_fit": np.array([1 / 250.0, 0.0]), "daq_fit_max_resid_ms": 0.0}
    fd = np.arange(20000) / 250.0 + 0.056                 # this camera starts 14 frames later
    g = o.other_cam_grid(tr, fd)
    assert np.array_equal(g["trial_id"], tr["trial_id"])
    assert np.array_equal(np.isnan(g["src_frame"]), np.isnan(src4))  # same coverage as cam4
    assert np.allclose(np.nanmedian(g["src_frame"] - src4), -14.0)
    assert np.allclose(g["daq_fit"], [1 / 250.0, 0.056]) and g["daq_fit_max_resid_ms"] < 1e-6


def test_set_references_name_their_bundle(tmp_path):
    assert fro.set_tag("master_pre_v1") == "master_pre_v1"
    assert fro.set_tag("fr_cam2_20261101/master_pre_v1") == "cam2-master_pre_v1"
    d = fro.factor_dir(None, set_name="fr_cam2_20261101/master_pre_v1")
    assert d.parts[-3:] == ("fr_cam2_20261101", "factors", "master_pre_v1")
    assert fro.factor_dir(None).parts[-3] == fro.BUNDLE


def test_session_keys_any_camera_in_video_order(tmp_path):
    (tmp_path / "proj").mkdir()
    for k in ("PS92_20260812_cam2_v2", "PS92_20260812_cam2_v1", "PS92_20260818_cam2", "PS93_20260818_cam2"):
        (tmp_path / "proj" / f"{k}.npz").write_bytes(b"")
    assert [f.stem for f in fro.session_keys(tmp_path, "PS92", "20260812")] == \
        ["PS92_20260812_cam2_v1", "PS92_20260812_cam2_v2"]
    assert [f.stem for f in fro.session_keys(tmp_path, "PS92", "20260818")] == ["PS92_20260818_cam2"]
    (tmp_path / "proj" / "PS92_20260818_cam4.npz").write_bytes(b"")
    with pytest.raises(ValueError):
        fro.session_keys(tmp_path, "PS92", "20260818")
