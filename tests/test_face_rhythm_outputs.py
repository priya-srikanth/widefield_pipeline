"""wfield_local.face_rhythm_outputs: joining trial (25 Hz) and session (5 Hz) projections, frame interpolation, the
frozen-set guard. Synthetic files only (no share access)."""
import numpy as np
import pytest

from wfield_local import face_rhythm_outputs as fro


def _proj(path, cues, R=2, spec="abc", n5=200, f_tr=None, f_se=None):
    t_ms = np.arange(-2000.0, 4000.0 + 1, 40.0)
    t_tr = np.asarray(cues)[:, None] + t_ms[None] / 1000.0
    ts = np.arange(n5) / 5.0 + 0.1
    f_tr = f_tr or (lambda t: np.stack([t, 2 * t], -1))
    f_se = f_se or (lambda t: np.stack([t, 2 * t], -1))
    np.savez(path, trials=f_tr(t_tr).astype(np.float32), trials_err=np.zeros(t_tr.shape, np.float32),
             session=f_se(ts).astype(np.float32), session_err=np.zeros(n5, np.float32), session_t_daq_s=ts,
             session_rate_hz=np.float32(5), spec_hash=spec, t_ms=t_ms.astype(np.float32), cue_daq_s=np.asarray(cues),
             trial_id=np.arange(len(cues)), position=np.array(["far_L"] * len(cues)))


def test_trials_replace_session_bins_inside_windows(tmp_path):
    f = tmp_path / "a.npz"
    _proj(f, [10.0], f_tr=lambda t: np.stack([np.full_like(t, 7.0), t], -1),
          f_se=lambda t: np.stack([np.zeros_like(t), t], -1))
    t, W = fro.combined_series([f])
    inside = (t >= 8.0) & (t <= 14.0)
    assert np.all(W[inside, 0] == 7.0)                      # only trial samples inside the window
    assert np.all(W[~inside, 0] == 0.0)                     # only session samples outside
    assert np.all(np.diff(t) > 0)
    assert np.isclose(np.diff(t[inside]).max(), 0.04)       # 25 Hz inside


def test_to_frames_interpolates_and_does_not_bridge_gaps():
    t = np.array([0.0, 0.2, 0.4, 2.0, 2.2])
    W = np.stack([t, -t], 1)
    ft = np.array([0.1, 0.3, 1.0, 2.1, 3.0])
    V = fro.to_frames(t, W, ft, max_gap_s=0.5)
    assert np.allclose(V[[0, 1, 3], 0], [0.1, 0.3, 2.1])
    assert np.isnan(V[2]).all() and np.isnan(V[4]).all()    # inside a 1.6 s gap / after the last sample


def test_split_session_videos_join_and_mixed_sets_refused(tmp_path):
    a, b = tmp_path / "v1.npz", tmp_path / "v2.npz"
    _proj(a, [10.0], n5=100)
    _proj(b, [60.0], n5=100, f_se=lambda t: np.stack([t + 50, 2 * (t + 50)], -1))
    t, W = fro.combined_series([a, b])
    assert len(t) == len(W) and np.all(np.diff(t) >= 0)
    _proj(b, [60.0], spec="other")
    with pytest.raises(ValueError, match="different frozen factor sets"):
        fro.combined_series([a, b])


def test_points_kind_reads_the_displacement_pcs(tmp_path):
    f = tmp_path / "a.npz"
    _proj(f, [10.0])
    z = dict(np.load(f))
    z["points_pc"] = np.arange(30, dtype=np.float32).reshape(10, 3)
    z["points_t_daq_s"] = np.arange(10) / 25.0
    np.savez(f, **z)
    t, W = fro.combined_series([f], kind="points", k=2)
    assert W.shape == (10, 2) and np.allclose(t, np.arange(10) / 25.0)
