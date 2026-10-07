"""null_potent / motion_state (Hasnain et al. 2025 method): planted stationary / moving subspaces are recovered,
normVE is in [0, 1] and high on the matching condition, the two-stage PCA control is orthogonal, the bimodal
threshold sits between the modes, frame labels carry the post-movement buffer, and per-frame motion energy rises
only when the picture changes."""
from __future__ import annotations

import numpy as np

from wfield_local import motion_state as MS
from wfield_local import null_potent as NP


def _planted(seed=0, N=20, T=4000):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((N, N)))
    A, B = Q[:, :3], Q[:, 3:6]                          # stationary dims, moving dims (orthogonal)
    Xs = rng.standard_normal((T, 3)) * 3 @ A.T + 0.2 * rng.standard_normal((T, N))
    Xm = rng.standard_normal((T, 3)) * 3 @ B.T + 0.2 * rng.standard_normal((T, N))
    return Xs, Xm, A, B


def _angle_cos(U, V):
    return np.linalg.svd(U.T @ V, compute_uv=False)     # cosines of principal angles


def test_joint_optimisation_recovers_planted_null_and_potent_subspaces():
    Xs, Xm, A, B = _planted()
    r = NP.fit_subspaces(Xs, Xm, d_null=3, d_pot=3)
    assert np.allclose(r["Q_null"].T @ r["Q_pot"], 0, atol=1e-8)
    assert _angle_cos(r["Q_null"], A).min() > 0.98 and _angle_cos(r["Q_pot"], B).min() > 0.98
    ve = r["normVE"]
    assert ve["null_stat"] > 0.95 and ve["pot_mov"] > 0.95 and ve["null_mov"] < 0.1 and ve["pot_stat"] < 0.1
    assert np.all(np.diff(r["objective"]) >= -1e-12)


def test_two_stage_pca_control_is_orthogonal_and_finds_the_planted_dims():
    Xs, Xm, A, B = _planted(1)
    r = NP.two_stage_pca(Xs, np.vstack([Xs, Xm]), k_null=3, k_pot=3)
    assert np.allclose(r["Q_null"].T @ r["Q_pot"], 0, atol=1e-8)
    assert _angle_cos(r["Q_null"], A).min() > 0.98 and _angle_cos(r["Q_pot"], B).min() > 0.95


def test_parallel_analysis_counts_the_planted_dimensions():
    Xs, Xm, *_ = _planted(2)
    assert NP.parallel_analysis(np.vstack([Xs, Xm]), n_shuffle=30) == 6


def test_bimodal_threshold_sits_between_the_modes():
    rng = np.random.default_rng(0)
    v = np.concatenate([np.exp(rng.normal(0, 0.2, 8000)), np.exp(rng.normal(2.5, 0.5, 2000))])
    t = MS.bimodal_threshold(v)
    assert np.exp(0.4) < t < np.exp(2.0)


def test_labels_flag_movement_and_buffer_the_frames_after_it():
    ft = np.arange(0, 3, 0.032)
    me = np.zeros(len(ft))
    me[20:25] = 10.0
    lab = MS.label_frames({"cam": me}, {"cam": 1.0}, ft, buffer_s=0.1)
    assert (lab[20:25] == 1).all() and (lab[25:28] == -1).all() and lab[28] == 0 and lab[10] == 0


def test_frame_scalar_rises_only_when_the_picture_changes(tmp_path):
    import cv2
    vid = tmp_path / "v.avi"
    wr = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"FFV1"), 250.0, (32, 32))
    for i in range(60):
        im = np.full((32, 32, 3), 50, np.uint8)
        if 30 <= i < 36:
            im[8:16, 8:16] = 200                        # a brief change
        wr.write(im)
    wr.release()
    v = MS.frame_scalar_worker(str(vid), 0, 60, ds=4, win=3, pct=99.0)
    assert np.nanmax(v[:24]) < 1 and np.nanmax(v[27:38]) > 50 and np.nanmax(v[42:57]) < 1
