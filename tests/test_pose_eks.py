"""Pure tests for `wfield_local.pose_eks`: frame matching on the DAQ clock and the EKS input/output
conversions. None of them imports ``eks`` (it lives in its own env), so they run in ``locanmf``."""
import numpy as np
import pandas as pd
import pytest

from wfield_local import pose_eks as PE


def _tpl(slope=20.0153, icept=-13000.0, fs=5000.0, n=2_000_000):
    return {"fs_daq": fs, "slope_daqSample_per_camFrame": slope, "intercept_daqSample": icept, "n_cam_frames": n}


def test_daq_round_trip_is_identity():
    t = _tpl()
    f = np.array([0, 1, 12345, 999_999], float)
    assert np.allclose(PE.daq_s_to_cam_frame(t, PE.cam_frame_to_daq_s(t, f)), f)


def test_match_frames_constant_offset_and_residual():
    """Same clock, intercepts 7.4 frames apart: every frame maps 7 frames on (nearest), residual -0.4 frame."""
    slope = 20.0
    a, b = _tpl(slope, 0.0), _tpl(slope, -7.4 * slope)
    g, r = PE.match_frames(a, b, np.arange(100, 110))
    assert np.array_equal(g, np.arange(107, 117))
    assert np.allclose(r, -0.4 * slope / 5000.0)
    assert np.all(np.abs(r) <= 0.5 * slope / 5000.0 + 1e-12)


def test_match_frames_drift_gives_a_skip_not_a_shift():
    """dst runs 1 % faster: over 300 frames the map must skip ~3 dst frames, never drift by more than half a frame."""
    a, b = _tpl(20.0, 0.0), _tpl(20.0 / 1.01, 0.0)
    g, r = PE.match_frames(a, b, np.arange(300))
    assert (np.diff(g) == 2).sum() in (2, 3, 4)
    assert np.abs(r).max() <= 0.5 * (20.0 / 1.01) / 5000.0 + 1e-12


def test_match_frames_clipped_to_recording():
    g, _ = PE.match_frames(_tpl(n=10), _tpl(n=10), np.array([5, 9, 50]))
    assert g.max() == 9


def _pose(T=6, K=4, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0, 600, (T, K, 3))
    p[..., 2] = rng.uniform(0, 1, (T, K))
    return p


def test_pose_frame_round_trip_through_lp_csv(tmp_path):
    p = _pose()
    df = PE.pose_to_frame(p, scorer="lp")
    f = tmp_path / "x.csv"
    df.to_csv(f)
    assert np.allclose(PE.pose_from_lp_csv(f), p)


def test_flat_frame_names_match_eks_loader():
    df = PE.flat_frame(_pose(T=3))
    assert list(df.columns[:3]) == ["nose_x", "nose_y", "nose_likelihood"]
    assert df.shape == (3, 12)


def test_mask_low_per_part():
    p = _pose()
    p[:, 2, 2] = 0.3
    p[:, 1, 2] = 0.9
    m = PE.mask_low(p, {"nose": 0.6, "jaw": 0.6, "tongue": 0.4, "spout": 0.6})
    assert np.isnan(m[:, 2, :2]).all() and np.isfinite(m[:, 1, :2]).all()
    assert np.array_equal(m[..., 2], p[..., 2])          # likelihood kept


def test_marker_stack_shape_floor_and_nan_refusal():
    a, b = _pose(seed=1), _pose(seed=2)
    a[0, 0, 2] = 0.0
    arr, cams = PE.marker_stack({"cam1": [a, b], "cam4": [b, a]})
    assert arr.shape == (2, 2, 6, 4, 3) and cams == ["cam1", "cam4"]
    assert arr[0, 0, 0, 0, 2] == pytest.approx(1e-3)
    a[1, 2, 0] = np.nan
    with pytest.raises(ValueError):
        PE.marker_stack({"cam1": [a, b]})
    with pytest.raises(ValueError):                      # unequal members across cameras
        PE.marker_stack({"cam1": [b, b], "cam4": [b]})


def test_floor_occluded_var_only_where_low():
    ens = np.zeros((1, 1, 3, 1, 5))
    ens[..., 2:4] = 4.0
    ens[0, 0, :, 0, 4] = [0.9, 0.2, 0.05]
    out = PE.floor_occluded_var(ens, cut=0.4, var=1000.0)
    assert np.allclose(out[0, 0, :, 0, 2], [4.0, 1000.0, 1000.0])
    assert np.allclose(ens[0, 0, :, 0, 2], 4.0)          # input untouched


def test_unpack_eks_frame_subtracts_obs_var():
    labels = ["x", "y", "likelihood", "x_ens_median", "y_ens_median", "x_ens_var", "y_ens_var",
              "x_posterior_var", "y_posterior_var"]
    cols = pd.MultiIndex.from_tuples([("ensemble-kalman_tracker", p, c) for p in PE.PARTS for c in labels])
    vals = np.tile(np.array([1, 2, 0.5, 1, 2, 10, 20, 15, 26], float), (5, 4))
    r = PE.unpack_eks_frame(pd.DataFrame(vals, columns=cols))
    assert r.pose.shape == (5, 4, 3) and np.allclose(r.var[0, 0], [15, 26])
    r2 = PE.unpack_eks_frame(pd.DataFrame(vals, columns=cols), includes_obs_var=True)
    assert np.allclose(r2.var[0, 0], [5, 6])


def test_pca_keep_quantile_ignores_unseen_part_and_clips():
    lik = np.ones((2, 100, 4))
    lik[:, 80:, 2] = 0.1                                # tongue seen in both views on 80 % of frames
    lik[0, :, 0] = 0.0                                  # cam1 nose never seen
    assert PE.pca_keep_quantile(lik, 0.4) == pytest.approx(50.0)     # 72 -> capped at 50
    lik[:, 20:, 2] = 0.1                                # tongue now 20 %
    assert PE.pca_keep_quantile(lik, 0.4) == pytest.approx(18.0)
