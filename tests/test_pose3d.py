"""`wfield_local.pose3d`: exact frame pairing, triangulation plumbing, lick runs and the aim-sign convention."""
import numpy as np
import pandas as pd

from wfield_local import pose3d as P3
from wfield_local import spout_world as SW

GEOM = {"dist_close": "2", "dist_far": "4", "az_center": "0.0", "az_left": "-60", "az_right": "60", "down_angle": "45",
        "head_roll": "0"}


def test_pair_is_an_exact_join_on_trial_and_time():
    ix1 = pd.DataFrame({"trial_id": [1, 1, 1, 2], "t_ms": [0.0, 4.0, 8.0, 0.0]})
    ix4 = pd.DataFrame({"trial_id": [1, 1, 2, 2], "t_ms": [4.0, 8.0, 0.0, 4.0]})
    i1, i4 = P3.pair(ix1, ix4)
    assert list(i1) == [1, 2, 3] and list(i4) == [0, 1, 2]


class _FakeGroup:
    """Two orthographic 'cameras': cam1 sees (x, y), cam4 sees (x, z)."""

    def triangulate(self, pts, undistort=True):
        return np.c_[pts[0][:, 0], pts[0][:, 1], pts[1][:, 1]]

    def project(self, X):
        return np.stack([X[:, [0, 1]], X[:, [0, 2]]])


def test_triangulate_leaves_nan_where_either_view_is_missing():
    a = np.array([[1.0, 2.0], [np.nan, np.nan], [3.0, 4.0]])
    b = np.array([[1.0, 5.0], [1.0, 1.0], [np.nan, np.nan]])
    X, rep = P3.triangulate(_FakeGroup(), a, b)
    assert np.allclose(X[0], [1, 2, 5]) and np.isnan(X[1:]).all()
    assert np.allclose(rep[0], 0) and np.isnan(rep[1:]).all()


def _frames(pos, path):
    n = len(path)
    return pd.DataFrame({"trial_id": 1, "position": pos, "t_ms": np.arange(n) * 4.0, "stop_ms": 1e4,
                         "tongue_ml": path[:, 0], "tongue_ap": path[:, 1], "tongue_dv": path[:, 2],
                         "tongue_rep1": 1.0, "tongue_rep4": 1.0})


def test_lick_runs_split_on_gaps_and_need_three_frames():
    g = SW.commanded_world(GEOM, "far_center")
    path = np.array([g * 0.5, g * 0.9, g, [np.nan] * 3, g, g * 0.5])
    L = P3.lick_peaks(_frames("far_center", path), {"geometry": GEOM})
    assert len(L) == 1 and L.n_frames.iloc[0] == 3
    assert np.isclose(L.miss_mm.iloc[0], 0.0) and np.isclose(L.reach_mm.iloc[0], np.linalg.norm(g))


def test_lat_err_is_positive_outward_on_both_sides_and_mouse_left_at_centre():
    for pos, shift, want in (("far_L", -0.5, +0.5), ("far_R", +0.5, +0.5), ("far_R", -0.5, -0.5),
                             ("far_center", -0.5, +0.5)):        # ml + = mouse RIGHT; centre: + = mouse left
        g = SW.commanded_world(GEOM, pos) + np.array([shift, 0, 0])
        L = P3.lick_peaks(_frames(pos, np.stack([g * 0.5, g * 0.8, g])), {"geometry": GEOM})
        assert np.isclose(L.lat_err_mm.iloc[0], want), pos


def test_june_scaling_keeps_the_centre_fixed():
    f = pd.DataFrame({"tongue_x": [332.0, 432.0], "tongue_y": [380.0, 380.0], "tongue_likelihood": [1.0, 1.0]})
    s = P3.scale_june_cam4(f, 0.885, (332.0, 380.0))
    assert np.allclose(s.tongue_x, [332.0, 332.0 + 88.5]) and np.allclose(s.tongue_y, 380.0)
    assert (s.tongue_likelihood == 1.0).all()
