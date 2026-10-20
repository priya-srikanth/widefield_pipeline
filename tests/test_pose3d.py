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


def test_disparity_shift_recovers_an_opposite_horizontal_offset():
    m1 = {p: (10.0 * k, 5.0 * k) for k, p in enumerate(SW.POSITIONS)}
    m4 = {p: (10.0 * k + 6.0, 2.0 * k) for k, p in enumerate(SW.POSITIONS)}   # cam4 sees x 6 px to the right
    d1, d4 = P3.disparity_shift(_FakeGroup(), m1, m4)
    assert np.isclose(d1, 0.0) and np.isclose(d4, -6.0)         # the fake triangulation takes x from cam1
    s = P3.shift_x(pd.DataFrame({"tongue_x": [1.0], "tongue_y": [2.0]}), 3.0)
    assert s.tongue_x.iloc[0] == 4.0 and s.tongue_y.iloc[0] == 2.0


class _Cam:
    """Just enough of aniposelib.Camera for `pnp_group`: pinhole K, no distortion, pose via Rodrigues."""

    def __init__(self, name, rvec, tvec):
        self.name, self.r, self.t = name, np.asarray(rvec, float), np.asarray(tvec, float)
        self.K = np.array([[5500.0, 0, 300], [0, 5500.0, 300], [0, 0, 1]])

    def get_name(self):
        return self.name

    def get_camera_matrix(self):
        return self.K

    def get_distortions(self):
        return np.zeros(5)

    def set_rotation(self, r):
        self.r = np.asarray(r, float)

    def set_translation(self, t):
        self.t = np.asarray(t, float)

    def project(self, X):
        import cv2
        return cv2.projectPoints(np.asarray(X, float), self.r, self.t, self.K, np.zeros(5))[0][:, 0]

    def get_extrinsics_mat(self):
        import cv2
        M = np.eye(4)
        M[:3, :3], M[:3, 3] = cv2.Rodrigues(self.r)[0], self.t
        return M


class _Group:
    def __init__(self, cams):
        self.cameras = cams

    def project(self, X):
        return np.stack([c.project(X) for c in self.cameras])

    def triangulate(self, pts, undistort=True):
        import cv2
        P = [c.K @ np.c_[cv2.Rodrigues(c.r)[0], c.t] for c in self.cameras]
        H = cv2.triangulatePoints(P[0], P[1], pts[0].T.astype(float), pts[1].T.astype(float))
        return (H[:3] / H[3]).T


def test_pnp_poses_each_camera_from_the_commanded_spouts_and_triangulates_in_mm():
    true = _Group([_Cam("cam1", [-0.8, 0.02, 0.0], [0.3, 1.0, 150.0]), _Cam("cam4", [1.5, 0.0, 0.0], [-0.2, 2.0, 160.0])])
    obj = np.array([SW.commanded_world(GEOM, p) for p in SW.POSITIONS])
    med = {c.get_name(): {p: tuple(c.project(obj)[k]) for k, p in enumerate(SW.POSITIONS)} for c in true.cameras}
    start = _Group([_Cam("cam1", [0, 0, 0], [0, 0, 100.0]), _Cam("cam4", [0, 0, 0], [0, 0, 100.0])])
    g, err = P3.pnp_group(start, med, GEOM, edge_offset=None)
    assert max(max(e.values()) for e in err.values()) < 1e-3
    body = np.array([[0.4, 1.9, -2.7]])                     # a tongue tip, mm
    X = g.triangulate(true.project(body), undistort=True)
    assert np.allclose(X, body, atol=1e-3)
    assert start.cameras[0].t[2] == 100.0                     # the input group is not modified


def test_single_view_frames_are_filled_on_the_seen_ray_with_depth_from_neighbours():
    g = _Group([_Cam("cam1", [-0.8, 0.02, 0.0], [0.3, 1.0, 150.0]), _Cam("cam4", [1.5, 0.0, 0.0], [-0.2, 2.0, 160.0])])
    t = np.arange(7) * 4.0
    truth = np.c_[0.1 * t, 1.0 + 0.05 * t, -2.0 - 0.02 * t]          # straight-line motion: interpolation is exact
    px = g.project(truth)
    a, b = px[0].copy(), px[1].copy()
    a[2:4] = np.nan                                                  # frames 2, 3: cam4 only
    b[4] = np.nan                                                    # frame 4: cam1 only
    X = g.triangulate(np.stack([a, b]), undistort=True)
    X[~(np.isfinite(a).all(1) & np.isfinite(b).all(1))] = np.nan
    F, src = P3.fill_single_view(X, a, b, g, np.ones(7, int), t, max_gap_ms=24.0)
    assert list(src) == [1, 1, 3, 3, 2, 1, 1]
    assert np.allclose(F, truth, atol=1e-3)
    for interp in ("pchip", "akima"):                                 # splines are exact on a straight line too
        Fs, _ = P3.fill_single_view(X, a, b, g, np.ones(7, int), t, max_gap_ms=24.0, interp=interp)
        assert np.allclose(Fs, truth, atol=1e-3), interp
    _, src2 = P3.fill_single_view(X, a, b, g, np.ones(7, int), t, max_gap_ms=2.0)   # no neighbour close enough
    assert list(src2) == [1, 1, 0, 0, 0, 1, 1]


def test_pnp_edge_offset_puts_each_camera_on_its_own_labelled_edge():
    true = _Group([_Cam("cam1", [-0.8, 0.02, 0.0], [0.3, 1.0, 150.0]), _Cam("cam4", [1.5, 0.0, 0.0], [-0.2, 2.0, 160.0])])
    off = {"cam4": (0.0, 0.0, 0.0), "cam1": (0.0, 0.0, -1.24)}
    obj = np.array([SW.commanded_world(GEOM, p) for p in SW.POSITIONS])
    med = {c.get_name(): {p: tuple(c.project(obj + off[c.get_name()])[k]) for k, p in enumerate(SW.POSITIONS)}
           for c in true.cameras}                                  # each camera sees ITS edge of the rod
    start = _Group([_Cam("cam1", [0, 0, 0], [0, 0, 100.0]), _Cam("cam4", [0, 0, 0], [0, 0, 100.0])])
    g, _ = P3.pnp_group(start, med, GEOM, edge_offset=off)
    body = np.array([[0.4, 1.9, -2.7]])
    assert np.allclose(g.triangulate(true.project(body), undistort=True), body, atol=1e-3)
    g0, _ = P3.pnp_group(start, med, GEOM, edge_offset=None)      # ignoring the edges: a constant shift appears
    assert np.linalg.norm(g0.triangulate(true.project(body), undistort=True) - body) > 0.05


def test_qc_drops_far_points_and_single_frame_spikes():
    t = np.arange(8) * 4.0
    X = np.c_[np.linspace(0.5, 3.0, 8), np.full(8, 1.0), np.full(8, -2.0)]
    X[3] = [6.0, 1.0, -2.0]                                         # one frame out and back by ~4 mm
    X[6] = [20.0, 0.0, 0.0]                                         # impossible reach
    Q, n = P3.qc_3d(X, "tongue", np.ones(8, int), t)
    assert np.isnan(Q[3]).all() and np.isnan(Q[6]).all() and np.isfinite(Q[[0, 1, 2, 4, 5, 7]]).all()
    assert n == {"tongue_qc_far": 1, "tongue_qc_behind": 0, "tongue_qc_spike": 1}
    Y = np.c_[np.full(4, 0.5), np.array([0.5, -0.8, -2.5, 0.4]), np.full(4, -2.0)]   # 2.5 mm behind the mouth: dropped
    Q2, n2 = P3.qc_3d(Y, "tongue", np.ones(4, int), np.arange(4) * 4.0)
    assert np.isnan(Q2[2]).all() and np.isfinite(Q2[[0, 1, 3]]).all() and n2["tongue_qc_behind"] == 1


def test_lick_events_bridge_short_gaps_and_split_long_ones():
    t = np.arange(20) * 4.0
    P = np.full((20, 3), np.nan)
    P[1:5] = [[0.1 * k, 1.0, -1.0 - 0.2 * k] for k in range(4)]   # lick 1, frames 1-4
    P[7:9] = [[0.0, 1.2, -2.5], [0.0, 1.1, -2.0]]                   # 8 ms gap (frames 5-6) -> same lick
    P[15:18] = [[0.0, 0.5, -0.8]] * 3                               # 24 ms later -> a second lick
    S = pd.DataFrame({"trial_id": 1, "position": "far_L", "t_ms": t, "stop_ms": 1e4,
                      "tongue_ml": P[:, 0], "tongue_ap": P[:, 1], "tongue_dv": P[:, 2]})
    E = P3.lick_events(S, max_gap_ms=12.0)
    assert len(E) == 2 and list(E.lick_idx) == [0, 1]
    assert E.on_ms.iloc[0] == 4.0 and E.off_ms.iloc[0] == 32.0 and E.t_ms.iloc[0] == 28.0   # peak = farthest frame


def test_pair_on_daq_time_survives_dropped_frames():
    """PS92 0908 (10-20): cam4 dropped frames, so its frame-index t_ms slides; pairing on the camera-timestamp time
    (t_daq_ms) keeps every surviving frame on its true partner and leaves the dropped ones unpaired."""
    t = 4.0 * np.arange(10)
    ix1 = pd.DataFrame({"trial_id": 1, "t_ms": t, "t_daq_ms": t + 0.3})
    keep = np.r_[0:3, 5:10]                                     # cam4 lost frames 3 and 4
    ix4 = pd.DataFrame({"trial_id": 1, "t_ms": 4.0 * np.arange(len(keep)), "t_daq_ms": t[keep] - 0.2})
    i1, i4 = P3.pair(ix1, ix4)
    assert i1.tolist() == keep.tolist() and i4.tolist() == list(range(len(keep)))
    i1, i4 = P3.pair(ix1.drop(columns="t_daq_ms"), ix4.drop(columns="t_daq_ms"))     # old exact join: slides
    assert i1.tolist() == list(range(len(keep)))
