"""`wfield_local.pose3d_side`: side-frame pairing, the water-drop rule, robust view rejection, the side cleaning params."""
import numpy as np
import pandas as pd

from wfield_local import pose3d_side as PS

ORDER = ["cam1", "cam4", "cam2", "cam3"]


def test_pair_side_exact_join_for_clips_and_true_time_for_sessions():
    S = pd.DataFrame({"trial_id": [1, 1, 1, 2], "t_ms": [0.0, 4.0, 8.0, 0.0]})
    ix = pd.DataFrame({"trial_id": [1, 1, 2], "t_ms": [4.0, 8.0, 0.0]})
    iS, iX = PS.pair_side(S, ix)
    assert list(iS) == [1, 2, 3] and list(iX) == [0, 1, 2]
    # whole sessions: nearest t_daq_ms within half a frame, so a dropped side frame leaves a gap, not a slide
    S2 = S.assign(t_daq_ms=[0.0, 4.0, 8.0, 0.0])
    ix2 = pd.DataFrame({"trial_id": [1, 1, 2], "t_ms": [0.0, 4.0, 0.0], "t_daq_ms": [0.5, 8.4, 0.3]})
    iS2, iX2 = PS.pair_side(S2, ix2)
    assert dict(zip(iS2, iX2)) == {0: 0, 2: 1, 3: 2}


def _tongue(n, side_xy, cam14=False):
    P = np.full((4, n, 2), np.nan)
    P[ORDER.index("cam2")] = side_xy
    if cam14:
        P[ORDER.index("cam1")] = 1.0
    return P


def test_droplet_rule_flags_a_static_side_only_tongue_and_nothing_else():
    n = 8
    static = np.tile([100.0, 200.0], (n, 1)) + np.random.default_rng(0).normal(0, 0.3, (n, 2))
    moving = np.c_[np.linspace(100, 160, n), np.full(n, 200.0)]
    trial = np.ones(n, int)
    scale = {"cam2": 1.0, "cam3": 1.0}
    assert PS.droplet_mask(_tongue(n, static), ORDER, trial, scale)[ORDER.index("cam2")].all()
    assert not PS.droplet_mask(_tongue(n, moving), ORDER, trial, scale).any()
    # cam1 / cam4 also see a tongue there -> a tongue held on the spout, not a drop
    assert not PS.droplet_mask(_tongue(n, static, cam14=True), ORDER, trial, scale).any()
    # too short to call
    assert not PS.droplet_mask(_tongue(3, static[:3]), ORDER, np.ones(3, int), scale).any()


class _Ortho3:
    """Three orthographic cameras: A sees (x, y), B sees (x, z), C sees (y, z)."""
    axes = [(0, 1), (0, 2), (1, 2)]

    def triangulate(self, pts, undistort=True):
        out = np.full((pts.shape[1], 3), np.nan)
        for i in range(pts.shape[1]):
            acc = [[] for _ in range(3)]
            for j, (a, b) in enumerate(self.axes):
                if np.isfinite(pts[j, i]).all():
                    acc[a].append(pts[j, i, 0])
                    acc[b].append(pts[j, i, 1])
            out[i] = [np.mean(v) if v else np.nan for v in acc]
        return out

    def project(self, X):
        return np.stack([X[:, list(ax)] for ax in self.axes])


def test_fuse_robust_drops_the_one_view_that_disagrees(monkeypatch):
    g = _Ortho3()
    monkeypatch.setattr(PS, "ray_dist", lambda g_, P, X: np.linalg.norm(g_.project(X) - P, axis=2))
    X0 = np.array([[1.0, 2.0, 3.0]])
    P = g.project(X0)                      # (3, 1, 2) consistent views
    P[2, 0] += [5.0, 5.0]                  # camera C fooled (a drop / a paw)
    X, used, _d = PS.fuse_robust(g, P, reject_mm=1.0)
    assert list(used[:, 0]) == [True, True, False]
    assert np.allclose(X[0], X0[0])
    # two views left that still disagree -> no point
    P2 = g.project(X0)
    P2[2] = np.nan
    P2[1, 0] += [9.0, 0.0]                 # along x, the coordinate both remaining views measure
    X2, used2, _ = PS.fuse_robust(g, P2, reject_mm=1.0)
    assert np.isnan(X2).all() and not used2.any()


def test_side_params_scale_px_thresholds_and_set_the_side_rules():
    from wfield_local import orofacial_clean as OC
    pt, pj, q = PS.side_params(0.5)
    base_t, base_j = OC.params("tongue", "cam4"), OC.params("jaw", "cam4")
    assert np.isclose(pt["x_range_pm"], 0.5 * float(base_t["x_range_pm"]))
    assert np.isclose(pj["x_range_pm"], 0.5 * float(base_j["x_range_pm"]))
    assert np.isclose(q["y_ceiling_px"], 0.5 * float(base_j["v34"]["y_ceiling_px"]))
    assert pt["max_gap_ms"] == 24.0 and pt["interp_veto_interlick"] is False
    assert q["x_flip_step_px"] is None                      # a profile view has no left / right flip
    assert float(base_j["v34"]["y_ceiling_px"]) == float(OC.params("jaw", "cam4")["v34"]["y_ceiling_px"])   # no mutation
