"""The spout world frame (wfield_local/spout_world.py): the command is ported exactly, a per-session similarity
fit recovers a known camera transform INCLUDING a June-like scale error, non-nominal (adaptive) samples never
define the frame, and the frame places them at their read-back distance."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from wfield_local import spout_world as SW

GEOM = {"dist_close": "2", "dist_far": "4", "az_center": "0.0", "az_left": "-60", "az_right": "60",
        "down_angle": "45", "head_roll": "0"}


def _rot(a, b, c):
    ca, sa, cb, sb, cc, sc = np.cos(a), np.sin(a), np.cos(b), np.sin(b), np.cos(c), np.sin(c)
    Rz = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
    Ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    Rx = np.array([[1, 0, 0], [0, cc, -sc], [0, sc, cc]])
    return Rz @ Ry @ Rx


def _session(rng, inflate=1.0, n=20, noise=0.05, adaptive_far_L=None):
    """Triangulated spout samples: commanded mm -> calibration space by a known rotation, translation and
    1/scale (``inflate`` > 1 = triangulated distances too large, as June is with the 09-11 calibration)."""
    R_true, t_cal = _rot(0.4, -0.3, 1.1), np.array([10.0, -40.0, 120.0])
    offset = np.array([0.3, -0.2, 0.4])                       # constant label-vs-tip offset (mm)
    rows = []
    for pos in SW.POSITIONS:
        dists = [SW.nominal_distance(GEOM, pos)] * n
        if pos == "far_L" and adaptive_far_L:
            dists += [adaptive_far_L] * n
        for d in dists:
            p_mm = SW.commanded_world(GEOM, pos, d) + offset     # camera space is right-handed
            X = inflate * (R_true.T @ p_mm) + t_cal + rng.normal(0, noise, 3)
            rows.append({"pos": pos, "X": X, "dist_mm": d})
    return pd.DataFrame(rows), R_true


def test_commanded_offset_is_the_firmware_formula():
    c = SW.commanded_offset(GEOM, "close_center")
    assert np.allclose(c, [0, -2 * np.cos(np.pi / 4), -2 * np.sin(np.pi / 4)])
    fl, fr = SW.commanded_offset(GEOM, "far_L"), SW.commanded_offset(GEOM, "far_R")
    assert fl[0] < 0 < fr[0] and np.isclose(np.linalg.norm(fl), 4) and np.isclose(fl[1], fr[1])
    assert np.isclose(np.linalg.norm(fl - fr), 2 * 4 * np.cos(np.pi / 4) * np.sin(np.pi / 3))


def test_fit_recovers_the_layout_and_a_june_like_scale_error():
    rng = np.random.default_rng(0)
    for inflate in (1.0, 1.07):
        pts, _ = _session(rng, inflate)
        w = SW.fit(pts, GEOM)
        assert np.isclose(w.scale, 1 / inflate, atol=0.01)
        assert max(w.residual_mm.values()) < 0.1
        assert np.allclose(w.distances.ratio, 1 / inflate, atol=0.02)


def test_the_left_handed_stage_frame_cannot_be_reached_by_a_rotation():
    """Why the fit is in (ml, ap, dv): the mirror-image stage layout fits badly however it is rotated."""
    pts, _ = _session(np.random.default_rng(4), noise=0.0)
    med, _, _ = SW.position_medians(pts, GEOM)
    X = np.stack([med[p] for p in SW.POSITIONS])
    for target, ok in ((SW.commanded_world, True), (SW.commanded_offset, False)):
        Y = np.stack([target(GEOM, p) for p in SW.POSITIONS])
        s, R, t = SW.umeyama(X, Y)
        res = np.linalg.norm((s * (R @ X.T)).T + t - Y, axis=1).max()
        assert (res < 1e-6) if ok else (res > 0.3)


def test_world_axes_signs():
    pts, _ = _session(np.random.default_rng(1), noise=0.0)
    w = SW.fit(pts, GEOM)
    med, _, _ = SW.position_medians(pts, GEOM)
    W = {p: w.to_world(med[p]) for p in med}
    assert W["far_R"][0] > 0 > W["far_L"][0]                  # ml: + mouse right
    assert W["far_center"][1] > W["close_center"][1] > 0      # ap: + out of the mouth toward the spouts
    assert W["far_center"][2] < W["close_center"][2] < 0      # dv: spouts are below the mouth


def test_adaptive_samples_never_define_the_frame_and_are_placed_at_their_readback_distance():
    pts, _ = _session(np.random.default_rng(2), adaptive_far_L=3.5)
    w = SW.fit(pts, GEOM)
    assert w.excluded == {"far_L": 20} and w.n["far_L"] == 20
    assert np.isclose(w.scale, 1.0, atol=0.01)
    chk = SW.step_check(w, pts, GEOM)
    assert len(chk) == 20 and chk.error_mm.abs().max() < 0.15


def test_too_few_positions_is_a_refusal():
    pts, _ = _session(np.random.default_rng(3))
    with pytest.raises(ValueError, match="spout positions"):
        SW.fit(pts[pts.pos.isin(["far_L", "far_R", "far_center", "close_L"])], GEOM)


def test_session_rig_and_stage_samples(tmp_path):
    (tmp_path / "gui_config.json").write_text(json.dumps(
        {"mouth": ["62.5", "24.7", "-66.0"], "geometry": GEOM, "adaptive": {"enabled": True}}), encoding="utf-8")
    pd.DataFrame({"device_t_ms": [1, 2, 3], "state": ["pre_cue", "settle", "pre_cue"], "trial_id": [1, 1, 2],
                  "pos_name": ["far_L", "far_L", "far_L"], "x_mm": [1, 2, 3], "y_mm": [1, 2, 3], "z_mm": [1, 2, 3],
                  "pos_dist_mm": [4.0, 4.0, 3.75]}).to_csv(tmp_path / "events.csv", index=False)
    rig = SW.session_rig(tmp_path)
    assert rig["adaptive"] and np.allclose(rig["mouth"], [62.5, 24.7, -66.0])
    s = SW.stage_samples(tmp_path)
    assert list(s.device_t_ms) == [1, 3] and list(s.nominal) == [True, False]
