"""lick_templates: patterns are baseline-subtracted windows, templates average by position, `compare` picks the
template a lick was built from, the executed-angle class follows the pre contact direction, stats are paired."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wfield_local import config
from wfield_local import lick_templates as LT


@pytest.fixture(autouse=True)
def _no_yaml(monkeypatch):
    monkeypatch.setattr(config, "defaults", lambda session=None: {})


def test_lick_vectors_baselines_and_layout():
    ft = np.arange(0, 20, 0.05)
    A = np.zeros((len(ft), 2))
    A[:, 0] = 1.0                                        # constant offset -> removed by either baseline
    A[(ft >= 10.0) & (ft < 10.5), 1] = 2.0               # a response after the onset at 10 s
    V, ok = LT.lick_vectors(A, ft, [10.0, 19.9], win_s=(0.0, 0.4), base_s=(-0.2, 0.0))
    assert ok.tolist() == [True, False]                  # the second runs off the recording
    v = V[0].reshape(2, -1)                              # region-major
    assert np.allclose(v[0], 0.0) and np.allclose(v[1], 2.0)
    V2, ok2 = LT.lick_vectors(A, ft, [10.0], win_s=(0.0, 0.4), baseline="pre_cue", cue_s=[9.0])
    assert ok2[0] and np.allclose(V2[0].reshape(2, -1)[1], 2.0)
    with pytest.raises(ValueError):
        LT.lick_vectors(A, ft, [10.0], baseline="pre_cue")


def _pre_post(rng):
    """Two pre 'positions' with distinct patterns and angles; post licks cued far_R but built from the close_R
    pattern at close_R's angle (= executed elsewhere)."""
    base = {"far_R": rng.normal(0, 1, 40), "close_R": rng.normal(0, 1, 40)}
    ang = {"far_R": -13.0, "close_R": 0.0}
    rows, V = [], []
    for pos in base:
        for _ in range(30):
            rows.append({"position": pos, "angle": ang[pos] + rng.normal(0, 1.5), "contact": True})
            V.append(base[pos] + rng.normal(0, 0.8, 40))
    pre = pd.DataFrame(rows)
    post = pd.DataFrame({"position": ["far_R"] * 10, "angle": rng.normal(0, 1.5, 10), "contact": False})
    Vpost = np.stack([base["close_R"] + rng.normal(0, 0.8, 40) for _ in range(10)])
    return pre, np.stack(V), post, Vpost


def test_compare_prefers_the_executed_template_when_the_pattern_came_from_it():
    pre, Vpre, post, Vpost = _pre_post(np.random.default_rng(0))
    assert LT.own_template_accuracy(Vpre, pre.position.to_numpy()) > 0.9
    D = LT.compare(pre, Vpre, post, Vpost, angle_tol_deg=4.0, class_deg=10.0, min_matched=5)
    assert (D.cls == "toward mouse-LEFT").all()          # angle ~0 vs the pre far_R contact median ~-13 -> image-right
    assert (D.r_executed > D.r_target).all() and (D.matched_mostly == "close_R").all()
    st = LT.paired_stats(D.r_executed - D.r_target, n_boot=500)
    assert st["n"] == 10 and st["ci_lo"] > 0 and st["p_wilcoxon"] < 0.01
    assert np.isnan(LT.paired_stats([0.1, 0.2])["mean"])


def _angle_world(rng, n_post=40, signal=1.0):
    """Pre licks whose pattern varies SMOOTHLY with executed angle (+ a per-position offset); post licks of two
    cued positions with within-position angle spread; ``signal`` scales the angle information in post patterns."""
    basis_a, basis_b = rng.normal(0, 1, 60), rng.normal(0, 1, 60)
    offs = {"far_R": rng.normal(0, 1, 60), "close_R": rng.normal(0, 1, 60)}
    def pat(pos, ang, s=1.0):
        return offs[pos] + s * (np.cos(np.radians(ang * 3)) * basis_a + np.sin(np.radians(ang * 3)) * basis_b) \
            + rng.normal(0, 0.7, 60)
    pre_rows, Vp = [], []
    for pos, mu in (("far_R", -12.0), ("close_R", 2.0)):
        for _ in range(80):
            a = mu + rng.normal(0, 7)
            pre_rows.append({"position": pos, "angle": a, "contact": True})
            Vp.append(pat(pos, a))
    post_rows, Vq = [], []
    for pos, mu in (("far_R", -2.0), ("close_R", 4.0)):
        for _ in range(n_post // 2):
            a = mu + rng.normal(0, 8)
            post_rows.append({"position": pos, "angle": a, "contact": False})
            Vq.append(pat(pos, a, signal))
    return pd.DataFrame(pre_rows), np.stack(Vp), pd.DataFrame(post_rows), np.stack(Vq)


def test_readout_test_detects_within_position_angle_and_is_calibrated():
    rng = np.random.default_rng(3)
    pre, Vp, post, Vq = _angle_world(rng, signal=1.0)
    c, T, _ = LT.angle_bin_templates(pre, Vp, 8)
    ro = LT.readout_angle(Vq, c, T)
    res = LT.within_position_angle_test(post.position, post.angle, ro, n_perm=500)
    assert res["r"] > 0.3 and res["p_perm"] < 0.01
    ps = []                                              # no angle information in post patterns -> uniform p
    for seed in range(150):
        r = np.random.default_rng(900 + seed)
        pre, Vp, post, Vq = _angle_world(r, signal=0.0)
        c, T, _ = LT.angle_bin_templates(pre, Vp, 8)
        ps.append(LT.within_position_angle_test(post.position, post.angle, LT.readout_angle(Vq, c, T),
                                                n_perm=199, seed=seed)["p_perm"])
    assert 0.0 <= np.mean(np.array(ps) < 0.05) <= 0.09


def test_bh():
    q = LT.bh([0.01, 0.04, 0.03, np.nan, 0.5])
    assert np.isnan(q[3]) and np.allclose(q[[0, 1, 2, 4]], [0.04, 0.16 / 3, 0.16 / 3, 0.5])   # m = 4 finite p
