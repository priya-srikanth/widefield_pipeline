"""The FaceRhythm analysis layer (`wfield_local/face_rhythm/`): constants, face speed, licks, statistics."""
import numpy as np
import pandas as pd

from wfield_local.face_rhythm import common, face, licks, stats


def test_positions_one_order():
    assert common.POSITIONS_CONTRA_FIRST == tuple(reversed(common.POSITIONS))
    assert {common.SIDE[p] for p in ("far_R", "close_R")} == {"contra"}
    assert {common.SIDE[p] for p in ("far_L", "close_L")} == {"ipsi"}
    g = common.position_groups(np.array(["far_R", "close_L", "far_center"]), "side")
    assert list(g) == ["contra", "centre", "ipsi"] and g["contra"].tolist() == [True, False, False]
    assert list(common.position_groups(np.array(["far_R"]), "position")) == list(common.POSITIONS_CONTRA_FIRST)


def test_side_speed_splits_halves():
    # 2 points, PC space = identity on the interleaved (x0, y0, x1, y1) layout; point 0 moves 1 px / s in x
    t = np.arange(5, dtype=float)
    pcs = np.zeros((5, 4), np.float32)
    pcs[:, 0] = t
    comps = np.eye(4, dtype=np.float32)
    c, i, S = face.side_speed(pcs, t, comps, np.array([True, False]), W=np.ones((5, 1)))
    assert np.isnan(c[0]) and np.allclose(c[1:], 1) and np.allclose(i[1:], 0)
    assert np.allclose(S[:, 0], [4, 0])                 # frames 1..4 summed (frame 0 has no velocity)


def test_window_weights_and_alignment():
    t = np.arange(0, 2, 0.04)
    w = face.window_weights(t, np.array([0.4, 0.44]), (0.0, 0.08), np.ones(len(t), bool))
    assert w[np.searchsorted(t, 0.44)] == 2               # two overlapping event windows count twice
    y = t.copy()
    m = face.aligned_mean(t, y, np.array([0.4, 0.8]), np.array([0.0, 0.04]))
    assert np.allclose(m, [0.6, 0.64])
    assert np.isnan(face.aligned_mean(t, y, np.array([5.0]), np.array([0.0]))).all()   # off the recording


def test_bout_classes_and_latency():
    L = pd.DataFrame({"daq_on_s": [10.3, 10.45, 11.5, 20.9, 21.0],
                      "trial_id": [1, 1, 1, 2, 2], "bout_idx": [0, 0, 1, 0, 0], "n_in_bout": [1, 2, 1, 1, 2],
                      "t_from_cue": [0.3, 0.45, 1.5, 0.9, 1.0], "cue_idx": [0, 0, 0, 1, 1]})
    C = licks.bout_classes(L, leadin=0.5, late_s=0.6)
    assert C["first"].tolist() == [True, False, False, True, False]
    assert C["later"].tolist() == [False, False, True, False, False]
    assert C["first_late"].tolist() == [False, False, False, True, False]
    assert C["carry"].tolist() == [False] * 5             # every onset follows >= 0.5 s without a lick
    assert np.allclose(licks.first_lick_latency(L, 3)[:2], [0.3, 0.9]) and np.isnan(licks.first_lick_latency(L, 3)[2])


def _sessions():
    rows = []
    for an, shift in (("PS92", -1.0), ("PS93", -0.5), ("PS94", -0.8)):
        for ep, n in (("pre", 3), ("acute", 2)):
            for s in range(n):
                v = 2.0 ** ((shift if ep == "acute" else 0.0) + 0.1 * s)
                rows.append({"animal": an, "epoch": ep, "label": f"{an}_{ep}{s}", "g": "x", "v": v})
    return pd.DataFrame(rows)


def test_log2_vs_pre_centres_pre():
    S = _sessions()
    S["l"] = stats.log2_vs_pre(S, "v", ["animal", "g"])
    assert np.allclose(S[S.epoch == "pre"].groupby("animal").l.mean(), 0)
    assert np.isclose(S[(S.animal == "PS92") & (S.epoch == "acute")].l.mean(), -1.0 + 0.05 - 0.1)


def test_epoch_change_is_boot_delta():
    from wfield_local import analysis_kit as ak
    S = _sessions()
    S["l"] = stats.log2_vs_pre(S, "v", ["animal", "g"])
    r = stats.epoch_change(S, "l", np.random.default_rng(0), epochs=("acute",))[0]
    pre = {an: list(g.l) for an, g in S[S.epoch == "pre"].groupby("animal")}
    post = {an: list(g.l) for an, g in S[S.epoch == "acute"].groupby("animal")}
    iv = ak.boot_delta(post, pre, np.random.default_rng(0))
    assert (r["delta"], r["ci_lo"], r["ci_hi"]) == (iv.point, iv.lo, iv.hi)      # EXACT: same draws (rule 9)
    assert r["n_below"] == 3 and r["excludes_0"]
    assert stats.tick_label("acute", r).startswith("acute\n")


def test_draws_do_not_change_the_interval_and_holm():
    from wfield_local import analysis_kit as ak
    post = {"A": [1.0, 1.2], "B": [0.8, 1.1, 0.9], "C": [1.3]}
    pre = {"A": [0.1, 0.0], "B": [0.2], "C": [-0.1, 0.1]}
    iv = ak.boot_delta(post, pre, np.random.default_rng(3), 500)
    iv2, o = ak.boot_delta(post, pre, np.random.default_rng(3), 500, draws=True)
    assert iv == iv2 and len(o) == 500                     # draws=True returns the SAME interval (same rng sequence)
    assert ak.boot_p(o) == 1 / 500                        # every draw > 0
    assert np.allclose(ak.holm([0.01, 0.04, 0.03, np.nan])[:3], [0.03, 0.06, 0.06])
    T = pd.DataFrame({"fam": ["x", "x", "x", "y"], "p_boot": [0.001, 0.03, 0.04, 0.04],
                      "excludes_0": [True, True, True, True]})
    assert stats.add_holm(T, ["fam"])["mark"].tolist() == ["**", "*", "*", "**"]      # x: .003 .06 .06; y: .04


def test_catalogue_points_at_real_scripts():
    from wfield_local.face_rhythm.__main__ import CATALOGUE, REPO, STAGES
    for name, a in CATALOGUE.items():
        f = REPO / "scripts" / f"{a.script}.py"
        assert f.exists(), (name, f)
        assert f.read_text(encoding="utf-8").lstrip().startswith('"""'), name      # every script says what it does
        assert a.stage in STAGES and a.question and a.outputs, name


def test_boot_delta_columns_is_boot_delta_per_column():
    from wfield_local import analysis_kit as ak
    g = np.random.default_rng(1)
    post = {a: g.normal(0.3, 1, (n, 5)) for a, n in (("A", 4), ("B", 9), ("C", 2), ("D", 6))}
    pre = {a: g.normal(0, 1, (n, 5)) for a, n in (("A", 3), ("B", 5), ("C", 7), ("D", 1))}
    pt, lo, hi, o, n = ak.boot_delta_columns(post, pre, np.random.default_rng(7), 300)
    for c in range(5):
        iv = ak.boot_delta({a: list(v[:, c]) for a, v in post.items()}, {a: list(v[:, c]) for a, v in pre.items()},
                           np.random.default_rng(7), 300)
        assert np.allclose([pt[c], lo[c], hi[c]], [iv.point, iv.lo, iv.hi], rtol=0, atol=1e-12)   # same draw
    assert n == 4 and o.shape == (300, 5)
    assert ak.bh([0.001, 0.02, 0.03, 0.5, np.nan]).tolist() == [True, True, True, False, False]
