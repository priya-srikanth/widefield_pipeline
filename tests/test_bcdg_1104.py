"""Priya 11-04, items B / C / D / G (DECISIONS 2026-11-04).

Pins:
  B  the discriminators' spout-frame centre rows: "working" by default (`spout_scaling.state_disc_centre`), "engaged"
     selectable; `precue_engagement_states._working_frame` centres on the camera working rows only.
  C  `spout_scaling.projection_arrays` under WIDEFIELD_PROJECTION_CENTRE=working centres each session on its working
     rows (a FrozenFrame over them), refuses to run without the masks, and is the identity while the frame is off.
  D  `pooled_frozen_loso` scores the camera WORKING rows of a working-trained model by default, and the engaged rows
     under WIDEFIELD_SECTION_D_SCORED=engaged (the pre-11-04 scored set, bit for bit).
  G  `working_trials.g_group` -- the ONE mapping from camera classes to section G's groups (names and sets pinned here
     for the rebuild port); `poststroke_compare._conf_panel` is the old inline confusion loop.
"""
import numpy as np
import pandas as pd
import pytest

from wfield_local import frozen_models as fm
from wfield_local import locanmf_frozen_decoder as fd
from wfield_local import poststroke_compare as pc
from wfield_local import spout_scaling as ss
from wfield_local import working_trials as wt


# ----------------------------------------------------------------------------------------------------------- G
def test_g_group_names_and_sets_are_pinned():
    assert wt.G_GROUPS == ("contact", "incomplete", "jaw_only", "none", "after_end")
    assert wt.G_WORKING == ("contact", "incomplete")
    assert set(wt.G_WORKING) == set(wt.WORKING_CLASSES[wt.IN_WORKING])


def test_g_group_maps_every_camera_class():
    cls = ["contact", "incomplete", "jaw_only", "no_response", "unknown", "no_video", "jaw_only", "no_response"]
    be = [True, True, True, True, True, True, False, False]
    assert list(wt.g_group(cls, be)) == ["contact", "incomplete", "jaw_only", "none", "none", "", "after_end",
                                         "after_end"]


def test_jaw_judged_flags_close_mid_defaults_and_unknown():
    cls = ["no_response", "no_response", "unknown", "jaw_only", "contact"]
    pos = ["close_center", "far_R", "far_R", "far_R", "close_center"]
    assert list(wt.g_jaw_judged(cls, pos)) == [False, True, False, True, True]


def test_row_groups_cut_after_the_last_camera_lick(tmp_path, monkeypatch):
    rows = [dict(label="PSX_0806", cue_idx=k, cls=c, position="far_R")
            for k, c in enumerate(["contact", "jaw_only", "incomplete", "no_response", "jaw_only", "unknown"])]
    rows += [dict(label="PSX_0807", cue_idx=k, cls="no_video", position="") for k in range(3)]
    f = tmp_path / "snap.csv.gz"
    pd.DataFrame(rows).to_csv(f, index=False)
    monkeypatch.setenv(wt.SNAPSHOT_ENV, str(f))
    wt._group_table.cache_clear()
    idx = {"PSX_0806": (np.array([0, 2]), np.array([1, 3, 4, 5])), "PSX_0807": (np.array([0]), np.array([1, 2]))}
    ge, gu, je, ju, have = wt.row_groups(["PSX_0806", "PSX_0807"], idx, np.array([0, 0, 1]), np.array([0] * 4 + [1] * 2))
    wt._group_table.cache_clear()
    assert have == {0}
    assert list(ge) == ["contact", "incomplete", ""]
    assert list(gu) == ["jaw_only", "after_end", "after_end", "after_end", "", ""]
    assert list(ju[:4]) == [True, True, True, False]


def test_conf_panel_is_the_old_inline_loop():
    rng = np.random.default_rng(0)
    labels = [0, 1, 2, 3, 4, 5]
    y, p = rng.integers(0, 6, 200), rng.integers(0, 6, 200)
    y[y == 3] = 4                                   # an empty true row stays NaN
    nl = rng.random(200) < .3
    M = np.full((6, 6), np.nan)
    n, n_nl = [], []
    for i, c in enumerate(labels):
        sel = y == c
        n.append(int(sel.sum()))
        n_nl.append(int(nl[sel].sum()))
        if sel.sum():
            for j, c2 in enumerate(labels):
                M[i, j] = float((p[sel] == c2).mean())
    got = pc._conf_panel(y, p, nl, labels)
    np.testing.assert_array_equal(np.array(got["matrix"]), M)
    assert got["n_per_true_position"] == n and got["n_nolick_per_true_position"] == n_nl
    assert "accuracy" not in got
    got = pc._conf_panel(y, p, nl, labels, jj=np.ones(200, bool))
    assert got["accuracy"] == float(np.mean(p == y)) and got["n_jaw_unjudged"] == 0


def test_switch_defaults(monkeypatch):
    for env in (pc.SECTION_G_GROUPS_ENV, fd.SECTION_D_SCORED_ENV, ss.STATE_DISC_CENTRE_ENV, ss.PROJECTION_CENTRE_ENV):
        monkeypatch.delenv(env, raising=False)
    assert pc.section_g_groups() == "camera"
    assert fd.section_d_scored() == "working"
    assert ss.state_disc_centre() == "working"
    assert ss.projection_centre() == "engaged"          # C is a re-check only: the 11-03 build stays the option
    monkeypatch.setenv(pc.SECTION_G_GROUPS_ENV, "daq")
    monkeypatch.setenv(fd.SECTION_D_SCORED_ENV, "engaged")
    monkeypatch.setenv(ss.STATE_DISC_CENTRE_ENV, "engaged")
    assert (pc.section_g_groups(), fd.section_d_scored(), ss.state_disc_centre()) == ("daq", "engaged", "engaged")
    monkeypatch.setenv(fd.SECTION_D_SCORED_ENV, "bogus")
    with pytest.raises(ValueError):
        fd.section_d_scored()


# ----------------------------------------------------------------------------------------------------------- C
def _pipe():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, C=0.5))


def _pool(seed=0, n_sess=5, n=120, n_u=40):
    rng = np.random.default_rng(seed)
    XE, YE, GE, XU, YU, GU = [], [], [], [], [], []
    for i in range(n_sess):
        y = rng.integers(0, 6, n)
        XE.append(rng.normal(size=(n, 8)) + np.eye(6, 8)[y] * 2 + rng.normal(size=8))
        YE.append(y)
        GE.append(np.full(n, i))
        yu = rng.integers(0, 6, n_u)
        XU.append(rng.normal(size=(n_u, 8)) + np.eye(6, 8)[yu] + rng.normal(size=8))
        YU.append(yu)
        GU.append(np.full(n_u, i))
    XE, YE, GE = np.vstack(XE), np.concatenate(YE), np.concatenate(GE)
    XU, YU, GU = np.vstack(XU), np.concatenate(YU), np.concatenate(GU)
    we = rng.random(len(YE)) < .9
    wu = rng.random(len(YU)) < .3
    wu[GU == 4] = we[GE == 4] = False             # session 4: no working rows (no camera classes)
    return XE, YE, GE, XU, YU, GU, we, wu


def test_projection_working_centre(monkeypatch):
    XE, YE, GE, XU, YU, GU, we, wu = _pool()
    monkeypatch.setenv(ss.PROJECTION_FRAME_ENV, "session_z")
    A, B = ss.projection_arrays(XE, YE, GE, XU, GU, [0, 1, 2], _pipe, work=(we, wu), YU=YU)
    assert A is XE and B is XU
    monkeypatch.setenv(ss.PROJECTION_FRAME_ENV, "spout_frame")
    monkeypatch.setenv(ss.PROJECTION_CENTRE_ENV, "working")
    monkeypatch.delenv(ss.SPOUT_SCALING_ENV, raising=False)
    with pytest.raises(ValueError):
        ss.projection_arrays(XE, YE, GE, XU, GU, [0, 1, 2], _pipe)
    A, B = ss.projection_arrays(XE, YE, GE, XU, GU, [0, 1, 2], _pipe, work=(we, wu), YU=YU)
    RX = np.vstack([XE[we], XU[wu]])
    RY = np.concatenate([YE[we], YU[wu]])
    RG = np.concatenate([GE[we], GU[wu]])
    ff = ss.FrozenFrame(RX, RY, RG, [0, 1, 2], _pipe, loso=True)
    np.testing.assert_allclose(A[GE == 3], ff.apply(XE[GE == 3], 3, "full"))
    np.testing.assert_allclose(B[GU == 0], ff.apply(XU[GU == 0], 0, "loso"))
    np.testing.assert_array_equal(A[GE == 4], XE[GE == 4])       # no working rows: production frame
    # the centre is label-free for a scored session: shuffling its labels leaves its rows unchanged
    YE2 = YE.copy()
    YE2[GE == 3] = np.random.default_rng(5).permutation(YE2[GE == 3])
    A2, _ = ss.projection_arrays(XE, YE2, GE, XU, GU, [0, 1, 2], _pipe, work=(we, wu), YU=YU)
    np.testing.assert_array_equal(A2[GE == 3], A[GE == 3])


# ----------------------------------------------------------------------------------------------------------- D
def test_scored_rows_stack_working_rows_with_their_own_blocks():
    XE, YE, GE, XU, YU, GU, we, wu = _pool()
    work = {"XE": XE, "YE": YE, "GE": GE, "XU": XU, "YU": YU, "GU": GU, "work_e": we, "work_u": wu}
    BE = np.arange(len(YE)) // 6
    Y, G, X, B, NL = fd._scored_rows(work, YE, GE, XE, XU, BE)
    assert len(Y) == we.sum() + wu.sum() and NL.sum() == wu.sum()
    np.testing.assert_array_equal(Y[:we.sum()], YE[we])
    np.testing.assert_array_equal(X[we.sum():], XU[wu])
    assert (B[:we.sum()] >= 0).all() and (B[we.sum():] < 0).all()


@pytest.fixture
def fake_loso(monkeypatch):
    XE, YE, GE, XU, YU, GU, we, wu = _pool(seed=3)
    kept = [f"PSX_08{10 + i}" for i in range(5)]
    BE = [np.arange(int((GE == i).sum())) // 6 for i in range(5)]
    monkeypatch.setattr(fd, "pool_sessions", lambda *a, **k: (XE, YE, GE, BE, XU, YU, kept, np.arange(8), GU))
    monkeypatch.setattr(fd.config, "session_phase", lambda an, d: "pre" if int(d) < 813 else "post")
    monkeypatch.setattr(fd.config, "animal_of", lambda lab: lab.split("_")[0])
    monkeypatch.setattr(fm, "_session_sig", lambda lab: "sig")
    monkeypatch.setattr(fm, "load_or_fit", lambda spec, fit, **k: (fit(), fm.STATUS_NEW))
    work = {"XE": XE, "YE": YE, "GE": GE, "XU": XU, "YU": YU, "GU": GU, "work_e": we, "work_u": wu}
    monkeypatch.setattr(fd, "pool_work", lambda *a, **k: work)
    monkeypatch.setattr(fd, "ood_control", lambda *a, **k: {})
    monkeypatch.setattr(fd.decode_ci, "frozen_ci", lambda *a, **k: None)
    monkeypatch.setenv(wt.WORKING_TRIALS_ENV, "in_working")
    return XE, YE, GE, XU, YU, GU, we, wu, kept


def test_section_d_scores_working_rows_and_engaged_on_request(fake_loso, monkeypatch):
    XE, YE, GE, XU, YU, GU, we, wu, kept = fake_loso
    monkeypatch.setenv(fd.SECTION_D_SCORED_ENV, "engaged")
    r_e = fd.pooled_frozen_loso(kept, verbose=False, features=object())
    monkeypatch.setenv(fd.SECTION_D_SCORED_ENV, "working")
    r_w = fd.pooled_frozen_loso(kept, verbose=False, features=object())
    assert r_e["scored_set"] == "engaged" and r_w["scored_set"] == "working"
    # the engaged-row scores are the same numbers either way (kept for robustness under working)
    assert r_w["per_session_engaged_scored"] == r_e["per_session"]
    # n scored: working rows of each scored session (session 4 has none and leaves the working-scored set)
    for i, lab in enumerate(kept[:4]):
        assert r_w["n_scored"][lab] == int(we[GE == i].sum() + wu[GU == i].sum())
        assert r_e["n_scored"][lab] == int((GE == i).sum())
    assert kept[4] not in r_w["per_session"] and kept[4] in r_e["per_session"]
    # confusion rows sum to the scored rows per true position
    for lab in kept[:4]:
        C = np.array(r_w["confusion"][lab])
        assert C.sum() == r_w["n_scored"][lab]
