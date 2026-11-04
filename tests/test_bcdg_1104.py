"""Priya 11-04, items B / C / D / G (DECISIONS 2026-11-04); main 4a37cbc8 ported to the cohort-1 rebuild.

Pins:
  B  the discriminators' spout-frame centre rows: "working" by default (`spout_scaling.state_disc_centre`), "engaged"
     selectable; `precue_engagement_states._working_frame` centres on the camera working rows only.
  C  `spout_scaling.projection_arrays` under WIDEFIELD_PROJECTION_CENTRE=working centres each session on its working
     rows (a FrozenFrame over them), refuses to run without the masks, and is the identity while the frame is off.
  D  `pooled_frozen_loso` scores the camera WORKING rows of a working-trained model by default, and the engaged rows
     under WIDEFIELD_SECTION_D_SCORED=engaged (the pre-11-04 scored set, bit for bit).
  G  `working_trials.g_group` -- the ONE mapping from camera classes to section G's groups (names and sets pinned here
     for the rebuild port); `poststroke_compare._conf_panel` is the old inline confusion loop.
REBUILD: `row_groups` reads the LIVE classes (`trial_classes.cue_groups_for`: raw class + end flag), so the rebuild's
stopped / early_run classes map through their raw class (early_run -> none, Priya 11-04) and after_end through the flag.
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


def test_group_table_cut_after_the_last_camera_lick(tmp_path, monkeypatch):
    """Main's snapshot table (kept for parity): the same groups as main's test."""
    rows = [dict(label="PSX_0806", cue_idx=k, cls=c, position="far_R")
            for k, c in enumerate(["contact", "jaw_only", "incomplete", "no_response", "jaw_only", "unknown"])]
    rows += [dict(label="PSX_0807", cue_idx=k, cls="no_video", position="") for k in range(3)]
    f = tmp_path / "snap.csv.gz"
    pd.DataFrame(rows).to_csv(f, index=False)
    wt._group_table.cache_clear()
    T = wt._group_table(str(f))
    wt._group_table.cache_clear()
    assert T["PSX_0807"] == {}
    assert [T["PSX_0806"][k][0] for k in range(6)] == ["contact", "jaw_only", "incomplete", "after_end", "after_end",
                                                       "after_end"]
    assert T["PSX_0806"][5][1] is False or not T["PSX_0806"][5][1]


def _fake_session(monkeypatch, raw_by_label, after_by_label):
    from wfield_local import trial_classes as tc
    monkeypatch.setattr(tc, "daq_cue_seconds", lambda lab: np.arange(len(raw_by_label[lab]), dtype=float))
    monkeypatch.setattr(tc, "cue_classes_for", lambda lab, cs: np.asarray(raw_by_label[lab], dtype=object))
    monkeypatch.setattr(tc, "cue_groups_for", lambda lab, cs: (
        np.asarray(raw_by_label[lab], dtype=object), np.asarray(after_by_label[lab], bool),
        np.asarray([c != "unknown" for c in raw_by_label[lab]], bool) if lab != "PSX_0807"
        else np.zeros(len(raw_by_label[lab]), bool)))


def test_row_groups_live_cut_after_the_last_camera_lick(monkeypatch):
    raw = {"PSX_0806": ["contact", "jaw_only", "incomplete", "no_response", "jaw_only", "unknown"],
           "PSX_0807": ["unknown"] * 3}
    after = {"PSX_0806": [False, False, False, True, True, True], "PSX_0807": [False] * 3}
    _fake_session(monkeypatch, raw, after)
    idx = {"PSX_0806": (np.array([0, 2]), np.array([1, 3, 4, 5])), "PSX_0807": (np.array([0]), np.array([1, 2]))}
    ge, gu, je, ju, have = wt.row_groups(["PSX_0806", "PSX_0807"], idx, np.array([0, 0, 1]),
                                         np.array([0] * 4 + [1] * 2), YE=np.array([3, 3, 3]), YU=np.array([3] * 6))
    assert have == {0}
    assert list(ge) == ["contact", "incomplete", ""]
    assert list(gu) == ["jaw_only", "after_end", "after_end", "after_end", "", ""]
    assert list(ju[:4]) == [True, True, True, False]


def test_g_group_rebuild_classes_route_through_their_raw_class(monkeypatch):
    """The rebuild's session-level classes: early_run (raw no_response) -> none; stopped (raw no_response / jaw_only)
    -> none / jaw_only inside the working period, after_end after the last camera lick; close-mid no_response is
    jaw-unjudged (kept apart by the flag)."""
    from wfield_local import trial_classes as tc
    raw = ["contact", "no_response", "no_response", "jaw_only", "incomplete", "no_response", "jaw_only", "no_response"]
    T = pd.DataFrame({"cue_daq": np.arange(8, dtype=float), "cls": raw, "position": ["far_R"] * 8})
    monkeypatch.setattr(tc, "session_trials", lambda lab, window=None: T)
    tc._CUE_GRP.clear()
    r, after, matched = tc.cue_groups_for("PSX_0806", np.arange(8, dtype=float))
    tc._CUE_GRP.clear()
    assert matched.all() and list(after) == [False] * 5 + [True] * 3
    assert list(wt.g_group(r, ~after)) == ["contact", "none", "none", "jaw_only", "incomplete", "after_end",
                                            "after_end", "after_end"]
    # an uncovered tail (video ends early): no end cutoff, as classify_session
    m = [True] * 2 + [False] * 6
    T2 = T.iloc[:2]
    monkeypatch.setattr(tc, "session_trials", lambda lab, window=None: T2)
    tc._CUE_GRP.clear()
    _r, after2, matched2 = tc.cue_groups_for("PSX_0806", np.arange(8, dtype=float))
    tc._CUE_GRP.clear()
    assert list(matched2) == m and not after2.any()
    assert list(wt.g_jaw_judged(["no_response", "no_response"], ["close_center", "far_R"])) == [False, True]
    assert wt.G_LABELS["after_end"] == "quit"


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
    for env in (pc.SECTION_G_GROUPS_ENV, fd.SECTION_D_SCORED_ENV, ss.STATE_DISC_CENTRE_ENV, ss.PROJECTION_CENTRE_ENV,
                ss.SCORE_CENTRE_ENV, ss.STATE_DISC_SCALING_ENV, ss.PROJECTION_FRAME_ENV):
        monkeypatch.delenv(env, raising=False)
    assert ss.state_disc_scaling() is None and ss.STATE_DISC_DEFAULT == "session_z"     # B stays OFF
    assert ss.projection_frame_on()                      # C ON since Priya's final 11-04 decision (Wt centre)
    # Priya via a8 11-04 (provisional): the scored-frame centre defaults to Wt; W / C / Cto / N selectable
    assert ss.score_centre_rows() == "wt" and ss.score_centre_tag() == "wt_centre"
    for v, k in (("working", "working"), ("contact", "contact"), ("cto", "cto"), ("n", "n")):
        monkeypatch.setenv(ss.SCORE_CENTRE_ENV, v)
        assert ss.score_centre_rows() == k
    assert ss.score_centre_tag() == "n_centre"
    monkeypatch.setenv(ss.SCORE_CENTRE_ENV, "bogus")
    with pytest.raises(ValueError):
        ss.score_centre_rows()
    monkeypatch.delenv(ss.SCORE_CENTRE_ENV)
    assert pc.section_g_groups() == "camera"
    assert fd.section_d_scored() == "working"
    assert ss.state_disc_centre() == "working"
    assert ss.projection_centre() == "wt"               # follows the centre decision; the frame itself stays OFF
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
    ce = np.where(we, "contact", "")
    cu = np.where(wu, "incomplete", "")
    work = (we, wu, ce, cu)
    A, B = ss.projection_arrays(XE, YE, GE, XU, GU, [0, 1, 2], _pipe, work=work, YU=YU)
    assert A is XE and B is XU
    monkeypatch.setenv("WIDEFIELD_TRIAL_GATE", "camera")
    monkeypatch.setenv(ss.PROJECTION_FRAME_ENV, "spout_frame")
    monkeypatch.setenv(ss.PROJECTION_CENTRE_ENV, "working")
    monkeypatch.delenv(ss.SPOUT_SCALING_ENV, raising=False)
    # no masks = the engaged-row frame (lick window / DAQ gate), which under the camera centre needs the engaged
    # rows' classes: it fails closed without them
    with pytest.raises((ValueError, ss.BalEmCentreRefused)):
        ss.projection_arrays(XE, YE, GE, XU, GU, [0, 1, 2], _pipe)
    A, B = ss.projection_arrays(XE, YE, GE, XU, GU, [0, 1, 2], _pipe, work=work, YU=YU)
    RX = np.vstack([XE[we], XU[wu]])
    RY = np.concatenate([YE[we], YU[wu]])
    RG = np.concatenate([GE[we], GU[wu]])
    ff = ss.FrozenFrame(RX, RY, RG, [0, 1, 2], _pipe, loso=True, cls_e=np.full(len(RX), "contact", dtype=object))
    np.testing.assert_allclose(A[GE == 3], ff.apply(XE[GE == 3], 3, "full"))
    np.testing.assert_allclose(B[GU == 0], ff.apply(XU[GU == 0], 0, "loso"))
    np.testing.assert_array_equal(A[GE == 4], XE[GE == 4])       # no working rows: production frame
    # the centre is label-free for a scored session: shuffling its labels leaves its rows unchanged
    YE2 = YE.copy()
    YE2[GE == 3] = np.random.default_rng(5).permutation(YE2[GE == 3])
    A2, _ = ss.projection_arrays(XE, YE2, GE, XU, GU, [0, 1, 2], _pipe, work=work, YU=YU)
    np.testing.assert_array_equal(A2[GE == 3], A[GE == 3])
    # the CONTACT centre (coordinator 11-04): same model, each session centred on its contact rows only
    monkeypatch.setenv(ss.PROJECTION_CENTRE_ENV, "contact")
    A3, _ = ss.projection_arrays(XE, YE, GE, XU, GU, [0, 1, 2], _pipe, work=work, YU=YU)
    ffc = ss.FrozenFrame(RX, RY, RG, [0, 1, 2], _pipe, loso=True, cls_e=np.full(len(RX), "contact", dtype=object))
    ffc.cm = np.concatenate([ce[we], cu[wu]]) == "contact"
    np.testing.assert_allclose(A3[GE == 3], ffc.apply(XE[GE == 3], 3, "full"))
    assert not np.allclose(A3[GE == 3], A[GE == 3])
    assert ss.projection_tag() == "spout_frame/contact"


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
    return _fake_loso(monkeypatch)


@pytest.fixture
def fake_loso_big(monkeypatch):
    return _fake_loso(monkeypatch, n=600)


def _fake_loso(monkeypatch, n=120):
    XE, YE, GE, XU, YU, GU, we, wu = _pool(seed=3, n=n)
    # REBUILD: the session without working rows (index 4) is a PRE session (as PS93_0813, no video): a POST session
    # with none is refused by the rebuild's working centre (review 7A NB4) instead of being left unscored
    kept = ["PSX_0810", "PSX_0811", "PSX_0812", "PSX_0815", "PSX_0809"]
    BE = [np.arange(int((GE == i).sum())) // 6 for i in range(5)]
    monkeypatch.setattr(fd, "pool_sessions", lambda *a, **k: (XE, YE, GE, BE, XU, YU, kept, np.arange(8), GU))
    monkeypatch.setattr(fd.config, "session_phase", lambda an, d: "pre" if int(d) < 813 else "post")
    monkeypatch.setenv("WIDEFIELD_TRIAL_GATE", "camera")
    monkeypatch.setattr(fd.config, "animal_of", lambda lab: lab.split("_")[0])
    monkeypatch.setattr(fm, "_session_sig", lambda lab: "sig")
    monkeypatch.setattr(fm, "load_or_fit", lambda spec, fit, **k: (fit(), fm.STATUS_NEW))
    work = {"XE": XE, "YE": YE, "GE": GE, "XU": XU, "YU": YU, "GU": GU, "work_e": we, "work_u": wu}
    monkeypatch.setattr(fd, "pool_work", lambda *a, **k: work)
    monkeypatch.setattr(fd, "ood_control", lambda *a, **k: {})
    monkeypatch.setattr(fd.decode_ci, "frozen_ci", lambda *a, **k: None)
    monkeypatch.setenv(wt.WORKING_TRIALS_ENV, "in_working")
    # this synthetic pool has < 20 working rows at some spouts, which the Wt centre refuses by design: the D tests
    # run on the label-free working centre (W); the Wt centre has its own tests below
    monkeypatch.setenv(ss.SCORE_CENTRE_ENV, "working")
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
    # n scored: working rows of each scored session (session 4 has none: no LOSO model, unscored under both sets)
    for i, lab in enumerate(kept[:4]):
        assert r_w["n_scored"][lab] == int(we[GE == i].sum() + wu[GU == i].sum())
        assert r_e["n_scored"][lab] == int((GE == i).sum())
    assert kept[4] not in r_w["per_session"] and kept[4] not in r_e["per_session"]
    assert r_w["scored_rows"] == "working" and r_e["scored_rows"] == "engaged"
    assert r_w["scored_centre"] == "working_rows"
    # confusion rows sum to the scored rows per true position
    for lab in kept[:4]:
        C = np.array(r_w["confusion"][lab])
        assert C.sum() == r_w["n_scored"][lab]


# ------------------------------------------------------------------------ the centre kinds (Priya via a8, 11-04)
def test_wt_is_the_equal_weight_mean_of_spout_means_and_refuses_thin_spouts():
    rng = np.random.default_rng(0)
    Y = np.repeat(np.arange(6), [20, 25, 30, 40, 60, 90])
    X = rng.normal(size=(len(Y), 4)) + Y[:, None]
    mu, sd = ss.true_label_centre(X, Y, "PSX_0901")
    np.testing.assert_allclose(mu, np.mean([X[Y == k].mean(0) for k in range(6)], 0))
    np.testing.assert_allclose(sd, X.std(0))
    with pytest.raises(ss.BalEmCentreRefused):                 # 19 rows at one spout: refused, no fill / fallback
        ss.true_label_centre(X[1:], Y[1:], "PSX_0901")


def test_cto_and_n_follow_the_check_definitions():
    rng = np.random.default_rng(1)
    RX, RY, RG, RC = [], [], [], []
    for s in range(3):
        y = np.tile(np.arange(6), 30)
        RX.append(rng.normal(size=(len(y), 3)) + y[:, None] * .3 + s)
        RY.append(y)
        RG.append(np.full(len(y), s))
        RC.append(np.where(rng.random(len(y)) < .7, "contact", "incomplete"))
    RX, RY, RG, RC = np.vstack(RX), np.concatenate(RY), np.concatenate(RG), np.concatenate(RC).astype(object)
    refs = ss.centre_refs(RX, RY, RG, RC, [0, 1, 2])
    m = RG == 2
    mu, sd = ss.scored_centre("cto", Xw=RX[m], Yw=RY[m], Cw=RC[m], refs=refs, train=[0, 1])
    c = m & (RC == "contact")
    d = np.nanmean([refs[s]["dC"] for s in (0, 1)], 0)
    p = np.array([(RY[c] == k).sum() for k in range(6)], float)
    np.testing.assert_allclose(sd, RX[c].std(0))
    np.testing.assert_allclose(mu, RX[c].mean(0) - sd * ((p / p.sum()) @ d))
    stats = {s: (rng.normal(size=3), rng.random(3) + .5) for s in range(3)}
    mu, sd = ss.scored_centre("n", Xw=RX[m], Yw=RY[m], Cw=RC[m], refs=refs, train=[0, 1], stats=stats, i=2)
    raw_c = np.mean([stats[s][0] + stats[s][1] * refs[s]["c"] for s in (0, 1)], 0)
    raw_sd = np.mean([stats[s][1] * refs[s]["sdw"] for s in (0, 1)], 0)
    np.testing.assert_allclose(mu, (raw_c - stats[2][0]) / stats[2][1])
    np.testing.assert_allclose(sd, raw_sd / stats[2][1])
    assert ss.scored_centre("wt", Xw=RX[:0], Yw=RY[:0], Cw=RC[:0]) is None   # no working rows: unscored


def test_wt_scores_section_d_and_carries_its_tag(fake_loso_big, monkeypatch):
    """Wt through pooled_frozen_loso on a pool with >= 20 working rows per spout: the scored frame equals
    (X - Wt centre) / working SD for a post session, and the result is tagged."""
    XE, YE, GE, XU, YU, GU, we, wu, kept = fake_loso_big     # >= 20 working rows per spout in every session
    monkeypatch.setenv(ss.SCORE_CENTRE_ENV, "wt")
    seen = {}
    real = fd.frozen_scored_frame

    def _spy(*a, **k):
        seen["X"] = real(*a, **k)
        return seen["X"]

    monkeypatch.setattr(fd, "frozen_scored_frame", _spy)
    r = fd.pooled_frozen_loso(kept, verbose=False, features=object())
    assert r["scored_centre"] == "wt_centre"
    i = 3                                                     # the post session
    Xw = np.vstack([XE[(GE == i) & we], XU[(GU == i) & wu]])
    Yw = np.concatenate([YE[(GE == i) & we], YU[(GU == i) & wu]])
    mu = np.mean([Xw[Yw == k].mean(0) for k in range(6)], 0)
    np.testing.assert_allclose(seen["X"][0][GE == i], (XE[GE == i] - mu) / Xw.std(0))


def test_g_group_final_keeps_stopped_licking_and_reads_a_stop_kind():
    """cue70 merge (coordinator 11-04): stopped_licking is its own group; with a run / tail marker a stopped trial of the
    quit tail is "after_end" ("quit") and one of a mid-session run "stopped"; without it the raw-class route holds."""
    final = ["contact", "stopped", "stopped", "stopped_licking", "stopped"]
    raw = ["contact", "no_response", "jaw_only", "contact", "no_response"]
    be = [True, True, True, True, False]
    assert list(wt.g_group_final(final, raw, be)) == ["contact", "none", "jaw_only", "stopped_licking", "after_end"]
    kind = ["", "run", "run", "tail", "tail"]
    assert list(wt.g_group_final(final, raw, be, stop_kind=kind)) == ["contact", "stopped", "stopped",
                                                                      "stopped_licking", "after_end"]
    assert wt.G_ALL == wt.G_GROUPS + ("stopped", "stopped_licking") and wt.G_LABELS["after_end"] == "quit"
