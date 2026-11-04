"""Coordinator BLOCKER 11-04 / main 48f2cc6d: for cue / precue the frozen spout decoder's TRAINING set, each session's
bal_em CENTRE, the pre panel and the scored set are the camera WORKING rows -- engaged AND no-lick rows of class
contact | incomplete, whatever the DAQ lick (a camera-incomplete trial has no DAQ lick, so it is a no-lick row). The
lick window keeps the engaged (lick-gate) rows."""
from __future__ import annotations

import numpy as np
import pytest

from wfield_local import grant_kit as gk
from wfield_local import spout_scaling as ss

C, I, J, NR = "contact", "incomplete", "jaw_only", "no_response"


@pytest.fixture
def camera(monkeypatch):
    monkeypatch.setenv("WIDEFIELD_TRIAL_GATE", "camera")
    monkeypatch.setenv("WIDEFIELD_EPOCH_FROZEN", "0")              # in-memory fits: nothing stored
    for k in (ss.CENTRE_ENV, ss.SPOUT_SCALING_ENV, "WIDEFIELD_WORKING_TRIALS"):
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def _bundle(align="cue", seed=0):
    """Three pre sessions (0, 1, 2) and one post (3); session 2 has NO camera classes (no video). Engaged rows are
    contact; the no-lick arm holds incomplete / jaw_only / no_response rows."""
    rng = np.random.default_rng(seed)
    GE = np.repeat(np.arange(4), 60)
    YE = np.tile(np.arange(6), 40)
    XE = rng.normal(size=(len(GE), 4)) + 0.9 * YE[:, None] + 0.3 * GE[:, None]
    GU = np.repeat(np.arange(4), 36)
    YU = np.tile(np.arange(6), 24)
    XU = rng.normal(size=(len(GU), 4)) + 0.9 * YU[:, None] + 0.3 * GU[:, None] + 2.0     # a shifted no-lick arm
    cu = np.tile(np.array([I] * 12 + [J] * 12 + [NR] * 12, object), 4)
    ce = np.full(len(GE), C, object)
    ce[GE == 2], cu[GU == 2] = "unknown", "unknown"
    return {"XE": XE, "YE": YE, "GE": GE, "XU": XU, "YU": YU, "GU": GU, "kept": ["A_1", "A_2", "A_3", "A_4"],
            "pre_i": {0, 1, 2}, "e_pre": np.isin(GE, [0, 1, 2]), "cls": (ce, cu), "align": align, "animal": "A",
            "not_eng": None}


def test_working_frame_trains_and_centres_on_engaged_and_nolick_working_rows(camera):
    # the label-free working centre (W) is what this pins; the 11-04 default (Wt) needs >= 20 working rows per spout,
    # which this 12-incomplete-row pool does not have (Wt is pinned in test_bcdg_1104)
    camera.setenv(ss.SCORE_CENTRE_ENV, "working")
    bd = _bundle()
    fr = gk._frozen_frame(bd, sel="working", store=False)
    assert fr["sel"] == "working"
    RX, RY, RG = fr["R"]
    # engaged contact rows + no-lick INCOMPLETE rows (never jaw_only / no_response); session 2 (no video) has none
    assert len(RX) == 3 * (60 + 12)
    assert sorted(np.unique(RG).tolist()) == [0, 1, 3] and fr["train_sessions"] == [0, 1]
    assert set(fr["loso"]) == {0, 1}
    # the no-lick incomplete rows are in the TRAINING rows of session 0
    assert np.isin(bd["XU"][(bd["GU"] == 0) & (bd["cls"][1] == I)], RX).all()
    # the centre of session 3 comes from its working rows, engaged AND no-lick
    mu, sd = ss.score_centre(fr["full"], RX[RG == 3], fr["pi_full"])
    np.testing.assert_allclose(fr["XEs"][bd["GE"] == 3], (bd["XE"][bd["GE"] == 3] - mu) / sd)
    # session 2 (no working rows) is neither trained on nor scored: left in the production frame
    np.testing.assert_array_equal(fr["XEs"][bd["GE"] == 2], bd["XE"][bd["GE"] == 2])


def test_engaged_frame_and_lick_window_keep_the_engaged_rows(camera):
    bd = _bundle()
    bd["kept"] = ["A_1", "A_2", "PS93_0813", "A_4"]          # no video: the engaged frame's contact-row fallback
    fr = gk._frozen_frame(bd, sel="engaged", store=False)
    assert fr["sel"] == "engaged" and len(fr["R"][0]) == 240
    bl = _bundle(align="lick")
    bl["kept"] = bd["kept"]
    lk = gk._frozen_frame(bl, sel="working", store=False)
    assert lk["sel"] == "engaged"                                   # no lick time without a lick


def test_jaw_arm_adds_jaw_only_rows(camera):
    camera.setenv("WIDEFIELD_WORKING_TRIALS", "in_working_jaw")
    camera.setenv(ss.SCORE_CENTRE_ENV, "working")          # W: this pool is too thin per spout for Wt
    fr = gk._frozen_frame(_bundle(), sel="working", store=False)
    assert len(fr["R"][0]) == 3 * (60 + 24)


def test_pool_work_masks_and_working_rows(camera, monkeypatch):
    from wfield_local import locanmf_frozen_decoder as lfd
    from wfield_local import trial_classes as tc
    bd = _bundle()

    class F:
        indices = {"A_1": 1}
    monkeypatch.setattr(tc, "pool_row_classes", lambda feat, kept, ne, nu: bd["cls"])
    w = lfd.pool_work(bd["XE"], bd["YE"], bd["GE"], bd["XU"], bd["YU"], bd["GU"], bd["kept"], F(), "cue")
    RX, RY, RG = lfd.working_rows(w)
    assert len(RX) == 3 * (60 + 12)
    assert lfd.pool_work(bd["XE"], bd["YE"], bd["GE"], bd["XU"], bd["YU"], bd["GU"], bd["kept"], F(), "lick") is None


def test_working_spec_records_the_selection(camera):
    kw = ss.spec_kw("session_bal_em", "sel-x")
    assert kw["scaling_params"]["trials"] == "sel-x" and kw["scaling_params"]["centre_set"] == "working_rows"
    assert "trials" not in ss.spec_kw("session_bal_em")["scaling_params"]
    with pytest.raises(ss.BalEmCentreRefused):
        ss.working_centre(None, np.zeros((5, 3)), None, "S")      # 1..19 working rows refuse
    assert ss.working_centre(None, np.zeros((0, 3)), None, "S") is None
