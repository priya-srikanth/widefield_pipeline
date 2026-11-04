"""Working = `trial_classes.in_working` on both sides (Priya 11-03; `wfield_local.working_trials`).

Pins: the snapshot's classes map onto pooled rows by cue index exactly as 5b's `scripts/s1_refit_locanmf_table.py`
does (reindex of the snapshot column, missing cue -> False); a no-video session has no working rows; the working
selection switches `_class_select` only when the masks are given and the selection is in_working; a working-trained
frozen model gets its own spec id, trains on working rows only and drops a pre session without camera classes; the
quit-tail selection reproduces the previous code.
"""
import numpy as np
import pandas as pd
import pytest

from wfield_local import frozen_models as fm
from wfield_local import grant_kit
from wfield_local import locanmf_frozen_decoder as fd
from wfield_local import working_trials as wt


@pytest.fixture
def snap(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    rows = []
    for lab in ("PSX_0806", "PSX_0807", "PSX_0808", "PSX_0901"):
        for k in range(60):
            cls = rng.choice(["contact", "incomplete", "jaw_only", "no_response"], p=[.7, .1, .1, .1])
            rows.append(dict(label=lab, cue_idx=k, cls=cls, position="far_R"))
    for k in range(40):
        rows.append(dict(label="PSX_0809", cue_idx=k, cls="no_video", position=""))
    D = pd.DataFrame(rows)
    from wfield_local import trial_classes as tc
    D["working_pl_lick"] = tc.in_working(D.cls, D.cue_idx, session=D.label, position=D.position, end="lick")
    f = tmp_path / "snap.csv.gz"
    D.to_csv(f, index=False)
    monkeypatch.setenv(wt.SNAPSHOT_ENV, str(f))
    wt._table.cache_clear()
    yield D
    wt._table.cache_clear()


def test_row_masks_reproduce_the_s1_reindex(snap):
    kept = ["PSX_0806", "PSX_0807", "PSX_0809", "PSX_0901"]
    rng = np.random.default_rng(1)
    idx, GE, GU = {}, [], []
    for i, lab in enumerate(kept):
        n = 40 if lab == "PSX_0809" else 60
        perm = rng.permutation(n)
        ie, inl = np.sort(perm[:45 if n == 60 else 30]), np.sort(perm[45 if n == 60 else 30:])
        idx[lab] = (ie, inl)
        GE += [i] * len(ie)
        GU += [i] * len(inl)
    we, wu, have = wt.row_masks(kept, idx, np.array(GE), np.array(GU))
    assert have == {0, 1, 3}                                    # the no-video session has no classes
    GE, GU = np.array(GE), np.array(GU)
    for i, lab in enumerate(kept):                              # 5b's construction, independently
        sn = snap[snap.label == lab].set_index("cue_idx")
        ie, inl = idx[lab]
        exp_e = sn["working_pl_lick"].reindex(ie).fillna(False).to_numpy(bool)
        exp_u = sn["working_pl_lick"].reindex(inl).fillna(False).to_numpy(bool)
        if lab == "PSX_0809":
            exp_e[:], exp_u[:] = False, False
        assert np.array_equal(we[GE == i], exp_e) and np.array_equal(wu[GU == i], exp_u)


def test_class_select_switches_only_with_masks_and_in_working(monkeypatch):
    se, su = np.array([1, 1, 0, 1], bool), np.array([1, 0, 1], bool)
    ne = np.array([0, 1, 0], bool)
    work = (np.array([1, 0, 1, 1], bool), np.array([1, 1, 0], bool))
    monkeypatch.setenv(wt.WORKING_TRIALS_ENV, "in_working")
    me, mu = grant_kit._class_select("working", se, su, ne, work=work)
    assert me.tolist() == [True, False, False, True] and mu.tolist() == [True, False, False]
    old = grant_kit._class_select("working", se, su, ne)
    assert old[0].tolist() == se.tolist() and old[1].tolist() == (su & ~ne).tolist()
    monkeypatch.setenv(wt.WORKING_TRIALS_ENV, "quit_tail")
    q = grant_kit._class_select("working", se, su, ne, work=work)
    assert q[0].tolist() == old[0].tolist() and q[1].tolist() == old[1].tolist()
    lk = grant_kit._class_select("lick", se, su, ne, work=work)              # the lick gate is untouched
    assert lk[0].tolist() == se.tolist() and not lk[1].any()


def test_working_trained_model_has_its_own_id_and_drops_sessions_without_classes(monkeypatch):
    seen = []
    monkeypatch.setattr(fm, "load_or_fit", lambda spec, fit, **k: (seen.append(spec), (fit(), "new"))[1])
    monkeypatch.setattr(fm, "_session_sig", lambda lab: "sig")
    monkeypatch.setattr(fm.config, "animal_of", lambda lab: lab.split("_")[0])
    rng = np.random.default_rng(2)
    kept = ["PSX_0806", "PSX_0807", "PSX_0808", "PSX_0809", "PSX_0901"]
    n = 120
    YE = np.tile(np.arange(6), 5 * n // 6)
    GE = np.repeat(np.arange(5), n)
    XE = rng.normal(size=(len(YE), 8)) + YE[:, None]
    YU = np.tile(np.arange(6), 5 * 10)
    GU = np.repeat(np.arange(5), 60)
    XU = rng.normal(size=(len(YU), 8)) + YU[:, None]
    we = rng.random(len(YE)) < .9
    wu = rng.random(len(YU)) < .5
    we[GE == 3], wu[GU == 3] = False, False                     # PSX_0809: no video
    work = {"XE": XE, "YE": YE, "GE": GE, "XU": XU, "YU": YU, "GU": GU, "work_e": we, "work_u": wu}
    m, _ = fd.frozen_decoder_models(XE, YE, GE, kept, [0, 1, 2, 3], align="cue", source="roi", log=lambda *a: None)
    mw, _ = fd.frozen_decoder_models(XE, YE, GE, kept, [0, 1, 2, 3], align="cue", source="roi", log=lambda *a: None,
                                     work=work)
    assert seen[1]["scaling_params"]["trials"] == wt.tag() and "trials" not in seen[0]["scaling_params"]
    assert fm.spec_id(seen[0]) != fm.spec_id(seen[1])
    assert seen[1]["train_labels"] == ["PSX_0806", "PSX_0807", "PSX_0808"]
    assert set(mw["loso"]) == {"PSX_0806", "PSX_0807", "PSX_0808"} and mw["trials"] == wt.tag()
    A, B = fd.frozen_scored_frame(mw, XE, GE, XU, GU, kept, [0, 1, 2], work=work)
    assert np.array_equal(A[GE == 3], XE[GE == 3])             # nothing to centre on: left as it was
    assert not np.array_equal(A[GE == 4], XE[GE == 4])


def test_pool_work_is_none_for_the_lick_window_and_quit_tail(monkeypatch):
    class F:
        indices = {}
    monkeypatch.setenv(wt.WORKING_TRIALS_ENV, "in_working")
    assert fd.pool_work(None, None, None, None, None, None, [], F(), "lick") is None
    monkeypatch.setenv(wt.WORKING_TRIALS_ENV, "quit_tail")
    assert fd.pool_work(None, None, None, None, None, None, [], F(), "cue") is None
    assert fd.pool_work(None, None, None, None, None, None, [], object(), "cue") is None
