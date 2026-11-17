"""bal_em EM center rows (Priya 11-03, "camera gate everywhere"; log 8.16) and the june_august_drift month model stored
in the bal_em frame (8.17)."""
from __future__ import annotations

import numpy as np
import pytest

from wfield_local import frozen_models as fm
from wfield_local import spout_scaling as ss


@pytest.fixture
def camera(monkeypatch):
    monkeypatch.setenv("WIDEFIELD_TRIAL_GATE", "camera")
    monkeypatch.delenv(ss.CENTRE_ENV, raising=False)
    monkeypatch.delenv(ss.SPOUT_SCALING_ENV, raising=False)
    return monkeypatch


def test_centre_rows_follow_the_gate(camera):
    assert ss.centre_rows() == "working"
    camera.setenv(ss.CENTRE_ENV, "engaged")
    assert ss.centre_rows() == "engaged"
    camera.setenv("WIDEFIELD_TRIAL_GATE", "daq")
    camera.delenv(ss.CENTRE_ENV)
    assert ss.centre_rows() == "engaged"
    camera.setenv(ss.CENTRE_ENV, "working")
    with pytest.raises(ss.BalEmCentreRefused, match="camera trial gate"):
        ss.centre_rows()
    camera.setenv(ss.CENTRE_ENV, "bogus")
    with pytest.raises(ss.BalEmCentreRefused):
        ss.centre_rows()


def test_centre_mask_is_the_working_classes_and_fails_closed(camera):
    cls = np.array(["contact", "incomplete", "jaw_only", "after_end", "unknown", "stopped", "early_run",
                    "no_response"], object)
    # Priya 11-03 late: working = contact + incomplete (jaw_only its own class)
    assert ss.centre_mask(cls).tolist() == [True, True, False, False, False, False, False, False]
    with pytest.raises(ss.BalEmCentreRefused, match="WORKING rows are required"):
        ss.centre_mask(None)
    camera.setenv(ss.CENTRE_ENV, "engaged")
    assert ss.centre_mask(cls) is None and ss.centre_mask(None) is None


def _pool(seed=0):
    rng = np.random.default_rng(seed)
    GE = np.repeat(np.arange(4), 120)
    YE = np.tile(np.arange(6), 80)
    XE = rng.normal(size=(len(GE), 5)) + 0.8 * YE[:, None] + 0.5 * GE[:, None]
    return XE, YE, GE


def _pipe():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))


def test_frozen_frame_centre_uses_only_the_working_rows(camera):
    XE, YE, GE = _pool()
    cls = np.where(np.arange(len(GE)) % 3 == 0, "unknown", "contact").astype(object)
    ff = ss.FrozenFrame(XE, YE, GE, [0, 1], _pipe, loso=False, cls_e=cls)
    XE2 = XE.copy()
    XE2[cls == "unknown"] += 50.0                 # junk on the NON-working rows must not move the centre
    ff2 = ss.FrozenFrame(XE2, YE, GE, [0, 1], _pipe, loso=False, cls_e=cls,
                         store=lambda fit_all: {"full": ff.full, "loso": {}}, labels=list("abcd"))
    np.testing.assert_allclose(ff.centre(3)[0], ff2.centre(3)[0])
    with pytest.raises(ss.BalEmCentreRefused):
        ss.FrozenFrame(XE, YE, GE, [0, 1], _pipe, loso=False)         # camera gate, no row classes: fails closed
    camera.setenv(ss.CENTRE_ENV, "engaged")
    e = ss.FrozenFrame(XE2, YE, GE, [0, 1], _pipe, loso=False, store=lambda fit_all: {"full": ff.full, "loso": {}}, labels=list("abcd"))
    assert e.cm is None
    assert not np.allclose(e.centre(3)[0], ff2.centre(3)[0])        # the engaged centre DOES see the junk rows


def test_contact_row_fallback_is_explicit_and_the_minimum_is_20(camera, capsys):
    """Fifth review D-B3: a session below MIN_CENTRE_ROWS working rows is REFUSED unless it is a configured contact
    fallback (PS93_0813, no video), whose center comes from its DAQ contact rows and is logged by name."""
    XE, YE, GE = _pool()
    me = GE == 2
    cm = np.ones(len(GE), bool)
    cm[me] = False
    np.testing.assert_array_equal(ss._centre_rows_of(XE, me, cm, "PS93_0813"), XE[me])
    assert "PS93_0813" in capsys.readouterr().out and "PS93_0813" in ss.FALLBACK_USED
    with pytest.raises(ss.BalEmCentreRefused, match="PS92_0818"):
        ss._centre_rows_of(XE, me, cm, "PS92_0818")
    cm[np.flatnonzero(me)[:19]] = True                     # 19 working rows: still refused
    with pytest.raises(ss.BalEmCentreRefused):
        ss._centre_rows_of(XE, me, cm, "PS92_0818")
    cm[np.flatnonzero(me)[19]] = True                      # 20: the working rows
    assert len(ss._centre_rows_of(XE, me, cm, "PS92_0818")) == 20
    assert ss.scaling_params()["centre_set_fallback"] == {"PS93_0813": "contact"}
    camera.setenv(ss.CENTRE_ENV, "engaged")
    assert "centre_set_fallback" not in ss.scaling_params()


def test_a_missing_classes_error_stops_the_confusions_block(camera):
    """Fifth review D-B2: the refusal is a SystemExit, so the per-class confusions' generic `except Exception` (and any
    other) cannot swallow it -- the run stops."""
    from wfield_local import position_coding_directions as pcd
    XE, YE, GE = _pool()
    XU, YU, GU = XE[:30], YE[:30], GE[:30]
    e_pre = GE < 2
    with pytest.raises(ss.BalEmCentreRefused):
        try:
            pcd._class_confusions(XE, YE, GE, XU, YU, e_pre, {0, 1}, lambda c, pos=None: np.zeros(30, bool),
                                  list("abcd"), GU=GU)
        except Exception:                                                  # noqa: BLE001 - the block under test
            pass


def test_centre_set_enters_the_spec(camera):
    kw = dict(align="cue", source="roi", train_labels=["PS92_0812"], post_s=2.0, train_data=None)
    w = fm.make_spec("PS92", "decoder", **kw, **ss.spec_kw())
    camera.setenv(ss.CENTRE_ENV, "engaged")
    e = fm.make_spec("PS92", "decoder", **kw, **ss.spec_kw())
    assert w["scaling_params"]["centre_set"] == "working" and e["scaling_params"]["centre_set"] == "engaged"
    assert fm.spec_id(w) != fm.spec_id(e)


def test_engaged_centre_under_camera_is_tagged(camera):
    from wfield_local import run_mode
    assert run_mode.emcentre_part() == ""
    camera.setenv(ss.CENTRE_ENV, "engaged")
    assert run_mode.emcentre_part() == "emc-engaged" and "emc-engaged" in run_mode.cache_parts()
    camera.setenv("WIDEFIELD_TRIAL_GATE", "daq")
    camera.delenv(ss.CENTRE_ENV)
    assert run_mode.emcentre_part() == ""


def test_scored_frame_refuses_without_centre_rows(camera):
    XE, YE, GE = _pool()
    ff = ss.FrozenFrame(XE, YE, GE, [0, 1], _pipe, loso=False, cls_e=np.array(["contact"] * len(GE), object))
    with pytest.raises(ss.BalEmCentreRefused, match="WORKING center rows"):
        ss.scored_frame(XE, GE, XE[:0], np.zeros(0, int), [2], lambda i: ff.full, lambda i: ff.pi["full"])


# ---------------------------------------------------------------------------------------------- 8.17 drift month model
def _drift_bundle():
    XE, YE, GE = _pool(3)
    kept = ["PS92_0606", "PS92_0607", "PS92_0812", "PS92_0813"]
    return {"XE": XE, "YE": YE, "GE": GE, "kept": kept, "pre_i": {0, 1, 2, 3}, "e_pre": np.ones(len(GE), bool),
            "basis_id": "b"}


def test_drift_month_model_is_stored_in_the_bal_em_frame(monkeypatch):
    jad = pytest.importorskip("scripts.june_august_drift")
    from wfield_local.locanmf_frozen_decoder import _pipe as fpipe
    monkeypatch.delenv(ss.SPOUT_SCALING_ENV, raising=False)
    monkeypatch.delenv("WIDEFIELD_TRIAL_GATE", raising=False)
    seen = []

    def fake(spec, fit, *, meta=None, refreeze=None, log=print):
        seen.append(spec)
        return fit(), fm.STATUS_NEW

    monkeypatch.setattr(fm, "load_or_fit", fake)
    monkeypatch.setattr(fm, "family_on", lambda env: True)
    bd = _drift_bundle()
    jad._DRIFT_MODELS.clear()
    m = jad._stored_month_model("PS92", bd, [2, 3], scaling=ss.SCALING_BAL_EM)
    assert seen[-1]["scaling"] == ss.SCALING_BAL_EM and seen[-1]["kind"] == "drift_decoder"
    XT = ss.train_frame(bd["XE"], bd["YE"], bd["GE"], [2, 3])
    tr = np.isin(bd["GE"], [2, 3])
    ref = fpipe().fit(XT[tr], bd["YE"][tr])
    np.testing.assert_allclose(m.predict_proba(XT[:20]), ref.predict_proba(XT[:20]))
    # session_z: the production-frame fit, a different spec (no scaling key)
    m0 = jad._stored_month_model("PS92", bd, [2, 3], scaling=None)
    assert "scaling" not in seen[-1] and m0 is not m
    # through the FrozenFrame hook the frame's full model IS the stored one: no in-memory fit under bal_em
    ff = ss.FrozenFrame(bd["XE"], bd["YE"], bd["GE"], [2, 3], fpipe, train_mask=bd["e_pre"], loso=False,
                        store=lambda fit_all: {"full": m, "loso": {}}, labels=dict(enumerate(bd["kept"])))
    assert ff.full is m
