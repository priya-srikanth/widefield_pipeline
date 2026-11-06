"""Priya 11-05, DECISIONS_PENDING_MERGE_2026-11-04 (f) 1: the state discriminators train WORKING (contact + incomplete)
vs NO_RESPONSE; STOPPED trials are scored as their own class and never trained on; jaw_only / after_end stay
scored-only.

Pins:
  * `precue_engagement_states.camera_train_sets`: no stopped / stopped_licking / jaw_only / after_end / unknown row is
    ever in the training set; contact + incomplete come from BOTH DAQ arms; the negatives are no_response only.
  * `disc_train_rows` with no no-lick-arm positives is exactly the pre-11-05 DAQ stack (so the DAQ model's id stands).
  * the camera-class model is stored under a NEW id (class tag in the align), also when the training arrays are equal.
  * section G's camera-class discriminator (`poststroke_compare.cam_balanced_sample`, 11-04) never draws a stopped /
    stopped_licking / after_end / jaw_only row.
"""
import inspect
from types import SimpleNamespace

import numpy as np

from wfield_local import frozen_models as fm
from wfield_local import poststroke_compare as pc
from wfield_local import precue_engagement_states as pes

ALL = ("contact", "incomplete", "jaw_only", "no_response", "unknown", "stopped", "stopped_licking", "after_end")


def test_camera_train_sets_never_train_on_stopped():
    ce = np.array(ALL, dtype=object)
    cu = np.array(ALL, dtype=object)
    on = np.ones(len(ALL), bool)
    ke, kpu, kd = pes.camera_train_sets(ce, cu, on, on, on)
    assert list(ce[ke]) == ["contact", "incomplete"]
    assert list(cu[kpu]) == ["contact", "incomplete"]          # the no-lick arm holds the incomplete licks
    assert list(cu[kd]) == ["no_response"]
    for c in pes.SCORED_ONLY:
        assert not ke[ce == c].any() and not kpu[cu == c].any() and not kd[cu == c].any()
    assert "stopped" in pes.SCORED_ONLY and "jaw_only" in pes.SCORED_ONLY and "after_end" in pes.SCORED_ONLY


def test_camera_train_sets_respect_the_windows():
    cu = np.array(["no_response", "no_response", "incomplete", "incomplete"], dtype=object)
    ce = np.array(["contact", "contact"], dtype=object)
    ke, kpu, kd = pes.camera_train_sets(ce, cu, np.array([True, False]), np.array([True, True, True, False]),
                                        np.array([True, False, True, True]))
    assert ke.tolist() == [True, False]
    assert kpu.tolist() == [False, False, True, False]
    assert kd.tolist() == [True, False, False, False]           # dis window (late, minus SPATIAL_MISS) still applies


def test_disc_train_rows_daq_stack_unchanged():
    rng = np.random.default_rng(0)
    XE, XU = rng.normal(size=(30, 4)), rng.normal(size=(20, 4))
    GE, GU = np.repeat([0, 1, 2], 10), np.repeat([0, 1], 10)
    ke, kd = rng.random(30) > .4, rng.random(20) > .5
    X, y, g = pes.disc_train_rows(XE, XU, GE, GU, ke, np.zeros(20, bool), kd)
    assert np.array_equal(X, np.vstack([XE[ke], XU[kd]]))
    assert np.array_equal(y, np.concatenate([np.zeros(int(ke.sum())), np.ones(int(kd.sum()))]))
    assert np.array_equal(g, np.concatenate([GE[ke], GU[kd]]))
    # the leave-one-session-out stack, as the old inline code built it
    X1, y1, _ = pes.disc_train_rows(XE, XU, GE, GU, ke, np.zeros(20, bool), kd, exclude=1)
    tr_e, tr_d = ke & (GE != 1), kd & (GU != 1)
    assert np.array_equal(X1, np.vstack([XE[tr_e], XU[tr_d]]))
    assert np.array_equal(y1, np.concatenate([np.zeros(int(tr_e.sum())), np.ones(int(tr_d.sum()))]))


def test_disc_train_rows_camera_positives_from_both_arms():
    XE, XU = np.zeros((3, 2)), np.ones((4, 2))
    GE, GU = np.zeros(3, int), np.zeros(4, int)
    X, y, g = pes.disc_train_rows(XE, XU, GE, GU, np.array([1, 1, 0], bool), np.array([1, 0, 0, 0], bool),
                                  np.array([0, 0, 1, 1], bool))
    assert len(X) == 5 and y.tolist() == [0, 0, 0, 1, 1]
    assert X[2].tolist() == [1.0, 1.0]                            # the no-lick-arm working row is on the engaged side


def test_camera_disc_is_a_new_frozen_id(monkeypatch):
    seen = []
    monkeypatch.setattr(fm, "load_or_fit", lambda spec, fit, **k: (seen.append(fm.spec_id(spec)) or ({}, "hit")))
    b = SimpleNamespace(basis_id="76d884873920")
    data = (np.arange(12.).reshape(4, 3), np.arange(4))
    for al in ("precue-disc", f"precue-disc-{pes.CAMERA_DISC_TAG}"):
        pes._stored("PS92", al, b, ["PS92_0810"], {0}, 3, data, lambda: {}, 2.0)
    assert len(seen) == 2 and seen[0] != seen[1]                  # same rows, different classes -> different id
    # run_animal stores the camera model under the tagged align (and the DAQ model under the old one)
    src = inspect.getsource(pes.run_animal)
    assert 'f"{align}-disc" + (f"-{CAMERA_DISC_TAG}" if camera else "")' in src
    assert "camera_train_sets(" in src and "class_scores(" in src


def test_class_scores_reports_every_class_separately():
    class M:
        def predict_proba(self, X):
            p = X[:, 0]
            return np.c_[1 - p, p]
    ce = np.array(["contact", "stopped"], dtype=object)
    cu = np.array(["no_response", "stopped", "jaw_only"], dtype=object)
    XE, XU = np.array([[0.1], [0.9]]), np.array([[0.8], [0.7], [0.5]])
    t = np.ones(2, bool), np.ones(3, bool)
    out = pes.class_scores(M(), XE, XU, ce, cu, *t, ~t[0], ~t[1])
    assert out["pre"]["stopped"]["n"] == 2 and abs(out["pre"]["stopped"]["p_disengaged"] - 0.8) < 1e-12
    assert out["pre"]["jaw_only"]["p_disengaged"] == 0.5 and out["post"]["contact"]["n"] == 0


def test_section_g_camera_discriminator_never_trains_on_stopped():
    rng = np.random.default_rng(1)
    n = 400
    groups = np.array(["contact", "incomplete", "jaw_only", "none", "after_end", "stopped", "stopped_licking"],
                      dtype=object)
    gE, gU = rng.choice(groups, n), rng.choice(groups, n)
    dT = {"XE": np.c_[np.arange(n), np.zeros(n)], "XU": np.c_[np.arange(n) + 10_000, np.zeros(n)],
          "YE": rng.integers(0, 2, n), "YU": rng.integers(0, 2, n), "GE": rng.integers(0, 3, n),
          "GU": rng.integers(0, 3, n), "pre_i": {0, 1, 2}}
    X, y, _g = pc.cam_balanced_sample(dT, gE, gU, [0, 1], np.random.RandomState(0))
    ids = X[:, 0].astype(int)
    drawn = np.where(ids >= 10_000, gU[np.clip(ids - 10_000, 0, n - 1)], gE[np.clip(ids, 0, n - 1)])
    assert set(drawn[y == 1]) <= {"contact", "incomplete"}
    assert set(drawn[y == 0]) == {"none"}
