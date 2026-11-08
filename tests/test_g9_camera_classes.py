"""Follow-on (b), Priya 11-04 (approved with guards): G9 coding-direction classes by CAMERA class.

Pins the guards: per spout only; thin cells (< 10) kept and flagged; each post class read against the SAME pre class,
else pre working (labeled); rows z-scored per session on CONTACT rows; the DAQ classes stay the default reading of every
consumer (`g9_daq_classes`), and `g9_old_new` reports old next to new.
"""
import numpy as np

from wfield_local import position_coding_directions as pcd


def _setup(seed=0):
    rng = np.random.default_rng(seed)
    P = pcd.BY_SEVERITY
    nE, nU = 600, 400
    GE, GU = rng.integers(0, 4, nE), rng.integers(0, 4, nU)
    en, un = rng.choice(P, nE), rng.choice(P, nU)
    XE, XU = rng.normal(size=(nE, 5)), rng.normal(size=(nU, 5))
    ge = rng.choice(["contact", "incomplete"], nE, p=[.9, .1]).astype(object)
    gu = rng.choice(["incomplete", "jaw_only", "none", "after_end"], nU).astype(object)
    gu[(GU < 2) & (gu == "after_end")] = "none"            # no pre after_end -> falls back to pre working
    W = {p: (rng.normal(size=5), -1.0, 1.0) for p in P}

    def proj(cls, X, D, sess=None):
        w, a, b = W[D]
        return pcd.project(X, w, a, b) if len(X) else np.empty(0)

    groups = (ge, gu, np.ones(nE, bool), np.ones(nU, bool), {0, 1, 2, 3})
    return proj, W, XE, XU, GE, GU, en, un, {0, 1}, ["A_0801", "A_0802", "A_0901", "A_0902"], groups


def test_camera_class_block_guards():
    proj, W, XE, XU, GE, GU, en, un, pre_i, kept, groups = _setup()
    r = pcd._camera_class_block(proj, W, XE, XU, GE, GU, en, un, pre_i, kept, groups)
    assert r["min_n"] == 10 and r["frame"] == "z-contact"
    for phase in ("pre", "post"):
        for _grp, per in r["pooled"][phase].items():
            assert set(per) == set(pcd.BY_SEVERITY)                 # per spout, never pooled over spouts
            for c in per.values():
                assert c["thin"] == (c["n"] < 10)
    # after_end has no pre rows: read against pre WORKING, labelled
    assert set(r["pre_reference_is"]["after_end"].values()) == {"working"}
    assert "same" in set(r["pre_reference_is"]["contact"].values())
    # a thin post cell has no contrast
    for grp in r["contrast"]:
        for P, v in r["contrast"][grp].items():
            if r["pooled"]["post"][grp][P]["thin"]:
                assert v is None


def test_contact_frame_uses_contact_rows_only():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(100, 3)) + 5
    G = np.zeros(100, int)
    grp = np.array(["contact"] * 50 + ["incomplete"] * 50, dtype=object)
    X[50:] += 10                                              # incomplete rows far away: must not move the centre
    Z = pcd._g9_contact_frame(X, G, grp, [0])
    np.testing.assert_allclose(Z[:50].mean(0), 0, atol=1e-9)


def test_g9_switch_and_old_new_report(monkeypatch):
    monkeypatch.delenv(pcd.G9_CLASSES_ENV, raising=False)
    assert pcd.g9_classes() == "camera"
    monkeypatch.setenv(pcd.G9_CLASSES_ENV, "daq")
    assert pcd.g9_classes() == "daq"
    proj, W, XE, XU, GE, GU, en, un, pre_i, kept, groups = _setup()
    cam = pcd._camera_class_block(proj, W, XE, XU, GE, GU, en, un, pre_i, kept, groups)
    rec = {"PSX": {"methods": {"dom": {"pooled": {"poststroke_lick": {"far_R": {"n": 30, "mean": .4, "low_n": False}}},
                                       "camera_classes": cam}}}}
    rows = pcd.g9_old_new(rec)
    assert {r["scheme"] for r in rows} == {"daq", "camera"}
    assert any(r["class"] == "post_incomplete" for r in rows)
