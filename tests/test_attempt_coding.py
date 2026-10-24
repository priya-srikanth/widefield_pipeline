"""attempt_coding: video attempt classes, own/off/spec scoring, within-session permutation contrast."""
import numpy as np

from wfield_local import attempt_coding as ac


def test_classify_video_classes():
    # engaged (XE) rows: pre hit, pre incomplete (not a reference), post hit, post incomplete
    ce, cu = ac.classify(np.array(["hit", "incomplete", "hit", "incomplete"]),
                         np.array(["no_try", "no_try", "jaw_only", "no_try", "hit", "unknown", "no_try"]),
                         np.array([True, True, False, False]),
                         np.array([True, False, False, False, False, False, False]),
                         np.array([False, False, False, True, False, True, False]))
    assert list(ce) == ["pre_hit", "", "hit", "incomplete"]
    # pre no-lick -> unused; post working no_try -> noattempt; jaw_only -> reaction; after the quit -> stopped;
    # attempted after max_rt -> late; unknown after the quit is not a class
    assert list(cu) == ["", "noattempt", "reaction", "stopped", "late", "", "noattempt"]


def test_own_axis_is_pole_normalised():
    rng = np.random.default_rng(0)
    d = 40
    mu = {P: rng.normal(size=d) * 2 for P in ac.SPOUTS}
    X = np.vstack([mu[P] + rng.normal(size=(60, d)) * 0.3 for P in ac.SPOUTS])
    sp = np.repeat(ac.SPOUTS, 60)
    W = ac.fit_axes(X, sp, "dom")
    own = ac.score(X, sp, W)
    assert abs(np.nanmean(own) - 1.0) < 1e-6           # pre-stroke hits at their own spout = the unit pole
    # a far_contra trial that looks like far_ipsi scores near the zero pole on the far_contra axis
    o = ac.score(mu["far_ipsi"][None], np.array(["far_contra"]), W)
    assert o[0] < 0.3
    assert np.isnan(ac.score(X[:1], np.array(["nowhere"]), W)[0])


def test_perm_test_within_session():
    rng = np.random.default_rng(1)
    v = np.concatenate([rng.normal(1, 1, 30), rng.normal(0, 1, 30), rng.normal(5, 1, 30), rng.normal(4, 1, 30)])
    cls = np.array((["a"] * 30 + ["b"] * 30) * 2)
    sess = np.repeat(["s1", "s2"], 60)
    diff, p, ns, na, nb = ac.perm_test(v, cls, sess, "a", "b", n_perm=500)
    assert ns == 2 and na == 60 and nb == 60
    assert 0.6 < diff < 1.4 and p < 0.01
    # a session missing one class does not enter
    diff2, _p, ns2, _a, _b = ac.perm_test(v[:90], cls[:90], sess[:90], "a", "b", n_perm=100)
    assert ns2 == 1


def test_map_classes_for_rows():
    from wfield_local import attempt_maps as am
    ce, cu = am.classes_for_rows(np.array(["hit", "incomplete"]), np.array(["no_try", "jaw_only", "no_try", "unknown"]),
                                 np.array([False, False, True, True]), pre=False)
    assert list(ce) == ["hit", "incomplete"] and list(cu) == ["noattempt", "reaction", "stopped", ""]
    ce, cu = am.classes_for_rows(np.array(["hit", "incomplete"]), np.array(["no_try"]), np.array([False]), pre=True)
    assert list(ce) == ["pre_hit", "pre_incomplete"] and list(cu) == [""]
    assert am._down(np.ones((540, 640))).shape == (270, 320)
