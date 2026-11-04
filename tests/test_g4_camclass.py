"""Follow-ons of the 11-04 B/C/D/G port (coordinator 11-04): (a) the CAMERA-CLASS G4 discriminator, (c) the lick-only
arm of `decode_matched` on camera working rows.

Pins:
  a  `cam_balanced_sample` = g4common.cam_balanced_sample (WORKING vs NONE, position-balanced, positives first, the same
     rng use); `g4_frame` z-scores each session on its CONTACT rows by default (working / engaged selectable, a centre
     with incomplete rows only under "working"); `g4_camclass` scores jaw_only / after_end as their own groups, flags
     thin cells (< 10) and reads each post group against the SAME pre group or, when that has no cell, pre WORKING
     (labelled); the stored discriminator's spec carries the class set and the centre.
  c  `_lickonly_working` trains on working rows only, scores the post session's working rows, and its stored spec is
     not the engaged-row arm's.
"""
import numpy as np
import pytest

from wfield_local import poststroke_compare as pc

GROUPS = ("contact", "incomplete", "jaw_only", "none", "after_end")


def _pool(seed=0, n_pre=4, n_post=1, n=240, n_u=160):
    rng = np.random.default_rng(seed)
    XE, YE, GE, XU, YU, GU, ge, gu = [], [], [], [], [], [], [], []
    for i in range(n_pre + n_post):
        y = rng.integers(0, 6, n)
        XE.append(rng.normal(size=(n, 6)) + np.eye(6)[y] + 0.5)
        YE.append(y)
        GE.append(np.full(n, i))
        ge.append(rng.choice(["contact", "incomplete"], n, p=[.9, .1]))
        yu = rng.integers(0, 6, n_u)
        g = rng.choice(["incomplete", "jaw_only", "none", "after_end"], n_u, p=[.3, .2, .3, .2])
        XU.append(rng.normal(size=(n_u, 6)) + np.eye(6)[yu] * .3 - 0.5 * (g == "none")[:, None])
        YU.append(yu)
        GU.append(np.full(n_u, i))
        gu.append(g)
    cat = np.concatenate
    d = {"XE": np.vstack(XE), "YE": cat(YE), "GE": cat(GE), "XU": np.vstack(XU), "YU": cat(YU), "GU": cat(GU),
         "kept": [f"PSX_08{10 + i}" for i in range(n_pre + n_post)], "pre_i": set(range(n_pre)),
         "post_i": set(range(n_pre, n_pre + n_post)), "align": "cue", "source": "roi", "_frozen_cache": {}}
    gE, gU = cat(ge).astype(object), cat(gu).astype(object)
    cg = {"grp_e": gE, "grp_u": gU, "jj_e": np.ones(len(gE), bool), "jj_u": np.ones(len(gU), bool),
          "have": set(range(n_pre + n_post)), "work_e": np.isin(gE, ["contact", "incomplete"]),
          "work_u": np.isin(gU, ["contact", "incomplete"])}
    return d, cg


@pytest.fixture
def g4env(monkeypatch):
    d, cg = _pool()
    monkeypatch.setattr(pc, "_camera_groups", lambda dd: cg)
    specs = []

    def _store(dd, kind, extra, fit, meta=None, scaling=None):
        specs.append((kind, extra, scaling))
        return fit()

    monkeypatch.setattr(pc, "_g_store", _store)
    monkeypatch.delenv(pc.G4_CENTRE_ENV, raising=False)
    monkeypatch.delenv(pc.G4_DISC_ENV, raising=False)
    return d, cg, specs


def test_g4_defaults_and_tag(monkeypatch):
    monkeypatch.delenv(pc.G4_CENTRE_ENV, raising=False)
    monkeypatch.delenv(pc.G4_DISC_ENV, raising=False)
    assert pc.g4_disc() == "camera" and pc.g4_centre() == "contact"
    assert pc.g4_frame_tag() == "camclass/z-contact/pos-contact+incomplete/neg-none"
    monkeypatch.setenv(pc.G4_CENTRE_ENV, "bogus")
    with pytest.raises(ValueError):
        pc.g4_centre()


def test_cam_balanced_sample_matches_the_check_script(g4env):
    d, cg, _ = g4env
    dT, _cg = pc.g4_frame(d)
    keep = [0, 1, 2, 3, 4, 5]
    X, y, g = pc.cam_balanced_sample(dT, cg["grp_e"], cg["grp_u"], keep, np.random.RandomState(0))
    # the g4common algorithm, inline
    rng = np.random.RandomState(0)
    XA = np.vstack([dT["XE"], dT["XU"]])
    Y = np.concatenate([dT["YE"], dT["YU"]])
    G = np.concatenate([dT["GE"], dT["GU"]])
    gg = np.concatenate([cg["grp_e"], cg["grp_u"]])
    m = np.isin(G, list(dT["pre_i"])) & np.isin(Y, keep)
    pos, neg = m & np.isin(gg, ["contact", "incomplete"]), m & (gg == "none")
    xs = []
    for c in keep:
        ie, iu = np.flatnonzero(pos & (Y == c)), np.flatnonzero(neg & (Y == c))
        n = min(len(ie), len(iu))
        if n < 5:
            continue
        xs += [XA[rng.choice(ie, n, replace=False)], XA[rng.choice(iu, n, replace=False)]]
    np.testing.assert_array_equal(X, np.vstack(xs))
    assert set(np.unique(gg[np.isin(np.arange(len(gg)), [])])) == set()
    assert y.sum() * 2 == len(y)


def test_g4_frame_centres_on_contact_rows(g4env, monkeypatch):
    d, cg, _ = g4env
    dT, _ = pc.g4_frame(d)
    i = 0
    rows = np.vstack([dT["XE"][(d["GE"] == i) & (cg["grp_e"] == "contact")],
                      dT["XU"][(d["GU"] == i) & (cg["grp_u"] == "contact")]])
    np.testing.assert_allclose(rows.mean(0), 0, atol=1e-9)
    monkeypatch.setenv(pc.G4_CENTRE_ENV, "engaged")
    dT2, _ = pc.g4_frame(d)
    np.testing.assert_array_equal(dT2["XE"], d["XE"])         # the production z frame: identity


def test_g4_camclass_scores_each_group_and_flags_thin_cells(g4env):
    d, cg, specs = g4env
    keep = [0, 1, 2, 3, 4, 5]
    r = pc.g4_camclass(d, keep)
    assert r["frame"].startswith("camclass/z-contact") and r["min_n"] == 10
    assert set(r["post"]) == set(GROUPS) | {"stopped", "stopped_licking"}     # rebuild extras (empty here)
    # working vs none separates: post none rows less working-like than post contact rows
    assert r["post"]["none"]["frac_spout_balanced"] < r["post"]["contact"]["frac_spout_balanced"]
    # every group read against the SAME group pre-stroke (all exist pre in this pool)
    assert all(r["post"][g]["pre_reference_is"] == g for g in GROUPS)
    # thin cells flagged, left out of the balanced mean
    for g in GROUPS:
        for v in r["post"][g]["per_spout"].values():
            assert v["thin"] == (v["n"] < 10)
    kind, extra, _sc = specs[-1]
    assert kind == "g_cam_disc" and "pos-contact+incomplete" in extra and "neg-none" in extra and "z-contact" in extra


def test_g4_falls_back_to_pre_working_when_a_group_is_absent_pre(g4env):
    d, cg, _ = g4env
    pre_rows = np.isin(d["GU"], list(d["pre_i"])) & (cg["grp_u"] == "after_end")
    cg["grp_u"][pre_rows] = "jaw_only"                         # no pre after_end rows at all
    r = pc.g4_camclass(d, [0, 1, 2, 3, 4, 5])
    assert r["post"]["after_end"]["pre_reference_is"] == "working"
    assert r["post"]["after_end"]["pre_reference"] == r["pre_working_reference"]["frac_spout_balanced"]


def test_lickonly_arm_on_working_rows(g4env, monkeypatch):
    d, cg, specs = g4env
    monkeypatch.setenv("WIDEFIELD_SPOUT_SCALING", "session_z")
    pos = [0, 1, 2, 4]
    clf, pre_pred, (pre_y, pre_g), (Xp, yp) = pc._lickonly_working(d, pos, None)
    nwork_post = int((cg["work_e"] & (d["GE"] == 4) & np.isin(d["YE"], pos)).sum()
                     + (cg["work_u"] & (d["GU"] == 4) & np.isin(d["YU"], pos)).sum())
    assert len(yp) == nwork_post and set(np.unique(yp)) <= set(pos)
    assert len(pre_y) == len(pre_pred) and set(np.unique(pre_g)) == {0, 1, 2, 3}
    kind, extra, _ = specs[-1]
    assert kind == "g_lickonly_decoder" and "-ffwork-" in extra
