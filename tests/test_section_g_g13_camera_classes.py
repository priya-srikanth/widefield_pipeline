"""Section G's G1-G3 state discriminators (`looks_like_which`, `fits_engaged_distribution`, `undetected_state_split`)
follow decision (f) 1 under the camera grouping (Priya 11-05; settings audit 11-08), as decided by Priya 11-08: train
WORKING (contact + incomplete, either DAQ arm) vs the RAW camera class NO_RESPONSE only (section G's "none" group split
by the raw class; "unknown" is never trained on, nor are stopped / stopped_licking / jaw_only / after_end rows); the
scored "no-lick" set is the rows whose raw class is no_response, the engaged comparison the camera WORKING rows; every
camera group (and no_response / unknown apart) is scored on its own; the stored ids carry the class set
("cls-work-v-noresp"). The DAQ grouping keeps the old training set, ids and scored sets; G4 keeps its "none" negative.
"""
import numpy as np
import pytest

from wfield_local import poststroke_compare as pc

KEEP = [0, 1, 2, 3, 4, 5]
NEVER = ("jaw_only", "after_end", "stopped", "stopped_licking")
SENTINEL = 1e3
#: raw class of the rows of each non-"none" group (stopped / after_end rows are raw no_response: scored, never trained)
RAW_OF = {"contact": "contact", "incomplete": "incomplete", "jaw_only": "jaw_only", "after_end": "no_response",
          "stopped": "no_response", "stopped_licking": "incomplete"}


def _pool(seed=0, n_pre=4, n_post=1, n=240, n_u=200):
    rng = np.random.default_rng(seed)
    XE, YE, GE, XU, YU, GU, ge, gu, re_, ru_ = [], [], [], [], [], [], [], [], [], []
    for i in range(n_pre + n_post):
        y = rng.integers(0, 6, n)
        XE.append(rng.normal(size=(n, 6)) + np.eye(6)[y] + 0.5)
        YE.append(y)
        GE.append(np.full(n, i))
        g = rng.choice(["contact", "incomplete", "stopped_licking", "none"], n, p=[.8, .1, .05, .05])
        ge.append(g)
        re_.append(np.array([RAW_OF.get(c) or rng.choice(["no_response", "unknown"]) for c in g], dtype=object))
        yu = rng.integers(0, 6, n_u)
        g = rng.choice(["incomplete", "jaw_only", "none", "after_end", "stopped"], n_u, p=[.25, .15, .3, .15, .15])
        XU.append(rng.normal(size=(n_u, 6)) + np.eye(6)[yu] * .3 - 0.5 * (g == "none")[:, None])
        YU.append(yu)
        GU.append(np.full(n_u, i))
        gu.append(g)
        ru_.append(np.array([RAW_OF.get(c) or rng.choice(["no_response", "unknown"], p=[.7, .3]) for c in g],
                            dtype=object))
    cat = np.concatenate
    gE, gU = cat(ge).astype(object), cat(gu).astype(object)
    rE, rU = cat(re_).astype(object), cat(ru_).astype(object)
    d = {"XE": np.vstack(XE), "YE": cat(YE), "GE": cat(GE), "XU": np.vstack(XU), "YU": cat(YU), "GU": cat(GU),
         "kept": [f"PSX_08{10 + i}" for i in range(n_pre + n_post)], "pre_i": set(range(n_pre)),
         "post_i": set(range(n_pre, n_pre + n_post)), "align": "cue", "source": "roi", "_frozen_cache": {}}
    # rows of the never-trained groups, and the "unknown" rows of "none", carry a sentinel in the last column: none may
    # reach a training sample
    d["XE"][np.isin(gE, NEVER) | (rE == "unknown"), -1] = SENTINEL
    d["XU"][np.isin(gU, NEVER) | (rU == "unknown"), -1] = SENTINEL
    cg = {"grp_e": gE, "grp_u": gU, "jj_e": np.ones(len(gE), bool), "jj_u": np.ones(len(gU), bool),
          "have": set(range(n_pre + n_post)), "work_e": np.isin(gE, ["contact", "incomplete"]),
          "work_u": np.isin(gU, ["contact", "incomplete"]), "raw_e": rE, "raw_u": rU}
    return d, cg


@pytest.fixture
def env(monkeypatch):
    d, cg = _pool()
    state = {"cg": cg}
    monkeypatch.setattr(pc, "_camera_groups", lambda dd: state["cg"])
    specs, samples = [], []

    def _store(dd, kind, extra, fit, meta=None, scaling=None):
        specs.append((kind, extra))
        return fit()

    real_sample = pc._g13_sample

    def _spy(*a, **k):
        got = real_sample(*a, **k)
        if got is not None:
            samples.append(got)
        return got

    monkeypatch.setattr(pc, "_g_store", _store)
    monkeypatch.setattr(pc, "_g13_sample", _spy)
    monkeypatch.delenv("WIDEFIELD_STATE_DISC_SCALING", raising=False)
    return d, state, specs, samples


def _n_post(d, cg, which):
    """Post rows at the KEEP spouts in a scored set, counted independently of `_g13_rows`."""
    pe = np.isin(d["GE"], list(d["post_i"])) & np.isin(d["YE"], KEEP)
    pu = np.isin(d["GU"], list(d["post_i"])) & np.isin(d["YU"], KEEP)
    if cg is None:
        return int((pe if which == "engaged" else pu).sum())
    if which == "engaged":
        return int((pe & np.isin(cg["grp_e"], ["contact", "incomplete"])).sum()
                   + (pu & np.isin(cg["grp_u"], ["contact", "incomplete"])).sum())
    return int((pe & (cg["grp_e"] == "none") & (cg["raw_e"] == "no_response")).sum()
               + (pu & (cg["grp_u"] == "none") & (cg["raw_u"] == "no_response")).sum())


def test_tag_and_negative_class():
    assert pc.G13_TAG == "cls-work-v-noresp"
    assert pc.G13_NEG == ("no_response",)
    # G4 is a separate, open question: its negative stays section G's whole "none" group
    assert pc.G4_NEG == ("none",) and pc.G4_POS == ("contact", "incomplete")


def test_sample_is_working_vs_no_response_never_unknown_or_stopped(env):
    d, state, _, _ = env
    cg = state["cg"]
    X, y, _g = pc._g13_sample(d, cg, KEEP, np.random.RandomState(0))
    assert not np.any(X[:, -1] == SENTINEL)                 # no unknown / stopped / jaw_only / after_end row
    ge, gu = pc._g13_split(cg)
    assert set(ge[cg["grp_e"] == "none"]) | set(gu[cg["grp_u"] == "none"]) == {"no_response", "unknown"}
    want = pc.cam_balanced_sample(d, ge, gu, KEEP, np.random.RandomState(0), neg=("no_response",))
    for a, b in zip((X, y), want[:2]):
        np.testing.assert_array_equal(a, b)
    # the negatives are exactly pre "none"-group rows of raw class no_response
    Xall = np.vstack([d["XE"], d["XU"]])
    neg_ok = np.concatenate([(cg["grp_e"] == "none") & (cg["raw_e"] == "no_response"),
                             (cg["grp_u"] == "none") & (cg["raw_u"] == "no_response")])
    pool = {r.tobytes() for r in Xall[neg_ok]}
    assert all(r.tobytes() in pool for r in X[y == 0])
    # DAQ grouping: the old sampler, draw for draw
    got = pc._g13_sample(d, None, KEEP, np.random.RandomState(0))
    old = pc.balanced_lick_sample(d, KEEP, np.random.RandomState(0))
    for a, b in zip(got, old):
        np.testing.assert_array_equal(a, b)


def test_g4_sampler_unchanged_keeps_none_with_unknown(env):
    d, state, _, _ = env
    cg = state["cg"]
    a = pc.cam_balanced_sample(d, cg["grp_e"], cg["grp_u"], KEEP, np.random.RandomState(0))
    b = pc.cam_balanced_sample(d, cg["grp_e"], cg["grp_u"], KEEP, np.random.RandomState(0), neg=pc.G4_NEG)
    for u, v in zip(a, b):
        np.testing.assert_array_equal(u, v)
    assert np.any(a[0][a[1] == 0][:, -1] == SENTINEL)         # G4's "none" still holds the unknown rows


@pytest.mark.parametrize("fn,kw,kind,extra", [
    ("looks_like_which", {}, "g_lick_disc", "keep0-1-2-3-4-5-seed0"),
    ("fits_engaged_distribution", {"n_boot": 200}, "g_lick_disc_folds", "keep0-1-2-3-4-5-seed0-v2")])
def test_camera_grouping_trains_on_camera_classes(env, fn, kw, kind, extra):
    d, state, specs, samples = env
    r = getattr(pc, fn)(d, KEEP, **kw)
    assert samples and all(not np.any(s[0][:, -1] == SENTINEL) for s in samples)
    assert specs[0] == (kind, f"{extra}-cls-work-v-noresp")
    tc = r["train_classes"]
    assert tc["groups"] == "camera" and tc["tag"] == "cls-work-v-noresp"
    assert tc["positive"] == ["contact", "incomplete"] and tc["negative"] == ["no_response"]
    assert set(tc["scored_only"]) == set(NEVER) | {"unknown"}
    assert "no_response" in tc["scored_sets"]["nolick"] and "WORKING" in tc["scored_sets"]["engaged"]
    post = r["by_class"]["post"]
    assert set(post) >= {"contact", "incomplete", "none", "jaw_only", "after_end", "stopped", "stopped_licking",
                         "no_response", "unknown"}
    assert post["stopped"]["n"] > 0 and post["stopped"]["frac_working_like"] is not None
    assert post["no_response"]["n"] > 0 and post["unknown"]["n"] > 0
    assert post["no_response"]["n"] + post["unknown"]["n"] == post["none"]["n"]


def test_scored_post_sets_are_camera_defined(env):
    d, state, _, _ = env
    cg = state["cg"]
    r = pc.looks_like_which(d, KEEP)
    assert r["n_post_undetected"] == _n_post(d, cg, "nolick")      # "none" + raw no_response rows, both DAQ arms
    assert r["n_post_engaged"] == _n_post(d, cg, "engaged")         # camera WORKING rows, both DAQ arms
    # the camera sets are not the DAQ arms
    assert r["n_post_undetected"] != _n_post(d, None, "nolick")
    assert r["n_post_engaged"] != _n_post(d, None, "engaged")
    f = pc.fits_engaged_distribution(d, KEEP, n_boot=200)
    assert f["n_post_nolick"] == _n_post(d, cg, "nolick")


def test_scored_no_lick_set_is_the_working_period_only(env):
    """after_end ("quit") and stopped rows of raw class no_response are NOT in the scored no-lick set (pre or post);
    by_class still reports them in their own groups."""
    d, state, _, _ = env
    cg = state["cg"]
    for sess in (d["pre_i"], d["post_i"]):
        me, mu = pc._g13_rows(d, cg, KEEP, sess, "nolick")
        for g in ("after_end", "stopped"):
            bad_u = (cg["grp_u"] == g) & (cg["raw_u"] == "no_response") & np.isin(d["GU"], list(sess))
            assert bad_u.any() and not np.any(mu & bad_u)
        assert set(cg["grp_e"][me]) | set(cg["grp_u"][mu]) == {"none"}
        assert set(cg["raw_e"][me]) | set(cg["raw_u"][mu]) == {"no_response"}
    raw_all = int(sum(((np.isin(d[G], list(d["post_i"])) & np.isin(d[Y], KEEP) & (cg[r] == "no_response")).sum()
                       for G, Y, r in (("GE", "YE", "raw_e"), ("GU", "YU", "raw_u")))))
    r = pc.looks_like_which(d, KEEP)
    assert r["n_post_undetected"] == _n_post(d, cg, "nolick") < raw_all
    assert r["n_post_undetected"] == r["by_class"]["post"]["no_response"]["n"]
    assert r["by_class"]["post"]["after_end"]["n"] > 0 and r["by_class"]["post"]["stopped"]["n"] > 0


def test_daq_grouping_keeps_the_old_training_set_ids_and_scored_sets(env):
    d, state, specs, samples = env
    state["cg"] = None
    r = pc.looks_like_which(d, KEEP)
    assert specs[0] == ("g_lick_disc", "keep0-1-2-3-4-5-seed0")
    assert r["train_classes"]["groups"] == "daq" and r["by_class"] is None
    assert any(np.any(s[0][:, -1] == SENTINEL) for s in samples)    # the DAQ no-lick side holds stopped rows, as before
    assert r["n_post_undetected"] == _n_post(d, None, "nolick")
    assert r["n_post_engaged"] == _n_post(d, None, "engaged")
    f = pc.fits_engaged_distribution(d, KEEP, n_boot=200)
    assert f["n_post_nolick"] == _n_post(d, None, "nolick")
    assert ("g_lick_disc_folds", "keep0-1-2-3-4-5-seed0-v2") in specs


def test_daq_grouping_values_match_the_daq_arms(env):
    """Under the DAQ grouping the scored values are the DAQ engaged / no-lick arms exactly (the old code path)."""
    d, state, _specs, samples = env
    state["cg"] = None
    r = pc.looks_like_which(d, KEEP)
    X, y, _g = samples[0]
    clf = pc.lick_pipe().fit(X, y)
    post_u = np.isin(d["GU"], list(d["post_i"])) & np.isin(d["YU"], KEEP)
    post_e = np.isin(d["GE"], list(d["post_i"])) & np.isin(d["YE"], KEEP)
    pre_e = np.isin(d["GE"], list(d["pre_i"])) & np.isin(d["YE"], KEEP)
    pre_u = np.isin(d["GU"], list(d["pre_i"])) & np.isin(d["YU"], KEEP)
    assert r["post_undetected_frac_classified_ENGAGED_like"] == float(clf.predict(d["XU"][post_u]).mean())
    assert r["CONTROL_post_engaged_frac_engaged_like"] == float(clf.predict(d["XE"][post_e]).mean())
    assert r["scale_pre_engaged"] == float(clf.predict(d["XE"][pre_e]).mean())
    assert r["scale_pre_undetected"] == float(clf.predict(d["XU"][pre_u]).mean())


def test_undetected_state_split_shares_the_looks_like_which_model(env):
    import inspect
    src = inspect.getsource(pc.undetected_state_split)
    assert '_g_store(d, "g_lick_disc", f"{_keep_tag(keep)}-seed{seed}{_g13_tag(cg)}"' in src
    assert "_g13_sample(dT, cg, keep, rng)" in src
    assert '_g13_rows(d, cg, keep, d["post_i"], "nolick")[1]' in src


def test_row_groups_with_raw_tells_no_response_from_unknown(monkeypatch):
    from wfield_local import trial_classes as tc
    from wfield_local import working_trials as wt
    raw = ["contact", "no_response", "unknown", "incomplete", "no_response", "unknown"]
    after = [False, False, False, False, True, True]
    monkeypatch.setattr(tc, "daq_cue_seconds", lambda lab: np.arange(len(raw), dtype=float))
    monkeypatch.setattr(tc, "cue_classes_for", lambda lab, cs, window=None: np.asarray(raw, dtype=object))
    monkeypatch.setattr(tc, "stop_kind_for", lambda lab, cs, window=None: np.full(len(raw), "", object))
    monkeypatch.setattr(tc, "cue_groups_for", lambda lab, cs: (np.asarray(raw, dtype=object),
                                                               np.asarray(after, bool), np.ones(len(raw), bool)))
    idx = {"PSX_0806": (np.array([0, 3]), np.array([1, 2, 4, 5]))}
    args = (["PSX_0806"], idx, np.array([0, 0]), np.array([0] * 4))
    kw = dict(YE=np.array([3, 3]), YU=np.array([3] * 4))
    base = wt.row_groups(*args, **kw)
    got = wt.row_groups(*args, **kw, with_raw=True)
    assert len(base) == 5 and len(got) == 7
    for a, b in zip(base[:4], got[:4]):
        np.testing.assert_array_equal(a, b)
    assert list(got[1]) == ["none", "none", "after_end", "after_end"]
    assert list(got[5]) == ["contact", "incomplete"]
    assert list(got[6]) == ["no_response", "unknown", "no_response", "unknown"]
