"""Section G's G1-G3 state discriminators (`looks_like_which`, `fits_engaged_distribution`, `undetected_state_split`)
follow decision (f) 1 under the camera grouping (Priya 11-05; settings audit 11-08): train WORKING (contact +
incomplete, either DAQ arm) vs NONE, never on stopped / stopped_licking / jaw_only / after_end rows; every camera group is
scored on its own; the stored ids carry the class set. The DAQ grouping keeps the old training set and ids.
"""
import numpy as np
import pytest

from wfield_local import poststroke_compare as pc

KEEP = [0, 1, 2, 3, 4, 5]
NEVER = ("jaw_only", "after_end", "stopped", "stopped_licking")
SENTINEL = 1e3


def _pool(seed=0, n_pre=4, n_post=1, n=240, n_u=200):
    rng = np.random.default_rng(seed)
    XE, YE, GE, XU, YU, GU, ge, gu = [], [], [], [], [], [], [], []
    for i in range(n_pre + n_post):
        y = rng.integers(0, 6, n)
        XE.append(rng.normal(size=(n, 6)) + np.eye(6)[y] + 0.5)
        YE.append(y)
        GE.append(np.full(n, i))
        ge.append(rng.choice(["contact", "incomplete", "stopped_licking"], n, p=[.85, .1, .05]))
        yu = rng.integers(0, 6, n_u)
        g = rng.choice(["incomplete", "jaw_only", "none", "after_end", "stopped"], n_u, p=[.25, .15, .3, .15, .15])
        XU.append(rng.normal(size=(n_u, 6)) + np.eye(6)[yu] * .3 - 0.5 * (g == "none")[:, None])
        YU.append(yu)
        GU.append(np.full(n_u, i))
        gu.append(g)
    cat = np.concatenate
    gE, gU = cat(ge).astype(object), cat(gu).astype(object)
    d = {"XE": np.vstack(XE), "YE": cat(YE), "GE": cat(GE), "XU": np.vstack(XU), "YU": cat(YU), "GU": cat(GU),
         "kept": [f"PSX_08{10 + i}" for i in range(n_pre + n_post)], "pre_i": set(range(n_pre)),
         "post_i": set(range(n_pre, n_pre + n_post)), "align": "cue", "source": "roi", "_frozen_cache": {}}
    # rows of the never-trained groups carry a sentinel in the last column: none may reach a training sample
    d["XE"][np.isin(gE, NEVER), -1] = SENTINEL
    d["XU"][np.isin(gU, NEVER), -1] = SENTINEL
    cg = {"grp_e": gE, "grp_u": gU, "jj_e": np.ones(len(gE), bool), "jj_u": np.ones(len(gU), bool),
          "have": set(range(n_pre + n_post)), "work_e": np.isin(gE, ["contact", "incomplete"]),
          "work_u": np.isin(gU, ["contact", "incomplete"])}
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
            samples.append(np.asarray(got[0]))
        return got

    monkeypatch.setattr(pc, "_g_store", _store)
    monkeypatch.setattr(pc, "_g13_sample", _spy)
    monkeypatch.delenv("WIDEFIELD_STATE_DISC_SCALING", raising=False)
    return d, state, specs, samples


def test_sample_is_working_vs_none_and_never_a_stopped_row(env):
    d, state, _, _ = env
    X, y, _g = pc._g13_sample(d, state["cg"], KEEP, np.random.RandomState(0))
    assert not np.any(X[:, -1] == SENTINEL)
    want = pc.cam_balanced_sample(d, state["cg"]["grp_e"], state["cg"]["grp_u"], KEEP, np.random.RandomState(0))
    for a, b in zip((X, y), want[:2]):
        np.testing.assert_array_equal(a, b)
    # DAQ grouping: the old sampler, draw for draw
    got = pc._g13_sample(d, None, KEEP, np.random.RandomState(0))
    old = pc.balanced_lick_sample(d, KEEP, np.random.RandomState(0))
    for a, b in zip(got, old):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("fn,kw,kind,extra", [
    ("looks_like_which", {}, "g_lick_disc", "keep0-1-2-3-4-5-seed0"),
    ("fits_engaged_distribution", {"n_boot": 200}, "g_lick_disc_folds", "keep0-1-2-3-4-5-seed0-v2")])
def test_camera_grouping_trains_on_camera_classes(env, fn, kw, kind, extra):
    d, _state, specs, samples = env
    r = getattr(pc, fn)(d, KEEP, **kw)
    assert samples and all(not np.any(X[:, -1] == SENTINEL) for X in samples)
    assert specs[0] == (kind, f"{extra}-{pc.G13_TAG}")
    tc = r["train_classes"]
    assert tc["groups"] == "camera" and tc["positive"] == ["contact", "incomplete"] and tc["negative"] == ["none"]
    assert set(tc["scored_only"]) == set(NEVER)
    post = r["by_class"]["post"]
    assert set(post) >= {"contact", "incomplete", "none", "jaw_only", "after_end", "stopped", "stopped_licking"}
    assert post["stopped"]["n"] > 0 and post["stopped"]["frac_working_like"] is not None


def test_daq_grouping_keeps_the_old_training_set_and_ids(env):
    d, state, specs, samples = env
    state["cg"] = None
    r = pc.looks_like_which(d, KEEP)
    assert specs[0] == ("g_lick_disc", "keep0-1-2-3-4-5-seed0")
    assert r["train_classes"]["groups"] == "daq" and r["by_class"] is None
    assert any(np.any(X[:, -1] == SENTINEL) for X in samples)    # the DAQ no-lick side holds stopped rows, as before


def test_undetected_state_split_shares_the_looks_like_which_model(env):
    import inspect
    src = inspect.getsource(pc.undetected_state_split)
    assert '_g_store(d, "g_lick_disc", f"{_keep_tag(keep)}-seed{seed}{_g13_tag(cg)}"' in src
    assert "_g13_sample(dT, cg, keep, rng)" in src
