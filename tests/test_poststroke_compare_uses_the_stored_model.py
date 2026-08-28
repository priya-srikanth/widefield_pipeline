"""`poststroke_compare` reads the stored pre-stroke decoder instead of refitting its own.

WHY. Six places built the same object. `pooled_frozen_loso` froze it; `crossed_confusion` refitted it
three times, `impaired_nolick_readout` once, and `decode_matched`'s all-trials arm once -- each with a
bare `_pipe().fit(XE[tr], YE[tr])`, agreeing with the frozen model only by coincidence. And
`poststroke_section_g` mutates ONLY `d["post_i"]` before calling back in, so the pre-stroke fit --
which does not depend on `post_i` at all -- was redone once per post-stroke session per arm per
alignment (Priya, 2026-08-28: "let's not refit independently if we are replicating the exact same
thing").

The redundancy is the cheap half of the problem. The expensive half is that two copies of "the same"
model can drift: that is exactly how the training contamination fixed on 2026-08-26 survived eight
days, because whichever copy you read looked defensible on its own.

SO THE TEST THAT MATTERS IS EQUIVALENCE, not that the call happens. Substituting a stored model is
only legitimate if it is the same object, so these assert that the frozen path and the old inline fit
produce IDENTICAL predictions -- and that the sites which are NOT the same model still fit locally.
"""
import numpy as np
import pytest
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict

from wfield_local import frozen_models as F
from wfield_local import poststroke_compare as pc
from wfield_local.locanmf_frozen_decoder import _pipe, frozen_decoder_models
from wfield_local.plot_lick_aligned_averages import DISPLAY_ORDER


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(F, "local_dir", lambda: tmp_path / "frozen")
    monkeypatch.setattr(F, "server_dir", lambda: None)
    monkeypatch.setattr(F, "_session_sig", lambda lab: f"sig::{lab}")
    return tmp_path


@pytest.fixture
def pool():
    """A pooled set shaped like `_pooled`'s: 4 pre-stroke sessions, 2 post-stroke."""
    rng = np.random.default_rng(11)
    K = len(DISPLAY_ORDER)
    kept = ["PS99_0601", "PS99_0602", "PS99_0603", "PS99_0604", "PS99_0817", "PS99_0818"]
    per = 60
    YE = np.array([DISPLAY_ORDER[i % K] for i in range(per * len(kept))])
    GE = np.repeat(np.arange(len(kept)), per)
    XE = rng.normal(size=(YE.size, 10)) + YE[:, None] * 0.5
    nu = 90
    YU = np.array([DISPLAY_ORDER[i % K] for i in range(nu)])
    GU = np.repeat(np.arange(len(kept)), nu // len(kept))
    XU = rng.normal(size=(nu, 10)) + YU[:, None] * 0.5
    return {"XE": XE, "YE": YE, "GE": GE, "BE": [np.zeros(per, int)] * len(kept),
            "XU": XU, "YU": YU.astype(int), "GU": GU, "kept": kept,
            "pre_i": {0, 1, 2, 3}, "post_i": {4, 5},
            "align": "cue", "source": "roi", "_frozen_cache": {}}


def test_the_stored_full_model_predicts_identically_to_the_inline_fit(isolated, pool):
    """The substitution is only legitimate if it is the same object."""
    tr = np.isin(pool["GE"], list(pool["pre_i"]))
    te = np.isin(pool["GE"], list(pool["post_i"]))
    inline = _pipe().fit(pool["XE"][tr], pool["YE"][tr]).predict(pool["XE"][te])
    models = pc.frozen(pool)
    assert models is not None, "the stored model was not reachable at all"
    assert np.array_equal(models["full"].predict(pool["XE"][te]), inline)


def test_the_stored_loso_arm_reproduces_cross_val_predict(isolated, pool):
    """The pre-stroke reference band came from `cross_val_predict(LeaveOneGroupOut)`. Keyed by
    LABEL, `models["loso"]` must reproduce it fold for fold -- an index-keyed lookup would depend on
    the order the caller assembled its pool, and a label does not."""
    tr = np.isin(pool["GE"], list(pool["pre_i"]))
    want = cross_val_predict(_pipe(), pool["XE"][tr], pool["YE"][tr],
                             cv=LeaveOneGroupOut(), groups=pool["GE"][tr])
    models = pc.frozen(pool)
    got = np.empty_like(want)
    gi = pool["GE"][tr]
    for i in sorted(pool["pre_i"]):
        m = gi == i
        got[m] = models["loso"][pool["kept"][i]].predict(pool["XE"][tr][m])
    assert np.array_equal(got, want)


def test_narrowing_post_i_does_not_refit(isolated, pool):
    """`poststroke_section_g` mutates only `post_i`. The pre-stroke training set does not depend on
    it, so isolating one session must not mint a second model -- that refit, once per session per
    arm per alignment, is the bulk of the redundancy."""
    pc.frozen(pool)
    before = sorted(q.name for q in (F.local_dir() / "PS99").iterdir())
    pool["post_i"] = {4}
    pool["_frozen_cache"] = {}                 # a fresh pool dict, as section G builds
    pc.frozen(pool)
    after = sorted(q.name for q in (F.local_dir() / "PS99").iterdir())
    assert before == after, f"narrowing post_i froze a new model: {set(after) - set(before)}"


def test_it_is_memoised_within_one_pool(isolated, pool):
    """Called once per arm per alignment; the load must not repeat."""
    a = pc.frozen(pool)
    b = pc.frozen(pool)
    assert a is b, "the pool re-loaded the stored model instead of reusing it"


def test_a_pool_that_cannot_be_keyed_falls_back_rather_than_guessing(isolated):
    """A shortcut must never become a new source of truth. An `excluded_labels` pool, or a dict a
    test assembled, has no recorded align/source -- the caller then fits locally exactly as before."""
    assert pc.frozen({"kept": ["PS99_0601"], "pre_i": {0}}) is None
    assert pc.frozen({"kept": ["PS99_0601"], "pre_i": {0}, "_frozen_cache": {}}) is None


def test_the_class_filtered_arm_still_fits_its_own_model():
    """`decode_matched`'s lick-only arm restricts to the PRESERVED positions -- 4-way for PS94 and
    PS95, so a different chance level. Serving it the 6-way frozen model would silently change what
    its accuracy means, which is worse than the duplication being removed."""
    import inspect
    src = inspect.getsource(pc.decode_matched)
    assert "frozen(d) if post_all_trials else None" in src, (
        "the lick-only arm must not be served the unfiltered frozen model")


def test_the_within_session_ceiling_still_fits_its_own_model():
    """`_within_accuracy` is a same-day ceiling, not the cross-day frozen model."""
    import inspect
    src = inspect.getsource(pc._within_accuracy)
    assert "_pipe()" in src and "frozen(" not in src


def test_the_pool_is_the_one_the_model_was_frozen_against():
    """If `_pooled` rebuilt the label list instead of calling `config.pooled_labels`, the two could
    differ -- and for ROI features `_align_many` intersects region x bin columns across the pool, so
    a different pool is a different `n_features`, a different spec id, and a permanent cache MISS
    rather than a hit. The failure would look like slowness, not like an error."""
    import inspect
    assert "config.pooled_labels(" in inspect.getsource(pc._pooled)


def test_n_features_is_part_of_the_identity(isolated, pool):
    """The guard behind all of the above: a pool that produced a different feature space must get a
    LOUD new model, never a silent mis-scoring by one frozen at another width."""
    first = pc.frozen(pool)
    assert first is not None
    wide = dict(pool, XE=np.hstack([pool["XE"], pool["XE"][:, :3]]), _frozen_cache={})
    _models, status = frozen_decoder_models(
        wide["XE"], wide["YE"], wide["GE"], wide["kept"], wide["pre_i"],
        align="cue", source="roi", log=lambda *a, **k: None)
    assert status != F.STATUS_HIT, "a 13-feature pool was served a 10-feature model"
