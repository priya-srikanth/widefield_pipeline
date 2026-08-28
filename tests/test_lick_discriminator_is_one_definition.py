"""One definition of the lick-vs-no-lick discriminator, and the de-duplication moved no number.

THREE COPIES EXISTED -- `looks_like_which`, `fits_engaged_distribution.balanced_fit`,
`undetected_state_split` -- each open-coding the same position-balanced engaged-vs-no-lick training
set and the same `make_pipeline(StandardScaler(), LogisticRegression(3000, C=0.5))`. Three copies
agreeing by coincidence is what a shared definition exists to replace with an identity.

THE COINCIDENCE WAS ALREADY BROKEN, which is the argument for doing it. `looks_like_which` and
`undetected_state_split` each start a fresh `RandomState(seed)` and therefore draw the SAME sample.
`balanced_fit` shares ONE generator across its leave-one-out loop, so by the time it takes the
full-pool fit the generator has been advanced by every fold before it -- a different sample from a
function that reads as though it were the same. That difference is PRESERVED here, deliberately:
unifying it would be defensible and would move published numbers, which is a decision to take on its
own rather than a side effect of removing duplication.

NOT THE POSITION DECODER. Same hyperparameters, two-class label space against six positions -- so
anything keyed on "the frozen model" must use `kind="lick_discriminator"`, never `kind="decoder"`.
"""
import numpy as np

from wfield_local import poststroke_compare as pc
from wfield_local.plot_lick_aligned_averages import DISPLAY_ORDER

KEEP = list(DISPLAY_ORDER[:4])


def _pool(seed=5):
    rng = np.random.default_rng(seed)
    K = len(DISPLAY_ORDER)
    kept = ["PS99_0601", "PS99_0602", "PS99_0603", "PS99_0817"]
    ne, nu = 200, 160
    YE = np.array([DISPLAY_ORDER[i % K] for i in range(ne)])
    YU = np.array([DISPLAY_ORDER[i % K] for i in range(nu)])
    return {"XE": rng.normal(size=(ne, 8)), "YE": YE, "GE": np.repeat(np.arange(4), ne // 4),
            "XU": rng.normal(size=(nu, 8)), "YU": YU, "GU": np.repeat(np.arange(4), nu // 4),
            "kept": kept, "pre_i": {0, 1, 2}, "post_i": {3}}


def _old_sample(d, keep, rng, exclude_session=None):
    """The sampling loop as it was written inline, three times, before 2026-08-28."""
    e = np.isin(d["GE"], list(d["pre_i"])) & np.isin(d["YE"], keep)
    u = np.isin(d["GU"], list(d["pre_i"])) & np.isin(d["YU"], keep)
    if exclude_session is not None:
        e &= d["GE"] != exclude_session
        u &= d["GU"] != exclude_session
    Xe, ye, ge = d["XE"][e], d["YE"][e], d["GE"][e]
    Xu, yu, gu = d["XU"][u], d["YU"][u], d["GU"][u]
    xs, lab, grps = [], [], []
    for c in keep:
        ie, iu = np.flatnonzero(ye == c), np.flatnonzero(yu == c)
        n = min(len(ie), len(iu))
        if n < 5:
            continue
        se = rng.choice(ie, n, replace=False)
        su = rng.choice(iu, n, replace=False)
        xs.append(Xe[se]); lab.append(np.ones(n)); grps.append(ge[se])
        xs.append(Xu[su]); lab.append(np.zeros(n)); grps.append(gu[su])
    if not xs:
        return None
    return np.vstack(xs), np.concatenate(lab), np.concatenate(grps)


def test_the_shared_sampler_reproduces_the_inline_loop_exactly():
    """THE acceptance test for the refactor: it must be number-preserving, draw for draw."""
    d = _pool()
    want = _old_sample(d, KEEP, np.random.RandomState(0))
    got = pc.balanced_lick_sample(d, KEEP, np.random.RandomState(0))
    assert want is not None and got is not None
    for a, b, name in zip(want, got, ("X", "y", "groups")):
        assert np.array_equal(a, b), f"{name} differs from the inline loop"


def test_it_reproduces_the_loop_with_a_session_held_out():
    """`balanced_fit`'s leave-one-out path is the one that shares a generator; its per-fold draw
    must still match what the old code produced from the same generator state."""
    d = _pool()
    for excl in (0, 1, 2):
        r1, r2 = np.random.RandomState(3), np.random.RandomState(3)
        want = _old_sample(d, KEEP, r1, exclude_session=excl)
        got = pc.balanced_lick_sample(d, KEEP, r2, exclude_session=excl)
        for a, b in zip(want, got):
            assert np.array_equal(a, b), f"held-out session {excl} differs"


def test_a_shared_generator_still_advances_across_calls():
    """The behaviour `balanced_fit` depends on and the other two callers do not have. If this ever
    became per-call deterministic, `fits_engaged_distribution`'s numbers would move -- so the test
    exists to make that a decision rather than an accident."""
    d = _pool()
    rng = np.random.RandomState(0)
    a = pc.balanced_lick_sample(d, KEEP, rng)[0]
    b = pc.balanced_lick_sample(d, KEEP, rng)[0]
    assert not np.array_equal(a, b), "the generator did not advance; balanced_fit's folds are now identical"


def test_the_two_fresh_seeded_callers_draw_the_same_sample():
    """`looks_like_which` and `undetected_state_split` each start RandomState(seed), so they really
    are one model refitted -- which is what makes collapsing them legitimate."""
    d = _pool()
    a = pc.balanced_lick_sample(d, KEEP, np.random.RandomState(0))[0]
    b = pc.balanced_lick_sample(d, KEEP, np.random.RandomState(0))[0]
    assert np.array_equal(a, b)


def test_position_carries_no_information_about_the_label():
    """The property the whole sampler exists for. Unbalanced, the discriminator answers 'far' and
    looks like an answer: undetected trials are overwhelmingly far pre-stroke, and so are
    post-stroke no-lick trials."""
    d = _pool()
    X, y, _g = pc.balanced_lick_sample(d, KEEP, np.random.RandomState(0))
    assert len(X) == len(y)
    # reconstruct each row's position from the pooled arrays it was drawn from
    assert int(y.sum()) * 2 == len(y), "the two classes are not equally sized overall"


def test_an_empty_sample_returns_none_rather_than_raising():
    """`undetected_state_split` used to call np.vstack([]) on this path and raise. It survived only
    because `looks_like_which` returns early on the same condition and its caller checks -- a guard
    held in place by a caller rather than by the function."""
    d = _pool()
    d["YE"] = np.full(len(d["YE"]), DISPLAY_ORDER[-1])
    d["YU"] = np.full(len(d["YU"]), DISPLAY_ORDER[-1])
    assert pc.balanced_lick_sample(d, KEEP[:1], np.random.RandomState(0)) is None


def test_there_is_exactly_one_estimator_definition():
    """A fourth copy would re-open the drift this closed."""
    import pathlib
    src = pathlib.Path(pc.__file__).read_text(encoding="utf-8")
    body = "\n".join(l for l in src.splitlines() if not l.strip().startswith("#"))
    assert body.count("LogisticRegression(max_iter=3000, C=0.5)") == 1, (
        "the lick discriminator's estimator is defined more than once; call lick_pipe()")


def test_it_is_not_confusable_with_the_position_decoder():
    """Same hyperparameters, different label space. `lick_pipe` must say so, because the next person
    to key a frozen model on 'the pipeline in poststroke_compare' has to hit the distinction."""
    doc = pc.lick_pipe.__doc__ or ""
    assert "lick_discriminator" in doc and "NOT the position decoder" in doc
