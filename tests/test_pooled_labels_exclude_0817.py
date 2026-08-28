"""PS92/PS93 8/17 is neither phase, and a pooled analysis must not carry it.

Priya, 2026-08-28: "we should drop PS92/93 8/17 from the pooled data (neither true pre- nor true
post-stroke)". `session_phase` already returned 'excluded' for both; what did not was the LABEL LIST
the pooled analyses were built from.

WHY IT WAS NOT MERELY UNTIDY. `pooled_frozen_loso` defines `post_i` as "every session that is not
pre-stroke", so an excluded session did not sit inertly in the pool -- it was scored by the frozen
decoder and REPORTED as a post-stroke day, in `per_session`, in `post_labels`, and in every figure
built from them. For ROI features `_align_many` also intersects region x bin columns across the whole
pool, so one extra session can shrink `n_features` for every other session too.

THE ROOT CAUSE was a date list standing in for a phase. `curated_dates(phase="all")` is cohort-wide
and cannot express "8/17 is post-stroke for PS94 and PS95 but excluded for PS92 and PS93" -- which is
the actual state of this cohort. Three call sites built the list independently and identically
wrongly, which is why the definition now lives in one place.
"""
import inspect

from wfield_local import config


def test_the_excluded_sessions_are_not_in_any_pool():
    labs = config.pooled_labels()
    assert "PS92_0817" not in labs and "PS93_0817" not in labs


def test_the_post_stroke_0817s_ARE_kept():
    """The same date IS post-stroke for the other two animals. A cohort-wide date exclusion would
    have thrown those away too, which is the thing a per-animal phase resolves."""
    labs = config.pooled_labels()
    assert "PS94_0817" in labs and "PS95_0817" in labs


def test_the_phase_really_is_excluded_for_those_two():
    """Pins the CONDITION the fix relies on, so the test keeps its meaning if the tagging moves."""
    assert config.session_phase("PS92", "0817") == "excluded"
    assert config.session_phase("PS93", "0817") == "excluded"
    assert config.session_phase("PS94", "0817") == "post"
    assert config.session_phase("PS95", "0817") == "post"


def test_pooled_labels_is_exactly_pre_plus_post():
    """No third category can leak in. If a new phase tag is ever added, this fails rather than
    silently pooling it."""
    assert sorted(config.pooled_labels()) == sorted(
        config.phase_labels("pre") + config.phase_labels("post"))


def test_filtering_by_animal_agrees_with_the_whole():
    whole = config.pooled_labels()
    for an in config.animals():
        assert config.pooled_labels(an) == sorted(x for x in whole if config.animal_of(x) == an)


def test_every_POOLED_call_site_uses_the_helper():
    """The regression that matters. Three modules built this list themselves; a fourth copy is how
    the pools diverge again -- and they HAD diverged: poststroke_compare already used pre+post while
    the nightly used the raw date list, so two analyses of the same cohort disagreed about which
    sessions exist.

    SCOPED TO MODULES THAT POOL. `precue_lickfree` iterates the same way but reports one ROW per
    session rather than fitting one model across them, so an excluded session there is a row a reader
    can see and discount, not a day silently relabelled post-stroke. Whether it should carry 8/17 is
    a separate question about the ENL definition, and is left to be asked rather than assumed.
    """
    from wfield_local import joint_xsession, locanmf_frozen_decoder, nightly_figs

    for mod in (joint_xsession, locanmf_frozen_decoder, nightly_figs):
        src = inspect.getsource(mod)
        for bad in ('s["label"].startswith(an) and s["label"][-4:] in dates',
                    's["label"].startswith(a) and s["label"][-4:] in set(from_list)'):
            assert bad not in src, f"{mod.__name__} still builds a pool from a raw date list"
