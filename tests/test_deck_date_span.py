"""A hand-run deck must cover the same dates as the nightly one.

`curated_dates()` became stroke-aware on 2026-08-17: its default is `phase="pre"`, so the bare call
returns only dates up to the lesion. `build_analysis_deck` defaulted to that bare call, under a
comment asserting that a hand-run deck therefore covered the same dates as the nightly. The comment
was true when written and false from the day the phase default landed.

The nightly never hit it, because it builds its own list (registered minus excluded) and passes it
in. So the defect was invisible to every automated run and only appeared when a person rebuilt the
deck by hand -- on 2026-08-22, publishing 249 slides over the nightly's 265 and losing the
post-stroke sections, which are the part of the study the deck exists to show.

Both directions are pinned here: the deck's span must reach the post-stroke dates, and it must
agree with the set the nightly computes.
"""
from wfield_local import config


def _nightly_date_list():
    """How nightly_figs builds its cross-session list: registered, minus the excluded ones."""
    exclude = set(config.date_policy().get("cross_session_exclude", []))
    registered = sorted({s["label"].split("_")[1] for s in config.load_sessions()})
    return [d for d in registered if d not in exclude]


def test_the_deck_default_reaches_the_post_stroke_dates():
    cut = config.stroke_cutoff()
    if cut is None:
        return                                  # whole cohort still pre-stroke: nothing to check
    dates = config.curated_dates(phase="all")
    assert any(d > cut for d in dates), (
        f"the deck's default date set stops at {dates[-1]!r} against a stroke cutoff of {cut!r}; "
        f"every post-stroke slide would be silently absent")


def test_the_deck_default_matches_what_the_nightly_passes():
    """The two must not be able to drift apart again."""
    assert config.curated_dates(phase="all") == _nightly_date_list()


def test_the_bare_call_really_is_pre_only():
    """Pinning WHY the explicit phase is needed, so nobody 'simplifies' it back to curated_dates()."""
    cut = config.stroke_cutoff()
    if cut is None:
        return
    assert all(d <= cut for d in config.curated_dates()), (
        "curated_dates() is expected to be pre-only by default; if that changed, the explicit "
        "phase='all' in build_analysis_deck may no longer be doing what its comment claims")


def test_build_analysis_deck_asks_for_all_phases():
    """Source-level, because the failure is a default that looks harmless at the call site."""
    import inspect

    from wfield_local import locanmf_analysis_deck as deck

    src = inspect.getsource(deck.build_analysis_deck)
    assert 'config.curated_dates(phase="all")' in src, (
        "build_analysis_deck must request phase='all' explicitly; the bare curated_dates() default "
        "drops every post-stroke date")
