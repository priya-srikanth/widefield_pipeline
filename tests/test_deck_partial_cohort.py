"""A night that ran two animals must not read as eight missing figures.

2026-08-23: 8/22 recorded PS92 and PS93 only. The deck's per-session slides iterate animals x dates,
so it constructed filenames for PS94_0822 and PS95_0822, counted all of them missing, and refused to
publish:

    !! analysis deck NOT PUBLISHED: 12 figure(s) missing, existing deck left untouched
       missing: locanmf_frozen_session_PS94_0822_roi_cue.png
       missing: locanmf_frozen_session_PS95_0822_roi_cue.png
       ...

Every one was for a session that does not exist. The completeness gate is worth keeping -- it is
what catches a step that genuinely failed -- so the fix is to stop it expecting the impossible,
never to loosen the threshold. Partial cohorts are now normal: 8/22 is two animals, and any night
where one mouse is rested will be too.
"""
import inspect

from wfield_local import config
from wfield_local import locanmf_analysis_deck as deck


def _registered_pairs():
    return {(config.animal_of(x["label"]), x["label"].split("_")[-1])
            for x in config.load_sessions()}


def test_the_cohort_really_is_partial_on_some_dates():
    """Pinning the CONDITION, so this test keeps its meaning as nights accumulate."""
    reg = _registered_pairs()
    dates = {d for _, d in reg}
    animals = {a for a, _ in reg}
    partial = [d for d in dates if sum((a, d) in reg for a in animals) < len(animals)]
    assert partial, "no partial night registered; this guard is untested against real data"


def test_the_deck_asks_whether_a_session_was_recorded():
    """Source-level: the per-session grids must be filtered by a registration check.

    The failure is silent in the sense that matters -- the deck reports a number, refuses, and the
    number is real; it is the EXPECTATION behind it that is wrong. That cannot be caught by
    counting."""
    src = inspect.getsource(deck.build_analysis_deck)
    assert "def have(" in src, "no registration check in the deck builder"
    assert src.count("if have(a, d)") >= 3, (
        "the per-session grids (frozen cue/precue, decoder cue/precue) must each skip animals that "
        "were not recorded on that date")


def test_no_unrecorded_session_would_be_requested():
    """The concrete case: PS94/PS95 on 8/22."""
    reg = _registered_pairs()
    assert ("PS92", "0822") in reg and ("PS93", "0822") in reg
    for a in ("PS94", "PS95"):
        assert (a, "0822") not in reg, f"{a}_0822 is registered; this test needs a new example"


def test_a_recorded_session_is_still_expected():
    """The guard must not turn into 'expect nothing'. 8/21 ran all four."""
    reg = _registered_pairs()
    for a in ("PS92", "PS93", "PS94", "PS95"):
        assert (a, "0821") in reg, "8/21 should have all four; if not, pick another full night"
