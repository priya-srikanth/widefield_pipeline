"""The lick-free CONTROL panel averages pre-stroke sessions; the per-session panels keep all.

WHY THIS PANEL IS NOT JUST A FIGURE. `decode.precue_lickfree` is the ENL DEFINITION -- applied inside
`locanmf_position_decoder` for every pre-cue analysis on every session, which is why
`session_cache.CACHE_VERSION` was bumped to v8 when it was adopted and why
`position_coding_directions` calls the pre-cue window "the clean one". Section G's pre-cue arm, the
plan/execution dissociation, rests on it.

So this panel is the evidence that restricting to lick-free windows does not destroy the signal. It
was averaging every session it was handed and the nightly hands it all phases -- 39% post-stroke per
animal by 2026-08-26. Post-stroke sessions have a degraded position code, so including them makes the
control argue for LESS than it should. A control works by being strong.

WHAT STAYS UNRESTRICTED, deliberately: the per-session panels. Post-stroke lick-free EXPOSURE is
worth seeing -- post-stroke animals lick less, so the ENL is quieter -- and dropping those sessions
would trade one blind spot for another.

Found because Priya rejected the finding that this module is not used for pre/post work: "I thought
it was just how we were restricting the ENL definition (so yes we do actually probably use this for
pre-stroke, post-stroke, and cross-session analyses)". Correct -- the earlier check asked whether the
module's OUTPUT is read as a baseline, which is the wrong question when the module validates a
criterion used everywhere.
"""
import inspect

from wfield_local import precue_lickfree as pl


def _figure_src() -> str:
    return inspect.getsource(pl.figure)


def test_the_aggregate_is_restricted_to_prestroke():
    src = _figure_src()
    assert 'phase_labels("pre")' in src, "the mean confusion is not restricted to pre-stroke"
    i_filter = src.index('phase_labels("pre")')
    i_mean = src.index("np.nanmean(np.stack(cms)")
    assert i_filter < i_mean, "the filter is applied after the mean, so it cannot restrict it"


def test_the_panel_states_its_composition():
    """The ambiguity that hid the frozen-model contamination was a label that did not say what it
    covered. 'mean over sessions' is exactly that label."""
    src = _figure_src()
    assert "PRE-stroke session(s)" in src, "the panel title does not say which phase it averages"
    assert "excluded here" in src, "the title does not say post-stroke sessions were excluded"


def test_per_session_panels_keep_every_session():
    """Restricting the aggregate must not silently drop post-stroke sessions from the whole figure --
    their lick-free exposure is a real observation."""
    src = _figure_src()
    # `good` is the unfiltered list and must still drive the per-session bars
    assert "for r in good]" in src or "for r in good ]" in src, (
        "per-session panels no longer iterate the full session list")


def test_the_criterion_itself_is_not_phase_restricted():
    """The DEFINITION applies to every session; only the control's aggregate is pre-only. Restricting
    the criterion would change what a post-stroke pre-cue window even is."""
    src = inspect.getsource(pl)
    assert "phase" not in src.split("def analyse")[1].split("def ")[0], (
        "analyse() became phase-aware; the lick-free criterion must apply to all sessions")
