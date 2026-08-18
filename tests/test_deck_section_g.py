"""Section G (post-stroke) must obey the constraints its own speaker note states.

Section G is the part of the deck most likely to be read out of context and quoted, and it has
already produced two wrong headlines: a PS94 "neural deficit" that was mostly trial composition, and
a working-vs-disengaged result resting on a class that was never validated. Both were caught by a
person reading, not by a check.

These tests pin the mechanical properties that would silently break the section:

  * pools selected by PHASE, never by date (PS92/PS93 8/17 would otherwise be swept in),
  * the retired disengagement filter staying retired,
  * the note's quoted config values still matching the config.

They deliberately do NOT try to check that the prose is good. See tests/test_deck_notes_match_config.py
for that boundary.
"""
from __future__ import annotations

import inspect

import pytest

from wfield_local import config
from wfield_local import locanmf_analysis_deck as deck
from wfield_local import poststroke_compare as pc

SRC = inspect.getsource(deck)
NOTE = deck.M_POSTSTROKE


# ------------------------------------------------------------------------------------------------
# pools must come from the phase resolver, never from a date
# ------------------------------------------------------------------------------------------------
def test_section_g_selects_the_post_pool_by_phase():
    assert 'config.phase_labels("post")' in SRC, (
        "Section G must build its post-stroke pool from phase_labels('post')")


def test_section_g_never_selects_sessions_by_date():
    """The tempting shortcut. PS92/PS93 8/17 is a real, registered, projectable session that belongs
    to NEITHER phase, so any date-based selection pools two animals whose lesion did not take."""
    g = SRC.split("# ---------------- G. POST-STROKE")[-1]
    for bad in ('endswith("0817")', "endswith('0817')", 'label.endswith(', '== "0817"'):
        assert bad not in g, f"Section G selects sessions by date ({bad}); use phase_labels('post')"


def test_excluded_sessions_are_named_on_a_slide_not_just_dropped():
    """Dropping them silently is how a pooled slide reads as covering the whole cohort."""
    assert 'session_phase(a, "0817") == "excluded"' in SRC
    assert "G7. Excluded sessions" in SRC, "the section must state what it left out"


def test_the_post_pool_is_exactly_the_two_lesioned_animals():
    assert set(config.phase_labels("post")) == {"PS94_0817", "PS95_0817"}


# ------------------------------------------------------------------------------------------------
# the retired disengagement filter must stay retired
# ------------------------------------------------------------------------------------------------
def test_poststroke_engagement_filtering_is_off():
    """Retired 2026-08-18: there is no valid post-stroke construction of "disengaged".

    A spared-position gate is defensible as a DESCRIPTIVE statistic and indefensible as a trial
    filter, because a local dip in response rate is indistinguishable from a run of motor failures --
    and in a severe stroke no spared reference position exists at all.
    """
    assert pc.POSTSTROKE_ENGAGEMENT_FILTERING is False


def test_the_note_says_there_is_no_post_stroke_disengaged_label():
    assert "NO POST-STROKE 'DISENGAGED' LABEL" in NOTE
    assert "UNINTERPRETABLE" in NOTE, (
        "the note must say the old working-vs-disengaged result is uninterpretable, not negative")


def test_no_deck_note_presents_the_disengagement_split_as_a_result():
    """It may be described as retired; it may not be quoted as evidence."""
    for name in dir(deck):
        if not name.startswith("M_"):
            continue
        txt = getattr(deck, name)
        if not isinstance(txt, str) or "disengaged" not in txt.lower():
            continue
        low = txt.lower()
        assert "retired" in low or "uninterpretable" in low or "descriptive" in low, (
            f"{name} mentions disengagement post-stroke without marking it retired")


def test_the_replacement_analysis_exists_and_needs_no_engagement_label():
    """G6 must be a real function, not just a claim in prose."""
    assert callable(pc.impaired_nolick_readout)
    src = inspect.getsource(pc.impaired_nolick_readout)
    assert "poststroke_engagement" not in src, (
        "the replacement must not depend on the retired engagement gate")


# ------------------------------------------------------------------------------------------------
# the note's quoted values must match the config
# ------------------------------------------------------------------------------------------------
def test_note_quotes_the_real_stroke_date():
    animals = config.animals()
    dates = {v.get("stroke_date") for v in animals.values() if v.get("stroke_date")}
    assert dates, "fixture check: at least one animal must carry a stroke_date"
    for d in dates:
        assert str(d) in NOTE, f"M_POSTSTROKE does not quote the configured stroke_date {d}"


def test_note_quotes_the_real_laser_power():
    for a in ("PS94", "PS95"):
        mw = config.animals()[a].get("stroke_laser_power_mW")
        if mw is not None:
            assert f"{mw} mW" in NOTE, f"M_POSTSTROKE does not quote {a}'s {mw} mW"


def test_note_states_the_matched_chance_level_and_that_it_is_not_comparable():
    """Position-matched slides are 4-way. Quoting them beside the 6-way numbers in sections A-F is
    the single easiest mistake to make with this section."""
    assert "4-way" in NOTE and "0.25" in NOTE
    assert "NOT comparable" in NOTE


def test_note_declares_the_pre_engaged_vs_post_all_mismatch():
    from wfield_local import nolick_analysis as na
    assert ("engaged_2s", "all_trials") in na.SANCTIONED_MISMATCHES
    assert "SANCTIONED_MISMATCHES" in NOTE


def test_note_carries_the_no_lick_is_not_no_protrusion_caveat():
    """The caveat that limits every no-lick conclusion in the section."""
    low = NOTE.lower()
    assert "is not 'no tongue protrusion'" in low, (
        "the note must state that an undetected lick is not an absent tongue protrusion")
    assert "DLC" in NOTE, "and must name what would replace the inference with a measurement"


def test_note_states_the_sample_size_honestly():
    assert "n = 1 post-stroke session" in NOTE or "ONE SESSION" in NOTE


# ------------------------------------------------------------------------------------------------
# the figures the section references must be producible
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("fn", ["fig_behaviour", "fig_matched", "fig_per_position",
                                "fig_confusion", "fig_identity", "fig_similarity",
                                "fig_nolick_readout"])
def test_every_section_g_figure_function_exists(fn):
    from wfield_local import plot_poststroke
    assert callable(getattr(plot_poststroke, fn))


def test_section_g_is_skipped_entirely_when_there_is_no_post_stroke_session():
    """Before the first lesion the whole section must be absent rather than empty.

    The guard is `if _post_labels and ...` -- an empty section with a divider and seven blank slides
    would look like a failed build.
    """
    assert "if _post_labels and (src /" in SRC
