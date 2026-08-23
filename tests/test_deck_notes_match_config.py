"""The deck's methodology notes must describe what the code ACTUALLY does.

Speaker notes are prose, and prose does not recompute. This project has now shipped, in the deck:
a warning that pre-cue numbers were "inflated ~2x" after they were corrected; a claim that no-lick
trials "decode at chance" rendered onto a figure TITLE; a cross-reference to the wrong section; a
transfer-cost table from before a bug that changed every number in it; and an "engaged" definition
that was never the one implemented. Every one was found by a person reading, not by a check.

A test cannot verify that prose is well written. It CAN verify that the specific numbers and setting
names the prose quotes still match `configs/defaults.yaml` — which is where the drift actually
happens, because the config changes and the sentence does not.

Keep these assertions to values a reader would ACT on. Pinning every phrase would make the notes
unmaintainable and the test a nuisance, and a nuisance test gets deleted.
"""
from __future__ import annotations

import pytest

from wfield_local import config
from wfield_local import locanmf_analysis_deck as deck

DP = config.defaults()["decode"]
ALL_NOTES = {n: getattr(deck, n) for n in dir(deck) if n.startswith("M_") and isinstance(getattr(deck, n), str)}


def test_engaged_definition_quotes_the_real_cut():
    """`engaged` is decode.max_rt_s, NOT the task's response window.

    The notes said "a lick inside the response window" until 2026-08-17. They are different numbers
    (2.0 s vs 3500 ms) and the difference is 2.15% of all rewarded hits, which is exactly why the
    late_rewarded category exists.
    """
    assert f"decode.max_rt_s = {DP['max_rt_s']:.1f} s" in deck.M_COMMON
    for name, txt in ALL_NOTES.items():
        assert "lick inside the response window" not in txt, (
            f"{name} still defines engaged as the response window; it is decode.max_rt_s")


def test_precue_note_states_whether_the_window_is_lick_free():
    """Whether the headline pre-cue number is lick-free is the single most load-bearing fact about
    it, and it changed on 2026-08-17."""
    lickfree = DP.get("precue_lickfree", False)
    if lickfree:
        assert "LICK-FREE" in deck.M_DECODE and "decode.precue_lickfree" in deck.M_DECODE
        assert "DROPPED" in deck.M_DECODE, "the note must say trials with no clean window are dropped"
    else:
        assert "LICK-FREE window ending" not in deck.M_DECODE, (
            "the note claims a lick-free pre-cue window but decode.precue_lickfree is off")


def test_bin_counts_in_the_notes_match_the_config():
    b = DP["bins"]
    assert f"precue/cue {b['precue']}" in deck.M_COMMON + deck.M_DECODE or \
           f"{b['precue']} x 0.5 s" in deck.M_DECODE, "pre-cue bin count not stated or stale"
    assert f"post-lick {b['lick']} x 0.25 s" in deck.M_DECODE or f"lick {b['lick']}" in deck.M_DECODE


def test_chance_level_matches_the_config():
    assert f"chance={DP['chance']:.3f}" in deck.M_DECODE or f"chance {DP['chance']:.3f}" in deck.M_DECODE


def test_precue_subbinning_is_not_advertised_as_established():
    """The +0.032 pilot gain did not replicate (+0.009, 23/44). No note may quote the old figure."""
    for name, txt in ALL_NOTES.items():
        if "+0.032" in txt:
            assert "did not replicate" in txt or "UNESTABLISHED" in txt, (
                f"{name} quotes the withdrawn +0.032 pre-cue gain without the retraction")


def test_no_note_claims_nolick_decodes_at_chance():
    """Retired 2026-08-17: the claim was wrong AND its null was wrong."""
    for name, txt in ALL_NOTES.items():
        low = txt.lower()
        if "decode at chance" in low or "decodes at chance" in low:
            assert "corrected" in low or "previously" in low, (
                f"{name} still asserts no-lick trials decode at chance")


def test_frozen_note_does_not_quote_the_pre_bug17_transfer_costs():
    """Bug 17 changed every ROI frozen number; the 8/11 magnitudes must not stand unqualified."""
    if "+0.102" in deck.M_FROZEN:
        assert "bug 17" in deck.M_FROZEN or "recomputed" in deck.M_FROZEN.lower(), (
            "M_FROZEN quotes the pre-bug-17 transfer costs as current")


def test_every_note_is_attached_to_at_least_one_slide():
    """A note nobody reads cannot be wrong, but it also cannot be right -- and it rots.

    Guards against a note constant surviving the deletion of the section that used it.
    """
    import inspect
    src = inspect.getsource(deck)
    for name in ALL_NOTES:
        if name in ("M_COMMON",):        # composed into the others rather than used directly
            continue
        uses = src.count(name)
        assert uses >= 2, f"{name} is defined but never attached to a slide"


#: Notes whose numbers rest on the ENGAGED / NO-LICK split. These are the ones that move when
#: decode.max_rt_s changes, so each must carry the warning that its prose predates the current cut.
#: M_HEMI, M_VESSEL, M_HEMIDYN and M_FIXEDSCALE are deliberately absent -- they read RAW
#: fluorescence and never split on a lick, so warning there would advertise a dependency they do
#: not have. M_COMMON carries its own longer version and reaches six more notes by embedding.
TRIAL_SPLIT_NOTES = ("M_EVOKED", "M_SPATIAL", "M_RECODING", "M_NOLICK", "M_PRECUE_CAVEAT",
                     "M_LICKFREE", "M_CODING_DIR")


@pytest.mark.parametrize("name", TRIAL_SPLIT_NOTES)
def test_notes_that_split_trials_carry_the_engaged_cut_warning(name):
    """A number resting on the engaged/no-lick boundary must say which boundary produced it.

    The cut moved 2.0 -> 3.5 s on 2026-08-21 and every earlier number shifts on rebuild. The warning
    went into M_COMMON, which reaches only the six notes that embed it -- these seven quote results
    from the same split and embed nothing, so they were silently exempt.
    """
    note = ALL_NOTES[name]
    assert "2.0 s" in note and "3.5 s" in note, (
        f"{name} quotes results from the engaged/no-lick split but does not say which cut produced "
        f"them; append M_GATE (or M_COMMON) when adding such a note")


def test_the_gate_warning_is_not_pasted_into_notes_that_do_not_split_trials():
    """The complement, so the warning keeps meaning something.

    A caveat attached to every note is read as boilerplate and stops being read at all. The raw-
    fluorescence notes have no engaged/no-lick dependency, and saying otherwise would send a reader
    looking for one.
    """
    for name in ("M_HEMI", "M_VESSEL", "M_HEMIDYN"):
        assert deck.M_GATE not in ALL_NOTES[name], (
            f"{name} reads raw fluorescence and does not split on a lick -- M_GATE does not apply")


def test_every_lick_aligned_note_says_which_lick():
    """ANALYSIS = one reference per TRIAL (first lick in the window). PREPROCESSING = one per LICK.

    Neither convention said so, and the silence made the two decks look contradictory: PS94 8/17
    far_R is n=83 on the preprocessing map and "not attempted" on fixed_scale_maps, because those 83
    licks belonged to fewer than 8 trials. The preprocessing note also claimed "First-lick-aligned
    maps" for a figure that has never been first-lick.
    """
    from wfield_local import locanmf_analysis_deck as deck

    unstated = [n for n, v in ALL_NOTES.items()
                if "lick" in v.lower() and deck._M_LICK_UNIT not in v and n != "_M_LICK_UNIT"]
    assert not unstated, (
        f"{sorted(unstated)} mention licks but do not state whether an n counts TRIALS or LICKS; "
        f"append _M_LICK_UNIT (or embed M_COMMON / M_GATE, which carry it)")


def test_the_preprocessing_deck_states_the_other_convention():
    """The complement: the preprocessing note must say EVERY lick, and must not claim first-lick."""
    from wfield_local import preprocess_deck as pdeck

    gate = pdeck._M_LICK_GATE
    assert "EVERY lick inside a trial" in gate
    assert "NOT the first lick of each trial" in gate
    for key in ("lick_maps", "lick_quietnorm", "lick_pairwise", "cue_vs_lick"):
        note = pdeck.METHOD_NOTES[key]
        assert gate in note, f"{key} is a lick figure and must carry the unit + gate note"
        assert "First-lick-aligned" not in note, (
            f"{key} said 'First-lick-aligned', which that figure has never been")
