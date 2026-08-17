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
