"""Deck section H places 19 grant-figure patterns, so the nightly must regenerate them.

THE RULE THIS ENFORCES, already stated in nightly_figs for the post-stroke stage: "If it is part of
the deck it is part of the nightly." Until 2026-08-27 `nightly_figs` never invoked `grant_figures`,
so every deck built section H from whatever PNGs happened to be on disk. A night nobody re-rendered
by hand shipped a section H that was silently a day stale -- the same failure that put twelve
section-G figures on a superseded basis in August.
"""
import inspect
import re

from wfield_local import locanmf_analysis_deck as ad
from wfield_local import nightly_figs


def test_nightly_figs_invokes_grant_figures():
    src = inspect.getsource(nightly_figs)
    assert re.search(r'cli\(\s*"wfield_local\.grant_figures"', src), (
        "nightly_figs must run grant_figures: deck section H places its output, and a deck input "
        "that no nightly step regenerates is frozen at the day it was last made by hand")


def test_grant_figures_runs_before_the_deck_is_built():
    """Order matters: the deck READS the PNGs grant_figures writes. Running it after would place
    the previous render's figures and look correct."""
    src = inspect.getsource(nightly_figs)
    grant = src.index('cli("wfield_local.grant_figures"')
    deck = src.index("build_analysis_deck(")
    assert grant < deck, "grant_figures must run BEFORE build_analysis_deck, not after"


def test_section_h_still_places_grant_figures():
    """If section H ever stops placing them, the nightly step above becomes dead weight and this
    test is the reminder to reconsider it -- rather than leaving an hours-long step nothing reads."""
    src = inspect.getsource(ad)
    assert src.count('("grant_') > 10, "section H no longer places grant figures; revisit the nightly step"
