"""Deck-note completeness, in both directions.

Priya, 2026-09-12: "verify that the deck notes are accurate and complete across all figure
families!!"

The first two versions of this audit each invented failures, and both mistakes are pinned here
because an audit that cries wolf gets ignored and then the real orphan sits on the share for weeks:

  * requiring a digit in a family name silently dropped `epoch_acc`/`epoch_accdelta` from the disk
    side, then reported their deck patterns as matching nothing;
  * auditing one figure root while the deck reads three reported the other two roots' patterns as
    stale -- 17 healthy patterns called failures.
"""
from __future__ import annotations

from wfield_local import deck_audit as da


def _deck(tmp_path, patterns):
    p = tmp_path / "deck.py"
    p.write_text("\n".join(f'    ("{x}", "t", "note"),' for x in patterns), encoding="utf-8")
    return p


def _figs(d, names):
    d.mkdir(parents=True, exist_ok=True)
    for n in names:
        (d / n).write_bytes(b"")
    return d


def test_a_family_needs_no_digit(tmp_path):
    """`epoch_acc` is a family; the first version dropped it and then blamed its deck pattern."""
    assert da.family_of("epoch_acc_by_position_cue_working.png") == "epoch_acc"
    assert da.family_of("epoch_10cdiagdelta_matrices_x.png") == "epoch_10cdiagdelta"


def test_epoch_families_are_the_token_after_the_prefix(tmp_path):
    """`epoch_10cdiag` is its own family, not a variant of `epoch_10` -- the deck places them apart."""
    assert da.family_of("epoch_10_best_match_acc_cue_working.png") == "epoch_10"
    assert da.family_of("epoch_10cdiag_x_cue_working.png") == "epoch_10cdiag"


def test_a_rendered_family_with_no_pattern_is_reported(tmp_path):
    figs = _figs(tmp_path / "f", ["epoch_5rmo_x_cue_lick.png", "epoch_9_y_cue_working.png"])
    deck = _deck(tmp_path, ["epoch_9_y_*_*.png"])
    r = da.audit([figs], deck)
    assert set(r["unplaced"]) == {"epoch_5rmo"}
    assert r["placed"] == 1


def test_a_pattern_matching_nothing_is_reported(tmp_path):
    figs = _figs(tmp_path / "f", ["epoch_9_y_cue_working.png"])
    deck = _deck(tmp_path, ["epoch_9_y_*_*.png", "gone_forever.png"])
    assert da.audit([figs], deck)["empty_patterns"] == ["gone_forever.png"]


def test_a_pattern_is_not_stale_because_its_root_was_not_audited(tmp_path):
    """The mistake that turned 17 healthy patterns into failures."""
    curated = _figs(tmp_path / "grant", ["epoch_9_y_cue_working.png"])
    other = _figs(tmp_path / "cue", ["section_g_counts_PS92.png"])
    deck = _deck(tmp_path, ["epoch_9_y_*_*.png", "section_g_counts*.png"])
    assert da.audit([curated], deck)["empty_patterns"] == ["section_g_counts*.png"]
    assert da.audit([curated], deck, all_fig_dirs=[curated, other])["empty_patterns"] == []


def test_the_family_side_stays_scoped_to_the_curated_dirs(tmp_path):
    """`all_fig_dirs` must not drag per-session figures into the family tally."""
    curated = _figs(tmp_path / "grant", ["epoch_9_y_cue_working.png"])
    other = _figs(tmp_path / "cue", ["section_g_counts_PS92.png"])
    deck = _deck(tmp_path, ["epoch_9_y_*_*.png", "section_g_counts*.png"])
    r = da.audit([curated], deck, all_fig_dirs=[curated, other])
    assert r["families"] == 1


def test_a_clean_set_reports_nothing(tmp_path):
    figs = _figs(tmp_path / "f", ["epoch_9_y_cue_working.png"])
    deck = _deck(tmp_path, ["epoch_9_y_*_*.png"])
    r = da.audit([figs], deck)
    assert not r["unplaced"] and not r["empty_patterns"]
