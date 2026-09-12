"""Splitting the built deck into a narrative deck and a per-session appendix.

Priya, 2026-09-12: "the deck is blowing up", then "go ahead with the split".

The deck is 744 slides and half of it is section G -- per animal x per session detail that section I
now supersedes in pooled, interval-carrying form. Splitting keeps the detail (it is what you want
when a pooled result looks odd) without it owning the file.

These pin the two things a split must not get wrong: a continuation slide has to travel with the
subsection it continues, and a section divider has to appear in BOTH files or neither deck reads as
a deck.
"""
from __future__ import annotations

import pytest

pptx = pytest.importorskip("pptx")

from wfield_local import deck_split as ds  # noqa: E402


def _deck(tmp_path, titles):
    prs = pptx.Presentation()
    blank = prs.slide_layouts[6]
    for t in titles:
        s = prs.slides.add_slide(blank)
        s.shapes.add_textbox(pptx.util.Inches(0.4), pptx.util.Inches(0.2),
                             pptx.util.Inches(9), pptx.util.Inches(1)).text_frame.text = t
    p = tmp_path / "d.pptx"
    prs.save(str(p))
    return p


TITLES = [
    "Spout-position decoding",                 # front matter, untagged
    "G. POST-STROKE -- the frozen model",      # divider
    "G9. PS92 -- ENL window, time course",     # appendix
    "part 2 of 6",                             # continuation of G9, untagged
    "G2. position-matched decoding",           # stays
    "I. POOLED EPOCH FIGURES",                 # divider
    "I4. decoding accuracy by position",       # stays
]


def test_a_continuation_slide_travels_with_its_subsection(tmp_path):
    """"part 2 of 6" carries no tag; leaving it behind would orphan it under the wrong heading."""
    tags = ds.slide_tags(pptx.Presentation(str(_deck(tmp_path, TITLES))))
    assert tags == ["", "G", "G9", "G9", "G2", "I", "I4"]


def test_the_split_moves_only_the_named_subsections(tmp_path):
    r = ds.split(_deck(tmp_path, TITLES), appendix_tags=("G9",))
    assert r["full"] == 7
    assert r["moved"] == 2                      # G9 and its continuation
    assert r["narrative"] == 5


def test_dividers_appear_in_both_files(tmp_path):
    """The appendix needs them to be navigable; the narrative needs them to still read as a deck."""
    r = ds.split(_deck(tmp_path, TITLES), appendix_tags=("G9",))
    nar = ds.slide_tags(pptx.Presentation(str(r["narrative_path"])))
    app = ds.slide_tags(pptx.Presentation(str(r["appendix_path"])))
    assert "G" in nar and "G" in app
    assert "G9" in app and "G9" not in nar
    assert "I4" in nar and "I4" not in app


def test_the_full_deck_is_left_alone(tmp_path):
    """A bad tag list must cost a re-split, never a re-render."""
    p = _deck(tmp_path, TITLES)
    before = p.read_bytes()
    ds.split(p, appendix_tags=("G9",))
    assert p.read_bytes() == before


def test_an_unknown_tag_moves_nothing(tmp_path):
    r = ds.split(_deck(tmp_path, TITLES), appendix_tags=("ZZ",))
    assert r["moved"] == 0 and r["narrative"] == r["full"]


def test_is_divider_only_matches_a_bare_section_letter():
    assert ds.is_divider("G") and not ds.is_divider("G9")
    assert not ds.is_divider("") and not ds.is_divider("G9c")


def test_a_section_without_numbered_subsections_stays_out_of_the_appendix(tmp_path):
    """THE BUG THE SYNTHETIC DECK MISSED, found by running on the real 744-slide file.

    Sections A-F have no numbered subsections, so their CONTENT slides carry the bare section letter
    by carry-forward. Treating every bare-letter tag as a divider swept all 94 of them into the
    appendix -- nothing lost, but the appendix stopped being per-session detail, which is its entire
    purpose. A divider is a slide whose OWN title is the bare letter.
    """
    titles = [
        "A. Per-animal WITHIN-DAY decoding",   # the divider
        "PS92 0818 decoding",                  # content, inherits "A"
        "PS93 0818 decoding",                  # content, inherits "A"
        "G. POST-STROKE",                      # divider
        "G9. PS92 ENL time course",            # appendix
    ]
    r = ds.split(_deck(tmp_path, titles), appendix_tags=("G9",))
    app = ds.slide_titles(pptx.Presentation(str(r["appendix_path"])))
    assert "PS92 0818 decoding" not in app, "section-A content must not ride along"
    assert "A. Per-animal WITHIN-DAY decoding" in app, "its divider still should"
    assert r["appendix"] == 3                  # two dividers + the one G9 slide


def test_divider_indices_uses_the_title_not_the_tag(tmp_path):
    titles = ["A. Section", "inherits A", "G9. thing"]
    assert ds.divider_indices(titles) == {0}
