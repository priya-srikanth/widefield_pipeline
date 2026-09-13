"""Split the built analysis deck into a NARRATIVE deck and a PER-SESSION APPENDIX.

Priya, 2026-09-12: "the deck is blowing up - can you suggest figures we might cut from it?", then
"go ahead with the split".

THE MEASUREMENT THAT MOTIVATED IT. The deck is 744 slides, and the intuition about where the bulk
sits was wrong in both directions -- mine and the obvious one. Counting the static filename patterns
in `locanmf_analysis_deck` gives 393, barely half, because the per-session sections build their
slides in loops. Counting the built file by section:

    G  372   post-stroke, per animal x per session        <- HALF THE DECK
    I  178   pooled epoch figures
    H   99   grant summary set
    D   51   cross-session frozen decoders / encoders
    A-C, E, F  44

and inside G, two subsections carry a quarter of the entire deck on their own: `G9` (113, per-animal
ENL time course) and `G9c` (72, per-session cross-position matrices).

WHY SPLIT RATHER THAN DELETE. Every subsection moved here is per-session detail that section I now
SUPERSEDES -- I is the pooled, epoch-stratified form of the same quantities, carrying bootstrap
intervals and corrected marks that a per-session panel cannot. But per-session detail is exactly
what is wanted when a pooled result looks odd and the question becomes "which session did that?".
Deleting it would answer the navigability problem by destroying the debugging one. So: two files,
nothing lost, and the main deck becomes something that can be opened.

WHY POST-HOC AND NOT A FLAG IN THE BUILDER. `build_analysis_deck` is one 3,000-line function whose
subsections are inline blocks; gating them would mean re-indenting hundreds of lines of a file two
windows are editing, to get a result this achieves by reading the slide titles the builder already
writes. The full deck is still produced and still published -- this derives two views of it, so a
mistake here costs a re-split and never a re-render.

THE TAG COMES FROM THE TITLE, WITH CARRY-FORWARD. Slides are titled "G9c. PS92 - ENL window, ..." and
continuation slides ("part 2 of 10") do not repeat the tag, so an untagged slide belongs to the last
tagged one. A section DIVIDER titled "G. POST-STROKE ..." tags as "G" and is kept in both files: the
appendix needs its dividers to be navigable, and the narrative needs them to still read as a deck.
"""
from __future__ import annotations

import argparse
import pathlib
import re

from pptx import Presentation

RID = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"

#: Subsections that move to the appendix. Every one is PER-SESSION or PER-ANIMAL detail whose
#: pooled, interval-carrying form lives in section I.
#:
#: Sizes at the 2026-09-12 build, and the reason each is here:
#:   G9  113  per-animal ENL time course, orthogonalised direction -- I's epoch bars pool it
#:   G9c  72  cross-position matrix PER SESSION -- I's matrix families are the pooled form
#:   G3   48  crossed confusion per animal per date -- I's 5c/5cr is the pooled epoch version
#:   G8f  40  pattern convergence / midline test, a TEN-PART series: a document, not slides
#:   G8d  24  pre/post maps on a common colour scale, three parts per animal
#:   G1b  12  "which positions still have trials at all", twelve parts
#:   G9e  12  early vs late rewarded trials, per animal
#:   G7c  10  small-lesion comparison, per session
#:   G9b   6  per-animal ENL companion
#: G4, G4b and G6 ADDED 2026-09-13 (Priya: "2 - push to appendix"), and for a reason distinct from
#: every tag above them. The others moved because section I SUPERSEDES them -- a pooled,
#: epoch-stratified form of the same quantity exists. These three move because they are
#: ILLEGIBLE ON A SLIDE, and measurably so rather than as an impression:
#:
#:     poststroke_G4_identity            2790 x 10979 px   aspect 1:3.94
#:     section_g_fits_engaged_<align>    3000 x  7350 px   aspect 1:2.45
#:     poststroke_G6_nolick_readout      3510 x  9360 px   aspect 1:2.67
#:
#: Fitted to a 16:9 slide those scale to 14-23%, which renders their fontsize-6.5 annotations at
#: ~1-1.5 pt. NO FONT CHANGE FIXES THAT; the cause is 48 per-session panels in one file, and the
#: only real remedies are ~6 panels per slide (≈8 slides per family), per-animal aggregation, or
#: the appendix. The appendix is where a reader can ZOOM, which is exactly what a 48-panel
#: per-session grid needs and what a slide cannot give -- and it costs no narrative slides, in a
#: deck that lost 46 of them the same day.
#:
#: THE FIGURES ARE NOT CUT. They are still rendered nightly and still placed; they move file.
APPENDIX_TAGS = ("G9", "G9b", "G9c", "G9e", "G3", "G8d", "G8f", "G1b", "G7c",
                 "G4", "G4b", "G6")


def slide_titles(prs):
    """First text run of each slide, which is the title the builder writes."""
    out = []
    for s in prs.slides:
        txt = ""
        for shape in s.shapes:
            if shape.has_text_frame and shape.text_frame.text.strip():
                txt = shape.text_frame.text.strip().splitlines()[0]
                break
        out.append(txt)
    return out


def slide_tags(prs):
    """Per-slide subsection tag ("G9c"), carried forward across untagged continuation slides."""
    tags, cur = [], ""
    for t in slide_titles(prs):
        m = re.match(r"^([A-Z]\d*[a-z]?)\.\s", t)
        if m:
            cur = m.group(1)
        tags.append(cur)
    return tags


def _drop(prs, idxs):
    """Remove slides by index, dropping their relationship so the media goes with them."""
    lst = prs.slides._sldIdLst
    entries = list(lst)
    for i in sorted(set(idxs), reverse=True):
        rid = entries[i].get(RID)
        try:
            prs.part.drop_rel(rid)
        except KeyError:                                               # already gone
            pass
        lst.remove(entries[i])


#: A divider slide's TITLE is a bare section letter: "G. POST-STROKE -- ...".
DIVIDER_RE = re.compile(r"^[A-Z]\.\s")


def divider_indices(titles):
    """Indices of the section-divider slides.

    NOT "every slide whose tag is one letter" -- that was the first rule and it was wrong on the
    real deck. Sections A-F have no numbered subsections, so their CONTENT slides carry the bare
    section letter by carry-forward and the rule swept all 94 of them into the appendix. A divider
    is a slide whose OWN TITLE is the bare letter; the slides that merely inherit it are content.
    """
    return {i for i, t in enumerate(titles) if DIVIDER_RE.match(t or "")}


def is_divider(tag):
    """True for a bare section-letter TAG. Kept for callers reasoning about tags rather than slides.

    Prefer `divider_indices`: a tag alone cannot tell a divider from a slide that inherited it.
    """
    return bool(tag) and len(tag) == 1 and tag.isalpha()


def split(full_path, narrative_out=None, appendix_out=None, appendix_tags=APPENDIX_TAGS):
    """Write the narrative and appendix decks beside ``full_path``. Returns a summary dict.

    Neither output is the input: the full deck is left exactly as built, so a bad tag list costs a
    re-split rather than a rebuild.
    """
    full_path = pathlib.Path(full_path)
    appendix_tags = set(appendix_tags)
    narrative_out = pathlib.Path(narrative_out or full_path.with_name(
        full_path.stem + "_narrative.pptx"))
    appendix_out = pathlib.Path(appendix_out or full_path.with_name(
        full_path.stem + "_per_session_appendix.pptx"))

    _prs = Presentation(str(full_path))
    titles = slide_titles(_prs)
    tags = slide_tags(_prs)
    to_appendix = [i for i, t in enumerate(tags) if t in appendix_tags]
    keep_in_appendix = set(to_appendix) | divider_indices(titles)

    prs = Presentation(str(full_path))
    _drop(prs, to_appendix)
    prs.save(str(narrative_out))

    prs = Presentation(str(full_path))
    _drop(prs, [i for i in range(len(tags)) if i not in keep_in_appendix])
    prs.save(str(appendix_out))

    return {"full": len(tags), "narrative": len(tags) - len(to_appendix),
            "appendix": len(keep_in_appendix), "moved": len(to_appendix),
            "narrative_path": narrative_out, "appendix_path": appendix_out,
            "narrative_mb": narrative_out.stat().st_size / 1e6,
            "appendix_mb": appendix_out.stat().st_size / 1e6}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("deck", nargs="?", help="built pptx (default: the analysis summary on labcams)")
    ap.add_argument("--narrative", default=None)
    ap.add_argument("--appendix", default=None)
    ap.add_argument("--tags", nargs="*", default=None,
                    help=f"subsections to move (default: {' '.join(APPENDIX_TAGS)})")
    ap.add_argument("--report", action="store_true",
                    help="print the per-subsection slide tally and exit without writing")
    a = ap.parse_args(argv)

    deck = a.deck
    if deck is None:
        from wfield_local.paths import PathResolver
        deck = pathlib.Path(PathResolver().root("labcams")) / "spout_position_analysis_summary.pptx"

    if a.report:
        import collections
        tags = slide_tags(Presentation(str(deck)))
        for t, n in collections.Counter(tags).most_common():
            print(f"  {n:4d}  {t or '(untagged)'}")
        return 0

    r = split(deck, a.narrative, a.appendix, a.tags or APPENDIX_TAGS)
    print(f"full {r['full']} slides -> narrative {r['narrative']} ({r['narrative_mb']:.0f} MB), "
          f"appendix {r['appendix']} ({r['appendix_mb']:.0f} MB); moved {r['moved']}")
    print(f"  {r['narrative_path']}")
    print(f"  {r['appendix_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
