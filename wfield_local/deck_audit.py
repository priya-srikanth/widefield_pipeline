"""Does every rendered figure family have a deck note, and does every deck note have a figure?

Priya, 2026-09-12: "verify that the deck notes are accurate and complete across all figure
families!!"

TWO FAILURES, OPPOSITE DIRECTIONS, AND BOTH WERE LIVE WHEN THIS WAS WRITTEN.

A family ON DISK WITH NO NOTE is either work that never reached the deck or a figure that should no
longer exist -- and the audit cannot tell which, which is the point of reporting rather than acting.
Both were present on 2026-09-12. `epoch_5rmo`/`epoch_5rmodelta` were the first: the pooled,
training-set-matched contrast, the cleanest single statement of the headline result, rendering
nightly and absent from the deck. `epoch_5rm`/`epoch_5rmdelta` were the second: `_frozen_vs_refit`
gained an `if not matched:` guard because the matched refit accuracy is IDENTICAL to the unmatched
one -- matching replaces the FROZEN model only -- so those slides "promised a second analysis that
did not exist". The guard stopped the render; nothing swept the files, so they sat on the share
looking current. `epoch_15sig_cluster_vs_musall.png` is a third kind again: no code in this
repository's history has ever produced that name.

A PATTERN WITH NO FIGURE is the reverse, and it is what a deck build's own completeness guard
catches at publish time -- but only for the machine it runs on. Reporting it here separates "this
figure was never rendered" from "this figure is missing FROM THIS BOX", which is the distinction
that cost a blocked deck rebuild on 2026-09-11.

THIS TOOL DOES NOT CHECK THAT A NOTE IS TRUE. Nothing can, mechanically; the two wrong notes found
on 2026-09-12 -- `epoch_13` describing a lick-bout anchor three commits after it became a post-cue
anchor, and `epoch_acc` enumerating three trial-class arms when there are five -- were found by
reading them against the code. Completeness is automatable and accuracy is not, so this closes the
half that can be, and says so rather than implying the other half is covered.
"""
from __future__ import annotations

import argparse
import collections
import fnmatch
import pathlib
import re

#: An epoch figure's family is the token after `epoch_`: `epoch_10cdiagdelta_...` is its own family,
#: not a variant of `epoch_10`, because that is how the deck places them -- one pattern per family.
#: A FAMILY NEED NOT CONTAIN A DIGIT. The first version of this required one, which silently dropped
#: `epoch_acc` and `epoch_accdelta` from the disk side and then reported their deck patterns as
#: matching nothing -- an audit that invents two failures is worse than no audit.
EPOCH_PREFIX = "epoch_"

#: Filenames the deck places by literal name rather than by family.
PATTERN_RE = re.compile(r'"([A-Za-z0-9_]*\*?[A-Za-z0-9_*]*\.png)"')

#: Every root the deck reads, filled in by `main`. Used only to decide whether a pattern is stale.
_ALL_ROOTS: list = []


def family_of(name):
    """The family a figure belongs to.

    Epoch figures are grouped by the token after `epoch_`, which is the unit the deck places. Every
    other figure is its own family under its stem: outside the epoch set the deck places by literal
    name or by a single glob, so a coarser grouping would merge families the deck treats separately.
    """
    if name.startswith(EPOCH_PREFIX):
        return "_".join(name.split("_")[:2])
    return pathlib.Path(name).stem


def families_on_disk(dirs):
    """``{family: [filename, ...]}`` over every PNG in ``dirs``."""
    out = collections.defaultdict(list)
    for d in dirs:
        d = pathlib.Path(d)
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.png")):
            out[family_of(f.name)].append(f.name)
    return dict(out)


def placement_patterns(deck_source):
    """Every figure glob the deck places, read out of the builder's source."""
    return sorted(set(PATTERN_RE.findall(pathlib.Path(deck_source).read_text(encoding="utf-8"))))


def audit(dirs, deck_source, *, all_fig_dirs=None):
    """``{"unplaced": {family: n}, "empty_patterns": [pattern, ...], "placed": n}``.

    ``unplaced`` is families whose files exist and which NO pattern matches -- the direction that
    hides both new work and retired figures. ``empty_patterns`` is the reverse.

    THE TWO DIRECTIONS ARE SCOPED DIFFERENTLY, AND MUST BE. ``dirs`` should be the CURATED sets
    (`grant_figures` and its `epoch` subdirectory), where every family is meant to reach the deck
    and an unplaced one is a finding. Pointing it at `cue_analysis` instead reports 3,157 families,
    because the per-session figures are placed by loops over animals and dates and each filename
    becomes its own family -- true, useless, and enough noise to bury the three real orphans.

    ``all_fig_dirs`` is every root the deck reads, and it is used ONLY for ``empty_patterns``: a
    pattern is not stale merely because it points at a directory this argument did not cover.
    Auditing one root and reporting the others as missing is how the first run of this turned 17
    healthy patterns into failures.
    """
    fams = families_on_disk(dirs)
    pats = placement_patterns(deck_source)
    names = [f.name for d in (all_fig_dirs or dirs)
             for f in pathlib.Path(d).glob("*.png")] if all_fig_dirs else [
        n for v in fams.values() for n in v]
    unplaced = {fam: len(files) for fam, files in sorted(fams.items())
                if not any(fnmatch.fnmatch(n, p) for n in files for p in pats)}
    empty = [p for p in pats if not any(fnmatch.fnmatch(n, p) for n in names)]
    return {"unplaced": unplaced, "empty_patterns": empty,
            "placed": len(fams) - len(unplaced), "families": len(fams)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dirs", nargs="*", default=None,
                    help="figure directories (default: grant_figures and grant_figures/epoch)")
    ap.add_argument("--deck", default=None, help="deck builder source file")
    a = ap.parse_args(argv)

    global _ALL_ROOTS
    dirs = a.dirs
    if dirs is None:
        from wfield_local.paths import PathResolver
        labcams = pathlib.Path(PathResolver().root("labcams"))
        # CURATED SETS for the family side; every root the deck reads for the pattern side.
        dirs = [labcams / "grant_figures", labcams / "grant_figures" / "epoch"]
        _ALL_ROOTS = dirs + [labcams / "locanmf_lick_pooled" / "cue_analysis"]
    deck = a.deck or (pathlib.Path(__file__).with_name("locanmf_analysis_deck.py"))

    r = audit(dirs, deck, all_fig_dirs=a.dirs or _ALL_ROOTS)
    print(f"{r['families']} figure families on disk; {r['placed']} placed in the deck")
    if r["unplaced"]:
        print("\nON DISK, NO DECK NOTE -- new work, or a figure the code no longer produces:")
        for fam, n in r["unplaced"].items():
            print(f"  {n:4d} png  {fam}")
    if r["empty_patterns"]:
        print("\nDECK PATTERN MATCHING NOTHING -- never rendered, or missing from THIS machine:")
        for p in r["empty_patterns"]:
            print(f"        {p}")
    if not r["unplaced"] and not r["empty_patterns"]:
        print("\nboth directions clean.")
    print("\nNOTE: this checks COMPLETENESS, never accuracy. A note can be present and wrong.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
