"""Render Section G's figures from `section_g_{arm}.json`, one panel per POST-STROKE SESSION.

Reads the JSON rather than recomputing, so a figure and the number quoted beside it cannot drift
apart: if they disagree, one of the two files is stale and that is visible rather than silent.

ONE PANEL PER SESSION, NOT PER ANIMAL. The previous renderer keyed every figure by animal and drew a
single "post" column, which was invisible while each animal had one post-stroke night. With two it
averages a day-1 and a day-2 brain, and for PS94 those differ more from each other than pre differs
from post. Panels are therefore labelled PS94_0817, PS94_0818, ... and the arm is in the filename.

BOTH ARMS ARE RENDERED, into separate files. They are not interchangeable and must not be compared
panel-to-panel: the ALL arm scores six positions at a fixed 1/6 chance, while the LICK-ONLY arm uses
each session's preserved positions, so its chance level moves with the animal's behaviour. That is
why `fig_matched` now takes a per-panel chance.

This replaces `make_section_g_figs.py`, which lived only in a scratchpad directory and opened with
`if config.session_phase(a, "0817") != "post": continue` -- so PS93 silently had no behaviour figure
at all, and the deck build reported it as a missing figure with no indication why.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from wfield_local import plot_poststroke as pp
from wfield_local.paths import PathResolver

#: alignment key in the JSON -> (the align string fig_confusion_alltrials expects, a filename tag)
CONFUSION_ALIGNS = (("post-cue", "cue"), ("pre-cue", "precue"))


def _by_session(rec, key):
    """{session label -> rec[key]} for the sessions that have it, in chronological order."""
    return {lab: r[key] for lab, r in sorted(rec.items()) if key in r and r.get(key)}


def render(rec, out, arm):
    """Every figure for one arm. Returns the list of paths written."""
    made = []
    tag = "all" if arm == "all" else "lickonly"
    arm_name = "ALL trials" if arm == "all" else "lick-only"

    counts = _by_session(rec, "counts")
    if counts:
        made.append(pp.fig_behaviour(
            counts, out, name=f"section_g_G1b_counts_{tag}.png",
            suptitle=("BEHAVIOUR FIRST: which positions each session still attempts. A position with "
                      "zero engaged trials has no lick-only decoding number at all, and reading one "
                      "as a neural deficit is how the first pass went wrong. Pre-stroke bars are the "
                      "per-session MEAN; post bars are that single session.")))

    matched = {lab: {k: v for k, v in r.items()
                     if k in ("post-cue", "post-lick", "pre-cue") and v}
               for lab, r in sorted(rec.items())}
    matched = {k: v for k, v in matched.items() if v}
    if matched:
        chance = {lab: rec[lab].get("chance", 1 / 6) for lab in matched}
        made.append(pp.fig_matched(
            matched, out, chance=chance, name=f"section_g_G2_matched_{tag}.png",
            suptitle=(f"Frozen PRE-stroke decoder, post arm = {arm_name}. One panel per POST-STROKE "
                      f"SESSION, against that animal's own pre-stroke band. "
                      + ("Every session is scored over all six positions, so the chance line is 1/6 "
                         "everywhere and the panels ARE comparable."
                         if arm == "all" else
                         "Each session uses ITS OWN preserved positions, so the chance line differs "
                         "between panels and accuracies must NOT be compared across them."))))

    sim = _by_session(rec, "postcue_pattern_similarity")
    if sim:
        # NAME CARRIES THE ARM. Both arms wrote poststroke_G5_similarity.png, so the second render
        # silently overwrote the first and the deck showed one arm labelled as whichever slide
        # happened to point at it.
        made.append(pp.fig_similarity(
            sim, out, name=f"section_g_G5_similarity_{tag}.png",
            suptitle=(f"Per-position correlation between the pre- and post-stroke mean patterns, "
                      f"post arm = {arm_name}. Decoding accuracy alone cannot separate a weakened "
                      f"code from a reorganised one; this can.")))

    for cond, align in CONFUSION_ALIGNS:
        conf = {lab: {align: r["confusion"][cond]}
                for lab, r in sorted(rec.items())
                if r.get("confusion", {}).get(cond)}
        if conf:
            made.append(pp.fig_confusion_alltrials(
                conf, out, align=align,
                name=f"section_g_G3b_confusion_{align}_{tag}.png"))
    return made


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", nargs="+", default=["all", "lickonly"], choices=["all", "lickonly"])
    ap.add_argument("--src", type=Path, default=None)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args(argv)
    src = args.src or Path(PathResolver().root("figures_working"))
    out = args.output or src
    for arm in args.arm:
        p = Path(src) / f"section_g_{arm}.json"
        if not p.exists():
            print(f"  SKIP {arm}: {p} not computed yet "
                  f"(run `python -m wfield_local.poststroke_section_g`)", flush=True)
            continue
        rec = json.load(open(p))
        print(f"\n=== {arm}: {len(rec)} post-stroke sessions", flush=True)
        for q in render(rec, out, arm):
            print(f"  wrote {Path(q).name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
