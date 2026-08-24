"""HOW MUCH OF EACH POSITION'S POST-STROKE SAMPLE IS PLACED AT THE CUE RATHER THAN AT A LICK?

    python scripts/nolick_fraction.py lick cue precue

In the LICK alignment a lick trial's window starts at its FIRST LICK, while a no-lick trial has no
lick to start from and `position_axes` places it at the CUE (it passes nolick_ref="cue"; the
would_be_lick reference belongs to `position_coding_directions`). So wherever the no-lick fraction is
high, the post-stroke sample is effectively CUE-aligned while the pre-stroke reference -- almost all
lick trials, at a 0.137-0.255 s reaction time -- sits near cue+0.2 s.

WHY THIS MATTERS AND WHERE. The no-lick fraction IS the impairment, so the effect is graded by
severity: measured 2026-08-24, post-stroke far_R is 96-97% no-lick in PS92 and PS94 against 4-19% at
the close positions. **At those positions the lick window is the cue window**, and the two must not
be quoted as independent replications of each other -- which they were, briefly, before this was
checked. Only the pre-cue window is independent there.

The cue and precue alignments are clean by construction (both arms are cue/precue-referenced, so
nothing is misaligned); they are accepted as arguments only so the fractions can be compared.

Counts come from the stored cell `n` -- the outcome-blind arm's total against the lick class's -- so
this reads the same JSON the results are quoted from rather than recomputing anything.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

POS = ["far_R", "far_center", "far_L", "close_R", "close_center", "close_L"]
ROOT = Path("E:/posaxes3")

for align in (sys.argv[1:] or ["lick"]):
    p = ROOT / f"position_axes_{align}_pool2.json"
    if not p.exists():
        print(f"\n### {align}: {p} not present")
        continue
    res = json.loads(p.read_text(encoding="utf-8"))
    print(f"\n### {align}: post-stroke NO-LICK fraction per position (outcome-blind arm)")
    for animal, r in res.items():
        alln, lickn = defaultdict(list), defaultdict(list)
        for key, rec in r["pairs"].items():
            a, b = key.split("|")
            ca = rec["pooled"].get("poststroke_all_working") or {}
            cl = rec["pooled"].get("poststroke_lick") or {}
            if "n" not in ca or "n" not in cl:
                continue
            for pos, i in ((a, 0), (b, 1)):
                alln[pos].append(ca["n"][i])
                lickn[pos].append(cl["n"][i])
        row = f"   {animal}  "
        for pos in POS:
            if not alln.get(pos):
                row += f"{pos}: --   "
                continue
            na, nl = max(alln[pos]), max(lickn[pos])
            row += f"{pos}: {1 - nl / na:.0%} ({nl}/{na})   "
        print(row)
