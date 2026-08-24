"""PER SPOUT POSITION x WINDOW x POST-STROKE BLOCK -- the whole picture in one table.

    python scripts/position_by_position.py precue cue lick        # engagement axis projected out
    ENGAGEMENT=raw python scripts/position_by_position.py precue  # and without it

(from a worktree, prefix PYTHONPATH=$(pwd) -- see this directory's README)

Reads the JSON `wfield_local.position_axes` writes (E:/posaxes3 by default) and reduces it to the
view the results are actually discussed in: for each animal, each spout position, each block of
post-stroke sessions, how far that position's coding has moved from its pre-stroke self.

THE AXIS MEASURE IS PAIRWISE, so "a position" is not a thing it measures directly. A per-position
number here is the MEDIAN over the five pairs that involve that position -- so a position reads low
when its contrasts against the others have moved, whatever the partner. That is the right reduction
for the question "did the code for THIS spout change", and it is not the same as a decoder's
per-class accuracy: it is about the GEOMETRY, not about how well the position can be read out.

CELLS ARE GATED exactly as everywhere else -- both reliabilities >= 0.5, and disattenuated ratios
above 1.0 dropped as at-ceiling. **A BLANK MEANS NOT MEASURABLE, NOT UNCHANGED**, and the difference
between windows is largely a POWER difference: the lick window keeps all five pairs everywhere while
the pre-cue window loses whole blocks. Reading a blank as "no change" would invert the comparison.
The cell count is printed beside every value for that reason.

Read each number against that animal's own null (`scripts/axis_holdout_null.py`), never against 1.0
-- and against the null of the SAME treatment, which is why the JSON stores `prestroke_null` and
`prestroke_null_raw` separately.

ENGAGEMENT=raw reads the cells computed WITHOUT projecting the engagement axis out. That projection
is not the neutral cleanup its name suggests: position axes sit at |cos| 0.61-0.89 to the engagement
axis, so for the worst pairs it discards most of the position axis. Neither treatment is right on
its own -- orth understates position structure, raw admits state contamination -- so a result that
appears in only one of them is a result about the projection.
"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

RAW = os.environ.get("ENGAGEMENT", "orth").lower() == "raw"
PKEY, SKEY = ("pooled_raw", "sessions_raw") if RAW else ("pooled", "sessions")
CLS = "poststroke_all_working"
#: the matched pooled-vs-held-out-two-sessions null, per animal (axis_holdout_null.py, 8/13 excluded)
NULL = {"PS92": 0.79, "PS93": 0.93, "PS94": 0.89, "PS95": 0.84}
POS = ["far_R", "far_center", "far_L", "close_R", "close_center", "close_L"]
ROOT = Path(os.environ.get("POSAXES_DIR", "E:/posaxes3"))


def usable(cell):
    """The disattenuated value if the cell can carry a verdict, else None."""
    if not cell:
        return None
    d, rp, rq = cell.get("disattenuated"), cell.get("r_pre"), cell.get("r_post")
    if d is None or rp is None or rq is None or min(rp, rq) < 0.5 or d > 1.0:
        return None
    return d


for align in (sys.argv[1:] or ["precue", "cue", "lick"]):
    p = ROOT / f"position_axes_{align}_pool2.json"
    if not p.exists():
        print(f"\n### {align}: {p} not present -- run position_axes for this alignment first")
        continue
    res = json.loads(p.read_text(encoding="utf-8"))
    print(f"\n{'=' * 100}\n### {align.upper()} WINDOW   "
          f"[engagement axis {'NOT projected out' if RAW else 'projected out'}]\n")
    for animal, r in res.items():
        if PKEY not in next(iter(r["pairs"].values()), {}):
            print(f"{animal}: no '{PKEY}' in this JSON -- rerun position_axes to store it")
            continue
        blocks = sorted({lab for rec in r["pairs"].values() for lab in rec[SKEY]})
        print(f"{animal}   null {NULL.get(animal, float('nan'))}")
        print(f"   {'position':<14}{'POOLED':>9}" + "".join(f"{b:>13}" for b in blocks))
        for pos in POS:
            vals = defaultdict(list)
            for key, rec in r["pairs"].items():
                if pos not in key.split("|"):
                    continue
                d = usable(rec[PKEY].get(CLS))
                if d is not None:
                    vals["POOLED"].append(d)
                for lab, cs in rec[SKEY].items():
                    d = usable(cs.get(CLS))
                    if d is not None:
                        vals[lab].append(d)
            row = f"   {pos:<14}"
            for scope in ["POOLED"] + blocks:
                v = vals.get(scope, [])
                row += (f"{np.median(v):>+8.2f}({len(v)})" if v else f"{'--':>12} ")
            print(row)
        print()
