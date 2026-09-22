"""ORTHOGONALISED vs RAW, side by side -- does a result survive the engagement projection or not?

    PYTHONPATH=$(pwd) python scripts/orth_vs_raw.py precue

Projecting the engagement axis out of every position axis was treated as a neutral cleanup. It is
not: measured 2026-08-24, position axes sit at |cos| 0.61-0.89 to that axis, so for the worst pairs
the projection discards most of the position axis and renormalises a small residual. The two
treatments are differently biased --

    ORTH   understates position structure, worst for CROSS-RING pairs, because the no-contact
           population is far-heavy (61-82%) and so `lick - no-lick` is partly a close-vs-far
           direction
    RAW    keeps that structure and admits whatever genuine state difference rides along with it

-- so neither is the answer alone. A result present in BOTH is robust to the choice; one present in
only one is a statement about the projection, and must be reported that way.

Each value is judged against the null of ITS OWN treatment (`prestroke_null` vs
`prestroke_null_raw`), because a raw cosine read against an orthogonalised null compares two
different measurements.

BY PAIR TYPE as well as by position, since the predicted damage is specifically cross-ring
(distance and diagonal pairs), and a uniform orth-vs-raw gap would mean something else is going on.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

CLS = "poststroke_all_working"
POS = ["far_R", "far_center", "far_L", "close_R", "close_center", "close_L"]
ROOT = Path("E:/posaxes4")


def usable(cell):
    if not cell:
        return None
    d, rp, rq = cell.get("disattenuated"), cell.get("r_pre"), cell.get("r_post")
    if d is None or rp is None or rq is None or min(rp, rq) < 0.5 or d > 1.0:
        return None
    return d


def null_of(r, key):
    """The stored matched null for one treatment (`prestroke_null[_raw].null_median`)."""
    n = r.get(key) or {}
    v = n.get("null_median")
    return float(v) if isinstance(v, (int, float)) else None


for align in (sys.argv[1:] or ["precue"]):
    p = ROOT / f"position_axes_{align}_pool2.json"
    if not p.exists():
        print(f"### {align}: {p} not present")
        continue
    res = json.loads(p.read_text(encoding="utf-8"))
    print(f"\n{'=' * 96}\n### {align.upper()}   orthogonalised vs raw\n")
    for animal, r in res.items():
        n_o, n_r = null_of(r, "prestroke_null"), null_of(r, "prestroke_null_raw")
        print(f"{animal}   null orth {n_o if n_o is None else round(n_o, 2)}   "
              f"raw {n_r if n_r is None else round(n_r, 2)}")
        by_pos = defaultdict(lambda: ([], []))
        by_type = defaultdict(lambda: ([], []))
        for key, rec in r["pairs"].items():
            a, b = key.split("|")
            do, dr = usable(rec["pooled"].get(CLS)), usable(rec.get("pooled_raw", {}).get(CLS))
            for pos in (a, b):
                if do is not None:
                    by_pos[pos][0].append(do)
                if dr is not None:
                    by_pos[pos][1].append(dr)
            if do is not None:
                by_type[rec["type"]][0].append(do)
            if dr is not None:
                by_type[rec["type"]][1].append(dr)
        print(f"   {'position':<14}{'orth':>10}{'raw':>10}{'delta':>9}")
        for pos in POS:
            o, w = by_pos.get(pos, ([], []))
            so = f"{np.median(o):+.2f}({len(o)})" if o else "--"
            sw = f"{np.median(w):+.2f}({len(w)})" if w else "--"
            dl = f"{np.median(w) - np.median(o):+.2f}" if o and w else ""
            print(f"   {pos:<14}{so:>10}{sw:>10}{dl:>9}")
        print(f"   {'pair type':<14}{'orth':>10}{'raw':>10}{'delta':>9}")
        for t in ("lateral", "lateral-centre", "distance", "diagonal"):
            o, w = by_type.get(t, ([], []))
            so = f"{np.median(o):+.2f}({len(o)})" if o else "--"
            sw = f"{np.median(w):+.2f}({len(w)})" if w else "--"
            dl = f"{np.median(w) - np.median(o):+.2f}" if o and w else ""
            print(f"   {t:<14}{so:>10}{sw:>10}{dl:>9}")
        print()
