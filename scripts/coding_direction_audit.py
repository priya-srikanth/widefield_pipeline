"""DOES ORTHOGONALISING THE CODING DIRECTIONS MOVE THEM TOWARD THE COVARIANCE-AWARE ANSWER?

    PYTHONPATH=$(pwd) python scripts/coding_direction_audit.py E:/cd_audit

THE AUDIT WAS DESIGNED WHEN THE ANALYSIS WAS (2026-08-20/21) and then never run. Difference-of-means
(`dom`) directions are heavily engagement-contaminated -- cos(w, engagement axis) 0.82/0.91/0.71/0.52
across the four animals, landing on a different position in each -- so the no-lick classes are
reported from `dom_orth`, the same directions with the engagement axis projected out. LOGISTIC (`lr`)
directions were clean from the start (|cos| <= 0.07) because they account for covariance, and were
kept as the independent check. This is that check.

THE LOGIC. `lr` is the reference: it reaches a near-uncontaminated direction WITHOUT any projection,
by a different route. So

    if  |dom_orth - lr|  <  |dom - lr|     the projection moves `dom` TOWARD the covariance-aware
                                          answer -- it is a correction, and the reported numbers
                                          are trustworthy
    if  |dom_orth - lr|  >=  |dom - lr|    the projection is not achieving what it claims, and the
                                          no-lick class values in the deck need revisiting

This is a stronger test than "the two agree", because `lr_orth` is also computed: if projecting a
direction that is ALREADY clean changes it materially, the projection is removing position structure
rather than engagement, and that shows up as lr_orth drifting away from lr.

Compares the POOLED per-position per-class means, which are what the deck figures show.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

#: `position_coding_directions` writes ONE coding_direction.json holding only the windows that run
#: asked for, so auditing cue+lick after ENL OVERWRITES the ENL result. Any
#: coding_direction*.json in the directory is read and merged here, and a per-window backup is the
#: way to keep an earlier run (coding_direction_ENL.json).
ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "E:/cd_audit")
CLASSES = ("prestroke_lick", "prestroke_nolick", "poststroke_lick",
           "poststroke_miss_working", "poststroke_stopped")


def pooled_values(method_block):
    """{(class, position): mean} for the cells the deck would draw."""
    out = {}
    for cls, byp in (method_block.get("pooled") or {}).items():
        for pos, cell in (byp or {}).items():
            m = cell.get("mean")
            if m is not None and not cell.get("low_n"):
                out[(cls, pos)] = float(m)
    return out


def compare(a, b):
    """(n, median |a-b|, correlation) over the cells both have."""
    keys = sorted(set(a) & set(b))
    if len(keys) < 3:
        return len(keys), None, None
    va = np.array([a[k] for k in keys])
    vb = np.array([b[k] for k in keys])
    r = float(np.corrcoef(va, vb)[0, 1]) if va.std() > 0 and vb.std() > 0 else float("nan")
    return len(keys), float(np.median(np.abs(va - vb))), r


everything = {}
for jf in sorted(ROOT.glob("coding_direction*.json")):
    for window, res in json.loads(jf.read_text(encoding="utf-8")).items():
        everything.setdefault(window, {}).update({k: v for k, v in res.items() if v})
for window, res in everything.items():
    print(f"\n{'=' * 96}\n### {window} window\n")
    verdicts = defaultdict(list)
    for animal, r in sorted(res.items()):
        if not r:
            continue
        meths = r.get("methods", {})
        if "lr" not in meths:
            print(f"{animal}: no 'lr' method stored -- rerun with --methods dom dom_orth lr lr_orth")
            continue
        vals = {m: pooled_values(meths[m]) for m in meths}
        n1, d1, r1 = compare(vals.get("dom", {}), vals["lr"])
        n2, d2, r2 = compare(vals.get("dom_orth", {}), vals["lr"])
        n3, d3, r3 = compare(vals.get("lr_orth", {}), vals["lr"])
        print(f"{animal}")
        print(f"   dom      vs lr   n={n1:<4} median |diff| "
              f"{'--' if d1 is None else f'{d1:.3f}'}   r {'--' if r1 is None else f'{r1:+.3f}'}")
        print(f"   dom_orth vs lr   n={n2:<4} median |diff| "
              f"{'--' if d2 is None else f'{d2:.3f}'}   r {'--' if r2 is None else f'{r2:+.3f}'}")
        print(f"   lr_orth  vs lr   n={n3:<4} median |diff| "
              f"{'--' if d3 is None else f'{d3:.3f}'}   r {'--' if r3 is None else f'{r3:+.3f}'}"
              "   <- projecting an ALREADY-CLEAN direction; large = the projection removes position")
        if d1 is not None and d2 is not None:
            better = d2 < d1
            verdicts["moves_toward_lr"].append(better)
            print(f"   -> orthogonalising moves dom {'TOWARD' if better else 'AWAY FROM'} lr "
                  f"({d1:.3f} -> {d2:.3f})")
        # where the projection does the most work, by class
        if "dom" in vals and "dom_orth" in vals:
            per_cls = defaultdict(list)
            for k in set(vals["dom"]) & set(vals["dom_orth"]):
                per_cls[k[0]].append(abs(vals["dom"][k] - vals["dom_orth"][k]))
            bits = "  ".join(f"{c}:{np.median(v):.2f}" for c, v in sorted(per_cls.items())
                             if c in CLASSES)
            print(f"   dom -> dom_orth shift by class: {bits}")
    if verdicts["moves_toward_lr"]:
        k = sum(verdicts["moves_toward_lr"])
        n = len(verdicts["moves_toward_lr"])
        print(f"\n   VERDICT [{window}]: orthogonalising moves dom toward lr in {k}/{n} animals")
