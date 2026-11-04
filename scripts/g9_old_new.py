"""G9 OLD (DAQ classes) -> NEW (camera classes) report, per animal (Priya 11-04 guard 5; run BEFORE any figure is switched).

Reads coding_direction records (`position_coding_directions.run_animal` JSON, the file the deck / grant figures read)
and prints, per animal x method x spout: the DAQ classes (poststroke_lick / miss_working / stopped) next to the camera
classes (pre / post x contact / incomplete / jaw_only / none / after_end, plus pre working), with n and the pooled CD
score; thin cells (< 10) marked. Writes a CSV next to --out. Read-only on its inputs.

Usage: python scripts/g9_old_new.py <coding_direction*.json> [...] [--out g9_old_new.csv] [--method dom]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from wfield_local import position_coding_directions as pcd


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("json", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--method", default=None, help="one CD method only (default: all)")
    a = ap.parse_args(argv)
    res = {}
    for p in a.json:
        D = json.loads(p.read_text())
        # either {animal: record} or a single record
        for an, r in (D.items() if "methods" not in D else [(D.get("animal", p.stem), D)]):
            res[an] = r
    T = pd.DataFrame(pcd.g9_old_new(res))
    if T.empty:
        print("no records")
        return 1
    if a.method:
        T = T[T.method == a.method]
    with pd.option_context("display.max_rows", 500, "display.width", 200):
        print(T.pivot_table(index=["animal", "method", "spout"], columns=["scheme", "class"], values="n",
                            aggfunc="sum").fillna(0).astype(int).to_string())
        print(T.pivot_table(index=["animal", "method", "spout"], columns=["scheme", "class"], values="mean").round(3)
              .to_string())
    if a.out:
        T.to_csv(a.out, index=False)
        print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
