"""Figs 1.6 (fig1p) and 1.15 (fig10p) only, from `rebuild_figs/fig_pooled.py`'s OWN draw code (the two blocks are
extracted from `fig_pooled.draw` source and executed in that module's namespace, so the bootstrap / seeds / layout are
the page's). fig_pooled.compute() also loads frozen-frame, Layer-2 and attempt inputs these two panels do not use.

    python g4run.py g4_pooled.py --tables <imaging_encoding root holding the three _bins folders> --out <dir>
"""
from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rebuild_figs"))
import fig_pooled as fp  # noqa: E402

BINS = "movement_first_subset-nokin_exec-polar_licks-combined_bins"


def block(src: str, start: str, stop: str) -> str:
    i, j = src.index(start), src.index(stop)
    lines = src[i:j].splitlines()
    ind = len(lines[1]) - len(lines[1].lstrip())
    lines[0] = " " * ind + lines[0]
    return "\n".join(x[ind:] if len(x) >= ind else x.lstrip() for x in lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    fd = {"bins": {f: pd.read_csv(a.tables / f"{BINS}{f}" / "tables" / "region_measures.csv")
                   for f in ("", "_bw0.5", "_bw2")}}
    src = inspect.getsource(fp.draw)
    code = (block(src, "# ---------------------------------------------------------------- 1p",
                  "# ---------------------------------------------------------------- 2c")
            + "\n" + block(src, "# ---------------------------------------------------------------- 10p",
                           "# ---------------------------------------------------------------- 12p"))
    ns = dict(fp.__dict__, fd=fd, OUT=a.out, TXT=[])
    fp._common.declare(fp.trial_set_sentence())
    exec(compile(code, "fig_pooled.draw[1p+10p]", "exec"), ns)
    (a.out / "pooled_numbers_1p_10p.txt").write_text("\n".join(ns["TXT"]))
    print("\n".join(ns["TXT"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
