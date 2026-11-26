"""Fig 17.1 PROPER (fig4p_decomp_pooled) from `rebuild_figs/fig_pooled.py`'s own 4p block, on a frozen-frame
decomposition table (`qc/lp_encoder_decompose_frozen_encoder_<tag>_joint_lick_by_animal.csv`) given explicitly.

    python g4run.py g4_pooled_4p.py --table <by_animal.csv> --out <dir>
"""
from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rebuild_figs"))
import fig_pooled as fp  # noqa: E402
from g4_pooled import block  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    fd = {"decompose": pd.read_csv(a.table)}
    src = inspect.getsource(fp.draw)
    code = block(src, "# ---------------------------------------------------------------- 4p",
                 "# ---------------------------------------------------------------- 5p")
    ns = dict(fp.__dict__, fd=fd, OUT=a.out, TXT=[])
    fp._common.declare(fp.trial_set_sentence())
    exec(compile(code, "fig_pooled.draw[4p]", "exec"), ns)
    (a.out / "pooled_numbers_4p.txt").write_text("\n".join(ns["TXT"]))
    print("\n".join(ns["TXT"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
