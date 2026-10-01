"""Lightning Pose label file: audit it, and remove all-blank rows before any single-view training.

    python -m wfield_local.lp_labels audit  <lp_dir>/CollectedData.csv
    python -m wfield_local.lp_labels clean  <lp_dir>/CollectedData.csv     # in place; keeps a .bak copy

THE EXPORT ROUTE (unchanged since 2026-09-28): `litpose convert` on DLC's TRAINING copy
(`dlc_train.train_project`, built by `dlc_train.stage`) -- never the labelling project. Since 2026-09-30
`stage` drops all-blank rows (`dlc_train.drop_unlabelled`), so a conversion from the training copy is
already clean. `clean` makes that a property of the LP file itself rather than of where it was converted
from: run it on every new LP label file, and `audit` reports whether it was needed.

WHY IT MATTERS MORE FOR LP THAN IT USED TO. Under `training.uniform_heatmaps_for_nan_keypoints: true` (on since
2026-09-30, the occlusion fix) every blank keypoint is trained toward a flat heatmap = "not visible". An
all-blank row -- a CONTEXT frame nobody labelled (`dlc_context_frames`, `dlc_hard_frames`), or the empty row
napari can leave behind -- would teach "nose, jaw, tongue and spout all hidden" on an ordinary frame. Before
the occlusion fix LP simply ignored blanks, so such a row was harmless; now it is not.

NOT FOR MULTI-VIEW. A multi-view LP label set must KEEP a moment's row in every camera's file (LP pairs views
row by row) and mark an unlabelled camera with a per-keypoint `visible` column = 0 instead. That export is
not written yet (DECISIONS.md 2026-09-30, "TARGET vs CONTEXT frames").
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from wfield_local.dlc_train import drop_unlabelled


def read(csv: Path) -> pd.DataFrame:
    return pd.read_csv(csv, header=[0, 1, 2], index_col=0)


def audit(df: pd.DataFrame) -> dict:
    """Rows, all-blank rows, and per-part filled counts (x present)."""
    x = df.xs("x", axis=1, level=2)
    x.columns = x.columns.get_level_values(-1)
    return {"rows": len(df), "all_blank": int((~df.notna().any(axis=1)).sum()),
            "filled": {p: int(x[p].notna().sum()) for p in dict.fromkeys(x.columns)}}


def clean(csv: Path) -> int:
    """Drop all-blank rows in place (``<csv>.bak`` keeps the original). Returns how many were dropped."""
    df = read(csv)
    kept, n = drop_unlabelled(df)
    if n:
        shutil.copy2(csv, csv.with_suffix(csv.suffix + ".bak"))
        kept.to_csv(csv)
    return n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["audit", "clean"])
    ap.add_argument("csv", type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "audit":
        print(audit(read(a.csv)))
    else:
        n = clean(a.csv)
        print(f"dropped {n} all-blank row(s)" + (f"; original kept as {a.csv.name}.bak" if n else "") + f" -> {audit(read(a.csv))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
