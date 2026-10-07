"""Import the per-animal body-weight sheet exports into `configs/animal_weights.csv`.

Source = the lab weight-tracking Google Sheet (one tab per animal), exported to CSV per tab as
`Animals - <PSxx>.csv`. Each tab has a two-row header: a section row (`... Weight (g) ...`) then a
column row whose `Date` column and `Pre` column (the pre-session body weight in grams) are what we
keep. We store only (animal, date, pre_weight_g); days-relative-to-stroke is DERIVED at read time
from `animals.yaml` (`wfield_local.animal_weights.load`), so the stroke date stays single-sourced.

Re-run whenever the sheet gains new weigh-ins:

    # export each tab to CSV (File -> Download -> CSV) into one folder, then:
    python -m scripts.import_animal_weights --src C:/Users/sabatini/Downloads
    python -m scripts.import_animal_weights --src DIR --out configs/animal_weights.csv
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import pandas as pd

ANIMALS = ("PS92", "PS93", "PS94", "PS95")
REPO = Path(__file__).resolve().parent.parent


def _parse_date(s) -> dt.date | None:
    s = str(s).strip()
    if not s or s.lower() == "nan":
        return None
    for fmt in ("%m/%d/%y", "%m/%d/%Y", "%Y-%m-%d"):
        try:
            return dt.datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def _one(csv_path: Path, animal: str) -> pd.DataFrame:
    # header row is the SECOND line (row 1 is the section header); columns Date + Pre.
    df = pd.read_csv(csv_path, header=1, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    if "Date" not in df.columns or "Pre" not in df.columns:
        raise SystemExit(f"{csv_path.name}: expected 'Date' and 'Pre' columns, got {list(df.columns)}")
    out = pd.DataFrame({
        "date": df["Date"].map(_parse_date),
        "pre_weight_g": pd.to_numeric(df["Pre"], errors="coerce"),
    }).dropna(subset=["date", "pre_weight_g"])
    out.insert(0, "animal", animal)
    out["date"] = out["date"].map(lambda d: d.strftime("%Y%m%d"))
    return out.sort_values("date")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=Path(r"C:/Users/sabatini/Downloads"),
                    help="folder holding `Animals - <PSxx>.csv` exports")
    ap.add_argument("--out", type=Path, default=REPO / "configs" / "animal_weights.csv")
    a = ap.parse_args(argv)

    frames = []
    for an in ANIMALS:
        p = a.src / f"Animals - {an}.csv"
        if not p.exists():
            print(f"!! missing {p} -- skipping {an}")
            continue
        f = _one(p, an)
        print(f"{an}: {len(f)} pre-weight rows  {f['date'].min()}..{f['date'].max()}")
        frames.append(f)
    if not frames:
        raise SystemExit("no per-animal CSVs found")
    allw = pd.concat(frames, ignore_index=True)[["animal", "date", "pre_weight_g"]]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    allw.to_csv(a.out, index=False)
    print(f"\nwrote {a.out}  ({len(allw)} rows, {allw['animal'].nunique()} animals)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
