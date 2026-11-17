"""Refactor pin (11-13, state-space cleanup): compare every table a verification run wrote under NEW (WF_SS_ROOT) with
the same relative path under OLD (the production outputs of the previous code). Tables only (csv / csv.gz); figures are
not compared byte-wise (PNG metadata differs) -- their numbers come from these tables.

  python pin_compare.py <NEW root> <OLD root> [--out report.csv]

Per file: rows / columns equal, and per numeric column the max |new - old| (NaN in the same places counts as equal);
text columns must match exactly. "exact" = every value identical; "float" = differences only below 1e-9 (summation
order); anything else is a CHANGE and is listed first."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def compare(fn, fo):
    a, b = pd.read_csv(fn, low_memory=False), pd.read_csv(fo, low_memory=False)
    if a.shape != b.shape or list(a.columns) != list(b.columns):
        return "SHAPE", f"new {a.shape} {list(a.columns)[:6]} vs old {b.shape} {list(b.columns)[:6]}", np.nan
    worst, where = 0.0, ""
    for c in a.columns:
        x, y = a[c], b[c]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            xv, yv = x.to_numpy(float), y.to_numpy(float)
            nan_x, nan_y = np.isnan(xv), np.isnan(yv)
            if (nan_x != nan_y).any():
                return "NAN", f"{c}: NaN pattern differs ({int((nan_x != nan_y).sum())} cells)", np.nan
            d = np.abs(xv - yv)[~nan_x]
            m = float(d.max()) if len(d) else 0.0
            if m > worst:
                worst, where = m, c
        elif not x.astype(str).equals(y.astype(str)):
            k = int((x.astype(str) != y.astype(str)).sum())
            bad = x.astype(str) != y.astype(str)
            return "TEXT", f"{c}: {k} cells differ (first: {x[bad].iloc[0]!r} vs {y[bad].iloc[0]!r})", np.nan
    if worst == 0:
        return "exact", "", 0.0
    return ("float" if worst < 1e-9 else "CHANGE"), where, worst


def main():
    new, old = Path(sys.argv[1]), Path(sys.argv[2])
    out = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else None
    rows = []
    for fn in sorted(p for p in new.rglob("*") if p.suffix in (".csv", ".gz") and "cache" not in p.parts):
        if fn.suffix == ".gz" and not fn.name.endswith(".csv.gz"):
            continue
        rel = fn.relative_to(new)
        fo = old / rel
        if not fo.exists():
            rows.append({"file": str(rel), "status": "NO OLD", "detail": "", "max_abs": np.nan})
            continue
        try:
            st, det, m = compare(fn, fo)
        except Exception as e:                                  # noqa: BLE001 -- report, never hide
            st, det, m = "ERROR", repr(e)[:200], np.nan
        rows.append({"file": str(rel), "status": st, "detail": det, "max_abs": m})
    R = pd.DataFrame(rows)
    order = {"SHAPE": 0, "NAN": 0, "TEXT": 0, "CHANGE": 1, "ERROR": 2, "NO OLD": 3, "float": 4, "exact": 5}
    R = R.sort_values(["status", "file"], key=lambda s: s.map(order) if s.name == "status" else s)
    with pd.option_context("display.width", 250, "display.max_rows", 500, "display.max_colwidth", 120):
        print(R.status.value_counts().to_string())
        print(R[~R.status.isin(["exact"])].to_string(index=False))
    if out:
        R.to_csv(out, index=False)


if __name__ == "__main__":
    main()
