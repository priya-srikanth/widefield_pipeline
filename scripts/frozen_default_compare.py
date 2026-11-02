"""Compare the data CSVs of two renders (docs/status/FROZEN_DEFAULT_20261102.md): the stored-frozen default against the
current (in-memory) renders, cell by cell, exact.

    python scripts/frozen_default_compare.py epoch <current epoch dir> <new epoch dir> [--tag NAME] [--out DIR]
    python scripts/frozen_default_compare.py rest  <current sessions.csv> <new sessions.csv> [--tag NAME] [--out DIR]

epoch: every `epoch_acc*.csv` the two `data/` folders share (bars + CIs, deltas, per-animal `_sessions`, `_meta`).
rest:  the 12.1 per-session table, plus the per-animal and cohort retained fractions derived from it the way
       `rest_frozen_decoder.main` derives them.
Numeric cells are compared EXACTLY (a tolerance would hide a changed model); the max |diff| is reported beside.
Writes `<out>/<tag>_cells.csv` (every differing cell) and `<tag>_summary.csv` (one row per file).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _cmp_frames(a: pd.DataFrame, b: pd.DataFrame, keys):
    """``(n_cells, n_diff, max_abs, diff_rows)`` over the shared columns, rows matched on ``keys``."""
    keys = [k for k in keys if k in a.columns and k in b.columns]
    if keys:
        m = a.merge(b, on=keys, how="outer", suffixes=("_cur", "_new"), indicator=True)
    else:
        if len(a) != len(b):
            return None
        m = a.add_suffix("_cur").join(b.add_suffix("_new"))
        m["_merge"] = "both"
    cols = [c for c in a.columns if c in b.columns and c not in keys]
    rows, n_cells, n_diff, mx = [], 0, 0, 0.0
    for _, r in m.iterrows():
        if r["_merge"] != "both":
            rows.append({**{k: r[k] for k in keys}, "column": "<row>", "cur": r["_merge"] == "left_only",
                         "new": r["_merge"] == "right_only", "abs_diff": np.nan})
            n_diff += 1
            continue
        for c in cols:
            x, y = r[f"{c}_cur"], r[f"{c}_new"]
            n_cells += 1
            xf, yf = pd.to_numeric(pd.Series([x]), errors="coerce")[0], pd.to_numeric(pd.Series([y]), errors="coerce")[0]
            if pd.notna(xf) and pd.notna(yf):
                if xf != yf:
                    n_diff += 1
                    mx = max(mx, abs(float(xf) - float(yf)))
                    rows.append({**{k: r[k] for k in keys}, "column": c, "cur": xf, "new": yf,
                                 "abs_diff": abs(float(xf) - float(yf))})
            elif not (pd.isna(x) and pd.isna(y)) and str(x) != str(y):
                n_diff += 1
                rows.append({**{k: r[k] for k in keys}, "column": c, "cur": x, "new": y, "abs_diff": np.nan})
    return n_cells, n_diff, mx, rows


def compare_epoch(cur: Path, new: Path):
    cd, nd = Path(cur) / "data", Path(new) / "data"
    names = sorted({p.name for p in cd.glob("epoch_acc*.csv")} & {p.name for p in nd.glob("epoch_acc*.csv")})
    only_cur = sorted({p.name for p in cd.glob("epoch_acc*.csv")} - {p.name for p in nd.glob("epoch_acc*.csv")})
    summ, cells = [], []
    for nm in names:
        a, b = pd.read_csv(cd / nm), pd.read_csv(nd / nm)
        # `_sessions` has ONE ROW PER SESSION and no session label, so (epoch, position, animal) is not a key: compare
        # those rows in file order (the renderer writes them in a fixed order), never by a merge that cross-joins them.
        keys = (["kind", "key"] if nm.endswith("_meta.csv") else
                [] if nm.endswith("_sessions.csv") else ["epoch", "position"])
        got = _cmp_frames(a, b, keys)
        if got is None:
            summ.append({"file": nm, "cells": np.nan, "differ": "shape", "max_abs": np.nan})
            continue
        n, d, mx, rows = got
        summ.append({"file": nm, "cells": n, "differ": d, "max_abs": mx})
        cells += [{"file": nm, **r} for r in rows]
    return summ, cells, only_cur


def _rest_tables(ps: pd.DataFrame):
    """Per-animal and cohort retained, as `rest_frozen_decoder.main` prints them."""
    order = [e for e in ("pre", "acute", "subacute", "chronic") if (ps.epoch == e).any()]
    pa = []
    for an, d in ps.groupby("animal"):
        pf = d[d.epoch == "pre"].above_chance_frac.mean()
        for e in order:
            v = d[d.epoch == e]
            if len(v):
                pa.append({"animal": an, "epoch": e, "n": len(v), "acc": v.acc.mean(),
                           "retained": v.above_chance_frac.mean() / pf})
    pre = ps[ps.epoch == "pre"].above_chance_frac.mean()
    co = [{"animal": "ALL", "epoch": e, "n": int((ps.epoch == e).sum()), "acc": ps[ps.epoch == e].acc.mean(),
           "retained": ps[ps.epoch == e].above_chance_frac.mean() / pre} for e in order]
    return pd.DataFrame(pa + co)


def compare_rest(cur: Path, new: Path):
    a, b = pd.read_csv(cur), pd.read_csv(new)
    n, d, mx, rows = _cmp_frames(a, b, ["label"])
    summ = [{"file": Path(new).name, "cells": n, "differ": d, "max_abs": mx}]
    ta, tb = _rest_tables(a), _rest_tables(b)
    n2, d2, mx2, rows2 = _cmp_frames(ta, tb, ["animal", "epoch"])
    summ.append({"file": "derived per-animal + cohort retained", "cells": n2, "differ": d2, "max_abs": mx2})
    return summ, [{"file": "sessions", **r} for r in rows] + [{"file": "retained", **r} for r in rows2], tb


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("what", choices=("epoch", "rest"))
    ap.add_argument("cur", type=Path)
    ap.add_argument("new", type=Path)
    ap.add_argument("--tag", default="compare")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    if a.what == "epoch":
        summ, cells, only_cur = compare_epoch(a.cur, a.new)
        if only_cur:
            print(f"  (not in the new render: {only_cur})")
    else:
        summ, cells, tb = compare_rest(a.cur, a.new)
        print(tb.round(3).to_string(index=False))
    S = pd.DataFrame(summ)
    print(f"[{a.tag}] {a.cur} vs {a.new}")
    print(S.to_string(index=False))
    tot = int(S["differ"].apply(lambda x: x if isinstance(x, (int, np.integer)) else 1).sum())
    print(f"[{a.tag}] files {len(S)}, differing cells {tot} -> {'IDENTICAL' if tot == 0 else 'DIFFERENT'}")
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        S.to_csv(a.out / f"{a.tag}_summary.csv", index=False)
        pd.DataFrame(cells or [{"file": ""}]).to_csv(a.out / f"{a.tag}_cells.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
