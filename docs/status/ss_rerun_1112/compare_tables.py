"""State-space Allen re-run (v20u, glue-free, 11-12) vs the glue-in outputs (backup allen_outputs_pre_v20u_20261112):
every CSV in the re-run folders, rows matched on the non-numeric columns. Read-only. Flags: CI (lo/hi) crossing 0,
sig / star columns changing, sign flips of point estimates whose CI excludes 0 in either run."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path("M:/MICROSCOPE/Priya/Widefield/state_space_recovery")
B = R / "_superseded_glue_20261109" / "allen_outputs_pre_v20u_20261112"
out = []
flags = []
for old in sorted(B.rglob("*.csv")):
    rel = old.relative_to(B)
    new = R / rel
    if not new.exists():
        out.append(f"| {rel} | MISSING in re-run | | |")
        continue
    a, b = pd.read_csv(old), pd.read_csv(new)
    if list(a.columns) != list(b.columns):
        out.append(f"| {rel} | columns differ ({len(a.columns)} -> {len(b.columns)}) | | |")
    keys = [c for c in a.columns if c in b.columns and not pd.api.types.is_numeric_dtype(a[c])]
    nums = [c for c in a.columns if c in b.columns and pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c])]
    if keys:
        a2 = a.assign(_o=a.groupby(keys, dropna=False).cumcount()); b2 = b.assign(_o=b.groupby(keys, dropna=False).cumcount())
        m = a2.merge(b2, on=keys + ["_o"], suffixes=("_g", "_n"))
    else:
        n = min(len(a), len(b)); m = a.iloc[:n].add_suffix("_g").join(b.iloc[:n].add_suffix("_n"))
    big = []
    for c in nums:
        if f"{c}_g" not in m:
            continue
        d = (m[f"{c}_n"] - m[f"{c}_g"]).abs()
        rel_ = d / m[f"{c}_g"].abs().clip(lower=1e-9)
        if d.max() > 1e-9:
            big.append(f"{c} max|d| {d.max():.3g} (med rel {np.nanmedian(rel_):.2f})")
    for lo, hi in (("lo", "hi"), ("ci_lo", "ci_hi"), ("delta_lo", "delta_hi")):
        if f"{lo}_g" in m and f"{hi}_g" in m:
            sg = (m[f"{lo}_g"] > 0) | (m[f"{hi}_g"] < 0)
            sn = (m[f"{lo}_n"] > 0) | (m[f"{hi}_n"] < 0)
            for i in np.flatnonzero((sg != sn).values):
                r = m.iloc[i]
                lab = " ".join(str(r[k]) for k in keys)
                est = [c for c in ("delta", "mean", "est", "value", "d") if f"{c}_g" in m]
                e = est[0] if est else None
                flags.append(f"| {rel} | {lab} | " + (f"{r[f'{e}_g']:+.3f} [{r[f'{lo}_g']:+.3f}, {r[f'{hi}_g']:+.3f}]{' *' if sg.iloc[i] else ''}"
                             f" | {r[f'{e}_n']:+.3f} [{r[f'{lo}_n']:+.3f}, {r[f'{hi}_n']:+.3f}]{' *' if sn.iloc[i] else ''} |" if e else
                             f"{'*' if sg.iloc[i] else ''} | {'*' if sn.iloc[i] else ''} |"))
    for sc in ("sig", "star", "stars", "mark"):
        if f"{sc}_g" in m:
            ch = m[f"{sc}_g"].astype(str) != m[f"{sc}_n"].astype(str)
            for i in np.flatnonzero(ch.values):
                r = m.iloc[i]
                flags.append(f"| {rel} | {' '.join(str(r[k]) for k in keys)} [{sc}] | {r[f'{sc}_g']} | {r[f'{sc}_n']} |")
    out.append(f"| {rel} | rows {len(a)} -> {len(b)}, matched {len(m)} | {'; '.join(big[:4])} |")
txt = ["# State-space Allen re-run vs glue-in (11-12)", "", "## Significance changes (CI crossing 0 / sig column)", "",
       "| table | row | glue-in | glue-free |", "|---|---|---|---|", *flags, "", "## All tables", "",
       "| table | rows | largest changes |", "|---|---|---|", *out]
Path(sys.argv[1]).write_text("\n".join(txt) + "\n", encoding="utf-8")
print(f"{len(flags)} significance changes, {len(out)} tables")
