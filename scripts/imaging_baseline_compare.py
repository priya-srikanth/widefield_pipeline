"""Imaging August-only pre baseline vs the default (all curated pre): cell-by-cell comparison of every epoch-family data
table both renders wrote (Priya 2026-10-18: "is the august-only baseline something we should do for the widefield data
too").

    python scripts/imaging_baseline_compare.py [--late grant_figures/epoch_latepre] [--default grant_figures/epoch]

Each table has epoch x position (or class / pair) rows with value, CI (lo / hi) and a significance mark. Per family:
cells, value change (late - default), and whether the significance mark agrees. Output: <late>/baseline_compare.csv
(+ summary printed and baseline_compare_summary.csv).
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def main() -> None:
    from wfield_local.paths import PathResolver
    lab = Path(PathResolver().root("labcams"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--late", default=str(lab / "grant_figures" / "epoch_latepre"))
    ap.add_argument("--default", default=str(lab / "grant_figures" / "epoch"))
    a = ap.parse_args()
    L, D = Path(a.late) / "data", Path(a.default) / "data"
    rows = []
    for f in sorted(L.glob("*.csv")):
        if f.name.endswith(("_meta.csv", "_sessions.csv")) or not (D / f.name).exists():
            continue
        x, y = pd.read_csv(f), pd.read_csv(D / f.name)
        if "value" not in x or "value" not in y:
            continue
        key = [c for c in x.columns if c not in ("value", "lo", "hi", "mark", "p", "n") and c in y.columns
               and x[c].dtype == object or c == "epoch"]
        key = list(dict.fromkeys(k for k in key if k in y.columns))
        m = x.merge(y, on=key, suffixes=("_late", "_def"), how="inner")
        fam = re.sub(r"_(cue|lick|precue)_(working|lick)$", "", f.stem)
        arm = f.stem[len(fam) + 1:] or "-"
        for _, r in m.iterrows():
            ml, md = str(r.get("mark_late", "") or ""), str(r.get("mark_def", "") or "")
            ml, md = ("" if ml == "nan" else ml), ("" if md == "nan" else md)
            rows.append({"family": fam, "arm": arm, **{k: r[k] for k in key}, "value_def": r.value_def,
                         "value_late": r.value_late, "diff": r.value_late - r.value_def,
                         "sig_def": bool(md.strip()), "sig_late": bool(ml.strip())})
    C = pd.DataFrame(rows)
    out = Path(a.late) / "baseline_compare.csv"
    C.to_csv(out, index=False, float_format="%.4g")
    post = C[C.epoch.astype(str) != "pre"] if "epoch" in C else C
    S = post.groupby("family").agg(cells=("diff", "size"), median_abs_diff=("diff", lambda v: v.abs().median()),
                                   max_abs_diff=("diff", lambda v: v.abs().max()),
                                   sig_both=("sig_def", lambda s: int((s & post.loc[s.index, "sig_late"]).sum())),
                                   sig_default_only=("sig_def", lambda s: int((s & ~post.loc[s.index, "sig_late"]).sum())),
                                   sig_late_only=("sig_late", lambda s: int((s & ~post.loc[s.index, "sig_def"]).sum())))
    S.to_csv(Path(a.late) / "baseline_compare_summary.csv", float_format="%.4g")
    pd.set_option("display.width", 200)
    print(S.round(4).to_string())
    print(f"post-stroke cells: {len(post)}; significance agrees in "
          f"{int((post.sig_def == post.sig_late).sum())} ({(post.sig_def == post.sig_late).mean():.1%})")
    print("->", out)


if __name__ == "__main__":
    main()
