"""Old (raw-dF tests) vs new (pre-normalised tests) significant-bin counts, per figure x row x column (2026-10-21,
`beta_maps.NORMALISE_TESTS_BY_PRE`; Priya: "archive the existing results so we don't lose them and can compare them").

Reads every `*_stats.csv` sidecar in the archive (`labcams/grant_figures/epoch/archive_rawdF_tests_20261021/data/`) and
its namesake in the live `labcams/grant_figures/epoch/data/`; writes `results/map_tests_raw_vs_normalised.csv` and
prints the cells whose significant fraction moved by >= 5 points of the tested bins.
Usage: PYTHONPATH=. python scripts/compare_map_tests.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def main() -> int:
    from wfield_local.paths import PathResolver
    from wfield_local.writeguard import assert_writable
    root = Path(PathResolver().root("cohort_out")) / "grant_figures" / "epoch"
    old_dir, new_dir = root / "archive_rawdF_tests_20261021" / "data", root / "data"
    rows = []
    for f in sorted(old_dir.glob("*_stats.csv")):
        g = new_dir / f.name
        if not g.exists():
            print(f"no new version: {f.name}")
            continue
        O, N = pd.read_csv(f), pd.read_csv(g)
        key = ["row", "col"]
        M = O[key + ["sig_bins", "n_bins", "suppressed"]].merge(
            N[key + ["sig_bins", "n_bins", "suppressed", "label"]], on=key, how="outer", suffixes=("_raw", "_norm"))
        M.insert(0, "figure", f.name.removesuffix("_stats.csv"))
        rows.append(M)
    if not rows:
        print("nothing to compare")
        return 1
    D = pd.concat(rows, ignore_index=True)
    for s in ("raw", "norm"):
        D[f"frac_{s}"] = D[f"sig_bins_{s}"] / D[f"n_bins_{s}"]
    D["delta_pts"] = 100 * (D.frac_norm - D.frac_raw)
    out = root / "results"
    assert_writable(out)
    out.mkdir(parents=True, exist_ok=True)
    D.to_csv(out / "map_tests_raw_vs_normalised.csv", index=False, float_format="%.4g")
    pd.set_option("display.width", 230)
    big = D[D.delta_pts.abs() >= 5].sort_values(["figure", "row", "col"])
    print(f"{len(D)} cells; {len(big)} moved by >= 5 points of tested bins:")
    print(big[["figure", "row", "col", "frac_raw", "frac_norm", "delta_pts"]].round(3).to_string(index=False))
    print(f"\n-> {out / 'map_tests_raw_vs_normalised.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
