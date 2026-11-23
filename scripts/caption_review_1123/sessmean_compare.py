"""Session medians -> session MEANS (Priya 11-23): cell-by-cell comparison of two six-position scalar family tables.

    python sessmean_compare.py <old c2_scalar_family_aligned.csv> <new ...csv> <out.csv>
Prints per metric the star counts and every star gained or lost; writes the merged table (old / new estimate, CI,
family q, star) for every cell.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

KEY = ["metric", "pos", "epoch"]
COLS = ["boot_change", "boot_lo", "boot_hi", "q_family", "star_aligned", "n_sessions"]


def main(old: str, new: str, out: str) -> None:
    O, N = pd.read_csv(old), pd.read_csv(new)
    M = O[KEY + COLS].merge(N[KEY + COLS], on=KEY, suffixes=("_old", "_new"), how="outer", indicator=True)
    assert (M["_merge"] == "both").all(), M[M["_merge"] != "both"]
    M = M.drop(columns="_merge")
    for c in ("star_aligned_old", "star_aligned_new"):
        M[c] = M[c].astype(bool)
    M["delta_est"] = M.boot_change_new - M.boot_change_old
    M["flip"] = np.where(M.star_aligned_old == M.star_aligned_new, "",
                         np.where(M.star_aligned_new, "GAINED", "LOST"))
    M.to_csv(out, index=False, float_format="%.5g")
    pd.set_option("display.width", 250)
    s = M.groupby("metric").agg(cells=("pos", "size"), stars_old=("star_aligned_old", "sum"),
                                stars_new=("star_aligned_new", "sum"),
                                max_abs_delta=("delta_est", lambda x: float(np.nanmax(np.abs(x)))))
    print(s.to_string())
    print("total stars", int(M.star_aligned_old.sum()), "->", int(M.star_aligned_new.sum()))
    f = M[M.flip != ""]
    print("\nflips:\n", f[KEY + ["boot_change_old", "boot_lo_old", "boot_hi_old", "q_family_old", "boot_change_new",
                               "boot_lo_new", "boot_hi_new", "q_family_new", "flip"]].round(4).to_string(index=False)
          if len(f) else "none")


if __name__ == "__main__":
    main(*sys.argv[1:4])
