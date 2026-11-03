"""Cohort-1 swap-mask re-run, family (c): RESTnl (and REST / RESTW / RAW) epoch-map amplitude ratios, adopted vs variant.

Ratio = RMS(epoch mean map) / RMS(pre mean map) per animal (`scripts/restnl_maps.amplitude_table`, August-only pre), read
from both renders' `data/restnl_amplitude_per_animal_<align>_<variant>.csv`. Per animal first (rule 8), every position;
the far-contra (far_R) acute / chronic cells are printed. Read-only on the adopted folders.

    python scripts/swapmask_compare_restnl.py --new-root <root>/restnl
"""
import argparse
from pathlib import Path

import pandas as pd

G = Path("M:/MICROSCOPE/Priya/Widefield/labcams/grant_figures")
ARMS = {"contact": ("epoch_restnl", ["precue_working", "cue_working", "lick_lick"]),
        "lp": ("LP_licks/epoch_restnl", ["cue_working", "lick_lick"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-root", type=Path, required=True)
    a = ap.parse_args()
    out = a.new_root / "compare_adopted"
    out.mkdir(parents=True, exist_ok=True)
    allrows = []
    for arm, (sub, units) in ARMS.items():
        for u in units:
            fa = G / sub / "data" / f"restnl_amplitude_per_animal_{u}.csv"
            fn = a.new_root / sub / "data" / f"restnl_amplitude_per_animal_{u}.csv"
            if not (fa.exists() and fn.exists()):
                print(f"MISSING {arm} {u}: adopted {fa.exists()} new {fn.exists()}")
                continue
            A, N = pd.read_csv(fa), pd.read_csv(fn)
            k = ["reference", "position", "epoch", "animal"]
            M = A[k + ["n_pre", "n_post", "ratio"]].merge(N[k + ["n_pre", "n_post", "ratio"]], on=k,
                                                         suffixes=("_adopted", "_swapmask"))
            M.insert(0, "unit", u)
            M.insert(0, "arm", arm)
            M["delta"] = M.ratio_swapmask - M.ratio_adopted
            allrows.append(M)
            for ref in ("restnl", "rest"):
                t = M[(M.reference == ref) & (M.position == "far_R")]
                p = t.pivot_table(index="epoch", columns="animal", values=["ratio_adopted", "ratio_swapmask"]).round(3)
                print(f"\n=== {arm} {u} {ref.upper()} far-contra ratio (epoch / pre), adopted vs swapmask ===\n"
                      f"{p.to_string()}")
            big = M[(M.reference == "restnl") & (M.delta.abs() > 0.05)]
            print(f"  {arm} {u} RESTnl cells (all positions) |delta| > .05: {len(big)} of "
                  f"{(M.reference == 'restnl').sum()}; max |delta| {M[M.reference == 'restnl'].delta.abs().max():.3f}")
    if allrows:
        pd.concat(allrows).to_csv(out / "restnl_ratio_compare.csv", index=False)
        print("wrote", out / "restnl_ratio_compare.csv")


if __name__ == "__main__":
    main()
