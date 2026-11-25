"""FINAL pooled six-position family (lead 11-23): FINAL 180 scalar cells + 18 live S12 recovery cells + 12 S12D mm cells.

One BH over all 210 cells (`orofacial_stats.apply_family_fdr`, MWU p only); star = q < 0.05 AND bootstrap CI excludes
0. Every cell keeps the p and CI of its producing `session_change_tests` call (4,000 draws, seed 0):
  scalar  c2_scalar_family_FINAL_ccB1123_boot4000_cam3on.csv (180)
  S12     recovery_mechanics_final_launch2v_boot4000_cam3on/c2_recovery_mechanics_v2_stats.csv, family members only
          (not the controls csm_false / cued_mean; step_b replaced by S12D mm) (18)
  S12D mm sixpos/data_item2b_alt/s12d_mm_family_stats.csv (s12d_mm_family_stats.py) (12)
Prints star changes vs each source's own star.

    python final_family_with_s12.py <scalar FINAL csv> <S12 stats csv> <S12D mm stats csv> <out csv>
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

CONTROLS = ("csm_false", "cued_mean")
COLS = ["metric", "pos", "epoch", "n_sessions", "n_pre_sessions", "p", "boot_change", "boot_lo", "boot_hi"]


def main(scalar: str, s12: str, mm: str, out: str) -> None:
    from wfield_local import orofacial_stats as st
    out = Path(out)
    if out.exists():
        sys.exit(f"refusing to overwrite {out}")
    A = pd.read_csv(scalar)
    B = pd.read_csv(s12)
    B = B[~B.metric.isin(CONTROLS) & (B.metric != "step_b")]
    C = pd.read_csv(mm)
    assert (len(A), len(B), len(C)) == (180, 18, 12), (len(A), len(B), len(C))
    P = pd.concat([A[COLS + ["panel"]].assign(block="scalar", star_source=A.star_aligned.astype(int)),
                   B[COLS].assign(block="S12", panel="S12 A-C, E", star_source=B.star.astype(int)),
                   C[COLS].assign(block="S12D_mm", panel="S12D mm", star_source=C.star.astype(int))],
                  ignore_index=True)
    Q = st.apply_family_fdr(P)
    P["q_family"], P["agree"] = Q.q_fdr.to_numpy(), Q.agree.to_numpy()
    P["star"] = (P.agree == "both").astype(int)
    P["fdr_family"] = "FINAL six-position family: 180 scalar + 18 S12 recovery + 12 S12D mm (one BH)"
    P.to_csv(out, index=False)
    for blk, g in P.groupby("block", sort=False):
        ch = g[g.star != g.star_source]
        print(f"{blk}: {len(g)} cells, stars {int(g.star_source.sum())} -> {int(g.star.sum())}, flips {len(ch)}")
        if len(ch):
            print(ch[["metric", "pos", "epoch", "p", "q_family", "boot_lo", "boot_hi", "star_source", "star"]]
                  .to_string(index=False))
    print(P[P.block != "scalar"][["block", "metric", "epoch", "boot_change", "boot_lo", "boot_hi", "q_family",
                                  "star"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:5])
