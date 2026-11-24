"""S12D mm (Priya 11-23): test the millimeter correction measures with the LIVE S12 rule, in the live S12 BH family.

Live S12 rule (`scripts/c2_recovery_mechanics_stats.py`): `orofacial_stats.session_change_tests` = session change from
each animal's own pre mean, session Mann-Whitney U (post-epoch sessions vs pre sessions) + animal-weighted paired
nested bootstrap (`boot_delta`, 4,000 draws, seed 0, one random stream per call); BH-FDR across ONE family = the
162-cell six-position scalar family + the S12 recovery endpoints; star (**) = q < 0.05 AND the CI excludes 0.

Here: the old S12D endpoint (step_b, 3 cells) leaves the family and the four S12D mm measures join it
(gain_miss_lat, gain_hit_lat, gain_diff_lat, binned_lat; 12 cells). Every other family cell keeps its p (MWU p does
not depend on the bootstrap), so only q moves. Session values: `s12d_alt_sessions.csv` (s12d_miss_correction_vs_pre.py,
12e055c3); pos = "all" (spout terms are inside the per-session fit).

    python s12d_mm_family_stats.py <s12d_alt_sessions.csv> <pooled family csv (162 + recovery)> <out dir>
Writes (refuses to overwrite) s12d_mm_family_stats.csv and s12d_mm_family_pooled.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

METS = ["gain_miss_lat", "gain_hit_lat", "gain_diff_lat", "binned_lat"]


def main(sess: str, family: str, out: str) -> None:
    from wfield_local import orofacial_stats as st
    out = Path(out)
    o1, o2 = out / "s12d_mm_family_stats.csv", out / "s12d_mm_family_pooled.csv"
    for p in (o1, o2):
        if p.exists():
            sys.exit(f"refusing to overwrite {p}")
    S = pd.read_csv(sess).rename(columns={"label": "date"}).assign(pos="all")
    # rule 8: per-animal changes first
    for m in METS:
        g = S.groupby(["animal", "epoch"])[m].mean().unstack("epoch")
        print(m, "\n", (g[["acute", "subacute", "chronic"]].sub(g["pre"], axis=0)).round(3).to_string())
    T = st.session_change_tests(S, METS, n_boot=4000, seed=0)
    F = pd.read_csv(family)
    assert len(F) == 183, len(F)
    base = F[F.metric != "step_b"].copy()
    cols = ["metric", "pos", "epoch", "p", "boot_change", "boot_lo", "boot_hi"]
    P = pd.concat([base[cols].assign(_new=0), T[cols].assign(_new=1)], ignore_index=True)
    Q = st.apply_family_fdr(P)
    P["q_family"], P["agree"] = Q.q_fdr.to_numpy(), Q.agree.to_numpy()
    old = F.set_index(["metric", "pos", "epoch"]).agree_pooled
    moved = P[(P._new == 0)].set_index(["metric", "pos", "epoch"])
    flips = int(((moved.agree == "both") != (old.reindex(moved.index) == "both")).sum())
    print(f"family: {len(P)} cells ({int((P._new == 0).sum())} kept + {int(P._new.sum())} S12D mm); "
          f"star flips among kept cells vs the live pooled family: {flips}")
    T = T.merge(P[P._new == 1][["metric", "epoch", "q_family", "agree"]], on=["metric", "epoch"],
                suffixes=("_metricBH", ""))
    T["star"] = (T.agree == "both").astype(int)
    T["fdr_family"] = "six-position scalar family (162 + S12 recovery, step_b replaced by S12D mm)"
    out.mkdir(parents=True, exist_ok=True)
    T.to_csv(o1, index=False)
    P.drop(columns="_new").assign(s12d_mm=P._new).to_csv(o2, index=False)
    print(T[["metric", "epoch", "n_sessions", "n_pre_sessions", "boot_change", "boot_lo", "boot_hi", "p", "q_family",
             "agree", "star"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:4])
