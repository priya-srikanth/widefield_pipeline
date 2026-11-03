"""Cohort-1 swap-mask re-run, family (d): quiet-period events (v2_quiet_lp definition, k = 4) adopted vs swap-mask variant.

Reads the two `quiet_events_sessions.csv` / `lmm_v2.csv` (same definition, same k) and writes, per animal FIRST (rule 8):
mean session rate per epoch (orofacial / limb, positive / both polarities) adopted vs variant, the epoch / pre fold, then
the LMM epoch terms side by side. Read-only on the adopted folder.

    python scripts/cohort2_qc/swapmask_compare_quiet.py --new <root>/quiet_events/v2_quiet_lp [--out <dir>]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ADOPTED = Path("M:/MICROSCOPE/Priya/Widefield/labcams/cohort1_quiet_spikes/v2_quiet_lp")
EP = ["pre", "acute", "subacute", "chronic"]
COLS = ["orofacial|rate", "orofacial|rate_both", "limb|rate", "limb|rate_both"]


def per_animal(S, tag):
    S = S[S.epoch.isin(EP)]
    g = S.groupby(["animal", "epoch"])[COLS].mean().reset_index()
    g["version"] = tag
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    out = a.out or a.new / "compare_adopted"
    out.mkdir(parents=True, exist_ok=True)
    A = pd.read_csv(ADOPTED / "quiet_events_sessions.csv")
    N = pd.read_csv(a.new / "quiet_events_sessions.csv")
    both = sorted(set(A.label) & set(N.label))
    print(f"sessions adopted {len(A)} new {len(N)} common {len(both)}")
    m = A.set_index("label").loc[both, COLS].join(N.set_index("label").loc[both, COLS], rsuffix="_new")
    for c in COLS:
        r = np.corrcoef(m[c], m[c + "_new"])[0, 1]
        print(f"  session-level {c}: r {r:.3f}, median new/adopted {np.median(m[c + '_new'] / m[c].replace(0, np.nan)):.3f}")
    P = pd.concat([per_animal(A, "adopted"), per_animal(N, "swapmask")])
    P.to_csv(out / "quiet_per_animal_epoch.csv", index=False)
    rows = []
    for c in COLS:
        t = P.pivot_table(index=["animal", "version"], columns="epoch", values=c)[EP]
        for e in EP[1:]:
            t[f"{e}/pre"] = t[e] / t["pre"]
        t = t.round(2)
        print(f"\n=== {c} (events / min, mean of sessions; fold = epoch / pre) ===\n{t.to_string()}")
        for (an, ver), row in t.iterrows():
            rows.append({"measure": c, "animal": an, "version": ver, **row.to_dict()})
    pd.DataFrame(rows).to_csv(out / "quiet_per_animal_fold.csv", index=False)
    LA, LN = pd.read_csv(ADOPTED / "lmm_v2.csv"), pd.read_csv(a.new / "lmm_v2.csv")
    key = ["model", "measure", "column", "term"]
    L = LA[key + ["coef", "se", "q"]].merge(LN[key + ["coef", "se", "q"]], on=key, suffixes=("_adopted", "_swapmask"))
    L = L[L.column.isin(["orofacial", "limb", "session"]) & (L.model == "epoch")]
    L.to_csv(out / "quiet_lmm_compare.csv", index=False)
    show = L[L.term.str.contains("epoch") & L.measure.isin(["rate", "rate_both"])]
    print("\n=== LMM epoch model (log10(rate + 0.2) vs August pre) ===")
    print(show.round(4).to_string(index=False))
    print("wrote", out)


if __name__ == "__main__":
    main()
