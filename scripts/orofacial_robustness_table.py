"""Robustness of every orofacial cohort cell to the baseline and to PS93 (Priya 2026-10-18).

    python scripts/orofacial_robustness_table.py [--run lp_r4_heatmap_20261013__clean_v36] [--n-boot 2000]

Re-runs the cohort's tests (`orofacial_stats.session_change_tests`: session Mann-Whitney + BH-FDR, animal-weighted
bootstrap) on the saved per-position table under four variants -- baseline all curated pre / August-only pre
(`orofacial_cohort.LATE_PRE_FROM`), x with / without PS93 (pre-existing right orofacial deficit) -- and writes, per
(metric, position, epoch), each variant's change and verdict plus `robust` = significant by BOTH tests in ALL FOUR.
Output: <LP_analysis>/<run>/cohort/tables/robustness/robustness.csv (+ robust_cells.md).
"""
from __future__ import annotations

import argparse

import pandas as pd


def main() -> None:
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local import writeguard
    from wfield_local.orofacial_stats import session_change_tests
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=oc.RUN)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    base = oo.root(PathResolver()) / a.run / "cohort"
    P = pd.read_csv(base / "tables" / "prepost" / "position_metrics.csv", dtype={"date": str})
    out = base / "tables" / "robustness"
    writeguard.assert_writable(out)
    out.mkdir(parents=True, exist_ok=True)
    ms = list(dict.fromkeys([*oc.COHORT_SCALARS, *(m for g in oc.POS_GROUPS.values() for m in g)]))
    late = ~((P.epoch == "pre") & (P.date.str[4:8] < oc.LATE_PRE_FROM))
    variants = {"all": P, "late": P[late], "all_noPS93": P[P.animal != "PS93"],
                "late_noPS93": P[late & (P.animal != "PS93")]}
    key = ["metric", "pos", "epoch"]
    R = None
    for name, d in variants.items():
        st = session_change_tests(d, ms, n_boot=a.n_boot)[key + ["mean_change", "agree", "n_animals"]]
        st = st.rename(columns={"mean_change": f"chg_{name}", "agree": f"agree_{name}", "n_animals": f"nan_{name}"})
        R = st if R is None else R.merge(st, on=key, how="outer")
    R["robust"] = (R[[f"agree_{v}" for v in variants]] == "both").all(axis=1)
    R["robust_both_baselines"] = (R[["agree_all", "agree_late"]] == "both").all(axis=1)
    R["holds_without_PS93"] = (R[["agree_all_noPS93", "agree_late_noPS93"]] == "both").all(axis=1)
    R.to_csv(out / "robustness.csv", index=False, float_format="%.4g")
    lines = ["# Orofacial cohort cells significant by both tests under both baselines (auto-generated)", "",
             "`without PS93` = still significant by both tests under both baselines with PS93 removed.", "",
             "| metric | position | epoch | change (all / Aug-only) | without PS93 |", "|---|---|---|---|---|"]
    for _, r in R[R.robust_both_baselines].sort_values(key).iterrows():
        lines.append(f"| {oc.LABEL.get(r.metric, r.metric)} | {r.pos} | {r.epoch} | {r.chg_all:+.3g} / {r.chg_late:+.3g} | "
                     f"{'yes' if r.holds_without_PS93 else 'NO (' + str(r.agree_all_noPS93) + ' / ' + str(r.agree_late_noPS93) + ')'} |")
    (out / "robust_cells.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    n = int(R.robust_both_baselines.sum())
    print(f"cells robust to both baselines: {n}; of those, holding without PS93: {int((R.robust_both_baselines & R.holds_without_PS93).sum())}")
    print(R[R.robust_both_baselines & ~R.holds_without_PS93][key + ["chg_all", "chg_all_noPS93", "agree_all_noPS93",
                                                                   "agree_late_noPS93"]].round(3).to_string(index=False))
    print("->", out)


if __name__ == "__main__":
    main()
