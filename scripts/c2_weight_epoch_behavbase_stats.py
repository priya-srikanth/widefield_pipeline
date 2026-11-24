"""Six-position cohort (PS92-95): body weight by epoch with the BEHAVIORAL pre-stroke baseline (Fig 3F; caption review
item 33, CC-B 2026-11-23).

Priya: the weight baseline must be exactly the pre-stroke sessions that define the behavioral baseline. Fig 3F
(c2_weight_epoch_unified_stats.py) normalized to every weight day from 0806 through the stroke day (day <= 0), which
adds the stroke day (0817) for PS92 / PS93 and 0813 for PS93 (no behavior session that day). Here:

- baseline days = the weight days whose date is a pre-stroke session of the six-position behavior tables
  (`position_metrics_fourcam.csv`, epoch == "pre", the sessions of every Fig 4 / S10 / S11 change-from-pre: August
  pre-stroke sessions, config.early_pre (June) excluded, date_policy exclusions applied);
- pct = weight as % of the mean over those days; the other pre-stroke weight days are dropped from panel and test;
- epochs and test unchanged (`c2_weight_epoch_unified_stats.unified`: session MWU, BH over the three epochs, paired
  animals -> days boot_delta, N_BOOT = 4000, '**' = both).

Writes (refuses to overwrite) under --out-dir: c2_weight_epoch_behavbase.csv (tests, old Fig 3F columns joined as old_*),
c2_weight_days_behavbase.csv (animal-days drawn), c2_weight_baseline_days.csv (per animal).
Usage: python scripts/c2_weight_epoch_behavbase_stats.py --out-dir DIR --positions CSV --old CSV [--weights CSV]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import c2_outcome_weight_epoch_stats as cow  # noqa: E402
import c2_weight_epoch_unified_stats as cwu  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--positions", type=Path, required=True)
    ap.add_argument("--old", type=Path, required=True)
    ap.add_argument("--weights", type=Path, default=cow.WEIGHTS_CSV)
    a = ap.parse_args(argv)
    outs = {k: a.out_dir / f"{k}.csv" for k in ("c2_weight_epoch_behavbase", "c2_weight_days_behavbase",
                                                 "c2_weight_baseline_days")}
    clash = [str(p) for p in outs.values() if p.exists()]
    if clash:
        sys.exit("refusing to overwrite " + ", ".join(clash))
    pm = pd.read_csv(a.positions, dtype={"date": str})
    pre = pm[pm.epoch == "pre"].groupby("animal").date.apply(set).to_dict()
    w = cow.weights(a.weights, cow._bounds())
    base = [d in pre.get(an, set()) for an, d in zip(w.animal, w.date)]
    w["is_base"] = base
    rows = []
    for an, g in w.groupby("animal"):
        rows.append({"animal": an, "behavior_pre_sessions": " ".join(sorted(x[4:] for x in pre[an])),
                     "weight_days_used": " ".join(sorted(g[g.is_base].date.str[4:])),
                     "old_pre_days": " ".join(sorted(g[g.day <= 0].date.str[4:])),
                     "dropped": " ".join(sorted(g[(g.day <= 0) & ~g.is_base].date.str[4:])),
                     "behavior_pre_without_weight": " ".join(sorted(x[4:] for x in pre[an] - set(g.date)))})
    bd = pd.DataFrame(rows)
    print(bd.to_string(index=False))
    w = w[(w.day > 0) | w.is_base].copy()
    w["pct"] = 100 * w.pre_weight_g / w.animal.map(w[w.day <= 0].groupby("animal").pre_weight_g.mean())
    st = cwu.unified(w)
    old = pd.read_csv(a.old)
    oc = ["boot_change", "boot_lo", "boot_hi", "q_fdr", "star", "n_pre_sessions"]
    st = st.merge(old[["epoch"] + oc].rename(columns={c: "old_" + c for c in oc}), on="epoch", how="left")
    am = w.groupby(["animal", "epoch"]).pct.mean().reset_index()
    st = st.merge(am.groupby("epoch").pct.mean().rename("animal_mean_pct").reset_index(), on="epoch")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    st.to_csv(outs["c2_weight_epoch_behavbase"], index=False)
    w.drop(columns="is_base").to_csv(outs["c2_weight_days_behavbase"], index=False)
    bd.to_csv(outs["c2_weight_baseline_days"], index=False)
    print(st[["epoch", "n_sessions", "n_pre_sessions", "boot_change", "boot_lo", "boot_hi", "q_fdr", "star",
              "old_boot_change", "old_boot_lo", "old_boot_hi", "old_q_fdr", "old_star", "animal_mean_pct"]]
          .round(4).to_string(index=False))
    print("lowest post-stroke day per animal (% of new baseline):\n"
          + w[w.day > 0].groupby("animal").pct.min().round(1).to_string())
    print("wrote", a.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
