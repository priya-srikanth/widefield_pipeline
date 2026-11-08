"""Does the within-session neural drift go with worsening behavior? (Priya 11-07: "PS94's subacute sessions ... move
strongly back toward the pre-stroke state from their first to their last trials ... does this correspond with
worsening performance over the session?")

Per session, on the same first-lick trials (state_space_pertrial.py's per-trial table, lick-aligned by default) joined
to the landing table: least-squares slopes on trial order (0 = first, 1 = last trial; within spout, i.e. each spout's
session mean removed) of
  s_x        neural position on the pre (0) -> chronic (1) line (fraction of pre -> chronic distance)
  s_contact  spout contact of the 3-D first lick (proportion)
  s_dside    first-lick lateral landing vs the animal's pre mean at that spout (mm; - = ipsilesional)
  s_fc_dside the same for far-contra trials only (>= 8 trials)
  s_rt       first-lick latency (ms)
then within-animal r of s_x with each behavior slope across post-stroke sessions.
Output: Widefield/state_space_recovery/pertrial{tag}/within_session_behavior_slopes.csv. Exploratory.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import state_space_recovery_trials as R  # noqa: E402
import state_space_sim as S  # noqa: E402

ROOT = Path("M:/MICROSCOPE/Priya/Widefield/state_space_recovery")
DIR = ROOT / f"pertrial{S.tag()}"               # run with WF_SS_ALIGN=lick for the page numbers (Fig 6.12)


def slope(g, c, demean=True):
    g = g.dropna(subset=[c])
    if len(g) < 15:
        return np.nan
    v = g[c] - (g.groupby("spout")[c].transform("mean") if demean else 0)
    return float(np.polyfit(g.order, v, 1)[0])


def main():
    T = pd.read_csv(DIR / "pertrial.csv.gz")
    L = pd.read_csv(R.LANDING)[["label", "row", "contact", "side_ml", "rt_ms"]].rename(columns={"row": "trial"})
    T = T.merge(L, on=["label", "trial"], how="left")
    pre = T[T.epoch == "pre"].groupby(["animal", "spout"]).side_ml.mean()
    T["dside"] = T.side_ml - T.set_index(["animal", "spout"]).index.map(pre).to_numpy()
    rows = []
    for (an, lab, ep), g in T.groupby(["animal", "label", "epoch"]):
        fc = g[(g.spout == "far_contra")].dropna(subset=["dside"])
        rows.append({"animal": an, "label": lab, "epoch": ep, "day": g.day.iloc[0], "s_x": slope(g, "x"),
                     "s_contact": slope(g, "contact"), "s_dside": slope(g, "dside"), "s_rt": slope(g, "rt_ms"),
                     "s_fc_dside": float(np.polyfit(fc.order, fc.dside, 1)[0]) if len(fc) >= 8 else np.nan,
                     "contact": g.contact.mean()})
    out = pd.DataFrame(rows)
    out.to_csv(DIR / "within_session_behavior_slopes.csv", index=False)
    with pd.option_context("display.width", 220, "display.precision", 3):
        print(out.groupby(["animal", "epoch"])[["s_x", "s_contact", "s_dside", "s_fc_dside", "s_rt"]].mean())
        for an, g in out[out.epoch != "pre"].groupby("animal"):
            print(an, {c: round(float(g[["s_x", c]].dropna().corr().iloc[0, 1]), 2)
                       for c in ("s_contact", "s_dside", "s_fc_dside", "s_rt")})


if __name__ == "__main__":
    main()
