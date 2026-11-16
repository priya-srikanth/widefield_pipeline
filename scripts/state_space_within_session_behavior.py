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
11-08 (the within-session-decline arm's open question, branch analysis/within-session-decline-quitting-20261107): the
same neural slopes (+ s_y, the acute-departure axis) joined session by session to THAT arm's behavior slopes
(`Widefield/within_session_decline/tables/session_slopes.csv`, per normalized session "_ord" = the same unit as s_x:
far-contra / all-far landing, latency, licks per trial), within-animal r per animal (rule 8) and pooled within-animal r
with a within-animal circular shift over days (a free shuffle until 11-16), pre and post-stroke separately.
Output: <rebuild root>/state_space_recovery/pertrial{tag}/within_session_behavior_slopes.csv, within_session_join.csv.
Exploratory.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import state_space_recovery_trials as R  # noqa: E402
import state_space_sim as S  # noqa: E402

from wfield_local import analysis_kit as kit  # noqa: E402

ROOT = S.ROOT
DIR = ROOT / f"pertrial{S.tag()}"               # run with WF_SS_ALIGN=lick for the page numbers (Fig 6.12)
OUT = DIR                                       # output folder (a verification run may point it elsewhere)
WSD = S.WSD_SLOPES      # the decline arm's table on the share root (was ROOT.parent: the rebuild root since 11-12)
JOIN = ["fc_dside_ord", "far_dside_ord", "lat_ord", "nlick_ord", "contact_ord"]


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
        rows.append({"animal": an, "label": lab, "epoch": ep, "day": g.day.iloc[0], "s_x": slope(g, "x"), "s_y": slope(g, "y"),
                     "s_contact": slope(g, "contact"), "s_dside": slope(g, "dside"), "s_rt": slope(g, "rt_ms"),
                     "s_fc_dside": float(np.polyfit(fc.order, fc.dside, 1)[0]) if len(fc) >= 8 else np.nan,
                     "contact": g.contact.mean()})
    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "within_session_behavior_slopes.csv", index=False)
    with pd.option_context("display.width", 220, "display.precision", 3):
        print(out.groupby(["animal", "epoch"])[["s_x", "s_contact", "s_dside", "s_fc_dside", "s_rt"]].mean())
        for an, g in out[out.epoch != "pre"].groupby("animal"):
            print(an, {c: round(float(g[["s_x", c]].dropna().corr().iloc[0, 1]), 2)
                       for c in ("s_contact", "s_dside", "s_fc_dside", "s_rt")})
    join(out)          # a missing decline-arm table raises (11-16: it was silently skipped since the 11-12 reroute)


def join(out, n_shuf=2000):
    rng = np.random.default_rng(0)
    W = pd.read_csv(S.require(WSD, "within-session-decline session slopes (scripts/within_session_decline.py analyze)"))
    J = out[["animal", "label", "epoch", "day", "s_x", "s_y"]].merge(W[["label", *JOIN]], on="label")
    if J.empty:
        raise ValueError(f"no session label of {DIR} matched {WSD}")
    rows = []
    for grp, G in (("pre", J[J.epoch == "pre"]), ("post", J[J.epoch != "pre"])):
        for nv in ("s_x", "s_y"):
            for bv in JOIN:
                g = G.dropna(subset=[nv, bv])
                per = {an: float(h[[nv, bv]].corr().iloc[0, 1]) for an, h in g.groupby("animal") if len(h) > 4}
                a = g[nv] - g.groupby("animal")[nv].transform("mean")
                b = (g[bv] - g.groupby("animal")[bv].transform("mean")).to_numpy()
                r0 = float(np.corrcoef(a, b)[0, 1])
                _, p = kit.within_shuffle_p(lambda x, b=b: np.corrcoef(x, b)[0, 1], a.to_numpy(), g.animal.to_numpy(),
                                            rng, n_shuf, order=g.day.to_numpy(), mode="circular")   # Priya 11-16
                rows.append({"sessions": grp, "neural": nv, "behavior": bv, "n": len(g), "r_within": r0,
                             "shuffle_p": p,
                             **{f"r_{an}": round(v, 2) for an, v in per.items()}})
    R_ = pd.DataFrame(rows)
    R_.to_csv(OUT / "within_session_join.csv", index=False)
    with pd.option_context("display.width", 220, "display.precision", 3, "display.max_rows", 100):
        print()
        print("== neural within-session slope vs the decline arm's behavior slopes (within animal) ==")
        print(R_.to_string(index=False))


if __name__ == "__main__":
    main()
