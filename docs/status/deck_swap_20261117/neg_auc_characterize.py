"""Step 1-2 tables of the ITI negative-AUC diagnostic: who is undefined, and why (reads features.pkl from
neg_auc_analyze). usage: python neg_auc_characterize.py <features.pkl>"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
F = pd.read_pickle(sys.argv[1])
POS = ["close_ipsi", "close_mid", "close_contra", "far_ipsi", "far_mid", "far_contra"]
F = F[F.spout.isin([*POS, "pooled"])]


def side(r, w):
    c, i = r[f"c_{w}"] <= 0, r[f"i_{w}"] <= 0
    return "both" if c and i else "contra" if c else "ipsi" if i else "-"


for w in ("0-1", "0-0.5"):
    F[f"und_{w}"] = (F[f"c_{w}"] <= 0) | (F[f"i_{w}"] <= 0)
    F[f"side_{w}"] = [side(r, w) for _, r in F.iterrows()]

print("### undefined share by mode x set x window (per-spout + pooled rows)")
print(F.groupby(["mode", "set"])[["und_0-1", "und_0-0.5"]].agg(["mean", "sum", "size"]).round(3).to_string())

for w in ("0-1", "0-0.5"):
    U = F[F[f"und_{w}"]]
    print(f"\n### window {w}: side of the non-positive AUC")
    print(pd.crosstab([U["mode"], U.set], U[f"side_{w}"]).to_string())
    print(f"\n### window {w}: by pair (share undefined), mode x set")
    print(F.groupby(["mode", "set", "pair"])[f"und_{w}"].mean().unstack("pair").round(3).to_string())

w = "0-1"
I = F[F.set == "iti"]
for mode in ("all", "dock"):
    J = I[I["mode"] == mode]
    print(f"\n### ITI {mode} 0-1: share undefined by animal x epoch (all pairs)")
    print(J.groupby(["animal", "epoch"])["und_0-1"].agg(["mean", "sum", "size"]).round(3).unstack("epoch").to_string())
    print(f"\n### ITI {mode} 0-1: share undefined by spout")
    print(J.groupby("spout")["und_0-1"].agg(["mean", "sum", "size"]).round(3).T.to_string())
    print(f"\n### ITI {mode} 0-1: lick count n, defined vs undefined (quantiles)")
    print(J.groupby("und_0-1").n.describe().round(1).to_string())
    print(f"\n### ITI {mode} 0-1 tjM1: undefined by animal x epoch")
    print(J[J.pair == "tjM1"].groupby(["animal", "epoch"])["und_0-1"].agg(["sum", "size"]).unstack("epoch").to_string())

# --- per offending side: near zero vs clearly negative, and cause features
print("\n\n### CAUSE FEATURES (offending sides of undefined rows, vs the same side of defined rows)")
recs = []
for _, r in F.iterrows():
    for s in ("c", "i"):
        for w in ("0-1", "0-0.5"):
            recs.append({"mode": r["mode"], "set": r.set, "pair": r.pair, "spout": r.spout, "animal": r.animal,
                         "epoch": r.epoch, "label": r.label, "n": r.n, "side": s, "w": w,
                         "auc": r[f"{s}_{w}"], "se": r[f"{s}_sem_{w}"], "auc_early": r[f"{s}_0-0.5"],
                         "auc_late": r[f"{s}_0.5-1"], "early": r[f"{s}_early"], "slope": r[f"{s}_slope"],
                         "drift": r[f"{s}_drift_{w}"], "peak": r[f"{s}_peak"], "rawbase": r[f"{s}_rawbase"],
                         "prelick": r[f"{s}_prelick"], "auc_bl": r[f"{s}_bl_{w}"], "auc_be": r[f"{s}_be_{w}"],
                         "since_stop": r.get("since_stop_med", np.nan), "since_dock": r.get("since_dock_med", np.nan),
                         "to_trial": r.get("to_trial_med", np.nan), "dock_in_base": r.get("frac_dock_in_base", np.nan),
                         "cue_in_win": r.get("frac_cue_in_win", np.nan), "gap": r.get("gap_med", np.nan)})
S = pd.DataFrame(recs)
S["neg"] = S.auc <= 0
S["z"] = S["auc"] / S["se"]
S["near0"] = S.z > -2                       # within 2 SEM of zero (per-lick AUC SEM)
S["undershoot"] = (S.auc_early > 0) & (S.auc_late < 0)
S["trend"] = (S.early > 0) & (S.slope < 0)   # signal was higher 1 s before the baseline and falling into it
S["detr_pos"] = (S.auc - S.drift) > 0        # removing the extrapolated pre-trend makes the AUC positive
S["bl_pos"] = S.auc_bl > 0
S["be_pos"] = S.auc_be > 0
S.to_pickle(sys.argv[1].replace("features.pkl", "sides.pkl"))
cols = ["near0", "undershoot", "trend", "detr_pos", "bl_pos", "be_pos"]
for (mode, st, w), g in S[S.set.isin(["iti", "cue"])].groupby(["mode", "set", "w"]):
    if mode == "dock" and st != "iti":
        continue
    neg, pos_ = g[g.neg], g[~g.neg]
    if not len(neg):
        continue
    print(f"\n-- {mode} {st} {w}: {len(neg)} non-positive sides (of {len(g)})")
    print("   share in non-positive sides: " + ", ".join(f"{c} {neg[c].mean():.2f}" for c in cols))
    print("   share in positive sides:     " + ", ".join(f"{c} {pos_[c].mean():.2f}" for c in cols
                                                       if c not in ("detr_pos", "bl_pos", "be_pos")))
    q = ["auc", "z", "peak", "early", "slope", "rawbase", "prelick", "auc_early", "auc_late", "n"]
    if st == "iti":
        q += ["since_stop", "since_dock", "to_trial", "dock_in_base", "cue_in_win", "gap"]
    print(pd.DataFrame({"non-positive": neg[q].median(), "positive": pos_[q].median()}).T.round(3).to_string())
    # hierarchical attribution
    a = np.select([neg.near0 & ~neg.undershoot & ~neg.trend, neg.undershoot & ~neg.trend, neg.trend & ~neg.undershoot,
                   neg.trend & neg.undershoot], ["(a) noise only: |z|<2, no undershoot/trend", "(c) undershoot",
                                                 "(b/d) falling pre-trend", "trend + undershoot"], "other (clearly neg, no pattern)")
    print("   attribution: " + pd.Series(a).value_counts().to_string().replace("\n", "; "))
