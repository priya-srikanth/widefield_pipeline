"""Post-transient undershoot per set (Priya 11-17 "this isn't biphasic for cued licks right?"): per session (spout-pooled
row) the min of the baseline-subtracted session-mean curve in 0.5..1.2 s, its peak 0..0.5 s, and the share of sessions /
session x spout rows whose 0.5..1.0 s AUC < 0. usage: python neg_auc_undershoot.py <all.pkl> <dock.pkl>"""
from __future__ import annotations

import pickle
import sys

import numpy as np
import pandas as pd

pd.set_option("display.width", 200)
POS = ["close_ipsi", "close_mid", "close_contra", "far_ipsi", "far_mid", "far_contra"]
A = pickle.load(open(sys.argv[1], "rb"))
D = pickle.load(open(sys.argv[2], "rb"))
L = A["L"]
w_late = (L >= 0.5 - 1e-9) & (L <= 1.0 + 1e-9)
w_tr = (L >= 0.5 - 1e-9) & (L <= 1.2 + 1e-9)
w_pk = (L >= 0) & (L <= 0.5 + 1e-9)
recs = []
for src, C in (("all", A["C"]), ("dock", D["C"])):
    for r in C.itertuples():
        if src == "dock" and r.set != "iti":
            continue
        st = "iti (docked)" if src == "dock" else ("iti (all)" if r.set == "iti" else r.set)
        for s, v in (("contra", r.curve_c), ("ipsi", r.curve_i)):
            recs.append({"set": st, "pair": r.pair, "spout": r.spout, "animal": r.animal, "epoch": r.epoch, "side": s,
                         "trough": float(np.nanmin(v[w_tr])), "peak": float(np.nanmax(v[w_pk])),
                         "auc_late": float(np.trapezoid(v[w_late], L[w_late])),
                         "auc_01": float(np.trapezoid(v[(L >= -1e-9) & (L <= 1.0 + 1e-9)], L[(L >= -1e-9) & (L <= 1.0 + 1e-9)]))})
U = pd.DataFrame(recs)
order = ["cued", "cue", "iti (all)", "iti (docked)"]
for pair in ("tjM1", "orofacial", None):
    g = U if pair is None else U[U.pair == pair]
    P = g[g.spout == "pooled"]
    S = g[g.spout.isin(POS)]
    t = pd.DataFrame({
        "sessions x sides": P.groupby("set").size(),
        "median peak 0-0.5": P.groupby("set").peak.median(),
        "median trough 0.5-1.2": P.groupby("set").trough.median(),
        "share trough < 0": P.groupby("set").trough.apply(lambda v: (v < 0).mean()),
        "share AUC 0.5-1 < 0 (sessions)": P.groupby("set").auc_late.apply(lambda v: (v < 0).mean()),
        "share AUC 0.5-1 < 0 (spout rows)": S.groupby("set").auc_late.apply(lambda v: (v < 0).mean()),
        "share AUC 0-1 <= 0 (spout rows)": S.groupby("set").auc_01.apply(lambda v: (v <= 0).mean()),
    }).reindex(order)
    print(f"\n### {pair or 'all five pairs'} (session = spout-pooled row; contra + ipsi sides pooled)")
    print(t.round(3).to_string())
P = U[(U.spout == "pooled") & (U.pair == "tjM1")]
print("\n### tjM1, share of sessions with AUC 0.5-1 < 0, by set x epoch (contra + ipsi)")
print(P.groupby(["set", "epoch"]).auc_late.apply(lambda v: (v < 0).mean()).unstack("epoch")
      .reindex(index=order, columns=["pre", "acute", "subacute", "chronic"]).round(2).to_string())
print("\n### tjM1 median trough 0.5-1.2 by set x animal (pre sessions)")
print(P[P.epoch == "pre"].groupby(["set", "animal"]).trough.median().unstack("animal").reindex(order).round(1).to_string())
