"""Event-level ITI check (neg_auc_diag step 3): does the 0.5..1 s dip depend on whether licking continues, on the
spout-dock / next-trial timing, or on the animal x epoch? usage: python neg_auc_events.py <ev_all.pkl> <ev_dock.pkl>"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

pd.set_option("display.width", 220)
for path, mode in ((sys.argv[1], "all-ITI"), (sys.argv[2], "docked")):
    E = pd.read_pickle(path)
    E["nf"] = pd.cut(E.nfol, [-1, 0, 2, 5, 10, 1e9], labels=["0", "1-2", "3-5", "6-10", ">10"])
    print(f"\n======== {mode}: {len(E)} bout starts; following licks (nfol) median {E.nfol.median():.0f}")
    for a in ("tjM1", "SSp-n"):
        cols = {f"{a}_c_0-0.5": "c 0-.5", f"{a}_c_0.5-1": "c .5-1", f"{a}_c_0-1": "c 0-1", f"{a}_c_peak": "c peak",
                f"{a}_c_trough": "c trough", f"{a}_i_0-1": "i 0-1"}
        g = E.groupby("nf", observed=True)
        t = g[list(cols)].mean().rename(columns=cols)
        t["n_ev"] = g.size()
        t["share_ev"] = (g.size() / len(E)).round(3)
        print(f"-- {a}: mean per-event AUC by following-lick count (baseline -1..-0.5 s; dF/F x1e3 s)")
        print(t.round(2).to_string())
    # timing (docked mainly)
    if E.since_dock.notna().any():
        E["dk"] = pd.cut(E.since_dock, [-99, 0, 0.5, 1.0, 1.5, 99], labels=["<0 (not docked)", "0-0.5", "0.5-1", "1-1.5", ">1.5"])
        E["tt"] = pd.cut(E.to_trial_start, [-99, 0, 0.5, 1.0, 2.0, 99], labels=["<0", "0-0.5", "0.5-1", "1-2", ">2"])
        for k, lab in (("dk", "time since spout dock (s)"), ("tt", "time to next trial start (s)")):
            g = E.groupby(k, observed=True)
            t = g[["tjM1_c_0-0.5", "tjM1_c_0.5-1", "tjM1_c_0-1", "tjM1_c_rawbase", "nfol"]].mean()
            t["n_ev"] = g.size()
            print(f"-- tjM1 contra by {lab}")
            print(t.round(2).to_string())
    # per animal x epoch: share of isolated / short bout starts and their mean late dip
    g = E.groupby(["animal", "epoch"])
    t = pd.DataFrame({"n_ev": g.size(), "share_nfol<=2": g.nfol.apply(lambda v: (v <= 2).mean()).round(2),
                      "median_nfol": g.nfol.median(), "tjM1_c_0-1": g["tjM1_c_0-1"].mean().round(2),
                      "tjM1_c_.5-1": g["tjM1_c_0.5-1"].mean().round(2)})
    print("-- per animal x epoch")
    print(t.to_string())
    # within-row regression: per-event 0..1 s AUC ~ nfol (log) per animal x epoch -> how much the dip is licking-dependent
    E["lnf"] = np.log1p(E.nfol)
    r = E.groupby(["animal", "epoch"]).apply(lambda h: np.corrcoef(h.lnf, h["tjM1_c_0-1"])[0, 1] if len(h) > 10 else np.nan,
                                             include_groups=False)
    print(f"-- corr(log1p nfol, tjM1 contra 0..1 s AUC) within animal x epoch: median {r.median():.2f}, "
          f"range {r.min():.2f}..{r.max():.2f}")
