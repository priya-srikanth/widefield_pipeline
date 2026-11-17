"""Does 0..1 s vs 0..0.5 s change the OTHER window-dependent lick-response results? Read-only comparison of existing
production test tables (ramp_window_tests.csv = slides 105-109 window sensitivity; perspout_auc_tests.csv = slides
113-114 per-spout AUC), all-ITI and docked-ITI runs."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

R = Path("M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/"
         "iti_licks")
TAG = ("hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_pre0801_sel-camera-cue_stop-endlick-cls-contact-incomplete-"
       "stopA2jexrt3500c70_daqlk")


def cmp(a: pd.DataFrame, b: pd.DataFrame, keys, what):
    m = a.merge(b, on=keys, suffixes=("_1", "_05"))
    s1 = (m.lo_1 > 0) | (m.hi_1 < 0)
    s5 = (m.lo_05 > 0) | (m.hi_05 < 0)
    same_sign = np.sign(m.delta_1) == np.sign(m.delta_05)
    print(f"{what}: {len(m)} cells; sig 0-1 {s1.sum()}, sig 0-0.5 {s5.sum()}, both {(s1 & s5).sum()}, "
          f"only 0-1 {(s1 & ~s5).sum()}, only 0-0.5 {(~s1 & s5).sum()}; sign agree {same_sign.mean():.3f}; "
          f"opposite-sign sig {((s1 & s5) & ~same_sign).sum()}")
    return m[(s1 != s5)]


for run, d in (("all-ITI", R / TAG), ("docked-ITI", R / "itidock_main_20261117" / TAG)):
    print(f"\n===== {run}: {d}")
    f = d / "ramp_window_tests.csv"
    if f.exists():
        W = pd.read_csv(f)
        for st in W.set.unique():
            a = W[(W.set == st) & (W.value == "auc_0_1_n")].drop(columns="value")
            b = W[(W.set == st) & (W.value == "auc_0_0.5_n")].drop(columns="value")
            diff = cmp(a, b, ["set", "roi", "epoch"], f"ramp_window_tests {st} (normalized AUC, change vs pre)")
            if len(diff):
                print(diff[["roi", "epoch", "delta_1", "lo_1", "hi_1", "delta_05", "lo_05", "hi_05"]].round(3).to_string(index=False))
    else:
        print("no", f)
    f = d / "perspout_auc_tests.csv"
    if f.exists():
        P = pd.read_csv(f)
        for st in P.set.unique():
            a = P[(P.set == st) & (P.window == "0-1")].drop(columns=["window", "cell", "sig"])
            b = P[(P.set == st) & (P.window == "0-0.5")].drop(columns=["window", "cell", "sig"])
            diff = cmp(a, b, ["set", "spout", "roi", "epoch"], f"perspout_auc_tests {st}")
            if len(diff) and len(diff) <= 25:
                print(diff[["spout", "roi", "epoch", "delta_1", "lo_1", "hi_1", "delta_05", "lo_05", "hi_05"]].round(2).to_string(index=False))
    else:
        print("no", f)
