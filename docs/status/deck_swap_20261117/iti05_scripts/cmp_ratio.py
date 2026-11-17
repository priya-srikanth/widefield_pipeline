"""0..1.0 vs 0..0.5 s for the ITI lateral ratio (all-ITI and docked), from the rebuild tables (both windows computed)."""
import pandas as pd
from pathlib import Path

I = Path("M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/iti_licks")
TAG = "hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_pre0801_sel-camera-cue_stop-endlick-cls-contact-incomplete-stopA2jexrt3500c70_daqlk"
OUT = Path(__file__).parent
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 500)
for run, d in (("all", I / TAG), ("dock", I / "lateral_ratio_dock_20261117" / TAG)):
    T = pd.read_csv(d / "lateral_ratio_tests.csv")
    T = T[(T.set == "iti") & (T.epoch != "pre")]
    T["c"] = [f"{x:+.2f}{'*' if s is True or s == 'True' else ''}" if pd.notna(x) else "" for x, s in zip(T.delta, T.sig)]
    P = T.pivot_table(index=["pair", "spout"], columns=["epoch", "window"], values="c", aggfunc="first")
    P = P.reindex(columns=pd.MultiIndex.from_product([["acute", "subacute", "chronic"], ["logr_0-1", "logr_0-0.5"]]))
    P.to_csv(OUT / f"cmp_ratio_tests_{run}.csv")
    print(f"######## {run}: ITI change vs pre, log2 contra/ipsi (0-1 | 0-0.5)")
    print(P.to_string())
    M = pd.read_csv(d / "lateral_ratio_models.csv")
    M = M[M.set == "iti"]
    Q = M.pivot_table(index=["pair", "test"], columns="window", values="p", aggfunc="first")
    fails = M[M.test.str.startswith("FAILED")][["pair", "window", "test"]]
    print(f"-- LR tests p ({run})\n" + Q.round(4).to_string())
    print("-- failed fits\n" + fails.to_string())
    Q.to_csv(OUT / f"cmp_ratio_models_{run}.csv")
    # star summary
    for w in ("logr_0-1", "logr_0-0.5"):
        t = T[T.window == w]
        print(run, w, "stars", int((t.sig.astype(str) == "True").sum()), "of", int(t.delta.notna().sum()))
