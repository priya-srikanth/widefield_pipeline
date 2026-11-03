"""Shared helpers for the like-for-like lick read-out LMM extractors (Figs 12.1, 13.6, 12.15 docked; Priya 2026-11-03).

The three published figures were drawn by scripts/lick_matched_readout.py (--ref gauss) and
scripts/lick_same_target_place.py under WFIELD_ITI_FILTER=dock, which writes into <iti_licks>/itidock_main/
(WFIELD_ITI_OUT_W); scripts/iti_dock_page_figures.py then adds the green banner and renames them <stem>_docked.png.
So the per-session tables of the DOCKED figures are <iti_licks>/itidock_main/{matched_gauss,sametarget}_{sessions,tests}.csv
(ITI rows = docked-spout ITI licks; cued rows unchanged).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/iti_licks/"
            "itidock_main")
EP = ("acute", "subacute", "chronic")


def args(doc: str) -> argparse.Namespace:
    from scripts.lmm_hidden import render
    ap = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(ROOT), help="folder holding the docked per-session / tests CSVs")
    ap.add_argument("--outdir", default=str(render.OUTDIR))
    return ap.parse_args()


def verify(rows: pd.DataFrame, tests: pd.DataFrame) -> pd.DataFrame:
    """Per cell x epoch: mean over animals of (animal post mean - animal pre mean) from OUR rows, against the published
    bootstrap point (tests.point) and its star (95 % CI excludes 0). `tests` must carry columns position, epoch, point,
    lo, hi."""
    out = []
    for c, g in rows.groupby("position", sort=False):
        pre = g[g.epoch == "pre"].groupby("animal").value.mean()
        for e in EP:
            am = (g[g.epoch == e].groupby("animal").value.mean() - pre).dropna()
            out.append(dict(position=c, epoch=e, ours=am.mean() if len(am) else np.nan, n_animals=len(am),
                            n_sessions=int((g.epoch == e).sum())))
    V = pd.DataFrame(out).merge(tests[["position", "epoch", "point", "lo", "hi"]], on=["position", "epoch"], how="left")
    V["diff"] = V.ours - V.point
    V["boot_star"] = np.where((V.lo > 0) | (V.hi < 0), "*", "")
    return V


def report(V: pd.DataFrame, T: pd.DataFrame, rows: pd.DataFrame) -> None:
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    print(f"max |extracted - published point| = {np.nanmax(np.abs(V['diff'])):.2e} over {V['diff'].notna().sum()} cells")
    n = rows.groupby("epoch").agg(sessions=("value", "size"), animals=("animal", "nunique"))
    print("rows per epoch (all cells pooled):\n" + n.to_string())
    M = V.merge(T[["position", "epoch", "beta", "p", "star"]], on=["position", "epoch"], how="left")
    M["star"] = M.star.fillna("")
    print(M[["position", "epoch", "ours", "point", "boot_star", "beta", "p", "star", "n_animals", "n_sessions"]]
          .round(4).to_string(index=False))
    print(f"LMM stars: {int((M.star != '').sum())} ({int((M.star == '**').sum())} **); "
          f"bootstrap stars: {int((M.boot_star != '').sum())}")
    d = M[M.star != M.boot_star]
    print("DIFFER (published -> LMM):")
    for r in d.itertuples():
        print(f"  {r.position} {r.epoch}: '{r.boot_star}' -> '{r.star}' (beta {r.beta:+.3f}, p {r.p:.4f})")
