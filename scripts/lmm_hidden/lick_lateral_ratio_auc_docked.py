"""LMM hidden-slide version of goal page Fig 14.9, `lick_lateral_ratio_auc_docked.png` (deck lr_ratio_cued_3panel = a crop
of its cued row): the contralesional / ipsilesional response ratio per spout and epoch, all 15 panels.

Source: scripts/lick_lateral_ratio.py (`compute`, `plot`) run under WFIELD_ITI_FILTER=dock with WFIELD_ITI_OUT_W =
<iti_licks>/itidock_main, collected + bannered by scripts/iti_dock_page_figures.py. The docked PNG was drawn from
<iti_licks>/itidock_main/lateral_ratio_{sessions,tests}.csv (iti row = docked-spout ITI bout starts; cued and cue rows
unchanged), so THAT sessions table is read here.

Value: per session x spout (>= 5 licks), `logr_0-1` = log2(contra AUC / ipsi AUC) over 0..1 s of the paired area's
response; spout cells = that spout's row, "all" = each session's mean over its six spout rows. Levels exist for every
session, pre included. Cells: set (cued, cue, iti) x area pair (ALM, tjM1, SSp-m, SSp-n, orofacial) = 15 panels x
(6 spouts + all) = 105 cells; Holm family = all 105 cells x 3 epochs (the published stars were uncorrected per cell).

Published test: per cell and epoch `analysis_kit.boot_delta` (paired nested animals > sessions bootstrap) of logr vs
pre, * = 95 % CI excludes 0 (only when >= 3 animals have both pre and that epoch); panel titles carry sessions-as-
observations likelihood-ratio mixed models (not redrawn here). The published figure plots LEVELS (mean +- SEM over
animals); this version plots the change vs each animal's own pre.

    python -m scripts.lmm_hidden.lick_lateral_ratio_auc_docked [--root <iti_licks>] [--outdir DIR]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.lmm_hidden import _paths, render

NAME = "lick_lateral_ratio_auc_docked"
ROOT = Path("M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/iti_licks")
VARIANT = "itidock_main"
WIN = "logr_0-1"              # cued / cue rows (0..1.0 s)
# per-set window (Priya 11-17): ITI row on 0..0.5 s; cued / cue rows keep 0..1.0 s (wf_cleanup2 lick_lateral_ratio.SET_WIN)
SET_WIN = {"cued": "logr_0-1", "cue": "logr_0-1", "iti": "logr_0-0.5"}
SETS = ("cued", "cue", "iti")
PAIRS = ("ALM", "tjM1", "SSp-m", "SSp-n", "orofacial")
POS = ("close_ipsi", "close_mid", "close_contra", "far_ipsi", "far_mid", "far_contra")
EP = ("acute", "subacute", "chronic")


def table(root: Path) -> pd.DataFrame:
    S0 = pd.read_csv(_paths.variant(root, VARIANT) / "lateral_ratio_sessions.csv")
    rows = []
    for a in PAIRS:             # panel grid: one row per area pair, columns cued / cue / iti
        for st in SETS:
            WIN = SET_WIN[st]
            S = S0.dropna(subset=[WIN])
            g = S[(S.set == st) & (S.pair == a)]
            G = g[g.spout.isin(POS)]
            parts = [(p, G[G.spout == p]) for p in POS]
            parts.append(("all", G.groupby(["label", "animal", "epoch"])[WIN].mean().reset_index()))
            for p, h in parts:
                rows.append(pd.DataFrame({"position": f"{st}|{a}:{p}", "panel": f"{a}, {st}",
                                          "epoch": h.epoch.to_numpy(), "animal": h.animal.to_numpy(),
                                          "label": h.label.to_numpy(), "value": h[WIN].to_numpy(float)}))
    D = pd.concat(rows, ignore_index=True)
    return D[np.isfinite(D.value)].reset_index(drop=True)


def animal_change(D: pd.DataFrame) -> pd.DataFrame:
    """Mean over animals (with pre and the epoch) of (animal's epoch mean - its pre mean): the published boot_delta
    point; NaN where fewer than 3 animals (the published script runs no test there)."""
    out = []
    for c, g in D.groupby("position", sort=False):
        a = g.groupby(["epoch", "animal"]).value.mean().unstack("animal")
        for ep in EP:
            if ep in a.index and "pre" in a.index:
                d = (a.loc[ep] - a.loc["pre"]).dropna()
                out.append(dict(position=c, epoch=ep, change=d.mean() if len(d) >= 3 else np.nan, n_animals=len(d)))
    return pd.DataFrame(out)


def published(root: Path) -> pd.DataFrame:
    T = pd.read_csv(_paths.variant(root, VARIANT) / "lateral_ratio_tests.csv")
    T = T[(T.window == T.set.map(SET_WIN)) & T.spout.isin([*POS, "all"]) & (T.epoch != "pre")].copy()
    T["position"] = T.set + "|" + T.pair + ":" + T.spout
    T["boot_star"] = np.where(T.sig.astype(str) == "True", "*", "")
    return T[["position", "epoch", "delta", "lo", "hi", "boot_star"]]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(_paths.iti_root(ROOT)))
    ap.add_argument("--outdir", default=str(render.OUTDIR))
    a = ap.parse_args(argv)
    root, outdir = Path(a.root), Path(a.outdir)
    D = table(root)
    outdir.mkdir(parents=True, exist_ok=True)
    f = outdir / f"{_paths.out_name(NAME)}_sessions.csv"
    D.to_csv(f, index=False)
    print("n sessions / animals per epoch, by set (cell 'all' of the ALM panel):")
    for st in SETS:
        g0 = D[D.position == f"{st}|ALM:all"]
        print(st, g0.groupby("epoch").agg(sessions=("label", "size"), animals=("animal", "nunique")).to_dict("index"))
    T = render.render(f, NAME, outdir, title="Contra / ipsi response ratio, log2 AUC (cued / cue rows 0..1.0 s, ITI row "
                      "0..0.5 s), change vs own pre -- docked ITI row (LMM)", ylabel="change in log2(contra / ipsi AUC)", cells=list(D.position.unique()),
                      labels={c: "all\nspouts" for c in D.position.unique() if c.endswith(":all")},
                      panel_by="panel", ncols=3)
    V = animal_change(D).merge(published(root), on=["position", "epoch"], how="outer")
    V = V.merge(T[["position", "epoch", "beta", "p", "star"]], on=["position", "epoch"], how="left")
    V["match"] = np.isclose(V.change, V.delta, atol=1e-9) | (V.change.isna() & V.delta.isna())
    pd.set_option("display.width", 220)
    print(V[~V.match].round(4).to_string(index=False))
    print(f"verification: {int(V.match.sum())}/{len(V)} cells match the published boot_delta point")
    V["boot_star"] = V.boot_star.fillna("")
    d = V[V.boot_star != V.star.str.replace("**", "*", regex=False)]
    print(f"LMM stars: {int((V.star != '').sum())} ({int((V.star == '**').sum())} **); bootstrap stars: "
          f"{int((V.boot_star != '').sum())}")
    print("cells where bootstrap * and LMM star differ (or ** vs *):")
    print(pd.concat([d, V[V.star == "**"]]).drop_duplicates(["position", "epoch"])
          [["position", "epoch", "delta", "lo", "hi", "boot_star", "beta", "p", "star"]].round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
