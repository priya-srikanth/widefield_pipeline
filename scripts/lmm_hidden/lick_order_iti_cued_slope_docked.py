"""LMM hidden-slide version of goal page Fig 13.2 / Key figure 2, `lick_order_iti_cued_slope_docked.png` (deck crops
o_lick_order_cued_row, o_lick_order_slopes): the LICK-ORDER SLOPE point panels (columns 4-5 of the published figure).

Source: scripts/iti_lick_order.py (`sessions`, `tests`, `plot_slope`) run under WFIELD_ITI_FILTER=dock with
WFIELD_ITI_OUT_W = <iti_licks>/itidock_main, collected + bannered by scripts/iti_dock_page_figures.py. The docked PNG
was drawn from <iti_licks>/itidock_main/lick_order_{sessions,tests}.csv (ITI rows = docked-spout ITI licks; cued rows
unchanged), so THAT sessions table is read here.

Value: per session, `<measure>_slope` = OLS slope (mm per lick) of the lick's value on min(lick order, 9), spout means
removed; measure in landing / decoded / lfl (like-for-like offset). Levels exist for every session, pre included.
Cells: set (ITI, cued) x scope (all spouts, far contra) x measure = 12 cells in 4 panels; epochs acute / subacute /
chronic vs each animal's own pre. Holm family = all 12 cells x 3 epochs (the published figure did no correction).

Published test: `analysis_kit.boot_delta` (paired nested animals > sessions bootstrap) of the slope change vs pre,
* = 95 % CI excludes 0 (dagger = an older sessions-as-observations mixed model, second opinion). The lick-order curves
(columns 1-3) are per-order traces and stay with their own per-order BH-FDR test: not drawn here.

    python -m scripts.lmm_hidden.lick_order_iti_cued_slope_docked [--root <iti_licks>] [--outdir DIR]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.lmm_hidden import render

NAME = "lick_order_iti_cued_slope_docked"
ROOT = Path("M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/iti_licks")
VARIANT = "itidock_main"
SETS = {"iti": "ITI", "trial": "cued"}
SCOPES = {"all spouts": "all", "far contra": "far contra"}
VALS = ("landing", "decoded", "lfl")
EP = ("acute", "subacute", "chronic")


def table(root: Path) -> pd.DataFrame:
    S = pd.read_csv(root / VARIANT / "lick_order_sessions.csv")
    rows = []
    for st, slab in SETS.items():
        for sc, sclab in SCOPES.items():
            g = S[(S.set == st) & (S.scope == sc)]
            for v in VALS:
                rows.append(pd.DataFrame({"position": f"{st}|{sc}:{v}", "panel": f"{slab}, {sclab}: lick-order slope",
                                          "epoch": g.epoch.to_numpy(), "animal": g.animal.to_numpy(),
                                          "label": g.label.to_numpy(), "value": g[f"{v}_slope"].to_numpy(float)}))
    D = pd.concat(rows, ignore_index=True)
    return D[np.isfinite(D.value)].reset_index(drop=True)


def animal_change(D: pd.DataFrame) -> pd.DataFrame:
    """Mean over animals of (animal's epoch mean - animal's pre mean): the published boot_delta point."""
    out = []
    for c, g in D.groupby("position", sort=False):
        a = g.groupby(["epoch", "animal"]).value.mean().unstack("animal")
        for ep in EP:
            if ep in a.index and "pre" in a.index:
                d = (a.loc[ep] - a.loc["pre"]).dropna()
                out.append(dict(position=c, epoch=ep, change=d.mean(), n_animals=len(d),
                                n_sessions=int((g.epoch == ep).sum())))
    return pd.DataFrame(out)


def published(root: Path) -> pd.DataFrame:
    T = pd.read_csv(root / VARIANT / "lick_order_tests.csv")
    T = T[T.measure.str.endswith("_slope") & (T.epoch != "pre")].copy()
    T["position"] = T.set + "|" + T.scope + ":" + T.measure.str.replace("_slope", "")
    T["boot_star"] = np.where((T.lo > 0) | (T.hi < 0), "*", "")
    return T[["position", "epoch", "point", "lo", "hi", "boot_star"]]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--outdir", default=str(render.OUTDIR))
    a = ap.parse_args(argv)
    root, outdir = Path(a.root), Path(a.outdir)
    D = table(root)
    outdir.mkdir(parents=True, exist_ok=True)
    f = outdir / f"{NAME}_sessions.csv"
    D.to_csv(f, index=False)
    print("n sessions / animals per epoch (first cell):")
    g0 = D[D.position == D.position.iloc[0]]
    print(g0.groupby("epoch").agg(sessions=("label", "size"), animals=("animal", "nunique")).to_string())
    labels = {c: c.split(":")[-1] + "\nslope" for c in D.position.unique()}
    T = render.render(f, NAME, outdir, title="Lick-order SLOPE (spout means removed), change vs own pre -- docked ITI "
                      "/ cued (LMM)", ylabel="change in slope vs pre (mm per lick)", cells=list(D.position.unique()),
                      labels=labels, panel_by="panel", ncols=2)
    V = animal_change(D).merge(published(root), on=["position", "epoch"], how="outer")
    V = V.merge(T[["position", "epoch", "beta", "p", "star"]], on=["position", "epoch"], how="left")
    V["match"] = np.isclose(V.change, V.point, atol=1e-9)
    pd.set_option("display.width", 220)
    print(V.round(4).to_string(index=False))
    print(f"verification: {int(V.match.sum())}/{len(V)} cells match the published boot_delta point")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
