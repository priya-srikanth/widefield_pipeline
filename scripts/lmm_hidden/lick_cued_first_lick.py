"""LMM hidden-slide version of goal page Fig 13.4, `lick_cued_first_lick.png` (deck o_cued_first_lick): FIRST cued lick
(lick_idx 0, launched before feedback) vs LATER cued licks, change vs pre.

Source: scripts/iti_first_lick.py (`sessions`, `tests`, `plot`) -> <iti_licks>/first_lick_sessions.csv (one row per
session x class x spout; the PNG and this table were written by the same run, 10-27 21:44). Not a docked figure: cued
licks only, read from <iti_licks> itself.

Value (exactly what the published `tests` bootstraps): per session x class, the mean over that session's spout rows
(scope "all spouts") or the far_contra row (scope "far contra") of
  landing = mean lateral landing (mm) - the animal's pre mean for that class x spout,
  decoded = mean decoded lateral position (frozen KEY-row decoder) - the same pre reference,
  lfl     = like-for-like offset (decoded vs pre licks that landed at the same place).
Each session row is a LEVEL against a fixed per-animal reference, so pre sessions carry rows too and the change vs each
animal's own pre mean is well defined (the published boot_delta takes the same pre rows).
Cells: scope (far contra, all spouts) x measure (landing, decoded, lfl) = 6 panels, x class (first, later) = 12 cells;
Holm family = all 12 cells x 3 epochs (the published figure did no correction and drew no stars).
The first - later difference quoted in the published suptitle is not a panel and is not drawn here.

Published test: `analysis_kit.boot_delta` (paired nested animals > sessions bootstrap), 95 % CI drawn, no stars.

    python -m scripts.lmm_hidden.lick_cued_first_lick [--root <iti_licks>] [--outdir DIR]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.lmm_hidden import _paths, render

NAME = "lick_cued_first_lick"
ROOT = Path("M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/iti_licks")
SCOPES = ("far contra", "all spouts")
VALS = {"landing": "landing shift", "decoded": "decoded shift", "lfl": "like-for-like offset"}
CLS = ("first", "later")
EP = ("acute", "subacute", "chronic")


def table(root: Path) -> pd.DataFrame:
    S = pd.read_csv(root / "first_lick_sessions.csv")
    rows = []
    for sc in SCOPES:
        g0 = S if sc == "all spouts" else S[S.spout == "far_contra"]
        g1 = g0.groupby(["animal", "label", "epoch", "cls"])[list(VALS)].mean().reset_index()
        for v, vlab in VALS.items():
            for c in CLS:
                g = g1[g1.cls == c]
                rows.append(pd.DataFrame({"position": f"{sc}|{v}:{c}", "panel": f"{sc}: {vlab} (mm)",
                                          "epoch": g.epoch.to_numpy(), "animal": g.animal.to_numpy(),
                                          "label": g.label.to_numpy(), "value": g[v].to_numpy(float)}))
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
                out.append(dict(position=c, epoch=ep, change=d.mean(), n_animals=len(d)))
    return pd.DataFrame(out)


def published(root: Path) -> pd.DataFrame:
    T = pd.read_csv(root / "first_lick_tests.csv")
    T = T[T.cls.isin(CLS)].copy()
    T["position"] = T.scope + "|" + T.measure + ":" + T.cls
    T["boot_star"] = np.where((T.lo > 0) | (T.hi < 0), "*", "")
    return T[["position", "epoch", "point", "lo", "hi", "boot_star"]]


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
    print("n sessions / animals per epoch, per cell:")
    print(D.groupby(["position", "epoch"]).agg(sessions=("label", "size"), animals=("animal", "nunique"))
          .unstack("epoch").to_string())
    labels = {c: c.split(":")[-1] + "\nlicks" for c in D.position.unique()}
    T = render.render(f, NAME, outdir, title="FIRST cued lick vs LATER licks, change vs own pre (LMM)",
                      ylabel="change vs pre (mm, + = contralesional)", cells=list(D.position.unique()),
                      labels=labels, panel_by="panel", ncols=3)
    V = animal_change(D).merge(published(root), on=["position", "epoch"], how="outer")
    V = V.merge(T[["position", "epoch", "beta", "p", "star"]], on=["position", "epoch"], how="left")
    V["match"] = np.isclose(V.change, V.point, atol=1e-9)
    pd.set_option("display.width", 220)
    print(V.round(4).to_string(index=False))
    print(f"verification: {int(V.match.sum())}/{len(V)} cells match the published boot_delta point")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
