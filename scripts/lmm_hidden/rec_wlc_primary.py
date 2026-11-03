"""LMM hidden-slide version of goal page Fig 14.18 S21b, `rec_wlc_primary.png` (deck slide 39 crop
rec_wlc_primary_body): WITHIN-LICK CORRECTION at the far-contra spout (cued post-cue licks, all orders and outcomes).

Source: scripts/within_lick_correction.py (`sessions_table` / `cell`, `fig_primary` = wlc_fig1_primary_far_contra.png,
copied to the page as rec_wlc_primary.png; summarise run of 10-30 16:15-16:27) -> <within_lick_correction>/sessions.csv,
one row per session x (context, group, order, outcome). Rows used: context cued, group far_contra, order any, outcome
any (exactly fig_primary's `sel`). Pre = AUGUST pre only (prepare() drops June pre before the table is built).
Values (per session, as the published panels): A e_L launch error (deg), B e_E endpoint error (deg), C corr = e_E - e_L
(deg), D csm_share, E csm_false (noise control), F g_w within-lick gain. Each is a session LEVEL (errors vs a fixed
per-animal pre-hit reference), so pre sessions carry rows and the change vs own pre is defined.
Cells: 6 panels x 1 cell; Holm family = all 6 x 3 epochs = 18 (the published ** was Holm over the 9 PRIMARY tests only
(A, C, D) -- primary_holm.csv; its * = boot CI excludes 0 AND >= 3/4 animals same sign).
Published test: `analysis_kit.boot_delta` via lick_correction.boot_table (boot.csv `change`).

    python -m scripts.lmm_hidden.rec_wlc_primary [--root <within_lick_correction>] [--outdir DIR]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.lmm_hidden import render

NAME = "rec_wlc_primary"
ROOT = Path("M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/"
            "within_lick_correction")
SEL = {"context": "cued", "group": "far_contra", "order": "any", "outcome": "any"}
METS = {"e_L": "A. launch e_L (deg)", "e_E": "B. endpoint e_E (deg)",
        "corr": "C. correction (deg)", "csm_share": "D. CSM share",
        "csm_false": "E. noise CSM share", "g_w": "F. within-lick gain"}
PRIMARY = ("e_L", "corr", "csm_share")
EP = ("acute", "subacute", "chronic")


def _sel(D: pd.DataFrame) -> pd.DataFrame:
    for k, v in SEL.items():
        D = D[D[k] == v]
    return D


def table(root: Path) -> pd.DataFrame:
    S = _sel(pd.read_csv(root / "sessions.csv"))
    S = S.groupby(["animal", "label", "epoch"])[list(METS)].mean().reset_index()   # one row per session (as the plot)
    rows = [pd.DataFrame({"position": f"{m}:far_contra", "panel": lab, "epoch": S.epoch.to_numpy(),
                          "animal": S.animal.to_numpy(), "label": S.label.to_numpy(), "value": S[m].to_numpy(float)})
            for m, lab in METS.items()]
    D = pd.concat(rows, ignore_index=True)
    return D[np.isfinite(D.value)].reset_index(drop=True)


def animal_change(D: pd.DataFrame) -> pd.DataFrame:
    out = []
    for c, g in D.groupby("position", sort=False):
        a = g.groupby(["epoch", "animal"]).value.mean().unstack("animal")
        for ep in EP:
            if ep in a.index and "pre" in a.index:
                d = (a.loc[ep] - a.loc["pre"]).dropna()
                out.append(dict(position=c, epoch=ep, change=d.mean(), n_animals=len(d)))
    return pd.DataFrame(out)


def published(root: Path) -> pd.DataFrame:
    B = _sel(pd.read_csv(root / "boot.csv"))
    B = B[B.metric.isin(list(METS)) & (B.epoch != "pre")].copy()
    H = pd.read_csv(root / "primary_holm.csv")
    hh = {(m, e): p for m, e, p in zip(H.metric, H.epoch, H.p_holm)}
    B["boot_star"] = [("**" if hh.get((m, e), 1) < 0.05 else "*") if bool(c) else ""
                      for m, e, c in zip(B.metric, B.epoch, B.change)]
    B["position"] = B.metric + ":far_contra"
    return B[["position", "epoch", "delta", "delta_lo", "delta_hi", "n_same", "boot_star"]]


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
    pd.set_option("display.width", 220)
    print("n sessions / animals per epoch, per cell:")
    print(D.groupby(["position", "epoch"]).agg(sessions=("label", "size"), animals=("animal", "nunique"))
          .unstack("epoch").to_string())
    cells = [f"{m}:far_contra" for m in METS]
    T = render.render(f, NAME, outdir, title="Fig 14.18 S21b: within-lick correction at the FAR-CONTRA spout (cued "
                      "post-cue licks, August pre), change vs own pre (LMM)",
                      ylabel="change vs pre", cells=cells, labels={c: "far\ncontra" for c in cells},
                      panel_by="panel", ncols=6)
    V = animal_change(D).merge(published(root), on=["position", "epoch"], how="outer")
    V = V.merge(T[["position", "epoch", "beta", "p", "star"]], on=["position", "epoch"], how="left")
    V["match"] = np.isclose(V.change, V.delta, atol=1e-6)
    print(V.round(4).to_string(index=False))
    print(f"verification: {int(V.match.sum())}/{len(V)} cells match the published boot_delta point")
    V.to_csv(outdir / f"{NAME}_verify.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
