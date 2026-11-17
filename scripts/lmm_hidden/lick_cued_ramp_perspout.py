"""LMM hidden-slide version of goal page Fig 14.7, `lick_cued_ramp_perspout.png` (deck lr_cued_perspout): the FIRST
cued lick response PER TARGET SPOUT, contralesional ALM / tjM1 / SSp-m and cortex.

The published image is a set of lick-aligned PETH curves (-2..+3 s) with per-time-bin bars (BH-FDR across bins); those
bins keep their family-wise test and are NOT re-tested here. What this figure re-tests is the scalar the caption quotes
for the same panels (`perspout_auc_tests.csv`, window 0-0.5): per session x spout x ROI, the response AUC 0..0.5 s after
the first cued lick, divided by the animal's mean PRE AUC at that spout.

Source: scripts/lick_cued_ramp.py (`_sessions`, `_pe`, `_spout_auc_tests`; run of 10-28 14:17, the PNG and
perspout_auc_tests.csv come from that run). Per-session values are recomputed (light) from the saved per-trial traces
<iti_licks>/cued_wide/<label>.npz (key `lick`, trials x LAGS_W x ROI, `pos` = target spout), exactly as `_pe`:
  sessions = stems of <iti_licks>/sessions_roi8fr (il.labels(), ROI v8), animal / epoch from that session file;
  per trial: minus the mean over -1.0..-0.5 s (BASE_SPOUT); session curve = mean over that spout's trials (>= 5 licks);
  AUC = trapezoid over 0..0.5 s; value = AUC / the animal's mean pre-session AUC at that spout and ROI (pre = 1).
Cells: 4 ROI panels x 6 spouts = 24 cells; Holm family = all 24 x 3 epochs (the published per-spout AUC test is
uncorrected: * = 95 % CI excludes 0).
Published test: `analysis_kit.boot_delta` (paired nested animals > sessions), animals with pre and that epoch, >= 3.

    python -m scripts.lmm_hidden.lick_cued_ramp_perspout [--root <iti_licks>] [--outdir DIR]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.lmm_hidden import _paths, render

NAME = "lick_cued_ramp_perspout"
ROOT = Path("M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/iti_licks")
SESS = "sessions_roi8fr"          # iti_licks ROI_VERSION 8 (SESS_NAME)
WIDE = "cued_wide"
ROIS = ["ALM_right", "tjM1_right", "SSp-m_right", "cortex"]     # lick_cued_ramp.SPOUT_ROIS
POS = ["close_ipsi", "close_mid", "close_contra", "far_ipsi", "far_mid", "far_contra"]
BASE = (-1.0, -0.5)               # BASE_SPOUT
WIN = (0.0, 0.5)                  # perspout_auc_tests window "0-0.5"
MIN_SPOUT = 5
EP = ("acute", "subacute", "chronic")


def _auc(v: np.ndarray, lags: np.ndarray, t0: float, t1: float) -> float:
    m = (lags >= t0 - 1e-9) & (lags <= t1 + 1e-9)
    return float(np.trapezoid(v[m], lags[m]))


def table(root: Path) -> pd.DataFrame:
    from scripts.lmm_hidden import _paths
    sd = _paths.sess_dir(root, SESS)
    labs = sorted(p.stem for p in sd.glob("*.npz"))
    if not labs:
        raise FileNotFoundError(sd)
    rows = []
    for x in labs:
        d = np.load(sd / f"{x}.npz", allow_pickle=False)
        an, ep = str(d["animal"]), str(d["epoch"])
        z = np.load(root / WIDE / f"{x}.npz")
        lags = np.round(np.asarray(z["lags"], float), 3)
        L, pos, cols = z["lick"].astype(float), np.asarray(z["pos"]).astype(str), list(z["cols"])
        bm = (lags >= BASE[0]) & (lags <= BASE[1])
        for p in POS:
            P = L[pos == p]
            if len(P) < MIN_SPOUT:
                continue
            P = P - np.nanmean(P[:, bm, :], 1, keepdims=True)
            m = np.nanmean(P, 0)
            for roi in ROIS:
                if roi in cols:
                    rows.append({"label": x, "animal": an, "epoch": ep, "roi": roi, "spout": p, "n": len(P),
                                 "auc": _auc(m[:, cols.index(roi)], lags, *WIN)})
    A = pd.DataFrame(rows)
    pre = A[A.epoch == "pre"].groupby(["roi", "spout", "animal"]).auc.mean()
    A["value"] = A.auc / np.array([pre.get((r, s, a), np.nan) for r, s, a in zip(A.roi, A.spout, A.animal)])
    A = A[np.isfinite(A.value)].copy()
    A["position"] = A.roi + ":" + A.spout
    A["panel"] = A.roi
    return A[["position", "panel", "epoch", "animal", "label", "value", "auc", "n"]].reset_index(drop=True)


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
    T = pd.read_csv(root / "perspout_auc_tests.csv")
    T = T[(T.set == "lick") & (T.window == f"{WIN[0]:g}-{WIN[1]:g}")].copy()
    T["position"] = T.roi + ":" + T.spout
    T["boot_star"] = np.where(T.sig.astype(bool), "*", "")
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
    pd.set_option("display.width", 220)
    print("n sessions / animals per epoch, per cell:")
    print(D.groupby(["position", "epoch"]).agg(sessions=("label", "size"), animals=("animal", "nunique"))
          .unstack("epoch").to_string())
    cells = [f"{r}:{p}" for r in ROIS for p in POS]
    T = render.render(f, NAME, outdir, title="Fig 14.7: first cued lick response PER SPOUT, AUC 0..0.5 s / the animal's "
                      "pre AUC at that spout, change vs own pre (LMM)",
                      ylabel="AUC 0..0.5 s / pre at spout,\nchange vs pre", cells=cells, panel_by="panel", ncols=2)
    V = animal_change(D).merge(published(root), on=["position", "epoch"], how="outer")
    V = V.merge(T[["position", "epoch", "beta", "p", "star"]], on=["position", "epoch"], how="left")
    V["match"] = np.isclose(V.change, V.delta, atol=1e-6)
    print(V.round(4).to_string(index=False))
    print(f"verification: {int(V.match.sum())}/{len(V)} cells match the published boot_delta point")
    V.to_csv(outdir / f"{NAME}_verify.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
