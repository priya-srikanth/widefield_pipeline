"""Shared per-session x spout response AUC for the lick-ramp CURVE figures (DECISIONS 2026-11-03 "Curve figures: mixed
model on PRE-SPECIFIED per-session summaries", row "lick-ramp time courses": the response AUC 0-0.5 s already used on
lick_cued_ramp_perspout).

The definition is `lick_cued_ramp_perspout.table` (constants BASE, WIN, MIN_SPOUT, POS and `_auc` are imported from
it, not re-stated); that function is hard-wired to the cued set and four ROIs, so its loop is generalised here
(set key = 'lick' (first cued lick) or 'iti' (ITI bout starts), any ROI list, an optional per-trial keep mask):
  per trial minus the mean over BASE (-1.0..-0.5 s); session curve = mean over that spout's trials (>= MIN_SPOUT);
  AUC = trapezoid over WIN (0..0.5 s); value = AUC / the animal's mean PRE-session AUC at that spout and ROI.
For 'iti' the spout is the PREVIOUS trial's spout (lick_cued_ramp.bout_prev_pos, cache <iti_licks>/bout_prev_pos.csv)
and only bout starts with gap >= 2.0 s (iti_licks.GAP_MAIN) are kept, as lick_cued_ramp._sessions; `subset` = a column
of <iti_licks>/iti_dock_audit.csv (e.g. 'dock') applies iti_licks.bout_subset_keep (match within 3 ms).
`check_cued` asserts that this reproduces lick_cued_ramp_perspout.table exactly on its four ROIs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.lmm_hidden import lick_cued_ramp_perspout as lcp

GAP_MAIN = 2.0          # iti_licks.GAP_MAIN
BOUT_TOL_S = 0.003      # iti_licks.BOUT_TOL_S
EP = lcp.EP
ROIS7 = ["ALM_right", "ALM_left", "tjM1_right", "tjM1_left", "SSp-m_right", "SSp-m_left", "cortex"]   # il.KEY_ROIS


def _prev_pos(root: Path) -> dict:
    C = pd.read_csv(root / "bout_prev_pos.csv", float_precision="round_trip")
    return {x: dict(zip(g.bout_t.to_numpy(float), g.prev_pos.fillna("").astype(str))) for x, g in C.groupby("label")}


def _subset(root: Path, col: str) -> dict:
    A = pd.read_csv(root / "iti_dock_audit.csv", usecols=["label", "onset", col]).dropna(subset=["onset"])
    return {x: (g.sort_values("onset").onset.to_numpy(float), g.sort_values("onset")[col].to_numpy(bool))
            for x, g in A.groupby("label")}


def _subset_keep(aud: dict, label: str, bout_t: np.ndarray) -> np.ndarray:
    """== iti_licks.bout_subset_keep."""
    if label not in aud or not len(bout_t):
        return np.zeros(len(bout_t), bool)
    on, fl = aud[label]
    j = np.clip(np.searchsorted(on, bout_t), 1, max(len(on) - 1, 1)) if len(on) > 1 else np.zeros(len(bout_t), int)
    jl = np.maximum(j - 1, 0)
    near = np.where(np.abs(on[j] - bout_t) <= np.abs(on[jl] - bout_t), j, jl)
    return (np.abs(on[near] - bout_t) <= BOUT_TOL_S) & fl[near]


def table(root: Path, key: str = "lick", rois=tuple(lcp.ROIS), subset: str = "") -> pd.DataFrame:
    labs = sorted(p.stem for p in (root / lcp.SESS).glob("*.npz"))
    if not labs:
        raise FileNotFoundError(root / lcp.SESS)
    pp = _prev_pos(root) if key == "iti" else None
    aud = _subset(root, subset) if subset else None
    rows = []
    for x in labs:
        d = np.load(root / lcp.SESS / f"{x}.npz", allow_pickle=False)
        an, ep = str(d["animal"]), str(d["epoch"])
        z = np.load(root / lcp.WIDE / f"{x}.npz")
        lags = np.round(np.asarray(z["lags"], float), 3)
        cols = list(z["cols"])
        if key == "lick":
            L, pos = z["lick"].astype(float), np.asarray(z["pos"]).astype(str)
        else:
            bt = np.asarray(d["bout_t"], float)
            keep = np.asarray(z["iti_gap"]) >= GAP_MAIN
            if aud is not None:
                keep &= _subset_keep(aud, x, bt)
            m = pp.get(x, {})
            pos = np.array([m.get(float(t), "") for t in bt], dtype=object)[keep].astype(str)
            L = z["iti"].astype(float)[keep]
        bm = (lags >= lcp.BASE[0]) & (lags <= lcp.BASE[1])
        for p in lcp.POS:
            P = L[pos == p]
            if len(P) < lcp.MIN_SPOUT:
                continue
            P = P - np.nanmean(P[:, bm, :], 1, keepdims=True)
            mu = np.nanmean(P, 0)
            for roi in rois:
                if roi in cols:
                    rows.append({"label": x, "animal": an, "epoch": ep, "roi": roi, "spout": p, "n": len(P),
                                 "auc": lcp._auc(mu[:, cols.index(roi)], lags, *lcp.WIN)})
    A = pd.DataFrame(rows)
    pre = A[A.epoch == "pre"].groupby(["roi", "spout", "animal"]).auc.mean()
    A["value"] = A.auc / np.array([pre.get((r, s, a), np.nan) for r, s, a in zip(A.roi, A.spout, A.animal)])
    A = A[np.isfinite(A.value)].copy()
    A["position"] = A.roi + ":" + A.spout
    A["panel"] = A.roi
    return A[["position", "panel", "epoch", "animal", "label", "value", "auc", "n"]].reset_index(drop=True)


def check_cued(root: Path, D: pd.DataFrame) -> bool:
    """This module's cued table on the four perspout ROIs == lick_cued_ramp_perspout.table, row for row."""
    ref = lcp.table(root).sort_values(["position", "label"]).reset_index(drop=True)
    got = D[D.panel.isin(lcp.ROIS)].sort_values(["position", "label"]).reset_index(drop=True)
    ok = len(ref) == len(got) and (ref.label == got.label).all() and np.allclose(ref.value, got.value, equal_nan=True)
    print(f"check vs lick_cued_ramp_perspout.table (4 ROIs): {'IDENTICAL' if ok else 'DIFFERENT'} "
          f"({len(got)} vs {len(ref)} rows)")
    return bool(ok)


def published(root: Path, set_: str) -> pd.DataFrame:
    """perspout_auc_tests.csv rows of one set (lick / iti), window 0-0.5 (all-ITI run; 4 ROIs)."""
    T = pd.read_csv(root / "perspout_auc_tests.csv")
    T = T[(T.set == set_) & (T.window == f"{lcp.WIN[0]:g}-{lcp.WIN[1]:g}")].copy()
    T["position"] = T.roi + ":" + T.spout
    return T[["position", "epoch", "delta", "sig"]]


def summary(D: pd.DataFrame, T: pd.DataFrame) -> None:
    pd.set_option("display.width", 220)
    print("n sessions / animals per epoch (all cells):")
    print(D.groupby("epoch").agg(sessions=("label", "nunique"), animals=("animal", "nunique")).reindex(["pre", *EP])
          .to_string())
    print("n sessions per cell x epoch (min / max over cells):")
    n = D.groupby(["position", "epoch"]).label.size().unstack("epoch").reindex(columns=["pre", *EP])
    print(n.agg(["min", "max"]).to_string())
    s = T[T.star != ""]
    print(f"LMM starred cells: {len(s)} ({(s.star == '**').sum()} **)")
    print(s[["position", "epoch", "star", "beta", "p"]].round(4).to_string(index=False))
