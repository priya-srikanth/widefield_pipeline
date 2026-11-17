"""Per-session, SPOUTS-POOLED lick response size for the hidden companions of PI-deck slide 94 ("Lick response larger
after stroke"): the quantity of the main slide, not the per-spout 0-0.5 s supplement (`_curve_auc`).

Definitions = wf_attempt scripts/lick_cued_ramp.py (`_sessions`, `_pe`, `compute`, `_windows`; RESP_W = (0.0, 1.0),
Priya 10-28 evening "0-1 s should be default now", commit d57b6098; constants re-stated below and checked by the
reproduction, not imported):
  per trial minus its mean over BASE_L (-1.5..-1.0 s); session PETH = mean over the session's trials (>= MIN_N = 10,
  all spouts); then minus the session PETH's mean over BASE_RESP (-1.0..-0.5 s), i.e. the response baseline;
  auc  = trapezoid over 0..1.0 s            (PRIMARY:   resp_auc / ramp_window_tests `auc_0_1_n`)
  peak = max over 0..1.5 s                  (SECONDARY: ramp_window_tests `peak_n`; Priya 11-03: AUC primary, peak
                                             secondary for regional response sizes)
  value = that / the animal's mean PRE-session value at that ROI (pre = 1).
key 'lick' = first cued lick of each engaged trial (<iti_licks>/cued_wide key `lick`); key 'iti' = ITI bout starts
(key `iti`) with gap >= 2.0 s (iti_licks.GAP_MAIN) and, for subset 'dock', flagged in <iti_licks>/iti_dock_audit.csv
(iti_licks.bout_subset_keep; the DOCKED-spout ITI set, primary since 10-29, folder <iti_licks>/itidock_main).
`check` compares the recomputed AUC with the published session table row for row and both changes with
ramp_window_tests.csv; the LMM is fitted only if every cell reproduces.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.lmm_hidden import _curve_auc as ca
from scripts.lmm_hidden import lick_cued_ramp_perspout as lcp

BASE_L = (-1.5, -1.0)        # iti_licks.BASE
BASE_RESP = (-1.0, -0.5)     # iti_licks.BASE_RESP
RESP_W = (0.0, 1.0)          # lick_cued_ramp.RESP_W
PEAK_W = (0.0, 1.5)          # lick_cued_ramp._windows peak
MIN_N = 10                   # lick_cued_ramp.MIN_N
EP = lcp.EP
ROIS7 = ca.ROIS7
LABELS = {"ALM_right": "ALM\ncontra", "ALM_left": "ALM\nipsi", "tjM1_right": "tjM1\ncontra",
          "tjM1_left": "tjM1\nipsi", "SSp-m_right": "SSp-m\ncontra", "SSp-m_left": "SSp-m\nipsi", "cortex": "cortex"}
# where each set's published tables live: (session table, folder of ramp_window_tests.csv, set name there)
PUB = {"lick": ("cued_ramp_auc_sessions.csv", "", "cued"),
       "iti_dock": ("itidock_main/iti_ramp_auc_sessions_wide.csv", "itidock_main", "iti")}


def table(root: Path, key: str = "lick", subset: str = "") -> pd.DataFrame:
    """One row per session x ROI: auc, peak, auc_n, peak_n (lick_cued_ramp.compute / _windows, spouts pooled)."""
    from scripts.lmm_hidden import _paths
    sd = _paths.sess_dir(root, lcp.SESS)
    labs = sorted(p.stem for p in sd.glob("*.npz"))
    if not labs:
        raise FileNotFoundError(sd)
    aud = ca._subset(root, subset) if subset else None
    rows = []
    for x in labs:
        d = np.load(sd / f"{x}.npz", allow_pickle=False)
        an, ep = str(d["animal"]), str(d["epoch"])
        z = np.load(root / lcp.WIDE / f"{x}.npz")
        lags = np.round(np.asarray(z["lags"], float), 3)
        cols = list(z["cols"])
        if key == "lick":
            P = z["lick"].astype(float)
        else:
            keep = np.asarray(z["iti_gap"]) >= ca.GAP_MAIN
            if aud is not None:
                keep &= ca._subset_keep(aud, x, np.asarray(d["bout_t"], float))
            P = z["iti"].astype(float)[keep]
        if len(P) < MIN_N:
            continue
        bl = (lags >= BASE_L[0]) & (lags <= BASE_L[1])
        br = (lags >= BASE_RESP[0]) & (lags <= BASE_RESP[1])
        pk = (lags >= PEAK_W[0]) & (lags <= PEAK_W[1])
        mu = np.nanmean(P - np.nanmean(P[:, bl, :], 1, keepdims=True), 0)
        for roi in ROIS7:
            if roi in cols:
                v = mu[:, cols.index(roi)]
                v = v - np.nanmean(v[br])
                rows.append({"label": x, "animal": an, "epoch": ep, "roi": roi, "n": len(P),
                             "auc": lcp._auc(v, lags, *RESP_W), "peak": float(np.nanmax(v[pk]))})
    A = pd.DataFrame(rows)
    for v in ("auc", "peak"):
        pre = A[A.epoch == "pre"].groupby(["roi", "animal"])[v].mean()
        A[v + "_n"] = A[v] / np.array([pre.get((r, a), np.nan) for r, a in zip(A.roi, A.animal)])
    return A


def as_sessions(A: pd.DataFrame, measure: str) -> pd.DataFrame:
    """render input: position = ROI, one panel, value = auc_n or peak_n."""
    D = A[np.isfinite(A[measure + "_n"])].copy()
    D["position"] = D.roi
    D["value"] = D[measure + "_n"]
    D["raw"] = D[measure]
    return D[["position", "epoch", "animal", "label", "value", "raw", "n"]].reset_index(drop=True)


def animal_table(D: pd.DataFrame) -> pd.DataFrame:
    """Rule 8: per ROI x animal, epoch mean - own pre mean (and n sessions); the cell change = mean over animals."""
    rows = []
    for (c, an), g in D.groupby(["position", "animal"], sort=False):
        m = g.groupby("epoch").value.agg(["mean", "size"])
        if "pre" not in m.index:
            continue
        r = {"position": c, "animal": an, "n_pre": int(m.loc["pre", "size"])}
        for ep in EP:
            r[ep] = m.loc[ep, "mean"] - m.loc["pre", "mean"] if ep in m.index else np.nan
            r[f"n_{ep}"] = int(m.loc[ep, "size"]) if ep in m.index else 0
        rows.append(r)
    return pd.DataFrame(rows)


def check(root: Path, A: pd.DataFrame, which: str) -> pd.DataFrame:
    """Reproduction: (1) recomputed AUC == the published session table resp_auc / resp_auc_n, row for row; (2) per
    cell change (mean over animals of animal epoch mean - own pre mean) == ramp_window_tests delta for auc_0_1_n AND
    peak_n. Returns the cell table; raises if anything differs."""
    sess, wdir, set_ = PUB[which]
    from scripts.lmm_hidden import _paths
    if which == "iti_dock":       # the docked tables' folder: LMM_HIDDEN_ITI_VARIANT (absolute path allowed), default itidock_main
        d = _paths.variant(root, "itidock_main")
        sess, wdir = str(d / Path(sess).name), str(d)
        print(f"[{which}] published docked tables: {d}", flush=True)
    if not (root / sess).exists() or not (root / wdir / "ramp_window_tests.csv").exists():
        if not _paths.allow_no_pub():
            raise SystemExit(f"[{which}] published table {root / sess} absent: no reproduction possible "
                             "(set LMM_HIDDEN_ALLOW_NO_PUB=1 to fit without it)")
        print(f"[{which}] !! NO REPRODUCTION CHECK: {root / sess} (or its ramp_window_tests.csv) does not exist on this "
              "tree; the LMM is fitted on the recomputed values only (LMM_HIDDEN_ALLOW_NO_PUB=1)", flush=True)
        out = []
        for meas in ("auc", "peak"):
            D = as_sessions(A, meas)
            for c, g in D.groupby("position", sort=False):
                a = g.groupby(["epoch", "animal"]).value.mean().unstack("animal")
                for ep in EP:
                    d = (a.loc[ep] - a.loc["pre"]).dropna() if ep in a.index else pd.Series(dtype=float)
                    out.append(dict(measure=meas, position=c, epoch=ep, change=d.mean(), n_animals=len(d),
                                    published=np.nan, boot_lo=np.nan, boot_hi=np.nan))
        V = pd.DataFrame(out)
        V["boot_star"] = ""
        V["match"] = False
        return V
    S = pd.read_csv(root / sess)
    M = S.merge(A, on=["label", "roi"], how="outer", suffixes=("_pub", ""), indicator=True)
    ok_rows = ((M._merge == "both").all() and np.allclose(M.resp_auc, M.auc, rtol=1e-9, atol=1e-9)
               and np.allclose(M.resp_auc_n, M.auc_n, rtol=1e-9, atol=1e-9))
    print(f"[{which}] session rows: {len(A)} recomputed vs {len(S)} in {sess}: "
          f"{'IDENTICAL (resp_auc, resp_auc_n)' if ok_rows else 'DIFFERENT'}")
    W = pd.read_csv(root / wdir / "ramp_window_tests.csv")
    W = W[W.set == set_]
    out = []
    for meas, col in (("auc", "auc_0_1_n"), ("peak", "peak_n")):
        D = as_sessions(A, meas)
        for c, g in D.groupby("position", sort=False):
            a = g.groupby(["epoch", "animal"]).value.mean().unstack("animal")
            for ep in EP:
                d = (a.loc[ep] - a.loc["pre"]).dropna()
                w = W[(W.roi == c) & (W.epoch == ep) & (W.value == col)]
                out.append(dict(measure=meas, position=c, epoch=ep, change=d.mean(), n_animals=len(d),
                                published=float(w.delta.iloc[0]) if len(w) else np.nan,
                                boot_lo=float(w.lo.iloc[0]) if len(w) else np.nan,
                                boot_hi=float(w.hi.iloc[0]) if len(w) else np.nan))
    V = pd.DataFrame(out)
    V["boot_star"] = np.where((V.boot_lo > 0) | (V.boot_hi < 0), "*", "")
    V["match"] = np.isclose(V.change, V.published, rtol=0, atol=1e-9)
    for meas in ("auc", "peak"):
        v = V[V.measure == meas]
        print(f"[{which}] {meas}: {int(v.match.sum())}/{len(v)} cells match ramp_window_tests.csv "
              f"({wdir or '.'}, set {set_})")
    if not (ok_rows and V.match.all()):
        print(V[~V.match].round(6).to_string(index=False))
        raise SystemExit(f"[{which}] reproduction FAILED: no model fitted")
    return V


def report(D: pd.DataFrame, T: pd.DataFrame, V: pd.DataFrame, measure: str, outdir: Path, name: str) -> pd.DataFrame:
    """Save the per-animal table (printed by `run` before the fit); print + save the LMM / bootstrap cell table."""
    pd.set_option("display.width", 220)
    P = animal_table(D)
    P.to_csv(outdir / f"{name}_animals.csv", index=False)
    C = V[V.measure == measure].drop(columns=["measure", "match"]).merge(
        T[["position", "epoch", "beta", "lo", "hi", "p", "df", "star"]].rename(columns={"lo": "lmm_lo", "hi": "lmm_hi"}),
        on=["position", "epoch"], how="left")
    C.to_csv(outdir / f"{name}_verify.csv", index=False)
    print(f"\n== {name}: per cell, published bootstrap vs LMM")
    print(C.round(4).to_string(index=False))
    s = C[C.star != ""]
    print(f"LMM starred: {len(s)} of {int(C.p.notna().sum())} tested ({(s.star == '**').sum()} **); "
          f"bootstrap *: {int((C.boot_star == '*').sum())}")
    return C


MEASURES = {  # measure -> (file suffix, title words, y label)
    "auc": ("_w0-1", "response AUC 0..1.0 s / the animal's pre (primary)", "AUC 0..1.0 s / pre,\nchange vs pre"),
    "peak": ("_peak", "secondary: peak 0..1.5 s / the animal's pre", "peak 0..1.5 s / pre,\nchange vs pre")}


def run(root: Path, outdir: Path, *, key: str, subset: str, which: str, name: str, what: str) -> dict:
    """Reproduce, then for each measure: per-animal table, LMM (one panel, 7 ROIs, ** Holm across the figure's
    7 x 3 = 21 tests) via render / lmm_change_fig; files <name>_w0-1* and <name>_peak*."""
    from scripts.lmm_hidden import render
    A = table(root, key=key, subset=subset)
    V = check(root, A, which)
    outdir.mkdir(parents=True, exist_ok=True)
    out = {}
    for meas, (suf, words, ylab) in MEASURES.items():
        nm = f"{name}{suf}"
        D = as_sessions(A, meas)
        f = outdir / f"{nm}_sessions.csv"
        D.to_csv(f, index=False)
        P = animal_table(D)
        pd.set_option("display.width", 220)
        print(f"\n== {nm}: PER ANIMAL first (rule 8): animal epoch mean - own pre mean; n sessions")
        print(P.round(3).to_string(index=False))
        T = render.render(f, nm, outdir, title=f"{what}, spouts pooled, {words}", ylabel=ylab, cells=ROIS7,
                          labels=LABELS)
        out[meas] = report(D, T, V, meas, outdir, nm)
    return out
