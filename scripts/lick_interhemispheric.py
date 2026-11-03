"""INTERHEMISPHERIC COORDINATION AROUND LICKS OVER RECOVERY (Priya item 3, 10-29): does contralesional orofacial cortex
become more / less coupled to ipsilesional cortex after the ipsilesional VLS lesion, and does it start to LEAD?

SETS (per session; signals = the iti_licks area traces: adopted hemo-corrected SVD, encoder slow-drift basis removed,
dF/F x1e3, sampled every 0.05 s on il.LAGS_D = -1..+1 s around the 3-D lick PEAK):
  cued      every first post-cue lick of a bout (firstlick_full/<label>.npz, uncapped; target spout = the trial's)
  iti_all   every decoder ITI lick (session files iti_X; spout = the PREVIOUS trial's, iti_pos)
  iti_far1  ITI licks >= 1 s after the response-window end and >= 1 s before the next spout arrival (the far1 rule of
            iti_lick_order, from iti_timing_audit.csv) -- the ~docked subset
  iti_gap2  ITI licks with no lick onset in the preceding 2 s (bout starts after quiet; pre-lick window lick-free)
  rest      quiet DOCKED rest (restdock05 rest mask x strict docked window, engaged period only:
            rest_by_position.rest_frames_by_position(docked=True)); docked rest periods are only ~1.0-1.5 s long, so
            ONE pseudo-event per period >= 0.95 s (its centre), window -0.5..+0.5 s (cached in
            <OUT_W>/interhemi_rest_w1/<label>.npz)
WINDOWS: w2 = -1..+1 s (lick sets: nc_post / nc_pre, cc lags +-0.5 s); w1 = -0.5..+0.5 s, identical code on lick sets
  and rest (nc_win = noise corr of the window mean, cc lags +-0.3 s); lick-vs-rest comparisons use w1 only.
PAIRS (homotopic, contralesional = image RIGHT = '_right', ipsilesional = '_left'): SSp-n, SSp-m, tjM1, ALM, SSs, MOp_rest
  (MOp minus its tjM1 part, il.with_rest; disjoint from tjM1), controls VISpm (VISp_left is not in any session's area
  set) and RSPd, and 'oro' = the primary pool
  ALM + tjM1 + SSp-m + SSp-n + SSs summed per hemisphere (lick_recovery_primary.POOL).
RESIDUALS ("pre-style model", fit within each session): every lag x hemisphere regressed on [1, spout dummies, lick angle]
  (angle = arcsin(side_ml / reach_mm); ITI: previous-trial spout; rest: intercept only) -> the spout / angle mean and the
  angle-linked part are removed; what is left is trial-to-trial variability.
METRICS (sign: + = CONTRALESIONAL hemisphere leads / predicts):
  1 nc_post  Pearson r across events between hemispheres of the residual post-lick response (mean 0.1..0.5 s minus
             -1.0..-0.5 s, il.BASE_RESP; the primary balance window); nc_pre = same for the mean over -1..0 s.
             Fisher z averaged where pooled (reported as r = tanh).
  2 cc(L)    pooled within-event cross-correlation of the residual traces (each event's trace de-meaned over its 2-s
             window), lag L = -0.5..+0.5 s, cc(L) = corr(c(t), i(t + L)): + L peak = contra leads. cc0 = zero lag,
             lag_peak = parabolic-interpolated argmax, asym = mean cc over +0.05..+0.30 s minus over -0.30..-0.05 s.
             gc_ci / gc_ic = Granger-style 5-fold (by event) cross-validated predictive gain (1 - SSE_full / SSE_own)
             of 3 past samples (0.15 s) of the other hemisphere over the target's own 3 past samples; null = the same
             with the driver taken from a shuffled event (mean of 5 shuffles); gc_dir = (gc_ci - null) - (gc_ic - null).
             Each on raw (tau 0) and deconvolved s = x + tau dx/dt (tau 0.6 / 1.0 s, revup_neural.deconv).
  3 rest     identical code on rest pseudo-events (non-lick baseline); lick-specific change = (lick - rest) per session.
  4 recovery within-animal partial r of each session metric with far-contra cued aim (aim_deficit_sessions via
             recmark_sessions) and with the cued contralesional share (balmatch_sessions bal_spout, matched-error
             within-spout balance), both controlling log(day), post sessions only; pooled = animal-demeaned partial r
             with a within-animal permutation p.
TIMING CAVEAT: GCaMP6s rise ~0.1-0.2 s, decay 0.5-1 s; hemispheres can differ in indicator expression / haemodynamics, so a
  lag or Granger asymmetry under ~0.1-0.2 s is NOT neural timing, and a CHANGE in it is only suggestive. Samples are
  interpolated at 20 Hz from the imaging frame rate. Nothing here is causal.
Statistics: per-animal tables FIRST (rule 8), then analysis_kit.boot_delta (paired nested animals -> sessions) of each
  post epoch vs August pre.
Outputs (<OUT_W>): interhemi_sessions.csv.gz, interhemi_animal.csv, interhemi_tests.csv, interhemi_recovery.csv,
  interhemi_rest_census.csv, figdata/lick_interhemi_*.pkl, figures/lick_interhemi_*.png.
Usage: python scripts/lick_interhemispheric.py [--jobs 4] [--extract-rest] [--summarise] [--plot-only]
"""
from __future__ import annotations

import argparse
import pickle
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iti_licks as il  # noqa: E402

OUT_W = Path("M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/iti_licks")
IN = il.OUT                                                  # inputs (read only)
REST_DIR = OUT_W / "interhemi_rest_w1"                      # (interhemi_rest/ = a first 2-s-window attempt: 0 windows)
FIGD, FIGS = OUT_W / "figdata", OUT_W / "figures"
LAGS = il.LAGS_D                                             # -1..+1 s, 0.05 s
DT = 0.05
# WINDOW FAMILIES. Docked rest periods are 1.0-1.5 s long (PS92_0806: 279 periods, median 1.25 s, max 1.55 s), so the
# 2-s lick window cannot be applied at rest. w2 = the full -1..+1 s lick window (lick sets only); w1 = the central
# -0.5..+0.5 s, the SAME code on lick sets and on rest (one rest pseudo-event per rest period >= 0.95 s, its centre) --
# every lick-vs-rest comparison uses w1.
LAGS_W1 = np.round(np.arange(-0.5, 0.5001, DT), 3)
IDX_W1 = np.flatnonzero((LAGS >= -0.5 - 1e-9) & (LAGS <= 0.5 + 1e-9))
MAXLW = {"w2": 10, "w1": 6}                                  # cross-correlation lags +-0.5 s / +-0.3 s
REST_MIN_SPAN = 0.95
# VISp_left (and VISl / VISal left) is absent from EVERY session's area set (left posterior-lateral visual cortex is off
# the stat mask in all 107 sessions), so the visual control pair is VISpm (bilateral in all sessions)
PAIRS = ["SSp-n", "SSp-m", "tjM1", "ALM", "SSs", "MOp_rest", "VISpm", "RSPd", "oro"]
POOL = ["ALM", "tjM1", "SSp-m", "SSp-n", "SSs"]
SETS = ["cued", "iti_all", "iti_far1", "iti_gap2", "rest"]
LICK_SETS = SETS[:-1]
TAUS = (0.0, 0.6, 1.0)
P_GC = 3
MIN_N = 30
N_SHUF = 5
EPS = ["pre", "acute", "subacute", "chronic"]
AN = ["PS92", "PS93", "PS94", "PS95"]
METRICS = ["nc_post", "nc_pre", "nc_win", "cc0", "cc_peak", "lag_peak", "asym", "gc_ci", "gc_ic", "gc_dir"]
NC = ("nc_post", "nc_pre", "nc_win")
N_BOOT = 2000


def _w(lo, hi, incl=False):
    return (LAGS >= lo - 1e-9) & ((LAGS <= hi + 1e-9) if incl else (LAGS < hi - 1e-9))


W_POST, W_BASE, W_PRE = _w(0.1, 0.5), _w(*il.BASE_RESP, incl=True), _w(-1.0, 0.0, incl=True)


# ------------------------------------------------------------------------------------------------ rest extraction
def rest_extract(label: str) -> dict:
    """Docked engaged rest windows of one session -> REST_DIR/<label>.npz (X (n, 41, cols) float16, centres t)."""
    from wfield_local import combined_encoding as ce
    from wfield_local import config
    from wfield_local import rest_by_position as rbp
    f = REST_DIR / f"{label}.npz"
    if f.exists():
        return {"label": label, "cached": True}
    s = next(x for x in config.load_sessions() if x["label"] == label)
    got = rbp._session_daq(s)                               # one DAQ read, reused by rest_frames_by_position below
    orig = rbp._session_daq
    rbp._session_daq = lambda _s: got
    try:
        per, info = rbp.rest_frames_by_position(s, 10 ** 9, docked=True, engaged_only=True)
    finally:
        rbp._session_daq = orig
    if info.get("error"):
        raise RuntimeError(f"{label}: rest mask {info['error']}")
    fs_samp = np.asarray(got[4], np.int64)
    idx = np.unique(np.concatenate(list(per.values()))) if per else np.zeros(0, int)
    V, ft, Ub, flat, parea, names, shape, pstat = ce.load_imaging(label)
    M = il.roi_masks(parea, pstat, names, flat, shape)
    cols = list(M)
    Ra = np.stack([Ub[M[n]].mean(0) for n in cols], 1).astype(np.float64)
    R = V.astype(np.float64) @ Ra
    Q = np.linalg.qr(ce._slow_basis(ft, ce.DRIFT_MIN_PERIOD_S))[0]
    R -= Q @ (Q.T @ R)
    R *= 1e3
    # rest frames -> DAQ seconds via the rest module's own frame map (5 kHz DAQ); periods = runs of consecutive frames
    tr = fs_samp[np.clip(idx, 0, len(fs_samp) - 1)] / 5000.0
    # clock check: each rest-frame time should sit on an imaging frame time (ft) within half a frame
    j = np.clip(np.searchsorted(ft, tr), 1, len(ft) - 1)
    clk = float(np.median(np.minimum(np.abs(tr - ft[j - 1]), np.abs(tr - ft[j])))) if len(tr) else np.nan
    brk = np.flatnonzero(np.diff(idx) > 1) + 1
    centres, spans = [], []
    for seg_t in np.split(tr, brk):
        if len(seg_t) >= 2:
            spans.append(seg_t[-1] - seg_t[0])
            if spans[-1] >= REST_MIN_SPAN:
                centres.append(0.5 * (seg_t[0] + seg_t[-1]))
    centres = np.asarray(centres, float)
    centres = centres[(centres + LAGS_W1[0] > ft[0]) & (centres + LAGS_W1[-1] < ft[-1])]
    X = il._sample(R, ft, centres, LAGS_W1).astype(np.float16) if len(centres) else np.zeros(
        (0, len(LAGS_W1), len(cols)), np.float16)
    rest_s = float(len(idx) * np.median(np.diff(ft)))
    REST_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, X=X, t=centres, cols=np.array(cols), rest_s=rest_s, n_periods=info["n_periods"],
                        reconstructed=bool(info["reconstructed"]), clock_med_s=clk, frame_dt=float(np.median(np.diff(ft))),
                        spans=np.asarray(spans, float))
    return {"label": label, "n_win": len(centres), "rest_s": rest_s, "clock_med_s": clk, "n_periods": len(spans),
            "span_med_s": float(np.median(spans)) if spans else np.nan, "reconstructed": bool(info["reconstructed"])}


# ------------------------------------------------------------------------------------------------ per-session metrics
def _pair_traces(X: np.ndarray, cols, label: str) -> dict:
    """{pair: (n, T, 2) [contra, ipsi]} from (n, T, cols)."""
    Xf, names = il.with_rest(X.astype(np.float64), cols, label)
    ix = {c: i for i, c in enumerate(names)}
    out = {}
    for p in PAIRS:
        if p == "oro":
            c = sum(Xf[..., ix[f"{a}_right"]] for a in POOL)
            i = sum(Xf[..., ix[f"{a}_left"]] for a in POOL)
        else:
            c, i = Xf[..., ix[f"{p}_right"]], Xf[..., ix[f"{p}_left"]]
        out[p] = np.stack([c, i], -1)
    return out


def _design(pos: np.ndarray | None, ang: np.ndarray | None, n: int) -> np.ndarray:
    cols = [np.ones(n)]
    if pos is not None:
        u = sorted(set(pos))
        cols += [(pos == v).astype(float) for v in u[1:]]
    if ang is not None:
        cols.append(ang)
    return np.column_stack(cols)


def _resid(Y: np.ndarray, D: np.ndarray) -> np.ndarray:
    sh = Y.shape
    Y2 = Y.reshape(sh[0], -1)
    return (Y2 - D @ np.linalg.lstsq(D, Y2, rcond=None)[0]).reshape(sh)


def _corr(a, b) -> float:
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else np.nan


def xcorr(E: np.ndarray, maxl: int) -> np.ndarray:
    """Pooled cc(L), L = -maxl..maxl samples; E (n, T, 2) de-meaned per event. + L = ipsi later = contra leads."""
    c, i = E[..., 0], E[..., 1]
    T = c.shape[1]
    out = np.empty(2 * maxl + 1)
    for k, L in enumerate(range(-maxl, maxl + 1)):
        a = c[:, max(0, -L):T - max(0, L)]
        b = i[:, max(0, L):T - max(0, -L)]
        out[k] = (a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum())
    return out


def cc_summary(cc: np.ndarray) -> dict:
    maxl = (len(cc) - 1) // 2
    L = np.arange(-maxl, maxl + 1) * DT
    k = int(np.argmax(cc))
    lag = L[k]
    if 0 < k < len(cc) - 1:
        den = cc[k - 1] - 2 * cc[k] + cc[k + 1]
        if den < 0:
            lag += DT * 0.5 * (cc[k - 1] - cc[k + 1]) / den
    pos, neg = (L >= 0.05 - 1e-9) & (L <= 0.30 + 1e-9), (L <= -0.05 + 1e-9) & (L >= -0.30 - 1e-9)
    return {"cc0": float(cc[maxl]), "cc_peak": float(cc[k]), "lag_peak": float(lag),
            "asym": float(cc[pos].mean() - cc[neg].mean())}


def _lagmat(x: np.ndarray, p: int) -> np.ndarray:
    """(n, T) -> (n, T - p, p) past samples t-1..t-p for targets t = p..T-1."""
    T = x.shape[1]
    return np.stack([x[:, p - k:T - k] for k in range(1, p + 1)], -1)


def _fit_pred(Z: np.ndarray, tgt: np.ndarray, trn: np.ndarray, te: np.ndarray) -> np.ndarray:
    Ztr = Z[trn].reshape(-1, Z.shape[-1])
    b = np.linalg.lstsq(np.c_[np.ones(len(Ztr)), Ztr], tgt[trn].ravel(), rcond=None)[0]
    Zte = Z[te].reshape(-1, Z.shape[-1])
    return np.c_[np.ones(len(Zte)), Zte] @ b


def _gc_gain(y: np.ndarray, drv: np.ndarray, folds: np.ndarray | None = None) -> float:
    """Predictive gain 1 - SSE_full / SSE_own of drv's past over y's own past. folds None = IN-SAMPLE (the default used:
    a 5-fold CV version swung by +-0.3 between fold seeds on one session -- one outlier event decides a fold -- while
    the in-sample gain with only P_GC extra parameters over ~n x 38 rows has a negligible fitting bias, and the
    event-shuffle null subtracts what remains)."""
    tgt = y[:, P_GC:]
    own = _lagmat(y, P_GC)
    full = np.concatenate([own, _lagmat(drv, P_GC)], -1)
    if folds is None:
        allm = np.ones(len(y), bool)
        yt = tgt.ravel()
        sse_r = float(((yt - _fit_pred(own, tgt, allm, allm)) ** 2).sum())
        sse_f = float(((yt - _fit_pred(full, tgt, allm, allm)) ** 2).sum())
        return 1 - sse_f / sse_r if sse_r > 0 else np.nan
    sse_r = sse_f = 0.0
    for f in np.unique(folds):
        te, trn = folds == f, folds != f
        yt = tgt[te].ravel()
        sse_r += float(((yt - _fit_pred(own, tgt, trn, te)) ** 2).sum())
        sse_f += float(((yt - _fit_pred(full, tgt, trn, te)) ** 2).sum())
    return 1 - sse_f / sse_r if sse_r > 0 else np.nan


def granger(E: np.ndarray, rng) -> dict:
    """Granger-style gains each way on events scaled to unit joint RMS (both hemispheres share each event's scale, so
    the relation between them is unchanged while a few huge events cannot dominate the pooled SSE)."""
    n = E.shape[0]
    sc = np.sqrt((E ** 2).mean((1, 2)))
    E = E / np.where(sc > 0, sc, 1)[:, None, None]
    c, i = E[..., 0], E[..., 1]
    g_ci, g_ic = _gc_gain(i, c), _gc_gain(c, i)
    nci, nic = [], []
    for _ in range(N_SHUF):
        perm = rng.permutation(n)
        while n > 1 and np.any(perm == np.arange(n)):
            perm = rng.permutation(n)
        nci.append(_gc_gain(i, c[perm]))
        nic.append(_gc_gain(c, i[perm]))
    nci, nic = float(np.mean(nci)), float(np.mean(nic))
    return {"gc_ci": g_ci, "gc_ic": g_ic, "gc_ci_null": nci, "gc_ic_null": nic, "gc_dir": (g_ci - nci) - (g_ic - nic)}


def _deconv(E: np.ndarray, tau: float) -> np.ndarray:
    from revup_neural import deconv
    if not tau:
        return E
    return np.moveaxis(deconv(np.moveaxis(E, 1, 0), tau), 0, 1)


def _angle(side, reach):
    side, reach = np.asarray(side, float), np.asarray(reach, float)
    return np.degrees(np.arcsin(np.clip(side / reach, -1, 1)))


def _set_arrays(label: str, meta: pd.DataFrame) -> dict:
    """{set: (X (n, T, cols), pos or None, angle or None, cols)}."""
    out = {}
    z = il._load(label)
    cols = [str(c) for c in z["cols"]]
    fl = np.load(IN / "firstlick_full" / f"{label}.npz", allow_pickle=False)
    ang = _angle(fl["side_ml"], fl["reach_mm"])
    ok = np.isfinite(ang)
    out["cued"] = (fl["X"][ok], fl["pos"].astype(str)[ok], ang[ok], [str(c) for c in fl["cols"]])
    t = np.round(z["iti_t"].astype(float), 4)
    m = meta.set_index("tk").reindex(t)
    ang = _angle(z["iti_side"], m.reach_mm.to_numpy(float))
    ok = np.isfinite(ang)
    pos = z["iti_pos"].astype(str)
    X = z["iti_X"]
    far1 = ((m.since_stop >= 1.0) & (m.to_strobe >= 1.0)).to_numpy(bool)
    gap2 = (m.gap >= 2.0).to_numpy(bool)
    out["iti_all"] = (X[ok], pos[ok], ang[ok], cols)
    out["iti_far1"] = (X[ok & far1], pos[ok & far1], ang[ok & far1], cols)
    out["iti_gap2"] = (X[ok & gap2], pos[ok & gap2], ang[ok & gap2], cols)
    rf = REST_DIR / f"{label}.npz"
    if rf.exists():
        r = np.load(rf, allow_pickle=False)
        out["rest"] = (r["X"], None, None, [str(c) for c in r["cols"]])
    return out


class Item(tuple):
    """(label, ITI meta table); repr = the label (fan_out logs the item)."""

    def __repr__(self):
        return str(self[0])


def session_metrics(item):
    """(label, meta) -> (rows DataFrame, {(set, win, pair, tau): cc curve})."""
    label, meta = item
    from wfield_local import config, epochs
    rng = np.random.default_rng(int(label[2:4]) * 10000 + int(label[-4:]))
    an, ep = config.animal_of(label), epochs.epoch_of(label)
    rows, curves = [], {}
    for st, (X, pos, ang, cols) in _set_arrays(label, meta).items():
        n = len(X)
        if n < MIN_N:
            continue
        D = _design(pos, ang, n)
        for p, Y in _pair_traces(X, cols, label).items():
            E = _resid(Y, D)
            wins = {}
            if X.shape[1] == len(LAGS):                      # lick sets: w2 (full window) and w1 (central 1 s)
                post = E[:, W_POST].mean(1) - E[:, W_BASE].mean(1)
                pre = E[:, W_PRE].mean(1)
                wins["w2"] = (E, {"nc_post": _fz(post), "nc_pre": _fz(pre)})
                E1 = E[:, IDX_W1]
            else:                                            # rest: w1 only
                E1 = E
            wins["w1"] = (E1, {"nc_win": _fz(E1.mean(1))})
            for win, (Ew, nc) in wins.items():
                Ed = Ew - Ew.mean(1, keepdims=True)
                for tau in TAUS:
                    Et = _deconv(Ed, tau)
                    Et = Et - Et.mean(1, keepdims=True)
                    cc = xcorr(Et, MAXLW[win])
                    curves[(st, win, p, tau)] = cc
                    r = {"label": label, "animal": an, "epoch": ep, "set": st, "win": win, "pair": p, "tau": tau,
                         "n": n, **{m: (nc.get(m, np.nan) if tau == 0 else np.nan) for m in NC}, **cc_summary(cc),
                         **granger(Et, rng)}
                    rows.append(r)
    return pd.DataFrame(rows), curves


def _fz(v: np.ndarray) -> float:
    """Fisher z of the across-event correlation between hemispheres (v (n, 2))."""
    return float(np.arctanh(np.clip(_corr(v[:, 0], v[:, 1]), -0.999, 0.999)))


def _meta() -> dict:
    """Per label: ITI lick table keyed on rounded t (reach from covref, gap / far1 times from the timing audit)."""
    C = pd.read_csv(IN / "covref_covariates.csv.gz", usecols=["set", "label", "t", "reach_mm"])
    C = C[C.set == "iti"].drop(columns="set")
    A = pd.read_csv(IN / "iti_timing_audit.csv", usecols=["label", "t", "gap", "since_stop", "to_strobe"])
    for d in (C, A):
        d["tk"] = np.round(d.t.astype(float), 4)
    M = C.drop(columns="t").merge(A.drop(columns="t"), on=["label", "tk"], how="outer")
    M = M.drop_duplicates(["label", "tk"])
    return {lab: g.drop(columns="label") for lab, g in M.groupby("label")}


# ------------------------------------------------------------------------------------------------ summaries
def _r(z):
    return np.tanh(z)


def summarise(S: pd.DataFrame) -> None:
    from wfield_local import analysis_kit as ak
    S = S[(S.epoch != "pre") | (S.label.str[-4:-2] == "08")].copy()
    # lick-specific: lick minus rest on the same session (w1, same pair / tau)
    K = ["label", "win", "pair", "tau"]
    R = S[S.set == "rest"].set_index(K)[METRICS]
    LS = []
    for st in LICK_SETS:
        x = S[(S.set == st) & (S.win == "w1")].set_index(K)
        d = x[METRICS] - R.reindex(x.index)
        d = d.join(x[["animal", "epoch"]]).reset_index()
        d["set"] = f"{st}-rest"
        LS.append(d)
    SS = pd.concat([S] + LS, ignore_index=True)
    SS.to_csv(OUT_W / "interhemi_sessions_with_lickminusrest.csv.gz", index=False)
    # per-animal table (rule 8)
    G = SS.groupby(["set", "win", "pair", "tau", "animal", "epoch"])
    A = G[METRICS].mean().join(G.size().rename("n_sess")).reset_index()
    for m in NC:
        A[m + "_r"] = _r(A[m])
    A.to_csv(OUT_W / "interhemi_animal.csv", index=False)
    rng = np.random.default_rng(0)
    T = []
    for (st, win, p, tau), g in SS.groupby(["set", "win", "pair", "tau"], sort=True):
        for m in METRICS:
            if g[m].notna().sum() == 0:
                continue
            pre = {a: v[m].dropna().to_list() for a, v in g[g.epoch == "pre"].groupby("animal")}
            pre = {a: v for a, v in pre.items() if v}
            lv = ak.boot_ci(pre, rng, n_boot=N_BOOT)
            for ep in EPS[1:]:
                post = {a: v[m].dropna().to_list() for a, v in g[g.epoch == ep].groupby("animal")}
                post = {a: v for a, v in post.items() if v}
                iv = ak.boot_delta(post, pre, rng, n_boot=N_BOOT)
                if iv is None:
                    continue
                dl = [np.mean(post[a]) - np.mean(pre[a]) for a in sorted(set(post) & set(pre))]
                T.append({"set": st, "win": win, "pair": p, "tau": tau, "metric": m, "epoch": ep,
                          "pre_level": lv.point if lv else np.nan,
                          "delta": iv.point, "lo": iv.lo, "hi": iv.hi, "n_animals": iv.n_animals,
                          "n_same_sign": int(sum(np.sign(x) == np.sign(iv.point) for x in dl)),
                          "excl0": iv.excludes_zero, "per_animal": " ".join(f"{x:+.3f}" for x in dl)})
    T = pd.DataFrame(T)
    T.to_csv(OUT_W / "interhemi_tests.csv", index=False)
    recovery(SS)


def recovery(SS: pd.DataFrame) -> None:
    """Within-animal partial r (controlling log day) of each session metric with cued far-contra aim and cued share."""
    R = pd.read_csv(IN / "recmark_sessions.csv", usecols=["label", "day", "aim"]).drop_duplicates("label")
    B = pd.read_csv(IN / "balmatch_sessions.csv")
    B = B[B.set == "trial"][["label", "bal_spout"]].rename(columns={"bal_spout": "share"})
    X = SS[SS.epoch != "pre"].merge(R, on="label", how="left").merge(B, on="label", how="left")
    X = X[X.day > 0]
    X["lday"] = np.log(X.day)
    rng = np.random.default_rng(1)
    out = []
    for (st, win, p, tau), g in X.groupby(["set", "win", "pair", "tau"], sort=True):
        if tau == 1.0:
            continue
        for m in METRICS:
            if g[m].notna().sum() == 0 or m in ("cc_peak", "gc_ci", "gc_ic"):
                continue
            for tgt in ("aim", "share"):
                h = g.dropna(subset=[m, tgt, "lday"])
                rs, res = {}, []
                for a, q in h.groupby("animal"):
                    if len(q) < 5:
                        continue
                    D = np.c_[np.ones(len(q)), q.lday]
                    ry = q[m].to_numpy() - D @ np.linalg.lstsq(D, q[m].to_numpy(), rcond=None)[0]
                    rx = q[tgt].to_numpy() - D @ np.linalg.lstsq(D, q[tgt].to_numpy(), rcond=None)[0]
                    rs[a] = _corr(ry, rx)
                    res.append((ry, rx))
                if len(rs) < 3:
                    continue
                ry = np.concatenate([a for a, _ in res])
                rx = np.concatenate([b for _, b in res])
                r0 = _corr(ry, rx)
                nul = []
                for _ in range(1000):
                    nul.append(_corr(np.concatenate([rng.permutation(a) for a, _ in res]), rx))
                pv = (1 + np.sum(np.abs(nul) >= abs(r0))) / (1 + len(nul))
                out.append({"set": st, "win": win, "pair": p, "tau": tau, "metric": m, "target": tgt, "r_pooled_within": r0, "p_perm": pv,
                            "n_animals": len(rs), "n_same_sign": int(sum(np.sign(v) == np.sign(r0) for v in rs.values())),
                            **{f"r_{a}": rs.get(a, np.nan) for a in AN}, "n_sess": len(h)})
    pd.DataFrame(out).to_csv(OUT_W / "interhemi_recovery.csv", index=False)


# ------------------------------------------------------------------------------------------------ figures
COL = {"pre": "#555555", "acute": "#d62728", "subacute": "#ff7f0e", "chronic": "#1f77b4"}


def _wrap(s, w=95):
    return "\n".join(textwrap.wrap(s, w))


def figures_compute() -> None:
    T = pd.read_csv(OUT_W / "interhemi_tests.csv")
    A = pd.read_csv(OUT_W / "interhemi_animal.csv")
    Rc = pd.read_csv(OUT_W / "interhemi_recovery.csv")
    FIGD.mkdir(parents=True, exist_ok=True)
    with open(FIGD / "lick_interhemi_tables.pkl", "wb") as fh:
        pickle.dump({"T": T, "A": A, "Rc": Rc}, fh)


def _win(st: str) -> str:
    return "w2" if st in LICK_SETS else "w1"


def _delta_panel(ax, T, st, m, tau, title, win=None):
    t = T[(T.set == st) & (T.metric == m) & (T.tau == tau) & (T.win == (win or _win(st)))]
    for k, ep in enumerate(EPS[1:]):
        q = t[t.epoch == ep].set_index("pair").reindex(PAIRS)
        x = np.arange(len(PAIRS)) + (k - 1) * 0.25
        ax.errorbar(x, q.delta, yerr=[q.delta - q.lo, q.hi - q.delta], fmt="o", ms=3, color=COL[ep], label=ep, lw=1)
        for xi, (_, rr) in zip(x, q.iterrows()):
            if rr.excl0 is True or rr.excl0 == "True":
                ax.text(xi, rr.hi, "*", ha="center", va="bottom", fontsize=8, color=COL[ep])
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(range(len(PAIRS)))
    ax.set_xticklabels(PAIRS, rotation=45, fontsize=7)
    ax.set_title(_wrap(title, 45), fontsize=8)


def plot_all() -> None:
    import matplotlib.pyplot as plt
    with open(FIGD / "lick_interhemi_tables.pkl", "rb") as fh:
        D = pickle.load(fh)
    T, A = D["T"], D["A"]
    FIGS.mkdir(parents=True, exist_ok=True)
    # 1 noise correlation
    fig, axs = plt.subplots(3, 5, figsize=(17, 10), sharey="row", layout="constrained")
    for j, st in enumerate(SETS):
        if st in LICK_SETS:
            _delta_panel(axs[0, j], T, st, "nc_post", 0.0, f"{st}: post-lick (0.1-0.5 s minus -1..-0.5 s) noise corr, change")
            _delta_panel(axs[1, j], T, st, "nc_pre", 0.0, f"{st}: pre-lick (-1..0 s) noise corr, change")
        else:
            axs[0, j].axis("off")
            axs[1, j].axis("off")
        _delta_panel(axs[2, j], T, st, "nc_win", 0.0, f"{st}: 1-s window (-0.5..+0.5 s) mean, noise corr, change", "w1")
    # legend in the empty rest panel (rest has no lick-locked rows), never on top of data
    axs[0, 4].legend(*axs[0, 0].get_legend_handles_labels(), loc="center", fontsize=9, title="change vs pre",
                     frameon=False)
    fig.suptitle(_wrap("Interhemispheric trial-to-trial (noise) correlation of residual responses (spout + angle mean "
                       "removed per session; Fisher z); change vs August pre, paired nested bootstrap 95% CI, * excludes 0; "
                       "bottom row = the same 1-s window on lick sets and on docked quiet rest (one pseudo-event per rest "
                       "period)", 160), fontsize=9)
    fig.savefig(FIGS / "lick_interhemi_noisecorr.png", dpi=130)
    plt.close(fig)
    # 2 lead measures (lick sets + rest + lick-rest), raw and deconvolved
    for tau in TAUS:
        fig, axs = plt.subplots(3, 5, figsize=(17, 10.5), layout="constrained")
        for j, st in enumerate(SETS):
            _delta_panel(axs[0, j], T, st, "asym", tau, f"{st}: xcorr asymmetry (+ = contra leads), change")
            _delta_panel(axs[1, j], T, st, "gc_dir", tau, f"{st}: Granger direction (+ = contra predicts ipsi more), change")
            _delta_panel(axs[2, j], T, st, "cc0", tau, f"{st}: zero-lag within-event coupling, change")
        fig.legend(*axs[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=3, fontsize=9,
                   frameon=False, title="change vs pre (* = CI excludes 0)", title_fontsize=9)
        fig.suptitle(_wrap(f"Within-event coupling and lead, tau = {tau} s ({'raw' if not tau else 'deconvolved s = x + tau dx/dt'}); "
                           "change vs August pre; GCaMP6s: lags < 0.1-0.2 s are not neural timing", 160), fontsize=9)
        fig.savefig(FIGS / f"lick_interhemi_lead_tau{tau:.1f}.png", dpi=130)
        plt.close(fig)
    # 3 lick-specific (lick - rest)
    fig, axs = plt.subplots(3, 4, figsize=(15, 10.5), layout="constrained")
    for j, st in enumerate(LICK_SETS):
        _delta_panel(axs[0, j], T, f"{st}-rest", "nc_win", 0.0, f"{st} minus rest: 1-s window noise corr change")
        _delta_panel(axs[1, j], T, f"{st}-rest", "cc0", 0.0, f"{st} minus rest: zero-lag within-event coupling change")
        _delta_panel(axs[2, j], T, f"{st}-rest", "asym", 0.6, f"{st} minus rest: xcorr asymmetry change (tau 0.6)")
    fig.legend(*axs[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=3, fontsize=9, frameon=False,
               title="change vs pre (* = CI excludes 0)", title_fontsize=9)
    fig.suptitle(_wrap("LICK-SPECIFIC change: per session (lick metric - docked-rest metric, both on the same 1-s window "
                       "code), change vs pre", 150), fontsize=9)
    fig.savefig(FIGS / "lick_interhemi_lickspecific.png", dpi=130)
    plt.close(fig)
    # 4 per-animal trajectories for the orofacial pool
    ms = [("nc_post", 0.0, None), ("nc_win", 0.0, "w1"), ("cc0", 0.0, "w1"), ("asym", 0.0, None), ("asym", 0.6, None),
          ("gc_dir", 0.6, None)]
    fig, axs = plt.subplots(len(ms), 5, figsize=(17, 2.5 * len(ms)), sharex=True, layout="constrained")
    empty = []
    for i, (m, tau, win) in enumerate(ms):
        for j, st in enumerate(SETS):
            ax = axs[i, j]
            q = A[(A.set == st) & (A.pair == "oro") & (A.tau == tau) & (A.win == (win or _win(st)))]
            if q[m].notna().sum() == 0:      # e.g. rest has no post-lick window: no data, use the slot for the legend
                ax.axis("off")
                empty.append((ax, f"{st}: no {m}"))
                continue
            for a in AN:
                v = q[q.animal == a].set_index("epoch").reindex(EPS)[m]
                ax.plot(range(4), v, "o-", ms=3, label=a)
            ax.axhline(0, color="k", lw=0.4)
            ax.set_title(f"{st}: {m} tau {tau} ({win or _win(st)})", fontsize=8)
            ax.set_xticks(range(4))
            ax.set_xticklabels(EPS, fontsize=7)
    if empty:
        empty[0][0].legend(*axs[0, 0].get_legend_handles_labels(), loc="center", fontsize=9, frameon=False,
                           title=f"animal\n({empty[0][1]})")
    else:
        fig.legend(*axs[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=4, fontsize=9,
                   frameon=False)
    fig.suptitle(_wrap("Orofacial pool (ALM+tjM1+SSp-m+SSp-n+SSs) contra vs ipsi: per-animal epoch means (session means; "
                       "noise corr in Fisher z)", 150), fontsize=9)
    fig.savefig(FIGS / "lick_interhemi_oro_animals.png", dpi=120)
    plt.close(fig)
    # 5 cross-correlation curves (oro, pooled over sessions per epoch, animal-weighted)
    f = FIGD / "lick_interhemi_xcorr.pkl"
    if f.exists():
        with open(f, "rb") as fh:
            CG = pickle.load(fh)
        C, grp = CG["curves"], CG["groups"]
        fig, axs = plt.subplots(len(TAUS), 5, figsize=(17, 8.5), sharex=True, layout="constrained")
        for i, tau in enumerate(TAUS):
            for j, st in enumerate(SETS):
                ax = axs[i, j]
                w = _win(st)
                L = np.arange(-MAXLW[w], MAXLW[w] + 1) * DT
                for ep in EPS:
                    cs = [np.mean([C[(lab, st, w, "oro", tau)] for lab in grp.get((a, ep), ())
                                   if (lab, st, w, "oro", tau) in C], 0) for a in AN]
                    cs = [c for c in cs if np.ndim(c) and np.all(np.isfinite(c))]
                    if cs:
                        ax.plot(L, np.mean(cs, 0), color=COL[ep], label=ep)
                ax.axvline(0, color="k", lw=0.4)
                ax.set_title(f"{st} tau {tau} ({w})", fontsize=8)
        axs[-1, 2].set_xlabel("lag (s); + = contralesional leads")
        fig.legend(*axs[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=4, fontsize=9,
                   frameon=False)
        fig.suptitle(_wrap("Orofacial pool: within-event cross-correlation of residual traces corr(c(t), i(t+L)), "
                           "mean over animals of session means", 150), fontsize=9)
        fig.savefig(FIGS / "lick_interhemi_xcorr.png", dpi=130)
        plt.close(fig)
    # 6 recovery
    Rc = D["Rc"]
    fig, axs = plt.subplots(2, 1, figsize=(16, 9), layout="constrained")
    qs = {}
    for ax, tgt in zip(axs, ("aim", "share")):
        q = Rc[(Rc.target == tgt) & (Rc.pair == "oro") & (Rc.win == Rc.set.map(_win))
               & (((Rc.tau == 0) & Rc.metric.isin(["nc_post", "nc_pre", "nc_win", "cc0"]))
                  | ((Rc.tau == 0.6) & Rc.metric.isin(["asym", "gc_dir"])))]
        q = q.reset_index(drop=True)
        qs[tgt] = q
        for k, a in enumerate(AN):
            ax.scatter(np.arange(len(q)) + (k - 1.5) * 0.12, q[f"r_{a}"], s=12, label=a)
        ax.scatter(np.arange(len(q)), q.r_pooled_within, marker="_", s=200, color="k", label="pooled within")
        ax.set_xticks(range(len(q)))
        ax.set_xticklabels([f"{s} {m}" for s, m in zip(q.set, q.metric)], fontsize=7, rotation=90)
        ax.set_xlim(-0.7, len(q) - 0.3)
        for xb in np.flatnonzero(q.set.to_numpy()[1:] != q.set.to_numpy()[:-1]) + 0.5:   # separate the event sets
            ax.axvline(xb, color="0.85", lw=0.6, zorder=0)
        ax.axhline(0, color="k", lw=0.4)
        ax.set_ylabel(f"partial r with {tgt}\n(| log day)")
    fig.legend(*axs[0].get_legend_handles_labels(), loc="outside lower center", fontsize=8, ncol=5, frameon=False)
    fig.suptitle(_wrap("Orofacial pool coupling metrics vs recovery, within animal (post sessions, log(day) partialled): "
                       "aim = far-contra cued aim (deg, - = ipsilesional); share = cued matched-error within-spout balance",
                       150), fontsize=9)
    fig.savefig(FIGS / "lick_interhemi_recovery.png", dpi=130)
    plt.close(fig)
    _recovery_simple(plt, qs)


def _recovery_simple(plt, qs: dict) -> None:
    """Readable summary of figure 6: pooled within-animal partial r as a set x metric heatmap (same numbers)."""
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.2), layout="constrained")
    for ax, (tgt, q) in zip(axs, qs.items()):
        sets = list(dict.fromkeys(q.set))
        mets = list(dict.fromkeys(q.metric))

        def piv(col, q=q, sets=sets, mets=mets):
            return q.pivot_table(index="set", columns="metric", values=col).reindex(index=sets, columns=mets)
        R, P, N, NA = piv("r_pooled_within"), piv("p_perm"), piv("n_same_sign"), piv("n_animals")
        im = ax.imshow(R.to_numpy(float), cmap="RdBu_r", vmin=-0.4, vmax=0.4, aspect="auto")
        for i in range(len(sets)):
            for j in range(len(mets)):
                r = R.iloc[i, j]
                if np.isfinite(r):
                    star = "*" if P.iloc[i, j] < 0.05 else ""
                    ax.text(j, i, f"{r:+.2f}{star}\n{int(N.iloc[i, j])}/{int(NA.iloc[i, j])}", ha="center",
                            va="center", fontsize=8)
        ax.set_xticks(range(len(mets)), mets, fontsize=9)
        ax.set_yticks(range(len(sets)), sets, fontsize=9)
        ax.set_title(f"partial r with {tgt} (| log day), pooled within animal", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.05).set_label("pooled within-animal r")
    fig.suptitle(_wrap("SIMPLE VIEW of the recovery figure: orofacial-pool coupling metric vs recovery, within animal "
                       "(post sessions, log(day) partialled). Cell = pooled within-animal r; * = permutation p < 0.05 "
                       "(uncorrected); k/n = animals with the same sign. aim = far-contra cued aim; share = cued "
                       "matched-error within-spout balance. Blank = metric not defined for that set.", 130), fontsize=9)
    fig.savefig(FIGS / "lick_interhemi_recovery_simple.png", dpi=130)
    plt.close(fig)


SIMPLE_PAIRS = [("oro", "orofacial\npool"), ("VISpm", "VISpm\n(control)"), ("RSPd", "RSPd\n(control)")]
SIMPLE_CTX = [("cued", "CUED licks"), ("rest", "DOCKED QUIET REST (no licks)"),
              ("cued-rest", "LICK-SPECIFIC: cued licks minus rest")]


def plot_simple() -> None:
    """ONE figure, key points only (Priya 10-29): interhemispheric noise correlation (same 1-s window, w1, raw), change
    vs pre, orofacial pool vs control pairs, during cued licks / at docked rest / lick minus rest; per-animal dots,
    standard epoch colours; plus one small panel for the recovery link. From the saved tables only."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from wfield_local.orofacial_cohort import ANIMAL_MARKER, EPOCH_LINE
    with open(FIGD / "lick_interhemi_tables.pkl", "rb") as fh:
        D = pickle.load(fh)
    T, Rc = D["T"], D["Rc"]
    t = T[(T.metric == "nc_win") & (T.tau == 0) & (T.win == "w1")]
    fig = plt.figure(figsize=(16, 5.6), layout="constrained")
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.8])
    axs = [fig.add_subplot(gs[0, k]) for k in range(3)]
    for ax in axs[1:]:
        ax.sharey(axs[0])
    x = np.arange(len(SIMPLE_PAIRS))
    for ax, (st, ttl) in zip(axs, SIMPLE_CTX):
        ax.axvspan(0.5, len(SIMPLE_PAIRS) - 0.5, color="0.94", zorder=0)
        for k, ep in enumerate(EPS[1:]):
            for xi, (p, _l) in zip(x, SIMPLE_PAIRS):
                r = t[(t.set == st) & (t.pair == p) & (t.epoch == ep)]
                if r.empty:
                    continue
                r = r.iloc[0]
                xx = xi + (k - 1) * 0.26
                pa = [float(v) for v in str(r.per_animal).split()]
                if len(pa) != len(AN):              # names are not stored; only draw when all four are present
                    pa = []
                for an, v in zip(AN, pa):            # boot_delta per-animal list is over sorted animals (all 4 here)
                    ax.plot(xx - 0.07, v, ANIMAL_MARKER[an], mfc="none", mec=EPOCH_LINE[ep], ms=4, alpha=0.75)
                ax.errorbar(xx + 0.05, r.delta, yerr=[[r.delta - r.lo], [r.hi - r.delta]], fmt="o",
                            color=EPOCH_LINE[ep], ms=5.5, lw=1.6)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(x, [lb for _p, lb in SIMPLE_PAIRS], fontsize=8.5)
        ax.set_xlim(-0.5, len(SIMPLE_PAIRS) - 0.5)
        ax.set_title(ttl, fontsize=9.5)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    axs[0].set_ylabel("change vs pre in interhemispheric noise correlation\n(contra vs ipsi homotopic, Fisher z; "
                      "+ = more coupled)", fontsize=8.5)
    # recovery link: pooled within-animal partial r (| log day) of the orofacial-pool coupling with cued far-contra aim
    # and with the cued contralesional share, during cued licks and at rest
    ax = fig.add_subplot(gs[0, 3])
    rows = [(st, tg) for st in ("cued", "rest") for tg in ("aim", "share")]
    q = Rc[(Rc.pair == "oro") & (Rc.metric == "nc_win") & (Rc.tau == 0) & (Rc.win == "w1")]
    for i, (st, tg) in enumerate(rows):
        r = q[(q.set == st) & (q.target == tg)]
        if r.empty:
            continue
        r = r.iloc[0]
        for an in AN:
            ax.plot(r[f"r_{an}"], i, ANIMAL_MARKER[an], mfc="none", mec="0.35", ms=5)
        ax.plot(r.r_pooled_within, i, "|", color="k", ms=16, mew=2.2)
        ax.text(0.98, i + 0.28, f"r = {r.r_pooled_within:+.2f}, p = {r.p_perm:.2f}", fontsize=7.5, ha="right",
                transform=ax.get_yaxis_transform())
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlim(-0.75, 0.75)
    ax.set_ylim(len(rows) - 0.4, -0.6)
    ax.set_yticks(range(len(rows)), [f"{'cued' if st == 'cued' else 'rest'} coupling\nvs {'far-contra aim' if tg == 'aim' else 'contra share'}"
                                     for st, tg in rows], fontsize=8)
    ax.set_xlabel("within-animal partial r (| log day), post sessions", fontsize=8)
    ax.set_title("RECOVERY LINK (orofacial pool)", fontsize=9.5)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    hd = [Line2D([], [], marker="o", ls="-", color=EPOCH_LINE[ep], ms=5.5) for ep in EPS[1:]]
    hd += [Line2D([], [], marker=ANIMAL_MARKER[a], ls="", mfc="none", color="0.4", ms=5) for a in AN]
    hd += [Line2D([], [], marker="|", ls="", color="k", ms=12, mew=2)]
    fig.legend(hd, [f"{ep} - pre (pooled, 95% CI)" for ep in EPS[1:]] + AN + ["pooled within-animal r"],
               loc="outside lower center", ncol=8, fontsize=8.5, frameon=False)
    pre = t[(t.set == "cued") & (t.pair == "oro")].pre_level.iloc[0]
    fig.suptitle(_wrap("Interhemispheric coupling over recovery: trial-to-trial (noise) correlation between homotopic "
                       "contra- and ipsilesional areas (residuals after the spout / angle mean; the same -0.5..+0.5 s "
                       "window for licks and rest). At quiet REST coupling rises in the orofacial pool AND in the "
                       "control areas (controls from acute on); during cued licks the orofacial rise (chronic) is no "
                       "larger than at rest, and lick minus rest is flat (orofacial) or falls (controls) -> a GLOBAL, "
                       "state-wide increase, not a lick- or orofacial-specific one. Orofacial pre level z = "
                       f"{pre:.2f} (r = {np.tanh(pre):.2f}). Grey band = control areas. Right: the orofacial coupling "
                       "does not track behavioural recovery. PROVISIONAL (n = 4)", 200), fontsize=9.5)
    fig.savefig(FIGS / "lick_interhemi_simple.png", dpi=120)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--extract-rest", action="store_true")
    ap.add_argument("--summarise", action="store_true", help="from interhemi_sessions.csv.gz")
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--simple-only", action="store_true", help="only lick_interhemi_simple.png, from saved tables")
    ap.add_argument("--labels", nargs="*")
    a = ap.parse_args()
    from wfield_local import analysis_kit as ak
    jobs = min(a.jobs, 4)
    t0 = time.time()
    if a.simple_only:
        plot_simple()
        return
    if a.plot_only:
        plot_all()
        plot_simple()
        return
    labs = a.labels or il.labels()
    if a.extract_rest:
        res, fail = ak.fan_sessions(labs, rest_extract, jobs=jobs)
        C = pd.DataFrame([v for _k, v in res])
        REST_DIR.mkdir(parents=True, exist_ok=True)
        if not a.labels:
            C.to_csv(OUT_W / "interhemi_rest_census.csv", index=False)
        print(C.describe(), f"\nfailed: {fail}", f"\n{time.time() - t0:.0f} s", flush=True)
        return
    if not a.summarise:
        M = _meta()
        items = [Item((x, M.get(x, pd.DataFrame(columns=["tk", "reach_mm", "gap", "since_stop", "to_strobe"])))) for x in labs]
        res, fail = ak.fan_sessions(items, session_metrics, jobs=jobs, key=lambda it: it[0])
        if fail:
            print(f"FAILED: {[(f[0][0] if isinstance(f[0], tuple) else f) for f in fail][:5]}", flush=True)
            if not a.labels:
                raise SystemExit("refusing to write a partial table")
        S = pd.concat([v[0] for _k, v in res], ignore_index=True)
        curves = {}
        for (lab, _m), (_df, cv) in res:
            for k, c in cv.items():
                curves[(lab, *k)] = c
        meta = S.drop_duplicates("label")[["label", "animal", "epoch"]]
        groups = {(an, ep): tuple(g.label) for (an, ep), g in meta.groupby(["animal", "epoch"])}
        tag = "" if not a.labels else "_test"
        S.to_csv(OUT_W / f"interhemi_sessions{tag}.csv.gz", index=False)
        if a.labels:
            print(S[(S.pair == "oro")].groupby(["set", "win", "tau"])[METRICS + ["n"]].mean().round(3).to_string())
            print(f"{time.time() - t0:.0f} s")
            return
        FIGD.mkdir(parents=True, exist_ok=True)
        with open(FIGD / "lick_interhemi_xcorr.pkl", "wb") as fh:
            pickle.dump({"curves": curves, "groups": groups}, fh)
        print(f"sessions done {time.time() - t0:.0f} s", flush=True)
    S = pd.read_csv(OUT_W / "interhemi_sessions.csv.gz")
    summarise(S)
    figures_compute()
    plot_all()
    plot_simple()
    print(f"total {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
