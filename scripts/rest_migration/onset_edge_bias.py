"""How badly does each drift estimator handle the EARLY BLEACHING ONSET?

THE ASYMMETRY (Priya, 2026-09-14). The polynomial is fitted globally, so it has data on both sides of
every point and no edge problem. A ROLLING estimator at t=0 has only the half-window [0, W/2], and a
MEDIAN over that half-window reports the middle of the first W/2 seconds -- a stretch over which the
signal is falling steeply, because the bleaching onset is the fastest, largest drift in the record
(and it is not always monotone: some sessions rise then fall). So the windowed family systematically
under-fits the onset and leaves it in the output.

This is a known property rather than an implementation quirk: LOCAL-CONSTANT estimators (median,
mean) carry O(h) boundary bias, LOCAL-LINEAR ones O(h^2). A median cannot extrapolate a slope; a local
linear fit can. So the principled fix is a local LINEAR estimator, at least near the edges -- not
reflection padding, which would fabricate a V out of a monotone decay, and not discarding the first
minutes, which throws away trials.

MEASURED HERE: the detrended residual's level in early bins, relative to the session's own median. A
good estimator leaves the early bins near zero; one that under-fits the onset leaves them high.

    python -m scripts.rest_migration.onset_edge_bias --sessions PS94_0819 PS94_0810
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from wfield_local import config
from wfield_local.hemo_variants import FS, FUNC, remove_drift

BINS_MIN = ((0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 5.0), (5.0, 10.0), (10.0, 20.0))


def robust_local_linear_trend(X, mask, win_s, stride_s=10.0, min_n=20, iters=2):
    """Rolling ROBUST LOCAL LINEAR trend -- the boundary-corrected counterpart of a rolling median.

    At each centre, fit value ~ a + b*(t - c) by least squares on the masked samples in the window,
    with `iters` rounds of MAD-based reweighting so a transient cannot drag the fit (the robustness
    the median was chosen for). Evaluating the fit AT the centre means that at an edge, where the
    window is one-sided, the slope carries the estimate to the boundary instead of the level being
    pinned to the middle of the available half-window.
    """
    X = np.asarray(X, float)
    K, T = X.shape
    m = np.asarray(mask, bool)
    m = m[:T] if m.size >= T else np.pad(m, (0, T - m.size))
    half = max(1, int(round(win_s * FS)) // 2)
    stride = max(1, int(round(stride_s * FS)))

    cs, vs = [], []
    for c in range(0, T, stride):
        a, b = max(0, c - half), min(T, c + half)
        idx = np.flatnonzero(m[a:b]) + a
        if idx.size < min_n:
            continue
        dt = (idx - c).astype(float) / FS
        A = np.stack([np.ones_like(dt), dt], 1)
        Y = X[:, idx]                                     # (K, n)
        w = np.ones_like(dt)
        beta = None
        for _ in range(iters + 1):
            Aw = A * w[:, None]
            beta, *_ = np.linalg.lstsq(Aw, (Y * w).T, rcond=None)    # (2, K)
            resid = Y - (A @ beta).T
            s = np.median(np.abs(resid - np.median(resid, 1, keepdims=True)), 1, keepdims=True)
            s = np.maximum(s * 1.4826, 1e-12)
            w = 1.0 / np.sqrt(1.0 + (np.median(np.abs(resid) / s, 0) / 3.0) ** 2)
        cs.append(float(c))
        vs.append(beta[0])                                 # intercept == value AT the centre
    if len(cs) < 2:
        return np.zeros_like(X)
    C = np.asarray(cs, float)
    V = np.stack(vs, 1)
    t = np.arange(T, dtype=float)
    return np.stack([np.interp(t, C, V[k]) for k in range(K)])


def run(label):
    from scripts.rest_migration.plot_session_residual import _brain_mean_op
    from scripts.rest_migration.rolling_detrend import rolling_masked_trend
    from scripts.rest_migration.worktrunc_result_impact import _mask_for

    s = next(x for x in config.load_sessions() if x["label"] == label)
    res = Path(s["mc"]) / "wfield_local_results"
    svt = np.load(res / "SVT.npy")
    a = svt[:, FUNC::2].astype(np.float64)
    n = a.shape[1]
    mask = _mask_for(s, n)
    if mask is None:
        print(f"  !! {label}: no fit mask", flush=True)
        return
    u_mean, _ = _brain_mean_op(s["mc"])
    raw = u_mean @ a
    t_min = np.arange(n) / FS / 60.0
    print(f"=== {label} ===  {t_min[-1]:.1f} min", flush=True)

    dets = {}
    print("  meegkit order 10 ...", flush=True)
    dets["meegkit"] = u_mean @ remove_drift(a.copy(), "meegkit_hpfit", mask)
    print("  ROLL300 (median) ...", flush=True)
    dets["ROLL300"] = raw - (u_mean @ rolling_masked_trend(a, mask, 300.0)[0])
    print("  LINEAR300 (robust local linear) ...", flush=True)
    dets["LIN300"] = raw - (u_mean @ robust_local_linear_trend(a, mask, 300.0))

    print(f"  RAW level by early bin (session median subtracted):", flush=True)
    base_raw = np.median(raw)
    for lo, hi in BINS_MIN:
        sl = (t_min >= lo) & (t_min < hi)
        if sl.sum():
            print(f"    {lo:4.1f}-{hi:4.1f} min  {np.median(raw[sl]) - base_raw:+.5f}", flush=True)

    print("  RESIDUAL after detrending -- near zero is good, large = onset left behind:", flush=True)
    hdr = "    bin (min)   " + "".join(f"{k:>12s}" for k in dets)
    print(hdr, flush=True)
    for lo, hi in BINS_MIN:
        sl = (t_min >= lo) & (t_min < hi)
        if not sl.sum():
            continue
        row = f"    {lo:4.1f}-{hi:4.1f}   "
        for k, v in dets.items():
            row += f"{np.median(v[sl]) - np.median(v):+12.5f}"
        print(row, flush=True)
    print("  |early residual| summed over the first 5 min (lower is better):", flush=True)
    early = (t_min < 5.0)
    for k, v in dets.items():
        c = v - np.median(v)
        print(f"    {k:10s} {np.abs(np.median(c[early])):.5f}   "
              f"(first 30 s: {np.median(c[t_min < 0.5]):+.5f})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="+", default=["PS94_0819", "PS94_0810"])
    a = ap.parse_args()
    for lab in a.sessions:
        try:
            run(lab)
        except Exception as ex:                                       # noqa: BLE001
            print(f"  !! {lab}: {type(ex).__name__} {ex}", flush=True)


if __name__ == "__main__":
    main()
