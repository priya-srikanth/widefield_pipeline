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

    python -m scripts.rest_migration.archive.onset_edge_bias --sessions PS94_0819 PS94_0810
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from wfield_local import config
from wfield_local.hemo_variants import FS, functional_channel, remove_drift

BINS_MIN = ((0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 5.0), (5.0, 10.0), (10.0, 20.0))




def run(label):
    from scripts.rest_migration.archive.rolling_detrend import robust_local_linear_trend, rolling_masked_trend
    from scripts.rest_migration.archive.worktrunc_result_impact import _mask_for
    from scripts.rest_migration.plot_session_residual import _brain_mean_op

    s = next(x for x in config.load_sessions() if x["label"] == label)
    res = Path(s["mc"]) / "wfield_local_results"
    svt = np.load(res / "SVT.npy")
    a = svt[:, functional_channel(s)::2].astype(np.float64)
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
