"""Does the polynomial's 50% cutoff really scale as duration/order? MEASURED, not assumed.

WHY IT MATTERS. A fixed-order polynomial is not a fixed filter: its basis functions live on the
record, so the shortest timescale it can track is set by duration/order. Sessions here run 95-154
min, so **the adopted variant's effective cutoff already varies ~1.6x across the cohort purely from
recording length** -- same nominal settings, different filter. Any fix that wants one timecourse for
every session must either use a WINDOWED estimator (window in seconds, length-independent by
construction) or SCALE THE ORDER per session as `order = c * duration / target`.

That formula is what `worktrunc_result_impact`'s FASTER arm depends on, and it had been asserted from
the shape of the basis rather than measured. This measures it.

METHOD. Take a real session's brain-mean 470 trace. For each (duration, order), add a pure sinusoid
of known period and amplitude, detrend with and without it, and recover the surviving sinusoid as
`detrend(x + s) - detrend(x)` -- the difference form, because meegkit's robust reweighting makes the
operator slightly non-linear and projecting the detrended sum directly would confound the drift with
the probe. Retention is the ratio of quadrature projections at that period.

    python -m scripts.rest_migration.cutoff_scaling --session PS92_0813
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from wfield_local import config
from wfield_local.hemo_variants import FS, FUNC, remove_drift

VARIANT = "meegkit_hpfit"
PERIODS_MIN = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0)


def _proj(x, period_n):
    """Quadrature amplitude of x at the given period (in samples)."""
    t = np.arange(x.size)
    c = np.cos(2 * np.pi * t / period_n)
    s = np.sin(2 * np.pi * t / period_n)
    return float(np.hypot(2 * (x * c).mean(), 2 * (x * s).mean()))


def retention(x, mask, order, period_min, amp):
    period_n = period_min * 60.0 * FS
    if period_n > 2 * x.size:                       # not even one cycle -- not estimable
        return np.nan
    probe = amp * np.sin(2 * np.pi * np.arange(x.size) / period_n)
    X = x[None, :]
    base = remove_drift(X.copy(), VARIANT, mask, order=order)[0]
    with_probe = remove_drift((X + probe[None, :]).copy(), VARIANT, mask, order=order)[0]
    return _proj(with_probe - base, period_n) / max(_proj(probe, period_n), 1e-12)


def half_cutoff(ret, periods):
    """Interpolate the period at which retention crosses 0.5 (log-period), or nan."""
    r = np.asarray(ret, float)
    p = np.asarray(periods, float)
    ok = np.isfinite(r)
    r, p = r[ok], p[ok]
    below = np.where(r < 0.5)[0]
    if not below.size or below[0] == 0:
        return np.nan
    i = below[0]
    lo, hi = r[i - 1], r[i]
    if abs(hi - lo) < 1e-12:
        return float(p[i])
    f = (lo - 0.5) / (lo - hi)
    return float(np.exp(np.log(p[i - 1]) + f * (np.log(p[i]) - np.log(p[i - 1]))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="PS92_0813")
    ap.add_argument("--orders", type=int, nargs="+", default=[5, 10, 20])
    ap.add_argument("--fracs", type=float, nargs="+", default=[1.0, 0.66, 0.5])
    a = ap.parse_args()

    from scripts.rest_migration.plot_session_residual import _brain_mean_op
    from scripts.rest_migration.worktrunc_result_impact import _mask_for

    s = next(x for x in config.load_sessions() if x["label"] == a.session)
    res = Path(s["mc"]) / "wfield_local_results"
    svt = np.load(res / "SVT.npy")
    aa = svt[:, FUNC::2].astype(np.float64)
    n_full = aa.shape[1]
    u_mean, _ = _brain_mean_op(s["mc"])
    trace_full = u_mean @ aa
    mask_full = _mask_for(s, n_full)
    amp = float(np.std(trace_full))
    print(f"{a.session}: {n_full/FS/60:.1f} min, probe amplitude {amp:.5f} (1 SD)\n", flush=True)

    rows = []
    for fr in a.fracs:
        n = int(n_full * fr)
        x, m = trace_full[:n].copy(), mask_full[:n].copy()
        dur = n / FS / 60.0
        for order in a.orders:
            ret = [retention(x, m, order, p, amp) for p in PERIODS_MIN]
            cut = half_cutoff(ret, PERIODS_MIN)
            pred = dur / order
            rows.append((dur, order, cut, pred))
            rs = "  ".join(f"{p:g}:{v:.2f}" if np.isfinite(v) else f"{p:g}:--"
                           for p, v in zip(PERIODS_MIN, ret))
            print(f"  dur {dur:6.1f} min  order {order:3d}   50% cutoff "
                  + (f"{cut:6.2f} min" if np.isfinite(cut) else "   >max ")
                  + f"   duration/order {pred:6.2f}", flush=True)
            print(f"      retention by period (min): {rs}", flush=True)

    ok = [(d, o, c, p) for d, o, c, p in rows if np.isfinite(c)]
    if len(ok) >= 2:
        c = np.array([r[2] for r in ok])
        p = np.array([r[3] for r in ok])
        k = float((c * p).sum() / (p * p).sum())          # best-fit cutoff = k * duration/order
        resid = c - k * p
        print(f"\nBEST FIT  cutoff = {k:.2f} x (duration/order)")
        print(f"  r = {np.corrcoef(c, p)[0,1]:.4f}   max |residual| {np.abs(resid).max():.2f} min "
              f"({100*np.abs(resid).max()/c.mean():.1f}% of the mean cutoff)")
        print(f"  so for a TARGET cutoff C on a session of duration D:  order = {k:.2f} * D / C")
        for D in (95.0, 120.0, 154.0):
            print(f"    D={D:5.1f} min -> order {k*D/43.0:5.1f} for 43 min, "
                  f"{k*D/13.0:5.1f} for 13 min")


if __name__ == "__main__":
    main()
