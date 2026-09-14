"""Does the STOPPED TAIL distort the drift fit inside the WORKING period?

THE RISK, and it is specific. The order-10 polynomial is a GLOBAL basis: its coefficients are set by
the whole record at once, so a tail full of large slow swings pulls the fit everywhere, including the
engaged period every analysis actually reads. In `PS94_0819` the fitted curve visibly wiggles at ~55,
65, 80 and 95 min -- all inside the stopped chunk. The question is whether those wiggles cost anything
BEFORE the stop.

THE TEST. Fit the drift twice on the same session: once on the FULL record (what production does) and
once on the WORKING PERIOD ALONE (truncated at the last engaged trial, so the tail cannot pull it).
Compare the two TRENDS inside the working period. If they agree, the tail is inert and the global fit
is safe. If they differ by an appreciable fraction of the signal, the tail is corrupting the analysed
period and the drift removal needs a local estimator rather than a faster global one.

NOTE THIS IS NOT ANSWERABLE BY RAISING THE POLYNOMIAL ORDER. Order 40 retains 0.22 at a 3000 s period
against order 10's 0.18 (hemo_variants) -- a global basis does not become local when you add terms.

    python -m scripts.rest_migration.stopped_tail_contamination --sessions PS94_0819 PS92_0821
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from wfield_local import config
from wfield_local.hemo_variants import FS, FUNC, remove_drift

VARIANT = "meegkit_hpfit"


def _last_engaged_frame(label, n):
    import glob as _g

    import pandas as pd

    an, mmdd = label.split("_")
    root = Path(config.resolver().resolve("behavior_out", "")) / "sessions" / an / f"2026{mmdd}"
    hits = _g.glob(str(root / "*_trials.csv"))
    if not hits:
        return None
    d = pd.read_csv(hits[0])
    if "engaged" not in d.columns or not len(d):
        return None
    eng = d[d["engaged"].astype(bool)]
    if not len(eng):
        return None
    return int(np.clip(float(eng["cue_s"].max()) * FS, 1, n - 1))


def _mask_for(s, n):
    from wfield_local.filter_acausality_test import MASK_SPEC, fit_mask
    from wfield_local.hemo_variants import VARIANTS
    from wfield_local.locanmf_crossanimal_dff import _frames as _fr
    from wfield_local.plot_lick_aligned_averages import _load_daq_events as _ll
    from wfield_local.plot_spout_trial_averages import _load_daq_events as _lc

    cue = _lc(s["h5"])
    lk = _ll(s["h5"], "lick_analog", 2.5, 1.0, (0.001, 0.020), 0.10)
    _c, _l, csmp = _fr(s, cue, lk)
    if csmp is None:
        return None
    m, _d = fit_mask(s, n, csmp, cue, **MASK_SPEC[VARIANTS[VARIANT]["mask"]])
    return m


def run(label):
    s = next(x for x in config.load_sessions() if x["label"] == label)
    res = Path(s["mc"]) / "wfield_local_results"
    svt = np.load(res / "SVT.npy")
    a = svt[:, FUNC::2].astype(np.float64)
    n = a.shape[1]
    k = _last_engaged_frame(label, n)
    if k is None:
        print(f"  !! {label}: no engaged trials")
        return
    mask = _mask_for(s, n)
    if mask is None:
        print(f"  !! {label}: no fit mask")
        return
    print(f"=== {label} ===  {n} frames ({n/FS/60:.1f} min), stops at {k/FS/60:.1f} min "
          f"(tail {(n-k)/FS/60:.1f} min)", flush=True)

    print("  fitting on the FULL record ...", flush=True)
    trend_full = a - remove_drift(a, VARIANT, mask)
    print("  fitting on the WORKING PERIOD ALONE ...", flush=True)
    trend_work = a[:, :k] - remove_drift(a[:, :k].copy(), VARIANT, mask[:k].copy())

    # compare the two trends inside the working period, per component then pooled
    d = trend_full[:, :k] - trend_work
    # centre: an overall constant offset between the two fits is absorbed downstream by the
    # zero-mean step, so it is NOT contamination -- only the SHAPE difference is.
    d = d - d.mean(1, keepdims=True)
    sig = a[:, :k] - a[:, :k].mean(1, keepdims=True)

    rms_d = float(np.sqrt((d ** 2).mean()))
    rms_sig = float(np.sqrt((sig ** 2).mean()))
    rms_trend = float(np.sqrt(((trend_work - trend_work.mean(1, keepdims=True)) ** 2).mean()))
    print(f"  trend SHAPE difference inside the working period:")
    print(f"    RMS difference        {rms_d:.6f}")
    print(f"    vs working signal RMS {rms_sig:.6f}   ->  {100*rms_d/rms_sig:.2f}% of the signal")
    print(f"    vs the trend's own RMS{rms_trend:.6f}   ->  {100*rms_d/max(rms_trend,1e-12):.2f}% "
          f"of the trend", flush=True)
    return rms_d / rms_sig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="+", required=True)
    a = ap.parse_args()
    out = []
    for lab in a.sessions:
        try:
            r = run(lab)
            if r is not None:
                out.append((lab, r))
        except Exception as ex:                                       # noqa: BLE001
            print(f"  !! {lab}: {type(ex).__name__} {ex}", flush=True)
    if out:
        print("\nSUMMARY -- trend shape difference as % of the working-period signal:")
        for lab, r in out:
            print(f"  {lab:12s} {100*r:6.2f}%")


if __name__ == "__main__":
    main()
