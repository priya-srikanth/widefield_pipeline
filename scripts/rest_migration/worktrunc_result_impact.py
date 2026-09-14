"""THE DECIDING MEASUREMENT: does stopped-tail contamination move a WITHIN-WORKING RESULT?

`stopped_tail_contamination` showed the global order-10 fit is pulled by a disengaged tail, changing
the subtracted TREND inside the engaged period by a median 16.3% of the signal. That is a statistic
about the trend, NOT about any result -- the decoder standardises per fold and reads trial-window
means, so a smooth slow difference is partly absorbed, while map AMPLITUDES would feel it more.

This rebuilds `SVTcorr` two ways for the same session and re-runs the same within-working analyses:

  PRODUCTION   drift fitted on the FULL record (what the pipeline does)
  WORKTRUNC    drift fitted on the WORKING PERIOD ALONE, then HELD CONSTANT past the last engaged
               trial -- never extrapolated, because an order-10 polynomial run past its fitted range
               diverges, and session-level consumers (`hemispheric_dynamics`, `crossday_intensity`,
               the photobleach panels) read the whole record.

NOTHING IS WRITTEN. Both products are built in memory and fed to `filter_acausality_test.roi_signal`
/ `.decode`, the harness built for exactly this kind of head-to-head. Allen-ROI features are used
deliberately: a LocaNMF source would need the decomposition refitted per variant, which would change
the basis as well as the data and confound the comparison.

Reported per session: position decode accuracy (cue- and lick-aligned) and the per-position map
amplitude above the `restw` baseline.

    python -m scripts.rest_migration.worktrunc_result_impact --sessions PS94_0819 PS93_0819
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np

from wfield_local import config
from wfield_local.hemo_variants import FS, FUNC, VARIANTS, remove_drift

VARIANT = "meegkit_hpfit"

#: ALIGNMENTS, and "precue" is here because the ADOPTION DECIDED ON IT. `filter_acausality_test.
#: analyse_session` reports `precue = decode(..., "precue")` and `postcue = decode(..., "cue")`, so a
#: harness that measures only "cue" and "lick" reproduces the head-to-head's POSTCUE column and never
#: touches the PRE-CUE one -- which is both the column where the windowed median lost in August
#: (0.306 against 0.352) and the headline scientific finding. Measuring the other two and declaring a
#: winner would answer a question nobody asked.
ALIGNMENTS = ("precue", "cue", "lick")

#: Target 50% cutoff (min) for the FASTER arm -- the "yellow line" timescale Priya sketched on
#: PS94_0819, roughly 8-12 min per wiggle. Production sits at ~43 min for a 150 min session.
TARGET_CUTOFF_MIN = 13.0


def _last_engaged_frame(label, n):
    import pandas as pd

    an, mmdd = label.split("_")
    root = Path(config.resolver().resolve("behavior_out", "")) / "sessions" / an / f"2026{mmdd}"
    hits = glob.glob(str(root / "*_trials.csv"))
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


def _detrend(x, mask, k, order=None, variant=None, win_s=None):
    """Drift-removed x. ``k=None`` fits the full record; otherwise fits [0,k) and HOLDS the trend
    constant past k rather than extrapolating the polynomial.

    ``variant`` selects the DRIFT ESTIMATOR: `meegkit_hpfit` is the adopted global polynomial,
    `detrend_hpfit` the masked WINDOWED MEDIAN (same mask, same high-passed T). Both hybrids, so the
    only thing that differs between them is how the trend is estimated.
    """
    v = variant or VARIANT
    kw = dict(order=order) if order is not None else {}
    if win_s is not None:
        kw["win_s"] = float(win_s)
    if k is None:
        return remove_drift(x, v, mask, **kw)
    trend = x[:, :k] - remove_drift(x[:, :k].copy(), v, mask[:k].copy(), **kw)
    pad = np.repeat(trend[:, -1:], x.shape[1] - k, axis=1)
    return x - np.concatenate([trend, pad], axis=1)


def build(s, svt, mask, T, k, order=None, variant=None, win_s=None):
    from wfield_local.filter_acausality_test import LP, _lp

    a = _detrend(svt[:, FUNC::2].astype(np.float64), mask, k, order, variant, win_s)
    b = _detrend(svt[:, (FUNC + 1) % 2::2].astype(np.float64), mask, k, order, variant, win_s)
    if LP < FS / 2:
        b = _lp(b)
    a = (a.T - np.nanmean(a, 1)).T
    b = (b.T - np.nanmean(b, 1)).T
    c = a - T @ b
    return (c.T - np.nanmean(c, 1)).T.astype(np.float32)


def _restw_amplitudes(s, svtc):
    """Per-position map amplitude (RMS across components) above the restw baseline, or None.

    RMS across components IS the spatial RMS of the map: U is orthonormal, so the component-space
    norm equals the pixel-space norm. No U load, and no dependence on which basis is used.
    """
    from wfield_local.locanmf_position_encoder import _engaged_frames
    from wfield_local.rest_by_position import rest_frames_by_position, restw_from_frames

    n = svtc.shape[1]
    per, info = rest_frames_by_position(s, n, docked=False)
    if info.get("error"):
        return None, info["error"]
    base, used = restw_from_frames(svtc, per, label=s["label"], verbose=False)
    if base is None:
        return None, f"only {len(used)}/6 positions"
    fr, y, post_n = _engaged_frames(s)
    out = {}
    for p in sorted(set(int(v) for v in y)):
        f = fr[y == p]
        f = f[f + post_n <= n]
        if f.size < 5:
            continue
        m = np.mean([svtc[:, i:i + post_n].mean(1) for i in f], axis=0) - base.reshape(-1)
        out[p] = float(np.sqrt((m ** 2).mean()))
    return out, None


def run(label):
    from wfield_local.filter_acausality_test import decode, patterns, roi_signal

    s = next(x for x in config.load_sessions() if x["label"] == label)
    res = Path(s["mc"]) / "wfield_local_results"
    allen = glob.glob(f"{res}/allen_aligned_affine8v1")[0]
    svt = np.load(res / "SVT.npy")
    n = svt[:, FUNC::2].shape[1]
    k = _last_engaged_frame(label, n)
    mask = _mask_for(s, n)
    if k is None or mask is None:
        print(f"  !! {label}: no engaged trials or no fit mask", flush=True)
        return None
    T = np.load(res / "T.npy").astype(np.float64)
    print(f"=== {label} ===  {n/FS/60:.1f} min, stops at {k/FS/60:.1f} min "
          f"(tail {(n-k)/FS/60:.1f} min)", flush=True)

    # THE CONFOUND CONTROL, and it is essential. WORKTRUNC is not only de-contaminated -- fitting
    # order 10 over a SHORTER record also makes the estimator FASTER (cutoff scales as
    # duration/order). For PS94_0819 that is ~13 min against the full record's ~30 min, so any gain
    # could be kinetics rather than de-contamination. MATCHED runs the FULL record at the order that
    # reproduces WORKTRUNC's effective cutoff, isolating the one variable.
    matched_order = int(round(10 * n / k))
    # A FIXED-CUTOFF arm, so the KINETICS question can be asked on sessions with NO stopped tail --
    # where WORKTRUNC and PRODUCTION coincide and cannot separate anything. Order 10 over 150 min is
    # a 43 min cutoff and cutoff scales as duration/order, so the order reaching `target` is
    # 10 * (duration/150) * (43/target).
    dur_min = n / FS / 60.0
    faster_order = max(2, int(round(10 * (dur_min / 150.0) * (43.0 / TARGET_CUTOFF_MIN))))
    print(f"  matched-cutoff order {matched_order} (worktrunc is order 10 over {100*k/n:.0f}% of the "
          f"record);  fixed-{TARGET_CUTOFF_MIN:.0f}min order {faster_order}", flush=True)

    # ARMS. (tag, truncate_frame, order, drift_variant, window_s)
    #
    # THE WINDOWED ARMS ARE THE POINT OF THIS VERSION. A masked windowed median only sees +/-W/2
    # around each sample, so a stopped tail CANNOT reach the working period -- which dissolves the
    # whole piecewise scheme: no truncation, no second product, no two-detrends-that-must-never-meet,
    # no order-scaling trap. Its cutoff is also set in SECONDS, so it is identical on a 95 min and a
    # 154 min session, where a fixed-order polynomial's is not (~1.6x spread across this cohort).
    #
    # `detrend_hpfit` LOST the 2026-08-13 head-to-head (pre-cue 0.306 vs 0.352). But that ran on 36
    # curated sessions that were essentially all PRE-STROKE, where long stopped tails barely exist --
    # pre-stroke tails top out at 23.8 min against 30-59 min post-stroke, because quitting early IS
    # the phenotype. So the comparison was decided on data where the polynomial's weakness could not
    # show. That is what this re-runs.
    arms = [("PRODUCTION", None, None, "meegkit_hpfit", None)]
    if k < int(0.97 * n):
        arms.append(("WORKTRUNC", k, None, "meegkit_hpfit", None))
    arms.append(("WIN600", None, None, "detrend_hpfit", 600.0))
    arms.append(("WIN300", None, None, "detrend_hpfit", 300.0))

    out = {}
    for tag, kk, oo, vv, ww in arms:
        print(f"  building {tag} ...", flush=True)
        c = build(s, svt, mask, T, kk, oo, vv, ww)
        sig, regs = roi_signal(allen, c)
        accs = {}
        for al in ALIGNMENTS:
            try:
                accs[al] = decode(s, sig, regs, al)
            except Exception as ex:                                   # noqa: BLE001
                print(f"    decode {al}: {type(ex).__name__} {str(ex)[:60]}", flush=True)
                accs[al] = np.nan
        amps, err = _restw_amplitudes(s, c)
        # THE SMEARING GUARD, not optional. Any candidate could "win" on accuracy by reintroducing
        # the zerophase artefact, so the shadow signature is measured on every arm: corr(pre-cue
        # pattern, post-cue pattern) across positions. zerophase gives -0.483 and is NEGATIVE in
        # 30/36 sessions -- a pre-cue pattern that is an inverted copy of the movement response,
        # which is what backwards smearing looks like. detrend_hpfit gave +0.525 and 0/36, the
        # CLEANEST of anything tested; meegkit_hpfit +0.570 and 2/36. A negative value here
        # disqualifies an arm regardless of its decode accuracy.
        try:
            shadow, pre_level = patterns(s, sig)
        except Exception as ex:                                       # noqa: BLE001
            print(f"    shadow test: {type(ex).__name__} {str(ex)[:50]}", flush=True)
            shadow, pre_level = np.nan, np.nan
        out[tag] = (accs, amps, err, shadow, pre_level)
        print(f"    shadow corr(pre,post) = {shadow:+.4f}"
              + ("   *** NEGATIVE -- smearing signature ***" if shadow < 0 else "   (ok)"),
              flush=True)
        astr = "  ".join(f"{a}={v:.4f}" for a, v in accs.items())
        print(f"    decode: {astr}"
              + (f"   restw amps: {err}" if err else
                 "   restw amps: " + " ".join(f"{p}:{v:.5f}" for p, v in sorted(amps.items()))),
              flush=True)

    ap, mp = out["PRODUCTION"][0], out["PRODUCTION"][1]
    print("  DECODE vs PRODUCTION:", flush=True)
    for al in ALIGNMENTS:
        line = f"    {al:4s}  prod {ap[al]:.4f}"
        for tag, *_ in arms[1:]:
            av = out[tag][0][al]
            line += f"   {tag} {av:.4f} ({av-ap[al]:+.4f})"
        print(line, flush=True)
    print("  SHADOW SIGNATURE corr(pre,post) -- negative = smearing, disqualifying:", flush=True)
    for tag, *_ in arms:
        print(f"    {tag:12s} {out[tag][3]:+.4f}", flush=True)
    for tag, *_ in arms[1:]:
        mw = out[tag][1]
        if not (mp and mw):
            continue
        common = sorted(set(mp) & set(mw))
        rel = [100 * (mw[p] - mp[p]) / mp[p] for p in common if mp[p] > 0]
        if rel:
            print(f"    map amplitude {tag}: mean {np.mean(rel):+.2f}%  "
                  f"max |{np.max(np.abs(rel)):.2f}|%  per-position "
                  + " ".join(f"{p}:{100*(mw[p]-mp[p])/max(mp[p],1e-12):+.1f}%" for p in common),
                  flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="+", required=True)
    a = ap.parse_args()
    for lab in a.sessions:
        try:
            run(lab)
        except Exception as ex:                                       # noqa: BLE001
            print(f"  !! {lab}: {type(ex).__name__} {ex}", flush=True)


if __name__ == "__main__":
    main()
