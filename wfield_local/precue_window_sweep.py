"""Pre-cue window shape and length, on DETRENDED data. (Priya, 2026-08-13)

The first window sweep was run on the current pipeline's SVTcorr and found the last 1.0-1.5 s carrying
~2x the position information of the first 1.0 s -- which reads as "the plan builds toward the cue" but
is also exactly the profile of the zero-phase filter's backward shadow (largest adjacent to the event,
decaying away). That sweep therefore measured the shadow's shape, not the plan's, and its conclusion
was withdrawn rather than acted on.

This repeats it on rebuilt SVTcorr so the question can actually be answered:

    rolling      4 x 0.5 s sub-bins concatenated -- a TIME COURSE over the window, not one number
    mean2.0      the pipeline's current feature: one mean over 2.0 s ending at the cue
    mean1.5      one mean over the last 1.5 s
    mean1.0      one mean over the last 1.0 s
    first1.0     the FIRST 1.0 s of the 2 s window -- the complement, and the arm that makes the
                 sweep interpretable rather than a tuning exercise

`first1.0` vs `mean1.0` is the comparison that matters: SAME duration, same trials, same folds, so
noise-averaging is matched and any difference is about WHERE in the window the information sits. If
the asymmetry was the filter shadow it should now be much smaller.

Features are Allen-ROI throughout, because LocaNMF components are fitted to SVTcorr and would need a
GPU refit per variant. The zerophase arm is run in the SAME ROI basis so the comparison is matched.

C is chosen by NESTED CV, since `rolling` has 4x the features of a mean and a fixed C would hand the
comparison to whichever arm it happened to suit.
"""
from __future__ import annotations

import argparse
import glob

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from wfield_local import config
from wfield_local.behavior_position import classify_cues_with_backup
from wfield_local.filter_acausality_test import (
    FS, FUNC, MASK_SPEC, fit_mask, roi_signal, svtcorr,
)
from wfield_local.locanmf_crossanimal_dff import _frames
from wfield_local.locanmf_cue_lick_analysis import SESSIONS
from wfield_local.plot_lick_aligned_averages import DISPLAY_ORDER, _load_daq_events
from wfield_local.plot_spout_trial_averages import _load_daq_events as _load_cue

CS = [0.02, 0.1, 0.5, 2.0]
WIN_S = 2.0
ARMS = [("rolling", ("bins", 4)), ("mean2.0", ("last", 2.0)), ("mean1.5", ("last", 1.5)),
        ("mean1.0", ("last", 1.0)), ("first1.0", ("first", 1.0))]


def featurize(W, spec):
    kind, val = spec
    wn = W.shape[2]
    if kind == "last":
        return W[:, :, max(0, wn - int(round(val * FS))):].mean(2)
    if kind == "first":
        return W[:, :, :max(1, int(round(val * FS)))].mean(2)
    e = np.linspace(0, wn, int(val) + 1).astype(int)
    return np.concatenate([W[:, :, a:b].mean(2) for a, b in zip(e[:-1], e[1:])], axis=1)


def windows(s, sig):
    """(n_trials, nROI, win_n) raw pre-cue traces, engaged trials, window ENDING at the cue."""
    cue = _load_cue(s["h5"])
    lk = _load_daq_events(s["h5"], "lick_analog", 2.5, 1.0, (0.001, 0.020), 0.10)
    cue_f, lick_f, _ = _frames(s, cue, lk)
    codes = classify_cues_with_backup(s, cue, verbose=False)
    ls = np.sort(np.asarray(lick_f))
    j = np.searchsorted(ls, cue_f, side="right")
    first = np.where(j < ls.size, ls[np.clip(j, 0, ls.size - 1)], -1)
    rt = first - cue_f
    maxrt = 2.0 * FS
    n, T = int(round(WIN_S * FS)), sig.shape[1]

    blk, b, prev = np.full(cue_f.size, -1, int), -1, None
    for k in range(cue_f.size):
        if codes[k] < 0:
            continue
        if prev is None or codes[k] != prev:
            b += 1
        blk[k] = b
        prev = int(codes[k])

    X, y, g = [], [], []
    for k in range(cue_f.size):
        if codes[k] < 0 or not (first[k] > 0 and 0 < rt[k] <= maxrt):
            continue                                   # engaged trials only, as the pipeline does
        c = int(cue_f[k])
        if c - n < 0 or c > T:
            continue
        X.append(sig[:, c - n:c])
        y.append(int(codes[k]))
        g.append(int(blk[k]))
    return np.asarray(X), np.asarray(y), np.asarray(g)


def score(X, y, g):
    ng = min(5, int(np.unique(g).size))
    if ng < 2 or len(y) < 60 or len(np.unique(y)) < len(DISPLAY_ORDER):
        return np.nan
    pred = np.empty_like(y)
    for tr, te in GroupKFold(ng).split(X, y, g):
        ni = min(4, int(np.unique(g[tr]).size))
        gs = GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
                          {"logisticregression__C": CS}, cv=GroupKFold(ni), n_jobs=1)
        gs.fit(X[tr], y[tr], groups=g[tr])
        pred[te] = gs.predict(X[te])
    return float(accuracy_score(y, pred))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="zerophase,strobedetrend")
    ap.add_argument("--from", dest="from_dates", default=None)
    ap.add_argument("--animals", nargs="+", default=None)
    a = ap.parse_args()

    modes = a.modes.split(",")
    dates = (set(config.expand_dates(a.from_dates, width=4)) if a.from_dates
             else set(config.curated_dates()))
    only = config.normalize_animals(a.animals) or sorted({x["label"][:4] for x in SESSIONS})
    labs = [x["label"] for x in SESSIONS if x["label"][:4] in set(only) and x["label"][-4:] in dates]

    rows = {m: [] for m in modes}
    for lab in labs:
        s = next(x for x in SESSIONS if x["label"] == lab)
        res = f"{s['mc']}/wfield_local_results"
        ad = glob.glob(f"{res}/allen_aligned_affine8v1")
        if not ad:
            continue
        try:
            svt = np.load(f"{res}/SVT.npy")
            T = np.load(f"{res}/T.npy").astype(np.float64)
        except Exception as ex:                                      # noqa: BLE001
            print(f"  !! {lab}: {type(ex).__name__} {str(ex)[:50]}", flush=True)
            continue
        cue = _load_cue(s["h5"])
        lk = _load_daq_events(s["h5"], "lick_analog", 2.5, 1.0, (0.001, 0.020), 0.10)
        _c, _l, csmp = _frames(s, cue, lk)
        for m in modes:
            msk = None
            if m in MASK_SPEC:
                if csmp is None:
                    continue
                msk, _d = fit_mask(s, svt[:, FUNC::2].shape[1], csmp, cue, **MASK_SPEC[m])
            sig, _regs = roi_signal(ad[0], svtcorr(svt, T, m, mask=msk))
            W, y, g = windows(s, sig)
            del sig
            if not len(y):
                continue
            r = {nm: score(featurize(W, sp), y, g) for nm, sp in ARMS}
            r["label"] = lab
            rows[m].append(r)
            print(f"  {lab:12s} {m:14s} n={len(y):4d}  "
                  + "  ".join(f"{nm}={r[nm]:.3f}" for nm, _ in ARMS), flush=True)

    for m in modes:
        rr = rows[m]
        if not rr:
            continue
        print(f"\n=== {m}: {len(rr)} sessions (chance 0.167) ===")
        base = np.array([x["mean2.0"] for x in rr], float)
        for nm, _ in ARMS:
            v = np.array([x[nm] for x in rr], float)
            d = v - base
            ok = np.isfinite(d)
            print(f"  {nm:9s} {np.nanmean(v):.3f}   vs mean2.0 {np.nanmean(d):+.3f}  "
                  f"(better in {int((d[ok] > 0).sum())}/{int(ok.sum())})")
        l1 = np.array([x["mean1.0"] for x in rr], float)
        f1 = np.array([x["first1.0"] for x in rr], float)
        print(f"  ASYMMETRY last1.0 - first1.0 = {np.nanmean(l1 - f1):+.3f}   "
              f"(the filter-shadow signature; matched duration, so noise-averaging is equal)")


if __name__ == "__main__":
    main()
