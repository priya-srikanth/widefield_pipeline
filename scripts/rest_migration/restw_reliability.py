"""SPLIT-HALF RELIABILITY of `rest` vs `restw` -- is equal weighting normalising to thin data?

THE CONCERN (Priya, 2026-09-14): "I worry normalizing to thin data will do more harm than good."

It is quantifiable and the frame counts understate it. `restw` takes a MEDIAN PER POSITION and
averages the six EQUALLY, so a position estimated from 200 frames gets the same influence as one
estimated from 3000. 200 frames is 6.4 s at 31.23 Hz, and calcium is heavily autocorrelated on ~1 s
timescales, so the EFFECTIVE sample size behind a thin position is nearer 6 than 200. Equal weighting
is the worst available weighting for that: it maximises the contribution of the noisiest estimates.

So `restw` trades a known, FIXED composition bias (17.5% per-position survival spread, stable across
epochs) for added VARIANCE. Which is worse is empirical, and this measures it.

THE MEASUREMENT. Split each position's labelled rest PERIODS -- not frames -- into two halves, build
each baseline from each half, and correlate the halves. Splitting by period matters: adjacent frames
inside one rest bout are nearly the same sample, so a frame-wise split would report a reliability that
the data does not have.

Reported per session: split-half correlation and relative RMS difference for both baselines, plus the
thinnest contributing position, so reliability can be read against how thin the session actually is.

    python -m scripts.rest_migration.restw_reliability [--limit N]
"""
from __future__ import annotations

import argparse

import numpy as np

from wfield_local import config, epochs
from wfield_local.rest_by_position import (
    MIN_FRAMES_PER_POSITION, MIN_POSITIONS_FOR_WEIGHTED, _session_daq, frame_samples)


def _periods_by_position(session, n_frames):
    """``{code: [frame_index_array, ...]}`` -- rest frames grouped by position AND BY PERIOD.

    Mirrors `rest_frames_by_position` but keeps each labelled rest period separate instead of
    concatenating, which is what makes an honest split possible.
    """
    from wfield_local.precue_engagement_states import engagement_gate

    rest, cs, codes, ts, fs_samp, _sync = _session_daq(session)
    f_of = np.asarray(fs_samp)[:n_frames]
    # engaged trials -- NB `engagement_gate` returns True for NOT-engaged
    responded = np.ones(len(codes), bool)
    eng = ~np.asarray(engagement_gate(np.arange(len(codes)), responded, codes), bool)

    out = {}
    starts, stops = [], []
    d = np.diff(rest.astype(np.int8))
    starts = np.flatnonzero(d == 1) + 1
    stops = np.flatnonzero(d == -1) + 1
    if rest[0]:
        starts = np.r_[0, starts]
    if rest[-1]:
        stops = np.r_[stops, rest.size]
    for a, b in zip(starts, stops):
        prev = np.searchsorted(cs, a, "right") - 1
        nxt = np.searchsorted(cs, b, "left")
        if prev < 0 or nxt >= len(codes) or prev >= len(codes):
            continue
        if codes[prev] != codes[nxt] or codes[prev] < 0:
            continue
        if not (eng[prev] and eng[nxt]):
            continue
        fr = np.flatnonzero((f_of >= a) & (f_of < b))
        fr = fr[fr < n_frames]
        if fr.size:
            out.setdefault(int(codes[prev]), []).append(fr)
    return out


def _rest_flat(V, idx):
    return np.median(V[:, idx], axis=1) if idx.size else None


def _restw(V, per_half):
    bases = []
    for c in sorted(per_half):
        idx = per_half[c]
        if idx.size < MIN_FRAMES_PER_POSITION // 2:      # half the floor, since this is half the data
            continue
        bases.append(np.median(V[:, idx], axis=1))
    if len(bases) < MIN_POSITIONS_FOR_WEIGHTED:
        return None
    return np.mean(np.stack(bases, 0), axis=0)


def run(label):
    from pathlib import Path

    s = next(x for x in config.load_sessions() if x["label"] == label)
    V = np.load(config.svtcorr_path(s["mc"]), mmap_mode="r")
    n = V.shape[1]
    per = _periods_by_position(s, n)
    if not per:
        return None
    rng = np.random.default_rng(0)
    halfA, halfB = {}, {}
    for c, periods in per.items():
        order = rng.permutation(len(periods))
        a = [periods[i] for i in order[0::2]]
        b = [periods[i] for i in order[1::2]]
        if a:
            halfA[c] = np.concatenate(a)
        if b:
            halfB[c] = np.concatenate(b)
    allA = np.concatenate([v for v in halfA.values()]) if halfA else np.array([], int)
    allB = np.concatenate([v for v in halfB.values()]) if halfB else np.array([], int)
    Vf = np.asarray(V[:, :], dtype=np.float64)

    rA, rB = _rest_flat(Vf, allA), _rest_flat(Vf, allB)
    wA, wB = _restw(Vf, halfA), _restw(Vf, halfB)
    # `per` maps position -> LIST OF PERIOD ARRAYS, so the frame count is a sum over periods.
    thin = min((sum(int(x.size) for x in v) for v in per.values()), default=0)

    def _pair(x, y):
        if x is None or y is None:
            return np.nan, np.nan
        r = float(np.corrcoef(x, y)[0, 1])
        rel = float(np.sqrt(((x - y) ** 2).mean()) / max(np.sqrt((x ** 2).mean()), 1e-12))
        return r, rel

    r_r, rel_r = _pair(rA, rB)
    r_w, rel_w = _pair(wA, wB)
    try:
        ep = epochs.epoch_of(label)
    except Exception:                                                  # noqa: BLE001
        ep = None
    print(f"  {label:12s} {str(ep):9s} thinnest {thin:5d}  "
          f"rest r={r_r:+.4f} relRMS={rel_r:.4f}   restw r={r_w:+.4f} relRMS={rel_w:.4f}",
          flush=True)
    return dict(label=label, epoch=ep, thin=thin, r_rest=r_r, rel_rest=rel_r,
                r_restw=r_w, rel_restw=rel_w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    want = set(config.phase_labels("pre") + config.phase_labels("post"))
    labs = [x["label"] for x in config.load_sessions() if x["label"] in want]
    if a.limit:
        labs = labs[: a.limit]
    rows = []
    for lab in labs:
        try:
            r = run(lab)
        except Exception as ex:                                        # noqa: BLE001
            print(f"  !! {lab}: {type(ex).__name__} {str(ex)[:70]}", flush=True)
            continue
        if r:
            rows.append(r)
    if not rows:
        print("nothing measured")
        return
    rr = np.array([r["r_rest"] for r in rows], float)
    rw = np.array([r["r_restw"] for r in rows], float)
    er = np.array([r["rel_rest"] for r in rows], float)
    ew = np.array([r["rel_restw"] for r in rows], float)
    ok = np.isfinite(rr) & np.isfinite(rw)
    print(f"\nSPLIT-HALF RELIABILITY, {int(ok.sum())} sessions")
    print(f"  rest   r median {np.median(rr[ok]):+.4f}   relRMS median {np.median(er[ok]):.4f}")
    print(f"  restw  r median {np.median(rw[ok]):+.4f}   relRMS median {np.median(ew[ok]):.4f}")
    print(f"  restw LESS reliable in {int((rw[ok] < rr[ok]).sum())}/{int(ok.sum())} sessions")
    print(f"  median relRMS ratio restw/rest: {np.median(ew[ok] / np.maximum(er[ok], 1e-12)):.2f}x")
    thin = np.array([r["thin"] for r in rows], float)[ok]
    if thin.size > 3:
        print(f"  corr(thinnest position, restw reliability) = "
              f"{np.corrcoef(thin, rw[ok])[0,1]:+.3f}  "
              f"(positive = thinner sessions are less reliable)")


if __name__ == "__main__":
    main()
