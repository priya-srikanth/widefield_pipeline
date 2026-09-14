"""Is the TIME-LOCAL rest baseline capturing real drift, or fitting noise?

Priya, 2026-09-13: *"do we need time local still? isn't the SVD already locally-corrected"*.

THE QUESTION IS SHARP AND THE PRIOR FAVOURS HER. The production signal removes slow drift with a
zero-phase 0.1 Hz Butterworth (docs/PREPROCESSING_DECISION.md) -- that is a HIGH-PASS at 0.1 Hz,
which removes everything slower than ~10 s. The time-local baseline bins a 30-60 min session into
12, so each bin spans 2.5-5 MINUTES. Structure on that timescale is three orders of magnitude below
the filter's corner and should already be gone. If it is, the time-local baseline is estimating
noise, with all the small-sample fragility that just invalidated the first buffer sweep.

WHY THE OBVIOUS EVIDENCE DOES NOT SETTLE IT. The number that would have been cited -- same position's
rest early-vs-late at RMS 0.00282 -- comes from `rest_position_vs_drift`, a control ALREADY
WITHDRAWN (DECISIONS.md, 2026-09-13): it never separated drift from noise, which was precisely its
flaw. Two estimates of the same thing from different halves of a session differ by noise alone.
Re-citing it here would repeat the error that produced the wrong "rest is drift" conclusion.

THE TEST, stated before the answer, and it is the same shape as the noise floor that fixed the
buffer sweep:

    Build the time-local baseline the real way -- bin BY TIME, median per bin, interpolate.
    Then build it again from the SAME rest frames with the time labels SHUFFLED, preserving the
    bin SIZES exactly. Compare how far each departs from a flat session mean.

    real >> shuffled  ->  there IS genuine slow structure; time-local is earning its cost
    real ~= shuffled  ->  the "across-time structure" is sampling noise, and a FLAT baseline is
                          both correct and more robust, because it averages over all rest frames
                          instead of 1/12 of them per estimate

A SECOND, INDEPENDENT READ: SMOOTHNESS. Real drift is slow, so consecutive bin medians should be
correlated; noise is white, so shuffled bins should not be. Lag-1 autocorrelation across the 12 bin
medians separates those even where the magnitudes are similar, and it cannot be faked by sample
size. Both are reported because either alone can mislead.

RUN:  python -m scripts.rest_migration.timelocal_needed [--limit 8] [--bins 12]
"""
from __future__ import annotations

import argparse
import glob
import sys
import time

import numpy as np


def _binned_medians(V, frames, edges, nbins):
    """``(K, nbins)`` median per bin over the given frame indices, NaN where a bin is empty."""
    out = np.full((V.shape[0], nbins), np.nan)
    for b in range(nbins):
        sel = frames[(frames >= edges[b]) & (frames < edges[b + 1])]
        if sel.size:
            out[:, b] = np.median(V[:, sel], axis=1)
    return out


def _interp_to_flatdiff(bm, cent, T, flat):
    """RMS departure of the interpolated baseline from the flat mean -- the quantity at issue."""
    x = np.arange(T)
    acc = 0.0
    n = 0
    for k in range(bm.shape[0]):
        ok = np.isfinite(bm[k])
        if ok.sum() < 2:
            continue
        y = np.interp(x, cent[ok], bm[k][ok])
        acc += float(np.mean((y - flat[k]) ** 2))
        n += 1
    return np.sqrt(acc / max(1, n))


def _lag1(bm):
    """Mean lag-1 autocorrelation across bins, over components. Real drift is smooth; noise is not."""
    r = []
    for k in range(bm.shape[0]):
        v = bm[k][np.isfinite(bm[k])]
        if v.size < 4:
            continue
        v = v - v.mean()
        d = float(np.dot(v, v))
        if d > 0:
            r.append(float(np.dot(v[:-1], v[1:])) / d)
    return float(np.mean(r)) if r else np.nan


def main() -> int:
    from wfield_local import config, joint_basis
    from wfield_local.locanmf_cue_lick_analysis import SESSIONS
    from wfield_local.quiet_periods import quiet_frame_path, quiet_variant
    from wfield_local.rest_by_position import rest_frames_by_position

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--bins", type=int, default=12)
    ap.add_argument("--perm", type=int, default=20)
    a = ap.parse_args()

    print(f"rest variant = {quiet_variant()!r};  bins = {a.bins};  shuffles = {a.perm}")
    want = set(config.phase_labels("pre") + config.phase_labels("post"))
    todo = [x for x in SESSIONS if x["label"] in want]
    step = max(1, len(todo) // max(1, a.limit))
    todo = todo[::step][: a.limit]

    rng = np.random.default_rng(0)
    t0 = time.time()
    rows = []
    for s in todo:
        lab = s["label"]
        qf = quiet_frame_path(s["mc"])
        if not qf:
            print(f"  .. {lab}: no rest frame mask")
            continue
        try:
            q = np.load(qf).astype(bool)
            _u, v = joint_basis._load_session(s["mc"])
        except Exception as ex:                                        # noqa: BLE001
            print(f"  !! {lab}: {type(ex).__name__} {str(ex)[:60]}")
            continue
        V = np.asarray(v)
        T = min(V.shape[1], q.shape[0])
        V = V[:, :T]
        fr = np.flatnonzero(q[:T])
        if fr.size < 4 * a.bins:
            print(f"  .. {lab}: only {fr.size} rest frames")
            continue

        edges = np.linspace(0, T, a.bins + 1)
        cent = (edges[:-1] + edges[1:]) / 2.0
        flat = np.median(V[:, fr], axis=1)

        bm_real = _binned_medians(V, fr, edges, a.bins)
        real_d = _interp_to_flatdiff(bm_real, cent, T, flat)
        real_r1 = _lag1(bm_real)

        # THE SHUFFLE: same frames, same bin SIZES, time labels destroyed. Preserving the sizes is
        # what makes this a control rather than a different experiment -- an unequal split would
        # change the per-bin noise and could produce the difference on its own.
        counts = [int(((fr >= edges[b]) & (fr < edges[b + 1])).sum()) for b in range(a.bins)]
        sd, sr = [], []
        for _ in range(a.perm):
            perm = rng.permutation(fr)
            bm = np.full((V.shape[0], a.bins), np.nan)
            i = 0
            for b, c in enumerate(counts):
                if c:
                    bm[:, b] = np.median(V[:, perm[i:i + c]], axis=1)
                i += c
            sd.append(_interp_to_flatdiff(bm, cent, T, flat))
            sr.append(_lag1(bm))
        shuf_d, shuf_r1 = float(np.mean(sd)), float(np.nanmean(sr))

        # ---------------------------------------------------------------- POSITION CONTROL
        # THE SECOND HYPOTHESIS, and the one the lag-1 result points at. The real bin medians vary
        # 20x more than sampling noise but are NOT SMOOTH -- which is not what photobleaching looks
        # like. Positions run in ~6-trial BLOCKS and rest CARRIES POSITION (observed/null 1.443,
        # 4/4 animals), so a bin dominated by one position has a different median for a reason that
        # has nothing to do with time.
        #
        # THE CONTROL: preserve each bin's POSITION COMPOSITION exactly -- for every (bin,
        # position) pair keep the same count -- but draw those frames from ANYWHERE in the session
        # that the position occurs. Time locality is destroyed; position mix is not.
        #
        #   real ~= position-control  ->  the structure is BLOCK COMPOSITION, not drift, and a
        #                                 time-local baseline is absorbing position signal
        #   real >>  position-control ->  genuinely temporal, and time-local earns its cost
        pos_d = np.nan
        try:
            per, pinfo = rest_frames_by_position(s, T)
            if per and not pinfo.get("error"):
                of_pos = {}
                for c_, idx in per.items():
                    of_pos[c_] = np.asarray(idx)
                pd_ = []
                for _ in range(a.perm):
                    bm = np.full((V.shape[0], a.bins), np.nan)
                    for b in range(a.bins):
                        take = []
                        for c_, idx in of_pos.items():
                            nb = int(((idx >= edges[b]) & (idx < edges[b + 1])).sum())
                            if nb:
                                take.append(rng.choice(idx, size=min(nb, idx.size),
                                                       replace=False))
                        if take:
                            sel = np.concatenate(take)
                            if sel.size:
                                bm[:, b] = np.median(V[:, sel], axis=1)
                    pd_.append(_interp_to_flatdiff(bm, cent, T, flat))
                pos_d = float(np.mean(pd_))
        except Exception as ex:                                        # noqa: BLE001
            print(f"     .. position control unavailable: {type(ex).__name__} {str(ex)[:50]}")

        rows.append((lab, real_d, shuf_d, real_r1, shuf_r1, fr.size, pos_d))
        print(f"  .. {lab:<12} n={fr.size:>6}  flat-departure real {real_d:.5f}  "
              f"time-shuffle {shuf_d:.5f} (x{real_d / max(1e-12, shuf_d):.1f})  "
              f"POSITION-control {pos_d:.5f} "
              f"(x{real_d / max(1e-12, pos_d):.2f})   "
              f"lag1 {real_r1:+.3f} vs {shuf_r1:+.3f}", flush=True)

    print(f"\n{'=' * 78}\nIS THE TIME-LOCAL REST BASELINE EARNING ITS COST?\n{'=' * 78}")
    if not rows:
        print("NOTHING TESTED -- a failed run, not a negative result.")
        return 1
    rd = np.array([r[1] for r in rows])
    sdv = np.array([r[2] for r in rows])
    rr = np.array([r[3] for r in rows])
    sr_ = np.array([r[4] for r in rows])
    print(f"sessions tested: {len(rows)}")
    print(f"\ndeparture from a FLAT baseline   real {rd.mean():.5f}   shuffled {sdv.mean():.5f}   "
          f"ratio {rd.mean() / max(1e-12, sdv.mean()):.2f}")
    print(f"sessions with real > shuffled:   {int((rd > sdv).sum())}/{len(rows)}")
    print(f"\nlag-1 autocorrelation of bins    real {np.nanmean(rr):+.3f}   "
          f"shuffled {np.nanmean(sr_):+.3f}")
    print(f"sessions with real > shuffled:   {int((rr > sr_).sum())}/{len(rows)}")
    pdv = np.array([r[6] for r in rows], float)
    ok = np.isfinite(pdv)
    if ok.any():
        print(f"\nPOSITION CONTROL (bin position-mix preserved, time locality destroyed)")
        print(f"  real {rd[ok].mean():.5f}   position-control {pdv[ok].mean():.5f}   "
              f"ratio {rd[ok].mean() / max(1e-12, pdv[ok].mean()):.2f}")
        print(f"  sessions with real > position-control: {int((rd[ok] > pdv[ok]).sum())}/{int(ok.sum())}")
        print("  ratio ~1  ->  the across-bin structure IS block/position composition, not drift:")
        print("                a time-local baseline is then absorbing POSITION signal, which is")
        print("                the self-subtraction trap that ruled out a per-position baseline")
        print("  ratio >>1 ->  genuinely temporal structure beyond position composition")
    print("\nHOW TO READ IT:")
    print("  ratio ~1 AND lag1 ~ shuffled  ->  no slow structure survives the 0.1 Hz high-pass;")
    print("                                    the time-local baseline is fitting NOISE and a flat")
    print("                                    one is better -- it uses every rest frame at once")
    print("  ratio >1 AND lag1 > shuffled  ->  real drift survives; time-local is earning its cost")
    print("  ratio >1 but lag1 ~ shuffled  ->  magnitude without smoothness = a sample-size effect,")
    print("                                    NOT drift. Trust lag1 here; it cannot be faked by n.")
    print(f"\n[done in {time.time() - t0:.0f}s]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
