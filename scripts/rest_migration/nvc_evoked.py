"""Cue-evoked RAW 415 and 470, at frame resolution -- task-evoked haemodynamics, and a test of
whether 415 is isosbestic for THIS cohort.

TWO QUESTIONS, ONE MEASUREMENT.

1. TASK-EVOKED COUPLING (Priya, 2026-09-19): is there less cue- or lick-evoked haemodynamic
   response after the stroke? The existing `hemo_map_control` answer is a SPATIAL correlation
   between maps, which is attenuation-prone and says nothing about amplitude. This is the
   amplitude, per epoch.

2. IS 415 ISOSBESTIC HERE? The literature disagrees on where GCaMP's neutral/anionic crossing
   sits. THREE ESTIMATES, and only the folk one puts it at or below our 415:

       conventional photometry practice     405-415
       Simpson et al. 2024, Neuron primer   420-430 for GCaMP6   (PMC10939905, Table 2)
       Barnett/Drobizhev 2017, GCaMP6m      440-450, and NO true isosbestic point

   Below the crossing calcium DECREASES fluorescence; above it increases. All three agree on
   that much, and two of the three put our 415 BELOW it.

   SIMPSON ET AL. ALSO NAME THE OBSERVABLE, from a 405 nm control against GCaMP6f: "the
   isosbestic control signal has significant negative bleed-through of the GCaMP signal, due to
   405 nm excitation not exactly matching the isosbestic point for GCaMP6f. This is evident as a
   NEGATIVE PEAK IN THE EVENT-ALIGNED AVERAGE." That is this module's statistic, described by
   someone who observed it -- and at 405 rather than 415, so it bounds the leak from above.

   **TIMING SETTLES IT AND AMPLITUDE CANNOT.** Calcium is fast (hundreds of ms); haemodynamics is
   slow (peaks 1-2 s, lasts seconds). So:

       415 below the crossing  ->  EARLY NEGATIVE deflection, then the slow positive one
       415 flat (isosbestic)   ->  no early dip, only the slow rise

   The +0.54% to +1.96% cue-evoked 415 rises already in DECISIONS cannot settle this: they are
   WINDOW AVERAGES over the whole post-cue period, which average an early dip away completely.
   That is precisely why they looked like a clean positive haemodynamic response.

THE NORMALISATION TRAP THIS MODULE EXISTS TO AVOID, and it has already been paid once here
(DECISIONS, 2026-08-18: a check reported +692% and -2881% evoked responses). `U @ SVT`
reconstructs the DEVIATION from each channel's mean -- the reconstructed means are ZERO -- so
dividing by them is division by ~0.

AND THERE IS AN OVER-CORRECTION FOR IT, WHICH THIS MODULE ALSO PAID. `U @ SVT` in this pipeline
is ALREADY FRACTIONAL, so the fix is not "divide by a different mean" -- it is DO NOT DIVIDE.
Dividing by `frames_average.npy` (raw camera counts, ~1.2e4) was the first version here and it is
wrong by four orders of magnitude; the output was all zeros to three decimals. See `session_traces`.

CHANNEL IDENTITY IS DERIVED, NOT ASSUMED, for the same reason: `SVTcorr` is the corrected BLUE
channel, so whichever half of `SVT` it correlates with IS blue. A docstring is not evidence.

PER-TRIAL BASELINE, so slow drift never enters. Each trial is expressed against its own pre-cue
window, which is what makes the raw (undetrended) channels usable here -- and the drift variants
are irrelevant to an event-triggered average on that baseline.

    python -m scripts.rest_migration.nvc_evoked [--animals PS92 ...] [--align cue|lick]
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

PRE_S, POST_S = 1.0, 4.0          # window around the event
BASE_S = (-1.0, -0.2)             # per-trial baseline, ending before any response
EARLY_S = (0.0, 0.4)              # calcium timescale
LATE_S = (1.0, 3.0)               # haemodynamic timescale


def session_traces(s):
    """``(t, pct470, pct415, cue_frames)`` -- brain-mean % traces and the cue frame indices."""
    from wfield_local.hemo_variants import FS, FUNC
    from scripts.rest_migration.plot_session_residual import _brain_mean_op

    res = Path(s["mc"]) / "wfield_local_results"
    svt = np.load(res / "SVT.npy")
    a0 = svt[:, FUNC::2].astype(np.float64)              # functional (blue) by convention
    b0 = svt[:, (FUNC + 1) % 2::2].astype(np.float64)    # the other one
    # DERIVE which is blue rather than trusting FUNC: SVTcorr IS the corrected blue channel.
    try:
        sc = np.load(res / "SVTcorr.npy").astype(np.float64)
        n = min(sc.shape[1], a0.shape[1], b0.shape[1])
        ra = abs(np.corrcoef(sc[0, :n], a0[0, :n])[0, 1])
        rb = abs(np.corrcoef(sc[0, :n], b0[0, :n])[0, 1])
        if rb > ra:
            a0, b0 = b0, a0
    except Exception:                                    # noqa: BLE001
        pass

    u_mean, _npix = _brain_mean_op(s["mc"])
    d470 = u_mean @ a0
    d415 = u_mean @ b0
    # NO DENOMINATOR. `U @ SVT` is ALREADY FRACTIONAL in this pipeline -- `plot_drift_estimators`
    # reports these traces with SD ~0.028 and the 2026-08-18 cue-triggered check reported +3.69%
    # for the same quantity, so x100 IS the percentage. Dividing by `frames_average` (raw camera
    # counts, ~1.2e4) was my first version and it is wrong by four orders of magnitude -- an
    # OVER-correction for the documented trap, which was dividing by the RECONSTRUCTED mean
    # (identically zero), not by any mean at all.
    pct470 = 100.0 * d470
    pct415 = 100.0 * d415

    from wfield_local.rest_by_position import _session_daq
    _rest, cs, _codes, _ts, fs_samp, _sync = _session_daq(s)
    n = d470.size
    f_of = np.asarray(fs_samp)[:n]
    cue_frames = np.searchsorted(f_of, np.asarray(cs))
    cue_frames = cue_frames[(cue_frames > int(PRE_S * FS) + 1)
                            & (cue_frames < n - int(POST_S * FS) - 1)]
    t = (np.arange(-int(PRE_S * FS), int(POST_S * FS)) / FS)
    return t, pct470, pct415, cue_frames


def evoked(trace, cue_frames, t):
    """Per-trial-baselined event-triggered average of `trace`."""
    from wfield_local.hemo_variants import FS

    lo, hi = -int(PRE_S * FS), int(POST_S * FS)
    seg = []
    for f in cue_frames:
        w = trace[f + lo:f + hi]
        if w.size != (hi - lo):
            continue
        b = w[(t >= BASE_S[0]) & (t < BASE_S[1])]
        if b.size:
            seg.append(w - b.mean())
    return (np.mean(seg, axis=0), len(seg)) if seg else (None, 0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--animals", nargs="+", default=None)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)

    from wfield_local import config, epochs
    from wfield_local.paths import PathResolver

    out_dir = a.out or (Path(PathResolver().root("labcams")) / "grant_figures" / "epoch")
    animals = a.animals or ["PS92", "PS93", "PS94", "PS95"]
    want = set(config.phase_labels("pre") + config.phase_labels("post"))

    rows, curves = [], defaultdict(list)
    for s in config.load_sessions():
        lab = s["label"]
        if lab not in want or config.animal_of(lab) not in animals:
            continue
        ep = epochs.epoch_of(lab)
        if not ep:
            continue
        try:
            got = session_traces(s)
        except Exception as ex:                                      # noqa: BLE001
            print(f"  !! {lab}: {type(ex).__name__} {str(ex)[:70]}", flush=True)
            continue
        if got is None:
            print(f"  .. {lab}: no channel means -- skipped", flush=True)
            continue
        t, p470, p415, cf = got
        e470, n1 = evoked(p470, cf, t)
        e415, n2 = evoked(p415, cf, t)
        if e470 is None or e415 is None:
            continue
        w = lambda e, lo, hi: float(np.mean(e[(t >= lo) & (t < hi)]))   # noqa: E731
        r = dict(label=lab, animal=config.animal_of(lab), epoch=ep, n_trials=n1,
                 e470_early=round(w(e470, *EARLY_S), 4), e470_late=round(w(e470, *LATE_S), 4),
                 e415_early=round(w(e415, *EARLY_S), 4), e415_late=round(w(e415, *LATE_S), 4))
        rows.append(r)
        curves[(config.animal_of(lab), ep)].append((e470, e415))
        print(f"   {lab:14s} {ep:9s} n={n1:4d}  470 early {r['e470_early']:+.3f} "
              f"late {r['e470_late']:+.3f}   415 early {r['e415_early']:+.3f} "
              f"late {r['e415_late']:+.3f}", flush=True)

    if not rows:
        print("no sessions -- a failed run, not a result")
        return 1
    q = out_dir / "epoch_16_nvc_evoked.csv"
    with open(q, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"\nwrote {q}")

    print(f"\n{'=' * 78}\nBY EPOCH -- session means (the isosbestic test is 415 EARLY)\n{'=' * 78}")
    print(f"  {'epoch':<10}{'n':>4}{'470 early':>11}{'470 late':>10}"
          f"{'415 early':>11}{'415 late':>10}")
    for ep in ("pre", "acute", "subacute", "chronic"):
        v = [r for r in rows if r["epoch"] == ep]
        if not v:
            continue
        f = lambda k: np.mean([r[k] for r in v])                       # noqa: E731
        print(f"  {ep:<10}{len(v):>4}{f('e470_early'):>+11.3f}{f('e470_late'):>+10.3f}"
              f"{f('e415_early'):>+11.3f}{f('e415_late'):>+10.3f}")
    print("\n415 EARLY < 0 and 415 LATE > 0  ->  415 sits BELOW the crossing (carries -Ca)")
    print("415 EARLY ~ 0 and 415 LATE > 0  ->  415 is flat; the rise is purely haemodynamic")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
