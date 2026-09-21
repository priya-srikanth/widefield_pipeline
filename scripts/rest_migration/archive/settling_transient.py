"""How long does the RECORDING-ONSET transient last, cohort-wide? Measure it, do not assume 30 s.

WHAT THIS IS. Every session opens with a large excursion that decays far too fast to be photobleaching
-- +0.05-0.07 against a session SD of ~0.027 on the two sessions plotted, and PS95_0823 at ~117 sigma
in five bursts. It SURVIVES the hemodynamic subtraction (PS94_0819's first minute swings to -0.075
against +/-0.03 for the rest of the session), so the 470/415 regression does not remove it, and it is
currently fitted as drift in every session in the cohort. Most likely LED or camera settling.

WHY MEASURE RATHER THAN PICK 30 s. "30 s" came from the bin width of a diagnostic on TWO sessions.
Fixing a config number from the bin someone happened to plot is the same error as the +0.028
acceptance bar, which was 1.96*se dressed up as an effect size. The decay constant is a property of
the cohort and should be read off it.

METHOD. Component 0 of the CORRECTED `SVTcorr` (the adopted variant) via mmap -- one row per session,
no U_atlas load; this is a TIMING question and component 0 carries the global mode.

IT MUST BE THE CORRECTED SIGNAL, NOT THE RAW ONE, and the first version of this script got that
wrong. Referencing raw 470 against a 5-20 min window made early bins sit high because of BLEACHING,
so the measurement reported when the bleaching decline entered the reference band -- 158-284 s on the
smoke test -- rather than when settling ended. The corrected signal has the polynomial drift removed,
so what remains at the onset is the transient itself. It is also what every analysis reads, and the
onset is known to SURVIVE the hemodynamic subtraction. A STABLE reference is taken from 5-20 min (after
any plausible settling, before end-of-session effects), as median and MAD. The trace is binned at
`--bin-s` over the first `--window-min`, and SETTLING TIME is the first bin after which EVERY
subsequent bin stays inside `--k` MADs of the stable median -- "first bin that is quiet AND stays
quiet", not "first bin that dips", which noise alone satisfies early.

INDEXED FROM START OF RECORDING, not from the first trial (Priya, 2026-09-14): the cause is hardware,
so it is anchored to acquisition start. Behaviour begins inside it -- median first cue 12.8 s, and
87/100 sessions have their first cue inside 30 s -- so any discard costs trials (0.33% at 30 s).

    python -m scripts.rest_migration.archive.settling_transient [--bin-s 2] [--k 4]
"""
from __future__ import annotations

import argparse

import numpy as np

from wfield_local import config
from wfield_local.hemo_variants import FS


def settling_time_s(x, bin_s, window_min, k, sustain_s=30.0,
                    stable_lo_min=5.0, stable_hi_min=20.0):
    """``(settle_s, peak_dev_in_mads, stable_mad)`` or ``(nan, nan, nan)`` if not estimable."""
    n = x.size
    lo, hi = int(stable_lo_min * 60 * FS), int(stable_hi_min * 60 * FS)
    if n < hi:
        hi = n
    if hi - lo < int(60 * FS):
        return np.nan, np.nan, np.nan
    ref = x[lo:hi]
    med = float(np.median(ref))
    mad = float(np.median(np.abs(ref - med))) * 1.4826
    if mad <= 0:
        return np.nan, np.nan, np.nan

    bn = max(1, int(round(bin_s * FS)))
    end = min(n, int(window_min * 60 * FS))
    nb = end // bn
    if nb < 3:
        return np.nan, np.nan, np.nan
    dev = np.abs(np.median(x[: nb * bn].reshape(nb, bn), axis=1) - med) / mad
    peak = float(dev.max())
    quiet = dev < k
    # FIRST BIN AFTER WHICH THE TRACE IS QUIET FOR A SUSTAINED RUN -- not "quiet for the rest of the
    # window", which was the second wrong version of this. Requiring quiet to the end of a 5 min
    # window means ONE burst at 4.9 min reports 294 s of settling, so the measurement became "time of
    # the last excursion anywhere in five minutes" rather than onset decay. The smoke test showed it
    # as a 0 s / 300 s bimodality with nothing between, which is not what a decay looks like.
    #
    # A sustained run also rejects the opposite error: a single bin dipping under threshold is
    # satisfied by noise almost immediately and would report one bin on every session.
    run = max(1, int(round(sustain_s / bin_s)))
    settle_bin = nb
    for i in range(nb):
        if quiet[i:i + run].all():
            settle_bin = i
            break
    return float(settle_bin * bin_s), peak, mad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin-s", type=float, default=2.0)
    ap.add_argument("--window-min", type=float, default=5.0)
    ap.add_argument("--k", type=float, default=4.0, help="MADs from the stable median")
    ap.add_argument("--sustain-s", type=float, default=30.0,
                    help="how long the trace must STAY quiet to count as settled")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    want = set(config.phase_labels("pre") + config.phase_labels("post"))
    sess = [s for s in config.load_sessions() if s["label"] in want and s.get("mc")]
    if a.limit:
        sess = sess[: a.limit]
    rows = []
    for s in sess:
        from pathlib import Path
        p = Path(config.svtcorr_path(s["mc"]))
        if not p.exists():
            continue
        try:
            x = np.asarray(np.load(p, mmap_mode="r")[0], dtype=np.float64)
        except Exception as ex:                                        # noqa: BLE001
            print(f"  !! {s['label']}: {type(ex).__name__}", flush=True)
            continue
        t, peak, mad = settling_time_s(x, a.bin_s, a.window_min, a.k, a.sustain_s)
        if not np.isfinite(t):
            continue
        rows.append((s["label"], t, peak))
        print(f"  {s['label']:12s} settles {t:6.1f} s   onset peak {peak:7.1f} MADs", flush=True)

    if len(rows) < 10:
        print(f"\nonly {len(rows)} sessions -- a failed run, not a result")
        return
    t = np.array([r[1] for r in rows])
    pk = np.array([r[2] for r in rows])
    print(f"\nSETTLING TIME, {len(rows)} sessions (bin {a.bin_s:g}s, threshold {a.k:g} MADs)")
    for q in (50, 75, 90, 95, 99):
        print(f"   p{q:<3d} {np.percentile(t, q):7.1f} s")
    print(f"   max  {t.max():7.1f} s   mean {t.mean():7.1f} s")
    for thr in (10, 20, 30, 60, 120):
        print(f"   settled within {thr:3d} s: {(t <= thr).sum():3d}/{len(t)} "
              f"({100*(t<=thr).mean():.0f}%)")
    print(f"\nONSET PEAK DEVIATION: median {np.median(pk):.1f} MADs, "
          f"p90 {np.percentile(pk,90):.1f}, max {pk.max():.1f}")
    print("\nWORST (longest to settle):")
    for lab, tt, p in sorted(rows, key=lambda r: -r[1])[:10]:
        print(f"  {lab:12s} {tt:6.1f} s   peak {p:7.1f} MADs")


if __name__ == "__main__":
    main()
