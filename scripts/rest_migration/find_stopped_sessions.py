"""Rank POST-STROKE sessions by having a STOPPED CHUNK -- the animal quits well before the recording
ends -- and by how far the raw fluorescence moves across that boundary.

WHY THIS RANKING AND NOT A SIMPLER ONE. The first version of this script ranked by tail-vs-middle
level, which CONFLATES two different things: ordinary photobleaching (a smooth decline every session
has) and a step at the moment the animal stops working. Only the second is the stress case for the
order-10 polynomial, because polynomials are GLOBAL basis functions -- a late step cannot be fitted
locally, so the fit either misses it or rings across the whole record reaching for it. Ranking by
tail level put three sessions on top that worked to the last minute.

So the stop is taken from BEHAVIOUR (the `engaged` column of the per-session trial table, the same
shared gate the rest definition uses) and the fluorescence change is measured ACROSS THAT BOUNDARY.

Reads row 0 of `SVT.npy` via mmap -- the dominant global component, enough to RANK. The plotting
script computes the proper brain-masked spatial mean for whichever session is picked.

    python -m scripts.rest_migration.find_stopped_sessions
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from wfield_local import config
from wfield_local.hemo_variants import FS, functional_channel


def _last_engaged_s(label):
    """Cue time (s) of the last ENGAGED trial, and the session's last trial time, or (None, None)."""
    import glob as _g

    import pandas as pd

    an, mmdd = label.split("_")
    root = Path(config.resolver().resolve("behavior_out", "")) / "sessions" / an / f"2026{mmdd}"
    hits = _g.glob(str(root / "*_trials.csv"))
    if not hits:
        return None, None
    d = pd.read_csv(hits[0])
    if "engaged" not in d.columns or "cue_s" not in d.columns or not len(d):
        return None, None
    eng = d[d["engaged"].astype(bool)]
    if not len(eng):
        return None, float(d["cue_s"].max())
    return float(eng["cue_s"].max()), float(d["cue_s"].max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-mmdd", default="0813", help="first post-stroke date (inclusive)")
    ap.add_argument("--min-stop-min", type=float, default=3.0,
                    help="minimum quiet tail (min) for a session to count as STOPPED")
    a = ap.parse_args()

    rows = []
    for s in config.load_sessions():
        mmdd = s["label"].split("_")[1]
        if mmdd < a.from_mmdd:
            continue
        svt_p = Path(s["mc"]) / "wfield_local_results" / "SVT.npy"
        if not svt_p.exists():
            continue
        last_s, _ = _last_engaged_s(s["label"])
        if last_s is None:
            continue
        try:
            f = np.asarray(np.load(svt_p, mmap_mode="r")[0, functional_channel(s)::2], dtype=np.float64)
        except Exception as ex:                                      # noqa: BLE001
            print(f"  !! {s['label']}: {type(ex).__name__} {str(ex)[:50]}", flush=True)
            continue
        n = f.size
        dur_min = n / FS / 60.0
        stop_min = last_s / 60.0
        tail_min = dur_min - stop_min
        if tail_min < a.min_stop_min:
            continue
        # level ACROSS the stop boundary, in units of the WORKING period's own scatter
        k = int(np.clip(last_s * FS, 1, n - 1))
        before = f[max(0, k - int(5 * 60 * FS)):k]
        after = f[k:]
        sd = float(np.std(before)) or 1e-12
        step = (float(np.median(after)) - float(np.median(before))) / sd
        rows.append((s["label"], stop_min, dur_min, tail_min, step))
        print(f"  {s['label']:12s} stops {stop_min:6.1f} of {dur_min:6.1f} min "
              f"(tail {tail_min:5.1f})  step across stop {step:+6.2f} SD", flush=True)

    if not rows:
        print("\nNO session has a stopped chunk longer than "
              f"{a.min_stop_min} min -- the animals worked to the end.")
        return
    rows.sort(key=lambda r: -abs(r[4]))
    print(f"\n{len(rows)} STOPPED session(s); largest |step across the stop| first:")
    for lab, sm, dm, tm, st in rows[:12]:
        print(f"  {lab:12s} {st:+6.2f} SD   stops {sm:6.1f}/{dm:6.1f} min, tail {tm:5.1f} min")


if __name__ == "__main__":
    main()
