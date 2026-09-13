"""How long a settle does the REST window need after the response window closes?

Priya, 2026-09-12: "should we use a settle of 0.5s then?"

WHAT THE SETTLE IS ACTUALLY FOR, which decides how long it must be. It is NOT protecting against the
cue-evoked response: the response window closes at cue + 3.5 s and first-lick latency is 0.14-0.26 s
pre-stroke, so the trial's evoked activity is several seconds past by then. It is not protecting
against consumption licking either -- the lick term already excludes 1 s before and 3 s after EVERY
lick onset, which covers a consumption bout far more thoroughly than a fixed settle could.

THE ONE THING IT UNIQUELY COVERS is reward delivery on a trial with NO licking. `reward_mode` is
`auto_after_delay`, so water arrives whether or not the animal responds (held only after >6
consecutive misses), and on a no-lick trial there is no lick onset for the lick buffer to key on.
If every reward lands inside [cue, cue + response_window] then the trial window already contains it
and the settle is nearly redundant; if rewards land after the window closes, the settle has to
reach past the latest of them.

Measures, on a balanced subsample:
  * reward time relative to its cue, and the fraction landing after the response window closes;
  * usable rest per trial at settles of 0.25 / 0.5 / 0.75 / 1.0 s, and what fraction of trials then
    admit a 1 s and a 2 s segment -- the quantity that decides the state decoder's window.
"""
from __future__ import annotations

import sys
import time

import numpy as np

from wfield_local import config, daq_io, epoch_figures as ef
from wfield_local.grant_figures import _day
from wfield_local.locanmf_cue_lick_analysis import SESSIONS

PER_EPOCH = 3
RESP_WIN_S = 3.5
SETTLES = (0.25, 0.5, 0.75, 1.0)


def main():
    import h5py

    t0 = time.time()
    want = set(config.phase_labels("pre") + config.phase_labels("post"))
    lat, gaps, seen = [], [], {}
    for s in SESSIONS:
        if s["label"] not in want or not s.get("h5"):
            continue
        an, mmdd = s["label"].split("_")
        d = _day(an, mmdd)
        if d is None:
            continue
        e = "pre" if int(d) <= 0 else ef.epoch_of_day(an, int(d))
        if e is None or seen.get(e, 0) >= PER_EPOCH:
            continue
        try:
            with h5py.File(s["h5"], "r") as f:
                fs = float(f.attrs["sample_rate_hz"])
                dnames, bits = daq_io.digital_bits(f)
                rv = daq_io.analog_channel(f, "reward_ttl", required=False)
            idx = {n: i for i, n in enumerate(dnames)}
            if "trial_start" not in idx or rv is None:
                continue
            cue = daq_io.rising_edges(bits[:, idx["cue"]]) / fs
            tstart = daq_io.rising_edges(bits[:, idx["trial_start"]]) / fs
            rew = daq_io.rising_edges((rv > 2.5).astype(np.int8)) / fs
        except Exception as ex:                                        # noqa: BLE001
            print(f"  !! {s['label']}: {type(ex).__name__} {str(ex)[:60]}", flush=True)
            continue
        seen[e] = seen.get(e, 0) + 1

        # each reward -> the most recent cue at or before it
        if rew.size and cue.size:
            j = np.searchsorted(cue, rew, side="right") - 1
            ok = j >= 0
            lat.append(rew[ok] - cue[j[ok]])
        nxt = np.searchsorted(tstart, cue + RESP_WIN_S, side="left")
        has = nxt < len(tstart)
        gaps.append(tstart[nxt[has]] - (cue[has] + RESP_WIN_S))
        print(f"  .. {s['label']} ({e})", flush=True)

    lat = np.concatenate(lat)
    g = np.concatenate(gaps)
    print(f"\n{sum(seen.values())} sessions {seen}, {time.time() - t0:.0f}s")

    print(f"\nREWARD LATENCY FROM ITS CUE (s), {lat.size} rewards:")
    for p in (1, 25, 50, 75, 95, 99, 100):
        print(f"   p{p:<4d} {np.percentile(lat, p):7.3f}")
    print(f"   rewards landing AFTER the {RESP_WIN_S:.1f}s response window closes: "
          f"{float(np.mean(lat > RESP_WIN_S)):.4f}")
    late = lat[lat > RESP_WIN_S]
    if late.size:
        print(f"   of those, how far past it: median {np.median(late - RESP_WIN_S):.2f}s, "
              f"p95 {np.percentile(late - RESP_WIN_S, 95):.2f}s, max {late.max() - RESP_WIN_S:.2f}s")

    print(f"\nUSABLE REST PER TRIAL by settle ({g.size} trials):")
    print(f"{'settle':>8}{'median':>9}{'p10':>8}{'>=1s':>8}{'>=2s':>8}{'total/trial':>13}")
    for st in SETTLES:
        u = g - st
        u = np.where(u > 0, u, 0.0)
        print(f"{st:>8.2f}{np.median(u):>9.2f}{np.percentile(u, 10):>8.2f}"
              f"{float((u >= 1).mean()):>8.3f}{float((u >= 2).mean()):>8.3f}{u.mean():>13.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
