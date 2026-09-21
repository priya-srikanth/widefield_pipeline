"""Where the trial's boundaries actually are, so the REST window can be anchored on a real event.

Priya, 2026-09-12: "I think we want trial stop + 1 until strobe?"

THE ENDPOINT NEEDS CHECKING BEFORE THAT IS IMPLEMENTED, because of an ordering the firmware docs
state explicitly (`daq_trials` module docstring): "firmware `startTrial` MOVES THE SPOUT, THEN calls
`emitPositionCode(...)`, which sets bit0/1/2 and only then pulses a 10 ms strobe". **The strobe
fires AFTER the spout has physically moved.** So a rest window ending AT the strobe contains the
spout movement -- a mechanical, auditory and possibly tactile event with its own cortical response
-- which is exactly what a baseline must not contain.

`trial_start` is a separate digital bit emitted at the TOP of `startTrial`, so it should precede the
movement. If it does, it is the correct endpoint, and the gap `strobe - trial_start` measures HOW
LONG THE SPOUT TAKES TO MOVE -- which is the quantity that should set any buffer, rather than a
guessed 1 s.

This reads DIGITAL channels only, so it is far cheaper than the analog passes.
"""
from __future__ import annotations

import sys
import time

import numpy as np

from wfield_local import config, daq_io
from wfield_local import epoch_figures as ef
from wfield_local.grant_figures import _day
from wfield_local.locanmf_cue_lick_analysis import SESSIONS

PER_EPOCH = 8


def main():
    import h5py

    t0 = time.time()
    want = set(config.phase_labels("pre") + config.phase_labels("post"))
    gaps, starts_before, resp_to_next, n_ok, n_nostart = [], [], [], 0, 0
    seen = {}
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
            idx = {n: i for i, n in enumerate(dnames)}
            if "trial_start" not in idx:
                n_nostart += 1
                continue
            cue = daq_io.rising_edges(bits[:, idx["cue"]]) / fs
            strobe = daq_io.rising_edges(bits[:, idx["spout_strobe"]]) / fs
            tstart = daq_io.rising_edges(bits[:, idx["trial_start"]]) / fs
        except Exception as ex:                                        # noqa: BLE001
            print(f"  !! {s['label']}: {type(ex).__name__} {str(ex)[:60]}", flush=True)
            continue
        seen[e] = seen.get(e, 0) + 1
        n_ok += 1

        # pair each strobe with the most recent trial_start at or before it
        j = np.searchsorted(tstart, strobe, side="right") - 1
        ok = j >= 0
        g = strobe[ok] - tstart[j[ok]]
        gaps.append(g)
        starts_before.append(float(np.mean(g > 0)))

        # how much room is there between the end of one trial's response window and the NEXT
        # trial_start? That interval IS the rest window under the proposed definition.
        rw = 3.5
        nxt = np.searchsorted(tstart, cue + rw, side="left")
        has = nxt < len(tstart)
        resp_to_next.append(tstart[nxt[has]] - (cue[has] + rw))
        print(f"  .. {s['label']} ({e})", flush=True)

    print(f"\n{n_ok} sessions ({seen}); {n_nostart} lacked a trial_start bit; "
          f"{time.time() - t0:.0f}s")

    g = np.concatenate(gaps)
    print(f"\nSTROBE - TRIAL_START (s) -- the spout's move time, {g.size} trials:")
    for p in (1, 5, 25, 50, 75, 95, 99):
        print(f"   p{p:<3d} {np.percentile(g, p):7.3f}")
    print(f"   trial_start PRECEDES the strobe on {float(np.mean(g > 0)):.4f} of trials")
    print(f"   per-session fraction preceding: min {min(starts_before):.4f}")

    r = np.concatenate(resp_to_next)
    r = r[np.isfinite(r)]
    print(f"\nREST WINDOW LENGTH under 'response-window end -> next trial_start' ({r.size} trials):")
    for p in (1, 5, 10, 25, 50, 75, 90):
        print(f"   p{p:<3d} {np.percentile(r, p):7.2f}")
    for w in (1.0, 1.5, 2.0, 3.0):
        print(f"   windows longer than {w:.1f}s after a 1 s settle: "
              f"{float(np.mean(r - 1.0 >= w)):.3f}")
    print(f"   NEGATIVE (next trial starts before the window closes): "
          f"{float(np.mean(r < 0)):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
