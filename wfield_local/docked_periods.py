"""The STRICT no-target interval: the spout is at the DOCK, stationary, nowhere near a position.

Priya, 2026-09-13: *"ok let's try the strict 'spout docked' interval"*, after establishing that the
spout RETRACTS between trials and that the behaviour log carries its timing.

WHY THE EXISTING REST WINDOW IS NOT THIS. `segmentation.rest` runs `cue + response_window + 0.5 s`
to the next `trial_start`. Measured against the behaviour log's own events (medians over six
sessions, 5/19-9/08):

    cue          0
    dock_start   3.68 - 3.88     the spout BEGINS retracting
    dock         4.61 - 4.80     the spout is AWAY   (~0.92 s of travel)
    trial_start  5.88 - 6.11     the next trial opens; the spout starts back out
    position     6.77 - 7.00     the spout ARRIVES at the next target

the rest window opens at ~4.0 s -- roughly 0.65 s BEFORE `dock`. So its first third contains the
spout physically retracting: a moving object the animal can see and track, whose trajectory STARTS
AT THE POSITION IT WAS JUST AT. That is a mundane explanation for position information in "rest",
and it has to be removed before the interesting one is worth entertaining.

THE DOCKED INTERVAL IS `dock` -> next `trial_start`: ~1.35 s in which there is no target AND no
spout movement. It is shorter than the rest window by design; the point is what it excludes.

WHAT IT IS NOT. `trial_start` -> `position` (~0.9 s) is a THIRD interval: still no target, but the
spout is travelling toward a position the block structure often makes predictable. That belongs to
an anticipation analysis, not to a baseline, and is deliberately outside the docked window.

STILL INTERSECTED WITH REST'S OTHER CONDITIONS. Docked says where the SPOUT is; it says nothing
about the animal. A docked interval in which the mouse is running, or licking at nothing, is not
rest. The caller intersects this with the existing not-running / not-licking mask, so "docked rest"
is strictly a SUBSET of rest -- which is what makes the comparison between them interpretable.

CLOCKS. `events.csv` is on the GUI device clock and everything else here is on DAQ samples.
`spout_behavior._sync_affine` already fits the mapping from the shared Arduino heartbeat (`sync` in
both streams) and REFUSES a fit whose rate is off by >1% or whose residual exceeds 10 ms. This
module reuses it rather than re-deriving it, and returns None wherever it refuses: a session whose
clocks cannot be aligned must drop out of a docked-window analysis, not fall back to the loose one.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def dock_events(session_dir: Path):
    """``(dock_s, trial_start_s, sync_s)`` on the GUI DEVICE clock, or None.

    One `dock` and one `trial_start` per trial, taken as the FIRST of each within a trial_id -- the
    GUI can emit a repeat when a move is retried (`move_aborted` appears in these logs), and the
    first is the one that matches the cycle every other event is timed against.
    """
    import pandas as pd

    p = Path(session_dir) / "events.csv"
    if not p.exists():
        return None
    try:
        ev = pd.read_csv(p, usecols=lambda c: c in ("device_t_ms", "event_name", "trial_id"))
    except Exception:                                                  # noqa: BLE001
        return None
    ev["device_t_ms"] = pd.to_numeric(ev["device_t_ms"], errors="coerce")
    ev["trial_id"] = pd.to_numeric(ev["trial_id"], errors="coerce")
    if "event_name" not in ev:
        return None

    def first(name):
        sub = ev[ev["event_name"] == name].dropna(subset=["device_t_ms", "trial_id"])
        return sub.groupby("trial_id")["device_t_ms"].min() / 1000.0 if len(sub) else None

    dk, tsx = first("dock"), first("trial_start")
    if dk is None or tsx is None or len(dk) < 20 or len(tsx) < 20:
        return None
    sy = ev[ev["event_name"] == "sync"]["device_t_ms"].dropna().to_numpy() / 1000.0
    return dk, tsx, np.sort(sy)


def docked_mask(session_dir: Path, daq_sync_samples, n_samples, fs=5000.0):
    """Boolean mask over DAQ SAMPLES that is True while the spout is docked, or None.

    True from each trial's `dock` until the NEXT `trial_start` -- the interval with no target present
    and no spout movement. The final trial's dock is included only if a later `trial_start` exists,
    so the mask never runs to the end of the recording on the strength of a missing event.
    """
    from wfield_local.spout_behavior import _sync_affine

    got = dock_events(session_dir)
    if got is None:
        return None
    dk, tsx, gui_sync = got
    daq_sync_s = np.asarray(daq_sync_samples, float) / float(fs)
    if daq_sync_s.size == 0 or gui_sync.size == 0:
        return None
    aff = _sync_affine(daq_sync_s, gui_sync)          # device = a*daq + b
    if aff is None:
        return None
    a, b = aff
    # INVERT to go device -> DAQ. `_sync_affine` is fitted in the direction the lick comparison
    # needs; the inverse is exact for an affine map and avoids fitting the same pair twice.
    def to_daq(t_dev):
        return (np.asarray(t_dev, float) - b) / a

    dock_daq = to_daq(dk.to_numpy())
    ts_daq = np.sort(to_daq(tsx.to_numpy()))
    m = np.zeros(int(n_samples), bool)
    n_used = 0
    for t0 in dock_daq:
        j = np.searchsorted(ts_daq, t0, "right")      # the next trial_start AFTER this dock
        if j >= ts_daq.size:
            continue
        aa, bb = int(round(t0 * fs)), int(round(ts_daq[j] * fs))
        if bb <= aa:
            continue
        aa, bb = max(0, aa), min(int(n_samples), bb)
        if bb > aa:
            m[aa:bb] = True
            n_used += 1
    return m if n_used >= 20 else None
