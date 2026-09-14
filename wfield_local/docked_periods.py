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

#: dock - cue, 95th percentile, per position index. THE FALLBACK ANCHOR when the GUI/DAQ clocks
#: cannot be aligned and the docked window has to be reconstructed from DAQ events alone.
#:
#: WHY THE 95th PERCENTILE AND NOT THE MEDIAN. The reconstruction has per-trial error -- dock minus
#: cue has a within-session sd of 0.17-0.40 s, because trial end timing varies -- and the two
#: directions of that error are NOT symmetric in cost. Starting LATE shortens the window; starting
#: EARLY puts spout retraction inside a window whose whole purpose is to exclude it, and the
#: retraction is POSITION-SPECIFIC (0.65-0.98 s by position), so an early start manufactures exactly
#: the position effect these analyses test for. A conservative anchor is therefore mandatory, not
#: fastidious: at p95 the reconstructed window opens after the true dock on ~95% of trials.
#:
#: MEASURED over 110 sessions, June onward, all four animals (median of per-session p95):
_DOCK_AFTER_CUE_P95 = {0: 4.703, 1: 4.862, 2: 4.861, 3: 5.092, 4: 5.369, 5: 5.214}

#: dock_start -> dock travel, per position, median over the same 110 sessions (across-session sd
#: 0.037-0.039 s). Kept because it is the machine constant the reconstruction rests on -- Zaber speed
#: is fixed, so travel is set by distance and distance is a property of the position.
_TRAVEL_S = {0: 0.775, 1: 0.975, 2: 0.976, 3: 0.649, 4: 0.949, 5: 0.951}


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
        aa, bb = round(t0 * fs), round(ts_daq[j] * fs)
        if bb <= aa:
            continue
        aa, bb = max(0, aa), min(int(n_samples), bb)
        if bb > aa:
            m[aa:bb] = True
            n_used += 1
    return m if n_used >= 20 else None


def docked_mask_reconstructed(cue_samples, position_codes, trial_start_samples, n_samples,
                              fs=5000.0):
    """Docked mask rebuilt from DAQ EVENTS ALONE, for sessions whose clocks will not align.

    Priya, 2026-09-13: *"the spout takes the same amount of time to move from position to dock for
    each position across sessions (zaber speed is always equal) - so we should be able to
    reconstruct"*, and this is that reconstruction. Two sessions need it -- PS93 6/6, whose sync fit
    `_sync_affine` refuses, and PS92 8/12, the crash+concat session whose device clock jumps at the
    splice -- and dropping them costs a pre-stroke session from an already small cohort.

    WINDOW: ``cue + _DOCK_AFTER_CUE_P95[position]`` to the next ``trial_start``. Both ends come from
    the DAQ; only the OFFSET is borrowed, and it is borrowed PER POSITION.

    PER POSITION OR NOT AT ALL. One shared constant would inject up to 0.33 s of position-dependent
    error into the window start, and the retraction it is meant to exclude is itself position-
    specific -- so a single constant would manufacture precisely the position effect these analyses
    exist to test. This is the difference between a reconstruction and a fabrication.

    CONSERVATIVE BY CONSTRUCTION: the p95 offset opens the window AFTER the true dock on ~95% of
    trials, trading window length for the guarantee that matters. See `_DOCK_AFTER_CUE_P95`.

    A TRIAL WHOSE POSITION IS UNKNOWN IS SKIPPED, not given the mean offset -- the mean is exactly
    the fabrication the per-position rule rejects.

    Returns None if fewer than 20 windows can be formed, so a caller cannot silently analyse a
    session on three reconstructed intervals.
    """
    cue = np.asarray(cue_samples, np.int64)
    codes = np.asarray(position_codes)
    ts = np.sort(np.asarray(trial_start_samples, np.int64))
    if cue.size == 0 or ts.size == 0 or codes.size < cue.size:
        return None
    m = np.zeros(int(n_samples), bool)
    n_used = 0
    for k in range(cue.size):
        off = _DOCK_AFTER_CUE_P95.get(int(codes[k])) if codes[k] >= 0 else None
        if off is None:
            continue
        t0 = cue[k] + round(off * fs)
        j = np.searchsorted(ts, t0, "right")
        if j >= ts.size:
            continue
        aa, bb = max(0, int(t0)), min(int(n_samples), int(ts[j]))
        if bb > aa:
            m[aa:bb] = True
            n_used += 1
    return m if n_used >= 20 else None


def docked_mask_any(session_dir, daq_sync_samples, cue_samples, position_codes,
                    trial_start_samples, n_samples, fs=5000.0):
    """``(mask, source)`` -- the measured docked window, else the reconstructed one, else None.

    THE SINGLE ENTRY POINT, and the reason it exists. `docked_mask` returns None whenever the GUI
    and DAQ clocks will not align, and every caller then has to remember to try
    `docked_mask_reconstructed`, and to remember that a reconstructed session must be REPORTED. Two
    call sites had already grown their own copy of that dance. The last time one rule had two
    implementations here -- the flat map baseline against the encoder's time-local one -- they
    disagreed for months and nothing raised.

    ``source`` is ``"log"`` or ``"reconstructed"``. IT IS NOT DECORATION: a reconstructed window is
    anchored on the per-position `dock - cue` p95 rather than on the logged dock, so it is
    conservative by construction and its start carries position-specific error that the measured
    window does not. Any result built over a mixed set has to be able to say which sessions took
    which path, and to be re-runnable without the reconstructed ones. `PS93_0606` and `PS92_0812`
    are the two that need it (2026-09-13); both clear their null on the reconstructed window, so
    the recovery is not propping up the cohort result.
    """
    m = docked_mask(session_dir, daq_sync_samples, n_samples, fs=fs) if session_dir else None
    if m is not None:
        return m, "log"
    # RECONSTRUCTION NEEDS THE POSITION CODES -- its anchor is the PER-POSITION `dock - cue` p95.
    # Without them there is no fallback: a single constant offset would inject up to 0.33 s of
    # position-dependent error into the window start, manufacturing exactly the position effect
    # these windows are used to test for. Better to return None and drop the session.
    if position_codes is None or len(np.asarray(position_codes)) == 0:
        return None, None
    m = docked_mask_reconstructed(cue_samples, position_codes, trial_start_samples, n_samples,
                                  fs=fs)
    return (m, "reconstructed") if m is not None else (None, None)


def behaviour_session_dir(label):
    """The BEHAVIOUR-LOG session directory for ``PS92_0606``-style label, or None.

    WHY THIS EXISTS. `docked_mask` reads the dock events out of the behaviour log, so it needs the
    log's session directory -- `Behavior_logs/Widefield/PS92_20260606_122520`. Both callers were
    handing it the DAQ `.h5`'s PARENT instead (`DAQ_recorder_output/20260606`), which contains no
    log, so `docked_mask` always returned None and every session silently fell through to the
    reconstructed path. Combined with `strobe_codes` failing on bit-packed input -- inside a bare
    `except` that set `position_codes = None` -- BOTH routes to a docked window were dead, and the
    only visible symptom was a note saying the session lacked position codes.

    That is why the `docked: true` switch raised on the first session it touched rather than
    producing a quietly wrong mask: `docked_mask_any` returns None when neither route works, and
    `rest_mask` refuses to write a mask named `docked` that is not.
    """
    from wfield_local import config
    from wfield_local.spout_behavior import discover_sessions

    try:
        animal, mmdd = str(label).split("_")[:2]
        cands = discover_sessions(config.resolver(), f"2026{mmdd}", [animal])
    except Exception:                                                  # noqa: BLE001
        return None
    return cands[0] if cands else None
