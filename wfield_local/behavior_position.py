"""Behavior-log spout-position BACKUP.

When the DAQ spout-strobe bits are faulty (the 2026-08 sessions had DAQ line5 `spout_bit1` dead — 0 edges —
which dropped close_R + far_center and mis-coded them onto other positions), the per-cue position from
`_classify_cues` is wrong. The behavior task controller logs the TRUE position per trial in `trials.csv`
(`pos_idx`, same 0-5 convention as POSITION_NAMES).

The behavior session starts AFTER the DAQ recording and the two use independent trial counters, so cue 0 is
not trial 0. We find the integer trial-offset that MAXIMIZES agreement with the positions the DAQ DID get
right — that offset absorbs the behavior-vs-DAQ startup difference; the alignment is then drift-free
(index-based, not a wall-clock offset that would slip over a long session). Cross-checked on 8/5: the DAQ
has exactly one cue per behavior trial, and the best offset reproduces the behavior log's own per-position
counts at 100% agreement on the 4 good positions. Only overrides when it validates (>=0.9 on the DAQ's good
positions) AND the DAQ is actually broken (<6 positions); otherwise returns the DAQ codes unchanged. If the
DAQ-cue and behavior-trial counts don't correspond 1:1 (e.g. the DAQ stopped before the behavior session
ended), validation fails and we keep the DAQ codes — so good sessions are never altered.

Behavior logs: M:/MICROSCOPE/Priya/Behavior_logs/Widefield/<mouse>_<YYYYMMDD>_*/trials.csv .
"""
from __future__ import annotations

import csv
import glob
import os
import re

import numpy as np

from wfield_local import config
from wfield_local.plot_spout_trial_averages import _classify_cues
from wfield_local.framemap_event_maps import _behavior_cue_codes  # recovered-position override

# Behavior-logs root. NOT a hardcoded mount: this used to be "M:/MICROSCOPE/Priya/Behavior_logs/Widefield"
# (the ANALYSIS box's mount), which silently broke the repair on every other machine -- the imaging box
# (MICROSCOPE = N:) and any helper box where M: is the STANDBY share. `_behavior_positions` then returned
# None and `classify_cues_with_backup` fell through to the unrepaired DAQ codes WITHOUT WARNING, leaving
# the dead-`spout_bit1` sessions (8/5-8/6) at 4 of 6 positions with ~1/3 of cues mislabelled. Resolved
# per-machine via configs/paths.yaml instead. Set BEH_ROOT_OVERRIDE to force a path (tests/ad-hoc).
BEH_ROOT_OVERRIDE = None
_BEH_ROOT = None


def beh_root():
    """This machine's behavior-logs root (configs/paths.yaml `behavior_logs`), resolved once."""
    global _BEH_ROOT
    if BEH_ROOT_OVERRIDE:
        return BEH_ROOT_OVERRIDE
    if _BEH_ROOT is None:
        _BEH_ROOT = config.resolver().root("behavior_logs")
    return _BEH_ROOT


def _behavior_dir(session):
    lab = session["label"]; mouse = lab[:4]
    m = re.search(r"/labcams/(\d{8})/", session["mc"].replace("\\", "/"))
    if not m:
        return None
    hits = sorted(glob.glob(f"{beh_root()}/{mouse}_{m.group(1)}_*"))
    return hits[0] if hits else None


def _is_true(v):
    return str(v).strip().lower() in ("1", "true", "yes", "t", "y")


def _scored_rows(rows):
    """Real, SCORED trials — one row per DAQ cue.

    Two filters, in order:
      1. ``device_trial_start_ms != device_trial_end_ms`` — drops never-closed rows.
      2. ``hit`` or ``miss`` must be set — drops the UNSCORED STUB rows that the task controller
         began emitting with **v47** (deployed 2026-08-10).

    On (2): v47 fixed the old ``pos_idx`` overwriting (docs/GUI_TRIALS_LOGGING.md) by closing the
    colliding row and opening a fresh one instead of overwriting it — but it KEEPS that stub in the
    CSV. The stub carries a duplicate ``trial_id``, a sub-second duration, and neither hit nor miss.
    Measured 2026-08-12: PS94 8/10 had 289 such rows (1009 -> 720) and 8/11 had 118 (554 -> 436),
    in both cases restoring the log EXACTLY to the DAQ cue count; pre-v47 PS94 8/6 has zero, so the
    filter is a no-op on older sessions.

    This matters because ``classify_cues_with_backup`` aligns the log to the DAQ by an integer
    trial-offset and only substitutes when agreement is >= 0.9. With the stubs left in, the two
    sequences differ in LENGTH and no offset can align them, so the repair silently fails to
    validate and a dead-strobe v47 session would be left with degraded positions and no fallback.
    """
    real = [r for r in rows if r.get("device_trial_start_ms") != r.get("device_trial_end_ms")]
    if not real or ("hit" not in real[0] and "miss" not in real[0]):
        return real                      # older schema without hit/miss -> nothing to filter on
    return [r for r in real if _is_true(r.get("hit")) or _is_true(r.get("miss"))]


def _behavior_positions(session):
    """True per-trial pos_idx (0-5) from the task controller's trials.csv, scored trials only."""
    d = _behavior_dir(session)
    if not d:
        return None
    f = os.path.join(d, "trials.csv")
    if not os.path.exists(f):
        return None
    try:
        with open(f, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        b = np.array([int(r["pos_idx"]) for r in _scored_rows(rows)], dtype=np.int16)
        return b if b.size >= 10 else None
    except Exception:
        return None


def classify_cues_with_backup(session, cue, verbose=True):
    daq = _classify_cues(cue["cue_samples"], cue["strobe_samples"], cue["strobe_codes"])
    # Explicit human-verified recovered-position CSV (e.g. PS93 8/5 cam1 recovery, when the DAQ strobe
    # bit is dead AND the behavior log is empty). Takes priority: same order+bitmask aligner the map
    # pipeline (_process) uses, so decoder/encoder and maps agree. Only set for known-broken sessions.
    bt = session.get("behavior_trials")
    if bt and os.path.exists(bt):
        try:
            rec = np.asarray(_behavior_cue_codes(bt, daq), dtype=np.int16)
            if verbose:
                print(f"  [recovered-positions] {session['label']}: using {os.path.basename(bt)}; "
                      f"positions {sorted(set(int(x) for x in rec if x >= 0))}", flush=True)
            return rec
        except SystemExit as ex:
            if verbose:
                print(f"  [recovered-positions] {session['label']}: {ex} -> fell back to DAQ/log", flush=True)
    good = sorted(set(int(x) for x in daq if x >= 0))
    if len(good) >= 6:
        return daq                                    # DAQ has all 6 positions -> trust it
    b = _behavior_positions(session)
    if b is None:
        # LOUD: the DAQ is short a position AND we found no log to repair it from. This used to be a
        # silent `return daq`, which is how a mis-mounted BEH_ROOT produced 4-position sessions with no
        # trace in the logs. Never fail silently here -- a wrong position label is unrecoverable downstream.
        if verbose:
            print(f"  [behavior-backup] *** {session['label']}: DAQ has only {good} ({len(good)}<6 positions) "
                  f"and NO usable trials.csv was found under {beh_root()} -> KEEPING DEGRADED DAQ CODES. "
                  f"Position labels for this session are WRONG for the missing positions; check the mount "
                  f"and configs/paths.yaml behavior_logs.", flush=True)
        return daq
    N = len(daq); rng = max(20, abs(len(b) - N) + 10)
    best = (-1.0, 0, None)
    for off in range(-rng, rng + 1):                  # behavior[j] -> daq[j+off]
        out = np.full(N, -1, dtype=np.int16)
        k = np.arange(len(b)) + off; ok = (k >= 0) & (k < N)
        out[k[ok]] = b[ok]
        val = (out >= 0) & np.isin(daq, good) & np.isin(out, good)
        if val.sum() < 20:
            continue
        sc = float((daq[val] == out[val]).mean())
        if sc > best[0]:
            best = (sc, off, out)
    sc, off, out = best
    if out is not None and sc >= 0.9:
        res = out.copy(); miss = res < 0; res[miss] = daq[miss]
        if verbose:
            print(f"  [behavior-backup] {session['label']}: DAQ had {good} (<6); behavior log aligned at "
                  f"trial-offset {off:+d} (good-pos agreement {sc:.2f}, n_cues={N}, n_beh={len(b)}); "
                  f"positions now {sorted(set(int(x) for x in res if x >= 0))}", flush=True)
        return res
    if verbose:
        print(f"  [behavior-backup] {session['label']}: DAQ missing positions but behavior alignment could "
              f"not be validated (best {sc:.2f}, n_cues={N}, n_beh={len(b) if b is not None else 0}) -> "
              f"kept DAQ codes", flush=True)
    return daq
