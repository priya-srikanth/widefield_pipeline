"""Canonical per-trial table reconstructed from the DAQ recorder ``.h5`` (behavior's trial source).

WHY THIS EXISTS — the task-controller GUI's ``trials.csv`` mislabels ``pos_idx``
-----------------------------------------------------------------------------
The GUI builds each trial row incrementally and lets **every later event row overwrite**
``pos_idx`` while the row is still open (``_update_trial_from_event_row``:
``if row.get("pos_idx","") != "": t["pos_idx"] = ...``), and an event with no explicit
``pos``/``idx`` falls back to the **live** device status (``latest_status["current_pos"]``).
A trial row is only finalized when the NEXT ``trial_start`` arrives, so any event that lands
after the firmware has already moved to the next position stamps that next position onto the
previous trial's row. Measured on complete logs: the disagreement hits **exactly** the trials
where the position changed (rate 1.000 vs 0.008 elsewhere) and ``gui[k-1] == daq[k]`` on 100%
of them — the GUI row is one trial ahead. ``pos_dist_mm_after_trial`` is corrupted the same way
(so an apparent "spout moved mid-trial" in the GUI log is a logging artifact, not a real move).
Unchanged from GUI v44 through v46 (identical code), so every session to date is affected.

The DAQ is correct by construction: firmware ``startTrial`` moves the spout, THEN calls
``emitPositionCode(currentTrialPos)``, which sets bit0/1/2 and only then pulses a 10 ms strobe
(``Behavior_MobileSpouts_Zaber_Arduino_v36.ino``). One strobe per trial, bits already settled,
~3 s before that trial's cue. The imaging side has always used this record
(``plot_spout_trial_averages._classify_cues``), so sourcing behavior from it here also makes the
two halves of the pipeline agree on trial identity by construction.

What we decode (all on the DAQ clock, sample rate from the file):
  cue (digital ``cue``)                  -> one trial per rising edge, the alignment point
  ``trial_start``                        -> trial onset; the pre-cue (ENL) window is [start, cue)
  ``spout_strobe`` + ``spout_bit0/1/2``  -> pos_idx, sampled at the strobe rising edge
  ``reward_ttl`` (analog)                -> reward delivery
  ``lick_analog``                        -> licks via the canonical ``lick_detection`` params

Scoring is OURS, not the GUI's: a trial is a hit if a lick falls in
``[cue, min(cue + response_window, next_cue)]``. The window is read PER SESSION from the
session's ``gui_config.json`` (``cue.response_window``, ms) because it is a task setting that
can be retuned — the sessions to date ran **3500 ms**, while ``defaults.yaml`` carried 2.0 s.

The behavior log stays the FALLBACK, never the primary: ``quality()`` gates the DAQ record
(one strobe per cue, codes in range, enough distinct positions), so a degraded strobe stream
(e.g. the Aug-2026 dead ``spout_bit1``) falls back to ``trials.csv`` rather than silently
producing wrong positions. Same policy as ``behavior_position.classify_cues_with_backup``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from wfield_local import daq_io
import pandas as pd

# pos_idx -> name, matching the task controller's own indexing (and spout_behavior.POSITIONS)
POS_NAMES = {0: "close_center", 1: "close_L", 2: "close_R", 3: "far_center", 4: "far_L", 5: "far_R"}
N_POS = 6


_rising = daq_io.rising_edges          # one implementation, in daq_io

def response_window_s(session_dir: Path, default: float) -> tuple[float, str]:
    """The session's real response window (s) from ``gui_config.json``; ``default`` if absent.

    Returns ``(seconds, source)``. The GUI stores it in MILLISECONDS under ``timing.response_window``
    (older/other layouts have used ``cue.response_window``; both are accepted), and writes its config
    values as STRINGS, hence the float() coercion.
    """
    for name in ("gui_config.json", "gui_config_end.json"):
        p = Path(session_dir) / name
        if not p.exists():
            continue
        try:
            cfg = json.loads(p.read_text())
        except Exception:
            continue
        for section in ("timing", "cue"):
            val = (cfg.get(section) or {}).get("response_window")
            if val in (None, ""):
                continue
            try:
                return float(val) / 1000.0, f"{name}:{section}"
            except (TypeError, ValueError):
                continue
    return float(default), "default"


def decode(h5_path: Path, lick_params: dict | None = None) -> dict:
    """Decode the DAQ digital/analog streams into event times (seconds on the DAQ clock)."""
    import h5py

    with daq_io.open_daq(h5_path) as f:
        fs, _created = daq_io.session_attrs(f)
        dnames, bits = daq_io.digital_bits(f)
        lick_v = daq_io.analog_channel(f, "lick_analog", required=False)
        reward_v = daq_io.analog_channel(f, "reward_ttl", required=False)

    idx = {n: i for i, n in enumerate(dnames)}
    cue = daq_io.rising_edges(bits[:, idx["cue"]])
    strobe = daq_io.rising_edges(bits[:, idx["spout_strobe"]])
    tstart = (daq_io.rising_edges(bits[:, idx["trial_start"]]) if "trial_start" in idx
              else np.array([], int))
    codes = daq_io.strobe_codes(bits, dnames, strobe)

    licks = np.array([], float)
    if lick_v is not None and lick_params is not None:
        from wfield_local.lick_detection import detect_licks

        det = detect_licks(lick_v, fs,
                           thresh_upper=lick_params["thresh_upper"],
                           thresh_lower=lick_params["thresh_lower"],
                           lockout_s=tuple(lick_params["lockout_falling_edge_s"]),
                           min_ili_s=lick_params.get("min_ili_ms", 40) / 1000.0)
        licks = np.asarray(det["lick_onsets"], dtype=float) / fs

    reward = (_rising((reward_v > 2.5).astype(np.int8)) / fs) if reward_v is not None else np.array([], float)
    return {"fs": fs, "cue_s": cue / fs, "strobe_s": strobe / fs, "trial_start_s": tstart / fs,
            "codes": codes, "lick_s": licks, "reward_s": reward, "h5": str(h5_path)}


def positions_for_cues(dec: dict) -> np.ndarray:
    """Position code per cue = the most recent strobe at or before it (-1 if none).

    Pairing by time rather than by index because the strobe count need not equal the cue count:
    ``moveToNamedPosition`` also emits a code on a manual/commanded move, so a session can carry
    extra strobes (e.g. the move that precedes the first trial). Same rule the imaging side uses
    (``plot_spout_trial_averages._classify_cues``), so both halves resolve identical positions.
    """
    j = np.searchsorted(dec["strobe_s"], dec["cue_s"], side="right") - 1
    out = np.where(j >= 0, dec["codes"][np.clip(j, 0, None)], -1).astype(int)
    return out


def quality(dec: dict, min_distinct_positions: int = 5) -> tuple[bool, str]:
    """Is this DAQ record trustworthy as the trial source? ``(ok, reason)``.

    Guards the failure mode the Aug-2026 dead ``spout_bit1`` produced: a strobe stream that
    decodes but collapses onto too few distinct positions. A caller that gets ``ok=False``
    must fall back to the behavior log rather than use these positions.
    """
    n_cue = dec["cue_s"].size
    if n_cue == 0:
        return False, "no cue pulses"
    if dec["strobe_s"].size == 0:
        return False, "no spout_strobe pulses"
    pos = positions_for_cues(dec)
    n_unpaired = int((pos < 0).sum())
    if n_unpaired:
        return False, f"{n_unpaired}/{n_cue} cues have no preceding strobe"
    if pos.max() >= N_POS:
        return False, f"position codes out of range [{int(pos.min())},{int(pos.max())}]"
    distinct = int(np.unique(pos).size)
    if distinct < min_distinct_positions:
        return False, f"only {distinct} distinct positions (dead strobe bit?)"
    extra = dec["strobe_s"].size - n_cue
    return True, (f"{n_cue} trials, {distinct} positions"
                  + (f", {extra} non-trial strobe(s)" if extra else ""))


def build_trials(dec: dict, response_window: float) -> pd.DataFrame:
    """Per-trial table scored from the DAQ: position, hit/miss, first-lick latency, reward.

    A trial's response window is capped at the next cue so a late lick can never be credited
    to two trials. Anticipatory licks = licks in the pre-cue (ENL) window ``[trial_start, cue)``,
    the DAQ-side equivalent of the GUI's ``pre_cue_reset_by_lick``.
    """
    cue = dec["cue_s"]
    n = cue.size
    codes = positions_for_cues(dec)
    licks, reward = dec["lick_s"], dec["reward_s"]
    nxt = np.append(cue[1:], np.inf)
    starts = dec["trial_start_s"]
    # each cue's own trial_start = the last one at or before it (ENL window opens there)
    if starts.size:
        j = np.searchsorted(starts, cue, side="right") - 1
        pre = np.where(j >= 0, starts[np.clip(j, 0, None)], cue)
    else:
        pre = cue

    rows = []
    for k in range(n):
        c, nx = cue[k], nxt[k]
        end = min(c + response_window, nx)
        w = licks[(licks >= c) & (licks <= end)]
        n_pre = int(((licks >= pre[k]) & (licks < c)).sum())
        rewarded = bool(((reward >= c) & (reward <= nx)).any())
        rows.append({
            "trial_id": k + 1,
            "pos_idx": codes[k],
            "pos_name": POS_NAMES.get(codes[k], "?"),
            "cue_s": c,
            "trial_start_s": pre[k],
            "lick_in_response_window": int(w.size > 0),
            "hit": int(w.size > 0),
            "miss": int(w.size == 0),
            "latency_s": float(w[0] - c) if w.size else np.nan,
            "n_licks_post": int(w.size),
            "n_licks_pre": n_pre,
            "reward_delivered": int(rewarded),
        })
    df = pd.DataFrame(rows)
    # the lick train rides along so the figures can work entirely on the DAQ clock — no
    # events.csv, no sync fit (and so a session whose events.csv is empty still analyses)
    df.attrs["lick_s"] = licks
    df.attrs["fs"] = dec["fs"]
    df["responded"] = df["lick_in_response_window"].astype(bool)
    # the DAQ cannot know the task's "free reward" designation; spout_behavior merges that
    # from the behavior log when one is available, and treats it as absent otherwise
    df["is_free"] = False
    df["free_designated"] = False
    df["source"] = "DAQ"
    return df


def trials_from_h5(h5_path: Path, session_dir: Path, lick_params: dict,
                   default_window_s: float) -> tuple[pd.DataFrame | None, dict]:
    """Full path: decode -> quality gate -> score. ``(trials|None, info)``; None => use the log."""
    dec = decode(h5_path, lick_params)
    ok, reason = quality(dec)
    win, win_src = response_window_s(session_dir, default_window_s)
    info = {"ok": ok, "reason": reason, "response_window_s": win, "window_source": win_src,
            "n_licks": int(dec["lick_s"].size), "h5": str(h5_path)}
    if not ok:
        return None, info
    return build_trials(dec, win), info
