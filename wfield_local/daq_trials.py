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

from wfield_local import config, daq_io
import pandas as pd

# pos_idx -> name, matching the task controller's own indexing (and spout_behavior.POSITIONS)
POS_NAMES = {0: "close_center", 1: "close_L", 2: "close_R", 3: "far_center", 4: "far_L", 5: "far_R"}
N_POS = 6


_rising = daq_io.rising_edges          # one implementation, in daq_io

def fallback_response_window_s() -> float:
    """THE fallback response window (s), `behavior.response_window_fallback_s`: the one value every resolver uses when a
    session's gui_config has none (code review 11-04 item 6; behaviour had 2.0 s and imaging 3.5 s until 11-05)."""
    from wfield_local import config
    return float(config.defaults()["behavior"]["response_window_fallback_s"])


def response_window_s(session_dir: Path, default: float | None = None) -> tuple[float, str]:
    """The session's real response window (s) from ``gui_config.json``; the configured fallback if absent.

    Production callers pass NO ``default`` (`tests/test_response_window_fallback.py` checks): the fallback is
    `fallback_response_window_s`, so two consumers cannot score "responded" with different windows. ``default`` stays
    for unit tests of the parser.

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
    return float(fallback_response_window_s() if default is None else default), "default"


def decode(h5_path: Path, lick_params: dict | None = None) -> dict:
    """Decode the DAQ digital/analog streams into event times (seconds on the DAQ clock)."""
    with daq_io.open_daq(h5_path) as f:
        fs, _created = daq_io.session_attrs(f)
        dnames, bits = daq_io.digital_bits(f)
        lick_v = daq_io.analog_channel(f, "lick_analog", required=False)
        reward_v = daq_io.analog_channel(f, "reward_ttl", required=False)
        end_v = daq_io.analog_channel(f, "trial_end", required=False)

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
                           min_ili_s=lick_params.get("min_ili_ms", config.LICK_MIN_ILI_MS_DEFAULT) / 1000.0)
        licks = np.asarray(det["lick_onsets"], dtype=float) / fs

    reward = (_rising((reward_v > 2.5).astype(np.int8)) / fs) if reward_v is not None else np.array([], float)
    # the TRIAL STOP: the DAQ analog `trial_end` TTL (rising edge at 2.5 V; `trial_windows` module docstring)
    trial_end = (_rising((end_v > 2.5).astype(np.int8)) / fs) if end_v is not None else np.array([], float)
    return {"fs": fs, "cue_s": cue / fs, "strobe_s": strobe / fs, "trial_start_s": tstart / fs,
            "codes": codes, "lick_s": licks, "reward_s": reward, "trial_end_s": trial_end, "h5": str(h5_path)}


#: licks / contacts count from cue + 70 ms (= trial_classes.CONTACT_FROM_CUE_S; the tongue cannot have responded earlier)
CONTACT_FROM_CUE_S = 0.070


def trial_stops(cue_s, lick_s, response_window: float, trial_end_s=None):
    """``(stop_s, stop_source)`` per cue -- THE TRIAL STOP (Priya 11-03): the DAQ `trial_end` TTL, the first one after the
    cue and before the next cue; without one, the TASK RULE: the first contact (from cue + 70 ms) within
    ``response_window`` of the cue + ``response_window`` (after the first lick the animal has the window to keep
    licking; DECISIONS e9a7dc31 on main), else cue + window. Capped at the next cue. Measured on 6 sessions (3,105
    trials, log 8.11): hits stop - first contact median 3.528 s, misses stop - cue median 3.525 s."""
    cue = np.asarray(cue_s, float)
    lick = np.sort(np.asarray(lick_s, float))
    nxt = np.append(cue[1:], np.inf)
    stop = np.full(cue.size, np.nan)
    src = np.array(["rule"] * cue.size, dtype=object)
    te = np.sort(np.asarray(trial_end_s if trial_end_s is not None else [], float))
    if te.size:
        k = np.searchsorted(te, cue, side="right")
        ok = (k < te.size) & (te[np.clip(k, 0, te.size - 1)] < nxt)
        stop[ok] = te[k[ok]]
        src[ok] = "daq_trial_end"
    for i in np.flatnonzero(~np.isfinite(stop)):
        j = np.searchsorted(lick, cue[i] + CONTACT_FROM_CUE_S, side="left")
        first = lick[j] if j < lick.size and lick[j] <= cue[i] + response_window else None
        stop[i] = min((first if first is not None else cue[i]) + response_window, nxt[i])
    return stop, src


def licks_to_stop(cue_s, stop_s, lick_s) -> np.ndarray:
    """LICKS PER TRIAL (Priya 11-03): TOTAL licks from cue + 70 ms to the trial stop, per trial."""
    lick = np.sort(np.asarray(lick_s, float))
    return (np.searchsorted(lick, np.asarray(stop_s, float), side="right")
            - np.searchsorted(lick, np.asarray(cue_s, float) + CONTACT_FROM_CUE_S, side="left")).astype(int)


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

    stop, stop_src = trial_stops(cue, licks, response_window, dec.get("trial_end_s"))
    n_trial = licks_to_stop(cue, stop, licks)
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
            "n_licks_post": int(w.size),                 # licks in the RESPONSE window (hit / latency scoring)
            "stop_s": float(stop[k]), "stop_source": stop_src[k],
            "n_licks_trial": int(n_trial[k]),            # LICKS PER TRIAL: cue -> trial stop (Priya 11-03)
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


def trials_from_h5(h5_path: Path, session_dir: Path, lick_params: dict) -> tuple[pd.DataFrame | None, dict]:
    """Full path: decode -> quality gate -> score. ``(trials|None, info)``; None => use the log. The window is the
    session's own (`response_window_s`, configured fallback); until 11-05 the caller passed 2.0 s as the fallback."""
    dec = decode(h5_path, lick_params)
    ok, reason = quality(dec)
    win, win_src = response_window_s(session_dir)
    info = {"ok": ok, "reason": reason, "response_window_s": win, "window_source": win_src,
            "n_licks": int(dec["lick_s"].size), "h5": str(h5_path)}
    if not ok:
        return None, info
    return build_trials(dec, win), info
