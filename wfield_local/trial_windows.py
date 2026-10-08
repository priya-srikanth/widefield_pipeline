"""Per-trial analysis windows from the trial's OWN timing: position strobe -> cue -> trial stop.

    from wfield_local import trial_windows as TW
    b = TW.trial_bounds(animal, date, rv)          # trial_id, pos_name, strobe_s, cue_s, stop_s, stop_source
    p_trial = TW.end_at_stop(p, stop_ms)           # a kinematics params dict with its post-cue ends at trial stop

WHY (Priya, 2026-10-01): the ported stroke_orofacial code hard-codes post-cue windows (lick detection to 3000 ms,
counts / dynamics / angles / jaw / mismatch to 5000 ms, detection slack to 8000 ms) from THEIR task. Ours has a
3.5 s response window -- "but should just define as interval between position strobe and trial stop, in case
this changes in future data". Measured (PS92 0821, PS93 0908, PS95 0917): trial stop is cue + 3.68-3.76 s
(median), 5th-95th percentile 3.53-5.37 s -- it VARIES per trial, so a fixed window is wrong for many trials.
The position strobe precedes the cue by a median 2.7-4.6 s with a tail to ~23 s (a pre-cue lick delays the cue).

WHERE THE TIMES COME FROM. strobe and cue: DAQ digital lines (`daq_trials.decode`; strobe = the most recent one
at or before the cue, the pipeline's pairing rule). trial stop: the DAQ ANALOG channel `trial_end` (a ~35 ms,
0-4 V TTL; rising edge at 2.5 V) -- the first edge after the cue and before the next cue. Fallbacks, in order,
with `stop_source` saying which: the behaviour log's `trial_stop_ttl` mapped onto the DAQ clock through the
shared Arduino heartbeat (`spout_behavior._sync_affine`); then cue + the session's response window
(`gui_config.json timing.response_window`). Checked on PS93 0908: DAQ trial_end vs the log-mapped stop, 510/510
trials, median -2.2 ms, max |difference| 4.3 ms. (A first version of this module said there was no DAQ line;
there is -- it is analog, not digital. Priya, 2026-10-01.)

THE RULE (`end_at_stop`), applied per trial when its stop is known:
  * RESPONSE windows end at the trial stop: kept-lick detection, peak velocity, lick-count apply, licking
    dynamics, angle, the bout table, the jaw detection window, the mismatch quiet-tongue window.
  * DETECTION-SLACK windows end at trial stop + `slack_ms` (3000 = their 8000 - 5000, so a lick straddling the
    end is still found whole): the trial slice and the lick-count detect window.
  * Window STARTS are unchanged (70 ms after the cue for licks: before that the tongue cannot have responded).
  * The 1-s lick-count bins (`lick_count_bin_win_ms`, 5 bins) stay fixed: they are a time-course readout, not a
    response window; a bin past the trial stop simply counts nothing.
  * No stop time -> the dict is returned unchanged, so code and tests written against the ported values keep
    those values.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd

#: tongue_kinematics keys whose END becomes the trial stop
TONGUE_END_AT_STOP = ("lick12_detect_win_ms", "peak_velocity_detect_win_ms", "lick_count_apply_win_ms",
                      "licking_dyn_apply_win_ms", "angle_apply_win_ms", ("bout_table", "detect_win_ms"))
#: tongue_kinematics keys whose END becomes trial stop + slack
TONGUE_END_AT_STOP_PLUS_SLACK = ("trial_slice_win_ms", "lick_count_detect_win_ms", ("bout_table", "plot_win_ms"))
DEFAULT_SLACK_MS = 3000.0


def _set_end(d: dict, key, end: float) -> None:
    if isinstance(key, tuple):
        d = d[key[0]]
        key = key[1]
    lo = float(d[key][0])
    d[key] = [lo, max(float(end), lo)]


def end_at_stop(p: dict, stop_ms: float | None, slack_ms: float = DEFAULT_SLACK_MS,
                end_at_stop_keys=TONGUE_END_AT_STOP, slack_keys=TONGUE_END_AT_STOP_PLUS_SLACK) -> dict:
    """A copy of ``p`` with response windows ending at ``stop_ms`` (ms after the cue) and slack windows at
    ``stop_ms + slack_ms``. ``stop_ms`` None/NaN -> ``p`` unchanged (the ported fixed windows)."""
    if stop_ms is None or not np.isfinite(stop_ms):
        return p
    q = copy.deepcopy(p)
    for k in end_at_stop_keys:
        _set_end(q, k, stop_ms)
    for k in slack_keys:
        _set_end(q, k, stop_ms + slack_ms)
    return q


def stop_ms_of(cue_frame: float, stop_frame, fps: float) -> float | None:
    """ms from the cue to the trial stop, or None when there is no stop frame."""
    if stop_frame is None or not np.isfinite(stop_frame) or not np.isfinite(cue_frame):
        return None
    return (float(stop_frame) - float(cue_frame)) * 1000.0 / float(fps)


def to_frames(t_s, tpl: dict) -> np.ndarray:
    """DAQ seconds -> camera frame (float), the alignment template's affine."""
    fs = float(tpl["fs_daq"])
    return (np.asarray(t_s, float) * fs - float(tpl["intercept_daqSample"])) / float(tpl["slope_daqSample_per_camFrame"])


def trial_bounds(animal: str, date: str, rv=None) -> pd.DataFrame:
    """Per trial (the DAQ trial table's rows): trial_id, pos_name, strobe_s, cue_s, stop_s (DAQ seconds),
    stop_source ('log_trial_stop_ttl' | 'cue+response_window')."""
    import json

    from wfield_local import daq_io, daq_trials
    from wfield_local.docked_periods import behaviour_session_dir
    from wfield_local.paths import PathResolver
    from wfield_local.spout_behavior import _sync_affine

    rv = rv or PathResolver()
    t = pd.read_csv(sorted((Path(rv.root("behavior_out")) / "sessions" / animal / date)
                           .glob(f"{animal}_{date}_*_trials.csv"))[-1])
    t = t[np.isfinite(t["cue_s"].astype(float))].sort_values("cue_s").reset_index(drop=True)
    h5 = sorted((Path(rv.root("daq_recorder_output")) / date).glob(f"{animal}_{date}_*.h5"))[0]
    dec = daq_trials.decode(h5)
    strobe = np.sort(np.asarray(dec["strobe_s"], float))
    j = np.searchsorted(strobe, t["cue_s"].to_numpy(float), side="right") - 1
    t["strobe_s"] = np.where(j >= 0, strobe[np.clip(j, 0, None)], np.nan)

    with daq_io.open_daq(h5) as f:
        fs, _ = daq_io.session_attrs(f)
        te = daq_io.analog_channel(f, "trial_end", required=False)
    daq_stop = (np.flatnonzero(np.diff((te > 2.5).astype(np.int8)) == 1) + 1) / fs if te is not None else None

    stop = None
    sd = behaviour_session_dir(f"{animal}_{date[4:]}")
    rw = 3.5
    if sd is not None and Path(sd).is_dir():
        sd = Path(sd)
        try:
            rw = float(json.loads((sd / "gui_config.json").read_text())["timing"]["response_window"]) / 1000.0
        except (OSError, KeyError, ValueError, TypeError):
            pass
        ev = pd.read_csv(sd / "events.csv", usecols=["device_t_ms", "event_name"])
        with daq_io.open_daq(h5) as f:
            names, bits = daq_io.digital_bits(f)
            fs, _ = daq_io.session_attrs(f)
        sync = daq_io.rising_edges(bits[:, names.index("sync")]) / fs
        ab = _sync_affine(sync, ev.loc[ev.event_name == "sync", "device_t_ms"].to_numpy(float) / 1000.0)
        if ab is not None:
            a, b = ab
            stop = np.sort((ev.loc[ev.event_name == "trial_stop_ttl", "device_t_ms"].to_numpy(float) / 1000.0 - b) / a)
    cue = t["cue_s"].to_numpy(float)
    nxt = np.r_[cue[1:], np.inf]
    out_stop, src = np.full(len(t), np.nan), np.array(["cue+response_window"] * len(t), dtype=object)
    # Lowest priority first, so the better source overwrites: log-mapped, then the DAQ trial_end line.
    for times, name in ((stop, "log_trial_stop_ttl"), (daq_stop, "daq_trial_end")):
        if times is None or len(times) == 0:
            continue
        times = np.sort(np.asarray(times, float))
        k = np.searchsorted(times, cue, side="left")
        ok = (k < len(times)) & (times[np.clip(k, 0, len(times) - 1)] < nxt)   # first stop after this cue,
        out_stop[ok] = times[k[ok]]                                             # and before the next one
        src[ok] = name
    miss = ~np.isfinite(out_stop)
    out_stop[miss] = cue[miss] + rw
    t["stop_s"], t["stop_source"] = out_stop, src
    return t[["trial_id", "pos_name", "strobe_s", "cue_s", "stop_s", "stop_source"]]


def spout_motion_spans(animal: str, date: str, rv=None, pad_pre_s: float = 0.05, pad_post_s: float = 0.2):
    """DAQ-second ``(t0, t1)`` arrays of the spans in which the MOTORISED spout is travelling, from the behaviour
    log: each trial's ``dock_start -> dock`` (retract to the dock) and ``trial_start -> position`` (out to the next
    target), padded ``pad_pre_s`` before and ``pad_post_s`` after; or None when the log is missing or its clock will not align to the
    DAQ (`spout_behavior._sync_affine` refuses).

    Priya, 2026-10-08: exclude ONLY the spout-moving frames from movement analyses, not the whole trial end -> next
    strobe gap -- the docked interval between them has no apparatus motion and is the most stationary part of the
    session. Checked against video (PS93 0814, `null_potent/PS93_0814/spout_move_eta.png`): motion energy rises
    within one imaging frame of the logged start, peaks again at deceleration by the logged end, and is back near
    baseline ~0.15-0.2 s after it -- hence the asymmetric pads. First event of each name per trial_id (the GUI can repeat a retried move; `docked_periods.dock_events`).
    """
    from wfield_local import daq_io
    from wfield_local.docked_periods import behaviour_session_dir
    from wfield_local.paths import PathResolver
    from wfield_local.spout_behavior import _sync_affine

    rv = rv or PathResolver()
    sd = behaviour_session_dir(f"{animal}_{date[4:]}")
    if sd is None or not (Path(sd) / "events.csv").exists():
        return None
    ev = pd.read_csv(Path(sd) / "events.csv", usecols=["device_t_ms", "event_name", "trial_id"])
    h5 = sorted((Path(rv.root("daq_recorder_output")) / date).glob(f"{animal}_{date}_*.h5"))[0]
    with daq_io.open_daq(h5) as f:
        names, bits = daq_io.digital_bits(f)
        fs, _ = daq_io.session_attrs(f)
    sync = daq_io.rising_edges(bits[:, names.index("sync")]) / fs
    ab = _sync_affine(sync, ev.loc[ev.event_name == "sync", "device_t_ms"].to_numpy(float) / 1000.0)
    if ab is None:
        return None
    a, b = ab

    def first(name):
        sub = ev[ev.event_name == name].dropna(subset=["device_t_ms", "trial_id"])
        return (sub.groupby("trial_id")["device_t_ms"].min() / 1000.0 - b) / a

    t0, t1 = [], []
    for s_name, e_name in (("dock_start", "dock"), ("trial_start", "position")):
        both = pd.concat([first(s_name), first(e_name)], axis=1, keys=["s", "e"]).dropna()
        both = both[(both.e > both.s) & (both.e - both.s < 3.0)]       # a real move: ~0.65-1.0 s (Zaber, fixed speed)
        t0.append(both.s.to_numpy() - pad_pre_s)
        t1.append(both.e.to_numpy() + pad_post_s)
    t0, t1 = np.concatenate(t0), np.concatenate(t1)
    o = np.argsort(t0)
    return t0[o], t1[o]
