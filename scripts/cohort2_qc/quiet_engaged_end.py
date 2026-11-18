"""Where the TASK ends in each session, for the quiet-event engaged gate (Priya 11-18, via a2 / 3c).

Before: `quiet_area_spikes.session_traces` ended "engaged" at the session's quit, and kept the WHOLE recording when the
quit was censored (no quit detected) -- so imaging after the task stopped counted as engaged quiet (PS92_0828: 6.0 of
8.4 quiet min and ~80 % of its events; PS93_0806 pre: an 11.7-min tail). Priya: when no quit is detected, engaged ends
at the LAST CUE + the session's response window. a2 (11-18): in PS92_0828 every Arduino line (sync heartbeat, cue,
trial_start) stops at 120.3 min while the DAQ and the PCO camera run to 127.5 min, so after that there is no sync and the
camera alignment is extrapolated -> clean task end = min(last cue + response window, last sync edge) (trial_start is
not used: it rises at the start of each trial, before its cue). Then Priya 11-18 (later): the analyzed time ends at the LAST SYNC PULSE -> QS_ENGAGED_END=lastsync
(primary); QS_ENGAGED_END=lastcue keeps the first rule (superseded); unset = old behavior (quit only).

Writes <cohort1_quiet_spikes>/engaged_end_20261118.csv, one row per session: last edges of cue / trial_start / sync /
pco_exposure (DAQ s), response window, task_end_s, and the post-task imaging tail (last PCO frame - task end, min);
sync_short_min = last PCO frame - last sync edge (sessions where the heartbeat stops well before the camera).

    PYTHONPATH=<worktree>;<worktree>/scripts python scripts/cohort2_qc/quiet_engaged_end.py [--jobs 6]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

CHANNELS = ("cue", "trial_start", "sync", "pco_exposure")


def table_path():
    from quiet_area_spikes import OUT
    return OUT / "engaged_end_20261118.csv"


def last_edges(h5):
    """{channel: last rising-edge time (DAQ s)} for CHANNELS, plus the recording length (s)."""
    import h5py
    with h5py.File(h5, "r") as h:
        names = [n.decode() for n in h["digital/channel_names"][:]]
        packed = h["digital/packed_samples"][:, 0]
        fs = float(h.attrs["sample_rate_hz"])
    out = {"daq_end_s": len(packed) / fs}
    for c in CHANNELS:
        if c not in names:
            out[f"last_{c}_s"] = np.nan
            continue
        line = (packed >> names.index(c)) & 1
        up = np.flatnonzero(np.diff(line.astype(np.int8)) > 0)
        out[f"last_{c}_s"] = (up[-1] + 1) / fs if len(up) else np.nan
    return out


def one(label):
    from wfield_local import config
    from wfield_local.analysis_kit import response_window_s
    s = next(x for x in config.load_sessions() if x["label"] == label)
    r = dict(label=label, **last_edges(s["h5"]), rw_s=float(response_window_s(s)))
    # NOT trial_start: it rises at the START of each trial, before that trial's cue, so its last edge would cut the last
    # trial off (PS93_0806: 7.5 s). The sync heartbeat stops when the task controller stops.
    r["task_end_s"] = float(np.nanmin([r["last_cue_s"] + r["rw_s"], r["last_sync_s"]]))
    r["tail_min"] = (r["last_pco_exposure_s"] - r["task_end_s"]) / 60
    r["sync_short_min"] = (r["last_pco_exposure_s"] - r["last_sync_s"]) / 60
    return r


def task_end_s(label, rule="lastcue"):
    """The session's analyzed-time end (DAQ s) from the table (computed for this one session if missing).
    rule "lastsync" (Priya 11-18, the PRIMARY rule: "the end of the analyzed time is the LAST SYNC PULSE") = the last
    rising edge of the sync heartbeat; "lastcue" (first 11-18 rule, superseded) = min(last cue + response window, last
    sync edge). The two differ by the sync-after-cue gap, at most ~22 s (table: sync stops 0-25 s after the last cue)."""
    import pandas as pd
    col = {"lastsync": "last_sync_s", "lastcue": "task_end_s"}[rule]
    f = table_path()
    if f.exists():
        T = pd.read_csv(f).set_index("label")
        if label in T.index:
            return float(T.loc[label, col])
    return float(one(label)[col])


def main():
    import pandas as pd
    from wfield_local import config
    from wfield_local.analysis_kit import curated_labels, fan_sessions
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    labels = [k for k in sorted(curated_labels()) if not config.early_pre(k)]
    res, fail = fan_sessions(labels, one, jobs=a.jobs, label="session")
    for x in fail:
        print("FAILED", x)
    T = pd.DataFrame([v for _, v in res]).sort_values("label")
    f = table_path()
    if f.exists():
        raise SystemExit(f"exists, not overwriting: {f}")
    T.to_csv(f, index=False)
    pd.set_option("display.width", 220)
    print(f"{len(T)} sessions -> {f}")
    print("\npost-task imaging tail > 1 min (last PCO frame - task end):")
    print(T[T.tail_min > 1][["label", "last_cue_s", "rw_s", "last_sync_s", "last_pco_exposure_s", "tail_min",
                             "sync_short_min"]].round(2).to_string(index=False))
    print("\nsync heartbeat stops > 1 min before the camera:")
    print(T[T.sync_short_min > 1][["label", "last_sync_s", "last_pco_exposure_s", "sync_short_min"]].round(2)
          .to_string(index=False))


if __name__ == "__main__":
    main()
