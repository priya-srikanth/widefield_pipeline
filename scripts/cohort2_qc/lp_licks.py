"""LP (video) lick and jaw-onset times on the DAQ clock, whole session, for the quiet-event chain (cohort1, 2026-11-01).

Priya 11-01: "replace all contact-based with LP-lick based where appropriate". The quiet scripts defined QUIET from DAQ
spout-CONTACT licks (quiet_spikes_v2.busy_frames); acutely many licks never touch the spout and are seen only in video.
This module is the one place those video licks are loaded for the quiet scripts. Not to be confused with
`wfield_local.lp_licks` (the imaging LP_licks family: per-trial post-cue gate, trial-window cam1 events only); this one
covers the WHOLE session and uses the cam1 whole-session table.

TABLES READ (all read-only):
  cam4   `<v36 run>/sessions/<animal>/<date8>/licks_all.csv` (`orofacial_outputs all-licks`: every cam4 LP lick of the
         trial windows, contexts pre_cue / post_cue / iti), column `daq_on_s`.
  cam1   `<dlc root>/session_poses/cam1_licks/<animal>_<date8>_events_all.csv` (`scripts/cam1_licks.py`, the cam1 2-D
         detector over the whole windows), column `daq_on_s`, keeping only CAM1-ONLY events (`cam4_matched` False:
         no overlap with any cam4 lick of licks_all) that are not water droplets (`droplet` False). If events_all is
         absent, the post-cue `<s>_events.csv` is used (trial id + `on_ms` from the cue, cue from windows.npz).
  jaw    `<v36 run>/imaging_encoding/qc/jaw_onsets.csv` (`scripts/jaw_onsets.py`): per trial `jaw_onset_ms` and
         `jaw_reonset_ms` (ms from the cue; cue response only -- there are no ITI jaw onsets).
DAQ CLOCK: `daq_on_s` in both lick tables is the absolute DAQ time (s since the DAQ recording start) of the onset frame,
  mapped from the camera timestamp through the camera <-> DAQ alignment template (`camera_sync`); this is the same clock
  as the DAQ event npz (`lick_onsets / fs`) and the imaging frame times used by busy_frames. Cue-relative times (jaw,
  post-cue cam1 fallback) are put on it with the trial's cue time from windows.npz (`orofacial_cohort._cue_table`:
  median of daq_s - t_ms over the trial's window frames). Checked 11-01 (PS92_0811): 93 % of LP onsets lie within
  0.25 s of a DAQ contact onset, median 5.6 ms BEFORE it (tongue out before contact): the clocks agree.
DEDUP: cam1 events that overlap a cam4 lick are already excluded by `cam4_matched`; after merging, any onset < 40 ms
  after the previous kept onset is dropped (the pipeline's physiological lick floor, defaults.yaml
  lick_detection.min_ili_ms = 40 ms), which removes residual cam4 / cam1 double counts of one lick.
COVERAGE: video is tracked only inside the trial windows (strobe - 0.5 s .. stop + 3 s, cut at the next trial; ~98-99 %
  of the video in most sessions, less in split sessions); `video_windows(label)` returns them, so a caller can tell
  'no LP lick' from 'not tracked'. No video (e.g. PS93_0813): empty array + a printed warning.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

MIN_ILI_S = 0.040
_JAW: dict = {}
_WARNED: set = set()


def _paths(label: str):
    from wfield_local import config
    from wfield_local import lp_licks as LPL
    from wfield_local.paths import PathResolver
    rv = PathResolver()
    sd, f4, f1 = LPL._inputs(label, rv)
    an, d = label.split("_")
    return sd, f4, f1, f1.with_name(f1.name.replace("_events.csv", "_events_all.csv")), \
        LPL._enc_dir(rv) / "qc" / "jaw_onsets.csv", f"{an}_{config.date8(d)}"


def _cues(sd: Path):
    from wfield_local import orofacial_cohort as oc
    return oc._cue_table(sd).set_index("trial_id").cue_daq


def _warn(label, what):
    if (label, what) not in _WARNED:
        _WARNED.add((label, what))
        print(f"lp_licks WARNING {label}: {what}")


def dedup(t, min_ili=MIN_ILI_S) -> np.ndarray:
    t = np.sort(np.asarray(t, float)[np.isfinite(t)])
    if not len(t):
        return t
    keep, last = [t[0]], t[0]
    for x in t[1:]:
        if x - last >= min_ili:
            keep.append(x)
            last = x
    return np.array(keep)


def lick_times(label: str, return_parts: bool = False):
    """Sorted LP video lick onsets (cam4 + cam1-only, deduplicated) in DAQ seconds; empty array if no video."""
    import pandas as pd
    sd, f4, f1, f1a, _, _ = _paths(label)
    if not f4.exists():
        _warn(label, f"no cam4 LP table ({f4}); returning no video licks")
        return (np.zeros(0), {"cam4": 0, "cam1": 0, "cam1_src": "none"}) if return_parts else np.zeros(0)
    t4 = pd.read_csv(f4, usecols=["daq_on_s"]).daq_on_s.to_numpy(float)
    t1, src = np.zeros(0), "none"
    if f1a.exists():
        E = pd.read_csv(f1a, usecols=["daq_on_s", "cam4_matched", "droplet"])
        only = ~E.cam4_matched.fillna(False).astype(bool) & ~E.droplet.fillna(False).astype(bool)
        t1, src = E.daq_on_s[only].to_numpy(float), "events_all"
    elif f1.exists() and (sd / "windows.npz").exists():
        cue = _cues(sd)
        E = pd.read_csv(f1)
        only = ~E.cam4_matched.fillna(False).astype(bool) & ~E.droplet.fillna(False).astype(bool)
        E = E[only & E.trial_id.isin(cue.index)]
        t1, src = E.trial_id.map(cue).to_numpy(float) + E.on_ms.to_numpy(float) / 1000.0, "events_postcue"
        _warn(label, "no cam1 events_all; using post-cue cam1 events only")
    else:
        _warn(label, "no cam1 events; cam4 LP licks only")
    t = dedup(np.r_[t4, t1])
    parts = {"cam4": int(np.isfinite(t4).sum()), "cam1": int(np.isfinite(t1).sum()), "cam1_src": src}
    return (t, parts) if return_parts else t


def jaw_onsets(label: str):
    """Sorted jaw onsets + re-onsets (cue response, both cameras' combined call) in DAQ seconds; None if unavailable."""
    import pandas as pd
    sd, _, _, _, fj, sess = _paths(label)
    if not fj.exists() or not (sd / "windows.npz").exists():
        return None
    key = (str(fj), fj.stat().st_mtime)
    if key not in _JAW:
        _JAW.clear()
        _JAW[key] = pd.read_csv(fj, usecols=["session", "trial_id", "jaw_onset_ms", "jaw_reonset_ms"])
    J = _JAW[key]
    J = J[J.session == sess]
    if not len(J):
        return None
    c = J.trial_id.map(_cues(sd)).to_numpy(float)
    t = np.r_[c + J.jaw_onset_ms.to_numpy(float) / 1000.0, c + J.jaw_reonset_ms.to_numpy(float) / 1000.0]
    return np.sort(t[np.isfinite(t)])


def video_windows(label: str):
    """(n x 2) DAQ-second spans of the tracked video trial windows; empty if no video."""
    import pandas as pd
    sd = _paths(label)[0]
    if not (sd / "windows.npz").exists():
        return np.zeros((0, 2))
    w = np.load(sd / "windows.npz")
    ix = pd.DataFrame({"trial_id": w["ix_trial_id"].astype(int), "daq_s": w["ix_daq_s"].astype(float)})
    return ix.groupby("trial_id").daq_s.agg(["min", "max"]).to_numpy(float)


if __name__ == "__main__":
    import sys
    for lab in sys.argv[1:] or ["PS92_0811", "PS93_0813"]:
        t, p = lick_times(lab, return_parts=True)
        j = jaw_onsets(lab)
        print(lab, len(t), p, "jaw", None if j is None else len(j), "windows", len(video_windows(lab)))
