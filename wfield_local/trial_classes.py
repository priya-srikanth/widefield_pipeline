"""THE trial classification: one behavioural class per trial, used by every analysis (Priya 11-03: "keep the 4
kinematics classes, jaw from cam4 OR cam1, time window cue to stop ... then we use this one definition for all").

What each trial is, first match wins, inside the chosen WINDOW:

  contact      the spout-contact sensor (DAQ) registered a lick                      -> the animal licked and hit
  incomplete   a lick seen on video (cam4 Lightning Pose lick, or a lick only cam1 saw; water droplets removed),
               or cam1 saw the tongue, but no spout contact                          -> licked and missed
  jaw_only     no lick; the jaw moved after the cue on cam4 OR cam1                  -> a reaction, NOT an attempt
               (Priya's 10-21 blind review: 0 / 24 such trials were attempted licks)
  no_response  no lick; the jaw was judged and stayed still                          -> no response at all
               AT close_center neither camera can judge the jaw (cam4's chin is hidden; cam1 failed the 10-21 blind
               review there): a trial with no lick, no contact and no tongue seen is NO_RESPONSE by default (Priya
               11-03: "if we don't see lick or jaw movement or spout contact ... default to no response"), flagged
               ``jaw_judged`` = False so it can be dropped as a sensitivity check. Jaw-only and no-response cannot be
               told apart there; jaw-only is ~.03-.05 of trials at the close spouts acutely.
  unknown      no lick, and the jaw could not be judged elsewhere: no jaw call on either camera, or a jaw already
               moving at the cue that never moved again (it may still have reacted)

Groupings (GATES), used to select trials:
  working      contact | incomplete | jaw_only   the animal responded to THIS trial's cue (imaging ENL + cue decoding)
  camera_lick  contact | incomplete              a lick was made (lick-aligned windows always need it)
  attempted    contact | incomplete              = camera_lick; the encoder page's "attempt" (10-21 rule)

WINDOW (keyword, default "cue_stop", Priya 11-03):
  "cue_stop"  licks from the cue to the trial stop; spout contacts from cue + 70 ms to the trial stop (the DAQ contact
              table built with the video outputs, `orofacial_outputs` / `pose_kinematics_demo.daq_contacts_ms`).
              This is the gate for ENL / pre-cue analyses too: it asks whether the animal responded to that trial's
              cue, while ENL decoding itself still uses pre-cue frames.
  "trial"     licks and contacts from the trial strobe to the stop, i.e. pre-cue licking counts as a lick. The jaw
              call is always the after-cue call (no pre-cue jaw call exists), so this option changes only the lick /
              contact classes. Contacts need the raw DAQ (slower, ~20-40 s per session).

Sources (all per session, nothing is recomputed here):
  v36 `sessions/<animal>/<date>/`: trials.csv (position, n_daq_contacts), licks.csv (cam4 licks, on_ms from the cue),
  windows.npz (cue / strobe / stop times on the DAQ clock);
  `session_poses/cam1_licks/<animal>_<date>_events.csv` (cam1 tongue events);
  `<v36>/imaging_encoding/qc/jaw_onsets.csv` (`scripts/jaw_onsets.py`: per-camera jaw-moved calls after the cue,
  onsets, the at-cue / re-onset flags).

Consumers import from HERE (never re-derive classes):
  scripts/jaw_only_incomplete_figs.py (kinematics page section 6, PI deck), scripts/trial_attempts.py (encoder-page
  attempt classes; keeps its own column names hit / no_try for its ~20 readers), the imaging section 1 table and the
  cohort-1 rebuild's working gate (`grant_kit._class_select`), and `responded(source="camera")` (consolidation A5).
Decision record: DECISIONS 2026-11-03; plan docs/status/REPO_CONSOLIDATION_PLAN_2026-11-03.md (A4).
"""
from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = ("contact", "incomplete", "jaw_only", "no_response", "unknown")
GATES = {"working": frozenset({"contact", "incomplete", "jaw_only"}),
         "camera_lick": frozenset({"contact", "incomplete"}),
         "attempted": frozenset({"contact", "incomplete"})}
WINDOWS = ("cue_stop", "trial")
DEFAULT_WINDOW = "cue_stop"
JAW_UNJUDGED_POSITIONS = ("close_center", "close_mid")   # neither camera can judge the jaw at this spout (10-21)
CONTACT_FROM_CUE_S = 0.070                               # DAQ contacts count from cue + 70 ms (daq_contacts_ms)
V36 = "lp_r4_heatmap_20261013__clean_v36"


def _flag(x: pd.Series) -> pd.Series:
    """Boolean column read safely: missing = False (used only for event flags, never for the jaw call)."""
    return x.astype("boolean").fillna(False).astype(bool)


def _paths(v36: Path | str | None, c1: Path | str | None) -> tuple[Path, Path]:
    from wfield_local.paths import PathResolver
    rv = PathResolver()
    v36 = Path(v36) if v36 else Path(rv.root("orofacial")) / V36 / "sessions"
    c1 = Path(c1) if c1 else Path(rv.root("dlc")) / "session_poses" / "cam1_licks"
    return v36, c1


@functools.lru_cache(maxsize=2)
def jaw_table(path: str) -> pd.DataFrame:
    """`jaw_onsets.csv`, indexed by (session = "PS92_20260818", trial_id): cam4_moved / cam1_moved (1 / 0 / NaN = not
    judged), onset_at_cue (the jaw was already moving at the cue), jaw_reonset_ms (its next movement, if any)."""
    want = ["session", "trial_id", "cam4_moved", "cam1_moved", "jaw_onset_ms", "onset_at_cue", "jaw_reonset_ms"]
    J = pd.read_csv(path, low_memory=False, usecols=lambda c: c in want)
    if "jaw_reonset_ms" not in J:
        J["jaw_reonset_ms"] = np.nan
    J["onset_at_cue"] = J.onset_at_cue.astype(str).str.lower().eq("true")
    for c in ("cam4_moved", "cam1_moved"):
        J[c] = pd.to_numeric(J[c].replace({"True": 1, "False": 0, True: 1, False: 0}), errors="coerce")
    return J.set_index(["session", "trial_id"]).sort_index()


def _trial_times(sd: Path) -> pd.DataFrame:
    """Per trial from windows.npz: cue time on the DAQ clock (s), strobe and stop (ms from the cue)."""
    w = np.load(sd / "windows.npz", allow_pickle=True)
    ix = pd.DataFrame({"trial_id": w["ix_trial_id"].astype(int), "t_ms": w["ix_t_ms"].astype(float),
                       "daq_s": w["ix_daq_s"].astype(float), "strobe_ms": w["ix_strobe_ms"].astype(float),
                       "stop_ms": w["ix_stop_ms"].astype(float)})
    ix["cue_daq"] = ix.daq_s - ix.t_ms / 1000.0
    return ix.groupby("trial_id", sort=True).agg(cue_daq=("cue_daq", "median"), strobe_ms=("strobe_ms", "first"),
                                                 stop_ms=("stop_ms", "first")).reset_index()


def _pre_cue_contacts(animal: str, date: str, times: pd.DataFrame) -> set:
    """Trials with a DAQ spout contact between the strobe and cue + 70 ms (only needed for window="trial")."""
    from wfield_local import dlc_frames
    from wfield_local.paths import PathResolver
    rv = PathResolver()
    sid = sorted((Path(rv.root("behavior_out")) / "sessions" / animal / date).glob("*_trials.csv"))[-1].name[:-11]
    licks = np.asarray(dlc_frames.lick_onsets(animal, date, sid, rv), float)
    out = set()
    for t, c, s in zip(times.trial_id, times.cue_daq, times.strobe_ms):
        lo, hi = c + s / 1000.0, c + CONTACT_FROM_CUE_S
        if np.any((licks >= lo) & (licks < hi)):
            out.add(int(t))
    return out


def session_trials(label: str, window: str = DEFAULT_WINDOW, v36: Path | str | None = None,
                   c1: Path | str | None = None, jaw_csv: Path | str | None = None) -> pd.DataFrame | None:
    """EVERY trial of one session (no engagement filter) with its class and the evidence behind it.

    label "PS92_0818" or "PS92_20260818". Returns None when the session has no v36 video outputs (no camera-defined
    classes exist; the caller must say what it does then -- there is deliberately no DAQ-only fallback here).
    Columns: trial_id, position, cue_daq (s), stop_ms, cls, window, has_cam1, n_contacts, n_licks, n_nocontact,
    first_lick_ms, jaw_cam4, jaw_cam1, jaw_at_cue, jaw_onset_ms, jaw_judged (False = the jaw call was not available;
    close_center no-response trials carry it, so they can be dropped as a sensitivity check).
    """
    if window not in WINDOWS:
        raise ValueError(f"window must be one of {WINDOWS}, got {window!r}")
    animal, date = label.split("_")
    date = date if len(date) == 8 else f"2026{date}"
    v36, c1 = _paths(v36, c1)
    sd = v36 / animal / date
    if not (sd / "trials.csv").exists() or not (sd / "windows.npz").exists():
        return None
    T = pd.read_csv(sd / "trials.csv", usecols=lambda c: c in ("trial_id", "position", "n_daq_contacts"))
    T = T.merge(_trial_times(sd), on="trial_id", how="left")
    lo_ms = (T.strobe_ms if window == "trial" else pd.Series(0.0, index=T.index)).to_numpy(float)
    lo = dict(zip(T.trial_id, lo_ms))
    hi = dict(zip(T.trial_id, T.stop_ms.to_numpy(float)))

    def in_win(df: pd.DataFrame) -> pd.Series:
        t = df.trial_id
        return (df.on_ms >= t.map(lo)) & (df.on_ms <= t.map(hi))

    # --- licks seen on video (cam4), in the window
    if (sd / "licks.csv").exists():
        L = pd.read_csv(sd / "licks.csv", usecols=["trial_id", "on_ms", "contact"])
    else:
        L = pd.DataFrame(columns=["trial_id", "on_ms", "contact"])
    L = L[in_win(L)]
    # --- cam1-only tongue events (no cam4 lick, no 3-D lick overlaps; droplets out), in the window
    fe = c1 / f"{animal}_{date}_events.csv"
    has_cam1 = fe.exists()
    if has_cam1:
        E = pd.read_csv(fe)
        for c in ("cam4_matched", "in_3d", "contact", "jaw_only_trial", "droplet"):
            E[c] = _flag(E[c]) if c in E else False
        seen = set(E.loc[E.jaw_only_trial & ~E.droplet, "trial_id"])        # cam1 saw a tongue, no lick counted it
        E = E[~E.cam4_matched & ~E.in_3d & ~E.droplet]
        E = E[in_win(E)] if "on_ms" in E else E.iloc[0:0]
    else:
        E, seen = pd.DataFrame(columns=["trial_id", "on_ms", "contact"]), set()
    t = T.trial_id
    T["n_licks"] = t.map(L.groupby("trial_id").size()).fillna(0) + t.map(E.groupby("trial_id").size()).fillna(0)
    nc4 = (~_flag(L.contact)).groupby(L.trial_id).sum() if len(L) else pd.Series(dtype=float)
    nc1 = (~_flag(E.contact)).groupby(E.trial_id).sum() if len(E) else pd.Series(dtype=float)
    T["n_nocontact"] = t.map(nc4).fillna(0) + t.map(nc1).fillna(0)
    first = pd.concat([L[["trial_id", "on_ms"]], E[["trial_id", "on_ms"]]]).groupby("trial_id").on_ms.min()
    T["first_lick_ms"] = t.map(first)
    # --- spout contact (DAQ sensor = primary, Priya 11-03): cue + 70 ms .. stop, from the contact table
    if "n_daq_contacts" not in T:
        raise ValueError(f"{label}: trials.csv has no n_daq_contacts (video outputs built without the DAQ contact "
                         "table) -- rebuild the session's outputs; no silent fallback to video-matched contact")
    T["n_contacts"] = pd.to_numeric(T.n_daq_contacts, errors="coerce").fillna(0)
    if window == "trial":
        pre = _pre_cue_contacts(animal, date, T)
        T["n_contacts"] = T.n_contacts + t.isin(pre).astype(int)
    # --- jaw, cam4 OR cam1, after the cue (jaw_onsets.csv); unjudgeable at close_center
    jaw_csv = Path(jaw_csv) if jaw_csv else v36.parent / "imaging_encoding" / "qc" / "jaw_onsets.csv"
    key = f"{animal}_{date}"
    J = jaw_table(str(jaw_csv)) if jaw_csv.exists() else None
    if J is not None and key in J.index.get_level_values(0):
        j = J.loc[key].reindex(t.to_numpy())
    else:
        j = pd.DataFrame(index=t.to_numpy(), columns=["cam4_moved", "cam1_moved", "jaw_onset_ms", "onset_at_cue",
                                                      "jaw_reonset_ms"], dtype=float)
    hidden = T.position.isin(JAW_UNJUDGED_POSITIONS).to_numpy()
    T["jaw_cam4"] = np.where(hidden, np.nan, pd.to_numeric(j.cam4_moved, errors="coerce").to_numpy(float))
    T["jaw_cam1"] = np.where(hidden, np.nan, pd.to_numeric(j.cam1_moved, errors="coerce").to_numpy(float))
    at_cue = j.onset_at_cue.fillna(False).astype(bool).to_numpy()
    reon = pd.to_numeric(j.jaw_reonset_ms, errors="coerce").to_numpy(float)
    on = pd.to_numeric(j.jaw_onset_ms, errors="coerce").to_numpy(float)
    T["jaw_at_cue"] = at_cue
    T["jaw_onset_ms"] = np.where(hidden, np.nan, np.where(at_cue, reon, on))
    judged = (T.jaw_cam4.notna() | T.jaw_cam1.notna()).to_numpy() & (~at_cue | np.isfinite(reon))
    moved = ((T.jaw_cam4 == 1) | (T.jaw_cam1 == 1)).to_numpy() & judged
    T["jaw_judged"] = judged
    T["cls"] = np.select([T.n_contacts.to_numpy() > 0,
                          (T.n_licks.to_numpy() > 0) | t.isin(seen).to_numpy(),
                          moved, judged | hidden],          # close_center, nothing seen -> no_response (11-03)
                         ["contact", "incomplete", "jaw_only", "no_response"], "unknown")
    T["window"] = window
    T["has_cam1"] = has_cam1
    return T[["trial_id", "position", "cue_daq", "stop_ms", "cls", "window", "has_cam1", "n_contacts", "n_licks",
              "n_nocontact", "first_lick_ms", "jaw_cam4", "jaw_cam1", "jaw_at_cue", "jaw_onset_ms", "jaw_judged"]]


def in_gate(cls, gate: str) -> np.ndarray:
    """Boolean per trial: class in GATES[gate] ("working", "camera_lick" or "attempted")."""
    return np.isin(np.asarray(cls, dtype=object), list(GATES[gate]))


def selection_tag(source: str = "camera", window: str = DEFAULT_WINDOW) -> str:
    """Short tag naming the trial-selection rule; put it in every cache key / frozen-model spec built on a gate, so
    runs with different sources or windows never share a cache entry (agreed with the cohort-1 rebuild, 11-03)."""
    return f"sel-{source}-{window}"
