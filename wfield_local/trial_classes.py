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

import contextlib
import contextvars
import functools
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = ("contact", "incomplete", "jaw_only", "no_response", "unknown")
GATES = {"working": frozenset({"contact", "incomplete", "jaw_only"}),
         "camera_lick": frozenset({"contact", "incomplete"}),
         "attempted": frozenset({"contact", "incomplete"}),
         # MISS WHILE WORKING on the NO-LICK arm (review M6, 11-03): responded but NO contact. A no-lick row (no DAQ
         # lick within max_rt) whose camera class is "contact" (a contact later in cue+70 ms .. stop) is in NEITHER
         # miss_working nor stopped -- to confirm with Priya (count them: `in_gate(cls, "working") & ~in_gate(cls,
         # "miss_working")`).
         "miss_working": frozenset({"incomplete", "jaw_only"})}
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


def selection_tag(source: str = "camera", window: str = DEFAULT_WINDOW, stopped: str | None = None) -> str:
    """Short tag naming the trial-selection rule; put it in every cache key / frozen-model spec built on a gate, so
    runs with different sources or windows never share a cache entry (agreed with the cohort-1 rebuild, 11-03).
    ``stopped`` = the camera stopped-rule name (`stopped_rule_name`, Priya 11-03), appended when given."""
    return f"sel-{source}-{window}" + (f"-{stopped}" if stopped else "")


# ------------------------------------------------------------------------------------------ the run's trial gate
# (cohort-1 rebuild 2026-11-03). WHICH RULE SELECTS "working" / "stopped" in the imaging families
# (`grant_kit._class_select` and the gates listed in `docs/status/REBUILD_CODE_20261103.md` item f):
#   daq     the DAQ quit-tail rule: working = engaged + unengaged outside the terminal quit period, stopped = the quit
#           period (`precue_engagement_states.engagement_gate`). UNTAGGED: every output made before 11-03.
#   camera  THIS module's GATES: working = contact | incomplete | jaw_only, stopped = no_response; "unknown" and
#           sessions without video (PS93_0813) are in NEITHER (no DAQ fallback; reconciliation D12). The lick class
#           stays the DAQ contact set in both modes (lick-aligned windows need a lick time).
# Setting: ``defaults.yaml trial_selection.gate`` (default daq), overridden by WIDEFIELD_TRIAL_GATE; the window by
# ``trial_selection.window`` / WIDEFIELD_TRIAL_WINDOW. Measures OF engagement (quit, decline, attempt rate) never use
# it. A5 HOOK: when `responded(source=...)` lands (e3, consolidation A5), `cue_classes_for` is the one place to route
# through it.
GATE_ENV = "WIDEFIELD_TRIAL_GATE"
WINDOW_ENV = "WIDEFIELD_TRIAL_WINDOW"
GATE_MODES = ("daq", "camera")
UNTAGGED_GATE = "daq"
#: CAMERA "STOPPED" (Priya 11-03, decided): a no_response trial that is part of a SUSTAINED stop. `cue_classes_for`
#: relabels such trials "stopped" (STOPPED_CLASS); every consumer selects stopped through STOPPED_CLASSES.
#:   rule "run" (PRIMARY)  a no_response trial is stopped only inside a run of >= `stop_run_min()` consecutive camera
#:                         no_response trials (`behavior.tail_min_misses`, 6 = the task auto_hold_after_miss threshold).
#:                         "unknown" trials neither break nor extend a run (skipped when counting) and are never
#:                         stopped. An ISOLATED no_response trial stays "no_response": in neither working nor stopped.
#:   rule "any" (SENSITIVITY)  every no_response trial is stopped (the earlier provisional rule).
#: Setting: ``trial_selection.stopped`` (run) / WIDEFIELD_TRIAL_STOPPED; the rule name enters `selection_tag`, so
#: caches and frozen specs of the two rules never meet.
STOPPED_CLASS = "stopped"
STOPPED_CLASSES = frozenset({STOPPED_CLASS})
STOPPED_ENV = "WIDEFIELD_TRIAL_STOPPED"
STOPPED_RULES = ("run", "any")
CUE_MATCH_S = 0.020          # a DAQ cue and a video trial are the same trial within 20 ms of cue time (as `lp_licks`)


# ------------------------------------------------------------------------------------------ FAIL CLOSED (11-03)
# Under gate=camera every trial-SELECTION gate must come from this module. A gate still on the DAQ quit-tail rule
# (`precue_engagement_states.engagement_gate`, the masks built on it) would put DAQ-selected trials into an output
# tagged sel-camera, so it REFUSES instead. Three legitimate DAQ-rule uses remain under camera, each opened explicitly
# with `daq_quit_rule(reason)`: a measure OF engagement (quit, engagement decline, attempt rate, the behaviour
# reporting gate), a labelled DAQ robustness row, and a camera-wired caller that builds the DAQ mask only for trial
# bookkeeping it then overrides. The register of every call site is tests/test_camera_gate_fail_closed.py.
class DaqGateRefused(SystemExit):
    """A SystemExit on purpose (as `frozen_models.LegacyStoreError`): the rest gates and many collectors wrap their gate
    in ``except Exception`` and FAIL OPEN, which would turn a refusal into a silently ungated session."""


_DAQ_RULE_OK: contextvars.ContextVar = contextvars.ContextVar("trial_classes_daq_rule_ok", default=None)


@contextlib.contextmanager
def daq_quit_rule(reason: str):
    """Allow the DAQ quit-tail rule inside this block under gate=camera, for ``reason`` (see the block comment)."""
    tok = _DAQ_RULE_OK.set(str(reason))
    try:
        yield
    finally:
        _DAQ_RULE_OK.reset(tok)


def require_gate(mode: str, where: str) -> None:
    """Raise `DaqGateRefused` unless the run's gate is ``mode`` (review M5): called at the top of a family that is NOT
    wired to the camera classes, so it refuses before any work (engagement_gate would refuse later anyway)."""
    if gate_mode() != mode:
        raise DaqGateRefused(f"{where} is not wired to the {gate_mode()} trial gate (it selects trials by the {mode} "
                             f"rule); run it with {GATE_ENV}={mode} or wire it to trial_classes")


def refuse_daq_gate(where: str) -> None:
    """Raise `DaqGateRefused` when the run's gate is camera and no `daq_quit_rule` block is open. No-op under daq."""
    if _DAQ_RULE_OK.get() is None and camera_gate_on():
        raise DaqGateRefused(
            f"camera trial gate ({GATE_ENV} / trial_selection.gate = camera): {where} is the DAQ quit-tail rule and is "
            "not wired to trial_classes, so its output would be mislabelled sel-camera. Wire it (trial_classes) or run "
            "with gate=daq; a measure OF engagement opens trial_classes.daq_quit_rule(reason).")


class RefusedMask:
    """Stands in for a DAQ quit-tail mask under gate=camera: any USE of it (numpy, ``~``, ``&``, ``len``, indexing)
    raises `DaqGateRefused`. Camera-wired readers never touch it (`grant_kit._class_select` reads ``cls``)."""
    __array_ufunc__ = None             # ndarray & mask -> NotImplemented -> our __rand__ (raises)

    def __init__(self, where: str):
        self.where = where

    def _refuse(self, *_a, **_k):
        raise DaqGateRefused(f"camera trial gate: the DAQ quit-tail mask {self.where} was used under gate=camera")

    __array__ = __invert__ = __and__ = __rand__ = __or__ = __ror__ = __xor__ = __rxor__ = _refuse
    __len__ = __getitem__ = __iter__ = __bool__ = __eq__ = __ne__ = _refuse
    __hash__ = None


_CLASS_SIG: dict = {}
#: functions whose source defines the camera classes; their code enters `class_source_signature` (review H4)
CLASS_CODE_ROOTS = ("session_trials", "jaw_table", "_trial_times", "_pre_cue_contacts", "cue_classes",
                    "cue_classes_for", "row_classes", "camera_rows", "pool_row_classes", "camera_select",
                    "camera_engaged_period", "in_gate")


def class_source_signature(v36: Path | str | None = None, c1: Path | str | None = None) -> str:
    """sha1[:10] of everything the camera classes are built from (review H4, 11-03): the v36 per-session trials.csv /
    licks.csv / windows.npz, the cam1 event files, jaw_onsets.csv (names, sizes, mtimes) and the SOURCE of the class
    functions (`CLASS_CODE_ROOTS`, through `map_collect.closure`). Enters every camera-gated per-session cache kind
    (`run_mode` part "csig", cache only), so a regenerated jaw_onsets / cam1 / v36 table or a fixed class rule starts
    fresh entries instead of serving stale ones -- the pattern of `lp_licks.cache_tag`. Memoised per process."""
    import hashlib
    import sys
    v36, c1 = _paths(v36, c1)
    key = (str(v36), str(c1))
    if key not in _CLASS_SIG:
        h = hashlib.sha1()

        def stat(f: Path) -> None:
            try:
                st = f.stat()
                h.update(f"{f.name}:{st.st_size}:{st.st_mtime_ns};".encode())
            except OSError:
                h.update(f"{f.name}:-;".encode())
        if v36.exists():
            for sd in sorted(p for a in v36.iterdir() if a.is_dir() for p in a.iterdir() if p.is_dir()):
                h.update(f"[{sd.parent.name}/{sd.name}]".encode())
                for nm in ("trials.csv", "licks.csv", "windows.npz"):
                    stat(sd / nm)
        for f in (sorted(Path(c1).glob("*_events.csv")) if Path(c1).exists() else []):
            stat(f)
        stat(v36.parent / "imaging_encoding" / "qc" / "jaw_onsets.csv")
        from wfield_local import map_collect
        h.update(map_collect.code_digest(map_collect.closure(sys.modules[__name__], *CLASS_CODE_ROOTS)).encode())
        _CLASS_SIG[key] = h.hexdigest()[:10]
    return _CLASS_SIG[key]


def camera_engaged_period(label: str, cue_s) -> np.ndarray:
    """``bool[n_cues]``: True for cues up to and including the session's LAST camera-working trial -- the camera rule's
    counterpart of the DAQ terminal quit period, for the REST-BASELINE gates (`rest_engagement`, `rest_by_position`),
    which ask "is this rest inside the session's working period", not "did this trial respond". PROVISIONAL (rebuild
    11-03, to confirm with Priya): the DAQ rule is a terminal collapse; this keeps that shape (one terminal boundary,
    so a mid-session no-response trial does not cut the baseline). A session with no working trial (e.g. no video:
    all "unknown") has NO working period: every trial is outside it (no DAQ fallback, as `camera_select`)."""
    w = in_gate(cue_classes_for(label, cue_s), "working")
    out = np.zeros(len(w), bool)
    if w.any():
        out[: int(np.flatnonzero(w)[-1]) + 1] = True
    return out


def _selection_cfg() -> dict:
    try:
        from wfield_local import config
        return config.defaults().get("trial_selection") or {}
    except Exception:                                # noqa: BLE001 - config unavailable (bare unit tests)
        return {}


def gate_mode() -> str:
    """"daq" (default) or "camera" -- the run's trial-selection rule (see the block comment above)."""
    import os
    v = os.environ.get(GATE_ENV, "").strip().lower() or str(_selection_cfg().get("gate") or UNTAGGED_GATE).lower()
    if v not in GATE_MODES:
        raise ValueError(f"{GATE_ENV} / trial_selection.gate = {v!r}; expected one of {GATE_MODES}")
    return v


def gate_window() -> str:
    import os
    v = os.environ.get(WINDOW_ENV, "").strip() or str(_selection_cfg().get("window") or DEFAULT_WINDOW)
    if v not in WINDOWS:
        raise ValueError(f"{WINDOW_ENV} / trial_selection.window = {v!r}; expected one of {WINDOWS}")
    return v


def stopped_rule() -> str:
    """"run" (primary, default) or "any" (sensitivity): the camera stopped rule (see STOPPED_CLASSES)."""
    import os
    v = os.environ.get(STOPPED_ENV, "").strip().lower() or str(_selection_cfg().get("stopped") or "run").lower()
    if v not in STOPPED_RULES:
        raise ValueError(f"{STOPPED_ENV} / trial_selection.stopped = {v!r}; expected one of {STOPPED_RULES}")
    return v


def stop_run_min() -> int:
    """Minimum run of consecutive camera no_response trials for "stopped" = ``behavior.tail_min_misses`` (6)."""
    try:
        from wfield_local import config
        return int((config.defaults().get("behavior") or {}).get("tail_min_misses", 6))
    except Exception:                                # noqa: BLE001 - config unavailable (bare unit tests)
        return 6


def stopped_rule_name() -> str:
    """The rule name in selection tags: "stopRun<n>" or "stopAny"."""
    return f"stopRun{stop_run_min()}" if stopped_rule() == "run" else "stopAny"


def apply_stopped_rule(cls, rule: str | None = None, min_run: int | None = None) -> np.ndarray:
    """``cls`` (camera class per trial, in SESSION ORDER, the full train) with the stopped no_response trials relabelled
    STOPPED_CLASS. "run": runs of consecutive no_response counted with "unknown" trials skipped (they neither break nor
    extend a run); any other class breaks a run. "any": every no_response."""
    rule = rule or stopped_rule()
    out = np.asarray(cls, dtype=object).copy()
    nr = out == "no_response"
    if rule == "any":
        out[nr] = STOPPED_CLASS
        return out
    n = int(min_run if min_run is not None else stop_run_min())
    idx = np.flatnonzero(out != "unknown")             # unknown trials are skipped when counting
    seq = nr[idx]
    i = 0
    while i < len(seq):
        if not seq[i]:
            i += 1
            continue
        j = i
        while j < len(seq) and seq[j]:
            j += 1
        if j - i >= n:
            out[idx[i:j]] = STOPPED_CLASS
        i = j
    return out


def camera_gate_on() -> bool:
    return gate_mode() == "camera"


def run_selection(source: str | None = None) -> str:
    """The selection tag of THIS run, always defined (frozen specs carry it): ``sel-daq-quittail`` for the DAQ rule,
    else ``selection_tag(source or "camera", gate_window())``."""
    if gate_mode() == "daq":
        # the LP-inclusive gate source changes WHICH trials the DAQ rule selects, so it is part of the selection a
        # frozen spec records (review M7, 11-03). Not in `run_selection_tag`: gate-LP runs carry their own folder /
        # cache tag (`gate_source.cache_tag`, lp_licks "_gateLP").
        from wfield_local import gate_source
        return "sel-daq-quittail" + ("-gateLP" if gate_source.is_lp() else "")
    return selection_tag(source or "camera", gate_window(), stopped_rule_name())


def run_selection_tag() -> str | None:
    """None for the untagged DAQ rule, else the run's selection tag (output folders, cache kinds; `run_mode`)."""
    return None if gate_mode() == UNTAGGED_GATE else selection_tag("camera", gate_window(), stopped_rule_name())


def cue_classes(table: pd.DataFrame | None, cue_s, tol: float = CUE_MATCH_S) -> np.ndarray:
    """Class per DAQ cue (``cue_s``, DAQ seconds), matched to ``session_trials`` rows by nearest cue time within
    ``tol``; "unknown" where nothing matches (or ``table`` is None). Matched by TIME, never by trial id (ids are offset
    in some sessions; `lp_licks`). (Moved here from scripts/jaw_only_incomplete_figs.py, which now calls it.)"""
    cue_s = np.asarray(cue_s, float)
    out = np.full(len(cue_s), "unknown", dtype=object)
    if table is None or not len(table):
        return out
    t = table.dropna(subset=["cue_daq"]).sort_values("cue_daq")
    tv, cv = t.cue_daq.to_numpy(float), t.cls.to_numpy(object)
    if not len(tv):
        return out
    j = np.clip(np.searchsorted(tv, cue_s), 1, len(tv) - 1) if len(tv) > 1 else np.zeros(len(cue_s), int)
    if len(tv) > 1:
        j = np.where(np.abs(tv[j - 1] - cue_s) <= np.abs(tv[j] - cue_s), j - 1, j)
    ok = np.abs(tv[j] - cue_s) <= tol
    out[ok] = cv[j[ok]]
    return out


_CUE_CLS: dict = {}


def cue_classes_for(label: str, cue_s, window: str | None = None) -> np.ndarray:
    """Camera class per DAQ cue of one session (``cue_s`` in DAQ seconds); all "unknown" for a session without video
    outputs. ``cue_s`` must be the session FULL cue train in order (the stopped rule counts runs along it): no_response
    trials in a sustained stop come back as STOPPED_CLASS (`apply_stopped_rule`, Priya 11-03). Memoised per process.
    THE A5 HOOK: switch the class source to `responded(source=...)` here."""
    import hashlib
    window = window or gate_window()
    cs = np.ascontiguousarray(np.asarray(cue_s, float))
    rule = (stopped_rule(), stop_run_min())
    key = (label, window, rule, hashlib.sha1(cs.tobytes()).hexdigest())
    if key not in _CUE_CLS:
        if len(_CUE_CLS) > 64:
            _CUE_CLS.clear()
        _CUE_CLS[key] = apply_stopped_rule(cue_classes(session_trials(label, window=window), cs), *rule)
    return _CUE_CLS[key]


def daq_cue_seconds(label: str) -> np.ndarray:
    """The session's DAQ cue times (s) -- the train every imaging trial index refers to (`_trial_features`)."""
    from wfield_local import config
    from wfield_local.plot_spout_trial_averages import _load_daq_events
    s = next(x for x in config.load_sessions() if x["label"] == label)
    cue = _load_daq_events(s["h5"])
    return np.asarray(cue["cue_samples"], float) / float(cue["sample_rate_hz"])


def row_classes(label: str, cue_index) -> np.ndarray:
    """Camera class of imaging rows given their DAQ cue INDICES (``features_with_indices().indices``); "unknown" for an
    out-of-range index or a session without video."""
    idx = np.asarray(cue_index, np.int64)
    cls = cue_classes_for(label, daq_cue_seconds(label))
    out = np.full(idx.size, "unknown", dtype=object)
    ok = (idx >= 0) & (idx < cls.size)
    out[ok] = cls[idx[ok]]
    return out


def camera_rows(label: str, idx_e, idx_n, variant: str):
    """(keep engaged rows, keep no-lick rows) for ONE session's trial features under the camera rule -- the per-session
    map builders' counterpart of `camera_select` (`beta_maps.session_maps`, `position_reference_maps._working_xy`,
    `hemo_map_control`). ``idx_e`` / ``idx_n`` = DAQ cue indices of the engaged / no-lick rows.
      working       engaged + no-lick rows whose class is in GATES["working"]
      miss_working  no-lick rows in GATES["working"] (incomplete / jaw-only: responded but no contact)
      stopped       no-lick rows in STOPPED_CLASSES
      lick          engaged rows unchanged (contact-only)"""
    ce, cu = row_classes(label, idx_e), row_classes(label, idx_n)
    ne, nn = len(ce), len(cu)
    if variant == "working":
        return in_gate(ce, "working"), in_gate(cu, "working")
    if variant == "miss_working":
        return np.zeros(ne, bool), in_gate(cu, "miss_working")     # no contact class (review M6)
    if variant == "stopped":
        return np.zeros(ne, bool), np.isin(cu, list(STOPPED_CLASSES))
    return np.ones(ne, bool), np.zeros(nn, bool)


def pool_row_classes(feat, kept, n_e: int, n_u: int):
    """(class per engaged row, class per unengaged row) for a pool stacked session by session in ``kept`` order
    (`locanmf_frozen_decoder.pool_sessions` with ``features_with_indices``): ``feat.indices[label]`` = (engaged cue
    indices, no-lick cue indices), row for row -- the bookkeeping `position_coding_directions._gate_all` relies on.
    Raises if the counts do not line up (never guesses)."""
    ce, cu = [], []
    for lab in kept:
        got = getattr(feat, "indices", {}).get(lab)
        if got is None:
            raise RuntimeError(f"camera trial gate: no trial indices for {lab}")
        ie, inl = got
        ce.append(row_classes(lab, ie))
        cu.append(row_classes(lab, inl))
    ce = np.concatenate(ce) if ce else np.zeros(0, object)
    cu = np.concatenate(cu) if cu else np.zeros(0, object)
    if len(ce) != n_e or len(cu) != n_u:
        raise RuntimeError(f"camera trial gate: {len(ce)}/{len(cu)} classified rows vs a pool of {n_e}/{n_u}")
    return ce, cu


def camera_select(variant: str, cls_e, cls_u, sess_e, sess_u):
    """The camera rule's (engaged-row mask, unengaged-row mask) for one trial class -- `grant_kit._class_select`'s
    camera branch. ``lick``: the engaged rows (DAQ contact) UNCHANGED -- a contact-only selection, so PS93_0813 (no
    video) stays in it; ``working``: rows of either arm in the working gate; ``stopped``: rows of either arm in
    STOPPED_CLASSES. "unknown" rows (no video, unmatched cue) are in neither working nor stopped."""
    ce = np.asarray(cls_e, dtype=object)
    cu = np.asarray(cls_u, dtype=object)
    se = np.asarray(sess_e, bool)
    su = np.asarray(sess_u, bool)
    if variant == "stopped":
        return se & np.isin(ce, list(STOPPED_CLASSES)), su & np.isin(cu, list(STOPPED_CLASSES))
    if variant == "working":
        return se & in_gate(ce, "working"), su & in_gate(cu, "working")
    return se, np.zeros(len(su), bool)
