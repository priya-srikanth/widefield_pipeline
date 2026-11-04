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
  working      contact | incomplete | jaw_only   the 11-03 AFTERNOON "working" (any reaction to the cue); since
                                                 11-03 late a ROBUSTNESS definition only, see below (as on main)
  camera_lick  contact | incomplete              a lick was made (lick-aligned windows always need it)
  attempted    contact | incomplete              = camera_lick; the encoder page's "attempt" (10-21 rule)

PRODUCTION WORKING (Priya 11-03 LATE): contact | incomplete only -- jaw-only trials are their OWN class, reported like
no-response (docs/status/JAW_ONLY_CHECKS_2026-11-03.md). The SELECTION is main's setting `working_trials`
(``WIDEFIELD_WORKING_TRIALS`` = in_working, production: the camera_lick classes | in_working_jaw, robustness: the
jaw-inclusive "working" gate | quit_tail = the DAQ rule, i.e. ``WIDEFIELD_TRIAL_GATE=daq`` here). The static "working"
gate and `in_working` below stay jaw-INCLUSIVE, exactly as on main (main's `working_trials` pins its s1 snapshot to
them); every rebuild consumer of the camera working set reads `in_sel(cls, "working" | "miss_working")`, which applies
the selection.

END OF SESSION (Priya 11-03): primary "working" = the working gate AND at or before the session's LAST CAMERA LICK
(contact or incomplete): `in_working(...)` / `before_session_end(..., anchor="lick")`. It drops the late, smaller
jaw-only trials of the sated tail (docs/status/QUIT_TAIL_JAW_ONLY_2026-11-03.md: 1.0 % of working; 0.8 / 1.8 / 1.0 /
0.8 % pre / acute / subacute / chronic) and, by construction, no lick. Robustness anchors: "contact" (last spout
contact; cuts real acute far-spout incomplete attempts, 94), "close" (last camera lick at the reference spouts close_L /
close_center), "none" (no cutoff). The classes and GATES themselves are unchanged; the cutoff is a separate step.

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
GATES = {"working": frozenset({"contact", "incomplete", "jaw_only"}),   # jaw-INCLUSIVE, as on main; see `in_sel`
         "camera_lick": frozenset({"contact", "incomplete"}),
         "attempted": frozenset({"contact", "incomplete"}),
         # MISS WHILE WORKING on the NO-LICK arm (review M6, 11-03): responded but NO contact. A no-lick row (no DAQ
         # lick within max_rt) whose camera class is "contact" (a contact later in cue+70 ms .. stop) is in NEITHER
         # miss_working nor stopped -- to confirm with Priya (count them: `in_gate(cls, "working") & ~in_gate(cls,
         # "miss_working")`).
         # the jaw-inclusive (in_working_jaw) form; `in_sel(cls, "miss_working")` applies the selection
         "miss_working": frozenset({"incomplete", "jaw_only"})}
WINDOWS = ("cue_stop", "trial")
DEFAULT_WINDOW = "cue_stop"
JAW_UNJUDGED_POSITIONS = ("close_center", "close_mid")   # neither camera can judge the jaw at this spout (10-21)
CONTACT_FROM_CUE_S = 0.070                               # DAQ contacts count from cue + 70 ms (daq_contacts_ms)
V36 = "lp_r4_heatmap_20261013__clean_v36"
REFERENCE_SPOUTS = ("close_L", "close_center")           # barely affected by the stroke: anchor of the "close" cutoff
END_ANCHORS = ("lick", "contact", "close", "none")
DEFAULT_END = "lick"                                      # primary end-of-session cutoff (Priya 11-03)


def _flag(x: pd.Series) -> pd.Series:
    """Boolean column read safely: missing = False (used only for event flags, never for the jaw call)."""
    return x.astype("boolean").fillna(False).astype(bool)


def _paths(v36: Path | str | None, c1: Path | str | None) -> tuple[Path, Path]:
    from wfield_local.paths import PathResolver
    rv = PathResolver()
    v36 = Path(v36) if v36 else Path(rv.root("orofacial")) / V36 / "sessions"
    c1 = Path(c1) if c1 else Path(rv.root("dlc")) / "session_poses" / "cam1_licks"
    return v36, c1


def _jaw_csv(v36: Path) -> Path:
    """`jaw_onsets.csv` of the run whose sessions folder is ``v36``: through `orofacial_outputs.encoding_root` (the
    rebuild root's seeded copy under one, `rebuild_root`)."""
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    v36 = Path(v36)
    if v36.parent == Path(PathResolver().input_root("orofacial")) / v36.parent.name:     # a run of the real root
        return oo.encoding_root(v36.parent.name) / "qc" / "jaw_onsets.csv"
    return v36.parent / "imaging_encoding" / "qc" / "jaw_onsets.csv"                      # an explicit (test) tree


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


def _daq_contact_onsets(animal: str, date: str) -> np.ndarray:
    """DAQ spout-contact (lick) onsets of one session, s on the DAQ clock (`dlc_frames.lick_onsets`)."""
    from wfield_local import dlc_frames
    from wfield_local.paths import PathResolver
    rv = PathResolver()
    sid = sorted((Path(rv.root("behavior_out")) / "sessions" / animal / date).glob("*_trials.csv"))[-1].name[:-11]
    return np.asarray(dlc_frames.lick_onsets(animal, date, sid, rv), float)


def trial_contact_times(animal: str, date: str, times: pd.DataFrame) -> pd.Series:
    """First DAQ contact per trial in cue + 70 ms .. stop, in ms from the cue (NaN = none): the contact the cue_stop
    class counts, with its TIME (decision 1's late / missed-contact split)."""
    licks = _daq_contact_onsets(animal, date)
    out = {}
    for t, c, s in zip(times.trial_id, times.cue_daq, times.stop_ms):
        lo, hi = c + CONTACT_FROM_CUE_S, c + s / 1000.0
        k = licks[(licks >= lo) & (licks <= hi)]
        out[int(t)] = float((k[0] - c) * 1000.0) if len(k) else np.nan
    return pd.Series(out)


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
    jaw_csv = Path(jaw_csv) if jaw_csv else _jaw_csv(v36)
    key = f"{animal}_{date}"
    if not jaw_csv.exists():
        from wfield_local import rebuild_root
        rebuild_root.require_seed(jaw_csv)          # a rebuild never silently drops the jaw calls (unseeded root)
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


#: the rebuild's names for the selection-dependent sets (`in_sel`)
SEL_GATES = ("working", "miss_working")


#: WHICH ROWS EACH FAMILY SCORES under the camera working selection (main 11-03 design, pending Priya; review 7A):
#: stamped on every figure of a camera-gated run (`lp_licks.stamp`) and written into section D / G result sidecars
DESIGN_NOTE = ("frozen decoders trained + centred on camera WORKING rows (contact + incomplete); scored sets: 5b / 5c = "
               "working rows on both sides; section D (pooled frozen) / G = engaged / all trials (main's 11-03 design); "
               "pattern families: post = working, pre reference = LICK trials")


def working_selection() -> str:
    """main's ``WIDEFIELD_WORKING_TRIALS`` (`working_trials.selection`): in_working (production, default) |
    in_working_jaw | quit_tail. Under the camera gate quit_tail is a contradiction and RAISES (the quit-tail rule is
    ``WIDEFIELD_TRIAL_GATE=daq`` in the rebuild)."""
    from wfield_local import working_trials as wt
    sel = wt.selection()
    if sel == wt.QUIT_TAIL and gate_mode() == "camera":
        raise DaqGateRefused(f"{wt.WORKING_TRIALS_ENV}=quit_tail under {GATE_ENV}=camera: the quit-tail rule is the DAQ "
                             f"gate -- set {GATE_ENV}=daq")
    return sel


def sel_classes(which: str = "working") -> frozenset:
    """The camera classes of the run's WORKING set (``which`` "working") or of its no-lick-arm MISS-while-working set
    (``which`` "miss_working" = working minus contact), under `working_selection` (main's `working_trials`
    WORKING_CLASSES): in_working = contact | incomplete; in_working_jaw = + jaw_only. quit_tail (DAQ gate) reads as
    in_working, as main's ``row_masks`` does for quit_tail callers asking for camera masks."""
    from wfield_local import working_trials as wt
    if which not in SEL_GATES:
        raise ValueError(f"in_sel: {which!r} is not one of {SEL_GATES}")
    sel = working_selection()
    w = wt.working_classes(sel if wt.is_camera(sel) else wt.IN_WORKING)
    return w if which == "working" else w - {"contact"}


def in_sel(cls, which: str = "working") -> np.ndarray:
    """Boolean per trial: class in the run's selection-dependent set (`sel_classes`)."""
    return np.isin(np.asarray(cls, dtype=object), list(sel_classes(which)))


def sel_cls_part() -> str | None:
    """main's spec-tag class part: "contact-incomplete" under in_working (production), None under in_working_jaw
    (the afternoon tag, unchanged)."""
    from wfield_local import working_trials as wt
    return "contact-incomplete" if working_selection() == wt.IN_WORKING else None


def before_session_end(cls, order, *, session=None, position=None, anchor: str = DEFAULT_END) -> np.ndarray:
    """Boolean per trial: at or before its session's LAST anchor trial (see END OF SESSION in the module docstring).

    cls / order (trial order within the session: trial_id or DAQ cue index) / session (labels; None = one session) /
    position (needed for anchor "close") are equal-length sequences. A session with no anchor trial keeps nothing
    (every trial is after the "last" one); anchor "none" keeps everything."""
    if anchor not in END_ANCHORS:
        raise ValueError(f"anchor must be one of {END_ANCHORS}, got {anchor!r}")
    cls = np.asarray(cls, dtype=object)
    if anchor == "none":
        return np.ones(len(cls), bool)
    order = np.asarray(order, float)
    sess = np.zeros(len(cls), int) if session is None else np.asarray(session, dtype=object)
    lick = in_gate(cls, "camera_lick")
    if anchor == "contact":
        mark = cls == "contact"
    elif anchor == "lick":
        mark = lick
    else:
        if position is None:
            raise ValueError('anchor "close" needs position')
        mark = lick & np.isin(np.asarray(position, dtype=object), REFERENCE_SPOUTS)
    last = pd.Series(order[mark]).groupby(pd.Series(sess[mark])).max()
    return order <= pd.Series(sess).map(last).fillna(-np.inf).to_numpy(float)


def in_working(cls, order, *, session=None, position=None, end: str = DEFAULT_END) -> np.ndarray:
    """The working gate (incl. jaw_only) AND before the session's end (`before_session_end`). Since 11-03 late the
    PRODUCTION working is this ``& in_gate(cls, "camera_lick")`` (jaw-only out; `working_trials.IN_WORKING`; the
    rebuild reads it through `in_sel`)."""
    return in_gate(cls, "working") & before_session_end(cls, order, session=session, position=position, anchor=end)


def selection_tag(source: str = "camera", window: str = DEFAULT_WINDOW, end: str | None = None,
                  stopped: str | None = None, cls: str | None = None) -> str:
    """Short tag naming the trial-selection rule; put it in every cache key / frozen-model spec built on a gate, so
    runs with different sources or windows never share a cache entry (agreed with the cohort-1 rebuild, 11-03).
    ``end`` = the end-of-session anchor when the selection uses one (e.g. "lick" -> "-endlick"); ``stopped`` = the
    camera stopped-rule name (`stopped_rule_name`). None leaves the tag as before, so existing keys do not move."""
    # ``cls`` = main's class part (`working_trials.tag`): "-cls-contact-incomplete" for the production in_working set
    # (Priya 11-03 late, jaw_only out), so no store / cache entry built on the jaw-inclusive set can be served; the
    # in_working_jaw robustness arm has none (its tag = the 11-03 afternoon one). Placed after the end part, as on main.
    return (f"sel-{source}-{window}" + (f"-end{end}" if end else "") + (f"-cls-{cls}" if cls else "")
            + (f"-{stopped}" if stopped else ""))


# ------------------------------------------------------------------------------------------ the run's trial gate
# (cohort-1 rebuild 2026-11-03). WHICH RULE SELECTS "working" / "stopped" in the imaging families
# (`grant_kit._class_select` and the gates listed in `docs/status/REBUILD_CODE_20261103.md` item f):
#   daq     the DAQ quit-tail rule: working = engaged + unengaged outside the terminal quit period, stopped = the quit
#           period (`precue_engagement_states.engagement_gate`). UNTAGGED: every output made before 11-03.
#   camera  THIS module's GATES: working = contact | incomplete, stopped = a sustained stop (`classify_session`,
#           >= 6 no_response); "unknown" and
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
#: the class a WORKING-gate trial gets in `cue_classes_for` when it lies after the session end (`before_session_end`,
#: anchor `end_anchor()`, Priya 11-03): in neither working nor stopped -- so every `in_sel(cls, "working")` consumer IS
#: `in_working(...)` on the session cue train
AFTER_END_CLASS = "after_end"
END_ENV = "WIDEFIELD_TRIAL_END"
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


def reraise_under_camera(ex: BaseException, where: str) -> None:
    """Review N9 (11-03): call in an ``except Exception`` that would DROP a session / animal and continue. Under the
    camera gate the error is re-raised (a camera-class data error must fail the render, not thin it silently); under
    daq nothing changes (the caller logs and continues, as before)."""
    if camera_gate_on():
        raise RuntimeError(f"camera trial gate: {where} failed ({type(ex).__name__}: {str(ex)[:200]}); a camera-gated "
                           "render does not drop data silently") from ex


def nolick_selection() -> str:
    """The selection a NO-LICK analysis records (Priya FINAL 11-03): under the camera gate its lick / contact
    definition is trial_classes' cue_stop window (cue -> trial stop, contacts from cue + 70 ms, the contact table), so
    the tag names cue_stop whatever `trial_selection.window` says; under daq the run's selection (unchanged)."""
    if not camera_gate_on():
        return run_selection()
    return selection_tag("camera", "cue_stop", end=end_anchor(), stopped=_stop_rest_name(), cls=sel_cls_part())


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
                    "camera_engaged_period", "in_gate", "before_session_end", "in_working", "classify_session", "_matched_cues",
                    "_effective_coverage")


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
        stat(_jaw_csv(v36))
        from wfield_local import map_collect
        h.update(map_collect.code_digest(map_collect.closure(sys.modules[__name__], *CLASS_CODE_ROOTS)).encode())
        _CLASS_SIG[key] = h.hexdigest()[:10]
    return _CLASS_SIG[key]


def camera_engaged_period(label: str, cue_s) -> np.ndarray:
    """``bool[n_cues]``: the session's camera WORKING PERIOD for the REST-BASELINE gates (`rest_engagement`,
    `rest_by_position`), which ask "is this rest inside the working period", not "did this trial respond": the cues
    from the session's FIRST camera lick to its LAST camera lick, inclusive (one start and one end boundary, so a
    mid-session no-response trial never cuts the baseline).

    * END (Priya 11-03, decision 2; T8 third review): the last camera lick of the CLASSIFIED train (`cue_classes_for`),
      always -- also with an uncovered video tail (rest after the last seen lick may already be sated). Under a
      robustness end anchor ("contact" / "close") the licks after that anchor are AFTER_END_CLASS in the classified
      train, so the period ends AT THE ANCHOR.
    * START: the first camera lick (``trial_selection.rest_start: first_lick``, in force; an open question for Priya --
      "session" starts it at the first cue).
    * A session with no camera lick (e.g. no video: all "unknown") has NO working period: every cue is outside it (no
      DAQ fallback, as `camera_select`)."""
    c = cue_classes_for(label, cue_s)
    out = _lick_span(c)
    if rest_start() == "session" and out.any():
        out[: int(np.flatnonzero(out)[-1]) + 1] = True
    return out


def _lick_span(cls) -> np.ndarray:
    """``bool[n]``: True from the first to the last camera_lick-class trial of ``cls`` (inclusive); all False without
    one."""
    c = np.asarray(cls, dtype=object)
    out = np.zeros(len(c), bool)
    lick = np.flatnonzero(in_gate(c, "camera_lick"))
    if len(lick):
        out[int(lick[0]): int(lick[-1]) + 1] = True
    return out


def _selection_cfg() -> dict:
    # NO SILENT "daq" (review, second pass 11-03): a config that exists but fails to load RAISES -- once
    # `trial_selection.gate: camera` is the yaml default, swallowing the error would silently run the DAQ rule.
    # Only a missing config module (a bare unit-test import) falls back to the documented defaults.
    try:
        from wfield_local import config
    except ImportError:
        return {}
    return config.defaults().get("trial_selection") or {}


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


def end_anchor() -> str:
    """The run's end-of-session anchor for camera "working": "lick" (primary, Priya 11-03) | "contact" | "close" |
    "none" (``trial_selection.end`` / WIDEFIELD_TRIAL_END)."""
    import os
    v = os.environ.get(END_ENV, "").strip().lower() or str(_selection_cfg().get("end") or DEFAULT_END).lower()
    if v not in END_ANCHORS:
        raise ValueError(f"{END_ENV} / trial_selection.end = {v!r}; expected one of {END_ANCHORS}")
    return v


def classify_session(cls, matched=None, position=None, *, rule: str | None = None, min_run: int | None = None,
                     end: str | None = None, cue_s=None, min_start_s: float | None = None,
                     jaw_bridges_all: bool | None = None) -> np.ndarray:
    """THE camera classes of one session, in cue order, after the session-level rules (Priya 11-03). ``cls`` = the raw
    per-cue class (`cue_classes`), ``matched`` = the cue was matched to a video trial (None = all matched).

    VIDEO COVERAGE (N6): trials outside the video stay "unknown". Both session-level rules use only the covered span:
      * END AT THE LAST CAMERA LICK (anchor ``end``): working-gate trials after it -> AFTER_END_CLASS. With an
        UNCOVERED TAIL (video ends before the last cue) the real last lick is unknown, so NO end cutoff is applied
        (a cutoff at the last covered lick would end working early).
      * SUSTAINED STOP (``rule`` "run"): a run with >= ``min_run`` no_response trials -> its no_response trials are
        STOPPED_CLASS. Runs are counted per COVERED SEGMENT (an uncovered gap ends a run; it never stitches two runs);
        in-video "unknown" trials neither break nor extend a run; a contact / incomplete trial (licking resumed) ENDS
        the run; a jaw_only trial anywhere BRIDGES it (Priya 11-03 late, jaw_only out of working: the after-last-lick
        T3 rule extended to the whole session): it neither breaks the run nor counts toward ``min_run`` and is never
        stopped. ``jaw_bridges_all`` (None = from the working selection): True under in_working (production); False
        under in_working_jaw, where jaw_only is working and the 11-03 afternoon rule holds (a jaw_only trial at or
        before the end anchor ENDS the run, one after it bridges). Rule "any": every no_response is stopped.
      * JAW_ONLY is its own class, reported like no_response: inside the working period it stays "jaw_only"; after the
        end anchor it becomes AFTER_END_CLASS.
      * TIME RULE (Priya 11-03): a stop counts only if its run STARTS >= ``min_start_s`` (`stop_min_start_s()`, 300 s)
        after the session's first cue (``cue_s``, the session cue times in s; None = no time rule). An earlier
        qualifying run -> EARLY_RUN_CLASS (in neither working nor stopped); under rule "any", every no_response
        before that time is EARLY_RUN_CLASS. Checked before adoption: 0 of the 159 current stopped runs started that
        early.
      * ISOLATED LAPSES (decision 3): a no_response trial not in a sustained stop stays "no_response" only INSIDE the
        working period (at or before the end anchor); after it -> AFTER_END_CLASS. So "no_response" = an isolated lapse.
    """
    rule = rule or stopped_rule()
    n_min = int(min_run if min_run is not None else stop_run_min())
    end = end or end_anchor()
    out = np.asarray(cls, dtype=object).copy()
    n = len(out)
    m = _effective_coverage(np.ones(n, bool) if matched is None else np.asarray(matched, bool))
    tail_uncovered = bool(m.any()) and int(np.flatnonzero(m)[-1]) < n - 1
    if end == "none" or tail_uncovered or not m.any():
        after = np.zeros(n, bool)
    else:
        after = ~before_session_end(out, np.arange(n), position=position, anchor=end)
    # JAW_ONLY BRIDGES a no_response run anywhere in the session (T3 extended; Priya 11-03 late: jaw_only is not
    # working): it does NOT count toward the 6 and is NOT stopped
    if jaw_bridges_all is None:
        jaw_bridges_all = "jaw_only" not in sel_classes("working")
    jaw_late = (out == "jaw_only") if jaw_bridges_all else ((out == "jaw_only") & after)
    nr = out == "no_response"
    # TIME RULE (Priya 11-03): runs starting before first cue + min_start_s are not stops (in neither class)
    if cue_s is not None and n:
        t = np.asarray(cue_s, float)
        if len(t) != n:
            raise ValueError(f"cue_s has {len(t)} cues for {n} classes")
        t_min = float(min_start_s if min_start_s is not None else stop_min_start_s())
        too_early = (t - np.nanmin(t)) < t_min
    else:
        too_early = np.zeros(n, bool)
    stop = np.zeros(n, bool)
    early = np.zeros(n, bool)
    if rule == "any":
        stop = nr & ~too_early
        early = nr & too_early
    else:
        seg_id = np.cumsum(np.r_[True, np.diff(m.astype(int)) != 0])   # covered / uncovered segments
        for sid in np.unique(seg_id[m]):
            idx = np.flatnonzero((seg_id == sid) & m & (out != "unknown"))
            seq = (nr | jaw_late)[idx]
            i = 0
            while i < len(seq):
                if not seq[i]:
                    i += 1
                    continue
                j = i
                while j < len(seq) and seq[j]:
                    j += 1
                run = idx[i:j]
                if int(nr[run].sum()) >= n_min:          # >= 6 NO_RESPONSE trials; bridging jaw_only not counted
                    first_nr = run[nr[run]]
                    (early if too_early[first_nr[0]] else stop)[first_nr] = True   # the run's START decides
                i = j
    out[early] = EARLY_RUN_CLASS
    out[stop] = STOPPED_CLASS
    late = after & ~stop & (in_gate(out, "working") | np.isin(out, ["no_response", "jaw_only"]))
    out[late] = AFTER_END_CLASS
    return out


#: T7 (third review): an unmatched stretch counts as UNCOVERED only when it is at least this many consecutive cues;
#: a shorter one (an isolated cue missing the 20 ms match, a trailing DAQ cue after the video stopped) stays "unknown"
#: inside the covered span -- skipped in stop runs, never a split, never a reason to drop the end cutoff
MIN_UNCOVERED_RUN = 5
#: a qualifying no_response run that STARTS within `stop_min_start_s()` of the session's first cue (Priya 11-03): in
#: neither working nor stopped
EARLY_RUN_CLASS = "early_run"


def _effective_coverage(matched) -> np.ndarray:
    """``matched`` with unmatched stretches shorter than MIN_UNCOVERED_RUN filled in (treated as covered)."""
    m = np.asarray(matched, bool).copy()
    if not m.any():
        return m
    i, n = 0, len(m)
    while i < n:
        if m[i]:
            i += 1
            continue
        j = i
        while j < n and not m[j]:
            j += 1
        if j - i < MIN_UNCOVERED_RUN:
            m[i:j] = True
        i = j
    return m


def session_coverage(label: str, cue_s, window: str | None = None) -> dict:
    """Video coverage of one session's cue train (N6): {n_cues, n_matched, frac, partial, tail_uncovered}."""
    m = _matched_cues(label, cue_s, window)
    e = _effective_coverage(m)                   # T7: stretches < MIN_UNCOVERED_RUN cues are not uncovered
    n = len(m)
    return {"label": label, "n_cues": n, "n_matched": int(m.sum()), "frac": float(m.mean()) if n else float("nan"),
            "partial": bool(n and not e.all()),
            "tail_uncovered": bool(e.any() and int(np.flatnonzero(e)[-1]) < n - 1)}


def _matched_cues(label: str, cue_s, window: str | None = None) -> np.ndarray:
    T = session_trials(label, window=window or gate_window())
    if T is None:
        return np.zeros(len(np.asarray(cue_s)), bool)
    return cue_classes(T.assign(cls="matched"), np.asarray(cue_s, float)) == "matched"


def coverage_note(path=None) -> str:
    """Caption text naming the partially video-covered sessions (N6), from the coverage table
    (`scripts/camera_coverage_tables.py`); says so when the table is missing. Used by `lp_licks.stamp` under camera."""
    from wfield_local import lp_licks
    p = Path(path) if path else lp_licks._enc_dir() / "qc" / "camera_coverage.csv"
    if not p.exists():
        from wfield_local import rebuild_root
        rebuild_root.require_seed(p)
        return "camera gate: video-coverage table missing (run scripts/camera_coverage_tables.py)"
    C = pd.read_csv(p)
    C = C[C.partial.astype(bool)]
    if not len(C):
        return "camera gate: every session fully video-covered"
    return ("camera gate: partial video coverage (covered trials only; end / stop rules within the covered span): "
            + ", ".join(f"{r.label} {100 * r.frac:.1f}%" for r in C.itertuples()))


def contact_nolick_counts(cls, nolick, animal, epoch) -> "pd.DataFrame":
    """Decision 1 (Priya 11-03): no-lick (DAQ) trials whose camera class is "contact" are in NEITHER miss-while-working
    nor stopped; this COUNTS them per animal x epoch (columns n_nolick, n_contact_nolick, share) so they can be checked
    (undetected DAQ contacts, or grooming / something else). Equal-length per-trial sequences."""
    D = pd.DataFrame({"animal": np.asarray(animal, object), "epoch": np.asarray(epoch, object),
                      "nolick": np.asarray(nolick, bool), "cls": np.asarray(cls, object)})
    D = D[D.nolick]
    g = D.groupby(["animal", "epoch"])
    out = pd.DataFrame({"n_nolick": g.size(), "n_contact_nolick": g.cls.apply(lambda c: int((c == "contact").sum()))})
    out["share"] = out.n_contact_nolick / out.n_nolick
    return out.reset_index()


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
    except ImportError:
        return 6
    return int((config.defaults().get("behavior") or {}).get("tail_min_misses", 6))   # a broken config RAISES


def stop_min_start_s() -> float:
    """A stop counts only if its run starts this long after the session's first cue (``trial_selection.
    stop_min_start_s``, 300 s = 5 min; Priya 11-03)."""
    import os
    v = os.environ.get("WIDEFIELD_STOP_MIN_START_S", "").strip()
    return float(v) if v else float(_selection_cfg().get("stop_min_start_s", 300.0))


def rest_start() -> str:
    """Where the camera rest working period STARTS (``trial_selection.rest_start``): "first_lick" (the session's first
    camera lick; the setting in force, an OPEN QUESTION for Priya) or "session" (the first cue)."""
    import os
    v = (os.environ.get("WIDEFIELD_REST_START", "").strip() or str(_selection_cfg().get("rest_start", "first_lick"))).lower()
    if v not in ("first_lick", "session"):
        raise ValueError(f"trial_selection.rest_start = {v!r}; expected first_lick or session")
    return v


def _stop_rest_name() -> str:
    """The stopped-rule name plus the rest-start setting when it is not the default (fifth review (b)): what every
    camera selection tag carries -- so cache keys, frozen specs and output folders separate the two rest starts."""
    return stopped_rule_name() + ("-restsession" if rest_start() == "session" else "")


def stopped_rule_name() -> str:
    """The rule name in selection tags: "stopRun<n>" or "stopAny" (+ "t<s>" when the stop start time is not 300 s)."""
    base = f"stopRun{stop_run_min()}" if stopped_rule() == "run" else "stopAny"
    t = stop_min_start_s()
    return base if t == 300.0 else f"{base}t{t:g}"


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
    return selection_tag(source or "camera", gate_window(), end=end_anchor(), stopped=_stop_rest_name(),
                         cls=sel_cls_part())


def run_selection_tag() -> str | None:
    """None for the untagged DAQ rule, else the run's selection tag (output folders, cache kinds; `run_mode`)."""
    return None if gate_mode() == UNTAGGED_GATE else selection_tag("camera", gate_window(), end=end_anchor(),
                                                                   stopped=_stop_rest_name(), cls=sel_cls_part())


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
    trials in a sustained stop come back as STOPPED_CLASS (`classify_session`, Priya 11-03). Memoised per process.
    THE A5 HOOK: switch the class source to `responded(source=...)` here."""
    import hashlib
    window = window or gate_window()
    cs = np.ascontiguousarray(np.asarray(cue_s, float))
    rule = (stopped_rule(), stop_run_min(), stop_min_start_s(), rest_start())
    end = end_anchor()
    jaw_all = "jaw_only" not in sel_classes("working")        # the working selection (main's WIDEFIELD_WORKING_TRIALS)
    key = (label, window, rule, end, jaw_all, hashlib.sha1(cs.tobytes()).hexdigest())
    if key not in _CUE_CLS:
        if len(_CUE_CLS) > 64:
            _CUE_CLS.clear()
        T = session_trials(label, window=window)
        # THE session-level rules (end at the last camera lick, sustained stop, isolated lapses; video coverage span):
        # `classify_session`. The "close" anchor needs each cue position (matched the same way as the class)
        pos = cue_classes(T.assign(cls=T["position"]) if T is not None else None, cs) if end == "close" else None
        matched = (cue_classes(T.assign(cls="matched"), cs) == "matched") if T is not None else np.zeros(len(cs), bool)
        _CUE_CLS[key] = classify_session(cue_classes(T, cs), matched, pos, rule=rule[0], min_run=rule[1], end=end,
                                         cue_s=cs, jaw_bridges_all=jaw_all)
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
      working       engaged + no-lick rows whose class is in the run's working set (`in_sel`; production contact |
                    incomplete)
      miss_working  no-lick rows in the run's miss-while-working set (`in_sel`; production incomplete: licked, no
                    contact)
      stopped       no-lick rows in STOPPED_CLASSES
      lick          engaged rows unchanged (contact-only)
    Any other variant RAISES (second pass 11-03: it used to fall through to "lick")."""
    if variant not in ("working", "miss_working", "stopped", "lick"):
        raise ValueError(f"camera_rows: unknown trial class {variant!r}")
    ce, cu = row_classes(label, idx_e), row_classes(label, idx_n)
    ne, nn = len(ce), len(cu)
    if variant == "working":
        return in_sel(ce, "working"), in_sel(cu, "working")
    if variant == "miss_working":
        return np.zeros(ne, bool), in_sel(cu, "miss_working")     # no contact class (review M6)
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
        # ONE definition of stopped (second pass 11-03): no-lick rows only, as `camera_rows("stopped")` -- an engaged
        # (DAQ-lick) row is never stopped, whatever its camera class
        return np.zeros(len(se), bool), su & np.isin(cu, list(STOPPED_CLASSES))
    if variant == "working":
        return se & in_sel(ce, "working"), su & in_sel(cu, "working")
    if variant == "lick":
        return se, np.zeros(len(su), bool)
    raise ValueError(f"camera_select: unknown trial class {variant!r} (lick / working / stopped)")
