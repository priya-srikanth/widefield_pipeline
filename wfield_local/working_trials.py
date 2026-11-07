"""The imaging decoders' WORKING trials (Priya 11-03), mapped onto pooled feature rows.

PRODUCTION SINCE 11-03 LATE (Priya): working = ``in_working & class in {contact, incomplete}`` -- the camera LICK
trials (contact or incomplete lick), cue -> trial stop, cut at the session's last camera lick. JAW-ONLY TRIALS ARE NOT
WORKING: they are reported as their own class, like no-response (docs/status/JAW_ONLY_CHECKS_2026-11-03.md: jaw_only
reads like no_response in PS94 / PS95 at cue and precue; adding them to training does not move the pre reference;
centering a post-stroke session on rows that include them drives PS95 close ipsi acute .975 -> .642). Every contact /
incomplete trial is at or before the last camera lick by construction, so the end cut only ever removed jaw_only /
no_response trials; it stays in the definition so the rule reads as written. ``WORKING_CLASSES`` names the sets.

The 11-03 AFTERNOON definition (in_working INCLUDING jaw_only) is the robustness arm
``WIDEFIELD_WORKING_TRIALS=in_working_jaw``; its frozen-model specs (tag without the class suffix) are the 29 models
published 11-03 afternoon, so it reproduces them exactly.

SUPPLEMENTAL SUCCESSFUL-LICK ARM (Priya 11-07): ``WIDEFIELD_WORKING_TRIALS=in_working_contact`` = in_working AND class ==
contact only (successful spout-contact licks), same camera working period and end cut (END = "lick"), tag suffix
"-cls-contact" so its frozen specs, cache keys and output folders never meet production's. Section G run under it is a
"successful licks only" supplement. Far-contra post-stroke trials are mostly INCOMPLETE licks, so this arm drops
positions: the lick-only arm's chance 1/n varies with the retained positions -- compare arms via the retained
above-chance fraction, not raw accuracy. Its no-lick "miss while working" set stays production's (incomplete; see
`trial_classes.sel_classes`), and the stop rule is production's (jaw_only counts toward a stop run).

History of the paragraph below: written for the afternoon definition.

Priya 11-03: the epoch families' "working" selection is the shared camera trial classification, not the DAQ lick +
miss-while-working set of the `flag_engagement` quit-tail gate. Working = jaw moved OR any camera lick (contact or
incomplete), window cue -> trial stop, cut at the session's LAST CAMERA LICK (`trial_classes.in_working`, end="lick"):
licks + attempted licks + jaw movements. AT close_center (close mid) the jaw cannot be judged on either camera, so
working there is licks only (contact or incomplete; `trial_classes.JAW_UNJUDGED_POSITIONS`). It applies on BOTH sides:
the pre-stroke training set and the pre-stroke panel are pre-stroke working trials too. Lick-aligned windows keep the
lick gate (a trial with no lick has nothing to align to). The quit-tail selection stays available as a robustness arm
(``WIDEFIELD_WORKING_TRIALS=quit_tail``).

THE CLASSES ARE READ, NEVER RE-DERIVED: the per-DAQ-cue A4 snapshot `scripts/s1_trial_class_snapshot.py` writes
(`<run>/imaging_encoding/qc/s1_trial_classes_cue_stop.csv.gz`, the table imaging section 1 indexes), with the
end-of-session cutoff recomputed here by 5b's `trial_classes.in_working` and checked equal to the snapshot's own
``working_pl_lick`` column. Pooled rows are matched to DAQ cues through the feature builder's ``.indices`` (engaged and
no-lick cue indices per session), exactly as `scripts/s1_refit_locanmf_table.py` does, so 5b's section-1 trial sets
are reproduced row for row (pinned in tests).

Sessions with NO camera classes (no behavior video: PS93_0813; the June sessions, outside the August-only baseline)
have no working trials: they drop out of every working training set and pre band.
"""
from __future__ import annotations

import functools
import os
from pathlib import Path

import numpy as np

WORKING_TRIALS_ENV = "WIDEFIELD_WORKING_TRIALS"
IN_WORKING = "in_working"                 # PRODUCTION (11-03 late): in_working & class in {contact, incomplete}
IN_WORKING_JAW = "in_working_jaw"         # robustness: the 11-03 afternoon in_working, jaw_only included
IN_WORKING_CONTACT = "in_working_contact"  # supplement (Priya 11-07): in_working & class == contact only
QUIT_TAIL = "quit_tail"                   # robustness: DAQ lick + miss-while-working (flag_engagement quit tail)
CAMERA_SELECTIONS = (IN_WORKING, IN_WORKING_JAW, IN_WORKING_CONTACT)
#: The camera classes a working trial may have, per camera selection (both also cut at the last camera lick). Production
#: = trial_classes.GATES["camera_lick"]; robustness = trial_classes.GATES["working"] (pinned in tests).
WORKING_CLASSES = {IN_WORKING: frozenset({"contact", "incomplete"}),
                   IN_WORKING_JAW: frozenset({"contact", "incomplete", "jaw_only"}),
                   IN_WORKING_CONTACT: frozenset({"contact"})}
#: the tag's class part per camera selection (None = the 11-03 afternoon tag, no class part)
CLS_PART = {IN_WORKING: "contact-incomplete", IN_WORKING_JAW: None, IN_WORKING_CONTACT: "contact"}
END = "lick"
SNAPSHOT_ENV = "WFIELD_S1_CLASSES"        # the same override 5b's s1 script reads


def selection() -> str:
    """``"in_working"`` (production since 11-03 late: contact + incomplete licks, cut at the last camera lick),
    ``"in_working_jaw"`` (the 11-03 afternoon definition, jaw_only included), ``"in_working_contact"`` (supplement,
    Priya 11-07: contact trials only) or ``"quit_tail"`` (the previous lick + miss-while-working set)."""
    v = os.environ.get(WORKING_TRIALS_ENV, IN_WORKING).strip().lower()
    if v in (IN_WORKING, "camera", "a4", "licks", "in_working_licks"):
        return IN_WORKING
    if v in (IN_WORKING_JAW, "in_working_incl_jaw", "with_jaw"):
        return IN_WORKING_JAW
    if v in (IN_WORKING_CONTACT, "contact", "contact_only", "in_working_contact_only"):
        return IN_WORKING_CONTACT
    if v in (QUIT_TAIL, "flag_engagement", "old"):
        return QUIT_TAIL
    raise ValueError(f"{WORKING_TRIALS_ENV}={v!r}: expected in_working, in_working_jaw, "
                     "in_working_contact or quit_tail")


def is_camera(sel: str | None = None) -> bool:
    """True when the working selection is one of the CAMERA selections (row masks from the A4 snapshot)."""
    return (selection() if sel is None else sel) in CAMERA_SELECTIONS


def working_classes(sel: str | None = None) -> frozenset:
    sel = selection() if sel is None else sel
    if sel not in WORKING_CLASSES:
        raise ValueError(f"{sel!r} is not a camera working selection")
    return WORKING_CLASSES[sel]


def tag(sel: str | None = None) -> str:
    """The selection tag that goes into every frozen-model spec trained on working trials (`trial_classes`).
    in_working_jaw keeps the 11-03 afternoon tag (so its specs are the published afternoon models); the production
    in_working adds the class set, so its spec ids differ from every afternoon model."""
    from wfield_local import trial_classes as tc
    sel = selection() if sel is None else sel
    base = tc.selection_tag("camera", tc.DEFAULT_WINDOW, end=END)
    if sel == IN_WORKING_JAW:
        return base
    if sel == IN_WORKING:
        return base + "-cls-contact-incomplete"
    if sel == IN_WORKING_CONTACT:
        return base + "-cls-contact"
    raise ValueError(f"{sel!r} is not a camera working selection")


def snapshot_path() -> Path:
    if os.environ.get(SNAPSHOT_ENV):
        return Path(os.environ[SNAPSHOT_ENV])
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    return oo.root(PathResolver()) / oc.RUN / "imaging_encoding" / "qc" / "s1_trial_classes_cue_stop.csv.gz"


@functools.lru_cache(maxsize=4)
def _table(path: str, sel: str = IN_WORKING):
    """{label: {cue_idx: working}} from the snapshot (sessions with no video map to an empty dict); working =
    in_working AND class in ``WORKING_CLASSES[sel]``."""
    import pandas as pd

    from wfield_local import trial_classes as tc
    D = pd.read_csv(path, usecols=["label", "cue_idx", "cls", "position", "working_pl_lick"])
    w = tc.in_working(D.cls, D.cue_idx, session=D.label, position=D.position, end=END)
    if not np.array_equal(w, D.working_pl_lick.to_numpy(bool)):
        raise RuntimeError(f"{path}: trial_classes.in_working no longer reproduces the snapshot's working_pl_lick "
                           "-- rebuild it with scripts/s1_trial_class_snapshot.py")
    # in_working (the jaw-INCLUSIVE robustness set; e3 keeps trial_classes' "working" gate that way on purpose) is what
    # the snapshot check above pins; PRODUCTION narrows it to the camera-lick gate (contact | incomplete)
    if sel == IN_WORKING:
        w = w & tc.in_gate(D.cls, "camera_lick")
    elif sel == IN_WORKING_CONTACT:                     # supplement (Priya 11-07): no GATES entry is contact-only
        w = w & (D.cls == "contact").to_numpy(bool)
    elif sel != IN_WORKING_JAW:
        raise ValueError(f"{sel!r} is not a camera working selection")
    D["w"] = w
    out = {}
    for lab, g in D.groupby("label"):
        out[lab] = {} if (g.cls == "no_video").all() else dict(zip(g.cue_idx.astype(int), g.w.astype(bool)))
    return out


@functools.lru_cache(maxsize=2)
def _cls_table(path: str):
    """{label: {cue_idx: class}} from the snapshot (the A4 classes; no-video sessions map to an empty dict)."""
    import pandas as pd
    D = pd.read_csv(path, usecols=["label", "cue_idx", "cls"])
    return {lab: ({} if (g.cls == "no_video").all() else dict(zip(g.cue_idx.astype(int), g.cls)))
            for lab, g in D.groupby("label")}


def row_classes(kept, indices, GE, GU):
    """``(cls_e, cls_u)``: the A4 camera class of every ENGAGED and NO-LICK row of a pool ("" where none: no video,
    or a cue the video has no trial for). For the reported 3-state discriminator variant (lick / jaw only / none)."""
    T = _cls_table(str(snapshot_path()))
    GE, GU = np.asarray(GE), np.asarray(GU)
    ce, cu = np.full(len(GE), "", dtype=object), np.full(len(GU), "", dtype=object)
    for i, lab in enumerate(kept):
        cls = T.get(lab) or {}
        if not cls:
            continue
        ie, inl = indices.get(lab, (np.zeros(0, int), np.zeros(0, int)))
        ce[np.flatnonzero(GE == i)] = [cls.get(int(k), "") for k in ie]
        cu[np.flatnonzero(GU == i)] = [cls.get(int(k), "") for k in inl]
    return ce, cu


# ------------------------------------------------------------------------------------------ section G groups
#: SECTION G's post-stroke outcome groups (main 4a37cbc8, Priya 11-04, item G): the camera class of the trial within
#: cue -> trial stop, and trials after the session's LAST CAMERA LICK as their own group ("after_end", replacing the DAQ
#: quit-tail "stopped"; captioned "quit", `G_LABELS`). Close mid follows e3's rule (`trial_classes`): no lick / contact
#: / tongue seen -> no response, flagged ``jaw_judged`` = False; an "unknown" trial (jaw not judgeable elsewhere) is also
#: "none" with ``jaw_judged`` = False. Main has NO stop rule; THE REBUILD routes its session-level classes through
#: `g_group` on the RAW per-cue class (`trial_classes.cue_groups_for`): early_run -> its raw class (no_response -> none;
#: Priya 11-04 "early_run folds into no_response"), stopped -> its raw class (no_response -> none, jaw_only -> jaw_only)
#: or after_end when it lies after the last camera lick, after_end -> after_end. So the five groups are main's exactly.
G_GROUPS = ("contact", "incomplete", "jaw_only", "none", "after_end")
#: the groups that are WORKING trials (the decoders' training / centre rows; section G's pre panel)
G_WORKING = ("contact", "incomplete")
#: REBUILD-ONLY groups, after merging rebuild/cue70 (coordinator 11-04): "stopped_licking" (a lick-touched trial inside a
#: stop run / the quit tail, cue70 3613cf0b: in neither stopped nor working) is its OWN group; "stopped" = a mid-session
#: stop RUN, by the class source's run / tail marker (`trial_classes.stop_kind_for`, cue70 88b1cb8e; `g_group_final`
#: ``stop_kind``); a stopped trial of the quit TAIL is "after_end" ("quit").
G_EXTRA = ("stopped", "stopped_licking")
G_ALL = G_GROUPS + G_EXTRA
#: caption labels (Priya 11-04: after_end reads "quit")
G_LABELS = {"contact": "contact", "incomplete": "incomplete", "jaw_only": "jaw only", "none": "no response",
            "after_end": "quit", "stopped": "stopped (mid-session run)", "stopped_licking": "stopped, licking"}


def g_group(cls, before_end) -> np.ndarray:
    """THE ONE MAPPING from camera classes to section G's groups (the port seam for the rebuild): per trial, its RAW
    `trial_classes` class (contact / incomplete / jaw_only / no_response / unknown) and whether it is at or before the
    session's last camera lick (`trial_classes.before_session_end`, anchor "lick"). Returns one of `G_GROUPS` per
    trial: "after_end" for every trial after the last camera lick, else contact / incomplete / jaw_only, and "none" for
    no_response and unknown. Anything else (no video) -> "". The rebuild's session-level classes (stopped, early_run,
    after_end) are NOT inputs here: `trial_classes.cue_groups_for` passes each trial's raw class and end flag."""
    cls = np.asarray(cls, dtype=object)
    before_end = np.asarray(before_end, bool)
    out = np.full(len(cls), "", dtype=object)
    for c in ("contact", "incomplete", "jaw_only"):
        out[cls == c] = c
    out[np.isin(cls, ["no_response", "unknown"])] = "none"
    out[(out != "") & ~before_end] = "after_end"
    return out


def g_group_final(final, raw, before_end, stop_kind=None) -> np.ndarray:
    """`g_group` on the RAW class, then the rebuild's session-level classes that must keep their own group
    (coordinator 11-04, for the cue70 merge): ``final`` == "stopped_licking" -> "stopped_licking"; with ``stop_kind``
    (per trial "run" / "tail" / "", the proposed marker from `classify_session`) a stopped trial of the quit TAIL ->
    "after_end" ("quit", also when A2 moves every post-last-lick trial into stopped) and of a mid-session RUN ->
    "stopped". Without ``stop_kind`` stopped / early_run trials keep the raw-class route."""
    out = g_group(raw, before_end)
    final = np.asarray(final, dtype=object)
    if stop_kind is not None:
        k = np.asarray(stop_kind, dtype=object)
        st = final == "stopped"
        out[st & (k == "tail")] = "after_end"
        out[st & (k == "run")] = "stopped"
    out[final == "stopped_licking"] = "stopped_licking"
    return out


def g_jaw_judged(cls, position) -> np.ndarray:
    """False where the jaw could not be judged: "unknown" trials, and no-response trials at the close-mid spout
    (`trial_classes.JAW_UNJUDGED_POSITIONS`), whose "none" is the default rather than an observation. ``cls`` = the RAW
    class (as `g_group`)."""
    from wfield_local import trial_classes as tc
    cls = np.asarray(cls, dtype=object)
    hidden = np.isin(np.asarray(position, dtype=object), list(tc.JAW_UNJUDGED_POSITIONS))
    return ~((cls == "unknown") | ((cls == "no_response") & hidden))


@functools.lru_cache(maxsize=2)
def _group_table(path: str):
    """{label: {cue_idx: (group, jaw_judged)}} from main's s1 snapshot (no-video sessions map to an empty dict). Kept
    for parity with main; the rebuild's pools use the LIVE classes (`row_groups`)."""
    import pandas as pd

    from wfield_local import trial_classes as tc
    D = pd.read_csv(path, usecols=["label", "cue_idx", "cls", "position"])
    be = tc.before_session_end(D.cls, D.cue_idx, session=D.label, position=D.position, anchor=END)
    D["g"] = g_group(D.cls, be)
    D["jj"] = g_jaw_judged(D.cls, D.position)
    out = {}
    for lab, g in D.groupby("label"):
        out[lab] = ({} if (g.cls == "no_video").all()
                    else dict(zip(g.cue_idx.astype(int), zip(g.g, g.jj.astype(bool)))))
    return out


def row_groups(kept, indices, GE, GU, YE=None, YU=None):
    """``(grp_e, grp_u, jj_e, jj_u, have)``: section G's group (`g_group`) and ``jaw_judged`` of every ENGAGED and
    NO-LICK row of a pool ("" / True where the session has no camera classes or the cue no trial), and the set of
    pooled session indices that HAVE camera classes. ``indices`` as in `row_masks`.

    REBUILD: the classes are LIVE (`trial_classes.cue_groups_for`, the same session-level rules as `cue_classes_for`),
    never main's s1 snapshot; ``YE`` / ``YU`` (spout codes) give each row's position for ``jaw_judged`` (close-mid
    defaults). A session without video (no matched cue: PS93_0813) is not in ``have``."""
    from wfield_local import trial_classes as tc
    from wfield_local.plot_lick_aligned_averages import POSITION_NAMES
    GE, GU = np.asarray(GE), np.asarray(GU)
    ge, gu = np.full(len(GE), "", dtype=object), np.full(len(GU), "", dtype=object)
    je, ju = np.ones(len(GE), bool), np.ones(len(GU), bool)
    have = set()
    for i, lab in enumerate(kept):
        ie, inl = indices.get(lab, (np.zeros(0, int), np.zeros(0, int)))
        me, mu = np.flatnonzero(GE == i), np.flatnonzero(GU == i)
        if len(ie) != len(me) or len(inl) != len(mu):
            raise RuntimeError(f"{lab}: {len(ie)} / {len(inl)} cue indices for {len(me)} / {len(mu)} pooled rows")
        cs = tc.daq_cue_seconds(lab)
        raw, after, matched = tc.cue_groups_for(lab, cs)
        if not np.any(matched):
            continue
        have.add(i)
        # the final classes keep stopped_licking apart; the run / tail marker (`stop_kind_for`, cue70) routes stopped
        # trials to "stopped" (mid-session run) / "after_end" (quit tail)
        grp_all = g_group_final(tc.cue_classes_for(lab, cs), raw, ~after, stop_kind=tc.stop_kind_for(lab, cs))
        for rows, idx, Y, g_, j_ in ((me, ie, YE, ge, je), (mu, inl, YU, gu, ju)):
            idx = np.asarray(idx, np.int64)
            ok = (idx >= 0) & (idx < len(raw))
            g_[rows[ok]] = grp_all[idx[ok]]
            if Y is not None:
                pos = np.array([POSITION_NAMES.get(int(v), str(v)) for v in np.asarray(Y)[rows[ok]]], dtype=object)
                j_[rows[ok]] = g_jaw_judged(raw[idx[ok]], pos)
            else:
                j_[rows[ok]] = raw[idx[ok]] != "unknown"
    return ge, gu, je, ju, have


def classes_available(label: str) -> bool:
    return bool(_table(str(snapshot_path()), IN_WORKING).get(label))


def row_masks(kept, indices, GE, GU, sel: str | None = None):
    """``(work_e, work_u, with_classes)``: working (``selection()``, or ``sel``: in_working & the selection's classes)
    per ENGAGED row and per NO-LICK row of a pool, and the set of pooled session indices that HAVE camera classes.
    ``indices[label] = (engaged cue idx, no-lick cue idx)``, row for row with that session's XE / XU rows (the feature
    builder's ``.indices``)."""
    sel = selection() if sel is None else sel
    if not is_camera(sel):
        sel = IN_WORKING          # quit_tail callers asking for masks get the production camera set
    T = _table(str(snapshot_path()), sel)
    GE, GU = np.asarray(GE), np.asarray(GU)
    we, wu = np.zeros(len(GE), bool), np.zeros(len(GU), bool)
    have = set()
    for i, lab in enumerate(kept):
        cls = T.get(lab) or {}
        ie, inl = indices.get(lab, (np.zeros(0, int), np.zeros(0, int)))
        me, mu = np.flatnonzero(GE == i), np.flatnonzero(GU == i)
        if len(ie) != len(me) or len(inl) != len(mu):
            raise RuntimeError(f"{lab}: {len(ie)} / {len(inl)} cue indices for {len(me)} / {len(mu)} pooled rows")
        if not cls:
            continue
        have.add(i)
        we[me] = [cls.get(int(k), False) for k in ie]
        wu[mu] = [cls.get(int(k), False) for k in inl]
    return we, wu, have


def projection_work(kept, features, GE, GU, align):
    """``(work_e, work_u, cls_e, cls_u)`` for `spout_scaling.projection_arrays` under
    ``WIDEFIELD_PROJECTION_CENTRE=working`` / ``contact`` (main 4a37cbc8, item C re-check), or None: the lick window
    (lick gate), the DAQ gate or a non-camera selection, the engaged center, the frame off, or a builder without cue
    ``.indices``. REBUILD: the classes are live (`trial_classes.pool_row_classes`), the working rows `in_sel` of them."""
    from wfield_local import spout_scaling as ss
    from wfield_local import trial_classes as tc
    if (align == "lick" or not tc.camera_gate_on() or not is_camera(tc.working_selection())
            or not ss.projection_frame_on() or ss.projection_centre() == "engaged"):
        return None                   # the lick window / the DAQ gate / the frame off: the engaged-row frame
    if not getattr(features, "indices", None):
        raise RuntimeError(f"projection frame ({ss.projection_tag()}): a camera {align} pool without cue indices "
                           "cannot take the working-row center -- build it with features_with_indices")
    ce, cu = tc.pool_row_classes(features, kept, len(GE), len(GU))
    return tc.in_sel(ce, "working"), tc.in_sel(cu, "working"), ce, cu


def eax_rows(kept, features, XE, XU, GE, GU):
    """``(XE, XU)`` in the ENGAGEMENT-AXIS frame (`spout_scaling.contact_z`: z on the camera contact rows), from the
    PRODUCTION rows; the inputs themselves under the DAQ gate or without cue indices."""
    from wfield_local import spout_scaling as ss
    from wfield_local import trial_classes as tc
    if not tc.camera_gate_on() or not getattr(features, "indices", None):
        return XE, XU
    ce, cu = tc.pool_row_classes(features, kept, len(GE), len(GU))
    return ss.contact_z(XE, XU, GE, GU, ce, cu)


def projection_cls_e(kept, features, GE, GU, align):
    """The camera class per ENGAGED row for the projection frame's engaged center (`spout_scaling.centre_mask` needs
    them under the camera center), or None (the DAQ gate / the frame off / no ``.indices``)."""
    from wfield_local import spout_scaling as ss
    from wfield_local import trial_classes as tc
    if not ss.projection_frame_on() or not tc.camera_gate_on() or not getattr(features, "indices", None):
        return None
    return tc.pool_row_classes(features, kept, len(GE), len(GU))[0]


def roi_features_with_indices(nolick_ref="cue"):
    """The default (ROI) `pool_sessions` feature builder, also keeping each session's cue indices (``.indices``),
    so a ROI pool can be mapped onto the camera classes. ROI features are not disk-cached (`trial_features_cached`
    computes non-LocaNMF sources), so this costs nothing extra."""
    from wfield_local.locanmf_position_decoder import trial_features_cached
    idx = {}

    def _feat(s, args):
        out = trial_features_cached(s, args, with_indices=True, nolick_ref=nolick_ref)
        idx[s["label"]] = (out[6], out[7])
        return out[:6]

    _feat.indices = idx
    _feat.basis_id = None
    return _feat
