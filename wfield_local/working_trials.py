"""The imaging decoders' WORKING trials (Priya 11-03), mapped onto pooled feature rows.

PRODUCTION SINCE 11-03 LATE (Priya): working = ``in_working & class in {contact, incomplete}`` -- the camera LICK
trials (contact or incomplete lick), cue -> trial stop, cut at the session's last camera lick. JAW-ONLY TRIALS ARE NOT
WORKING: they are reported as their own class, like no-response (docs/status/JAW_ONLY_CHECKS_2026-11-03.md: jaw_only
reads like no_response in PS94 / PS95 at cue and precue; adding them to training does not move the pre reference;
centring a post-stroke session on rows that include them drives PS95 close ipsi acute .975 -> .642). Every contact /
incomplete trial is at or before the last camera lick by construction, so the end cut only ever removed jaw_only /
no_response trials; it stays in the definition so the rule reads as written. ``WORKING_CLASSES`` names the sets.

The 11-03 AFTERNOON definition (in_working INCLUDING jaw_only) is the robustness arm
``WIDEFIELD_WORKING_TRIALS=in_working_jaw``; its frozen-model specs (tag without the class suffix) are the 29 models
published 11-03 afternoon, so it reproduces them exactly.

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

Sessions with NO camera classes (no behaviour video: PS93_0813; the June sessions, outside the August-only baseline)
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
QUIT_TAIL = "quit_tail"                   # robustness: DAQ lick + miss-while-working (flag_engagement quit tail)
CAMERA_SELECTIONS = (IN_WORKING, IN_WORKING_JAW)
#: The camera classes a working trial may have, per camera selection (both also cut at the last camera lick). Production
#: = trial_classes.GATES["camera_lick"]; robustness = trial_classes.GATES["working"] (pinned in tests).
WORKING_CLASSES = {IN_WORKING: frozenset({"contact", "incomplete"}),
                   IN_WORKING_JAW: frozenset({"contact", "incomplete", "jaw_only"})}
END = "lick"
SNAPSHOT_ENV = "WFIELD_S1_CLASSES"        # the same override 5b's s1 script reads


def selection() -> str:
    """``"in_working"`` (production since 11-03 late: contact + incomplete licks, cut at the last camera lick),
    ``"in_working_jaw"`` (the 11-03 afternoon definition, jaw_only included) or ``"quit_tail"`` (the previous lick +
    miss-while-working set)."""
    v = os.environ.get(WORKING_TRIALS_ENV, IN_WORKING).strip().lower()
    if v in (IN_WORKING, "camera", "a4", "licks", "in_working_licks"):
        return IN_WORKING
    if v in (IN_WORKING_JAW, "in_working_incl_jaw", "with_jaw"):
        return IN_WORKING_JAW
    if v in (QUIT_TAIL, "flag_engagement", "old"):
        return QUIT_TAIL
    raise ValueError(f"{WORKING_TRIALS_ENV}={v!r}: expected in_working, in_working_jaw or quit_tail")


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
