"""The imaging decoders' WORKING trials = `trial_classes.in_working` (Priya 11-03), mapped onto pooled feature rows.

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
IN_WORKING = "in_working"
QUIT_TAIL = "quit_tail"
END = "lick"
SNAPSHOT_ENV = "WFIELD_S1_CLASSES"        # the same override 5b's s1 script reads


def selection() -> str:
    """``"in_working"`` (production since 11-03) or ``"quit_tail"`` (the previous lick + miss-while-working set)."""
    v = os.environ.get(WORKING_TRIALS_ENV, IN_WORKING).strip().lower()
    if v in (IN_WORKING, "camera", "a4"):
        return IN_WORKING
    if v in (QUIT_TAIL, "flag_engagement", "old"):
        return QUIT_TAIL
    raise ValueError(f"{WORKING_TRIALS_ENV}={v!r}: expected in_working or quit_tail")


def tag() -> str:
    """The selection tag that goes into every frozen-model spec trained on working trials (`trial_classes`)."""
    from wfield_local import trial_classes as tc
    return tc.selection_tag("camera", tc.DEFAULT_WINDOW, end=END)


def snapshot_path() -> Path:
    if os.environ.get(SNAPSHOT_ENV):
        return Path(os.environ[SNAPSHOT_ENV])
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    return oo.root(PathResolver()) / oc.RUN / "imaging_encoding" / "qc" / "s1_trial_classes_cue_stop.csv.gz"


@functools.lru_cache(maxsize=2)
def _table(path: str):
    """{label: {cue_idx: in_working}} from the snapshot (sessions with no video map to an empty dict)."""
    import pandas as pd

    from wfield_local import trial_classes as tc
    D = pd.read_csv(path, usecols=["label", "cue_idx", "cls", "position", "working_pl_lick"])
    w = tc.in_working(D.cls, D.cue_idx, session=D.label, position=D.position, end=END)
    if not np.array_equal(w, D.working_pl_lick.to_numpy(bool)):
        raise RuntimeError(f"{path}: trial_classes.in_working no longer reproduces the snapshot's working_pl_lick "
                           "-- rebuild it with scripts/s1_trial_class_snapshot.py")
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
    return bool(_table(str(snapshot_path())).get(label))


def row_masks(kept, indices, GE, GU):
    """``(work_e, work_u, with_classes)``: in_working per ENGAGED row and per NO-LICK row of a pool, and the set of
    pooled session indices that HAVE camera classes. ``indices[label] = (engaged cue idx, no-lick cue idx)``, row for
    row with that session's XE / XU rows (the feature builder's ``.indices``)."""
    T = _table(str(snapshot_path()))
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
