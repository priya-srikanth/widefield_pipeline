"""How the FROZEN SPOUT-POSITION decoder normalises a session: ``session_bal_em`` (production since 2026-11-03).

Priya, 11-03: "I would adopt B everywhere" -- arm B of the spout-guard check (DECISIONS 2026-11-03, 24e9dac1; the
class-mix check fa70b651), at every alignment (precue, cue, lick), with NO guard. DECISIONS 2026-11-03 "Spout-position
frozen decoder: bal_em adopted".

THE PROBLEM IT FIXES. `locanmf_frozen_decoder.pool_sessions` z-scores each session with the mean / SD of its ENGAGED
trials (a lick within max_rt), an outcome-gated set, and applies the same transform to the session's no-lick trials.
Acutely far contra is .2-3 % of the engaged set (vs ~1/6 pre), so the session mean leaves far contra and every trial is
pushed toward the far-contra pattern: production UNDER-states the acute far-contra loss (~.10 cue/working) and
OVER-states the near-position losses (~.05-.07).

THE FIX (the same EM centring as the state decoder; primitives in `label_shift`):
  * TRAINING (pre-stroke sessions; labels are legitimate): each session centred on the EQUAL-weight mean of its TRUE
    class means, divided by production's SD (over its engaged trials).
  * SCORING (every scored session, pre held-out and post; NO labels, by signature): production's SD over the engaged
    trials; start from production's mean; 3 passes of frozen-model posteriors -> Saerens EM priors (training prior =
    the model's training class mix) -> soft class means of the engaged trials -> centre = their equal-weight (1/6)
    mean -> re-z-score. The engaged-set centre and SD are applied to the session's no-lick trials too, as production
    does. No guard (the G1 pre-geometry guard was NOT adopted: Priya rejected applying pre-stroke structure
    selectively); ``guard: None`` is recorded in the spec.

WORKS IN THE PRODUCTION Z FRAME. Every caller already holds `pool_sessions`' z-scored matrices. Both transforms are
affine per session with the SAME SD, and a (soft or true) class mean commutes with an affine map, so re-centring the
z-scored rows is algebraically identical to doing it on the raw features (checked against the raw-feature arm B:
max |centre difference| ~1e-15). Zero-SD features stay zero, as in production.

``WIDEFIELD_SPOUT_SCALING=session_z`` reproduces the pre-11-03 production run exactly: no transform, and the frozen spec
carries no ``scaling`` key, so the old stored models (and their ids) are what it loads.

LIMITS (DECISIONS): when a spout is nearly absent from the engaged set (acute far contra) about 30-40 % of the bias
remains, in the conservative direction (the far-contra loss is still slightly under-stated); at precue the EM is weak
and only ~37 % of the artefact is removed.
"""
from __future__ import annotations

import os

import numpy as np

from wfield_local import label_shift

SPOUT_SCALING_ENV = "WIDEFIELD_SPOUT_SCALING"
SCALING_SESSION_Z = "session_z"          # pre-11-03 production (engaged-trial z-score only); NO spec key
SCALING_BAL_EM = "session_bal_em"
DEFAULT_SCALING = SCALING_BAL_EM
#: The six spout codes the position decoder is trained on (`plot_lick_aligned_averages.DISPLAY_ORDER` as integers).
N_CLASSES = 6
BAL_EM_PASSES = label_shift.BAL_EM_PASSES


def spout_scaling():
    """``"session_bal_em"`` (default) or None (= ``session_z``, the pre-11-03 engaged-trial z-score alone)."""
    v = os.environ.get(SPOUT_SCALING_ENV, DEFAULT_SCALING).strip().lower()
    if v in ("session_bal_em", "bal_em", "balem"):
        return SCALING_BAL_EM
    if v in ("session_z", "z", "zscore", "none", "production_z", "old"):
        return None
    raise ValueError(f"{SPOUT_SCALING_ENV}={v!r}: expected session_bal_em or session_z")


def scaling_params():
    """The spec ``scaling_params`` for the spout decoder's bal_em. Everything that defines the transform is recorded,
    including the guard that is OFF."""
    return {"centre_set": "engaged", "guard": None, "min_mass": None, "n_outer": BAL_EM_PASSES,
            "train_centre": "true_class_means_equal_weight"}


def spec_kw(scaling="env"):
    """``{"scaling": ..., "scaling_params": ...}`` for `frozen_models.make_spec`; EMPTY under session_z so that spec
    (and its id) is byte-identical to the pre-11-03 one."""
    if scaling == "env":
        scaling = spout_scaling()
    if scaling is None:
        return {}
    return {"scaling": scaling, "scaling_params": scaling_params()}


# ------------------------------------------------------------------------------------------------ training side
def train_frame(XE, YE, GE, sessions, classes=None):
    """A copy of the (production-z) engaged matrix with every TRAINING session's rows re-centred on the equal-weight
    mean of its TRUE class means. Rows of other sessions are left untouched (they are never trained on)."""
    classes = list(range(N_CLASSES)) if classes is None else list(classes)
    XE = np.asarray(XE, float)
    out = XE.copy()
    GE, YE = np.asarray(GE), np.asarray(YE)
    for i in sessions:
        m = GE == i
        if m.any():
            Xi = XE[m]
            out[m] = (Xi - label_shift.balanced_centre(Xi, YE[m], classes)) / label_shift.session_sd(Xi)
    return out


def class_prior(y, classes=None):
    """The model's TRAINING class mix (the EM's training prior), in class-code order 0..5."""
    classes = list(range(N_CLASSES)) if classes is None else list(classes)
    return label_shift.class_prior([np.asarray(y)], classes)


# ------------------------------------------------------------------------------------------------ scoring side
def score_centre(model, Xe, pi_train):
    """``(mu, sd)`` for ONE scored session from its ENGAGED rows only. Takes NO labels, by signature: nothing about the
    scored session's true positions can reach its centre."""
    Xe = np.asarray(Xe, float)
    if len(Xe) < 2:                       # nothing to estimate from: production's centring
        return (Xe.mean(0) if len(Xe) else 0.0), (label_shift.session_sd(Xe) if len(Xe) else 1.0)
    if [int(c) for c in model.classes_] != list(range(len(pi_train))):
        raise ValueError(f"bal_em needs a model over classes 0..{len(pi_train) - 1}, got {list(model.classes_)}")
    _Z, info = label_shift.bal_em_centre(model, Xe, pi_train, n_outer=BAL_EM_PASSES, min_mass=None,
                                         return_info=True)
    return info["centre"], info["sd"]


def score_session(model, Xe, Xu, pi_train):
    """``(Ze, Zu)``: one session's engaged and no-lick rows in the bal_em scoring frame (centre + SD from the engaged
    rows, applied to both, as production does). No labels."""
    mu, sd = score_centre(model, Xe, pi_train)
    Ze = (np.asarray(Xe, float) - mu) / sd
    Zu = (np.asarray(Xu, float) - mu) / sd if Xu is not None and len(Xu) else (
        np.zeros((0, np.asarray(Xe).shape[1])) if Xu is not None else None)
    return Ze, Zu


def scored_frame(XE, GE, XU, GU, sessions, model_of, prior_of):
    """Copies of the engaged / no-lick matrices with each session in ``sessions`` moved into the bal_em scoring frame,
    centred with ITS scoring model (``model_of(i)``: the LOSO model for a held-out pre session, the full model for a
    post session) and that model's training prior (``prior_of(i)``). Sessions without a model are left as they are."""
    XE = np.asarray(XE, float)
    XU = np.asarray(XU, float)
    GE, GU = np.asarray(GE), np.asarray(GU)
    XEs, XUs = XE.copy(), XU.copy()
    for i in sessions:
        mdl = model_of(i)
        if mdl is None:
            continue
        me = GE == i
        mu_ = (GU == i) if len(GU) else np.zeros(0, bool)
        Ze, Zu = score_session(mdl, XE[me], XU[mu_] if mu_.any() else None, prior_of(i))
        XEs[me] = Ze
        if mu_.any():
            XUs[mu_] = Zu
    return XEs, XUs
