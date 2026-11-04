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


#: The engagement-STATE discriminators (lick vs no-lick: section G looks_like_which / fits_engaged_distribution /
#: undetected_state_split, precue_engagement_states C / D) can be scored with bal_em over the two STATE classes, but it is
#: OFF by default: measured 11-03 (labcams/decoder_mix_checks/spout_balem_production/summarise_extension_log.txt) it
#: breaks the discriminators' own pre-stroke reference -- held-out PRE engaged trials read engaged-like .92 -> .64 (cue)
#: and .80 -> .43 (precue) -- because these discriminators separate the states only weakly (AUC ~.6-.8), so the EM
#: cannot recover a session's state mix and the centre lands between the states. Needs Priya's decision; selectable
#: with WIDEFIELD_STATE_DISC_SCALING=state_bal_em.
STATE_DISC_SCALING_ENV = "WIDEFIELD_STATE_DISC_SCALING"
SCALING_STATE_BAL_EM = "state_bal_em"
#: The SPOUT bal_em frame for the state discriminators (Priya 11-03, item B): each session's rows centred exactly as
#: the frozen SPOUT decoder centres them (EM over the six spouts, label-free; training sessions on their true spout
#: class means), so the spout-mix shift of the engaged set is removed without centring on the state contrast itself.
SCALING_SPOUT_FRAME = "spout_frame"
#: STAYS session_z after the 11-04 re-check under the jaw-out working definition (Priya's criterion: pre checks within
#: the per-animal between-session noise AND no pre reference worse by > .03). One cell missed: PS93's cue held-out pre
#: engaged reference .869 -> .850 (-.019, between-session SD .016; nothing worse than .03). DECISIONS 2026-11-04;
#: labcams/decoder_mix_checks/bcdg_1104/summarise_B_log.txt.
STATE_DISC_DEFAULT = "session_z"


def state_disc_scaling():
    """``"spout_frame"`` / ``"state_bal_em"`` / None (the production engaged-trial z-score) for the engagement-state
    discriminators (``WIDEFIELD_STATE_DISC_SCALING``; default `STATE_DISC_DEFAULT`). ``spout_frame`` applies only while
    the spout decoder itself is on bal_em."""
    v = os.environ.get(STATE_DISC_SCALING_ENV, STATE_DISC_DEFAULT).strip().lower()
    if v in ("state_bal_em", "bal_em"):
        return SCALING_STATE_BAL_EM
    if v in ("spout_frame", "spout", "spout_bal_em"):
        return SCALING_SPOUT_FRAME if spout_scaling() else None
    if v in ("session_z", "z", "none", "off"):
        return None
    raise ValueError(f"{STATE_DISC_SCALING_ENV}={v!r}: expected spout_frame, state_bal_em or session_z")


#: WHICH ROWS centre a session in the discriminators' ``spout_frame`` (Priya 11-04, item B): "working" (default) = the
#: camera WORKING rows (contact + incomplete, engaged AND no-lick rows; `working_trials`), exactly the rows the cue /
#: precue spout decoders are trained and centred on (`locanmf_frozen_decoder.frozen_scored_frame`); "engaged" = every
#: DAQ-engaged row (what `precue_engagement_states` used before 11-04; robustness). Section G's discriminators
#: (`poststroke_compare._spout_train_d` / `frozen_X`) already centred on the working rows; the lick window always uses
#: the lick gate (engaged rows).
STATE_DISC_CENTRE_ENV = "WIDEFIELD_STATE_DISC_CENTRE"


def state_disc_centre() -> str:
    v = os.environ.get(STATE_DISC_CENTRE_ENV, "working").strip().lower()
    if v in ("working", "work", "camera"):
        return "working"
    if v in ("engaged", "daq", "old"):
        return "engaged"
    raise ValueError(f"{STATE_DISC_CENTRE_ENV}={v!r}: expected working or engaged")


def scaling_params():
    """The spec ``scaling_params`` for the spout decoder's bal_em. Everything that defines the transform is recorded,
    including the guard that is OFF."""
    return {"centre_set": "engaged", "guard": None, "min_mass": None, "n_outer": BAL_EM_PASSES,
            "train_centre": "true_class_means_equal_weight"}


def spec_kw(scaling="env", trials=None):
    """``{"scaling": ..., "scaling_params": ...}`` for `frozen_models.make_spec`; EMPTY under session_z with the
    engaged training set, so that spec (and its id) is byte-identical to the pre-11-03 one. ``trials`` = the training
    trial selection when it is NOT the engaged set (the camera working trials, `working_trials.tag()`, Priya 11-03):
    recorded as ``scaling_params["trials"]``, so a working-trained model never shares an id with an engaged one."""
    if scaling == "env":
        scaling = spout_scaling()
    if scaling is None:
        return {} if trials is None else {"scaling_params": {"trials": trials}}
    params = scaling_params()
    if trials is not None:
        params["trials"] = trials
        params["centre_set"] = "working"
    return {"scaling": scaling, "scaling_params": params}


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
class _Ordinal:
    """A view of a fitted classifier whose classes are re-indexed 0..K-1 in ``model.classes_`` order, so the EM can run
    on a model over a SUBSET of the spouts (the class-filtered lick-only arm of section G) or over non-integer state
    codes. Only ``predict_proba`` and ``classes_`` are used by `label_shift.bal_em_centre`."""

    def __init__(self, model):
        self.model = model
        self.classes_ = np.arange(len(model.classes_))

    def predict_proba(self, Z):
        return self.model.predict_proba(Z)


def score_centre(model, Xe, pi_train, *, mu0=None, sd=None):
    """``(mu, sd)`` for ONE scored session from its centre-set rows (the ENGAGED rows for a spout decoder). Takes NO
    labels, by signature: nothing about the scored session's true classes can reach its centre.

    ``pi_train`` is the model's training class mix: in class-code order 0..5 for the six-spout model (the 11-03
    production call, unchanged), or in ``model.classes_`` order for any other model (a spout subset, a 2-class state
    discriminator). ``mu0`` / ``sd`` fix the starting centre and the SD (see `label_shift.bal_em_centre`)."""
    Xe = np.asarray(Xe, float)
    if len(Xe) < 2:                       # nothing to estimate from: production's centring
        if mu0 is not None:
            return np.asarray(mu0, float), (1.0 if sd is None else np.asarray(sd, float))
        return (Xe.mean(0) if len(Xe) else 0.0), (label_shift.session_sd(Xe) if len(Xe) else 1.0)
    codes = [int(c) for c in model.classes_]
    if codes == list(range(len(pi_train))):
        m = model
    elif len(codes) == len(pi_train):
        m = _Ordinal(model)
    else:
        raise ValueError(f"pi_train has {len(pi_train)} entries for a model over classes {list(model.classes_)}")
    _Z, info = label_shift.bal_em_centre(m, Xe, pi_train, n_outer=BAL_EM_PASSES, min_mass=None,
                                         return_info=True, mu0=mu0, sd=sd)
    return info["centre"], info["sd"]


def model_prior(model, y):
    """The training class mix of ``y`` in ``model.classes_`` order (the EM prior for any model)."""
    y = np.asarray(y)
    return np.array([float(np.mean(y == c)) for c in model.classes_])


class FrozenFrame:
    """One pre-stroke-trained spout decoder over a pooled set, and the frames it is trained and scored in.

    For the in-memory copies of the frozen spout-position decoder that the first 11-03 adoption had left on session_z
    (DECISIONS 2026-11-03, "bal_em extended to every cross-session decoder"). ``XE`` / ``YE`` / ``GE`` are the
    PRODUCTION-frame engaged rows (each session z-scored on its engaged trials) with a session id per row;
    ``train_sessions`` the pre-stroke ids; ``pipe`` the estimator factory; ``classes`` the spout codes the model is
    trained on (default: every code in the training rows; a subset for a class-filtered arm, whose training rows the
    caller restricts with ``train_mask``).

      * ``XT``            training frame: each training session centred on the equal-weight mean of its TRUE class
                          means (over ``classes``), production SD
      * ``full``          fitted on every training row; ``loso[i]`` on all but session ``i`` (``loso=True``)
      * ``centre(i, m)``  the label-free EM centre of session ``i``'s engaged rows (ALL of them, whatever the arm
                          scores) under model ``m`` ("full" / "loso"), production SD; ``apply`` moves any rows of
                          session ``i`` (engaged or no-lick) into that frame

    Under ``WIDEFIELD_SPOUT_SCALING=session_z`` ``XT`` is ``XE`` itself and every centre is the identity, so callers are
    bit-identical to the pre-11-03 code; the models are then exactly the fits the callers always made.
    """

    def __init__(self, XE, YE, GE, train_sessions, pipe, *, classes=None, train_mask=None, loso=True, scaling="env"):
        self.scaling = spout_scaling() if scaling == "env" else scaling
        self.XE, self.YE, self.GE = np.asarray(XE, float), np.asarray(YE), np.asarray(GE)
        self.train_sessions = sorted(train_sessions)
        tm = np.isin(self.GE, self.train_sessions)
        if train_mask is not None:
            tm &= np.asarray(train_mask, bool)
        self.train_mask = tm
        if classes is None:
            classes = sorted(int(c) for c in np.unique(self.YE[tm]))
        self.classes = list(classes)
        self._tc = {}
        if self.scaling:
            self.XT = self.XE.copy()
            for i in self.train_sessions:
                m = self.GE == i
                rows = m & tm
                if rows.any():
                    c = label_shift.balanced_centre(self.XE[rows], self.YE[rows], self.classes)
                    sd = label_shift.session_sd(self.XE[m])
                    self.XT[m] = (self.XE[m] - c) / sd
                    self._tc[i] = (c, sd)
        else:
            self.XT = self.XE
        self.full = pipe().fit(self.XT[tm], self.YE[tm])
        self.pi = {"full": model_prior(self.full, self.YE[tm])}
        self.loso = {}
        if loso:
            for i in self.train_sessions:
                tr = tm & (self.GE != i)
                if not tr.any() or len(np.unique(self.YE[tr])) < 2:
                    continue
                self.loso[i] = pipe().fit(self.XT[tr], self.YE[tr])
                self.pi[("loso", i)] = model_prior(self.loso[i], self.YE[tr])
        self._c = {}

    def model(self, which="full", i=None):
        return self.full if which == "full" else self.loso[i]

    def centre(self, i, which="full"):
        key = (i, which)
        if key not in self._c:
            if not self.scaling:
                self._c[key] = None
            else:
                pi = self.pi["full"] if which == "full" else self.pi[("loso", i)]
                self._c[key] = score_centre(self.model(which, i), self.XE[self.GE == i], pi)
        return self._c[key]

    def apply(self, X, i, which="full"):
        """Rows of session ``i`` (production frame) in the scoring frame of model ``which``. No labels."""
        if not self.scaling:
            return X
        mu, sd = self.centre(i, which)
        return (np.asarray(X, float) - mu) / sd

    def train_apply(self, X, i):
        """Rows of TRAINING session ``i`` (any rows: engaged or no-lick) in the training frame (its true-class-mean
        centre and SD). The input under session_z, or for a session that is not a training session."""
        if not self.scaling or i not in self._tc:
            return X
        c, sd = self._tc[i]
        return (np.asarray(X, float) - c) / sd

    def apply_rows(self, X, G, which="full"):
        """``apply`` for a stack of rows from several sessions (``G`` = their session ids)."""
        if not self.scaling:
            return X
        X = np.asarray(X, float)
        G = np.asarray(G)
        out = X.copy()
        for i in np.unique(G):
            m = G == i
            out[m] = self.apply(X[m], i, which)
        return out


# ------------------------------------------------------------------------- engagement-state discriminators
def state_train_frame(XE, GE, XU, GU, sel_e, sel_u, sessions, scaling="env"):
    """TRAINING frame for an engaged-vs-no-lick (state) discriminator: each training session centred on the
    equal-weight mean of its TRUE state-class means (class "engaged" = its ``sel_e`` engaged rows, class "no-lick" =
    its ``sel_u`` no-lick rows), SD unchanged (production engaged-trial SD; these are production-z rows). The same
    shift is applied to all of that session's rows. The input arrays themselves unless opted in
    (``WIDEFIELD_STATE_DISC_SCALING=state_bal_em``; default OFF, see `state_disc_scaling`)."""
    sc = state_disc_scaling() if scaling == "env" else scaling
    if sc != SCALING_STATE_BAL_EM:
        return XE, XU
    XE, XU = np.asarray(XE, float), np.asarray(XU, float)
    GE, GU = np.asarray(GE), np.asarray(GU)
    sel_e, sel_u = np.asarray(sel_e, bool), np.asarray(sel_u, bool)
    oE, oU = XE.copy(), XU.copy()
    for i in sessions:
        me, mu = GE == i, GU == i
        a, b = me & sel_e, mu & sel_u
        if not a.any() or not b.any():
            continue
        c = 0.5 * (XE[a].mean(0) + XU[b].mean(0))
        oE[me] = XE[me] - c
        oU[mu] = XU[mu] - c
    return oE, oU


def state_centre(model, X_all, pi_train):
    """SCORING centre for a state discriminator, label-free: EM over the session's rows of BOTH states (``X_all`` =
    its engaged and no-lick rows, production-z), starting from production's centre (0) with production's SD (1),
    equal weight over the state classes. Returns the centre (a vector to subtract)."""
    X_all = np.asarray(X_all, float)
    D = X_all.shape[1]
    mu, _sd = score_centre(model, X_all, pi_train, mu0=np.zeros(D), sd=np.ones(D))
    return mu


def state_scored(model, pi_train, XE, GE, XU, GU, sel_e, sel_u, sessions, scaling="env"):
    """Copies of ``XE`` / ``XU`` with each session in ``sessions`` moved into the state discriminator's EM scoring
    frame (centre from that session's ``sel_e`` engaged + ``sel_u`` no-lick rows). The inputs unless opted in
    (``WIDEFIELD_STATE_DISC_SCALING=state_bal_em``; default OFF)."""
    sc = state_disc_scaling() if scaling == "env" else scaling
    if sc != SCALING_STATE_BAL_EM:
        return XE, XU
    XE, XU = np.asarray(XE, float), np.asarray(XU, float)
    GE, GU = np.asarray(GE), np.asarray(GU)
    sel_e, sel_u = np.asarray(sel_e, bool), np.asarray(sel_u, bool)
    oE, oU = XE.copy(), XU.copy()
    for i in sessions:
        me, mu = GE == i, GU == i
        rows = np.vstack([XE[me & sel_e], XU[mu & sel_u]])
        if len(rows) < 2:
            continue
        c = state_centre(model, rows, pi_train)
        oE[me] = XE[me] - c
        oU[mu] = XU[mu] - c
    return oE, oU


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


# ------------------------------------------------------------------------- axes / CDs / pattern templates
#: The PROJECTION frame (Priya 11-03, item C): coding directions, pole projections, the engagement axis, pairwise
#: position axes, attempt axes and the pattern / template families project rows centred exactly as the frozen spout
#: decoder centres them, so a session's class mix no longer shifts every projection. ``spout_frame`` uses the
#: ENGAGED-row EM centre of the lick-gate (engaged-trained) spout model for every alignment -- label-free for every
#: session (pre sessions by their leave-one-out model), and no camera needed, so no session drops out of a projection;
#: ``session_z`` = the production engaged-trial z-score (the previous behaviour). Default set by the C check.
PROJECTION_FRAME_ENV = "WIDEFIELD_PROJECTION_FRAME"
PROJECTION_DEFAULT = "session_z"


def projection_frame_on() -> bool:
    v = os.environ.get(PROJECTION_FRAME_ENV, PROJECTION_DEFAULT).strip().lower()
    if v in ("spout_frame", "spout", "bal_em"):
        return spout_scaling() is not None
    if v in ("session_z", "z", "off", "none"):
        return False
    raise ValueError(f"{PROJECTION_FRAME_ENV}={v!r}: expected spout_frame or session_z")


#: WHICH ROWS centre a session in the projection frame (Priya 11-04, item C re-check): "engaged" (the 11-03 build: the
#: lick-gate spout model's engaged-row EM centre) or "working" (the camera WORKING rows, contact + incomplete, engaged and
#: no-lick, as the cue / precue decoders: `grant_kit._frozen_frame(bd, "working")`; the lick window keeps the lick
#: gate). Only read while the projection frame is on (default OFF; Priya decides from the 11-04 re-check numbers).
PROJECTION_CENTRE_ENV = "WIDEFIELD_PROJECTION_CENTRE"


def projection_centre() -> str:
    v = os.environ.get(PROJECTION_CENTRE_ENV, "engaged").strip().lower()
    if v in ("engaged", "daq", "lick_gate"):
        return "engaged"
    if v in ("working", "work", "camera"):
        return "working"
    raise ValueError(f"{PROJECTION_CENTRE_ENV}={v!r}: expected engaged or working")


def projection_arrays(XE, YE, GE, XU, GU, pre_sessions, pipe, work=None, YU=None):
    """``(XE, XU)`` for projection: every session moved into the spout decoder's scoring frame (centre = label-free
    EM centre of its ENGAGED rows, or of its camera WORKING rows under ``WIDEFIELD_PROJECTION_CENTRE=working`` with
    ``work`` = the pool's ``(work_e, work_u)`` masks and ``YU`` its no-lick labels; pre sessions under their
    leave-one-out model, others under the full model). The inputs themselves when the projection frame is off. Under
    the working centre a session with no working rows stays in the production frame, as `grant_kit._frozen_frame`."""
    if not projection_frame_on():
        return XE, XU
    if projection_centre() == "working" and work is not None and work[0] is not None:
        we, wu = np.asarray(work[0], bool), np.asarray(work[1], bool)
        GE, GU = np.asarray(GE), np.asarray(GU)
        RX = np.vstack([np.asarray(XE, float)[we], np.asarray(XU, float)[wu]])
        RY = np.concatenate([np.asarray(YE)[we], np.asarray(YU)[wu].astype(np.asarray(YE).dtype)])
        RG = np.concatenate([GE[we], GU[wu]])
        pre = sorted(i for i in pre_sessions if np.any(RG == i))
        ff = FrozenFrame(RX, RY, RG, pre, pipe, loso=True)
        XEp, XUp = np.array(XE, float), np.array(XU, float)
        for i in np.unique(RG):
            if i in set(pre_sessions) and i not in ff.loso:
                continue
            which = "loso" if i in ff.loso else "full"
            XEp[GE == i] = ff.apply(np.asarray(XE, float)[GE == i], i, which)
            if len(GU) and np.any(GU == i):
                XUp[GU == i] = ff.apply(np.asarray(XU, float)[GU == i], i, which)
        return XEp, XUp
    if projection_centre() == "working":
        raise ValueError("WIDEFIELD_PROJECTION_CENTRE=working needs the pool's working-row masks (work=, YU=); this "
                         "caller has none -- pass them or use the engaged centre")
    ff = FrozenFrame(XE, YE, GE, sorted(pre_sessions), pipe, loso=True)
    XEp, XUp = np.array(XE, float), np.array(XU, float)
    GE, GU = np.asarray(GE), np.asarray(GU)
    for i in np.unique(GE):
        which = "loso" if i in ff.loso else "full"
        XEp[GE == i] = ff.apply(XE[GE == i], i, which)
        if len(GU) and np.any(GU == i):
            XUp[GU == i] = ff.apply(XU[GU == i], i, which)
    return XEp, XUp
