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


#: WHICH ROWS THE EM CENTRE IS FITTED ON (Priya 11-03, "camera gate everywhere"): ``working`` = the session's camera
#: WORKING rows (contact | incomplete inside the working period -- Priya 11-03 late, jaw_only is its own class: never
#: jaw_only, after_end, unknown, stopped, early_run), the default under the camera trial gate; ``engaged`` = every DAQ-engaged row of the session (main's
#: 11-03 rule), the default under the DAQ gate and the alternative under the camera gate. Label-free either way, never
#: no-lick rows. Setting ``bal_em.centre_rows`` (auto | working | engaged) / WIDEFIELD_BALEM_CENTRE. It enters the
#: stored-model spec (``scaling_params.centre_set``); a camera run with the engaged centre is tagged (`run_mode`).
CENTRE_ENV = "WIDEFIELD_BALEM_CENTRE"
CENTRE_ROWS = ("working", "engaged")
#: a session's working-row centre needs at least this many camera-working engaged rows (every curated session except
#: the no-video PS93_0813 has >= 20; fifth review D-B3)
MIN_CENTRE_ROWS = 20


class BalEmCentreRefused(SystemExit):
    """A bal_em centre that cannot be computed by the run's rule (fifth review D-B2): a SystemExit, so no generic
    ``except Exception`` can swallow it into a silently ungated or engaged-centred result."""


def centre_fallback() -> dict:
    """{session label: "contact"} -- the sessions whose bal_em centre comes from their DAQ CONTACT (lick-class) rows
    because they have no camera-working rows (no video; ``bal_em.centre_fallback_contact``, default PS93_0813).
    Priya's rule keeps such a session in lick-aligned, contact-only selections, so its centre is fitted on exactly the
    rows those selections score (never no-lick rows). Empty unless the centre is the camera working set."""
    if centre_rows() != "working":
        return {}
    try:
        from wfield_local import config
        labs = (config.defaults().get("bal_em") or {}).get("centre_fallback_contact", ["PS93_0813"])
    except ImportError:
        labs = ["PS93_0813"]
    return {str(x): "contact" for x in (labs or [])}


def centre_rows() -> str:
    """"working" or "engaged" for this run (see CENTRE_ROWS); "auto" = working under the camera gate, else engaged."""
    v = os.environ.get(CENTRE_ENV, "").strip().lower()
    if not v:
        try:
            from wfield_local import config
            v = str((config.defaults().get("bal_em") or {}).get("centre_rows", "auto")).lower()
        except ImportError:
            v = "auto"
    from wfield_local import trial_classes
    cam = trial_classes.camera_gate_on()
    if v == "auto":
        return "working" if cam else "engaged"
    if v not in CENTRE_ROWS:
        raise BalEmCentreRefused(f"bal_em.centre_rows / {CENTRE_ENV} = {v!r}; expected auto, working or engaged")
    if v == "working" and not cam:
        raise BalEmCentreRefused("bal_em.centre_rows = working needs the camera trial gate (WIDEFIELD_TRIAL_GATE=camera)")
    return v


def centre_mask(cls_e):
    """``bool[n engaged rows]`` -- the rows the EM centre is fitted on: the camera WORKING rows (contact | incomplete;
    ``cls_e`` = the camera
    class per engaged row, `trial_classes.pool_row_classes`) under ``centre_rows() == "working"``; None (= every
    engaged row) under "engaged". FAILS CLOSED: "working" without row classes raises."""
    if centre_rows() != "working":
        return None
    if cls_e is None:
        raise BalEmCentreRefused("bal_em centre: the camera WORKING rows are required (bal_em.centre_rows = working) "
                                 "but the caller passed no camera classes for its engaged rows")
    from wfield_local import trial_classes
    return trial_classes.in_sel(cls_e, "working")


#: sessions whose centre came from the contact-row fallback in this process (for result sidecars / captions)
FALLBACK_USED: set = set()


def _centre_rows_of(XE, me, cm, label=None):
    """Engaged rows of one session the centre uses: ``me & cm`` (``cm`` None = all of ``me``). Under the working centre
    a session needs >= MIN_CENTRE_ROWS working rows; one below that is REFUSED (BalEmCentreRefused) unless it is a
    configured contact fallback (`centre_fallback`: no video, e.g. PS93_0813), whose centre then comes from its DAQ
    contact (lick-class) rows -- the engaged rows here -- with a log line naming it."""
    if cm is None:
        return XE[me]
    sel = me & np.asarray(cm, bool)
    if sel.sum() >= MIN_CENTRE_ROWS:
        return XE[sel]
    if label is not None and str(label) in centre_fallback():
        FALLBACK_USED.add(str(label))
        print(f"  [bal_em] {label}: {int(sel.sum())} camera-working rows -> centre from its {int(me.sum())} DAQ contact "
              f"(lick-class) rows (bal_em.centre_fallback_contact; recorded as centre_set_fallback)", flush=True)
        return XE[me]
    raise BalEmCentreRefused(f"bal_em centre: session {label or '?'} has {int(sel.sum())} camera-working engaged rows "
                             f"(< {MIN_CENTRE_ROWS}) and is not a configured contact fallback")


#: The engagement-STATE discriminators (lick vs no-lick: section G looks_like_which / fits_engaged_distribution /
#: undetected_state_split, precue_engagement_states C / D) can be scored with bal_em over the two STATE classes, but it is
#: OFF by default: measured 11-03 (labcams/decoder_mix_checks/spout_balem_production/summarise_extension_log.txt) it
#: breaks the discriminators' own pre-stroke reference -- held-out PRE engaged trials read engaged-like .92 -> .64 (cue)
#: and .80 -> .43 (precue) -- because these discriminators separate the states only weakly (AUC ~.6-.8), so the EM
#: cannot recover a session's state mix and the centre lands between the states. Needs Priya's decision; selectable
#: with WIDEFIELD_STATE_DISC_SCALING=state_bal_em.
STATE_DISC_SCALING_ENV = "WIDEFIELD_STATE_DISC_SCALING"
SCALING_STATE_BAL_EM = "state_bal_em"


def state_disc_scaling():
    """``"state_bal_em"`` when opted in, else None (the production engaged-trial z-score, default)."""
    v = os.environ.get(STATE_DISC_SCALING_ENV, "session_z").strip().lower()
    if v in ("state_bal_em", "bal_em"):
        return SCALING_STATE_BAL_EM
    if v in ("session_z", "z", "none", "off"):
        return None
    raise ValueError(f"{STATE_DISC_SCALING_ENV}={v!r}: expected state_bal_em or session_z")


def scaling_params():
    """The spec ``scaling_params`` for the spout decoder's bal_em. Everything that defines the transform is recorded,
    including the guard that is OFF."""
    out = {"centre_set": centre_rows(), "guard": None, "min_mass": None, "n_outer": BAL_EM_PASSES,
           "train_centre": "true_class_means_equal_weight"}
    fb = centre_fallback()
    if fb:                                           # the explicit fallback enters the spec id (fifth review D-B3)
        out["centre_set_fallback"] = dict(sorted(fb.items()))
        out["centre_min_rows"] = MIN_CENTRE_ROWS
    return out


def spec_kw(scaling="env", trials=None):
    """``{"scaling": ..., "scaling_params": ...}`` for `frozen_models.make_spec`; EMPTY under session_z with the
    engaged training set, so that spec (and its id) is byte-identical to the pre-11-03 one. ``trials`` = the training
    trial selection when it is NOT the engaged set (the camera WORKING trials, engaged AND no-lick rows: the run's
    `trial_classes.run_selection`; as main 48f2cc6d `spec_kw(scaling, trials)`): recorded as
    ``scaling_params["trials"]`` with ``centre_set`` "working_rows", so a working-trained model never shares an id with
    an engaged one."""
    if scaling == "env":
        scaling = spout_scaling()
    if scaling is None:
        return {} if trials is None else {"scaling_params": {"trials": trials}}
    params = scaling_params()
    if trials is not None:
        params["trials"] = trials
        params["centre_set"] = "working_rows"        # engaged + no-lick camera working rows (coordinator 11-04)
        params.pop("centre_set_fallback", None)      # no contact fallback for a working-trained model: no rows = unscored
        params["centre_min_rows"] = MIN_CENTRE_ROWS
    return {"scaling": scaling, "scaling_params": params}


#: sessions left OUT of a working-trained model in this process because they have no camera working rows (no video:
#: PS93_0813), for result sidecars / captions (review 7A NB3)
WORKING_EXCLUDED: set = set()


def is_post_session(label) -> bool:
    """True for a POST-stroke session by its PHASE (`config.session_phase`), never by membership of a caller's (possibly
    narrowed) pre list (review 8A B1): a pre session with 0 working rows is excluded and logged; only a POST one raises."""
    from wfield_local import config
    lab = str(label)
    return config.session_phase(config.animal_of(lab), lab.split("_")[-1]) == "post"


def note_working_excluded(label, where=""):
    """Record (and log once) a session excluded from the working selection: no camera working rows."""
    lab = str(label)
    if lab not in WORKING_EXCLUDED:
        print(f"  [working] {lab}: no camera working rows (no video) -> EXCLUDED from the working selection "
              f"(not trained on, not centred, not scored){' -- ' + where if where else ''}", flush=True)
    WORKING_EXCLUDED.add(lab)


def working_centre(model, C, pi_train, label=None):
    """``(mu, sd)``: the label-free EM centre of ONE session from its camera WORKING rows ``C`` (engaged AND no-lick
    rows of class contact | incomplete, main 48f2cc6d / coordinator 11-04), or None when it has none (the session is
    then not scored by a working-trained model: no video, PS93_0813). 1..MIN_CENTRE_ROWS-1 rows RAISE."""
    C = np.asarray(C, float)
    if not len(C):
        return None
    if len(C) < MIN_CENTRE_ROWS:
        raise BalEmCentreRefused(f"bal_em centre: session {label or '?'} has {len(C)} camera-working rows "
                                 f"(< {MIN_CENTRE_ROWS})")
    return score_centre(model, C, pi_train)


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

    def __init__(self, XE, YE, GE, train_sessions, pipe, *, classes=None, train_mask=None, loso=True, scaling="env",
                 store=None, cls_e=None, labels=None):
        self._lab = ({int(k): str(v) for k, v in (labels.items() if hasattr(labels, "items") else enumerate(labels))}
                     if labels is not None else {})
        self.scaling = spout_scaling() if scaling == "env" else scaling
        # the rows each scored session's EM centre is fitted on (`centre_mask`; camera WORKING rows under the camera gate)
        self.cm = centre_mask(cls_e) if self.scaling else None
        self.XE, self.YE, self.GE = np.asarray(XE, float), np.asarray(YE), np.asarray(GE)
        self.train_sessions = sorted(train_sessions)
        tm = np.isin(self.GE, self.train_sessions)
        if train_mask is not None:
            tm &= np.asarray(train_mask, bool)
        self.train_mask = tm
        if classes is None:
            classes = sorted(int(c) for c in np.unique(self.YE[tm]))
        self.classes = list(classes)
        if self.scaling:
            self.XT = self.XE.copy()
            for i in self.train_sessions:
                m = self.GE == i
                rows = m & tm
                if rows.any():
                    c = label_shift.balanced_centre(self.XE[rows], self.YE[rows], self.classes)
                    self.XT[m] = (self.XE[m] - c) / label_shift.session_sd(self.XE[m])
        else:
            self.XT = self.XE
        def _fit_all():
            out = {"full": pipe().fit(self.XT[tm], self.YE[tm]), "loso": {}}
            if loso:
                for i in self.train_sessions:
                    tr = tm & (self.GE != i)
                    if tr.any() and len(np.unique(self.YE[tr])) >= 2:
                        out["loso"][i] = pipe().fit(self.XT[tr], self.YE[tr])
            return out

        # STORE HOOK (rebuild x main merge, 11-03; CLAUDE.md rule 10): ``store(fit_all)`` returns the frozen store's
        # payload -- fitted by ``fit_all`` on THIS training frame on a miss -- so a caller keeps its stored,
        # fingerprinted models; None = the in-memory fits (main's behaviour). THE STORED LOSO IS KEYED BY SESSION
        # LABEL (merge review B1): pooled ids differ between pools (the LP pool lacks PS93_0813; a non-pre session
        # entering a pool shifts ranks), so ``labels`` (id -> label) is REQUIRED with a store; ids are mapped to
        # labels before storing and back after loading, and a stored label absent from this pool is ignored.
        if store is not None:
            if labels is None:
                raise ValueError("FrozenFrame(store=...) needs labels= (pooled id -> session label): stored LOSO "
                                 "models are keyed by label, never by pooled index")
            lab_of = {int(k): str(v) for k, v in (labels.items() if hasattr(labels, "items") else enumerate(labels))}
            id_of = {v: k for k, v in lab_of.items()}

            def _fit_labelled():
                got = _fit_all()
                return {"full": got["full"], "loso": {lab_of[int(i)]: m for i, m in got["loso"].items()}}

            raw = store(_fit_labelled)
            pay = {"full": raw["full"], "loso": {id_of[k]: m for k, m in (raw.get("loso") or {}).items() if k in id_of}}
        else:
            pay = _fit_all()
        self.full = pay["full"]
        self.pi = {"full": model_prior(self.full, self.YE[tm])}
        self.loso = {}
        for i, m in (pay.get("loso") or {}).items():
            self.loso[i] = m
            self.pi[("loso", i)] = model_prior(m, self.YE[tm & (self.GE != i)])
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
                self._c[key] = score_centre(self.model(which, i),
                                            _centre_rows_of(self.XE, self.GE == i, self.cm, self._lab.get(int(i))), pi)
        return self._c[key]

    def apply(self, X, i, which="full"):
        """Rows of session ``i`` (production frame) in the scoring frame of model ``which``. No labels."""
        if not self.scaling:
            return X
        mu, sd = self.centre(i, which)
        return (np.asarray(X, float) - mu) / sd

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
    if not sc:
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
    if not sc:
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


def score_session(model, Xe, Xu, pi_train, Xc=None):
    """``(Ze, Zu)``: one session's engaged and no-lick rows in the bal_em scoring frame (centre + SD from ``Xc``, the
    centre rows -- default the engaged rows -- applied to both). No labels."""
    mu, sd = score_centre(model, Xe if Xc is None else Xc, pi_train)
    Ze = (np.asarray(Xe, float) - mu) / sd
    Zu = (np.asarray(Xu, float) - mu) / sd if Xu is not None and len(Xu) else (
        np.zeros((0, np.asarray(Xe).shape[1])) if Xu is not None else None)
    return Ze, Zu


def scored_frame(XE, GE, XU, GU, sessions, model_of, prior_of, centre_mask=None, session_names=None):
    """Copies of the engaged / no-lick matrices with each session in ``sessions`` moved into the bal_em scoring frame,
    centred with ITS scoring model (``model_of(i)``: the LOSO model for a held-out pre session, the full model for a
    post session) and that model's training prior (``prior_of(i)``). Sessions without a model are left as they are.
    ``centre_mask`` = `centre_mask(cls_e)` (the camera working rows); None is refused under centre_rows "working"."""
    if centre_mask is None and centre_rows() == "working":
        raise BalEmCentreRefused("bal_em scored_frame: camera WORKING centre rows required (bal_em.centre_rows = "
                                 "working)")
    # session NAMES (pooled id -> label), for the centre fallback; never trial class labels
    lab = ({int(k): str(v) for k, v in (session_names.items() if hasattr(session_names, "items")
                                         else enumerate(session_names))}
           if session_names is not None else {})
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
        Ze, Zu = score_session(mdl, XE[me], XU[mu_] if mu_.any() else None, prior_of(i),
                               Xc=_centre_rows_of(XE, me, centre_mask, lab.get(int(i))))
        XEs[me] = Ze
        if mu_.any():
            XUs[mu_] = Zu
    return XEs, XUs
