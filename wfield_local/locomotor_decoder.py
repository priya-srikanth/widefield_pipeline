"""The FROZEN pre-stroke behavioural-state decoder: does cortex still report quiet/running/licking?

THE SPECIFICITY CONTROL, and it has to be the FROZEN arm rather than the within-session one.

Priya, 2026-09-11, asked for "evidence that not *all* decoding/encoding degrades post-stroke". The
obvious version of that -- refit a state decoder inside each session and show it stays high -- does
not work, and the reason is measured rather than suspected: within session, binary running-vs-quiet
decodes at AUROC 0.99-1.00 and the three-way problem at macro-AUROC 0.98-1.00. That is a CEILING,
and a ceiling cannot demonstrate preservation; a reviewer reads it as "the task was too easy to
fail" and is right to.

The position claim rests on a FROZEN PRE-STROKE MODEL failing on post-stroke data. So the control
has to be the same object: freeze a pre-stroke state decoder, apply it to each post-stroke session,
and put its accuracy on the same axis as the frozen position decoder's. Cross-session generalisation
is the hard part of the position result, and the state arm has to survive exactly that.

    frozen POSITION decoder, by epoch    0.886 pre -> 0.523 acute  (collapses)
    frozen STATE decoder, by epoch       <- if this holds, the deficit is specific

WHY IT IS CREDIBLE THAT IT HOLDS, and why that is not circular: the lesion is ventrolateral
STRIATAL. No cortex is damaged, the window is the same, the basis is the same, and the animals run
MORE acutely than before (5.1% of session against 3.1% pre-stroke), so the state arm is not being
rescued by having more data at baseline than after.

BALANCED ACCURACY AND MACRO AUROC, never raw accuracy. The classes are wildly unbalanced within a
session -- PS94 8/06 holds 12 running segments against 324 quiet -- and the balance CHANGES across
epochs (quiet runs 3.4% of a pre-stroke session, 15.1% acutely, 0.7% chronically). Raw accuracy
under a shifting base rate is not comparable across the epochs this figure exists to compare.

THE BOOTSTRAP MUST CLUSTER BY PERIOD. Segments from one bout are not independent observations; the
same rule the position families apply to trials inside a scheduler block.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from wfield_local import locomotor_state as ls

#: Fewest segments of EVERY class a session needs before it is scored at all. A session that holds
#: three quiet segments can produce a balanced accuracy, and it would be noise with a number on it.
MIN_PER_CLASS = 15


#: Per-session z-scoring of the segment features (2026-11-02). `segment_features` has always said the caller applies
#: the session-level z-score "exactly as for trials" (`locanmf_frozen_decoder.pool_sessions`), and the frozen spec
#: recorded zscore=True, but `by_animal_day` never applied it: every session's raw joint-basis features went straight
#: into a scaler fitted on pooled PRE-STROKE data, so any session-wide gain / offset change moved all segments toward
#: one class together. Measured (labcams/state_decoder_ps94_check/): z-scoring lifts PS94 chronic balanced accuracy
#: .64 -> .78 and leaves PS92 / PS93 / PS95 within ~.02; a within-session refit stays .97-.99 in every PS94 session,
#: so the rest of PS94's loss is a changed cortical LICKING pattern (licking read as quiet .49 chronic), not scaling
#: and not the state definitions. Set WIDEFIELD_STATE_ZSCORE=0 to reproduce the pre-fix (unscaled) run.
STATE_ZSCORE_ENV = "WIDEFIELD_STATE_ZSCORE"

#: WHICH per-session scaling the state decoder uses (2026-11-03). Priya, 11-03: "ok bal_em it is" -- the PRODUCTION
#: frozen state decoder is now ``session_bal_em`` (DECISIONS 2026-11-03, "bal_em adopted as the production state
#: decoder"). ``session_z`` (the 11-02 plain per-session z-score) stays selectable and its stored models stay in the
#: store untouched: ``WIDEFIELD_STATE_SCALING=session_z`` reproduces it, ``WIDEFIELD_STATE_ZSCORE=0`` (or
#: ``WIDEFIELD_STATE_SCALING=none``) the pre-11-02 unscaled run.
STATE_SCALING_ENV = "WIDEFIELD_STATE_SCALING"
SCALING_SESSION_Z = "session_z"
SCALING_BAL_EM = "session_bal_em"
DEFAULT_SCALING = SCALING_BAL_EM

#: bal_em outer passes (posteriors -> EM priors -> soft class means -> re-centre -> re-predict). 3, as the diagnostic
#: (labcams/state_decoder_ps94_check/prebal/analyse_prebal.py); 6 passes moved no per-animal number by > .02.
BAL_EM_PASSES = 3
#: OPTIONAL minimum-mass guard, DEFAULT OFF (None) -- Priya has not asked for one. When set to a fraction (e.g. 0.05),
#: a pass whose EM-estimated class share falls below it centres on the TRAINING class mix instead of equal weights
#: (equal weighting amplifies a nearly absent class: PS92 chronic running ~3 %, PS95 acute ~1 %). Setting it changes
#: the spec id (`bal_em_params` -> spec ``scaling_params``, which records min_mass even when OFF), so a guarded model
#: can never be read as an unguarded one.
BAL_EM_MIN_MASS = None


def session_zscore_on() -> bool:
    import os
    return os.environ.get(STATE_ZSCORE_ENV, "1").strip().lower() not in ("0", "false", "no", "off")


def state_scaling():
    """The scaling the state decoder runs under: ``"session_bal_em"`` (default), ``"session_z"`` or None (unscaled).
    ``WIDEFIELD_STATE_ZSCORE=0`` still means unscaled whatever ``WIDEFIELD_STATE_SCALING`` says."""
    import os
    if not session_zscore_on():
        return None
    v = os.environ.get(STATE_SCALING_ENV, DEFAULT_SCALING).strip().lower()
    if v in ("session_bal_em", "bal_em", "balem"):
        return SCALING_BAL_EM
    if v in ("session_z", "z", "zscore"):
        return SCALING_SESSION_Z
    if v in ("none", "0", "off", "unscaled", "raw"):
        return None
    raise ValueError(f"{STATE_SCALING_ENV}={v!r}: expected session_bal_em, session_z or none")


def bal_em_params(min_mass="default", n_outer=None):
    """The spec ``scaling_params`` for bal_em: ``{"min_mass": ..., "n_outer": ...}``, min_mass recorded even when None
    (off). ``"default"`` = the module default `BAL_EM_MIN_MASS`."""
    return {"min_mass": BAL_EM_MIN_MASS if min_mass == "default" else min_mass,
            "n_outer": BAL_EM_PASSES if n_outer is None else int(n_outer)}


def session_zscore(X):
    """Each feature z-scored over THIS session's segments (mean / SD over all its classes), as `pool_sessions` does
    for trials; zero-SD features are left centred. Note the session's class mix enters the mean (as the trial mix
    does for `pool_sessions`)."""
    X = np.asarray(X, float)
    if not len(X):
        return X
    mu, sd = X.mean(0), X.std(0)
    return (X - mu) / np.where(sd > 0, sd, 1.0)


def _session_sd(X):
    sd = np.asarray(X, float).std(0)
    return np.where(sd > 0, sd, 1.0)


def balanced_zscore(X, y, classes=ls.THREE_WAY):
    """TRAINING-side centring for bal_em: centre at the EQUAL-weight mean of the session's class means (TRUE labels --
    pre-stroke training labels are allowed), divide by the production per-session SD (all segments). Only ever applied
    to the pre-stroke sessions a model is trained on; a scored session goes through `bal_em_centre`."""
    X = np.asarray(X, float)
    y = np.asarray(y)
    if not len(X):
        return X
    mu = np.mean([X[y == c].mean(0) for c in classes if (y == c).sum()], 0)
    return (X - mu) / _session_sd(X)


def class_prior(ys, classes=ls.THREE_WAY):
    """Class proportions of the pooled TRAINING labels (the frozen model's training prior, for the EM)."""
    y = np.concatenate([np.asarray(v) for v in ys]) if len(ys) else np.array([])
    return np.array([(y == c).mean() if len(y) else np.nan for c in classes], float)


def _proba_full(model, Z, K):
    P = model.predict_proba(np.asarray(Z))
    full = np.zeros((len(Z), K))
    for j, c in enumerate(model.classes_):
        full[:, int(c)] = P[:, j]
    return full


def saerens_em(P, pi_train, n_iter=200, tol=1e-8):
    """Label-shift EM (Saerens, Latinne & Decaestecker 2002): re-estimate a session's class priors from the model's
    posteriors ``P`` (n x K) alone. Returns (adjusted posteriors, estimated priors). No label enters."""
    pi_train = np.asarray(pi_train, float)
    pi = pi_train.copy()
    Q = P
    for _ in range(n_iter):
        Q = P * (pi / pi_train)
        Q = Q / Q.sum(1, keepdims=True)
        new = Q.mean(0)
        if np.abs(new - pi).max() < tol:
            pi = new
            break
        pi = new
    return Q, pi


def bal_em_centre(model, X, pi_train, *, n_outer=None, min_mass=None, return_info=False):
    """SCORING-side centring for bal_em -- LABEL-FREE (DECISIONS 2026-11-03). Takes no labels, by signature.

    Start from the production centring (session mean / session SD); then ``n_outer`` passes of: the frozen model's
    posteriors -> Saerens EM priors (training prior = ``pi_train``) -> SOFT class means of the RAW features from the
    adjusted posteriors -> centre at their EQUAL-weight mean (classes with no mass dropped), keeping production's
    per-session SD -> re-z-score. Returns the scaled features (and, with ``return_info``, the estimated priors, the
    centre and whether the optional min-mass guard fired). Exactly the ``bal_em`` arm of
    labcams/state_decoder_ps94_check/prebal/analyse_prebal.py (``prebal_centre(..., w_target=[1/3]*3)``).

    ``min_mass`` (default None = OFF): if set, a pass whose EM-estimated share of any class is below it centres on
    the training mix ``pi_train`` instead of equal weights."""
    n_outer = BAL_EM_PASSES if n_outer is None else int(n_outer)
    X = np.asarray(X, float)
    pi_train = np.asarray(pi_train, float)
    K = len(pi_train)
    sd = _session_sd(X)
    mu = X.mean(0)
    Z = (X - mu) / sd
    pi_hat, guard = pi_train, False
    for _ in range(n_outer):
        Q, pi_hat = saerens_em(_proba_full(model, Z, K), pi_train)
        mass = Q.sum(0)
        muc = np.array([(Q[:, c:c + 1] * X).sum(0) / mass[c] if mass[c] > 1e-9 else mu for c in range(K)])
        ok = mass > 1e-9
        tgt = np.full(K, 1.0 / K)
        if min_mass is not None and np.any(pi_hat < float(min_mass)):
            tgt, guard = pi_train, True
        w = tgt * ok / (tgt * ok).sum()
        mu = (w[:, None] * muc).sum(0)
        Z = (X - mu) / sd
    if return_info:
        return Z, {"pi_hat": pi_hat, "centre": mu, "guard": guard}
    return Z


def train_transform(X, y, scaling):
    """How a PRE-STROKE TRAINING session's features are scaled under ``scaling``."""
    if scaling is None:
        return np.asarray(X)
    if scaling == SCALING_SESSION_Z:
        return session_zscore(X)
    if str(scaling).startswith(SCALING_BAL_EM):
        return balanced_zscore(X, y)
    raise ValueError(f"unknown state scaling {scaling!r}")


def score_transform(model, X, scaling, pi_train=None, *, min_mass=None):
    """How a SCORED session's features are scaled under ``scaling``. Takes no labels: nothing about a scored
    session's true classes can reach its centre."""
    if scaling is None:
        return np.asarray(X)
    if scaling == SCALING_SESSION_Z:
        return session_zscore(X)
    if str(scaling).startswith(SCALING_BAL_EM):
        return bal_em_centre(model, X, pi_train, min_mass=min_mass)
    raise ValueError(f"unknown state scaling {scaling!r}")


def _pipe():
    """The same shape as the position decoder's: standardise, then multinomial logistic."""
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))


def fit_frozen(Xs, ys, classes=ls.THREE_WAY):
    """Fit one state decoder on POOLED pre-stroke segments. None if a class is missing."""
    if not len(ys):
        return None
    y = np.asarray(ys)
    if any((y == c).sum() < MIN_PER_CLASS for c in classes):
        return None
    return _pipe().fit(np.asarray(Xs), np.array([list(classes).index(v) for v in y]))


def frozen_state_models(animal: str, basis_id: str, got: dict, scaling="env", min_mass="default"):
    """``({"full": model, "loso": {pre_label: model}}, status)`` -- the PRE-STROKE state decoder as a FROZEN,
    STORED REFERENCE (`frozen_models.load_or_fit`), never refitted once stored.

    Priya, 2026-10-15: "establish in the code for future cohorts that we will make a frozen pre-stroke decoder
    that is cached/stored and reused" -- the same discipline as the position decoders (docs/FROZEN_MODELS.md, CLAUDE.md
    rule 10). Until now this decoder was refitted in memory on every render, so adding a pre-stroke session (PS92
    0812, 2026-10-15) silently moved the reference every post-stroke score was computed against. Now: the identity
    is animal + kind ``state`` + the joint-basis id + the feature count + the SORTED pre-stroke labels with each
    session's data signature; a changed pre-stroke set is reported as SPEC-CHANGED and frozen under a new id, the
    old one untouched. ``got`` = {label: (X, y, groups, day)}; pre-stroke = day <= 0. The LOSO models (each pre
    session scored by a model that never saw it) are part of the same frozen payload."""
    import numpy as np

    from wfield_local import frozen_models as fm
    if scaling == "env":
        scaling = state_scaling()
    pre = {lab: v for lab, v in got.items() if v[3] <= 0}
    n_feat = int(next(iter(pre.values()))[0].shape[1])
    # SCALING IS PART OF THE IDENTITY (2026-11-02): models fitted on per-session z-scored features carry
    # scaling="session_z"; the pre-fix models (unscaled features under a spec that already said zscore=True) keep
    # their old id and stay in the store as the record (WIDEFIELD_STATE_ZSCORE=0 reproduces them).
    # 2026-11-03: PRODUCTION = scaling="session_bal_em" (trained on class-balanced pre sessions; scored with the
    # label-free EM centre). A NEW id; the session_z and unscaled models stay in the store untouched.
    # ``got`` holds RAW features; the training transform is applied here, by scaling. The pooled order is the
    # order ``got`` arrived in (as before 11-03, so the session_z fit is the same fit).
    # The min-mass guard acts only at SCORING time, but it is part of the identity (Priya 11-03: the spec records
    # min_mass), so a guarded and an unguarded run can never share a stored reference.
    params = bal_em_params(min_mass) if scaling == SCALING_BAL_EM else None
    spec = fm.make_spec(animal, "state", align="segments", source=f"joint_{basis_id}", train_labels=list(pre),
                        basis_id=basis_id, n_features=n_feat, post_s=ls.SEGMENT_S, zscore=True,
                        train_data={lab: v[:3] for lab, v in sorted(pre.items())},
                        scaling=scaling, scaling_params=params)
    tx = {lab: train_transform(v[0], v[1], scaling) for lab, v in pre.items()}

    def fit():
        full = fit_frozen(np.vstack([tx[lab] for lab in pre]), np.concatenate([pre[lab][1] for lab in pre]))
        loso = {}
        for lab in sorted(pre):
            keep = [l2 for l2 in pre if l2 != lab]
            if keep:
                loso[lab] = fit_frozen(np.vstack([tx[l2] for l2 in keep]),
                                       np.concatenate([pre[l2][1] for l2 in keep]))
        out = {"full": full, "loso": loso}
        if scaling is not None and str(scaling).startswith(SCALING_BAL_EM):
            # THE EM'S TRAINING PRIOR travels with the model: the pooled pre-stroke class mix of the sessions that
            # trained it (for a LOSO model, never the held-out one). Training labels only.
            out["pi_train"] = {"full": class_prior([pre[lab][1] for lab in pre]),
                               "loso": {lab: class_prior([pre[l2][1] for l2 in pre if l2 != lab])
                                        for lab in sorted(pre)}}
        return out

    return fm.load_or_fit(spec, fit, meta={"classes": list(ls.THREE_WAY), "n_pre_sessions": len(pre)})


def score(model, X, y, classes=ls.THREE_WAY):
    """``{"balacc":…, "auroc":…, "n":…, "per_class":{…}}`` or None when a class is absent.

    PER-CLASS RECALL IS RETURNED ALONGSIDE THE SUMMARY, because a pooled balanced accuracy can hide
    the interesting failure: a state decoder that keeps quiet and running but loses LICKING would be
    reporting a change in the task-engaged state specifically, which is a different claim from
    "cortex still reports behavioural state" and should not be averaged into it.
    """
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score

    if model is None or not len(y):
        return None
    y = np.asarray(y)
    present = [c for c in classes if (y == c).sum()]
    if len(present) < 2:
        return None
    yi = np.array([list(classes).index(v) for v in y])
    P = model.predict_proba(np.asarray(X))
    full = np.zeros((len(yi), len(classes)))
    for j, c in enumerate(model.classes_):
        full[:, int(c)] = P[:, j]
    pred = full.argmax(1)
    # RAW COUNTS, so epochs can be POOLED BY SUMMING. The same reason `_class_confusions` stores
    # counts rather than rates: a mean of per-session rates and a rate computed on pooled counts are
    # different numbers, and only the second is what a confusion panel draws.
    K = len(classes)
    conf = np.zeros((K, K), float)
    for t, q in zip(yi, pred):
        conf[int(t), int(q)] += 1
    out = {"n": len(yi),
           "balacc": float(balanced_accuracy_score(yi, pred)),
           "per_class": {c: float((pred[y == c] == list(classes).index(c)).mean())
                         for c in present},
           "confusion": conf,
           "counts": {c: int((y == c).sum()) for c in classes}}
    try:
        # ONE-VS-REST OVER THE CLASSES ACTUALLY PRESENT. Passing all three to `roc_auc_score` when
        # a session holds two raises rather than returning a partial score, and a session missing a
        # class is common here -- chronic sessions run 0.7% quiet.
        idx = [list(classes).index(c) for c in present]
        sub = full[:, idx]
        sub = sub / np.clip(sub.sum(1, keepdims=True), 1e-12, None)
        out["auroc"] = float(roc_auc_score(
            np.array([present.index(v) for v in y]), sub if len(present) > 2 else sub[:, 1],
            multi_class="ovr", average="macro") if len(present) > 2 else
            roc_auc_score(np.array([present.index(v) for v in y]), sub[:, 1]))
    except Exception:                                                  # noqa: BLE001
        out["auroc"] = float("nan")
    out["chance_balacc"] = 1.0 / len(present)
    out["classes_present"] = list(present)
    return out


#: Imaging frame rate the segment windows are cut at. The corrected (paired) rate, matching what
#: `_trial_features` uses for the trial arms -- the two must agree or a "1 s window" is two lengths.
FS_IMG = 31.23


def collect_animal(an, fs_img=FS_IMG):
    """``(basis_id, got)`` for one animal: every pre + post session's RAW segment features on the joint basis,
    ``got = {label: (X_raw, y, groups, day)}``. The expensive half of `by_animal_day` (projection onto the basis);
    no scaling is applied here -- that is `score_animal`'s job, by scaling. None when the basis is unavailable."""
    from wfield_local import behavior_events as be
    from wfield_local import config, joint_locanmf
    from wfield_local import locomotor_features as lf
    from wfield_local.grant_figures import _day
    from wfield_local.locanmf_cue_lick_analysis import SESSIONS
    from wfield_local.paths import PathResolver

    rv = PathResolver()
    try:
        basis = joint_locanmf.load(an, sessions=SESSIONS)
    except Exception as ex:                                            # noqa: BLE001
        print(f"  !! state {an}: basis {type(ex).__name__} {str(ex)[:70]}", flush=True)
        return None
    want = {x for x in config.phase_labels("pre") + config.phase_labels("post") if x.startswith(an)}
    got = {}
    for s in [x for x in SESSIONS if x["label"] in want]:
        lab = s["label"]
        mmdd = lab.split("_")[-1]
        ev = be.get_or_compute(rv, an, config.date8(mmdd))
        if not ev:
            continue
        try:
            sig, _r = joint_locanmf.BasisSource(basis, s).signal()
            X, y, g = lf.segment_features(s, np.asarray(sig), ev, fs_img=fs_img)
        except Exception as ex:                                        # noqa: BLE001
            print(f"  !! state {lab}: {type(ex).__name__} {str(ex)[:70]}", flush=True)
            continue
        if not len(y):
            continue
        d = _day(an, mmdd)
        if d is None:
            continue
        got[lab] = (X, y, g, int(d))
    return basis.basis_id, got


def score_animal(an, basis_id, got, scaling="env", min_mass="default", models=None):
    """``(rec, status, models)``: score one animal's sessions with its FROZEN state decoder under ``scaling``.

    ``rec = {"PRE": [score, ...], day: score}``; PRE is leave-one-session-out, post-stroke days use the model frozen
    on ALL pre-stroke segments. Each scored session's features are scaled by `score_transform`, which never sees
    that session's labels (bal_em: label-free EM centre). ``models`` lets a caller pass an already-loaded payload."""
    if scaling == "env":
        scaling = state_scaling()
    if min_mass == "default":
        min_mass = BAL_EM_MIN_MASS
    status = "given"
    if models is None:
        models, status = frozen_state_models(an, basis_id, got, scaling=scaling, min_mass=min_mass)
    pis = models.get("pi_train") or {}
    rec = {"PRE": []}
    for lab, (X, y, _g, d) in sorted(got.items()):
        if d <= 0:
            m = models["loso"].get(lab)
            pi = (pis.get("loso") or {}).get(lab)
            if m is None:
                continue
        else:
            m = models["full"]
            pi = pis.get("full")
        if m is None:
            continue
        if scaling is not None and str(scaling).startswith(SCALING_BAL_EM) and pi is None:
            # A bal_em payload always carries its training prior; rebuild it from the TRAINING labels if not.
            tr = [l2 for l2, v in got.items() if v[3] <= 0 and (d > 0 or l2 != lab)]
            pi = class_prior([got[l2][1] for l2 in tr])
        Z = score_transform(m, X, scaling, pi, min_mass=min_mass)
        r = score(m, Z, y)
        if r is None:
            continue
        if d <= 0:
            rec["PRE"].append(r)
        else:
            rec[d] = r
    return rec, status, models


def by_animal_day(fs_img=FS_IMG):
    """``({animal: {"PRE": [score, ...], day: score}}, days)`` for the FROZEN state decoder, under the scaling the
    environment selects (`state_scaling`; production = ``session_bal_em`` since 2026-11-03).

    PRE IS LEAVE-ONE-SESSION-OUT and post-stroke days are scored by a model frozen on ALL pre-stroke
    segments, which is exactly the discipline `_collect_5c` uses for position. Without the LOSO the
    pre column would be a session scored partly against itself and the baseline would be too easy --
    the failure `_pre_reference` exists to prevent.

    CACHED (in process, keyed on the scaling too) BECAUSE IT PROJECTS EVERY SESSION ONTO THE JOINT BASIS, which is
    the expensive step (~10 min for the cohort) and is shared by every figure that reads this.
    """
    return _by_animal_day(fs_img, state_scaling(), BAL_EM_MIN_MASS)


@lru_cache(maxsize=4)
def _by_animal_day(fs_img, scaling, min_mass):
    from wfield_local.grant_figures import ANIMALS

    out, all_days = {}, set()
    for an in ANIMALS:
        c = collect_animal(an, fs_img)
        if c is None:
            continue
        basis_id, got = c
        if not any(v[3] <= 0 for v in got.values()):
            continue
        rec, status, models = score_animal(an, basis_id, got, scaling=scaling, min_mass=min_mass)
        print(f"  state {an}: {status} ({len(models['loso'])} pre-stroke sessions; scaling={scaling})", flush=True)
        all_days.update(d for d in rec if d != "PRE")
        if len(rec) > 1 and rec["PRE"]:
            out[an] = rec
    return out, sorted(all_days)
