"""Label-free class-balanced centring ("bal_em") for a FROZEN decoder: the shared primitives.

ONE definition, used by both frozen decoders that adopted it (CLAUDE.md rule 9):

  * the behavioural-STATE decoder (`locomotor_decoder`, production since 2026-11-03, "ok bal_em it is");
  * the SPOUT-POSITION decoder (`spout_scaling`, production since 2026-11-03, "I would adopt B everywhere").

WHY. A frozen decoder scores each session after a per-session normalisation (mean / SD of that session's samples).
The session mean depends on the session's CLASS MIX, so when a class becomes rare post-stroke (far contra on the
engaged set, running in PS94) the centre moves toward the remaining classes and every sample is pushed toward the
rare class's pattern. bal_em replaces that mean by the EQUAL-weight mean of the session's class means, estimated
WITHOUT labels: the frozen model's posteriors -> Saerens, Latinne & Decaestecker (2002) EM re-estimates the session's
class priors -> soft class means from the prior-adjusted posteriors -> equal-weight centre (production SD kept) ->
re-predict; ``n_outer`` passes. DECISIONS 2026-11-03 (state: "bal_em adopted"; spout: "spout-position bal_em
adopted").

Moved here verbatim from `locomotor_decoder` (2026-11-03) so the spout decoder reuses the same code rather than a
copy; `locomotor_decoder` re-exports every name, and `tests/test_state_bal_em.py` pins that the state decoder is
bit-identical.
"""
from __future__ import annotations

import numpy as np

#: Outer passes (posteriors -> EM priors -> soft class means -> re-centre -> re-predict). 3, as both diagnostics.
BAL_EM_PASSES = 3


def session_sd(X):
    """Per-feature SD over the session's rows; zero-SD features get 1 (left centred, as `pool_sessions` does)."""
    sd = np.asarray(X, float).std(0)
    return np.where(sd > 0, sd, 1.0)


def proba_full(model, Z, K):
    """``model.predict_proba`` spread into K columns indexed by the integer class code (a class the model never saw
    gets probability 0)."""
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


def bal_em_centre(model, X, pi_train, *, n_outer=None, min_mass=None, return_info=False, mu0=None, sd=None):
    """SCORING-side centring for bal_em -- LABEL-FREE (DECISIONS 2026-11-03). Takes no labels, by signature.

    Start from the production centring (session mean / session SD); then ``n_outer`` passes of: the frozen model's
    posteriors -> Saerens EM priors (training prior = ``pi_train``) -> SOFT class means of the RAW features from the
    adjusted posteriors -> centre at their EQUAL-weight mean (classes with no mass dropped), keeping production's
    per-session SD -> re-z-score. Returns the scaled features (and, with ``return_info``, the estimated priors, the
    centre, the SD and whether the optional min-mass guard fired). Exactly the ``bal_em`` arm of
    labcams/state_decoder_ps94_check/prebal/analyse_prebal.py (``prebal_centre(..., w_target=[1/3]*3)``) and arm B of
    labcams/decoder_mix_checks/spout_guard/analyse_guard.py.

    ``min_mass`` (default None = OFF): if set, a pass whose EM-estimated share of any class is below it centres on
    the training mix ``pi_train`` instead of equal weights.

    ``mu0`` / ``sd`` (default None = the session mean / SD of ``X``, the behaviour above): a fixed starting centre and a
    fixed SD, for a caller whose production normalisation comes from a DIFFERENT row set than the one the classes are
    balanced over (the engagement-state discriminators: production frame = the engaged trials, state classes = all
    trials; `spout_scaling.state_centre`)."""
    n_outer = BAL_EM_PASSES if n_outer is None else int(n_outer)
    X = np.asarray(X, float)
    pi_train = np.asarray(pi_train, float)
    K = len(pi_train)
    sd = session_sd(X) if sd is None else np.asarray(sd, float)
    mu = X.mean(0) if mu0 is None else np.asarray(mu0, float)
    Z = (X - mu) / sd
    pi_hat, guard = pi_train, False
    for _ in range(n_outer):
        Q, pi_hat = saerens_em(proba_full(model, Z, K), pi_train)
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
        return Z, {"pi_hat": pi_hat, "centre": mu, "sd": sd, "guard": guard}
    return Z


def balanced_centre(X, y, classes):
    """TRAINING-side centre: the EQUAL-weight mean of the session's TRUE class means (classes absent from the session
    dropped). Training labels only -- a SCORED session goes through `bal_em_centre`."""
    X = np.asarray(X, float)
    y = np.asarray(y)
    return np.mean([X[y == c].mean(0) for c in classes if (y == c).sum()], 0)


def class_prior(ys, classes):
    """Class proportions of the pooled TRAINING labels (the frozen model's training prior, for the EM)."""
    y = np.concatenate([np.asarray(v) for v in ys]) if len(ys) else np.array([])
    return np.array([(y == c).mean() if len(y) else np.nan for c in classes], float)
