"""GLM-HMM ENGAGEMENT STATES from trial outcomes, after Ashwood et al. 2022 / Yin, Melin, ... Churchland (Neuron 2025)
(Priya 11-16: "our 'states' if we are defining them as GLM-HMM as in the Churchland paper, should come from ALL trials,
not just engaged or working trials"; design choices 11-16: 4 outcome classes, FROZEN pooled pre-stroke fit primary).

    python -m wfield_local.glm_hmm --select-k              # held-out-session likelihood for K = 1..4 (pre sessions)
    python -m wfield_local.glm_hmm --k 3                   # frozen pre fit + every session's per-trial state posteriors

MODEL. A hidden Markov chain over trials (K states, transition matrix A, initial pi); in state k the trial outcome y in
{contact, incomplete, jaw_only, no_response} (the camera trial classes, `trial_classes.session_trials`) is a
MULTINOMIAL LOGISTIC function of the trial's inputs x with state-specific weights W_k (contact = reference class):
    P(y = c | x, z = k) = softmax_c(W_k x),   x = [animal one-hot (per-animal bias), spout dummies (5),
                                                  previous outcome dummies (3), spout moved since the last trial].
No session-time input on purpose: a drift term would stand in for the states. Trials of class "unknown" have no
observation (their emission likelihood is 1); their state is still inferred from the chain.

DATA = ALL TRIALS of every session (not engaged / working trials: the disengaged stretches are what the model must
find). Each session is one sequence.

FIT. EM: forward-backward per session (scaled), transitions from expected counts + a sticky Dirichlet prior
(STICKY on the diagonal, 1 elsewhere), emissions by weighted multinomial logistic regression with an L2 prior
(L2_PRIOR) per state (L-BFGS). N_RESTARTS random initialisations; the best final log-likelihood is kept.

FRAMES (as every other readout here):
  frozen  (PRIMARY) one model pooled over the four animals' AUGUST pre sessions (`is_august_pre`), per-animal bias.
          Pre sessions are scored LEAVE-ONE-SESSION-OUT (refit without that session, warm-started from the full fit);
          every other session is scored by the full pre model unchanged.
  refit   one model per animal over ALL its curated sessions (comparison: states may then absorb post-stroke
          behaviour).
States are named after fitting by their P(contact) at the average input (state 0 = most engaged); the label map is
saved with the outputs.

Output per trial: label, trial_id, cls, position, frame, K, p_state0..p_state{K-1}, state (argmax if its posterior
> STATE_P, else -1 = unassigned, the paper's 0.8 rule), loglik.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

POSITIONS = ("close_L", "close_R", "close_center", "far_L", "far_R", "far_center")
STICKY = 10.0
L2_PRIOR = 1.0
N_RESTARTS = 5
MAX_ITER = 200
TOL = 1e-5
STATE_P = 0.8
#: number of leading X columns whose weights are SHARED across states (design v2: the animal offsets); 0 = v1 (every
#: column state-specific). Set by `apply_design`.
N_SHARED = 0


# ------------------------------------------------------------------------------------------------------------ model ----
def _log_emission_one(Wj, X, y, Bj=None, ns: int = 0):
    """[T, K] log P(y_t | x_t, z = k) of ONE categorical output; Wj [K, C-1, D - ns] on the state-specific columns
    X[:, ns:]; Bj [C-1, ns] = weights SHARED by all states on X[:, :ns]; y = -1 -> 0 (no observation)."""
    K, T = Wj.shape[0], X.shape[0]
    out = np.zeros((T, K))
    obs = np.flatnonzero(y >= 0)
    off = X[:, :ns] @ Bj.T if ns else 0.0
    for k in range(K):
        eta = np.concatenate([np.zeros((T, 1)), X[:, ns:] @ Wj[k].T + off], axis=1)
        lp = eta - logsumexp(eta, axis=1, keepdims=True)
        out[obs, k] = lp[obs, y[obs]]
    return out


def _log_emission(W, X, Y, B=None, ns: int = 0):
    """Sum over the (conditionally independent given the state) outputs: W = list of [K, C_j-1, D-ns], Y [T, J]."""
    return sum(_log_emission_one(Wj, X, Y[:, j], B[j] if ns else None, ns) for j, Wj in enumerate(W))


def _forward_backward(logpi, logA, logB):
    """Posteriors gamma [T, K], expected transitions xi_sum [K, K], log-likelihood. SCALED recursion in probability
    space (Rabiner): each step's emissions are divided by their row max (added back to the log-likelihood), so nothing
    under- / overflows; ~50x faster than a per-step logsumexp (measured on the synthetic test)."""
    T, K = logB.shape
    mx = logB.max(1, keepdims=True)
    B = np.exp(logB - mx)
    A, pi = np.exp(logA), np.exp(logpi)
    alpha = np.empty((T, K))
    c = np.empty(T)
    a = pi * B[0]
    c[0] = a.sum()
    alpha[0] = a / c[0]
    for t in range(1, T):
        a = (alpha[t - 1] @ A) * B[t]
        c[t] = a.sum()
        alpha[t] = a / c[t]
    beta = np.empty((T, K))
    beta[-1] = 1.0
    for t in range(T - 2, -1, -1):
        beta[t] = (A @ (B[t + 1] * beta[t + 1])) / c[t + 1]
    gamma = alpha * beta
    gamma /= gamma.sum(1, keepdims=True)
    if T > 1:
        xi_sum = A * (alpha[:-1].T @ (B[1:] * beta[1:] / c[1:, None]))
    else:
        xi_sum = np.zeros((K, K))
    return gamma, xi_sum, float(np.log(c).sum() + mx.sum())


def _fit_emission(W0, X, y, w, n_cls, offset=None):
    """Weighted multinomial logistic regression (reference class 0) with an L2 prior; W [C-1, D]; ``offset`` [T, C-1]
    = a fixed linear term (the shared-weight part in design v2)."""
    obs = y >= 0
    X, y, w = X[obs], y[obs], w[obs]
    off = 0.0 if offset is None else offset[obs]
    D = X.shape[1]
    Y = np.eye(n_cls)[y]

    def f(v):
        Wk = v.reshape(n_cls - 1, D)
        eta = np.concatenate([np.zeros((len(X), 1)), X @ Wk.T + off], axis=1)
        lse = logsumexp(eta, axis=1)
        nll = -(w * ((Y * eta).sum(1) - lse)).sum() + 0.5 * (v ** 2).sum() / L2_PRIOR
        P = np.exp(eta - lse[:, None])
        g = -((w[:, None] * (Y - P))[:, 1:]).T @ X + Wk / L2_PRIOR
        return nll, g.ravel()
    r = minimize(f, W0.ravel(), jac=True, method="L-BFGS-B")
    return r.x.reshape(n_cls - 1, D)


class GLMHMM:
    """K-state input-output HMM: every output j is a state-specific multinomial logistic function of the inputs."""

    def __init__(self, K: int, n_cls=None, seed: int = 0, n_shared: int | None = None):
        self.K, self.C, self.rng = K, list(n_cls if n_cls is not None else n_classes()), np.random.default_rng(seed)
        self.ns = N_SHARED if n_shared is None else n_shared
        self.W = self.pi = self.A = None
        self.B = None

    def _init(self, W1):
        ns = self.ns
        self.B = [w[:, :ns].copy() for w in W1] if ns else None
        self.W = [np.stack([w[:, ns:] + self.rng.normal(0, 0.5, w[:, ns:].shape) for _ in range(self.K)]) for w in W1]
        self.pi = np.full(self.K, 1.0 / self.K)
        A = np.full((self.K, self.K), 0.05 / max(self.K - 1, 1))
        np.fill_diagonal(A, 0.95 if self.K > 1 else 1.0)
        self.A = A / A.sum(1, keepdims=True)

    def e_step(self, seqs):
        out, ll = [], 0.0
        logpi, logA = np.log(self.pi + 1e-300), np.log(self.A + 1e-300)
        for X, Y in seqs:
            g, xs, l_ = _forward_backward(logpi, logA, _log_emission(self.W, X, Y, self.B, self.ns))
            out.append((g, xs))
            ll += l_
        return out, ll

    def m_step(self, seqs, post):
        K = self.K
        self.pi = np.maximum(sum(g[0] for g, _ in post), 1e-12)
        self.pi /= self.pi.sum()
        cnt = sum(xs for _, xs in post) + 1.0 + np.eye(K) * (STICKY - 1.0)
        self.A = cnt / cnt.sum(1, keepdims=True)
        X = np.vstack([s[0] for s in seqs])
        Y = np.vstack([s[1] for s in seqs])
        G = np.vstack([g for g, _ in post])
        ns = self.ns
        if not ns:
            self.W = [np.stack([_fit_emission(Wj[k], X, Y[:, j], G[:, k], self.C[j]) for k in range(K)])
                      for j, Wj in enumerate(self.W)]
            return
        # design v2: coordinate step -- state weights given the shared offsets, then the shared offsets given the
        # state weights (one weighted regression over the K-fold stacked trials, each copy weighted by its posterior)
        Xa, Xs = X[:, :ns], X[:, ns:]
        for j in range(len(self.W)):
            off = Xa @ self.B[j].T
            self.W[j] = np.stack([_fit_emission(self.W[j][k], Xs, Y[:, j], G[:, k], self.C[j], off)
                                  for k in range(K)])
            off_k = np.vstack([Xs @ self.W[j][k].T for k in range(K)])
            self.B[j] = _fit_emission(self.B[j], np.tile(Xa, (K, 1)), np.tile(Y[:, j], K), G.T.ravel(), self.C[j],
                                      off_k)

    def fit(self, seqs, init: "GLMHMM | None" = None, n_restarts: int = N_RESTARTS):
        D = seqs[0][0].shape[1]
        X = np.vstack([s[0] for s in seqs])
        Y = np.vstack([s[1] for s in seqs])
        W1 = [_fit_emission(np.zeros((c - 1, D)), X, Y[:, j], np.ones(len(X)), c) for j, c in enumerate(self.C)]
        best = None
        for r in range(1 if init is not None else n_restarts):
            if init is not None:
                self.W, self.pi, self.A = [w.copy() for w in init.W], init.pi.copy(), init.A.copy()
                self.B = [b.copy() for b in init.B] if init.B is not None else None
            else:
                self.rng = np.random.default_rng(1000 + r)
                self._init(W1)
            prev = -np.inf
            for _ in range(MAX_ITER):
                post, ll = self.e_step(seqs)
                self.m_step(seqs, post)
                if ll - prev < TOL * abs(ll):
                    break
                prev = ll
            _post, ll = self.e_step(seqs)
            if best is None or ll > best[0]:
                best = (ll, [w.copy() for w in self.W], self.pi.copy(), self.A.copy(),
                        [b.copy() for b in self.B] if self.B is not None else None)
        self.ll, self.W, self.pi, self.A, self.B = best
        return self

    def loglik(self, seqs) -> float:
        return self.e_step(seqs)[1]

    def loglik_by_output(self, seqs) -> np.ndarray:
        """Expected log-likelihood of each output under the posteriors (for per-output pseudo-R2)."""
        post, _ = self.e_step(seqs)
        out = np.zeros(len(self.C))
        for (X, Y), (g, _) in zip(seqs, post):
            for j, Wj in enumerate(self.W):
                out[j] += (g * _log_emission_one(Wj, X, Y[:, j], self.B[j] if self.ns else None, self.ns)).sum()
        return out

    def posteriors(self, seqs):
        return [g for g, _ in self.e_step(seqs)[0]]

    def engagement_order(self, X_mean) -> np.ndarray:
        """State indices sorted by P(class 0 of output 0 = attempt / contact) at the average input."""
        pc = []
        for k in range(self.K):
            eta = np.r_[0.0, self.W[0][k] @ X_mean[self.ns:] + (self.B[0] @ X_mean[:self.ns] if self.ns else 0.0)]
            pc.append(np.exp(eta[0] - logsumexp(eta)))
        return np.argsort(pc)[::-1]


# ------------------------------------------------------------------------------------------------------------- data ----
#: OUTCOME coding (Priya 11-16: "unclear to me if contact and incomplete cued should be different"): "3" (DEFAULT) merges
#: them into ATTEMPT -- engagement states should not encode the post-stroke aim deficit, and the frozen pre model has seen
#: few incompletes; "4" keeps them apart (sensitivity). Class 0 is the reference (attempt / contact).
OUTCOME_ENV = "GLMHMM_OUTCOME"
OUTCOMES = {"3": {"contact": 0, "incomplete": 0, "jaw_only": 1, "no_response": 2},
            "4": {"contact": 0, "incomplete": 1, "jaw_only": 2, "no_response": 3}}
RUN_LEVELS = (0.0, 0.5)          # running fraction of a window: 0 -> still, (0, 0.5) -> some, >= 0.5 -> running
#: the observations of one TRIAL CYCLE (ITI before the trial + its pre-cue + its response; Priya 11-16: "we don't want
#: to only look at cued-response periods", "running might come out as its own state", "cued vs spontaneous may be their
#: own classes"); each a categorical output with its own state-dependent logistic emission
OUTPUTS = ("outcome", "precue_lick", "iti_lick", "run_precue", "run_iti")


def outcome_mode() -> str:
    v = os.environ.get(OUTCOME_ENV, "3")
    if v not in OUTCOMES:
        raise ValueError(f"{OUTCOME_ENV}={v!r}; expected one of {sorted(OUTCOMES)}")
    return v


def n_classes() -> list[int]:
    return [len(set(OUTCOMES[outcome_mode()].values())), 2, 2, 3, 3]


def _run_frac(a, b, starts, stops) -> float:
    """Fraction of [a, b) (s) inside running bouts; NaN for an empty / undefined window."""
    if not (np.isfinite(a) and np.isfinite(b)) or b <= a:
        return np.nan
    ov = np.clip(np.minimum(stops, b) - np.maximum(starts, a), 0, None).sum()
    return float(ov / (b - a))


def _run_level(f) -> int:
    if not np.isfinite(f):
        return -1
    return 0 if f <= RUN_LEVELS[0] else (1 if f < RUN_LEVELS[1] else 2)


def session_sequence(label: str, animals) -> tuple[np.ndarray, np.ndarray, pd.DataFrame] | None:
    """(X [T, D], Y [T, len(OUTPUTS)], cycle table) of ALL trials of one session, in trial order. One row per trial
    CYCLE: the ITI before the trial (previous stop .. strobe), its pre-cue (strobe .. cue) and its response (cue ..
    stop). Licks = every camera lick (LP licks_all, DAQ time); running = the canonical DAQ running bouts."""
    from wfield_local import behavior_events as be
    from wfield_local import trial_classes as tc
    from wfield_local.paths import PathResolver
    S = tc.session_trials(label)
    if S is None or len(S) < 20:
        return None
    S = S.sort_values("trial_id").reset_index(drop=True)
    animal, date = label.split("_")
    rv = PathResolver()
    ev = be.get_or_compute(rv, animal, date)
    if ev is None:
        return None
    fs = float(ev["fs"])
    rs, rp = np.asarray(ev["running_starts"]) / fs, np.asarray(ev["running_stops"]) / fs
    # 11-22: THE session lick table (`session_licks`: the fused four-camera train when selected, else cam4 licks_all)
    from wfield_local import session_licks as SL
    lk = SL.session_onsets_s(label, rv)

    def any_lick(a, b):
        if not (np.isfinite(a) and np.isfinite(b)) or b <= a:
            return -1
        return int(np.searchsorted(lk, b) > np.searchsorted(lk, a))
    cue = S.cue_daq.to_numpy(float)
    strobe = cue + S.strobe_ms.to_numpy(float) / 1000.0
    stop = cue + S.stop_ms.to_numpy(float) / 1000.0
    prev_stop = np.r_[np.nan, stop[:-1]]
    omap = OUTCOMES[outcome_mode()]
    y_out = S.cls.map(omap).fillna(-1).astype(int).to_numpy()
    rows = []
    for t in range(len(S)):
        fp, fi = _run_frac(strobe[t], cue[t], rs, rp), _run_frac(prev_stop[t], strobe[t], rs, rp)
        rows.append((y_out[t], any_lick(strobe[t], cue[t]), any_lick(prev_stop[t], strobe[t]), _run_level(fp),
                     _run_level(fi), fp, fi))
    R = np.array(rows, dtype=float)
    Y = R[:, :5].astype(int)
    # INPUTS: per-animal bias, spout dummies, previous outcome dummies, spout moved since the last trial
    A = np.array([[1.0 if animal == a else 0.0 for a in animals]] * len(S))
    P = np.zeros((len(S), len(POSITIONS) - 1))
    for t, i in enumerate(S.position.map({p: i for i, p in enumerate(POSITIONS)}).fillna(-1).astype(int)):
        if i > 0:
            P[t, i - 1] = 1.0
    nc = n_classes()[0]
    prev = np.zeros((len(S), nc - 1))
    for t in range(1, len(S)):
        if y_out[t - 1] > 0:
            prev[t, y_out[t - 1] - 1] = 1.0
    pos = S.position.to_numpy()
    moved = np.r_[0.0, (pos[1:] != pos[:-1]).astype(float)]
    X = np.hstack([A, P, prev, moved[:, None]])
    T = S[["trial_id", "position", "cls"]].assign(label=label, run_frac_precue=R[:, 5], run_frac_iti=R[:, 6],
                                                   precue_lick=Y[:, 1], iti_lick=Y[:, 2])
    return X, Y, T


def is_august_pre(label: str) -> bool:
    from wfield_local import epochs, run_mode
    return epochs.epoch_of(label) == "pre" and label[-4:] >= run_mode.PRIMARY_PRE_FROM


def _load_job(item):
    lab, animals = item
    return session_sequence(lab, list(animals))


def load_all(labels, animals, jobs: int = 4) -> dict:
    """{label: (X, Y, table)} for every session that has trial classes, events and LP licks (fanned over sessions)."""
    from wfield_local import analysis_kit as ak
    res, fails = ak.fan_sessions([(lab, tuple(animals)) for lab in labels], _load_job, jobs=jobs,
                                 key=lambda it: it[0])
    for f in fails:
        print("  load failed:", f)
    return {it[0]: v for it, v in res if v is not None}


def out_dir() -> Path:
    env = os.environ.get("GLMHMM_OUT")
    if env:
        return Path(env)
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    from wfield_local import orofacial_cohort as oc
    return Path(oo.root(PathResolver())) / oc.RUN / "glm_hmm" / "v1"


# ------------------------------------------------------------------------------------------------------------- runs ----
def _null_ll(train, test, n_cls) -> np.ndarray:
    """Per-output held-out log-likelihood of the NULL model: each output's marginal class frequencies in the training
    sessions (no inputs, no states) -- the denominator of the pseudo-R2."""
    Ytr = np.vstack([y for _x, y in train])
    Yte = np.vstack([y for _x, y in test])
    out = np.zeros(len(n_cls))
    for j, c in enumerate(n_cls):
        cnt = np.bincount(Ytr[Ytr[:, j] >= 0, j], minlength=c) + 1.0
        lp = np.log(cnt / cnt.sum())
        yj = Yte[Yte[:, j] >= 0, j]
        out[j] = lp[yj].sum()
    return out


def select_k(data: dict, ks=(1, 2, 3, 4, 5), n_folds: int = 4, seed: int = 0, labels=None) -> pd.DataFrame:
    """Held-out fit by number of states over session folds of the August PRE sessions (Priya 11-16: "look at explained
    variance with different state numbers and choose"): total held-out log-likelihood, its gain over K = 1 in bits per
    trial cycle, and per output the McFadden pseudo-R2 = 1 - ll_model / ll_null (null = marginal class frequencies;
    the per-output model term is the posterior-expected log-likelihood, so the outputs' shares are comparable)."""
    if labels is None:
        pre = sorted(lab for lab in data if is_august_pre(lab))
        rng = np.random.default_rng(seed)
        fold = dict(zip(pre, rng.permutation(np.arange(len(pre)) % n_folds)))
    else:                       # given sessions (per animal, all epochs): folds INTERLEAVED in date order, so every
        pre = sorted(labels)    # fold spans pre / acute / subacute / chronic
        fold = {lab: i % n_folds for i, lab in enumerate(pre)}
    nc = n_classes()
    rows = []
    for f in range(n_folds):
        tr = [data[lab][:2] for lab in pre if fold[lab] != f]
        te = [data[lab][:2] for lab in pre if fold[lab] == f]
        nll = _null_ll(tr, te, nc)
        n_cyc = sum(len(y) for _x, y in te)
        for K in ks:
            m = GLMHMM(K).fit(tr)
            r = {"K": K, "fold": f, "ll": m.loglik(te), "n_cycles": n_cyc}
            for j, (o, l_) in enumerate(zip(OUTPUTS, m.loglik_by_output(te))):
                r[f"pR2_{o}"] = 1.0 - l_ / nll[j]
            r["pR2_total"] = 1.0 - sum(m.loglik_by_output(te)) / nll.sum()
            rows.append(r)
    R = pd.DataFrame(rows)
    if (R.K == 1).any():
        base = R[R.K == 1].set_index("fold").ll
        R["bits_per_cycle_vs_K1"] = (R.ll - R.fold.map(base)) / R.n_cycles / np.log(2)
    else:                               # a run of higher K only: compare through the pseudo-R2 (same folds, same null)
        R["bits_per_cycle_vs_K1"] = np.nan
    return R


def _table(m: GLMHMM, labs, data, frame, order, post=None):
    rows = []
    post = post if post is not None else m.posteriors([data[lab][:2] for lab in labs])
    for lab, g in zip(labs, post):
        g = g[:, order]
        T = data[lab][2].copy()
        for k in range(m.K):
            T[f"p_state{k}"] = g[:, k]
        T["state"] = np.where(g.max(1) > STATE_P, g.argmax(1), -1)
        T["frame"], T["K"] = frame, m.K
        rows.append(T)
    return pd.concat(rows, ignore_index=True)


def run_frozen(data: dict, K: int) -> tuple[pd.DataFrame, dict]:
    pre = sorted(lab for lab in data if is_august_pre(lab))
    full = GLMHMM(K).fit([data[lab][:2] for lab in pre])
    Xm = np.vstack([data[lab][0] for lab in pre]).mean(0)
    order = full.engagement_order(Xm)
    # pre sessions: leave-one-session-out, warm-started from the full fit
    loso = []
    for lab in pre:
        m = GLMHMM(K).fit([data[x][:2] for x in pre if x != lab], init=full)
        o = m.engagement_order(Xm)
        loso.append(_table(m, [lab], data, "frozen", o))
    post_labs = sorted(lab for lab in data if lab not in set(pre))
    tables = loso + ([_table(full, post_labs, data, "frozen", order)] if post_labs else [])
    params = {"K": K, "pre": pre, "W": [w[order].tolist() for w in full.W], "A": full.A[np.ix_(order, order)].tolist(),
              "pi": full.pi[order].tolist(), "ll": full.ll, "outputs": list(OUTPUTS), "outcome_mode": outcome_mode(),
              "n_classes": n_classes()}
    return pd.concat(tables, ignore_index=True), params


def run_refit(data: dict, K: int) -> pd.DataFrame:
    out = []
    for an in sorted({lab.split("_")[0] for lab in data}):
        labs = sorted(lab for lab in data if lab.startswith(an + "_"))
        m = GLMHMM(K).fit([data[lab][:2] for lab in labs])
        order = m.engagement_order(np.vstack([data[lab][0] for lab in labs]).mean(0))
        out.append(_table(m, labs, data, "refit", order))
    return pd.concat(out, ignore_index=True)


#: DESIGN (Priya 11-17: "fix the design issues"). v1 = X [animal one-hot, spout dummies, previous-outcome dummies,
#: moved], every weight state-specific -- the pooled fit let states absorb animal style (state-specific animal bias)
#: and explained no-response STREAKS inside engaged states (previous outcome predicts outcome), so only 30-60 % of
#: quit-tail trials landed in the disengaged state. v2 (DEFAULT) = X [animal offsets (all but the first animal) SHARED
#: across states, then a state intercept, spout dummies, moved]: no previous-outcome input, so a streak of
#: non-responses has to be carried by the state.
DESIGNS = ("v1", "v2")


def apply_design(data: dict, design: str, n_animals: int) -> dict:
    """Rebuild X of every session for ``design`` from the v1 columns (no reload); sets N_SHARED."""
    global N_SHARED
    if design == "v1":
        N_SHARED = 0
        return data
    if design != "v2":
        raise ValueError(design)
    npos, nprev = len(POSITIONS) - 1, n_classes()[0] - 1
    out = {}
    for lab, (X, Y, T) in data.items():
        A = X[:, :n_animals]
        P = X[:, n_animals:n_animals + npos]
        moved = X[:, n_animals + npos + nprev:]
        out[lab] = (np.hstack([A[:, 1:], np.ones((len(X), 1)), P, moved]), Y, T)
    N_SHARED = n_animals - 1
    return out


def sequence_key(labels) -> str:
    """Cache key of the per-session sequences: the label set + outcome mode AND (11-22, reconciled plan item 4: a
    refit after the classes / licks changed reused the OLD sequences, keyed on labels only) the identity of what the
    sequences are built from -- the trial classes' sources (`trial_classes.class_source_signature`: cam4 run tables,
    cam1 events, jaw_onsets, class code) and the lick set (`session_licks.signature`, "" on the legacy source) -- plus
    this module's code."""
    import hashlib
    import inspect
    import sys

    from wfield_local import session_licks as SL
    from wfield_local import trial_classes as tc
    src = inspect.getsource(sys.modules[__name__])
    parts = ["|".join(labels), outcome_mode(), tc.class_source_signature(), SL.signature(),
             hashlib.sha1(src.encode()).hexdigest()[:10]]
    return hashlib.sha1("#".join(parts).encode()).hexdigest()[:10]


def load_cached(labels, animals, od: Path) -> dict:
    """`load_all` with a pickle in the output dir keyed by `sequence_key` (reloading 120 sessions costs minutes and does
    not change between design / K runs)."""
    import pickle
    key = sequence_key(labels)
    f = od / f"sequences_v1_{key}.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    data = load_all(labels, animals)
    f.write_bytes(pickle.dumps(data))
    return data


def august_on(data: dict) -> list[str]:
    return sorted(lab for lab in data if lab[-4:] >= "0801")


def model_params(m: "GLMHMM", order, labs, data) -> dict:
    """The fitted model in ``order`` (the table's state numbering): state weights W, shared offsets B, transitions A,
    initial pi, the mean input (for predicted emission profiles) and the coding needed to read them."""
    return {"K": m.K, "labels": list(labs), "W": [w[order].tolist() for w in m.W],
            "B": [b.tolist() for b in m.B] if m.ns else None, "n_shared": int(m.ns),
            "A": m.A[np.ix_(order, order)].tolist(), "pi": m.pi[order].tolist(), "ll": m.ll,
            "X_mean": np.vstack([data[lab][0] for lab in labs]).mean(0).tolist(), "outputs": list(OUTPUTS),
            "positions": list(POSITIONS), "outcome_mode": outcome_mode(), "n_classes": n_classes()}


def run_pooled(data: dict, K: int) -> tuple[pd.DataFrame, dict]:
    """ONE model over ALL animals and ALL August-onward epochs (Priya 11-17: per-animal states are not comparable --
    the quit tail sat in a different state in each animal). Shared emissions = shared state definitions; the
    per-animal bias columns of X absorb each animal's baseline rates, so states are not forced to encode identity
    (the pooled-PRE fit's failure, 11-16, had no post-stroke variation to model)."""
    labs = august_on(data)
    m = GLMHMM(K).fit([data[lab][:2] for lab in labs])
    order = m.engagement_order(np.vstack([data[lab][0] for lab in labs]).mean(0))
    return _table(m, labs, data, "pooled_all", order), model_params(m, order, labs, data)


def main(argv=None) -> int:
    from wfield_local import analysis_kit as ak
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--select-k", action="store_true")
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--per-animal", action="store_true",
                    help="K selection PER ANIMAL over ALL its August-onward sessions (Priya 11-16: pooled pre states "
                         "encoded animal identity)")
    ap.add_argument("--pooled-all", action="store_true",
                    help="ONE model over all animals, all August-onward epochs (shared states, per-animal bias "
                         "inputs); with --select-k: folds interleaved by session within animal")
    ap.add_argument("--ks", default="1,2,3,4,5", help="--select-k: the K values (folds are fixed, so runs add up)")
    ap.add_argument("--design", default="v2", choices=DESIGNS, help="see DESIGNS (v2 = shared animal offsets, no "
                    "previous-outcome input)")
    ap.add_argument("--params-only", action="store_true", help="--pooled-all --k K: refit and write the model "
                    "parameters only, after checking the refit reproduces the posterior table on disk")
    ap.add_argument("--force", action="store_true", help="--pooled-all: overwrite an existing states / params table "
                    "(refused otherwise; GLMHMM_OUT=<new folder> writes a new one)")
    a = ap.parse_args(argv)
    labels = sorted(ak.curated_ids())
    animals = sorted({lab.split("_")[0] for lab in labels})
    od = out_dir()
    od.mkdir(parents=True, exist_ok=True)
    data = apply_design(load_cached(labels, animals, od), a.design, len(animals))
    sfx = "" if a.design == "v1" else f"_{a.design}"
    print(f"design {a.design}: X width {next(iter(data.values()))[0].shape[1]}, shared columns {N_SHARED}", flush=True)
    print(f"{len(data)} sessions, {sum(len(v[1]) for v in data.values())} trials; out {od}")
    if a.pooled_all:
        if a.select_k:
            ks = tuple(int(k) for k in a.ks.split(","))
            R = select_k(data, ks=ks, labels=august_on(data))
            kt = "" if ks == (1, 2, 3, 4, 5) else "_K" + "-".join(map(str, ks))
            R.to_csv(od / f"select_k_pooled_all{sfx}{kt}.csv", index=False)
            print(R.groupby("K")[["bits_per_cycle_vs_K1", "pR2_total", *[f"pR2_{o}" for o in OUTPUTS]]].mean().round(4),
                  flush=True)
        for K in ([a.k] if a.k else ([] if a.select_k else [3, 4])):
            T, params = run_pooled(data, K)
            if a.params_only:              # same seed + data as the table on disk: write the parameters beside it
                old = pd.read_csv(od / f"states_pooled_all_K{K}{sfx}.csv.gz")
                dmax = float(np.abs(old[[f"p_state{k}" for k in range(K)]].to_numpy()
                                    - T[[f"p_state{k}" for k in range(K)]].to_numpy()).max())
                print(f"K{K}: max |posterior - table on disk| = {dmax:.2e}", flush=True)
                if dmax > 1e-6:
                    raise SystemExit("refit does not reproduce the table on disk; parameters NOT written")
            else:
                f = od / f"states_pooled_all_K{K}{sfx}.csv.gz"
                if f.exists() and not a.force:
                    # 11-22 (reconciled plan item 4): the production states / params feed the engagement / TIM
                    # chain; a refit on new classes or licks goes to a new GLMHMM_OUT unless replacing on purpose
                    raise SystemExit(f"{f} exists: --force to overwrite, or GLMHMM_OUT=<new folder>")
                T.to_csv(f, index=False)
            (od / f"params_pooled_all_K{K}{sfx}.json").write_text(json.dumps(params, indent=1))
            print("written", od / f"params_pooled_all_K{K}{sfx}.json", flush=True)
        return 0
    if a.select_k and a.per_animal:
        out = []
        for an in animals:
            labs = [lab for lab in data if lab.startswith(an + "_") and lab[-4:] >= "0801"]
            R = select_k(data, labels=labs)
            R.insert(0, "animal", an)
            out.append(R)
            print(an, len(labs), "sessions")
            print(R.groupby("K")[["bits_per_cycle_vs_K1", "pR2_total"]].mean().round(4), flush=True)
        pd.concat(out).to_csv(od / "select_k_per_animal.csv", index=False)
    elif a.select_k:
        R = select_k(data)
        R.to_csv(od / "select_k.csv", index=False)
        print(R.groupby("K")[["bits_per_cycle_vs_K1", "pR2_total", *[f"pR2_{o}" for o in OUTPUTS]]].mean().round(4))
    if a.k:
        F, params = run_frozen(data, a.k)
        F.to_csv(od / f"states_frozen_K{a.k}.csv.gz", index=False)
        (od / f"params_frozen_K{a.k}.json").write_text(json.dumps(params, indent=1))
        run_refit(data, a.k).to_csv(od / f"states_refit_K{a.k}.csv.gz", index=False)
        print("written", od)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
