"""Movement-NULL and movement-POTENT subspaces of population activity (Hasnain et al. 2025, after Elsayed et al. 2016).

Hasnain, Birnbaum, Ugarte Nunez, Hartman, Chandrasekaran & Economo (2025) Nat Neurosci, Methods / Subspace
identification (single-trial data) (Priya, 2026-10-07):

    [Q_null, Q_pot] = argmax 1/2 Tr(Q_null' C_stat Q_null) / sum_{i<=d_null} s_stat(i)
                           + 1/2 Tr(Q_pot'  C_mov  Q_pot)  / sum_{i<=d_pot}  s_mov(i)
    subject to Q_null' Q_pot = 0, Q_null' Q_null = I, Q_pot' Q_pot = I

C_stat / C_mov = covariance of activity at STATIONARY / MOVING time points (motion energy, `motion_state`);
s = singular values of each covariance (the normaliser = the most variance d dimensions could capture, so each term
is a normalised variance explained, normVE, in [0, 1]); d = min(N / 2, 20). They solved it with manopt (MATLAB);
here: Riemannian gradient ascent on the Stiefel manifold of Q = [Q_null | Q_pot] (N x (d_null + d_pot)), QR
retraction, initialised from the leading eigenvectors -- the same problem, small (N ~ 80 LocaNMF components).

Also their conservative control (`two_stage_pca`): Q_null = the first 5 PCs of stationary activity; remove it;
Q_pot = the first 5 PCs of the residual -- "ensures dynamics observed during stationarity are contained within the
movement-null subspace prior to assigning any dynamics to the movement-potent subspace."
Reconstruction X_sub = X Q Q' (their X_subspace = X Q', X_recon = X_subspace Q, with Q stored row-wise there).
"""
from __future__ import annotations

import numpy as np


def _cov(X):
    Xc = np.asarray(X, float) - np.mean(X, 0)
    return Xc.T @ Xc / max(len(Xc) - 1, 1)


def norm_ve(Q: np.ndarray, C: np.ndarray) -> float:
    """Tr(Q' C Q) / (sum of the top-d eigenvalues of C): the fraction of the best possible d-dim variance."""
    d = Q.shape[1]
    ev = np.sort(np.linalg.eigvalsh(C))[::-1]
    return float(np.trace(Q.T @ C @ Q) / ev[:d].sum())


def fit_subspaces(X_stat: np.ndarray, X_mov: np.ndarray, d_null: int | None = None, d_pot: int | None = None,
                  n_iter: int = 3000, lr: float = 0.5, tol: float = 1e-9, seed: int = 0) -> dict:
    """Jointly fit orthogonal Q_null (stationary variance) and Q_pot (moving variance). Returns Q_null (N x
    d_null), Q_pot (N x d_pot), normVE of each subspace on each condition, the objective trace."""
    C_s, C_m = _cov(X_stat), _cov(X_mov)
    N = C_s.shape[0]
    d = min(N // 2, 20)
    d_null, d_pot = d_null or d, d_pot or d
    ev_s, U_s = np.linalg.eigh(C_s)
    ev_m, U_m = np.linalg.eigh(C_m)
    a = np.sort(ev_s)[::-1][:d_null].sum()
    b = np.sort(ev_m)[::-1][:d_pot].sum()
    # init: top stationary PCs, then top moving PCs orthogonalised against them
    init = np.hstack([U_s[:, ::-1][:, :d_null], U_m[:, ::-1][:, :d_pot]])
    init += 1e-3 * np.random.default_rng(seed).standard_normal(init.shape)
    Q, _ = np.linalg.qr(init)

    def f(Q):
        return 0.5 * np.trace(Q[:, :d_null].T @ C_s @ Q[:, :d_null]) / a + \
            0.5 * np.trace(Q[:, d_null:].T @ C_m @ Q[:, d_null:]) / b

    hist = [f(Q)]
    for _ in range(n_iter):
        G = np.hstack([C_s @ Q[:, :d_null] / a, C_m @ Q[:, d_null:] / b])     # Euclidean gradient (x1, the 1/2
        sym = 0.5 * (Q.T @ G + G.T @ Q)                                          # and the 2 of d tr(Q'CQ) cancel)
        R = G - Q @ sym                                                          # Riemannian gradient (Stiefel)
        Qn, _ = np.linalg.qr(Q + lr * R)
        Qn *= np.sign(np.diag(Qn.T @ Q))[None, :]                               # keep column signs stable
        fn = f(Qn)
        if fn < hist[-1]:
            lr *= 0.5
            if lr < 1e-6:
                break
            continue
        Q = Qn
        hist.append(fn)
        if hist[-1] - hist[-2] < tol:
            break
    Qn_, Qp_ = Q[:, :d_null], Q[:, d_null:]
    return {"Q_null": Qn_, "Q_pot": Qp_, "objective": np.array(hist),
            "normVE": {"null_stat": norm_ve(Qn_, C_s), "null_mov": norm_ve(Qn_, C_m),
                       "pot_stat": norm_ve(Qp_, C_s), "pot_mov": norm_ve(Qp_, C_m)}}


def two_stage_pca(X_stat: np.ndarray, X_all: np.ndarray, k_null: int = 5, k_pot: int = 5) -> dict:
    """Their conservative control: Q_null = first k_null PCs of stationary activity; Q_pot = first k_pot PCs of
    ALL activity after removing the Q_null reconstruction (orthogonal to Q_null by construction)."""
    Xs = np.asarray(X_stat, float) - np.mean(X_stat, 0)
    _, _, vt = np.linalg.svd(Xs, full_matrices=False)
    Qn = vt[:k_null].T
    Xa = np.asarray(X_all, float) - np.mean(X_all, 0)
    Rz = Xa - (Xa @ Qn) @ Qn.T
    _, _, vt2 = np.linalg.svd(Rz, full_matrices=False)
    Qp = vt2[:k_pot].T
    Qp -= Qn @ (Qn.T @ Qp)
    Qp, _ = np.linalg.qr(Qp)
    return {"Q_null": Qn, "Q_pot": Qp}


def reconstruct(X: np.ndarray, Q: np.ndarray, center=None) -> np.ndarray:
    """Activity reconstructed from subspace Q: (X - c) Q Q' + c (c = ``center``, default the column mean)."""
    X = np.asarray(X, float)
    c = X.mean(0) if center is None else center
    return (X - c) @ Q @ Q.T + c


def parallel_analysis(X: np.ndarray, n_shuffle: int = 200, q: float = 95.0, seed: int = 0) -> int:
    """Their dimensionality upper bound: eigenvalues above the q-th percentile of a null made by shuffling each
    column's time bins independently (1,000 shuffles in the paper; ``n_shuffle`` here)."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X, float) - np.mean(X, 0)
    ev = np.sort(np.linalg.eigvalsh(_cov(X)))[::-1]
    null = np.empty((n_shuffle, len(ev)))
    for k in range(n_shuffle):
        Xs = np.column_stack([rng.permutation(c) for c in X.T])
        null[k] = np.sort(np.linalg.eigvalsh(_cov(Xs)))[::-1]
    return int(np.sum(ev > np.percentile(null, q, axis=0)))


def between_position_cov(F: np.ndarray, y: np.ndarray, n_split: int = 50, seed: int = 0) -> np.ndarray:
    """SPLIT-HALF between-position covariance of trial features ``F`` (trials x N) with labels ``y``: position means
    from two random halves of each position's trials, centred across positions, B = (M1' M2 + M2' M1) / 2,
    averaged over ``n_split`` splits. Trial noise is independent between halves, so it does not inflate B (a plain
    B from all trials carries noise / n_trials in every dimension, which pulls every subspace toward d / N)."""
    rng = np.random.default_rng(seed)
    labs = np.unique(y)
    N = F.shape[1]
    B = np.zeros((N, N))
    for _ in range(n_split):
        M1, M2 = [], []
        for c in labs:
            ii = rng.permutation(np.flatnonzero(y == c))
            h = len(ii) // 2
            M1.append(F[ii[:h]].mean(0))
            M2.append(F[ii[h:2 * h]].mean(0))
        M1 = np.array(M1) - np.mean(M1, 0)
        M2 = np.array(M2) - np.mean(M2, 0)
        B += (M1.T @ M2 + M2.T @ M1) / 2
    return B / n_split


def subspace_fraction(Q: np.ndarray, B: np.ndarray) -> float:
    """Tr(Q' B Q) / Tr(B): the share of the between-position variance ``B`` lying in subspace Q (random
    expectation d / N). Hasnain et al. project coding directions onto the subspaces; this is the same question for
    all position contrasts at once (B's column space = the span of the per-position coding directions)."""
    return float(np.trace(Q.T @ B @ Q) / np.trace(B))
