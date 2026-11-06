"""The maths: non-negative CP by HALS (numpy, chunked), projection with spatial + spectral fixed, the FR normaliser, pooled EVR, and factor matching / similarity."""
from __future__ import annotations

import numpy as np

from .config import CHUNK, EPS
from .files import rows_of


class _scaled:
    """Lazy (n, 2P, F) float32 view of a (n, 2, P, F) float16 memmap divided by the animal's scale. ``on_chunk(start,
    rows)`` sees every slice read (rows incl. NaN ones): a second statistic computed in the SAME pass over the file
    (`cmd_project`: the window power means ride on the projection's read of the session spectrogram)."""

    def __init__(self, S, s, on_chunk=None):
        self.S, self.s, self.on_chunk = S, s, on_chunk
        self.shape = (S.shape[0], S.shape[-3] * S.shape[-2], S.shape[-1])

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, sl):
        x = rows_of(self.S[sl]) / self.s
        if self.on_chunk is not None and isinstance(sl, slice):
            self.on_chunk(sl.start or 0, x)
        return x


# ------------------------------------------------------------------------------------------------- non-negative CP
def _mttkrp(X, A, B, C, mode: str) -> np.ndarray:
    """Matricised tensor times Khatri-Rao product for X (T, P, F) ~ sum_r C[:, r] o A[:, r] o B[:, r]."""
    T, R = X.shape[0], A.shape[1]
    if mode == "C":
        out = np.empty((T, R), np.float64)
        for s in range(0, T, CHUNK):
            out[s:s + CHUNK] = np.einsum("tpr,pr->tr", X[s:s + CHUNK] @ B, A)
        return out
    out = np.zeros((A.shape[0] if mode == "A" else B.shape[0], R), np.float64)
    for s in range(0, T, CHUNK):
        x, c = X[s:s + CHUNK], C[s:s + CHUNK]
        if mode == "A":
            out += np.einsum("tpr,tr->pr", x @ B, c)
        else:
            out += np.einsum("tfr,tr->fr", np.swapaxes(x, 1, 2) @ A, c)
    return out


def _hals_cols(Fm: np.ndarray, M: np.ndarray, G: np.ndarray) -> None:
    for r in range(Fm.shape[1]):
        Fm[:, r] = np.maximum(EPS, Fm[:, r] + (M[:, r] - Fm @ G[:, r]) / max(G[r, r], EPS))


def ncp_hals(X: np.ndarray, rank: int, seed: int, n_iter: int = 300, tol: float = 1e-6, log=None) -> dict:
    """Non-negative CP of X (T, P, F) by HALS. Returns A (P, R), B (F, R) with unit-norm columns, C (T, R), rel_err
    (Frobenius, uncentred: EVR = 1 - rel_err^2, the paper's definition) and the error trace."""
    X = X.astype(np.float32, copy=False)
    T, P, F = X.shape
    rng = np.random.default_rng(seed)
    nX2 = float(sum(np.einsum("tpf,tpf->", X[s:s + CHUNK], X[s:s + CHUNK], dtype=np.float64)
                    for s in range(0, T, CHUNK)))
    A, B, C = rng.random((P, rank)), rng.random((F, rank)), rng.random((T, rank))
    A /= np.linalg.norm(A, axis=0)
    B /= np.linalg.norm(B, axis=0)
    G = (A.T @ A) * (B.T @ B)
    C *= np.sqrt(nX2 / max(float(np.sum(G * (C.T @ C))), EPS))
    trace = []
    for it in range(n_iter):
        A32, B32 = A.astype(np.float32), B.astype(np.float32)
        _hals_cols(A, _mttkrp(X, A32, B32, C.astype(np.float32), "A"), (B.T @ B) * (C.T @ C))
        A32 = A.astype(np.float32)
        _hals_cols(B, _mttkrp(X, A32, B32, C.astype(np.float32), "B"), (A.T @ A) * (C.T @ C))
        na, nb = np.linalg.norm(A, axis=0), np.linalg.norm(B, axis=0)
        A /= na
        B /= nb
        C *= na * nb
        G = (A.T @ A) * (B.T @ B)
        Mc = _mttkrp(X, A.astype(np.float32), B.astype(np.float32), None, "C")
        _hals_cols(C, Mc, G)
        err2 = nX2 - 2 * float(np.sum(Mc * C)) + float(np.sum(G * (C.T @ C)))
        trace.append(np.sqrt(max(err2, 0.0) / nX2))
        if log and (it % 25 == 0):
            log(f"    it {it}: rel err {trace[-1]:.4f}")
        if it > 10 and abs(trace[-2] - trace[-1]) < tol * trace[-1]:
            break
    return {"A": A, "B": B, "C": C, "rel_err": trace[-1], "trace": np.array(trace), "n_iter": it + 1}


def normaliser(rows: np.ndarray, freqs: np.ndarray, f: float, one_over_f: float) -> np.ndarray:
    """FR's per-bin divisor D = f m + (1 - f) (VQT_Analyzer: m = mean over xy x points of the power summed over
    frequencies, before the 1/f weighting) recovered from SAVED normalised rows (n, 2P, F), unscaled: their summed
    power n = m / D, so m = (1 - f) n / (1 - f n)."""
    n = (np.asarray(rows, np.float64) / freqs[None, None, :] ** one_over_f).sum(-1).mean(-1)
    m = (1 - f) * n / np.maximum(1 - f * n, 1e-9)
    return (f * m + (1 - f)).astype(np.float32)


def project_rows(X, A: np.ndarray, B: np.ndarray, n_iter: int = 60, row_fn=None):
    """Temporal weights (n, R) >= 0 per row of X (n, P, F) with A, B fixed (exact per-row NNLS by coordinate descent on
    the R x R system), each row's relative error ||x - x^|| / ||x||, and ||x||^2 (for pooled EVR). NaN rows -> NaN.
    ``row_fn(x_chunk) -> (rows,)``: a per-row statistic computed in the same pass (returned 4th; else 3 values)."""
    n, R = X.shape[0], A.shape[1]
    W, E, N2 = np.full((n, R), np.nan, np.float32), np.full(n, np.nan, np.float32), np.full(n, np.nan)
    RF = np.full(n, np.nan, np.float32) if row_fn is not None else None
    G = (A.T @ A) * (B.T @ B)
    A32, B32 = A.astype(np.float32), B.astype(np.float32)
    for s in range(0, n, CHUNK):
        x = np.asarray(X[s:s + CHUNK], np.float32)
        ok = np.isfinite(x).all(axis=(1, 2))
        if not ok.any():
            continue
        x = x[ok]
        M = np.einsum("tpr,pr->tr", x @ B32, A32).astype(np.float64)
        c = np.maximum(EPS, np.linalg.lstsq(G, M.T, rcond=None)[0].T)
        for _ in range(n_iter):
            for r in range(R):
                c[:, r] = np.maximum(0.0, c[:, r] + (M[:, r] - c @ G[:, r]) / max(G[r, r], EPS))
        nx2 = np.einsum("tpf,tpf->t", x, x, dtype=np.float64)
        e2 = nx2 - 2 * np.sum(M * c, 1) + np.einsum("tr,rs,ts->t", c, G, c)
        idx = np.arange(s, min(s + CHUNK, n))[ok]
        W[idx], N2[idx] = c, nx2
        E[idx] = np.sqrt(np.maximum(e2, 0) / np.maximum(nx2, EPS))
        if row_fn is not None:
            RF[idx] = row_fn(x)
    return (W, E, N2) if row_fn is None else (W, E, N2, RF)


def pooled_evr(E: np.ndarray, N2: np.ndarray) -> float:
    """Uncentred EVR over rows from `project_rows`' per-row errors: 1 - sum(err^2 ||x||^2) / sum(||x||^2)."""
    ok = np.isfinite(E) & np.isfinite(N2)
    return float(1 - np.sum(E[ok] ** 2 * N2[ok]) / max(np.sum(N2[ok]), EPS))


def match(ref: list, F: list) -> tuple[np.ndarray, np.ndarray]:
    """Hungarian match of F's components to ref's on the product of per-mode cosines. -> (order: for each ref
    component the matched F column, cos (n_modes, R_ref) per mode for the matched pairs). When ref has MORE components
    than F (a rank-12 set against a rank-10 one) the extra ref components are unmatched: order -1, cos NaN."""
    from scipy.optimize import linear_sum_assignment
    cs = []
    for a, b in zip(ref, F):
        a = a / np.maximum(np.linalg.norm(a, axis=0), EPS)
        b = b / np.maximum(np.linalg.norm(b, axis=0), EPS)
        cs.append(a.T @ b)
    i, j = linear_sum_assignment(-np.prod(cs, axis=0))
    order = np.full(cs[0].shape[0], -1)
    order[i] = j
    cos = np.full((len(cs), cs[0].shape[0]), np.nan)
    cos[:, i] = [c[i, j] for c in cs]
    return order, cos


def similarity(F1: list, F2: list) -> float:
    """Stability (Williams et al. 2018): mean over matched components of the product of per-mode cosines."""
    _, cos = match(F1, F2)
    return float(np.prod(cos, axis=0).mean())


def match_sets(F1: list, F2: list) -> tuple[np.ndarray, np.ndarray]:
    """`match` of set 1's components to set 2's (spatial x spectral cosine; same mesh points): for each set-1
    component the matched set-2 column and the (2, R1) cosines."""
    return match(F1, F2)
