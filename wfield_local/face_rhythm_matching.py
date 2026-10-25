"""Matching FaceRhythm factors across INDEPENDENT fits (Priya 10-24: "the factor-matching analysis like the VLS stroke
methods paper, spatial loading and frequency similarity"; FACE_RHYTHM_PLAN "morphing").

Two non-negative CP fits of the same face (shared mesh: point i is the same face location in every session) have
factors in arbitrary ORDER and SCALE, so before comparing them each factor of fit B is paired with one of fit A. The
standard (tensortools `kruskal_align`, the 2pRAM-cohort manuscript, this repo's `face_rhythm_factors.match`) is the
HUNGARIAN assignment on a similarity between every pair of factors, built from the per-mode cosines of the modes the two
fits SHARE -- spatial (points) and spectral (frequencies); the temporal mode is never shared across sessions.

Choices made explicit here (each changes the answer):
  * ``on``: what the assignment maximises -- "product" (spatial x spectral cosine, the default and the fit's own rule:
    a pair must agree in BOTH where and how fast), "spatial", "spectral", or "sum". Report spatial and spectral cosines
    separately whatever the assignment used: "same place, new rhythm" and "same rhythm, new place" are the readouts.
  * ``spatial``: "vector" = the (x, y) displacement loadings, cosine over 2P (direction-sensitive: a lateral and a
    vertical movement of the same region differ); "magnitude" = |(x, y)| per point, cosine over P (WHERE only).
  * One-to-one is FORCED: when a factor has SPLIT (one ref factor -> two in the other fit) or two have MERGED, the
    assignment still pairs something. ``free`` gives each ref factor's best partner WITHOUT the one-to-one constraint and
    ``margin`` its similarity minus the runner-up's: free != assigned, or a small margin, flags a split / merge.
  * A cosine means nothing without a reference: ``chance`` = the same match against factors with points (spatial) and
    frequencies (spectral) PERMUTED (non-negative vectors have positive cosines by construction: chance is not 0), and
    the PRE-STROKE FLOOR = the matched cosines between pre-stroke sessions of the same animal (`pairwise`).

All inputs are non-negative, so there is no sign ambiguity; columns are unit-normalised before any cosine.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12
ON = ("product", "spatial", "spectral", "sum")


def _unit(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, float)
    return X / np.maximum(np.linalg.norm(X, axis=0), EPS)


def spatial_view(A: np.ndarray, mode: str = "vector") -> np.ndarray:
    """(2P, R) stacked x / y loadings (the fits' A; or (2, P, R) as in master.npz) -> the spatial vectors compared."""
    A = np.asarray(A, float)
    if A.ndim == 3:
        A = A.reshape(-1, A.shape[-1])
    if mode == "vector":
        return A
    if mode == "magnitude":
        P = A.shape[0] // 2
        return np.hypot(A[:P], A[P:])
    raise ValueError(f"spatial mode {mode!r}: vector | magnitude")


def cosines(A1, B1, A2, B2, spatial: str = "vector") -> tuple[np.ndarray, np.ndarray]:
    """Every-pair cosines (R1, R2): spatial and spectral."""
    cs = _unit(spatial_view(A1, spatial)).T @ _unit(spatial_view(A2, spatial))
    cf = _unit(B1).T @ _unit(B2)
    return cs, cf


def _score(cs: np.ndarray, cf: np.ndarray, on: str) -> np.ndarray:
    if on not in ON:
        raise ValueError(f"on={on!r}: {' | '.join(ON)}")
    return {"product": cs * cf, "spatial": cs, "spectral": cf, "sum": (cs + cf) / 2}[on]


def match(A_ref, B_ref, A, B, on: str = "product", spatial: str = "vector") -> dict:
    """Hungarian match of fit (A, B)'s factors to the reference's. Works for unequal ranks (a ref factor with no partner
    gets -1 / NaN). -> dict of per-REF-factor arrays: ``order`` (matched column of the other fit), ``cos_spatial`` /
    ``cos_spectral`` / ``score`` of the matched pair, ``free`` (best partner without the one-to-one constraint),
    ``margin`` (best minus second-best score over the other fit's factors), and the full ``score_matrix``."""
    from scipy.optimize import linear_sum_assignment
    cs, cf = cosines(A_ref, B_ref, A, B, spatial)
    S = _score(cs, cf, on)
    i, j = linear_sum_assignment(-S)
    R = S.shape[0]
    order = np.full(R, -1)
    order[i] = j
    pick = lambda M: np.array([M[k, order[k]] if order[k] >= 0 else np.nan for k in range(R)])  # noqa: E731
    srt = np.sort(S, axis=1)
    return {"order": order, "cos_spatial": pick(cs), "cos_spectral": pick(cf), "score": pick(S),
            "free": S.argmax(1), "margin": srt[:, -1] - (srt[:, -2] if S.shape[1] > 1 else 0.0), "score_matrix": S}


def chance(A_ref, B_ref, A, B, on: str = "product", spatial: str = "vector", n: int = 200, seed: int = 0) -> dict:
    """Matched cosines against the other fit's factors with POINTS (spatial) and FREQUENCIES (spectral) permuted --
    keeps each factor's loading distribution, destroys where / how fast. -> per-ref-factor 95th percentiles over ``n``
    shuffles of the matched spatial / spectral cosine (the chance line to read a matched cosine against)."""
    rng = np.random.default_rng(seed)
    A = np.asarray(A, float).reshape(-1, np.shape(A)[-1])
    P = A.shape[0] // 2
    sp, sf = [], []
    for _ in range(n):
        pp = rng.permutation(P)
        As = np.concatenate([A[:P][pp], A[P:][pp]])                     # same permutation for x and y of a point
        Bs = np.asarray(B, float)[rng.permutation(len(B))]
        m = match(A_ref, B_ref, As, Bs, on, spatial)
        sp.append(m["cos_spatial"])
        sf.append(m["cos_spectral"])
    return {"cos_spatial_p95": np.nanpercentile(sp, 95, axis=0), "cos_spectral_p95": np.nanpercentile(sf, 95, axis=0)}


def columns(m: dict, assign: str = "hungarian") -> np.ndarray:
    """The other fit's column standing for each reference factor: ``hungarian`` = the one-to-one assignment (``order``);
    ``best`` = each reference factor's best-scoring column, allowed to repeat (Priya 10-25: after the stroke the extra
    leftward licking is SPLIT over several own factors, and a one-to-one assignment can then hand a factor a poor column
    -- PS95 acute lick_left got a ~12 Hz factor while its lick_left-like column went to lick_ongoing_7Hz)."""
    if assign not in ("hungarian", "best"):
        raise ValueError(f"assign={assign!r}: hungarian | best")
    return np.asarray(m["order"] if assign == "hungarian" else m["free"])


def pairwise(fits: dict, A_ref, B_ref, on: str = "product", spatial: str = "vector",
             assign: str = "hungarian") -> "pd.DataFrame":  # noqa: F821
    """Every pair of fits {key: (A, B)}, each first aligned to the REFERENCE (so factor k means the reference's factor k
    in both), then compared factor by factor: one row per (key_a, key_b, k) with the spatial / spectral cosine between
    fit a's and fit b's factor k. Pre-vs-pre pairs of one animal = the PRE-STROKE FLOOR for any post-vs-pre number.
    Aligning both to the reference (rather than matching a to b directly) keeps the factor identity fixed across pairs.
    ``assign``: see `columns`."""
    import pandas as pd
    keys = sorted(fits)
    al = {}
    for k in keys:
        m = match(A_ref, B_ref, *fits[k], on=on, spatial=spatial)
        A, B = fits[k]
        A = np.asarray(A, float).reshape(-1, np.shape(A)[-1])
        cols = [c if c >= 0 else None for c in columns(m, assign)]
        al[k] = ([_unit(spatial_view(A, spatial))[:, c] if c is not None else None for c in cols],
                 [_unit(np.asarray(B, float))[:, c] if c is not None else None for c in cols])
    rows = []
    for a_i, a in enumerate(keys):
        for b in keys[a_i + 1:]:
            for k in range(len(al[a][0])):
                if al[a][0][k] is None or al[b][0][k] is None:
                    continue
                rows.append({"key_a": a, "key_b": b, "factor": k,
                             "cos_spatial": float(al[a][0][k] @ al[b][0][k]),
                             "cos_spectral": float(al[a][1][k] @ al[b][1][k])})
    return pd.DataFrame(rows)


def loading_asymmetry(A, home: np.ndarray, midline_x: float) -> np.ndarray:
    """Per factor: (contra - ipsi) / (contra + ipsi) of its SPATIAL LOADING magnitude |(x, y)| summed over the mesh
    points on each side of the face midline (Priya 10-24: asymmetry "both ways" -- this is the loading way; the other is
    the tracked points' movement, `face_rhythm_readouts.movement`). cam4: IPSILESIONAL = IMAGE RIGHT (x > midline),
    all lesions left. On the frozen basis this is a constant of the master; it can only change in a session's OWN fit,
    so per session it says which way a factor's shape moved (towards / away from the contralesional face), where the
    matched angle says only how much."""
    m = spatial_view(A, "magnitude")                                   # (P, R)
    ipsi = np.asarray(home)[:, 0] > midline_x
    c, i = m[~ipsi].sum(0), m[ipsi].sum(0)
    return (c - i) / np.maximum(c + i, EPS)
