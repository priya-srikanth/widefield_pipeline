"""A control that DOES NOT WORK at this rank, kept so it is not rebuilt.

THE QUESTION IT WAS BUILT FOR. Every cross-session result in this deck is expressed in a joint
LocaNMF basis fitted on PRE-STROKE sessions, with post-stroke days PROJECTED onto it. Priya,
2026-09-12: "the shared locanmf basis is built only on pre-stroke sessions ... ok to use?" The worry
is real in principle -- a pre-stroke basis can only express pre-stroke spatial patterns, so
structure the lesion creates would be discarded silently, and discarded structure is exactly a
candidate for "where the code moved to".

THE PLAN WAS to decode spout position from the RESIDUAL -- the part of a session's activity
orthogonal to the basis -- and see whether the basis throws away anything position-informative.

IT CANNOT ANSWER THAT, AND THE REASON IS STRUCTURAL:

    session SVD rank K = 100        joint basis dims = 100        subspace overlap = 0.996-0.997

The basis is a 100-dimensional subspace and the DATA IS ALSO 100-DIMENSIONAL, because `SVTcorr` is
already truncated to rank 100 in preprocessing. The basis is therefore very nearly a ROTATION of the
data rather than a reduction of it, and "the residual" is the misalignment between two 100-dim
subspaces: 0.3% of the energy, with an ISOTROPIC eigenvalue spectrum (0.029, 0.028, 0.028, 0.028,
0.028, 0.027 -- a noise floor, not a coding subspace).

AND IT PRODUCED A CONVINCING WRONG ANSWER, which is why this file survives. Feeding 95 near-null
directions to a pipeline whose `StandardScaler` z-scores each feature independently RESURRECTS them
all to unit variance, reconstructing a re-mix of the session's entire temporal space. "The residual"
then decoded position at 0.93 balanced accuracy pre-stroke against a chance of 0.167 -- a number
that looks like a discovery and is an artefact of the construction. Restoring the singular-value
scaling (below, and it is the correct arithmetic) changed the decode NOT AT ALL, because the scaler
undoes it; that invariance is what exposed the error.

WHAT THE INVESTIGATION DID ESTABLISH, and it is the better answer to the original question:

  * THE BASIS IS NOT THE BOTTLENECK. 100 dimensions spanning 99.7% of a 100-dimensional dataset
    discards almost nothing. The real dimensionality reduction is the rank-100 SVD in preprocessing,
    which is UPSTREAM of every analysis here and applies identically to pre- and post-stroke
    sessions, so whatever it discarded cannot bias a pre-versus-post comparison.
  * THE MISALIGNMENT DOES GROW after the lesion -- 0.29-0.37% pre-stroke against 0.48-0.80% acutely
    -- which is a small, real drift outside the pre-stroke subspace, worth reporting as such and not
    as evidence of relocation.

TO ACTUALLY TEST WHAT PREPROCESSING DISCARDS you would need the pre-SVD movie. That is a much larger
undertaking and is not this.

`residual_signal` below is left correct (singular values included) for anyone who wants the residual
for a different purpose. Do not use it to argue that the basis loses position information.
"""
from __future__ import annotations

import numpy as np

#: Residual directions kept. Comparable in count to the basis's 95 components, so the two arms are
#: given a similar number of features and a difference between them cannot be a dimensionality
#: artefact -- the decoder sees 95 x bins either way.
N_RESIDUAL = 95


def residual_signal(session, basis, n_comp=N_RESIDUAL):
    """``(C_res, frac_var)`` -- residual component timecourses and the energy fraction they carry.

    ``C_res`` is ``(n_comp, T)``: the session's activity projected onto the top residual directions,
    i.e. onto the orthogonal complement of the basis's pixel subspace.

    RETURNS THE ENERGY FRACTION TOO, because a residual that carries 1% of the variance and decodes
    at chance means something different from one that carries 1% and decodes well -- the second says
    the basis discards a small but INFORMATIVE slice, which is the finding this module exists to be
    able to report.
    """
    from wfield_local import joint_basis

    u, v = joint_basis._load_session(session["mc"])
    Uj = np.load(basis.root / "U_joint.npy")
    if u.shape[0] != Uj.shape[0]:
        raise ValueError(f"{session['label']} is not on the basis pixel grid "
                         f"({u.shape[0]} px vs {Uj.shape[0]})")
    uu = np.asarray(u.T @ u, dtype=np.float64)                 # (K, K)
    P = np.asarray(Uj.T @ u, dtype=np.float64)                 # (100, K)
    G = uu - P.T @ P                                           # residual Gram, (K, K)
    # SYMMETRISED before eigendecomposition: `uu - P.T P` is symmetric in exact arithmetic and
    # drifts by ~1e-7 in float, which `eigh` does not mind but `argsort` on complex eigenvalues
    # would. Cheap insurance on a K x K matrix.
    G = 0.5 * (G + G.T)
    w, V = np.linalg.eigh(G)
    order = np.argsort(w)[::-1][:n_comp]
    # NEGATIVE EIGENVALUES ARE NUMERICAL ZEROS of a rank-deficient residual, not directions.
    keep = [i for i in order if w[i] > 1e-9 * max(float(w.max()), 1e-30)]
    if not keep:
        return np.zeros((0, v.shape[1]), np.float32), 0.0
    W = V[:, keep]                                             # (K, m)
    # SCALED BY THE SINGULAR VALUES, and this is the whole correctness of the construction.
    #
    # `G = u_res^T u_res = Z S^2 Z^T`, so the residual ACTIVITY projected onto its own orthonormal
    # spatial directions is `S Z^T v`, not `Z^T v`. Dropping S normalises every direction to unit
    # spatial norm -- which amplifies the near-NULL directions to the same scale as the real ones,
    # and since the session SVD has rank 100 and the basis is also 100-dimensional, `Z^T v` then
    # spans essentially the session's ENTIRE temporal space. The first version did exactly that and
    # "the residual" decoded position at 0.93 pre-stroke, which was not the residual decoding at
    # all: it was a re-mix of the whole session. The flat eigenvalue spectrum (0.029, 0.028, 0.028,
    # 0.028, 0.028, 0.027 -- isotropic, i.e. a noise floor) is what gave it away.
    C = np.asarray(np.sqrt(w[keep])[:, None] * (W.T @ v), dtype=np.float32)   # (m, T)
    # ENERGY, WEIGHTED BY THE TEMPORAL GRAM, exactly as `Basis.project` computes
    # `variance_captured` -- so this is 1 - VC and the two numbers are commensurable. The first
    # version multiplied the coefficient Gram by the residual timecourse Gram, which is not the
    # residual energy at all and reported 10.4% where the true figure is ~1%.
    Wt = np.asarray(v @ v.T, dtype=np.float64)                 # (K, K)
    total = float(np.sum(uu * Wt))
    res = float(np.sum(G * Wt))
    return C, (res / total if total > 1e-30 else 0.0)
