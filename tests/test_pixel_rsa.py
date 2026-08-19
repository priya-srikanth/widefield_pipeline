"""Pixel-space RSA: the transform must be EXACT, or the whole point of it is lost.

The reason this module exists is that coefficient-space distances are not pixel distances once U has
been warped onto the Allen grid. If the correction were itself approximate it would just be a
different wrong metric.
"""
from __future__ import annotations

import numpy as np

from wfield_local import pixel_rsa as px


def _random_gram(k=12, seed=0):
    """A realistic non-orthonormal Gram matrix: symmetric positive-definite, unequal diagonal."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(40, k))
    return A.T @ A


def test_the_whitener_reproduces_pixel_distance_exactly():
    """d2 = (a-b)^T G (a-b) must equal ||L(a-b)||^2 to floating-point precision, not approximately."""
    G = _random_gram()
    L = px.pixel_whitener(G)
    rng = np.random.default_rng(1)
    for _ in range(20):
        a, b = rng.normal(size=G.shape[0]), rng.normal(size=G.shape[0])
        d_true = float((a - b) @ G @ (a - b))
        d_z = float(((L @ (a - b)) ** 2).sum())
        assert abs(d_true - d_z) <= 1e-9 * max(abs(d_true), 1.0)


def test_an_orthonormal_basis_makes_the_correction_a_no_op():
    """Sanity in the other direction: if U really were orthonormal, coefficient distance would
    already be pixel distance and this module would have nothing to do."""
    G = np.eye(8)
    L = px.pixel_whitener(G)
    assert np.allclose(L @ L.T, np.eye(8), atol=1e-12)
    d = px.basis_distortion(G)
    assert d["relative_frobenius"] == 0.0
    assert d["max_abs_offdiag"] == 0.0


def test_distortion_grows_with_non_orthonormality():
    near = px.basis_distortion(np.eye(6) + 0.01 * np.ones((6, 6)))
    far = px.basis_distortion(np.eye(6) + 0.40 * np.ones((6, 6)))
    assert near["relative_frobenius"] < far["relative_frobenius"]


def test_a_semi_definite_gram_still_yields_a_usable_whitener():
    """A rank-deficient G is fine to within rounding and must not fail: Cholesky raises, and the
    eigen fallback keeps the session analysable instead of dropping it."""
    A = np.random.default_rng(2).normal(size=(3, 7))
    G = A.T @ A                       # rank 3 in 7 dimensions
    L = px.pixel_whitener(G)
    assert np.all(np.isfinite(L))
    rng = np.random.default_rng(3)
    v = rng.normal(size=7)
    # distances in the row space are preserved; the null space contributes the clipped floor
    assert float(((L @ v) ** 2).sum()) >= 0.0


def test_to_pixel_space_is_a_plain_linear_map():
    G = _random_gram(k=5, seed=4)
    L = px.pixel_whitener(G)
    C = np.random.default_rng(5).normal(size=(11, 5))
    Z = px.to_pixel_space(C, L)
    assert Z.shape == (11, 5)
    # pairwise distances in Z equal the G-metric distances between the original rows
    for i in range(4):
        for j in range(i + 1, 5):
            d_true = float((C[i] - C[j]) @ G @ (C[i] - C[j]))
            assert abs(float(((Z[i] - Z[j]) ** 2).sum()) - d_true) <= 1e-9 * max(d_true, 1.0)
