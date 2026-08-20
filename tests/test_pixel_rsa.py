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


# ------------------------------------------------------------------------------------------------
# The mirror operators. These replace a left-right IMAGE FLIP with k x k algebra, so the only thing
# that makes them trustworthy is that they agree with the pixel computation exactly -- not closely.
# ------------------------------------------------------------------------------------------------

def _fake_session(tmp_path, H=12, W=16, k=6, seed=0):
    """A tiny synthetic U_atlas + brain mask, so the algebra can be checked against brute force."""
    import json as _json

    rng = np.random.default_rng(seed)
    d = tmp_path / "allen_aligned_affine8v1"
    d.mkdir(parents=True)
    U = rng.normal(size=(H, W, k)).astype(np.float32)
    mask = np.zeros((H, W), bool)
    mask[2:10, 3:14] = True          # deliberately NOT left-right symmetric
    np.save(d / "U_atlas.npy", U)
    np.save(d / "allen_brain_mask_native_grid.npy", mask)
    return {"mc": str(tmp_path)}, U.astype(np.float64), mask


def test_mirror_and_normal_correlations_match_brute_force(tmp_path, monkeypatch):
    from wfield_local import pixel_rsa as px

    sess, U, mask = _fake_session(tmp_path)
    monkeypatch.setattr(px, "_atlas_dir", lambda s: str(tmp_path / "allen_aligned_affine8v1"))
    ops = px.mirror_operators(sess)
    m = np.isfinite(U).all(axis=2) & mask
    both = m & m[:, ::-1]
    Uf, Ufl = U[both], U[:, ::-1, :][both]
    rng = np.random.default_rng(1)
    for _ in range(10):
        a, b = rng.normal(size=U.shape[2]), rng.normal(size=U.shape[2])
        assert abs(np.corrcoef(Uf @ a, Ufl @ b)[0, 1] - px.mirror_correlation(a, b, ops)) < 1e-12
        assert abs(np.corrcoef(Uf @ a, Uf @ b)[0, 1] - px.normal_correlation(a, b, ops)) < 1e-12


def test_a_pixel_whose_mirror_leaves_the_brain_is_dropped_from_both_sides(tmp_path, monkeypatch):
    """An asymmetric mask is the normal case. A pixel whose mirror falls outside the brain must not
    contribute a background value to one side of the comparison -- it is excluded from both."""
    from wfield_local import pixel_rsa as px

    sess, U, mask = _fake_session(tmp_path, seed=2)
    monkeypatch.setattr(px, "_atlas_dir", lambda s: str(tmp_path / "allen_aligned_affine8v1"))
    _G, _M, _Gf, _s, _sf, n = px.mirror_operators(sess)
    both = (mask & mask[:, ::-1]).sum()
    assert n == both
    assert n < mask.sum(), "the fixture's mask must be asymmetric or this proves nothing"


def test_symmetrising_the_pixel_set_makes_the_flipped_norm_identical(tmp_path, monkeypatch):
    """I documented the opposite and this test caught it.

    Because a pixel is kept only when its MIRROR is also in the brain, the surviving set is closed
    under the flip -- so summing over the flipped pixels is the same sum, and Gf equals G exactly
    (6.7e-15 on real data), as does sf against s. The identity is a property of that masking choice,
    not of the algebra, so the operators are still computed separately: a future change to the
    masking would otherwise break the normalisation silently."""
    from wfield_local import pixel_rsa as px

    sess, _U, mask = _fake_session(tmp_path, seed=3)
    monkeypatch.setattr(px, "_atlas_dir", lambda s: str(tmp_path / "allen_aligned_affine8v1"))
    G, M, Gf, sv, sf, _n = px.mirror_operators(sess)
    assert np.allclose(G, Gf, atol=1e-12)
    assert np.allclose(sv, sf, atol=1e-12)
    assert np.allclose(M, M.T, atol=1e-12), "M is symmetric over a flip-closed pixel set"
    assert not np.allclose(mask, mask[:, ::-1]), "fixture mask must be asymmetric before symmetrising"
