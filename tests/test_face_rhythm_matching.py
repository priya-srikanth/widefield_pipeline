"""face_rhythm_matching: Hungarian alignment of independent FR fits, split flag, chance, pairwise floor."""
import numpy as np

from wfield_local import face_rhythm_matching as fm


def _fit(rng, P=40, F=12, R=4):
    return rng.random((2 * P, R)), rng.random((F, R))


def test_recovers_permutation_and_scale():
    rng = np.random.default_rng(0)
    A, B = _fit(rng)
    perm = np.array([2, 0, 3, 1])
    m = fm.match(A, B, 3 * A[:, perm], 0.5 * B[:, perm])
    np.testing.assert_array_equal(perm[m["order"]], np.arange(4))
    np.testing.assert_allclose(m["cos_spatial"], 1)
    assert (m["free"] == m["order"]).all()


def test_unequal_ranks_and_split_flag():
    rng = np.random.default_rng(1)
    A, B = _fit(rng, R=5)
    m = fm.match(A, B, A[:, :4], B[:, :4])                      # ref has one more factor: it stays unmatched
    assert (m["order"] == -1).sum() == 1 and np.isnan(m["cos_spatial"]).sum() == 1
    A2 = np.column_stack([A[:, 0], A[:, 0], A[:, 2]])           # factor 0 duplicated ("split"), factor 1 missing
    B2 = np.column_stack([B[:, 0], B[:, 0], B[:, 2]])
    m = fm.match(A[:, :3], B[:, :3], A2, B2)
    assert m["free"][1] != m["order"][1] or m["margin"][0] < 1e-9


def test_chance_below_identity_and_pairwise_floor():
    rng = np.random.default_rng(2)
    A, B = _fit(rng)
    ch = fm.chance(A, B, A, B, n=30)
    assert (ch["cos_spatial_p95"] < 0.999).all()
    fits = {"a": (A, B), "b": (A[:, ::-1], B[:, ::-1])}
    P = fm.pairwise(fits, A, B)
    assert len(P) == 4 and np.allclose(P.cos_spatial, 1)


def test_magnitude_view_halves_points():
    A = np.arange(12, dtype=float).reshape(6, 2)
    assert fm.spatial_view(A, "magnitude").shape == (3, 2)
