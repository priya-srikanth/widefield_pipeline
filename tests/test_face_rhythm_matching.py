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


def test_tiebreak_uses_spectrum_only_among_spatial_ties():
    rng = np.random.default_rng(3)
    A, B = _fit(rng, R=3)
    # own fit: col 0 = ref 0's place with a foreign rhythm; col 1 = ref 0's place (spatial .99+) with ref 0's rhythm
    A2 = np.column_stack([A[:, 0], A[:, 0] + 0.01 * A[:, 1], A[:, 2]])
    B2 = np.column_stack([B[:, 2], B[:, 0], B[:, 1]])
    m = fm.match(A, B, A2, B2, on="spatial")
    assert fm.columns(m, "best")[0] == 0 and fm.columns(m, "tiebreak")[0] == 1
    assert fm.columns(m, "tiebreak", tie=0.0)[0] == 0
    ok = fm.recovered(m, fm.columns(m, "best"), 0.8, 0.999)
    assert not ok[0]


def test_merge_reference_sums_members():
    rng = np.random.default_rng(4)
    A, B = _fit(rng, R=4)
    A2, B2, nm = fm.merge_reference(A, B, ["a", "b", "c", "d"], {"bd": ("b", "d"), "zz": ("c", "missing")})
    assert nm == ["a", "bd", "c"] and A2.shape[1] == 3 and B2.shape[1] == 3
    u = A / np.linalg.norm(A, axis=0)
    np.testing.assert_allclose(A2[:, 1], u[:, 1] + u[:, 3])


def test_cluster_mass_finds_a_patch_and_not_noise():
    from wfield_local.face_rhythm import stats as frs
    xx, yy = np.meshgrid(np.arange(20.0), np.arange(20.0))
    home = np.column_stack([xx.ravel(), yy.ravel()])
    adj = frs.mesh_adjacency(home)
    assert len(adj[210]) == 8
    rng = np.random.default_rng(5)
    null = rng.normal(size=(300, len(home)))
    obs = rng.normal(size=len(home))
    patch = (np.abs(home[:, 0] - 10) <= 2) & (np.abs(home[:, 1] - 10) <= 2)
    obs[patch] += 3.0
    r = frs.map_permutation(obs, null, adj)
    assert r["sig_cluster"][patch].mean() > 0.8 and r["sig_cluster"][~patch].mean() < 0.02
    r0 = frs.map_permutation(rng.normal(size=len(home)), null, adj)
    assert r0["sig_cluster"].sum() < 30


def test_session_shuffle_and_drift_split():
    from wfield_local.face_rhythm import stats as frs
    xx, yy = np.meshgrid(np.arange(15.0), np.arange(15.0))
    home = np.column_stack([xx.ravel(), yy.ravel()])
    adj = frs.mesh_adjacency(home)
    rng = np.random.default_rng(2)
    patch = (np.abs(home[:, 0] - 7) <= 2) & (np.abs(home[:, 1] - 7) <= 2)
    per = []
    for _ in range(4):
        pre, post = rng.normal(size=(8, len(home))), rng.normal(size=(5, len(home)))
        post[:, patch] += 1.5
        per.append((np.vstack([pre, post]), 8))
    r = frs.session_shuffle_test(per, adj, 300, rng)
    assert r["sig_cluster"][patch].mean() > 0.8 and r["n_sessions"] == 20 and r["n_ref"] == 32
    X, n = frs.drift_split(np.arange(14.0).reshape(7, 2))
    assert n == 3 and len(X) == 7
    assert [list(a) for a in frs.chain_adjacency(3)] == [[1], [0, 2], [1]]
