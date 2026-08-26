"""`_fast_rdm` must be a faster ROUTE to `_crossnobis_within`, never a second estimator.

An interval computed under a different whitener rule does not describe the number printed beside
it. For these RDMs that is not a rounding matter: fixing the whitener across draws moves the
whole-RDM correlation by up to 0.6 and its post-minus-PRE contrast from -0.07 to -0.51. Figure 8d
already shipped an interval that did not contain its own estimate; these tests are what stop the
same thing happening to 8b.
"""
import numpy as np
import pytest

from wfield_local import grant_figures as gf

L = gf.CONF_LABELS


def _fake(n_pos=6, n_trials=40, n_feat=25, seed=0):
    """Trials with real structure: a per-position mean plus correlated noise."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n_feat, n_feat))
    cov_root = A @ A.T / n_feat + np.eye(n_feat) * 0.5
    out = {}
    for k, q in enumerate(L[:n_pos]):
        mu = rng.normal(scale=1.5, size=n_feat) * (1 + 0.3 * k)
        out[q] = mu + rng.normal(size=(n_trials, n_feat)) @ cov_root
    return out


def test_lw_cov_is_bit_identical_to_sklearn():
    sk = pytest.importorskip("sklearn.covariance")
    R = _fake(n_trials=60, n_feat=20, seed=3)[L[0]]
    assert np.allclose(gf._lw_cov(R), sk.LedoitWolf().fit(R).covariance_, rtol=0, atol=1e-12)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fast_rdm_reproduces_crossnobis_within(seed):
    """THE LOAD-BEARING TEST. Same rng state in, same matrix out."""
    pytest.importorskip("scipy.linalg")
    src = _fake(seed=seed)
    a = gf._crossnobis_within(src, np.random.default_rng(99), L)
    b = gf._fast_rdm(src, np.random.default_rng(99), L)
    assert np.isfinite(a).sum() == 30, "a 6x6 RDM has 30 off-diagonal entries"
    assert np.allclose(a, b, equal_nan=True, rtol=1e-6, atol=1e-8), (
        f"max |diff| = {np.nanmax(np.abs(a - b)):.3e}")


def test_fast_rdm_handles_a_missing_position():
    src = _fake()
    src.pop(L[2])
    D = gf._fast_rdm(src, np.random.default_rng(0), L)
    assert np.isnan(D[2]).all() and np.isnan(D[:, 2]).all()
    assert np.isfinite(D[0, 1]), "the positions that ARE present must still be scored"


def test_fast_rdm_declines_rather_than_guessing():
    assert np.isnan(gf._fast_rdm({}, np.random.default_rng(0), L)).all()
    one = {L[0]: np.random.default_rng(0).normal(size=(20, 8))}
    assert np.isnan(gf._fast_rdm(one, np.random.default_rng(0), L)).all(), (
        "one position is not a distance")


def test_rdm_scores_matches_the_shape_of_the_figures():
    src, ref = _fake(seed=0), _fake(seed=1)
    D = gf._fast_rdm(src, np.random.default_rng(5), L)
    Dref = gf._fast_rdm(ref, np.random.default_rng(5), L)
    whole, rows = gf._rdm_scores(D, Dref)
    assert np.isfinite(whole) and -1 <= whole <= 1
    assert rows.shape == (6,) and np.isfinite(rows).all()


def test_rdm_scores_declines_a_row_built_from_three_points():
    """A row is five numbers; below four usable ones a correlation estimates nothing."""
    D = np.full((6, 6), np.nan)
    Dref = np.full((6, 6), np.nan)
    for j in (1, 2, 3):
        D[0, j] = D[j, 0] = float(j)
        Dref[0, j] = Dref[j, 0] = float(j) * 2
    _w, rows = gf._rdm_scores(D, Dref)
    assert np.isnan(rows[0]), "three partners is not a row correlation"


def test_rdm_scores_row_is_the_off_diagonal_only():
    """The diagonal of a within-set RDM is zero by construction and must not enter its own row."""
    rng = np.random.default_rng(0)
    D = rng.uniform(1, 2, (6, 6))
    D = (D + D.T) / 2
    Dref = D.copy()
    np.fill_diagonal(D, 0.0)
    np.fill_diagonal(Dref, 99.0)          # a wildly different diagonal must change nothing
    _w, rows = gf._rdm_scores(D, Dref)
    assert np.allclose(rows, 1.0), "a row correlated with itself is 1 whatever the diagonal says"


def test_excludes_zero():
    assert gf._excludes_zero((0.1, 0.4, 0.2))
    assert gf._excludes_zero((-0.4, -0.1, -0.2))
    assert not gf._excludes_zero((-0.1, 0.4, 0.2))
    assert not gf._excludes_zero(None)
