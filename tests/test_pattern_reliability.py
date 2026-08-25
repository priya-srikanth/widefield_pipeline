"""The claims figures 7/7b/8/8b make, tested on synthetic data where the answer is known.

Each figure asserts something specific about its estimator -- that a split-half correlation measures
the ceiling, that disattenuation recovers a true correlation, that crossnobis is unbiased by trial
noise, that a second-order RDM correlation is invariant to a global gain change. Those are the load
bearing claims: if any of them is false the figure argues the opposite of what it says. None of them
can be checked on the real data, where the truth is what is being estimated.
"""
import numpy as np
import pytest

from wfield_local import grant_figures as gf

LABELS = gf.CONF_LABELS


def _cond(rng, n, mu, noise):
    return mu + rng.normal(0, noise, size=(n, len(mu)))


def _set(rng, mus, n=80, noise=1.0):
    return {q: _cond(rng, n, mus[q], noise) for q in mus}


def _mus(rng, dim=40, scale=1.0):
    return {q: rng.normal(0, scale, size=dim) for q in LABELS}


def test_split_half_is_a_ceiling_not_a_signal():
    """~1 when the pattern repeats, ~0 when there is nothing but noise."""
    rng = np.random.default_rng(0)
    mu = rng.normal(0, 1, size=40)
    clean = mu + rng.normal(0, 0.05, size=(200, 40))
    assert gf._split_half(clean, np.random.default_rng(1)) > 0.95

    pure_noise = rng.normal(0, 1, size=(200, 40))
    assert abs(gf._split_half(pure_noise, np.random.default_rng(1))) < 0.25

    assert np.isnan(gf._split_half(np.zeros((2, 40)), np.random.default_rng(1)))


def test_disattenuation_recovers_a_known_correlation():
    """THE ARITHMETIC BEHIND 7b. E[r_obs] ~ r_true * sqrt(rel_a * rel_b); dividing it out returns
    r_true. If this were false, 7b's right-hand panel would be a decoration rather than a control."""
    rng = np.random.default_rng(3)
    dim, n = 60, 60
    a_mu = rng.normal(0, 1, size=dim)
    # a second pattern with a KNOWN true correlation to the first
    r_true = 0.6
    perp = rng.normal(0, 1, size=dim)
    perp -= perp @ a_mu / (a_mu @ a_mu) * a_mu
    b_mu = r_true * a_mu / np.std(a_mu) + np.sqrt(1 - r_true ** 2) * perp / np.std(perp)

    A = a_mu + rng.normal(0, 3.0, size=(n, dim))
    B = b_mu + rng.normal(0, 3.0, size=(n, dim))
    raw = float(np.corrcoef(A.mean(0), B.mean(0))[0, 1])
    rel_a = gf._split_half(A, np.random.default_rng(4))
    rel_b = gf._split_half(B, np.random.default_rng(5))
    dis = raw / np.sqrt(rel_a * rel_b)

    assert raw < r_true - 0.1, f"noise should attenuate: raw={raw:.3f}"
    assert abs(dis - r_true) < 0.2, f"disattenuated={dis:.3f} vs true={r_true}"


def test_crossnobis_is_unbiased_when_there_is_no_difference():
    """Two conditions drawn from the SAME distribution are at distance 0 in expectation, however
    noisy the trials are. A non-cross-validated squared distance is positive here, and that bias is
    the entire reason figure 8 exists beside figure 6."""
    rng = np.random.default_rng(7)
    mu = rng.normal(0, 1, size=30)
    src = {q: mu + rng.normal(0, 2.0, size=(120, 30)) for q in LABELS}
    D = gf._crossnobis_within(src, np.random.default_rng(8), LABELS)
    vals = gf._triu_vals(D)
    vals = vals[np.isfinite(vals)]
    assert len(vals) == 15
    # centred on zero; the naive alternative would be strictly positive and large
    assert abs(np.mean(vals)) < 0.5 * np.std(vals) + 1e-9, f"mean={np.mean(vals):.3f}"

    naive = []
    for i, a in enumerate(LABELS):
        for b in LABELS[i + 1:]:
            d = src[a].mean(0) - src[b].mean(0)
            naive.append(d @ d)
    assert np.mean(naive) > 0, "the biased estimator really is positive here"


def test_crossnobis_separates_real_conditions():
    """Unbiasedness is worthless if the estimator cannot see a difference that is there."""
    rng = np.random.default_rng(11)
    src = _set(rng, _mus(rng, 30), n=120, noise=1.0)
    D = gf._crossnobis_within(src, np.random.default_rng(12), LABELS)
    vals = gf._triu_vals(D)
    assert np.nanmin(vals) > 0, "distinct conditions should be at positive distance"


def test_second_order_rdm_is_gain_invariant_and_cross_set_distance_is_not():
    """THE CLAIM THAT SEPARATES 8b FROM 6 AND 8, and the reason the pair is worth building.

    Figure 6's headline is that own-position similarity drops at EVERY position -- which is exactly
    what a uniform amplitude change would produce. 8b is the measure that cannot be fooled by one.
    """
    rng = np.random.default_rng(13)
    mus = _mus(rng, 40)
    pre = _set(rng, mus, n=120, noise=1.0)
    # post = the SAME geometry, every response scaled down. Nothing has moved relative to anything.
    gain = 0.4
    post = {q: v * gain for q, v in _set(rng, mus, n=120, noise=1.0).items()}

    Dpre = gf._crossnobis_within(pre, np.random.default_rng(14), LABELS)
    Dpost = gf._crossnobis_within(post, np.random.default_rng(15), LABELS)
    a, b = gf._triu_vals(Dpost), gf._triu_vals(Dpre)
    ok = np.isfinite(a) & np.isfinite(b)
    second_order = float(np.corrcoef(a[ok], b[ok])[0, 1])
    assert second_order > 0.9, f"a pure gain change must leave the RDM correlation alone: {second_order:.3f}"

    # the cross-set distance, by contrast, is moved by the same change
    same = gf._crossnobis_cross(_set(rng, mus, n=120, noise=1.0), pre,
                                np.random.default_rng(16), LABELS)
    scaled = gf._crossnobis_cross(post, pre, np.random.default_rng(16), LABELS)
    assert np.nanmean(np.diag(scaled)) > np.nanmean(np.diag(same)), (
        "a gain change SHOULD move the cross-set distance -- that is the exposure 8b exists to "
        "cover, and if it did not appear here the two figures would not be measuring different "
        "things")


def test_split_half_matrix_shape_and_missing_positions():
    rng = np.random.default_rng(17)
    src = _set(rng, _mus(rng, 25), n=40)
    src.pop("far_R")                                    # a position the lesion emptied
    M = gf._split_half_matrix(src, np.random.default_rng(18))
    assert M.shape == (6, 6)
    k = LABELS.index("far_R")
    assert np.all(np.isnan(M[k, :])) and np.all(np.isnan(M[:, k]))
    assert np.isfinite(M[0, 0])


@pytest.mark.parametrize("n", [0, 1, 3])
def test_estimators_return_nan_rather_than_raising_on_too_few_trials(n):
    """Post-stroke impaired positions really do fall to a handful of trials; a figure must blank
    those cells, not crash the render that also holds the other fifteen animals-windows."""
    rng = np.random.default_rng(19)
    Z = rng.normal(size=(n, 20))
    assert np.isnan(gf._split_half(Z, np.random.default_rng(20)))
    D = gf._crossnobis_within({q: Z for q in LABELS}, np.random.default_rng(21), LABELS)
    assert np.all(np.isnan(gf._triu_vals(D)))


def test_reliability_is_spearman_brown_corrected():
    """`_split_half` measures a HALF-length mean; figure 6 correlates the FULL-length one.

    Skipping the projection makes every reliability too low, and since 7b divides by
    sqrt(rel_post * rel_pre), too low a denominator makes every disattenuated correlation too high
    -- it would manufacture the "the code moved" verdict the panel exists to test.
    """
    rng = np.random.default_rng(23)
    mu = rng.normal(0, 1, size=40)
    Z = mu + rng.normal(0, 2.5, size=(120, 40))
    sh = gf._split_half(Z, np.random.default_rng(24))
    rel = gf._reliability(Z, np.random.default_rng(24))
    assert 0 < sh < 1
    assert rel > sh, "the full-length mean must be MORE reliable than a half-length one"
    assert abs(rel - 2 * sh / (1 + sh)) < 1e-9
    assert np.isnan(gf._reliability(np.zeros((2, 10)), np.random.default_rng(25)))


def test_split_half_matrix_is_symmetric():
    """M[P,Q] and M[Q,P] estimate ONE quantity -- how similar positions P and Q look within a
    session -- and differ only in which random half of each landed on which side.

    The first version of figure 7 drew corr(A_P, B_Q) and left it, so the matrix was visibly
    asymmetric and that asymmetry was pure estimation noise rendered as structure, in a figure whose
    whole job is to quantify estimation noise (Priya spotted it in the first render).
    """
    rng = np.random.default_rng(31)
    src = _set(rng, _mus(rng, 30), n=60, noise=1.5)
    M = gf._split_half_matrix(src, np.random.default_rng(32))
    fin = np.isfinite(M)
    assert fin.all()
    assert np.allclose(M, M.T, atol=1e-12), "off-diagonal must not depend on cell ordering"

    # the diagonal is still the two-half reliability, NOT 1: a mean correlated with itself is 1 by
    # definition and would make the whole figure vacuous
    assert np.all(np.diag(M) < 0.999)


def test_split_half_asymmetry_is_reported_not_drawn():
    """The discarded difference is a real noise read, so it stays available as a number."""
    rng = np.random.default_rng(33)
    noisy = _set(rng, _mus(rng, 30), n=20, noise=4.0)
    clean = _set(rng, _mus(rng, 30), n=400, noise=0.5)
    a_noisy = gf._split_half_asymmetry(noisy, np.random.default_rng(34))
    a_clean = gf._split_half_asymmetry(clean, np.random.default_rng(34))
    assert a_noisy > a_clean, f"noisier data must disagree more: {a_noisy:.3f} vs {a_clean:.3f}"
