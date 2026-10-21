"""The map significance tests see the scale the pooled picture is drawn on (`beta_maps.NORMALISE_TESTS_BY_PRE`,
2026-10-21; DECISIONS 10-21 afternoon, map statistics entry).

    one constant              `pre_scale` IS the constant `within_animal_pooled` divides by -- one definition
    brightness-invariant      multiplying one animal's maps (pre AND post) by a constant cannot move a normalised
                              contour, for the nested bootstrap, the cluster permutation and vs-zero alike
    raw still available       normalise=False reproduces the old raw-dF contour exactly
    labelled                  a normalised contour says so ("[pre-normalised]")

The factor is a power of two so the scaled and unscaled draws are bit-identical, not merely close.
"""
import numpy as np
from scipy import ndimage

from wfield_local import beta_maps as bm


def _noise(rng, scale=1.0):
    return ndimage.gaussian_filter(rng.standard_normal(bm.MAP_SHAPE), 8) * scale


def _arms(seed, effect=0.0):
    rng = np.random.default_rng(seed)
    bump = np.zeros(bm.MAP_SHAPE)
    bump[220:320, 250:380] = effect
    pre = {f"A{i}": [_noise(rng) + 0.5 for _ in range(4)] for i in range(4)}
    post = {f"A{i}": [_noise(rng) + 0.5 + bump for _ in range(3)] for i in range(4)}
    return pre, post


def _brighter(by, an, k=4.0):
    return {a: [m * (k if a == an else 1.0) for m in v] for a, v in by.items()}


def test_pre_scale_is_the_constant_the_pooled_picture_uses():
    pre, post = _arms(0)
    sc = bm.pre_scale(pre)
    pre_p, post_p, _d, _r, animals = bm.within_animal_pooled(pre, post)
    want = np.mean([np.mean(post[a], axis=0) / sc[a] for a in animals], axis=0)
    np.testing.assert_array_equal(post_p, want)


def test_pre_scale_leaves_a_degenerate_animal_unscaled():
    z = np.zeros(bm.MAP_SHAPE)
    assert bm.pre_scale({"A": [z, z]})["A"] == 1.0


def test_normalised_nested_contour_ignores_one_animals_brightness():
    pre, post = _arms(1, effect=3.0)
    k0, lab0 = bm.significance_contour(pre, post, method="nested", n_boot=300, normalise=True)
    k1, _ = bm.significance_contour(_brighter(pre, "A0"), _brighter(post, "A0"), method="nested", n_boot=300,
                                    normalise=True)
    assert k0 is not None and k0.any(), "the planted effect should be found"
    np.testing.assert_array_equal(k0, k1)
    assert lab0.startswith("[pre-normalised]")


def test_normalised_cluster_contour_ignores_one_animals_brightness():
    pre, post = _arms(2, effect=3.0)
    k0, lab0 = bm.significance_contour(pre, post, method="cluster", n_perm=60, normalise=True)
    k1, _ = bm.significance_contour(_brighter(pre, "A0"), _brighter(post, "A0"), method="cluster", n_perm=60,
                                    normalise=True)
    assert k0 is not None and k0.any()
    np.testing.assert_array_equal(k0, k1)
    assert lab0.startswith("[pre-normalised]")


def test_raw_contour_is_unchanged_when_normalise_is_off():
    pre, post = _arms(3, effect=3.0)
    got, lab = bm.significance_contour(pre, post, method="nested", n_boot=300, normalise=False)
    ref = bm._significance_contour(pre, post, method="nested", n_boot=300, n_perm=500, alpha=0.05, seed=0,
                                   mask=None)
    np.testing.assert_array_equal(got, ref[0])
    assert not lab.startswith("[pre-normalised]")


def test_cluster_normalise_off_matches_the_original_statistic():
    pre, post = _arms(4, effect=3.0)
    a = bm.cluster_permutation(pre, post, n_perm=40)
    b = bm.cluster_permutation(pre, post, n_perm=40, normalise=False)
    np.testing.assert_array_equal(a, b)


def test_vs_zero_scale_by_ignores_one_animals_brightness():
    rng = np.random.default_rng(5)
    bump = np.zeros(bm.MAP_SHAPE)
    bump[220:320, 250:380] = 3.0
    pre = {f"A{i}": [_noise(rng) + 0.5 for _ in range(4)] for i in range(4)}
    by = {a: [_noise(rng) + bump for _ in range(3)] for a in pre}
    k0, lab0 = bm.vs_zero_contour(by, n_boot=300, scale_by=pre)
    k1, _ = bm.vs_zero_contour(_brighter(by, "A1"), n_boot=300, scale_by=_brighter(pre, "A1"))
    np.testing.assert_array_equal(k0, k1)
    assert lab0.startswith("[pre-normalised]")
