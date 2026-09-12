"""The cluster-permutation test must not find structure where there is no brain.

WHY THIS FILE EXISTS. The first version of `beta_maps.cluster_permutation` computed
`t = |mean| / (se + 1e-12)` over the whole 540x640 frame. 138,387 of those pixels are outside the
brain, where every animal's map is ~0, so both the numerator and the denominator vanished and `t`
was unbounded. Priya, 2026-09-12, looking at the output: "this contour of significance looks like
total artifact ... hard to believe nothing else is significant?" -- and both halves were the same
bug, because the enormous edge clusters entered the permutation NULL as well and pushed the
cluster-mass threshold past every real interior effect.

The three properties pinned here are the three the fix rests on, and each one failed before it:

    1. nothing is ever flagged outside the brain mask
    2. a pure null flags nothing at all -- the test is calibrated, not merely quieter
    3. a real, localised, in-mask effect IS flagged -- it did not buy (1) and (2) with blindness
"""
import numpy as np
import pytest
from scipy import ndimage

from wfield_local import beta_maps as bm


def _mask():
    """A blob mask with a wide empty border -- the geometry the artefact needed."""
    m = np.zeros(bm.MAP_SHAPE, bool)
    m[120:420, 150:500] = True
    return m


def _noise(rng, mask, scale=1.0):
    """Spatially smooth noise, zero outside the mask -- what an off-brain pixel actually holds."""
    return ndimage.gaussian_filter(rng.standard_normal(bm.MAP_SHAPE), 8) * scale * mask


def _arms(rng, mask, n_animals=4, n_pre=4, n_post=3, effect=None):
    pre = {f"A{i}": [_noise(rng, mask) for _ in range(n_pre)] for i in range(n_animals)}
    post = {f"A{i}": [_noise(rng, mask) + (0 if effect is None else effect)
                      for _ in range(n_post)] for i in range(n_animals)}
    return pre, post


def test_nothing_is_flagged_outside_the_brain_mask():
    """The border artefact itself. Pixels with no brain in them cannot carry a cluster."""
    rng = np.random.default_rng(0)
    mask = _mask()
    pre, post = _arms(rng, mask)
    keep = bm.cluster_permutation(pre, post, n_perm=60, mask=mask)
    assert keep is not None
    assert not (keep & ~mask).any(), (
        f"{int((keep & ~mask).sum())} pixels flagged OUTSIDE the brain mask -- the edge artefact "
        "is back, and it suppresses interior effects as well as inventing border ones")


def test_a_pure_null_flags_nothing():
    """Calibration, not quietness: with no effect anywhere, the test must return an empty mask."""
    rng = np.random.default_rng(1)
    mask = _mask()
    pre, post = _arms(rng, mask)
    keep = bm.cluster_permutation(pre, post, n_perm=200, mask=mask)
    assert int(keep.sum()) == 0, f"{int(keep.sum())} pixels flagged under a pure null"


def test_a_real_localised_effect_survives():
    """The other half of the bargain -- the fix must not have been bought with blindness."""
    rng = np.random.default_rng(2)
    mask = _mask()
    effect = np.zeros(bm.MAP_SHAPE)
    effect[200:300, 250:380] = 3.0
    effect *= mask
    pre, post = _arms(rng, mask, effect=effect)
    keep = bm.cluster_permutation(pre, post, n_perm=200, mask=mask)
    assert keep[240, 300], "the planted effect was not detected"
    assert not (keep & ~mask).any()
    # And it stays LOCALISED: the flagged area should look like the effect, not like the mask.
    assert keep.sum() < 0.6 * mask.sum(), (
        f"{int(keep.sum())} of {int(mask.sum())} in-mask pixels flagged for a "
        f"{int((effect > 0).sum())}-pixel effect")


def test_the_denominator_floor_stops_a_zero_variance_pixel_from_seeding_a_cluster():
    """The border bug's interior twin: four animals agreeing exactly is not infinite evidence.

    A pixel where every animal returns the same value has se = 0. Without a floor its `t` is
    whatever `1e-12` allows, which is enormous, and that is true INSIDE the brain too -- so the
    mask alone does not fix it. `SE_FLOOR_PCT` is the second half of the fix.
    """
    mask = _mask()
    rng = np.random.default_rng(3)
    pre, post = _arms(rng, mask)
    # A tiny patch where all animals are identical pre and identical post, differing by a hair.
    for an in pre:
        for m in pre[an]:
            m[300:306, 300:306] = 0.0
        for m in post[an]:
            m[300:306, 300:306] = 1e-6
    keep = bm.cluster_permutation(pre, post, n_perm=120, mask=mask)
    assert not keep[303, 303], (
        "a zero-variance pixel with a 1e-6 difference was called significant -- the denominator "
        "floor is not doing its job")


def test_it_refuses_to_run_without_a_mask_rather_than_falling_back():
    """A silent fallback to the whole frame is indistinguishable from the fix working."""
    rng = np.random.default_rng(4)
    pre, post = _arms(rng, _mask())
    with pytest.raises(ValueError, match="brain mask"):
        bm.cluster_permutation(pre, post, n_perm=5, mask=np.zeros(bm.MAP_SHAPE, bool))


def test_the_cluster_forming_threshold_comes_from_the_df():
    """t=2.0 is p=0.14 at df=3. The threshold must track the number of animals, not be a constant.

    Pinned because the hard-coded 2.0 did not look wrong -- it looks like a conventional choice,
    and the only symptom was 18.8% of the brain above threshold for a position with no change.
    """
    from scipy import stats

    for n in (3, 4, 5):
        want = float(stats.t.ppf(1 - bm.CLUSTER_FORMING_P / 2, n - 1))
        assert want > 2.0, f"df={n - 1} threshold {want:.2f} is no stricter than the old 2.0"
    assert abs(float(stats.t.ppf(1 - bm.CLUSTER_FORMING_P / 2, 3)) - 3.182) < 0.01


def test_positive_and_negative_clusters_are_labelled_apart():
    """An increase and a decrease that touch must not be summed into one cluster.

    Under `|t|` they merge wherever they meet, and the merged mass then clears a threshold neither
    would clear alone -- so a strong increase can carry an adjacent decrease over the line with it.
    """
    rng = np.random.default_rng(5)
    mask = _mask()
    effect = np.zeros(bm.MAP_SHAPE)
    effect[200:300, 200:330] = 4.0        # a large increase
    effect[200:300, 330:340] = -0.30      # a weak decrease, touching it
    effect *= mask
    pre, post = _arms(rng, mask, effect=effect)
    keep = bm.cluster_permutation(pre, post, n_perm=200, mask=mask)
    assert keep[250, 260], "the strong increase should survive"
    assert not keep[250, 334], (
        "the weak adjacent decrease was carried over the threshold by the increase it touches -- "
        "the polarities are being merged")
