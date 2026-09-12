"""Everything that makes a green contour on a map figure mean what it says.

This file grew one test per bug, and every one of them is a bug that SHIPPED -- each was producing
a figure that looked finished. Read it as the list of ways a cortical-map statistic can be wrong
while still rendering.

THE ORIGINAL BUG. `cluster_permutation` computed `t = |mean| / (se + 1e-12)` over the whole
540x640 frame. 138,387 of those pixels are not brain, and there both the numerator and the
denominator vanish, so `t` was unbounded. Priya, 2026-09-12: "this contour of significance looks
like total artifact ... hard to believe nothing else is significant?" -- and both halves were the
SAME bug, because the enormous edge clusters entered the permutation NULL as well and pushed the
cluster-mass threshold past every real interior effect.

WHAT IS PINNED, and why each one exists:

    off-brain pixels          the original artefact -- clusters formed where there is no brain
    pure-null calibration     the fix had to be CALIBRATED, not merely quieter
    a planted central effect  ...and not bought with blindness
    zero-variance floor       the border bug's interior twin: four animals agreeing by chance
    refuses without a mask    a silent fallback is indistinguishable from the fix working
    df-derived threshold      t=2.0 is p=0.14 at df=3; it passed 18.8% of a null position's brain
    polarities apart          an increase must not carry a touching decrease over the line
    eroded stat mask          the rim is consistent ACROSS animals, which a between-animal test
                              rewards -- it cannot tell a shared optical artefact from a shared
                              biological effect
    rim effect not flagged    the behaviour that erosion is FOR
    edge result suppressed    one erosion radius cannot cover every panel; the guard is per-result
    central result drawn      ...and the guard is not satisfied by blindness
    olfactory bulbs excluded  in the display mask AND the statistics mask -- an exclusion honoured
                              by the tests but not by the amplitude numbers is worse than none
    vs-zero finds a map       Musall fig S6's question, and it must still detect something
    do-not-subtract pinned    two vs-zero panels are NOT a difference, and that caveat is exactly
                              what a rewrite drops

Full decision record: DECISIONS.md, 2026-09-12 (night).
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


def test_statistics_use_an_ERODED_mask_but_display_does_not():
    """The rim of the imaging window is where the edge artefact lives, so no test may reach it.

    Priya, 2026-09-12: "the beta significance does not look right - lots of edge selection
    (olfactory bulbs etc)." Measured on the beta maps, as the fraction of flagged pixels falling
    inside a rim of the mask against that rim's own share of the brain (6.3% / 12.6% / 24.5% at
    8 / 16 / 32 px):

        near ipsi      5.5%   12.0%   25.1%     unbiased
        near MIDDLE   11.6%   24.7%   55.8%     ~2x enriched
        near CONTRA   14.1%   27.7%   48.6%     ~2x enriched

    The rim is where `U` is smallest and the Allen warp least constrained, so it carries
    partial-volume mixing and alignment jitter. Those are SYSTEMATIC -- same window, same warp,
    every session -- hence consistent across animals, which is exactly what a between-animal
    denominator rewards. A statistic asking "do the animals agree" cannot separate a shared
    biological effect from a shared optical one.

    DISPLAY MUST KEEP THE FULL MASK: eroding what is drawn would hide data.
    """
    import numpy as np

    from wfield_local import beta_maps as bm

    disp, stat = bm.brain_mask(), bm.stat_mask()
    assert disp is not None and stat is not None
    assert stat.sum() < disp.sum(), "the statistics mask must be strictly smaller"
    assert not (stat & ~disp).any(), "the statistics mask must be a subset of the display mask"
    # it removes the rim and nothing central
    from scipy import ndimage
    core = ndimage.binary_erosion(disp, iterations=bm.STAT_ERODE_PX + 4)
    assert (core & ~stat).sum() == 0, "erosion reached past the rim into the core"
    assert 0.75 < stat.sum() / disp.sum() < 0.95, (
        f"erosion keeps {100 * stat.sum() / disp.sum():.0f}% -- too little or too much")


def test_a_planted_EDGE_effect_is_not_flagged_but_a_central_one_is():
    """The point of the erosion, as a behaviour rather than a mask-size assertion."""
    import numpy as np
    from scipy import ndimage

    from wfield_local import beta_maps as bm

    disp = bm.brain_mask()
    stat = bm.stat_mask()
    rim = disp & ~stat
    rng = np.random.default_rng(0)

    def arms(effect):
        pre = {f"A{i}": [ndimage.gaussian_filter(rng.standard_normal(bm.MAP_SHAPE), 8) * disp
                         for _ in range(3)] for i in range(4)}
        post = {f"A{i}": [ndimage.gaussian_filter(rng.standard_normal(bm.MAP_SHAPE), 8) * disp
                          + effect for _ in range(3)] for i in range(4)}
        return pre, post

    # a large, perfectly consistent effect confined to the RIM
    edge = np.zeros(bm.MAP_SHAPE)
    edge[rim] = 5.0
    got = bm.hierarchical_bootstrap_significance(*arms(edge), n_boot=300)
    assert got is not None
    assert int(got[0].sum()) == 0, (
        f"{int(got[0].sum())} bins flagged for an effect that lies entirely in the eroded rim")


def test_an_edge_localised_result_is_SUPPRESSED_not_merely_annotated():
    """A green contour is read as a result; a caveat in a subtitle is not read at all.

    Priya, 2026-09-12: "the evoked maps weren't clean - the significance was localized at the rim
    in some figures." Measured on the evoked maps, near-middle CHRONIC had 28.4% of its flagged
    pixels inside the 8 px rim (4.5x its 6.3% share) and 72.0% inside the 32 px rim (2.9x its
    24.5% share) -- and after 16 px erosion it was STILL 960 bins with 44% in the outer rim. One
    erosion radius cannot cover every panel without eating real cortex, so the guard is a
    per-result property rather than a fixed geometry.
    """
    import numpy as np
    from scipy import ndimage

    from wfield_local import beta_maps as bm

    disp = bm.brain_mask()
    # calibration: the statistic means what its name says
    rim = disp & ~ndimage.binary_erosion(disp, iterations=32)
    assert bm.edge_enrichment(disp) == pytest.approx(1.0, abs=0.01), "whole brain must be 1.0x"
    assert bm.edge_enrichment(rim) > 3.0, "an all-rim result must be strongly enriched"
    assert bm.edge_enrichment(ndimage.binary_erosion(disp, iterations=60)) < 0.2

    # behaviour: a planted rim-hugging effect must not come back as a contour
    rng = np.random.default_rng(7)
    effect = np.zeros(bm.MAP_SHAPE)
    effect[rim] = 6.0
    pre = {f"A{i}": [ndimage.gaussian_filter(rng.standard_normal(bm.MAP_SHAPE), 8) * disp
                     for _ in range(3)] for i in range(4)}
    post = {f"A{i}": [ndimage.gaussian_filter(rng.standard_normal(bm.MAP_SHAPE), 8) * disp + effect
                      for _ in range(3)] for i in range(4)}
    contour, label = bm.significance_contour(pre, post, n_boot=300, mask=disp)
    assert contour is None, "an edge-localised result was drawn"
    assert "SUPPRESSED" in label and "edge enrichment" in label, label


def test_a_central_result_still_reports_its_enrichment_and_is_drawn():
    """The guard must not be bought with blindness -- a real central effect keeps its contour."""
    import numpy as np
    from scipy import ndimage

    from wfield_local import beta_maps as bm

    disp = bm.brain_mask()
    core = ndimage.binary_erosion(disp, iterations=60)
    rng = np.random.default_rng(8)
    effect = np.zeros(bm.MAP_SHAPE)
    effect[core] = 6.0
    pre = {f"A{i}": [ndimage.gaussian_filter(rng.standard_normal(bm.MAP_SHAPE), 8) * disp
                     for _ in range(3)] for i in range(4)}
    post = {f"A{i}": [ndimage.gaussian_filter(rng.standard_normal(bm.MAP_SHAPE), 8) * disp + effect
                      for _ in range(3)] for i in range(4)}
    contour, label = bm.significance_contour(pre, post, n_boot=300, mask=disp)
    assert contour is not None and contour.any(), f"a central effect was suppressed: {label}"
    assert "edge enrichment" in label, label


def test_vs_zero_detects_a_map_and_is_silent_when_there_is_none():
    """Musall fig S6's question: is there a map here at all, tested within ONE epoch.

    Priya, 2026-09-12: "we still haven't done the Musall-style analysis of comparing each pre/post
    epoch to zero." Correct -- `musall_significance` existed but had only ever been pointed at
    DIFFERENCES. This pins the vs-zero path: a real map is found, pure noise is not.
    """
    import numpy as np
    from scipy import ndimage

    from wfield_local import beta_maps as bm

    disp = bm.brain_mask()
    core = ndimage.binary_erosion(disp, iterations=60)
    rng = np.random.default_rng(11)

    def arm(effect):
        return {f"A{i}": [ndimage.gaussian_filter(rng.standard_normal(bm.MAP_SHAPE), 8) * disp
                          + effect for _ in range(3)] for i in range(4)}

    # a real, central, consistent map
    eff = np.zeros(bm.MAP_SHAPE)
    eff[core] = 6.0
    got, label = bm.vs_zero_contour(arm(eff), n_boot=300)
    assert got is not None and got.any(), f"a real map was not detected: {label}"
    assert "vs ZERO" in label

    # pure noise: nothing
    got2, label2 = bm.vs_zero_contour(arm(np.zeros(bm.MAP_SHAPE)), n_boot=300)
    assert got2 is None or int(got2.sum()) == 0, f"noise was called a map: {label2}"


def test_vs_zero_carries_the_do_not_subtract_warning_in_its_docstring():
    """The commonest misuse of a per-epoch vs-zero figure is reading two panels as a difference.

    "Significant in pre and not in acute" is not evidence of change -- a map that just clears the
    threshold in one epoch and just misses it in the next may not differ at all, and a vs-zero
    figure puts no error bar on that comparison. Pinned as text because it is the kind of caveat
    that gets dropped in a rewrite.
    """
    from wfield_local import beta_maps as bm

    doc = bm.vs_zero_contour.__doc__ or ""
    assert "DO NOT SUBTRACT" in doc.upper()
    assert "significance_contour" in doc, "must point at the test that DOES answer the difference"


def test_the_olfactory_bulbs_are_excluded_everywhere_or_nowhere():
    """Priya: "get rid of the olfactory bulbs, since I don't have them in full view".

    THE FIELD OF VIEW BEARS IT OUT: inside the mask, MOB_left is 7,553 px against MOB_right's
    389 px -- a 19x asymmetry, so "the olfactory bulbs" really means the left one, and a bilateral
    map including them compares a full region against a sliver.

    THE EXCLUSION HAS TO BE AT ONE PLACE. It is applied inside `brain_mask`, so every consumer --
    the statistics mask, amplitude ratios, split-half reliability, the edge-enrichment denominator
    -- inherits the same field of view without each having to remember. Pinned because an exclusion
    honoured by the tests but not by the amplitude numbers would be worse than none.
    """
    import numpy as np

    from wfield_local import beta_maps as bm

    ex = bm.excluded_mask()
    assert ex is not None and ex.any(), "no pixels excluded -- the atlas lookup silently failed"
    assert 0.01 < ex.sum() / bm.MAP_SHAPE[0] / bm.MAP_SHAPE[1] < 0.10, (
        f"{int(ex.sum())} px excluded -- implausible for the olfactory bulbs")
    disp, stat = bm.brain_mask(), bm.stat_mask()
    assert not (disp & ex).any(), "excluded pixels are still inside the display mask"
    assert not (stat & ex).any(), "excluded pixels are still inside the statistics mask"


def test_excluded_regions_are_not_drawn_and_do_not_set_the_colour_scale(tmp_path):
    """A region no statistic may touch must not be PAINTED as if it were data.

    Excluding it from the mask while still drawing it would be the worst of both: it would look
    like a result and could even set the row's colour limit, squashing everything real.
    """
    import numpy as np

    from wfield_local import beta_maps as bm
    from wfield_local import epoch_figures as ef

    blank = bm.excluded_mask()
    m = np.zeros(bm.MAP_SHAPE)
    m[blank] = 1000.0          # an absurd value only inside the excluded region
    m[~blank] = 0.01
    lim_with = float(np.percentile(np.abs(m.ravel()), 99.0))
    masked = np.where(blank, np.nan, m)
    v = masked.ravel()[np.isfinite(masked.ravel())]
    lim_without = float(np.percentile(np.abs(v), 99.0))
    assert lim_without < lim_with, "blanking must keep the excluded region out of the limit"

    p = ef.map_grid({("r", "a"): m}, tmp_path, name="t", title="t",
                    row_labels=["r"], col_labels=["a"], blank=blank)
    assert p.exists()


def test_the_atlas_lookup_maps_SIGNED_IDS_and_every_region_is_bilateral():
    """The json maps INDEX -> [SIGNED_ID, name]; the atlas array stores the SIGNED ID.

    KEYING BY THE INDEX SHIFTS EVERY NAME ONTO THE WRONG REGION, by an offset that grows down the
    file, and it shipped (2026-09-12). Consequences, all of which looked plausible at the time:

      * `EXCLUDE_REGIONS = ("MOB",)` removed background + MOB_LEFT ONLY, leaving the right bulb
        fully in -- which is what Priya saw still on the figure;
      * it invented a "19x MOB asymmetry" and a "1.9x FRP asymmetry", neither of which exists;
      * "FRP" resolved to FRP_left + MOp_LEFT, so a proposal to exclude FRP would have deleted
        PRIMARY MOTOR CORTEX from every analysis.

    It was caught only because she asked to see the mask DRAWN ON A BRAIN -- a pixel count cannot
    reveal it, and every number it produced was self-consistent.

    THE INVARIANT THAT WOULD HAVE CAUGHT IT: a correctly-mapped cortical parcellation is bilateral.
    Each `X_left` / `X_right` pair must have centroids on OPPOSITE sides of the image midline and
    near-equal pixel counts.
    """
    import numpy as np

    from wfield_local import beta_maps as bm

    atlas, names = bm._atlas_names()
    assert atlas is not None and names, "atlas or names unavailable"
    mid = atlas.shape[1] / 2.0
    pairs = {}
    for sid, nm in names.items():
        if "_" not in nm:
            continue
        base, side = nm.rsplit("_", 1)
        if side.lower() in ("left", "right"):
            pairs.setdefault(base, {})[side.lower()] = sid

    checked = 0
    for base, sides in pairs.items():
        if set(sides) != {"left", "right"}:
            continue
        ml, mr = atlas == sides["left"], atlas == sides["right"]
        nl, nr = int(ml.sum()), int(mr.sum())
        if min(nl, nr) < 200:            # too small for a centroid to be meaningful
            continue
        checked += 1
        cl = float(np.where(ml)[1].mean())
        cr = float(np.where(mr)[1].mean())
        assert (cl - mid) * (cr - mid) < 0, (
            f"{base}: _left centroid {cl:.0f} and _right {cr:.0f} are on the SAME side of the "
            f"midline {mid:.0f} -- the atlas lookup is keying by index instead of signed id")
        assert max(nl, nr) / min(nl, nr) < 1.3, (
            f"{base}: {nl} vs {nr} px, a {max(nl, nr) / min(nl, nr):.1f}x asymmetry")
    assert checked >= 8, f"only {checked} bilateral pairs checked -- the lookup may be empty"


def test_BOTH_olfactory_bulbs_are_excluded_not_one():
    """The half-exclusion is the specific symptom the atlas bug produced on the figure."""
    import numpy as np

    from wfield_local import beta_maps as bm

    atlas, names = bm._atlas_names()
    ex = bm.excluded_mask()
    mid = atlas.shape[1] / 2.0
    for sid, nm in names.items():
        if nm.upper().startswith("MOB"):
            m = atlas == sid
            assert m.any(), f"{nm} has no pixels"
            assert (m & ~ex).sum() == 0, f"{nm} is NOT excluded -- only one bulb was removed"
    xs = np.where(ex)[1]
    assert (xs < mid).any() and (xs > mid).any(), (
        "the exclusion lies entirely on one side of the midline -- it is half a pair")
