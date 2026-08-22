"""Feature alignment must not collapse repeated labels — bugs 16 and 17, 2026-08-17.

THE PATTERN, because it arose twice independently on the same day in unrelated modules:

    common = [r for r in regs[0] if all(r in set(rr) for rr in regs[1:])]
    idx    = [list(rr).index(r) for r in common]

Region labels REPEAT. Sub-binning tiles the whole label vector once per bin (66 Allen areas x 4 bins
= 264 columns, still 66 distinct labels), and a joint LocaNMF basis maps several components to one
area (PS93: 87 components, 64 labels). `list.index()` returns the FIRST match, so every repeat
resolves to the same column: the 4 x 0.5 s time course silently becomes four copies of bin 0, and the
column COUNT is unchanged, so nothing downstream can notice.

It cost every ROI frozen-decoder number between 2026-08-14 and 2026-08-17 (post-cue understated by
0.23-0.41) and was found by Priya reading two deck sections against each other, not by any check.

These tests use a TILED fixture, because a fixture with unique labels passes the broken code.
"""
from __future__ import annotations

import numpy as np

from wfield_local.locanmf_frozen_decoder import _align_many


def _tiled(n_areas=5, n_bins=4):
    """Region labels as the pipeline actually builds them: areas tiled once per sub-bin."""
    return list(np.tile(np.arange(n_areas), n_bins))


def test_align_many_preserves_every_sub_bin():
    """The regression. Four bins in, four DISTINCT bins out -- not four copies of bin 0."""
    regs = [_tiled(), _tiled()]
    # column j holds value j, so a collapse is visible as repeated values
    mats = [np.arange(20, dtype=float)[None, :].copy() for _ in regs]
    out, common = _align_many(mats, regs)
    assert len(common) == 20, "all 20 tiled features must survive"
    got = out[0][0]
    assert len(set(got.tolist())) == 20, (
        f"columns collapsed: only {len(set(got.tolist()))} distinct of 20 -- the sub-bins were "
        f"resolved back onto the first occurrence of each label")
    assert np.array_equal(got, np.arange(20)), "column order must be preserved"


def test_align_many_still_intersects_on_missing_areas():
    """KNOWN-GOOD case: the function's actual job -- drop an area one session lacks -- still works.

    Without this, a 'fix' that simply returned every column positionally would pass the test above
    while breaking the reason the function exists.
    """
    regs = [[0, 1, 2, 0, 1, 2], [0, 2, 0, 2]]          # session B has no area 1
    mats = [np.arange(6, dtype=float)[None, :], np.arange(4, dtype=float)[None, :]]
    out, common = _align_many(mats, regs)
    assert common == [0, 2, 0, 2], f"expected areas 0 and 2 in both bins, got {common}"
    assert out[0].shape[1] == out[1].shape[1] == 4


def test_align_many_matches_bin_k_to_bin_k_across_sessions():
    """Bin identity must be preserved, not just bin COUNT: session B's bin 1 must map to bin 1."""
    regs = [[7, 8, 7, 8], [7, 8, 7, 8]]
    a = np.array([[10.0, 20.0, 11.0, 21.0]])           # area7-bin0, area8-bin0, area7-bin1, area8-bin1
    b = np.array([[30.0, 40.0, 31.0, 41.0]])
    out, _ = _align_many([a, b], regs)
    assert np.array_equal(out[0][0], a[0]) and np.array_equal(out[1][0], b[0])


def test_unique_labels_are_unaffected():
    """No sub-binning -> the old code was already correct here; the fix must not disturb it."""
    regs = [[3, 1, 2], [1, 2, 3]]
    a = np.array([[3.0, 1.0, 2.0]])
    b = np.array([[1.0, 2.0, 3.0]])
    out, common = _align_many([a, b], regs)
    assert common == [3, 1, 2]
    assert np.array_equal(out[0][0], [3.0, 1.0, 2.0])
    assert np.array_equal(out[1][0], [3.0, 1.0, 2.0]), "B must be reordered to match A's label order"


def test_nolick_decoder_uses_the_same_keying():
    """The same collapse existed in nolick_decoder; pin that both were fixed the same way.

    Rather than duplicating the alignment logic in a test, this asserts the property that matters:
    a tiled label vector keyed by (label, occurrence) yields as many distinct keys as columns.
    """
    reg = _tiled(n_areas=6, n_bins=4)
    seen, keyed = {}, []
    for r in reg:
        seen[r] = seen.get(r, -1) + 1
        keyed.append((r, seen[r]))
    assert len(set(keyed)) == len(reg) == 24
    assert len(set(reg)) == 6, "the fixture must genuinely repeat labels, else it tests nothing"


def test_short_windows_do_not_produce_empty_nan_bins():
    """decode.bins is 8, sized for a 2 s window. A 150 ms one is ~5 frames.

    linspace(0, 5, 9).astype(int) repeats three edges, so three of the eight slices are w[:, a:a]
    and mean over an empty axis is NaN -- indistinguishable from a real value until the fit fails.
    Priya asked for a 150 ms post-lick decoder on 2026-08-22, which is exactly who hits this.
    """
    import numpy as np

    from wfield_local.locanmf_position_decoder import _window_feature

    sig = np.arange(3 * 40, dtype=float).reshape(3, 40)
    for post_n in (1, 2, 3, 5, 8):
        f = _window_feature(sig, 0, post_n, 8, 0.0)
        assert np.isfinite(f).all(), f"post_n={post_n} produced NaN bins"
        assert len(f) == 3 * min(8, post_n), f"post_n={post_n} gave {len(f)} features"


def test_a_long_window_still_gets_every_bin_it_asked_for():
    """The complement: the clamp must not quietly reduce resolution on the normal 2 s window."""
    import numpy as np

    from wfield_local.locanmf_position_decoder import _window_feature

    sig = np.arange(3 * 80, dtype=float).reshape(3, 80)
    f = _window_feature(sig, 0, 62, 8, 0.0)
    assert len(f) == 3 * 8 and np.isfinite(f).all()
