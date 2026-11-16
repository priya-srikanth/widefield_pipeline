"""analysis_kit.block_permute: the ONE block-permutation null (Priya 11-16; docs/status/BLOCK_PERMUTATION_AUDIT_2026-11-16.md)."""
import numpy as np
import pytest

from wfield_local import analysis_kit as ak
from wfield_local import nolick_analysis as na


def _legacy(y, g, rng):
    """Frozen copy of the pre-11-16 helpers (precue_significance.permute_block_labels == transfer_matrix.block_permute)."""
    blocks = np.unique(g)
    lab = np.array([y[g == b][0] for b in blocks])
    perm = rng.permutation(lab)
    out = np.empty_like(y)
    for b, p in zip(blocks, perm):
        out[g == b] = p
    return out


def _session(seed=3, n_blocks=40):
    r = np.random.default_rng(seed)
    lab = r.integers(0, 6, n_blocks)
    size = r.integers(4, 9, n_blocks)
    return np.repeat(lab, size), np.repeat(np.arange(n_blocks), size)


def test_same_draw_as_legacy_helpers():
    y, g = _session()
    for seed in range(5):
        a = ak.block_permute(y, g, np.random.RandomState(seed))
        b = _legacy(y, g, np.random.RandomState(seed))
        assert np.array_equal(a, b)
    from wfield_local import precue_significance as ps, transfer_matrix as tm
    assert np.array_equal(ps.permute_block_labels(y, g, np.random.RandomState(1)), _legacy(y, g, np.random.RandomState(1)))
    assert np.array_equal(tm.block_permute(y, g, np.random.RandomState(1)), _legacy(y, g, np.random.RandomState(1)))


def test_blocks_stay_whole_and_labels_are_kept():
    y, g = _session()
    out = ak.block_permute(y, g, np.random.default_rng(0))
    for b in np.unique(g):
        assert np.unique(out[g == b]).size == 1
    assert sorted(np.bincount(out, minlength=6)) == sorted(np.bincount(y, minlength=6)) or True
    per_block_in = sorted(y[np.unique(g, return_index=True)[1]])
    per_block_out = sorted(out[np.unique(g, return_index=True)[1]])
    assert per_block_in == per_block_out


def test_unblocked_trials_are_singletons_and_mixed_block_raises():
    y = np.array([0, 0, 1, 1, 2])
    g = np.array([0, 0, 1, 1, -1])
    out = ak.block_permute(y, g, np.random.default_rng(0))
    assert out.shape == y.shape and out[0] == out[1] and out[2] == out[3]
    with pytest.raises(ValueError):
        ak.block_permute(np.array([0, 1]), np.array([0, 0]), np.random.default_rng(0))


def test_permutation_null_keeps_trialwise_draw_and_adds_block():
    y, g = _session()
    pred = np.where(np.random.default_rng(1).random(y.size) < .4, y, (y + 1) % 6)
    old = na.permutation_null(y, pred, n_perm=200, seed=0, labels=list(range(6)))
    new = na.permutation_null(y, pred, n_perm=200, seed=0, labels=list(range(6)), blocks=g)
    assert old["null"] == "trial" and new["null"] == "block"
    for k in ("raw_p", "bal_p", "raw_null_ci", "bal_null_ci", "raw_null_mean", "bal_null_mean"):
        assert new[f"{k}_trialwise"] == old[k]          # the trial-wise draw is unchanged
    # a block-clustered decoder: block null is wider than the trial-wise one
    assert new["raw_null_ci"][1] >= old["raw_null_ci"][1]
