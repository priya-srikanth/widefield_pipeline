"""Chronic fix (Priya 11-26): the epoch is subsampled to B's trial count when it holds more than pre can supply."""
from __future__ import annotations

import numpy as np

from wfield_local import cd_migration_null as mig
from wfield_local import cd_overlap_null as ov

POS = [1, 0, 2, 4, 3, 5]


def test_stratified_fold_subsample_hits_target_and_keeps_every_session():
    rng = np.random.default_rng(0)
    sessions = [(f"s{j}", [(np.zeros((3, 4)), 10) for _ in range(8)]) for j in range(6)]   # 480 trials
    chunks, n, k = ov.stratified_fold_subsample(sessions, 200, rng)
    assert abs(n - 200) <= 10 and k == 6 and sum(c[1] for c in chunks) == n


def test_stratified_trials_is_proportional():
    rng = np.random.default_rng(1)
    parts = [np.ones((100, 6)), 2 * np.ones((300, 6))]
    got = mig.stratified_trials(parts, 40, rng)
    assert got.shape == (40, 6) and int((got[:, 0] == 1).sum()) == 10


def _tab(n_post=6, n_post_trials=200, seed=0):
    rng = np.random.default_rng(seed)
    tab = {f"PS00_08{k:02d}": {"epoch": "pre", "rows": {p: rng.normal(0, 1, (60, 6)) for p in POS}} for k in range(8)}
    for k in range(n_post):
        tab[f"PS00_09{k:02d}"] = {"epoch": "chronic", "rows": {p: rng.normal(0, 1, (n_post_trials, 6)) for p in POS}}
    return tab


def test_match_epoch_shares_the_trial_count_and_leaves_b_unchanged():
    a = mig.epoch_null(_tab(), POS, "chronic", n_draw=30, rng=np.random.default_rng(3))
    b = mig.epoch_null(_tab(), POS, "chronic", n_draw=30, rng=np.random.default_rng(3), match_epoch=True)
    assert np.array_equal(a["n_B"], b["n_B"])                 # B's draws are those of the unmatched run
    assert np.array_equal(b["n_E_sub"], b["n_B"])              # observed and null share one count
    assert b["epoch_subsampled_row"].all()
    np.testing.assert_allclose(a["null"], b["null"])


def test_match_epoch_is_a_no_op_when_b_can_cover_the_epoch():
    t = _tab(n_post=1, n_post_trials=20)
    a = mig.epoch_null(t, POS, "chronic", n_draw=30, rng=np.random.default_rng(3))
    b = mig.epoch_null(t, POS, "chronic", n_draw=30, rng=np.random.default_rng(3), match_epoch=True)
    np.testing.assert_array_equal(a["obs"], b["obs"])
    assert not b["epoch_subsampled_row"].any()
