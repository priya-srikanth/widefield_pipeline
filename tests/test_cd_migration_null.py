"""cd_migration_null: the trial-matched pre-vs-pre null on synthetic per-trial window means."""
from __future__ import annotations

import numpy as np

from wfield_local import cd_migration_null as m

POS = [1, 0, 2, 4, 3, 5]


def _tab(shift=0.0, seed=0, n_pre=8, n_post=2, n_pre_trials=60, n_post_trials=10):
    rng = np.random.default_rng(seed)
    tab = {}
    for k in range(n_pre):
        tab[f"PS00_08{k:02d}"] = {"epoch": "pre", "rows": {p: rng.normal(0, 1, (n_pre_trials, 6)) for p in POS}}
    for k in range(n_post):
        rows = {p: rng.normal(0, 1, (n_post_trials, 6)) for p in POS}
        rows[5][:, 3] += shift                    # far_R trials on far_L's direction
        tab[f"PS00_09{k:02d}"] = {"epoch": "acute", "rows": rows}
    return tab


def test_null_matches_trials_and_keeps_groups_disjoint():
    r = m.epoch_null(_tab(), POS, "acute", n_draw=50, rng=np.random.default_rng(1))
    assert r["n_E"].tolist() == [20.0] * 6
    assert (r["n_B"] == 20).all()                  # B trimmed to E's trial count in every row
    assert r["k_B"].max() <= 4                      # never more than half the pre sessions


def test_shift_clears_and_no_shift_does_not():
    i, j = POS.index(5), 3                        # row = position 5 trials, column 3 = the shifted column
    big = m.epoch_null(_tab(shift=2.0), POS, "acute", n_draw=200, rng=np.random.default_rng(2))
    assert big["frac_null_ge"][i, j] <= m.CLEAR_FRAC
    none = m.epoch_null(_tab(shift=0.0, seed=3), POS, "acute", n_draw=200, rng=np.random.default_rng(2))
    assert none["frac_null_ge"][i, j] > m.CLEAR_FRAC


def test_undersized_b_is_reported_not_hidden():
    r = m.epoch_null(_tab(n_post=6, n_post_trials=200), POS, "acute", n_draw=20, rng=np.random.default_rng(4))
    assert (r["match_frac"] < 1).all()             # 1200 target vs at most 4 x 60 = 240 in B
