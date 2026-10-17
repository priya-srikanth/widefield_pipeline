"""orofacial_stats: BH-FDR, session MWU table, cluster permutation."""
import numpy as np
import pandas as pd

from wfield_local import orofacial_stats as st


def test_bh_fdr_matches_hand_computation():
    q = st.bh_fdr([0.01, 0.04, 0.03, np.nan, 0.5])
    # sorted p: .01 .03 .04 .5 (m = 4) -> .04 .06 .0533 .5 -> monotone from the top: .04 .0533 .0533 .5
    assert np.allclose(q[[0, 2, 1, 4]], [0.04, 0.0533333, 0.0533333, 0.5])
    assert np.isnan(q[3])


def _panel(shift: float, n_animals=3, n_pre=6, n_post=4, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for a in range(n_animals):
        off = rng.normal(0, 5)                                   # animal-specific level, removed by the delta
        for k in range(n_pre):
            rows.append({"animal": f"A{a}", "date": f"pre{k}", "epoch": "pre", "pos": "far_ipsi",
                         "m": off + rng.normal(0, 1)})
        for k in range(n_post):
            rows.append({"animal": f"A{a}", "date": f"post{k}", "epoch": "acute", "pos": "far_ipsi",
                         "m": off + shift + rng.normal(0, 1)})
    return pd.DataFrame(rows)


def test_session_change_detects_a_shift_and_not_noise():
    hit = st.session_change_tests(_panel(5.0), ["m"], epochs=("acute",), n_boot=300)
    assert bool(hit.headline_sig.iloc[0]) and bool(hit.boot_sig.iloc[0]) and hit.agree.iloc[0] == "both"
    assert 4.0 < hit.mean_change.iloc[0] < 6.0
    null = st.session_change_tests(_panel(0.0, seed=3), ["m"], epochs=("acute",), n_boot=300)
    assert not bool(null.headline_sig.iloc[0])


def test_cluster_permutation_finds_the_shifted_bins_only():
    rng = np.random.default_rng(1)
    n_pre, n_post, bins = 12, 12, 40
    curves = rng.normal(0, 1, (n_pre + n_post, bins))
    curves[n_pre:, 15:25] += 3.0                                  # post differs in bins 15-24
    is_post = np.r_[np.zeros(n_pre, bool), np.ones(n_post, bool)]
    animal = np.r_[np.repeat(["A", "B", "C"], 4), np.repeat(["A", "B", "C"], 4)]
    m = st.cluster_permutation(curves, is_post, animal, n_perm=300)
    assert m[17:23].all() and not m[:10].any() and not m[30:].any()
    m0 = st.cluster_permutation(rng.normal(0, 1, (24, bins)), is_post, animal, n_perm=300)
    assert not m0.any()
