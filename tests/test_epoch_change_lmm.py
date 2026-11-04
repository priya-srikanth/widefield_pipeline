"""analysis_kit.epoch_change_lmm / lmm_stars: the change-vs-pre test for figure stars (Priya 2026-11-03)."""
import numpy as np
import pandas as pd

from wfield_local import analysis_kit as ak


def _data(effect_acute, effect_chronic, seed=0, spread=0.02):
    rng = np.random.default_rng(seed)
    rows = []
    for a in ["PS92", "PS93", "PS94", "PS95"]:
        base = rng.normal(0, 0.05)
        for ep, eff, n in (("pre", 0.0, 10), ("acute", effect_acute, 4), ("chronic", effect_chronic, 8)):
            dev = 0.0 if ep == "pre" else rng.normal(0, spread)
            rows += [(a, ep, base + eff + dev + rng.normal(0, 0.02)) for _ in range(n)]
    return pd.DataFrame(rows, columns=["animal", "epoch", "value"])


def test_recovers_changes_and_df():
    T = ak.epoch_change_lmm(_data(-0.5, 0.0)).set_index("epoch")
    assert list(T.index) == ["acute", "chronic"]                      # subacute absent -> no row
    assert abs(T.loc["acute", "beta"] + 0.5) < 0.05
    assert T.loc["acute", "df"] == 3 and T.loc["acute", "p"] < 0.01
    assert T.loc["acute", "lo"] < T.loc["acute", "beta"] < T.loc["acute", "hi"]
    assert abs(T.loc["chronic", "beta"]) < 0.05


def test_one_animal_is_not_tested():
    D = _data(-0.5, 0.0)
    D = D[~((D.epoch == "acute") & (D.animal != "PS92"))]                # only PS92 has acute sessions
    T = ak.epoch_change_lmm(D).set_index("epoch")
    assert np.isnan(T.loc["acute", "p"]) and T.loc["acute", "n_animals"] == 1
    assert np.isfinite(T.loc["chronic", "p"])


def test_lmm_stars_holm_across_family():
    p = np.array([0.001, 0.02, 0.5, np.nan] + [0.9] * 14)               # 18 cells, one NaN
    s = ak.lmm_stars(p)
    assert list(s[:4]) == ["**", "*", "", ""]
