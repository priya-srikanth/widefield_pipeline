"""Pins wfield_local/mixed_models.py: recovery of a known step / log-day slope / deficit slope on synthetic sessions,
the random-effects ladder and its fixed-intercept fallback, and the covariate rules (None epochs dropped, June pre
dropped, deficit 0 for pre). No share access (epochs are monkeypatched)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wfield_local import mixed_models as mm


def _panel(seed=0, step=-10.0, slope=4.0, noise=1.0, animal_sd=3.0):
    rng = np.random.default_rng(seed)
    rows = []
    for an in ("PS92", "PS93", "PS94", "PS95"):
        base = rng.normal(20, animal_sd)
        for day in (-8, -6, -5, -3, -2, -1, 1, 2, 3, 5, 7, 11, 15, 22, 29, 36, 43, 50):
            lday = np.log(max(day, 1)) if day > 0 else 0.0
            post = float(day > 0)
            rows.append({"label": f"{an}_{day}", "animal": an, "day": day, "post": post, "lday": lday,
                         "epoch": "pre" if day < 0 else ("acute" if day <= 7 else "chronic"),
                         "y": base + post * (step + slope * lday) + rng.normal(0, noise)})
    return pd.DataFrame(rows)


def test_time_model_recovers_step_and_slope():
    L = _panel()
    r, re = mm.fit_lmm(L, "y ~ post + post:lday", "post:lday")
    assert abs(r.fe_params["post"] - (-10)) < 1.0 and abs(r.fe_params["post:lday"] - 4) < 0.5
    assert re in ("~post + post:lday", "~post", "~1", "fixed")


def test_lmm_table_groups_fdr_and_rule8(capsys):
    L = pd.concat([_panel(1).assign(window="cue"), _panel(2, step=0.0, slope=0.0).assign(window="lick")])
    L["deficit"] = 0.0
    T = mm.lmm_table(L, by=["window"], models=("time",))
    out = capsys.readouterr().out
    assert "per-animal epoch means" in out                        # rule 8 printed first
    assert set(T.term) == {"pre_level", "post", "post:lday"}
    cue = T[(T.window == "cue") & (T.term == "post")].iloc[0]
    lick = T[(T.window == "lick") & (T.term == "post")].iloc[0]
    assert cue.p < 1e-3 and lick.p > 0.01
    assert T.q.notna().all()


def test_behaviour_model_recovers_deficit_slope():
    L = _panel(3, step=0.0, slope=0.0)
    rng = np.random.default_rng(3)
    L["deficit"] = np.where(L.post == 1, rng.uniform(0, 1, len(L)), 0.0)
    L["y"] = L.y + 5.0 * L.post - 20.0 * L.deficit
    r, _re = mm.fit_lmm(L, "y ~ post + post:deficit", "post:deficit")
    assert abs(r.fe_params["post"] - 5) < 1.0 and abs(r.fe_params["post:deficit"] - (-20)) < 1.5


def test_fixed_fallback_when_random_intercept_singular():
    L = _panel(4, animal_sd=0.0, noise=1e-9)                        # identical animals, no noise -> singular RE
    r, re = mm.fit_lmm(L, "y ~ post + post:lday", "post:lday")
    assert re in ("fixed", "~1", "~post", "~post + post:lday")
    assert np.isfinite(r.fe_params["post"])
    if re == "fixed":                                               # without the fallback the same group fails
        r2, re2 = mm.fit_lmm(L, "y ~ post + post:lday", "post:lday", fallback=False)
        assert re2 == "failed" and r2 is None


def test_add_time_covariates_drops_none_epochs_and_june_pre(monkeypatch):
    from wfield_local import epochs
    ep = {"PS92_0606": "pre", "PS92_0811": "pre", "PS92_0817": None, "PS92_0819": "acute"}
    dy = {"PS92_0606": -72, "PS92_0811": -6, "PS92_0817": 0, "PS92_0819": 2}
    monkeypatch.setattr(epochs, "epoch_of", lambda x: ep[x])
    monkeypatch.setattr(epochs, "days_since_stroke", lambda x: dy[x])
    L = mm.add_time_covariates(pd.DataFrame({"label": list(ep), "y": [1.0, 2.0, 3.0, 4.0]}))
    assert list(L.label) == ["PS92_0811", "PS92_0819"]               # June pre and the excluded 0817 dropped
    assert list(L.post) == [0.0, 1.0] and np.isclose(L.lday.iloc[1], np.log(2))
    D = mm.add_deficit(L, {"PS92_0811": 0.4, "PS92_0819": 0.7})
    assert list(D.deficit) == [0.0, 0.7]                             # pre forced to 0


@pytest.mark.parametrize("model", ["time", "behaviour", "epoch"])
def test_summary_table_runs(model):
    L = _panel(5)
    L["window"] = "cue"
    L["deficit"] = np.where(L.post == 1, 0.5, 0.0)
    T = mm.lmm_table(L, by=["window"], models=(model,), print_animals=False)
    S = mm.summary_table(T, ["window"])
    assert len(S) == 1 and "re" in S.columns
