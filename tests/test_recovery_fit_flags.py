"""A11 (Priya 11-07): recovery-fit quality flags, half-recovery RELATIVE TO PRE, and the hit-gated minimum.

  - recovery_timing.recovery_metrics: a series whose fit hits a parameter bound / is carried by one session -> flag
    "bound" / "unstable" and the headline NaN; a clean exponential -> "ok" and the half-recovery-to-pre day within 5 %;
    a series that plateaus below half of pre (PS94's aim) -> "not_reached" even though tau ln 2 is short.
  - analysis_kit.MIN_HITS: a code-on-hits epoch value from 8 hits -> NaN, from 31 hits -> the value (recovery_timing,
    decoder_retention_sessions).
Synthetic inputs only, no share reads.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = ("recovery_timing", "decoder_retention_sessions")


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for k in list(os.environ):
        if k.startswith(("WIDEFIELD_", "WFIELD_", "COMBINED_ENCODING_")):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    yield
    for m in SCRIPTS:
        sys.modules.pop(m, None)


DAYS = np.array([1, 2, 3, 4, 5, 7, 9, 11, 14, 18, 22, 28, 35, 42, 49], float)


def _curve(d, a, b, tau):
    return a - (a - b) * np.exp(-d / tau)


def test_clean_exponential_ok_and_half_to_pre_within_5pct():
    import recovery_timing as rt
    a, b, tau = 1.0, 0.0, 6.0                        # recovers fully to pre: half-to-pre day = tau ln 2
    z = _curve(DAYS, a, b, tau) + np.random.default_rng(0).normal(0, 0.01, len(DAYS))
    r = rt.recovery_metrics(DAYS, z, DAYS >= 28)
    truth = tau * np.log(2)
    assert r["flag"] == "ok"
    assert r["half_to_pre"] == pytest.approx(truth, rel=0.05)
    assert r["half_to_pre_interp"] == pytest.approx(truth, rel=0.15)
    assert r["loo_lo"] <= r["half_to_pre"] <= r["loo_hi"]


def test_partial_plateau_is_not_half_recovery_to_pre():
    import recovery_timing as rt
    z = _curve(DAYS, 0.45, 0.0, 4.4)                 # PS94 aim: plateau below half of pre; tau ln 2 ~ 3 d
    r = rt.recovery_metrics(DAYS, z, DAYS >= 28)
    assert r["t_to_own_plateau_half"] == pytest.approx(4.4 * np.log(2), rel=0.05)
    assert r["flag"] == "not_reached" and np.isnan(r["half_to_pre"]) and np.isnan(r["half_to_pre_interp"])
    z = _curve(DAYS, 0.6, 0.0, 4.0)                  # plateau 0.6: half-to-pre is LATER than half of own plateau
    r = rt.recovery_metrics(DAYS, z, DAYS >= 28)
    assert r["flag"] == "ok"
    assert r["half_to_pre"] == pytest.approx(-4.0 * np.log(1 - 0.5 / 0.6), rel=0.05)
    assert r["half_to_pre"] > r["t_to_own_plateau_half"]


def test_bound_case_flagged_and_headline_nan():
    """PS95 behavior (A11): one very low acute session then a near-flat high series drives b to its -2 bound."""
    import recovery_timing as rt
    d = np.array([1, 2, 3, 4, 6, 8, 10, 13, 17, 21, 27, 34, 41], float)
    z = np.r_[0.05, 0.95, 1.02, 0.97, 1.0, 1.03, 0.98, 1.01, 0.99, 1.0, 1.02, 0.97, 1.0]
    p = rt.fit_params(d, z)
    assert rt.at_bound(p)
    r = rt.recovery_metrics(d, z, d >= 28)
    assert r["flag"] in ("bound", "unstable")
    assert np.isnan(r["half_to_pre"]) and np.isfinite(r["half_to_pre_fit"])
    assert rt.fmt_half(r).startswith("unresolved")


def test_unstable_and_few_points():
    import recovery_timing as rt
    d = np.array([1, 2, 3, 4], float)
    r = rt.recovery_metrics(d, np.array([0.1, 0.4, 0.7, 0.8]), np.zeros(4, bool))
    assert r["flag"] == "unstable"                   # a LOO fit on 3 points fails
    r = rt.recovery_metrics(d[:3], np.array([0.1, 0.4, 0.7]), np.zeros(3, bool))
    assert r["flag"] == "few_points" and np.isnan(r["half_to_pre"])
    r = rt.recovery_metrics(DAYS, np.ones_like(DAYS), DAYS >= 28, gate="few_hits")
    assert r["flag"] == "few_hits" and np.isnan(r["complete"]) and rt.fmt_half(r) == "n.a. (<20 hits)"


def _series(n_acute_hits: int) -> pd.DataFrame:
    rows = [{"animal": "PS94", "label": f"PS94_p{i}", "epoch": "pre", "day": -5 + i, "beh": 1.0, "neu": 1.0,
             "aim": 0.0, "n_hits": 30} for i in range(3)]
    rows.append({"animal": "PS94", "label": "PS94_a", "epoch": "acute", "day": 1, "beh": 0.2, "neu": 0.3,
                 "aim": -30.0, "n_hits": n_acute_hits})
    for i, dd in enumerate((5, 8, 12, 16)):
        rows.append({"animal": "PS94", "label": f"PS94_s{i}", "epoch": "subacute", "day": dd, "beh": 0.8, "neu": 0.6,
                     "aim": -15.0, "n_hits": 15})
    for i, dd in enumerate((30, 40)):
        rows.append({"animal": "PS94", "label": f"PS94_c{i}", "epoch": "chronic", "day": dd, "beh": 0.95, "neu": 0.65,
                     "aim": -12.0, "n_hits": 12})
    return pd.DataFrame(rows)


def test_min_hits_gates_code_on_hits_recovered_fraction():
    import recovery_timing as rt
    from wfield_local import analysis_kit as ak
    assert ak.MIN_HITS == 20 and not ak.hits_ok([8]) and ak.hits_ok([31]) and ak.hits_ok([12, 12])
    lo = rt.per_animal_recovery(_series(8)).set_index("animal").loc["PS94"]
    hi = rt.per_animal_recovery(_series(31)).set_index("animal").loc["PS94"]
    assert np.isnan(lo.neu_acute) and np.isnan(lo.neu_recov_chronic) and lo.neu_flag == "few_hits"
    assert hi.neu_acute == pytest.approx(0.3)
    assert hi.neu_recov_chronic == pytest.approx((0.65 - 0.3) / (1 - 0.3))      # chronic: 2 x 12 = 24 hits
    assert hi.neu_complete == pytest.approx(hi.neu_recov_chronic)
    # behaviour series are not hit-gated
    assert lo.aim_recov_chronic == pytest.approx((-12 + 30) / 30) and lo.beh_recov_chronic == pytest.approx(0.75 / 0.8)
    # the pooled-epoch helper drops the gated cell
    S = _series(8)
    assert len(rt._hit_gated(S[S.epoch == "acute"], "neu")) == 0
    assert len(rt._hit_gated(_series(31).query("epoch == 'acute'"), "neu")) == 1
    assert len(rt._hit_gated(S[S.epoch == "acute"], "beh")) == 1


def test_decoder_retention_hit_cell_gated_failed_not():
    import decoder_retention_sessions as dr
    v8 = pd.DataFrame({"ret": [0.4], "n": [8]})
    v31 = pd.DataFrame({"ret": [0.4, 0.5], "n": [16, 15]})
    assert len(dr._cell(v8, "hit")) == 0 and len(dr._cell(v31, "hit")) == 2
    assert len(dr._cell(v8, "failed")) == 1
