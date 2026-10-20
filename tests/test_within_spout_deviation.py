"""Pins the within-spout deviation check (scripts/within_spout_deviation.py): a response driven by the EXECUTED angle gives
within ~ between; one driven by the TARGET (spout) gives within ~ 0. Synthetic trials, no share access."""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts import within_spout_deviation as W

SPOUT_LAT = {"a": -30.0, "b": -10.0, "c": 10.0, "d": 30.0, "e": -20.0, "f": 20.0}


def _trials(seed: int = 0, n: int = 600):
    rng = np.random.default_rng(seed)
    pos = rng.choice(list(SPOUT_LAT), n)
    tgt = np.array([SPOUT_LAT[p] for p in pos])
    lat = tgt + rng.normal(0, 6, n)                       # executed = target + lick-to-lick scatter
    T = pd.DataFrame({"pos": pos, "lat_mean": lat, "reach_mean": rng.normal(10, 1, n),
                      "lat_first": lat, "reach_first": rng.normal(10, 1, n), "n_licks": rng.integers(3, 9, n)})
    return T, tgt, lat, rng


def test_executed_response_within_equals_between():
    T, _tgt, lat, rng = _trials()
    R = (0.02 * lat + rng.normal(0, 0.05, len(lat)))[:, None]
    s = W.slopes(R, T, "mean")
    assert abs(s["b_within"][0] - 0.02) < 0.004
    assert abs(s["b_between"][0] - 0.02) < 0.004
    assert s["terc_R"][0, 0] < s["terc_R"][1, 0] < s["terc_R"][2, 0]


def test_target_response_has_no_within_slope():
    T, tgt, lat, rng = _trials(1)
    R = (0.02 * tgt + rng.normal(0, 0.05, len(lat)))[:, None]
    s = W.slopes(R, T, "mean")
    assert abs(s["b_within"][0]) < 3 * s["se_within"][0]
    assert abs(s["b_between"][0] - 0.02) < 0.004       # across spouts it still looks lateral-tuned


def test_spouts_with_too_few_trials_dropped():
    T, _tgt, lat, rng = _trials(2, n=60)
    T = T[(T.pos != "a") | (np.arange(len(T)) < 5)]       # few 'a' trials
    s = W.slopes(np.ones((len(T), 1)), T, "mean")
    assert s == {} or s["n_spouts"] <= 6
