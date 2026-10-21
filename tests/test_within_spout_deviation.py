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


def _sided(seed=3, n=900):
    rng = np.random.default_rng(seed)
    pos = rng.choice(["close_contra", "far_contra", "close_ipsi", "far_ipsi", "close_mid", "far_mid"], n)
    lat = rng.normal(0, 5, n)
    T = pd.DataFrame({"pos": pos, "lat_mean": lat, "reach_mean": rng.normal(10, 1, n),
                      "n_licks": rng.integers(3, 9, n), "latency": rng.uniform(0.1, 1, n)})
    return T, lat, W.side_sign(pos), rng


def test_dirmid_separates_direction_from_midline():
    T, lat, s, rng = _sided()
    R = (0.03 * lat * s + rng.normal(0, 0.05, len(lat)))[:, None]       # medial licks evoke more, no direction
    d = W.dirmid(R, T)
    assert abs(d["b_mid"][0] - 0.03) < 0.005 and abs(d["b_dir"][0]) < 3 * d["se_dir"][0]
    R = (0.02 * lat + rng.normal(0, 0.05, len(lat)))[:, None]           # pure direction
    d = W.dirmid(R, T)
    assert abs(d["b_dir"][0] - 0.02) < 0.005 and abs(d["b_mid"][0]) < 3 * d["se_mid"][0]


def test_split_half_reliability():
    rng = np.random.default_rng(4)
    n = 400
    true = rng.normal(0, 4, n)
    T = pd.DataFrame({"pos": rng.choice(list(SPOUT_LAT), n), "lat_odd": true + rng.normal(0, 4, n),
                      "lat_even": true + rng.normal(0, 4, n)})
    r, rel, m = W.split_half(T)
    assert 0.4 < r < 0.6 and abs(rel - 2 * r / (1 + r)) < 1e-9 and m == n


def test_per_spout_and_contact_covariate(monkeypatch):
    """Per-spout slopes recover each spout's own slope; the contact covariate absorbs a touch response that would
    otherwise load on the lateral deviation."""
    import numpy as np
    import pandas as pd
    from scripts import within_spout_deviation as ws
    rng = np.random.default_rng(0)
    n = 60
    pos = np.repeat(["far_contra", "far_ipsi"], n)
    lat = rng.normal(0, 5, 2 * n)
    touch = (lat > 0).astype(float)                        # deviation toward the spout = a touch
    slope = np.where(pos == "far_contra", 0.2, -0.2)
    R = (slope * lat + 3.0 * touch + rng.normal(0, 0.1, 2 * n))[:, None]
    T = pd.DataFrame({"pos": pos, "lat_mean": lat, "reach_mean": rng.normal(0, 1, 2 * n), "n_licks": 5.0,
                      "latency": 0.2, "n_contact": touch})
    monkeypatch.setenv(ws.CONTROLS_ENV, "contact")
    P = ws.per_spout(R, T, "mean")
    assert abs(P["far_contra"][0][0] - 0.2) < 0.03 and abs(P["far_ipsi"][0][0] + 0.2) < 0.03
    monkeypatch.setenv(ws.CONTROLS_ENV, "")
    assert ws.per_spout(R, T, "mean")["far_contra"][0][0] > 0.3      # touch leaks into the slope without it
