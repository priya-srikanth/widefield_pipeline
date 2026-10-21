"""Pins the within-spout deviation check (scripts/within_spout_deviation.py): a response driven by the EXECUTED angle gives
within ~ between; one driven by the TARGET (spout) gives within ~ 0. Synthetic trials, no share access."""
from __future__ import annotations

import os

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


def test_variants_one_load_each_fit_sees_its_env(monkeypatch):
    """VARIANTS_ENV: one load per session, every variant fitted with its own CONTROLS_ENV / TRIALS_ENV, env restored."""
    from types import SimpleNamespace

    from scripts import within_spout_deviation as ws
    from wfield_local import combined_encoding as ce
    loads = []
    monkeypatch.setattr(ce, "load_imaging", lambda lab: loads.append(lab) or (np.zeros((10, 2), np.float32),
                                                                              np.arange(10.0), None, None, None,
                                                                              {}, (1, 1), None))
    trials = pd.DataFrame({"engaged": [True], "cue_s": [1.0]})
    monkeypatch.setattr(ce, "load_behaviour", lambda *a: SimpleNamespace(trials=trials))
    monkeypatch.setattr(ce, "_slow_basis", lambda ft, p: np.ones((len(ft), 1)))
    monkeypatch.setattr(ws, "trial_table", lambda *a: pd.DataFrame({"pos": ["far_contra"]}))
    monkeypatch.setattr(ws, "commanded", lambda *a: {})
    monkeypatch.setattr(ws, "_fit_session", lambda *a: (os.environ[ws.CONTROLS_ENV], os.environ[ws.TRIALS_ENV]))
    monkeypatch.setenv(ws.VARIANTS_ENV, "base,contact,contact_hits")
    monkeypatch.setenv(ws.CONTROLS_ENV, "keep")
    r = ws.session_check("PS92_0806")
    assert loads == ["PS92_0806"]
    assert r["variants"] == {"base": ("", "all"), "contact": ("contact", "all"), "contact_hits": ("contact", "hits")}
    assert os.environ[ws.CONTROLS_ENV] == "keep"


def test_contact_adjusted_between(monkeypatch):
    """With CONTROLS_ENV=contact the across-spout slope uses contact-adjusted spout means: a response driven only by
    contact, whose rate falls toward one side, gives a between slope without the adjustment and ~0 with it."""
    from scripts import within_spout_deviation as ws
    rng = np.random.default_rng(1)
    rows = []
    for p, cl in SPOUT_LAT.items():
        n = 80
        lat = cl + rng.normal(0, 5, n)
        rate = 0.5 + cl / 80.0                                 # contact rate rises toward the + side
        rows.append(pd.DataFrame({"pos": p, "lat_mean": lat, "reach_mean": rng.normal(10, 1, n), "n_licks": 5.0,
                                  "latency": 0.2, "n_contact": rng.binomial(4, rate, n).astype(float)}))
    T = pd.concat(rows, ignore_index=True)
    R = (2.0 * T.n_contact.to_numpy() + rng.normal(0, 0.2, len(T)))[:, None]
    monkeypatch.setenv(ws.CONTROLS_ENV, "")
    raw = ws.slopes(R, T, "mean")["b_between"][0]
    monkeypatch.setenv(ws.CONTROLS_ENV, "contact")
    adj = ws.slopes(R, T, "mean")["b_between"][0]
    assert raw > 0.05 and abs(adj) < 0.2 * raw


def test_add_z_rescales_within_animal():
    """b_z = b x the animal's pre lateral SD / pre response SD: two animals with the same effect in SD units but different
    amplitudes and angle scales get the same b_z; b_zs uses each session's own SDs."""
    from scripts import within_spout_mixed as wm
    rows = []
    for an, amp, ang in (("PS92", 1.0, 1.0), ("PS94", 4.0, 3.0)):
        for lab in (f"{an}_0806", f"{an}_0807", f"{an}_0920"):              # two pre sessions, one chronic
            rows.append({"label": lab, "variant": "mean", "spouts": "far_ipsi", "area": "MOs_left",
                         "b_within": 0.5 * amp / ang, "lat_sd": 2.0 * ang, "resp_sd": 1.0 * amp})
    D = wm.add_z(pd.DataFrame(rows))
    assert np.allclose(D.b_z, 1.0) and np.allclose(D.b_zs, 1.0)
