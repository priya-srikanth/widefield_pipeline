"""scripts/c2_speed_smooth_robustness.py: the per-lick measures (pipeline speed, SG5 speed, jitter, view shares) and the
noise-by-epoch tables, on synthetic data."""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.c2_speed_smooth_robustness import NOISE, central_speed, lick_measures, noise_tables


def _lick(noise=0.0, seed=0, views=3):
    t = np.arange(0, 80, 4.0)                                    # 20 frames, 4 ms
    r = 4.0 * np.sin(np.pi * t / 80.0)                           # out to 4 mm and back
    P = np.c_[r, np.zeros_like(r), np.zeros_like(r)]
    P = P + np.random.default_rng(seed).normal(0, noise, P.shape)
    v = np.full(len(t), views)
    return t, P, v


def test_clean_lick_raw_equals_pipeline_central_difference():
    t, P, v = _lick()
    m = lick_measures(t, P, v, 0, 76)
    sp = central_speed(t, P)
    k = int(np.argmax(np.linalg.norm(P, axis=1)))
    assert np.isclose(m["speed_raw"], np.nanmax(sp[:k + 1]))
    # SG order 2 on a smooth path changes the peak only slightly; a sine has no high-frequency content
    assert abs(m["speed_sg5"] - m["speed_raw"]) < 0.05 * m["speed_raw"]
    assert m["f_le2v_pro"] == 0 and m["f_fill_pro"] == 0 and m["speedpk_le2v"] == 0


def test_noise_raises_raw_peak_and_jitter_more_than_sg5():
    t, P0, v = _lick()
    m0 = lick_measures(t, P0, v, 0, 76)
    raw, sg = [], []
    jit = []
    for s in range(30):
        t, P, v = _lick(noise=0.15, seed=s)
        m = lick_measures(t, P, v, 0, 76)
        raw.append(m["speed_raw"] - m0["speed_raw"])
        sg.append(m["speed_sg5"] - m0["speed_sg5"])
        jit.append(m["jit_pro"])
    assert np.mean(raw) > np.mean(sg) > -1          # raw peak inflated by jitter, smoothed peak much less
    assert np.mean(jit) > m0["jit_pro"]


def test_view_shares_and_too_short():
    t, P, v = _lick(views=2)
    v[3] = -1
    m = lick_measures(t, P, v, 0, 76)
    assert m["f_le2v"] == 1.0 and 0 < m["f_fill_pro"] < 1
    assert lick_measures(t, P, v, 0, 4) is None                 # < 3 frames in the window


def test_noise_tables_change_vs_own_pre():
    rows = []
    for a, base in (("A1", 0.40), ("A2", 0.50)):
        for e, add in (("pre", 0.0), ("acute", 0.05), ("subacute", 0.02), ("chronic", 0.0)):
            for d in range(2):
                for _ in range(5):
                    rows.append({"session": f"{a}_{e}{d}", "animal": a, "date": f"{e}{d}", "epoch": e,
                                 "jit_pro": base + add, "f_le2v_pro": 0.2, "f_le2v": 0.2, "f_fill_pro": 0.1,
                                 "speedpk_le2v": float(_ % 2), "speed_raw": 200.0, "speed_sg5": 195.0})
    S, A, E = noise_tables(pd.DataFrame(rows))
    assert len(S) == 16 and (S.gap_raw_sg5 == 5.0).all() and np.allclose(S.speedpk_le2v, 0.4)   # share, 2 of 5
    assert set(NOISE) <= set(E.columns)
    e = E.set_index("epoch")
    assert np.isclose(e.loc["acute", "jit_pro_chg"], 0.05) and e.loc["acute", "jit_pro_n_up"] == 2
    assert np.isclose(e.loc["pre", "jit_pro_chg"], 0.0) and np.isclose(e.loc["chronic", "jit_pro_chg"], 0.0)
    assert (A.n_sessions == 2).all()
