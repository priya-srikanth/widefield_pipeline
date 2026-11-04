"""Priya 11-04: first-lick timing of the lick-aligned arms (lp_licks.first_lick_timing, LP_DAQ_TOL_MS = 50 ms).
CLOSE spouts: LP onset missing or > 50 ms after the first DAQ contact in the response window -> the contact time
(timing_source "daq"). FAR spouts: LP time kept, grooming_suspect when > 50 ms late. Classes unchanged."""
import numpy as np
import pandas as pd

from wfield_local import lp_licks as LL


def test_close_and_far_rules():
    pos = ["close_mid", "close_ipsi", "close_contra", "far_mid", "far_ipsi", "close_L", "far_R", "close_mid"]
    lp = [200.0, np.nan, 160.0, 300.0, 140.0, 250.0, np.nan, np.nan]
    ct = [120.0, 130.0, 130.0, 120.0, 120.0, 210.0, 150.0, np.nan]
    stop = [3600.0] * 8
    T = LL.first_lick_timing(pos, lp, ct, stop)
    assert T.timing_source.tolist() == ["daq", "daq", "lp", "lp", "lp", "lp", "", ""]
    assert np.allclose(T.first_lick_ms.to_numpy()[:6], [120.0, 130.0, 160.0, 300.0, 140.0, 250.0])
    assert T.grooming_suspect.tolist() == [False, False, False, True, False, False, False, False]


def test_tolerance_boundary_and_response_window():
    tol = LL.LP_DAQ_TOL_MS
    assert tol == 50.0
    T = LL.first_lick_timing(["close_mid"] * 2, [150.0, 150.1], [100.0, 100.0], [3600.0] * 2)
    assert T.timing_source.tolist() == ["lp", "daq"]                        # exactly +50 ms is not late
    # a contact outside the response window (before cue + 70 ms, after min(max_rt, stop)) is not used
    T = LL.first_lick_timing(["close_mid"] * 3, [np.nan, np.nan, 4000.0], [60.0, 3600.0, 3000.0],
                             [5000.0, 5000.0, 2900.0])
    assert T.timing_source.tolist() == ["", "", "lp"] and not T.grooming_suspect.any()


def test_session_lp_uses_the_fallback_time(monkeypatch):
    A = pd.DataFrame({"label": "PS92_0806", "trial_id": [0, 1], "cls": ["hit", "hit"], "cam4": [np.nan, 300.0],
                      "combined": [np.nan, np.nan], "pos": ["close_mid", "far_mid"], "stop_ms": [3600.0, 3600.0],
                      "contact": [110.0, 120.0]})
    tm = LL.first_lick_timing(A.pos, A.combined.fillna(A.cam4), A.contact, A.stop_ms)
    A = pd.concat([A, tm], axis=1)
    cue = pd.DataFrame({"cue_daq": [10.0, 20.0]}, index=pd.Index([0, 1], name="trial_id"))
    monkeypatch.setattr(LL, "lp_lick_times", lambda label, rv=None: (np.array([20.3]), cue))
    monkeypatch.setattr(LL, "_attempts", lambda rv=None: A)
    out = LL._session_lp("PS92_0806", np.array([10000, 20000]), 1000.0, 3.5)
    assert out["attempted"].tolist() == [True, True]                         # LP-missing close hit: now timed
    assert np.allclose(out["first_s"], [0.110, 0.300])                      # DAQ time / LP time kept at far


def test_tolerance_enters_the_cache_token(monkeypatch, tmp_path):
    monkeypatch.setattr(LL, "attempts_path", lambda rv=None: tmp_path / "none.csv")
    monkeypatch.setattr(LL, "_cam1_signature", lambda rv=None: "x")
    assert LL.spec_token().startswith(f"lp{LL.LP_VERSION}t50:")
    monkeypatch.setattr(LL, "LP_DAQ_TOL_MS", 30.0)
    assert LL.spec_token().startswith(f"lp{LL.LP_VERSION}t30:")


def test_attempts_table_without_the_columns_is_refused(monkeypatch, tmp_path):
    p = tmp_path / "trial_attempts.csv"
    pd.DataFrame({"label": ["a"], "trial_id": [0], "cls": ["hit"], "cam4": [1.0], "combined": [1.0]}).to_csv(p, index=False)
    monkeypatch.setattr(LL, "attempts_path", lambda rv=None: p)
    LL._ATTEMPTS.clear()
    try:
        LL._attempts()
    except LL.LPUnavailable as e:
        assert "contact" in str(e) or "pos" in str(e)
    else:
        raise AssertionError("expected LPUnavailable")
    LL._ATTEMPTS.clear()
