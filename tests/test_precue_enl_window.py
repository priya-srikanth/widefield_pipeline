"""The pre-cue LEVEL on the lick-free ENL window (Priya 11-09, `scripts/lick_precue_restw.py`): the trial selection
(`enl_select` = `enl_window.sliding_lags` -> `lp_licks.enl_window_starts_s` + the LP-verdict and working masks), the
window length / cache naming per lick source, the references, and the window means.

Synthetic licks only: nothing reads real data."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


@pytest.fixture
def lp(monkeypatch):
    monkeypatch.setenv("WIDEFIELD_LICK_SOURCE", "lp")
    monkeypatch.delenv("WIDEFIELD_ENL_WINDOW_S", raising=False)
    yield


@pytest.fixture
def pr(lp):
    import lick_precue_restw as m
    return m


def _one(pr, licks, strobe=-3.0, has=True, working=True, win=1.0):
    """One trial with the cue at t = 100 s; licks / strobe in s from the cue. Returns (keep, lo, hi)."""
    cue = 100.0
    k, lo, hi = pr.enl_select(np.array([cue]), np.array([cue + strobe]), cue + np.asarray(licks, float),
                              np.array([has]), np.array([working]), win)
    return bool(k[0]), float(lo[0]), float(hi[0])


def test_window_length_follows_lick_source(pr, monkeypatch):
    assert pr.enl_win_s() == 1.0 and pr.enl_cache() == "precue_enl_lp1s"
    monkeypatch.setenv("WIDEFIELD_LICK_SOURCE", "daq")
    assert pr.enl_win_s() == 2.0 and pr.enl_cache() == "precue_enl_daq2s"


def test_clean_trial_keeps_the_cue_ending_window(pr):
    assert _one(pr, [-3.5, 0.2]) == (True, -1.0, 0.0)        # licks outside [-1, 0) do not move it


def test_lick_in_window_slides_to_the_latest_clean_gap(pr):
    keep, lo, hi = _one(pr, [-0.5])
    assert keep and lo == pytest.approx(-1.5, abs=2e-3) and hi == pytest.approx(-0.5, abs=2e-3)


def test_no_lick_free_window_is_dropped(pr):
    keep, lo, hi = _one(pr, np.arange(-2.9, 0.0, 0.5))     # a lick every 0.5 s from the strobe to the cue
    assert not keep and np.isnan(lo) and np.isnan(hi)


def test_settle_after_strobe_guard(pr):
    # the gap [strobe, -0.2) holds 1.4 s: the window [-1.2, -0.2) starts >= strobe + 0.3 -> kept ...
    assert _one(pr, [-0.2], strobe=-1.6)[0]
    # ... with the strobe at -1.4 it would start 0.2 s after the strobe -> DROPPED (never clipped)
    assert not _one(pr, [-0.2], strobe=-1.4)[0]


def test_window_must_lie_on_the_trace_grid(pr):
    # clean gaps: [-1.05, -0.1) is too short; the next one gives [-2.05, -1.05), earlier than LAGS[0] = -2 -> dropped
    assert not _one(pr, [-1.05, -0.1], strobe=-5.0)[0]
    assert _one(pr, [-1.0, -0.1], strobe=-5.0)[0]            # [-2.0, -1.0) fits


def test_missing_strobe_keeps_only_an_unslid_window(pr):
    assert _one(pr, [], strobe=np.nan)[:1] == (True,)
    assert not _one(pr, [-0.5], strobe=np.nan)[0]


def test_verdict_and_working_masks_drop(pr):
    assert not _one(pr, [], has=False)[0]                     # no LP verdict
    assert not _one(pr, [], working=False)[0]                 # not a camera working trial


def test_selection_is_the_shared_search(pr):
    """enl_select's windows are exactly lp_licks.enl_window_starts_s's (rule 9: one search)."""
    from wfield_local import lp_licks
    rng = np.random.default_rng(0)
    cue = np.arange(20) * 10.0 + 5.0
    strobe = cue - rng.uniform(1.5, 4.0, cue.size)
    licks = np.sort(rng.uniform(0, 205, 120))
    k, lo, _hi = pr.enl_select(cue, strobe, licks, np.ones(cue.size, bool), np.ones(cue.size, bool), 1.0)
    w0 = lp_licks.enl_window_starts_s(cue, strobe, licks, 1.0)
    ok = np.isfinite(w0) & (w0 >= strobe + 0.3 - 1e-9) & (w0 - cue >= -2.0 - 1e-9)
    assert np.array_equal(k, ok)
    assert np.allclose(lo[k], np.where(np.abs(w0 - cue + 1.0) < 1e-3, -1.0, w0 - cue)[k])


def test_window_means_match_the_fixed_window_on_unslid_trials(pr):
    from wfield_local import enl_window
    rng = np.random.default_rng(1)
    X = rng.normal(size=(3, len(pr.LAGS), 2))
    lo, hi = np.array([-1.0, -1.5, np.nan]), np.array([0.0, -0.5, np.nan])
    p10 = enl_window.window_mean(X, pr.LAGS, lo, hi)
    p05 = enl_window.window_mean(X, pr.LAGS, hi - pr.LAST_S, hi)
    fixed = (pr.LAGS >= pr.WIN[0]) & (pr.LAGS <= pr.WIN[1])
    assert np.allclose(p10[0], X[0, fixed].mean(0))           # unslid = the old -1..0 s mean
    m2 = (pr.LAGS >= pr.WIN2[0]) & (pr.LAGS <= pr.WIN2[1])
    assert np.allclose(p05[0], X[0, m2].mean(0))              # last 0.5 s of an unslid window = the old -0.5..0 s
    assert np.allclose(p10[1], X[1, (pr.LAGS >= -1.5 - 1e-9) & (pr.LAGS <= -0.5 + 1e-9)].mean(0))
    assert np.isnan(p10[2]).all()


def test_references(pr):
    z = {"cols": np.array(["a", "b"]), "restnl_wr": np.array([1.0, 2.0]), "restnl_wr_ok": True,
         "restnl_strict": np.array([5.0, 6.0]), "restnl_strict_ok": False, "restw": np.array([3.0, 4.0]),
         "restw_ok": True}
    base, ok = pr.ref_base(z, "restnl")
    assert ok and np.array_equal(base, [1.0, 2.0])            # the active rule (wr)
    base, ok = pr.ref_base(z, "session")
    assert ok and np.array_equal(base, [0.0, 0.0])            # the session-mean twin: no subtraction
    assert pr.ref_base(z, "restw")[1]
    assert {pr.ref_sfx(r) for r in pr.REFS} == {"_restnl", "_sessmean", "_restw"}


def test_reference_and_window_flags(pr, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["x"])
    assert (pr.window_mode(), pr.reference()) == ("enl", "restnl")
    monkeypatch.setattr(sys, "argv", ["x", "--rest-variant", "session"])
    assert pr.reference() == "session"
    monkeypatch.setattr(sys, "argv", ["x", "--precue-window", "fixed"])
    assert pr.reference() == "restw"                          # the old default, unchanged
    monkeypatch.setattr(sys, "argv", ["x", "--precue-window", "fixed", "--rest-variant", "session"])
    with pytest.raises(SystemExit):
        pr.reference()
