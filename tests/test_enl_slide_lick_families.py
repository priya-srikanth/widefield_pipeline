"""Priya 11-04 ~20:00 (RUNNING_LIST_2026-11-01; encoding handoff 00-COMPACT-25 queued item 1): the between-trial lick ENL
family (lick_enl_angle_decode, lick_enl_target_test, lick_enl_offset_explore, the prelaunch / revup pre-cue parts) and
fr_enl_decode SLIDE, like the imaging ENL decoders, with the guard start >= strobe + 0.3 s and inside the family's span.

Pins:
  * `enl_window.sliding_lags` IS the decoders' search (`lp_licks.enl_window_starts_s` -> `precue_window_start`), plus
    the guard; clean trials keep exactly [-W, end_lag]; a lick slides the window to the latest clean gap; too-early
    slides are dropped, never clipped;
  * `window_mean`, `strobe_for_cues`;
  * every family reads the sliding window (and the angle decoder's "slide" rule is primary, with new frozen ids).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wfield_local import enl_window as ew
from wfield_local import lp_licks

REPO = Path(__file__).resolve().parents[1]


def test_clean_trial_keeps_the_cue_ending_window():
    lo, hi = ew.sliding_lags([100.0], [97.0], np.array([90.0, 101.0]), 1.0, end_lag=-0.10, span_lo=-2.0)
    assert lo[0] == -1.0 and hi[0] == pytest.approx(-0.10)


def test_lick_in_window_slides_to_the_latest_clean_gap():
    # lick at cue - 0.4 s: the latest clean 1 s gap ends at that lick -> window [-1.4, -0.4]
    lo, hi = ew.sliding_lags([100.0], [97.0], np.array([99.6]), 1.0, end_lag=-0.10, span_lo=-2.0)
    assert lo[0] == pytest.approx(-1.4, abs=1.5e-3) and hi[0] == pytest.approx(-0.4, abs=1.5e-3)
    # same answer as the decoders' search itself (one definition)
    w0 = lp_licks.enl_window_starts_s(np.array([100.0]), np.array([97.0]), np.array([99.6]), 1.0)
    assert lo[0] == pytest.approx(w0[0] - 100.0)


def test_guard_drops_slides_into_the_spout_arrival_or_out_of_span():
    # strobe at cue - 1.6 s: a lick at -0.4 slides the window to [-1.4, -0.4], which starts 0.2 s after the strobe
    lo, _ = ew.sliding_lags([100.0], [98.4], np.array([99.6]), 1.0, span_lo=-3.0)
    assert np.isnan(lo[0])
    lo, _ = ew.sliding_lags([100.0], [98.4 - 0.2], np.array([99.6]), 1.0, span_lo=-3.0)      # 0.4 s after: kept
    assert lo[0] == pytest.approx(-1.4, abs=1.5e-3)
    # traces start at -1.2 s: the slid window would leave them -> dropped, never clipped
    lo, hi = ew.sliding_lags([100.0], [96.0], np.array([99.6]), 1.0, span_lo=-1.2)
    assert np.isnan(lo[0]) and np.isnan(hi[0])
    # a 2 s window on a grid that starts at -2 s can never slide (fr_enl_decode's 2 s arms = the old fixed rule)
    lo, _ = ew.sliding_lags([100.0], [96.0], np.array([99.5]), 2.0, span_lo=-2.0)
    assert np.isnan(lo[0])


def test_no_strobe_keeps_only_an_unslid_window():
    lo, _ = ew.sliding_lags([100.0, 200.0], [np.nan, np.nan], np.array([199.6]), 1.0, span_lo=-2.0)
    assert lo[0] == -1.0 and np.isnan(lo[1])


def test_window_mean_uses_each_trials_window():
    lags = np.round(np.arange(-2.0, 0.0001, 0.1), 3)
    X = np.broadcast_to(lags[None, :, None], (3, len(lags), 2)).copy()
    out = ew.window_mean(X, lags, np.array([-1.0, -1.5, np.nan]), np.array([-0.1, -0.5, np.nan]))
    assert out[0, 0] == pytest.approx(np.mean(lags[(lags >= -1.0) & (lags <= -0.1)]))
    assert out[1, 1] == pytest.approx(-1.0) and np.isnan(out[2]).all()


def test_strobe_for_cues_matches_by_cue_time(monkeypatch):
    from wfield_local import trial_classes as tc
    T = pd.DataFrame({"trial_id": [1, 2], "cue_daq": [100.0, 110.0], "strobe_ms": [-3000.0, -2500.0],
                      "stop_ms": [3500.0, 3500.0]})
    monkeypatch.setattr(tc, "_trial_times", lambda sd: T)
    monkeypatch.setattr(tc, "_paths", lambda a, b: (Path("v36"), Path("c1")))
    got = ew.strobe_for_cues("PS92_0810", [100.005, 110.0, 150.0])
    assert got[0] == pytest.approx(97.0) and got[1] == pytest.approx(107.5) and np.isnan(got[2])


@pytest.mark.parametrize("script", ["lick_enl_angle_decode", "revup_neural", "fr_enl_decode"])
def test_families_read_the_sliding_window(script):
    src = (REPO / "scripts" / f"{script}.py").read_text(encoding="utf-8")
    assert "enl_window.sliding_lags(" in src


def test_angle_family_slide_is_primary_with_new_ids(monkeypatch):
    import sys
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    sys.modules.pop("lick_enl_angle_decode", None)
    import lick_enl_angle_decode as ead
    assert ead.LF_PRIMARY == "slide" and ead.LF_RULES[0] == "slide" and "2s" in ead.LF_RULES
    assert ead.fcols(1.0, "slide", ["a"]) == ["s10:a"] and ead.fcols(0.5, "2s", ["a"]) == ["w05:a"]
    T = pd.DataFrame({"angle": [1.0, 2.0, 3.0], "slo10": [-1.0, np.nan, -1.3], "nl2": [0, 0, 1], "n_between": 0,
                      "epoch": "pre"})
    assert len(ead.select(T, 1.0, "slide")) == 2 and len(ead.select(T, 1.0, "2s")) == 2
    with pytest.raises(SystemExit):                                  # an old (fixed-window) trial table is refused
        ead.select(T.drop(columns=["slo10"]), 1.0, "slide")
    # the frozen decoders' identity carries the rule: enl_w1_lfslide is never the old enl_w1_lf2s
    src = (REPO / "scripts" / "lick_enl_angle_decode.py").read_text(encoding="utf-8")
    assert 'align=f"enl_w{W:g}_lf{lf}{variant}"' in src
    for s in ("lick_enl_target_test", "lick_enl_offset_explore"):
        t = (REPO / "scripts" / f"{s}.py").read_text(encoding="utf-8")
        assert "ead.LF_PRIMARY" in t and "ead.fcols(" in t
