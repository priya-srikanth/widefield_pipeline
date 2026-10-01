"""tongue_jaw_mismatch: candidate = zero tongue licks in (70, 5000) ms AND jaw_pass_qc, as in stroke_orofacial.

Constructed trials for each branch of the rule, then parity of the transcribed slope detector against THEIR
`_detector._slope_detect_lick_peaks` (loaded by file path; it imports only numpy) with the mismatch knobs.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wfield_local import config
from wfield_local import orofacial_clean as oc
from wfield_local import tongue_jaw_mismatch as tjm

FPS = 250.0
SRC = Path(__file__).resolve().parents[2] / "stroke_orofacial_pipeline" / "src" / "stroke_orofacial" / "dlc_kinematics"
N = 30000


@pytest.fixture(autouse=True)
def _no_yaml_overrides(monkeypatch):
    monkeypatch.setattr(config, "defaults", lambda session=None: {})


def _jaw():
    return np.where(np.arange(N) % 2 == 0, 0.5, -0.5)          # baseline mean 0, sd 0.5


def _lick(y, at, amp=60.0):
    """Triangular protrusion: 10 frames up (6 px/frame = 1500 px/s), 10 frames down, peak at frame ``at``."""
    y[at - 10:at + 1] += np.linspace(0, amp, 11)
    y[at + 1:at + 11] += np.linspace(amp, 0, 11)[1:]


CUES = [2000, 5000, 8000, 11000, 14000, 17000, 20000]
POS = ["far_L", "close_L", "far_center", "close_center", "close_R", "far_R", "far_L"]


def _session():
    jaw, ty, tx = _jaw(), np.zeros(N), np.zeros(N)
    # trial 1: jaw opens, tongue quiet                 -> CANDIDATE
    jaw[CUES[0] + 50:CUES[0] + 100] += 15
    # trial 2: jaw opens, tongue licks at 400 ms       -> not (the animal licked)
    jaw[CUES[1] + 50:CUES[1] + 100] += 15
    _lick(ty, CUES[1] + 100)
    # trial 3: no jaw, tongue quiet                    -> not (nothing moved)
    # trial 4: no jaw, tongue licks                    -> not
    _lick(ty, CUES[3] + 100)
    # trial 5: jaw opens, tongue twitch fast enough to be a rise (300 px/s) but below min_peak_y_abs
    # (12 < 20 px) -> CANDIDATE
    jaw[CUES[4] + 50:CUES[4] + 100] += 15
    _lick(ty, CUES[4] + 100, amp=12.0)
    # trial 6: jaw opens, a lick whose peak (40 ms) precedes the 70 ms quiet window -> CANDIDATE
    jaw[CUES[5] + 50:CUES[5] + 100] += 15
    _lick(ty, CUES[5] + 10)
    # trial 7: jaw opens, tongue untracked (baseline fill -> NaN) through the window -> CANDIDATE
    jaw[CUES[6] + 50:CUES[6] + 100] += 15
    _lick(ty, CUES[6] + 100)
    ty[CUES[6]:CUES[6] + 1300] = np.nan
    tx[CUES[6]:CUES[6] + 1300] = np.nan
    trials = pd.DataFrame({"trial_id": np.arange(1, 8) * 10, "cue_frame": np.array(CUES, float), "position": POS})
    return jaw, tx, ty, trials


def test_rule_on_constructed_trials():
    jaw, tx, ty, trials = _session()
    res = tjm.classify(jaw, tx, ty, FPS, trials)
    t = res.trials.set_index("trial_id")
    assert t["candidate_no_lick_with_jaw_move"].tolist() == [True, False, False, False, True, True, True]
    assert t["jaw_pass_qc"].tolist() == [True, True, False, False, True, True, True]
    assert t["n_licks_in_quiet_win"].tolist() == [0, 1, 0, 1, 0, 0, 0]
    assert t.loc[20, "tongue_peak_times_ms"] == [400.0]
    assert res.metadata["max_licks_for_quiet"] == 0 and res.metadata["quiet_tongue_win_ms_lo"] == 70.0
    assert res.metadata["jaw_direction_mode"] == "above_or_below"


def test_from_clean_masks_tongue_baseline_fill_but_not_jaw():
    jaw, tx, ty, trials = _session()
    ty7 = ty.copy()
    fm_t = np.zeros(N, np.int8)
    gap = slice(CUES[6], CUES[6] + 1300)
    fm_t[gap] = oc.FILL_BASELINE
    ty7[gap] = 0.0                                             # how Clean stores a baseline fill: exactly 0
    tx7 = np.where(fm_t == oc.FILL_BASELINE, 0.0, tx)

    def mk(x, y, fm, part):
        z = np.zeros(N)
        return oc.Clean(part, z, z, z, z, z, z, z, x, y, fm, 0.0, 0.0, FPS)

    res = tjm.classify_from_clean(mk(np.zeros(N), jaw, np.zeros(N, np.int8), "jaw"),
                                  mk(tx7, ty7, fm_t, "tongue"), trials)
    assert res.trials["candidate_no_lick_with_jaw_move"].tolist() == [True, False, False, False, True, True, True]


def test_precomputed_jaw_table_must_cover_the_trials():
    jaw, tx, ty, trials = _session()
    from wfield_local import jaw_kinematics as jk
    table, _ = jk.jaw_pertrial(jaw, FPS, trials.iloc[:3])
    with pytest.raises(KeyError):
        tjm.classify(jaw, tx, ty, FPS, trials, jaw_table=table)


def test_summary_by_position():
    jaw, tx, ty, trials = _session()
    s = tjm.summarize_by_position(tjm.classify(jaw, tx, ty, FPS, trials).trials).set_index("position")
    assert list(s.index) == list(tjm.POSITIONS)
    assert s.loc["far_L", "n_trials"] == 2 and s.loc["far_L", "n_candidates"] == 2
    assert s.loc["close_L", "n_candidates"] == 0 and s.loc["close_L", "n_jaw_moved"] == 1
    assert s["n_trials"].sum() == 7 and s["n_candidates"].sum() == 4


def test_yaml_block_deep_merges(monkeypatch):
    monkeypatch.setattr(config, "defaults", lambda session=None: {
        "orofacial_kinematics": {"mismatch": {"tongue_detector": {"min_peak_y_abs": 5.0}}}})
    p = tjm.params()
    assert p["tongue_detector"]["min_peak_y_abs"] == 5.0 and p["tongue_detector"]["rise_min_ms"] == 16.0
    jaw, tx, ty, trials = _session()
    # The 12 px twitch now counts as a lick, so trial 5 stops being a candidate.
    assert not tjm.classify(jaw, tx, ty, FPS, trials).trials.iloc[4]["candidate_no_lick_with_jaw_move"]


# --------------------------------------------------------------------------- parity with the source detector

def _load_source_detector():
    path = SRC / "_detector.py"
    if not path.exists():
        pytest.skip(f"source repo not found at {path}")
    spec = importlib.util.spec_from_file_location("_src_detector", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("smooth", [False, True])
def test_detector_parity_with_source(smooth):
    src = _load_source_detector()
    rng = np.random.default_rng(11)
    cfg = {**tjm.DEFAULTS["tongue_detector"], "do_smooth_y_for_detection": smooth}
    dp = tjm._detector_params(cfg)
    pre_f, post_f = 0, 1250
    t_ms = np.arange(-pre_f, post_f + 1) * (1000.0 / FPS)
    n_peaks_seen = 0
    for _ in range(300):
        y = rng.normal(0, 2.0, post_f + 1)
        for at in rng.integers(15, post_f - 15, rng.integers(0, 8)):
            amp = rng.uniform(5, 80)
            w = int(rng.integers(3, 15))
            y[at - w:at + 1] += np.linspace(0, amp, w + 1)
            y[at + 1:at + w + 1] += np.linspace(amp, 0, w + 1)[1:][: len(y[at + 1:at + w + 1])]
        x = rng.normal(0, 1.0, post_f + 1)
        y[rng.random(post_f + 1) < 0.02] = np.nan
        x[rng.random(post_f + 1) < 0.01] = np.nan
        ours = tjm.slope_detect_lick_peaks(y, x, t_ms, FPS, detect_win_ms=(70.0, 5000.0), detector_params=dp)
        theirs = src._slope_detect_lick_peaks(y_t=y, x_t=x, t_ms=t_ms, fps=FPS, detect_win_ms=(70.0, 5000.0),
                                              detector_params=dp)
        for f in ("peak_times_ms", "peak_y", "peak_x", "rise_starts", "turnover", "peak_indices"):
            assert np.array_equal(getattr(ours, f), getattr(theirs, f), equal_nan=True), f
        n_peaks_seen += len(theirs)
    assert n_peaks_seen > 100                                  # the comparison exercised real detections
