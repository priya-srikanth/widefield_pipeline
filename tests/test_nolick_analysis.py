"""Tests for the no-detected-lick arm.

The property that matters most is the one the old code got wrong: a metric and a null that stay
honest when BOTH the trials and the predictions are skewed. Two of the checks below are deliberately
run on a KNOWN-GOOD case (balanced trials, unbiased predictions) as well as the known-bad one --
twice now a guard in this repo fired on everything because it was only ever tested against the case
it was written for.
"""
from __future__ import annotations

import numpy as np
import pytest

from wfield_local import nolick_analysis as na
from wfield_local.plot_lick_aligned_averages import DISPLAY_ORDER

CH = na.CHANCE


def _skewed_truth(n=4000, seed=0):
    """PS93-like: no-lick trials concentrated on far_center / far_L."""
    p = np.array([0.077, 0.059, 0.035, 0.251, 0.490, 0.088])
    return np.random.RandomState(seed).choice(DISPLAY_ORDER, size=n, p=p / p.sum())


def _biased_pred(n=4000, seed=1):
    """Decoder that ignores the input and predicts with its own (different) skew."""
    q = np.array([0.106, 0.103, 0.102, 0.259, 0.224, 0.206])
    return np.random.RandomState(seed).choice(DISPLAY_ORDER, size=n, p=q / q.sum())


def test_balanced_accuracy_null_is_one_sixth_despite_both_skews():
    """The headline metric's null expectation is 1/6 no matter how skewed either side is.

    This is the whole reason balanced accuracy is the headline rather than raw accuracy: raw
    accuracy's null moves with the skews (see the next test), this one cannot.
    """
    y, p = _skewed_truth(), _biased_pred()
    assert na.balanced_accuracy(y, p) == pytest.approx(CH, abs=0.03)


def test_raw_accuracy_null_exceeds_uniform_chance_when_both_are_skewed():
    """Independent, information-free predictions score ABOVE 1/6 -- the bug being retired."""
    y, p = _skewed_truth(), _biased_pred()
    raw = float((y == p).mean())
    assert raw > CH + 0.02, "the confound should be visible in this fixture, else it tests nothing"
    null = na.permutation_null(y, p, n_perm=200, seed=0)
    # the permutation null reproduces it, so nothing is left over
    assert null["raw_null_mean"] == pytest.approx(raw, abs=0.02)
    assert null["raw_p"] > 0.05, "no real signal here, so the corrected test must NOT be significant"


def test_uniform_chance_flag_and_corrected_flag_disagree_on_the_confounded_case():
    """The exact failure the module exists to retire, pinned so it cannot come back."""
    y, p = _skewed_truth(), _biased_pred()
    r = na.evaluate_arm(y, p, n_perm=200)
    assert r["above_uniform_chance_DEPRECATED"] is True
    assert r["above_null_raw"] is False


def test_known_good_case_balanced_trials_and_unbiased_predictions():
    """A guard that fires on everything is ignored. On clean data the corrected null sits at 1/6."""
    rng = np.random.RandomState(3)
    y = rng.choice(DISPLAY_ORDER, size=3000)
    p = rng.choice(DISPLAY_ORDER, size=3000)
    null = na.permutation_null(y, p, n_perm=200, seed=0)
    assert null["raw_null_mean"] == pytest.approx(CH, abs=0.02)
    assert null["bal_null_mean"] == pytest.approx(CH, abs=0.02)


def test_real_signal_survives_the_corrected_null():
    """And it must still DETECT information when information is present, skew or no skew."""
    y = _skewed_truth(n=3000)
    rng = np.random.RandomState(5)
    p = np.where(rng.rand(y.size) < 0.55, y, rng.choice(DISPLAY_ORDER, size=y.size))
    r = na.evaluate_arm(y, p, n_perm=200)
    assert r["above_null_raw"] and r["above_null_balanced"]
    assert r["bal_p"] < 0.01


def test_majority_class_floor_beats_a_weak_decoder_on_skewed_trials():
    y = _skewed_truth()
    assert na.majority_class_floor(y) == pytest.approx(0.49, abs=0.03)


def test_match_profile_hits_the_target_distribution():
    y = _skewed_truth(n=6000)
    target = {"close_L": 1 / 6, "close_center": 1 / 6, "close_R": 1 / 6,
              "far_L": 1 / 6, "far_center": 1 / 6, "far_R": 1 / 6}
    draws = na.match_profile(y, target, seed=0, n_draws=5)
    assert draws, "a uniform target must be reachable from this fixture"
    d = draws[0]
    frac = np.array([(y[d] == c).mean() for c in DISPLAY_ORDER])
    assert np.allclose(frac, 1 / 6, atol=0.01)
    # bounded by the scarcest position: close_R is 3.5% of 6000 ~= 210, so ~6*210 total at most
    assert len(d) <= 6 * int(0.036 * 6000) + 6


def test_match_profile_returns_empty_when_a_target_position_is_absent():
    y = np.array([c for c in DISPLAY_ORDER if c != DISPLAY_ORDER[0]] * 10)
    assert na.match_profile(y, {na.POSITION_NAMES[c]: 1 / 6 for c in DISPLAY_ORDER}) == []


def test_balanced_accuracy_skips_absent_classes_rather_than_scoring_them_zero():
    y = np.array([DISPLAY_ORDER[0]] * 10 + [DISPLAY_ORDER[1]] * 10)
    p = y.copy()
    assert na.balanced_accuracy(y, p) == pytest.approx(1.0)


SIG = {"bal_p": 0.001}
NS = {"bal_p": 0.9}


def test_interpretation_maps_the_CONTRAST_to_the_right_hypothesis():
    hi, lo = {"survival_ratio": 0.8}, {"survival_ratio": 0.2}
    assert "PLAN INTACT" in na.interpret(hi, lo, SIG, SIG)
    assert "NO PLAN FORMED" in na.interpret(hi, lo, NS, NS)          # ratio good, level absent
    assert "UNEXPECTED" in na.interpret({"survival_ratio": 0.6}, {"survival_ratio": 0.55}, SIG, SIG)
    assert "indeterminate" in na.interpret({"survival_ratio": float("nan")}, lo, SIG, SIG)


def test_absolute_thresholds_would_mislabel_PS92_but_the_contrast_does_not():
    """The real case that broke the first version.

    PS92 survives 0.401 pre-cue vs 0.143 post-cue -- a LARGER dissociation (2.8x) than PS93's
    0.627/0.357 (1.8x) -- but an independent 0.5 cut on each ratio called PS92 "no plan formed" and
    PS93 "plan intact". Both are the same phenomenon at different strengths.
    """
    ps92 = na.interpret({"survival_ratio": 0.401}, {"survival_ratio": 0.143}, SIG, SIG)
    ps93 = na.interpret({"survival_ratio": 0.627}, {"survival_ratio": 0.357}, SIG, SIG)
    assert "PLAN INTACT" in ps92 and "PLAN INTACT" in ps93


def test_no_dissociation_is_named_rather_than_forced_into_a_hypothesis():
    r = na.interpret({"survival_ratio": 0.30}, {"survival_ratio": 0.28}, SIG, SIG)
    assert "NO CLEAR DISSOCIATION" in r


def test_missing_permutation_p_is_declared_not_assumed():
    r = na.interpret({"survival_ratio": 0.8}, {"survival_ratio": 0.2})
    assert "PLAN INTACT" in r and "unverified" in r


def test_survival_ratio_uses_above_chance_part_not_raw_ratio():
    """A raw ratio treats 1/6 as zero and would call a dead arm '30% preserved'."""
    eng = {"balanced_accuracy": CH + 0.60}
    nol = {"balanced_accuracy": CH + 0.00}
    assert na.compare_arms(eng, nol)["survival_ratio"] == pytest.approx(0.0, abs=1e-9)


def test_empty_arm_does_not_raise():
    r = na.evaluate_arm(np.array([], int), np.array([], int), n_perm=10)
    assert r["n"] == 0 and np.isnan(r["accuracy"])


# ---------------------------------------------------------------------------------------------
# the session bootstrap that REPLACES the 1.5x threshold
# ---------------------------------------------------------------------------------------------
def _raw(n_sess, n_per, precue_strength, cue_strength, seed=0):
    """Synthetic (y, pred, session) triples with a controllable amount of real signal."""
    rng = np.random.RandomState(seed)
    out = {}
    for arm, strength in (("engaged", 0.75), ("nolick", None)):
        ys, ps, ss = [], [], []
        for s in range(n_sess):
            y = rng.choice(DISPLAY_ORDER, size=n_per)
            k = strength if strength is not None else 0.0
            p = np.where(rng.rand(n_per) < k, y, rng.choice(DISPLAY_ORDER, size=n_per))
            ys.append(y); ps.append(p); ss.append(np.full(n_per, f"S{s}", dtype=object))
        out[arm] = (np.concatenate(ys), np.concatenate(ps), np.concatenate(ss))
    return out


def _raw_with(nolick_strength, n_sess=11, n_per=60, seed=0):
    rng = np.random.RandomState(seed)
    out = {}
    for arm, k in (("engaged", 0.75), ("nolick", nolick_strength)):
        ys, ps, ss = [], [], []
        for s in range(n_sess):
            y = rng.choice(DISPLAY_ORDER, size=n_per)
            p = np.where(rng.rand(n_per) < k, y, rng.choice(DISPLAY_ORDER, size=n_per))
            ys.append(y); ps.append(p); ss.append(np.full(n_per, f"S{s}", dtype=object))
        out[arm] = (np.concatenate(ys), np.concatenate(ps), np.concatenate(ss))
    return out


def test_dissociation_ci_detects_a_real_precue_advantage():
    """pre-cue no-lick carries signal, post-cue does not -> difference CI excludes zero."""
    pre = _raw_with(0.45, seed=1)
    cue = _raw_with(0.02, seed=2)
    r = na.dissociation_ci(pre, cue, n_boot=300, seed=0)
    assert r["excludes_zero"] is True
    assert r["direction"] == "precue > postcue"
    assert r["p_one_sided"] < 0.05


def test_dissociation_ci_reports_no_effect_when_both_arms_match():
    """KNOWN-GOOD case: identical strength in both windows must NOT be called a dissociation."""
    pre = _raw_with(0.30, seed=3)
    cue = _raw_with(0.30, seed=4)
    r = na.dissociation_ci(pre, cue, n_boot=300, seed=0)
    assert r["excludes_zero"] is False


def test_dissociation_ci_resamples_SESSIONS_not_trials():
    """A trial bootstrap would give a far narrower interval; pin that sessions are the unit.

    Same data, but relabelled so every trial is its own 'session'. If the function were secretly
    resampling trials the two intervals would agree; because it clusters, the real (11-session)
    interval must be the WIDER one.
    """
    pre, cue = _raw_with(0.45, seed=5), _raw_with(0.05, seed=6)
    wide = na.dissociation_ci(pre, cue, n_boot=300, seed=0)
    per_trial = {}
    for k, (y, p, _s) in pre.items():
        per_trial[k] = (y, p, np.array([f"T{i}" for i in range(y.size)], dtype=object))
    per_trial_cue = {}
    for k, (y, p, _s) in cue.items():
        per_trial_cue[k] = (y, p, np.array([f"T{i}" for i in range(y.size)], dtype=object))
    narrow = na.dissociation_ci(per_trial, per_trial_cue, n_boot=300, seed=0)
    w = wide["difference_ci"][1] - wide["difference_ci"][0]
    n = narrow["difference_ci"][1] - narrow["difference_ci"][0]
    assert w > n, f"session-clustered interval ({w:.3f}) must exceed the trial-level one ({n:.3f})"


def test_dissociation_ci_refuses_when_there_are_too_few_sessions():
    pre, cue = _raw_with(0.4, n_sess=2, seed=7), _raw_with(0.1, n_sess=2, seed=8)
    r = na.dissociation_ci(pre, cue, n_boot=100)
    assert r["n_boot"] == 0 and "fewer than 3 sessions" in r["note"]


def test_dissociation_ci_handles_missing_raw_arrays():
    assert na.dissociation_ci(None, None)["n_boot"] == 0


# ---------------------------------------------------------------------------------------------
# the response window is a HARD CEILING: nothing is analysed after it (spout is moving)
# ---------------------------------------------------------------------------------------------
from wfield_local.nolick_decoder import category_for_rt as _cat_rt  # noqa: E402


def test_nothing_is_categorised_past_the_response_window():
    """A lick at 4 s lands while the spout is already moving -> undetected, never its own arm."""
    assert _cat_rt(4.0, 2.0, 3.5) == "undetected"
    assert _cat_rt(3.6, 2.0, 3.5) == "undetected"
    assert _cat_rt(10.0, 2.0, 3.5) == "undetected"


def test_late_rewarded_is_exactly_the_gap_between_the_cut_and_the_window():
    assert _cat_rt(2.5, 2.0, 3.5) == "late_rewarded"
    assert _cat_rt(3.5, 2.0, 3.5) == "late_rewarded"
    assert _cat_rt(1.9, 2.0, 3.5) == "engaged"


def test_engaged_cut_cannot_exceed_the_response_window():
    """With the cut at the window there is no late arm at all, and nothing spills past it."""
    assert _cat_rt(3.4, 3.5, 3.5) == "engaged"
    assert _cat_rt(3.6, 3.5, 3.5) == "undetected"
    # a cut wider than the window must be clamped, not honoured
    assert _cat_rt(4.0, 10.0, 3.5) == "undetected"


def test_absent_lick_is_undetected():
    assert _cat_rt(float("nan"), 2.0, 3.5) == "undetected"
    assert _cat_rt(-1.0, 2.0, 3.5) == "undetected"
    assert _cat_rt(None, 2.0, 3.5) == "undetected"


# ---------------------------------------------------------------------------------------------
# the pre/post-stroke criterion guard
# ---------------------------------------------------------------------------------------------
def test_comparing_different_criteria_raises():
    """engaged-only pre vs all-trials post is the confound that mimics a stroke effect."""
    with pytest.raises(ValueError, match="criteria differ"):
        na.assert_comparable({"criterion": "engaged_2s"}, {"criterion": "all_trials"})


def test_missing_criterion_raises_rather_than_assuming():
    with pytest.raises(ValueError, match="without a recorded"):
        na.assert_comparable({"criterion": "engaged_2s"}, {})


def test_same_criterion_is_comparable():
    assert na.assert_comparable({"criterion": "all_trials"}, {"criterion": "all_trials"})


def test_reference_position_engagement_restricts_to_spared_positions():
    codes = np.array([DISPLAY_ORDER[0]] * 10 + [DISPLAY_ORDER[3]] * 10)
    cat = np.array(["engaged"] * 8 + ["undetected"] * 2 + ["undetected"] * 10, dtype=object)
    r = na.reference_position_engagement(codes, cat, [DISPLAY_ORDER[0]])
    assert r["n"] == 10 and r["engaged_rate"] == pytest.approx(0.8)
    # the impaired position's total failure must not drag the motivation estimate down
    allpos = na.reference_position_engagement(codes, cat, DISPLAY_ORDER)
    assert allpos["engaged_rate"] == pytest.approx(0.4)


def test_reference_position_engagement_handles_absent_positions():
    r = na.reference_position_engagement(np.array([DISPLAY_ORDER[0]]), np.array(["engaged"]),
                                         [DISPLAY_ORDER[5]])
    assert r["n"] == 0
