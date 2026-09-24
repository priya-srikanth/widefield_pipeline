"""`enl_lick_control` -- is the ENL position code a sensory code or a spout-arrival lick tail?

The control has to be able to return the BAD answer. `test_clean_at_null_is_stated_outright` builds
the worst case -- position decodable only on trials that had a preceding lick -- and requires the
verdict to say so, because that is the finding the module exists to be able to report and the one a
silently-averaged summary would hide.
"""
import numpy as np
import pytest

from wfield_local import enl_lick_control as lc


def _arm(y, g, X, lead):
    return {"X": np.asarray(X, float), "y": np.asarray(y), "g": np.asarray(g),
            "lead_lick": np.asarray(lead, bool)}


def _build(n, rng, tmpl, strength, lead_strength, lead_frac=0.5, nblk=20, ncomp=20):
    """Trials where the SIGNAL differs between lead-lick and lick-free trials by construction."""
    y = rng.integers(0, 6, n)
    g = rng.integers(0, nblk, n)
    lead = rng.random(n) < lead_frac
    amp = np.where(lead, lead_strength, strength)[:, None]
    return _arm(y, g, amp * tmpl[y] + rng.normal(size=(n, ncomp)), lead)


def test_split_arms_cuts_each_arm_by_its_own_flag():
    rng = np.random.default_rng(0)
    tmpl = rng.normal(size=(6, 20))
    pooled = {"success": _build(400, rng, tmpl, 1.0, 1.0, lead_frac=0.25),
              "miss_working": _build(200, rng, tmpl, 1.0, 1.0, lead_frac=0.75)}
    out = lc.split_arms(pooled)
    assert set(out) == {"success_clean", "success_lead", "miss_working_clean", "miss_working_lead"}
    assert len(out["success_clean"]["y"]) + len(out["success_lead"]["y"]) == 400
    # the flag is per arm, so the two arms split at different rates
    assert len(out["success_lead"]["y"]) < len(out["success_clean"]["y"])
    assert len(out["miss_working_lead"]["y"]) > len(out["miss_working_clean"]["y"])


def test_stopped_is_not_split():
    """Priya framed this control as hit + working trials; `stopped` is too scarce to halve."""
    rng = np.random.default_rng(1)
    tmpl = rng.normal(size=(6, 20))
    pooled = {"success": _build(300, rng, tmpl, 1.0, 1.0),
              "miss_working": _build(200, rng, tmpl, 1.0, 1.0),
              "stopped": _build(100, rng, tmpl, 1.0, 1.0)}
    assert not any(k.startswith("stopped") for k in lc.split_arms(pooled))


def test_only_within_arm_pairs_are_compared():
    """`success` vs `miss_working` is a SEPARATE question (readout 4); it must not appear here."""
    rng = np.random.default_rng(2)
    tmpl = rng.normal(size=(6, 20)) * 3.0
    pooled = {"success": _build(1200, rng, tmpl, 1.0, 1.0),
              "miss_working": _build(600, rng, tmpl, 1.0, 1.0)}
    r = lc.analyse(pooled, n_perm=200, n_boot=200)
    keys = set(r["bootstrap"]["differences"])
    assert keys == {"success_lead_minus_success_clean",
                    "miss_working_lead_minus_miss_working_clean"}
    assert not any("success" in k and "miss_working" in k for k in keys)


def test_a_clean_control_reports_no_difference():
    """Same signal on both groups -> the lead lick is not what the decoder reads."""
    rng = np.random.default_rng(3)
    tmpl = rng.normal(size=(6, 20)) * 3.0
    pooled = {"success": _build(1500, rng, tmpl, 1.0, 1.0),
              "miss_working": _build(700, rng, tmpl, 1.0, 1.0)}
    r = lc.analyse(pooled, n_perm=200, n_boot=300)
    assert r["arms"]["success_clean"]["above_null_balanced"]
    d = r["bootstrap"]["differences"]["success_lead_minus_success_clean"]
    assert d["p"] > 0.05
    assert "do not change it" in lc.verdict_line(r)


def test_clean_at_null_is_stated_outright():
    """THE BAD ANSWER. Position present ONLY on trials with a preceding lick.

    If the control cannot say this in words, it is not a control -- a reader would see one arm
    above null and conclude the code is sensory.
    """
    rng = np.random.default_rng(4)
    tmpl = rng.normal(size=(6, 20)) * 3.0
    pooled = {"success": _build(2000, rng, tmpl, 0.0, 1.4, lead_frac=0.5),
              "miss_working": _build(800, rng, tmpl, 0.0, 1.4, lead_frac=0.5)}
    r = lc.analyse(pooled, n_perm=200, n_boot=300)
    assert not r["arms"]["success_clean"]["above_null_balanced"]
    assert "CLEAN TRIALS AT NULL" in lc.verdict_line(r)


def test_an_underpowered_clean_arm_is_could_not_test_not_absence():
    rng = np.random.default_rng(5)
    tmpl = rng.normal(size=(6, 20)) * 3.0
    # almost every trial has a lead lick, as in PS92 (85% pre-stroke)
    pooled = {"success": _build(1500, rng, tmpl, 1.0, 1.0, lead_frac=0.94),
              "miss_working": _build(400, rng, tmpl, 1.0, 1.0, lead_frac=0.94)}
    r = lc.analyse(pooled, n_perm=150, n_boot=150)
    assert r["power"]["success_clean"]["underpowered"]
    assert "could not test" in lc.verdict_line(r)


def test_lead_fraction_is_reported_per_arm():
    rng = np.random.default_rng(6)
    tmpl = rng.normal(size=(6, 20))
    pooled = {"success": _build(1000, rng, tmpl, 1.0, 1.0, lead_frac=0.2),
              "miss_working": _build(500, rng, tmpl, 1.0, 1.0, lead_frac=0.8)}
    r = lc.analyse(pooled, n_perm=100, n_boot=100)
    assert r["lead_fraction"]["success"] == pytest.approx(0.2, abs=0.05)
    assert r["lead_fraction"]["miss_working"] == pytest.approx(0.8, abs=0.05)


def test_no_clean_success_to_train_on_is_skipped_not_crashed():
    rng = np.random.default_rng(7)
    tmpl = rng.normal(size=(6, 20))
    pooled = {"success": _build(300, rng, tmpl, 1.0, 1.0, lead_frac=1.0),
              "miss_working": _build(200, rng, tmpl, 1.0, 1.0, lead_frac=1.0)}
    r = lc.analyse(pooled, n_perm=50, n_boot=50)
    assert "skipped" in r and "lick-free success" in r["skipped"]
