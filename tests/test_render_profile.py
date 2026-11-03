"""The cohort-1 rebuild render profile yields the intended run (coordinator 11-03, Priya's decisions)."""
from __future__ import annotations

import pytest

from wfield_local import render_profile

NAME = "cohort1_rebuild_20261103"
KEYS = ("WIDEFIELD_HEMO_VARIANT", "WIDEFIELD_LOCANMF_PROJECT", "WIDEFIELD_TRIAL_GATE", "WIDEFIELD_TRIAL_WINDOW",
        "WIDEFIELD_TRIAL_END", "WIDEFIELD_TRIAL_STOPPED", "WIDEFIELD_LP_FROZEN", "WIDEFIELD_FROZEN_MODEL_DIR",
        "WIDEFIELD_SESSION_CACHE", "WIDEFIELD_PRE_FROM", "WIDEFIELD_LICK_SOURCE", "WIDEFIELD_GATE_SOURCE",
        "WIDEFIELD_MAP_REFS", "WIDEFIELD_JOINT_PROJECT_ALL", "WIDEFIELD_ENL_WINDOW_S")


@pytest.fixture
def profile(monkeypatch):
    for k in KEYS:
        monkeypatch.delenv(k, raising=False)
    d = render_profile.load(NAME)
    for k, v in d["env"].items():
        monkeypatch.setenv(k, v)
    return d


def test_profile_yields_the_intended_run_tag(profile, monkeypatch):
    from wfield_local import config, locanmf_session, run_mode, trial_classes
    monkeypatch.setattr(trial_classes, "class_source_signature", lambda *a, **k: "sig")
    assert config.hemo_variant() == "meegkit_hpfit_lag_refitT_swapmask"
    assert locanmf_session.projection_on()
    assert trial_classes.gate_mode() == "camera" and trial_classes.end_anchor() == "lick"
    assert trial_classes.stopped_rule_name() == "stopRun6"
    assert run_mode.tag() == profile["notes"]["expected_run_tag"]
    assert run_mode.cache_tag() == profile["notes"]["expected_run_tag"] + "_csig-sig"
    from wfield_local import frozen_models
    assert not frozen_models.daq_trained_in_lp()


def test_profile_stores_are_fresh_and_separate(profile):
    env = profile["env"]
    # fourth review A: both stores live in the profile's REBUILD ROOT
    assert env["WIDEFIELD_FROZEN_MODEL_DIR"] == profile["rebuild_root"] + "/frozen_models"
    assert env["WIDEFIELD_SESSION_CACHE"] == profile["rebuild_root"] + "/session_cache"
    assert env["WIDEFIELD_FROZEN_MODEL_DIR"].rstrip("/\\") != "C:/Users/SabatiniLab/frozen_models"
    assert "rebuild" in env["WIDEFIELD_SESSION_CACHE"]
    assert "WIDEFIELD_PRE_FROM" not in env                     # Aug-only pre is per decoder block, never global
    assert profile["block_args"]["epoch_grant_figures"] == ["--pre-from", "0801"]
    assert profile["notes"]["encoder_drift_cutoff_s"] == 600


def test_shell_lines():
    lines = render_profile.shell_lines(NAME, "powershell")
    assert '$env:WIDEFIELD_TRIAL_GATE = "camera"' in lines
    assert any(ln.startswith("export WIDEFIELD_HEMO_VARIANT=") for ln in render_profile.shell_lines(NAME, "bash"))


def test_drift1500_arm_has_its_own_tag(monkeypatch):
    """Priya 11-03: 600 s primary; the 1500 s drift cutoff is a sensitivity ARM with its own output / cache tag."""
    from wfield_local import run_mode
    monkeypatch.delenv("COMBINED_ENCODING_DRIFT_MIN_PERIOD_S", raising=False)
    assert run_mode.drift_part() == ""
    env = {}
    d = render_profile.apply(NAME, env, arm="drift1500")
    assert env["COMBINED_ENCODING_DRIFT_MIN_PERIOD_S"] == "1500"
    assert "_drift1500s_" in d["notes"]["expected_run_tag"]
    monkeypatch.setenv("COMBINED_ENCODING_DRIFT_MIN_PERIOD_S", "600")
    assert run_mode.drift_part() == ""                                   # the primary value adds no part
    import sys
    if "wfield_local.combined_encoding" in sys.modules:                  # imported at 600 s: a 1500 s env is refused
        monkeypatch.setenv("COMBINED_ENCODING_DRIFT_MIN_PERIOD_S", "1500")
        with pytest.raises(RuntimeError, match="imported"):
            run_mode.drift_part()
    left = render_profile.apply(NAME, {"COMBINED_ENCODING_DRIFT_MIN_PERIOD_S": "1500"})
    assert "COMBINED_ENCODING_DRIFT_MIN_PERIOD_S" in left["cleared"]     # the primary profile clears a leftover arm
    assert "drift" in run_mode._CACHE
