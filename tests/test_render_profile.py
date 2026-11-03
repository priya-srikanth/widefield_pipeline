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
    assert "frozen_models_rebuild" in env["WIDEFIELD_FROZEN_MODEL_DIR"]
    assert env["WIDEFIELD_FROZEN_MODEL_DIR"].rstrip("/\\") != "C:/Users/SabatiniLab/frozen_models"
    assert "rebuild" in env["WIDEFIELD_SESSION_CACHE"]
    assert "WIDEFIELD_PRE_FROM" not in env                     # Aug-only pre is per decoder block, never global
    assert profile["block_args"]["epoch_grant_figures"] == ["--pre-from", "0801"]
    assert profile["notes"]["encoder_drift_cutoff_s"] == 600


def test_shell_lines():
    lines = render_profile.shell_lines(NAME, "powershell")
    assert '$env:WIDEFIELD_TRIAL_GATE = "camera"' in lines
    assert any(ln.startswith("export WIDEFIELD_HEMO_VARIANT=") for ln in render_profile.shell_lines(NAME, "bash"))
