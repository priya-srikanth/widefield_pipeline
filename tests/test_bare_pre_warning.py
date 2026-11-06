"""Bare-run warning (Priya 11-06, Q1): a run with no WIDEFIELD_PRE_FROM, no render profile and no explicit "all" that
pools a June pre session warns ONCE that the cohort's August-only cut is not applied. It never changes a phase."""
import warnings

import pytest

from wfield_local import config, epoch_audit


@pytest.fixture
def bare(monkeypatch):
    for k in ("WIDEFIELD_PRE_FROM", "WIDEFIELD_REBUILD_ROOT", "WIDEFIELD_COHORT"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(config, "_FULL_PRE_CHOSEN", False)
    monkeypatch.setattr(config, "_BARE_PRE_WARNED", False)
    return monkeypatch


def _phase_warnings(date="0607"):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ph = config.session_phase("PS92", date)
    return ph, [x for x in w if "bare run" in str(x.message)]


def test_bare_run_warns_once_and_keeps_phase(bare):
    ph, w = _phase_warnings()
    assert ph == "pre" and len(w) == 1 and "0806" in str(w[0].message)
    assert "--pre-from 0801" in str(w[0].message)          # the primary run value: existing folders, no re-render
    assert _phase_warnings()[1] == []                      # once per process


def test_august_pre_does_not_warn(bare):
    assert _phase_warnings("0806") == ("pre", [])


@pytest.mark.parametrize("how", ["env", "empty", "profile", "all", "require_full", "full_cohort"])
def test_deliberate_choices_are_quiet(bare, how):
    if how == "env":
        bare.setenv("WIDEFIELD_PRE_FROM", "0801")
        assert config.session_phase("PS92", "0607") == "excluded"
    if how == "empty":
        bare.setenv("WIDEFIELD_PRE_FROM", "")              # nightly_figs: "" = include June on purpose
    if how == "profile":
        bare.setenv("WIDEFIELD_REBUILD_ROOT", "C:/x")
    if how == "all":
        config.apply_pre_from("all")
    if how == "require_full":
        config.require_full_pre("test")
    if how == "full_cohort":
        with epoch_audit.full_cohort():
            assert _phase_warnings()[1] == []
        return
    assert _phase_warnings()[1] == []
