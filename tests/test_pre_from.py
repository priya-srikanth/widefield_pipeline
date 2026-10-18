"""`WIDEFIELD_PRE_FROM` (late-baseline sensitivity mode, 2026-10-18): unset = nothing changes; set = early pre
sessions are 'excluded' everywhere; the epoch audit suspends it."""
import os

from wfield_local import config, epoch_audit


def _labels():
    return (config.phase_labels("pre", machine="analysis"), config.phase_labels("post", machine="analysis"),
            config.curated_dates(machine="analysis", phase="pre"))


def test_unset_is_a_no_op(monkeypatch):
    monkeypatch.delenv("WIDEFIELD_PRE_FROM", raising=False)
    a = _labels()
    monkeypatch.setenv("WIDEFIELD_PRE_FROM", "")
    assert _labels() == a


def test_set_drops_only_early_pre(monkeypatch):
    monkeypatch.delenv("WIDEFIELD_PRE_FROM", raising=False)
    pre, post, dates = _labels()
    monkeypatch.setenv("WIDEFIELD_PRE_FROM", "0801")
    pre2, post2, dates2 = _labels()
    assert post2 == post                                       # post-stroke untouched
    assert pre2 == [lab for lab in pre if lab.split("_")[1] >= "0801"]
    assert dates2 == [d for d in dates if d >= "0801"]
    assert all(config.session_phase(lab.split("_")[0], lab.split("_")[1]) == "excluded"
               for lab in pre if lab.split("_")[1] < "0801")


def test_epoch_audit_suspends_it(monkeypatch):
    monkeypatch.setenv("WIDEFIELD_PRE_FROM", "0801")
    with epoch_audit.full_cohort():
        assert "WIDEFIELD_PRE_FROM" not in os.environ
    assert os.environ["WIDEFIELD_PRE_FROM"] == "0801"
