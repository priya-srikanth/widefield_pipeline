"""Epoch boundaries are a property of the COHORT; a caller's `--only` subset must not change them.

2026-09-28: `nightly_figs 20260922 --only PS92` exported WIDEFIELD_ONLY_ANIMALS=PS92 and then derived
the epochs. `config.phase_labels` / `pooled_labels` honour that variable, so PS93/94/95 had no series,
and the rule wrote chronic_from=None for PS93 and PS95 into the SHARED epoch_boundaries.json.
"""
from __future__ import annotations

import os

from wfield_local import config, epoch_audit


def test_full_cohort_suspends_the_subset_and_restores_it(monkeypatch):
    monkeypatch.setenv("WIDEFIELD_ONLY_ANIMALS", "PS92")
    monkeypatch.setenv("WIDEFIELD_ONLY_DATES", "0922")
    assert config.pooled_labels("PS93") == [], "the subset is in force outside"
    with epoch_audit.full_cohort():
        assert "WIDEFIELD_ONLY_ANIMALS" not in os.environ and "WIDEFIELD_ONLY_DATES" not in os.environ
        assert config.pooled_labels("PS93"), "inside, every animal's sessions are visible"
        assert any(lab.startswith("PS95") for lab in config.phase_labels("pre"))
    assert os.environ["WIDEFIELD_ONLY_ANIMALS"] == "PS92" and os.environ["WIDEFIELD_ONLY_DATES"] == "0922"


def test_full_cohort_restores_even_when_the_body_raises(monkeypatch):
    monkeypatch.setenv("WIDEFIELD_ONLY_ANIMALS", "PS92")
    try:
        with epoch_audit.full_cohort():
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert os.environ["WIDEFIELD_ONLY_ANIMALS"] == "PS92"


def test_audit_under_a_subset_still_derives_every_animal(monkeypatch):
    """Against the real cohort table when this box can see it; the derived spec must name all four."""
    import pytest

    if not epoch_audit._cohort_path().exists():
        pytest.skip("no cohort table on this box")
    monkeypatch.setenv("WIDEFIELD_ONLY_ANIMALS", "PS92")
    rep = epoch_audit.audit()
    assert rep["available"]
    spec = epoch_audit.derived_spec(rep)
    assert set(spec) == {"PS92", "PS93", "PS94", "PS95"}
    for animal, r in rep["chronic"].items():
        for name, ser in r["series"].items() if "series" in r else []:
            assert "no usable" not in str(ser.get("note", "")), (animal, name, ser)
