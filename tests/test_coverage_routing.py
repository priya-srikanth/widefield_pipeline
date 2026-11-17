"""Coordinator 11-04: camera_coverage_tables writes camera_coverage.csv INSIDE the rebuild root under the profile (the
untagged path `trial_classes.coverage_note` reads), and a missing table is LOUD in a tagged run."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def tagged(monkeypatch):
    for k in list(os.environ):
        if k.startswith(("WIDEFIELD_", "WFIELD_", "COMBINED_ENCODING_")):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("WIDEFIELD_TRIAL_GATE", "camera")
    monkeypatch.setenv("WIDEFIELD_PRE_FROM", "0801")
    return monkeypatch


def test_script_writes_inside_the_rebuild_root_instead_of_refusing():
    src = (REPO / "scripts" / "camera_coverage_tables.py").read_text(encoding="utf-8")
    assert "if _tagged and not _rr.active():" in src                     # refuses only OUTSIDE a rebuild root
    assert "_rm.register_data_store(qc / n)" in src and "coverage_path().name" in src


def test_registered_data_file_is_writable_in_a_tagged_run_and_its_sibling_is_not(tagged):
    from wfield_local import run_mode, writeguard
    qc = Path("M:/MICROSCOPE/Priya/Widefield/rebuild_x/LP_analysis/run/imaging_encoding/qc")
    with pytest.raises(writeguard.WriteGuardError):
        writeguard.assert_writable(qc / "camera_coverage.csv")
    run_mode.register_data_store(qc / "camera_coverage.csv")
    try:
        writeguard.assert_writable(qc / "camera_coverage.csv")            # the registered FILE
        with pytest.raises(writeguard.WriteGuardError):
            writeguard.assert_writable(qc / "camera_coverage_other.csv")  # exact file only, not a prefix
        with pytest.raises(writeguard.WriteGuardError):
            writeguard.assert_writable(qc / "trial_attempts.csv")
    finally:
        run_mode._DATA_STORES.discard(run_mode._norm(qc / "camera_coverage.csv"))


def test_missing_coverage_is_loud_in_a_tagged_run(tagged, tmp_path):
    from wfield_local import trial_classes as TC
    with pytest.raises(FileNotFoundError, match="camera_coverage_tables"):
        TC.coverage_note(tmp_path / "camera_coverage.csv")
    tagged.delenv("WIDEFIELD_PRE_FROM")
    tagged.delenv("WIDEFIELD_TRIAL_GATE")
    assert "missing" in TC.coverage_note(tmp_path / "camera_coverage.csv")   # the default run: a caption, as before


def test_stamp_reraises_a_missing_coverage_table_in_a_tagged_run(tagged, monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local import lp_licks
    from wfield_local import trial_classes as TC

    def boom(*a, **k):
        raise FileNotFoundError("camera_coverage.csv")
    monkeypatch.setattr(TC, "coverage_note", boom)
    fig = plt.figure()
    try:
        with pytest.raises(FileNotFoundError):
            lp_licks.stamp(fig)
    finally:
        plt.close(fig)
