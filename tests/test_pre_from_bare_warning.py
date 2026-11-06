"""A bare cohort-1 hand run (no render profile, no WIDEFIELD_PRE_FROM) pools June + August pre sessions; `config.pre_from`
prints ONE loud stderr WARNING per process naming the August-only default (A4, Priya 11-06). Nothing else changes.
cohort2, a render profile, an explicit WIDEFIELD_PRE_FROM (even empty), `apply_pre_from(<value>)` and
`require_full_pre` print nothing."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from wfield_local import config, epoch_audit, paths

ENV = "WIDEFIELD_PRE_FROM"


@pytest.fixture
def bare(monkeypatch):
    """A fresh process state: nothing chosen, nothing warned, no profile, cohort1."""
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.delenv("WIDEFIELD_REBUILD_ROOT", raising=False)
    monkeypatch.delenv(paths.COHORT_ENV, raising=False)
    monkeypatch.setattr(config, "_PRE_FROM_CHOSEN", False)
    monkeypatch.setattr(config, "_PRE_FROM_WARNED", False)
    return monkeypatch


def _warnings(capsys):
    return capsys.readouterr().err.count("WARNING [pre-stroke baseline]")


def test_bare_cohort1_warns_once(bare, capsys):
    labs = config.phase_labels("pre", machine="analysis")       # many pre_from() calls
    for _ in range(5):
        assert config.pre_from() is None
    err = capsys.readouterr().err
    assert err.count("WARNING [pre-stroke baseline]") == 1
    assert "JUNE + AUGUST" in err and "WIDEFIELD_PRE_FROM=0801" in err and "0806" in err
    assert any(lab.endswith("_0606") or lab.endswith("_0607") for lab in labs)   # selection unchanged: June still in


def test_warning_changes_no_selection(bare, capsys):
    a = (config.phase_labels("pre", machine="analysis"), config.curated_dates(machine="analysis"))
    bare.setattr(config, "_PRE_FROM_CHOSEN", True)                # silenced: same pools
    assert (config.phase_labels("pre", machine="analysis"), config.curated_dates(machine="analysis")) == a


def test_cohort2_silent(bare, capsys):
    bare.setenv(paths.COHORT_ENV, "cohort2")
    config.phase_labels("pre", machine="analysis")
    config.pre_from()
    assert _warnings(capsys) == 0


def test_profile_silent(bare, capsys, tmp_path):
    bare.setenv("WIDEFIELD_REBUILD_ROOT", str(tmp_path))
    assert config.pre_from() is None
    assert _warnings(capsys) == 0


@pytest.mark.parametrize("val", ["0801", ""])
def test_explicit_env_silent(bare, capsys, val):
    bare.setenv(ENV, val)
    config.pre_from()
    with epoch_audit.full_cohort():                               # a later suspension is not a bare run
        assert config.pre_from() is None
    assert _warnings(capsys) == 0


def test_apply_pre_from_all_silent(bare, capsys):
    config.apply_pre_from("all")
    assert config.pre_from() is None
    assert _warnings(capsys) == 0


def test_require_full_pre_silent(bare, capsys):
    config.require_full_pre("june_august_drift")
    assert config.pre_from() is None
    assert _warnings(capsys) == 0


def test_once_per_fresh_process():
    """End to end in a clean interpreter: one warning for a bare run, none with WIDEFIELD_PRE_FROM set."""
    root = Path(__file__).resolve().parents[1]
    code = ("from wfield_local import config\n"
            "config.phase_labels('pre', machine='analysis'); config.pooled_labels(machine='analysis')\n")
    base = {k: v for k, v in os.environ.items() if not k.startswith(("WIDEFIELD_", "WFIELD_", "COMBINED_ENCODING_"))}
    base["PYTHONPATH"] = str(root)
    r = subprocess.run([sys.executable, "-c", code], env=base, cwd=root, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    assert r.stderr.count("WARNING [pre-stroke baseline]") == 1
    r2 = subprocess.run([sys.executable, "-c", code], env=dict(base, WIDEFIELD_PRE_FROM="0801"), cwd=root,
                        capture_output=True, text=True, timeout=300)
    assert r2.returncode == 0, r2.stderr
    assert "WARNING [pre-stroke baseline]" not in r2.stderr
