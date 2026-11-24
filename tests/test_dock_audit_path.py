"""iti_licks.dock_audit_path: the docked-ITI filters read a rebuilt audit from OUT_W / PATCH_W / IN_W (lane E 11-24),
and OUT/iti_dock_audit.csv when no redirect is set (unchanged default)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import iti_licks as il  # noqa: E402


def _mk(d: Path) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    (d / "iti_dock_audit.csv").write_text("label,k,dock\n")
    return d / "iti_dock_audit.csv"


def test_default_reads_production(tmp_path, monkeypatch):
    monkeypatch.delenv("WFIELD_ITI_OUT_W", raising=False)
    monkeypatch.setattr(il, "OUT", tmp_path / "prod")
    assert il.dock_audit_path() == tmp_path / "prod" / "iti_dock_audit.csv"


def test_patch_folder_wins_over_upstream_and_production(tmp_path, monkeypatch):
    monkeypatch.setattr(il, "OUT", tmp_path / "prod")
    monkeypatch.setattr(il, "OUT_W", tmp_path / "readers")
    (tmp_path / "readers").mkdir()
    monkeypatch.setenv("WFIELD_ITI_OUT_W", str(tmp_path / "readers"))
    monkeypatch.setenv("WFIELD_ITI_PATCH_W", str(tmp_path / "patch"))
    monkeypatch.setenv("WFIELD_ITI_IN_W", str(tmp_path / "nt"))
    _mk(tmp_path / "prod"); _mk(tmp_path / "nt")
    assert il.dock_audit_path() == tmp_path / "nt" / "iti_dock_audit.csv"
    p = _mk(tmp_path / "patch")
    assert il.dock_audit_path() == p
    monkeypatch.delenv("WFIELD_ITI_PATCH_W"); monkeypatch.delenv("WFIELD_ITI_IN_W")
    (tmp_path / "prod2").mkdir()
    monkeypatch.setattr(il, "OUT", tmp_path / "prod2")
    assert il.dock_audit_path() == tmp_path / "prod2" / "iti_dock_audit.csv"
