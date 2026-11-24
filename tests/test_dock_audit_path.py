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


def test_redirected_subset_run_refuses_missing_copy(tmp_path, monkeypatch):
    import pytest
    monkeypatch.setattr(il, "ITI_SUBSET", "dock")
    monkeypatch.setattr(il, "OUT_W", tmp_path / "readers_dock")
    monkeypatch.setenv("WFIELD_ITI_OUT_W", str(tmp_path / "readers_dock"))
    with pytest.raises(FileNotFoundError, match="dock copy"):
        il.pred_file("zscore_check/where_preds_z.csv")
    (tmp_path / "readers_dock" / "zscore_check").mkdir(parents=True)
    q = tmp_path / "readers_dock" / "zscore_check" / il.tagged("where_preds_z.csv", il.PRED_TAG)
    q.write_text("x\n")
    assert il.pred_file("zscore_check/where_preds_z.csv") == q


def test_truncated_dock_audit_refused_under_the_fused_train(tmp_path, monkeypatch):
    import pytest

    from wfield_local import session_licks as SL
    old = tmp_path / "old.csv"
    old.write_text("label,k,dock\nA,0,True\n")
    new = tmp_path / "new.csv"
    new.write_text("label,k,dock,iti_class\nA,0,True,docked\n")
    monkeypatch.setattr(SL, "fused_on", lambda: True)
    with pytest.raises(SystemExit, match="truncated"):
        il.refuse_truncated_dock_audit(old)
    il.refuse_truncated_dock_audit(new)
    monkeypatch.setattr(SL, "fused_on", lambda: False)
    il.refuse_truncated_dock_audit(old)          # cam4 lick set: the old audit is the right one


def test_nonlinear_preds_and_sm_coherence_read_through_the_redirect():
    root = Path(__file__).resolve().parents[1] / "scripts"
    t = (root / "lick_nonlinear_preds.py").read_text(encoding="utf-8")
    assert 'il.OUT / "lick_order_preds' not in t and 'in_path_x("lick_order_preds' in t
    t = (root / "sm_coherence_cca.py").read_text(encoding="utf-8")
    assert 'pd.read_csv(ihc.IN / "iti_dock_audit.csv"' not in t and "refuse_truncated_dock_audit" in t


def test_enlv_table_freeze_is_opt_in():
    t = (Path(__file__).resolve().parents[1] / "scripts" / "lick_enlv_balance_offset.py").read_text(encoding="utf-8")
    assert '"--allow-freeze" not in sys.argv' in t and "LOAD_ONLY_ENV" in t


def test_prediction_caches_are_written_under_out_x_only():
    root = Path(__file__).resolve().parents[1] / "scripts"
    import re
    for f in root.glob("*.py"):
        t = f.read_text(encoding="utf-8", errors="replace")
        assert not re.search(r"to_csv\(il\.pred_file\(", t), f"{f.name} writes through pred_file (a READ path)"
