"""tools/protected_mtime_audit.py: stat-only listing of files changed in the protected roots after a time (rule 11)."""
from __future__ import annotations

import importlib.util
import os
import textwrap
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _tool():
    spec = importlib.util.spec_from_file_location("protected_mtime_audit", ROOT / "tools" / "protected_mtime_audit.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture
def tree(tmp_path, monkeypatch):
    share = tmp_path / "Priya"
    prod = share / "Widefield" / "prod"
    (prod / "a" / "b").mkdir(parents=True)
    old, new, deep = prod / "old.csv", prod / "a" / "new.csv", prod / "a" / "b" / "deep.csv"
    for f in (old, new, deep):
        f.write_text("x", encoding="utf-8")
    t = time.time()
    os.utime(old, (t - 7200, t - 7200))
    reg = tmp_path / "reg.yaml"
    reg.write_text(textwrap.dedent("""
        share:
          - path: Widefield/prod
            kind: production
          - path: Widefield/missing_root
            kind: patch
        local: []
        """), encoding="utf-8")
    for k in ("WIDEFIELD_FROZEN_MODEL_DIR",):
        monkeypatch.delenv(k, raising=False)
    from wfield_local import protected_roots as pr
    monkeypatch.setattr(pr, "_REG", None)
    return share, reg, t, (old, new, deep)


def test_lists_only_files_after_since(tree, capsys):
    share, reg, t, (old, new, deep) = tree
    rc = _tool().main(["--since", str(t - 600), "--registry", str(reg), "--share-root", str(share)])
    out = capsys.readouterr().out
    assert rc == 1
    assert str(new) in out and str(deep) in out and str(old) not in out


def test_depth_and_until(tree, capsys):
    share, reg, t, (old, new, deep) = tree
    _tool().main(["--since", str(t - 600), "--depth", "1", "--registry", str(reg), "--share-root", str(share)])
    out = capsys.readouterr().out
    assert str(new) in out and str(deep) not in out
    rc = _tool().main(["--since", str(t - 9000), "--until", str(t - 3600), "--registry", str(reg),
                       "--share-root", str(share)])
    out = capsys.readouterr().out
    assert rc == 1 and str(old) in out and str(new) not in out


def test_no_hits_exit_zero_and_time_formats(tree):
    share, reg, t, _ = tree
    m = _tool()
    assert m.main(["--since", str(t + 3600), "--registry", str(reg), "--share-root", str(share)]) == 0
    assert abs(m.parse_time("2026-11-24 14:04") - time.mktime(time.strptime("2026-11-24 14:04", "%Y-%m-%d %H:%M"))) < 1
    m.parse_time("14:04")
    with pytest.raises(ValueError):
        m.parse_time("yesterday")
