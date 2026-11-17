"""`o2_state_space_lnmf.enc_share`: the encoding bundle's folder, followed through a move-aside manifest (11-16: Fig 9.10's
script failed after `enc_rebuild_20261104b` was moved to `_superseded_glue_20261109/o2/` on 11-09). Fake roots only."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))


@pytest.fixture
def L(monkeypatch, tmp_path):
    import o2_state_space_lnmf as L
    monkeypatch.setattr(L.S, "REBUILD", tmp_path / "rebuild_x")
    (tmp_path / "rebuild_x" / "o2").mkdir(parents=True)
    return L


def _bundle(d: Path) -> Path:
    d.mkdir(parents=True)
    (d / "env.json").write_text("{}")
    return d


def _manifest(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["group", "kind", "source", "dest", "n_files", "bytes", "status", "time"])
        for src, dst, st in rows:
            w.writerow(["A", "dir", src, dst, 1, 1, st, "t"])


def test_live_bundle_wins(L):
    live = _bundle(L.S.REBUILD / "o2" / "enc_a")
    assert L.enc_share("enc_a") == live


def test_moved_bundle_rerooted_on_manifest(L):
    sup = L.S.REBUILD / "_superseded_glue_x"
    moved = _bundle(sup / "o2" / "enc_a")
    # recorded under another drive letter: re-rooted on the manifest's own folder
    _manifest(sup / "MANIFEST.csv", [(r"Q:\old\rebuild_x\o2\enc_a", r"Q:\old\rebuild_x\_superseded_glue_x\o2\enc_a",
                                      "moved")])
    assert L.enc_share("enc_a") == moved


def test_missing_raises(L):
    sup = L.S.REBUILD / "_superseded_glue_x"
    _bundle(sup / "o2" / "enc_a")
    _manifest(sup / "MANIFEST.csv", [(r"Q:\r\o2\enc_a", r"Q:\r\_superseded_glue_x\o2\enc_a", "skipped")])
    with pytest.raises(FileNotFoundError):
        L.enc_share("enc_a")
    with pytest.raises(FileNotFoundError):
        L.enc_share("enc_b")


def test_two_moved_copies_raise(L):
    for name in ("_superseded_a", "_superseded_b"):
        sup = L.S.REBUILD / name
        _bundle(sup / "o2" / "enc_a")
        _manifest(sup / "M.csv", [(r"Q:\r\o2\enc_a", rf"Q:\r\{name}\o2\enc_a", "moved")])
    with pytest.raises(FileNotFoundError, match="several"):
        L.enc_share("enc_a")
