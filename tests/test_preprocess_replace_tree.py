"""The push's rmtree+copytree must survive an SMB share that holds the old directory open.

2026-09-28: the PS92_0922 redo pushed 73 minutes of results and then died at the last item,
``motion_qc`` -- rmtree had returned, the share still held the directory "delete pending", and
copytree's mkdir raised FileExistsError. Everything after the push (maps, photobleach, xall) was
left unrun. `_replace_tree` copies INTO a surviving directory and retries a refused copy.
"""
from __future__ import annotations

import shutil

import pytest

from wfield_local import preprocess


def _tree(root, files):
    root.mkdir(parents=True, exist_ok=True)
    for name, body in files.items():
        (root / name).write_text(body)
    return root


def test_replaces_the_destination_contents(tmp_path):
    src = _tree(tmp_path / "src", {"a.png": "new-a", "b.json": "new-b"})
    dst = _tree(tmp_path / "dst", {"a.png": "OLD", "stale.png": "gone"})
    preprocess._replace_tree(src, dst)
    assert sorted(p.name for p in dst.iterdir()) == ["a.png", "b.json"]
    assert (dst / "a.png").read_text() == "new-a"


def test_a_destination_the_share_will_not_release_is_copied_INTO(tmp_path, monkeypatch):
    """rmtree 'succeeds' but the directory is still there (delete pending, contents gone)."""
    src = _tree(tmp_path / "src", {"a.png": "new-a"})
    dst = _tree(tmp_path / "dst", {"a.png": "OLD"})
    real_rmtree = shutil.rmtree

    def rmtree_leaving_the_dir(path, ignore_errors=False, **kw):
        real_rmtree(path, ignore_errors=ignore_errors, **kw)
        dst.mkdir()                                     # the share keeps the (now empty) directory alive

    monkeypatch.setattr(preprocess.shutil, "rmtree", rmtree_leaving_the_dir)
    preprocess._replace_tree(src, dst)
    assert (dst / "a.png").read_text() == "new-a"


def test_a_refused_copy_is_retried_and_then_succeeds(tmp_path, monkeypatch):
    src = _tree(tmp_path / "src", {"a.png": "new-a"})
    dst = tmp_path / "dst"
    real_copytree = shutil.copytree
    calls = []

    def flaky(s, d, **kw):
        calls.append(1)
        if len(calls) < 3:
            raise PermissionError(13, "Access is denied", str(d))
        return real_copytree(s, d, **kw)

    monkeypatch.setattr(preprocess.shutil, "copytree", flaky)
    monkeypatch.setattr(preprocess.time, "sleep", lambda s: None)
    preprocess._replace_tree(src, dst, retries=4)
    assert len(calls) == 3 and (dst / "a.png").read_text() == "new-a"


def test_gives_up_with_instructions_after_the_retries(tmp_path, monkeypatch):
    src = _tree(tmp_path / "src", {"a.png": "x"})
    monkeypatch.setattr(preprocess.shutil, "copytree",
                        lambda s, d, **kw: (_ for _ in ()).throw(FileExistsError(17, "exists", str(d))))
    monkeypatch.setattr(preprocess.time, "sleep", lambda s: None)
    with pytest.raises(SystemExit, match="--skip-preprocess"):
        preprocess._replace_tree(src, tmp_path / "dst", retries=2)
