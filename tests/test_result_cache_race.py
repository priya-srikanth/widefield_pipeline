"""result_cache publishes atomically and tolerates a concurrent writer (2026-10-21: WinError 32 dropped four sessions
from the first parallel LP_licks render)."""
import os

from wfield_local import result_cache


def test_failed_publish_returns_value_and_leaves_no_tmp(tmp_path, monkeypatch):
    def boom(a, b):
        raise PermissionError(32, "in use")
    monkeypatch.setattr(os, "replace", boom)
    v = result_cache.cached(tmp_path, "k", [], "v1", lambda: {"x": 1})
    assert v == {"x": 1}
    assert not list(tmp_path.glob("*.tmp"))


def test_tmp_name_is_per_process(tmp_path):
    result_cache.cached(tmp_path, "k", [], "v1", lambda: 3)
    assert (tmp_path / "k.pkl").exists() and not list(tmp_path.glob("*.tmp"))
    assert result_cache.cached(tmp_path, "k", [], "v1", lambda: 4) == 3      # served from the cache
