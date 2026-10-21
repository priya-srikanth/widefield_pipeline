"""Pins combined_encoding's local input cache (CACHE_ENV): off when unset, one build per key, a changed source file or
COMBINED_ENCODING_* switch makes a new entry. No share access."""
from __future__ import annotations

import os
import time

from wfield_local import combined_encoding as ce


def test_cache_builds_once_and_refreshes_on_change(tmp_path, monkeypatch):
    src = tmp_path / "src.npy"
    src.write_bytes(b"a")
    calls = []

    def build():
        calls.append(1)
        return {"n": len(calls)}
    monkeypatch.delenv(ce.CACHE_ENV, raising=False)
    assert ce._cached("k", "PS92_0806", [src], build) == {"n": 1}          # off: plain build, nothing written
    monkeypatch.setenv(ce.CACHE_ENV, str(tmp_path / "cache"))
    assert ce._cached("k", "PS92_0806", [src], build) == {"n": 2}
    assert ce._cached("k", "PS92_0806", [src], build) == {"n": 2}          # hit
    time.sleep(0.01)
    src.write_bytes(b"bb")                                                   # source changed -> new entry
    os.utime(src, None)
    assert ce._cached("k", "PS92_0806", [src], build) == {"n": 3}
    monkeypatch.setenv(ce.EXEC_ENV, "polar")                                 # a switch changed -> new entry
    assert ce._cached("k", "PS92_0806", [src], build) == {"n": 4}
    assert len(list((tmp_path / "cache" / "k").glob("*.pkl"))) == 3
