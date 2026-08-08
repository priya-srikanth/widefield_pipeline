"""Unit tests for wfield_local.session_cache (no real data needed)."""
from pathlib import Path

from wfield_local import session_cache as sc


def _mk_session(tmp_path, label="PSXX_9999", cbytes=b"0" * 10):
    final = tmp_path / "mc" / "locanmf_affine8v1_final"
    final.mkdir(parents=True, exist_ok=True)
    (final / f"{label}_locanmf_C.npy").write_bytes(cbytes)
    h5 = tmp_path / f"{label}.h5"
    h5.write_bytes(b"h5")
    return {"label": label, "mc": str(tmp_path / "mc").replace("\\", "/"), "h5": str(h5)}


def _cfile(session):
    return Path(session["mc"], "locanmf_affine8v1_final", f"{session['label']}_locanmf_C.npy")


def test_roundtrip_and_hit(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "CACHE_DIR", tmp_path / "cache")
    s = _mk_session(tmp_path)
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return {"v": 123}

    r1 = sc.cached(s, "k", compute)
    r2 = sc.cached(s, "k", compute)
    assert r1 == r2 == {"v": 123}
    assert calls["n"] == 1  # second call was a cache hit


def test_invalidate_on_input_change(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "CACHE_DIR", tmp_path / "cache")
    s = _mk_session(tmp_path)
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return calls["n"]

    sc.cached(s, "k", compute)
    _cfile(s).write_bytes(b"1" * 99)  # simulate LocaNMF re-run (size+mtime change)
    sc.cached(s, "k", compute)
    assert calls["n"] == 2


def test_params_and_kind_are_separate(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "CACHE_DIR", tmp_path / "cache")
    s = _mk_session(tmp_path)
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return calls["n"]

    sc.cached(s, "k", compute, params="A")
    sc.cached(s, "k", compute, params="B")  # different params -> recompute
    sc.cached(s, "other", compute)          # different kind -> recompute
    assert calls["n"] == 3


def test_no_cache_env_bypasses(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setenv("WIDEFIELD_NO_CACHE", "1")
    s = _mk_session(tmp_path)
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return 1

    sc.cached(s, "k", compute)
    sc.cached(s, "k", compute)
    assert calls["n"] == 2  # always recompute, never read/write


def test_prune_superseded_signature(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "CACHE_DIR", tmp_path / "cache")
    s = _mk_session(tmp_path)
    sc.cached(s, "k", lambda: 1)
    _cfile(s).write_bytes(b"x" * 77)  # new signature
    sc.cached(s, "k", lambda: 2)
    files = list((tmp_path / "cache").glob(f"{s['label']}__k__*.pkl"))
    assert len(files) == 1  # old signature pruned, only the current one remains
