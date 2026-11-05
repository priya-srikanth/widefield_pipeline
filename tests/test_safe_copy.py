"""safe_copy.verified_copy -- the one verified copy behind archive_day and camera_nightly uploads (rule 0).

Each scenario is also run through both tools' `_copy_one` so the two call sites cannot drift apart again.
"""
from __future__ import annotations

import hashlib
import os

import pytest

from wfield_local import archive_day, camera_nightly, safe_copy


def _src(tmp_path, data=b"acquired bytes " * 1000):
    p = tmp_path / "src" / "a.h5"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return p, hashlib.sha256(data).hexdigest()


def test_copy_ok_returns_measured_source_hash(tmp_path):
    src, digest = _src(tmp_path)
    dst = tmp_path / "dst" / "sub" / "a.h5"
    st, sh = safe_copy.verified_copy(src, dst, readback=True)
    assert (st, sh) == ("ok", digest)
    assert dst.read_bytes() == src.read_bytes()
    assert not (tmp_path / "dst" / "sub" / "a.h5.part").exists()
    assert src.exists()                                   # never moves / deletes the source


def test_size_only_copy_ok(tmp_path):
    src, digest = _src(tmp_path)
    dst = tmp_path / "dst" / "a.h5"
    assert safe_copy.verified_copy(src, dst) == ("ok", digest)


def test_existing_same_size_is_skip_and_readback_hashes_it(tmp_path):
    src, digest = _src(tmp_path)
    dst = tmp_path / "dst" / "a.h5"
    safe_copy.verified_copy(src, dst)
    assert safe_copy.verified_copy(src, dst) == ("skip", None)          # nothing measured -> no hash
    assert safe_copy.verified_copy(src, dst, readback=True) == ("skip", digest)


def test_same_size_corrupt_destination_detected_by_readback(tmp_path):
    src, digest = _src(tmp_path)
    dst = tmp_path / "dst" / "a.h5"
    dst.parent.mkdir(parents=True)
    dst.write_bytes(b"X" * src.stat().st_size)
    assert safe_copy.verified_copy(src, dst, readback=True) == ("FAIL", digest)
    # size-only cannot see it: that is the documented limit of readback=False
    assert safe_copy.verified_copy(src, dst) == ("skip", None)


def test_truncated_destination_is_recopied(tmp_path):
    src, digest = _src(tmp_path)
    dst = tmp_path / "dst" / "a.h5"
    dst.parent.mkdir(parents=True)
    dst.write_bytes(src.read_bytes()[:100])
    assert safe_copy.verified_copy(src, dst, readback=True) == ("ok", digest)
    assert dst.read_bytes() == src.read_bytes()


def _truncating_replace(real):
    def _rep(a, b):
        real(a, b)
        with open(b, "r+b") as fh:
            fh.truncate(10)
    return _rep


def test_destination_truncated_during_copy_is_FAIL(tmp_path, monkeypatch):
    src, digest = _src(tmp_path)
    dst = tmp_path / "dst" / "a.h5"
    monkeypatch.setattr(safe_copy.os, "replace", _truncating_replace(os.replace))
    assert safe_copy.verified_copy(src, dst) == ("FAIL", digest)


def test_destination_bytes_changed_during_copy_is_FAIL_with_readback(tmp_path, monkeypatch):
    src, digest = _src(tmp_path)
    dst = tmp_path / "dst" / "a.h5"
    real = os.replace

    def _flip(a, b):
        real(a, b)
        with open(b, "r+b") as fh:
            fh.write(b"Z")                                # same size, different bytes
    monkeypatch.setattr(safe_copy.os, "replace", _flip)
    assert safe_copy.verified_copy(src, dst, readback=True) == ("FAIL", digest)


def test_missing_source_refuses_and_creates_nothing(tmp_path):
    dst = tmp_path / "dst" / "sub" / "a.h5"
    with pytest.raises(FileNotFoundError):
        safe_copy.verified_copy(tmp_path / "nope.h5", dst, readback=True)
    assert not (tmp_path / "dst").exists()


def test_unreadable_pair_is_FAIL_not_a_match(tmp_path, monkeypatch):
    """camera_nightly used to compare _sha256(src) != _sha256(dst): two unreadable files (None, None) PASSED."""
    src, _ = _src(tmp_path)
    dst = tmp_path / "dst" / "a.h5"
    safe_copy.verified_copy(src, dst)
    monkeypatch.setattr(safe_copy, "sha256", lambda p: None)
    assert safe_copy.verified_copy(src, dst, readback=True) == ("FAIL", None)


def test_sha256_unreadable_is_None(tmp_path):
    assert safe_copy.sha256(tmp_path / "missing") is None
    assert safe_copy.size(tmp_path / "missing") == -1


# ---- both call sites use it ---------------------------------------------------------------------------------
def test_archive_day_copy_one(tmp_path):
    src, digest = _src(tmp_path)
    dst = str(tmp_path / "n" / "a.h5")
    assert archive_day._copy_one(str(src), dst, verify=True) == ("ok", digest)
    assert archive_day._copy_one(str(src), dst, verify=True) == ("skip", digest)
    assert archive_day._copy_one(str(src), dst, verify=False) == ("skip", None)
    assert archive_day._copy_one(str(src), dst, verify=True, big=True) == ("skip", None)   # big = size-only
    with open(dst, "r+b") as fh:
        fh.write(b"Q")
    assert archive_day._copy_one(str(src), dst, verify=True) == ("FAIL", digest)
    assert archive_day._sha256(str(src)) == digest


def test_camera_nightly_copy_one(tmp_path, monkeypatch):
    src, _ = _src(tmp_path)
    dst = str(tmp_path / "m" / "a.h5")
    assert camera_nightly._copy_one(str(src), dst, dry=True) == "dry"
    assert not os.path.exists(dst)
    assert camera_nightly._copy_one(str(src), dst, dry=False, verify=True) == "ok"
    assert camera_nightly._copy_one(str(src), dst, dry=False, verify=True) == "skip"
    with open(dst, "r+b") as fh:
        fh.write(b"Q")
    assert camera_nightly._copy_one(str(src), dst, dry=False, verify=True) == "FAIL"
    assert camera_nightly._copy_one(str(src), dst, dry=False, verify=False) == "skip"
    # at/above HASH_READBACK_MAX the read-back is skipped (size-only)
    monkeypatch.setattr(camera_nightly, "HASH_READBACK_MAX", 1)
    assert camera_nightly._copy_one(str(src), dst, dry=False, verify=True) == "skip"
