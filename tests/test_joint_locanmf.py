"""Tests for the persisted joint LocaNMF basis.

The point of the module is that a basis is a FIXED reference frame with honest provenance, so the
tests target exactly that: the id must change when the inputs change, must NOT change when they don't,
and a saved basis must never be silently replaced by a refit.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from wfield_local import config
from wfield_local import joint_locanmf as jl


@pytest.fixture
def fake_sessions(tmp_path):
    """Two sessions with real files on disk, so the mtime/size signatures are meaningful."""
    out = []
    for lab in ("PS99_0101", "PS99_0102"):
        mc = tmp_path / lab
        (mc / "wfield_local_results" / "allen_aligned_affine8v1").mkdir(parents=True)
        np.save(mc / "wfield_local_results" / "allen_aligned_affine8v1" / "U_atlas.npy",
                np.zeros((4, 4, 3), np.float32))
        # write the SVTcorr the CONFIGURED hemo variant resolves to, not the bare literal: the
        # signature follows config.svtcorr_path, so a fixture pinned to the bare file would make
        # both signatures read MISSING and the id compare equal for the wrong reason.
        svt = Path(config.svtcorr_path(mc))
        svt.parent.mkdir(parents=True, exist_ok=True)
        np.save(svt, np.zeros((3, 10), np.float32))
        out.append({"label": lab, "mc": str(mc)})
    return out


def test_basis_id_is_stable_for_identical_inputs(fake_sessions):
    assert jl.basis_id(fake_sessions, rank=100) == jl.basis_id(fake_sessions, rank=100)


def test_basis_id_ignores_session_ORDER_but_not_membership(fake_sessions):
    """Order is an accident of how the caller listed sessions; membership is the basis."""
    assert jl.basis_id(fake_sessions, 100) == jl.basis_id(list(reversed(fake_sessions)), 100)
    assert jl.basis_id(fake_sessions[:1], 100) != jl.basis_id(fake_sessions, 100)


def test_basis_id_changes_with_rank_and_seed(fake_sessions):
    base = jl.basis_id(fake_sessions, 100)
    assert jl.basis_id(fake_sessions, 200) != base
    assert jl.basis_id(fake_sessions, 100, seed=7) != base


def test_basis_id_changes_when_an_input_file_changes(fake_sessions):
    """THE case that motivates signatures: a re-preprocess upstream must invalidate the basis, and no
    mtime of the basis's own files would reveal it."""
    before = jl.basis_id(fake_sessions, 100)
    svt = config.svtcorr_path(fake_sessions[0]["mc"])
    np.save(svt, np.zeros((3, 11), np.float32))          # different size -> different signature
    assert jl.basis_id(fake_sessions, 100) != before


def test_missing_input_is_recorded_not_silently_ignored(fake_sessions):
    sig = jl._session_sig({"label": "x", "mc": "/nonexistent/path"})
    assert "MISSING" in sig


def test_load_raises_when_no_basis_exists(tmp_path, monkeypatch):
    """A missing basis must be an explicit build. Falling back to a refit is what the module exists
    to prevent."""
    monkeypatch.setattr(jl, "BASIS_DIR", tmp_path)
    with pytest.raises(FileNotFoundError):
        jl.load("PS99")


def test_load_returns_newest_and_specific_ids_still_reachable(tmp_path, monkeypatch):
    monkeypatch.setattr(jl, "BASIS_DIR", tmp_path)
    for bid, when, ncomp in (("aaa111", "2026-08-01T00:00:00Z", 90),
                             ("bbb222", "2026-08-20T00:00:00Z", 95)):
        root = tmp_path / "PS99" / bid
        root.mkdir(parents=True)
        (root / "manifest.json").write_text(json.dumps(
            {"basis_id": bid, "animal": "PS99", "labels": ["PS99_0101"], "ncomp": ncomp,
             "rank": 100, "seed": 0, "built_utc": when}))
    assert jl.load("PS99").basis_id == "bbb222"          # newest by build time, not by name
    assert jl.load("PS99", "aaa111").ncomp == 90         # the superseded one is still readable


def test_signal_rejects_a_session_not_in_the_basis(tmp_path, monkeypatch):
    monkeypatch.setattr(jl, "BASIS_DIR", tmp_path)
    root = tmp_path / "PS99" / "ccc333"
    root.mkdir(parents=True)
    (root / "manifest.json").write_text(json.dumps(
        {"basis_id": "ccc333", "animal": "PS99", "labels": ["PS99_0101"], "ncomp": 5,
         "rank": 100, "seed": 0, "built_utc": "2026-08-01T00:00:00Z"}))
    with pytest.raises(KeyError):
        jl.load("PS99").signal("PS99_9999")


def test_stale_basis_is_detected_not_silently_reused(fake_sessions, tmp_path, monkeypatch):
    """basis_id makes two bases distinguishable on disk; it does NOT stop a stale one being LOADED.

    On 2026-08-14 the SVTcorr variant flipped and LocaNMF was refit, and every joint figure kept being
    built on bases fitted to superseded data while the ROI figures had already moved -- visible only by
    recomputing the id. `load()` returns the newest basis regardless, so the check has to be explicit.
    """
    class _B:
        labels = [s["label"] for s in fake_sessions]
        basis_id = "deadbeef0000"
        ncomp = 100
        manifest = {"rank": 100}

    ok, exp = jl.basis_is_current(_B(), fake_sessions)
    assert not ok and exp and exp != "deadbeef0000"

    _B.basis_id = exp                      # a basis built from exactly these inputs is current
    ok2, _ = jl.basis_is_current(_B(), fake_sessions)
    assert ok2


def test_basis_with_unknown_labels_is_not_current(fake_sessions):
    """A basis naming sessions we cannot resolve cannot be verified, so it must not pass as current."""
    class _B:
        labels = ["PS99_9999"]
        basis_id = "whatever0000"
        ncomp = 100
        manifest = {"rank": 100}

    ok, exp = jl.basis_is_current(_B(), fake_sessions)
    assert not ok and exp is None


def test_a_freshly_built_basis_reads_as_CURRENT(fake_sessions, tmp_path, monkeypatch):
    """basis_id hashes the rank ARGUMENT (build is called with rank=None) while the manifest stores
    the RESOLVED rank. Recomputing with the resolved value marked all four freshly-built bases stale --
    a guard that flags everything is worthless."""
    class _B:
        labels = [s["label"] for s in fake_sessions]
        ncomp = 90
        manifest = {"rank": 100}
        basis_id = jl.basis_id(fake_sessions, None)      # as build() actually computes it

    ok, _ = jl.basis_is_current(_B(), fake_sessions)
    assert ok, "a basis built with rank=None must not read as stale"

    class _Old(_B):
        basis_id = "0000deadbeef"
    ok2, exp = jl.basis_is_current(_Old(), fake_sessions)
    assert not ok2 and exp, "a genuinely stale basis must still be caught"
