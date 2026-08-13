"""Tests for the persisted joint LocaNMF basis.

The point of the module is that a basis is a FIXED reference frame with honest provenance, so the
tests target exactly that: the id must change when the inputs change, must NOT change when they don't,
and a saved basis must never be silently replaced by a refit.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

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
        np.save(mc / "wfield_local_results" / "SVTcorr.npy", np.zeros((3, 10), np.float32))
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
    svt = f"{fake_sessions[0]['mc']}/wfield_local_results/SVTcorr.npy"
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
