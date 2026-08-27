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


@pytest.fixture(autouse=True)
def _no_server(monkeypatch):
    """Every test below is about the LOCAL directory, so the server is switched off by default.

    Without this, `load` would reach MICROSCOPE on every one of them: slow, and it would make the
    suite pass or fail on what happens to be published rather than on what the test set up. The
    fallback gets its own tests, which turn it back on explicitly.
    """
    monkeypatch.setattr(jl, "server_basis_dir", lambda: None)


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


# --------------------------------------------------------------------------------------------
# The server fallback.
#
# WHY IT EXISTS (Priya, 2026-08-26). The bases live in a LOCAL directory, so the behavior box ran the
# 8/24 and 8/25 analysis and then could not build a single joint-basis figure -- its deck hit the
# completeness gate and refused. `publish_basis` put them on the share; these cover the loader half.
# --------------------------------------------------------------------------------------------

def _write_basis(root, bid, when, ncomp=90, animal="PS99"):
    d = root / animal / bid
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps(
        {"basis_id": bid, "animal": animal, "labels": ["PS99_0101"], "ncomp": ncomp,
         "rank": 100, "seed": 0, "built_utc": when}))
    return d


def test_a_basis_only_on_the_server_is_found(tmp_path, monkeypatch):
    """The behavior box's exact situation: nothing local, the basis published to MICROSCOPE."""
    local, server = tmp_path / "local", tmp_path / "server"
    local.mkdir()
    _write_basis(server, "srv001", "2026-08-20T00:00:00Z", ncomp=95)
    monkeypatch.setattr(jl, "BASIS_DIR", local)
    monkeypatch.setattr(jl, "server_basis_dir", lambda: server)
    assert jl.load("PS99").basis_id == "srv001"
    assert jl.load("PS99", "srv001").ncomp == 95      # by explicit id too, not only "newest"


def test_the_newest_wins_across_roots(tmp_path, monkeypatch):
    """NOT local-first. Preferring local wholesale would serve a superseded reference frame on
    whichever box was behind -- an artifact whose name asserts a currency nothing checks, which is
    the same shape as the frozen-model contamination this repo already had once."""
    local, server = tmp_path / "local", tmp_path / "server"
    _write_basis(local, "old111", "2026-08-01T00:00:00Z")
    _write_basis(server, "new222", "2026-08-20T00:00:00Z")
    monkeypatch.setattr(jl, "BASIS_DIR", local)
    monkeypatch.setattr(jl, "server_basis_dir", lambda: server)
    assert jl.load("PS99").basis_id == "new222"

    # ...and the converse, so the test cannot pass by always choosing the server.
    _write_basis(local, "newest333", "2026-08-25T00:00:00Z")
    assert jl.load("PS99").basis_id == "newest333"


def test_the_local_copy_is_preferred_when_both_have_the_same_id(tmp_path, monkeypatch):
    """A basis_id is a hash of its own inputs, so the same id in both roots is the same bytes. Read
    the one that is not 180 MB of footprints over SMB."""
    local, server = tmp_path / "local", tmp_path / "server"
    _write_basis(local, "same999", "2026-08-20T00:00:00Z")
    _write_basis(server, "same999", "2026-08-20T00:00:00Z")
    monkeypatch.setattr(jl, "BASIS_DIR", local)
    monkeypatch.setattr(jl, "server_basis_dir", lambda: server)
    got = jl.load("PS99")
    assert got.basis_id == "same999"
    assert local in got.root.parents


def test_an_unreachable_server_does_not_break_loading(tmp_path, monkeypatch):
    """The share is down or unmounted on this box. A local basis must still load."""
    local = tmp_path / "local"
    _write_basis(local, "loc555", "2026-08-20T00:00:00Z")
    monkeypatch.setattr(jl, "BASIS_DIR", local)
    monkeypatch.setattr(jl, "server_basis_dir", lambda: tmp_path / "does_not_exist")
    assert jl.load("PS99").basis_id == "loc555"


def test_a_half_copied_basis_is_not_a_candidate(tmp_path, monkeypatch):
    """Publishing is a file-by-file copy, so a directory can exist mid-flight with an unreadable or
    absent manifest. That must not shadow a good basis, and must not raise."""
    local, server = tmp_path / "local", tmp_path / "server"
    _write_basis(local, "good777", "2026-08-01T00:00:00Z")
    (server / "PS99" / "partial888").mkdir(parents=True)
    (server / "PS99" / "partial888" / "manifest.json").write_text("{not json")
    monkeypatch.setattr(jl, "BASIS_DIR", local)
    monkeypatch.setattr(jl, "server_basis_dir", lambda: server)
    assert jl.load("PS99").basis_id == "good777"


def test_missing_everywhere_still_raises_and_names_both_places(tmp_path, monkeypatch):
    """A missing basis must stay an explicit build. The message has to say where it looked, or the
    fix ("publish it from the box that has it") is not discoverable from the error."""
    monkeypatch.setattr(jl, "BASIS_DIR", tmp_path / "local")
    monkeypatch.setattr(jl, "server_basis_dir", lambda: tmp_path / "server")
    with pytest.raises(FileNotFoundError) as ex:
        jl.load("PS99")
    assert "publish_basis" in str(ex.value)


def test_listing_says_which_root_answered(tmp_path, monkeypatch):
    """"Missing" and "on the share but not here" are different problems with different fixes."""
    local, server = tmp_path / "local", tmp_path / "server"
    _write_basis(local, "loc111", "2026-08-01T00:00:00Z")
    _write_basis(server, "srv222", "2026-08-20T00:00:00Z")
    monkeypatch.setattr(jl, "BASIS_DIR", local)
    monkeypatch.setattr(jl, "server_basis_dir", lambda: server)
    got = {r["basis_id"]: r["origin"] for r in jl.listing()}
    assert got == {"loc111": "local", "srv222": "server"}
