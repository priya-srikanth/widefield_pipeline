"""The joint basis must SPAN the sessions it was built from, and weight them equally.

Synthetic data only -- the maths is what is being pinned, not the rig.
"""
from __future__ import annotations

import numpy as np

from wfield_local import joint_basis


def _fake(npix, K, T, rng, scale=1.0):
    U = rng.standard_normal((npix, K)).astype(np.float32)
    V = (rng.standard_normal((K, T)) * scale).astype(np.float32)
    return U, V


def test_joint_basis_spans_every_session(monkeypatch):
    """Rank >= total independent structure -> each session reconstructs essentially exactly."""
    rng = np.random.default_rng(0)
    npix, K = 400, 6
    sess = {f"s{i}": _fake(npix, K, 200 + 30 * i, rng) for i in range(3)}
    monkeypatch.setattr(joint_basis, "_load_session", lambda mc: sess[mc])

    Uj, Vj, _ = joint_basis.build_joint_basis(list(sess), rank=K * len(sess),
                                              labels=list(sess), verbose=False)
    assert Uj.shape == (npix, K * len(sess))
    for lab, (U, V) in sess.items():
        err = np.linalg.norm(U @ V - Uj @ Vj[lab]) / np.linalg.norm(U @ V)
        assert err < 1e-4, f"{lab} not spanned by the joint basis (rel err {err:.2e})"


def test_normalize_stops_a_loud_session_dominating(monkeypatch):
    """A session with 100x the amplitude must not monopolise the shared basis."""
    rng = np.random.default_rng(1)
    npix, K = 300, 4
    quiet = _fake(npix, K, 300, rng, scale=1.0)
    loud = _fake(npix, K, 300, rng, scale=100.0)
    sess = {"quiet": quiet, "loud": loud}
    monkeypatch.setattr(joint_basis, "_load_session", lambda mc: sess[mc])

    def quiet_err(normalize):
        Uj, Vj, _ = joint_basis.build_joint_basis(["quiet", "loud"], rank=K, labels=["quiet", "loud"],
                                                  normalize=normalize, verbose=False)
        U, V = quiet
        return np.linalg.norm(U @ V - Uj @ Vj["quiet"]) / np.linalg.norm(U @ V)

    assert quiet_err(normalize=True) < quiet_err(normalize=False), \
        "normalizing must improve the quiet session's representation"


def test_rank_defaults_to_max_session_k(monkeypatch):
    rng = np.random.default_rng(2)
    sess = {"a": _fake(200, 5, 100, rng), "b": _fake(200, 9, 100, rng)}
    monkeypatch.setattr(joint_basis, "_load_session", lambda mc: sess[mc])
    Uj, _, _ = joint_basis.build_joint_basis(list(sess), labels=list(sess), verbose=False)
    assert Uj.shape[1] == 9


def test_mismatched_pixel_grids_are_rejected(monkeypatch):
    rng = np.random.default_rng(3)
    sess = {"a": _fake(200, 4, 50, rng), "b": _fake(250, 4, 50, rng)}
    monkeypatch.setattr(joint_basis, "_load_session", lambda mc: sess[mc])
    try:
        joint_basis.build_joint_basis(list(sess), labels=list(sess), verbose=False)
    except ValueError as e:
        assert "common pixel grid" in str(e)
    else:
        raise AssertionError("must refuse sessions on different grids")
