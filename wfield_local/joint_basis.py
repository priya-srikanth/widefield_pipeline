"""One shared low-rank basis across an animal's sessions — the input a JOINT LocaNMF needs.

The problem it solves: each session has its OWN SVD basis ``U_s`` (npix x K) and ``V_s`` (K x T_s),
and LocaNMF run per session therefore returns components that differ in count AND identity. Any
quantity that is not invariant to the basis — notably an RDM built as ``1 - corr`` between position
patterns — is then not comparable across sessions. (Measured 2026-08-12: PS94_0811 sat at z=-8.4 in
per-session LocaNMF space and z=-0.5 in Allen-ROI space. See DECISIONS.md "RSA basis".)

Concatenating the sessions is the principled fix, and it is CHEAP if done right. The naive route --
reconstruct each session's ``U_s V_s`` into pixel space and stack along time -- is
``npix x sum(T_s)``, about 345600 x 1.6M float32 = 2.2 TB. Never do that.

Instead use the fact that the left singular vectors of ``M = [U_1 V_1 | U_2 V_2 | ...]`` depend on
the sessions only through ``U_s (V_s V_s^T) U_s^T``. So build the thin matrix

    B = [ U_1 (V_1 V_1^T)^(1/2) | U_2 (V_2 V_2^T)^(1/2) | ... ]        (npix x S*K)

whose SVD has the SAME left singular vectors as ``M``. With K=100 and 8 sessions that is
345600 x 800 -- seconds, not terabytes. Then each session re-expresses exactly, with no
reconstruction:

    V_joint_s = U_joint^T U_s V_s = (U_joint^T U_s) V_s                (contract to K_j x K first)

Feed ``U_joint`` plus the concatenated ``V_joint`` to LocaNMF once and every session shares one set
of footprints: component *i* is the same cortical patch on every day.

This is exact for the span; it is a change of basis, not an approximation, up to the rank kept.
"""
from __future__ import annotations

import glob

import numpy as np


def _load_session(mc):
    """(U_flat (npix,K), SVT (K,T)) for one session, on the shared Allen grid, NaNs zeroed."""
    ad = glob.glob(f"{mc}/wfield_local_results/allen_aligned_affine8v1")
    if not ad:
        raise FileNotFoundError(f"no allen_aligned_affine8v1 under {mc}")
    U = np.load(f"{ad[0]}/U_atlas.npy")
    SVT = np.load(f"{mc}/wfield_local_results/SVTcorr.npy")
    return np.nan_to_num(U.reshape(-1, U.shape[2])).astype(np.float32), SVT.astype(np.float32)


def _whiten_factor(V):
    """(V V^T)^(1/2) via eigendecomposition -- the K x K factor that makes B's SVD match M's."""
    C = np.asarray(V @ V.T, dtype=np.float64)
    C = 0.5 * (C + C.T)                       # symmetrize away round-off
    w, Q = np.linalg.eigh(C)
    w = np.clip(w, 0.0, None)
    return (Q * np.sqrt(w)) @ Q.T


def build_joint_basis(mcs, rank=None, labels=None, normalize=True, verbose=True):
    """Shared basis over several sessions.

    ``mcs`` = motion_corrected dirs (one per session). Returns
    ``(U_joint (npix, R), {label: V_joint (R, T_s)}, singular_values)``.

    ``rank`` defaults to the max per-session K, which keeps the joint basis the same size as any one
    session's — enough to span the shared structure without inflating the LocaNMF problem.

    ``normalize=True`` scales each session's block to unit Frobenius norm before the joint SVD.
    Without it a session contributes in proportion to its total energy (length x dF/F scale), so the
    longest / highest-variance days dominate the shared basis and short or quiet ones are poorly
    represented. Measured on PS94: the unnormalized basis gave markedly WORSE cross-session RDM
    consistency than either Allen-ROI or a frozen reference basis, which is an artifact of exactly
    this. Equal weighting is what "a basis for THIS ANIMAL's pre-stroke activity" should mean.
    """
    labels = list(labels) if labels is not None else [str(i) for i in range(len(mcs))]
    Us, Vs = [], []
    for mc in mcs:
        u, v = _load_session(mc)
        Us.append(u); Vs.append(v)
    npix = Us[0].shape[0]
    if any(u.shape[0] != npix for u in Us):
        raise ValueError("sessions are not on a common pixel grid (need allen_aligned_affine8v1)")
    R = int(rank or max(u.shape[1] for u in Us))

    blocks = []
    for u, v in zip(Us, Vs):
        blk = u @ _whiten_factor(v).astype(np.float32)
        if normalize:
            blk = blk / max(float(np.linalg.norm(blk)), 1e-12)
        blocks.append(blk)
    B = np.concatenate(blocks, axis=1)
    if verbose:
        print(f"[joint_basis] B = {B.shape} (npix x sum K)  -> SVD, keeping rank {R}", flush=True)
    Uj, sv, _ = np.linalg.svd(B, full_matrices=False)
    Uj = np.ascontiguousarray(Uj[:, :R])

    Vj = {}
    for lab, u, v in zip(labels, Us, Vs):
        Vj[lab] = (Uj.T @ u) @ v              # (R,K)(K,T) -- never touches pixel-space time series
    if verbose:
        tot = float((sv ** 2).sum())
        kept = float((sv[:R] ** 2).sum())
        print(f"[joint_basis] {len(mcs)} sessions, rank {R}, "
              f"variance retained {100 * kept / max(tot, 1e-12):.2f}%", flush=True)
    return Uj, Vj, sv


def reconstruction_error(mc, Uj, Vj_s, n_probe=200, seed=0):
    """Relative error of the joint basis for ONE session, on a random subset of timepoints.

    Compares ``U_s V_s`` against ``U_joint V_joint_s`` at ``n_probe`` random frames (pixel space is
    only materialized for those columns). Returns ``||diff|| / ||orig||`` -- near 0 means the joint
    basis loses nothing for this session.
    """
    u, v = _load_session(mc)
    rs = np.random.RandomState(seed)
    idx = rs.choice(v.shape[1], size=min(n_probe, v.shape[1]), replace=False)
    orig = u @ v[:, idx]
    approx = Uj @ Vj_s[:, idx]
    return float(np.linalg.norm(orig - approx) / max(np.linalg.norm(orig), 1e-12))
