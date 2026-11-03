"""Per-session LocaNMF time courses C -- stored, or PROJECTED onto the stored footprints (rebuild 2026-11-03, item j).

THE ONE PLACE a per-session ``<label>_locanmf_C.npy`` is read (`load_C`); every consumer in wfield_local goes through
it (pinned by tests/test_locanmf_session.py, which greps for direct loads).

    WIDEFIELD_LOCANMF_PROJECT unset (DEFAULT): `load_C` returns the stored C from `config.locanmf_dir` -- byte for
        byte what the consumers read before (they called ``np.load`` on the same path). The directory follows the ONE
        hemo switch (`config.locanmf_dir_name`), so under a non-default hemo variant with no decomposition of its own
        the load FAILS (FileNotFoundError) rather than serving the adopted C under the variant's label.
    WIDEFIELD_LOCANMF_PROJECT=1: footprints A (and the region labels) are the STORED decomposition's
        (``defaults.yaml locanmf.output_dir_name``); C is the least-squares projection of the SELECTED variant's SVTcorr
        (`config.svtcorr_path`) onto A, ``C = pinv(A_in' A_in) A_in' U_in SVT`` restricted to the pixels inside any
        footprint (LocaNMF writes NaN outside each footprint's region; within the mask NaN = 0). The same operator as
        `joint_locanmf.Basis._operator` and scripts/locanmf_refit_test/refit_vs_projection.project_C; measured
        11-02 to reproduce the stored C of PS93_0826 at median r .9994 (min .977) on the adopted data.

Why projection and not a refit: a LocaNMF refit is not reproducible on this box (95 vs 93 components for the adopted
PS93_0826), so a refit would confound the hemo change with a new decomposition. Which one the rebuild uses is Priya's
decision (refit vs projection, LOCANMF_REFIT_TEST_20261103.md); this path is needed either way.

Projected C is cached per session in the session cache (kind ``lnmfproj-C``, tagged by `run_mode.cache_tag`, keyed on
the SVTcorr / U / stored C signatures); the projection runs a U_atlas load and one (ncomp, K) x (K, T) product.
"""
from __future__ import annotations

import glob

import numpy as np

from wfield_local import config

#: Bump when the projection arithmetic changes (enters the cache kind).
PROJECTION_VERSION = 1


def projection_on() -> bool:
    return config.locanmf_projection_on()


def stored_path(s: dict, what: str = "C") -> str:
    """``<locanmf_dir>/<label>_locanmf_<what>.npy`` (A, C or regions) of the decomposition `config.locanmf_dir` names."""
    return f"{config.locanmf_dir(s['mc'])}/{s['label']}_locanmf_{what}.npy"


def projection_operator(A, U_flat) -> tuple[np.ndarray, np.ndarray]:
    """``(op (ncomp, K), inside (npix,))``: ``op @ SVT`` is the least-squares C on fixed footprints ``A`` (H, W, ncomp).

    Contracted as ``pinv(A'A) (A' U)`` so the (npix, T) movie is never formed (~276 GB for a long session)."""
    Araw = np.asarray(A, dtype=np.float32).reshape(-1, np.asarray(A).shape[-1])
    inside = np.isfinite(Araw).any(axis=1)
    Ai = np.nan_to_num(Araw[inside]).astype(np.float64)
    Ui = np.nan_to_num(np.asarray(U_flat)[inside]).astype(np.float64)
    G = Ai.T @ Ai
    return np.linalg.pinv(G) @ (Ai.T @ Ui), inside


def project_C(A, U_flat, SVT) -> np.ndarray:
    """(ncomp, T) float64 -- C of the fixed footprints ``A`` for the movie ``U_flat @ SVT``."""
    op, _inside = projection_operator(A, U_flat)
    return op @ np.asarray(SVT, dtype=np.float64)


def _project_session(s: dict) -> np.ndarray:
    ad = glob.glob(f"{s['mc']}/wfield_local_results/allen_aligned_affine8v1")
    if not ad:
        raise FileNotFoundError(f"no allen_aligned_affine8v1 under {s['mc']}")
    U = np.load(f"{ad[0]}/U_atlas.npy")
    U_flat = U.reshape(-1, U.shape[-1])
    A = np.load(stored_path(s, "A"), mmap_mode="r")
    SVT = np.load(config.svtcorr_path(s["mc"]))
    return project_C(A, U_flat, SVT).astype(np.float32)


def load_C(s: dict) -> np.ndarray:
    """The session's C (ncomp, T): stored (default; the array ``np.load`` returns) or projected (see the docstring)."""
    if not projection_on():
        return np.load(stored_path(s, "C"))
    from wfield_local import session_cache
    return session_cache.cached(s, f"lnmfproj-C-p{PROJECTION_VERSION}", lambda: _project_session(s), verbose=False)
