"""RSA on the SVD MAPS THEMSELVES -- pixel-space geometry, with no parcellation anywhere in it.

Priya, 2026-08-19: "can we do RSA on the SVD maps themselves?"

WHY THIS IS NOT JUST "RUN THE RDM ON SVT". The obvious shortcut is to compute distances between the
per-position SVT coefficient vectors and call it pixel space, on the grounds that the map is `U @ svt`
and U is a fixed linear map. That is only valid if U's columns are ORTHONORMAL, and after Allen
registration they are not:

    UtU on PS94_0812, over 345,600 finite pixels, k=100
        diagonal            0.782 - 1.296   (mean 1.158)
        max |off-diagonal|  0.362           (31% of the mean diagonal)
        ||UtU - I||_F / sqrt(k) = 0.234

The affine warp to the Allen grid resamples the spatial components, and resampling does not preserve
orthogonality. So a coefficient-space RDM silently applies the WRONG METRIC -- it treats components
as orthogonal and equally scaled when they are neither, off by ~23% in Frobenius norm. That is the
same class of error as an unnormalised RDM: it looks like a geometry and is partly a basis artefact.

WHAT THIS DOES INSTEAD. The exact pixel-space squared distance between two coefficient vectors is

    d2(a, b) = (a - b)^T G (a - b),    G = U^T U over the brain mask

and G is only k x k, so the whole thing is cheap and needs no pixel maps in memory. Taking the
Cholesky factor G = L^T L and transforming z = L a makes ordinary Euclidean distance in z EXACTLY
pixel distance -- so every existing crossnobis / RDM routine can be reused unchanged on z, with no
approximation and no special-casing downstream.

The mask matters: G computed over the whole frame is dominated by background pixels, which carry
registration edge artefacts and no signal. G is taken over the Allen brain mask only.

WHAT IT IS FOR. Every geometry result in the deck currently rests on a parcellation -- Allen ROIs or
LocaNMF components. If the pre/post convergence and the midline null survive in pixel space, they are
not artefacts of how the cortex was divided up. If they do not, that is worth knowing before any of
it is written up.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from wfield_local import config
from wfield_local.paths import PathResolver


def _atlas_dir(s):
    d = glob.glob(f"{s['mc']}/wfield_local_results/allen_aligned_affine8v1")
    return d[0] if d else None


def brain_mask(atlas_dir) -> np.ndarray | None:
    """Boolean (H, W) mask of pixels inside the Allen atlas, or None if no atlas is available.

    Background is excluded deliberately: it carries registration edge artefacts and no signal, and
    including it lets those pixels dominate the Gram matrix that defines the metric.
    """
    for name in ("atlas_labels.npy", "atlas.npy", "allen_atlas.npy"):
        p = Path(atlas_dir) / name
        if p.exists():
            a = np.load(p)
            if a.ndim == 2:
                return np.isfinite(a) & (a != 0)
    return None


def gram(session, mask=None) -> np.ndarray | None:
    """``U^T U`` over the brain mask: the metric that turns coefficient distance into PIXEL distance.

    Returned as a k x k matrix, so nothing pixel-sized is ever held in memory.
    """
    ad = _atlas_dir(session)
    if ad is None:
        return None
    U = np.load(f"{ad}/U_atlas.npy")
    H, W, k = U.shape
    F = U.reshape(-1, k)
    m = np.isfinite(F).all(axis=1)
    if mask is None:
        mask = brain_mask(ad)
    if mask is not None and mask.shape == (H, W):
        m &= mask.reshape(-1)
    return (F[m].T @ F[m]).astype(np.float64)


def pixel_whitener(G, eps=1e-8) -> np.ndarray:
    """``L`` with ``G = L^T L``, so Euclidean distance in ``z = L a`` IS pixel distance.

    Cholesky where possible; an eigen-decomposition fallback keeps a numerically semi-definite G
    usable rather than failing on a matrix that is fine to within rounding.
    """
    G = np.asarray(G, float)
    G = 0.5 * (G + G.T)
    try:
        return np.linalg.cholesky(G).T
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(G)
        w = np.clip(w, eps, None)
        return (V * np.sqrt(w)).T


def to_pixel_space(coeffs, L) -> np.ndarray:
    """Map (n_trials, k) SVT coefficients into the space where Euclidean distance = pixel distance."""
    return np.asarray(coeffs, float) @ np.asarray(L, float).T


def basis_distortion(G) -> dict:
    """How far the coefficient metric is from the pixel metric -- the number that justifies all this.

    ``relative_frobenius`` is ||G - I||_F / sqrt(k): zero when the basis is orthonormal, so a
    coefficient-space RDM is exact; large when it is not, so a coefficient-space RDM is measuring
    the basis as much as the brain.
    """
    G = np.asarray(G, float)
    k = G.shape[0]
    d = np.diag(G)
    off = G - np.diag(d)
    return {"k": int(k),
            "diag_min": float(d.min()), "diag_max": float(d.max()), "diag_mean": float(d.mean()),
            "max_abs_offdiag": float(np.abs(off).max()),
            "offdiag_over_diag": float(np.abs(off).max() / max(d.mean(), 1e-12)),
            "relative_frobenius": float(np.linalg.norm(G - np.eye(k)) / np.sqrt(k))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--animals", nargs="+", default=None)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args(argv)
    out = args.output or Path(PathResolver().root("figures_working"))
    rows = {}
    for s in config.analysis_sessions(animals=args.animals):
        G = gram(s)
        if G is None:
            continue
        rows[s["label"]] = basis_distortion(G)
        r = rows[s["label"]]
        print(f"  {s['label']:11s} k={r['k']:3d}  diag {r['diag_min']:.3f}-{r['diag_max']:.3f}  "
              f"max|off|={r['max_abs_offdiag']:.3f}  ||G-I||/sqrt(k)={r['relative_frobenius']:.3f}",
              flush=True)
    p = Path(out) / "pixel_rsa_basis_distortion.json"
    json.dump(rows, open(p, "w"), indent=1, default=float)
    print(f"\nwrote {p}  ({len(rows)} sessions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
