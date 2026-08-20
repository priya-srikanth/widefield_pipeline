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
    p = Path(atlas_dir) / "allen_brain_mask_native_grid.npy"
    if p.exists():
        return np.load(p).astype(bool)
    # fall back to the atlas labels, which define the same territory a different way
    for name in ("allen_area_atlas_native_grid.npy", "atlas_labels.npy", "atlas.npy"):
        q = Path(atlas_dir) / name
        if q.exists():
            a = np.load(q)
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


def session_geometry(session, align="cue", post_all_trials=True):
    """Per-position pixel-space patterns and the crossnobis RDM for ONE session.

    The features are the RAW SVD coefficients (`source="svt"`), so nothing is averaged into regions
    or components first -- then mapped through the Gram whitener, after which Euclidean distance IS
    pixel distance. Bins are averaged to the WINDOW MEAN, which is what "the SVD map" means and what
    fixed_scale_maps shows; keeping them would measure the temporal profile as well as the map.

    Everything downstream (the crossnobis estimator, the position-matched pre-stroke band) is the
    same code `spatial_reorganisation` uses on ROI features, so the pixel and parcellated versions
    differ in the FEATURES ALONE and can be read against each other.
    """
    import numpy as np

    from wfield_local.locanmf_frozen_decoder import _args
    from wfield_local.locanmf_position_decoder import _trial_features
    from wfield_local.plot_lick_aligned_averages import DISPLAY_ORDER, POSITION_NAMES
    from wfield_local.spatial_reorganisation import MIN_TRIALS_PER_POS, _crossnobis

    G = gram(session)
    if G is None:
        return None
    L = pixel_whitener(G)
    X, y, g, Xn, yn, reg = _trial_features(session, _args(source="svt", align=align, post_s=2.0))
    lab = session["label"]
    if (post_all_trials
            and config.session_phase(config.animal_of(lab), lab.split("_")[-1]) == "post"
            and len(yn)):
        X = np.vstack([X, Xn])
        y = np.concatenate([y, yn])
        g = np.concatenate([g, np.arange(g.max() + 1, g.max() + 1 + len(yn))])
    if len(y) < 30:
        return None
    k = int(np.unique(reg).size)
    A = np.asarray(X, float).reshape(len(y), -1, k).mean(axis=1)      # window-mean map
    Z = to_pixel_space(A, L)                                          # Euclidean here = pixel space

    labels = [c for c in DISPLAY_ORDER if (y == c).sum() >= MIN_TRIALS_PER_POS]
    D = _crossnobis(Z, np.asarray(y), np.asarray(g), labels)
    if D is None:
        return None
    iu = np.triu_indices(len(labels), k=1)
    vals = D[iu][np.isfinite(D[iu])]
    return {"label": lab, "animal": config.animal_of(lab), "date": lab.split("_")[-1],
            "align": align, "positions": [POSITION_NAMES[c] for c in labels],
            "crossnobis": D.tolist(), "n_pairs": int(vals.size),
            "mean_distance": float(vals.mean()) if vals.size else float("nan"),
            "pattern": {POSITION_NAMES[c]: Z[y == c].mean(axis=0).tolist() for c in labels}}


def collect(animals=None, align="cue", post_all_trials=True):
    rows = []
    for s in config.analysis_sessions(animals=animals):
        try:
            r = session_geometry(s, align=align, post_all_trials=post_all_trials)
        except Exception as ex:                                       # noqa: BLE001
            print(f"  {s['label']}: skip ({type(ex).__name__} {str(ex)[:60]})", flush=True)
            continue
        if r is None:
            continue
        r["phase"] = config.session_phase(r["animal"], r["date"])
        rows.append(r)
        print(f"  {r['label']} {align:6s} pixel crossnobis {r['mean_distance']:.4f}  "
              f"({len(r['positions'])} positions)", flush=True)
    return sorted(rows, key=lambda r: (r["animal"], r["date"]))


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
