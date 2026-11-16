"""The cohort SVD basis read as pixel maps and Allen regions: ONE loader for the state-space map scripts (11-13 cleanup).

Before 11-13, eight scripts (chronic_maps, part_maps, rest_decomp, shift_behavior / carried / precue / rawref /
spout_timing) each re-loaded the basis, rebuilt the split-area masks and re-derived the region weights and the pixel-RMS
unit inline. They now call this module; the values are unchanged (same files, same arithmetic, same dict order).

  basis()            U_cohort (brain-mask px x K, orthonormal), atlas, mask, the mask's flat pixel index and the SPLIT
                     area masks (ROI v8: ALM / tjM1 carved out, `area_basis.basis_masks(...)["split"]`), cached per
                     process. Read-only by convention: callers must not modify the arrays in place.
  region_weights()   {"<region> <side>": K-vector}: the pixel-weighted mean of U_cohort over a region's areas on one
                     side, so (session x K) @ w = the region's mean dF/F. Sides: contra = right, ipsi = left (all four
                     lesions are left).
  pre_unit_px()      the PIXEL RMS of an animal's pre template (spout means minus their grand mean): the unit every
                     map / region readout of these scripts is expressed in. Not chronic_maps.full_unit (RMS of the pre
                     mean response, not of its spout deviations) and not change_parts' unit -- different quantities.
  roi_outlines()     full-frame boolean masks of areas to outline on the maps (ALM / tjM1 by default).

Inputs: `state_space_sim.SVD_BASIS` (U_cohort.npy, mask.npy) and `state_space_sim.BUNDLE` (atlas.npy) -- WF_SS_BASIS=svd.
"""
from __future__ import annotations

import functools
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import state_space_sim as S  # noqa: E402

SIDES = {"contra": "right", "ipsi": "left"}         # readout side -> hemisphere suffix (lesion left = ipsilesional)


@functools.lru_cache(maxsize=1)
def basis():
    """SimpleNamespace(Uc, atlas, mask, flat, split, npix) of the cohort SVD basis, loaded once per process."""
    from wfield_local import area_basis, beta_maps
    Uc = np.load(S.SVD_BASIS / "U_cohort.npy").astype(np.float64)
    atlas = np.load(S.BUNDLE / "atlas.npy")
    mask = np.load(S.SVD_BASIS / "mask.npy").reshape(atlas.shape)
    flat = np.flatnonzero(mask.ravel())
    split = area_basis.basis_masks(area_basis.roi_masks(atlas.ravel()[flat], beta_maps.stat_mask().ravel()[flat],
                                                        dict(S._atlas_names()), flat, mask.shape))["split"]
    return SimpleNamespace(Uc=Uc, atlas=atlas, mask=mask, flat=flat, split=split, npix=Uc.shape[0])


def region_weights(regions=None, sides=SIDES):
    """{"<region> <side>": K-vector} for {region: [areas]} (default `state_space_sim.SHIFT_REGIONS`); an area absent on
    a side is skipped. Order = regions, then sides, as given."""
    b = basis()
    W = {}
    for reg, areas in (S.SHIFT_REGIONS if regions is None else regions).items():
        for side, hemi in sides.items():
            px = np.logical_or.reduce([np.asarray(b.split[f"{a}_{hemi}"], bool) for a in areas
                                       if f"{a}_{hemi}" in b.split])
            W[f"{reg} {side}"] = b.Uc[px].mean(0)
    return W


def pre_unit_px(Tpre, npix=None):
    """Pixel RMS of a pre template's spout deviations (spouts x bins x K; spouts with a NaN dropped)."""
    ok = np.isfinite(Tpre).all(axis=(1, 2))
    B0 = Tpre[ok] - S.flat(Tpre[ok]).mean(0)
    return float(np.sqrt((B0 ** 2).sum() / B0.size * B0.shape[-1] / (npix or basis().npix)))


def roi_outlines(keys=("ALM_left", "ALM_right", "tjM1_left", "tjM1_right")):
    """Full-frame boolean masks of the given split areas (drawn as outlines on the pixel maps)."""
    b = basis()
    out = []
    for k in keys:
        f = np.zeros(b.mask.size, bool)
        f[b.flat[b.split[k]]] = True
        out.append(f.reshape(b.mask.shape))
    return out
