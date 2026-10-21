"""Section -> atlas plate transform.

Fitted on the INTACT (contralesional) hemisphere only, anchored on the anatomical midline (Priya, 2026-10-21:
"do scaling by the contra hemisphere only, aligning midline"), with independent x and y scale:

    x_mm = (x - midline_x) * kx * s            (negative = left = lesioned side)
    y_mm = ty + (y - y_top) * ky * s           (y from the plate's page top)
    s    = 5.2 um / px  (level-3 crop)

Score = median distance (mm) from the section's hemisphere outline (midline cut excluded) to the plate's brain
outline + from its contralesional striatum outline (QuPath) to the plate's CPu outline. Boundary distance, not area
overlap: overlap scoring over-sized every section (Priya). kx, ky in 0.82-1.15 (an unconstrained fit collapsed
PS95 s5 to 0.71 on a torn ventral edge). When striatum and cortex edge disagree the result favours the striatum
(Priya chose this explicitly; PS93's cortex edge then sits slightly inside the atlas outline). The lesioned
hemisphere is carried by the same transform and never rescaled.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from wfield_local.histology_map import UM_PER_PX_L3 as UM
from wfield_local.histology_map.sections import rasterize

DS = 2              # fit on the plate raster at PX_PER_MM / DS
K_RANGE = (0.82, 1.15)


def _edge(m):
    return m & ~ndi.binary_erosion(m)


def atlas_half(R):
    b = R["brain"][::DS, ::DS]; c = R["cpu"][::DS, ::DS]; mid = b.shape[1] // 2
    return b[:, mid:], c[:, mid:], R["ppm"] / DS


def fit_section(T, plate_rasters: dict, figures):
    """Best (err_mm, fig, kx, ky, ty) over the candidate figures (sorted best first) + y_top."""
    tm = T["tissue"]; sh = tm.shape; x0 = T["x0"]; xs = np.arange(sh[1])[None, :]
    tis = tm & (xs >= x0)
    ey, ex = np.nonzero(_edge(tis) & (xs > x0 + 3))
    cy, cx = np.nonzero(_edge(rasterize(T["gc"], sh)))
    y_top = int(np.nonzero(tis.any(1))[0].min()); s = UM / 1000
    out = []
    for f in figures:
        b, c, ppm = atlas_half(plate_rasters[f])
        db = ndi.distance_transform_edt(~_edge(b)); dc = ndi.distance_transform_edt(~_edge(c))
        a_top = np.nonzero(b.any(1))[0].min() / ppm

        def score(kx, ky, ty, ppm=ppm, db=db, dc=dc):
            def med(yy, xx, dist):
                r = ty * ppm + (yy - y_top) * ky * s * ppm; q = (xx - x0) * kx * s * ppm
                ok = (r >= 0) & (r < dist.shape[0]) & (q >= 0) & (q < dist.shape[1])
                return np.median(dist[r[ok].astype(int), q[ok].astype(int)]) if ok.mean() >= 0.9 else 1e9
            return (med(ey, ex, db) + med(cy, cx, dc)) / ppm

        best = None
        for kx in np.arange(K_RANGE[0], K_RANGE[1] + 0.01, 0.03):
            for ky in np.arange(K_RANGE[0], K_RANGE[1] + 0.01, 0.03):
                for dty in np.arange(-0.4, 0.41, 0.1):
                    v = score(kx, ky, a_top + dty)
                    if best is None or v < best[0]:
                        best = (v, kx, ky, a_top + dty)
        _, kx0, ky0, ty0 = best
        for kx in np.arange(kx0 - 0.05, kx0 + 0.051, 0.01):
            for ky in np.arange(ky0 - 0.05, ky0 + 0.051, 0.01):
                for ty in np.arange(ty0 - 0.1, ty0 + 0.101, 0.02):
                    v = score(kx, ky, ty)
                    if v < best[0]:
                        best = (v, kx, ky, ty)
        out.append(dict(err_mm=float(best[0]), figure=int(f), kx=float(best[1]), ky=float(best[2]), ty=float(best[3])))
    out.sort(key=lambda d: d["err_mm"])
    return out, y_top


def to_mm(x, y, rec):
    """Standardised section px -> plate mm, using a sections.json record."""
    s = UM / 1000
    return (np.asarray(x) - rec["midline_x"]) * rec["kx"] * s, rec["ty"] + (np.asarray(y) - rec["y_top"]) * rec["ky"] * s
