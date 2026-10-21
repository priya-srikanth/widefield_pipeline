"""Automatic stroke footprint: a SUGGESTION for the painter, not the published mask.

Rules agreed with Priya (2026-10-21), each one a correction of an earlier version:
  * Evidence of stroke = NeuN-blank, OR GFAP scar ring with low NeuN, OR the autofluorescent necrotic core
    (bright in every channel), each judged against the MIRROR point on the intact side (min / max over ~75 um).
    Normal GPe (higher GFAP, lower neuron density on BOTH sides) is therefore not flagged; a necrotic core counts
    wherever it is, GPe included. Ventricles (no DAPI) are not NeuN-blank tissue.
  * "Absent" territory = the contralesional striatum mirrored onto the lesioned side, ANCHORED on the surviving
    striatum's intact medial + dorsomedial border (atrophy / ex-vacuo ventricle enlargement shifts the surviving
    striatum, so a plain mirror misplaces it), never rescaled, minus the surviving striatum (Priya's QuPath
    outline, less any evidence inside it).
  * Footprint = the visible lesion (evidence connected to the core, within 1.5 mm), always fully included, spilled
    LATERALLY then DORSALLY (not radially) into the absent territory. Medial and usually dorsal striatum are spared.
    Because of encephalomalacia the absent area exceeds the visible lesion -- expected.
On PS92-95 these suggestions overlapped Priya's painted masks at IoU 0.3-0.7 (2026-10-21), mostly where she added
the scar ball and tissue inferior to the striatum; treat them as a starting point.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from shapely import affinity
from skimage import measure, morphology

from wfield_local.histology_map import UM_PER_PX_L3 as UM
from wfield_local.histology_map.sections import rasterize


def _boundary_pts(g, n=400):
    p = max(getattr(g, "geoms", [g]), key=lambda q: q.area)
    return np.array([p.exterior.interpolate(t, normalized=True).coords[0] for t in np.linspace(0, 1, n, endpoint=False)])


def anchor_mirror(gc, gi, x0, max_shift=120, max_rot=10):
    """Mirror the contra striatum about x0 and place it rigidly on the ipsi outline's medial + dorsomedial border."""
    mc = affinity.scale(gc, -1, 1, origin=(x0, 0))
    P = _boundary_pts(gi); c = np.array(gi.centroid.coords[0]); v = P - c; R = np.abs(v).max()
    A = P[(v[:, 0] > 0.25 * R) | ((v[:, 1] < 0) & (v[:, 0] > -0.2 * R))]
    best = None
    for rot in np.arange(-max_rot, max_rot + 0.1, 2.5):
        tree = cKDTree(_boundary_pts(affinity.rotate(mc, rot, origin=gi.centroid), 600))
        for dx in np.arange(-max_shift, max_shift + 1, 8):
            for dy in np.arange(-max_shift, max_shift + 1, 8):
                sc = np.median(tree.query(A - [dx, dy])[0])
                if best is None or sc < best[0]:
                    best = (sc, rot, dx, dy)
    _, rot, dx0, dy0 = best
    tree = cKDTree(_boundary_pts(affinity.rotate(mc, rot, origin=gi.centroid), 600))
    for dx in np.arange(dx0 - 8, dx0 + 9, 2):
        for dy in np.arange(dy0 - 8, dy0 + 9, 2):
            m = np.median(tree.query(A - [dx, dy])[0])
            if m < best[0]:
                best = (m, rot, dx, dy)
    _, rot, dx, dy = best
    return affinity.translate(affinity.rotate(mc, rot, origin=gi.centroid), dx, dy)


def evidence(T):
    """Boolean maps (evidence, autofluorescent core), lesioned side only, mirror-referenced."""
    img, tm, x0 = T["img"], T["tissue"], T["x0"]; sh = tm.shape
    sm = [ndi.gaussian_filter(ch, 3) for ch in img]
    cm = rasterize(T["gc"], sh); ref = [np.median(ch[cm]) for ch in sm]
    dapi, gfap, neun = [ch / (r + 1e-9) for ch, r in zip(sm, ref)]
    xs = np.arange(sh[1])[None, :]; xm = np.clip(np.round(2 * x0 - xs).astype(int), 0, sh[1] - 1)[0]
    neun_m = ndi.grey_erosion(neun[:, xm], size=15); gfap_m = ndi.grey_dilation(gfap[:, xm], size=15)
    autof = (dapi > 2.0) & (neun > 1.5) & (gfap > 3.0)
    blank = (neun / (neun_m + 1e-6) < 0.55) & (neun_m > 0.6) & (dapi > 0.2)
    scar = (gfap > 2.0 * np.maximum(gfap_m, 1)) & (neun < 1.2 * np.maximum(neun_m, 0.3))
    ev = morphology.opening((autof | blank | scar) & (xs < x0) & tm, morphology.disk(2))
    return ev, autof


def suggest(T, max_um=1500):
    """dict(stroke, seed, missing, mirror, flags) in the standardised section frame."""
    sh = T["tissue"].shape
    mirror = anchor_mirror(T["gc"], T["gi"], T["x0"])
    mcm = rasterize(mirror, sh); im = rasterize(T["gi"], sh)
    ev, autof = evidence(T)
    ev_in = ev & ndi.binary_dilation(mcm | im, iterations=4)
    missing = mcm & ~(im & ~ev_in)
    missing = morphology.remove_small_objects(morphology.opening(missing, morphology.disk(3)), 100)
    lab = measure.label(ev); near = ndi.binary_dilation(mcm, iterations=30)
    cand = [p for p in measure.regionprops(lab) if near[lab == p.label].any()]
    if not cand:
        return dict(stroke=missing.copy(), seed=np.zeros(sh, bool), missing=missing, mirror=mirror,
                    flags=["no lesion evidence: using the absent area"])
    core = max(cand, key=lambda p: (autof[lab == p.label].sum(), p.area)); cy, cx = core.centroid
    yy, xx = np.mgrid[0:sh[0], 0:sh[1]]
    seed = np.isin(lab, [p.label for p in cand]) & (np.hypot(yy - cy, xx - cx) * UM <= max_um)
    seed = measure.label(ndi.binary_fill_holes(morphology.closing(seed, morphology.disk(5))))
    seed = seed == (seed[int(cy), int(cx)] or np.argmax(np.bincount(seed[seed > 0])))
    allowed = missing | seed
    lat = np.array([[0, 0, 0], [1, 1, 0], [0, 0, 0]], bool)   # leftward = lateral (lesion is image-left)
    dor = np.array([[0, 0, 0], [0, 1, 0], [0, 1, 0]], bool)   # upward = dorsal
    s = seed.copy()
    for _ in range(50):
        before = s.sum()
        for se in (lat, dor):
            for _ in range(400):
                s2 = (s | ndi.binary_dilation(s, structure=se)) & allowed
                if s2.sum() == s.sum():
                    break
                s = s2
        if s.sum() == before:
            break
    s = ndi.binary_fill_holes(morphology.closing(s, morphology.disk(3)))
    flags = []
    unreached = (missing & ~s).sum() / max(missing.sum(), 1)
    if unreached > 0.3:
        flags.append(f"{unreached:.0%} of the absent area is not connected to the lesion")
    return dict(stroke=s, seed=seed, missing=missing, mirror=mirror, flags=flags)
