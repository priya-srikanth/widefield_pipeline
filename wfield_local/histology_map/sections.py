"""Load and standardise histology sections for the stroke map.

A section = a level-3 crop of the VS200 scan (``ets.crop_jobs``) + its two QuPath striatum outlines (the pair from
``scripts/import_histology_areas.py``; ipsi = the smaller outline). Standardisation, in order:

  1. FLIP so the lesion is on the image LEFT (Priya: "stroke should always be on the left, regardless of
     histology orientation"; flipped mounts are mirrored back).
  2. ROTATE so the midline is vertical: the angle that maximises mirror overlap of the tissue mask
     (``symmetry_midline``). Only the angle is used from it.
  3. MIDLINE = the dorsal interhemispheric fissure (``fissure_x``), NOT the mirror-symmetry axis: atrophy of the
     lesioned hemisphere pulls the symmetry axis 0.1-0.3 mm toward the intact side (measured on PS92-95,
     2026-10-21), which then mis-scales the intact hemisphere against the atlas.

Every number a published figure depends on (flip, angle, rotation centre, midline, crop of the painter image) is
stored in ``configs/histology_map/<cohort>/sections.json`` and re-used, so painted masks stay registered.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from shapely import affinity
from skimage import filters, measure, morphology, transform

from wfield_local.histology_map import UM_PER_PX_L3 as UM


def stretch(a, lo=1, hi=99.5):
    p = np.percentile(a[a > 0] if (a > 0).any() else a, (lo, hi))
    return np.clip((a - p[0]) / (p[1] - p[0] + 1e-9), 0, 1)


def rasterize(geom, shape):
    from skimage.draw import polygon as dp
    m = np.zeros(shape, bool)
    for p in getattr(geom, "geoms", [geom]):
        x, y = p.exterior.xy; rr, cc = dp(np.array(y), np.array(x), shape); m[rr, cc] = True
        for h in p.interiors:
            x, y = h.xy; rr, cc = dp(np.array(y), np.array(x), shape); m[rr, cc] = False
    return m


def load_raw(crop_npz, ann, row):
    """Crop image (C, H, W float) + ipsi / contra outlines (shapely, crop px) for one section-table row."""
    z = np.load(crop_npz); img = z["img"].astype(np.float32); org = z["origin"]; ds = int(z["ds"])
    a = ann[ann.entry_id == row.entry_id]
    pick = lambda x, y: a.iloc[((a.cx_px - x) ** 2 + (a.cy_px - y) ** 2).argmin()].geometry
    tf = lambda g: affinity.scale(affinity.translate(g, -org[0], -org[1]), 1 / ds, 1 / ds, origin=(0, 0))
    return img, tf(pick(row.ipsi_cx_px, row.ipsi_cy_px)), tf(pick(row.contra_cx_px, row.contra_cy_px))


def tissue_mask(dapi):
    s = ndi.gaussian_filter(dapi, 3)
    return ndi.binary_fill_holes(morphology.closing(s > filters.threshold_otsu(s[s > 0]) * 0.5, morphology.disk(5)))


def section(img, gi, gc):
    """Step 1: lesion LEFT + the tissue mask of the section that holds the outlines."""
    H, W = img.shape[1:]
    flip = gi.centroid.x > gc.centroid.x
    if flip:
        img = img[:, :, ::-1].copy()
        gi, gc = [affinity.scale(g, -1, 1, origin=(W / 2, 0)) for g in (gi, gc)]
    tm = tissue_mask(img[0]); lab = measure.label(tm)
    keep = {lab[int(g.centroid.y), int(g.centroid.x)] for g in (gi, gc)} - {0}
    return dict(img=img, gi=gi, gc=gc, tissue=ndi.binary_fill_holes(np.isin(lab, list(keep))), flip=bool(flip))


def symmetry_midline(S):
    """Step 2: rotation (+ a symmetry x, used only as the search centre for the fissure)."""
    tm = S["tissue"]; ys, xs = np.nonzero(tm); cy = ys.mean(); best = None
    c0 = (S["gi"].centroid.x + S["gc"].centroid.x) / 2
    small = tm[::4, ::4].astype(float)
    for ang in np.arange(-8, 8.1, 1.0):
        rot = transform.rotate(small, ang, center=(np.mean(xs) / 4, cy / 4), order=0) > 0.5
        for x0 in np.linspace(c0 - 60, c0 + 60, 41):
            c = x0 / 4; w = int(min(c, rot.shape[1] - c)) - 1
            if w < 20:
                continue
            L = rot[:, int(c - w):int(c)]; R = rot[:, int(c):int(c + w)][:, ::-1]
            iou = (L & R).sum() / max((L | R).sum(), 1)
            if best is None or iou > best[0]:
                best = (iou, ang, x0)
    return dict(angle=float(best[1]), x_sym=float(best[2]), cx=float(np.mean(xs)), cy=float(cy))


def standardise(S, angle, cx, cy):
    """Rotate image, tissue and outlines (skimage CCW == shapely -angle in y-down coordinates)."""
    c = (cx, cy)
    img = np.stack([transform.rotate(ch, angle, center=c, order=1, preserve_range=True) for ch in S["img"]])
    tm = transform.rotate(S["tissue"].astype(float), angle, center=c, order=0) > 0.5
    rg = lambda g: affinity.rotate(g, -angle, origin=c)
    return dict(S, img=img, tissue=tm, gi=rg(S["gi"]), gc=rg(S["gc"]))


def fissure_x(tm, x_guess, half_um=1200):
    """Step 3: x of the dorsal interhemispheric fissure = deepest dip of the top tissue boundary near x_guess."""
    H, W = tm.shape
    xs = np.arange(max(int(x_guess - half_um / UM), 0), min(int(x_guess + half_um / UM), W))
    top = np.array([np.nonzero(tm[:, x])[0].min() if tm[:, x].any() else np.nan for x in xs], float)
    top = ndi.gaussian_filter1d(np.nan_to_num(top, nan=np.nanmin(top)), 3)   # empty columns are not a notch
    return float(xs[int(np.argmax(top))])


def prepare(img, gi, gc, stored=None):
    """Full standardisation. ``stored`` (a sections.json record) pins flip / angle / centre / midline so painted
    masks stay registered; without it everything is estimated."""
    S = section(img, gi, gc)
    if stored is not None:
        if bool(stored["flip"]) != S["flip"]:
            raise ValueError("flip disagrees with sections.json -- the QuPath outlines changed?")
        T = standardise(S, stored["angle"], stored["cx"], stored["cy"])
        T["x0"] = stored["midline_x"]
        return T, stored
    M = symmetry_midline(S)
    T = standardise(S, M["angle"], M["cx"], M["cy"])
    T["x0"] = fissure_x(T["tissue"], M["x_sym"])
    rec = dict(flip=S["flip"], angle=M["angle"], cx=M["cx"], cy=M["cy"], x_sym=M["x_sym"], midline_x=T["x0"])
    return T, rec
