"""THE AREA BASIS of every Allen-ROI analysis (Priya 11-04: "all Allen-ROI analyses use the split area basis", the
iti_licks basis of 10-27). ONE definition, used by scripts/iti_licks.py (and its importers) and the combined encoding
model (`combined_encoding`): moved here verbatim from iti_licks.roi_masks on 2026-11-04.

  roi_masks   {name: bool over brain pixels}: every Allen area on the eroded stat mask + the v8 rim (thin lateral rim,
              ALM front line), ALM / tjM1 per hemisphere (anterolateral 40 % of MOs / 50 % of MOp), and "cortex".
  split       PRIMARY: every area except cortex and whole MOs / MOp; ALM + tjM1 in; per hemisphere the DISJOINT rests
              MOs_rest = MOs - ALM, MOp_rest = MOp - tjM1 (as pixel masks; iti_licks' rest = (parent - f sub) / (1 - f) is
              the same pixel mean).
  plain       robustness: every area except cortex, ALM / tjM1 out (whole MOs / MOp in).

The masks need `beta_maps.brain_mask()` (share files), so on O2 they are built at BUNDLE time (`session_area_masks`) and
carried in the inputs; never recomputed there.
"""
from __future__ import annotations

import os

import numpy as np

#: ROI version (Priya 10-27; the version history is in scripts/iti_licks.py): WFIELD_ROI_VERSION, default "8" (FINAL)
ROI_VERSION = os.environ.get("WFIELD_ROI_VERSION", "8")
# VERSION HISTORY (Priya 10-27; v1 / v2 fractions, v3 thin anterior rim abandoned; moved from iti_licks):
# v4 (Priya 10-27: "give the orofacial S1 strip"; "I don't want the front edge slivers - only expand laterally"):
# v3 is ABANDONED (front-edge slivers). v4 = the thin rim ONLY at the LATERAL edge: a pixel of the 16-px rim is
# kept when it is >= RIM_ANT_PX inside the brain mask AND the mask ends within 16 px LATERALLY (moving away from
# the midline along its row) -- i.e. it is rim because of the lateral edge, not the front / back edge -- AND it
# lies anterior of RIM_LAT_AP_MIN (the lateral MOp / tjM1 edge and the orofacial S1 strip; posterior rim kept out)
# v5 (Priya 10-27 sketch on lick_rim_overlay: ALM reaches the FRONT edge): as v4 + the anterior thin-rim MOs pixels
# that continue the ALM wedge (same anterolateral rank threshold as ALM) are added to ALM (and MOs); no other
# front-edge pixels (no PL / FRP / medial MOs slivers)
# v6 (Priya 10-27: "keep the lateral extent but I'm worried about edge artefact anteriorly"): as v4, but a lateral-
# edge pixel is NOT added when it is also within 16 px of the FRONT edge (the mask ends within 16 px going
# anterior) -- removes v4's detached anterolateral ALM sliver; nothing added at the front edge (v5 rejected)
# v7 (Priya 10-27: "I don't want to be right up on FRP, medium rim is better"; "v6 doesn't have the lateral
# extent in tjM1 and SSp-m"): lateral thin rim as v4 (full lateral extent) + ALM continued toward the front edge
# with a MEDIUM rim:
# front-edge MOs pixels >= RIM_FRONT_PX inside the brain mask that pass the ALM anterolateral rank are added to
# ALM (and MOs); ALM / tjM1 kept as one connected region each
# v8 (Priya 10-27: "v7 just needs to extend the front horizontal line until it intersects the lateral edge - no
# zig zag"): as v7, but ALM's front boundary is ONE horizontal line -- the most anterior row of v7's medium-rim
# front -- carried laterally to the thin (4-px) lateral rim; ALM = MOs pixels at or behind that row that pass the
# ALM rank, within the 4-px-eroded brain mask, >= FRP_GAP_PX from FRP; one connected region
#: (ROI fraction of the parent area, thin-rim rule) per ROI version
ROI_SPEC = {
    "1": ({"ALM": 0.4, "tjM1": 0.4}, ""),
    "2": ({"ALM": 0.4, "tjM1": 0.5}, ""),
    "3": ({"ALM": 0.4, "tjM1": 0.5}, "ant"),
    "4": ({"ALM": 0.4, "tjM1": 0.5}, "lat"),
    "5": ({"ALM": 0.4, "tjM1": 0.5}, "lat+almfront"),
    "6": ({"ALM": 0.4, "tjM1": 0.5}, "lat-nofront"),
    "7": ({"ALM": 0.4, "tjM1": 0.5}, "lat-almfront8"),
    "8": ({"ALM": 0.4, "tjM1": 0.5}, "lat-almline"),
}
RIM_AP_MIN, RIM_ANT_PX = 1.0, 4
RIM_FRONT_PX = 8
FRP_GAP_PX = 10           # 0.19 mm
RIM_LAT_AP_MIN = -0.5
SPLIT_PAIRS = (("MOs", "ALM"), ("MOp", "tjM1"))           # (parent area, anterolateral sub-ROI)
HEMIS = ("left", "right")
BASES = ("split", "plain")
PRIMARY = "split"


def roi_masks(parea, pstat, names, flat, shape, frac: dict | None = None, version: str | None = None) -> dict:
    """{name: boolean over brain pixels}: every Allen area (stat mask) + ALM / tjM1 per hemisphere (module docstring).
    ROI_THIN_ANT_RIM: "lat" (v4) = the 16-px rim is kept at the LATERAL edge (anterior of RIM_LAT_AP_MIN) except the
    outer RIM_ANT_PX; "ant" (v3, abandoned) = the same anterior of RIM_AP_MIN."""
    ROI_FRAC, ROI_THIN_ANT_RIM = ROI_SPEC[str(version or ROI_VERSION)]
    H, W = (int(x) for x in shape)
    r, c = flat // W, flat % W
    if ROI_THIN_ANT_RIM:
        from scipy import ndimage

        from wfield_local import beta_maps
        from wfield_local.allen_register import BREGMA_OFFSET, RESOLUTION
        bmask = beta_maps.brain_mask()
        thin = ndimage.binary_erosion(bmask, iterations=RIM_ANT_PX).ravel()[flat]
        ap_mm = (BREGMA_OFFSET[1] - r) * RESOLUTION
        if ROI_THIN_ANT_RIM.startswith("lat"):
            step = beta_maps.STAT_ERODE_PX
            out_c = np.where(c >= BREGMA_OFFSET[0], c + step, c - step)          # 16 px further from the midline
            ok_c = (out_c >= 0) & (out_c < W)
            lat_edge = ~ok_c | ~bmask[r, np.clip(out_c, 0, W - 1)]
            add = lat_edge & (ap_mm > RIM_LAT_AP_MIN)
            if ROI_THIN_ANT_RIM == "lat-nofront":
                up_r = r - step                                                   # 16 px anterior (image up)
                front_edge = (up_r < 0) | ~bmask[np.clip(up_r, 0, H - 1), c]
                add &= ~front_edge
        else:                                                                     # "ant" (v3, abandoned)
            add = ap_mm > RIM_AP_MIN
        front = thin & ~pstat & ~add & (ap_mm > RIM_AP_MIN) & (parea != 0)   # anterior thin-rim candidates (v5)
        if ROI_THIN_ANT_RIM == "lat-almline":                                    # v8: pool for the line rule
            frp_ = np.zeros(H * W, bool)
            frp_[flat[np.isin(parea, [k for k, v in names.items() if str(v).startswith("FRP")])]] = True
            gap_ = ndimage.distance_transform_edt(~frp_.reshape(H, W)).ravel()[flat] >= FRP_GAP_PX
            pool4 = thin & gap_ & (parea != 0)
            front8 = front & ndimage.binary_erosion(bmask, iterations=RIM_FRONT_PX).ravel()[flat] & gap_
        if ROI_THIN_ANT_RIM == "lat-almfront8":                                  # v7: medium front rim
            front &= ndimage.binary_erosion(bmask, iterations=RIM_FRONT_PX).ravel()[flat]
            # ... and >= FRP_GAP_PX from the frontal pole (Priya: "I don't want to be right up on FRP")
            frp = np.zeros(H * W, bool)
            frp[flat[np.isin(parea, [k for k, v in names.items() if str(v).startswith("FRP")])]] = True
            front &= ndimage.distance_transform_edt(~frp.reshape(H, W)).ravel()[flat] >= FRP_GAP_PX
        pstat = pstat | (thin & add & (parea != 0))   # labelled cortex only
    m = {names.get(int(a), str(a)): (parea == a) & pstat for a in sorted(np.unique(parea[pstat])) if a != 0}
    left = np.zeros(len(flat), bool)
    right = np.zeros(len(flat), bool)
    for n, mk in m.items():
        left |= mk & n.endswith("_left")
        right |= mk & n.endswith("_right")
    mid = (np.median(c[left]) + np.median(c[right])) / 2.0 if left.any() and right.any() else W / 2.0
    for src, dst in (("MOs", "ALM"), ("MOp", "tjM1")):
        for h in ("left", "right"):
            mk = m.get(f"{src}_{h}")
            if mk is None or not mk.any():
                continue
            ant = -r[mk].astype(float)                      # anterior = top of the image
            lat = np.abs(c[mk] - mid).astype(float)
            u = (ant - ant.mean()) / (ant.std() or 1) + (lat - lat.mean()) / (lat.std() or 1)
            sel = np.zeros(len(flat), bool)
            thr = np.quantile(u, 1 - (frac or ROI_FRAC)[dst])
            sel[np.flatnonzero(mk)[u >= thr]] = True
            if ROI_THIN_ANT_RIM in ("lat+almfront", "lat-almfront8") and dst == "ALM":
                # front-edge MOs pixels of this hemisphere that continue the wedge: same rank, same z-scoring
                cand = front & (parea == next(k for k, v in names.items() if v == f"MOs_{h}"))
                ua = (-r[cand].astype(float) - ant.mean()) / (ant.std() or 1) \
                    + (np.abs(c[cand] - mid).astype(float) - lat.mean()) / (lat.std() or 1)
                idx = np.flatnonzero(cand)[ua >= thr]
                sel[idx] = True
                m[f"MOs_{h}"] = m[f"MOs_{h}"].copy()
                m[f"MOs_{h}"][idx] = True
            if ROI_THIN_ANT_RIM == "lat-almline" and dst == "ALM":
                code = next(k for k, v in names.items() if v == f"MOs_{h}")

                def _u(ix, ant=ant, lat=lat):
                    return (-r[ix].astype(float) - ant.mean()) / (ant.std() or 1) \
                        + (np.abs(c[ix] - mid).astype(float) - lat.mean()) / (lat.std() or 1)
                c8 = front8 & (parea == code)
                c8 = np.flatnonzero(c8)[_u(c8) >= thr]
                line = int(r[c8].min()) if len(c8) else int(r[sel].min())          # v7's most anterior ALM row
                pool = pool4 & (parea == code) & (r >= line)
                idx = np.flatnonzero(pool)[_u(pool) >= thr]
                sel[idx] = True
                sel &= r >= line
                m[f"MOs_{h}"] = m[f"MOs_{h}"].copy()
                m[f"MOs_{h}"][idx] = True
            if ROI_THIN_ANT_RIM in ("lat-nofront", "lat-almfront8", "lat-almline"):   # v6-v8: one connected ROI
                from scipy import ndimage
                g = np.zeros(H * W, bool)
                g[flat[sel]] = True
                lab_, n_ = ndimage.label(g.reshape(H, W))
                if n_ > 1:
                    keep = np.argmax(np.bincount(lab_.ravel())[1:]) + 1
                    sel &= (lab_.ravel()[flat] == keep)
            m[f"{dst}_{h}"] = sel
    m["cortex"] = pstat.copy()
    if ROI_THIN_ANT_RIM in ("lat+almfront", "lat-almfront8", "lat-almline"):
        m["cortex"] |= m.get("ALM_left", False) | m.get("ALM_right", False)
    return m


def basis_masks(M: dict) -> dict:
    """roi_masks output -> {"split": {name: mask}, "plain": {name: mask}} (module docstring); cortex in neither."""
    split = {n: m for n, m in M.items() if n != "cortex" and not n.startswith(("MOs_", "MOp_"))}
    for src, dst in SPLIT_PAIRS:
        for h in HEMIS:
            if f"{src}_{h}" in M and f"{dst}_{h}" in M:
                split[f"{src}_rest_{h}"] = M[f"{src}_{h}"] & ~M[f"{dst}_{h}"]
    plain = {n: m for n, m in M.items() if n != "cortex" and not n.startswith(("ALM_", "tjM1_"))}
    return {"split": split, "plain": plain}


def tag(version: str | None = None) -> str:
    """The basis name outputs carry: "split" + ROI version, e.g. "splitr8"."""
    return f"{PRIMARY}r{version or ROI_VERSION}"


def session_area_masks(parea, pstat, names, flat, shape, version: str | None = None) -> dict:
    """Everything the encoder needs, as compact pixel INDEX arrays (int32 into the brain-pixel vector): {"version",
    "tag", "areas": {name: idx} (the union of both bases), "split": [names], "plain": [names], "cortex": idx (roi_masks
    cortex: stat mask + v8 rim + ALM)}. Built where the share is visible (bundle time)."""
    v = str(version or ROI_VERSION)
    M = roi_masks(parea, pstat, names, flat, shape, version=v)
    b = basis_masks(M)
    areas = {}
    for k in BASES:
        for n, m in b[k].items():
            areas.setdefault(n, np.flatnonzero(m).astype(np.int32))
    return {"version": v, "tag": tag(v), "areas": areas, "split": list(b["split"]), "plain": list(b["plain"]),
            "cortex": np.flatnonzero(M["cortex"]).astype(np.int32)}
