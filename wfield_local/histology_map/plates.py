"""Paxinos & Franklin plates -> outline-only line art + mm-calibrated brain / CPu masks.

Source: ``The Mouse Brain in Stereotaxic Coordinates`` (2nd ed.) PDF in the lab's Atlases folder. Coronal plates are
Illustrator vector art, one figure per page (Fig 13-48 = PDF pages 63-98 cover the striatum, Bregma +2.10 to
-2.06); photomicrographs follow at PDF page = figure + 182 (odd figures Nissl, even AChE). Copy the plate range out
of the 146 MB atlas once with ``qpdf "<atlas>" --pages . 63-98 -- plates.pdf`` and point ``plates_pdf`` at it.

What is kept, and why (Priya, 2026-10-21: "outlines only, no letter identifiers"):
  * the section drawing only: black strokes inside the outer brain contour's box (frame, ticks, grid, sagittal
    inset and navigation buttons fall outside it);
  * NOT label leader lines (single straight 2-point segments, except the vertical midline);
  * NOT the small unlabelled circle every plate carries in the right caudoputamen (~1.5 mm lateral, ~0.12 mm wide):
    a reference mark of the drawing, not anatomy;
  * NOT line art outside the brain outline (Fig 30 has a stray mark above the brain);
  * filled shapes (the ventricles) are drawn light grey so they do not compete with stroke colours.
Grid lines (grey, 1 mm) give the calibration: ~71.2 pt / mm; x = 0 is the central vertical grid line.
"""
from __future__ import annotations

import re

import numpy as np
from scipy import ndimage as ndi

PX_PER_MM = 60


def _is_black(c):
    return c is not None and max(c) < 0.05


def _polys(x, n=12):
    """Flatten one PDF drawing's path items to polylines (list of Nx2 arrays, PDF points)."""
    polys, cur = [], []
    for it in x["items"]:
        if it[0] == "l":
            a, b = it[1], it[2]; seg = [(a.x, a.y), (b.x, b.y)]
        elif it[0] == "c":
            P = np.array([[q.x, q.y] for q in it[1:5]]); t = np.linspace(0, 1, n)[:, None]
            seg = list((1 - t) ** 3 * P[0] + 3 * (1 - t) ** 2 * t * P[1] + 3 * (1 - t) * t ** 2 * P[2] + t ** 3 * P[3])
        elif it[0] == "re":
            r = it[1]; seg = [(r.x0, r.y0), (r.x1, r.y0), (r.x1, r.y1), (r.x0, r.y1), (r.x0, r.y0)]
        elif it[0] == "qu":
            q = it[1]; seg = [(q.ul.x, q.ul.y), (q.ur.x, q.ur.y), (q.lr.x, q.lr.y), (q.ll.x, q.ll.y), (q.ul.x, q.ul.y)]
        else:
            continue
        seg = [tuple(map(float, s)) for s in seg]
        if cur and np.hypot(cur[-1][0] - seg[0][0], cur[-1][1] - seg[0][1]) < 1e-3:
            cur += seg[1:]
        else:
            if len(cur) > 1: polys.append(np.array(cur))
            cur = list(seg)
    if len(cur) > 1: polys.append(np.array(cur))
    return polys


def plate(page) -> dict:
    """One PDF page -> dict(fig, bregma, pt_per_mm, midline_x, lines, fills, page)."""
    txt = page.get_text()
    bregma = float(re.search(r"Bregma (-?[0-9.]+) mm", txt).group(1))
    fig = int(re.search(r"FIGURE (\d+)", txt).group(1))
    dr = page.get_drawings()
    gx = sorted({round(x["rect"].x0, 2) for x in dr if x.get("color") and 0.4 < x["color"][0] < 0.6 and x["rect"].width < 1})
    mid = gx[len(gx) // 2] if len(gx) % 2 else float(np.mean(gx[len(gx) // 2 - 1:len(gx) // 2 + 1]))
    lines = [dict(polys=_polys(x), width=x["width"], dashed=bool(x.get("dashes") and "[]" not in x["dashes"]),
                  rect=x["rect"]) for x in dr if _is_black(x.get("color")) and x["type"] in ("s", "fs")]
    fills = [_polys(x) for x in dr if x["type"] in ("f", "fs") and _is_black(x.get("fill"))]
    return dict(fig=fig, bregma=bregma, pt_per_mm=float(np.median(np.diff(gx))), midline_x=mid,
                lines=lines, fills=fills, page=page)


def load_plates(pdf_path) -> dict:
    """{figure number: plate dict} for every plate page in the PDF."""
    import pymupdf as fz
    doc = fz.open(str(pdf_path))
    out = {}
    for i in range(len(doc)):
        try:
            P = plate(doc[i])
        except (AttributeError, ValueError):
            continue
        out[P["fig"]] = P
    out["_doc"] = doc        # keep the document alive: pages reference it
    return out


def brain_lines(P):
    """Section line art only (see module docstring). Returns (lines, fills, outer_rect)."""
    import pymupdf as fz
    mx, ppmm = P["midline_x"], P["pt_per_mm"]
    centred = lambda L: abs((L["rect"].x0 + L["rect"].x1) / 2 - mx) < 25
    cand = [L for L in P["lines"] if not L["dashed"] and L["width"] > 0.8 and centred(L)]
    if not cand:   # a few plates (Fig 15) draw the outline as two separate halves meeting at the midline: use their union
        halves = [L for L in P["lines"] if not L["dashed"] and L["width"] > 0.8 and L["rect"].x0 < mx + 8 and L["rect"].x1 > mx - 8
                  and L["rect"].width > 100]
        if len(halves) >= 2:
            cand = [dict(halves[0], rect=fz.Rect(halves[0]["rect"]))]
            for L in halves[1:]: cand[0]["rect"] |= L["rect"]
    if not cand:   # a few plates draw the outline at a lighter weight
        cand = [L for L in P["lines"] if not L["dashed"] and L["width"] > 0.5 and centred(L) and L["rect"].width < 11 * ppmm]
    outer = max(cand, key=lambda L: L["rect"].width * L["rect"].height)["rect"]
    bb = fz.Rect(outer.x0 - 2, outer.y0 - 2, outer.x1 + 2, outer.y1 + 2)
    keep = [L for L in P["lines"] if bb.contains(L["rect"])]

    def is_leader(L):
        if len(L["polys"]) != 1 or len(L["polys"][0]) != 2:
            return False
        (x0, _), (x1, _) = L["polys"][0]
        return not (abs(x0 - x1) < 0.5 and abs(x0 - mx) < 3)

    def is_refmark(L):
        r = L["rect"]; x = ((r.x0 + r.x1) / 2 - mx) / ppmm
        return (not L["dashed"]) and r.width < 10 and r.height < 10 and 1.2 < x < 1.9 and len(L["polys"]) == 1

    keep = [L for L in keep if not is_leader(L) and not is_refmark(L)]
    fills = [f for f in P["fills"] if f and bb.contains(fz.Rect(*f[0].min(0), *f[0].max(0)))]
    return keep, fills, outer


def raster(P, ppm=PX_PER_MM) -> dict:
    """Masks on an mm grid: column = (x_mm + 6) * ppm, row = y_mm * ppm (y from the page top).
    brain = largest filled region; cpu = flood fill from the 'CPu' label, mirrored to both sides."""
    from PIL import Image, ImageDraw
    keep, fills, _ = brain_lines(P)
    W, H = int(12 * ppm), int(9 * ppm)
    tf = lambda a: [((x - P["midline_x"]) / P["pt_per_mm"] * ppm + 6 * ppm, y / P["pt_per_mm"] * ppm) for x, y in a]
    im = Image.new("L", (W, H), 0); d = ImageDraw.Draw(im)
    for L in keep:
        for a in L["polys"]: d.line(tf(a), fill=255, width=2)
    for fl in fills:
        for a in fl:
            if len(a) > 2: d.polygon(tf(a), fill=255)
    lines = np.array(im) > 0
    lab, _ = ndi.label(~lines); bg = lab[0, 0]
    brain = ndi.binary_fill_holes((lab != bg) | lines)
    bl, n = ndi.label(brain)
    if n > 1:      # keep the section itself, not stray marks
        brain = bl == np.argmax(np.bincount(bl.ravel())[1:]) + 1
    cpu = np.zeros_like(lines)
    for w in P["page"].get_text("words"):
        if w[4] == "CPu":
            cx = ((w[0] + w[2]) / 2 - P["midline_x"]) / P["pt_per_mm"] * ppm + 6 * ppm
            cy = (w[1] + w[3]) / 2 / P["pt_per_mm"] * ppm
            for dy in range(12):             # the label sits on the region: first non-line pixel below its centre
                yy, xx = int(cy) + dy, int(cx)
                if 0 <= yy < H and not lines[yy, xx] and lab[yy, xx] != bg:
                    cpu |= lab == lab[yy, xx]; break
    cpu |= cpu[:, ::-1]
    return dict(brain=brain, cpu=cpu, lines=lines, ppm=ppm)


def draw(ax, P, lw=0.5, color="k", dashed_alpha=0.6, fill_color="0.88", brain=None, ppm=PX_PER_MM):
    """Draw a plate in mm (x from the midline, y from the page top) onto ``ax``; ``brain`` (raster mask) drops
    line art outside the outline."""
    keep, fills, _ = brain_lines(P)
    to_mm = lambda a: np.c_[(a[:, 0] - P["midline_x"]) / P["pt_per_mm"], a[:, 1] / P["pt_per_mm"]]
    if brain is not None:
        bd = ndi.binary_dilation(brain, iterations=4)

        def inside(a):
            b = to_mm(a)
            c = np.clip(((b[:, 0] + 6) * ppm).astype(int), 0, bd.shape[1] - 1)
            r = np.clip((b[:, 1] * ppm).astype(int), 0, bd.shape[0] - 1)
            return bd[r, c].mean() > 0.5
        keep = [dict(L, polys=[a for a in L["polys"] if inside(a)]) for L in keep]
        fills = [[a for a in fl if inside(a)] for fl in fills]
    for L in keep:
        for a in L["polys"]:
            b = to_mm(a)
            ax.plot(b[:, 0], b[:, 1], color=color, lw=lw * (0.6 if L["dashed"] else 1),
                    ls=(0, (2, 1)) if L["dashed"] else "-", alpha=dashed_alpha if L["dashed"] else 1,
                    solid_capstyle="round")
    for fl in fills:
        for a in fl:
            b = to_mm(a); ax.fill(b[:, 0], b[:, 1], color=fill_color, lw=0, zorder=0)
    ax.set_aspect("equal")
    if not ax.yaxis_inverted():
        ax.invert_yaxis()
    ax.axis("off")


def left_brain_polygon(R):
    """The plate's LEFT hemisphere (x <= 0) as a shapely polygon in mm, for clipping stroke footprints."""
    from shapely.geometry import Polygon, box
    from skimage import measure
    ppm = R["ppm"]
    c = max(measure.find_contours(np.pad(R["brain"], 1).astype(float), 0.5), key=len)
    return Polygon([(x / ppm - 6, y / ppm) for y, x in c]).buffer(0).intersection(box(-7, -1, 0, 10))
