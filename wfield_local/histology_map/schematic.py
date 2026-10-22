"""Stroke-extent schematic: painted masks on outline-only Paxinos plates, lesion left, animals overlaid.

Per section: painted mask (painter px) -> standardised section px -> plate mm with that section's fit
(``fit.to_mm``), unioned per animal per plate, clipped to the plate's left hemisphere. Plates = every AP level that
has a section (Priya's figure assignments). Levels BETWEEN an animal's measured levels are filled by signed-distance
interpolation of the two nearest measured footprints (Priya: "strokes shouldn't skip sections"); those are drawn
dashed and lighter and labelled "interp." -- they are estimates, not measurements.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from shapely.geometry import Polygon
from shapely.ops import transform as stf, unary_union
from skimage import measure
from skimage.draw import polygon as _dp

from wfield_local.histology_map import fit as F, painter, plates as PLT

G = 0.02                                   # mm grid for interpolation
GX = np.arange(-6, 1e-4, G); GY = np.arange(0, 9, G)


def _rast(P):
    m = np.zeros((len(GY), len(GX)), bool)
    for g in getattr(P, "geoms", [P]):
        x, y = g.exterior.xy
        rr, cc = _dp((np.array(y) - GY[0]) / G, (np.array(x) - GX[0]) / G, m.shape); m[rr, cc] = True
    return m


def _sdf(m):
    return ndi.distance_transform_edt(~m) - ndi.distance_transform_edt(m)


def _poly(m):
    cs = [c for c in measure.find_contours(np.pad(m, 1).astype(float), 0.5) if len(c) > 4]
    return unary_union([Polygon([(GX[0] + (x - 1) * G, GY[0] + (y - 1) * G) for y, x in c]).buffer(0) for c in cs])


def interpolate(P0, P1, w):
    """Shape between two footprints (w = 0 -> P0, 1 -> P1) by signed-distance blending."""
    return _poly(((1 - w) * _sdf(_rast(P0)) + w * _sdf(_rast(P1))) < 0)


def footprints(sections: dict, masks_dir: Path):
    """{figure: {animal: [polygon mm, ...]}} + per-section rows."""
    byplate, rows = {}, []
    for sid, rec in sorted(sections.items()):
        mp = Path(masks_dir) / f"{sid}.json"
        if not mp.exists() or rec.get("exclude_from_schematic"):
            continue
        doc = json.loads(mp.read_text())
        polys = painter.mask_polygons_px(doc, rec)
        mm = unary_union([stf(lambda x, y, rec=rec: F.to_mm(x, y, rec), p) for p in polys]).buffer(0)
        byplate.setdefault(rec["figure"], {}).setdefault(rec["animal"], []).append(mm)
        rows.append(dict(section=sid, animal=rec["animal"], figure=rec["figure"], bregma=rec["bregma"],
                         area_mm2=round(float(mm.area), 3)))
    return byplate, rows


def footprints_from_polygons(poly_dir):
    """{figure: {animal: [polygon mm]}} + rows from the atlas editor's saved polygons (one JSON per section:
    section_id, animal, figure, poly [[x_mm, y_mm], ...] or polys [[[x_mm, y_mm], ...], ...] in the plate frame, exclude). These are Priya's hand-edited
    shapes (2026-10-22) and supersede the painted-mask route when present."""
    byplate, rows = {}, []
    for f in sorted(Path(poly_dir).glob("*.json")):
        d = json.loads(f.read_text()); d = d.get("data", d)
        if d.get("exclude"):
            continue
        polys = d.get("polys") or [d["poly"]]          # v4 (2026-10-22): several separate stroke regions per section
        if not polys or not polys[0]:
            continue
        g = unary_union([Polygon(q).buffer(0) for q in polys if len(q) >= 3])
        byplate.setdefault(int(d["figure"]), {}).setdefault(d["animal"], []).append(g)
        rows.append(dict(section=d["section_id"], animal=d["animal"], figure=int(d["figure"]), area_mm2=round(float(g.area), 3)))
    return byplate, rows


def fill_gaps(byplate):
    interp = {}
    animals = sorted({a for v in byplate.values() for a in v})
    for a in animals:
        meas = {f: unary_union(v[a]) for f, v in byplate.items() if a in v}
        fs = sorted(meas)
        for f in range(fs[0], fs[-1] + 1):
            if f in meas:
                continue
            lo = max(x for x in fs if x < f); hi = min(x for x in fs if x > f)
            byplate.setdefault(f, {})[a] = [interpolate(meas[lo], meas[hi], (f - lo) / (hi - lo))]
            interp[(f, a)] = True
    return interp


def render(sections, masks_dir, plates_pdf, colors, out_stem, title, ncol=4, interpolate_gaps=True, per_line=4,
           polygons_dir=None, style="filled"):
    """style: "filled" (translucent fill + thin outline) or "outline" (coloured outlines only, no fill; Priya
    2026-10-22: overlapping fills made animals hard to tell apart)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    matplotlib.rcParams["svg.fonttype"] = "none"; matplotlib.rcParams["pdf.fonttype"] = 42
    byplate, rows = footprints_from_polygons(polygons_dir) if polygons_dir else footprints(sections, masks_dir)
    interp = fill_gaps(byplate) if interpolate_gaps else {}
    PL = PLT.load_plates(plates_pdf)
    figs = sorted(byplate)
    nrow = int(np.ceil(len(figs) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, (3.9 + 0.25 * (len(colors) > per_line)) * nrow))
    for ax in np.ravel(axs)[len(figs):]:
        ax.axis("off")
    for ax, f in zip(np.ravel(axs), figs):
        P = PL[f]; R = PLT.raster(P)
        PLT.draw(ax, P, lw=0.45, brain=R["brain"], ppm=R["ppm"])
        LB = PLT.left_brain_polygon(R)
        for a in sorted(byplate[f]):
            U = unary_union(byplate[f][a]).intersection(LB); ii = interp.get((f, a), False)
            for g in getattr(U, "geoms", [U]):
                if g.is_empty or not hasattr(g, "exterior"):
                    continue
                x, y = g.exterior.xy
                if style == "filled":
                    ax.fill(x, y, color=colors[a], alpha=0.22 if ii else 0.35, lw=0)
                    ax.plot(x, y, color=colors[a], lw=1.0, alpha=0.9, ls=(0, (3, 2)) if ii else "-")
                else:
                    ax.plot(x, y, color=colors[a], lw=1.4 if ii else 1.9, alpha=0.95,
                            ls=(0, (3, 2)) if ii else "-", solid_joinstyle="round")
        yy, xx = np.nonzero(R["brain"]); pp = R["ppm"]
        ax.set_xlim(xx.min() / pp - 6.15, xx.max() / pp - 5.85); ax.set_ylim(yy.max() / pp + 0.15, yy.min() / pp - 0.15)
        n = {a: len(v) for a, v in byplate[f].items()}
        lab = lambda a, f=f, n=n: f"{a} interp." if interp.get((f, a)) else f"{a} n={n[a]}"
        labs = [lab(a) for a in sorted(n)]                     # wrap: 8-animal cohorts overflow one line
        ax.set_title(f"Bregma {P['bregma']:+.2f} mm\n" + "\n".join("  ".join(labs[i:i + per_line])
                     for i in range(0, len(labs), per_line)), fontsize=10 if len(labs) <= per_line else 8.5,
                     linespacing=1.4)
    handles = ([Patch(color=colors[a], alpha=0.5, label=a) for a in sorted(colors)] if style == "filled" else
               [Line2D([0], [0], color=colors[a], lw=2.2, label=a) for a in sorted(colors)])
    if interp:
        handles.append(Line2D([0], [0], color="0.3", ls=(0, (3, 2)), label="interpolated between measured sections"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=9)
    fig.suptitle(title, fontsize=12)
    plt.tight_layout(rect=(0, 0.04, 1, 0.96), h_pad=2.5)
    out_stem = Path(out_stem); out_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(out_stem.with_suffix("." + ext), dpi=300)
    plt.close(fig)
    Path(str(out_stem) + "_sections.json").write_text(json.dumps(rows, indent=1))
    return rows, interp
