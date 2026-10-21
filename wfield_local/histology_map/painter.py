"""The Stroke Painter: a claude.ai artifact page where Priya paints / corrects each section's stroke mask.

The page (``painter_template.html``) shows each standardised section (lesion left) as three switchable colour
layers (NeuN red, GFAP green, DAPI blue) + outlines (her surviving-striatum outline, the mirrored normal striatum),
Claude's suggestion (yellow, read-only) and HER mask (magenta; paint / erase / undo / start-from-Claude), with a
progress bar of her mask against the absent striatal area. Saving writes ``masks/<animal>_s<n>`` to the artifact's
database ({mask: PNG data-URL, note, w, h, saved_at}); those documents, copied into
``configs/histology_map/<cohort>/masks/``, are the published masks.

Painter pixel (px, py) -> standardised section px: ``x = crop_x0 + px / scale``, ``y = crop_y0 + py / scale``
(crop + scale stored per section in sections.json).
"""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from wfield_local.histology_map import UM_PER_PX_L3 as UM
from wfield_local.histology_map.sections import stretch

SCALE = 0.42
TEMPLATE = Path(__file__).with_name("painter_template.html")


def _enc(im, fmt="JPEG", **kw):
    b = io.BytesIO(); im.save(b, fmt, **kw)
    return f"data:image/{'jpeg' if fmt == 'JPEG' else 'png'};base64," + base64.b64encode(b.getvalue()).decode()


def decode_mask(data_url) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(base64.b64decode(data_url.split(",")[1]))).convert("L")) > 127


def section_payload(sid, animal, section, T, suggestion, mirror, flags=(), excluded=False):
    """One section's page data + its crop record (x0, y0, scale) for sections.json."""
    ys, xs = np.nonzero(T["tissue"])
    x0, y0, x1, y1 = max(xs.min() - 15, 0), max(ys.min() - 15, 0), xs.max() + 15, ys.max() + 15
    W, H = int((x1 - x0) * SCALE), int((y1 - y0) * SCALE)
    d = dict(id=sid, animal=animal, section=int(section), w=W, h=H, excluded=bool(excluded), flags=list(flags),
             target_mm2=round(float(suggestion["missing"].sum() * (UM / 1000) ** 2), 3))
    for name, c, col in (("neun", 2, 0), ("gfap", 1, 1), ("dapi", 0, 2)):
        g = (stretch(T["img"][c]) * 255).astype(np.uint8)[y0:y1, x0:x1]
        rgb = np.zeros(g.shape + (3,), np.uint8); rgb[..., col] = g if name != "dapi" else (g * 0.8).astype(np.uint8)
        d[name] = _enc(Image.fromarray(rgb).resize((W, H), Image.LANCZOS), quality=80)
    ol = Image.new("RGBA", (W, H), (0, 0, 0, 0)); dr = ImageDraw.Draw(ol)
    for g, col, dash in ((T["gi"], (255, 255, 255, 255), False), (mirror, (80, 220, 255, 255), True)):
        for p in getattr(g, "geoms", [g]):
            pts = [((x - x0) * SCALE, (y - y0) * SCALE) for x, y in p.exterior.coords]
            if dash:
                for i in range(0, len(pts) - 1, 4): dr.line(pts[i:i + 3], fill=col, width=1)
            else:
                dr.line(pts, fill=col, width=1)
    d["outlines"] = _enc(ol, "PNG", optimize=True)
    m = Image.fromarray((suggestion["stroke"][y0:y1, x0:x1] * 255).astype(np.uint8)).resize((W, H), Image.NEAREST)
    d["mask"] = _enc(m, "PNG", optimize=True)
    return d, dict(crop_x0=int(x0), crop_y0=int(y0), scale=SCALE)


def build_page(payloads, cohort, out_html):
    t = TEMPLATE.read_text().replace("__COHORT__", cohort)
    Path(out_html).write_text(t.replace("__DATA__", json.dumps(payloads).replace("</", "<\\/")))
    return out_html


def mask_polygons_px(mask_doc, rec):
    """A saved painter mask -> shapely polygons in the standardised section frame."""
    from shapely.geometry import Polygon
    from skimage import measure
    m = decode_mask(mask_doc["mask"])
    out = []
    for c in measure.find_contours(np.pad(m, 1).astype(float), 0.5):
        if len(c) <= 4:
            continue
        p = Polygon(c[:, ::-1] - 1).buffer(0)
        if p.area > 20:
            out.append(p)
    tf = lambda xy: (rec["crop_x0"] + xy[0] / rec["scale"], rec["crop_y0"] + xy[1] / rec["scale"])
    from shapely.ops import transform as stf
    return [stf(lambda x, y: tf((np.asarray(x), np.asarray(y))), p) for p in out]
