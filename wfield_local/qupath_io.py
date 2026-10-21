"""Read annotation outlines from a QuPath project without QuPath or Java.

QuPath (checked on 0.6.0) saves each image's objects to ``data/<entryID>/data.qpdata`` as a Java
``ObjectOutputStream``. Annotation ROIs appear in one of two encodings:

  ``GeometryROI$WKBSerializationProxy`` -> a ``byte[]`` holding WKB (Polygon / MultiPolygon);
                                            brush / wand / merged annotations
  ``PolygonROI$SerializationProxy``      -> ``float[] x`` then ``float[] y`` of equal length,
                                            back to back; plain polygon-tool annotations

Every Java array is ``TC_ARRAY (0x75)`` + class descriptor (the full ``0x72`` descriptor the first
time a type is written, a ``0x71`` back-reference after) + ``int32`` length + data. We walk every
array in the stream and keep those that decode to an area geometry. Coordinates are full-resolution
image PIXELS; ``server.json`` gives the micron calibration.

Verified 2026-10-21 on the PS92-95 project: annotation counts equal QuPath's own
``summary.json`` counts on every image, and areas equal the ``GeometryStats.area`` /
``ClosedShapeStatistics.areaCached`` values QuPath stores beside each ROI (e.g. 3815749.5 px^2
stored vs 3815750 px^2 recomputed).

Annotation names and classes are NOT read: this project's annotations are unnamed and
unclassified, so callers identify them by position (see ``scripts/import_histology_areas.py``).

    from wfield_local import qupath_io
    df = qupath_io.project_annotations("/Users/Priya2/Documents/data/Immuno/PS92-95_widefield_stroke_cohort")
    # image, entry_id, ann, kind, area_px, area_um2, cx_px, cy_px, (geometry)
"""
from __future__ import annotations

import json
import re
import struct
from pathlib import Path

import numpy as np
import pandas as pd
from shapely import wkb
from shapely.geometry import Polygon

_ARRAY = re.compile(rb"\x75(?:\x72\x00\x02\[[BF].{8}\x02\x00\x00\x78\x70|\x71\x00\x7e..)", re.S)


def _arrays(raw: bytes):
    """(start, data_offset, length) of every array in the stream, in stream order."""
    out = []
    for m in _ARRAY.finditer(raw):
        p = m.end()
        n = struct.unpack(">i", raw[p:p + 4])[0]
        if n > 1 and p + 4 + n <= len(raw):
            out.append((m.start(), p + 4, n))
    return out


def read_qpdata(path) -> list:
    """Area geometries (shapely, pixel coordinates) of the annotations in one ``data.qpdata``."""
    raw = Path(path).read_bytes()
    arrs = _arrays(raw)
    geoms, skip = [], set()
    for k, (_, d, n) in enumerate(arrs):
        if k in skip:
            continue
        try:                                            # byte[] WKB
            g = wkb.loads(raw[d:d + n])
            if g.geom_type in ("Polygon", "MultiPolygon") and g.area > 0:
                geoms.append(("wkb", g))
                continue
        except Exception:                               # noqa: BLE001 -- not WKB, try float[] pair
            pass
        if k + 1 < len(arrs):                           # float[] x immediately followed by float[] y
            s2, d2, n2 = arrs[k + 1]
            if n2 == n and n >= 3 and s2 == d + 4 * n:
                x = np.frombuffer(raw[d:d + 4 * n], ">f4")
                y = np.frombuffer(raw[d2:d2 + 4 * n], ">f4")
                g = Polygon(np.c_[x, y]).buffer(0)
                if g.area > 0:
                    geoms.append(("polygon", g))
                    skip.add(k + 1)
    return geoms


def _summary_count(entry_dir: Path) -> int | None:
    try:
        h = json.loads((entry_dir / "summary.json").read_text())["hierarchy"]
        return int(h.get("objectTypeCounts", {}).get("Annotations", 0))
    except (OSError, KeyError, ValueError):
        return None


def project_annotations(project_dir, keep_geometry: bool = False) -> pd.DataFrame:
    """One row per annotation across every image in a QuPath project that has a ``data.qpdata``.

    Raises if the number decoded for an image differs from QuPath's own ``summary.json`` count,
    so a ROI encoding this reader does not know about fails loudly instead of being dropped.
    """
    project_dir = Path(project_dir)
    proj = json.loads((project_dir / "project.qpproj").read_text())
    rows = []
    for im in proj["images"]:
        eid = str(im.get("entryID"))
        edir = project_dir / "data" / eid
        if not (edir / "data.qpdata").exists():
            continue
        cal = json.loads((edir / "server.json").read_text())["metadata"]["pixelCalibration"]
        um2_per_px = cal["pixelWidth"]["value"] * cal["pixelHeight"]["value"]
        geoms = read_qpdata(edir / "data.qpdata")
        expect = _summary_count(edir)
        if expect is not None and expect != len(geoms):
            raise ValueError(f"{im.get('imageName')} (entry {eid}): decoded {len(geoms)} annotations, "
                             f"QuPath summary.json says {expect}")
        for i, (kind, g) in enumerate(geoms):
            row = dict(image=im.get("imageName"), entry_id=int(eid), ann=i, kind=kind,
                       area_px=g.area, area_um2=g.area * um2_per_px,
                       cx_px=g.centroid.x, cy_px=g.centroid.y)
            if keep_geometry:
                row["geometry"] = g
            rows.append(row)
    return pd.DataFrame(rows)
