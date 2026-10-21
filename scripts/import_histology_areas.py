"""Import ipsi/contra striatal (caudoputamen) areas from the PS92-95 QuPath project.

Source = the QuPath project Priya annotates (DAPI/GFAP/NeuN, VS200 10x scans), one unnamed,
unclassified outline per striatum, two per section:

    /Users/Priya2/Documents/data/Immuno/PS92-95_widefield_stroke_cohort

Writes `configs/histology_striatum_area.csv`, one row per SECTION:
    animal, section, image, entry_id, slide_section, ipsi_area_um2, contra_area_um2,
    ipsi_cx_px, ipsi_cy_px, contra_cx_px, contra_cy_px

How the two outlines of a section are identified, since QuPath stores no names:
  * PAIRING: mutual nearest neighbours by centroid. Every pair is asserted to sit on one slide row
    (|dy| < ROW_TOL_PX) at hemisphere spacing (MIN_DX_PX < |dx| < MAX_DX_PX), so a mis-pairing
    fails the import instead of producing a plausible ratio.
  * IPSI = THE SMALLER OUTLINE (Priya, 2026-10-21: "the one with smaller area will always be ipsi
    -- some sections are mounted in flipped orientation"). Image side cannot be used: PS93's sections
    and PS92's first section are mounted flipped. Checked against the GFAP scar on every slide
    thumbnail on 2026-10-21, and it agreed on all 24 sections. Consequence: ipsi/contra <= 100% by
    construction, so this measures lesion-driven loss, not left/right asymmetry in an unlesioned
    animal.
  * SECTION ORDER is reading order on the slide (row, then x); it is NOT a verified AP order.

The animal is parsed from the image name (`PSxx_...`). Re-run whenever the QuPath annotations
change (QuPath must have SAVED the image, which is what writes data.qpdata):

    python -m scripts.import_histology_areas
    python -m scripts.import_histology_areas --project /path/to/qupath_project --out configs/histology_striatum_area.csv
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from wfield_local import qupath_io

REPO = Path(__file__).resolve().parent.parent
DEFAULT_PROJECT = Path("/Users/Priya2/Documents/data/Immuno/PS92-95_widefield_stroke_cohort")
ANIMALS = ("PS92", "PS93", "PS94", "PS95")
ROW_TOL_PX = 1500          # the two hemispheres of one section sit on one row (measured |dy| <= ~620 px)
MIN_DX_PX, MAX_DX_PX = 3000, 9000   # hemisphere spacing at 0.65 um/px (measured 5.5k-7.6k px)


def pair_sections(ann: pd.DataFrame) -> list[tuple[int, int]]:
    """Pair the outlines of ONE image into sections; returns (image-left idx, image-right idx), reading order."""
    c = ann[["cx_px", "cy_px"]].to_numpy()
    if len(c) % 2:
        raise ValueError(f"{ann.image.iloc[0]}: odd number of outlines ({len(c)}); every section needs two")
    d = np.hypot(*(c[:, None, :] - c[None, :, :]).transpose(2, 0, 1))
    np.fill_diagonal(d, np.inf)
    nn = d.argmin(1)
    pairs = set()
    for i, j in enumerate(nn):
        if nn[j] != i:
            raise ValueError(f"{ann.image.iloc[0]}: outline {i} has no mutual nearest neighbour")
        dx, dy = abs(c[i, 0] - c[j, 0]), abs(c[i, 1] - c[j, 1])
        if dy > ROW_TOL_PX or not (MIN_DX_PX < dx < MAX_DX_PX):
            raise ValueError(f"{ann.image.iloc[0]}: outlines {i},{j} do not look like one section (dx={dx:.0f}, dy={dy:.0f})")
        pairs.add(tuple(sorted((i, j), key=lambda k: c[k, 0])))
    row_h = np.median([abs(c[a, 1] - c[b, 1]) for a, b in pairs]) + 2 * ROW_TOL_PX
    return sorted(pairs, key=lambda p: (round(c[p[0], 1] / row_h), c[p[0], 0]))


def build(project: Path) -> pd.DataFrame:
    ann = qupath_io.project_annotations(project)
    rows = []
    for (eid, image), g in ann.groupby(["entry_id", "image"], sort=True):
        m = re.match(r"(PS\d+)", image)
        if not m or m.group(1) not in ANIMALS:
            continue
        g = g.reset_index(drop=True)
        for n, (a, b) in enumerate(pair_sections(g), 1):
            ip, co = (a, b) if g.area_um2[a] < g.area_um2[b] else (b, a)
            rows.append(dict(animal=m.group(1), image=image.split(" - ")[0], entry_id=eid, slide_section=n,
                             ipsi_area_um2=g.area_um2[ip], contra_area_um2=g.area_um2[co],
                             ipsi_cx_px=g.cx_px[ip], ipsi_cy_px=g.cy_px[ip],
                             contra_cx_px=g.cx_px[co], contra_cy_px=g.cy_px[co]))
    df = pd.DataFrame(rows).sort_values(["animal", "entry_id", "slide_section"], kind="stable")
    df.insert(1, "section", df.groupby("animal").cumcount() + 1)
    return df.reset_index(drop=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", type=Path, default=DEFAULT_PROJECT, help="QuPath project folder (holds project.qpproj)")
    ap.add_argument("--out", type=Path, default=REPO / "configs" / "histology_striatum_area.csv")
    a = ap.parse_args(argv)
    df = build(a.project)
    df.to_csv(a.out, index=False, float_format="%.2f")
    pct = 100 * df.ipsi_area_um2 / df.contra_area_um2
    print(f"[histology] {len(df)} sections from {df.image.nunique()} images -> {a.out}")
    print(pct.groupby(df.animal).agg(["size", "mean", "min", "max"]).round(1).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
