"""Stroke-schematic geometry: anatomical midline, section -> plate transform, gap interpolation, cohort data."""
import json
from pathlib import Path

import numpy as np
from shapely.geometry import Point, box

from wfield_local.histology_map import fit, schematic, sections

REPO = Path(__file__).resolve().parents[1]


def test_fissure_found_not_symmetry_axis():
    # two "hemispheres" of different width (left atrophied); the fissure notch is at x=100
    tm = np.zeros((200, 260), bool)
    tm[40:180, 30:100] = True; tm[40:180, 101:230] = True     # gap column = fissure
    tm[40:60, 100] = False; tm[60:180, 100] = True           # notch only at the dorsal surface
    assert abs(sections.fissure_x(tm, x_guess=130, half_um=1000) - 100) <= 1


def test_to_mm_left_is_negative_and_scales_from_midline():
    rec = dict(midline_x=500, y_top=100, kx=0.9, ky=1.1, ty=1.0)
    x, y = fit.to_mm([500, 400], [100, 200], rec)
    assert x[0] == 0 and x[1] < 0
    assert np.isclose(x[1], -100 * 0.9 * 5.2e-3) and np.isclose(y[1], 1.0 + 100 * 1.1 * 5.2e-3)


def test_interpolation_is_between_its_ends():
    a = Point(-2, 4).buffer(0.4); b = Point(-2, 4).buffer(0.8)
    m = schematic.interpolate(a, b, 0.5)
    assert a.area < m.area < b.area
    assert m.intersection(box(-7, 0, 0, 9)).area > 0.95 * m.area


def test_cohort_data_consistent():
    d = REPO / "configs" / "histology_map" / "PS92-95"
    secs = json.loads((d / "sections.json").read_text())
    masks = {p.stem for p in (d / "masks").glob("*.json")}
    assert set(secs) == masks
    for sid, r in secs.items():
        assert r["figure"] in r["priya_figures"]                 # the level is one Priya gave
        assert 0.79 <= r["kx"] <= 1.16 and 0.79 <= r["ky"] <= 1.16
        assert r["animal"] == sid.split("_s")[0]
