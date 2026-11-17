"""wfield_local/area_basis.py (Priya 11-04: every Allen-ROI analysis on the split basis = iti_licks' basis, ONE
function): split / plain sets, disjoint rests, compact masks for the O2 bundle, and the encoder's refusal of bundled
inputs without them. Synthetic grids only (ROI version 2 has no v8 rim, so no share access)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from wfield_local import area_basis as ab

REPO = Path(__file__).resolve().parents[1]


def _grid():
    """10 x 12 grid, every pixel brain; left half / right half; MOs top rows, MOp next, SSp-m bottom."""
    H, W = 10, 12
    flat = np.arange(H * W)
    r, c = flat // W, flat % W
    names = {1: "MOs_left", 2: "MOs_right", 3: "MOp_left", 4: "MOp_right", 5: "SSp-m_left", 6: "SSp-m_right"}
    side = np.where(c < W // 2, 0, 1)
    parea = np.where(r < 4, 1 + side, np.where(r < 7, 3 + side, 5 + side))
    return parea, np.ones(len(flat), bool), names, flat, (H, W)


def test_split_and_plain_sets_and_disjoint_rests():
    M = ab.roi_masks(*_grid(), version="2")
    b = ab.basis_masks(M)
    assert set(b["split"]) == {"SSp-m_left", "SSp-m_right", "ALM_left", "ALM_right", "tjM1_left", "tjM1_right",
                               "MOs_rest_left", "MOs_rest_right", "MOp_rest_left", "MOp_rest_right"}
    assert set(b["plain"]) == {"MOs_left", "MOs_right", "MOp_left", "MOp_right", "SSp-m_left", "SSp-m_right"}
    for src, dst in ab.SPLIT_PAIRS:
        for h in ab.HEMIS:
            rest, sub, par = b["split"][f"{src}_rest_{h}"], M[f"{dst}_{h}"], M[f"{src}_{h}"]
            assert not (rest & sub).any() and np.array_equal(rest | sub, par) and sub.any() and rest.any()
    assert ab.tag("8") == "splitr8" and ab.PRIMARY == "split"


def test_session_area_masks_are_compact_indices():
    S = ab.session_area_masks(*_grid(), version="2")
    assert S["version"] == "2" and S["tag"] == "splitr2"
    assert all(v.dtype == np.int32 for v in S["areas"].values())
    assert set(S["areas"]) == set(S["split"]) | set(S["plain"])
    M = ab.roi_masks(*_grid(), version="2")
    assert np.array_equal(S["areas"]["ALM_left"], np.flatnonzero(M["ALM_left"]))
    assert np.array_equal(S["cortex"], np.flatnonzero(M["cortex"]))


def test_encoder_refuses_bundled_inputs_without_masks():
    from wfield_local import combined_encoding as ce
    g = _grid()
    with pytest.raises(RuntimeError, match="no area_masks"):
        ce.area_masks_for({"label": "PS92_0806"}, g[0], g[1], g[2], g[3], g[4])
    am = {"areas": {}, "split": [], "plain": [], "cortex": np.arange(3), "tag": "splitr8", "version": "8"}
    assert ce.area_masks_for({"area_masks": am}, *g[:2], g[2], g[3], g[4]) is am     # used as bundled, not recomputed


def test_iti_licks_delegates_to_the_one_function():
    src = (REPO / "scripts" / "iti_licks.py").read_text(encoding="utf-8")
    assert "return area_basis.roi_masks(" in src and "ndimage.binary_erosion" not in src.split("def roi_masks(")[1][:800]
    assert "ROI_FRAC, ROI_THIN_ANT_RIM = area_basis.ROI_SPEC[ROI_VERSION]" in src
    ce_src = (REPO / "wfield_local" / "combined_encoding.py").read_text(encoding="utf-8")
    assert "area_basis.session_area_masks(" in ce_src and "(parea == a) & pstat for a in areas" not in ce_src
