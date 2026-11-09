"""Area set of interhemi_crossarea (11-09 fix of the l_xa_restnl crash at `area_order`).

The rebuild_20261103 session files carry 62 decoder-basis areas (the painted glue mask is not readable there), and the
run took ONE session's names (`res[0]`) to index every session's matrices. Pinned here: (1) the analysis area set is the
glue-mask set, 24 contralesional + 21 ipsilesional, no left VISp / VISl / VISal and no lateral VIS / AUD / TEa;
(2) the lick sets define a session's areas and the rest set is cut to them BY NAME (extra rest areas dropped, a missing
one is an error); (3) matrices are placed in the common order by name, so two sessions whose column orders differ give
the same cells.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import interhemi_crossarea as ica  # noqa: E402

LAT = ["VISli", "VISpl", "VISpor", "AUDp", "AUDd", "AUDpo", "TEa"]


def _cols(bases, hemis=("left", "right")):
    """Session columns in the Allen layout (whole MOs / MOp + ALM / tjM1), as the extraction writes them."""
    return [f"{b}_{h}" for h in hemis for b in bases] + ["cortex"]


def _session_bases():
    full = [b for b in ica.ORDER_BASE if b not in ("ALM", "tjM1", "MOs_rest", "MOp_rest")] + ["MOs", "MOp", "ALM", "tjM1"]
    return full + LAT


def test_area_set_is_glue_mask_set():
    left = sorted(n for n in ica.AREA_SET if n.endswith("_left"))
    right = sorted(n for n in ica.AREA_SET if n.endswith("_right"))
    assert (len(right), len(left)) == (24, 21)
    for b in ("VISp", "VISl", "VISal"):
        assert f"{b}_left" not in ica.AREA_SET and f"{b}_right" in ica.AREA_SET
    assert not any(n.rpartition("_")[0] in LAT for n in ica.AREA_SET)


def test_lick_sets_define_areas_and_rest_is_cut_by_name():
    cols = _cols(_session_bases())
    rest_cols = list(reversed(cols))                      # same areas, other column order
    sets = {"cued": (None, None, None, cols), "iti_all": (None, None, None, cols), "rest": (None, None, None, rest_cols)}
    names, dropped = ica.session_areas(sets)
    assert set(names) == set(ica.AREA_SET)
    assert {"VISp_left", "VISl_left", "VISal_left", "TEa_right", "AUDp_left"} <= set(dropped)
    all_rest = ica.il.area_names(rest_cols)
    sel = ica.area_select(all_rest, names, "rest")
    assert [all_rest[i] for i in sel] == names           # rest columns follow the lick-set names exactly
    with pytest.raises(RuntimeError, match="lacks lick-set areas"):
        ica.area_select([n for n in all_rest if n != "SSp-m_left"], names, "rest")


def test_lick_sets_must_agree():
    a = _cols(_session_bases())
    b = [c for c in a if c != "RSPd_left"]
    with pytest.raises(RuntimeError, match="disagree"):
        ica.session_areas({"cued": (None, None, None, a), "iti_all": (None, None, None, b)})


def test_place_matches_by_name_not_by_first_session():
    names1 = ["SSp-m_right", "ALM_left", "VISpm_left"]
    names2 = ["VISpm_left", "SSp-m_right"]                # other order, one area absent
    order = ica.common_order([names1, names2])
    assert order == ["ALM_left", "VISpm_left", "SSp-m_right"]   # ipsi block first, ORDER_BASE order
    rng = np.random.default_rng(0)
    M1 = rng.normal(size=(3, 3)).astype(np.float32)
    P1 = ica.place(M1, names1, order)
    # the same pair (VISpm_left, SSp-m_right) in session 2's own order lands in the same cell
    M2 = np.array([[1, 0.3], [0.3, 1]], np.float32)
    P2 = ica.place(M2, names2, order)
    i, j = order.index("VISpm_left"), order.index("SSp-m_right")
    assert P1[i, j] == M1[names1.index("VISpm_left"), names1.index("SSp-m_right")]
    assert P2[i, j] == np.float32(0.3) and np.isnan(P2[order.index("ALM_left")]).all()
