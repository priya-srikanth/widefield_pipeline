"""Model-comparison figures (frozen_fr_compare.epoch_lines / bars; Priya 11-07): one model-coloured line per model from
`cat_points.MODEL_COLORS`; the per-animal values are OPEN markers, shape = animal, colour = epoch, with no per-animal
line across epochs."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
ffc = pytest.importorskip("frozen_fr_compare")
from wfield_local import cat_points as cp  # noqa: E402


def _B():
    rows = []
    for fr, off in (("noFR", 0.0), ("FR-A", 0.01)):
        for ep, v in (("acute", -0.05), ("subacute", -0.02), ("chronic", 0.01)):
            rows.append({"frame": fr, "area": "cortex", "epoch": ep, "point": v + off, "lo": v - 0.03, "hi": v + 0.03,
                         **{a: v + off + 0.01 * k for k, a in enumerate(ffc.AN)}})
    return pd.DataFrame(rows)


def test_model_palette_used_by_frozen_fr_compare():
    assert set(ffc.COL.values()) <= set(cp.MODEL_COLORS)


def test_epoch_lines_animals_are_epoch_coloured_open_markers():
    fig, ax = plt.subplots()
    ser = [("noFR", "noFR", ffc.COL["noFR"]), ("FR-A", "FR-A", ffc.COL["FR-A"])]
    ffc.epoch_lines(ax, _B(), ser, "cortex")
    joined = [ln for ln in cp.no_cross_epoch_lines(ax)]
    # the only multi-point lines are the two model lines, in model colours
    assert len(joined) == 2
    assert {mcolors.to_hex(ln.get_color()) for ln in joined} == {mcolors.to_hex(c) for _q, _l, c in ser}
    animal = [ln for ln in ax.get_lines() if ln.get_marker() in {cp.animal_marker(a) for a in ffc.AN}
              and ln.get_linestyle() == "None" and len(ln.get_xdata()) == 1]
    assert len(animal) == 2 * 3 * len(ffc.AN)
    assert all(ln.get_markerfacecolor() == "none" for ln in animal)
    epc = {mcolors.to_hex(cp.epoch_color(e)) for e in ("acute", "subacute", "chronic")}
    assert {mcolors.to_hex(ln.get_markeredgecolor()) for ln in animal} == epc
    plt.close(fig)


def test_bars_animal_dots_open_epoch_coloured():
    fig, ax = plt.subplots()
    ffc.bars(ax, _B(), ["noFR", "FR-A"], ["cortex"], "y", "t")
    dots = [ln for ln in ax.get_lines() if ln.get_linestyle() == "None"]
    assert dots and all(ln.get_markerfacecolor() == "none" for ln in dots)
    assert not [c for c in ax.collections if np.asarray(c.get_offsets()).size]      # no black scatter dots left
    plt.close(fig)
