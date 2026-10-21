"""Section pairing + ipsi rule for the QuPath striatal-area import, and the plotted quantity."""
import pandas as pd
import pytest

from scripts.import_histology_areas import pair_sections
from wfield_local import histology_area


def _ann(rows):
    return pd.DataFrame([dict(image="PSxx", cx_px=x, cy_px=y, area_um2=a) for x, y, a in rows])


def test_pairs_by_row_and_reading_order():
    # two rows of two sections; second section of row 1 mounted flipped (small outline on the right)
    ann = _ann([(1000, 5000, 2.0), (7000, 5100, 4.0), (20000, 4900, 4.1), (26000, 5000, 1.9),
                (1100, 15000, 2.2), (7100, 15300, 4.2)])
    pairs = pair_sections(ann)
    assert pairs == [(0, 1), (2, 3), (4, 5)]
    ipsi = [a if ann.area_um2[a] < ann.area_um2[b] else b for a, b in pairs]
    assert ipsi == [0, 3, 4]                    # smaller outline, whichever image side it is on


def test_rejects_outlines_that_are_not_one_section():
    with pytest.raises(ValueError):
        pair_sections(_ann([(1000, 5000, 2.0), (1200, 12000, 4.0)]))      # stacked, not side by side
    with pytest.raises(ValueError):
        pair_sections(_ann([(1000, 5000, 2.0), (7000, 5000, 4.0), (9000, 5000, 3.0)]))   # odd count


def test_pct_and_per_animal(tmp_path):
    p = tmp_path / "h.csv"
    pd.DataFrame(dict(animal=["A", "A", "B"], section=[1, 2, 1],
                      ipsi_area_um2=[1.0, 3.0, 2.0], contra_area_um2=[2.0, 4.0, 2.5])).to_csv(p, index=False)
    df = histology_area.load(p)
    assert df.ipsi_contra_pct.tolist() == pytest.approx([50.0, 75.0, 80.0])
    s = histology_area.per_animal(df)
    assert s.loc["A", "mean"] == pytest.approx(62.5) and s.loc["A", "n_sections"] == 2
