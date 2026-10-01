"""`lp_labels`: an all-blank row must not survive into a Lightning Pose single-view label file."""
import numpy as np
import pandas as pd

from wfield_local.lp_labels import audit, clean, read


def _write(path, rows):
    cols = pd.MultiIndex.from_product([["Priya"], ["nose", "jaw"], ["x", "y"]], names=["scorer", "bodyparts", "coords"])
    idx = [f"labeled-data/cam4_x/img{i:07d}.png" for i in range(len(rows))]
    pd.DataFrame(rows, index=idx, columns=cols, dtype=float).to_csv(path)


def test_clean_drops_blank_rows_and_keeps_backup(tmp_path):
    p = tmp_path / "CollectedData.csv"
    nan = np.nan
    _write(p, [[1, 2, 3, 4], [nan, nan, nan, nan], [1, 2, nan, nan]])
    assert audit(read(p))["all_blank"] == 1
    assert clean(p) == 1
    a = audit(read(p))
    assert a["rows"] == 2 and a["all_blank"] == 0 and a["filled"] == {"nose": 2, "jaw": 1}
    assert (tmp_path / "CollectedData.csv.bak").exists()


def test_clean_is_a_no_op_on_a_clean_file(tmp_path):
    p = tmp_path / "CollectedData.csv"
    _write(p, [[1, 2, 3, 4]])
    assert clean(p) == 0 and not (tmp_path / "CollectedData.csv.bak").exists()
