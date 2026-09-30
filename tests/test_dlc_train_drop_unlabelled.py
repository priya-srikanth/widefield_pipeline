"""`dlc_train.drop_unlabelled`: an all-blank row must never reach DLC, which reads blanks as "absent"."""
import numpy as np
import pandas as pd

from wfield_local.dlc_train import drop_unlabelled


def _labels(rows):
    cols = pd.MultiIndex.from_product([["Priya"], ["nose", "jaw"], ["x", "y"]],
                                      names=["scorer", "bodyparts", "coords"])
    idx = pd.MultiIndex.from_tuples([("labeled-data", "cam4_x", f"img{i:07d}.png") for i in range(len(rows))])
    return pd.DataFrame(rows, index=idx, columns=cols, dtype=float)


def test_all_blank_rows_dropped_partial_rows_kept():
    nan = np.nan
    df = _labels([[1, 2, 3, 4],            # fully labelled
                  [nan, nan, nan, nan],    # context frame never labelled -> must go
                  [1, 2, nan, nan]])       # jaw occluded -> stays, blank = absent
    out, n = drop_unlabelled(df)
    assert n == 1
    assert list(out.index.get_level_values(2)) == ["img0000000.png", "img0000002.png"]
    assert np.isnan(out.iloc[1]["Priya", "jaw", "x"])


def test_nothing_to_drop():
    out, n = drop_unlabelled(_labels([[1, 2, 3, 4]]))
    assert n == 0 and len(out) == 1
