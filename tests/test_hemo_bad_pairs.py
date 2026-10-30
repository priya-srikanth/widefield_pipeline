import numpy as np
import pytest

from wfield_local import hemo_variants as hv


def test_interp_pairs_replaces_only_bad_columns():
    x = np.vstack([np.arange(10.0), 2 * np.arange(10.0)])
    x[:, 4] = 999.0                                   # a swapped frame
    bad = np.zeros(10, bool)
    bad[4] = True
    y = hv._interp_pairs(x, bad)
    assert np.allclose(y[:, 4], [4.0, 8.0])
    assert np.array_equal(np.delete(y, 4, 1), np.delete(x, 4, 1))
    assert x[0, 4] == 999.0                           # input not modified


def test_interp_pairs_no_bad_is_identity():
    x = np.random.default_rng(0).normal(size=(3, 20))
    assert hv._interp_pairs(x, np.zeros(20, bool)) is x


def test_write_refuses_bad_pairs_without_tag(tmp_path):
    with pytest.raises(ValueError):
        hv.write({"mc": str(tmp_path), "label": "X"}, "meegkit_hpfit", bad_pairs=np.zeros(5, bool))
