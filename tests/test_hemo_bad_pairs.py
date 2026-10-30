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


def test_advance_half_sample_is_exact_and_amplitude_preserving():
    n = 4000
    i = np.arange(n)
    f = 0.31                                              # cycles / sample (~10 Hz at 31.23 Hz)
    x = np.sin(2 * np.pi * f * i)[None, :]
    y = hv._advance_half_sample(x)
    want = np.sin(2 * np.pi * f * (i + 0.5))
    mid = slice(50, n - 50)                               # the first / last ~1.5 s ring a little
    assert np.allclose(y[0, mid], want[mid], atol=1e-3)
    assert abs(y[0, mid].std() / want[mid].std() - 1) < 1e-3


def test_lag_variants_need_refit(tmp_path):
    with pytest.raises(ValueError):
        hv.compute({"mc": str(tmp_path), "label": "X", "h5": "x"}, "meegkit_hpfit_lag", refit_t=False)
