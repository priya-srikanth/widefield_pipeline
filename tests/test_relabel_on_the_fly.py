"""RelabeledDat (on-the-fly TTL relabel) must feed motion correction bytes IDENTICAL to the
materialized cleanpairs .dat, so the motion-corrected .bin is unchanged.

Skipped where wfield isn't installed (run_wfield_motion imports it) — i.e. runs on the imaging box.
"""
import numpy as np
import pytest

pytest.importorskip("wfield")  # run_wfield_motion imports wfield.io / motion_correct_fixed
from wfield_local.run_wfield_motion import RelabeledDat  # noqa: E402


def _materialized(raw_flat, pairs):
    """Exactly write_trimmed_dat's gather: out[i,0]=raw[pairs[i,0]], out[i,1]=raw[pairs[i,1]]."""
    out = np.empty((len(pairs), 2, raw_flat.shape[1], raw_flat.shape[2]), raw_flat.dtype)
    out[:, 0] = raw_flat[pairs[:, 0]]
    out[:, 1] = raw_flat[pairs[:, 1]]
    return out


def test_relabeled_dat_matches_materialized(tmp_path):
    rng = np.random.RandomState(0)
    ntot, h, w = 60, 6, 5
    raw = rng.randint(0, 65535, size=(ntot, h, w)).astype(np.uint16)
    p = tmp_path / "raw.dat"
    raw.tofile(p)
    raw_flat = np.memmap(p, mode="r", dtype=np.uint16, shape=(ntot, h, w))
    pairs = np.array([[0, 1], [2, 3], [10, 11], [20, 25], [58, 59], [7, 8], [30, 31]], dtype=np.int64)

    rd = RelabeledDat(raw_flat, pairs)
    mat = _materialized(raw_flat, pairs)

    assert rd.shape == (len(pairs), 2, h, w)
    assert rd.dtype == np.uint16
    assert len(rd) == len(pairs)
    assert np.array_equal(np.array(rd[:]), mat)                      # full gather
    # the exact access patterns motion_correct uses: dat[a:b] slices
    for a, b in [(0, 3), (2, 7), (1, 5), (0, len(pairs))]:
        assert np.array_equal(np.array(rd[a:b]), mat[a:b]), (a, b)
