"""`pose_windows` must return the SAME tables a CSV read does, and rebuild when a CSV is newer.

The sidecar replaced a dozen scripts' direct `pd.read_csv` of windows_index / windows_LP, so any
drift between the two reads would move every cam1 window result without a visible change.
"""
import os

import numpy as np
import pandas as pd
import pytest

from wfield_local import pose_windows as pw


def _write_csvs(d, n=12, lp=True):
    rng = np.random.default_rng(0)
    ix = pd.DataFrame({
        "trial": np.arange(n),
        "position": np.array(["close_L", "far_R", "close_center"] * (n // 3)),
        "stop_source": np.array(["daq", "lick"] * (n // 2)),
        "t_ms": rng.normal(size=n) * 1000,
        "daq_s": 1e4 + rng.random(n),                       # DAQ seconds need float64
    })
    ix.to_csv(d / "windows_index.csv", index=False)
    if lp:
        cols = pd.MultiIndex.from_tuples(
            [("tongue", "x"), ("tongue", "y"), ("tongue", "likelihood"), ("jaw", "x")],
            names=["bodyparts", "coords"])
        P = pd.DataFrame(rng.random((n, 4)).astype(np.float32) * 300, columns=cols)
        # DLC / LP layout: a scorer row above (bodyparts, coords), so the reader uses header=[1, 2]
        top = pd.MultiIndex.from_tuples([("scorer",) + c for c in cols], names=["scorer", "bodyparts", "coords"])
        P.set_axis(top, axis=1).to_csv(d / "windows_LP.csv")
    return ix


def test_index_matches_csv_read(tmp_path):
    _write_csvs(tmp_path)
    ref = pd.read_csv(tmp_path / "windows_index.csv")
    got = pw.index(tmp_path)
    assert list(got.columns) == list(ref.columns)
    for c in ("trial", "position", "stop_source"):
        assert got[c].tolist() == ref[c].tolist()
    np.testing.assert_array_equal(got["daq_s"].to_numpy(), ref["daq_s"].to_numpy())   # exact, float64 kept
    assert (tmp_path / pw.SIDECAR).exists()


def test_index_usecols_keeps_file_order(tmp_path):
    _write_csvs(tmp_path)
    got = pw.index(tmp_path, usecols=["t_ms", "trial"])
    assert list(got.columns) == ["trial", "t_ms"]


def test_lp_matches_csv_read_to_float32(tmp_path):
    _write_csvs(tmp_path)
    ref = pd.read_csv(tmp_path / "windows_LP.csv", header=[1, 2], index_col=0)
    got = pw.lp(tmp_path)
    assert list(got.columns) == list(ref.columns)
    np.testing.assert_allclose(got.to_numpy(), ref.to_numpy(), atol=1e-3)            # decimal print of float32


def test_lp_missing_raises(tmp_path):
    _write_csvs(tmp_path, lp=False)
    assert not pw.has_lp(tmp_path)
    with pytest.raises(FileNotFoundError):
        pw.lp(tmp_path)


def test_newer_csv_triggers_rebuild(tmp_path):
    _write_csvs(tmp_path)
    pw.index(tmp_path)
    side = tmp_path / pw.SIDECAR
    old = side.stat().st_mtime
    os.utime(side, (old - 100, old - 100))                  # sidecar now older than the CSVs
    assert pw._stale(tmp_path)
    assert pw._build_one(tmp_path).endswith("built")
    assert not pw._stale(tmp_path)
    assert pw._build_one(tmp_path).endswith("current")
