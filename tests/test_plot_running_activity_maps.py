"""Tests for the quiet/running SVD activity-map producer (helper + guard)."""
import numpy as np
import pytest

from wfield_local import plot_running_activity_maps as pram


def test_mask_from_edges():
    m = pram._mask_from_edges([2, 7], [4, 9], 10)
    assert m.tolist() == [False, False, True, True, False, False, False, True, True, False]
    assert pram._mask_from_edges([], [], 5).sum() == 0


def test_display_limit():
    lim = pram._display_limit([np.array([-3.0, 1.0, 2.0])], 100.0)
    assert lim == pytest.approx(3.0)
    assert pram._display_limit([np.array([])], 99.0) == 1e-6


def test_main_errors_without_events(tmp_path):
    argv = ["--label", "x", "--events", str(tmp_path / "missing.npz"),
            "--wfield-results", str(tmp_path), "--allen-dir", str(tmp_path),
            "--daq-h5", str(tmp_path / "d.h5"), "--output", str(tmp_path / "out")]
    with pytest.raises(SystemExit):
        pram.main(argv)
