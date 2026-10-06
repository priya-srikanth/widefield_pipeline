"""video_motion: camera frames land in the right imaging bins, motion energy is the mean |frame difference| per
bin, and the SVD recovers a single moving region as the first component with its time course."""
from __future__ import annotations

import numpy as np

from wfield_local import video_motion as VM


def test_frame_bins_use_midpoint_edges_and_mark_outside_frames():
    ft = np.arange(0, 1.0, 0.032)
    b = VM.frame_bins(np.array([-0.5, 0.0, 0.015, 0.017, 0.5, 2.0]), ft)
    assert list(b) == [-1, 0, 0, 1, int(round(0.5 / 0.032)), -1]


def test_downsample_area_averages_grey_bgr():
    im = np.zeros((16, 16, 3), np.uint8)
    im[:8, :8] = 80
    d = VM.downsample(im, 8)
    assert d.shape == (2, 2) and d[0, 0] == 80 and d[1, 1] == 0


def test_motion_energy_and_svd_find_the_moving_patch(tmp_path):
    rng = np.random.default_rng(0)
    n, h, w = 4000, 32, 32
    amp = np.abs(np.sin(np.arange(n) / 37.0)) * 60          # the patch flickers with this envelope
    frames = []
    for i in range(n):
        im = np.full((h, w, 3), 50, np.uint8)
        im[:8, :8] = 50 + (amp[i] * (i % 2)).astype(np.uint8)   # alternating -> motion energy ~ amp
        im += rng.integers(0, 2, im.shape, dtype=np.uint8)
        frames.append(im)
    bins = np.arange(n) // 8                                 # 8 camera frames per imaging bin
    mm, filled = VM.binned_motion_energy(iter(frames), bins, n // 8, tmp_path / "me.npy", ds=8, progress_every=0)
    assert filled[1:].all()
    res = VM.motion_svd(mm, filled, k=3, n_fit=400)
    comp = np.abs(res["components"][0].reshape(4, 4))
    assert np.argmax(comp) == 0 and comp[0, 0] > 3 * comp[2:, 2:].max()
    env = amp.reshape(-1, 8).mean(1)
    assert abs(np.corrcoef(res["timecourses"][1:, 0], env[1:])[0, 1]) > 0.95
