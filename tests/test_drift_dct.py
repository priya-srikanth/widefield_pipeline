"""wfield_local.drift_dct: the block-protected cosine drift fit keeps block means and removes slow drift."""
import numpy as np
from scipy.fft import dct

from wfield_local import drift_dct as dd


def _synthetic(seed=0, fs=10.0, minutes=120, block_s=300.0):
    rng = np.random.default_rng(seed)
    n = int(minutes * 60 * fs)
    t = np.arange(n) / fs
    segs, a, code = [], 0, 0
    while a < n:
        b = min(n, a + int(block_s * fs))
        segs.append(dict(start=a, stop=b, code=code % 6, n_trials=6, first_cue=0, last_cue=0))
        a, code = b, code + 1
    levels = rng.normal(0, 1, 6)
    blk = np.zeros(n)
    for sg in segs:
        blk[sg["start"]:sg["stop"]] = levels[sg["code"]]
    drift = 3 * np.cos(2 * np.pi * t / (minutes * 60)) + 2 * np.cos(2 * np.pi * t / 2400.0)
    x = blk + drift + rng.normal(0, 0.3, n)
    mask = rng.random(n) < 0.3
    return x[None, :], blk, drift, segs, mask, fs


def test_basis_is_orthonormal_dct():
    B = dd.dct_basis(1000, 10.0, 20.0)
    assert np.allclose(B.T @ B, np.eye(B.shape[1]), atol=1e-10)
    x = np.random.default_rng(1).normal(size=1000)
    assert np.allclose(B.T @ x, dct(x, norm="ortho")[:B.shape[1]], atol=1e-10)


def test_protected_keeps_block_means_and_removes_drift():
    x, blk, drift, segs, mask, fs = _synthetic()
    y = dd.remove(x, mask, fs, 20 * 60.0, segs, protect=True)[0]
    bm = lambda s: np.array([s[g["start"]:g["stop"]].mean() for g in segs])  # noqa: E731
    clean = blk + (drift - drift.mean())
    # block means after removal track the true block levels much better than the drifting input does
    r_after = np.corrcoef(bm(y), bm(blk))[0, 1]
    r_before = np.corrcoef(bm(x[0]), bm(blk))[0, 1]
    assert r_after > 0.95 and r_after > r_before
    assert np.var(y - blk - (y - blk).mean()) < 0.2 * np.var(clean - blk)


def test_only_slow_dct_coefficients_change():
    x, _blk, _drift, segs, mask, fs = _synthetic(seed=2)
    y = dd.remove(x, mask, fs, 15 * 60.0, segs, protect=True)[0]
    k = dd.kmax_for(x.shape[1], fs, 15 * 60.0)
    d = dct(x[0], norm="ortho") - dct(y, norm="ortho")
    assert np.allclose(d[k + 1:], 0, atol=1e-8) and abs(d[0]) < 1e-8


def test_position_protection_keeps_position_means_and_removes_drift():
    x, blk, drift, segs, mask, fs = _synthetic(seed=3, block_s=90.0)
    y = dd.remove(x, mask, fs, 15 * 60.0, segs, protect="position")[0]
    codes = np.array([g["code"] for g in segs])
    pm = lambda s: np.array([np.mean([s[g["start"]:g["stop"]].mean() for g in segs if g["code"] == q])  # noqa: E731
                             for q in range(6)])
    t = pm(blk)
    assert np.allclose(pm(y) - pm(y).mean(), t - t.mean(), atol=0.15)
    assert np.var(y - blk - (y - blk).mean()) < 0.2 * np.var(drift)
    assert codes.size > 6
