import numpy as np
import pytest

from wfield_local import swap_repair as sr


def _movie(ratio, T=500, H=256, W=256, seed=0):
    """Synthetic (T, 2, H, W) uint16 movie: slot 0 = 415, slot 1 = 470, with distinct spatial patterns."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W] / H
    brain = ((yy - 0.5) ** 2 + (xx - 0.5) ** 2) < 0.18
    vessels = 1 - 0.3 * (np.abs(np.sin(9 * xx + 4 * yy)) < 0.15)
    F415 = 2000 + 14000 * brain * (0.8 + 0.4 * yy) * vessels
    F470 = 2000 + 14000 * ratio * brain * (1.1 - 0.4 * xx) * (1 - 0.6 * (1 - vessels))
    slow = 0.02 * np.sin(np.arange(T) / 40.0)
    mov = np.empty((T, 2, H, W), np.float32)
    for c, F in ((0, F415), (1, F470)):
        mov[:, c] = F[None] * (1 + slow[:, None, None])
    mov += rng.normal(0, 1, mov.shape).astype(np.float32) * np.sqrt(mov)      # shot noise
    truth = mov.copy()
    mov[100, 0] = truth[100, 1]                                 # single full swap in the 415 slot
    mov[200, 0], mov[200, 1] = truth[200, 1], truth[200, 0]     # both slots exchanged
    mov[300, 1, :96] = truth[300, 0, :96]                       # rolling-shutter band in the 470 slot
    mov[400, 0] = truth[400, 0] + truth[400, 1] - 2000          # dual exposure in the 415 slot
    clip = lambda a: np.clip(np.round(a), 0, 65535).astype(np.uint16)
    return clip(mov), clip(truth), clip(mov.mean(0))


@pytest.mark.parametrize("ratio", [2.0, 1.0])            # RiboL1-like (470 ~2x 415) and GCaMP6s-like (equal)
def test_detects_each_defect_and_nothing_else(ratio):
    mov, _truth, fa = _movie(ratio)
    plan = sr.detect(mov, fa, func=1, verbose=False)
    bad = plan["bad"]
    assert bad[100, 0] and not bad[100, 1]
    assert plan["exchange"][200]
    assert bad[300, 1] and not plan["full"][300, 1]            # partial, not full
    assert bad[400, 0]
    expected = {(100, 0), (200, 0), (200, 1), (300, 1), (400, 0)}
    found = {(int(t), int(c)) for t, c in zip(*np.nonzero(bad))}
    assert found == expected, found - expected


def test_repair_restores_frames():
    mov, truth, fa = _movie(2.0)
    plan = sr.detect(mov, fa, func=1, verbose=False)
    rep = sr.RepairedDat(mov, plan)
    assert not hasattr(rep, "filename") and rep.shape == mov.shape and len(rep) == len(mov)
    blk = rep[95:405]
    assert np.array_equal(blk[200 - 95], truth[200])                      # exchanged exactly
    for t, c in ((100, 0), (300, 1), (400, 0)):                           # interpolated: close to the truth
        err = np.abs(blk[t - 95, c].astype(float) - truth[t, c]).mean() / truth[t, c].mean()
        assert err < 0.02
    assert np.array_equal(rep[50:60], mov[50:60])                         # untouched elsewhere
    fa2 = sr.corrected_average(mov, rep, fa)
    assert np.abs(fa2 - truth.mean(0)).mean() < np.abs(fa.astype(float) - truth.mean(0)).mean()


def test_summary_counts():
    mov, _t, fa = _movie(2.0)
    s = sr.summary(sr.detect(mov, fa, func=1, verbose=False))
    assert s["pairs_exchanged"] == 1 and s["frames_interpolated"] == 3 and s["bad_415_slot"] == 3


def test_approximate_svd_runs_on_repaired_view():
    from wfield.decomposition import approximate_svd
    mov, truth, fa = _movie(2.0, T=500, H=128, W=128)
    plan = sr.detect(mov, fa, func=1, verbose=False)
    rep = sr.RepairedDat(mov, plan)
    fa2 = sr.corrected_average(mov, rep, fa).astype(np.float32)
    U, SVT = approximate_svd(rep, fa2, k=5, nframes_per_bin=5, nbinned_frames=50, nframes_per_chunk=50)
    assert U.shape == (128, 128, 5) and SVT.shape == (5, 2 * 500)
    m = U.reshape(-1, 5).mean(0) @ SVT                    # interleaved 415, 470 columns
    s415 = m[0::2]
    jump = abs(s415[100] - np.median(s415[95:106]))
    assert jump < 0.02                                   # the planted 415 swap (+~100%) is gone
