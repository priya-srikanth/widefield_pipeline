"""A session recorded with a DIFFERENT CAMERA ROI is placed at its offset at native scale, not stretched
(2026-10-29; cohort1 June 1-3 and cohort2 10-29 PS118 / PS120 / PS121 have such frames)."""
import cv2
import numpy as np
import pytest

from wfield_local.cross_day_align import _register_different_roi


def _sensor(seed=0, shape=(620, 720)):
    """Vessel-like texture: smoothed noise plus thin random lines, on a smooth illumination gradient."""
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur(rng.normal(size=shape).astype(np.float32), (0, 0), 3)
    for _ in range(120):
        p = rng.integers(0, shape[::-1], size=2); q = p + rng.integers(-80, 80, size=2)
        cv2.line(img, tuple(int(v) for v in p), tuple(int(v) for v in q), float(rng.uniform(2, 4)), 2)
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    return img + 5 + 0.004 * xx + 0.003 * yy


@pytest.mark.parametrize("mov_box", [(80, 120, 462, 464),    # smaller ROI, offset (cohort1 06-03-like)
                                     (10, 20, 540, 640)])    # LARGER ROI that contains the reference (06-01-like)
def test_different_roi_is_placed_not_stretched(mov_box):
    S = _sensor()
    r0, c0, h, w = 50, 100, 460, 480
    ref = S[r0:r0 + h, c0:c0 + w]
    mr, mc, mh, mw = mov_box
    mov = S[mr:mr + mh, mc:mc + mw]
    mask = cv2.erode((ref > np.percentile(ref, 30)).astype(np.uint8), np.ones((9, 9), np.uint8))
    w2x3, ncc, method, cov = _register_different_roi(mov, ref, mask)
    # expected: a session pixel (r, c) sits at reference (r + mr - r0, c + mc - c0) -- pure translation
    assert np.allclose(w2x3[:, :2], np.eye(2), atol=2e-3), w2x3
    assert w2x3[1, 2] == pytest.approx(mr - r0, abs=0.5) and w2x3[0, 2] == pytest.approx(mc - c0, abs=0.5)
    assert ncc > 0.95, (ncc, method)
    placed = cv2.warpAffine(mov, w2x3, (w, h), flags=cv2.INTER_LINEAR)
    inner = cv2.erode(cov.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    assert np.abs(placed - ref)[inner].mean() < 0.05 * np.abs(ref).mean()
    # coverage = exactly the overlap of the two ROIs on the reference grid
    exp = np.zeros((h, w), bool)
    exp[max(0, mr - r0):min(h, mr - r0 + mh), max(0, mc - c0):min(w, mc - c0 + mw)] = True
    assert (cov != exp).mean() < 0.01
