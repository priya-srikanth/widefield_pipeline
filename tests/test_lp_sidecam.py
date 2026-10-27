import numpy as np
import pandas as pd
from PIL import Image

from wfield_local import lp_sidecam as S


def test_transform_round_trip_both_sizes():
    for cam in ("cam2", "cam3"):
        for w in (600, 800):                 # June side videos are 800 x 600, August onward 600 x 450
            x, y = np.array([41.0, 300.5, 599.0]), np.array([91.0, 200.0, 320.0])
            mx, my = S.to_model(x, y, cam, w)
            rx, ry = S.to_raw(mx, my, cam, w)
            assert np.allclose(rx, x) and np.allclose(ry, y)


def test_label_follows_the_pixel():
    """A bright spot at a raw position lands at the transformed position (flip, crop and the June resize)."""
    for cam in ("cam2", "cam3"):
        for size, (xr, yr) in (((800, 600), (650, 250)), ((600, 450), (480, 200)), ((800, 600), (90, 200))):
            a = np.zeros(size[::-1], np.float32)
            a[yr - 3:yr + 4, xr - 3:xr + 4] = 255
            mx, my = S.to_model(float(xr), float(yr), cam, size[0])
            out = np.asarray(S.transform_image(Image.fromarray(a.astype(np.uint8)), cam), np.float32)
            assert out.shape == (S.OUT, S.OUT)
            if not (5 <= mx < S.OUT - 5 and 5 <= my < S.OUT - 5):
                continue                                   # this test point is outside this camera's crop
            ys, xs = np.nonzero(out > 0)
            w = out[ys, xs]
            assert abs((xs * w).sum() / w.sum() - mx) < 0.6 and abs((ys * w).sum() / w.sum() - my) < 0.6


def test_unknown_frame_size_refuses():
    import pytest
    with pytest.raises(SystemExit):
        S.to_model(1.0, 1.0, "cam2", 640)


def test_keep_rows_drops_blank_and_single_part():
    lab = pd.DataFrame({f"{p}_{c}": [np.nan] * 3 for p in S.PARTS for c in ("x", "y")}, index=["a", "b", "c"])
    lab.loc["b", ["nose_x", "nose_y"]] = 1.0                                   # one part only
    lab.loc["c", ["nose_x", "nose_y", "jaw_x", "jaw_y"]] = 1.0
    kept, partial = S.keep_rows(lab)
    assert list(kept.index) == ["c"] and list(partial.index) == ["b"]
