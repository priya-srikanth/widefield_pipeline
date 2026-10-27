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
    """A bright pixel at a labelled position must sit at the transformed label after the image transform."""
    for cam, size in (("cam3", (800, 600)), ("cam3", (600, 450)), ("cam2", (600, 450))):
        a = np.zeros(size[::-1], np.uint8)
        a[150, 700 if size[0] == 800 else 500] = 255
        mx, my = S.to_model(700 if size[0] == 800 else 500, 150, cam, size[0])
        out = np.asarray(S.transform_image(Image.fromarray(a), cam))
        assert out.shape == (S.CROP[3], S.CROP[2]) or not (0 <= mx < S.CROP[2])
        if 0 <= mx < S.CROP[2]:
            assert out[int(my), int(mx)] == 255


def test_keep_rows_drops_blank_and_single_part():
    lab = pd.DataFrame({f"{p}_{c}": [np.nan] * 3 for p in S.PARTS for c in ("x", "y")}, index=["a", "b", "c"])
    lab.loc["b", ["nose_x", "nose_y"]] = 1.0                                   # one part only
    lab.loc["c", ["nose_x", "nose_y", "jaw_x", "jaw_y"]] = 1.0
    kept, partial = S.keep_rows(lab)
    assert list(kept.index) == ["c"] and list(partial.index) == ["b"]
