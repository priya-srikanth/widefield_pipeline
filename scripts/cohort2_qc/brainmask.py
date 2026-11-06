"""Brain mask for native-space cohort2 analysis (2026-10-30).

The QC scripts used `binary_erosion(F470 > 0.45 * F470.max(), 4)`. In the RiboL1 mice the 470 mean image has bright
posterior/lateral hotspots that set the max, so the dimmer, mottled ANTERIOR cortex (ALM / tjM1 territory) fell
below the cut and was fragmented away (Priya 10-30: "why is so much anterior cortex ... missing in 121 and 122").
Here the cut is relative to the camera dark level: F > dark + FRAC * (p99.5 - dark), the largest connected
component, holes filled, a light opening for the ragged edge. Vessels inside the brain are kept (filled).
Pass ``res`` (the wfield_local_results dir) to also drop camera-saturated pixels (``clip_mask.npy``, written by the
pre-SVD pass since 2026-10-30; wfield_local.swap_repair, CLIPPED PIXELS). Absent file = nothing dropped.
"""
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_fill_holes, binary_opening, label

FRAC = 0.2


def brain_mask(fa, func: int = 1, frac: float = FRAC, res=None) -> np.ndarray:
    im = np.asarray(fa[func], np.float64)
    dark = min(im[:5, :5].mean(), im[:5, -5:].mean(), im[-5:, :5].mean(), im[-5:, -5:].mean())
    m = im > dark + frac * (np.percentile(im, 99.5) - dark)
    m = binary_opening(m, iterations=3)
    lab, n = label(m)
    if n > 1:
        m = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    m = binary_fill_holes(m)
    if res is not None and (Path(res) / "clip_mask.npy").exists():
        m &= ~np.load(Path(res) / "clip_mask.npy").astype(bool)       # fill BEFORE the cut: clipped stays out
    return m
