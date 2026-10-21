"""Per-FRAME motion energy in a few face ROIs (whisker pads), at the camera rate -- for a per-trial "reacted" call
(Priya 10-21: "they may also react with nose or whisker movement"). `video_motion` bins whole-frame motion energy to
the IMAGING frames for the encoding model; this keeps the 250 Hz camera clock so the jaw rule
(`jaw_kinematics.jaw_onsets`) can be run on it, trial by trial, like the jaw and nose.

Interim measure: FaceRhythm (point tracking on a cross-animal master face mesh, Priya's colleague's stroke_orofacial
work) is the intended long-term whisker readout; this is deliberately simple so it can be swapped out.

ROIs (`rois`): two rectangles, image-LEFT and image-RIGHT of the session-median LP nose, offsets per camera (checked on
PS92-95 frames 10-21: cam4 = the whisker pads beside the snout, clear of the rod; cam1 = the whisker shafts against the
dark background). Each ROI is split into vertical STRIPS of ``STRIP_PX`` so the spout rod (cam1: it crosses the
whisker field at far spouts) can be dropped per trial afterwards by its LP x. Value per frame = mean |frame -
previous frame| over a strip of the ``ds``-downsampled grey image.

ARTEFACT (10-21): the cam4 videos carry an ENCODER KEYFRAME on every frame whose index % 12 == 0 (20.8 Hz): that frame's
motion energy is ~33 % higher than the other 11 phases (flat), a static background corner peaks at exactly 12.01
frames, and the pads correlate with it (r 0.48). Fix = interpolate the motion energy over frames with index % 12 == 0
(`scripts/attempt_review.py`): 20.8 Hz power down 4-8x, 62.5 Hz harmonic removed (a 5-frame median or a background-corner
detector only shrank it). The 10 ms-binned pilot averages (`scripts/whisker_motion.py traces`) include it as a small
offset. Any frame-difference / optical-flow measure on these videos (FaceRhythm included) can carry a 20.8 Hz component.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

#: (dx_outer, dx_inner, dy_top, dy_bottom) from the nose, full-resolution px
OFFSETS = {"cam4": (250, 110, -20, 110), "cam1": (290, 140, -60, 170)}
STRIP_PX = 16
SIDES = ("img_left", "img_right")


def rois(cam: str, nose_xy: tuple[float, float], shape: tuple[int, int]) -> list[tuple[int, int, int, int]]:
    """[(x0, y0, x1, y1) image-left, image-right], clipped to the frame (rows, cols = ``shape``)."""
    dxo, dxi, dy0, dy1 = OFFSETS[cam]
    nx, ny = nose_xy
    H, W = shape
    y0, y1 = int(np.clip(ny + dy0, 0, H)), int(np.clip(ny + dy1, 0, H))
    return [(int(np.clip(nx - dxo, 0, W)), y0, int(np.clip(nx - dxi, 0, W)), y1),
            (int(np.clip(nx + dxi, 0, W)), y0, int(np.clip(nx + dxo, 0, W)), y1)]


def strip_edges(roi: tuple[int, int, int, int]) -> np.ndarray:
    """Full-resolution x edges of the strips of one ROI (last strip may be narrower)."""
    x0, _, x1, _ = roi
    return np.unique(np.r_[np.arange(x0, x1, STRIP_PX), x1])


def _grey(im: np.ndarray, ds: int) -> np.ndarray:
    g = im[..., 0] if im.ndim == 3 else im
    H, W = (g.shape[0] // ds) * ds, (g.shape[1] // ds) * ds
    return g[:H, :W].reshape(H // ds, ds, W // ds, ds).mean((1, 3), dtype=np.float32)


def _chunk(path: str, out_path: str, f0: int, f1: int, ds: int, roi_list: list, n_strips: int) -> int:
    """Frames [f0, f1) -> rows f0..f1 of the (n_frames, n_roi, n_strips) float16 memmap. Decoding starts at f0 - 1
    so frame f0 has its predecessor (frame 0 stays NaN). Seeking is frame-accurate on these videos
    (`video_motion._chunk_worker`, checked 2026-10-06)."""
    import cv2
    mm = np.load(out_path, mmap_mode="r+")
    # CROP to the ROIs' bounding band first: the grey conversion + downsample of the whole frame cost 60 % of the time
    Y0, Y1 = min(r[1] for r in roi_list), max(r[3] for r in roi_list)
    X0, X1 = min(r[0] for r in roi_list), max(r[2] for r in roi_list)
    cuts = []
    for r in roi_list:
        e = (strip_edges(r) - X0) // ds
        cuts.append(((r[1] - Y0) // ds, max((r[3] - Y0) // ds, (r[1] - Y0) // ds + 1), e))
    cap = cv2.VideoCapture(path)
    start = max(f0 - 1, 0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    prev, n = None, 0
    buf = np.full((min(f1 - f0, 4096), len(roi_list), n_strips), np.nan, np.float32)
    b0 = f0
    for i in range(start, f1):
        ok, im = cap.read()
        if not ok:
            break
        g = _grey(im[Y0:Y1, X0:X1], ds)
        if prev is not None and i >= f0:
            d = np.abs(g - prev)
            k = i - b0
            for j, (ya, yb, e) in enumerate(cuts):
                col = d[ya:yb].mean(0)
                w = np.maximum(np.diff(e), 1)
                buf[k, j, :len(w)] = np.add.reduceat(col[:e[-1]], e[:-1]) / w
            n += 1
            if k == len(buf) - 1:
                mm[b0:b0 + len(buf)] = buf
                b0 += len(buf)
                buf[:] = np.nan
        prev = g
    k = min(f1, b0 + len(buf)) - b0
    if k > 0:
        mm[b0:b0 + k] = buf[:k]
    cap.release()
    mm.flush()
    return n


def roi_motion(path: Path, roi_list: list, out_path: Path, ds: int = 2, n_workers: int = 4) -> np.memmap:
    """(n_frames, n_roi, n_strips) float16 per-frame motion energy for one video, decoded in ``n_workers`` chunks."""
    import cv2
    from concurrent.futures import ProcessPoolExecutor
    cap = cv2.VideoCapture(str(path))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    n_strips = max(len(strip_edges(r)) - 1 for r in roi_list)
    mm = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float16, shape=(n_frames, len(roi_list), n_strips))
    mm[:] = np.nan
    del mm
    edges = np.linspace(0, n_frames, n_workers + 1).astype(int)
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = [ex.submit(_chunk, str(path), str(out_path), int(a), int(b), ds, roi_list, n_strips)
                for a, b in zip(edges[:-1], edges[1:])]
        for fu in futs:
            fu.result()
    return np.load(out_path, mmap_mode="r")


def trial_trace(me: np.ndarray, roi: tuple[int, int, int, int], rod_x: float | None, pad_px: float = 30.0) -> np.ndarray:
    """(n,) mean over one ROI's strips, dropping strips within ``pad_px`` of the rod's x (``me``: (n, n_strips))."""
    e = strip_edges(roi)
    keep = np.ones(len(e) - 1, bool)
    if rod_x is not None and np.isfinite(rod_x):
        keep = (e[1:] < rod_x - pad_px) | (e[:-1] > rod_x + pad_px)
    keep = keep[:me.shape[1]]
    if not keep.any():
        return np.full(len(me), np.nan)
    with np.errstate(invalid="ignore"):
        return np.nanmean(me[:, :len(keep)][:, keep].astype(np.float32), 1)
