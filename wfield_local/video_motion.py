"""Video MOTION-ENERGY components per camera, on the imaging-frame clock -- Musall-style "uninstructed movement"
regressors for the movement encoding models (`movement_encoding`).

Priya, 2026-10-06: "let's try the video regressors" (after comparing our DLC-keypoint movement model with Musall,
Kaufman, ... Churchland 2019, whose movement regressors were mostly high-dimensional video components and whose
UNINSTRUCTED movements explained the most single-trial variance). Our DLC regressors describe only tongue and jaw;
cam2 / cam3 see the body (paws, posture, wheel) and cam1 / cam4 the whole face (whiskers, nose), so motion energy
from all four captures what the keypoints miss.

Per camera, ONE decoding pass (`binned_motion_energy`): every video frame -> grey, downsampled ``ds``x by area
averaging -> motion energy |f_t - f_(t-1)| -> averaged into IMAGING-frame bins on the DAQ clock (camera alignment
template; bin edges = midpoints between imaging frames, `movement_encoding.bin_to_frames`'s rule) -> one row per
imaging frame in a float16 memmap. Then (`motion_svd`) the spatial components from a row subsample (randomized SVD,
column-centred) and every row projected onto them -> time courses (n_imaging_frames, k), the regressors. Facemap's
"motion SVD", binned before rather than after the projection so the memmap is imaging-frame sized (~275 k rows per
session instead of 2.2 M).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def frame_bins(cam_t_s: np.ndarray, frame_times_s: np.ndarray) -> np.ndarray:
    """Imaging-frame bin of each camera frame (DAQ s); -1 outside the imaging coverage."""
    ft = np.asarray(frame_times_s, float)
    half = np.median(np.diff(ft)) / 2.0
    edges = np.concatenate([[ft[0] - half], (ft[1:] + ft[:-1]) / 2.0, [ft[-1] + half]])
    k = np.searchsorted(edges, np.asarray(cam_t_s, float), side="right") - 1
    return np.where((k >= 0) & (k < len(ft)), k, -1)


def downsample(im: np.ndarray, ds: int) -> np.ndarray:
    """Grey frame (first channel of a grey-as-BGR video) area-averaged by ``ds`` in each axis, float32."""
    g = im[..., 0] if im.ndim == 3 else im
    h, w = (g.shape[0] // ds) * ds, (g.shape[1] // ds) * ds
    return g[:h, :w].reshape(h // ds, ds, w // ds, ds).mean(axis=(1, 3), dtype=np.float32)


def binned_motion_energy(frames, bins: np.ndarray, n_bins: int, out_path: Path, ds: int = 8,
                         progress_every: int = 200_000) -> tuple[np.memmap, np.ndarray]:
    """``frames`` an iterator of video frames in order (frame 0, 1, ...), ``bins`` the imaging bin of each frame
    (`frame_bins`). Returns (memmap (n_bins, n_px) float16 of mean motion energy per bin, bool mask of bins that got
    any frame). Frames are time-ordered, so a bin is accumulated in RAM and written once."""
    prev, mm, cur, acc, n_acc = None, None, -1, None, 0
    filled = np.zeros(n_bins, bool)
    for i, im in enumerate(frames):
        f = downsample(im, ds)
        if mm is None:
            mm = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float16, shape=(n_bins, f.size))
        if prev is not None and i < len(bins) and bins[i] >= 0:
            me = np.abs(f - prev).ravel()
            b = int(bins[i])
            if b != cur:
                if cur >= 0 and n_acc:
                    mm[cur] = acc / n_acc
                    filled[cur] = True
                cur, acc, n_acc = b, np.zeros_like(me), 0
            acc += me
            n_acc += 1
        prev = f
        if progress_every and i and i % progress_every == 0:
            print(f"   {out_path.stem}: {i} frames", flush=True)
    if cur >= 0 and n_acc:
        mm[cur] = acc / n_acc
        filled[cur] = True
    mm.flush()
    return mm, filled


def motion_svd(mm: np.ndarray, filled: np.ndarray, k: int = 50, n_fit: int = 60_000, seed: int = 0,
               chunk: int = 20_000) -> dict:
    """Spatial components from ``n_fit`` evenly spaced filled rows (column-centred, randomized SVD) and every
    filled row's projection. Returns {"timecourses" (n_bins, k) NaN where not filled, "components" (k, n_px),
    "mean" (n_px,), "explained" (k,) fraction of the fitted rows' variance}."""
    from sklearn.utils.extmath import randomized_svd
    rows = np.flatnonzero(filled)
    fit_rows = rows[np.linspace(0, len(rows) - 1, min(n_fit, len(rows))).round().astype(int)]
    X = np.asarray(mm[fit_rows], np.float32)
    mu = X.mean(0)
    X -= mu
    _, s, vt = randomized_svd(X, n_components=k, random_state=seed)
    explained = s ** 2 / float((X ** 2).sum())
    tc = np.full((len(filled), k), np.nan, np.float32)
    for a in range(0, len(rows), chunk):
        r = rows[a:a + chunk]
        tc[r] = (np.asarray(mm[r], np.float32) - mu) @ vt.T
    return {"timecourses": tc, "components": vt.astype(np.float32), "mean": mu, "explained": explained}


def video_frames(path: Path):
    """Every frame of a video, in order (READ-ONLY)."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    try:
        while True:
            ok, im = cap.read()
            if not ok:
                return
            yield im
    finally:
        cap.release()


# --------------------------------------------------------------------------- parallel (Priya 2026-10-06)

def _chunk_worker(path: str, bins: np.ndarray, out_path: str, f0: int, f1: int, ds: int) -> np.ndarray:
    """Frames [f0, f1) of one video into the shared memmap (already created); decoding starts at f0 - 1 so frame
    f0 has its predecessor. Chunk edges sit on imaging-bin changes, so no bin is split between workers. Returns
    the bins this chunk filled. Seeking is frame-accurate on these FMP4 videos (checked 2026-10-06)."""
    import cv2
    mm = np.load(out_path, mmap_mode="r+")
    filled = []
    cap = cv2.VideoCapture(str(path))
    start = max(f0 - 1, 0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    prev, cur, acc, n_acc = None, -1, None, 0
    for i in range(start, f1):
        ok, im = cap.read()
        if not ok:
            break
        f = downsample(im, ds)
        if prev is not None and i >= f0 and bins[i] >= 0:
            me = np.abs(f - prev).ravel()
            b = int(bins[i])
            if b != cur:
                if cur >= 0 and n_acc:
                    mm[cur] = acc / n_acc
                    filled.append(cur)
                cur, acc, n_acc = b, np.zeros_like(me), 0
            acc += me
            n_acc += 1
        prev = f
    if cur >= 0 and n_acc:
        mm[cur] = acc / n_acc
        filled.append(cur)
    cap.release()
    mm.flush()
    return np.asarray(filled, int)


def chunk_edges(bins: np.ndarray, n_chunks: int) -> list[tuple[int, int]]:
    """[(f0, f1), ...] covering all frames, each edge moved forward to the next frame whose imaging bin differs
    from its predecessor's (so a bin never straddles two chunks)."""
    n = len(bins)
    edges = [0]
    for k in range(1, n_chunks):
        e = max(int(round(k * n / n_chunks)), edges[-1] + 1)
        while e < n and bins[e] == bins[e - 1]:
            e += 1
        if e < n and e > edges[-1]:
            edges.append(e)
    edges.append(n)
    return list(zip(edges[:-1], edges[1:]))


def binned_motion_energy_parallel(path: Path, bins: np.ndarray, n_bins: int, out_path: Path, ds: int = 8,
                                  n_workers: int = 6) -> tuple[np.memmap, np.ndarray]:
    """`binned_motion_energy` split over ``n_workers`` processes (each decodes its own frame range)."""
    from concurrent.futures import ProcessPoolExecutor
    first = downsample(next(video_frames(path)), ds)
    mm = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float16, shape=(n_bins, first.size))
    del mm
    filled = np.zeros(n_bins, bool)
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = [ex.submit(_chunk_worker, str(path), bins, str(out_path), f0, f1, ds)
                for f0, f1 in chunk_edges(bins, n_workers)]
        for fu in futs:
            filled[fu.result()] = True
    return np.load(out_path, mmap_mode="r"), filled
