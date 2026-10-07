"""Moving vs stationary, per imaging frame, from video MOTION ENERGY -- the Hasnain et al. 2025 definition.

Hasnain, Birnbaum, Ugarte Nunez, Hartman, Chandrasekaran & Economo (2025) Nat Neurosci, "Separating cognitive and
motor processes in the behaving mouse", Methods / Motion energy (Priya, 2026-10-07, who supplied the text):
  "The motion energy for a given frame and pixel was defined as the absolute value of the difference between the
   median value of the pixel across the next 5 frames (12.5 ms) and the median value of the pixel across the
   previous 5 frames (12.5 ms). Motion energy for each frame was then converted to a single value by taking the
   99th percentile (~700 pixels) of motion energy values across the frame. ... A threshold above which an animal
   was classed as moving, was defined on a per session basis manually. Motion energy distributions, per session,
   were bimodal ... The threshold was set as the motion energy value separating these two modes."

OURS (documented differences): 250 fps cameras, so the median windows are ``win`` = 3 frames (12 ms, their time
scale; 5 frames would be 20 ms); frames area-downsampled by ``ds`` first; the threshold is set AUTOMATICALLY at the
density valley between the two modes of log motion energy (`bimodal_threshold`), per session and camera, with a
histogram figure to check it and a per-session override in configs (`motion_state.thresholds`) -- theirs was set
by eye. Each camera's per-frame value is then reduced to the imaging clock by the MAX over the camera frames in an
imaging frame (any movement inside the frame counts).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def frame_scalar_worker(path: str, f0: int, f1: int, ds: int, win: int, pct: float) -> np.ndarray:
    """Per-frame motion energy for frames [f0, f1) of one video: |median(next win) - median(previous win)| per
    pixel, then the ``pct`` percentile over pixels. Frames without a full window on both sides are NaN. Decodes
    from f0 - win so chunk edges are exact (`video_motion.chunk_edges` splits the video)."""
    import cv2

    from wfield_local import video_motion as VM
    cap = cv2.VideoCapture(str(path))
    start = max(f0 - win, 0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    buf = []
    out = np.full(f1 - f0, np.nan, np.float32)
    # frame i's value needs frames i-win+1 .. i (previous, inclusive of i) and i+1 .. i+win (next)
    for i in range(start, f1 + win):
        ok, im = cap.read()
        if not ok:
            break
        buf.append(VM.downsample(im, ds))
        if len(buf) > 2 * win:
            buf.pop(0)
        c = i - win                                   # the frame whose window just completed
        if len(buf) == 2 * win and f0 <= c < f1:
            prev = np.median(np.stack(buf[:win]), 0)
            nxt = np.median(np.stack(buf[win:]), 0)
            out[c - f0] = np.percentile(np.abs(nxt - prev), pct)
    cap.release()
    return out


def session_frame_scalar(path: Path, n_frames: int, ds: int = 4, win: int = 3, pct: float = 99.0,
                         workers: int = 6) -> np.ndarray:
    """`frame_scalar_worker` over the whole video, split into ``workers`` frame ranges (processes)."""
    from concurrent.futures import ProcessPoolExecutor
    edges = np.linspace(0, n_frames, workers + 1).round().astype(int)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(frame_scalar_worker, str(path), int(a), int(b), ds, win, pct)
                for a, b in zip(edges[:-1], edges[1:])]
        return np.concatenate([f.result() for f in futs])


def to_imaging(values: np.ndarray, cam_t_s: np.ndarray, frame_times_s: np.ndarray) -> np.ndarray:
    """Camera-frame values -> imaging frames, MAX over the camera frames inside each imaging frame (NaN if none)."""
    from wfield_local import video_motion as VM
    b = VM.frame_bins(cam_t_s, frame_times_s)
    out = np.full(len(frame_times_s), -np.inf)
    ok = (b >= 0) & np.isfinite(values)
    np.maximum.at(out, b[ok], values[ok])
    out[~np.isfinite(out)] = np.nan
    return out


def bimodal_threshold(values: np.ndarray, n_grid: int = 512) -> float:
    """The motion-energy value at the density VALLEY between the two modes of log(values) (2-component Gaussian
    mixture on log ME; the valley = minimum of the mixture density between the two means). Falls back to the
    posterior-equality point if the density has no interior minimum."""
    from sklearn.mixture import GaussianMixture
    v = np.asarray(values, float)
    v = v[np.isfinite(v) & (v > 0)]
    lv = np.log(v)[:, None]
    gm = GaussianMixture(2, random_state=0).fit(lv)
    m = np.sort(gm.means_.ravel())
    grid = np.linspace(m[0], m[1], n_grid)[:, None]
    dens = np.exp(gm.score_samples(grid))
    k = int(np.argmin(dens))
    if 0 < k < n_grid - 1:
        return float(np.exp(grid[k, 0]))
    post = gm.predict_proba(grid)
    hi = int(np.argmax(gm.means_.ravel()))
    return float(np.exp(grid[int(np.argmin(np.abs(post[:, hi] - 0.5))), 0]))


def label_frames(me_by_cam: dict, thresholds: dict, frame_times_s: np.ndarray, lick_s=None, running=None,
                 run_thresh: float | None = None, buffer_s: float = 0.3) -> np.ndarray:
    """Per imaging frame: 1 = moving, 0 = stationary, -1 = excluded (no data, or within ``buffer_s`` AFTER a
    moving frame: GCaMP decays over a few hundred ms, so activity just after a movement is not 'stationary').
    Moving = any camera above its threshold, or a DAQ lick in the frame, or running above ``run_thresh``."""
    ft = np.asarray(frame_times_s, float)
    mov = np.zeros(len(ft), bool)
    have = np.zeros(len(ft), bool)
    for cam, v in me_by_cam.items():
        fin = np.isfinite(v)
        have |= fin
        mov |= fin & (v > thresholds[cam])
    if lick_s is not None and len(lick_s):
        from wfield_local import video_motion as VM
        b = VM.frame_bins(np.asarray(lick_s, float), ft)
        mov[b[b >= 0]] = True
    if running is not None and run_thresh is not None:
        mov |= np.abs(np.asarray(running, float)) > run_thresh
    lab = np.where(mov, 1, 0).astype(int)
    lab[~have] = -1
    n_buf = int(round(buffer_s / np.median(np.diff(ft))))
    if n_buf > 0:
        ends = np.flatnonzero(mov[:-1] & ~mov[1:])
        for e in ends:
            seg = slice(e + 1, min(e + 1 + n_buf, len(lab)))
            lab[seg] = np.where(lab[seg] == 0, -1, lab[seg])
    return lab
