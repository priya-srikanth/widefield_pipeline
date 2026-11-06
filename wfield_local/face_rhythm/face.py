"""Face movement from the frozen point PCs, independent of the factors.

Each projected session (`proj/` ``points_pc``) carries 20 PCs of the centred (x, y) displacement of the tracked mesh
points (components in the set's `master.npz` ``points_components``, INTERLEAVED x0, y0, x1, y1, ...). Velocity =
d(PCs)/dt @ components -> per-point speed |v| (px / s). The 20 PCs carry most but not all point variance (~95 %,
`master.npz` ``points_explained``), so small movements are under-represented equally on both face halves.

Used by the asymmetry, aligned-map and bout-split readouts (FINDINGS 5e, 5f).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

#: frames per chunk of the speed reconstruction (bounds memory: frames x points floats)
CHUNK = 20000


def master_points(fdir: Path) -> tuple[np.ndarray, np.ndarray]:
    """(point components (k, 2P) float32, contra (P,) bool) of a factor set. cam4: image LEFT of the master's LP-nose
    midline = CONTRALESIONAL (all lesions left)."""
    from wfield_local import face_rhythm_readouts as frr
    M = np.load(Path(fdir) / "master.npz")
    return M["points_components"].astype(np.float32), M["home"][:, 0] <= float(frr.MASTER_MIDLINE_X)


def side_speed(pcs: np.ndarray, t: np.ndarray, comps: np.ndarray, contra: np.ndarray,
               W: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """(contra, ipsi) mean point speed per frame (px / s) from the frozen point PCs. ``comps`` (k, 2P) on the INTERLEAVED
    (x0, y0, x1, y1, ...) displacement layout of `points_stream`; ``contra`` (P,) bool. ``W`` (frames, m) event weights
    (how many events' windows cover each frame) -> also (P, m) per-point weighted SUMS of speed (NaN frames skipped)."""
    dt = np.diff(t)
    c = np.full(len(t), np.nan, np.float32)
    i = np.full(len(t), np.nan, np.float32)
    P = comps.shape[1] // 2
    S = np.zeros((P, W.shape[1])) if W is not None else None
    for c0 in range(1, len(t), CHUNK):
        c1 = min(c0 + CHUNK, len(t))
        dv = (pcs[c0:c1] - pcs[c0 - 1:c1 - 1]) / dt[c0 - 1:c1 - 1, None]          # PC velocity
        v = (dv @ comps).reshape(-1, P, 2)
        sp = np.sqrt((v ** 2).sum(-1))                                           # (frames, P)
        c[c0:c1] = sp[:, contra].mean(1)
        i[c0:c1] = sp[:, ~contra].mean(1)
        if W is not None:
            ok = np.isfinite(sp).all(1)
            S += sp[ok].T.astype(np.float64) @ W[c0:c1][ok]
    return c, i, S


def window_weights(t: np.ndarray, ev: np.ndarray, win: tuple[float, float], ok: np.ndarray) -> np.ndarray:
    """Per frame: how many events' [ev + win] windows cover it (frames with NaN speed, ``~ok``, get 0) -- so a weighted
    mean over frames is the mean over events of each event's window mean (overlapping lick windows count per lick)."""
    w = np.zeros(len(t) + 1)
    np.add.at(w, np.searchsorted(t, ev + win[0]), 1)
    np.add.at(w, np.searchsorted(t, ev + win[1], side="right"), -1)
    return np.cumsum(w)[:-1] * ok


def aligned_mean(t: np.ndarray, y: np.ndarray, ev: np.ndarray, lags: np.ndarray, fs: float = 25.0) -> np.ndarray:
    """Mean over events of ``y`` at the nearest frame to event + lag (NaN off the recording: no frame within 1.5
    frames of ``fs``)."""
    if not len(ev):
        return np.full(len(lags), np.nan)
    q = ev[:, None] + lags[None]
    k = np.clip(np.searchsorted(t, q), 1, len(t) - 1)
    k = np.where(np.abs(t[k - 1] - q) < np.abs(t[k] - q), k - 1, k)
    out = y[k].astype(float)
    out[np.abs(t[k] - q) > 1.5 / fs] = np.nan
    return np.nanmean(out, 0)


def asymmetry(contra: np.ndarray, ipsi: np.ndarray) -> np.ndarray:
    """(contra - ipsi) / (contra + ipsi): > 0 = more contralesional movement."""
    return (contra - ipsi) / (contra + ipsi)
