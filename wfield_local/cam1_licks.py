"""cam1 2-D LICK EVENTS (Priya 10-19: "let's add a cam1 2d lick detector") -- the incomplete licks at the lips that cam4
and the 3-D track miss.

Why a 2-D detector: a tongue tip seen by cam1 alone, away from any both-camera frame, has no 3-D point (the depth-prior
check, `scripts/pose3d_depth_prior_check.py`: 0.6-1.7 mm error on licks < 1 mm -- no usable depth), so the combined 3-D
track cannot contain it. Counting and sizing such licks needs no depth: cam1 (from below) sees the tongue leave the mouth.

    from wfield_local import cam1_licks as C1
    E = C1.detect(out_px, lat_px, present, trial_id, t_ms, src_frame, stop_ms, position)

An EVENT = a run of PRESENT cam1 tongue frames (after the cam1 cleaning: measured, or bridged by its <= 24 ms gap rule)
within one trial, consecutive video frames, cue (0 ms) .. trial stop, >= ``min_frames`` frames. Per event: on / off,
the PEAK = the frame farthest from the cam1 mouth (`spout_frame.from_medians_lr` origin), its distance / ``out``
(ventral, toward the spouts) / ``lat`` (+ = mouse's left) in px, and ``size_class``: ``lips`` when the peak never gets
more than ``lips_px`` below the mouth (``out``; `scripts/cam1_lick_aim.py`'s "dorsal runs", 10-18: a pale bump at the
lips, mouth nearly closed) else ``out``. The 20 px default is the 10-18 value -- PROPOSED, to be set from the blind review.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_FRAMES = 3
LIPS_PX = 20.0
CONTACT_TOL_MS = 20.0


def detect(out_px, lat_px, present, trial_id, t_ms, src_frame, stop_ms, position, *, min_frames: int = MIN_FRAMES,
           lips_px: float = LIPS_PX) -> pd.DataFrame:
    out_px, lat_px = np.asarray(out_px, float), np.asarray(lat_px, float)
    t, stop = np.asarray(t_ms, float), np.asarray(stop_ms, float)
    tid, src, pos = np.asarray(trial_id), np.asarray(src_frame), np.asarray(position)
    live = np.asarray(present, bool) & np.isfinite(out_px) & np.isfinite(lat_px) & (t >= 0) & (t <= stop)
    k = np.flatnonzero(live)
    cols = ["trial_id", "position", "on_ms", "off_ms", "t_peak_ms", "n_frames", "peak_dist_px", "peak_out_px",
            "peak_lat_px", "max_out_px", "size_class", "lick_idx"]
    if not len(k):
        return pd.DataFrame(columns=cols)
    brk = np.r_[True, (tid[k[1:]] != tid[k[:-1]]) | (src[k[1:]] - src[k[:-1]] != 1)]
    starts = np.flatnonzero(brk)
    ends = np.r_[starts[1:], len(k)]
    rows = []
    for a, b in zip(starts, ends):
        kk = k[a:b]
        if len(kk) < min_frames:
            continue
        d = np.hypot(out_px[kk], lat_px[kk])
        pk = kk[int(np.argmax(d))]
        mo = float(np.max(out_px[kk]))
        rows.append({"trial_id": int(tid[kk[0]]), "position": str(pos[kk[0]]), "on_ms": t[kk[0]], "off_ms": t[kk[-1]],
                     "t_peak_ms": t[pk], "n_frames": len(kk), "peak_dist_px": float(d.max()),
                     "peak_out_px": out_px[pk], "peak_lat_px": lat_px[pk], "max_out_px": mo,
                     "size_class": "lips" if mo <= lips_px else "out"})
    E = pd.DataFrame(rows, columns=cols[:-1])
    E["lick_idx"] = E.groupby("trial_id").cumcount() if len(E) else []
    return E


def overlaps(E: pd.DataFrame, W: pd.DataFrame, on: str = "on_ms", off: str = "off_ms", tol_ms: float = 0.0) -> np.ndarray:
    """Per event of ``E``: does any window of ``W`` (same trial_id; ``on`` .. ``off``) overlap it, +- ``tol_ms``?"""
    win = {k: g[[on, off]].to_numpy(float) for k, g in W.groupby("trial_id")}
    res = []
    for r in E.itertuples():
        w = win.get(r.trial_id)
        res.append(bool(w is not None and ((w[:, 0] <= r.off_ms + tol_ms) & (w[:, 1] >= r.on_ms - tol_ms)).any()))
    return np.array(res, bool)


def contact_flags(E: pd.DataFrame, contacts: dict, tol_ms: float = CONTACT_TOL_MS) -> np.ndarray:
    """A DAQ spout-contact onset (``contacts`` = {trial_id: [ms from cue]}) within the event +- ``tol_ms``."""
    res = []
    for r in E.itertuples():
        c = np.asarray(contacts.get(r.trial_id, []), float)
        res.append(bool(((c >= r.on_ms - tol_ms) & (c <= r.off_ms + tol_ms)).any()))
    return np.array(res, bool)


def mm_per_px(lat_by_pos: dict, ml_by_pos: dict) -> float:
    """cam1 image scale at the spouts: |ml(far_L) - ml(far_R)| (commanded, mm) / |lat(far_L) - lat(far_R)| (image px,
    along the cam1 frame's left-right axis). The tongue at the lips sits at about the spout tips' depth; reach toward the
    camera is foreshortened, so mm values are a SIZE scale, not 3-D distances."""
    d_px = abs(lat_by_pos["far_L"] - lat_by_pos["far_R"])
    return float(abs(ml_by_pos["far_L"] - ml_by_pos["far_R"]) / d_px) if d_px > 0 else float("nan")
