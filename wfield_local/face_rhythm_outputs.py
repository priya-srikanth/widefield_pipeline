"""FaceRhythm factor time courses for the imaging analyses (the frozen master pre-stroke factor set,
`scripts/face_rhythm_factors.py`; plan: docs/FACE_RHYTHM_PLAN.md B).

Per task, `project` writes ``<bundle>/factors/<set>/proj/<key>.npz``: ``trials`` (trial, 151 bins at 25 Hz on the DAQ
grid cue -2 .. +4 s, R), ``session`` (5 Hz bins, R) with ``session_t_daq_s``, per-bin reconstruction errors, and the
set's ``spec_hash``. A session recorded as two videos (PS92_0812) has two tasks (``..._v1``, ``..._v2``); they are
joined here.

``combined_series``  one time series per component on the DAQ clock: the 25 Hz trial windows where they exist (engaged
                     trials), the 5 Hz whole-session bins everywhere else (spout arrival before cue -2 s, after +4 s,
                     between trials). The FR spectrogram envelopes change slowly (bandwidth ~f / Q) and GCaMP smooths
                     further, so the 5 Hz edges cost little; the trial windows keep the full 25 Hz.
``kind="points"``     the dense-POSITION family instead: the frozen PCA of the 25 Hz point displacements
                     (``points_pc``, first ``k`` PCs) -- the paper's "points + FR" comparison.
``fr_signals``       ``{name: (frame_times_s, values)}`` on the imaging frames: linear interpolation (as the paper did),
                     NaN across gaps longer than ``max_gap_s``, and BLANKED outside position strobe .. trial end like
                     the video motion energy (`scripts.movement_encoding_session.video_signals`), so FR and ME enter
                     the encoding model over the same frames.
``source_files``     the npz files behind a session, for the input-cache key (`combined_encoding._cached`): a refrozen
                     or reprojected set must invalidate cached inputs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

BUNDLE = "fr_cam4_20261021"
SET = "master_pre_v1"


def factor_dir(rv=None, bundle: str = BUNDLE, set_name: str = SET) -> Path:
    from wfield_local import o2_inference as oi
    from wfield_local.paths import PathResolver
    return oi.bundle_dir(rv or PathResolver(), bundle) / "factors" / set_name


def session_keys(fdir: Path, animal: str, date: str) -> list[Path]:
    """The proj files of one session, in video order (``_v1`` before ``_v2``)."""
    base = f"{animal}_{date}_cam4"
    fs = [fdir / "proj" / f"{base}.npz"] + sorted((fdir / "proj").glob(f"{base}_v*.npz"))
    return [f for f in fs if f.exists()]


def source_files(fdir: Path, animal: str, date: str) -> list[Path]:
    return [fdir / "master.npz", *session_keys(fdir, animal, date)]


def labels(fdir: Path) -> list[str]:
    """Component names: ``labels.json`` in the set folder if written (e.g. {"0": "sniff", ...}), else fr0..fr<R-1>."""
    R = int(np.load(fdir / "master.npz")["rank"])
    lab = {}
    if (fdir / "labels.json").exists():
        lab = json.loads((fdir / "labels.json").read_text())
    return [f"fr_{lab.get(str(k), k)}" for k in range(R)]


def combined_series(files: list[Path], kind: str = "factors", k: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(t_daq_s (n,), W (n, R)) sorted by time. factors: trial-window 25 Hz samples + session 5 Hz samples outside
    them; points: the 25 Hz displacement PCs (first ``k``)."""
    ts, ws, hashes = [], [], set()
    for f in files:
        z = np.load(f, allow_pickle=False)
        hashes.add(str(z["spec_hash"]))
        if kind == "points":
            if "points_pc" not in z.files:
                raise FileNotFoundError(f"{f}: no points_pc (factor set frozen without the points basis)")
            ts.append(z["points_t_daq_s"].astype(float))
            ws.append(z["points_pc"][:, :k].astype(float))
            continue
        tr, cue, tms = z["trials"].astype(float), z["cue_daq_s"].astype(float), z["t_ms"].astype(float) / 1000.0
        t_tr = cue[:, None] + tms[None, :]
        lo, hi = t_tr[:, 0] - 0.02, t_tr[:, -1] + 0.02                     # half a 25 Hz bin
        ts_s, w_s = z["session_t_daq_s"].astype(float), z["session"].astype(float)
        inside = np.zeros(len(ts_s), bool)
        for a, b in zip(lo, hi):
            inside |= (ts_s >= a) & (ts_s <= b)
        ts += [t_tr.ravel(), ts_s[~inside]]
        ws += [tr.reshape(-1, tr.shape[-1]), w_s[~inside]]
    if len(hashes) > 1:
        raise ValueError(f"projections from different frozen factor sets: {sorted(hashes)}")
    t, W = np.concatenate(ts), np.concatenate(ws)
    ok = np.isfinite(t) & np.isfinite(W).all(1)
    o = np.argsort(t[ok], kind="stable")
    return t[ok][o], W[ok][o]


def to_frames(t: np.ndarray, W: np.ndarray, frame_times_s, max_gap_s: float = 0.5) -> np.ndarray:
    """Linear interpolation of each column onto the frames; NaN where the nearest samples on either side are more
    than ``max_gap_s`` apart (a missing stretch is not bridged) or the frame is outside the samples."""
    ft = np.asarray(frame_times_s, float)
    out = np.full((len(ft), W.shape[1]), np.nan)
    if len(t) < 2:
        return out
    i = np.searchsorted(t, ft)
    ok = (i > 0) & (i < len(t))
    ok[ok] &= (t[i[ok]] - t[i[ok] - 1]) <= max_gap_s
    for k in range(W.shape[1]):
        out[ok, k] = np.interp(ft[ok], t, W[:, k])
    return out


def keep_mask(animal: str, date: str, rv, frame_times_s) -> np.ndarray:
    """Frames inside position strobe .. trial end (the video motion energy's mask)."""
    from wfield_local import trial_windows as TW
    ft = np.asarray(frame_times_s, float)
    b = TW.trial_bounds(animal, date, rv)
    start = np.where(np.isfinite(b.strobe_s), b.strobe_s, b.cue_s - 0.5).astype(float)
    stop = b.stop_s.to_numpy(float)
    keep = np.zeros(len(ft), bool)
    for a0, a1 in zip(start, stop):
        if np.isfinite(a0) and np.isfinite(a1):
            keep |= (ft >= a0) & (ft <= a1)
    return keep


def fr_signals(animal: str, date: str, rv, frame_times_s, fdir: Path | None = None, blank: bool = True,
               max_gap_s: float = 0.5, kind: str = "factors", k: int | None = None) -> dict:
    """{name: (frame_times_s, values)} -- one per factor (``kind="points"``: per displacement PC) -- on the imaging
    frames (see the module docstring)."""
    fdir = fdir or factor_dir(rv)
    files = session_keys(fdir, animal, date)
    if not files:
        raise FileNotFoundError(f"no FaceRhythm projection for {animal}_{date} in {fdir / 'proj'}")
    ft = np.asarray(frame_times_s, float)
    t, W = combined_series(files, kind, k)
    V = to_frames(t, W, ft, max_gap_s)
    if blank:
        V[~keep_mask(animal, date, rv, ft)] = np.nan
    names = labels(fdir) if kind == "factors" else [f"frpt_{j}" for j in range(V.shape[1])]
    return {nm: (ft, V[:, j]) for j, nm in enumerate(names)}
