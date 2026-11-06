"""TRUE DAQ times and frame loss for a FaceRhythm task, from the camera's per-frame timestamps (10-23).

The runner (`o2_face_rhythm.py`, ``fr_runner.py``) stamps its 5 Hz session bins and 25 Hz points bins with a LINEAR
frame -> DAQ fit (``daq_fit``). That is exact while the camera drops (almost) nothing -- every cam4 task but one fits
to <= 5 ms -- and wrong when it drops many frames: PS92_20260908 cam4 lost 11.8 % (57 % in its last tenth) and its
bin stamps were off by a median 3.2 s, up to 841 s. The TRIAL grid was always right (each grid time placed at the
camera's own frame through its per-frame timestamps, `face_rhythm_cams.frames_at`).

Here, per task, from the per-frame DAQ times (`face_rhythm_cams.frame_daq_seconds`, through the camera's alignment
template, right across dropped frames):

  t5, t25      each session / points bin's DAQ time = the per-frame time interpolated at the bin's centre frame
               (``frame_center`` in the outputs).
  drop5, drop25, drop_trials   the LOCAL FRAME LOSS around each bin / trial-grid time: 1 - (frames recorded within
               +-LOSSY_HALF_S) / (frames expected at the camera rate). FaceRhythm treats frames as evenly spaced, so
               missing frames compress time inside its spectrogram windows (the lowest frequencies span ~2 s):
               bins with loss > LOSSY_FRAC are LOSSY -> NaN downstream (fit rows, projections, window power).
               NaN times / loss = frames OUTSIDE the camera's DAQ segment (e.g. PS92_0812 v1 ran ~39 min, the DAQ
               recorded its first ~10): unalignable, treated as lossy.

Cached per task in ``<bundle>/timing/<key>.npz`` (TIMING_VERSION). Every reader of bin times goes through `load`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

TIMING_VERSION = "fr-timing-v1"
LOSSY_FRAC = 0.02                        # > 2 % of frames missing within +-LOSSY_HALF_S -> the bin is lossy
LOSSY_HALF_S = 2.0
FPS = 250.0


def local_drop(fd: np.ndarray, t: np.ndarray, half_s: float = LOSSY_HALF_S, fps: float = FPS) -> np.ndarray:
    """Fraction of frames MISSING within +-``half_s`` of each DAQ time ``t`` (per-frame DAQ times ``fd``); a window
    reaching past the recording is judged on its covered part; NaN outside the recording."""
    f = np.sort(fd[np.isfinite(fd)])
    t = np.asarray(t, float)
    lo, hi = np.maximum(t - half_s, f[0]), np.minimum(t + half_s, f[-1])
    n = np.searchsorted(f, hi, side="right") - np.searchsorted(f, lo, side="left")
    exp = (hi - lo) * fps + 1
    out = np.clip(1.0 - n / np.maximum(exp, 1.0), 0.0, 1.0)
    return np.where(np.isfinite(t) & (t >= f[0]) & (t <= f[-1]), out, np.nan)


def bin_times(fd: np.ndarray, frame_center: np.ndarray) -> np.ndarray:
    """DAQ time at fractional frame positions (per-frame times interpolated; NaN outside / in segment gaps)."""
    ok = np.isfinite(fd)
    i = np.flatnonzero(ok)
    out = np.interp(frame_center, i, fd[ok], left=np.nan, right=np.nan)
    return out


def compute(fd: np.ndarray, fc5: np.ndarray, fc25: np.ndarray, grid_daq_s: np.ndarray) -> dict:
    """Everything `load` returns, from the per-frame DAQ times and the outputs' bin centres + the trial grid's DAQ
    times (n_trials, n_t)."""
    t5, t25 = bin_times(fd, fc5), bin_times(fd, fc25)
    g = np.asarray(grid_daq_s, float)
    return {"t5": t5, "t25": t25, "drop5": local_drop(fd, t5), "drop25": local_drop(fd, t25),
            "drop_trials": local_drop(fd, g.ravel()).reshape(g.shape),
            "dropped_frac": float(1 - np.isfinite(fd).sum() / ((np.nanmax(fd) - np.nanmin(fd)) * FPS + 1)),
            "version": TIMING_VERSION, "lossy_frac": LOSSY_FRAC, "lossy_half_s": LOSSY_HALF_S}


def path(bundle_dir: Path, key: str) -> Path:
    return Path(bundle_dir) / "timing" / f"{key}.npz"


def build(rv, bundle_dir: Path, row) -> dict:
    """Compute + cache one task (``row``: its tasks.tsv row: key, animal, date, cam, video)."""
    from wfield_local import face_rhythm_cams as frc
    bundle_dir = Path(bundle_dir)
    fd = frc.frame_daq_seconds(rv, row.animal, row.date, row.cam, row.video)
    od = bundle_dir / "out"
    with np.load(od / f"{row.key}_session.npz") as z:
        fc5 = z["frame_center"].astype(float)
    with np.load(od / f"{row.key}_points.npz") as z:
        fc25 = z["frame_center"].astype(float)
    with np.load(od / f"{row.key}_trials.npz") as z:
        grid = z["cue_daq_s"][:, None].astype(float) + z["t_ms"][None, :].astype(float) / 1000.0
    out = compute(fd, fc5, fc25, grid)
    dst = path(bundle_dir, row.key)
    dst.parent.mkdir(exist_ok=True)
    tmp = dst.with_suffix(".part.npz")
    np.savez(tmp, **out)
    tmp.replace(dst)
    return out


def load(bundle_dir: Path, key: str) -> dict | None:
    """The cached timing of a task (None if missing or an older version -> run `o2_face_rhythm.py timing`)."""
    f = path(bundle_dir, key)
    if not f.exists():
        return None
    with np.load(f) as z:
        d = {k: z[k] for k in z.files}
    return d if str(d.get("version")) == TIMING_VERSION else None


def lossy(drop: np.ndarray) -> np.ndarray:
    """True where the local frame loss exceeds LOSSY_FRAC (NaN loss = outside the recording -> lossy)."""
    d = np.asarray(drop, float)
    return ~(d <= LOSSY_FRAC)
