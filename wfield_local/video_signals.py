"""Video motion-energy regressors on the imaging frames, blanked outside position strobe .. trial end -- library code.

Moved VERBATIM from `scripts/movement_encoding_session.py` on 2026-11-05 (code review 11-04, modularity item 1: `combined_encoding` imported it from scripts/); its one function-local import now reads `session_dir` from `wfield_local.pose_sessions` (same object). The script re-exports it.
"""
from __future__ import annotations
import numpy as np


def video_signals(animal: str, date: str, rv, frame_times_s, cams, k: int = 30) -> dict:
    """{name: (t_s, values)} motion-energy components per camera (`scripts.video_motion_session` output, top ``k``)
    on the imaging frames, BLANKED (NaN = contributes nothing) outside position strobe .. trial end: the spout is
    repositioned between trial end and the next strobe, and spout motion is the TARGET (Priya 2026-10-06: "masking
    spout repositioning sounds good. using position-strobe to trial-end will eliminate this problem"); a stationary
    spout makes no motion energy, so close-spout pixels stay in on far-spout trials."""
    from wfield_local.pose_sessions import session_dir
    from wfield_local import trial_windows as TW
    ft = np.asarray(frame_times_s, float)
    b = TW.trial_bounds(animal, date, rv)
    start = np.where(np.isfinite(b.strobe_s), b.strobe_s, b.cue_s - 0.5).astype(float)
    ok = np.isfinite(start) & np.isfinite(b.stop_s.to_numpy(float))
    keep = np.zeros(len(ft), bool)
    for a0, a1 in zip(start[ok], b.stop_s.to_numpy(float)[ok]):
        keep |= (ft >= a0) & (ft <= a1)
    out = {}
    d = session_dir(rv, f"{animal}:{date}:full")[2]
    for cam in cams:
        z = np.load(d / f"video_motion_{cam}.npz")
        if not np.allclose(z["frame_times_s"][:5], ft[:5]):
            raise ValueError(f"{cam}: motion energy was binned on a different imaging clock")
        tc = np.array(z["timecourses"][:, :k], float)
        tc[~keep] = np.nan
        for j in range(tc.shape[1]):
            out[f"video_{cam}_pc{j}"] = (ft, tc[:, j])
    return out
