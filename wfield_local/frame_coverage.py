"""PER-TRIAL CAMERA FRAME COVERAGE -- the gate for camera-derived kinematics (Priya 11-21; refined 11-22).

WHY. PS92 0908 lost 78-89 % of the frames of ALL FOUR cameras after minute ~104 (docs/EXPERIMENT_ERRORS.md
2026-09-08; the N18 miss of trial 306 was a real camera drop, not a missed lick). On such a trial "no video lick" or a
lick with a few 3-D frames says nothing about the animal, so camera licks of those trials must not enter the 3-D
kinematics.

RULE. Coverage is measured on the UNION of the cameras that can contribute to the fused / 3-D lick (coordinator 11-22,
from Priya's question): a 4 ms frame slot of the trial's window counts as covered when ANY of those cameras captured a
frame in it. The fusion already tolerates one camera missing (D2: any camera counts), so only frames missing in all /
most views remove a trial -- PS92 0908's drops are simultaneous across the four cameras and are still gated. A trial
with union coverage < `MIN_FRAME_COVERAGE` (0.90) is gated: its camera licks are skipped by the kinematics, and the
reason is recorded per trial (`GATE_REASON`, the union coverage and each camera's own coverage).

WHAT IS COUNTED. Each camera's OWN frame log: the Bonsai CSV's per-frame timestamps through that camera's alignment
template (`camera_sync.frame_daq_seconds`; a dropped frame is a missing CSV row, so it is a missing time here). The
trial window is the one every camera's 2-D / 3-D windows are cut on (`o2_inference.trial_frame_spans_ts`): position
strobe (or cue - 0.5 s, the earlier) - 0.5 s .. trial stop + 3 s, cut at the next trial's start. Expected frames =
the window length at `FPS`; a camera's coverage = its frames in the window / expected (capped at 1).

Cameras: `camera_licks.CAMS`, minus a side camera the four-camera 3-D run DROPPED for the session (its side gate;
`fusion_cams`) and minus a camera with no template / CSV (it cannot contribute).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

#: a trial whose union frame coverage is below this is gated for camera-derived kinematics (Priya 11-21)
MIN_FRAME_COVERAGE = 0.90
#: frame slot width / expected frame rate (the Blackfly cameras run at ~249.8 fps; slots of 1 / 250 s)
FPS = 250.0
PRE_S, POST_S = 0.5, 3.0             # = `o2_inference.build_windows` defaults
GATE_REASON = f"frame_coverage<{MIN_FRAME_COVERAGE:.2f}"


def trial_windows_s(b: pd.DataFrame, pre_s: float = PRE_S, post_s: float = POST_S) -> pd.DataFrame:
    """Per trial of ``b`` (`trial_windows.trial_bounds`: trial_id, cue_s, strobe_s, stop_s; DAQ s) the window
    (t0_s, t1_s) every camera's windows are cut on (`o2_inference.trial_frame_spans_ts`, by time), cut at the next
    trial's start."""
    b = b[np.isfinite(b.cue_s) & np.isfinite(b.stop_s)].sort_values("cue_s").reset_index(drop=True)
    st = np.where(np.isfinite(b.strobe_s), np.minimum(b.strobe_s, b.cue_s - pre_s), b.cue_s - pre_s) - pre_s
    en = b.stop_s.to_numpy(float) + post_s
    en[:-1] = np.minimum(en[:-1], st[1:])
    return pd.DataFrame({"trial_id": b.trial_id.astype(int).to_numpy(), "t0_s": st, "t1_s": en})


def coverage_table(win: pd.DataFrame, frame_times: dict, fps: float = FPS,
                   min_cov: float = MIN_FRAME_COVERAGE) -> pd.DataFrame:
    """Per trial window of ``win`` (trial_id, t0_s, t1_s): ``n_expected`` frame slots, ``cov_<cam>`` (that camera's
    frames in the window / expected, <= 1) for every camera of ``frame_times`` ({cam: frame DAQ times, s}),
    ``cov_union`` (share of slots in which ANY camera has a frame), ``n_cams``, ``gated`` (cov_union < ``min_cov``)
    and ``gate_reason``. Pure (tested)."""
    t0, t1 = win.t0_s.to_numpy(float), win.t1_s.to_numpy(float)
    n_exp = np.maximum(np.round((t1 - t0) * fps).astype(int), 0)
    out = win[["trial_id"]].copy()
    out["n_expected"] = n_exp
    covered = [np.zeros(n, bool) for n in n_exp]
    cams = sorted(frame_times)
    for cam in cams:
        t = np.sort(np.asarray(frame_times[cam], float))
        t = t[np.isfinite(t)]
        cov = np.zeros(len(win))
        for i in range(len(win)):
            if n_exp[i] == 0:
                continue
            a, b = np.searchsorted(t, t0[i], "left"), np.searchsorted(t, t1[i], "left")
            cov[i] = min(1.0, (b - a) / n_exp[i])
            slot = np.floor((t[a:b] - t0[i]) * fps).astype(int)
            covered[i][slot[(slot >= 0) & (slot < n_exp[i])]] = True
        out[f"cov_{cam}"] = cov
    out["cov_union"] = [float(c.mean()) if len(c) else np.nan for c in covered]
    out["n_cams"] = len(cams)
    out["gated"] = ~(out.cov_union >= min_cov)
    out["gate_reason"] = np.where(out.gated, GATE_REASON, "")
    return out


def fusion_cams(rv, animal: str, date: str) -> tuple:
    """The cameras that can contribute to the session's fused / 3-D licks: `camera_licks.CAMS` minus a side camera the
    four-camera 3-D run dropped (``side_dropped`` of `pose3d_dir()`'s production summary, when that table exists)."""
    from wfield_local import camera_licks as CL
    from wfield_local.pose3d import pose3d_dir
    drop: set = set()
    try:
        d3 = pose3d_dir(rv, "4cam")
        for f in sorted(d3.glob("summary_production*.csv")):
            S = pd.read_csv(f)
            if "side_dropped" in S and "session" in S:
                r = S[S.session == f"{animal}_{date}"]
                for v in r.side_dropped.dropna().astype(str):
                    drop |= {c for c in v.replace("+", ",").split(",") if c.strip()}
    except (OSError, ValueError):
        pass
    return tuple(c for c in CL.CAMS if c not in drop)


def camera_frame_times(rv, animal: str, date: str, cam: str) -> np.ndarray | None:
    """DAQ times (s) of every frame camera ``cam`` logged on ``date`` (all its recordings / segments), or None when it
    has no alignment template or frame log."""
    from wfield_local import camera_sync as cs
    tpls = cs.templates_for(rv, date, animal, cam)
    parts = []
    for _rec, ts in tpls.items():
        for t in ts:                         # a split day: one template per DAQ segment, each valid inside its segment
            try:
                d = cs.frame_daq_seconds(rv, animal, date, t)
            except (ValueError, FileNotFoundError, OSError):
                continue
            seg = t.get("segment_daq_sec")
            if seg is not None and len(ts) > 1:
                d = d[(d >= float(seg[0])) & (d <= float(seg[1]))]
            parts.append(d)
    return np.sort(np.concatenate(parts)) if parts else None


def session_coverage(rv, animal: str, date: str, cams: tuple | None = None) -> pd.DataFrame:
    """`coverage_table` of one session (every trial of `trial_windows.trial_bounds`) on ``cams`` (default
    `fusion_cams`). Cached on the frame logs + templates (`result_cache`)."""
    import sys

    from wfield_local import camera_sync as cs
    from wfield_local import result_cache, session_cache
    cams = tuple(cams) if cams is not None else fusion_cams(rv, animal, date)
    ins = []
    for cam in cams:
        d = Path(rv.root("alignment_templates")) / cam / animal
        ins += sorted(d.glob(f"{date}*.npz"))
        ins += sorted((Path(rv.root("behavior_cameras")) / date / animal).glob(f"{cam}_*.csv"))
    ver = result_cache.code_version(sys.modules[__name__], cs, extra=",".join(cams))
    return result_cache.cached(Path(session_cache.CACHE_DIR) / "frame_coverage", f"{animal}_{date}", ins, ver,
                               lambda: _session_coverage(rv, animal, date, cams))


def _session_coverage(rv, animal: str, date: str, cams: tuple) -> pd.DataFrame:
    from wfield_local import trial_windows as TW
    b = TW.trial_bounds(animal, date, rv)
    ft = {c: t for c in cams if (t := camera_frame_times(rv, animal, date, c)) is not None}
    T = coverage_table(trial_windows_s(b), ft)
    T.insert(0, "session", f"{animal}_{date}")
    T["cams_used"] = "+".join(sorted(ft))
    T["cams_missing"] = "+".join(sorted(set(cams) - set(ft)))
    return T
