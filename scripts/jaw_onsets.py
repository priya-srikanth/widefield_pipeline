"""Per-trial JAW MOVEMENT ONSET, both cameras, on the DAQ clock (encoding window via Priya 10-20: "per-trial jaw movement
onset (ms from cue, DAQ clock, camera that saw it), so jaw-only trials get an attempt latency and we can require onset
after the cue").

    python -m scripts.jaw_onsets [--jobs 3] [--sessions PS92:20260818 ...] [--all]

Rule = the "jaw moved" call itself (`jaw_kinematics.jaw_onsets`, the same parameters / snippet as `jaw_pertrial`:
|y - pre-cue baseline (-500 .. 0 ms)| > max(4 sd, 4 px), sustained >= 5 of 8 frames, detect window cue .. the trial's
own stop); onset = the first suprathreshold frame of the first window where that test passes. Inputs, as
`scripts/trial_attempts.py`:
  cam4  v36 cleaned jaw (``clean.npz`` jaw_y per window row; hidden jaw = NaN); NOT used at close_center (chin hidden)
  cam1  LP jaw y from ``session_poses/<a>_<d>_cam1_full/windows_LP.csv`` (likelihood < 0.6 -> NaN; template-time
        windows from `o2_inference collect`, re-collected 10-20 night)
Trial windows are laid end to end with NaN gaps (no snippet reaches a neighbour); the onset frame's ``t_daq_ms`` (camera
timestamp through the alignment template, ms from the DAQ cue) is the onset time.

Output `<v36 run>/imaging_encoding/qc/jaw_onsets.csv`, one row per trial: session, animal, date, epoch, trial_id,
position, cam4_moved / cam1_moved (1 / 0 / NaN = cannot judge), cam4_onset_ms / cam1_onset_ms, jaw_onset_ms + jaw_onset_cam
(the EARLIER of the cameras that saw a movement), onset_at_cue (the jaw was already deflected at the cue frame:
onset 0 -- not an after-cue movement), cam4_reonset_ms / cam1_reonset_ms / jaw_reonset_ms + jaw_reonset_cam (at-cue
trials: the first NEW movement after the jaw settled, `jaw_kinematics.jaw_onsets`). 10-21: cam1 floor 15 px (cam4 4 px;
Priya's blind reviews), BOTH cameras masked (NaN) at close_center.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

V36 = "lp_r4_heatmap_20261013__clean_v36"
CAM1_JAW_CUT = 0.6                      # as trial_attempts.CAM1_JAW_CUT
#: close_center: chin hidden in cam4 AND the cam1 jaw point jumps under the rod (blind review 10-21: cam1 call at chance)
#: -> both cameras masked there (encoding window masks it too)
JAW_UNRELIABLE = ("close_center",)
OVERRIDES = {"cam4": {"degenerate_as_unknown": True},
             # cam1 floor 15 px (Priya's blind reviews 10-21, frames + clips: cam4's 4 px is jitter at cam1's larger
             # scale; < 12 px mostly still, >= 16 px nearly always moved, 12-16 a coin flip)
             "cam1": {"degenerate_as_unknown": True, "deflection_threshold_min_px": 15.0}}
FPS = 250.0


def _laid_out(y: np.ndarray, idx: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    """Trial windows end to end with NaN gaps -> (trace, trials table for `jaw_onsets`, row of ``idx`` per sample)."""
    gap = int(np.ceil(6000 / 4))
    parts, rows, src, off = [], [], [], 0
    for tid, g in idx.groupby("trial_id", sort=False):
        k = g.index.to_numpy()
        t = g.t_ms.to_numpy(float)
        c = int(np.argmin(np.abs(t)))
        st = g.stop_ms.iloc[0]
        s = int(np.searchsorted(t, st)) if np.isfinite(st) else c + 875
        parts += [y[k], np.full(gap, np.nan)]
        src += [k, np.full(gap, -1)]
        rows.append({"trial_id": int(tid), "position": str(g.position.iloc[0]), "cue_frame": float(off + c),
                     "stop_frame": float(off + max(s, c))})
        off += len(k) + gap
    return np.concatenate(parts), pd.DataFrame(rows), np.concatenate(src)


def _camera(y: np.ndarray, idx: pd.DataFrame, cam: str) -> pd.DataFrame:
    from wfield_local import jaw_kinematics as jk
    trace, trials, src = _laid_out(y, idx)
    O = jk.jaw_onsets(trace, FPS, trials, OVERRIDES[cam])
    td = idx.t_daq_ms.to_numpy(float)
    for col, ic in (("onset_ms", "onset_idx"), ("reonset_ms", "reonset_idx")):
        row = np.where(O[ic] >= 0, src[np.clip(O[ic], 0, None)], -1)
        O[col] = np.where(row >= 0, td[np.clip(row, 0, None)], np.nan)
    O["position"] = trials.position.to_numpy()
    return O


def session(job) -> pd.DataFrame:
    """One session's table, CACHED on its input files (`result_cache`; 10-21): a rerun recomputes only new / changed
    sessions."""
    import sys

    from wfield_local import jaw_kinematics, result_cache
    s, root_o, root_d = job
    a, d = s.split("_")
    sd = Path(root_o) / V36 / "sessions" / a / d
    d1 = Path(root_d) / "session_poses" / f"{s}_cam1_full"
    inputs = [sd / "windows.npz", sd / "clean.npz", d1 / "windows_index.csv", d1 / "windows_LP.csv"]
    ver = result_cache.code_version(jaw_kinematics, sys.modules[__name__])
    return result_cache.cached(Path(root_o) / V36 / "imaging_encoding" / "qc" / "_cache" / "jaw_onsets", s, inputs, ver,
                               lambda: _session(job))


def _session(job) -> pd.DataFrame:
    s, root_o, root_d = job
    from wfield_local import epochs
    from wfield_local import pose_windows as pw
    a, d = s.split("_")
    sd = Path(root_o) / V36 / "sessions" / a / d
    w = np.load(sd / "windows.npz", allow_pickle=False)
    ix4 = pd.DataFrame({k: w[f"ix_{k}"] for k in ("trial_id", "position", "t_ms", "t_daq_ms", "stop_ms")})
    y4 = np.load(sd / "clean.npz")["jaw_y"].astype(float)
    C4 = _camera(y4, ix4, "cam4").rename(columns={"jaw_pass_qc": "cam4_moved", "onset_ms": "cam4_onset_ms",
                                                  "onset_frames": "cam4_onset_frames", "reonset_ms": "cam4_reonset_ms"})
    out = C4[["trial_id", "position", "cam4_moved", "cam4_onset_ms", "cam4_onset_frames", "cam4_reonset_ms"]].copy()
    d1 = Path(root_d) / "session_poses" / f"{s}_cam1_full"
    out["cam1_moved"], out["cam1_onset_ms"], out["cam1_onset_frames"], out["cam1_reonset_ms"] = (np.nan,) * 4
    if (d1 / "windows_LP.csv").exists():
        ix1 = pw.index(d1)                           # binary sidecar of windows_index.csv / windows_LP.csv (10-21)
        if "t_daq_ms" in ix1:                       # template-time windows only (pre-10-20 collects lack it)
            P = pw.lp(d1)
            y1 = P[("jaw", "y")].to_numpy(float)
            y1[P[("jaw", "likelihood")].to_numpy(float) < CAM1_JAW_CUT] = np.nan
            C1 = _camera(y1, ix1, "cam1").set_index("trial_id")
            out["cam1_moved"] = out.trial_id.map(C1.jaw_pass_qc)
            out["cam1_onset_ms"] = out.trial_id.map(C1.onset_ms)
            out["cam1_onset_frames"] = out.trial_id.map(C1.onset_frames)
            out["cam1_reonset_ms"] = out.trial_id.map(C1.reonset_ms)
    bad = out.position.isin(JAW_UNRELIABLE)
    for c in ("cam4_moved", "cam4_onset_ms", "cam4_onset_frames", "cam4_reonset_ms", "cam1_moved", "cam1_onset_ms",
              "cam1_onset_frames", "cam1_reonset_ms"):
        out.loc[bad, c] = np.nan
    o4 = np.where(out.cam4_moved == 1, out.cam4_onset_ms, np.nan)
    o1 = np.where(out.cam1_moved == 1, out.cam1_onset_ms, np.nan)
    both = np.c_[o4, o1]
    out["jaw_onset_ms"] = np.nanmin(np.where(np.isfinite(both), both, np.inf), axis=1)
    out.loc[~np.isfinite(out.jaw_onset_ms), "jaw_onset_ms"] = np.nan
    out["jaw_onset_cam"] = np.where(~np.isfinite(out.jaw_onset_ms), "",
                                    np.where(np.isfinite(o4) & (o4 <= np.where(np.isfinite(o1), o1, np.inf)), "cam4", "cam1"))
    f4 = np.where(out.jaw_onset_cam == "cam4", out.cam4_onset_frames, out.cam1_onset_frames)
    out["onset_at_cue"] = np.where(np.isfinite(out.jaw_onset_ms), f4 == 0, False)
    # re-onset (encoding window via Priya 10-21): for at-cue trials, the earliest NEW movement after the jaw settled,
    # on any camera that saw a movement (`jaw_kinematics.jaw_onsets`); NaN otherwise
    r4 = np.where(out.cam4_moved == 1, out.cam4_reonset_ms, np.nan)
    r1 = np.where(out.cam1_moved == 1, out.cam1_reonset_ms, np.nan)
    rr = np.c_[r4, r1]
    re = np.nanmin(np.where(np.isfinite(rr), rr, np.inf), axis=1)
    out["jaw_reonset_ms"] = np.where(out.onset_at_cue & np.isfinite(re), re, np.nan)
    out["jaw_reonset_cam"] = np.where(~np.isfinite(out.jaw_reonset_ms), "",
                                      np.where(np.isfinite(r4) & (r4 <= np.where(np.isfinite(r1), r1, np.inf)), "cam4", "cam1"))
    out.insert(0, "session", s)
    out.insert(1, "animal", a)
    out.insert(2, "date", d)
    out.insert(3, "epoch", epochs.epoch_of(f"{a}_{d[4:]}"))
    return out.drop(columns=["cam4_onset_frames", "cam1_onset_frames"])


def main(argv=None) -> int:
    from wfield_local import analysis_kit as ak
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--sessions", nargs="*", default=None, metavar="ANIMAL:YYYYMMDD")
    ap.add_argument("--all", action="store_true", help="every v36 session, not just the curated cohort")
    ap.add_argument("--out", type=Path, default=None, help="write here instead (checks; the default file is WATCHED)")
    a = ap.parse_args(argv)
    rv = PathResolver()
    ro, rd = Path(rv.root("orofacial")), Path(rv.root("dlc"))
    sess = sorted(f"{p.parent.name}_{p.name}" for p in (ro / V36 / "sessions").glob("*/*")
                  if (p / "clean.npz").exists())
    if a.sessions:
        sess = [s for s in sess if s.replace("_", ":", 1) in a.sessions]
    elif not a.all:
        sess = [s for s in sess if s in ak.curated_ids()]
    res, fails = ak.fan_sessions([(s, str(ro), str(rd)) for s in sess], session, jobs=a.jobs, key=lambda j: j[0])
    for s, e in fails:
        print(f"{s}: FAILED {e}")
    T = pd.concat([r for _, r in res], ignore_index=True)
    out = a.out or ro / V36 / "imaging_encoding" / "qc" / "jaw_onsets.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    T.to_csv(out, index=False, float_format="%.2f")
    pd.set_option("display.width", 220)
    print(T.groupby(["animal", "epoch"]).agg(trials=("trial_id", "size"), cam4_moved=("cam4_moved", "mean"),
                                             cam1_moved=("cam1_moved", "mean"), onset_med=("jaw_onset_ms", "median"),
                                             at_cue=("onset_at_cue", "mean")).round(3).to_string())
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
