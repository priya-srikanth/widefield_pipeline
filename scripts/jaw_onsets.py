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

10-21 later, APPENDED columns (existing ones unchanged; Priya's attempt review: 0 / 24 trials judged attempts, all 11
jaw_only = jaw responses WITHOUT opening / other movement / still; walking below the running threshold moves the jaw):
  cam4_open_px / cam1_open_px   largest jaw OPENING from the pre-cue median, cue .. stop (+y in BOTH views: the chin
                                drops in cam4 and opens downward from below in cam1), px
  cam4_up_px / cam1_up_px       largest deflection the OTHER way (jaw up / clench), px
  jaw_dir                       moved trials: "open" / "up" from the onset camera (open_px >= up_px -> open); "" else
  cam4_contrast / cam1_contrast mean |frame-to-frame jaw speed| cue .. min(stop, 1500 ms) / the same over -1000 .. 0 ms:
                                ~1 = the jaw only did what it was already doing (e.g. walking jitter), > 1 = a cue response
  tread_pre_mm_s / tread_post_mm_s  mean DAQ treadmill speed -1000 .. 0 ms and cue .. stop (`behavior_events.session_speed`;
                                running threshold 3 mm/s, rest floor ~0.2: sub-running steps show as ~0.5-3)

10-23, APPENDED 3-D columns (Priya: "use 3-D jaw ... as well as 3-D nose"), DESCRIPTIVE -- the per-camera call above stays
the jaw-moved classifier: on her 24 reviewed trials (side spouts) 3-D caught 5 / 7 of her jaw calls with 1 / 5 false
(per-camera 4 / 7, 0 / 5), and 3-D nose separated her "other movement" calls (median 0.40 mm vs jaw 0.16, still 0.11).
From `session_poses/pose3d/<s>_frames.csv.gz` (both cameras + one-camera fill, mm, stage frame: ml + = mouse right,
ap + = forward, dv + = up); per trial, displacement from the pre-cue (-500 .. 0 ms) median over cue .. stop.
BLANK at close_center / far_center (the rod crosses the face in cam1: 3-D jaw / nose < 15 % of pre-cue frames) and
where < 50 % of the window has a 3-D point. Per part P in {jaw, nose}:
  P3d_disp_mm      95th percentile of the 3-D displacement (robust size of the movement)
  P3d_rest_mm      the same over the pre-cue frames (that trial's resting noise; compare with disp)
  P3d_down_mm / P3d_up_mm       largest downward / upward displacement (jaw: down = opening, up = clench)
  P3d_back_mm      largest backward displacement (nose: head pulled back)
  P3d_side_ipsi_mm sideways displacement at the largest movement, + = mouse LEFT = ipsilesional (all four animals)
  P3d_moved / P3d_onset_ms  the per-camera onset rule (`jaw_kinematics.jaw_onsets`, sustained 5 / 8 frames, > 4 sd) on
                   the 3-D displacement, floor 0.10 mm; onset on the DAQ clock (PROVISIONAL: floor not tuned)
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
#: 3-D columns (10-23): positions where the rod hides the face in cam1 -> blank; onset floor in MICROMETRES (the onset
#: rule's floor parameter is in trace units; the 3-D trace is displacement in um)
CENTRE = ("close_center", "far_center")
OVERRIDE_3D = {"degenerate_as_unknown": True, "deflection_threshold_min_px": 100.0}


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


def _extras(y: np.ndarray, ix: pd.DataFrame, cam: str) -> pd.DataFrame:
    """Per trial: opening / upward deflection (px) and the post / pre jaw-speed contrast (see the module docstring)."""
    rows = []
    for tid, g in ix.groupby("trial_id", sort=False):
        k = g.index.to_numpy()
        t = g.t_daq_ms.to_numpy(float)
        v = y[k]
        st = float(g.stop_ms.iloc[0]) if np.isfinite(g.stop_ms.iloc[0]) else 3500.0
        pre = (t >= -500) & (t < 0) & np.isfinite(v)
        win = (t >= 0) & (t <= st) & np.isfinite(v)
        op = up = ct = np.nan
        if pre.sum() >= 5 and win.any():
            dv = v[win] - np.nanmedian(v[pre])
            op, up = float(max(np.nanmax(dv), 0.0)), float(max(-np.nanmin(dv), 0.0))
        sp = np.abs(np.diff(v))
        tm = t[1:]
        a_ = (tm >= 0) & (tm <= min(st, 1500.0)) & np.isfinite(sp)
        b_ = (tm >= -1000) & (tm < 0) & np.isfinite(sp)
        if a_.sum() >= 5 and b_.sum() >= 5:
            ct = float(np.mean(sp[a_]) / max(np.mean(sp[b_]), 1e-6))
        rows.append({"trial_id": int(tid), f"{cam}_open_px": op, f"{cam}_up_px": up, f"{cam}_contrast": ct})
    return pd.DataFrame(rows)


def _threed(s: str, root_d: str) -> pd.DataFrame | None:
    """Per trial 3-D jaw / nose displacement columns (module docstring, 10-23); None without 3-D frames."""
    from wfield_local import jaw_kinematics as jk
    f = Path(root_d) / "session_poses" / "pose3d" / f"{s}_frames.csv.gz"
    if not f.exists():
        return None
    cols = ["trial_id", "position", "t_ms", "stop_ms", "t_daq_ms"] + [f"{p}_{a}" for p in ("jaw", "nose")
                                                                     for a in ("ml", "ap", "dv")]
    F = pd.read_csv(f, usecols=cols).drop_duplicates(["trial_id", "t_ms"]).sort_values(["trial_id", "t_ms"])
    F = F.reset_index(drop=True)
    out = {}
    for part in ("jaw", "nose"):
        X = F[[f"{part}_ml", f"{part}_ap", f"{part}_dv"]].to_numpy(float)
        dist = np.full(len(F), np.nan)
        rows = []
        for tid, g in F.groupby("trial_id", sort=False):
            k = g.index.to_numpy()
            t = g.t_ms.to_numpy(float)
            st = float(g.stop_ms.iloc[0]) if np.isfinite(g.stop_ms.iloc[0]) else 3500.0
            ok = np.isfinite(X[k]).all(1)
            pre, win = (t >= -500) & (t < 0) & ok, (t >= 0) & (t <= st)
            r = {"trial_id": int(tid)}
            if str(g.position.iloc[0]) in CENTRE or pre.sum() < 20 or (win & ok).sum() < max(20, 0.5 * win.sum()):
                rows.append(r)
                continue
            D = X[k] - np.median(X[k][pre], axis=0)
            d = np.linalg.norm(D, axis=1)
            dist[k] = d * 1000.0                                          # um, for the onset rule
            w = win & ok
            i = int(np.argmax(np.where(w, d, -1.0)))
            r.update({f"{part}3d_disp_mm": float(np.percentile(d[w], 95)),
                      f"{part}3d_rest_mm": float(np.percentile(d[pre], 95)),
                      f"{part}3d_down_mm": float(max(-D[w, 2].min(), 0.0)), f"{part}3d_up_mm": float(max(D[w, 2].max(), 0.0)),
                      f"{part}3d_back_mm": float(max(-D[w, 1].min(), 0.0)), f"{part}3d_side_ipsi_mm": float(-D[i, 0])})
            rows.append(r)
        R = pd.DataFrame(rows).set_index("trial_id")
        if f"{part}3d_disp_mm" not in R:
            R[f"{part}3d_disp_mm"] = np.nan
        trace, trials, src = _laid_out(dist, F[["trial_id", "position", "t_ms", "stop_ms", "t_daq_ms"]])
        O = jk.jaw_onsets(trace, FPS, trials, OVERRIDE_3D)
        row = np.where(O["onset_idx"] >= 0, src[np.clip(O["onset_idx"], 0, None)], -1)
        O["onset_ms"] = np.where(row >= 0, F.t_daq_ms.to_numpy(float)[np.clip(row, 0, None)], np.nan)
        O = O.set_index(trials.trial_id.to_numpy())
        has = R[f"{part}3d_disp_mm"].notna()
        R[f"{part}3d_moved"] = O.jaw_pass_qc.reindex(R.index).where(has)
        R[f"{part}3d_onset_ms"] = O.onset_ms.reindex(R.index).where(has & (R[f"{part}3d_moved"] == 1))
        out[part] = R
    return out["jaw"].join(out["nose"], how="outer")


def _treadmill(a: str, d: str, ix4: pd.DataFrame) -> pd.DataFrame:
    """Per trial mean treadmill speed (mm/s) -1000 .. 0 ms and cue .. stop, on the DAQ clock."""
    from wfield_local import behavior_events, daq_io
    ts, sp = behavior_events.session_speed(daq_io.session_h5(a, d))
    rows = []
    for tid, g in ix4.groupby("trial_id", sort=False):
        t = g.t_daq_ms.to_numpy(float)
        cue_s = float(np.median(g.daq_s.to_numpy(float) - t / 1000.0))          # the trial's cue on the DAQ clock
        st = float(g.stop_ms.iloc[0]) if np.isfinite(g.stop_ms.iloc[0]) else 3500.0

        def mean(lo, hi, cue_s=cue_s):
            m = (ts >= cue_s + lo / 1000.0) & (ts < cue_s + hi / 1000.0)
            return float(np.mean(sp[m])) if m.any() else np.nan
        rows.append({"trial_id": int(tid), "tread_pre_mm_s": mean(-1000, 0), "tread_post_mm_s": mean(0, st)})
    return pd.DataFrame(rows)


def session(job) -> pd.DataFrame:
    """One session's table, CACHED on its input files (`result_cache`; 10-21): a rerun recomputes only new / changed
    sessions."""
    import sys

    from wfield_local import jaw_kinematics, result_cache
    s, root_o, root_d = job
    a, d = s.split("_")
    sd = Path(root_o) / V36 / "sessions" / a / d
    d1 = Path(root_d) / "session_poses" / f"{s}_cam1_full"
    from wfield_local import daq_io
    try:
        h5 = daq_io.session_h5(a, d)
    except Exception:
        h5 = None
    inputs = [sd / "windows.npz", sd / "clean.npz", d1 / "windows_index.csv", d1 / "windows_LP.csv"] + \
        ([Path(h5)] if h5 else []) + [Path(root_d) / "session_poses" / "pose3d" / f"{s}_frames.csv.gz"]
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
    ix4 = pd.DataFrame({k: w[f"ix_{k}"] for k in ("trial_id", "position", "t_ms", "t_daq_ms", "stop_ms", "daq_s")})
    y4 = np.load(sd / "clean.npz")["jaw_y"].astype(float)
    C4 = _camera(y4, ix4, "cam4").rename(columns={"jaw_pass_qc": "cam4_moved", "onset_ms": "cam4_onset_ms",
                                                  "onset_frames": "cam4_onset_frames", "reonset_ms": "cam4_reonset_ms"})
    out = C4[["trial_id", "position", "cam4_moved", "cam4_onset_ms", "cam4_onset_frames", "cam4_reonset_ms"]].copy()
    d1 = Path(root_d) / "session_poses" / f"{s}_cam1_full"
    out["cam1_moved"], out["cam1_onset_ms"], out["cam1_onset_frames"], out["cam1_reonset_ms"] = (np.nan,) * 4
    X = _extras(y4, ix4, "cam4").set_index("trial_id")
    X1 = None
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
            X1 = _extras(y1, ix1, "cam1").set_index("trial_id")
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
    # APPENDED 10-21 (see the docstring): direction, cue contrast, treadmill -- masked where the jaw is unknown
    for cam, XX in (("cam4", X), ("cam1", X1)):
        for c in ("open_px", "up_px", "contrast"):
            out[f"{cam}_{c}"] = out.trial_id.map(XX[f"{cam}_{c}"]) if XX is not None else np.nan
            out.loc[bad, f"{cam}_{c}"] = np.nan
    oc = out.jaw_onset_cam.to_numpy(str)
    op = np.where(oc == "cam4", out.cam4_open_px, out.cam1_open_px)
    up = np.where(oc == "cam4", out.cam4_up_px, out.cam1_up_px)
    out["jaw_dir"] = np.where(oc == "", "", np.where(op >= up, "open", "up"))
    try:
        T = _treadmill(a, d, ix4).set_index("trial_id")
        out["tread_pre_mm_s"] = out.trial_id.map(T.tread_pre_mm_s)
        out["tread_post_mm_s"] = out.trial_id.map(T.tread_post_mm_s)
    except Exception:
        out["tread_pre_mm_s"] = out["tread_post_mm_s"] = np.nan
    # APPENDED 10-23: 3-D jaw / nose (descriptive; module docstring)
    T3 = _threed(s, root_d)
    for part in ("jaw", "nose"):
        for c in ("disp_mm", "rest_mm", "down_mm", "up_mm", "back_mm", "side_ipsi_mm", "moved", "onset_ms"):
            col = f"{part}3d_{c}"
            out[col] = out.trial_id.map(T3[col]) if T3 is not None and col in T3 else np.nan
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
