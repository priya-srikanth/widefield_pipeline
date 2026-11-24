"""NULL TEST of the 3-D jaw onset floor per spout (CC-B 11-23; companion of scripts/jaw3d_all_validate.py).

    python scripts/jaw3d_floor_null.py [--jobs 5] [--labels PS92_0818 ...]

The production 3-D jaw call (`scripts/jaw_onsets._threed`): displacement of the 3-D jaw point from its pre-cue
(-500 .. 0 ms) median; "moved" = `jaw_kinematics.jaw_onsets` (|d - baseline| > max(4 sd, FLOOR), >= 5 of 8 frames)
over cue .. stop. Here the SAME rule runs on two 1,000 ms windows per trial, for floors 0.10-0.40 mm:
  pseudo  a PSEUDO-CUE 1,000 ms before the cue: baseline -1,500 .. -1,000 ms, detect -1,000 .. 0 ms (the spout has been
        in place for > 1.6 s; nothing in the task happens there). Trials with any fused lick peak in -1,500 .. 0 ms are
        dropped from the null, so its "moved" rate is the false-positive rate of the call on a still, lick-free face.
  real  the real cue, baseline -500 .. 0 ms, detect 0 .. 1,000 ms (same length as the null).
Output <review>/jaw3d_qc/floor_null_trials.csv.gz (label, trial_id, position, window, floor_mm, moved, disp95_mm,
precue_lick) and floor_null_summary.csv. Writes only there.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

P3D = Path("M:/MICROSCOPE/Priya/DeepLabCut/Widefield/session_poses/pose3d_4cam_3net_20261121_sync60")
OUT = Path("M:/MICROSCOPE/Priya/stroke_orofacial_pipeline_outputs/vls_stroke_paper/ccB_caption_review_2026-11-23/jaw3d_qc")
QC = Path("M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v37_sync60/imaging_encoding/qc")
FLOORS_MM = (0.10, 0.15, 0.20, 0.25, 0.30, 0.40)
WINDOWS = {"pseudo": 1000.0, "real": 0.0}     # shift of the detect window's zero (ms before the cue)


def session_job(label: str) -> pd.DataFrame | None:
    from scripts import jaw_onsets as jo
    from wfield_local import jaw_kinematics as jk
    from wfield_local import session_licks as SL
    a, d = label.split("_")
    s = f"{a}_2026{d}" if len(d) == 4 else label
    f = P3D / f"{s}_frames.csv.gz"
    if not f.exists():
        return None
    F = pd.read_csv(f, usecols=["trial_id", "position", "t_ms", "stop_ms", "t_daq_ms", "jaw_ml", "jaw_ap", "jaw_dv"])
    F = F.drop_duplicates(["trial_id", "t_ms"]).sort_values(["trial_id", "t_ms"]).reset_index(drop=True)
    L = SL.session_licks(label, include_cam1_only=True)
    pk = pd.to_numeric(L.t_ms, errors="coerce")
    pre_lick = set(L.trial_id[(pk >= -1500) & (pk < 0)].astype(int))
    X = F[["jaw_ml", "jaw_ap", "jaw_dv"]].to_numpy(float)
    ok_all = np.isfinite(X).all(1)
    out = []
    for wname, shift in WINDOWS.items():
        tt = F.t_ms.to_numpy(float) + shift                       # pseudo-cue at t = 0
        dist = np.full(len(F), np.nan)
        rows = {}
        for tid, g in F.groupby("trial_id", sort=False):
            k = g.index.to_numpy()
            t = tt[k]
            ok = ok_all[k]
            pre, win = (t >= -500) & (t < 0) & ok, (t >= 0) & (t <= 1000)
            if pre.sum() < 20 or (win & ok).sum() < max(20, 0.5 * win.sum()):
                continue
            D = X[k] - np.median(X[k][pre], axis=0)
            dd = np.linalg.norm(D, axis=1)
            dist[k] = dd * 1000.0
            rows[int(tid)] = float(np.percentile(dd[win & ok], 95))
        idx = F[["trial_id", "position", "t_ms", "stop_ms", "t_daq_ms"]].copy()
        idx["t_ms"] = tt
        idx["stop_ms"] = 1000.0
        trace, trials, _src = jo._laid_out(dist, idx)
        for fl in FLOORS_MM:
            O = jk.jaw_onsets(trace, jo.FPS, trials, {"degenerate_as_unknown": True,
                                                       "deflection_threshold_min_px": fl * 1000.0})
            O = O.set_index(trials.trial_id.to_numpy())
            for tid, disp in rows.items():
                out.append((label, tid, str(trials.position[trials.trial_id == tid].iloc[0]), wname, fl,
                            O.jaw_pass_qc.get(tid, np.nan), disp, tid in pre_lick))
    return pd.DataFrame(out, columns=["label", "trial_id", "position", "window", "floor_mm", "moved", "disp95_mm",
                                      "precue_lick"])


def twod_job(label: str) -> pd.DataFrame | None:
    """The SAME pseudo-cue / real windows for the production 2-D calls (cam4 4 px, cam1 15 px; `jaw_onsets.OVERRIDES`),
    so the 3-D call's background rate is read against the per-camera calls' own (``--twod``)."""
    from scripts import jaw_onsets as jo
    from wfield_local import pose_windows as pw
    from wfield_local.paths import PathResolver
    rv = PathResolver()
    a, d = label.split("_")
    d = d if len(d) == 8 else f"2026{d}"
    sd = Path(rv.root("orofacial")) / "lp_r4_heatmap_20261013__clean_v37_sync60" / "sessions" / a / d
    if not (sd / "clean.npz").exists():
        return None
    w = np.load(sd / "windows.npz", allow_pickle=False)
    ix4 = pd.DataFrame({k: w[f"ix_{k}"] for k in ("trial_id", "position", "t_ms", "t_daq_ms", "stop_ms")})
    cams = {"cam4": (np.load(sd / "clean.npz")["jaw_y"].astype(float), ix4)}
    d1 = Path(rv.root("dlc")) / "session_poses" / f"{a}_{d}_cam1_full"
    if (d1 / "windows_LP.csv").exists():
        ix1 = pw.index(d1)
        if "t_daq_ms" in ix1:
            P = pw.lp(d1)
            y1 = P[("jaw", "y")].to_numpy(float)
            y1[P[("jaw", "likelihood")].to_numpy(float) < jo.CAM1_JAW_CUT] = np.nan
            cams["cam1"] = (y1, ix1.reset_index(drop=True))
    out = []
    for cam, (y, ix) in cams.items():
        for wname, shift in WINDOWS.items():
            idx = ix[["trial_id", "position", "t_ms", "t_daq_ms", "stop_ms"]].copy()
            idx["t_ms"] = idx.t_ms.to_numpy(float) + shift
            idx["stop_ms"] = 1000.0
            O = jo._camera(y, idx, cam)
            for r in O.itertuples():
                out.append((label, int(r.trial_id), r.position, cam, wname, r.jaw_pass_qc))
    return pd.DataFrame(out, columns=["label", "trial_id", "position", "cam", "window", "moved"])


def main(argv=None) -> int:
    from wfield_local import run_mode as _rm_guard
    _rm_guard.require_default("jaw3d_floor_null")
    from wfield_local import analysis_kit as ak
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--twod", action="store_true", help="the 2-D per-camera calls instead (floor_null_2d_trials)")
    a = ap.parse_args(argv)
    os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
    labels = a.labels or sorted(pd.read_csv(QC / "s1_trial_classes_cue_stop.csv.gz", usecols=["label"]).label.unique())
    res, fails = ak.fan_sessions(labels, twod_job if a.twod else session_job, jobs=a.jobs)
    for s, e in fails:
        print(f"{s}: FAILED {e}")
    T = pd.concat([r for _, r in res if r is not None], ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    name = ("floor_null_2d_trials" if a.twod else "floor_null_trials") + ("_subset" if a.labels else "") + ".csv.gz"
    T.to_csv(OUT / name, index=False, float_format="%.4g")
    print(f"{len(T)} rows, {T.label.nunique()} sessions -> {OUT / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
