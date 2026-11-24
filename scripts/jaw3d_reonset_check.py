"""3-D jaw call DIAGNOSTICS for the six-spout switch (CC-B 11-23; Priya's S10A video verdicts on clips 15 / 28-30).

    python scripts/jaw3d_reonset_check.py [--jobs 5] [--labels PS92_0809 ...]

Recomputes the PRODUCTION 3-D jaw call per trial exactly as `scripts/jaw_onsets._threed` (jaw part; displacement from
the pre-cue -500 .. 0 ms median, `jaw_kinematics.jaw_onsets` over cue .. stop, floor 0.20 mm, coverage rules) and adds
what the production table does not store:
  jaw3d_reonset_ms   the 3-D RE-ONSET for a jaw already moving at the cue: the same rule the 2-D calls use
                     (`jaw_onsets`: the first window that fails the density test, then the next window that passes)
  base_sd_um / thr_um / thr_term   the baseline sd, the APPLIED threshold max(4 sd, floor) and which term set it
                     ("sd_scaled" = the trial's own pre-cue noise, "px_floor" = the 0.20 mm floor)
  peak_um            max |displacement - baseline| over the detect window
The replica's moved / onset / at-cue are written next to the production values (``prod_*``) so the replication is
checked on every trial. Output <review>/jaw3d_qc/jaw3d_reonset_trials.csv.gz (writes only there).
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


def session_job(label: str) -> pd.DataFrame | None:
    from scripts import jaw_onsets as jo
    from wfield_local import jaw_kinematics as jk
    a, d = label.split("_")
    s = f"{a}_2026{d}" if len(d) == 4 else label
    f = P3D / f"{s}_frames.csv.gz"
    if not f.exists():
        return None
    F = pd.read_csv(f, usecols=["trial_id", "position", "t_ms", "stop_ms", "t_daq_ms", "jaw_ml", "jaw_ap", "jaw_dv"])
    F = F.drop_duplicates(["trial_id", "t_ms"]).sort_values(["trial_id", "t_ms"]).reset_index(drop=True)
    X = F[["jaw_ml", "jaw_ap", "jaw_dv"]].to_numpy(float)
    dist = np.full(len(F), np.nan)
    has = {}
    for tid, g in F.groupby("trial_id", sort=False):
        k = g.index.to_numpy()
        t = g.t_ms.to_numpy(float)
        st = float(g.stop_ms.iloc[0]) if np.isfinite(g.stop_ms.iloc[0]) else 3500.0
        ok = np.isfinite(X[k]).all(1)
        pre, win = (t >= -500) & (t < 0) & ok, (t >= 0) & (t <= st)
        if pre.sum() < 20 or (win & ok).sum() < max(20, 0.5 * win.sum()):
            continue
        dd = np.linalg.norm(X[k] - np.median(X[k][pre], axis=0), axis=1)
        dist[k] = dd * 1000.0
        has[int(tid)] = float(np.percentile(dd[win & ok], 95))
    trace, trials, src = jo._laid_out(dist, F[["trial_id", "position", "t_ms", "stop_ms", "t_daq_ms"]])
    O = jk.jaw_onsets(trace, jo.FPS, trials, jo.OVERRIDE_3D)
    td = F.t_daq_ms.to_numpy(float)
    for col, ic in (("onset_ms", "onset_idx"), ("reonset_ms", "reonset_idx")):
        row = np.where(O[ic] >= 0, src[np.clip(O[ic], 0, None)], -1)
        O[col] = np.where(row >= 0, td[np.clip(row, 0, None)], np.nan)
    # baseline / threshold / peak per trial, from the same snippet geometry as jaw_onsets
    p = jk._resolve(jo.OVERRIDE_3D)
    extra = []
    for r in jk._iter_trials(trials):
        stop_ms = jk.TW.stop_ms_of(r.cue_frame, r.stop_frame, jo.FPS)
        det = p["detect_win_ms"] if stop_ms is None else (p["detect_win_ms"][0], max(stop_ms, p["detect_win_ms"][0]))
        lo, hi = min(p["baseline_win_ms"][0], det[0]), max(p["baseline_win_ms"][1], det[1])
        pre_f, post_f = abs(jk._ms_to_frames(lo, jo.FPS)), abs(jk._ms_to_frames(hi, jo.FPS))
        t_ms = np.arange(-pre_f, post_f + 1) * (1000.0 / jo.FPS)
        c = jk.cue_frame_index(r.cue_frame)
        yw = np.full(pre_f + post_f + 1, np.nan) if c is None else jk._snippet(trace, c, pre_f, post_f, len(trace))
        bm, bs = jk._compute_jaw_baseline(yw, t_ms, baseline_win_ms=p["baseline_win_ms"])
        thr, term = jk._compute_jaw_threshold(bs, sd_mult=p["sd_mult"], min_px=p["min_px"])
        m = jk._window_mask(t_ms, det) & np.isfinite(yw)
        peak = float(np.nanmax(np.abs(yw[m] - bm))) if m.any() and np.isfinite(bm) else np.nan
        extra.append({"trial_id": r.trial_id, "base_sd_um": bs, "thr_um": thr, "thr_term": term, "peak_um": peak})
    E = pd.DataFrame(extra).set_index("trial_id")
    O = O.set_index(trials.trial_id.to_numpy())
    rows = []
    for tid, disp in has.items():
        o = O.loc[tid]
        moved = o.jaw_pass_qc
        at = bool(moved == 1 and o.onset_frames == 0)
        rows.append({"label": label, "session": s, "trial_id": tid, "position": str(trials.position[trials.trial_id == tid].iloc[0]),
                     "jaw3d_moved": moved, "jaw3d_onset_ms": o.onset_ms if moved == 1 else np.nan, "jaw3d_at_cue": at,
                     "jaw3d_reonset_ms": o.reonset_ms if at else np.nan, "disp95_mm": disp, **E.loc[tid].to_dict()})
    R = pd.DataFrame(rows)
    J = pd.read_csv(QC / "jaw_onsets.csv", usecols=["session", "trial_id", "jaw3d_moved", "jaw3d_onset_ms", "jaw3d_at_cue"],
                    low_memory=False)
    J = J[J.session == s].rename(columns={"jaw3d_moved": "prod_moved", "jaw3d_onset_ms": "prod_onset_ms",
                                          "jaw3d_at_cue": "prod_at_cue"}).drop(columns="session")
    return R.merge(J, on="trial_id", how="left")


def main(argv=None) -> int:
    from wfield_local import run_mode as _rm_guard
    _rm_guard.require_default("jaw3d_reonset_check")
    from wfield_local import analysis_kit as ak
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--labels", nargs="*", default=None)
    a = ap.parse_args(argv)
    os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
    labels = a.labels or sorted(pd.read_csv(QC / "s1_trial_classes_cue_stop.csv.gz", usecols=["label"]).label.unique())
    res, fails = ak.fan_sessions(labels, session_job, jobs=a.jobs)
    for s, e in fails:
        print(f"{s}: FAILED {e}")
    T = pd.concat([r for _, r in res if r is not None], ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    name = "jaw3d_reonset_trials" + ("_subset" if a.labels else "") + ".csv.gz"
    T.to_csv(OUT / name, index=False, float_format="%.4g")
    print(f"{len(T)} trials, {T.label.nunique()} sessions -> {OUT / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
