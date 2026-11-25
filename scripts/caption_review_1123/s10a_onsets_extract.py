"""S10A onsets (caption review 11-23, ccB helper): per-session extraction of the raw signals behind the rev-up "first
movement", so every threshold / baseline / matching variant is computed offline from one cache (no re-extraction).

Per session (label "PS92_0806" as in the rev-up table) one pickle ``<out>/<label>.pkl``:
  fr      FaceRhythm master_pre_v1 (cam4) lick vigor = sum of the six lick factors (`revup_behaviour.LICK`), ABSOLUTE
          (x the bin normalizer D, as `fr_signals(absolute=True)`), UNBLANKED; the raw combined series (t_daq_s, v):
          25 Hz inside the trial windows (cue - 2 .. cue + 4 s), 5 Hz session samples outside them
          (`face_rhythm_outputs.combined_series`). The production z / onset are recomputed from it by linear
          interpolation onto cue + `revup_behaviour.LAGS_CUE` (`to_frames`, max gap 0.5 s), exactly as `session()`.
  jaw3d   the four-camera 3-D jaw (`session_poses/pose3d_4cam_3net_20261121_sync60/<s>_frames.csv.gz`): per trial the
          3-D displacement from the pre-cue (-500 .. 0 ms) median, in um, the quantity `scripts/jaw_onsets.py` runs
          its onset rule on (floor 0.20 mm), plus the vertical component (dv, + = up), for t_ms in -1000 .. 6000;
          time = t_daq_ms (ms from the cue on the DAQ clock).
  cam1    cam1 2-D detector events (`cam1_licks/<s>_events_all.csv`) with: cam4_matched (production flag: overlaps a
          cam4 2-D lick), fused_matched (11-23: overlaps a lick of THE fused four-camera train, `session_licks`
          without cam1-only rows, same overlap rule `cam1_licks.overlaps`), keep_old / keep_new (`cam1_licks.keep_mask`
          with each flag: droplets + far-from-mouth labels out).
  licks   the fused train (cam1-only rows dropped = the rev-up first-lick list): trial_id, on_ms, daq_on_s, cams.

    python s10a_onsets_extract.py <out_dir> [--labels ...] [--jobs 2]
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

P3D = Path("M:/MICROSCOPE/Priya/DeepLabCut/Widefield/session_poses/pose3d_4cam_3net_20261121_sync60")
REVUP = Path("M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v37_sync60/imaging_encoding/"
             "revup_fourcam_cam3on/revup_trials.csv.gz")
T_LO, T_HI = -1000.0, 6000.0


def _fr(an: str, date: str, rv):
    import revup_behaviour as rb
    from wfield_local import face_rhythm_outputs as fro
    fdir = fro.factor_dir(rv, set_name=rb.FR_SET)
    files = fro.session_keys(fdir, an, date)
    if not files:
        return None
    t, W = fro.combined_series(files, "factors", None, True)
    names = fro.labels(fdir)
    v = np.zeros(len(t))
    for k in rb.LICK:
        v = v + W[:, names.index(k)]
    return {"t": t.astype(np.float64), "v": v.astype(np.float32)}


def _jaw3d(an: str, date: str):
    f = P3D / f"{an}_{date}_frames.csv.gz"
    if not f.exists():
        return None
    F = pd.read_csv(f, usecols=["trial_id", "t_ms", "t_daq_ms", "jaw_ml", "jaw_ap", "jaw_dv"])
    F = F.drop_duplicates(["trial_id", "t_ms"]).sort_values(["trial_id", "t_ms"]).reset_index(drop=True)
    X = F[["jaw_ml", "jaw_ap", "jaw_dv"]].to_numpy(float)
    t = F.t_ms.to_numpy(float)
    keep = (t >= T_LO) & (t <= T_HI)
    d = np.full(len(F), np.nan)
    dv = np.full(len(F), np.nan)
    for _tid, g in F.groupby("trial_id", sort=False):
        k = g.index.to_numpy()
        ok = np.isfinite(X[k]).all(1)
        pre = (t[k] >= -500) & (t[k] < 0) & ok
        if pre.sum() < 20:
            continue
        D = X[k] - np.median(X[k][pre], axis=0)
        d[k] = np.linalg.norm(D, axis=1) * 1000.0
        dv[k] = D[:, 2] * 1000.0
    return {"trial_id": F.trial_id.to_numpy()[keep].astype(np.int32), "t_daq_ms": F.t_daq_ms.to_numpy(float)[keep],
            "d_um": d[keep].astype(np.float32), "dv_um": dv[keep].astype(np.float32)}


def _cam1(an: str, date: str, rv, A: pd.DataFrame):
    from wfield_local import cam1_licks as C1
    from wfield_local import lp_licks
    f = lp_licks.cam1_events_dir(rv) / f"{an}_{date}_events_all.csv"
    if not f.exists():
        return None
    E = pd.read_csv(f, usecols=lambda c: c in ("trial_id", "on_ms", "off_ms", "t_peak_ms", "daq_on_s", "cam4_matched",
                                               "in_3d", "droplet", "spread_px", "range_px", "med_out_px", "context"))
    E = E[np.isfinite(E.daq_on_s)].reset_index(drop=True)
    m4 = C1.flag_col(E["cam4_matched"])
    mf = C1.overlaps(E, A[["trial_id", "on_ms", "off_ms"]])
    out = pd.DataFrame({"trial_id": E.trial_id.to_numpy(int), "on_ms": E.on_ms.to_numpy(float),
                        "off_ms": E.off_ms.to_numpy(float), "daq_on_s": E.daq_on_s.to_numpy(float),
                        "cam4_matched": m4, "fused_matched": np.asarray(mf, bool),
                        "in_3d": C1.flag_col(E["in_3d"]) if "in_3d" in E else False,
                        "keep_old": C1.keep_mask(E, m4, label=f"{an}_{date}"),
                        "keep_new": C1.keep_mask(E, np.asarray(mf, bool), label=f"{an}_{date}")})
    return out


def one(label: str, out: str) -> dict:
    f = Path(out) / f"{label}.pkl"
    if f.exists():
        return {"label": label, "cached": True}
    t0 = time.time()
    from wfield_local import config
    from wfield_local import session_licks as SL
    from wfield_local.paths import PathResolver
    rv = PathResolver()
    an, date = config.animal_of(label), "2026" + label.split("_")[-1]
    A = SL.session_licks(f"{an}_{date}", rv)
    Af = SL.session_licks(f"{an}_{date}", rv, include_cam1_only=True)
    d = {"label": label, "fr": _fr(an, date, rv), "jaw3d": _jaw3d(an, date), "cam1": _cam1(an, date, rv, A),
         "licks": A[["trial_id", "on_ms", "off_ms", "daq_on_s", "daq_peak_s", "cams"]].reset_index(drop=True),
         "n_cam1_only_fused": int(len(Af) - len(A)), "signature": SL.signature(rv)}
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp")
    tmp.write_bytes(pickle.dumps(d))
    tmp.replace(f)
    return {"label": label, "cached": False, "s": round(time.time() - t0, 1)}


def _one(item):
    return one(*item)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--labels", nargs="*")
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args()
    labels = a.labels or sorted(pd.read_csv(REVUP, usecols=["label"]).label.unique())
    from wfield_local import parallel
    res, fails = parallel.fan_out([(x, a.out) for x in labels], _one, jobs=a.jobs, label="session")
    res = sorted(res, key=lambda t: str(t[0]))            # fan_out returns COMPLETION order (rule 9)
    for _it, r in res:
        print(r, flush=True)
    for it, e in fails:
        print("FAILED", it, e, flush=True)


if __name__ == "__main__":
    main()
