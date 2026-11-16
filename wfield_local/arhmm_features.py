"""Whole-session behaviour feature matrices for the AR-HMM (port of Hillary Tong's AR-HMM pipeline,
github.com/htong1/Priya_ARHMM2026, adapted to this rig; decisions 11-16 in docs/status/ENCODING_DESIGN_TIM_STATES_2026-11-16.md).

    python -m wfield_local.arhmm_features --all [--jobs 4]       # one npz per curated session

ONE ROW PER 40 ms BIN (25 Hz) of DAQ time over the session's pose coverage. Columns (FEATURES):
  tongue_ml / _ap / _dv, tongue_ext   3-D tongue tip, mm, mouth-centred (`pose3d` frames; ml signed so + = CONTRALESIONAL
                                      -- the pose3d frame is + = mouse right, lesions are left in cohort 1, sign taken from
                                      `config` stroke laterality); IN THE MOUTH (cam4 tongue_x == 0 and no 3-D point) = 0;
                                      out of the mouth but not triangulated = NaN
  jaw_ml / _ap / _dv                  3-D jaw, mm; gaps filled from cam4 2-D jaw by a per-session linear map (flag
                                      ``jaw_filled``, fit R2 ``jaw_fill_r2``)
  nose_x / _y                        cam4 2-D, mouth-relative px (3-D nose missing on 42-84 % of frames); x signed
                                      contralesional-positive by its correlation with the 3-D tongue ml
  run_fwd, run_back                    treadmill speed mm/s split by sign (backward = paw push; frozen-encoder v7 note)
  fr_<label> x 12                      FaceRhythm `master_pre_v1` factors, ABSOLUTE expression (weights x the bin's FR
                                       normaliser, the encoder's default): 25 Hz in cue -2 .. +4 s, the 5 Hz session bins
                                       elsewhere, interpolated -- the 5 Hz bins interpolated to 25 Hz reproduce the 25 Hz
                                       series at r >= 0.99 for every factor (4 sessions checked 11-16; FR factors are
                                       spectral-power envelopes)
Hillary's version (PS46-50) used 12.5 Hz, 2-D DLC tongue / jaw (6 features, weighted x2), 10 FR consensus factors and
treadmill speed / acceleration; the rate here is 25 Hz because licking is 6-7 Hz (Priya 11-16, "Yes, as proposed").

Gaps: pose NaN runs <= MAX_INTERP_S are linearly interpolated; longer ones stay NaN and the model step splits the
session into segments there. Task events are NOT features (they label syllables afterwards, as in her pipeline).
Output: <out_dir>/sessions/<label>.npz with X (bins x features), t_daq_s, features, nan_frac.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = 1
BIN_S = 0.04
MAX_INTERP_S = 0.2
POSE_PARTS = ("tongue",)           # 3-D tongue; jaw = 3-D filled from cam4, nose = cam4 2-D (see session_features)


def out_dir() -> Path:
    env = os.environ.get("ARHMM_OUT")
    if env:
        return Path(env)
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    return Path(oo.root(PathResolver())) / oc.RUN / "arhmm" / f"features_v{VERSION}"


def contra_sign(animal: str) -> float:
    """+1 when the pose3d ml axis (+ = mouse right) is CONTRALESIONAL (left lesion), -1 for a right lesion."""
    from wfield_local import config
    lat = str(config.animals()[animal].get("stroke_laterality", "L")).upper()[:1]
    return 1.0 if lat == "L" else -1.0


def _bin(t, v, edges):
    """Mean of v in each bin (NaN-aware); t, v 1-D."""
    i = np.searchsorted(edges, t, side="right") - 1
    ok = (i >= 0) & (i < len(edges) - 1) & np.isfinite(v)
    s = np.bincount(i[ok], v[ok], minlength=len(edges) - 1)
    n = np.bincount(i[ok], minlength=len(edges) - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, s / np.maximum(n, 1), np.nan)


def _interp_short(x, max_n):
    """Linear interpolation over NaN runs of at most ``max_n`` bins; longer runs stay NaN."""
    x = x.copy()
    nan = ~np.isfinite(x)
    if not nan.any() or nan.all():
        return x
    idx = np.arange(len(x))
    filled = np.interp(idx, idx[~nan], x[~nan])
    run_id = np.cumsum(np.r_[True, nan[1:] != nan[:-1]])
    for r in np.unique(run_id[nan]):
        m = run_id == r
        if m.sum() <= max_n and m.argmax() > 0 and (len(x) - 1 - idx[m][-1]) > 0:
            x[m] = filled[m]
    return x


def session_features(label: str) -> dict | None:
    from wfield_local import behavior_events as be
    from wfield_local import face_rhythm_outputs as fro
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    from wfield_local.pose3d import pose3d_dir
    from wfield_local.spout_behavior import _daq_h5_for
    rv = PathResolver()
    animal, date = label.split("_")
    fp = pose3d_dir(rv) / f"{animal}_{date}_frames.csv.gz"
    sd = Path(oo.session_dir(rv, oc.RUN, animal, date))
    if not fp.exists() or not (sd / "windows.npz").exists():
        return None
    cols = ["trial_id", "t_ms", "t_daq_ms"] + [f"{p}_{a}" for p in ("tongue", "jaw") for a in ("ml", "ap", "dv")]
    F = pd.read_csv(fp, usecols=cols)
    # tongue IN THE MOUTH = 0 (cam4 convention): cam4 tongue_x == 0 on the paired frame and no 3-D point
    W = np.load(sd / "windows.npz", allow_pickle=True)
    C = np.load(sd / "clean.npz", allow_pickle=True)
    cam4 = pd.DataFrame({"trial_id": W["ix_trial_id"], "t_ms": W["ix_t_ms"], "c4x": C["tongue_x"]})
    F = F.merge(cam4.drop_duplicates(["trial_id", "t_ms"]), on=["trial_id", "t_ms"], how="left")
    # pose3d ``t_daq_ms`` is CUE-RELATIVE (each frame's camera-template time from its trial's cue, `o2_inference`); the
    # absolute DAQ time adds the trial's cue (`trial_classes.session_trials` cue_daq, the DAQ clock of every other stream)
    from wfield_local import trial_classes as tc_
    S = tc_.session_trials(f"{animal}_{date}")
    if S is None:
        return None
    cue_of = dict(zip(S.trial_id.to_numpy(), S.cue_daq.to_numpy(float)))
    t = F.trial_id.map(cue_of).to_numpy(float) + F.t_daq_ms.to_numpy(float) / 1000.0
    ok_t = np.isfinite(t)
    t0, t1 = np.nanmin(t), np.nanmax(t)
    edges = np.arange(t0, t1 + BIN_S, BIN_S)
    tc = edges[:-1] + BIN_S / 2
    sgn = contra_sign(animal)
    cols_out, X = [], []
    tongue_in = (F.c4x.to_numpy(float) == 0) & ~np.isfinite(F.tongue_ml.to_numpy(float))
    # 3-D TONGUE (complete: in the mouth = 0, triangulated otherwise; the direction feature)
    for a in ("ml", "ap", "dv"):
        v = np.where(tongue_in, 0.0, F[f"tongue_{a}"].to_numpy(float))
        X.append(_bin(t[ok_t], (sgn * v if a == "ml" else v)[ok_t], edges))
        cols_out.append(f"tongue_{a}")
    ml, ap, dv = (F[f"tongue_{a}"].to_numpy(float) for a in ("ml", "ap", "dv"))
    ext = np.where(tongue_in, 0.0, np.sqrt(ml ** 2 + ap ** 2 + dv ** 2))
    X.append(_bin(t[ok_t], ext[ok_t], edges))
    cols_out.append("tongue_ext")
    # JAW = 3-D (mm; Priya 11-16: "is 3d jaw not better than cam4"), its gaps FILLED from cam4: on PS92_0822 the 3-D
    # jaw is cam1-only on 75 % of frames, two-camera on 7 %, missing on 16 % in long runs (median 0.7 s; 98 % of missing
    # frames in runs > 0.2 s), so interpolation cannot fill it; cam4 2-D jaw exists on > 99 %. Per session, each 3-D
    # axis is regressed on [1, cam4 jaw x, y] over bins with both, and missing bins take the prediction
    # (``jaw_filled`` = 1 there). NOSE = cam4 2-D (3-D nose missing on 42-84 % of frames). Lateral sign of every cam4
    # x set FROM THE DATA: cam4 tongue x vs the 3-D contralesional tongue ml on the same bins (contra-positive).
    tw = W["ix_daq_s"].astype(float)
    okw = np.isfinite(tw)
    parts = list(W["parts"])
    nose = W["pose"][:, parts.index("nose"), :]
    mouth = np.asarray(C["tongue_origin"], float)
    c4 = {"c4_tongue_x": np.asarray(C["tongue_x"], float),
          "jaw_x": np.asarray(C["jaw_x"], float), "jaw_y": np.asarray(C["jaw_y"], float),
          "nose_x": np.where(nose[:, 2] >= 0.5, nose[:, 0] - mouth[0], np.nan),
          "nose_y": np.where(nose[:, 2] >= 0.5, nose[:, 1] - mouth[1], np.nan)}
    B = {k: _bin(tw[okw], v[okw], edges) for k, v in c4.items()}
    both = np.isfinite(B["c4_tongue_x"]) & np.isfinite(X[0]) & (X[3] > 0)
    xs = float(np.sign(np.corrcoef(B["c4_tongue_x"][both], X[0][both])[0, 1])) if both.sum() > 100 else np.nan
    if not np.isfinite(xs) or xs == 0:
        return None
    jx, jy = xs * B["jaw_x"], B["jaw_y"]
    J3 = {a: _bin(t[ok_t], (sgn * F[f"jaw_{a}"].to_numpy(float) if a == "ml" else F[f"jaw_{a}"].to_numpy(float))[ok_t],
                  edges) for a in ("ml", "ap", "dv")}
    D2 = np.column_stack([np.ones(len(jx)), jx, jy])
    ok2 = np.isfinite(jx) & np.isfinite(jy)
    filled = np.zeros(len(jx), bool)
    jaw_fit_r2 = {}
    for a, v in J3.items():
        fit = ok2 & np.isfinite(v)
        if fit.sum() < 500:
            return None
        beta = np.linalg.lstsq(D2[fit], v[fit], rcond=None)[0]
        pred = D2 @ beta
        jaw_fit_r2[a] = float(1 - np.var(v[fit] - pred[fit]) / np.var(v[fit]))
        miss = ~np.isfinite(v) & ok2
        filled |= miss
        X.append(np.where(miss, pred, v))
        cols_out.append(f"jaw_{a}")
    for k in ("nose_x", "nose_y"):
        X.append(xs * B[k] if k.endswith("_x") else B[k])
        cols_out.append(k)
    nan_pose = np.mean(~np.isfinite(np.column_stack(X)), axis=0)
    X = [_interp_short(x, int(round(MAX_INTERP_S / BIN_S))) for x in X]
    # treadmill (canonical DAQ speed)
    h5 = _daq_h5_for(rv, animal, date)
    if h5 is None:
        return None
    ts, sp = be.session_speed(Path(h5))
    s = _bin(ts, sp, edges)
    X += [np.clip(s, 0, None), np.clip(-s, 0, None)]
    cols_out += ["run_fwd", "run_back"]
    # FaceRhythm, absolute expression, interpolated onto the bins
    fd = fro.factor_dir(rv)
    files = fro.session_keys(fd, animal, date)            # the per-session projections (not the set's master.npz)
    if not files:
        return None
    tf, Wf = fro.combined_series(files, absolute=True)
    names = fro.labels(fd)
    for k in range(Wf.shape[1]):
        v = np.interp(tc, tf, Wf[:, k], left=np.nan, right=np.nan)
        X.append(v)
        cols_out.append(names[k] if k < len(names) else f"fr_{k}")
    X = np.column_stack(X).astype(np.float32)
    return {"X": X, "t_daq_s": tc, "features": np.array(cols_out), "nan_frac_pose_raw": nan_pose,
            "jaw_filled": filled, "jaw_fill_r2": np.array([jaw_fit_r2[a] for a in ("ml", "ap", "dv")]),
            "cam4_x_sign": xs,
            "nan_frac": np.mean(~np.isfinite(X), axis=0), "label": label, "version": VERSION}


def _job(label: str):
    f = out_dir() / "sessions" / f"{label}.npz"
    if f.exists() and not os.environ.get("ARHMM_FORCE"):
        return str(f)
    d = session_features(label)
    if d is None:
        return None
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_name(f.stem + f".tmp{os.getpid()}.npz")
    np.savez_compressed(tmp, **d)
    os.replace(tmp, f)
    return str(f)


def main(argv=None) -> int:
    from wfield_local import analysis_kit as ak
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("labels", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args(argv)
    labels = sorted(ak.curated_ids()) if a.all else a.labels
    res, fails = ak.fan_sessions(labels, _job, jobs=a.jobs)
    print(f"{sum(v is not None for _k, v in res)} sessions written, {len(fails)} failed -> {out_dir()}")
    for f in fails:
        print("  FAIL", f)
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
