"""Whole-session behaviour feature matrices for the AR-HMM (port of Hillary Tong's AR-HMM pipeline,
github.com/htong1/Priya_ARHMM2026, adapted to this rig; decisions 11-16 in docs/status/ENCODING_DESIGN_TIM_STATES_2026-11-16.md).

    python -m wfield_local.arhmm_features --all [--jobs 4]       # one npz per curated session

ONE ROW PER 40 ms BIN (25 Hz) of DAQ time over the session's pose coverage. Columns (FEATURES):
  tongue_ml / tongue_ap / tongue_ext  3-D tongue (`combined_encoding.tongue_3d_traces`, the encoder's TONGUE=3d):
                                      lateral (signed + = CONTRALESIONAL), forward, 3-D distance; in the mouth = 0
  jaw_x / jaw_y, nose_x / nose_y     cam4 (the encoder's jaw / nose: cleaned jaw, nose minus mouth, likelihood >= 0.6),
                                      x signed contralesional-positive (sign from the cam4-vs-3-D tongue correlation)
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

VERSION = 2
BIN_S = 0.04
MAX_INTERP_S = 0.2


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
    """The ENCODER'S production behaviour variables (`combined_encoding.load_behaviour` with TONGUE=3d; Priya 11-16:
    "don't reinvent the wheel - what did we use for our encoding analyses?"): 3-D tongue via
    `combined_encoding.tongue_3d_traces` (lateral ml, forward ap, 3-D distance; in the mouth = 0, untriangulated = NaN),
    cam4 cleaned jaw x / y (v35 jaw cleaning, rest-relative), cam4 nose minus the mouth (likelihood >= 0.6), the
    canonical DAQ treadmill speed -- the same loaders, so the AR-HMM and the encoders see identical behaviour. Added
    here: 25 Hz binning of the whole session, lateral columns signed contralesional-positive, running split by sign,
    FR factors (below)."""
    from wfield_local import behavior_events as be
    from wfield_local import combined_encoding as ce
    from wfield_local import face_rhythm_outputs as fro
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    from wfield_local.spout_behavior import _daq_h5_for
    rv = PathResolver()
    animal, date = label.split("_")
    sd = Path(oo.session_dir(rv, oc.RUN, animal, date))
    f3 = ce.pose3d_frames_path(rv, animal, date)
    h5 = _daq_h5_for(rv, animal, date)
    if not f3.exists() or not (sd / "clean.npz").exists() or h5 is None:
        return None
    S = oo.load_session(rv, oc.RUN, animal, date)
    t_pose = S["index"].daq_s.to_numpy(float)
    cl, pose = S["clean"], S["pose"]
    o = np.load(sd / "clean.npz")["tongue_origin"].astype(float)
    F3 = pd.read_csv(f3, usecols=["trial_id", "t_ms", "tongue_ml", "tongue_ap", "tongue_dv"])
    c4x = cl.tongue_x.to_numpy(float)
    tx, ty, prot = ce.tongue_3d_traces(S["index"], c4x, F3)
    jx, jy = cl.jaw_x.to_numpy(float), cl.jaw_y.to_numpy(float)
    nlik = pose.nose_likelihood.to_numpy(float) >= 0.6
    nx = np.where(nlik, pose.nose_x.to_numpy(float) - o[0], np.nan)
    ny = np.where(nlik, pose.nose_y.to_numpy(float) - o[1], np.nan)
    ok_t = np.isfinite(t_pose)
    edges = np.arange(np.nanmin(t_pose), np.nanmax(t_pose) + BIN_S, BIN_S)
    tc = edges[:-1] + BIN_S / 2
    sgn = contra_sign(animal)                                # pose3d ml: + = mouse right
    # cam4 image x (jaw, nose) -> contralesional-positive, sign FROM THE DATA: cam4 tongue x vs the 3-D ml on the same
    # frames (cam4 face video: ipsilesional = image right, memory image-side-conventions; checked, not assumed)
    both = np.isfinite(c4x) & np.isfinite(tx) & (np.abs(tx) > 0)
    xs = float(np.sign(np.corrcoef(c4x[both], tx[both])[0, 1])) * sgn if both.sum() > 1000 else np.nan
    if not np.isfinite(xs) or xs == 0:
        return None
    traces = {"tongue_ml": sgn * tx, "tongue_ap": ty, "tongue_ext": prot,
              "jaw_x": xs * jx, "jaw_y": jy, "nose_x": xs * nx, "nose_y": ny}
    cols_out, X = [], []
    for k, v in traces.items():
        X.append(_bin(t_pose[ok_t], v[ok_t], edges))
        cols_out.append(k)
    nan_pose = np.mean(~np.isfinite(np.column_stack(X)), axis=0)
    X = [_interp_short(x, int(round(MAX_INTERP_S / BIN_S))) for x in X]
    ts, sp = be.session_speed(Path(h5))                       # the encoder's `session_speed`
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
