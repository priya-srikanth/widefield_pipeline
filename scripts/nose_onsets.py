"""Per-trial NOSE MOVEMENT call + onset, both cameras, on the DAQ clock -- a candidate "reacted" class kept SEPARATE
from attempts (Priya 10-21: "sometimes nose could maybe be helpful? (they may also react with nose or whisker
movement?)"; decision: a reaction is not an attempted lick, so nothing here feeds `trial_attempts`).

    python -m scripts.nose_onsets [--jobs 3] [--sessions PS92:20260818 ...] [--all]

Rule = the jaw rule (`jaw_kinematics.jaw_onsets`, same windows and snippet as `scripts.jaw_onsets`): |p - pre-cue
baseline (-500 .. 0 ms)| > max(4 sd, floor), sustained >= 5 of 8 frames, cue .. the trial's own stop -- applied to the
nose x AND y separately; moved = either axis passes, onset = the earlier axis. Floors PROVISIONAL (the jaw's: cam4 4 px,
cam1 15 px) until a blind clip review with the nose traces beside it. Inputs: cam4 raw LP nose from v36 ``windows.npz``
(``pose`` part "nose"), cam1 LP nose from ``session_poses/<a>_<d>_cam1_full/windows_LP.csv``; likelihood < 0.6 -> NaN.

RESULT 10-21 -> NOT USED for any attempted / reacted class (Priya: "we won't use nose for the attempted/reacted
class"). Over the cue .. stop window the call is not specific: cam4 nose "moved" on hit 95 % (onset median 99 ms),
incomplete 87 %, jaw_only 80 %, no_try 56 % (853 ms) -- and on no_try trials 44-74 % even PRE-stroke, ~1 s after the cue
(sniffing / drift caught by a 3.5 s window). cam1 nose is hidden by the rod at centre spouts. Reactions are to come from
FaceRhythm components instead (docs/FACE_RHYTHM_PLAN.md). Kept as a descriptive table only.

Output `<v36 run>/imaging_encoding/qc/nose_onsets.csv` (NOT jaw_onsets.csv, which the encoding window watches): session,
animal, date, epoch, trial_id, position, cam4_nose_moved / cam1_nose_moved (1 / 0 / NaN), cam4_nose_onset_ms /
cam1_nose_onset_ms, nose_onset_ms + nose_onset_cam, nose_at_cue.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from wfield_local import pose_windows

from scripts.jaw_onsets import FPS, V36, _laid_out

NOSE_CUT = 0.6
OVERRIDES = {"cam4": {"degenerate_as_unknown": True},
             "cam1": {"degenerate_as_unknown": True, "deflection_threshold_min_px": 15.0}}


def _axis(v: np.ndarray, idx: pd.DataFrame, cam: str) -> pd.DataFrame:
    from wfield_local import jaw_kinematics as jk
    trace, trials, src = _laid_out(v, idx)
    O = jk.jaw_onsets(trace, FPS, trials, OVERRIDES[cam])
    td = idx.t_daq_ms.to_numpy(float)
    row = np.where(O.onset_idx >= 0, src[np.clip(O.onset_idx, 0, None)], -1)
    O["onset_ms"] = np.where(row >= 0, td[np.clip(row, 0, None)], np.nan)
    return O.set_index("trial_id")[["jaw_pass_qc", "onset_ms", "onset_frames"]]


def _camera(x: np.ndarray, y: np.ndarray, idx: pd.DataFrame, cam: str) -> pd.DataFrame:
    """moved = x OR y passes (NaN only when both cannot judge); onset = the earlier passing axis."""
    X, Y = _axis(x, idx, cam), _axis(y, idx, cam)
    mx, my = X.jaw_pass_qc.to_numpy(float), Y.jaw_pass_qc.to_numpy(float)
    moved = np.where((mx == 1) | (my == 1), 1.0, np.where(np.isnan(mx) & np.isnan(my), np.nan, 0.0))
    ox = np.where(mx == 1, X.onset_ms, np.inf)
    oy = np.where(my == 1, Y.onset_ms.reindex(X.index).to_numpy(float), np.inf)
    fx = np.where(mx == 1, X.onset_frames, np.inf)
    fy = np.where(my == 1, Y.onset_frames.reindex(X.index).to_numpy(float), np.inf)
    on = np.minimum(ox, oy)
    return pd.DataFrame({"moved": moved, "onset_ms": np.where(np.isfinite(on), on, np.nan),
                         "onset_frames": np.where(np.isfinite(on), np.where(ox <= oy, fx, fy), np.nan)},
                        index=X.index)


def session(job) -> pd.DataFrame:
    s, root_o, root_d = job
    from wfield_local import epochs
    a, d = s.split("_")
    sd = Path(root_o) / V36 / "sessions" / a / d
    w = np.load(sd / "windows.npz", allow_pickle=False)
    ix4 = pd.DataFrame({k: w[f"ix_{k}"] for k in ("trial_id", "position", "t_ms", "t_daq_ms", "stop_ms")})
    k = list(w["parts"]).index("nose")
    P4 = w["pose"][:, k, :].astype(float)
    P4[P4[:, 2] < NOSE_CUT, :2] = np.nan
    C4 = _camera(P4[:, 0], P4[:, 1], ix4, "cam4")
    out = pd.DataFrame({"trial_id": C4.index, "cam4_nose_moved": C4.moved.to_numpy(),
                        "cam4_nose_onset_ms": C4.onset_ms.to_numpy(), "_f4": C4.onset_frames.to_numpy()})
    out["position"] = out.trial_id.map(ix4.groupby("trial_id").position.first())
    out["cam1_nose_moved"], out["cam1_nose_onset_ms"], out["_f1"] = np.nan, np.nan, np.nan
    d1 = Path(root_d) / "session_poses" / f"{s}_cam1_full"
    if (d1 / "windows_LP.csv").exists():
        ix1 = pose_windows.index(d1)
        if "t_daq_ms" in ix1:
            P = pose_windows.lp(d1)
            lo = P[("nose", "likelihood")].to_numpy(float) < NOSE_CUT
            x1, y1 = P[("nose", "x")].to_numpy(float), P[("nose", "y")].to_numpy(float)
            x1[lo], y1[lo] = np.nan, np.nan
            C1 = _camera(x1, y1, ix1, "cam1")
            out["cam1_nose_moved"] = out.trial_id.map(C1.moved)
            out["cam1_nose_onset_ms"] = out.trial_id.map(C1.onset_ms)
            out["_f1"] = out.trial_id.map(C1.onset_frames)
    o4 = np.where(out.cam4_nose_moved == 1, out.cam4_nose_onset_ms, np.inf)
    o1 = np.where(out.cam1_nose_moved == 1, out.cam1_nose_onset_ms, np.inf)
    on = np.minimum(o4, o1)
    out["nose_onset_ms"] = np.where(np.isfinite(on), on, np.nan)
    out["nose_onset_cam"] = np.where(~np.isfinite(on), "", np.where(o4 <= o1, "cam4", "cam1"))
    f = np.where(out.nose_onset_cam == "cam4", out._f4, out._f1)
    out["nose_at_cue"] = np.where(np.isfinite(on), f == 0, False)
    out.insert(0, "session", s)
    out.insert(1, "animal", a)
    out.insert(2, "date", d)
    out.insert(3, "epoch", epochs.epoch_of(f"{a}_{d[4:]}"))
    cols = ["session", "animal", "date", "epoch", "trial_id", "position", "cam4_nose_moved", "cam4_nose_onset_ms",
            "cam1_nose_moved", "cam1_nose_onset_ms", "nose_onset_ms", "nose_onset_cam", "nose_at_cue"]
    return out[cols]


def main(argv=None) -> int:
    from wfield_local import analysis_kit as ak
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--sessions", nargs="*", default=None, metavar="ANIMAL:YYYYMMDD")
    ap.add_argument("--all", action="store_true", help="every v36 session, not just the curated cohort")
    a = ap.parse_args(argv)
    rv = PathResolver()
    ro, rd = Path(rv.root("orofacial")), Path(rv.root("dlc"))
    sess = sorted(f"{p.parent.name}_{p.name}" for p in (ro / V36 / "sessions").glob("*/*")
                  if (p / "windows.npz").exists())
    if a.sessions:
        sess = [s for s in sess if s.replace("_", ":", 1) in a.sessions]
    elif not a.all:
        sess = [s for s in sess if s in ak.curated_ids()]
    res, fails = ak.fan_sessions([(s, str(ro), str(rd)) for s in sess], session, jobs=a.jobs, key=lambda j: j[0])
    for s, e in fails:
        print(f"{s}: FAILED {e}")
    T = pd.concat([r for _, r in res], ignore_index=True)
    out = ro / V36 / "imaging_encoding" / "qc" / "nose_onsets.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    T.to_csv(out, index=False, float_format="%.2f")
    pd.set_option("display.width", 220)
    print(T.groupby(["animal", "epoch"]).agg(trials=("trial_id", "size"), cam4=("cam4_nose_moved", "mean"),
                                             cam1=("cam1_nose_moved", "mean"), onset_med=("nose_onset_ms", "median"),
                                             at_cue=("nose_at_cue", "mean")).round(3).to_string())
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
