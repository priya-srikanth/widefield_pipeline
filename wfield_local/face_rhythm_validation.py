"""FaceRhythm factor IDENTITY checks against independent behavior (Priya 10-22: "compare these to lick timing and treadmill
traces to see if we can refine" the clip-based names). Computation only; figures + CLI: scripts/face_rhythm_validate.py.

Per projected session (``proj/<key>.npz`` of a factor set; ABSOLUTE expression = weights x FR normalizer D):

``lick_table``       every lick of the session (`session_licks`: the fused four-camera train; DAQ onset ``daq_on_s``, executed
                     angle ``peak_angle_deg``, + = the animal's LEFT) with its direction class (left > +ANGLE_DEG,
                     right < -ANGLE_DEG, center between), the trial's spout ``position``, its NUMBER within its bout
                     (``n_in_bout``, 1 = first; v36 ``bout_idx``), time from the cue of the engaged trial it falls in,
                     and the instantaneous lick FREQUENCY (1 / interval to the next lick of the same trial, if
                     < MAX_ILI_S).
``lick_triggered``   each factor's expression around every lick (LAGS, from the 25 Hz trial grid; licks whose window
                     lies inside cue -2..+4 s of a projected engaged trial). -> (n_licks, n_lags, R).
``bin_table``        per 40 ms bin of the trial grid: expression of every factor + lick rates by direction (licks / s,
                     Gaussian SMOOTH_S), mean lick frequency, time from cue, treadmill speed. The correlation table is
                     computed from it.
``running_triggered`` expression around treadmill RUNNING-BOUT onsets (behavior_events, the canonical bouts) on the
                     whole-session series (25 Hz in trial windows, 5 Hz elsewhere, `face_rhythm_outputs.combined_series`)
                     + the correlation with speed OUTSIDE lick periods (> LICK_FREE_S from any LP lick).
``lick_epoch_session`` per session and executed DIRECTION: licks per engaged trial in the response window
                     (0..RESP_S after the cue) and each factor's expression AT the lick (mean over 0..PER_LICK_S after
                     onset) -- splits a post-stroke change of a factor into FEWER licks vs DIFFERENT licks
                     (`scripts/face_rhythm_lick_epochs.py`).
Treadmill speed (`behavior_events.session_speed`, the one definition) is CACHED per session at 25 Hz
(`result_cache`, keyed on the DAQ .h5): the 5 kHz channel is read once.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from wfield_local.lp_licks import input_run as _input_run  # noqa: E402
V36 = _input_run()  # 11-22: the input / adopted cam4 run (lp_licks.input_run; was the literal v36 run, plan item 3)
ANGLE_DEG = 12.0                         # |executed angle| above this = a left / right lick (cam4 image angle)
MAX_ILI_S = 0.35                         # next lick within this -> an instantaneous frequency
LAGS = np.arange(-1.0, 1.0 + 1e-9, 0.04)
SMOOTH_S = 0.1                           # Gaussian sigma for lick rates on the 40 ms grid
LICK_FREE_S = 1.0
RUN_LAGS = np.arange(-3.0, 5.0 + 1e-9, 0.2)
SPEED_VERSION = "fr-validate-speed-v1"
RESP_S = 4.0                             # response window: cue .. +4 s (as `face_rhythm_readouts.WINDOWS`)
PER_LICK_S = 0.2                         # expression AT a lick: mean over 0..+0.2 s after onset
DIRECTIONS = ("left", "centre", "right")


LICK_COLS = ["trial_id", "position", "bout_idx", "daq_on_s", "peak_angle_deg"]
#: DIRECTION SOURCE (11-23, four-camera rule; CC-B audit violation 5). Env ``FRV_DIRECTION``: unset / "cam4" = the cam4
#: image angle ``peak_angle_deg`` (as before; licks cam4 missed -> "unknown"); "3d" = the four-camera 3-D lick angle of
#: the six-position kinematics, atan2(side_ml, hypot(out_ap, down_dv)) (= the fused kinematics' ``midline`` column, the
#: Fig 4D aim angle), signed + = CONTRALESIONAL (`arhmm_features.contra_sign`), taken only where the lick has >= 2
#: camera views (the kinematics rule) and joined by LICK KEY to the fused lick table; other licks -> "unknown". The same
#: +-ANGLE_DEG rule classifies it (right / contralesional > +12 deg, left / ipsilesional < -12 deg); internally it is
#: stored as ``peak_angle_deg`` = -angle so the cam4 sign convention (+ = the animal's left) and the class code are shared.
DIRECTION_ENV = "FRV_DIRECTION"
_KIN3D: dict = {}


def _kin3d_angles(rv, label: str) -> pd.Series:
    """3-D lick angle (deg, + = contralesional) by lick_key for ``label`` (n_views >= 2), from the adopted run's
    ``kinematics_fused/licks.csv.gz`` (all contexts; memoised per process)."""
    from wfield_local import pose3d
    from wfield_local.arhmm_features import contra_sign
    f = pose3d.pose3d_dir(rv) / "kinematics_fused" / "licks.csv.gz"
    if str(f) not in _KIN3D:
        K = pd.read_csv(f, usecols=["session", "lick_key", "n_views", "side_ml", "out_ap", "down_dv"])
        K = K[(K.n_views >= 2) & K[["side_ml", "out_ap", "down_dv"]].notna().all(1)]
        K = K.assign(ang=np.degrees(np.arctan2(K.side_ml, np.hypot(K.out_ap, K.down_dv))))
        _KIN3D.clear()
        _KIN3D[str(f)] = {s: g.set_index("lick_key").ang for s, g in K.groupby("session")}
    a = _KIN3D[str(f)].get(label, pd.Series(dtype=float))
    return a * contra_sign(label.split("_")[0])


def _session_lick_rows(rv, animal: str, date: str) -> pd.DataFrame:
    """THE session lick set (`session_licks.session_licks`: the fused four-camera train since 11-22, cam4 licks_all.csv
    with fusion off) with the cam4 columns this module needs. On the fused train, ``peak_angle_deg`` (cam4 image angle,
    the direction class) and ``bout_idx`` come from cam4 licks_all.csv joined by LICK KEY (a fused lick cam4 saw keeps
    cam4's peak); licks cam4 did not see get NaN -> direction "unknown", no bout number (11-23, four-camera re-run)."""
    from wfield_local import session_licks as sl
    A = sl.session_licks(f"{animal}_{date}", rv)
    if (A.source == "cam4").all():
        return A[LICK_COLS]
    f = Path(rv.root("orofacial")) / V36 / "sessions" / animal / date / "licks_all.csv"
    C = pd.read_csv(f, usecols=["trial_id", "t_ms", "bout_idx", "peak_angle_deg"])
    C = C.assign(lick_key=sl.lick_key(C.trial_id, C.t_ms)).drop_duplicates("lick_key").set_index("lick_key")
    for c in ("bout_idx", "peak_angle_deg"):
        A[c] = C[c].reindex(A.lick_key.to_numpy()).to_numpy()
    if os.environ.get(DIRECTION_ENV, "cam4") == "3d":
        A["peak_angle_deg"] = -_kin3d_angles(rv, f"{animal}_{date}").reindex(A.lick_key.to_numpy()).to_numpy()
    return A[LICK_COLS]


def lick_table(rv, animal: str, date: str, cues: np.ndarray) -> pd.DataFrame:
    """LP licks of the whole session with direction class, time from the nearest preceding engaged cue (NaN when the
    lick is outside cue -2..+4 s of any of ``cues``) and instantaneous frequency."""
    L = _session_lick_rows(rv, animal, date).dropna(subset=["daq_on_s"])
    L = L.sort_values("daq_on_s").reset_index(drop=True)
    # lick NUMBER within its bout (v36 bout_idx within the trial; 1 = the bout's first lick); NaN without a bout
    g = L.dropna(subset=["bout_idx"]).groupby(["trial_id", "bout_idx"]).cumcount() + 1
    L["n_in_bout"] = np.nan
    L.loc[g.index, "n_in_bout"] = g
    a = L.peak_angle_deg.to_numpy(float)
    L["direction"] = np.where(a > ANGLE_DEG, "left", np.where(a < -ANGLE_DEG, "right", "centre"))
    L.loc[~np.isfinite(a), "direction"] = "unknown"
    c = np.sort(np.asarray(cues, float))
    t = L.daq_on_s.to_numpy(float)
    k = np.searchsorted(c, t + 2.0, side="right") - 1                    # latest cue with cue - 2 <= t
    rel = np.where(k >= 0, t - c[np.clip(k, 0, None)], np.nan)
    L["t_from_cue"] = np.where((rel >= -2.0) & (rel < 4.0), rel, np.nan)
    L["cue_idx"] = np.where(np.isfinite(L.t_from_cue), k, -1)
    nxt = np.r_[t[1:] - t[:-1], np.inf]
    same = np.r_[L.trial_id.to_numpy()[1:] == L.trial_id.to_numpy()[:-1], False]
    L["freq_hz"] = np.where(same & (nxt < MAX_ILI_S), 1.0 / np.where(nxt > 0, nxt, np.inf), np.nan)
    return L


def _trial_series(z, absolute: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(cues (n_tr,), t_s (151,) from cue, W (n_tr, 151, R)) of a proj file, cue-sorted."""
    W = z["trials"].astype(float) * (z["trials_D"][..., None] if absolute else 1.0)
    o = np.argsort(z["cue_daq_s"])
    return z["cue_daq_s"][o].astype(float), z["t_ms"].astype(float) / 1000.0, W[o]


def lick_triggered(z, L: pd.DataFrame, absolute: bool = True) -> tuple[np.ndarray, pd.DataFrame]:
    """(X (n, len(LAGS), R), the licks used) -- expression around licks whose whole window is inside a projected trial."""
    cues, ts, W = _trial_series(z, absolute)
    ci = np.searchsorted(cues, L.daq_on_s.to_numpy(float) + 2.0, side="right") - 1
    rel = L.daq_on_s.to_numpy(float) - cues[np.clip(ci, 0, None)]
    ok = (ci >= 0) & (rel + LAGS[0] >= ts[0]) & (rel + LAGS[-1] <= ts[-1])
    out = np.empty((ok.sum(), len(LAGS), W.shape[-1]))
    for n, (i, r) in enumerate(zip(ci[ok], rel[ok])):
        for k in range(W.shape[-1]):
            out[n, :, k] = np.interp(r + LAGS, ts, W[i, :, k])
    return out, L[ok].assign(t_rel=rel[ok]).reset_index(drop=True)


def _gauss_rate(times: np.ndarray, grid: np.ndarray, sigma: float) -> np.ndarray:
    """Events / s on ``grid`` (Gaussian kernel, sigma s)."""
    if len(times) == 0:
        return np.zeros(len(grid))
    d = grid[:, None] - times[None, :]
    m = np.abs(d) < 4 * sigma
    return (np.exp(-0.5 * (d / sigma) ** 2) * m).sum(1) / (sigma * np.sqrt(2 * np.pi))


def bin_table(z, L: pd.DataFrame, speed: tuple[np.ndarray, np.ndarray] | None, absolute: bool = True) -> pd.DataFrame:
    """One row per 40 ms bin of every projected trial: f0..f<R-1>, lick rates by direction, lick frequency, time from
    cue, speed."""
    cues, ts, W = _trial_series(z, absolute)
    R = W.shape[-1]
    lt = L.daq_on_s.to_numpy(float)
    rows = []
    for i, c in enumerate(cues):
        g = c + ts
        near = (lt > g[0] - 1) & (lt < g[-1] + 1)
        Li = L[near]
        d = {f"f{k}": W[i, :, k] for k in range(R)}
        d["t_from_cue"] = ts
        d["trial"] = np.full(len(ts), i)
        for dname in ("left", "centre", "right"):
            d[f"rate_{dname}"] = _gauss_rate(Li.daq_on_s[Li.direction == dname].to_numpy(float), g, SMOOTH_S)
        d["rate_all"] = _gauss_rate(Li.daq_on_s.to_numpy(float), g, SMOOTH_S)
        fq = Li.dropna(subset=["freq_hz"])
        d["lick_freq_hz"] = np.interp(g, fq.daq_on_s, fq.freq_hz, left=np.nan, right=np.nan) if len(fq) > 1 else \
            np.full(len(g), np.nan)
        if len(fq) > 1:                                                   # only where a bout is ongoing
            d["lick_freq_hz"][d["rate_all"] < 1.0] = np.nan
        d["speed_mm_s"] = np.interp(g, *speed) if speed is not None else np.full(len(g), np.nan)
        rows.append(pd.DataFrame(d))
    return pd.concat(rows, ignore_index=True)


def session_speed_25hz(rv, animal: str, date: str) -> tuple[np.ndarray, np.ndarray] | None:
    """(DAQ s, treadmill speed mm/s) at 25 Hz, cached (`result_cache`, keyed on the DAQ .h5)."""
    from wfield_local import behavior_events as be
    from wfield_local import result_cache
    from wfield_local.spout_behavior import _daq_h5_for
    h5 = _daq_h5_for(rv, animal, date)
    if h5 is None:
        return None
    cache = Path(rv.root("dlc")) / "face_rhythm" / "cam4" / "validation" / "_cache" / "speed"
    return result_cache.cached(cache, f"{animal}_{date}", [h5], SPEED_VERSION,
                               lambda: be.session_speed(Path(h5), step_s=0.04))


def running_triggered(rv, animal: str, date: str, files: list[Path], L: pd.DataFrame,
                      speed: tuple[np.ndarray, np.ndarray] | None, absolute: bool = True) -> dict:
    """Expression around running-bout onsets (whole-session series) and its correlation with speed outside licks."""
    from wfield_local import behavior_events as be
    from wfield_local import face_rhythm_outputs as fro
    t, X = fro.combined_series(files, absolute=absolute)
    ev = be.get_or_compute(rv, animal, date)
    on = np.asarray(ev["running_starts"], float) / float(ev["fs"]) if ev is not None else np.empty(0)
    on = on[(on + RUN_LAGS[0] > t[0]) & (on + RUN_LAGS[-1] < t[-1])]
    TA = np.stack([np.stack([np.interp(o + RUN_LAGS, t, X[:, k]) for k in range(X.shape[1])], 1) for o in on]) \
        if len(on) else np.empty((0, len(RUN_LAGS), X.shape[1]))
    r_speed = np.full(X.shape[1], np.nan)
    if speed is not None:
        lt = L.daq_on_s.to_numpy(float)
        j = np.searchsorted(lt, t)
        dmin = np.minimum(np.abs(t - lt[np.clip(j - 1, 0, len(lt) - 1)]), np.abs(lt[np.clip(j, 0, len(lt) - 1)] - t)) \
            if len(lt) else np.full(len(t), np.inf)
        sp = np.interp(t, *speed)
        m = (dmin > LICK_FREE_S) & np.isfinite(sp) & np.isfinite(X).all(1)
        if m.sum() > 50:
            r_speed = np.array([np.corrcoef(X[m, k], sp[m])[0, 1] for k in range(X.shape[1])])
    return {"run_onsets": len(on), "run_ta": TA, "r_speed_lickfree": r_speed}


def session(job) -> dict:
    """Everything for one projected session (module-level for `analysis_kit.fan_sessions`)."""
    fdir, animal, date = job
    from wfield_local import face_rhythm_outputs as fro
    from wfield_local.paths import PathResolver
    rv = PathResolver()
    files = fro.session_keys(Path(fdir), animal, date)
    zs = [np.load(f) for f in files]
    cues = np.concatenate([z["cue_daq_s"] for z in zs])
    L = lick_table(rv, animal, date, cues)
    speed = session_speed_25hz(rv, animal, date)
    lta, used, bins = [], [], []
    for z in zs:
        x, u = lick_triggered(z, L)
        lta.append(x)
        used.append(u)
        bins.append(bin_table(z, L, speed))
    run = running_triggered(rv, animal, date, files, L, speed)
    return {"animal": animal, "date": date, "epoch": str(zs[0]["epoch"]), "lta": np.concatenate(lta),
            "licks": pd.concat(used, ignore_index=True), "bins": pd.concat(bins, ignore_index=True), **run}


def lick_epoch_summary(zs: list, L: pd.DataFrame, absolute: bool = True) -> dict:
    """Per direction: licks per engaged trial in the response window, per-lick expression (R,) and the mean
    lick-triggered curve (len(LAGS), R); plus every factor's mean response-window expression per trial (R,).
    ``zs``: the session's proj files (loaded), ``L``: its `lick_table`."""
    n_tr = sum(len(z["cue_daq_s"]) for z in zs)
    resp = []
    X, used = [], []
    for z in zs:
        _, ts, W = _trial_series(z, absolute)
        resp.append(np.nanmean(W[:, (ts >= 0) & (ts < RESP_S)], axis=1))          # (n_tr, R)
        x, u = lick_triggered(z, L, absolute)
        X.append(x)
        used.append(u)
    X = np.concatenate(X)
    U = pd.concat(used, ignore_index=True)
    R = X.shape[-1] if len(X) else resp[0].shape[-1]
    at = (LAGS >= 0) & (LAGS <= PER_LICK_S + 1e-9)
    inresp = (U.t_rel >= 0) & (U.t_rel < RESP_S) if len(U) else np.zeros(0, bool)
    out = {"n_trials": n_tr, "resp_expr": np.nanmean(np.concatenate(resp), axis=0)}
    for d in DIRECTIONS:
        m = np.asarray(inresp & (U.direction == d)) if len(U) else np.zeros(0, bool)
        out[f"n_{d}"] = int(m.sum())
        out[f"licks_per_trial_{d}"] = m.sum() / max(n_tr, 1)
        out[f"per_lick_{d}"] = X[m][:, at].mean(axis=(0, 1)) if m.sum() else np.full(R, np.nan)
        out[f"lta_{d}"] = X[m].mean(0) if m.sum() else np.full((len(LAGS), R), np.nan)
    return out


def lick_epoch_session(job) -> dict:
    """`lick_epoch_summary` for one projected session (module-level for `analysis_kit.fan_sessions`)."""
    fdir, animal, date = job
    from wfield_local import face_rhythm_outputs as fro
    from wfield_local.paths import PathResolver
    rv = PathResolver()
    zs = [np.load(f) for f in fro.session_keys(Path(fdir), animal, date)]
    L = lick_table(rv, animal, date, np.concatenate([z["cue_daq_s"] for z in zs]))
    return {"animal": animal, "date": date, "epoch": str(zs[0]["epoch"]), **lick_epoch_summary(zs, L)}
