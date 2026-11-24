"""Step 1: per-session FRAME tables for TIM, motion energy (ME), stillness and running; six-position cohort (PS92-PS95).
PROMPT_engagement_movement_both_cohorts.md items 4, 5, 10b, 11 (review only; widefield_pipeline / manuscript untouched).

WINDOWS (Priya 11-20 00:25), every stream, by DAQ time:
  PRE-CUE   = the ENL, max(strobe, cue - 2 s) .. cue (configs precue_post_s = 2.0; a shorter ENL is kept as is);
              per trial ``precue_lick`` = a camera lick (LP licks_all daq_on_s) inside it (excluded downstream).
  RESPONSE  = cue .. cue + 3.5 s (fixed).
  ITI       = spout DOCKED: dock + 0.2 s .. next trial_start - 0.05 s (`trial_windows.docked_spans`, the strict docked
              interval of the restdock05 / docked rest term); frames within the rest lick buffer [-0.5, +1.0] s of
              any camera lick flagged (``iti_lick``, excluded downstream). Owner = the trial that just ended.
Codes: per 1 = ITI, 2 = pre-cue, 3 = response, 0 = none; own = owner trial_id; tb = time bin of the period
  (pre-cue / response 100 ms from the cue; ITI 250 ms from the docked start); tmin = minutes from the first cue.

STREAMS
  kp  `arhmm_features` v3 (whole session, 25 Hz): DLC/LP 3-D tongue (tongue_ml/ap/ext, mouth = 0), cam4 jaw (x, y),
      nose minus mouth (x, y); per part frame-to-frame Euclidean change, > mean + 5 SD of the session's part -> NaN
      (Yin et al. STAR Methods); treadmill run_fwd / run_back (mm/s). No paw keypoints exist in this run.
  fr  FaceRhythm point mesh, EACH SESSION'S OWN tracking (`<bundle>/out/<key>_points.npz` ``disp``, 1,282 points):
      per-bin per-point speed (px/s), mean over all points / contralesional half / ipsilesional half (master midline;
      cam4 image LEFT = contralesional); FR true bin times + lossy bins -> NaN (`face_rhythm_timing`, as
      `factors.project.points_stream`); > mean + 5 SD -> NaN.
  pc  the session's `proj` ``points_pc``: the session's own centered mesh displacement projected on the 20 frozen
      point PCs of the master (`factors.project.points_stream`; ~95 % of point variance) -> FaceRhythm TIM traces.
  tp  TIM traces of `task_independent_movement.frame_table` (tongue / jaw x, y in the mouth frame from clean.npz, nose
      mouth-relative; 40 ms bins), placed on DAQ time as cue + t_ms.
Output: <cache>/<label>.npz; existing files are skipped (never overwritten).
Usage: WF_REPO=... python s1_extract_frames.py <cache_dir> [--jobs 4] [--labels ...] [--licks fused|cam4]
       [--jaw-nose 3d|cam4]

FOUR-CAMERA RULE (Priya 11-23; CC-B audit violation 6, 2026-11-23). Two switches, defaults = the four-camera build:
  --licks fused (default): every lick exclusion (pre-cue trials with a lick; ITI frames within [-0.5, +1.0] s of a
      lick) reads the fused four-camera lick ONSETS, `session_licks.session_onsets_s(label)` (all contexts, daq_on_s,
      cam1-only events left out as everywhere outside the trial-class path). ``--licks cam4`` = the cam4 LP
      licks_all.csv daq_on_s, as the frames_v3 cache (11-22). The cam4-list ITI buffer is also stored as
      ``kp_nearlick_cam4`` / ``fr_nearlick_cam4`` and the cam4 window flag as ``win_lick_cam4`` (diagnostics only).
  --jaw-nose 3d (default): ``kp_d_jaw`` / ``kp_d_nose`` = frame-to-frame 3-D Euclidean change of the four-camera jaw /
      nose (`pose3d` adopted run frames, jaw_ml/ap/dv, nose_ml/ap/dv, mm; triangulated from 3-4 views on > 99 % of
      frames), placed on DAQ time through the cam4 pose index (trial_id, t_ms -> daq_s), averaged in the same 25 Hz
      bins as the features, NaN runs <= 0.2 s interpolated (`arhmm_features._interp_short`), then the same > mean + 5 SD
      drop. ``--jaw-nose cam4`` = the cam4 2-D jaw / nose columns of the features file (frames_v3). The 2-D versions
      are always stored too as ``kp2d_d_jaw`` / ``kp2d_d_nose`` (s4 reads them with S4_KP_JAWNOSE=2d).
  The tongue is the features file's 3-D tongue (four-camera; ARHMM_OUT selects features_v3 or the cam3-on features_v4).
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.expanduser(os.environ.get("WF_REPO", "~/Github/widefield_pipeline")))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BUNDLE = Path(r"M:\MICROSCOPE\Priya\DeepLabCut\Widefield\o2\fr_cam4_20261021")
FDIR = BUNDLE / "factors" / "master_pre_v1"
KP_PARTS = {"tongue": ("tongue_ml", "tongue_ap", "tongue_ext"), "jaw": ("jaw_x", "jaw_y"), "nose": ("nose_x", "nose_y")}
PRECUE_S, RESP_S = 2.0, 3.5
LICK_BUF = (0.5, 1.0)
TB = {2: 0.1, 3: 0.1, 1: 0.25}
CHUNK = 20000


def drop5(x):
    x = np.asarray(x, np.float32).copy()
    m, s = np.nanmean(x), np.nanstd(x)
    x[x > m + 5 * s] = np.nan
    return x


def windows(label, TR, licks):
    """Per trial and docked span: (code, owner, t0, t1, anchor) rows."""
    from wfield_local import trial_windows as tw
    animal, date = label.split("_")
    cue = TR.cue_daq.to_numpy(float)
    strobe = cue + TR.strobe_ms.to_numpy(float) / 1000.0
    rows = []
    for k in range(len(TR)):
        a = np.nanmax([cue[k] - PRECUE_S, strobe[k]]) if np.isfinite(strobe[k]) else cue[k] - PRECUE_S
        rows.append((2, TR.trial_id.iat[k], a, cue[k], cue[k]))
        rows.append((3, TR.trial_id.iat[k], cue[k], cue[k] + RESP_S, cue[k]))
    sp = tw.docked_spans(animal, date)
    n_dock = 0
    if sp is not None:
        for a0, a1 in zip(*sp):
            k = np.searchsorted(cue, a0) - 1
            if k >= 0:
                rows.append((1, TR.trial_id.iat[k], a0, a1, a0))
                n_dock += 1
    W = pd.DataFrame(rows, columns=["code", "own", "t0", "t1", "anchor"])
    W["lick"] = lick_flag(W, licks)
    return W, n_dock


def lick_flag(W, licks):
    """Per window row: any lick onset in [t0, t1)."""
    return np.array([np.any((licks >= r.t0) & (licks < r.t1)) for r in W.itertuples()]) if len(licks) else np.zeros(len(W), bool)


def label_frames(t, W, licks, first_cue):
    n = len(t)
    per = np.zeros(n, np.int8); own = np.full(n, -1, np.int32); tb = np.full(n, -1, np.int16)
    for r in W.itertuples():
        i0, i1 = np.searchsorted(t, r.t0), np.searchsorted(t, r.t1)
        per[i0:i1], own[i0:i1] = r.code, r.own
        tb[i0:i1] = np.floor((t[i0:i1] - r.anchor) / TB[r.code]).astype(np.int16)
    # lick buffer inside the ITI
    near = np.zeros(n, bool)
    for lk in licks:
        near[np.searchsorted(t, lk - LICK_BUF[0]):np.searchsorted(t, lk + LICK_BUF[1])] = True
    return {"per": per, "own": own, "tb": tb, "nearlick": near, "tmin": ((t - first_cue) / 60.0).astype(np.float32)}


def kp3d_jaw_nose(label, t):
    """3-D jaw and nose (mm) of the adopted four-camera run, binned onto the feature bin centres ``t`` (25 Hz)."""
    from wfield_local import arhmm_features as af
    from wfield_local import combined_encoding as ce
    from wfield_local import orofacial_cohort as oc
    from wfield_local import orofacial_outputs as oo
    from wfield_local.paths import PathResolver
    rv = PathResolver()
    animal, date = label.split("_")
    f3 = ce.pose3d_frames_path(rv, animal, date)
    ix = oo.load_session(rv, oc.RUN, animal, date)["index"][["trial_id", "t_ms", "daq_s"]]
    cols = ["jaw_ml", "jaw_ap", "jaw_dv", "nose_ml", "nose_ap", "nose_dv"]
    J = ix.merge(pd.read_csv(f3, usecols=["trial_id", "t_ms", *cols]).drop_duplicates(["trial_id", "t_ms"]),
                 on=["trial_id", "t_ms"], how="left")
    td = J.daq_s.to_numpy(float)
    edges = np.r_[t - af.BIN_S / 2, t[-1] + af.BIN_S / 2]
    ok = np.isfinite(td)
    nmax = int(round(af.MAX_INTERP_S / af.BIN_S))
    out, cov = {}, {}
    for part in ("jaw", "nose"):
        P = np.column_stack([af._interp_short(af._bin(td[ok], J[f"{part}_{a}"].to_numpy(float)[ok], edges), nmax)
                             for a in ("ml", "ap", "dv")])
        cov[part] = float(np.isfinite(P).all(1).mean())
        out[part] = drop5(np.r_[np.nan, np.sqrt((np.diff(P, axis=0) ** 2).sum(1))])
    return out, cov, str(f3)


def job(item):
    label, cache = item
    out = Path(cache) / f"{label}.npz"
    if out.exists():
        return str(out)
    from wfield_local import arhmm_features as af
    from wfield_local import trial_classes as tc
    from wfield_local import face_rhythm_timing as frt
    from wfield_local import orofacial_outputs as oo
    from wfield_local import orofacial_cohort as oc
    from wfield_local import task_independent_movement as tim
    from wfield_local.paths import PathResolver
    from wfield_local.face_rhythm import face as frface
    from wfield_local.face_rhythm.factors.files import npz_memmap, npz_small
    animal, date = label.split("_")
    TR = tc.session_trials(label)
    TR = TR.dropna(subset=["cue_daq"]).sort_values("trial_id").reset_index(drop=True)
    L = pd.read_csv(Path(oo.session_dir(PathResolver(), oc.RUN, animal, date)) / "licks_all.csv", usecols=["daq_on_s"])
    licks_cam4 = np.sort(L.daq_on_s.dropna().to_numpy(float))
    lick_src = os.environ.get("S1_LICKS", "fused")
    if lick_src == "fused":
        from wfield_local import session_licks as SL
        licks = SL.session_onsets_s(label)
        lick_tag = f"fused:{SL.fused_run()}"
    else:
        licks, lick_tag = licks_cam4, "cam4:licks_all.csv"
    W, n_dock = windows(label, TR, licks)
    fc = float(np.nanmin(TR.cue_daq))
    res = {"win_code": W.code.to_numpy(np.int8), "win_own": W.own.to_numpy(np.int32), "win_t0": W.t0.to_numpy(),
           "win_t1": W.t1.to_numpy(), "win_lick": W.lick.to_numpy(bool), "n_dock": n_dock, "first_cue": fc,
           "win_lick_cam4": lick_flag(W, licks_cam4), "lick_source": lick_tag, "n_licks": len(licks),
           "n_licks_cam4": len(licks_cam4)}
    # ---- kp
    F = np.load(Path(af.out_dir()) / "sessions" / f"{label}.npz", allow_pickle=True)
    cols = [str(c) for c in F["features"]]
    X, t = F["X"].astype(np.float32), F["t_daq_s"].astype(float)
    res["kp_t"] = t
    for p, cs in KP_PARTS.items():
        P = X[:, [cols.index(c) for c in cs]]
        res[f"kp_d_{p}"] = drop5(np.r_[np.nan, np.sqrt((np.diff(P, axis=0) ** 2).sum(1))])
    res["kp2d_d_jaw"], res["kp2d_d_nose"] = res["kp_d_jaw"], res["kp_d_nose"]
    res["features_file"] = str(Path(af.out_dir()) / "sessions" / f"{label}.npz")
    if os.environ.get("S1_JAWNOSE", "3d") == "3d":
        d3, cov3, f3 = kp3d_jaw_nose(label, t)
        res["kp_d_jaw"], res["kp_d_nose"] = d3["jaw"], d3["nose"]
        res.update(kp_source="tongue 3-D (features); jaw, nose 3-D four-camera", kp3d_frames=f3,
                   kp3d_cov_jaw=cov3["jaw"], kp3d_cov_nose=cov3["nose"])
    else:
        res["kp_source"] = "tongue 3-D (features); jaw, nose cam4 2-D (features)"
    res["run_fwd"], res["run_back"] = X[:, cols.index("run_fwd")], X[:, cols.index("run_back")]
    res.update({f"kp_{k}": v for k, v in label_frames(t, W, licks, fc).items()})
    res["kp_nearlick_cam4"] = label_frames(t, W, licks_cam4, fc)["nearlick"]
    # ---- fr (own tracking)
    key = f"{label}_cam4"
    pf = BUNDLE / "out" / f"{key}_points.npz"
    if pf.exists():
        comps, contra = frface.master_points(FDIR)
        D = npz_memmap(pf, "disp")
        n, P = D.shape[0], D.shape[1]
        tm = frt.load(BUNDLE, key)
        t25 = tm["t25"][:n] if tm is not None else npz_small(pf, ("t_daq_s",))["t_daq_s"][:n]
        bad = frt.lossy(tm["drop25"][:n]) if tm is not None else np.zeros(n, bool)
        inv_dt = 1.0 / np.median(np.diff(t25))
        sa, sc, si = (np.full(n, np.nan, np.float32) for _ in range(3))
        prev = None
        for c0 in range(0, n, CHUNK):
            x = np.asarray(D[c0:c0 + CHUNK], np.float32)
            full = x if prev is None else np.concatenate([prev[None], x])
            s_ = np.sqrt((np.diff(full, axis=0) ** 2).sum(-1)) * inv_dt
            if prev is None:
                s_ = np.vstack([np.full((1, P), np.nan, np.float32), s_])
            sl = slice(c0, c0 + len(x))
            sa[sl], sc[sl], si[sl] = s_.mean(1), s_[:, contra].mean(1), s_[:, ~contra].mean(1)
            prev = x[-1]
        for a in (sa, sc, si):
            a[bad[:n]] = np.nan
        res.update(fr_t=t25, fr_all=drop5(sa), fr_contra=drop5(sc), fr_ipsi=drop5(si))
        res.update({f"fr_{k}": v for k, v in label_frames(t25, W, licks, fc).items()})
        res["fr_nearlick_cam4"] = label_frames(t25, W, licks_cam4, fc)["nearlick"]
    no_tim = bool(os.environ.get("S1_NO_TIM"))
    pj = FDIR / "proj" / f"{key}.npz"
    if pj.exists() and not no_tim:
        with np.load(pj, allow_pickle=True) as z:
            pc, tpc = z["points_pc"].astype(np.float32), z["points_t_daq_s"].astype(float)
        res.update(pc_t=tpc, pc=pc.astype(np.float16))
        res.update({f"pc_{k}": v for k, v in label_frames(tpc, W, licks, fc).items()})
    # ---- tp (TIM traces, mouth frame)
    sd = None if no_tim else tim.session_dir(label)
    if sd is not None and (sd / "windows.npz").exists() and (sd / "clean.npz").exists():
        G = tim.frame_table(sd)
        cmap = dict(zip(TR.trial_id, TR.cue_daq))
        tt = G.trial_id.map(cmap).to_numpy(float) + G.t_ms.to_numpy(float) / 1000.0
        ok = np.isfinite(tt)
        G, tt = G[ok], tt[ok]
        o = np.argsort(tt, kind="stable")
        res["tp_t"] = tt[o]
        res["tp"] = G[list(tim.TRACES)].to_numpy(np.float32)[o]
        res.update({f"tp_{k}": v for k, v in label_frames(tt[o], W, licks, fc).items()})
    tmp = out.with_name(out.stem + f".tmp{os.getpid()}.npz")
    np.savez(tmp, **res)
    os.replace(tmp, out)
    return str(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cache")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--labels", nargs="*")
    ap.add_argument("--no-tim", action="store_true",
                    help="skip the TIM-only streams (pc = FaceRhythm point PCs, tp = TIM traces; read only by s2)")
    ap.add_argument("--licks", choices=("fused", "cam4"), default="fused",
                    help="lick list for the exclusions: fused four-camera onsets (default) or cam4 licks_all (frames_v3)")
    ap.add_argument("--jaw-nose", choices=("3d", "cam4"), default="3d",
                    help="jaw / nose keypoint ME source: four-camera 3-D (default) or cam4 2-D (frames_v3)")
    a = ap.parse_args()
    os.environ["S1_LICKS"], os.environ["S1_JAWNOSE"] = a.licks, a.jaw_nose   # inherited by the pool workers
    if a.no_tim:
        os.environ["S1_NO_TIM"] = "1"           # inherited by the pool workers
    from wfield_local import behavior_states as B
    import trial_scaffold
    Path(a.cache).mkdir(parents=True, exist_ok=True)
    pool_labels = (B.load_collected()["trials"].label.unique() if trial_scaffold.collected_available()
                   else trial_scaffold.glmhmm_labels())
    labels = a.labels or sorted(lab for lab in pool_labels if B.session_epoch(lab))
    from multiprocessing import Pool
    import time
    t0 = time.time()
    with Pool(a.jobs) as pool:
        for i, r in enumerate(pool.imap_unordered(job, [(lab, a.cache) for lab in labels])):
            print(f"{i + 1}/{len(labels)} {time.time() - t0:.0f}s {r}", flush=True)


if __name__ == "__main__":
    main()
