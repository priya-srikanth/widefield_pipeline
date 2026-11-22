"""Numbers for the six-position Methods on the FOUR-CAMERA production run (Priya 11-22: "do the measurements"); fills the
"not verified" items of vls_stroke_paper/NOTE_CC_4cam_methods_2026-11-22.md.

    python scripts/pose3d_methods_numbers.py reproj  [--jobs 3]   # PnP spout reprojection, all four cameras, every session
    python scripts/pose3d_methods_numbers.py fill    [--jobs 3]   # one-camera fill held-out error (8 sessions)
    python scripts/pose3d_methods_numbers.py lateral               # cam4 2-D lateral vs 3-D lateral, per lick
    python scripts/pose3d_methods_numbers.py ili                   # pre-stroke within-bout ILI per animal
    python scripts/pose3d_methods_numbers.py daq     [--jobs 3]   # fused post-cue licks vs DAQ contacts

Read-only on the production outputs (`pose3d_4cam_3net_20261121_sync60`); writes to its `qc_methods_numbers_20261122/`.

reproj: the four-camera run poses every camera per session from its six parked-spout medians (`pose3d_side.fuse_session`:
cam1 / cam4 medians as `pose3d.session_3d` computes them, June cam4 rescaled; side medians from the whole-session side
model) with `pose3d_side.pose_all` (= `pose3d.pnp_group`); this recomputes exactly that and reports, per camera, the
mean / rms / max reprojection over the six spouts (image px; sides also in 600-px units, the side gate's scale).
fill: the protocol of the two-camera numbers in `pose3d.FILL_GAP_MS` / `session_3d` (`fill_interp`): on frames with >= 2
used views, hold out runs of 8 / 16 / 24 ms (tongue) or 24 / 200 ms (jaw) bracketed by multi-view frames, keep ONE of the
run's used views, fill with `pose3d_side.fill_n` (the production filler) and compare with the triangulated point (mm).
lateral: four-camera post-cue licks with 3-D support that cam4 saw, joined to cam4's own lick (lick_key = trial + cam4 peak);
Pearson r of cam4 ``lr_px`` (+ = mouse left) vs -``side_ml`` (3-D, mm), per animal (the D11 check, `scripts/pose3d_crosscheck`).
ili: consecutive 3-D extension peaks within a trial, <= 300 ms apart, August pre-stroke sessions, median per animal.
daq: per session, fused post-cue licks (cam1-only excluded, the session table) vs DAQ contact onsets (behavior events) in
the same windows (cue + 70 ms .. trial stop): fraction of DAQ contacts with a fused lick peak within +-X ms, and fraction
of fused licks with a DAQ contact within +-X ms (X = 50 and 100 ms).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

P3D = Path("M:/MICROSCOPE/Priya/DeepLabCut/Widefield/session_poses/pose3d_4cam_3net_20261121_sync60")
OUT = P3D / "qc_methods_numbers_20261122"
SIDE_BUNDLE = "lp_side_r2b_prod_20261118"
KIN = P3D / "kinematics_fused_postcue" / "licks.csv.gz"
FILL_SESSIONS = ["PS92:20260807", "PS92:20260818", "PS93:20260812", "PS93:20260821",
                 "PS94:20260813", "PS94:20260825", "PS95:20260812", "PS95:20260821"]
GAPS = {"tongue": (8.0, 16.0, 24.0), "jaw": (24.0, 200.0)}
FRAME_MS = 4.0


def _sessions() -> list[str]:
    s = pd.read_csv(next(P3D.glob("summary_production*.csv")), usecols=["session"]).session
    return sorted(f"{x.split('_')[0]}:{x.split('_')[1]}" for x in s)


# ------------------------------------------------------------------------------------------------------------ reproj
def _reproj_worker(item: str) -> list[dict]:
    os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
    from scripts.apparatus_mouth import _rig
    from scripts.pose3d import PROD, _cgroup, _june_factor
    from wfield_local import o2_inference as oi
    from wfield_local import pose3d as P3
    from wfield_local import pose3d_side as PS
    from wfield_local import spout_frame as SF
    from wfield_local.paths import PathResolver
    animal, date = item.split(":")
    rv = PathResolver()
    c1 = oi.session_windows(rv, PROD["cam1"], animal, date, "cam1")
    c4 = oi.session_windows(rv, PROD["cam4"], animal, date, "cam4")
    if c1 is None or c4 is None:
        return []
    cg = _cgroup(rv)
    flat4 = c4[1]
    jf = _june_factor(date)
    if jf:
        K = cg.cameras[1].get_camera_matrix()
        flat4 = P3.scale_june_cam4(flat4, jf, (K[0, 2], K[1, 2]))
    meds = {"cam1": SF.ruler_medians(c1[1], c1[0]), "cam4": SF.ruler_medians(flat4, c4[0])}
    widths = {}
    for c in PS.SIDE:
        sw = PS.side_windows(rv, SIDE_BUNDLE, animal, date, c)
        if sw is None:
            return []
        meds[c] = SF.ruler_medians(sw[1], sw[0])
        widths[c] = PS.side_width(rv, animal, date, c)
    rig = _rig(rv, animal, date)
    _g, perr = PS.pose_all(cg, {c: meds[c] for c in ("cam1", "cam4", "cam2", "cam3")}, rig["geometry"])
    gate = PS.side_gate(perr, widths)
    rows = []
    for cam, e in perr.items():
        v = np.array(list(e.values()), float)
        r = {"session": f"{animal}_{date}", "animal": animal, "date": date, "cam": cam, "n_spouts": int(np.isfinite(v).sum()),
             "mean_px": float(np.nanmean(v)), "rms_px": float(np.sqrt(np.nanmean(v ** 2))), "max_px": float(np.nanmax(v))}
        if cam in gate:
            r.update({"rms600": gate[cam]["rms600"], "max600": gate[cam]["max600"], "side_gate_pass": bool(gate[cam]["pass"])})
        rows.append(r)
    return rows


def run_reproj(jobs: int) -> None:
    from wfield_local import analysis_kit as ak
    res, fail = ak.fan_sessions(_sessions(), _reproj_worker, jobs=jobs, label="reproj")
    R = pd.DataFrame([r for _k, rows in res for r in rows])
    R.to_csv(OUT / "reproj_per_session.csv", index=False)
    ok = R[R.get("side_gate_pass", pd.Series(True, index=R.index)).fillna(True).astype(bool)]
    S = ok.groupby("cam").agg(n_sessions=("session", "nunique"), mean_px_min=("mean_px", "min"),
                              mean_px_med=("mean_px", "median"), mean_px_max=("mean_px", "max"),
                              rms_px_min=("rms_px", "min"), rms_px_med=("rms_px", "median"), rms_px_max=("rms_px", "max"),
                              max_px_max=("max_px", "max")).round(2)
    S.to_csv(OUT / "reproj_summary.csv")
    print(S.to_string(), "\nfailed:", [f[0] for f in fail])


# -------------------------------------------------------------------------------------------------------------- fill
def _fill_worker(item: str) -> list[dict]:
    os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
    from wfield_local import pose3d_side as PS
    from wfield_local.paths import PathResolver
    animal, date = item.split(":")
    Z = PS.fuse_session(PathResolver(), animal, date, side_bundle=SIDE_BUNDLE)
    F = Z["F"]
    trial, t = F.trial_id.to_numpy(), F.t_ms.to_numpy(float)
    rng = np.random.default_rng(0)
    rows = []
    for part, gaps in GAPS.items():
        gs, pts = Z["parts"][part]["g"], Z["pts"][part]
        X, used, _d = PS.fuse_robust(gs, pts, PS.REJECT_MM[part])
        multi = np.isfinite(X).all(1)
        same = np.r_[True, (trial[1:] == trial[:-1]) & np.isclose(np.diff(t), FRAME_MS)]  # frame i continues i-1
        for gap in gaps:
            n = int(round(gap / FRAME_MS))
            # candidate start i: frames i-1 .. i+n all multi and contiguous within one trial
            ok = np.zeros(len(X), bool)
            m_run = pd.Series(multi & same).rolling(n + 2).sum().to_numpy()
            for i in range(1, len(X) - n):
                if multi[i - 1] and m_run[i + n] == n + 2 and same[i:i + n + 1].all():
                    ok[i] = True
            cand = np.flatnonzero(ok)
            rng.shuffle(cand)
            taken = np.zeros(len(X), bool)
            starts = []
            for i in cand:
                lo, hi = max(0, i - n - 3), min(len(X), i + 2 * n + 3)
                if not taken[lo:hi].any():
                    taken[i:i + n] = True
                    starts.append(i)
                if len(starts) >= 600:
                    break
            if not starts:
                continue
            Xm, um = X.copy(), used.copy()
            keep_cam = []
            for i in starts:
                cams = np.flatnonzero(used[:, i:i + n].all(1))
                j = int(rng.choice(cams)) if len(cams) else int(np.flatnonzero(used[:, i])[0])
                Xm[i:i + n] = np.nan
                um[:, i:i + n] = False
                um[j, i:i + n] = True
                keep_cam.append(j)
            Xf, nv = PS.fill_n(gs, Xm, pts, um, trial, t, PS.FILL_MS[part])
            order = [c.get_name() for c in gs.cameras]
            for i, j in zip(starts, keep_cam):
                e = np.linalg.norm(Xf[i:i + n] - X[i:i + n], axis=1)
                rows.append({"session": f"{animal}_{date}", "animal": animal, "part": part, "gap_ms": gap,
                             "kept_cam": order[j], "filled_frac": float(np.isfinite(e).mean()),
                             "err_med_mm": float(np.nanmedian(e)) if np.isfinite(e).any() else np.nan,
                             "err_max_mm": float(np.nanmax(e)) if np.isfinite(e).any() else np.nan})
    return rows


def run_fill(jobs: int) -> None:
    from wfield_local import analysis_kit as ak
    res, fail = ak.fan_sessions(FILL_SESSIONS, _fill_worker, jobs=jobs, label="fill")
    R = pd.DataFrame([r for _k, rows in res for r in rows])
    R.to_csv(OUT / "fill_heldout_runs.csv", index=False)
    S = R.groupby(["part", "gap_ms", "animal"]).agg(runs=("err_med_mm", "size"), filled=("filled_frac", "mean"),
                                                    err_med_mm=("err_med_mm", "median"),
                                                    err_p90_mm=("err_med_mm", lambda v: np.nanpercentile(v, 90))).round(3)
    S.to_csv(OUT / "fill_heldout_summary.csv")
    C = R.groupby(["part", "gap_ms", "kept_cam"]).err_med_mm.median().round(3).unstack()
    C.to_csv(OUT / "fill_heldout_by_kept_cam.csv")
    print(S.to_string(), "\n", C.to_string(), "\nfailed:", [f[0] for f in fail])


# ----------------------------------------------------------------------------------------------------------- lateral
def run_lateral() -> None:
    from wfield_local import lp_licks
    from wfield_local import orofacial_outputs as oo
    from wfield_local import session_licks as SL
    from wfield_local.paths import PathResolver
    rv = PathResolver()
    K = pd.read_csv(KIN, usecols=["session", "animal", "trial_id", "lick_key", "seen_cam4", "kin_ok", "side_ml", "epoch"])
    K = K[K.seen_cam4.astype(bool) & K.kin_ok.astype(bool) & np.isfinite(K.side_ml)]
    rows = []
    for s, g in K.groupby("session"):
        a, d = s.split("_")
        f = oo.session_dir(rv, lp_licks.input_run(), a, d) / "licks.csv"
        if not f.exists():
            continue
        L = pd.read_csv(f, usecols=["trial_id", "t_ms", "lr_px"])
        L["lick_key"] = SL.lick_key(L.trial_id, L.t_ms)
        M = g.merge(L[["lick_key", "lr_px"]], on="lick_key", how="inner")
        M = M[np.isfinite(M.lr_px)]
        rows.append(M.assign(neg_ml=-M.side_ml))
    M = pd.concat(rows, ignore_index=True)
    per = M.groupby("animal").apply(lambda x: pd.Series({"n_licks": len(x), "r": np.corrcoef(x.lr_px, x.neg_ml)[0, 1]}))
    per_s = M.groupby(["animal", "session"]).apply(
        lambda x: np.corrcoef(x.lr_px, x.neg_ml)[0, 1] if len(x) >= 50 else np.nan).rename("r").reset_index()
    per["session_r_median"] = per_s.groupby("animal").r.median()
    per = per.round(3)
    per.to_csv(OUT / "lateral_cam4_vs_3d.csv")
    per_s.to_csv(OUT / "lateral_cam4_vs_3d_sessions.csv", index=False)
    print(per.to_string(), "\nmatched licks:", len(M))


# --------------------------------------------------------------------------------------------------------------- ili
def run_ili() -> None:
    K = pd.read_csv(KIN, usecols=["session", "animal", "date", "epoch", "trial_id", "daq_peak_s", "kin_ok", "t_peak_3d"])
    K = K[(K.epoch == "pre") & (K.date.astype(str).str[-4:] >= "0801") & K.kin_ok.astype(bool) & np.isfinite(K.t_peak_3d)]
    K = K.sort_values(["session", "trial_id", "t_peak_3d"])
    d = K.groupby(["session", "trial_id"]).t_peak_3d.diff()
    K["ili"] = d
    I = K[(K.ili > 0) & (K.ili <= 300)]
    S = I.groupby("animal").agg(ili_ms_median=("ili", "median"), n_intervals=("ili", "size"),
                                n_sessions=("session", "nunique")).round(1)
    used = pd.read_csv(P3D / "phase" / "vls_manuscript_20261122_fused" / "c2_pre_ili_ms.csv").set_index("animal")
    S["phase_cycle_ms_used"] = used.ili_ms
    S.to_csv(OUT / "pre_ili_by_animal.csv")
    print(S.to_string())


# --------------------------------------------------------------------------------------------------------------- daq
def _daq_worker(item: str) -> list[dict]:
    os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
    from wfield_local import behavior_events as be
    from wfield_local import session_licks as SL
    from wfield_local import trial_classes as tc
    from wfield_local.paths import PathResolver
    animal, date = item.split(":")
    label = f"{animal}_{date[4:]}"
    rv = PathResolver()
    try:
        A = SL.session_licks(label, rv)
    except Exception:
        return []
    T = tc.session_trials(label)
    ev = be.get_or_compute(rv, animal, date)
    if T is None or ev is None:
        return []
    on = np.sort(np.asarray(ev["lick_onsets"], float) / float(ev["fs"]))
    T = T[np.isfinite(T.cue_daq) & np.isfinite(T.stop_ms)]
    lo, hi = T.cue_daq.to_numpy(float) + 0.070, T.cue_daq.to_numpy(float) + T.stop_ms.to_numpy(float) / 1000.0
    tid = T.trial_id.to_numpy()
    P = A[A.context.astype(str) == "post_cue"]
    pk = P.set_index("trial_id").daq_peak_s
    out = {"session": f"{animal}_{date}", "animal": animal, "date": date, "n_fused": 0, "n_daq": 0}
    for X in (50, 100):
        out[f"daq_matched_{X}"] = 0
        out[f"fused_contact_{X}"] = 0
    for k in range(len(tid)):
        d = on[(on >= lo[k]) & (on <= hi[k])]
        f = np.sort(np.atleast_1d(pk.get(tid[k], pd.Series(dtype=float))).astype(float))
        f = f[np.isfinite(f)]
        out["n_daq"] += len(d)
        out["n_fused"] += len(f)
        for X in (50, 100):
            x = X / 1000.0
            if len(f) and len(d):
                out[f"daq_matched_{X}"] += int((np.abs(d[:, None] - f[None, :]).min(1) <= x).sum())
                out[f"fused_contact_{X}"] += int((np.abs(f[:, None] - d[None, :]).min(1) <= x).sum())
    return [out]


def run_daq(jobs: int) -> None:
    from wfield_local import analysis_kit as ak
    sess = sorted(pd.read_csv(KIN, usecols=["session"]).session.unique())
    items = [f"{s.split('_')[0]}:{s.split('_')[1]}" for s in sess]
    res, fail = ak.fan_sessions(items, _daq_worker, jobs=jobs, label="daq")
    R = pd.DataFrame([r for _k, rows in res for r in rows])
    ep = pd.read_csv(KIN, usecols=["session", "epoch"]).drop_duplicates("session").set_index("session").epoch
    R["epoch"] = R.session.map(ep)
    R.to_csv(OUT / "daq_vs_fused_sessions.csv", index=False)
    g = R.groupby(["animal", "epoch"])[["n_fused", "n_daq", "daq_matched_50", "daq_matched_100",
                                        "fused_contact_50", "fused_contact_100"]].sum()
    for X in (50, 100):
        g[f"daq_matched_frac_{X}"] = g[f"daq_matched_{X}"] / g.n_daq
        g[f"fused_with_contact_frac_{X}"] = g[f"fused_contact_{X}"] / g.n_fused
    g["fused_per_daq"] = g.n_fused / g.n_daq
    g = g.round(3)
    g.to_csv(OUT / "daq_vs_fused_summary.csv")
    tot = R[["n_fused", "n_daq", "daq_matched_50", "daq_matched_100", "fused_contact_50", "fused_contact_100"]].sum()
    print(g.to_string(), "\nTOTAL", tot.to_dict(), "\nfailed:", [f[0] for f in fail])
    (OUT / "daq_vs_fused_total.json").write_text(json.dumps({k: int(v) for k, v in tot.items()}, indent=1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("what", choices=("reproj", "fill", "lateral", "ili", "daq"))
    ap.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    {"reproj": lambda: run_reproj(a.jobs), "fill": lambda: run_fill(a.jobs), "lateral": run_lateral, "ili": run_ili,
     "daq": lambda: run_daq(a.jobs)}[a.what]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
