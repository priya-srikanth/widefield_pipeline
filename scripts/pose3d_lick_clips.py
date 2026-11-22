"""Four-camera LICK example clips from the PRODUCTION outputs (Priya 11-22: "can you make new example 4-camera clips with
licks, showing each camera view and the 3d projection? These can replace the older clips you made before we ran the
full cam2/3 models.").

Unlike `scripts.pose3d_side_video` (11-18: pilot side predictions on short clip folders, fusion recomputed there), every
number drawn here comes from a production run:
  * 3-D points, views, 3-D licks  `session_poses/<p3d>/<a>_<d>_frames.csv.gz` / `_licks.csv` (whole-session four-camera
    run, side model `lp_side_r2b_prod_20261118` from the O2 bundle);
  * per-camera lick events and the fused lick train  `session_poses/camera_licks/<licks>/<a>_<d>_events.csv` /
    `_fused.csv` (``keep`` / ``reason`` = the table-gaps rule: side-only licks need cam2 AND cam3);
  * DAQ lick contacts  `behavior_events.get_or_compute` (the canonical DAQ detector, 40 ms floor).
The per-camera CLEANED 2-D keypoints and which of them the robust fusion used are not stored by the production run, so
`pose3d_side.fuse_session` is re-run for the session under the SAME routing env as the production run (``--preset``);
the script then CHECKS that its fused 3-D equals the production frames table on every rendered frame (and that the
recomputed 3-D licks equal `_licks.csv` on the clip trials) and refuses to render if not. The 3-D drawn is always the
production table's.

    conda activate dlc        # MKL_THREADING_LAYER=SEQUENTIAL, PYTHONPATH=<worktree>
    python -m scripts.pose3d_lick_clips PS93:20260814 PS93:20260821 ... --preset 3net_20261121_sync60 \
        --bout PS93:20260814 --iti PS93:20260814 --out <folder> [--jobs 2]

Each frame: header (animal, date, epoch, trial, spout, time from cue, slow-motion factor); the four raw views (cam1
below, cam4 front, cam2 / cam3 sides; tiles fitted to the nose, the six commanded spouts and every drawn point, as
`pose3d_side_video --fit`) with the cleaned 2-D keypoints (dot = used in the 3-D point, ring = rejected by the robust
fusion) and the production 3-D point reprojected (x); the 3-D pose seen from cam2 (profile) and front-oblique
(commanded spouts as rings, the trial's target filled, 120 ms tongue / jaw trails); and a LICK TIMELINE: trial period
(pre-cue / cue .. trial stop / ITI), DAQ contacts, each camera's own lick events (cam4, cam1, cam2, cam3; side events
hollow when a side guard flagged them), the fused lick train (filled = kept, coloured by source; hollow = dropped, with
its reason) and the 3-D licks (``M`` = two tongue runs merged by the 60 ms video-lick floor; `pose3d.lick_peaks` counts
3-D licks only between the cue and the trial stop, so the row is empty in the ITI), with a moving cursor and the current
lick's source spelled out. ``used`` = the production table's per-camera reprojection is defined (``<part>_rep_<cam>``;
a one-view fill, views = -1, shows its one camera's point as used). 25 fps from 250 fps video (10x slow motion), H.264 / yuv420p (PowerPoint).

TRIAL SELECTION (seeded, `SEED` + crc32 of the session; never chosen by eye):
  * one trial per spout position, cue -100 .. +700 ms. Candidates: every trial window covers the clip; >= 2 KEPT fused
    licks peak in 0 .. 700 ms; every fused lick in the clip is kept and seen by >= 3 cameras; >= 15 frames with a 3-D
    tongue. Candidates are shuffled with the seed and the first one whose four videos have every frame is taken. No
    candidate -> the rule relaxes (>= 2 cameras per lick, then >= 1 kept lick), stated in the README table.
  * ``--bout``: >= 7 consecutive kept fused licks, each seen by >= 3 cameras, peak-to-peak <= 220 ms, inside one trial
    after the cue: clip = first peak - 150 .. +1350 ms (1.5 s).
  * ``--iti``: >= 4 consecutive kept fused licks peak-to-peak <= 250 ms, all after the trial stop (ITI); preferred
    where the sync-span windows reach beyond the old trial windows (stop + 3 s), so the whole-session coverage shows.
  * ``--merge``: a 3-D lick that the video-lick floor made from >= 2 tongue runs (peaks closer than the floor), every
    fused lick in the clip kept: clip = its onset - 250 .. + 450 ms (the ``M`` marker in the 3-D row).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

SIDE_BUNDLE = "lp_side_r2b_prod_20261118"
SEED = 20261122
OUT_FPS = 25
TRAIL_MS = 120.0
HOLD_FRAMES = 12
DLC = Path("M:/MICROSCOPE/Priya/DeepLabCut/Widefield")
LP = Path("M:/MICROSCOPE/Priya/Widefield/LP_analysis")
PREP_DIR = Path("C:/Users/SabatiniLab/lp_stage/clips_20261122/prep")
#: the production runs' routing env (copied from their launchers: the 11-21 cohort run, running list "step 3"; the 11-22
#: sync-span re-run, lp_stage/rerun_20261122/chain.py ENV); `fuse_session` must see the same windows / cam4 lick run
PRESETS = {
    "3net_20261121": {
        "p3d": "pose3d_4cam_3net_20261121",
        "licks": f"{SIDE_BUNDLE}_3net_20261121_gapfix", "floor_ms": 40.0,     # run before the 60 ms floor (f2071544)
        "env": {"WIDEFIELD_LICK_INPUT_RUN": "lp_r4_heatmap_20261013__clean_v37",
                "WIDEFIELD_CAM1_LICKS_DIR": "cam1_licks_v37", "WIDEFIELD_POSE3D_SOURCE": "4cam",
                "WIDEFIELD_POSE3D_TAG": "_3net_20261121"}},
    "3net_20261121_sync60": {
        "p3d": "pose3d_4cam_3net_20261121_sync60",
        "licks": f"{SIDE_BUNDLE}_3net_20261121_sync60_gapfix", "floor_ms": 60.0,
        "env": {"WIDEFIELD_WINDOWS_SPAN": "sync", "WIDEFIELD_RUN_LABEL": "60",
                "WIDEFIELD_LICK_INPUT_RUN": "lp_r4_heatmap_20261013__clean_v37_sync60",
                "WIDEFIELD_CAM1_LICKS_DIR": "cam1_licks_v37_sync60", "WIDEFIELD_POSE3D_SOURCE": "4cam",
                "WIDEFIELD_POSE3D_TAG": "_3net_20261121_sync60"}},
}
PARTS = ("nose", "jaw", "tongue", "spout")
CAMS_STRIP = ("cam4", "cam1", "cam2", "cam3")
W = 1200
STRIP_L = 118                      # px of row labels left of the timeline
#: fused-lick source classes -> BGR
SRC_COL = {"all cameras": (235, 235, 235), "multi-view": (235, 170, 80), "cam1-only": (0, 150, 255),
           "cam4-only": (230, 60, 230), "side-only, both side views": (240, 240, 0)}
CAM_COL = {"cam4": (120, 215, 255), "cam1": (160, 230, 160), "cam2": (230, 160, 230), "cam3": (240, 200, 140)}
POS_ORDER = ["close_L", "close_center", "close_R", "far_L", "far_center", "far_R"]


# ------------------------------------------------------------------------------------------------ inputs

def load_tables(p3d: Path, lk: Path, animal: str, date: str) -> dict:
    s = f"{animal}_{date}"
    return {"S": pd.read_csv(p3d / f"{s}_frames.csv.gz"), "L": pd.read_csv(p3d / f"{s}_licks.csv"),
            "U": pd.read_csv(lk / f"{s}_fused.csv"), "E": pd.read_csv(lk / f"{s}_events.csv", low_memory=False)}


def cue_daq_s(cam4_run: str, animal: str, date: str) -> dict:
    """{trial_id: DAQ second of the cue} from the cam4 run's windows (``ix_daq_s`` - ``ix_t_daq_ms``: every frame's DAQ
    time and its DAQ time from the cue), so DAQ contacts / rewards land on the clip's cue clock."""
    w = np.load(LP / cam4_run / "sessions" / animal / date / "windows.npz", allow_pickle=True)
    d = pd.DataFrame({"tid": w["ix_trial_id"], "c": w["ix_daq_s"] - w["ix_t_daq_ms"] / 1000.0})
    return d.groupby("tid").c.median().to_dict()


def source_class(r) -> str:
    if not bool(r.keep):
        return f"dropped: {r.reason}"
    cams = str(r.cams)
    if int(r.n_views) == 4:
        return "all cameras"
    if cams == "cam1":
        return "cam1-only"
    if cams == "cam4":
        return "cam4-only"
    if cams == "cam2+cam3":
        return "side-only, both side views"
    return "multi-view"


def _rng(session: str) -> np.random.Generator:
    return np.random.default_rng(SEED + zlib.crc32(session.encode()))


def trial_candidates(T: dict, session: str, t0: float, t1: float) -> dict:
    """{position: [(trial_id, rule), ...]} in seeded order, strict rule first (module doc)."""
    S, U = T["S"], T["U"]
    span = S.groupby("trial_id").t_ms.agg(["min", "max"])
    tongue = S[(S.t_ms >= t0) & (S.t_ms <= t1) & (S.tongue_views != 0)].groupby("trial_id").size()
    W_ = U[(U.off_ms >= t0) & (U.on_ms <= t1)]
    out = {}
    rng = _rng(session)
    for pos, g in S.drop_duplicates("trial_id").groupby("position"):
        tids = sorted(g.trial_id.tolist())
        tids = [t for t in tids if t in span.index and span.loc[t, "min"] <= t0 and span.loc[t, "max"] >= t1]
        rules = {}
        for tid in tids:
            w = W_[W_.trial_id == tid]
            kept = w[w.keep.astype(bool) & (w.t_peak_ms >= 0) & (w.t_peak_ms <= t1)]
            nt = int(tongue.get(tid, 0))
            if len(kept) >= 2 and w.keep.astype(bool).all() and (w.n_views >= 3).all() and nt >= 15:
                rules[tid] = "strict"
            elif len(kept) >= 2 and w.keep.astype(bool).all() and (w.n_views >= 2).all():
                rules[tid] = "relaxed: >= 2 cameras per lick"
            elif len(kept) >= 1:
                rules[tid] = "relaxed: >= 1 kept lick"
        lst = []
        for rule in ("strict", "relaxed: >= 2 cameras per lick", "relaxed: >= 1 kept lick"):
            c = np.array([t for t, r in rules.items() if r == rule])
            lst += [(int(t), rule) for t in rng.permutation(c)] if len(c) else []
        out[pos] = lst
    return out


def merge_candidates(rv, T: dict, session: str, floor_ms: float) -> list:
    """Seeded list of (trial_id, onset_ms, n_runs) 3-D licks merged by the floor (module doc): `pose3d.lick_peaks` with
    the production cuts and no floor, against the production `_licks.csv`."""
    from scripts.apparatus_mouth import _rig
    from wfield_local import camera_licks as CL
    from wfield_local import pose3d as P3
    animal, date = session.split(":")
    S = T["S"]
    L0 = P3.lick_peaks(S, _rig(rv, animal, date), cuts=CL.event_cuts(CL.production_events(rv, animal, date, span="all"),
                                                                     S=S), min_ili_ms=0.0)
    U = T["U"]
    out = []
    for tid, l in T["L"].groupby("trial_id"):
        l0 = L0[L0.trial_id == tid]
        u = U[U.trial_id == tid]
        for r in l.itertuples():
            a, b = float(r.t_onset_ms), float(r.t_onset_ms) + 4.0 * (int(r.n_frames) - 1)
            n = int(((l0.t_peak_ms >= a) & (l0.t_peak_ms <= b)).sum())
            w = u[(u.off_ms >= a - 250) & (u.on_ms <= a + 450)]
            if n >= 2 and len(w) and w.keep.astype(bool).all():
                out.append((int(tid), a, n))
    rng = _rng(session + "_merge")
    return [out[k] for k in rng.permutation(len(out))]


def bout_candidates(T: dict, session: str, n_min: int = 7, ili_max: float = 220.0, iti: bool = False) -> list:
    """Seeded list of (trial_id, t_start, t_end, n_licks, beyond_old_window) lick bouts (module doc)."""
    U = T["U"].sort_values(["trial_id", "t_peak_ms"])
    stop = T["S"].drop_duplicates("trial_id").set_index("trial_id").stop_ms
    out = []
    for tid, g in U.groupby("trial_id"):
        if iti:
            g = g[g.t_peak_ms > stop.get(tid, np.inf) + 100]
        else:
            g = g[g.t_peak_ms >= 0]
        ok = g.keep.astype(bool).to_numpy() & ((g.n_views >= 3).to_numpy() | iti)
        pk = g.t_peak_ms.to_numpy(float)
        i = 0
        while i < len(g):
            if not ok[i]:
                i += 1
                continue
            j = i + 1
            while j < len(g) and ok[j] and pk[j] - pk[j - 1] <= ili_max:
                j += 1
            if j - i >= n_min:
                beyond = bool(pk[i] > stop.get(tid, np.inf) + 3000.0)
                out.append((int(tid), float(pk[i]), float(pk[j - 1]), j - i, beyond))
            i = j
    rng = _rng(session + ("_iti" if iti else "_bout"))
    out = [out[k] for k in rng.permutation(len(out))]
    if iti:                                          # beyond the old trial windows first (sync-span coverage)
        out = [b for b in out if b[4]] + [b for b in out if not b[4]]
    return out


# ------------------------------------------------------------------------------------------------ prep (fusion re-run)

def video_complete(Z: dict, rows: np.ndarray) -> bool:
    from wfield_local import pose3d_side as PS
    F = Z["F"]
    if not (np.isfinite(F.src1.to_numpy(float)[rows]).all() and np.isfinite(F.src4.to_numpy(float)[rows]).all()):
        return False
    for c in PS.SIDE:
        if c in Z["stats"]["side_dropped"]:
            continue
        if (F[f"i_{c}"].to_numpy()[rows] < 0).any():
            return False
    return True


def prep_session(rv, session: str, T: dict, clips: list, floor_ms: float) -> dict:
    """Fuse the session again (cleaned 2-D + used flags), check it against the production table, and extract every
    clip's frames. ``clips``: [{"name", "kind", "cands": [(trial, t0, t1, rule), ...]}] -> the first candidate with
    complete video is taken."""
    from scripts.pose3d_figures import OBLIQUE
    from scripts.pose3d_side_video import view_from
    from wfield_local import camera_licks as CL
    from wfield_local import o2_inference as oi
    from wfield_local import pose3d as P3
    from wfield_local import pose3d_side as PS
    from wfield_local import spout_world as SW
    animal, date = session.split(":")
    Z = PS.fuse_session(rv, animal, date, side_bundle=SIDE_BUNDLE)
    S = T["S"]
    F, order = Z["F"], Z["order"]
    if len(F) != len(S) or not (np.array_equal(F.trial_id.to_numpy(), S.trial_id.to_numpy())
                                and np.allclose(F.t_ms.to_numpy(float), S.t_ms.to_numpy(float))):
        raise RuntimeError(f"{session}: re-run fusion frames != production frames ({len(F)} vs {len(S)})")
    cams = {c.get_name(): c for c in Z["g"].cameras}
    side = Z["side"]
    src = {"cam1": F.src1.to_numpy(float), "cam4": F.src4.to_numpy(float),
           **{c: np.where(F[f"i_{c}"].to_numpy() >= 0,
                          side[c][0].src_frame.to_numpy()[np.clip(F[f"i_{c}"].to_numpy(), 0, None)], np.nan)
              for c in PS.SIDE}}
    tgt = {p: np.asarray(SW.commanded_world(Z["rig"]["geometry"], p), float) for p in SW.POSITIONS}
    Xs = {p: S[[f"{p}_ml", f"{p}_ap", f"{p}_dv"]].to_numpy(float) for p in PARTS}
    allX = np.vstack([Xs[p] for p in PARTS] + [np.array(list(tgt.values()))])
    lo, hi = np.nanpercentile(allX, 1, axis=0) - 0.5, np.nanpercentile(allX, 99, axis=0) + 0.5
    nose_w = np.nanmedian(Xs["nose"], axis=0)
    out = {"session": session, "order": order, "tgt": tgt, "lo": lo, "hi": hi,
           "views": [(*view_from(cams["cam2"]), "3-D seen from cam2 (profile)"),
                     (OBLIQUE[0], OBLIQUE[1], "3-D oblique (front-right, above)")],
           "nose_px": {c: PS.project(cams[c], nose_w[None])[0] for c in order},
           "mouth_px": {c: PS.project(cams[c], np.zeros((1, 3)))[0] for c in order},
           "spout_px": {c: PS.project(cams[c], np.array(list(tgt.values()))) for c in order},
           "side_dropped": list(Z["stats"]["side_dropped"]),
           "videos": {"cam1": str(oi.videos_for(rv, animal, date, "cam1")[0]),
                      "cam4": str(oi.videos_for(rv, animal, date, "cam4")[0]),
                      **{c: str(sorted((Path(rv.root("behavior_cameras")) / date / animal).glob(f"{c}_*.avi"))[0])
                         for c in PS.SIDE}},
           "clips": [], "checks": []}
    tid_all, t_all = F.trial_id.to_numpy(), F.t_ms.to_numpy(float)
    chosen_trials = []
    for spec in clips:
        for (tr, t0, t1, rule) in spec["cands"]:  # noqa: B007 (used after the loop)
            rows = np.flatnonzero((tid_all == tr) & (t_all >= t0) & (t_all <= t1))
            if len(rows) and video_complete(Z, rows):
                break
        else:
            print(f"  {session} {spec['name']}: no candidate with complete video")
            continue
        clip = {"name": spec["name"], "kind": spec["kind"], "trial": int(tr), "t0": t0, "t1": t1, "rule": rule,
                "rows": rows, "position": str(F.position.iloc[rows[0]]), "t_ms": t_all[rows],
                "stop_ms": float(S.stop_ms.iloc[rows[0]]),
                "cue_daq_ms": float(np.nanmedian(S.t_daq_ms.to_numpy(float)[rows] - t_all[rows])),
                "src": {c: src[c][rows] for c in order}, "pts": {}, "used": {}, "rep": {}, "X": {}, "nviews": {},
                "trail_X": {}}
        # check: the re-run fusion's 3-D == the production table on every rendered frame
        for p in PARTS:
            Xz, Xp = Z["parts"][p]["X"][rows], Xs[p][rows]
            same_nan = np.array_equal(np.isfinite(Xz).all(1), np.isfinite(Xp).all(1))
            ok = np.isfinite(Xp).all(1)
            dmax = float(np.nanmax(np.abs(Xz[ok] - Xp[ok]))) if ok.any() else 0.0
            used_z = Z["parts"][p]["used"][:, rows]
            used_p = np.vstack([np.isfinite(S[f"{p}_rep_{c}"].to_numpy(float)[rows]) for c in order])
            out["checks"].append({"clip": spec["name"], "part": p, "frames": len(rows), "same_missing": same_nan,
                                  "max_abs_mm": dmax, "used_agree": float((used_z == used_p).mean())})
            if not same_nan or dmax > 0.002:            # the table is written with 3 decimals
                raise RuntimeError(f"{session} {spec['name']} {p}: fusion re-run != production ({dmax:.4f} mm)")
            gcam = {cc.get_name(): cc for cc in Z["parts"][p]["g"].cameras}
            clip["pts"][p] = Z["pts"][p][:, rows]
            clip["used"][p] = used_p
            clip["X"][p] = Xp
            clip["nviews"][p] = S[f"{p}_views"].to_numpy()[rows]
            rep = np.full((len(order), len(rows), 2), np.nan)
            for j, c in enumerate(order):
                if ok.any():
                    rep[j, ok] = PS.project(gcam[c], Xp[ok])
            clip["rep"][p] = rep
        # trails need 120 ms before the first rendered frame
        pre = np.flatnonzero((tid_all == tr) & (t_all >= t0 - TRAIL_MS) & (t_all < t0))
        clip["trail_t"] = t_all[pre]
        clip["trail_X"] = {p: Xs[p][pre] for p in ("tongue", "jaw")}
        out["clips"].append(clip)
        chosen_trials.append(int(tr))
    # 3-D licks on the clip trials, with and without the 60 ms floor (`pose3d.lick_peaks` as the production run calls
    # it); the floor version must equal the production _licks.csv there
    trs = sorted(set(chosen_trials))
    Ssub = S[S.trial_id.isin(trs)].reset_index(drop=True)
    cuts = CL.event_cuts(CL.production_events(rv, animal, date, span="all"), S=Ssub)
    L60 = P3.lick_peaks(Ssub, Z["rig"], cuts=cuts, min_ili_ms=floor_ms)
    L0 = P3.lick_peaks(Ssub, Z["rig"], cuts=cuts, min_ili_ms=0.0)
    Lp = T["L"][T["L"].trial_id.isin(trs)]
    same = len(L60) == len(Lp) and np.allclose(L60.t_peak_ms.to_numpy(float), Lp.t_peak_ms.to_numpy(float))
    out["checks"].append({"clip": "3-D licks", "part": "tongue", "frames": len(L60), "same_missing": same,
                          "max_abs_mm": np.nan, "used_agree": np.nan})
    out["lick_check"] = (same, len(L60), len(Lp))
    out["L0"], out["floor_ms"] = L0, floor_ms
    return out


# ------------------------------------------------------------------------------------------------ drawing

def _x(t, t0, t1):
    return int(round(STRIP_L + (t - t0) / (t1 - t0) * (W - STRIP_L - 12)))


def strip_background(clip: dict, T: dict, daq_ms: np.ndarray, L0: pd.DataFrame, floor_ms: float,
                     rew_ms: np.ndarray) -> tuple[np.ndarray, list]:
    """The timeline (static part) and the fused licks of the clip [(on, off, label, kept)]."""
    import cv2
    tr, t0, t1, stop = clip["trial"], clip["t0"], clip["t1"], clip["stop_ms"]
    rows = ["trial", "DAQ contact", *CAMS_STRIP, "fused (4 cams)", "3-D (cue-stop)"]
    rh, top = 19, 26
    h = top + rh * len(rows) + 40
    im = np.full((h, W, 3), 18, np.uint8)
    y = {r: top + k * rh for k, r in enumerate(rows)}
    F_ = cv2.FONT_HERSHEY_SIMPLEX
    for r in rows:
        cv2.putText(im, r, (6, y[r] + 13), F_, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.line(im, (STRIP_L, y[r] + rh - 1), (W - 12, y[r] + rh - 1), (45, 45, 45), 1)
    X = lambda t: _x(np.clip(t, t0, t1), t0, t1)     # noqa: E731
    # trial period
    yy = y["trial"]
    cv2.rectangle(im, (X(t0), yy + 3), (X(min(0, t1)), yy + rh - 4), (70, 70, 70), -1)
    if t1 > 0:
        cv2.rectangle(im, (X(0), yy + 3), (X(stop), yy + rh - 4), (60, 120, 40), -1)
    if t1 > stop:
        cv2.rectangle(im, (X(stop), yy + 3), (X(t1), yy + rh - 4), (90, 60, 60), -1)
    for lab, a, b in (("pre-cue", t0, 0), ("cue .. trial stop", 0, stop), ("ITI", stop, t1)):
        a_, b_ = max(a, t0), min(b, t1)
        if b_ - a_ > 0.12 * (t1 - t0):
            cv2.putText(im, lab, (X(a_) + 4, yy + 13), F_, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    for tl, lab in ((0.0, "cue"), (stop, "stop"), (stop + 3000.0, "old window end")):
        if t0 <= tl <= t1:
            col = (255, 255, 255) if lab == "cue" else (150, 150, 150)
            for yd in range(top, y["3-D (cue-stop)"] + rh, 6):
                cv2.line(im, (X(tl), yd), (X(tl), yd + 3), col, 1)
            cv2.putText(im, lab, (X(tl) + 2, top - 4), F_, 0.36, col, 1, cv2.LINE_AA)
    for t in rew_ms[(rew_ms >= t0) & (rew_ms <= t1)]:                 # water delivery (DAQ reward line)
        cv2.drawMarker(im, (X(t), yy + rh // 2), (255, 200, 0), cv2.MARKER_TRIANGLE_DOWN, 11, 2, cv2.LINE_AA)
    # DAQ contacts
    for t in daq_ms[(daq_ms >= t0) & (daq_ms <= t1)]:
        cv2.line(im, (X(t), y["DAQ contact"] + 3), (X(t), y["DAQ contact"] + rh - 4), (80, 80, 255), 2)
    # per-camera events
    E = T["E"]
    E = E[(E.trial_id == tr) & (E.off_ms >= t0) & (E.on_ms <= t1)]
    for _, e in E.iterrows():
        c = str(e.cam)
        if c not in y:
            continue
        guard = bool(e.get("static", False) is True or str(e.get("static", "")) == "True"
                     or str(e.get("far", "")) == "True")
        cv2.rectangle(im, (X(e.on_ms), y[c] + 4), (max(X(e.off_ms), X(e.on_ms) + 1), y[c] + rh - 5), CAM_COL[c],
                      1 if guard else -1)
        if np.isfinite(e.t_peak_ms) and t0 <= e.t_peak_ms <= t1:
            cv2.line(im, (X(e.t_peak_ms), y[c] + 2), (X(e.t_peak_ms), y[c] + rh - 3), (255, 255, 255), 1)
    # fused train
    U = T["U"]
    U = U[(U.trial_id == tr) & (U.off_ms >= t0) & (U.on_ms <= t1)]
    fused = []
    yf = y["fused (4 cams)"]
    for _, u in U.iterrows():
        lab = source_class(u)
        kept = bool(u.keep)
        col = SRC_COL.get(lab, (235, 170, 80)) if kept else (140, 140, 140)
        cv2.rectangle(im, (X(u.on_ms), yf + 3), (max(X(u.off_ms), X(u.on_ms) + 1), yf + rh - 4), col,
                      -1 if kept else 1)
        if not kept:
            cv2.putText(im, str(u.reason)[:22], (X(u.on_ms), yf + 13), F_, 0.32, (200, 200, 200), 1, cv2.LINE_AA)
        elif lab not in ("all cameras",):
            short = {"cam1-only": "c1", "cam4-only": "c4", "side-only, both side views": "S2"}.get(lab, "")
            if short:
                cv2.putText(im, short, (X(u.on_ms) + 1, yf + 13), F_, 0.32, (0, 0, 0), 1, cv2.LINE_AA)
        fused.append((float(u.on_ms), float(u.off_ms), lab if lab != "multi-view" else f"multi-view ({u.cams})",
                      kept, str(u.cams)))
    # 3-D licks (production table) + merges by the 60 ms floor (>= 2 no-floor licks inside one)
    L = T["L"][T["L"].trial_id == tr]
    l0 = L0[L0.trial_id == tr]
    y3 = y["3-D (cue-stop)"]
    for _, l_ in L.iterrows():
        a, b = float(l_.t_onset_ms), float(l_.t_onset_ms) + 4.0 * (int(l_.n_frames) - 1)
        if b < t0 or a > t1:
            continue
        nparts = int(((l0.t_peak_ms >= a) & (l0.t_peak_ms <= b)).sum())
        col = (90, 90, 255) if nparts < 2 else (0, 165, 255)
        cv2.rectangle(im, (X(a), y3 + 3), (max(X(b), X(a) + 1), y3 + rh - 4), col, -1)
        if t0 <= l_.t_peak_ms <= t1:
            cv2.line(im, (X(l_.t_peak_ms), y3 + 1), (X(l_.t_peak_ms), y3 + rh - 2), (255, 255, 255), 1)
        if nparts >= 2:
            cv2.putText(im, "M", (X(a) + 2, y3 + 14), F_, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    # time axis
    ya = y["3-D (cue-stop)"] + rh + 2
    step = 100 if t1 - t0 <= 1600 else 250
    for t in np.arange(np.ceil(t0 / step) * step, t1 + 1e-6, step):
        cv2.line(im, (X(t), ya), (X(t), ya + 4), (180, 180, 180), 1)
        cv2.putText(im, f"{t:+.0f}", (X(t) - 14, ya + 16), F_, 0.34, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(im, "ms from cue", (6, ya + 16), F_, 0.36, (180, 180, 180), 1, cv2.LINE_AA)
    xl = 6
    for lab, col, fill in (("fused: all cameras", SRC_COL["all cameras"], True),
                           ("multi-view", SRC_COL["multi-view"], True), ("c1 = cam1-only", SRC_COL["cam1-only"], True),
                           ("c4 = cam4-only", SRC_COL["cam4-only"], True),
                           ("S2 = side-only, both side views", SRC_COL["side-only, both side views"], True),
                           ("dropped (reason)", (140, 140, 140), False),
                           (f"3-D: M = runs merged by the {floor_ms:.0f} ms floor", (0, 165, 255), True),
                           ("DAQ contact", (80, 80, 255), True), ("reward (trial row)", (255, 200, 0), True)):
        cv2.rectangle(im, (xl, ya + 24), (xl + 12, ya + 33), col, -1 if fill else 1)
        cv2.putText(im, lab, (xl + 16, ya + 33), F_, 0.34, (200, 200, 200), 1, cv2.LINE_AA)
        xl += 22 + int(cv2.getTextSize(lab, F_, 0.34, 1)[0][0])
    return im, fused


def fit_crops(P: dict, clips: list) -> tuple[dict, dict]:
    from scripts.pose3d_side_video import CROP, FIT_MARGIN
    crop, centre = {}, {}
    for j, c in enumerate(P["order"]):
        pts = [P["nose_px"][c][None], P["spout_px"][c]]
        for cl in clips:
            for p in PARTS:
                q = cl["pts"][p][j][cl["used"][p][j]]
                pts += [q, cl["rep"][p][j]]
        A = np.vstack(pts)
        A = A[np.isfinite(A).all(1)]
        lo_, hi_ = A.min(0) - FIT_MARGIN, A.max(0) + FIT_MARGIN
        crop[c] = int(max(CROP[c], (hi_ - lo_).max()))
        centre[c] = (lo_ + hi_) / 2
    return crop, centre


class H264:
    """Raw BGR frames -> ffmpeg (imageio-ffmpeg's binary) libx264 yuv420p, no console window."""
    def __init__(self, path: Path, w: int, h: int, fps: int = OUT_FPS):
        import imageio_ffmpeg
        self.p = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo",
                                   "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
                                   "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                                   str(path)], stdin=subprocess.PIPE, creationflags=0x08000000)
        self.wh = (w, h)

    def write(self, im):
        assert (im.shape[1], im.shape[0]) == self.wh
        self.p.stdin.write(np.ascontiguousarray(im).tobytes())

    def close(self):
        self.p.stdin.close()
        if self.p.wait() != 0:
            raise RuntimeError("ffmpeg failed")


def render(P: dict, T: dict, clips: list, f_out: Path, daq_s: np.ndarray, rew_s: np.ndarray, cues: dict, epoch: str,
           stills: Path | None) -> list:
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from scripts.pose3d_side_video import BGR, HEX, TILE, Reader
    animal, date = P["session"].split(":")
    order = P["order"]
    crop, centre = fit_crops(P, clips)
    readers = {c: Reader(P["videos"][c]) for c in order}
    lo, hi, tgt = P["lo"], P["hi"], P["tgt"]
    fig = plt.figure(figsize=(W / 100, 3.0), dpi=100)
    axes = [fig.add_subplot(1, 2, k + 1, projection="3d") for k in range(2)]
    vw, written = None, []
    F_ = cv2.FONT_HERSHEY_SIMPLEX
    for cl in clips:
        c0 = 1000.0 * cues[cl["trial"]]
        bg, fused = strip_background(cl, T, daq_s * 1000.0 - c0, P["L0"], P["floor_ms"], rew_s * 1000.0 - c0)
        pos, tr = cl["position"], cl["trial"]
        n = len(cl["rows"])
        first_peak = None
        for k in range(n):
            t = cl["t_ms"][k]
            tiles = []
            for j, c in enumerate(order):
                s_ = cl["src"][c][k]
                im = readers[c].get(int(s_)) if np.isfinite(s_) else None
                if im is None:
                    im = np.zeros((600, 800, 3), np.uint8)
                    cv2.putText(im, "no frame", (20, 60), F_, 1.0, (0, 0, 255), 2)
                # the crop is shrunk to the tile, so marker sizes are set in TILE px (scaled by the crop)
                sc = max(1.0, min(crop[c], *im.shape[:2]) / TILE)
                for p in PARTS:
                    r_ = cl["rep"][p][j, k]
                    if np.isfinite(r_).all():
                        cv2.drawMarker(im, (int(round(r_[0])), int(round(r_[1]))), BGR[p], cv2.MARKER_TILTED_CROSS,
                                       int(16 * sc), max(1, int(round(1.5 * sc))), cv2.LINE_AA)
                for p in PARTS:
                    q = cl["pts"][p][j, k]
                    if np.isfinite(q).all():
                        used = bool(cl["used"][p][j, k])
                        xy = (int(round(q[0])), int(round(q[1])))
                        if used:
                            cv2.circle(im, xy, int(4 * sc), (0, 0, 0), -1, cv2.LINE_AA)
                            cv2.circle(im, xy, int(3 * sc), BGR[p], -1, cv2.LINE_AA)
                        else:
                            cv2.circle(im, xy, int(6 * sc), BGR[p], max(1, int(round(1.5 * sc))), cv2.LINE_AA)
                cs = min(crop[c], *im.shape[:2])
                x0 = int(np.clip(centre[c][0] - cs // 2, 0, im.shape[1] - cs))
                y0 = int(np.clip(centre[c][1] - cs // 2, 0, im.shape[0] - cs))
                tile = cv2.resize(im[y0:y0 + cs, x0:x0 + cs], (TILE, TILE))
                name = {"cam1": "cam1 (below)", "cam4": "cam4 (front)", "cam2": "cam2 (side)",
                        "cam3": "cam3 (side)"}[c] + (" -- dropped by side gate" if c in P["side_dropped"] else "")
                cv2.putText(tile, name, (6, 18), F_, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                tiles.append(tile)
            # 3-D panels
            sel_pre = cl["trail_t"] >= t - TRAIL_MS
            sel = (cl["t_ms"] <= t) & (cl["t_ms"] >= t - TRAIL_MS)
            for ax, (el, az, name) in zip(axes, P["views"]):
                ax.cla()
                for q, c_ in tgt.items():
                    ax.scatter(*c_, s=40, facecolor=HEX["spout"] if q == pos else "none", edgecolor="0.4", lw=0.8)
                for p in PARTS:
                    Xp = cl["X"][p]
                    if p in ("tongue", "jaw"):
                        A = np.vstack([cl["trail_X"][p][sel_pre], Xp[sel]])
                        A = A[np.isfinite(A).all(1)]
                        if len(A) > 1:
                            ax.plot(A[:, 0], A[:, 1], A[:, 2], color=HEX[p], lw=1.5)
                    if np.isfinite(Xp[k]).all():
                        ax.scatter(*Xp[k], s=36, color=HEX[p], edgecolor="k", lw=0.5)
                ax.set_xlim(lo[0], hi[0])
                ax.set_ylim(lo[1], hi[1])
                ax.set_zlim(lo[2], hi[2])
                ax.set_box_aspect(hi - lo)
                ax.set_xlabel("ml (+R)", fontsize=7)
                ax.set_ylabel("ap (out)", fontsize=7)
                ax.set_zlabel("dv (up)", fontsize=7)
                ax.tick_params(labelsize=6)
                ax.view_init(elev=el, azim=az)
                ax.set_title(name, fontsize=8)
            fig.subplots_adjust(left=0.0, right=1.0, bottom=0.03, top=0.92, wspace=0.0)
            fig.canvas.draw()
            im3 = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba())[:, :, :3], cv2.COLOR_RGB2BGR)
            # timeline + cursor + current lick
            st = bg.copy()
            xc = _x(t, cl["t0"], cl["t1"])
            cv2.line(st, (xc, 20), (xc, st.shape[0] - 22), (0, 255, 255), 1)
            cur = [f for f in fused if f[0] <= t <= f[1]]
            if cur:
                on, off, lab, kept, cams = cur[0]
                txt = f"current lick: {lab}" + (f"  [{cams}]" if lab in ("all cameras", "cam1-only", "cam4-only",
                                                                          "side-only, both side views") else "")
                col = SRC_COL.get(lab, (235, 170, 80)) if kept else (160, 160, 160)
                if first_peak is None and kept:
                    first_peak = k
            else:
                txt, col = "current lick: none", (150, 150, 150)
            cv2.putText(st, txt, (STRIP_L, 16), F_, 0.5, col, 1, cv2.LINE_AA)
            # header
            bar = np.zeros((46, W, 3), np.uint8)
            nv = "  ".join(f"{p} {int(cl['nviews'][p][k])}" for p in PARTS)
            what = {"trials": "", "bout": "  [lick bout]",
                    "iti": f"  [ITI bout, {t - cl['stop_ms']:+.0f} ms after trial stop]",
                    "merge": f"  [3-D lick merged by the {P['floor_ms']:.0f} ms floor]"}[cl["kind"]]
            cv2.putText(bar, f"{animal}  {date[:4]}-{date[4:6]}-{date[6:]}  {epoch}  trial {tr}  spout {pos}  "
                        f"t = {t:+.0f} ms from cue{what}   10x slow motion (250 fps shown at 25 fps)",
                        (8, 18), F_, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(bar, f"dot = keypoint used, ring = rejected, x = production 3-D reprojected | nose yellow, jaw "
                        f"green, tongue red, spout blue | 3-D views: {nv}", (8, 38), F_, 0.4, (220, 220, 220), 1,
                        cv2.LINE_AA)
            frame = np.vstack([bar, np.hstack(tiles), cv2.resize(im3, (W, int(W * im3.shape[0] / im3.shape[1]))), st])
            if frame.shape[0] % 2:
                frame = np.vstack([frame, np.zeros((1, W, 3), np.uint8)])
            if vw is None:
                vw = H264(f_out, frame.shape[1], frame.shape[0])
            vw.write(frame)
            if stills is not None and (k == first_peak or k == n // 2):
                tt = f"_t{tr}" if cl["kind"] == "trials" else ""
                p_ = stills / f"{f_out.stem}{tt}_{'lick' if k == first_peak else 'mid'}_{t:+.0f}ms.png"
                cv2.imwrite(str(p_), frame)
                written.append(p_)
        for _ in range(HOLD_FRAMES):
            vw.write(frame)
    if vw is not None:
        vw.close()
    plt.close(fig)
    return written


# ------------------------------------------------------------------------------------------------ one session

def run_session(item) -> dict:
    """Module level (fan_sessions pickles it). item = (session, preset, out, t0, t1, bout, iti, stills, trials)."""
    session, preset, out, t0, t1, bout, iti, stills, do_trials, merge = item
    os.environ.update(PRESETS[preset]["env"])
    import pickle

    from wfield_local import behavior_events as BE
    from wfield_local import epochs
    from wfield_local.paths import PathResolver
    rv = PathResolver()
    animal, date = session.split(":")
    pr = PRESETS[preset]
    T = load_tables(DLC / "session_poses" / pr["p3d"], DLC / "session_poses" / "camera_licks" / pr["licks"], animal, date)
    sets = []
    cands = trial_candidates(T, session, t0, t1) if do_trials else {}
    sets.append(("trials", [{"name": pos, "kind": "trials", "cands": [(tr, t0, t1, r) for tr, r in cands.get(pos, [])]}
                            for pos in POS_ORDER if pos in cands]))
    if bout:
        b = bout_candidates(T, session, n_min=7, ili_max=220.0)
        sets.append(("bout", [{"name": "bout", "kind": "bout",
                               "cands": [(tr, a - 150.0, a + 1350.0, f"bout: {n} licks") for tr, a, _e, n, _b in b]}]))
    if iti:
        b = bout_candidates(T, session, n_min=4, ili_max=250.0, iti=True)
        sets.append(("iti", [{"name": "iti", "kind": "iti",
                              "cands": [(tr, a - 150.0, min(e + 250.0, a + 1350.0),
                                         f"ITI bout: {n} licks{', beyond old window' if bb else ''}")
                                        for tr, a, e, n, bb in b]}]))
    if merge:
        m = merge_candidates(rv, T, session, pr["floor_ms"])
        sets.append(("merge", [{"name": "merge", "kind": "merge",
                                "cands": [(tr, a - 250.0, a + 450.0, f"3-D lick merged from {n} runs by the floor")
                                          for tr, a, n in m]}]))
    PREP_DIR.mkdir(parents=True, exist_ok=True)
    f_prep = PREP_DIR / f"{preset}_{animal}_{date}_{t0:+.0f}_{t1:+.0f}{'_tr' if do_trials else ''}{'_b' if bout else ''}{'_i' if iti else ''}{'_m' if merge else ''}.pkl"
    if f_prep.exists():
        P = pickle.loads(f_prep.read_bytes())
    else:
        P = prep_session(rv, session, T, [c for _, cl in sets for c in cl], pr["floor_ms"])
        f_prep.write_bytes(pickle.dumps(P))
    if not P["lick_check"][0]:
        raise RuntimeError(f"{session}: recomputed 3-D licks != production _licks.csv {P['lick_check'][1:]}")
    ev = BE.get_or_compute(rv, animal, date)
    daq_s = BE.lick_onsets_s(ev) if ev is not None else np.array([])
    rew_s = np.asarray(ev["reward_samples"], float) / float(ev["fs"]) if ev is not None else np.array([])
    cues = cue_daq_s(pr["env"]["WIDEFIELD_LICK_INPUT_RUN"], animal, date)
    epoch = epochs.epoch_of(f"{animal}_{date[4:]}") or "no epoch"
    epoch = {"pre": "pre-stroke"}.get(epoch, epoch)
    out = Path(out)
    st_dir = out / "stills" if stills else None
    if st_dir:
        st_dir.mkdir(parents=True, exist_ok=True)
    rows, files = [], []
    for kind, specs in sets:
        names = {s["name"] for s in specs}
        cl = [c for c in P["clips"] if c["kind"] == kind and c["name"] in names]
        if not cl:
            continue
        tr_tag = "" if kind == "trials" else f"_t{cl[0]['trial']}"
        f_out = out / f"clip4cam_{animal}_{date}_{kind}{tr_tag}.mp4"
        files += render(P, T, cl, f_out, daq_s, rew_s, cues, epoch, st_dir)
        for c in cl:
            U = T["U"]
            u = U[(U.trial_id == c["trial"]) & (U.off_ms >= c["t0"]) & (U.on_ms <= c["t1"])]
            rows.append({"file": f_out.name, "session": f"{animal}_{date}", "epoch": epoch, "kind": kind,
                         "position": c["position"], "trial": c["trial"], "t0_ms": round(c["t0"]),
                         "t1_ms": round(c["t1"]), "rule": c["rule"], "fused_licks": len(u),
                         "kept": int(u.keep.astype(bool).sum()),
                         "sources": "; ".join(f"{k} {v}" for k, v in u.apply(source_class, axis=1).value_counts().items()),
                         "side_dropped": "+".join(P["side_dropped"])})
    return {"rows": rows, "checks": [{"session": session, **c} for c in P["checks"]], "stills": [str(f) for f in files]}


def main(argv=None) -> int:
    from wfield_local import analysis_kit as ak
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="+", metavar="ANIMAL:YYYYMMDD")
    ap.add_argument("--preset", required=True, choices=sorted(PRESETS))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--t0", type=float, default=-100.0)
    ap.add_argument("--t1", type=float, default=700.0)
    ap.add_argument("--bout", nargs="*", default=[], help="sessions that also get a lick-bout clip")
    ap.add_argument("--iti", nargs="*", default=[], help="sessions that also get an ITI lick-bout clip")
    ap.add_argument("--merge", nargs="*", default=[], help="sessions that also get a floor-merged 3-D lick clip")
    ap.add_argument("--no-trials", nargs="*", default=[], help="sessions rendered for --bout / --iti only")
    ap.add_argument("--stills", action="store_true", help="also write PNG stills (first lick peak + mid clip)")
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args(argv)
    o = str(a.out).replace("\\", "/")
    if o.upper().startswith(("M:", "N:")) and not o.startswith("M:/MICROSCOPE/Priya/"):
        raise SystemExit("--out on a share must be under M:/MICROSCOPE/Priya/")
    a.out.mkdir(parents=True, exist_ok=True)
    items = [(s, a.preset, str(a.out), a.t0, a.t1, s in a.bout, s in a.iti, a.stills, s not in a.no_trials,
              s in a.merge)
             for s in a.sessions]
    res, fails = ak.fan_sessions(items, run_session, jobs=min(a.jobs, len(items)))
    rows = [r for _, v in res for r in v["rows"]]
    checks = [r for _, v in res for r in v["checks"]]
    # one table per folder over every invocation: rows of the clips this run wrote replace their old rows
    for name, new, key in (("clips.csv", rows, ["session", "file", "position"]),
                           ("checks_vs_production.csv", checks, ["session", "clip", "part"])):
        D = pd.DataFrame(new)
        f = a.out / name
        if f.exists() and f.stat().st_size > 2:
            old = pd.read_csv(f)
            if len(D):
                k_old = old[key].astype(str).agg("|".join, axis=1)
                old = old[~k_old.isin(set(D[key].astype(str).agg("|".join, axis=1)))]
            D = pd.concat([old, D], ignore_index=True)
        if len(D):
            D = D.sort_values(key[:2], kind="mergesort")
        D.to_csv(f, index=False)
    for _, v in res:
        for f in v["stills"]:
            print("still", f)
    for it, err in fails:
        print("FAILED", it[0], err)
    print(pd.DataFrame(rows).to_string())
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
