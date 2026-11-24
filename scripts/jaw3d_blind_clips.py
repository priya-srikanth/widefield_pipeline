"""VIDEO CLIPS for the six-spout 3-D vs 2-D jaw blind review (CC-B 11-23; sample from `jaw3d_blind_review sample6`).

    python scripts/jaw3d_blind_clips.py <blind_sixpos folder> [--jobs 3] [--ids Q001 ...]

Per trial of key.csv two mp4 clips, cue - 0.5 s .. response-window end (min(cue + 3.5 s, trial stop)) + 0.2 s, every
250-fps frame written at 62.5 fps (0.25x); front (cam4), below (cam1) and side (cam2) synced on the DAQ clock
(camera_sync templates; the S10A clip tool's frame reader, `scripts/caption_review_1123/s10a_video_qc.py`):
  clips_blind/<id>.mp4    video + the time from the cue ONLY (no animal, spout, call or trace): score these first
  clips_reveal/<id>.mp4   the same video + a strip with BOTH jaw signals, each against its own applied threshold:
                          3-D jaw displacement (mm; threshold max(4 sd, 0.20 mm) from the production rule, floor
                          line), cam4 jaw y and cam1 jaw y (px from the pre-cue mean, + = opening; threshold
                          max(4 sd, 4 px) / max(4 sd, 15 px) of the pre-cue -500 .. 0 ms), with the 3-D and 2-D onset
                          ticks, the first lick (fused list) and a cursor; title = mouse / epoch / spout.
Raw videos are only read. Writes only into the given folder.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

P3D = Path("M:/MICROSCOPE/Priya/DeepLabCut/Widefield/session_poses/pose3d_4cam_3net_20261121_sync60")
RUN = "lp_r4_heatmap_20261013__clean_v37_sync60"
REON = Path("M:/MICROSCOPE/Priya/stroke_orofacial_pipeline_outputs/vls_stroke_paper/ccB_caption_review_2026-11-23/"
            "jaw3d_qc/jaw3d_reonset_trials.csv.gz")
MOUSE = {"PS92": "#9", "PS93": "#10", "PS94": "#11", "PS95": "#12"}
CAMS = ("cam4", "cam1", "cam2")
TILE_H = 280
COL = {"j3": (0, 140, 255), "c4": (255, 120, 200), "c1": (120, 255, 120), "cue": (0, 220, 255), "lick": (255, 255, 255)}


def _vq():
    f = Path(__file__).resolve().parent / "caption_review_1123" / "s10a_video_qc.py"
    spec = importlib.util.spec_from_file_location("s10a_video_qc", f)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _traces(r: dict) -> dict:
    """{name: (t_daq_ms, value, threshold, onset_ms)} for the 3-D jaw and the two 2-D cameras of one trial."""
    from scripts import jaw_onsets as jo
    from wfield_local import pose_windows as pw
    from wfield_local.paths import PathResolver
    rv = PathResolver()
    a, d = r["session"].split("_")
    tid = int(r["trial_id"])
    out = {}
    F = pd.read_csv(P3D / f"{r['session']}_frames.csv.gz", usecols=["trial_id", "t_ms", "t_daq_ms", "jaw_ml", "jaw_ap",
                                                                      "jaw_dv"])
    g = F[F.trial_id == tid].drop_duplicates("t_ms").sort_values("t_ms")
    X = g[["jaw_ml", "jaw_ap", "jaw_dv"]].to_numpy(float)
    t = g.t_ms.to_numpy(float)
    pre = (t >= -500) & (t < 0) & np.isfinite(X).all(1)
    if pre.sum() >= 5:
        dd = np.linalg.norm(X - np.median(X[pre], axis=0), axis=1)
        bm = float(np.mean(dd[pre]))                     # the rule is |d - pre-cue mean| > max(4 sd, 0.20 mm)
        out["3-D jaw (mm)"] = (g.t_daq_ms.to_numpy(float), dd, r.get("thr_um", np.nan) / 1000.0 + bm,
                               r.get("jaw3d_onset_ms", np.nan), 0.2 + bm,
                               _fix3d_line(dd[pre], r), r.get("fix_jaw3d_ms", np.nan))
    sd = Path(rv.root("orofacial")) / RUN / "sessions" / a / d
    w = np.load(sd / "windows.npz", allow_pickle=False)
    m = w["ix_trial_id"] == tid
    y4 = np.load(sd / "clean.npz")["jaw_y"].astype(float)[m]
    cams = {"cam4": (w["ix_t_ms"][m].astype(float), w["ix_t_daq_ms"][m].astype(float), y4, 4.0)}
    d1 = Path(rv.root("dlc")) / "session_poses" / f"{a}_{d}_cam1_full"
    if (d1 / "windows_LP.csv").exists():
        ix1 = pw.index(d1)
        if "t_daq_ms" in ix1:
            P = pw.lp(d1)
            y1 = P[("jaw", "y")].to_numpy(float)
            y1[P[("jaw", "likelihood")].to_numpy(float) < jo.CAM1_JAW_CUT] = np.nan
            k = (ix1.trial_id.to_numpy() == tid)
            cams["cam1"] = (ix1.t_ms.to_numpy(float)[k], ix1.t_daq_ms.to_numpy(float)[k], y1[k], 15.0)
    for cam, (tm, td, y, floor) in cams.items():
        p = (tm >= -500) & (tm < 0) & np.isfinite(y)
        if p.sum() < 5:
            continue
        mu, s = float(np.mean(y[p])), float(np.std(y[p]))
        on = _cam_onset(r, cam, "")
        thr_f = _fix2d_line(y[p], mu, floor, r.get("fix_variant", ""))
        out[f"{cam} jaw y (px)"] = (td, y - mu, max(4 * s, floor), on, floor, thr_f, _cam_onset(r, cam, "fix_"))
    return out


def _cam_onset(r: dict, cam: str, pre: str) -> float:
    """The camera's onset under the table ``pre`` ("" production, "fix_" the fixed variant): its re-onset when the jaw
    was already moving at the cue (the 2-D rule), else its onset; NaN when it did not move."""
    if r.get(f"{pre}{cam}_moved", 0) != 1:
        return np.nan
    re_ = r.get(f"{pre}{cam}_reonset_ms", np.nan)
    return re_ if np.isfinite(re_) else r.get(f"{pre}{cam}_onset_ms", np.nan)


def _fix2d_line(base: np.ndarray, mu: float, floor: float, v: str):
    """(centre offset from the pre-cue mean, threshold) of a camera under the fixed variant ``v``; None = no variant."""
    if not v:
        return None
    from scripts import jaw_onsets as jo
    if v == "mad":
        med = float(np.median(base))
        return med - mu, max(4 * 1.4826 * float(np.median(np.abs(base - med))), floor)
    return 0.0, max(min(4 * float(np.std(base)), jo.CAP_MULT * floor), floor)


def _fix3d_line(base: np.ndarray, r: dict):
    """(centre, threshold) of the 3-D displacement under the fixed variant (threshold from jaw3d_reonset_trials_<v>)."""
    v = r.get("fix_variant", "")
    if not v:
        return None
    c = float(np.median(base)) if v == "mad" else float(np.mean(base))
    return c, r.get("fix_thr_um", np.nan) / 1000.0


def render(item) -> dict:
    r, folder = item
    import cv2
    import imageio_ffmpeg
    from wfield_local.paths import PathResolver
    vq = _vq()
    rv = PathResolver()
    a, d = r["session"].split("_")
    from wfield_local import trial_classes as tc
    tt = tc._trial_times(Path(rv.root("orofacial")) / RUN / "sessions" / a / d)
    cue = float(tt.cue_daq[tt.trial_id == int(r["trial_id"])].iloc[0])
    end_ms = float(np.fmin(3500.0, r["stop_ms"])) if np.isfinite(r["stop_ms"]) else 3500.0
    t_lo, t_hi = cue - 0.5, cue + end_ms / 1000.0 + 0.2
    base = vq._frames(rv, a, d, "cam4", t_lo, t_hi)
    if base is None:
        return {"id": r["id"], "status": "no cam4 frames"}
    fd4, k4, p4 = base
    master = fd4[k4]
    vids = {"cam4": (k4, vq._read(p4, k4))}
    for cam in CAMS[1:]:
        got = vq._frames(rv, a, d, cam, t_lo, t_hi)
        if got is None:
            continue
        fd, _k, p = got
        j = np.clip(np.searchsorted(fd, master), 1, len(fd) - 1)
        j = np.where(np.abs(fd[j - 1] - master) < np.abs(fd[j] - master), j - 1, j)
        vids[cam] = (j, vq._read(p, j))
    tiles = {}
    for cam, (_j, fr) in vids.items():
        h, w = next(iter(fr.values())).shape[:2]
        tiles[cam] = (int(round(w * TILE_H / h)), TILE_H)
    W_ = sum(v[0] for v in tiles.values())
    W_ += W_ % 2
    tr = _traces(r)
    t0, t1 = -500.0, end_ms + 200.0
    L, Rm, PH = 150, 20, 56
    SH = 12 + PH * len(tr) + 26
    X = lambda t: int(L + (t - t0) / (t1 - t0) * (W_ - L - Rm))   # noqa: E731
    strip = np.zeros((SH, W_, 3), np.uint8)
    keys = list(tr)
    for i, name in enumerate(keys):
        td, v, thr, on, floor, fx, on_f = tr[name]
        col = COL["j3"] if name.startswith("3-D") else COL["c4"] if name.startswith("cam4") else COL["c1"]
        y0, y1 = 8 + i * PH, 8 + i * PH + PH - 8
        ok = np.isfinite(v) & (td >= t0) & (td <= t1)
        hi = max(thr * 2.0, float(np.nanmax(np.abs(v[ok]))) if ok.any() else thr * 2)
        if fx is not None and name.startswith("3-D") and np.isfinite(fx[1]):
            hi = max(hi, (fx[0] + fx[1]) * 1.3)
        lo = -hi if not name.startswith("3-D") else 0.0
        yy = lambda u, lo=lo, hi=hi, y0=y0, y1=y1: int(y1 - (np.clip(u, lo, hi) - lo) / (hi - lo) * (y1 - y0))  # noqa: E731
        cv2.rectangle(strip, (L, y0), (W_ - Rm, y1), (60, 60, 60), 1)
        for s in ((1,) if name.startswith("3-D") else (1, -1)):
            cv2.line(strip, (L, yy(s * thr)), (W_ - Rm, yy(s * thr)), (150, 150, 150), 1, cv2.LINE_AA)
        if name.startswith("3-D"):
            cv2.line(strip, (L, yy(floor)), (W_ - Rm, yy(floor)), (90, 90, 90), 1, cv2.LINE_4)
        if fx is not None:                         # FIXED rule: cyan threshold line(s) and a lower cyan tick
            cf, tf = fx
            if np.isfinite(tf):
                lv = [cf + tf] if name.startswith("3-D") else [cf + tf, cf - tf]
                for u in lv:
                    cv2.line(strip, (L, yy(u)), (W_ - Rm, yy(u)), (255, 255, 0), 1, cv2.LINE_AA)
            if np.isfinite(on_f):
                cv2.line(strip, (X(on_f), y0 + (y1 - y0) // 2), (X(on_f), y1), (255, 255, 0), 2, cv2.LINE_AA)
        pts = np.array([[X(x), yy(u)] for x, u in zip(td[ok], v[ok])], np.int32)
        if len(pts) > 1:
            cv2.polylines(strip, [pts], False, col, 1, cv2.LINE_AA)
        vq._txt(strip, name, (4, y0 + 14), 0.38, col)
        fmt = (lambda u: f"{u:.2f}") if name.startswith("3-D") else (lambda u: f"{u:.0f}")
        # 3-D: both numbers are LINE positions (pre-cue centre + threshold), as drawn; cameras: the threshold
        ftxt = (f" / {fmt(fx[0] + fx[1] if name.startswith('3-D') else fx[1])}"
                if fx is not None and np.isfinite(fx[1]) else "")
        vq._txt(strip, f"thr {fmt(thr)}{ftxt}", (4, y0 + 32), 0.34, (170, 170, 170))
        if np.isfinite(on):
            cv2.line(strip, (X(on), y0), (X(on), y1), col, 2, cv2.LINE_AA)
        ot = (f"{on:.0f}" if np.isfinite(on) else "-") + (
            (" / " + (f"{on_f:.0f}" if np.isfinite(on_f) else "-")) if fx is not None else "")
        vq._txt(strip, f"onset {ot}", (4, y0 + 48), 0.34, col)
    cv2.line(strip, (X(0), 4), (X(0), SH - 24), COL["cue"], 1)
    lk = r.get("first_lick_ms", np.nan)
    if np.isfinite(lk):
        cv2.line(strip, (X(lk), 4), (X(lk), SH - 24), COL["lick"], 1)
    legend = "gray = applied threshold (3-D dark gray = 0.20 mm floor); thick tick = that signal's onset; yellow = cue"
    if r.get("fix_variant"):
        legend = ("gray / colored tick = CURRENT rule; cyan line / lower cyan tick = FIXED rule (" + r["fix_variant"]
                  + " noise + 3-D re-onset); text = current / fixed; yellow = cue")
    vq._txt(strip, legend, (4, SH - 8), 0.34, (200, 200, 200))
    title = f"{r['id']}  mouse {MOUSE[a]}  {r['epoch']}  {r['pos']}"
    out = {}
    kinds = ("reveal_fixed",) if r.get("fix_variant") else ("blind", "reveal")
    for kind in kinds:
        H = TILE_H + (SH if kind != "blind" else 0)
        H += H % 2
        od = Path(folder) / ("clips_reveal" if kind == "reveal_fixed" else f"clips_{kind}")
        od.mkdir(exist_ok=True)
        f_out = od / (f"{r['id']}_reveal_fixed.mp4" if kind == "reveal_fixed" else f"{r['id']}.mp4")
        proc = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo",
                                 "-pix_fmt", "bgr24", "-s", f"{W_}x{H}", "-r", "62.5", "-i", "-", "-c:v", "libx264",
                                 "-pix_fmt", "yuv420p", "-crf", "20", str(f_out)], stdin=subprocess.PIPE)
        for n, tm in enumerate(master):
            im = np.zeros((H, W_, 3), np.uint8)
            x = 0
            for cam in CAMS:
                if cam not in vids:
                    continue
                j, fr = vids[cam]
                src = fr.get(int(j[n]))
                w, h = tiles[cam]
                if src is not None:
                    im[:TILE_H, x:x + w] = cv2.resize(src, (w, h), interpolation=cv2.INTER_AREA)
                vq._txt(im, cam, (x + 6, 18), 0.45)
                x += w
            tms = (tm - cue) * 1000
            vq._txt(im, f"{r['id']}   t = {tms:+.0f} ms from cue (0.25x)", (6, TILE_H - 10), 0.55, (0, 255, 255))
            if kind != "blind":
                vq._txt(im, title, (6, 40), 0.45, (220, 220, 220))
                s = strip.copy()
                if t0 <= tms <= t1:
                    cv2.line(s, (X(tms), 0), (X(tms), SH - 24), (0, 255, 255), 1)
                im[TILE_H:TILE_H + SH] = s
            proc.stdin.write(im.tobytes())
        proc.stdin.close()
        proc.wait()
        out[kind] = f_out.name
    return {"id": r["id"], "status": "ok", "frames": len(master), "cams": ",".join(vids)}


def main(argv=None) -> int:
    from wfield_local import run_mode as _rm_guard
    _rm_guard.require_default("jaw3d_blind_clips")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--fixed", choices=("mad", "cap"), default=None,
                    help="write ONLY clips_reveal/<id>_reveal_fixed.mp4: current AND fixed-rule thresholds / onsets")
    a = ap.parse_args(argv)
    os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
    K = pd.read_csv(a.folder / "key.csv")
    QC = Path("M:/MICROSCOPE/Priya/Widefield/LP_analysis") / RUN / "imaging_encoding" / "qc"
    jcols = ["session", "trial_id", "cam4_moved", "cam1_moved", "cam4_onset_ms", "cam1_onset_ms", "cam4_reonset_ms",
             "cam1_reonset_ms"]
    J = pd.read_csv(QC / "jaw_onsets.csv", low_memory=False, usecols=jcols)
    if a.fixed:
        JQ = REON.parent
        Fx = pd.read_csv(JQ / f"jaw_onsets_{a.fixed}.csv", low_memory=False,
                         usecols=jcols + ["jaw3d_moved", "jaw3d_onset_ms", "jaw3d_at_cue", "jaw3d_reonset_ms"])
        at = Fx.jaw3d_at_cue.astype(str).str.lower().eq("true")
        Fx["jaw3d_ms"] = np.where(Fx.jaw3d_moved == 1, np.where(at, Fx.jaw3d_reonset_ms, Fx.jaw3d_onset_ms), np.nan)
        Fx = Fx.drop(columns=["jaw3d_moved", "jaw3d_onset_ms", "jaw3d_at_cue", "jaw3d_reonset_ms"])
        Fx = Fx.rename(columns={c: f"fix_{c}" for c in Fx.columns if c not in ("session", "trial_id")})
        Rf = pd.read_csv(JQ / f"jaw3d_reonset_trials_{a.fixed}.csv.gz", usecols=["session", "trial_id", "thr_um"])
        Fx = Fx.merge(Rf.rename(columns={"thr_um": "fix_thr_um"}), on=["session", "trial_id"], how="left")
    K = K.drop(columns=[c for c in ("cam4_moved", "cam1_moved") if c in K]).merge(J, on=["session", "trial_id"],
                                                                                    how="left")
    if a.fixed:
        K = K.merge(Fx, on=["session", "trial_id"], how="left")
        K["fix_variant"] = a.fixed
    if REON.exists():
        R = pd.read_csv(REON, usecols=["session", "trial_id", "thr_um", "base_sd_um"])
        K = K.merge(R, on=["session", "trial_id"], how="left")
    from wfield_local import session_licks as SL
    fl = {}
    for lab in sorted(K.label.unique()):
        Lk = SL.session_licks(lab)
        pk = pd.to_numeric(Lk.t_ms, errors="coerce")
        fl.update({(lab, int(t)): float(v) for t, v in Lk[pk >= 70].groupby("trial_id").on_ms.min().items()})
    K["first_lick_ms"] = [fl.get((lab, int(t)), np.nan) for lab, t in zip(K.label, K.trial_id)]
    if a.ids:
        K = K[K.id.isin(a.ids)]
    from wfield_local import parallel
    items = [(r.to_dict(), str(a.folder)) for _, r in K.sort_values("id").iterrows()]
    res, fails = parallel.fan_out(items, render, jobs=a.jobs, label="clip")
    rows = sorted([x[1] for x in res], key=lambda d: d["id"]) + [{"id": it[0]["id"], "status": f"failed: {e}"}
                                                                   for it, e in fails]
    name = "clip_status" + (f"_fixed_{a.fixed}" if a.fixed else "") + ("_subset" if a.ids else "") + ".csv"
    pd.DataFrame(rows).sort_values("id").to_csv(a.folder / name, index=False)
    print(pd.DataFrame(rows).status.value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
