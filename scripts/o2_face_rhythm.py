"""FaceRhythm (face-rhythm 0.3.3) point tracking + VQT spectrograms on HMS O2, one task per session x cam4 video
(docs/FACE_RHYTHM_PLAN.md step 5).

    python scripts/o2_face_rhythm.py bundle --name fr_cam4_20261021 --user ps150 [SESSIONS ...]
    python scripts/o2_face_rhythm.py collect --name fr_cam4_20261021

bundle (this desktop): per session, from the v36 cam4 ``windows.npz`` (``ix_src_frame``, ``ix_t_daq_ms``), the
fractional source-video frame of every trial-grid time (cue -2 .. +4 s on the DAQ clock, 40 ms = the spectrogram's
25 Hz) and a linear source-frame -> DAQ-seconds fit for the whole-session outputs -> ``trials/<a>_<d>_cam4.npz``;
plus params.json, the runner, an sbatch array (CPU, ``short``) and COMMANDS.md -> <DeepLabCut/Widefield>/o2/<name>/.
Default sessions = every curated session with v36 cam4 windows. The ROIs come from
``<dlc>/face_rhythm/cam4/rois/<a>_<d>/ROIs.h5`` (the master mesh warped to the session; made in another window) --
the bundle only records whether each exists yet; COMMANDS.md copies that tree next to the runner.
collect (this desktop): checks every task's outputs (shapes from the npy headers, nothing loaded) and writes
``<dlc>/face_rhythm/cam4/o2_<name>_index.csv`` (paths + violation fractions). The npz files stay in the bundle's
``out/`` (~10-20 GB per session at ~1500 points), not copied again.

Runner (O2, per task): tracking with the paper's values (PARAMS["tracking"]; LK criteria, halting frames and CLAHE as
the 10-21 pilot) in ``segment_frames`` (30 min) segments that each restart at the home positions, so the tracked
points never exceed one segment in RAM; each segment is tracked from ``warmup_frames`` before its start (discarded,
lets the mesh settle and gives the VQT its left margin) to ``margin_frames`` past its end. Each segment's points go
through FR's ``VQT_Analyzer.transform`` in ``chunk_frames`` chunks with ``padding='same'``, so 25 Hz bin j = frames
[10j, 10j+10) exactly; each chunk keeps only the bins it owns. Outputs (float16, ``out/<a>_<d>_cam4_*.npz``):
  trials   spec (trial, time, xy, point, freq) on the DAQ grid ``t_ms`` (linear between the two nearest 25 Hz bins,
           NaN where the trial's window or the video does not reach)
  session  spec (bin, xy, point, freq) at 5 Hz (mean of 5 consecutive 25 Hz bins) + ``t_daq_s``
  points   disp (bin, point, xy) = 25 Hz block mean of tracked - home (``home`` float32 alongside) + ``viol_frac``
           (fraction of points frozen as violating, per bin)
  summary  json: violation fraction (FR's: frames with >= 1 violating point), per segment fps / violations.
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

V36 = "lp_r4_heatmap_20261013__clean_v36"
CAM = "cam4"
T_MS = np.arange(-2000.0, 4000.0 + 1, 40.0)          # trial grid on the DAQ clock (ms from cue); 40 ms = 25 Hz bins
PILOT_POINTS = 1514                                    # 10-21 pilot (point_spacing 9) -- size estimates until ROIs exist
PARAMS = {
    "fps": 250,
    "segment_frames": 30 * 60 * 250,                   # restart at home every 30 min (RAM: frames x points x 8 B)
    "warmup_frames": 2500,                             # 10 s; relaxation 0.01/frame settles in ~100 frames
    "margin_frames": 520,                              # >= half the longest VQT filter (1001 frames at 1 Hz) + 1 bin
    "chunk_frames": 20000,                             # VQT chunk (multiple of 50: 5 Hz blocks align)
    "session_ds_bins": 5,                              # 25 Hz -> 5 Hz
    "reader": {"buffer_size": 500, "prefetch": 1, "posthold": 1, "backend": "decord"},
    "tracking": {
        "params_optical_flow": {"method": "lucas_kanade", "mesh_rigidity": 0.1, "mesh_n_neighbors": 24,
                                "relaxation": 0.01,
                                "kwargs_method": {"winSize": [40, 40], "maxLevel": 3, "criteria": [3, 3, 0.03]}},
        "params_outlier_handling": {"threshold_displacement": 300, "framesHalted_before": 10,
                                    "framesHalted_after": 10},
        "params_clahe": {"clipLimit": 40.0, "tileGridSize": [150, 150]},
    },
    "vqt": {
        "params_VQT": {"Fs_sample": 250, "Q_lowF": 4.0, "Q_highF": 20.0, "F_min": 1.0, "F_max": 60, "n_freq_bins": 36,
                       "window_type": "hann", "symmetry": "center", "taper_asymmetric": True, "downsample_factor": 10,
                       "padding": "same", "fft_conv": True, "fast_length": True, "take_abs": True, "filters": None,
                       "plot_pref": False},
        "batch_size": 10, "normalization_factor": 0.95, "spectrogram_exponent": 1.0, "one_over_f_exponent": 0.5,
    },
}

RUNNER = r'''"""Runs on O2: FaceRhythm tracking + VQT for one session x camera (task = SLURM_ARRAY_TASK_ID). See
scripts/o2_face_rhythm.py in widefield_pipeline for the design; params.json holds every setting."""
import json, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse
import torch
import cv2
import face_rhythm as fr

b = Path(__file__).parent
task = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ["SLURM_ARRAY_TASK_ID"])
T = pd.read_csv(b / "tasks.tsv", sep="\t", dtype=str)
r = T.iloc[task - 1]
P = json.loads((b / "params.json").read_text())
key = f"{r.animal}_{r.date}_{r.cam}"
out = b / "out"; out.mkdir(exist_ok=True)
summ = out / f"{key}_summary.json"
if summ.exists():
    print("done already", summ); sys.exit(0)
nc = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
torch.set_num_threads(nc); cv2.setNumThreads(nc)
t_start = time.time()
try:
    import resource
    rss = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6    # Linux: kB -> GB (peak)
except ImportError:                                                         # Windows smoke tests
    import psutil
    rss = lambda: psutil.Process().memory_info().peak_wset / 1e9

# ROIs: the master mesh warped onto this session (point_positions) + background mask
rp = b / "rois" / f"{r.animal}_{r.date}" / "ROIs.h5"
if not rp.exists():
    sys.exit(f"no ROIs.h5 at {rp} -- run the rois rsync in COMMANDS.md step 1 once it exists")
rois = fr.rois.ROIs(select_mode="file", path_file=str(rp), verbose=1)
if rois.point_positions is None:
    sys.exit(f"{rp} has no point_positions (expected the master mesh warped to this session)")
home = np.ascontiguousarray(rois.point_positions, dtype=np.float32)
NP = len(home)
mask = rois[1] if len(rois) > 1 else rois[0]

R = P["reader"]
reader = fr.helpers.BufferedVideoReader(paths_videos=[str(r.video_o2)], buffer_size=R["buffer_size"],
                                        prefetch=R["prefetch"], posthold=R["posthold"], method_getitem="by_video",
                                        backend=R["backend"], verbose=1)
n_vid = int(reader.num_frames_total)
if n_vid != int(r.n_frames) and os.environ.get("FR_ALLOW_NFRAMES_MISMATCH") != "1":
    sys.exit(f"video has {n_vid} frames, the LP manifest {r.n_frames}: trial frames would not line up")
if tuple(mask.shape) != tuple(reader.frame_height_width):
    sys.exit(f"ROI mask {mask.shape} != video {reader.frame_height_width}")
n = min(n_vid, int(os.environ.get("FR_MAX_FRAMES", n_vid)))                 # FR_MAX_FRAMES: smoke tests only

Tk = P["tracking"]
of = json.loads(json.dumps(Tk["params_optical_flow"]))
of["kwargs_method"] = {**of["kwargs_method"], "winSize": tuple(of["kwargs_method"]["winSize"]),
                       "criteria": tuple(of["kwargs_method"]["criteria"])}
clahe = {**Tk["params_clahe"], "tileGridSize": tuple(Tk["params_clahe"]["tileGridSize"])}
pt = fr.point_tracking.PointTracker(buffered_video_reader=reader, point_positions=home, rois_masks=mask,
                                    contiguous=False, params_optical_flow=of, params_clahe=clahe,
                                    params_outlier_handling=Tk["params_outlier_handling"], verbose=1)
video = pt.videos[0]
HB = int(Tk["params_outlier_handling"]["framesHalted_before"])
Vq = P["vqt"]
vqt = fr.spectral_analysis.VQT_Analyzer(params_VQT=Vq["params_VQT"], batch_size=Vq["batch_size"], device="cpu",
                                        normalization_factor=Vq["normalization_factor"],
                                        spectrogram_exponent=Vq["spectrogram_exponent"],
                                        one_over_f_exponent=Vq["one_over_f_exponent"], verbose=0)
freqs = np.asarray(vqt.vqt_model.freqs, np.float32)
NF = len(freqs)
DS = int(Vq["params_VQT"]["downsample_factor"])
half_k = vqt.vqt_model.filters.shape[-1] // 2
SEG, W, M, CH, D5 = (int(P[k]) for k in ("segment_frames", "warmup_frames", "margin_frames", "chunk_frames",
                                         "session_ds_bins"))
m = int(np.ceil((half_k + 1) / DS)) * DS                                   # VQT margin, whole bins
assert W >= m and M >= m + DS and SEG % (DS * D5) == 0 and CH % (DS * D5) == 0, (W, M, m, SEG, CH)


def fmt(frames):
    return pt._format_decordTorchVideo_for_opticalFlow(vid=frames, mask=pt.mask)


def track(f0, f1):
    """Frames [f0, f1) from the home positions: FR's per-frame step (LK, violation freezing + rewind, mesh,
    relaxation) exactly as PointTracker._track_points_singleVideo, on a frame range. -> points (n, NP, 2) float32,
    violating points per frame."""
    nn = f1 - f0
    pts = np.zeros((nn, NP, 2), np.float32)
    pt.violations_currentVideo = scipy.sparse.lil_matrix((nn, NP), dtype=np.bool_)
    pt._frames_freeze_currentVideo = None
    pt._pointIdx_violations_current = np.zeros(NP, dtype=bool)
    pt._violation_event = False
    frame_prev = fmt(video.get_frames_from_continuous_index(f0))[0]
    points_prev = home.copy()
    i = 0
    video.set_iterator_frame_idx(f0)
    while i < nn:
        for frame in video:
            pt.i_frame = i
            frame_new = fmt(frame[None, ...])[0]
            pts[i] = pt._track_points_singleFrame(frame_new=frame_new, frame_prev=frame_prev, points_prev=points_prev)
            frame_prev, points_prev = frame_new, pts[i]
            if pt._violation_event:
                pt._violation_event = False
                i = max(i - HB, 0)
                frame_prev = fmt(video.get_frames_from_continuous_index(f0 + max(i - 1, 0)))[0]
                video.set_iterator_frame_idx(f0 + i)
                points_prev = pts[i]
                break
            i += 1
            if i >= nn:
                break
    nv = np.asarray(pt.violations_currentVideo.tocsr().sum(1)).ravel().astype(np.int32)
    return pts, nv


# trial grid -> fractional source frames (bundle); 25 Hz bin j is centred on frame 10 j + 4.5
Z = np.load(b / "trials" / f"{key}.npz")
fr_tr = Z["src_frame"]
u = (fr_tr - (DS - 1) / 2) / DS
with np.errstate(invalid="ignore"):
    j0_tr = np.floor(u)
w_tr = (u - j0_tr).astype(np.float32)
n_bins = n // DS
n5 = n_bins // D5
ok_tr = np.isfinite(u) & (j0_tr >= 0) & (j0_tr + 1 < n_bins)
j0_tr = np.where(ok_tr, j0_tr, -1).astype(np.int64)
n_tr, n_t = fr_tr.shape

wd = b / "work" / key; wd.mkdir(parents=True, exist_ok=True)
prog_f = wd / "progress.json"
prog = json.loads(prog_f.read_text()) if prog_f.exists() else {"done": [], "segments": [], "n": n, "n_points": NP}
assert prog["n"] == n and prog["n_points"] == NP, "work/ from a different run; move it aside"
mode = "r+" if prog["done"] else "w+"
mm = lambda f, shape, dt: np.lib.format.open_memmap(wd / f, mode=mode, dtype=dt, shape=shape)
S_tr = mm("trials.npy", (n_tr, n_t, 2, NP, NF), np.float16)
S_se = mm("session.npy", (n5, 2, NP, NF), np.float16)
D_pt = mm("disp.npy", (n_bins, NP, 2), np.float16)
V_pt = mm("viol.npy", (n_bins,), np.float16)
filled = np.load(wd / "filled.npy") if (wd / "filled.npy").exists() and prog["done"] else np.zeros((n_tr, n_t), bool)

edges = list(range(0, n, SEG)) + [n]
if len(edges) > 2 and n - edges[-2] < SEG // 4:                            # fold a short tail into the last segment
    edges.pop(-2)
segs = list(zip(edges[:-1], edges[1:]))
print(f"{key}: {n} frames, {NP} points, {len(segs)} segments, {n_tr} trials; VQT margin {m}", flush=True)

for si, (s, e) in enumerate(segs):
    if si in prog["done"]:
        continue
    a, z = max(s - W, 0), min(e + M, n)
    t0 = time.time()
    pts, nv = track(a, z)
    t_track = time.time() - t0
    for c0 in range(s, e, CH):
        c1 = min(c0 + CH, e)
        jb0, jb1 = c0 // DS, min(c1 // DS, n_bins)
        if jb1 <= jb0:
            continue
        jbx = min(jb1 + 1, n_bins)                                       # + 1 bin for the trial interpolation
        i0, i1 = max(jb0 * DS - m, a), min(jbx * DS + m, z)
        assert i0 % DS == 0, i0
        S, x, _ = vqt.transform(pts[i0 - a:i1 - a], home)               # (2, NP, NF, bins), x in frames from i0
        assert abs(float(x[0]) - (DS - 1) / 2) < 1e-3, x[:3]
        S = S[..., jb0 - i0 // DS: jbx - i0 // DS]                        # global bins [jb0, jbx)
        nb = jb1 - jb0
        q0, q1 = jb0 // D5, min(jb1 // D5, n5)
        if q1 > q0:
            blk = S[..., (q0 * D5 - jb0):(q1 * D5 - jb0)]
            S_se[q0:q1] = blk.reshape(2, NP, NF, q1 - q0, D5).mean(-1).transpose(3, 0, 1, 2).astype(np.float16)
        d = pts[jb0 * DS - a: jb1 * DS - a] - home
        D_pt[jb0:jb1] = d.reshape(nb, DS, NP, 2).mean(1).astype(np.float16)
        V_pt[jb0:jb1] = (nv[jb0 * DS - a: jb1 * DS - a].reshape(nb, DS).mean(1) / NP).astype(np.float16)
        ti, ki = np.nonzero(ok_tr & (j0_tr >= jb0) & (j0_tr < jb1))
        if len(ti):
            jj = j0_tr[ti, ki] - jb0
            ww = w_tr[ti, ki]
            v = S[..., jj] * (1 - ww) + S[..., jj + 1] * ww                 # (2, NP, NF, n_sel)
            S_tr[ti, ki] = v.transpose(3, 0, 1, 2).astype(np.float16)
            filled[ti, ki] = True
        del S
    own = nv[s - a: e - a]
    st = {"segment": si, "start": s, "end": e, "tracked_from": a, "tracked_to": z, "track_fps": (z - a) / t_track,
          "violation_fraction": float((own > 0).mean()), "mean_points_violating": float(own.mean() / NP),
          "minutes": (time.time() - t0) / 60, "peak_rss_gb": rss()}
    del pts, nv
    [A.flush() for A in (S_tr, S_se, D_pt, V_pt)]                          # (no loop name left holding a map)
    np.save(wd / "filled.npy", filled)
    prog["done"].append(si); prog["segments"].append(st)
    prog_f.write_text(json.dumps(prog, indent=1))
    print(f"  seg {si + 1}/{len(segs)}: {st['track_fps']:.0f} fps, violations {st['violation_fraction']:.3f}, "
          f"{st['minutes']:.1f} min, peak {st['peak_rss_gb']:.1f} GB", flush=True)

# unfilled trial samples -> NaN; DAQ times from the bundle's source-frame -> DAQ fit
ti, ki = np.nonzero(~filled)
if len(ti):
    S_tr[ti, ki] = np.float16(np.nan)
    S_tr.flush()
f2d = lambda f: Z["daq_fit"][0] * f + Z["daq_fit"][1]
fc25 = np.arange(n_bins) * DS + (DS - 1) / 2
fc5 = np.arange(n5) * DS * D5 + (DS * D5 - 1) / 2
meta = dict(home=home, freqs=freqs, video=str(r.video), n_frames=n, fps=float(P["fps"]))


def save(name, **arrs):
    dst = out / f"{key}_{name}.npz"
    tmp = dst.with_suffix(".npz.part")
    with open(tmp, "wb") as fh:
        np.savez(fh, **arrs, **meta)
    with np.load(tmp) as chk:
        assert all(chk[k].shape == np.shape(v) for k, v in arrs.items() if np.ndim(v) <= 2)
    os.replace(tmp, dst)
    return dst


save("trials", spec=S_tr, dims="trial,time,xy,point,freq", t_ms=Z["t_ms"].astype(np.float32),
     trial_id=Z["trial_id"], position=Z["position"], cue_daq_s=Z["cue_daq_s"], src_frame=fr_tr, filled=filled)
save("session", spec=S_se, dims="bin,xy,point,freq", frame_center=fc5.astype(np.float32),
     t_daq_s=f2d(fc5), rate_hz=np.float32(P["fps"] / (DS * D5)))
save("points", disp=D_pt, dims="bin,point,xy", viol_frac=V_pt, frame_center=fc25.astype(np.float32),
     t_daq_s=f2d(fc25), rate_hz=np.float32(P["fps"] / DS))
segs_st = prog["segments"]
frames_own = np.array([x["end"] - x["start"] for x in segs_st], float)
res = {"key": key, "video": r.video, "n_frames": n, "n_points": NP, "n_trials": int(n_tr),
       "violation_fraction": float(np.dot(frames_own, [x["violation_fraction"] for x in segs_st]) / frames_own.sum()),
       "mean_points_violating": float(np.dot(frames_own, [x["mean_points_violating"] for x in segs_st])
                                      / frames_own.sum()),
       "trial_samples_filled": float(filled.mean()), "segments": segs_st, "hours": (time.time() - t_start) / 3600,
       "peak_rss_gb": rss(), "face_rhythm": fr.__version__, "params": P}
# the work memmaps are this task's own scratch copies of what is now in out/ -> remove them
del S_tr, S_se, D_pt, V_pt
for f in ("trials.npy", "session.npy", "disp.npy", "viol.npy", "filled.npy", "progress.json"):
    (wd / f).unlink(missing_ok=True)
wd.rmdir()
summ.write_text(json.dumps(res, indent=1))
print(f"saved {key}: violations {res['violation_fraction']:.3f}, trial samples filled "
      f"{res['trial_samples_filled']:.1%}, {res['hours']:.1f} h, peak {res['peak_rss_gb']:.1f} GB", flush=True)
'''


def trial_frames(w) -> dict:
    """Per trial (session order): the fractional source frame at every ``T_MS`` time (np.interp of ix_src_frame on
    ix_t_daq_ms within the trial's window; NaN outside it), cue time on the DAQ clock, position; plus a linear
    source-frame -> DAQ-seconds fit over every window row (for the whole-session outputs)."""
    tid, src = w["ix_trial_id"], w["ix_src_frame"].astype(float)
    tq, daq, pos = w["ix_t_daq_ms"], w["ix_daq_s"], w["ix_position"].astype(str)
    ids, first, inv = np.unique(tid, return_index=True, return_inverse=True)
    order = np.argsort(first)
    rows = np.argsort(inv, kind="stable")
    groups = np.split(rows, np.cumsum(np.bincount(inv))[:-1])
    F = np.full((len(ids), len(T_MS)), np.nan)
    cue = np.full(len(ids), np.nan)
    p = np.empty(len(ids), dtype=object)
    for o, k in enumerate(order):
        g = groups[k]
        g = g[np.argsort(tq[g], kind="stable")]
        t, f = tq[g], src[g]
        keep = np.r_[True, np.diff(t) > 0]
        F[o] = np.interp(T_MS, t[keep], f[keep], left=np.nan, right=np.nan)
        cue[o] = np.median(daq[g] - tq[g] / 1000.0)
        p[o] = pos[g[0]]
    uf, ui = np.unique(src, return_index=True)
    fit = np.polyfit(uf, daq[ui], 1)
    resid_ms = float(np.abs(np.polyval(fit, uf) - daq[ui]).max() * 1000)
    return {"src_frame": F, "trial_id": ids[order], "cue_daq_s": cue, "position": p.astype(str), "t_ms": T_MS,
            "daq_fit": fit, "daq_fit_max_resid_ms": resid_ms}


def default_sessions(ro: Path) -> list[tuple[str, str]]:
    from wfield_local import analysis_kit as ak
    cur = ak.curated_ids()
    return [(p.parent.name, p.name) for p in sorted((ro / V36 / "sessions").glob("*/*"))
            if (p / "windows.npz").exists() and f"{p.parent.name}_{p.name}" in cur]


def rois_dir(rv) -> Path:
    return Path(rv.root("dlc")) / "face_rhythm" / CAM / "rois"


def _n_points(h5: Path) -> int | None:
    try:
        import h5py
        with h5py.File(h5, "r") as f:
            return int(f["point_positions"].shape[0])
    except Exception:                                                  # noqa: BLE001
        return None


def cmd_bundle(name: str, user: str, sessions: list[tuple[str, str]] | None, seg_min: float) -> Path:
    from wfield_local import o2_inference as oi
    from wfield_local import writeguard
    from wfield_local.paths import PathResolver
    rv = PathResolver()
    ro = Path(rv.root("orofacial"))
    d = oi.bundle_dir(rv, name)
    writeguard.assert_writable(d.parent)
    (d / "trials").mkdir(parents=True, exist_ok=True)
    (d / "logs").mkdir(exist_ok=True)
    s = oi.scratch(user)
    rows = []
    for animal, date in sessions or default_sessions(ro):
        sd = ro / V36 / "sessions" / animal / date
        try:
            man = json.loads((sd / "manifest.json").read_text())
            vids = man["videos"]
            if len(vids) != 1:
                raise ValueError(f"{len(vids)} videos (one task = one video; windows' src frames index one)")
            v = vids[0]
            vp = Path(rv.root("behavior_cameras")) / date / animal / v["video"]
            if not vp.exists():
                raise FileNotFoundError(vp)
            tr = trial_frames(np.load(sd / "windows.npz"))
        except Exception as ex:                                        # noqa: BLE001
            print(f"  !! {animal}_{date}: {type(ex).__name__}: {ex} -- skipped")
            continue
        np.savez_compressed(d / "trials" / f"{animal}_{date}_{CAM}.npz", **tr)
        h5 = rois_dir(rv) / f"{animal}_{date}" / "ROIs.h5"
        n_pts = _n_points(h5) if h5.exists() else None
        rows.append({"animal": animal, "date": date, "cam": CAM, "video": v["video"], "n_frames": int(v["n_frames"]),
                     "n_trials": len(tr["trial_id"]), "rois_ready": h5.exists(), "n_points": n_pts or "",
                     "video_share": oi.o2_path(vp, rv),
                     "video_o2": f"{s}/o2_videos/{date}/{animal}/{v['video']}"})
        cov = np.isfinite(tr["src_frame"]).mean()
        print(f"  {animal}_{date}: {v['n_frames']} frames, {len(tr['trial_id'])} trials, grid covered {cov:.1%}, "
              f"DAQ fit resid {tr['daq_fit_max_resid_ms']:.1f} ms, ROIs {'yes' if h5.exists() else 'not yet'}")
    T = pd.DataFrame(rows)
    T.to_csv(d / "tasks.tsv", sep="\t", index=False, lineterminator="\n")
    p = json.loads(json.dumps(PARAMS))
    p["segment_frames"] = int(round(seg_min * 60 * p["fps"] / 50)) * 50
    (d / "params.json").write_text(json.dumps(p, indent=1), encoding="utf-8", newline="\n")
    (d / "fr_runner.py").write_text(RUNNER, encoding="utf-8", newline="\n")
    rel = sorted({"/".join(Path(x).parts[-3:]) for x in T.video_share})
    (d / "videos.txt").write_text("".join(f"{x}\n" for x in rel), encoding="utf-8", newline="\n")

    # sizes (float16) and time, from the pilot: 100 fps tracking at 1514 points, VQT ~3200 frames/s
    npt = pd.to_numeric(T.n_points, errors="coerce").fillna(PILOT_POINTS)
    nf = p["vqt"]["params_VQT"]["n_freq_bins"]
    gb = (T.n_trials * len(T_MS) * 2 * npt * nf * 2 + T.n_frames / 50 * 2 * npt * nf * 2
          + T.n_frames / 10 * npt * 2 * 2) / 1e9
    hours = T.n_frames * (1 + (p["warmup_frames"] + p["margin_frames"]) / p["segment_frames"]) / 100 / 3600 \
        + T.n_frames / 3200 / 3600
    ram = p["segment_frames"] * npt.max() * 8 / 1e9 + 2 * p["reader"]["buffer_size"] * 680 * 680 * 3 / 1e9 + 2.5
    n_ready = int(T.rois_ready.sum())

    b = f"{s}/o2/{name}"
    share_b = oi.o2_path(d, rv)
    cam_root = oi.o2_path(Path(rv.root("behavior_cameras")), rv)
    share_rois = oi.o2_path(rois_dir(rv), rv)
    sb = f"""#!/bin/bash -l
#SBATCH -J fr_{name}
#SBATCH -p short
#SBATCH -c 4
#SBATCH --mem=20G
#SBATCH -t 0-12:00:00
#SBATCH --array=1-{len(T)}%24
#SBATCH -o {b}/logs/%A_%a.out
#SBATCH -e {b}/logs/%A_%a.err
# Expected per task (pilot rates): {hours.min():.1f}-{hours.max():.1f} h, peak ~{ram:.0f} GB. Resumable per 30-min segment:
# a task that hits the time limit continues where it stopped when resubmitted with the same --array index.
set -eo pipefail
module load conda/miniforge3/24.11.3-0
conda activate fr
echo "task $SLURM_ARRAY_TASK_ID on $(hostname), $(python --version)"
python {b}/fr_runner.py
"""
    (d / "run_fr.sbatch").write_text(sb, encoding="utf-8", newline="\n")
    cmds = f"""# O2 FaceRhythm: {name} ({len(T)} tasks = session x {CAM}; ROIs ready for {n_ready})

Per task: {hours.min():.1f}-{hours.max():.1f} h at the pilot's rates, peak RAM ~{ram:.0f} GB, outputs ~{gb.median():.0f} GB
(float16; {gb.sum() / 1000:.1f} TB for all {len(T)}{"" if n_ready == len(T) else f", {PILOT_POINTS} points assumed where no ROIs yet"}).
Scratch per running task: up to 2x its outputs (work memmaps, removed once the npz are written).

## 0. Once: the env  (login node, interactive: srun --pty -p interactive -c 2 --mem 8G -t 1:00:00 bash)
```bash
module load conda/miniforge3/24.11.3-0
conda create -y -n fr python=3.12 && conda activate fr
pip install face-rhythm==0.3.3
python -c "import face_rhythm as fr, decord; print(fr.__version__)"
```

## 1. Transfer node  (ssh {user}@transfer.rc.hms.harvard.edu)
```bash
mkdir -p {b} {s}/o2_videos
rsync -ah {share_b}/ {b}/
rsync -ah --files-from={b}/videos.txt {cam_root}/ {s}/o2_videos/
rsync -ah {share_rois}/ {b}/rois/          # <a>_<d>/ROIs.h5; rerun when more sessions get ROIs
ls {b}/rois | wc -l; wc -l {b}/videos.txt
```
(cam4 videos already on scratch from the LP runs are skipped by rsync.)

## 2. Login node  (ssh {user}@o2.hms.harvard.edu) -- one task first, then the rest
```bash
sbatch --array=1 {b}/run_fr.sbatch
squeue -u {user}
tail -n 5 {b}/logs/*_1.out
```
The log prints one line per 30-min segment (fps, violations, peak RAM). After the first segment, check the fps
against the pilot's 100; when task 1 has written `{b}/out/*_summary.json`:
```bash
sbatch --array=2-{len(T)}%24 {b}/run_fr.sbatch
```
Tasks without ROIs exit at once with "no ROIs.h5"; resubmit their indices after step 1's rois rsync.
Smoke test (optional, ~2 min): `FR_MAX_FRAMES=6000 python {b}/fr_runner.py 1` in an interactive job, then move its
`out/*` aside before the real task 1.

## 3. Transfer node, when all are done
```bash
ls {b}/out/*_summary.json | wc -l          # expect {len(T)}
rsync -ah {b}/out/ {share_b}/out/
```

## 4. Desktop
`python scripts/o2_face_rhythm.py collect --name {name}`
"""
    (d / "COMMANDS.md").write_text(cmds, encoding="utf-8", newline="\n")
    print(f"{len(T)} tasks -> {d}\n{cmds}")
    return d


def _npz_shapes(f: Path) -> dict:
    """Array shapes from the npy headers inside an npz (nothing loaded)."""
    out = {}
    with zipfile.ZipFile(f) as z:
        for nm in z.namelist():
            with z.open(nm) as fh:
                ver = np.lib.format.read_magic(fh)
                rd = np.lib.format.read_array_header_1_0 if ver == (1, 0) else np.lib.format.read_array_header_2_0
                shape = rd(fh)[0]
            out[nm[:-4]] = shape
    return out


def cmd_collect(name: str) -> None:
    from wfield_local import o2_inference as oi
    from wfield_local import writeguard
    from wfield_local.paths import PathResolver
    rv = PathResolver()
    d = oi.bundle_dir(rv, name)
    T = pd.read_csv(d / "tasks.tsv", sep="\t", dtype=str)
    rows = []
    for r in T.itertuples():
        key = f"{r.animal}_{r.date}_{r.cam}"
        sj = d / "out" / f"{key}_summary.json"
        fs = {k: d / "out" / f"{key}_{k}.npz" for k in ("trials", "session", "points")}
        if not sj.exists() or not all(f.exists() for f in fs.values()):
            print(f"  missing {key}")
            continue
        S = json.loads(sj.read_text())
        sh = {k: _npz_shapes(f) for k, f in fs.items()}
        exp_tr = (int(r.n_trials), len(T_MS), 2, S["n_points"])
        if tuple(sh["trials"]["spec"][:4]) != exp_tr:
            print(f"  !! {key}: trials spec {sh['trials']['spec']} != {exp_tr}")
            continue
        rows.append({"animal": r.animal, "date": r.date, "cam": r.cam, "n_points": S["n_points"],
                     "n_trials": int(r.n_trials), "violation_fraction": S["violation_fraction"],
                     "mean_points_violating": S["mean_points_violating"],
                     "trial_samples_filled": S["trial_samples_filled"], "hours": S["hours"],
                     "peak_rss_gb": S["peak_rss_gb"], "session_bins": sh["session"]["spec"][0],
                     **{f"{k}_npz": str(f) for k, f in fs.items()}})
    if not rows:
        print("nothing collected")
        return
    I = pd.DataFrame(rows)
    dst = Path(rv.root("dlc")) / "face_rhythm" / CAM / f"o2_{name}_index.csv"
    writeguard.assert_writable(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    I.to_csv(dst, index=False, float_format="%.4g")
    print(I.groupby("animal")[["violation_fraction", "mean_points_violating", "trial_samples_filled"]]
          .agg(["mean", "max"]).round(3).to_string())
    print(f"collected {len(I)} of {len(T)} -> {dst}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bundle")
    b.add_argument("--name", required=True)
    b.add_argument("--user", required=True)
    b.add_argument("--seg-min", type=float, default=30.0, help="tracking segment (restart at home), minutes")
    b.add_argument("sessions", nargs="*", help="animal:YYYYMMDD (default: every curated session with v36 windows)")
    c = sub.add_parser("collect")
    c.add_argument("--name", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "bundle":
        ss = [tuple(x.split(":")[:2]) for x in a.sessions] if a.sessions else None
        cmd_bundle(a.name, a.user, ss, a.seg_min)
    else:
        cmd_collect(a.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
