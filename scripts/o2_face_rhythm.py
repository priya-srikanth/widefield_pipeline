"""FaceRhythm (face-rhythm 0.3.3) point tracking + VQT spectrograms on HMS O2, one task per session x cam4 video
(docs/FACE_RHYTHM_PLAN.md step 5).

    python scripts/o2_face_rhythm.py bundle --name fr_cam4_20261021 --user ps150 [SESSIONS ...]
    python scripts/o2_face_rhythm.py collect --name fr_cam4_20261021

bundle (this desktop; sessions fanned over cores, `analysis_kit.fan_sessions`): per session, from the v36 cam4
``windows.npz`` (``ix_src_frame``, ``ix_t_daq_ms``), the fractional source-video frame of every trial-grid time (cue
-2 .. +4 s on the DAQ clock, 40 ms = the spectrogram's 25 Hz) and a linear source-frame -> DAQ-seconds fit for the
whole-session outputs -> ``trials/<a>_<d>_cam4.npz``; plus params.json, the runner, an sbatch array (CPU, ``short``)
and COMMANDS.md -> <DeepLabCut/Widefield>/o2/<name>/. Default sessions = every curated session with v36 cam4 windows;
trial windows for ENGAGED trials only (the cohort tables' gate, `orofacial_cohort.engagement_keep`, keeps ~90 %;
``--all-trials`` for every trial). The ROIs come from ``<dlc>/face_rhythm/cam4/rois/<a>_<d>/ROIs.h5`` (the master
mesh warped to the session; made in another window) -- the bundle only records whether each exists yet; COMMANDS.md
copies that tree next to the runner.
collect (this desktop): checks every task's outputs (shapes from the npy headers, nothing loaded) and writes
``<dlc>/face_rhythm/cam4/o2_<name>_index.csv`` (paths + violation fractions). The npz files stay in the bundle's
``out/`` (~10-30 GB per session at ~1500 points), not copied again.

Runner (O2, per task): tracking with the paper's values (PARAMS["tracking"]; LK criteria, halting frames and CLAHE as
the 10-21 pilot) in ``segment_frames`` (10 min) segments that each restart at the home positions. Segments are
therefore independent and run in ``workers`` parallel processes (FR's tracking loop is one Python thread per frame,
so one process cannot use more than ~1-2 cores); each worker writes only the bins its segment owns into shared
float16 memmaps (``work/<key>/``), the parent records finished segments (``progress.json``: a resubmitted task resumes)
and writes the npz outputs at the end. Each segment is tracked from ``warmup_frames`` before its start (discarded:
lets the mesh settle, gives the VQT its left margin) to ``margin_frames`` past its end; its points go through FR's
``VQT_Analyzer.transform`` in ``chunk_frames`` chunks with ``padding='same'``, so 25 Hz bin j = frames [10j, 10j+10)
exactly and no full-length spectrogram is ever held. Outputs (float16, ``out/<a>_<d>_cam4_*.npz``):
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
PILOT_FPS, PILOT_VQT_FPS = 100.0, 3200.0               # 10-21 pilot, one process, 1514 points
PARAMS = {
    "fps": 250,
    "workers": 4,                                      # parallel segments per task (2 threads each)
    "segment_frames": 10 * 60 * 250,                   # restart at home every 10 min (RAM/worker: frames x points x 8 B)
    "warmup_frames": 2500,                             # 10 s; relaxation 0.01/frame settles in ~100 frames
    "margin_frames": 520,                              # >= half the longest VQT filter (1001 frames at 1 Hz) + 1 bin
    "chunk_frames": 10000,                             # VQT chunk (multiple of 50: 5 Hz blocks align)
    "session_ds_bins": 5,                              # 25 Hz -> 5 Hz
    "reader": {"buffer_size": 250, "prefetch": 1, "posthold": 1, "backend": "decord"},
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

RUNNER = r'''"""Runs on O2: FaceRhythm tracking + VQT for one session x camera (task = SLURM_ARRAY_TASK_ID).

Segments restart at the home positions, so they are independent and run in parallel worker processes (params
"workers", or FR_WORKERS; threads per worker = SLURM_CPUS_PER_TASK // workers). Each worker writes only the bins its
segment owns into shared float16 memmaps in work/<key>/; the parent records finished segments in progress.json (a
resubmitted task resumes) and writes the npz outputs. Design: scripts/o2_face_rhythm.py in widefield_pipeline."""
import json, os, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np

B = Path(__file__).resolve().parent
S = None                                                                    # this process's state (`setup`)


def peak_gb():
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6    # Linux: kB -> GB
    except ImportError:                                                     # Windows smoke tests
        import psutil
        return psutil.Process().memory_info().peak_wset / 1e9


def task_row(task):
    import pandas as pd
    return pd.read_csv(B / "tasks.tsv", sep="\t", dtype=str).iloc[task - 1]


def setup(task, threads, mode="r+"):
    """One process's state: ROIs, video reader, FR tracker + VQT, derived sizes, trial grid, work memmaps
    (``mode="w+"`` creates them -- the parent, before any worker starts)."""
    import cv2
    import torch
    import face_rhythm as fr
    torch.set_num_threads(threads)
    cv2.setNumThreads(threads)
    r = task_row(task)
    P = json.loads((B / "params.json").read_text())
    s = SimpleNamespace(r=r, P=P, key=f"{r.animal}_{r.date}_{r.cam}")
    rp = B / "rois" / f"{r.animal}_{r.date}" / "ROIs.h5"
    if not rp.exists():
        sys.exit(f"no ROIs.h5 at {rp} -- run the rois rsync in COMMANDS.md step 1 once it exists")
    rois = fr.rois.ROIs(select_mode="file", path_file=str(rp), verbose=0)
    if rois.point_positions is None:
        sys.exit(f"{rp} has no point_positions (expected the master mesh warped to this session)")
    s.home = np.ascontiguousarray(rois.point_positions, dtype=np.float32)
    s.NP = len(s.home)
    mask = rois[1] if len(rois) > 1 else rois[0]
    R = P["reader"]
    reader = fr.helpers.BufferedVideoReader(paths_videos=[str(r.video_o2)], buffer_size=R["buffer_size"],
                                            prefetch=R["prefetch"], posthold=R["posthold"], method_getitem="by_video",
                                            backend=R["backend"], verbose=0)
    n_vid = int(reader.num_frames_total)
    if n_vid != int(r.n_frames) and os.environ.get("FR_ALLOW_NFRAMES_MISMATCH") != "1":
        sys.exit(f"video has {n_vid} frames, the LP manifest {r.n_frames}: trial frames would not line up")
    if tuple(mask.shape) != tuple(reader.frame_height_width):
        sys.exit(f"ROI mask {mask.shape} != video {reader.frame_height_width}")
    s.n = min(n_vid, int(os.environ.get("FR_MAX_FRAMES", n_vid)))          # FR_MAX_FRAMES: smoke tests only
    Tk = P["tracking"]
    of = json.loads(json.dumps(Tk["params_optical_flow"]))
    of["kwargs_method"] = {**of["kwargs_method"], "winSize": tuple(of["kwargs_method"]["winSize"]),
                           "criteria": tuple(of["kwargs_method"]["criteria"])}
    clahe = {**Tk["params_clahe"], "tileGridSize": tuple(Tk["params_clahe"]["tileGridSize"])}
    s.pt = fr.point_tracking.PointTracker(buffered_video_reader=reader, point_positions=s.home, rois_masks=mask,
                                          contiguous=False, params_optical_flow=of, params_clahe=clahe,
                                          params_outlier_handling=Tk["params_outlier_handling"], verbose=0)
    s.video = s.pt.videos[0]
    s.HB = int(Tk["params_outlier_handling"]["framesHalted_before"])
    Vq = P["vqt"]
    s.vqt = fr.spectral_analysis.VQT_Analyzer(params_VQT=Vq["params_VQT"], batch_size=Vq["batch_size"],
                                              device="cpu", normalization_factor=Vq["normalization_factor"],
                                              spectrogram_exponent=Vq["spectrogram_exponent"],
                                              one_over_f_exponent=Vq["one_over_f_exponent"], verbose=0)
    s.freqs = np.asarray(s.vqt.vqt_model.freqs, np.float32)
    s.NF = len(s.freqs)
    s.DS = int(Vq["params_VQT"]["downsample_factor"])
    s.SEG, s.W, s.M, s.CH, s.D5 = (int(P[k]) for k in ("segment_frames", "warmup_frames", "margin_frames",
                                                       "chunk_frames", "session_ds_bins"))
    s.m = int(np.ceil((s.vqt.vqt_model.filters.shape[-1] // 2 + 1) / s.DS)) * s.DS   # VQT margin, whole bins
    assert s.W >= s.m and s.M >= s.m + s.DS and s.SEG % (s.DS * s.D5) == 0 and s.CH % (s.DS * s.D5) == 0
    s.n_bins = s.n // s.DS
    s.n5 = s.n_bins // s.D5
    # trial grid -> fractional source frames (bundle); 25 Hz bin j is centred on frame 10 j + 4.5
    with np.load(B / "trials" / f"{s.key}.npz") as Z:
        s.Z = {k: Z[k] for k in Z.files}
    u = (s.Z["src_frame"] - (s.DS - 1) / 2) / s.DS
    with np.errstate(invalid="ignore"):
        j0 = np.floor(u)
    s.w_tr = (u - j0).astype(np.float32)
    s.ok_tr = np.isfinite(u) & (j0 >= 0) & (j0 + 1 < s.n_bins)
    s.j0_tr = np.where(s.ok_tr, j0, -1).astype(np.int64)
    s.wd = B / "work" / s.key
    s.wd.mkdir(parents=True, exist_ok=True)

    def mm(f, shape, dt):
        if mode == "w+":
            return np.lib.format.open_memmap(s.wd / f, mode="w+", dtype=dt, shape=shape)
        a = np.lib.format.open_memmap(s.wd / f, mode="r+")
        assert a.shape == shape, (f, a.shape, shape)
        return a

    n_tr, n_t = u.shape
    s.S_tr = mm("trials.npy", (n_tr, n_t, 2, s.NP, s.NF), np.float16)
    s.S_se = mm("session.npy", (s.n5, 2, s.NP, s.NF), np.float16)
    s.D_pt = mm("disp.npy", (s.n_bins, s.NP, 2), np.float16)
    s.V_pt = mm("viol.npy", (s.n_bins,), np.float16)
    edges = list(range(0, s.n, s.SEG)) + [s.n]
    if len(edges) > 2 and s.n - edges[-2] < s.SEG // 4:                    # fold a short tail into the last segment
        edges.pop(-2)
    s.segs = list(zip(edges[:-1], edges[1:]))
    return s


def init_worker(task, threads):
    global S
    S = setup(task, threads, mode="r+")


def fmt(frames):
    return S.pt._format_decordTorchVideo_for_opticalFlow(vid=frames, mask=S.pt.mask)


def track(f0, f1):
    """Frames [f0, f1) from the home positions: FR's per-frame step (LK, violation freezing + rewind, mesh,
    relaxation) exactly as PointTracker._track_points_singleVideo, on a frame range. -> points (n, NP, 2) float32,
    violating points per frame."""
    import scipy.sparse
    pt, video, NP = S.pt, S.video, S.NP
    nn = f1 - f0
    pts = np.zeros((nn, NP, 2), np.float32)
    pt.violations_currentVideo = scipy.sparse.lil_matrix((nn, NP), dtype=np.bool_)
    pt._frames_freeze_currentVideo = None
    pt._pointIdx_violations_current = np.zeros(NP, dtype=bool)
    pt._violation_event = False
    frame_prev = fmt(video.get_frames_from_continuous_index(f0))[0]
    points_prev = S.home.copy()
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
                i = max(i - S.HB, 0)
                frame_prev = fmt(video.get_frames_from_continuous_index(f0 + max(i - 1, 0)))[0]
                video.set_iterator_frame_idx(f0 + i)
                points_prev = pts[i]
                break
            i += 1
            if i >= nn:
                break
    video.delete_all_slots()
    nv = np.asarray(pt.violations_currentVideo.tocsr().sum(1)).ravel().astype(np.int32)
    return pts, nv


def run_segment(si):
    """Track segment si, VQT it chunk by chunk, write the bins it owns. -> (si, stats, filled trial samples)."""
    s0, e = S.segs[si]
    DS, D5, NP, NF, m, home = S.DS, S.D5, S.NP, S.NF, S.m, S.home
    a, z = max(s0 - S.W, 0), min(e + S.M, S.n)
    t0 = time.time()
    pts, nv = track(a, z)
    t_track = time.time() - t0
    fti, fki = [], []
    for c0 in range(s0, e, S.CH):
        c1 = min(c0 + S.CH, e)
        jb0, jb1 = c0 // DS, min(c1 // DS, S.n_bins)
        if jb1 <= jb0:
            continue
        jbx = min(jb1 + 1, S.n_bins)                                     # + 1 bin for the trial interpolation
        i0, i1 = max(jb0 * DS - m, a), min(jbx * DS + m, z)
        assert i0 % DS == 0, i0
        X, x, _ = S.vqt.transform(pts[i0 - a:i1 - a], home)             # (2, NP, NF, bins), x in frames from i0
        assert abs(float(x[0]) - (DS - 1) / 2) < 1e-3, x[:3]
        X = X[..., jb0 - i0 // DS: jbx - i0 // DS]                        # global bins [jb0, jbx)
        nb = jb1 - jb0
        q0, q1 = jb0 // D5, min(jb1 // D5, S.n5)
        if q1 > q0:
            blk = X[..., (q0 * D5 - jb0):(q1 * D5 - jb0)]
            S.S_se[q0:q1] = blk.reshape(2, NP, NF, q1 - q0, D5).mean(-1).transpose(3, 0, 1, 2).astype(np.float16)
        d = pts[jb0 * DS - a: jb1 * DS - a] - home
        S.D_pt[jb0:jb1] = d.reshape(nb, DS, NP, 2).mean(1).astype(np.float16)
        S.V_pt[jb0:jb1] = (nv[jb0 * DS - a: jb1 * DS - a].reshape(nb, DS).mean(1) / NP).astype(np.float16)
        ti, ki = np.nonzero(S.ok_tr & (S.j0_tr >= jb0) & (S.j0_tr < jb1))
        if len(ti):
            jj = S.j0_tr[ti, ki] - jb0
            ww = S.w_tr[ti, ki]
            v = X[..., jj] * (1 - ww) + X[..., jj + 1] * ww                 # (2, NP, NF, n_sel)
            S.S_tr[ti, ki] = v.transpose(3, 0, 1, 2).astype(np.float16)
            fti.append(ti)
            fki.append(ki)
        del X
    [A.flush() for A in (S.S_tr, S.S_se, S.D_pt, S.V_pt)]
    own = nv[s0 - a: e - a]
    st = {"segment": si, "start": s0, "end": e, "tracked_from": a, "tracked_to": z, "track_fps": (z - a) / t_track,
          "violation_fraction": float((own > 0).mean()), "mean_points_violating": float(own.mean() / NP),
          "minutes": (time.time() - t0) / 60, "worker_peak_rss_gb": peak_gb()}
    cat = lambda L: np.concatenate(L) if L else np.zeros(0, np.int64)
    return si, st, cat(fti), cat(fki)


def save(out, name, meta, **arrs):
    dst = out / f"{S.key}_{name}.npz"
    tmp = dst.with_suffix(".npz.part")
    with open(tmp, "wb") as fh:
        np.savez(fh, **arrs, **meta)                                       # streams memmaps; uncompressed (float16)
    os.replace(tmp, dst)


def main():
    global S
    task = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ["SLURM_ARRAY_TASK_ID"])
    r = task_row(task)
    key = f"{r.animal}_{r.date}_{r.cam}"
    out = B / "out"
    out.mkdir(exist_ok=True)
    summ = out / f"{key}_summary.json"
    if summ.exists():
        print("done already", summ)
        return
    P = json.loads((B / "params.json").read_text())
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
    nw = max(1, int(os.environ.get("FR_WORKERS", P["workers"])))
    threads = max(1, cpus // nw)
    t_start = time.time()
    wd = B / "work" / key
    prog_f = wd / "progress.json"
    prog = json.loads(prog_f.read_text()) if prog_f.exists() else {"done": []}
    resume = bool(prog["done"])
    S = setup(task, threads if nw == 1 else 1, mode="r+" if resume else "w+")
    if not resume:
        prog = {"done": [], "segments": [], "n": S.n, "n_points": S.NP}
    assert prog["n"] == S.n and prog["n_points"] == S.NP, "work/ is from a different run; move it aside"
    shape_tr = S.Z["src_frame"].shape
    filled = np.load(wd / "filled.npy") if resume and (wd / "filled.npy").exists() else np.zeros(shape_tr, bool)
    todo = [i for i in range(len(S.segs)) if i not in prog["done"]]
    print(f"{key}: {S.n} frames, {S.NP} points, {len(S.segs)} segments ({len(todo)} to do), {shape_tr[0]} trials, "
          f"{min(nw, max(len(todo), 1))} workers x {threads} threads", flush=True)

    def record(res):
        si, st, ti, ki = res
        filled[ti, ki] = True
        np.save(wd / "filled.npy", filled)
        prog["done"].append(si)
        prog["segments"].append(st)
        prog_f.write_text(json.dumps(prog, indent=1))
        print(f"  seg {si + 1}/{len(S.segs)}: {st['track_fps']:.0f} fps, violations {st['violation_fraction']:.3f}, "
              f"{st['minutes']:.1f} min, worker peak {st['worker_peak_rss_gb']:.1f} GB "
              f"[{len(prog['done'])}/{len(S.segs)} done, {(time.time() - t_start) / 60:.0f} min]", flush=True)

    if nw == 1 or len(todo) <= 1:
        import cv2
        import torch
        torch.set_num_threads(cpus)
        cv2.setNumThreads(cpus)
        for si in todo:
            record(run_segment(si))
    else:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(min(nw, len(todo)), initializer=init_worker,
                                          initargs=(task, threads)) as pool:
            for res in pool.imap_unordered(run_segment, todo):
                record(res)

    # unfilled trial samples -> NaN; DAQ times from the bundle's source-frame -> DAQ fit
    ti, ki = np.nonzero(~filled)
    if len(ti):
        S.S_tr[ti, ki] = np.float16(np.nan)
        S.S_tr.flush()
    Z, DS, D5 = S.Z, S.DS, S.D5
    f2d = lambda f: Z["daq_fit"][0] * f + Z["daq_fit"][1]
    fc25 = np.arange(S.n_bins) * DS + (DS - 1) / 2
    fc5 = np.arange(S.n5) * DS * D5 + (DS * D5 - 1) / 2
    meta = dict(home=S.home, freqs=S.freqs, video=str(r.video), n_frames=S.n, fps=float(P["fps"]))
    t_w = time.time()
    save(out, "trials", meta, spec=S.S_tr, dims="trial,time,xy,point,freq", t_ms=Z["t_ms"].astype(np.float32),
         trial_id=Z["trial_id"], position=Z["position"], cue_daq_s=Z["cue_daq_s"], src_frame=Z["src_frame"],
         filled=filled, **{k: Z[k] for k in ("gate", "n_trials_all") if k in Z})
    save(out, "session", meta, spec=S.S_se, dims="bin,xy,point,freq", frame_center=fc5.astype(np.float32),
         t_daq_s=f2d(fc5), rate_hz=np.float32(P["fps"] / (DS * D5)))
    save(out, "points", meta, disp=S.D_pt, dims="bin,point,xy", viol_frac=S.V_pt,
         frame_center=fc25.astype(np.float32), t_daq_s=f2d(fc25), rate_hz=np.float32(P["fps"] / DS))
    segs = prog["segments"]
    fo = np.array([x["end"] - x["start"] for x in segs], float)
    wavg = lambda k: float(np.dot(fo, [x[k] for x in segs]) / fo.sum())
    res = {"key": key, "video": r.video, "n_frames": S.n, "n_points": S.NP, "n_trials": int(shape_tr[0]),
           "violation_fraction": wavg("violation_fraction"), "mean_points_violating": wavg("mean_points_violating"),
           "trial_samples_filled": float(filled.mean()), "segments": segs, "workers": nw, "threads": threads,
           "hours": (time.time() - t_start) / 3600, "write_minutes": (time.time() - t_w) / 60,
           "peak_rss_gb": peak_gb() + min(nw, len(S.segs)) * max(x["worker_peak_rss_gb"] for x in segs),
           "face_rhythm": __import__("face_rhythm").__version__, "params": P}
    # the work memmaps are this task's own scratch copies of what is now in out/ -> remove them
    for k in ("S_tr", "S_se", "D_pt", "V_pt"):
        delattr(S, k)
    import gc
    gc.collect()
    for f in ("trials.npy", "session.npy", "disp.npy", "viol.npy", "filled.npy", "progress.json"):
        (wd / f).unlink(missing_ok=True)
    wd.rmdir()
    summ.write_text(json.dumps(res, indent=1))
    print(f"saved {key}: violations {res['violation_fraction']:.3f}, trial samples filled "
          f"{res['trial_samples_filled']:.1%}, {res['hours']:.2f} h (write {res['write_minutes']:.1f} min), "
          f"peak ~{res['peak_rss_gb']:.1f} GB", flush=True)


if __name__ == "__main__":
    main()
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


def engaged_only(tr: dict, animal: str, date: str, sd: Path) -> tuple[dict, str]:
    """``tr`` cut to the engaged trials (`orofacial_cohort.engagement_keep`, the cohort tables' gate: the behaviour
    pipeline's `engaged` flag, ids matched to ours by id / DAQ cue time / order). Ungated (all trials) when the session
    has no usable flag; the label says which."""
    from wfield_local import orofacial_cohort as oc
    keep, label = oc.engagement_keep(animal, date, sd)
    if keep is None:
        return tr, label
    m = np.isin(tr["trial_id"], sorted(keep))
    return {k: (v[m] if k in ("src_frame", "trial_id", "cue_daq_s", "position") else v) for k, v in tr.items()}, label


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


def bundle_session(job) -> dict:
    """One session's trial grid (-> ``trials/``) and tasks.tsv row; raises on anything that makes it unrunnable.
    Module-level for `fan_sessions` (spawn pickles by name)."""
    from wfield_local import o2_inference as oi
    from wfield_local.paths import PathResolver
    animal, date, all_trials, d, s = job
    rv = PathResolver()
    sd = Path(rv.root("orofacial")) / V36 / "sessions" / animal / date
    vids = json.loads((sd / "manifest.json").read_text())["videos"]
    if len(vids) != 1:
        raise ValueError(f"{len(vids)} videos (one task = one video; windows' src frames index one)")
    v = vids[0]
    vp = Path(rv.root("behavior_cameras")) / date / animal / v["video"]
    if not vp.exists():
        raise FileNotFoundError(vp)
    tr = trial_frames(np.load(sd / "windows.npz"))
    n_all = len(tr["trial_id"])
    tr, gate = (tr, "off (all trials)") if all_trials else engaged_only(tr, animal, date, sd)
    np.savez_compressed(Path(d) / "trials" / f"{animal}_{date}_{CAM}.npz", gate=gate, n_trials_all=n_all, **tr)
    h5 = rois_dir(rv) / f"{animal}_{date}" / "ROIs.h5"
    n_pts = _n_points(h5) if h5.exists() else None
    return {"animal": animal, "date": date, "cam": CAM, "video": v["video"], "n_frames": int(v["n_frames"]),
            "n_trials": len(tr["trial_id"]), "n_trials_all": n_all, "gate": gate, "rois_ready": h5.exists(),
            "n_points": n_pts or "", "video_share": oi.o2_path(vp, rv),
            "video_o2": f"{s}/o2_videos/{date}/{animal}/{v['video']}",
            "grid_covered": float(np.isfinite(tr["src_frame"]).mean()),
            "daq_fit_resid_ms": tr["daq_fit_max_resid_ms"]}


def estimates(T: pd.DataFrame, p: dict) -> dict:
    """Per-task hours, peak RAM and output GB at the pilot's rates (``PILOT_POINTS`` where no ROIs yet)."""
    npt = pd.to_numeric(T.n_points, errors="coerce").fillna(PILOT_POINTS)
    nf, nw = p["vqt"]["params_VQT"]["n_freq_bins"], p["workers"]
    gb = (T.n_trials * len(T_MS) * 2 * npt * nf * 2 + T.n_frames / 50 * 2 * npt * nf * 2
          + T.n_frames / 10 * npt * 2 * 2) / 1e9
    ovh = 1 + (p["warmup_frames"] + p["margin_frames"]) / p["segment_frames"]
    scale = npt / PILOT_POINTS
    hours = (T.n_frames * ovh * scale / PILOT_FPS + T.n_frames * scale / PILOT_VQT_FPS) / nw / 3600 \
        + gb / 0.2 / 3600                                                # final npz write at ~200 MB/s
    seg = p["segment_frames"] * 1.25 + p["warmup_frames"] + p["margin_frames"]
    worker = (seg * npt.max() * 8 + 3 * p["reader"]["buffer_size"] * 680 * 680 * 3
              + 3 * 2 * npt.max() * nf * (p["chunk_frames"] / 10 + 110) * 4) / 1e9 + 0.8
    return {"gb": gb, "hours": hours, "ram": nw * worker + 1.5, "worker": worker}


def cmd_bundle(name: str, user: str, sessions: list[tuple[str, str]] | None, seg_min: float,
               all_trials: bool = False, workers: int = 4, jobs: int = 4) -> Path:
    from wfield_local import analysis_kit as ak
    from wfield_local import o2_inference as oi
    from wfield_local import writeguard
    from wfield_local.paths import PathResolver
    rv = PathResolver()
    d = oi.bundle_dir(rv, name)
    writeguard.assert_writable(d.parent)
    (d / "trials").mkdir(parents=True, exist_ok=True)
    (d / "logs").mkdir(exist_ok=True)
    s = oi.scratch(user)
    ss = sessions or default_sessions(Path(rv.root("orofacial")))
    res, fails = ak.fan_sessions([(a, dt, all_trials, str(d), s) for a, dt in ss], bundle_session, jobs=jobs,
                                 key=lambda j: (j[0], j[1]))
    for j, e in fails:
        print(f"  !! {j[0]}_{j[1]}: {e} -- skipped")
    T = pd.DataFrame([row for _, row in res])
    for row in T.itertuples():
        print(f"  {row.animal}_{row.date}: {row.n_frames} frames, {row.n_trials}/{row.n_trials_all} trials "
              f"[{row.gate}], grid covered {row.grid_covered:.1%}, DAQ fit resid {row.daq_fit_resid_ms:.1f} ms, "
              f"ROIs {'yes' if row.rois_ready else 'not yet'}")
    T.drop(columns=["grid_covered", "daq_fit_resid_ms"]).to_csv(d / "tasks.tsv", sep="\t", index=False,
                                                                lineterminator="\n")
    p = json.loads(json.dumps(PARAMS))
    p["segment_frames"] = int(round(seg_min * 60 * p["fps"] / 50)) * 50
    p["workers"] = int(workers)
    (d / "params.json").write_text(json.dumps(p, indent=1), encoding="utf-8", newline="\n")
    (d / "fr_runner.py").write_text(RUNNER, encoding="utf-8", newline="\n")
    rel = sorted({"/".join(Path(x).parts[-3:]) for x in T.video_share})
    (d / "videos.txt").write_text("".join(f"{x}\n" for x in rel), encoding="utf-8", newline="\n")

    E = estimates(T, p)
    hours, gb = E["hours"], E["gb"]
    t_lim = int(min(12, max(3, np.ceil(2.5 * hours.max()))))            # pilot rates are a desktop guess: 2.5x room
    cpus = 2 * p["workers"]
    mem = int(min(64, np.ceil(E["ram"] * 1.25 / 4) * 4))
    n_ready = int(T.rois_ready.sum())
    b = f"{s}/o2/{name}"
    share_b = oi.o2_path(d, rv)
    cam_root = oi.o2_path(Path(rv.root("behavior_cameras")), rv)
    share_rois = oi.o2_path(rois_dir(rv), rv)
    sb = f"""#!/bin/bash -l
#SBATCH -J fr_{name}
#SBATCH -p short
#SBATCH -c {cpus}
#SBATCH --mem={mem}G
#SBATCH -t 0-{t_lim:02d}:00:00
#SBATCH --array=1-{len(T)}%24
#SBATCH -o {b}/logs/%A_%a.out
#SBATCH -e {b}/logs/%A_%a.err
# {p['workers']} segment workers x 2 threads. Expected per task at the pilot's rates: {hours.min():.1f}-{hours.max():.1f} h,
# peak ~{E['ram']:.0f} GB. Resumable per segment: a task that hits the limit continues where it stopped when
# resubmitted with the same --array index.
set -eo pipefail
module load conda/miniforge3/24.11.3-0
conda activate fr
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
echo "task $SLURM_ARRAY_TASK_ID on $(hostname), $(python --version)"
python {b}/fr_runner.py
"""
    (d / "run_fr.sbatch").write_text(sb, encoding="utf-8", newline="\n")
    cmds = f"""# O2 FaceRhythm: {name} ({len(T)} tasks = session x {CAM}; ROIs ready for {n_ready})

Per task ({p['workers']} parallel 10-min segments x 2 threads = {cpus} CPUs): {hours.min():.1f}-{hours.max():.1f} h at the
pilot's rates, peak RAM ~{E['ram']:.0f} GB, outputs ~{gb.median():.0f} GB (float16; {gb.sum() / 1000:.1f} TB for all
{len(T)}{"" if n_ready == len(T) else f", {PILOT_POINTS} points assumed where no ROIs yet"}). Scratch per running task:
up to 2x its outputs (work memmaps, removed once the npz are written).

## 0. Once: the env  (interactive job: srun --pty -p interactive -c 2 --mem 8G -t 1:00:00 bash)
CPU-only torch (the default wheel pulls ~4 GB of CUDA libraries, and pip's cache filled /home on 10-21):
```bash
module load conda/miniforge3/24.11.3-0
conda create -y -n fr python=3.12 && conda activate fr
export PIP_CACHE_DIR={s}/pip_cache TMPDIR={s}/tmp && mkdir -p $PIP_CACHE_DIR $TMPDIR
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install face-rhythm==0.3.3
python -c "import face_rhythm as fr, decord, torch; print(fr.__version__, torch.__version__)"
```

## 1. Transfer node  (ssh {user}@transfer.rc.hms.harvard.edu)
```bash
mkdir -p {b} {s}/o2_videos
rsync -ah {share_b}/ {b}/
rsync -ah --files-from={b}/videos.txt {cam_root}/ {s}/o2_videos/
rsync -ah {share_rois}/ {b}/rois/          # <a>_<d>/ROIs.h5; fails until the folder exists; rerun as ROIs arrive
ls {b}/rois | wc -l; wc -l {b}/videos.txt
```
(cam4 videos already on scratch from the LP runs are skipped by rsync.)

## 2. Login node  (ssh {user}@o2.hms.harvard.edu) -- one task first, then the rest
```bash
sbatch --array=1 {b}/run_fr.sbatch          # or the row (1-based, tasks.tsv) of the first session with ROIs
squeue -u {user}
tail -n 5 {b}/logs/*_1.out
```
The log prints one line per finished segment (fps per worker, violations, worker peak RAM). Check the fps against
the pilot's 100 and `sacct -j <jobid> --format=MaxRSS,Elapsed,TotalCPU` against the request; when task 1 has written
`{b}/out/*_summary.json`:
```bash
sbatch --array=2-{len(T)}%24 {b}/run_fr.sbatch
```
Tasks without ROIs exit at once with "no ROIs.h5"; resubmit their indices after step 1's rois rsync.
Smoke test (optional, ~3 min): `FR_MAX_FRAMES=6000 FR_WORKERS=1 python {b}/fr_runner.py 1` in an interactive job,
then move its `out/*` aside before the real task 1.

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
                     "n_trials": int(r.n_trials), "gate": getattr(r, "gate", ""),
                     "violation_fraction": S["violation_fraction"],
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
    b.add_argument("--seg-min", type=float, default=10.0, help="tracking segment (restart at home), minutes")
    b.add_argument("--workers", type=int, default=4, help="parallel segments per O2 task (2 CPUs each)")
    b.add_argument("--jobs", type=int, default=4, help="sessions in parallel on this desktop (shared box: keep low)")
    b.add_argument("--all-trials", action="store_true",
                   help="trial windows for every trial (default: engaged trials only, the cohort tables' gate)")
    b.add_argument("sessions", nargs="*", help="animal:YYYYMMDD (default: every curated session with v36 windows)")
    c = sub.add_parser("collect")
    c.add_argument("--name", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "bundle":
        ss = [tuple(x.split(":")[:2]) for x in a.sessions] if a.sessions else None
        cmd_bundle(a.name, a.user, ss, a.seg_min, a.all_trials, a.workers, a.jobs)
    else:
        cmd_collect(a.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
