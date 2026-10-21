"""Whisker-pad motion energy per camera frame (`wfield_local.roi_motion`), and its cue-aligned per-trial traces.

    python -m scripts.whisker_motion run PS94:20260810 ... [--cams cam4 cam1] [--workers 6]
    python -m scripts.whisker_motion traces PS94:20260810 ...       # cue-aligned per-trial table + figure

run     one video per session x camera -> <dlc>/session_poses/whisker_me/<a>_<d>_<cam>.npz: ``me`` (n_frames, 2 ROIs
        [image-left, image-right], n_strips) float16, ``rois`` (x0, y0, x1, y1 full-res), ``nose_xy`` (session-median
        LP nose, likelihood > 0.9), video name. Interim measure until FaceRhythm (see roi_motion).
traces  per trial on the camera's own template-time windows (cam4: v36 ``windows.npz``; cam1: ``<s>_cam1_full``):
        each ROI's mean over strips clear of the spout rod (LP spout x, trial median, +-30 px), in 10 ms bins from -500
        to +1500 ms around the cue -> whisker_me/<a>_<d>_<cam>_trials.npz + a figure by attempt class (from
        trial_attempts.csv), to choose a per-trial rule.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from wfield_local import pose_windows

V36 = "lp_r4_heatmap_20261013__clean_v36"
CAMS = ("cam4", "cam1")
BIN_MS, T0, T1 = 10, -500, 1500


def _nose(s: str, cam: str, ro: Path, rd: Path):
    a, d = s.split("_")
    if cam == "cam4":
        w = np.load(ro / V36 / "sessions" / a / d / "windows.npz")
        P = w["pose"][:, list(w["parts"]).index("nose"), :]
    else:
        L = pose_windows.lp(rd / "session_poses" / f"{s}_cam1_full")
        P = L[[("nose", "x"), ("nose", "y"), ("nose", "likelihood")]].to_numpy(float)
    ok = P[:, 2] > 0.9
    return float(np.median(P[ok, 0])), float(np.median(P[ok, 1]))


def _video(s: str, cam: str, rv) -> Path | None:
    a, d = s.split("_")
    root = Path(rv.root("microscope")) / "Behavior_cameras" / "widefield"
    v = sorted(glob.glob(str(root / d / a / f"{cam}_*.avi")))
    return Path(v[-1]) if v else None


def cmd_run(sessions, cams, workers):
    import cv2

    from wfield_local import roi_motion as RM
    from wfield_local.paths import PathResolver
    rv = PathResolver()
    ro, rd = Path(rv.root("orofacial")), Path(rv.root("dlc"))
    od = rd / "session_poses" / "whisker_me"
    od.mkdir(parents=True, exist_ok=True)
    scratch = Path.home() / "lp_stage" / "whisker_me"
    scratch.mkdir(parents=True, exist_ok=True)
    for s in sessions:
        for cam in cams:
            dst = od / f"{s}_{cam}.npz"
            if dst.exists():
                print(f"{s} {cam}: exists")
                continue
            if cam == "cam1" and not (rd / "session_poses" / f"{s}_cam1_full" / "windows_LP.csv").exists():
                print(f"{s} cam1: no cam1 LP (nose) yet")
                continue
            v = _video(s, cam, rv)
            if v is None:
                print(f"{s} {cam}: no video")
                continue
            cap = cv2.VideoCapture(str(v))
            shape = (int(cap.get(4)), int(cap.get(3)))
            cap.release()
            nose = _nose(s, cam, ro, rd)
            R = RM.rois(cam, nose, shape)
            import time
            t = time.time()
            mm = RM.roi_motion(v, R, scratch / f"{s}_{cam}.npy", ds=2, n_workers=workers)
            np.savez_compressed(dst, me=np.asarray(mm), rois=np.asarray(R), nose_xy=np.asarray(nose), video=v.name)
            del mm
            (scratch / f"{s}_{cam}.npy").unlink(missing_ok=True)
            print(f"{s} {cam}: {(time.time() - t) / 60:.1f} min -> {dst.name}", flush=True)


def _windows(s: str, cam: str, ro: Path, rd: Path) -> pd.DataFrame:
    a, d = s.split("_")
    if cam == "cam4":
        w = np.load(ro / V36 / "sessions" / a / d / "windows.npz")
        ix = pd.DataFrame({k: w[f"ix_{k}"] for k in ("trial_id", "position", "src_frame", "t_daq_ms")})
        ix["spout_x"] = w["pose"][:, list(w["parts"]).index("spout"), 0]
    else:
        sp = rd / "session_poses" / f"{s}_cam1_full"
        ix = pose_windows.index(sp, usecols=["trial_id", "position", "src_frame", "t_daq_ms"])
        L = pose_windows.lp(sp)
        ix["spout_x"] = np.where(L[("spout", "likelihood")].to_numpy(float) > 0.6, L[("spout", "x")].to_numpy(float),
                                 np.nan)
    return ix


def trial_traces(s: str, cam: str, ro: Path, rd: Path):
    """(trial_ids, positions, (n_trials, 2, n_bins) cue-aligned ME per ROI) on the DAQ clock."""
    from wfield_local import roi_motion as RM
    Z = np.load(rd / "session_poses" / "whisker_me" / f"{s}_{cam}.npz")
    me, R = Z["me"], [tuple(r) for r in Z["rois"]]
    ix = _windows(s, cam, ro, rd)
    edges = np.arange(T0, T1 + BIN_MS, BIN_MS)
    tids, poss, out = [], [], []
    for tid, g in ix.groupby("trial_id", sort=True):
        f = g.src_frame.to_numpy(int)
        f = f[(f >= 0) & (f < len(me))]
        if not len(f):
            continue
        rod = float(np.nanmedian(g.spout_x)) if g.spout_x.notna().any() else None
        t = g.t_daq_ms.to_numpy(float)[:len(f)]
        k = np.digitize(t, edges) - 1
        ok = (k >= 0) & (k < len(edges) - 1)
        row = np.full((2, len(edges) - 1), np.nan)
        for j in range(2):
            v = RM.trial_trace(me[f, j, :], R[j], rod)
            if ok.any():
                sm = np.bincount(k[ok], np.nan_to_num(v[ok]), len(edges) - 1)
                n = np.bincount(k[ok], np.isfinite(v[ok]), len(edges) - 1)
                with np.errstate(invalid="ignore"):
                    row[j] = sm / n
        tids.append(int(tid))
        poss.append(str(g.position.iloc[0]))
        out.append(row)
    return np.asarray(tids), np.asarray(poss), np.asarray(out), edges


def cmd_traces(sessions, cams):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from wfield_local.paths import PathResolver
    rv = PathResolver()
    ro, rd = Path(rv.root("orofacial")), Path(rv.root("dlc"))
    od = rd / "session_poses" / "whisker_me"
    A = pd.read_csv(ro / V36 / "imaging_encoding" / "qc" / "trial_attempts.csv", usecols=["label", "trial_id", "cls"])
    for s in sessions:
        got = [c for c in cams if (od / f"{s}_{c}.npz").exists()]
        if not got:
            continue
        fig, axes = plt.subplots(len(got), 2, figsize=(10, 3.2 * len(got)), squeeze=False)
        cl = A[A.label == f"{s[:4]}_{s[9:13]}"].set_index("trial_id").cls
        for i, cam in enumerate(got):
            tids, poss, X, edges = trial_traces(s, cam, ro, rd)
            np.savez_compressed(od / f"{s}_{cam}_trials.npz", trial_id=tids, position=poss, me=X, edges=edges)
            t = (edges[:-1] + edges[1:]) / 2
            cls = pd.Series(tids).map(cl).fillna("?").to_numpy()
            for j, side in enumerate(("image-left", "image-right")):
                ax = axes[i, j]
                for c, col in (("hit", "k"), ("incomplete", "tab:blue"), ("jaw_only", "tab:orange"),
                               ("no_try", "tab:red"), ("unknown", "0.6")):
                    m = cls == c
                    if m.sum() < 3:
                        continue
                    base = np.nanmean(X[m, j][:, t < 0], 1, keepdims=True)
                    ax.plot(t, np.nanmean(X[m, j] - base, 0), color=col, lw=1.2, label=f"{c} ({m.sum()})")
                ax.axvline(0, color="0.5", lw=0.6)
                ax.set_title(f"{s} {cam} {side} pad", fontsize=8)
                ax.set_xlabel("ms from cue (DAQ)", fontsize=7)
                ax.set_ylabel("motion energy - pre-cue", fontsize=7)
                ax.legend(fontsize=6, frameon=False)
        fig.tight_layout()
        p = od / f"{s}_traces_by_class.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        print("->", p)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "traces"])
    ap.add_argument("sessions", nargs="+", metavar="ANIMAL:YYYYMMDD")
    ap.add_argument("--cams", nargs="+", default=list(CAMS), choices=CAMS)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(argv)
    sess = [x.replace(":", "_") for x in a.sessions]
    if a.cmd == "run":
        cmd_run(sess, a.cams, a.workers)
    else:
        cmd_traces(sess, a.cams)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
