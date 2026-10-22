"""FaceRhythm prep (docs/FACE_RHYTHM_PLAN.md): per-session reference faces + LP landmarks, for registering every session
onto one MASTER face per camera.

    python -m scripts.face_rhythm_prep frames [--cam cam4|cam1|cam2|cam3] [--jobs 4] [--sessions PS92:20260818 ...]

frames  per curated session (v36 cam4 windows; locanmf env): the MEDIAN of ~300 pre-cue frames (``N_TRIALS`` trials x every ``EVERY``-th frame) (-1500 .. -500 ms from
        the cue; tongue absent, LP tongue likelihood < 0.5 -> mouth closed), sampled evenly over the session, and the LP
        landmarks the registration is checked with and the ROI is drawn against: nose and jaw at rest (same frames),
        per-position spout tip (median over 0 .. 500 ms after the cue, spout in place) and the tongue's reach (95th
        percentile radius of tongue points, likelihood > 0.9, around the jaw). ->
        <dlc>/face_rhythm/<cam>/frames/<a>_<d>.npz (median uint8, landmarks) + a contact sheet per animal.
        OTHER CAMERAS (`wfield_local.face_rhythm_cams`): the same cam4-LP rest moments, mapped to the camera's frames
        through its alignment template (`face_rhythm_cams.frames_at`); no LP landmarks there (NaN / empty) -- registration uses clicks.
Registration (session -> animal template -> master) runs in the ``fr`` env: `scripts/face_rhythm_register.py`.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

V36 = "lp_r4_heatmap_20261013__clean_v36"
N_TRIALS, EVERY = 30, 25          # 30 trials x ~10 rest frames (every 100 ms) = ~300 frames
PRE = (-1500.0, -500.0)


def out_dir(rd: Path, cam: str) -> Path:
    return rd / "face_rhythm" / cam


def session(job) -> dict:
    """Median rest face of one session in ``cam``. Rest = cam4 LP (tongue absent, ``PRE`` from the cue) for EVERY view:
    another camera reads the same moments through its own template (`face_rhythm_cams.frames_at`); trials at ``rest_exclude``
    positions are dropped (cam1: the centre spout rod is in front of the face). LP landmarks only for cam4."""
    s, cam, ro, rd, root_vid = job
    import cv2

    from wfield_local import face_rhythm_cams as frc
    C = frc.cam(cam)
    a, d = s.split("_")
    w = np.load(Path(ro) / V36 / "sessions" / a / d / "windows.npz")
    parts = [str(p) for p in w["parts"]]
    pose = w["pose"]
    t, src, pos = w["ix_t_ms"], w["ix_src_frame"].astype(int), w["ix_position"].astype(str)
    tg = pose[:, parts.index("tongue"), :]
    rest = (t >= PRE[0]) & (t < PRE[1]) & (tg[:, 2] < 0.5) & ~np.isin(pos, C["rest_exclude"])
    idx = np.flatnonzero(rest)
    if len(idx) < 50:
        return {"session": s, "error": f"only {len(idx)} rest frames"}
    vids = sorted(glob.glob(str(Path(root_vid) / d / a / f"{cam}_*.avi")))
    if not vids:
        return {"session": s, "error": "no video"}
    if cam != "cam4":                                                   # cam4 frames -> this camera's frames by DAQ time
        from wfield_local.paths import PathResolver
        fd = frc.frame_daq_seconds(PathResolver(), a, d, cam, Path(vids[-1]).name)
        f_cam = frc.frames_at(fd, w["ix_daq_s"][idx], nearest=True)
        idx, src = idx[f_cam >= 0], src.copy()
        src[idx] = f_cam[f_cam >= 0]
        if len(idx) < 50:
            return {"session": s, "error": f"only {len(idx)} rest frames inside {Path(vids[-1]).name}"}
    # ONE SEEK PER TRIAL (300 random seeks over the network took ~15 min a session): N_TRIALS trials spread over the
    # session, each read forward through its rest frames (grab() skips cheaply), keeping every EVERY-th frame
    tid = w["ix_trial_id"][idx]
    trials = np.unique(tid)
    trials = trials[np.linspace(0, len(trials) - 1, min(N_TRIALS, len(trials))).astype(int)]
    cap = cv2.VideoCapture(vids[-1])
    ims = []
    for tr in trials:
        f = np.sort(src[idx[tid == tr]])
        want = set(f[::EVERY].tolist())
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f[0]))
        for fr in range(int(f[0]), int(f[-1]) + 1):
            if fr in want:
                ok, im = cap.read()
                if ok:
                    ims.append(im[..., 0])
            elif not cap.grab():
                break
    cap.release()
    med = np.median(np.stack(ims), 0).astype(np.uint8)

    def lm(part, m, cut=0.9):
        P = pose[m, parts.index(part), :]
        P = P[P[:, 2] > cut]
        return np.median(P[:, :2], 0) if len(P) else np.full(2, np.nan)

    nose, jaw, tips, reach = np.full(2, np.nan), np.full(2, np.nan), {}, np.nan
    if C["lp_landmarks"]:                                               # LP coordinates are cam4 pixels
        nose, jaw = lm("nose", rest), lm("jaw", rest)
        for p in np.unique(pos):
            tips[p] = lm("spout", (pos == p) & (t >= 0) & (t < 500))
        T = tg[(tg[:, 2] > 0.9), :2]
        reach = float(np.percentile(np.hypot(*(T - jaw).T), 95)) if len(T) and np.isfinite(jaw).all() else np.nan
    o = out_dir(Path(rd), cam) / "frames"
    o.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(o / f"{s}.npz", median=med, n_frames=len(ims), video=Path(vids[-1]).name, nose=nose, jaw=jaw,
                        tongue_reach_px=reach, tip_positions=np.array(list(tips)),
                        tip_xy=np.array([tips[k] for k in tips]).reshape(-1, 2))
    return {"session": s, "n_frames": len(ims), "nose_x": nose[0], "nose_y": nose[1], "jaw_x": jaw[0],
            "jaw_y": jaw[1], "tongue_reach_px": reach}


def contact_sheets(rd: Path, cam: str, sessions: list[str]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    o = out_dir(rd, cam)
    for a in sorted({s[:4] for s in sessions}):
        ss = [s for s in sessions if s.startswith(a) and (o / "frames" / f"{s}.npz").exists()]
        n = len(ss)
        if not n:
            continue
        c = 8
        fig, axes = plt.subplots(int(np.ceil(n / c)), c, figsize=(2 * c, 2.1 * np.ceil(n / c)), squeeze=False)
        for ax in axes.flat:
            ax.axis("off")
        for ax, s in zip(axes.flat, ss):
            Z = np.load(o / "frames" / f"{s}.npz")
            ax.imshow(Z["median"], cmap="gray")
            for xy in Z["tip_xy"]:
                ax.plot(*xy, "o", ms=2, color="tab:orange")
            ax.plot(*Z["nose"], "+", color="c", ms=5)
            ax.plot(*Z["jaw"], "+", color="lime", ms=5)
            ax.set_title(s[5:], fontsize=7)
        fig.suptitle(f"{a} {cam}: median pre-cue rest face per session (+ nose, + jaw, o spout tips)", fontsize=9)
        fig.tight_layout()
        fig.savefig(o / f"frames_{a}.png", dpi=90)
        plt.close(fig)


def main(argv=None) -> int:
    from wfield_local import analysis_kit as ak
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["frames"])
    ap.add_argument("--cam", default="cam4", choices=["cam4", "cam1", "cam2", "cam3"])
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--sessions", nargs="*", default=None, metavar="ANIMAL:YYYYMMDD")
    a = ap.parse_args(argv)
    rv = PathResolver()
    ro, rd = Path(rv.root("orofacial")), Path(rv.root("dlc"))
    root_vid = Path(rv.root("microscope")) / "Behavior_cameras" / "widefield"
    sess = sorted(f"{p.parent.name}_{p.name}" for p in (ro / V36 / "sessions").glob("*/*")
                  if (p / "windows.npz").exists())
    sess = [s for s in sess if s in ak.curated_ids()]
    if a.sessions:
        sess = [s for s in sess if s.replace("_", ":", 1) in a.sessions]
    res, fails = ak.fan_sessions([(s, a.cam, str(ro), str(rd), str(root_vid)) for s in sess], session, jobs=a.jobs,
                                 key=lambda j: j[0])
    for j, e in fails:
        print(f"{j[0]}: FAILED {e}")
    T = pd.DataFrame([r for _, r in res])
    o = out_dir(rd, a.cam)
    T.to_csv(o / "frames_summary.csv", index=False, float_format="%.1f")
    contact_sheets(rd, a.cam, sess)
    print(T.groupby(T.session.str[:4]).agg(n=("session", "size"), nose_x=("nose_x", "std"),
                                           nose_y=("nose_y", "std")).round(1).to_string())
    print("->", o)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
