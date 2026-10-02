"""Session-level cam4 pose predictions on a TRIAL SUBSET: whole trials (position strobe -0.5 s .. trial stop +3 s),
N per spout position, cut into one clip, then DLC round 3 (+ prior) and the LP occlusion model on that clip.

    python -m scripts.session_poses clip  PS93:20260814 PS93:20260821 PS93:20260908 [--per-position 10]
    python -m scripts.session_poses dlc   PS93:20260814 ...      # resumable (chunks), GPU
    python -m scripts.session_poses merge PS93:20260814 ...      # chunks -> windows_DLC.csv
    python -m scripts.session_poses from-scan PS93:20260819 ...  # round-4 scan cache -> <a>_<d>_r4/ (no GPU)
    # LP: WSL only, 16-frame DALI chunks + memory guard (96 blue-screened this box twice):
    #   litpose predict <model> windows.mp4 --overrides dali.base.predict.sequence_length=16 \\
    #       dali.context.predict.sequence_length=16      -> copy video_preds csv to windows_LP.csv

Priya, 2026-10-01: run the kinematics session-level on 1-3 sessions (one pre, one acute, one chronic).
WHY A SUBSET, NOT THE WHOLE VIDEO: trials run back to back (strobe -0.5 .. stop +3 s covers ~all of a 2-2.5 h
video, 1.8-2.2 M frames) and DLC round 3 (ResNet-50-GN at 680x680) runs ~42 fps on this RTX 5060 whatever the
batch size (forward pass 16 ms/frame; fp16 / cudnn.benchmark no faster), i.e. 12-15 h per session. Full sessions
are for a server / a faster inference path (crop to the face, GPU preprocessing) -- see DECISIONS 2026-10-01.
Outputs in <share>/DeepLabCut/Widefield/session_poses/<animal>_<date>/: windows_index.csv (trial_k, trial_id,
position, src_frame, t_ms from cue, plus strobe/stop ms), windows.mp4 (h264 crf 12, both models read this same
file), dlc_chunks/, windows_DLC.csv. Source video READ-ONLY.
"""
from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

CHUNK = 20_000            # frames per DLC checkpoint
SEED = 7
PRE_S, POST_S = 0.5, 3.0  # before the strobe (or cue -0.5 s if earlier), after the trial stop


def _ffmpeg() -> str:
    """ffmpeg on PATH, else imageio-ffmpeg's bundled binary (this desktop has no system ffmpeg on Windows)."""
    import shutil
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def out_dir(rv, animal, date) -> Path:
    from wfield_local import dlc_project
    d = dlc_project.project_dir(rv).parent / "session_poses" / f"{animal}_{date}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def session_dir(rv, spec: str) -> tuple[str, str, Path]:
    """``animal:date`` -> the trial-subset clip folder; ``animal:date:tag`` -> ``<animal>_<date>_<tag>`` (e.g. tag
    ``full`` = every trial from a whole-video O2 run, `o2_inference collect`)."""
    animal, date, *tag = spec.split(":")
    from wfield_local import dlc_project
    name = f"{animal}_{date}" + (f"_{tag[0]}" if tag else "")
    return animal, date, dlc_project.project_dir(rv).parent / "session_poses" / name


def choose(b: pd.DataFrame, per_position: int) -> pd.DataFrame:
    """``per_position`` trials per spout position, spread over the session (evenly spaced in time order)."""
    b = b[np.isfinite(b.cue_s) & np.isfinite(b.stop_s)].sort_values("cue_s")
    out = []
    for _, g in b.groupby("pos_name"):
        k = np.unique(np.linspace(0, len(g) - 1, min(per_position, len(g))).round().astype(int))
        out.append(g.iloc[k])
    return pd.concat(out).sort_values("cue_s").reset_index(drop=True)


def cmd_clip(rv, sessions, per_position: int) -> None:
    import cv2

    from wfield_local import dlc_frames as DF
    from wfield_local import trial_windows as TW
    for animal, date in sessions:
        d = out_dir(rv, animal, date)
        if (d / "windows.mp4").exists():
            print(f"{animal} {date}: clip exists", flush=True)
            continue
        tpl = dict(np.load(Path(rv.root("alignment_templates")) / "cam4" / animal / f"{date}.npz", allow_pickle=True))
        vid = sorted((Path(rv.root("behavior_cameras")) / date / animal).glob("cam4_*.avi"))[0]
        tr = choose(TW.trial_bounds(animal, date, rv), per_position)
        cap = cv2.VideoCapture(str(vid))                         # READ-ONLY
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        tmp = d / "windows.part.mp4"
        ff = subprocess.Popen([_ffmpeg(), "-nostdin", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                               "-s", f"{w}x{h}", "-r", "250", "-i", "-", "-c:v", "libx264", "-crf", "12",
                               "-preset", "veryfast", "-pix_fmt", "yuv420p", str(tmp)], stdin=subprocess.PIPE)
        idx, pos = [], -1
        for k, r in enumerate(tr.itertuples()):
            t0 = min(r.strobe_s, r.cue_s - PRE_S) if np.isfinite(r.strobe_s) else r.cue_s - PRE_S
            f0, f1 = DF.frame_of(tpl, t0 - PRE_S, 0.0), DF.frame_of(tpl, r.stop_s + POST_S, 0.0)
            fc = DF.frame_of(tpl, r.cue_s, 0.0)
            if pos != f0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
            for f in range(f0, f1):
                ok, im = cap.read()
                if not ok:
                    break
                ff.stdin.write(im.tobytes())
                idx.append((k, int(r.trial_id), r.pos_name, f, (f - fc) * 4.0,
                            (r.strobe_s - r.cue_s) * 1000.0, (r.stop_s - r.cue_s) * 1000.0, r.stop_source))
            pos = f1
        cap.release()
        ff.stdin.close()
        ff.wait()
        tmp.rename(d / "windows.mp4")
        pd.DataFrame(idx, columns=["trial_k", "trial_id", "position", "src_frame", "t_ms", "strobe_ms", "stop_ms",
                                   "stop_source"]).to_csv(d / "windows_index.csv", index=False)
        print(f"{animal} {date}: {len(tr)} trials, {len(idx)} frames -> {d / 'windows.mp4'}", flush=True)


def cmd_dlc(rv, sessions) -> None:
    import cv2

    from wfield_local import dlc_prior
    from wfield_local.dlc_hard_frames import pose_predictor
    predict = None
    with dlc_prior.apply(rv, cam="cam4"):
        for animal, date in sessions:
            d = out_dir(rv, animal, date)
            n = len(pd.read_csv(d / "windows_index.csv"))
            cd = d / "dlc_chunks"
            cd.mkdir(exist_ok=True)
            cap, pos = cv2.VideoCapture(str(d / "windows.mp4")), 0
            for c0 in range(0, n, CHUNK):
                c1 = min(n, c0 + CHUNK)
                f = cd / f"{c0:08d}_{c1:08d}.npz"
                if f.exists():
                    continue
                predict = predict or pose_predictor(rv)
                if pos != c0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, c0)
                t, out, buf, k = time.time(), [], [], c0
                while k < c1:
                    ok, im = cap.read()
                    if not ok:
                        break
                    buf.append(im)
                    k += 1
                    if len(buf) == 64 or k == c1:
                        out.append(predict(buf))
                        buf = []
                pos = k
                P = np.concatenate(out).astype(np.float32)
                np.savez_compressed(f, f0=c0, pose=P)
                print(f"{animal} {date}: {c0}-{k} of {n} ({len(P) / (time.time() - t):.0f} fps)", flush=True)
            cap.release()


def cmd_from_scan(rv, sessions) -> None:
    """The round-4 scan's cached windows (`dlc_hard_frames`: 2 trials / position, cue -1 .. +3.5 s; DLC round 3 +
    prior in round4_scan/<a>_<d>_dlc.npz, LP occlusion model in %USERPROFILE%/lp_clips/round4/lp/<a>_<d>.csv, same
    windows in the same order) -> `session_poses/<a>_<d>_r4/` (windows_index.csv, windows_DLC.csv, windows_LP.csv),
    so every session script reads them as `animal:date:r4`. No clip: frames come from the original video
    (src_frame). Windows end at cue + 3.5 s, i.e. before most trial stops -- post-cue measures there are truncated."""
    from wfield_local import dlc_frames as DF
    from wfield_local import dlc_hard_frames as H
    from wfield_local import dlc_project
    from wfield_local import trial_windows as TW
    scan = dlc_project.project_dir(rv).parent / "round4_scan"
    for animal, date in sessions:
        wins = H.load_poses(scan / f"{animal}_{date}_dlc.npz")
        tpl = dict(np.load(Path(rv.root("alignment_templates")) / "cam4" / animal / f"{date}.npz", allow_pickle=True))
        b = TW.trial_bounds(animal, date, rv).set_index("trial_id")
        idx, poses = [], []
        for k, (f0, tr, P) in enumerate(wins):
            fc = DF.frame_of(tpl, tr["cue_s"], 0.0)
            r = b.loc[int(tr["trial_id"])]
            for j in range(len(P)):
                idx.append((k, int(tr["trial_id"]), tr["pos_name"], f0 + j, (f0 + j - fc) * 4.0,
                            (r.strobe_s - r.cue_s) * 1000.0, (r.stop_s - r.cue_s) * 1000.0, r.stop_source))
            poses.append(P)
        d = out_dir(rv, animal, f"{date}_r4")
        pd.DataFrame(idx, columns=["trial_k", "trial_id", "position", "src_frame", "t_ms", "strobe_ms", "stop_ms",
                                   "stop_source"]).to_csv(d / "windows_index.csv", index=False)
        P = np.concatenate(poses)
        cols = pd.MultiIndex.from_product([["DLC_round3"], H.PARTS, ["x", "y", "likelihood"]],
                                          names=["scorer", "bodyparts", "coords"])
        pd.DataFrame(P.reshape(len(P), -1), columns=cols).to_csv(d / "windows_DLC.csv")
        lp = H.LOCAL_CLIPS / "lp" / f"{animal}_{date}.csv"
        n_lp = None
        if lp.exists():
            L = pd.read_csv(lp, header=[0, 1, 2], index_col=0)
            n_lp = len(L)
            if n_lp == len(P):
                L.to_csv(d / "windows_LP.csv")
        print(f"{animal} {date}: {len(wins)} trials, {len(P)} frames -> {d}  (LP {n_lp} rows"
              f"{'' if n_lp == len(P) else ' -- MISMATCH, not copied'})")


def cmd_merge(rv, sessions) -> None:
    from wfield_local.dlc_hard_frames import PARTS
    for animal, date in sessions:
        d = out_dir(rv, animal, date)
        n = len(pd.read_csv(d / "windows_index.csv"))
        ch = sorted((d / "dlc_chunks").glob("*.npz"))
        P = np.concatenate([np.load(f)["pose"] for f in ch])
        assert len(P) == n, f"{animal} {date}: {len(P)} predicted of {n} -- run dlc again"
        cols = pd.MultiIndex.from_product([["DLC_round3"], PARTS, ["x", "y", "likelihood"]],
                                          names=["scorer", "bodyparts", "coords"])
        pd.DataFrame(P.reshape(len(P), -1), columns=cols).to_csv(d / "windows_DLC.csv")
        print(f"{animal} {date}: {n} frames -> {d / 'windows_DLC.csv'}")


def main(argv=None) -> int:
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["clip", "dlc", "merge", "from-scan"])
    ap.add_argument("sessions", nargs="+", help="animal:date")
    ap.add_argument("--per-position", type=int, default=10)
    a = ap.parse_args(argv)
    rv = PathResolver()
    sess = [tuple(s.split(":")) for s in a.sessions]
    {"clip": lambda: cmd_clip(rv, sess, a.per_position), "dlc": lambda: cmd_dlc(rv, sess),
     "merge": lambda: cmd_merge(rv, sess), "from-scan": lambda: cmd_from_scan(rv, sess)}[a.cmd]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
