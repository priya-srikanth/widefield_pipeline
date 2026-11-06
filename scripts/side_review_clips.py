"""Side-camera (cam2 + cam3) review clips for the shared side model: cut short trial windows in MODEL space, then render
the model's predictions as one labelled clip, cam2 | cam3 side by side on the SAME trial and time from cue (Priya
2026-10-27: "should we look at some labeled clips").

    python -m scripts.side_review_clips cut    PS93:20260814 PS93:20260821 PS95:20260606 [--per-position 2] --out <dir>
    litpose predict <model> <dir>/*/*_cam[23].mp4 --skip_viz
    python -m scripts.side_review_clips render PS93:20260814 ... --out <dir> --preds <model>/video_preds [--tag crop2]

`cut`: ``--per-position`` trials per spout position (`session_poses.choose`, deterministic), each cue ``--pre-ms`` ..
cue + ``--post-ms``, frame of the cue from EACH camera's own alignment template (so cam2 frame k and cam3 frame k are the
same moment, give or take one 4-ms frame). Frames go through `lp_sidecam.transform_image` (cam3 flipped, per-frame-size
crop, June rescaled to the August scale), so the clip is exactly what the model sees. Writes <animal>_<date>_cam2.mp4 / _cam3.mp4 (unique stems: litpose names predictions by stem)
(h264 crf 12, 250 fps) + index.csv (frame, trial_k, trial_id, position, t_ms, src_frame_cam2, src_frame_cam3). Source
videos READ-ONLY.
`render`: points where likelihood >= cut (tongue 0.4, others 0.6, as cam1), 5-frame trails, frames upscaled 2x; header =
trial, position, ms from cue. -> <dir>/<animal>_<date>/labeled_<tag>.mp4 at ``--fps`` playback.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from wfield_local import lp_sidecam as S

SIDE = ("cam2", "cam3")
COL = {"nose": (0, 220, 255), "jaw": (60, 220, 60), "tongue": (60, 60, 255), "spout": (255, 220, 0)}   # BGR
CUT = {"nose": 0.6, "jaw": 0.6, "tongue": 0.4, "spout": 0.6}
TRAIL = 5


def _ffmpeg_writer(path: Path, w: int, h: int, fps: float, gray: bool = False):
    from scripts.session_poses import _ffmpeg
    return subprocess.Popen([_ffmpeg(), "-nostdin", "-y", "-loglevel", "error", "-f", "rawvideo",
                             "-pix_fmt", "gray" if gray else "bgr24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
                             "-c:v", "libx264", "-crf", "12", "-preset", "veryfast", "-pix_fmt", "yuv420p", str(path)],
                            stdin=subprocess.PIPE)


def cmd_cut(rv, sessions, per_position: int, pre_ms: float, post_ms: float, out: Path) -> None:
    import cv2
    from PIL import Image

    from scripts.session_poses import choose
    from wfield_local import dlc_frames as DF
    from wfield_local import trial_windows as TW
    from wfield_local.writeguard import assert_writable
    n0, n1 = int(round(-pre_ms / 4.0)), int(round(post_ms / 4.0))
    for animal, date in sessions:
        d = out / f"{animal}_{date}"
        assert_writable(d)
        d.mkdir(parents=True, exist_ok=True)
        tr = choose(TW.trial_bounds(animal, date, rv), per_position)
        src = {}
        for cam in SIDE:
            tpl = dict(np.load(Path(rv.root("alignment_templates")) / cam / animal / f"{date}.npz", allow_pickle=True))
            vid = sorted((Path(rv.root("behavior_cameras")) / date / animal).glob(f"{cam}_*.avi"))[0]
            src[cam] = [DF.frame_of(tpl, r.cue_s, 0.0) for r in tr.itertuples()]
            cap = cv2.VideoCapture(str(vid))                     # READ-ONLY
            ff = _ffmpeg_writer(d / f"{animal}_{date}_{cam}.part.mp4", S.OUT, S.OUT, 250, gray=True)
            for fc in src[cam]:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fc + n0)
                for _ in range(n0, n1):
                    ok, im = cap.read()
                    if not ok:
                        im = np.zeros((int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 3),
                                      np.uint8)
                    g = S.transform_image(Image.fromarray(im[..., 0]), cam)
                    ff.stdin.write(np.asarray(g, np.uint8).tobytes())
            cap.release()
            ff.stdin.close()
            ff.wait()
            (d / f"{animal}_{date}_{cam}.part.mp4").replace(d / f"{animal}_{date}_{cam}.mp4")
        rows = [(k, int(r.trial_id), r.pos_name, j * 4.0, src["cam2"][k] + j, src["cam3"][k] + j)
                for k, r in enumerate(tr.itertuples()) for j in range(n0, n1)]
        pd.DataFrame(rows, columns=["trial_k", "trial_id", "position", "t_ms", "src_frame_cam2", "src_frame_cam3"]) \
            .rename_axis("frame").to_csv(d / "index.csv")
        print(f"{animal} {date}: {len(tr)} trials x {n1 - n0} frames -> {d}", flush=True)


def _read_preds(csv: Path) -> pd.DataFrame:
    df = pd.read_csv(csv, header=[0, 1, 2], index_col=0)
    sc = df.columns.get_level_values(0)[0]
    return pd.DataFrame({f"{p}_{c}": df[(sc, p, c)].to_numpy() for p in S.PARTS for c in ("x", "y", "likelihood")})


def cmd_render(sessions, out: Path, preds: Path, tag: str, fps: float) -> None:
    import cv2
    for animal, date in sessions:
        d = out / f"{animal}_{date}"
        idx = pd.read_csv(d / "index.csv")
        caps = {c: cv2.VideoCapture(str(d / f"{animal}_{date}_{c}.mp4")) for c in SIDE}
        pr = {c: _read_preds(preds / f"{animal}_{date}_{c}.csv") for c in SIDE}     # litpose: <video stem>.csv
        sz = 2 * S.OUT
        ff = _ffmpeg_writer(d / f"labeled_{tag}.mp4", 2 * sz, sz + 24, fps)
        for i, r in idx.iterrows():
            panels = []
            for c in SIDE:
                ok, im = caps[c].read()
                im = cv2.resize(im if ok else np.zeros((S.OUT, S.OUT, 3), np.uint8), (sz, sz), interpolation=cv2.INTER_LINEAR)
                for p in S.PARTS:
                    for lag in range(TRAIL - 1, -1, -1):
                        j = i - lag
                        if j < 0 or idx.trial_k[j] != r.trial_k:
                            continue
                        x, y, lk = (pr[c][f"{p}_{k}"][j] for k in ("x", "y", "likelihood"))
                        if lk >= CUT[p]:
                            cv2.circle(im, (int(round(2 * x + 0.5)), int(round(2 * y + 0.5))), 6 if lag == 0 else 2,
                                       COL[p], 2 if lag == 0 else -1, cv2.LINE_AA)
                cv2.putText(im, c + (" (flipped)" if c == "cam3" else ""), (6, sz - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1, cv2.LINE_AA)
                panels.append(im)
            frame = np.zeros((sz + 24, 2 * sz, 3), np.uint8)
            frame[24:] = np.hstack(panels)
            cv2.putText(frame, f"{animal} {date}  trial {r.trial_id}  {r.position}  {r.t_ms:+.0f} ms from cue   "
                               "nose yellow  jaw green  tongue red  spout cyan", (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1, cv2.LINE_AA)
            ff.stdin.write(frame.tobytes())
        for c in caps.values():
            c.release()
        ff.stdin.close()
        ff.wait()
        print(f"{animal} {date}: -> {d / f'labeled_{tag}.mp4'}", flush=True)


def main(argv=None) -> int:
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["cut", "render"])
    ap.add_argument("sessions", nargs="+", help="animal:date")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-position", type=int, default=2)
    ap.add_argument("--pre-ms", type=float, default=200.0)
    ap.add_argument("--post-ms", type=float, default=1500.0)
    ap.add_argument("--preds", type=Path, help="render: the model's video_preds folder")
    ap.add_argument("--tag", default="side")
    ap.add_argument("--fps", type=float, default=30.0)
    a = ap.parse_args(argv)
    sessions = [tuple(s.split(":")) for s in a.sessions]
    if a.cmd == "cut":
        cmd_cut(PathResolver(), sessions, a.per_position, a.pre_ms, a.post_ms, a.out)
    else:
        cmd_render(sessions, a.out, a.preds, a.tag, a.fps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
