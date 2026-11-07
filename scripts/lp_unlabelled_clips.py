"""Unlabelled clips for Lightning Pose's semi-supervised losses (temporal, pose-PCA).

    conda activate dlc
    python -m scripts.lp_unlabelled_clips                # -> <dlc root>/lightning-pose/<project>/videos/

Each clip is cut from a CUE so it contains the whole failure surface in one window: ENL, cue, the
3.5 s response window with its licking, and the following inter-trial gap where the spout moves.
Sampling only lick-dense windows (as the throughput check clips do) would train the temporal loss on
the licking and never on the interval where the spout dropout happened.

Sessions are the LABELED ones (from the frame manifest), one per animal x epoch, so the unlabelled
distribution matches the supervised one across all four animals and all four epochs. Seeded, and
idempotent: an existing clip is kept, so re-running after the label set grows adds nothing and
removes nothing. Recovered into the repo 2026-09-28 from the 2026-09-26 scratch script that cut the
15 clips in `lightning-pose/cam4-2026-09-26/videos/` (7,500 frames each, 0.59 GB).

SIDE CAMERAS (`--cam cam2 cam3`, 2026-11-04): the clip is written in MODEL space, exactly what the shared side model sees
(`lp_sidecam.transform_image`: cam3 flipped left-right, the per-frame-size square crop, June resized to the August scale),
via the matching ffmpeg filter chain; one clip per camera per picked session, `<video stem>_t<s>.mp4`.
`--all-sessions --windows N` (Priya 11-04: "plenty of temporal loss training frames"): every labeled session of the
camera, N non-overlapping cue-anchored windows each, seeded per session.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from wfield_local.dlc_frames import manifest_path
from wfield_local.paths import PathResolver

DUR_S, PRE_S, SEED = 30.0, 3.0, 92


def _ffmpeg() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def pick_sessions(manifest: pd.DataFrame, cam: str = "cam4") -> pd.DataFrame:
    """One labeled session per (animal, epoch): the first in a deterministic sort."""
    m = manifest[manifest.cam == cam].drop_duplicates("video_stem")[["animal", "date", "video_stem", "epoch"]]
    return m.sort_values(["epoch", "animal"]).groupby(["animal", "epoch"], as_index=False).first()


def side_filter(cam: str, width: int) -> str:
    """ffmpeg -vf chain equal to `lp_sidecam.transform_image` (flip cam3, crop, resize to lp_sidecam.OUT)."""
    from wfield_local import lp_sidecam as S
    x0, y0, side = S.crop_for(width)
    f = (["hflip"] if cam == "cam3" else []) + [f"crop={side}:{side}:{x0}:{y0}"]
    return ",".join(f + ([f"scale={S.OUT}:{S.OUT}:flags=bilinear"] if side != S.OUT else []))


def _width(video: Path) -> int:
    import cv2                     # imageio_ffmpeg cannot read the meta of these AVIs
    cap = cv2.VideoCapture(str(video))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()
    return w


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", default="cam4-2026-09-26", help="LP project dir name under <dlc root>/lightning-pose/")
    ap.add_argument("--cam", nargs="+", default=["cam4"], help="cam4, or the side cameras: cam2 cam3")
    ap.add_argument("--all-sessions", action="store_true",
                    help="every labeled session of the camera, not one per animal x epoch")
    ap.add_argument("--windows", type=int, default=1,
                    help="with --all-sessions: non-overlapping 30 s windows per session (per-session seed)")
    a = ap.parse_args(argv)
    rv = PathResolver()
    out = Path(rv.root("dlc")) / "lightning-pose" / a.project / "videos"
    out.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(manifest_path(rv), dtype=str)
    for cam in a.cam:
        if a.all_sessions:
            _cut_all(rv, man, cam, out, a.windows)
        else:
            _cut(rv, man, cam, out)
    clips = list(out.glob("*.mp4"))
    print(f"\n{len(clips)} clips, {sum(f.stat().st_size for f in clips) / 1e9:.2f} GB, {DUR_S * 250:.0f} frames each")
    return 0


def _cut(rv, man: pd.DataFrame, cam: str, out: Path) -> None:
    picked = pick_sessions(man, cam)
    rng = np.random.default_rng(SEED)
    print(f"{cam}: {len(picked)} sessions: " + ", ".join(f"{r.animal}/{r.epoch}" for r in picked.itertuples()))
    for r in picked.itertuples():
        tcsv = sorted((Path(rv.root("behavior_out")) / "sessions" / r.animal / r.date).glob("*_trials.csv"))
        vids = sorted((Path(rv.root("behavior_cameras")) / r.date / r.animal).glob(f"{cam}_*.avi"))
        if not tcsv or not vids:
            print(f"  {r.video_stem}: missing trials or video -> skip")
            continue
        cue = pd.read_csv(tcsv[0])["cue_s"].to_numpy()
        mid = cue[(cue > 60) & (cue < cue.max() - 60)]
        if not len(mid):
            continue
        _write(cam, vids[0], out / f"{r.video_stem}_t{int(float(rng.choice(mid)) - PRE_S)}.mp4")


def _cut_all(rv, man: pd.DataFrame, cam: str, out: Path, windows: int) -> None:
    """Every labeled session of `cam`, `windows` non-overlapping cue-anchored windows each. The seed is per session
    (SEED + the video stem), so adding sessions or re-running never moves an existing session's windows; a window that
    starts within DUR_S of an existing clip of that session (e.g. from the one-per-animal x epoch run) is skipped."""
    import zlib
    m = man[man.cam == cam].drop_duplicates("video_stem").sort_values(["animal", "date"])
    print(f"{cam}: {len(m)} sessions x {windows} windows")
    for r in m.itertuples():
        tcsv = sorted((Path(rv.root("behavior_out")) / "sessions" / r.animal / r.date).glob("*_trials.csv"))
        vids = sorted((Path(rv.root("behavior_cameras")) / r.date / r.animal).glob(f"{cam}_*.avi"))
        if not tcsv or not vids:
            print(f"  {r.video_stem}: missing trials or video -> skip")
            continue
        cue = pd.read_csv(tcsv[0])["cue_s"].to_numpy()
        mid = cue[(cue > 60) & (cue < cue.max() - 60)]
        have = [int(p.stem.rsplit("_t", 1)[1]) for p in out.glob(f"{r.video_stem}_t*.mp4")]
        rng = np.random.default_rng([SEED, zlib.crc32(r.video_stem.encode())])
        for c in rng.permutation(mid):
            if len(have) >= windows:
                break
            t0 = int(float(c) - PRE_S)
            if all(abs(t0 - h) >= DUR_S for h in have):
                _write(cam, vids[0], out / f"{r.video_stem}_t{t0}.mp4")
                have.append(t0)


def _write(cam: str, video: Path, dst: Path) -> None:
    if dst.exists():
        print(f"  {dst.name}: exists")
        return
    t0 = float(dst.stem.rsplit("_t", 1)[1])
    vf = ["-vf", side_filter(cam, _width(video))] if cam in ("cam2", "cam3") else []
    subprocess.run([_ffmpeg(), "-hide_banner", "-loglevel", "error", "-ss", f"{t0:.3f}", "-i", str(video),
                    "-t", f"{DUR_S}", *vf, "-c:v", "libx264", "-crf", "15", "-preset", "veryfast",
                    "-pix_fmt", "yuv420p", "-an", str(dst)], check=True)
    print(f"  {dst.name}  ({dst.stat().st_size / 1e6:.0f} MB)  from cue at {t0 + PRE_S:.0f}s")


if __name__ == "__main__":
    raise SystemExit(main())
