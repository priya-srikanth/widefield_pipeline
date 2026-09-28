"""Unlabelled clips for Lightning Pose's semi-supervised losses (temporal, pose-PCA).

    conda activate dlc
    python -m scripts.lp_unlabelled_clips                # -> <dlc root>/lightning-pose/<project>/videos/

Each clip is cut from a CUE so it contains the whole failure surface in one window: ENL, cue, the
3.5 s response window with its licking, and the following inter-trial gap where the spout moves.
Sampling only lick-dense windows (as the throughput check clips do) would train the temporal loss on
the licking and never on the interval where the spout dropout happened.

Sessions are the LABELLED ones (from the frame manifest), one per animal x epoch, so the unlabelled
distribution matches the supervised one across all four animals and all four epochs. Seeded, and
idempotent: an existing clip is kept, so re-running after the label set grows adds nothing and
removes nothing. Recovered into the repo 2026-09-28 from the 2026-09-26 scratch script that cut the
15 clips in `lightning-pose/cam4-2026-09-26/videos/` (7,500 frames each, 0.59 GB).
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from wfield_local.dlc_frames import staging_root
from wfield_local.paths import PathResolver

DUR_S, PRE_S, SEED = 30.0, 3.0, 92


def _ffmpeg() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def pick_sessions(manifest: pd.DataFrame, cam: str = "cam4") -> pd.DataFrame:
    """One labelled session per (animal, epoch): the first in a deterministic sort."""
    m = manifest[manifest.cam == cam].drop_duplicates("video_stem")[["animal", "date", "video_stem", "epoch"]]
    return m.sort_values(["epoch", "animal"]).groupby(["animal", "epoch"], as_index=False).first()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", default="cam4-2026-09-26", help="LP project dir name under <dlc root>/lightning-pose/")
    ap.add_argument("--cam", default="cam4")
    a = ap.parse_args(argv)
    rv = PathResolver()
    out = Path(rv.root("dlc")) / "lightning-pose" / a.project / "videos"
    out.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(staging_root(rv) / "frame_manifest.csv", dtype=str)
    picked = pick_sessions(man, a.cam)
    rng = np.random.default_rng(SEED)
    print(f"{len(picked)} sessions: " + ", ".join(f"{r.animal}/{r.epoch}" for r in picked.itertuples()))
    for r in picked.itertuples():
        tcsv = sorted((Path(rv.root("behavior_out")) / "sessions" / r.animal / r.date).glob("*_trials.csv"))
        vids = sorted((Path(rv.root("behavior_cameras")) / r.date / r.animal).glob(f"{a.cam}_*.avi"))
        if not tcsv or not vids:
            print(f"  {r.video_stem}: missing trials or video -> skip")
            continue
        cue = pd.read_csv(tcsv[0])["cue_s"].to_numpy()
        mid = cue[(cue > 60) & (cue < cue.max() - 60)]
        if not len(mid):
            continue
        t0 = float(rng.choice(mid)) - PRE_S
        dst = out / f"{r.video_stem}_t{int(t0)}.mp4"
        if dst.exists():
            print(f"  {dst.name}: exists")
            continue
        subprocess.run([_ffmpeg(), "-hide_banner", "-loglevel", "error", "-ss", f"{t0:.3f}", "-i", str(vids[0]),
                        "-t", f"{DUR_S}", "-c:v", "libx264", "-crf", "15", "-preset", "veryfast",
                        "-pix_fmt", "yuv420p", "-an", str(dst)], check=True)
        print(f"  {dst.name}  ({dst.stat().st_size / 1e6:.0f} MB)  from cue at {t0 + PRE_S:.0f}s")
    clips = list(out.glob("*.mp4"))
    print(f"\n{len(clips)} clips, {sum(f.stat().st_size for f in clips) / 1e9:.2f} GB, {DUR_S * 250:.0f} frames each")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
