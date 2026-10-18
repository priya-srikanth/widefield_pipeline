"""Labelled review clips of RAW pose predictions on a `session_poses` trial clip, one panel per model, the SAME frame in
every panel (Priya 2026-10-18: labelled clips for the cam1 round-2 model). Replaces the one-off
`inference_check_20261012_round4/render_models_grid.py`; for CLEANED traces use `scripts.cleaned_label_clips` (cam4 cleaning).

    python -m scripts.model_label_clips PS93:20260814:cam1 --model "DLC c1r2={d}/dlc_r2/*.csv" [--model "DLC c1r1={d}/dlc_r1/*.csv"] \\
        [--per-position 2] [--pre-ms 200] [--post-ms 1500] [--fps 30] [--tag r1_vs_r2]

``{d}`` = the clip folder. Trials: ``--per-position`` per spout position, evenly spread over the session (seeded order, the same
for every run). Points drawn where likelihood >= the camera's analysis cutoff (cam1: tongue 0.4, others 0.6 -- DECISIONS
2026-10-05; cam4: 0.4 for all, `orofacial_clean`), with 5-frame trails; nose yellow, jaw green, tongue red, spout cyan; a
hollow marker = below cutoff (shown only with ``--show-low``). Header: model, trial, position, ms from cue. Writes
``<clip folder>/eval/labeled_<tag>.mp4`` (h264, ``--fps`` playback of 250-fps video). Source clip READ-ONLY.
"""
from __future__ import annotations

import argparse
import glob
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

PARTS = ["nose", "jaw", "tongue", "spout"]
COL = {"nose": (0, 220, 255), "jaw": (60, 220, 60), "tongue": (60, 60, 255), "spout": (255, 220, 0)}   # BGR
CUTS = {"cam1": {"nose": 0.6, "jaw": 0.6, "tongue": 0.4, "spout": 0.6}, "cam4": {p: 0.4 for p in PARTS}}
TRAIL = 5


def pick_trials(idx: pd.DataFrame, per_position: int) -> list[int]:
    """``per_position`` trial_k per position, evenly spaced through the session."""
    out = []
    for _, g in idx.groupby("position"):
        ks = np.sort(g.trial_k.unique())
        out += list(ks[np.unique(np.linspace(0, len(ks) - 1, min(per_position, len(ks))).round().astype(int))])
    return sorted(out)


def main(argv=None) -> int:
    import cv2

    from scripts.session_poses import _ffmpeg, read_index, session_dir
    from wfield_local import orofacial_clean as oc
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", metavar="ANIMAL:YYYYMMDD[:cam1]")
    ap.add_argument("--model", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--per-position", type=int, default=2)
    ap.add_argument("--pre-ms", type=float, default=200.0)
    ap.add_argument("--post-ms", type=float, default=1500.0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--show-low", action="store_true", help="also draw below-cutoff points (hollow)")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args(argv)
    rv = PathResolver()
    _, _, d = session_dir(rv, a.session)
    cam = "cam1" if a.session.endswith(":cam1") else "cam4"
    cuts = CUTS[cam]
    idx = read_index(d)
    models = []
    for spec in a.model:
        name, pat = spec.split("=", 1)
        hits = glob.glob(pat.replace("{d}", str(d)))
        if len(hits) != 1:
            raise SystemExit(f"{pat!r}: {len(hits)} matches, need exactly one")
        P = oc.read_pose(Path(hits[0]))
        models.append((name, {p: P[[f"{p}_x", f"{p}_y", f"{p}_likelihood"]].to_numpy(float) for p in PARTS}))
    keep = idx.trial_k.isin(pick_trials(idx, a.per_position)) & idx.t_ms.between(-a.pre_ms, a.post_ms)
    rows = idx.index[keep].to_numpy()
    cap = cv2.VideoCapture(str(d / "windows.mp4"))                       # READ-ONLY
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ncol = min(3, len(models))
    nrow = int(np.ceil(len(models) / ncol))
    out = d / "eval" / f"labeled_{a.tag or '_'.join(n.replace(' ', '') for n, _ in models)}.mp4"
    out.parent.mkdir(exist_ok=True)
    ff = subprocess.Popen([_ffmpeg(), "-nostdin", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                           "-s", f"{w * ncol}x{h * nrow}", "-r", str(a.fps), "-i", "-", "-c:v", "libx264", "-crf", "18",
                           "-pix_fmt", "yuv420p", str(out)], stdin=subprocess.PIPE)
    pos = -1
    for r in rows:
        if r != pos + 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(r))
        ok, im = cap.read()
        pos = r
        if not ok:
            break
        grid = np.zeros((h * nrow, w * ncol, 3), np.uint8)
        same_trial = idx.trial_k.to_numpy()
        for m, (name, P) in enumerate(models):
            pan = im.copy()
            for p in PARTS:
                for k in range(max(0, r - TRAIL + 1), r + 1):
                    if same_trial[k] != same_trial[r]:
                        continue
                    x, y, lk = P[p][k]
                    if not np.isfinite(x) or (lk < cuts[p] and not a.show_low):
                        continue
                    rad, th = (5, -1) if k == r else (2, -1)
                    if lk < cuts[p]:
                        rad, th = (5, 1) if k == r else (2, 1)
                    cv2.circle(pan, (int(round(x)), int(round(y))), rad, COL[p], th)
            ir = idx.loc[r]
            cv2.putText(pan, f"{name}", (8, 22), 0, 0.6, (255, 255, 255), 2)
            cv2.putText(pan, f"trial {int(ir.trial_id)} {ir.position} {ir.t_ms:+.0f} ms", (8, 44), 0, 0.5,
                        (255, 255, 255), 1)
            rr, cc = divmod(m, ncol)
            grid[rr * h:(rr + 1) * h, cc * w:(cc + 1) * w] = pan
        ff.stdin.write(grid.tobytes())
    cap.release()
    ff.stdin.close()
    ff.wait()
    print(f"{len(rows)} frames ({len(rows) / a.fps / 60:.1f} min at {a.fps:g} fps) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
