#!/usr/bin/env python3
"""Standalone cam4 Lightning Pose inference for HMS O2 (or any GPU box): one whole video per call. NO
widefield_pipeline imports -- `o2_inference bundle --kind lp` copies this file next to the model, so it runs in a
plain `lp` env (lightning-pose 2.4.2) on O2.

    python o2_lp_runner.py --model <bundle>/model --video <cam4_*.avi> --out <dir> [--reader opencv]
    python o2_lp_runner.py --model <bundle>/model --video <avi> --bench 2000        # fps on the first N frames

Priya, 2026-10-15: LP r4 (single-frame heatmap) is the production cam4 model; "start prepping the code we'll need to
run LP on O2 across sessions". The counterpart of `o2_pose_runner.py` (DLC), with the same OUTPUT CONTRACT, so
`o2_inference collect` handles both: <stem>_<scorer>.csv (3-row header, one row per video frame), <stem>_<scorer>.npz
(pose (n, parts, 3) = x, y, likelihood; parts), <stem>_done.json (n_frames, fps, wall_s, scorer, batch, reader).

* Prediction = LP's own `Model.predict_on_video_file` (compute_metrics=False: the metrics need the training data
  module; we only want the predictions).
* READER: our cam4 videos are MPEG-4 part 2 in AVI (FMP4), which DALI's GPU decoder is not known to read, so the
  default is "opencv" (CPU decode -- the same path as the desktop's Windows-native LP inference). DALI and OpenCV
  preprocess slightly differently (LP's own docs): never mix readers within one analysis, and check O2 output
  against the desktop on a shared session (`o2_inference collect` reports it for PS93 0814).
* NOT resumable inside a video (LP predicts a file in one pass). A finished video is skipped on a re-run (done.json),
  so an array requeue repeats only unfinished videos.
* LP "not visible" points are parked near the image centre at low likelihood -- downstream always applies a
  likelihood cut (the cleaning does).
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np


def lp_csv_to_pose(csv_path: Path) -> tuple[np.ndarray, list[str], str]:
    """LP's prediction CSV (3-row header scorer / bodyparts / coords, index = frame) -> (pose (n, parts, 3), parts,
    scorer). Reads only x / y / likelihood, in the CSV's bodypart order."""
    import pandas as pd
    df = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)
    scorer = str(df.columns.get_level_values(0)[0])
    df.columns = df.columns.droplevel(0)
    parts = list(dict.fromkeys(df.columns.get_level_values(0)))
    pose = np.stack([df[p][["x", "y", "likelihood"]].to_numpy(np.float32) for p in parts], axis=1)
    return pose, parts, scorer


def write_outputs(out: Path, stem: str, pose: np.ndarray, parts: list[str], scorer: str, info: dict) -> None:
    """The shared o2 output contract (see the module docstring)."""
    import pandas as pd
    out.mkdir(parents=True, exist_ok=True)
    cols = pd.MultiIndex.from_product([[scorer], parts, ["x", "y", "likelihood"]],
                                      names=["scorer", "bodyparts", "coords"])
    pd.DataFrame(pose.reshape(len(pose), -1), columns=cols).to_csv(out / f"{stem}_{scorer}.csv")
    np.savez_compressed(out / f"{stem}_{scorer}.npz", pose=pose, parts=np.array(parts))
    (out / f"{stem}_done.json").write_text(json.dumps({**info, "scorer": scorer, "n_frames": int(len(pose))},
                                                      indent=1), encoding="utf-8")


def default_scorer(model_dir: Path) -> str:
    """The bundle's scorer (bundle.json next to the model dir, e.g. LP_r4_heatmap_20261013); else LP_<model dir>.
    In a bundle the model dir is just `model`, so its name alone says nothing."""
    bj = model_dir.parent / "bundle.json"
    if bj.exists():
        return str(json.loads(bj.read_text(encoding="utf-8"))["scorer"])
    return f"LP_{model_dir.name}"


def n_video_frames(video: Path) -> tuple[int, float]:
    import cv2
    cap = cv2.VideoCapture(str(video))
    return int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), float(cap.get(cv2.CAP_PROP_FPS))


def first_frames_clip(video: Path, n: int, dst: Path) -> Path:
    """The first ``n`` frames of ``video`` as an mp4 (for --bench / smoke tests)."""
    import cv2
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 250.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for _ in range(n):
        ok, im = cap.read()
        if not ok:
            break
        vw.write(im)
    vw.release()
    return dst


def run_video(model_dir: Path, video: Path, out: Path, *, reader: str = "opencv", precision: str = "fp32",
              scorer: str | None = None) -> dict:
    from lightning_pose.api.model import Model
    stem = video.stem
    if (out / f"{stem}_done.json").exists():
        print(f"{stem}: done already ({out / (stem + '_done.json')}), skipping", flush=True)
        return json.loads((out / f"{stem}_done.json").read_text(encoding="utf-8"))
    n_vid, fps = n_video_frames(video)
    print(f"{stem}: {n_vid} frames at {fps} fps, reader {reader}, precision {precision}", flush=True)
    model = Model.from_dir(str(model_dir), precision=precision)
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out) as tmp:                  # same filesystem as the outputs
        model.predict_on_video_file(str(video), output_dir=tmp, compute_metrics=False, reader=reader)
        pose, parts, lp_scorer = lp_csv_to_pose(Path(tmp) / f"{stem}.csv")
    wall = time.time() - t0
    if len(pose) != n_vid:
        raise SystemExit(f"{stem}: {len(pose)} predicted rows vs {n_vid} video frames -- not writing outputs")
    info = {"fps": fps, "wall_s": round(wall, 1), "batch": None, "reader": reader, "precision": precision,
            "model_dir": str(model_dir), "lp_scorer": lp_scorer}
    write_outputs(out, stem, pose, parts, scorer or default_scorer(Path(model_dir)), info)
    print(f"{stem}: {len(pose)} frames in {wall / 3600:.2f} h ({len(pose) / max(wall, 1e-9):.0f} fps) -> {out}",
          flush=True)
    return info


def bench(model_dir: Path, video: Path, n: int, readers=("opencv", "dali"), precision: str = "fp32") -> None:
    """Frames/s on the first ``n`` frames for each reader (a reader that fails is reported, not fatal)."""
    from lightning_pose.api.model import Model
    model = Model.from_dir(str(model_dir), precision=precision)
    with tempfile.TemporaryDirectory() as tmp:
        clip = first_frames_clip(video, n, Path(tmp) / "bench.mp4")
        for r in readers:
            try:
                t0 = time.time()
                model.predict_on_video_file(str(clip), output_dir=tmp, compute_metrics=False, reader=r)
                print(f"reader {r}: {n / (time.time() - t0):.0f} fps over {n} frames (incl. model warm-up)",
                      flush=True)
            except Exception as e:                                   # noqa: BLE001 -- report and go on
                print(f"reader {r}: FAILED ({type(e).__name__}: {e})", flush=True)
    print("NB: the bench clip is re-encoded mp4v, not the original AVI -- run one real video before trusting a "
          "non-opencv reader on the AVIs.", flush=True)


def env_report(require_gpu: bool = True) -> None:
    """Print torch / CUDA / GPU / LP / OpenCV versions, and stop if no GPU is visible (a job silently running LP on
    CPU would take days per video). This IS the GPU check of the O2 commands -- no interactive srun needed."""
    import importlib.metadata as md

    import cv2
    import torch
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE"
    print(f"torch {torch.__version__} | cuda {torch.cuda.is_available()} | gpu {gpu} | "
          f"lightning-pose {md.version('lightning-pose')} | cv2 {cv2.__version__}", flush=True)
    if require_gpu and not torch.cuda.is_available():
        raise SystemExit("no GPU visible to torch -- check --gres and the env's CUDA build")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True, help="LP model dir (config.yaml + tb_logs/.../*.ckpt)")
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--reader", default="opencv", choices=("opencv", "dali", "pynvvc"))
    ap.add_argument("--precision", default="fp32", choices=("fp32", "fp16", "bf16"))
    ap.add_argument("--scorer", default=None, help="scorer name in the outputs (default LP_<model dir name>)")
    ap.add_argument("--bench", type=int, default=None, help="time the first N frames per reader; write nothing")
    ap.add_argument("--allow-cpu", action="store_true", help="tests only: run without a GPU")
    a = ap.parse_args(argv)
    env_report(require_gpu=not a.allow_cpu)
    if a.bench:
        bench(a.model, a.video, a.bench, precision=a.precision)
        return 0
    if a.out is None:
        ap.error("--out is required unless --bench")
    run_video(a.model, a.video, a.out, reader=a.reader, precision=a.precision, scorer=a.scorer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
