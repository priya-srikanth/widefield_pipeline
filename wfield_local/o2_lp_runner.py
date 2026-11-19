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
* LP "not visible" points are parked near the image center at low likelihood -- downstream always applies a
  likelihood cut (the cleaning does).
* SIDE CAMERAS (``--side``, 2026-11-18, side model r2b): the shared cam2 / cam3 model was trained on TRANSFORMED frames
  (`wfield_local.lp_sidecam.transform_image`: cam3 mirrored, square crop by frame width, June 800 px resized to 256),
  so each frame is transformed here (`side_transform`, a dependency-free copy -- tests/test_o2_inference.py pins it to
  `lp_sidecam`) and written to a temporary mp4 with the SAME encoder settings as the review clips the model was
  validated on (`scripts/side_review_clips.py`: gray, libx264 crf 12 veryfast, yuv420p), then predicted. Outputs stay
  in MODEL space; `pose3d_side.side_windows` maps them back to the raw frame. The camera comes from the video name.
  ``--frames A:B`` (local smoke tests only) predicts that frame span and writes ``<stem>_fA-B`` outputs.
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


#: `wfield_local.lp_sidecam` CROPS / OUT, copied (this file imports nothing from the package): frame width ->
#: (x0, y0, square side) of the crop; the square is then resized to SIDE_OUT px when it is not already that size.
SIDE_CROPS = {600: (0, 50, 256), 800: (0, 62, 317)}
SIDE_OUT = 256


def side_cam(video: Path) -> str:
    cam = video.name.split("_", 1)[0]
    if cam not in ("cam2", "cam3"):
        raise SystemExit(f"{video.name}: --side needs a cam2_* / cam3_* video")
    return cam


def side_transform(gray: np.ndarray, cam: str) -> np.ndarray:
    """One gray frame (h, w) uint8 -> the side model's input (SIDE_OUT, SIDE_OUT) uint8, exactly as
    `lp_sidecam.transform_image` (mirror, crop, PIL BILINEAR resize). Mirror + crop are numpy slices (11-18 profile:
    PIL's fromarray / transpose / crop ran 1330 fps and set the pace of the whole transform; slicing 8950 fps, same
    pixels); only June's 317 -> 256 resize goes through PIL, so it stays PIL's BILINEAR."""
    w = gray.shape[1]
    if w not in SIDE_CROPS:
        raise SystemExit(f"side frame width {w} has no crop (known: {sorted(SIDE_CROPS)})")
    x0, y0, s = SIDE_CROPS[w]
    if cam == "cam3":                                   # mirror, then crop columns x0 .. x0 + s of the MIRRORED frame
        out = gray[y0:y0 + s, w - 1 - x0::-1][:, :s] if x0 == 0 else gray[y0:y0 + s, ::-1][:, x0:x0 + s]
    else:
        out = gray[y0:y0 + s, x0:x0 + s]
    if s != SIDE_OUT:
        from PIL import Image
        return np.asarray(Image.fromarray(np.ascontiguousarray(out)).resize((SIDE_OUT, SIDE_OUT), Image.BILINEAR),
                          dtype=np.uint8)
    return np.ascontiguousarray(out)


def _ffmpeg() -> str:
    import shutil
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg                 # pip install imageio-ffmpeg (O2 `lp` env, COMMANDS.md step 0)
    return imageio_ffmpeg.get_ffmpeg_exe()


def write_side_video(video: Path, dst: Path, cam: str, start: int = 0, stop: int | None = None) -> int:
    """Frames [start, stop) of ``video`` transformed (`side_transform`) into ``dst`` (mp4, the review clips' encoder
    settings). Returns the number of frames written."""
    import subprocess

    import cv2
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 250.0
    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    ff = subprocess.Popen([_ffmpeg(), "-nostdin", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "gray",
                           "-s", f"{SIDE_OUT}x{SIDE_OUT}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-crf", "12",
                           "-preset", "veryfast", "-pix_fmt", "yuv420p", str(dst)], stdin=subprocess.PIPE)
    # decode in a background thread (OpenCV releases the GIL while decoding), transform + pipe here: decode is then the
    # pace (~2700 fps on the desktop) instead of decode + transform in series
    import queue
    import threading
    q: queue.Queue = queue.Queue(maxsize=256)
    n_read = (stop - start) if stop is not None else None

    def reader():
        k = 0
        while n_read is None or k < n_read:
            ok, im = cap.read()
            if not ok:
                break
            q.put(im[..., 0])
            k += 1
        q.put(None)
    th = threading.Thread(target=reader, daemon=True)
    th.start()
    n = 0
    while (g := q.get()) is not None:
        ff.stdin.write(side_transform(g, cam).tobytes())
        n += 1
    th.join()
    ff.stdin.close()
    if ff.wait() != 0:
        raise SystemExit(f"ffmpeg failed writing {dst}")
    return n


def run_video(model_dir: Path, video: Path, out: Path, *, reader: str = "opencv", precision: str = "fp32",
              scorer: str | None = None, side: bool = False, frames: tuple[int, int] | None = None) -> dict:
    from lightning_pose.api.model import Model
    stem = video.stem if frames is None else f"{video.stem}_f{frames[0]}-{frames[1]}"
    if (out / f"{stem}_done.json").exists():
        print(f"{stem}: done already ({out / (stem + '_done.json')}), skipping", flush=True)
        return json.loads((out / f"{stem}_done.json").read_text(encoding="utf-8"))
    n_vid, fps = n_video_frames(video)
    if frames is not None:
        n_vid = min(frames[1], n_vid) - frames[0]
    print(f"{stem}: {n_vid} frames at {fps} fps, reader {reader}, precision {precision}"
          f"{f', side transform ({side_cam(video)})' if side else ''}", flush=True)
    model = Model.from_dir(str(model_dir), precision=precision)
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    extra = {}
    # same filesystem as the outputs; ignore_cleanup_errors: on Windows LP's reader still holds the temporary side mp4
    # open at exit (11-18 smoke test) -- the leftover tmp* folder is excluded from the copy-back (COMMANDS.md step 5)
    with tempfile.TemporaryDirectory(dir=out, ignore_cleanup_errors=True) as tmp:
        src = video
        if side or frames is not None:
            if not side:
                raise SystemExit("--frames is for side smoke tests (it re-encodes the span through the side transform)")
            src = Path(tmp) / f"{stem}.mp4"
            n_w = write_side_video(video, src, side_cam(video), *(frames or (0, None)))
            extra = {"side_cam": side_cam(video), "side_transform": "lp_sidecam (mirror cam3, crop, 256)",
                     "side_encode": "gray libx264 crf12 veryfast yuv420p", "transform_s": round(time.time() - t0, 1)}
            if n_w != n_vid:
                raise SystemExit(f"{stem}: wrote {n_w} transformed frames vs {n_vid} expected -- not predicting")
        model.predict_on_video_file(str(src), output_dir=tmp, compute_metrics=False, reader=reader)
        pose, parts, lp_scorer = lp_csv_to_pose(Path(tmp) / f"{src.stem}.csv")
    wall = time.time() - t0
    if len(pose) != n_vid:
        raise SystemExit(f"{stem}: {len(pose)} predicted rows vs {n_vid} video frames -- not writing outputs")
    info = {"fps": fps, "wall_s": round(wall, 1), "batch": None, "reader": reader, "precision": precision,
            "model_dir": str(model_dir), "lp_scorer": lp_scorer, **extra,
            **({"frames": list(frames)} if frames is not None else {})}
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
    ap.add_argument("--side", action="store_true", help="side model: transform cam2 / cam3 frames first (docstring)")
    ap.add_argument("--frames", default=None, metavar="A:B", help="smoke tests: only frames [A, B) (with --side)")
    a = ap.parse_args(argv)
    env_report(require_gpu=not a.allow_cpu)
    if a.bench:
        bench(a.model, a.video, a.bench, precision=a.precision)
        return 0
    if a.out is None:
        ap.error("--out is required unless --bench")
    fr = tuple(int(x) for x in a.frames.split(":")) if a.frames else None
    run_video(a.model, a.video, a.out, reader=a.reader, precision=a.precision, scorer=a.scorer, side=a.side,
              frames=fr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
