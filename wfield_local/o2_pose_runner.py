#!/usr/bin/env python3
"""Standalone cam4 pose inference for the HMS O2 cluster (or any GPU box): DLC round-N model + the spatial prior,
whole video, resumable in chunks. NO widefield_pipeline imports -- `o2_inference.bundle` copies this file next to
the model, so it runs in a plain DeepLabCut-3 env on O2.

    python o2_pose_runner.py --bundle <bundle dir> --video <cam4_*.avi> --out <dir> [--batch 16] [--chunk 25000]
    python o2_pose_runner.py --bundle <dir> --video <avi> --bench 2000          # fps per batch size, writes nothing

PORTED FROM stroke_orofacial_pipeline `dlc_inference_o2/runner.py` (the generated `run_dlc_batch.py`), and changed:
  * Prediction is OURS, not `deeplabcut.analyze_videos`: the same code path as `dlc_hard_frames.pose_predictor`
    (PoseModel built from the train dir's pytorch_config.yaml, ImageNet normalisation, `get_predictions`) with
    the per-bodypart heatmap prior of `dlc_prior.masking` (boxes from the bundle's prior.json), so O2 output is
    identical to local output (tested: tests/test_o2_inference.py). analyze_videos would skip the prior.
  * DLC 3 / PyTorch (theirs: TensorFlow, allow_growth, shuffle 5, snapshot renames) -- none of that applies.
  * One video per call (the SBATCH job ARRAY fans out over sessions), resumable: every CHUNK frames are written
    as soon as they are done and skipped on a re-run, so a time-limit kill or a requeue loses at most one chunk.
  * Output: <stem>_<scorer>.csv in DLC's 3-row-header format (scorer / bodyparts / coords, one row per VIDEO
    FRAME, frame 0 onward) + the same as .npz, plus <stem>_done.json (frame count, fps, model, prior).
  * --bench: frames/s for several batch sizes on the real video (Priya: batch 1 was fastest in earlier tests --
    measure it on the node instead of assuming).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def load_bundle(bundle: Path) -> dict:
    """bundle.json: {"parts": [...], "scorer": str, "pytorch_config": rel path, "snapshot": rel path,
    "prior": {part: [x0, x1, y0, y1]} (full-frame px; parts without a box are not masked)}."""
    b = json.loads((bundle / "bundle.json").read_text(encoding="utf-8"))
    b["pytorch_config"] = str(bundle / b["pytorch_config"])
    b["snapshot"] = str(bundle / b["snapshot"])
    return b


def install_prior(parts: list[str], box: dict) -> None:
    """Patch DLC's HeatmapPredictor so every heatmap is masked to its part's box before the argmax --
    `dlc_prior.masking` (partial=True, offset 0), transcribed. Process-wide for this runner, on purpose."""
    import torch
    from deeplabcut.pose_estimation_pytorch.models.predictors.single_predictor import HeatmapPredictor

    if not box:
        return
    original = HeatmapPredictor.forward
    cache: dict = {}

    def patched(self, stride, outputs):
        hm = outputs["heatmap"]
        if hm.shape[1] != len(parts):
            raise SystemExit(f"prior has {len(parts)} bodyparts, heatmap has {hm.shape[1]} channels")
        h, w = hm.shape[-2:]
        s = float(stride)
        key = (h, w, s, str(hm.device), hm.dtype)
        if key not in cache:
            gy, gx = torch.meshgrid(torch.arange(h, device=hm.device, dtype=hm.dtype) * s,
                                    torch.arange(w, device=hm.device, dtype=hm.dtype) * s, indexing="ij")
            drop = torch.zeros((len(parts), h, w), device=hm.device, dtype=torch.bool)
            for k, bp in enumerate(parts):
                if bp not in box:
                    continue
                x0, x1, y0, y1 = box[bp]
                ok = (gx >= x0) & (gx <= x1) & (gy >= y0) & (gy <= y1)
                if not bool(ok.any()):
                    raise SystemExit(f"prior box for {bp} covers no heatmap cell at stride {s}")
                drop[k] = ~ok
            cache[key] = drop
        hm = hm.clone()
        hm.masked_fill_(cache[key].unsqueeze(0), -1e4)
        outputs = dict(outputs)
        outputs["heatmap"] = hm
        return original(self, stride, outputs)

    HeatmapPredictor.forward = patched


def build_predictor(b: dict, batch: int = 16, device: str | None = None):
    """``predict(frames_bgr) -> (N, n_parts, 3)`` [x, y, p] -- `dlc_hard_frames.pose_predictor`, transcribed."""
    import cv2
    import torch
    from deeplabcut.pose_estimation_pytorch.config import read_config_as_dict
    from deeplabcut.pose_estimation_pytorch.models import PoseModel

    cfg = read_config_as_dict(b["pytorch_config"])
    model = PoseModel.build(cfg["model"])
    model.load_state_dict(torch.load(b["snapshot"], map_location="cpu", weights_only=True)["model"])
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.eval().to(device)
    install_prior(list(b["parts"]), {k: tuple(v) for k, v in (b.get("prior") or {}).items()})

    def predict(frames) -> np.ndarray:
        out = []
        for k in range(0, len(frames), batch):
            a = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames[k:k + batch]]).astype(np.float32) / 255.0
            t = torch.from_numpy(((a - MEAN) / STD).transpose(0, 3, 1, 2)).to(device)
            with torch.inference_mode():
                out.append(model.get_predictions(model(t))["bodypart"]["poses"][:, 0].cpu().numpy())
        return np.concatenate(out)

    return predict


def run_video(b: dict, video: Path, out: Path, *, batch: int = 16, chunk: int = 25_000, read_n: int = 64,
              max_frames: int | None = None, device: str | None = None) -> Path:
    """Whole video (or the first ``max_frames``) -> chunk npz files -> final csv / npz / done.json."""
    import cv2
    out.mkdir(parents=True, exist_ok=True)
    cd = out / f"{video.stem}_chunks"
    cd.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(str(video))                        # READ-ONLY
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames is not None:
        n = min(n, int(max_frames))
    predict, pos, t_all, done_frames = None, 0, time.time(), 0
    for c0 in range(0, n, chunk):
        c1 = min(n, c0 + chunk)
        f = cd / f"{c0:09d}_{c1:09d}.npz"
        if f.exists():
            continue
        predict = predict or build_predictor(b, batch=batch, device=device)
        if pos != c0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, c0)
        t, res, buf, k = time.time(), [], [], c0
        while k < c1:
            ok, im = cap.read()
            if not ok:
                break
            buf.append(im)
            k += 1
            if len(buf) == read_n or k == c1:
                res.append(predict(buf))
                buf = []
        pos = k
        P = np.concatenate(res).astype(np.float32) if res else np.empty((0, len(b["parts"]), 3), np.float32)
        np.savez_compressed(f, f0=c0, pose=P)
        done_frames += len(P)
        print(f"{video.name}: {c0}-{k} of {n} ({len(P) / max(time.time() - t, 1e-9):.0f} fps)", flush=True)
        if k < c1:
            print(f"  video ended early at frame {k} (header said {n})", flush=True)
            n = k
            break
    cap.release()
    chunks = sorted(cd.glob("*.npz"))
    P = np.concatenate([np.load(c)["pose"] for c in chunks]) if chunks else np.empty((0, len(b["parts"]), 3))
    if len(P) != n:
        raise SystemExit(f"{video.name}: {len(P)} frames predicted of {n} -- re-run to resume")
    base = f"{video.stem}_{b['scorer']}"
    np.savez_compressed(out / f"{base}.npz", pose=P, parts=np.array(b["parts"]))
    write_csv(out / f"{base}.csv", P, b["parts"], b["scorer"])
    (out / f"{video.stem}_done.json").write_text(json.dumps({
        "video": str(video), "n_frames": int(n), "scorer": b["scorer"], "parts": b["parts"],
        "prior": b.get("prior"), "snapshot": Path(b["snapshot"]).name, "batch": batch,
        "wall_s": round(time.time() - t_all, 1)}, indent=1), encoding="utf-8")
    print(f"-> {out / (base + '.csv')} ({n} frames)", flush=True)
    return out / f"{base}.csv"


def write_csv(path: Path, P: np.ndarray, parts: list[str], scorer: str) -> None:
    """DLC's 3-row header (scorer / bodyparts / coords), index = video frame -- what `orofacial_clean.read_pose`
    and the local scripts read. Written with numpy (no pandas needed on the node)."""
    hdr = [",".join(["scorer"] + [scorer] * (3 * len(parts))),
           ",".join(["bodyparts"] + [p for p in parts for _ in range(3)]),
           ",".join(["coords"] + ["x", "y", "likelihood"] * len(parts))]
    flat = P.reshape(len(P), -1)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(hdr) + "\n")
        np.savetxt(fh, np.column_stack([np.arange(len(P)), flat]), delimiter=",",
                   fmt=["%d"] + ["%.4f"] * flat.shape[1])


def bench(b: dict, video: Path, n: int, batches=(1, 4, 8, 16, 32, 64)) -> None:
    import cv2
    import torch
    cap = cv2.VideoCapture(str(video))
    frames = [cap.read()[1] for _ in range(n)]
    cap.release()
    for bs in batches:
        try:
            p = build_predictor(b, batch=bs)
            p(frames[:max(bs, 16)])
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            t = time.time()
            p(frames)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            mem = torch.cuda.max_memory_allocated() / 2 ** 20 if torch.cuda.is_available() else float("nan")
            print(f"batch {bs:3d}: {n / (time.time() - t):7.1f} fps, peak {mem:.0f} MiB", flush=True)
        except RuntimeError as e:                              # OOM on a large batch: report and stop
            print(f"batch {bs}: failed ({str(e)[:80]})", flush=True)
            break


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=25_000)
    ap.add_argument("--max-frames", type=int, default=None, help="only the first N frames (tests)")
    ap.add_argument("--bench", type=int, default=None, help="time N frames at several batch sizes; write nothing")
    a = ap.parse_args(argv)
    b = load_bundle(a.bundle)
    if a.bench:
        bench(b, a.video, a.bench)
        return 0
    if a.out is None:
        ap.error("--out is required unless --bench")
    run_video(b, a.video, a.out, batch=a.batch, chunk=a.chunk, max_frames=a.max_frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
