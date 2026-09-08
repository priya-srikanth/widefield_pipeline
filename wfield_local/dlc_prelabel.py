"""Seed DLC's labels with the 2pRAM network's predictions, so labelling is CORRECTION not blank-page.

The donor network (``MICROSCOPE/Priya/DeepLabCut/DLC_train_config``, iteration-5 shuffle5, test RMSE
2.70 px) was trained on a view this rig does not have — the old ``video2`` framed the whole head at
~2.5x less zoom than ``cam4``. DLC does not rescale at inference, so run on native cam4 frames it
finds the nose on **8%** of them. Feed it the same frames resized to ``dlc.prelabel.scale`` and it
finds the nose on **97%**. Everything here follows from that: the weights are fine, the apparent
size was wrong.

Measured on the 312 extracted cam4 frames, 2026-09-08, at scale 0.45 (fraction of frames above 0.6):

    nose          0.97   dead on the nose pad; x holds 336-341 px across all six spout positions,
                         which is what head fixation says it must do
    L_whiskers_*  0.95-1.00     R_whiskers_2  0.99     R_whiskers_1/3  0.81-0.85
    jaw           0.78   correct at the chin when the mouth is visible
    spout         0.75   max-of-(L_spout, R_spout); lands at x=255/300/343/346/397/439 for
                         far_R/close_R/far_center/close_center/close_L/far_L -- it TRACKS the moving
                         spout, which is the non-trivial part
    tongue        0.13   NOT WRITTEN -- see below
    L_eye/R_eye   0.60   NOT WRITTEN -- hallucinated, the eyes are outside cam4's field of view

**Two categories are deliberately withheld, and the likelihood does not tell them apart.** The eyes
are not in frame at all, yet come back at 0.44-0.79 in the top corners. The tongue fires at 0.99 on
the SPOUT while the tongue is out beside it. Its overall rate is honest — out on 27% of `early`
(+0.4 s) frames versus 1% of ENL frames, which is licking behaviour, not failure — but a
confidently misplaced point is worse than a blank one, because a point that is already placed
invites being accepted rather than checked. Blank cells are what the labelling GUI shows as "you
must place this".

RUN THIS FROM THE ``dlc`` ENV, not ``locanmf``: DeepLabCut is installed only there (``pip install -e
. --no-deps`` makes this repo importable from it). The donor project is copied locally before use
and the original on MICROSCOPE is opened READ-ONLY — DLC writes into a project directory as a matter
of course, so pointing it at the original would modify irreplaceable source data.

CLI::

    conda activate dlc
    python -m wfield_local.dlc_prelabel --cam cam4              # predict + write CollectedData
    python -m wfield_local.dlc_prelabel --cam cam4 --dry-run    # report coverage, write nothing
    python -m wfield_local.dlc_prelabel --cam cam4 --scale 0.6  # override the sweep's choice
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from wfield_local import config
from wfield_local.dlc_frames import bodyparts, out_root
from wfield_local.paths import PathResolver
from wfield_local.writeguard import assert_writable

SCORER = "Priya"          #: DLC keys labels by scorer; matches the donor project's own


def _cfg() -> dict:
    return (config.defaults().get("dlc") or {})


def donor() -> dict:
    return _cfg().get("donor") or {}


def prelabel_cfg() -> dict:
    return _cfg().get("prelabel") or {}


def ensure_local_donor(force: bool = False) -> Path:
    """Copy the donor project's config + one snapshot to local disk. Returns the local config path.

    THE ORIGINAL IS NEVER USED IN PLACE. `analyze_videos` writes into the project directory as a
    matter of course, and that project is irreplaceable source data on MICROSCOPE (CLAUDE.md rule 1).
    Only the files inference needs are copied — one snapshot, not all five — so this is ~95 MB
    rather than 556 MB.
    """
    d = donor()
    src, dst = Path(d["project"]), Path(d["local_copy"])
    model = f"dlc-models-pytorch/iteration-5/video2Jan26-trainset80shuffle{d['shuffle']}"
    dataset = "training-datasets/iteration-5/UnaugmentedDataSet_video2Jan26"
    wanted = [
        (f"{model}/train/pytorch_config.yaml", True),
        (f"{model}/train/{d['snapshot']}", True),
        (f"{model}/test/pose_cfg.yaml", True),
        (f"{dataset}/metadata.yaml", True),
        (f"{dataset}/Documentation_data-video2_80shuffle{d['shuffle']}.pickle", True),
    ]
    cfg_path = dst / "config.yaml"
    if cfg_path.exists() and not force and all((dst / rel).exists() for rel, _ in wanted):
        return cfg_path

    assert_writable(dst)
    for rel, _ in wanted:
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        if not (dst / rel).exists() or force:
            shutil.copy2(src / rel, dst / rel)
    (dst / "videos").mkdir(exist_ok=True)
    (dst / "labeled-data").mkdir(exist_ok=True)

    import yaml
    cfg = yaml.safe_load((src / "config.yaml").read_text(encoding="utf-8"))
    cfg["project_path"] = str(dst).replace("\\", "/")
    # -1 (not "best"): DLC 3.0's ProjectConfig validates snapshotindex as int|'all', and only the
    # one snapshot we copied is present, so -1 selects it.
    cfg["snapshotindex"] = -1
    cfg["video_sets"] = {}
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    mp = dst / model / "train/pytorch_config.yaml"
    m = yaml.safe_load(mp.read_text(encoding="utf-8"))
    m["metadata"]["project_path"] = cfg["project_path"]
    m["metadata"]["pose_config_path"] = str(mp).replace("\\", "/")
    mp.write_text(yaml.safe_dump(m, sort_keys=False), encoding="utf-8")
    return cfg_path


def frame_index(cam: str, rv=None) -> pd.DataFrame:
    """Every extracted frame for ``cam``, in a stable order: columns ``video_stem, image, path``."""
    root = out_root(rv) / "labeled-data"
    rows = [{"video_stem": d.name, "image": p.name, "path": str(p)}
            for d in sorted(root.iterdir()) if d.is_dir() and d.name.startswith(cam)
            for p in sorted(d.glob("img*.png"))]
    return pd.DataFrame(rows, columns=["video_stem", "image", "path"])


def build_video(index: pd.DataFrame, scale: float, dest: Path) -> tuple[int, int]:
    """Write the frames as one video at ``scale``. Returns the (width, height) actually used.

    A video rather than DLC's image API because `analyze_videos` is the path that is exercised in
    production, on O2, against the real recordings -- a pre-label produced through a different code
    path could differ from what the same weights will do later and nobody would know why.

    Sizes are rounded to a multiple of 32: the backbone's output stride is 16, and a size that is not
    a clean multiple leaves the mapping from heatmap peak back to input pixel dependent on DLC's
    internal padding.
    """
    import cv2

    first = cv2.imread(index.iloc[0]["path"])
    h0, w0 = first.shape[:2]
    dw, dh = (max(32, round(w0 * scale / 32) * 32), max(32, round(h0 * scale / 32) * 32))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    assert_writable(dest.parent)
    dest.parent.mkdir(parents=True, exist_ok=True)
    # FFV1 is lossless. A lossy intermediate would put compression artefacts between the frames the
    # network sees here and the frames it will see in production.
    vw = cv2.VideoWriter(str(dest), cv2.VideoWriter_fourcc(*"FFV1"), 25.0, (dw, dh))
    for p in index["path"]:
        vw.write(cv2.resize(cv2.imread(p), (dw, dh), interpolation=interp))
    vw.release()
    return dw, dh


def predict(cfg_path: Path, video: Path, dest: Path) -> pd.DataFrame:
    """Run the donor over ``video``; returns the per-frame prediction table (bodypart, coord)."""
    import deeplabcut

    d = donor()
    deeplabcut.analyze_videos(str(cfg_path), [str(video)], videotype="avi",
                              shuffle=int(d["shuffle"]),
                              trainingsetindex=int(d["trainingsetindex"]),
                              save_as_csv=False, destfolder=str(dest), batch_size=8, gputouse=0)
    h5 = sorted(dest.glob(video.stem + "*.h5"))
    if not h5:
        raise RuntimeError(f"analyze_videos produced no .h5 for {video}")
    out = pd.read_hdf(h5[-1])
    out.columns = out.columns.droplevel(0)          # drop the scorer level
    return out.reset_index(drop=True)


def to_labels(pred: pd.DataFrame, scale_x: float, scale_y: float) -> pd.DataFrame:
    """Donor predictions -> this project's bodyparts, in ORIGINAL pixels, blanked below threshold.

    Returns one column pair per entry of ``dlc.bodyparts``, NaN where nothing is written -- NaN is
    what the labelling GUI reads as "unplaced", and every withheld cell is deliberate.
    """
    pc = prelabel_cfg()
    thresh = float(pc.get("min_likelihood", 0.6))
    never = set(pc.get("never_prelabel") or [])
    spout_from = list(pc.get("spout_from") or [])
    n = len(pred)
    out = {}
    for bp in bodyparts():
        x = np.full(n, np.nan)
        y = np.full(n, np.nan)
        if bp in never:
            out[bp] = (x, y)
            continue
        if bp == "spout" and spout_from:
            # Whichever donor spout fires harder: which one responds depends on where the moving
            # spout is, and the max-of-two demonstrably tracks the actual position.
            liks = np.stack([pred[(s, "likelihood")].to_numpy() for s in spout_from])
            pick = liks.argmax(axis=0)
            lk = liks.max(axis=0)
            px = np.stack([pred[(s, "x")].to_numpy() for s in spout_from])
            py = np.stack([pred[(s, "y")].to_numpy() for s in spout_from])
            rows = np.arange(n)
            cx, cy = px[pick, rows], py[pick, rows]
        elif (bp, "likelihood") in pred.columns:
            lk = pred[(bp, "likelihood")].to_numpy()
            cx, cy = pred[(bp, "x")].to_numpy(), pred[(bp, "y")].to_numpy()
        else:
            out[bp] = (x, y)
            continue
        ok = lk >= thresh
        x[ok], y[ok] = cx[ok] / scale_x, cy[ok] / scale_y
        out[bp] = (x, y)
    cols = pd.MultiIndex.from_tuples([(SCORER, bp, c) for bp in out for c in ("x", "y")],
                                     names=["scorer", "bodyparts", "coords"])
    data = np.column_stack([v for bp in out for v in out[bp]])
    return pd.DataFrame(data, columns=cols)


def write_labels(labels: pd.DataFrame, index: pd.DataFrame, rv=None, dry: bool = False) -> list[Path]:
    """Split by source folder and write DLC ``CollectedData_<scorer>.{h5,csv}``.

    REFUSES to overwrite an existing CollectedData file. Once a human has corrected labels, a
    re-run that replaced them would silently discard the work this whole module exists to save.
    """
    root = out_root(rv) / "labeled-data"
    todo = []
    for stem, g in index.groupby("video_stem", sort=True):
        dest_h5 = root / stem / f"CollectedData_{SCORER}.h5"
        if dest_h5.exists():
            print(f"[dlc_prelabel] {stem}: CollectedData already exists -> NOT overwritten",
                  flush=True)
            continue
        sub = labels.iloc[g.index].copy()
        sub.index = pd.MultiIndex.from_tuples([("labeled-data", stem, img) for img in g["image"]])
        print(f"[dlc_prelabel] {stem}: {len(g)} frames, "
              f"{int(sub.notna().to_numpy().sum() // 2)} points seeded"
              f"{' (dry-run)' if dry else ''}", flush=True)
        todo.append((dest_h5, sub))

    if dry or not todo:
        return []
    # BEFORE the first write, not at it: DLC's labelling GUI reads the .h5, so a run that wrote
    # CSVs and then died on the first `to_hdf` would leave folders that look labelled and are not.
    # pytables ships with the `dlc` env and not with `locanmf`, which is the likely mistake. Checked
    # here rather than on entry so a run with nothing left to write still succeeds.
    try:
        import tables  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Writing DLC labels needs pytables, which `locanmf` does not have. Run this from the "
            "`dlc` env: conda activate dlc && python -m wfield_local.dlc_prelabel"
        ) from exc

    written = []
    for dest_h5, sub in todo:
        assert_writable(dest_h5.parent)
        sub.to_hdf(dest_h5, key="df_with_missing", mode="w")
        sub.to_csv(dest_h5.with_suffix(".csv"))
        written.append(dest_h5)
    return written


def coverage(labels: pd.DataFrame) -> pd.Series:
    """Fraction of frames that got a point, per bodypart -- what the labeller still has to place."""
    return pd.Series({bp: float(labels[(SCORER, bp, "x")].notna().mean()) for bp in bodyparts()})


def run(cam="cam4", scale=None, rv=None, dry=False, workdir=None):
    rv = rv or PathResolver()
    scale = float(scale if scale is not None else prelabel_cfg().get("scale", 0.45))
    index = frame_index(cam, rv)
    if index.empty:
        print(f"[dlc_prelabel] no extracted {cam} frames -- run dlc_frames first", flush=True)
        return None
    work = Path(workdir or Path(donor()["local_copy"]).parent / "prelabel")
    video = work / f"{cam}_scale{scale:.2f}.avi"
    print(f"[dlc_prelabel] {len(index)} {cam} frames at scale {scale:.2f}", flush=True)
    dw, dh = build_video(index, scale, video)

    import cv2
    first = cv2.imread(index.iloc[0]["path"])
    sx, sy = dw / first.shape[1], dh / first.shape[0]

    pred = predict(ensure_local_donor(), video, work / "out")
    labels = to_labels(pred, sx, sy)
    cov = coverage(labels)
    print("[dlc_prelabel] seeded fraction per bodypart:", flush=True)
    for bp, v in cov.items():
        note = "  <- place by hand" if v == 0 else ""
        print(f"    {bp:16s} {v:5.1%}{note}", flush=True)
    write_labels(labels, index, rv, dry)
    return labels


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cam", default="cam4")
    ap.add_argument("--scale", type=float, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--machine", default=None)
    args = ap.parse_args(argv)
    run(args.cam, args.scale, PathResolver(machine=args.machine), args.dry_run, args.workdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
