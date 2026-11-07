"""Side cameras (cam2 + cam3) -> ONE Lightning Pose project: cam3 flipped left-right, every frame cropped to the face.

    conda activate dlc            # pandas + PIL; no DLC import
    python -m wfield_local.lp_sidecam stage C:/Users/SabatiniLab/lp_stage/lp_side_pilot_20261027
    python -m wfield_local.lp_sidecam unseen C:/Users/SabatiniLab/lp_stage/lp_side_pilot_20261027   # Parts B/C targets
    python -m wfield_local.lp_sidecam audit                  # what is labeled in the side folders, nothing written
    conda activate dlc            # pytables, for DLC's .h5
    python -m wfield_local.lp_sidecam prelabel <preds.csv> [--dry-run]   # seed the unlabelled side folders

PRE-LABELS (`prelabel`, Priya 2026-11-04: crop2 pre-labels the round-2 side targets, Parts B / C + the jaw sequences).
``<preds.csv>`` = `litpose predict <crop2 model> <dest>/<list>.csv` on the frames `unseen` staged (model space). Each
point with likelihood >= `PRELABEL_CUT` (tongue 0.4, others 0.6, as the review clips) goes back to the RAW frame
(`to_raw`, the frame's own width); below the cut it stays BLANK for the labeller to place or leave hidden. Written as
DLC ``CollectedData_Priya`` through `dlc_prelabel.write_labels` (staging) + `dlc_project.sync_frames`, so a folder
that already holds labels is NEVER touched -- the seeds are a starting point to correct, not labels.

THE PLAN (DECISIONS 2026-10-13 night, `dlc_sidecam`): one shared model for both side views, cam3 flipped at
TRAINING and INFERENCE time; labeling is on the raw images. The four parts are midline / single structures, so a
flip needs no left / right label swap.

THE TRANSFORM (`to_model` / `to_raw`), applied identically to labels, images and later to videos:
  1. cam3 only: x -> W - 1 - x, W = that frame's own width (pixel centers; PIL FLIP_LEFT_RIGHT does the same).
  2. crop the square CROPS[W] = (x0, y0, side) of the (flipped) frame and resize it to OUT x OUT (256; a multiple of
     128, LP's check). Model coordinates = (x - x0 + 0.5) * OUT / side - 0.5 (PIL's pixel-center resize mapping).

ONE CROP PER FRAME SIZE (Priya 2026-10-27: "prioritize the august crop, or design different crops"). The JUNE side
videos are 800 x 600, August onward 600 x 450, and the face is 1.24x larger in June (June / August median nose-spout
and nose-jaw distance per animal 1.16-1.29 on both measures, all four animals -- a camera set-up change, not an animal
difference). So:
  * AUGUST (W 600) = the primary data (the August-only pre baseline is primary, DECISIONS 2026-10-18): 256 x 256 at
    NATIVE resolution, (x0, y0) = (0, 50), set from the August points only (Part A labels + the pilot's confident
    predictions on 13 unseen sessions, cam3 flipped: x 52-185, y 91-275) -> margins 52 left (the frame edge), 71
    right, 41 top, 41 bottom.
  * JUNE (W 800): a 317-px square (= 256 x 1.24) resized to 256, so the June face reaches the network at the AUGUST
    scale; y0 = 62 puts it at August's height (fit on the per-part medians); x0 = 0 rather than the fitted -39 (no
    padding: a shift is cheap for a CNN, scale is not). June points land at x 33-148, y 54-210 of the 256 box.
The first pilot (one 384 box for both sizes, no rescaling) is superseded by this one (DECISIONS 2026-10-27).

WHICH ROWS. All-blank rows never reach training (`dlc_train.drop_unlabelled`; under LP's uniform-heatmap rule a
blank row teaches "everything hidden"). Rows with ONE part only are dropped too: the nose is visible on every listed
Part A frame (143 / 143), so a lone part is far more likely a half-finished frame than a real three-part occlusion,
and a partial frame's blanks would be read as hidden (the guide's "never label only one part"). CONTEXT frames that
were labeled completely are kept (Priya 2026-10-27; allowed since 10-13: "context stays blank, or is labeled
completely").
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PARTS = ["nose", "jaw", "tongue", "spout"]
SIDE = ("cam2", "cam3")
SCORER = "Priya"
#: frame width -> (x0, y0, side) of the square crop in the (cam3-flipped) frame; resized to OUT -- module docstring
CROPS = {600: (0, 50, 256), 800: (0, 62, 317)}
OUT = 256
MIN_PARTS = 2
PRELABEL_CUT = {"nose": 0.6, "jaw": 0.6, "tongue": 0.4, "spout": 0.6}


def labelling_project() -> Path:
    from wfield_local import dlc_project
    return dlc_project.project_dir()


def crop_for(width: int) -> tuple[int, int, int]:
    if width not in CROPS:
        raise SystemExit(f"no side-camera crop for frame width {width} (known: {sorted(CROPS)})")
    return CROPS[width]


def to_model(x, y, cam: str, width: int):
    """Raw image coordinates -> model (flipped, cropped, resized) coordinates. Scalars or arrays."""
    x0, y0, side = crop_for(width)
    if cam == "cam3":
        x = width - 1 - x
    f = OUT / side
    return (x - x0 + 0.5) * f - 0.5, (y - y0 + 0.5) * f - 0.5


def to_raw(x, y, cam: str, width: int):
    """Inverse of `to_model`: model predictions back onto the raw video frame."""
    x0, y0, side = crop_for(width)
    f = side / OUT
    x, y = (x + 0.5) * f - 0.5 + x0, (y + 0.5) * f - 0.5 + y0
    if cam == "cam3":
        x = width - 1 - x
    return x, y


def transform_image(im, cam: str):
    from PIL import Image
    if cam == "cam3":
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    x0, y0, side = crop_for(im.size[0])
    im = im.crop((x0, y0, x0 + side, y0 + side))
    return im if side == OUT else im.resize((OUT, OUT), Image.BILINEAR)


def read_labels(folder: Path) -> pd.DataFrame:
    """One row per image: <part>_x / <part>_y columns, index = image name."""
    df = pd.read_csv(folder / f"CollectedData_{SCORER}.csv", header=[0, 1, 2], index_col=[0, 1, 2])
    out = pd.DataFrame(index=df.index.get_level_values(2))
    for p in PARTS:
        for c in ("x", "y"):
            out[f"{p}_{c}"] = df[(SCORER, p, c)].to_numpy()
    return out


def keep_rows(lab: pd.DataFrame, min_parts: int = MIN_PARTS) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = lab[[f"{p}_x" for p in PARTS]].notna().sum(axis=1)
    return lab[n >= min_parts], lab[(n > 0) & (n < min_parts)]


def side_folders(proj: Path, only: list[str] | None = None) -> list[Path]:
    fs = sorted(p for p in (proj / "labeled-data").iterdir()
                if p.is_dir() and p.name.startswith(SIDE) and (p / f"CollectedData_{SCORER}.csv").is_file())
    return [f for f in fs if not only or f.name in only]


def stage(dest: Path, proj: Path | None = None, only: list[str] | None = None) -> pd.DataFrame:
    """Write ``dest/labeled-data/<folder>/<img>.png`` (transformed) + ``dest/CollectedData.csv``. Returns a per-folder
    summary. Refuses if a kept point falls outside the crop (the box would be wrong for that session)."""
    from PIL import Image
    proj = proj or labelling_project()
    rows, summary = [], []
    for f in side_folders(proj, only):
        cam = f.name[:4]
        lab, partial = keep_rows(read_labels(f))
        if lab.empty:
            continue
        (dest / "labeled-data" / f.name).mkdir(parents=True, exist_ok=True)
        widths = set()
        for img, r in lab.iterrows():
            im = Image.open(f / img)
            widths.add(im.size)
            transform_image(im, cam).save(dest / "labeled-data" / f.name / img)
            rec = {"path": f"labeled-data/{f.name}/{img}"}
            for p in PARTS:
                x, y = to_model(r[f"{p}_x"], r[f"{p}_y"], cam, im.size[0])
                if x == x and not (0 <= x < OUT and 0 <= y < OUT):
                    raise SystemExit(f"{f.name}/{img}: {p} at model ({x:.0f}, {y:.0f}) is outside the "
                                     f"{crop_for(im.size[0])} crop")
                rec[f"{p}_x"], rec[f"{p}_y"] = x, y
            rows.append(rec)
        summary.append({"folder": f.name, "kept": len(lab), "dropped_partial": len(partial),
                        "sizes": ",".join(f"{w}x{h}" for w, h in sorted(widths)),
                        **{p: int(lab[f"{p}_x"].notna().sum()) for p in PARTS}})
    if not rows:
        raise SystemExit("no labeled side-camera frames found")
    out = pd.DataFrame(rows).set_index("path")
    cols = pd.MultiIndex.from_tuples([(SCORER, p, c) for p in PARTS for c in ("x", "y")],
                                     names=["scorer", "bodyparts", "coords"])
    # UNNAMED index: a named one makes pandas write an extra "path,,,," row under the three header rows, which LP
    # then reads as a frame called "path" (first pilot run, 2026-10-27: FileNotFoundError in the dataloader)
    out = pd.DataFrame(out[[f"{p}_{c}" for p in PARTS for c in ("x", "y")]].to_numpy(), index=list(out.index),
                       columns=cols)
    out.to_csv(dest / "CollectedData.csv")
    s = pd.DataFrame(summary)
    s.to_csv(dest / "staging_summary.csv", index=False)
    return s


def stage_unseen(dest: Path, exclude: set[str], proj: Path | None = None) -> int:
    """The listed TARGET frames (`side_assignment.csv`, label == True) of every side folder NOT in ``exclude``,
    transformed like the training frames, plus a blank-label ``dest/unseen.csv`` for `litpose predict`. These are
    sessions the pilot never saw: the out-of-session check, and later pre-labels for SIDECAM Parts B / C."""
    from PIL import Image
    from wfield_local.dlc_sidecam import ASSIGN_NAME
    proj = proj or labelling_project()
    a = pd.read_csv(proj.parent / ASSIGN_NAME)
    listed = a.label & ~(a.deferred if "deferred" in a else False)
    a = a[listed & ~a.video_stem.isin(exclude)]
    paths = []
    for (stem, cam), g in a.groupby(["video_stem", "cam"]):
        (dest / "unseen" / stem).mkdir(parents=True, exist_ok=True)
        for img in sorted(g.image):
            transform_image(Image.open(proj / "labeled-data" / stem / img), cam).save(dest / "unseen" / stem / img)
            paths.append(f"unseen/{stem}/{img}")
    cols = pd.MultiIndex.from_tuples([(SCORER, p, c) for p in PARTS for c in ("x", "y")],
                                     names=["scorer", "bodyparts", "coords"])
    pd.DataFrame(float("nan"), index=pd.Index(paths), columns=cols).to_csv(dest / "unseen.csv")
    return len(paths)


def prelabel(preds_csv: Path, rv=None, dry: bool = False, replace_unchecked: bool = False) -> int:
    """Seed the side folders listed in ``preds_csv`` (rows ``unseen/<stem>/<img>``) with confident predictions, for
    the LISTED targets only (`side_assignment.csv`: label and not deferred). ``replace_unchecked``: a folder whose
    live CollectedData is still EXACTLY its earlier seed file (nobody has edited it) is re-seeded; any other labeled
    folder is left alone."""
    from PIL import Image

    from wfield_local import dlc_prelabel as DP
    from wfield_local import dlc_project
    from wfield_local.dlc_context_frames import untouched_folders
    from wfield_local.paths import PathResolver
    rv = rv or PathResolver()
    pr = pd.read_csv(preds_csv, header=[0, 1, 2], index_col=0)
    pr.columns = pr.columns.droplevel(0)
    proj = labelling_project()
    from wfield_local.dlc_frames import out_root, staging_root
    from wfield_local.dlc_sidecam import ASSIGN_NAME
    a = pd.read_csv(out_root(rv) / ASSIGN_NAME)
    listed = set(zip(a.video_stem, a.image)) if "deferred" not in a else \
        set(zip(a.video_stem[a.label & ~a.deferred], a.image[a.label & ~a.deferred]))
    free = set(untouched_folders("cam2", rv)) | set(untouched_folders("cam3", rv))
    if replace_unchecked:
        from wfield_local.dlc_sidecam_guide import unchecked_seeds
        stale = []
        for stem in sorted({str(p).replace("\\", "/").split("/")[1] for p in pr.index} - free):
            live = proj / "labeled-data" / stem / f"CollectedData_{SCORER}.csv"
            seed = staging_root(rv) / stem / f"CollectedData_{SCORER}.csv"
            if not (live.is_file() and seed.is_file()):
                continue
            rows_live = pd.read_csv(live, header=[0, 1, 2], index_col=[0, 1, 2]).index
            if unchecked_seeds(stem, proj, rv, PARTS) >= {ix[2] for ix in rows_live}:
                stale.append(stem)
            else:
                print(f"{stem}: live labels differ from the seeds (edited) -> left alone")
        for stem in stale:
            print(f"{stem}: unedited seeds -> re-seeded{' (dry-run)' if dry else ''}")
            if not dry:
                for d in (proj / "labeled-data" / stem, staging_root(rv) / stem):
                    for ext in ("h5", "csv"):
                        (d / f"CollectedData_{SCORER}.{ext}").unlink(missing_ok=True)
        free |= set(stale)
    rows, idx = [], []
    for path in pr.index:
        _, stem, img = str(path).replace("\\", "/").split("/")
        if stem not in free or (stem, img) not in listed:
            continue
        cam = stem.split("_", 1)[0]
        with Image.open(proj / "labeled-data" / stem / img) as im:
            width = im.size[0]
        rec = {}
        for part in PARTS:
            x, y, lk = (float(pr.loc[path, (part, c)]) for c in ("x", "y", "likelihood"))
            if lk >= PRELABEL_CUT[part]:
                rec[(SCORER, part, "x")], rec[(SCORER, part, "y")] = to_raw(x, y, cam, width)
        rows.append(rec)
        idx.append({"video_stem": stem, "image": img})
    cols = pd.MultiIndex.from_tuples([(SCORER, p, c) for p in PARTS for c in ("x", "y")],
                                     names=["scorer", "bodyparts", "coords"])
    labels = pd.DataFrame(rows, columns=cols, dtype=float)
    index = pd.DataFrame(idx)
    print(f"{len(index)} frames in {index.video_stem.nunique()} unlabelled folders; seeded points per part: "
          + ", ".join(f"{p} {int(labels[(SCORER, p, 'x')].notna().sum())}" for p in PARTS))
    written = DP.write_labels(labels, index, rv, dry=dry)
    if written:
        folders, images, n = dlc_project.sync_frames(dlc_project.project_dir(rv), rv, cams=list(SIDE))
        print(f"wrote {len(written)} seed files; synced {n} into the labeling project")
    return len(written)


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["stage", "unseen", "audit", "prelabel"])
    ap.add_argument("dest", nargs="?", type=Path)
    ap.add_argument("--only", nargs="+", default=None, help="folder names (default: every labeled side folder)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--replace-unchecked", action="store_true",
                    help="prelabel: re-seed folders whose labels are still exactly the earlier seeds")
    a = ap.parse_args(argv)
    if a.cmd == "prelabel":
        if a.dest is None:
            ap.error("prelabel needs the predictions csv")
        prelabel(a.dest, dry=a.dry_run, replace_unchecked=a.replace_unchecked)
        return 0
    proj = labelling_project()
    if a.cmd == "audit":
        for f in side_folders(proj, a.only):
            lab, partial = keep_rows(read_labels(f))
            if len(lab) or len(partial):
                print(f"{f.name}: {len(lab)} kept, {len(partial)} partial (dropped): {list(partial.index)}")
        return 0
    if a.dest is None:
        ap.error(f"{a.cmd} needs a destination directory")
    if a.cmd == "unseen":            # every side folder the project at ``dest`` was NOT trained on
        trained = set(pd.read_csv(a.dest / "staging_summary.csv").folder)
        print(f"{stage_unseen(a.dest, trained, proj)} unseen target frames -> {a.dest / 'unseen.csv'}")
        return 0
    s = stage(a.dest, proj, a.only)
    print(s.to_string(index=False))
    print(f"total {s.kept.sum()} frames ({s.dropped_partial.sum()} partial dropped) -> {a.dest / 'CollectedData.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
