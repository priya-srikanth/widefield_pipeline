"""Assemble the DLC project so labelling is one command away.

`dlc_frames` extracts frames and `dlc_prelabel` seeds them, but both write into a plain directory on
the share; DeepLabCut needs a PROJECT around that -- a `config.yaml` naming the bodyparts, and
`labeled-data/<video-stem>/` folders it recognises. This builds and refreshes it, idempotently, so
the answer to "what do I run to label?" is two commands rather than a recipe.

**The project's `bodyparts` is the UNION across views; each view still only gets what it can see.**
DLC has one bodypart list per project, so the union is the only thing that can go in it — but a
label placed for a part a camera cannot see is invented data, and a network trained on invented
points learns to hallucinate. `dlc_frames.bodyparts(cam)` is the per-view truth and this module
prints it, because the GUI cannot enforce it.

**Labels are copied in, never linked, and never overwritten.** A `CollectedData_*.h5` in the project
is the only record of manual work; a refresh that clobbered one would destroy exactly what this
pipeline exists to save. New frames are added, existing labels are left alone.

**`project_path` is rewritten for the machine it is opened on.** DLC stores an absolute path in
`config.yaml`, and MICROSCOPE is `N:` on the imaging box and `M:` on the analysis box, so a project
created on one and opened on the other points at nothing.

CLI (from the ``dlc`` env)::

    python -m wfield_local.dlc_project --create        # build it, or refresh in place
    python -m wfield_local.dlc_project                 # refresh + repoint, no creation
    python -m wfield_local.dlc_project --label         # ... and open the labelling GUI
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from wfield_local import config
from wfield_local.dlc_frames import bodyparts, cameras, out_root, role
from wfield_local.paths import PathResolver
from wfield_local.writeguard import assert_writable

TASK = "widefield"
SCORER = "Priya"


def project_dir(rv=None) -> Path:
    """``<behavior_cameras>/dlc/<Task>-<scorer>-<date>`` -- beside the frames it labels.

    On the SHARE rather than a local disk: the labels are the expensive artefact in this pipeline,
    they are produced interactively on whichever box someone is sitting at, and a project on one
    box's C: drive is invisible to the other and to any backup.
    """
    root = out_root(rv)
    existing = sorted(p for p in root.glob(f"{TASK}-{SCORER}-*") if p.is_dir())
    if existing:
        return existing[-1]
    date = (config.defaults().get("dlc") or {}).get("project_date", "2026-09-08")
    return root / f"{TASK}-{SCORER}-{date}"


def frame_dirs(rv=None, cams=None) -> list[Path]:
    """Extracted-frame folders, one per source video, for the requested cameras."""
    cams = tuple(cams or cameras())
    src = out_root(rv) / "labeled-data"
    return sorted(p for p in src.iterdir()
                  if p.is_dir() and p.name.startswith(cams) and any(p.glob("img*.png")))


def write_config(proj: Path, rv=None) -> Path:
    """Create or refresh ``config.yaml``: bodyparts = the union, project_path = this machine."""
    import yaml

    cfg_path = proj / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    dlc_cfg = config.defaults().get("dlc") or {}
    cfg.update({
        "Task": TASK, "scorer": SCORER, "multianimalproject": False, "identity": None,
        "project_path": str(proj).replace("\\", "/"),
        "bodyparts": bodyparts(),                      # the UNION; per-view subsets are enforced
        "skeleton": [], "skeleton_color": "black",     # by the labeller, see print_plan()
        "pcutoff": 0.6, "dotsize": 6, "alphavalue": 0.7, "colormap": "rainbow",
        "TrainingFraction": [0.8], "iteration": cfg.get("iteration", 0),
        "default_net_type": "resnet_50", "default_augmenter": "default", "engine": "pytorch",
        "snapshotindex": -1, "detector_snapshotindex": -1,
        "batch_size": 8, "detector_batch_size": 8,
        "cropping": False, "start": 0, "stop": 1,
        "numframes2pick": int((dlc_cfg.get("frames") or {}).get("per_session", 24)),
        "move2corner": True, "corner2move2": [50, 50],
        # Frames come from dlc_frames, never from DLC's own appearance clustering, so there is
        # nothing for video_sets to do -- and pointing it at 12 GB recordings on the share invites
        # the GUI to open one.
        "video_sets": cfg.get("video_sets") or {},
    })
    assert_writable(proj)
    proj.mkdir(parents=True, exist_ok=True)
    for sub in ("labeled-data", "training-datasets", "dlc-models-pytorch", "videos"):
        (proj / sub).mkdir(exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return cfg_path


def sync_frames(proj: Path, rv=None, cams=None) -> tuple[int, int, int]:
    """Copy frames and any seeded labels into the project. Returns (folders, images, labels).

    NEVER overwrites a CollectedData file that is already in the project: once a human has corrected
    labels, the copy on the share is the stale one, and refreshing must not run backwards.
    """
    dest_root = proj / "labeled-data"
    assert_writable(dest_root)
    folders = images = labels = 0
    for src in frame_dirs(rv, cams):
        dest = dest_root / src.name
        dest.mkdir(parents=True, exist_ok=True)
        folders += 1
        for img in sorted(src.glob("img*.png")):
            if not (dest / img.name).exists():
                shutil.copy2(img, dest / img.name)
                images += 1
        if (src / f"CollectedData_{SCORER}.h5").exists() \
                and not (dest / f"CollectedData_{SCORER}.h5").exists():
            _copy_labels_padded(src, dest)
            labels += 1
    return folders, images, labels


def _copy_labels_padded(src: Path, dest: Path) -> None:
    """Copy seeded labels, padding to the project's FULL bodypart list with NaN.

    `dlc_prelabel` writes only the bodyparts a camera can see -- ten on cam4 -- but a DLC project
    has ONE bodypart list and its readers expect every column to be present. A short table is a
    shape mismatch waiting to happen somewhere in DLC rather than a documented subset. NaN is the
    same statement in the format DLC understands: unplaced.

    The padded columns are the ones the labeller is told NOT to place (`print_plan`); they exist so
    the file is well-formed, not as an invitation.
    """
    import numpy as np
    import pandas as pd

    d = pd.read_hdf(src / f"CollectedData_{SCORER}.h5")
    have = {c[1] for c in d.columns}
    for bp in bodyparts():
        if bp not in have:
            for coord in ("x", "y"):
                d[(SCORER, bp, coord)] = np.nan
    d = d.reindex(columns=pd.MultiIndex.from_tuples(
        [(SCORER, bp, c) for bp in bodyparts() for c in ("x", "y")],
        names=["scorer", "bodyparts", "coords"]))
    d.to_hdf(dest / f"CollectedData_{SCORER}.h5", key="df_with_missing", mode="w")
    d.to_csv(dest / f"CollectedData_{SCORER}.csv")


def print_plan(proj: Path, rv=None, cams=None) -> None:
    """What to place in each folder -- the per-view subset the GUI cannot enforce."""
    dirs = frame_dirs(rv, cams)
    by_cam: dict[str, int] = {}
    for p in dirs:
        by_cam[p.name.split("_")[0]] = by_cam.get(p.name.split("_")[0], 0) + 1
    print(f"\n[dlc_project] {proj}", flush=True)
    print(f"[dlc_project] project bodyparts (the union DLC needs): {', '.join(bodyparts())}",
          flush=True)
    print("[dlc_project] PLACE ONLY WHAT EACH VIEW CAN SEE:", flush=True)
    for cam in sorted(by_cam):
        print(f"    {cam} ({role(cam) or '?'}, {by_cam[cam]} folders): "
              f"{', '.join(bodyparts(cam))}", flush=True)
    print("    a point placed for a part a camera cannot see is invented data, and a network "
          "trained on it learns to hallucinate", flush=True)


def run(create=False, cams=None, rv=None, label=False) -> Path:
    rv = rv or PathResolver()
    proj = project_dir(rv)
    if not (proj / "config.yaml").exists() and not create:
        raise SystemExit(f"No project at {proj}. Run with --create first.")
    cfg = write_config(proj, rv)
    folders, images, labels = sync_frames(proj, rv, cams)
    print(f"[dlc_project] {folders} folders, +{images} images, +{labels} label files -> {cfg}",
          flush=True)
    print_plan(proj, rv, cams)
    if label:
        open_gui(cfg)
    return proj


def open_gui(cfg: Path) -> None:
    """Open DLC's labelling GUI, or explain what to install.

    ``pip install deeplabcut[pytorch]`` gets inference and training but NOT the Qt GUI, and the
    resulting failure is an ``AttributeError`` on ``label_frames`` several frames below a
    ``ModuleNotFoundError: qtpy`` -- which reads as a broken project rather than a missing extra.
    """
    import deeplabcut

    print("\n[dlc_project] opening the labelling GUI ...", flush=True)
    try:
        deeplabcut.label_frames(str(cfg))
    except (AttributeError, ImportError) as exc:
        raise SystemExit(
            f"DeepLabCut has no GUI in this environment ({exc}).\n"
            f"  conda activate dlc && pip install \"deeplabcut[gui]\"\n"
            f"Then re-run. The project itself is built and unaffected:\n  {cfg}"
        ) from exc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--create", action="store_true", help="create the project if it does not exist")
    ap.add_argument("--cam", action="append", default=None,
                    help="restrict to these cameras (repeatable; default: all with frames)")
    ap.add_argument("--label", action="store_true", help="open the DLC labelling GUI afterwards")
    ap.add_argument("--machine", default=None)
    args = ap.parse_args(argv)
    run(args.create, args.cam, PathResolver(machine=args.machine), args.label)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
