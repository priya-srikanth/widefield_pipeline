"""Retrain the cam4 orofacial network on the REFINED labels only — DLC's refine-and-retrain loop.

DeepLabCut's own refinement cycle is

    label -> create_training_dataset -> train_network -> evaluate_network -> analyze_videos
          -> extract_outlier_frames -> refine_labels -> merge_datasets (bumps ``iteration``) -> again

and this module is that cycle's TRAIN end. Every step is DeepLabCut's own API —
``create_training_dataset``, ``train_network``, ``evaluate_network``, ``WeightInitialization``,
and ``iteration`` as the round counter. Nothing here reimplements a step DLC already has.

**WHAT HAS BEEN REFINED IS NOT WHAT IS LABELLED.** `dlc_prelabel` seeds ten bodyparts on cam4 from
the 2pRAM donor, and a seeded point is a *prediction*, not a label. As of 2026-09-21 the refinement
pass covered `nose`, `jaw`, `tongue` and `spout` across all 15 cam4 folders; the six whisker columns
are still raw donor output. Measured on the one folder that kept a pre-refinement backup
(`cam4_2026-06-06T12_25_18/…h5.bak-20260916-161730`):

    nose    moved on 16/16 frames, median 3.42 px      L_whiskers_1..3  1/16 frames, median 0.00 px
    spout   moved on 14/16 frames, median 0.91 px      R_whiskers_1..3  1/16 frames, median 0.00 px
    jaw     moved on  8/12 frames, median 0.66 px
    tongue  moved on  4/4  frames, median 3.37 px

So training on the union would fit six channels to the donor's own guesses about a view 2.5x more
zoomed than the one it learned — the hallucination `dlc_frames.bodyparts` exists to prevent, arriving
by a different door. `dlc.train.bodyparts` is the list of parts a HUMAN has actually placed, and it
is the contract: widen it when the labelling widens, not before.

**THIS TRAINS IN A SEPARATE PROJECT, AND THE COPY DIRECTION IS THE OPPOSITE OF `dlc_project`'s.**
The labelling project is the single source of truth for labels; this one is derived and disposable,
and every run re-syncs labels FORWARD from it and overwrites what is here. `dlc_project` refuses to
overwrite a `CollectedData` file because there the human edit is downstream; here it is upstream, and
refusing would train on stale labels. Do not label in the training project.

A separate project rather than an edit in place, for three reasons that are all about the OTHER
views:

  * `config.yaml` carries ONE bodypart list for the whole project. Cutting the live one to four
    parts would make `cam2`/`cam3` — 8 parts each, including the eyes, and not yet labelled at all —
    impossible to label.
  * `dlc_project.write_config` rewrites `bodyparts` to the union on EVERY run, so an in-place edit
    is silently reverted the next time anyone opens the GUI, and a training set rebuilt after that
    would quietly pull the whisker seeds back in without erroring.
  * ``iteration`` is project-wide. Bumping it in the live project to version a cam4 refinement round
    would also re-version the other views' work -- `cam2`/`cam3` have no labels yet, and `cam1` has
    one partly-labelled session (`cam1_2026-06-06T12_25_18`, 2026-09-10: jaw 71/72, tongue 38/72,
    hand-placed since `dlc_prelabel` seeds cam4 only).

**THE DONOR SUPPLIES WEIGHTS, NOT AUGMENTATION.** `dlc_orofacial.md` step 3 says to add scale
augmentation because the donor trained with ``affine.scaling: [1.0, 1.0]`` and
``ResizeFromDataSizeCollate(min_scale=0.4, max_scale=1.0)`` — its own scale and smaller, never
larger. That is true of the donor's config and is not something this module has to fix: DLC 3.0.1
builds a FRESH ``pytorch_config.yaml`` from its own templates, whose default
(``config/base/aug_default.yaml``) is already ``affine.scaling: [0.5, 1.25]`` — two-sided — with
448x448 ``crop_sampling`` rather than the donor's collate. Only the `snapshot_path` weights come
across. `dlc.train.scaling` is the knob if you want to reach further down toward the donor's
apparent size; the default is DLC's own, on purpose.

**THE SPLIT HOLDS OUT WHOLE SESSIONS, NOT RANDOM FRAMES.** DLC's default split is uniform over
frames, and on THIS labelling set that measures the wrong thing. `dlc.frames.lick_offsets_s` samples
four points of a single ~80 ms protrusion, and `dlc_frames`'s own pruning measurement found 99–100%
of within-onset frame pairs closer in appearance than the 5th percentile of between-onset pairs. A
uniform split therefore puts near-duplicate frames on both sides of it and reports a test RMSE that
is optimistic about the only thing the number is for — a session the network has not seen. Sessions
are held out whole, one per epoch in a seeded order, so `test rmse` is a generalisation estimate and
`--per-epoch` can ask the question the runbook actually cares about:

    "a network that tracks a healthy mouse well and a hemiparetic one badly reads as a deficit
     and is not one"

which is a per-EPOCH error table, not a scalar.

RUN THIS FROM THE ``dlc`` ENV (DeepLabCut is installed only there; `pip install -e . --no-deps`
makes this repo importable from it).

CLI::

    conda activate dlc
    python -m wfield_local.dlc_train --dry-run     # stage + audit + print the split; train nothing
    python -m wfield_local.dlc_train               # stage, create the training set, train, evaluate
    python -m wfield_local.dlc_train --evaluate    # re-evaluate the newest snapshot, no training
    python -m wfield_local.dlc_train --evaluate --review   # what to relabel, + annotated crops
    python -m wfield_local.dlc_train --evaluate --guide    # ... and rebuild the labeller's guide
    python -m wfield_local.dlc_train --iteration 1 # next refinement round (see `next_round`)
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from wfield_local import config, dlc_project
from wfield_local.dlc_frames import staging_root
from wfield_local.paths import PathResolver
from wfield_local.writeguard import assert_writable

SCORER = dlc_project.SCORER
TASK = dlc_project.TASK


# --------------------------------------------------------------------------------------- config

def _cfg() -> dict:
    return (config.defaults().get("dlc") or {})


def train_cfg() -> dict:
    return _cfg().get("train") or {}


def parts() -> list[str]:
    """The bodyparts a HUMAN has placed — `dlc.train.bodyparts`, and nothing inferred.

    Deliberately not derived from the label tables. "Has a non-NaN value" cannot distinguish a
    refined label from a donor seed (both are just coordinates), so any automatic rule here would
    silently train on seeds the moment `dlc_prelabel` filled a new column. A hand-maintained list
    is the only honest form this can take; `audit()` prints what is actually in the files beside it
    so a disagreement is visible rather than assumed away.
    """
    bps = [str(b) for b in (train_cfg().get("bodyparts") or [])]
    if not bps:
        raise SystemExit("dlc.train.bodyparts is empty — nothing has been declared as refined.")
    return bps


def cam() -> str:
    return str(train_cfg().get("cam", "cam4"))


def train_project(rv=None) -> Path:
    """``<dlc>/training/<labelling project>-<tag>`` — derived, and OUT OF THE LABELLING GLOB.

    The subdirectory is not tidiness. `dlc_project.project_dir` globs
    ``<Task>-<scorer>-*`` in the dlc root and returns the LAST match, so a sibling named
    ``widefield-Priya-2026-09-08-orofacial`` sorts after the real project and becomes the labelling
    target — `dlc_project --label` then opens the GUI on this disposable four-bodypart copy, and any
    label placed there is destroyed by the next `stage()`. That happened on 2026-09-21, on the first
    real run. `project_dir` now also matches the date shape rather than `*`; both guards stay,
    because either one alone is a single edit away from being undone by someone who does not know
    about the other.
    """
    src = dlc_project.project_dir(rv)
    return src.parent / "training" / f"{src.name}-{train_cfg().get('tag', 'orofacial')}"


# ------------------------------------------------------------------------------- donor weights

def donor_bodyparts() -> list[str]:
    """The donor's bodyparts IN HEAD-CHANNEL ORDER, read from its ``pytorch_config.yaml``.

    From the model config's ``metadata.bodyparts``, not the project's ``config.yaml``: a
    conversion array indexes the trained HEAD, and the model config is the file that records what
    that head's channels are. (The donor's `config.yaml` is also unreliable to parse — its
    bodyparts list has comment blocks interleaved through it.)
    """
    import yaml

    from wfield_local.dlc_prelabel import donor

    d = donor()
    mp = (Path(d["project"]) / "dlc-models-pytorch/iteration-5"
          / f"video2Jan26-trainset80shuffle{d['shuffle']}/train/pytorch_config.yaml")
    meta = (yaml.safe_load(mp.read_text(encoding="utf-8")) or {}).get("metadata") or {}
    return [str(b) for b in (meta.get("bodyparts") or [])]


def donor_snapshot() -> Path:
    """The donor snapshot to initialise from; prefers the local copy, falls back to the share.

    Read-only either way — DLC ``torch.load``s this path and writes only into the NEW project — so
    reading it off MICROSCOPE breaks no ground rule. The local copy is preferred because it is
    faster and because `dlc.donor.local_copy` is per-machine (`E:` exists on the imaging box and
    not on this one), so requiring it would make training machine-dependent for no gain.
    """
    from wfield_local.dlc_prelabel import donor

    d = donor()
    rel = (f"dlc-models-pytorch/iteration-5/video2Jan26-trainset80shuffle{d['shuffle']}"
           f"/train/{d['snapshot']}")
    local = Path(d.get("local_copy") or "") / rel
    if local.is_file():
        return local
    shared = Path(d["project"]) / rel
    if not shared.is_file():
        raise SystemExit(f"Donor snapshot not found:\n  {local}\n  {shared}")
    return shared


def conversion(bodyparts: list[str] | None = None) -> tuple[list[int], list[str]] | None:
    """``(conversion_array, bodyparts)`` mapping THIS project's parts onto donor head channels.

    ``None`` when any requested part has no donor counterpart, because DLC requires one entry per
    project bodypart when ``with_decoder=True`` — a partial mapping is not a thing it accepts, so
    the caller falls back to a re-initialised head rather than guessing a channel.

    The mapping itself is `dlc.donor.conversion` (target -> donor name), restricted to the parts
    being trained. On the four refined parts that is ``nose->nose, jaw->jaw, tongue->tongue,
    spout->R_spout`` = ``[0, 1, 2, 12]``, and dropping the whiskers makes this a BETTER transfer
    than the ten-part version would have been: every channel that survives is one whose supervision
    is a human label, rather than six more initialised from the donor to be fitted back to the
    donor's own guesses.
    """
    bps = list(bodyparts or parts())
    name_of = {str(k): str(v) for k, v in ((_cfg().get("donor") or {}).get("conversion") or {}).items()}
    order = donor_bodyparts()
    arr: list[int] = []
    for bp in bps:
        donor_name = name_of.get(bp)
        if donor_name is None or donor_name not in order:
            return None
        arr.append(order.index(donor_name))
    return arr, bps


def weight_init(bodyparts: list[str] | None = None):
    """DLC's ``WeightInitialization`` for the donor snapshot, carrying the conversion array.

    **``with_decoder`` is False HERE and flipped on at train time** (`pytorch_updates`), which looks
    like a workaround and is the only correct way to do this in DLC 3.0.1. Two different pieces of
    DLC read this object:

      * `create_training_dataset` branches on ``with_decoder`` alone, and if it is True routes to
        ``make_super_animal_finetune_config`` -> ``PoseConfig.build_for_superanimal_finetune``, which
        raises ```WeightInitialization.dataset` is required for fine-tuning SuperAnimal models`` and
        would build the model config from SUPERANIMAL templates. Our donor is a custom project, not
        a SuperAnimal model, and we want DLC's ordinary resnet_50 + `aug_default` templates — the
        two-sided scaling in the module docstring is exactly what that branch would discard.
      * `PoseModel.build` (``models/model.py``) honours ``with_decoder`` and ``conversion_array``
        for ANY snapshot — it loads the backbone, then calls ``head.convert_weights`` with the
        conversion tensor. Nothing there is SuperAnimal-specific.

    So the config is built through the normal path with the array already stored in
    ``train_settings.weight_init``, and `train_network` flips ``with_decoder`` on so the second
    reader does the mapping. Both readers are DLC's; the only thing we do is not let the first one
    mistake a custom donor for a SuperAnimal model.

    ``bodyparts`` is deliberately NOT set: DLC's validator rejects it without ``conversion_array``
    and only uses it to length-check the array, which `conversion` has already done by construction.
    """
    from deeplabcut.core.weight_init import WeightInitialization

    conv = conversion(bodyparts)
    snap = donor_snapshot()
    if conv is None:
        print("[dlc_train] no complete donor mapping for these bodyparts — backbone only, "
              "head re-initialised", flush=True)
        return WeightInitialization(snapshot_path=snap, with_decoder=False)
    arr, bps = conv
    order = donor_bodyparts()
    print(f"[dlc_train] donor init: {', '.join(f'{b}<-{order[i]}' for b, i in zip(bps, arr))}",
          flush=True)
    return WeightInitialization(snapshot_path=snap, with_decoder=False,
                                conversion_array=np.asarray(arr, dtype=int))


def decoder_wanted(bodyparts: list[str] | None = None) -> bool:
    """Whether the donor's HEAD channels can be carried over, not just its backbone."""
    return conversion(bodyparts) is not None


# ----------------------------------------------------------------------------------- staging

def _label_files(proj: Path, which: str) -> list[Path]:
    root = proj / "labeled-data"
    if not root.is_dir():
        return []
    return sorted(p / f"CollectedData_{SCORER}.h5" for p in root.iterdir()
                  if p.is_dir() and p.name.startswith(which)
                  and (p / f"CollectedData_{SCORER}.h5").is_file())


def subset_labels(df: pd.DataFrame, bodyparts: list[str]) -> pd.DataFrame:
    """Keep only ``bodyparts``, in that order.

    Belt and braces: DLC would do this for us. ``merge_annotateddatasets`` ends with
    ``AnnotationData.reindex(cfg["bodyparts"], axis=1, level="bodyparts")``, which drops columns
    outside the config list (pinned in `tests/test_dlc_train.py`, because it is an implementation
    detail of DLC's rather than a documented promise). Doing it here as well makes the staged files
    SELF-DESCRIBING — `check_labels` and the GUI on this project show four parts, not twelve with
    eight blanked — so what the network was trained on is legible from the project on disk.
    """
    lvl = df.columns.names.index("bodyparts")
    out = df.reindex(bodyparts, axis=1, level=lvl)
    missing = [b for b in bodyparts if b not in set(out.columns.get_level_values("bodyparts"))]
    if missing:
        raise SystemExit(f"bodyparts absent from the labels: {', '.join(missing)}")
    return out


def stage(bodyparts=None, which=None, rv=None, iteration: int | None = None) -> Path:
    """Build/refresh the training project from the labelling project. Returns its path.

    Labels are re-copied FORWARD every run and the copy here is overwritten — see the module
    docstring. Images are copied once (cheap, ~240 PNGs) rather than linked, because a Windows
    symlink needs admin and a project with dangling images fails deep inside a dataloader.
    """
    import yaml

    rv = rv or PathResolver()
    bps, which = list(bodyparts or parts()), which or cam()
    src, dest = dlc_project.project_dir(rv), train_project(rv)
    if not (src / "config.yaml").is_file():
        raise SystemExit(f"No labelling project at {src}. Run dlc_project first.")

    assert_writable(dest)
    for sub in ("labeled-data", "training-datasets", "dlc-models-pytorch", "videos"):
        (dest / sub).mkdir(parents=True, exist_ok=True)

    folders = images = 0
    for h5 in _label_files(src, which):
        sd, dd = h5.parent, dest / "labeled-data" / h5.parent.name
        dd.mkdir(parents=True, exist_ok=True)
        folders += 1
        for img in sorted(sd.glob("img*.png")):
            if not (dd / img.name).exists():
                shutil.copy2(img, dd / img.name)
                images += 1
        d = subset_labels(pd.read_hdf(h5), bps)
        d.to_hdf(dd / f"CollectedData_{SCORER}.h5", key="df_with_missing", mode="w")
        d.to_csv(dd / f"CollectedData_{SCORER}.csv")

    tc = train_cfg()
    cfg_path = dest / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    cfg.update({
        "Task": TASK, "scorer": SCORER, "multianimalproject": False, "identity": None,
        # Rewritten every run: MICROSCOPE is N: on the imaging box and M: on the analysis box, and
        # DLC stores this absolutely (same reason as dlc_project).
        "project_path": str(dest).replace("\\", "/"),
        "bodyparts": bps,                       # THE REFINED SUBSET -- the whole point of this file
        "skeleton": [], "skeleton_color": "black",
        "pcutoff": float(tc.get("pcutoff", 0.6)), "dotsize": 6, "alphavalue": 0.7,
        "colormap": "rainbow",
        # TrainingFraction is REWRITTEN by create_dataset() to the fraction the session hold-out
        # actually produces: DLC looks the derived fraction up in this list
        # (`cfg["TrainingFraction"].index(trainFraction)`) and raises if it is absent.
        "TrainingFraction": cfg.get("TrainingFraction") or [float(tc.get("training_fraction", 0.8))],
        "iteration": int(cfg.get("iteration", 0) if iteration is None else iteration),
        "default_net_type": str(tc.get("net_type", "resnet_50")),
        "default_augmenter": "default", "engine": "pytorch",
        "snapshotindex": -1, "detector_snapshotindex": -1,
        "batch_size": int(tc.get("batch_size", 8)), "detector_batch_size": 8,
        "cropping": False, "start": 0, "stop": 1,
        "numframes2pick": int((_cfg().get("frames") or {}).get("per_session", 24)),
        "move2corner": True, "corner2move2": [50, 50],
        "video_sets": cfg.get("video_sets") or {},
    })
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(f"[dlc_train] staged {folders} {which} folders, +{images} images, "
          f"{len(bps)} bodyparts -> {dest}", flush=True)
    return dest


# -------------------------------------------------------------------------------------- audit

def labels(proj: Path, which: str | None = None) -> pd.DataFrame:
    """Every staged label row for the project, concatenated — DLC's own on-disk order."""
    frames = [pd.read_hdf(h5) for h5 in _label_files(proj, which or cam())]
    if not frames:
        raise SystemExit(f"No labels staged in {proj}/labeled-data")
    return pd.concat(frames).sort_index()


def audit(proj: Path, which: str | None = None) -> pd.DataFrame:
    """Filled-point counts per bodypart per session. Printed on every run, and here is why.

    `parts()` is a declaration, not a measurement, so this is the measurement standing next to it.
    A part declared refined but blank on most frames, or a part quietly filled by a re-run of
    `dlc_prelabel`, both show up here as a number that does not match what you believe.

    A blank is NOT a negative example: DLC masks NaN keypoints out of the loss, so a part is learned
    from the frames where it is placed and nowhere else. That makes the tongue's count the one to
    read carefully — it is out for only ~70 ms per lick, so a low count is the BEHAVIOUR rather than
    unfinished work (`dlc.frames.lick_fraction` 0.5 with one tongue-in offset of four puts ~90 of
    240 cam4 frames in the tongue-out set, which is what 93 placed tongues means).
    """
    rows = {}
    for h5 in _label_files(proj, which or cam()):
        d = pd.read_hdf(h5)
        rows[h5.parent.name] = {
            bp: int(d.loc[:, (slice(None), bp, "x")].notna().sum().sum())
            for bp in dict.fromkeys(d.columns.get_level_values("bodyparts"))
        }
        rows[h5.parent.name]["frames"] = len(d)
    out = pd.DataFrame(rows).T
    return out[["frames"] + [c for c in out.columns if c != "frames"]]


def print_audit(proj: Path, which: str | None = None) -> pd.DataFrame:
    a = audit(proj, which)
    total = a.sum()
    print(f"\n[dlc_train] labels staged for training ({len(a)} sessions, {int(total['frames'])} frames):",
          flush=True)
    for bp in [c for c in a.columns if c != "frames"]:
        n, d = int(total[bp]), int(total["frames"])
        print(f"    {bp:<8s} {n:5d} / {d} frames  ({100 * n / max(d, 1):.0f}%)", flush=True)
    print("    blanks are MASKED OUT of the loss, not taught as absence", flush=True)
    return a


# -------------------------------------------------------------------------------------- split

def session_epochs(rv=None) -> dict[str, str]:
    """``video_stem -> epoch`` from `dlc_frames`' own ``frame_manifest.csv``."""
    man = staging_root(rv) / "frame_manifest.csv"
    if not man.is_file():
        return {}
    m = pd.read_csv(man, usecols=["video_stem", "epoch"], dtype=str)
    return dict(zip(m["video_stem"], m["epoch"]))


def split(proj: Path, which=None, fraction=None, seed=None, rv=None) -> tuple[list[int], list[int]]:
    """Hold out whole SESSIONS, spread across epochs. Returns ``(train_idx, test_idx)``.

    Indices are positions into ``labels(proj)``, which is the same order
    ``merge_annotateddatasets`` produces (per-folder read, ``pd.concat``, ``sort_index``) — that
    equality is what makes these indices mean what DLC will think they mean, and
    `tests/test_dlc_train.py` pins it.

    Sessions rather than frames because four frames of one lick are four views of one pose (see the
    module docstring), so a uniform split reports a test error measured partly on the training set.
    One session per epoch in a seeded order, until the test share is reached: with 15 cam4 sessions
    at one per animal x epoch, ~20% is 3 sessions, and taking them from different epochs is what
    lets `per_epoch_error` say anything about post-stroke frames at all.
    """
    tc = train_cfg()
    frac = float(fraction if fraction is not None else tc.get("training_fraction", 0.8))
    rng = np.random.default_rng(int(seed if seed is not None else tc.get("seed", 42)))
    df = labels(proj, which)
    stems = np.array([ix[1] for ix in df.index])            # ('labeled-data', <stem>, <image>)

    epochs = session_epochs(rv)
    by_epoch: dict[str, list[str]] = {}
    for stem in dict.fromkeys(stems):
        by_epoch.setdefault(epochs.get(stem, "unknown"), []).append(stem)
    for v in by_epoch.values():
        rng.shuffle(v)

    want = len(df) * (1.0 - frac)
    held: list[str] = []
    order = sorted(by_epoch)
    while sum(int((stems == s).sum()) for s in held) < want:
        took = False
        for ep in order:                                     # one per epoch per pass, round-robin
            pool = [s for s in by_epoch[ep] if s not in held]
            if not pool:
                continue
            held.append(pool[0])
            took = True
            if sum(int((stems == s).sum()) for s in held) >= want:
                break
        if not took:                                         # every session held out
            break

    test_mask = np.isin(stems, held)
    if test_mask.all() or not test_mask.any():
        raise SystemExit(f"Session hold-out degenerate ({len(held)} of {len(by_epoch)} epochs' "
                         f"sessions held): too few sessions for fraction {frac}.")
    test_idx = np.flatnonzero(test_mask).tolist()
    train_idx = np.flatnonzero(~test_mask).tolist()
    print(f"\n[dlc_train] held-out sessions ({len(test_idx)} frames, "
          f"{100 * len(test_idx) / len(df):.0f}%):", flush=True)
    for s in held:
        print(f"    {s}  [{epochs.get(s, 'unknown')}]  {int((stems == s).sum())} frames", flush=True)
    print(f"    train {len(train_idx)} frames from {len(set(stems)) - len(held)} sessions", flush=True)
    return train_idx, test_idx


# --------------------------------------------------------------------- training dataset + train

def create_dataset(proj: Path, bodyparts=None, which=None, shuffle=1, rv=None):
    """``deeplabcut.create_training_dataset`` with the donor init and the session hold-out."""
    import yaml

    import deeplabcut

    cfg_path = proj / "config.yaml"
    train_idx, test_idx = split(proj, which, rv=rv)

    # DLC derives the fraction from the indices and then looks it up in cfg["TrainingFraction"]
    # (`.index(trainFraction)`), so a hold-out that does not land on a listed value raises. Write
    # the achieved fraction in before calling rather than forcing the split to hit a round number.
    frac = round(len(train_idx) / (len(train_idx) + len(test_idx)), 2)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if frac not in (cfg.get("TrainingFraction") or []):
        cfg["TrainingFraction"] = [frac]
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    deeplabcut.create_training_dataset(
        str(cfg_path),
        Shuffles=[shuffle],
        net_type=str(train_cfg().get("net_type", "resnet_50")),
        augmenter_type="default",
        weight_init=weight_init(bodyparts),
        trainIndices=[train_idx], testIndices=[test_idx],
        userfeedback=False,          # else `input()` blocks on any re-run of the same shuffle
        engine=deeplabcut.core.engine.Engine.PYTORCH,
    )
    return shuffle, frac


def pytorch_updates(bodyparts: list[str] | None = None) -> dict:
    """``pytorch_cfg_updates`` for ``train_network`` — DLC applies these by dotted path
    (``model_cfg.set_nested``).

    Only what we mean to move off DLC's template. `scaling` defaults to DLC's own ``[0.5, 1.25]``
    (see the module docstring: the donor's one-sided augmentation is not inherited, so there is
    nothing to repair) and is exposed because reaching further DOWN is the one principled change —
    the donor's features live at ~0.4x cam4's apparent size.

    ``train_settings.weight_init.with_decoder`` is the flip described in `weight_init`: the array is
    already in the model config, and this is what makes `PoseModel.build` apply it.
    """
    tc = train_cfg()
    updates: dict = {}
    if decoder_wanted(bodyparts):
        updates["train_settings.weight_init.with_decoder"] = True
    scaling = tc.get("scaling")
    if scaling:
        updates["data.train.affine.scaling"] = [float(scaling[0]), float(scaling[1])]
    if tc.get("dataloader_workers") is not None:
        # DLC's default is 0. Raising it helps here only because the frames live on SMB rather than
        # a local disk; on Windows the workers are spawned, so a non-zero value is a real change in
        # failure modes and stays opt-in.
        updates["train_settings.dataloader_workers"] = int(tc["dataloader_workers"])
    if tc.get("eval_interval") is not None:
        updates["runner.eval_interval"] = int(tc["eval_interval"])
    return updates


def train(proj: Path, shuffle=1, epochs=None, save_epochs=None, device=None, bodyparts=None):
    """``deeplabcut.train_network``. The donor weights are already in the model config; the
    ``with_decoder`` flip that makes DLC APPLY the head mapping rides in on ``pytorch_updates``."""
    import deeplabcut

    tc = train_cfg()
    kw = dict(shuffle=shuffle, trainingsetindex=0,
              batch_size=int(tc.get("batch_size", 8)),
              epochs=int(epochs if epochs is not None else tc.get("epochs", 200)),
              save_epochs=int(save_epochs if save_epochs is not None
                              else tc.get("save_epochs", 25)))
    if device:
        kw["device"] = device
    updates = pytorch_updates(bodyparts)
    if updates:
        kw["pytorch_cfg_updates"] = updates
    print(f"\n[dlc_train] train_network({', '.join(f'{k}={v}' for k, v in kw.items())})", flush=True)
    deeplabcut.train_network(str(proj / "config.yaml"), **kw)


def evaluate(proj: Path, shuffle=1, plotting=False):
    """``deeplabcut.evaluate_network`` with per-keypoint errors.

    ``per_keypoint_evaluation=True`` because one scalar RMSE over four parts hides the two that
    matter: the tongue is the hardest part on the rig and the jaw is occluded at the spout exactly
    when the mouth is open, so a good mean can sit on top of either being bad.
    """
    import deeplabcut

    deeplabcut.evaluate_network(str(proj / "config.yaml"), Shuffles=[shuffle], trainingsetindex=0,
                                plotting=plotting, per_keypoint_evaluation=True,
                                pcutoff=float(train_cfg().get("pcutoff", 0.6)))


# ------------------------------------------------------------------------------ per-epoch error

def _xy(df: pd.DataFrame, bp: str) -> np.ndarray:
    """``(n, 2)`` x/y for one bodypart, from a 3- OR 4-level DLC column index.

    The two tables being compared do not have the same column depth. Labels carry
    ``(scorer, bodyparts, coords)``; `evaluate_network`'s predictions carry
    ``(scorer, individuals, bodyparts, coords)`` — DLC inserts `individuals` even for a
    single-animal project. Selecting positionally therefore put the bodypart name on the
    `individuals` level and raised ``KeyError('nose')`` on the first real evaluation
    (2026-09-21), after a full training run had already completed. Selection is BY LEVEL NAME,
    which is correct for either depth and for whatever DLC adds next.
    """
    sub = df.xs(bp, axis=1, level="bodyparts")
    return np.column_stack([sub.xs(c, axis=1, level="coords").to_numpy(float).ravel()
                            for c in ("x", "y")])


def _newest_predictions(proj: Path) -> Path | None:
    cands = sorted((proj / "evaluation-results-pytorch").rglob("*.h5"),
                   key=lambda p: p.stat().st_mtime) if (proj / "evaluation-results-pytorch").is_dir() else []
    return cands[-1] if cands else None


def per_epoch_error(proj: Path, which=None, rv=None) -> pd.DataFrame | None:
    """Error per EPOCH per bodypart, SPLIT BY train/test — the runbook's acceptance criterion.

    "Judge the result on test rmse, and separately on the post-stroke frames: the failure mode that
    matters is a network that tracks a healthy mouse well and a hemiparetic one badly, which reads
    as a deficit and is not one." That comparison is not a number `evaluate_network` produces, and
    it is the one that decides whether any orofacial result is real, so it is computed here from the
    predictions DLC wrote and `frame_manifest.csv`'s epoch column.

    **THE TRAIN/TEST COLUMN IS NOT COSMETIC.** `evaluate_network` predicts on EVERY labelled image,
    train and test alike, so pooling them makes each epoch's error depend on what share of that
    epoch's sessions happened to be held out — an artefact of the split reported as a property of
    the epoch, which is the same class of error the table exists to catch. Read the `test` rows for
    generalisation; the `train` rows only say whether the fit took.

    Read it as a CONTRAST across epochs, not against a threshold. Degrading tracking and genuine
    recovery are the same measurement until this table says the error is flat. Note which epochs the
    hold-out actually reached: at 20% over four epochs only three sessions fit, so one epoch has no
    test frames and its row is silent rather than reassuring.
    """
    pred_path = _newest_predictions(proj)
    if pred_path is None:
        print("[dlc_train] no evaluation predictions yet — run --evaluate first", flush=True)
        return None
    pred = pd.read_hdf(pred_path)
    truth = labels(proj, which)

    # The split is seeded and deterministic, so recomputing it here recovers the same membership
    # DLC was handed rather than parsing it back out of the Documentation pickle.
    _, test_idx = split(proj, which, rv=rv)
    test_frames = set(truth.index[test_idx])

    idx = truth.index.intersection(pred.index)
    if not len(idx):
        print(f"[dlc_train] predictions in {pred_path.name} share no frames with the labels",
              flush=True)
        return None
    truth, pred = truth.loc[idx], pred.loc[idx]

    epochs = session_epochs(rv)
    stems = np.array([ix[1] for ix in idx])
    ep_of = np.array([epochs.get(s, "unknown") for s in stems])
    is_test = np.array([ix in test_frames for ix in idx])
    rows = []
    have = set(pred.columns.get_level_values("bodyparts"))
    for bp in dict.fromkeys(truth.columns.get_level_values("bodyparts")):
        if bp not in have:
            print(f"[dlc_train] {bp} absent from the predictions — skipped", flush=True)
            continue
        t, p = _xy(truth, bp), _xy(pred, bp)
        err = np.sqrt(((t - p) ** 2).sum(axis=1))
        ok = np.isfinite(err)
        for ep in sorted(set(ep_of)):
            for name, sel in (("test", is_test), ("train", ~is_test)):
                m = ok & sel & (ep_of == ep)
                if m.any():
                    rows.append({"bodypart": bp, "epoch": ep, "split": name, "n": int(m.sum()),
                                 "rmse_px": float(np.sqrt((err[m] ** 2).mean())),
                                 "median_px": float(np.median(err[m]))})
    out = pd.DataFrame(rows)
    if out.empty:
        return None
    for name in ("test", "train"):
        part = out[out["split"] == name]
        if part.empty:
            continue
        print(f"\n[dlc_train] {name} rmse (px) by epoch, from {pred_path.name}:", flush=True)
        print(part.pivot(index="bodypart", columns="epoch", values="rmse_px").round(2).to_string(),
              flush=True)
    print("\n    read the TEST rows: a part whose error rises with epoch will read as a "
          "post-stroke deficit and is not one", flush=True)
    print("    an epoch missing from the test table had no held-out session, so it is untested "
          "rather than fine", flush=True)
    return out


def worst_frames(proj: Path, n: int = 15, which=None, rv=None) -> pd.DataFrame | None:
    """The frames to re-open in ``refine_labels``, worst prediction-vs-label error first.

    This is the refine loop's INPUT, and DLC has no call for it: `extract_outlier_frames` finds
    outliers in a video's predictions (temporal jumps), which needs analysed videos and says nothing
    about a LABEL being wrong. Comparing the network against its own training labels does, and a
    large error on a TRAIN frame is the informative case — the network had every chance to fit it and
    could not, so the label is the thing more likely to be wrong.

    That is not hypothetical. The first real run (2026-09-21) put the `subacute` train tongue RMSE at
    53.36 px, entirely from ONE frame at 266 px (`cam4_2026-08-26T12_25_41/img1796416.png`, likelihood
    0.012): the tongue was labelled at x=619 on a 680 px frame while every other tongue in that
    session sits at x=338-373 and the nose, jaw and spout are all at x=244-323. A misclick on the far
    edge of the frame. The median tongue error over all 93 labelled tongues was 1.91 px.

    **Read `likelihood` next to the error.** Low error + low likelihood is a hard frame; LARGE error
    + low likelihood is the network refusing the label, which is the signature above.
    """
    pred_path = _newest_predictions(proj)
    if pred_path is None:
        print("[dlc_train] no evaluation predictions yet — run --evaluate first", flush=True)
        return None
    pred, truth = pd.read_hdf(pred_path), labels(proj, which)
    idx = truth.index.intersection(pred.index)
    if not len(idx):
        return None
    truth, pred = truth.loc[idx], pred.loc[idx]

    epochs = session_epochs(rv)
    _, test_idx = split(proj, which, rv=rv)
    test_frames = set(truth.index[test_idx])
    have = set(pred.columns.get_level_values("bodyparts"))
    rows = []
    for bp in dict.fromkeys(truth.columns.get_level_values("bodyparts")):
        if bp not in have:
            continue
        err = np.sqrt(((_xy(truth, bp) - _xy(pred, bp)) ** 2).sum(axis=1))
        lik = (pred.xs(bp, axis=1, level="bodyparts")
                   .xs("likelihood", axis=1, level="coords").to_numpy(float).ravel()
               if "likelihood" in set(pred.columns.get_level_values("coords"))
               else np.full(len(err), np.nan))
        for i, ix in enumerate(idx):
            if np.isfinite(err[i]):
                rows.append({"bodypart": bp, "err_px": float(err[i]), "likelihood": float(lik[i]),
                             "split": "test" if ix in test_frames else "train",
                             "epoch": epochs.get(ix[1], "unknown"), "session": ix[1], "image": ix[2]})
    out = pd.DataFrame(rows).sort_values("err_px", ascending=False).reset_index(drop=True)
    print(f"\n[dlc_train] worst {min(n, len(out))} frames (prediction vs label):", flush=True)
    print(out.head(n).to_string(index=False, float_format=lambda v: f"{v:.2f}"), flush=True)
    print("    a LARGE error on a TRAIN frame with LOW likelihood is the network refusing the label "
          "— check that label before blaming the network", flush=True)
    print(f"    open one with: deeplabcut.refine_labels(r'{proj / 'config.yaml'}')", flush=True)
    return out


#: How a frame's fault is classified, from the error and the network's confidence. The label position
#: itself is NOT a criterion: a tongue label 100 px from its session's median is a long protrusion,
#: not a mistake, and a distance-from-median rule flagged 13 cam4 frames of which 12 were fine
#: (2026-09-21). What separates the cases is whether the network DISAGREES, and how sure it is.
VERDICTS = {
    "DELETE": "big error, network sure there is nothing there — the point is stray",
    "REPLACE": "big error, network CONFIDENT elsewhere — the label is in the wrong place",
    "DECIDE": "big error, network unsure — an ambiguous landmark; pick one and hold it",
    "ADD": "no label, network confident — a part that is visible and was never placed",
}


def classify(proj: Path, which=None, rv=None, bodyparts=None, tol: float = 8.0,
             sure: float = 0.7, absent: float = 0.05) -> pd.DataFrame:
    """One row per (frame, bodypart) needing review, with a VERDICT — see `VERDICTS`.

    Grouped by fault rather than ranked by error, because the three faults take three different
    actions and a single ranking interleaves them. The `lick` column matters as much as the frame:
    `dlc.frames.lick_offsets_s` samples four points of one protrusion and pruning keeps a lick
    WHOLE, so a fault that repeats across a lick's frames is a LANDMARK fault and the whole lick gets
    re-placed. A fault on one frame of a lick is a slip.
    """
    pred_path = _newest_predictions(proj)
    if pred_path is None:
        print("[dlc_train] no evaluation predictions yet — run --evaluate first", flush=True)
        return pd.DataFrame()
    pred, truth = pd.read_hdf(pred_path), labels(proj, which)
    idx = truth.index.intersection(pred.index)
    truth, pred = truth.loc[idx], pred.loc[idx]

    man_path = staging_root(rv) / "frame_manifest.csv"
    man = pd.read_csv(man_path, dtype=str) if man_path.is_file() else pd.DataFrame()
    # The manifest is metadata, not data: without it every frame still classifies, it just loses
    # epoch/animal/lick. Missing columns must therefore degrade rather than raise.
    if {"video_stem", "image", "cam"} <= set(man.columns):
        man = man[man["cam"] == (which or cam())].set_index(["video_stem", "image"])
    else:
        man = pd.DataFrame()
    # 250 fps: the four offsets are onset + {-4, 0, +8, +16} frames, so the onset is recoverable
    # and a lick has a stable id across its frames.
    offsets = {"lick-16": -4, "lick+0": 0, "lick+32": 8, "lick+64": 16}

    _, test_idx = split(proj, which, rv=rv)
    test_frames = set(truth.index[test_idx])
    have = set(pred.columns.get_level_values("bodyparts"))
    has_lik = "likelihood" in set(pred.columns.get_level_values("coords"))

    rows = []
    for bp in (bodyparts or dict.fromkeys(truth.columns.get_level_values("bodyparts"))):
        if bp not in have:
            continue
        t, p = _xy(truth, bp), _xy(pred, bp)
        lik = (pred.xs(bp, axis=1, level="bodyparts").xs("likelihood", axis=1, level="coords")
                   .to_numpy(float).ravel() if has_lik else np.full(len(t), np.nan))
        for i, ix in enumerate(idx):
            placed = bool(np.isfinite(t[i]).all())
            err = float(np.sqrt(((t[i] - p[i]) ** 2).sum())) if placed else np.nan
            if placed and err <= tol:
                continue
            if not placed and not (lik[i] > 0.6):
                continue
            verdict = ("ADD" if not placed else
                       "DELETE" if lik[i] < absent else
                       "REPLACE" if lik[i] > sure else "DECIDE")
            m = man.loc[(ix[1], ix[2])] if len(man) and (ix[1], ix[2]) in man.index else None
            phase = None if m is None else m["phase"]
            rows.append({
                "verdict": verdict, "bodypart": bp, "session": ix[1], "image": ix[2],
                "phase": phase,
                "lick": (f"{ix[1]}#{int(m['frame']) - offsets[phase]}"
                         if m is not None and phase in offsets else None),
                "epoch": None if m is None else m["epoch"],
                "animal": None if m is None else m["animal"],
                "position": None if m is None else m["position"],
                "split": "test" if ix in test_frames else "train",
                "err_px": err, "likelihood": float(lik[i]),
                "dx": t[i][0] - p[i][0] if placed else np.nan,
                "dy": t[i][1] - p[i][1] if placed else np.nan,
                "label_x": t[i][0], "label_y": t[i][1],
                "pred_x": p[i][0], "pred_y": p[i][1]})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    order = {"DELETE": 0, "REPLACE": 1, "DECIDE": 2, "ADD": 3}
    return (out.assign(_o=out.verdict.map(order))
               .sort_values(["_o", "err_px"], ascending=[True, False])
               .drop(columns="_o").reset_index(drop=True))


def print_review(proj: Path, which=None, rv=None, tol: float = 8.0) -> pd.DataFrame:
    """`classify` grouped by verdict, plus the whole-lick roll-up."""
    d = classify(proj, which, rv, tol=tol)
    if d.empty:
        print("\n[dlc_train] nothing over tolerance — no frames to review", flush=True)
        return d
    cols = ["bodypart", "session", "image", "phase", "epoch", "animal", "position", "split",
            "err_px", "likelihood", "dx", "dy"]
    for v, why in VERDICTS.items():
        g = d[d.verdict == v]
        if not len(g):
            continue
        print(f"\n{'=' * 104}\n{v} ({len(g)}) — {why}\n{'=' * 104}", flush=True)
        print(g[cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"), flush=True)

    licks = (d[d.lick.notna()].groupby(["bodypart", "lick"])
              .agg(frames=("image", "size"), max_err=("err_px", "max"),
                   mean_lik=("likelihood", "mean"), epoch=("epoch", "first"),
                   animal=("animal", "first"), split=("split", "first")))
    licks = licks[licks.frames >= 2].sort_values("max_err", ascending=False)
    if len(licks):
        print(f"\n{'=' * 104}\nWHOLE LICKS — 2+ frames of one protrusion are wrong, so it is the "
              f"LANDMARK, not the frame\n{'=' * 104}", flush=True)
        print(licks.to_string(float_format=lambda x: f"{x:.2f}"), flush=True)
        print("    re-place these licks TOGETHER: seeing the tongue move is what makes 'the tip' "
              "identifiable at all", flush=True)
    return d


def review_images(proj: Path, which=None, rv=None, dest: Path | None = None, tol: float = 8.0,
                  pad: int = 110) -> Path | None:
    """Write one annotated crop per frame to review — LABEL in red, PREDICTION in cyan.

    A coordinate table cannot settle which of the two is right; only the image can. These are
    written so the review is "look at this folder" rather than "open the GUI 12 times and find the
    frame". `refine_labels`/`label_frames` is still where the FIX happens.
    """
    import cv2

    d = classify(proj, which, rv, tol=tol)
    dest = dest or (proj / "review")
    assert_writable(dest)
    dest.mkdir(parents=True, exist_ok=True)
    # Clear FIRST, and clear even when there is nothing to write. A round that fixes everything
    # leaves an empty table, and an early return would leave the PREVIOUS round's crops sitting
    # there to be read as current — the one output of this whole loop that must not go stale.
    for old in dest.glob("*.png"):
        old.unlink()
    if d.empty:
        print(f"\n[dlc_train] nothing to review — {dest} cleared", flush=True)
        return dest

    n = 0
    for _, r in d.iterrows():
        img_path = proj / "labeled-data" / r.session / r.image
        im = cv2.imread(str(img_path))
        if im is None:
            continue
        cx = float(r.pred_x if not np.isfinite(r.label_x) else (r.label_x + r.pred_x) / 2)
        cy = float(r.pred_y if not np.isfinite(r.label_y) else (r.label_y + r.pred_y) / 2)
        h, w = im.shape[:2]
        x0, y0 = max(0, int(cx - pad)), max(0, int(cy - pad))
        x1, y1 = min(w, int(cx + pad)), min(h, int(cy + pad))
        crop = im[y0:y1, x0:x1].copy()
        if np.isfinite(r.label_x):
            cv2.drawMarker(crop, (int(r.label_x - x0), int(r.label_y - y0)), (0, 0, 255),
                           cv2.MARKER_CROSS, 18, 2)
        cv2.circle(crop, (int(r.pred_x - x0), int(r.pred_y - y0)), 6, (255, 255, 0), 2)
        crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        cv2.putText(crop, f"{r.verdict} {r.bodypart} err={r.err_px:.0f} p={r.likelihood:.2f}",
                    (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(crop, f"{r.phase} {r.epoch}", (6, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
        out = dest / f"{r.verdict}_{r.bodypart}_{r.session}_{Path(r.image).stem}.png"
        cv2.imwrite(str(out), crop)
        n += 1
    print(f"\n[dlc_train] {n} annotated crops -> {dest}", flush=True)
    print("    RED CROSS = your label,  CYAN CIRCLE = the network. Decide which is right.", flush=True)
    return dest


def next_round(proj: Path) -> None:
    """How the next refinement round attaches — DLC's loop, not a new one.

    Nothing to automate yet (it needs analysed videos, which is `dlc.o2.*`'s job), so this prints
    the four DLC calls rather than pretending to a step the pipeline has not ported. The important
    part is that `iteration` is the counter: `merge_datasets` bumps it, so each round's training
    set and model folder are separate and an earlier network stays reproducible.
    """
    print(f"""
[dlc_train] next refinement round (DeepLabCut's own loop):

    import deeplabcut as d
    cfg = r"{proj / 'config.yaml'}"
    d.analyze_videos(cfg, [<video>], save_as_csv=False)
    d.extract_outlier_frames(cfg, [<video>], outlieralgorithm="jump", epsilon=20)
    d.refine_labels(cfg)              # fix the network's own worst frames
    d.merge_datasets(cfg)             # merges + BUMPS `iteration`
    python -m wfield_local.dlc_train  # re-stage at the new iteration and retrain

Widen `dlc.train.bodyparts` only when a human has actually placed the new part — the whisker
columns on cam4 are still donor seeds, and nothing here can tell a seed from a label.""",
          flush=True)


# ------------------------------------------------------------------- seeding the new frames

def seed_pending(proj: Path | None = None, rv=None, shuffle: int = 1, dry: bool = False,
                 pcutoff: float | None = None) -> int:
    """Seed the UNLABELLED frames with THIS network's predictions. Returns points written.

    Frames added by a later `dlc_frames` run arrive blank, and blank-page labelling is several
    times slower than correction. `dlc_prelabel` exists for exactly this and seeds from the 2pRAM
    DONOR -- which was right when there was nothing better, and is not any more: on this view the
    donor needed a 0.45 rescale to find the nose at all and still found the tongue on about half of
    true tongue-out frames, where the network trained here sits at a median 1.5-1.9 px.

    **SEEDED ONLY ABOVE `pcutoff`, blank below it, and that is the whole design.** These frames were
    chosen by APPEARANCE DIVERSITY -- deliberately the poses the set has least of -- so they are
    where this network is least reliable, and its own confidence is the right gate. `nose` and
    `spout` will seed nearly everywhere and cost only a glance; the tongue stays blank precisely
    where the network is unsure, which is where an independent judgement is worth having.

    **The honest cost:** seeding from our own network means the labeller nudges its predictions
    rather than placing fresh, which biases the labels toward what the network already believes.
    That is standard `refine_labels` practice and the confidence gate bounds it, but it is real, and
    it is strongest on the tongue -- the part with the fewest training frames and the least settled
    landmark. `--worst` on the NEXT round is the check: a seeded frame that nobody moved and that
    the network then fits perfectly is not evidence.

    EXISTING ROWS ARE NEVER TOUCHED. Rows are added for pending images only, and every pre-existing
    row is asserted byte-identical before the file is written -- the 240 corrected labels are the
    expensive artefact here and this is the one function that opens their file for writing.
    """
    import numpy as np

    from wfield_local import dlc_prelabel, dlc_review_guide

    rv = rv or PathResolver()
    proj = proj or train_project(rv)
    live = dlc_project.project_dir(rv)
    bps = parts()
    cut = float(pcutoff if pcutoff is not None else train_cfg().get("pcutoff", 0.6))

    pending = dlc_review_guide.pending_frames(live, cam(), bps)
    if not pending:
        print("[dlc_train] no unlabelled frames -- nothing to seed", flush=True)
        return 0
    index = pd.DataFrame([{"video_stem": stem, "image": img,
                           "path": str(live / "labeled-data" / stem / img)}
                          for stem, imgs in sorted(pending.items()) for img in imgs])
    print(f"[dlc_train] seeding {len(index)} unlabelled frames in {len(pending)} folders "
          f"from {proj.name} (pcutoff {cut})", flush=True)

    work = proj / "_seed"
    work.mkdir(parents=True, exist_ok=True)
    video = work / "pending.avi"
    import cv2
    w0, h0 = cv2.imread(index.iloc[0]["path"]).shape[1::-1]
    dw, dh = dlc_prelabel.build_video(index, 1.0, video)
    # build_video rounds to a multiple of 32 (the backbone's stride), so predictions come back in
    # the ROUNDED frame and have to be mapped home. 680 -> 672 is a 1.2% shrink, which is 4 px at
    # the edge of the frame -- small, and silently wrong if not undone.
    sx, sy = w0 / dw, h0 / dh

    import deeplabcut
    for stale in list(work.glob("pending*.h5")) + list(work.glob("pending*.pickle")):
        stale.unlink()
    deeplabcut.analyze_videos(str(proj / "config.yaml"), [str(video)], videotype="avi",
                              shuffle=shuffle, trainingsetindex=0, save_as_csv=False,
                              destfolder=str(work), batch_size=8)
    h5 = sorted(work.glob("pending*.h5"))
    if not h5:
        raise RuntimeError("analyze_videos produced no .h5 for the pending frames")
    pred = pd.read_hdf(h5[-1])
    if len(pred) != len(index):
        raise RuntimeError(f"{len(pred)} predictions for {len(index)} frames -- these pair BY "
                           f"POSITION, so a mismatch would seed the wrong images.")

    # Pull x / y / likelihood per bodypart ONCE, BY LEVEL NAME. Doing it positionally inside
    # the row loop put 'likelihood' on the `scorer` level and raised KeyError -- the same
    # mistake, and the same fix, as `_xy` above.
    have = set(pred.columns.get_level_values("bodyparts"))
    chan = {}
    for bp in bps:
        if bp not in have:
            print(f'    {bp}: not predicted by this network -- left blank', flush=True)
            continue
        sub = pred.xs(bp, axis=1, level="bodyparts")
        chan[bp] = tuple(sub.xs(c, axis=1, level="coords").to_numpy(float).ravel()
                         for c in ("x", "y", "likelihood"))

    written = 0
    for stem, g in index.groupby("video_stem", sort=True):
        dest = live / "labeled-data" / stem / f"CollectedData_{SCORER}.h5"
        old = pd.read_hdf(dest) if dest.is_file() else None
        cols = old.columns if old is not None else pd.MultiIndex.from_tuples(
            [(SCORER, b, c) for b in bps for c in ("x", "y")],
            names=["scorer", "bodyparts", "coords"])
        rows = {}
        for pos, img in zip(g.index, g["image"]):
            vals = {}
            for bp, (xs_, ys_, lk_) in chan.items():
                if lk_[pos] >= cut:
                    vals[(SCORER, bp, "x")] = xs_[pos] * sx
                    vals[(SCORER, bp, "y")] = ys_[pos] * sy
            rows[("labeled-data", stem, img)] = {c: vals.get(c, np.nan) for c in cols}
        add = pd.DataFrame.from_dict(rows, orient="index", columns=cols)
        add.index = pd.MultiIndex.from_tuples(add.index)
        n = int(np.isfinite(add.to_numpy(float)).sum() // 2)
        written += n
        print(f"    {stem}: {len(add)} frames, {n} points seeded", flush=True)
        if dry:
            continue
        out = pd.concat([old, add]).sort_index() if old is not None else add.sort_index()
        if old is not None:
            # the guarantee, checked rather than trusted
            pd.testing.assert_frame_equal(out.loc[old.index], old)
        assert_writable(dest.parent)
        out.to_hdf(dest, key="df_with_missing", mode="w")
        out.to_csv(dest.with_suffix(".csv"))
    if not dry:
        # A SEED LOOKS EXACTLY LIKE A LABEL on disk -- that is the premise this whole module is
        # built on -- so "which frames has a person actually checked?" cannot be recovered from the
        # label files afterwards. Record it at the one moment it is known. The guide reads this to
        # keep listing the seeded frames as work-to-do; without it they would silently disappear
        # from the page the instant they were seeded, which is the opposite of what is wanted.
        rec = work / "seeded_frames.csv"
        pd.DataFrame([{"session": s_, "image": i_} for s_, imgs in sorted(pending.items())
                      for i_ in imgs]).to_csv(rec, index=False)
        print(f"[dlc_train] recorded {len(index)} seeded frames -> {rec}", flush=True)
    print(f"[dlc_train] {written} points seeded{' (dry-run)' if dry else ''}", flush=True)
    return written


def seeded_frames(proj: Path | None = None, rv=None) -> "dict[str, list[str]]":
    """``{session: [image, ...]}`` that `seed_pending` filled in and NOBODY HAS CHECKED YET.

    Cleared by hand (delete `_seed/seeded_frames.csv`) once the labeller has been through them.
    There is no way to detect "checked" from the files: a seed and a corrected label are the same
    four numbers, which is exactly why `dlc.train.bodyparts` is a hand-maintained contract too.
    """
    proj = proj or train_project(rv)
    rec = proj / "_seed" / "seeded_frames.csv"
    if not rec.is_file():
        return {}
    df = pd.read_csv(rec, dtype=str)
    return {s_: sorted(g["image"]) for s_, g in df.groupby("session")}


# ---------------------------------------------------------------------------------------- run

def run(bodyparts=None, which=None, rv=None, dry=False, shuffle=1, epochs=None, save_epochs=None,
        device=None, evaluate_only=False, iteration=None, plotting=False, worst=15,
        review=False, guide=False) -> Path:
    rv = rv or PathResolver()
    bps = list(bodyparts or parts())
    proj = stage(bps, which, rv, iteration)
    print_audit(proj, which)

    if evaluate_only:
        evaluate(proj, shuffle, plotting)
        per_epoch_error(proj, which, rv)
        if worst:
            worst_frames(proj, worst, which, rv)
        if review or guide:
            print_review(proj, which, rv)
            review_images(proj, which, rv)
        if guide:
            from wfield_local import dlc_review_guide
            dlc_review_guide.build(proj, rv)
        return proj

    if dry:
        split(proj, which, rv=rv)
        conv = conversion(bps)
        print(f"\n[dlc_train] donor snapshot : {donor_snapshot()}", flush=True)
        print(f"[dlc_train] conversion_array: {conv[0] if conv else 'none (head re-initialised)'}",
              flush=True)
        print(f"[dlc_train] pytorch updates : {pytorch_updates(bps)}", flush=True)
        print("[dlc_train] --dry-run: nothing trained", flush=True)
        return proj

    create_dataset(proj, bps, which, shuffle, rv)
    train(proj, shuffle, epochs, save_epochs, device, bps)
    evaluate(proj, shuffle, plotting)
    per_epoch_error(proj, which, rv)
    if worst:
        worst_frames(proj, worst, which, rv)
    if review or guide:
        print_review(proj, which, rv)
        review_images(proj, which, rv)
    if guide:
        from wfield_local import dlc_review_guide
        dlc_review_guide.build(proj, rv)
    next_round(proj)
    return proj


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", action="append", default=None,
                    help="override dlc.train.bodyparts (repeatable). ONLY parts a human has placed.")
    ap.add_argument("--cam", default=None, help="camera to train on (default: dlc.train.cam)")
    ap.add_argument("--dry-run", action="store_true",
                    help="stage, audit, print the split and the donor mapping; train nothing")
    ap.add_argument("--evaluate", action="store_true",
                    help="evaluate the newest snapshot and report per-epoch error; do not train")
    ap.add_argument("--shuffle", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--save-epochs", type=int, default=None)
    ap.add_argument("--iteration", type=int, default=None,
                    help="refinement round (DLC's `iteration`); default: keep the project's")
    ap.add_argument("--device", default=None, help="e.g. cuda:0 (default: DLC's auto)")
    ap.add_argument("--worst", type=int, default=15,
                    help="how many worst prediction-vs-label frames to list (0 = none)")
    ap.add_argument("--review", action="store_true",
                    help="classify the frames needing review (DELETE/REPLACE/DECIDE/ADD) and "
                         "write annotated crops to <project>/review/")
    ap.add_argument("--guide", action="store_true",
                    help="also (re)generate CORRECTION_GUIDE.html on the share, for whoever "
                         "is doing the labelling")
    ap.add_argument("--plotting", action="store_true", help="write evaluation overlay images")
    ap.add_argument("--machine", default=None)
    args = ap.parse_args(argv)
    run(args.part, args.cam, PathResolver(machine=args.machine), args.dry_run, args.shuffle,
        args.epochs, args.save_epochs, args.device, args.evaluate, args.iteration, args.plotting,
        args.worst, args.review, args.guide)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
