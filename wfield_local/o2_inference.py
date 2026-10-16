"""Whole-session cam4 pose inference on HMS O2: build a self-contained BUNDLE on the share, print the commands to
run it, and collect the results back into the session-analysis format.

    python -m wfield_local.o2_inference bundle  --name r3_ps93 --user ab123 PS93:20260814 PS93:20260821 ...
    python -m wfield_local.o2_inference collect --name r3_ps93               # after the outputs are copied back

PORTED FROM stroke_orofacial_pipeline `dlc_inference_o2/` (batch.py, o2_job.py, runner.py, transfer.py; Priya
2026-10-02: "it will need refining and editing but was a way to run dlc on the server. lets port and modify").
KEPT: Python GENERATES the ssh / rsync / sbatch commands and you paste them -- nothing here logs in to O2 or
handles a password (O2 needs DUO; their 2026-04-18 decision). The two-hop data path (research.files is visible
only on the TRANSFER node: stage to /n/scratch there, compute on GPU nodes, copy back there). One GPU per job.
CHANGED:
  * The bundle lives ON THE SHARE (`<DeepLabCut/Widefield>/o2/<name>/`), so the transfer node pulls it straight
    from /n/files -- nothing has to be copied from this desktop to O2.
  * Prediction = `o2_pose_runner.py` (copied into the bundle): our model + the `dlc_prior` heatmap prior, i.e.
    the same numbers as local inference (theirs: `deeplabcut.analyze_videos`, which would skip the prior;
    TensorFlow, shuffle 5, allow_growth, snapshot renaming -- all gone with DLC 3 / PyTorch).
  * A job ARRAY, one task per video (theirs: one job looping over every video for 9.5 h), each task resumable in
    chunks, with --requeue: a preempted or time-limited task resumes where it stopped.
  * Videos: our cam4 tree (`Behavior_Cameras/Widefield/<date>/<animal>/cam4_*.avi`), not their 2pRAM `video2_*`.
  * `collect` checks every video's frame count against the predictions and turns the whole-video CSV into the
    per-trial windows the session analyses read (`session_poses/<animal>_<date>_full/`).
O2 facts to re-check before a big run (they change): partition names / time limits (`sinfo`), the conda module
name, and which CUDA the GPU nodes' driver supports (pick the torch wheel to match).
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from wfield_local import config

SHARE_PREFIX_O2 = "/n/files/Neurobio/MICROSCOPE/Priya"   # research.files path of the `microscope` root on O2


def params(kind: str = "dlc") -> dict:
    """`o2_inference` settings; for ``kind="lp"`` the `lp:` block overrides the shared (partition, scratch, ...) keys."""
    base = dict(config.defaults().get("o2_inference") or {})
    lp = base.pop("lp", None) or {}
    return {**base, **lp, "kind": kind} if kind == "lp" else {**base, "kind": "dlc"}


def true_case(path: Path) -> Path:
    """``path`` with every existing component spelled as it is ON DISK.

    Windows / SMB resolve names case-insensitively, so `configs/paths.yaml` has carried
    ``Behavior_Cameras/Widefield`` for months while the folders are ``Behavior_cameras/widefield`` (found
    2026-10-05). O2 is Linux: a path that works here can fail there. Components that do not exist (yet) are
    kept as given."""
    path = Path(path)
    out = Path(path.anchor)
    for part in path.parts[len(path.anchor and path.parts[:1]):]:
        try:
            match = next((c.name for c in out.iterdir() if c.name.lower() == part.lower()), part)
        except OSError:
            match = part
        out = out / match
    return out


def o2_path(local, rv) -> str:
    """A path under the `microscope` share root -> the same path as O2's transfer node sees it, in the
    on-disk case (`true_case`)."""
    root = true_case(Path(rv.root("microscope"))).as_posix().rstrip("/")
    p = true_case(Path(local)).as_posix()
    if not p.lower().startswith(root.lower()):
        raise ValueError(f"{p} is not under the share root {root}; O2 cannot see it")
    return SHARE_PREFIX_O2 + p[len(root):]


def scratch(user: str, p: dict | None = None) -> str:
    p = p or params()
    return p["scratch_root"].format(u0=user[0], user=user)


def bundle_dir(rv, name: str) -> Path:
    from wfield_local import dlc_project
    return dlc_project.project_dir(rv).parent / "o2" / name


def videos_for(rv, animal: str, date: str) -> list[Path]:
    return sorted((Path(rv.root("behavior_cameras")) / date / animal).glob("cam4_*.avi"))


def _first_video(user: str, p: dict, tasks: pd.DataFrame) -> str:
    r = tasks.iloc[0]
    return f"{scratch(user, p)}/o2_videos/{r.date}/{r.animal}/{r.video}"


def write_videos_list(d: Path, tasks: pd.DataFrame) -> Path:
    """videos.txt for `rsync --files-from`: each video's path relative to the camera root (<date>/<animal>/<file>),
    LF line endings (read on Linux)."""
    rel = [str(v).rsplit("/", 3) for v in tasks.video_o2]
    out = d / "videos.txt"
    out.write_text("".join(f"{r[1]}/{r[2]}/{r[3]}\n" for r in rel), encoding="utf-8", newline="\n")
    return out


def smoke_sbatch_text(p: dict, user: str, name: str, video_first: str) -> str:
    """LP smoke test as a BATCH job: the runner's GPU check + `--bench 2000` on the first video. Queued with the
    array behind it (`--dependency=afterok`), so no one waits at an interactive srun."""
    b = f"{scratch(user, p)}/o2/{name}"
    return f"""#!/bin/bash -l
#SBATCH -J smoke_{name}
#SBATCH -p {p['partition']}
#SBATCH --gres={p['gres']}
#SBATCH -c {p['cpus_per_task']}
#SBATCH --mem={p['mem']}
#SBATCH -t 0:45:00
#SBATCH -o {b}/logs/smoke_%j.out
#SBATCH -e {b}/logs/smoke_%j.err
# Generated by wfield_local.o2_inference -- regenerate rather than edit.
set -eo pipefail
module load {p['conda_module']}
conda activate {p['conda_env']}
nvidia-smi || true
python {b}/o2_lp_runner.py --model {b}/model --video {video_first} --bench 2000
"""


def runner_line(p: dict, b: str, video: str, out: str) -> str:
    """The prediction command for one video: DLC (`o2_pose_runner.py`, chunked + prior) or LP (`o2_lp_runner.py`)."""
    if p.get("kind") == "lp":
        return (f"python {b}/o2_lp_runner.py --model {b}/model --video {video} \\\n"
                f"    --out {out} --reader {p['reader']} --precision {p['precision']}")
    return (f"python {b}/o2_pose_runner.py --bundle {b} --video {video} \\\n"
            f"    --out {out} --batch {p['batch']} --chunk {p['chunk']}")


def sbatch_text(p: dict, user: str, name: str, n_tasks: int) -> str:
    s = scratch(user, p)
    b = f"{s}/o2/{name}"
    run = runner_line(p, b, f"{s}/o2_videos/$DATE/$ANIMAL/$VIDEO", f"{b}/out/${{ANIMAL}}_${{DATE}}")
    return f"""#!/bin/bash -l
#SBATCH -J {'lp' if p.get('kind') == 'lp' else 'pose'}_{name}
#SBATCH -p {p['partition']}
#SBATCH --gres={p['gres']}
#SBATCH -c {p['cpus_per_task']}
#SBATCH --mem={p['mem']}
#SBATCH -t {p['time']}
#SBATCH --array=1-{n_tasks}%{p['max_concurrent']}
#SBATCH --requeue
#SBATCH -o {b}/logs/%A_%a.out
#SBATCH -e {b}/logs/%A_%a.err
{f"#SBATCH --exclude={p['exclude_nodes']}" if p.get('exclude_nodes') else ''}
# Generated by wfield_local.o2_inference -- regenerate rather than edit. One array task = one video; a re-run or
# requeue skips finished videos (DLC also resumes inside a video from the last finished chunk).
# Login shell (-l) + `conda activate`, as the working 2025-11 O2 script; no `set -u` (conda's activate scripts reference unset variables).
set -eo pipefail
module load {p['conda_module']}
conda activate {p['conda_env']}
LINE=$(awk -v n="$SLURM_ARRAY_TASK_ID" -F'\\t' 'NR==n+1' {b}/tasks.tsv)
ANIMAL=$(echo "$LINE" | cut -f2); DATE=$(echo "$LINE" | cut -f3); VIDEO=$(echo "$LINE" | cut -f4)
echo "task $SLURM_ARRAY_TASK_ID: $ANIMAL $DATE $VIDEO on $(hostname)"; nvidia-smi || true
{run}
"""


def commands_text(p: dict, user: str, name: str, share_b: str, tasks: pd.DataFrame) -> str:
    s = scratch(user, p)
    b = f"{s}/o2/{name}"
    # ONE rsync over a file list (videos.txt in the bundle, paths relative to the camera root), not a line per video:
    # rsync skips files already on scratch, so it only fills gaps (all 288 cam1 + cam4 videos to 10-06 were staged
    # on 2026-10-06; the scratch purge counts 45 days ON scratch)
    cam_root = tasks.video_o2.iloc[0].rsplit("/", 3)[0]
    v0 = f"{s}/o2_videos/{tasks.date.iloc[0]}/{tasks.animal.iloc[0]}/{tasks.video.iloc[0]}"
    step0, step2 = (_lp_steps if p.get("kind") == "lp" else _dlc_steps)(p, user, name, b, v0)
    if p.get("kind") == "lp":
        # NO INTERACTIVE GPU WAIT (Priya 2026-10-15: every srun waits for an allocation): the smoke test is a batch
        # job and the array is queued behind it with afterok -- two sbatch lines, then walk away
        step23 = f"""## 2-3. Login node: queue the smoke test (GPU check + benchmark, logs/smoke_<job>.out) and the array behind it.
##      The array starts only if the smoke job succeeds (afterok); if it fails, read the smoke log, fix, resubmit both.
ssh {user}@o2.hms.harvard.edu
mkdir -p {b}/logs
SMOKE=$(sbatch --parsable {b}/smoke.sbatch) && echo "smoke job $SMOKE"
sbatch --dependency=afterok:$SMOKE {b}/run_pose.sbatch
#   later: cat {b}/logs/smoke_$SMOKE.out  -> torch/CUDA/GPU line, then fps per reader"""
    else:
        step23 = f"""## 2. FIRST RUN: smoke test + benchmark on a GPU node (proves the env runs OUR runner + prior)
ssh {user}@o2.hms.harvard.edu
srun -p {p['partition']} --gres={p['gres']} -t 0:30:00 --mem={p['mem']} --pty bash
module load {p['conda_module']}; conda activate {p['conda_env']}
{step2}

## 3. Login node: submit the array
ssh {user}@o2.hms.harvard.edu
mkdir -p {b}/logs && sbatch {b}/run_pose.sbatch"""
    return f"""# O2 run `{name}` ({p.get('kind', 'dlc').upper()}) -- {len(tasks)} video(s). Paste these yourself; nothing here logs in for you.
# Generated by `python -m wfield_local.o2_inference bundle`. Bundle on the share: {share_b}

{step0}

## 1. Transfer node: bundle -> scratch, then fill any gaps in the staged videos (research.files is visible only here)
##    Videos already on scratch (staged 2026-10-06) are skipped; only missing ones copy (e.g. sessions after 10-06).
ssh {user}@{p['transfer_host']}
mkdir -p {b} {s}/o2_videos
rsync -ah --partial {share_b}/  {b}/
rsync -ah --partial --info=progress2 --files-from={b}/videos.txt {cam_root}/  {s}/o2_videos/

{step23}

## 4. Monitor
squeue -u {user}
tail -f {b}/logs/*_1.out
ls {b}/out/*/*_done.json | wc -l          # finished videos (of {len(tasks)})

## 5. Transfer node: results -> the share (into the bundle folder)
ssh {user}@{p['transfer_host']}
rsync -ah --partial --exclude {"'tmp*'" if p.get('kind') == 'lp' else "'*_chunks'"} {b}/out/  {share_b}/out/

## 6. This desktop: check + convert to the session-analysis format
python -m wfield_local.o2_inference collect --name {name}{'''
#   LP, first run: compare O2 against the desktop's prediction of the same model on a session_poses clip, e.g.
#   --compare PS93:20260814=C:/Users/SabatiniLab/lp_stage/r3_vs_r4_lp/r4heatmap/PS93_20260814_windows.csv''' if p.get('kind') == 'lp' else ''}

## 7. Clean up THIS run's scratch once collected. NEVER remove the staged videos ({s}/o2_videos) from here:
##    they are shared by every run, and stay until Priya confirms ALL analysis on the server is finished.
rm -rf {b}/out/*/*_chunks
"""


def _dlc_steps(p: dict, user: str, name: str, b: str, v0: str) -> tuple[str, str]:
    step0 = f"""## 0. Check the EXISTING env (Priya's O2 login already has `{p['conda_env']}` -- the env the 2025-11 stroke_orofacial
##    O2 runs used, PyTorch DLC 3 models). Our runner needs DLC 3.x (tested with {p['deeplabcut_version']}).
ssh {user}@o2.hms.harvard.edu
srun -p {p['partition']} --gres={p['gres']} -t 0:20:00 --mem=8G --pty bash
module load {p['conda_module']}
conda activate {p['conda_env']}
python -c "import torch, deeplabcut, cv2; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'dlc', deeplabcut.__version__, 'cv2', cv2.__version__)"
#   needs: cuda True and dlc 3.x. Step 2 (--bench) is the real check -- it builds the model and runs the prior.
#   FALLBACK only if the env is DLC 2.x / broken (do NOT modify the existing env -- other analyses use it):
#     conda create -y -n dlc3 python=3.10 && conda activate dlc3 && pip install "deeplabcut=={p['deeplabcut_version']}"
#     then set o2_inference.conda_env: dlc3 in configs/defaults.yaml and regenerate the bundle"""
    step2 = f"""python {b}/o2_pose_runner.py --bundle {b} --video {v0} --bench 2000
#   -> if another batch size is clearly faster: set o2_inference.batch in configs/defaults.yaml, then
#      `python -m wfield_local.o2_inference bundle --refresh --name {name} --user {user} <same sessions>`
#      and re-run the bundle rsync in step 1 (only the text files change)"""
    return step0, step2


def _lp_steps(p: dict, user: str, name: str, b: str, v0: str) -> tuple[str, str]:
    step0 = f"""## 0. ONE TIME: create an `{p['conda_env']}` env with Lightning Pose {p['lightning_pose_version']} (do NOT touch the
##    `deeplabcut` env -- other analyses use it). A CPU node is enough to install.
ssh {user}@o2.hms.harvard.edu
srun -p short -t 1:00:00 -c 4 --mem=16G --pty bash
module load {p['conda_module']}
conda create -y -n {p['conda_env']} python=3.10 && conda activate {p['conda_env']}
pip install "lightning-pose=={p['lightning_pose_version']}" "opencv-python-headless=={p['opencv_version']}"
#   (lightning-pose does NOT install OpenCV; the runner reads the AVIs with it -- the first O2 smoke job, 2026-10-15,
#   failed on `import cv2`. If the env already exists: conda activate {p['conda_env']} && pip install the second package.)
exit
#   (The GPU check is step 2: the runner prints torch / CUDA / LP versions first and stops if no GPU is visible.
#   lightning-pose already installs DALI (nvidia-dali-cuda110) -- do NOT add another DALI build. Our cam4 AVIs are
#   MPEG-4 part 2, which DALI is not known to decode, so the runner uses reader {p['reader']}; the smoke job's bench
#   also tries dali and just reports if it fails.)"""
    step2 = f"""python {b}/o2_lp_runner.py --model {b}/model --video {v0} --bench 2000
#   -> fps per reader on the first 2000 frames (re-encoded clip). Keep reader {p['reader']} unless another is clearly
#      faster AND a real video run with it matches the desktop (`collect --compare`); readers preprocess differently,
#      so never mix them within one run."""
    return step0, step2


def cmd_bundle(rv, name: str, user: str, sessions: list[tuple[str, str]], refresh: bool = False,
               kind: str = "dlc") -> Path:
    from wfield_local import dlc_prior, dlc_train
    from wfield_local.dlc_iti_frames import current_snapshot
    d = bundle_dir(rv, name)
    if (d / "bundle.json").exists():
        kind = json.loads((d / "bundle.json").read_text(encoding="utf-8")).get("kind", "dlc")   # as bundled
    p = params(kind)
    runner = "o2_lp_runner.py" if kind == "lp" else "o2_pose_runner.py"
    if (d / "bundle.json").exists():
        if not refresh:
            raise SystemExit(f"{d} exists -- pick a new --name (a bundle is a record of what ran), or --refresh to "
                             f"rewrite only its generated text (sbatch, COMMANDS.md, runner)")
        # --refresh: regenerate the TEXT only -- the model, prior and task list stay exactly as bundled
        tasks = pd.read_csv(d / "tasks.tsv", sep="\t", dtype=str).drop(columns="task")
        # Same videos, re-SPELT in the on-disk case (`true_case`; bundles made before 2026-10-05 carry
        # paths.yaml's Behavior_Cameras/Widefield, which a case-sensitive O2 path lookup would not find).
        local = {(a, dt): {v.name: v for v in videos_for(rv, a, dt)} for a, dt in zip(tasks.animal, tasks.date)}
        tasks["video_o2"] = [o2_path(local[(a, dt)][v], rv) if v in local[(a, dt)] else o2
                             for a, dt, v, o2 in zip(tasks.animal, tasks.date, tasks.video, tasks.video_o2)]
        tasks.insert(0, "task", np.arange(1, len(tasks) + 1))
        tasks.to_csv(d / "tasks.tsv", sep="\t", index=False)
        write_videos_list(d, tasks)
        tasks = tasks.drop(columns="task")
        shutil.copy2(Path(__file__).with_name(runner), d / runner)
        if kind == "lp":            # the provenance record follows the job text (reader / precision / GPU type)
            bj = json.loads((d / "bundle.json").read_text(encoding="utf-8"))
            bj.update(reader=p["reader"], precision=p["precision"], gres=p["gres"])
            (d / "bundle.json").write_text(json.dumps(bj, indent=1), encoding="utf-8")
        (d / "run_pose.sbatch").write_text(sbatch_text(p, user, name, len(tasks)), encoding="utf-8", newline="\n")
        if p.get("kind") == "lp":
            (d / "smoke.sbatch").write_text(smoke_sbatch_text(p, user, name, _first_video(user, p, tasks)),
                                            encoding="utf-8", newline="\n")
        (d / "COMMANDS.md").write_text(commands_text(p, user, name, o2_path(d, rv), tasks), encoding="utf-8",
                                       newline="\n")
        print(f"refreshed text in {d}")
        return d
    rows = []
    for animal, date in sessions:
        vids = videos_for(rv, animal, date)
        if not vids:
            print(f"   no cam4 video for {animal} {date} -- skipped (see docs/EXPERIMENT_ERRORS.md)")
            continue
        rows += [{"animal": animal, "date": date, "video": v.name, "video_o2": o2_path(v, rv)} for v in vids]
    tasks = pd.DataFrame(rows)
    if kind == "lp":
        return _bundle_lp(rv, d, p, name, user, tasks)
    td, snap = current_snapshot(dlc_train.train_project(rv), None)
    box = dlc_prior.active(rv, cam="cam4") or {}
    (d / "model").mkdir(parents=True)
    shutil.copy2(td / "pytorch_config.yaml", d / "model" / "pytorch_config.yaml")
    shutil.copy2(snap, d / "model" / snap.name)
    shutil.copy2(Path(__file__).with_name("o2_pose_runner.py"), d / "o2_pose_runner.py")
    scorer = f"DLC_{td.parent.parent.name}_{snap.stem}"          # e.g. DLC_iteration-2_snapshot-best-160
    (d / "bundle.json").write_text(json.dumps({
        "parts": list(dlc_train.parts()), "scorer": scorer, "pytorch_config": "model/pytorch_config.yaml",
        "snapshot": f"model/{snap.name}", "prior": {k: [float(v) for v in b] for k, b in box.items()},
        "source_train_dir": str(td), "created_for": user}, indent=1), encoding="utf-8")
    tasks.insert(0, "task", np.arange(1, len(tasks) + 1))
    tasks.to_csv(d / "tasks.tsv", sep="\t", index=False)
    write_videos_list(d, tasks)
    (d / "run_pose.sbatch").write_text(sbatch_text(p, user, name, len(tasks)), encoding="utf-8", newline="\n")
    if p.get("kind") == "lp":
        (d / "smoke.sbatch").write_text(smoke_sbatch_text(p, user, name, _first_video(user, p, tasks)),
                                        encoding="utf-8", newline="\n")
    (d / "COMMANDS.md").write_text(commands_text(p, user, name, o2_path(d, rv), tasks), encoding="utf-8",
                                   newline="\n")
    print(f"bundle -> {d}  ({len(tasks)} videos; model {td.parent.parent.name}/{snap.name}; prior on "
          f"{', '.join(box) or 'nothing'})\nnext: follow {d / 'COMMANDS.md'}")
    return d


def widefield_sessions() -> list[tuple[str, str]]:
    """(animal, YYYYMMDD) for every registered widefield session, sorted -- Priya 2026-10-15: pose inference only for
    dates with widefield data (the camera also ran on pre-widefield days, 0527 / 0529 / 0601, and after imaging ended,
    PS94 1008). A session with no cam4 video (PS93 0813, docs/EXPERIMENT_ERRORS.md) is reported, not fatal."""
    out = set()
    for s in config.load_sessions():
        animal, mmdd = s["label"].split("_")
        # neither baseline nor post-stroke (`configs/animals.yaml` exclude: PS92 / PS93 0817, the day after the
        # first, small lesion that did not take) -> not analysed, so not tracked (Priya 2026-10-15)
        if config.session_phase(animal, mmdd) == "excluded":
            continue
        out.add((animal, "2026" + mmdd))
    return sorted(out)


def lp_model_dir(rv, p: dict | None = None) -> Path:
    """The production LP model on the share (`o2_inference.lp.model_rel`, relative to DeepLabCut/Widefield)."""
    from wfield_local import dlc_project
    p = p or params("lp")
    return dlc_project.project_dir(rv).parent / p["model_rel"]


def _bundle_lp(rv, d: Path, p: dict, name: str, user: str, tasks: pd.DataFrame) -> Path:
    """LP bundle: the model dir's config.yaml + checkpoint(s) under model/ (config's data paths pointed at the scratch
    copy -- `compute_metrics=False` predictions do not read them, but a stale desktop path is a trap), the runner,
    bundle.json (kind lp, scorer, parts), tasks.tsv, run_pose.sbatch, COMMANDS.md."""
    import yaml
    src = lp_model_dir(rv, p)
    ckpts = sorted(src.glob("tb_logs/**/checkpoints/*.ckpt"))
    if not (src / "config.yaml").exists() or not ckpts:
        raise SystemExit(f"{src}: needs config.yaml and tb_logs/**/checkpoints/*.ckpt")
    (d / "model").mkdir(parents=True)
    for c in ckpts:
        dst = d / "model" / c.relative_to(src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(c, dst)
    cfg = yaml.safe_load((src / "config.yaml").read_text(encoding="utf-8"))
    o2_model = f"{scratch(user, p)}/o2/{name}/model"
    cfg["data"]["data_dir"] = o2_model
    cfg["data"]["video_dir"] = o2_model
    cfg["model"]["losses_to_use"] = []          # inference only: no unlabelled-video loss (needs DALI + videos)
    (d / "model" / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8", newline="\n")
    shutil.copy2(Path(__file__).with_name("o2_lp_runner.py"), d / "o2_lp_runner.py")
    scorer = f"LP_{src.name}"
    (d / "bundle.json").write_text(json.dumps({
        "kind": "lp", "parts": list(cfg["data"]["keypoint_names"]), "scorer": scorer, "model": "model",
        "checkpoints": [str(c.relative_to(src)) for c in ckpts], "reader": p["reader"], "precision": p["precision"],
        "source_model_dir": str(src), "created_for": user}, indent=1), encoding="utf-8")
    tasks.insert(0, "task", np.arange(1, len(tasks) + 1))
    tasks.to_csv(d / "tasks.tsv", sep="\t", index=False)
    write_videos_list(d, tasks)
    tasks = tasks.drop(columns="task")
    (d / "run_pose.sbatch").write_text(sbatch_text(p, user, name, len(tasks)), encoding="utf-8", newline="\n")
    if p.get("kind") == "lp":
        (d / "smoke.sbatch").write_text(smoke_sbatch_text(p, user, name, _first_video(user, p, tasks)),
                                        encoding="utf-8", newline="\n")
    (d / "COMMANDS.md").write_text(commands_text(p, user, name, o2_path(d, rv), tasks), encoding="utf-8",
                                   newline="\n")
    print(f"LP bundle -> {d}  ({len(tasks)} videos; model {src.name}, reader {p['reader']})\n"
          f"next: follow {d / 'COMMANDS.md'}")
    return d


def compare_to_clip(pose: np.ndarray, parts: list[str], clip_dir: Path, clip_csv: Path, cut: float = 0.4) -> str:
    """O2 whole-video predictions vs a desktop prediction of the same model on a `session_poses` clip (rows =
    the clip's windows_index src_frame): per part, the median / p95 distance where both are confident, and how often
    they agree on confident-or-not. The readers differ (O2 OpenCV on the AVI; desktop DALI / OpenCV on the
    re-encoded clip), so expect small differences, not zero."""
    idx = pd.read_csv(clip_dir / "windows_index.csv")
    df = pd.read_csv(clip_csv, header=[0, 1, 2], index_col=0)
    df.columns = df.columns.droplevel(0)
    n = min(len(idx), len(df))
    src = idx.src_frame.to_numpy()[:n]
    ok = src < len(pose)
    out = []
    for k, part in enumerate(parts):
        if part not in df.columns.get_level_values(0):
            continue
        a = pose[src[ok], k]
        b = df[part][["x", "y", "likelihood"]].to_numpy(float)[:n][ok]
        both = (a[:, 2] >= cut) & (b[:, 2] >= cut)
        dist = np.hypot(a[both, 0] - b[both, 0], a[both, 1] - b[both, 1])
        agree = np.mean((a[:, 2] >= cut) == (b[:, 2] >= cut))
        out.append(f"{part}: median {np.median(dist):.2f} px, p95 {np.percentile(dist, 95):.2f} px "
                   f"(n {both.sum()}), confident-or-not agree {100 * agree:.1f} %" if both.any() else f"{part}: none")
    return "; ".join(out)


def trial_frame_spans(b: pd.DataFrame, tpl: dict, n_full: int, pre_s: float = 0.5, post_s: float = 3.0) -> list:
    """[(f0, f1, f_cue), ...] per trial of ``b`` (sorted by cue): position strobe - pre_s .. trial stop + post_s,
    CUT at the next trial's start, so no video frame belongs to two trials (Priya, 2026-10-06: "just cut at the next
    trial's start"). Before the cut, a trial's last 3 s overlapped the next trial's strobe window (~3 % of frames
    twice in a whole session) -- harmless for the 10-06 analyses, but anything summing over rows double-counted."""
    from wfield_local import dlc_frames as DF
    spans = []
    for r in b.itertuples():
        t0 = min(r.strobe_s, r.cue_s - pre_s) if np.isfinite(r.strobe_s) else r.cue_s - pre_s
        f0, f1 = DF.frame_of(tpl, t0 - pre_s, 0.0), DF.frame_of(tpl, r.stop_s + post_s, 0.0)
        spans.append([max(0, f0), min(n_full, f1), DF.frame_of(tpl, r.cue_s, 0.0)])
    for k in range(len(spans) - 1):
        spans[k][1] = min(spans[k][1], spans[k + 1][0])
    return [tuple(x) for x in spans]


def assign_trials_to_videos(cue_s: np.ndarray, videos: list) -> np.ndarray:
    """Index into ``videos`` [(pose, template, name), ...] of the video that covers each cue: its frame lies inside
    that video, and inside the template's DAQ segment when it has one (split days). -1 = no video covers it."""
    from wfield_local import dlc_frames as DF
    who = np.full(len(cue_s), -1)
    for v, (pose, tpl, _name) in enumerate(videos):
        seg = tpl.get("segment_daq_sec")
        for i, c in enumerate(cue_s):
            if who[i] >= 0:
                continue
            f = DF.frame_of(tpl, c, 0.0)
            if 0 <= f < len(pose) and (seg is None or seg[0] <= c <= seg[1]):
                who[i] = v
    return who


def full_to_windows(pose: np.ndarray, parts: list[str], scorer: str, animal: str, date: str, out: Path, rv,
                    pre_s: float = 0.5, post_s: float = 3.0, model: str = "DLC", videos: list | None = None) -> int:
    """Whole-video predictions ``pose`` (n_frames, n_parts, 3) -> `session_poses`-format windows for EVERY trial
    (`trial_frame_spans`: position strobe - pre_s .. trial stop + post_s, cut at the next trial's start):
    windows_index.csv (trial_k, trial_id, position, src_frame, t_ms, strobe_ms, stop_ms, stop_source) +
    windows_<model>.csv with the matching rows; the analyses then run unchanged. Takes the runner's .npz ARRAY,
    not its CSV: exact, and no 2-million-row parse (and a CSV whose first data row is empty would be misread by
    pandas' 3-row-header reader as an index-name row -- found 2026-10-02 while testing this with a NaN-padded
    file)."""
    ix, sel = build_windows(pose, animal, date, rv, pre_s=pre_s, post_s=post_s, videos=videos)
    out.mkdir(parents=True, exist_ok=True)
    ix.to_csv(out / "windows_index.csv", index=False)
    if not len(sel):
        sel = np.zeros((0, len(parts), 3), np.float32)
    cols = pd.MultiIndex.from_product([[scorer], list(parts), ["x", "y", "likelihood"]],
                                      names=["scorer", "bodyparts", "coords"])
    pd.DataFrame(sel.reshape(len(sel), -1), columns=cols).to_csv(out / f"windows_{model}.csv")
    return int(ix.trial_k.nunique()) if len(ix) else 0


def build_windows(pose, animal: str, date: str, rv, pre_s: float = 0.5, post_s: float = 3.0,
                  videos: list | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    """``(windows_index, pose rows)`` for EVERY trial -- the arrays `full_to_windows` writes, without writing them
    (`orofacial_outputs` stores them in its own layout). See `full_to_windows` for the window and split-day rules."""
    from wfield_local import trial_windows as TW
    # ``videos`` (OURS, 2026-10-15, PS92 0812): [(pose, template, video name), ...] for a SPLIT day -- each trial is
    # cut from the video that covers its cue (`assign_trials_to_videos`), windows cut at the next trial of the SAME
    # video, and windows_index gains a `video` column. Default: one video with the date's template.
    if videos is None:
        tpl = dict(np.load(Path(rv.root("alignment_templates")) / "cam4" / animal / f"{date}.npz", allow_pickle=True))
        videos = [(pose, tpl, None)]
    b = TW.trial_bounds(animal, date, rv)
    b = b[np.isfinite(b.cue_s) & np.isfinite(b.stop_s)].sort_values("cue_s").reset_index(drop=True)
    who = assign_trials_to_videos(b.cue_s.to_numpy(float), videos)
    if (who < 0).any():
        print(f"   {animal} {date}: {int((who < 0).sum())} trial(s) not covered by any video -- left out")
    from wfield_local import camera_sync as cs
    idx, sels = [], []
    k = 0
    for v, (pose_v, tpl_v, name_v) in enumerate(videos):
        bv = b[who == v].reset_index(drop=True)
        # ``daq_s`` (OURS, 2026-10-16): each frame's DAQ time from its camera TIMESTAMP (`camera_sync.frame_daq_seconds`),
        # so imaging-aligned analyses need no template. ``t_ms`` stays the nominal 4-ms-per-frame offset from the cue
        # frame (what every kinematics function expects); the two agree to < 0.01 ms on drop-free recordings.
        daq_v = cs.frame_daq_seconds(rv, animal, date, tpl_v)
        for r, (f0, f1, fc) in zip(bv.itertuples(), trial_frame_spans(bv, tpl_v, len(pose_v), pre_s, post_s)):
            for f in range(f0, f1):
                idx.append((k, int(r.trial_id), r.pos_name, f, (f - fc) * 4.0, (r.strobe_s - r.cue_s) * 1000.0,
                            (r.stop_s - r.cue_s) * 1000.0, r.stop_source, float(daq_v[f]), name_v))
            sels.append(pose_v[np.arange(f0, f1)])
            k += 1
    ix = pd.DataFrame(idx, columns=["trial_k", "trial_id", "position", "src_frame", "t_ms", "strobe_ms", "stop_ms",
                                    "stop_source", "daq_s", "video"])
    if len(videos) == 1:
        ix = ix.drop(columns="video")
    sel = np.concatenate(sels) if sels else np.zeros((0, 0, 3), np.float32)
    return ix, sel


def cmd_collect(rv, name: str, compare: dict | None = None) -> None:
    """Check each finished video's frame count and write `session_poses/<a>_<d>_full/` windows (windows_DLC.csv or,
    for an LP bundle, windows_LP.csv -- the same windows_index either way). ``compare``: {"ANIMAL:DATE": clip csv}
    -> `compare_to_clip` against a desktop prediction on that session's `session_poses/<a>_<d>/` clip."""
    import cv2
    d = bundle_dir(rv, name)
    kind = json.loads((d / "bundle.json").read_text(encoding="utf-8")).get("kind", "dlc")
    tasks = pd.read_csv(d / "tasks.tsv", sep="\t", dtype=str)
    from wfield_local import dlc_project
    sp = dlc_project.project_dir(rv).parent / "session_poses"
    from wfield_local import camera_sync as cs
    for (animal, date), g in tasks.groupby(["animal", "date"], sort=True):
        od = d / "out" / f"{animal}_{date}"
        vids, ok_all = [], True
        for r in g.itertuples():
            done = od / f"{Path(r.video).stem}_done.json"
            if not done.exists():
                print(f"{animal} {date} {r.video}: NOT DONE (no {done.name})")
                ok_all = False
                continue
            info = json.loads(done.read_text(encoding="utf-8"))
            v = next(x for x in videos_for(rv, animal, date) if x.name == r.video)
            n_local = int(cv2.VideoCapture(str(v)).get(cv2.CAP_PROP_FRAME_COUNT))
            npz = od / f"{Path(r.video).stem}_{info['scorer']}.npz"
            ok = info["n_frames"] == n_local and npz.exists()
            print(f"{animal} {date} {r.video}: {info['n_frames']} frames predicted, video {n_local} "
                  f"{'OK' if ok else 'MISMATCH'} ({info['wall_s'] / 3600:.1f} h, batch {info['batch']})")
            ok_all &= ok
            if ok:
                vids.append((r.video, np.load(npz), info))
        if not ok_all or not vids:
            continue                                  # a split day is collected only when every part is done
        # each video with ITS recording's template (split days have one per recording -- camera_sync)
        tpls = cs.templates_for(rv, date, animal, "cam4")
        rec = lambda name: Path(name).stem.split("_", 1)[1]           # noqa: E731  cam4_<recording>.avi
        missing = [n for n, _, _ in vids if rec(n) not in tpls]
        if missing:
            print(f"   {animal} {date}: no alignment template for {missing} -- not collected")
            continue
        parts = [str(x) for x in vids[0][1]["parts"]]
        videos = [(z["pose"], t, n) for n, z, _ in vids for t in tpls[rec(n)]]   # one entry per template / segment
        n = full_to_windows(videos[0][0], parts, vids[0][2]["scorer"], animal, date, sp / f"{animal}_{date}_full",
                            rv, model="LP" if kind == "lp" else "DLC", videos=videos if len(videos) > 1 else None)
        print(f"   -> {sp / (animal + '_' + date + '_full')} ({n} trials)")
        key = f"{animal}:{date}"
        if compare and key in compare and len(videos) == 1:
            print(f"   vs desktop clip: {compare_to_clip(videos[0][0], parts, sp / f'{animal}_{date}', Path(compare[key]))}")


def main(argv=None) -> int:
    from wfield_local.paths import PathResolver
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bundle")
    b.add_argument("--name", required=True)
    b.add_argument("--user", required=True, help="O2 (HMS) username")
    b.add_argument("sessions", nargs="*", help="animal:date (or --all-widefield)")
    b.add_argument("--all-widefield", action="store_true",
                   help="every REGISTERED widefield session (configs/sessions.yaml) -- the production set")
    b.add_argument("--refresh", action="store_true", help="existing bundle: rewrite only sbatch / COMMANDS / runner")
    b.add_argument("--kind", choices=("dlc", "lp"), default="dlc",
                   help="lp = the production Lightning Pose model (o2_inference.lp.model_rel)")
    c = sub.add_parser("collect")
    c.add_argument("--name", required=True)
    c.add_argument("--compare", action="append", default=[], metavar="ANIMAL:DATE=CSV",
                   help="desktop prediction of the same model on that session's session_poses clip")
    a = ap.parse_args(argv)
    rv = PathResolver()
    if a.cmd == "bundle":
        sess = widefield_sessions() if a.all_widefield else [tuple(s.split(":")) for s in a.sessions]
        if not sess and not a.refresh:
            ap.error("give animal:date sessions or --all-widefield")
        cmd_bundle(rv, a.name, a.user, sess, refresh=a.refresh, kind=a.kind)
    else:
        cmd_collect(rv, a.name, dict(x.split("=", 1) for x in a.compare))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
