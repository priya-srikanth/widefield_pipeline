# Runbook — whole-session pose inference on HMS O2

First run end to end on 2026-10-05 (Priya + Claude): round-3 cam4 DLC model, PS93 0814, one video. Code:
`wfield_local/o2_inference.py` (builds the bundle, prints the commands, collects results) and
`wfield_local/o2_pose_runner.py` (runs on O2; same predictor + prior as this desktop). Settings:
`configs/defaults.yaml` `o2_inference`. Nothing in Python logs in to O2 — O2 needs DUO, so every command below is
pasted by a person.

## The shape of it

```
this desktop                     O2 transfer node                O2 login / GPU nodes
bundle -> share (M:)  ------->   /n/files (the share)  ---rsync--> /n/scratch (yours)  ---> sbatch array
collect <- share (M:) <-------   /n/files  <---rsync---------------  /n/scratch/out
```

* **The lab share is visible ONLY on the transfer nodes** (`ssh ps150@transfer.rc.hms.harvard.edu`, prompt
  `transfer0x`) at `/n/files/Neurobio/MICROSCOPE/Priya/...`. On a login node (`login0x`) the same path does not
  exist — the first 10-05 rsync failed exactly so.
* **Scratch:** `/n/scratch/users/p/ps150` (create once: `/n/cluster/bin/scratch_create_directory.sh`). 25 TiB,
  files deleted **45 days after last modification**, no backups. Visible from login, transfer and compute nodes.
* **O2 paths are case-sensitive; the share's Windows view is not.** `configs/paths.yaml` says
  `Behavior_Cameras/Widefield`, the folders are `Behavior_cameras/widefield`. `o2_inference.true_case` writes
  every path in its on-disk spelling (fixed 10-05, before the first copy).
* **DUO:** every `ssh` asks for a push. If pushes stop arriving, re-enable DUO push in the HMS account settings
  (needed on 10-05). Keep one terminal per host open rather than re-connecting.

## Environment (checked 10-05)

| item | value |
|---|---|
| conda module | `conda/miniforge3/24.11.3-0` (worked; HMS docs say non-`miniconda3` modules were hidden in 2024 — if `module load` fails, `module spider miniconda3` and update `o2_inference.conda_module`) |
| env | `deeplabcut` (Priya's): DeepLabCut **3.0.0rc13**, torch 2.9.0+cu128, OpenCV 4.11 — runs our runner + prior (benchmark and 5000-frame test passed). Do NOT modify it; if it ever breaks, make a separate `dlc3` env (COMMANDS.md step 0). |
| GPU seen | `gpu` partition, NVIDIA **L40S 46 GB**, driver CUDA 13.2 (>= the 12.8 torch needs) |
| speed | `--bench 2000`: batch 1 98.1 fps, 4 89.2, 8 75.0, 16 70.9, 32 68.7, 64 67.2 → **batch 1** (CPU decode is the limit, not the GPU). Real runs: 66 fps on a 5000-frame test, **88 fps** sustained in the array job; a cam4 session is ~2.2 M frames (PS93 0814: 2 209 351) → **~7 h per session**. |

## Partitions (HMS docs, Oct 2026 — re-check with `sinfo`)

* `gpu` — open to all, but only ~20 cards (V100 16 GB, L40S 48 GB): interactive `srun` can wait a long time.
* `gpu_quad` — 142 cards, open to labs with a pre-clinical appointment (Neurobiology qualifies); test access with
  `srun -p gpu_quad --gres=gpu:1 -t 0:05:00 --mem=2G --pty nvidia-smi`. Not yet tested by us.
* `gpu_requeue` — idle purchased cards, may be killed and requeued at any time, 24 h limit. Fine for short tests,
  and for our jobs (`--requeue`, resumable per 25 000-frame chunk).
* Checks that need no GPU (imports, versions): `srun -p short -t 0:15:00 --mem=8G --pty bash` allocates at once.
  The runner itself falls back to CPU but is far too slow there (a 300-frame CPU test timed out in 30 min).
* `exit` from an `srun` shell ENDS that allocation; to add a shell to a live one: `srun --jobid=<id> --overlap --pty bash`.

## Procedure

1. **Desktop:** `python -m wfield_local.o2_inference bundle --name <name> --user ps150 <ANIMAL:YYYYMMDD> ...` →
   `<DeepLabCut/Widefield>/o2/<name>/` with the model, runner, `tasks.tsv`, `run_pose.sbatch`, `COMMANDS.md`.
   After a settings change: `bundle --refresh` (rewrites text only; the model and video list stay).
2. **Transfer node:** rsync the bundle and the videos to scratch (COMMANDS.md step 1).
3. **First time on a new env / node type — smoke test on a GPU node** (COMMANDS.md step 2): `--bench 2000`, then
   `--out <scratch>/test_out --max-frames 5000` and `ls` the output (expect `<video>_<scorer>.csv`, `.npz`,
   `_done.json`, a `_chunks/` dir).
4. **Login node:** `sbatch <scratch bundle>/run_pose.sbatch` (all videos) or `sbatch --array=1 ...` (one).
   Monitor: `squeue -u ps150`; `tail -n 5 <bundle>/logs/*_<task>.out`; `.err` should hold only DLC's
   "light mode" notice. The job survives logging out.
5. **Transfer node:** rsync `out/` back into the bundle on the share (COMMANDS.md step 5).
6. **Desktop:** `python -m wfield_local.o2_inference collect --name <name>` → `session_poses/<a>_<d>_full/`.
7. Clean scratch videos and `_chunks/` once collected.

## Status 2026-10-05

`r3_PS93_20261002` (round-3 cam4 model, 3 PS93 videos): smoke test passed; array task 1 (0814) submitted as job
55345148 and running. Tasks 2-3 need their videos copied first. This is a PIPELINE test — the production run
waits for the next cam4 model (labelling round 4).

## Next on O2

* Lightning Pose **training** there (the 4-5-model ensembles the multi-camera EKS needs): a new `lp` env
  (lightning-pose 2.4.2 + NVIDIA DALI, which runs on O2's Linux nodes), and a training bundle analogous to the
  pose bundle. Test with one short training run before the real ensemble.
* Try `gpu_quad` access; if granted, set `o2_inference.partition: gpu_quad`.
