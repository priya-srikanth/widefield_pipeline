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
  files deleted **45 days after they were put on scratch** (time on scratch, not the files' dates -- Priya 10-15), no backups. Visible from login, transfer and compute nodes.
  **Never change file dates on scratch** (`touch` etc.): HMS treats artificial modification times as a policy
  violation. Copy with COMMANDS.md's `rsync -ah`.
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
6. **Desktop**, from the repo folder (`C:\Users\SabatiniLab\Github\widefield_pipeline`), `dlc` env (h5py added
   10-06) or `locanmf`: `python -m wfield_local.o2_inference collect --name <name>` → `session_poses/<a>_<d>_full/`.
   Checks predicted frames == the video's frame count, then splits the whole video into every trial's window
   (strobe - 0.5 s .. stop + 3 s; neighbouring windows overlap a few s) in the `session_poses` format.
7. Clean `_chunks/` once collected. Staged videos stay until Priya confirms all server analysis is finished.

## Lightning Pose (the production cam4 model, from 2026-10-15)

Priya chose **LP r4** (single-frame heatmap, round-4 labels; `docs/POSE_MODELS.md`) as the cam4 production model. The
same bundle -> scratch -> array -> collect shape, with `--kind lp`:

* `python -m wfield_local.o2_inference bundle --kind lp --name <name> --user ps150 <ANIMAL:YYYYMMDD> ...` copies the
  model from the share (`DeepLabCut/Widefield/lightning-pose/cam4-2026-10-13/models/r4_heatmap_20261013/`, set by
  `o2_inference.lp.model_rel`), points its config's data paths at the scratch copy, and writes the runner
  `o2_lp_runner.py`, `run_pose.sbatch` and `COMMANDS.md`.
* **New env on O2** (`lp`, lightning-pose 2.4.2): COMMANDS.md step 0. Never Priya's `deeplabcut` env. LP already
  installs DALI (`nvidia-dali-cuda110`); do not add a second DALI build.
* **Reader = OpenCV** (CPU decode). Our cam4 AVIs are MPEG-4 part 2 (FMP4), which DALI's GPU decoder is not known
  to read. LP's readers preprocess slightly differently, so one run uses one reader, and the first run is checked
  against the desktop: `collect --compare PS93:20260814=<desktop LP csv for session_poses/PS93_20260814>`.
* **Not resumable inside a video** (LP predicts a file in one pass); a requeue skips finished videos. Speed on O2 is
  unmeasured: `--bench 2000` in step 2. If CPU decode limits as it did for DLC (~90-100 fps), ~6-8 h per session.
* Output contract = the DLC runner's (`<stem>_<scorer>.npz/.csv`, `_done.json`), so `collect` is shared; it writes
  `windows_LP.csv` next to `windows_index.csv` in `session_poses/<a>_<d>_full/`.
* Tested 2026-10-15 on this desktop (CPU, 300 frames): the bundled model + runner reproduce the share model's
  predictions exactly. Not yet run on O2.
* Pilot bundle ready: `o2/lp_r4_pilot_20261015` (PS93 0814 + 0821, both with desktop LP r4 clip predictions for
  `--compare`).
* **No interactive GPU waits (2026-10-15):** the LP smoke test (GPU check + `--bench 2000`) is a batch job,
  `smoke.sbatch`, and the array is queued behind it with `--dependency=afterok` -- two `sbatch` lines, then walk away.
  The only interactive step is creating the `lp` env once, on the `short` CPU partition.
* **Video staging = one `rsync --files-from=<bundle>/videos.txt`**: it skips videos already on scratch (all 288
  cam1 + cam4 to 10-06 were staged 2026-10-06) and copies only missing ones. Staged videos are SHARED by every run --
  **never delete `o2_videos` until Priya confirms all analysis on the server is finished** (Priya, 2026-10-15).
* **Partition gpu_quad** (Priya's `gpuquad_qos`; 5-day limit; no per-user cap listed), array throttle %12;
  production = ONE array over every registered widefield session (`bundle --kind lp --all-widefield`, 138 videos).
* **Scratch purge and file dates (Priya, 2026-10-15):** the 45-day purge counts time ON SCRATCH, not the files'
  modification dates, so `rsync -ah` (which keeps the original August dates) is right -- keep it, and never alter
  file dates. The 10-06 video copies were staged on 10-06; step 1's rsync re-stages anything already purged.

## LP production run — the exact steps (as run 2026-10-15)

The bundle's own `COMMANDS.md` (`<DeepLabCut/Widefield>/o2/<name>/COMMANDS.md`) is the copy-paste source; it is
regenerated by `bundle --refresh`. In order, and where each runs:

| step | node | what |
|---|---|---|
| 0 | `short` CPU partition (once) | create the `lp` env: `pip install "lightning-pose==2.4.2" "opencv-python-headless==4.12.0.88"` (LP brings DALI cuda110 itself) |
| 1 | transfer node | rsync the bundle to scratch; `rsync --files-from=<bundle>/videos.txt` fills any gaps in the staged videos (dry run with `-n --itemize-changes` to see what would copy -- `--info=progress2`'s %/ETA count the skipped files too) |
| 2-3 | login node | pilot / new setup only: `SMOKE=$(sbatch --parsable smoke.sbatch)` then `sbatch --dependency=afterok:$SMOKE run_pose.sbatch`; a validated setup: just `sbatch run_pose.sbatch` |
| 4 | login node | `squeue -u ps150`; `sacct -j <id> --format=JobID%16,State,ExitCode,Elapsed -X`; finished: `ls out/*/*_done.json \| wc -l`; raise parallelism live: `scontrol update JobId=<id> ArrayTaskThrottle=24`; opening to any GPU type (`Gres=gpu:1`) is fine ONLY with the V100 nodes excluded (status 10-16) |
| 5 | transfer node | copy back, incremental, any time: `rsync -ah --partial --exclude 'tmp*' <scratch bundle>/out/ <share bundle>/out/` (`tmp*` = a running video's temporary folder) |
| 6 | this desktop | `python -m wfield_local.o2_inference collect --name <name>` (dlc env; on the analysis desktop set `WIDEFIELD_FIGURES_WORKING` / `WIDEFIELD_SESSION_CACHE`); checks every frame count against the share |
| 7 | -- | staged videos are NEVER deleted until Priya confirms all server analysis is finished |

Timing (L40S, DALI): ~210 fps, i.e. wall time ~ frames / 210 (June sessions ~1.2-1.5 h, August ~2.4-2.9 h). Fair
share is charged on elapsed time x resources, not on the requested limit; a shorter request (6 h now the default) only
helps the job backfill sooner.

## Status 2026-10-15 — LP r4 production RUNNING

* **Pilot `lp_r4_pilot_20261015`** (PS93 0814 / 0821, gpu_quad L40S, reader dali): 2,209,351 and 1,794,061 frames in
  2.9 / 2.4 h (~210 fps); frame counts match the videos. `collect --compare` vs the desktop's LP r4 clip predictions:
  median 0.15-0.45 px, p95 0.4-1.7 px (tongue largest), confident-or-not agreement >= 99.7 % on every keypoint ->
  O2 reproduces the desktop. Collected into `session_poses/PS93_<d>_full/windows_LP.csv` (575 / 540 trials).
  Smoke bench (Quadro RTX 8000): opencv ~59 / dali ~105 fps -> DALI + L40S chosen.
* **Production `lp_r4_prod_20261015`**: 137 videos (every registered widefield session; PS93 0813 has none; PS92
  0812 contributes both files), submitted 2026-10-15 as array job **56464442** (`%12`, L40S, dali). Expect ~370
  GPU-h, ~31 h of running time at 12 at once. All 137 videos verified on scratch (only the two 10-06 videos
  needed copying; PS92 1006's 10-06 copy was incomplete). Pilot outputs copied into its `out/` (skipped).
* First O2 smoke job failed on `import cv2` (lightning-pose does not install OpenCV) -- fixed in COMMANDS step 0.
* **2026-10-16: never let an LP task onto the V100 nodes.** With L40S scarce, the pending tasks were opened to any GPU
  type (`scontrol update JobId=56464442 Gres=gpu:1`). The gpu_quad Tesla V100S nodes (`compute-g-17-[147-152]`) then
  failed 43 tasks in ~30 s each: `CUDA error: no kernel image is available for execution on the device` -- the `lp`
  env's torch has no Volta (sm_70) kernels -- and since a failed task frees its node at once, the queue drained onto
  them. RTX 8000, A40, A100 (incl. MIG) and L40S work. Fix: hold, `ExcNodeList=compute-g-16-[254-255],compute-g-17-[147-152]`,
  release, resubmit only the failed ids (`sbatch --array=<ids>%12 --exclude=... --gres=gpu:1 -t 6:00:00 run_pose.sbatch`;
  the runner skips videos with a done.json). `o2_inference.exclude_nodes` now carries those nodes for every new bundle.
  Symptom to watch: many tasks `CG` / `FAILED` under a minute -> `sacct ... --format=JobID,NodeList,Elapsed`.
  Root cause check: `module load conda/miniforge3/24.11.3-0; conda activate lp; python -c "import torch;
  print(torch.cuda.get_arch_list())"` -> `sm_75 ... sm_120`, no `sm_70`. Re-check after any rebuild of the env.
* **Coverage check (run after any failure / resubmit, and at the end)** -- by OUTPUTS, not sacct states (a task can end
  FAILED, NODE_FAIL, TIMEOUT, CANCELLED ...):
  ```
  cd /n/scratch/users/p/ps150/o2/<bundle>
  awk -F'	' 'NR>1{sub(/\.avi$/,"",$4); print $1, "out/"$2"_"$3"/"$4"_done.json"}' tasks.tsv     | while read id f; do [ -f "$f" ] || echo $id; done | sort -n > /tmp/notdone
  squeue -u ps150 -r -h -t PD,R,CF -o %i | sed 's/.*_//' | sort -n > /tmp/queued
  comm -23 <(sort /tmp/notdone) <(sort /tmp/queued) | sort -n | paste -sd,   # UNCOVERED: resubmit these ids
  sort /tmp/queued | uniq -d | paste -sd,                                     # queued twice: scancel one array
  ```
  Resubmit with `sbatch --array=<ids>%12 --exclude=<exclude_nodes> --gres=gpu:1 -t 6:00:00 run_pose.sbatch` (flags
  override the file's #SBATCH lines). Submit it ONCE -- a second identical sbatch queues every id twice (56481741, 10-16).
* **Time limit vs GPU type (10-16).** LP is ~210 fps on L40S but ~105 fps on RTX 8000, and NOT resumable inside a video.
  At 6 h, the 8 longest videos (> 2.27 M frames: tasks 12, 13, 16, 17, 44-47) time out on an RTX 8000 and lose
  everything. Users cannot RAISE a time limit with scontrol (only lower it), so: scancel the at-risk pending ids and
  resubmit them with `-t 10:00:00` (done for 44-47 and 117, which needs ~5.5-6 h). `o2_inference.lp.time` is now 10 h.
  Per-video RTX 8000 estimate = frames / 105 / 3600 h (frames = the camera CSV's row count).
* **Stuck `(JobArrayTaskLimit)` (10-16).** After several `scontrol update` edits (Gres, TimeLimit, throttle) array
  56464442's pending tasks sat at `JobArrayTaskLimit` with ZERO of its tasks running, although `ArrayTaskThrottle` read
  24 -- a corrupted throttle state, not the throttle working. Fix: `scontrol update JobId=<id> ArrayTaskThrottle=0`
  (0 = no limit); the reason cleared to `(None)` and then the normal queue. Fallback if that fails: `scancel <id>` (only
  pending tasks remain) and resubmit those ids with the usual flags (finished videos are skipped).
* **Fair share** collapses after a day of ~24 GPUs (0.43 -> 0.022 on 10-16, recovering to 0.068 by evening): expect
  long `(Priority)` waits afterwards. Check with `sshare -U -l` / `sprio -u <user>`.
* **Video one frame shorter than its camera CSV** (7 of the first 49: PS92 0607, 0805, 0810, 0812 part 2, 0911, PS93
  0606, PS94 0820): Bonsai logged a timestamp for a last frame that never reached the .avi. Harmless -- O2 predicts
  every frame in the video, video frame k = CSV row k. Compare `done.json` n_frames with the VIDEO's frame count.

## Status 2026-10-06

Task 1 (PS93 0814) finished: 2,209,351 frames in 7.0 h, collected (575 trials). Against the local round-3 run on
the 239,564 shared frames: median 0.2-0.4 px, p95 0.7-1.5 px (decode / batch noise) -- O2 reproduces the desktop.
All 288 cam1 + cam4 session videos copied to `o2_videos/<date>/<animal>/` with `rsync -ah` (the include filter
also took the 6 calibration-folder videos; harmless).

## Status 2026-10-05

`r3_PS93_20261002` (round-3 cam4 model, 3 PS93 videos): smoke test passed; array task 1 (0814) submitted as job
55345148 and running. Tasks 2-3 need their videos copied first. This is a PIPELINE test — the production run
waits for the next cam4 model (labelling round 4).

## Next on O2

* Lightning Pose **training** there (the 4-5-model ensembles the multi-camera EKS needs): a new `lp` env
  (lightning-pose 2.4.2 + NVIDIA DALI, which runs on O2's Linux nodes), and a training bundle analogous to the
  pose bundle. Test with one short training run before the real ensemble.
* Try `gpu_quad` access; if granted, set `o2_inference.partition: gpu_quad`.

## ViT (DINOv2) backbones need transformers 5.17.0 (2026-10-18)

Training `vitb_dinov2` with the O2 `lp` env's newer transformers died at the first validation batch (37 s) with
`TypeError: cannot unpack non-iterable int object` in `modeling_dinov2.interpolate_pos_encoding` -- Lightning Pose
2.4.2 sets `patch_embeddings.patch_size` to an int. Fix (interactive node, not the login node):
`conda activate lp && pip install "transformers==5.17.0"`, then resubmit only that array task
(`sbatch --array=1 train.sbatch` after removing the failed model dir). ResNet / temporal-context models are unaffected.
A batch job's `.err` shows no epoch progress bar; read `models/<name>/train_status.json` (completed / total epochs).
