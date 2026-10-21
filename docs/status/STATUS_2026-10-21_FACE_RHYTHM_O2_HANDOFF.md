# FaceRhythm on O2 — handoff (2026-10-21)

Plan: [`docs/FACE_RHYTHM_PLAN.md`](../FACE_RHYTHM_PLAN.md) (steps 1-7; status section at its end). This doc covers step 4
(master ROI, drawing tool) and step 5 (the O2 bundle), as left at the end of 10-21 morning. Registration (step 3) and
the ROI warp are the pose/3-D window's (`scripts/face_rhythm_register.py`, `scripts/face_rhythm_warp_rois.py`,
`STATUS_2026-10-21_POSE_CAM1_JAW_HANDOFF.md`).

## Where things are

| what | where |
|---|---|
| bundle / collect script | `scripts/o2_face_rhythm.py` (commits 66cf985, cf880bb, ab86352, 6a2bfc8) |
| ROI drawing notebook | `notebooks/face_rhythm_draw_roi.ipynb` (`fr` env; JupyterLab, not VS Code -- see pitfalls) |
| bundle on the share | `M:\MICROSCOPE\Priya\DeepLabCut\Widefield\o2\fr_cam4_20261021\` (108 tasks) |
| bundle on O2 | `/n/scratch/users/p/ps150/o2/fr_cam4_20261021/` (an EARLIER build: re-rsync before submitting, see below) |
| ROIs the runner reads | `<dlc>/face_rhythm/cam4/rois/<animal>_<YYYYMMDD>/ROIs.h5` -> rsynced to `<bundle>/rois/` |
| outputs | `<bundle>/out/<key>_{trials,session,points}.npz` + `_summary.json`; stay in the bundle's `out/` (not copied again) |
| O2 env | `fr` in `/home/ps150/.conda/envs/fr`: face-rhythm 0.3.3, torch 2.14.1+cpu (installed 10-21) |

## What the runner does (one task = session x cam4 video)

* **Tracking** with FR's own per-frame step (LK + violation freezing/rewind + mesh + relaxation, called on a frame range;
  `PointTracker` itself allocates the whole video). Values = the plan's "paper" set: mesh_rigidity 0.1, 24 neighbours,
  relaxation 0.01, winSize 40, maxLevel 3, threshold 300 px; LK criteria (3, 3, 0.03), halting 10/10, CLAHE default,
  decord -- as the 10-21 pilot (`fr_pilot.py`). **Not validated on this rig** (see open items).
* **10-min segments**, each restarting at the home positions, tracked from 10 s before (warm-up, discarded) to 520
  frames after (VQT margin). Independent, so **4 segment workers x 2 threads per task** (spawn pool); each writes only
  its own bins into shared float16 memmaps in `work/<key>/`; the parent records finished segments in `progress.json`
  -> a resubmitted task resumes. Parallel output is bit-identical to single-process (6000-frame smoke test).
* **VQT** (Fs 250, 1-60 Hz, 36 bins, Q 4 -> 20, ds 10, normalisation 0.95, 1/f^0.5) per 10 000-frame chunk with
  `padding='same'` so 25 Hz bin j = frames [10j, 10j+10) exactly.
* **Outputs (float16)**: `trials` spec (trial, time, xy, point, freq) on the DAQ grid cue -2 .. +4 s / 40 ms (from the
  v36 cam4 `windows.npz` `ix_src_frame` / `ix_t_daq_ms`; linear between the two nearest bins, NaN outside the window);
  `session` spec at 5 Hz (mean of 5 bins) + `t_daq_s`; `points` disp at 25 Hz (minus `home`) + `viol_frac`; summary json
  with FR's violation fraction. ~24-32 GB per session at ~2000 points; ~2.6-3.4 TB for the cohort.
* **Trials = engaged only** (`orofacial_cohort.engagement_keep`, the cohort tables' reference gate; keeps 90.2 %;
  `--all-trials`). **June left out** (`--include-june`): registration fell back to a similarity fit after the camera
  change and gets no warped ROIs. **PS92_20260812** (two cam4 videos) = two tasks `_v1` / `_v2` (`ix_video`).
* sbatch (from the bundle's estimates): `short`, `-c 8`, `--mem 36G`, `-t 6 h`, `%24`. Estimates are the desktop
  pilot's rates (100 fps / process); **calibrate on the first O2 task** (`sacct -j <id> --format=MaxRSS,Elapsed`),
  then lower `--mem` / `-t` in `run_fr.sbatch` so tasks schedule sooner.

## State at handoff and next steps (in order)

1. **Master face is being rebuilt** (pose/3-D window, Priya's go-ahead): the registration had picked post-stroke
   templates for all four animals and PS93_20260821 (ACUTE, the right-orofacial-deficit animal) as master. Asked:
   August pre-stroke templates, master among PS92 / PS94 / PS95, Priya's saved polygons (`rois/PS93_20260821/ROIs.h5`,
   09:45, 2 007 points; the registered draft kept as `ROIs_20261021_093559.h5`) carried onto the new master, existing
   ROIs moved aside rather than overwritten.
2. **Priya re-checks the master ROI** in the notebook: `SESSION` = new master, `EDIT = True`; overlay = every
   session's LP spout tips / nose / jaw in master px (`register/master_overlay.npz`). Face reaches the eyes; mask ~25
   px outside the face on top and sides, bottom just above the highest close-spout (orange) dots.
3. **Warp** to sessions (pose window: `face_rhythm_warp_rois.py to-sessions`).
4. **Re-run the bundle** if anything about sessions changed (`python scripts/o2_face_rhythm.py bundle --name
   fr_cam4_20261021 --user ps150`), then on the transfer node rsync the bundle, the videos (`--files-from videos.txt`)
   and `rois/`. **Never rsync a new bundle while tasks run**: spawned workers re-read `tasks.tsv`, so a changed task
   order would hand a worker the wrong session.
5. **Open decision -- tracking-parameter test before the full array** (proposed, not yet answered): pilot session,
   first ~30 000 frames, three sets (the plan's paper values; the old 2pRAM cohort run in
   `MICROSCOPE\Priya\FaceRhythm\fr_run_20250520_latent_bundle` -- rigidity 0.002 / 90 neighbours / relaxation 0.002 /
   winSize 50 / maxLevel 5 / threshold 120; FR's `params_pipeline_basic.json`), scored against LP nose / jaw in the
   trial windows + violation fraction + rest drift. Needs a params override in the runner + a scoring script.
6. **Pilot one task** (current task numbers: PS93_20260821 = 39, PS94_20260810 = 58 -- re-check after any rebuild:
   `awk -F'\t' '$1=="<key>" {print NR-1}' tasks.tsv`), read its per-segment log lines (fps per worker, violations,
   worker peak RAM), then `sbatch --array=<rest>%24`.
7. **Copy back** `out/` to the share bundle; `python scripts/o2_face_rhythm.py collect --name fr_cam4_20261021` ->
   `<dlc>/face_rhythm/cam4/o2_fr_cam4_20261021_index.csv`.

## Pitfalls met today

* **O2 /home quota**: `pip install face-rhythm` pulled CUDA torch (~4 GB of nvidia wheels) and pip's cache had grown to
  15 GB -> "No space left on device". Fix (now in COMMANDS.md): CPU torch first from
  `https://download.pytorch.org/whl/cpu`, `PIP_CACHE_DIR` / `TMPDIR` on scratch, `pip cache purge`. `quota` is retired:
  `quota-v2`. There is also an older `face-rhythm` env in /home (unknown version) -- the sbatch activates `fr`.
* **FR's ROI GUI** is a Plotly freehand widget: VS Code's notebook renderer fails ("Cannot read properties of undefined
  (reading 'ipywidgetsKernel')") and it cannot place vertices by click. The notebook uses matplotlib `PolygonSelector`
  (`%matplotlib widget`) in JupyterLab instead: click vertices, drag to edit, no vertex insertion (paste a `refine()`
  resampler if needed). Its first overlay showed the mask in near-white instead of red, and it seeded points from the
  face polygon alone (points outside the mask) -- both fixed: points = face AND mask.
* **The 08/06 sessions' trial ids** disagree with the behaviour tables: dead `spout_bit1` -> behaviour from the GUI
  log, 1-based; `engagement_keep` matches by order (99.8-100 % position agreement). Not an error.
* **Split sessions** carry `ix_video` in `windows.npz` with `ix_src_frame` per video; `face_rhythm_prep.py` reads
  `vids[-1]` -- wrong for PS92_20260812's first video (that file is another window's; not changed here).
* **Templates must be pre-stroke** (see step 1) -- the registration picked by medoid score alone.
* The trial-window output is the big one (n_trials x 151 x 2 x points x 36 x 2 B; 720 trials -> ~30 GB). Fewer points
  (spacing 10-11 -> ~1350-1650) or x+y combined would cut it.
* A JupyterLab server was left running for the notebook (port 8899, started from the `fr` env).
