# FaceRhythm — handoff (2026-10-22, ~09:30)

Supersedes the CURRENT STATE block of [`STATUS_2026-10-21_FACE_RHYTHM_O2_HANDOFF.md`](STATUS_2026-10-21_FACE_RHYTHM_O2_HANDOFF.md)
(still the record of the parameter choice, the O2 runner, pitfalls and the PS92 preview). HOW everything runs:
[`docs/FACE_RHYTHM_PIPELINE.md`](../FACE_RHYTHM_PIPELINE.md) (stages, files, run order, the "Other cameras" section). WHY:
`docs/FACE_RHYTHM_PLAN.md` + DECISIONS.md (10-21 evening).

Paths: `<dlc>` = `M:\MICROSCOPE\Priya\DeepLabCut\Widefield`; `<bundle>` = `<dlc>\o2\fr_cam4_20261021` (share) /
`/n/scratch/users/p/ps150/o2/fr_cam4_20261021` (O2).

## 1. cam4 production array (O2 57168778) — outputs arriving

* Priya copies batches from O2 (`rsync -ah --exclude '*.part' <scratch>/out/ <share>/out/`). By ~09:30: PS92 all 28 tasks,
  PS93 ~23 of 26, PS94 arriving. ~19 GB per task (trials 9-12, session 6-9, points ~1); ~2 TB for all 108; the link
  runs ~240 MB/s. Sessions take ~4.3 h on O2 (12 segments = 3 rounds of ~85 min), as estimated.
* **Every task checked so far is clean**: whole video covered, 25 Hz bins = frames // 10, 0 % NaN bins, 40 ms DAQ steps,
  trials 100 % filled, violations 0. Tracking QC (PS92 all epochs): interior folds 0.03-0.06 % of sampled bins in every
  epoch; rim folds 28-65 % are near-flat SLIVER triangles at the mesh edge flipping with sub-pixel jitter (area 0-8 px^2,
  min angle 0-2 deg; rim points move like interior points; factors carry the rim's share of points, 13-23 %) -> no
  effect on the analysis; median displacement 1.33 px pre vs 1.11-1.20 post.
* **QC is now `python scripts/o2_face_rhythm.py qc --name fr_cam4_20261021 [--jobs 3]`** (10-22 rewrite): integrity from
  npy HEADERS only (never loads the spectrograms), tracking QC from CONTIGUOUS block reads, tasks in parallel, per-task
  results CACHED in `<bundle>/qc_cache/` (`result_cache`, keyed on the four files' size + mtime + `QC_VERSION`) -> a rerun
  after a copy reads only the new tasks; skips tasks still being copied. 58 tasks cached at ~09:15 (stopped there to
  free the share for the pilot; rerun when the copy is done). Output: `<dlc>/face_rhythm/cam4/o2_fr_cam4_20261021_qc.png`
  + `data/` CSV with a `problems` column. (The 10-21/22 scratch check loaded each session's 6-9 GB spectrogram for its
  shape -- ~400 GB of reads; retired.)

## 2. PILOT factor set `pilot_pre_PS92_PS93` (running LOCALLY; Priya: no O2 fair share)

`<bundle>/factors/pilot_pre_PS92_PS93/`, role = **pilot** (stamped in plan.json + master.npz; never the cohort master;
`plan` refuses a restricted set without `--role pilot`). Plan: `face_rhythm_factors.py plan --set pilot_pre_PS92_PS93
--animals PS92 PS93 --with-outputs --role pilot --trials-per-animal 30 --session-trials 30 --inits 3 --n-iter 150
--ranks 10` (smaller than the O2 design: the box had 2-5 GB free; rank 10 chosen by Priya to match the preview and the
paper -- the rank sweep belongs to master_pre_v1). 51 tasks (PS92 28 + PS93 23).

* **Fit (rank 10)**: EVR 0.807 (uncentred), stability across 3 starts 0.87, 33 min. Leave-one-animal-out (two animals:
  fit ONE, project the other -- weak): held-out err 0.54 / 0.52 vs in-fit 0.43 / 0.45 (EVR ~0.71 vs ~0.81), factor
  similarity 0.77 (PS92 out) / 0.90 (PS93 out). **Animal-to-animal transfer costs ~0.10 EVR, a pre day ~0.006** ->
  the master set needs all four animals.
* **Leave-one-pre-DAY-out (`loso`, 15 days)** -- the noise floor: each day's SESSION rows (the rows `fitsession` scores
  every session on) projected on a fit without that day vs the full fit. 7 / 15 done at 09:19: held-out EVR 0.792-0.815
  vs in-fit 0.792-0.816 (gap 0.000-0.009), similarity to the full fit 0.87-1.00. Report: `report/loso.csv`,
  `loso_by_rank.csv`, rank_sweep panel 4.
* **Scoring rule (Priya 10-22)**: post-stroke sessions are projected on the FULL pre-stroke fit; each PRE-STROKE session on
  the fit that LEFT ITS DAY OUT (`loso/rank10_<day>.npz`, components matched to the full fit's order, cosines in
  `basis_cos`), so pre-stroke numbers are never in-sample. `proj/` and `own/` files record `basis` = full / loso;
  `project` refuses a pre task whose day basis is missing. (Animal scale and the dense-position PCA still use all pre
  days -- a scalar and a position basis.)
* **Running** (two local drivers, logs + copies of the drivers in the set's `logs/`): `run_pilot_local.py` (build -> fit
  -> loso 1..15 -> select -> project all -> fitsession all, 1 start) and `run_pilot_project_early.py` (froze rank 10 at
  09:04 and projects post tasks now, pre tasks as their loso basis lands; the encoding window's keys first). Projected by
  09:18: PS92_0820, PS92_0818 (acute, full basis; trial err median 0.47). ~13-14 min per task while the share is busy.
  Both drivers are resumable (finished outputs are skipped). Then: rerun `report`, `face_rhythm_readouts.py` on the set.
* **Figures (queued, start automatically)**: when PS92_0806 / 0807 (pre) are projected: `report --absolute --face
  <dlc>/face_rhythm/cam4/master_face.png` -> `report/factor_gallery_abs.png` (spatial map coloured by movement direction,
  blue lateral / red vertical, size = weight; spectrum; pre-stroke cue-aligned traces by position, L blue / R red /
  centre grey, close solid / far dashed) + `factor_epochs_abs.png` (cue-aligned by epoch, per animal); and CLIPS, 8
  examples per factor, engaged trials cue -2..+4 s, PS92_0806 -> `<dlc>/face_rhythm/cam4/factor_clips/pilot_pre_PS92_PS93/`.
  **NO names carried over** (Priya: factor numbers may shift between fits; she names them again from spatial,
  frequency and temporal extent). `report --match <set>` exists as a hint, off by default.

**Encoding window (sabatinilab-d4)** knows: use `--fr pilot_pre_PS92_PS93` (folder `_fr-pilot_pre_PS92_PS93`), pilot
only, pre = loso basis; its watcher waits for proj/PS92_0812 (v1+v2), PS92_0820, PS93_0812. It also has the PS92 rank-10
PREVIEW as a set now: `<bundle>/factors/preview_r10_PS92/` (master, labels, proj for PS92_0806 / 0812 v1+v2; in-sample,
no points basis) -> `--fr preview_r10_PS92`.

## 3. Other cameras (cam1 / cam2 / cam3) — prepped, waiting on Priya's clicks

Code (10-22): `wfield_local/face_rhythm_cams.py` + `--cam` on prep / register / warp / O2 bundle; pipeline doc section
"Other cameras". Done: rest faces for all 119 sessions x 3 cameras (cam4-LP rest moments mapped through each camera's
template; ~8 min per camera) and `register --stage templates` (master PS94 in every view; click images
`<dlc>/face_rhythm/<cam>/register_inputs/click_<s>.png`). **Next (Priya)**: `notebooks/face_rhythm_click_landmarks.ipynb`
(fr JupyterLab, port 8899 already running) -> `register --cam <cam>` -> ROI on the master (`face_rhythm_draw_roi.ipynb`,
`CAM`) -> `warp_rois to-sessions --cam` -> one-session O2 pilot (`o2_face_rhythm.py bundle --cam <cam>`). Side views show
the forepaws (a grooming route later). Suggested order cam2 + cam3, then cam1.

## 4. Efficiency changes (10-22, Priya: "parallelization, cache/sidecars")

* `qc`: see 1. `project`: ONE pass over the 6-9 GB session spectrogram (window power means ride on the projection's read,
  `_scaled(on_chunk=)` + `ClassMeans`) and the 25 Hz points STREAMED in blocks (`points_stream`; peak ~200 MB instead of
  ~5 GB per task); a test pins both to the in-memory path. `loso` = its own array (rank x day) instead of serial in
  `fit`. Per-task stages fan out two at a time locally.
* Still I/O-bound: a projection reads ~20 GB from the share; with the O2 copy, the encoding renders and several of our
  readers at once it ran ~13 min per task. Keep our concurrent share readers to ~3.

## 5. Backups (all on the share under MICROSCOPE/Priya)

Factor sets, frames, registration, click images, qc_cache, proj / loso / own, reports and clips are written to the share
directly. Copied from local scratch 10-22: the preview's fit files (`factors/preview_r10_PS92/`: rank10.npz,
scales.json, fit_index.npz, report/), the master face image (`<dlc>/face_rhythm/cam4/master_face.png`), the pilot
drivers + logs (`factors/pilot_pre_PS92_PS93/logs/`; logs are a snapshot -- recopy at the end). Not copied (reproducible
/ test only): the preview fit_tensor (from its plan.json) and the rank-6 real-data dry-run set.

## 6. Next, in order

1. Pilot finishes (loso 8-15, all projections, fitsession) -> `report` (+ `--absolute`) -> Priya reviews gallery + clips
   and names the factors -> `labels.json` -> `face_rhythm_readouts.py --set pilot_pre_PS92_PS93` (EVR gap vs the loso
   noise floor, epochs, asymmetry, movement maps). PS93 clips once its pre projections exist.
2. Array done -> final copy -> `qc` (cached) -> `collect` -> **master_pre_v1 on all four animals** (`plan` without
   `--animals`; rank sweep 6-16 + loso; O2 if fair share allows, else locally at the pilot's size) -> select -> project /
   fitsession -> readouts -> naming -> encoding with `--fr master_pre_v1` (re-run anything built on the pilot).
3. Planned checks (unchanged): lick factors by EXECUTED direction (LP per-lick angle); 5 vs 25 Hz sufficiency; lick
   rhythm vs time with LP lick rate; nose_up_prep vs attempt classes; left early / late asymmetry in the master set.
4. Other cameras: clicks -> registration -> ROIs -> pilots.

## Pitfalls (new)

* **Pushes race**: the pre-push suite takes ~18 min and other windows push meanwhile -> `cannot lock ref`; fetch, rebase,
  push again. Commit your own edits first (a dirty tree stops the rebase).
* **Do not check outputs by loading arrays** -- `_npz_shapes` / `npz_memmap`; a strided memmap over the share is slow
  (read contiguous blocks).
* Background share listings (`ls` of out/) can hang for minutes while copies and readers are busy.
