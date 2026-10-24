# FaceRhythm — handoff (2026-10-22, ~09:30; UPDATES ~13:15 and ~12:10 first)

**Findings so far: [`docs/FACE_RHYTHM_FINDINGS.md`](../FACE_RHYTHM_FINDINGS.md)** (living, status-tagged). Decisions:
DECISIONS.md 2026-10-22 "FaceRhythm: pilot factor set, pre-stroke scored held-out, naming per fit, figures + methods".

## UPDATE 10-23 (read first)

* **AR-HMM (Hillary, github.com/htong1/Priya_ARHMM2026, private; clone in scratch `ext/`) assessed 10-23**: frozen
  pre-stroke sticky AR-HMM (jax-moseq, K 24, 12.5 Hz) on FR consensus + DLC tongue / jaw + treadmill, PS46-50.
  DECIDED (Priya 10-23): LOW PRIORITY, not ported for imaging. FR factors already give continuous regressors for
  specific behaviours (whisk f7, nose / sniff f2 / f3 / f8, prep jaw f6, bout initiation f1, lick factors by side), and
  the encoder already separates failed vs successful licks (LP lick-onset kernel for every lick + DAQ contact kernel
  for hits + executed angle / reach modulation; trial hit / incomplete classes in `trial_attempts`). What an AR-HMM
  would add is narrow: discrete onsets / bouts (testable with onset kernels on thresholded factors), state
  combinations (interaction terms), and a global per-session recovery score (frozen-model log-lik; the FR EVR gap,
  FINDINGS 6, already covers the vocabulary half). Revisit only for that score or a state interaction linear
  regressors miss. Its headline ("licks and misses") cross-checked behaviourally: FINDINGS 5c. Open, for the encoding
  window if wanted: split the lick-onset kernel by outcome (hit / miss) to test a PRE-lick (preparation) difference.

* **~21:05 10-23: master_pre_v1 FIT + `proj/` BACK ON THE SHARE** (Priya rsync): fit 131 min on O2, EVR .792 (err
  .4564), 8 starts on two solutions (5 at .4564, 3 at .4571; per-factor min across starts .66-.72 for f4 / f5 / f10 =
  the factors that differ in the .4571 solution, median >= .99 for all), stability .998; leave-one-animal-out held-out
  err .468-.504 vs in-fit .452-.464, factor similarity .957-.984; split half .990 (per factor min .969). `proj/`
  108 / 108 (`basis` full_in_sample 32 / full 76; only gaps PS92_0812 v1 73% lossy + no speed, PS92_0908 5% lossy).
  d4 TOLD (waits for labels.json + a master grouping; box full). Loso 60 / 62 files copied mid-run; project_loso +
  fitsession still to come -> rsync again. NOW: master report (`--absolute` and share, `--match` the pilot) ->
  clips + validation -> Priya names.
* **O2 chain SUBMITTED 10-23 ~19:00** (`run_chain.sh 10`): build DONE (27448 rows, 6862 per animal, 10.1 GB);
  fit 57382949_10 running; select 57382951, split half 57382950, project 57382952 (-> encoder), loso 57382953,
  project_loso 57382954, fitsession 57382955 pending on dependencies. Bring back: transfer node `rsync -ah --exclude
  fit_tensor.npy $B/factors/master_pre_v1/ $S/factors/master_pre_v1/` (after project, then after fitsession). Then: tell
  d4 `proj/` is ready (full frozen basis, `basis` = full / full_in_sample); report + clips + validation; Priya names.
* Previous state (10-23 ~18:30): master_pre_v1 RE-PLANNED with equal days (DECISIONS 10-23 evening #1-3), rank 10
  only, and set up for O2: Priya runs `factors/master_pre_v1/COMMANDS.md` (transfer rsync -> `run_chain.sh 10`). Local
  driver stopped. Encoder gets `proj/` (full frozen basis, all 108) as soon as the project array finishes.
* Previous state (10-23 17:10): master sweep DONE (8 / 10 / 12), **rank 10 chosen** (DECISIONS 10-23 evening); local
  driver running `run_master_local.py rank 10` (log `scratchpad/master_local.log`): refit + LOAO -> splithalf -> loso 31
  days -> select -> project 108 -> fitsession -> report. Resumable (skips finished outputs). Then: report + clips +
  validation on the master; Priya names the factors; redo FINDINGS 5 / 5b / 6 on four animals.
* Earlier state: master_pre_v1 PLANNED (counts in PIPELINE 'Time windows and the fit
  sample'; still = canonical REST, run = canonical running bouts, equal pre-cue rows per animal) and SWEEP RUNNING. Plan was (`--trials-per-animal 120 --trial-ds 5
  --between-frac 0.556 --run-share 0.5 --arrival --inits 8 --ranks 8 10 12`; it reads treadmill speed for all 108
  sessions, cached) -> check the drawn counts per animal (printed; ~3600 trial + ~1500 pre-cue + ~2000 run / still rows)
  -> `run_master_local.py sweep` (build + ranks 8 / 10 / 12) -> Priya picks the rank -> `run_master_local.py rank R`
  (fit with leave-one-animal-out -> `splithalf` -> loso -> select -> project 108 -> fitsession).
  DONE 10-23: pilot PS92_0908 + PS92_0812_v1 re-projected + fitsession with true times (old files in
  `proj|own/_superseded_linear_timing/`; plan.json session_rows of the two redrawn, old plan in
  `logs/plan_before_timing_fix_20261023.json`); d4 told (reruns PS92_0908 in run A). Pilot readouts / validation /
  lick_epochs NOT yet re-rendered with them (small effect: 2 of 51 tasks). Decisions: DECISIONS 10-23 #1-7.

* **~later 10-23: master_pre_v1 RE-PLANNED** (sweep stopped before any fit): windows + balance in PIPELINE "Time windows
  and the fit sample" (DECISIONS 10-23 #4-6): true bin times (`o2_face_rhythm.py timing`, run for all 108: only
  PS92_0908 lossy / mis-stamped; PS92_0812 v1 partly outside its DAQ segment), pre-cue rows (spout arrival U cue -4..-2
  s), between rows half run / half still, 20 trials per position per animal. Then restart `run_master_local.py sweep`.
* Pilot: PS92_0908 + PS92_0812_v1 to re-project with true times (`scratchpad/fix_pilot_timing.py`; old files ->
  `_superseded_linear_timing/`); then tell d4 (it reruns PS92_0908 in run A).

* **cam4 array COMPLETE**: 108 / 108 on the share, qc 108 / 108 clean (`o2_face_rhythm.py qc`), `collect` done
  (`face_rhythm/cam4/o2_fr_cam4_20261021_index.csv`).
* **master_pre_v1 RUNNING LOCALLY** (Priya: "local, trimmed"): plan = 4 animals x 120 trials (20 per position, fit rows
  every 160 ms, `--trial-ds 4`) + 1110 between bins, 8 starts, ranks 8 / 10 / 12, loso jobs for 31 pre days.
  Driver `factors/master_pre_v1/logs/run_master_local.py` (scratch copy runs; log `master_local.log` in the scratchpad):
  `sweep` = build + fit-only ranks + report -> STOP for Priya's rank -> `rank R` = refit R with leave-one-animal-out,
  loso at R, select, project 108, fitsession. Waits for >= 20 GB free RAM before heavy stages (Priya: may ask other
  windows for RAM). Old 10-21 O2 plan kept as `logs/plan_20261021_o2_design.json`.
* **Multi-view factor sets prepped** (DECISIONS 10-23 #3; PIPELINE "Other cameras" step 7): `plan --views cam1=... `;
  waits on cam1 / cam2 / cam3 runs (Priya's landmark clicks -> register -> ROIs -> O2).
* Pilot labels written 10-22 (FINDINGS 4); encoding (d4) uses grouping `pilot_v1` (fr_groups, pilot set only).

## UPDATE 10-22 ~14:00

* **Rank sweep complete**: 14 EVR 0.827 / stability 0.89; 16 0.835 / 0.92 (report re-rendered). No elbow; rank 12 the
  most reproducible. Rank 12 vs 10 (FINDINGS 2): lick factors unchanged; the extra two split SLOW movement (f8 by
  frequency, f6 / f2). Pilot stays rank 10.
* **Per-factor stability** in `fit` (Priya: "ok"): every start's A / B saved (`A_starts`, `B_starts`), per-factor
  matched similarity in rank*.json + `report/stability_per_factor.csv` + dots on the rank_sweep stability panel;
  `--inits` default 5 -> 8 for master_pre_v1. Pilot rank files predate it (no per-factor values).
* Projection tasks 26 / 28 logged rc 1 (the temp-file collision): outputs verified valid.
* ~14:20: fitsession 51 / 51 done; readouts run (`<dlc>/face_rhythm/cam4/readouts/pilot_pre_PS92_PS93/`): EVR gap
  acute above every pre session in both animals, PS93 back to pre by subacute, PS92 slightly above chronically
  (FINDINGS 6). Local pilot run COMPLETE; nothing running.
* ~14:45: per-lick by epoch DONE (`scripts/face_rhythm_lick_epochs.py`, FINDINGS 5b): right-lick COUNT recovers to pre
  chronically in both animals, f9 PER right lick stays 0.4-0.6 -> the contralesional deficit is in how licks are made.
* ~15:20: pilot `labels.json` WRITTEN (Priya; FINDINGS 4; f5 = lick_fast_early_Lbias, kept in the encoder's left
  group). Encoder columns are now `fr_<name>` (d4 told; fr_groups unaffected). Pilot figures re-rendered with names.
  Scratchpad test folders deleted (Priya OK; preview_r10 clips backed up to factors/preview_r10_PS92/clips first).
  QC 95 / 108 clean. Next: last 13 PS95 tasks -> final qc + collect -> master_pre_v1 (8 starts, per-factor stability) when PS94 / PS95 are in.

## UPDATE 10-22 ~13:15

* **Pilot projections DONE: 51 / 51** verified (pre = `loso` basis, post = `full`); encoding window (d4) told. The early
  and main drivers collided on 4 tasks (shared temp file; results intact) -> `_atomic_savez` now uses a per-process
  temp file. Still running locally: `fitsession` (main driver, 1 start, 2 at a time) and the rank sweep (6, 8, 10, 12
  done; 14, 16 running: rank 12 EVR 0.819, stability 0.997 vs rank 10 0.807 / 0.87).
* **Figures restyled** (Priya: bigger fonts, no overlaps, methods documented): `wfield_local/face_rhythm_figures.py`
  (style, constrained layout, legends outside, short titles; `save()` writes each figure's METHODS into its folder's
  README.md). Applied to the factor report (gallery, epochs, rank sweep), validation, readouts and the QC figure.
* **Validation re-cuts** (Priya: "we can do 1-3"): `factor_vs_licks.png` now has direction x early / late, LICK NUMBER
  IN BOUT (1st / 2nd-3rd / 4th+), SPOUT POSITION, and lick frequency; running onsets in `factor_vs_running.png`. Read
  the identities and the first epoch changes in FINDINGS sections 4-5: f1 = bout INITIATION (not "right / central
  later"); f9 right / contralesional licking DOWN in both animals and still down chronically; f6 preparatory UP in
  both.
* **Next**: when fitsession finishes -> `report` (+ `--absolute`) -> `face_rhythm_readouts.py --set pilot_pre_PS92_PS93`
  (EVR gap vs the loso noise floor, epoch tables, asymmetry, movement maps) -> Priya confirms names -> labels.json;
  per-lick expression by epoch (does f9 drop because of fewer licks or different licks?); master_pre_v1 when PS94 / PS95
  are in.

## UPDATE 10-22 ~12:10 (sections 1-6 below are the 09:30 state)

* **Array outputs on the share: 74 tasks (PS92 28, PS93 26, PS94 20), integrity 74 / 74 ok**, 0 NaN bins, 0 violations,
  interior folds 0-0.15 % in every animal x epoch. Median displacement (px, pre -> acute -> subacute -> chronic): PS92
  1.33 -> 1.11 -> 1.16 -> 1.18; PS93 1.19 -> **0.80** -> 1.21 -> 1.36 (the deficit animal moves least acutely); PS94
  **2.06** -> 1.33 -> 2.35 -> 2.63 (~2x the others: a more mobile animal or a larger face in px; the per-animal scale
  absorbs a constant). Priya may lift the array throttle (`scontrol update JobId=57168778 ArrayTaskThrottle=24`; total
  CPU-h and fair-share charge are the same either way).
* **Pilot `pilot_pre_PS92_PS93`**: leave-one-day-out DONE (15 / 15): held-out EVR PS92 median 0.798 (0.788-0.815) vs in
  fit 0.805, gap median 0.006 (max 0.009); PS93 0.805 (0.773-0.813) vs 0.804, gap 0.001 (max 0.004); similarity to the
  full fit 0.84-1.00. **Noise floor: leaving a pre day out costs <= 0.01 EVR** (animal-to-animal transfer ~0.10).
  Projections 44 / 51 at 12:00 (the rest + fitsession continue in the local drivers). PS93 pre now projected too.
* **Gallery** (`report/factor_gallery_abs.png`, `factor_gallery.png`): WHERE (weight on the master face) + HOW (blue
  lateral / red vertical, opacity = weight) + spectrum + pre-stroke cue-aligned traces by position; y units
  "a.u. (weight x D; per-animal scale)" -- the weight is the factor's amplitude in the bin (spatial + spectral columns
  unit-norm) on the spectrogram divided by the animal's one scale; comparable within an animal across sessions, not a
  physical unit. **Epoch figures** (`factor_epochs[_abs].png`): per animal, NORMALISED so each factor's pre-stroke mean
  curve max = 1. Clips: 8 examples per factor, PS92_0806, `<dlc>/face_rhythm/cam4/factor_clips/pilot_pre_PS92_PS93/`.
* **Priya's provisional pilot labels (from the clips; NOT carried from the preview -- factor numbers shift between fits)**:
  f0 left licks later in trials; f1 right / central licks later; f2 sniff / whisk / jaw; f3 vertical nose late after
  the cue / around a late lick attempt or spontaneous lick; f4 close-R / central licks early (faster?); f5 left licks
  early (faster?); f6 weak, jaw (preparatory before a lick?); f7 nose wiggle / sniff / whisk while walking; f8 nose
  wiggle with a few treadmill steps, sometimes around a little licking; f9 right licking early (faster). No labels.json
  written yet.
* **Identity check vs behaviour** (`scripts/face_rhythm_validate.py --set pilot_pre_PS92_PS93`, computation
  `wfield_local/face_rhythm_validation.py`; LP licks_all with executed angle, + = animal's left; canonical treadmill
  speed cached per session) -> `<dlc>/face_rhythm/cam4/validation/pilot_pre_PS92_PS93/` (factor_vs_licks_running.png,
  factor_correlations.png + data/). First run, PS92_0806 + 0807 only: f0 LEFT-selective (left-rate r 0.49, right ~0),
  strongest at SLOW lick rates (5-6 Hz); f1 weakly lick-linked, LATE (t-from-cue r 0.17), centre > R ~ L; f2 sharp peak
  at RUNNING ONSET; f3 peaks ~0.3 s BEFORE early licks + doubles after running onset; f4 CENTRE / RIGHT (r 0.52 / 0.27),
  peak at 6 Hz (slow, sustained), not early-fast; f5 EARLY + FAST (7-8 Hz), all directions with a left bias; f6 ~0.35 s
  BEFORE early licks, all directions (preparatory); f7 strongest RUNNING factor (speed r 0.56 in trials, 0.68 outside
  licks); f8 running-linked (0.32 / 0.38); f9 RIGHT-selective (r 0.49, left ~0), early AND late, peak ~7 Hz. Caveat:
  absolute expression includes overall face movement, so every factor correlates positively with speed outside licks
  (0.16-0.68) -- read speed relatively. **Rerun now that PS92's other pre days and PS93 pre are projected.**
* **Rank sweep (Priya: no leave-one-out -- "what rank is enough" in pre-stroke sessions)**: `fit --rank R --no-loao` on
  the pilot tensor, ranks 6 8 12 14 16 (3 starts) running locally after rank 10; rank 6: EVR 0.767, stability 0.84; rank
  10: 0.807, 0.87. Read `report/rank_sweep.png` / `.csv` (EVR elbow + stability across starts). The cohort rank is
  chosen on master_pre_v1.
* Commits through e633e00 pushed. Scratchpad: ~20 GB of superseded local test outputs (frtest*, preview_r10, dryrun)
  proposed for deletion, awaiting Priya's OK.

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
