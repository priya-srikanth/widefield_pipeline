# Handoff — 2026-10-13: movement-removed ENL trajectory (118 sessions), null/potent closed, cam4 round 4 (DLC + LP, temporal-context LP), camera templates

**START HERE.** Supersedes `STATUS_2026-10-07_MOVEMENT_SUBSPACES_HANDOFF.md` (still the reference for the 10-06/07
movement models, video regressors and the multi-view LP pilot set-up). Reasoning and numbers: `DECISIONS.md`
addenda 2026-10-08 (three), 2026-10-12, 2026-10-13 (two). Analysis desktop; M: = MICROSCOPE.

## 0. RUNNING (2026-10-13 ~11:00)
1. **Round-4 DLC on the two comparison clips** (`session_poses/r3_vs_r4_dlc.py`, dlc env, GPU): PS93 0814 DONE,
   PS93 0821 running; outputs `session_poses/<session>/r4_eval/windows_DLC_r4.csv`; log `session_poses/r3_vs_r4_dlc.log`.
2. **QUEUED behind it: LP round-4 pipeline** (`C:\Users\SabatiniLab\lp_stage\queue_lp_r4.ps1` waits for DLC cmd
   PID 27492, stages the two clips to `lp_stage\r3_vs_r4_clips\`, then runs `C:\Users\SabatiniLab\lp_r4_pipeline.sh`
   in WSL): trains `/root/lp/cam4-2026-10-13/models/r4_heatmap_20261013` (~2 h) and `r4_mhcrnn_20261013` (~4-6 h),
   then predicts round-3 LP, r4 heatmap and r4 mhcrnn on both clips (16-frame DALI chunks + 7 GB guard). Log
   `lp_stage\lp_r4_pipeline.log`; predictions copied to `lp_stage\r3_vs_r4_lp\<model>\`. Expected this evening.
3. NEXT when both finish: five-column figures per session (`python -m scripts.pose_cue_traces compare
   <session_poses dir> --model "DLC r3=..." --model "DLC r4=..." --model "LP r3=..." --model "LP r4=..."
   --model "LP r4 context=..." [--cleaned]`) -> `<session>/r4_eval/cue_traces_compare[_cleaned].png`; plus lick
   detection vs DAQ spout contacts for each model (the measure the executed-angle test depends on). Show Priya.

## 1. Analyses since 10-07 — why, what, result

### A. Movement-null / movement-potent subspaces (Hasnain et al. 2025) — CLOSED as a supplementary control
* **Why:** test whether the pre-cue (ENL) position code lives in activity dimensions that do not drive movement.
* **What changed along the way (all Priya's calls or fixes, DECISIONS 10-08):** spout exclusion = only the logged
  motorised moves (`dock_start->dock`, `trial_start->position`, `trial_windows.spout_motion_spans`, pads set from
  motion energy aligned to the moves); z-score baseline = the docked interval (`trial_windows.docked_spans`; the old
  pre-strobe window sat in the spout's travel); size-matched random-subspace baselines; subspaces fit WITHIN each
  area (fitting globally leaked other areas into MO); primary readout = share of split-half between-position
  variance in each subspace (`null_potent.between_position_cov` / `subspace_fraction`); SVD basis option (global
  100 comps + per-area pixel PCA, Musall's basis) because LocaNMF gives only 8-16 comps per area.
* **Result:** position-related VARIANCE is mostly movement-potent in every area and window (both bases), while
  position is about equally DECODABLE outside those directions; the MO pre-cue "null" result was a small-basis
  artefact. On the 24-session subset the potent share does not change consistently across epochs or agree between
  bases. **Decision (Priya 10-08, "does this add a lot?"): no** -- the movement regression answers the movement
  confound more directly; null/potent stays a supplementary control, no further work.
* **Side finding worth following:** pixel bases decode position better than LocaNMF (ENL all 0.48 vs 0.40, MO 0.47
  vs 0.32; one session). Check against spatial leakage with a non-motor control area before reading anything in.

### B. ENL movement-removed position decoding across recovery — the main result
* **Why:** the goal is how VLS stroke changes cortical motor representations; the confound to rule out is that a
  post-stroke change in the position code is a change in movement that cortex reflects.
* **What:** `scripts/enl_movement_multisession.py` -- pre-cue 2 s lick-free window, running + 4-camera ENL motion
  energy (30 PCs / camera, lags 0 / 0.1 / 0.2 s), every movement model fit inside each CV fold on training trials;
  arms raw / residual / top-k movement-predicted dims projected out / video-only. Validated on PS93 0814 (residual
  0.28 vs 0.26 from the whole-session model).
* **Result (118 curated sessions, >= 60 trials; DECISIONS 10-12 + 10-13 correction):** video + running decode
  position at 0.86-0.97 throughout; after removal position stays above chance in most sessions; across-animal raw /
  residual: pre 0.40 / 0.26, acute 0.45 / 0.31, subacute 0.33 / 0.23, chronic 0.46 / 0.28; **the residual tracks
  the raw code session by session (r 0.92 / 0.91 / 0.80 / 0.77, PS92-95)** including the subacute dip and chronic
  recovery, more tightly than video decodability does (r 0.39 / 0.71 / 0.39 / 0.71). Residual keeps ~40-50 % of the
  above-chance code; PS95 chronic is the exception (0.16). Figure
  `labcams/enl_movement_multisession/enl_movement_trajectory.png`. Caveats: linear removal, video may over-remove
  (lower bound), small acute n.
* **Fixes made to get there:** `trial_bounds` DAQ-cue fallback for sessions whose saved trial table has no DAQ
  clock (dead strobe bit 08-05/06, 08-12 crash, 09-29) or none (PS93 0805 -> recovered-positions CSV); trials
  outside imaging coverage dropped (PS95 0813); **sessions missing any camera are refused** (PS93 0813 had run
  running-only and was in the first 112 -- corrected 10-13).

### C. Camera alignment templates for every analysed session (Priya: "we should have or make them")
* Built with `python -m wfield_local.camera_sync <date> --only <animals>` for the 14 analysed sessions that had none
  (06-02/03/05, 08-05, 09-29): 55 / 56 good (rms 1.15-1.16 ms). **PS92 0605 cam4 FAILS quality** (165k dropped
  frames, 3.3 s residual; saved, flagged). **PS93 0813: no video was recorded** -- experiment error, logged in
  `docs/EXPERIMENT_ERRORS.md`. June 1-5 and 8/5 are outside the curated set (`animals.yaml cross_session_exclude`),
  so those templates matter only for non-curated work.

### D. cam4 labelling round 4 -> DLC round 4
* Labels: Part 1 (cam4 round 4) 128 / 131 done 10-09 (3 blanks, possibly all-occluded); Part 2 cam1 round-1 290 /
  376; Part 3 cam1 round 2 0 / 254 (status from the label files via `dlc_cam1_guide` helpers).
* **DLC round 4** (`wfield_local.dlc_train`, 540 frames / 21 sessions, best snapshot 170; iteration-2 `trainset74`,
  round 3 kept as `trainset71` best-160). Per-part test error is NOT comparable to round 3's (different held-out
  sessions). **Fair test = the 70 frames held out by BOTH** (PS93 0606, PS95 0821, PS94 0921;
  `inference_check_20261012_round4/heldout_both.py`): round 3 -> 4 median error nose 3.2 -> 3.1, jaw 3.2 -> 3.3,
  spout 1.8 -> 1.9, tongue 6.6 -> 6.8 px; jaw errors > 10 px halve (9.8 -> 4.9 %); tongue found 91 -> 85 %, tongue
  false detections on blank frames 8.1 -> 5.4 %. **Round 4 is not measurably better** -- more conservative on the
  tongue. Review videos (6 clips, round 4 and round 3 | round 4 side by side, `inference_check_20261012_round4/
  labeled_clips/`) + `tongue_round3_only_frames.png`: about half of round 4's tongue drops are clear protrusions
  with confidence just under 0.6 (under-confident), half are genuinely ambiguous.
* **Decision:** stop judging models by pixel error; compare on the downstream measure (lick detection vs DAQ
  contacts, angle at peak protrusion) on one pre and one acute session, DLC and LP, rounds 3 and 4 (§0).

### E. Lightning Pose — "temporal loss" is not "temporal inference" (Priya 10-13: "I thought we were")
* Every LP model so far (cam4 round 3, cam1, both multi-view arms) used `losses_to_use: [temporal]` = a TRAINING
  penalty on frame-to-frame jumps in unlabelled video; inference was single-frame (`model_type: heatmap`, or
  `heatmap_multiview_transformer`, which shares across cameras, not time). Temporal context at inference =
  `heatmap_mhcrnn` (frames t-2..t+2), never trained until now. The 09-30 labelling plan kept context frames named
  for it. LP 2.4.2 has NO multi-view + context model; the route to both is per-camera context models + the
  calibration-free multi-camera EKS.
* Round-4 LP project `/root/lp/cam4-2026-10-13` (`litpose convert` of the DLC training copy staged to
  `lp_stage\cam4_r4_dlc_trainingcopy`; 540 rows, 0 all-blank; videos symlinked to round 3's); configs
  `config_r4_heatmap.yaml` (identical to round 3 but the labels) and `config_r4_mhcrnn.yaml`.

### F. Multi-view LP pilot — evaluated 10-08 (see the 10-07 handoff §3A): arm (b) = arm (a); not adopted.
  Priya: the multi-view cam1 jaw through the spout rod is closer than single-view but not perfect.

**Preliminary (PS93 0814, 4 models; `r4_eval/cue_traces_compare[_cleaned].png`):** jaw shape identical across models;
DLC round 4 loses tongue coverage (peak ~0.35 vs ~0.45 for DLC round 3); LP round 4 has the fewest wild tongue points
and the best cleaned tongue coverage (~0.58); DLC round 3's cleaned jaw x has many -30 px spikes. LP round 4 jaw x
side-flips at the close positions -- labels confirmed correct by Priya (DECISIONS 10-13 evening). LP r4 temporal-
context model retraining (batch-size fix; `lp_stage/lp_r4_fix.log`), round-3 LP on 0821 re-run from its own dir.

**10-13 evening:** the machine CRASHED at 19:01 (bugcheck 0x20001, HYPERVISOR_ERROR, as on 09-30) ~2 min into the
WSL `heatmap_mhcrnn` training. Measured natively: context models need > 7.8 GB at batch 2 / 640 px once the backbone
is unfrozen; batch 1 / 640 px = 4.6 GB. Now running WINDOWS-NATIVE: supervised `heatmap_mhcrnn`, batch 1, 640 px
(`lp_stage/lp_win_r4_mhcrnn.ps1`, log `lp_stage/lp_win_r4_mhcrnn.log` + `.out`; project `lp_stage/lp_cam4_r4_win`),
then predicts both session clips + the six review clips -> `lp_stage/r3_vs_r4_lp/r4mhcrnn_sup_win/`. Native Windows
has no DALI -> no unlabelled-video temporal LOSS; the semi-supervised context model needs WSL: do it ATTENDED, batch
1 + `torch.cuda.set_per_process_memory_fraction`, after a short monitored test. Still to run: the review-clip LP
predictions (r3, r4 heatmap) and a round-3 context model (`lp_r4_followon.sh` died in the crash).
**Side cameras:** labelling plan + `SIDECAM_GUIDE.html` built (DECISIONS 10-13 night).

**10-14 early -- comparison DONE (DECISIONS 2026-10-14):** five models on both sessions (`r4_eval/cue_traces_compare*.png`,
`model_lick_compare.csv`, `model_angle_compare.csv`); LP r4 heatmap best all-rounder; the (supervised, Windows) context
model removes jaw side-flips but is under-confident on tongue. Windows-native inference copies of the WSL LP models:
`lp_stage/lp_models_win/` (configs repointed, `losses_to_use: []`). NEXT: semi-supervised context model in WSL,
attended; Priya to judge `lp_r4_extra_nocontact_licks.png` and `tongue_frames_all_models.png`.

## 2. Decisions (10-08 .. 10-13)
* Null/potent closed as a supplementary control (Priya). Priorities: (1) ENL movement-removed trajectory -- done;
  (2) executed-angle vs target (MUST REVISIT) once tracking allows; (3) tracking.
* ENL summaries use the CURATED set (`analysis_kit.curated_labels()`) with >= 60 trials; never a session with an
  incomplete movement model.
* Camera templates exist for every analysed session with video; PS93 0813 has none (no video).
* Model choice for cam4 is made on lick detection / kinematics, not pixel error. Evaluate the LP temporal-context
  model before more labelling.

## 3. Pending / to-do
1. §0 comparison -> figures + lick detection -> pick the cam4 model -> production O2 run over all sessions.
2. If the context model helps the tongue: cam1 context model, then per-camera context + EKS as the 2-camera route.
3. Executed-angle vs target readout (needs 1).
4. Pixel-vs-LocaNMF decoding gap: non-motor control area (spatial-leakage check).
5. Labelling: cam1 Part 2 (4 empty folders) and Part 3 (round 2) -- worth it only if the model comparison says
   labels, not architecture, are the limit.
6. Optional: tongue likelihood cutoff for round 4 (it is under-confident on clear protrusions at 0.6).
7. FaceRhythm (Hakim et al., the paper Priya cited with Musall): reading notes + proposal in `docs/FACERHYTHM_PROPOSAL.md`
   (moved from `lp_stage/facerhythm/` 10-13) -- FR components as video regressors alongside / instead of ME PCs.

## 4. Pitfalls (new)
1. **GNU sed and Python string edits mangle Windows paths** (`\U`, `\L` = case conversion; `\n` becomes a newline
   in a non-raw string). Use the Edit tool, or build paths with `chr(92)`.
2. **DLC 3 takes the project root from the config file's FOLDER**, not `project_path`: a temporary config for an
   older shuffle must sit inside the training project (delete it after -- `check_round4.py` does, in `finally`).
3. **`deeplabcut.analyze_images` writes `image_predictions_*.h5` into `labeled-data/`** of the training project --
   remove them after (Priya OK'd deleting the two made 10-13).
4. **WSL cannot see MICROSCOPE today** (`mount -t drvfs` "Operation not supported"; earlier handoffs used `/mnt/n`):
   stage inputs on C: for anything run in WSL.
5. **A session with missing inputs must fail loudly**: PS93 0813 ran with an incomplete movement model and nothing
   flagged it until its video-only accuracy (chance) was inspected.
6. **Test errors from different held-out splits are not comparable** across DLC rounds; score on frames held out by
   both (the split is seeded by session and changes when new sessions join).

## 5. Where things are
| what | where |
|---|---|
| ENL trajectory CSV / figure | `M:\...\Widefield\labcams\enl_movement_multisession\` |
| null/potent outputs (closed) | `M:\...\Widefield\labcams\null_potent\` (summaries v1-v4 archived; `_svd` files) |
| round-4 DLC review | `M:\...\DeepLabCut\Widefield\inference_check_20261012_round4\` (clips, labeled_clips, summary CSV, `heldout_both/`) |
| round 3 vs 4 session comparison | `M:\...\DeepLabCut\Widefield\session_poses\PS93_20260814\r4_eval\`, `...\PS93_20260821\r4_eval\` |
| LP round 4 | WSL `/root/lp/cam4-2026-10-13/` (models `r4_heatmap_20261013`, `r4_mhcrnn_20261013`) |
| DLC round 4 log | `C:\Users\SabatiniLab\lp_stage\dlc_train_cam4_round4.log` |
| new code | `trial_windows.{spout_motion_spans,docked_spans,_daq_cue_trials}`, `null_potent.{between_position_cov,subspace_fraction}`, `scripts/null_potent_session.py --basis svd`, `scripts/pose_cue_traces.py compare` |
