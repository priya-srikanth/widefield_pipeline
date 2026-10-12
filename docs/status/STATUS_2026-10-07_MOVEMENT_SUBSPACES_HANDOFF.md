# Handoff — 2026-10-07 (evening): movement-removed position coding (residual + ENL + null/potent subspaces), video regressors, multi-view LP pilot, cohort registration

**START HERE.** Supersedes `STATUS_2026-10-06_FULLSESSION_MOVEMENT_HANDOFF.md` for current state (still the
reference for the 10-06 orofacial cleaning changes: windows cut at the next trial, Rule 4 swap, contact classes,
smooth kernels / split licks). Tracking / labelling / O2 background: `STATUS_2026-10-05_CAM1_LP_EKS_O2_HANDOFF.md`.
Reasoning and numbers: `DECISIONS.md` addenda 2026-10-06 (night) and 2026-10-07 (several). Analysis desktop;
M: = MICROSCOPE (the behaviour box calls it N:). Everything movement-related below is ONE pre-stroke session
(PS93 0814, whole-session O2 poses from the ROUND-3 cam4 model) unless stated -- pipeline results, not findings.

## 0. STATE (2026-10-12) -- nothing running
* ENL movement-removed decoding: DONE on 112 / 132 sessions -> RESULT in DECISIONS 2026-10-12 (the residual tracks
  the raw pre-cue code across recovery, r 0.77-0.92 per animal). To do: fix the 12 code failures (`KeyError
  'cue_s'` on 0805 / 0806 / 0812 / 0929; PS93 0805, PS95 0813 index errors) and rerun those (resumable); the 8
  early-June sessions have no camera alignment templates (data gap).
* Null/potent 24-session subset, LocaNMF + SVD: DONE; CLOSED as a supplementary control (no epoch change).
* 10-12: failures fixed (`trial_bounds` DAQ-cue fallback, coverage drop; b8dc9e4) and camera templates built for the
  14 analysed sessions that lacked them (55 / 56 good; PS92 0605 cam4 FAILS quality -- 165k dropped frames; PS93
  0813 has NO VIDEO, logged in docs/EXPERIMENT_ERRORS.md). Rerun of the 20 failed sessions: log
  `M2_enl_multisession_fixes.log`. **Summaries use the CURATED set only** (`analysis_kit.curated_labels()`, 120
  sessions: excludes 6/1-6/5 and 8/5 per animals.yaml `cross_session_exclude`); the 10-12 112-session result was
  already all-curated. The June / 8-5 rows in the CSV are extra, not for the trajectory.
* 10-12: cam4 ROUND-4 DLC retrain RUNNING (`wfield_local.dlc_train`, dlc env, GPU; log
  `C:\Users\SabatiniLab\lp_stage\dlc_train_cam4_round4.log`; 540 frames / 21 sessions, 200 epochs ~1.5 min each).
  Held-out split changed (6 sessions, 142 frames, incl. round-4 PS94 0921) -- not the same test frames as round 3.
  NEXT: per-part test RMSE, review clips, then the production O2 run.

## 0b. PRIORITIES (Priya, 2026-10-08 night -- see the DECISIONS addendum of that name)
Null/potent is DEMOTED to a control: finish the running subset, check whether the potent share of between-position
variance changes across epochs, then stop. Next, in order:
1. **ENL movement-removed decoding across sessions / epochs** (`scripts/enl_movement_multisession.py`). VALIDATED
   10-09 on PS93 0814 (n 475): raw 0.43, residual 0.28 (whole-session model gave 0.26), video_only 0.95 (0.96);
   projecting out the top-3 movement-predicted dims removes 91 % of movement-predicted and 65 % of total ENL
   variance, decoding 0.38; top-8: 99 % / 89 %, 0.32. RUNNING 10-09 07:45 on all 132 regime-B sessions (the 24
   subset first; order in `lp_stage/full_0814_logs/compare/enl_order.txt`), cmd PID 23116, log
   `M_enl_multisession.log`, ~10 min / session, resumable; output
   `<labcams>/enl_movement_multisession/enl_movement_multisession.csv`.
2. **Executed-angle vs target readout** (MUST REVISIT; needs the next cam4 model + whole-session poses).
3. **Tracking**: cam4 round-4 labels -> retrain -> production O2 run.
Side check: pixel-PCA vs LocaNMF decoding gap (MO pre-cue 0.47 vs 0.32) against a non-motor control area.

**Labelling status (checked 2026-10-08 from the label files; nothing saved since 10-05 17:03):** Part 1 cam4 round 4
21 / 131 (PS93 acute 21/22; five folders untouched) -- ON THE CRITICAL PATH for priorities 2-3; Part 2 cam1 round-1
290 / 376 (4 empty folders: PS93 0606, PS92 0820, PS95 0821, PS94 0903); Part 3 cam1 round 2 0 / 254. Re-check with
the helpers in `wfield_local/dlc_cam1_guide.py` (`_round4_folders`, `_folders`, `_round2_folders` + `folder_status`).

## 1. Findings since the 10-06 handoff (PS93 0814 whole session unless noted)
**Movement model comparison** (CV R^2 of the full model; smooth kernels, split licks, retuned cleaning):
| model | MOp_L | MOp_R | MOs | SSp |
|---|---|---|---|---|
| DLC tongue/jaw/licks | 0.10 | 0.03 | 0.11-0.19 | 0.33-0.37 |
| + DAQ running speed | 0.43 | 0.27 | 0.27-0.32 | 0.36-0.40 |
| + video (4 cams x 30 ME PCs, strobe..trial end) | 0.58 | 0.35 | 0.38-0.44 | 0.45-0.49 |
Running unique 0.13-0.33 in motor cortex; video unique 0.07-0.15 everywhere; target (cue per position) unique ~0.

**Residual position test (cue -0.5..+1.5 s window, spans ENL tail + cue + first licks), full movement model,
200 perms:** position-specific falls 10-40x vs DLC-only; ipsilesional MOp_L / MOs_L ~0 (p 0.96-1.00); tiny remnant
(+0.0002..0.0004) in SSp-m (p <= 0.02), MOp_R, SSp-n_R. Angle beyond position tiny but detectable (+0.0004..0.0012).
Priya: unsurprising -- in the cue/lick window the target IS the executed movement.

**ENL (2 s pre-cue, lick-free) position decoding** (`scripts/enl_movement_removed_decode.py`; the project's ENL
decoder, 484 engaged trials, chance ~0.17, all p = 0.001): raw 0.40 (SSp 0.43, MO 0.32); - DLC 0.38; - DLC +
running 0.38; - DLC + running + video **0.26 (SSp 0.30, MO 0.25)**. Video + running alone decode position 0.96.
`scripts/enl_video_spout_check.py`: face cams 0.92 / 0.94, body cams 0.46, flat across the 2 s (no post-strobe
settling). Data-driven spout masks FAILED (posture + block-time drift, 30-55 % of pixels; per-position mask leaked
position 0.46 -> 0.76-0.80). Priya: a stationary spout makes ~no motion energy; remaining apparatus routes =
occlusion / contrast by the rod => the video may OVER-remove => **0.26-0.30 is a LOWER BOUND: a pre-cue, lick-free
position code survives removal of tongue, jaw, running and all four cameras** (one session; linear removal).

**Behaviour (cohort, through 10/5-10/6):** plateau holds in all four (far_R hit PS92 0.96-1.00, PS93 0.96-1.00,
PS94 0.87-0.99 wobbling as before, PS95 0.98-1.00). Neural plateau: the behaviour box is producing it (stability
test after its render) -- not run here.

## 2. Decisions (10-07)
* Running speed is ONE definition: `behavior_events.treadmill_speed` / `session_speed` (configs
  segmentation.treadmill), used by running bouts and the movement models.
* Video regressors: no spatial masks (`video_motion.masks: {}`); the motorised spout is excluded in TIME (video
  blanked outside position strobe .. trial end); the treadmill is kept (only the paws move it).
* Movement-removed position: next steps are (1) ENL across sessions with a running + video model (DLC adds ~nothing
  in the lick-free ENL) -- `scripts/enl_movement_multisession.py` WRITTEN, NOT RUN / NOT TESTED, superseded in
  priority by (2); (2) **movement-null / movement-potent subspaces exactly per Hasnain et al. 2025** (below).
* **Null/potent method (Priya supplied Hasnain et al. 2025 Methods):** stationarity from MOTION ENERGY (per pixel
  |median next N - median previous N| frames, 99th percentile over pixels per frame; N = 3 at our 250 fps = their
  12.5 ms; per-session threshold at the valley of the bimodal distribution -- theirs manual, ours automatic + figure
  + override), plus DAQ licks / running count as moving, 0.3 s post-movement buffer excluded (GCaMP); subspaces by
  their joint normalised-variance objective with orthogonality (Stiefel gradient ascent = their manopt), d =
  min(N/2, 20); two-stage PCA control (5 + 5); parallel-analysis dimensionality reported; fit on all labelled frames
  (label-free => not circular for position decoding); readout = our position decoder on X Q Q' reconstructions,
  ENL and post-cue windows. Code: `wfield_local/motion_state.py`, `wfield_local/null_potent.py` (6 tests),
  `scripts/null_potent_session.py`.
* `grant_kit._day` = calendar days (was month*31; October sessions were +1). Parity tests skip on a STALE
  stroke_orofacial checkout instead of failing (the behaviour box's is ~90 commits behind).
* MUST REVISIT (Priya, "very important"): the executed-angle readout test with complete models -- conditions in
  DECISIONS 2026-10-07 and §3 below.

## 3. Lines of investigation -- state and next steps
**A. Tracking models**
* cam4 DLC: round-3 model in production (O2 run on PS93 0814 done, matches desktop). Waits on labelling round 4
  (student, CAM1_GUIDE Part 1) -> retrain -> production O2 run over all sessions (videos for cam1 + cam4 already on
  scratch; never touch scratch file dates).
* cam1 DLC best-40 (own project) + single-view LP (not better than DLC); cam1 round 2 labels pending (Part 3).
* **Multi-view LP pilot EVALUATED 2026-10-08** (`lp_stage/mv_eval/`: per-split pixel error, `trace_metrics.py` on the
  16 synced clip pairs, `cam1_jaw_disagreements.png`). Arm (b) (calibration + 3-D aug) = arm (a): no gain from
  calibration with 105 / 526 triangulable moments. Versus single-view LP: nose / spout / cam4 jaw the same (2-3 px);
  tongue no better (visible 8-9 % of frames, 20-31 % of visible frames jump > 15 px, all models). ONE difference:
  cam1 jaw through spout-rod occlusion -- multi-view confident on 99 % of frames vs single-view 87 % (single-view
  correctly says not visible). Priya: the multi-view occluded jaw is "definitely closer to the right position than
  the single-view, but not perfect (often a little too far down and maybe off to the side) -- plausible but
  probably not perfect". => not adopted for production yet; its likelihood cannot flag occluded points (keep the
  split-rule `visible` ourselves); revisit after round-4 labels add paired tongue / jaw moments. Original note:
  Multi-view LP pilot (cam1 + cam4, 526 moments, split-rule `visible`): arm (a) done, arm (b) running. Evaluate;
  arm (b) understates calibration until more PAIRED labels exist (3-D augmentation needs >= 3 keypoints
  triangulated per moment: 105 / 526 today). Patch masking on (LP multi-view default).
* FaceRhythm (Priya 10-07: "we will probably eventually also want to try"): bioRxiv 10.1101/2025.09.10.675423 --
  read, compare with motion-energy PCs as video regressors; prior lab runs exist at
  `M:\MICROSCOPE\Priya\FaceRhythm\` (`facerhythm_stroke_biomarker_exp`, `fr_run_20250520_latent_bundle`).
  **Handed to a separate session 10-07 18:45** (reading + proposal only) -- don't duplicate; look for its write-up.
**B. Orofacial cleaning / kinematics** (10-06): Rule 4 swap (tip hops across the straddled spout), contact classes
  (lick window peak -60..+120 ms; grooming = long touch without the tongue), angle at peak protrusion, figures in
  image orientation. Pending: cross-session contact survey (`scripts/contact_survey.py`, not built) before fixing
  contact / grooming thresholds; `segmentation.grooming` (old proxy, OFF) to be superseded.
**C. Movement models / position coding**
  1. NOW: null/potent PS93 0814 (thresholds -> Priya) -> cohort overnight -> per animal / epoch: does the ENL
     position code sit in the movement-NULL subspace pre-stroke, and does that change after the stroke?
  2. ENL running + video residual across sessions (script ready, untested) -- complementary to 1.
  3. Movement-null by regression (2pRAM-style top-k) -- deprioritised in favour of the Hasnain method.
  4. Nonlinear movement model (their ANN control: 28 % vs 11 % VE over ridge) -- later robustness check.
  5. Per-period residual test (ENL / cue / lick) -- partly answered by the ENL decode.
  6. MUST REVISIT: executed-angle vs target readout (`lick_template_match`) once (i) the next cam4 model, (ii)
     whole-session O2 poses for pre AND post sessions, (iii) retuned cleaning, (iv) full movement model variant,
     (v) straddled-spout licks flagged.
**D. Imaging pipeline / cohort**: 8 October sessions registered (10/1, 10/2, 10/5, 10/6); behaviour box rendering.

## 4. Pitfalls (new; earlier lists still apply)
1. **CRLF**: `configs/sessions.yaml` is stored CRLF in the index; `git am` strips CR by default -> "patch does not
   apply". Use `git am -3 --keep-cr` for the behaviour box's patches.
2. **Drive letters differ by machine**: MICROSCOPE is M: here, N: on the behaviour box -- translate handoff paths.
3. **Never delete on MICROSCOPE without Priya's OK** (she OK'd `_handoff/register_1006`, deleted 10-07).
4. **Data-driven spout masks from mean frames capture posture + block-time drift**, not the rod; per-position masks
   leak position through the mask itself. Use a UNION mask, and geometric (keypoint / dark-rod) masks if needed.
5. **Long jobs**: launch detached (`Start-Process cmd.exe /c ...`); bash background tasks die at 2 h. Stop jobs by
   command line / parent PID only (shared box); a killed parent leaves ProcessPool workers -- stop them by
   `ParentProcessId`.
6. Heredocs with backslashes / `\n` inside Python strings break (Windows paths, f-strings): write edit scripts with
   the Write tool and run them.
7. PMC links hit a CAPTCHA for WebFetch; use Europe PMC (`ebi.ac.uk/europepmc/webservices/rest/...`) or bioRxiv.
8. The pre-push hook runs the full suite (~3 min); a failing test from someone else's change blocks every push --
   fix the cause, never `--no-verify`.

## 5. Where things are
| what | where |
|---|---|
| PS93 0814 whole-session outputs | `M:\...\DeepLabCut\Widefield\session_poses\PS93_20260814_full\` |
| comparison / ENL logs | `C:\Users\SabatiniLab\lp_stage\full_0814_logs\compare\` |
| null/potent outputs | `M:\...\Widefield\labcams\null_potent\` |
| multi-view LP data / launcher | `C:\Users\SabatiniLab\lp_stage\multiview-pilot-20261006\`, `train_lp_multiview_fg.sh` (copy in `scripts/wsl/`) |
| new code (10-07) | `wfield_local/{motion_state,null_potent}.py`, `scripts/{null_potent_session,enl_movement_removed_decode,enl_video_spout_check,enl_movement_multisession}.py` |
