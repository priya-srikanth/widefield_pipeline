# Handoff — 2026-10-15 late / 10-16: LP r4 production on O2, organised orofacial outputs, split-session fixes

**START HERE.** Supersedes `STATUS_2026-10-15_JAW_CLEANING_MODEL_CHOICE_HANDOFF.md` as the entry point (still the
reference for jaw cleaning v35, the cohort jaw-steering readout and the model choice; its pitfalls still apply).
Model register: `docs/POSE_MODELS.md`. O2 procedure: `runbooks/o2_pose_inference.md` (LP section + steps table).
Reasoning: `DECISIONS.md` 2026-10-15 entries (incl. "(night)"); split sessions: `docs/EXPERIMENT_ERRORS.md`
2026-08-12 entry ("Full audit", "PS92 0604"). Analysis desktop; M: = MICROSCOPE.

**Open items waiting on Priya: [`OPEN_QUESTIONS_PRIYA.md`](OPEN_QUESTIONS_PRIYA.md)** (keep it current).
Cohort pre- vs post-stroke kinematics: `python -m wfield_local.orofacial_cohort` -> `LP_analysis/<run>/cohort/`
(DECISIONS 2026-10-16 entries). Lick-floor QC: `python -m scripts.lick_floor_qc`.

## 0. RUNNING / IN PROGRESS
0. **O2 pose-model TRAINING job 56522565** (`DeepLabCut/Widefield/o2/lp_train_20261016`, built by
   `scripts/o2_lp_train_bundle.py`): ViT-B DINOv2 / ResNet-50 b8 / ctx-semi b4, all 640 px, round-4 labels, then
   predicts the eval clips; score with `scripts/model_lick_compare.py` like the 10-14 models. Local ViT-S DINOv2 512 px
   training in WSL (`lp_train_vit_r4_wsl.sh`, log `lp_stage/lp_train_vit_r4_wsl.log`).
1. **O2 LP r4 production — array job `56464442`** (`DeepLabCut/Widefield/o2/lp_r4_prod_20261015`, 137 videos = every
   registered widefield session with video; gpu_quad, `--gres=gpu:l40s:1`, reader DALI, throttle %12, time limit
   lowered to 6 h by Priya). ~210 fps on L40S -> wall time = frames / 210 (June ~1.2-1.5 h, August ~2.4-2.9 h).
   On 10-16 only ~3 tasks were running (`(Priority)`: waiting for L40S), so Priya UPDATED THE PENDING TASKS to any GPU
   type (`scontrol update JobId=56464442 Gres=gpu:1`) and TimeLimit 6 h -- later tasks may run on RTX 8000 / A40 /
   V100 (~2x slower per video on RTX 8000; results identical). sprio then: priority 772,575 (fair share 426,126,
   partition 342,857, age 3,576). Never submit a duplicate array over the same videos.
   **10-16: the V100S nodes (`compute-g-17-[147-152]`) failed 43 tasks** (torch in `lp` has no sm_70 kernels) -> job
   given `ExcNodeList=compute-g-16-[254-255],compute-g-17-[147-152]`; the failed ids resubmitted as array **56481998**
   (duplicate 56481741 cancelled); throttle raised to 24. Check coverage with the runbook's done.json / squeue check. The bundle on the share still
   says `gpu:l40s:1` / 24 h in run_pose.sbatch (the live job was edited, not the file).
2. **Processing as results arrive.** Priya copies back (COMMANDS step 5, transfer node,
   `rsync -ah --partial --exclude 'tmp*' <scratch>/out/ <share bundle>/out/`); then here:
   `python -m wfield_local.orofacial_outputs process --bundle lp_r4_prod_20261015 --jobs 4` (dlc env, with
   `WIDEFIELD_FIGURES_WORKING=C:/Users/SabatiniLab/cue_lick` and `WIDEFIELD_SESSION_CACHE=C:\Users\SabatiniLab\.widefield_session_cache`).
   Idempotent (skips sessions with a manifest), ~100 s per session since 10-16 (pre-cue + ITI passes), rebuilds
   `index.csv`. 10-16 additions: `ix_daq_s` (per-frame DAQ time), `licks_precue.csv`, `licks_iti.csv` (DECISIONS
   2026-10-16); the sessions processed before them are re-run once with `--force`. **12 sessions done**
   (PS92 0602-0608, 0805-0807; PS93 0814, 0821).

## 1. What happened since the 10-15 handoff (and why)
1. **Production model = LP r4 alone** (Priya). After cleaning, the EKS ensemble's raw jitter advantage vanished
   (one-frame tongue spikes 2 vs 0-1 per 10k frames); LP r4 misses clear tongue only where the tip is behind the rod
   (2 / 16 of Priya's frames, 32-36 ms gaps, interpolated). LP r3 ctx-semi scored: not a contender (29 / 58 reviewed
   real licks, jumpiest tongue). Model archived on the share: `lightning-pose/cam4-2026-10-13/models/r4_heatmap_20261013/`.
2. **Cleaning tweaks adopted** (Priya QC'd sheets): tongue inter-lick veto (a gap entered retracting and left
   protruding, both ends within 80 px of the mouth -> mouth fill, not a bridge); jaw-off-tongue (jaw within 25 px of a
   tongue point > 60 px from the resting jaw -> not visible). Accuracy unchanged or better (LP r4 recall 98.5 -> 98.8 %
   pre). Tongue keeps the mouth-zero fill (presence / absence of the tongue in the trace); jaw hidden = NaN.
   Kinematics per lick now also carry `protrusion_max_measured_px`, `interp_frac`, `peak_interp` (5.6 % of licks have
   an interpolated / imputed peak; measured-only max protrusion = all-frames value). "Interpolated" = PCHIP gap fill in
   `orofacial_clean`; "imputed" = the v7 detector's parabola / PCHIP apex for a flattened or occluded peak.
3. **O2 LP path built and validated.** `wfield_local/o2_lp_runner.py` + `o2_inference bundle --kind lp` (+
   `--all-widefield`), smoke test as a batch job with the array queued `--dependency=afterok` (no interactive GPU
   waits), one `rsync --files-from=videos.txt` for staging (videos staged 10-06 are reused; staged videos are NEVER
   deleted until Priya says server analysis is finished). Pilot (PS93 0814 / 0821): O2 vs desktop median 0.15-0.45 px,
   p95 <= 1.7 px, confident-call agreement >= 99.7 %. Reader DALI (bench 105 vs OpenCV 59 fps on an RTX 8000; DALI is
   also how LP r4 was evaluated); L40S requested. O2 facts: Priya has `gpuquad_qos`; gpu_quad limit 5 days; `gpu_qos`
   caps the `gpu` partition at 34 CPU / 420 GB per user; MaxArraySize 10001; fair share counts USED time x resources.
4. **PS92 0812 (crash + concat) fixed end to end and back in every analysis** (Priya): (a) six code paths took part 1's
   truncated DAQ file by name sort -> `daq_io.session_h5` (registered / `_concat`); the lick cache was rebuilt (662 ->
   5,840 licks; that was the "22 ENL trials"); (b) camera templates per recording against the concat DAQ; (c) the
   padded gap read as a 2,441-s running bout -> `daq_io.recorded_mask`. ENL re-run (210 trials, raw 0.40 / residual
   0.28; summaries 119 sessions, pre raw 0.40 -> 0.41, else unchanged). State decoder re-rendered with 0812.
   Main LocaNMF analyses always used the concat file and are unchanged.
5. **PS92 0604 (the other concat): camera templates PER DAQ SEGMENT.** One camera file ran through a 27-s DAQ restart;
   the two segment fits differ by a constant 1.62 s because `concat_split_session` padded the gap from file
   timestamps. Imaging / behaviour unaffected (all on the DAQ clock; behaviour log maps by sync pulses). Split days
   have NO single `<date>.npz` any more (copies moved aside as `.split_day_ambiguous_20261015`) so single-template code
   fails loudly; split-aware code uses `camera_sync.templates_for` (lists per recording).
6. **Frozen pre-stroke decoders are the rule for future cohorts** (Priya): the behavioural-STATE decoder now goes
   through `frozen_models.load_or_fit` (`locomotor_decoder.frozen_state_models`, kind `state`, full + LOSO models);
   frozen on this desktop (11 pre sessions per animal, PS92's incl. 0812) and the 4 models PUBLISHED to
   `labcams/frozen_models` (SHA-verified). This desktop is the publishing box for STATE models; the behaviour box for
   position models. `docs/FROZEN_MODELS.md` §7, CLAUDE.md rule 10.
7. **Organised orofacial outputs on the share** (`wfield_local/orofacial_outputs.py`, root `orofacial` =
   `M:/MICROSCOPE/Priya/Widefield/LP_analysis/`; moved from `Priya/Orofacial/Widefield` on 10-16, Priya): RUN-versioned (`lp_r4_heatmap_20261013__clean_v35`; `run.json` holds
   a params digest and refuses mixing) and SESSION-first (`sessions/<animal>/<date>/`: manifest.json, windows.npz,
   clean.npz, licks / trials / jaw_trials / tongue_jaw_mismatch CSVs, qc.png); `cohort/{tables,figures}/<analysis>/`
   for cross-session work; `index.csv` + `README.md` at the top. `load_session(rv, run, animal, date)` reads one back.
   Improves on stroke_orofacial's type-first tree (one session over ~8 folders, overwrite in place).
8. **Whole-session QC (PS93 0814 / 0821, O2):** spout 100 % confident, six tight clusters (SD 0.3-2.3 px), no drift;
   nose 100 %, frame steps median 0.2-0.3 px; jaw 98-100 %; tongue visible in a median 23 % (pre) / 7 % (acute) of
   post-cue frames, long zero stretches at session ends (disengagement); acute: far positions nearly flat.

## 2. Decisions (10-15 afternoon .. 10-16)
LP r4 alone for production; cleaning v35 incl. the two tweaks; tongue fill kept, jaw hidden NaN; flag interpolated
peaks rather than drop; pose inference only for registered widefield sessions (minus `excluded` phases); split days
get per-segment templates and no single template; frozen + stored pre-stroke decoders for every reference (state
added); orofacial outputs run-versioned + session-first on the share; O2: DALI + L40S, one array, %12, 6 h per video,
staged videos kept until Priya says.

## 3. Code added / changed (all pushed; last commit at the time of writing in `git log`)
`wfield_local/o2_lp_runner.py` (new), `o2_inference.py` (`--kind lp`, `--all-widefield`, smoke job, files-from,
`build_windows`, split-day collect, `compare_to_clip`), `orofacial_outputs.py` (new), `orofacial_clean.py` (inter-lick
veto, `jaw_off_tongue`, jaw v35, hidden_fill), `tongue_kinematics.py` (measured-only fields), `camera_sync.py`
(`build_split_templates`, `concat_segments`, `templates_for` -> lists, `run` split-aware, `_match_daq` prefers concat),
`daq_io.py` (`session_h5`, `recorded_mask`), `behavior_events.py` (gap masking), `locomotor_state.py` (no exclusion),
`locomotor_decoder.py` (`frozen_state_models`), `scripts/enl_movement_multisession.py` (motion energy per recording /
segment), `scripts/cleaned_label_clips.py` (new), `scripts/jaw_lateral.py`, `scripts/model_lick_compare.py`,
`scripts/pose_cue_traces.py` (`--trials`), configs (`o2_inference.lp`, cleaning switches, paths `orofacial`).
Tests: `test_orofacial_outputs.py` (new), `test_o2_inference.py`, `test_camera_sync.py`, `test_orofacial_clean.py`,
`test_locomotor_state.py`.

## 4. Pending / to-do (in order) -- updated 10-16 evening

1. Keep processing O2 results as they are copied back: `python -m wfield_local.orofacial_outputs process --bundle
   lp_r4_prod_20261015 --jobs 4` then `python -m wfield_local.orofacial_cohort --jobs 6` (NO overrides needed any more).
   88 / 136 sessions done (PS92 complete, PS94 30, PS95 10 pre only, PS93 4 -- PS93 / PS95 post are mostly the V100
   re-runs queued on O2). Re-read every cohort figure when PS93 + PS95 are in: all results so far are PRELIMINARY.
2. Score the O2-trained models (job 56522565: ViT-B DINOv2 / ResNet-50 b8 / ctx-semi, all 640 px) when copied back:
   `scripts/model_lick_compare.py PS93:<d> --model ... --review lp_stage/lick_review/review_scored.csv` on 0814 + 0821
   (predictions land in `models/<name>/video_preds/`), against LP r4 and the local ViT-S baseline (POSE_MODELS.md).
3. **Executed-angle vs target readout (MUST REVISIT)**: the kinematic side now exists in `orofacial_cohort`
   (angle_tuning, angle_gain_far, ipsi_bias_deg, Family 8.2 cue_angle*); the imaging-side readout test
   (`lick_template_match`) on whole sessions is still to do.
4. ~~Jaw lateral readout with LP r4 on whole sessions~~ DONE 10-16/17 (`wfield_local/orofacial_jaw_lateral.py`): jaw
   steering toward the target abolished acutely in 4 / 4 animals (-12.2 px [-13.8, -10.7]), not a shorter-reach effect;
   recovery timing differs by animal (DECISIONS 2026-10-16 jaw entry + 10-17 update). Re-run as sessions arrive.
5. Other cohort analyses onto `orofacial_outputs`: contact classes, licking dynamics / bout structure by epoch.
6. Side cameras, pixel-vs-LocaNMF decoding gap, FaceRhythm -- unchanged.
Open items waiting on Priya: `OPEN_QUESTIONS_PRIYA.md` (clean-up PowerShell, O2 copy-backs).

## 4b. 10-16 afternoon / evening (all in DECISIONS 2026-10-16)

- Outputs moved to `Priya/Widefield/LP_analysis`; `wfield_local/orofacial_cohort.py` = the pre- vs post-stroke cohort
  pass (session / position / lick-set metrics, lick phase time-aligned + phase-normalised (interpolated, stroke_orofacial
  method), phase portraits, x-y trajectories (camera view, SEM ribbons), lick 1 vs 2, lick walk (angle at peak +
  max angle in lick, paired lick 1 vs 3), mismatch, Family 8.1 / 8.2, minus-pre versions; spout-angle reference,
  lick selection in titles, in-mouth included / excluded stated).
- Pre-cue + ITI lick tables and per-frame DAQ time added to every session (`licks_precue.csv`, `licks_iti.csv`,
  `ix_daq_s`). Lick floor QC: the 20 px floor is moot -- LP never tracks a tongue within ~20 px of the mouth.
- O2: V100S nodes excluded (no sm_70), 10 h limit, stuck-throttle fix, fair share collapsed to 0.022 then 0.068.
- Pose models: ViT-S DINOv2 512 px (local) not better than LP r4; O2 training bundle `scripts/o2_lp_train_bundle.py`.
- This desktop's machine profile fixed (rule-10 check passed). Camera templates built for PS94 / PS95 0604; PS94 /
  PS95 0601 unalignable (rig camera problem).
- Preliminary findings (3 animals with acute data): acute far contra -- licks / trial ~1, contact 1-22 %, protrusion
  -30..-48 px, isolated licks (neighbour fraction 0.94 -> 0.38-0.56), lick 2 delayed (+240-630 ms), tongue angle while
  out ~+40 deg toward ipsi across the response window (PS92, PS94), recovering by chronic in PS92. Pre-stroke: later
  licks drift toward the middle at the close spouts in all four animals.

## 5. Pitfalls (new; older lists still apply)
1. **This desktop resolves local roots to the behaviour box** (`C:/Users/sabatini`): set `WIDEFIELD_FIGURES_WORKING`
   and `WIDEFIELD_SESSION_CACHE` before any render or processing. A render without them FAILED EVERY SESSION AND
   EXITED 0 -- grep logs for `PermissionError`; never trust the exit code.
2. **Never pick a DAQ file by name sort** -- use `daq_io.session_h5`. Never assume one camera template per date --
   use `camera_sync.templates_for` (split days: per recording / per segment).
3. **A model's "zero flips" can be abstention**; report coverage with any flip count.
4. **`--info=progress2`'s % / ETA count skipped files** -- use `rsync -n --itemize-changes` to see what would copy.
5. **Shell-edited Python with `\n` inside strings / heredocs keeps breaking** (real newlines land in source; quote
   mismatches in `bash -c`): write edit scripts with the Write tool and run them, or use the Edit tool.
6. **Stale bundle text on scratch:** after `bundle --refresh` on the share, the scratch copy changes only when Priya
   re-runs the bundle rsync (step 1); `bundle.json` now follows refreshes (reader / precision / gres).
7. **LP needs OpenCV installed separately** (`opencv-python-headless`); LP already brings DALI -- do not add another.
8. Interactive `srun` on O2 waits for an allocation -- keep O2 steps to `sbatch` on the login node.
