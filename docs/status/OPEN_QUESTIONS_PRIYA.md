# Waiting on Priya — running list

Kept by Claude; answer any item whenever convenient (a line in chat is enough); resolved items move to the bottom.

## To run (only you can: my deletes on the share are blocked by the session's safety check)

1. **Clean-up of `M:\MICROSCOPE\Priya\DeepLabCut\Widefield`** — everything below is approved and checked; the unique
   pieces were moved out first (round-3 evals -> `lightning-pose\_evals_round3\`; `_frame_staging`'s provenance files
   and older label versions -> `_label_history\frame_staging_snapshot_2026-09-16\`). PowerShell:

   ```powershell
   $W = "M:\MICROSCOPE\Priya\DeepLabCut\Widefield"
   Remove-Item -Recurse -Force "$W\o2\r3_PS93_20261002", "$W\o2\lp_r4_pilot_20261015", "$W\_frame_staging_unassigned", "$W\round4_scan", "$W\cam1_round2_scan", "$W\_guide_archive", "$W\_frame_staging"
   Remove-Item -Force "$W\LABELLING_GUIDE.bak-2026*.html"
   Remove-Item -Recurse -Force "$W\lightning-pose\cam4-2026-09-26", "$W\lightning-pose\cam4-2026-09-28"
   $S = "$W\session_poses"
   Remove-Item -Recurse -Force "$S\PS92_20260821_r4", "$S\PS92_20260824_r4", "$S\PS93_20260819_r4", "$S\PS93_20260826_r4", "$S\PS94_20260921_r4", "$S\PS95_20260917_r4", "$S\PS94_20260908", "$S\PS93_20260814_full", "$S\PS93_20260821_full", "$S\_pre_full_20261006"
   Remove-Item -Recurse -Force "$W\o2\lp_train_20261016\hf_cache"
   ```

   `session_poses\PS93_20260908` is KEPT (PS93's chronic clip in the jaw lateral readout). The last line removes a
   broken partial weight cache in the training bundle; the working one is `hf_home`.
2. **O2 pose-model training** — job **56522565** (3 tasks: ViT-B DINOv2 640 px b8, ResNet-50 640 px b8, context
   ctx-semi 640 px b4), queued behind inference + other users. When `models/*/DONE` appear: copy back (bundle
   `COMMANDS.md` step 4) and tell me; I score them against LP r4.
3. **O2 inference** — copy back finished videos (`lp_r4_prod_20261015` COMMANDS step 5) whenever more finish; final
   coverage check when `squeue -u ps150` is empty (runbook: done.json / squeue check -> `not done: 0`).
   Status 10-16 evening: 88 / 136 sessions done; gpu_quad full; fair share 0.068 (recovering).

4. **O2 cam1 round-2 LP training** (2026-10-18) — bundle `o2/lp_train_cam1_r2_20261018` (heatmap 640 b4 = LP r4
   recipe, ctx-semi 640 b4; 634 cam1 labels). Steps in its `COMMANDS.md`; when `models/*/DONE` appear, copy back and
   tell me. DLC cam1 round 2 trains locally (iteration 1).

## Decisions (none pending right now)

## FYI

- PS94 0601 and PS95 0601 cannot be camera-aligned (~280k dropped cam4 frames each, same day -> rig camera problem on
  06-01); outside the curated set.
- PS92 0908: cam4 stopped ~14 min before the DAQ — its last 68 trials have no pose data (EXPERIMENT_ERRORS).

## Resolved

- 2026-10-16 — figure clarity: spout-angle reference on raw-angle plots, lick selection in titles, quantity+unit y
  labels, SEM ribbons on trajectories / portraits, max-angle-during-lick walk (Priya).
- 2026-10-16 — Family 8.2 equivalent built (`cue_*` figures, three in-mouth versions) (Priya: yes).
- 2026-10-16 — machine profile fixed (detected by `C:/Users/SabatiniLab`, both drive letters listed); frozen-model
  store unchanged (`C:\Users\SabatiniLab\frozen_models`, 8 files); overrides no longer needed (Priya: yes).
- 2026-10-16 — ViT-S DINOv2 (local, 512 px): not better than LP r4 (precise but misses faint licks; jumpy on the pre
  clip). Baseline for the O2 ViT-B 640 px run. `docs/POSE_MODELS.md`.
- 2026-10-16 — tiny (< 20 px) incomplete licks: not needed (Priya). LP never tracks them (no labels that close);
  jaw-without-tongue covers attempts. Floor stays 20 px (moot).
- 2026-10-16 — lick-phase angle convention: show both (Priya); captions say what each reads (in-mouth excluded = given
  the tongue is out; included = also how much it is out).
- 2026-10-16 — old LP folders: delete, keep round-3 evals (Priya; evals moved).
- 2026-10-16 — session_poses: delete older runs only (Priya) — minus PS93_20260908 (needed; item 1).
- 2026-10-16 — _frame_staging: verified — frames identical to the DLC project; only label snapshots + provenance files
  were unique (moved). Delete in item 1.
- 2026-10-16 — outputs location: `Priya/Widefield/LP_analysis` (Priya).
