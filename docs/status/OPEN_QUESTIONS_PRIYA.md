# Waiting on Priya — running list

Kept by Claude; answer any item whenever convenient (a line in chat is enough); resolved items move to the bottom.

## To run (only you can: my deletes on the share are blocked by the session's safety check)

1. **Clean-up of `M:\MICROSCOPE\Priya\DeepLabCut\Widefield`** — everything below is approved and checked; the unique
   pieces were moved out first (round-3 evals -> `lightning-pose\_evals_round3\`; `_frame_staging`'s provenance files
   and older label versions -> `_label_historyrame_staging_snapshot_2026-09-16\`). PowerShell:
   ```powershell
   $W = "M:\MICROSCOPE\Priya\DeepLabCut\Widefield"
   Remove-Item -Recurse -Force "$W\o2
3_PS93_20261002", "$W\o2\lp_r4_pilot_20261015", "$W\_frame_staging_unassigned", "$W
ound4_scan", "$W\cam1_round2_scan", "$W\_guide_archive", "$W\_frame_staging"
   Remove-Item -Force "$W\LABELLING_GUIDE.bak-2026*.html"
   Remove-Item -Recurse -Force "$W\lightning-pose\cam4-2026-09-26", "$W\lightning-pose\cam4-2026-09-28"
   $S = "$W\session_poses"
   Remove-Item -Recurse -Force "$S\PS92_20260821_r4", "$S\PS92_20260824_r4", "$S\PS93_20260819_r4", "$S\PS93_20260826_r4", "$S\PS94_20260921_r4", "$S\PS95_20260917_r4", "$S\PS94_20260908", "$S\PS93_20260814_full", "$S\PS93_20260821_full", "$S\_pre_full_20261006"
   Remove-Item -Recurse -Force "$W\o2\lp_train_20261016\hf_cache"
   ```
   (`session_poses\PS93_20260908` is KEPT: PS93's chronic clip in the jaw lateral readout. The last line removes a
   broken partial weight cache I left in the training bundle; the working one is `hf_home`.)
2. **O2 pose-model training** — SUBMITTED 2026-10-16 as job **56522565** (3 tasks: ViT-B DINOv2 640 px b8, ResNet-50
   640 px b8, context ctx-semi 640 px b4). When `models/*/DONE` appear: copy back (COMMANDS step 4) and tell me.
3. **O2 inference** — copy back finished videos (COMMANDS step 5 of `lp_r4_prod_20261015`) whenever more finish;
   final coverage check when `squeue -u ps150` is empty (runbook: done.json / squeue check -> `not done: 0`).

## In progress on my side (no action needed)

- Family 8.2 equivalent (cue-aligned lateral tongue position minus pre) — building (Priya: yes).
- Machine profile for this desktop in `paths.yaml` — fixing, with the frozen-model store check (Priya: yes).
- ViT-S DINOv2 512 px training locally (WSL, guarded); scored against LP r4 when done.

## FYI

- PS94 0601 and PS95 0601 cannot be camera-aligned / have no template (early June; outside the curated set).
- PS92 0908: cam4 stopped ~14 min before the DAQ — its last 68 trials have no pose data (EXPERIMENT_ERRORS).

## Resolved

- 2026-10-16 — tiny (< 20 px) incomplete licks: not needed (Priya). LP never tracks them (no labels that close);
  jaw-without-tongue covers attempts. Floor stays 20 px (moot).
- 2026-10-16 — lick-phase angle convention: show both (Priya).
- 2026-10-16 — old LP folders: delete, keep round-3 evals (Priya; evals moved).
- 2026-10-16 — session_poses: delete older runs only (Priya) — minus PS93_20260908 (needed; see item 1).
- 2026-10-16 — _frame_staging: verified — frames identical to the DLC project; only label snapshots + provenance
  files were unique (moved). Delete in item 1.
- 2026-10-16 — outputs location: `Priya/Widefield/LP_analysis` (Priya).
