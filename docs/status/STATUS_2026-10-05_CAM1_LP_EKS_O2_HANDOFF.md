# Handoff — 2026-10-05 (evening): cam1 DLC + LP, cam1 labelling round 2, 3-D spout frame, multi-camera EKS, first O2 run

**START HERE for tracking / 3-D / O2.** Continues `STATUS_2026-10-05_IMAGING_MOVEMENT_HANDOFF.md` (imaging-side
analyses; still the entry point for those) and `STATUS_2026-09-30_LP_OCCLUSION_ROUND4_HANDOFF.md` (cam4 models,
labelling rules). Reasoning: `DECISIONS.md`, the three 2026-10-05 tracking entries (afternoon: cam1 audit / spout
frame / multi-view plan; round 2; evening: LP vs DLC, EKS, calibration, O2). Analysis desktop; M: = MICROSCOPE.

## 0. RESUME CHECKLIST

**Running / waiting at hand-off:**
1. **O2 job 55345148 task 1** — round-3 cam4 DLC on PS93 0814 (2.21 M frames, 88 fps, ~7 h, from ~17:40).
   When `squeue -u ps150` is empty: rsync `out/` back (runbook step 5), then
   `python -m wfield_local.o2_inference collect --name r3_PS93_20261002`. A pipeline test; production waits for the
   next cam4 model.
2. **cam1 LP on the round-2 scan windows** (WSL, `/mnt/c/Users/SabatiniLab/lp_predict_cam1_r2.sh`, log
   `C:\Users\SabatiniLab\lp_stage\lp_predict_cam1_r2.log`, output `C:\Users\SabatiniLab\lp_clips\cam1_round2\lp\`).
   Then: `python -m wfield_local.dlc_hard_frames picks --cam cam1 --lp-dir <that folder>` -> show Priya counts + a
   contact sheet -> `python -m wfield_local.dlc_hard_frames extract --cam cam1` -> `python -m wfield_local.dlc_cam1_guide`.
   The scan (6 sessions, cam1 DLC best-40 + cam1 prior) is cached in `DeepLabCut/Widefield/cam1_round2_scan/`.
3. **Calibrated multicam EKS: done, WORSE than calibration-free** (hidden cam1 tongue error 42-45 vs 17-18 px,
   spikes back, cam1 nose behind the rod 95-155 px off; EKS's likelihood-blind 3-D initialisation lets occluded
   cam1 guesses set the depth). Use calibration-free multicam EKS with an explicit `smooth_param` (~10). See DECISIONS.

**Labelling (student, worksheet `CAM1_GUIDE.html`, archived round-1 page in `_guide_archive/`):** Part 1 cam4
round 4 (in progress now), Part 2 the four empty cam1 folders (PS93 0606, PS92 0820, PS95 0821, PS94 0903),
Part 3 cam1 round 2 (61 pair frames staged; the cam1 hard frames (b) to be added after Priya sees them).

**Then, in order (Priya's plan):** retrain cam1 DLC (+ cam4 after round 4) -> cam1 single-view LP retrain ->
anipose cam1+cam4 + `spout_world` frame -> multi-view LP (`heatmap_multiview_transformer`, explicit `visible`
column with the SPLIT RULE, calibration arms a/b) -> 4-5-member ensembles on O2 -> multi-camera EKS -> 3-D.

## 1. Results today (numbers in DECISIONS)
* **cam1 labels clean** (no partial sessions; nose blank = rod, spout blank = tongue over the tip). 2 frames all
  occluded (`dlc.train.all_occluded`).
* **cam1 DLC best-40** (own project `training/...-orofacial-cam1`): test nose 2.3 / spout 2.9 px good; jaw 11.9 /
  tongue 13.8 px overfit (7 training sessions; PS93 0904 worst).
* **cam1 tongue through the lick:** 85 % of open-mouth frames at p >= 0.4, ~1.5 confident runs per lick; far_L
  (sideways) worst. LP is not better like for like and spikes more raw; DLC and LP disagree > 15 px on 28-43 % of
  confident frames because they pick different ends of the tongue — a label-definition problem.
* **3-D spout frame** (`wfield_local/spout_world.py`): within-session spout distances match the stage command to
  1-2 % (Aug-Sep); stage axes are LEFT-handed -> fit in (ml, ap, dv); per-session residual 0.03-0.08 mm; June ~5 %
  too large -> per-session scale.
* **EKS plumbing works**; auto smoothing over-smooths (set s); posterior variance uncalibrated; cam4 fills cam1.
* **O2 works** (DLC rc13 env, L40S, batch 1, 88 fps).

## 2. Plans / proposed methods (decided or proposed today)
* **Multi-view LP labels — split rule** (decided): hidden in EVERY labelled view -> visible 1; hidden here but
  labelled in the other view -> visible 0; unpaired view rows -> visible 0; `all_occluded` -> visible 1.
* **Calibration in multi-view LP**: arm (a) 3-D augmentation only vs (b) + projection losses at low weight; keep (b)
  only if better on held-out sessions.
* **Calibration in EKS**: the nose supports it (reprojects 8.7 px, like the other parts), but the calibrated smoother
  was worse (see above). Revisit only with a likelihood-weighted 3-D initialisation or with cam2/3 added.
* **June** (proposed): 3-D output corrected by the per-session spout fit; if the calibrated smoother is worse on
  June, refine June extrinsics from the spout positions + paired labels (no new recording).
* **Adaptive sessions** (PS93 0817 / 0820, PS92 0820): per-sample distance from events.csv pre_cue rows (trial ids
  lag on trial_start); only 4 mm far_L defines the frame; shorter far_L samples are a step check. Frame -> event
  join by TIME via the controller clock on the DAQ clock (to build; reuse `daq_trials`).
* **Cutoffs on cam1**: tongue 0.4, jaw / nose / spout 0.6.
* **cam1 round 2 (b) picks**: incomplete_tongue (30-70 px opening band, validated on the scan: tongue seen 17-30 %
  there vs ~90 % above 70 px), tongue_dropout, unsure_tongue, jaw_near_spout, erratic, tricky_spout, plus 2 licks x
  rise / peak / fall per session; LP-disagreement picks once the LP predictions land.

## 3. Pitfalls (new)
1. **PowerShell eats quotes** in inline python and git messages: write scripts / messages to files (`git commit -F`).
2. **`wsl -e bash script` kills nohup'd children** when it returns: launch long WSL jobs via a hidden Windows
   `Start-Process wsl.exe ...` with the work in the foreground.
3. **The cam1 training project is separate** (`-orofacial-cam1`); `dlc_train.train_project(rv, cam)`, `dlc_prior.boxes`
   and `dlc_hard_frames.pose_predictor(cam=...)` must get the camera, or they read cam4's.
4. **`config.donor.project` says N:** — `dlc_prelabel.donor()` rebases it; never type drive letters.
5. **O2:** the share is only on transfer nodes; paths are case-sensitive; `exit` ends an `srun` allocation; `gpu` has
   ~20 cards (try `short` for non-GPU checks, `gpu_requeue` / `gpu_quad` for GPU).
6. **EKS:** no missing data (NaN poisons the filter); auto `smooth_param` unreliable; variance not a quality signal.
7. **A fit pooled over sessions in absolute stage coordinates gives a spurious scale** (regression dilution) — fit
   per session, mouth-relative, in world (right-handed) axes.

## 4. Where things are
| what | where |
|---|---|
| cam1 DLC model | `M:\...\DeepLabCut\Widefield\training\widefield-Priya-2026-09-08-orofacial-cam1\` (best-40) |
| cam1 LP model | WSL `/root/lp/cam1-2026-10-05/models/cam1_occl_uniform_20261005/` (epoch 144) |
| labelled clips (DLC p60 / tongue40, DLC vs LP, disagreement sheet) | `M:\...\Widefield\inference_check_20261005_cam1\` |
| EKS test outputs | `M:\...\Widefield\eks_test_20261005\`; scratch `C:\Users\SabatiniLab\lp_stage\eks_test\` |
| round-2 scan / rows | `M:\...\Widefield\cam1_round2_scan\` |
| O2 bundle | `M:\...\Widefield\o2\r3_PS93_20261002\` (COMMANDS.md), runbook `runbooks/o2_pose_inference.md` |
| worksheet | `M:\...\Widefield\CAM1_GUIDE.html` (`wfield_local/dlc_cam1_guide.py`) |
