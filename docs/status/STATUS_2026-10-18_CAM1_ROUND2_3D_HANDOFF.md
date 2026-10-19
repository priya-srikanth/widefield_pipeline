# Handoff — 2026-10-18: cam1 round-2 retrain, cam1 clean traces, the 3-D apparatus mouth, cam4 eps35

**START HERE for cam1 tracking / cleaning / 3-D.** Continues `STATUS_2026-10-16_O2_PRODUCTION_OUTPUTS_HANDOFF.md` (cam4
production, LP_analysis) and the cam1 background in `STATUS_2026-10-05_CAM1_LP_EKS_O2_HANDOFF.md`. The model register with
every setting, reason and score is **`docs/POSE_MODELS.md`** (cam4 sections 1-4, **cam1 Part B**, **evaluation plan + status
Part C**, **3-D / cam1 cleaning Part D, D1-D7**). Waiting-on-Priya list: `docs/status/OPEN_QUESTIONS_PRIYA.md`.
Analysis desktop; M: = MICROSCOPE.

## 0. RESUME CHECKLIST

**UPDATE 2026-10-19 (supersedes everything below where they differ):**
* DECIDED: cam4 production stays LP r4 (DECISIONS 2026-10-19; eps35 and ViT-B not adopted). cam1 production model = **LP
  ctx-semi** (`mhcrnn_semi_640_b4`; POSE_MODELS Part B: most faint licks after cleaning, no emergence lag; plain LP c1r2 lags
  the emerging tip, eps40 loses acute faint licks). No new cam1 labelling round now (targeted round kept as a known fix:
  water drops, part-out tongue, faint licks cam1 misses). cam1 cleaning: max gap 24 ms, veto off (D8). Nose: blank at the
  centre positions by design (rod occludes it).
* RUNNING on O2: cam1 production `o2/lp_cam1_ctx_prod_20261019` (137 cam1 videos, `o2_inference bundle --kind lp --cam cam1`),
  smoke job + array (afterok), submitted ~07:40 10-19; all videos were already on scratch. Array **56830841**: smoke 145 fps on L40S
  (median video ~3 h, longest ~4.5 h); only ~16 L40S exist in gpu_quad (we held 10), so Priya opened the LIVE job to any
  GPU for the PENDING tasks 11-137 (`scontrol update JobId=56830841 Gres=gpu:1 TimeLimit=16:00:00`, 10-19, verified;
  running tasks 1-10 refused the change and stay L40S / 10 h; V100 exclude list kept; throttle 24). The
  bundle's run_pose.sbatch still says l40s / 10 h (the live job was edited, not the file). If tasks fail on a new node
  type, add it to ExcNodeList. Next: smoke log fps -> run time;
  copy back with COMMANDS.md step 5; `python -m wfield_local.o2_inference collect --name lp_cam1_ctx_prod_20261019 --compare
  PS93:20260814=M:/MICROSCOPE/Priya/DeepLabCut/Widefield/o2/lp_train_cam1_r2_20261018/models/mhcrnn_semi_640_b4/video_preds/PS93_20260814_cam1.csv`
  -> `session_poses/<a>_<d>_cam1_full/`. Then: per-session 3-D spouts + apparatus mouth from both cameras, READY:
  `python -m scripts.apparatus_mouth production --jobs 4` (dlc env; reads both O2 bundles directly, split days handled,
  June cam4 x0.885 about the principal point -> `session_poses/apparatus_mouth_production{,_chosen}.csv`). cam1's principal
  point (200, 200) CHECKED 10-19: pinned from the sensor ROI, correct (POSE_MODELS D5); v36 June sessions need NO re-run
  (the scaling centre cancels in every 2-D measure). Then cam1 per-session outputs (cleaned traces + cam4-timed lateral landing).
  Per-frame 3-D, READY (POSE_MODELS D10): `python -m scripts.pose3d production --jobs 4` (dlc env) -> `session_poses/pose3d/`
  frames + per-lick reach / miss / lateral error against the target spout; tested on the PS93 clips (`pose3d clips`).
* Local GPU free (LP review inference stopped 10-19; an orphaned queue loop from the previous session had kept running --
  pitfall 17).
* 17. **Background loops survive a session restart even when reported "stopped"**: after a restart, find leftover queue
  loops by command line (e.g. `Get-CimInstance Win32_Process` filtered on the script / clip path) and stop the loop
  BEFORE its child, or it starts the next item.

**UPDATE 2026-10-18 evening (superseded by 10-19 above where they differ):**
* O2 `56718665_1` cam1 eps40: trained 300 / 300, then CRASHED in LP's post-training labelled-frame eval (DataLoader workers
  died together, core dumps; MaxRSS 17 GB of 64 -- node / shared memory, not the model). Recovered WITHOUT retraining by a
  predict-only job `predict_eps40.sbatch` in the bundle (`litpose predict ... --skip_viz` + `touch DONE`, excludes
  compute-g-17-161); submitted by Priya ~19:00. No labelled-frame pixel error for eps40 (not needed).
* O2 `56700781_1` cam4 ViT-B: trained 300 / 300 (val loss still falling, 0.19 vs eps35 best 0.11), predicting the 8 eval videos
  (2 done at 17:11), its sbatch predates `--skip_viz` so labelled videos are rendered after -> DONE late.
* O2 `56709413_2` cam1 ctx-semi: epoch 173 / 300 at ~18:45, 2.9 min / epoch -> training ends ~01:00, DONE ~02:00.
* Local GPU: the LP review inference was STOPPED at 18:35 after clip 1 (eps35 PS92 0821 done, 2 h 17 min per clip) so cam1 DLC
  r2 could run on June 0606 x 4 + PS94 0814 / 0821 / 0907, then r1 PS93 0908 (log `lp_stage\cam1_dlc_eval_june_ps94_20261018.log`;
  r1 on PS94 dropped -- r2 clearly better). The LP review restarts automatically after DLC prints ALL DONE, 0821 sessions first
  (both models), so a 3-session review is possible early. Full 12 clips ~ 16-25 h.
* Auto-scoring watcher (session scratchpad `score_when_ready.sh`): when Priya's rsync brings `PS93_2026{0814,0821,0908}_cam1.csv`
  for eps40 -> `cam1_model_compare ... --tag _eps40_1018`; `PS93_2026{0814,0821}_windows.csv` for ViT-B -> `model_lick_compare
  ... --tag _vitb_1018` (with LP r4, eps35, ctx-semi, Priya's review). If the session is gone, run those by hand (§4).
* Copy-back now with `--exclude 'core.*'` too (the crashed job's 4 x 3.7 GB dumps were copied once; deleted on both sides).
* New this evening: cam1 max gap 24 ms + veto off (POSE_MODELS D8, re-rendered clip `..._gap24.mp4` for Priya's QC); cam1
  lateral-landing readout incl. cam4-timed peaks (D9; PS93's apparent rightward shift = 0814 baseline artefact, PS93 shifts
  ipsi acutely over all sessions; cameras agree once the tongue is > 30 px out; cam1 also sees tiny licks cam4 does not
  count -- 31-41 % of at-the-lips cam1 events have no cam4 lick: put them in the cam1 blind review).

**Running at hand-off (check these first):**

| what | where | how to check | then |
|---|---|---|---|
| O2 `56709413_2` -- cam1 LP ctx-semi 640 (`mhcrnn_semi_640_b4`) | `/n/scratch/users/p/ps150/o2/lp_train_cam1_r2_20261018` | login node: `squeue -u ps150`; `ls .../models/*/DONE` | Priya copies back (below); score with `scripts.cam1_model_compare` |
| O2 `56718665_1` -- cam1 LP eps40 (`heatmap_640_b4_eps40`) | same bundle | same | same |
| O2 `56700781_1` -- cam4 ViT-B DINOv2 640 rerun | `/n/scratch/users/p/ps150/o2/lp_train_20261016` | same | score with `scripts.model_lick_compare ... --review` like eps35 |
| local GPU -- DLC c1r1 on the PS93 cam1 clips (0814 / 0821 / 0908) | log `C:\Users\SabatiniLab\lp_stage\cam1_dlc_eval_20261018.log` -> `session_poses/PS93_<d>_cam1/dlc_r1/` | `grep "^==\|ALL DONE"` the log | automatically followed by the next row |
| local GPU, queued -- DLC c1r2 on June 0606 cam1 clips (4 animals) + PS94 0814 / 0821 / 0907 (r2 then r1) | log `lp_stage\cam1_dlc_eval_june_ps94_20261018.log` (starts when the row above prints ALL DONE) | same | June: `python -m scripts.apparatus_mouth clips PS9x:20260606 ...` (dlc env) for the June cam4 variants; PS94: rescore |

| local GPU -- cam4 **LP r4 and LP r4 eps35** (Windows-native) on PS92 / PS94 / PS95 x 0814 / 0821 cam4 clips, for the eps35 BLIND REVIEW | clips copied to `C:\Users\SabatiniLab\lp_stage\review_clips_1018\<s>_windows.mp4`; predictions -> `lp_stage\lp_models_win\{r4_eps35_o2,r4_heatmap_20261013}\video_preds\<s>_windows.csv` (eps35 first, then LP r4) | `ls` those video_preds folders (12 csvs when done) | build the review (§5 item 0) |

The local runner is `C:\Users\SABATI~1\AppData\Local\Temp\claude\...\scratchpad\cam1_dlc_eval.py` (session scratchpad, not the
repo): `python cam1_dlc_eval.py <scratch dir> r2/PS94_20260814 r1/...` runs DLC round 1 / 2 on `session_poses/<s>_cam1/windows.mp4`
through `config_eval_r{1,2}.yaml` in the cam1 training project. If the scratchpad is gone, the loop is ~20 lines: DLC
`analyze_videos(config_eval_rX.yaml, [windows.mp4], shuffle=1, trainingsetindex=0, destfolder=<clip>/dlc_rX, save_as_csv=True)`.

**Copying O2 outputs back (Priya runs; transfer node `ssh ps150@transfer.rc.hms.harvard.edu`):**
```
S=/n/scratch/users/p/ps150/o2
F=/n/files/Neurobio/MICROSCOPE/Priya/DeepLabCut/Widefield/o2
rsync -ah --partial --exclude labeled_videos --exclude 'core.*' $S/lp_train_cam1_r2_20261018/models/ $F/lp_train_cam1_r2_20261018/models/
rsync -ah --partial --exclude labeled_videos --exclude 'core.*' $S/lp_train_20261016/models/ $F/lp_train_20261016/models/
rsync -ah --partial --exclude 'tmp*' $S/lp_r4_prod_20261015/out/ $F/lp_r4_prod_20261015/out/      # production inference, if more finished
```
A model is finished when `models/<name>/DONE` exists; its eval predictions are `models/<name>/video_preds/*.csv` (they can be
copied before DONE -- `litpose predict` renders labelled videos after predicting, which is what held DONE back).

**Then, in order:**
1. Score every new cam1 model on PS93 (+ PS94 when its DLC is done) -- command in §4; add to the cleaned multi-model clip.
2. LP c1r1 (the 10-05 WSL model) on the new cam1 clips -- WSL, `dali.base.predict.sequence_length=16` + memory guard (never the
   default 96: two blue-screens). Then the cam1 candidate set is complete.
3. June 3-D test (§2 D5): which June cam4 correction makes the triangulation work.
4. Priya: cam1 cleaned clip QC (centre-position bridging, water) -> settle `orofacial_clean_cam1`; cam4 eps35 production decision.

## 1. What was done today (all committed and pushed)

* **cam1 round-2 labels complete** (634 frames / 21 sessions; Part 2's four empty folders + Part 3's 254 targets; 3 blanks).
* **DLC cam1 round 2 trained** locally: `training/...-orofacial-cam1`, iteration-1 `trainset72`, best-30 (round 1 kept at
  iteration-0). Held-out test (7 sessions, every epoch): nose 2.6 / jaw 9.9 / tongue 12.3 / spout 1.9 px; no post-stroke penalty.
* **cam1 LP round 2 on O2**: bundle `o2/lp_train_cam1_r2_20261018` (heatmap 640 b4 = the LP r4 recipe; ctx-semi 640 b4; add-on
  eps40). `scripts/o2_lp_train_bundle.py --cam cam1`, add-ons `--only <model> --configs-only`, predict `--skip_viz` from now on.
* **cam4 add-on LP r4 eps35** trained on O2 and scored (POSE_MODELS "cam4 LP r4 eps35"): precision 96 / 94 % vs LP r4's 82 / 88,
  recall 99.5 / 99.2, share of real faint licks 89 / 97 vs 100 / 97.
* **cam1 eval clips**: `scripts.session_poses clip --cam cam1` cuts the SAME trials as the cam4 clips: PS93 0814 / 0821 / 0908,
  PS94 0814 / 0821 / 0907 (60 trials), June 0606 x 4 animals (24 trials).
* **Scoring / review tools**: `scripts/cam1_model_compare.py` (agreed cam1 metrics), `scripts/model_label_clips.py` (raw, any
  models side by side), `scripts/cleaned_label_clips.py --cam cam1` (cleaned), `scripts/cam1_clean_qc.py` (mouth / jaw /
  tongue_mouth QC sheets), `scripts/apparatus_mouth.py` (3-D mouth).
* **Repo fixes**: frame manifest moved (pitfall 1); per-camera cleaning overlay; `spout_frame` cam1 functions; a spout_frame
  docstring that had cam1's view backwards was corrected.

## 2. Decisions (Priya, 2026-10-18)

| decision | detail / where |
|---|---|
| retrain cam1 as DLC + LP, LP on O2 only | Part B |
| LP temporal epsilon measured per camera | cam1 [10, 15, 40, 10] (p99 of genuine motion, B3); cam4 add-on [10, 10, 35, 10] |
| cam1 evaluation metrics | Part C (epoch gate; tongue at DAQ contacts; tip teleports > 60 px + coverage; cam1 vs cam4 onset timing; jaw coverage + jumps; blind visual review; pixel error sanity; practicality) |
| 3-D is an analysis output | Part D route: cam1 cleaning -> multi-view LP candidate -> anipose vs calibration-free EKS |
| **cam1 mouth = the L / R spout-line intersection, no offset** (`spout_frame.from_medians_lr`) | apparatus-only, so post-stroke tongue changes cannot move it (the tongue-path point was rejected for that reason; kept as QC). D6 |
| adaptive (shorter far_L) trials stay in the cam1 mouth | the stage moves along the same ray -> same line (pinned by a test) |
| jaw-off-tongue rule OFF on cam1 | Priya's QC: those frames are mostly real jaw |
| cam1 needs no June rescale | cam1 June spout ruler 0.995 of Aug+ (cam4 12.9 % larger), D3 |
| non-curated sessions are in no analysis | checked: every orofacial cohort analysis reads `analysis_kit.curated_sessions` |

**Config (proposed, pending Priya's QC of the cleaned clip): `configs/defaults.yaml` `orofacial_clean_cam1`** -- jaw lk 0.6, jaw
y ceiling 160 px, PS93 flip rule off, jaw-off-tongue off, between-lick veto near 40 px / speed 1.7 px. Everything else = cam4's
values (measured to remove / fill about the same on cam1, D7). A SEPARATE block on purpose (pitfall 4).

## 3. Findings

1. **cam1 sees the centre spout line end-on** (close->far 3.4 px vs L / R 48; cam4 56 / 66 / 68): cam1 sees LR and DV, hardly AP.
   Centre-spout angles are ill-defined on cam1; cam1's axis = the L / R bisector. Good for 3-D (the cameras see AP very differently).
2. **The apparatus mouth moves with the spout set point** (D5): triangulated spouts fit the command to 0.05 mm; the commanded mouth
   projected into cam1 moves 24 px between PS93 days, the same as the 2-D cam1 spout intersection, because the gui_config mouth set
   point differs by up to 0.70 mm between days (nose moved 0.78 mm relative to it). **Spout set-point jitter across days** -- noted
   in POSE_MODELS: spout-relative measures unaffected; mouth-relative ones (protrusion, in-mouth cutoff, smallest licks) exposed.
3. **cam4 re-set during 2026-08-05**: PS94 13:10 scale 0.80, PS95 16:04 0.77, PS92 / PS93 from 18:21 at 1.00; the 08-05
   calibration video (17:58) shows the Aug+ setup, so it cannot fix June. v36 leaves PS94 / PS95 0805 unscaled -- non-curated, no
   result affected (FYI in OPEN_QUESTIONS).
4. **June 3-D**: v36's scaling is about the image origin -- right for 2-D px measures, not valid triangulation input as is. Test
   pending (variants: raw / x0.885 about origin, principal point, cam4 mouth), cam4 June from the O2 output.
5. **cam1 geometry from below**: the tongue comes out DOWN the image over the chin (an earlier note said "up" -- wrong, corrected);
   the jaw opens down (dy p99 139 px); the tongue's reach is foreshortened (distance from the mouth ~0.55 x cam4's), so cam1 px
   thresholds measured from the mouth must come from cam1 distributions (cam4's 20 px in-mouth cutoff ~ 10-11 px on cam1).
6. **Water drops make false cam1 tongue detections** (static points at the bottom of a hanging drop, late sated trials; DLC c1r2
   23 % static in no-contact trials). Spout tip itself unaffected (< 2 px). Fix list in D4.
7. **First cam1 scores (PS93, DLC c1r2 vs LP c1r2 heatmap)**: LP finds more contacts (100 / 99.2 / 99.2 %) and has the steadier,
   better-covered jaw; teleports vary by session for BOTH models (DLC 9.5 / 4.0 / 17.4, LP 5.1 / 17.4 / 12.9 per 10k), so the
   acute LP rise is session noise, not an epoch effect (confirm on PS94).
8. **cam1 cleaned clip** (`session_poses/PS93_20260814_cam1/eval/cleaned_labels_cam1_dlc2_lp2.mp4`): at the CENTRE positions the
   cleaning can bridge a between-lick gap (trial 251 +188 ms: interpolated tongue under the spout, mouth closed) -- the centre
   spouts sit on the cam1 mouth, so only the veto's speed test is left there.

## 4. Commands

```
# score cam1 models (locanmf env; use the DLC .csv -- no pytables here)
python -m scripts.cam1_model_compare PS93:20260814=pre PS93:20260821=acute PS93:20260908=chronic \
   --model "DLC c1r2={d}/dlc_r2/*.csv" --model "DLC c1r1={d}/dlc_r1/*.csv" \
   --model "LP c1r2 heatmap=M:/MICROSCOPE/Priya/DeepLabCut/Widefield/o2/lp_train_cam1_r2_20261018/models/heatmap_640_b4/video_preds/{s}_cam1.csv" \
   --cam4 "cam4 DLC r3={c4}/windows_DLC.csv" --tag _<date>
# cleaned multi-model clip, same 16 trials (2 / position, +1 per centre position, 2 late sated)
python -m scripts.cleaned_label_clips M:/.../session_poses/PS93_20260814_cam1 --cam cam1 --model "NAME=CSV" ... \
   --trials 1 508 12 507 26 251 494 65 457 8 439 22 264 428 568 575 --win -0.3 2.0 --fps 50 --tag <tag>
# cam4 models (LP r4 vs O2 candidates, with Priya's lick review)
python -m scripts.model_lick_compare PS93:20260814 --model "LP r4=C:/Users/SabatiniLab/lp_stage/r3_vs_r4_lp/r4heatmap/PS93_20260814_windows.csv" \
   --model "NAME=M:/.../o2/lp_train_20261016/models/<model>/video_preds/PS93_20260814_windows.csv" \
   --review C:/Users/SabatiniLab/lp_stage/lick_review/review_scored.csv --tag _<tag>
# apparatus mouth / June test (dlc env: aniposelib)
python -m scripts.apparatus_mouth clips PS93:20260814 PS92:20260606 ...
```

## 5. To-do

0. **eps35 blind review (Priya agreed 10-18).** When the 12 csvs above exist:
   `python -m scripts.lick_review_candidates PS92:20260814 PS92:20260821 PS94:20260814 PS94:20260821 PS95:20260814 PS95:20260821
   --model "LP r4=C:/Users/SabatiniLab/lp_stage/lp_models_win/r4_heatmap_20261013/video_preds/{s}_windows.csv"
   --model "LP r4 eps35=C:/Users/SabatiniLab/lp_stage/lp_models_win/r4_eps35_o2/video_preds/{s}_windows.csv"
   --out C:/Users/SabatiniLab/lp_stage/lick_review_eps35` (tiers = exact model sets: LP r4 only / eps35 only / both, by
   position; hidden contact-lick controls; key.csv kept off the page). Page: rebuild from `lp_stage/lick_review/lick_review.html`
   (the 10-14 review; answers in its `lick_review` collection) with the new candidates and a NEW collection name, publish
   (load the artifact-design / artifact-capabilities skills first), give Priya the link; score precision / share of real per
   model as in the 10-14 review (POSE_MODELS section 3). Decision rule to discuss: switch only if eps35 keeps its precision
   gain across animals without losing real faint licks; if it wins, re-infer cam4 together with the cam1 production run on O2.
   Priya 10-18 on the four-model cam4 clips: "they don't seem meaningfully different" -- consistent with the numbers (the
   difference is confined to faint no-contact licks).
1. Score cam1 ctx-semi + eps40 (O2), DLC c1r1, LP c1r1 on PS93 + PS94; per-epoch tables; cleaned multi-model clip; blind cam1
   review sheet for Priya (include late sated trials -- water). Then Priya picks the cam1 model.
2. cam1 cleaning: Priya's QC of the cleaned clip; a centre-position bridging fix if it is common; measure a static-tongue water
   guard's cost before proposing it; re-check every cam1 value on the chosen model.
3. June 3-D: run `apparatus_mouth clips` on the four June sessions once their cam1 DLC is done; pick the June cam4 correction.
4. Next labelling round (cam1 and cam4): water-drop frames with the tongue blank (hard negatives).
5. Quantify spout set-point jitter across the cohort (nose in each session's mouth-centred frame, D5 method).
6. cam4: Priya decides on LP r4 eps35 vs LP r4 (and ctx-semi 640, ViT-B when scored); check the cam4 review's "false detections
   on the spout rod" for water drops.
7. Production cam1 inference on O2 once a model is chosen: generalise `o2_inference` bundles to `--cam cam1` (window builder reads
   cam4 templates). O2 scratch purges files untouched ~45 days -- the videos copied 10-06 may go around 11-20.
8. 3-D: multi-view LP candidate (round-2 cam1 + round-4 cam4), anipose vs calibration-free EKS, the 3-D mouth as the shared origin.

**Clips made today (for Priya):** cam1 cleaned DLC c1r2 vs LP c1r2, 16 trials -- `session_poses/PS93_20260814_cam1/eval/
cleaned_labels_cam1_dlc2_lp2.mp4`; cam1 raw DLC c1r2 -- `.../eval/labeled_dlc_c1r2.mp4`; cam4 cleaned LP r4 / eps35 / ctx-semi /
ResNet-50 -- `session_poses/PS93_2026081{4,21}/r4_eval/cleaned_labels_cam4_r4_eps35_ctx_rn50.mp4`. QC sheets: `.../eval/qc_*.png`.
**eps35 on Windows**: `lp_stage/lp_models_win/r4_eps35_o2` (O2 checkpoint; data_dir -> `lp_cam4_r4_win`, `losses_to_use: []`);
validated against O2's own predictions (median 0.04-0.11 px, confidence agreement 99.8-100 %).

## 6. Pitfalls (new today)

1. **The frame manifest moved** to `<DeepLabCut/Widefield>/frame_manifest.csv` (`dlc_frames.manifest_path`); the 10-16
   clean-up of `_frame_staging` had taken it, and `dlc_train.session_epochs` silently returned `{}` -> an epoch-blind split. It
   now raises. `side_assignment.csv` moved too.
2. **DLC resets `project_path` to the config's own folder**: a config copy elsewhere points at an empty project ("Couldn't find
   any shuffles"). The eval configs live in the cam1 training project (`config_eval_r1.yaml`, `config_eval_r2.yaml`).
3. **locanmf has no pytables**: read DLC `.csv`, not `.h5` (the dlc env has both).
4. **`orofacial_outputs.params_digest` hashes the whole `orofacial_clean` block** and the cam4 production run refuses to continue
   when it changes -> per-camera values go in `orofacial_clean_<cam>` (test pins the digest).
5. **`litpose predict` renders labelled videos by default**: hours on 220k-frame clips, DONE held back. `--skip_viz` in the
   bundle now; `--exclude labeled_videos` when copying.
6. **Reading a prediction file while DLC is still writing it** gives a short file (an IndexError later): check `len(P) >= len(idx)`.
7. **The tongue "> 40 px" jump count is genuine motion on cam1** (40 = p99): teleports are > 60 px.
8. **Shared clone with the other window**: commit only your own files; a push can land through the other window's push ("cannot
   lock ref ... is at <your commit>" is harmless); the pre-push suite takes 11+ min, longer while the GPU job and clip cutting run.
9. **Python heredocs that write Python code**: `"\n"` inside them became real newlines in the target file -- edit such lines with
   the Edit tool.
10. **`squeue` truncates job names** (`lp_train_cam1` shows as `lp_train`); use `sacct --format=JobID,JobName%30,...`.
11. **cam1 labels mark the spout tip's BOTTOM edge, cam4 the top**: constant offset in 3-D (absorbed by the fit), ~15-20 px toward
    the jaw for the cam1 mouth.
12. **Python strings holding Windows paths**: `"C:\Users..."` in a non-raw string is a \U escape (SyntaxError) -- use raw strings or the Edit tool.
13. **A bundle job that dies after training must NOT be resubmitted as is**: `train.sbatch` skips training only when
    `train_status.json` says COMPLETED; a crash in the post-training eval leaves EVALUATING -> a resubmit RETRAINS from scratch.
    Use a predict-only sbatch (`predict_eps40.sbatch` pattern).
14. **Local GPU is 8 GB**: LP Windows predict (~7 GB) + DLC analyze_videos together -> DLC CUDA OOM (crashed r1 PS93 0908).
    Run them one after the other. "Pausing" LP = stop + restart: suspended processes keep their GPU memory; the predict loop
    skips finished clips, the clip in progress is lost.
15. **cam1 geometry for any lick measure**: image up / down = dorsal / ventral, spouts nearly level with the mouth, the tongue
    reaches DOWN 50-100 px; tongue-vs-spout angles are meaningless on cam1, and the most ventral frame is not peak
    protrusion -- time peaks from cam4 (same 4 ms DAQ grid, exact frame match). cam4 is a FRONTAL view, not a side view.
16. **Single-day baselines mislead**: PS93's close_center lick side is bimodal across sessions (0814 is an odd pre day). Compare
    epochs over all curated sessions, never one clip day each.
