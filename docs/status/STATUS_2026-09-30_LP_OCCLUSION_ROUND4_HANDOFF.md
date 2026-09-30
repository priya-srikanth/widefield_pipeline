# Handoff — 2026-09-30: Lightning Pose occlusion fix, labelling round 4 (cam4) + cam1 context, two machine crashes

**START HERE to continue cam4/cam1 orofacial tracking.** Supersedes §00/§0 of
`STATUS_2026-09-28_DLC_LIGHTNING_POSE_HANDOFF.md` (still the background: WSL setup, DALI, MSE→KL, clip history).
Reasoning for every decision: `DECISIONS.md`, the 2026-09-30 entries (four of them, incl. the addenda).
Analysis desktop `MNB-SABA-N40713`. **After the 16:23/16:26 crash-reboots: `M:` = MICROSCOPE,
`N:` = standby (`collaborations\Priya`).** Take paths from `PathResolver`, never type a drive letter.

---

## 0. RESUME CHECKLIST — what is running / what to do next, in order

1. **Round-4 cam4 pipeline (running detached, started 17:13).** `dlc_hard_frames scan` (DLC round 3 + prior
   over 12 trial windows × 6 sessions, ~9 min/session) then `dlc_hard_frames clips` (same windows → one
   lossless .avi per session in `%USERPROFILE%\lp_clips\round4\`). Logs: scratchpad `round4_scan.log`,
   `round4_clips.log`. Cache: `M:\MICROSCOPE\Priya\DeepLabCut\Widefield\round4_scan\` (`<animal>_<date>_dlc.npz`,
   `sessions.csv`). Sessions: PS92 0821 acute, PS92 0824 subacute, PS93 0819 acute, PS93 0826 subacute,
   PS94 0921 chronic, PS95 0917 chronic. Re-running `scan` skips cached sessions.
2. **Then LP on the same windows** (WSL, GPU, **16-frame chunks — never the default 96**, see §3 pitfall 1):
   ```bash
   # per session clip: ffmpeg -nostdin -i /mnt/c/Users/SabatiniLab/lp_clips/round4/<a>_<d>.avi -c:v libx264 -crf 12 -pix_fmt yuv420p <a>_<d>.mp4
   litpose predict /root/lp/cam4-2026-09-28/models/round3_occl_uniform_20260930 <clips>.mp4 \
     --overrides dali.base.predict.sequence_length=16 dali.context.predict.sequence_length=16
   # copy video_preds/<a>_<d>.csv to one folder (on /mnt/c), then on Windows:
   python -m wfield_local.dlc_hard_frames picks --lp-dir <that folder>   # DLC + LP-focused picks -> round4_rows.csv
   python -m wfield_local.dlc_hard_frames extract                        # PNGs, manifest, sync cam4, matched cam1
   python -m wfield_local.dlc_cam1_guide                                  # rebuild the worksheet (Priya asked: rebuild after pull)
   ```
   Run the GPU memory guard around `litpose predict` (scratchpad `lpcue.sh` has it: kill if > 6800 MiB).
   Expected size: ≤ ~7 DLC picks + ≤ 2–4 LP picks per session, each with context (see §2), ~100–130 frames to label.
3. **Priya reviews** `lp_vs_dlc_cue_traces\PS93_20260908\*_cue_traces_DLC_vs_LP[_median5].png` (made 17:25).
   Offered next: a zoomed version (few trials, both models on one axis, 0–1 s) — too dense at 24 × 1000 frames.
4. **Student is labelling cam1** from `CAM1_GUIDE.html` (now the combined cam1 + cam4-round-4 worksheet, same
   filename). 257 cam1 frames to label (16 targets/folder + 66 suggested lick frames).
5. **After round-4 labels come back:** retrain DLC (`dlc_train`, which now drops all-blank rows) AND LP (occlusion
   config); build the LP label export with the all-blank drop (not written yet — §4).
6. Unrelated, still pending from 09-28: `python -m scripts.chronic_stability` once stage 2 for 0928 has landed.

## 1. Findings

### Scientific / tracking
* **LP never learned "not visible" until today.** `training.uniform_heatmaps_for_nan_keypoints` defaults false;
  blanks were dropped from the loss. Epoch-170 KL model: jaw rode the tongue edge (jaw conf 0.99 with tongue
  out vs DLC 0.41) and a tongue appeared at rest (13–35 % of DLC's no-tongue frames). The earlier "better jaw
  confidence than DLC" was this failure — retracted.
* **Fixed model `round3_occl_uniform_20260930` (best ckpt epoch 185):** PS93 jaw conf with tongue out median
  **0.00** (DLC 0.41), false tongue at rest **0.5–1.1 %**, frame 114 tongue 0.85 → 0.00; real tongue still
  detected (median conf 0.99–1.00). Nose/spout agree with DLC 0.8–3.3 px. Val RMSE steady 3.7–3.9 px over the
  last 50 epochs (vs 2–19 px swings in the 09-29 run).
* **LP vs DLC round 3 now ~on par, not clearly better.** Incomplete licks (12–30 px mouth openings): LP 5/9,
  5/10, 21/50 vs DLC 4/9, 4/10, 16/50 (PS92/PS93/PS95) — **neither solves them; round-4 labels are the fix**.
  Full licks ~all. Raw LP jaw has more >15 px jumps than median-filtered DLC (22 vs 7 on PS95). LP tongue
  confident more often (20–33 % vs 14–26 % of frames) and jumpier. Priya: "LP looks better to me on PS93 unseen".
  Priya also saw LP put the tongue off the distal tip on sideways licks (PS95 0907 frame 402) and the jaw on
  the tongue edge there → LP-focused round-4 picks (disagreement) + a sideways-lick tip rule in the guide.
* **Why the 09-29 validation swung 2–19 px:** a few tongue frames flipping between two heatmap peaks on a 40-frame
  val set (measured: test mean 3.9 → 3.0 px without its worst 5 points, all tongue); BatchNorm drift inferred.
* **Spout contact durations** for the cam1 lick targets: 44–128 ms (post-stroke spread wider than pre-stroke 65–95).

### Practical (machine / tooling)
* **GPU memory spill** made the 09-29 run 7 min/epoch: 7.6/8.0 GB dedicated + 5.5 GB "shared GPU memory"
  (Windows pages to RAM instead of OOM). Batch 4+4 → 0.08 GB shared, ~25 s/epoch (17×).
* **Two blue screens today (16:08, 16:23), bugcheck 0x20001 HYPERVISOR_ERROR**, both during `litpose predict`
  in WSL with DALI's default 96-frame predict chunks (7.8 GB on the card). Only crashes in 30 days. With
  16-frame chunks: peak 2.7 GB alone / 4.9 GB beside a DLC job, identical output (median diff 1e-4), no crash
  in three runs. Dumps: `C:\Windows\Minidump\093026-*.dmp`. Training (DALI sequence 4) never crashed.
* **Drive letters swapped after the reboot** (N: MICROSCOPE → standby). A hard-coded `N:` copy created a stray
  tree on standby; copied to M:, byte-verified, stray removed.
* **Early stopping was never on** in LP (`training.early_stopping` defaults false; we only set patience).
* **napari can leave an all-empty row** in CollectedData (cam1 June folder has one) — DLC would train it as
  "everything absent". `dlc_train.drop_unlabelled` now removes such rows.
* **A background task with a time limit kills its children** — the first round-4 scan died after 4/6 sessions
  with all rows in memory. Long jobs: detached `nohup sh -c "…" &` + per-session cache.
* **`ffmpeg` reading a piped script consumes it** — use `-nostdin`. WSL cannot mount the SMB drives
  (`mount -t drvfs M:` fails) — stage through `/mnt/c`.
* `litpose recommend` exists and was run (vits_dino 256 px, AdamW 5e-5, log_weight 11, uniform heatmaps on).

## 2. Decisions (all in DECISIONS.md 2026-09-30)
* LP config: `uniform_heatmaps_for_nan_keypoints: true`; `losses_to_use: [temporal]` (PCA off — no confidence
  masking); temporal `prob_threshold` 0.5; batch 4+4; `early_stopping: false` explicitly; KL kept; resolution
  kept (not the recommender's 256 px — incomplete-lick tongue tip is a few px); log_weight 5 kept for now.
* NVIDIA "Sysmem Fallback Policy" left at default (global; would crash 2pRAM jobs instead of slowing them).
* **TARGET vs CONTEXT frames.** Target = label completely (blank = occluded). Context = neighbours to scrub;
  blank, or labelled completely, never partly. Suggested ("context_label") = every ~3rd context frame, flexible
  — "the goal is the toughest frames". Manifest `category` is the only record (`round4`, `context`,
  `context_label`, originals). Filenames stay `img%07d.png`.
* Blank semantics: DLC = absent always → all-blank rows dropped; LP single-view = occluded → all-blank rows must
  be dropped at export; LP multi-view = keep the row, `visible`=0, needs a `visible` column + shared image names;
  anipose needs no paired labels. Matching cam1 frames are EXTRACTED for every new cam4 moment but kept off the
  student's worksheet (`_frame_staging_unassigned/`).
* cam1: 16 targets/folder unchanged (the cam4 pairing); context ±4 around measured contact onset AND end of each
  lick target in the 11 unlabelled folders (330 frames); 66 promoted to suggested.
* cam4 round 4: six post-stroke sessions (2/epoch); DLC kinds incomplete_tongue (cap 4, ±4), erratic_tongue
  (1, ±4), erratic_jaw (1, ±2), tricky_spout (1, none); LP kinds in priority order disagree_tongue, disagree_jaw,
  lp_erratic_tongue, lp_erratic_jaw (1 each). No full-lick sequences (Priya).
* One worksheet (`CAM1_GUIDE.html`) for cam1 + cam4 round 4; rebuilt after the round-4 frames are pulled (no
  scheduled task — Priya).
* GPU prediction in WSL only with 16-frame chunks + memory guard.
* Filtering: compare like for like — DLC's `filterpredictions` default (median, window 5) applied to both.
  LP's EKS (ensemble of ~4–5 seeds) is the stronger option, planned after round 4 (~8 h GPU).

## 3. Pitfalls to avoid
1. **Never run `litpose predict` with the default DALI predict chunk (96)** on this 8 GB card — it crashed the
   machine twice. `--overrides dali.base.predict.sequence_length=16 dali.context.predict.sequence_length=16`.
2. "100 % GPU, no crash" is not health on Windows — check `Get-Counter '\GPU Adapter Memory(*)\Shared Usage'`.
3. Type no drive letters; they move on reboot. `net use` shows the mapping.
4. A context frame labelled partly is read as "other parts occluded" by both models.
5. Never stage frames nobody should label under `_frame_staging/cam1_*` — `sync_frames` copies them to the student.
6. An LP config key that is only a partner of a switch does nothing (`early_stop_patience` without
   `early_stopping: true`; `patch_mask: null` crashes). Check how LP reads a key before relying on it.
7. LP's epoch-best checkpoint is chosen on `val_supervised_loss`; with occlusion on that includes the occluded
   targets — do not re-select by val RMSE (visible points only).
8. Long GPU jobs: detached + per-session cache; a time-limited background task kills them.
9. The repo test `test_no_hardcoded_machine_paths` blocks pushes on typed paths — use `Path.home()`/PathResolver.
10. Other windows leave uncommitted edits (today: `scripts/chronic_stability.py`, `nightly_figs.py`,
    `PRELIM_DATA_VLS_STROKE.md`, `rotation_maps.py`, `region_reference_contrast.py`): commit only your own files.

## 4. To do (not started)
* LP label export for single-view training applying the all-blank drop (before the round-4 LP retrain).
* Multi-view export: per-keypoint `visible` column (2/1/0), shared image name per DAQ instant; verify a
  not-labelled view row trains correctly (checked in LP code only).
* EKS ensemble after round 4; log_weight 11 comparison; `freeze_until_epoch` 60; vits_dino at 384 px.
* Zoomed cue-trace figure; split LP–DLC tongue disagreement by spout position.
* Guide line for sideways-lick tongue tip is in the round-4 "disagree_tongue" text only — add to
  `LABELLING_GUIDE.html` too when it is next regenerated.
* cam1/cam4 nose offset (241 same-instant frames) once cam1 is labelled.

## 5. Where things are
| what | where |
|---|---|
| LP occlusion model | WSL `/root/lp/cam4-2026-09-28/models/round3_occl_uniform_20260930/` (best `epoch=184-step=15170-best.ckpt`) |
| its clip predictions + labelled videos | `M:\…\DeepLabCut\Widefield\lightning-pose\cam4-2026-09-28\eval_occl_20260930\` |
| epoch-170 (pre-fix) model outputs | `…\eval_ep170_20260930\` |
| DLC round 3 review clips | `M:\…\DeepLabCut\Widefield\inference_check_20260928_round3\labeled_clips\` |
| cue-trace comparison | `M:\…\DeepLabCut\Widefield\lp_vs_dlc_cue_traces\PS93_20260908\` (`scripts/pose_cue_traces.py`) |
| worksheet | `M:\…\DeepLabCut\Widefield\CAM1_GUIDE.html` (`wfield_local/dlc_cam1_guide.py`) |
| round-4 cache / rows | `M:\…\DeepLabCut\Widefield\round4_scan\` |
| matched cam1 frames (unassigned) | `M:\…\DeepLabCut\Widefield\_frame_staging_unassigned\` (after `extract`) |
| code | `wfield_local/dlc_hard_frames.py`, `dlc_context_frames.py`, `dlc_train.drop_unlabelled`, `scripts/lp_eval_cpu.py`, `configs/lightning_pose_cam4.yaml` |
