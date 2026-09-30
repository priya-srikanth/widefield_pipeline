# Handoff — cam4 orofacial tracking: DLC round 3 in use; Lightning Pose (KL) training in WSL2; cam1 labelling under way

**START HERE to continue the DLC / Lightning Pose work.** Written 2026-09-28 (evening) on the analysis
desktop (`MNB-SABA-N40713`, profile `analysis_desktop`, RTX 5060 8 GB, `N:` = MICROSCOPE). Everything
below is either in git, on the DLC share `N:\MICROSCOPE\Priya\DeepLabCut\Widefield\` (Mac:
`/Volumes/Neurobio/MICROSCOPE/Priya/DeepLabCut/Widefield/`), or named here with its path.

Decision owner: Priya. Her standing choices that shape this: *"let's try using lightning pose, keeping
open the ability to incorporate the additional cameras"* (2026-09-26); WSL2 before O2 as the first try;
prior ON for tongue and jaw only (2026-09-28); train/test split by whole sessions (do not change).

---

## 00. UPDATE 2026-09-30 — READ THIS FIRST; it supersedes parts of §0 below

Reasoning for all of it: `DECISIONS.md`, the two 2026-09-30 entries. In short:

1. **The 09-29 KL run is superseded.** It never learned "not visible" (jaw rides the tongue, tongue at
   rest; `uniform_heatmaps_for_nan_keypoints` defaults false). Stopped 14:00 at ~epoch 235; its best
   checkpoint (epoch 170) is kept as `models/eval_ep170_snapshot/` and its CPU predictions on the three
   review clips are on the share at `lightning-pose/cam4-2026-09-28/eval_ep170_20260930/` (README there).
2. **Current run: `models/round3_occl_uniform_20260930`** (WSL, launched 14:00; log
   `scratchpad\lp_train_occl.log`): blank = occluded, PCA off, temporal prob_threshold 0.5, batch 4+4,
   ~25 s/epoch, runs all 300 epochs (early stopping is OFF by design — it had never been on). Evaluate
   as in §0a step 4, adding: jaw confidence while the tongue is out (must drop, like DLC's 0.41) and
   confident tongue at rest (must be near 0). CPU prediction recipe:
   `scripts/lp_eval_cpu.py` (patches `torch.load` to CPU; the snapshot's config needs
   `losses_to_use: []` because semi-supervised configs refuse to build without CUDA).
3. **GPU memory spill** was the reason for 7 min/epoch; check shared GPU memory in the first minutes of
   any run (`Get-Counter '\GPU Adapter Memory(*)\Shared Usage'`).
4. **Labelling, target vs context frames.** cam1: 330 context frames added to the 11 untouched folders
   around measured contact onset/end; `CAM1_GUIDE.html` regenerated (targets by slider position). cam4
   round 4 (incomplete licks ±4, erratic tongue ±4, erratic jaw ±2, tricky spout): picker still TO BUILD
   — needs the GPU once LP finishes. Remaining TO DO list: end of the second 2026-09-30 DECISIONS entry.
5. `dlc_train.stage` now drops all-blank label rows (`drop_unlabelled`); keep it that way.

## 0. RESUME CHECKLIST (rewritten 2026-09-29 ~13:00, for the session that picks this up)

**State in one paragraph.** DLC round 3 (iteration-2 / snapshot-best-160) is the cam4 model in use, prior
ON for tongue+jaw. A Lightning Pose semi-supervised run on the round-3 labels is **training in WSL2 and
learning** (§1d: the first run never learned — MSE heatmap loss stalls on our 160×160 maps — fixed by
`heatmap_loss_type: kl`). The student is labelling **cam1** from `CAM1_GUIDE.html` (§0b). Nothing else of
ours is running on the box (the other `python.exe` processes are 2pRAM jobs — never kill python by name).

### 0a. The Lightning Pose run

1. **Run:** `litpose train config.yaml --output_dir /root/lp/cam4-2026-09-28/models/round3_semisup_kl_20260929`,
   launched 2026-09-29 08:54, anchored by a background Windows-side `wsl.exe`. Log:
   `C:\Users\SabatiniLab\AppData\Local\Temp\claude\C--Users-SabatiniLab\cdd1e016-6bb3-4346-8600-f08116ccf61d\scratchpad\lp_train_round3_kl.log`
   (progress uses `\r`: `tr -d '\0' < log | tr '\r' '\n' | grep -E "^Epoch" | tail -1`). Alive = `nvidia-smi`
   ~7.9 GB / 100 %, epoch counter advancing (~7.1 min/epoch). Status file:
   `/root/lp/cam4-2026-09-28/models/round3_semisup_kl_20260929/train_status.json` (`completed`/`total`).
2. **Progress when this was written:** val RMSE (LP's frame-level split, 40 frames, every 5 epochs):
   73.9, 72.3, 79.8, 70.1 (epochs 5–20, backbone frozen until 20) → **5.2 (25) → 3.9 (30)**. Unsupervised
   weight ramps +0.01/epoch, full at ~epoch 100. Ends at early stopping (patience 20 validations on
   `val_supervised_loss`) or the 300-epoch cap (~20:00 on 09-30). No measured end time; overnight is a guess.
   Read TensorBoard scalars on CPU without disturbing training: `scratchpad/lp_tb_scalars.py` (set its `run`
   path) under `CUDA_VISIBLE_DEVICES=''` in the WSL `lp` env.
3. **If it crashed / OOM:** traceback in the log. OOM → `training.train_batch_size: 4` and
   `dali.context.train.batch_size: 4` in BOTH `configs/lightning_pose_cam4.yaml` and the live copy
   (`scratchpad/lp_config_wsl.yaml` → `/root/lp/cam4-2026-09-28/config.yaml`; the live copy = reference with its
   three paths rewritten to `/root/lp/cam4-2026-09-28`). Relaunch anchored from Windows (Git Bash):
   `nohup wsl.exe -d Ubuntu-24.04 -u root -- bash -c "source /root/miniforge3/etc/profile.d/conda.sh && conda activate lp && cd /root/lp/cam4-2026-09-28 && litpose train config.yaml --output_dir <new dir>" > <log> 2>&1 &`
   To stop a run: `pkill -f 'bin/litpose'` inside WSL (a pattern matching the `bash -c` wrapper kills the
   wrapper first). Earlier failed runs kept for the record: `models/FAILED_mse_flat_round3_semisup_20260928/`
   (and `FAILED_collapsed_…` copy).
4. **When it finishes** (log shows `Best model path` / predictions; the model dir holds `tb_logs/…/checkpoints/*.ckpt`,
   `predictions.csv`, the `config.yaml` as used):
   a. **Copy the model dir to the share**: `lightning-pose/cam4-2026-09-28/models/round3_semisup_kl_20260929/`.
   b. **Predict on the three review clips** (already in WSL: `/root/lp/review_clips/*.mp4`):
      `litpose predict <model_dir> /root/lp/review_clips/*.mp4` (check `--help` for the output flag). Copy CSVs
      beside the DLC ones in `inference_check_20260928_round3/labeled_clips/` on the share; render labelled
      videos if LP offers it, else overlay with the same style as the DLC clips.
   c. **Score on OUR terms**: per clip × part, % frames with likelihood < 0.6; **jaw confidence and continuity
      through lick bouts** vs DLC round 3 (`*_snapshot_best-160_filtered.csv` in `labeled_clips/`) — the gain LP
      is for; spout NOT over-smoothed (where both confident, positions agree ~1 px; between-trial moves intact).
      Labelled-frame error: LP's `predictions.csv`, its 42 test frames only, stated as frame-level (most frames of
      our four held-out sessions are in LP's TRAIN set — not comparable to DLC's session hold-out). A clean
      session-level LP number needs a second run with a session-forced split.
   d. **Write up**: runbook "Lightning Pose, first result" + DECISIONS; a row per LP-analysed clip in §3 with
      the checkpoint path; commit + push (full suite runs on push; two tests need N:/M: mounted — after a reboot
      they need Priya's HMS sign-in).

### 0b. cam1 labelling (student, in progress from 2026-09-29)

* Worksheet: **`CAM1_GUIDE.html`** on the DLC share, generated by `python -m wfield_local.dlc_cam1_guide`
  (re-run to refresh what is still blank). 15 folders, balanced order (first 8 = every animal × every epoch);
  14 empty × 16 frames + the June folder `cam1_2026-06-06T12_25_18` (86 frames: jaw 71, tongue 38, **no nose,
  no spout on any frame**, 15 blank; its file still lists the retired 12-part set — napari shows 12 names there).
* Parts on cam1: **nose, jaw, tongue, spout**. Nose confirmed by Priya 2026-09-29 with the caveat that the cam1
  (ventral) and cam4 (frontal) nose tips need not be the same 3-D point. `LABELLING_GUIDE.html`'s stale cam1
  text was corrected (backups `LABELLING_GUIDE.bak-20260929-*.html`).
* **What the nose caveat means downstream** (answered 2026-09-29): single-view LP — no effect; LP
  `pca_multiview` — fine (learned from paired labels); LP 3-D reprojection losses — would fight the real offset,
  so exclude the nose from them or use `pca_multiview`; anipose — a fixed ~mm-scale offset (estimate, not
  measured) that cancels from displacements/speeds; check the reprojection-error threshold doesn't drop nose
  frames. **Measure it** once cam1 is labelled: triangulate the 241 same-instant cam1/cam4 frames and compare
  per-part reprojection error. Jaw/tongue (deforming) are the bigger multi-view concern, not the nose.

### 0c. Decisions made in this stretch that the next session should not re-open

* **Heatmap loss is `kl`** (MSE-on-softmax stalls at a flat map on 160×160 maps; measured). Resolution kept
  (downsample 2 = 4 px cells, finer than DLC's ~8 px) — changing the loss, not the resolution, was deliberate.
* **Spout stays OUT of `pca_singleview`** (columns 0,1,2 = nose, jaw, tongue). Measured 2026-09-29 on the labelled
  frames: with the spout in, it dominates the top components (SD ~62 px vs jaw 21) and the error the loss would
  penalise on correct poses rises 9.8 → 25.0 px; spout is uncorrelated with pose (r = 0.03 vs jaw x). The spout
  still gets the supervised and temporal losses. Caveat: the pose PCA is fitted on only ~110 frames (those with
  the tongue visible, all mid-lick) — weakest where the mouth is closed. A better spout constraint, later, is the
  DAQ's known spout position per trial.
* LP's `temporal` loss epsilon 10 px is above real spout motion per frame at 250 fps (p99 ~3 px measured on the
  check clips), so it should not flatten real spout moves — verify at 4c.

### 0d. Unrelated, pending from 2026-09-28

Once the analysis box's stage 2 for 0928 has landed: `python -m scripts.chronic_stability` (PS92_0922 back in the
cohort), then re-read the PS92 chronic cells in `docs/PRELIM_DATA_VLS_STROKE.md` and
`docs/status/STATUS_2026-09-28_CHRONIC_STABILITY.md`.

Everything below is background for these steps.

---

## 1d. UPDATE 2026-09-29 ~11:00 — the overnight run NEVER LEARNED; cause found and fixed (heatmap loss mse -> kl); relaunched

* **Symptom** (TensorBoard scalars of `models/FAILED_mse_flat_round3_semisup_20260928/`): `val_supervised_rmse`
  83.85 px at every one of 16 validations over 84 epochs / 10 h; `train_heatmap_mse_loss` 0.0509 constant =
  the energy of the target Gaussians, i.e. a FLAT predicted map. The epoch-4 checkpoint predicts uniform
  heatmaps (max 1.7e-4 vs uniform 3.9e-5) → soft-argmax at the image centre → ~80 px from every landmark.
  Killed at epoch 84.
* **Ruled out**: data (image tensors normalised correctly, labels land on target-heatmap peaks, visibility 2
  for present points / 0 for NaN tongue), precision (LP's Trainer is fp32), optimizer groups (head lr 1e-3).
* **Cause, measured**: `heatmap_loss_type: mse` is MSE on a SOFTMAX-normalised map; its gradient carries a
  factor of the predicted probability, which is 1/25,600 at init on our 160×160 maps (640 px at
  downsample 2 — LP's shipped configs run 64–96 px maps). Head gradient norm ~1e-7; Adam random-walks on
  augmented batches. Head-only, same init, real batches, 300 steps: **mse 153 → 149 px, confidence 0.00;
  kl 153 → 112 px, confidence 0.41**. fp16 autocast would kill mse entirely (an aside; trainer is fp32).
* **Fix**: `model.heatmap_loss_type: kl` in the reference and live configs (reason inline). Relaunched
  ~11:05 as `models/round3_semisup_kl_20260929/`, log `scratchpad/lp_train_round3_kl.log`. Same speed
  expected (~7 min/epoch). **Check the FIRST validations**: `val_supervised_rmse` must be well below 80 and
  falling by epoch 10; the flat run read 83.85 from the first one.
* Probe scripts (this session's scratchpad): `lp_tb_scalars.py` (read TB scalars), `lp_probe_vis.py`
  (dataset sample + checkpoint output), `lp_loss_compare.py` (head-only mse vs kl on real batches).
* `pkill -f 'litpose train'` inside a `wsl -- bash -c "… litpose train …"` kills its own shell first;
  use a pattern that does not match the wrapper (e.g. `pkill -f 'bin/litpose'`).

## 1. Where things stand

### DeepLabCut — round 3 is current and good to use

| | round 2 (iteration-1, best-060) | **round 3 (iteration-2, best-160)** |
|---|---|---|
| labelled frames / sessions | 360 / 15 | **407 / 15** (+47 between-trial spout frames) |
| train / test | 264 / 96 | 287 / 120 (same held-out sessions) |
| test error, DLC units (px): nose · jaw · tongue · spout | 3.21 · 3.77 · 7.48 · 1.85 | 3.33 · 4.12 · **6.44** · **1.74** |
| spout dropout (p < 0.6), unseen PS93_0908 clip | 9.8 % | **0.1 %** |
| spout dropout, held-out PS95_0907 clip | 2.5 % | 1.2 % |

Per-epoch test RMSE for the spout: acute 1.86, subacute 1.99, chronic 1.62, pre 2.92. Acute remains
the best or near-best epoch on every part (no false-deficit mode). Full tables: `runbooks/dlc_orofacial.md`
"Third result (2026-09-28)"; check results `inference_check_20260928_round3/README.md` on the share.

The training project is `training/widefield-Priya-2026-09-08-orofacial/` (`config.yaml` says
`iteration: 2`); anything that runs inference off that config — including the O2 path, which pins no
snapshot of its own — picks up `dlc-models-pytorch/iteration-2/widefield-trainset71shuffle1/train/snapshot-best-160.pt`.
Round 2's model is untouched under `iteration-1/`.

**Held-out sessions (seed 42, `dlc.train.training_fraction` 0.74):** `cam4_2026-08-20T16_31_02` (PS92,
acute), `cam4_2026-08-21T15_02_47` (PS95, subacute), `cam4_2026-09-07T17_08_48` (PS95, chronic; holds
24 of the new spout frames), `cam4_2026-06-06T18_02_39` (PS92, pre). Score every model on these.

**Spatial prior:** `dlc.prior.enabled: true`, `parts: [tongue, jaw]`; `dlc_prior.apply()` is the one
entry point (no-op when off). Measured inert on the review clips (0 of ~7,900 confident points outside
a box) — it is insurance against the rare confident-wrong peak. Nose and spout stay unmasked until a
FULL-SESSION run confirms their boxes never mask a real point (the spout's must hold the between-trial
transit, measured x 279–416 on two animals).

**Open on the DLC side:** the 31-point correction list (`CORRECTION_GUIDE.html`, regenerated from
round 3: 1 DELETE, 15 REPLACE, 7 DECIDE, 8 ADD); the tongue landmark convention (`DECIDE`); pre-stroke
jaw ~8.7 px in the 0606 held-out session (a labelling-convention problem, not the network). Each retrain
is ~45 min here: `conda activate dlc; python -m wfield_local.dlc_train --iteration <next>`.

### Lightning Pose — built, validated, blocked on a Linux-only dependency; WSL2 staged

Why LP at all (DECISIONS.md 2026-09-26): the jaw is lost at maximum mouth opening, human and network
agree it is occluded there (88 % of human-blank jaw frames are also network-unsure), so more cam4 labels
cannot fix it. LP's **temporal and pose-PCA losses act during training on unlabelled frames**, so the
network can learn to stay coherent through the occlusion instead of having it interpolated afterwards
(a post-hoc interpolation was built and rejected: it flattens jaw-opening amplitude, biased towards
"less movement" on post-stroke animals). Multi-view (cam1 sees the jaw when cam4 cannot) is the
follow-on, and the data already support it (§4).

What exists:

* **`lp` conda env** on this box — python 3.10, torch 2.11.0+cu128 (Blackwell `sm_120`; a cu124 build
  has no kernels), lightning-pose 2.4.2, CUDA visible. README "Per-machine environments" has the
  install order. Do not install LP into `dlc`.
* **Converted project** `lightning-pose/cam4-2026-09-26/` on the share: `CollectedData.csv` (**360
  rows — round 2's labels; STALE, see §2 step 3**), `config.yaml` (live copy), `videos/` (15
  unlabelled 30 s clips, §3), `smoke/` (an aborted smoke run).
* **Config** `configs/lightning_pose_cam4.yaml` — the version-controlled reference copy, with every
  validator trap found on 09-26 written into it (resize dims must be multiples of 128 → 640;
  `max_steps` must be ABSENT not null; `training.num_gpus` passes validation then raises; camera-
  dependent losses need `heatmap_multiview_transformer` + `imgaug: dlc` + `imgaug_3d: true` or are
  silently dropped). Key settings: `model_type: heatmap`, `backbone: resnet50`, `losses_to_use:
  [temporal, pca_singleview]`, PCA over nose/jaw/tongue only (the spout is apparatus and moves by
  design — indices 0,1,2), batch 8, 50–300 epochs, `train_prob 0.8 / val 0.1 / test 0.1`.

**The blocker:** `lightning_pose.utils.device.require_cuda_for_semi_supervised` refuses ANY unsupervised
loss unless `nvidia.dali` imports, and DALI has no Windows wheel (verified: the PyPI stubs fail with
"Didn't find wheel"). Supervised-only would run here and is pointless. The `pynvvc -> dali -> opencv`
fallback in `data/video/factory.py` is the PREDICTION reader and does not apply — do not re-derive
that mistake.

**WSL2 state (this box), superseded by §1b:** `wsl --install` was run elevated on 09-26; WSL 2.7.14 + kernel 6.18.33.2
present, Ubuntu-24.04 staged with `--no-launch`. `wsl --status` today: "Default Version: 2 … no
installed distributions" — i.e. exactly the staged-but-unlaunched state. **The Virtual Machine Platform
driver loads only at boot, so the next step is a reboot.** Firmware virtualization is fine despite
`Win32_Processor.VirtualizationFirmwareEnabled = False`: a hypervisor is already running (VBS status 2,
Credential Guard), and that flag is a reporting artefact from inside it. Do not send anyone into the BIOS.

---

## 1b. UPDATE 2026-09-28 20:10 — the box was rebooted; WSL2 works, INCLUDING DALI's GPU video decode

Done after the reboot, all from this session, all inside `Ubuntu-24.04` as root:

* `wsl --install -d Ubuntu-24.04 --no-launch` registered the distro (the 09-26 staging had not
  persisted); first launch with `wsl -d Ubuntu-24.04 -u root -- …` skips the user-creation prompt.
  Ubuntu 24.04.5, kernel 6.18.33.2-microsoft-standard-WSL2, **RTX 5060 visible via the Windows
  driver**, ext4 955 GB free, 24 CPUs, 31 GB RAM.
* **`lp` env inside WSL** at `/root/miniforge3/envs/lp` (script: this session's
  `scratchpad/setup_lp_wsl.sh`; 3 min end to end): python 3.10 → torch 2.11.0+cu128 (CUDA true) →
  lightning-pose 2.4.2 → **nvidia-dali-cuda120 2.3.0** (`--extra-index-url https://pypi.nvidia.com`).
  `require_cuda_for_semi_supervised` imports; `litpose` CLI on PATH.
* **The remaining unknown is settled: DALI's GPU video reader decodes under WSL2.**
  `fn.readers.video(device="gpu")` on a 680×680 review clip returned a (1, 8, 680, 680, 3) batch. That
  is the exact call LP's unlabelled-frame pipeline makes (`lightning_pose/data/video/dali.py:135`).
  The `device="cpu"` variant fails — the legacy reader is GPU-only — which is irrelevant to LP.
* Staged in ext4: `/root/lp/cam4-2026-09-28/config.yaml` (the reference config with its three paths
  rewritten to `/root/lp/cam4-2026-09-28`) and the three review clips in `/root/lp/review_clips/`.

**Still needed, and why it stopped here:** N: and M: were "Unavailable" after the reboot and prompt
for HMS credentials, which this session cannot supply. Until Priya signs in once, the round-3 labels
cannot be re-converted, the 15 unlabelled clips cannot be copied into WSL, and the two share-dependent
tests in the pre-push hook fail (the PathResolver falls to the wrong profile without the mounts), so
commits after c86448c sit local. Resume at §2 step 3.

Run things in WSL from Windows with a SCRIPT FILE, not an inline `bash -lc "…"`: `$(…)` and quotes are
mangled between Git Bash and WSL (measured: `python: command not found` from a command that worked
verbatim inside WSL). Pattern that works:
`wsl.exe -d Ubuntu-24.04 -u root -- bash -c "tr -d '
' < /mnt/c/<path>/x.sh > /root/x.sh && bash /root/x.sh"`.

## 1c. UPDATE 2026-09-28 22:30 — round-3 labels converted, project on the share, training RUNNING in WSL

* `litpose convert` on the round-3 training copy (run from the Windows `lp` env; the converter is pure
  pandas): **407 rows**, nose 407 / jaw 340 / tongue 164 / spout 396 — identical to `dlc_train`. Published
  as `lightning-pose/cam4-2026-09-28/` on the share (labels + `config.reference.yaml` + README; the 15
  clips stay in `cam4-2026-09-26/videos/`). ext4 copy: `/root/lp/cam4-2026-09-28/` (761 MB with clips).
* **Two more config traps, both fixed in `configs/lightning_pose_cam4.yaml` with the reason inline:**
  `training.patch_mask: null` → must be ABSENT (`"init_epoch" in cfg.training.patch_mask` on a null
  raises TypeError); and a top-level **`callbacks.anneal_weight` section is REQUIRED** for any
  semi-supervised run (`AnnealWeight(**cfg.callbacks.anneal_weight)`; omegaconf: "Key 'callbacks' is
  not in struct"). LP's default config is not in the wheel, which is why 09-26's validation missed it.
* **WSL detaches nothing you start inside it.** `nohup … &` inside a `wsl.exe -- bash -c` session dies
  when that session ends. Working pattern: run the *Windows-side* `wsl.exe` in the background as the
  anchor: `nohup wsl.exe -d Ubuntu-24.04 -u root -- bash -c "… litpose train …" > log 2>&1 &`.
* **Launched 22:27:** `litpose train config.yaml --output_dir /root/lp/cam4-2026-09-28/models/round3_semisup_20260928`
  — data module 407 images, LP split train 325 / val 40 / test 42 (LP's own frame-level split; NOT our
  metric, see §2 step 7), PCA kept 3/6 components (95.4 %), `SemiSupervisedHeatmapTracker` resnet50,
  losses temporal + pca_singleview, batch 8 + DALI context batch 8. GPU at 7.9 / 8.1 GB at start — if it
  OOMs, drop `training.train_batch_size` and `dali.context.train.batch_size` to 4 and note it. Log:
  this session's `scratchpad/lp_train_round3.log`; outputs under the model dir (hydra).
* Not yet done: evaluation on OUR held-out sessions and the three review clips (§2 step 7), copying the
  model back to the share, the runbook "Lightning Pose, first result" section.

## 2. The procedure to proceed (WSL2 route)

0. ~~Reboot the box~~ DONE 2026-09-28 (§1b). Steps 1–2 DONE too; resume at step 3. Originally: reboot when nothing important is running (check `Get-CimInstance Win32_Process` for
   other users' `cellpose_gpu` / 2pRAM jobs — this machine is shared; never kill python by name).
1. **Launch Ubuntu-24.04** (`wsl -d Ubuntu-24.04`, first launch creates the user). Confirm the GPU is
   passed through: `nvidia-smi` inside WSL must show the RTX 5060 using the WINDOWS driver (610.88).
   **Never install an NVIDIA driver inside WSL** — it breaks the passthrough. Only the CUDA toolkit
   goes inside, and DALI's wheel bundles what it needs.
2. **Env inside WSL** (mirror the Windows `lp` recipe): miniforge → `conda create -n lp python=3.10`
   → `pip install torch --index-url https://download.pytorch.org/whl/cu128` FIRST → `pip install
   lightning-pose` → `pip install nvidia-dali-cuda120` (the piece that cannot install on Windows) →
   `python -c "import nvidia.dali, lightning_pose"`.
3. **Re-convert the project from the ROUND-3 training copy** so the 47 spout frames are in:
   `litpose convert /mnt/n/.../training/widefield-Priya-2026-09-08-orofacial --lp_dir <new lp_dir>`
   (any env with pandas). Expect **407 rows**, 4 bodyparts, fills nose 407 / jaw 340 / tongue 164 /
   spout 396 — the same counts `dlc_train --dry-run` prints. Name the new dir by date
   (`lightning-pose/cam4-2026-09-<dd>/`), keep `cam4-2026-09-26/` as the round-2 record.
4. **Copy the project into WSL's ext4** (`~/lp/cam4-…`, ~0.75 GB incl. clips) rather than training over
   `drvfs`/SMB — faster and no share credentials. Reuse the 15 clips from `cam4-2026-09-26/videos/`
   (they are unlabelled video; the label set does not change them).
5. **Write the live config** from `configs/lightning_pose_cam4.yaml` with `data.data_dir`,
   `data.video_dir` and `eval.test_videos_directory` pointed at the ext4 copy. Keep everything else in
   step with the reference copy; if you change a setting, change the reference copy in git too.
6. **Train**: `litpose train <config.yaml>` (semi-supervised). Watch the first epoch's memory on the
   8 GB card; `train_batch_size: 8` matched DLC, but LP's unlabelled batches add to it — drop to 4 if
   it OOMs and note it in the config.
7. **Evaluate on OUR terms, not LP's.** LP's split is uniform over frames, so its own test number is
   optimistic (near-duplicate lick frames on both sides — the reason `dlc_train.split` holds out whole
   sessions). Score the LP model on the four held-out sessions above and on the three review clips
   (§3), with the same metrics: per-part test error, per-epoch RMSE, spout dropout, and — the point of
   LP — jaw confidence across licks (round 2 measured 0.90 → 0.28 across a lick; round 3 drops the jaw
   on 0.7 / 19.6 / 9.5 % of frames on the three clips). `litpose predict` on the clips, then compare
   against `labeled_clips/*_filtered.csv` frame by frame.
8. **Record** the result in `runbooks/dlc_orofacial.md` (a "Lightning Pose, first result" section) and
   DECISIONS.md, and decide with Priya whether LP replaces DLC for cam4 inference or stays a jaw filler.

**Fallback if IT blocks WSL:** O2. Linux GPUs, DALI installs normally, and `dlc.o2.*` + the sbatch/rsync
scaffolding from the DLC inference port already exist. Project, clips and config transfer unchanged.

---

## 3. Clips — where they are and which network they came from

All on the DLC share `N:\MICROSCOPE\Priya\DeepLabCut\Widefield\` (Mac: `/Volumes/Neurobio/MICROSCOPE/Priya/DeepLabCut/Widefield/`). All cam4, 680×680, 250 fps.

| clip(s) | location | frames / length | purpose | network they were analysed with |
|---|---|---|---|---|
| `PS92_0820_acute_day1_HELDOUT_round3_labeled.mp4` | `inference_check_20260928_round3/labeled_clips/` | 5,000 / 20 s, densest-licking window (t = 1439 s, frame 360175 of `cam4_2026-08-20T16_31_02.avi`) | **review clip** — held-out acute session (day 1) | **round 3**, iteration-2 / best-160, prior ON tongue+jaw, median-filtered, p ≥ 0.6, 5-frame trails |
| `PS93_0908_chronic_UNSEEN_round3_labeled.mp4` | same | 5,000 / 20 s, from frame 740321 of `cam4_2026-09-08T11_10_48.avi` | **review clip** — session never labelled; where round 2 dropped the spout on 9.8 % of frames | round 3, as above |
| `PS95_0907_chronic_HELDOUT_round3_labeled.mp4` | same | 5,000 / 20 s, from frame 346831 of `cam4_2026-09-07T17_08_48.avi` | **review clip** — held-out chronic session carrying 24 of the new spout frames | round 3, as above |
| `*_snapshot_best-160_filtered.csv` (3) | same | — | the predictions drawn on the clips above | round 3 |
| `PS93_20260908_UNSEEN_f740321DLC_…best-160.csv`, `PS95_…best-160.csv` | `inference_check_20260928_round3/` | — | check-clip predictions, **full frame, NO prior** (the like-for-like comparison with round 2) | round 3 |
| `round3_vs_round2_spout_PS93.png` | same | — | montage: cyan = round 2, magenta = round 3 on the frames round 2 dropped | rounds 2 and 3 |
| `PS93_…_snapshot_best-60_filtered.csv`, `PS95_…best-60_filtered.csv` (+ `crop384x512_prior/`, `fullframe_prior/`) | `inference_check_20260926/` | same two 20 s clips | round-2 check: full frame vs crop, with/without prior | **round 2**, iteration-1 / best-060 |
| 15 × `cam4_<stem>_t<sec>.mp4` | `lightning-pose/cam4-2026-09-26/videos/` | 7,500 / 30 s each, cut 3 s before a mid-session cue (ENL → cue → response → ITI spout move), one per labelled session | **unlabelled clips for LP's semi-supervised losses**; also `eval.test_videos_directory` | none — unlabelled by design (chosen with the trial tables, not a network) |
| source `.mp4` clips (3) + round-2 analysis of two of them | this session's scratchpad `…/scratchpad/clips/` (temporary, not on the share) | — | inputs to the above | round 2 files: best-060 |

The review clips were cut by `pick_clips.py`-style logic (densest 20 s of lick onsets, `dlc_frames.lick_onsets`
+ `frame_of`) and rendered by `review_clips.py` (both were scratch; `review_clips.py`'s logic is: `analyze_videos`
under `dlc_prior.apply()`, `filterpredictions`, `create_labeled_video(filtered=True, trailpoints=5)`).
The LP unlabelled clips were cut by `scripts/lp_unlabelled_clips.py` (committed 2026-09-28 from the 09-26
scratch script; seed 92, one clip per animal × epoch).

**When LP has a model, add a row here for each clip you analyse with it, with its checkpoint path.**

---

## 4. The multi-view extension, when you get there

* `dlc.frames.anchor_cam` is `cam4`, so every camera's frames were sampled at the SAME DAQ instant via
  its own alignment template: **241 frames exist in both cam1 and cam4 at the same (trial, phase)**.
* cam1 (looks up at the snout underside) has one partly-labelled session with jaw on 71/72 rows — it
  sees the jaw exactly when cam4 cannot. Labelling more cam1 is the prerequisite, not a blocker.
* LP's `pca_multiview` needs no calibration; `heatmap_multiview_transformer` can take the intrinsics /
  extrinsics / distortions from the 2026-09-11 four-camera solve (`dlc_calibration`, `dlc_anipose`).
* Carry the repo's caveat: no landmark is the same physical point from two views, and for deforming
  parts the offset changes with posture. Using cam1 to FILL cam4's occluded jaw frames is sound;
  calling the result a 3D jaw is a separate claim needing its own check.

---

## 5. Things that will bite (all measured, all in git — read before "improving" them)

* **Labels live in the LABELLING project only** (`widefield-Priya-2026-09-08/labeled-data/<stem>/CollectedData_Priya.*`).
  `training/…-orofacial/` is overwritten from it on every `dlc_train` run; LP's `CollectedData.csv` is a
  converted copy. Never correct a label anywhere but the labelling project.
* **Split by session, seed 42.** A frame-level split reports a test error partly measured on the training
  set (four frames of one 80 ms lick are near-duplicates). If a held-out frame gets *corrected toward a
  prediction*, either move it into train (seed change) or say so when quoting the number.
* **`dlc_train` prints two tables in different units**: DLC's per-keypoint block is MEAN euclidean error
  despite saying rmse; `per_epoch_error` is true RMSE. Compare like with like.
* **Between-trial spout frames are picked by the network, not a time offset** (`dlc_iti_frames`): scan
  each position-change gap at ~21 Hz, keep the largest spout-x jump and the lowest likelihood, two gaps
  per position, seed 92. The 09-26 picks are `docs/dlc_iti_rows_20260926.csv`. In cam4 the entire six-
  position excursion is ~120 px around frame centre — "the spout is always near the middle" is geometry.
* **This machine is shared.** Other sessions' `cellpose_gpu` / 2pRAM jobs run here for days. Identify a
  process by its command line before touching it (memory note `analysis-desktop-shared-box`).
* **Round-2's "retracted spout found in the right place, just under-confident" was half wrong**: deep in
  the retraction it sat on fur 40–80 px above the tip (montage above). Low confidence WAS the correct
  signal; the cutoff was doing its job. Treat "position right, confidence wrong" claims with suspicion.

## 6. Commands, in one place

```powershell
conda activate dlc
python -m wfield_local.dlc_train --dry-run                      # stage labels, audit, print the split
python -m wfield_local.dlc_train --iteration 3                  # next DLC round (~45 min)
python -m wfield_local.dlc_train --evaluate --guide --iteration 3   # re-evaluate + CORRECTION_GUIDE.html
python -m wfield_local.dlc_iti_frames --sessions PS9x:YYYYMMDD --extract   # more between-trial frames
python -m wfield_local.dlc_spout_guide                          # the page for the labeller
python -m scripts.lp_unlabelled_clips                           # LP unlabelled clips (idempotent)
# Lightning Pose (inside WSL, env lp):  litpose convert … ;  litpose train <config> ;  litpose predict …
```

Repo state at handoff: `main` at c86448c (all of 2026-09-28 pushed). Related records: `runbooks/dlc_orofacial.md`
(the full DLC runbook, results rounds 1–3), DECISIONS.md entries 2026-09-26 (why LP; WSL2 attempt),
`configs/lightning_pose_cam4.yaml`, README "Per-machine environments".
