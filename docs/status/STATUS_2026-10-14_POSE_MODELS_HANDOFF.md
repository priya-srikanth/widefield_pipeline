# Handoff — 2026-10-14 (late morning): cam4 pose-model selection (DLC / LP x round 3 / 4, temporal-context LP, EKS, human lick review), side-camera labelling plan, WSL GPU safety

**START HERE.** Supersedes `STATUS_2026-10-13_ENL_TRAJECTORY_TRACKING_R4_HANDOFF.md` (still the reference for the
movement-removed ENL trajectory result and the null/potent closure). **The model register is `docs/POSE_MODELS.md`**
(every cam4 model, settings + rationale, results, the selection rule) — read it before touching tracking. Reasoning:
`DECISIONS.md` addenda 2026-10-13 (x4), 2026-10-14 (x5, incl. a CORRECTION). Analysis desktop; M: = MICROSCOPE.
Anything meant for Priya goes on the server, not `C:\Users\SabatiniLab\lp_stage` (local; she cannot see it).

## 0. RUNNING (2026-10-14 10:55)
1. **LP r4 ctx-semi** — round-4 SEMI-SUPERVISED temporal-context LP (`heatmap_mhcrnn` + unlabelled-video temporal loss),
   WSL, **512 px**, batch 1 (+5-frame video batch), temporal-loss epsilon **[nose 10, jaw 10, tongue 35, spout 10]**,
   through `/root/lp/lp_capped.py` (PyTorch memory cap 0.75) + a 1-s guard at 6.8 GB. Epoch 116 at 10:54, ~1.9 min /
   epoch, 5.5 GB steady, val supervised loss 0.160; early stopping likely ends it this afternoon; then it predicts the two
   session clips + six review clips. Script `C:\Users\SabatiniLab\lp_train_semi_ctx_wsl.sh`; log
   `lp_stage\lp_train_semi_ctx_wsl.log`; model `/root/lp/cam4-2026-10-13/models/r4_mhcrnn_semi_512_20261014`;
   predictions -> `lp_stage\r3_vs_r4_lp\r4mhcrnn_semi_512\`.
2. **QUEUED: LP r3 ctx-semi** (same recipe, round-3 labels) waits for (1)'s `done`: `lp_train_semi_ctx_r3_wsl.sh`, log
   `lp_stage\lp_train_semi_ctx_r3_wsl.log`, model `/root/lp/cam4-2026-09-28/models/r3_mhcrnn_semi_512_20261014`.
3. NEXT when (1) finishes: score it exactly like the others — `scripts/model_lick_compare.py` (contact recall, licks,
   angles), jump counts, `pose_cue_traces compare`, and against Priya's lick review (`lp_stage/lick_review/
   review_scored.csv`; detection = a kept lick within 40 ms of a reviewed candidate). Questions: is it confident on clear
   tongues (the supervised one was not)? does it keep the jaw fix? Then EKS (fixed) of LP r4 + ctx-semi, and the
   ">= 2 models" agreement count. Add to `docs/POSE_MODELS.md`.

## 1. Findings since the 10-13 handoff
* **ENL trajectory** (10-12 / 10-13, unchanged): residual tracks raw across recovery, 118 curated sessions, r 0.77-0.92.
* **DLC round 4 (540 labels) is not better than round 3** on frames held out by both; more conservative on the tongue
  (recall 97.1 % vs 99.3 %). More single-frame labels of the same kind no longer pay off.
* **Five-model comparison** (PS93 0814 pre, 0821 acute; same mouth, stops, DAQ contacts; `model_lick_compare.py`):
  contact recall DLC r3 99.3/99.6, DLC r4 97.1/97.1, LP r3 99.0/99.6, LP r4 98.5/99.6, LP r4 ctx-sup 96.6/96.3 %.
  Angle at peak protrusion on contact licks agrees within ~1-2 deg across all models. Jaw x side-flips (> 30 px, pre):
  DLC r3 40, DLC r4 40, LP r3 17, LP r4 29, ctx-sup 3. LP r3 has many tongue outliers (133 jumps).
* **Jaw side-flips at the close positions** are a genuine ambiguity: the rod covers the chin and either visible side is
  a correct label (Priya reviewed 59 round-4 jaw labels: all correct). The temporal-context model suppresses the flips.
* **Supervised temporal-context model** (Windows-native, 640 px, no unlabelled-video loss): removes jaw flips,
  tightest angles, but UNDER-CONFIDENT on clearly visible tongues (p 0 on several; confident on 9 / 16 of Priya's
  dropped-tongue frames vs LP r3 15, LP r4 12).
* **Human lick review** (Priya, 88 blind candidates + 8 controls, all controls judged real): of no-contact candidates
  58 real, 19 not, 3 unsure; single-model detections mostly false (15 / 20). Precision / share of real found: DLC r3
  91/67, DLC r4 85/88, LP r3 100/59, **LP r4 85/98**, ctx-sup 100/69 %. Priya: many real ones are barely visible and
  missing some is acceptable; and **several LP r4 "false positives" are defensible as teeny licks** (so LP r4's true
  precision is likely ~90 %). Review page: claude.ai artifact `UURfzx99WnTxUEAbegZhsy` (answers in its `lick_review`
  collection); sheets + scores on the server: `inference_check_20261012_round4/lick_review/`.
* **CORRECTION — the first EKS ensemble's advantage was an artefact** (see §2). Fixed EKS (LP r4 + ctx-sup) ~ LP r4:
  recall 98.8 / 99.2 %, review precision 88 % / real found 98 %, kinematics intact at smooth_param 100 (peak vy 5988 vs
  6011 px/s), jaw flips 17 vs 30 (partial: when both models are confident on opposite sides of the rod EKS splits it).
* **Agreement rules for no-contact licks** (on the review): LP r4 alone 85 / 98; ">= 2 models" 93 / 91; LP r4 AND ctx
  100 / 67 %.

## 2. Decisions (10-13 .. 10-14)
* **Judge models downstream, not by pixel error** — selection rule in `docs/POSE_MODELS.md` §4: epoch-bias gate;
  contact recall >= 99 % over FINDABLE contacts (found by >= 1 model; a contact no cam4 model finds is unobservable)
  with no spout position < ~97 %; then human-reviewed incomplete-lick precision / coverage, angle spread + unbiased
  kinematics, stability, held-out pixel error (sanity), practicality.
* **Current leaning (Priya 10-14):** borderline teeny licks are ambiguous, not errors -> use **fixed EKS (or LP r4)**
  for detection and positions; the **">= 2 models agree" count is a SENSITIVITY analysis**, not a filter: any result on
  lick counts / incomplete licks is reported both ways. Final choice waits for the ctx-semi models and >= 4 more
  sessions (chronic + other animals: PS93 0908, PS94 0818, PS95 0821, PS94 0921).
* **The EKS fix** (`scripts/eks_two_model.py`): LP reports "not visible" as a low-likelihood point PARKED AT THE IMAGE
  CENTRE (~340, 338). The first ensemble passed those points to EKS relying on the occlusion variance floor; they still
  pulled the smoothed track toward the centre (EKS tongue up-right of LP r4's, likelihood ~0.5 = mean of 1.0 and 0.0),
  losing faint licks and suppressing others. Now, per keypoint and frame, a member below the cut (0.4) takes the MEAN
  POSITION OF THE CONFIDENT MEMBERS (so it neither pulls nor vetoes); its likelihood stays low; frames with no confident
  member are untouched. EKS's output "likelihood" (~0.5 when one member says hidden) is its averaged score, above the 0.4
  cleaning cut.
* **Temporal-loss epsilon per keypoint** [10, 10, 35, 10] (Priya asked to raise the penalty threshold): genuine tongue
  motion p99 23-34 px/frame, jaw 4-6, nose / spout 1-3; 10 px penalised ~the fastest quarter of real licks.
* **Side cameras (cam2 + cam3):** one SHARED model, cam3 flipped left-right at training / inference (labelling on raw
  images; the four parts are midline so no L/R swap); each moment labelled in ONE side camera, every 5th in both
  (332 frames vs 602 doubled); same moments as cam4 (anchor) incl. the 61 cam4 round-4 moments; context frames stay
  blank. Worksheet `DeepLabCut/Widefield/SIDECAM_GUIDE.html` (Part A pilot 8 sessions / 144 frames, B round-4 62,
  C rest 126), separate from the student's CAM1_GUIDE. Crop side views to the face at training (~2-3x faster).
* **Camera templates** exist for every analysed session with video (built 14; PS92 0605 cam4 fails quality; PS93 0813
  has no video — EXPERIMENT_ERRORS.md).
* **ENL summaries:** curated set, >= 60 trials, never a session with an incomplete movement model.

## 3. Code added / changed (all pushed; last `ef79a3f`, plus the commit carrying this handoff)
| file | what |
|---|---|
| `scripts/model_lick_compare.py` | per model: contact recall (unique contacts in a lick window), kept / no-contact licks, angle at peak protrusion by position (contact licks + all) |
| `scripts/pose_cue_traces.py compare [--cleaned]` | cue-aligned tongue / jaw traces, any number of models; cleaned tongue zero = ONE mouth for all models |
| `scripts/eks_two_model.py` | single-camera EKS over two (or more) LP / DLC csvs, fixed sub-cut handling, DLC-format output |
| `scripts/lick_review_candidates.py` | blind candidate licks (union over models, balanced) + controls -> candidates.json / key.csv |
| `scripts/lick_review_sheets.py` | LP r4 false-positive / EKS false-negative sheets with legend (LP r4 orange circle, old EKS green diamond, fixed EKS magenta cross) |
| `wfield_local/dlc_sidecam.py`, `dlc_sidecam_guide.py` | side-camera pairs + assignment + worksheet; `dlc_context_frames` takes cam2 / cam3; tests `tests/test_dlc_sidecam.py` |
| `wfield_local/trial_windows.py` | DAQ-cue fallback (`_daq_cue_trials`), `spout_motion_spans`, `docked_spans` |
| `scripts/enl_movement_multisession.py` | refuses sessions missing a camera; drops trials outside imaging coverage |
| docs | `docs/POSE_MODELS.md` (register), `docs/FACERHYTHM_PROPOSAL.md`, EXPERIMENT_ERRORS (PS93 0813 no video) |
Not in the repo (review / one-off): `inference_check_20261012_round4/{check_round4.py, heldout_both.py,
tongue_frames_all_models.py, render_models_grid.py}` on the DLC share; WSL / Windows launchers in `C:\Users\SabatiniLab\`
(`lp_train_semi_ctx_*_wsl.sh`, `lp_memtest_wsl.sh`) and `lp_stage\*.ps1`; `/root/lp/lp_capped.py` in WSL.

## 4. Pending / to-do
1. Score LP r4 ctx-semi (and r3 ctx-semi) as in §0.3; update `docs/POSE_MODELS.md`; decide the cam4 production model
   with Priya (current leaning: fixed EKS LP r4 + best context model, agreement count as sensitivity).
2. Run the chosen model(s) on >= 4 more sessions (chronic + other animals) and re-check the selection rule per epoch.
3. Optionally rescore precision with Priya's teeny-lick calls (she can name trials, or add a "tiny lick" option to the
   review page).
4. Production run (O2 or desktop) on the chosen model for all sessions -> executed-angle vs target readout (MUST REVISIT).
5. Side cameras: student labels SIDECAM_GUIDE Part A; then a shared flipped model (crop to the face), then Parts B / C.
6. Jaw x side-flip rule in cleaning (proposed; Priya wants visual QC before adopting) — may be unnecessary if the
   context model / EKS handle the jaw.
7. Pixel-vs-LocaNMF decoding gap (non-motor control area); FaceRhythm proposal.
8. cam1 labelling Parts 2 / 3 only if a model comparison says labels, not architecture, are the limit.

## 5. Pitfalls (new — earlier lists still apply)
1. **WSL GPU = hypervisor crash on overshoot** (bugcheck 0x20001; 09-30 x2, 10-13). Measure peak memory natively first
   (OOM is clean there); run WSL GPU jobs via `lp_capped.py` + guard; temporal-context models need batch 1 (and 512 px
   semi-supervised); prediction in 16-frame chunks; never start an untested WSL config unattended.
2. **LP's "not visible" = a point parked at the image centre** at low likelihood. Never use LP positions without a
   likelihood filter, and never hand them unmasked to a smoother (the EKS artefact).
3. **Windows-native LP**: inference and supervised training only (no DALI); inference copies of WSL models need
   `losses_to_use: []` and `data_dir` repointed (`lp_stage/lp_models_win/`).
4. **DLC 3 takes the project root from the config's folder**; `analyze_images` writes `image_predictions_*.h5` into
   the training project's labeled-data (delete after, with Priya's OK).
5. **Test errors from different held-out splits are not comparable** across DLC rounds; use frames held out by all.
6. **`pkill -f <pattern>` inside `wsl -e bash -c "..."` kills its own shell** when the pattern is in its command line —
   use a bracket pattern (`"[l]p_capped.py"`). Editing a RUNNING bash script is unsafe — stop and restart it.
7. **Background waiters that grep logs need strict patterns** (`^== .* done$`), or a stray word starts GPU work early.
8. **GNU sed `\U` / `\L` and Python `\n` in edit scripts mangle Windows paths / strings** — use the Edit tool or
   `chr(92)`.
9. **Video-file timestamps can differ by a second between cameras of one session** (PS94 0820 cam2 / cam3): key sessions
   by animal + date.
10. Review material for Priya goes on the server (`M:\...\inference_check_20261012_round4\`), with a legend on every
    sheet.
