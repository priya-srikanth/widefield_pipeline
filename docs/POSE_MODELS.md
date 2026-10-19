# Orofacial pose models (cam4 front view, cam1 from below) — register, settings, results, and how we choose

Living record of every pose model trained for the widefield cohort -- cam4 in sections 1-4, cam1 in Part B: what it is,
the settings that matter and why, where it lives, how it scored, and the rule for picking the production model.
**Part C = the evaluation plan and status for both cameras (models in training, what is scored, what is left).**
Reasoning in `DECISIONS.md` (2026-09-28, 09-30, 10-05, 10-13 x3, 10-14); evaluation scripts named below. Last updated 2026-10-18.

Keypoints everywhere: **nose, jaw, tongue, spout** (`dlc.cameras.cam4.bodyparts`). Labels follow the
TARGET vs CONTEXT rule (DECISIONS 2026-09-30): a blank on a labelled frame means *hidden*; context frames stay blank.

## 1. The models

| id | framework / type | labels | resolution | batch | unlabelled-video loss | trained on | where |
|---|---|---|---|---|---|---|---|
| **DLC r3** | DLC 3.0.1, ResNet-50, single frame | round 3: 407 frames / 15 sessions | 680 (native) | DLC default | — | Windows GPU | `training/widefield-Priya-2026-09-08-orofacial`, iteration-2 `trainset71`, `snapshot-best-160` |
| **DLC r4** | same | round 4: 540 frames / 21 sessions | 680 | DLC default | — | Windows GPU | same project, iteration-2 `trainset74`, `snapshot-best-170` |
| **LP r3** | Lightning Pose 2.4.2 `heatmap`, ResNet-50, single frame | round 3 | 640 | 4 | temporal, eps 10 px | WSL | `/root/lp/cam4-2026-09-28/models/round3_occl_uniform_20260930`; Windows inference copy `lp_stage/lp_models_win/` |
| **LP r4** | LP `heatmap` | round 4 | 640 | 4 | temporal, eps 10 px | WSL | `/root/lp/cam4-2026-10-13/models/r4_heatmap_20261013` (best epoch 159); copy in `lp_stage/lp_models_win/` |
| **LP r4 ctx-sup** | LP `heatmap_mhcrnn` (frames t-2..t+2 at inference) | round 4 | 640 | 1 | **none** (supervised only) | **Windows native** | `lp_stage/lp_cam4_r4_win/models/r4_mhcrnn_sup_win_20261013` (300 epochs) |
| **EKS r4+ctx** | single-camera Ensemble Kalman Smoother over LP r4 + LP r4 ctx-sup | — | — | — | — | CPU (env `eks`) | `scripts/eks_two_model.py`; outputs `lp_stage/r3_vs_r4_lp/eks_r4_heat_ctx*/` |
| **LP r4 ctx-semi** *(training)* | LP `heatmap_mhcrnn` | round 4 | **512** | 1 (+5-frame video batch) | temporal, eps **[10, 10, 35, 10]** | WSL, capped | `/root/lp/cam4-2026-10-13/models/r4_mhcrnn_semi_512_20261014` |
| **LP r3 ctx-semi** *(queued)* | same recipe | round 3 | 512 | 1 (+5) | eps [10, 10, 35, 10] | WSL, capped | `/root/lp/cam4-2026-09-28/models/r3_mhcrnn_semi_512_20261014` |

Shared LP settings (all LP models): `uniform_heatmaps_for_nan_keypoints: true` (a blank = trained as hidden),
KL heatmap loss, `pca_singleview` OFF, `downsample_factor 2`, early-stop patience 20, `max_epochs 300`.

## 2. Settings that matter, and why

* **Occlusion = flat heatmap** (`uniform_heatmaps_for_nan_keypoints: true`, LP; DLC only masks blanks). LP then
  reports a hidden part at low likelihood and parks it at the **image centre (~340, 338 px)** — a "not visible"
  answer, NOT a position. Every consumer must filter by likelihood; the cleaning step does (below).
* **Likelihood cut 0.4 and gap interpolation** (`orofacial_clean`): below 0.4 a point is missing; tongue gaps up to
  **56 ms** and jaw gaps up to 120 ms are PCHIP-interpolated from the neighbouring good frames. A brief
  low-confidence dip mid-lick is filled; a model that is under-confident through most of a lick cannot be rescued.
* **Tongue zero = the mouth**: the least-squares intersection of the three close->far spout lines (`spout_frame`),
  one per session; jaw keeps its own resting baseline.
* **Temporal-loss epsilon per keypoint** (from 2026-10-14): the loss penalises frame-to-frame jumps larger than
  epsilon on unlabelled video. Measured genuine motion (confident frames, PS93 0814 / 0821): tongue median 7-10,
  p95 17-25, **p99 23-34**, p99.9 29-46 px/frame; jaw p99 4-6; nose / spout p99 1-3. The old single 10 px
  penalised roughly the fastest quarter of real tongue movement, so tongue = 35 px (~p99), nose / jaw / spout stay
  10 px (side-flips of 30-60 px and teleports are still penalised).
* **Temporal-context models need memory.** 5 frames per sample: > 7.8 GB at batch 2 / 640 px once the backbone
  unfreezes; batch 1 / 640 px = 4.6 GB (supervised); semi-supervised adds a 5-frame video batch: > 6 GB at 640 px,
  **5.5 GB at 512 px**. Hence 512 px for the semi-supervised context models (a confound vs the 640 px models).
* **GPU safety (WSL).** An 8 GB overshoot under WSL spills and crashes the hypervisor (bugcheck 0x20001: 2026-09-30
  x2 from 96-frame prediction chunks, 2026-10-13 from context-model training). Rules: measure peak memory natively on
  Windows first (OOM there is a clean kill); in WSL run through `/root/lp/lp_capped.py`
  (`torch.cuda.set_per_process_memory_fraction`, default 0.75) plus a 1-s guard killing at 6.8 GB; prediction
  always in 16-frame chunks; never start an untested WSL GPU config unattended.
* **Windows-native LP** works for supervised training and for inference (OpenCV video reader), not for the
  unlabelled-video loss (needs DALI, Linux only). Inference copies of WSL models need `losses_to_use: []` and
  `data_dir` repointed to a Windows copy of the labels (LP's post-prediction metrics read them).
* **EKS** (`smooth_param`): the auto fit over-smooths the tongue; 10 keeps lick extent but lowers peak tongue speed
  ~11 % (pre) / 19 % (acute); **100 keeps speed within ~5 %** and still removes the jaw side-flips. A member below
  the 0.4 cut gets the occlusion variance floor (no pull on the track).

## 3. Results to date (PS93 0814 pre-stroke, PS93 0821 acute; 60 trials each, 10 per position)

Scripts: `scripts/model_lick_compare.py` (contact recall, licks, angles; same mouth / trial stops / DAQ contacts for
every model), `scripts/pose_cue_traces.py compare` (cue-aligned traces), jump counts and the frames sheets in
`DeepLabCut/Widefield/inference_check_20261012_round4/` and `session_poses/<session>/r4_eval/`.

| | DLC r3 | DLC r4 | LP r3 | LP r4 | LP r4 ctx-sup | EKS r4+ctx (s 10) |
|---|---|---|---|---|---|---|
| contact recall pre / acute (%) | 99.3 / 99.6 | 97.1 / 97.1 | 99.0 / 99.6 | 98.5 / 99.6 | 96.6 / 96.3 | 99.0 / 99.2 |
| licks without contact, pre | 119 | 130 | 79 | 149 | 59 | 78 (95 at s 100) |
| jaw x jumps > 30 px, pre | 40 | 40 | 17 | 29 | 3 | 0 |
| tongue x jumps > 30 px, pre | 33 | 16 | 133 | 29 | 24 | 1 |
| Priya's 16 dropped-tongue frames, confident | 16* | 0* | 15 | 12 | 9 | — |
| peak tongue vy, contact licks, pre (px/s) | 6053 | — | — | 6011 | 6095 | 5354 (5723 at s 100) |

\* by selection. Angle at peak protrusion on contact licks: all models within ~1-2 deg per position; ctx-sup and EKS
have the tightest spread. Held-out pixel error (70 frames held out by both DLC rounds): DLC r3 ~ DLC r4.

**Findings.** (1) More single-frame labels (round 4) did not improve DLC; DLC r4 is more conservative on the tongue.
(2) LP r4 is the best single-frame all-rounder but its jaw switches sides of the spout rod at the close positions —
a genuine ambiguity (Priya confirmed the jaw labels correct). (3) The temporal-context model removes those jaw flips
and tightens angles but, trained without the unlabelled-video loss, is under-confident on clearly visible tongues.
(4) EKS over LP r4 + context keeps LP r4's tongue recall, the context model's jaw, and (at smooth_param 100) the
kinematics — the strongest option so far. (5) LP r4's extra no-contact licks look mostly like incomplete licks with
some false detections on the spout rod — pending Priya's review.

### Human review of no-contact licks (Priya, 2026-10-14)

88 candidates (40 per session, PS93 0814 / 0821, balanced over spout position and over how many models found each,
blind to model) + 8 hidden contact-lick controls. Controls: 8 / 8 judged real (reviewer calibrated). No-contact
candidates: 58 real, 19 not a lick, 3 unsure. Detections found by ONE model only were mostly false (15 / 20); found by
>= 4 models, all real. Priya: many real ones are "barely visible but there -- not the end of the world if our model
can't capture absolutely all of these", so PRECISION matters more than catching every faint lick.

| model | detected | real | not a lick | precision | share of the sample's real licks found |
|---|---|---|---|---|---|
| DLC r3 | 44 | 39 | 4 | 91 % | 67 % |
| DLC r4 | 62 | 51 | 9 | 85 % | 88 % |
| LP r3 | 34 | 34 | 0 | 100 % | 59 % |
| LP r4 | 69 | 57 | 10 | 85 % | 98 % |
| LP r4 ctx-sup | 42 | 40 | 0 | 100 % | 69 % |
| **EKS r4+ctx** (s 10) | 47 | 44 | 1 | **98 %** | 76 % |

The false detections come from LP r4 and DLC r4 (alone or together) and DLC r3; the real licks EKS misses are mostly
faint ones only LP r4 / DLC r4 found. The "share found" is over this disagreement-weighted sample, not the population
(the sample over-represents licks few models find). Reading: EKS r4+ctx gives the best precision / coverage balance
for incomplete licks; LP r4 alone finds nearly all of them at ~15 % false. Scored with `lp_stage/lick_review/
review_scored.csv` (answers in the review artifact's `lick_review` collection).

**CORRECTION (2026-10-14, from Priya's request for LP r4 false-positive / EKS false-negative sheets,
`lp_stage/lick_review/lp_r4_false_positives.png`, `eks_false_negatives.png`).** The EKS r4+ctx numbers above were
largely an ARTEFACT: when the context model said "not visible" its point is parked at the image centre, and the
occlusion variance floor did not stop that point pulling the smoothed track -- EKS tongue points sat up-right of LP
r4's (toward the centre) at likelihood ~0.5 (the mean of 1.0 and 0.0), so faint real licks fell below the detector
and some false ones were suppressed. Its 98 % precision and its 0 jaw flips came from that drag, not from good
judgement. Fixed (`scripts/eks_two_model.py`: a member below the cut takes the confident members' mean position, so
it neither pulls nor vetoes): EKS fixed behaves like LP r4 -- contact recall 98.8 / 99.2 %, review precision 88 %,
share of real 98 %, kinematics intact at smooth_param 100 (peak vy 5988 vs LP r4 6011 px/s), jaw x flips 17 vs LP
r4's 30 (pre). The jaw fix is only partial: when both models are confident but on opposite sides of the rod, EKS
splits the difference.

**What the review actually supports -- an explicit agreement rule for licks without contact** (contact licks are
confirmed by the DAQ anyway), on the 80 reviewed candidates:

| rule | precision | share of real found |
|---|---|---|
| LP r4 alone | 85 % | 98 % |
| detected by >= 2 models | 93 % | 91 % |
| LP r4 AND context model | 100 % | 67 % |

### LP r4 ctx-semi (semi-supervised temporal context, finished 2026-10-14 18:24)

`heatmap_mhcrnn` + the unlabelled-video temporal loss (epsilon [10, 10, 35, 10]), round-4 labels, WSL, 512 px, batch 1,
300 epochs (no early stop), memory-capped. Predictions `lp_stage/r3_vs_r4_lp/r4mhcrnn_semi_512/`; scored with
`scripts/model_lick_compare.py --review` (all models in one run, `session_poses/PS93_<d>/r4_eval/
model_lick_compare_semi_1014.csv`); EKS = fixed `eks_two_model.py`, LP r4 + ctx-semi, smooth_param 100
(`eks2_r4_semi_s100/`). Figures `r4_eval/cue_traces_compare_semi_cleaned.png`. Jump counts here are a NEW definition
(raw frames, both >= 0.4, contiguous source frames, identical for every model) and are not comparable with the table above.

| pre / acute | LP r4 | ctx-sup | **ctx-semi** | EKS r4+sup | EKS r4+semi |
|---|---|---|---|---|---|
| contact recall (%) | 98.5 / 99.6 | 96.6 / 96.3 | **97.6 / 97.1** | 98.8 / 99.2 | 98.5 / 99.6 |
| licks without contact | 149 / 88 | 59 / 81 | 138 / 95 | 144 / 88 | 160 / 95 |
| jaw x jumps > 30 px | 38 / 5 | 6 / 0 | **0 / 0** | 21 / 1 | 22 / 0 |
| tongue x jumps > 60 px | 3 / 1 | 1 / 2 | 4 / **16** | 2 / 0 | 3 / 4 |
| review: real / not detected (both sessions) | 57 / 10 | 40 / 0 | **55 / 6** | 57 / 8 | 58 / 10 |
| review precision / share of real | 85 / 98 % | 100 / 69 % | **90 / 95 %** | 88 / 98 % | 85 / 100 % |

**Reading.** The unlabelled-video loss fixed the supervised context model's under-confidence (share of real licks 69 ->
95 %) and kept its jaw: **zero jaw side-flips** in both sessions (LP r4 38 / 5). It is the most precise detector that
still finds nearly all reviewed licks (90 % / 95 %). Against it: contact recall 97.6 / 97.1 % (below the 99 % gate; LP
r4 98.5 / 99.6) and one-frame tongue teleports on the acute session (16 > 60 px; visible as vertical spikes in the
cleaned tongue-x panel). Angles at peak (contact licks) agree with the other models within ~1-2 deg, spread as tight as
ctx-sup. EKS r4+semi inherits LP r4's recall but NOT the clean jaw (22 flips pre): where both members are confident on
opposite sides of the rod, the ensemble splits the difference. **Natural candidate: choose per keypoint -- jaw from
ctx-semi, tongue from LP r4 or EKS r4+semi** (to decide with Priya; check which contacts ctx-semi misses first).

**Caveat on "zero jaw flips" (2026-10-15).** Part of it is ABSTENTION: on PS93 0814 the ctx-semi jaw is confident
(p >= 0.4) on only 55 % / 62 % of post-cue frames at close_L / close_center (LP r4 76 / 81, DLC r3 84 / 80; all
models >= 92 % at the other positions; acute 81 / 97 vs LP r4 92 / 98). On the DLC flip events (`session_poses/
jaw_lateral/jaw_flip_qc_models.png`): when the mouth is nearly closed and DLC jumps to the image-LEFT point (~300 px),
LP r4 mostly stays confident on the correct side (~355 px) or says "not visible"; ctx-semi mostly says "not visible".
When the mouth is wide open with the jaw beside / behind the rod, LP r4 and ctx-semi BOTH say "not visible" (trained
blank = occluded) while DLC tracks it (+60 px, judged correct by Priya except one). Jaw coverage per position belongs
in the selection rule next to flip counts.

**PS93 jaw anatomy (Priya, 2026-10-15).** PS93's jaw has two candidate "lowest points": the pronounced true one,
deviated toward the MOUSE'S LEFT (image-right) -- what the labels mostly mark -- and a smaller local one on the mouse's
right (image-left). When the true one is not clearly visible, models pick the other: that is the PS93 jaw side-flip
(every right-column event in `jaw_flip_qc.png`; one left-column event, f3489). Attributed to PS93's pre-existing right
orofacial deficit. Consequences: (1) for PS93 an image-left jaw jump is a wrong landmark, an image-right excursion
usually real -- the cleaning's flip rule therefore only refills a run when the frames around it are on the correct point (flanks agree
and sit near rest; x only) -- adopted 2026-10-15; (2) PS93's jaw lateral readout (to-do 10)
mixes jaw shape with movement, and a "change" can be the model choosing differently -- read it from a model that picks
consistently, and treat PS93's static leftward bias as anatomy.

### LP r3 ctx-semi (round-3 labels, same recipe as r4 ctx-semi; finished 2026-10-15 16:50)

Scored with the cleaning tweaks ON (`model_lick_compare_r3semi_1015.csv`; LP r4 contact recall reads 98.8 % pre here,
98.5 % before the tweaks). Pre / acute: contact recall 97.3 / 99.6 %; review real / not 10 / 1 and 19 / 0 (finds 29 of
the 58 reviewed real licks; LP r4 57); raw tongue x jumps > 30 px 310 / 253 (the jumpiest model; LP r4 35 / 45); jaw x
jumps > 30 px 23 / 4 (LP r3 38 / 8). **Not a contender**: conservative on faint licks and jumpy on the tongue; the
round-4 labels matter more than temporal context. Closes the r3-vs-r4 context comparison.

### Jaw source after cleaning v35 (2026-10-15)

PS93 0814, cleaned jaw (v35), post-cue 70-3000 ms: coverage (frames with a measured or interpolated value) / jumps > 30 px:
DLC r3 98.8 % / 26; DLC r4 95.4 % / 7; LP r4 96.2 % / 0; ctx-sup 89.6 % / 0; ctx-semi 90.0 % / 0; EKS r4+semi 95.6 % / 0
(close_L coverage: DLC r3 97, LP r4 81, ctx-semi 61 %). PS93 0821: all >= 97 % coverage, <= 6 jumps. Priya judged LP r4's
jaw points correct on trial 47 close_center (1.4-2.1 s), where ctx-semi abstains and DLC r3 twice took PS93's fake
mouse-right minimum. **LP r4 = best jaw source. PRODUCTION MODEL = LP r4 alone (Priya, 2026-10-15)**; after cleaning, EKS's raw tongue-jitter
advantage is gone (one-frame spikes 2 vs 0-1 per 10k frames). Model archived at `DeepLabCut/Widefield/lightning-pose/
cam4-2026-10-13/models/r4_heatmap_20261013/`; O2 inference: runbooks/o2_pose_inference.md (Lightning Pose).

### LP r4 ViT-S DINOv2 (vision-transformer backbone, local, 2026-10-16)

`vits_dinov2`, otherwise the LP r4 recipe (round-4 labels, temporal loss), **512 px** (LP needs multiples of 128; the
8-GB card), batch 4, WSL guarded; training cut at epoch 270 / 300 by a machine restart -> best checkpoint epoch 254
(LP r4's best: 159). `lp_train_vit_r4_wsl.sh` / `lp_predict_vit_r4_wsl.sh`; predictions `lp_stage/r3_vs_r4_lp/
r4_vits_dinov2_512/`; scored `session_poses/PS93_<d>/r4_eval/model_lick_comparevits_1016.csv`.

| pre / acute | LP r4 | ctx-semi | ViT-S 512 |
|---|---|---|---|
| contact recall (%) | 98.8 / 99.6 | 98.0 / 97.1 | 98.0 / 98.3 |
| review: share of real licks found (%) | 100 / 97 | 89 / 100 | 67 / 90 |
| review precision (%) | 82 / 88 | 89 / 91 | 95 / 93 |
| tongue x jumps > 30 px (raw) | 35 / 45 | 31 / 58 | 294 / 57 |
| jaw x jumps > 30 px (raw) | 38 / 5 | 0 / 0 | 140 / 12 |

**Reading.** Not better than LP r4: the most precise detector so far but misses faint licks (like temporal context),
and very jumpy on the pre clip only. Angles at peak agree with LP r4 within ~1-2 deg. Not a final verdict on ViTs:
512 px vs 640, the SMALL ViT, training cut at 254 epochs. The O2 run (job 56522565: ViT-B DINOv2 640 px b8, ResNet-50
640 px b8, ctx-semi 640 px b4; `scripts/o2_lp_train_bundle.py`) is the real test, scored with the same protocol.

### O2-trained 640 px models (job 56522565, scored 2026-10-18)

Trained on O2 from `scripts/o2_lp_train_bundle.py` (`DeepLabCut/Widefield/o2/lp_train_20261016/models/`), predictions
on the bundle's eval clips; scored with the same protocol (`scripts.model_lick_compare ... --tag o2_1018`;
`session_poses/PS93_<d>/r4_eval/model_lick_compareo2_1018.csv`).

| pre / acute | LP r4 | ctx-semi 512 (local) | ViT-S 512 (local) | ResNet-50 640 (O2) | ctx-semi 640 (O2) |
|---|---|---|---|---|---|
| contact recall (%) | 98.8 / 99.6 | 98.0 / 97.1 | 98.0 / 98.3 | 99.0 / 99.2 | 98.5 / 97.9 |
| review: share of real licks found (%) | 100 / 97 | 89 / 100 | 67 / 90 | 85 / 87 | 100 / 97 |
| review precision (%) | 82 / 88 | 89 / 91 | 95 / 93 | 96 / 96 | 87 / 91 |
| jaw x jumps > 30 px (raw) | 38 / 5 | 0 / 0 | 140 / 12 | 114 / 23 | 7 / 3 |
| tongue x jumps > 30 px (raw) | 35 / 45 | 31 / 58 | 294 / 57 | 48 / 33 | 52 / 57 |

**Reading.** ctx-semi at 640 px is the first model at least as good as LP r4 on everything that matters: same share of
real licks, higher precision, far steadier jaw (raw tongue jumps slightly higher; cleaning removes one-frame jumps);
angles at peak within ~1-2 deg of LP r4. ResNet-50 640: most precise, misses faint licks, jumpy jaw on the pre clip.
Small review (~27-31 real licks per session). NOT switched: a production change means re-running all O2 inference,
cleaning and every downstream analysis for a modest gain -- revisit with the ViT-B result. ViT-B DINOv2 640 FAILED at
the first validation batch: the O2 `lp` env's transformers is incompatible with LP 2.4.2's ViT backbone
(`patch_embeddings.patch_size` int; local transformers 5.17.0 works) -- pin and resubmit array task 1.

### cam4 models training now (2026-10-18) -- to score with section 4's protocol

| id | settings | why | where / job |
|---|---|---|---|
| **ViT-B DINOv2 640** (rerun) | `vitb_dinov2`, 640 px, batch 8, temporal eps 10, round 4 | the first run died on the O2 `transformers` clash; the real test of a ViT backbone after the inconclusive local ViT-S (512 px, small, cut at 254 epochs) | `o2/lp_train_20261016/models/vitb_dinov2_640_b8`, job 56700781 |
| **LP r4 eps35** | LP r4 recipe exactly (heatmap, ResNet-50, 640 px, batch 4, round 4) but temporal eps **[nose 10, jaw 10, tongue 35, spout 10]** | LP r4 predates the 10-14 per-keypoint epsilon: its 10 px tongue limit penalises roughly the fastest quarter of real tongue steps (cam4 tongue p99 23-34 px/frame). Only the context models ever got 35, so this isolates the epsilon's effect on the production recipe (Priya, 10-18) | `o2/lp_train_20261016/models/heatmap_640_b4_eps35`, `train_heatmap_640_b4_eps35.sbatch` |

If LP r4 eps35 beats LP r4, switching production means re-running O2 inference, cleaning and every downstream analysis -- Priya's call once
the numbers are in.

## 4. How we choose (proposed 2026-10-14; Priya to confirm)

The model is for the ANALYSES — executed angle, lick timing / counts, kinematics across recovery — so it is judged on
those, on sessions it never saw, across epochs, in this order:

1. **No epoch-dependent bias (gate).** A model whose accuracy changes between pre / acute / chronic manufactures or
   hides a deficit. Every metric below is computed per epoch; a model that degrades post-stroke relative to pre is
   out, whatever its average.
2. **Contact recall (gate), over FINDABLE contacts.** A contact is found when a kept lick's window (onset - 60 ms ..
   max(offset, peak + 120 ms)) contains it, so a few frames hidden behind the spout around contact are tolerated: the
   lick is detected from the visible part of the way out / back, with tongue gaps <= 56 ms interpolated. Where the
   tongue is hidden for nearly the WHOLE lick, no cam4 model can find it, so recall is computed over contacts that at
   least one model finds (a contact missed by every model = unobservable from cam4, excluded). Gate: >= 99 % overall
   AND no spout position below ~97 %; a tripped position gets an image check of its misses. Measured 2026-10-14
   (PS93 0814 / 0821): only 1 / 0 contacts missed by every model; 23 / 15 missed by some model and found by another
   (the tongue WAS trackable there), concentrated in DLC r4 and the ctx-sup model at close_L / close_center.
3. **Incomplete-lick accuracy (primary; Priya reviews, 2026-10-14).**** Precision and recall of no-contact licks against a HUMAN-REVIEWED sample
   (~40 candidate licks per session marked real / not / unsure by Priya, blind to model; 4 contact licks per session as hidden positive controls). Candidates: `scripts/lick_review_candidates.py` (key with the detecting models kept apart in `lp_stage/lick_review/key.csv`); review page: claude.ai artifact `UURfzx99WnTxUEAbegZhsy` (answers in its `lick_review` collection). This is where models differ most and what matters
   most post-stroke; no automatic measure substitutes for it.
4. **Angle and kinematics (primary).** Angle at peak protrusion on contact licks: per-position spread (tighter =
   less noise) with medians agreeing across models; peak velocity and protrusion within ~5 % of the unsmoothed
   models (no smoothing bias).
5. **Stability, WITH coverage:** jaw side-flips and tongue teleports (jumps > 30 / 60 px) per 10k frames, always next to
   the fraction of frames each model reports (confident raw and cleaned, per spout position) -- a model can reach zero
   flips by abstaining (ctx-semi at close_L / close_center, 2026-10-15). Count AFTER the production cleaning too.
6. **Pixel error on frames held out by every model** — a sanity check only.
7. **Practicality:** inference cost for the full cohort (desktop / O2) and simplicity (one model beats an ensemble
   at equal performance).

Test set: at least one pre, one acute and one chronic session for two animals, none labelled (the PS93 0814 / 0821
clips plus e.g. PS93 0908, PS94 0818, PS95 0821, PS94 0921); the review clips for the visual check.

---

# Part B -- cam1 (view from below), round 2 (2026-10-18)

Same keypoints (**nose, jaw, tongue, spout**), same TARGET vs CONTEXT rule. Analysis cutoffs on cam1: **tongue 0.4, jaw / nose
/ spout 0.6** (DECISIONS 2026-10-05, evening). cam1 is for the executed-angle checks and the cam1 + cam4 3-D / multi-camera EKS
route; cam4 stays the primary tongue view. Background: `docs/status/STATUS_2026-10-05_CAM1_LP_EKS_O2_HANDOFF.md`.

## B1. Labels

| round | frames / sessions | what was added | why |
|---|---|---|---|
| 1 (10-05) | 294 / 11 (DLC staging) | lick-phase targets + spout-contact context frames | first cam1 model |
| **2 (done 10-18)** | **634 / 21** | Part 2: the four empty round-1 folders (PS93 0606, PS92 0820, PS95 0821, PS94 0903). Part 3 (254 targets, 6 new sessions): cam1 frames at cam4 round 4's 61 picks; 106 hard frames from the round-1 DLC / LP scan (incomplete tongue in the 30-70 px opening band, dropouts, unsure tongue, jaw near spout, erratic, tricky spout, rise / peak / fall); 90 tongue "also label" neighbours (Priya: 12 ms matters for the tongue) | round 1 overfit jaw / tongue (7 training sessions) and was ambiguous about the tongue tip from below |

Fill (round 2): nose 427, jaw 593, tongue 310, spout 605 of 634. Three targets left blank (2 in PS92 0821, 1 in PS95 0821) are
dropped. Frame provenance: `<DeepLabCut/Widefield>/frame_manifest.csv` (moved out of `_frame_staging` 10-18; `dlc_frames.manifest_path`).

## B2. The models

| id | type | labels | settings | why | where |
|---|---|---|---|---|---|
| **DLC c1r1** | DLC 3.0.1 ResNet-50, single frame | round 1 | 2pRAM donor init, DLC defaults, session-held-out split | first cam1 model | `training/...-orofacial-cam1`, iteration-0 `trainset71`, best-40 |
| **DLC c1r2** | same | round 2 | same recipe, `dlc_train --cam cam1 --iteration 1` (round 1 kept) | the DLC arm of the round-2 retrain | iteration-1 `trainset72`, **best-30** |
| **LP c1r1** | LP `heatmap`, ResNet-50 | round 1 | 640 px, batch 4, temporal eps 10, early stopping OFF | first cam1 LP; not better than DLC on cam1 (10-05) | WSL `/root/lp/cam1-2026-10-05/models/cam1_occl_uniform_20261005` (epoch 144) |
| **LP c1r2 heatmap** | LP `heatmap`, ResNet-50 | round 2 | **cam4 LP r4 recipe exactly**: 600 -> 640 px, batch 4, temporal eps 10, early-stop patience 20, blanks = uniform heatmap | like-for-like with the cam4 production model | O2 `o2/lp_train_cam1_r2_20261018/models/heatmap_640_b4`, job 56709413_1 |
| **LP c1r2 ctx-semi** | LP `heatmap_mhcrnn` (frames t-2..t+2 at inference) | round 2 | 640 px, batch 4 + 8-frame context batch, temporal eps [10, 10, 35, 10] (cam4's 10-14 values) | the 640 px context model matched LP r4 on cam4 with the steadiest jaw | `.../models/mhcrnn_semi_640_b4`, job 56709413_2 |
| **LP c1r2 eps40** | LP `heatmap`, ResNet-50 | round 2 | LP r4 recipe with temporal eps **measured on cam1: [nose 10, jaw 15, tongue 40, spout 10]** | the temporal loss should penalise only implausible jumps; cam1 sees faster tongue / jaw motion than cam4 (B3) | `.../models/heatmap_640_b4_eps40`, `train_heatmap_640_b4_eps40.sbatch` |

All LP c1r2 models train on all 634 labels (LP holds out frames, not sessions) with round 1's 15 unlabelled clips (one per animal
x epoch) for the temporal loss. Project `/root/lp/cam1-2026-10-18` (`litpose convert` of the DLC training copy, then
`lp_labels clean`); bundle `scripts/o2_lp_train_bundle.py --cam cam1` (add-ons: `--only <model> --configs-only`). No ViT for cam1
unless the cam4 ViT-B 640 rerun beats LP r4.

## B3. Temporal-loss epsilon measured on cam1 (10-18)

Genuine frame-to-frame motion in px/frame, from DLC predictions (no temporal loss), both frames confident (cam1 cutoffs),
consecutive SOURCE frames only. DLC c1r2 on PS93 0814 (60 trials; ~2 x 10^5 steps per part, 19k for the tongue), pooled with DLC
c1r1 on the six round-1 clips (PS92 / PS93 / PS95):

| part | p50 | p95 | p99 | p99.9 | cam4 p99 (10-14) |
|---|---|---|---|---|---|
| nose | 0.2 | 1.5 | 2.6 | 4.3 | 1-3 |
| jaw | 0.5 | 5.6 | 10.9 (pooled 11.3) | 21.4 | 4-6 |
| tongue | 5.4 | 20.7 | 37.2 (pooled 40.2) | 60.3 | 23-34 |
| spout | 0.1 | 2.3 | 3.5 | 5.3 | 1-3 |

Same rule as cam4: epsilon ~ p99 for the moving parts, 10 px floor otherwise -> **[10, 15, 40, 10]** (jaw 15 sits between p99 and
p99.9, so large jaw jumps are still penalised). cam1 looks up at a tongue and jaw moving toward it, hence the faster image motion.
Caveat: the tongue p99 includes some tip-end switches (the cam1 tip ambiguity), so 40 may be slightly generous; it is still below
p99.9 (60). To re-check on PS93 0821 (acute).

## B4. Results so far

**DLC c1r2, held-out sessions** (7 sessions, 176 frames, every epoch; DLC's mean error): nose 2.6, jaw 9.9, tongue 12.3, spout
1.9 px. Per-epoch test RMSE, tongue: acute 14.6 / chronic 12.0 / pre 21.5 / subacute 18.0; jaw: 14.2 / 10.0 / 13.1 / 10.0. No
post-stroke penalty (pre is worst, driven by PS93 0606 frames, one tongue 70 px off); subacute spout 6.2 px (one session) to check.
Not comparable with round 1's test numbers (different held-out sessions).

**Frames held out by BOTH DLC rounds** (40 frames: PS95 0606 pre, PS95 0817 acute):

| part | c1r1 median px / > 10 px / found | c1r2 median px / > 10 px / found |
|---|---|---|
| nose | 1.8 / 0 % / 100 % | 3.4 / 0 % / 100 % |
| jaw | 10.2 / 53 % / 78 % | **6.5 / 33 % / 83 %** |
| tongue (13 labelled) | 8.3 / 38 % / 92 % | 12.0 / 54 % / 92 % |
| spout | 1.8 / 0 % / 97 % | 1.3 / 0 % / 97 % |

Jaw better; tongue not, on too few frames to say -- consistent with the tip-definition ambiguity rather than missing labels.

**Round 1 vs round 2 vs LP on the PS93 cam1 clips** (`scripts.cam1_model_compare`, tag `_r1r2lp_1018`; 0908 r1 pending):

| pre / acute | DLC c1r1 | DLC c1r2 | LP c1r2 heatmap |
|---|---|---|---|
| contacts seen (%) | 98.8 / 95.9 | 99.8 / 97.1 | **100 / 99.2** |
| tongue > 60 px per 10k | 30 / 39 | 9.5 / **4.0** | **5.1** / 17.4 |
| jaw coverage post-cue (%) | 74 / 80 | 86 / 85 | **95 / 87** |
| jaw > 20 px per 10k | 39 / 33 | 12 / 3.3 | **3.3 / 3.2** |
| jaw coverage, centre positions pre (%) | 35-36 | 54-69 | **82-90** |

Round 2's labels cut teleports and jaw jumps 3-10x and fixed most of the centre-position jaw. Priya (labelled clip): LP is
mostly good but sometimes sits to one side of the tongue as it emerges. NB the plain LP c1r2 uses the LP r4 recipe's temporal
epsilon of 10 px for EVERY keypoint (tongue too) -- below cam1's genuine fast tongue motion (p99 40 px, B3), so it may hold the
emerging tip back; eps40 (same model, tongue 40) and ctx-semi (temporal context, tongue 35) test that.

**LP c1r2 eps40 scored (10-18, tag `_eps40_1018`; O2 job crashed in the post-training labelled-frame eval, predictions from a
predict-only job, so no pixel-error table).** Pre / acute / chronic, eps40 vs plain LP c1r2: contacts seen 99.5 / 98.8 / 99.0
vs 100 / 99.2 / 99.2; tongue > 60 px per 10k 5.6 / 20.6 / 15.6 vs 5.1 / 17.4 / 12.9; tongue coverage 20.7 / 10.5 / 25.9 vs
21.0 / 12.2 / 28.9 %; jaw jumps 3.2 / 2.4 / 6.1 vs 3.3 / 3.2 / 5.3; onset within 12 ms of cam4 69.6 / 56.3 / 68.3 vs 66.6 /
54.8 / 65.2 %. **Essentially the same model**: the cam1-measured tongue epsilon buys slightly better cam1-cam4 timing at
slightly more teleports and a little less coverage. Whether it fixes the "beside the emerging tongue" offset is a visual
question: `session_poses/PS93_20260814_cam1/eval/cleaned_labels_cam1_dlc2_lp2_eps40_gap24.mp4` (same 16 trials, incl. 507).

**All five cam1 candidates on PS93 (10-19, tag `_all5_1019`; ctx-semi = `heatmap_mhcrnn`, tongue eps 35).** Pre / acute /
chronic, tongue > 60 px per 10k: DLC c1r1 30 / 39 / 36, DLC c1r2 9.5 / 4.0 / 17.4, LP 5.1 / 17.4 / 12.9, eps40 5.6 / 20.6 /
15.6, **ctx-semi 10.8 / 18.0 / 12.5**; contacts seen ctx-semi 99.8 / 99.2 / 99.8; jaw coverage 95 / 84 / 99; jaw jumps 4.2 /
2.3 / 3.6. The three LP variants are statistically indistinguishable on these measures; the choice between them is the
visual one (emergence offset): `.../eval/cleaned_labels_cam1_4models_gap24.mp4` (DLC c1r2, LP, eps40, ctx-semi).
**Priya's QC of that clip (10-19):** all four good on clear licks; plain LP c1r2 sits MEDIAL / up in the tongue body as the
tongue emerges and corrects to the tip more slowly than the others. Measured (PS93 0814 / 0821 / 0908, 197 emergences,
offset down the image toward the tip vs the median of the other three): LP c1r2 -1.7 / -1.7 / -1.8 px at 12-20 / 24-44 /
48-96 ms, the only model consistently above the tip; eps40 -0.4 .. -0.1, ctx-semi +0.1 .. +0.7, DLC c1r2 +1.2 .. +2.4. Small
on average (> 10 px misses 0-3 % of frames for every model) but systematic -- the 10 px tongue epsilon holding the emerging tip
back. -> plain LP c1r2 dropped; cam1 choice between LP eps40 and ctx-semi (tied on the scores; eps40 cheaper on O2, fewer
pre teleports). Frames where ALL models miss the tip (e.g. trial 507 +904 ms, tongue part-out / retracting) -> next cam1
labelling round, with the water-drop frames.
**cam1 nose (10-19).** Steady when tracked (median step 0.2 px, p99 ~2 px, no jumps) but "confident" in only 40-68 % of
frames -- BY DESIGN: at the two CENTRE positions the spout rod covers the nose from below, and the labels leave it blank there
(nose labelled in 1 % of centre frames vs 91-100 % at the sides), so the models correctly say "not visible" (ctx-semi nose
lk >= 0.5 at centre 2-4 %, sides 80-94 %). Acute PS93 0821 side positions are lower (58-69 % at close_L / far_L / far_R).
DLC's nose likelihoods are also less decisive (median 0.91 pre, 0.27 acute) -- use lk >= 0.5, not 0.9, for the nose. Fine
as a session-level head reference (median of side-position frames); not a per-frame signal at the centre.
**Faint licks (10-19, Priya: "but eps40 misses faint licks?"):** share of cam4 LP r4 licks (PS93 clip trials; LP r4 found
99-100 % of real faint licks in both blind reviews) with a confident cam1 tongue within +-20 ms of cam4's peak, pre / acute /
chronic. No-contact (n = 151 / 88 / 165): **ctx-semi 90.7 / 89.8 / 95.8**, LP c1r2 88.7 / 87.5 / 94.5, eps40 86.8 / **79.5** /
92.1, DLC c1r2 81.5 / 80.7 / 96.4. Contact licks: all 96-100 %. eps40 repeats eps35's cam4 pattern (loses faint licks, most
acutely); **ctx-semi sees the most faint licks in every epoch AND fixes the emergence lag -> the recommended cam1 model**
(cost: temporal-context inference is slower on O2). Clip re-rendered with an outlined mouth X (the grey X vanished on the pale
tongue / water under the spout): `.../eval/cleaned_labels_cam1_4models_gap24_v2.mp4`.

**Cross-camera agreement on a second animal (10-19, PS94 0814 / 0821 / 0907, cam1 DLC c1r2, cam4-timed peaks).** cam1 lateral
vs cam4 peak angle, Spearman, all licks: far_L 0.61, close_L 0.79, close_center 0.64, far_center 0.90, close_R 0.73, **far_R
0.93** (PS93 DLC c1r2: close_R 0.35, far_R 0.23). The right-side disagreement is PS93-specific (its near-mouth events); with
the tongue > 30 px out PS93 reaches 0.75-0.78 there too.

---

# Part C -- Evaluation plan and status, both cameras (2026-10-18; cam1 metrics agreed by Priya 10-18; she picks the winner)

Every candidate is scored by the SAME scripts on the SAME clips, never by its own training metrics (LP holds out frames, not
sessions, so its training numbers are not comparable with DLC's).

**cam4.** Candidates: LP r4 (production), ctx-semi 640, ResNet-50 640 b8, ViT-B 640 (rerun), LP r4 eps35. Protocol = section 4
(epoch gate, contact recall >= 99 %, Priya's blind lick review, angles / kinematics, stability with coverage, practicality) on the
PS93 0814 / 0821 trial clips + review clips (`scripts.model_lick_compare ... --review`).

**cam1.** Candidates: DLC c1r1, DLC c1r2, LP c1r1, LP c1r2 heatmap, ctx-semi, eps40. Test clips:
`session_poses/PS93_{20260814,20260821,20260908}_cam1/windows.mp4` (pre / acute / chronic; the SAME 60 trials as the cam4 clips,
cut with `scripts.session_poses clip --cam cam1`; no cam1 labels in these sessions). Second animal still to add (gate 1 asks for
two): PS94 0814 / 0821 / 0908 (no PS94 cam1 labels on those dates). Metrics, in order:

1. **Epoch gate.** Every metric per epoch; a model that degrades post-stroke is out.
2. **Tongue at DAQ contacts.** Share of spout contacts with a confident (>= 0.4) cam1 tongue in the contact window, per position
   (the cam1 analogue of contact recall; far_L, sideways from below, was round 1's worst).
3. **Tip consistency.** One-frame tongue jumps > 40 px per 10k confident frames, next to coverage; plus agreement of cam1 vs cam4
   lick timing (onset / peak on the DAQ clock) for the same licks.
4. **Jaw.** Coverage per position next to jaw jumps > 20 px per 10k frames (abstaining must not pass for stability).
5. **Visual check.** Priya reviews a blind sheet of the frames where the candidates disagree most.
6. **Pixel error** on frames held out by every DLC round (sanity only; the LP models trained on all labels have none).
7. **Practicality.** Inference cost for the cohort; one model beats an ensemble at equal performance.

**Status**

| step | cam4 | cam1 |
|---|---|---|
| trained | LP r4, ctx-semi 640, ResNet-50 640 done; ViT-B rerun running (56700781); LP r4 eps35 to submit | DLC c1r1, c1r2 done; LP c1r2 heatmap + ctx-semi running (56709413); eps40 to submit |
| predictions on the test clips | each O2 task predicts the bundle's eval clips | DLC c1r1 / c1r2 running locally (`session_poses/PS93_<d>_cam1/dlc_r{1,2}/`); LP c1r2 models predict them on O2; LP c1r1 to run |
| scored | LP r4 / ctx-semi 640 / ResNet-50 640 (10-18) | not yet -- cam1 scoring script (metrics 2-4) to write |
| Priya's review | lick review done for the earlier models | not yet |
| decision | production = LP r4 (10-15); revisit after ViT-B + eps35 | not yet |

---

# Part D -- 3-D output and cam1 clean traces (Priya 2026-10-18: 3-D IS an analysis output)

## D1. The 3-D route (order)

1. **Single-view models chosen per camera** (Parts C / B, in progress).
2. **cam1 clean traces** (D2): gaps filled consistently and one-frame tip-end switches removed BEFORE triangulation -- the
   10-05 calibrated EKS failed because occluded / wrong cam1 points set the 3-D depth (likelihood-blind initialisation).
3. **Multi-view LP candidate** (cam1 round 2 + cam4 round 4, split-rule `visible`, no calibration -- the 10-06 pilot found
   calibration added nothing). Why retry: the pilot's one win was the cam1 jaw behind the spout rod (99 vs 87 % confident),
   exactly the centre-position jaw gap measured on DLC c1r2 (jaw confident 69 % close_center, 54 % far_center vs >= 96 %
   elsewhere); the pilot used older labels on both cameras. LP 2.4.2 has no multi-view + context model, so if a context model
   wins single-view, the multi-camera route is per-camera context models + the calibration-free multi-camera EKS
   (`pose_eks`, smooth_param explicit). Scored like the single-view candidates.
4. **anipose triangulation** (`wfield_local/dlc_anipose.py`, cam1-cam4 calibration: 22 board poses, 6-9 px reprojection on
   labels) on the chosen cleaned traces, likelihood-filtered first; compared with the calibration-free multi-camera EKS.
   Checks: reprojection error per keypoint, the spout tip's 3-D position against the commanded stage positions
   (`spout_world`, within-session spout distances matched the command to 1-2 %), 3-D lick extent / angle per epoch.

## D2. cam1 clean traces

`orofacial_clean` now takes a camera (`params(bodypart, cam)`, `clean_windows(..., cam=)`, CLI `--cam`): cam4 = the
top-level `orofacial_clean` values, untouched (pinned by tests); another camera = those values overlaid with
`orofacial_clean_<cam>` in `configs/defaults.yaml` (a separate block: `orofacial_outputs.params_digest` hashes `orofacial_clean`). **No cam1 overlay is set yet** -- the values must come from cam1
data with Priya's QC, as cam4's v35 did.

**Baseline diagnostic (cam4 values on DLC c1r2, PS93 0814, 60 trials, 10-18).** cam1 looks UP at the underside of the
snout: nose at the top, mouth / jaw low in the image (jaw median y 456), the spout rod running down the middle of the face
to the mouth (hence the nose blanks and the centre-position jaw occlusion). The tongue comes out of the mouth DOWN the
image, over the chin (labelled tongue median y 431 vs mouth y 389; QC frames 10-18) -- an earlier note here said "up the
image", which was wrong.

* **Mouth origin (QC 10-18, `scripts.cam1_clean_qc mouth`, `session_poses/PS93_20260814_cam1/eval/qc_mouth_origin.png`):**
  cam1 looks almost straight down the CENTRE spout line (close->far 3.4 px vs L / R 48 px; cam4 56 / 66 / 68). The L and R
  lines meet within 0.3 px at (315, 389), on the lower lip just under the centre spout tip; both centre spouts lie 4-7 px
  from it. Projection keeps incidence, so this IS the mouth's image (Priya to confirm by eye). BUT `spout_frame.ap_axis`
  comes from the centre line, which is noise on cam1 -> a cam1 axis must be the L / R bisector; cam1 angles are LR / DV, not
  AP. Good for 3-D: the two cameras see AP very differently (cam4 56 px, cam1 ~3 px).
* **Tongue:** confident on 9 % of clip frames; cleaning keeps 9 % raw, interpolates 2 %, fills 89 % with the baseline
  (tongue in the mouth). Steps > 40 px rise from 160 raw to 806 after cleaning -- mostly the jump between the baseline fill and
  a protruded tongue at every lick onset / offset, so on cam1 the metric must exclude fill transitions.
* **Jaw:** confident 95 %; v34 halves jaw steps > 20 px (629 -> 284). (The cleaning's "lowest-y" X0/Y0 lands at y 372 vs a
  median of 456; v34 itself uses per-trial rests from cue -500 .. 0 ms, which are direction-free.)
* **Jaw QC (10-18, `scripts.cam1_clean_qc jaw`, `.../PS93_20260814_cam1/eval/qc_jaw.png`):** the cam1 jaw opens DOWN the image,
  like cam4 (only 3 % of tongue-out frames move up). Displacement from the trial's rest, tongue out: dy median 77, p95 123,
  **p99 139 px**; |dx| p99 60 px (tongue not out: dy p99 84). So cam4's `y_ceiling_px` 125 would wipe ~2 % of real opening on
  cam1 -> propose **160**; `x_ceiling_px` 75 clears |dx| p99. Jaw within 25 px of a confident out tongue: 1.3 % of tongue-out
  frames; the examples look like the jaw point sliding onto the tongue tip where the protruded tongue covers the chin (Priya's
  clip, 10-18: "sometimes jaw ends up tracking tongue") -- Priya to confirm before cam4's `jaw_off_tongue` rule is carried over.

**To tune, each with a QC figure for Priya:** (1) tongue origin = cam1 mouth (spout-line intersection, checked on frames);
(2) jaw baseline direction / resting jaw on cam1; (3) the px thresholds marked "OLD RIG" (x-range, jaw ceilings, frame-jump
thresholds) rescaled from cam1 motion (B3: tongue p99 40, jaw p99 11 px/frame); (4) the PS93 jaw-flip rules (cam4 anatomy,
v35) OFF on cam1 until shown to apply; (5) jaw cutoff 0.6 (cam1 analysis cutoff). Tune on DLC c1r2 now; re-check on the
chosen cam1 model.

## D3. June camera scale -- cam1 vs cam4 (10-18)

cam4 June 0602-0608 is on a 12.9 % larger scale (v36, DECISIONS 2026-10-18 evening: x 0.8853 before cleaning). **cam1 is
not**: the same ruler idea (pairwise distances between the six spout positions, per session, relative to the Aug+ median for
that pair) on the LABELLED cam1 spout tips gives June 0606 = 0.988 / 0.994 / 0.996 / 1.009 (PS93 / PS92 / PS94 / PS95; median
0.995) vs Aug+ median 1.000 (IQR 0.999-1.008, 17 sessions). So cam1 gets NO June factor (`px_scale` 1.0); only cam4's
configuration changed. To confirm with the v36 method (no-lick ENL frames, `spout_frame.ruler_*`) on DLC predictions of the
June cam1 sessions -- labelled spouts sit on lick frames (pushed 2-5 px), though a ratio of pairwise distances is robust to that.

**Consequence for 3-D.** Triangulation uses ABSOLUTE image positions (the 09-11 calibration describes Aug+ cam4). v36
multiplies June cam4 x, y about the image ORIGIN (pixel 0, 0): right for 2-D px measures relative to the mouth, but every
point also moves (an image-centre point by ~39 px), so v36 pose is not triangulation input as is. Candidate June corrections,
cheapest first (Priya 10-18: "can we use the scaled image mouth-centered?"):
1. **Mouth-centred scaling** (x 0.8853 about the session's mouth, mouth kept in place). If the June -> Aug change was a zoom
   about centre c, the residual error is (1 - 0.885) x |mouth - c| -- ~3-6 px if the mouth is 30-50 px from the optical
   centre, within the calibration's 6-9 px reprojection error.
2. **Spout-fitted similarity** (scale + rotation + shift, 4 parameters from the six spout medians onto the Aug+ reference
   positions) -- contains 1 as a special case; needed if the shift matters.
3. **June cam4 recalibration** (intrinsics / extrinsics from the spout positions with cam1 fixed), only if 1-2 fail (camera
   moved rather than zoomed).
Tests: spout residuals in the cam4 image after 1 vs 2; triangulated June spout distances vs the commanded stage distances
(Aug+ match 1-2 %). The 2026-08-05 calibration video (board geometry unrecorded; cam4 4 poses, cam1-cam4 22 joint poses)
may show what changed (zoom vs move, zoom centre) IF it was taken in the June configuration -- check its date against the
cam4 ruler first. cam1 needs no correction. Likely source of the 10-05 "June ~5 % too large" 3-D spout fit.

**2026-08-05 = the day cam4 was re-set (measured 10-18, v36 ruler on the O2 LP r4 output).** cam4 scale vs Aug+ by session:
PS94 13:10 **0.80**, PS95 16:04 **0.77**, PS92 18:21 1.00, PS93 20:20 1.00 (June 0606 PS92 0.88; 0806 PS92 / PS94 1.00). The
camera was being adjusted that afternoon and reached the Aug+ setting by 18:21. The 08-05 calibration video was recorded at
**17:58** -- after the zoomed sessions, before the first Aug+-scale one -- so it almost certainly shows the Aug+ configuration,
NOT June's: no help for the June geometry; at most a second Aug+ calibration (board layout to resolve) for a rig-unmoved check
against 09-11. **v36 leaves PS94 / PS95 0805 unscaled** (`scale_before` 20260801) although they are 0.80 / 0.77 -- outside the
curated set (8/5 excluded) and every orofacial cohort analysis reads `analysis_kit.curated_sessions`, so NO result is
affected (checked 10-18); only those two LP_analysis session folders carry px measures inflated 25-30 %.

## D4. cam1 mouth, first attempt: where the tongue paths meet (superseded by D6) -- and results so far

**Mouth (option 1 now, option 2 for 3-D).** Priya (10-18): the spout-line intersection is not the cam1 mouth -- cam1 labels
the spout tip's BOTTOM edge (cam4 the top front edge), so the lines meet toward the jaw, and the tongue comes out "above" it.
`spout_frame.tongue_path_origin`: per spout position, the line from the median lick-onset tongue point (first confident frame
after >= 3 hidden ones, > 70 ms after the cue) toward that position's median spout tip; their least-squares meeting point.
DLC c1r2, PS93: **(310, 369) pre, (310, 367) acute** (paths meet to 8-10 px rms) -- 2 px apart, while the spout-label
intersection moved 20 px (389 -> 409) and the mouth relative to the centre spout tip moved 25 px: the spouts move between
sessions, the head under a fixed camera does not. cam1 axis = `from_medians_lr_bisector` (L / R far->close bisector).
Centre-spout angles are ill-defined on cam1 (the spouts sit almost on the mouth, end-on); read cam1 angles in 3-D.
Option 2 (3-D): cam4's apparatus mouth (lines meet to 0.34 px) triangulated / projected into cam1.

**Water (Priya 10-18).** Spout tip, late in the trial (>= 1500 ms, tongue not out) minus pre-cue ENL, per position: < 2 px on
cam1 and cam4, both PS93 sessions -- DLC c1r2 stays on the rod end inside a hanging drop (Priya's frame, trial 540). Still to
check: spout drift and pre-cue false tongue detections against TRIAL NUMBER (drops grow in late, sated trials).

**First scores, LP c1r2 heatmap vs DLC c1r2** (`cam1_model_compare_1018_dlc2_lp.csv`; PS93 pre / acute):

| | DLC c1r2 | LP c1r2 heatmap |
|---|---|---|
| tongue seen at DAQ contacts | 99.8 / 97.1 % | 100 / 99.2 % |
| tongue teleports > 60 px / 10k confident | 9.5 / 4.0 | 5.1 / **17.4** |
| tongue steps > 40 px / 10k (mostly genuine fast motion: 40 = p99) | 72.5 / 25.5 | 108.7 / 66.2 |
| jaw confident close_center / far_center | 69 / 54 ; 70 / 70 % | 90 / 82 ; 63 / 60 % |
| jaw jumps > 20 px / 10k | 12.3 / 3.3 | 3.3 / 3.2 |
| tongue onset vs cam4 (median abs) | 12 / 12 ms | 8 / 12 ms |

LP finds more contacts and is steadier pre, but its tongue teleports triple on the acute session -- an EPOCH-dependent
difference (gate 1); not called on one animal: needs PS93 0908 and PS94, and the ctx-semi / eps40 models. The > 40 px count
was first proposed as the tip-jump metric; at the cam1 p99 it counts ~1 % of real steps, so > 60 px (p99.9) is the teleport count.

**Water makes false TONGUE detections (Priya's frames 10-18: PS93 0814 trial 540, a large drop under the tip late in the
session; DLC c1r2 puts a confident tongue point at the bottom of the drop, spout correctly on the rod end).** Post-cue, trials
with no DAQ contact vs with contacts (PS93 0814): tongue confident 1.9-2.6 % vs 21-24 % of frames; DLC c1r2's no-contact
detections are slow and STATIC (median step 2.2 px, 23 % of frames < 1 px; contact trials 7.1 px, 4 %), LP c1r2's mostly not
(8 % static). 0821 (acute) is mixed -- many of its 31 no-contact trials carry real incomplete licks. Small in frames, but it
lands on licks without contact, the measure that matters most. Fixes, strongest first: (1) drop frames (drop under the tip,
tongue blank) as hard negatives in the next labelling round (late, sated trials); (2) a cleaning guard against static tongue
runs at the spout tip -- measure what it costs real licks first; (3) include late, sated trials in the cam1 lick review.
Check the cam4 LP r4 "false detections on the spout rod" (lick review 10-14) for drops too.

## D5. Option 2 built: the commanded (apparatus) mouth in each camera, from 3-D (10-18)

Priya 10-18: the cam1 mouth must not depend on the tongue paths (they change after the stroke) -- use the apparatus.
`spout_world.apparatus_mouth`: triangulate the six parked spout medians (cam1 + cam4, 09-11 calibration), fit the similarity
frame to the COMMANDED layout (`spout_world.fit`, mouth-centred stage mm from gui_config), and project the frame's origin --
the stage's mouth set point -- into each camera. `scripts/apparatus_mouth.py` (clips: cam1 DLC c1r2 + cam4 DLC r3 trial clips,
or the O2 LP r4 whole-session output for cam4 where there is no clip -- June).

| PS93 | cam1 mouth | - cam1 nose | cam4 mouth - cam4 nose | fit residual | distance ratios | nose in mouth frame (ml, ap, dv) mm | stage set point y |
|---|---|---|---|---|---|---|---|
| 0814 | (323, 388) | (+8, +250) | (-3, +82) | 0.05 mm | 0.97-1.00 | (+0.10, +6.47, +2.87) | 24.934 |
| 0821 | (313, 408) | (0, +274) | (-4, +85) | 0.05 mm | 0.95-1.00 | (-0.02, +7.25, +3.02) | 25.634 |

**Reading.** The triangulation is good (spouts within 0.05 mm of the command; reprojection 6-8 px), and the apparatus mouth
moves 24 px in cam1 between the two days -- the SAME shift as the cam1 2-D spout-line intersection, which therefore tracks the
apparatus mouth (within ~8 px) and was never noise. The cause is in gui_config: the mouth set point entered for 0821 differs by
0.70 mm in stage y, and the nose moved 0.78 mm relative to it -- the head stayed, the set point moved. cam4 hardly sees that
direction (its view runs along it), hence its stable +82 / +85. So the apparatus mouth = the task's reference by design, with
the day-to-day precision of the set point (~0.7 mm here); the tongue-path mouth is anatomical and stable (2 px) but
behaviour-dependent. Priya's choice (10-18): apparatus. Labelled frames cannot test June (2-4 positions per session in both
cameras; the fit needs 5): June cam1 0606 clips (4 animals, 4 trials / position) are cut and queued for DLC c1r2; cam4 June from
the O2 LP r4 output; the June variants (raw / x0.885 about origin, principal point, cam4 mouth) are then compared by
reprojection error and fit residual.

**June 3-D test (10-19, `apparatus_mouth clips` on the four 0606 sessions, cam1 DLC c1r2 + cam4 from the O2 LP r4 output;
`session_poses/apparatus_mouth_clips_june_1019.csv`).** Raw June cam4: triangulated spout distances 0.90-0.96 of commanded
(fit scale 0.950-0.957), residual 0.07-0.12 mm (Aug: 0.98-0.99, 0.03-0.05 mm). Scaling June cam4 by 0.885 brings the scale to
1.01-1.03 and the residual to 0.05-0.07 mm whichever centre is used -- BUT v36's centre (the image ORIGIN) moves the cam1 mouth
~20 px sideways relative to the nose (mouth - nose x -17 .. -18 vs +2 .. +15 raw and +8 / +11 in August), impossible for a
head-fixed mouse. Scaling about the cam4 PRINCIPAL POINT or the cam4 MOUTH keeps the cam1 mouth where it is (+3 / +2 px).
**For 3-D: scale June cam4 about the principal point (or the cam4 mouth -- indistinguishable here), not the image origin.**
v36's origin-centred scaling stays right for 2-D px measures. Reprojection error June 9-10 px in every variant (Aug 8).
**No v36 re-run (10-19):** the scaling centre only adds a constant shift (1 - f) x c to every point; every 2-D measure is a
difference (tongue / jaw relative to the mouth or spout frame, reach, lateral error, speed, angles; the jaw x / y ceilings are
measured from a baseline, `orofacial_clean.py` jaw rules), so it cancels. Only triangulation sees absolute positions, and the 3-D
step reads RAW June cam4 from the O2 output and scales it about the principal point itself.

**The calibration's principal points are not suspicious (checked 10-19).** They were PINNED (`fix_principal_point: true`), not
fitted: `dlc_anipose.measured_principal_point` = sensor centre minus the recorded ROI offset (`defaults.yaml dlc.sensor`,
Blackfly 1280 x 1024). cam1: ROI at (440, 312) -> (640 - 440, 512 - 312) = (200, 200); cam4: (640 - 308, 512 - 132) = (332, 380).
A free fit put them outside the frame (the 09-10 recording), hence the pin. With f ~ 5500 px a 100 px error equals a ~1 deg
rotation, which the extrinsics absorb; the August spout fits (0.05 mm, ratios 0.97-1.00) confirm the geometry.

## D6. DECISION (Priya 10-18): the cam1 mouth = the L / R spout-line intersection, no offset

`spout_frame.from_medians_lr`: origin where the L and R far->close spout lines meet, axis = their bisector (cam1 sees the centre
line end-on). Why: apparatus-only (the tongue-path point depends on tongue behaviour, which changes after the stroke); it
tracks the 3-D apparatus mouth within 5-8 px (D5), and DLC / LP and parked / all-frame spout medians give the same point within
2-3 px. Its day-to-day movement relative to the head is real set-point variation (~0.7 mm, D5), acceptable for a task-defined
reference. It sits ~15-20 px toward the jaw from where the tongue appears (cam1 labels the spout tip's bottom edge) -- kept,
no shift; cam1 thresholds measured from the mouth are set relative to this point. Adaptive trials need no filter (Priya: the
stage moves along the same ray, so a shorter far_L lies on the same line; pinned by a test). Works for June unchanged (cam1 has
no June scale change, D3). The tongue-path point stays as a QC comparison only.

**NOTE -- spout position jitter across days (Priya 10-18).** The spout layout is placed relative to a mouth set point entered
each session, and that placement jitters across days relative to the head: PS93 0814 vs 0821, the set point differs by 0.70 mm
(stage y) and the triangulated nose moved 0.78 mm relative to it (D5); cam1 sees this as ~24 px, cam4 hardly (it views along
that direction). Consequences:
* Measures relative to the SPOUT TIP -- where the tongue is aiming (angle to the target, executed vs target, contact) -- are
  unaffected by construction: the reference moves with the target.
* Measures relative to the MOUTH origin (protrusion distance, the in-mouth / tongue-out threshold, mouth-zero fills) inherit
  the jitter, because the origin follows the apparatus, not the anatomy. Most exposed: the SMALLEST licks -- where an
  incomplete lick near the lips falls relative to the origin (and to a fixed px threshold) can shift by up to ~0.7 mm between
  days. Read day-to-day changes in small-lick extent with this in mind; compare within a session where possible.
* Not yet quantified across the cohort: triangulating the nose into each session's mouth-centred frame (as in D5) gives the
  set-point jitter per session for every session with cam1 + cam4 spout positions.

## D7. The other tongue / jaw cleaning stages on cam1 (10-18; PS93 0814 / 0821, cam1 mouth = D6, cam4 LP r4 as reference)

Kept as cam4: the x-range filter (drops <= 0.02 % of confident cam1 points), the isolated-point rule (cam1 DLC drops 1.4-2.2 %,
LP 0.4-1.4 %, cam4 0.5-0.8 % -- single-frame flicker, the rule working), gap filling (tongue 1.9-4.8 % of post-cue frames
filled, cam4 2.5-3.4 %; jaw similar), the jaw rest ("lowest y" = closed: the cam1 jaw also opens down). CHANGED (proposed): the
between-lick veto's `interlick_near_px` 80 -> **40** (percentile-matched: cam4's 80 px = its 42nd-55th percentile of tongue distance; the same percentiles on cam1 = 44-45 px DLC / 40 px LP; 40 also stays below the close side spouts' ~48 px from the cam1 mouth; the centre spouts are ~5 px from it, end-on, so there only the speed test separates a between-lick pause from a slide round the rod) and `interlick_speed_px` 3 -> 1.7 -- cam1 sees the tongue's reach foreshortened
(distance from the mouth median 36-50 vs 73-98 px, p95 71-106 vs 156-186), so 80 px covered 81-98 % of cam1 tongue frames and
the veto would also fire on slides round the spout rod. Same foreshortening for ANALYSIS thresholds measured from the mouth:
cam4's 20 px in-mouth cutoff corresponds to ~10-11 px on cam1 -- set cam1 thresholds from cam1 distributions, not cam4's px.
**Superseded for gap filling by D8** (max gap 24 ms, veto off).

## D8. cam1 gap length: cam4's 56 ms bridges licks on cam1 (10-18, Priya's QC of the cleaned clip, trial 251)

`python -m scripts.cam1_clean_qc gaps PS93:20260814 PS93:20260821 PS93:20260908 --model <csv> --tag _<model>` lists every
tongue gap the cam1 cleaning would interpolate (cam4's 56 ms, veto off; post-cue, within a trial) and asks cam4 (side view,
DLC r3, same trial and cue-relative time) whether the tongue was out: a gap where cam4's tongue is not confident for >= half
of it was the tongue back in the mouth -> interpolating it BRIDGES two licks. Outputs `session_poses/cam1_gap_check_<model>.csv`.

* By gap length (LP c1r2; DLC c1r2 the same shape): <= 12 ms -> cam4 says in-mouth 24-38 % (mostly real dropouts, tongue out);
  24-40 ms -> 93-97 %; 40-56 ms -> 98-100 %. **Side and centre positions alike** -- not a centre-only problem. On cam4 the
  same-length gaps are the tip hidden behind the rod (hence 56 ms there); from below the rod hides little.
* The between-lick veto caught only ~10 % of the bridges (it needs a retract-then-protrude in distance from the mouth, which
  cam1's foreshortened reach rarely shows) and at max gaps <= 32 ms it blanks more real tongue than it saves.
* Bridged + lost frames per session (0814 / 0821 / 0908): LP c1r2 56 ms + veto 1713 / 675 / 2431 -> **24 ms, no veto 83 /
  260 / 227**; DLC c1r2 1643 / 1088 / 2656 -> 291 / 548 / 419. 16-24 ms tie; 24 keeps more measured-tongue continuity.
* Acute has the most cam1 gaps > 24 ms that cam4 sees as tongue OUT (16-23 % vs 1-3 % pre / chronic) -- the residual cost of
  24 ms falls on acute (LP 79 lost frames vs 0-7); still far below the bridging it removes (607 -> 181).
* **Proposed and written to `orofacial_clean_cam1.tongue`: `max_gap_ms: 24`, `interp_veto_interlick: false`** (cam4 digest
  unchanged). Re-rendered clip for QC: `session_poses/PS93_20260814_cam1/eval/cleaned_labels_cam1_dlc2_lp2_gap24.mp4`.
* **Why it matters for lick counts (10-19, Priya: "can't the xy shape separate two bridged licks?").** Not reliably: the
  return to the mouth is what the gap is missing, and PCHIP between two near-spout points never dips back. Counted on PS93
  (LP c1r2, cam4 production lick peaks as truth): continuous cleaned cam1 tongue runs that contain >= 2 cam4 licks hide
  **25.0 / 6.2 / 23.1 %** of cam4 licks (pre / acute / chronic) under 56 ms + veto vs **1.4 / 1.5 / 0.6 %** under 24 ms. The old
  rule's merging was also EPOCH-DEPENDENT (acute licking has longer pauses), so it would have manufactured an epoch effect in
  any cam1 lick count, duration or rhythm measure. (A shape-based detector could split a few of the merged runs; these are
  continuous runs, not detector output.)

**Water (Priya 10-18):** the jaw and sometimes the tongue get dragged onto a hanging drop when the drop hides where the point
would be (reflections pull the keypoint; e.g. PS93 0814 trial 22 far_center +84 / +180 ms, jaw on the drop's lower edge). A
"jaw above its trial rest" rule is NOT a clean water signature: confident post-cue jaw > 20 px above the trial's pre-cue rest is
2-7 % of frames, but only 2-24 % of those are in the last quarter of trials (where drops build up) -- something else dominates.
No water guard proposed yet; next labelling round: water-drop frames with the hidden point left blank (hard negatives).

## D9. First cam1 behavioural readout -- lateral landing of licks (10-18, PRELIMINARY: one animal, model not chosen)

`python -m scripts.cam1_lick_aim PS93:20260814=pre PS93:20260821=acute PS93:20260908=chronic --model <LP c1r2 csv> --tag _lp_c1r2`
-> `session_poses/cam1_lick_aim_lp_c1r2{.csv,_summary.csv,.png}`. cam1 GEOMETRY (checked on frames): image up / down =
dorsal / ventral; the spouts sit almost level with the mouth and differ only laterally; the tongue reaches DOWN under the spout
tip, 50-100 px below the mouth. An angle of tongue vs spout direction is therefore meaningless on cam1 (first version: -50 ..
-220 deg); only LATERAL comparisons are valid. Lick = cleaned tongue run (D8 values) reaching > 20 px below the mouth; peak =
its most ventral frame; `lat_err_px` = peak lateral - spout lateral (+ = outward of the spout; centre: + = mouse's left).
Runs that never get below the mouth (a pale bump at the lips, LP confident, mouth nearly closed) are counted, not licks.

Median lateral error, contact licks (px; pre / acute / chronic): close_L -6 / -29 / -26; close_center (+ = left) +18 / +1 / -7;
close_R -41 / -14 / -16; far_L -39 / -49 / -60; far_R -32 / - / -50; far_center +1 / 0 / -3.
Lesions are LEFT (ipsi = mouse left = cam1 IMAGE RIGHT). On these three clips PS93's licks look shifted to the mouse's RIGHT
after the stroke, and PS93's cam4 angles on the same trials agree -- **but that is a BASELINE ARTEFACT, corrected 10-18 (Priya:
"I thought PS93 also shifted LEFT")**. Over all curated sessions (median of session medians, cam4 contact licks, + = ipsi)
PS93 shifts LEFT acutely like the cohort: close_center +10.0, far_center +9.5, far_R +5.9 deg (acute - all pre). 0814 is an odd
pre day: PS93's close_center angle is BIMODAL across sessions -- about -10 deg on most days, +5 .. +10 on 0814, 0818, 0820, 0915,
0918, 1002 -- while close_L / close_R do not jump on those days (so not a reference-frame flip): PS93 seems to lick the OTHER
SIDE of the centre spout on some days, and cam1 shows the same (close_center lateral +18 px on 0814, ~+1 on 0821). A cam1
epoch comparison needs several sessions per epoch, not one clip day each. (Nose lateral in the cam1 mouth frame moved +6.5 px
between these days: not the spout set point.) NB cam4 is a FRONTAL view (face from the front, tongue down toward the spouts),
not a side view as written in places above.
**But the PEAK definition is not trustworthy on cam1 (Priya 10-18: "peak protrusion may not be the furthest distance on cam1
since it's at a funny angle"):** the clouds lie along diagonals and pre-stroke licks reach deeper, so the most-ventral frame's
lateral position covaries with reach; at close_R the pre-stroke "peaks" sit on the opposite side of the mouth from the spout
(the tongue drooping / curling, not aiming). At the DAQ CONTACT frame the shifts are smaller (close_L -18 / -16, close_center
+1 / -10, close_R -2 / -8 px).

**cam4-TIMED peaks (10-18, `--cam4-licks <production licks.csv>`, tag `_lp_c1r2_cam4timed`).** Each cam4 lick's protrusion peak
(production LP r4 v36, `t_ms`) -> the cam1 frame at the same cue-relative time. Both clips are cut on the template-aligned DAQ
clock on one 4 ms grid: 99.99 % of cam1 frames have a cam4 frame at EXACTLY the same t_ms (worst case one frame; the template is
finer still). (The "~12 ms" in B4 is cross-camera agreement of two models' onset estimates -- tracking, not clock.) cam1 sees the
tongue at cam4's peak for 79-100 % of licks. Per lick, cam1 lateral vs cam4 peak angle: Spearman 0.77 pooled; within position
0.74-0.88 (far_L, close_L, both centres), close_R 0.45, far_R -0.02. **The right-hand disagreement is near-mouth events**
(frames: tongue at the lips late in the trial, cam4 angle -65 / -73 deg -- the angle of a point at the mouth is unstable):
restricted to cam1 tongue > 30 px below the mouth, rho = far_R 0.80, close_R 0.83, close_L 0.75, close_center 0.81, far_L 0.92,
far_center 0.66. So the two cameras measure the same lateral aim once the tongue is really out -- use a protrusion floor for any
angle / lateral measure. Contact licks, lateral error px on the three clip days (pre / acute / chronic): close_L -17 / -34 / -35;
close_center (+ = left) +20 / +1 / -8; close_R -40 / -21 / -16 -- the apparent rightward shift is the 0814 baseline (above).

### cam4 LP r4 eps35 (O2, scored 2026-10-18)

LP r4 recipe (heatmap, ResNet-50, 640 px, batch 4, round 4) with temporal epsilon [nose 10, jaw 10, tongue 35, spout 10] -- the
only difference from LP r4 (which used 10 px for every keypoint). `o2/lp_train_20261016/models/heatmap_640_b4_eps35`; scored
`session_poses/PS93_<d>/r4_eval/model_lick_compare_eps35_1018.csv`.

| pre / acute | LP r4 | **LP r4 eps35** | ctx-semi 640 | ResNet-50 640 b8 |
|---|---|---|---|---|
| contact recall (%) | 98.8 / 99.6 | **99.5 / 99.2** | 98.5 / 97.9 | 99.0 / 99.2 |
| review precision (%) | 82 / 88 | **96 / 94** | 87 / 91 | 96 / 96 |
| review share of real licks (%) | 100 / 97 | 89 / 97 | 100 / 97 | 85 / 87 |
| no-contact licks | 149 / 88 | 127 / 94 | 138 / 86 | 119 / 84 |
| jaw x jumps > 30 px | 38 / 5 | 52 / 5 | 7 / 3 | 114 / 23 |
| tongue x jumps > 60 px | 3 / 1 | 1 / 1 | 1 / 0 | 4 / 2 |

**Reading.** The 35 px tongue epsilon makes the production recipe more PRECISE (pre: 1 false lick in the review vs 6) at
nearly the same coverage (misses 3 of 27 real faint licks pre), with tighter all-lick angle spreads (e.g. far_center pre IQR
3.3 vs 15.6 deg, far_R 6.8 vs 17.7) and slightly more jaw jumps pre. A candidate to replace LP r4 -- Priya's call (switching =
re-running O2 inference, cleaning and every downstream analysis); weigh against ctx-semi 640 (steadiest jaw, finds every real
lick, lower precision). ViT-B 640 still training.

**Blind review on three more animals (Priya 10-19; PS92 / PS94 / PS95, 21 Aug acute; both models run Windows-native on the
cam4 clips; `lp_stage/lick_review_eps35/`, page https://claude.ai/artifact/FyZaEb6Laub48h87YAgUqP, collection
`lick_review_eps35`; 120 no-contact candidates + 12 hidden contact controls, all 12 judged real).** The models agree on 104
candidates (102 real, 2 unsure); LP r4 alone found 14 (7 real, 5 not a lick -- all 5 in PS94 -- 2 unsure); eps35 alone 2 (1
real, 1 unsure). Precision LP r4 95.6 % (PS92 / PS94 / PS95 100 / 87.5 / 100) vs eps35 **100 %**; share of real no-contact
licks found LP r4 **99.1 %** vs eps35 93.6 % (92.1 / 88.6 / 100). **eps35 trades misses for false positives**: on these
sessions it drops 5 false licks and loses 7 real faint ones (of 110). Across both reviews (PS93 pre / acute + these three) eps35
is consistently more precise and consistently misses some real faint licks -- not a clear enough win to re-run production;
the faint / incomplete licks it misses are the post-stroke ones that matter.
**DECISION (Priya 10-19): production stays on LP r4; eps35 not adopted** -- reasons and revisit conditions in DECISIONS.md 2026-10-19.

### cam4 ViT-B DINOv2 640 b8 (O2 rerun, scored 2026-10-19 -- NOT a candidate)

`o2/lp_train_20261016/models/vitb_dinov2_640_b8`, tag `_vitb_1019` (PS93 0814 pre / 0821 acute, Priya's 10-14 review). Contact
recall 96.8 / 97.9 % (LP r4 98.8 / 99.6); review precision 100 / 90.6 % but **finds only 48 % of the real faint licks pre**
(LP r4 100 %) -- 71 no-contact licks pre vs LP r4's 149, then 133 vs 88 acute (inconsistent across sessions); tongue x jumps
> 30 px 267 / 137 (LP r4 35 / 45), > 60 px 12 / 8 (3 / 1); jaw x jumps > 30 px 137 pre (38). Validation loss was still falling
at 300 epochs (0.19 vs ResNet-50 eps35's 0.11). Worse than LP r4 on every axis that matters; no reason to train it longer
for production.

## D10. Per-frame 3-D tongue / jaw and per-lick 3-D aim (10-19, PRELIMINARY: PS93 trial clips)

`wfield_local/pose3d.py` + `python -m scripts.pose3d {clips | production}` (dlc env; output `session_poses/pose3d/`:
`<a>_<d>_frames.csv.gz`, `<a>_<d>_licks.csv`, `summary_<source>.csv`). aniposelib `CameraGroup.triangulate` (undistort +
DLT) with the 09-11 calibration -- the spout fit's own call, not anipose's filter / optimisation pipeline (two cameras, no
rigid limbs; multi-camera EKS is the next step if tips prove noisy). Frames pair EXACTLY on (trial_id, t_ms) (same 4 ms
DAQ grid); each camera is cleaned with its own rules first and tongue / jaw are triangulated only where BOTH cameras
MEASURED the point (no interpolated point enters); nose / spout raw at likelihood >= 0.9. The per-session spout fit
(`spout_world`) gives (ml, ap, dv) mm, origin = the stage's mouth set point. June cam4: x 0.885 about the principal point
before cleaning. `production` reads both O2 outputs (`o2_inference.session_windows`, split days handled).

**PS93 clips (cam1 ctx-semi eval clip + cam4 LP r4 O2; 0814 pre / 0821 acute / 0908 chronic).** 97 % of cam1 frames pair;
spout fit 0.05-0.07 mm, scale 0.986-0.992; reprojection 6-9 px for EVERY part, the tongue included (= the spout's, so the
cam1 "tip from below" and cam4 "tip from the front" triangulate as one point); 841 / 448 / 1091 3-D licks.

**The aim must be measured against the target spout, not from the origin.** The set point is not where the tongue
leaves the mouth (resting jaw at (ml, ap, dv) = (-0.2, 1.2, -1.4) mm, i.e. at the close_center spout), so angles from it were
11-48 deg off target with no sensible structure. At its extension peak the tongue is 1.5-3 mm BELOW the spout tip (the hanging
drop; Priya's "the tongue reaches down" on cam1) and medial to it. Per lick now: ``miss_mm`` = closest 3-D approach of the tip
to the commanded spout tip; ``lat_err_mm`` at that frame (+ = outward, centre + = mouse left, the cam1 2-D sign);
``reach_mm`` at the peak.

**Validation: contact licks come closer** (median miss, pre): far_L 1.53 vs 3.07 mm without contact, far_R 0.93 vs 1.63,
close_L 1.32 vs 1.71, close_R 0.88 vs 1.17 -- the measure tracks real accuracy. Contact licks still keep the tip ~1 mm
MEDIAL of the spout on BOTH sides (lat_err -1.0 close_L, -0.8 far_R): a constant calibration / label offset would have
opposite signs on L and R, so this is physical -- the side of the tongue, not the tip, touches the spout.
**PS93 acute:** NO far_R contact licks on 0821; far_R no-contact miss 1.63 -> 2.53 mm, lat_err -1.11 -> -1.98 (medial =
toward mouse left = ipsi), partial recovery at 0908 (2.14 / -1.44). One animal, trial clips: wait for production.

**First PRODUCTION run (10-19, PS92: 7 June + 4 August sessions, the first 12 cam1 O2 videos; 0812 is a split day still
running).** `collect`: every frame count matches the video. Whole sessions pair 99.99 % (both cameras cut by
`build_windows`) and give 3,300-7,700 3-D licks per session (vs 450-1,100 on the trial clips). August pre (0805-0807):
spout fit 0.05-0.07 mm, reprojection 5-11 px for every part, tongue included. Pre-stroke aim matches PS93's pattern: the
tip's closest approach stays medial of the lateral spouts on both sides (lat_err -0.5 close, -0.8 to -0.9 far), miss
0.5-1.4 mm. **PS92 acute day 1 (0818, stroke after the 0817 session):** far_R miss 1.2 -> 3.23 mm, lat_err -0.9 -> -3.03
mm (medial = toward mouse left = ipsi) -- the same contra-far deficit as PS93 0821; every other position within the pre range.

**June cam4 is NOT one configuration (`apparatus_mouth production`, 10-19).** Curated June (0606-0608): scaling about the
principal point is best, as in the 0606 test (reprojection 7-10 px, cam1 mouth - nose x +4 / +2 / 0 like August), and
`pose3d` uses it. But 0602-0604 fit better scaled about the image ORIGIN (v36): reprojection 4-5 vs 15-17 px; there the
principal-point variant puts the cam1 mouth - nose x at +18 / +22. So cam4 was moved or re-zoomed between 0604 and 0606.
0601-0605 are outside the curated set (`cross_session_exclude`), so no analysis is affected; 3-D for them would need a
per-session cam4 correction (D3 option 2, spout-fitted similarity). **PS92 0605 fails** (fit residual 1.5 mm, distance
ratios to 8; cam4 cuts 15 % fewer frames) -- its cam4 template is the known quality failure. June reach is ~5-10 % longer
than August at far positions (scale after correction 1.015-1.023): residual June geometry or early-training behaviour;
compare June only within June.

**Cameras posed per session from the commanded spouts (PnP) -- the 3-D default since 10-19.** Priya's video QC (10-19, PS92
0807 far_center): the reprojected 3-D point sat ~10 px from the 2-D tongue, to the RIGHT in cam1 and the LEFT in cam4.
Measured over whole sessions it is the same for EVERY part (cam1 / cam4 red - green x: tongue +9.9 / -9.4, spout +8.9 /
-8.5, jaw +8.1 / -7.7, nose +9.5 / -9.4 px), horizontal only, and varies by session (5.5-10.5 px; PS92 June, August and
acute, PS93 clips): a sideways error in the 09-11 extrinsics plus day-to-day camera drift. **Not the spout edge labels**
(cam1 inferior, cam4 superior edge, Priya asked): they differ in dv / ap, which lies along the epipolar direction (cameras
below and in front), so triangulation absorbs them into depth with no residual -- and nose / jaw carry no edge label.
Fix (Priya: "use the ground-truth spout positions"): `pose3d.pnp_group` poses EACH camera per session by PnP (cv2 SQPnP +
LM) on the six parked spout medians against the commanded positions (mm); intrinsics / distortion stay 09-11. Points come
out directly in stage mm. Result (`pose3d production`, 11 PS92 sessions; PS93 clips):

| | 09-11 extrinsics | per-session PnP |
|---|---|---|
| spout reprojection px (cam1 / cam4) | 6-10 / 6-10 | 0.5-1.1 / 0.6-1.2 |
| nose, jaw px | 8-10 | 0.5-3.0, 0.8-1.8 |
| tongue px | 8-11 | 3.3-4.3 (PS93 2.8-4.0); 0606: 7 |
| June 0602-0604 tongue px (cam4 moved, D10 above) | 13-18 | 3.8-4.2 -- PnP absorbs the moved camera |
| PS92 0605 | broken | still broken (cam4 spouts 19 px: its cam4 template is the known failure) |

Lick measures barely move (miss / lat_err within ~0.05 mm; PS92 0818 far_R miss 3.24 mm, lat_err -2.99), so the earlier
readouts stand. The edge labels go into each camera's pose; their difference (~ rod width, mostly depth) is a small constant
shift of every body point. The old route (09-11 extrinsics + per-session `disparity_shift` + similarity fit) stays as
``method="calib"``. Single-camera frames (~40 % of tongue frames; cam1 alone fixes ml / ap, cam4 alone ml / dv) are the next
step: ray + depth from neighbouring 3-D frames, or multi-camera EKS. Video: `pose3d_figures video --view cam4` (3-D panel
seen from cam4's PnP position; also cam1 / hybrid).

**Single-camera frames filled (`pose3d.fill_single_view`, default since 10-19; Priya: "use the known geometry to resolve
occlusions in one view").** ~40 % of tongue frames are seen by ONE camera (PS92 0807: 18 % cam1-only, 25 % cam4-only; jaw
29-34 %). That camera fixes the point to its ray (cam4 alone: ml + dv; cam1 alone: ml + ap -- ml, the aim coordinate, is
known either way); the depth along the ray comes from the nearest both-camera frames of the same trial within 24 ms
(interpolated in time), moved to the closest point on the ray; ``{part}_src`` 1 both / 2 cam1 / 3 cam4 / 0 none.
**Held-out check (PS92 0807, one camera hidden on both-camera frames):** median 3-D error 0.14-0.15 mm at 4 ms gaps, 0.22-0.25
mm at 24 ms (p90 0.4 -> 0.65); ml 0.08-0.09 mm at every gap (= the cameras' tongue disagreement, 3-4 px); the error grows
along the axis the seen camera cannot measure. Coverage of tongue frames seen by either camera: 61 % -> 88 %.
**It changes the aim readout -- CORRECTION to the "tip ~1 mm medial at contact" reading above.** The closest approach to
the spout is often a ONE-camera frame (the spout itself hides the tip from one view at contact), so both-camera-only misses
were biased high: PS92 0807 far_L 1.34 -> 0.90 mm, far_center 0.57 -> 0.36, close ~1.1 -> 0.7-0.8; lat_err far_L -0.87 ->
-0.62, close -0.5 -> -0.3 / -0.4; reach at the far positions +0.3-0.5 mm (peaks were one-camera frames too). Licks per session
change little (runs that were split by a one-camera gap now merge). The tip still passes medial of the lateral spouts, by
less (0.3-0.9 mm). **The acute deficit stands:** PS92 0818 far_R miss 3.19 mm / lat_err -2.93 (pre 1.1-1.2 / -0.7 to -0.9);
PS93 far_R 0.97 / 2.48 / 1.40 mm pre / acute / chronic. Unfilled summaries kept in `session_poses/pose3d/nofill_20261019/`.
Videos mark filled points orange (red = both cameras).

## D11. cam4 readouts redone in 3-D (10-19, PRELIMINARY: PS92 production pre x 9 + acute 0818; PS93 clips)

**Cross-check, lick by lick (`scripts/pose3d_crosscheck.py`; each cam4 v36 lick matched to the 3-D frame at its peak, 93-97 %
matched on production sessions).** cam4 LATERAL is faithful: `lr_px` vs -ml r 0.95-0.97, slope 34-37 px / mm (= cam4's own
scale); peak angle vs -ml r 0.77-0.90; tip-to-spout r 0.93-0.97. cam4's 'out' axis (`ap_px`, `protrusion_px`) is the DOWN
axis: r 0.99 with -dv and -0.4..+0.1 with the true out-of-the-mouth ap. So the cam4 lateral / angle / jaw-steering findings
stand; anything about how far the tongue comes OUT is invisible to cam4.

**3-D readouts on the cam4 lick events (`scripts/pose3d_kinematics.py`).** Same licks (cam4 on .. off windows); the peak is
redefined in 3-D as the frame farthest from the mouth exit (differs from the cam4 peak frame on 54 % of licks, |dt| p90 16
ms). MOUTH EXIT anchored to the head (Priya 10-19: not the jaw, which changes post-stroke): the session's resting nose + a
fixed per-animal offset = median (resting jaw - resting nose) over its August pre sessions (PS92: (-0.05, -5.19, -4.32) mm).
Angles: the exit sits AT the close spouts (~1 mm), so angles are defined for the FAR spouts only (close: miss / lat_err);
a horizontal azimuth is ill-conditioned (licks go 50-75 deg down) -- use the frontal side tilt atan2(ml, down) vs the target's
(error + = ipsi), the sagittal forward tilt, and the 3-D angle. Jaw: as `orofacial_jaw_lateral` but in mm, cue + 70 ms .. the
trial's stop (Priya 10-19), opening = 3-D distance from the trial's pre-cue rest, wide > 1.5 mm.

PS92, acute 0818 vs its August pre sessions (range of 3; June range in brackets where it matters):

| measure | Aug pre | acute |
|---|---|---|
| far_R miss (mm) | 1.09-1.21 | **3.20** |
| far_R lateral error (mm; - = medial) | -0.84..-0.89 | **-2.96** |
| frontal error toward ipsi (deg): far_R / far_L / far_center | +35 / -35 / +4 | **+94 / -21 / +10** (all shift ipsi; far_R most) |
| reach far_L / far_R (mm) | 3.7-4.1 / 3.8-4.0 | **3.2 / 3.0** |
| of which down (dv) far_L / far_R | 3.4-3.8 / 3.5-3.6 | **2.5 / 2.3** |
| of which out (ap) far_R | 0.48-0.60 [June -0.01..2.76] | 0.95 |
| peak 3-D speed far_L / far_R / far_center (mm/s) | 230-255 | **206 / 157 / 192** |
| widest jaw opening far_R / far_center (mm) | 3.6-4.2 / 3.0 | **2.2 / 1.7** |
| jaw FAR_DIR (mm; + = toward target) | 0.26-0.27 [June 0.04-0.27] | **0.02** |

Readings (one animal acute; PS93 only on the clip subset, ~10 % of its licks pair): the acute ipsi shift (cam4 finding 2),
largest at far contra, holds in mm; the shorter acute reach is LESS DOWNWARD extension with no loss in forward extension;
peak tongue speed drops at every far spout; the jaw opens less and stops steering (cam4 finding 3: -11.5 px ~ -0.33 mm at
34 px / mm; here -0.25 mm). Pre-stroke, licks at the lateral spouts are ~35 deg steeper (more downward) than the line to the
spout. The cohort run (curated sessions, per-animal tables, `analysis_kit.boot_delta`) waits for the cam1 production output.

**REVISED 10-19 (later): the mouth is the spout "mouth zero" -- the table above is SUPERSEDED.** The mouth set point is placed at
the rig by bringing the spout as close as possible to the front of the mouth, so it IS the mouth and the targets sit 2 / 4 mm
from it (Priya). The table above used the resting JAW label (the chin, ~1.4 mm in front of and 1.3 mm below it) as the exit,
which made the close spouts look like they sat on the mouth; a nose-anchored variant was rejected too (the nose and jaw can
both move post-stroke; Priya: "don't use resting nose"). `pose3d_kinematics` now uses the origin (= the set point) only;
angles are defined at every position (targets 2 / 4 mm away). Post-stroke the set point was placed ~0.5 mm closer
(slack jaw, EXPERIMENT_ERRORS) -- every set-point-relative number describes the task as placed that day.
Change vs August pre (PS92 production / PS93 clip subset), far_R: miss +2.1 / +1.5 mm; lateral error -2.1 / -1.1 mm (medial);
frontal tilt error toward ipsi +34 / +15 deg; 3-D angle error +29 / +22 deg; reach -0.2 / -1.3 mm = more forward (+0.95 /
+0.73) and less downward (-0.76 / -1.84); peak speed -83 / -57 mm/s. Jaw FAR_DIR -0.25 / -0.22 mm acute (PS93 chronic -0.17);
widest opening -1.0 to -2.3 mm at the far spouts. Same direction in both animals: acute licks reach less DOWN and more
FORWARD, aimed ipsi, slower; the jaw opens less and stops steering.

## D12. Error distance from the spout (10-19; Priya: "a separate analysis") -- `scripts/pose3d_spout_error.py`

Referenced to the spout itself, so the post-stroke set-point change does not enter. Three errors per cam4 lick event (Priya
10-19: closest approach or peak? -- "for close spouts, normal peak lick is well below the spout"), kept side by side:
(1) CLOSEST APPROACH to the spout tip (water access); (2) PEAK ENDPOINT (3-D extension peak) vs the spout tip; (3) peak
endpoint vs the animal's OWN August-pre median endpoint for that position (error in the motor-control sense, no assumption
about where on / under the spout the target is). Inputs: each camera's cleaned tongue (measured points only) + the
single-camera fill. Splits: DAQ contact (cam4 ``contact``), first lick of the trial vs later. Figures: spout-centred scatter
of closest-approach points (front and side), distance-to-spout traces from lick onset.
Pre (August) medians: closest approach 0.4-1.3 mm; peak 1.5-3.2 mm from the spout and BELOW it (dv -1.3 to -3.0) at every
position -- Priya's point; peak vs own endpoint 0.7-1.4 mm (lick-to-lick spread). Acute far_R, all three grow in both
animals (PS92 / PS93): +2.1 / +1.5, +1.7 / +0.5, +2.2 / +1.3 mm, laterally medial (-2.6 / -1.3 mm at the endpoint); acute
endpoints sit higher than habitual (dv +0.5 to +1.8 mm; PS93 most positions, PS92 far L / R); close spouts: closest approach
~unchanged (still reached), endpoints shift by tenths of a mm. One animal per epoch so far (PS93 on the clip subset).

**Spout edge labels built into the camera posing (10-19, Priya).** The mouth zero was set with the UPPER spout tip at the
mouth, so the commanded point is the rod's superior edge (cam4's keypoint) and cam1's keypoint (inferior edge) sits one rod
diameter below. Rod diameter measured on parked frames (horizontal profile behind the rounded end, PnP px -> mm): cam1
1.23-1.24 mm (IQR +-0.03), cam4 1.27-1.31 (low contrast), PS92 0806 / 0807 / 0818 -> `pose3d.ROD_DIAMETER_MM` 1.24;
`pose3d.EDGE_OFFSET_MM` cam4 (0, 0, 0), cam1 (0, 0, -1.24). Effect (PS92): spout fit unchanged (0.03-0.07 mm); body points move
~0.5-0.9 mm back (toward the mouth) and 0.1-0.2 mm down relative to the spouts; errors vs the animal's own endpoint change
<= 0.05 mm (changes / comparisons unaffected, as predicted); reprojection cannot arbitrate (the offset lies along depth: nose
slightly better, jaw slightly worse) -- the correction rests on the rig setup + the measured rod. New pre absolute values
(0807): closest approach to the upper tip 0.65-1.1 mm, the tip then ~0.25-1.0 mm short of it on the mouth side; peak 1.4-2.6
mm below it. Acute far_R unchanged (miss 3.15 vs 0.97; lateral -2.98 vs -0.79). The fit check (`_identity_world`) shifts
cam1 by its offset before triangulating the spouts (without it every session read > 0.2 mm and was skipped downstream).
Pre-correction tables kept in `session_poses/pose3d/noedge_20261019/`.

**Single-camera fill: Akima, not linear (10-19, Priya asked about spline interpolation).** Held-out 3-D error (PS92 0807 /
0818, one camera hidden on both-camera frames), linear -> PCHIP -> Akima: 8 ms gaps 0.136-0.144 -> 0.126-0.138 ->
0.127-0.137 mm; 16 ms 0.164-0.176 -> 0.151-0.165 -> 0.145-0.162; 24 ms 0.220-0.272 -> 0.213-0.245 -> 0.203-0.231 (largest
gain on long cam1-only gaps). Akima's knot slopes follow the straighter side of a bend and are not zeroed at a turn (PCHIP
flattens the extension peak inside a gap). Used only for two-sided gaps (both-camera frames within 24 ms on both sides,
same trial, >= 3 knots); one-sided gaps hold the nearest point; the estimate is then moved onto the seen camera's ray.
`session_3d(fill_interp="akima")`. Results after the rerun: PS92 acute far_R miss +2.3 / lateral -2.25 mm; PS93 acute far_R
+1.0 / -1.0 (~20 licks on the clip subset, so it moves with method details; PS92 stable).

**3-D QC, angles and figures (10-19 evening).** (1) Plausibility gates (`pose3d.qc_3d`): tongue > 8 mm from the mouth zero,
> 2 mm BEHIND it (ap < -2: Priya, "impossible (negative from mouth)"; ap < 0 is common to ~-1 mm = a tip emerging under the
upper lip), jaw > 5 mm from its session median, single-frame out-and-back spikes > 1.5 mm; filled points dropped when the
seen camera's ray misses the neighbours' estimate by > 1 mm. Source of the PS92 0607 outliers: PAW-in-face frames -- cam1
puts 'tongue' on the digits (likelihood 0.91-0.92), cam4 on the incisors in the open mouth (0.97-0.99); reprojection does not
flag them (error along depth). Hard negatives for a future labelling round: paw-in-face frames, tongue blank, both cameras.
(2) Direction: 3-D angle error vs the spout direction (consistent across the set-point move: the spouts are commanded from
it, Priya) + the angle OUT OF THE MIDLINE PLANE through the mouth zero and the centre spouts (= ml 0; asin(ml / |r|), + =
ipsi) + forward tilt within it. (3) Figures: `scripts/pose3d_summary_fig.py` (per session over days from the stroke, August
pre band, June as open markers in its own segment), `scripts/pose3d_phase.py` (peak-aligned 3-D distance, phase portrait
radial speed vs distance, midline angle and distance vs normalised phase, mean paths in front / side / top planes on
normalised phase where >= half the licks have a 3-D point). PS92 (9 Aug pre, 3 acute): acute licks are shifted ipsi from
ONSET through the whole lick at every position (+5..+13 deg at close / centre, ~+30 deg at far_R, where the mean path crosses
to the mouse's left of the midline); far_R protrusion peak radial speed ~100 vs ~160 mm/s; centre licks reach further acutely.

**Combined-track licks (`pose3d.lick_events`, `pose3d_kinematics --licks combined`; Priya 10-19).** Lick events from the 3-D
track (both cameras or one camera + neighbours, after QC), cue .. trial stop, gaps <= 12 ms bridged, >= 3 frames; flagged
``cam4_matched`` (overlaps a cam4 lick window) and DAQ ``contact``. Adds 3-7 % licks per session (pre 3.1-6.6 %, acute
2.4-3.3 %). The added licks (PS92 August): reach 3.4 vs 4.3 mm, 8 vs 18 3-D frames, peak speed 119 vs 184 mm/s, and 47 %
touched the spout (DAQ) vs 86 % -- real small licks cam4's detection missed. The earlier 31-41 % of cam1 at-the-lips events
without a cam4 lick (D9) are mostly cam1-ONLY (no both-camera frame within 24 ms) and still have no 3-D point: they need the
depth prior (held-out test next). Output `session_poses/pose3d/kinematics_combined/`.

**Jaw one-camera fill limit = 200 ms (10-19 late; `pose3d.FILL_GAP_MS`, `scripts/pose3d_jaw_gap_check.py`).** Held-out (PS92
0807 / 0818, one camera hidden on both-camera jaw frames): median 0.03-0.05 mm at 24 ms gaps, 0.04-0.07 mm at 200 ms (p90 0.10
-> 0.20-0.34). One-camera jaw frames are mostly long stretches: 1-3 % lie in gaps <= 24 ms, 13-23 % in gaps <= 200 ms (the rest
-- e.g. the rod over the chin in cam4 for whole centre-spout trials -- stay single-camera). Tongue stays at 24 ms.
