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
`orofacial_clean.cameras.<cam>` in `configs/defaults.yaml`. **No cam1 overlay is set yet** -- the values must come from cam1
data with Priya's QC, as cam4's v35 did.

**Baseline diagnostic (cam4 values on DLC c1r2, PS93 0814, 60 trials, 10-18).** cam1 looks UP at the underside of the
snout: nose at the top, mouth / jaw low in the image (jaw median y 456), the spout rod running down the middle of the face
to the mouth (hence the nose blanks and the centre-position jaw occlusion). The tongue protrudes mostly UP the image
(toward the nose) and sideways.

* **Mouth origin:** the cam1 spout-line intersection (as on cam4) lands at (315, 389) -- plausible, to confirm on frames.
* **Tongue:** confident on 9 % of clip frames; cleaning keeps 9 % raw, interpolates 2 %, fills 89 % with the baseline
  (tongue in the mouth). Steps > 40 px rise from 160 raw to 806 after cleaning -- mostly the jump between the baseline fill and
  a protruded tongue at every lick onset / offset, so on cam1 the metric must exclude fill transitions.
* **Jaw:** confident 95 %; the data-driven baseline ("lowest y") lands at y 372 while the jaw's median is y 456 -- the cam4
  assumption about which image direction is "closed" does not hold from below. v34 halves jaw steps > 20 px (629 -> 284).

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

**Consequence for 3-D.** v36 multiplies June cam4 x, y about the image origin: right for 2-D px measures (protrusion, speed,
relative to the mouth), WRONG as triangulation input (absolute positions move; the 09-11 calibration describes Aug+ cam4).
With cam1 unchanged, June 3-D needs a June cam4 calibration: refit cam4 intrinsics / extrinsics from the six spout positions
(known 3-D points, commanded and read back to 0.001 mm) with cam1 as the fixed reference -- the 10-05 proposal, now with a
cause (and the likely source of that day's "June ~5 % too large" in the 3-D spout fit). Aug+ sessions use the 09-11
calibration as is. Feed triangulation UN-rescaled pose (divide by `px_scale`).
