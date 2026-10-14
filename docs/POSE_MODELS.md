# Orofacial pose models (cam4 front view) — register, settings, results, and how we choose

Living record of every cam4 pose model trained for the widefield cohort: what it is, the settings that matter and
why, where it lives, how it scored, and the rule for picking the production model. Reasoning in `DECISIONS.md`
(2026-09-28, 09-30, 10-13 x3, 10-14); evaluation scripts named below. Last updated 2026-10-14.

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
5. **Stability:** jaw side-flips and tongue teleports (jumps > 30 / 60 px) per 10k frames.
6. **Pixel error on frames held out by every model** — a sanity check only.
7. **Practicality:** inference cost for the full cohort (desktop / O2) and simplicity (one model beats an ensemble
   at equal performance).

Test set: at least one pre, one acute and one chronic session for two animals, none labelled (the PS93 0814 / 0821
clips plus e.g. PS93 0908, PS94 0818, PS95 0821, PS94 0921); the review clips for the visual check.
