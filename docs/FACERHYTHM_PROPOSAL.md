# FaceRhythm as video regressors in the movement models — reading notes + proposal (2026-10-07)

> **HISTORICAL (2026-10-07 reading notes).** The FaceRhythm work that followed is documented in
> [`FACE_RHYTHM_PIPELINE.md`](FACE_RHYTHM_PIPELINE.md) (how / run order) and [`FACE_RHYTHM_PLAN.md`](FACE_RHYTHM_PLAN.md)
> (why / decisions); read those first.

Written 2026-10-07 by a separate session outside the repo (to avoid clashing with concurrent commits) and moved
into `docs/` on 2026-10-13 at Priya's request, unchanged apart from this note. Nothing was run except read-only peeks
at existing outputs. NOTE (2026-10-13): this is the "Hakim et al." paper Priya referred to on 10-08 alongside
Musall / Churchland; and since this proposal, the null/potent subspace line was closed as a supplementary control
and the ENL movement-removed trajectory (running + motion-energy PCs) was run on 118 sessions -- see
`docs/status/STATUS_2026-10-13_ENL_TRAJECTORY_TRACKING_R4_HANDOFF.md`. Context: handoff `STATUS_2026-10-07_MOVEMENT_SUBSPACES_HANDOFF.md` §3A.

## 1. The paper (Hakim, Jaggi, Heo, ... Musall, Sabatini; bioRxiv 10.1101/2025.09.10.675423 v4, 2026-07-09)

**Pipeline.** Dense mesh of ~500-3000 points on a face ROI -> Lucas-Kanade optical flow with three corrections
(mesh "springs" to neighbours, relaxation toward the start position, freezing of points whose displacement
jumps, e.g. during grooming) -> per point, x and y, a variable-Q transform (VQT) magnitude spectrogram
(log-spaced bins, window length L = Q * fs / f) -> normalisation (frame-wise power, optional 1/f boost) ->
non-negative CP / TCA on the (2P x F x T) tensor. Each component = spatial factor (which points) x spectral
factor (which frequencies) x **temporal factor (amplitude envelope over time) -> that is the regressor**.
Rank 4-12 is interpretable; 10 is their default. Components are not orthogonal, so co-active behaviours
(whisking + sniffing) separate into different components.

**The result that bears on us — Musall et al. 2019 widefield data (7 mice, 9 sessions, ridge per pixel, cvEVR):**

| regressors | cvEVR |
|---|---|
| nose + whisk + jaw ROI motion energy | 0.143 |
| 10 ME PCs | 0.144 |
| **10 FR components** | **0.245** |
| 200 ME PCs | 0.279 |
| raw point-tracking (all points, high-dim) | 0.33 |
| point-tracking + 10 FR | 0.38 |
| everything (+ ME PCs + ROI ME) | 0.39 |

Read-outs: (i) at matched rank FR >> ME; (ii) FR alone does **not** beat many ME PCs; (iii) ME adds ~0.01 once
point-tracking + FR are in, so **point-tracking + FR subsumes ME**; (iv) in their ablations, the spectrogram
step matters most and nnTCA mainly buys interpretability, not R^2.

**Phase-variant vs phase-invariant (2p M1).** Above ~0.5 Hz, M1 is better fit by the spectral **envelope** of a
movement; below ~0.5 Hz, by the **position/displacement itself** (phase-variant). 93 % of neurons correlate with
some FR envelope and 66 % are phase-locked to some raw movement. Practical consequence: FR envelopes alone discard
the slow and postural part. That part has to come from position-like regressors (point-tracking PCs, DLC, ME).

Other parts that matter less for us: CS identity decoded from FR at 0.83-0.92; BMI reward-time AUC 0.90-0.93 from
FR and from raw points alike; FR temporal factors match M1 NMF factors one-to-one (82 % of pairs significant).
Speed: ~200-300 frames/s on a GPU for a 1280x1024 video, dominated by tracking. Code: `pip install face-rhythm`
(github.com/RichieHakim/face-rhythm). Causal (right-asymmetric) VQT kernels are available (§5.14.2).

## 2. The existing runs on M:\MICROSCOPE\Priya\FaceRhythm\

* **`fr_run_20250520_latent_bundle/`** is Rich's O2 run (wandb, `run_pipeline_basic.py`), 2025-05-21. It covers
  the **old 2pRAM cohort only** (PS46-PS50, Feb-Apr 2025), `cam4` (~140 sessions) and `cam2`, 250 fps,
  600x800, ~1.01 M frames (~67 min) per session. Size: **3.5 TB**, almost all of it `VQT_Analyzer.h5`
  (11 GB/session; the spectrogram is (2, 918, 30, 50631) float32).
  Per session: `TCA.h5` (factors: points 1836x10, frequency 30x10, time 50631x10 at 12.5 Hz), `ROIs.h5` (two ROIs,
  918 points), and params, config and log files.
  Parameters: point_spacing 10, LK winSize 50 / maxLevel 5, CLAHE, VQT 0.5-60 Hz, 30 bins, Q 3->10, centred
  Hann (6 s window at 0.5 Hz), downsample 20 (-> 12.5 Hz), normalization 0.6, 1/f 0.5, rank-10 NN-CP-HALS
  (random init). Runtime: 2.4 h per session on an O2 GPU.
  NB its mesh and relaxation values (0.002 / 0.002) differ from the paper's production values (0.1 / 0.01). It
  is an older package version, so copy the current package template rather than this one.
* Example components (PS48 0301, cam4): two sparse vertical ~5-6 Hz components active ~10 % of the time (licking),
  a horizontal ~7 Hz one (whisking), a 16 Hz one, several 1-2 Hz ones, and two slow ones near the 0.5 Hz floor.
  Temporal factors are mostly decorrelated (median r -0.09, max 0.76).
* **`facerhythm_stroke_biomarker_exp/`** holds **no FR output**. It has the old cohort's wavesurfer files,
  `session_processing3` behaviour products, and four grant example frames (PS51).
* **None of this is our widefield cohort (PS92-95) or rig** (four Blackflys, a moving spout). The existing runs
  are useful as a parameter and runtime reference and for what lick/whisk components look like, but **cannot**
  be used as regressors. `stroke_orofacial_pipeline` has FR deferred to "Phase 11+" for that old cohort.

## 3. Proposal: ADD FR for the face cameras first; replace face-camera ME only if the test below earns it

Current model (PS93 0814): DLC tongue/jaw/licks + DAQ running + 4 cams x 30 ME PCs (blanked outside strobe..trial
end). Full R^2 MOp_L 0.58, video unique 0.07-0.15. ENL position decode after removal: **0.26 (lower bound)**.
Video + running alone decode position at 0.96, and the face-cam ME alone at 0.92 / 0.94.

**Why add, not replace:** in the paper FR's gain is at low rank. It needs a position-like partner for the
<0.5 Hz/postural variance, and that partner is what ME PCs currently supply. The body cams (cam2/3: paws,
posture) are not FR's domain, so they keep ME.

**Arms** (same session, CV, blanking, smooth kernels, split licks as the 10-07 comparison; face = cam1 + cam4):

| arm | regressors |
|---|---|
| A (current) | DLC + running + ME(cam1-4 x 30) |
| B replace | DLC + running + ME(cam2,3 x 30) + FR(cam1, cam4; rank 10 each) |
| C add | A + FR(cam1, cam4) |
| D replace + position | B + face point-tracking PCs (phase-variant partner, from the same FR tracking) |

Read: full cvR^2 and unique per group, per area; rank sweep of FR (6-16) as in their hierarchical regression.

**Criterion to adopt B or D as the default:** cvR^2 within ~0.01 of A (or better) in MOp/MOs/SSp, with far fewer
face regressors. The second, and the more important, criterion is in §4.

## 4. The rig-specific risk: the spout, and why blanking is not enough for FR

* The VQT window is L = Q*fs/f. At 0.5 Hz, Q 3, that is a **6 s** (centred, +/-3 s) kernel. The spout is
  repositioned before the strobe and arrives ~3 s before the cue. A low-frequency envelope sampled inside
  strobe..trial end therefore integrates the repositioning motion *before* the strobe. Blanking the regressor's
  output does not stop that. ME has no such memory (one frame difference per bin). This matters most for the
  ENL window (2 s pre-cue).
* Mitigations, in order: (1) **draw the face ROI to exclude the spout's path and rod** (points then move only
  through occlusion). (2) Raise F_min for the face cams (e.g. 1-2 Hz, a 1.5-3 s window) and let point-tracking
  PCs carry the slow part (arm D). (3) Use causal VQT kernels so nothing after a bin leaks in (it still looks
  back).
* **Required check before any ENL use** (the analogue of `enl_video_spout_check.py`): decode position from FR
  temporal factors in the 2 s pre-cue lick-free window. Face-cam ME scores 0.92-0.94 here. If FR is much lower
  and the model R^2 holds, the **0.26 lower bound becomes less conservative** (less over-removal). That is the
  main scientific payoff, beyond interpretability. If FR also scores ~0.9, look at the spatial factors of the
  components that carry position: FR can show *which* face points and frequencies carry it, which ME PCs cannot.

## 5. Logistics (nothing heavy here)

* **Run on O2, not this desktop** (shared box, 8 GB card, past blue screens). cam1 + cam4 videos for all sessions
  are already on O2 scratch (never touch file dates).
* Our sessions are ~2.2 M frames (~147 min), twice the old runs. Expect **~5 h per camera-session on a GPU** and a
  ~24 GB spectrogram at 12.5 Hz (more at the imaging rate). Do not keep the VQT on MICROSCOPE: keep TCA.h5 +
  ROIs + point-tracking (decimated) and delete the scratch spectrogram.
* Temporal factors -> imaging frames: use the camera alignment templates (camera -> DAQ time) and the same
  bin-edge rule as `video_motion.frame_bins`, with a VQT downsample near the imaging rate. Fill the time index
  explicitly; dropped frames shift a free-running index.
* Cohort use later: per-session TCA gives session-specific components (fine within a session, like the ME SVD).
  For pre/post-stroke comparisons, fit spatial + spectral factors per animal on registered ROIs (roicat, as in
  the paper), then solve only the temporal factors per session. The lick-rhythm components are themselves a
  candidate stroke readout for PS93 (right orofacial deficit).
* Pilot: PS93 0814, cam1 + cam4, two O2 jobs, then the arms in §3 and the check in §4.

## 6. Open questions for Priya

1. Ask Rich for his current widefield-rig parameter template / ROI-alignment notebook (he ran the Musall data)?
2. Face cams = cam1 + cam4 only, or also the face part of cam2/3?
3. Is the "replace" criterion above acceptable, or is FR wanted only as an interpretable add-on?
4. Spout handling: ROI exclusion + F_min raise (my preference) vs. causal kernels.
