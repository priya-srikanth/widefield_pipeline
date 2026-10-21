# FaceRhythm on the widefield sessions — plan (2026-10-21)

Priya 10-21: "we will probably eventually want to use FaceRhythm as part of this ... matching points across all session
dates and animals using cross-animal / cross-session alignment (morph across animals to fit a single 'master' mouse face
mesh for each camera)". Repo: https://github.com/RichieHakim/face-rhythm (v0.3.3, cloned read-only to
`Github/face-rhythm`); paper: Hakim et al., bioRxiv 10.1101/2025.09.10.675423 v4 (2026-07-09). Env: conda `fr`
(Python 3.12, `pip install face-rhythm==0.3.3`; Windows -> decord video backend).

Interim measures until this lands: whisker-pad ROI motion energy (`wfield_local/roi_motion.py`,
`scripts/whisker_motion.py`) and the nose call (`scripts/nose_onsets.py`).

## What FR does (repo + paper)

| step | what | paper production values | repo defaults |
|---|---|---|---|
| ROIs | one face outline, filled with a point grid (500-3000 points) + a background mask | ~930-1985 points | `point_spacing` 9 px |
| tracking | Lucas-Kanade per frame -> violation freezing -> mesh correction -> relaxation to home | lambda_mesh 0.1, lambda_relax 0.01, threshold 300 px, winSize ~40, maxLevel 3, 24 neighbours | 0.025, 0.0015, 150 px, [20,20], 2, 8 |
| spectral | VQT of each point's x and y displacement, magnitude, per-frame power normalisation (gamma 0.95), 1/f^0.5 | Q ~4 -> ~20, output ~15 Hz | Fs 120 (!), 1-60 Hz, 36 bins, Q 4 -> 10, downsample 20 |
| decomposition | non-negative CP (HALS), (2P points) x frequency x time | rank 10-12; ONE PER SESSION, cross-animal by Hungarian matching of spatial + spectral factors | rank 12, "separate" per session |
| multi-session | mean frame per session registered to a template (RoMa / ECC + DeepFlow non-rigid), template points warped onto each session | ROICaT rigid alignment | `demo_set_rois_multisession.ipynb` |

Not in the paper: a moving lick spout (occlusion), joint fits across sessions, projecting new sessions onto fixed
factors, multi-camera rigs.

## Our numbers

* 250 Hz, cam4 680x680 (frontal), cam1 600x600 (ventral); ~1.8-2.1 M frames (~2 h) per video; 137 sessions x 2 cameras
  (cam1 only where it exists).
* Tracking holds frames x points x 2 float32 in RAM: 2 M x ~800 x 8 B ~ 13 GB per video (+ video buffer) -> O2 CPU
  tasks with >= 48 GB, or tracked in ~30 min segments restarting at the home positions (relaxation pulls there anyway).
* Speed: paper ~67 fps on CPU for 1280x1024; ours is ~1/3 the pixels -> expect ~150-250 fps -> ~2.5-4 h per video ->
  ~1000 CPU-h for the cohort (O2 `short` partition, no GPU fairshare needed). Benchmark first (step 1).
* Spectrograms at full length are ~30 GB per session at 25 Hz -> keep trial windows (cue -2 .. +4 s) at ~25 Hz plus a
  whole-session copy at ~5 Hz.

## Plan

1. **Pilot + benchmark (local, now).** One session per camera through FR with the paper's tracking values, Fs 250:
   fps, memory, violation fraction, and how points behave at the spout rod and the tongue.
2. **Session frames.** Per session x camera a median face from pre-cue frames with the mouth closed (LP tongue absent),
   CLAHE (FR `Image_preparation_pipeline.apply_clahe`).
3. **Registration -> master face per camera.** Within animal: session frame -> animal template (an August pre session),
   ECC (euclidean/affine) then DeepFlow non-rigid (`fr.rois.Image_Aligner`). Across animals: animal template -> master
   (one animal's template, or an iterated mean of the four), affine + non-rigid. Compose session -> master. QC is
   QUANTITATIVE: warp each session's median LP keypoints (nose, jaw, spout tip at rest) through the field and report the
   error against the master's (px), plus a GIF of warped faces.
4. **Master ROIs** (Priya, once per camera, FR GUI on the master face): the face outline + background mask. Exclude the
   rod's path (cam4: below the lower lip; cam1: the rod crosses the face at centre spouts -> either a narrower ROI or
   post-hoc masking of points within LP spout x +- r per frame).
5. **O2 bundle** (`scripts/o2_face_rhythm.py bundle / collect`, the `o2_video_me` pattern): per session x camera task =
   warped ROIs.h5 + video path -> FR tracking + VQT -> trial-window spectrograms on the DAQ clock (alignment templates,
   as everything else) + decimated session spectrogram + violation fraction. Priya runs it.
6. **Decomposition.** Recommended: ONE joint non-negative CP per camera on the master-mesh points, fitted on a balanced
   subsample of trial windows (August pre + every epoch, all four animals), then every session's temporal factors by
   non-negative least squares with the spatial + spectral factors fixed -> components comparable across sessions and
   animals by construction (the paper's per-session fits + Hungarian matching is the fallback / check).
7. **Use.** Per-trial "reacted" calls and latencies from the whisking / sniffing / nose components (alongside the jaw
   rule), and component time courses as regressors in the encoding window's models.

## Decisions for Priya

* camera order: cam4 first (whisker pads in view, rod below the face), cam1 second?
* master face: one animal's template, or a mean of the four after registration?
* ROI: drawn by Priya on the master (recommended) or generated from LP landmarks?
* decomposition: joint fit + projection (recommended) vs per-session + matching (paper)?
* does the colleague's stroke_orofacial master mesh code exist somewhere we can reuse (Flea3 rig, different animals)?

## Status (10-21 evening)

* **Parameters chosen: `midlk12`** (DECISIONS.md 10-21 evening): rigidity 0.03 / 24, relaxation 0.005, LK 50 px / 5
  levels, 12 px (1282 points). Master re-seeded at 12 px and re-warped to all 106 sessions; bundle rebuilt (8 workers x
  2 threads, 56 GB, est. 2.2-4.5 h per session). **Pilot task 87 running on O2**; array may be submitted alongside.
* Next: step 6 (decomposition) and step 7 (use) -- see the integration plan once written.

## Status (10-21 late morning)

Handoff for steps 4-5: [`docs/status/STATUS_2026-10-21_FACE_RHYTHM_O2_HANDOFF.md`](status/STATUS_2026-10-21_FACE_RHYTHM_O2_HANDOFF.md).

* **1 pilot** done (desktop: 100 fps at 1514 points, VQT ~3200 frames/s).
* **2 frames / 3 registration**: pose/3-D window (`face_rhythm_prep.py`, `face_rhythm_register.py`). The first master
  (PS93_20260821) was post-stroke (acute) -- rebuilt from August PRE templates: master **PS95_20260812**.
* **4 master ROI**: drawn by Priya with `notebooks/face_rhythm_draw_roi.ipynb` (click polygons; face to the eyes, mask
  ~25 px outside, bottom above the close-spout tips); to be re-checked on the new master, then warped to sessions
  (`face_rhythm_warp_rois.py`). Decided: one animal's template as master (not a mean); ROI drawn by Priya.
* **Encoder keyframe line** (20.83 Hz, cam4 FMP4 GOP 12) notched in the runner before the VQT (handoff pitfalls).
* **Tracking-parameter test** (10-21 afternoon, PS95_0812 30k frames, 8 sets scored + 3 queued): recommendation
  `mids12` (rigidity 0.03 / 24, relaxation 0.005, LK 40 px / 3, 12 px spacing); round 3 (mids12 mesh + oldrun's LK)
  pending; factors identical across sets (sniff, whisking, shared + L / R lick-direction components). Production not
  started; first full-session pilot OOM-killed (fixed). Handoff has the table.
* **5 O2 bundle** built (`scripts/o2_face_rhythm.py`, `fr_cam4_20261021`, 108 tasks: June out, engaged trials,
  PS92_0812 as two tasks); waiting on the warped ROIs. Open: a tracking-parameter test (paper vs the old 2pRAM run vs
  FR defaults, scored against LP) before the full array.
