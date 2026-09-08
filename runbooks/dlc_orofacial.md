# Runbook — DeepLabCut orofacial tracking on the widefield rig

Design + rationale: [`../DECISIONS.md`](../DECISIONS.md) §"DLC on the widefield rig". Parameters:
`configs/defaults.yaml dlc.*`. Module reference:
[`../wfield_local/README.md`](../wfield_local/README.md) §"DLC orofacial tracking".

**This is a NEW project, not a re-run of the 2pRAM one.** The earlier cohort's project at
`MICROSCOPE/Priya/DeepLabCut/DLC_train_config` is the weight donor and the read-only reference; it is
never modified from here. Its network is genuinely good (test RMSE 2.70 px) on a view this rig does
not have: `cam4` is ~2–2.5× more zoomed than the old `video2`, and the training augmentation only
ever scaled *down* from its own view, so those weights are an initialisation and not a predictor.

---

## Step 0 — is 3D possible yet? (run this before labelling anything)

```powershell
conda activate locanmf
python -m wfield_local.dlc_calibration          # newest camera_calibration_<YYYYMMDD>/
```

Read the **poses** column, not the frame counts. A board held still in front of a camera is one
view of it however many frames that fills, and a calibration is constrained by distinct views. On
`camera_calibration_20260805` at 50 Hz:

| cam | usable frames | **poses** | px/bit | | pair | frames | **poses** |
|---|---|---|---|---|---|---|---|
| cam1 | 6350 | 20 | 5.2 | | cam1-cam4 | 6163 | 22 OK |
| cam2 | 70 | 16 | **2.1** | | cam1-cam2 | 69 | 16 OK |
| cam3 | 8 | 7 | ~2 | | cam2-cam4 | 70 | 16 OK |
| cam4 | 8969 | **4** | 5.2 | | cam1-cam3 | 8 | 7 thin |
| | | | | | cam3-cam4 | 8 | 7 thin |
| | | | | | cam2-cam3 | 0 | 0 none |

So the board **was** presented to the side views, repeatedly — 12 separate episodes each, spread
across the whole recording. Two things are wrong with it, and neither is "you didn't sweep enough":

1. **The board is too small for `cam2`/`cam3`.** Their markers come out at ~2.1 px per code cell
   against `cam4`'s 5.1; a DICT_4X4 marker is 6 cells across and stops decoding below ~3. The squares
   are found and then rejected — 58 rejected candidates in one cam2 frame. No detector setting fixes
   this. **Print the board ~2.5x larger, or hold it that much closer to the side cameras.**
2. **`cam4` was held, not swept.** 8,969 usable frames and **four** distinct poses. It looks like the
   best camera in the rig on every frame-based measure and has the least-constrained intrinsics.

### Re-recording the calibration

0. **Print the new board.** Ready to print on the share at
   `Behavior_Cameras/calibration_boards/charuco_9x7_10mm_a4.pdf` (94 x 74 mm board on A4; an A3
   variant at 154 x 112 mm is beside it), or regenerate with `python -m wfield_local.dlc_board`.

   Its markers are **8 mm against the current ~1.2 mm** — sized from the recording, not from paper:
   in cam2 the printed pattern spans ~145 px while the card it is taped to (a 30 mm cage plate,
   38.1 mm) spans ~390 px. That puts cam2 near 14 px per code cell instead of 2.1. It is
   deliberately NOT a full page: ~20x would leave `cam4`, whose field is the snout alone, seeing two
   squares.

   **Print at 100% / "actual size", then measure the 100 mm ruler on the sheet.** "Fit to page" is
   on by default in most dialogs and would rescale the board — which changes nothing about the
   images and everything about the millimetres they imply, with nothing downstream to reveal it.
   The board carries its own spec (squares, mm, dictionary) so the geometry cannot be separated from
   the object, and a pattern-free **mount tab** for taping to the cage plate so no marker is covered.

### "Is it better for the board to be too big?"

**Yes — err big, because the two failure modes are not symmetric.** Too big means `cam1`/`cam4` see
only PART of the board, and ChArUco is built for exactly that: every marker carries a unique id, so
a partial view is still uniquely located and yields a pose. It costs corners per frame, a precision
cost repaid by more poses. Too small means `cam2`/`cam3` cannot DECODE a marker at all — the square
is found, rejected, and the frame contributes nothing. That one is a cliff, not a slope.

But size does not buy latitude. Both bounds scale with the board, so the usable window of working
distances is always about **4x** wide — bigger boards *move* it further out rather than widening it
(`working_window`, pinned in `test_making_the_board_bigger_MOVES_the_window_it_does_not_widen_it`).
As a multiple of the distance the 08-05 board was held at:

| board | marker | usable window |
|---|---|---|
| current | 1.2 mm | **0.18–0.73x** — excludes 1.0, i.e. no usable distance where it was held |
| **new A4** 9x7 | 8.0 mm | **1.2–4.8x** |
| new A3 11x8 | 10.5 mm | 1.5–6.4x |
| 2x bigger | 16 mm | 2.3–9.7x |
| 4x bigger | 32 mm | 4.6–19.4x |

So the board is not chosen by "how big can I print" but by **where you can physically stand**. The
A4 board wants ~1.2–4.8x the old standoff, which is inside the enclosure; a 32 mm-marker board would
need 4.6x minimum and probably puts you outside it. Print the A3 as well and use whichever suits the
distance you can actually achieve — it costs a sheet of paper.

**Pre-flight before the real sweep:** record ~30 s, run
`python -m wfield_local.dlc_calibration --step 5`, and read the `px/bit` column. Above ~5 on every
camera means the size and standoff are right; then do the full sweep.

1. Animal off the rig. All four cameras recording, as in a session.
3. **Move it constantly**: pause ~0.5 s per pose, then change position AND tilt. Aim for **≥20
   distinct poses per camera** and **≥15 shared per pair**. Holding it steady adds frames, not poses,
   which is exactly the mistake in the 08-05 recording.
4. Work the pairs deliberately: `cam3↔cam1` and `cam3↔cam4` are the thin ones (7 poses each), and
   `cam2↔cam3` has none. `cam2↔cam3` face opposite sides and may share no field of view at all —
   that is fine, the graph only has to be *connected*, not complete.
5. Save as `Behavior_Cameras/camera_calibration_<YYYYMMDD>/` — `dlc_calibration` picks up the newest
   by the date in the name, with no config edit.
6. **Write down the board's physical geometry**: squaresX, squaresY, square length mm, marker length
   mm. Not recoverable from the video, and without it a reconstruction has no metric scale. Put it in
   `configs/defaults.yaml dlc.board.*` when you have it.
7. Re-run step 0 with `--step 5` and confirm `RESULT: pair graph CONNECTED` with every camera at
   ≥20 poses.

---

## Step 1 — extract the labelling set

```powershell
python -m wfield_local.dlc_frames --cohort --dry-run    # inspect first
python -m wfield_local.dlc_frames --cohort
```

Frames land in DLC's own layout at
`Behavior_Cameras/Widefield/dlc/labeled-data/<video-stem>/img<FRAME>.png`, with
`frame_manifest.csv` beside it giving every frame's animal / date / camera / epoch / trial / spout
position / trial category / phase.

Two kinds of frame per session, per camera:

* **24 cue-locked** — 6 spout positions × 4 within-trial phases (ENL, Cue, early, late).
* **36 lick-locked** — 6 DAQ lick onsets (one per position) × 6 offsets spanning the tongue-out
  epoch (`−16, 0, +16, +32, +48, +64 ms`). These exist because the tongue is out for only ~70 ms
  per lick, so fixed cue offsets catch it by accident: of 96 ground-truthed frames, 8 landed within
  40 ms of a lick. The phase name records the offset (`lick+32`) so a labelled frame traces back
  to a point in the cycle.

Current cohort set: **3401 frames — cam4 927, cam1 926, cam2 774, cam3 774** (1248 cue-locked,
2153 lick-locked; the `lick` phase with no offset is an earlier single-offset batch, still valid).

`dlc.frames.lick_per_session` is the knob if that is more tongue frames than you want to label; the
offsets themselves were measured (see DECISIONS.md) and a symmetric ±28 ms window misses the peak.

Re-running is safe and idempotent: selection is seeded per session, and an image that already exists
is never overwritten. Add a single date later with `python -m wfield_local.dlc_frames <YYYYMMDD>`.

**Check the manifest covers every epoch before labelling.** If an epoch is missing, the run said so
in capitals and the cause is almost always a missing or failed `camera_sync` alignment template for
one of the two cameras.

---

## Step 1b — seed the labels from the 2pRAM network (`cam4` ONLY)

**On `cam4` you do not have to label from scratch.** The donor transfers once the frames are resized
to the apparent size it was trained at — `dlc.prelabel.scale = 0.45`. At native scale it finds the
nose on 8% of frames; at 0.45, on 97%.

**On the other three views it does not transfer at all**, and this was measured across a scale sweep
rather than assumed:

| view | best case (fraction ≥0.6) |
|---|---|
| `cam1` bottom | jaw 5.9%, tongue 0.3%, **spout 0%** |
| `cam2` side | jaw 14%, tongue 1%, eye ~0%, spout 38% |

`cam4` is the only view whose geometry resembles the old frontal `video2`; the others are new
viewpoints. On `cam2` one number looks like transfer — `L_whiskers_3` at 76% — and is not: overlaying
the most confident predictions puts them all on the snout region generally, with jaw, tongue, eye and
spout never firing. The network recognises "front of a face" and nothing more.

So `cam1`/`cam2`/`cam3` are labelled without a seed. Their frames are extracted and stratified
exactly the same way, so nothing else about the workflow changes. If seeding them matters, the option
worth trying is DLC 3's **SuperAnimal-Quadruped**, trained across many animals in side view — it
carries eye and nose but not tongue or spout, so it would seed part of the set.

```powershell
conda activate dlc
python -m wfield_local.dlc_prelabel --cam cam4 --dry-run   # coverage report, writes nothing
python -m wfield_local.dlc_prelabel --cam cam4
```

Writes `CollectedData_Priya.{h5,csv}` into each `labeled-data/` folder — **~82% of points seeded
across the 927 cam4 frames.** Refuses to overwrite an existing CollectedData unless you pass
`--force`, so a re-run cannot discard corrections.

Seeded fraction, by within-trial phase — the phase matters more than the average:

| phase | n | tongue | jaw | nose | spout |
|---|---|---|---|---|---|
| ENL (−0.6 s) | 78 | 2.6% | 91% | 96% | 69% |
| Cue (+0.05 s) | 78 | 5.1% | 89% | 99% | 78% |
| early (+0.4 s) | 78 | 35% | 58% | 96% | 78% |
| late (+1.5 s) | 78 | 14% | 74% | 96% | 74% |
| lick −16 ms | 77 | 44% | 43% | 97% | 92% |
| lick 0 ms | 77 | 34% | 25% | 97% | 91% |
| **lick +16 ms** | 77 | **58%** | 26% | 97% | 78% |
| lick +32 ms | 77 | 51% | 23% | 97% | 79% |
| **lick +48 ms** | 77 | **58%** | 33% | 97% | 81% |
| lick +64 ms | 77 | 36% | 47% | 97% | 90% |

**The tongue is found where the tongue is out** (61% on lick-locked frames) and is near-silent when
it is in (2.6% on ENL). The **jaw** is the part that needs you most: it falls to 13% on lick frames,
because the chin point the donor learned is occluded once the mouth is open at the spout.

**The eyes stay blank on purpose** — outside cam4's field of view entirely, yet returned at
0.44–0.79 in the top corners. A hallucination is not a weak detection and no threshold separates
them. A blank cell is what the labelling GUI shows as "place this".

**The tongue seed has a consistent offset.** It lands at the tongue–spout CONTACT — the lower edge
of the tongue where it meets the tube — not at the tip or centroid. That is where the donor's own
label sat on the old rig. Consistent bias is the easy kind: decide once where you want `tongue` to
be, and every seeded point needs the same nudge in the same direction. **Decide it before you
start**, because that choice defines the kinematic variable for the whole study.

So the manual pass is: **place the jaw on lick frames, fill the missing tongues, nudge the seeded
ones to your chosen landmark, check the rest.** Expect to move some whisker points too — they land
on the whisker field, but their anatomical identity across the two rigs is not guaranteed.

### First-time `dlc` env setup (this box)

`locanmf` is left alone (CLAUDE.md rule 6); DeepLabCut goes in its own env.

```powershell
conda create -n dlc python=3.10 -y
conda activate dlc
pip install "deeplabcut[pytorch]"
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0+cu128 torchvision==0.26.0+cu128
pip install -e C:\Users\SabatiniLab\Github\widefield_pipeline --no-deps    # makes wfield_local importable
```

The cu128 wheels are not optional — the RTX 5060 is Blackwell and the default torch build is CPU-only
here. Measured: 217 fps at 320×320, 66 fps at 672×672.

---

## Step 1c — what to label FIRST, and what to leave alone

The frames exist for all four views, but the work is not evenly distributed and not all of it is
useful yet:

| view | frames | parts | points | seeded | **by hand** | 3D today? |
|---|---|---|---|---|---|---|
| `cam4` front | 927 | 10 | 9,270 | 7,642 | **1,628** | yes, with cam1 |
| `cam1` bottom | 926 | 3 | 2,778 | 0 | **2,778** | yes, with cam4 |
| `cam2` side_left | 774 | 7 | 5,418 | 0 | **5,418** | not until recalibration |
| `cam3` side_right | 774 | 7 | 5,418 | 0 | **5,418** | not until recalibration |

**Label `cam4` + `cam1` now.** They are the one pair the existing calibration solves (1225 co-visible
board frames), and between them they carry `jaw`, `tongue` and `spout` — which is the entire set that
can be reconstructed in 3D today. 4,406 placements gets you 3D orofacial kinematics.

**Leave `cam2`/`cam3` until the calibration is re-recorded.** They are 71% of the remaining work and
none of it converts into 3D yet: the whiskers pair `cam4` with a side view, and `cam4↔cam2` currently
has 11 co-visible frames. Deferring costs nothing — the frames are extracted and the selection is
deterministic, so they will be exactly these frames afterwards.

**Do not label the side views "just jaw/tongue/spout" as a cheaper first pass.** It halves the
placements but discards the whiskers, which are the main reason to have side views at all: they are
triangulatable once calibration is fixed, a profile is the better view of whisking than a frontal one,
and PS93's phenotype includes minimal right whisking (`cam3`). And DLC labels per FRAME — adding a
bodypart later means reopening all 774 frames, so a partial pass is not less work, it is the same
work split in two with a revisit tax.

`nose`, `L_eye` and `R_eye` are single-view and stay 2D whatever happens. The eyes are still worth
placing when you do label the side views: one extra point on a frame already open, and the only rigid
landmark those views have if 3D ever fails.

---

## Step 2 — create the DLC project and adopt the frames

In the DLC environment (not `locanmf` — DeepLabCut is not installed there):

```python
import deeplabcut as d

cfg = d.create_new_project("widefield", "Priya", [<one video path per stem>],
                           working_directory=r"...\DeepLabCut", copy_videos=False)
```

The project's `bodyparts:` must be the **union** across views (twelve):

```
nose, jaw, tongue,
L_whiskers_1, L_whiskers_2, L_whiskers_3,
R_whiskers_1, R_whiskers_2, R_whiskers_3,
spout, L_eye, R_eye
```

but **only place what each view can actually see** (`configs/defaults.yaml dlc.cameras.<cam>.bodyparts`):

| view | role | place these |
|---|---|---|
| `cam4` | front | nose, jaw, tongue, L/R_whiskers_1-3, spout |
| `cam1` | bottom | jaw, tongue, spout — **no nose, no whiskers**: it looks UP at the underside |
| `cam2` | side_left | L_eye, jaw, tongue, L_whiskers_1-3, spout |
| `cam3` | side_right | R_eye, jaw, tongue, R_whiskers_1-3, spout |

A label for a part a camera cannot see is not merely wasted — it is invented, and a network trained
on invented points learns to hallucinate. **Names stay shared where views overlap**, which is what
makes 3D possible: `jaw`, `tongue` and `spout` are in all four views and are the parts triangulation
can reconstruct cohort-wide; whiskers pair `cam4` with one side view; `nose` is `cam4` only and stays
2D.

`L_spout` became `spout` (one spout, six positions) and `R_spout` is gone. The eyes survive only on
the side views — they were the centering fiducial in the old pipeline, not a result, and the
replacement is the 3D world frame, which head fixation makes static without any fiducial at all.
**`cam3` is the right-side view**, so it is the one carrying PS93's right orofacial deficit; that
laterality is a measurement decision, not a naming convention.

Copy `Behavior_Cameras/Widefield/dlc/labeled-data/*` into the project's `labeled-data/` — images AND
the `CollectedData_Priya.{h5,csv}` written in step 1b. The folder names are already the video stems
DLC expects, so the labelling GUI opens with the seeded points in place. Do NOT run `extract_frames`,
which would add appearance-clustered frames beside the designed set.

```python
d.label_frames(cfg)      # the manual step: place the tongue, check the rest
d.check_labels(cfg)
```

---

## Step 3 — train, starting from the 2pRAM snapshot

The donor is
`DeepLabCut/DLC_train_config/dlc-models-pytorch/iteration-5/video2Jan26-trainset80shuffle5/train/snapshot-best-060.pt`
(ResNet-50 group-norm, HeatmapHead, bottom-up). DLC 3.x initialises from a custom snapshot with a
bodypart conversion table; the head is re-initialised for the new keypoint set either way, so the
conversion table only carries the parts that survive:

```
nose -> nose,  jaw -> jaw,  tongue -> tongue,
L_whiskers_{1,2,3} -> L_whiskers_{1,2,3},  R_whiskers_{1,2,3} -> R_whiskers_{1,2,3},
L_spout -> spout
(R_spout, L_eye, R_eye: dropped)
```

**Add scale augmentation.** The donor trained with `affine.scaling: [1.0, 1.0]` and
`ResizeFromDataSizeCollate(min_scale=0.4, max_scale=1.0)` — it saw its own scale and smaller, never
larger, and the new view is larger. Leaving that unchanged wastes the main advantage of starting from
these weights.

Judge the result on `test rmse`, and separately on the **post-stroke** frames: the failure mode that
matters is a network that tracks a healthy mouse well and a hemiparetic one badly, which reads as a
deficit and is not one.

---

## Step 4 — inference on O2

Pattern ported from `DeepLabCut/code/20251112_DLC_batch_video_analysis_O2.ipynb` (paramiko/scp →
`/n/scratch`, `sbatch -p gpu --gres=gpu:1 --cpus-per-task=10 --mem=30G`, `module load
conda/miniforge3`, `conda activate deeplabcut`, a runner looping `analyze_videos` →
`filterpredictions` one video at a time, then pull the H5/CSV back). Parameters live in
`configs/defaults.yaml dlc.o2.*`.

**Budget it before submitting.** Measured throughput is ~86 frames/s (the old cohort's ~925k-frame
videos took ~3 h each at `batchsize 1`). A widefield session-camera is ~1.5 M frames at 250 fps:

| scope | per camera-session | `cam4` across the cohort |
|---|---|---|
| whole recording | ~4.8 h | ~31 GPU-days |
| cue-aligned windows (−1 s → +3.5 s) | ~1 h | ~7 GPU-days |

**Windows first** (Priya, 2026-09-08): enough to validate the network, and whole sessions only once
the bodypart set and thresholds are settled. Note what windowing costs — inter-trial behaviour
(grooming, spontaneous licking, treadmill bouts) is not in the output, and the quiet-period baseline
the widefield analysis uses lives in the ITI.

Not yet ported: the deployment module itself. `dlc.o2.*` holds its parameters.

---

## Where things live

| what | where |
|---|---|
| calibration recordings | `Behavior_Cameras/camera_calibration_<YYYYMMDD>/` |
| calibration survey report | `charuco_survey_<recording>.{csv,txt}` beside the recording |
| labelling frames + manifest | `Behavior_Cameras/Widefield/dlc/labeled-data/` |
| 2pRAM donor project (READ-ONLY) | `MICROSCOPE/Priya/DeepLabCut/DLC_train_config/` |
| 2pRAM inference outputs (READ-ONLY) | `MICROSCOPE/Priya/DeepLabCut/O2_analysis/` |
