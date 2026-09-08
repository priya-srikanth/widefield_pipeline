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

Read the **pair** table, not the per-camera table. A camera that sees the board on 90% of frames but
never at the same instant as another cannot be placed relative to it, and the per-camera numbers look
excellent in exactly that case. The verdict line is the answer.

As of `camera_calibration_20260805` the pair graph has three components — `cam1+cam4`, `cam2`,
`cam3` — so **3D reconstruction is blocked on a re-recorded calibration**. 2D labelling on
`cam4`/`cam1` is not blocked and can proceed in parallel.

### Re-recording the calibration

1. Animal off the rig. All four cameras recording, as in a session.
2. Move the ChArUco board slowly through poses where **two cameras see it at once**, and work every
   pair you can: `cam1↔cam4` (already fine), `cam2↔cam1`, `cam2↔cam4`, `cam3↔cam1`, `cam3↔cam4`.
   `cam2↔cam3` face opposite sides and may share no field of view at all — that is fine, the graph
   only has to be *connected*, not complete.
3. The side views see the board small and oblique across a much wider FOV, which is why they failed
   before. Hold it **closer to cam2/cam3**, or print a **larger** board.
4. Slowly. At 250 fps motion blur is not the limit, but a board swept through a pose in three frames
   contributes three near-identical views.
5. Save as `Behavior_Cameras/camera_calibration_<YYYYMMDD>/` — `dlc_calibration` picks up the newest
   by the date in the name, with no config edit.
6. **Write down the board's physical geometry**: squaresX, squaresY, square length mm, marker length
   mm. Not recoverable from the video, and without it a reconstruction has no metric scale. Put it in
   `configs/defaults.yaml dlc.board.*` when you have it.
7. Re-run step 0 and confirm `RESULT: pair graph CONNECTED`.

---

## Step 1 — extract the labelling set

```powershell
python -m wfield_local.dlc_frames --cohort --dry-run    # inspect first
python -m wfield_local.dlc_frames --cohort
```

624 frames — 13 animal × epoch cells × 24 frames × 2 cameras — into DLC's own layout at
`Behavior_Cameras/Widefield/dlc/labeled-data/<video-stem>/img<FRAME>.png`, with
`frame_manifest.csv` beside it giving every frame's animal / date / camera / epoch / trial / spout
position / trial category / phase.

Re-running is safe and idempotent: selection is seeded per session, and an image that already exists
is never overwritten. Add a single date later with `python -m wfield_local.dlc_frames <YYYYMMDD>`.

**Check the manifest covers every epoch before labelling.** If an epoch is missing, the run said so
in capitals and the cause is almost always a missing or failed `camera_sync` alignment template for
one of the two cameras.

---

## Step 1b — seed the labels from the 2pRAM network (so labelling is correction, not blank-page)

**You do not have to label from scratch.** The donor network transfers to `cam4` once the frames are
resized to match the apparent size it was trained at — `dlc.prelabel.scale = 0.45`. At native scale
it finds the nose on 8% of frames; at 0.45, on 97%.

```powershell
conda activate dlc
python -m wfield_local.dlc_prelabel --cam cam4 --dry-run   # coverage report, writes nothing
python -m wfield_local.dlc_prelabel --cam cam4
```

Writes `CollectedData_Priya.{h5,csv}` into each `labeled-data/` folder — **~83% of points seeded
across the 465 cam4 frames.** Refuses to overwrite an existing CollectedData unless you pass
`--force`, so a re-run cannot discard corrections.

Seeded fraction, by within-trial phase — the phase matters more than the average:

| phase | n | tongue | jaw | nose | spout |
|---|---|---|---|---|---|
| ENL (−0.6 s) | 78 | 2.6% | 91% | 96% | 69% |
| Cue (+0.05 s) | 78 | 5.1% | 89% | 99% | 78% |
| early (+0.4 s) | 78 | 35% | 58% | 96% | 78% |
| late (+1.5 s) | 78 | 14% | 74% | 96% | 74% |
| **lick-locked** | **153** | **61%** | **13%** | 98% | 84% |

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

## Step 2 — create the DLC project and adopt the frames

In the DLC environment (not `locanmf` — DeepLabCut is not installed there):

```python
import deeplabcut as d

cfg = d.create_new_project("widefield", "Priya", [<one video path per stem>],
                           working_directory=r"...\DeepLabCut", copy_videos=False)
```

Then edit the project `config.yaml` to match `configs/defaults.yaml dlc.bodyparts` exactly:

```
nose, jaw, tongue,
L_whiskers_1, L_whiskers_2, L_whiskers_3,
R_whiskers_1, R_whiskers_2, R_whiskers_3,
spout
```

Ten bodyparts, the same list for `cam4` and `cam1` — triangulation matches keypoints by NAME, so a
set that differs per view cannot be lifted to 3D later without re-labelling. `L_spout` became
`spout` (one spout, six positions); `R_spout` is gone; `L_eye`/`R_eye` are gone because they are out
of frame on both snout views. **The eye was the centering fiducial in the old pipeline, not a
result** — the replacement is the 3D world frame, which head fixation makes static without any
fiducial at all.

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
