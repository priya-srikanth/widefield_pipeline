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

### Why "several squares" on the snout cameras is not enough

The solve consumes ChArUco CORNERS, and corners grow as (squares-1)^2:

| squares visible | interior corners | equations | left over for intrinsics |
|---|---|---|---|
| 3x3 | 4 | 8 | **2** |
| 4x4 | 9 | 18 | 12 |
| 6x6 | 25 | 50 | 44 |

Every view spends **6 of its equations on its own board pose** (3 rotation, 3 translation); only
what is left constrains the shared intrinsics, which is the entire reason for having many views.
`cam1`'s best frame ever shows 5 corners — 10 equations, 6 on its own pose, **4 contributing to the
calibration**. A 25-corner view contributes 44. And a small planar patch is ill-conditioned on top
of that: focal length and distance trade off against each other unless the board is seen large and
at varied tilt, and four corners cannot break that ambiguity.

### The board size, not the distance, is the thing to change

Moving the board further back would fix the corner count and introduce two worse problems: the
cameras are focused on the MOUSE, so a board at twice that distance is out of focus and its corners
localise badly; and a calibration is most accurate where its data was, so calibrating at a depth the
animal never occupies extrapolates into the working volume.

There is a board that satisfies every camera AT the working distance. The constraints are
`marker >= 1.86 mm` (cam2/cam3 decode floor, 9.7 px/mm) and `square <= ~4.5 mm` (cam1's 20 mm field
must hold 4-5 squares), which leaves a real window:

| file | board | square | marker | cam2 px/cell | cam1 corners | cam4 corners |
|---|---|---|---|---|---|---|
| **`charuco_6x6_26mmboard_4.3mmsq.pdf`** | 26 mm | 4.33 mm | 3.29 mm | **5.3** | **9** | 16 |
| `charuco_7x7_26mmboard_3.7mmsq.pdf` | 26 mm | 3.71 mm | 2.82 mm | 4.6 | 16 | 25 |

**Start with the 6x6 at 26 mm.** Nine corners per view over 20+ views is a sound intrinsics solve
(12 equations each towards it), and its 0.55 mm per code cell prints more reliably than the 7x7's
0.47 mm. If your printer holds that detail cleanly, the 7x7 gives cam1 sixteen corners instead of
nine — check a print under magnification before deciding.

This is now the **printer**, not the cameras, that sets the floor: 3.3 mm markers put 0.55 mm per
code cell on paper, which wants 1200 dpi rather than 600.

### Re-recording the calibration

0. **Print the new board.** Ready to print in
   `Behavior_Cameras/calibration_boards/`, all on A4 unless noted:

   | file | board | squares | marker | markers | cam2 px/cell | working window |
   |---|---|---|---|---|---|---|
   | `charuco_5x5_40mmboard_8.0mmsq.pdf` | 40 x 40 mm | 5x5 | 6.1 mm | 12 | 11.1 | 0.88-3.68x |
   | **`charuco_6x6_40mmboard_6.7mmsq.pdf`** | **40 x 40 mm** | **6x6** | **5.1 mm** | **18** | **9.2** | **0.74-3.07x** |
   | `charuco_7x7_40mmboard_5.7mmsq.pdf` | 40 x 40 mm | 7x7 | 4.3 mm | 24 | 7.9 | 0.63-2.63x |
   | `charuco_5x5_50mmboard_10.0mmsq.pdf` | 50 x 50 mm | 5x5 | 7.6 mm | 12 | 13.8 | 1.10-4.60x |
   | `charuco_7x7_50mmboard_7.1mmsq.pdf` | 50 x 50 mm | 7x7 | 5.4 mm | 24 | 9.9 | 0.79-3.29x |
   | `charuco_9x9_50mmboard_5.6mmsq.pdf` | 50 x 50 mm | 9x9 | 4.2 mm | 40 | 7.7 | 0.61-2.56x |
   | `charuco_9x7_10mm_a4.pdf` | 94 x 74 mm | 9x7 | 8.0 mm | 31 | 17.5 | 1.16-4.84x |
   | `charuco_11x8_14mm_a3.pdf` (A3) | 154 x 112 mm | 11x8 | 10.5 mm | 44 | 19.1 | 1.54-6.36x |

   (`cam2 px/cell` at the distance the 08-05 board was held. The bar is ~3; that board managed 2.1.)

   > **SUPERSEDED 2026-09-10 — print `charuco_7x7_26mmboard_3.7mmsq.pdf`.** The 40 mm 6x6 was
   > recorded and measured: `cam2`/`cam3` are fine on it, but `cam1` never exceeds **6** ChArUco
   > corners and `cam4` **9**, against aniposelib's floor of 9 for intrinsics. cam1's 600 px frame
   > spans **~20 mm** at its measured 30 px/mm, so the board has to put 4+ squares inside 20 mm.
   > 7x7 at 26 mm gives cam1 ~16 corners and still leaves the side views 4.6 px/cell (floor ~3);
   > 6x6 at 26 mm lands cam1 on exactly 9, i.e. the threshold with no margin. Full table in
   > `DECISIONS.md` "Which board to print". The paragraph below is kept as the reasoning that led to
   > the 40 mm choice.

   **Start with the 40 mm 6x6.** At 40 mm the board is flush with the 30 mm cage plate (38.1 mm), so
   it adds nothing to the rig's footprint — and a smaller board moves its working window CLOSER,
   which is the direction you want when space is tight. Every 40 mm variant still clears the decode
   bar on cam2 by 2-4x, and 6x6 keeps 18 markers with a window that contains 1.0, so the distance
   you already worked at stays valid.

   Going below ~4 mm markers starts to be limited by the PRINTER rather than the cameras: the 7x7 at
   40 mm puts 0.72 mm per code cell on paper, which a 600 dpi laser handles but where toner spread
   begins to soften the edges. Check a print under magnification before committing to a session.

   Regenerate any size with
   `python -m wfield_local.dlc_board --size-mm 50 --squares 7x7`.

   **One board in frame at a time.** They all draw on DICT_4X4_50, so two boards in view share
   marker ids and a repeated id is a pose that can solve in two places.

   Their markers are **4-10 mm against the current ~1.2 mm** — sized from the recording, not from paper:
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
3b. **SWEEP THROUGH DEPTH, and tilt well off parallel.** This is a *separate* requirement from the
   board size, and the 09-10 recording met the size requirement for `cam2`/`cam3` and failed this
   one on all four cameras. Every pose sat at a similar depth and tilt, so focal length, principal
   point and distortion traded off against one another: solved freely, **three of four cameras put
   their principal point outside their own sensor** while reporting 0.9-1.9 px reprojection error.
   Move the board **toward and away from each camera** across the full range it stays in focus, and
   hold it at steep angles (≥ 30° off the image plane), not just face-on. Reprojection error cannot
   detect the failure this prevents — `dlc_anipose` checks for it explicitly and refuses to write a
   calibration that shows it.
4. Work the pairs deliberately: `cam3↔cam1` and `cam3↔cam4` are the thin ones (7 poses each), and
   `cam2↔cam3` has none. `cam2↔cam3` face opposite sides and may share no field of view at all —
   that is fine, the graph only has to be *connected*, not complete.
5. Save as `Behavior_Cameras/camera_calibration_<YYYYMMDD>/` — `dlc_calibration` picks up the newest
   by the date in the name, with no config edit.
6. **Write down the board's physical geometry**: squaresX, squaresY, square length mm, marker length
   mm. Not recoverable from the video, and without it a reconstruction has no metric scale.
   `dlc_board` writes a `board.yaml` sidecar — drop a copy into the recording folder. This is not a
   formality: the 08-05 recording's geometry was never written down and **cannot be recovered**
   (marker ids 0-37 fit a 7x11, a 9x9 and an 8x10 board equally well), which is the single reason it
   cannot contribute to a pooled solve today.
7. Re-run step 0 with `--step 5` and confirm `RESULT: pair graph CONNECTED` with every camera at
   ≥20 poses, then `python -m wfield_local.dlc_anipose --gate` for the thresholds the solver
   actually applies (see step 0c).

---

## Step 0c — the solve, and pooling more than one recording

`dlc_calibration` (step 0) says whether a recording is worth solving. `dlc_anipose` solves it, with
aniposelib's joint bundle adjustment over all four cameras rather than a chain of pairs.

```powershell
conda activate dlc
python -m wfield_local.dlc_anipose --gate                    # what the SOLVER accepts, no solve
python -m wfield_local.dlc_anipose                           # newest recording
python -m wfield_local.dlc_anipose --dir <A> --dir <B>       # pool INTRINSICS across recordings
python -m wfield_local.dlc_anipose --dir <A> --dir <B> --assume-rig-unmoved   # ... and extrinsics
```

**Run `--gate` before `--dir`-ing anything together.** aniposelib is stricter than our step-0 gate:
it initialises intrinsics only from views with **≥9** ChArUco corners and admits a row to the bundle
adjustment only at **≥8**. `dlc_calibration` screens at 6. A camera can clear step 0 and still be
dropped by the solver, which is what `--gate` exists to show you first.

Detections are cached beside the recording as `anipose_rows_<digest>.pkl`, keyed by board spec +
frame step + detector settings. The first run decodes four ~650 MB videos off the share and takes
tens of minutes; re-solving with different pooling options after that is seconds. `--refresh`
forces a re-detect.

### What pools and what does not

| | pools across recordings? | assumption |
|---|---|---|
| **intrinsics** (f, principal point, distortion) | **yes, freely** — even across different board sizes | nobody refocused or swapped a lens |
| **extrinsics** (where the cameras are relative to each other) | only behind `--assume-rig-unmoved` | the rig did not move between recordings |

Intrinsics pool because `calibrateCamera` takes object points *per view*, so a 40 mm board and a
26 mm board sit in one solve. This is the point of pooling: **no single board size works for all
four cameras** — one big enough for `cam2`/`cam3` to decode overflows the snout views, one small
enough for `cam1`/`cam4` falls below the decode floor on the side views.

Extrinsics are a fact about where the cameras were *on the day*. `--assume-rig-unmoved` is **tested,
not trusted**: any pair solvable in both recordings is solved separately in each and compared
(tolerance 1.0° / 2.0 mm). **If no pair is observed in both, the assumption is untestable and the
flag is refused** — untestable is not the same as true.

Extrinsics also never pool recordings that used *different boards*: aniposelib indexes object points
by ChArUco corner id against one geometry, and both our boards are DICT_4X4_50 with overlapping ids,
so mixing them would read one board's corners against the other's millimetres — silently.

### The check that reprojection error cannot make

`dlc_anipose` refuses a calibration whose **principal point lands outside the frame** (or >0.2 image
widths off centre), or whose **|k1| > 1**, *regardless of how good the reprojection error looks*.
On the 09-10 recording three of four cameras failed this at 0.9-1.9 px RMS. The model fits; it is
not identifiable, because the board never moved through depth. See step 0 item 3b — the fix is at
the rig, not in the solver.

`--fix-principal-point` pins it to the frame centre and rescues `cam2`/`cam3`. It is an
**assumption, not a measurement**: these frames are ROIs off a larger sensor, so the optical axis is
at the ROI centre only if the ROI is centred on the sensor. Use it to get moving, not to ship.

### Can the two recordings we already have be combined? No.

Measured 2026-09-10, full record in `DECISIONS.md`:

- **`camera_calibration_20260805`** — its `board.yaml` is all zeros. The geometry was never written
  down and is not recoverable (ids 0-37 fit several layouts), so it contributes nothing to a metric
  solve. Even granting it: `cam2` = 11 frames at 2.1 px/cell, `cam3` = 3 frames, `cam4` = **4
  distinct poses**. Only `cam1` has a real sweep.
- **`Widefield_calibration_20260910`** — `cam2`/`cam3` are fine; `cam1` reaches **0** views with ≥6
  corners (max 5 ever) and `cam4` reaches **4** (max 8). The 40 mm board is too large in the snout
  views: a fragment carries markers but no interpolated corners.
- **Together**: `cam4` is covered by neither, and no pair is observed in both, so the rig-unmoved
  test cannot even run.

The unblocking recording is the **26 mm board, swept through depth**. Its intrinsics then pool with
09-10's for `cam2`/`cam3`; if it is recorded close enough in time that the rig-unmoved test passes,
its extrinsics pool too.

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
pip install "deeplabcut[pytorch,gui]"
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0+cu128 torchvision==0.26.0+cu128
pip install -e C:\Users\SabatiniLab\Github\widefield_pipeline --no-deps    # makes wfield_local importable
```

**`[gui]` is not optional if you intend to label.** `deeplabcut[pytorch]` alone gets inference and
training but no Qt, and the failure surfaces as `AttributeError: 'label_frames' is unavailable`
several frames below a `ModuleNotFoundError: qtpy` — which reads as a broken project rather than a
missing extra. Add it later with `pip install "deeplabcut[gui]"`; nothing else needs redoing.

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
| `cam2` side_left | 774 | 8 | 6,192 | 0 | **6,192** | **yes** |
| `cam3` side_right | 774 | 8 | 6,192 | 0 | **6,192** | **yes** |

**SUPERSEDED 2026-09-12: the "not until recalibration" rows above were true of the AUGUST board and
are not true now.** The calibration was re-recorded on 2026-09-11 with the 26 mm 7x7 board and ALL
FOUR cameras solve — `cam2` and `cam3` resolve 36 corners in a single view, more than any other
camera, against the ~2.1 px/cell failure that blocked them in August. See DECISIONS 2026-09-12 for
the measured reconstruction precision (square 3.709 mm, SD 41 um across frames, -1.0% scale bias).
Nothing is blocked on calibration; this table's ordering is a PRIORITY, not a possibility.

**Label `cam4` + `cam1` first** — they carry `jaw`, `tongue` and `spout`, the orofacial parts the
study is about, and `cam4` is already 82% seeded so it is the cheapest place to build an eye for the
landmarks. **Then `cam2`/`cam3`**, which add the eyes and the whiskers.

**Part counts changed 2026-09-12**: `nose` was added to `cam1`, `cam2` and `cam3` (9ac523c), so the
side views carry 8 parts rather than 7 and `cam1` carries 4 rather than 3.

**SUPERSEDED 2026-09-22 — the whiskers are DEFERRED, so the side views ARE now "just
nose/jaw/tongue/spout" plus one eye each.** `dlc.cameras.*.bodyparts` dropped from twelve names to
six. The paragraph below is kept because its reasoning is still correct and is the thing to re-read
when the whiskers come back.

> **Do not label the side views "just jaw/tongue/spout" as a cheaper first pass.** It halves the
> placements but discards the whiskers, which are the main reason to have side views at all: they are
> triangulatable once calibration is fixed, a profile is the better view of whisking than a frontal one,
> and PS93's phenotype includes minimal right whisking (`cam3`). And DLC labels per FRAME — adding a
> bodypart later means reopening all 774 frames, so a partial pass is not less work, it is the same
> work split in two with a revisit tax.

**What changed is the ORDER, not the verdict.** Six points per frame over 774 frames per side view is
the largest single labelling cost in the project, on the parts whose landmark is least well defined —
a numbered whisker has to be the *same* whisker frame to frame — and for an analysis nobody has
written yet. The orofacial four are what the study turns on, and they were the bottleneck: the tongue
had 71 training frames against the nose's 192.

**The revisit tax above is real and is being accepted knowingly.** Coming back to whiskers means
reopening those frames. Against that: nothing is lost in the meantime — the donor's seeded whisker
values are still in the label files, the side-view frames are already extracted, and the columns are
inert rather than deleted. Add the names back to `dlc.cameras.<cam>.bodyparts` to resume.

`L_eye` and `R_eye` are single-view and stay 2D — one side camera each. They are still worth placing:
one extra point on a frame already open, and the only rigid landmark those views have if 3D ever
fails. `nose` is NO LONGER single-view (2026-09-12); it is in all four.

---

## Step 2 — LABEL

Everything upstream is done: the project exists on the share with the frames and cam4's seeds in it.
Two commands.

```powershell
conda activate dlc
cd C:\Users\SabatiniLab\Github\widefield_pipeline
python -m wfield_local.dlc_project --cam cam4 --cam cam1 --label
```

That refreshes the project (adds any new frames, repoints `project_path` to this machine's drive
letter, never touches a label you have edited), prints the per-view bodypart list, and opens DLC's
labelling GUI. Without `--label` it does everything except open the GUI.

The project is `Behavior_Cameras/Widefield/dlc/widefield-Priya-2026-09-08/` — **60 folders,
1150 images** (measured 2026-09-25): 15 per camera, `cam4` 360 images and fully labelled on all four
parts, `cam1` 310 with ONE folder partly done, `cam2`/`cam3` 240 each with no `CollectedData` file at
all. The earlier "26 folders, 1853 images, 13 of them cam4" counted a different extraction.

### If the GUI does not stay open

Two failures, both environment rather than project:

* `AttributeError: 'label_frames' is unavailable because DeepLabCut was loaded without GUI
  dependencies` — `pip install "deeplabcut[gui]"`.
* `QThread: Destroyed while thread 'StatusChecker' is still running`, window flashes and closes —
  DLC's `label_frames` builds a napari viewer and returns it **without starting a Qt event loop**,
  which is fine from IPython (the shell already runs one) and fatal from `python -m`. `dlc_project`
  calls `napari.run()` after it, which supplies the loop and blocks until you close the window. Any
  other script driving a `deeplabcut.gui` entry point needs the same — they are all written for
  interactive use.

`--folder <video-stem>` opens a particular session first; by default DLC opens the alphabetically
first folder, which is a cam1 June session.

### Place ONLY what each view can see

DLC has one bodypart list per project, so `config.yaml` carries the union of all twelve. The GUI
shows all twelve on every frame and **cannot enforce the per-view subset** — that part is yours:

| view | folders | place these | leave EMPTY |
|---|---|---|---|
| `cam4` front | 15 | nose, jaw, tongue, spout | everything else |
| `cam1` bottom | 15 | nose, jaw, tongue, spout | everything else |
| `cam2` side_left | 15 | nose, jaw, tongue, spout | everything else |
| `cam3` side_right | 15 | nose, jaw, tongue, spout | everything else |

**UPDATED 2026-09-25 — EVERY VIEW IS THE SAME FOUR PARTS NOW.** This table used to send the labeller
to six whiskers on `cam4` and to leave `nose` empty on `cam1`, and both instructions were already
wrong when read: `nose` joined all four views on 2026-09-12, the whiskers were deferred on
2026-09-22, and the eyes on 2026-09-25 (Priya: *"we aren't using the whisker or eye points for
now"*). The live project's `config.yaml` carries six names today and will carry four after the next
`dlc_project` run. **A runbook that asks for a part the config no longer lists is worse than a stale
one** — the GUI would happily accept those points and `dlc.train.bodyparts` would then be a claim
nobody checked.

**Why the eyes are empty on cam4:** they are outside its field of view — sampled at four separated
timepoints in PS94 and one in PS95, never in frame. The donor network does return them at 0.44–0.79
in cam4's empty top corners, which is a hallucination and exactly why they are not seeded.
~~they get filled on `cam2` (L_eye) and `cam3` (R_eye)~~ — **not any more.** The eyes were deferred
on 2026-09-25 along with the whiskers, so no view asks for one. Nothing was stranded: `cam2` and
`cam3` had never been labelled at all, so not a single eye had been placed. See
`dlc.cameras` in `configs/defaults.yaml` for the reasoning and for how to bring them back.

A point placed for a part a camera cannot see is invented data, and a network trained on invented
points learns to hallucinate.

### What the work actually is

* **cam4** — mostly CHECKING; ~82% is placed. The **jaw** needs the most attention: it falls to 13%
  on lick frames because the chin point the donor learned is occluded once the mouth is open at the
  spout. Fill the missing tongues and nudge the seeded ones to your chosen landmark.
* **cam1** — from scratch, and FOUR parts, not three: `nose` joined it on 2026-09-12. ~~It is the
  best tongue view on the rig~~ — corrected 2026-09-13, the spout occludes the tongue TIP from below
  and `cam4` is the better tongue view; see `dlc.cameras.cam1` for why that changes what a cam1
  tongue label means for triangulation.

**Decide the tongue landmark before you start.** The seeds land at the tongue–spout CONTACT (where
the donor's own label sat), not the tip or centroid. Consistent bias is easy to correct — but pick
one and hold to it, because that choice defines the kinematic variable for the whole study.

Save often; the GUI writes `CollectedData_Priya.h5` per folder and that file is the only record of
the work.

```python
import deeplabcut as d
d.check_labels(cfg)      # renders your labels onto the frames for a visual audit
```

## Step 3 — train on the REFINED labels, starting from the 2pRAM snapshot

```powershell
conda activate dlc
python -m wfield_local.dlc_train --dry-run     # stage + audit + print the split; train nothing
python -m wfield_local.dlc_train               # stage, create the training set, train, evaluate
```

`wfield_local/dlc_train.py` is DLC's own refine-and-retrain loop — `create_training_dataset`,
`train_network`, `evaluate_network`, `WeightInitialization`, and `iteration` as the round counter.
Nothing in it reimplements a step DeepLabCut has.

### It trains the parts a HUMAN placed, which is not the same as the parts that have values

`dlc.train.bodyparts` = `[nose, jaw, tongue, spout]` as of 2026-09-21. `dlc_prelabel` seeds **ten**
bodyparts on `cam4`, and a seed is a *prediction*: the refinement pass covered those four across all
15 folders and left the six whisker columns as the donor wrote them. Measured against the one
surviving pre-refinement backup (`cam4_2026-06-06T12_25_18/…h5.bak-20260916-161730`):

| part | frames moved > 0.5 px | median displacement |
|---|---|---|
| `nose` | 16/16 | 3.42 px |
| `spout` | 14/16 | 0.91 px |
| `jaw` | 8/12 | 0.66 px |
| `tongue` | 4/4 | 3.37 px |
| `L/R_whiskers_1..3` | **1/16 each** | **0.00 px** |

Training the union would fit six head channels to the donor's guesses about a view 2.5x more zoomed
than the one it learned. **Nothing in code can tell a seed from a label** — both are just coordinates
— so that config list is the contract, and `dlc_train` prints the on-disk counts beside it every run.
**Widen it when the labelling widens, not before.**

Dropping the whiskers makes the transfer *better*, not worse: the conversion table shrinks to

```
nose -> nose,  jaw -> jaw,  tongue -> tongue,  spout -> R_spout      = head channels [0, 1, 2, 12]
```

and every channel that survives is one whose supervision is a human label rather than a re-fit of the
donor's own output. (R, not L: it clears 0.6 on 53% of frames against L's 23%.) The head is
re-initialised for any part without a donor counterpart either way.

### Fill counts, and how to read the tongue's

```
nose    240 / 240      jaw    197 / 240
spout   235 / 240      tongue  93 / 240
```

DLC masks NaN keypoints out of the loss, so a part is learned only from the frames where it is placed
— **a blank is not taught as absence.** That makes 93 tongues *complete*, not 39% done:
`dlc.frames.lick_fraction` 0.5 with one tongue-in offset of four puts ~90 of 240 frames in the
tongue-out set. `jaw` at 197 is the part with real room left, for the reason Step 2 gives — the chin
point is occluded once the mouth is open at the spout.

### It trains in a SEPARATE project, and the copy direction is the opposite of Step 2's

The training project is `training/widefield-Priya-2026-09-08-orofacial`, in a **subdirectory** beside
the labelling one. Not an edit in place, for three reasons that are all about the other views:

* `config.yaml` holds **one** bodypart list, so cutting the live one to four would make `cam2`/`cam3`
  — 8 parts each, including the eyes, not yet labelled at all — impossible to label.
* **`dlc_project.write_config` rewrites `bodyparts` to the union on EVERY run.** An in-place edit is
  silently reverted the next time anyone opens the GUI, and a training set rebuilt after that would
  quietly pull the whisker seeds back in *without erroring*. This is the trap.
* `iteration` is project-wide, so bumping it for a cam4 round would re-version the unlabelled work.

**Labels are copied FORWARD into it every run and overwritten** — the opposite of Step 2's rule,
because there the human edit is downstream of the copy and here it is upstream. Do not label in the
training project.

### The split holds out whole SESSIONS

DLC's default split is uniform over frames, which on this set measures the wrong thing:
`dlc.frames.lick_offsets_s` samples four points of one ~80 ms protrusion, and 99–100% of within-onset
frame pairs are closer in appearance than the 5th percentile of between-onset pairs. A uniform split
puts near-duplicates on both sides and reports a test RMSE that is optimistic about the only thing
the number is for — a session the network has not seen.

Sessions are held out whole, one per epoch in a seeded round-robin. At 0.8 that is 3 of 15:

```
cam4_2026-08-20T16_31_02  [acute]     cam4_2026-09-07T17_08_48  [chronic]
cam4_2026-06-06T18_02_39  [pre]       train 192 frames from 12 sessions
```

**`subacute` gets no held-out session at 20%** — three sessions do not cover four epochs — so it is
untested rather than fine. Lower `dlc.train.training_fraction` to reach it.

### Judge it on the per-epoch table, not the scalar

```powershell
python -m wfield_local.dlc_train --evaluate     # re-evaluate the newest snapshot, no training
```

`evaluate_network` runs with `per_keypoint_evaluation=True` (one RMSE over four parts hides the tongue
and the jaw, which are the two that matter), and `per_epoch_error` then answers the question this
study actually turns on:

> the failure mode that matters is a network that tracks a healthy mouse well and a hemiparetic one
> badly, which reads as a deficit and is not one

That is a per-**epoch** contrast, and not a number DLC produces. It is reported **split by train/test**
because `evaluate_network` predicts on every labelled image: pooling them makes each epoch's error
depend on what share of that epoch's sessions happened to be held out — an artefact of the split
reported as a property of the epoch, which is the same mistake the table exists to catch. Read the
`test` rows.

### CORRECTED: you do not have to add scale augmentation by hand

This step used to say to add it, because the donor trained with `affine.scaling: [1.0, 1.0]` and
`ResizeFromDataSizeCollate(min_scale=0.4, max_scale=1.0)` — its own scale and smaller, never larger,
against a new view that is larger. True of the donor's config, and **not inherited**:
`create_training_dataset` builds a *fresh* `pytorch_config.yaml` from DLC 3.0.1's own templates, whose
default is already `affine.scaling: [0.5, 1.25]` — two-sided — with 448×448 `crop_sampling` instead of
the collate. Only the snapshot's **weights** come across. `dlc.train.scaling` states that value
explicitly rather than inheriting it silently, and is the knob if you want to reach further *down*
toward the donor's apparent size (~0.4x cam4's).

### First result (2026-09-21), and how to read it

> **SUPERSEDED 2026-09-25 by round 2 — see "Second result" below.** Kept because the way it is READ
> is still how to read the next one, and because the round-1 numbers are the baseline the round-2
> table is measured against. The network it describes is still on disk at `iteration-0`.


`snapshot_best-90`, scored on the three held-out sessions:

| part | test RMSE | | part | test RMSE |
|---|---|---|---|---|
| `spout` | **2.19 px** | | `jaw` | **4.70 px** |
| `nose` | **4.17 px** | | `tongue` | 10.27 px |

**The tongue's 10.27 px is six frames, not a tongue problem.** Median error over all 93 labelled
tongues is **1.91 px**; exactly 6 frames exceed 20 px. `--worst` ranks them:

```
bodypart  err_px  likelihood split    epoch                  session          image
  tongue  266.56        0.01 train subacute cam4_2026-08-26T12_25_41 img1796416.png
  tongue   33.40        0.61  test  chronic cam4_2026-09-07T17_08_48 img0674392.png
  tongue   25.59        0.63  test      pre cam4_2026-06-06T18_02_39 img0389083.png
```

* **The 266 px one is a misclick.** The tongue is at x=619 on a 680 px frame while every other tongue
  in that session sits at x=338–373 and that frame's own nose/jaw/spout are at x=244–323. It is a
  TRAIN frame at likelihood 0.012 — the network had every chance to fit it and refused, so the label
  is the thing that is wrong. It alone produces the `subacute` train tongue RMSE of 53 px.
* **Four more are two whole licks** (`img0389079/083/091`, `img0674384/392/400`) at likelihood
  0.46–0.63, i.e. unsure rather than confidently wrong. This is "decide the tongue landmark before you
  start" cashing out: one lick labelled at a different point of the tongue reads as error, and pruning
  keeps a lick whole so they arrive in threes.

**`spout` and `nose` are flat across epochs** (2.30/2.35/2.99 and 4.69/4.91/3.82 for acute/chronic/pre)
— no sign of the false-deficit failure mode. `jaw` is *better* acute (3.36) than pre/chronic
(6.40/7.43), the opposite direction. The tongue cannot be read at 5–9 test frames per epoch until
those six frames are resolved. Full table in `DECISIONS.md`.

**Next is a refine round, not more epochs.** Train loss kept falling (0.0052 → 0.0003) while test RMSE
plateaued by ~epoch 20.

### Which frames to relabel, and how

```powershell
python -m wfield_local.dlc_train --evaluate --review
```

Classifies every frame over tolerance by **what is wrong**, and writes an annotated crop per frame to
`<training project>/review/` — **red cross = your label, cyan circle = the network**. The verdict comes
from the network's CONFIDENCE, not the error size, because the same 20 px error means different things:

| verdict | error | network | what it means | action |
|---|---|---|---|---|
| `DELETE` | big | says nothing is there (`p < 0.05`) | a point placed where the part is not | remove the point |
| `REPLACE` | big | **confident** elsewhere (`p > 0.7`) | the label departs from your own convention | move it |
| `DECIDE` | big | unsure (`0.05–0.7`) | genuinely ambiguous landmark | pick one and hold it |
| `ADD` | — | confident, nothing labelled | a visible part nobody placed | add a point |

**`REPLACE` reads the network as your own consensus.** It learned the landmark from the other ~190
frames, so where it is confident and disagrees, that frame is the odd one out — not the network.

**Do NOT use label position as the test.** A tongue 100 px from its session's median is a long
protrusion, not a mistake; a distance-from-median rule flags 13 cam4 frames of which 12 are fine.

**A fault that repeats across a lick's frames is a LANDMARK fault.** `lick_offsets_s` samples four
points of one protrusion and pruning keeps a lick whole, so `--review` rolls those up: re-place such a
lick **together**, because seeing the tongue move is what makes "the tip" identifiable at all.

### Handing the list to whoever is labelling

```powershell
python -m wfield_local.dlc_train --evaluate --guide
```

Writes **`CORRECTION_GUIDE.html`** next to `LABELLING_GUIDE.html` on the share — the page the
labeller actually works from. Every flagged frame appears with its **crop embedded** (red cross =
their label, cyan circle = the network) and what to do to it in the vocabulary
`LABELLING_GUIDE.html` already taught (the `+` tool, the select arrow, `Delete`,
`File -> Save Selected Layer(s)...`).

**It is written for a MacBook, because that is where the labelling happens.** On her own laptop she
has `napari` + `napari-deeplabcut` and NOT DeepLabCut and NOT this repo, so
`python -m wfield_local.dlc_project --label` is not a command she can run at all. The Mac route is
`conda activate label` -> `napari` -> *File -> Open Folder...*, which means what the page has to give
her per session is a **path to paste** into `Cmd+Shift+G`, not a command line — so each session block
carries its full `/Volumes/Neurobio/.../labeled-data/<stem>`, and the rig `--folder` command is kept
in a collapsed note for whoever is at the rig instead. It also repeats the two mount traps the main
guide documents: `_frame_staging` holds identical images under identical names and only fails at SAVE
time, and a doubled `Neurobio-1` mount is the usual cause of "that path does not exist".

**It is generated, never edited.** The list changes every refinement round, so a hand-maintained copy
would go stale in the worst way — telling someone to move a point that has already been moved. Rerun
the command after each retrain; the page rebuilds from the network's own output. A round with nothing
to fix renders as "Nothing to fix" rather than as the previous round's list.

Two things it tells the labeller that nothing else does:

* **Do not work in the `training/` copy.** `dlc_train.stage()` overwrites its labels from the
  labelling project on every run, so a correction made there is destroyed by the next retrain and
  nothing warns you. The commands in the guide always open the real project.
* **Do not settle the tongue landmark alone.** `DECIDE` frames are flagged precisely because it is
  not obvious; that choice defines the kinematic variable for the whole study.

`LABELLING_GUIDE.html` has a pointer section to it (*After the first pass: fixing the flagged
labels*), so the permanent instructions and the per-round list stay separate — the first is
hand-written and stable, the second is regenerated and disposable.

### Fix them in the LABELLING project, never the training copy

```powershell
python -m wfield_local.dlc_project --label --folder cam4_2026-09-07T17_08_48
```

The training project is overwritten by the next `dlc_train` run — anything corrected there is lost.
`dlc_project` opens the real one. Then re-run `dlc_train`, which re-stages the corrected labels
forward automatically.

### The 2026-09-21 review list (23 frames, 4 sessions)

**1 to DELETE.** `cam4_2026-08-26T12_25_41/img1796416.png` — tongue at x=619 on a 680 px frame,
`p=0.012`. The crop is bare fur and whiskers: no mouth, no tongue. A misclick on the cheek. This one
frame produces the entire `subacute` train tongue RMSE of 53 px.

**3 whole licks to re-place**, each wrong on 3+ of its frames:

| lick | part | what it looks like |
|---|---|---|
| `cam4_2026-09-07T17_08_48#674384` | tongue | 8–33 px, `p=0.34–0.61`. Label sits on the tongue body/tip; the network puts it at the tongue–spout CONTACT. This is the landmark question, unresolved. |
| `cam4_2026-06-06T18_02_39#389083` | tongue | 22–26 px, `p=0.46–0.63`. The x-offset **flips sign** between consecutive frames of one protrusion — the label is not tracking one point. |
| `cam4_2026-09-07T17_08_48#1253168` | jaw | 5–14 px at `p=0.78–0.96`, offset **growing monotonically** through the lick (−4.5 → −6.8 → −10.2 → −13.1 px). The label follows the chin contour as the mouth opens instead of holding the landmark. Its `lick+0` frame is also the missing tongue below — re-do this lick whole. |

**3 to ADD** (`img1253168` tongue `p=0.73`, `img0811162` tongue `p=0.72`, `img0233849` jaw `p=0.65`).
`evaluate_network` scores only what was labelled, so a MISSED part is invisible to it — this is the one
category no error metric can surface.

**Also worth a spot-check:** `lick-16` carries a tongue label on **53%** of its frames, the same rate as
`lick+0`, although `dlc.frames.lick_offsets_s` designed `−16 ms` as the tongue-IN hard negative
("committed to the lick, tongue still in"). Either the tongue really is emerging by then — plausible,
the sensor fires on contact — or some `lick-16` frames carry a tongue that is not out. Two of the
flagged frames are `lick-16`. Worth deciding once, since it sets what the hard negative teaches.

### Second result (2026-09-25) — round 2, and the one new hazard

`snapshot_best-60`, `iteration-1`, trained on 264 frames from 11 sessions after the second labelling
round took the set from **240 to 360 frames** and corrected 84 points on frames that already existed.
Every part improved on the SAME three sessions round 1 was scored on:

| part | round 1 | round 2 (same 3) | | part | round 1 | round 2 (same 3) |
|---|---|---|---|---|---|---|
| `spout` | 2.19 px | **1.79** | | `jaw` | 4.70 px | **4.01** |
| `nose` | 4.17 px | **3.29** | | `tongue` | 10.27 px | **6.94** |

**`subacute` is tested for the first time** (`dlc.train.training_fraction` 0.8 -> 0.74 buys a fourth
held-out session; see the config comment for why 0.73 does not), and the per-epoch TEST table answers
the question the study turns on:

| part | acute | subacute | chronic | pre | | part | acute | subacute | chronic | pre |
|---|---|---|---|---|---|---|---|---|---|---|
| `jaw` | **2.65** | 3.75 | 4.53 | 8.90 | | `spout` | **1.57** | 2.30 | 1.71 | 2.80 |
| `nose` | 3.51 | 3.44 | 5.02 | **2.82** | | `tongue` | **4.97** | 10.62 | 12.68 | 9.86 |

**`acute` is the BEST epoch on three of four parts.** The false-deficit failure mode — a network that
tracks a healthy mouse well and a hemiparetic one badly — is absent, and this is the first round in
which `subacute` could be asked at all.

**THE TWO TABLES `dlc_train` PRINTS ARE IN DIFFERENT UNITS.** DLC's per-keypoint block reports MEAN
euclidean error despite calling it `rmse`; `per_epoch_error` reports true RMSE. The comparison table
above is in DLC's units because that is what the round-1 table was. Do not read one against the
other — on this run that alone is the difference between a `jaw` of 3.77 and one of 5.61.

**The tongue is still a TAIL, not a level.** Test RMSE 9.70 px, test MEDIAN 5.80, and 6.96 with the
worst five frames removed; over all 160 labelled tongues the median is 2.81 px and just 2 exceed
20 px.

#### The hazard that is new this round: the review list has moved onto the TEST set

Round 2's list is **1 `DELETE`, 15 `REPLACE`, 5 `DECIDE` — every one a held-out frame** — plus 21
`ADD` (15 train, 7 test). That is what a fitted network does: it agrees with the frames it learned,
so disagreement concentrates where it did not. The trap is what happens next.

**Correcting a TEST label toward a confident prediction and then re-scoring the same split measures
how well the labels were moved onto the network.** `REPLACE`'s justification — the network learned
the landmark from the other ~190 frames, so a confident disagreement means that frame is the odd one
out — is sound for a TRAIN frame and circular for a test one.

**Make the corrections anyway; move the measurement.** Before scoring round 3 either change
`dlc.train.seed` so those frames land in train, where the corrections are ordinary supervision, or
keep the split and report the number knowing what is in it. What must not happen is round 3's test
error being quoted as a clean generalisation estimate without either.

#### Still unresolved: the tongue landmark

`cam4_2026-06-06T18_02_39#389083` and `cam4_2026-09-07T17_08_48#674384` were flagged in round 1,
touched in round 2, and are flagged **again**. They are not drifting labels — they are a convention
nobody has settled. `DECIDE` is 5 frames and it is the only category retraining cannot clear. Two
whole licks in the newly held-out `subacute` session (`#346227`, `#526451`) join them.

### The next refinement round

DLC's loop, and `dlc_train` prints these on completion:

```python
import deeplabcut as d
cfg = ".../dlc/Widefield/training/widefield-Priya-2026-09-08-orofacial/config.yaml"
d.analyze_videos(cfg, [video]);  d.extract_outlier_frames(cfg, [video], outlieralgorithm="jump")
d.refine_labels(cfg)        # fix the network's own worst frames
d.merge_datasets(cfg)       # merges + BUMPS `iteration`
# then: python -m wfield_local.dlc_train      (re-stages at the new iteration and retrains)
```

`iteration` is the round counter, so each round's training set and model folder are separate and an
earlier network stays reproducible.

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
