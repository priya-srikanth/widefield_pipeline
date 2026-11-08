# Running list 2026-11-08 (a8, overnight): June pre-stroke sessions as a reference

Priya 11-07 ~midnight: "please continue to proceed through tasks, keep a running list of findings, and any questions or
decisions you need me for". Tasks (Priya "sure start tasks 1-4"): 1 = 3-D kinematics June reference; 2 = June trial
classes; 3 = FaceRhythm June face regions; 4 = FaceRhythm June O2 run. Background: `docs/POSE_MODELS.md` D13,
`docs/status/JUNE_COVERAGE_2026-11-07.md`.

## DECISIONS / QUESTIONS FOR PRIYA

1. **Submit the June FaceRhythm O2 bundle?** `DeepLabCut/Widefield/o2/fr_cam4_june_20261108/` (12 tasks, same
   settings as production fr_cam4_20261021: 10-min segments, 4 workers, engaged trials; COMMANDS.md inside). ROIs are
   in place (`face_rhythm/cam4/rois/PS9x_2026060x/`, `rois_ready` True for all 12). Before submitting, please look at
   `face_rhythm/cam4/register/qc_june_rois.png` (June warped regions next to each animal's first August session).
2. **PS95 06/07 and 06/08 mesh points off the frame.** 4 (06/07) and 29 (06/08) of the 1282 mesh points land outside
   the 680 x 680 frame (top edge; PS92 06/08 has 24 within 8 px of the edge). Proposal: keep the mesh as is (point i
   must stay the same face location in every session) and exclude off-frame points from the June readouts (NaN), as
   frozen / violating points are already treated. Alternative: leave PS95 06/08 out of June FR.
3. (pending results) whether June goes on the page / deck as a reference panel.

## FINDINGS (newest last)

- **June face alignment is as good as August's at the landmarks** (registration QC, `register/qc_landmarks.csv`):
  median error nose 2.8 vs 3.5 px, jaw 7.0 vs 5.0, spout tips 6.4 vs 7.6 (June vs August). The dense optical-flow step
  fell back to similarity + deepflow in June (the face is 1.17-1.24x larger), but landmark errors are in August's range.
- **June trial classes compute from existing inputs.** The shared `trial_classes` code classifies June on demand
  (PS92 06/06: 291 working-period trials, 283 contact / 1 incomplete / 4 jaw-only / 3 no-response); only the session
  LISTS of the snapshot / summary scripts skip June. So the 3-D kinematics gate (A20) already has June working periods.

## DONE (newest last)

- Task 3: `scripts/face_rhythm_warp_rois.py to-sessions --june` (June only; the 107 existing ROIs untouched) ->
  12 June `rois/<s>/ROIs.h5`; `qc-june` -> `register/qc_june_rois.png`.
- Task 4: June bundle built (not submitted; O2 submission is Priya's).

## RUNNING / NEXT

- Task 2: class check on all 12 June sessions + 8 August sessions for comparison (running).
- Task 1: `scripts/pose3d_june_reference.py` (June vs own August pre, within-animal shuffles, 10k; post epochs drawn
  alongside) running.
