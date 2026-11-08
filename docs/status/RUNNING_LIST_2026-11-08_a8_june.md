# Running list 2026-11-08 (a8, overnight): June pre-stroke sessions as a reference

Priya 11-07 ~midnight: "please continue to proceed through tasks, keep a running list of findings, and any questions or
decisions you need me for". Tasks (Priya "sure start tasks 1-4"): 1 = 3-D kinematics June reference; 2 = June trial
classes; 3 = FaceRhythm June face regions; 4 = FaceRhythm June O2 run. Background: `docs/POSE_MODELS.md` D13,
`docs/status/JUNE_COVERAGE_2026-11-07.md`.

## DECISIONS / QUESTIONS FOR PRIYA

1. ~~Submit the June FaceRhythm O2 bundle?~~ **SUBMITTED by Priya 11-08 ~00:30.** When done: transfer-node rsync of
   `out/` (bundle COMMANDS.md section 3), then `python scripts/o2_face_rhythm.py collect --name fr_cam4_june_20261108`,
   then project onto frozen `master_pre_v1` + readouts (a8). 11-08 ~00:40: June tasks pending on O2 with
   `AssocGrpCpuLimit` (group CPU cap); Priya HELD the remaining pooled-null tasks (`scontrol hold
   58716929_[1992-2000]`) so June runs first. **To do: `scontrol release 58716929_[1992-2000]` once the June tasks
   start** (the chronic null rescore waits on those tasks). Original item: `DeepLabCut/Widefield/o2/fr_cam4_june_20261108/` (12 tasks, same
   settings as production fr_cam4_20261021: 10-min segments, 4 workers, engaged trials; COMMANDS.md inside). ROIs are
   in place (`face_rhythm/cam4/rois/PS9x_2026060x/`, `rois_ready` True for all 12). Before submitting, please look at
   `face_rhythm/cam4/register/qc_june_rois.png` (June warped regions next to each animal's first August session).
2. **DECIDED (Priya 11-08): option 1 -- NaN the off-frame / folded rim points in the June readouts; PS95 June stays in.**
   Original item: **PS95 06/07 and 06/08 mesh points off the frame.** 4 (06/07) and 29 (06/08) of the 1282 mesh points land outside
   the 680 x 680 frame (top edge; PS92 06/08 has 24 within 8 px of the edge). Proposal: keep the mesh as is (point i
   must stay the same face location in every session) and exclude off-frame points from the June readouts (NaN), as
   frozen / violating points are already treated. Alternative: leave PS95 06/08 out of June FR.
3. **Whether June goes on the page / deck as a reference panel** (task 1 results below). Proposal: one HIDDEN
   supporting slide / one page panel ("June -> August learning changes are ~1/5 of the acute change; chronic
   endpoint / lateral residuals exceed them; jaw opening does not"). Nothing changed on the page or deck.

## FINDINGS (newest last)

- **June face alignment is as good as August's at the landmarks** (registration QC, `register/qc_landmarks.csv`):
  median error nose 2.8 vs 3.5 px, jaw 7.0 vs 5.0, spout tips 6.4 vs 7.6 (June vs August). The dense optical-flow step
  fell back to similarity + deepflow in June (the face is 1.17-1.24x larger), but landmark errors are in August's range.
- **June trial classes compute from existing inputs.** The shared `trial_classes` code classifies June on demand
  (PS92 06/06: 291 working-period trials, 283 contact / 1 incomplete / 4 jaw-only / 3 no-response); only the session
  LISTS of the snapshot / summary scripts skip June. So the 3-D kinematics gate (A20) already has June working periods.

- **June trial classes look like August's (task 2).** All 12 June sessions, with 8 August sessions (0806, 0810) for
  comparison:
  - Working period covers 92-100 % of video trials in June; August 0806/0810 ranges 69-100 % (PS94/95 0806 quit
    early).
  - Contact share of working-period trials: June 93-99 %, August 87-100 %.
  - Incomplete licks are somewhat rarer in June: PS93 17-22 vs 40-71 in August; others 1-10 vs 2-11.
  - So June has a normal working period, and the A20 gate keeps ~all June trials.
  - Table: scratchpad `june_classes_check.csv` (copied into the June reference folder with task 1's outputs).

- **Task 1: June vs own August pre, 3-D kinematics** (`session_poses/pose3d/june_reference/`: june_vs_august.csv,
  sessions_delta.csv, june_reference.png, june_classes_check.csv; 3 June + 7-8 August sessions per animal; within-animal
  session shuffles 10k; Holm over the 6 positions per measure; values below = mean over animals of the per-animal mean
  delta vs own August pre).
  - **June differs from August far less than acute does.** Median |June| / |acute| over measure x position = 0.19.
    At far_R (the deficit spout): closest approach June +0.13 vs acute +1.75 mm; 3-D angle error +2.4 vs +23.5 deg;
    endpoint side -0.20 vs -2.28 mm; lateral error -0.13 vs -1.71 mm.
  - **June-vs-August cells passing Holm (14 of 54 drawn):**
    - jaw opens LESS in June at close_L / close_R / far_R (-0.22 / -0.30 / -0.35 mm; z -1.4 to -2.4);
    - tongue FASTER in June at far_L / far_center (+10 / +13 mm/s);
    - close_L reaches further and lower (+0.29 / +0.25 mm);
    - far_R: angle error +2.4 deg (z +3.8) and endpoint side -0.20 mm;
    - closest approach close_R +0.09 / far_center -0.11 mm; lateral error far_center +0.11 mm.
  - **Read against the post epochs:**
    - post-stroke tongue SLOWING (acute -22 to -56 mm/s, chronic -4 to -23) runs opposite to June's direction, so
      it is not a return to the earlier training stage;
    - chronic endpoint-out / lateral residuals (out close_R +0.39, far_R +0.47 mm; lateral far_center +0.36 mm)
      are 3-5x June's (<= 0.13), so they exceed the learning-stage difference;
    - chronic jaw-opening deficits (-0.10 to -0.34 mm) are the SAME size as June's (-0.2 to -0.35), so a
      chronic jaw-opening change is within the training-stage range.
  - Peak-timing row (dt_peak_ms) kept in the table but not drawn (moves in whole 4 ms frames; medians ~all 0).
  - Script fix: the classes were built twice per session (licks + jaw trials; ~35 min serial, killed by my 30-min
    timeout on the first run); now precomputed once, 2 workers (~15 min); `--replot` redraws from the CSVs.

- **June FaceRhythm O2 outputs: complete and clean except PS95's mesh rim** (Priya copied out/ 11-08; `o2_face_rhythm.py
  collect` -> `face_rhythm/cam4/o2_fr_cam4_june_20261108_index.csv`; `qc --jobs 2` -> `o2_fr_cam4_june_20261108_qc.png`
  + data/ csv). Compared with each animal's August pre sessions of the production run:
  - integrity 12 / 12; no constraint violations, no NaN bins, trial samples 100 % filled;
  - interior mesh folds as rare as August (<= 0.13 % of bins; August <= 0.32 %);
  - displacement in August's range (median 1.1-2.2 px vs 1.1-2.4; p99 15-27 vs 15-25 px);
  - **PS95 June: a rim fold in every bin (August 80-92 %) and 3x August's p99 folded fraction (0.7 % vs 0.24 % of
    triangles).** It is the same place as the off-frame mesh points at the top edge (decision 2). PS92 / PS93 / PS94
    rims are at or below August.
  - So decision 2 (NaN the off-frame / rim points in the June readouts, or leave PS95 June out) now matters for PS95
    all three days, not just 06/08.

## DONE (newest last)

- Task 3: `scripts/face_rhythm_warp_rois.py to-sessions --june` (June only; the 107 existing ROIs untouched) ->
  12 June `rois/<s>/ROIs.h5`; `qc-june` -> `register/qc_june_rois.png`.
- Task 4: June bundle built; Priya submitted it on O2 (11-08 ~00:30).

## RUNNING / NEXT

- Task 2: DONE (see findings).
- Task 1: DONE (see findings). Task 4: outputs collected + QC'd (see findings); next = project onto frozen
  master_pre_v1 + June readouts, waits on decision 2 for PS95.
