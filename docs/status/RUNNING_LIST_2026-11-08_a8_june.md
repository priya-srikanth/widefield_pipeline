# Running list 2026-11-08 (a8, overnight): June pre-stroke sessions as a reference

Priya 11-07 ~midnight: "please continue to proceed through tasks, keep a running list of findings, and any questions or
decisions you need me for". Tasks (Priya "sure start tasks 1-4"): 1 = 3-D kinematics June reference; 2 = June trial
classes; 3 = FaceRhythm June face regions; 4 = FaceRhythm June O2 run. Background: `docs/POSE_MODELS.md` D13,
`docs/status/JUNE_COVERAGE_2026-11-07.md`.

## DECISIONS / QUESTIONS FOR PRIYA

0. **June FR must RE-RUN on O2 (11-08 morning).** Priya: "june face mesh fits are off - they look shifted image left (face
   right) and aren't scaled to be bigger with the face". CAUSE: the June LP landmarks in `face_rhythm/cam4/frames/` are
   run v36 coordinates (June pose x 0.8853 about (0, 0), the June camera-scale correction), not June image px; the
   landmark similarity read them as image px, so the June fields had no scale and were 50-77 px off (master nose / jaw /
   tips vs each June session's own un-rescaled LP). FIX: `scripts/face_rhythm_register.py --june-only` (face-masked
   image affine June -> the animal's August template, DeepFlow on top, landmarks / 0.8853; PS94 06/06 = fixed median
   scale + shift search). New June fields: face ECC vs master 0.63-0.72 -> 0.88-0.95 (August 0.87-0.95); nose 0.9-4.5
   px, jaw 1.3-12, tips 3-16 (`register/qc_june_fields.csv`); June face 1.10-1.14x wide, 1.04-1.12x tall (camera
   scale 1.129). June ROIs re-warped (`register/qc_june_rois.png`); NO mesh point off the frame any more (decision 2
   moot). Old fields -> `register/fields_june_superseded_20261108/`; old O2 outputs -> bundle
   `out_superseded_wrongmesh_20261108/` (+ its qc_cache); old QC -> `face_rhythm/cam4/superseded_june_wrongmesh_20261108/`.
   Re-run = move scratch out/ + work/ aside, rsync the new rois, resubmit 1-12 (commands given to Priya 11-08).
   August fields / ROIs / outputs untouched. 3-D (task 1) unaffected: it uses the v36 coordinates consistently.
   **Re-run 11-08 midday:** 9 / 12 copied back by ~13:00. Not yet copied: PS93 06/07 and 06/08, PS94 06/06; PS92 06/08
   trials were still copying. All 9 have violations <= 0.02 %, trial samples 100 % filled, and mesh home 54-61 px
   (median) from the old wrong mesh, all inside the frame. So the new meshes were used. Next = collect + qc once all 12
   are copied.

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
   MOOT after the June re-registration (no point off the frame); the rule stays for any that appear.
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

- **June FaceRhythm RE-RUN (correct meshes), 11-08 afternoon: clean, and the PS95 rim problem is gone.** All 12 copied
  back; `collect` 12 / 12; `qc --jobs 2` -> `face_rhythm/cam4/o2_fr_cam4_june_20261108_qc.png` + data/ csv; `timing`
  (dropped frames median 0.08 %, max 0.10 %). Compared with the wrong-mesh run and with each animal's August pre:
  - violations 0, no NaN bins, trial samples 100 % filled;
  - interior folds <= 0.4 % of bins (August pre <= 0.3 %);
  - p99 folded fraction 0.001-0.002 (wrong mesh: PS95 0.005-0.007; August pre <= 0.003);
  - rim folds: PS95 0.51-0.87 of bins (wrong mesh 1.00), PS92 0.55-0.72, PS93 0.27-0.42, PS94 0.14-0.24; all within
    August pre (up to 0.92);
  - **displacement is LARGER than August:** median 1.1-2.6 px vs August pre 1.0-2.4. By animal: PS92 2.2-2.3 vs
    1.2-1.5; PS95 2.3-2.6 vs 1.4-2.0. Up 1.2-1.35x from the wrong mesh, which sat partly off the face. The face-scale
    difference (~1.13x) explains only part of it (PS92 ~1.6x), so June face movement is larger in px even after
    scale. Readouts must be scale-normalized and interpreted as a training-stage difference, not tracking.
  - So decision 2 is moot for the tracking as well: no PS95 June exclusion is needed.
  Projection onto frozen `master_pre_v1`: `scripts/face_rhythm_june_project.py` (separate set folder
  `o2/fr_cam4_june_20261108/factors/master_pre_v1_june/`: byte copy of master.npz + a 12-task June plan; production set
  untouched) -> `master_pre_v1_june/proj/`. DONE (12 / 12, ~10 min each). Reconstruction error (relative, trial
  median): June 0.44-0.48, the same as acute (0.44-0.49); held-out August pre 0.42-0.46, chronic 0.43-0.46. So the
  frozen basis describes June about as well as acute: June faces are not fully inside the August-pre space.
- **June FR factor peaks vs own August pre** (`scripts/face_rhythm_june_reference.py --split side` ->
  `readouts/master_pre_v1/june_reference/`: june_peak_reference_side.png, data/june_peak_{sessions,stats}_side.csv;
  the 5d peak quantity, June peaks divided by face scale sqrt(sx sy), boot_delta 10k, Holm over 12 factors):
  - **Absolute amplitude: June is higher on almost every factor** (+0.3 to +0.6 log2; 26 / 36 factor x side cells
    pass). It is a GLOBAL gain, not factor-specific. The median over factors per animal is PS92 +0.43, PS93 +0.52,
    PS94 +0.20, PS95 +0.15 (chronic +0.16 to +0.39). It survives the face-scale division, so camera geometry alone
    does not explain it; lighting, contrast or focus could (the June camera setup differed), and so could more
    vigorous June movement. Absolute June amplitudes are therefore NOT a clean reference.
  - **Composition (scale-free: each factor's log2 peak minus the session's mean over the 12 factors, vs own pre)
    removes the gain.** June differs from August ~1/3 as much as acute does (median |June| / |acute| 0.32). The cells
    where all 4 animals move the same way:
    - run_onset_jaw LOWER (-0.22 to -0.40 in all three sides);
    - prep_jaw LOWER at contra / ipsi (-0.18 / -0.24);
    - sniff_prelick HIGHER at center / ipsi (+0.22 / +0.29);
    - lick_left higher at center (+0.25).
    The jaw factors converge with task 1 (3-D: the June jaw opens less).
  - **The contralesional lick deficit is NOT a training-stage difference:** lick_right composition June ~0 (-0.02 to
    +0.01) vs acute -0.49 to -0.83 and chronic -0.22 to -0.79. The chronic lick_fast_close and lick_ongoing_7Hz rises
    (+0.3 to +0.65) are also absent in June (-0.2 to +0.15).
  - The composition numbers are a scratch analysis (per-animal means + sign counts, no bootstrap yet). Superseded by
    the drift-floor run below.
- **June as a ~2-month DRIFT FLOOR (Priya 11-08: "only as drift maybe"; "ok" to the plan 11-09).**
  `scripts/face_rhythm_june_drift.py` -> `june_reference/`: june_drift_{composition,asymmetry}.png,
  data/june_drift_*.csv. Scale-free readouts only: factor composition, and the 5e contra / ipsi speed asymmetry (June
  sessions through the same `asym_aligned.job`; production from its cached traces). Per cell: change vs own August
  pre, then boot_delta(epoch, June). "Beyond floor" = that interval excludes 0 with the change's sign, and the change
  itself passes (Holm).
  - **Asymmetry: every contralesional-spout loss is beyond the June floor, acute AND chronic.** Cue and lick aligned,
    close_R and far_R (chronic lick far_R -0.14, cue far_R -0.10, close_R -0.08 / -0.09; June -0.06 to +0.02). The
    chronic GAIN at close_L (ipsilesional) is also beyond the floor (cue +0.03, lick +0.05). Bout-aligned changes are
    not beyond the floor (June bout far_R -0.08 is wide).
  - **Composition:**
    - lick_right loss is beyond the floor at center and contra spouts in all three post epochs (chronic -0.74 /
      -0.79; June ~0);
    - chronic rises are beyond the floor: lick_fast_close (+0.35 to +0.65, all sides) and lick_ongoing_7Hz at center
      (+0.43); also chronic nose_prelick_slow at center (-0.27);
    - acute jaw / prep rises (run_onset_jaw, prep_jaw, +0.3 to +0.57) are beyond the floor;
    - **chronic run_onset_jaw (-0.23 / -0.40) and ipsi sniff_prelick (+0.22) are NOT beyond the floor:** June moves
      the same way by the same amount (-0.22 to -0.40; +0.29). So these chronic changes are inside the training-stage
      range, matching the 3-D result that chronic jaw opening sits inside June's range.
  - Proposed slide (hidden, supporting; not made): "Chronic contralesional face changes exceed a 2-month pre-stroke
    drift floor; chronic jaw changes do not", with task 1's 3-D numbers in the same frame.
  - Optional next: the June-only cohort factor fit on O2 for the lick_right SHAPE angle vs June (not prepared yet).
- **11-09 (Priya "1. yes 2. yes"):**
  1. **Hidden deck slide 95 ADDED** (after the FR per-session drift slide 94; deck now 202 slides): "Chronic
     contralesional changes exceed a 2-month pre-stroke drift floor; jaw changes do not". Picture
     fig:june_drift_summary (`scripts/june_drift_summary_fig.py`; FR asymmetry + composition + 3-D in one figure, 3-D
     beyond-floor bootstrap in data/june_drift_3d.csv). Notes: 7-line Data block + numbers + source. Scripts beside
     the deck: `add_june_drift_slide_1109.py` (backup_before_june_drift_1109), then layout fixes
     `fix_june_drift_slide_1109b.py` (full-width bullet band) and `_1109c.py` (14 pt); hidden-state asserts passed.
     3-D chronic vs June: endpoint out close_R / far_R +0.39 / +0.47 mm and lateral error close_R / far_R beyond the
     floor (Holm); jaw opening close_L / far_R (-0.22 / -0.34 mm) = June's (-0.22 / -0.35).
  2. **June-only cohort factor fit READY for O2** (`scripts/face_rhythm_june_fit.py plan`): session_rows drawn with
     the production params and seed rule (60 trials per session, as August), code deployed, `run_fitjune.sbatch` (5
     tasks: COHORT + 4 LOAO, 16 CPU / 96 GB) and `run_fitsession8_june.sbatch` (12 per-session own fits, as production
     own8), commands in the set folder's COMMANDS_JUNE_FIT.md. SUBMITTED by Priya 11-09 (queued behind frf_null). When done: rsync own_epoch_cohort/ + own8/ back, then angle June fit vs production COHORT pre / preEarly (drift_floor matching).
- **11-09 Priya: "can we also do the frozen pre-stroke analysis on june sessions (using august frozen reference) - for
  fr and other analyses".** FR: done (frozen projection + peak / composition / asymmetry readouts; per-session own fits
  queued above). Other arms with an August-frozen reference: imaging position decoder (5r / 5ro frozen-vs-refit),
  behavioral-state decoder, frozen encoder (needs June encoding inputs on O2), no-lick decoder, CD trajectories.
  `june_august_drift.py` exists but runs the OTHER direction (trains on June, scores August). Scope question to Priya.
- **SUPERSEDED (wrong June meshes, item 0):** June FaceRhythm O2 outputs: complete and clean except PS95's mesh rim (Priya copied out/ 11-08; `o2_face_rhythm.py
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
