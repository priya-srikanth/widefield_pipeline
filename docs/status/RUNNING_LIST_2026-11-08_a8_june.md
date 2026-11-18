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
     11-09: own8 (12 per-session June fits) DONE + copied (EVR own .82-.83 vs frozen .77-.79). frf_june (cohort) still queued.
     Per-session floor test: scripts/face_rhythm_june_session_shape.py (June scored vs own Aug pre like the main test;
     epoch vs June within-animal shuffle; -> readouts/master_pre_v1/june_reference/session_shape_june/). 200-perm smoke
     run: lick_right spatial angle acute +14.6 / subacute +7.6 beyond June (+3.8), chronic +5.5 NOT; side_spec acute /
     subacute beyond, chronic not; lick_left angles ~ June (June +8.9 itself). 
     10k RESULT (data/epoch_vs_june.csv, june_vs_pre.csv, june_session_scalars.png): lick pair -- NOTHING survives Holm
     over the 24-test family vs June (uncorrected # only). lick_right: spatial angle acute p.003 / subacute p.029
     beyond June (+3.8, n.s. vs pre), chronic +5.5 p.21 not; spectral acute only (p.007); side_spec acute / subacute
     beyond (p.007 / .014), chronic -0.134 vs June -0.119 (June itself Holm-sig vs pre, p.001) -> not. lick_left: June
     already +8.9 deg spatial / +9.8 spectral vs Aug pre (Holm-sig): no spatial angle beyond June; spectral acute /
     subacute #; side_spec acute / subacute #; chronic LOSS 30/43 vs June 12/12 # (p.010). June vs pre also Holm-sig
     spatial angle for lick_ongoing_7Hz, slow_jaw, lick_fast_close. READ: acute / subacute lick-pair shape changes exceed
     2-month drift (uncorrected); chronic lick-pair shape residuals are within it, except chronic lick_left loss.
     Same picture as the 3-D / composition floor for jaw. NOT yet reported to Priya -> no deck / page change.
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

## 11-09 late: June COHORT shape floor (frf_june done, 5 fits on the share)
`scripts/face_rhythm_june_cohort_shape.py` -> `readouts/master_pre_v1/june_reference/cohort_shape_june/`
(june_cohort_shape.png, data/june_cohort_shape.csv). June fit (COHORT + 4 LOAO) angled vs production preEarly / pre with
drift_floor.py's v2 matching; DESCRIPTIVE (pooled fits). lick_right, COHORT (LOAO range):
- spatial: noise 3.0, Aug drift 5.2, JUNE vs early Aug 15.6 (17.6-27.6); acute 25.0, subacute 22.0, chronic 26.2.
  Epoch > its own June floor in 4/5 fits acute and chronic, 3/5 subacute; the June jackknife spread overlaps the epochs.
- spectral: June 33.9 (22-42) is LARGER than every epoch (19.6 / 20.7 / 13.3): 0/5 fits beyond June.
- So the lick_right spatial change sits modestly above a 2-month pre-stroke floor (not clearly), and the spectral change
  is within it. Fits with the per-session result (acute / subacute beyond June, chronic not). Not yet on deck / page.

- 11-09 late: Priya: June floor figures go on the FR page as a collapsible supplement only (no deck). PUBLISHED FR page v46 (supplement to Fig 7a.1, id #june-floor, both June figures + a status row). Source: session scratchpad fr_page/ (copy at M:/MICROSCOPE/Priya/Widefield/fr_page_src/fr_page/index_v46.html). FINDINGS 5 + FACE_RHYTHM_FINDINGS 6e-June updated.

## 11-12: FR pooled null, all 500 permutations (frf_null500 complete, 3000 / 3000 files on the share)
Priya's transfer-node loop copied own_epoch_cohort_null/ back (count + pre/post pairing checked locally). Scored with
`python scripts/face_rhythm_cohort_null.py` (rule v2, defaults) -> readouts/master_pre_v1/epoch_shape_cohort_v2/
(cohort_null_*.png + data/cohort_null.csv OVERWRITTEN; the previous outputs, acute ~500 / subacute + chronic 100, are in
epoch_shape_cohort_v2/n100_backup_20261107/). Acute numbers unchanged (were already ~500).
- LICK PAIR holds and tightens: lick_right Holm < .05 on all four measures at all three epochs (angle / asym / side_spec,
  p .002-.010); lick_left passes at subacute + chronic (not acute, p .27-.39), side_spec subacute .028.
- SPECIFICITY OF THE SHAPE CHANGE NO LONGER HOLDS for angles: of the other ten factors, angle_spatial Holm < .05 for
  4 at subacute (lick_ongoing_7Hz, lick_fast_close, whisk_run, lick_slow_late) and 4 at chronic (prelick_nose,
  lick_fast_close, whisk_run, lick_slow_late at Holm .050 exactly), 3 acute as before (prelick_nose, whisk_run,
  lick_slow_late); angle_spectral whisk_run at every epoch. These had sat at the 100-perm floor (Holm .07-.10).
- What STAYS specific: the LATERALIZED measures. asym: no other factor passes Holm in any epoch (lick_right passes all
  three, lick_left chronic). side_spec is defined for the lick pair only.
- Proposed reading (NOT yet in FINDINGS 5.10 / page, waiting on Priya): many factors change shape (within-animal null
  includes time drift; see June floor 6e-June), but only the lick factors become less lateralized / less side-specific.
- NaN rows: lick_ongoing_7Hz and lick_fast_close acute (master factor not recovered in the acute fit), as before.
- 11-12 (Priya asked): lick_left MATCH STABILITY in the pooled null, rule v2 (spatial best; ties within .05 of the best
  spatial cos -> best spectral; recovered at spatial >= .7, spectral >= .5; columns may repeat). Acute shuffled fits
  (500 pre / 500 post): runner-up within .05 in 67 / 120, tie-break changed the pick in 40 / 61, lick_left shares its
  column with another lick unit (lick_ongoing_7Hz, lick_slow_late, lick_fast_close) in 19 / 34, spatial cos < .8 in
  25 / 36, not recovered 0 / 3. Subacute + chronic: 0 on every count (cos median .99). The REAL acute cohort fit is
  itself near-tied (margin .026, cos .86; pre / subacute / chronic margins .19 / .18 / .34). So the wide acute null is
  partly matching ambiguity; the chronic pooled lick_left asymmetry (-.14, MORE lateralized) is not a matching effect.
  Pooled-vs-per-session direction disagreement for chronic lick_left remains (per session +.10, 3 / 4 own fits +.27).
- 11-12 (Priya: gut-check the lick_left match): `scripts/face_rhythm_lick_left_match_check.py` ->
  readouts/master_pre_v1/lick_left_match_check/ (pooled_lick_left.png, session_mean_lick_left.png,
  sessions_lick_left_<animal>.png). POOLED: pre / subacute / chronic pick a compact ipsilesional mouth-corner blob like
  the master (cos .94-.99, margins .07-.35); every ACUTE pooled fit (COHORT + 4 LOAO, and LOAO-PS92 subacute) picks a
  diffuse column (cos .71-.88, margin <= .05) whose runner-up is just as diffuse -> acute pooled lick_left has no clean
  match. Chronic pooled column is as compact as pre or more (asym -.83) -> the pooled "more lateralized" is real for
  that fit. PER SESSION: pre fits mostly clean; post fits often flagged (margin < .05: PS94 acute 5/5, chronic 5/8,
  PS95 subacute 4/5), the pick and runner-up are similar (mean asym -.39 vs -.42; only 4 / 41 flagged picks share a
  column with another lick unit). Post-stroke matched columns keep the ipsilesional blob, weaker, plus a contralesional
  cheek spot and lower lip (also in unflagged chronic sessions, e.g. PS94 0910 / 1001 / 1005). Asym change vs own pre,
  UNFLAGGED sessions only: chronic PS92 +.09, PS93 -.04, PS94 +.22, PS95 .00 (all recovered: +.11 / -.06 / +.29 /
  +.06); acute has 0-1 clean session per animal. So the per-session contralesional shift is real in PS94, small
  elsewhere, and partly carried by sessions where lick_left is split over two similar diffuse columns.

## 11-13: lick_left asymmetry downgraded (Priya "ok")
- FINDINGS headline + 5.10 and FACE_RHYTHM_FINDINGS 6e: the per-session lick_left contralesional shift is now
  PROVISIONAL (mainly PS94; clean-match chronic change +.09 / -.04 / +.22 / .00). Robust lick_left result = loss of side
  specificity + shape change (diffuse / split after the stroke). Acute pooled lick_left has no clean match.
- FR page: headline row, overview row, 7a.1, 7.10, 7.13 reworded; new Figs 7a.10 (session_mean_lick_left) and 7a.11
  (PS94 session sheet). Published with the 500-permutation edits (v47).

## 11-16: lick_left per-session shift by animal x epoch (correction to the 11-13 downgrade)
Asym change vs own pre (per-session, recovered sessions): acute +.26 / +.13 / +.27 / +.25 (4 / 4), subacute
+.02 / +.20 / +.24 / +.25 (3 / 4), chronic +.11 / -.06 / +.29 / +.06 (PS92-95). Taking the RUNNER-UP column instead
gives the same or a larger shift (acute +.15 / -.01 / +.29 / +.22; subacute +.46 / -.02 / +.40 / +.34; chronic +.27 /
+.16 / +.34 / +.49), so the near-ties do not create the shift. Clean-only sets are 0-2 sessions acute / subacute (not
informative). The 11-13 "clear in PS94 only" was based on chronic alone and overstated. Proposed to Priya: acute +
subacute STANDS; chronic small (PROVISIONAL). Awaiting her ok before re-editing FINDINGS / page / deck.

## 11-16: flagged lick_left sessions, pick vs runner-up (Priya: "is the runner up a better match?")
New figures flagged_lick_left_<animal>.png + data/flagged_lick_left.csv (match-check script). Flags: near-tie PS92 12,
PS93 8, PS94 13, PS95 3; low cosine (.70-.80, still counted as recovered) ONLY in PS95 (5); not recovered PS93 11, PS95 4.
Runner-up has a higher spatial cosine in 22 / 58 flagged sessions (frequency tie-break chose the pick); its asymmetry is
about the same. PS95: no good column at all in most flagged sessions (pick .72-.78, runner-up .56-.76, both diffuse);
0818 subacute: the spatially best column (cos .86) looks like the master but fails on frequency (.31). Recovery at
spatial cos >= .8 instead of .7: PS92-94 unchanged; PS95 loses acute (1 -> 0 sessions) and subacute (5 -> 1). Asym change
at >= .8: acute +.26 / +.13 / +.27 / - ; subacute +.02 / +.20 / +.24 / +.17 (1 session); chronic +.11 / -.06 / +.27 / .00.

## 11-16: own-fit lick columns, trial-aligned expression by spout (Priya: matched vs runner-up on lick-left trials?)
`scripts/face_rhythm_own_temporal.py` (new): each session's trial spectrograms projected on all 12 own8 columns (unit-norm
A / B, absolute = weight x D, 12 trials per spout; cache ~/wfield_cache/fr_own_temporal/) -> own_temporal_lick.png +
per animal + data/own_temporal_lick.csv in lick_left_match_check/. Response window (cue 0..+2 s) ipsi / contra spout
ratio, mean of animals: lick_left MATCHED 4.7 pre -> 2.5 / 2.3 / 2.7 (ipsi trials fall a little, contra trials rise);
RUNNER-UP 1.3 / 1.8 / 1.4 / 1.3 (not side-selective). lick_right matched contra/ipsi 5.7 pre, stays selective; its
runner-up ~1. Post-stroke FLAGGED sessions: matched column more left-selective than the runner-up in 27 / 35 (median
ratio 2.3 vs 1.3); PS92 10 / 10, PS93 5 / 5, PS94 8 / 13 (2.2 vs 2.3), PS95 4 / 7. PS95 subacute matched lick_left is
strongly expressed on left-spout trials (Priya: lick_left is there behaviorally; my "not really there" was wrong -- the
footprint changed, the factor did not vanish). PS93 0822 / 0824: matched column more left-selective (1.42 vs 1.07; 1.31
vs .35) -> the tie-break picked the right column in both.

## 11-16 (later): own-fit lick columns -- SEM bands, cohort palette, by-spout layout, stats (Priya)
`face_rhythm_own_temporal.py` now: spout colors = `spout_behavior.position_style`; band = SEM across animals;
own_temporal_lick[_by_position][_<animal>].png + own_temporal_lick_stats.png (response window cue 0..+2 s, log2 vs own
pre, boot_delta, Holm over 18 per row) + data/own_temporal_lick_stats.csv. Preview from 12 trials / spout (suffix _n12);
ALL-TRIALS cache running (~3.5 h, 50,929 trials, network-bound), then re-render without suffix.
Preview stats (matched columns): lick_right falls on far contra trials at every epoch (-2.61** / -.86** / -.69**) and
far mid (-1.22** / -.47** / -.54*), close contra unchanged; lick_left RISES on close contra acutely (+1.31**) and far
contra subacutely (+.69*), falls on close ipsi chronically (-.50**) and far mid acutely (-.76**). Runner-ups: one Holm
pass (lick_right runner-up close contra subacute -.55**).

## 11-16: lick_left asymmetry wording applied (Priya "sure")
STANDS acute (4 / 4) + subacute (3 / 4), chronic small / PROVISIONAL (mainly PS94): FINDINGS headline + 5.10,
FACE_RHYTHM_FINDINGS 6e + change log; PI deck slide 90 (new bullet + notes line; backup
Overview_2026-11-06.bak_pre_a8_lickleft_20261116.pptx); FR page text edited LOCALLY (headline row, overview row, 7a.1,
7a.10, 7.13), published together with the own_temporal figures after the all-trials re-render.
Stale, not mine to edit without asking: deck slide 163 (Big picture) status line still says "lick_left subacute + chronic
(cohort permutation null, 100 permutations per epoch, 10-26)".
- 11-16: deck slide 163 notes status clause updated (Priya "yes ok"; backup Overview_2026-11-06.bak_pre_a8_s163_20261116.pptx).
- 11-16: FR page v48 published: lick_left wording (STANDS acute + subacute), new section-7a block "Does the matched factor
  fire on its own spout's trials?" -- Figs 7a.12 own_temporal_lick (key), 7a.13 stats, 7a.14 by_position_norm (pre =
  leave-one-out, SEM across pre sessions), supporting 7a.15-7a.27 (absolute by spout, per animal, flagged pick vs
  runner-up, session sheets, pooled matches); status row added. Figures are the 12-trials-per-spout preview: SWAP for the
  all-trials render (same ids / file names) and update the 7a.13 numbers + status row when bkwv2ffli finishes.
- 11-16: BLOCK-PERMUTATION audit (window a2; Priya: block permutation everywhere it makes sense). FR arm: only
  scripts/fr_enl_decode.py is affected (FR page Fig 6.1, PROVISIONAL): its per-session null permutes position labels
  trial-wise against the block-CV predictions -> HOLD claims until a2's shared block helper lands, then re-run with block
  permutation. All other FR nulls shuffle session labels within animal (unaffected); own_temporal stats = bootstrap.
- 11-16: a2 audit done (branch analysis/tim-encdesign-1116, docs/status/BLOCK_PERMUTATION_AUDIT_2026-11-16.md, a7ec0482).
  Class A in the FR / ENL arm (trial-wise spout-label null vs block-CV predictions -> block permutation): fr_enl_decode.py:118,
  enl_fr_removed_decode.py:119, enl_movement_multisession.py:253, enl_video_spout_check.py:142, movement_position_angle.py:79
  (+98). HOLD new p values on these until a2's trial-wise vs block-wise measurement + shared analysis_kit.block_permute.
  Session-label shuffles fine. Trend caveat (two session series both trending over days): the only FR association test
  is session_shape_tests lick_right map change vs right-lick share (Spearman, parametric p, within epoch; rho -.14 n.s.) --
  a shared trend would inflate, not hide, an association, so the null result stands; detrend if it is ever re-tested.
- 11-16 ~15:10: all-trials run STOPPED at 64 / 108 (window 3c: box at 2.3 GB free; my 6 workers held ~42 GB of memory-mapped
  spectrogram pages). Fix in session_curves: plain per-trial file reads (verified byte-identical to the memmap); measured
  0.42 GB working set per worker (was 8-10 GB). Resumed with --jobs 4 (completed sessions kept in the cache).

### 11-16 ~16:50 -- all-trials lick timing done; FR page v49
All-trials cache (108 sessions, 0 errors) + render (`face_rhythm_own_temporal.py --per-pos 0`). Numbers move little vs the
12-trial preview; three marks changed: lick_left far contra subacute +.78 (* -> **), lick_left close ipsi subacute -.32
(none -> *), lick_right runner-up far contra acute (* -> none). Matched lick_right far contra -2.54** / -.84** / -.58**;
lick_left close contra acute +1.36**. Left / right spout response ratio (both spouts per side, sessions pooled per epoch):
matched lick_left 5.0 pre, 2.6 / 2.3 / 3.1 post; runner-up 1.2-1.7 (the 12-trial "4.7, 2.3-2.7" came from the first
preview version). Flagged post sessions where the matched column is more left-selective: 25 / 35 (was 27 / 35; PS92 10/10,
PS93 4/5, PS94 7/13, PS95 4/7). Conclusion unchanged. Page v49 (1794866489-396c): figures 7a.12-7a.20 swapped to all
trials, captions / take line / status row rewritten (no 12-trial wording left). FINDINGS + FACE_RHYTHM_FINDINGS numbers
updated.

### 11-16 ~17:30 -- per-trial / per-lick readouts (Priya: "start per-trial analyses"; far-contra lick_left question)
New `scripts/face_rhythm_per_trial.py` (frozen master `proj/`, LP v36 licks; session-level animal bootstrap, Holm over
spouts) -> readouts/master_pre_v1/per_trial/ (data/per_trial_{trials,licks}.pkl, _sessions.csv, _stats.csv). RESULTS
(NOT on the page; to Priya first):
- Far-contra leftward licks are real: trials with >= 1 leftward lick pre .01-.09 -> acute .22-.67 (4/4), subacute
  .05-.81, chronic .02-.59 (PS94 persists).
- FROZEN master lick_left on far-contra trials DOES rise acutely: trial mean 0-2 s +.73** log2 (4/4 up), subacute
  +1.16**, chronic +.81**. Per leftward lick no change (-.11 acute, n.s.) -> the rise = more leftward licks, not bigger.
  So the flat acute far-contra cell in own-fit Fig 7a.13 (+.01) is a property of the per-session own-fit matched column,
  not of the behavior or the master factor. Peak of the TRIAL AVERAGE also hides it (+.21 n.s. acute) while the median
  PER-TRIAL peak shows it (+.72**): leftward licks on ~half the far-contra trials, late and scattered (per-trial peak
  time +1.16 s).
- Isolated licks: isolated leftward licks express ~30 % less lick_left than in-bout ones acutely (ratio .59-.71, 4/4;
  pre .67-.96); acutely 15 % of far-contra leftward licks are isolated (pre 2 %). Partial contribution, not the cause.
- Window 0-4 s vs 0-2 s: same conclusions (far_R acute +.94** vs +.73**).
- Per-trial vs trial-average amplitude: same direction everywhere; acute losses smaller per trial (lick_right far_R
  -1.83** vs -2.43**). CAVEAT: per-trial peak of a flat trial = max of noise (positive floor), which compresses losses;
  per-trial timing (COM / peak time) of near-flat trials drifts to the window middle, so acute "later" lick_right
  per-trial timing (+.47 s) is confounded by the amplitude loss. NEXT: condition on movement (trials with a lick of
  the unit's direction): prevalence x amplitude-given-lick, timing only on those trials.
Fig 7.3: per-epoch shading removed, pooled squares + stars centred on the epoch (Priya 11-16); page v50. Same shading
removed in code from epoch_shape_angles (page 7.11 / 8.18) -- takes effect at their next render.

### 11-16 ~18:00 -- movement-conditioned per-trial readouts; FR page renumbered (v51)
Conditioned on movement (trials with >= 1 lick in the unit's direction; prevalence = fraction of such trials):
- lick_right at far contra: prevalence -.87** / -.36** / -.07 (fraction); amplitude GIVEN a right lick -.83** / -.58** /
  -.61** log2 (4/4 below every epoch) -> the chronic deficit is per-movement amplitude, not prevalence. Timing: first
  right lick later +1.21 s acute (+.41 subacute, +.08 chronic n.s.); lick_right peak relative to the first right lick
  unchanged (-.25 n.s. acute) -> acute "later" expression follows later licking (reaction time), not re-timing.
  Chronic: peak +.24 s* / COM +.32 s** with first lick on time (peak - first lick +.12 n.s.) -- weak.
- lick_left at far contra: prevalence +.38** / +.27** / +.12; amplitude given a left lick unchanged (-.06 / +.24 / +.11
  n.s.) -> the rise is MORE trials with leftward licks, not bigger ones. On far-contra trials WITHOUT a leftward lick
  lick_left still rises subacute / chronic (+.77** / +.60**), and at close contra (leftward licks ~absent) it rises on
  no-left-lick trials +.65* / +.81** / +.98** -> contralesional-target licks made with the lick_left face pattern
  (fits FINDINGS 5d "ipsilesional lick pattern used for contralesional targets").
- Caveat: no-lick trials also fall acutely for lick_right (-.79**): less face movement overall acutely.
FR page v51: sections renumbered in page order 7a->7, 7b->8, 7->9, 8->10, 9->11 (Priya: "7a, 7, and 7 is confusing");
checked: 0 link mismatches, titles = data-num, no non-figure text changed. Doc pointers updated. PI deck notes on slides
87 / 89 / 93 / 94 still cite the OLD numbers (8.10, 7.1, 7.2, 7.12 -> now 10.10, 9.1, 9.2, 9.12; #f- ids still valid):
ASK Priya before editing the deck.

### 11-16 ~18:30 -- per-trial block on the page (v52); deck notes renumbered (Priya "yes add to page" / "and update deck")
FR page v52: section 2 gains "Per trial: how often vs how large" (Fig 2.5 = per_trial/per_trial_conditioned.png) + a
status row. Figure fixes before publishing: factor expression floored at 1 before log2 (a near-zero session gave -20),
and a cell is tested only with a pre baseline in >= 3 animals (lick_left given a leftward lick at close contra: 2 / 4,
drawn as "not tested", out of Holm). Key numbers unchanged by either fix.
PI deck (backup Overview_2026-11-06.bak_pre_a8_renumber_20261116.pptx): FR page numbers in notes of slides 87 (8.10 ->
10.10), 89 (7.1 -> 9.1; Figs 7.11-7.15 -> 9.11-9.15), 90 (section 7a -> 7), 93 (7.2 -> 9.2; 7.15 -> 9.15), 94 (7.12 ->
9.12, x2). Each #f- id checked: v50 number = old note number, v52 number = new. No slide text touched.

### 11-16 late -- to-do pass (Priya "go ahead with the to-do's"; deck slide yes; delete superseded n12 yes; hold ok)
- Page v53 / v54: Fig 2.6 (amplitude three ways + isolated licks), Fig 5.5 (per-trial lick_right timing; section 5 take +
  status row rewritten: acute later expression follows later first right lick; chronic close contra peaks .50 s later
  relative to it, *), Fig 6.2 (spout-in-mesh control; section 6 note sentence replaced), Fig 7.13 caption sentence
  (own-fit vs frozen far-contra lick_left), Figs 9.11 / 10.18 redrawn without epoch shading (v1 tables, numbers same).
- Deck: new slide 81 (per-trial; backup .bak_pre_a8_pertrial_slide_20261116.pptx). ALL SLIDES AFTER 80 MOVED +1 (206).
- FINDINGS FR item 6 + FACE_RHYTHM_FINDINGS 5g: timing claim refined (per-trial split, chronic re-timing PROVISIONAL).
- DELETED (superseded derived, Priya "ok to delete if superseded"), 118 files:
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS92_20260806_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS92_20260807_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS92_20260809_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS92_20260810_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS92_20260811_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS92_20260812_cam4_v1_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS92_20260812_cam4_v2_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS92_20260813_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS92_20260814_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS92_20260818_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS92_20260819_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS92_20260820_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS92_20260821_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS92_20260822_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS92_20260824_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS92_20260826_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS92_20260828_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS92_20260901_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS92_20260904_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS92_20260908_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS92_20260911_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS92_20260915_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS92_20260918_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS92_20260922_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS92_20260925_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS92_20260929_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS92_20261002_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS92_20261006_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS93_20260806_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS93_20260807_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS93_20260809_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS93_20260810_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS93_20260811_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS93_20260812_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS93_20260814_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS93_20260818_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS93_20260819_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS93_20260820_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS93_20260821_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS93_20260822_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS93_20260824_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS93_20260826_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS93_20260828_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS93_20260901_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS93_20260904_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS93_20260908_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS93_20260911_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS93_20260915_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS93_20260918_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS93_20260922_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS93_20260925_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS93_20260929_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS93_20261002_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS93_20261006_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS94_20260806_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS94_20260807_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS94_20260809_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS94_20260810_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS94_20260811_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS94_20260812_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS94_20260813_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS94_20260814_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS94_20260817_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS94_20260818_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS94_20260819_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS94_20260820_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS94_20260821_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS94_20260823_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS94_20260825_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS94_20260827_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS94_20260831_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS94_20260903_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS94_20260907_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS94_20260910_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS94_20260914_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS94_20260917_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS94_20260921_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS94_20260924_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS94_20260928_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS94_20261001_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS94_20261005_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS95_20260806_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS95_20260807_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS95_20260809_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS95_20260810_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS95_20260811_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS95_20260812_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS95_20260813_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS95_20260814_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS95_20260817_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS95_20260818_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS95_20260819_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS95_20260820_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS95_20260821_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS95_20260823_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS95_20260825_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS95_20260827_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS95_20260831_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS95_20260903_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS95_20260907_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS95_20260910_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS95_20260914_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS95_20260917_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS95_20260921_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS95_20260924_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS95_20260928_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS95_20261001_cam4_n12.npz
    C:/Users/SabatiniLab/wfield_cache/fr_own_temporal/PS95_20261005_cam4_n12.npz
    M:/MICROSCOPE/Priya/DeepLabCut/Widefield/face_rhythm/cam4/readouts/master_pre_v1/lick_left_match_check/data/own_temporal_lick_n12.csv
    M:/MICROSCOPE/Priya/DeepLabCut/Widefield/face_rhythm/cam4/readouts/master_pre_v1/lick_left_match_check/data/own_temporal_lick_stats_n12.csv
    M:/MICROSCOPE/Priya/DeepLabCut/Widefield/face_rhythm/cam4/readouts/master_pre_v1/lick_left_match_check/own_temporal_lick_PS92_n12.png
    M:/MICROSCOPE/Priya/DeepLabCut/Widefield/face_rhythm/cam4/readouts/master_pre_v1/lick_left_match_check/own_temporal_lick_PS93_n12.png
    M:/MICROSCOPE/Priya/DeepLabCut/Widefield/face_rhythm/cam4/readouts/master_pre_v1/lick_left_match_check/own_temporal_lick_PS94_n12.png
    M:/MICROSCOPE/Priya/DeepLabCut/Widefield/face_rhythm/cam4/readouts/master_pre_v1/lick_left_match_check/own_temporal_lick_PS95_n12.png
    M:/MICROSCOPE/Priya/DeepLabCut/Widefield/face_rhythm/cam4/readouts/master_pre_v1/lick_left_match_check/own_temporal_lick_by_position_n12.png
    M:/MICROSCOPE/Priya/DeepLabCut/Widefield/face_rhythm/cam4/readouts/master_pre_v1/lick_left_match_check/own_temporal_lick_by_position_norm_n12.png
    M:/MICROSCOPE/Priya/DeepLabCut/Widefield/face_rhythm/cam4/readouts/master_pre_v1/lick_left_match_check/own_temporal_lick_n12.png
    M:/MICROSCOPE/Priya/DeepLabCut/Widefield/face_rhythm/cam4/readouts/master_pre_v1/lick_left_match_check/own_temporal_lick_stats_n12.png

### 11-16 late -- Priya on open items; deck "Data:" pass
- June vs August-frozen: "just for drift sub-analyses" -> DECISIONS 2026-11-16. il.OUT item = the 11-08 render-queue
  audit, dropped from the FR list. "Data:" pass: Priya "yes go ahead" -> 21 slides written (8 CONFIRMED provenance rows +
  13 FR slides), 32 / 206 now carry the block; INFERRED / UNKNOWN rows not written (~150 slides wait). Record
  docs/status/DECK_DATA_BLOCK_PASS_2026-11-16.csv; slide 4 conflict (confirmed row vs notes on the trial gate) left for
  the behavior arm.
- 11-16 late, Priya "confirm provenance": re-ran 34's provenance scripts on today's render logs ->
  FIGURE_PROVENANCE_20261116_rerun.csv (+ _RERUN.md): CONFIRMED 45 -> 257 (deck 8 -> 38); Data: blocks on 26 more slides
  (deck 58 / 206). Remaining deck rows need per-figure work (no script named 48, older kept runs 47, missing fields 20,
  steps not done ~30).
- 11-16 late, Priya "yes do the per figure sweep" + "and update re-rendered figures": 6 read-only helpers, one per deck
  slide group (126 picture slides without a Data: block; inputs + outputs in a8 scratchpad sweep/group*.json ->
  out_group*.json): 7 fields + level LOG (render log) / CODE (code + output file, no log) / UNRESOLVED, plus the current
  render path vs the deck image source (stale yes / no) and crop. Next: spot-check, write blocks (CODE labelled as such),
  swap stale images (as-is ones directly; crops reproduced), record.
- 11-16 night, per-figure sweep DONE (6 helpers + 1 staleness check; outputs docs/status/provenance_sweep_20261116/):
  126 swept slides = 39 traced to a render log, 76 to code + output file, 11 unresolved. Spot-checked 3 claims against the
  logs (slide 13 decoder .37->.14 etc., slide 21 chronic Holm .057, slide 141 chronic -.288 / -.269): all exact.
  Data: blocks on 38 more slides (image not stale; Correction line tagged [traced: render log] or [traced: code + output
  file, no run log]); deck 96 / 206 with the block (DECK_DATA_BLOCK_PASS3_2026-11-16.csv; backup _datablock3).
  STALE IMAGES: 109 slides (DECK_STALE_FIGURES_2026-11-16.csv): A 21 swap + recheck numbers, B 53 numbers / text change,
  C 35 no rebuilt version. NO image swapped; per-slide approval asked from Priya. 3c: not editing the PI deck; holds
  slide 13 picture, balance 135-140 / 203 (C6), state-space SVD / LocaNMF slides (pre-glue inputs).

### 11-17 — slide 21 (frozen position decoder) normalization check (Priya: "per-animal normalization or raw?")
- Rebuild test (g7_fig_state_position_r.log, 480accea) = boot_delta PAIRED vs each animal's own pre, on the
  chance-normalized scale retained = (bal_acc - chance)/(1 - chance); an absolute difference, not divided by own pre.
- Relative to own pre (post / own-pre mean - 1, then boot_delta + Holm over 6; `provenance_sweep_20261116/slide21_relative_check.py`):
  position -.42 / -.17 / -.09, chronic p .023, Holm .046 (*); on raw balanced accuracy -.34 / -.13 / -.07, Holm .045.
  Per animal chronic: PS92 -.01, PS93 -.10, PS94 -.20, PS95 -.06 (4/4 negative, PS92 ~0). Chronic is borderline either way
  (Holm .057 absolute vs .046 relative); not a normalization artifact. Old slide's "all **" (-.41/-.19/-.09) came from the
  pre-rebuild inputs, not the scale.

### 11-17 — deck swap pass started (Priya: 1. swap A+B + rewrite text/notes, 2. run C renders, 3. replace held slides with updated versions)
- Deck backup Overview_2026-11-06.bak_pre_a8_swap_20261117.pptx; plan + per-slide spec schema docs/status/deck_swap_20261117/README.md;
  6 helpers: A+B specs in 4 batches; C renders (lmm_hidden companions on branch lmm-hidden-20261103; others on main). 162/163 = cohort2 QC, skip.
- Slide 21 old vs new decomposed (same script on the pre-rebuild epoch/data): old pre had 11 sessions / animal (incl. June 06-06..08),
  rebuild 8 (August from 08-01). Pre levels .850/.744/.923/.915 -> .829/.681/.922/.908 (PS93 -.06); acute .451/.288/.466/.607 ->
  .499/.309/.515/.629 (all higher). Both shrink the drop (-.405 -> -.347; chronic -.093 ** -> -.077 n.s.). Not decomposed by factor:
  the gate is not isolated (Priya: camera gate adds non-contact-lick trials, expected to LOWER accuracy; post went UP).
- 11-17: slide 21 normalized-to-own-pre FIGURE (provenance_sweep_20261116/slide21_relative_figure.py, same data + drawing as
  fig_state_position_sessions.py): .../rebuild_figs/<tag>/rel_own_pre_20261117/state_position_sessions_rel_own_pre.png.
  With the figure script's seed (20261102) chronic position Holm .054 (n.s.); the quick check (seed 0) gave .046 -> chronic sits
  ON the .05 line, seed-dependent; report it as borderline, not as a star either way.

### 11-17 — deck swap pass: A+B applied (63 slides)
- Applied to Overview_2026-11-06.pptx (backup .bak_pre_a8_swap_20261117.pptx) by docs/status/deck_swap_20261117/apply_specs.py
  from specs/batch1-4.json; logs apply_log_batch1_2.csv, apply_log_batch4_s61.csv, apply_log_batch3.csv. Slide 21 applied with
  Priya's OK ("ok to replace slide 21, with our ongoing investigation"): absolute scale, chronic "borderline", relative numbers +
  decomposition pointer in notes. Slide 61 re-drawn on current code (contra-left flip) from rebuild figdata. 147 widened (box override).
- Main-point changes applied: 20 (chronic Bonferroni cells now far ipsi + far contra), 21, 101 (chronic position-free = medial loss),
  117 (far spouts still down chronically, new title), 197 (per-trial; pre separation weak in all 4).
- NOT applied: 17 (LMM companion, queued to the lmm_hidden render helper), 157 (quiet task events, queued C render);
  105 top / 122 / 125 / 128 / 130 / 131 / 133 / 200 / 203 = docked-spout ITI set never rendered on the rebuild -> docked arm
  (iti_dock_page_figures + lick_matched_readout gauss + lick_lateral_ratio, WFIELD_ITI_FILTER=dock) queued as group C.
- Helper flags to follow up: l_offset_terms_g log labels a flat session mean "mean of animals" (scripts/lick_offset_terms_recovery.py
  ~L85); e_first_lick_g froze NEW iti_first_lick decoders (SPEC-CHANGED, pre set changed); slide 112 figure stars are LMM;
  purple trial-set stamp whited out of several crops where it overprinted figure text; 139 = capped set until WFIELD_BAL_FULL render.
- 11-17: 3c (Priya OK) lifted the hold on state-space SVD / LocaNMF-basis slides (11-08 tables were already glue-free; re-run byte
  identical; M:/.../state_space_recovery/svd_lnmf_gluefree_20261117/logs/RUN_LOG.md). Checked the PI deck: no slide uses SS Fig 6.3 /
  6.4 (text + notes search), so nothing to change here.
- 11-17: slide 21 now carries BOTH figures (Priya: "add the per-animal pre-normalized versions"): absolute on top, each session /
  its animal's own pre mean below (legend rows cut; swap_inputs/s021b_state_position_rel_own_pre.png); caption rewritten;
  backup .bak_pre_a8_s21rel_20261117.pptx; script deck_swap_20261117/slide21_add_relative.py.
- 11-17 INCIDENT: a PowerPoint GUI process (PID 43556, started 11-16 16:20) has Overview_2026-11-06.pptx open (no lock file). My COM
  export of the LIVE path attached to it, showed the 11-16 in-memory version, and its Close() may have closed that window.
  Saving from that window would overwrite today's deck edits (and 11-16 Data: blocks after 16:20). export_slides.ps1 now refuses
  live-deck paths; told Priya.
- Stale-code check (3c 11-17): `python scripts/...` from a worktree imports the main checkout's wfield_local (663 behind). My slide 21
  runs did; the modules used differ only by ADDED functions in analysis_kit, so the numbers stand. Helpers told to set PYTHONPATH.
- 11-17: group C LMM companions applied (15 slides: 5, 12, 17, 19, 106, 107, 114, 121, 127, 136, 137, 142, 145, 150, 151) from
  specs/groupC_lmm.json (renders .../grant_figures/<tag>/lmm_hidden_20261117/, branch lmm-hidden-20261103 pushed, e3f6e02c; 20-hit
  gate now applied to 121/142/145). Main-point changes: 12 (acute far contra LMM degenerate, not tested; 4/4 animals still drop),
  145 (acute code + gap not tested under the 20-hit gate; chronic gap -.29 *), 151 (acute orofacial-ipsi lick residual now *).
  Backup .bak_pre_a8_groupC_lmm_20261117.pptx. Waiting: 108/109/123/129/134/138/201 (docked ITI arm), 111/116 (no rebuild input).
- 11-17: group C other applied (10 slides: 104, 110, 115, 119, 139 full set, 154, 155, 198, 199, 202) from specs/groupC_other.json
  (139 text edits re-targeted to the batch-4 text; caption now "full set"). Backup .bak_pre_a8_groupC_other_20261117.pptx.
  Need Priya: 22 (no session_z frozen model on the rebuild; options keep / swap to epoch_13 state figs / fit new frozen models),
  157 (event-linked weaker pre-cue code now present PRE-stroke too: pre strict -.020 p .001; "after the stroke" claim fails).
  Docked arm landed (matched_gauss_dock, lateral_ratio_dock, lick_order_dock, itidock_main _20261117); specs being redone:
  docked_batch3.json (105/122/125/128/130/131/133), docked_batch4.json (200/203), groupC_lmm_docked.json (108/109/111/116/123/129/134/138/201).
  Deleted 3 superseded helper tries (stale-package / LP-default first tries), 125 files listed in deck_swap_20261117/DELETED_SUPERSEDED_20261117.txt.
  Slide 12 LMM: acute far contra SE NaN = singular animal x epoch variance (4 animals -.61..-.67, one pre value per animal);
  wording/refit decision with Priya. Slide 145: PS92 (10) / PS94 (8) acute hits < 20, PS95 none -> only PS93 acutely.
- 11-17: docked 200 + 203 applied (specs/docked_batch4.json; backup .bak_pre_a8_docked_20261117.pptx). 203 docked ITI balance up acutely
  in every area (0-1 s: ALM +.50, tjM1 +.88, SSp-m +.88), chronically only ALM; tjM1 spout x epoch LR fit degenerate (-inf, p 1, printed in
  the panel title); render lacks the old green DOCKED banner (notes + Data: say docked).
- 11-17: docked 105 / 122 / 125 / 128 / 130 / 131 / 133 applied (specs/docked_batch3.json). 130: docked ITI lateral r no significant
  recovery (was "partial"); ITI offset 3/4 subacute (was 4/4 every epoch). 125 weaker subacutely.
- 11-17 Priya decisions applied: 22 kept as PRE-REBUILD RECORD (notes line; Data: lines may describe the rebuild step), 157 rewrite
  ("weaker code at every epoch, pre-stroke included"), 12 wording (degenerate cell, see bootstrap); per-session-pre LMM version of 12
  queued (5r sidecars have 8/7/8/8 pre sessions). Backup .bak_pre_a8_priya1117_20261117.pptx; script deck_swap_20261117/priya_1117_edits.py.
  145: hit gate stays MIN_HITS = 20 (A11, 11-07); the 10-trial rule is D7 = refit floor, a different rule. Proposal to Priya: show
  low-hit acute cells as hollow / labeled, untested.
  203: docked ITI tjM1 0-1 s spout x epoch LR = -inf: full mixed model failed to converge (MLE on boundary); pre tjM1 docked balance
  very heterogeneous across animals (PS93 / PS94 to -1.8 / -2.0 at close spouts, PS92 / PS95 ~0) with few docked licks per cell;
  0-0.5 s fits (p .016). A fit failure, not a result.
- 11-17: LMM round 2 applied (15: 12 per-session pre [acute far contra ** p .0002], 108, 109, 111, 116, 123, 129, 134, 136, 137, 138, 201
  docked; 121 / 142 / 145 with <20-hit cells shown, not tested). specs/groupC_lmm_docked.json; branch lmm-hidden-20261103 d44d8e25;
  backup .bak_pre_a8_lmm_round2_20261117.pptx. Helper rule slip (rewrote 3 input csvs in its own render folder, identical content; fixed a551070e).
- 11-17: neg-AUC diagnostic first output: tjM1 ITI trace figures in .../iti_licks/neg_auc_diag_20261117/. Undefined rows = small short
  lick transient (peak ~8-10 vs ~20) then a dip below baseline at 0.5-1.0 s (to ~-8..-14); defined rows stay up after 0.5 s
  (following licks / next trial). Clearest PS94 pre, also PS93. Tests of alternatives still running.
- 11-17: low-hit cells shown on main 141 / 143 / 144 / 146 (specs/lowhit_main.json; code 997573fb --show-low-hits; renders
  attempt_coding/<tag>/redraw_lowhit_20261117/; tables byte-identical). 147 not hit-gated. Open for Priya: 146 day-bin means INCLUDE
  low-hit sessions (bin 2 .34, bin 5 .61, bin 6-7 .68; never hit-gated) - gate them? Slide 120 (key_figure_BC code on hits) not yet marked.
  Backup .bak_pre_a8_lowhit_main_20261117.pptx.
- 11-17 Priya: ITI AUC + balance -> 0..0.5 s (DECISIONS 2026-11-17); helper re-rendering (*_iti05_20261117, specs/iti05.json); 3c told
  (goal page owner). 146: keep day bins ungated ("20 hits per epoch is very different than 20 hits per session"), low-hit points noted
  (already so). 120: low-hit marking queued (specs/lowhit_120.json).
- 11-17: cued balance -> cued angle done (49d1f416; .../iti_licks/bal_full_cuedang_20261117/): LOAO .68 vs days .08 (cued -> aim .66 vs
  .23); cued angle r .94 with aim -> cued balance mostly a lick-direction readout. Spec specs/s139_cuedang.json NOT applied: goes to
  Priya first (conclusion-level; spec replaces the caveat bullet).
- 11-17 neg-AUC diagnostic: cued / cue NOT biphasic (tjM1 0.5-1.2 s trough +22 / +26 above baseline; 0 / 214 curves below baseline).
  ITI dips: share of tjM1 sessions with 0.5-1 s AUC < 0 = 9 % all-ITI, 24 % docked, and EPOCH-DEPENDENT: all-ITI 26 / 12 / 0 / 0 %,
  docked 43 / 30 / 18 / 9 % (pre / acute / subacute / chronic) -> the 0..1 s ITI integral is pulled down mostly in PRE, inflating
  post-stroke ITI increases vs own pre. 52 % of ITI bout starts are single licks (short transient). Figures neg_auc_diag_20261117/.
- 11-17 Priya: full audit of ITI-window impacts requested (code, outputs, pages, decks, findings docs) -> helper writes
  docs/status/deck_swap_20261117/ITI_WINDOW_AUDIT_20261117.md and re-renders everything feeding the deck or a page; page items go
  to their owners (goal page = 3c). Slide 139 4-column figure: reorder cued left / ITI right (queued). Balance sign answered: stored
  balance = contra - ipsi (+ = contralesional hemisphere larger = looks like an IPSILESIONAL lick in the pre code); slope +13.4 deg per
  log2 unit vs aim (- = ipsilesional): sessions whose activity looks more ipsilesional land LESS ipsilesionally (angle-matched
  reference = executed angle, so a target-like signal reads as the landing error).
- 11-17 Priya: spout-matched + docked ITI angle markers. Spout-matched ITI angle already existed in learned_compensation.py (T2 same
  previous spout + phase; T3 docked D_off) but behavior-only, never on the rebuild, and NOT the recovery-marker axis (iti_ang =
  unmatched session mean of all ITI licks). Helper adds iti_ang_matched / iti_ang_dock / iti_ang_dock_matched (+ docked ITI
  balance) and far-contra within-spout raw balance vs angle (cued; docked ITI by previous spout) -> ITI_MATCHED_ANGLE_MARKERS.md.
- 11-17 slide 139 decisions (Priya): keep the caveat bullet + add the cued-angle result as its own bullet; layout 3a (bullets on top,
  4-panel figure full width); title waits for the matched / docked ITI results ("lick direction IS recovery, so that's ok").
  Apply once the ITI-marker panels are final (panel set may change).
- Docked ITI marker was ALREADY run pre-rebuild (itifilter_dock/recprim_*.csv; FINDINGS ~L852): docked ITI balance LOST recovery
  tracking (LOAO -0.11 vs days .15; all-ITI .42) and the ITI balance shift itself (acute -0.11 ns vs -0.34). Helper told to reproduce
  on the rebuild via the same ITI_SUBSET hook and report side by side.
- 11-17 Priya Q "does balance still predict overall lick angle?": rebuild l_bal_angle_lmm_r2 (all-ITI, session x previous spout,
  0..1 s resp): ITI dec pre slope strong in every pair (ALM -.20, tjM1 -.26, SSp-m -.28, SSp-n -.51 per deg, p < 1e-4; mildly
  flatter post); ITI resp weak (ALM n.s., tjM1 -.010 p .02, SSp-m -.024). Cued: strong in every pair (~30 % flatter post). Docked
  version of this LMM never run -> added to the matched-ITI helper (ITI resp at 0..0.5 s).
- 11-17 matched / docked ITI markers done (1fc7145b; ITI_MATCHED_ANGLE_MARKERS.md): ALL-ITI balance predicts every behavioral ITI axis
  incl. docked + spout-matched angle (lick-weighted cvR2 .53 vs days .04; matched .51 vs .19). DOCKED balance: cvR2 <= .05 vs days
  .15-.22 (per-animal r .49/.35/.78/-.26 to ITI angle: 3/4 same sign, PS95 reverses; docked = ~35 % of ITI licks -> noisier).
  Cued far contra (target fixed): raw resp balance shift at the same angle +.24 log2 (+.23 inside pre range) survives.
  Balance predicts raw angle across spouts pre and post (dec + resp, docked too).
- 11-17 documented (Priya "document the figures and findings of the far-contra versions"): FINDINGS.md recovery-marker block gains
  "Recovery-marker checks 11-17" + "Far contra only, target / previous spout fixed" entries (figures + numbers); detailed record
  docs/status/deck_swap_20261117/ITI_MATCHED_ANGLE_MARKERS.md section 3.
- 11-17 ITI 0..0.5 s done (9c245793; lmm 77af6d25; audit ITI_WINDOW_AUDIT_20261117.md; specs/iti05.json = 105, 108, 110, 111, 135,
  136-138, 203, NOT applied: conclusion changes go to Priya first). ITI increases shrink ~1/3-1/2, none reverses; several cells gain *.
  Changes: 135 caveat flips (14.13 matched-angle ITI balance: ALM +.06 p .005, orofacial +.04 ns; was orofacial +.15 p .010); hidden
  108 (0 -> 5/21 *), 110 (share kept ~2/3 -> ~1/2, * every epoch), 111 (0 -> 2/21), 138 (13 -> 14); docked tjM1 LR fits (p .016),
  ALM loses (.084); page 14.29 "chronic orofacial only" fails (chronic adds SSp-ll/-tr/-ul). 113/114 = cued only (no ITI). Page list
  sent to 3c (hold page edits until Priya OKs).
- 11-17 docked rebuild of decoded-vs-executed figures queued (matched curves incl. gap2dock + cued first lick, angle versions,
  recovery link 16.3-16.5, where scatter / delta, per-lick error) -> DOCKED_DECODED_VS_EXECUTED_20261117.md.
- 11-17 Priya OK: ITI 0..0.5 s slide updates applied (105, 108, 110, 111, 135 [notes caveat], 136-138, 203; specs/iti05.json; backup
  .bak_pre_a8_iti05_20261117.pptx). FINDINGS (4 places) + ENCODING_FINDINGS (1) ITI 0..1 s numbers replaced (one owner, per 3c).
  3c told to swap goal-page figures. Curves figure: footer overlap + dashed (3-animal) acute docked ITI line -> fix queued with the
  docked-figure helper.
- 11-17 status: handoff item 00-A8-1117 written (FR handoff section 2). Slide 21 decomposition INTERIM: trial gate drives the change
  (B_gate -.397 ~ old -.405); pre set / scaling / lick detector ~no effect; hemo + LocaNMF variants running.
- 11-17 bounded indices done (e83b055e; --idx flag, defaults byte-identical; section 5 of ITI_MATCHED_ANGLE_MARKERS.md; FINDINGS
  sentence). dec's negative ITI same-angle shifts (incl. docked far contra -1.82) VANISH in dec_idx (+.031 ns) -> pre-lick level /
  gain, not evoked balance. Cued far-contra shift survives in dec_idx (+.095, inside range +.063, 4/4) and resp_idx (+.079).
  resp vs resp_idx identical in sign / significance everywhere. Unit test for bounded_index requested. 113 / 114 cued per-spout
  window -> 0..1.0 s queued (Priya OK). Slide 139 title options sent to Priya.
- 11-17 Priya: idx figures as a collapsible goal-page supplement -> captions + paths in deck_swap_20261117/PAGE_SUPPLEMENT_IDX_20261117.md,
  sent to 3c (page owner; its account cannot reach the page right now). dec / dec_idx ITI bout starts use BOUT_GAP 1 s (resp uses
  GAP_MAIN 2 s) -> -1.0..-0.5 s baseline can include the previous lick for 1-2 s gaps; gap >= 2 s variant queued.
- 11-17 slide 139 applied (Priya title "Cued ipsi-contra balance places an unseen animal on recovery"; caveat kept; cued-angle and
  ITI / docked bullets; 4-column figure full width below bullets; caption at bottom; notes + Data: from specs/s139_cuedang.json;
  script deck_swap_20261117/s139_apply.py; backup .bak_pre_a8_s139_20261117.pptx; checked as an exported image).
- 11-17 overlap audit (Priya "ensure text isn't obscured by or obscuring figures"): deck_swap_20261117/overlap_check.py (bbox) over
  all 206 slides -> 6 flagged, checked as exported images: only slide 20 was a real problem (right figure ran off the slide,
  legend cut) -> resized to fit. 3 / 18 / 37 / 105 / 147: frames overlap whitespace only. In-image issue not caught by bbox:
  147 x-axis labels garbled where the purple footer was whited out (re-render with footer margin queued via curves helper).
- 11-17 docked decoded-vs-executed done (59223261, 74518c9f; DOCKED_DECODED_VS_EXECUTED_20261117.md): docked ITI like-for-like
  offset +0.30 / +0.21 / +0.27 mm (all ITI +0.36 / +0.24 / +0.27; after quiet +0.22 / +0.18 / +0.21); angle +4.3 / +2.6 / +3.4 deg;
  raw within-animal r(ITI offset, aim) gone on docked (+0.03 vs -0.33 all ITI), survives detrended (-0.30, p .013). Dashed acute =
  PS95 has no acute docked curve (53 licks). Footer overlap fixed (footer_bottom helper). gap >= 2 s dec_idx variant: conclusions
  unchanged; docked all-spout dec_idx now +.036 (p .002) like resp_idx. Supplement caption licks line corrected (DAQ lick times,
  LP 3-D angle). Helper commits landed inside another window's commit 74518c9f (shared index) - content verified on main.
- 11-17 slides 113 / 114 cued per-spout window -> 0..1.0 s applied (specs/cued01.json; 514e5438, lmm 7081a49c; backup
  .bak_pre_a8_cued01_20261117.pptx). 113: acute far contra now -.12 n.s. (tjM1), far mid loses * acutely; title holds. 114 (hidden
  LMM): 31 -> 22 / 72 starred, no acute star left. Slide 147 footer-fixed image applied (445902eb). Goal Fig 14.7 path for 3c.
  Waiting on Priya: lick-source wording fixes (3c audit LICK_SOURCE_AUDIT_20261117.md).
- 11-17 lick-source wording fixes (3c audit section 5) prepared: deck_swap_20261117/licksrc_apply.py, dry-run on a scratch copy
  places all 6 WRONG + 27 VAGUE edits (slide 102 needed the "pre-cue(ENL)" spelling). NOT applied to the live deck yet: 3c relayed
  Priya's OK, waiting for Priya's own confirmation in this window. RESTnl map wording waits for the RESTnlc1 figures.
- 11-17 Priya: "ok to fix lick sources on slides" -> 33 edits applied (licksrc_apply.py; apply_log_licksrc.csv; backup
  .bak_pre_a8_licksrc_20261117.pptx). "you can delete stale backups" -> 17 a8 deck backups deleted (list
  deck_swap_20261117/DELETED_DECK_BACKUPS_20261117.txt); kept .bak_pre_a8_swap_20261117 (before today) and
  .bak_pre_a8_licksrc_20261117 (before the last edit); .bak_pre_rebuild_20261116 not ours, untouched.
  Pooled cued response 0-1 s vs own pre (fold - 1): tjM1 contra +.51* / +.40* / +.41*; tjM1 ipsi acute +.37 n.s.
- 11-17 slide 21 decomposition done (ba24aebf; SLIDE21_DECOMPOSITION.md): trial gate explains the change (old gate on the rebuild
  -.397 / -.191 / -.112); ~-.02 from the old fitted-vs-projected joint time courses; pre set / scaling / lick detector none.
  Priya OK -> slide 21 notes line replaced with the result.
- 11-17 Priya OK -> FR page v55 published: 12 lick-source caption fixes (3c list; FR lick windows = LP video cam4 only,
  licks_all.csv; Fig 6.1 ENL window cam4 + DAQ). Backup scratchpad fr_page/index_v54_pre_licksrc.html.
- 11-17 late: grant panel check resolved (grants/k08_2026 render_fig4_rows.row_DE; boot_star source has no acute ipsi tjM1 star;
  matches the rebuild). Handoff 00-A8-1117b updated for compaction. Nothing of a8's running; worktree clean.

### 11-17 late (a8): like-for-like combined figure
- Priya asked: SEM as a faded band; one multi-panel figure instead of repeated cued panels; loclin with a Gaussian kernel;
  rows with vs without like-for-like matching; thinner translucent curves, short-line legend, shared axes; dashed notes
  under the figure; strict = each animal's own pre curve of the same lick set (her question on post landings outside the
  pre zone); no unity line ('too busy'); slide title not ITI-only.
- Done: figure (panels_loclin_first_20261117), goal page v317 (Fig 12.2 + S2, brief card), deck slide 125, FINDINGS 4d0k,
  DECISIONS entry, handoff 00-A8-1117c. Bars identical in both rows; ITI shifted every epoch on every ITI set.
- Later: figure retitled 'Decoded vs landed lateral position' (page v318); non-linear version -> page Fig 12.2 S2 (v319)
  + hidden deck slide 126 (deck now 207 slides; every slide after 125 +1). Quiet-event re-run (cam1 far-from-mouth guard,
  Priya OK via 3c) queued: waiting for 3c's commit on fix/cam1-farmouth-guard-1117 (worktree wf_c1guard1117 has the
  guard edits + tests/test_cam1_keep_mask.py, uncommitted at 19:2x). Production env: cohort1_quiet_spikes/
  v2_quiet_lp_20261117/logs/quiet_v2.log. Nothing of a8's running.
