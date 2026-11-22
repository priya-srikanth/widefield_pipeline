# Running list 2026-11-21: four-camera 3-D re-run (overnight, window de)

Branch `pilot/side4cam-wholesession` (worktree `C:/Users/SabatiniLab/Github/wf_side4cam_pilot`), NOT pushed: Priya reviews
in the morning. Plan: `STATUS_2026-11-18_CAM23_RERUN_INVENTORY.md` (sections 0c, 2-4), `MANUSCRIPT_C2_REFRESH.md`, and the
Mac prompt `vls_stroke_paper/PROMPT_c2_4cam_3d_refresh.md` (Phase A gate -> B tables -> C READY file).

## Decisions needed (Priya) -- newest first

- **3-D lick floor: 40 ms applied as authorized, but 40-80 ms looks like fragments too (small).** Pilot four-camera 3-D
  licks under the adopted split, peak-to-peak intervals (PS93 0814 / PS94 0817): <40 ms 37 / 18 (now merged), 40-60 ms
  27 / 5, 60-80 ms 22 / 6; cam4's own lick detector has 0 / 0 below 60 ms and 10 / 1 at 60-80 ms; the 3-D median is
  160 / 168 ms. So ~0.9 % of PS93 3-D licks (40-80 ms) are probably still split. Option: raise the 3-D floor to 60 or
  80 ms (cam4's empirical minimum) -- a parameter (`pose3d.lick_peaks(min_ili_ms=)`), re-applied from the frames
  without re-running the 3-D. Running with 40 ms until you decide. Script: `lp_stage/pilot4cam/split/ili_hist.py`.

## Authorizations (Priya, 11-21 ~05:30)

1. Run from this branch; no push until she reviews.
2. 40 ms min inter-lick floor on every 3-D lick track (check the 40-60 ms intervals first; stop if they look like fragments).
3. Regenerated outputs go to new folders / run tags; scripts that rewrite pages / tables in place: keep a copy of the old
   version beside it. Superseded outputs stay; list them for review + deletion when done.
4. Delete local `lp_stage/lp_side_r2b_prod_local/out/tmp*` once the merge into the bundle's out/ is verified; clear local
   disk as needed (ask if unsure).
5. Kinematics before neural analyses (pending manuscript).

## Log

- 05:00 all 274 side tasks done and verified (frame counts == timestamp CSVs; O2 1-206, 227-230, 44 on the share;
  desktop 207-226 + 263-274; other computer 231-262). Task 208 (PS95 0603 cam3): 1 more video frame than timestamp
  lines -- check the template handling at collect.
- 05:30 local outputs copied into the bundle's out/ (96 files); other computer's out_helper copying.
- 05:40 merged origin/main (PR #2: `docs/MANUSCRIPT_C2_REFRESH.md`, c2 correlation scripts) into the branch (d93fa42a;
  DECISIONS.md conflict = two additions, both kept).
- 05:45 three independent blast-radius reviewers started (forward from inputs / backward from manuscript + pages /
  recent commits + plan audit); reports to `C:/Users/SabatiniLab/lp_stage/blast_radius/`.
- June side-view spout-fit check running (Priya: same lenses June to August).
- 06:10 all 274 verified in the bundle's out/ (273 OK + task 208 known off-by-one: the video has one frame past the last
  timestamp line; frame ids are contiguous, so it is the trailing frame). Local temp folders deleted (48 GB).
- 06:15 collect for every session started (`o2_inference collect --missing-only --jobs 6`; log
  `lp_stage/rerun_20261121/logs/collect_side.log`).
- 06:30 01fa06ad: 40 ms floor in `pose3d.lick_peaks` (merges sub-floor runs over the union of frames; tests) +
  `WIDEFIELD_POSE3D_TAG` (run tag `_3net_20261121` -> `session_poses/pose3d_4cam_3net_20261121`; never on 2cam).
- 06:35 Phase A prep agent: parked fixes in the 3-D path + one-session smoke test of the tagged production run.
- ~07:45 Phase A prep done (agent). 3-D layer: ENL ruler-median poses + commanded aim already in pose3d / pose3d_side;
  per-frame spout edge offset (bddf02e7) in HEAD and in the four-camera fuse (spout = cam1 edge offset); commanded 3-D
  spout table (cf7c4cad, prep_3d_inputs) present. 2-D-only spout items (go with v37): orofacial_outputs.py:184 ->
  ruler_medians, tongue_kinematics.trial_spout_reference, ~17 whole-window scripts, whisker_motion, heatmaps spout_ref.
  Close-spout jaw: NOT the 3-D layer -- the 4-cam frames carry it (PS93 0814 cue..stop jaw 3-D 100 % at close_center,
  >= 3 views 94 %; 2-cam 93 %), the decision sits in jaw_onsets (JAW_UNRELIABLE = close_center mask, source-blind) and
  trial_classes jaw_judged (L1d / L2, needs a source-aware rule + review). Production lick split = event_cuts(S) + 40 ms
  floor (PS93 0814: 5624 3-D licks).
  New: June gate replaces the blanket June skip (side_gate: rms <= 1.5 / max <= 2.0 px600; failing camera dropped);
  BUG FIXED: the 4-cam path skipped the June cam4 rescale (fuse_session now passes _june_factor); run tag routing:
  run_mode.pose3d_part (p3d-4cam_3net_20261121, output-only) + camera_licks folder tagged; within_lick_correction /
  lick_correction read pose3d_dir(). Smoke (tagged folder): PS93 0814 ~12 min, 5624 licks; PS93 0607 + PS92 0602 471 s
  with --jobs 2 (June cam4 scaled; PS92 0602 cam3 dropped, rms600 1.65 / max 2.14; PS93 0607 both pass 0.41 / 0.52).
  OPEN for the coordinator: (1) 3-D kinematics (`pose3d_kinematics`) defines licks by the v36 cam4 tables (licks.csv /
  licks_all.csv on the 3-D track; also reads literal V36) -- the fused four-camera lick train / new split reach
  kinematics only with a new lick mode (a lick-identity change: lick_uid joins in iti_licks / frozen_encoder /
  stray_licks / lick_correction key on licks_all rows) -> decision, not built; (2) many downstream scripts still read
  literal session_poses/pose3d (c2_scalar_family_aligned, c2_engagement_tim/s4, face_prelaunch_aim, kinematic_normality,
  coupling_vigour_sm, enl_lick_audit, lick_firstlick_extract, lick_vigor_perspout, june_drift_summary_fig, stray_licks,
  pose3d_jaw_depth_model_check, ...) -- they would silently read the 2-cam tables in a 4cam run; (3) runbook step 1 says
  the 4-cam run writes session_poses/pose3d/ and never sets WIDEFIELD_POSE3D_SOURCE=4cam (coordinator fixes the doc).
- ~08:30 side collect DONE: 270 videos OK (+ PS93 0814 / PS94 0817 from the pilot); `session_poses/<a>_<d>_cam2_full` /
  `_cam3_full`. Not collected: PS94 0601 and PS95 0601 side views (no side alignment template; both outside the curated
  set). Task 208 (PS95 0603 cam3, one trailing frame without a timestamp) collected OK.
- ~09:00 STEP 1 (v37 2-D chain) DONE (e11f3978 + DECISIONS entry): `LP_analysis/lp_r4_heatmap_20261013__clean_v37`, 133
  sessions (= v36), 69 min + 1 min all-licks; all frames from ENL medians. v37 vs v36: licks median 0.0 %, licks_all
  0.01 %, origin 0.8 px median; outlier PS92 0929 (v36 close_R spout median 24 px off -> v36 mouth origin 37.7 px off;
  v37 correct; licks +4 %). Table `lp_stage/rerun_20261121/v37_vs_v36.csv`. RUN not flipped.
- ~10:30 step 2 DONE (cam1 re-detection, v37): bbbf862e adds opt-in routing `WIDEFIELD_LICK_INPUT_RUN` (cam4 run) +
  `WIDEFIELD_CAM1_LICKS_DIR` (versioned cam1 folder) to `lp_licks._inputs` / `scripts.cam1_licks` (unset = unchanged;
  set = in `spec_token`); orofacial_cohort.RUN not flipped. Output `session_poses/cam1_licks_v37` (125 sessions; the
  production folder has 119). ~90 s / session, 6 parallel. Pass 1 `in_3d` = the 2-cam 3-D (pass 2 after the four-camera
  3-D). vs production `cam1_licks`: post-cue events median -0.03 % (min -5.1 %), events_all median -0.1 % (min -6.9 %);
  8 sessions lose > 3 % of events_all (PS94 0823 / 0831, PS95 0812 / 0823 / 0825 / 0827 / 0903 / 0910), all with
  droplet-flagged events down ~5-15x (e.g. PS95 0812 349 -> 55) and the cam4-matched share up (0.80 -> 0.86): the 11-18
  detached-static drop mask (`mask_detached`, before cleaning) removing parked labels the production folder predates.
  Table: `lp_stage/rerun_20261121/checks/cam1_compare_cam1_licks_v37.csv`.
- 7b88afb9: the 3-D production run writes `<s>_side_events.csv` (tagged camera_licks folder) so the fusion
  (`camera_licks session --saved`) does not rebuild the fused session a second time (~8 min / session saved).
- ~10:35 step 3 started: `scripts.pose3d production --all --four-cam lp_side_r2b_prod_20261118 --jobs 6` (dlc env;
  SOURCE=4cam, TAG=_3net_20261121, v37 + cam1_licks_v37 routing); log `lp_stage/rerun_20261121/logs/pose3d_4cam_full.log`.
- ~13:40 step 3 DONE: four-camera 3-D for all 126 sessions (same set as the two-camera `pose3d/`), 0 errors,
  `session_poses/pose3d_4cam_3net_20261121/` (+ summary_production_lp_side_r2b_prod_20261118.csv; side events in
  `camera_licks/lp_side_r2b_prod_20261118_3net_20261121/`). Median per session: tongue in 3-D 11.6 % of frames (cam1 +
  cam4 alone 7.7 %), jaw 99.94 % (min PS95 0820 91.4 %), fit residual 0.05 mm. Side gate dropped cam3 in 4 sessions:
  PS92 0602 / 0603 (as predicted), PS95 0818 (rms600 1.56) and PS95 0820 (rms600 1.26, max over 2.0) -- PS95 0818 / 0820
  are next to the PS95 cam2 jaw-confidence dip (0819-0827): worth a look at the side camera set-up those days.
  3-D LICKS: four-camera = 1.015 (August) / 1.009 (June) per cam4 post-cue lick (session medians); two-camera production
  = 1.113 / 1.255. Peak-to-peak intervals < 40 ms: two-camera 45,282, four-camera 0 (the floor); 40-80 ms: two-camera
  40,571, four-camera 6,034 (~1 % of cam4 licks: the 40 vs 60-80 ms floor decision above). So the session-level drop vs
  the two-camera counts (median -8 %, June ~ -21 %) is the two-camera track's fragments, not lost licks. PS92 0605:
  two-camera 867 licks (cam1 + cam4 tongue 1.5 % of frames) -> four-camera 2,874 (the side views rescue the session).
  Tables: `lp_stage/rerun_20261121/checks/{pose3d_4cam_vs_2cam,licks_vs_cam4}.csv`.
- ~13:45 step 4 started: cam1 pass 2 (`in_3d` against the four-camera 3-D), then the fusion (`--saved`).
- ~14:40 step 4 DONE. cam1 pass 2 (`in_3d` against the four-camera 3-D) into `cam1_licks_v37` (125 sessions; events_all
  1,045,468, in_3d 56.7 %, droplets 6,751). Fusion (`camera_licks session --saved`, ~15 s / session instead of ~8.5 min,
  D2 revised, trial-window span "all"), 126 sessions, 0 failures, `camera_licks/lp_side_r2b_prod_20261118_3net_20261121/`
  (`<s>_fused.csv`, `<s>_events.csv`, `<s>_side_events.csv`, `views.csv`). Pooled: 1,103,693 licks; all four cameras
  76.8 %; cam1-only 7.0 %; cam4-only 2.5 %; guarded-only dropped 5,479; side-only kept 14,696 (1.3 %; session median
  0.42 %). Outliers: PS93 0805 95 % side-only (no behavior log -> no cam1 / cam4 events; exclude it from fused-train
  analyses, as it is from the cam4 chain); PS95 August sessions 3-10 % side-only kept (PS95 0820 9.8 %, its cam3 dropped by
  the gate) -- same PS95 0818-0827 window as the cam2 jaw-confidence dip: look at the PS95 side cameras on those days
  before trusting PS95 August fused counts.
  D3 NOTE: the fusion covers the collected TRIAL WINDOWS (strobe - 0.5 s .. stop + 3 s, cut at the next trial; ~99.7 % of
  the video, pilot); the video before the first and after the last trial is not in any camera's windows, so "full
  session" needs those edges added at collect (all cameras) -- not done (a collect-level change; not a 3-D step).
- ~15:30 PHASE A GATE (dc884bbf `scripts/pose3d_phaseA_gate.py`, dlc env: pandas 2.3.3 / numpy 2.2.6 / scipy 1.15.3 /
  sklearn 1.7.2), 50 pre-stroke sessions (31 curated August), tables `session_poses/pose3d_4cam_3net_20261121/qc_phaseA/`.
  1 COVERAGE PASS: tongue 3-D share >= two-camera in 50 / 50 sessions, jaw in 49 / 50 (PS92 0605 jaw 0.992 vs 0.994;
  its tongue 2.5 % -> 12.3 %). August medians, tongue (all frames) 2cam -> 4cam: PS92 10.1 -> 12.5 %, PS93 12.6 -> 13.0,
  PS94 11.1 -> 12.3, PS95 5.7 -> 6.2; >= 2-view tongue 6.8 / 9.2 / 8.1 / 3.8 -> 11.2 / 11.3 / 11.4 / 5.6 %; jaw 80-99 ->
  99.3-99.95 %. Reprojection is larger in cam1 / cam4 by construction (over-determined fit; the 11-04 approved fusion
  numbers): August tongue cam1 3.0-3.9 -> 5.3-6.7 px, cam4 2.8-3.7 -> 4.2-5.8, side 1.7-2.2; jaw cam1 1.0-1.7 -> 6.8-11.7
  (cam1 chin label), cam4 0.9-1.6 -> 2.7-5.4, side 1.3-2.4. June cam4 rep 12-16 px (June cam4 rescale) and the two
  cam3-dropped sessions (PS92 0602 / 0603: cam4 11 px, cam2 5.4) are the worst; 3-D point 2cam vs 4cam on common frames
  median 0.15-0.20 mm tongue (June 0.28-0.35), 0.26-0.55 mm jaw (PS95 August largest).
  2 AGREEMENT PASS: own 3-D licks matched (peaks <= 30 ms) 85 % of 2cam / 96 % of 4cam licks (median; min 4cam 0.20 =
  PS92 0605, 2,874 vs 867 licks); identical cam4 windows 212,401 licks. Per animal r (identical licks) lateral
  0.986-0.992, midline angle 0.979-0.985, reach 0.966-0.988; mean offset 4cam - 2cam lateral -0.013..+0.037 mm, angle
  +0.06..+0.46 deg, reach +0.10..+0.18 mm (side views place the tip further out, as at 11-04).
  3 DECODING PASS (pose3d_vs_cam4.decode, LOSO August pre, identical licks): 2cam 0.805 / 0.674 / 0.862 / 0.737
  (= production table exactly) -> 4cam 0.886 / 0.706 / 0.875 / 0.801 (PS92-95); own 3-D licks 0.891 / 0.702 / 0.878 /
  0.810; chance 0.17.
  4 SUBMOVEMENT PASS on the stated criteria, FLAGGED: 4cam threshold calibrates to theta_sp 40 (2cam 50; theta_ml 50
  both). Real > noise surrogate in 4 / 4 (cued, mean of animals 0.166 vs 0.048; 2cam 0.159 vs 0.050). Far lateral >
  close lateral in 3 / 4 (PS93 fails in both runs) BUT the contrast shrinks: 0.211 vs 0.147 (2cam 0.301 vs 0.059); close
  spouts gain CSMs (close_ipsi 0.078 -> 0.203, close_mid 0.057 -> 0.181), far lose (far_ipsi 0.32 -> 0.23, far_contra
  0.28 -> 0.20); PS95 close_mid 0.37 > far. The four-camera track changes the CSM measure -> review before step 7
  (recovery mechanics). Extraction pickles in qc_phaseA/within_lick_correction (not the routed folder, so Phase B cannot
  reuse them by skip-if-exists); two-camera = cached production extraction, read only.
- Phase C: `vls_stroke_paper/READY_desktop_c2_4cam_20261121.md` written (Phase A pass, gate 4 flagged; Phase B NOT run,
  every table PENDING; 8 questions for Priya; superseded-output list). STOPPED here pending Priya's decisions.
- Priya's answers to the READY file's questions (11-21 evening):
  - Q2: ADOPT the four-camera / v37 run as the new cohort default. Flip `orofacial_cohort.RUN` (+ pose3d source / tag
    default) as one commit once the consumers are rewired and the `lick_uid` key join is in; branch only until she reviews.
  - Q3 (floor): decide from example traces; diagnostic running (`qc_phaseA/ili_floor_examples/`).
  - Q4 (submovements): review first; example traces + artifact tests running (`qc_phaseA/csm_examples/`).
  - Q5 (close-spout jaw): use the four cameras -- replace the source-blind close_center mask in `jaw_onsets` /
    `trial_classes` jaw_judged with a source-aware rule (four-camera frames judge the jaw there).
  - Q1: explained; awaiting her choice.
- Priya (11-21 evening, cont.). Camera naming: cam4 = FRONT, not bottom.
  - Q1: (b) -- the fused four-camera train defines the 3-D kinematics lick set, tolerant of missing camera frames (any
    camera counts per D2; a lick keeps whatever 3-D frames exist; kinematics NaN only where too few 3-D frames).
  - Q6: PS93 0805 has the h5 + a reconstructed trial table (`behavior_trials`, cam1 recovery). It is in
    `cross_session_exclude` (outside the curated window), so it does not affect results; its camera lick tables are
    missing (the cam4 / cam1 chain did not pick up the reconstructed trials) -- low priority.
  - Q7: look at examples -> diagnostic running (`qc_phaseA/ps95_side_week/`: daily side-camera frames + side-only licks).
  - Q8: full session = first to last sync pulse (all cameras); collect window to be widened and only the head / tail
    stretches run through 2-D -> 3-D -> fusion.
- 14:05 (11-21) ILI-floor diagnostic done (eef212a0, `scripts/pose3d_ili_floor_examples.py`; README + figs in
  `pose3d_4cam_3net_20261121/qc_phaseA/ili_floor_examples/`): 40-60 ms pairs are 85 % fragments of ONE protrusion
  (retraction median -0.09), 60-80 ms mixed (some true double pumps), controls clean; 60 ms floor -0.57 % licks, 80 ms
  -0.93 %. Suggested (not adopted) retraction rule instead of a time floor -- decision for Priya.
- 11-21 Q7 PS95 side week (`scripts/pose3d_side_week_check.py`, `qc_phaseA/ps95_side_week/README.md`): side views did NOT
  move in the week (cam2 <= 2.3 px vs 08-12; cam3 shifted ~9-10 px once on 08-17 in PS95 AND PS94, then stable); of 21
  sampled PS95 0818-0827 side-only licks 17 = paw sweeping into view, 2 grooming, 1 water drop, 1 real (close_center);
  near-mouth (<= 4 mm) & >= 20 ms side-only stay ~0.1 % all week (post-hoc rule, 5/5 real kept, 1/36 false).
  Recommend: keep side geometry + gate, add a side-only guard in `camera_licks` before adopting the four-camera train.
- 11-21 Q4 CSM diagnostic (`scripts/pose3d_csm_examples.py` + `pose3d_csm_fixes.py`, `qc_phaseA/csm_examples/README.md`):
  close-spout CSM gain = mostly a measure-edge artifact: 33-75 % of four-camera close CSM peaks are on the run's LAST frame
  (zero-padded `_peaks` edge; tail into the mouth, one-view fills / side-only, labels on the lip); not counting it takes
  close ipsi / mid 0.20 / 0.18 -> 0.014 / 0.009 (2cam 0.035 / 0.018). Residual view-subset bias after production offsets
  0.008-0.035 mm (<< 0.16 mm / frame), bias correction no effect; switch replay +1-2 % false CSMs; switch-aware surrogate
  0.047 -> 0.058 (theta 40 -> 50). Same-threshold pattern persists (H3 out); far loss on the same licks = two-view noise
  in 2cam (cam1+cam4-only far contra 0.25 vs 4cam 0.16). EM Kalman / RTS smoother FAILS injection (keeps 18-61 % of real
  pulses) -> not adopted. Recommend: no-last-frame-peak rule in both runs + switch-aware noise control; Priya to decide
  whether CSM share stays a step-7 primary readout (4cam CSMs 1-3 % under the rule).
- 11-21 Q4 follow-up, gate 4 under the edge rule (`gate4_edge/`, README Part 3): first/last-frame peaks not counted for real
  AND surrogates. 4cam real 1.2-3.2 % vs false 0.2-0.9 % per animal (theta at the grid floor, 30), far > close 4/4; the rule
  also removes 71-89 % of the 2cam CSMs (pre, theta 50; 107-session check) and flattens the acute changes (PS94 / PS95
  down -> flat), so the manuscript two-camera CSM numbers need an e2 re-run. Recommend: CSMs descriptive only.
- 2026-11-21 side-only guard (Priya: "shape rule, validated"): blind fresh sample of 63 side-only licks -> shape rule (<= 4 mm AND >= 20 ms) keeps 12/25 real, 6/37 false = FAILS, NOT adopted (camera_licks unchanged, no re-run). Two-view fallback 19/25 real, 6/37 false, 0 paws / drops -> Priya to decide. Real side-only licks are mostly cam4/cam1 table gaps (PS93 0805 has no cam4/cam1 events: 6,941 = 47 % of side-only). qc_phaseA/side_only_guard/ (README, blind_sheet/ + blind_answer_key.csv); script 3f250436.
- READY file sections 6-7 added. Section 6: the CSM end-of-track artifact affects the CURRENT two-camera manuscript
  numbers (71-89 % of pre CSMs are edge peaks; the PS94 / PS95 acute drop disappears), so step 7 is PENDING. Section 7:
  Priya's answers and the floor / PS95 / side-guard diagnostics.
- 2026-11-21 CSM edge-peak fix (Priya: "yes fix the measure"; 025237ec): `_peaks` no zero padding, grid 10-200, PEAKS_VERSION 2. Two-camera re-run (theta 50 -> 20 mm/s) + S16 stats under new names, `Widefield/manuscript_stats/csmfix_20261121/` (README + cmp_*.csv): S16 csm_share acute / subacute stars lost (acute +0.16, surrogate +0.08), chronic -0.07** kept (surrogate also down); 0 flips in the 162 scalar cells; non-CSM cells unchanged. Four-camera (cam4 v36 lick windows) in `imaging_encoding/p3d-4cam_3net_20261121_daqlk/within_lick_correction_cam4v36licks_csmfix_20261121/`. DECISIONS entry added.
- 2026-11-21 table gaps (Priya: "go ahead with your table gaps rule"; 2e51b2c5, 376c5e8f): all 14,696 side-only licks diagnosed -- 47 % PS93 0805 (no cam4 / cam1 tables; chain fixed, tables built in `_sync` folders), 36 % not tracked by cam4 / cam1 (paws), 12 % tracked but not called, 5 % pieces of counted licks, 1 lick outside the windows. Tracking-confirms rule FAILED its blind check (30 new: 13 real / 15 paw-grooming-drop), not adopted; adopted: side-only needs cam2 AND cam3, cut-off pieces dropped (blind 19 / 25 real, 8 / 85 false, 0 / 46 paws). Fusion `..._3net_20261121_gapfix/`: side-only 7,755 -> 1,432 (125 sessions). Full-session windows already cover the sync span to a median 0.9 s; `WIDEFIELD_WINDOWS_SPAN=sync` coded, cohort re-run (~6.6 h) NOT launched -> Priya. README qc_phaseA/table_gaps/.
- 11-21 late: Priya chose a 60 ms 3-D lick floor (DAQ stays 40 ms). Code: pose3d.POSE3D_MIN_ILI_MS / camera_licks.MIN_ILI_MS; DECISIONS entry; READY Q3 updated. Applies from the Phase B licks run.
- 11-21 late: N18 (PS92 0908 t306) miss = REAL camera frame drops (EXPERIMENT_ERRORS 2026-09-08: all 4 cams, cliff at min ~104, 78-89 % dropped by the end). Curated sessions with >0.5 % drops: PS92 0908 only (PS92 0605 15 %, PS94 0603 2.5 % are outside the curated window). Grooming contact rule: Priya says skip if dicey (few bouts) -- deferred.
- 11-22: consumer rewiring for Phase B on branch `pilot/side4cam-rewire` (code + tests only; DECISIONS 2026-11-22; runbook
  `docs/MANUSCRIPT_C2_REFRESH.md` rewritten with the Phase B command list). Found: the 11-21 fused / events / side-events
  CSVs are 4-significant-digit (ITI times 10 ms) -> refused by the new loader; Phase B re-fuses (`_gapfix60`). Open for
  Priya: frame-coverage gate on trial classes too (now kinematics only); blind check of the 4cam close-spout jaw calls;
  3-D re-run to remove the side-event rounding (optional). Adoption = the last commit on that branch (droppable).
