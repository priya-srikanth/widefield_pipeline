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
