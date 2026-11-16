# Render queue audit, 2026-11-08 (read-only)

Scope: the 191 unique step names in `scratchpad/pending.txt`. I ran no scripts and edited no queue or repo file; this report is the only file written. I spot-checked four findings by grep: simple_key_figures LC, lick_rim_overlay sessions_roi2, restnl_rescue precue_restnl, and the producer of key_figure sametarget_tests.

**How each step was audited.** I used the definition that owns the step's claim: `logs/<name>.claim/` holds RESERVED_v7, _v8, _v8b, _v8c, _v9y or _HOLD. When no claim exists, I used the newest queue file. Code was read in that queue's `W=` worktree:
- 7b_a7: wf_render_daqlk
- 8*_a7: wf_render_v8 (v8d, v8f)
- 9u_a7: v9s
- 9y: v9y
- 9zg: v9zg
- 9zh: v9zh
- 9zi: v9zi

In the tables, OUT = `.../imaging_encoding/iti_licks/<TAG>`, E or ENC = `.../imaging_encoding/<TAG>`, and IE = untagged `imaging_encoding`.

**State at about 02:45.**
- `a_partial_ls` holds HEAVY.lock (since 11-07 21:47, at 52/107). `t_frame_te_basis`, `l_balimg`, `gb_sessions_r`, `g5_epoch_15r_r`, `l_stopped_restw`, `l_block_compare` and others are queued behind it.
- `g5_section_g_r4` is running (16/88 units).
- `e_basis` is running, but its output is superseded by `e_basis_r`.
- Several steps hold `RESERVED_HOLD` claims and will not run until released: `d_fig_goal_summary*`, `g6_grant`, `g7_section_g_figs`, `g7_fig_pose_validation`, `b_imaging_lmm`, `b_remap_lmm`, `b_remap_checks`, `e_mm`, `e_lmm_*`, `e_rlink_hit`, `l_session_levels_z`, `l_levels_compare`, `l_key_figure`.

## Counts (191 unique names)

| verdict | n |
|---|---|
| NO_PRODUCER (includes wrong-path reads where the producer writes elsewhere) | 19 |
| ARG_MISMATCH | 8 |
| UNDECLARED_DEP, will break or produce silently wrong output | 14 |
| UNDECLARED_DEP, harmless now (producer already done or runs earlier in the same serial lane) | 17 |
| DEP_FAILED, will be SKIPPED (stale dep name) | 2 |
| BOX_DEFAULT_PATH (secondary; works on this box) | 2 |
| UNCLEAR | 2 |
| SUPERSEDED | 21 |
| OK | ~106 |

The counts are approximate where a row carries two verdicts.

## A. Will fail, or produce wrong output, as queued

| step | verdict | deps state | evidence | suggested fix |
|---|---|---|---|---|
| **e_fpa_extract** (RESERVED_v7) | UNDECLARED_DEP: e_pat_extract | e_iti_main done, so it is **ready now** in v7b lane E | Reads `pat.FEAT = ENC/prelaunch_aim_tracking/features` (face_prelaunch_aim.py:46,78-79,165; prelaunch_aim_tracking.py:44). The folder is absent. Its producer e_pat_extract (8_a7:540) is pending. With 0 labels, `S.sort_values` (:219) raises. 7b_a7:516 declares only e_iti_main. | Add e_pat_extract to the deps, or HOLD the step. |
| e_fpa_fit | UNDECLARED_DEP: e_pat_extract | e_fpa_extract pending | face_prelaunch_aim.py:238 reads S26F/<lab>.npz. 7b_a7:517. | Add e_pat_extract. |
| e_fpa_summarise | UNDECLARED_DEP: l_recov_markers | e_fpa_fit pending | Reads OUT_W/recmark_sessions.csv (face_prelaunch_aim.py:502). 7b_a7:518. | Add l_recov_markers (and e_pat_summarise, which is optional). |
| e_fpa_offset | UNDECLARED_DEP: e_dock_prep_dock | e_fpa_fit pending | Calls face_offset_partial.run, which reads `OUT/itifilter_dock/baltarget_licks.csv.gz` (face_offset_partial.py:175; face_prelaunch_aim.py:917-919). It also duplicates l_face_offset_partial, which writes the same output folder. | Add e_dock_prep_dock, e_contrib_split and e_split_order_all, or drop it in favor of l_face_offset_partial. |
| l_face_offset_partial | UNDECLARED_DEP: e_dock_prep_dock | e_fpa_fit and l_balmatch pending | face_offset_partial.py:175-176 reads itifilter_dock/baltarget_licks.csv.gz (lick_balance_matched_error.py:90). 8_a7:545. | Add e_dock_prep_dock. |
| **b_imaging_lmm** (HOLD) | ARG_MISMATCH (reads untagged paths) + UNDECLARED_DEP | none declared in 8_a7:291 | Last log: `FileNotFoundError IE\qc\trial_attempts_sessions.csv`. The file exists only in IE/qc/<sel>/. In v8, imaging_lmm.py:174 calls `mm.deficit(E/"qc")` with E untagged. `trend_table` (:46) reads IE/qc/position_coding_trend_recovery_sessions.csv, which m_trend_recovery wrote to ENC/qc. `behaviour_table` (:60) reads untagged cohort/tables/prepost, while k_cohort wrote cohort/<TAG>/. v9zn fixes only the deficit path (:184); trend_table still reads E/qc (v9zn :47). | Run on v9zn or newer **and** route trend_table and behaviour_table to the tagged dirs. Declare m_trend_recovery, g5_recovery_cue, g5_sessions_long_r and k_cohort. |
| b_remap_lmm (HOLD) | ARG_MISMATCH (deficit qc dir) | 8_a7:504: a_allsession_frA done, b_specificity pending | remap_lmm.py:76 calls `mm.deficit(E/"qc")` with E = ENC. ttc and ttc_log need trial_attempts_sessions.csv, which exists only in IE/qc/<sel>/. In v9zn (:77, `deficit_qc_dir` = attempts_dir), ttc works but `contact` breaks, because its CSV lives in ENC/qc. The 7b_a7:337 version also lacks the b_specificity dep (reads specificity_control/<TAG>/<F>_frA/sessions.csv, remap_lmm.py:38). | Choose the qc dir per deficit, or copy the file. Keep the 8_a7 deps. |
| b_remap_checks (HOLD) | ARG_MISMATCH (same as b_remap_lmm) | 8_a7:505: b_svf_frA done | remap_checks.py:89 (v8 :110) has the same `mm.deficit(E/"qc")` with `--deficits ttc_log ttc contact`. | Same fix as b_remap_lmm. |
| b_residual_roi | NO_PRODUCER | none | residual_roi.py:100-101 reads `IE/specificity_control/<TAG>/gate-attempted_v7g3_t3d_spont/sessions.csv`. The only specificity step runs `--tag ${F}_frA` (7b_a7:314). | Queue `specificity_control.py --tag "$F" --layer2 allsession_encoder_v7g3_t3d_spont_ref` (noFR) and add it as a dep. |
| b_encoding_key_figures | NO_PRODUCER | none | encoding_key_figures.py:41 runs `pd.concat` over the glob ENC/movement_first/sessions/*_regions.csv. That folder does not exist (only suffixed movement_first_* folders do), and it would come from unsuffixed combined_encoding (combined_encoding.py:2344), which is never queued. The resulting ValueError is not caught; only FileNotFoundError is (:143-146). | Point it at an existing movement_first_<suffix> folder, or catch ValueError. |
| b_mnull_maps | UNDECLARED_DEP: b_fr_musall_v2 | none declared; b_fr_musall_v2 pending | movement_null_plan_maps.py:259,269 read ENC/frozen_fr_compare/musall_{sessions,change}_v2.csv (written at frozen_fr_musall_v2.py:78,95). Neither file exists yet. | Add the dep b_fr_musall_v2. |
| b_residual_roi_frA | UNDECLARED_DEP: b_specificity | none; b_specificity pending | residual_roi.py:100-101 reads specificity_control/<TAG>/<F>_frA/sessions.csv. It is safe only while b_specificity (cont) succeeds earlier in the same lane. | Add the dep b_specificity. |
| **x_enlv_table** (RESERVED_v8) | NO_PRODUCER: `lick_enlv_extract.py --regress` | ENL_SLIDE_FIX and x_enlv_readers pending | lick_enlv_balance_offset.py:130 loads `enlv_licks/sessions_reg/<lab>.npz`. Only `lick_enlv_extract.py --regress` writes those (:228, :353-354), and no queue runs it. | Add `x_enlv_regress cont x_enlv_extract "" "frozen" $PY scripts/lick_enlv_extract.py --regress --jobs 3` as a dep, and add the command to HEAVY_RE. |
| x_enlv_balance (RESERVED_v8c) | NO_PRODUCER + UNDECLARED_DEP | x_enlv_table pending | compare() at lick_enlv_balance_offset.py:369-398 reads itifilter_notdock/ balmatch, ci_spout, recprim and matched_loclin. No notdock balerr, ci, subsets or recov steps exist. The dock producers (e_dock_balerr_dock, e_dock_ci_dock, e_dock_recov_dock, defined in the 7b_a7:494-501 loop) are not declared. arrival() :428 also globs sessions_reg. | Add notdock balerr, ci, subsets and recov steps (copy the 7b loop) and declare them plus the dock ones; also apply the regress fix. |
| l_enlv_target_code | UNDECLARED_DEP: x_enlv_balance | x_enlv_table pending | lick_enlv_target_code.py:56 reads enlv_sigma.csv, written by lick_enlv_balance_offset.py:839 (x_enlv_balance). The step's dep is x_enlv_table (8_a7:476). | Change the dep to x_enlv_balance. |
| l_code_rotation | NO_PRODUCER: subspace/noisematch with LICK_CODE_FAMILY=enlv | l_code_noisematch not started | lick_code_rotation_summary.py:104 always loads noisematch_enlv.pkl (TYPES :36-38). No queue sets LICK_CODE_FAMILY. | Add l_code_subspace_enlv and l_code_noisematch_enlv (env `frozen LICK_CODE_FAMILY=enlv`) and make this step depend on them. |
| l_code_subspace | UNDECLARED_DEP: e_june_ref (done) + unneeded dep | x_enlv_table pending (broken chain) | The default family reads OUT/june_ref (:133, :263). The ENLv files are read only by `_load_enlv`, which this run never calls. Because the step is stop-mode and depends on the broken ENLv chain, it will be SKIPPED. | Replace x_enlv_table with e_june_ref in the deps (8_a7:478). This also unblocks l_code_noisematch. |
| **l_simple_key_figures** | ARG_MISMATCH (path) | many pending | simple_key_figures.py:39-40 set `LC = E/"iti_licks"/"lick_correction"` = IE/<TAG>/iti_licks/... The producer writes `il.OUT_W/lick_correction` = IE/iti_licks/<TAG>/lick_correction (lick_correction.py:67-68). fig_lick_dots (:238-240) will raise FileNotFoundError. Confirmed by grep. | Code fix: `LC = il.OUT_W/"lick_correction"`, `FIG_LC = il.OUT_W/"figures"`. |
| l_bal_offset_primary | NO_PRODUCER + UNDECLARED_DEP: l_recov_primary | l_bal_regions pending | lick_balance_offset_primary.py:43-46 reads recprim_sessions.csv **and** recprim_sessions_inside.csv with no exists-check. `_inside` is written only with WFIELD_BAL_INSIDE=1 (lick_recovery_primary.py:43,86), which no queue sets. recprim comes from l_recov_primary, which is not declared (8_a7:395). | Add the l_recov_primary dep. Then either queue an inside variant (subsets + recovery_primary with WFIELD_BAL_INSIDE=1) or skip `_inside` when it is missing. |
| **l_arrival_restnl** | NO_PRODUCER: arrival_to_cue_traces.py without --rest-variant | deps done; the old failure was archived, so it **will re-run and fail again** | arrival_to_cue_traces.py:591-593: the restnl variant skips extraction and reads OUT/arrival_to_cue/sessions/<lab>.npz (:195). The archived log shows FileNotFoundError PS92_0806.npz. | Add `l_arrival_restw` (`arrival_to_cue_traces.py --jobs 2`, deps l_enl_lick_audit,l_precue_restw) as a dep. |
| **l_restnl_rescue** | ARG_MISMATCH (path) + UNDECLARED_DEP | declared deps done; the old failure was archived, so it **will re-run** | restnl_rescue.py:258 reads `il.OUT/"precue_restnl"`, but the producer writes `restnl.cache("precue_restnl")` = precue_restnl_wr (lick_precue_restw.py:70; restnl.py:59-61). Confirmed by grep. It also needs l_precue_restnl, l_c2l_extract_restnl and l_c2l_restnl outputs (:191, :407-409), none of which are declared. The 107 stale rescue/sessions npz from 01:50 lack the c2l fields and are skipped silently (:278). | Read `il.OUT_W/restnl.cache("precue_restnl")`. Add deps l_precue_restnl,l_c2l_extract_restnl,l_c2l_restnl. Clear the rescue cache or bump VERSION (:50). |
| l_c2l_precue_restnl | UNDECLARED_DEP: l_balimg | l_c2l_extract_restnl not started | lick_cue_to_lick_precue.py:106 reads OUT_W/balimg_f0.csv unguarded. The writer is lick_balance_imaging_control.py:59 (l_balimg, not done). | Add the dep l_balimg. |
| **l_stopped_restw** (RESERVED_v9y, waiting for HEAVY.lock) | NO_PRODUCER (wrong path) + UNDECLARED_DEP: l_precue_restw | l_enl_lick_audit done | v9y stopped_trial_traces.py:887 → movement_precue (:950) reads `il.OUT.parent/"qc"/"jaw_onsets.csv"` = imaging_encoding/iti_licks/qc/ (absent). The file is at imaging_encoding/qc/jaw_onsets.csv (paths.yaml:215). It also reads OUT_W/precue_restw (:340), which is undeclared; that producer is done, but a missing file would silently set restw_ok=False. | Use `run_mode.behaviour_qc_dir()/"jaw_onsets.csv"` and add the dep l_precue_restw. |
| l_stopped_restnl | NO_PRODUCER (same path) | l_stopped_restw pending | Same :887, via :1049-1053. 9y:236. | Same fix. |
| l_disengage | NO_PRODUCER (wrong path) | l_restw_check done → ready | disengagement_tests.py:965 reads `E.parent/"cohort"/"tables"/"prepost"/"position_metrics.csv"`, which is under imaging_encoding/ and missing. The real file is LP_analysis/...v36/cohort/<TAG>/tables/prepost_working/position_metrics.csv (k_cohort, done). | Point it at the orofacial_cohort table dir for the run. |
| l_disengage_daq | NO_PRODUCER (same path) | l_disengage pending | Same :965. | Same fix. |
| l_session_levels | UNDECLARED_DEP: l_basis_extract_r + old-code SPATIAL | deps done → ready, but will fail or produce empty output | It reads OUT/basis_svdbasis/sessions (iti_where_decoder.py:161), which is absent because l_basis_extract failed 107/107. In v8, SPATIAL = ..._spont_frA_te_basis/fit (iti_decoder_patterns.py:42), which no step writes; this is fixed only in v9zg (cc70519e). | Add `l_session_levels_r` in the v9zg queue, depending on l_basis_extract_r, and repoint l_session_levels_z_r to it. Otherwise z_r will be SKIPPED. |
| l_key_figure_r2 | NO_PRODUCER: scripts/lick_same_target_place.py | l_refit and d_decoder_retention_r2 done; e_recov_meas pending | key_figure.py:79 (v9zh) reads OUT/sametarget_tests.csv. Its only writer is lick_same_target_place.py:204, which appears only in v8_missing_steps_draft.sh, not in any queue (confirmed). The step also reads matched_gauss_tests.csv (:77) from e_split_gauss_all, which is done but undeclared. | Queue lick_same_target_place.py (frozen; dep e_split_order_all) and add it plus e_split_gauss_all to the deps. |
| e_pat_stats_maps_r | UNDECLARED_DEP: l_basis_extract_r | t_frame_te_basis pending | iti_decoder_pattern_stats.py:64 → `wd.load("svdbasis")` reads basis_svdbasis/sessions. A missing file returns None and the session is dropped silently. The result is cached permanently in decoder_pattern_sessions.pkl (:326-331). It is safe today only because l_basis_extract_r (stop mode) comes first in the same serial lane (9zg:283,289). | Add the dep l_basis_extract_r. Delete the pkl if this step ever runs early. |
| e_pat_stats_maps (v9u, claimed) | ARG_MISMATCH (old code) | t_frame_te_basis pending | v9s iti_decoder_patterns.py:43 uses `encoder_run("_te_basis")`, which gives ..._spont_frA_te_basis. t_frame_te_basis writes ..._frA_te_basis (7b_a7:380,383). It already failed this way (logs/e_pat_stats_maps.log_nodep_2332). | Mark it skipped and release the claim; e_pat_stats_maps_r replaces it. |
| e_roi_foci_r2 | NO_PRODUCER: scripts/iti_spout_pattern_stats.py | chain pending | lick_roi_foci.py:29-32,59 call load_fd("lick_cued_spout_pattern_stats"), which raises SystemExit if missing (figdata.py:47-50). The writer is iti_spout_pattern_stats.py:128,242, and no queue runs it. | Queue that script as a dep, or drop the cued-spout panels. |
| e_roi_cand_r | ARG_MISMATCH (chain runs the old code); no _r2 exists | will be skipped via e_pat_stats_maps | lick_roi_candidates.py:88,93. 9u_a7:239. 9zg defines no e_roi_cand_r2. | Add `e_roi_cand_r2` to the 9zg chain. |
| e_rim_overlay_r | NO_PRODUCER: iti_licks.py with WFIELD_ROI_VERSION=2 | e_roi_cand_r (will skip) | lick_rim_overlay.py:31,48-49 hard-code `il.XOUT/"sessions_roi2"`, which is never produced (the default ROI version is 8). `pre[0]` then raises IndexError (:57). Confirmed by grep. | Use il.XOUT/il.SESS_NAME (lick times are identical across ROI versions), and add an _r2 step. |
| l_offset_where_r | NO_PRODUCER: contrib/order with WFIELD_LICK_DECODER=svd/locanmf | l_basis_frame_sd_r pending | lick_offset_where.py:44-47,131-134 need offset_contrib/terms_{locanmfFM,svdF}.csv, which is SystemExit if missing. Only the areas, areas_split and nonlinear decoders are queued. | Queue the svd and locanmf order and contrib steps after l_basis_extract_r, or limit BASES to areas_splitr8. |
| e_checkfigs (8b_a7:285) | DEP_FAILED → SKIPPED + UNDECLARED_DEP | e_iti_hemo_sum skipped; e_rlink_angle_z failed | The script reads no hemo output. It does read e_rmse output (iti_check_figures.py:121,477), decoder_retention via _days (recovery_timing.py:48, from d_decoder_retention_g1106), and the angle_z sessions now produced by e_rlink_angle_z_r (:176). | Add a new step e_checkfigs_r with deps e_iti_main,e_iti_f0,e_3axis,e_where_z,e_rmse,e_rlink_angle_z_r,d_decoder_retention_g1106. |
| g5_section_g_contact (9zh:257) | DEP_FAILED → SKIPPED | dep g5_section_g_r2 failed | deps_ok returns 1 on .failed. The step needs ordering only. | Change the dep to `after:g5_section_g_r4`, and optionally add the r4 WIDEFIELD_SESSION_CACHE token. |
| g7_section_g_figs_contact (9zh:258) | ARG_MISMATCH (output collision) | g5_section_g_contact (will skip) | It runs `--output "$FW" --suffix _contactonly`, but section_g_figures.py:236-271 writes fixed names and only the retained figure uses the suffix (:275, :287-288). It would overwrite the r4 primary PNGs. | Use `--output "$FW/contactonly"`, or `--only retained --suffix _contactonly`. |
| e_dock_compare (8c_a7:298) | UNDECLARED_DEP (silent missing rows) | many dock deps have no claim or state | lick_iti_dock_compare.py:163-219 reads balmatch_tests, ci_spout_epoch_tests, itidock_sspn_* and itifilter_notdock/matched_loclin_tests, each behind an exists() check, so missing producers just drop rows. | Add l_balmatch, l_ci_spout, e_dock_sspn0 and e_dock_loclin_notdock. |
| d_fig_goal_summary_f3 (HOLD) | UNDECLARED_DEP (wrong declared producer; file exists) | d_attempt_maps_rest done | fig_goal_summary.py:94 reads hemispheres_lick*.csv, written only by `attempt_maps --plot-only` (attempt_maps.py:382,394-395). The file on disk came from d_attempt_maps_plot_r (9r_a7). | Change the dep to d_attempt_maps_plot_r and release the hold. |
| e_mm / e_lmm_all (HOLD) | UNDECLARED_DEP + stale dep name | e_rlink_angle_z failed, e_iti_hemo_sum skipped | iti_mixed_models.py:57 reads angle_z output, now from e_rlink_angle_z_r. :70 reads hemo_check/variant_resp_sessions.csv, which is absent because e_iti_hemo failed. imm.matched reads matched_gauss_sessions.csv (:103, from e_split_gauss_all, undeclared). | Release only after e_iti_hemo and e_iti_hemo_sum re-run. Change e_rlink_angle_z to e_rlink_angle_z_r and add e_split_gauss_all. |

## B. Undeclared but harmless now

In each row the producer is already `.done`, or it runs earlier in the same serial lane. Adding the dep is housekeeping only.

| step | missing dep(s) | evidence |
|---|---|---|
| b_exec_vs_decoder, b_exec_vs_far_contra, b_exec_vs_far_mid | d_attempt_pp_lickcue (done) | exec_vs_neural_match.py:83,86 read attempt_coding_pp/<TAG>/trials_lick_*. |
| b_fr_musall_v2 | v_pixmus_noFR_spont (done) | frozen_fr_musall_v2.py:46-49 |
| b_fr_compare (8b_a7:295) | a_allsession_ref, a_allsession_frA, a_relabel_noFR (done) | frozen_fr_compare.py:262,281 |
| g6_grant_r4 | g5_joint_xsession_r, g5_pcd (done) | grant_behaviour.py:248,307; grant_confusion.py:104,339. Note: g6_grant_r4 does not match HEAVY_RE (the regex matches `epoch_grant_figures`, not `wfield_local.grant_figures`), so it runs `-j 4` unlocked. |
| gb_sessions_r, gb_summarise_r | gb_quiet_lp, gb_frcache_copy (done) | gb_common.py:82,86; gain_bio.py:268; summarise_gain_bio.py:75 |
| e_recov_meas, e_recov_marker, e_iti_lick_rate | d_decoder_retention_g1106 (done) | recovery_timing.py:48 via `_days` / `series` |
| l_bal_blind | e_split_order_all (done) | lick_balance_blind_decoder.py:241 |
| l_sspn_checks | e_contrib_split (done) | lick_sspn_checks.py:100,113 |
| l_enl_target | d_attempt_lp (done) | lick_enl_target_test.py:915. Also BOX_DEFAULT_PATH: softmax_neural_vs_lick.py:30 hard-codes `C:/Users/SabatiniLab/wfield_cache/pose3d_inputs`, bypassing WFIELD_INPUT_CACHE. It works on this box. |
| l_reembed_r | e_june_ref (done) | reembedding_test.py:61-62 |
| d_fig_goal_summary_f1 (HOLD) | no queue producer; file present | fig_goal_summary.py:65 reads region_measures.csv from O2 combined_encoding (11-05). Release the hold. |
| l_bal_contact | optional recprim_sessions_full.csv (try/except) | lick_balance_contact.py:192-198. Non-fatal. |

**Order-only dep worth removing.** l_area_maps_split and l_area_maps_split_cued depend on e_iti_area_maps but read nothing it writes (iti_area_maps.py:49,183,189). Removing that dep unblocks the l_recov_markers chain, which currently waits behind l_balimg → HEAVY.lock → e_iti_area_maps in lane v8-L0L1EL8L10.

## C. UNCLEAR

| step | note |
|---|---|
| l_bal_subsets_full | It is queued without WFIELD_BAL_FULL=1 (8_a7:389), so it writes the untagged balsub_* files, which every consumer reads. Confirm the untagged primary is intended; if `_full` is wanted, set the env on this step **and** on l_recov_primary. |
| e_lmm_response (HOLD) | Reads variant_resp_sessions.csv (iti_lmm_figures.py:64-72), which is absent until e_iti_hemo is re-run. The v8 `--deficit ttc` path is fixed on main (v9zn:188). Release after the hemo re-run. |

## D. SUPERSEDED (leave held, skip, or don't relaunch)

- **Section G re-runs:** g6_grant, g6_grant_r, g6_grant_r2, g6_grant_r3, g7_section_g_figs, g7_section_g_figs_r, g7_section_g_figs_r2, g7_section_g_figs_r3. Their section_g deps failed; the r4 versions replace them.
- **Pose validation:** g7_fig_pose_validation is replaced by k_pose_validation (.done).
- **ENL move figure:** x_fig_enl_move (RESERVED_v7) is replaced by x_fig_enl_move_v8 (.done, same command). Its `--csv` path matches where x_enl_move wrote, so there is no path mismatch.
- **e_rlink_hit:** replaced by e_rlink_hit_r (.done).
- **e_basis:** running now, but its svdbasis input is empty; e_basis_r overwrites basis_decoder.csv.
- **ROI _r steps:** e_roi_check_r and e_roi_foci_r are replaced by the _r2 versions (but see e_roi_foci_r2 in section A).
- **v8 session-level steps:** l_session_levels_z and l_levels_compare are replaced by their _r versions; the v8 copies use the old SPATIAL and the failed frame_sd.
- **Key figure:** l_key_figure and l_key_figure_r are replaced by l_key_figure_r2.
- **t_te_readout:** already satisfied. The 11-06 00:08 run exited 0 and its outputs exist; the marker was archived as `.done_nopartial_0120`. Restore `.done` or let it re-run (about 3 minutes).

## E. OK

The input files of every step below resolve to a declared producer (directly or transitively), or to existing raw or seeded inputs. Some are still blocked by a broken step upstream.

- **A/B/T/Z:** a_partial_ls (running), b_fr_musall, b_posinfo, b_matched_residual, b_partial_frA, b_peth_trial_maps, b_kernel_maps, b_r2_maps, b_spout_maps, b_spout_maps_contrast, b_resid_trend, b_rsig_all, b_rsig_all_pa, b_residual_roi_spout, b_specificity, b_svf_frA_plot, b_layer2_matched_n, b_fig_layer2, b_fig_rsplit, b_fig_resid_per_animal, z_fig_pooled (8b_a7), t_frame_te_basis (its calib input now exists), t_frame_frlag, t_contest_lines.
- **E:** e_iti_area_maps (ready; cued_wide holds 107 files), e_basis_r, e_dock_prep_gap2dock, e_dock_prep_notdock, e_dock_loclin_gap2dock, e_dock_loclin_notdock, e_dock_sspn0, e_lc_run, e_lc_summarise, e_lc_plot, e_pat_extract (v8), e_pat_fit, e_pat_summarise, e_wlc_run, e_wlc_summarise, e_wlc_plot, e_roi_check_r2.
- **L, basis and lick-correction chain:** l_basis_extract_r, l_basis_frame_sd_r, l_lc_extract (ready, unclaimed; lane L8 has not reached it), l_lc, l_pht_run, l_pht_gate, l_pht_summarise, l_pht_plot, l_pst_extract, l_pst_fit, l_pst_refit, l_pst_summarise, l_pst_plot.
- **L, balance and recovery family:** l_area_maps_split, l_area_maps_split_cued, l_recov_markers (8b), l_bal_target_model (8b), l_balmatch (8b), l_bal_reference (ready), l_balimg (waiting for HEAVY.lock), l_bal_regions, l_bal_recovery, l_bal_variants, l_bal_nested, l_recov_primary, l_bal_order, l_bal_order_primary, l_bal_offset_lmm, l_bal_angle_lmm, l_bal_maps, l_bal_explainer, l_bal_trial, l_ci_spout, l_area_tuning, l_balimg_primary.
- **L, ENL and code family:** l_enl_offset (8b), l_iti_prelick, l_offset_terms, x_enl_slide_fix_mark, x_enlv_extract, x_enlv_readers, x_enlv_curves (blocked), l_code_noisematch (blocked), l_block_compare.
- **L, rest/c2l, interhemispheric and nonlinear:** l_c2l_extract_restnl, l_c2l_restnl, l_precue_restnl, l_precue_restnl_traces, l_disengage_add (blocked), l_interhemi_rest, l_interhemi, l_interhemi_plot, l_xa, l_xa_plot, l_cv_sm, l_cv_sm_plot, l_session_levels_z_r (blocked; needs repointing), l_levels_compare_r, l_nonlinear, l_nonlinear_preds, l_nl_order, l_nl_gauss, l_nl_loclin, m_reembed_plot_r.
- **G, D, gb:** g5_section_g_r4 (running), g7_section_g_figs_r4, g5_epoch_15r_r, gb_stats_r (it failed before on the gain_stats.py:58 assert; the code is unchanged, so it relies on the moved sessions being recomputed), gb_fig_r, d_fig_goal_summary_f2 (HOLD), d_fig_goal_summary (aggregate, HOLD).

## Recurring patterns

1. **`<routed root>.parent/...` paths.** Under a run TAG these land inside imaging_encoding. Affected: l_stopped_*, l_disengage*, and the qc dir in imaging_lmm, remap_lmm and remap_checks.
2. **Old-code SPATIAL / svdbasis consumers.** l_session_levels and e_pat_stats_maps run on v8/v9s code; only v9zg has the fix.
3. **Producers that exist as scripts but are never queued.** Affected: lick_enlv_extract --regress, lick_same_target_place, iti_spout_pattern_stats, arrival_to_cue (restw), specificity_control on the spont tag, enlv code family, svd/locanmf contrib.
4. **No BOX_DEFAULT_PATH failure found.** Two secondary hard-coded reads work on this box: softmax_neural_vs_lick.py:30 (C:/.../pose3d_inputs) and face_prelaunch_aim.py:53-54 (DeepLabCut share, read-only).
