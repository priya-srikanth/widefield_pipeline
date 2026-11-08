# Figure rules audit, 2026-11-03

Priya 11-03: *"can cohort metrics minus pre contra (and similar figures) (1) have no lines connecting spout positions
-- review all figures across pages to ensure this -- and (2) have faded dots for session data with each? can increase
lateral offset between epochs if needed"*.

Rules applied (categorical x = spout position or epoch):

1. No line joins categories (spouts or epochs). An epoch summary = marker + interval. Time-continuous x (ms from cue,
   lick phase, days) keeps its traces.
2. Every summary marker shows the session values behind it as faded OPEN markers: shape = animal, color = epoch,
   alpha ~0.4, seeded jitter.
3. Epochs within a spout are spread laterally (`cat_points.epoch_offsets`, 0.26 apart for 3 epochs). Canonical spout
   order, stars in the top band (`sig_marks.star_top` + `reserve_headroom`), + = contralesional.

Shared code: **`wfield_local/cat_points.py`** (`epoch_offsets`, `session_dots`, `summary_point`, `add_animal_entries`)
-- use it for any new categorical-x summary figure.

Re-renders read stored tables only (no statistics recomputed); each re-render asserts that the numbers it plots / the
sidecar CSVs are unchanged.

<!-- TABLES BELOW -->

## Kinematics page (DKcKRJKbhBoratxxRc6qZv) -- NOT published here (another helper owned it); new PNGs beside the originals

All checked: every plotted mean / SEM equals the pre-11-03 code (commit 5371c420) on the same stored tables (exact),
summary values written as a sidecar `<name>_nolines.csv`. Code: 25b6cd4a, 910a3ab9, 7144bea3.

| page fig | page file | new file (`M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/cohort/figures/`...) | script : function | status |
|---|---|---|---|---|
| 1.1 | k_metrics_contra.png | prepost/cohort_metrics_minus_pre_contra_nolines.png | wfield_local/orofacial_cohort.py : fig_cohort_scalars (+ _stars) | FIXED |
| 1.2 | k_kin_pos.png | prepost/cohort_epochs_by_position_kinematics_nolines.png (+ _contra_nolines, the + = contra version) | orofacial_cohort : fig_cohort_epochs_by_position | FIXED (session dots now the default) |
| 1.3 | k_acc_pos_contra.png | prepost/cohort_epochs_by_position_accuracy_contra_nolines.png | same | FIXED (deck slide 25 swapped by the kinematics helper) |
| 1.6 | k_aim_first_all.png | -- | aim_first_vs_all.py (scratch only, not in the repo) | NOT FIXED: script not in the repo |
| 2.9 | l_pos_first_contra.png | prepost/by_position_licks_first_contra_nolines.png | orofacial_cohort : _per_animal_figure / _by_pos_panel | FIXED |
| 2.10 | l_pos_all_contra.png | prepost/by_position_licks_all_contra_nolines.png | same | FIXED |
| 3.1 | k_angle.png | prepost/angle_tuning_contra_nolines.png | orofacial_cohort : fig_angle_tuning | FIXED |
| 5.1 | k_jaw_epochs.png | jaw_lateral/epochs_nolines.png | wfield_local/orofacial_jaw_lateral.py : fig_epochs (`--plot-only --suffix _nolines`) | FIXED (FAR_BIAS panel still labeled + = ipsi: target sign, unchanged) |
| 8.1 | k_w2.png | w2/epochs_by_position_nolines.png | wfield_local/orofacial_w2.py : figures (`--plot-only`) -> fig_cohort_epochs_by_position | FIXED |
| 10.1 | pose_validation_sessions.png | -- | (cam4 2-D -> 3-D per animal; x = method, not spout / epoch) | borderline, left |
| style only | 1.4 k_aim_spout, 3.6 p_angle, 7.5 p_3d_contra | -- | session dots filled, colored by animal | not changed |

Shared fixes in `orofacial_cohort._tidy`: a legend moved next to the last panel of its row is no longer rebuilt from that
panel's own artists, and explicit legend entries survive the move. `_spout_markers` no longer joins the spout-angle bars.


## FaceRhythm page (X8NDdQNHi3zMP77PBrZAC1, read at 1793125970-97fe) -- re-rendered IN PLACE (old = `<stem>_superseded_1103.png`); NOT yet republished

R = `M:/MICROSCOPE/Priya/DeepLabCut/Widefield/face_rhythm/cam4/readouts/master_pre_v1/`, V = `M:/MICROSCOPE/Priya/DeepLabCut/Widefield/face_rhythm/cam4/validation/master_pre_v1/lick_epochs/`. All stored CSVs byte-identical (sha256). Commits 9f228199, 4376d3ba, f1053aa4.

| fig | file | script : function | status |
|---|---|---|---|
| 1.1, 2.1, 5.1-5.4 | R/peak/{peak_side_headline, peak_position_headline, peak_cohort_by_epoch, peak_cohort_by_epoch_side, timing_by_epoch_com_side, timing_by_epoch_com_position}.png | face_rhythm_peak : headline / cohort / timing | FIXED (canonical spout order) |
| 2.2, 2.3, 2.4 | V/{per_lick_stats, per_lick_by_epoch, licks_per_trial}.png | face_rhythm_lick_epochs : per_lick_stats / main | FIXED |
| 3.1 | R/asym_aligned/asym_aligned_change.png | face_rhythm_asym_aligned : main | FIXED |
| 3.5, 3.11 | R/aligned_maps/halves_{cue,lick}.png | face_rhythm_aligned_maps (--halves-only) | FIXED |
| 3.13 | R/asym_aligned/bout_split/bout_split_pre_window.png | face_rhythm_bout_split : main | FIXED |
| 3.14, 8.19 | R/matching_spatial_vector_best/{asymmetry_two_ways, match_by_day_pooled}.png | face_rhythm_match | FIXED |
| 4.1, 4.2 | R/{evr_epochs, evr_own_vs_frozen}.png | face_rhythm_evr_epochs / face_rhythm_readouts | FIXED |
| 6.1 | R/enl_decode/fr_enl_decode.png | fr_enl_decode : figure | FIXED |
| 7.3 | R/session_shape/session_shape_summary.png | face_rhythm_session_shape : figures | FIXED |
| 7.10, 7.11, 8.16, 8.18 | R/epoch_shape_cohort/, R/epoch_shape/ epoch_shape_{asymmetry,angles}.png | face_rhythm_epoch_shape : main | FIXED (lines removed; one fit per epoch, rule 2 n/a) |
| 7.12, 8.15, 8.17 | -- | drift_floor / epoch_shape pooled | pass rule 1; no session level |
| 9.1-9.4 | R/{task_vs_nontask_pooled, task_vs_nontask}.png, R/nontask_state/nontask_state.png, R/push_check/push_check_pooled.png | face_rhythm_readouts / nontask_state / push_check | FIXED |

## Imaging page (PvraHiF82BFXur1hnNUCeV) -- new `*_nolines.png` beside the originals; NOT yet republished

| fig | new file | script | status |
|---|---|---|---|
| 9.6-9.8 | `M:/MICROSCOPE/Priya/Widefield/labcams/grant_figures/epoch/cd_loo/cd_loo_pooled_window_{precue,cue,lick}_nolines.png` | scripts/cd_loo_pooled.py --plot-only | FIXED (2dbae096, b5a9d889) |
| 9.1 | `M:/MICROSCOPE/Priya/Widefield/labcams/grant_figures/epoch/cd_cim_geometry_lick_cortexonly_vs_ceiling_nolines.png` | scripts/cd_geometry_figure.py --suffix | FIXED (per animal only) |
| 12.1 | `M:/MICROSCOPE/Priya/Widefield/labcams/grant_figures/epoch/epoch_15f_rest_frozen_restdock05_durmatched_gapmatched_nolines.png` | rest_frozen_decoder.py --from-csv | FIXED |
| 3.3 | M:/MICROSCOPE/Priya/Widefield/imaging_page_src/s_per_animal_regen_20261102/s_per_animal_nolines.png | scripts/imaging_page/s_per_animal.py | FIXED |
| 2.1 d_within_precue | `M:/MICROSCOPE/Priya/Widefield/labcams/grant_figures/epoch/epoch_acc_by_position_precue_working_nolines.png` | epoch_figures.bar_row via scripts/replot_epoch_from_sidecars.py | FIXED (282dea74, cd4b32c8) |
| 2.2 d_enl1s | `M:/MICROSCOPE/Priya/Widefield/labcams/grant_figures/DAQ_ENL1s/epoch/epoch_acc_by_position_precue_working_nolines.png` | same | FIXED |
| 2.3 d_acc_cue | `M:/MICROSCOPE/Priya/Widefield/labcams/grant_figures/epoch/epoch_acc_by_position_cue_working_nolines.png` | same | FIXED |
| 2.4 d_lp_lick | `M:/MICROSCOPE/Priya/Widefield/labcams/grant_figures/LP_licks/epoch/epoch_acc_by_position_lick_lick_nolines.png` | same | FIXED |
| 2.4b-2.8 (LPfinal variants) | `M:/MICROSCOPE/Priya/Widefield/labcams/grant_figures/LPfinal/trained_{lp,daq}/<variant>/epoch/epoch_acc_by_position_precue_working_nolines.png` (2.8 = trained_lp/LP_licks_gateLP) | same | FIXED |
| 3.1 d_state | `M:/MICROSCOPE/Priya/Widefield/labcams/grant_figures/epoch/epoch_13n_state_vs_position_cue_nolines.png` | epoch_state._fig_13_state | recoloured only: no per-session values stored, NO session dots |
| 3.2 d_state_delta | `M:/MICROSCOPE/Priya/Widefield/labcams/grant_figures/epoch/epoch_13delta_state_decoder_cue_nolines.png` | contrast_panel (+ per-session deltas) | FIXED |
| 4.2 d_refitgap | `M:/MICROSCOPE/Priya/Widefield/labcams/grant_figures/epoch/epoch_5rmgapdelta_frozen_vs_refit_cue_working_nolines.png` | contrast_panel | FIXED |
| 6.1 e_amp | `M:/MICROSCOPE/Priya/Widefield/labcams/grant_figures/epoch/epoch_11amp_encoder_amplitude_cue_working_nolines.png` | bar_row | FIXED |
| 6.2 e_ceiling | `M:/MICROSCOPE/Priya/Widefield/labcams/grant_figures/epoch/epoch_11c_encoder_ceiling_cue_working_nolines.png` | bar_row | FIXED |
| 6.3 e_captured | `M:/MICROSCOPE/Priya/Widefield/labcams/grant_figures/epoch/epoch_11cfrac_encoder_captured_by_position_cue_working_nolines.png` | bar_row | FIXED |
| 10.1 r_crossnobis | `M:/MICROSCOPE/Priya/Widefield/labcams/grant_figures/epoch/epoch_8diagdelta_matrices_crossnobis_cue_working_nolines.png` | contrast_panel | FIXED |
| 5.2 | fig2b_attempt_decoder_epochs | scratch fig_goal_epochs.py (encoding window) | NOT FIXED: script not in repo |

Shared changes (epoch_figures, all epoch families on their next render): `bar_row` session dots open / shape = animal / color = epoch; stars via `sig_marks.star_top` in alternating rows (overlap fixed); `contrast_panel(points=)` per-session deltas; `EPOCH_GREY` = `orofacial_cohort.EPOCH_LINE` (gray ramp replaced). Not re-rendered (no sidecar path; recolour on the next normal render): transfer_matrix.epoch_color users (rotation_maps), cd_trajectories.EPOCH_COLOR users.

## Encoding page section 23 (this window): `*_nolines.png` beside originals, commit 458f128b
23.4 `.../v2_quiet_lp/stillness/quiet_stillness_nolines.png`; 23.5 `.../task_events/t13_precue_event_vs_outcome_nolines.png`;
23.13 `.../hemi_amp_recovery/quiet_hemi_amp_recovery_nolines.png`; 23.15 `.../task_events/t12_task_window_event_rates_nolines.png`;
23.22 `.../cohort1_quiet_spikes/attempt_effort/attempt_effort_nolines.png`; 23.23 `.../task_events_followup/t13_followup_exclusion_nolines.png`
(root `M:/MICROSCOPE/Priya/Widefield/labcams/cohort1_quiet_spikes/`). 23.5 / 23.23: no per-session values stored.

## Encoding page (sabatinilab-34): not re-plotted here

Code is in `C:/Users/SabatiniLab/Github/wf_attempt/scripts/` (branch analysis/attempt-coding), in analysis-specific
scripts (no shared plotting module), so these figures are for window 34 to fix. R1 = lines across categories, R2 = no session dots.

### Figs 13.10 - 14.18 (+ supplements)

| script : function | fig | rule |
|---|---|---|
| iti_overcorrection : plot(angle=True) | 13.10 | R1 + R2 |
| iti_hemo_check : draw_variants | 14.3 | R1 + R2 |
| lick_balance_target_model : plot | 14.x | R1 + R2 |
| lick_balance_matched_error : plot | 14.x | R1 + R2 |
| lick_ci_spout_epoch : plot (epoch lines across the 6 spouts) | 14.x | R1 + R2 |
| lick_correction : plot (both functions of that name; rows with lines across epochs) | 14.x | R1 + R2 |
| lick_balance_order_primary : _plot; lick_balance_order : plot | 14.14 / 14.15 | R1 + R2 (R1 only if lick-order bins 0..9+ count as categories) |
| lick_cued_ramp : _draw_spout_windows | 14.7 | R2 |
| lick_vigor_perspout : plot | 14.8 | R2 |
| lick_lateral_ratio : plot / plot_sides / plot_rings | 14.9 - 14.12 | R2 |
| lick_code_subspace : figures | 14.x | R2 |
| lick_code_rotation_summary : main | 14.x | R2 (+ minor R1: per-animal null-to-observed lines) |
| lick_code_noisematch : figures | 14.x | R2 (+ same minor R1) |
| lick_area_tuning : plot_decomp_dots | 14.x | R2 |
| simple_key_figures : fig_lick_dots | 14.x | R2 |
| lick_interhemispheric : plot_simple | 14.x | R2 |
| interhemi_crossarea : plot1 / plot2 | 14.x | R2 |
| coupling_vigour_sm : plot1 | 14.x | R2 |
| (borderline, passed) lick_recovery_primary* cross-validated R2 bars | 14.18 | one held-out value per bar |

### Figs 14.18 - 14.21 (+ supplements)

| script : function | rule |
|---|---|
| sm_coherence_cca : plot_itpc / plot_ctrl | R1 |
| sm_direction_lag : plot_lags | R1 |
| learned_compensation : fig6_t3 (lines joining the 6 spouts) | R1 |
| prelick_hemisphere_tuning : fig_primary (lines across epochs) | R1 |
| lick_enl_offset_explore : figure_a | R1 |
| coupling_vigour_sm : plot2 | R2 |
| kn_hits_deviation : A | R2 |
| prelaunch_aim_tracking : cmd_plot | R2 |
| face_prelaunch_aim : plot_all | R2 |
| face_offset_partial | R2 |
| disengagement_tests : _delta_panel / _pair_panel (t1, t3, t4 movement + neural, a2, a3, t4b) | R2 |
| lick_balance_trial | R2 |
| lick_balance_imaging_control(_primary) | R2 |
| lick_enl_target_test : _fig_classmatched_epochs | R2 |
| (style only) lick_enl_angle_decode, lick_balance_recovery, lick_recovery_markers_size, lick_*_recovery_search | session dots filled, colored by animal |

### Key figures, Figs 0.1 - 5.2

**Root cause for most of these:** two helpers in `scripts/frozen_fr_compare.py` -- `epoch_lines` (lines across epochs,
per-animal faint lines instead of session dots) and `bars` (gray dots = animals, not sessions). At least 9 encoding
scripts import them (contest_lines, te_readout, partial_refit, frozen_claims, frozen_fr_musall, frozen_posinfo, ...).
Fixing the two helpers fixes most rows below. Not in `wfield_local`.

| fig | image | rule | script : function |
|---|---|---|---|
| Key 1 | key_figure.png | R1 + R2 | iti_licks : key_figures |
| Key 2 | lick_order_iti_cued_slope_docked.png | R2 (cols 4-5) | iti_lick_order : plot_slope |
| 0.1, 0.7 | frozen_fr_compare_lines_{cue,lick}.png | R1 + R2 | frozen_fr_compare : lines_figure (epoch_lines) |
| 0.2, 0.8 | frozen_musall_{cue,lick}_simple_v2.png | R2 | frozen_fr_musall : main (bars); cited frozen_fr_musall_v2.py not in repo |
| 0.3, 0.4, 0.9, 0.10 | frozen_claims_{contest,peth}_{cue,lick}_simple.png | R2 | frozen_claims : main (bars) |
| 1.2 | te_unique_lines_frame.png | R1 + R2 | te_readout : main (epoch_lines) |
| 1.3, 1.4 | te_partial_refit_lines.png, partial_refit_lines.png | R1 + R2 | partial_refit : summarize (epoch_lines) |
| 1.5, 4.7 | contest_landing_vs_target.png, contest_selffit_remap.png | R1 + R2 | contest_lines : figure (epoch_lines) |
| 3.11 | epoch_kernels_r.png | R1 + R2 | epoch_kernels : figures |
| 1.6, 1.15 | fig1p_contest_pooled.png, fig10p_width_pooled.png | R2 | no local source (O2 script) |
| 1.13 | fig1_binned_contest.png | R1 | no local source (O2) |
| 1.14 | fig10_softmax_width.png | R2 | no local source (O2) |
| 2.4, 2.5 | frozen_posinfo_split_{cue,lick}.png | R2 | frozen_posinfo : main (bars) |

### Figs 19.5 - 23.24

| fig | image | rule | script : function |
|---|---|---|---|
| 19.5 | fig5b_layer2_random_splits.png | R2 (borderline: dots are 20 random-split fits) | plotting code not found; data from layer2_random_splits : main |
| 19.6 | fig6_layer2_groups.png | R2 | plotting code not found |
| 20.1 | fig14_musall_task_aligned.png | R2 | frozen_musall_figures : main (musall_task_aligned_*) |
| 20.2 | fig15b_musall_alone_unique_lines.png | R1 + R2 | frozen_musall_figures : main (musall_alone_vs_unique_lines_*) |
| 21.1, 21.5, 21.6 | fig16_fr_{cortex,SSp-m_left,SSp-m_right}.png | R2 | fr_encoding_summary : main |
| 21.4, 21.11 | fr_trial_peth_{chronic,pre}.png | R2 | fr_trial_peth : main |
| 21.7, 21.8 | fr_musall_cortex_{cue,lick}_modeA.png | R2 | fr_musall : main |
| 22.1 | fr_direction_mechanism_summary.png | R2 | fr_direction_mechanism : summary_figure |
| 22.2 | contest_fr_lags.png | R1 + R2 | contest_lines : figure |
| 23.4 | q_stillness.png | R2 (borderline; x = face-speed bins) | cohort2_qc/quiet_stillness : figure |
| 23.5 | q_task_outcome.png | R2 | cohort2_qc/quiet_task_events : t13_figure |
| 23.13 | q_hemi_amp.png | R2 | cohort2_qc/quiet_hemi_amp_recovery : figure |
| 23.15 | q_task_rates.png | R2 (bottom-right panel) | cohort2_qc/quiet_task_events : t12_figure |
| 23.22 | q_effort.png | R2 | cohort2_qc/attempt_effort : figure |
| 23.23 | (inline image) | R2 | cohort2_qc/quiet_task_events_followup : figure |

### Figs 5.3 - 11.13

| Fig | png | rules | script:function | note |
|---|---|---|---|---|
| 6.2 | fig9i_cls_animals | 2 | coupling_by_class.py:main | animal markers only |
| 6.4 | fig9m_offtarget | 2 | exec_vs_neural_match.py:main | |
| 6.6 | fig9h_trial_by_landing | 2 | trial_neural_vs_exec.py:main | |
| 6.10 | fig9n_offtarget_by_spout | 2 | exec_vs_neural_match.py:main | stacked bars, no dots |
| 7.5 | fig3z_encoder_offtarget_by_spout | 2 | same as 6.10 (src = encoder) | |
| 8.1 | fig2_attempt_decoder | 2 | SCRATCH fig_goal_summary.py | per-animal bars |
| 8.2 | fig2b_attempt_decoder_epochs | 1+2 | SCRATCH fig_goal_epochs.py | class lines across epochs |
| 8.3 | fig2c_decoder_pooled | 2 | SCRATCH fig_pooled.py | |
| 9.1 | fig3_attempt_amplitude | 2 | SCRATCH fig_goal_summary.py | |
| 10.2b | lick_refit_decoders | 1+2 | lick_refit_decoders.py:plot | "o-" across epochs |
| 10.3 | lick_direction_decoding | 1+2 | iti_licks.py:direction_draw | right column |
| 11.1 | lick_check_3axis | 1 | iti_check_figures.py:epoch_draw | animal + mean lines across epochs |
| 11.9 | lick_where_angle_reach | 1 | iti_check_figures.py:epoch_draw | |
| 11.10 | lick_matched_angle_reach | 1 | iti_check_figures.py:epoch_draw | |
| 11.13 | lick_area_unique_maps_iti / _cued | 2 | lick_area_unique_maps.py:plot | BLOCKS bars |

Fixing `iti_check_figures.epoch_draw` covers 11.1, 11.9 and 11.10 at once.

Borderline: 10.6 lick_iti_rate. Its pooled mean line is drawn across epochs on a days axis.
The goal figures fig2* and fig3_attempt_amplitude come from one-off scratch scripts that are not in the repo (C:/Users/SabatiniLab/AppData/Local/Temp/claude/C--Users-SabatiniLab/cdd1e016-6bb3-4346-8600-f08116ccf61d/scratchpad/). Fixing iti_check_figures.epoch_draw covers 11.1, 11.9 and 11.10.

### Figs 11.14 - 13.9

| Fig | png | rules | script:function | note |
|---|---|---|---|---|
| 11.14 | lick_block_compare | 2 | lick_block_compare.py:plot | animal dots only |
| 11.14 + supp | lick_block_ladder_shapley(_oroSSs) | 1+2 | lick_block_compare.py:plot (2nd savefig) | ladder lines over area-block steps |
| 11.20 | lick_nonlinear_decoder | 1+2 | lick_nonlinear_decoder.py:plot | lines across epochs |
| 11.21 | lick_behaviour_decoder | 1+2 | lick_behaviour_decoder.py:plot | lines across epochs, no dots |
| 12.14 | lick_iti_filter_compare | 2 | lick_iti_filter_compare.py:plot | animal markers only |
| 12.15 + supp | lick_sametarget_place(_nonlinear) | 1+2 | lick_same_target_place.py:draw | COVERAGE column |
| 12.16 | lick_covref_offset | 2 | lick_covariate_reference.py:draw | |
| 12.17 | lick_offset_terms_recovery | 2 | lick_offset_terms_recovery.py:plot | top row bars with no dots (cvR2 row borderline) |
| supp 12.17 | lick_iti_prelick | 1+2 | lick_iti_prelick.py:plot | row 3 window bins |
| 12.18 | lick_balance_reference | 2 | lick_balance_reference.py:plot | |
| 12.20 | lick_balance_regions | 2 | lick_balance_regions.py:plot | middle panel |
| 12.21 | balblind_offset / balblind_subsets | 2 | lick_balance_blind_decoder.py:plot | |
| 13.1 + supp | lick_order_iti_cued(_nonlinear) | 2 | iti_lick_order.py:plot_classes | |
| 13.2 | lick_order_iti_cued_slope_nonlinear | 2 (+1 borderline: lick-order bins) | iti_lick_order.py:plot_slope | |
| 13.4 | lick_cued_first_lick | 1+2 | iti_first_lick.py:plot | lines across epochs |
| 13.5 | lick_cued_overcorrection | 1+2 | iti_overcorrection.py:plot | animal lines across epochs |
| 13.7 | lick_position_per_spout(_byepoch) | 2 | lick_position_per_spout.py:plot / plot_by_epoch | |
| 13.8 | lick_decoded_decomposition | 1+2 | iti_decoded_decomposition.py:plot | |
| 13.9 | lick_cued_misdirected_licks | 2 | cued_misdirected_licks.py:plot | row 1 only |


### Supplements after 14.21 - Fig 15.12

| fig | image | rule | script : function |
|---|---|---|---|
| supp 14.21 | revup_late_engaged_simple.png | R2 | wf_attempt revup_followup : plots |
| supp 14.21 | enl_lick_audit.png | R2 | wf_attempt enl_lick_audit : figure |
| supp 14.21 | lick_enlv_target.png | R1 (panel 6) + R2 | wf_attempt lick_enlv_target_code : plot |
| supp 14.21 | lick_like_for_like_all.png | R2 | wf_attempt lick_enlv_curves : fig_all |
| supp 14.21 | stopped_precue_levels_restnl.png | R1 + R2 | wf_attempt stopped_trial_traces : draw_levels |
| supp 14.21 | lick_cue_to_lick_A_level_simple_restnl.png | R2 | wf_attempt lick_cue_to_lick : draw_A_simple |
| supp 14.21 | lick_cue_to_lick_D_simple_restnl.png | R2 | wf_attempt lick_cue_to_lick_precue : draw_decomp_simple |
| supp 14.21 | restw_check_results_moved.png | R2 | wf_attempt restw_strict_check : draw |
| 14.24 | lick_ramp_baseline_sens.png | R2 | wf_attempt lick_cued_ramp : baseline_sens |
| 14.32 | lick_iti_check_f0.png | R1 | widefield_pipeline scripts/iti_check_figures.py : f0 |
| 14.34 | lick_iti_hemo_check_edge_vessel.png | R1 + R2 (bottom row) | widefield_pipeline scripts/iti_hemo_check.py : fig_edge_vessel |
| 14.35 | lick_iti_hemo_check_state_gap.png | R1 + R2 | widefield_pipeline scripts/iti_hemo_check.py : fig_state_gap |
| 15.3 | lick_session_shape_d.png | R2 | widefield_pipeline scripts/iti_session_cluster_tests.py : plot_shape_d |

### Figs 15.13 - 19.4

| fig | image | rule | script : function |
|---|---|---|---|
| 16.3 | lick_recovery_link_pooled_aim_z_docked.png | R1 + R2 | scripts/iti_recovery_link.py : plot_pooled |
| 17.1 | fig4p_decomp_pooled.png | R2 | producer not tracked (frozen_encoder_gate-attempted_v7g3_t3d inputs) |
| 17.2 | fig4_decomp_epochs.png | R2 | producer not tracked |
| 18.3 | fig12p_residual_trend_pooled.png | R2 (epoch side bars) | producer not tracked (nearest frozen_residual_trend.py) |
| 19.1 | fig5p_layer2_pooled.png | R2 | producer not tracked (inputs from frozen_fr_compare layer2_change_vs_prelate.csv) |
| 19.2 | fig5c_layer2_random_splits_pooled.png | R2 (borderline: dots = random splits) | producer not tracked |
| 19.3 | fig6p_layer2_groups_pooled.png | R2 | producer not tracked |

Deck specs: `build_inputs/figure_rules_swaps.json` (slides 24, 33, 57, 100; applied) and `figure_rules_swaps_2.json`
(slides 56, 58, 10 d_acc_cue, 19 d_lpf_enl1_lp), crops by `build_inputs/figure_rules_crops.py [2]`.
