# Figures cut from the results page (analyses still exist)

Priya 2026-10-29: cut graphs with tons of tiny thin bars / unreadable figures from the results page
(https://claude.ai/artifact/Fnpn2KHknMwh3r13xADH7d), but keep a record so the analyses can be reached for when a
question needs them. The figure files, scripts and CSVs are all kept; only the page `<img>` was removed.
Outputs live under `M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/iti_licks/`
unless noted.

| figure (PNG) | was on page | what it shows | answers questions like | script / data |
|---|---|---|---|---|
| lick_area_tuning_raw.png | Fig 14.18 S8 | raw per-area lick-direction tuning slope (mixes target and landed angle), per epoch | "does area X's raw tuning to lick angle change?" | scripts/lick_area_tuning.py; areatune_*.csv |
| lick_area_tuning_decomp.png (full) | Fig 14.18 S8 | per-area offset / gain / tuning-width decomposition, all areas x 3 lick sets (replaced by `_simple`, 7 areas, cued + ITI-after-quiet) | per-area decomposition for areas or the far1 set not in the simple version | scripts/lick_area_tuning.py; areatune_*.csv |
| lick_matched_pixel_areas.png | Fig 14.18 S8 | a-priori area test for the lick-matched pixel maps (9 areas x 3 epochs x 12 panels) | "is the matched-pixel effect significant in area X?" | scripts/lick_matched_pixel_maps.py; pixmatch_*.csv |
| lick_correction_prediction.png | Fig 14.18 S9 | does activity predict the correction lick (dcvR2 ~0.01-0.04, 4 series x 12 panels); null: next-command variability, no error feedback, no post-stroke increase | "does cortex predict the next lick's correction?" | scripts/lick_correction.py; lick_correction/*.csv |
| lick_interhemi_recovery.png (full strip) | Fig 14.18 S10 | 44-category strip of within-animal r between interhemispheric coupling and recovery (replaced by `_simple` heatmap) | coupling-recovery r for a specific area / measure | scripts/lick_interhemispheric.py; interhemi_*.csv |
| lick_enl_offset_explore.png | Fig 14.21 S4 | ENL acute offset exploration: area pairs (12), control bars (10), history / face controls (8 panels) | "which area pair / control carries the ENL pre-cue offset?" | scripts/lick_enl_offset_explore.py; enl_offx_*.csv, enl_hist_*.csv |
| lick_enlv_sweep.png (full) | Fig 14.21 S10 | ENLv sensitivity: 21 event-definition variants x 7 measures (replaced by `_simple`, 7 x 4) | results for a variant not in the simple version (e.g. w05_15, on15_clean, first_corr) | scripts/lick_enlv_balance_offset.py; enlv_licks/enlv_tests.csv |
| revup_phase_shares.png (full) | Fig 14.21 S6 | rev-up trial phase shares, 1x6 panels x 16 bars incl. late_spout (replaced by `_simple`) | late_spout phase share | scripts/revup_behaviour.py; revup/revup_cells.csv |
| revup_neural_slowonly_contrasts.png (full) | Fig 14.21 S7 | slow-only contrasts, 10 areas x 4 epochs x 4 animals x 8 panels (replaced by `_simple`) | slow-trial contrast in a non-key area | scripts/revup_neural.py; revup/revup_neural_tests.csv |
| revup_late_engaged.png (full) | Fig 14.21 S8 | late licks after the spout left under 4 engagement gates (replaced by `_simple`, canonical gate) | late-lick share under the other gates | scripts/revup_followup.py; revup/revup_late_engaged*.csv |
| revup_neural_precue_vs_latency.png | Fig 14.21 S7 | pre-cue level vs first-lick latency, 6 spouts x 4 epochs x 4 animals x 10 areas (mostly rho ~ +/-0.1, no spout pattern) | "does pre-cue activity predict latency?" | scripts/revup_neural.py; revup/revup_precue_latency*.csv |

## 2026-10-29 late: secondary figures cut (Priya: "we've added way too many figures - keep the ones that show important points")

Each supplement keeps its key figure(s); these secondary views were removed from the page (files, scripts, CSVs kept; the scripts are named in each supplement's caption).

| figure (PNG) | was on page |
|---|---|
| lick_ci_spout_epoch_farcontra.png | Fig 14.18 S5 |
| lick_code_subspace_axes.png | Fig 14.18 S6 |
| lick_code_subspace_areas.png | Fig 14.18 S6 |
| lick_code_power.png | Fig 14.18 S7 |
| lick_area_tuning_curves_trial.png | Fig 14.18 S8 |
| lick_area_tuning_curves_iti.png | Fig 14.18 S8 |
| lick_matched_pixel_animals_trial.png | Fig 14.18 S8 |
| lick_matched_pixel_animals_iti.png | Fig 14.18 S8 |
| lick_interhemi_lickspecific.png | Fig 14.18 S10 |
| lick_interhemi_lead_tau0.6.png | Fig 14.18 S10 |
| lick_interhemi_xcorr.png | Fig 14.18 S10 |
| lick_interhemi_oro_animals.png | Fig 14.18 S10 |
| lick_cued_precue_restw_traces_farcontra_change.png | Fig 14.21 S2 |
| lick_cued_precue_restw_traces_change.png | Fig 14.21 S2 |
| lick_enl_target_offtarget.png | Fig 14.21 S5 |
| lick_enl_target_decodability.png | Fig 14.21 S5 |
| lick_enl_target_tracking.png | Fig 14.21 S5 |
| revup_phase_shares_simple.png | Fig 14.21 S6 |
| revup_fr_vigour.png | Fig 14.21 S6 |
| revup_recovery.png | Fig 14.21 S6 |
| revup_neural_slowonly_contrasts_simple.png | Fig 14.21 S7 |
| revup_jaw_metrics.png | Fig 14.21 S8 |
| revup_jaw_traces_cue.png | Fig 14.21 S8 |
| lick_enlv_perspout.png | Fig 14.21 S10 |
| lick_enlv_sweep_simple.png | Fig 14.21 S10 |
| lick_enlv_first_later.png | Fig 14.21 S10 |
| lick_enlv_arrival.png | Fig 14.21 S10 |
| lick_enlv_curves_angle.png | Fig 14.21 S10 |
| stopped_class_stopped_cue_restw.png | Fig 14.21 S11 |
| stopped_compare_jaw_vs_nojaw_cue_restw.png | Fig 14.21 S11 |
| stopped_compare_jaw_aligned_jaw_restw.png | Fig 14.21 S11 |
| stopped_examples_PS93.png | Fig 14.21 S11 |
| lick_cue_to_lick_A_level_own_contra.png | Fig 14.21 S12 |
| lick_cue_to_lick_B_farcontra_on_pre.png | Fig 14.21 S12 |
| lick_cue_to_lick_C_w03_contra.png | Fig 14.21 S12 |
| lick_cue_to_lick_D_raw_traces.png | Fig 14.21 S12 |
| arrival_to_cue_per_spout_enl.png | Fig 14.21 S13 |
| arrival_to_cue_heatmap_PS92_all.png | Fig 14.21 S13 |
| arrival_to_cue_heatmap_pooled_far_contra.png | Fig 14.21 S13 |
| lick_cue_to_lick_B_pre_models.png (32 panels) | Fig 14.21 S12 | full threshold test per area x hemisphere (replaced by lick_cue_to_lick_threshold_simple.png, pooled contra orofacial) |
| lick_cue_to_lick_A_level_contra.png (8 areas x 7 spouts) | Fig 14.21 S12 | replaced by lick_cue_to_lick_A_level_simple.png (4 areas, all spouts + far contra) |
| lick_cue_to_lick_D_decomposition.png (16 x 5) | Fig 14.21 S12 | replaced by lick_cue_to_lick_D_simple.png (6 areas x 3 columns) |
| lick_enlv_curves.png, lick_enlv_curves_gain.png | Fig 14.21 S10 | replaced by lick_like_for_like_all.png (all six lick sets + gain/offset) |
| lick_acad_midline_control.png (whole supplement 12.17 S2, NON-PAGE SUPPLEMENTAL, Priya 10-29) | Fig 12.17 S2 | is the ACAd left/right result real cortex? midline-distance, vessel, registration, neighbour controls: undetermined, not ACAd-specific (sinus strip; shares variance with MOs / SSp-n) | scripts/lick_acad_midline_control.py |

## 2026-10-29 late: crowded-bar scan of the whole page (many thin bars; key numbers stay in the captions)

| figure (PNG) | was on page | what it shows |
|---|---|---|
| lick_recovery_markers_cv.png | Fig 14.18 S14 | LOAO / blocked cvR2 grids for the retired balance definition |
| lick_balance_variants.png | Fig 14.18 S14 | cvR2 across balance definitions |
| lick_recovery_markers_areas.png | Fig 14.18 S15 | per-area contra / ipsi / size / balance cvR2 |
| lick_offset_where_bars.png | Fig 11.16 | offset carried per area family x side x basis |
| lick_sspn_checks.png | Fig 12.17 S1 | is SSp-n special: refit without SSp-n, weight-free area comparison, face |
| lick_balance_contact.png | Fig 14.18 S2 | contact vs miss first licks (balance not mainly touch reafference) |
| lick_recovery_markers_iti_axis.png | Fig 14.18 S16 | ITI licks on their own recovery axis, cvR2 |
| lick_enlv_compare.png | Fig 14.21 S10 | ENLv vs cued vs ITI (all / docked / moving) offsets and balance, replaced by lick_like_for_like_all |
| lick_iti_dock_compare.png | Fig 14.18 S11 | ITI results on docked vs all / moving licks (docked versions of the main ITI figures coming) |
| lick_matched_pixel_trial.png / _iti.png | Fig 14.18 S8 | replaced by *_epochs.png (raw pre / acute / subacute / chronic + change maps) |
| lick_area_tuning_decomp_simple.png | Fig 14.18 S8 | replaced by lick_area_tuning_decomp_dots.png |
| lick_interhemi_noisecorr.png, lick_interhemi_recovery_simple.png | Fig 14.18 S10 | replaced by lick_interhemi_simple.png (orofacial vs VISpm / RSPd, cued vs rest, recovery link) |
| lick_correction_behaviour.png (full, relabelled) | Fig 14.18 S9 | replaced by lick_correction_behaviour_compact.png (pre vs chronic, all spouts + far contra) |
