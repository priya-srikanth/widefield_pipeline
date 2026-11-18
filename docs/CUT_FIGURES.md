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
| lick_acad_midline_control.png (whole supplement 12.17 S2, NON-PAGE SUPPLEMENTAL, Priya 10-29) | Fig 12.17 S2 | is the ACAd left/right result real cortex? midline-distance, vessel, registration, neighbor controls: undetermined, not ACAd-specific (sinus strip; shares variance with MOs / SSp-n) | scripts/lick_acad_midline_control.py |

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

## 2026-10-29 late (Priya approved): images removed from numbered figures (caption + numbers kept on page)

| figure (PNG) | Fig |
|---|---|
| spec_groups.png | see caption |
| matched_residual.png | see caption |
| selffit_cue.png | see caption |
| selffit_lick.png | see caption |
| te_selffit_lick.png | see caption |
| lick_area_importance.png | see caption |
| fr_direction_areas_cue.png | see caption |
| fr_direction_areas_lick.png | see caption |
| frozen_musall_cue/lick, frozen_claims_contest_cue/lick, frozen_claims_peth_cue/lick (.png) | Fig 0.2/0.8, 0.3/0.9, 0.4/0.10 | replaced by *_simple.png (key areas; note SSp-n absent from saved tables; unique-R2 table has only cortex / SSp-m / MOp) |
| partial_refit_bars_cue.png | Fig 1.12 | image removed; numbers in caption + partial_refit_cue_numbers.md |
| lick_correction_spouts.png (4 heatmaps) | Fig 14.18 S9 | replaced by lick_correction_spouts_dots.png |

## 2026-10-30

| figure (PNG) | was on page | what it shows | answers questions like | script / data |
|---|---|---|---|---|
| early_predictors.png | Fig 0.13 (Priya: "not useful") | early features (lesion %, day-1 residuals, acute retention, coupling) vs chronic outcomes, Spearman per animal, n=4; only lesion size orders recovery speed perfectly | "does lesion size / day-1 signal predict recovery?" | scripts/early_predictors.py |

## 2026-11-16: Cortical Spout Coding page (IMG, https://claude.ai/artifact/PvraHiF82BFXur1hnNUCeV), v53 / v54

Priya 11-16 (caption pass Q9: "yes, off the page, into CUT_FIGURES; files kept"): the DAQ ENL 2 s arms (retired), the
DAQ-trained frozen-decoder twins of the LP figures, the DAQ spout-contact map twins of the LP maps (B11), and the pre-cue
Haufe maps came off the IMG page. Files below are the page's published names (the page's source copies; the producing
scripts and their CSVs are kept). Every cut figure is on DAQ spout contact; the LP-lick / camera working-trial version
named in the last column stays on the page.

| figure (PNG) | was on page | what it shows | answers questions like | script / data |
|---|---|---|---|---|
| d_refit_precue.png | IMG Fig 1.1 | within-session refit 6-way decoder, pre-cue 2 s (DAQ ENL), per position x epoch; no acute loss at any position, far contra ~0.3-0.4 in every epoch | "is the low pre-cue far-contra code specific to the frozen decoder?" | wfield_local/epoch_accuracy.py (`_frozen_vs_refit`) on grant_confusion._refit_pred; epoch family 5r (10-07) |
| d_within_precue_nolines.png | IMG Fig 2.1 | frozen pre-stroke decoder, pre-cue 2 s (DAQ ENL); far contra low even at baseline (pre ENL .51, cue .90, lick .95) | "how decodable is position before the cue on the 2 s window?" | wfield_local/epoch_accuracy.py; on the page: LP 1.0 / 1.5 s ENL (Figs 2.6, 2.8) |
| d_enl1s_nolines.png | IMG Fig 2.2 | frozen decoder, DAQ ENL 1 s; far contra roughly halves acutely 4 / 4 (.29 -> .15, .33 -> .16, .54 -> .30, .40 -> .14) | ENL 1 s far-contra loss on the DAQ lick definition | wfield_local/epoch_accuracy.py (arm grant_figures/DAQ_ENL1s); on the page: Fig 2.8 |
| d_lp_lick_daq.png | IMG Fig 2.4b | frozen decoder, lick window, LP lick gate, decoder trained on DAQ-contact pre trials: acute far contra .90 -> .26, 4 / 4 (first run 11-02) | "does the Fig 2.4 lick-window loss depend on which lick definition trained the decoder?" (no) | wfield_local/epoch_accuracy.py (WIDEFIELD_LICK_SOURCE=lp, WIDEFIELD_EPOCH_FROZEN=1, stored frozen decoders) |
| d_lpf_enl15_daq_nolines.png | IMG Fig 2.5 | final LP arm, ENL 1.5 s, DAQ-trained frozen decoder: far contra .48 -> .21 acute **, 4 / 4 | DAQ-trained twin of Fig 2.6 | scripts/lp_final_render.py decoders --source lp --trained daq --window 1.5 |
| d_lpf_enl1_daq_nolines.png | IMG Fig 2.7 | final LP arm, ENL 1.0 s, DAQ-trained frozen decoder: far contra .42 -> .20 acute **, 4 / 4 | DAQ-trained twin of Fig 2.8 | scripts/lp_final_render.py decoders --source lp --trained daq --window 1.0 |
| map_REST_precue_contact.png, map_RESTNL_precue_contact.png | IMG Fig 7.1, 7.1b | pre-cue 2 s contact-gate response maps, pooled, REST / RESTnl reference (RESTnl: acute far-contra rise 4 / 4, PS94 flips) | anticipatory position map on the 2 s DAQ window | wfield_local/epoch_maps.py (position_reference_maps.py, beta_maps.py); scripts/restnl_maps.py --arm contact |
| drift_REST_precue_contact.png, drift_RESTNL_precue_contact.png | IMG Fig 7.2, 7.2b | June -> August drift for the 2 s contact pre-cue maps (chronic far-contra fall drift-sized; chronic far middle .53 beyond drift; RESTnl acute far contra 1.80) | "is a pre-cue map change beyond the June -> August drift?" (DAQ arm) | scripts/map_drift_prepre.py [--refs restnl] |
| map_REST_enl1s_contact.png, map_RESTNL_enl1s_contact.png | IMG Fig 7.5, 7.5b | DAQ ENL 1 s contact maps, pooled (REST acute far contra 1.67; RESTnl 2.04, 4 / 4) | contact twin of the LP 1 s pre-cue map (Fig 7.3) | wfield_local/epoch_maps.py; scripts/restnl_maps.py --arm enl1s |
| drift_REST_enl1s_contact.png, drift_RESTNL_enl1s_contact.png | IMG Fig 7.6, 7.6b | drift for the DAQ ENL 1 s maps (far-middle chronic .35 / .49 beyond drift) | DAQ ENL 1 s drift check | scripts/map_drift_prepre.py [--refs restnl] |
| map_REST_cue_contact.png, map_RESTNL_cue_contact.png | IMG Fig 7.7, 7.7b | cue-window contact maps, pooled (acute far contra .48 REST / .53 RESTnl of pre, far middle .66 / .76, near ipsi up to 1.43) | contact twin of the cue map (Fig 7.10 / 7.10b) | wfield_local/epoch_maps.py; scripts/restnl_maps.py --arm contact |
| map_REST_cue_contact_by_animal.png, map_RESTNL_cue_contact_by_animal.png | IMG Fig 7.8, 7.8b | cue-window contact far-contra maps per animal (acute loss 4 / 4) | per-animal contact cue maps | wfield_local/epoch_maps.py (family 15rpa); scripts/restnl_maps.py --arm contact |
| drift_REST_cue_contact.png, drift_RESTNL_cue_contact.png | IMG Fig 7.9, 7.9b | cue contact drift (far-contra acute .49 / .53 beyond drift .95 / .93; far middle not separable from drift) | contact cue drift check | scripts/map_drift_prepre.py [--refs restnl] |
| map_PRECUE_cue_contact.png | IMG Fig 7.12 | cue contact maps, PRECUE reference: cue-evoked far-contra increment .18 of pre acutely, far middle .52 | contact twin of the PRECUE cue map | wfield_local/epoch_maps.py |
| drift_PRECUE_cue_contact.png | IMG Fig 7.13 | PRECUE cue contact drift (far contra .18 and far middle .55 beyond drift) | contact PRECUE drift check | scripts/map_drift_prepre.py |
| map_REST_lick_contact.png, map_RESTNL_lick_contact.png | IMG Fig 7.16, 7.16b | lick-window contact maps; acute far-contra panel empty (< 20 contact trials); RESTnl near positions 1.42-1.67, far ipsi 1.50 | lick-window maps on DAQ contact (other positions) | wfield_local/epoch_maps.py; scripts/restnl_maps.py --arm contact |
| drift_REST_lick_contact.png, drift_RESTNL_lick_contact.png | IMG Fig 7.17, 7.17b | lick contact drift; no acute far-contra estimate; near-spout acute rises, opposite to drift | contact lick drift check | scripts/map_drift_prepre.py [--refs restnl] |
| map_PRECUE_lick_contact.png | IMG Fig 7.21 | lick contact maps, PRECUE reference; acute far contra empty | contact twin of Fig 7.23 | wfield_local/epoch_maps.py |
| drift_PRECUE_lick_contact.png | IMG Fig 7.22 | PRECUE lick contact drift: far-contra increment grew June -> August (1.32, learning); no acute estimate | "did the cue-referenced lick map change before the stroke?" | scripts/map_drift_prepre.py |
| haufe_precue.png | IMG Fig 14.1 | Haufe decoder maps, pre-cue window, pooled: weak (pre split-half r .73-.96), only near middle's acute change significant | "where does the pre-cue decode come from?" | wfield_local/epoch_maps.py (maps and tests wfield_local/beta_maps.py) |
| haufez_precue.png | IMG Fig 14.4 | pre-cue Haufe maps tested against zero: few significant bins in any epoch | pre-cue Haufe bins per position / epoch | wfield_local/epoch_maps.py (beta_maps.py) |
| haufepa_precue.png | IMG Fig 14.7 | far-contra pre-cue Haufe map per animal: acute change does not replicate (2 / 4 up, PS94 down) | per-animal pre-cue Haufe maps | wfield_local/epoch_maps.py (beta_maps.py) |

## Figdata that must not be used for a `--plot-only` redraw

- `lick_cue_to_lick_B_{pre_models,exponents_all,exponents_far_contra}_tau0.pkl` in the rebuild's
  `.../iti_licks/<rebuild tag>/cue_to_lick/figdata/` (written 2026-11-17 19:48) hold the tau-0.6 data of the LAST run
  only (restnlc1): before the 11-18 fix, `figdata.fd_path`'s `Path(stem).stem` cut "..._tau0.6_<variant>" to "..._tau0",
  so every rest variant overwrote the same file. The same-named files in the pre-rebuild
  `lp_r4_heatmap_20261013__clean_v36/.../cue_to_lick/figdata/` (10-30) are equally ambiguous. Priya decided (11-18) not
  to regenerate them. Each tau-0.6 PNG was drawn right after its own save, so it matches its variant unless two variants ran at the same time. New runs write
  `..._tau0p6<variant>.pkl` (`lick_cue_to_lick.fd_stem`).
