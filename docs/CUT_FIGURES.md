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
