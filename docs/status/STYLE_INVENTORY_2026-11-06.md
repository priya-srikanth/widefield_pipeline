# Off-style per-session plots, inventory 2026-11-06 (read-only; a8 helper)

House style (Priya): sessions = OPEN markers, SHAPE = animal, COLOR = epoch; no lines joining sessions across epochs;
stars in the top band (`sig_marks.star_top` + `reserve_headroom`); no top spine. Already fixed: `mixed_models.plot_fits`
on 34's `fix/log-deficit-primary-20261105` (905f3ca0 / b185ad70 / ef663dc7) -> its callers imaging_lmm, encoding_mixed,
iti_lmm_figures, remap_checks, remap_lmm. Nothing below has been changed. A shared session-marker helper would stop
new drift (proposed, not built).

## wfield_local/
- `epoch_figures.py:756` `session_points`: filled, color = animal; 7 call sites; `bar_row(dot_style="animal")` keeps it on purpose.
- `epoch_figures.py:2489` `timecourse_panel` (:2548-2550): per session vs day, color = animal, filled, lines across epochs.
- `recovery_trajectory.py:396` `figure` (:415, :417): F / G vs day "o-" across epochs, color = animal (hard-coded `_animal_colours`, :384). `figure_by_animal` (:515, :524) is an F-G trajectory by design.
- `locanmf_decoder_weights.py:182`: lines with markers, color = animal.
- `histology_area.py:75`: filled, color = animal (per-animal values; low priority).

## scripts/ -- color = animal, filled, lines across epochs / days
`face_rhythm_match.py:324`, `face_rhythm_peak.py:215`, `face_rhythm_readouts.py:191`, `:381`,
`trial_neural_vs_exec.py:202-203`, `lick_aim_deficit.py:81` (shape = animal, but lines), `recovery_timing.py:146`,
`lick_cue_to_lick.py:633`.

## scripts/ -- color = animal, filled dots, no lines
`kinematics_pooled_figs.py:107`, `:399`; `face_rhythm_epoch_shape.py:421`, `:458`; `combined_encoding_figures.py:896`;
`exec_angle_spread_check.py:130`; `enl_fr_removed_decode.py:176`; `encoding_key_figures.py:50` (own `ANIMAL_COL`, :27);
`mvar_regions.py:442`, `:483`; `reembedding_test.py:813`; `slowband_caveat_figure.py:61`, `:74`;
`position_block_durations.py:86-87`; `revup_behaviour.py:750`; `lick_cue_to_lick.py:885`; `restw_strict_check.py:520`;
`misdirected_lick_decode.py:435`, `:480`; `early_predictors.py:127` (per-animal).

## Other
- Open markers but color = animal: `cohort2_qc/quiet_area_spikes.py:235`.
- Per-animal epoch-mean lines (summaries, lower priority): `encoding_key_figures.py:48`, `:78`; `enl_fr_removed_decode.py:165`;
  `lick_cue_to_lick.py:525`; `rebuild_figs/fig_s_per_animal.py:101`; `recovered_20261103/per_animal_fig.py:13`;
  `reembedding_test.py:693`, `:787`; `face_rhythm_readouts.py:49`.

Line numbers are as of the helper's base 4afd2552 (wf_attempt-style tree; scripts live in the main repo).
