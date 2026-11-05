# Figure data + `--plot-only` for the ITI / cued figure scripts (2026-10-27)

Priya 10-27: *"ensure all models / data analysis is being saved so if we want to change plotting style we can re-make
plots without re-doing analysis"*.

## Convention (`scripts/figdata.py`)

- The analysis step of a figure script writes ONE pickle per figure (sometimes per script) holding everything the figure
  draws -- point estimates, CIs, per-session / per-animal values, curves, map arrays and the imaging grid needed to place
  pixels -- to `<il.OUT>/figdata/<figure stem>.pkl`
  (`il.OUT = M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/iti_licks`).
  The pickle is named after the PNG it feeds; PNG names are unchanged.
- `--plot-only` loads ONLY that pickle and redraws: no data load, no bootstrap, no permutation, no fit. A restyle takes
  seconds and no number can shift. If the pickle is missing it stops with "run the script once without --plot-only".
- The draw function takes the figure-data object (`plot(D)`); computation that used to sit inside the draw function moved
  to the compute step.
- Helpers: `save_fd(stem, obj)` (atomic write), `load_fd(stem)`, `plot_only()`, `fd_path(stem)`, and for cortex maps
  `imaging_grid()` -> picklable `dict(pix, shape, ol)` + `grid_fns(grid)` -> `(to_img, ol)`.
- To restyle: edit the draw function, then `python scripts/<script>.py --plot-only`.

"verified" = the script was run normally, then with `--plot-only`, and the two PNGs were compared pixel by pixel
(`identical` = every pixel equal), and also against the PNG that existed before this change.

Run with: `export PYTHONPATH=$(cygpath -w <repo>) PYTHONWARNINGS=ignore WFIELD_INPUT_CACHE=C:/Users/SabatiniLab/wfield_cache`
and the `locanmf` python.

## Table

| script | figdata file(s) (under il.OUT/figdata/) | plot-only command | verified | notes |
|---|---|---|---|---|
| iti_session_pattern_tests.py | lick_session_pattern_maplevel, lick_session_pattern_similarity, lick_session_pattern_change_{position,within_spout}, lick_{cued,iti}_session_pattern_change_spout | `python scripts/iti_session_pattern_tests.py --plot-only` (7.6 s) | yes, all 6 identical to the normal run AND to the pre-change PNGs | Was: --plot-only still re-ran change_maps (per-pixel bootstrap). Now the change-map results (point, sig, sig_max, frac, n per key x epoch x kind) + whole-map table + grid are cached. The 4 change-map families now run in parallel (`parallel.fan_out`, one family per worker; each family keeps its own rng(0), as the serial loop did, so the draws are unchanged -- confirmed by the identical PNGs). Full run 12 min. `--reuse-maplevel` keeps the saved permutation CSV. |
| iti_decoder_pattern_stats.py | lick_decoder_pattern_stats_{lat,sag} | `python scripts/iti_decoder_pattern_stats.py --plot-only` (4.8 s) | yes, identical (normal + before) | per-axis test results + stats table + grid. The held-out session maps were already cached (decoder_pattern_sessions.pkl); the bootstrap (~7 min) is what --plot-only now skips. |
| iti_spout_pattern_stats.py | lick_cued_spout_pattern_stats, lick_iti_spout_pattern_stats | `python scripts/iti_spout_pattern_stats.py [--iti] --plot-only` (4 s) | yes, identical (both sets; normal + before) | figdata stem follows the figure name, so cued / --iti do not collide. |
| iti_pattern_area_stats.py | lick_pattern_area_stats_position, lick_{cued,iti}_pattern_area_stats_spout | `python scripts/iti_pattern_area_stats.py --plot-only` (7.8 s) | yes, identical (normal + before) | figdata = the area table (incl. per-animal area means) as a DataFrame. |
| iti_session_cluster_tests.py | lick_session_cluster_position, lick_{cued,iti}_session_cluster_spout, lick_session_shape_d, lick_{cued,iti}_session_shape_days | `python scripts/iti_session_cluster_tests.py --plot-only` (17 s) | yes, all 6 identical (normal + before) | Old --plot-only read maps.pkl (good) but still loaded imaging for the outline, took n_perm from argv, and the shape figures read session_pattern_tests' CSVs at draw time; now all from figdata. New `--reuse-maps` rebuilds the figdata from session_cluster/maps.pkl without the TFCE permutations (31 s); the plain run still re-runs them. |
| iti_spout_animal_panels.py | lick_cued_spout_animal_panels | `python scripts/iti_spout_animal_panels.py --plot-only` (4.6 s) | yes, identical (normal + before) | per animal x spout x epoch mean map, n, r to pre. Also now closes its figure. |
| figdata.py (new helper) | -- | -- | -- | save_fd / load_fd / plot_only / fd_path + imaging_grid / grid_fns (replaces the 3 inline copies of the stat-mask -> image placement in the scripts above). |
| iti_check_figures.py | lick_check_rmse, lick_iti_check_f0, lick_check_3axis, lick_check_error_per_lick, lick_{iti,cued}_check_error_per_lick{,_angle,_xyz}, lick_iti_check_{rmse,error,f0,3axis}_sessions, lick_cued_check_{rmse,error}_sessions (16) | `python scripts/iti_check_figures.py --plot-only` (23 s; normal 1 m 32 s) | yes, 16/16 identical to the normal run | 15/16 also identical to the pre-change PNG; `lick_iti_check_f0_sessions` differs from the snapshot only because the snapshot was drawn by older code (row-3 title "angle / pre" vs the current "change vs pre (deg)"). New `epoch_compute` / `epoch_draw` split (used by iti_where_angle_reach); `epoch_panel` and `_days` kept for other callers. f0_severity table prints only on a normal run. |
| iti_licks.py | lick_iti_rois, lick_iti_ramp_peth{,_isolated}, lick_direction_decoding, lick_frozen_timecourse, lick_frozen_summary, lick_iti_ramp_sessions, lick_iti_ramp_summary, lick_iti_kinematics_control, lick_iti_face_control, fr_lick_variance, lick_binned_sweep, lick_binned_summary (13) | `python scripts/iti_licks.py --plot-only` (all 13, 25 s); add a family flag (`--key`, `--controls`, `--binned`, `--ramp`, `--plot-direction`, `--plot-frozen-tc`, `--rois-only`) to redraw one family | yes, 13/13 identical (and to the snapshot) | Normal runs used the per-family flags (`--key` ~6 min, `--binned` ~15 min). New shared helpers `sig_compute`/`sig_draw`, `dots_compute`/`dots_draw` (`sig_bars`, `session_dots` unchanged). The old `--plot-direction` / `--plot-frozen-tc` flags still redraw from the CSVs + re-run the per-bin tests, and now also refresh the figdata. The full `summarise` (per-session decoding) was not re-run; its figures go through the verified code from the existing CSVs. |
| iti_area_maps.py | lick_iti_area_maps, lick_iti_area_peth, lick_iti_area_groups, lick_iti_area_compare | `python scripts/iti_area_maps.py --plot-only` (5 s; normal 1 m 54 s); `--compare` for the compare figure only | yes, 4/4 identical (and to the snapshot) | maps figdata = area image, names, outline, per-panel change values (atlas lookup stays in the draw step: a lookup, not a statistic). |
| iti_hemo_check.py | lick_iti_hemo_check_{variants,peth,edge_vessel,f0_areas,state_gap} | `python scripts/iti_hemo_check.py --plot-only` (5 s; normal 7 min) | yes, 5/5 identical (and to the snapshot) | Existing --plot-only used to re-read the CSVs + per-session cache and re-run every bootstrap CI; now figdata only. Normal run via `--summarise` (extraction is cached per session). |
| iti_where_scatter.py | lick_{cued,iti}_where_scatter, lick_where_delta (+ `_z` versions with --z) | `python scripts/iti_where_scatter.py [--z] --plot-only` (~13 s) | yes, 6/6 identical to the normal run | Old --plot-only re-read where_preds.csv and re-ran the delta bootstrap. Delta PNGs identical to the snapshot; the 4 scatter PNGs differ from the snapshot in the title (it now names `lick_where_delta*.png`, post-rename) and, for non-z, sub-pixel because the normal run regenerated where_preds.csv; numbers / labels match on visual check. |
| iti_where_per_spout.py | lick_where_per_spout{,_z} | `python scripts/iti_where_per_spout.py [--z] --plot-only` | yes (both, and snapshot) | tests T + per-session changes S; draw does mean +- SEM + seeded jitter only. |
| iti_where_angle_reach.py | lick_where_angle_reach | `python scripts/iti_where_angle_reach.py --plot-only` | yes (and snapshot) | via iti_check_figures `epoch_compute`/`epoch_draw`. |
| iti_where_drift.py | lick_where_drift, lick_where_drift_june | `python scripts/iti_where_drift.py [--june] --plot-only` | yes (both, and snapshot) | per-session residuals, tests, drift-fit curves, largest pre gap; no ridge fit in plot-only. |
| iti_where_mae.py | lick_where_mae | `python scripts/iti_where_mae.py --plot-only` | yes (and snapshot) | |
| iti_overcorrection.py | lick_cued_overcorrection{,_angle} | `python scripts/iti_overcorrection.py [--angle] --plot-only` | yes (both, and snapshot) | |
| iti_first_lick.py | lick_cued_first_lick | `python scripts/iti_first_lick.py --plot-only` | yes (and snapshot) | bootstrap table T is all the figure draws. |
| iti_recovery_measure.py | lick_cued_recovery_measure | `python scripts/iti_recovery_measure.py --plot-only` | yes (and snapshot) | compute() / plot(S). |
| iti_recovery_marker.py | lick_recovery_marker | `python scripts/iti_recovery_marker.py --plot-only` (4 s; normal 110 s) | yes (and snapshot) | post sessions, per-animal r, 5000-permutation table. |
| iti_recovery_link.py | lick_recovery_link_{tag}, lick_recovery_link_pooled_{tag}, lick_recovery_link_sessions_{tag} (tag = angle_z, lateral_z) | `python scripts/iti_recovery_link.py --beh angle --z --plot-only` (and `--beh lateral --z`) | yes, 6/6 identical (and snapshot) | Old --plot-only re-ran the pooled bootstrap from CSV and skipped the course figure. figdata root follows --src. The `hit` tag was not run (no hit figures exist). |
| cued_misdirected_licks.py | lick_cued_misdirected_licks | `python scripts/cued_misdirected_licks.py --plot-only` | yes (and snapshot) | per-session table. |
| iti_extension_dist.py | lick_extension_dist | `python scripts/iti_extension_dist.py --plot-only` | yes (and snapshot) | every histogram sample, thresholds, threshold table. |
| iti_basis_decoder.py | lick_basis_decoder | `python scripts/iti_basis_decoder.py --plot-only` | yes (normal `--summarise` run, and snapshot) | per-lag significance strip now via `il.sig_compute`/`sig_draw`. |
| iti_decoder_contrib.py | lick_decoder_contrib | `python scripts/iti_decoder_contrib.py --plot-only` | yes (and snapshot) | |
| iti_decoder_patterns.py | lick_decoder_patterns_{lat,sag}, lick_decoder_patterns_animals_lat | `python scripts/iti_decoder_patterns.py --plot-only` | yes, 3/3 (and snapshot) | mean maps, colour limits, n, similarities, grid. |
| iti_spout_patterns.py | lick_cued_spout_patterns | `python scripts/iti_spout_patterns.py --plot-only` | yes (and snapshot) | |
| iti_lick_rate.py | lick_iti_rate | `python scripts/iti_lick_rate.py --plot-only` | yes (figdata built from the saved CSVs; plot-only PNG identical to the existing PNG and the snapshot) | The figure draws only R / T / C, all three already saved as iti_lick_rate{,_tests,_recovery}.csv, so the figdata was built from them (pixel-identical redraw). A normal re-run (started 15:01, still running when this doc was written: serial per-session behaviour load + 5000-perm loop, ~2 h) will overwrite the pickle with its own result when it ends. Follow-up: fan `session_row` out with analysis_kit.fan_sessions (rule 6). |
| lick_gain_prediction.py | -- | -- | -- | Makes no figure (the docstring names figures/lick_gain_prediction.png but there is no plotting code); not edited. |
| no figure (skipped) | -- | -- | -- | iti_where_3axis, iti_where_beyond_pull, iti_where_rmse_split, iti_where_signed_error, iti_where_spread_matched, iti_contact_check, iti_f0_check, iti_farcontra_pattern, iti_june_ref, iti_nmatch_check, iti_zscore_check, iti_session_patterns (tables / caches only). |
| NOT touched (another worker) | -- | -- | no | iti_lick_order, iti_decoded_decomposition, iti_lmm_figures, iti_mixed_models, iti_where_decoder (+ lick_matched_readout, wfield_local/mixed_models). These still need the same treatment. |

## Left undone / caveats

- The five scripts owned by the other worker (iti_lick_order, iti_decoded_decomposition, iti_lmm_figures,
  iti_mixed_models, iti_where_decoder) are not converted.
- `--plot-only` means "no statistic recomputed". A few draw steps still do trivial arithmetic on the stored per-session
  values (mean +- SEM, seeded jitter, atlas lookup), but none loads data, fits or resamples.
- A plain (non `--plot-only`) run still recomputes and overwrites the figdata. Where an analysis is slow there is a
  reuse flag: `iti_session_pattern_tests --reuse-maplevel`, `iti_session_cluster_tests --reuse-maps`; the held-out
  session-map pickles (decoder_pattern_sessions.pkl, spout_pattern_sessions{,_iti}.pkl) were already reused.
- iti_where_scatter's normal run regenerated `where_preds.csv` (derived; read by per_spout / mae / others). The
  regenerated file reproduces the delta, per_spout and mae figures pixel-identically.
- iti_licks `summarise` (full per-session decoding) was not re-run; iti_recovery_link `hit` tag not run (no figure).
- The figdata pickles hold pandas / numpy objects; loading them needs a compatible pandas (locanmf env).
- Nothing committed (per instructions).
