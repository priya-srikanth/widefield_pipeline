# Section C render (O2-dependent encoder figures), cohort-1 rebuild, 11-05

Code: wf_rebuild `rebuild/cohort1-20261103`, commits 24d604cc + 19d3cf71 (REBUILD_CODE 8.56), on top of 31ec5c1a.
Profile `cohort1_rebuild_20261103` (`--check` exit 0), `WFIELD_INPUT_CACHE=C:/Users/SabatiniLab/wfield_cache`,
PYTHONPATH = repo + repo/scripts. Inputs: the collected O2 runs of bundle `enc_rebuild_20261104b` under

`E = M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_pre0801_sel-camera-cue_stop-endlick-cls-contact-incomplete-stopA2jexrt3500c70_daqlk`

Every output is under E (or `<rebuild root>/rebuild_figs/<run tag>/` for the rebuild_figs scripts). No script refused and
none wrote outside the rebuild root. Every script reports 107 sessions = PS92 27, PS93 26, PS94 27, PS95 27 (4 animals).
At most 2 scripts ran at once, with workers <= 3; free RAM stayed >= 13 GB.

## Code changes (REBUILD_CODE 8.56)
- The FR scripts had the 10-25 hand-collected folder names hard-coded (`..._subset-fr_exec-polar_licks-combined_frg-<m>`,
  and a separate `_tr` run). `collect` writes `movement_first_fr-master_pre_v1_frg-<m>_subset-fr_exec-polar_licks-combined`,
  and fr_A itself carries the trial-to-trial models. The new helper `fr_groups.run_folder` tries the new name first and
  falls back to the old one. Test added.
- `combined_encoding_figures`: the subset-kind checks now ignore the `_exec-...` suffix. Before this,
  `..._subset-trmodels_exec-...` and `..._subset-shufpos-_exec-...` / `droppos-` would have gone to the wrong suite.
- `fr_direction_mechanism --bundle`: under a rebuild root this flag is now required; the default was the adopted
  enc_v36_full_20261024 bundle.
- `--jobs` added to `fr_direction`, `fr_pixel_significance` and `variance_maps`, which had 6 / 8 workers hard-coded.
- `encoding_mixed`: now reads the suffixed main and droppos runs.

## Per script (python = locanmf env; cwd = wf_rebuild)
| script / command | exit | runtime | output (under E unless noted) | PNGs |
|---|---|---|---|---|
| `fr_musall.py --mode A --windows lick cue` | 0 | 18 min | fr_encoding_summary/figures (+ fr_musall_pixel_maps_modeA.npz) | 6 |
| `fr_pixel_significance.py --mode A --windows lick cue --jobs 3` | 0 | 25 min | fr_encoding_summary/pixel_significance | 2 |
| `fr_pixel_maps_sig.py --mode A` | 0 | 1 min | fr_encoding_summary/figures | 2 |
| `fr_contra_orofacial.py --mode A` | 0 | 28 min | fr_encoding_summary/contra_orofacial | 2 |
| `fr_encoding_summary.py` | 0 | 8 min | fr_encoding_summary/figures | 5 |
| `fr_direction.py --jobs 3` | 0 | 15 min | fr_encoding_summary/direction | 4 |
| `fr_trial_peth.py` (reads fr_A) | 0 | 35 min | fr_encoding_summary/trial_peth | 2 |
| `variance_maps.py --jobs 3` | 0 | 4 min | variance_maps | 2 |
| `fr_direction_mechanism.py --jobs 3 --bundle <rebuild root>/o2/enc_rebuild_20261104b/inputs` | 0 | 4 min | fr_encoding_summary/direction_mechanism | 2 |
| `combined_encoding_figures.py --order movement_first_subset-nokin_exec-polar_licks-combined_bins` | 0 | 8 min | <order>/figures, tables | 5 |
| ... `_bins_bw0.5` | 0 | 9 min | same | 5 |
| ... `_bins_bw2` | 0 | 10 min | same | 5 |
| `... --order movement_first_exec-polar_licks-combined` (main) | 0 | 88 min | same | 43 |
| `... --order movement_first_exec-polar_licks-combined_nolicksall` | 0 | 71 min | same | 43 |
| `... --order movement_first_subset-nodir_exec-polar_licks-combined` | 0 | 5 min | same | 5 |
| `... --order movement_first_subset-shufpos-_exec-polar_licks-combined` | 0 | 8 min | same | 6 |
| `... --order movement_first_subset-droppos-_exec-polar_licks-combined` | 0 | 8 min | same | 6 |
| `... --order movement_first_subset-trmodels_exec-polar_licks-combined` | 0 | <1 min | same | 3 |
| `... --order musall_subset-shuf_exec-polar_licks-combined` | 0 | <1 min | same | 1 |
| `... --order movement_first_subset-drop_exec-polar_licks-combined` | 0 | <1 min | none ("drop models only -> encoding_mixed", by design) | 0 |
| `rebuild_figs/fig_softmax.py` | 0 | <1 min | <rebuild root>/rebuild_figs/<run tag>/ (fig9, fig9b, fig10) | 3 |
| `rebuild_figs/fig_softmax_stats.py` | 0 | <1 min | same (fig9c + csv) | 1 |
| `encoding_mixed.py` (main + droppos + 3-D bins; 17 measures x 324 rows, LMM fallback ladder used) | 0 | 93 min | qc/encoding_mixed{,_summary}.csv, figures/mixed_models | 6 |

## Skipped / partial
- `rebuild_figs/fig_pooled.py` was not run, because its section-B inputs are not rendered yet in this run:
  - `qc/lp_encoder_decompose_*_joint_lick_by_animal.csv`
  - `allsession_encoder_v7g3_t3d_ref/`
  - `attempt_coding/<tag>/summary_{lick,cue}.csv`
  - `frozen_encoder_gate-attempted_v7g3_t3d/residual_trend_sessions.csv`

  The binned-contest tables it reads (bins / bw0.5 / bw2 `region_measures.csv`) are now in place.
- `encoding_mixed`: the rebuild has no separate cam4 contest run (`movement_first_subset-nokin`) and no `_win` runs. The
  `cam4:*` measures are therefore absent, and the 2 `cam4:u_target` panels plus the `cam4:ceiling` panel of the
  6-panel mixed-model figures are blank. The main run already contains `nokin:*` models and the extra windows. Whether
  those should feed the `cam4` measures is a decision for Priya / the coordinator; I did not make that change.
- droppos: it arrived mid-render (107 sessions) and is included in combined_encoding_figures (droppos) and encoding_mixed.
  No other section C script reads it.

## Looked odd / to check
- **Null calibration above 5 % on some FR map tests** (`fr_pixel_significance`, 50 random pre half-splits; share with at
  least 1 false bin):
  - full_fr cue 16 px: 0.12
  - unique_position lick 32 px: 0.10
  - full_fr cue 32 px, unique_fr lick, unique_position cue 32 / lick 16: 0.08

  The others are 0-0.06. variance_maps calibrates at 0-0.02.
- **Fixed conclusion text in a suptitle.** fig10_softmax_width.png has the hard-coded sentence "Chronic target lead holds
  across widths (PS94 ties at 2)". In this render, PS94 chronic lick at width 2 has executed slightly above target, so
  the sentence may no longer describe the data. This is caption/conclusion text, so I left it unchanged.
- **Styles that predate the conventions** (I made no plotting changes, so these are unchanged):
  - The droppos / shufpos `position_each_maps_*` rows run close contra, mid, ipsi, then far contra, mid, ipsi. This is
    not the canonical order (ipsi, mid, contra).
  - fig10 still draws a top spine.
- No figure was empty or all-NaN among those spot-checked:
  - fig10
  - position_each_maps_lick (droppos)
  - the logs' per-epoch tables

  `fr_encoding_summary`: |full(main) - full(FR run)| max 1e-5, so both runs share the same base model, as expected.
