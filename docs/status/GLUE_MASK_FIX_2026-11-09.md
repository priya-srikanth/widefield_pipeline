# Rebuild glue-mask bug: fix, cache keys, audit, re-run plan (2026-11-09)

Priya approved 11-09 ~11:40: fix in shared code, then audit which outputs change. Background: RUNNING_LIST_2026-11-08_STATE_SPACE
item 12, RUNNING_LIST_2026-11-06 11:35.

## The bug

Under `cohort1_rebuild_20261103`, `paint_exclusion.mask_dir()` resolved to `<rebuild>/labcams/grant_figures`, where no mask was
painted. `beta_maps.painted_exclusion()` swallowed the missing file and returned an all-False mask. Measured on origin/main
2009bf45 under the profile: union 0 px, brain 192,110, stat 166,974. After the fix: union 40,663, brain 151,447, stat 129,770
(the same as with no profile). `component_exclusion.excluded_pixels` had the same fallback: under the profile every animal
got bulbs only (15,106 px), where the right values are PS92 50,885 / PS93 39,689 / PS94 42,787 / PS95 45,383. So the CD
trajectories' occluded-component drop (`mask_occluded=True`, the default) also ran without the glue.

## Fix (branch fix/rebuild-glue-mask-1109)

- `paint_exclusion.mask_dir` uses `PathResolver.input_root("cohort_out")`. The masks are hand-made inputs, read in place and never
  copied. The fence now allows exactly `grant_figures/exclusion_mask_painted_*.npy` (`paths.yaml rebuild.labcams_inputs`, a
  relative-path glob, `rebuild_root._labcams_input`); the rest of grant_figures stays refused.
- `paint_exclusion.load_mask` raises when a mask is missing, unreadable, the wrong shape, or empty. It is used by both
  `beta_maps.painted_exclusion` and `component_exclusion.excluded_pixels`. The one explicit opt-out is
  `cohorts.yaml painted_exclusion: none` (cohort2, which has no painted masks), so cohort 2 is never given cohort 1 masks.
- `combined_encoding._load_imaging` raises instead of falling back to the raw Allen mask or an all-True stat mask.
- `beta_maps.mask_signature()` is a sha1 of the brain, stat and painted-union mask content. It is now part of the
  encoder-input cache key (`_cached("imaging")`).
- `scripts/iti_licks.py` session extracts store `mask_sig`. A cached extract on different masks is refused. An extract with no
  mask_sig is refused inside a rebuild root (all of those were built with the glue in) and accepted at the default root.
- csig (`trial_classes.class_source_signature`) under the profile: 3d7bbe921e both before and after the fix (trial classes do
  not depend on the mask).

## Stale-read protection (branch fix/mask-keyed-caches-1109, Priya 11-09 ~11:45)

Every cache whose content depends on the brain / stat / painted / component-exclusion masks now records the masks it
was built on, and a reader refuses a mismatch. The helper is `wfield_local/mask_keys.py`:

- `signature()` is `beta_maps.mask_signature()`. Under the profile and at the default root it is `5f9eb36f36b2`.
- `exclusion_signature(animal)` is the sha1 of `component_exclusion.excluded_pixels`.
- `check` / `check_npz` / `check_obj` / `stamp`, plus a sidecar `<cache>.mask_sig` (`write_sidecar` / `check_sidecar`) for
  pooled caches whose format other readers depend on.

The rule is the same everywhere, as in `iti_licks.check_extract_masks`:

- A digest that differs from the current masks raises `RuntimeError`.
- A cache with NO digest is refused inside a rebuild root (all of those were built with the glue in) and accepted at the
  default root.
- Nothing is ever deleted or moved by the code.

csig is unchanged (3d7bbe921e). No cache-key source module was edited. Tests: `tests/test_mask_keyed_caches.py` and
`tests/test_mask_keyed_extracts_more.py`.

What is protected:

- **O2 encoding bundle.**
  - At bundle time `mask_sig` is recorded in `bundle.json` and in every `inputs/<label>.pkl`.
  - On the desktop, `o2_bundle.verify_masks` runs on resume, `check`, `collect` and `refresh-code`.
  - The runner's `o2_bundle.check_inputs` compares each pickle with `bundle.json`'s own record, because O2 has no share.
  - A rebuild-root bundle without `mask_sig` is refused.
- **ITI-family per-session extracts.** The writer stamps `mask_sig`; it is checked at the skip-if-exists branch and at
  the readers. The extracts:
  - `iti_licks` `sessions*/` (every `il._load`), `lick_matched_pixel_maps`, `lick_cue_to_lick_extract` (both variants),
    `lick_interhemispheric` rest.
  - `restw_strict_check` (and `restnl.level`), `lick_precue_restw` (`precue_restw` / `precue_restnl[_wr]`),
    `disengagement_tests` `img_*`, `iti_hemo_check`.
  - `lick_cued_ramp` `cued_wide`, `lick_firstlick_extract` `firstlick_full`, `revup_neural`, `lick_precue_restw_traces`,
    `arrival_to_cue_traces`, `stopped_trial_traces`.
  - `lick_correction`, `lick_enlv_extract` (`sessions`, `sessions_reg`), `prelick_hemisphere_tuning`,
    `prelick_svd_tuning`, `prelaunch_aim_tracking`.
  - `iti_basis_decoder` (svdbasis), `within_spout_deviation`, `lick_template_test`, `spout_unique_maps`.
  - `restnl_rescue`. Its reuse also requires that the recorded cue-to-lick source equals the c2l extract present now.
    That was the cause of the 11-09 "No objects to concatenate": all 107 rescue files predated the c2l extract.
- **Pooled caches (sidecar).**
  - The `zscore_check` SD and split tables, checked in `iti_licks._table`.
  - `decoder_pattern_sessions.pkl` and `spout_pattern_sessions*.pkl`, plus their readers.
  - `session_levels/patterns_gap2_enc.pkl`, `acad_extract.pkl`, `partial_refit/fits.pkl`.
- **frozen_encoder.**
  - `fit/pre_stats.pkl` records, `spatial_<animal>.pkl` and `pre_stats_roi.pkl` entries store `mask_sig`. They are
    checked on `--reuse-stats` / `--fit-from` / `--trials-only` / `--roi-targets` and on every `_spatial_of` load.
  - `epoch_kernels` `stats/<label>.pkl` likewise.
- **state_space_sim.**
  - The Allen binned cache moves to `<cache>/m<signature>/<align>/`. The old `cache/<align>/` is never read.
  - Every ITI extract read is checked as a rebuild-root product, even without the profile.
  - The keep-mask file becomes `lnmf_keep_<animal>_<bid>_x<exclusion digest>.npy`.
  - `WF_SS_BASIS=svd` refuses an SVD basis whose `mask.npy` differs from the current brain mask (ss_svdbm passes). The
    chronic-map RESP cache is SVD-only and inherits that check.
- **frozen_models `DAQ_RUN_FP`.** The LP arm's DAQ-trained lookup ignored `train_fp`, so it could serve a glue-in DAQ
  model. Manifests now record `mask_sig` (spec ids unchanged). `find_daq_trained` skips models on other masks, and models
  without `mask_sig` inside a rebuild root, and raises when only such models match.
- **Not protected; move these aside instead** (section below): the `*_preds_*r8.csv*` decoder prediction tables,
  `balimg_sessions.csv` / `balimg_primary_licks.csv.gz`, curbd_pilot fits, and `combined_encoding --skip-done` outputs.
  Also not protected: the state-space O2 LocaNMF bundle (its components are dropped at load by the keyed keep-mask).

## Move-aside list (READ-ONLY scan 11-09, `scripts/scan_mask_stale.py`; stale / total = files without the current mask_sig)

Path prefixes:

- `RE` = `M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding`
- `X` = `RE/iti_licks/hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_pre0801`
- `W` = `RE/iti_licks/hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_pre0801_sel-camera-cue_stop-endlick-cls-contact-incomplete-stopA2jexrt3500c70_daqlk`
- `E` = `RE/hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_pre0801_sel-camera-cue_stop-endlick-cls-contact-incomplete-stopA2jexrt3500c70_daqlk`

Refused now. These must be moved aside (or rebuilt into a new folder) before the re-run:

| folder | stale / total |
|---|---|
| `X/sessions_roi8fr` (the ITI extracts; also state_space_sim's ITI input) | 107/107 |
| `X/zscore_check/session_sd_roi8_rimlat_almline.csv` + `session_sd_split_roi8_rimlat_almline.csv` (no sidecar) | 2 files |
| `W/june_ref/sessions_roi8fr` | 12/12 |
| `W/june_ref/session_sd_roi8_rimlat_almline.csv` + `session_sd_split_roi8_rimlat_almline.csv` | 2 files |
| `W/cued_wide` | 107/107 |
| `W/firstlick_full` | 107/107 |
| `W/interhemi_rest_w1_restnl` | 107/107 |
| `W/precue_restw` | 107/107 |
| `W/precue_restnl_wr` | 107/107 |
| `W/precue_restw_traces` | 107/107 |
| `W/precue_restnl_traces_wr` | 107/107 |
| `W/cue_to_lick/sessions` | 107/107 |
| `W/cue_to_lick/sessions_restnl_wr` | 107/107 |
| `W/restw_check/sessions` | 107/107 |
| `W/restw_check/rescue/sessions` | 107/107 |
| `W/revup/sessions` | 107/107 |
| `W/stopped_traces/sessions` | 107/107 |
| `W/lick_correction/sessions` | 107/107 |
| `W/enlv_licks/sessions` | 107/107 |
| `W/enlv_licks/sessions_reg` | 107/107 |
| `W/hemo_check/cache_roi8` | 107/107 |
| `W/basis_svdbasis/sessions` | 107/107 |
| `W/decoder_pattern_sessions.pkl`, `W/spout_pattern_sessions.pkl`, `W/session_levels/patterns_gap2_enc.pkl` | 3 files |
| `E/disengagement/sessions` (`img_*.npz`) | 107/107 |
| `E/prelick_hemisphere_tuning/sessions` | 107/107 |
| `E/prelick_svd_tuning/features` | 107/107 |
| `E/prelaunch_aim_tracking/features` | 107/107 |
| `E/qc/within_spout_deviation_lick/sessions` | 107/107 |
| `E/frozen_encoder_*/fit/` (pre_stats* / spatial_*), 20 of the 22 run folders | 54 pickles |
| `E/frozen_encoder_gate-attempted_v7g3_t3d_spont_frA_ls_basis/epoch_kernels/stats` | 76/76 |
| `partial_refit/fits.pkl` in `E/frozen_encoder_gate-attempted_v7g3_t3d_frA_te`, `..._frA_te__nocalib_failed_20261105`, `..._spont_frA`, `..._spont_frA_ls` | 4 files |
| `E/frozen_encoder_gate-attempted_v7g3_t3d_spont_frA_ls_basis/spout_unique_cache_joint.npz` | 1 file |
| `M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/o2/enc_rebuild_20261104b` (`bundle.json` has no mask_sig) | 107 inputs |

Not guarded, so a re-run would REUSE them. Move these aside so they are rebuilt:

- `W/first_lick_preds_splitr8.csv.gz`
- `W/lick_order_preds_r8.csv.gz`
- `W/lick_order_preds_areas_splitr8{,_far1,_gap2,_gap2far1}.csv.gz`
- `W/where_preds_splitr8.csv`
- `W/where_3axis_preds_splitr8.csv.gz`
- `W/where_angle_reach_preds_splitr8.csv.gz`
- `W/zscore_check/where_preds_z_splitr8.csv`
- `W/balimg_sessions.csv`
- `W/balimg_primary_licks.csv.gz`
- `E/movement_first_*` and `E/musall_*` session outputs, which `combined_encoding --skip-done` would reuse: not cache
  folders, but glue-in run outputs.
- `curbd_pilot/fits*`, if present (none found under the rebuild root).

Orphaned (never read again; optional tidy-up):

- `M:/MICROSCOPE/Priya/Widefield/state_space_recovery/cache/{cue,cue25,lick,splice}`: the Allen binned cache, 107 files
  per alignment.
- `.../state_space_recovery/cache_lnmf/lnmf_keep_PS9{2,3,4,5}_<bid>.npy`

Refused only by the LP-arm DAQ cross-check (`WIDEFIELD_LP_FROZEN=daq`):

- the 2,861 manifests in `M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/frozen_models`. Do not move the store; re-freeze
  the DAQ run instead.

Follow-up for branch `feat/precue-enl-window-1109`: its new `precue_enl_lp1s` cache in `lick_precue_restw.py` needs the
same `mask_keys.check_obj` at its skip (its diff ~L289/301) and reader (~L343) and `**mask_keys.stamp()` at its write
(~L321).

## Other cache notes (11-09)

- In-process `lru_cache` on `painted_exclusion`: restart any long-lived process (Jupyter) after the merge.
- Keys that already change: `frozen_models` (`train_fp` hashes the training arrays, and CD includes `drop`; the `daq_trained_in_lp` constant
  `DAQ_RUN_FP` lookup is now mask-checked through the manifest, above), `cd_trajectories` `cdcourse` (hashes `w`), and everything keyed on the bytes of
  `beta_maps.py` (that key moves incidentally with this commit).
- Not mask-dependent (full-grid maps, masks applied later): session_cache `prm` / `bm` / `tf-*` / `cdarms`, the restnl npz,
  and `o2_state_space_svd` (built with the guarded correct mask).


## Glue effect on area traces (rebuild input cache, glue in, against the canonical stat mask)

Three sessions: PS93_0811, PS93_0819, PS95_0820. Raw output:
`scratchpad/glue_audit/area_trace_glue.txt`. Areas are Allen area x stat mask; the ROI rim rule is not applied.

- **17 area x hemisphere targets drop out completely** (VISp_left, VISl/al/li/pl/por_left, VISli/pl/por_right, AUDd/p/po both
  hemispheres, TEa both hemispheres). They exist only in glue-in outputs.
- 90-98 % of the pixels go: VISl_right, SSs_right, VISrl_left, SSs_left, VISpm_left. Trace r = 0.92-0.99, and the SD changes
  by 0.9-1.6x.
- 17-40 % go: VISal_right, VISp_right, RSPagl_left, SSp-n_right, SSp-bfd_left. r >= 0.994, SD 1.03-1.17x.
- The "cortex" mean target loses 22 % of its pixels (r = 0.999, SD 1.14-1.16x). The encoder reports its readouts on this
  target.
- Even "no-glue" areas lose a few pixels, because the stat erosion also eats a 16 px rim around the glue: SSp-n_left 9 %,
  VISa_left 6 %, SSp-m_left 5 %, SSp-bfd_right 7 %. Unchanged: MOs, MOp, ACAd, PL, RSPv, SSp-un/ul/ll/tr. ALM / tjM1 ROIs are anterior and not measured here (expected unchanged).
