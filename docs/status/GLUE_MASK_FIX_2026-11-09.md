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

## Mask-dependent caches whose key does NOT see the mask (listed, not changed)

- O2 encoding bundle `inputs/<label>.pkl` (`scripts/o2_encoding.py` skip-if-exists; `o2_bundle.identity` has no mask). Fix
  proposal: record `mask_signature` in the bundle at bundle time (computed locally, since O2 has no share) and compare it in
  `verify` from the bundle's own record. Rebuild `enc_rebuild_20261104b` inputs under a new bundle name.
- Other skip-if-exists extracts on `load_imaging` + `roi_masks`: `lick_matched_pixel_maps`, `lick_cue_to_lick_extract`,
  `lick_interhemispheric`, `restw_strict_check`, `lick_precue_restw`, `disengagement_tests`, `iti_hemo_check`. Fix proposal: the
  same `mask_sig` field plus `iti_licks.check_extract_masks`.
- `frozen_encoder` `fit/pre_stats.pkl` / `spatial_<animal>.pkl` (`--reuse-stats` / `--fit-from`) and `epoch_kernels`
  `stats/<label>.pkl`. These are existence-keyed. Fix proposal: store `mask_signature` in the pickle and check it on reuse.
- `state_space_sim` `lnmf_keep_<animal>_<bid>.npy` (from `component_exclusion.occluded`) and the binned / RESP / FR caches
  built from the ITI extracts. These are path-keyed. Fix proposal: put the mask digest in the filename.
- In-process `lru_cache` on `painted_exclusion`: restart any long-lived process (Jupyter) after the merge.
- Keys that already change: `frozen_models` (`train_fp` hashes the training arrays, and CD includes `drop`; the exception is the
  `daq_trained_in_lp` constant `DAQ_RUN_FP`), `cd_trajectories` `cdcourse` (hashes `w`), and everything keyed on the bytes of
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
