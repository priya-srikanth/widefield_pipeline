# Rebuild order B: inventory traced from the code to the outputs (cohort 1, swapmask + Aug-only pre + LP-trained)

The inventory is PARTIAL. The four helper inventories (grant / epoch family keys, scripts/, decks / pages, wf_attempt-only scripts) had not reported at handback. Rows marked UNSURE in the CSV must be completed before any render.

## 0. Code first (no render may start before this)
The rebuild branch is analysis/frozen-default @ 1553db6a. It LACKS the swapmask overrides:
- b334316a: `combined_encoding.hemo_dir()`, FROZEN_OUT_ROOT, RESTNL_OUT_ROOT, QS_OUT_ROOT;
- 4813c3cf: `joint_locanmf.project_all()`;
- 3cd00814: the compare scripts.

All three are on analysis/cohort1-swapmask only. The quiet-events code lives on main (widefield_pipeline 910a3ab9) and differs from frozen-default:
- QS_LICK_SOURCE, cohort2_qc/lp_licks.py, quiet_sync_4to8, quiet_task_events_followup, quiet_state_response, quiet_lp_sensitivity, quiet_event_movie, quiet_f0_noise and quiet_breathing exist only there;
- cat_points.py, jaw_only_incomplete_figs.py and trial_class_raw_share_fig.py are also main-only.

Merge them all into one rebuild branch. Then fix these:
1. **Encoder routing.** `combined_encoding.hemo_dir()` reads only the env var. If the switch is made by editing defaults.yaml `hemo.variant`, the encoder and quiet events stay on ADOPTED while every other family moves. Fix it either way:
   - make hemo_dir follow `config.hemo_variant()`; or
   - keep `hemo.variant` unchanged and use the env var everywhere.
2. **Joint-basis projection.** `project_all()` compares the env var with defaults.yaml. If defaults.yaml is flipped and the env var is unset, it returns False. The in-fit pre sessions would then be served the STORED adopted C, so variants mix inside one decoder. Fix it in one of two ways:
   - set WIDEFIELD_JOINT_PROJECT_ALL=1 for the whole rebuild; or
   - compare against the variant the basis was built on (record it in the basis manifest).
3. **Planned cache changes.**
   - Drop WIDEFIELD_PRM_LEGACY_DATE8. restnl_maps.setup_env and lp_final_render both FORCE it to 1, so remove it there too.
   - Convert precue_engagement_states to the frozen store.
   - Narrow the map-cache code digest.
4. **lp_final_render `--trained` default.** It is `daq`. The new rule (LP arm trained on LP trials) needs `--trained lp` (WIDEFIELD_LP_FROZEN=lp), or flip the default.
5. **Aug-only pre for the decoders.** Either set WIDEFIELD_PRE_FROM=0801 for every decoder render, or make it the default.
   - restnl_mask build REFUSES PRE_FROM, so run it in its own step without the env var.
   - epoch_audit.full_cohort suspends PRE_FROM on purpose.
   - drift_decoder and 12.1b need June. Keep them as labeled exceptions.

## 1. Variant data
`hemo_meegkit_hpfit_lag_refitT_swapmask` is DONE for all 120 curated cohort-1 sessions (log C:/Users/SabatiniLab/swapmask_rerun_logs/build_summary.jsonl). Do NOT rebuild it. Every frozen spec_id and session-cache signature hashes the SVTcorr mtime:size, so a rebuild would orphan every downstream key.

## 2. Caches (fresh dirs; delete the old ones afterwards from a kept list)
- **WIDEFIELD_SESSION_CACHE.** Point it at a NEW dir, e.g. `.widefield_session_cache_rebuild`. `session_cache.cached` prunes every other signature of label x kind, and the restnl_mask cache prunes the same way.
  - Kinds hit: tf-*, prm-*, beta-map kinds, cd / arms, jointvc-*, per_session / rdm_rel / crossnobis / hemi.
  - Existing dirs: `.widefield_session_cache`, `_restnl`, `_restnl_glp3e0db850`, `_swapmask`, `_swapmask_ctl`, `_swapmask_restnl`.
- **WFIELD_INPUT_CACHE** (C:/Users/SabatiniLab/wfield_cache: imaging/ 335 pkl, behaviour/, gate_lp_check/, pose3d_inputs/).
  - After the hemo_dir merge it is keyed on the source path, so it is safe to share. It is never pruned.
  - BEFORE that merge, the imaging/ entries are adopted data under any env.
- **restnl baselines.** `python -m wfield_local.restnl_mask --jobs 4` with NO WIDEFIELD_PRE_FROM, run in each gate mode the families use. The gate-LP mode gets its own `_restnl_glp<sha>` dir.

## 3. Frozen fits (new store, or new spec ids in the existing one)
- Use WIDEFIELD_FROZEN_MODEL_DIR = a NEW local dir. `find()` still consults the M: server store, which is harmless only because the spec ids differ.
- Never pass `refreeze=`. The variant / Aug-only / LP-trained specs are SIBLINGS of the adopted ones (same kind / align / source prefix), so `refreeze` would rename the published adopted models to `.SUPERSEDED_*`.
- Expect SPEC-CHANGED log lines ("N -> N sessions", nothing added). Under Aug-only they will read "removed June".
- **Order:**
  1. DAQ arms first: the LP arm under daq_trained_in_lp RAISES on a miss.
  2. Then LP-trained (WIDEFIELD_LP_FROZEN=lp).
- **Kinds in the store** (C:/Users/SabatiniLab/frozen_models and cue_lick/frozen_models_lpfinal):
  - attempt_axes, cd_directions, coding_decoder, coding_directions, decoder (roi / joint), drift_decoder, encoder (roi);
  - enl_angle, enl_spout, epoch_decoder, face_aim (~300+ per animal), g_lick_disc(_folds), g_lickonly_decoder, iti_readout;
  - lick_code, lick_reader, nolick_reference (roi / joint), prelaunch_aim, prelaunch_svd, reembed, rest_frozen, state.
- **frozen_encoder is NOT in the store.** It refits pre stats into `<out>/fit/`. Never use `--reuse-stats` into a dir holding adopted stats: the reuse check is not keyed on the variant.

## 4. Families (render into NEW folders; per-family logs)
1. RESTnl maps, 15r / 15rpa / 14 maps.
2. Frozen epoch decoders, DAQ: acc / accdelta + the other epoch families. Use `--output` (output_tag carries no hemo or PRE_FROM tag).
3. LP arm: lp_final_render with `--trained lp`.
4. rest_frozen 12.1 / 12.1b.
5. Section G, nolick, cd, coding, attempt, drift.
6. frozen_encoder, then the scripts/frozen_* figures and the per-session combined_encoding + FR-in-encoder.
7. Quiet events (main code; QS_OUT_ROOT, QS_LICK_SOURCE=daq+lp).
8. ITI and wf_attempt goal-page scripts (UNSURE list).

## 5. Decks / pages
1. Analysis deck (`locanmf_analysis_deck` via deck_registry): reads `figures_working`, not the share.
2. Imaging page, encoding page (section 23), goal page, FR page.
3. PI deck: coordinate with the owning session (sabatinilab-e3). The per-slide map is not done.

## Env vars
| Variable | Value / note |
|---|---|
| WIDEFIELD_HEMO_VARIANT | meegkit_hpfit_lag_refitT_swapmask |
| WIDEFIELD_JOINT_PROJECT_ALL | 1 (belt and braces) |
| WIDEFIELD_PRE_FROM | 0801 (decoders; NOT for the restnl_mask build) |
| WIDEFIELD_LICK_SOURCE / WIDEFIELD_GATE_SOURCE | lp, LP arm |
| WIDEFIELD_LP_FROZEN | lp (LP-trained) |
| WIDEFIELD_ENL_WINDOW_S | per arm |
| WIDEFIELD_MAP_REFS | per the restnl / lp renders |
| WIDEFIELD_SESSION_CACHE | new dir |
| WIDEFIELD_FROZEN_MODEL_DIR | new dir |
| WFIELD_INPUT_CACHE | C:/Users/SabatiniLab/wfield_cache |
| FROZEN_OUT_ROOT / RESTNL_OUT_ROOT / QS_OUT_ROOT | new roots |
| WIDEFIELD_MAP_JOBS | <= 4 |
| WIDEFIELD_*_FROZEN | leave unset (stored = default) |

## Traps
1. **Branch.** frozen-default lacks hemo_dir / project_all / output-root overrides (swapmask branch only). Quiet-events code is newest on main.
2. **Encoder reads stay adopted.** `combined_encoding.load_imaging` hard-codes `hemo_{ADOPTED}` on frozen-default, main and attempt-coding. This covers the encoder, frozen encoder, FR-in-encoder, quiet events, ITI and many wf_attempt scripts. Even after the merge, hemo_dir reads the env var only, not defaults.yaml.
3. **Projection switch.** `project_all()` is False if the switch is made in defaults.yaml without the env var, so in-fit pre sessions use stored adopted C.
4. **Per-session LocaNMF families.** They read `locanmf_affine8v1_hemo_meegkit_hpfit` C = ADOPTED REGARDLESS of the variant (nightly A-C, RSA, cross-mouse, `--source locanmf`). Their session-cache signature changes anyway, so they recompute on unchanged data under a "variant" label. A LocaNMF refit is not reproducible (95 vs 93 components). Decide the scope explicitly.
5. **Spec ids can be mislabelled.** Frozen spec ids hash the svtcorr_path mtime:size, never the data actually used. Any kind whose features come through load_imaging (enl_angle, enl_spout, lick_code, lick_reader, prelaunch_*, reembed, face_aim / partial_FR*, areas_split_r8 / allen_split_r8 / svd_blk8 sources) gets a NEW spec id under the variant while still fitting on ADOPTED data, unless hemo_dir is merged. iti_readout is the exception: its data_sig hashes the actual features.
6. **Cache pruning.**
   - `session_cache.cached` prunes other signatures of label x kind: a variant run in the shared cache deletes the adopted tf / prm / beta caches.
   - restnl_mask prunes its own dir the same way.
   - map_collect kinds do not include the hemo variant.
7. **Output dirs are overwritten in place.** None of these carry a hemo or PRE_FROM tag, so a rebuild overwrites the adopted figures:
   - grant_figures/epoch (lp_licks.output_tag has no hemo / PRE_FROM tag);
   - combined_encoding.out_dir;
   - frozen_encoder (no FROZEN_OUT_ROOT on frozen-default);
   - cohort1_quiet_spikes (hard-coded M: path; per-session npz rewritten);
   - epoch_restnl.
8. **`refreeze=` supersedes adopted models.** The new specs are siblings, so `refreeze` renames the published models.
9. **LP arm before its DAQ models.** An LP run before the matching DAQ (variant, Aug-only) models exist raises (daq_trained_in_lp); with WIDEFIELD_LP_FROZEN=lp it fits LP-trained instead. Choose deliberately.
10. **lp_final_render forces settings.** It sets WIDEFIELD_PRM_LEGACY_DATE8=1 and its own frozen_models_lpfinal store; restnl_maps also forces LEGACY_DATE8. Its `--trained` default is daq.
11. **PRE_FROM conflicts.** restnl_mask raises under WIDEFIELD_PRE_FROM, so build the baselines first, separately. drift_decoder and 12.1b need June.
12. **Joint basis rebuild.** `joint_locanmf.load()` returns the NEWEST basis per animal. Building a basis under the variant would silently move EVERY joint-basis family (adopted renders included) onto the new basis. Keep 76d884 or pass the basis id explicitly.
13. **Hard-coded Aug-only pre.** frozen_encoder and many scripts (frozen_*, fr_*, encoder_trialavg_r2, contest_lines ...) hard-code `>= '0801'`, so they are already Aug-only. The config families default to June+Aug.
14. **Drift removal differs by family.** The encoder family removes drift with a >= 600 s cosine basis, as do the ITI scripts, iti_basis_decoder and stray_licks. The frozen epoch decoders and the other decoder / CD / nolick / section G families have NO slow-drift removal (open decision for Priya). Quiet events use only a 2-s running-median detrend.
15. **Diagnostics hard-coded to adopted.** slowband_decode_check, o2_video_me, iti_hemo_check and quiet_hemo_check load `hemo_{ADOPTED}` directly.
16. **Deck source.** The analysis deck reads the local figures_working dir, not the share.

## ADDENDUM 2026-11-03 (coverage completed)

Added 530 CSV rows (548 in total):
- every frozen-default `scripts/` file (389) and the 11 main-only files;
- the 121 wf_attempt-only files;
- 8 family / nightly rows merged from the family-key report.

How the classifications were made:
- frozen-default: grep of imaging entry points, with propagation through local imports. 37 behavior-only false positives (they import iti_licks / combined_encoding only for helpers) were checked by hand and moved to indirect / behavior.
- wf_attempt-only: per-file read by a helper sweep.

### Counts
| Group | REDO | NO | UNSURE | Notes |
|---|---|---|---|---|
| frozen-default scripts (389) | 209 | 169 | 11 | NO includes 25 cohort-2-only and 10 hemo-QC diagnostics |
| main-only scripts (11) | 7 | 4 | 0 | quiet events |
| wf_attempt-only (122, incl. untracked s1_refit_basis_table) | 99 | 15 | 8 | |
| family / nightly rows (original 18 + 8 new) | 20 | 2 | 4 | |
| **Total (548 rows)** | 335 | 190 | 23 | |

### New traps
17. **Per-session caches that skip when the file exists.** Many writers return early if the output file exists, so a rebuild silently REUSES adopted data unless the dirs are pruned or redirected:
    - quiet_area_spikes `OUT/sessions`;
    - iti_licks.extract `il.OUT/<SESS_NAME>`;
    - cued_wide, enlv_licks, firstlick_full, cue_to_lick, precue_restw / restnl, revup, stopped_traces, restw_check, lick_correction, prelick_* FEAT, prelaunch_aim FEAT, arrival_to_cue, disengagement img_*;
    - lick_matched_pixel_maps' scratchpad lmp_cache.

    Redirect with WFIELD_ITI_OUT_W where it exists. It exists ONLY on wf_attempt, not on frozen-default or main, and lick_interhemispheric hard-codes its OUT_W anyway.
18. **figdata `--plot-only` replays stale data.** The figdata pickle cache (wf_attempt `scripts/figdata.py`) has NO data signature, so `--plot-only` replays the adopted data. The same holds for `restnl_maps --plot-only`, lp_final_render `--plot-only` and replot_map_figures (bundles). Never plot-only during the rebuild until the bundles are regenerated.
19. **ADOPTED, defaults.yaml and the env var must move together.** Changing them separately breaks in three ways:
    - ADOPTED alone: frozen specs keep their train_sigs (svtcorr_path), so models are served as frozen-hit and the OLD-correction model is reused.
    - env / defaults.yaml alone: load_imaging stays adopted while spec ids change, so the store gets new "variant" models fitted on adopted data.
    - The wf_attempt list scripts keep xy_sha1 in the payload, but a mismatch only WARNS.
    - Fix: one switch that drives all three (or merge hemo_dir), plus a fresh frozen store.
20. **wf_attempt cannot BUILD or properly read the variant.**
    - Its hemo_variants lacks meegkit_hpfit_lag, iso_shift, bad_pairs and the tag argument. It also lacks swap_repair, restnl_mask and the clip-mask removal.
    - load_imaging reads `allen_brain_mask_native_grid.npy`, so the brain mask a run gets depends on WHICH BRANCH last wrote allen_aligned_affine8v1, not on the reading branch.
    - Merge before rendering the goal-page families.
21. **Manifest hard-coded to the old variant.** lick_balance_imaging_control(+_primary) and lick_acad_midline_control read `hemo_meegkit_hpfit/manifest.json` by literal name, and raw SVT.npy that has no swap repair.
22. **15e cannot be fixed by a render.** It reads npz files baked at PREPROCESSING time. It needs a preprocess-map regeneration under the variant.
23. **JSON-fed grant figures.** 2 / 2b / 4, 3a / 5b, 3b / 5, section_g_figures and miss_vs_stopped draw whatever mode last wrote the fixed-name JSON. Regenerate the JSON in the FINAL mode immediately before the render.
24. **Output dirs ignore the run mode.** PRE_FROM, HEMO_VARIANT, MAP_REFS and *_FROZEN=0 runs all write to the default dirs. The family-key report found 13 writers without lp_licks routing or writeguard. Always pass `--output` / OUT_ROOT.
25. **Per-session C ignores the hemo variant.** It follows WIDEFIELD_LOCANMF_VARIANT, so the nightly A-C / RSA / cross-mouse families cannot move without a LocaNMF refit (scope decision).

### Adopted-only diagnostics, and what they feed
| Script | Feeds | Verdict |
|---|---|---|
| slowband_decode_check (direct `hemo_{hv.ADOPTED}`) | slowband_caveat_figure -> imaging page Fig 1b (d_slowband) and goal page fig1b | REDO; parameterize the variant |
| iti_hemo_check (adopted vs orig vs 415) | normalizes ITI responses used by iti_mixed_models -> ITI section 14 figures | REDO with the "adopted" arm = swapmask |
| quiet_hemo_check (T10) | quoted in quiet-events section 15 / encoding section 23 text | REDO |
| s1_refit_basis_table (wf_attempt; direct `hemo_{ADOPTED}` + `ce._load_imaging`, no drift removal) | imaging page section 1 refit decoder | REDO |
| lick_balance_imaging_control(+_primary), lick_acad_midline_control | Fig 14.19 | REDO |
| o2_video_me | reads only a frame count | NO |
| rest_migration/plot_session_residual | 09-13 diagnostic, not on a page | UNSURE |
| cohort2_qc hemo-QC scripts (cohort1_impact, indicator_*, iso_*, raw470_vs_corrected, running_sign_check, slow_is_bleach, quiet_modes) | compare corrections by design | NO |

### Extract -> consumer order for the goal-page lick families (after the encoder + iti_licks.extract)
1. iti_zscore_check / iti_area_split_sd, iti_june_ref, iti_basis_decoder / frame_sd, iti_lick_order (KEY preds), iti_where_decoder, lick_covariate_reference.
2. Encoder re-summaries: encoder_split_rows -> frozen_fr_musall_v2, movement_null_plan_maps -> simple_key_figures.
3. lick_cued_ramp (cued_wide) -> lateral_ratio / vigor / ramp_animals / balance_imaging_control.
4. lick_precue_restw -> _traces -> restw_strict_check -> restnl.py -> precue_restw (restnl) -> revup_neural -> lick_cue_to_lick_extract -> cue_to_lick(_precue) -> restnl_rescue -> arrival_to_cue / stopped_trial_traces / disengagement_tests.
5. lick_enlv_extract (lick_reader) -> enlv_balance_offset -> enlv_curves / target_code; lick_enl_angle_decode (enl_angle) -> enl_offset_explore / enl_target_test (enl_spout; Part A needs attempt_coding_pp LP).
6. lick_firstlick_extract -> balance_subsets (FULL), balance_contact.
7. The balance / recovery family on the KEY preds.
8. lick_offset_contrib_map -> offset_where / terms_recovery / sspn_checks / acad_midline / iti_prelick (load-only lick_reader, so step 5 must come first).
9. Decoders: refit -> area_importance -> unique_maps / block_compare; nonlinear -> nonlinear_preds; key_figure.
10. lick_code_subspace (lick_code) -> noisematch -> rotation_summary; reembedding_test (reembed).
11. Session-level maps.
12. Interhemispheric chain.
13. lick_correction -> prelick_hemisphere_tuning -> prelick_svd_tuning (prelaunch_svd) -> prelaunch_aim_tracking (prelaunch_aim) -> face_prelaunch_aim -> face_offset_partial (face_aim).
14. Dock subsets.
15. Pilots (UNSURE).

### Reconciliation with the frozen-default scripts helper (arrived after the rows above were written)
The helper's counts are 193 REDO / 30 UNSURE / 166 NO; my CSV has 209 / 11 / 169. The gap is in borderline cases.

| Script(s) | Helper | My CSV |
|---|---|---|
| hemo / channel diagnostics: rest_migration channel_*, rest_coupling, nvc_evoked, evoked_hrf_latency, plot_session_residual | UNSURE | REDO |
| cohort-1 QC: epileptiform_*, prestroke_quiet_videos | UNSURE | NO |
| frozen_group_leakage, qc_running_survey, qc_running_push_survey | NO | REDO (template_job loads imaging but uses only behavior) |
| video_motion_session | NO | REDO (LocaNMF C used only for frame times) |

The helper's verdict should win for these. If the CSV disagrees, read the code.

#### Extra traps from the helper
- **O2 encoder bundles are not rebuilt.** `M:/.../DeepLabCut/Widefield/o2/enc_v36_full_20261024/inputs/*.pkl` holds adopted imaging, and inputs already in a bundle are never rebuilt. This affects o2_encoding, misdirected_lick_decode and fr_direction_mechanism. Build a new bundle.
- **Quiet v2 hard-codes a 0806 cutoff.** The quiet v2 scripts use `>= "0806"` for the August pre (August starts at 0806 after the 0805 exclusion), not 0801. quiet_area_spikes v1 uses June+Aug.
- **PRE_FROM is required by some scripts and refused by others.**
  - cd_loo_sessions and epoch_sessions_long REQUIRE WIDEFIELD_PRE_FROM=0801.
  - map_drift_prepre and restnl_mask REFUSE it.
  - So they cannot share one env block.
- **summary_figures_1021 imports from another repo.** It puts C:/Users/SabatiniLab/Github/widefield_pipeline on sys.path, so it imports `wfield_local` from main.
- **More skip-if-exists caches to prune:**
  - frozen_encoder_<tag>/session_vs_frozen/sessions/*.json;
  - specificity_control `--reuse` sessions.csv;
  - residual_roi spout cache; epoch_kernels/stats/*.pkl; layer2_random_splits part csv;
  - cohort_out/enl_fr_removed_decode/sessions; enl_movement_multisession done-set;
  - frozen_kernel_maps vessel cache; spout_unique_maps cache; map_test_calibration `--cache`;
  - iti_licks basis_*, june_ref, zscore_check/session_sd.csv, session_patterns, decoder_pattern_stats, lick_order_preds*, hemo_check/cache.
- **Rerun publish_cache.py afterwards.** It copies the session cache, so republish after the re-render.
- **Map-level preprocessing products** (rebuild_lick_maps, rest_migration/regen_lick_maps, regen_running_maps) are REDO for the preprocessing deck.
