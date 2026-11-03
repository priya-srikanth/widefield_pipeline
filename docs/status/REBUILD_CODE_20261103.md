# Cohort-1 rebuild: CODE preparation log (2026-11-03)

Worktree `C:/Users/SabatiniLab/Github/wf_rebuild`, branch `rebuild/cohort1-20261103` off `origin/main` 11e8a846.
NOT pushed. main and the other worktrees were not touched. No result renders; only the step-3 reproduction checks
(scratch outputs). Decisions NOT taken here: the drift-step design / cutoff, and LocaNMF refit vs projection.

## 1. Integration merges (`git merge --no-ff`, in order)

| # | branch (tip at merge) | merge commit | conflicts and resolution |
|---|---|---|---|
| 1 | analysis/restnl-maps (1cb58c9c) | 89664325 | none |
| 2 | analysis/lp-final (5822cbad) | 2b182c76 | `wfield_local/epoch_figures.py`, 4 hunks. (i) significance marks: main = top band for the default open-dot style (epoch-coloured, alternating rows), lp-final = top band for every style (one row per post epoch, y ticks trimmed). KEPT main's default path exactly; lp-final's top band now covers `dot_style="animal"`, which main still drew above the bar (both Priya 10-30). (ii) `reserve_headroom`: same split. (iii) `contrast_panel` tail: kept main's `write_session_values` AND lp-final's `save_figdata`. (iv) end of module: kept main's `replot_from_sidecars` family AND lp-final's `save_figdata` / `replot_figdata` (independent additions). |
| 3 | analysis/rest-decoder-nl (7c53214a) | 22959cec | none |
| 4 | analysis/frozen-default (876c87b0) | 6622a3c3 | `wfield_local/frozen_models.py` `make_spec` tail: kept main's `scaling` field (state decoder session_z) AND frozen-default's `lp_trained()` rule (LP-trained default, Priya 11-03). |
| 5 | analysis/cohort1-swapmask (2c3fccfb) | db711802 | none |
| 4b | analysis/frozen-default again (1922a47b, section G conversion, committed there after the first merge; coordinator 11-03) | 4b1919f0 | none |

NOT merged: analysis/cohort1-swapmask-enc (old-commit comparison arm). wf_frozendef's uncommitted section G edits were
left out; they were later COMMITTED on frozen-default (1922a47b) and merged as 4b. NOTE: helpers kept committing in
wf_swapmask (drift check, LocaNMF test) -- anything after 2c3fccfb is NOT in this branch.

Tests after the merges (frozen copy of db711802, 4 shards, 4,284 tests): see section 4.
Merge breakage found and fixed: `tests/test_rest_frozen_decoder_nl.py::test_only_collect_changed` (rest-decoder-nl
pins every function of `rest_frozen_decoder` to 0e592a20; main changed `_plot` -- animal colours from config, the
cohort-2 change). Fixed by allowing `_plot` (and the rebuild's routed `_epoch_dir`) in its CHANGED set, commit 335f23f3.

## 2. Code changes (commits on rebuild/cohort1-20261103)

| commit | items |
|---|---|
| 250b801e | a, b, d, i, j + the run-mode core (`run_mode`, `trial_classes` run gate) |
| 509cd2bd | c (writer routing) |
| 335f23f3 | e, f, g, h, k + fixed-name JSONs + the merge test fix |
| ce5ecf6a | step 3 scripts |

(The items are entangled across shared modules, so they share commits; each commit message lists its items.)
DEFAULT BEHAVIOUR: every default path, cache key, output folder and file name is unchanged (pinned in
`tests/test_rebuild_switches.py`), EXCEPT the two changes that are cold-cache / new-id by design: (e) every frozen spec
id, and (h) the map / prm / restnl cache digests.

### a. One hemo switch
- `defaults.yaml hemo.variant` (+ `WIDEFIELD_HEMO_VARIANT`) = `config.hemo_variant()` now drives:
  `config.svtcorr_path` (as before), `hemo_variants.ADOPTED` (the READ suffix, resolved at import; `adopted()` at call
  time), `combined_encoding.hemo_dir` / `load_imaging` (was env-only: trap T2), `config.locanmf_dir_name` (follows the
  variant: `locanmf_affine8v1_hemo_<variant>`; was always adopted: T7; `WIDEFIELD_LOCANMF_VARIANT` still wins).
- New: `config.hemo_dir_name`, `hemo_manifest_path`, `hemo_tag`, `UNTAGGED_HEMO_VARIANT = meegkit_hpfit` (the variant
  untagged outputs hold; fixed on purpose so flipping defaults.yaml still tags), `default_hemo_variant`.
- Builders use the new `hemo_variants.BUILD_DEFAULT` (a VARIANTS key): `hemo_variants --variant` default,
  `cohort2_qc/swapmask_1029.py`, `swapmask_cohort1.py`.
- Scripts: `slowband_decode_check` reads `config.svtcorr_path`; `iti_hemo_check` labels the selected variant and gets a
  selectable reference row `ITI_HEMO_REF_VARIANT=<suffix>` (own cache dir); `iti_mixed_models` follows (`il.OUT`
  routed); `postcue_window_test --read-selected` reads the built variant (it recomputes VARIANTS keys otherwise);
  `encoding_drift_check`, `quiet_hemo_check`, `o2_video_me` follow via load_imaging / ADOPTED.
- Test: all paths agree for a non-default variant set by env AND by defaults.yaml.
- NOT DONE (blocked by the worktree rule): `lick_balance_imaging_control(+_primary)`, `lick_acad_midline_control`,
  `s1_refit_basis_table` exist only on wf_attempt (analysis/attempt-coding) and depend on wf_attempt-only scripts.
  They hard-code `hemo_meegkit_hpfit/manifest.json` / `hemo_{ADOPTED}`. Fix there: use `config.hemo_manifest_path(res)`
  / `config.svtcorr_path(mc)` (both now exist), or merge attempt-coding into the rebuild.

### b. Joint basis
- `defaults.yaml joint_basis.pins` (NOT under `locanmf:`, whose items enter every basis_id): PS92 76d884873920, PS93
  0141bb830372, PS94 a5e70d5ecd36, PS95 a9170190179c (the 08-15 bases, meegkit_hpfit; = the newest on disk today).
  `joint_locanmf.load` serves the pin, never the newest (T13); an explicit `bid` still wins; unpinned animals
  (cohort 2) unchanged.
- `project_all(basis)`: read variant != `basis_hemo_variant(basis)` (manifest field, recorded by `build` from now on;
  else the pin's; else legacy meegkit_hpfit). So a defaults.yaml flip now projects in-fit sessions (T3).
  `WIDEFIELD_JOINT_PROJECT_ALL=1` still forces it.
- `Basis.signal` REFUSES stored C under project_all. This exposed a latent mixing bug: `nolick_decoder._joint_signal`
  served stored C to in-fit sessions regardless of the variant -- fixed.
- Tests: pin wins over a newer basis; unpinned keeps newest; under a variant the in-fit session is projected in
  BasisSource and signal_or_project and the stored C is refused; nolick projects.

### c. Output tagging
- `wfield_local/run_mode.py`: parts hemo (`hv-<v>`), lnmf (`lnmfproj`), pre (`pre<MMDD>`), selection
  (`sel-camera-<window>`), refs (`refs-<names>`), inmem (`inmem-<families>` for any `WIDEFIELD_<F>_FROZEN=0`),
  lptrained (`lpdaqtrained`). `lp_licks.output_tag` = LP/ENL/gate tag + these parts, so `lp_licks.output_dir` routes,
  `writeguard` REFUSES a tagged run's share write outside its tag folder, and `stamp` labels figures.
- Routed (were writing to default folders): cd_trajectories, combined_encoding.out_dir, frozen_encoder.out_root,
  quiet events root (`QS_OUT_ROOT` still wins), encoding_drift_check, slowband (qc), chronic_drift_vs_reorg,
  enl_movement_multisession (skip-if-exists), imaging_lmm, null_potent_session, replot_map_figures, 18 rest_migration
  writers, `scripts/iti_licks` OUT -> `iti_licks/<tag>` (skip-if-exists extracts), preprocess_deck output.
- Fixed-name JSONs: `section_g.json`, `coding_direction.json`, `joint_xsession_decoder_<align>.json` written and read
  as `run_mode.keyed_name` (grant 2/2b/4, 3a/5b, 3b/5 via `grant_kit._mode_name`; section_g_figures; miss_vs_stopped).
- Per-session cache keys: see d.
- Not exhaustive: the reconciler's list of "13 writers" was not in the inventory files, so the set above came from a
  grep of share writers with no writeguard / output_dir call. Any writer that calls `assert_writable` is covered by
  the refusal anyway. Remaining wf_attempt-only skip-if-exists writers (T18 list) need the same `run_mode.subdir` there.

### d. Prune guards
- `session_cache.effective_kind`: kind += `run_mode.cache_tag()` (hemo variant, LocaNMF projection, selection; tf-*
  included as agreed), so the label x kind prune can never delete another mode's entries and a variant can share the
  cache dir. Readers that glob the cache (`BasisSource.variance_captured`) use it too.
- `restnl_mask` file stem `<label>__restnl-<tag>` in a tagged run; the prune globs its own stem only.
- CACHE_VERSION not bumped (keys change only under a tag). restnl_mask's own `_code()` digest moved (module edited):
  its baselines are cold once.

### e. make_spec fingerprint (FREEZE_VERSION 2)
- `train_data` REQUIRED; spec gets `train_fp` (`data_fingerprint`), `selection`, `hemo_variant`. All 15 kinds pass
  their training rows (list in docs/FROZEN_MODELS.md section 8).
- `_write` raises `LegacyStoreError` (a SystemExit, so per-animal `except Exception` cannot hide it) when the store
  holds v1 models for the animal: the rebuild MUST set `WIDEFIELD_FROZEN_MODEL_DIR` to a fresh store. The existing store
  `C:/Users/SabatiniLab/frozen_models` was not written; tests use temp stores.
- LP DAQ-trained cross-check: `train_fp = "from-daq-run"`, found by `find_daq_trained` (all fields but fingerprint /
  selection); ambiguous -> raise.
- Lick-landing read-outs (coordinator 11-03, DECISIONS fa70b651): only the SPEC of `frozen_readouts.pre_loso_ridge`
  changed; `ridge_fit` / `fit_all` / `data_sig` pinned by source sha1 and stored payload == in-memory fit (exact). Their
  per-session frame-level SD scaling lives in the wf_attempt read-out scripts and was not touched.
- Tests `tests/test_frozen_fingerprint.py` (11) + updated callers in 4 existing test files.

### f. Camera "working"
- `defaults.yaml trial_selection.gate: daq` (untagged) | camera; `WIDEFIELD_TRIAL_GATE`, `WIDEFIELD_TRIAL_WINDOW`
  (default cue_stop). KEPT DAQ AS THE DEFAULT so the default stays identical until the rebuild flips it (one line).
- `trial_classes`: `gate_mode`, `run_selection` / `run_selection_tag`, `cue_classes` (moved from
  scripts/jaw_only_incomplete_figs, which delegates; equality pinned), `cue_classes_for` (THE A5 HOOK: route through
  `responded()` here when e3 lands it), `row_classes`, `pool_row_classes`, `camera_select`, `camera_rows`.
- Wired: `grant_kit._class_select` (+ `_pooled_bundle["cls"]`; callers in grant_kit, grant_confusion, grant_similarity,
  transfer_arms; `rule="daq"` for s1_refit_locanmf_table's quit-tail robustness row; raises without classes),
  `beta_maps.session_maps`, `position_reference_maps._working_xy` (working / miss_working / stopped),
  `hemo_map_control`, `position_coding_directions` (miss-working / stopped). Lick class = contact-only, unchanged
  (PS93_0813 stays); "unknown" (no video, unmatched cue; PS92_0908 17.5 %) is in neither working nor stopped.
- Camera "stopped" = class no_response. CHOICE TO CONFIRM with Priya (the instruction defined working, not stopped).
- NOT YET WIRED (each builds its own working / engaged class): `nolick_decoder.categorize`, `cd_trajectories` gate,
  `attempt_coding.classify` / `attempt_maps` (LP attempt classes), `poststroke_compare.poststroke_engagement`,
  `precue_engagement_states` states 5/6, the rest-baseline gates (rest_engagement / rest_by_position). Under
  `gate=camera` they still use the DAQ rule -- their outputs ARE tagged sel-camera, so a render would mislabel them.
  Must be wired (or refuse under camera) before a camera render.
- Measures of engagement (quit, decline, attempt rate) untouched.

### g. August-only pre per render block
- `--pre-from MMDD|all` on `epoch_grant_figures` and `grant_figures` (`config.apply_pre_from`); no config default.
  Outputs tagged `pre<MMDD>`. `june_august_drift` refuses PRE_FROM (`config.require_full_pre`); 12.1b
  (`rest_frozen_decoder_nl`) already refuses it and selects its pre set itself.

### h. Cache rebuild items
- `WIDEFIELD_PRM_LEGACY_DATE8` removed (position_reference_maps, lp_final_render, restnl_maps).
- Map digests narrowed: `map_collect.closure(module, *roots)` = the roots + their in-module transitive references +
  the module's imports; `_raw_maps_code` / `beta_maps._session_maps_code` hash only the functions the map builders call
  (quiet_variant / quiet_frame_path / quiet_baseline_svt, engaged_frame_mask, rest_frames_by_position /
  restw_from_frames / MIN_POSITIONS_FOR_WEIGHTED, engagement_gate). TRAP: `cache_kind` still hashes the WHOLE
  defaults.yaml -- any yaml edit (this branch added two blocks) colds every map cache.
- `precue_engagement_states`: position decoder, state discriminator and its D2 LOSO refits through the store (kind
  engagement_states; `WIDEFIELD_ENGSTATES_FROZEN=0` = in memory, identical fits).
- 5rm / paired_matched: ONE seed-keyed spec per animal x align x class (kind matched_frozen, align carries the seed
  base, payload keyed `pool|n_target|seed`, fingerprint over the whole bundle); hit = replay, miss = fit in memory as
  before and store. Numbers identical by construction (same rng seeds); NOT yet verified on data (no render allowed).
- `lp_final_render --trained` defaults to lp for the LP source (T16).

### i. nolick summarize KeyError -- fixed, test added.

### j. Per-session LocaNMF projection path (default OFF)
- `wfield_local/locanmf_session.load_C`: every per-session C load in wfield_local (10 sites) + movement_encoding_session
  goes through it (grep test). Default = `np.load` of the same file. `WIDEFIELD_LOCANMF_PROJECT=1`: A / regions from the
  stored decomposition, C = LS projection of the selected variant's SVTcorr onto A (inside-footprint pixels;
  session-cached, tagged `lnmfproj`). Exactness tested on synthetic data. Not run on real data (decision pending).

### k. Preprocessing map npz
- `config.maps_dir_tag()` = `affine8v1` + `_hv-<variant>` under a non-adopted variant: `preprocess._maps_commands`
  (cue / lick / running dirs), `position_evoked_maps.session_npz` (15e), `preprocess_deck` discovery. Registration
  (`allen_aligned_affine8v1`) and quiet masks stay untagged (hemo-independent).

## 3. Step 3: lost / share-only figure scripts (`scripts/rebuild_figs/`, helper agent)

All reproduce the current figure pixel for pixel on the current data; `--plot-only` is pixel-identical to the full run.
Scratch: `C:/Users/SABATI~1/AppData/Local/Temp/claude/C--Users-SabatiniLab/f082ee0c-b33c-49a8-a537-dcee3bb7c84c/scratchpad/rebuild_figs/`.

| figure | script | from | numbers | image |
|---|---|---|---|---|
| imaging 0.2 | fig_behaviour_hits.py | share fig_behaviour_hits.py | CSV diff 0 | identical |
| imaging 1.4 | fig_decoding_sessions.py (+ --pre-col/--post-col) | share | CSV diff 0 | identical |
| imaging 3.3 | fig_s_per_animal.py | per_animal_tables + state_table_zscored + per_animal_fig | tables diff 0 | identical under matplotlib 3.11.2 (fr / roicat env); ~7 % antialias pixels under locanmf 3.10.9 (original too) |
| imaging 3.4 | fig_state_position_sessions.py | share | CSV diff 0; caption stats match | identical |
| imaging 3.5 | fig_state_ps94_diagnostic.py | share (not _balem) | CSV / confusions diff 0 | identical (run with widefield_pipeline's wfield_local: needs v1 state models; default BLAS threads) |
| imaging 4.4 | fig_frozen_refit_pooled.py | share | 3 CSV sections diff 0 | identical |
| imaging 5.1 / goal 8.1 | fig_goal_summary.py | recovered | - | identical (also fig1 / fig3 of the goal page) |
| imaging 5.2 / goal 8.2 | fig_goal_epochs.py | recovered | - | identical |
| imaging 11.2 | fig_enl_move.py | rebuilt from PNG + caption + CSV | - | identical (10-13 CSV backup; live CSV adds PS92_0812 = 0.08 % px) |

Findings: BLAS thread count changed 27 PS92 / PS93 rows of 3.5's in-memory class-balanced arm by 1-2 segments
(balanced accuracy <= .005) -- contradicts CLAUDE.md's "bitwise across thread counts" for that `fit_frozen` path.
3.3 suptitle says "engaged trials" (inputs are working trials). 5.1 / 5.2 suptitles hold fixed 10-24 text.

## 4. Test results

**After the five merges** (frozen copy of db711802, 4 shards, ~4,100 run + 173 skipped, 1 h 30 m): 2 failures, both
merge breakage, both fixed in tests only:
- `test_rest_frozen_decoder_nl::test_only_collect_changed` (main's `_plot` change; commit 335f23f3);
- `test_reference_registry::test_no_caption_without_a_reference` (restnl-maps' DERIVED reference `restnl` has a
  caption; the test now allows `DERIVED_REFERENCES`; commit e947bf1c).

**Rebuild changes**: new tests `tests/test_rebuild_switches.py` (37) + `tests/test_frozen_fingerprint.py` (11): all
pass. Targeted existing suites (config, swapmask overrides, session cache, joint LocaNMF, LP gate, LP licks,
writeguard, trial classes, map collect, combined encoding, await / batch LocaNMF: 160 passed; frozen / rest / nolick /
section G / stopped: see the full run below). Two existing tests asserted the old identity by design and were updated
(LP DAQ cross-check equality now by `base_id`; the CD reference test passes `train_data`), e947bf1c.

**Full suite on the final code**: see the line appended at the end of this section.

## 4b. analysis/attempt-coding (coordinator, 11-03): TRIAL MERGE ONLY, NOT RESOLVED

Trial `git merge --no-ff --no-commit analysis/attempt-coding` (tip 1ce49f85, 605 commits ahead of main) onto e947bf1c:
**60 conflicted files** (52 add/add, 8 content) -> above the ~30 threshold, so the merge was ABORTED unresolved.
- 52 add/add: the ITI / lick family scripts (`scripts/iti_*` x45, `cued_misdirected_licks`, `early_predictors`,
  `goal_page_reorder`, `lick_aim_deficit`, `lick_gain_prediction`, `lick_matched_readout`, `recovery_timing`,
  `residual_roi`, `specificity_control`). CAUSE: they were cherry-picked onto main around 10-27 (different commit ids,
  e.g. iti_licks d767c8a9 on main vs b47c5bee on attempt-coding) and then developed on BOTH sides: attempt-coding
  kept developing them (10-27 .. 11-01: docked-spout ITI primary, masked joint LocaNMF, ALM / tjM1 split, ...), while
  main / frozen-default converted them to the stored `frozen_readouts` helper (c61ac5ce, 11-03) and this rebuild
  routed `iti_licks.OUT` / `iti_hemo_check`. Largest: iti_licks 68 hunks, iti_recovery_link 23, iti_check_figures 22,
  lick_matched_readout 20, iti_session_cluster_tests 20, iti_lick_order 19, iti_hemo_check 18.
- 8 content: DECISIONS.md, docs/FINDINGS.md, docs/POSE_MODELS.md, STATUS_2026-10-26_ENCODING_HANDOFF.md,
  SVD_GAUSS_SUPERSEDE_QUEUE_2026-10-27.md, pyproject.toml, `wfield_local/{frozen_encoder (4 hunks), lp_licks (2),
  map_collect (1), session_cache (1), sig_marks (1)}.py`.
- Suggested resolution (needs an owner of the goal-page scripts): for each script take attempt-coding's newer body,
  then re-apply (1) c61ac5ce's `frozen_readouts.pre_loso_ridge` conversion (and its lick-landing scaling pin), (2) the
  rebuild's run-mode routing (`il.OUT` via `run_mode.subdir`, `iti_hemo_check` reference row), (3) the item a fixes for
  the wf_attempt-only hard-coded-ADOPTED scripts. The five wfield_local files are small and resolvable by reading.

## 4c. Imaging section 1 (from session 5b), run order -- needs attempt-coding merged first
1. main `scripts/s1_trial_class_snapshot.py`;
2. `s1_refit_basis_table.py` (wf_attempt only), under LP (`WIDEFIELD_LICK_SOURCE=lp WIDEFIELD_ENL_WINDOW_S=2.0`) and
   under DAQ;
3. main `scripts/s1_refit_locanmf_table.py` (its quit-tail row now forces `rule="daq"` in `_class_select`);
4. main `docs/results_pages/imaging/p_img_s1_table_20261103.py`.
Steps 2-4 read SVTcorr / the joint basis (so the hemo switch, project_all and a fresh frozen store apply).
`s1_refit_basis_table` hard-codes `hemo_{hv.ADOPTED}` and `ce._load_imaging` (now both follow the switch once merged).

## 5. Blocked / for Priya / open
- Drift-step design + cutoff, LocaNMF refit vs projection: not touched (the variant build, D9 removal of the
  encoder / ITI / stray_licks drift steps, and `build_swapmask_cohort1.py` parameterisation wait on them).
- wf_attempt-only scripts (item a list; T18 skip-if-exists writers): need edits on attempt-coding or its merge.
- Camera gate: wire (or make refuse) the not-yet-wired gates listed under f; confirm camera "stopped" = no_response;
  flip `trial_selection.gate` at render time.
- Deck captions (deck_registry / deck_text / locanmf_analysis_deck) still state the old rules (T11): not edited.

## 6. New traps
1. Every frozen id changed (FREEZE_VERSION 2): on this branch ANY fit into the old store stops the run
   (LegacyStoreError). Set `WIDEFIELD_FROZEN_MODEL_DIR` to a fresh store for every render; `find()` still consults
   the M: store (harmless: ids differ).
2. Tags are relative to meegkit_hpfit / the DAQ gate, NOT to the current defaults: after flipping defaults.yaml to the
   final variant and camera gate, every output lands under `.../hv-<variant>_sel-camera-cue_stop/...` (by design).
   Pass `--output` / OUT roots anyway.
3. `lp_licks.output_dir` now also routes non-LP modes; a writer that builds its path from `output_tag()` must take the
   whole tag as ONE path segment (writeguard checks `/<tag>/`).
4. Map cache keys hash the whole defaults.yaml (`map_collect.cache_kind`): any yaml edit colds them.
5. `hemo_variants.ADOPTED` is resolved at IMPORT: set `WIDEFIELD_HEMO_VARIANT` before the process starts (in-process
   code should call `adopted()` / `config.hemo_dir_name()`).
6. Under a non-default variant without `WIDEFIELD_LOCANMF_PROJECT=1`, every per-session LocaNMF consumer FAILS
   (directory `locanmf_affine8v1_hemo_<variant>` does not exist) -- deliberate, until the LocaNMF decision.
7. Camera gate: half the gates are not wired yet (section f); a camera run of those families is mislabelled.
8. 3.3 / 3.5 reproduction depend on matplotlib version and BLAS threading (section 3).
9. The scratch test worktrees `<scratchpad>/wf_mergetest` and `wf_headtest` are detached worktrees of this repo:
   remove with `git worktree remove` when done.

## 7. Final test record (appended)
Full suite on a frozen copy of 335f23f3 (3 shards, ~54 min): 4,159 passed, 175 skipped, 8 failed. The 8 were all
tests that asserted the OLD design and were updated afterwards, then re-run green on the branch tip:
- test_lp_licks x2 (LP DAQ cross-check spec equality -> base_id), e947bf1c;
- test_frozen_default::test_cd_reference_stored_equals_the_old_inline_fit x4 (train_data now required), e947bf1c;
- test_reference_registry::test_no_caption_without_a_reference (DERIVED restnl; merge breakage), e947bf1c;
- test_frozen_default::test_collect_5c_..._paired_matched (5rm now stored by design; new test pins stored == replayed
  == in-memory records), d919eb55.
Branch tip d919eb55 + this doc. Scratch test worktrees removed.

## 8. Second helper (11-03 afternoon): attempt-coding merge, camera gate, map key, review fixes

### 8.1 Merge of analysis/attempt-coding (tip 734226d4, committed only): 3988cbe3
60 conflicts. **52 add/add.** Rule: attempt-coding's body, then this branch's OWN commits re-applied with a 3-way
`git merge-file` (base = the commit's parent blob, new = the commit's blob). "Own" commits were found with
`git log --cherry-mark --right-only attempt-coding...HEAD -- <files>`: 509cd2bd, 250b801e, c61ac5ce, 4fbfaa51, 34b3013b
(335f23f3 touches none of the 52). For every file without an own commit, HEAD's blob is in attempt-coding's history,
so attempt-coding's version is a strict descendant.

| file(s) | resolution |
|---|---|
| 39 scripts: cued_misdirected_licks, early_predictors, goal_page_reorder, iti_area_maps, iti_check_figures, iti_decoded_decomposition, iti_decoder_pattern_stats, iti_decoder_patterns, iti_extension_dist, iti_farcontra_pattern, iti_june_ref, iti_lick_rate, iti_lmm_figures, iti_mixed_models, iti_overcorrection, iti_pattern_area_stats, iti_recovery_link / marker / measure, iti_session_cluster_tests, iti_session_pattern_tests, iti_spout_animal_panels / pattern_stats / patterns, iti_where_beyond_pull / drift / mae / per_spout / rmse_split / signed_error / spread_matched, iti_zscore_check, lick_matched_readout, recovery_timing, residual_roi, specificity_control | attempt-coding byte for byte (no own commit) |
| iti_decoder_contrib, iti_where_decoder, iti_where_scatter | attempt-coding + c61ac5ce applied cleanly |
| iti_basis_decoder | c61 by hand: frozen_readouts import + attempt-coding's `wd.basis_dir` / `il.SESS_NAME` lines |
| iti_contact_check | c61 by hand: attempt-coding's `il.split_only` + `il.area_features` features, stored fit |
| iti_first_lick, iti_where_3axis, lick_gain_prediction | c61 by hand (import hunk; the now-unused `feature_names` import dropped) |
| iti_lick_order | c61 by hand: stored fit, then attempt-coding's `CONTRIB_HOOK` |
| iti_where_angle_reach | c61 by hand (k=3 on attempt-coding's 4-column Y: the same fit of the first three columns) + CONTRIB_HOOK |
| iti_licks | c61 clean; 509cd2bd by hand: `OUT = run_mode.subdir(E / "iti_licks")` at the head of attempt-coding's OUT_W / ITI_SUBSET / ROI-version block |
| iti_hemo_check | 250b801e by hand: attempt-coding's ROI-version cache name, `_ref-<variant>` suffix only under ITI_HEMO_REF_VARIANT (default = attempt-coding's path) |
| lick_aim_deficit | attempt-coding (`rm.aim_licks()` reads the animal column, so 34b3013b's `[:4]` fix is moot) |
| wfield_local/sig_marks | = HEAD (code identical on both sides; HEAD's comment kept) |

**8 content.** `frozen_encoder`: attempt-coding's `frame` Musall variant and `--out-root`, AND the rebuild's routed
`out_root()` (`--out-root` wins, also for `--summarise-drop`). `lp_licks`, `session_cache`, `map_collect`: HEAD
(attempt-coding had nothing new in those hunks; gate-LP is already in `effective_kind` / `output_tag`). `pyproject.toml`:
both per-file ignores. Docs, both sides in date order:
- DECISIONS: HEAD's 10-25 FaceRhythm entry; the 10-27 side-camera addenda, then attempt-coding's 10-27 / 10-28 entries,
  HEAD's 10-29 / 10-30, attempt-coding's 11-02 evening, HEAD's 11-02 late .. 11-03. Every heading present once (checked).
- FINDINGS: one-side additions; the 4d0 "Behaviour on the same cued licks" line takes attempt-coding's 10-28 rewording.
- POSE_MODELS: attempt-coding's 11-01 local-cleanup note stays with the cam4 / cam1 parts it describes, then HEAD's
  Part E (side cameras; those models were not deleted).
- 10-26 encoding handoff, SVD_GAUSS queue: attempt-coding's additions.

After the merge (c17be8f5): `lick_balance_imaging_control(+_primary)` and `lick_acad_midline_control` read
`config.hemo_manifest_path(res)`; `s1_refit_basis_table` reads `config.svtcorr_path(mc)`. Same paths by default.
Later: 12 attempt-coding `make_spec` calls had no `train_data` and would have raised under FREEZE_VERSION 2
(face_offset_partial x2, face_prelaunch_aim, lick_code_subspace, lick_enl_angle_decode, lick_enl_target_test,
lick_enlv_extract, lick_iti_prelick, prelaunch_aim_tracking x2, prelick_svd_tuning, reembedding_test). Each now passes its
own training-data digest (the `xy_sha1` the script already computed, or the rows). The prelaunch_aim_tracking cued_PRE
re-lookup reuses the spec made in the same run (the fingerprint cannot be rebuilt there; skipped loudly if absent).
NOT converted to `frozen_readouts` (inline pre-LOSO ridge in attempt-coding-only scripts): face_offset_partial
(fit_dec), lick_balance_blind_decoder, lick_nonlinear_decoder.

**Default path of the goal-page scripts.** 39 / 52 are attempt-coding byte for byte. In the other 13 the only changes
are the stored read-out (pinned equal to the inline loop by c61's tests), the run-mode root (identity by default) and the
iti_hemo_check cache name (= attempt-coding's by default). REAL-DATA IDENTITY CHECK (read-only, scratch frozen store):
merged `iti_first_lick.predictions()` vs the committed share table `iti_licks/first_lick_preds_splitr8.csv.gz`
(160,166 licks x 7): every column identical, `dec` max |diff| 4.4e-16 (float summation order).
TRAP: the opt-out `WIDEFIELD_ITI_FROZEN=0` adds the run-mode part `inmem-iti`, so `il.OUT` moves to
`iti_licks/inmem-iti/`, where the per-session extracts are not (FileNotFoundError, loud). Any non-default mode of the
ITI family needs its extracts rebuilt in its own folder, including modes that do not change them (inmem, pre, refs).

**Tests on the merge (c17be8f5, 6 shards, frozen copy):** 4,359 passed, 176 skipped, 11 failed. All 11 are
`test_parallel_results_are_ordered` on attempt-coding scripts and FAIL IDENTICALLY ON THE attempt-coding TIP (checked in
a detached worktree): fan_out results consumed unsorted (curbd_pilot x2, iti_decoder_pattern_stats, lick_code_subspace)
and runtime globals that spawned workers never see (`SFX`: arrival_to_cue_traces, lick_cue_to_lick, revup_neural,
stopped_trial_traces; `JOBS`: lick_iti_prelick x3). Not fixed here (goal-page owner; SFX may change output names).

### 8.2 Camera gate (8d169197; review fixes 5361b2fa)
WIRED under gate=camera: nolick_decoder.categorize (-> cd_trajectories classes, enl_decode arms, the nolick undetected
split; an undetected cue in neither miss_working nor stopped leaves the categorisation), poststroke_engagement /
nolick_state_vector / undetected_state_split, precue_engagement_states states 5 / 6 (+ D1), attempt_coding.classify,
attempt_maps.classes_for_rows, position_axes, grant_confusion post_working panels, and the rest-baseline gates
(rest_engagement.engaged_by_cue / engaged_frame_mask, rest_by_position._engaged_trials; through them rest_frozen_decoder,
rest_position_decode, rest_position_permutation, shared_position_projection, rest_block_boundary, cd_trajectories'
rest baseline, attempt_maps' time-local rest). Previously wired: grant_kit, beta_maps, position_reference_maps,
hemo_map_control, position_coding_directions.
FAIL CLOSED: `engagement_gate` raises `trial_classes.DaqGateRefused` (a SystemExit, so the fail-open rest gates cannot
swallow it) under camera unless `daq_quit_rule(reason)` is open. `_gate_all` and the grant bundle hand out a
`RefusedMask`; the real mask stays in `bundle["not_eng_daq"]` for `rule="daq"`. Exempt (measures, diagnostics, labelled
DAQ rows): spout_behavior.flag_engagement, disengagement_tests, quit_point, gate_lp_check, lp_gate_option_counts, the
s1 tables' quit-tail rows. Refuse up front (`require_gate`): enl_states.states_for, enl_state_counts,
stopped_trial_traces, channel_position_maps; engagement_axis_balance / nocontact_census refuse on the mask.
Register of every call site: tests/test_camera_gate_fail_closed.py (a new call site fails until classified).
CAMERA "STOPPED", DECIDED (Priya 11-03, replaces the provisional "stopped = every no_response"): a no_response trial is
stopped only inside a run of >= `behavior.tail_min_misses` (6 = the task auto_hold_after_miss threshold) consecutive
camera no_response trials; "unknown" trials are skipped when counting (they neither break nor extend a run) and are
never stopped; an isolated no_response is in NEITHER working nor stopped. Implemented once, in
`trial_classes.apply_stopped_rule`, applied by `cue_classes_for` (sustained no_response relabelled class "stopped";
`STOPPED_CLASSES = {"stopped"}`), so every stopped consumer follows it: section 11 / ENL states (`enl_states.states_for`,
now wired with `cam_cls`; precue_engagement_states 5 / 6), epoch 12s / 12b and every map / decoder stopped arm
(`camera_rows`, `camera_select`), miss_vs_stopped (coding_direction.json from position_coding_directions), the nolick
categories. Sensitivity: `trial_selection.stopped: any` / `WIDEFIELD_TRIAL_STOPPED=any` (every no_response). The rule
name is in the selection tag (`sel-camera-cue_stop-stopRun6` / `-stopAny`), so caches and frozen specs separate.
Tests: `test_sustained_stop_rule`, `test_enl_states_camera_branch` (tests/test_camera_gate_fail_closed.py).
NOTE: `cue_classes_for` must be given the session's FULL cue train (every current caller does), since runs are counted
along it. `scripts/enl_state_counts.py` still refuses under camera (it builds its own states).
CAMERA "WORKING" ENDS AT THE SESSION LAST CAMERA LICK (Priya 11-03; main b506c1e3, merged 25ce7efb): the merge kept
main's `before_session_end` / `in_working` and the rebuild's gate. `cue_classes_for` applies the cutoff once
(`apply_session_end`: working-gate trials after the anchor -> class "after_end", in neither working nor stopped, after
the sustained-stop rule), so every `in_gate(cls, "working")` consumer, `grant_kit._class_select` / `camera_select`
included, equals `in_working(cls, cue order, end=...)` (pinned: test_session_end_relabel_equals_in_working). Anchor
`trial_selection.end: lick` / `WIDEFIELD_TRIAL_END` (contact / close / none = robustness); it is in the selection tag:
`sel-camera-cue_stop-endlick-stopRun6`. Effect per main: -1.0 % of working trials (1.8 % acute), all jaw-only after
the last lick.
FULL SUITE on a frozen copy of 86280b8a (6 shards, ~52 min): 4,402 passed, 176 skipped, 12 failed. 11 = the
test_parallel_results_are_ordered cases that fail identically on the attempt-coding tip (section 8.1); 1 =
test_config::test_svtcorr_path_follows_the_configured_hemo_variant, which asserted the OLD "empty env = zerophase"
semantics that review L3 changed on purpose (test updated in the commit after 25ce7efb). The later commits (sustained
stop 464226fd, end-of-session merge 25ce7efb) were tested on the touched suites (camera gate, trial classes, rebuild
switches, stopped class, map key, state bal_em, frozen models / fingerprint: all pass), not on a second full run.
Still PROVISIONAL, for Priya:
2. miss-while-working = incomplete | jaw_only (a no-lick row with a camera "contact" is in neither; review M6);
3. rest-baseline "working period" = every cue up to the session's LAST camera-working trial (keeps the DAQ rule's single
   terminal boundary);
4. under the camera gate attempt coding's "noattempt" class is EMPTY (no_try IS no_response, which is never
   working), so the noattempt-vs-stopped contrast cannot be formed; isolated no_try trials are now in neither class.

### 8.3 Map-cache key (e04143c6; M4 widening 5361b2fa)
`map_collect.cache_kind` hashes `MAP_YAML_KEYS` = decode, lick_detection, segmentation, preprocess.maps.tag, hemo.variant
(every defaults() access in the builder modules, read by hand) instead of all of defaults.yaml. trial_selection is left
out on purpose (the selection is in the kind). tests/test_map_cache_key.py: unrelated keys keep the key, builder keys
move it, and the builder modules are re-read for new sections. One-time cold rebuild of bm-* / prm-* at the next render
(the gate edits move the code digests too).

### 8.4 Review findings (docs/status/REBUILD_CODE_REVIEW_20261103.md, 8c6edf55)
| id | fix | commit |
|---|---|---|
| H1 | `make_spec(lp_only=True)` for attempt_axes / iti_readout: always the real-row fingerprint, never DAQ_RUN_FP | 5361b2fa |
| H2 | 13 wfield_local figures_working writers, grant_kit._fig_root, enl_decode_figure, enl_sparsity_figure routed by `run_mode.subdir`; writeguard tag-checks the figures_working root | 5361b2fa |
| H3 | attempt_coding / attempt_maps out_dir routed (their skip-if-exists files are per mode) + assert_writable | 8d169197, 5361b2fa |
| H4 | `trial_classes.class_source_signature` (v36 trials / licks / windows, cam1 events, jaw_onsets, class code) = cache-only run_mode part `csig` under camera | 5361b2fa |
| M1 | run_mode part `projall` (output + cache) | 5361b2fa |
| M2 | await_locanmf / batch_locanmf / nightly_figs refuse under WIDEFIELD_LOCANMF_PROJECT=1 | 5361b2fa |
| M3 | keyed_name / subdir use the full `lp_licks.output_tag()` | 5361b2fa |
| M4 | prm digest + quiet_periods.{response_window_s, rest_mask, trial_exclusion, widen_bool_sparse, quiet_dir} and analysis_kit.response_window_s; sufficiency test | 5361b2fa |
| M5 | see 8.2 | 5361b2fa |
| M6 | GATES["miss_working"] = incomplete / jaw_only in every no-lick split | 5361b2fa |
| M7 | find_daq_trained matches selection; run_selection carries `-gateLP` | 5361b2fa |
| M8 | 5rm under the LP DAQ cross-check: in memory, logged, never stored | 5361b2fa |
| M9 | origin/main 65d9aee3 merged (trial in a separate worktree; 4 conflicts: make_spec scaling_params + train_data, locomotor_decoder bal_em + train_data, s1 table fanned unit_job + rule="daq" + not_eng_daq, FROZEN_MODELS state row); test_state_bal_em's pinned pre-rebuild ids relaxed (every id changed by design); Fig 3.4 provenance text corrected | e74a4579 + the L commit |
| M10 | lp_final_compare derives the final maps folder through lp_final_render.setup + lp_licks.output_dir. NOT re-pointed: readers of attempt_maps_restnl / epoch_restnl made under PRE_FROM (logged) | 5361b2fa |
| L1 | `_same_records` length checks; the frozen_readouts test re-reads a HIT and checks a one-row change misses | L commit |
| L2 | o2_video_me, quiet_hemo_check, iti_hemo_check read the switch at call time; locanmf_lick_aligned refuses under projection. The cohort2_qc scripts with a literal `hemo_meegkit_hpfit` (channel_drift, early_fade, perpixel_snr, processing_stages, session_qc, slowband_behaviour, slow_is_bleach, slow_spectrum, stage_traces, var_bands, spout_maps, quiet_modes, running_sign_check, indicator_kinetics_spread, and the variant-comparison scripts) stay literal: QC comparisons BY DESIGN | L commit |
| L3 | hemo_variants: no fallback (the import fails if config cannot load); an empty WIDEFIELD_HEMO_VARIANT = unset | L commit |
| L4 | pinned by a test (all-NaN cue table -> "unknown") | L commit |
| L5 | logged: `_BUNDLE_CACHE` / lru caches ignore in-process env changes (tests / notebooks only) | - |
| L6 | logged: joint-basis edge cases (explicit superseded bid without a manifest field -> legacy label; a missing pin raises) | - |

### 8.5 Review, second pass (REBUILD_CODE_REVIEW_20261103.md "Second pass", 630c277b)
All in the "second pass" commit after 630c277b; tests in `tests/test_review_second_pass.py` unless noted.
| id | fix |
|---|---|
| N1 | LP-only families route on `run_mode.lp_family_tag()` = the output tag without its plain "LP_licks" part (`subdir(lp_family=True)`): under LP alone attempt_coding / attempt_maps(_restnl) write the pre-rebuild folders again; every other mode gets its own. Readers resolve through the WRITER: fig_goal_summary, fig_goal_epochs, early_predictors, recovery_timing, precue_class_matched, decoder_retention_sessions, lick_enl_target_test (+ attempt_coding_pp via lp_family), kinematic_normality, cohort2_qc/quiet_task_events call `attempt_coding.out_dir()` / `attempt_maps.out_dir()`. deck_registry's relative paths are the DEFAULT run's and match again (the deck is built from the default run). Tests: default LP folder; no literal reader path; reader == writer under 4 modes |
| N2 | `base_id` compares the selection without its `-gateLP` suffix (`_daq_lookup_selection`): the gate-LP LP run finds the DAQ model frozen under `sel-daq-quittail`; a camera-gate run still does not. Test: the lp_final pair |
| N3 | writeguard's figures_working check exempts content-keyed stores: `WIDEFIELD_FROZEN_MODEL_DIR`, `frozen_models.local_dir()`, `session_cache.CACHE_DIR`, any `frozen_models*` segment. Test |
| N4 | `run_mode.routed_output(path)`: the output tag appended unless already a segment, folder made, `assert_writable` called. Used by the 13 H2 writers (also for an explicit `--output`), section_g_figures `--src`, enl figures, nightly_figs' `out` (so it passes `<fw>/<tag>` down) and joint_xsession `--output`. Test: writer == `grant_kit._fig_root` and no double tag |
| N5 | `routed_output` / `subdir(make=True)` create the tagged folder (default run: nothing new). Test |
| N6 | NOT DONE: waiting for Priya (whole-session exclusion of PS92_0908) |
| N7 | NOT DONE: waiting for Priya (jaw-only after the last lick vs no-response runs) |
| N8 | `restnl_mask._key` + `map_yaml_subset()` + `_helper_code()` (= the closures of the prm digest, `position_reference_maps._raw_maps_parts`). Test |
| N9 | `map_collect.collect` RAISES under the camera gate when any session job failed (all failures listed); the per-animal / per-session drops in grant_confusion 5b, attempt_maps, hemo_map_control, the time-local rest call `trial_classes.reraise_under_camera`. Under daq: unchanged (log and continue). Test |
| N10 | specificity_control and residual_roi read `frozen_encoder.out_root()/frozen_encoder_<tag>` and write `run_mode.subdir(<E>/specificity_control)` (+ assert_writable); recovery_timing / early_predictors read attempt_coding through `out_dir()`, iti outputs through `il.OUT`, specificity / attempt_coding_pp through `run_mode.subdir`. The other ~44 unrouted pre-merge writers repo-wide: NOT done (logged) |
| N11 | `il.XOUT` = `iti_licks/` + only the DATA parts (hemo, lnmf, projall) holds the per-session extracts and their zscore_check SD tables (`_load`, `labels`, `_sd_dir`, the extract step, iti_area_split_sd / iti_basis_decoder / iti_where_decoder / lick_rim_overlay); every output stays in the fully tagged `il.OUT`. A PRE_FROM / MAP_REFS / in-memory / LP / gate-LP run reads the extracts of its data. Test. (Other caches under OUT, e.g. the prediction caches `pred_file`, still follow OUT.) |
| H2 (partial) | routed writers now call `assert_writable` (via `routed_output`); nightly `--output` no longer bypasses routing (N4) |
| H4 (partial) | attempt_coding `trials_*` and attempt_maps `maps_*` skip-if-exists files are keyed by `run_mode.keyed_name(..., cache=True)` (csig under camera). Their downstream readers use the plain names, so under camera they fail loudly rather than read stale files (logged) |
| M4 (gap) | `_raw_maps_code` = `code_digest(*_raw_maps_parts())`; the sufficiency test now reads THAT list and also requires daq_io.{rising_edges, strobe_codes} and behavior_events.get_or_compute (added) (test_map_cache_key) |
| M6 (gap) | attempt_coding / attempt_maps noattempt / reaction split on `miss_working` ("late" stays on working) |
| M8 (label) | the 5rm column under the LP DAQ cross-check is printed as "LP-ROW fits, not the DAQ-trained model" |
| M10 | `lp_final_render.settings()` returns the env, `scoped()` applies it for a block; `setup()` (the render driver) still applies it; lp_final_compare uses `scoped()`. Test |
| L1 | pinned fingerprint value instead of a self-comparison; one-row-change -> new id tests for attempt_axes, engagement_states, epoch/position decoder (`frozen_decoder_models`) in addition to iti_readout. The `_read` stub of test_collect_5c and the other 11 kinds: NOT done (logged) |
| LOW | scripts/iti_nmatch_check.py deleted (archive copy kept; removed from ITI_SCRIPTS). ONE stopped: `camera_select("stopped")` returns no engaged rows (= camera_rows). Unknown trial classes RAISE in camera_select / camera_rows / _class_select. `_selection_cfg` / `stop_run_min`: a config that fails to load RAISES (only ImportError falls back). `engaged_frame_mask` under camera with no working period keeps no frame. Register regex also covers `reference_engagement` / `session_quit` callers (analysis_kit, behavior_clips = "measure"). NOT done (logged): hard-coded old encoder / basis run labels in iti_licks / iti_basis_decoder / iti_decoder_patterns; `WFIELD_ITI_OUT_W` override; rule-edge wording ("terminal by construction"); bm / prm coverage of `_args` / detect_licks / classify_cues / docked_periods, `session_signature` of regime / fmdir / frame map / gui_config, hemo_map_control's svt415 key; `.str[:4]` animal ids |

### 8.6 Priya's decisions, 11-03 (fix round 3; tests/test_priya_decisions_1103.py)
| # | decision | implementation |
|---|---|---|
| 1 | no-lick rows with camera class "contact": in NEITHER miss-while-working nor stopped; count them | `GATES["miss_working"]` excludes contact (already); `trial_classes.contact_nolick_counts` + `scripts/camera_coverage_tables.py` -> `camera_contact_nolick.csv` (per animal x epoch, with the cause split) and `camera_contact_nolick_trials.csv` |
| 1b (final window) | the no-lick analysis counts licks / contacts EXACTLY as cue_stop (cue -> trial stop, contacts from cue + 70 ms, the contact table) | `nolick_decoder.categorize` under camera: classes from `cue_classes_for(..., window="cue_stop")`; a DAQ-"undetected" cue whose cue_stop class is contact becomes `late_rewarded` (a contact trial, out of the no-lick arm), so the two definitions cannot disagree. `trial_classes.nolick_selection()` (`sel-camera-cue_stop-...`) enters the nolick reference spec and the session-features cache kind (camera only; default keys unchanged). The superseded "response window" variant was written and REMOVED before commit (no fork of the classifier) |
| 1c (check) | the count table is a check that the disagreement is 0; residuals with cause; trial-stop lengths | causes: late_after_response_window / cue_frame / refractory_merge / definition_mismatch (never a sensor miss: contact IS the DAQ sensor); `camera_stop_lengths.csv` (trials, stops > 3.6 s, their median, median of all). SCRATCH CHECK, 3 sessions (PS92_0908, PS93_0818, PS94_0821; read-only, outputs in scratch): the categorisation's response window = 3.5 s in all (per-session, A1); disagreements 0 / 1 / 0, the one (PS93_0818 acute) = late_after_response_window (now a contact trial); trials with stop > 3.6 s 289/321, 373/554, 376/600, median of those 3.76 / 3.94 / 3.72 s, median of all 3.72 / 3.77 / 3.66 s. The full table needs a run of the script (it writes into the share qc folder) |
| 2 | rest-baseline working period ends at the LAST CAMERA LICK | `camera_engaged_period`: cues up to the last camera_lick-class cue. SUPERSEDED in part: T8 (8.8) ends it at the last camera lick also with an uncovered tail, and the fourth review (8.10) starts it at the first camera lick |
| 3 | attempt "noattempt" = no_response inside the working period, not in a sustained stop | `classify_session`: a non-stopped no_response after the end anchor -> after_end, so "no_response" = an isolated lapse; attempt_coding / attempt_maps label it "noattempt" (contrast: isolated lapse vs sustained stop) |
| 4 | the 7 parallel-ordering scripts | curbd_pilot x2 / iti_decoder_pattern_stats sort the fan_out results; lick_code_subspace docstring no longer reads as a call; SFX resolved at import from argv (spawned workers re-import with the parent argv) in arrival_to_cue_traces / lick_cue_to_lick / revup_neural / stopped_trial_traces; lick_iti_prelick JOBS -> `_RUN["jobs"]`. test_parallel_results_are_ordered: 261 passed |
| 5 | face_offset_partial, lick_balance_blind_decoder, lick_nonlinear_decoder stored | face_offset_partial.fit_dec = `frozen_readouts.fit_all` inside its stored face_aim spec (fingerprinted); lick_balance_blind = `frozen_readouts.pre_loso_ridge`; lick_nonlinear = the whole linear + kernel-ridge reference through `load_or_fit` (kind lick_nonlinear, lp_only fingerprint; WIDEFIELD_ITI_FROZEN=0 = in memory) |
| 6 (N6) | KEEP PS92_0908 for its covered trials; session rules within the covered span; mark partial coverage | `classify_session(cls, matched)`: uncovered cues stay unknown; an uncovered TAIL disables the end cutoff; stop runs per covered segment (an uncovered gap never stitches runs); `session_coverage` + `camera_coverage.csv`; `lp_licks.stamp` adds `trial_classes.coverage_note()` (names partial sessions) to every camera-gate figure. Scratch check: PS92_0908 389 cues, 321 covered (82.5 %), uncovered tail |
| 7 (N7) | a jaw-only after the last camera lick does not break a no-response run; it counts toward the stop | SUPERSEDED by T3 (8.8, Priya): the jaw-only trial BRIDGES the run but is NOT counted and NOT stopped (after_end); a stop needs >= 6 no_response. The probe `[contact, NR x3, jaw, NR x3, jaw, NR x2]` = one bridged run, 8 stopped + 2 after_end |

**Full suite on a frozen copy of b32225f1 (6 shards, ~48 min): 4,441 passed, 176 skipped, 0 failed** (the 11 parallel-ordering cases now pass; fcae2ee2 before this round: 4,425 passed, 176 skipped, 11 failed = those cases).

### 8.7 Before any render (coordinator 11-03)
1. **Unrouted share writers.** A static scan of `scripts/*.py` (writes + a share / figures_working root + no routing)
   found 128. Classified (register: `tests/test_share_writers_mode_safe.py`, a new unclassified writer fails):
   - 75 REFUSE in any non-default mode (`run_mode.require_default` at the top of `main()`, or after the imports for the
     4 module-level scripts layer2_matched_n, lick_contact_by_epoch, spout_mid_maps, summary_figures_1021). Includes
     camera_coverage_tables, jaw_onsets, trial_attempts, s1_trial_class_snapshot: run those WITHOUT the render profile
     (their outputs do not depend on the hemo variant / projection; camera_coverage_tables takes `--out`).
   - 1 routed: kinematic_normality (`run_mode.subdir(..., lp_family=True)`).
   - 10 INDIRECT: every write goes through an already-routed root (il.OUT / attempt_coding.out_dir()): coupling_vigour_sm,
     iti_first_lick, lick_correction, lick_firstlick_extract, lick_vigor_perspout, restw_strict_check, session_pace_table,
     decoder_retention_sessions, precue_class_matched, recovery_timing.
   - 42 BEHAVIOUR: behaviour / pose / FaceRhythm / camera products with no imaging input (same file in every mode).
   Also: `fr_musall.E` (imported by ~36 encoder-family scripts) = `frozen_encoder.out_root()` (the old literal by default,
   imaging_encoding/<tag>/ in a tagged run), so they read / write the encoder runs of their own mode. Not scanned:
   scripts/ subfolders (rest_migration routed earlier, 509cd2bd; cohort2_qc is cohort-2 QC; archive).
2. **ITI scripts.** `iti_licks.encoder_run(suffix)` = `frozen_encoder.out_root()/frozen_encoder_gate-attempted_v7g3_t3d_frA
   <suffix>` replaces the hard-coded run labels (iti_licks x2, iti_basis_decoder / iti_decoder_patterns `_te_basis`);
   `WFIELD_ITI_OUT_W` is tagged (`run_mode.subdir(<override>)`).
3. **Cache-key gaps.** CLOSED: bm / prm digests (+ the RESTnl key, which reuses the prm closures) now hash
   locanmf_frozen_decoder._args, lick_detection.detect_licks, behavior_position.classify_cues_with_backup,
   docked_periods.docked_mask_any (and daq_io / behavior_events, second pass). LISTED, cannot affect a rebuild render:
   `session_signature` not statting sessions.yaml regime / fmdir, the cleanpairs frame map and gui_config.json, and
   hemo_map_control's `svt415:rank{K}` key ignoring the session_overrides functional_channel -- every rebuild cache entry
   is NEW (its kind carries the hv- / lnmfproj / selection / csig tags, so no pre-rebuild entry can be served) and none of
   these inputs is edited during the rebuild; a later edit of one of them requires clearing the rebuild cache (the
   profile's separate WIDEFIELD_SESSION_CACHE makes that one folder).
4. **Attempt readers.** `attempt_coding.trials_name(align, animal)` is the writer's mode-keyed name; the 10 readers
   (quiet_task_events, coupling_by_class, decoder_retention_sessions, early_predictors, exec_vs_neural_match,
   lick_enl_target_test x3, precue_class_matched, fig_goal_summary, softmax_neural_vs_lick, trial_neural_vs_exec) use it,
   so the rebuild reads the new files. attempt_maps `maps_*` are read only inside attempt_maps (same keyed name).
5. **Render profile** `configs/render_profiles/cohort1_rebuild_20261103.yaml` + `wfield_local/render_profile.py`
   (`--shell powershell|bash` prints the env, `--check` prints the run tag): WIDEFIELD_HEMO_VARIANT =
   meegkit_hpfit_lag_refitT_swapmask (Priya: no new drift step), WIDEFIELD_LOCANMF_PROJECT=1 (Priya: projection, no
   refit), gate camera / cue_stop / end lick / stopped run (stopRun6), WIDEFIELD_LP_FROZEN=lp (LP-trained), a fresh
   frozen store `C:/Users/SabatiniLab/frozen_models_rebuild_20261103`, a separate session cache
   `C:/Users/SabatiniLab/.widefield_session_cache_rebuild_20261103`; Aug-only pre as per-block `--pre-from 0801`
   (epoch_grant_figures, grant_figures), never global; encoder drift cutoff recorded at 600 s.
   tests/test_render_profile.py: run tag = `hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_sel-camera-cue_stop-endlick-
   stopRun6` (cache tag + `_csig-<sha>`).

6. **Coverage tables, all curated sessions** (`scripts/camera_coverage_tables.py` at 5c17d76a, run WITHOUT the render
   profile into a scratch folder; nothing on the share). Partly covered: PS93_0813 (0/553, no video), PS92_0904
   (377/378), PS92_0908 (321/389, tail uncovered). Every contact / no-lick disagreement is `late_after_response_window`
   (the 3.5 s window), so with the no-lick window unified on cue -> trial stop the residual is 0. Disagreements per
   animal (pre / acute / subacute / chronic): PS92 1/1/1/1, PS93 6/3/3/2, PS94 1/2/0/1, PS95 1/1/0/0.
   Trial stops > 3.6 s (n trials; n > 3.6 s; median of those ms; median of all ms):

   | animal | pre | acute | subacute | chronic |
   |---|---|---|---|---|
   | PS92 | 4044; 3465; 3803.0; 3762.5 | 2681; 1475; 3802.0; 3651.0 | 593; 471; 3755.0; 3688.6 | 3894; 3524; 3771.1; 3736.3 |
   | PS93 | 5396; 4713; 3827.6; 3792.1 | 2294; 1258; 3912.9; 3657.8 | 1616; 1193; 3820.4; 3730.4 | 6068; 5542; 3791.6; 3778.4 |
   | PS94 | 6197; 5711; 3707.8; 3697.6 | 3643; 1924; 3719.7; 3634.8 | 3000; 2486; 3698.7; 3680.8 | 4800; 4469; 3709.0; 3696.6 |
   | PS95 | 6950; 6259; 3715.6; 3690.6 | 720; 493; 3739.8; 3694.6 | 4275; 2941; 3701.4; 3667.6 | 6600; 5848; 3681.4; 3675.6 |

   Acute has a clearly smaller share of trials longer than 3.6 s (53-68 % vs 86-92 % pre): relevant for any
   licks-per-trial read-out (FOR PRIYA).

### 8.8 Review, third pass (REBUILD_CODE_REVIEW_20261103.md "Third pass", 62d1dbc4): 3ef70b5b
- **T1** LP-family roots (`run_mode.subdir(lp_family=True)` / `routed_output(lp_family=True)`) and registered data stores
  (`run_mode.register_data_store`) pass the writeguard at their routed place; `lp_family_root_ok` checks every root
  (was: first only). attempt_coding `--out` routed like `out_dir`. Probe: writable in every LP mode.
- **T2** ITI: `E_BASE` (= `frozen_encoder.base_root()`, unrouted) for iti_licks / residual_roi / specificity_control /
  remap_lmm, so the specificity_control and ITI folders are tagged once; `il.XOUT` extract folders (SESS_NAME,
  zscore_check) registered data stores; SD tables through `il.sd_table_path` (iti_zscore_check, iti_area_split_sd).
- **T3 (Priya)** a jaw-only trial after the last lick BRIDGES a no-response run (not counted, not stopped); a stop needs
  >= 6 no_response. Effect (below).
- **T4** missing RESTnl baselines are built (`restnl_mask.baseline_or_build`); attempt_maps raises on empty maps;
  position_reference_maps raises when no RESTnl baseline exists.
- **T5** nightly_figs output and deck_out routed; locanmf_analysis_deck `assert_writable` before save.
- **T6** kinematic_normality reads `il.OUT`; remap_lmm writes into the specificity_control writer's folder.
- **T7** only >= 5 consecutive unmatched cues count as uncovered (`MIN_UNCOVERED_RUN`; `_effective_coverage`).
- **T8** the rest period ends at the last camera lick, always (`camera_engaged_period`).
- **T9** s1_trial_class_snapshot built from `classify_session` (working = primary, end lick); face_offset_partial
  "before" fits stored (`pre_loso_ridge`; decoder via `frozen_readouts.fit_all`).
- **T10** `tests/test_parallel_ordering_strict.py` (AST: fan_out result sorted; runtime state visible to workers);
  5 wfield_local sites sorted (joint_xsession, orofacial_outputs x2, poststroke_section_g, precue_lickfree).
- **LOW** warm-up rule; a session with no camera lick -> all no_response are warm-up (no stops); rule-edge tests
  (tests/test_priya_decisions_1103.py); stale docstrings; poststroke no-lick window cue_stop; cache sort key with kind.
- **DECISIONS.md**: "2026-11-03 (night) -- Camera trial gate for the cohort-1 rebuild" collects the decided rules.

**Classifier effect, 5c17d76a vs 3ef70b5b** (the 119 curated sessions with trial tables, read-only, scratch CSV;
CORRECTED 11-03 late after the fourth-review addendum): 513 trials in 44 sessions change class. Cohort: stopped
2339 -> 1830, after_end 72 -> 581, no_response 2674 -> 2670, warm-up 0 -> 4. The 509 stopped -> after_end trials are
almost all the BRIDGING JAW-ONLY TRIALS THEMSELVES leaving the stopped arm (the old rule labelled a bridging jaw-only
trial stopped; T3 makes it after_end): the review counts 485 jaw_only + 24 no_response on its 139-session set. In
PS94_0825 and PS93_0925 every moved trial is jaw-only and every stopped no_response trial is unchanged. (The CSV's
`old_jaw_only` / `new_jaw_only` columns count only trials still LABELLED jaw_only, i.e. inside working, which is why
they are equal.) Stopped and after_end are both outside working, so the working / camera-lick sets do not change.
- PS94_0825 (subacute, 600 cues): stopped 40 -> 12, after_end 0 -> 28 (28 bridging jaw-only trials); contact 440,
  incomplete 27, working jaw_only 55, no_response 38 unchanged.
- PS93_0925 (chronic, 600 cues): stopped 86 -> 38, after_end 0 -> 48 (48 bridging jaw-only trials); contact 477,
  incomplete 20, working jaw_only 13, no_response 4 unchanged.
- Stopped per animal x epoch, old -> new (pre / acute / subacute / chronic): PS92 0/268/0/0 -> 0/217/0/0;
  PS93 41/232/77/86 -> 24/208/52/38; PS94 225/386/156/94 -> 157/328/116/67; PS95 187/0/355/232 -> 135/0/341/147.
- **Warm-up (PROVISIONAL, for Priya)**: 4 trials in 4 of 119 curated sessions (the review's 5 on 139 sessions adds
  PS94_0805, which is outside the curated set: `date_policy.cross_session_exclude` has 0805) -- PS94 acute 2 (PS94_0817, PS94_0820), PS93
  subacute 1 (PS93_0824), PS95 subacute 1 (PS95_0818). Labelled neither class.

**Tests.** 50f59ff8 full suite (5 shards): 4,450 passed, 176 skipped, 0 failed. 3ef70b5b: see 8.9.

**Not done (LOW, listed):** o2_face_rhythm segment completion order; grant_figures failure order; the N2 symmetric
strip; the N3 `frozen_models*` file-segment exemption; T6 early_predictors mixes families (consistent within a family);
camera_coverage_tables forces gate=daq for its own comparison; `.str[:4]` animal ids in a few scripts.

**For Priya:** warm-up rule (provisional); the no-camera-lick session rule (all warm-up, no stops); MIN_UNCOVERED_RUN = 5;
the contact no-lick counts above; the acute trial-length difference.

### 8.9 Final test record (3ef70b5b; ffdf94ab after it is docs only)
Full suite, 5 shards, frozen worktree copy: 4,476 passed, 176 skipped, 0 failed.

### 8.10 Fourth review (a1b6064a + addendum 0fc49ee8): camera-class items 831f8ff3, rebuild root c812a3c0
**Camera classes (831f8ff3).**
- The rest working period now runs from the session's FIRST camera lick to its LAST (`camera_engaged_period` via
  `_lick_span`). Warm-up trials are outside it. Under the robustness anchors (contact / close) it ends at that anchor,
  because licks after the anchor are after_end in the classified train.
- A stop run ends where licking resumes: a contact or incomplete ends it, and a jaw-only trial before the end anchor
  ends it too. An in-video unknown is skipped. So `[C, NR x3, U, NR x3, C]` is one stop of 6 and
  `[C, NR x3, C, NR x3, C]` is no stop. Both are tested.
- The stale docstrings / comments at trial_classes :449, :455-457 and :517-518 are rewritten, and so is the module
  comment "stopped = no_response".
- `apply_stopped_rule` / `apply_session_end` (the old rule) are REMOVED. Their tests now call `classify_session`.
- New tests:
  - MIN_UNCOVERED_RUN exactly 5, with 4 vs 5 unmatched cues at the head, middle and tail (2-4 fail).
  - NR3 J NR2 (no stop) and NR3 J NR3 (stop).
  - No bridging when the video tail is uncovered.
  - A real T8 test through `cue_classes_for` with an uncovered tail.
  - The robustness anchor ends the rest period.
  - Rule "any" excludes warm-up.
  - test_7 renamed to `..._bridges_but_is_not_counted`.
- The DECISIONS entry gains:
  - no-video sessions (PS93_0813);
  - late_rewarded;
  - "any" excludes warm-up;
  - no jaw-only bridging with a missing tail;
  - robustness anchors and the rest period;
  - a stop ends where licking resumes;
  - the rest period starts at the first camera lick.
- Log 8.8 is corrected: the moved trials are the bridging jaw-only trials. Rows 2 and 7 of 8.6 are marked superseded.
- Warm-up count: PS94_0805 is outside the curated set (`cross_session_exclude` has 0805). That accounts for the
  review's 5 vs our 4.

**A. Rebuild root (c812a3c0).** Implemented in `configs/paths.yaml` `rebuild:` and `wfield_local/paths.py`, with the
fence in `wfield_local/rebuild_root.py`. The profile key `rebuild_root: M:/MICROSCOPE/Priya/Widefield/rebuild_20261103`
sets `WIDEFIELD_REBUILD_ROOT`, and then:

| what | where under the profile |
|---|---|
| figures_working | `<R>/figures_working` |
| cue_analysis_out (analysis_figures) | `<R>/analysis_figures` |
| cohort_out: grant_figures, frozen_models (server), joint_bases (publish), the deck, epoch_boundaries.json, chronic_stability, ... | `<R>/labcams` |
| `<LP_analysis>/<run>/imaging_encoding`: encoder runs, iti_licks, attempt_coding / attempt_maps, qc tables, kinematic_normality, ... | `<R>/LP_analysis/<run>/imaging_encoding` (`PathResolver.derived` / `orofacial_outputs.encoding_root`) |
| frozen store / session cache / RESTnl caches (beside the cache) / map npz (in the cache) | `<R>/frozen_models`, `<R>/session_cache`, `<R>/session_cache_restnl_<tag>` |
| rebuild_figs default --out | `<R>/rebuild_figs` |
| INPUTS, read in place, never written: labcams session folders (SVTcorr, hemo_*, U_atlas, LocaNMF footprints), the joint bases, DAQ, behaviour logs and behavior_summary, cameras, dlc / session_poses, the v36 pose tables (`<run>/sessions`, `<run>/cohort`) | unchanged |
| inputs inside derived trees: epoch_boundaries.json; imaging_encoding/qc jaw_onsets / nose_onsets / trial_attempts / camera_coverage | COPIED by `python -m wfield_local.rebuild_root init --profile cohort1_rebuild_20261103` (sha256, `REBUILD_ROOT.json`; a different existing copy refuses) |

- **Fence.** A `sys.addaudithook` hook, installed by `import wfield_local` whenever the env var is set (subprocesses
  inherit it). It sees every Python `open` / `os.*` / `shutil` call:
  - A WRITE on a data tree outside R is refused. Data trees are any MICROSCOPE / standby path, this box's
    figures_working, and the default `.widefield_session_cache*` / `frozen_models*` / `joint_bases` beside it.
  - Exceptions: temp / scratch, and the hemo-variant build step (`WIDEFIELD_REBUILD_VARIANT_BUILD=1`, which may write
    `hemo_*` dirs only).
  - A READ of a default derived product is refused: the old figures_working / analysis_figures, any
    `<run>/imaging_encoding`, every labcams top-level entry that is not a dated session folder, joint_bases or xday,
    and the default caches / stores.
  - `writeguard.assert_writable` applies the same write rule.
  - Not seen: C-level writers (h5py, cv2) and `Path.exists` (an exists check on a default path is True, but the open
    that follows is refused).
- **Rewrite.** About 75 imaging_encoding constructions now go through `oo.encoding_root` / `lp_licks._enc_dir`
  (literal `M:/...` strings, `oo.root(..) / RUN / "imaging_encoding"`, `_run_dir() / "imaging_encoding"`). They are
  byte-identical in a default run on this box. The joint bases use `input_root`.
- **Behaviour-only writers** (behaviour events, spout_behavior, DLC frames, ...) are not rebuild outputs. Under the
  profile their writes to the share are refused. Run them without the profile.

**B.**
- **R1.** rest_frozen_decoder_nl:
  - OUT_ROOT goes through `run_mode.subdir`.
  - The feature cache lives under the run's figures_working.
  - `_code_key` adds `run_mode.cache_tag()`. The default key is unchanged.
  - PAGE_SESSIONS is read from `rest_frozen_decoder._epoch_dir()`.
- **R2.** The deck's section K resolves through the run's cohort_out and `derived()`. A tagged build outside a
  rebuild root with no grant_dir RAISES. Under the profile, src and grant_dir resolve into R. Remaining:
  - the section K legends carry pre-rebuild numbers;
  - its attempt-map entries point at the default attempt_maps names, so a rebuild deck skips them (optional slides).
- **R3.** rebuild_figs:
  - the epoch sidecars come from `_common.epoch_data_dir()` (lp_licks.output_dir);
  - goal_summary fig1 comes from the combined-encoding writer's folder;
  - the unrouted inputs (the enl_move `.bak` CSV, the PS94 state cache) and the slowband CSV outside a rebuild root
    must be given explicitly in a non-default run.
- **R4.** attempt_maps: maps / counts / hemispheres / figures all use `product_name` (the cache key). `--figures`
  raises on a missing keyed maps file. fig_goal_summary reads `hemispheres_path`.
- **R5.** `require_default` uses `lp_licks.output_tag()`, so LP, gate-LP and ENL runs refuse. `render_profile.apply`
  CLEARS every leftover WIDEFIELD_* / WFIELD_* (`cleared` lists them). The shell lines start with a clearing line.
  `--check` prints the run tag and the full output tag and exits 1 on a mismatch.
- **O2.** The guard moved out of the RUNNER text into `o2_face_rhythm.main`. frozen_running_compare refuses.

**C.** `tests/test_rebuild_root.py` covers:
- A subprocess probe under the profile, with R in scratch: old grant / figures_working / imaging_encoding writes
  raise in `open()` and in writeguard, old derived reads raise, and a derived read resolves inside R.
- Rule units, init, the package-import hook, R1-R5, and O2.
- The writer register for `scripts/` subfolders + `wfield_local/`: 32 RESOLVED (land in R) and 45 LITERAL (fenced:
  refused under the profile, mostly cohort2_qc / one-offs). A new writer fails until classified.
- test_share_writers_mode_safe's SHARE pattern now includes the new helpers.

- **Unseeded root fails loudly.** Two inputs are silently optional for their reader: the jaw calls (a missing
  jaw_onsets.csv would drop every jaw_only class) and the coverage table. If either is missing INSIDE the rebuild
  root, it raises `rebuild_root.require_seed` and asks for `init`.

**Not run:** `rebuild_root init` writes to M:, so it was not run. Run it once before the render.
