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

### 8.11 Priya 11-03 (evening): drift arm, licks per trial, 5-min stop rule (bc57f786 + this commit)
**Drift cutoff (2d103d2c cherry-pick of 25e294d2, bc57f786).**
- 600 s stays primary.
- `COMBINED_ENCODING_DRIFT_MIN_PERIOD_S=1500` is a SENSITIVITY arm. It adds `run_mode.drift_part()` (`drift1500s`) to
  the output tag AND the cache keys, and the ITI extracts (`il.XOUT`) are keyed by it too.
- `drift_part` refuses when combined_encoding was imported with another value.
- The profile gained `arms: drift1500` (encoder spout contra - ipsi cells, ITI pre-cue level readouts): `--arm drift1500`
  / `apply(arm=)`, expected tag `hv-..._lnmfproj_drift1500s_sel-...`. COMBINED_ENCODING_* leftovers are cleared.

**Licks per trial = TOTAL licks from cue + 70 ms to the TRIAL STOP; lick rate from ILIs.**
- `daq_trials.decode` now reads the analog `trial_end` TTL.
- `daq_trials.trial_stops` returns that TTL. Without one it falls back to the task rule: first contact (from
  cue + 70 ms) + the response window, or cue + window for a miss. Capped at the next cue.
- `daq_trials.licks_to_stop` counts the licks.
- The DAQ trial table gains `stop_s`, `stop_source` and `n_licks_trial`. `n_licks_post` stays the response-window count
  for hit / latency scoring.
- Readouts moved to this window:
  - `spout_behavior.lick_microstructure` (licks_per_trial; its lick_rate_hz = 1 / median ILI over the same licks,
    relabelled "(1 / median ILI)");
  - `engagement_decomposition.session_trials` (`n_licks_trial`, `stop_sample`);
  - `quit_prodrome` licks per trial (was cue -> next cue, which counted ITI licks) and its per-trial ILI;
  - `lick_bout_structure` per-trial licks / bouts / ILIs (was cue -> next cue).
- No count / duration per-trial rate exists in these readouts. NOT changed, listed for Priya:
  - the session-level licks-per-minute curves (quit dynamics, `early_lpm`);
  - `enl_lick_rates` (pre-cue ENL);
  - FaceRhythm validation's fixed response window;
  - `face_rhythm_lick_epochs`.
  Also: the epoch CHRONIC rule reads licks/trial from the cohort behaviour table. The rebuild SEEDS the existing
  epoch_boundaries.json, so its boundaries do not move unless the epochs are re-derived after the behaviour tables are
  rebuilt.

**Trial stop check** (DAQ trial_end vs first DAQ contact from cue + 70 ms; 6 sessions PS92_0821, PS93_0908, PS94_0818,
PS95_0917, PS93_0818, PS92_0908; read-only):

| | n | median | 5-95 % | min | max |
|---|---|---|---|---|---|
| hits: stop - first contact | 2,405 | 3.528 s | [3.390, 3.718] s | 0.032 s | 3.845 s |
| misses: stop - cue | 700 | 3.525 s | [3.525, 3.684] s | 3.523 s | 3.748 s |

- Of the hits, 69.3 % fall in [3.50, 3.60] s, 23.2 % in (3.60, 3.75] s, and 0.6 % are above 3.75 s.
- 49 trials are off the rule by more than 250 ms, all hits:
  - SHORT stops, 0.03-3.2 s after the first contact (e.g. PS92_0821 trials 69, 99, 100, 145, 168; PS93_0908 76, 82,
    100, 173; PS93_0818 66, 71, 202, 363; PS92_0908 30, 72, 108, 113, 122). The task's timer seemingly started on an
    earlier lick than the first DAQ contact after cue + 70 ms (a lick inside the first 70 ms, or one the DAQ
    discriminator did not register).
  - A few at 3.75-3.85 s.
- So the rule holds at the median, the measured TTL is used wherever it exists, and the rule is only a fallback.

**5-min stop rule (requested FIRST; not implemented).** Current classifier, 119 curated sessions: 159 stopped runs,
1,830 stopped trials. **0 runs (0 trials) start within 5 min of the session's first cue.** The earliest starts are
PS95_0820 at 46.0 min (6 trials), PS95_0820 at 54.2, PS92_0818 at 55.2, PS95_0820 at 57.7 and PS94_0819 at 58.9. So
the rule would move no current stopped time. Per the coordinator's order it waits for the go-ahead after this report.
Implementing it means replacing the warm-up rule, which today covers 4 no_response trials before the first lick. With
the time rule those 4 become isolated lapses (no_response), unless they sit in a >= 6 run, which none do.

### 8.12 Test record (f58c5d86)
Full suite, 6 shards, frozen worktree copy: 4,517 passed, 176 skipped, 0 failed. The b422ae83 run in between had 11
failures, all `test_combined_encoding_options::test_out_dir_names`; they were fixed in bc57f786, where encoding_root
outside a rebuild root is `oo.root(rv) / run / imaging_encoding`.

### 8.13 The 5-min stop rule (Priya 11-03, GO after the 0 / 159 check)
- `classify_session(..., cue_s=)`: a qualifying run (>= 6 no_response) is a STOP only if it starts >= 5 min after the
  session's first cue (`trial_selection.stop_min_start_s: 300`). An earlier run is `early_run`, in neither class.
  Under rule "any", every no_response before 5 min is `early_run`.
- `cue_classes_for` passes the cue train and keys its memo on the setting. The selection tag stays `stopRun6` at 300 s;
  another value adds `t<s>`.
- The warm-up class is REMOVED. The 4 former warm-up trials are isolated lapses (no_response).
- The rest working period keeps its start at the first camera lick as the named setting `trial_selection.rest_start:
  first_lick` (`session` = the first cue; OPEN for Priya).
- DECISIONS entry rewritten for the time rule, the rest start setting and rule "any".
- Tests: time rule (early run, a straddling run is decided by its start, isolated lapse before the first lick, length
  check), rule "any" with the time rule, the rest_start setting.

### 8.14 Merge of origin/main (e8491dc8) into the rebuild: merge commit, done in a scratch worktree, then fast-forwarded
Main brings:
- A3 (paired animal draws, `epoch_change_lmm` / `lmm_stars`);
- bal_em: `spout_scaling.FrozenFrame` and the label-free EM centre on every cross-session spout decoder; the state
  discriminators stay on session_z;
- A1 per-session response window, and the A2 / A4 amendment;
- e9a7dc31 DECISIONS (spout hold 3.5 s, lick counts to the trial stop).

Rule for every resolution: keep the rebuild's camera gate / selection tag / rebuild-root routing / FREEZE_VERSION 2
store AND main's bal_em frames and A3.

**Shared mechanism: `FrozenFrame(..., store=)`** (`spout_scaling`). The hook `store(fit_all)` returns {"full",
"loso": {session id: model}}: from the frozen store, fitted by `fit_all` on the FrozenFrame TRAINING frame on a miss.
So the rebuild's stored, fingerprinted models are the bal_em models, and each spec carries `spout_scaling.spec_kw`
(the scaling) on top of the train_data fingerprint.

**Per-file resolutions** (rebuild side = HEAD, main side = origin/main):

| file | rebuild side | main side | kept |
|---|---|---|---|
| DECISIONS.md | camera-gate entry | 7 new 11-03 entries | both, in order |
| locanmf_frozen_decoder.py | `decoder_spec(YE, GE)` + train_data | `scaling` + `spec_kw` | `decoder_spec(XE, kept, pre_i, *, YE, GE, ..., scaling)` with train_data AND spec_kw; main's XT training frame + pi_train; main's lint fixes |
| cohort2_qc/quiet_task_events_followup.py | YE / GE kw | scaling comment | both |
| grant_confusion.py + grant_kit.py | 5b / 5c through `_stored_epoch_decoder` | `grant_kit._frozen_frame` (bal_em frame, memoised on the bundle) for 5b / 5c / 5r / 5rm | main's frame everywhere; the store moved INSIDE it |
| nolick_decoder.py | `_stored_nolick_reference` | FrozenFrame | FrozenFrame with `store=` `_stored_nolick_reference(fit=, scaling=)` (spec + scaling kw); LOSO from `ff.loso` in both scalings; `res["frozen_model"]` provenance kept |
| precue_engagement_states.py | `_stored` position decoder and discriminator | FrozenFrame; state frames XEt / XUt | see below |
| position_coding_directions.py | `models=` stored confusion decoder | FrozenFrame (needs GU) | see below |
| poststroke_compare.py | `_g_store` section G references | `frozen_X`, FrozenFrame lick-only, `_lick_scored` state frames, balanced_fit -> (clf, y) | see below |
| scripts/june_august_drift.py | stored month model | FrozenFrame + matched EM frame | see below |

Details for the rows marked "see below":
- **grant_confusion.py + grant_kit.py.** `_frozen_frame(bd, an, align)` asks `_stored_epoch_decoder(..., XT=, scaling=)`.
  That spec gains the scaling and fits on XT; with WIDEFIELD_EPOCH_FROZEN=0 the same fits run in memory. The memo key
  carries the scaling, store/opt-out, animal and window. The bundle records "animal" / "align".
  - 5rm `matched_frozen` keeps its store; its spec gains the scaling, and main's rec/_stack gains `pool_tag`.
  - `_class_select(..., cls=bd["cls"])` (the camera gate) is kept in every 5c site.
- **precue_engagement_states.py.** Position decoder: FrozenFrame with `store=_stored(... "posdec-ff" ...)` (new key,
  since the payload is now {"full","loso"}). Discriminator: the stored {"full","loso"} payload (trained on main's XEt / XUt),
  scored with main's `state_scored`; `_stored` passes the scaling kw. States 5 / 6 keep the rebuild's camera-gate masks
  (`nl_eng` / `nl_not`), not main's `not_eng_nl`.
- **position_coding_directions.py.** `_class_confusions(..., GU=, store=)`; FrozenFrame over `_confusion_pipe`
  (main's local pipe is the same estimator) with the store hook; `stored_confusion_decoder(fit=, scaling=)`. The pre
  LOSO comes from `ff.loso` in both scalings.
- **poststroke_compare.py.** `_g_store` and `frozen_X` both kept. The lick-only arm is main's FrozenFrame with
  `store=_g_store("g_lickonly_decoder", "<keep>-ff")`. `g_lick_disc` / `g_lick_disc_folds` stay stored (their specs
  carry `state_disc_scaling`; the folds key is `-v2`, since the payload is now (clf, y)), scored with main's
  `_lick_scored`.
- **scripts/june_august_drift.py.** Under bal_em the FrozenFrame model, in memory. The stored month model is the
  session_z fit and serves only under session_z. LISTED: drift_payload is to be stored in the bal_em frame.

**Tests adapted:**
- test_frozen_default's source tests now check the merged structure (store inside the frame / hook).
- test_frozen_default's coding equality test uses the store hook.
- test_spout_bal_em's two pinned-id tests became contract checks: FREEZE_VERSION 2 ids differ from main's literals,
  as already done for test_state_bal_em.

**bal_em and the camera gate.** The EM centre is NOT computed on the gated scored subset. For every scored session it
comes from ALL of that session's engaged (DAQ lick-arm, `pool_sessions` XE) rows under its scoring model; it is
label-free and never uses no-lick rows. It is then applied to whichever rows the gate scores (camera working / stopped
via `_class_select`). That is main's design ("the label-free EM centre of ALL its engaged rows, whatever the arm
scores"). Under the camera gate the centre therefore includes DAQ-engaged rows that the camera calls after_end or
unknown. FOR PRIYA / the coordinator: whether the centre should use only camera-working rows.

**in_working ON HOLD (main docs/status/SPOUT_BALEM_PAGE_EDITS_2026-11-03.md section 6).** Answered by option (b):
Aug-only pre, June gone, PS93_0813 in the lick class only. The rebuild already selects post-stroke rows by camera
class in `_class_select`, and the pre panel / training set stay DAQ-engaged licking trials of the Aug pre sessions, so
nothing extra is wired.

**A5.** It lands with this branch (trial_classes `cue_classes_for` is the single class source). There is no
`responded()` on main.

**Post-merge fixes** (the suite on 698c4e57 had 4 failures):
- 5c "refit" and "paired_matched" asked the store for the epoch decoder after the merge. `_frozen_frame(..., store=)`
  now keeps those two modes in memory, as before the merge.
- Main's new `scripts/lmm_curve_summaries.py` (pose / kinematics LMM, no imaging) is classified BEHAVIOUR in the
  writer register.
- `lmm_curve_summaries` and `kinematics_pooled_figs.fig_phase` wrote figure sidecars in the flat directory (main
  commits 8e2b82e6 / ceb8d2b4; `test_no_writer_bypasses_figure_layout`). Both now use `figure_layout.sidecar_for`,
  and the readers are the same functions.

### 8.15 Test record after the main merge (c6f9be2d)
Full suite, 6 shards, the merge worktree: 4,550 passed, 178 skipped, 0 failed. This includes the pinned
test_analysis_kit exact-equality tests and main's new test files.

### 8.16 bal_em centre rows = the rows the gate scores (Priya 11-03, camera gate everywhere)
- `spout_scaling.centre_rows()`: setting `bal_em.centre_rows` (auto | working | engaged), env WIDEFIELD_BALEM_CENTRE.
  "auto" means working under the camera gate and engaged under the DAQ gate. "working" under the DAQ gate raises.
- `centre_mask(cls_e)` is the camera WORKING classes (contact / incomplete / jaw_only). The classified train already
  turns after_end, stopped and early_run into other classes, and unknown is not working. It FAILS CLOSED: under
  "working" a site without camera classes raises (in `FrozenFrame` and `scored_frame`).
- A session with fewer than 2 working rows (no video) falls back to all its engaged rows, and the log says so.
- Sites wired:
  - `grant_kit._frozen_frame` / `_frozen_score_rows` (bundle `cls`);
  - `locanmf_frozen_decoder.frozen_scored_frame(cls_e=)` with the new `engaged_row_classes` (from feature indices or
    `trial_features_cached(with_indices=True)`), called by `pooled_frozen_loso`, `attempt_coding` and section G
    `frozen_X`;
  - `FrozenFrame(cls_e=)` in nolick, position_coding_directions, precue_engagement_states, the section G lick-only arm
    and june_august_drift.
- nolick carries a per-row `cam_working` flag through `session_features`. Its cache kind gains `cam_working` under
  camera, and a stale entry raises.
- NOT wired: `scripts/cohort2_qc/quiet_task_events_followup.py` calls `score_centre` directly; it is a cohort-2
  script, outside the cohort-1 render.
- **Identity:** `scaling_params.centre_set` = the centre rows, so the two centres have different spec ids.
- **Tags:** the profile's run tag is UNCHANGED (default auto = working). Camera gate + engaged adds `emc-engaged` to the
  output and cache tags.
- State discriminators (state_bal_em, default off) are unchanged.
- DECISIONS entry added. Tests: tests/test_balem_centre_rows.py.

### 8.17 june_august_drift: the month model stored in the bal_em frame
- `drift_payload(..., scaling=)` fits on `spout_scaling.train_frame` under bal_em; its spec carries `spec_kw`.
- `decoder_rows` gets the model through the `FrozenFrame` store hook (`_stored_month_model(..., scaling=)`), so under
  bal_em it no longer fits in memory. WIDEFIELD_DRIFT_FROZEN=0 is the in-memory opt-out.
- The size-matched draws of that script are in memory, as before the merge.
- Tests: test_drift_month_model_is_stored_in_the_bal_em_frame; test_frozen_default's drift source test updated.

### 8.18 Test record (4a8db872 + the test fix after it)
Full suite at 4a8db872, 6 shards: 4,556 passed, 178 skipped, 2 failed. Both failures were source-text tests in
test_frozen_default that matched the old call text (`store=_store)` / `store=_store_conf)`, now followed by `cls_e=`).
Those assertions were updated in the next commit, and both tests pass.

### 8.19 Merge-review fixes (B1, N1, N2, N4, N8; N3 not done)
- **B1 (blocker): label-keyed LOSO in every FrozenFrame store.**
  - `FrozenFrame(store=, labels=)`: `labels` (pooled id -> session label) is REQUIRED with a store.
  - The hook maps LOSO keys id -> label before `load_or_fit` stores and label -> id after loading. A stored label
    absent from the pool is ignored.
  - Sites: nolick_reference, coding_decoder, engagement_states posdec-ff, g_lickonly_decoder -ff and the drift month
    hook; the precue `-disc` payload's LOSO is keyed `kept[si]`.
  - The shift this fixes: the LP pool without PS93_0813 scored each later PS93 pre session with the wrong LOSO model.
    `frozen_decoder_models`, `_stored_epoch_decoder` and `drift_payload` were already label-keyed.
  - The dead `poststroke_compare._fit_lickonly` is removed.
  - Test: `tests/test_merge_review_b1.py` (a pool missing a middle pre session; fails on the old code).
- **N1:** the nolick_reference train_data now also carries the pooled label set.
- **N2:** `_collect_5c`'s outer except and `run_animal`'s confusion except call `trial_classes.reraise_under_camera`.
- **N4:** june_august_drift's size-matched draws are stored like 5rm `matched_frozen`: one `drift_matched` spec per
  animal (bundle fingerprint + scaling), keyed by (group, session, n_target, seed); a hit replays, a miss stores. The
  comment now says so.
- **N8:** the `_pooled_bundle` cache key adds the gate mode and selection tag; the `_frozen_frame` memo adds
  `centre_rows`.
- **N3 not done:** in "refit" / "paired_matched" the frame's models are still fitted. `rec` calls `model.predict`
  before branching on the mode, the pre loop passes `fr["loso"][i]`, and a matched model missing a spout falls back
  to XEs, so a fit=False frame is not a small change.

### 8.20 Fifth-review non-blockers (N5, a-i)
- **N5:** `run_mode.require_input_tag(path, what)` refuses an input folder whose run tag (a tag-like path segment)
  is not this run's, and prints the tag it read. Applied to a3_lmm_imaging_stars, a3_lmm_preview_fig, lmm_change_fig
  and lmm_per_lick_fig:
  - the tag goes into their output CSVs (`input_run_tag`) or a figure footer;
  - they are registered in test_share_writers_mode_safe (ARGPATH_READERS);
  - a3_lmm_imaging_stars with no fittable cell now prints a clear message and writes an empty CSV (was KeyError 'p').
- **(a)** spout_behavior's licks per trial uses the session's own response window (`daq_trials.response_window_s(..,
  3.5)`). The GUI-fallback sessions used the yaml 2.0 s. Behavioural test (fails on the old code): a lick 3.0 s after
  the first contact counts.
- **(b)** rest_start enters the selection tag (`-restsession` when "session"; the default first_lick adds nothing),
  and with it run_mode tags, cache keys, frozen specs and the `_CUE_CLS` memo. Env overrides: WIDEFIELD_REST_START and
  WIDEFIELD_STOP_MIN_START_S.
- **(c)** s1_trial_class_snapshot passes `cue_s` to classify_session (the 5-min rule); its "warm-up" comment is
  rewritten.
- **(d)** Tests:
  - the 299.9 / 300.0 s edge;
  - `cue_classes_for` applies the time rule;
  - the `stopRun6t600` suffix;
  - `-restsession` in the run and cache tags.
- **(e)** Labels now read "licks / trial (cue + 70 ms to stop)" (spout_behavior POS_METRICS and LICK_METRICS,
  behavior_deck, epoch_behaviour). The 11-03 DECISIONS line "should cut at the response window" is rewritten to the
  cue + 70 ms to stop rule.
- **(f)** DECISIONS: a session with no camera lick can have stopped trials (a run of >= 6 no_response starting >= 300 s);
  no curated session is affected.
- **(g)** quit_prodrome and lick_bout_structure use `daq_trials.CONTACT_FROM_CUE_S`.
- **(h)** quit_prodrome includes the last trial (`range(len(tt))`; the frac denominator is n - 1). Its ILI loop and
  lick_bout_structure no longer skip the last trial; the next cue is only a cap (the session end for the last trial).
- **(i) Lick detectors (report only; a decision for the coordinator, CLAUDE.md rule 9).**
  - Detector A: `lick_detection.detect_licks` with the yaml `lick_detection` params (2.5 / 0.5 V, lockout
    (0.001, 0.020) s, min ILI 40 ms). Used by `daq_trials.decode` (spout_behavior, the DAQ trial table, trial stops),
    `dlc_frames.lick_onsets` (trial_classes' DAQ contacts) and `behavior_events` (same params).
  - Detector B: `plot_lick_aligned_averages._load_daq_events(h5, "lick_analog", 2.5, 1.0, (0.001, 0.020), 0.10)`
    (lower threshold 1.0 V, 100 ms refractory). Used by `analysis_kit.lick_samples` (quit_prodrome,
    lick_bout_structure) and `engagement_decomposition.session_trials` (`n_licks_trial`).
  - Counts from cue + 70 ms to the stop:

    | session | trials | A | B | trials that differ (B - A) |
    |---|---|---|---|---|
    | PS93_0908 | 511 | 5,304 | 5,203 | 101 (-1 each) |
    | PS94_0818 | 600 | 1,486 | 1,484 | 2 (-1 each) |

  - They DIFFER, so nothing was unified.
- **N3** stays undone (8.19).

### 8.21 Fifth review scope D + E blockers (D-B1, D-B2, D-B3, N1) and the N5 report
- **D-B1.** Section G `frozen(d)` no longer catches store errors. A WriteGuardError, the LP "loads DAQ-frozen"
  refusal, an unreadable entry or a centre refusal now propagates instead of printing "fitting locally" and refitting
  in memory, untagged. A pool that records no identity (`frozen` / `_g_store`) is still fitted in memory in the
  DEFAULT run, but raises `SectionGRefused` (a SystemExit) in any tagged run. The WIDEFIELD_SECTION_G_FROZEN=0 opt-out
  is already tagged `inmem-section_g` by run_mode.
- **D-B2.** `spout_scaling.BalEmCentreRefused(SystemExit)` is raised by `centre_rows` (bad setting), `centre_mask`
  (no classes) and `scored_frame` (no mask). A generic `except Exception` cannot swallow it. The reraise_under_camera
  fixes in `_collect_5c` and `run_animal` stay. Test: the per-class confusions call under the camera gate without
  classes stops the run even inside `try / except Exception`.
- **D-B3.** The minimum is now 20 working rows (`MIN_CENTRE_ROWS`).
  - The only fallback is explicit: `bal_em.centre_fallback_contact: ["PS93_0813"]` (no video). Its centre comes from
    its DAQ contact (lick-class) rows, the rows its lick-aligned, contact-only selections score; never no-lick rows.
  - The fallback is recorded in the spec (`scaling_params.centre_set_fallback` + `centre_min_rows`, so the spec id
    changes), in every `figure_meta` sidecar (`bal_em,centre_fallback_sessions`), in the camera-gate caption
    (`lp_licks.stamp`), and in a log line naming the session.
  - Any other session below 20 raises. A session in a WORKING-gated selection cannot be a fallback session: those
    sessions have no camera-working rows, so `_class_select` never selects them. A session with 1-19 working rows
    that is not in the list raises at centre time.
  - DECISIONS line rewritten.
  - Session labels now reach every centre site (`FrozenFrame(labels=)` and `scored_frame(labels=)` in all callers).
  - Not done: the fallback list is not written into every result JSON, only figure sidecars, captions, specs and logs.
- **N1.** quiet_task_events_followup raises `BalEmCentreRefused` under the working centre (it has no mask).
- **N5 (report only).** File: `C:/Users/SABATI~1/AppData/Local/Temp/claude/C--Users-SabatiniLab/f082ee0c-b33c-49a8-a537-dcee3bb7c84c/scratchpad/n5_report.txt`.
  The REDO families of MASTER_MANIFEST outside epoch_grant_figures / grant_figures were classified heuristically by
  source text.
  - 33 scripts select their pre set from all pre sessions with no PRE_FROM / Aug-only marker. They would use
    June + August pre under the profile. Among them: rest_migration rest_frozen_decoder, rest_coupling,
    transfer_arms, rotation_maps, reference_family_figure, quit_point, flatpool_vs_restw; locomotor_decoder,
    grant_encoder, cd_trajectories, chronic_stability, chronic_drift_vs_reorg, lick_cued_ramp, lick_iti_prelick.
  - 5 scripts write LP_licks outputs, so their render block must set WIDEFIELD_LICK_SOURCE=lp. Three of them never
    read the switch themselves: epoch_accuracy, epoch_maps, locanmf_analysis_deck. The other two are lp_final_render
    and restnl_maps.

### 8.22 Test record (a06f7fb7 + the rename after it)
Full suite at a06f7fb7, 6 shards: 4,572 passed, 178 skipped, 1 failed. The failure was
`test_spout_bal_em::test_scoring_takes_no_labels_by_signature`: `scored_frame` had gained a `labels` parameter (session
names for the centre fallback), a name that test reserves for trial labels. It is renamed `session_names`, and the test
and the centre suites pass.

### 8.23 Lick detector for counts (i), per-block LP env (N5b), pre-set report (N5a)
- **(i), coordinator decision.** Every lick COUNT now uses detector A through the new `analysis_kit.lick_count_samples`
  (`daq_trials.decode` with the yaml `lick_detection` params, the detector of the trial table, stops and camera
  contacts).
  - Consumers of the B discriminator (`analysis_kit.lick_samples` / `_load_daq_events(.., 0.10)`) were split:
    - COUNTS: `analysis_kit.session_behavior` (-> quit_prodrome, lick_bout_structure, disengagement_tests) and
      `engagement_decomposition.session_trials` (n_licks_trial, n_licks_pre, cum_licks). These now use A.
    - IMAGING ALIGNMENT / windows, unchanged: locanmf_* (decoder, encoder, cue_lick, lick_aligned, firstlick,
      weights, dff), precue_lickfree / precue_window_sweep, locomotor_features, rt_drift, nolick_decoder.categorize,
      gate_source, fixed_scale_maps, channel_position_maps, evoked_hrf_latency, nvc_evoked, enl_lick_rates, fr_enl_decode.
  - engagement_decomposition's RESPONSE detection (hit / latency -> the engagement gate and the quit point) stays on
    B, so `test_lp_gate_option`'s pinned quit is unchanged.
  - DECISIONS entry added; tests in test_licks_per_trial.
- **N5b.** The profile has a `block_env:` (lp_final_render, restnl_maps, epoch_accuracy, epoch_maps,
  analysis_deck_lp: WIDEFIELD_LICK_SOURCE=lp), applied by `render_profile.apply(..., block=)` and printed by
  `--block`. `lp_licks.refuse_lp_folder` stops a write into an LP_licks folder when the switch is unset. It is called
  in `epoch_figures._save_png_svg` and `write_values` (every epoch family, epoch_accuracy and epoch_maps included), in
  `figure_layout.sidecar_for` (every sidecar), and in `build_analysis_deck`. Test in test_render_profile.
- **N5a (report, no code).** Runtime check under the profile (no PRE_FROM):
  - The selectors: `config.phase_labels("pre")` = 44 sessions incl. 12 June; `analysis_kit.curated_labels` = 120
    incl. 12 June; `epochs.epoch_of("PS92_0606")` = "pre". With WIDEFIELD_PRE_FROM=0801: 32 pre sessions, 0 June, and
    June's epoch_of is None.
  - **June + August (27, + cohort2_qc/quiet_area_spikes):** every flagged script whose pre set comes from those selectors: axis_holdout_null,
    chronic_drift_vs_reorg, chronic_stability, engagement_axis_balance, lick_iti_prelick, lick_template_test,
    nocontact_census, fig_state_ps94_diagnostic, reembedding_test, cd_trajectories, grant_encoder, locomotor_decoder,
    and rest_migration channel_evoked_sign, evoked_hrf_latency, flat_vs_timelocal, flatpool_vs_restw, nvc_evoked,
    quit_point, regen_lick_maps, regen_running_maps, rest_coupling, rest_frozen_decoder, rotation_maps,
    shared_position_projection, substitute_code, transfer_arms, uniform_shift_check. cohort2_qc/quiet_area_spikes is
    cohort-1 + 2 QC, not a baseline for this render.
  - **Pre from upstream tables (4):** lick_code_noisematch, lick_cued_ramp, reference_family_figure and
    face_prelaunch_aim take "pre" from upstream per-session tables' epoch labels. They inherit the upstream run's pre
    set, so they follow PRE_FROM wherever the upstream does. Not verified per table.
  - **Not a pre baseline:** regen_lick_maps / regen_running_maps use phase_labels only to list sessions to rebuild.
    They are "June + Aug" only in the sense that they rebuild June sessions' maps too.
  - **Profile-wide PRE_FROM=0801** would take:
    - `WIDEFIELD_PRE_FROM: "0801"` in the profile env, which changes the expected run tag (adds `pre0801`; every
      output and cache tag moves, and the frozen specs change with the training labels);
    - block exemptions that UNSET it for june_august_drift (which refuses PRE_FROM) and 12.1b
      (rest_frozen_decoder_nl, its own June + Aug pre set). `block_env` needs a "remove" value (e.g. null -> pop);
    - `epoch_audit.full_cohort` already suspends PRE_FROM, so epoch boundaries do not move.

### 8.24 Test record (3da8e480) and the pinned quit test
Full suite at 3da8e480: 4,570 passed, 178 skipped, 6 failed. All 6 were `test_lp_gate_option` pins of the quit's
LICK-COUNT descriptives (quit_cum_licks, early_lpm, early_lpt). Those moved on purpose with the detector-A decision.
The gate and the quit point (hit-based) are still pinned exactly; only those three count fields are now skipped
(COUNT_DESCRIPTIVES). The file passes (18 passed).

### 8.25 Test record (8b67097d)
Full suite, 6 shards: 4,576 passed, 178 skipped, 0 failed.

### 8.26 Re-verify at 3a772a26: non-blockers
1. **disengagement_tests.** The per-trial loop is now `_trial_rows(.., lk, lk_resp)`. The RESPONSE measures (hit /
   hit35, latency, n_post, n_pre2, and `coll` through hit) use `analysis_kit.lick_samples` (unchanged); the COUNTS
   (ILIs, bouts, licks / bout, deceleration) use detector A. Test (real session PS93_0908): the response measures equal
   the pre-change values (one detector for all), and licks / bout moved.
2. **Sidecar fallback.** `figure_meta` writes `bal_em,centre_fallback_sessions` = the sessions whose contact-row
   fallback was actually USED (`spout_scaling.FALLBACK_USED`; an empty value when none was). This is process-level,
   accumulated up to the write. The caption likewise names only the fallbacks used.
3. **render_profile.** `--check` honours `--block`: the block's env is applied, the run tag must match, and the full
   tag must end with it (an LP block prefixes LP_licks). An unknown block name (in neither block_env nor block_args)
   raises KeyError. Tests added.
4. **_collect_5c.** The lru cache moved to `_collect_5c_cached(align, variant, mode, _run)`. `_run` = (full output tag,
   bal_em scaling, centre rows), filled in by `_collect_5c`. Source tests point at `_collect_5c_cached`; grant_figures
   re-exports it. Test added.

### 8.27 Priya 11-03 late: August-only pre profile-wide (5b6f35df)
- The render profile sets `WIDEFIELD_PRE_FROM=0801` (run tag part `pre0801`, in both expected run tags). The per-block
  `--pre-from` arguments are removed, so the env is the only source.
- `block_env` null unsets PRE_FROM for `june_august_drift` and `rest_frozen_decoder_nl` (12.1b). `render_profile.check`
  strips `_pre...` from the expected tag for such a block.
- Epoch boundaries come from the full pre set: `epoch_audit.full_cohort` lists WIDEFIELD_PRE_FROM in SUBSET_ENV and
  suspends it (confirmed).
- The four inherit-upstream scripts:
  - `iti_licks`: its extracts are now keyed by pre (`run_mode.subdir(..., include=(... "pre" ...))`).
  - The ITI consumers read those keyed extracts.
  - `reference_family_figure` reads routed, pre-tagged epoch outputs.
  - `face_prelaunch_aim` reads `encoding_root` under the profile.

### 8.28 Jaw-only out of working, matched to main 48f2cc6d (5b6f35df, then db33dac2)
Priya 11-03 late: working = contact + incomplete; jaw_only is its own class, like no_response. The first version
(5b6f35df) narrowed `GATES["working"]` and added a `wkci` tag part. db33dac2 replaces that with main's form so that the
merge is trivial:
- **Setting.** `wfield_local/working_trials.py` is copied from main and is byte-identical (same blob 1a56bb25).
  `WIDEFIELD_WORKING_TRIALS` takes the values:
  - `in_working` (production, the default; the profile sets it);
  - `in_working_jaw` (the jaw-inclusive robustness arm);
  - `quit_tail` (the DAQ rule, i.e. `WIDEFIELD_TRIAL_GATE=daq` here). Under the camera gate it raises `DaqGateRefused`.
- **Classes.**
  - As on main, the static `trial_classes.GATES["working"]` and `in_working` stay jaw-inclusive.
  - Every rebuild consumer of the camera working set reads `trial_classes.in_sel(cls, "working" | "miss_working")`,
    which applies the selection:
    - production: working = contact | incomplete; miss_working = incomplete;
    - in_working_jaw: working adds jaw_only; miss_working = incomplete | jaw_only.
  - The consumers are: camera_rows, camera_select, spout_scaling.centre_mask (the bal_em centre), nolick_decoder,
    attempt_coding, attempt_maps, enl_states, position_axes, position_coding_directions, poststroke_compare and
    precue_engagement_states.
  - Under the camera gate, a jaw_only trial inside the working period keeps its own label ("reaction" in attempt
    coding).
- **Tag.** The tag uses main's class part, placed after the end part:
  `sel-camera-cue_stop-endlick-cls-contact-incomplete-stopRun6`. The drift arm carries the same part.
  - `in_working_jaw` keeps the afternoon tag `sel-camera-cue_stop-endlick-stopRun6`.
  - `working_trials.tag()` on this branch returns main's strings exactly.
  - FREEZE_VERSION stays 2.
- **Stop runs.** Main has no stop-run rule: its working set is not cut by stops, and it has no stopped class in
  trial_classes. There is therefore nothing to match, and the coordinator's default stays:
  - Production: a jaw_only trial bridges a run anywhere, does not count toward the 6, and is never stopped.
  - Under `in_working_jaw` the afternoon rule holds: a jaw_only trial at or before the last camera lick ends the run,
    and one after it bridges.
  - `cue_classes_for` memo keys carry this choice.
- **5c cache.** `grant_confusion._collect_5c`'s key now carries `trial_classes.run_selection()` and
  `working_trials.selection()` explicitly. Before this, it carried them only through the output tag (the a8 pitfall).
- **Kept differences from main:**
  - Main narrows working in `working_trials.row_masks`, which reads its s1 snapshot. The rebuild narrows it in
    trial_classes classes computed live (`cue_classes_for`), because the rebuild does not read the snapshot. The class
    semantics are the same.
  - The rebuild's tag also carries `stopRun6` (main has no stop rule).
  - The `in_sel` / `sel_classes` / `sel_cls_part` helpers are rebuild-only. On merge, main's grant_kit and
    locanmf_frozen_decoder use `working_trials.is_camera()`, and the rebuild's camera branch stays in trial_classes.
- **Tests:** `tests/test_working_selection_main.py` (static gates = main's; production / jaw-arm sets; tags incl.
  `working_trials.tag()`; quit_tail raises under camera; bridging by selection; 5c key carries the selection) plus updated
  pins in test_trial_classes, test_rebuild_switches, test_camera_gate_fail_closed, test_review_second_pass.

### 8.29 Dry count (re-verify item 5) under the new working definition
Read-only: the profile applied, the rebuild root unset, scratch caches. For each of the 108 curated sessions the count is
the ENGAGED rows (the decoder's DAQ lick-arm category) whose camera class is production working. These are the rows the
bal_em centre is fitted on, before the per-alignment window drop. Errors: 0. The lowest 12, minimum first:

| session | epoch | cues | engaged | engaged & working | unknown |
|---|---|---|---|---|---|
| PS93_0813 | pre | 553 | 482 | 0 (no video: contact fallback, 482 rows) | 553 |
| PS94_0819 | acute | 600 | 174 | 174 | 0 |
| PS92_0812 | pre | 225 | 221 | 221 | 0 |
| PS93_0820 | acute | 600 | 235 | 235 | 0 |
| PS94_0818 | acute | 600 | 239 | 239 | 0 |
| PS92_0820 | acute | 559 | 243 | 243 | 0 |
| PS93_0821 | acute | 540 | 259 | 259 | 0 |
| PS92_0826 | subacute | 268 | 260 | 260 | 0 |
| PS92_1002 | chronic | 270 | 269 | 269 | 0 |
| PS92_0929 | chronic | 272 | 272 | 272 | 0 |
| PS92_0824 | subacute | 325 | 273 | 273 | 0 |
| PS93_0819 | acute | 600 | 277 | 277 | 1 |

- No session other than PS93_0813 is below 20; the minimum is 174 (PS94_0819).
- By epoch (min / median), excluding PS93_0813 from pre:

  | epoch | min | median |
  |---|---|---|
  | pre | 221 | 474 |
  | acute | 174 | 291.5 |
  | subacute | 260 | 424 |
  | chronic | 269 | 490 |

- An engaged row is a DAQ contact, so its camera class is contact, and contact is working under both selections.
  Engaged and engaged & working differ in only three sessions:
  - PS93_0813 (no video);
  - PS92_0908: 373 vs 309, 68 unknown;
  - PS92_0904: 378 vs 377.
- The jaw change therefore does not move the centre counts.
- As flagged before, the centre (engaged XE rows) is effectively the contact rows. The checks' "jaw as its own class"
  cell centres on contact + incomplete rows.

### 8.30 No-video DAQ fallback: agreement check FAILS, so it is not wired
The check covers the 31 August pre sessions with video, for all four animals. PS93_0813 has no video. The working period
is defined two ways:
- **Camera (new definition).** Cues up to the last camera lick, minus stopped and early_run trials. This is
  `cue_classes_for` at 5b6f35df, the same as production in_working.
- **DAQ fallback.** Cues up to the last DAQ hit, minus DAQ stops (a run of 6 or more DAQ no-response trials starting at
  least 300 s after the first cue).

The measures are the DAQ-only behaviour rates: hit and no-response rate (a DAQ lick from cue + 70 ms to min(cue + 3.5 s,
next cue)) and the median latency of hits. The pass rule is at most 2 pp for hit and no-response rates and at most 20 ms
for latency, on every session.

Result: **FAIL on 9 sessions**: PS93_0814, PS94_0806, PS94_0811, PS94_0812, PS94_0814, PS95_0806, PS95_0807,
PS95_0811 and PS95_0812.
- The largest is PS95_0806 at +15.9 pp.
- Latency agrees exactly everywhere: both periods contain the same hits.
- The fallback always reads higher on hit rate. A camera incomplete or jaw_only trial is a DAQ miss: it extends a DAQ
  miss run, but it ends a camera no_response run (incomplete) or is not counted in it (jaw_only). So DAQ stops are
  longer and more frequent, and they remove misses that the camera period keeps. The camera period can also end later,
  at a last incomplete lick after the last DAQ hit.

Per the rule, the fallback is NOT wired: no `trial_selection.daq_fallback_sessions` setting, and PS93_0813 stays out of
the DAQ-only behaviour rates under the camera gate. Script: scratchpad `daq_fallback_check.py`.

Agreement table (camera working period vs DAQ fallback; d = DAQ - camera):

| session | n cam | n DAQ | hit cam | hit DAQ | d hit (pp) | no-resp cam | no-resp DAQ | d no-resp (pp) | median lat cam / DAQ (ms) | pass |
|---|---|---|---|---|---|---|---|---|---|---|
| PS92_0806 | 380 | 380 | 0.929 | 0.929 | +0.00 | 0.071 | 0.071 | +0.00 | 567.8 / 567.8 | yes |
| PS92_0807 | 430 | 430 | 0.986 | 0.986 | +0.00 | 0.014 | 0.014 | +0.00 | 287.2 / 287.2 | yes |
| PS92_0809 | 431 | 424 | 0.956 | 0.972 | +1.58 | 0.044 | 0.028 | -1.58 | 243.2 / 243.2 | yes |
| PS92_0810 | 404 | 404 | 0.995 | 0.995 | +0.00 | 0.005 | 0.005 | +0.00 | 220.6 / 220.6 | yes |
| PS92_0811 | 386 | 386 | 0.997 | 0.997 | +0.00 | 0.003 | 0.003 | +0.00 | 199.4 / 199.4 | yes |
| PS92_0812 | 225 | 225 | 0.982 | 0.982 | +0.00 | 0.018 | 0.018 | +0.00 | 234.6 / 234.6 | yes |
| PS92_0813 | 339 | 339 | 0.997 | 0.997 | +0.00 | 0.003 | 0.003 | +0.00 | 243.8 / 243.8 | yes |
| PS92_0814 | 310 | 310 | 0.984 | 0.984 | +0.00 | 0.016 | 0.016 | +0.00 | 225.4 / 225.4 | yes |
| PS93_0806 | 530 | 530 | 0.919 | 0.919 | +0.00 | 0.081 | 0.081 | +0.00 | 451.6 / 451.6 | yes |
| PS93_0807 | 547 | 547 | 0.888 | 0.888 | +0.00 | 0.112 | 0.112 | +0.00 | 399.3 / 399.3 | yes |
| PS93_0809 | 566 | 566 | 0.951 | 0.951 | +0.00 | 0.049 | 0.049 | +0.00 | 360.0 / 360.0 | yes |
| PS93_0810 | 559 | 559 | 0.866 | 0.866 | +0.00 | 0.134 | 0.134 | +0.00 | 441.7 / 441.7 | yes |
| PS93_0811 | 491 | 489 | 0.945 | 0.949 | +0.39 | 0.055 | 0.051 | -0.39 | 279.3 / 279.3 | yes |
| PS93_0812 | 493 | 483 | 0.921 | 0.940 | +1.91 | 0.079 | 0.060 | -1.91 | 233.1 / 233.1 | yes |
| PS93_0814 | 534 | 512 | 0.908 | 0.947 | +3.90 | 0.092 | 0.053 | -3.90 | 212.8 / 212.8 | **no** |
| PS94_0806 | 326 | 315 | 0.951 | 0.984 | +3.32 | 0.049 | 0.016 | -3.32 | 172.8 / 172.8 | **no** |
| PS94_0807 | 444 | 444 | 0.930 | 0.930 | +0.00 | 0.070 | 0.070 | +0.00 | 185.0 / 185.0 | yes |
| PS94_0809 | 616 | 616 | 0.994 | 0.994 | +0.00 | 0.006 | 0.006 | +0.00 | 178.6 / 178.6 | yes |
| PS94_0810 | 695 | 693 | 0.955 | 0.958 | +0.28 | 0.045 | 0.042 | -0.28 | 159.2 / 159.2 | yes |
| PS94_0811 | 411 | 399 | 0.956 | 0.985 | +2.88 | 0.044 | 0.015 | -2.88 | 147.4 / 147.4 | **no** |
| PS94_0812 | 488 | 466 | 0.934 | 0.979 | +4.41 | 0.066 | 0.021 | -4.41 | 145.3 / 145.3 | **no** |
| PS94_0813 | 647 | 647 | 0.998 | 0.998 | +0.00 | 0.002 | 0.002 | +0.00 | 146.6 / 146.6 | yes |
| PS94_0814 | 672 | 650 | 0.960 | 0.992 | +3.25 | 0.040 | 0.008 | -3.25 | 140.0 / 140.0 | **no** |
| PS95_0806 | 519 | 434 | 0.813 | 0.972 | +15.92 | 0.187 | 0.028 | -15.92 | 135.3 / 135.3 | **no** |
| PS95_0807 | 666 | 651 | 0.937 | 0.959 | +2.16 | 0.063 | 0.041 | -2.16 | 134.8 / 134.8 | **no** |
| PS95_0809 | 682 | 682 | 0.993 | 0.993 | +0.00 | 0.007 | 0.007 | +0.00 | 140.6 / 140.6 | yes |
| PS95_0810 | 706 | 698 | 0.953 | 0.964 | +1.09 | 0.047 | 0.036 | -1.09 | 143.6 / 143.6 | yes |
| PS95_0811 | 629 | 614 | 0.962 | 0.985 | +2.35 | 0.038 | 0.015 | -2.35 | 139.0 / 139.0 | **no** |
| PS95_0812 | 605 | 552 | 0.850 | 0.931 | +8.16 | 0.150 | 0.069 | -8.16 | 144.2 / 144.2 | **no** |
| PS95_0813 | 870 | 870 | 0.900 | 0.900 | +0.00 | 0.100 | 0.100 | +0.00 | 174.8 / 174.8 | yes |
| PS95_0814 | 609 | 599 | 0.946 | 0.962 | +1.58 | 0.054 | 0.038 | -1.58 | 139.5 / 139.5 | yes |

### 8.31 Read-only validation of the jaw rule (the checks' "jaw as its own class" cell)
- **Setup:**
  - `grant_confusion._collect_5c("cue", "working", "frozen")` at 5b6f35df (= production in_working);
  - under the profile, with the rebuild root unset, scratch caches, and `WIDEFIELD_EPOCH_FROZEN=0` (in-memory fits,
    nothing stored);
  - joint basis, bal_em centre.
- **Measures:**
  - pre panel = the pre-stroke LOSO record per animal;
  - acute close ipsi = the close_L trials pooled over each animal's acute days (`epoch_of_day`).
- The checks average session means instead, so the aggregation differs slightly. Script: scratchpad
  `jaw_validation.py`.

| animal | pre panel | acute close ipsi (n) | main 48f2cc6d: pre / acute close ipsi |
|---|---|---|---|
| PS92 | .864 | .898 (315) | .854 / .829 |
| PS93 | .769 | .664 (253) | .724 / .629 |
| PS94 | .936 | .828 (430) | .926 / .619 |
| PS95 | .935 | .966 (119) | .921 / .832 |
| **mean** | **.876** (checks .856) | **.839** (checks .727) | |

- **Pre panel:** same ordering and about the same size, +.02.
- **Acute close ipsi:** the rebuild reads higher, +.11 on the animal mean. Most of the gap is PS94 (+.21) and PS95
  (+.13).
- **Expected sources** (not separated):
  - the rebuild's data: the swap-mask hemo variant and the LocaNMF projection;
  - pooled trials vs session means (PS94 has 6 acute sessions, PS95 one);
  - the centre and training rows. In the rebuild the bal_em centre is the engaged (contact) rows and the frozen
    training set is the engaged pre rows. The checks' cell trains and centres on contact + incomplete. Flagged, not
    restructured.
- **Direction:** the jaw rule does not lower the rebuild's acute close-ipsi below the checks' cell.

### 8.32 Test record (db33dac2 + the leak fix)
Full suite, 6 shards, at db33dac2: 4,580 passed, 179 skipped, 8 failed. All 8 had one cause. `scripts/disengagement_tests.py`
runs `os.environ.setdefault("WIDEFIELD_PRE_FROM", "0801")` at import, so `test_disengagement_detectors` (8.26) leaked
pre0801 into the later lp_licks, rebuild_root and share_writers tests of its shard. The test now sets the variable
through monkeypatch before the import. Re-run of that shard: 646 passed, 26 skipped, 0 failed. Total 4,588 passed,
179 skipped, 0 failed.

### 8.33 Coordinator BLOCKER 11-04: training, centre, pre panel and scored set on camera working rows (as main 48f2cc6d)
- **Problem.** For cue and precue, the frozen spout decoder was trained only on the DAQ-engaged (contact) pre rows, and
  each session's bal_em centre came from those rows too. So camera-INCOMPLETE trials, which have no DAQ lick and are
  no-lick rows, never entered the training set or the centre. They did enter the scored set through `_class_select`.
  The 8.31 validation ran on that code.
- **Now (mirrors main).** For cue and precue, the training set, the centre, the pre panel and the scored set are the
  camera WORKING rows on both arms: every engaged or no-lick row whose class is in the run's working set
  (`trial_classes.in_sel`; production contact | incomplete, whatever the DAQ lick).
- **Lick window.** It keeps the lick class (DAQ contacts), because a trial without a lick has no lick time. The DAQ gate
  keeps the engaged model.
- **Sites changed**, each matched to main:
  - `grant_kit`: new `_working_masks` and `_uses_working`; `_frozen_frame(bd, an, align, store, sel)`.
    - `sel="working"` trains on R = XE working rows plus XU working rows of the pre sessions that have them.
    - Each session is centred from its own R rows (LOSO model for a pre session) via `spout_scaling.working_centre`
      (at least 20 rows, otherwise it raises). A session with no R rows is neither trained on nor scored.
    - It returns `R`, `RT` (R in the training frame), `train_sessions` and `work`.
    - The memo key carries the selection. `_frozen_score_rows(..., centre_rows=)` is used by the matched arm.
  - `grant_confusion`:
    - 5b: the working frame. Its pre panel is the held-out pre session's working rows (engaged + no-lick).
    - 5c: `sel="working"` for the working variant. The pre loop runs over the train sessions and scores the held-out
      session's working rows (`if variant == "stopped" or uw`). The matched-draw pools are R rows, and the matched
      spec carries the selection.
    - `_stored_epoch_decoder(..., R=)`: the spec's training data are the R rows, plus `trials`.
  - `locanmf_frozen_decoder`:
    - `working_rows`, `pool_work` (rebuild: live classes via `pool_row_classes`, not main's s1 snapshot);
    - `decoder_spec(trials=)` and `frozen_decoder_models(work=)`, which train on R and record `trials` / `pre_labels`;
    - `frozen_scored_frame(work=)`, which centres from R rows;
    - `pooled_frozen_loso`: ROI features with indices; pre band = sessions with a LOSO model; per-session values, CI and
      confusion over scored sessions only; `frozen_model_id` from the R rows.
  - `poststroke_compare` (section G):
    - `_pooled` uses ROI features with indices (`_features`);
    - `frozen` passes `work`, and `frozen_X` centres from R rows;
    - `decode_matched` and `crossed_confusion` drop pre sessions without a LOSO model.
  - `attempt_coding` (scoring and `--freeze`) and `joint_xsession` (decoder features with indices).
  - `spout_scaling`: `spec_kw(scaling, trials)` (main's signature; `centre_set: working_rows`) and `working_centre`.
- **Where the rebuild differs from main:**
  - Classes are computed live (`cue_classes_for`) rather than read from main's s1 snapshot.
  - The lick-gate (engaged) model centres on its working engaged rows, with the PS93_0813 contact fallback. Main
    centres it on all engaged rows. This is the rebuild's 11-03 centre rule and is unchanged.
  - The other `FrozenFrame` sites keep their engaged-row training. Main does not change them either: nolick_decoder,
    position_coding_directions, precue_engagement_states, poststroke `decode_matched`'s lick-only arm and
    june_august_drift.
- **Dry count** (re-run). The centre rows of the working frame are the engaged + no-lick rows of class contact |
  incomplete, counted per curated session before the window drop. Errors: 0.
  - No session other than PS93_0813 is below 20. PS93_0813 has 0 rows (no video) and is not trained on or scored.
  - The lowest sessions:
    - PS92_0812 (pre): **224**
    - PS94_0819 (acute): 249
    - PS92_0826 (subacute): 265
    - PS92_1002 (chronic): 270
    - PS92_0929 (chronic): 272
    - PS95_0820 (subacute): 284
  - By epoch (min / median, and no-lick working rows added):

    | epoch | min | median | no-lick working rows added |
    |---|---|---|---|
    | pre | 224 | 509 | 391 |
    | acute | 249 | 370 | 1,324 |
    | subacute | 265 | 446 | 360 |
    | chronic | 270 | 497 | 225 |

- **Validation** (`_collect_5c("cue", "working")`, in memory, session means per animal, then the animal mean).
  - The `paired` mode gives per-session records. Column 0 is the frozen prediction, and the refit column is stubbed.
  - (b) uses the checks' data: production hemo variant, no LocaNMF projection, PRE_FROM 0801, the profile's rules
    otherwise.
  - (a) uses the full rebuild profile.

| | pre panel PS92 / 93 / 94 / 95 -> mean | acute close ipsi PS92 / 93 / 94 / 95 -> mean |
|---|---|---|
| checks (main 48f2cc6d) | .854 / .724 / .926 / .921 -> **.856** | .829 / .629 / .619 / .832 -> **.727** |
| (b) rebuild code, checks' data | .853 / .724 / .926 / .922 -> **.856** | .829 / .629 / .619 / .832 -> **.727** |
| (a) rebuild profile | .855 / .729 / .934 / .922 -> **.860** | .818 / .584 / .717 / .815 -> **.734** |
| (a) - (b), the data effect | +.002 / +.005 / +.008 / .000 -> +.004 | -.011 / -.045 / +.098 / -.017 -> +.007 |

- (b) reproduces the checks to .001 on every animal, so the rebuild code has no code difference on this cell.
- The data effect (swap-mask hemo plus the LocaNMF projection) is small on the pre panel. On acute close ipsi it is
  mixed in sign: PS94 is +.10 and PS93 is -.05, giving +.007 on the mean.
- 8.31's +.11 came from the contact-only training and centre plus trial pooling, not from the data.
- Files: scratchpad `jaw_val/{sessions,animals}_a.csv` and `jaw_val_b/{sessions,animals}_b.csv`; script
  `r4/jaw_validation2.py`.
- **Tests:**
  - New `tests/test_working_rows_training.py`:
    - the working frame trains on and centres from engaged + no-lick working rows;
    - a no-video session is neither trained nor scored;
    - the engaged frame and the lick window keep the engaged rows;
    - the jaw arm adds jaw_only rows;
    - `pool_work` / `working_rows`;
    - the spec records the selection;
    - `working_centre` refuses 1–19 rows.
  - Updated source pins in test_frozen_default, test_stopped_trial_class (main's wording) and test_merge_review_b1
    (the pool now carries XU).

### 8.34 Test record (2c59f85b)
Full suite, 6 shards: 4,593 passed, 179 skipped, 0 failed.

### 8.35 Review 7A / 7B at 6a85304d (cf94e7c3, plus a test pin)
Following the coordinator's revision, the rebuild follows MAIN's 11-03 design pending Priya. The section D / G scored
sets and the pattern families' pre reference were therefore NOT converted to working rows (DECISIONS 2026-11-04).

**7A**
- **B1 (5b).** `_from_json` carries main's guard verbatim (`if _wt.is_camera(): return None`). 5b therefore always
  recomputes from `_frozen_frame(bd, an, align, sel="working")`: working-trained, with the pre panel on the held-out
  working rows. The panel titles and suptitle name camera WORKING (c+i) under the camera gate.
- **B2.** Kept as main. Section D (`pooled_frozen_loso`, also behind joint_xsession) and section G (`crossed_confusion`,
  `decode_matched`) score engaged / all trials with the working-trained model. This is recorded:
  - `pooled_frozen_loso` result: `scored_rows` = "engaged", `model_trials`;
  - `section_g_design.json` beside section_g.json;
  - `trial_classes.DESIGN_NOTE` in the corner stamp of every camera-gated figure.
- **attempt_coding.** Its pre reference is `pre_hit` (the LP attempt classes). It is an attempt-class analysis by
  design and is left as it is; its frozen model is working-trained (DECISIONS).
- **NB1.** The `_collect_7` pre reference stays LICK trials, as main. The captions say so via
  `_class_note(v, pattern=True)` in grant_similarity / geometry / encoder / matching.
- **NB2.** `_class_note("working")` describes the active selection: "camera WORKING trials: contact + incomplete, cue to
  the session's last camera lick; jaw_only excluded" (+ jaw_only under in_working_jaw). The DAQ text is unchanged.
- **NB3.** PS93_0813 is now visible as excluded:
  - `spout_scaling.note_working_excluded` logs "[working] PS93_0813: ... EXCLUDED from the working selection" once;
  - figure sidecars carry `working,working_excluded_sessions`, and the corner caption names it;
  - the store log line names the lick source AND the selection.
- **NB4.** A post session with 0 working rows raises `BalEmCentreRefused`, in both `frozen_scored_frame(work=)` and
  `_frozen_frame`. Pre sessions without rows are excluded and logged.
- **NB6.** The `_collect_7` bundle failure calls `reraise_under_camera`. `_collect_7` is also keyed by the selection
  (`_collect_7_cached(..., _run)`), the same pitfall as 5c.
- **NB7.** `working_trials.py` stays byte-identical to main (pinned by a test); the rebuild note is in DECISIONS.

**7B**
- **B1 (june_august_drift).**
  - A tagged run is allowed only with PRE_FROM unset (the profile's block); a PRE_FROM run raises.
  - It writes to `run_mode.june_august_drift_dir` (`chronic_stability/<own tag>/june_august_drift`).
  - It reads the stroke twin from the primary run via `primary_chronic_csv` → `run_mode.primary_output_dir`
    (pre0801), and refuses if that file is missing.
  - The deck reads Fig 4.3 through `june_august_drift_dir`, so it no longer depends on the relative `../../` path.
- **B2.** New `block_env map_drift_prepre: {WIDEFIELD_PRE_FROM: null}`. The script sets its own arm's lick source.
- **B3.** reembedding_test selects June from the curated set with PRE_FROM suspended (`run_mode.with_env`). It raises
  on an empty June set. Its pickle cache is keyed by the output tag.
- **N1.** New `block_env restnl_mask: {WIDEFIELD_PRE_FROM: null}`.
- **N2.** rest_frozen_decoder_nl `--compare` reads page 12.1's 15f from the primary tag (`page_sessions()`). Its labels
  say "Aug pre" in a tagged run.
- **N3 (deck).** The default `grant_dir` of a tagged run is `lp_licks.output_dir(grant_figures)`, i.e.
  `grant_figures/<tag>` and its `epoch`.
  - Sections H and I count registered figures they cannot find as MISSING (a tagged build only).
  - Section K resolves the LP / DAQ_ENL1s epoch families and the LP attempt maps under their blocks' tags
    (`_tagged_k`).
  - Sections A–G read `src` (figures_working, inside the rebuild root).
  - Section J reads the tagged epoch dir.
- **New helpers:** `run_mode.with_env`, `primary_output_dir`, `june_august_drift_dir`, `PRIMARY_PRE_FROM`.
- **Tests:** `tests/test_review_7ab.py` (13); updated pin in test_one_pooling_per_animal_alignment (`_collect_7_cached`).

**Validation (extended).**
- Code from a CLEAN export of cf94e7c3 (`git archive`; imports checked to come from the export).
- The STORED path: `WIDEFIELD_EPOCH_FROZEN=1`, store = scratch `fm/`; the models were frozen there (new spec ids).
- Run as `_collect_5c(align, "working")`, session means per animal, then the animal mean.
- (b) uses the checks' data: production hemo, no projection, PRE_FROM 0801. (a) uses the rebuild profile.

| window | arm | pre panel PS92 / 93 / 94 / 95 → mean | acute close ipsi PS92 / 93 / 94 / 95 → mean |
|---|---|---|---|
| cue | checks | .854 / .724 / .926 / .921 → .856 | .829 / .629 / .619 / .832 → .727 |
| cue | (b) | .853 / .724 / .926 / .922 → **.856** | .829 / .629 / .619 / .832 → **.727** |
| cue | (a) | .855 / .729 / .934 / .922 → .860 | .818 / .584 / .717 / .815 → .734 |
| precue | checks | → .559 | → .567 (PS95 .513) |
| precue | (b) | .552 / .517 / .696 / .468 → **.558** | .729 / .476 / .556 / .513 → **.568** |
| precue | (a) | .564 / .508 / .704 / .481 → .564 | .590 / .473 / .610 / .529 → .551 |

- (b) reproduces the checks for cue (to .001) and precue (.558 / .568 vs .559 / .567; PS95 .513 exact).
- The stored path gives the same numbers as the in-memory path of 8.33 for cue.
- The data effect (a) − (b):
  - cue: pre +.004, acute close ipsi +.007;
  - precue: pre +.006, acute close ipsi −.017 (PS92 −.139, PS93 −.003, PS94 +.054, PS95 +.016).
- A store note: the frozen-models lookup reported SPEC-CHANGED against an earlier PS95 model frozen on 11 sessions
  (with June). The new August-only model was frozen under its own id in the scratch store; nothing was overwritten.

**Test record.** Full suite at cf94e7c3, 6 shards: 4,605 passed, 179 skipped, 1 failed. The failure was the source
pin of `_collect_7` (it is now `_collect_7_cached`); the pin is updated and the file passes (3 passed).

### 8.36 Review 8A / 8B at 11817e27 (fcbcd535, d8e44e6c)

**8A**
- **B1 (regression from NB4).** `pooled_frozen_loso` narrows `pre_i` to the sessions that have a LOSO model, which
  drops PS93_0813. `frozen_scored_frame` then took the narrowed list for the pre / post split and raised on PS93_0813 as
  a "post" session with 0 working rows.
  - Fix: pre vs post is decided by the session's PHASE (`spout_scaling.is_post_session` → `config.session_phase`).
    A PRE session with 0 working rows is excluded and logged; only a POST session raises.
  - `pooled_frozen_loso` also passes the un-narrowed pre list (`pre_all`).
  - `grant_kit._frozen_frame` applies the same rule.
  - Section G (`frozen_X` passes the full `d["pre_i"]`) and attempt_coding (full `pre_list`) were checked; both are
    correct under the rule.
  - Test: the `pooled_frozen_loso` sequence with a no-video pre session (synthetic pool, real config phases) does not
    raise, logs EXCLUDED, and leaves PS93_0813 out of `per_session`. A post session with 0 rows still raises.
- **NB-b.** The 5b titles say "(c+i)" or "(c+i+jaw)" from the selection.
- **NB-c.** `trial_classes.design_note()` builds its text from `sel_class_text()`.
- **NB-d.** The excluded sessions are now computed from the session classes (`trial_classes.working_excluded_sessions`:
  pooled sessions without camera classes; memoised), not from process state.
  - They are used by the corner stamp, the epoch-figure sidecars (`figure_meta`) and `section_g_design.json`.
  - DECISIONS now says "corner stamp + epoch-figure sidecars". `grant_kit._save` writes no meta sidecar.
- **NB-e.** The `test_review_7ab` fixture clears every owned env variable.
- **NB-f.** The store log names the gate: "engaged rows (camera gate | DAQ gate: DAQ lick-arm trials)".

**8B**
- **B1.**
  - june_august_drift uses `root("cohort_out")`, the derived tree, not the labcams input tree.
  - The deck resolves Fig 4.3 through the same helper on the same root (`_grant_root.parent` = cohort_out).
  - The test uses a cohort_out root.
- **B1b.**
  - `recovery_trajectory.run` writes by default to `lp_licks.output_dir(grant_figures/epoch)`, the tagged epoch dir
    that june_august_drift's `primary_chronic_csv` reads.
  - `scripts/chronic_stability.py` is a REDO row (MASTER_MANIFEST M05577). Its `require_default` refusal is replaced
    by tagged paths: it reads `grant_figures/<tag>/epoch/data` and writes `chronic_stability/<tag>/`.
  - chronic_drift_vs_reorg (Fig 10.2, M00086) was not touched in this batch.
- **N1 / N5.** `run_mode.june_august_drift_dir` and `primary_output_dir` clear WIDEFIELD_LICK_SOURCE, the ENL window and
  WIDEFIELD_GATE_SOURCE, so they resolve the DAQ-arm folders from inside an LP / ENL block (tested).
- **N2.** Section K's DAQ_ENL1s entry unsets the lick source, so under the LP deck it resolves the DAQ ENL figure
  (tested).
- **N3.** In a tagged build, a missing tagged grant dir or epoch dir counts every registered H / I figure as MISSING.
- **N4.** These legends with PRE-REBUILD numbers are marked `REBUILD-TODO(numbers)` in deck_registry, for the
  replacement pass:
  - section K: epoch_acc_by_position_lick_lick, epoch_accdelta_by_position_lick_lick,
    epoch_acc_by_position_cue_working, epoch_5c_frozen_confusion_lick_lick,
    DAQ_ENL1s epoch_acc_by_position_precue_working, epoch_15rpa_position_RESTref_by_animal_lick_lick,
    epoch_15r_position_RESTref_lick_lick, attempt_maps_lick_far_contra, attempt_maps_lick_far_contra_minus_pre,
    attempt_maps_cue_far_contra (10 entries);
  - `_DRIFT_GROUP` (Fig 4.3).
- **N6.** The rest_frozen_decoder_nl docstring now says that `--pre all` is the PRE-REBUILD page set and that the
  rebuild's page 12.1 is August-only, read via `page_sessions`.
- **N7.** Page sources name map_drift_prepre's rebuild folders (`<run tag>`): p_img_s7v3.py, p_img_licksrc.py,
  prov_s7.md.
- **Tests from tests/.** test_review_7ab now resolves paths from `Path(__file__).parents[1]` and runs git with
  `cwd=REPO`.
  - Running the suite with cwd = tests/ imports `wfield_local` from the editable install (the main checkout) unless
    PYTHONPATH names the worktree. The tests/-cwd suite below sets it.

**Test record.** Full suite at d8e44e6c, 6 shards, run twice: from the repo root, and with cwd = tests/ (PYTHONPATH = the
worktree). Each run: 4,612 passed, 179 skipped, 1 failed.
- The failure is `test_rebuild_root::test_subfolder_and_package_writers_are_classified`: recovery_trajectory is now
  routed (8B B1b), so it leaves the scanner's RESOLVED list. The list is updated in 2ddcf9fb.
- At 2ddcf9fb, test_rebuild_root + test_review_7ab pass from both cwds (37 passed each).

### 8.37 Pre-render follow-ups (d394dcbc)

**1. chronic_drift_vs_reorg (Fig 10.2, M00086).** It now writes `chronic_stability/<tag>/` (`run_mode.subdir`), like
chronic_stability.py, and calls `assert_writable` first.
- It reads no epoch data file: its inputs are the pooled bundles (`grant_kit._collect_7`), which follow the run.
- Test added.

**REDO / NO scan.** Scanned every producing script named in MASTER_MANIFEST REDO / NO rows for `require_default` and
for share writers without routing. The scan is heuristic: a writer AND a share root AND no run_mode / output_dir /
output_tag routing. Results:
- **REDO, refuses under the profile (`require_default`), 47 scripts, all in G5 families** (fr_musall /
  fr_direction_mechanism are in the O2 batch; within_lick_correction is in G4 extracts):
  best_match_destination, combined_encoding_figures, compare_map_tests, coupling_by_class, encoder_trialavg_r2,
  encoding_key_figures, encoding_mixed, enl_fr_removed_decode, enl_lick_audit, exec3d_pilot_figure,
  exec_angle_spread_check, exec_vs_neural_match, face_prelaunch_aim, fr_direction_mechanism, fr_encoding_summary,
  fr_musall, frozen_area_maps, frozen_group_leakage, frozen_musall_figures, frozen_pixel_maps, frozen_r2_maps,
  frozen_residual_significance, frozen_residual_trend, frozen_running_compare, frozen_spout_maps,
  imaging_baseline_compare, imaging_per_animal_tables, layer2_matched_n, layer2_random_splits, lick_interhemispheric,
  lick_template_test, lp_encoder_decompose, misdirected_lick_decode, position_coding_trend_recovery,
  prelaunch_aim_tracking, prelick_hemisphere_tuning, qc_running_speed, relabel_confusion, slowband_caveat_figure,
  softmax_neural_vs_lick, spout_mid_maps, spout_reliability, summary_figures_1021, trial_neural_vs_exec,
  within_lick_correction, within_spout_deviation, within_spout_mixed.
  - These were made to refuse on purpose (fourth review R5: fixed default names).
  - Under the profile each one stops before writing, so the render fails loudly rather than contaminating anything.
  - They are NOT fixed in this batch: 47 per-script output routings are not a small change, and the output layout
    (`<root>/<tag>/...`) is a decision for the coordinator. Proposal in the report.
- **REDO, possible untagged writer, 22 scripts** (heuristic): cohort2_qc/{attempt_effort, cohort1_subset_headline,
  quiet_f0_noise, quiet_lick_lfo, quiet_task_events, quiet_task_events_followup}, coupling_vigour_sm,
  decoder_retention_sessions, iti_first_lick, jaw_only_incomplete_figs, lick_correction, lick_firstlick_extract,
  lick_template_match, lick_vigor_perspout, movement_encoding_session, movement_position_angle, precue_class_matched,
  recovery_timing, rest_migration/rest_baseline_epoch_drift, restw_strict_check, wfield_local/restnl_mask, and
  archive/iti_nmatch_check.
  - Under a tagged run, `writeguard.assert_writable` refuses any share write outside `/<tag>/`, so these too fail
    loudly at write time if they really are unrouted. Each needs a look; some write keyed stores or take `--out`.
- **REDO, missing file:** `scripts/iti_nmatch_check.py`. It now lives in `scripts/archive/`.
- **NO rows:** 100 flagged, not fixed (no re-render planned).

**2. Which code a render imports.** The env's editable install (`pip install -e .`) points at the MAIN checkout. A
render imports the wf_rebuild worktree only when the worktree is first on sys.path:
- **Always set `PYTHONPATH=C:/Users/SabatiniLab/Github/wf_rebuild`.**
  - `python scripts/x.py` puts `scripts/` (not the cwd) on sys.path, so cwd = wf_rebuild alone is NOT enough for
    scripts run by path.
  - `python -m wfield_local.x` from the wf_rebuild root works without it (cwd first on sys.path), but PYTHONPATH covers
    both.
- The profile records it as `code_root: C:/Users/SabatiniLab/Github/wf_rebuild`.
- `render_profile --check` prints `wfield_local: <file>` and fails (exit 1) when a rebuild-root profile is active and
  the imported package is neither in code_root nor another LINKED WORKTREE of the same repository. The MAIN checkout
  (a `.git` directory) is always refused; test / review worktrees of the branch are accepted.
- Limitation: a command run on MAIN's code never reaches this check. Main has no render_profile / rebuild_root, so it
  would silently ignore the profile's env and write to the default locations. PYTHONPATH on every render command is
  the guard.
- Test added.

**Test record.** Full suite at d394dcbc, 6 shards, from the repo root with PYTHONPATH = the worktree: 4,615 passed,
179 skipped, 0 failed.

### 8.38 Routing the refusing REDO scripts and the render-runner fixes (coordinator 11-04)

**Layout.**
- A routed script's default output dir gains the run tag as a subfolder. Its readers resolve the same helper:
  - `run_mode.encoding_out_root(run)`: `<imaging_encoding>/<tag>`, the root the encoders write to (`combined_encoding`,
    `frozen_encoder.out_root`);
  - `lp_licks.output_dir`: `grant_figures/<tag>/epoch`;
  - `run_mode.subdir` / `routed_output`: `<dir>/<tag>`; attempt_coding_pp uses the LP-family tag;
  - the ITI family uses `il.OUT` / `il.OUT_W`.
- Run-independent behaviour tables are read untagged from `run_mode.behaviour_qc_dir()` (seeded): jaw / nose onsets,
  precue_contacts, first_lick_latency_late_licked_trials, position_block_summary.

**Commits:**
- 41cfe67e: init's scoped seed allow-list.
- f31b72b1: `python -m rebuild_root` runs in the imported module.
- 38f81c0c: camera_coverage_tables produces the coverage table inside the root, and a missing table is loud.
- b97a39f9: family 1, the encoder family (29 scripts).
- f49709ca + c22507df: trial_attempts (REDO).
  - It runs on the run's camera classes, and its denominator is the working period.
  - Its tables live in `run_mode.attempts_dir()`: `qc/<selection tag>/`, shared by every arm and block.
  - Every reader goes through `lp_licks.attempts_path()`. The pre-rebuild table is dropped from the seeds.
- a6e4e512: manifest. compare_map_tests and imaging_baseline_compare are NO; trial_attempts is REDO.
- 74986bd6: family 2, fr_* (4 scripts).
- f8e690b9: family 3 (layer2_*, summary_figures_1021, within_lick_correction) and the upstream allsession_encoder.
- 934f9ccf: the other 10 refusers.
- 09290e7f: learned_compensation (REDO, whole script) and misdirected_lick_counts (REDO). The counts depend on the
  engagement gate, so a camera run now produces its own, on the working period.
- 5cc29488: the 22-writer triage.
  - Real ones fixed:
    - the quiet family root (it used the labcams INPUT tree);
    - cohort1_subset_headline;
    - rest_baseline_epoch_drift;
    - the restnl_mask reader.
  - False positives are routed upstream.
  - iti_nmatch_check is NO (retired ROI-v3 one-off). Fig 10.1's producer is corrected to iti_licks.py.

**Pose-tree writers (621c81de).** These four now write, under a rebuild root, to
`<rebuild_root>/session_poses_derived/<tag>/<script>/` through `run_mode.session_poses_out`:
lick_template_match, movement_encoding_session, movement_position_angle, jaw_only_incomplete_figs.
- Per-session outputs go into `<...>/<script>/<session>/`.
- Without a rebuild root they keep their old place.
- Pose INPUTS are still read in place.
- No downstream REDO script reads their outputs (grep: none).

**Render order constraints:**
1. `rebuild_root init`. Re-run it for the new seeds.
2. RESTnl baselines.
3. camera_coverage_tables.
4. trial_attempts. It must run before:
   - the LP arm, attempt_coding and attempt_maps;
   - combined_encoding with a non-reference gate;
   - the mixed models;
   - misdirected_lick_counts, then misdirected_lick_decode;
   - every attempts reader.
5. within_lick_correction, then learned_compensation run, then prelaunch_aim_tracking, then face_prelaunch_aim.
6. attempt_coding with `--out <imaging_encoding>/attempt_coding_pp` before its readers.

**Tests.**
- `tests/test_routed_redo_scripts.py`:
  - every routed script has no require_default and uses a routing helper;
  - the encoding root equals the encoder's tagged root;
  - every attempts reader reads this run's table (LP blocks included);
  - the triage fixes.
- `tests/test_trial_attempts.py`: camera classes and the working denominator.
- `tests/test_coverage_routing.py`.
- `tests/test_rebuild_root.py`: init under the fence, and the CLI run as `__main__`.

**Test record.** Full suite, 6 shards, from the repo root with PYTHONPATH = the worktree:
- at 5cc29488: 4,685 passed, 179 skipped, 0 failed;
- at 621c81de: 4,690 passed, 179 skipped, 0 failed.

### 8.39 Page figure scripts into the repo (RECONCILIATION section 7; coordinator 11-04)

**Sources.** The scratchpad copies backed up at
`M:/MICROSCOPE/Priya/Widefield/encoding_goal_page_src/scratchpad_scripts_backup_20261104/S1_cdd1e016/`, and the PI
source `M:/MICROSCOPE/Priya/Widefield/presentations/_source_PI_overview_2026-11-02/`.

**Where they live.** All go into `scripts/rebuild_figs/`. That is the folder of the recreated page scripts, and it
already holds fig_goal_summary, fig_goal_epochs, fig_behaviour_hits, fig_decoding_sessions, fig_frozen_refit_pooled,
fig_state_position_sessions and fig_state_ps94_diagnostic.

**The `_common` contract.**
- `--out` / `--plot-only`.
- Figdata: `<out>/data/<name>_figdata.pkl`.
- The default `--out` is now run-tagged: `<rebuild_root>/rebuild_figs/<tag>/`.

Every script keeps its logic (computation, seeds, layout, styling); only paths, args and figdata change.

**Group 1 (1281b799), goal-page Layer 2 / pooled:**
- fig_layer2: Figs 5 / 6.
- fig_rsplit: Fig 5b. The summary CSV now goes to `--out`; it was written into the input folder.
- fig_pooled: 1p / 2c / 4p / 5p / 5c / 6p / 10p / 12p. The original body is wrapped in `draw()`; the figdata is the
  input tables, so `--plot-only` re-runs the same seeded bootstraps.
- Inputs: this run's `allsession_encoder_<tag>`, encoder runs (`run_mode.encoding_out_root`, `frozen_encoder.out_root`),
  `attempt_coding.out_dir()` and lp_encoder_decompose qc.

**Group 2 (7607f07a), softmax / residual:**
- fig_softmax: 9 / 9b / 10.
- fig_softmax_stats: 9c.
- fig_resid_per_animal: 8.
- fig_resid16: 7c / 7d.
- fig_resid_spout: 7. It shares fig_resid16.panel, which the originals duplicated.
- Inputs: this run's binned-contest and frozen-encoder runs.

**Group 3 (9a3f578d), PI deck / imaging + kinematics pages:**
- fig_farcontra_attempts_sessions:
  - attempt rate and completion come from THIS run's `trial_attempts_sessions.csv` (camera classes);
  - the cam1 incomplete table, a pose output, is read in place through the resolver.
- fig_pose_validation_sessions: pose3d and v36 lick inputs are read in place through the resolver.
- fig_decoding_sessions: its default input is now slowband_decode's tagged output.
- fig_behaviour_hits: its input is resolved through `behavior_out`.

**Manifest.** 43 rows now carry `producing_script` = `scripts/rebuild_figs/<x>.py` and the exact `command`. These
include the PI-deck and kinematics-page duplicates. Verdicts are unchanged: softmax 9 / 9b / 9c and pose_validation
stay NO.

**Smoke tests** (`tests/test_page_fig_scripts.py`, 32):
- no hard-coded machine / adopted-run / scratch path in code;
- the import and `--help` contract;
- `--plot-only` draws from synthetic figdata (layer2, rsplit, softmax_stats, resid16, resid_spout);
- the default `--out` is run-tagged;
- inputs follow the run.

**Commands.** Run each under the render profile env with `PYTHONPATH=<render worktree>`, from the render worktree:

    python scripts/rebuild_figs/fig_layer2.py
    python scripts/rebuild_figs/fig_rsplit.py
    python scripts/rebuild_figs/fig_pooled.py
    python scripts/rebuild_figs/fig_softmax.py
    python scripts/rebuild_figs/fig_softmax_stats.py
    python scripts/rebuild_figs/fig_resid_per_animal.py
    python scripts/rebuild_figs/fig_resid16.py --bin 16      (and --bin 32 for the 32 px page rows)
    python scripts/rebuild_figs/fig_resid_spout.py --spout far_contra --bin 16
    python scripts/rebuild_figs/fig_farcontra_attempts_sessions.py
    python scripts/rebuild_figs/fig_pose_validation_sessions.py

Prerequisites (render order):
- layer2 / rsplit / pooled: allsession_encoder, layer2_matched_n, layer2_random_splits, attempt_coding,
  lp_encoder_decompose (joint lick), frozen_residual_trend, and the binned-contest runs `_bins{,_bw0.5,_bw2}`.
- resid*: frozen_residual_significance at the matching `--bin`.
- farcontra: trial_attempts.

**Test record.** Full suite at 9a3f578d, 6 shards, from the repo root with PYTHONPATH = the worktree: 4,722 passed,
179 skipped, 0 failed.

### 8.40 Render blocker: content-keyed stores refused by the tag rule (7b5d3334)

**Symptom** (render at 621c81de). Every arm of `epoch_grant_figures` failed at the first collect with "5c PS92 ...
WriteGuardError: refusing a <tag> run's write outside a <tag> directory". `frozen_models._write` calls
`assert_writable(d.parent)`. Under the profile the store is `<rebuild_root>/frozen_models`, a share path that is
neither under `/<tag>/` nor a registered data store.

**Fix.** The write guard's tag rule now also accepts `writeguard.in_keyed_store(c)`. The stores are computed every time
from their definitions:
- `WIDEFIELD_FROZEN_MODEL_DIR` (incl. lp_final_render's `frozen_models_lpfinal`) and `frozen_models.local_dir()`;
- the session cache;
- every `rebuild.stores` entry;
- `WFIELD_INPUT_CACHE` when it lies under the root;
- the RESTnl caches (`<session cache>_restnl*`).

**Audit of the write sites.**
- Only `frozen_models._write` calls the guard among the stores.
- session_cache, map_collect (in the session cache), result_cache (LP licks, in the session cache) and the RESTnl cache
  do not call the guard; the fence allows them inside the root.
- The attempts tables and the coverage tables are registered data files (8.38).

**Test** (`test_every_content_keyed_store_is_writable_in_a_tagged_rebuild_run`).
- It runs in a subprocess under the applied profile, with a share-like temp rebuild root (`.../MICROSCOPE/Priya/...`).
- It calls the real write paths: `frozen_models.load_or_fit`, `session_cache.cached`, `result_cache.cached`, the RESTnl
  cache write, and the lpfinal store path. All are allowed.
- An untagged `grant_figures/epoch` write is still refused.
- On the old guard the test fails with the render's error.

**Test record.** Full suite at 7b5d3334, 6 shards, from the repo root with PYTHONPATH = the worktree: 4,723 passed,
179 skipped, 0 failed.

### 8.41 Stop rule: jaw_only counts toward the 6 (Priya 11-04; c3cb3c0c)

**Rule (production, in_working).**
- A stop run = >= 6 consecutive camera trials, each no_response OR jaw_only, starting >= 300 s after the first cue.
- Every trial of the run is stopped, jaw_only included.
- Unknown trials are skipped; a contact / incomplete trial ends a run.
- An earlier qualifying run is early_run (all of its trials).
- in_working_jaw keeps the 11-03 rule (>= 6 no_response; a jaw_only trial at or before the end ends the run, after it it
  bridges without counting).

**Code.**
- `classify_session(jaw_counts=None)` replaces `jaw_bridges_all`; None = `trial_classes.jaw_counts_in_stop()` (True
  unless jaw_only is in the working selection).
- The tag part is `stopRun6j` under production (`stopRun6` under in_working_jaw), so no cache, store or attempts table of
  the old rule is reused. The `cue_classes_for` key carries the flag. The profile's expected tags were updated.
- trial_attempts: stopped trials (jaw_only included) leave the attempt-rate denominator; cls_raw keeps the trial class.
- DECISIONS: both stop-rule bullets rewritten.

**Tests.**
- J,NR,J,NR,NR,NR at >= 300 s gives 6 stopped; a lone J stays jaw_only; the 299.9 / 300.0 s edge with a jaw-first run.
- An unknown inside a jaw-bearing run is skipped.
- The in_working_jaw rule is unchanged.
- The tag strings.
- The attempts denominator / cls_raw.

**Old vs new on the 120 curated sessions** (under the profile; script and table in the session scratchpad,
`jaw_count_compare2.py`):
- Stop runs 177 -> 283; stopped trials 3,267 -> 5,944, of which 2,353 are jaw_only (+324 no_response trials from runs
  that reach 6 only with jaw counting).
- early_run: 0 under both rules. Working counts are unchanged in every session.
- Acute stopped: PS92 417 -> 708, PS93 396 -> 661, PS94 543 -> 1,012, PS95 6 -> 60.
- August pre stopped: 10 -> 17, 30 -> 53, 188 -> 310, 214 -> 425.

**Test record.** Related files 144 passed. Full suite: not run at c3cb3c0c; the next record (on code that holds it) is in 8.43.

### 8.42 Split area basis in the encoder; rebuild-aware O2 bundles (fb3b6304, a6a29dd7, a7aec872)

**Imaging input cache (fb3b6304).** The imaging entry is keyed on `_load_imaging` + `hemo_dir` source and beta_maps
(plus the sources' mtimes). Before, it hashed all of combined_encoding.py and every COMBINED_ENCODING_* / OROFACIAL_*
switch, so any edit discarded ~1 GB per session of share reads. That let the O2 bundle's imaging be pre-warmed while the
code changed: 107 sessions in 23.4 min (4 jobs).

**Split basis (a6a29dd7; Priya 11-04: every Allen-ROI analysis uses the iti_licks split basis).**
- `iti_licks.roi_masks` moved verbatim to `wfield_local/area_basis.py` (bit-identical on PS92_0806 / 0807).
- iti_licks delegates to it; the (fraction, rim) per ROI version is `area_basis.ROI_SPEC`; iti_licks keeps the file
  names.
- `session_area_masks` returns compact pixel indices for:
  - split (primary): every area except cortex and whole MOs / MOp, + ALM, tjM1, MOs_rest, MOp_rest;
  - plain (robustness): ALM / tjM1 out, whole MOs / MOp in;
  - the v8 cortex.
- combined_encoding: one region row per area of either basis, with a `basis` column. The same areas go into Ra (peth /
  sse / sse_trial), + `cortex_stat` (the old stat-mask cortex) + `cortex` (v8). `res["area_basis"]` names the basis
  (splitr8).
- The masks are built at BUNDLE time (the v8 rim reads `beta_maps.brain_mask()` from the share) and stored in each
  inputs pickle. run_session on bundled inputs refuses inputs without them.
- No folder suffix: about ten page scripts hard-code the encoder folder names. The stopRun6j tag separates the outputs.
- Mask sizes on PS92_0806 (pixels): cortex 166,974 -> 172,322; SSp-m_left 2,486 -> 3,271; MOs_left 15,613 -> 16,381;
  MOp_right 8,190 -> 8,688; SSp-bfd and VISp unchanged.

**O2 bundles (a7aec872; coordinator 11-04).** `scripts/o2_encoding.py --profile <profile>` applies the profile before
any import.
- Location: the bundle goes to `<rebuild root>/o2/<name>`, a registered data area of the root.
- env.json = `wfield_local.o2_bundle.settings()`: the full profile env + COMBINED_ENCODING_* / OROFACIAL_*. Local store
  paths and per-submit options are excluded; files go to tables/; a directory value is refused.
- bundle.json = run identity (run tag, selection, area basis), full commit, attempts table (copied to tables/ for
  provenance; the gate is inside each Behaviour).
- Runner: clears the O2 shell's settings, applies env.json before importing wfield_local, and refuses unless the bundled
  code recomputes the identity.
- code/ = `git archive` of the recorded commit, refused when wfield_local / configs are dirty.
- check: audits the bundled run for share reads (sys audit hook), needs max |dr2| < 1e-4 and equal row counts.
- collect: refuses a bundle of another run, and writes to the routed `combined_encoding.out_dir`.
- COMMANDS.md: the 14 runs of `FULL_RUNS` (fr_A_tr dropped: fr_A carries the FR trial-to-trial outputs in this code).

**Tests.** test_area_basis (sets, disjoint rests, compact masks, refusal, delegation); test_o2_encoding_env (env round
trip under the profile, tag refusal, runner order, 14 runs, root location); test_input_cache (imaging key).

### 8.43 Every camera trial class from the O2 encoder; bundle fixes (branch rebuild/encoder-classes, 38436a2b..HEAD)

Goal (Priya 11-04): report every camera trial class from the O2 encoder outputs: hit (contact), incomplete, jaw_only,
no_try, stopped, after_end, early_run, unknown, plus cls_raw and the close-mid `jaw_judged` flag. Before this change
run_session kept only the engaged trials (E). It saved per-position averages and summed SSE with no trial ids, so a
non-engaged class was never scored and contact could not be split from incomplete.

**`--profile` wiped the encoder settings (38436a2b).** `render_profile.apply` clears every COMBINED_ENCODING_* variable
the profile does not set. The first enc_rebuild_20261104 build therefore lost LICKS / TONGUE / EXEC and the EXEC3D /
TARGET3D / BINS_CALIB tables (OROFACIAL_GATE survived: not an owned prefix).
- `o2_encoding.apply_profile` re-applies the caller's COMBINED_ENCODING_* / OROFACIAL_* (and WFIELD_INPUT_CACHE, never
  recorded) on top of the profile.
- An arm-owned setting left in the shell (COMBINED_ENCODING_DRIFT_MIN_PERIOD_S) is refused: it would re-tag the run.
- Test: env.json carries the six settings and the tables.
- The rebuilt bundle's env.json has all six.
- Other `--profile` users: only `rebuild_root init / show` calls `apply`, and it reads no encoder setting.
  frozen_encoder / allsession_encoder / frozen_relabel take no `--profile`. They get the profile from
  `render_profile --shell` eval, whose clear line also unsets COMBINED_ENCODING_*. ENCODER_RENDER_COMMANDS.md already
  says to export the extra env AFTER the eval; a launcher that exports first loses it the same way.

**Every cue trial scored (75f6a40c).** The fit is unchanged: training rows = E's frames, the same folds, penalties and
models.
- Scored-only trials = every cue trial not in E. Their frames are built like E's (strobe - 0.5 .. stop + 1 s, concatenated
  in time) and never fitted.
- Their design rows get the fit rows' transforms: z statistics, the orthogonalisation betas (`_residualise(extra=...)`),
  and the slow-drift coefficients (Q extended as S_x R^-1; frames after the last fitted trial extrapolate the cosine
  basis).
- They are predicted by each model's full-data beta.
- Position kernels are NOT engagement-gated: raw_blocks builds cue / arrival / position over all trials, so these trials
  carry their real position, cue, lick, contact and movement regressors.
- Not scorable on these trials: shuffled blocks (`shuf`, a circular shift of the fit rows) and label-permuted / relabelled
  blocks (shufpos-*, shufside, shufdist, droppos-*), whose labels are drawn over the fitted trials only. Their models give
  NaN there (`tr_scorable`). In the main time-resolved set that is shuf:task / instructed / uninstructed / pos_enl /
  pos_cue / exec_angle / position, shufside:position and shufdist:position. Every raw / orth model is scored: full,
  alone:*, drop:*, fr:*, nokin:*.

**`sessions/<label>_trials.npz`.** The session npz / regions / json are untouched; the json only gains
`n_trials_scored_only` and `trials_mode`.
- Trial table `t_*`: trial_id, pos, engaged, fitted, fold (-1 = scored only), cue_s, stop_s, first_lick_s,
  timing_source, grooming_suspect, lp_first_lick_s, daq_first_contact_s, and cue_row / lick_row / lplick_row / row_lo /
  row_hi (the frame rows of each trial).
- `sse_bins_<cue|lick>` (models, trials, 0.5 s bins, areas), `sst_bins_*` (about the fit-row mean, as `sst`), `nfr_bins_*`.
- Mode full adds `frames` (parts, rows, areas): data, full, task, instructed, uninstructed, pos (full-data betas) and
  "cv" (held-out on fitted rows), for fitted rows then scored-only rows, in Ra space (all areas + cortex_stat + cortex).
  This is the minimal set that rebuilds `peth` / `peth_lick` and the full model's `sse` / `sse_trial` per class at any
  alignment (frame-level, not aligned: one copy serves both alignments). Other models: binned per-trial SSE only (their
  per-class trial-to-trial R2 would need every model's prediction, about 3x the size).
- LICK TIMING (Priya 11-04).
  - Close spouts use the first DAQ contact (cue + 70 ms .. min(cue + max_rt, stop)) when the first LP onset is missing or
    more than `LP_DAQ_TOL_S` = 50 ms after it (timing_source daq).
  - Far spouts keep the LP onset and set grooming_suspect when it is late; with no LP onset they use the contact.
  - `lplick_row` is the session npz's LP-only alignment, so peth_lick / sse_lick stay exactly reproducible.
- Classes are NOT in the O2 run. `wfield_local/encoder_trials.py` joins cls / cls_raw / attempted / canonical by
  (label, trial_id) from the run's trial_attempts.csv (`jaw=True` adds jaw_judged from `trial_classes.session_trials`,
  a share read). It returns per-class PETHs (`class_peth`), binned SSE / SST per model (`class_sse`), the full model
  frame-resolved with trial-to-trial R2 (`class_full_fine`), and a table (`class_summary`).
- `COMBINED_ENCODING_TRIALS` = off | sse | full (float16 frames) | full32. Unset = auto:
  - full for the unsubset run (TAG main);
  - sse when a subset run has a time-resolved model that main does not compute identically (another ORDER, nokin / bins,
    fr);
  - off when all its time-resolved models are main's (trmodels, drop, droppos, shufpos, nodir).

**Engagement from a shipped table (4f44ed35; coordinator 11-04).** A later class-rule change (the 70 ms contact floor,
the stop-rule guard) must not force a 1.5 h input rebuild.
- Submit option ATTEMPTS=tables/trial_attempts.csv -> COMBINED_ENCODING_ATTEMPTS. `run_session` recomputes
  trials.engaged with engagement_keep's attempted rule, refusing a table that misses any trial of the session.
- run.json and the session json record the table and its sha256.
- refresh-code ships the run's current table (an earlier, different one is kept as trial_attempts_<sha12>.csv) and records
  its sha256 in bundle.json. New bundles' submit lines carry ATTEMPTS.
- With the baked table it reproduces the flags exactly: PS93_0814 509 / 575, PS93_0818 480 / 554 (table sha256
  1dee914250a0...), and a synthetic run gives identical outputs.
- Nothing else in the inputs depends on the classes: trial windows (`trial_windows.trial_bounds`: DAQ + log), licks /
  pose / cam1 events (LP run), DAQ contacts, imaging and the area masks are class-free.

**O2 bundle.**
- `refresh-code --name <n>` (or `--bundle <dir>`) replaces only code/ (+ enc_runner.py, the commit in COMMANDS.md); inputs
  are never touched.
- It refuses a dirty tree, and refuses unless the NEW code, in a clean subprocess with only env.json, computes the recorded
  identity.
- bundle.json gets the new commit, `commit_inputs` and `code_refreshes`.
- Not run on the real bundle.
- `check` also compares the two runs' per-trial outputs (NaN patterns exactly, numbers < 1e-4 relative; `compare_trials`).

**Identical outputs (SMOKE model set = full + every TR model; original = 571b3955's combined_encoding.py, same inputs,
4 BLAS threads).** Every array of the session npz (maps 265 x 48,385, peth, peth_lick, sse / sst / sse_trial / sst_trial,
cue and lick), the regions csv (14,280 rows) and every shared json key are BIT-IDENTICAL:
- PS93_0814, on scratch inputs built with the run_bundle.sh env after the profile, AND on the rebuilt bundle's inputs;
- PS93_0818, on the rebuilt bundle's inputs and env.json;
- including the final code (0223f7a8, default auto mode).
Full cvR2 (cortex, all frames): 0.5264 / 0.4054 both ways.

**Sizes (float16 frames, npz compressed).**
- Per session: full = 242 MB (0818) .. 279 MB (0814); sse = 60-64 MB with the 19 main time-resolved models (about 3.2 MB
  per model).
- 107 sessions x the auto modes:

  | TAG | auto mode | per session | 107 sessions |
  | --- | --- | --- | --- |
  | main | full | ~260 MB | ~28 GB |
  | bins_w1 / bins_w05 / bins_w2 | sse (11 models) | ~37 MB | ~4 GB each, ~12 GB |
  | musall_shuf | sse (8 models) | ~27 MB | ~3 GB |
  | fr_A / fr_B / fr_S / fr_D | sse (7 models) | ~23 MB | ~2.5 GB each, ~10 GB |
  | droppos, shufpos, nodir, drop, trmodels | off (main holds their time-resolved models) | 0 | 0 |
  | **total** | | | **~53 GB** |

- For comparison, full in all 14 TAGs would be about 400 GB.

**Memory (peak commit, SMOKE model set, bundle inputs).**
- PS93_0818: original 25.6 GiB, new 26.2 GiB.
- PS93_0814: original 28.3 GiB, new 28.9 GiB.
- The change adds +0.6 GiB; the frames are built one part at a time into a preallocated float16 array.
- The BASELINE is already 28 GiB on a 575-trial session, close to the sbatch's 32G. Larger sessions or the full main model
  set may need --mem=40G (COMMANDS.md already says to resubmit OOM tasks with 64G).

**Smoke per-class numbers (full model, cortex, -3..+3 s; scored-only trials by the full-data model).** cvR2 is against
the fit-row mean, as `sst`. A scored-only class whose mean differs from the session's (stopped, late in the session) can
score high from that offset alone. Trial-to-trial R2 removes each position's class mean.

PS93_0814 (pre):

| cls | n | fitted | cue cvR2 | cue trial R2 | lick cvR2 |
| --- | --- | --- | --- | --- | --- |
| hit | 485 | 485 | 0.616 | 0.442 | 0.624 (2 on the DAQ contact; 1 far grooming_suspect) |
| incomplete | 24 | 24 | 0.511 | 0.571 | 0.476 |
| jaw_only | 6 | 0 | 0.347 | -0.066 | - |
| no_try | 7 | 0 | 0.445 | 0.557 | - |
| stopped | 53 | 0 | 0.714 | 0.705 | - |

- jaw_judged False: 88 hit, 4 no_try, 6 stopped.

PS93_0818 (acute):

| cls | n | fitted | cue cvR2 | cue trial R2 | lick cvR2 |
| --- | --- | --- | --- | --- | --- |
| hit | 381 | 381 | 0.553 | 0.412 | 0.562 |
| incomplete | 99 | 99 | 0.386 | 0.390 | 0.448 |
| jaw_only | 19 | 0 | 0.609 | 0.605 | - (1 trial with an LP onset) |
| no_try | 29 | 0 | 0.535 | 0.554 | - |
| stopped | 26 | 0 | 0.609 | 0.628 | - |

- 6 hit trials are lick-aligned on the DAQ contact (close spouts); grooming_suspect 0.
- jaw_judged False: 84 hit, 9 no_try, 4 stopped.

after_end / early_run / unknown do not occur in these two sessions. Every class with a finite cue is scored. A class
with no lick has no lick-aligned trace.

**Tests.**
- New: tests/test_encoder_trials.py (11: identity with per-trial outputs on, every trial scored and never trained,
  reader rebuilds peth / peth_lick / sse / sse_trial from engaged classes, lick timing rule, float16 bound, class join /
  summary, helper exactness, mode rule, re-derived engagement x3).
- tests/test_o2_encoding_env.py: +6 (profile keeps settings, refresh-code replaces only code / refuses identity change
  and a dirty tree / ships the attempts table, check's compare_trials, runner ATTEMPTS).
- Related suites: 189 passed, 1 skipped.
- Full suite, 3 shards on 75f6a40c..c08c7396: 4,763 passed, 172 skipped, 0 failed.
- After the lick-timing rule (0223f7a8): test_encoder_trials + test_o2_encoding_env 24 / 24.

**Lick-free enl window (f13c608d, aa844490; Priya 11-04).**
- "enl" is now, per engaged trial, the ENL decoders' own lick-free window. The search is
  `precue_window_start` / `lickfree_window`, moved verbatim to `wfield_local/enl_window.py` and re-exported by
  `locanmf_position_decoder` (one definition; the decoder module's sklearn / matplotlib / registry imports kept it off O2).
  The fixed window ending at the cue is kept if it is clean; otherwise the window slides to the latest clean gap after the
  strobe. Trials with neither are left out and counted in json `enl_window`.
- Length / source = the decoders' spec: `lp_licks.enl_window_s()` = 1.0 s under LP, else `decode.precue_post_s` = 2.0 s.
  `COMBINED_ENCODING_ENL_SOURCE=lp|daq` overrides it. The profile clears WIDEFIELD_LICK_SOURCE and the bundle env does
  not set it, so the bundle resolves to DAQ unless this is set.
- Licks seen by the search are only what the pickle carries:
  - lp = licks.csv + licks_all pre-cue / ITI / post_extra + cam1-only, i.e. `lp_lick_times` UNEDITED. The decoders'
    `edit_train` moves attempts-table licks; it touches 8 / 3,532 (PS93_0814) and 4 / 3,198 (PS93_0818) ENL licks, and
    unedited keeps the inputs class-free.
  - daq = the whole-window contacts (`dlc_frames.lick_onsets`).
- The old window stays as "enl_fixed2s" (max(strobe, cue - 2 s) .. cue, ungated).
- Pickles before 11-04 carry no pre-cue licks (licks.csv starts at the cue; the contact table at cue + 70 ms): 0 cam4 /
  0 DAQ ENL licks in the 11-04 PS93_0814 pickle vs 3,524 in the LP train. There "enl" falls back to the fixed window and
  the json says so.
- LICKS_ALL "score" = load the context licks for scoring only (no kernels, no folder suffix).
- Readers of window "enl" whose meaning changes:
  - scripts/combined_encoding_figures.py: WINDOWS, and a WIN_DESC caption "pre-cue 2 s (ENL)" that is now wrong;
  - scripts/encoding_key_figures.py:42 (cortex enl full vs drop:position);
  - scripts/encoding_mixed.py:29;
  - scripts/fr_encoding_summary.py:72 (default windows).
  - Unchanged: the resp / respdev "|enl" maps (strobe + 0.3 .. cue, not the window; position_coding_trend_recovery), and
    frozen_encoder / allsession_encoder (own windows).

**LICKS_ALL=split = production (Priya 11-04, option B; f13c608d, 2064c4f4, 377f160e).**
- `o2_encoding bundle --licks-all split` is now the default; env.json records COMBINED_ENCODING_LICKS_ALL. The pickles carry
  ctx (pre_cue / iti / post_extra) and the whole-window contacts, and keep the cue-table contacts.
- Submit-time `LICKS_ALL=none` (new TAG `main_nolicksall`, TRIALS=sse) drops the kernels from the same pickles. The design
  picks the contacts at design time, so this path is exactly the old model. collect routes by the run's own LICKS_ALL.
- Folders: split -> `<order>_licksall-split`, main_nolicksall -> `<order>`. The page scripts that hard-code the default
  folder would read the ROBUSTNESS run unless pointed at the split folder: a decision before collect.
- licks_all source: `_context_licks` (combined_encoding.py `_context_licks`, `sd / "licks_all.csv"`, sd =
  `orofacial_outputs.session_dir(rv, run, ...)`) = `LP_analysis/lp_r4_heatmap_20261013__clean_v36/sessions/<animal>/<date>/
  licks_all.csv`. Present for all 107 bundle sessions.
- Every lick in exactly ONE kernel (`split_licks`):
  - post-cue = licks.csv (every row post-cue; the 0-70 ms licks, 14 / 4 on PS93_0814 / 0818, are here: the 70 ms floor is
    for counting contacts, not modelling) + the pre-cue / ITI-table rows whose onset is post-cue (`post_extra`, 11 per
    session, >= 60 ms from any licks.csv lick, before this in NO kernel) + cam1-only post-cue;
  - pre-cue = licks_all pre_cue;
  - ITI = licks_all iti.
  - cam1-only onsets are routed by time. Those outside their own trial that lie within CAM1_DUP_TOL_S = 50 ms of a cam4
    pre-cue / ITI lick are dropped as duplicates (cam4_matched only compares the same trial_id): 13 / 22 (PS93_0814),
    6 / 11 (PS93_0818).
  - DAQ contacts: one contact kernel over the whole trial windows (strobe - 0.5 .. stop + 3 s, raw channel; covers every
    fit row), incl. 0-70 ms and pre-cue contacts.
- Three variables: licks = cued post-cue licks + contacts (instructed); licks_pre, licks_iti (UNINSTRUCTED); all penalty
  group lick. New model drop:licks_pre+licks_iti, plus the generic drop / alone / shuf / task+ of each. The nodir / nokin
  bases keep the spontaneous lick-timing kernels.
- Caveat (accepted): the tongue / jaw traces stay instructed over the whole trial.
- Stacked vs three variables is NOT identical: the uninstructed group is orthogonalised against instructed, so the ridge
  sees another basis (same alphas). Full-model region R2, max |d| = 0.0015 (PS93_0814) / 0.0008 (PS93_0818); full cvR2
  0.54037 vs 0.54041 and 0.43161 vs 0.43166.
- Group maps / peth parts follow the local GROUP_OF (uninstructed now includes the spontaneous licks).
- Readers that hard-code the lick variable:
  - scripts/combined_encoding_figures.py:31-33 (own VARS / GROUP_OF copies, no licks_pre / iti);
  - scripts/curbd_pilot.py:131-134 (`ce.GROUP_OF`, static);
  - frozen_encoder.split_blocks refuses LICKS_ALL (fine for O2; a frame render must not inherit LICKS_ALL);
  - every reader of drop:instructed / drop:uninstructed / alone:* / shuf:* (the group contents changed);
  - drop:licks = cued licks only.
- sbatch `--mem` is now a bundle option, default 48G (32G is below the measured peaks).

**Smokes (split-built scratch inputs; one at a time; enl = DAQ 2.0 s, the bundle default).**
- Unset path vs the original 571b3955 code: every non-enl output is BIT-IDENTICAL (maps 234 keys, every npz array, 12,240
  region rows, shared json keys) and enl_fixed2s = the old enl rows exactly (2,040 rows), on both sessions. The json
  changes `window_frames` and adds `enl_window`.
- Cortex, models from ONLY=drop:; unique = full - drop.

| | PS93_0814 nolicksall | PS93_0814 split | PS93_0818 nolicksall | PS93_0818 split |
| --- | --- | --- | --- | --- |
| full cvR2, all frames | 0.5264 | 0.5404 | 0.4054 | 0.4316 |
| full, enl (lick-free) / enl_fixed2s | 0.4593 / 0.4595 | 0.4709 / 0.4712 | 0.2439 / 0.2480 | 0.2816 / 0.2853 |
| drop:pos_enl, enl / fixed2s | 0.0085 / 0.0086 | 0.0019 / 0.0019 | 0.0029 / 0.0036 | 0.0064 / 0.0068 |
| unique licks (cued), all / enl | 0.0027 / -0.0009 | 0.0048 / -0.0029 | 0.0129 / 0.0018 | 0.0189 / 0.0037 |
| unique licks_pre, all / enl | - | 0.0077 / 0.0013 | - | 0.0132 / 0.0191 |
| unique licks_iti, all / enl | - | 0.0024 / -0.0009 | - | 0.0070 / 0.0065 |
| unique pre + iti, all / enl | - | 0.0091 / 0.0023 | - | 0.0171 / 0.0261 |
| DAQ window: fixed / slid / none | 501 / 8 / 0 of 509 | same | 468 / 10 / 2 of 480 | same |
| full, enl with ENL_SOURCE=lp (1.0 s) | 0.4723 | 0.4794 | 0.1536 | 0.1897 |
| drop:pos_enl, enl LP | 0.0072 | 0.0029 | 0.0044 | 0.0037 |
| unique licks_pre / pre + iti, enl LP | - | 0.0016 / 0.0023 | - | 0.0252 / 0.0242 |

- DAQ: the ENL resets on any contact, so the DAQ lick-free window is nearly the fixed one.
- LP 1.0 s: 408 fixed / 94 slid / 7 none (PS93_0814), 399 / 80 / 1 (PS93_0818). LP 2.0 s would drop 94 / 103 trials.
- Runtime split vs nolicksall: 11.0 vs 10.1 min, 9.6 vs 8.7 min (+~10 %).
- Peak commit split vs nolicksall: 31.5 vs 30.0 GiB, 28.7 vs 27.3 GiB (+1.4 GiB). The unset SMOKE + per-trial run peaked
  at 30.5 GiB (PS93_0814).

**Tests (this addendum).**
- tests/test_encoder_licks_all.py, 10 tests: exactly one kernel per lick, split variables vs unset from the same pickle,
  score mode, the enl window equals the decoders' search, the moved search re-exported, run_session enl rows, groups /
  penalties / drop models, ENL_SOURCE override, bundle default + runner override.
- test_combined_encoding_options::test_licks_all_modes now asserts the three-variable layout.
- Full suite, 3 shards on 377f160e: 4,762 passed, 173 skipped, 2 failed.
  - test_lesion_line_and_window: a hard-coded max_rt fallback; fixed in 148af7ec.
  - test_review_7ab::test_working_trials_stays_byte_identical_to_main: origin/main changed working_trials.py on 11-04
    (4a37cbc8), so the file differs from the rebuild base too. Not this branch.

**Coordinator decisions, 11-04 evening.**
- Memory: 48G accepted.
- Folders: in a TAGGED (rebuild) run, LICKS_ALL=split is the PLAIN folder (`movement_first`, plus each TAG's subset
  suffix), and the no-spontaneous-kernel run is `_nolicksall` (`licks_all_suffix`). collect routes every TAG by the run's
  own LICKS_ALL: main -> movement_first, main_nolicksall -> movement_first_nolicksall, the subset TAGs -> their usual
  names.
  - Untagged (legacy) runs keep the old names, so adopted folders keep their meaning.
  - Nothing is overwritten: the rebuild root has no combined_encoding outputs yet (imaging_encoding holds only qc/) and
    o2/enc_rebuild_20261104/out is empty.
  - The library default stays LICKS_ALL unset, so frozen_encoder and frame renders never inherit it. split_blocks refuses
    it, and the ENCODER_RENDER_COMMANDS env blocks (ENV-noFR / frA / frozen) do not set it. Production split comes from
    the bundle (`--licks-all split` default -> env.json).
- ENL source: LP 1.0 s lick-free is the production default (`ENL_SOURCE_DEFAULT = "lp"`, and `bundle --enl-source lp`
  writes it to env.json). daq and decoders stay as options.
- Readers to fix at render time (captions / variables):
  - scripts/combined_encoding_figures.py: WINDOWS / WIN_DESC "pre-cue 2 s (ENL)", and its own VARS / GROUP_OF without
    licks_pre / licks_iti;
  - scripts/encoding_key_figures.py:42;
  - scripts/encoding_mixed.py:29;
  - scripts/fr_encoding_summary.py:72;
  - scripts/curbd_pilot.py:131-134 (static `ce.GROUP_OF`).
- Related tests after these changes: 256 passed, 1 skipped (encoder, O2, profile, rebuild root, classes, decoder / ENL
  suites).

### 8.44 Cue + 70 ms, response window, stop rule A2, LP first-lick timing (branch rebuild/cue70; Priya 11-04)

Branch `rebuild/cue70` from 571b3955 (worktree `wf_rebuild_cue70`). Commits: 9f0d6efe (rules 1 + 2), 395a9124 (A2),
0988a1cf (LP timing), 5463a0b6 (run-break hook), 3613cf0b (stopped-time licks, excise). Written as 8.45 on the branch; renumbered 8.44 at the merge into
rebuild/cohort1-20261103 (merge order: 8.43 encoder-classes, 8.44 cue70, 8.45 / 8.46 enl-window). Nothing was rendered and no table on M: was rebuilt. Scratch (scripts, CSVs, logs):
`C:/Users/SABATI~1/AppData/Local/Temp/claude/C--Users-SabatiniLab/f082ee0c-b33c-49a8-a537-dcee3bb7c84c/scratchpad/cue70/`.

**Rule 1: cue .. cue + 70 ms is neither pre- nor post-cue** (widefield_pipeline docs/status/CUE70_CHECK_2026-11-04.md).
- One definition in `trial_classes`: `POSTCUE_START_MS` = 70 (= `CONTACT_FROM_CUE_S` x 1000, kept),
  `PRECUE_END_AFTER_CUE_MS` = 0, `EXCLUDED_CONTEXT` = "excluded_0_70". `pose_kinematics_demo` imports them;
  `daq_contacts_ms` and `orofacial_cohort._first_contacts` use the constant (same value, same floats).
- Camera licks are judged by PEAK (`t_ms`), never onset: onsets at 60-70 ms are real responses.
- `trial_classes._pre_cue_contacts` (window "trial" only): strobe .. the cue (was .. cue + 70 ms).
- `pose_kinematics_demo.precue_and_excluded_licks`: ONE detection pass, strobe .. cue + 70 ms (unchanged, so every lick
  and its features stay as they were), split on the peak by `split_precue_by_peak`: peak < cue -> `licks_precue.csv`,
  cue <= peak < cue + 70 -> `licks_excluded_0_70.csv` (new table; manifest `n_excluded_0_70_licks`; `load_session`
  key). `precue_licks` = the pre-cue part. Every peak still lands in exactly one table (docstring updated).
- `orofacial_cohort`: `precue_rate_hz` / `precue_trial_frac` numerator = `split_precue_by_peak(licks_precue.csv)`, so
  the v36 tables written before 11-04 (which still hold the 0-70 ms peaks) give the new numbers without a re-render.
  Denominator `_windows` pre = (0 - strobe) (was 70 - strobe); response = stop - 70 (unchanged). New column
  `n_excluded_0_70_licks`.
- `orofacial_outputs.lick_context(peak_ms, strobe, stop)`: by PEAK; CONTEXTS = pre_cue / excluded_0_70 / post_cue / iti.
  Before, it split on the ONSET at 0 while the tables split on the peak at 70. The old onset label is `onset_context`.
  `all_licks_session` writes both (`context` by peak, `onset_context`) and reads `licks_excluded_0_70.csv` when present.
- Encoder lists: `combined_encoding._context_licks` and `frozen_encoder` read `oo.encoder_context(A)` = `onset_context`
  when present, else `context` (a pre-11-04 licks_all, whose context IS the onset label). So a re-rendered licks_all
  keeps every lick in the encoder's pre-cue / ITI lists exactly as before; only the analysis label changes.
- `scripts/cam1_licks.py` events_all: `context` by `t_peak_ms` + `onset_context`. Deck caption (orofacial_deck) rewritten.

**Rule 2: the response window** (census CONTACT_DAQ_MISMATCH_20261104.md). Camera classes (`session_trials`, both
windows) count only cue + 70 ms (lick PEAK) .. min(cue + max_rt, stop) (`response_end_ms`; max_rt = `decode.max_rt_s`
= 3.5 s through `max_rt_ms()`, the DAQ engaged rule's source):
- cam4 licks by `t_ms`; cam1-only events and the cam1 "saw a tongue" rows by `t_peak_ms` (an untimed "saw" row keeps its
  call; an events file without `t_peak_ms` is refused);
- contacts = `n_daq_contacts` (cue + 70 .. stop) minus the DAQ contacts in (cue + max_rt, stop] (`_late_contacts`; no
  DAQ read for a trial whose stop is <= max_rt);
- jaw: a moved call counts only when its (re-)onset is <= the window end; a late-only movement = judged and still
  (no_response). The jaw calls themselves (`scripts/jaw_onsets.py`) are cue .. stop and unchanged; a moved call with no
  onset time keeps its call.
- The last-camera-lick end anchor follows the classes: a lick after max_rt no longer makes a lick trial, so it no longer
  sets the session's last lick (the coordinator's default, adopted). Stop runs follow from the classes.
- The new functions enter `CLASS_CODE_ROOTS`, so every camera-gated cache starts fresh.

**Rule 3: stop rule A2 "both sides" PRIMARY** (wf_attempt docs/status/rebuild_20261103/STOP_RULE_GUARD_CENSUS_R2_20261104.md).
- `classify_session(rule="both")` -> `_classify_guarded`. Runs = >= 6 no_response / jaw_only before the last camera
  lick, over the non-unknown sequence.
  - A run with no reference-spout trial (`REFERENCE_SPOUTS`) is kept (raw classes) when the nearest reference-spout
    trial before AND after it are both raw "contact"; otherwise it is stopped.
  - Runs with a reference trial are stopped.
  - Everything after the last camera lick (not unknown) is stopped, no guard, also with an uncovered video tail.
  - A qualifying run starting < 300 s keeps its raw classes: early_run is folded into no_response / jaw_only. It was
    NOT folded before: rule "run" still emits early_run (the cohort has 0 such runs).
  - Runs are not split at uncovered gaps (as r2). Under in_working_jaw, jaw_only ends a run.
- Arms (`WIDEFIELD_TRIAL_STOPPED` / `trial_selection.stopped`): both (default + profile), run (A0, the 11-04 morning
  rule, code path unchanged), long (A3, runs >= 11, no guard), any. Profile arms `stopA0` / `stopA3`.
- Tags: every camera rule name now ends in the class-window token `rt<max_rt ms>c<floor ms>` = `rt3500c70`.
  - Primary `...-cls-contact-incomplete-stopA2jexrt3500c70` (was `...-stopRun6j`).
  - A0 `stopRun6jrt3500c70`, A3 `stopA3n11jexrt3500c70`, any `stopAnyrt3500c70`, in_working_jaw `stopA2exrt3500c70`.
  - Profile expected tags updated.
- `cue_classes_for` passes positions to the guard. `trial_attempts` already passes positions + cue times.
- **Reproduction of r2.py**: `_classify_guarded` on r2's own `srg/trials.csv` (56,979 trials, 107 sessions; early_run
  folded). A2: 0 trials differ, 5,611 stopped (r2: 5,611). A3: 0 differ, 5,175 stopped. Script `repro_r2.py`.
- Run-break hook (for the pending "no licks in the stopped time" rule): `classify_session(..., breaks=bool[n])` ends a
  run at a True trial in every rule (default None = no change); `licks_in_spans(lo, hi, lick_s)` builds it. Not wired.

**Rule 4: LP first-lick timing for lick-aligned arms** (wf_attempt .../lp_missed_contact/LP_MISSED_CONTACT_20261104.md).
- `lp_licks.first_lick_timing(pos, lp_ms, contact_ms, stop_ms)`, `LP_DAQ_TOL_MS` = 50. Contact = first DAQ contact in
  cue + 70 .. min(max_rt, stop).
  - CLOSE spouts: LP onset missing or > contact + 50 -> the contact time, `timing_source` = daq.
  - FAR spouts: the LP time, `grooming_suspect` = True when > contact + 50. Classes unchanged.
- `_session_lp` uses it (an LP-missing close hit is now attempted, with the DAQ time). `_attempts` refuses a table
  without pos / stop_ms / contact. `first_lick_latency.session_latency` (so trial_attempts too) carries `first_lick_ms`,
  `timing_source`, `grooming_suspect`. `LP_VERSION` 2 and the tolerance enter `spec_token` (every LP cache key).
- Counts on the existing attempts table (`.../qc/sel-...-stopRun6j/trial_attempts.csv`, 57,048 trials; LP = combined,
  else cam4):
  - close-spout raw hits with DAQ timing: **55** (close_mid 34, close_ipsi 17, close_contra 4, incl. PS93_0810 t403,
    the close_contra paw trial), all LP-late, none LP-missing. Report: 56 + 1.
  - far `grooming_suspect`: **6** (far_mid 4, far_ipsi 2). Report: 7. The missed one is PS95_1005 t482: on the report's
    windows.npz cue clock its first contact is at ~70 ms; on the trial_bounds clock of the `contact` column it is
    under 70 ms, so the first contact used is 356 ms and the LP onset (156 ms) is not late.
  - The 2 missing close_ipsi trials are presumably the same clock difference (<= 14 ms); not checked one by one.
  - 17 more close trials get DAQ timing with LP missing; all are class unknown (not attempted, so never used).

**Class census, full cohort** (107 sessions of `labels107.txt`, 57,048 DAQ cues, under the profile; before = 571b3955
code, after = 5463a0b6; scripts `census.py` / `census_cmp.py`; per-cue table `merged_census_after.csv`; per animal x
epoch x spout `transitions_final_animal_epoch_spout.csv`). Session-level classes (`cue_classes_for`):

| before \ after | incomplete | jaw_only | no_response | stopped |
|---|---|---|---|---|
| contact | 17 | 6 | 2 | 0 |
| incomplete | 0 | 7 | 5 | 4 |
| jaw_only | 0 | 0 | 7 | 5 |
| no_response | 0 | 0 | 0 | 3 |
| stopped | 0 | 171 | 143 | 0 |
| after_end | 0 | 0 | 0 | 43 |

- Rules 1 + 2 alone (A0 kept): 53 raw-class changes (contact -> incomplete 17, -> jaw_only 6, -> no_response 2;
  incomplete -> jaw_only 10, -> no_response 6; jaw_only -> no_response 12). Stopped 5,882 -> 5,895; stop runs
  277 -> 276.
- On PS92_0806 / 0811 and PS93_0925 / 1002 (both windows), rule 1 alone changes 0 classes. With rule 2, 1 trial changes
  (PS93_0925, cue_stop: contact -> incomplete, its only contact after 3.5 s).
- All rules: working (contact | incomplete) 49,111 -> 49,087 (-24); stopped 5,882 -> 5,623; after_end 43 -> 0 (all now
  stopped); jaw_only 1,059 -> 1,231; no_response 883 -> 1,037; stopped stretches 277 -> 258.
- Restored from stopped by the A2 guard, per epoch (close ipsi / mid / contra, far ipsi / mid / contra):
  - acute 0 / 0 / 8 / 69 / 68 / 85;
  - subacute 0 / 0 / 17 / 13 / 14 / 17;
  - chronic 0 / 0 / 5 / 8 / 2 / 8;
  - pre 0.
- Newly stopped (not counting after_end): pre 20, acute 14, subacute 4, chronic 17.

| animal | epoch | stopped>jaw | stopped>no_resp | after_end>stopped | contact>incompl | contact>jaw | contact>no_resp | incompl>jaw | incompl>no_resp | incompl>stopped | jaw>no_resp | jaw>stopped | no_resp>stopped |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PS92 | pre | 0 | 0 | 5 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| PS92 | acute | 18 | 12 | 5 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| PS92 | subacute | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| PS92 | chronic | 0 | 0 | 0 | 1 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| PS93 | pre | 0 | 0 | 1 | 5 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 |
| PS93 | acute | 4 | 2 | 0 | 2 | 1 | 0 | 1 | 0 | 2 | 2 | 1 | 2 |
| PS93 | subacute | 0 | 0 | 0 | 2 | 1 | 0 | 2 | 0 | 0 | 0 | 0 | 0 |
| PS93 | chronic | 0 | 0 | 6 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| PS94 | pre | 0 | 0 | 4 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| PS94 | acute | 82 | 64 | 2 | 2 | 0 | 0 | 1 | 2 | 0 | 3 | 0 | 0 |
| PS94 | subacute | 7 | 4 | 2 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| PS94 | chronic | 8 | 15 | 3 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| PS95 | pre | 0 | 0 | 4 | 0 | 0 | 0 | 1 | 0 | 1 | 1 | 4 | 1 |
| PS95 | acute | 31 | 17 | 2 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| PS95 | subacute | 21 | 29 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| PS95 | chronic | 0 | 0 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

**Encoder unchanged.**
- Check: `combined_encoding._load_behaviour("PS92", "20260806")` under the profile + the bundle env (run_bundle.sh vars,
  set after `render_profile.apply`), pickled before (571b3955) and after. Every field compared (arrays exact, frames by
  `assert_frame_equal`).
- IDENTICAL with LICKS_ALL unset (the bundle's setting) and with `COMBINED_ENCODING_LICKS_ALL=split`.
- IDENTICAL with split reading a licks_all.csv rebuilt into scratch by the new `all_licks_session`: 7,059 licks, same
  rows and order; `onset_context` == the on-disk `context` for every row; new `context`: pre_cue 2,398, excluded_0_70 22,
  post_cue 3,680, iti 959.
- After the A2 commit the attempts table moves with the tag, so the final check pointed `lp_licks.attempts_path` at the
  existing stopRun6j table: IDENTICAL. Only `engaged` can move, through the regenerated attempts table.

**Readers and what changes.**
- Immediately (code only): orofacial_cohort precue rate / frac / n_precue_licks (+ the new n_excluded column);
  trial_classes classes (rules 2 / 3); the window "trial" pre-cue contacts; trial_attempts (classes, A2, timing
  columns); the lp_licks first-lick time; cue_classes_for and every camera-gated consumer (new tag).
- After a licks_all re-render (the on-disk `context` changes):
  - pre_cue readers: enl_lick_audit, lick_enlv_extract, lick_enlv_balance_offset;
  - iti readers: iti_lick_rate, lick_cue_to_lick_extract, revup_neural, learned_compensation, restnl_rescue;
  - post_cue readers: kinematic_normality, prelick_hemisphere_tuning, prelaunch_aim_tracking, lick_firstlick_extract,
    lick_correction, iti_licks, stray_licks.
  - The last four (plus lick_enlv_extract) assert that the pose3d kinematics_all `context` matches licks_all by lick_uid,
    so they REFUSE until pose3d_kinematics (which copies `context`) is rebuilt too.
  - lp_licks, restnl_mask, epoch_maps and face_rhythm_* read every lick: unaffected. The encoder: unaffected
    (onset_context).
- Unchanged cue + 70 users: daq_trials.CONTACT_FROM_CUE_S and its users (spout_behavior:589, lick_bout_structure,
  quit_prodrome); revup_behaviour:164, tongue_contact_recall:28, tongue_nocontact_qc:109 (literal 0.07);
  tongue_kinematics:99-104 / defaults.yaml:1146-1153 (post-cue windows); orofacial_jaw_lateral, orofacial_w2,
  spout_frame; combined_encoding contact_s (daq_contacts_ms, same value).

**Where licks_precue.csv / licks_all.csv live, and what a re-render needs.**
- Both are per-session files of the v36 LP run:
  `M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/sessions/<animal>/<date>/`.
- licks_precue.csv is written by `orofacial_outputs process` (the whole pose pass from the O2 bundle predictions; it
  rewrites licks.csv, trials.csv, clean.npz ... too). licks_all.csv is written by `orofacial_outputs all-licks`
  (`_all_licks_worker`; reads only the session tables, cheap).
- Under cohort1_rebuild_20261103 only `<run>/imaging_encoding` of the orofacial root is routed (paths.yaml
  rebuild.derived_subtrees). `<run>/sessions` is an INPUT, so both writers target the original LP_analysis tree, and
  `writeguard.assert_writable` REFUSES them under the profile. Without the profile they would overwrite v36 in place.
- A re-render is not needed for the numbers. The detection pass is unchanged, so licks_excluded_0_70 = the peak split of
  today's licks_precue (done on read), and the new `context` = `lick_context(t_ms, ...)` on today's licks_all. If the
  tables are wanted, the clean route is a routed product under `<run>/imaging_encoding/` (or a new run folder), plus a
  pose3d_kinematics rebuild for the lick_uid readers.

**Tests.**
- New: tests/test_cue70_response_window.py (13), tests/test_stop_rule_a2.py (16), tests/test_lp_daq_timing_fallback.py
  (5); lick_context / onset_context / encoder_context in test_orofacial_outputs.
- Updated: the test_trial_classes fixture (peaks); tag strings in 9 files (stopRun6j -> stopA2jexrt3500c70);
  test_trial_attempts (A2: the tail is stopped, the early run folded).
- Related files at 5463a0b6: 241 passed; at 3613cf0b: 352 passed + the pre-existing failure below.

**Rule 5: stopped time holds no licks, EXCISE** (Priya 11-04, leaning; commit 3613cf0b; wf_attempt
docs/status/rebuild_20261103/STOPPED_LICKS_CENSUS_20261104.md, scripts scratchpad/slc/).
- The runs and the quit tail are formed exactly as A2 (or A3). Then every stopped trial TOUCHED by a lick becomes
  `stopped_licking` (`STOPPED_LICKING_CLASS`): in NEITHER stopped (`STOPPED_CLASSES`) NOR the working period, so it is
  out of the hit denominator and out of the stopped arms (trial_attempts `RUN_EXCLUDED`), but reportable as a class.
  The other run trials stay stopped; runs are not re-formed.
- **Touched (the census definition):** a lick anywhere in the trial's own span, strobe (missing: cue - 3 s) .. stop,
  or in the ITI on EITHER side of it (previous stop .. its strobe, its stop .. the next strobe). Licks = camera licks
  by PEAK (v36 `licks_all.csv` `daq_peak_s`, every context) + DAQ contacts (`dlc_frames.lick_onsets`). cam1-only events
  are not used (post-cue only; census: they add nothing in stopped spans). A lick belongs to the trial whose strobe is
  the last one at or before it (`span_lick_matrix`, categories precue / 0_70 / response / after_max_rt / iti_after =
  the census's `bin_events`).
- Options (env / `trial_selection`, named in the tag):
  - `WIDEFIELD_STOP_LICKS` / `stop_licks` = excise (default, tag "ex") | strict (tag "sb": the census's strict break,
    runs cut at lick break points and re-formed, touched trials leave them; the quit tail is not touched) | none.
  - `WIDEFIELD_STOP_TAIL` / `stop_tail` = lick_trial (default: the tail starts after the last camera-lick TRIAL) |
    lick_event (tag "te": the tail starts at the last trial with a lick break point, the census's tail-event). Open
    with Priya.
  - A guarded rule without span lick counts RAISES unless none + lick_trial (no silent pre-11-04 classes). A0 ("run")
    does not use lick times.
- Spans are passed by `cue_classes_for` (`cue_span_licks`, the DAQ cue train; unmatched cues strobe = cue - 3 s,
  stop = cue + max_rt), `trial_attempts` (trial rows) and `scripts/s1_trial_class_snapshot.py`. licks_all.csv enters
  `class_source_signature`.
- Tags: primary `...-cls-contact-incomplete-stopA2jexrt3500c70` (one run-tag change covering max_rt + cue70 + A2 +
  excise); strict `stopA2jsbrt3500c70`, tail-event `stopA2jextert3500c70`; profile arms `stopStrict` / `stopTailEvent`
  (with `stopA0` / `stopA3`).
- **section G `g_group`** lives on rebuild/bcdg-port (724862f9), not on this branch: at merge it must map
  `stopped_licking` to its own group (it routes stopped through the raw class today, which would put these trials back
  in none / jaw_only).
- **Reproduction** (`repro_slc.py` on the census's own gather.pkl + srg trials, pre-11-04 classes): A2 + excise =
  5,338 stopped + **273 stopped_licking (pre 50, acute 141, subacute 51, chronic 31)**; A2 hit rates identical (max
  |diff| 0 over animal x epoch; cohort pooled pre .956, acute .667); strict = 5,255 stopped (cohort pooled pre .953,
  acute .652, as the census); strict + tail-event 5,228; excise + tail-event 5,332 + 271.
- **On the final code** (census `census_final.csv`, 57,048 cues): stopped_licking **276** (pre 51, acute 143, subacute
  51, chronic 31; the census's pending-rule count on trial rows is 275), all of them A2-stopped trials; nothing else
  moves; hit rates identical to A2 without the excise (cohort pooled pre .9559, acute .6668, subacute .892, chronic
  .9731).

**Combined transition table, all five rules** (571b3955 -> 3613cf0b, session-level classes, 57,048 cues):

| before \ after | incomplete | jaw_only | no_response | stopped | stopped_licking |
|---|---|---|---|---|---|
| contact | 17 | 6 | 2 | 0 | 0 |
| incomplete | 0 | 7 | 5 | 1 | 3 |
| jaw_only | 0 | 0 | 7 | 5 | 0 |
| no_response | 0 | 0 | 0 | 3 | 0 |
| stopped | 0 | 171 | 143 | 0 | 266 |
| after_end | 0 | 0 | 0 | 36 | 7 |

stopped 5,882 -> 5,347; stopped_licking 0 -> 276; after_end 43 -> 0; working 49,111 -> 49,087. stopped_licking per
animal (pre / acute / subacute / chronic): PS92 8 / 44 / 0 / 0, PS93 8 / 39 / 22 / 7, PS94 23 / 58 / 3 / 9, PS95 12 / 2 /
26 / 15. Per animal x epoch x spout: `transitions_final_animal_epoch_spout.csv` (A2 without the excise) and
`merged_census_final.csv` (per cue).

**Encoder (final code, 3613cf0b):** PS92_0806 Behaviour IDENTICAL again (LICKS_ALL unset and split; attempts table
pinned to the existing stopRun6j table, since the new tag's table does not exist until trial_attempts is regenerated).

**Test record (3613cf0b, full suite, 4 shards run one at a time):** 4,790 passed, 171 skipped, 1 failed
(731 / 1,819 + 1 failed / 1,088 / 1,152). The failure is pre-existing and unrelated:
`test_review_7ab.py::test_working_trials_stays_byte_identical_to_main` compares wfield_local/working_trials.py with
origin/main, which has moved (origin/main 2eeead86 vs 571b3955 1a56bb25; this branch does not touch the file).
New tests in 3613cf0b: test_stop_rule_a2.py now 24 (span categories, touched definition, excise, strict, tail-event,
tags / raise, session_span_licks), test_trial_attempts.py +1 (stopped_licking leaves both sets).

**Per-cue stop kind (for the bcdg port g_group_final; coordinator 11-04).** `trial_classes.stop_kind_for(label, cue_s)` ->
"run" (a mid-session stop run) / "tail" (the quit tail after the last camera lick) / "" per DAQ cue, memoised with
`cue_classes_for` (`_CUE_KIND`, shared key `_cue_key`). `classify_session(..., return_kind=True)` returns (classes,
kind); `_classify_guarded` now returns both. stopped_licking trials keep the kind of the run / tail they came from.
Under rule "run" / "any" every stopped trial is "run" (no stopped tail there). trial_attempts writes a `stop_kind`
column (camera-gated runs). Tests: test_stop_rule_a2 +3 (a run, a tail and a stopped_licking trial in each; A3; A0;
`stop_kind_for` with and without a prior `cue_classes_for`), test_trial_attempts (stop_kind in both camera tests).
Related files: 367 passed + the pre-existing working_trials failure.

### 8.45 ENL window follows the lick source: ONE definition (Priya 11-04; branch rebuild/enl-window off 571b3955)

Written as 8.43 / 8.44 on the branch; renumbered 8.45 / 8.46 at the merge into rebuild/cohort1-20261103 (8.47).

Decision: every ENL analysis follows the lick source. LP licks give 1.0 s; DAQ licks give `decode.precue_post_s` = 2.0 s.
`WIDEFIELD_ENL_WINDOW_S` still overrides both (the matched DAQ 1.0 s arm, the LP 1.5 / 2.0 s sensitivity arms).
CODE + TESTS ONLY: no renders, nothing written to M:. Worktree `wf_rebuild_enl`.

**The definition (`wfield_local/lp_licks.py`).**
- `enl_post_s(source=None)`: THE window. The override if set; else 1.0 under LP; else the config window. `source`
  forces a lick source ("daq" for code that reads DAQ contact only).
- `post_s_for(align)`: `decode.{align}_post_s`, except for the pre-cue window (`precue`, `precue-*`), which is
  `enl_post_s()`.
- `gate_licks_s(label, cue_s, daq_lick_s, max_rt)`: the lick train an ENL lick-free gate must use, for scripts that
  gate on times.
  - DAQ: the DAQ train.
  - LP: `session_lp`'s edited train (the one `_trial_features` uses); cues without an LP verdict are dropped.
- `enl_window_s()` and `apply_enl_window` are unchanged. They stay None / identity on the default run, so default keys
  do not move.

**Sites (before -> after).**

| site | before | after | gate source |
|---|---|---|---|
| `precue_engagement_states.py:204` run_animal default | `precue_post_s` (2.0 under LP too; features were already 1.0 via `apply_enl_window`, the spec via make_spec) | `enl_post_s()` for precue; `cue` keeps 2.0 | `_trial_features`: same source |
| `scripts/enl_movement_removed_decode.py:104` | `precue_post_s` | `enl_post_s()`; output name keyed by the run tag | `_trial_features`: same source |
| `scripts/enl_movement_multisession.py:137` + `enl_table` | 2.0, DAQ licks always | `enl_post_s()`, `gate_licks_s` | **MISMATCH FIXED** (an LP window would have met DAQ licks) |
| `scripts/enl_fr_removed_decode.py:60` | 2.0 (via EM.enl_table, DAQ licks) | `enl_post_s()` + the fixed `enl_table` | **MISMATCH FIXED** |
| `scripts/enl_video_spout_check.py:142` + `enl_trials` | 2.0, DAQ licks always | `enl_post_s()`, `gate_licks_s`; output names keyed by the run tag | **MISMATCH FIXED** |
| `scripts/enl_lick_rates.py:51-53` | `precue_post_s`, DAQ licks always | `enl_post_s()`; LP train + LP coverage under LP; the `--cache` CSV records (source, window) and a mismatch is refused | **MISMATCH FIXED** |
| `scripts/cohort2_qc/quiet_task_events.py:57` | `ENL_S = 2.0` | `enl_post_s(source="daq")` = 2.0; each npz records `enl_s` | DAQ masks; see OPEN 1 |
| `scripts/within_spout_deviation.py:52` | `(-1.0, 0.0)` | `(-enl_post_s(source="lp"), 0)` = (-1, 0) | NOT lick-gated; see OPEN 2 |
| `wfield_local/precue_lickfree.py` PRE_S | 2.0 | `enl_post_s(source="daq")` = 2.0 | DAQ window, DAQ licks (unchanged) |
| `wfield_local/cd_trajectories.py:1158` | `decode.{align}_post_s` (2.0 under LP) | `post_s_for(align)` + `apply_enl_window` | **WINDOW FIXED**: under LP, `session_arms` slid a 2 s lick-free window over the LP train and fitted 2 s window means |
| `cd_overlap_null.py:343`, `scripts/component_exclusion_audit.py:60` | same | `post_s_for(align)` | same fix (they share cd_trajectories' arms cache) |
| `attempt_coding.py:223/384`, `attempt_maps.py:110`, `position_axes.py:243`, `position_coding_directions.py:772` | `decode.{align}_post_s` | `post_s_for(align)` | value was already corrected downstream (`apply_enl_window` / make_spec); now one read |
| `nightly_figs.py:256` (`--post-s`), `locanmf_analysis_deck.py:285` (notes text), `scripts/null_potent_session.py:273`, `scripts/locanmf_refit_test/refit_vs_projection.py:155` | config window | `post_s_for` / `enl_post_s` | `_trial_features`: same source |
| `scripts/epoch_sessions_long.py` CD pre-cue readout `[-2, 0)` | fixed 2 s | the dump's own fitted window (`res["win_s"]`); a DAQ dump keeps the label `[-2,0) s` | n/a |
| `scripts/rebuild_figs/fig_enl_move.py` | provenance note "[cue - 2 s, cue]" | note updated | n/a |

**Gate-source check (task 3).**
- The core builders take the window and the licks from the same switch (WIDEFIELD_LICK_SOURCE): `_trial_features`,
  `nolick_decoder.categorize` / `session_features`, `precue_window_start`. No mismatch there.
- Mismatches were only in scripts that gate on their own lick times. They read `dlc_frames.lick_onsets` /
  `_load_daq_events`, both DAQ:
  - enl_movement_multisession and enl_fr_removed_decode (shared `enl_table`);
  - enl_video_spout_check;
  - enl_lick_rates.
- All four now gate through `gate_licks_s` / `session_lp`.
- The cd_trajectories family had the right licks and the wrong window (fixed above).

**Other hard-coded windows (task 1 sweep), status.**
- `combined_encoding.py:1192`: the "enl" R2 scoring window, `max(strobe, cue - 2.0) .. cue`. It is a scoring window,
  not lick-gated. NOT changed, because changing it moves the O2 encoder outputs. Decision needed: OPEN 4.
- `combined_encoding.py:1663`: the maps' ENL state = strobe + 0.3 s .. cue. Unrelated to the lick-free window;
  unchanged.
- `configs/defaults.yaml decode.precue_lickfree_check_s: 2.0`: no code reads it (the `precue_lick` flag uses the
  effective window). Dead key; left.
- FaceRhythm windows `(-2, 0)` (face_rhythm/factors/config PRECUE_S, face_rhythm_readouts, fr position maps, fr_peak,
  fr asym): behaviour windows on a projection grid that starts at cue - 2 s, not lick-gated ENL analyses. Left.
- `scripts/fr_enl_decode.py`: explicit W = 1 and 2 s, gate = LP OR DAQ. Left (explicit sweep).
- `orofacial_w2.py` / `orofacial_heatmaps.py`: "NO-LICK ENL frames cue - 2 s .. cue" with the LP tongue gate
  (behaviour). Left as a 2 s LP-gated behaviour window. Flag for Priya if "every ENL analysis" includes behaviour.
- `scripts/lick_enl_angle_decode.py`, `lick_enl_target_test.py`: explicit W = 1.0 / 0.5 s, lick-free rule 2 s on the
  iti_licks behaviour licks. Left (own family, explicit).
- `scripts/enl_lp_census.py`, `gate_lp_check.py`: censuses of 2.0 / 1.5 / 1.0 / 0.5 s windows by both sources. Left.
- `scripts/s1_refit_basis_table.py`, `s1_refit_locanmf_table.py`: their docstrings run `WIDEFIELD_LICK_SOURCE=lp
  WIDEFIELD_ENL_WINDOW_S=2.0` (ENL pinned at 2 s, Priya 11-03). The override is still honoured. This conflicts with
  11-04 unless it is meant as the table's sensitivity row: OPEN 3.
- `state_spout_check.py:49`, `disengagement_tests.py:455/1348`, `revup_neural.py:303`, `iti_licks.py CUE_EXCL`:
  behaviour / exclusion windows, not ENL lick-free windows. Left.
- Function defaults `post_s=2.0` (beta_maps, nolick_decoder, enl_decode, enl_lick_control, grant_confusion,
  poststroke_compare, ...): each reaches its features through `trial_features_cached` / `_trial_features` /
  `nolick_decoder.session_features`, which apply `apply_enl_window` BEFORE the key. Frozen specs go through
  `make_spec`. So they resolve to the shared window; left.

**1.5 s (task 5).** No widefield code uses a 2pRAM-style 1.5 s ENL default. 1.5 s appears only:
- as an explicit `WIDEFIELD_ENL_WINDOW_S` sensitivity arm (`lp_final_render`, tests);
- in sweeps (`precue_window_sweep`, `enl_lp_census`).

**Frozen ids / cache keys (task 2).**
- Frozen specs: unchanged. `make_spec` already replaced a pre-cue post_s with `enl_window_s()`; the callers now pass
  that value themselves, so every id stays the same (tested: `test_frozen_spec_carries_the_window`).
- tf-* / nolick / map caches: unchanged (the window was already in the key via `apply_enl_window` /
  `map_collect.cache_kind`). Map digests checked, base vs branch: bm 503fe1b00ac5, prm 94846e2d13ab, restnl
  47b715c3d0df, all identical.
- CHANGED under LP / an override only, because the value is now the real window:
  - cd_trajectories `cdarms-*` (post_s), `cdfit*` (win_n) and courses (win_s), shared by cd_overlap_null and
    component_exclusion_audit;
  - cold once; the old LP entries held 2 s windows;
  - default DAQ keys unchanged.
- New identity:
  - enl_lick_rates `--cache` CSV: lick_source and post_s columns;
  - quiet_task_events per-session npz: `enl_s` (files without it read as 2.0, so existing files stay valid);
  - enl_movement_removed_decode / enl_video_spout_check: file names via `run_mode.keyed_name` (unchanged on the default
    run).
- Side effect: the LP lick-times local sidecar (`lp_lick_times`, keyed on lp_licks' source) recomputes once, with the
  same values.
- GAP, not fixed (outside these paths): some session-cache kinds do not carry the window (`evoked_amp__*`,
  `fixed_scale_maps__*`, `spatial_area_matrix__*`, `within_acc__*`). They are shared between a source's default window
  and an override arm (DAQ vs DAQ_ENL1s; LP 1.0 vs LP + 1.5 / 2.0).
  - Fix: an ENL tag in `session_cache.effective_kind` when an override differs from the source default.
  - Left because it would cold-start the override arms (incl. the s1_refit LP + 2.0 tables).

**O2 encoder (task 6).**
- combined_encoding, o2_encoding, frozen_encoder and o2_bundle use only `lp_licks.attempts_path` and
  `lp_licks.output_dir`. Both are unchanged, and none of the switched modules is imported by them.
- No bundle input changes.
- The maps' ENL state (strobe + 0.3 s .. cue) is unrelated, as expected.
- BUT the encoder has a second ENL quantity: the "enl" R2 scoring window, `cue - 2.0 .. cue` (bounded at the strobe).
  It is hard-coded and does not depend on the source. Left unchanged so the bundles stay valid: OPEN 4.

**OPEN (for Priya).**
1. `quiet_task_events` is a COHORT-1 script (PS92-95), despite its folder.
   - Its `ev_enl` / `npre_enl` are "the decoder's window", and the T13 p_true comes from `attempt_coding` (LP-only,
     1.0 s).
   - Kept on DAQ 2.0 s as briefed.
   - Following LP needs MIN_CLEAN_S (1.0 s of clean frames inside the window) re-decided; otherwise only fully clean
     1 s windows survive.
2. `within_spout_deviation --window enl` is 1.0 s (= LP) but is not lick-gated by any source. Its trials are engaged
   and its licks are the bundle's video licks; the ENL is contact-free by task design only.
3. The s1_refit tables pin ENL at 2.0 s under LP (11-03); 11-04 says LP gives 1.0 s.
4. The combined_encoding "enl" scoring window is fixed at 2 s on this branch. Closed at the merge: encoder-classes
   (8.43) made it the decoders' lick-free search (LP 1.0 s; `enl_fixed2s` kept), and its length is now `enl_post_s` (8.47).

**Tests.**
- New `tests/test_enl_window_source.py` (43 tests):
  - the window under DAQ / LP / override / a forced source; `post_s_for`; agreement with `apply_enl_window`; the
    frozen spec;
  - `gate_licks_s` under both sources;
  - `enl_table` / `enl_trials` gating on the window's source (fake session);
  - each switched site resolving to the shared window (at runtime where cheap, static otherwise);
  - a repo-wide check that only lp_licks (and precue_engagement_states' `cue` branch) read the config window
    directly.
- `test_share_writers_mode_safe`: enl_video_spout_check moved out of BEHAVIOUR (it now routes).

**Test record (cb7b1f49).** New file: 43 passed. Related set (23 files): 414 passed. Full suite, 4 shards run one at
a time (RAM checked before each): 4,787 passed, 171 skipped, 1 failed.
- Shard 0: 848 passed, 27 skipped, 1 failed.
- Shard 1: 1,918 passed, 27 skipped.
- Shard 2: 1,144 passed, 117 skipped.
- Shard 3: 877 passed.
- The failure is `test_review_7ab::test_working_trials_stays_byte_identical_to_main`, and it was already there. It
  compares against the LIVE origin/main, where `working_trials.py` moved in 4a37cbc8 (11-04 B/C/D/G). This branch does
  not touch the file: its blob equals the one at 571b3955.

### 8.46 Lick-free SLIDING ENL for every ENL predictor; quiet_task_events and s1 refit on LP 1.0 s (Priya 11-04)

Priya's decisions, all relayed by the coordinator on 11-04:
- quiet_task_events (cohort 1) switches to the LP no-lick 1.0 s ENL.
- Any ENL prediction must use a LICK-FREE ENL.
- The window SLIDES with the decoders' own search, and a trial is dropped only when no lick-free window exists. This
  is neither "fully clean trials" nor a fixed-window gate.
- The s1 refit tables switch to LP 1.0 s, with 2.0 s kept only as a labelled sensitivity arm.

Commits:
- 9efc3c22: s1 refit.
- c73d04ee: quiet_task_events.
- 8d912ae5: ENL predictors.

**The slide (`lp_licks.enl_window_starts_s`, new).** It runs the decoders' own `precue_window_start` on a 1 ms grid
(that function works in integer frames); it is not a new rule:
- the fixed window ending at the cue, if it holds no lick;
- otherwise the latest lick-free gap after the strobe that fits the window;
- otherwise NaN, and the trial is dropped.

The licks come from `gate_licks_s` (the window's source). As in `_trial_features`, a CLEAN fixed window is kept even
when it starts before the strobe (pinned in a test). The scripts that required `strobe <= cue - W` before still do.

**Per site: did it already slide?**

| site | before | now |
|---|---|---|
| s1_refit_basis_table / s1_refit_locanmf_table | already slid (`_trial_features`); post_s literal 2.0, run pinned at ENL 2.0 | `post_s_for` -> LP 1.0 s; 2.0 s = sensitivity arm `WIDEFIELD_ENL_WINDOW_S=2.0` (folder LP_licks_ENL2s); CSV `enl_s` |
| quiet_task_events `ev_enl` / `npre_enl` | fixed DAQ 2 s, NOT lick-gated (window frames only masked) | LP 1.0 s, gated + slid; trials without a window are out of the ENL rows |
| within_spout_deviation `--window enl` | fixed [cue - 1, cue), UNGATED | `enl_post_s()` of the run's source, gated + slid |
| enl_movement_multisession / enl_fr_removed_decode (`enl_table`) | gated (8.45), fixed window, dropped on a lick | slid (`w0_s`); features, lead-in, end-0.5 s arm on the slid window |
| enl_video_spout_check (`enl_trials`) | gated (8.45), fixed | slid; motion energy on the slid window |
| `_trial_features` / nolick / cd_trajectories / attempt_coding / quiet_task_events_followup | already slid | unchanged |
| precue_lickfree (Section C control) | already searches (`lickfree_window`) | unchanged |
| enl_lick_rates | a lick-rate measurement, not a predictor (`no_clean` uses the search) | unchanged |

**Gated but NOT sliding, left unchanged (decision needed).** These use a fixed window and drop on a lick:
- iti_licks-family ENL predictors: `lick_enl_angle_decode`, `lick_enl_target_test`, `lick_enl_offset_explore`, and
  the pre-cue parts of `prelaunch_aim_tracking` / `revup_neural`.
  - They use a fixed [-W, -0.10] window on the cue-aligned traces and drop any trial with a lick in [cue - L, cue)
    (L = W or 2 s).
  - They document why they do not slide: the traces start at -2 s, and a slid window mixes in the spout-arrival
    transient.
  - They are frozen-decoder families with their own outputs, so converting them is a separate job.
- `fr_enl_decode`: fixed W = 1 / 2 s, dropped on an LP or DAQ lick. The FaceRhythm grid starts at cue - 2 s, so a
  slide could only use [max(strobe, cue - 2), cue).
- Not predictors, so left as they are:
  - `combined_encoding`'s "enl" R2 scoring window (8.45, OPEN 4);
  - the `epoch_sessions_long` CD readout (a projection of directions fitted on the slid windows).

**quiet_task_events: what "clean" means, and MIN_CLEAN.**
- CLEAN = the detection mask, not the lick-free test:
  - no DAQ contact lick within TASK_BUF_S = 0.5 s (plus LP licks when QS_LICK_SOURCE=daq+lp);
  - no running bout within 0.5 s;
  - no swapped pair within 0.5 s;
  - finite frames.
- So the LP gate does not make the requirement redundant. It stays, scaled with the window:
  `MIN_CLEAN_ENL_S = 0.5 x ENL_S` = 0.5 s of 1.0 s. That is the same fraction as the 1.0 s of 2.0 s used until 11-04.
  The strata bins are scaled the same way (`CLEAN_ENL_BINS`; identical at 2 s).
- `MIN_CLEAN_S` (whole pre-cue, `ev_pre`) is unchanged.

Cheap count, read-only:
- Sources: the existing 108 task_events npz (exact old counts), the events npz and the LP tables.
- New clean time is approximated in continuous time: no swapped-pair / NaN frames. Check: old approx 49,790 vs exact
  49,113, so the approximation is about 1.4 % high.
- Totals:
  - engaged trials with a pre-cue window: 51,455;
  - old (DAQ 2 s, >= 1.0 s clean): 49,113;
  - LP lick-free 1 s window exists: 50,407 (6,768 of them slid);
  - new, >= 0.5 s clean: 48,919;
  - new, whole window clean (>= 1.0 s): 11,134.

So 0.5 x W keeps the old trial count, whereas a fully clean window would keep about 22 %. Per animal x epoch:

| animal | epoch | trials | old 2 s, >=1.0 | LP window | slid | new, >=0.5 | new, >=1.0 |
|---|---|---|---|---|---|---|---|
| PS92 | pre | 2920 | 2577 | 2910 | 494 | 2652 | 661 |
| PS92 | acute | 2091 | 1945 | 2086 | 261 | 1958 | 973 |
| PS92 | subacute | 593 | 568 | 587 | 83 | 561 | 136 |
| PS92 | chronic | 3962 | 3886 | 3863 | 987 | 3835 | 829 |
| PS93 | pre | 4236 | 3885 | 3630 | 967 | 3441 | 1415 |
| PS93 | acute | 1755 | 1677 | 1739 | 278 | 1666 | 1073 |
| PS93 | subacute | 1390 | 1327 | 1383 | 299 | 1338 | 531 |
| PS93 | chronic | 5939 | 5811 | 5914 | 1085 | 5840 | 1424 |
| PS94 | pre | 4249 | 4090 | 4220 | 379 | 4120 | 454 |
| PS94 | acute | 2740 | 2581 | 2721 | 235 | 2574 | 1168 |
| PS94 | subacute | 2528 | 2388 | 2504 | 285 | 2373 | 314 |
| PS94 | chronic | 4467 | 4296 | 4408 | 454 | 4279 | 446 |
| PS95 | pre | 5032 | 4744 | 5027 | 106 | 4981 | 405 |
| PS95 | acute | 720 | 715 | 718 | 33 | 717 | 298 |
| PS95 | subacute | 3007 | 2905 | 2995 | 183 | 2916 | 432 |
| PS95 | chronic | 5826 | 5718 | 5702 | 639 | 5668 | 575 |

PS93 pre loses the most: 606 of its 4,236 trials have no LP lick-free 1 s window, and 551 of those are PS93_0813,
which has no video and so no LP verdict.

**quiet_task_events identity.**
- The run's lick source defaults to LP (`os.environ.setdefault`, as in the followup).
- Each npz records `enl_s`, `enl_src` and `enl_w0` (window start per trial). A pre-11-04 file reads as the fixed
  2 s window.
- Outputs go to `task_events_<src>lf<W>s/`; the DAQ 2 s run keeps `task_events/`. The followup imports TE / TE_SUFFIX
  and keys its own folder the same way.
- Under LP, V2 itself is already routed to `labcams/LP_licks/...` by the run tag, so the old DAQ files can never be
  read.

**within_spout_deviation.**
- `ENL_W = enl_post_s()` of the run's source: LP 1.0 s; a DAQ-env run now gets 2.0 s (it was 1.0 s, ungated).
- Licks: `dlc_frames.lick_onsets` (DAQ), or the LP train via `gate_licks_s`.
- The window is slid (`enl_window_starts`); trials without one are dropped.
- Folder `within_spout_deviation[_..]_enllf<W>s` (was `_enl`), so old ungated per-session npz are never reused. No
  reader depends on the old name.

Before / after under LP: 46,495 -> 46,092 trials (99.1 %), from a cheap count on behaviour only, all curated sessions
from 08-06. PS93_0813 is excluded in both, as it has no behaviour bundle (no video).

| animal | pre | acute | subacute | chronic |
|---|---|---|---|---|
| PS92 | 2690 -> 2680 | 1485 -> 1480 | 540 -> 534 | 3806 -> 3777 |
| PS93 | 3272 -> 3226 | 1142 -> 1133 | 1215 -> 1208 | 5724 -> 5705 |
| PS94 | 4088 -> 4059 | 1980 -> 1962 | 2428 -> 2405 | 4433 -> 4374 |
| PS95 | 4798 -> 4793 | 512 -> 510 | 2689 -> 2677 | 5693 -> 5569 |

**Tests.**
- `tests/test_enl_window_source.py`, now 49 tests:
  - the slide equals `precue_window_start`;
  - `enl_table` / `enl_trials` slide on the window's source (a fake session with a slid, a no-verdict and a
    no-window trial; `w0_s` checked);
  - within_spout's window follows the run source and drops only trials with no lick-free window;
  - quiet_task_events npz fields and keyed folders, and within_spout's keyed folder.
- Full-suite record: below.

**Test record (8d912ae5).** Full suite, 4 shards run one at a time (RAM >= 9.5 GB free before each): 4,790 passed,
171 skipped, 1 failed.
- Shard 0: 848 passed, 27 skipped, 1 failed.
- Shard 1: 1,921 passed, 27 skipped.
- Shard 2: 1,144 passed, 117 skipped.
- Shard 3: 877 passed.
- The one failure is the same as in 8.45 and was already there: `test_review_7ab::test_working_trials_stays_byte_identical_to_main`,
  which compares against the live origin/main; working_trials.py is untouched here.

### 8.47 Merge of encoder-classes, cue70 and enl-window into rebuild/cohort1-20261103 (11-04)

CODE + TESTS ONLY: no renders, nothing written to M:. Merged with `git merge --no-ff`, in this order, onto 571b3955:
rebuild/encoder-classes (825020c2, merge 3c4eb932, no conflicts), rebuild/cue70 (88b1cb8e, merge 28939c46), rebuild/enl-window
(f056c223, this merge). rebuild/bcdg-port is NOT merged (follow-ups pending); its log sections (written as 8.44 / 8.45) became
8.48 / 8.49 when it was merged after this one (8.50).

**Log numbering.** encoder-classes = 8.43, cue70 = 8.44 (was 8.45), enl-window = 8.45 / 8.46 (were 8.43 / 8.44). Cross-references
inside the enl sections were renumbered with them; 8.41's "see 8.43" now says no full suite ran at c3cb3c0c.

**Encoder lick contexts (cue70 x encoder-classes).** `_context_licks` reads cue70's ONSET label (`oo.encoder_context`:
`onset_context`, else `context` for a pre-11-04 licks_all) for pre_cue / iti AND for encoder-classes' `post_extra`
(post-cue onsets held by the pre-cue / ITI tables). So a licks_excluded_0_70 row (a licks_precue row before 11-04) keeps
its kernel and LICKS_ALL=split still puts every lick in exactly one kernel. On PS92_0806, relabelling the existing
licks_all by peak (22 licks_precue rows with a 0-70 ms peak -> excluded_0_70) leaves the three lists unchanged (2,397 /
967 / 15); a peak-label read would have moved 11 licks out of pre_cue and 11 out of post_extra. test_cue70_response_window's
encoder test now carries `table` and checks post_extra.

**One LP-vs-DAQ tolerance.** `combined_encoding.LP_DAQ_TOL_S` = `lp_licks.LP_DAQ_TOL_MS` / 1000 (0.05 s, unchanged).
The two rules are otherwise NOT the same: at a FAR spout with no LP onset the encoder's per-trial alignment falls back to
the DAQ contact (`timing_source` daq), while `lp_licks.first_lick_timing` leaves the trial untimed. Decision needed.
Both read max_rt from `decode.max_rt_s`.

**One ENL search, one ENL length.** The search is `enl_window.precue_window_start` (encoder-classes moved it there
verbatim). The decoders re-export it, the combined encoder calls it on frames, and enl-window's `lp_licks.enl_window_starts_s`
(time-based scripts, 1 ms grid) now imports it from `enl_window` (it went through the decoder module, which pulls in
sklearn). The length: `enl_window.enl_spec` and the encoder's `COMBINED_ENCODING_ENL_SOURCE` override both take
`lp_licks.enl_post_s` (they re-implemented it, reading `precue_post_s` directly). Same numbers under the production
settings. One change: an encoder run with ENL source daq and WIDEFIELD_ENL_WINDOW_S set now uses that window, like every
other ENL analysis (8.45). Test: test_enl_window_source::test_one_enl_search_and_length_for_decoders_encoder_and_time_scripts.
8.45 OPEN 4 (the encoder's fixed 2 s enl window) is closed by 8.43.

**Smoke (PS92_0806, rebuild profile + the r4 run_bundle.sh env applied after `render_profile.apply`).** `_load_behaviour`
at the merged tip vs the encoder-classes tip (825020c2), the same attempts table for both (the stopRun6j one; the
stopA2jexrt3500c70 table is not built yet): LICKS_ALL unset and split are BIT-IDENTICAL (every Behaviour field;
split lists 2,397 pre_cue / 967 iti / 15 post_extra; 381 trials, 364 engaged, 3,680 licks). The session's licks_all is
pre-11-04 (no onset_context), so the onset label cannot differ there; the relabel check above covers a re-rendered one.
The run tag under cohort1_rebuild_20261103 is ...stopA2jexrt3500c70 (= the profile's expected_run_tag).

**Test record (2387cf20).** Full suite, 4 shards run one at a time (>= 45 GB free before each): 4,868 passed,
173 skipped, 1 failed (shards: 1,902 + 140 skipped; 949 + 27; 883 + 5; 1,134 + 1, 1 failed). The failure is the known
`test_review_7ab::test_working_trials_stays_byte_identical_to_main` (bcdg-port replaces it).

### 8.48 Port of main 4a37cbc8 (B / C / D / G, Priya 11-04) + follow-ons (branch rebuild/bcdg-port, from 571b3955)

Written as 8.44 / 8.45 on the branch; renumbered 8.48 / 8.49 at the merge into rebuild/cohort1-20261103 (8.50).

Code and tests only: no render, nothing written to M:. Worktree `wf_rebuild_bcdg`.

**Why a hand port.** `git cherry-pick -n 4a37cbc8` conflicts in 8 of 11 files. The main cause: the rebuild never
took main 26fd5b44's items B / C (spout-frame state discriminators, the projection frame). Those were default OFF;
the rebuild ported only the working-row training from that commit (8.33). a8's B / C edits sit on that missing
infrastructure, so it is ported here first, still default OFF.

**Defaults that land.**
- Section D is scored on working rows (`WIDEFIELD_SECTION_D_SCORED=working`).
- Section G is grouped by camera class (`WIDEFIELD_SECTION_G_GROUPS=camera`).
- The state discriminators stay z: `STATE_DISC_DEFAULT = session_z`, so B is OFF.
- The projection frame stays off: `WIDEFIELD_PROJECTION_FRAME=session_z`, so C is OFF. Its working / contact centre
  is an option only.
- G4's primary is the camera-class discriminator (`WIDEFIELD_G4_DISC=camera`).
- The G9 camera-class block is computed alongside the DAQ classes (`WIDEFIELD_G9_CLASSES=camera`); every figure
  still reads the DAQ classes.

**Port map (main -> rebuild).**

| main (4a37cbc8 / 26fd5b44) | rebuild location | note |
|---|---|---|
| `working_trials.G_GROUPS / G_WORKING / g_group / g_jaw_judged / _group_table` | same | verbatim; `G_LABELS` added (after_end reads "quit") |
| `working_trials.row_groups` (s1 snapshot) | same name, LIVE classes | via the new `trial_classes.cue_groups_for` (raw class, after-end flag, matched), the same end and coverage rule as `classify_session`. Stopped and early_run map through their RAW class: early_run -> none; stopped -> none / jaw_only, or after_end when past the last camera lick. after_end -> after_end. So the five groups are main's. Positions come from YE / YU for `jaw_judged`. A session with no matched cue is not in `have`. |
| `working_trials.projection_work` | same | live classes; returns `(we, wu, cls_e, cls_u)`; + `projection_cls_e` (the engaged centre needs classes under the camera gate) |
| `locanmf_frozen_decoder.section_d_scored / SECTION_D_SCORED_ENV / _scored_rows` + scoring block | same | Reconciled with 8.33, which already TRAINED and centred on working rows but scored engaged rows (`scored_rows: "engaged"`). The field now reports the scored set. New fields: `scored_set`, `n_scored`, `n_scored_from_nolick`, `per_session_engaged_scored`, `scored_centre` (+ `confusion_by_class`, panel commit). |
| `spout_scaling.state_disc_centre / STATE_DISC_CENTRE_ENV` | same | |
| `spout_scaling.projection_centre / PROJECTION_CENTRE_ENV / projection_arrays(work=, YU=)` | same + `cls_e=`, `labels=` | engaged / working / **contact** (coordinator 11-04); `projection_tag` |
| 26fd5b44: `SCALING_SPOUT_FRAME`, `STATE_DISC_DEFAULT`, `state_disc_scaling` spout option, `FrozenFrame.train_apply`, `PROJECTION_FRAME_ENV`, `projection_frame_on` | `spout_scaling` | the state_bal_em helpers now act only on state_bal_em |
| `grant_kit._proj_arrays` (+ `_session_trials` through it) | `grant_kit` | `_frozen_frame(..., centre=)`: working or contact centre; the memo key carries it |
| `grant_similarity` (2 sites) | same | 26fd5b44 |
| `poststroke_compare.section_g_groups, _camera_groups, _conf_panel, _loso_rows, _by_class_panels, _state_frame_tag` | same | `_camera_groups` gates on the camera gate + `working_selection()` (the rebuild's tests); `_by_class_panels` adds `group_labels` |
| `poststroke_compare` crossed_confusion / decode_matched / `_proj_X` / looks_like_which / fits_engaged | same | + 26fd5b44 `_spout_train_d`, `_lick_train` / `_lick_scored` spout branch, `pattern_similarity` in `_proj_X` (memo keyed by `projection_tag`) |
| `poststroke_section_g._counts` by_class | same | design sidecar `section_g_design.json` now says camera grouping, `scored_centre`, `state_disc_frame` |
| `precue_engagement_states._working_frame` + spout-frame block in `run_animal` | `_working_frame` + `_spout_frame_block` | Kept to a few lines in `run_animal`: the bookkeeping moved into `_spout_frame_block`, and the `post_s` default line (204) is untouched for rebuild/enl-window. The stored discriminator's spec scaling = `spout_frame/<working or engaged>` when ON; unchanged (None) when OFF. |
| `position_coding_directions.run_animal`, `position_axes.run_animal` | same | projection with masks + classes; the confusion block and its STORED decoder keep the production rows (`XE0`) |
| `attempt_coding` (26fd5b44 projection) | same | **follow-on (d)**: masks and classes passed, so every centre works |

**Conflicts and how they were resolved.**
- DECISIONS.md: not cherry-picked. The rebuild's own 11-04 entry stands; this log is the record.
- `grant_kit`: main replaced `_frozen_frame`. The rebuild's version (store, centre rule, working centre) is kept, and
  `_proj_arrays` is added before it.
- `locanmf_frozen_decoder`: the rebuild's working training, `frozen_scored_frame(work=)` and PS93_0813 rules are kept;
  main's scoring block is grafted on.
- `poststroke_compare` / `precue_engagement_states`: the rebuild's stored references (`_g_store`, `_stored`) are kept;
  main's branches are added around them.
- `position_*`, `spout_scaling`: the 26fd5b44 code was missing, so it was added.

**Coordinator 11-04 additions.**
- **Scored-frame centre kind: `WIDEFIELD_SPOUT_SCORE_CENTRE`** (Priya 11-04, **FINAL: Wt**; also set in the rebuild profile).
  - Kinds: `wt` (DEFAULT) | `working` (W, the label-free bal_em that was the rebuild rule until today) | `contact` (C)
    | `cto` | `n`.
  - Definitions: main `docs/status/CENTRE_MIXSHIFT_2026-11-04.md` and its `centre_mixshift_1104/cmcommon.Engine`.
    Implemented in `spout_scaling.scored_centre` / `true_label_centre` / `centre_refs` / `kind_centre_fn`.
    - **Wt** = the equal-weight mean of the per-spout means of the session's WORKING rows (spout = DAQ strobe label),
      SD over those rows; no EM, no pre fill.
    - **Cto** = the mean of the contact rows − sd · (true spout share of the contact rows) · δC. δC = the pooled pre
      contact-row offsets from each training session's balanced working centre, in working-SD units, averaged over the
      scoring model's training sessions (nan → 0).
    - **N** = one fixed raw-unit frame from the scoring model's PRE training sessions (mean of m_s + s_s·c_s, mean of
      s_s·sdw_s), expressed in each session's z. It needs `pool_sessions`' per-session z stats, now registered per XE
      object (`register_pool_stats` / `pool_stats_of`; the return value is unchanged).
  - **Min-rows guard (Priya 11-04).** Wt needs >= 20 (MIN_CENTRE_ROWS) working rows at EVERY spout and REFUSES
    (BalEmCentreRefused) otherwise. The check's "drop spouts < 5 and renormalise" is deliberately NOT ported: no fill,
    no fallback. A session with no working rows (PS93_0813) is unscored as before; a post session with none raises.
    Dry count of the minimum below.
  - Applies to: `frozen_scored_frame` (sections D, G, attempt, joint) and `grant_kit._frozen_frame` (5b / 5c and the
    working projection).
    - On the lick-only working arm (a `pos`-spout model): wt is the balanced centre over the `pos` spouts; cto and n
      refuse there.
    - The lick window / engaged models are unchanged.
  - The models are unchanged (training frame unchanged); only the scored frame moves.
  - The kind is carried in `frozen_X`'s memo key (`X<tag>`), the `_frozen_frame` memo key (`_c<kind>`),
    `pooled_frozen_loso` `scored_centre` (e.g. `wt_centre`), the section G sidecar and `_state_frame_tag`.
  - **Known caveat (CENTRE_MIXSHIFT 3d):** Wt drifts like W under a high incomplete share, about −.024 overall
    [−.042, −.011] (cue, acute mix). That is mechanism (ii), the incomplete trials' own pattern, not the EM.
- **Projection centre `WIDEFIELD_PROJECTION_CENTRE`** = `wt` (default, following the decision) | engaged | working |
  contact | cto | n. It is used only while `WIDEFIELD_PROJECTION_FRAME=spout_frame`; the frame stays OFF by default.
  The G4 engagement discriminator stays z on contact rows (`WIDEFIELD_G4_CENTRE=contact`).
- **(a) G4 camera-class discriminator** (`poststroke_compare.g4_camclass`; section G record `g4_camclass_{precue,cue}`,
  `g4_primary`). Ported from g4common.cam_balanced_sample.
  - Training: pre WORKING (contact + incomplete) vs NONE, position-balanced. jaw_only and after_end are scored as their
    own groups, never trained on.
  - Frame: z on CONTACT rows by default (`WIDEFIELD_G4_CENTRE` = contact | working | engaged; engaged = the production
    z, identity).
  - Scores are spout-balanced per group. Cells with n < 10 are kept, flagged `thin`, and left out of the means.
  - Each post group is read against the same pre group, else against pre working (labelled).
  - Stored as kind `g_cam_disc`. The spec extra carries keep, seed, `camclass/z-<centre>/pos-contact+incomplete/
    neg-none` and the run's selection tag.
  - The DAQ G4 readouts stay as robustness. `WIDEFIELD_G4_DISC=daq` drops the camera block. The G4 FIGURES still draw
    the DAQ readouts: no figure function reads `g4_camclass_*` yet.
- **(c)** `decode_matched` lick-only arm under the camera grouping: `_lickonly_working`.
  - It trains on pre working rows at the kept spouts; the pre band is their working rows (LOSO); the post arm is the
    post session's working rows.
  - Stored as kind `g_lickonly_decoder`, align `...-ffwork-<run_selection>`: new ids.

**New frozen-model identities.** Ids are content hashes of the training data, so they exist only after a render. Spec
changes:
- `g_lickonly_decoder` with align `<cue|precue>-keep<spouts>-ffwork-<selection tag>-f<cols>`: NEW, one per animal x
  window x preserved-spout set.
- `g_cam_disc`: NEW kind, one per animal x window x keep x seed x centre.
- Unchanged ids (spec unchanged):
  - the stored cue / precue spout decoders (`decoder`): the score centre is not in the spec;
  - `g_lick_disc` / `g_lick_disc_folds` / `engagement_states` while B is OFF;
  - `attempt_axes` / `coding_decoder` while C is OFF. With C on, the projected rows change the training-data
    fingerprint, so ids change automatically.
- FREEZE_VERSION stays 2; the class set rides in the spec extra.

**Not ported / left as is.**
- joint_xsession's section D decoder still scores engaged rows. a8 changed `pooled_frozen_loso` only.
- main's s1-snapshot `row_groups` behaviour is replaced by live classes. `_group_table` is kept for parity.
- B's / C's re-check numbers (bcdg_1104) are main-data numbers, not re-run here.

**Tests.**
- `tests/test_bcdg_1104.py` (a8's file, adapted):
  - g_group pins and the live `row_groups`;
  - stopped / early_run through `cue_groups_for`, including the uncovered-tail rule;
  - the contact score / projection centre;
  - D with a no-video PRE session: a post session without working rows raises in the rebuild.
- `tests/test_g4_camclass.py`: cam sample = g4common; contact z frame; groups, thin flags and pre-reference fallback;
  spec extra; lick-only working arm.

**(b) G9 by camera class (fecc69c9, its own commit; drop it to revert).** Priya approved this with guards.
- `position_coding_directions.g9_classes` / `WIDEFIELD_G9_CLASSES` (camera | daq).
- Under camera, each method record gains `camera_classes`:
  - classes: pre / post x the five groups, plus pre working;
  - per-spout CD scores (pre rows on their held-out direction);
  - n floor 10: thin cells are kept and flagged, never averaged;
  - `pre_reference_is`: "same" when the pre class has a cell, else "working";
  - `contrast` = post - matched pre (None for thin cells);
  - frame `z-contact`: rows re-z-scored per session on CONTACT rows.
- The DAQ classes are still written (`g9_daq_classes`). `g9_old_new()` and `scripts/g9_old_new.py` report old -> new
  per animal (counts and pooled CD scores).
- **No figure switches in this commit.** These stay on the DAQ classes until the report is read: grant_confusion 5b,
  miss_vs_stopped, position_axes, the deck G9 slides and scripts/best_match.py. Each needs a per-spout / thin-cell
  layout, so all are flagged.
- Camera-class CONFUSIONS (`_class_confusions` per camera class) are not built; only the CD scores are.

**Contact vs incomplete panel (d15fe9af, its own commit).**
- `wfield_local/contact_incomplete_panel.py`:
  - rows = cue / precue, columns = epochs; x = canonical spouts, contact | incomplete side by side;
  - open session dots (shape = animal, colour = epoch), no lines across epochs;
  - pre reference per class, else pre working (dotted, labelled); chance 1/6;
  - cells with n < 10 are OMITTED and counted ("n<10:k");
  - stars in the top band from `analysis_kit.boot_delta`; no top spine;
  - figdata + `--plot-only`.
- Data:
  - section G: `by_class.post`;
  - section D: the new `pooled_frozen_loso` `confusion_by_class` (scored working rows split into contact /
    incomplete).
- The centre is the producer's (printed tag).
- Test: test_contact_incomplete_panel.

**Wt per-spout dry count** (read-only, under the cohort1_rebuild profile; working cues per spout per curated session,
August pre). The imaging pool can drop a few window-edge cues, so these are upper bounds by a few rows. The table is
the minimum per animal x epoch; spout names are DAQ (L = ipsi, R = contra).

| animal | epoch | close_L | close_center | close_R | far_L | far_center | far_R |
|---|---|---|---|---|---|---|---|
| PS92 | pre | 36 | 41 | 29 | 47 | 33 | 38 |
| PS92 | acute | 52 | 55 | 53 | 49 | 47 | 46 |
| PS92 | subacute | 48 | 41 | 45 | 41 | 48 | 37 |
| PS92 | chronic | 42 | 41 | 44 | 45 | 40 | 42 |
| PS93 | pre | 73 | 84 | 80 | 79 | 76 | 80 |
| PS93 | acute | 53 | 52 | 60 | 45 | 49 | 43 |
| PS93 | subacute | 58 | 69 | 63 | 56 | 68 | 64 |
| PS93 | chronic | 69 | 68 | 70 | 70 | 71 | 68 |
| PS94 | pre | 57 | 49 | 54 | 48 | 52 | 55 |
| PS94 | acute | 43 | 51 | 48 | 39 | 34 | **21** |
| PS94 | subacute | 74 | 74 | 76 | 74 | 62 | 75 |
| PS94 | chronic | 86 | 81 | 84 | 82 | 85 | 84 |
| PS95 | pre | 65 | 74 | 81 | 71 | 66 | 73 |
| PS95 | acute | 119 | 113 | 113 | 106 | 53 | 41 |
| PS95 | subacute | 56 | 50 | 49 | 45 | 41 | 40 |
| PS95 | chronic | 57 | 72 | 63 | 59 | 56 | 62 |

- No session has a spout below 20. The tightest cell is PS94 acute far contra at 21, so a dropped edge cue or two could
  trip the refusal there.
- PS93_0813 has no working rows and stays unscored.
- Script: scratchpad `dry_count_wt.py` (outside the repo).

**cue70 merge (rebuild/cue70 read only, 3613cf0b / 395a9124).**
- `stopped_licking` (a lick-touched stop trial) gets its own group (`g_group_final`).
- Under A2, `_classify_guarded` marks the tail only internally (`tail = idx >= ts`); no run / tail marker leaves the
  function.
- **Needed from cue70:** a `stop_kind` per cue ("run" / "tail" / ""), exposed as
  `trial_classes.stop_kind_for(label, cue_s)`, memoised like `cue_classes_for`. With it, g_group_final sends a tail
  stop to "after_end" (labelled "quit") and a run stop to "stopped". Without it, stopped trials keep the raw-class
  route, and under A2 every post-last-lick trial would become none / jaw_only.
- `G_ALL = G_GROUPS + ("stopped", "stopped_licking")` is iterated by every by-class consumer.

**Commits (branch rebuild/bcdg-port, from 571b3955):**
- 724862f9: the port + (a) / (c) / (d).
- fecc69c9: (b) G9, separable.
- d15fe9af: the contact / incomplete panel, separable.
- 361c83bd: the centre kinds (Wt default) + the cue70 hooks.
- This log.

**Wt final (Priya 11-04).**
- `WIDEFIELD_SPOUT_SCORE_CENTRE=wt` and `WIDEFIELD_PROJECTION_CENTRE=wt` are now in
  `configs/render_profiles/cohort1_rebuild_20261103.yaml` (the code defaults were already wt). The run tag is
  unchanged.
- Robustness kinds: `n`, `cto`. The engagement axis / G4 stays z on contact rows.
- Method note:
  - ~-.024 drift under a high incomplete share (CENTRE_MIXSHIFT_2026-11-04 3d);
  - the rest-centred frame was tested and REJECTED (REST_CENTRE_2026-11-04: it absorbs the post-stroke rest-state
    change; chronic far contra ~.29 vs .72).
- **Open (for Priya / the coordinator):**
  - The projection FRAME itself is still OFF (`WIDEFIELD_PROJECTION_FRAME=session_z`): Wt only acts on CD / pattern
    rows once it is switched on. And `position_coding_directions` projects the engagement axis from the same rows, so
    "eax stays z" needs a per-readout split before the frame goes on.
  - The centre kind is in the scored-frame tags and memo keys but NOT in the run tag, so N / Cto robustness arms
    rendered under this profile would write into the primary folders. They need a run-tag part first, so no arm was
    added.

**Test record (361c83bd, 6 shards, one at a time):** 4,777 passed, 171 skipped, 0 failed. Related files during the
port: 379 passed. After the profile edit: profile / o2 / rebuild_root / 7ab / bcdg / switches: 108 passed.

### 8.49 Follow-ups to 8.48 (coordinator 11-04): C ON, scoring run-tag parts, G4 figures, joint D, G9 estimate

**1. C ON with Wt, plus a per-readout split (5c62a272).**
- `WIDEFIELD_PROJECTION_FRAME` now defaults to `spout_frame`; `session_z` is the old frame and carries the tag part
  `projz`. The projection centre is `wt`.
- **Readers, and the frame each gets:**

  | reader | frame |
  |---|---|
  | `position_coding_directions` directions / poles / pairwise / cross matrices / camera_classes | Wt |
  | its engagement axis (used to orthogonalise `_orth` directions and in the cos diagnostics) | z on CONTACT rows |
  | its frozen-decoder confusions (`coding_decoder`) | production rows |
  | `position_axes` pairwise axes | Wt |
  | `position_axes` `e_ax` | z on contact rows |
  | `attempt_coding` attempt axes | Wt |
  | `grant_kit._session_trials` (pattern / template families, `grant_similarity` x 2) | Wt |
  | `poststroke_compare.pattern_similarity` (`_proj_X`) | Wt |

  - The contact-row z frame comes from `spout_scaling.contact_z` / `working_trials.eax_rows`.
  - The production rows are the bundle's own XE / XU.
- **Lick window and DAQ gate:** these pools have no working masks, so they keep the engaged-row frame.
  `projection_work` REFUSES a camera cue / precue pool that lacks cue indices.
- **Frozen ids that change under C ON:** `coding_directions` (projected rows + the eax rows in the fingerprint) and
  `attempt_axes` (projected X).
  - `coding_decoder`, the spout decoders and the epoch decoders are unchanged.
  - Exact ids exist only after a render (content hashes).
- **Not done:** the stored `position_axes` references (none are frozen) and the grant pattern `_pre_reference` (not
  frozen) need no ids.

**2. Run-tag parts (bf5a6d69).**
- `run_mode.scoring_part` builds one output-tag part `sc-<a>+<b>`. It is empty for the default run, so the primary tag
  is unchanged. It covers:
  - `ctrN` / `ctrCto` / `ctrW` / `ctrC`: scored-frame centre;
  - `projz`, `pctr<K>`: projection frame / centre;
  - `dEng`: section D scored on engaged rows;
  - `gDaq`: section G grouped by DAQ;
  - `g4Daq`, `g4cWorking` / `g4cEngaged`: G4 disc / centre;
  - `g9Daq`: G9 classes;
  - `sdSpout` / `sdcEng` / `sdStatebal`: state-disc frame.
- It is not in the cache keys, because none of these change a per-session cached function.
- Profile arms:
  - `centreN` -> `..._stopRun6j_sc-ctrN+pctrN`;
  - `centreCto` -> `..._stopRun6j_sc-ctrCto+pctrCto`.

**3. G4 figures (8089793a).**
- `wfield_local/g4_figures.py`:
  - camera-class discriminator PRIMARY;
  - DAQ `fits_engaged` panel labelled ROBUSTNESS;
  - figdata + `--plot-only`; stars in the top band; no top spine.
- `section_g_figures` renders it before the DAQ G4 set, and the deck places it before G4 identity.
- **Bug fixed** in the 8.48 contact / incomplete panel: `T.align` is `DataFrame.align` (a method), so every filter came
  back empty. The test now asserts that the dots are drawn.

**4. joint_xsession section D.** No change needed. It calls `pooled_frozen_loso` with the decoder's
cue-index features (8.33), so it has scored the working rows since 724862f9, with the same `scored_set` /
`scored_centre` fields.

**5. G9 figures (not switched).** Estimate for the per-spout / thin-cell layouts, after the render and the
`g9_old_new.py` read:

| item | estimate |
|---|---|
| grant_confusion 5b class panels -> class x spout grid with the n floor | ~0.5 day |
| miss_vs_stopped -> per-spout matched contrast | ~0.5 day |
| position_axes post-class projections | ~0.5 day |
| deck G9 slides (text + placement) | ~0.5 day |
| `scripts/best_match.py` | ~2 h |
| tests | ~0.5 day |
| **total** | **~2.5-3 days**, plus camera-class confusions per spout (`_class_confusions` per group), ~0.5 day if wanted |

**Fix (1d59741c).** `grant_kit._proj_arrays` raised on a bundle with no working masks (DAQ gate). It now keeps the
engaged-row frame, as `projection_arrays` does. Two mask-bookkeeping tests use minimal bundles, so they pin
`WIDEFIELD_PROJECTION_FRAME=session_z`.

**Test record.**
- Full suite at 8089793a (6 shards, one at a time): 4,793 passed, 171 skipped, 5 failed. All 5 failures were the
  `_proj_arrays` case above.
- After 1d59741c, shards 00 and 05 (where they sat) were re-run: 1,636 + 578 passed.
- Total: **4,798 passed, 171 skipped, 0 failed**.

### 8.50 Merge of rebuild/bcdg-port into rebuild/cohort1-20261103 (11-04)

CODE + TESTS ONLY. rebuild/bcdg-port (81dbf4cc) merged with `git merge --no-ff` after 8.47's three merges. Its log
sections (8.44 / 8.45 on the branch) are 8.48 / 8.49; their internal cross-references were renumbered with them.

**Conflicts.**
- `configs/render_profiles/cohort1_rebuild_20261103.yaml`: both sides added arms. Kept cue70's (drift1500 on the new tag,
  stopA0, stopA3, stopStrict, stopTailEvent) and bcdg's centreN / centreCto, whose expected tags were computed on the old
  ...stopRun6j base and are now ...stopA2jexrt3500c70_sc-ctrN+pctrN / _sc-ctrCto+pctrCto. Every arm's tag, recomputed with
  `render_profile.apply(arm=...)` + `run_mode.tag()`, equals its expected_run_tag (primary, drift1500, stopA0, stopA3,
  stopStrict, stopTailEvent, centreN, centreCto). The primary env's SPOUT_SCORE_CENTRE / PROJECTION_CENTRE = wt (the default)
  adds no tag part.
- This log (sections appended on both sides).
- Auto-merged, checked: trial_classes (bcdg's `cue_groups_for` beside cue70's `stop_kind_for`), test_review_7ab
  (cue70's A2 profile env / tag, bcdg's names check in place of the byte-identity test), and the enl overlaps
  (attempt_coding, position_axes, position_coding_directions, precue_engagement_states: line 204's `enl_post_s` post_s
  default is enl's).

**Section G groups wired to cue70.** `working_trials.row_groups` calls `trial_classes.stop_kind_for` directly (was a
`getattr` hook for the then-missing function). `g_group_final`: stopped_licking -> its own group (run or tail); a
stopped trial of the quit tail (stop_kind "tail") -> "after_end" (captioned "quit"); of a mid-session run -> "stopped";
working trials keep their class. Test: test_stop_rule_a2::test_section_g_groups_split_run_tail_and_stopped_licking (a
synthetic session through the live classes: run of 7 with one lick-touched trial, tail of 3 with one).
  test_bcdg_1104's `_fake_session` also fakes `stop_kind_for` (its fake `cue_classes_for` had no `window` argument, which
  the real `stop_kind_for` passes).

**Test record (9fb2d964).** bcdg-related files: 213 passed. Full suite, 4 shards run one at a time (>= 38 GB free
before each): 4,923 passed, 173 skipped, 0 failed (shards: 926 + 28 skipped; 785 + 5; 2,251; 961 + 140).

### 8.51 FROZEN encoder: spontaneous lick kernels, design v7g3_t3d_spont (Priya 11-04; branch rebuild/frozen-licks off c149decf)

CODE + TESTS ONLY: no renders, nothing written to M:. Worktree `wf_rebuild_frozenlicks`. Smoke outputs in the session
scratchpad (`.../scratchpad/fl/`).

**Why.** The frozen encoder's v7g3_t3d design has post-cue lick kernels only (licks.csv starts at the cue), and
`split_blocks` refused LICKS_ALL. So in its all-frames maps, pre-cue (ENL) lick activity could be credited to the target
(pos_enl, whose ENL state spans strobe .. cue).

**Design v7g3_t3d_spont (`FROZEN_LICKS_ALL=split`).**
- Reuses the combined encoder; none of its logic is copied. `frozen_encoder.load_behaviour` / `raw_blocks` call
  `combined_encoding.load_behaviour` / `raw_blocks` with `COMBINED_ENCODING_LICKS_ALL=split` set ONLY around those calls
  (`run_mode.with_env`). Routing = `split_licks` (8.43): every lick in exactly ONE kernel.
  - `lick_onset` (the cued part of the encoder's `licks`) = licks.csv. That is every post-cue row, the 0-70 ms licks
    included: the 70 ms floor is for counting. It also takes the pre-cue / ITI-table rows with a post-cue onset, and
    cam1-only post-cue onsets.
  - `licks_pre` = licks_all pre-cue by the ONSET label (`oo.encoder_context`), plus cam1-only pre-cue.
  - `licks_iti` = licks_all ITI, plus cam1-only ITI. A cam1 onset within 50 ms of a cam4 pre-cue / ITI lick is dropped.
  - `contact` = DAQ contacts over the whole trial windows.
- Groups.
  - `lick_onset` / `contact` keep v7g3_t3d's groups; this is the encoder's instructed `licks`.
  - `licks_pre` / `licks_iti` are UNINSTRUCTED: the frozen group "other", the encoder's uninstructed set (nose,
    running, push). So `no_other` = drop:uninstructed.
  - Penalty group "lick" (= the encoder's PEN_GROUP_OF).
  - `split_blocks` checks these against `ce.SPLIT_LICK_VARS` / `PEN_GROUP_OF` at run time.
  - The new variables are not split into spout-locked / deviation parts.
- New variants:
  - `no_spont` drops both new variables (= drop:licks_pre+licks_iti). It is in the default variant set.
  - `novar_licks_pre` / `novar_licks_iti` are opt-in, like every novar_.
  - New PETH / evoked part "spont": a subset of "other", like spout / pos.
- Lick timing (Priya 11-04 rule 4): the first lick of each trial (template time axis, `first_f`, the lick window, the
  Musall lick window, trial_parts) = `combined_encoding.trial_first_lick`, factored out of `_TrialScorer`.
  - The scorer now calls it; its outputs are identical, covered by test_encoder_trials.
  - The rule: the first camera lick from the cue. At a close spout, an LP onset that is missing or more than 50 ms after
    the first DAQ contact (cue + 70 ms .. min(cue + max_rt, stop)) is replaced by that contact; at a far spout the LP
    onset is kept.
  - npz gets `trial_timing_source`; the json gets design / licks_all / spont_counts / lick_timing_sources.
- Ids, never shared with v7g3_t3d.
  - The saved statistics' run tag gains `|licksall-split`, so `--reuse-stats` / `--fit-from`, allsession_encoder and
    epoch_kernels refuse a v7g3_t3d fit.
  - The FRAME tag must carry a "spont" segment iff the design is split (`check_tag`). This is enforced in
    frozen_encoder (`--tag`, `--fit-from`, `--alphas-from`), frozen_relabel / allsession_encoder (`--from` and `--tag`),
    and partial_refit / session_vs_frozen / epoch_kernels (`--tag`).
  - frozen_relabel also refuses pre statistics of the other design.
  - `design_tag("...v7g3_t3d...")` inserts "_spont" after v7g3_t3d under split: gate-attempted_v7g3_t3d_spont,
    ..._spont_frA, ..._spont_basis, allsession_encoder_v7g3_t3d_spont_ref, frozen_relabel_v7g3_t3d_spont.
  - Folder name: "_ls" was not used because it already means FROZEN_LICKSPOUT (`_frA_ls`).
- Refused: FROZEN_DESIGN=te together with split (te routes the pre-cue / ITI licks itself; te runs need
  FROZEN_LICKS_ALL=none). An inherited COMBINED_ENCODING_LICKS_ALL is refused in both designs.
- Unset / "none" = v7g3_t3d, byte-identical (smoke below).

**Profile.**
- `cohort1_rebuild_20261103.yaml` env: `FROZEN_LICKS_ALL: split` (the rebuild default for frozen frames). It is not a
  run-tag part: the designs are kept apart by the frame tag.
- Arm `frozenNoSpont` (`FROZEN_LICKS_ALL: none`) = the labelled robustness row v7g3_t3d. Its own frame tag is
  gate-attempted_v7g3_t3d[...] with no "spont"; its run tag equals the primary's.
- ENCODER_RENDER_COMMANDS.md is NOT in this branch: it lives on analysis/attempt-coding (wf_attempt), not edited. Needed
  there:
  - frame tags `gate-attempted_v7g3_t3d` -> `gate-attempted_v7g3_t3d_spont` (+ the _frA / _basis / _ls / Layer 2 /
    relabel tags);
  - the te commands need `export FROZEN_LICKS_ALL=none`;
  - a robustness block = the same commands under `--arm frozenNoSpont` with the old tags.

**Frozen vs combined: trial / frame selection.**
- Consistent:
  - Trials = `T.engaged & finite cue` from the same `load_behaviour`. Under OROFACIAL_GATE=attempted, engaged = the run's
    camera-class trial_attempts (stopA2jexrt3500c70 tag), so the stop / quit rules are the camera classes'.
  - Frames = strobe - 0.5 .. stop + 1 s, as run_session.
  - Same drift basis, same lick sets / routing; the 0-70 ms licks are kept in the post-cue kernel.
- Differences, reported and not changed:
  1. The frozen encoder fits AND scores only the engaged trials. The combined encoder also scores every other cue trial
     (8.43 classes); there is no class-resolved frozen output.
  2. Locally, engaged comes from the tag's attempts table (`lp_licks.attempts_path`). O2 bundles use the shipped
     `COMBINED_ENCODING_ATTEMPTS` table. Same rule, but the copy can differ if the table is regenerated.
  3. The v7g3_t3d design keeps the LP-only first lick (= the combined session npz's peth_lick). Only the spont design
     applies the per-trial DAQ timing, so lick-window numbers of the two designs are not like-for-like (2 / 6 trials
     moved on PS93_0814 / 0818).
  4. Smoke PS93_0818: 478 fitted trials vs the combined encoder's 480 engaged (8.43). Not chased.
  5. FAR spout with no LP onset: the encoder rule (daq fallback) is used; `lp_licks.first_lick_timing` leaves such a
     trial untimed (the 8.47 open decision).
  6. The tongue / jaw traces stay instructed over the whole trial (as the encoder, 8.43 caveat).

**Readers.**
- Updated (frame-tag defaults / literals now go through `design_tag`, so under the profile they read the spont frames):
  contest_lines, early_predictors, epoch_kernels, exec_vs_neural_match, frozen_claims, frozen_fr_compare,
  frozen_fr_musall(_v2), frozen_kernel_maps, frozen_peth_trial_maps, frozen_posinfo, frozen_residual_significance,
  frozen_residual_trend, iti_licks, layer2_matched_n, layer2_random_splits, matched_residual, movement_null_plan_maps,
  partial_refit, rebuild_figs/{fig_layer2, fig_pooled, fig_resid16, fig_resid_per_animal, fig_resid_spout, fig_rsplit},
  relabel_confusion, remap_checks, remap_lmm, residual_roi, session_vs_frozen, simple_key_figures, specificity_control,
  spout_unique_maps, state_spout_check.
- Left alone: te_readout (te is never spont) and swapmask_compare_encoder (a historical adopted-run comparison).
- Group / variable lists:
  - frozen_pixel_unique_sig: + other / spont unique.
  - frozen_r2_maps: + spont unique, optional, so v7g3_t3d runs still draw.
  - frozen_pixel_musall: unique + no_other / no_spont; novar + licks_pre / licks_iti.
  - frozen_musall_figures: UNIQUE + spont (NaN on v7g3_t3d).
  - partial_refit `model_label` names the spont design.
  - curbd_pilot uses `ce.group_of()`: under LICKS_ALL=split the static GROUP_OF silently dropped licks_pre / iti.
- No reader crashed on the new variables. Every reader selects parts / columns by name, and FGROUP / FPEN come from the
  env. A frame read under the other env fails loudly (KeyError or the tag / run-tag guard).
- Meaning changes, not edited (captions to fix at render):
  - "other" now includes the spontaneous licks: frozen_peth_trial_maps, movement_null_plan_maps, frozen_posinfo
    (noposinfo).
  - Under split, `lick_onset` / `novar_lick_onset` = post-cue (cued) licks only, and contact = whole-window DAQ contacts:
    frozen_kernel_maps, epoch_kernels docstrings.
  - lp_encoder_decompose gains a d_spont column (overlaps "other").
- Fixed output folders shared by both arms (they overwrite each other if both are rendered in one run root):
  `E/frozen_fr_compare` (frozen_claims, frozen_fr_compare, frozen_fr_musall*, frozen_posinfo), `E/contest_lines`,
  `E/movement_null_plan`, `qc/frozen_group_leakage.csv`. They need a design suffix before rendering the robustness row.
- Combined-encoder readers (8.43 list, unchanged here): combined_encoding_figures VARS / GROUP_OF copies,
  lp_encoder_decompose docstring and its `drop_licks` folder (cued licks only under split), fr_musall /
  fr_contra_orofacial / fr_trial_peth captions ("uninstructed" now includes the spontaneous licks), lick_template_test
  (`R["licks"]` = cued).

**Smoke (PS93: 7 August pre sessions fitted, PS93_0814 pre = LOSO, PS93_0818 acute = all-pre fit).**
- Settings: rebuild profile + ENV-noFR (OROFACIAL_GATE=attempted, LICKS=combined, TONGUE=3d), `--jobs 2
  --no-drop-spout`, output to scratch only.
- Variants: joint, no_target / timing / lick / contact / other / task / movement; spont run also no_spont + novar_licks_pre / _iti.
- Reading the table:
  - Cortex R2, held out. unique = joint - no_<g>.
  - Windows: all frames / cue (cue .. + 1 s) / lick (first lick - 0.1 .. + 1 s).
  - target = pos_enl + pos_cue; task = timing + target.

| PS93_0814 (pre) | v7g3_t3d all / cue / lick | v7g3_t3d_spont all / cue / lick |
| --- | --- | --- |
| full | 0.6486 / 0.5618 / 0.4591 | 0.6587 / 0.5711 / 0.4726 |
| target (position) unique | 0.0071 / 0.0115 / 0.0192 | 0.0068 / 0.0126 / 0.0192 |
| timing (cue) unique | 0.0014 / 0.0020 / 0.0033 | 0.0021 / 0.0018 / 0.0040 |
| task unique | 0.0106 / 0.0215 / 0.0331 | 0.0107 / 0.0215 / 0.0333 |
| lick (cued deviation) unique | 0.0715 / 0.0295 / 0.0189 | 0.0515 / 0.0110 / 0.0070 |
| contact unique | 0.0017 / 0.0042 / 0.0063 | 0.0008 / 0.0010 / 0.0009 |
| other (uninstructed) unique | 0.3169 / 0.3996 / 0.4711 | 0.3238 / 0.4292 / 0.5151 |
| spont (licks_pre + licks_iti) unique | - | 0.0100 / 0.0238 / 0.0348 |
| licks_pre / licks_iti unique (all) | - | 0.0097 / 0.0013 |

| PS93_0818 (acute) | v7g3_t3d all / cue / lick | v7g3_t3d_spont all / cue / lick |
| --- | --- | --- |
| full | 0.4127 / 0.4184 / 0.2625 | 0.4335 / 0.4213 / 0.2577 |
| target (position) unique | -0.0022 / 0.0058 / -0.0255 | -0.0050 / 0.0044 / -0.0304 |
| timing (cue) unique | 0.0042 / 0.0032 / 0.0062 | 0.0042 / 0.0008 / 0.0084 |
| task unique | 0.0011 / -0.0003 / -0.0188 | -0.0009 / -0.0058 / -0.0212 |
| lick (cued deviation) unique | 0.0959 / 0.0950 / 0.1561 | 0.0441 / 0.0524 / 0.0905 |
| contact unique | 0.0013 / 0.0002 / 0.0007 | 0.0022 / 0.0011 / 0.0045 |
| other (uninstructed) unique | 0.1014 / 0.1440 / 0.2012 | 0.1216 / 0.1632 / 0.2361 |
| spont (licks_pre + licks_iti) unique | - | 0.0138 / 0.0045 / 0.0050 |
| licks_pre / licks_iti unique (all) | - | 0.0132 / 0.0018 |

- Kernel counts (every lick in one kernel):

  | session | post-cue | pre-cue | ITI | whole-window DAQ contacts | lick timing LP / DAQ |
  | --- | --- | --- | --- | --- | --- |
  | PS93_0814 | 6,032 | 3,508 | 879 | 6,936 | 507 / 2 |
  | PS93_0818 | 3,814 | 3,192 | 937 | 5,381 | 472 / 6 |

  Penalties are unchanged (task 1e5, lick 1e4, orofacial 1e3, uninstructed 100).
- Readout:
  - Full R2 rises 0.010 (pre) / 0.021 (acute) in all frames.
  - The spontaneous licks' unique R2 is ~0.01 (all frames), almost all pre-cue.
  - Position (target) unique barely moves, in all frames and the cue window: 0.0071 -> 0.0068 pre, -0.0022 -> -0.0050
    acute. So little pre-cue lick variance had been credited to the target in these two sessions.
  - What the spontaneous kernels take is mostly from the cued lick group's deviation part (lick unique 0.072 -> 0.052
    pre, 0.096 -> 0.044 acute): pre-cue / ITI licks had been absorbed by the post-cue lick_onset / tongue terms.
  - "other" grows because it now contains them.
  - The lick-window rows are not like-for-like (the spont design moves 2 / 6 trials to the DAQ contact).
- Runtime (template + stats + fit + predict 2), same box while the O2 bundle built:
  - spont 14 m 13 s (split behaviour cache cold);
  - v7g3_t3d new code 13 m 13 s; base code 12 m 57 s.
- Peak RAM, root + workers summed:
  - working set 7.2 GB, private 8.6 GB (spont); 9.3 / 10.9 GB (v7g3_t3d, new code);
  - minimum free on the box 10.0 / 6.3 GB.
- **v7g3_t3d reproduces exactly.** The same run on the base code (c149decf, detached scratch worktree) vs this branch
  with FROZEN_LICKS_ALL=none: 46 / 46 files, 214 items compared (every npz array, json, fit npz, pre_stats / templates
  pickles, csv), 0 differ.

**Tests.**
- New tests/test_frozen_spont_licks.py, 12 tests:
  - v7g3_t3d unchanged;
  - groups / penalties = the encoder's;
  - raw blocks route every lick like the encoder;
  - the v7g3_t3d raw blocks are byte-identical;
  - an inherited COMBINED_ENCODING_LICKS_ALL is refused;
  - no_spont / novar / no_other columns;
  - design_tag / check_tag, and main refuses a mismatched tag;
  - te + split refused;
  - first-lick timing in both designs;
  - profile default + arm;
  - pre-statistics design guard.
- Related files (frozen, encoder trials, licks_all, relabel, allsession, profile, cue70, LP-DAQ timing): 165 passed.
- Full suite, 4 shards run one at a time on 910c408b (>= 33 GB free before each): 4,933 passed, 173 skipped, 1 failed.
  - Shards: 1,959 + 4 skipped; 1,223 + 168; 890 + 1; 861 + 1 failed.
  - The failure: test_rebuild_root's writer classifier read frozen_relabel's new `run_mode` check as "routed". Fixed
    by moving the check into `frozen_encoder.stats_design_ok` (next commit).
  - test_rebuild_root + test_frozen_relabel + test_frozen_spont_licks then: 34 passed.

### 8.52 FROZEN encoder: per-trial, per-class outputs (Priya 11-05; branch rebuild/frozen-trials off 34425934)

CODE + TESTS ONLY: no renders, nothing written to M:. Worktree `wf_rebuild_frozenlicks`. Smoke outputs in the session
scratchpad (`.../scratchpad/fl/tr_new`, `tr_old`, `old_posthoc`).

**Why.** The frozen headline: the pre-stroke model predicts post-stroke activity worse. Acute sessions hold more
incomplete trials, so part of the drop could be class composition. This file gives trial-to-trial R2 by class within
the working trials: contact (cls "hit") vs "incomplete".

**The file.** Every session the frozen encoder scores (Allen-area targets: the areas and the frA frames; basis / grid
runs write none) gets `<run>/joint/sessions/<label>_trials.npz`, next to the session npz.
- Layout = the combined encoder's `<label>_trials.npz` (8.43). It is built by the combined encoder's own
  `_TrialScorer` (`trial_rows`, `trial_bins`, `_binsum`, `trial_first_lick`), so `wfield_local/encoder_trials` reads it.
  - `t_*` trial table: trial_id, pos, cue_s, stop_s, first_lick_s, timing_source, grooming_suspect, lp / daq times,
    rows.
  - `sse_bins_<cue|lick>` (models, trials, 0.5 s bins, areas), `sst_bins_*` / `nfr_bins_*`, `rd_mean` (the SST
    reference), `bin_edges_s`, `off`.
- Rows = the scored trials. That is the engaged = working trials (contact + incomplete under the camera gate).
  - Every row is a held-out prediction: pre sessions leave-one-session-out, post sessions the all-pre fit.
  - `t_frozen_from` = loso | all_pre; `t_heldout` = True.
- Classes: `t_cls` / `t_cls_raw` / `t_attempted` / `t_canonical`, joined (`encoder_trials.join_classes`) from the run
  tag's trial_attempts table (`lp_licks.attempts_path`). Trials without a row (or no table) = "missing".
- Lick alignment = `trial_first_lick` in both designs, as the encoder's file. The session npz keeps its own alignment.
- `tr_models`: "full" (= the joint model; the reader's name) + the unique-share variants. `tr_variants` = the frozen
  names.
  - Default (`TRIAL_VARIANTS`): joint, no_timing, no_target, no_task, no_lick, no_contact, no_other, no_movement, plus
    no_spont (spont design) and no_fr_all / no_execution where fitted.
  - `FROZEN_TRIAL_VARIANTS=all` adds no_exec_locked / no_exec_within / no_spout / task_only / no_lickspout / no_fr /
    no_fr_dir / no_fr_locked / no_posinfo / no_noposinfo. A comma list picks variants.
  - Unique share per class = joint - variant from the summed binned SSE.
- PER-CLASS TRIAL-TO-TRIAL R2 WITHOUT FRAMES. `sse_trial_bins_<align>` / `sst_trial_bins_<align>` (trials, bins,
  areas): the joint model's terms with the trial means removed within each spout x class (`trial_groups` "pos+cls";
  `trial_to_trial_bins`).
  - `encoder_trials.class_summary` uses them when the file has no frames. This is a thin branch; the frame path is
    unchanged.
  - Test: equal to the frame path (`class_full_fine`) for every class and both alignments.
- `FROZEN_TRIALS` = off | sse (default) | full (+ frames: data and the joint held-out "cv", float16, about +45 MB per
  session) | full32.
- Size, default sse, float32, 59 areas:
  - 8 variants (v7g3_t3d): 29.3 MB (PS93_0814, 509 trials) / 27.5 MB (PS93_0818, 478).
  - 9 variants (spont): 31.9 / 30.0 MB.
  - About 2.5 MB per variant plus 5 MB.
  - The combined encoder's file is 16-27 MB. The default set is kept to the headline shares for that reason; float16
    would halve it (checked: unique shares change by < 3e-5) but was not adopted, to keep the encoder's dtype.
- `load_trial_parts` skips the `_trials` files. No other frozen reader globs `sessions/*.npz` (they read the json).

**Post hoc (`--trials-only`).** An existing frame (e.g. v7g3_t3d, rendering now without the file) gets the files from its
OWN `fit/pre_stats.pkl` + `templates.pkl`.
- Fits = `fit_animal` from the saved Gram terms (deterministic: the same alphas and betas). Only the trial variants are
  fitted.
- Nothing else is written: no fit npz, no session npz / json, no drop-spout / Musall tables.
- It refuses missing / mismatched saved statistics (it never recomputes), basis targets, `--fit-from` / `--alphas-from`,
  and FROZEN_TRIALS=off.
- Smoke: on a copy of the earlier v7g3_t3d smoke frame (made by the pre-8.52 code), the post hoc files are BIT-IDENTICAL
  to the files of a full run on this code (2 files, 96 arrays). The frame's own 46 files stay unchanged.
- Cost: 1 m 25 s for PS93 (8 trial variants, 2 sessions, `--jobs 2`), peak 6.1 GB working set (2 workers). About 40-60 s
  per session per worker, dominated by `session_design` (imaging + behaviour loads from the input cache).
- Estimate: a full frame (107 sessions) at 4 workers is about 25-35 min, ~3 GB per worker. The frA frame is similar plus
  the FR load. Not run on a real frame.

**Existing outputs byte-identical** (smoke PS93_0814 / 0818, 7 August pre sessions, as 8.51; this code vs the 8.51 smoke
outputs, `_trials` files excluded):
- v7g3_t3d: 46 / 46 files, 214 items, 0 differ.
- v7g3_t3d_spont: 61 / 61 files, 314 items, 0 differ.
- The only new file is `<label>_trials.npz`.
- Runtime with the file: spont 10 m 46 s, v7g3_t3d 8 m 55 s (`--jobs 2`; caches warm). Peak 10.4 GB working set (spont,
  2 workers).

**Smoke class table** (joint model, cortex, held out, cue-aligned -3 .. +3 s; lick-aligned in brackets). cvR2 = against
the session mean, as `sst`; trial R2 = trial-to-trial, spout x class means removed.

| session | design | class | n | cvR2 | trial-to-trial R2 |
| --- | --- | --- | --- | --- | --- |
| PS93_0814 pre (LOSO) | v7g3_t3d | contact | 485 | 0.658 (0.653) | 0.539 (0.508) |
| | | incomplete | 24 | 0.675 (0.686) | 0.714 (0.686) |
| | v7g3_t3d_spont | contact | 485 | 0.661 (0.656) | 0.543 (0.513) |
| | | incomplete | 24 | 0.699 (0.711) | 0.734 (0.711) |
| PS93_0818 acute (all-pre) | v7g3_t3d | contact | 380 | 0.469 (0.474) | 0.387 (0.351) |
| | | incomplete | 98 | 0.367 (0.422) | 0.369 (0.363) |
| | v7g3_t3d_spont | contact | 380 | 0.480 (0.482) | 0.399 (0.360) |
| | | incomplete | 98 | 0.358 (0.416) | 0.361 (0.359) |

- Acute, one session: the contact trials alone already fall from ~0.54 to ~0.39 trial-to-trial (0.66 -> 0.47 cvR2).
  Incomplete trials are not predicted worse than contact trials trial-to-trial (0.37 vs 0.39), but they are on cvR2
  (0.37 vs 0.47).
- Pre has only 24 incomplete trials.
- One animal, one session pair: composition can only be judged on the full frame.

**Tests.**
- New tests/test_frozen_trials.py, 7 tests:
  - layout and binned SSE = the encoder's definition;
  - the reader's class_summary on a frozen file;
  - sse mode has no frames but gives the per-class trial R2;
  - the binned per-class trial R2 = the frame path;
  - FROZEN_TRIALS env;
  - --trials-only refuses without saved statistics and for basis targets;
  - load_trial_parts skips trials files.
- Related (frozen trials / spont / encoder, encoder trials, licks_all, relabel, allsession, rebuild_root): 92 passed.
- Full suite, 4 shards one at a time on 984116bd (>= 44 GB free before each): 4,942 passed, 173 skipped, 0 failed
  (1,978 + 5 skipped; 1,212 + 167; 890 + 1; 862).

### 8.53 Imaging licks = THE behaviour DAQ lick detector (Priya 11-04/05; branch rebuild/daq-lick-detector off 34425934)

CODE + TESTS ONLY: no renders, nothing written to M:. Worktree `wf_rebuild_lickdet`. Scratch outputs (onset counts,
trial_attempts preview) in the session scratchpad (`.../scratchpad/lickdet/`).

**Why.** LICK_DETECTOR_COMPARE_2026-11-04 (widefield_pipeline main a69fb9c3): the imaging profile
`(2.5, 1.0, (0.001, 0.020), 0.10)` and the behaviour profile (yaml `lick_detection`: 2.5 / 0.5 V, 40 ms floor) differ
ONLY by imaging's 0.10 s refractory (the 1.0 vs 0.5 V offset moves 0 onsets in 16 sessions). The refractory drops 0.58 %
of licks, 94 % of them inside a camera lick (real 10-11 Hz licking), unevenly (PS95 1.4 %). 0 hits change; 0 RTs as
imaging computes them (10 / 6,520 with a cue + 70 ms floor). Priya approved switching imaging before the render.

**One accessor.** `config.lick_detector()` = the yaml profile as `{thresh_upper, thresh_lower, lockout_s,
refractory_s = 0.0, min_ili_s}`; `config.lick_detector_args()` = the positional 4-tuple; `config.lick_min_ili_s()` =
the floor. `plot_lick_aligned_averages._load_daq_events(h5, channel, thresh_upper=None, ..., refractory_s=None)`: every
detector parameter defaults to the accessor, so a call site passes only `(h5, "lick_analog")`. `daq_io.lick_onsets`
applies the floor (`config.lick_min_ili_s`). Its `lick_detection` record now also carries `min_ili_s`.

**Call sites changed (47 literal calls in 39 files; each now `_load_daq_events(h5, "lick_analog")`, gate_source keeps
`allow_lp=False`).**
- wfield_local (18 files / 26 calls): analysis_kit.lick_samples, filter_acausality_test (2), fixed_scale_maps,
  gate_source (`_daq_contact_samples`), hemo_variants, locanmf_crossanimal_dff, locanmf_cue_lick_analysis,
  locanmf_decoder_weights (4), locanmf_dff_by_position, locanmf_encoding_model, locanmf_firstlick_aligned,
  locanmf_position_decoder (`_trial_features`), locanmf_position_encoder (3), locomotor_features, nolick_decoder,
  precue_lickfree (2), precue_window_sweep (2), rt_drift.
- scripts (21 files / 21 calls): cohort2_qc/{cohort1_impact, compare_hemo_variants_1029, epileptiform_gallery,
  epileptiform_videos, indicator_compare, indicator_kinetics_spread, native_maps_1029, prestroke_quiet_videos,
  quiet_modes, running_sign_check, seizure_events_video, spout_compare_1029, spout_maps}, drift_check/drift_cutoff_check,
  enl_lick_rates, enl_lp_census, gate_lp_check, rest_migration/{channel_position_maps, evoked_hrf_latency, nvc_evoked,
  plot_session_residual}.
- CLI defaults `--refractory-s 0.10` -> `config.lick_detector()["refractory_s"]` (0.0; thresholds / lockout were already
  yaml): framemap_event_maps, locanmf_lick_aligned, plot_frame_alignment_comparison, plot_lick_aligned_averages,
  quiet_periods, roi_activity.
- Other hard-coded 0.10: rest_migration/relaxed_buffer_percentile (detect_licks refractory) -> the accessor;
  camera_coverage_tables `REFRACTORY_S` (the "refractory_merge" cause) -> max(refractory, floor) = 40 ms (its test moved
  the synthetic lick to 1 frame before the cue).
- Left alone, deliberately: scripts/archive/lick_bout_decoder.py, scripts/rest_migration/archive/{stopped_tail_contamination,
  worktrunc_result_impact}.py (archived; they reproduce archived results as they ran). `analysis_kit.lick_count_samples`
  (already the behaviour detector via `daq_trials.decode`; it stays the DAQ train under an LP lick source). The
  behaviour-side readers (behavior_events, daq_trials, dlc_frames, dlc_context_frames, contact_classes, spout_behavior,
  restnl_mask, rest_by_position, restw_strict_check) already used the yaml profile.

**min_ili fallback.** `config.LICK_MIN_ILI_MS_DEFAULT = 40` is the one fallback when the yaml has no `min_ili_ms`:
behavior_events / restnl_mask / rest_by_position / restw_strict_check / position_survival fell back to 0, daq_trials /
dlc_context_frames / spout_behavior to 40 (spout_behavior, quiet_periods, plot_frame_alignment_comparison now call
`config.lick_min_ili_s()`). The yaml sets 40, so no output moves; a missing key no longer splits the paths.

**Caches and run tag.**
- `session_cache.CACHE_VERSION` 13 -> 14 (`_trial_features`: lick-free pre-cue windows, lick-aligned events; every
  cached kind; map_collect's map kinds go through it).
- `gate_source.GATE_VERSION` 1 -> 2 (DAQ contact train of the LP-inclusive gate).
- `scripts/disengagement_tests.BEH_VERSION` 2 -> 3, `scripts/gate_lp_check.VERSION` 1 -> 2 (their own pickles).
- Not bumped, checked: `lp_licks.LP_VERSION` (its DAQ contact fallback = `dlc_frames.lick_onsets` = `daq_trials.decode`
  with the yaml profile, unchanged); the WFIELD_INPUT_CACHE encoder inputs (behaviour inputs use DAQ contacts from
  dlc_frames; their key hashes daq_trials.py, whose edit re-keys them once anyway); `dlc_frames` lick cache (params
  digest of the yaml, unchanged); behavior_events SCHEMA_VERSION, RESTnl / restw caches (yaml profile already);
  trial_attempts / qc tables (keyed by the SELECTION tag, which does not change: they never read the imaging detector,
  see the preview below); frozen stores (specs fingerprint the training data, which moves wherever the lick train
  does).
- RUN TAG: new `run_mode` part `lickdet` = `daqlk` (`config.LICK_DETECTOR_TAG`), placed right after the selection part,
  in output tags AND cache parts, present iff any other output part is (the default untagged run stays byte-identical;
  its per-session caches moved with v14). `_TAG_PREFIXES` gains "daqlk". New rebuild run tag:
  `hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_pre0801_sel-camera-cue_stop-endlick-cls-contact-incomplete-stopA2jexrt3500c70_daqlk`
  (cache tag `hv-..._lnmfproj_sel-...-stopA2jexrt3500c70_daqlk_csig-<sha>`); every arm's `expected_run_tag` in
  `cohort1_rebuild_20261103.yaml` gains `_daqlk` (before `_sc-...` in the centre arms). An O2 bundle recorded under the
  old tag (enc_rebuild_20261104b) is refused by `o2_bundle.verify` under this code.

**Onset counts, before (old imaging profile) / after (THE DAQ detector).** `tests/fixtures/lick_detector_onsets_20261105.csv`,
17 sessions: the comparison's 16 (every animal x pre / acute / subacute / chronic; all 16 equal its
task1_per_session.csv n_IMG / n_BEH exactly; per animal 30,683 / 30,785, 27,395 / 27,542, 24,080 / 24,139,
19,658 / 19,940; total 101,816 / 102,406) + PS93_0908 (7,910 / 8,023). Largest: PS95_0917 +117 (6,997 -> 7,114),
PS95_0811 +99 (2.1 %); PS94_0819 is the one session with an onset the old detector had and the new lacks (+7 / -1).

**trial_attempts preview (scratch only; an audit hook refused any write on M: / N:).** PS95_0917, PS93_0908, PS94_0819
(1,711 trials) rebuilt by `scripts/trial_attempts.session_trials` under the rebuild profile with this code, with the
34425934 code, and compared with the current `o2/enc_rebuild_20261104b/tables/trial_attempts.csv`: 0 class changes
(cls / cls_raw / attempted / canonical / stop_kind), 0 first_lick_ms (RT) changes, 0 timing_source changes, 0 contact /
n_licks changes in either comparison. Expected: every DAQ contact in the attempts path comes from `dlc_frames.lick_onsets`
(the behaviour detector already).

**Tests.**
- New tests/test_lick_detector_daq.py, 12 tests: the accessor = yaml; one 40 ms min_ili fallback and no reader with its
  own; no imaging literal / 0.1 CLI refractory left outside archive/; `_load_daq_events` defaults keep two licks 95 ms
  apart and a re-touch 50 ms later (old detector dropped both) and drop a 30 ms double; `analysis_kit.lick_samples`
  passes only `(h5, "lick_analog")`; the cache bumps; `daqlk` in every tagged run, never the default, right after the
  selection part; profile tags; the fixture = the comparison doc's per-animal table; real-session counts on PS95_0917
  and PS94_0819.
- Updated pins: test_licks_per_trial (lick_samples), test_disengagement_detectors (the two trains are now the same),
  test_priya_decisions_1103 (refractory_merge at 40 ms), and the tag strings in test_rebuild_switches,
  test_camera_gate_fail_closed, test_review_7ab, test_stop_rule_a2, test_o2_encoding_env.
- Related files: 49 + 385 passed. Full suite on cd35e594, 4 shards one at a time (>= 31 GB free before each):
  4,946 passed, 173 skipped, 0 failed (1,078 + 27 skipped; 772 + 5; 2,144 + 1; 952 + 140).

**Open (for Priya / the coordinator).**
1. The O2 encoder (combined_encoding / frozen_encoder) never reads the imaging detector (DAQ contacts come from
   dlc_frames = the behaviour detector), so its outputs do not change, but the new run tag makes `o2_bundle.verify`
   refuse enc_rebuild_20261104b and routes encoder outputs to the `_daqlk` folder. Re-run it, or exempt the encoder?
2. Preprocessing-side lick-aligned products (imaging box: `plot_lick_aligned_averages` / `locanmf_lick_aligned` CLIs ->
   `lick_aligned_affine8v1`, `locanmf_lick_aligned_affine8v1/*.npz`, read by locanmf_lick_pool / cards / event_master /
   contralateral) carry no detector tag; existing files are the 0.10 s ones. Regenerate, or leave as legacy?
3. The default (untagged) run has no `daqlk` part: its skip-if-exists outputs made under the old detector are not
   separated (per-session caches are, by v14).
4. cohort2_qc scripts were switched too ("everywhere"); confirm cohort 2 follows.

### 8.54 O2 bundles recorded without `daqlk` are accepted (coordinator 11-05; branch rebuild/daqlk-bundle-accept off 90b7ddd5)

CODE + TESTS ONLY: nothing written to M:. Scratch: `.../scratchpad/lickdet/` (enc_inputs/, a separate enc_cache/).

**Definitive check (8.53 Q1).** Bundle enc_rebuild_20261104b (code c149decf, old imaging detector, run tag without
`daqlk`) vs its inputs rebuilt under 90b7ddd5:
- How: `scripts/o2_encoding._inputs_worker` (the bundle path), env = the bundle's env.json (@tables read from the bundle),
  `o2_encoding.apply_profile("cohort1_rebuild_20261103")`, cams cam1,cam4, FR master_pre_v1, a SEPARATE scratch
  WFIELD_INPUT_CACHE, WIDEFIELD_NO_CACHE=1, an audit hook refusing any write on M: / N:. The attempts table read is
  the one the bundle recorded (sha256 bef06b0a..., unchanged). Identity computed: the 8.53 tag with `_daqlk`, the same
  selection, area basis splitr8.
- Result, array by array (every ndarray / DataFrame / Series / scalar of the input dict and the Behaviour object):
  PS95_0917 342 leaves, PS94_0819 342, PS93_0814 444 (video): **0 differences**. PS95_0917 and PS94_0819 are
  byte-identical files. PS93_0814 is 253 bytes shorter: the pickle memo of numpy dtype objects (the bundle pickled a
  fresh `dtype('i8' / 'f8' / 'b1')` per array where the rebuild re-used one); every array buffer is the same, and the two
  loaded objects re-pickle to identical bytes.

**Acceptance.** `o2_bundle.NON_INPUT_TAG_PARTS = ("daqlk",)`: an explicit, documented list of run-tag parts that are not
encoder inputs (each entry needs a check like the one above). `o2_bundle.verify` accepts a recorded identity whose
selection and area basis match exactly and whose run tag equals the current tag with exactly those parts removed
(`tag_without_non_input_parts`, by `run_mode.parts()` names, no prefix / fuzzy match; one way only: a daqlk-tagged
bundle under a run without daqlk still refuses). It returns the CURRENT identity + `recorded_run_tag` /
`accepted_parts` and prints the acceptance. `check` / `collect` (via `_refuse_other_run`) and a refreshed-code runner
use the same rule. `collect` routes to `combined_encoding.out_dir` under the current profile, i.e. the `_daqlk` encoding
folder, and writes `collected_<bundle>__<TAG>.json` (recorded / current tag, accepted parts) beside `sessions/`.
On O2 nothing changes for a resubmit: the bundled c149decf runner computes the recorded tag itself.

**Tests.** New tests/test_o2_bundle_non_input_parts.py (12): the list is exactly ("daqlk",); the current tag = bundle
b's + `_daqlk`; exact identity unchanged; the daqlk-less bundle accepted (verify + `_refuse_other_run`); refused when
another part is missing, the stop rule differs, `_daqlkx` / `_daql`, the selection or the area basis differs; one-way;
collect into the `_daqlk` folder with the provenance json. With test_o2_encoding_env: 25 passed.
- Full suite on 23726438 (= 90b7ddd5 + this), 4 shards one at a time (27-41 GB free before each; frozen lanes running):
  4,965 passed, 173 skipped, 0 failed (935 + 28 skipped; 813 + 5; 2,247; 970 + 140).

### 8.55 `o2_encoding collect --move` (Priya 11-05: moving our derived outputs within her M: folder is ok)
- `collect --move` moves `out/<TAG>/sessions/*` into the routed run folder (`os.replace`, a rename on the same share; a
  cross-volume move falls back to copy + delete) instead of copying, so a ~40 GB run is not stored twice.
- Everything else is unchanged: identity check first (`_refuse_other_run`, incl. the 8.54 `daqlk` acceptance), the JSON
  trailing-byte repair (now on the moved file), the provenance json (+ `"moved": true`). The bundle's `out/<TAG>/` keeps
  `run.json` and gets `MOVED_TO.txt` (destination + file count); its `sessions/` is left empty.
- Default (no flag) still copies. Test: `test_collect_move_leaves_no_second_copy`; O2 tests 26 passed, ruff clean.
- First use: the bundle-b runs after main / main_nolicksall (those two were copied; the bundle copies are deleted after a
  full hash check, approved by Priya 11-05).
