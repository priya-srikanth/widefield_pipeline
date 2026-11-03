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
