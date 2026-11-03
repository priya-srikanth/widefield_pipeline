# Independent review of the cohort-1 rebuild code (2026-11-03)

**Tip reviewed: `85af6485`** on `rebuild/cohort1-20261103`, range `origin/main..85af6485`:
- merges 89664325, 2b182c76, 22959cec, 6622a3c3, db711802, 4b1919f0;
- work commits 250b801e, 509cd2bd, 335f23f3, ce5ecf6a, e947bf1c, d919eb55, plus the doc commits e93911c1 and 85af6485.

**NOT reviewed:** commits that landed while the review ran, 734226d4, **3988cbe3** (the attempt-coding merge) and **c17be8f5** (wf_attempt-only scripts follow the hemo switch). Review them before a render. In particular:
- the five conflicted wfield_local files of that merge: lp_licks, session_cache, frozen_encoder, map_collect, sig_marks;
- the iti_* scripts' frozen_readouts / run-mode routing.

**Method:** code was read from clean `git archive` exports. During the review, the wf_rebuild working tree was in the middle of the attempt-coding trial merge (UU / AA paths), so nothing was imported from it. I had three read-only helpers, each limited to one process. My own checks:
- tests on a clean export of 85af6485;
- four reproductions written to scratch.

I made no code changes and pushed nothing. The basis is the author's log (`docs/status/REBUILD_CODE_20261103.md`) plus the decisions and traps in wf_attempt `COHORT1_SWAPMASK_RERUN_20261102.md` and `rebuild_20261103/RECONCILIATION.md`.

## Findings, by severity

### HIGH

**H1. The attempt_axes DAQ cross-check is stored under the constant fingerprint `"from-daq-run"`, so for that kind the fingerprint stops protecting the model.**
- Where: `wfield_local/attempt_coding.py:122-126` together with `frozen_models.py:220` (make_spec) and `:467-491` (load_or_fit).
- Scenario, under `WIDEFIELD_LICK_SOURCE=lp WIDEFIELD_LP_FROZEN=daq`:
  1. make_spec sets `train_fp = DAQ_RUN_FP`.
  2. stored_axes then sets `spec["lick_source"]="lp"`, which skips the "a DAQ cross-check must never fit" refusal.
  3. On a miss (and with no LP-default model for find_daq_trained), the model is fitted on LP rows and `_write`n with `train_fp="from-daq-run"`.
  4. Every later cross-check HITs it through `find()`, even after the video-hit rows change.
- iti_readout has the same pattern but is safe, because `data_sig` still carries the data hash.
- Fix: when a caller forces lick_source=lp, make_spec must fingerprint the real data (e.g. a `lp_only=True` argument). Or add `spec["data_fp"]=data_fingerprint(...)` in stored_axes.

**H2. Local outputs (`figures_working`) are never tag-checked, so tagged runs overwrite default-named PNGs that the deck reads.**
- Where: `writeguard.py:88-96`. The tag check runs only for MICROSCOPE / standby paths, and only when a caller calls `assert_writable`. On this box `figures_working` = `C:/Users/SabatiniLab/cue_lick`.
- Probe: under `WIDEFIELD_HEMO_VARIANT=<variant>`, `is_writable(".../cue_lick/coding_direction_cue_logistic_PS92.png")` is True.
- Writers with no routing: `position_coding_directions` PNGs (`:1169`, `:1272`, out `:2021`; this is a camera-wired family), `section_g_figures` (`:271-274`), `miss_vs_stopped` (`:132,143`), `precue_engagement_states` (`:427-452`; its `precue_states_<align>.json` is also not keyed), `position_axes` (`:429-457`), hemispheric_dynamics / intensity, pixel_rsa, spatial_reorganisation, fixed_scale_maps, evoked_amplitude, rt_drift, vessel_contrast, enl_decode_figure, enl_sparsity_figure.
- Only the fixed-name JSONs were keyed.
- Fix: route each default `out` through `lp_licks.output_dir`, or `run_mode.require_default()` in their `main()`. Extend the guard's tag check to `figures_working`, and add a guard test on a local path.

**H3. Share writers with no routing and no guard, including skip-if-exists files (T18).**
- Where: `attempt_maps.out_dir` (`:197`) and `attempt_coding.out_dir` (`:365`) build `lp_licks._run_dir()/imaging_encoding/...` directly.
- `attempt_maps` skips an existing `maps_{align}_{an}.npz` (`:353-356`). `attempt_coding` reuses `trials_{align}_{an}.csv.gz` (`:387-389`).
- Scenario: a variant, lnmfproj or camera run serves the adopted npz / csv, then rewrites summary, contrasts, figures and run.json in the default folder.
- The log's statement that the unwired families' "outputs ARE tagged sel-camera" is false for attempt_coding, attempt_maps, precue_engagement_states and position_axes.
- Fix: wrap both in `lp_licks.output_dir` / `run_mode.subdir` and call `assert_writable`.

**H4. Camera-gate caches do not key on the camera class inputs or on the trial_classes code (stale caches).**
- What is right: `effective_kind` adds `sel-camera-<window>`, so camera and DAQ entries never collide.
- The gap: the classes are built from v36 `trials.csv` / `licks.csv` / `windows.npz`, `cam1_licks/*_events.csv` and `imaging_encoding/qc/jaw_onsets.csv` (`trial_classes.session_trials`). None of these enters `session_signature`, the tag, or the bm / prm / tf digests.
- No `trial_classes` function is in `_raw_maps_code` or `_session_maps_code`; 32 of its names are reachable from the map builders.
- Compare `lp_licks.cache_tag`, which carries the attempts table and a cam1 signature for exactly this reason.
- Scenario: jaw_onsets or cam1 are regenerated, or `camera_rows` is fixed, and warm `bm-*/prm-*/tf-*-sel-camera-*` entries are served with no warning.
- Fix: add a signature of the class sources to `run_selection_tag()` under camera. Add `map_collect.closure(trial_classes, "camera_rows", "row_classes", "camera_select", "cue_classes_for", "session_trials")` to both digests.

### MED

**M1. Controls run with `WIDEFIELD_JOINT_PROJECT_ALL=1` on the default variant are served caches built from stored C.**
- Where: `run_mode.py:84-93`. The cache parts are hemo, lnmf and selection; projall is not one of them.
- Kinds built from `basis.basis_id` rather than `BasisSource.key` (the only key carrying `:projall`): `cd_trajectories.py:667, 905, 937, 1209, 1260`, `enl_decode.py:238`, `scripts/rest_migration/transfer_arms.py:120`.
- Scenario: the "adopted control of the same method" silently equals the default. Its frozen specs fingerprint those cached features, so stored models are served as well.
- Under a non-default variant this is safe, because the hv- tag separates the caches.
- Fix: add a `projall` run-mode part (cache and output), or use `BasisSource.key` everywhere.

**M2. Under `WIDEFIELD_LOCANMF_PROJECT=1`, the LocaNMF poller / nightly can refit over the adopted per-session decomposition.**
- Where: `await_locanmf.py:130-150` and `nightly_figs.py:326,563`. Under projection, `config.locanmf_dir` keeps the adopted name while `svtcorr_path` points at the variant.
- Scenario: the stale check sees a newer SVTcorr, and `batch_locanmf` fits the variant data into `locanmf_affine8v1_hemo_meegkit_hpfit`, moving the old fit aside.
- Fix: refuse (`run_mode.require_default`) in await_locanmf / batch_locanmf / nightly when `locanmf_projection_on()`.

**M3. Fixed-name JSON keys ignore the LP, ENL and gate-LP modes.**
- Where: `run_mode.keyed_name` (`run_mode.py:121-125`) uses `run_mode.tag()`, not `lp_licks.output_tag()`.
- Probe: `WIDEFIELD_LICK_SOURCE=lp` gives `section_g.json`; `WIDEFIELD_GATE_SOURCE=lp` gives `coding_direction.json`.
- Scenario: an LP section-G or coding run overwrites the default JSON, and the next default grant 3a / 3b / 5 / 5b render draws LP data. T21 is only partly fixed.
- `run_mode.subdir` (iti_licks OUT) has the same gap. Under LP + a variant it also builds a segment that writeguard rejects (loud, but inconsistent).
- Fix: key both on `lp_licks.output_tag()`.

**M4. The narrowed prm digest dropped a dependency that origin/main covered.**
- origin/main hashed the whole `quiet_periods` module. At the tip, `rest_by_position:158,179` calls `quiet_periods.response_window_s` (the REST-frame engagement window), which is outside every closure, because `closure()` follows same-module names only.
- `rest_mask`, `trial_exclusion` and `widen_bool_sparse` are also reached. They are lower risk, since they sit behind the events cache.
- The only digest test (`test_map_digest_closure_is_narrow`) checks narrowness, not sufficiency.
- Fix: add these roots, and add a cross-module call-graph test.

**M5. More unwired DAQ-rule gates than the log lists.**
- These would run mislabelled under `gate=camera`:
  - `position_axes.py:248-259` (miss / stopped from `_gate_all`; untagged)
  - `enl_states.states_for:126` (`scripts/enl_state_counts`)
  - `scripts/rest_migration/channel_position_maps.py:162` (tagged sel-camera, but on the DAQ quit rule)
  - `rest_frozen_decoder` / `rest_position_decode` / `shared_position_projection` (`engaged_by_cue`)
- These are in addition to the ones the log already lists: nolick_decoder.categorize, cd_trajectories, attempt_coding / maps, poststroke_engagement, precue_engagement_states, rest_engagement / rest_by_position.
- Fix: a `require_gate("daq")` helper that raises under camera, called in each of these, until they are wired. A tag does not protect the untagged ones (H2 / H3).

**M6. Camera `miss_working` includes contact-class rows, which contradicts its docstring.**
- Where: `trial_classes.camera_rows` "miss_working" (`:353`) and `position_coding_directions` `nl_working = in_gate(_cu, "working")` (`:782-785`). Both use GATES["working"], which includes `contact`. The docstring says "responded but no contact".
- Scenario: a trial with no DAQ lick within max_rt but a camera "contact" (cue+70 ms to stop) is counted as miss-while-working.
- Fix: use `{incomplete, jaw_only}`, or decide it with Priya and count such rows. The test covers only no-contact rows.

**M7. `find_daq_trained` ignores `selection`.**
- Where: `frozen_models.py:148`, `_DAQ_LOOKUP_IGNORES = ("train_fp", "selection")`.
- Scenario: a camera-gate LP cross-check silently loads a stored DAQ model trained under `sel-daq-quittail`.
- hemo_variant IS matched (it is in base_id).
- `WIDEFIELD_GATE_SOURCE=lp` is not in the selection tag at all.
- Fix: stop ignoring `selection`, and add the gate-source tag to `run_selection()`.

**M8. In the LP DAQ cross-check, 5rm (`matched_frozen`) can drop whole animals.**
- Where: `grant_confusion.py:746-884`.
- `mm_old` uses `find()` only. That always misses (train_fp is DAQ_RUN_FP), so the draws are fitted in memory on LP rows.
- `load_or_fit(mm_spec, ...)` then either returns a DAQ payload that is discarded, or raises "loads DAQ-frozen models". The outer `except Exception` turns the raise into a "!! 5c" line and drops the animal from 5r / 5rm.
- PS93 always takes the raise path: train labels are not overridden to the DAQ pre set, and PS93_0813 has no video.
- Fix: under `daq_trained_in_lp`, skip the matched store (in memory, logged), or use the lookup's hit.

**M9. The branch is behind origin/main on the state decoder, and the recreated Fig 3.4 input has already moved to main's decoder.**
- origin/main has 15 commits past the merge base 11e8a846, including 9547a773 ("bal_em is the PRODUCTION frozen state decoder") and af638910.
- Main's `make_spec` has `scaling_params`; main's `locomotor_decoder.py:270-272` calls make_spec WITHOUT `train_data`, so it raises TypeError once merged here. DECISIONS fa70b651, cited in FROZEN_MODELS §8 and in a test, exists only on main.
- Measured: the share sidecar `labcams/grant_figures/epoch/data/epoch_13_state_decoder_cue_sessions.csv` was rewritten 11-03 11:05 (bal_em). As a result, `scripts/rebuild_figs/fig_state_position_sessions.py`:
  - now reproduces `build_inputs/state_position_sessions_balem.{csv,png}` exactly;
  - no longer reproduces `state_position_sessions.{csv,png}` (11-02 21:39, the image its docstring calls the page image): CSV max |diff| 0.307, 1.2 % of pixels differ.
  - Its docstring still describes the input as the "11-02 per-session z-scored re-render".
- Meanwhile the rebuild's own state spec (`locomotor_decoder.py:111`) uses session_z.
- Fix: merge main before G3. The merged state spec should keep `scaling` + `scaling_params` and pass `train_data`. Then fix the 3.4 provenance text (and say which image is the page's).

**M10. Moving established non-default outputs leaves hard-coded readers stale.**
- `lp_final_render.setup` sets PRE_FROM=0801 and MAP_REFS, so its output moves to `LPfinal/LP_licks_gateLP_pre0801_refs-.../epoch`.
- `scripts/lp_final_compare.py:42-43` still reads `LP_licks_gateLP/epoch/...`, so after the rebuild it silently compares against the pre-rebuild file.
- Readers of `attempt_maps_restnl` / `epoch_restnl` made under PRE_FROM have the same problem.
- Fix: derive these paths through `lp_licks.output_dir` under the same env.

### LOW

**L1. Tests that do not test their claim.**
- In `tests/test_frozen_fingerprint.py`:
  - `test_stored_readout_payload_equals_the_in_memory_fit` compares fit_all with fit_all. On a fresh store, load_or_fit returns the payload it just fitted, never a re-read. It should call twice and compare the HIT.
  - `test_spec_carries_fingerprint` asserts `train_fp == data_fingerprint(X)` (the same code path).
- In `tests/test_frozen_default.py`:
  - `test_collect_5c_matched_draws_stored_in_one_seed_keyed_spec`: "stored" and "mem" are both in-memory draws, and `_read` is stubbed to return the same object, not an unpickled copy.
  - `_same_records` (`:150`) zips without a length check.
- There is no per-kind test that perturbing one training row changes the spec id. Only the generic make_spec test exists, so the 15 call sites were verified by reading only.
- Not tested in `test_rebuild_switches.py`: H2, H4, M3, M4, M6; camera integration in beta_maps / `_working_xy` / hemo_map_control / position_coding_directions; skip-if-exists routing; `--pre-from` wiring inside the `main()`s.
- There are no tolerance-vs-exact problems: every pinned comparison is `array_equal` / `==`.

**L2. A few remaining reads bypass the hemo switch.**
- Import-time `hemo_{ADOPTED}` in `scripts/o2_video_me.py:74`, `quiet_hemo_check.py:113` and `iti_hemo_check.py:128`. These go stale if the env is flipped in-process.
- `scripts/cohort2_qc/channel_drift.py:62,82` and about 12 other cohort2_qc scripts use the literal `hemo_meegkit_hpfit`. This is cohort-2 QC and probably intentional, but it does not follow the switch.
- `locanmf_lick_aligned.py:116` loads C directly; it is exempted in the grep test. Its untagged output dir is globbed by locanmf_contralateral (`:49`), locanmf_lick_pool, lick_cards and event_master. Under projection, contralateral mixes an adopted lick arm with a projected cue arm.

**L3. Silent fallbacks.**
- `hemo_variants.py:256-261`: `except Exception: ADOPTED = BUILD_DEFAULT` silently ignores the env if config fails to load.
- `config.py:214`: `WIDEFIELD_HEMO_VARIANT=""` means zerophase. Fix with `os.environ.get(...) or None`.

**L4. `cue_classes` edge case.**
- The moved `cue_classes` adds an `if not len(tv): return out` early return. An all-NaN `cue_daq` table now gives "unknown" where origin/main raised IndexError. Untested.

**L5. Caches keyed without the run mode.**
- `grant_kit._BUNDLE_CACHE` (an, align) and the lru-cached `_collect_5c` / `position_reference_maps.maps_by_epoch` ignore run mode and env. This matters only for in-process env changes (tests, notebooks).

**L6. Joint-basis edge cases.**
- An explicit `load(bid=<08-13 superseded id>)` with no manifest field falls back to the legacy "meegkit_hpfit" label (`joint_locanmf.py:418-427`).
- A pinned id missing on a box now raises FileNotFoundError; on main the newest was served.
- The "ignoring newer" list includes older bases.

**Process note (resolved).** While this review ran, the wf_rebuild checkout held the unresolved attempt-coding trial merge (UU / AA on about 60 paths), so tests run from that tree tested code that was not 85af6485. It is now committed as 3988cbe3. Keep trial merges in separate worktrees.

## Claims verified OK

1. **Defaults unchanged.** Resolved with no env vars, origin/main vs tip:
   - identical: `svtcorr_path`, `hemo_variants.ADOPTED` (`meegkit_hpfit`), `locanmf_dir_name`, `combined_encoding.hemo_dir` / `load_imaging`, `maps_dir_tag` (`affine8v1`), `run_mode.tag()` / `cache_tag()` = "", `lp_licks.output_tag()` None, `keyed_name` / `subdir` identity, `effective_kind` unchanged, `project_all()` False;
   - defaults.yaml edits are additive only (`joint_basis`, `trial_selection.gate: daq`);
   - joint-basis pins equal what origin/main's newest-basis rule returns today for all four animals;
   - `locanmf_session.load_C` default is the same file and the same `np.load` at every replaced site;
   - `_class_select` under daq is byte-identical (frozen-copy test, exact).
2. **One hemo switch.** Flipping `defaults.yaml hemo.variant` alone (scratch copy) moved svtcorr_path, ADOPTED (at import), locanmf_dir_name, hemo_dir, maps_dir_tag, run_mode tag, effective_kind and project_all. Traps T2 / T3 / T4 / T7 are fixed in wfield_local. No direct ADOPTED / literal-variant / env reads remain in wfield_local. The exceptions are in scripts (L2).
3. **Joint basis.** Under a non-default variant every consumer goes through a variant-aware path: BasisSource, `_joint_signal` with the project_all check (the nolick fix is real), `signal_or_project`, iti_basis_decoder. Every cache is hv-tagged. `Basis.signal` refuses stored C under project_all. Every `load()` call honours the pin, and nothing else globs basis dirs.
4. **Fingerprint.**
   - All 15 kinds + matched_frozen hash the rows actually fitted, with the same mask as the fit (per-kind list in review B's notes).
   - `data_fingerprint` hashes dtype + shape + bytes, sorted dicts, ordered lists and exact scalar reprs.
   - selection and hemo_variant are read at call time; hemo is in base_id.
   - `LegacyStoreError` is a SystemExit.
   - Lick-landing read-outs: `ridge_fit` / `fit_all` / `data_sig` are unchanged since c61ac5ce. The frame-SD scaling scripts (`iti_zscore_check`, `iti_area_split_sd`, the SD division in `iti_lick_order`) are untouched in range, so they are bit-identical by construction.
   - precue_engagement_states in-memory fits are the same deterministic fits as main.
   - The 5rm payload key and rng use are identical on hit and miss.
5. **Output tagging (share side).** The previously unrouted share writers are routed: grant_figures/epoch, combined_encoding.out_dir, frozen_encoder.out_root, the cohort-1 quiet-spikes root, epoch_restnl, and the preprocess map npz. Fixed-name JSON readers and writers use the same `keyed_name`. Exceptions: H2, H3, M3.
6. **Prune guard.** The `{lab}__{kind}__*` prune cannot match a tagged kind (`__` separator). The restnl stem glob is the same. No other delete crosses modes.
7. **Camera gate rules.**
   - The lick class is contact-only, so PS93_0813 is kept.
   - "unknown" is in neither working nor stopped.
   - Every `_class_select` caller passes `cls` and raises without it; s1_refit uses `rule="daq"`.
   - Engagement measures are untouched (spout_behavior, orofacial_cohort, engagement_by_epoch are not in the diff).
   - Camera lines are not `# [gate-source hook]`-marked, so the digest sees them.
   - `cue_classes` logic equals main's, except L4.
8. **Merges.**
   - 2b182c76 `epoch_figures.py`: only the 4 conflict hunks differ from `git merge-tree`. main's open-dot top band is the default; lp-final's band covers `dot_style="animal"`; `write_session_values` + `save_figdata`, and `replot_from_sidecars` + `replot_figdata`, are all kept.
   - 6622a3c3 `make_spec`: `scaling` and `lp_trained()` are both kept.
   - 4b1919f0: a clean auto-merge, with nothing reverted.
9. **Tests.** On a clean export of 85af6485 I ran 31 files in 3 shards (rebuild_switches, frozen_fingerprint, swapmask_overrides, config, session_cache, writeguard, trial_classes, map_collect, combined_encoding_options, joint_locanmf, joint_basis, lp_gate_option, one_engagement_gate, frozen_default, frozen_models, lp_licks, reference_registry, rest_frozen_decoder(_nl), frozen_encoder, frozen_* guards, joint_xsession, nolick_*, stopped_trial_class, deck_section_g, lick_iti_gate). Result: **513 passed, 4 skipped, 0 failed.**
10. **Reproductions.** Re-run from the clean export into scratch (`<scratchpad>/review_repro/`) and compared with `compare_figs.py` against `presentations/_source_PI_overview_2026-11-02/build_inputs/`:

| fig | script | CSV | image |
|---|---|---|---|
| 0.2 | fig_behaviour_hits | max diff 0 | pixel-identical |
| 1.4 | fig_decoding_sessions | max diff 0 | pixel-identical |
| 4.4 | fig_frozen_refit_pooled | 3 sections, max diff 0 | pixel-identical |
| 3.4 | fig_state_position_sessions | 0 vs `_balem` (11-03); **0.307 vs the 11-02 21:39 original** | identical to `_balem`; 1.2 % pixels vs the original (M9) |
