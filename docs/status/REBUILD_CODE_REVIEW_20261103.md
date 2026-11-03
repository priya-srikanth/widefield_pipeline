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

---

# Second pass (2026-11-03 evening): tip **f7bb4302**

**Range reviewed:** everything after 85af6485.
- Merge commits: 3988cbe3 (attempt-coding), e74a4579 and 25ce7efb (origin/main).
- Fix commits: c17be8f5, 8d169197, e04143c6, 5361b2fa, 86280b8a, 464226fd, f7bb4302.

**Method:** same as the first pass, on a clean `git archive` of f7bb4302 with three read-only helpers.
- I did not trust the log.
- Code was imported from the export. I checked the import path, because the editable install points to `widefield_pipeline`, and the cwd shadows PYTHONPATH.

**Full suite at f7bb4302** (6 shards, ~22 min wall clock): **4,402 passed, 182 skipped, 11 failed.**
- The 11 failures are exactly the known `test_parallel_results_are_ordered` cases:
  - curbd_pilot x2, iti_decoder_pattern_stats, lick_code_subspace (results used unsorted);
  - arrival_to_cue_traces, lick_cue_to_lick, lick_iti_prelick x3, revup_neural, stopped_trial_traces (runtime globals that workers never see).
- Nothing else failed. The test_config failure from 86280b8a is fixed.

## 1. Status of the first-pass findings

| id | status | evidence (f7bb4302) | remaining |
|---|---|---|---|
| H1 | FIXED | `frozen_models.py:227-232` `lp_only` fingerprints real rows. Callers: `attempt_coding.py:134-137`, `frozen_readouts.py:69-71`. No other kind sets lick_source after make_spec. | - |
| H2 | PARTIAL, plus regressions N3 / N4 / N5 | 13 modules routed via `run_mode.subdir`. `writeguard.py:90,100-107` now checks figures_working. | Routed writers don't call `assert_writable`. `nightly_figs.py:685,835-853` passes `--output <fw>`, which bypasses the routing. |
| H3 | FIXED | `attempt_coding.py:378-381,397`, `attempt_maps.py:211-216,359` | Moved the canonical folder (N1) |
| H4 | FIXED for session caches; PARTIAL for skip-if-exists | `trial_classes.py:369-405` csig (v36 / cam1 / jaw_onsets stats + class-code closure) enters `cache_tag` (`run_mode.py:96-108`) | `attempt_maps maps_*.npz` (`:371`) and `attempt_coding trials_*.csv.gz` (`:403`) have no csig. csig costs ~7 s per spawned worker. |
| M1 | FIXED | `run_mode.py:91-94` (projall in both the output and cache parts) | - |
| M2 | FIXED | refusals at `await_locanmf.py:317`, `batch_locanmf.py:68`, `nightly_figs.py:451` | - |
| M3 | FIXED as asked; regression N1 | `run_mode.py:143-171` | - |
| M4 | FIXED for the named roots | `position_reference_maps.py:620-632` | The test rebuilds its own root list (it does not read what `_raw_maps_code` hashes). `daq_io.rising_edges` and `behavior_events.get_or_compute` are still unhashed (low). |
| M5 | FIXED | `require_gate` in enl_state_counts:89, channel_position_maps:917, stopped_trial_traces:857. `enl_states.states_for:127-139` is wired. | - |
| M6 | FIXED for the listed splits | `GATES["miss_working"]` (`trial_classes.py:70`) | `attempt_coding.py:96` and `attempt_maps.py:77` still split no-lick rows on "working", which includes contact (low) |
| M7 | REGRESSION (N2) | `_DAQ_LOOKUP_IGNORES=("train_fp",)`, `run_selection` adds `-gateLP` | - |
| M8 | FIXED | `grant_confusion.py:756-764`: in memory, nothing stored | The 5rm "DAQ-trained" column is LP-row fits. Label it. |
| M9 | FIXED | origin/main merged (e74a4579). `locomotor_decoder.py:270-273` passes scaling + scaling_params + raw train_data. The 3.4 docstring is corrected. | - |
| M10 | PARTIAL | `lp_final_compare.py:39-49` uses `final_maps_dir` | That calls `lp_final_render.setup()`, which mutates `os.environ` for the whole process. Readers of attempt_maps_restnl / epoch_restnl were not re-pointed. |
| L1 | PARTIAL | `test_frozen_fingerprint.py:128-150` adds a real second-call HIT and a row perturbation for iti_readout. `_same_records` now checks lengths. | `test_spec_carries_fingerprint...:42` still uses the same code path. The `test_collect_5c` `_read` stub still returns the in-memory object (`test_frozen_default.py:227`). No perturbation test for the other 14 kinds. |
| L2 | FIXED | o2_video_me / quiet_hemo_check / iti_hemo_check resolve at call time. locanmf_lick_aligned refuses under projection. | cohort2_qc literals are intentional |
| L3 | FIXED | `config.py:213-214`, `hemo_variants.py:206-210` | - |
| L4 | FIXED | pinned by `test_cue_classes_all_nan_table_is_unknown` | - |
| L5, L6 | NOT FIXED (logged) | - | - |

## 2. New findings, ranked

### HIGH

**N1. The canonical attempt-coding / attempt-maps outputs moved, and readers still read the old folder (from the M3 + H3 fixes).**
- Both modules require `WIDEFIELD_LICK_SOURCE=lp` (`attempt_coding.py:210`, `attempt_maps.py:180`).
- `out_dir()` now routes on the full `lp_licks.output_tag()` (`attempt_coding.py:381`, `attempt_maps.py:215`). So EVERY legitimate run writes to `.../attempt_coding/LP_licks`, `attempt_maps/LP_licks` or `attempt_maps_restnl/LP_licks` (probe), while `run_mode.tag()` is "".
- These readers still read the old folder and silently get the PRE-REBUILD files:
  - `scripts/rebuild_figs/fig_goal_summary.py:66,81`, `fig_goal_epochs.py:35`
  - `early_predictors.py:29`, `recovery_timing.py:33`, `precue_class_matched.py:25`, `decoder_retention_sessions.py:23`
  - `lick_enl_target_test.py:894`, `kinematic_normality.py:272`, `cohort2_qc/quiet_task_events.py:350`
  - `deck_registry.py:2393-2409`
- Fix: for LP-only families, route on `run_mode.tag()`, or the output tag minus the plain-LP part. Add a test that `out_dir()` under LP alone equals origin/main's path.

**N2. The lp_final DAQ-trained cross-check cannot find its model (from the M7 fix).**
- `lp_final_render.setup(source="daq")` sets no gate-LP (`lp_final_render.py:54-57`), so the DAQ models are stored with `selection="sel-daq-quittail"`.
- The LP run (`--source lp --trained daq`, gate LP) asks for `"sel-daq-quittail-gateLP"` (`trial_classes.py:524-529`).
- `find_daq_trained` now matches selection (`frozen_models.py:153`), so it misses, and `load_or_fit` raises "loads DAQ-frozen models" (`:500`). The whole `trained_daq` arm fails.
- Fix: match the camera-vs-DAQ part of the selection but not the gate source (strip `-gateLP`, or compare `run_selection_tag()`). Add a test where the DAQ spec has gate=daq and the LP spec has gate=lp.

**N3. writeguard refuses the lp_final frozen store (from the H2 fix).**
- `lp_final_render.setup` puts `WIDEFIELD_FROZEN_MODEL_DIR` at `<figures_working>/frozen_models_lpfinal` (`:66-67`).
- `frozen_models._write` calls `assert_writable(d.parent)` (`:569`). Under the LP / gate-LP / PRE_FROM tag, the new fw check refuses it (probe: `is_writable(...cue_lick\frozen_models_lpfinal\PS92)` = False).
- Any first freeze in that store raises WriteGuardError. Any store or cache a user puts under figures_working during a tagged run is refused too.
- Fix: limit the fw check to figure / JSON outputs (exempt `frozen_models.local_dir()` and the session cache), or move the lp_final store beside figures_working.

### MED

**N4. In a tagged run, writers given an explicit `--output` and the readers disagree.**
- Grant readers use `_fig_root()` = `fw/<tag>`: `grant_kit.py:626`, `grant_confusion.py:102,171,324`, `grant_encoder.py:546,627`, `grant_behaviour.py:248,307`.
- `joint_xsession.py:262` (`--output` required), and every H2 writer when nightly passes `--output <fw>`, write `fw/<keyed>` instead.
- Readers then skip silently (`grant_confusion:102`). A tagged nightly also overwrites default-named PNGs in fw, because nightly has no run_mode guard.
- Fix: nightly routes `out` through `run_mode.subdir` (or `require_default`). joint_xsession warns when `--output` lacks the tag.

**N5. The new tagged figures_working folder is never created.**
- `poststroke_section_g.py:325-329` runs the full `collect()` and then fails at `open()` with FileNotFoundError.
- The same applies to miss_vs_stopped:132/144, section_g_figures, evoked_amplitude, fixed_scale_maps, hemispheric_*, pixel_rsa, spatial_reorganisation, vessel_contrast. They succeed only if another writer created the folder first.
- Fix: mkdir in `subdir`, or in each `main()`.

**N6. D12 is implemented per trial, not per session.**
- D12 (RECONCILIATION section 5) says sessions without full video (PS93_0813; PS92_0908 at 82.5 %) are EXCLUDED from working-class cells and listed in captions.
- The code only makes unmatched cues "unknown" (`trial_classes.cue_classes:539-556`), so PS92_0908's matched trials enter working and stopped.
- PS92_0908's missing video is in its tail. That is where the new rules act:
  - `apply_stopped_rule` skips unknowns and stitches separate no-response runs together;
  - the last-camera-lick anchor falls before the real last lick, so working ends early.
- No caption list exists.
- Fix: a per-session coverage threshold in `cue_classes_for` that makes the whole session unknown for working and stopped, plus a caption list. Or Priya amends D12 to the trial-level reading.

**N7. Jaw-only trials after the last lick still break stopped runs (an ordering consequence; needs Priya).**
- `cue_classes_for:577-581` applies the stopped rule BEFORE the session-end rule. A jaw-only trial after the last lick is dropped from working (after_end) but still breaks a no-response run.
- Probe: `[contact, NR x3, jaw, NR x3, jaw, NR x2]` gives 0 stopped and 10 trials in neither class.
- This undercounts stopped in the sated tail. Options:
  - (a) treat after_end like unknown when counting runs;
  - (b) count them as no_response;
  - (c) keep this behaviour and document it.

**N8. The RESTnl cache key has no config or helper-code coverage (pre-existing; not covered by e04143c6).**
- `restnl_mask._key:211-220` hashes VERSION, its own source / constants, the session signature and three file stats.
- It reads `lick_detection` (`:88`) and `segmentation.rest.min_rest_s` (`:135`), and calls rest_by_position / rest_engagement / quiet_periods / engagement_gate code. None of that is hashed.
- After a yaml edit, bm / prm rebuild but the old RESTnl baselines are still served and subtracted from them.
- Fix: put `map_collect.map_yaml_subset()` and the same code closures into `_key`.

**N9. Camera-gate data errors inside the per-session map workers drop the session with one log line.**
- `map_collect.collect` and `parallel.fan_out` catch Exception (`map_collect.py:166-175`, `parallel.py:74,89`). Callers print `!! beta-map/ref-map` and continue (`beta_maps.py:1673-1675`, `position_reference_maps.py:758-760`).
- The same happens per animal in grant_confusion:435, attempt_maps:172, hemo_map_control:196 and position_reference_maps:261.
- `DaqGateRefused` (a SystemExit) does propagate, including from spawned workers. So refusals fail closed, but camera data errors give a silently partial render.
- Fix: under camera, fail the render when `failures` is non-empty, or write the failures into the sidecar.

**N10. Two goal-page scripts read the default encoder run and write default folders in a tagged run.**
- `scripts/specificity_control.py:113-115` and `scripts/residual_roi.py:98,173-175` build `run = E/f"frozen_encoder_{tag}"` and `out = E/...` directly.
- `frozen_encoder.out_root()` now puts a tagged run under `imaging_encoding/<tag>/`. So these scripts read the old-variant run and overwrite the default output, with no writeguard call.
- More unrouted writers: `recovery_timing.py:32-33,277` (writes `E/attempt_coding` and reads `E/iti_licks` directly, not `il.OUT`), `early_predictors.py:28-29`. Repo-wide, about 44 scripts write under `E` / `M:` with no routing; most predate the merge.
- Fix: route through `run_mode.subdir` / `frozen_encoder.out_root()`, or `run_mode.require_default()`.

**N11. `il.OUT` serves the skip-if-exists extracts and the outputs from one folder (`scripts/iti_licks.py:50`).**
- Since M3, any of PRE_FROM, MAP_REFS, `<F>_FROZEN=0`, LP, or gate-LP moves the extracts. `labels()` then raises FileNotFoundError. It fails loudly, but this is broader than the ITI_FROZEN case section 8.1 documents.
- Fix: put the extracts under a cache-only tag (hemo / lnmf / selection / projall), and put the full tag only on `OUT_W`.

### LOW
- **Retired script resurrected.** `scripts/iti_nmatch_check.py` is back next to `scripts/archive/iti_nmatch_check.py`. attempt-coding retired it in 3ab92858; git treated the move as add / rename, so it was not among the 60 conflicts. It is still in `tests/test_frozen_default.py:1126 ITI_SCRIPTS`.
- **Old encoder / basis runs hard-coded.** `iti_licks.py:715,2210` and `iti_basis_decoder.py:46` / `iti_decoder_patterns.py:42` use the old variant's encoder-run label list and te basis. Under a hemo tag, new data is projected onto the old basis with no warning.
- **Untagged ITI output override.** `WFIELD_ITI_OUT_W` (`iti_licks.py:54`) overrides run_mode untagged. `face_offset_partial.py:35` sets it.
- **Two definitions of "stopped".** `camera_select("stopped")` (`trial_classes.py:653-654`) includes engaged-arm rows with camera class stopped; `camera_rows("stopped")` (`:619-620`) never does. So grant 12s / 12b and prm / pcd use different stopped sets. Return zeros for engaged rows.
- **Pre-first-cue rest frames.** A session with no working trial (PS93_0813) keeps pre-first-cue frames as "engaged" rest in `engaged_frame_mask` (`rest_engagement.py:141-167`). The time-local REST baseline then differs in construction from every other session. Under camera, return all-False.
- **Silent "daq" on config failure.** `_selection_cfg` (`trial_classes.py:422-427`) and `stop_run_min` (`:478-482`) catch every exception, so a config failure silently means "daq" / 6. Dangerous once `gate: camera` is the yaml default. Catch ImportError only, or export `WIDEFIELD_TRIAL_GATE` to workers.
- **Gate exemptions inside shared functions.** `spout_behavior.reference_engagement:706` and `quit_point.session_quit:122` open `daq_quit_rule` themselves. A future imaging caller would get the DAQ rule unrefused and unregistered. Add both to the register regex.
- **Unknown variants fall through.** `camera_select` / `_class_select` treat an unknown variant as "lick" (e.g. `"miss_working"` returns every engaged row). Raise instead.
- **Rule edges to confirm with Priya.** A 6+ no-response run at session START, or a mid-session pause, is labelled stopped; comments still say "terminal by construction" (`position_coding_directions.py:1733`, `position_axes.py:274`). A video session with no camera lick at all has no working trials.
- **Cache-key coverage gaps** (code and files, pre-existing).
  - The bm / prm digests do not cover `locanmf_frozen_decoder._args`, `lick_detection.detect_licks`, `classify_cues_with_backup`, `docked_periods` or `daq_io`.
  - `session_signature` does not stat sessions.yaml `regime` / `fmdir`, the cleanpairs frame map, or `gui_config.json`.
  - hemo_map_control's `svt415:rank{K}` key ignores the `functional_channel` override.
- **`.str[:4]` animal ids** in recovery_timing:320, lick_correction:193, lick_rim_overlay:41 (would break on cohort-2 ids).

## 3. Verified OK in this pass
- **Default behaviour** (no env), probed: `tag()` / `cache_tag()` "", `output_tag` None, `keyed_name` / `subdir` identity, `run_selection` = `sel-daq-quittail`, `effective_kind` identity, figures_working writable.
- **Attempt-coding merge (3988cbe3).**
  - The 9 hand-merged scripts are all OK: iti_basis_decoder, iti_contact_check, iti_first_lick, iti_where_3axis, lick_gain_prediction, iti_lick_order, iti_where_angle_reach, iti_licks, iti_hemo_check.
    - Each differs from the attempt-coding tip only by c61ac5ce's hunks, plus the run_mode line (iti_licks) and the reference row (iti_hemo_check). So every attempt-coding change survives.
    - The pre_loso_ridge conversions keep the alphas, training data, LOSO set, and scaler + RidgeCV.
    - Frame-SD / lick-window-SD division happens once, before the fit, unchanged, and is what `data_sig` hashes.
    - masked LocaNMF and CONTRIB_HOOK are kept.
  - The 39 byte-for-byte scripts are strict descendants of the rebuild side's blobs.
  - Across all 52 scripts:
    - no def is lost from either side;
    - no frozen_readouts / run_mode / svtcorr_path / output_dir / trial_classes use was dropped;
    - no duplicate defs, conflict markers or broken cross-script imports;
    - every non-test make_spec passes train_data.
  - wfield_local: lp_licks / session_cache keep both sides (gate-LP + run mode), map_collect keeps code_digest + `_Closure`, frozen_encoder keeps `out_root()` + `frame` moments + `--out-root`.
- **c17be8f5:** lick_balance_imaging_control(+_primary), lick_acad_midline_control and s1_refit_basis_table read `config.hemo_manifest_path` / `config.svtcorr_path`. No `hemo_{ADOPTED}` / literal adopted read remains outside cohort2_qc and the intended `iti_hemo_check.py:159` reference row.
- **Camera gate: no hard fail-open path.**
  - Every imaging selection site is wired or fails closed: grant_kit `_class_select` and its callers, beta_maps, position_reference_maps, hemo_map_control, position_coding_directions, position_axes, precue states, nolick categorize, poststroke_engagement, attempt_coding / maps, enl_states, rest_engagement, rest_by_position.
  - `DaqGateRefused` (a SystemExit) survives `except Exception` and fan_out.
  - Engagement measures (spout_behavior, quit_point, disengagement_tests, s1 quit-tail rows labelled DAQ) are ungated by design.
  - The gate-site register test is `tests/test_camera_gate_fail_closed.py`.
- **Stopped rule** (`apply_stopped_rule:490-514`), by synthetic probes:
  - a run of 6 or more (5 stays no_response) relabels every trial of the run;
  - runs at session start and end count;
  - unknown neither breaks nor extends a run and is never stopped;
  - any other class breaks a run;
  - empty and all-unknown inputs work.
- **End-of-session cutoff:**
  - the anchor is the last camera lick (contact or incomplete), with `<=`;
  - only working trials become after_end; stopped is not cut;
  - no-video sessions stay unknown;
  - the memo key carries window / rule / min_run / anchor;
  - unknown is never in working or stopped, nor counted as no_response.
- **Map-cache key (e04143c6).**
  - Every `defaults()` read in the 23 map-builder modules and config helpers falls under the hashed subset (decode / lick_detection / segmentation / preprocess.maps.tag / hemo.variant).
  - trial_selection.* and behavior.tail_min_misses reach the key through the selection tag; SVTcorr / LocaNMF through session-signature paths.
  - animals.yaml is used only at aggregation.
  - The only yaml left out entirely is session_overrides.yaml (functional_channel). It is covered only through the regenerated SVTcorr / U mtimes, except in hemo_map_control (LOW above).
