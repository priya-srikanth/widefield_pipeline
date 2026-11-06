# Code review of `widefield_pipeline` (main at fc09de78, 2026-11-04)

Read-only review. Seven reviewers each covered part of the repo, and I merged their findings. I then opened the cited lines for every high finding and for a sample of mediums. The cohort-1 rebuild branches were not reviewed.

## Bottom line

- The foundations are good. `analysis_kit`, `parallel`/`fan_sessions`, the caches, `frozen_models`, `writeguard` and `conftest.py` are well built, and the A1–A5 consolidations are visible in the code.
- The repo is **not yet "one definition per quantity."** Several things a scientist would call one quantity still have two or more definitions in code, and some of them disagree: the lick detector, the August-only baseline, the response-window fallback, bootstrap p and Holm, and the LP run name.
- About 60% of the important fixes touch files the rebuild branches edit, so they must wait for the merge. The safe-now work is mostly docs, dead code, plotting style and a few isolated modules.
- For a new neuroscientist the main obstacle is **finding things**. Core pieces live in private names, in figure modules or in `scripts/`. The library imports code from `scripts/`. `scripts/` has about 300 files and an index covering about 13 of them.
- Rough scale: about 2 weeks of focused work after the rebuild merges, then larger refactors done one at a time.

---

## Can give a wrong or inconsistent number (verified)

Terms used once: a **literal** is a value typed directly into code instead of read from `configs/*.yaml`. **Latent** means it is wrong in principle but does not change any number today.

### 1. Two lick detectors, and the config claims to be the only one
- **Where:** `configs/defaults.yaml:158-162` says `lick_detection` is "shared by behavior + imaging (single source)" with `thresh_lower: 0.5`. `wfield_local/analysis_kit.py:469` hard-codes `_load_daq_events(h5, "lick_analog", 2.5, 1.0, (0.001, 0.020), 0.10)`, i.e. a 1.0 V lower threshold plus a 100 ms refractory. I found that exact literal in **39 files** across `wfield_local/` and `scripts/` (the 09-21 doc said ~35). `lick_detection.detect_licks` also defaults to `thresh_lower=1.0` (`lick_detection.py:52`). The config readers use different fallbacks for `min_ili_ms`: `behavior_events.py:142` uses 0, `daq_trials.py:109` uses 40. The `behavior_events` comment at :139 says its call is "identical to imaging's", which is false. `DECISIONS.md:62` states 2.5/1.0 as the pipeline rule, which contradicts the yaml.
- **Why it matters:** the lower threshold sets where each lick ends, and the lockout is timed from that end. So the two paths can keep slightly different lick-onset sets. Imaging hits, RTs and lick-aligned trials can therefore differ from the behaviour `*_trials.csv` at the margins. `DLC_IN_WIDEFIELD_ANALYSES.md` §1 already records this, and §6 Q1 is open.
- **Fix:** first you decide whether imaging keeps its own detector (§6 Q1). Then put both detectors in `defaults.yaml` as named profiles (`behaviour`, `imaging`), each with one sentence on why it exists. Route all 39 call sites through one accessor, and include the profile in cache keys. Pin per-session onset counts before and after.
- **Effort:** medium (1–2 days plus the measurement). **Wait for rebuild: yes.** The lick source is rebuild territory, and the rebuild adds `lick_count_samples`. Fixing the false `behavior_events` comment is free and can go now.

### 2. The August-only baseline has many spellings, and hand re-renders silently use June+August
- **Where:**
  - `nightly_figs.py:80` holds `IMAGING_PRE_FROM = "0801"` and is the only place that sets `WIDEFIELD_PRE_FROM` (`:517`). `config.pre_from()` (`config.py:410-420`) returns None when the env var is unset.
  - `orofacial_cohort.py:1536,1543` keeps a second copy (`LATE_PRE_FROM`), compared as MMDD strings.
  - Other copies: `mixed_models` (`'0806'`), about 76 inline `x[-4:] >= '0806'` comparisons in about 46 script files, `lp_trial_counts` (`'0801'`), month == `'08'` tests (`lick_contact_by_epoch:21,31`), and drop-June tests (`lick_path_softmax:103`, `lick_trajectories_3d:75`, `o2_face_rhythm:615`).
- **Why it matters:**
  - CLAUDE.md tells you to re-render one family with `epoch_grant_figures --only <key>`. Run that way, the pre bands **and the frozen training sets** include June. That gives a different `spec_id` and different numbers from the nightly render of the same figure, with no error.
  - The 0801 and 0806 cut dates agree only because 0805 is excluded by `date_policy`. The month-08 tests would drop a September pre session, and the drop-June tests would keep a July one.
  - All of these are MMDD string comparisons, which break for cohort 2 across New Year.
- **Fix:** one configured value per cohort (for example `cohorts.yaml cohort1.pre_from: '0801'`). Make `config.pre_from()` fall back to it, with env `""` meaning "all pre", and compare with `config.date_key`. Delete `IMAGING_PRE_FROM` and `late_pre_from`, and point the scripts at one `in_baseline(label)`. Add a test that the curated label set is identical under the old and new spellings.
- **Effort:** medium (core small; script sweep 1 day). **Wait: yes.** The rebuild adds `config.apply_pre_from`, so fold this into it.

### 3. `hemo_variants.compute/write/CLI` default to the cohort 470 slot, not the per-session one
- **Where:** `hemo_variants.py:277` (`compute(..., func=FUNC)`) slices `svt[:, func::2]` at `:301`. `write()` at `:400` and the `--functional-channel` CLI default at `:441` also use `FUNC`. Five diagnostics call `hv.compute` without `func`: `precue_significance:80`, `hemo_residual_check:77`, `postcue_window_test:107`, `precue_window_sweep:170`, `filter_acausality_test:335`.
- **Why it matters:** for PS92_0828 (slot 0, in the curated post set) these diagnostics build a corrected signal with 415 and 470 swapped. That is the exact trap CLAUDE.md describes. A hand `--sessions PS92_0828 --write` would write a swapped variant folder. Production is safe, because `preprocess` passes the right slot.
- **Fix:** default `func=None` and resolve it to `functional_channel(session)` inside `compute`, and do the same in the CLI. Add a test that PS92_0828 uses slot 0.
- **Effort:** small (an hour). **Wait: yes.** The rebuild edits this file.

### 4. `run_locanmf` defaults disagree with `defaults.yaml`, and the README's direct command uses them
- **Where:** `run_locanmf.py:83-84` sets `--loc-thresh 70.0` and `--r2-thresh 0.99`, and the usage example at `:48-50` says the same. `defaults.yaml` says `r2_thresh 0.95` and `loc_thresh 80`. `batch_locanmf.py:69-74` reads the config correctly, so the nightly is fine. `wfield_local/README.md` §17 documents a direct `run_locanmf` command with no thresholds.
- **Why it matters:** re-running one session by hand from the README fits a different decomposition from every other session, with no warning.
- **Fix:** make the argparse defaults read `config.defaults()['locanmf']`, fix the example, and add a test that both CLIs' defaults equal the yaml.
- **Effort:** small. **Wait: no.** Nightly behaviour is unchanged. First check that the rebuild's LocaNMF-projection work does not edit this file.

### 5. A missing or undecodable DAQ file silently gives zero spout contacts
- **Where:**
  - `dlc_frames.py:241-249` returns an empty array, with only a print, when there is no `.h5` or when `decode` raises.
  - `scripts/pose_kinematics_demo.py:28-37` (`daq_contacts_ms`) then finds no contacts.
  - `tongue_kinematics.py:1216-1217` writes `n_daq_contacts = 0`.
  - `trial_classes.py:192-196` raises only if the column is *missing*. A column of zeros passes and becomes `fillna(0)`.
- **Why it matters:** every real contact trial in that session would be reclassified as incomplete, jaw_only or no_response, with no error. I did not check whether any current v36 session hit this path.
- **Fix:** return None or NaN on a missing DAQ, write NaN `n_daq_contacts`, and map NaN to the `unknown` class. Add a test. Also run a one-off check of v36 `trials.csv` for sessions whose contacts are all zero.
- **Effort:** small. **Wait: yes** (trial classes). The one-off check of existing outputs is read-only and can be done now.

### 6. The response-window fallback is 2.0 s in behaviour and 3.5 s in imaging
- **Where:**
  - `spout_behavior.py:304-306` passes `behavior.licking.response_window_s` (2.0), and `:504-505` uses a literal 2.0.
  - `nolick_decoder.py:82` uses `DEFAULT_RESPONSE_WINDOW_S = 3.5`, and `quiet_periods` uses `decode.max_rt_s` (3.5).
  - `defaults.yaml:190-194` says the task "has run 3500 ms, never 2.0 s".
  - Literal 3.5 s or 4.0 s windows appear in `scripts/trial_attempts.py:42` (which feeds `mixed_models.behaviour_deficit`), `lick_contact_by_epoch.py:37` (4000 ms) and two cohort2_qc scripts.
  - `tests/test_response_window.py` covers only one consumer.
- **Why it matters:** this is latent while every session has a `gui_config.json`. A session without one (crashed, cohort 2) would be scored 2.0 s for behaviour and 3.5 s for imaging under the same word, "responded". `lick_contact_by_epoch` already uses a different window today.
- **Fix:** one fallback key, read inside `daq_trials.response_window_s` so callers cannot pass their own. Rename the 2.0 s key to `count_window_s`. Extend the test so every resolver gives the same answer when there is no gui_config. Check `git blame` on the 4000 ms before changing it.
- **Effort:** small. **Wait: yes** for `nolick_decoder`. `spout_behavior` and `daq_trials` are not on the rebuild list.

### 7. Bootstrap p-value and Holm/BH copies in scripts give different stars
- **Where:**
  - The `ak.boot_p` floor is `1/n` (`analysis_kit.py:276-283`).
  - `iti_decoder_pattern_stats.py:139` (and three sibling iti scripts) use `2*min(...) + 1/N_BOOT`, which is always larger.
  - `coupling_by_class.py:105` has no floor, so it can return p = 0.
  - The `a3_lmm_*` Holm copies (e.g. `a3_lmm_heterogeneous_stars.py:13-17`) skip NaN p-values in their running maximum but still count them in the family size. So a NaN p comes back with a finite adjusted value. They are fed `het_p` and `p_changes`, which can be NaN (`:44-48`).
  - `iti_decoder_pattern_stats._bh` counts NaN in the family.
  - `lick_templates.bh` returns q-values, while `ak.bh` returns a yes/no mask.
  - `recovery_timing._boot_mean_of_animals` (`:165-175`) is a third LEVEL weighting (animal-weighted, n = 2000) that feeds the 10-25 pooled figures. Rule 9 says levels use `boot_ci`.
- **Why it matters:** the same draws give different p-values and stars depending on which script made the table.
- **Fix:** replace with `ak.boot_p`/`boot_p_columns`/`holm`/`bh`, add `ak.bh_q` for q-values, then re-run the four iti tables and diff the stars. The `a3_lmm_*` scripts were the side-runs behind the adopted `ak.epoch_change_lmm`, so archive them rather than fixing them. For `recovery_timing`, either use `boot_ci` or add an explicit `weight='animal'` option to `analysis_kit`. That choice is yours.
- **Effort:** small–medium. **Wait: no.** These are scripts only.

### 8. LP run name typed out in 46 scripts and 9 library modules
- **Where:** `lp_r4_heatmap_20261013__clean_v36` appears in 46 non-archive scripts (mostly as full `M:/...` paths) and 9 `wfield_local` files (`orofacial_cohort.py:67` `RUN`, `trial_classes.py:80` `V36`, `orofacial_jaw_lateral.py:34`, `face_rhythm_validation.py:34`, three `deck_registry` paths, ...). There is no yaml key.
- **Why it matters:** when the rebuild or the next 3-D re-run produces a new run, scripts that use `oc.RUN` move and these keep reading v36. A page or deck could then mix runs with no error.
- **Fix:** add `orofacial.run` in `defaults.yaml` and one accessor plus an `encoding_dir()` helper. Make every script take `--run` defaulting to it. Do this in the same commit that switches to the rebuild run.
- **Effort:** small–medium. **Wait: yes.** It must coincide with the run switch.

### 9. Deck prose carries hand-typed results
- **Where:** `deck_registry.py` has about 353 decimal literals against 19 `{{SELF:}}` tokens, `deck_text.py` about 206 with no tokens, and `locanmf_analysis_deck.py` about 170. Examples are the per-animal numbers in `_CD_LOO` and `_DRIFT_GROUP`, some marked "PROVISIONAL 2026-09-12". These counts are the reviewer's; I checked that the pattern exists, not the totals.
- **Why it matters:** after the rebuild re-renders the figures, many captions will still state old values, and `deck_audit` does not check them. In September `deck_values` measured 171 such numbers; the count has roughly tripled.
- **Fix:** after the merge, convert the "WHAT IT SHOWS" blocks to tokens one family at a time, starting with the families the rebuild changes. Until then, run a grep listing literal-number captions whose figure was re-rendered.
- **Effort:** large (spread over time). **Wait: yes.**

**Downgraded or dropped from high, and why:**
- *Epoch bars (pooled level) vs delta panel (animal-weighted change)*, `epoch_figures.py:2838-2898`. Verified, but it is the deliberate rule-9 convention, so it is a decision rather than a defect. Moved to medium: either state it in subtitles and legends, or make levels animal-weighted, which moves every bar. Your call.
- *Position estimator copied about 12 times.* Verified: `max_iter` is 2000, 3000 or 4000 (`locanmf_frozen_decoder.py:71`, `position_coding_directions.py:483`, `beta_maps.py:288`), and `make_spec` (`frozen_models.py:87`) records no C or max_iter. These differ only in optimiser stopping, so predictions are unlikely to move. The real risk is that someone edits `_pipe`'s C and the stored frozen models are served unchanged. Moved to medium.
- *Library imports `scripts/`* (`analysis_kit.py:513-514` and others). Verified, but it is coupling, not a wrong number. Moved to Modularity, top item.
- *"working / engaged / stopped" mean several trial sets.* Verified (`trial_classes.py:261-264` vs `working_trials.py:45`). Pinned tests keep them consistent today, so it is a naming trap: medium.
- Nothing was dropped outright. Every high finding I opened matched the code.

---

## Redundancy to consolidate

| Concept | Copies found | Agree? | Canonical home | Fix |
|---|---|---|---|---|
| DAQ lick detector | yaml profile + literal in 39 files + `detect_licks` default | **No** (0.5 V vs 1.0 V, refractory) | `defaults.yaml lick_detection.{behaviour,imaging}` + one accessor | Item 1 |
| August-only pre cut | nightly_figs, config env, orofacial_cohort, mixed_models, about 76 inline, month tests | Today yes; 0801 vs 0806 | one cohort key + `config.pre_from()` | Item 2 |
| Response window fallback | 4 resolvers, 3 fallbacks, about 6 literals | **No** (2.0 vs 3.5 vs 4.0) | `daq_trials.response_window_s` | Item 6 |
| Bootstrap p / Holm / BH | ak + 4 iti + coupling + face_rhythm_session_shape + 4 a3 Holm + 3 BH | **No** | `analysis_kit` | Item 7 |
| Level bootstrap | `ak.boot_ci`, `epoch_figures.scalar_value_draws` (copy), `recovery_timing` (animal-weighted), `session_lick_summary` (median) | Partly | `ak.boot_ci(draws=True)` | Add `draws=`, pin |
| N_BOOT | 4000 / 2000 / 1000 / 400 / 200 / 5000 | No (different CI resolution) | `defaults.yaml bootstrap:` | Per family, pin |
| Position estimator `StandardScaler+LogReg(C=0.5)` | about 12 inline; private `_pipe` imported 21 times | max_iter differs; locomotor C=1.0 | public `position_estimator()` + `decode.estimator` yaml | Add to spec only when it differs from today's values |
| Block-CV with <2 groups | 65 copies, 3 fallbacks (NaN / ungrouped / stratified) | No, by design in places | `block_cv_predict(on_too_few=...)` | Callers state their choice |
| Feature `_args()` (pre_s=1.0, post_s=2.0, FS=31.23) | about 14 `_args` + about 19 namespaces; FS in 46 files | Yes today | `feature_args()` from yaml | Keep named exceptions (nolick max_rt=2.0) |
| Per-session z-score | 4 implementations | Yes | `label_shift.session_zscore` | Pin `pool_sessions` |
| LP run name | 46 scripts + 9 modules | Yes today | yaml + `oc.RUN` | Item 8 |
| Allen dir `affine8v1` | about 100 hits in 61 files; 2 yaml keys | Yes | `config.allen_dir()` | Sweep + guard test |
| Cam1-only lick filter | `trial_classes:178` vs `lp_licks:253` | **No** (`~in_3d` missing in lp_licks; verified) | `cam1_licks.cam1_only()` | Bump `LP_VERSION`, report the change in lick counts |
| Contact window cue+70 ms | yaml + tongue_kinematics + demo constant + about 12 inline + 4 helpers | Yes | `daq_contacts.py` | Move out of `scripts/` |
| Reference spouts | 2 constants + 2 inline | Yes | gate module | Extend guard test |
| Spout code map / order / ipsi naming | 7 code maps, about 5 orders, 12 L→ipsi dicts | Orders differ by purpose; `face_rhythm/common` "canonical" is not | `spouts.py` from `stroke_laterality` | Delete dead `matched_residual.POSMAP` (wrong keys) |
| "Orofacial cortex" area set | 5 memberships (4-area id dict ×9, ALM/tjM1 sets, 6- and 9-area sets, 2-area) | **No** | `configs/roi_groups.yaml` | Name the group in figure titles; no data moves |
| Animal marker | `cat_points`, `orofacial_cohort`, 10–11 scripts | Yes for cohort 1; **PS118 = PS121 = '>'** (verified: 317 % 6 = 311 % 6 = 5) | `animals.yaml marker:` | Safe now |
| Animal / epoch colours | `config.animal_color` vs 6 locanmf dicts + Dark2 in `mixed_models` + about 15 epoch palettes | No (Dark2 vs tab; `session_lick_summary` epochs use animal colours) | `style.py` | Safe now |
| Star headroom | `orofacial_cohort._mark_rows/_tidy` vs `sig_marks` | Same arithmetic, different de-duplication | `sig_marks` | Safe now |
| PNG+SVG save | 3 helpers | No (only orofacial keeps editable SVG text) | `figure_layout.save` | Safe now |
| Wilson CI | 2 | No (NaN vs 0 at n=0; wrong docstring) | `analysis_kit.wilson` | Safe now |
| Verified copy + SHA-256 | `archive_day`, `camera_nightly` | Drifted (buffers, read-back rule, no hash for cameras) | `safe_copy.py` | Safe now |
| Disk caches | `session_cache`, `result_cache`, `map_collect` (4 stat-sig helpers, opposite disable switches) | Different invalidation | one cache module | Wait |
| "Which sessions" APIs | 6 in config/ak + inline in `nightly_figs:522` | No (nightly list ignores phase and pre_from) | documented table + merge | Check pools before changing; can move cross-mouse/RSA |

---

## Modularity / readability (biggest maintainability costs)

1. **The library imports `scripts/`.** `analysis_kit.session_behavior` (`:513-514`) imports the quit and trial definitions from `scripts/rest_migration/`. `orofacial_cohort`, `orofacial_outputs` and `combined_encoding` import `scripts/pose_kinematics_demo` (a script that calls itself "a smoke test ... NOT a result") for the canonical DAQ contacts. `pose3d_side` imports private names from `scripts/pose3d` and `scripts/apparatus_mouth`. `scripts/README.md:5` says the opposite. Archiving a "one-off" therefore breaks production numbers. Fix: move these into `wfield_local` (`engagement.py`, `daq_contacts.py`, `pose3d_io.py`), leave one-line shims in the scripts, and add a test that bans `import scripts` in `wfield_local`. Medium effort. Wait (pose/encoder/gate files).
2. **There is no obvious answer to "where is the decoder?"** The estimator is a private `_pipe`. The features are private functions in `locanmf_position_decoder`. Position names and the lick loader are in a *plotting* module (`plot_lick_aligned_averages`, 53 importers). The session registry is `SESSIONS` (48 importers), which is loaded **when the module is imported**. Because of that, it freezes whatever animal or cohort env vars were set at that moment, and `epoch_audit.full_cohort()` cannot undo it. Fix: a re-export facade, `decoding.py`, with a docstring walking through "features → estimator → CV → frozen model → scoring". Replace the import-time `SESSIONS` with calls at the point of use. Medium. Wait.
3. **Very long functions and modules.** `build_analysis_deck` is 1,776 lines in one function (`locanmf_analysis_deck.py:473`). It was 3,892 on 09-21 and has grown back by about 100. `nightly_figs.main` is about 680 lines of inline stages. `combined_encoding.run_session` is 621 lines. `epoch_figures.py` is 3,165 lines covering sidecar I/O, epoch assignment, statistics, layout and rendering, and its docstring is stale ("three panels", "NOTHING HERE RECOMPUTES"). `beta_maps` holds the repo-wide pixel-map significance library. Fix: one function per deck section with a context object, verified with `scripts/deck_fingerprint.py`. A declarative step table for the nightly that keeps each incident comment as a `why=` string. Split modules with re-exports. Large. Wait.
4. **Analysis lives in figure modules.** Epoch renderers call 15 private `grant_figures._collect_*` names. `_collect_5c` scores decoders. `conftest.patch_grant` exists because this coupling hung the test suite for 16 minutes. Fix: a public `pooled_collectors.py` with documented return shapes. Large. Wait.
5. **Behaviour switches with no registry.** About 58 env vars change what a lick, gate, baseline or encoder design means, and none are in yaml. Two examples of the cost:
   - `combined_encoding.out_dir` leaves out `COMBINED_ENCODING_TONGUE`, so 3-D-tongue and cam4-tongue runs write the same folder. `frozen_encoder` does include it.
   - `working_trials.row_masks` silently swaps a quit_tail selection for the production working set (`:241-242`). `OROFACIAL_GATE` is read at import.

   Fix: `switches.py` with `active_switches()` stamped into figures and cache keys, and make `row_masks` raise instead of swapping. Short term, add the missing switches to `out_dir`. Medium. Wait.
6. **Gate names collide.** `trial_classes.in_working` (`:261`) includes jaw_only, while production `working_trials.IN_WORKING` excludes it. The behaviour timeline's "WORKING" is a third set, and the imaging gate a fourth. Rename (`reacted`, `camera_working`, `reference_collapse`, ...) and add a gate table to ARCHITECTURE. Medium. Wait.
7. **Caches miss config changes.** `session_signature` hashes no yaml, and four past version bumps were needed for exactly this. The frozen-model spec does not record the trial rule (max_rt, precue_lickfree, lick profile) or the estimator settings. Fix: a digest of the config blocks the cached code reads; add new spec fields only when they differ from today's values, so current ids stay the same. Medium. Wait. Read `FROZEN_MODELS.md` first.
8. **Two large hub modules and an ambiguous alias.** `spout_behavior` (1,954 lines) and `orofacial_cohort` (2,805 lines, 124 importers). The alias `oc` means `orofacial_cohort` in 77 files and `orofacial_clean` in 27.
9. **Cohort-2 traps not yet triggered** (verified where marked):
   - `analysis_kit.curated_ids` (`:438`) slices `lab[5:]`. This is verified, and it breaks on the 5-character `PS118`. It also returns a **set**, whose iteration order is not fixed (it depends on Python's hash seed). That is the rule-9 order trap if it ever feeds a pool.
   - `nightly_figs` sorts MMDD strings.
   - `preprocess` builds the year by hand instead of using `config.date8`.

   One-line fixes each. Wait.

---

## Docs a new lab member would trip on

- **No single starting point.** There are 8 "START HERE" and 25 "superseded" pointers in CLAUDE.md. The 11-03 trial-gate and repo-audit handoffs are linked from no entry doc. `docs/status/README.md` was last updated 09-22, and README says it holds 16 handoffs; there are 96. README's key-docs list omits `FINDINGS.md`. Fix: one START_HERE with FINDINGS first, one current handoff per arm, and a table "to change analysis X: module / config key / registry entry / test / `--only` command".
- **`ARCHITECTURE.md` is stale** (09-22: "162 modules"; there are now 256). It never mentions trial_classes, working_trials, frozen_encoder, pose3d, orofacial, face_rhythm, cat_points or sig_marks.
- **Machine and env instructions describe the other box.** These are wrong on this desktop:
  - `CONDA_PREFIX=C:/Users/sabatini/...` (rule 4 and the "Reference" env line);
  - the git path in the analysis runbook;
  - the cache default in README;
  - `conda activate label` in the DLC runbook.

  `analysis_desktop` and its M:/N: swap are not described, and the two torch instructions (cu124 vs cu128) do not say which box each is for.
- **`STATUS_2026-09-21_ENGINEERING.md` is still "START HERE for code" but its inventory is stale.** The grant/epoch split is done. The deck function is 1,776 lines, not "~3,900". The `phase_labels("pre")` filter repeats are 48. Direct lick loads are about 29–39.
- **CLAUDE.md rule 6 says 4 of 11 rest loops were converted.** The four expensive ones are all converted now. The rule also says `enl_states` is unit-tested; it has no test, and its CLI only raises "not wired yet".
- **`scripts/README.md`** covers about 13 of about 300 files and says scripts are never imported by the package. Page-figure producers sit flat beside 27 `*_check.py` probes. Fix: a generated `scripts/INDEX.md` grouped by page and figure, tagged producer / diagnostic / superseded.
- **Smaller items:**
  - README roadmap items are already done.
  - "tests/ (22)" should be 243.
  - The CLAUDE.md figure path is garbled ("labcamsnalysis_figures").
  - 17 script paths cited in DECISIONS have moved to `archive/` or are gone.
  - The deck title slide says "Curated pre-stroke sessions" and "chance = 0.17" for an all-phase deck.
  - The behaviour-deck caption still names `flag_engagement`.
  - `paths.py:50-54` describes the machine-detection order wrongly.

---

## Good patterns to keep

- **`analysis_kit` and `tests/test_analysis_kit.py`:** the `Interval` type, `boot_ci` for levels vs `boot_delta` for changes with each weighting explained, `paired_animal_draws` as the single change rule, and frozen-copy tests with **exact** equality. This is the template for every refactor below.
- **`parallel.fan_out` / `fan_sessions`:** module-level workers, failures reported rather than swallowed, BLAS pinned, results sorted.
- **Safe caching:** atomic publish in `session_cache`/`result_cache`. `feature_cache_kind`, `lp_licks.spec_token` and `trial_classes.selection_tag` put every result-changing input into the key. `apply_enl_window` runs before the key is built.
- **`frozen_models`:** a spec change is loud (`SPEC-CHANGED`), and new fields are added only when given, so ids stay byte-stable.
- **`writeguard`** plus the re-verify at `archive_day`'s delete sites. Keep both checks: the size comparison alone passes when both files are missing.
- **`conftest.py`:** refuses writes to the share, gives each test a cold cache, pins epochs, and `patch_grant` refuses a patch that lands nowhere. Plus the "one definition" guard tests (`test_one_engagement_gate`, `test_engaged_cut_comes_from_config`, `test_response_window`, `test_working_trials` pinning against `GATES`).
- **Deterministic figures:** stable seeds (`_seed_for`, no `hash()`), uncorrected and Bonferroni marks from one set of draws, and guards that refuse a star carried by one animal or a cohort with mixed lesion sides.
- **`trial_classes.py`**, with its one-case-per-rule `test_trial_classes`, which a neuroscientist can read. `imaging_page/s_per_animal.py` re-checks against stored tables before drawing.
- **Long comments that name a past bug at the spot it would recur** (for example `poststroke_compare:1013-1018` on max_rt=2.0, and the deck's `phase="all"` comment). Every refactor must carry these forward.

---

## Proposed order of work

**Rule for every item that touches a computation:** pin the outputs before and after, the way `tests/test_analysis_kit.py` does (exact equality on draws, session lists, onset counts, sidecars, or a `deck_fingerprint`). A changed number is a decision to record in DECISIONS, never a side effect.

**Phase 0: safe now (docs, dead code, files outside the rebuild). About 3–4 days total.**
1. `run_locanmf` defaults from yaml, plus a test (0.5 h). Check first that the rebuild does not touch it.
2. Read-only check of v36 `trials.csv` for sessions with all-zero `n_daq_contacts` (1 h).
3. Script statistics: replace the boot_p/Holm/BH copies with `ak.*`, archive the `a3_lmm_*` side-runs, diff the iti star tables (0.5 day). Decide `recovery_timing`'s weighting.
4. `safe_copy.py` for `archive_day` and `camera_nightly` (0.5 day).
5. Plotting style module: markers in `animals.yaml` (fixes PS118 = PS121), animal and epoch colours, one star-headroom path, one save helper with editable SVG text, tests for `cat_points` and `sig_marks` (1 day).
6. Tests: drop `importorskip` on in-repo modules, add `pythonpath`/`testpaths` and `slow`/`live_data` markers to pyproject, add `pose_windows`/`docked_periods` tests, and change `test_no_hardcoded_machine_paths` to recurse (`rglob`) (1 day).
7. Docs: START_HERE, per-box env table, fix the CLAUDE.md env/rule-6/enl_states lines, refresh the 09-21 inventory, `scripts/INDEX.md`, README roadmap, DECISIONS paths, the `behavior_events` comment (1 day). Rewrite captions rather than annotating them.
8. Dead code: `orofacial_cohort.engaged_trial_ids` (an unguarded reader), `flag_engagement` (keep the ban test), `matched_residual.POSMAP`, the `enl_states` stub CLI, and, once you confirm, the orphan `locanmf_cue_auc/firstlick_aligned/event_master` and `locanmf_decoder_ablation/qc/xsession_generalization` (0.5 day).
9. The `spout_behavior` engagement-timeline plot and the gate's WINDOW/MIN_RATE from yaml; `daq_trials` reward threshold from config (0.5 day).

**Phase 1: right after the rebuild merges. About 2 weeks.**
1. Lick detector profiles plus the sweep (needs your §6 Q1 decision).
2. One August-only baseline key, folded into `apply_pre_from`.
3. Response-window single fallback.
4. LP run name into yaml, switched at the same moment as the rebuild run.
5. `hemo_variants` per-session slot default.
6. Fail-loud DAQ contacts and NaN → `unknown`; one cam1-only filter (bump `LP_VERSION`).
7. Cohort-2 one-liners (`curated_ids`, `date_key` sort, `date8`).
8. Gate renames and the gate table.
9. Library-imports-scripts moves, with shims and a ban test.
10. Encoder `out_dir` includes the tongue and calib switches.
11. Literal-number caption grep after the re-render.

**Phase 2: larger refactors, one per commit, each pinned.**
- Public `decoding.py` facade, `position_estimator`, `feature_args`, `block_cv_predict`, `session_zscore`. Add the estimator and trial rule to the frozen spec, added only when they differ from today's values.
- Config digest in `session_cache`; one cache module.
- `build_analysis_deck` split into sections; nightly step table.
- `pooled_collectors.py`; split `epoch_figures`/`beta_maps`/`spout_behavior`/`orofacial_cohort` with re-exports.
- `EncoderRunConfig` and an env-switch registry.
- Promote ITI and quiet-events helpers from `scripts/` into `wfield_local` (after the quiet-events deck window).
- Deck captions to tokens, one family at a time.
- `figures:` and `bootstrap:` yaml blocks (trial floors, N_BOOT).

**Decisions that need you:** imaging lick detector (§6 Q1); epoch bars pooled vs animal-weighted, or label them; encoder ENL scoring 2.0 s vs LP 1.0 s; `recovery_timing` level weighting; whether the per-animal trend-line panels in `orofacial_cohort.fig_epochs` are page material; deleting the 26 unregistered epoch PNGs and the orphan modules.

---

## Coverage

- **Read closely (by reviewers):** core infrastructure (config, paths, analysis_kit, parallel, both caches, map_collect, writeguard, sig_marks, nightly, camera_nightly, lick_detection, cat_points), trial_classes, the cited sections of the decoder and encoder modules, epoch_figures, grant_kit, deck_registry, spout_behavior, orofacial_cohort, lp_licks, working_trials, conftest, and a sample of tests and entry docs.
- **I re-checked myself:** every high finding at its cited lines (lick literal and yaml, pre_from, hemo_variants, run_locanmf, the DAQ contact chain, the response-window fallbacks, boot_p/Holm copies, run-name counts, library→scripts imports, epoch level vs delta, estimator max_iter and spec), plus mediums (cam1 filter, gate names, marker collision, curated_ids, build_analysis_deck length, engaged_trial_ids).
- **Sampled, or covered by grep only:** about a seventh of `scripts/` read in depth, the rest pattern-swept. Most test bodies. Large parts of `locanmf_analysis_deck`, `nightly_figs`, `combined_encoding`, `position_coding_directions`, `iti_licks`. Counts such as "about 560 caption numbers", "65 CV copies" and "58 env vars" are the reviewers' greps, not re-counted.
- **Not reviewed:** the rebuild branches (by instruction); `face_rhythm/` internals; `histology_map/`; dlc_train/anipose/calibrate bodies; `orofacial_clean` and `tongue_detect` detector bodies; `scripts/archive`. No code or tests were run, so no claim rests on a test result. The performance claims (the cohorts.yaml re-parsing, BLAS pinning in the parent process) are reasoned from the code, not measured.
