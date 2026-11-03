# Cache all frozen decoders: inventory, conversions, verification (11-02, later)

Priya 11-02: *"yes cache all frozen decoders"*; under the LP lick arm *"we should probably have a version of both daq
contact and lp lick-based"*. Follows `docs/status/FROZEN_DEFAULT_20261102.md` (epoch acc family + 12.1 stored by
default). Worktree `C:/Users/SabatiniLab/Github/wf_frozendef`, branch `analysis/frozen-default`, on top of 1d0fc040.
No page was edited, nothing was published, and nothing on M: was overwritten, deleted or renamed. All renders are new
folders under `G/frozen_default_20261102/allstored/` (G = `M:/MICROSCOPE/Priya/Widefield/labcams/grant_figures`).

## 1. Code

| file | change | opt-out |
|---|---|---|
| `wfield_local/grant_confusion.py` | `_collect_5c` mode **`paired`** (families 5r, 5ro; also `recovery_trajectory`, `scripts/prelim_numbers_frozen_vs_refit.py`) takes its frozen column (full model for post days, LOSO model per pre session) from `_stored_epoch_decoder`, the acc family's stored `epoch_decoder`. Modes `refit` and `paired_matched` do not ask the store (section 2). | `WIDEFIELD_EPOCH_FROZEN=0` (same family, same model) |
| same | **5b** (`fig_confusion_pre_post_working`, recompute path) takes its full + LOSO models from `_stored_epoch_decoder` (same bundle, rows, estimator and spec as acc). New data sidecar `data/grant_5b_confusion_working_<align>.csv` (pre / post_all / post_working counts per animal, the post_working panel split by session, and a `decoder` provenance row store / memory); the figure is unchanged. | `WIDEFIELD_EPOCH_FROZEN=0` |
| same | `_stored_epoch_decoder(..., tag=)` only labels the log line (`[5c]` / `[5r]` / `[5b]`). | - |
| `wfield_local/frozen_models.py` | `siblings()` groups by kind / align / source **and `post_s` and `lick_source`** (from the manifest spec). Fixes the false SPEC-CHANGED between 1.0 s and 2.0 s ENL models and between DAQ- and LP-trained models; `refreeze` now retires only same-window same-source models. No spec, spec_id or stored model changes (pinned). | - |
| `scripts/frozen_default_compare.py` | `--glob` (e.g. `epoch_5r*.csv`, `grant_5b_*.csv`); 5b long tables compared in file order with the provenance row dropped. | - |
| `tests/test_frozen_default.py` | +11 tests (30 in total; section 4). | - |

**Cache keys untouched.** `grant_confusion` and `frozen_models` are not hashed by `map_collect.code_digest` (digested:
`beta_maps.session_maps` / `_runs_to_blocks` / `_quit_mask`, `joint_basis._load_session`, `position_reference_maps`
map builders, and the whole modules `precue_engagement_states`, `quiet_periods`, `rest_engagement`,
`rest_by_position`, `restnl_mask`; `rest_frozen_decoder_nl._code_key` hashes `rest_frozen_decoder._collect` /
`_frame_samples`). None of those was edited. `_collect` / `_frame_samples` / `trial_features_cached` unchanged.

## 2. Inventory: every pre-stroke reference fitted per run (wfield_local/, scripts/; archive dirs skipped)

Line numbers are at this commit. A = frozen pre-stroke reference scored on post-stroke data.

### Converted here (A)
| where | what | family / consumer |
|---|---|---|
| `grant_confusion._collect_5c` mode `paired` (full l.736, LOSO l.818) | frozen column of frozen-vs-refit | epoch 5r, 5ro; `recovery_trajectory`; `prelim_numbers_frozen_vs_refit` |
| `grant_confusion.fig_confusion_pre_post_working` (full l.372, LOSO l.400) | 5b full + LOSO | `grant_figures --only 5b` |

### Already stored before this commit
`grant_confusion._stored_epoch_decoder` (acc / 5c / 5cr's `_epoch_arm`), `locanmf_frozen_decoder.frozen_decoder_models`
/ `pooled_frozen_encoder` (kind decoder / encoder), `locomotor_decoder.frozen_state_models` (state, 13s),
`rest_frozen_decoder(.py/_nl.py)` (rest_frozen, 12.1).

### NOT converted, by design (B / C)
- `_collect_5c` mode `refit`: no frozen column at all (the in-memory full fit there is computed and discarded).
- `_collect_5c` mode `paired_matched` (5rm, 5rmo; `scripts/june_august_drift.py`): `_matched_frozen` (l.~546) fits a
  size-matched RANDOM subset of pre blocks, seeded per SCORED session, i.e. one model per post session x variant. A
  resampled control, not one reference; storing it means hundreds of models per arm, and under the LP arm a
  DAQ-trained draw would need the DAQ pool loaded in the LP run. Listed as a decision (section 7).
- `rest_frozen_decoder` `matched_frozen` (l.~1016) / `_nl` matched (l.~373), `substitute_code` l.238: same reasoning.
- `_refit_pred`, `refit_predictions`, `transfer_matrix._matched` (15g), `rotation_maps` (15h), `enl_decode`
  transfer / shared, `iti_where_drift`: per-epoch / per-session refits by design.
- Within-session CV decoders (`decoder_c_sweep`, `precue_*`, `locanmf_decoder_qc` / `_ablation`, `locanmf_position_
  decoder` l.718, `poststroke_compare.recoding_test`, `pooled_frozen_loso` within arm, `misdirected_lick_decode`,
  `rest_position_decode`, `session_vs_frozen`, ...).
- Not pre -> post (D): per-session maps / encoders (`beta_maps` 14m, `combined_encoding`, `movement_encoding`),
  pre-only or June baselines (`cross_day_transfer`, `locanmf_xsession_generalization`, `locanmf_cross_mouse`),
  geometry / RSA, `position_axes` (cosine between a pre and a refit post axis), `allsession_encoder`,
  `grant_kit._pre_reference` (mean-pattern templates, not an estimator; JOBS 6 / 7 / 8 / 10 / 11, mat / scal / 10e /
  10cs / 12s / 12b), the joint basis and FaceRhythm factor fits (persisted elsewhere).

### NOT converted: frozen pre-stroke references that are still fitted in memory (A), listed for a decision
Each needs its own `kind` (different estimator settings or feature space from `epoch_decoder`), its own spec, and its
own verification run; together they are the large part of "cache all", so they are listed rather than converted here.

| # | where (file:line) | what | feeds | size / note |
|---|---|---|---|---|
| 1 | `position_coding_directions.run_animal` l.598: `_class_confusions` LOSO l.505, full l.511 | position decoder (C=0.5, max_iter 2000, not `_pipe`) on pre engaged, scores post lick / miss_working / stopped / early / late | `coding_direction.json` -> JOBS 3a, 5b (`_from_json`, preferred over the recompute when the JSON exists), `miss_vs_stopped`; nightly | small (own kind) |
| 1b | same, l.717 / 731 / 754 / 873 / 898, `_diagnostics` l.385, `_disc().fit` l.239 | engagement axis, per-position CDs (full + LOSO), pairwise CDs, 4 methods | same | medium |
| 2 | `precue_engagement_states.run_animal` l.196: `_pipe().fit` l.263, `_disc().fit` l.286 | position decoder + state discriminator, pre -> all states | CLI only | small, **but the module is hashed into the beta-map AND prm map-cache digests: any edit colds ~33 GB of map caches. Not touched; needs Priya.** |
| 3 | `nolick_decoder.analyse_animal` l.527: LOSO l.665, full l.672 | pre engaged -> post late_rewarded / undetected / working / disengaged | `build_reference` from `nightly_figs` l.693 / 716 | small (own feature space `session_features`) |
| 4 | `poststroke_compare`: `decode_matched` l.267 / 269; `looks_like_which` l.316, `undetected_state_split` l.809 (`lick_pipe`); `fits_engaged_distribution` l.421 (`balanced_fit`); fallbacks l.565 / 577 / 602 / 872 when `frozen(d)` is None | lick-only arm restricted to the post session's preserved positions (spec depends on the post session); lick discriminator on a seeded balanced pre sample; balanced fit with an RNG that advances across folds (freezing changes published numbers unless the RNG is kept) | `poststroke_section_g` -> `section_g.json` -> JOBS 3b, 5, `section_g_figures` | medium; ambiguous specs |
| 5 | `attempt_coding.run_animal` l.161: `fit_axes` full l.219, LOSO l.224 | per-spout CD axes on pre video hits | standalone CLI (decoder half already stored) | small |
| 6 | `cd_trajectories.analyse_animal` l.1049 (l.1111-1184) | z-score stats, CIM, `fit_directions` on pre success trials, projected frame by frame | nightly l.914 / 934; `scripts/cd_loo_sessions.py` | medium |
| 7 | `scripts/june_august_drift.py decoder_rows` l.113 | August-pre-only decoder, refitted per scored session | script | small; different training set from `epoch_decoder` (needs its own id) |
| 8 | `frozen_encoder.fit_animal` l.1169 (+ `build_templates` l.727; `frozen_relabel` l.181, `scripts/partial_refit.py` l.119 private pickles) | frozen ridge encoder on August pre | encoding page | large |
| 9 | ITI / cued-lick read-outs (RidgeCV, LOSO per pre + full pre for post), ~12 copies: `iti_licks.py` l.352 / 729 / 1283, `iti_basis_decoder` l.116, `iti_contact_check` l.64, `iti_first_lick` l.95, `iti_lick_order` l.130, `iti_nmatch_check` l.48, `iti_where_decoder` l.170, `iti_where_3axis` l.61, `iti_where_angle_reach` l.66, `iti_where_scatter` l.65, `iti_decoder_contrib` l.58, `lick_gain_prediction` l.61 | pre fit -> post | scripts | small each; a shared stored helper first |

Suggested order: 1 (feeds 3a / 5b nightly), 4 (3b / 5), 3, 5 / 6, 7, 9 (one helper), 8; 2 only with a decision on
the digest.

## 3. Verification: real renders, stored vs in-memory (exact, cell by cell)

Renders: `epoch_grant_figures --only 5r 5ro --output <dir>` (5r and 5ro share the one paired collector, so they were
rendered together; 5 arms) and `grant_figures --only 5b -j 1 --output <dir>`. LP arm = the `G/LP_licks` settings
(`WIDEFIELD_LICK_SOURCE=lp`, DAQ gate, June + August pre, `WIDEFIELD_ENL_WINDOW_S=1.0`); LP outputs land in
`allstored/LP_licks/<family>_<arm>/` (`lp_licks.output_dir`). Comparisons: `allstored/compare/<tag>_{summary,cells}.csv`
(`scripts/frozen_default_compare.py epoch A B --glob ...`).

| comparison | training set | result |
|---|---|---|
| 5r / 5ro DAQ 2 s: stored (`5r_daq_stored`, 40 / 40 `frozen-hit`) vs `=0` (`5r_daq_inmemory`) | same | **IDENTICAL**: 75 CSVs, 0 differing cells (PNG byte-identical, checked on cue / working) |
| 5b DAQ 2 s: stored (8 / 8 `frozen-hit`) vs `=0` | same | **IDENTICAL**: both sidecars (5280 cells each), both PNGs byte-identical |
| 5r / 5ro LP: stored LP-trained (`WIDEFIELD_LP_FROZEN=lp`, 40 hits) vs `=0` (in memory = LP-trained) | same | **IDENTICAL**: 75 CSVs, 0 cells |
| 5b LP: stored LP-trained vs `=0` | same | **IDENTICAL** |
| 5r / 5ro LP: **stored DAQ-trained (new default)** vs `=0` | different | DIFFERENT, expected: 5069 cells, all in the gap / overall files (every `5r_refit_by_position` file, i.e. the refit column, is identical) |
| 5b LP: stored DAQ-trained vs `=0` | different | DIFFERENT, expected: 4376 count cells |

Every stored lookup was a `frozen-hit` on the models the acc family froze earlier today (same spec ids: e.g. cue PS92
`epoch_decoder_cue_joint76d884_c77678327fbf`; LP ENL 1.0 s DAQ-trained `..._precue_joint76d884_5b9b980e58d1`, LP-trained
`..._08d4f7bce4cf`). No model was fitted, no SPEC-CHANGED fired. The 5b pooled post-cue far-contra recalls equal the acc
family's cue / working bars (DAQ: pre .896, acute .322, subacute .631, chronic .740; LP below), so 5b, 5r and acc now
score with one model in each setting.

## 4. Tests

`tests/test_frozen_default.py`: 30 pass (19 earlier + 11 new):
- `_collect_5c` mode `paired` reaches the store once; its records (frozen AND refit columns, the pre LOSO list and every
  post day) equal the opt-out's EXACTLY (variants working, lick); the paired frozen column equals mode `frozen`'s
  predictions; modes `refit` / `paired_matched` never ask the store.
- 5b end to end (synthetic bundle, no data): stored and opt-out sidecars `assert_frame_equal(check_exact=True)`; one store
  call per window with the acc spec; per-session split sums to the drawn panel; source pins.
- LP arm: the spec is DAQ-trained (no `lick_source`, every pre session of the animal) by default and LP-trained
  (`lick_source=lp`, LP-kept pre) with `WIDEFIELD_LP_FROZEN=lp`; the two ids differ.
- `siblings()`: 1.0 s / 2.0 s / LP-trained models are not siblings, a changed training set at the same window still is;
  `load_or_fit` returns frozen-new (not SPEC-CHANGED) across windows, SPEC-CHANGED within, and a hit afterwards; the
  four spec_ids are unchanged and distinct.

Related suites (frozen_models, store-move, frozen_vs_refit, class_confusions, grant_kit, recovery_trajectory,
lp_licks, stopped class, one pooling, ood, poststroke stored model, substitute_code, analysis / preprocess deck,
locomotor state, twinned axes, rest_frozen_decoder_nl): **275 / 275 pass** (5.1 min). ruff clean on every changed file.

## 5. LP arm, both trainings (far contra; LP-trained -> DAQ-trained default)

### 5r gap = refit - frozen (positive = the within-session decoder beats the frozen one)
Per animal = mean over sessions (`_sessions.csv`); pooled = the bar with its mark. The refit column is identical in both
(it never touches the store); only the frozen column moves.

| arm | animal | pre | acute | subacute | chronic |
|---|---|---|---|---|---|
| lick / lick | PS92 | -0.079 -> -0.067 | +0.411 -> +0.534 | +0.149 -> +0.200 | +0.056 -> +0.128 |
|  | PS93 | -0.039 -> -0.044 | +0.319 -> +0.308 | +0.097 -> +0.102 | +0.047 -> +0.136 |
|  | PS94 | -0.054 -> -0.046 | +0.041 -> +0.067 | +0.197 -> +0.167 | +0.298 -> +0.307 |
|  | PS95 | -0.039 -> -0.026 | +0.200 -> +0.250 | -0.038 -> -0.029 | -0.032 -> -0.013 |
|  | pooled | -0.049 -> -0.042 | +0.255** -> +0.308** | +0.103* -> +0.103* | +0.083* -> **+0.131\*\*** |
| ENL working | PS92 | -0.099 -> -0.054 | +0.114 -> +0.150 | +0.007 -> +0.014 | +0.038 -> +0.084 |
|  | PS93 | -0.045 -> -0.047 | +0.013 -> +0.006 | -0.029 -> -0.058 | -0.017 -> -0.051 |
|  | PS94 | -0.134 -> -0.140 | +0.046 -> +0.012 | +0.077 -> +0.020 | +0.068 -> +0.020 |
|  | PS95 | -0.085 -> -0.077 | +0.167 -> +0.167 | +0.010 -> +0.001 | +0.040 -> +0.048 |
|  | pooled | -0.085 -> -0.077 | +0.076** -> +0.074** | +0.021 -> -0.006 | +0.035* -> +0.024* |
| ENL lick | PS92 | -0.099 -> -0.054 | +0.142 -> +0.174 | +0.072 -> +0.064 | +0.029 -> +0.075 |
|  | PS93 | -0.045 -> -0.047 | +0.030 -> +0.023 | -0.043 -> -0.073 | -0.024 -> -0.058 |
|  | PS94 | -0.134 -> -0.140 | +0.017 -> -0.010 | +0.065 -> +0.008 | +0.094 -> +0.046 |
|  | PS95 | -0.085 -> -0.077 | -0.075 -> -0.075 | -0.002 -> -0.006 | +0.037 -> +0.045 |
|  | pooled | -0.085 -> -0.077 | +0.069* -> +0.072* | +0.017 -> -0.010 | +0.036** -> +0.025* |
| cue working | PS92 | -0.089 -> -0.094 | +0.270 -> +0.270 | +0.113 -> +0.113 | +0.057 -> +0.031 |
|  | PS93 | -0.094 -> -0.082 | +0.151 -> +0.153 | +0.033 -> +0.055 | -0.004 -> -0.003 |
|  | PS94 | -0.081 -> -0.085 | +0.063 -> +0.106 | +0.147 -> +0.161 | +0.232 -> +0.246 |
|  | PS95 | -0.061 -> -0.054 | +0.217 -> +0.167 | -0.043 -> -0.008 | -0.011 -> -0.014 |
|  | pooled | -0.078 -> -0.075 | +0.154** -> +0.166** | +0.059* -> +0.084* | +0.060** -> +0.057** |
| cue lick | PS92 | -0.089 -> -0.094 | +0.329 -> +0.330 | +0.050 -> +0.050 | +0.042 -> +0.016 |
|  | PS93 | -0.094 -> -0.082 | +0.129 -> +0.136 | +0.022 -> +0.045 | +0.019 -> +0.020 |
|  | PS94 | -0.081 -> -0.085 | +0.062 -> +0.123 | +0.166 -> +0.172 | +0.243 -> +0.254 |
|  | PS95 | -0.061 -> -0.054 | +0.225 -> +0.150 | -0.102 -> -0.067 | -0.014 -> -0.017 |
|  | pooled | -0.078 -> -0.075 | +0.178** -> +0.197** | +0.046 -> **+0.068\*** | +0.065** -> +0.062** |

The frozen accuracy behind these gaps is the acc family's (same model): the LP-arm far-contra frozen tables are
`FROZEN_DEFAULT_20261102.md` section 5 (lick / lick acute pooled .314 -> .260 etc.).

**Mark changes, 5r / 5ro, LP-trained -> DAQ-trained** (all positions; bar files):
- 5r gap, far contra: cue lick subacute none -> \*; lick / lick chronic \* -> \*\*; ENL lick chronic \*\* -> \*.
- 5r gap, other positions: cue working subacute fI \*\* -> \*; lick / lick acute nC none -> \*\*, chronic nC none -> \*,
  chronic fI \* -> none, chronic fM \*\* -> \*; ENL working acute fI \* -> none, chronic fI \*\* -> \*.
- 5ro overall: lick / lick subacute gap \* -> none; ENL lick subacute frozen \*\* -> \*.

Reading: the DAQ-trained frozen decoder leaves a LARGER lick-window far-contra gap (acute pooled +.255 -> +.308; PS92
+.41 -> +.53; chronic +.08 -> +.13, mostly PS92 / PS93), i.e. the same direction as the acc drop (.314 -> .260): the
within-session decoder is untouched, the frozen one does worse. The ENL arms move by <= .03 pooled, cue arms by <= .025.
Acute far-contra gap > 0 holds in every arm pooled under both trainings.

### 5b confusions, far-contra row recall (pooled post-stroke panel split by epoch from the sidecar)
5b has no marks and no epoch split on the figure; the epoch columns are the sidecar's per-session counts summed per
epoch (they sum to the drawn "quit period REMOVED" panel = "all post"). pre = the LOSO panel.

| window | animal | pre | acute | subacute | chronic | all post |
|---|---|---|---|---|---|---|
| post-cue | PS92 | 0.876 -> 0.879 | 0.262 -> 0.262 | 0.562 -> 0.562 | 0.733 -> 0.759 | 0.569 -> 0.584 |
|  | PS93 | 0.806 -> 0.795 | 0.136 -> 0.133 | 0.440 -> 0.419 | 0.685 -> 0.683 | 0.538 -> 0.532 |
|  | PS94 | 0.895 -> 0.900 | 0.397 -> 0.355 | 0.659 -> 0.643 | 0.603 -> 0.590 | 0.561 -> 0.540 |
|  | PS95 | 0.943 -> 0.937 | 0.283 -> 0.333 | 0.755 -> 0.717 | 0.883 -> 0.887 | 0.797 -> 0.791 |
|  | pooled | 0.883 -> 0.880 | 0.283 -> 0.272 | 0.652 -> 0.627 | 0.733 -> 0.736 | 0.622 -> 0.615 |
| ENL (1.0 s) | PS92 | 0.291 -> 0.248 | 0.152 -> 0.117 | 0.131 -> 0.119 | 0.264 -> 0.217 | 0.217 -> 0.177 |
|  | PS93 | 0.327 -> 0.331 | 0.159 -> 0.166 | 0.237 -> 0.263 | 0.346 -> 0.380 | 0.293 -> 0.320 |
|  | PS94 | 0.543 -> 0.546 | 0.292 -> 0.319 | 0.316 -> 0.373 | 0.291 -> 0.340 | 0.298 -> 0.343 |
|  | PS95 | 0.402 -> 0.395 | 0.150 -> 0.150 | 0.290 -> 0.300 | 0.341 -> 0.337 | 0.310 -> 0.310 |
|  | pooled | 0.400 -> 0.392 | 0.206 -> 0.207 | 0.279 -> 0.306 | 0.316 -> 0.327 | 0.285 -> 0.297 |

These equal the acc family's LP cue working / ENL working far-contra cells under each training (FROZEN_DEFAULT section
5), as they must. DAQ arm (stored = in memory): post-cue pooled .896 / .322 / .631 / .740, ENL 2 s .507 / .251 / .340 /
.410 (`compare/5b_farcontra_5b_daq_stored.csv`).

## 6. Store, runs (11-02, EST; shared box: <= 4 BLAS threads in total = 2 processes x 2, free RAM >= 40 GB throughout)

- `local_dir()` = `C:/Users/SabatiniLab/frozen_models`, exists. **Store count 2067, unchanged**: every lookup in every
  render was a hit; this commit writes no model. The server store was not touched; nothing published.
- `siblings()` change: warning / supersede scope only; no spec_id or stored model changes (pinned).

| run | wall |
|---|---|
| 5b DAQ stored / in memory | 4.6 / 3.1 min |
| 5b LP DAQ-trained / LP-trained / in memory | 5.6 / 5.1 / 3.7 min |
| 5r + 5ro DAQ stored (5 arms) | 41.6 min |
| 5r + 5ro DAQ in memory | 33.7 min |
| 5r + 5ro LP DAQ-trained / LP-trained / in memory | 46.0 / 42.9 / 36.2 min |
| tests: test_frozen_default (30) / related suites (275) | 3.7 / 5.1 min |

(5r renders ran two at a time, so their walls include sharing the CPU.)

## 7. Open / for Priya

1. The ten groups of frozen pre-stroke references in section 2 "NOT converted" (about 25 fit sites) are still fitted in memory per run. Converting
   them is the remaining part of "cache all"; each needs its own kind / spec and a verification render. Suggested order
   is in section 2. `precue_engagement_states` needs a decision first: editing it colds ~33 GB of map caches.
2. `paired_matched` (5rm / 5rmo): store the size-matched draws (one per scored session), or keep them as an in-memory
   resampled control. Under the LP arm they are currently LP-trained (the LP pool), unlike the rest of the family.
3. Under the LP arm the 5r far-contra lick-window gap grows with the DAQ-trained default (section 5); re-render the page's
   5r / 5b LP figures if those are quoted.
4. Commit: on `analysis/frozen-default`; nothing pushed or merged.

---

# PASS 2 (11-02, evening): the in-memory references of section 2, converted

Same worktree / branch, on top of 8debc35e. Brief: convert section 2's "still fitted in memory" list in the suggested
order, each family with its own `kind`, its own opt-out, exact pinned tests and a real render stored vs opt-out. No page
was edited, nothing was published, nothing on M: was overwritten, deleted or renamed: every render is a NEW folder under
`G/frozen_default_20261102/allstored2/<family>/` and every comparison table is in `allstored2/compare/<tag>_{summary,
cells}.csv` (`scripts/frozen_default_compare.py tree A B`, new mode: every json / npz / csv / csv.gz both folders share,
leaf by leaf, exact; provenance keys such as `frozen_model` dropped).

**Cache keys untouched (checked before each edit).** Digested sources: `map_collect.code_digest` (beta_maps
`session_maps` / `_runs_to_blocks` / `_quit_mask`, `joint_basis._load_session`, `position_reference_maps` builders,
whole modules `precue_engagement_states`, `quiet_periods`, `rest_engagement`, `rest_by_position`), `restnl_mask`
(module), `rest_frozen_decoder_nl._code_key` (`_collect`, `_frame_samples`). None of the modules edited here is among
them; no `session_cache` kind is built from source text. `cd_trajectories.courses_cache_kind` hashes the direction
VALUES, and the stored directions equal the in-memory ones, so the CD courses caches stay warm (DAQ).

Shared code: `frozen_models.family_on(env)` (one opt-out rule: 0 / false / no / off) and `frozen_models.
daq_trained_in_lp()` (WIDEFIELD_LICK_SOURCE=lp without WIDEFIELD_LP_FROZEN=lp: load the DAQ-trained model, whose
training set is the DAQ arm's pre set; `load_or_fit` refuses to fit it in the LP run).

Renders ran against SNAPSHOTS of the worktree (copied to the scratchpad before each batch), one BLAS thread per process,
at most 4 processes in total.

## P2.1 Conversions

| # | family | file | kind (spec align carries) | opt-out | LP arm |
|---|---|---|---|---|---|
| 1 | no-lick reference (`nolick_decoder.analyse_animal`: full + LOSO) | `wfield_local/nolick_decoder.py` | `nolick_reference` (`<align>-rt<2 / respwin>-f<digest of the pooled feature columns>`; source roi / joint + basis id) | `WIDEFIELD_NOLICK_FROZEN=0` | DAQ-trained default (train = the animal's pre sessions in the date list), LP-trained with `WIDEFIELD_LP_FROZEN=lp` |
| 2 | attempt-coding per-spout axes (`attempt_coding.run_animal`: full + LOSO, method dom) | `wfield_local/attempt_coding.py` | `attempt_axes` (`<align>-<methods>`; joint basis id) | `WIDEFIELD_ATTEMPT_FROZEN=0` | **LP-only by nature** (the reference class is the VIDEO hit; the module refuses to run without the LP source): spec always `lick_source=lp` |
| 3 | June -> August drift decoder (`scripts/june_august_drift.decoder_rows`, the full (not size-matched) frozen column) | `scripts/june_august_drift.py` | `drift_decoder`, ONE spec per month (`cue-june` / `cue-aug`; the two training sets never share an id); payload full + LOSO, so every `score` training set (June LOSO, all June, Aug LOSO, all Aug) is served by one stored fit | `WIDEFIELD_DRIFT_FROZEN=0` | DAQ-trained default (train = the animal's DAQ pre sessions of that month), LP-trained with `=lp` |
| 4 | position coding directions (`position_coding_directions.run_animal`): the per-class confusion decoder (StandardScaler + LR C=0.5, max_iter 2000) and every pre-stroke direction (engagement axis; per-position axes + poles + raw axes; their LOSO refits; pairwise axes + their LOSO refits) | `wfield_local/position_coding_directions.py` (`fit_confusion_decoder` / `fit_reference` hold the fits, moved verbatim out of `_class_confusions` / `run_animal`; the opt-out runs the same functions) | `coding_decoder` (align) and `coding_directions` (`<align>-<methods>`) | `WIDEFIELD_CODING_FROZEN=0` (both) | DAQ-trained default (train = every pre session of the animal), LP-trained with `=lp` |

## P2.2 Verification: real renders, stored vs opt-out (exact, cell by cell)

| family | render | training set | result |
|---|---|---|---|
| 1 nolick | `build_reference` as the nightly calls it (dates = every curated date, all phases), PS93 + PS94, roi basis, both cuts, both windows (8 analyses, 39 post sessions scored); stored (`alldates_PS93_PS94/daq_stored`: 1 frozen-new + 7 frozen-hit on models the pre-date render had just frozen) vs `=0` | same | **IDENTICAL**: 37801 leaves |
| 2 attempt | `attempt_coding --out` (LP, all 4 animals x precue / cue / lick), stored (12 frozen-new) vs `=0` | same (LP) | **IDENTICAL**: 19 files (12 trial tables, 3 summaries, 3 contrasts, run.json), 0 of ~3.25 M leaves differ |
| 3 drift | `june_august_drift --reuse-distance` (DAQ), stored (8 frozen-new) vs `=0` | same | **IDENTICAL**: decoder_rows / distance_rows / summary_by_animal, 1984 cells |
| 4 coding | `position_coding_directions -j 1` (DAQ; 4 animals x ENL / cue / lick; methods dom + dom_orth), three renders: stored (24 frozen-new), `=0`, and the UNMODIFIED code at 8debc35e (`daq_head_8debc35e`) | same | **IDENTICAL in every model-derived leaf** (582360 leaves of `coding_direction.json`; PNGs not compared). The only differing leaves are `sessions[i]/variance_captured`, a joint-basis PROJECTION diagnostic computed before any model, which jitters at <= 9e-8 between ANY two runs: 3 leaves stored vs `=0`, 60 HEAD vs `=0`, 57 HEAD vs stored. So the refactor reproduces the old code and the store reproduces the in-memory fit |

Render notes, family 1: `python -m wfield_local.nolick_decoder` crashes after its first animal at 8debc35e
(`nolick_analysis.summarize`, print only, raises KeyError on an arm that has `n` but no `balanced_accuracy`, i.e. any
arm under 30 trials) -- pre-existing, not touched; the nightly calls the same `build_reference`, so its no-lick stage
fails the same way when that arm occurs. The renders used a scratch driver (`nolick_render.py`) that wraps
`summarize`, memoises `session_features` so the opt-out pass re-reads the stored pass's features, and calls
`build_reference` twice in one process (stored, then `=0`). Nothing computed is changed by the driver.
