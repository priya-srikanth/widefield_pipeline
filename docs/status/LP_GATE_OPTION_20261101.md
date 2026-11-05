# LP-INCLUSIVE ENGAGEMENT GATE: an option, no renders (2026-11-01)

Priya 11-01, after `GATE_LP_CHECK_20261101.md`: *"should we do it anyway for consistency? most of our final analyses
will be based on LP licks"*. Answer: yes for the LP arm; the DAQ arms stay as they are. This adds the option. Nothing was
rendered. PROVISIONAL, n = 4 (PS92-95; cohort 2 not touched).

## The switch

`WIDEFIELD_GATE_SOURCE = daq` (default) `| lp`. The module is `wfield_local/gate_source.py`, and its docstring is the
full record.

* **daq (unset):** every gate is unchanged. `gate_source.gate_responded` returns its input object (no copy, no I/O),
  and no row column is added. The real-session tests below pin this exactly.
* **lp:** for the engaged / working / stopped gate and the S28 quit point **only**, a trial counts as a response if it
  has any of the following:
  * (a) the caller's own response, unchanged (the engaged arm, the DAQ hit, ...);
  * (b) a **DAQ spout contact** in [cue, min(cue + window, next cue)]. This is read with
    `_load_daq_events(..., allow_lp=False)`, so it is contact even inside an LP_licks render;
  * (c) an **LP attempt**: trial_attempts class **hit or incomplete** (`lp_licks.ATTEMPT_CLASSES`), matched to the DAQ
    cue by cue time, with the first LP lick (combined, else cam4) at most `window` after the cue. **jaw_only, no_try and
    unknown do not count** (jaw_only = a reaction, not an attempt; Priya's 10-21 review).
  * A cue with no LP verdict gets (a) or (b) only. That means no attempts row, an unmatched cue, or a session without
    video (PS93_0813).
  * `window` is the caller's own response window: `decode.max_rt_s` = 3.5 s for the imaging gates, the session's
    response window (3.5 s) for the rest gates, and `resp_s` = 2.0 s for the S28 quit rows.
* The switch never moves a trial between the lick and no-lick arms (that is `WIDEFIELD_LICK_SOURCE`). It never changes
  a hit rate, latency or lick count. In the S28 quit dict, `hit_before`, `hit_after` and `step_sharpness` stay on the
  row's own `hit`, and only the quit point moves.

## Where it is wired

The same one-line `gate_source.gate_responded(...)` call sits at each place a gate's `responded` vector is built.

| hook | what consumes it |
|---|---|
| `precue_engagement_states._session_tables` | `position_coding_directions._gate_all` (lick / miss_working / stopped), `attempt_coding`, `grant_kit`, `position_axes`, the precue-state analysis |
| `beta_maps._quit_mask` | 14 / 15r working and miss_working / stopped maps, `attempt_maps` (stopped), `position_reference_maps`, `hemo_map_control` |
| `nolick_decoder.categorize` (session engagement column) | `poststroke_compare` and other categorize users |
| `poststroke_compare.poststroke_engagement` | post-stroke engagement split |
| `rest_by_position._engaged_trials`, `rest_engagement.engaged_by_cue` | the rest-baseline gates (15r rest-referenced maps, rest_migration scripts) |
| `engagement_decomposition.session_trials` adds `gate_hit` (only under the switch); `quit_point.session_quit` scores the gate on it | S28 quit, `analysis_kit.session_behavior` |

Not wired, and stated in the module:
* the behaviour-pipeline gates `spout_behavior.reference_engagement` / `flag_engagement` (behaviour figures,
  `behavior_clips`);
* `orofacial_cohort.engagement_keep`, which reads the behaviour table's `engaged` column and already has its own
  `attempted` gate options;
* DAQ-arm diagnostic scripts that call `engagement_gate` directly (`disengagement_tests`, `enl_state_counts`,
  `stopped_trial_traces`, `channel_position_maps`).

Other files changed:
* `plot_lick_aligned_averages._load_daq_events`: new `allow_lp=True` keyword (default = unchanged).
* `session_cache.cached`: the gate tag.
* `map_collect.code_digest`: drops marked hook lines.
* `lp_licks`: the `_gateLP` output tag and stamp, plus a docstring pointer.
* `scripts/lp_gate_option_counts.py`: the counts below.
* `tests/test_lp_gate_option.py`.

## Found while wiring: the LP arm's gate was not purely DAQ

Under `WIDEFIELD_LICK_SOURCE=lp`, the shared lick loader (`plot_lick_aligned_averages._load_daq_events`) already returns
the LP train. So in the LP_licks arm:
* the imaging gate was already **LP-only**, because its engaged arm is the LP-attempted arm;
* so were `nolick_decoder.categorize` and `session_trials`, and therefore the S28 quit.
* The **rest-baseline gates** (`rest_engagement`, `rest_by_position`) read the canonical DAQ events / lick channel
  directly, so they stayed **DAQ-only**.

The switch puts all of them on one DAQ-or-LP rule. The counts below show that, for the imaging gate, LP-only and
DAQ-or-LP are **identical in every session**.

## Cache keys (default caches and frozen-model ids unchanged)

The gate source enters a key **only when it is `lp`**. This is the pattern of `lp_licks.cache_tag` /
`frozen_models.LP_FROZEN_ENV`.

* **`session_cache.cached`** appends `gate_source.cache_tag()` (`glp<sha>`) under the switch. The tag carries
  `GATE_VERSION` plus the LP inputs' identity. It covers every session-cached gate consumer: map_collect `bm-*` /
  `prm-*` pixel maps, categorize, CD arms. Trial features (`tf-*`) are exempt: they are built before any gate is applied,
  so they stay shared and warm.
* **map_collect source digests** (`beta_maps._session_maps_code`, `position_reference_maps._raw_maps_code`) hash the
  source of `_quit_mask`, `precue_engagement_states`, `rest_engagement` and `rest_by_position`.
  * Every hook line carries `# [gate-source hook]`, and `code_digest` drops marked lines.
  * Measured: `bm` 08dac9e3216a and `prm` 81053b92244b are identical at HEAD and in the working tree, so the warm DAQ and
    LP_licks map caches stay warm.
  * A test enforces that every hook line in a digested module is marked. Bump `GATE_VERSION` when the rule or a hook
    changes.
* **Frozen models** are not touched. Every frozen decoder trains on pre-stroke trial features, which no gate selects, so
  the spec deliberately does not carry the gate and a gate-LP render reuses the same references.
* **Other caches:**
  * lru_caches need nothing, because the env is fixed per process;
  * `grant_kit._boot_cached` is keyed on its data;
  * the one-off cost is the LP lick-times sidecar (`<cache>/lp_licks/*__times.pkl`). It is keyed on `lp_licks.py`'s
    bytes, so it recomputes once (CSV reads per session). Its values are unchanged.
* **Outputs:** `lp_licks.output_tag()` gains `_gateLP` (`LP_licks_gateLP`, `DAQ_gateLP`). As a result, writeguard
  refuses a gate-LP run's write into the plain LP_licks or default folders, and every figure is stamped.

## The pinned test

`tests/test_lp_gate_option.py`: **18 passed** (4 min 07 s; 6 tests read the share). Related suites (rest_engagement, one_engagement_gate, session_cache, cache_isolation, map_collect, lp_licks, writeguard, analysis_kit, nolick_analysis, stopped_trial_class): 131 passed.

* **Default byte-identical.** On PS92_0818 (acute), PS94_0921 (chronic) and PS95_0814 (pre), with the switch off:
  * the img gate through the hook equals `gate_lp_check`'s cached `ne_img` **exactly**;
  * `session_trials` rows carry no `gate_hit`;
  * `session_quit` equals the cached `quit_daq` dict key by key.
  The cache was written 11-01 by the pre-option code.
* **Default identity, synthetic:**
  * `gate_responded` returns the same object;
  * `beta_maps._quit_mask` with the switch off equals the pre-option computation;
  * the session-cache kind is unchanged by default, tagged under the switch, and `tf-*` is never tagged;
  * `output_tag` behaves as described;
  * every hook line in a digested module is marked.
* **Switch semantics:**
  * OR by cue index;
  * the windows: a DAQ contact is bounded by window and next cue, an LP attempt by its first lick;
  * a dropped cue or a non-attempt class adds nothing.
* **Switch reproduces gate_lp_check on real data.** On the same three sessions:
  * the gate equals the cached `ne_img_lp` exactly;
  * the quit equals the cached `quit_lp`, except `hit_before`, `hit_after` and `step_sharpness`, which gate_lp_check
    computed on an overwritten `hit`.

Run:

```
cd C:/Users/SabatiniLab/Github/wf_attempt && PYTHONPATH="$(cygpath -w $PWD);$(cygpath -w $PWD/scripts)" PYTHONWARNINGS=ignore MPLBACKEND=Agg WFIELD_INPUT_CACHE=C:/Users/SabatiniLab/wfield_cache WIDEFIELD_FIGURES_WORKING=C:/Users/SabatiniLab/cue_lick /c/Users/SabatiniLab/miniconda3/envs/locanmf/python.exe -m pytest tests/test_lp_gate_option.py -q
```

## What the switch moves: per animal first

Source: `python scripts/lp_gate_option_counts.py`.
* No new extraction: it reads gate_lp_check's cache (`wfield_cache/gate_lp_check/`), whose LP rule is this option's rule
  (verified exactly above on 3 sessions).
* Population: curated PS92-95, August-only pre, epochs from `epochs.epoch_of`, 108 sessions, full behavioural trial list.
* Per-session CSV: `M:/.../lp_r4_heatmap_20261013__clean_v36/imaging_encoding/gate_lp_check/lp_gate_option_counts.csv`.

Column definitions:
* **flip** = trials whose img-gate status (engaged vs not, 3.5 s) differs between DAQ contact and DAQ-or-LP. The rule
  only adds responses, so every flip is not-engaged to engaged.
* **quits moved** = S28 quit (2.0 s) at a different trial. **lost** = quit becomes censored.

| animal | epoch | sessions | trials | not-engaged DAQ -> DAQ-or-LP | trials flipped (sessions) | quits DAQ -> DAQ-or-LP | quits moved | lost | mean / max shift (min) |
|---|---|---|---|---|---|---|---|---|---|
| PS92 | pre | 8 | 2920 | 0 -> 0 | 0 | 0 -> 0 | 0 | 0 | 0 / 0 |
| PS92 | acute | 5 | 2681 | 590 -> 573 | 17 (1) | 5 -> 5 | 1 | 0 | 0.53 / 2.65 |
| PS92 | subacute | 2 | 593 | 0 -> 0 | 0 | 0 -> 0 | 0 | 0 | 0 / 0 |
| PS92 | chronic | 12 | 3963 | 0 -> 0 | 0 | 0 -> 0 | 0 | 0 | 0 / 0 |
| PS93 | pre | 8 | 4315 | 40 -> 35 | 5 (1) | 1 -> 1 | 0 | 0 | 0 / 0 |
| PS93 | acute | 4 | 2294 | 539 -> 517 | 22 (1) | 3 -> 3 | 1 | 0 | 1.00 / 3.99 |
| PS93 | subacute | 3 | 1616 | 226 -> 216 | 10 (1) | 2 -> 2 | 1 | 0 | 0.52 / 1.55 |
| PS93 | chronic | 12 | 6068 | 129 -> 86 | 43 (1) | 1 -> 1 | 1 | 0 | 0.59 / 7.02 |
| PS94 | pre | 8 | 4557 | 308 -> 304 | 4 (1) | 4 -> 4 | 1 | 0 | 0.08 / 0.63 |
| PS94 | acute | 6 | 3643 | 903 -> 894 | 9 (1) | 4 -> 4 | 1 | 0 | 0.23 / 1.41 |
| PS94 | subacute | 5 | 3000 | 472 -> 399 | 73 (4) | 5 -> 5 | 3 | 0 | 2.14 / 5.36 |
| PS94 | chronic | 8 | 4800 | 306 -> 269 | 37 (2) | 5 -> 4 | 2 | 1 | 0.72 / 5.28 |
| PS95 | pre | 8 | 5556 | 518 -> 514 | 4 (1) | 6 -> 6 | 1 | 0 | 0.08 / 0.62 |
| PS95 | acute | 1 | 720 | 0 -> 0 | 0 | 0 -> 0 | 0 | 0 | 0 / 0 |
| PS95 | subacute | 7 | 4275 | 1268 -> 1268 | 0 | 6 -> 6 | 0 | 0 | 0 / 0 |
| PS95 | chronic | 11 | 6600 | 751 -> 730 | 21 (2) | 6 -> 6 | 0 | 0 | 0 / 0 |

Pooled sums by epoch:

| epoch | trials | not-engaged DAQ -> DAQ-or-LP | flipped | quits moved / lost |
|---|---|---|---|---|
| pre | 17348 | 866 -> 853 | 13 (0.07 %) | 2 / 0 |
| acute | 9338 | 2032 -> 1984 | 48 (0.5 %) | 3 / 0 |
| subacute | 9484 | 1966 -> 1883 | 83 (0.9 %) | 4 / 0 |
| chronic | 21431 | 1186 -> 1085 | 101 (0.5 %) | 3 / 1 |

* **Reproduces gate_lp_check.**
  * 12 of 108 sessions move their quit, all later.
  * PS94_0921 (chronic) is the one quit lost. Its 34 not-engaged trials all become engaged.
  * The per-animal mean shifts equal GATE_LP_CHECK Q3: PS92 acute 0.53; PS93 1.00 / 0.52 / 0.59; PS94 0.08 / 0.24 /
    2.14 / 0.72 (0.23 here is rounding); PS95 pre 0.08.
  * Largest single shifts: PS93_0925 +7.0 min, PS94_0825 +5.4, PS94_0921 +5.3, PS93_0820 +4.0.
* **Moved trials are few and concentrated.**
  * 16 sessions change any img-gate status. The largest are PS93_0925 (43), PS94_0825 (34), PS94_0921 (34) and
    PS92_0818 (17).
  * Pooled, 0.07 % of pre and 0.5-0.9 % of post trials flip.
  * Only PS94 has flips in more than 2 sessions of an epoch (subacute, 4 of 5).
* **For the LP arm the imaging gate does not move at all.**
  * The script also computes the gate the LP_licks arm uses today (LP attempt only, on LP-covered trials). It is
    identical to DAQ-or-LP in all 108 sessions (0 trials flip).
  * So DAQ contact without an LP attempt never decides the gate at the reference spouts.
  * In the LP arm, the switch changes:
    * the **rest-baseline gates**: DAQ-only today, so those sessions move by the "flipped" column above;
    * the **S28 quit**: LP train today, DAQ-or-LP under the switch, not separately computed here;
    * the trials without an LP verdict, which get DAQ contact.
  * Their imaging maps / classes should therefore be unchanged except through the rest baseline. That is a prediction,
    not a measurement, since nothing was rendered.

## How to run an LP-arm render with it

Set both switches. Everything else is the usual command, for example:

```
WIDEFIELD_LICK_SOURCE=lp WIDEFIELD_GATE_SOURCE=lp python -m wfield_local.epoch_grant_figures --only 15r ...
```

* Outputs land under `.../LP_licks_gateLP/...`; writeguard refuses anything else.
* Session-cache entries get the `-glp<sha>` tag. Trial features are shared with the plain LP_licks render, so only the
  per-session gate consumers (pixel maps, categorize, CD arms) are cold.
* Frozen models are reused as-is.
* With `WIDEFIELD_GATE_SOURCE=lp` alone, a DAQ-arm render gets the DAQ-or-LP gate under `DAQ_gateLP`. Not requested.

## Limits

* Trial universe for the counts is the full behavioural list, not the imaging-kept subset (coverage, lick-free drops).
* The LP-arm S28 quit under today's rule (LP train) is not computed here. Doing so needs `session_trials` rows under the
  LP source, which means an h5 read per session.
* The switch-on path is checked exactly against gate_lp_check on 3 sessions. The other 105 rely on the rule being the
  same code path.
* In the LP_licks arm, trials without an LP verdict are dropped from the imaging features entirely, so (b) cannot add
  them back to the imaging gate. They are only added in the rest and quit gates, which use the full cue list.
