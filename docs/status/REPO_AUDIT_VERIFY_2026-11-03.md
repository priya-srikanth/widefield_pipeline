# VERIFY: independent check of AUDIT.md (2026-11-03)

This was a read-only check. No repo file was edited. The only things run were git read commands, greps, file listings on M:, and
four small pandas/json reads: 8 `tongue_jaw_mismatch.csv` files, the `response_window` field of every
`Behavior_logs/Widefield/*/gui_config.json` (258 files), `config.load_sessions()` and `rest_engagement._response_window_s` on 3
sessions. Unless stated otherwise, line numbers are on **main** (`widefield_pipeline`); "attempt" means `wf_attempt`
(branch `analysis/attempt-coding`). Both trees were clean (`git status` empty). Merge-base `2c6fcd89`, main at `f1053aa4`,
attempt at `3834b3d9`.

---

## Claim 1: the two branches hold diverged copies of the shared package. **CONFIRMED** (with small corrections)

- **Counts.** I ran `git diff --name-only <mb> main|HEAD -- scripts wfield_local` and took `comm -12`. Main touched **297** files
  (the report says 295; main has moved since). Attempt touched **218**. **84** files were touched on both, and **60** of those
  still differ (`git diff --quiet main HEAD`). The 8 `wfield_local` files that differ are exactly the ones listed:
  frozen_encoder, lp_licks, nolick_decoder, orofacial_cohort, rest_by_position, rest_engagement, sig_marks, stray_licks.
- **Gate hook.** Confirmed. `wfield_local/gate_source.py` exists only on attempt. Its hooks are at `rest_engagement.py:68-71`,
  `rest_by_position.py:183-185`, `nolick_decoder.py:257-258`, plus `beta_maps.py:331`, `poststroke_compare.py:725` and
  `precue_engagement_states.py:181`. With the default `daq` setting they are the identity (`gate_source.py:11-12`, the same
  object is returned, and this is pinned by a test). **IMPACT:** behaviour changes only under `WIDEFIELD_GATE_SOURCE=lp`, and
  outputs then go to `*_gateLP` folders. So this is latent for the default cohort-1 numbers.
  - One correction to the report's table: "main = DAQ contact only" is not quite right. Under `WIDEFIELD_LICK_SOURCE=lp`,
    main's `nolick_decoder.categorize` already uses **LP licks only**, because `_load_daq_events` swaps the train in
    (`plot_lick_aligned_averages.py:62-72`). Main's rest gates stay DAQ-only. So main is itself internally mixed in the LP arm
    (`gate_source.py` docstring, "WHAT THE LP ARM'S GATE WAS BEFORE").
- **`"2026"+mmdd`.** Confirmed exactly. Patterns: `["']2026["'] *\+` gave 79 on attempt and 32 on main; `f["']2026\{` gave 21 on
  attempt and 35 on main. Totals: attempt **100**, main **67**. `date8\(` gave attempt **0**, main **29**.
  - The deeper cause: attempt's `config.py` has **no `date8` at all**. `git diff main HEAD -- wfield_local/config.py` shows
    -124 lines on the attempt side, so attempt never received the cohort2 / year-aware config.
- **`[:4]` animal id.** Confirmed with `label["']?\]?\[:4\]|lab\[:4\]|label\[:4\]`: attempt **96**, main **36**. (Any `[:4]` at
  all: attempt 234, main 139.) The example `nolick_decoder.py:131/505/552` is correct on attempt; main uses
  `split("_")[0]` there.
- **IMPACT of both.** None for cohort 1 (PS92-95 are 4-character ids, all dates are in 2026). Cohort 2 breaks on attempt, both
  through the ids and through the missing config.
- **Scripts not pinning `sys.path`.** Confirmed: **150 of 391** top-level `scripts/*.py` on attempt. 348 of them import
  `wfield_local`, and 150 of those have no `sys.path.insert|append`.
  - The editable install points at main:
    `locanmf/Lib/site-packages/__editable___widefield_pipeline_0_1_0_finder.py` maps `wfield_local` to
    `C:\Users\SabatiniLab\Github\widefield_pipeline\wfield_local`.
  - `iti_recovery_measure.py:36` and `lick_cued_aim_by_spout.py:31` insert only `scripts/`. So they **do** resolve to main's
    package unless `PYTHONPATH` is set. The report left this UNVERIFIED; it is now resolved.
  - Attempt's CLAUDE.md:745 says "no PYTHONPATH".
  - **IMPACT:** real for any attempt-branch number produced by `python scripts/x.py` without `PYTHONPATH`. Whether a
    published number was affected depends on how each run was launched, which the code cannot show.

## Claim 2: the "August-only pre" cut-off is hard-coded as 0806 in some places and 0801 in others. **PARTLY**

- **Locations.** `mixed_models.py:63/:78` (`>= "0806"`) are correct on both trees. `frozen_encoder.py:1641/1736` are **main's**
  line numbers; on attempt they are :1658/:1762 (the report says the numbers are from attempt). `LATE_PRE_FROM` is at
  `orofacial_cohort.py:1530` on main and :1480 on attempt, not :1504.
- **Counts** (`[<>]=? *["']0806["']` and the same for 0801): attempt has 22 / 58, main has 68 / 51. So "at least 60" holds.
- **Does 0805 reach a ">= 0801" reader?** **No, for cohort 1, on the current data.**
  - `epochs.epoch_of("PS9x_0805")` returns "pre", because `session_phase` ignores `cross_session_exclude`
    (`config.py:423-437`).
  - But every 0801 reader I traced gets its labels from the curated set:
    - `frozen_encoder.main --all` goes through `combined_encoding.sessions_available()` (`combined_encoding.py:1767`), which
      calls `ak.curated_sessions`, which calls `phase_labels`, which applies `cross_session_exclude` ("0805",
      `animals.yaml:193`).
    - `orofacial_cohort.collect` uses `ak.curated_sessions` (attempt :613).
    - `frozen_group_leakage.py:72/79` uses `ce.sessions_available()`. That is curated, **not a raw glob** as the report says.
  - The raw-glob readers (`frozen_r2_maps.py:72/220`, `frozen_spout_maps.py:48`, `frozen_area_maps.py:102`,
    `frozen_encoder.summarise_drop:1640`, `iti_licks.run_labels:711-713`) glob session files that the curated `--all` run wrote.
  - I listed **every** `imaging_encoding/frozen_encoder_*/*/sessions/` plus `dropspout/sessions` on M: (85,364 entries). There
    is **no 0805 file** and no June file. The main run dir `frozen_encoder_gate-attempted_v7g3_t3d_frA/joint/sessions` holds
    107 sessions, all August or later.
  - I did not list `fr_direction`'s `FRD` folders.
- **IMPACT:** latent for cohort 1. 0801 and 0806 select the same sessions because the inputs never contain 0805. It would bite
  only if someone ran `frozen_encoder` with 0805 named explicitly. For cohort 2 every MMDD string compare is wrong, as the
  report says.

## Claim 3: `iti_recovery_measure.py` mixes baselines (all-pre offsets, August-pre deficit). **REFUTED in practice; wording misleading**

- The code is as described. `:61` sets `ref = p[p.epoch == "pre"]...mean()` (all pre in the input). `:121`, `aim_deficit`,
  calls `mixed_models.behaviour_deficit` (`>= "0806"`, `mixed_models.py:78`). The docstring at `:16-17` says "all-pre-lick
  mean ... as approved".
- **But the input has no June sessions.** `where_preds_z` is built over `iti_licks.run_labels()`
  (`iti_licks.py:711-713`), which is the frozen-encoder run's session jsons filtered `>= "0801"`. June is a separate reference
  set (`iti_june_ref.py`, `june_ref/`).
- I checked the published `.../imaging_encoding/iti_licks/recovery_measure_sessions.csv` (written Oct 27 23:09). It has 31 pre
  rows, all **0806-0814**: PS92 8, PS93 7, PS94 8, PS95 8. So "all pre" equals "August pre (>= 0806)", and the two baselines
  are the **same set**. `weighted_abs` and the pre noise floor are unaffected.
- **Documented?** Not as a mixed baseline. The docstring's "all-pre" is literally true but implies June. DECISIONS 2026-10-27
  (late, ~line 22159) records the first-lick choice only. I recommend rewording the docstring.
- **Residual subtlety the report missed:** `weighted()` renormalises over the spouts that have at least `MIN` licks. So the
  "constant per animal x spout" reference shift is **not** constant per session when the set of spouts present changes. This
  is small, and it applies whatever the baseline is.
- The ad-hoc `EPS` / `POS` lists (`:41-42`) are confirmed; they match `POS_ORDER`.

## Claim 4: "responded" is defined three ways. **CONFIRMED** (and the three differ by more than the report says)

| | lick source | window | response window |
|---|---|---|---|
| `rest_engagement._responded_per_cue` (`:154-176`) | canonical `behavior_events` npz (config 2.5/0.5 V, 40 ms floor) | `[cue, cue+w]`, searchsorted, inclusive | `_response_window_s`: in practice **3.5 s**, see (g) |
| `rest_by_position._engaged_trials` (`:142-189`) | re-detects from raw `lick_analog` with `detect_licks` and the config params | `[c, c+w)`, Python loop | `quiet_periods.response_window_s(session.get("session_dir"))`; `session_dir` is never set, so the default `decode.max_rt_s` = **3.5 s** |
| `nolick_decoder.categorize` (`:172-262`) | `_load_daq_events(..., 2.5, 1.0, ..., 0.10)` (1.0 V offset, 100 ms bout refractory), at **frame** resolution; LP train under `LICK_SOURCE=lp` | first lick after the cue frame | `responded = engaged OR late_rewarded`, so it runs up to `response_window_for(s)` (gui_config, 3.5 s) |

- All three feed `engagement_gate` with the same reference (`close_L` / `close_center`, `precue_engagement_states.py:63`).
- The two rest gates should agree closely (same detector parameters, 3.5 s). `categorize` can differ through its 1.0 V /
  100 ms / frame-binning detector.
- **IMPACT:** possibly small differences in the quit boundary between the rest baselines and the imaging no-lick arm for
  cohort 1. That is unmeasured. Not a demonstrated published-number change.

## Claim 5: there is no single map of where shared quantities are defined. **CONFIRMED**

- `POS_ORDER`, `lp_licks`, `gate_source`, `sig_marks`, `cat_points`, `mixed_models` and `jaw_pass_qc` appear in none of
  `docs/ARCHITECTURE.md`, `README.md`, `wfield_local/README.md`, `scripts/README.md` or `CLAUDE.md`, on either tree.
  `orofacial_cohort` appears only in README.md.
- No cohort-1 number is affected. This is a clarity issue only.

---

## Coordinator add-ons

### (a) Two trial classifiers. **CONFIRMED that they differ. They do not currently feed the same figure.**

- `trial_attempts.py:114` (same line on both trees) uses these inputs:
  - jaw from `jaw_onsets.csv`: `cam4_moved` OR `cam1_moved`;
  - `close_mid` / close_center masked (`:45`);
  - a jaw-at-cue / re-onset rule;
  - DAQ `contact`, and `n_licks`.
  - Classes: hit / incomplete / jaw_only / no_try / unknown.
- `jaw_only_incomplete_figs.classify:64-88` (**main only**, new on 11-02/03) uses these inputs:
  - cam4 v36 `licks.csv` contact, plus cam1-only events with droplets removed;
  - the cam4 `jaw_pass_qc` from `tongue_jaw_mismatch.csv`;
  - no close_center mask and no jaw-at-cue rule.
  - Classes: contact / incomplete / jaw_only / no_response / unknown.
- **Consumers.** `trial_attempts` feeds `lp_licks.ATTEMPT_CLASSES` (hence the LP gate), `mixed_models` behaviour indices
  (ttc / attempt rate), `frozen_encoder`, `attempt_coding` and others (22 importers). `jaw_only_incomplete_figs` feeds only
  itself and `trial_class_raw_share_fig.py`, i.e. kinematics page Fig 6.1.
  - Its `GATES` ("working" for the imaging page, `:91-104`) is **not yet imported** by any imaging code (grep across both trees).
- **IMPACT:** "jaw_only" and "incomplete" shares on the kinematics page will not equal `trial_attempts`' `frac_jaw_only` /
  `incomplete` for the same cohort-1 session. That is a reader-facing inconsistency, not a wrong number. It becomes material
  once `GATES` is adopted for the imaging page, as the comment at `:94-96` plans.

### (b) `jaw_pass_qc` read with `.astype(bool)`, so NaN becomes True. **CONFIRMED but small; the report's line list is partly wrong**

- Real sites:
  - `cam1_incomplete_licks.py:56` (both trees);
  - `orofacial_cohort.py:501` on main (and attempt :501), which sets `jq`. `:526`/`:546` use `jq` (the `jaw_no_lick_frac`
    session and per-position values).
- `cam1_licks.py:110/179` filter `candidate_no_lick_with_jaw_move.astype(bool)`, **not** `jaw_pass_qc`.
- **Data.** I read 8 cohort-1 `tongue_jaw_mismatch.csv` files (pre, acute and later; all four animals):
  - `jaw_pass_qc` NaN in **18 of 4,130 trials (0.44%)**, in 4 of 8 sessions. Example: PS93_0818 has True 465 / False 84 /
    NaN 5, and `astype(bool)` gives 470.
  - The column is read as object dtype holding Python bools plus float NaN. These are not strings, so "False" does not
    become True.
  - `candidate_no_lick_with_jaw_move` is NaN on 2 trials, and both become True.
- **The bigger semantic issue.** `jaw_pass_qc` is the jaw-MOVED call, not a QC gate (`jaw_kinematics.py:28,186`). So
  `jaw_no_lick_frac` = (jaw moved, no lick) / (jaw moved), not / all trials. `jaw_only_incomplete_figs` fixed exactly this on
  11-03 (`07385b37`), but `orofacial_cohort` and `cam1_incomplete_licks` still use it as a gate.
- **IMPACT:** the NaN part changes `jaw_no_lick_frac` by at most about 1 trial per session. The denominator part defines
  what the published `jaw_no_lick_frac` means for cohort 1 and should be checked against how the page words it.

### (c) Epoch bootstraps weight by session, not animal. **CONFIRMED**

- `epoch_figures.scalar_contrast_draws:2797-2834`:
  - `point` = flat mean over **all** session rows per epoch (session-weighted). It includes animals that are missing from the
    other epoch.
  - The draws pool resampled sessions over the animals present in **both** epochs (session-weighted mean).
- `contrast_draws:1630-1676`: `point` = `stat` on the pooled trials of all animals (trial-weighted). The draws skip animals
  missing an arm.
- `analysis_kit.boot_delta:131-175`, by contrast, gives an animal-weighted mean of per-animal differences over animals in both.
- `analysis_kit.py:21-25` documents why the matrix reducers are kept separate. It does not document that they weight
  differently.
- **IMPACT:** session weighting applies to cohort-1 epoch-grant figures (PS95 had about 7 of 15 pre sessions in one table,
  per `boot_delta`'s own docstring). The animal-mismatch part is latent while all 4 animals have every epoch. I did not check
  `matrix_bootstrap.py` or `epoch_behaviour` in detail.

### (d) DAQ lick detection: 1.0 V + 100 ms vs 0.5 V + 40 ms. **PARTLY: real, but already documented, and the count is off**

- `_load_daq_events(` is called at **38** sites on main (multiline-aware `rg -U`). **28** of them pass the literal
  `2.5, 1.0, (0.001, 0.020), 0.10` (attempt: 29). The report's figure of about 48 is too high.
- The config (`defaults.yaml:158-162`) is 2.5 / 0.5 V with a 40 ms floor.
- **Headline paths:**
  - the imaging decode / `_trial_features` (`locanmf_position_decoder.py:453`) and `nolick_decoder.categorize` use 1.0 V /
    0.10 s;
  - behaviour (`daq_trials`, `behavior_events`), the rest gates and DLC use the config.
- Already recorded in DECISIONS.md:19184-19194 and `docs/DLC_IN_WIDEFIELD_ANALYSES.md:21-23` ("imaging hits / RTs can already
  differ from `*_trials.csv`"), and deliberately left. **IMPACT:** a known, documented divergence, not a new finding.

### (e) Response window 2.0 s vs 3.5 s. **PARTLY: most 2.0 s sites are not the response window, but one family is**

- **Where it is recorded:** `Behavior_logs/Widefield/<ANIMAL>_<YYYYMMDD>_<HHMMSS>/gui_config.json`, field
  `timing.response_window`, in **milliseconds as a string**.
- **Values.** I read all **258** `gui_config.json` files. **214 = "3500"** and **44 = "3000"**. All 44 of the 3000 ms files are
  May-2026 training sessions (PS92-95, 0505-0521), before any registered imaging session.
  - Spot checks across epochs: PS92 0601/0805, PS93 0805/0901/1002, PS94 0810, PS95 0820/1001 are all 3500.
  - Cohort 2: PS118 1019/1102 and PS121 1019/1102 are 3500.
  - So **3.5 s holds in every June-onward session, including cohort 2**. Priya is right for the analysed data. The "every
    session" claim in code comments is false only for May training.
- **Readers that use the data:**
  - `daq_trials.response_window_s(session_dir, default)` (`daq_trials.py:58-81`) is the correct single resolver: it divides ms
    by 1000 and accepts `timing` or `cue`. Used by spout_behavior / daq_trials scoring.
  - `nolick_decoder.response_window_for` (`:112+`) calls it via the `behavior_trials` parent or the date from the DAQ path.
- **Readers that look like data readers but never are:**
  - `quiet_periods.response_window_s(session.get("session_dir"))`: session dicts have no `session_dir` key (`load_sessions`
    keys: fmdir, h5, label, mc, regime), so it always returns `decode.max_rt_s` = 3.5 ("default").
  - `rest_engagement._response_window_s`: see (g).
- **Hard-coded 3.5:** `decode.max_rt_s` (`defaults.yaml:132`), `trial_attempts.RESP_MS = 3500` (`:50`).
- **The 2.0 s sites and what each one is:**
  - `defaults.yaml:194` `behavior.licking.response_window_s: 2.0`: the lick count / rate window, and the hit/miss fallback
    only. The fallback never fires, since every folder has a gui_config.
  - **`analysis_kit.RESP_S = 2.0` (`:56`), `session_behavior(resp_s=RESP_S)` (`:360`), `quit_point.session_quit(resp_s=2.0)`
    (`:103`), `engagement_decomposition.DEFAULT_RESP_S = 2.0` (`:67`), `channel_position_maps._quit_trials(resp_s=2.0)`
    (`:138`):** these ARE used as the response window, i.e. "responded" for the S28 quit point and the engagement
    decomposition. This is the real 2.0-vs-3.5 inconsistency. A lick at 2-3.5 s counts as a miss for the quit rule but as a hit
    in behaviour and the gates. `gate_source.py` calls it "`resp_s` (2.0 s) for the S28 quit rows". The comment at
    `analysis_kit.py:54-55` only says it matches rest_migration; there is no decision saying 2.0 is intended.
    **IMPACT:** a cohort-1 published quantity (the S28 quit point / licks-at-quit) is defined on 2.0 s.
  - `nvc_evoked.LICK_RESP_S = 2.0` (`:79`): an analysis cut-off ("lick this soon after the cue = lick trial"), not the window.
  - `nolick_decoder.py:109, :526` `max_rt=2.0` and `poststroke_compare.py:721` `args.max_rt = 2.0`: the engaged / late arm
    boundary. `responded` = engaged | late_rewarded still extends to the real window. This is deliberate and documented at
    `poststroke_compare.py:715-720`.
- **Natural single home:** `daq_trials.response_window_s` (already correct). `rest_engagement._response_window_s` should call
  it rather than be the home.

### (f) Five hand-written BH implementations, one counting NaN p in m. **CONFIRMED (5); the NaN point is moot at its call site**

- The five: `analysis_kit.bh:205`, `lick_templates.bh:227`, `orofacial_stats.bh_fdr:26`, `iti_decoder_pattern_stats._bh:99`,
  `within_spout_deviation._bh:524`. No `statsmodels` / `scipy` BH is used anywhere.
- Four of them drop non-finite p before ranking. `iti_decoder_pattern_stats._bh` uses `n = len(p)`, so NaN would count in m
  (conservative).
- Its only caller (`:139-140`) builds `p = min(1, 2*min(mean(B<=0), mean(B>=0)) + 1/N_BOOT)`, which can never be NaN.
- The opposite hazard: an all-NaN bootstrap column would give p = 1/N_BOOT, i.e. **significant**. Unverified whether any
  column of `B` is NaN.
- Since 10-27 that page leads with the FWER max-statistic (`sig_max`); BH `frac_sig` is still tabulated. **IMPACT:** none shown
  for cohort 1.

### (g) `rest_engagement._response_window_s` ms-vs-s bug. **CONFIRMED latent; it currently returns 3.5 s by accident**

- `:178-211`: if a `gui_config.json` is found under `session["behavior_dir"]` or `session["mc"]`, it returns
  `float(t["response_window"])`, which is **3500 (ms) treated as seconds**. That would make every cue "responded" and switch
  the gate off.
- It is never reached: session dicts have no `behavior_dir`, and none of the `mc` folders I checked (PS92_0606, PS92_0811,
  PS93_0818, PS93_1006) holds a `gui_config.json`.
- The fallback `config.defaults()["behavior"]["response_window_s"]` raises a KeyError, because the key lives at
  `behavior.licking.response_window_s`. So the `except` returns the hard-coded **3.5**.
- I measured this: `_response_window_s` returns 3.5 for PS92_0606, PS93_0818 and PS95_0812.
- **IMPACT:** none today. But two latent bugs cancel. Fixing the KeyError alone would give **2.0 s**, and registering a
  `behavior_dir` would give 3500 s. Route it through `daq_trials.response_window_s`.

---

## Issues the report missed (not a full audit)

1. **Attempt's `config.py` lacks `date8` and the cohort2 / year-aware logic** (124 lines behind main). The attempt branch
   cannot run cohort 2 correctly at all, independent of the 100 `"2026"+` sites.
2. **Main's LP arm is itself mixed** (only under `WIDEFIELD_LICK_SOURCE=lp`, so latent for the default numbers). Imaging gates
   and `categorize` use LP licks only, while `rest_engagement` / `rest_by_position` stay DAQ-only. This is documented only in
   attempt's `gate_source.py` docstring.
3. **`quiet_periods.response_window_s` is never fed a `session_dir`**, so the rest-baseline trial windows use the config's
   `max_rt_s` (3.5) rather than the data. The result is correct today only because every session is 3500 ms.
4. **May-2026 training sessions ran 3000 ms.** Code comments say "every session ran 3500 ms". This is harmless for imaging
   (no May session is registered), but wrong if any behaviour learning-curve figure scores May with a 3.5 s fallback. I did
   not check whether it does.
5. `orofacial_cohort` / `cam1_incomplete_licks` still use `jaw_pass_qc` as a filter. That is the same denominator bug fixed in
   `jaw_only_incomplete_figs` on 11-03 (see (b)).
