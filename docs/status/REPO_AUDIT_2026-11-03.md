# Redundancy and clarity audit: widefield_pipeline (main) and wf_attempt (branch analysis/attempt-coding)

**Date:** 2026-11-03.

**Method: read-only.** I edited no repo files and ran no analyses. The only commands used were grep, `git diff` / `git log`, and directory listings on M:.

**Line numbers** are from the trees as read on 11-03. On main, `orofacial_cohort.py` and `epoch_figures.py` had uncommitted edits in progress, so their line numbers can drift by 20-80 lines. Search by function name instead.

**Sources.** Four read-only sub-audits ran in parallel, plus my own pass. The sub-audits covered:
- epochs, curation and the engagement gate;
- spout order, signs, colours and stars;
- licks, trial classes, jaw, 3-D pose and FaceRhythm loaders;
- bootstrap, corrections, LMMs and paths.

**Confidence tags:**
- **[verified]**: I re-read the code myself.
- Untagged: taken from a sub-audit's reads and greps.
- **UNVERIFIED**: nobody read it.

**Repos:** MAIN = `C:/Users/SabatiniLab/Github/widefield_pipeline`, ATT = `C:/Users/SabatiniLab/Github/wf_attempt`.

---

## 0. Summary by risk

| # | concept | problem | could results differ? |
|---|---|---|---|
| 1 | two copies of the shared package | ATT's `wfield_local/` lags main (no cohort-year support, no `cat_points`) and adds `gate_source` hooks; 60 shared files differ. Worktree scripts can silently import main's package. | **Yes.** The engagement gate differs by checkout and env var. Cohort2 breaks on ATT. |
| 2 | trial classes | Two classifiers with different contact source, lick window, jaw cameras, masks and gate (`trial_attempts.py` vs `jaw_only_incomplete_figs.classify`). | **Yes.** Class shares differ. |
| 3 | `jaw_pass_qc` | Three places still use it as a gate and convert NaN to True via `.astype(bool)`. | **Yes.** The jaw-only fraction has the wrong denominator, and unknown-jaw trials count as jaw-only. |
| 4 | change-vs-pre bootstrap weighting | `epoch_figures.scalar_contrast_draws` / `contrast_draws`, `matrix_bootstrap.delta_draws` and `epoch_behaviour` weight by session or trial. `analysis_kit.boot_delta` (rule 9) weights by animal. | **Yes.** Point estimates and CIs on the epoch families (8g/10/10b/11, accuracy, behaviour) differ from `boot_delta` whenever session counts differ between animals. |
| 5 | DAQ lick identity | Imaging path uses `thresh_lower 1.0, refractory 0.10` (48 copies). Config and `behavior_events` use `0.5` with no refractory. CLI modules use `0.5 + 0.10`. | **Yes, small.** Onset sets differ, so hits, RTs and lick-aligned trials differ at the margins. |
| 6 | response window and "hit" | Three values: 2.0 s (`analysis_kit.RESP_S`, the S28 quit family, the imaging "engaged" cut in `decoder_weights` / `position_encoder`), 3.5 s (`decode.max_rt_s`, the task window), and the trial-stop TTL (`trial_windows`). | **Yes.** "Hit" means different things in different families. |
| 7 | multiple comparisons | Five hand-written BH implementations, one of which counts NaN p-values in m. Bonferroni m is chosen ad hoc (x3, x8, x12, x18, x36, per figure). FDR families differ per `lmm_table` caller. | **Yes.** Significance calls are not comparable across figures. |
| 8 | mixed models | About 17 ad-hoc `smf.mixedlm` calls fit by ML on post-only `log(day)`. Canonical `fit_lmm` uses REML, a random-effects ladder, and `log(clip(day,1))` with pre sessions included. | **Yes.** Recovery-slope p-values come from different models. |
| 9 | sign conventions | Three meanings of "image right". `to_contra` would double-flip for a right-lesioned animal. About 25 `pos_key(p, "L")` literals. One script infers the sign from the data. | Not today (all 8 animals are left-lesioned). **Latent** for any R-lesion animal. |
| 10 | August-only pre baseline | Cut-off is "0801" in about 45 sites and "0806" in about 25-38, alongside `config.pre_from()` and `oc.LATE_PRE_FROM`. | **Not today.** No 0805 or June files on disk in the globbed folders (§3). Breaks for cohort2 because of MMDD string compares. |
| 11 | spout order, colours, markers, stars | At least 7 spout orders, 4 position palettes, about 12 epoch palettes, 5+ animal palettes (some reuse another animal's colour), and 6+ meanings of `*` / `**`. | Numbers unaffected, but figures can be misread (e.g. PS94 drawn in PS93's red). |
| 12 | paths | The v36 run root is hard-coded in about 20-23 files. Two hard-coded `N:` paths survive the M/N swap; one is a write destination. | `publish_cache` would write to standby. |
| 13 | clarity | No "where is X defined" map. ARCHITECTURE.md omits every behavioural and orofacial definition. Several misleading names. | n/a |

---

## 1. Two copies of the shared package (highest structural risk)

**[verified]** The two branches diverged at merge-base `2c6fcd89`. Since then:
- main has touched 295 files under `scripts/` and `wfield_local/`; ATT has touched 218.
- **84 files were touched on both sides, and 60 of them now differ.** They include `wfield_local/{frozen_encoder, lp_licks, nolick_decoder, orofacial_cohort, rest_by_position, rest_engagement, sig_marks, stray_licks}.py` and 44 `scripts/iti_*.py`. Example: `iti_area_maps.py` is 248 lines on main and 572 on ATT.
- The branches are synced by hand-copying. ATT's `sig_marks.py` says it was "copied 11-01" from main, yet the files still differ.

| concept | MAIN | ATT | effect |
|---|---|---|---|
| engagement-gate "responded" | DAQ contact only | `gate_source.gate_responded` hooks in `precue_engagement_states:181`, `rest_engagement:68-71`, `rest_by_position:183-185`, `nolick_decoder:257-258` and `quit_point:119`. With `WIDEFIELD_GATE_SOURCE=lp`, "responded" = DAQ contact OR LP attempt. `gate_source.py` exists only on ATT. | Same function name, different gated trial sets |
| cohort year | `config.date8` / `date_key` (handles New-Year rollover); `epochs._date` uses `date8` | **No `date8` at all.** `epochs._date = dt.date(2026, …)` (ATT `epochs.py:416`). `boundaries_path` → `root("labcams")` instead of `root("cohort_out")` (`:315`). `"2026"+mmdd` in 19 `wfield_local` modules (e.g. `rest_engagement:165`, `rest_by_position:135`, `orofacial_cohort:615`, `lp_licks._inputs`, `stray_licks:124`). | Cohort2 dates after 12-31 break, and the epoch-boundary file lives in a different place |
| animal id | `split("_")[0]` / `config.animal_of` | `label[:4]` in `nolick_decoder:131,505,685`, `locanmf_decoder_weights`, `locanmf_position_encoder` and `analysis_kit.curated_ids:304` | PS118 becomes "PS11" |
| jaw-angle sign | `orofacial_cohort.to_contra`, `fig_angle_tuning(contra=)`, `kinematics_pooled_figs` default "+ = contralesional" | none of these; "+ = ipsi" | Same data, opposite printed sign |
| figure helpers | `cat_points.py`, `camera_view.py` | absent | — |

**Silent wrong-copy imports.** MAIN `scripts/README.md:30-45` warns that `python scripts/x.py` run from a worktree imports **main's** `wfield_local` through the editable install. **[verified]** 150 of the 391 ATT scripts import `wfield_local` without pinning `sys.path` to their own checkout. They include newer scripts such as `decoder_retention_sessions.py`, `recovery_timing.py`, `revup_behaviour.py`, `precue_class_matched.py`, `frozen_musall_figures.py` and `pose3d_kinematics.py`. `iti_recovery_measure.py:36` and `lick_cued_aim_by_spout.py:31` insert `scripts/` rather than the repo root; which package that resolves to is UNVERIFIED.

**Fix:** merge or rebase `analysis/attempt-coding` onto main so `wfield_local/` exists once. Until then, stamp each figure with `wfield_local.__file__` and the commit (via `figure_meta`).

---

## 2. Trial classes and `jaw_pass_qc` (the example that started this audit)

### 2a. Two classifiers

**Canonical producer: `scripts/trial_attempts.py:114-115`.** It writes `<v36>/imaging_encoding/qc/trial_attempts.csv`:
`cls = np.select([contact.notna(), n_licks>0, ok&mv, ok], ["hit","incomplete","jaw_only","no_try"], "unknown")`.
- **hit** = DAQ contact from cue+70 ms to stop.
- **incomplete** = a cam4 `licks.csv` lick with `on_ms` in 0..stop (:77), or a cam1-only event (`~cam4_matched & ~in_3d & ~droplet`, :87).
- **jaw** = `jaw_onsets.csv`, cam4 OR cam1 (cam1 floor 15 px), masked at close_mid (`JAW_UNRELIABLE`, :45). A jaw already deflected at the cue counts only through its re-onset.
- **gate** = `engagement_keep(gate="reference")`, forced (:92).
- There is no `wfield_local/trial_attempts.py`; the producer is the script itself.

Consumers that read this table consistently:
- `lp_licks.py:49,331`
- `attempt_coding.classify:76-92`
- `attempt_maps.classes_for_rows:63-72`
- ATT `gate_source.extra:121-166`
- `cohort2_qc/attempt_effort.py:33`, `quiet_task_events.py:400-406`
- `relabel_fc_by_class.py:85`, `summary_figures_1021.py:24`, `whisker_motion.py:156`
- ATT `stopped_trial_traces.py:181,263`

One small difference among them: `attempt_coding` labels hits in the no-lick rows "late", while `attempt_maps` drops them.

**Second classifier: MAIN `scripts/jaw_only_incomplete_figs.py:64-88` `classify`.** It feeds `trial_classes/sessions.csv` and `trial_class_raw_share_fig.py`. It differs from `trial_attempts` in every input:

| | trial_attempts | jaw_only_incomplete_figs |
|---|---|---|
| contact | raw DAQ contact, cue+70 ms to stop | `licks.csv` `contact` (video lick matched within ±8 ms) or cam1 `contact` (±20 ms). A DAQ contact with no video lick becomes jaw_only / no_response. |
| lick window | 0..stop | **none** |
| jaw | `jaw_onsets`, cam4 OR cam1, close_mid masked, at-cue rule | `tongue_jaw_mismatch.jaw_pass_qc`, cam4 only (4 px), **no mask**, no at-cue rule |
| extra rule | — | a cam1 tongue on a mismatch candidate counts as incomplete (:82) |
| gate | `gate="reference"` | `oc.engagement_keep(animal, date, sd)` with no `gate=` (:97), so it follows env `OROFACIAL_GATE`. Under `attempted`, the jaw_only and no_response shares come out empty. |
| class names | hit / no_try | contact / no_response |

**Results differ.**

**Fix:**
- Make `trial_attempts.csv` the only class table. `jaw_only_incomplete_figs` should read `cls` from it, with extra columns added there if the figure needs them.
- Move the classifier body into a `wfield_local` module (proposed: `wfield_local/trial_classes.py`, or inside `attempt_coding`) so scripts import a function rather than depend on another script.
- The deck-script copy mentioned in the brief was not found. `trial_class_all_engaged` appears only in a docstring (UNVERIFIED; possibly a scratch script).

### 2b. `jaw_pass_qc` means "jaw moved", but some readers use it as a gate

Definition (`jaw_kinematics.py:177-186`): True when |y − mean(−500..0)| > max(4·sd, 4 px) on at least 5 of 8 frames in 0..stop; NaN when the baseline is degenerate.

| reader | reads it as | problem |
|---|---|---|
| `scripts/cam1_incomplete_licks.py:56-59` **[verified]** | a gate: `q = M[... & M.jaw_pass_qc.astype(bool)]`, then `jaw_only_cam4 = cand.mean()` over q | The denominator is jaw-moved trials. **`.astype(bool)` turns NaN into True**, so unknown-jaw trials enter. `candidate…astype(bool)` also turns NaN into True. |
| `wfield_local/orofacial_cohort.py:501` (also :526, :546) **[verified :501]** | the same gate | `jaw_no_lick_frac` has the jaw-moved denominator, but the label at :83 says "frac trials" |
| `scripts/cam1_licks.py:110,179` | `candidate_no_lick_with_jaw_move.astype(bool)` | NaN becomes True; feeds the "seen" rule in 2a |
| `kinematics_checks.py:122` | a QC pass rate | NaN counted in n |
| ATT `coupling_vigour_sm.py:122` | a trial selection | harmless |
| correct readers (moved call, NaN-aware) | `tongue_jaw_mismatch:188-190,255`, `jaw_only_incomplete_figs:77`, `session_lick_summary:84`, `pose_kinematics_demo:209,221`, `jaw_onsets:172,226,240`, `nose_onsets:47,53` | — |

**Fix:**
- Add a helper in `jaw_kinematics` that returns a nullable boolean (`pd.array(..., "boolean")`), and stop calling `.astype(bool)` on this column.
- Rename the column `jaw_moved`, keeping `jaw_pass_qc` as an alias, so the name stops suggesting QC.
- Fix two stale comments: `orofacial_cohort.py:~1580` says "attempted" includes jaw_only (it does not), and `jaw_onsets.py:59` cites `trial_attempts.CAM1_JAW_CUT`, which does not exist.

---

## 3. Epochs, baseline and curation

**Canonical definitions:**
- `configs/animals.yaml`: stroke dates (PS92/93 0817, also excluded for them; PS94/95 0816) and per-animal epoch bounds.
- `epochs.EPOCHS` (:97), `epochs.epoch_of` (:434), `days_since_stroke` (:421).
- `epoch_boundaries.json` overrides the YAML at run time (contents UNVERIFIED).
- `config.session_phase` (MAIN :423) applies `pre_from()`.

### August-only pre: one rule, five spellings
- `config.pre_from()`
- `orofacial_cohort.LATE_PRE_FROM = "0801"`
- literal `>= "0801"`: about 40 MAIN / 44 ATT files, e.g. `frozen_encoder.py:1641,1736`, the `frozen_*_maps` scripts, `fr_direction:61`, `kinematics_pooled_figs`, `iti_licks:1397`
- literal `>= "0806"`: about 38 MAIN / 18 ATT files, e.g. **`mixed_models.py:63,78`** (the LMM pre filter and the `behaviour_deficit` recovery axis), `allsession_encoder:156`, `frozen_relabel:186`, `trial_attempts.py:163,194`, `encoding_mixed:109`, most `cohort2_qc/quiet_*`
- `"20260806"` / `"20260801"` (`prep_3d_inputs:28`, `pose3d_*`) and `>= "0800"` (`spout_behavior:1567`)

**Checked on disk (11-03).** I listed the encoder session folders that the "0801" readers glob:
- `frozen_encoder_gate-attempted_{v7g3_t3d, v7g3_t3d_frA, v7basis}/joint/sessions` (214 files each)
- `movement_first_fr-master_pre_v1_…combined{,_frg-A}/sessions` (321 files each)

None contains a 0805 or June file, and August pre is 0806/0807/0809 for all four animals in every folder. **So for these readers, 0801 and 0806 give the same result today.**
- A second listing covered **every** `frozen_encoder_*/*/sessions` and `movement_first_fr-master_pre_v1*/sessions` folder under `imaging_encoding` and found **no `_0805` file in any of them**.
- Not checked (UNVERIFIED): `qc_running_survey:12`, `qc_running_lick_jitter:44`, `state_spout_check:87`.
- Remaining risk 1: 0805 is excluded only by `cross_session_exclude`, which `epoch_of` does not apply, so `epoch_of("PSxx_0805") == "pre"`.
- Remaining risk 2: every MMDD string compare is wrong for cohort2, whose dates run from 1029 into the next year.

### Other epoch and curation findings
- **Two baselines inside one measure [verified].** ATT `scripts/iti_recovery_measure.py` takes its offset reference from **all** pre sessions (:16, :61). Its deficit `aim_deficit` (:122) uses the **August** pre, via `mixed_models.behaviour_deficit`. Change vs pre is unaffected, but `weighted_abs` and the "pre noise floor" are not.
- **Day-0 edge case.** `epoch_figures.epoch_of_day` (:550) and `grant_behaviour:122` treat pre as `day < 0`; `animal_weights._epoch_for` uses `day <= 0`. `epoch_of` calls day 0 "pre", but `epoch_of_day(0)` returns None. Latent: the only day-0 sessions are excluded.
- **Hard-coded pre/post splits.** `cohort2_qc/quiet_modes.py:116` (`< "0816"`) and `slowband_decode_check.py:99` (`<= "0814"`) ignore per-animal stroke dates. An archived default `--from-mmdd "0813"` is wrong.
- **Re-declared constants.**
  - `("pre","acute","subacute","chronic")` is re-typed in about 25 places per tree (e.g. `mixed_models:52`, `orofacial_cohort:68`, `animal_weights:45`, `iti_recovery_measure:41`).
  - Session-list snapshots: `cohort1_subset_headline.py:26-29`, `lick_floor_qc.py:23-24`.
  - Chronic-rule fallbacks in `epochs.py` (`k_res 1.5`, `flat_mode "rate"`) disagree with the YAML (`0.4`, `drift`). They are inactive while the YAML keys exist.
- **Curation.**
  - `set(phase_labels("pre")+phase_labels("post"))` is still spelled out in 44 files per tree. It skips `curated_sessions`' filter that drops sessions whose `epoch_of` is None.
  - `("PS92","PS93","PS94","PS95")` is hard-coded in 88 MAIN / 126 ATT files.
  - **MAIN `analysis_kit.curated_ids:304` uses `lab[5:]`**, which raises on PS118 dates. ATT uses `lab[:4]`.
  - **MAIN `mixed_models.py:59` uses `str[:4]`**, which gives "PS11" for cohort2.

### Single home
`epochs` should own `EPOCHS`, `epoch_of` and `epoch_of_day` (delete the `epoch_figures` copy), plus a new `baseline_labels(mode="august"|"all")`. That function should use `config.date_key` and one constant, `defaults.yaml baseline.august_from`. `mixed_models`, `frozen_encoder`, `orofacial_cohort` and every script with a literal cut would import them.

---

## 4. Engagement gate, "responded", response window and "hit"

**Canonical rule:** `precue_engagement_states.engagement_gate` (MAIN :123). Reference spouts = close_L and close_center, window 15 trials, minimum rate 0.5, terminal collapse only. It returns True for NOT engaged.

Every caller uses that rule, but **each feeds it a different "responded" vector**:

| caller | responded = | lick source | window |
|---|---|---|---|
| `rest_engagement._responded_per_cue` **[verified]** | any lick in `[cue, cue+w]` (searchsorted, inclusive) | `behavior_events` (config 2.5 / 0.5 / lockout / min_ili 40 ms, no refractory) | `_response_window_s`, which ends up at 3.5 (see the bug below) |
| `rest_by_position._engaged_trials` **[verified]** | any lick in `[c, c+w)` (half-open) | re-detected from raw `lick_analog` with the same config parameters; `fs` defaults to 5000 rather than the h5 rate | `quiet_periods.response_window_s(session.get("session_dir"))`. `session_dir` is never set, so this is `decode.max_rt_s` = 3.5. |
| `nolick_decoder.categorize` **[verified]** | category in {engaged, late_rewarded}: first lick after the cue with 0 < rt ≤ response window | `_load_daq_events(..., 2.5, 1.0, (0.001,0.020), 0.10)`: thresh_lower 1.0 and a **100 ms refractory**, in **imaging-frame units** | session window 3.5 |
| `precue_engagement_states._session_tables` and `beta_maps._quit_mask`, used by `position_coding_directions`, `attempt_coding`, `grant_kit`, `position_axes`, `attempt_maps` and others | trial in the ENGAGED feature arm (first DAQ lick ≤ `decode.max_rt_s`) | decoder lick path | 3.5 |
| `spout_behavior.reference_engagement` | `engagement_gate` plus a terminal tail of 6 reference misses | DAQ trial table `lick_in_response_window` | `[cue, min(cue+rw, next_cue)]` |
| `analysis_kit.session_behavior`, `quit_point`, `engagement_decomposition`, `channel_position_maps._quit_trials` | hit = DAQ lick in `[cue, min(cue+resp_s, next)]` | — | **2.0 s** (`analysis_kit.RESP_S = 2.0` **[verified]**) |

### Do the three "responded" definitions pick the same trials?
Answered from the code logic only; nothing was run.

**`rest_engagement` vs `rest_by_position`: the same trials in practice.** Same detector, same parameters, same 3.5 s window. They can differ in only two cases:
- a lick landing exactly on sample `cue+w` (inclusive vs half-open window);
- a session whose h5 sample rate is not 5000 Hz, because `rest_by_position` hard-codes `fs=5000` when converting the window.

Neither truncates the window at the next cue.

**`nolick_decoder` vs the other two: nearly the same trials, but not by construction.** Four differences:
- **(a) Different lick train.** thresh_lower is 1.0 and there is a 100 ms refractory. A lick within 100 ms of a preceding lick is dropped, so a cue preceded by an ITI lick can lose its first post-cue lick. The next lick (~150 ms later) usually still falls inside the window.
- **(b) Frame units.** RT is computed in imaging frames, so the window edge is quantised to a frame. A lick in the same frame as the cue (rt = 0) is classed "undetected".
- **(c) Position codes.** Cues with no usable position code get `cat = ""` and count as not responded.
- **(d) Env switches.** Under `WIDEFIELD_LICK_SOURCE=lp` it uses the LP lick train. On ATT under `WIDEFIELD_GATE_SOURCE=lp` it also counts LP attempts.

Expect agreement on the large majority of trials, with differences at the edges. Whether those differences ever move the gate's terminal-collapse point is UNVERIFIED; that needs a run.

### Latent units bug [verified]
`rest_engagement._response_window_s` returns `float(t["response_window"])` from `gui_config.json` **without dividing by 1000**. The GUI stores milliseconds (`daq_trials.response_window_s` does divide by 1000). If the file were found, the window would be 3500 s, every cue would count as "responded", and the gate would never trip.

Its config fallback also misfires: it reads `defaults()["behavior"]["response_window_s"]`, which raises KeyError because the key actually lives at `behavior.licking.response_window_s` (= 2.0). It reaches 3.5 only through the `except` branch.

**Harmless today.** I listed the `mc` dirs of PS92-95 on 0812, 0820 and 0911 and found no `gui_config.json`, and session dicts have no `behavior_dir`.

### Other window findings
- **2 s vs 3.5 s "engaged".** `locanmf_decoder_weights.py:287,345,450,491` and `locanmf_position_encoder.py:168,260` hard-code `rt <= 2*FS`, contradicting `decode.max_rt_s: 3.5` (`defaults.yaml:132`).
- `trial_windows.py:136-154` ends the window at the trial-stop TTL (3.53-5.37 s).
- `face_rhythm_validation.RESP_S = 4.0`.
- 3.5 is re-declared in `dlc_iti_frames:40`, `position_coding_trend_recovery:34`, `trial_attempts:50`, `rest_migration/settle:35`, `cohort2_qc/spout_maps:43`, `indicator_kinetics_spread:59` and ATT `gate_lp_check:54` (which also sets `QUIT_RESP = 2.0`).
- Dead config: `behavior.engagement.window_trials` / `min_response_rate` are never read; the module constants are used instead.

### Single home
- `rest_engagement` (or `precue_engagement_states`) should own one `responded_per_cue(label, cue_samples, window="task")`, built on `behavior_events` and `daq_trials.response_window_s`. `rest_by_position` and `nolick_decoder` (for its gate input) would call it.
- Name the two windows as two constants in one place: `TASK_WINDOW_S` (3.5, per session) and `QUIT_HIT_WINDOW_S` (2.0, the S28 family). Then a reader can see which one a figure used.

---

## 5. Licks

### Two (really three) DAQ lick identities [verified config values]
- **Config / behaviour path** (`behavior_events`, `daq_trials`, `spout_behavior`, `rest_by_position`): `thresh_upper 2.5, thresh_lower 0.5, lockout [0.001, 0.020], min_ili 40 ms`, no refractory (`defaults.yaml:158-162`).
- **Imaging path:** `_load_daq_events(h5, "lick_analog", 2.5, 1.0, (0.001,0.020), 0.10)`, i.e. **thresh_lower 1.0 and refractory 0.10 s**. It appears 48 times in 40 MAIN files and 37 times in 29 ATT files, with the same numbers in every copy. `analysis_kit.lick_samples` (:324) wraps it.
- **CLI modules** (`plot_lick_aligned_averages:226`, `locanmf_lick_aligned:244`, `roi_activity:117`, `framemap_event_maps:451`, `quiet_periods:377`): thresh_lower 0.5 with `--refractory-s 0.10`.

CLAUDE.md says behaviour and imaging share one lick identity; on these paths they do not.

Two related traps:
- Two different functions are both named `_load_daq_events`: one loads licks (`plot_lick_aligned_averages`), the other cues (`plot_spout_trial_averages`).
- Under `WIDEFIELD_LICK_SOURCE=lp`, the lick loader silently returns the LP train (`plot_lick_aligned_averages.py:62-73`).

### Video licks
- **"cam1-only" filter.** `trial_attempts:87`, `jaw_only_incomplete_figs:73`, `cam1_incomplete_licks:110` and `cam1_licks:203` use `~cam4_matched & ~in_3d & ~droplet`. **`lp_licks._read_times:253` and `cohort2_qc/lp_licks.py` omit `~in_3d`** and read `licks_all.csv` / `events_all`; the cohort2 version also adds a 40 ms dedup.
- **DAQ-contact match tolerance** is ±8 ms in `licks.csv` and `contact_classes`, but ±20 ms in `cam1_licks.CONTACT_TOL_MS`.

### Single home
- `behavior_events` for DAQ licks. Either make `analysis_kit.lick_samples` read the canonical identity, or declare the imaging identity a deliberate second one with its own name (`IMAGING_LICK_PARAMS` in `defaults.yaml`). Then sweep the 48 copies.
- `lp_licks` should own the cam1-only filter as a function.

---

## 6. Jaw, 3-D pose and FaceRhythm loaders

### Jaw
Canonical: `jaw_kinematics` (baseline mean/sd over −500..0, threshold max(4·sd, 4 px), 5 of 8 frames, window 0..stop). Other definitions:

| where | opening / baseline | thresholds | masks |
|---|---|---|---|
| `jaw_onsets.py:63-71,111-161` | `open_px` vs the pre-cue **median**; 3-D uses the 95th percentile | cam1 floor 15 px, likelihood < 0.6 → NaN; 3-D 0.10 mm; at least 20 pre frames and 50% coverage | close_center (2-D) and far_center (3-D) |
| `jaw_only_incomplete_figs.py:115-120` | 3-D `jaw_open` vs the per-trial pre median, −500..3000 | at least 5 frames, at least 2 trials | **no** centre blanking, **no** coverage gate |
| `orofacial_jaw_lateral.py:35-87` | **no per-trial baseline**; projected on the AP axis, 70..3000 | `OPEN_PX 50`, `OPEN_FIT_PX 15` | filled frames dropped |

About 30 other files touch raw jaw coordinates; their thresholds are UNVERIFIED.

### 3-D pose
`pose3d.py` has **no loader function**, so every consumer reads `session_poses/pose3d/<s>_frames.csv.gz` itself, with inconsistent hygiene:

| dedup | `failed_sessions` filter | files |
|---|---|---|
| yes | yes | `cam1_licks`, `jaw_only_incomplete_figs`, `pose3d_cue_traces`, `pose3d_kinematics`, `pose3d_close_dir_sheet` |
| yes | no | `jaw_onsets`, `lick_trajectories_3d`, `pose3d_phase`, `pose3d_spout_error`, `prep_3d_inputs`, `combined_encoding`, ATT `face_prelaunch_aim`, `within_lick_correction` |
| no | no | `lick_path_softmax:48`, `pose3d_check_sheet:44`, `pose3d_figures`, `pose_chain` |

`contact_direction_predictability:18` reads `kinematics/` where every other script reads `kinematics_all/`.

**Single home:** add `pose3d.load_frames(label, dedup=True, drop_failed=True)` and `pose3d.load_lick_kinematics()`.

### FaceRhythm
Canonical: `face_rhythm_outputs` (`BUNDLE`, `SET="master_pre_v1"`, `factor_dir`, `fr_signals`) and `fr_groups.GROUPINGS["master_v1"]`.
- Bypassed with a direct `np.load(fdir/"proj"/…)` in `face_rhythm_asym_aligned:70`, `bout_split:51`, `nontask_state:52`, `push_check:85`, `factor_clips:97` and `cohort2_qc/quiet_eta_split:95`.
- A third helper, `frc.sessions(...)`, is UNVERIFIED.
- Factor indices are hard-coded in ATT `disengagement_tests:59` and `face_prelaunch_aim:56-57,229`. They match today but would not follow a refreeze.
- The set name is hard-coded in about 10 output-folder names.

---

## 7. Statistics: bootstrap, corrections, LMMs, permutations

**Canonical:** `analysis_kit`, with N_BOOT 4000 and percentiles 2.5/97.5:
- `boot_ci`: flat pool, for levels.
- `boot_delta`: animal-weighted and paired, for changes.
- `boot_p` = max(2·min, 1/n).
- `bh` and `holm`.

### Changes computed session-weighted (the pattern behind the 09-20 retraction)
- **`epoch_figures.scalar_contrast_draws` [verified]:**
  - The point estimate is the flat mean over all sessions of epoch a minus epoch b, **including animals missing from one epoch**, while the draws use only the animals present in both.
  - Each draw concatenates session values, so it is session-weighted. n_boot = 2000.
  - It feeds `epoch_kit._scalar_figure`, i.e. every change-from-pre panel in families 8g/10/10b/11.
- `epoch_figures.contrast_draws` is trial-weighted; used by `epoch_accuracy` and `epoch_behaviour`.
- `matrix_bootstrap.delta_draws` is session-weighted.
- `epoch_behaviour._behaviour_contrast` says so explicitly: "session-weighted, a six-session animal weighs 6x".

These may be deliberate (the docstrings argue that a session is the unit), but they contradict CLAUDE.md rule 9. **Results differ** from `boot_delta` whenever animals have unequal session counts.

### Other bootstrap differences
- **Levels computed animal-weighted:** ATT `recovery_timing._boot_mean_of_animals:168` and `lick_cued_aim_by_spout.boot:59`, unlike the `boot_ci` flat pool.
- **n_boot** takes the values 200, 400, 500, 1000, 2000, 4000 and 5000 across about 40 sites. 2000 is the most common, e.g. `epoch_kit:85`, `matrix_bootstrap:35`, `orofacial_stats:52`.
- **CI construction:**
  - Bonferroni-widened percentiles inside the bootstrap: `lick_path_softmax:85` and `softmax_neural_vs_lick:98` (0.05/36), `lick_cued_aim_by_spout:85` (0.025/18).
  - SE + z: `beta_maps:1046`.
  - Medians: `session_lick_summary`, `jaw_lateral`.
- **Bootstrap p formula:** `2·min + 1/N` in `iti_decoder_pattern_stats:177` (and the files that import it); `2·min` with no floor in `coupling_by_class`; canonical is `max(2·min, 1/n)`.

### Multiple-comparison corrections
- **BH:** five hand-written versions and no statsmodels anywhere:
  - `analysis_kit.bh`
  - `orofacial_stats.bh_fdr`
  - `lick_templates.bh`
  - ATT `within_spout_deviation._bh`
  - **`iti_decoder_pattern_stats._bh`, which counts NaN p-values in m** (more conservative).
- **BH families** differ per `lmm_table` caller:
  - default `("window",)`
  - `("source",)` and `("kind",)` in other callers
  - `()` in ATT `iti_lmm_figures`, `iti_mixed_models` and `lick_cued_aim_by_spout`
  - `encoding_mixed:119` and `imaging_lmm:118` apply BH a second time.
- **Bonferroni m:**

  | m | where |
  |---|---|
  | bars per figure (docstring says 12) | `epoch_kit:182`, `epoch_figures.contrast_marks` |
  | up to 36 cells | `matrix_bootstrap` |
  | pixels or resels | `beta_maps`, `frozen_pixel_unique_sig` |
  | fixed x8 | `poststroke_compare:899`, `fr_contra_orofacial:167` |
  | x18 | `jaw_only_incomplete_figs:164`, `lick_cued_aim_by_spout:85` |
  | x36 | `lick_path_softmax:85`, `softmax_neural_vs_lick:98` |
  | x3 | `face_prelaunch_aim:551` |

- `disengagement_tests` replaces NaN with 1.0 before Holm, which inflates m.

### LMMs
- **Canonical `mixed_models.fit_lmm`:** REML; a re_formula ladder (`~post+slope` → `~post` → `~1`) with an OLS fallback; `lday = log(clip(day,1))` with pre = 0; pre sessions ≥ "0806".
- **About 17 ATT ad-hoc calls** (`lick_balance_*`, `lick_recovery_*`, `lick_correction:397`, `revup_behaviour:483`, `face_prelaunch_aim:513`, …): ML, `logday = np.log(day)` on post-only rows, random intercept only, no convergence check.
- `lick_balance_imaging_control*`: REML with `C(epoch, Treatment('pre'))`.
- `learned_compensation:256`: adds a session variance component and the lbfgs/powell optimizers.
- MAIN `cohort2_qc/quiet_state_response:263`: **groups = session**, with animal as a fixed effect.
- `mixed_models.py:218` hard-codes a Dark2 animal palette.

### Permutations
- Two cluster-permutation implementations: `beta_maps:663` (n = 500) and `orofacial_stats:122` (n = 1000, but called with 500).
- Four copies of the sign-flip / max-stat test.
- Within-animal n_perm is 200, 500, 1000, 2000 or 5000 depending on the file.

### Single home
- `analysis_kit` for every bootstrap and correction. Add `boot_delta_sessions` only if session weighting is a stated choice, and name it as such.
- One `bh(p, nan="drop")`; delete the other four.
- `mixed_models` owns every LMM: add `fit_lmm(..., method="ML")` and a `post_only` option rather than inline calls.
- One `cluster_permutation`, shared by `beta_maps` and `orofacial_stats`.

---

## 8. Spout order, sign conventions, colours, markers, stars

### Spout order
- **Canonical:** `orofacial_cohort.POS_ORDER` (:69): close ipsi / mid / contra, then far ipsi / mid / contra.
- The code→name maps are defined five times and agree (`roi_activity:45`, `daq_trials:52`, `plot_spout_trial_averages:26`, …).
- The canonical list is re-typed as a literal in about 40 scripts.
- **Competing orders:**
  - far-contra first: `grant_kit.py:54` `POS` (the same file has the canonical `CONF_LABELS` at :697), `best_match.py:40`
  - contra first: `misdirected_lick_decode:35`, `frozen_spout_maps:25`
  - left to right: `face_rhythm/common.py:18`, which calls itself "the one canonical order" even though the same file defines the real canonical order at :23
  - left to right with the mid ring swapped: `lick_templates:37`, `tongue_jaw_mismatch:101`, `jaw_lateral:263`
  - alphabetical: `epoch_kernels:48`, `frozen_kernel_maps:27`
  - centre first: `spout_world:62`
- The same spout is called "near middle" (`epoch_figures._ANATOMICAL`), "close mid" or "centre" depending on the file.

### Sign conventions
All of these agree today, because all 8 animals have `stroke_laterality: "L"`.
- "Image right" means **ipsi** in face video but the **contra hemisphere** in brain maps (consistent with the user's memory note on image sides). This is a labelling hazard.
- **Latent double flip.** `orofacial_cohort._cue_traces:433` and `session_metrics:500` already flip to + = ipsi. `to_contra` then flips the same "image-signed" columns again. For a non-left lesion the code prints a warning instead of fixing it.
- **Hard-coded "L":**
  - `pos_key(p, "L")` in about 25 sites (`stray_licks:63`, `orofacial_heatmaps:449`, `iti_first_lick:83`, `imaging_lmm:92`, …)
  - `_pos_color(p, "L")` in 8 places in `orofacial_cohort`
  - about 10 literal rename dicts (`attempt_coding.LESION_NAME:58`, `disengagement_tests:57`, `gate_lp_check:51`, …)
  - `face_rhythm/common.SIDE`
  - about 30 attempt scripts that map `_right` → contra
- `kinematics_pooled_figs._sg` flips the 3-D measures by −1 without reading laterality, while `fig_angle` in the same file does read it.
- **ATT `lick_balance_target_model.py:129-130` infers the sign from the data** (`np.sign` of the pre far-contra mean).

### Colours
- **Epoch colours.**
  - Canonical: `orofacial_cohort.EPOCH_LINE` (:113).
  - A second sanctioned palette, `epoch_figures.EPOCH_GREY`, plus the new `epoch_dot_color`.
  - About 12 other palettes. Several collide with animal colours: `cam1_lick_aim:35`, `session_lick_summary:90`, `reembedding_test:71`, `gate_lp_check:458`, and the seaborn sets.
- **Animal colours.**
  - Canonical: `config.animal_color()`, reading `animals.yaml` (CLAUDE.md rule 3).
  - A Dark2 palette appears in about 16 places, including `mixed_models.py:218`.
  - Wrong mappings: `locanmf_decoder_weights.py:175` draws **PS94 in PS93's red**; also `quiet_area_spikes:211`, `sm_coherence_cca:723`, `restnl_rescue:501`, `restw_strict_check:505`.
  - Six retired `ANIMAL_COLOR` dicts are still in the code.
- **Position colours.**
  - Canonical: `spout_behavior.position_style`.
  - Reversed-lightness or green-centre palettes in `pose3d_figures:25`, `pose3d_phase`, `iti_where_scatter:36`, `lick_trajectories_3d:112`, `spout_reference_frame:37`.

### Markers
`ANIMAL_MARKER` is defined in both `cat_points:25` and `orofacial_cohort:115`, and re-typed in about 28 scripts. The values agree.

### Stars
Canonical: `sig_marks.star_top` / `reserve_headroom`. What `*` and `**` mean varies:
- CI excludes 0 / survives Bonferroni: `epoch_figures`, `matrix_bootstrap`, `jaw_only_incomplete_figs`
- CI / survives Holm: `face_rhythm/stats`, ATT prelick and within_lick scripts
- p-value tiers on q or raw p: `cohort2_qc/quiet_*`, `plot_poststroke`
- `orofacial_cohort` alone uses three rules: the `STAR` dict (`*` both, `(*)` MWU only, `b`), CI only, and ad-hoc per-animal text
- `†` = LMM: `iti_lick_order`

Placement:
- ATT `within_lick_correction.py:653` defines a fallback `star_top`, which six scripts import.
- Ad-hoc `ax.text("*")` appears in 1 wfield_local module, 12 MAIN scripts and 24 ATT scripts. `star_top` is used in 4, 26 and 13 respectively.

### Single home
- A small `wfield_local/spouts.py` (or a section of `position_axes`) owns `POS_ORDER`, the code↔name maps, `pos_key`, `to_contra` and `ipsi_sign(animal)`, all reading `stroke_laterality`. Move these out of `orofacial_cohort`.
- `cat_points` owns the epoch, animal and position colours and the markers.
- `sig_marks` owns star placement and one documented legend: "`*` = 95% CI excludes 0; `**` = survives <named correction>". Each figure states its correction.

---

## 9. Paths

Canonical: `PathResolver` / `config.resolver()`.
- **Two hard-coded `N:` paths after the M/N swap** (N is standby now):
  - `scripts/rest_migration/publish_cache.py:33`: `DEFAULT_DEST = "N:/MICROSCOPE/…/session_cache_v12"`. This is a **write** target and would land on standby.
  - MAIN `cohort2_qc/swap_raw_check.py:33`.
- **The v36 run root is a literal** `M:/…/lp_r4_heatmap_20261013__clean_v36/imaging_encoding` in about 19 MAIN / 23 ATT files (`coupling_by_class:31`, `early_predictors:25`, `recovery_timing:29`, `learned_compensation:43`, `prelaunch_aim_tracking:40`, …). The resolver form would be `resolver().root("orofacial") / oc.RUN / "imaging_encoding"`.
- About 45 other hard-coded `M:/` and `E:/` roots in `cohort2_qc`, FaceRhythm `ROOT` literals, and `session_cache.py:115` falling back to `C:/Users/sabatini`.

**Single home:** add `orofacial_outputs.encoding_root()` and have scripts call it.

---

## 10. Clarity for non-coders

- **Docstrings are good.** Only `scripts/histology_stroke_extent_figure.py` lacks one. Scripts without a usage/run line in their first 80 lines: 22 of 280 on MAIN, 54 of 391 on ATT.
- **There is no "where is each quantity defined" page.**
  - `docs/ARCHITECTURE.md`'s Foundation table lists config, paths, writeguard, epochs, parallel and analysis_kit.
  - None of `POS_ORDER`, the trial classes, `jaw_pass_qc`, `lp_licks`, `gate_source`, `sig_marks`, `cat_points` or `mixed_models` appears in ARCHITECTURE.md, README.md, wfield_local/README.md or scripts/README.md (grep).
  - CLAUDE.md rule 9 lists only `analysis_kit`'s contents.
- **Definitions live in large modules with unrelated names.** Spout order and epoch colours sit in `orofacial_cohort` (~2,600 lines). The recovery axis sits in `mixed_models`. The trial classifier is a *script*.
- **Misleading names:**
  - `jaw_pass_qc` (it means "jaw moved")
  - `docs/WHERE_THE_CODE_MOVES.md` (it is about cortex, not code layout)
  - `face_rhythm/common.POSITIONS`, labelled "the one canonical order" but not canonical
  - two different functions named `_load_daq_events`
  - `scripts/README.md` calls `scripts/` "one-off utilities … not imported". In fact it holds about 280-390 primary analyses, and scripts import each other (e.g. `iti_licks.pred_file`).
- **Stale comments and docstrings:** `orofacial_cohort:~1580`; `jaw_onsets:59`; the comment at `epochs.py:104` contradicts its docstring; the `config.stroke_date` docstring says "last pre-stroke session".

---

## 11. Proposed target layout (consolidate into existing modules)

| concept | owner | call sites to switch |
|---|---|---|
| session registry, `date8` / `date_key` / `animal_of`, cohorts | `config` | ban `[:4]`, `"2026"+`, `lab[5:]` and MMDD string compares (~100 ATT / ~70 MAIN sites; `mixed_models:59`, `analysis_kit.curated_ids`) |
| epochs, days since stroke, `epoch_of_day`, **baseline set** | `epochs` (+ `defaults.yaml baseline.august_from`) | `mixed_models:63,78`, `frozen_encoder:1641,1736`, `oc.LATE_PRE_FROM`, ~80 literal cuts, `epoch_figures.epoch_of_day` |
| curated sessions, bootstrap, change-vs-pre, BH / Holm, `boot_p` | `analysis_kit` | the `epoch_figures` / `matrix_bootstrap` / `epoch_behaviour` contrasts, 4 BH copies, ~30 script `_boot` helpers |
| engagement gate, `responded_per_cue`, the two named windows | `rest_engagement` (+ `gate_source` after the merge) | `rest_by_position._engaged_trials`, `nolick_decoder.categorize` gate input, the 2 s cuts in `locanmf_decoder_weights` / `position_encoder`, `RESP_S` users |
| DAQ licks (one named imaging identity) | `behavior_events` + `analysis_kit.lick_samples` | 48 incantation copies |
| video licks, cam1-only filter | `lp_licks` | `lp_licks._read_times`, `trial_attempts`, `cam1_*`, `jaw_only_incomplete_figs` |
| **trial / attempt classes** | new `wfield_local/trial_classes.py` (the body of `scripts/trial_attempts.py`) | `jaw_only_incomplete_figs.classify` (delete), `attempt_coding`, `attempt_maps`, `lp_licks` |
| jaw-moved call (nullable bool), jaw opening | `jaw_kinematics` | `cam1_incomplete_licks:56`, `orofacial_cohort:501/526/546`, `cam1_licks:110/179`, `orofacial_jaw_lateral` |
| 3-D frame loader | `pose3d` | ~18 direct readers |
| FaceRhythm factors and groups | `face_rhythm_outputs` + `fr_groups` | ~10 direct `np.load` / glob readers |
| spout order, names, ipsi/contra, sign | `position_axes` (or a new `spouts.py`) | ~40 literal lists, ~25 `pos_key(p,"L")`, `to_contra`, `kinematics_pooled_figs._sg` |
| colours and markers | `cat_points` | `EPOCH_LINE` / `EPOCH_GREY`, ~12 epoch and ~16 animal palettes, 28 marker dicts |
| stars | `sig_marks` | the ATT `within_lick_correction` fallback, ~37 ad-hoc `ax.text("*")` |
| LMMs and the deficit axis | `mixed_models` | ~17 ad-hoc `smf.mixedlm` calls |
| output roots | `orofacial_outputs.encoding_root()` + `PathResolver` | ~23 v36 literals, the `publish_cache` `N:` path |

Then add one table, "quantity → function → module", to `docs/ARCHITECTURE.md`, and point CLAUDE.md rule 9 at it.

---

## 12. Open / UNVERIFIED

- Whether the gate's terminal-collapse point ever differs between the three "responded" inputs (needs a run).
- The contents of `epoch_boundaries.json`.
- Whether 0805 files exist for the non-encoder raw readers (`qc_running_*`, `state_spout_check`). All encoder folders were checked: none has a 0805 file.
- The bodies of most ATT `iti_*` / `lick_*` scripts (only grepped).
- The internals of `lp_multiview`, `dlc_anipose` and `face_rhythm_readouts`; the `frc` helper.
- Where the deck's "trial class" copy lives (not found).
- Housekeeping: a read-only `ls` loop over the encoder folders ran in the background. It finished, wrote nothing, and found no 0805 files.
