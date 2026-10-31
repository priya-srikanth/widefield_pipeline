# DISENGAGEMENT PRE-REGISTRATION — why do animals quit faster after the stroke, and what is the quit state?

**Written 2026-10-31, BEFORE any of the four tests below was computed.** Priya's question (10-31): *why do animals
disengage / quit faster after the stroke, and what is the disengaged state?* Cohort PS92-PS95 only (cohort 2,
PS118 / 120-122, is never used here). Stage 1 = this document only; stage 2 (`scripts/disengagement_tests.py`) runs
after Priya / the coordinating session says go, and appends **Outcome** + dated **Deviations** at the bottom. Everything
here is **PROVISIONAL, n = 4 animals**.

---

## 0. What is already known (the starting point, not re-tested)

| claim | number | source |
|---|---|---|
| most of the acute drop in responding is quitting | responded 0.94 -> 0.64 all trials; 0.98 -> 0.82 engaged | brief 10-31 |
| quitting is TIME-driven, not lick-budget-driven | log(licks)~log(rate) +0.72 [+0.36, +1.06]; log(time) -0.13 | STATUS_2026-09-20, `quit_point` |
| quit prodrome | -18.7 licks/min [-22.9, -13.7] in the 10 min before, vs time-matched controls | `quit_prodrome` |
| within-session engagement decline steeper subacute | -6.8 near / -5.8 far vs own pre | STATUS_2026-09-20 |
| persistent bout-MAINTENANCE deficit | within-bout deceleration +15-25 ms every epoch, 4/4; steepens within session acutely (+10.1 [+0.9, +19.1]) | `lick_bout_structure`, FINDINGS 7 |
| hit rate falls within post sessions while launch aim does not | -0.13 | brief 10-31 |
| stopped (quit-period) trials: NO cue-evoked response, only a flat raised level | rise vs rest mostly the reference falling; late-session confound flagged | FINDINGS "Stopped-trial traces 10-29", `scripts/stopped_trial_traces.py` |

The step framing of `quit_point` holds: a quit is a THRESHOLDED readout, so which accumulated quantity (time, licks,
water) is conserved at it — or whether none is and the threshold itself moved — is the question.

---

## 1. Data sources — verified 10-31 (what exists, exactly)

### 1a. Sessions (curated, AUGUST-ONLY pre; `WIDEFIELD_PRE_FROM=0801`, `analysis_kit.curated_sessions`)

All 108 sessions pass `session_behavior` (>= 60 trials). Shared epoch boundaries (`epochs.epoch_of`).

| animal | pre (Aug) | acute | subacute | chronic |
|---|---|---|---|---|
| PS92 | 8 | 5 | **2** | 12 |
| PS93 | 8 | 4 | 3 | 12 |
| PS94 | 8 | 6 | 5 | 8 |
| PS95 | 8 | **1** | 7 | 11 |

### 1b. Quit detection survey (existing detector, measured 10-31 BEFORE any test; `session_quit` via `session_behavior`)

Sessions with a detected (confirmed, non-recovering) quit / sessions; median session length (min):

| animal | pre | acute | subacute | chronic |
|---|---|---|---|---|
| PS92 | **0/8** (148) | 5/5 (115) | 0/2 (120) | 0/12 (120) |
| PS93 | 1/8 (151) | 3/4 (121) | 2/3 (121) | 1/12 (120) |
| PS94 | 4/8 (95) | 4/6 (103) | 5/5 (101) | 5/8 (100) |
| PS95 | 6/8 (110) | **0/1** (120) | 6/7 (96) | 6/11 (96) |

**Consequences fixed now, before any outcome is seen:** (i) PS92 has NO pre-stroke quit and PS93 one, so no test may
need a pre-stroke QUIT in every animal; (ii) PS95's single acute session is censored, so an acute contrast that needs
a quit has 3 animals at most; (iii) PS94 / PS95 quit often in EVERY epoch including pre, PS92 only acutely. Tests 1-2
are therefore built on quantities defined for every session; tests 3-4 are quit-vs-matched-control contrasts pooled
over epochs.

Session termination (events.csv): post sessions mostly end at `target_trials_reached` (100 / position) or
`max_duration_reached` (120-150 min); pre sessions ran 74-167 min, often stopped by hand at the quit (the
right-censoring argument of `quit_point`). **One session ended at the water limit** (`water_limit_reached`,
PS95_0809 pre, 1500 µl delivered, 677 hits / 5 misses): censored, not a quit.

### 1c. Water (Test 1) — what is logged

* **Per-reward NOMINAL volume** `reward_ul` in each session's `gui_config.json` (1.5-2.9 µl; ~2.0 typical) with valve
  time `reward_ms` (8-20 ms); **mid-session changes are logged** in `config_changes.csv` (`set_ack task.reward_ul`,
  GUI-time-stamped; e.g. PS93_0814 2.9 -> 2.5 µl in the first 2 min). `reward_ul` is a calibration NUMBER the
  experimenter enters, not a measured drop.
* **Every delivery**: `events.csv` `reward` rows (`reward_type` auto / manual) and the DAQ `reward_ttl` channel (what
  `session_quit` already counts as `quit_cum_rewards`). `summary_end.json` `water_ul` = the GUI's own running sum
  (e.g. PS92_0821: 397 rewards, 794.0 µl); `water_limit_ul` 1500 in every session.
* **Delivered is not consumed.** `reward_mode auto_after_delay`, delay 0: the drop is delivered ~15 ms after the cue
  on EVERY trial whether or not the animal licks (withheld only by `auto_hold_after_miss` after 6 consecutive misses),
  and the spout docks at a wick after the 3.5 s window. A miss therefore leaves the drop uncollected. **Collected
  water** (the satiety proxy) = deliveries on trials with a DAQ spout lick inside the session's response window
  (3.5 s, `gui_config timing.response_window`) x the `reward_ul` in force; **delivered water** is reported beside it.
  Partial collection and drop-size error are unmeasurable (limit).
* **Body weight**: `configs/animal_weights.csv` (`wfield_local/animal_weights.py`), PRE-session weight, 49 dates per
  animal from 08-01 on, covering every analysed session day. **No water-schedule / home-cage supplement record exists
  in the repo or configs** (searched); none is invented.
* Free / manual rewards: `free_rewards` 0 in every summary; `manual_reward` events exist in a few sessions (PS93_0813
  19); manual deliveries are counted as delivered and as collected only on a responded trial.
* Adaptive spout distance was ON in some PS93 pre sessions (0810-0813) and PS93_0820 acute, PS92_0820, PS94_0811
  (`adapt_advance` / `adapt_decrease` events): recorded as a limit, not excluded.

### 1d. Movement (Test 4)

* **Running**: DAQ treadmill, `behavior_events.session_speed` (calibrated mm/s; running = > 3 mm/s bouts as
  `compute_events`). Every session.
* **Face**: FaceRhythm cam4, frozen master set `master_pre_v1` (rank 12), whole-session projections at 5 Hz
  (`<o2>/fr_cam4_20261021/factors/master_pre_v1/proj/<animal>_<date>_cam4.npz`, key `session` (n, 12) +
  `session_t_daq_s`; 108 files; PS93_0813 has no video). Groups from `fr_groups.master_v1`: lick factors 2, 4, 5, 7, 8,
  11; "other" (uninstructed) 0, 1, 3, 6, 9, 10.
* **cam4 / cam1 video motion energy**: whole-session files exist for only 23 sessions
  (`session_poses/*_full/video_motion_cam{1,4}.npz`), cam2 / cam3 for 1 — **not cohort-wide; not used**.
* **PUPIL: NO.** Evidence: no pupil or eye keypoint in any pose model (the side-camera worksheet `SIDECAM_GUIDE.html`
  says "Leave eyes and whiskers empty"; cam4 / cam1 models are tongue / jaw / nose / spout); `grep -ri pupil` over the
  repo finds only the 10-26 handoff line that schedules this check; FaceRhythm cam2 / cam3 (the views that contain the
  eye, `FACE_RHYTHM_PIPELINE.md`) are prepped only. The eye IS visible in raw cam2 / cam3 video, so a pupil measure is
  possible in principle; per the brief no tracker is built.

### 1e. Cortex (Test 4)

* Imaging through `combined_encoding.load_imaging` with `WFIELD_INPUT_CACHE=C:/Users/SabatiniLab/wfield_cache`
  (adopted `meegkit_hpfit` SVTcorr: order-10 robust polynomial detrend, 50 % cutoff ~43 min); area traces = the
  ROI v8 masks (`iti_licks.roi_masks`, ALM / tjM1 split from MOs / MOp), x1e3, as `stopped_trial_traces`.
* **Reference = RESTnl** (`scripts/restnl.py`, rescue rule `s20_f50_wr`, 107 / 107 sessions; lick-free rest, ENGAGED
  period only). RESTW kept as sensitivity.
* **The extra 600-s slow-drift basis that `stopped_trial_traces` / `lick_precue_restw_traces` remove
  (`DRIFT_MIN_PERIOD_S`) is NOT removed for the level measures here**: over 10-20 min windows it would absorb part of a
  step at the quit, which is the object. Sensitivity: with the 600-s basis removed (= the existing construction).

### 1f. Engagement-gate status (REST_ENGAGEMENT_AUDIT)

The quit period IS the object here, so no consumer below gates it away silently. Every window is defined explicitly
(section 2). The **reference** (RESTnl) stays engagement-gated — it must not contain the quit period, or the contrast
would subtract the state being measured. Per-trial behavioural covariates are measured BEFORE the quit onset only.

---

## 2. Definitions

* **Trials**: DAQ cues with repaired positions (`engagement_decomposition.session_trials`); positions named by lesion
  side (all lesions LEFT: L = ipsi, R = contra; `orofacial_cohort.POS_ORDER`).
* **Quit onset**: the existing detector, unchanged (`quit_point.session_quit` through `analysis_kit.session_behavior`):
  first trial of the terminal, NON-recovering collapse of the REFERENCE-spout (close ipsi + close mid) response rate
  (`precue_engagement_states.engagement_gate`, 15 reference trials, < 0.5), backdated to the trial after the last
  reference response. Scored at the detector's 2.0 s window so every earlier quit number stays comparable. The
  reference spouts are unaffected by the motor deficit, so the quit is not a far-contra miss run.
* **Censored session**: no detected quit; observation = whole session (a LOWER bound on time / water engaged).
* **Quit period**: [quit onset, quit onset + 10 min] (and to session end for descriptive traces).
* **Prodrome window**: [quit onset - 10 min, quit onset).
* **Time-matched control window** (tests 3-4): the SAME clock interval in another session of the SAME animal and SAME
  epoch that has no quit before the end of that interval (`quit_prodrome` panel B construction). A quit with no such
  control is dropped and counted. Sensitivity: controls from any epoch of the same animal.
* **Lapse** (trial-level disengagement, test 2): a reference-spout (close ipsi / close mid) trial with no DAQ lick in
  the 3.5 s task window, BEFORE the quit onset.
* **Response / hit for covariates**: DAQ lick inside the session's 3.5 s task window (what the animal experienced as
  getting the drop); the 2.0 s quit-detector scoring is used only for the quit point.
* **Collected water**: section 1c. **Local collected reward rate**: collected µl / min over the preceding 5 min.
* **Block**: run of consecutive same-position trials (`balanced_block_cycles`, 4-8 trials, scheduled by the task, NOT
  chosen by the animal).

---

## 3. The four tests

### TEST 1 — SATIETY

Measures per session: collected water at quit onset (censored: at session end); delivered water at the same point;
rewards (deliveries) and hits before the quit; total collected / delivered per session; collected water per minute
(the accrual rate); pre-session body weight as % of the animal's August pre mean.

**PRIMARY P1**: collected water at quit-or-end, **acute minus pre**, `boot_delta` (animal-weighted), per-animal table
first.

Why this survives the censoring: pre sessions are more censored and ran longer (PS92 0 / 8 quits, sessions to 148 min),
so measured pre UNDERSTATES true pre water-at-quit (an animal that collected X µl and kept working had a threshold
> X). A post SHORTFALL is therefore conservative; a post EXCESS can be censoring alone. Caveat checked per animal: the
argument needs pre at least as censored as acute — true for PS92 / PS93 / PS94, **false for PS95** (pre 6 / 8 quits,
acute 0 / 1), so PS95's acute cell is reported but flagged.

Secondary: same for subacute and chronic; Kaplan-Meier "still engaged" against cumulative collected water per animal x
epoch (descriptive); accrual rate pre vs post; delivered-water version; weight % pre per epoch and its session-level
relation to water at quit-or-end (descriptive, n = 4).

**Decision rule**
* P1 < 0 (Holm-surviving, CI excludes 0) and negative in >= 3 / 4 animals -> **post animals quit after LESS water than
  they collected while still working pre: not the same satiety point.** The threshold moved (malaise / motivation, or
  a lowered satiety set-point — this test cannot separate those two).
* P1 95 % CI inside +/- 15 % of the pooled pre mean -> **consistent with satiety at the same water level**; quitting
  earlier in time would then need faster water accrual post (checked against the accrual rate — if accrual is SLOWER
  post, satiety cannot explain earlier quits).
* P1 > 0 -> uninterpretable (differential censoring), stated as such.

### TEST 2 — FRUSTRATION / REWARD RATE

The causal handle: **blocks are scheduled by the task**, so a far-contra block is an EXOGENOUS run of trials that,
post-stroke, the animal fails (acute far-contra hit ~0.03) and so collects no water (and can trip the 6-miss reward
hold). Pre-stroke the same block is not a failure run. A falling local reward rate driven by the animal's own
disengagement is circular; one driven by the schedule is not.

**PRIMARY P2**: per session, lapse rate in REFERENCE blocks that immediately follow a **far-contra** block, minus lapse
rate in reference blocks that immediately follow any other non-reference block (close contra, far ipsi, far mid);
trials before the quit onset only; a session needs >= 3 reference blocks in each arm. **Acute minus pre**,
`boot_delta`.

Secondary:
* the same with recent FAILURES (count of no-lick trials at contra spouts in the previous 16 trials) instead of block
  identity (endogenous; descriptive);
* **quit-triggered schedule**: fraction of far-contra trials in the 16 trials before each quit onset minus that
  session's scheduled fraction (exogenous; > 0 would mean quits follow far-contra runs), per animal x epoch;
* **trial-level logistic** of lapse on recent far-contra exposure, recent failures by side, local collected reward
  rate, time in session, cumulative collected water — fitted within animal x epoch where >= 10 lapses, coefficients
  shown per animal (descriptive; not tested);
* **foraging threshold**: giving-up rate = local collected reward rate in the 5 min before each quit onset. Compared
  with the animal's PRE-stroke distribution of 5-min collected rates during engaged working (all August pre sessions,
  windows before any quit). Reported: each post quit's giving-up rate as a percentile of the animal's pre working
  distribution; the pre 5th / 50th percentiles drawn on the figure. No pre-vs-post test of the giving-up rate itself
  (PS92 has no pre quit).

**Decision rule**
* P2 > 0 (Holm-surviving) in >= 3 / 4 animals, with the pre contrast ~0 -> **failure runs at the deficit spout
  provoke disengagement post-stroke: frustration / local reward rate contributes.**
* P2 null -> no evidence that scheduled failure runs drive lapses (with the power limit stated).
* Threshold: post giving-up rates BELOW the animal's pre 5th percentile in >= 3 / 4 animals -> **consistent with an
  unchanged threshold that the motor deficit pushes the reward rate below** (disengagement follows from the deficit).
  AT OR ABOVE the pre 5th percentile in >= 3 / 4 -> **animals quit while still earning at rates they worked through
  pre: the threshold moved (motivation / state).** Mixed -> no conclusion. Stated limit: the 5 min before a quit
  overlaps the prodrome, so a low giving-up rate can be the start of disengagement rather than its cause; the
  block test (P2) is the one that is not circular.

### TEST 3 — MOTOR FATIGUE

Measures in the prodrome window vs time-matched control windows, on REFERENCE-spout responded trials only (no
far-contra composition effect): within-bout deceleration (last-third minus first-third median ILI, bouts >= 6 licks,
gap 0.5 s — `lick_bout_structure`), median within-bout ILI (`quit_prodrome`, >= 4 licks), licks per bout, first-lick
latency.

**PRIMARY P3**: within-bout deceleration, prodrome window minus time-matched control, **all quit sessions pooled over
epochs** (August pre included), paired within animal (`boot_delta(quit windows, control windows)`).

Secondary: the other three measures; by epoch (descriptive); the same with controls matched on local collected reward
rate as well as time (+/- 20 %) — the "beyond reward rate" version; a 30-min quit-aligned trajectory in 5-min bins.

**Decision rule**
* P3 > 0 (Holm-surviving) in >= 3 / 4 animals AND still > 0 in >= 3 / 4 after reward-rate matching -> **motor
  slowing precedes the quit beyond time in session and reward rate: fatigue contributes to the quit.**
* P3 null while the lick-count prodrome (-18.7 licks/min) stands -> **the prodrome is disengagement (fewer licks),
  not a slower tongue**; fatigue stays a within-session-decline contributor only (DECISIONS 09-20), not the
  terminator.

### TEST 4 — THE STATE

Windows: quit period vs time-matched engaged control; prodrome vs control; 30 min before to 20 min after the quit in
5-min bins (descriptive trajectory).

Measures: cortical PRE-CUE level (-2..0 s before each cue; the spout is already at the mouth) minus the session's
RESTnl level, GLOBAL (mean of all ROI v8 areas) and per area (contra / ipsi tjM1, ALM, SSp-m, SSp-n; VISp, RSP as
non-orofacial references); trial-to-trial SD of the global pre-cue level within the window (slow fluctuation sampled
once per trial, free of cue-evoked responses); cue-evoked response (0..0.5 s minus -0.5..0 s) on reference trials,
trend approaching the quit; running (mean speed, fraction of time > 3 mm/s); face movement (summed activation of the
FaceRhythm "other" factors, and all 12, between-trial samples only so the lick factors' task drive does not dominate).
Descriptive only: 0.01-0.05 Hz band power of the continuous global signal (it falls mechanically when evoked
responses stop, so it is not interpreted alone).

**PRIMARY P4**: global pre-cue level (RESTnl-referenced), quit period minus time-matched control, all quit sessions
pooled over epochs, paired within animal (`boot_delta`). Imaging must cover the window; sessions whose imaging ends
early are dropped and counted.

**Decision rule** (one state label per measure set, pre-specified)
* P4 > 0 (Holm-surviving, >= 3 / 4) -> the raised tonic level of stopped trials is a property of the disengaged
  STATE, not of late session time (the confound flagged 10-29). P4 null -> it was the late-session / reference
  confound.
* Movement + fluctuation pattern (each >= 3 / 4 animals to count):
  * running down AND "other" face movement down AND pre-cue SD up -> **quiescent / low-arousal disengagement**;
  * running up OR "other" face movement up -> **active disengagement (doing something else)**;
  * neither -> **no movement signature: the animal stops licking but is otherwise unchanged**.
  No pupil means arousal is inferred, not measured — said on every figure that uses the word.
* Cue-response trend: a decline in the 10 min before the quit beyond controls (>= 3 / 4) -> cortex disengages from
  the cue BEFORE the behavioural quit; else the loss of the cue response is concurrent with the stop.

---

## 4. Statistics

* **Per-animal table FIRST** for every result (ground rule 8): n sessions / n quits / value per animal x epoch, printed
  before any bootstrap. Anything not in >= 3 / 4 animals individually does not survive, whatever the CI says.
* `analysis_kit`: `boot_delta` (paired animals -> sessions, animal-weighted) for every change and every
  quit-vs-control contrast; `boot_ci` only for levels; `boot_p` on the `draws=True` draws; `holm`. N_BOOT 4000, one
  seeded rng, sessions collected in `curated_sessions(order="sorted")` order via `fan_sessions` (<= 4 workers).
* **Primary family = {P1, P2, P3, P4}, Holm at 0.05.** Everything else is secondary / descriptive and labelled so;
  secondary families get BH only if a table needs it, never promoted.
* Single-session cells (PS95 acute, PS92 subacute 2) are printed and flagged; their uncertainty does not enter the CI.
* Sensitivity (reported, not tested): 80-min common horizon (`quit_point --horizon-min`; keeps PS94 pre 6 / 8);
  RESTW instead of RESTnl; 600-s basis removed; controls from any epoch; delivered instead of collected water;
  2.0 s instead of 3.5 s response window for covariates.
* **No frozen pre-stroke model is planned** (P1-P4 are contrasts, the threshold reference is a percentile). If stage 2
  ends up applying any model fitted on pre to post, it goes through `frozen_models.load_or_fit` (store
  `C:/Users/SabatiniLab/frozen_models`, `local_dir()` printed first) and is logged as a deviation.

---

## 5. Figures planned (all regenerable from `figdata` pickles; `--plot-only` / `plot` mode)

Outputs: `M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/disengagement/`
(`figures/`, `figdata/`, `tables/`). Conventions: sessions = open markers, shape = animal (PS92 o, PS93 s, PS94 ^,
PS95 D), colour = epoch (`orofacial_cohort.EPOCH_LINE`), pooled = filled epoch-colour marker, June not used (August-only
pre); no figure numbers in images; no epoch connecting lines; 50 / 95 % marked per epoch on distributions; stars in the
top band (`star_top` + `reserve_headroom`); canonical spout order; titles say BEHAVIOUR or NEURAL; censored sessions
carry an upward tick (lower bound).

1. BEHAVIOUR — water collected at quit-or-end per session by epoch (P1), + delivered; per-animal KM against water.
2. BEHAVIOUR — water accrual rate and pre-session weight (% pre) by epoch.
3. BEHAVIOUR — far-contra-block lapse contrast per session by epoch (P2); quit-triggered far-contra exposure.
4. BEHAVIOUR — giving-up rate of each post quit on the animal's pre working-rate distribution (5th / 50th marked).
5. BEHAVIOUR — quit-aligned deceleration / ILI / licks per bout / latency vs time-matched controls (P3).
6. NEURAL — quit-aligned global and per-area pre-cue level vs controls (P4); pre-cue SD; cue-response trend.
7. BEHAVIOUR — quit-aligned running and FaceRhythm movement vs controls.
8. **BEHAVIOUR (descriptive, pre-registered as such)** — licks per responded trial by spout (POS_ORDER) and epoch.
   Source = the quantity behind FINDINGS orofacial claim 6 ("chronic: more licks per responded trial"):
   `orofacial_cohort` `licks_per_resp` (cam4 LP licks, cue + 70 ms .. DAQ trial stop, engaged trials), read from its
   per-session table, not recomputed; DAQ-contact count per responded trial as a second panel. No test.

---

## 6. Limits (stated now)

* n = 4 animals; quits are sparse pre (PS92 0, PS93 1) and PS95 acute is one censored session.
* Water is NOMINAL (`reward_ul`), collection is inferred from a lick in the window; home-cage water is not logged.
* Pre vs post recording policy differs (pre stopped by hand / longer; post 120 min / 100 per position); handled by the
  censoring argument and the time-matched designs, not removed.
* The quit point is a detector output; quits near a session's end are unconfirmable (censored).
* Adaptive spout distance in a few sessions (section 1c).
* No pupil; arousal is inferred from movement and cortical fluctuations.
* Test 4 cannot separate "state causes stopping" from "stopping causes the state" (no licks / cue responses ->
  different cortical input); the time-matched control removes clock time, not that.

---

## 7. Decisions needed from Priya (before stage 2)

1. **Primary epoch for P1 / P2 = acute** (consistent with every other result). PS95 acute is ONE censored session, PS92
   subacute two. Alternative: an "early post" = acute + subacute contrast as primary. Default if no answer: acute.
2. **Satiety proxy = COLLECTED water** (drop delivered at the cue on every trial, uncollected on a miss) rather than
   delivered water. Default: collected primary, delivered secondary.
3. **Test 3 / 4 controls = same animal, same epoch, time-matched**; quits without one are dropped (PS92 loses its
   latest acute quit). Default: as stated, any-epoch controls as sensitivity.
4. `star_top` / `reserve_headroom` live in `scripts/within_lick_correction.py`, not `wfield_local/sig_marks.py` as the
   brief says. Default: import them from there (no copy); moving them into `sig_marks` is a separate shared-module edit.

**RESOLVED 10-31 before any test (Claude, on Priya's behalf; Priya has not reviewed -- listed in the handoff):**
1. PRIMARY epoch for P1 / P2 = **EARLY POST (acute + subacute pooled)**, because PS95 acute is a single censored session
   and PS92 subacute two, so an acute-only primary cannot meet the >= 3/4-animal rule as designed. Acute alone and
   subacute alone are reported as pre-specified SECONDARY cells; chronic as secondary. P3 / P4 unchanged (all quits pooled).
2. Collected water primary, delivered secondary (default).
3. Same animal, same epoch, time-matched controls; any-epoch controls as sensitivity (default).
4. Import star_top / reserve_headroom from scripts/within_lick_correction.py (default).

---

## Outcome

*(stage 2 — not yet run)*

## Deviations

*(dated entries appended in stage 2)*
