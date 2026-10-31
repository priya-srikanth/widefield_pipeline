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

**Stage 2 run 2026-10-31** (`scripts/disengagement_tests.py`; outputs
`<v36>/imaging_encoding/disengagement/{sessions,tables,figdata,figures}`). 108 sessions behaviour, 107 imaging
(PS93_0813 has no imaging input row). PROVISIONAL, n = 4. Per-animal tables are in the run log and `tables/`.

### Primary family (Holm over P1-P4)

| test | estimate [95 % CI] | Holm p | per animal (PS92 / 93 / 94 / 95) | verdict vs rule |
|---|---|---|---|---|
| P1 collected water at quit-or-end, early post - pre | **-290 µl [-428, -154]** | **0.001** | -160 / -349 / -259 / -392 (4/4 < 0) | **SURVIVES: not the same satiety point** |
| P2 lapse after far-contra block - after other block, early post - pre | +0.037 [-0.014, +0.092] | 0.27 | +0.059 / +0.023 / +0.013 / +0.054 (4/4 > 0, small) | not supported |
| P3 within-bout deceleration, prodrome - time-matched control, ms | -5.6 [-15.2, +3.6] | 0.27 | -8.9 / +2.0 / +1.4 / -17.0 (2/2 split) | null: no fatigue prodrome |
| P4 global pre-cue level (RESTnl), quit period - time-matched control | **+3.65 [+1.85, +5.30]** | **0.001** | +2.1 / +3.5 / +5.1 / +3.9 (4/4 > 0) | survives as tested, **but it is running (post hoc, Test 4)** |

### Test 1 — satiety: quitting is NOT satiety at the pre-stroke water level

Mean collected water at quit-or-end, µl (PS95 acute = 1 censored session):

| animal | pre | acute | subacute | chronic |
|---|---|---|---|---|
| PS92 | 712 | 561 | 532 | 644 |
| PS93 | 1010 | 564 | 796 | 960 |
| PS94 | 988 | 538 | 957 | 1050 |
| PS95 | 1160 | 908 | 753 | 1020 |

* Early post -290 µl, 4/4 animals. Acute -327 [-476, -178] (4/4), subacute -210 [-382, -45] (4/4), chronic -53 n.s.
  (back near pre).
* The shortfall holds where pre has real quits (PS94 4/8 and PS95 6/8 pre quits), not only through the censoring
  argument.
* Collected water ACCRUES SLOWER acutely: -2.1 µl/min [-3.9, -0.3]; 3/4 animals, PS92 +0.4. Satiety would then predict
  LATER quits, but the quits come earlier.
* Body weight is lower after the stroke: -4.8 % of August pre in the early post period [-6.3, -3.3], 4/4, and -5.7 %
  chronically. Lighter, if anything thirstier, animals quit after LESS water.
* Sensitivities:
  * Delivered water: early -180 [-329, -24]; acute alone n.s.
  * **80-min common horizon is WEAKER: early -105 [-232, +15], 3/4 (PS92 +24).** Inside the first 80 min the water
    collected differs less, because most post quits fall at 70-100 min.
* **Reading (PROVISIONAL):** post-stroke animals stop after collecting ~30 % less water than they took without
  quitting pre, while gaining water more slowly and weighing less. Satiety at an unchanged set-point does not
  explain the quit. Per the rule, the threshold moved. This test cannot separate malaise / motivation from a lowered
  satiety set-point.

### Test 2 — frustration / reward rate: not supported; threshold test inconclusive

* P2: 4/4 positive but small, CI spans 0.
* Acute cell 4/4 positive (+0.002 to +0.085); subacute and chronic ~0 or negative. The 2.0 s scoring gives the same
  picture.
* Lapses rise acutely after EVERY non-reference block: after a far-contra block 0.10-0.18 vs pre 0.003-0.10; after
  other blocks 0.07-0.14 vs pre 0.007-0.024. This is general disengagement, not specific to failure runs.
* Quit-triggered far-contra exposure (16 trials before the quit, minus that session's schedule): no consistent sign.
  Quits do not follow far-contra runs.
* Lapse logistic (descriptive):
  * The far-contra exposure coefficient is NEGATIVE in every acute cell (-0.5 to -0.8).
  * Recent contra failures are positive acutely (+0.3 to +1.4), but those are endogenous.
  * Local reward rate is negative almost everywhere, including pre (circular).
* **Foraging threshold:** each animal's post-stroke giving-up rates against its pre-stroke working distribution:

  | animal | share of post quits at / above the pre 5th percentile | median pre percentile |
  |---|---|---|
  | PS92 | 0.8 | 74 |
  | PS93 | 1.0 | 41 |
  | PS94 | 0.5 | 4.5 |
  | PS95 | 0.5 | 4.3 |

  Two animals quit while earning at rates they worked through pre; two sit at their pre floor. **Mixed, so no
  conclusion per the rule**; PS92 / PS93 point to a moved threshold.

### Test 3 — motor fatigue: no motor slowing before the quit

* P3 null (2/2).
* Rate-matched controls: -3.5 n.s. Any-epoch controls: -2.0 n.s.
* Within-bout ILI is SHORTER before the quit: -6.1 ms [-11.2, -1.1], 4/4.
* Licks per bout are fewer: -1.5 [-2.9, -0.1], 3/4 (PS92 +0.2); the trajectories show them falling over the last
  ~10 min.
* First-lick latency n.s.
* Caveat: shorter bouts end before their late, slower intervals, so a shorter median ILI can be bout length rather
  than a faster tongue.
* **Reading (per the rule): the prodrome is disengagement (shorter bouts / fewer licks), not a slower tongue;**
  fatigue is not the terminator.

### Test 4 — the state: active (running), not quiescent; cortex does not disengage before the stop

* **Movement:**
  * Running is up in the quit period, 4/4: speed +2.1 mm/s [+1.3, +3.3], running fraction +0.17 [+0.10, +0.28].
  * FaceRhythm "other" factors n.s. (2/4); all factors n.s.
  * Per the rule: **active disengagement (doing something else: running)**, not quiescent / low-arousal.
* **Cortex:**
  * P4 +3.65 (4/4), any-epoch controls +5.6 (4/4). Bilateral in every orofacial area, 4/4: contra tjM1 / ALM / SSp-m /
    SSp-n +3.8 / +2.7 / +5.3 / +5.0, ipsi +3.4 / +2.6 / +5.8 / +5.4. VISpm / RSPd are 3/4 (PS92 negative), n.s.
  * The no-basis within-session version is n.s.: +2.5, 3/4, PS95 -3.5.
  * **POST HOC (deviation 6):** on pre-cue windows with NO running the sign REVERSES: -7.2 [-10.3, -4.3], 4/4
    (-11.2 / -5.4 / -8.4 / -4.0). It is the same with no pre-cue lick as well (-7.3, 4/4).
  * **So the raised quit-period level of the stopped-trial traces is the running that comes with the quit. When the
    animal is still, the disengaged cortex sits BELOW its engaged, time-matched level.**
  * Trial-to-trial SD of the pre-cue level is up +9.3 [+6.3, +12.4], 4/4, and 0.01-0.05 Hz global power is up 4/4
    although evoked responses stop. Both are likely helped by running bouts; not read alone.
* **Before the quit:**
  * The cue response does not decline: prodrome -0.7 n.s., 2/4. The trajectory is flat until the quit, then drops at
    once.
  * Pre-cue level, running and face are unchanged in the 10 min before.
  * Per the rule: **cortex loses the cue response together with the behavioural stop, not before it.**
* No pupil measure exists, so arousal is not claimed.

### Synthesis (PROVISIONAL, n = 4)

After the stroke, animals stop working after less water than they collected without stopping before, while
thirstier (lower weight) and while earning water more slowly. Satiety at the old set-point is excluded.

* Failure runs at the deficit spout do not trigger the quit.
* No motor slowing precedes it: bouts shorten, the rhythm does not slow.
* Two of four animals quit while still earning at rates they worked through pre-stroke.
* The quit state is ACTIVE: the animal runs. The raised cortical level seen on stopped trials is that running; still
  disengaged cortex is lower than engaged.

The data favour a moved quitting threshold (motivation / malaise / state) over satiety, frustration or fatigue as
the terminator. It is back near pre chronically (P1 chronic n.s.).

**Licks per responded trial (descriptive figure):** chronic is higher than pre at every spout in both sources (cam4 LP
and DAQ contacts). Acute far contra is lowest (LP ~3 vs pre ~8.5).

## Deviations

**2026-10-31 (stage 2).** All were decided before the primary numbers were read, except 6.
1. **P4 construction.** The prereg made the no-extra-basis trace primary with RESTnl as reference. RESTnl's level
   exists only for the 600-s-detrended construction (`restw_strict_check` builds it on that trace), so subtracting it
   from a no-basis trace mixes constructions. PRIMARY therefore uses the detrended trace (RESTnl-consistent, = the
   stopped-trial-traces construction). The no-basis trace is the sensitivity, referenced within session to its own
   engaged pre-cue level of the first 30 min, as a difference against controls. The basis can absorb part of a
   10-min step, so the primary is if anything conservative.
2. RESTW sensitivity not run (RESTnl is the adopted reference); open.
3. ROI v8 has no `VISp_left` (blanked edge area), so **VISpm** (both hemispheres) replaces VISp as the visual
   reference. RSP = RSPd.
4. PS93_0813 has no imaging-input row (`KeyError` in the split table), so it is dropped from the Test 4 imaging
   measures only. It has no FaceRhythm video either (known).
5. Water volumes: DAQ deliveries equal the events.csv reward rows in 99/108 sessions, and the volume sum then equals
   the GUI `water_ul` to < 1 %. In the 9 mismatches (incl. PS92_0812 concat, PS95_0827) each delivery takes the
   session's mean nominal volume.
6. **POST HOC (exploratory, labelled on the figure):** `level_still` (pre-cue windows with every 1-s treadmill bin
   < 3 mm/s) and `level_quiet` (also no DAQ lick in the 2 s before the cue). Added AFTER seeing running up 4/4,
   because locomotion raises widefield dF/F globally. Not in the Holm family. It reverses the reading of P4. It
   needed a behaviour re-extract (`BEH_VERSION` 2: + `n_pre2`). All primary numbers were identical after it.
7. The 80-min horizon sensitivity counts a quit only if onset <= 80 min. It does not re-run the detector on the
   truncated record (`quit_point --horizon-min` does), so it is an approximation.
8. Bout measures use the post-cue licks inside the task window only (bout gap 0.5 s; deceleration on bouts >= 6 licks,
   last-third minus first-third median ILI). Prodrome ILI = mean of per-trial medians (trials >= 4 licks).
9. Trajectory bins (5 min) require each control to be engaged through the END of that bin. The quit period uses
   [quit, quit + 10 min] cut at session end, needing >= 5 min.
10. FaceRhythm "between trials" excludes [cue - 0.5, cue + 4 s] and lossy samples, in 10-s bins.
11. Lapse logistic: sklearn, effectively unpenalised (C = 1e6), standardised covariates, cells with >= 10 lapses and
    >= 10 non-lapses.
12. The giving-up rate is also computed for pre quits (drawn, descriptive). The pre working distribution = 5-min
    collected rate at every trial >= 5 min into a session and before any quit, all August pre sessions.
13. P3 / P4 figures pool epochs, so they use an animal-shape legend with a black pooled marker (no epoch colour).
    "+" marks a secondary cell whose 95 % CI excludes 0; "*" marks only a Holm-significant primary.

---

## ADDENDUM 10-31 (follow-up) — PRE-REGISTERED BEFORE ANY OF ITS ANALYSES WAS RUN

Priya's follow-up of 10-31 (asks 1-3) and her hypothesis for Test 4b. Stage 1 = this section only. Stage 2 appends
"Addendum outcome" and dated deviations below it. Outputs go to the same `disengagement/` folder (figdata stem
`disengagement_addendum`). All earlier rules still apply.

### A0. DATA-COLLECTION LIMIT: how pre-stroke sessions ended (Priya, 10-31)

Priya's account of the recordings:
* (a) Pre-stroke she often STOPPED a session by hand when the animal quit. A pre session with no detected quit may
  therefore END AT the quit, with the quit tail cut off; its "session end" is then effectively the quit point, not
  censoring.
* (b) PS92 and PS93 were very licky (many ENL violations) and were usually stopped FOR TIME, so their pre session
  ends are genuinely time-censored.
* (c) From cohort 2 on she lets animals run to the quit.

**Every result in this file that uses a pre "session end" must say which reading it uses.**

**END CLASSIFICATION RULE** (fixed now; it uses only the logs and the last trials, never an outcome measure):

| class | rule | count of 32 August pre |
|---|---|---|
| Q detected quit | `session_quit` finds the terminal non-recovering collapse | 11 (PS93 1, PS94 4, PS95 6) |
| W water limit | `water_limit_reached` in events.csv | 1 (PS95_0809) |
| T stopped by rule / for time (censored) | `max_duration_reached` or `target_trials_reached` with no detected quit; OR a manual stop (no rule event) with the reference spouts still responding (last 5 reference trials >= 0.8 responded, no trailing reference miss); PS92_0812 (crash + concat) is T | 16 |
| H likely hand-stopped AT a quit | manual stop, no detected quit, and the end is collapsing: last 5 reference trials <= 0.4 responded, OR >= 2 trailing reference misses, OR last 12 trials (all spouts) <= 0.5 responded | 2 (PS92_0809: ref 0.2, 2 trailing misses, all-spout 0.08; PS94_0807: ref 0.2, 4 trailing misses) |
| U unknown | manual stop, none of the above | 2 (PS92_0806: stopped at 83 min, ref 0.8, all-spout 0.58; PS95_0813: ref 0.6, 1 trailing miss) |

Fit with Priya's account: PS92 / PS93 pre ends are T except PS92_0809 (H) and PS92_0806 (U). That matches (b)
"stopped for time", with one PS92 session that looks hand-stopped at a collapse. The same rule is applied to post
sessions for symmetry (post mostly end by rule: target trials / max duration).

**Readings:**
* PRIMARY: Q and H are events (quit at quit onset / session end); T, W and U are censored.
* Bounds: (i) U also an event (end = quit); (ii) H also censored (end = censored).
* Both bounds are run for every addendum survival result.
* Uncertainty: the H / U split rests on the last 5-12 trials of a session and can misclassify a slow animal near its
  end. That is why U exists and both readings are reported.

**Consequence for earlier numbers:**
* P1's per-session value (water at quit-or-end) does not depend on the reading; its INTERPRETATION does.
* PS92 / PS93 pre values are time-censored LOWER BOUNDS (T), so their pre water / time is at least what is shown.
* Where pre is H or Q the value is the quit itself.

### A1. Licks per responded trial, normalised to each animal's own pre (ask 1)

* Per session x spout: log2(value / that animal x spout's August-pre mean), with both sources:
  * cam4 LP: `orofacial_cohort` `licks_per_resp`;
  * DAQ: contacts in the task window on responded trials before the quit.
* Canonical spout order; epochs. Per-animal table (animal x spout x epoch means) printed first.
* Descriptive except ONE pre-specified contrast, **P8: chronic far-contra log2 ratio, cam4 LP source**, chronic vs
  pre (`boot_delta`; each pre session's own log2 ratio, so pre is ~0 by construction).
* Rule: P8 > 0 (Holm in family A, >= 3/4 animals) means far-contra responded trials carry more licks chronically
  than the same animal's pre. DAQ is the sensitivity.
* Figure: BEHAVIOUR, log2 ratio by spout and epoch, sessions as open markers, pooled filled, 0 line, both sources
  side by side.

### A2. Time to quitting, and water, as survival (ask 2)

* Time = minutes from the session's first cue to quit onset (Q / H events) or to session end (censored).
* Kaplan-Meier per animal x epoch, in MINUTES and in collected WATER, under the primary reading. The two bounds are
  drawn as dashed curves. Censored ends are ticked and coloured by class (T / W / U).
* **P5 (primary, family A): restricted engaged time min(time to quit-or-end, 90 min), early post (acute + subacute)
  minus pre, `boot_delta`.**
  * Why 90 min: post sessions end by rule at ~95-125 min, so a 90-min restriction compares equal observation windows
    and removes the pre long-session policy confound that the unrestricted value carries.
  * Every session except two is observed to 90 min; the two censored before 90 (PS92_0806 U at 83 min, PS94_0807 H
    at 83 min) enter as their observed value under the primary reading and as 90 min under bound (ii).
  * Secondary cells: acute, subacute, chronic. Unrestricted time to quit-or-end is descriptive, with PS92 / PS93 pre
    marked as time-censored lower bounds on the figure.
  * Rule: P5 < 0 (Holm, >= 3/4) means post animals stop working sooner within a common 90-min window.
* P1 (water) is NOT re-tested. It is re-reported under the three readings and with the exact 80-min horizon (A3iii).
* **Family A (Holm): {P5, P6, P8}.** P1 stays in the original family.

### A3. Open items (ask 3)

**(i) Running-regressed cortical level — P6 (primary, family A).**
* Within each session: per-trial pre-cue level (detrended construction, minus RESTnl, each area) regressed on running
  over the 2 s pre-cue window (mean speed and running fraction) plus the mean speed of the preceding 10 s, with a
  QUIT-PERIOD INDICATOR in the same regression. The running coefficients are then estimated within state rather than
  absorbing the state difference. Fit on all trials of the session.
* Residual level = level minus the fitted running terms only. Then quit period [quit, +10 min] minus time-matched
  engaged controls (same animal, same epoch), all quits pooled, exactly as P4. Global = `cortex`; areas secondary.
* Rule:
  * P6 < 0 (Holm, >= 3/4) confirms the post-hoc "still disengaged cortex sits below engaged".
  * P6 > 0 means the still-trial result was a selection artefact and a raised level survives running removal.
  * Null means no level difference once running is removed.
* Sensitivity: the regression fitted on engaged trials only.

**(ii) RESTW-reference sensitivity of P4, P6 and the still-trial measures.**
* RESTW level per area from `iti_licks/precue_restw/<label>.npz` (`restw`, `restw_ok`; 107 files).
* Descriptive: same sign and >= 3/4 means reference-robust.

**(iii) Exact 80-min horizon.**
* Re-detect quits with `precue_engagement_states.engagement_gate` on each record CUT at 80 min, as `quit_point
  --horizon-min 80` does. Sessions shorter than 80 min are dropped.
* Water at quit-or-80 and time at quit-or-80, early post minus pre. Sensitivity only; replaces the approximation of
  deviation 7.

### A4. TEST 4b — is quit-period running SELF-ACTIVATION to get back to licking? (Priya's hypothesis)

The alternative is that running is a generic disengaged / exploratory state.
* Running bouts: 1-s bins with mean speed > 3 mm/s, >= 2 s consecutive (gaps < 1 s merged).
* Still periods: >= 2 s with every bin < 3 mm/s.
* All within the quit period (quit onset to session end), unless stated.
* Pooled over epochs (all quits; pre quits = PS93 1, PS94 4, PS95 6, PS92 0), with a post-only cell reported beside.

**(1) RE-ENGAGEMENT — P7a.**
* For each running bout and each still period in a quit period, take the NEXT trial whose cue falls after the period
  ends, and ask whether it was responded (DAQ lick in the 3.5 s window).
* Still periods are matched to bouts: same session, the still period nearest in time with duration >= the bout's.
* Per session: P(responded | after running) minus P(responded | after still); `boot_delta(run, still)` paired within
  animal; per-animal table first.
* Secondary: the same with "any DAQ lick within 30 s of the period end", and a discrete-time hazard of the first
  responded trial after the period.

**(2) GAIN — P7b.**
* Trials whose preceding 10 s (before the cue) contained >= 1 s of running vs trials whose preceding 10 s were fully
  still.
* Response probability (P7b), within QUIT periods, paired within animal.
* Secondary: the same within ENGAGED periods (so the contrast is not just state); first-lick latency and the global /
  orofacial cue response on the same split.

**(3) SPECIFICITY — P7c.**
* Per session, the z-scored (within-session SD) pre-cue level of each area regressed on pre-cue running speed (all
  trials).
* Orofacial excess = mean beta over ALM, tjM1, SSp-m, SSp-n (both hemispheres) minus mean beta over VISpm and RSPd
  (both hemispheres).
* P7c: excess > 0 over sessions, `boot_delta(orofacial betas, reference betas)` paired within animal.
* Secondary: is the excess larger post than pre (`boot_delta`, early post vs pre; descriptive)?

**(4) STROKE-SPECIFIC — descriptive only.**
* Running fraction, bout rate and bout duration in quit periods: post quits vs the only pre quits (PS94 n = 4,
  PS95 n = 6; PS93 n = 1, PS92 none).
* At most 2-3 animals, so it cannot pass the 3/4 rule.

**Family B (Holm, its own): {P7a, P7b, P7c}.**

**Decision rule (Priya's, fixed in advance):**
* SELF-ACTIVATION requires (P7a OR P7b) Holm-significant AND positive in >= 3/4 animals, AND P7c Holm-significant
  and positive (orofacial excess).
* Otherwise GENERIC disengagement. Two named partial outcomes:
  * "running precedes re-engagement without an orofacial-specific activation": (1) or (2) passes, (3) does not;
  * "running activates orofacial cortex preferentially but does not bring licking back": (3) passes, (1) and (2) do
    not.
* Limits:
  * Quit-period responded trials are rare, so P7a / P7b may be thin; per-animal counts of periods and trials are
    printed first, and a session needs >= 3 run and >= 3 still periods.
  * Running and re-engagement can share a cause (a general arousal fluctuation); this design cannot separate
    "running causes re-engagement" from "both follow arousal". No pupil.

### A5. Figures planned (addendum)
* BEHAVIOUR: normalised licks per responded trial (A1).
* BEHAVIOUR: KM in minutes and in water per animal, primary reading + bounds (A2).
* BEHAVIOUR: restricted engaged time by epoch + deltas (P5).
* NEURAL: running-regressed quit-period level, global + areas, with RESTnl and RESTW versions (P6, A3ii).
* BEHAVIOUR: 80-min horizon re-detection (A3iii).
* BEHAVIOUR + NEURAL: Test 4b (re-engagement after run vs still; gain; orofacial-vs-reference running betas; pre vs
  post quit running).
* Figure defaults as before.

### A6. Decisions for Priya
1. The END CLASSIFICATION rule (A0) and its thresholds, especially whether PS92_0809 (H) and PS92_0806 / PS95_0813 (U)
   are read correctly. Default: as above, bounds always shown.
2. Restriction horizon for P5 = 90 min. Default 90 (the latest time nearly every session is observed).
3. Test 4b pooled over epochs (all quits) with a post-only cell beside it. Default pooled; pre has only PS94 / PS95
   quits.
