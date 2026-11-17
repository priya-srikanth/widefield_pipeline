# Within-session decline of aim and latency vs quitting (cohort1, 2026-11-07)

Branch `analysis/within-session-decline-quitting-20261107`. Script `scripts/within_session_decline.py`
(extract / analyze / plot). Outputs `M:/MICROSCOPE/Priya/Widefield/within_session_decline/` (`trials.csv.gz`,
`tables/`, `figures/`). Cohort1 only (PS92-95), August-only pre, curated sessions. **Exploratory, n = 4 animals.**

Priya (11-07), on the state-space observation that first-lick latency slows and far-contra landing drifts
ipsilesionally within post-stroke sessions: *"this is important; i wonder how this relates to the animals 'quitting'
after stroke. can we look into this"*.

## 0. Decisions (Priya, 2026-11-08, relayed by the coordinating session)

1. **Slope unit: per HOUR primary, per 100 TRIALS the robustness row; per normalized session dropped.**
   Rationale: post-stroke sessions are shorter (subacute working span median 98 min vs pre 122), so a per-normalized-
   session slope biases against the stroke effect; per trial mixes the drift with the trial rate, which itself changes
   after the stroke; the pacing tests (R7) say the drift is clock / state-like. Priya's addendum (11-08): *"trial
   lengths can vary post-stroke though (enl licking changes)"* -> let the data choose: the unit whose slope is
   independent of the session's trial rate becomes primary. **Test and outcome (section 3, R0): neither unit's slope
   depends on trial rate for any measure (all within-animal |r| <= 0.31, shuffle p >= 0.12); the one duration
   dependence (latency drift vs the pre-cue / ENL wait, early post) has the SAME negative sign under both units, which
   neither pacing model predicts, so it is a session-state covariate, not a unit artifact. The data do not
   discriminate; per hour stays primary on the rationale above, per trial is the robustness row and agrees in sign
   in every animal for the headline cells.** The normalized-order slope is still computed only to check the
   reproduction of the state-space numbers.
2. **Wording (adopted):** "the far landing pattern drifts ipsilesionally within the session, largest at far contra"
   -- not "the impaired direction degrades".
3. **Adopted:** the latency rise before quits is a general sign of quitting (present pre-stroke), not a stroke effect;
   the aim drift is not a quit prodrome.
4. **Satiety vs time:** between-session / between-animal comparisons only (R7); no manipulation planned.
5. **Placement:** a behavior-page subsection next to the disengagement tests (S28). The behavior page belongs to
   another window: this branch only PREPARES the subsection, `docs/results_pages/within_session_decline/section.html`
   (not published; the page owner inserts it after the S28 block and uploads the figures).

## 1. What was already established (not re-tested here)

| claim | number | source |
|---|---|---|
| Acute orofacial motor deficit in the lick RHYTHM (median ILI) | +12.5 ms [+4.6, +19.9], 4/4, flat across the session (already in Q1), recovered by subacute | STATUS_2026-09-20, DECISIONS 09-20 |
| ILI does not rise within a session | 0 / 16 epoch x position cells | same |
| BUT "no within-session motor fatigue" was revised the same day | within-bout deceleration +15-25 ms at every epoch incl. chronic (4/4); it steepens across the session acutely (+10.1 [+0.9, +19.1]); bouts shorten across the session subacute / chronic near; the decline in licks / trial is paced about half by licks and half by the clock | DECISIONS "THE WITHIN-SESSION DECLINE IS PARTLY FATIGUE AFTER ALL" (09-20), `lick_bout_structure` |
| Within-session ENGAGEMENT decline (licks / trial), not the quit tail | Q5-Q1 pre -3.8, acute -8.3, subacute -10.7, chronic -3.5; steepening vs own pre significant SUBACUTE (-6.8 near / -5.8 far) | STATUS_2026-09-20 |
| The QUIT is time-paced, not lick-budget-paced | log(quit licks) on log(early rate) +0.72 [+0.36, +1.06]; log(quit time) -0.13 | `quit_point` |
| Quit prodrome in LICK RATE | -18.7 licks/min [-22.9, -13.7] in the 10 min before, vs time-matched controls | `quit_prodrome` |
| Animals quit sooner early post | KM restricted mean -15.1 min [-25.4, -5.6] 4/4, -77 trials 4/4; chronic ~0 | disengagement tests (S28) addendum |
| Satiety at the quit | collected water at quit-or-end early post -290 ul (Holm .001, 4/4) -> quits at LESS water; on equal-length (80 min) data -105 ul [-225, +10] 3/4 -> satiety not rejected; weight -5 % | disengagement P1 + addendum |
| Fatigue at the quit | P3 n.s.: bouts shorten before quits, tongue does not slow (within-bout ILI) | disengagement P3 |
| Frustration (far-contra block lapses) | +0.037 n.s. | disengagement P2 |
| Quit state in cortex | no prodrome in cortex / face / running; cue response falls AT the quit | disengagement P4 |
| Latency tracks aim within animals | r -0.72 to -0.92 | revup_behaviour (FINDINGS) |

**What was NOT established:** (i) whether the 3-D AIM (lateral landing) or the first-lick LATENCY degrade across a
session -- every earlier within-session test used DAQ lick counts / intervals, never where the tongue lands or how
soon the first lick comes; (ii) whether aim / latency show a prodrome before the quit (the S28 prodrome tests looked at
lick rate, within-bout ILI, cortex, face, running); (iii) whether a session's aim drift predicts how early it quits;
(iv) whether the drift is specific to the impaired (far-contra) direction; (v) a time vs effort vs water separation for
aim / latency (the 09-20 pacing test was for licks / trial and for the quit).

**Reconciling with "no within-session motor fatigue".** That line (CLAUDE.md pointer to STATUS_2026-09-20) is stale:
it was about the lick-rhythm PERIOD (median ILI), which a brainstem CPG holds fixed, and DECISIONS revised it the same
day. Aim (where the first lick lands) and latency (how soon the first lick starts) are different measures from ILI:
a slower start and a drifting aim can coexist with a fixed cycle period. Nothing in the earlier work contradicts the
new observation; the question is what paces it.

## 2. What was done here

`scripts/within_session_decline.py` (extract ~14 min fanned over 6 workers on the shared box; analyze / plot ~1 min).
Per trial (56,979 trials, 107 curated sessions with video; PS93 pre has 7, PS93_0813 has no video):

* trials, DAQ cue times, camera classes = `trial_classes.session_trials` / `daq_cue_seconds`; **working** = inside the
  camera working period (`camera_period_trials`, to the last camera lick); **quit** = `trial_classes.camera_quit`.
* first lick = 3-D `licks.csv.gz` lick_idx 0: lateral landing `side_ml` (mm, - = ipsilesional), peak time after the cue
  (latency, ms), contact, peak speed, reach; camera licks / trial from the trial table.
* aim offset `dside` = side_ml minus the animal's August-pre mean at that spout; latency / contact / speed likewise as
  deviations from the animal's pre spout mean.
* cumulative collected water (ul), camera licks and far-contra trials BEFORE each trial, from the disengagement-tests
  cache (`disengagement/sessions/beh_<label>.pkl`; DAQ responded trials x reward_ul in force; cache version 2 = the
  pre-11-05 DAQ lick detector, 0.6 % of licks differ; t0 checked = 0.000 s offset in every session).

Analyses: Q1 session slopes (per hour of elapsed time = primary; per 100 trials of the session = robustness row,
decision 1; multi-spout measures demeaned within spout; >= 15 trials, >= 8 for one spout; "far" = the three far spouts
pooled), change vs own pre with `boot_delta`; slope-unit test (11-08, `times` stage -> `trial_times.csv.gz`: per
session the trial rate = working trials per working hour, the median cue -> next cue cycle, and its parts: pre-cue
wait strobe -> cue incl. ENL-lick resets, cue -> trial end, the rest = ITI; within-animal Spearman of each slope with
each, shuffle null, by epoch group; pooled elasticity, descriptive); the slope
level with `boot_ci`; the same with the last 15 min before the quit removed. Q2 prodrome: per quitting session the
change from [-30, -10) to [-10, 0) min before the camera quit, minus the same clock windows in the same animal's
other sessions of the epoch still working then (time-matched, as `quit_prodrome`); quit-aligned 5-min bins. Q3 time
vs effort vs water: per animal x epoch, session x spout fixed effects, R2 of each predictor alone and unique R2; plus
session-level instruments (early water accrual rate, drop size, early lick rate) vs the per-hour drift, within-animal
Spearman with a within-animal shuffle (5,000). Q4 per-hour drift vs minutes to quit-or-end and vs working-trial count
(within-animal Spearman + shuffle). Q5 per-spout per-hour slopes vs own pre; within-animal r with the far-contra aim
recovery axis (`aim_deficit_sessions.csv` farcontra_first).

Reproduction check: per-session normalized-order slopes correlate with the state-space CSV at r = 0.96 (latency),
0.99 (far-contra aim), 1.00 (all-spout aim), 0.99 (contact), n = 107 sessions -- same observation, now on the repo's
working-period / 3-D definitions.

Outputs (`M:/MICROSCOPE/Priya/Widefield/within_session_decline/`): `trials.csv.gz`; `tables/session_slopes.csv`,
`session_slopes_cut15.csv`, `q1_slope_levels.csv`, `contrasts_vs_pre.csv`, `q2_prodrome_sessions.csv`,
`q2_prodrome_levels.csv`, `q2_quit_aligned_bins.csv`, `q3_pacing_fe.csv`, `q3_predictor_correlations.csv`,
`q3_session_accrual.csv` (incl. the per-session trial rate / durations), `unit_rate_assoc.csv`, `unit_elasticity.csv`,
`unit_durations_vs_pre.csv`, `q5_spout_slopes.csv`, `../trial_times.csv.gz`, `within_animal_assoc.csv`, `session_table.csv`, `analyze_log.txt`
(every per-animal table, printed before each bootstrap); figures `wsd_fig1_slopes_per_hour`, `wsd_fig1b_slopes_per_trial`, `wsd_fig2_quit_aligned`,
`wsd_fig3_pacing`, `wsd_fig4_drift_vs_quit`, `wsd_fig5_per_spout` (.png + svg/).

Camera quits (quit / sessions): PS92 pre 0/8, acute 4/5, subacute 0/2, chronic 0/12; PS93 1/7, 3/4, 1/3, 1/12;
PS94 5/8, 5/6, 4/5, 4/8; PS95 4/8, 0/1, 3/7, 5/11.

## 3. Results (n = 4; stars = boot_delta 95 % CI excludes 0; per-animal in the tables)

**R0. Slope unit (decision 1; tables `unit_rate_assoc.csv`, `unit_elasticity.csv`, `unit_durations_vs_pre.csv`).**
* Trial length does change post-stroke, through the PRE-CUE (ENL) wait only: cue -> next cue cycle acute -1.95 s
  [-5.19, -0.06] (3/4 shorter; PS92 -6.6 s, its pre wait 10.6 s -> 3.7 s), chronic -1.30 s (4/4); pre-cue wait acute
  -2.03 s [-5.29, -0.01] (3/4), chronic -1.18 (3/4); cue -> trial end -0.05 s (4/4, tiny); ITI +0.13 / -0.03 / -0.07
  n.s. Trial rate per working hour: acute -4, subacute -24, chronic +11 trials/h, all n.s. (between-animal range
  165-350 trials/h).
* Slope vs trial rate, within animal (mean r over animals, shuffle p): latency per hour -0.15 / -0.29 / -0.04 (pre /
  early post / chronic), per trial -0.15 / -0.31 / -0.04; far-contra aim per hour -0.01 / +0.01 / +0.02, per trial
  +0.11 / 0.00 / 0.00; all-far aim per hour +0.14 / -0.09 / +0.14, per trial +0.03 / -0.09 / +0.08. None
  significant, neither unit better.
* Duration parts: the only dependence is latency drift vs the cycle / pre-cue wait EARLY POST: per hour r -0.60
  (p .001) with the cycle, -0.45 (p .013) with the pre-cue wait; per trial -0.51 (p .006) and -0.36 (p .058); not
  with cue -> trial end or the ITI, and not pre or chronic. Clock pacing predicts per-hour r = 0 and per-trial r > 0;
  trial pacing predicts per-hour r < 0 and per-trial r = 0; both units are negative here, so sessions with long
  pre-cue waits (more waiting / ENL licking) simply drift less -- a session state, not the unit. Aim drift: no
  dependence on any part (|r| <= 0.26).
* Pooled elasticities are unstable (|e| 0.4-62; the aim mean slopes are near 0 and within-animal rate varies only
  +-10-20 %), so not used.
* Per-trial vs per-hour, headline cells, per animal (all same sign): far-contra aim subacute per hour -0.20 mm/h
  (PS92 -0.10, PS93 -0.22, PS94 -0.16, PS95 -0.33) vs per trial -0.074 mm / 100 trials [-0.119, -0.027] 4/4
  (-0.073, -0.087, -0.043, -0.092); latency acute +126 ms/h (+45, +263, +157, +39) vs +40 ms / 100 trials [+4, +80]
  4/4 (+5, +93, +48, +12); latency subacute +57 ms/h 4/4 n.s. vs +26 / 100 trials 3/4 n.s. (PS94 -0.3); all-far aim
  subacute -0.097 mm/h [-0.200, -0.002] 4/4 vs -0.041 / 100 trials [-0.088, -0.000] 4/4; camera licks / trial subacute
  -3.5 /h vs -1.4 / 100 trials, both 4/4. Fig 1b = Fig 1 per 100 trials.

**R1. Latency slows within the session in every epoch, including pre, and more after the stroke (Fig 1).** Slope
level per hour: pre +61 ms/h [+24, +114] (4/4 > 0), acute +205 [+130, +285] (4/4), subacute +113 [+58, +174] (4/4),
chronic +37 [-1, +75] (3/4). Change vs own pre: **acute +126 ms/h [+37, +230], 4/4** (PS92 +45, PS93 +263, PS94 +157,
PS95 +39); subacute +57 [-17, +130], 4/4 positive, CI includes 0; chronic -22 n.s. Unchanged with the last 15 min
before the quit removed (acute +125 [+41, +232] 4/4; subacute +69 [-8, +145] 4/4).

**R2. Far-contra aim drifts ipsilesionally within subacute sessions in 4/4 (Fig 1).** Slope level subacute -0.32 mm/h
[-0.50, -0.11], 4/4 < 0; change vs pre **-0.20 mm/h [-0.36, -0.07], 4/4** (PS92 -0.10, PS93 -0.22, PS94 -0.16, PS95
-0.33). Acute -0.15 n.s. (PS93 -0.55, PS95 -0.46; PS92 +0.07, PS94 +0.33); chronic -0.04 n.s. (3/4 negative). Holds
with the last 15 min before the quit removed (subacute -0.21 [-0.37, -0.07] 4/4). Other spouts pooled: n.s. at every
epoch (-0.04 to -0.07).

**R3. Not specific to the impaired direction (Fig 5).** Per spout, aim drift change vs pre: subacute far contra -0.20
(4/4), **far ipsi -0.16 [-0.33, -0.01] 4/4**, close mid -0.12 (3/4, n.s.), close contra -0.10 (3/4, n.s.); acute
close contra -0.09 [-0.19, -0.01] 4/4; **chronic far ipsi -0.17 [-0.30, -0.03] 4/4**; far mid ~0; the three far
spouts pooled, subacute -0.097 mm/h [-0.200, -0.002] 4/4. Wording (decision 2): **the far landing pattern drifts
ipsilesionally within the session, largest at far contra.** Latency drift is also broad: acute far contra +188 [+12, +394] 4/4, far ipsi +210 [+57, +376] 4/4;
subacute close ipsi +55 [+1, +115] 4/4.

**R4. Contact and vigor barely move (Fig 1).** First-lick contact slope: subacute -0.026 /h [-0.057, +0.004], 4/4
negative (level -0.026 [-0.045, -0.007]) -- small; acute / chronic ~0. First-lick peak speed: no change vs pre at any
epoch (acute -1.5, subacute +1.7, chronic -2.8 mm/s per h, all n.s.; pre itself speeds up +2.5 mm/s/h 4/4). Reach:
acute -0.14 mm/h [-0.32, +0.01] 3/4, subacute -0.11 n.s. Camera licks / trial fall within every session (pre -2.0 /h,
4/4) and faster subacute: **-3.5 /h [-4.8, -2.5] 4/4** vs pre -- the 09-20 licks / trial decline, reproduced on
camera licks. So the latency / aim drift is not accompanied by a slowing or weakening tongue.

**R5. The aim drift is not a quit prodrome; latency is, in every epoch (Fig 2).** Change from [-30, -10) to [-10, 0)
min before the camera quit, minus time-matched windows in the same animal's still-working sessions: latency pre
+150 ms [+90, +214] (3/3 animals with pre quits: PS93 +168, PS94 +190, PS95 +90), chronic +141 [+14, +427] (3/3),
acute +75 (2 animals), subacute -85 (1 animal). Far-contra aim in the same windows: no ipsilesional prodrome (chronic
+0.27 mm, i.e. contralesional; other cells 1-2 animals). Contact falls before the quit pre (-0.09, 3/3) and chronic
(-0.09, 3/3). Licks / trial fall toward the quit in every epoch (the known lick-rate prodrome). Thin: acute and
subacute prodrome cells have 1-2 animals with both a quit >= 35 min in and a time-matched control.

**R6. A session's drift does not predict how early it quits (Fig 4).** Within-animal Spearman, per-hour drift vs
minutes to quit-or-end: far-contra aim pre -0.11 (p .59), early post (acute + subacute) +0.13 (p .48), chronic +0.32
(p .048; i.e. more ipsilesional drift -> earlier end, 4/4 positive but small); latency drift pre -0.30 (p .12), early
post -0.22 (p .25), chronic -0.29 (p .07) -- negative on average in every epoch including pre, but not consistent
across animals (pre PS93 +0.50; chronic PS92 +0.26, PS94 +0.02), so at most a weak, non-stroke-specific tendency. Quitting sessions only: nothing (2-3 animals). Drift vs working-trial count: n.s.

**R7. Time vs effort vs water cannot be separated within a session (Fig 3).** After session x spout fixed effects
the four clocks (elapsed time, cumulative camera licks, cumulative collected water, cumulative far-contra trials)
correlate r = 0.96-1.00 in every animal x epoch. Each alone explains the same small share (latency 0.1-4 % of
within-session variance; far-contra aim up to 13 % in PS93 acute); unique R2 is < 3 % for every predictor except a
far-contra-trial count term in a few cells (PS95 pre 7.5 %), and no predictor wins consistently. Between sessions,
the per-hour drift does NOT scale with water: early water accrual rate vs far-contra aim drift early post r -0.16
(p .40), vs latency drift -0.19 (p .32); drop size vs latency drift early post **-0.39 (p .04, 4/4 negative)** --
bigger drops, LESS latency drift, the opposite of a satiety pacing (weak instrument: drops 1.8-2.1 ul post-stroke).
Between animals: PS94 / PS95 collect water ~2.5x faster than PS92 (pre ~11-12 vs ~4.3 ul/min) yet do not drift
faster per hour (pre latency drift PS92 +107, PS93 +38, PS94 +48, PS95 +48 ms/h; acute PS95 the slowest, +87).
So the drift is paced like a clock (or a fixed per-session state change), not like accumulated water; a lick-paced
(effort) component cannot be excluded because licks and time are inseparable here, and early lick rate vs drift is
null too (early post r +0.25 aim, -0.23 latency, both n.s.).

**R8. The within-session aim drift is a different quantity from the session's aim level.** Within animal, a
session's far-contra aim (aim_deficit_sessions farcontra_first, deg) correlates r = 0.94 (early post) / 0.84
(chronic) with our far-contra mean dside (mm) -- same measure, sanity check -- but NOT with its within-session drift
(r -0.16 early post, +0.05 chronic, n.s.) nor with the latency drift. Sessions with a worse far-contra aim do not
drift more within the session.

## 4. Interpretation

* **Reconciliation with "no within-session motor fatigue":** that finding was about the lick-rhythm period (median
  ILI), and was already revised on 09-20. The new measures agree with the revised picture: what declines within a
  session is the START (latency) and, post-stroke, the AIM of the first lick, together with bout length / licks per
  trial; the tongue's cycle period, peak speed and contact barely move. No contradiction.
* **Relation to quitting:** the aim drift and the quit are separate. The aim drift is present well away from the quit
  (same size with the last 15 min removed), shows no prodrome, and does not predict when the session ends (except a
  weak chronic tendency). Latency is the one first-lick measure that rises steeply in the last 10 min before a quit,
  and it does so pre-stroke as much as after -- a general sign of disengagement, not of the stroke.
* **Fatigue vs satiety vs time:** within a session the three cannot be separated (collinear). Across sessions and
  animals the drift does not scale with water intake rate or drop size, which argues against satiety PACING it; the
  quit itself was already shown to be clock-paced. Peak speed / contact do not decline, which argues against a
  peripheral motor fatigue of the tongue. Best description: a time-dependent (or per-session state) decline in
  readiness to start (latency) and in lateral calibration (aim), amplified after the stroke (latency acute,
  ipsilesional aim drift subacute), and returning to the pre-stroke slope chronically. Central fatigue / effort
  cost cannot be ruled out because effort (licks) is collinear with time.
* **Impaired direction:** the ipsilesional drift is largest at far contra but appears at far ipsi too; read it as a
  drift of the whole landing pattern toward the ipsilesional side (a calibration that relaxes back toward the
  lesion-biased default as the session goes on), not as degradation specific to the impaired direction.

## 5. Caveats

* n = 4 animals; PS95 acute = 1 session, PS92 subacute = 2. Bootstrap p is crude (`boot_p`); no Holm across this
  exploratory family -- R1 acute and R2 subacute survive any reasonable reading (4/4 and CI clear); R3's far-ipsi
  and acute close-contra cells, R5, R6 chronic and R7's drop-size r are each one test among many.
* Prodrome cells are thin (pre quits only in PS93 1 / PS94 / PS95; acute / subacute 1-2 animals with both a quit and
  a time-matched control).
* Latency = 3-D first-lick PEAK after the cue (includes non-contact licks); the DAQ-contact latency would read a
  missed far-contra lick as a long latency.
* Water = collected nominal drops (delivered on responded trials); consumption is not measured. Session water cache
  is version 2 (old DAQ detector).
* Drop size is set by the experimenter and may respond to the animal's performance (endogenous); its range is narrow.
* Per-hour slopes assume linear drift; the quit-aligned curves suggest latency accelerates near the quit.

## 6. Open questions

* Does the ipsilesional drift also appear in LATER licks of a trial (the chronic later-lick ipsilesional reading in
  FINDINGS) and does it reset after the ITI -- i.e. is it a slow within-session state or trial-history (repeated
  far-contra blocks)? A block-position analysis (first vs last trials of each far-contra block) would split them.
* Neural counterpart: the state-space branch's within-session drift toward the pre state -- is it the same sessions /
  animals as R2 (PS94 subacute)? Join `session_slopes.csv` to the state-space per-session table.
* A satiety manipulation (pre-session water, or smaller drops on some days) is the only clean way to separate
  satiety from time.
