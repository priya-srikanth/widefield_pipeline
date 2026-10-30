# Learned, context-specific compensation (behaviour only): design written BEFORE any post-stroke number (2026-10-30)

Working idea (Priya 10-30; RECOVERY_MECHANISMS_LIT_20261030.md section 1a, "learned, context-specific compensation" and
"Why do cued licks recover but ITI licks do not?"): recovery at far contra is a RE-AIMED FEED-FORWARD LAUNCH (FINDINGS 4
item 21: launch error -25.7 deg acute -> -3.2 n.s. chronic, no within-lick correction), learned for CUED licks, while the
deficit persists in self-initiated ITI licks (so far a CORTICAL readout: ITI decoder offset, docked-spout set). Three
behaviour-only tests, no imaging.
Code: `scripts/learned_compensation.py` (`run` / `summarise` / `plot`, `--plot-only`). Outputs only under
`M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/learned_compensation/`.
Looked at before this was written: column names / file structure, the session list with epochs and days, the spout block
structure of two sessions (blocks of 4-8 trials of one spout, ~10-12 far-contra blocks per session, random block order),
the lick-context definitions (below) and the join coverage of one PRE session (PS92_0807: 99 % of ITI licks matched
to `iti_dock_audit.csv`). No post-stroke outcome number.

## 0. Data (fixed)
* Sessions: the 107 sessions of `within_lick_correction/licks_scored.csv.gz` (= `analysis_kit.curated_ids()` passing
  the pose3d gates; PS92-95 only, cohort2 PS118 / 120-122 NEVER; **pre = AUGUST pre only**, June excluded everywhere,
  incl. references). Epochs `wfield_local.epochs.epoch_of` (animal-specific); day = `epochs.days_since_stroke`.
* Licks: `licks_scored.csv.gz` (one row per post-cue / ITI lick with a 3-D track; sign already forced so + =
  CONTRALESIONAL = mouse right). LAUNCH angle theta_L (first 12 ms, SG-smoothed), ENDPOINT angle theta_E (extension
  peak); e_L / e_E = minus the animal's August-pre CONTACT post-cue mean at that spout (within_lick_correction prereg 2).
  A launch score needs `onset_ok` and finite theta_L.
* Trials: `imaging_encoding/qc/trial_attempts.csv` (`trial_id`, `pos`, `cls`, `canonical` = the reference engagement
  gate, `stop_ms` from the cue). ENGAGED trial = `canonical`. HIT = `cls == "hit"` (a DAQ contact). Trial order =
  `trial_id`. Spout BLOCK = a run of consecutive trials (all trials, engaged or not) at one spout; `first_in_block` = the
  trial is the first of its block (a spout switch just happened).
* Lick contexts (`orofacial_outputs.lick_context`): post_cue = cue .. DAQ trial stop; pre_cue = spout in place before
  the cue (NOT used: it has a target and was not extracted in 3-D by within_lick_correction); **ITI = the spout away**
  (after the stop: retraction, docked, outbound move to the next target, before its arrival).
* ITI cycle: each ITI lick (licks_all `daq_on_s` via `lick_uid`) is joined (|dt| < 2 ms) to
  `iti_licks/iti_dock_audit.csv` (per-lick `phase`, `since_stop`; log-clock events, reconstruction for ~0.6 % of
  licks). PHASES: RETRACT (`pre_retract` + `retract`: spout moving from the target back to the dock, ~0-0.9 s after the
  stop), DOCKED (`docked`: spout parked at the dock, stationary, no target; ~0.9-2.3 s; Priya's primary ITI set), OUTBOUND
  (spout moving to the next target, ~2.3-4 s). Other phases and unmatched licks are dropped (counted). PREVIOUS TRIAL = the
  last trial whose DAQ stop (cue + `stop_ms`, cue on the DAQ clock from `windows.npz`) is <= the lick onset; its spout =
  `prev_pos`, its engagement / outcome are the trial's. NEXT TRIAL = the one after it (`next_pos`). The dock is ONE
  location for all spouts; "relative to the prior spout" therefore means relative to where the spout WAS, now vacated.
  Note: the ITI context covers only ~0-4 s after the stop (the spout arrives at the next target ~3 s after the stop;
  licks after that are pre_cue), so the example bins "0-2 / 2-5 / > 5 s" are replaced by the task's own cycle phases
  (+ descriptive 0.5 s since-stop bins up to 4 s).
* ITI licks enter only if the PREVIOUS trial is engaged (`canonical`).

## T1. Learning signature (within-session gain, overnight loss) -- PRIMARY
* Trials: engaged FAR-CONTRA trials. Readout per trial: e_L of the FIRST post-cue lick (lick_idx 0; the feed-forward launch
  before any feedback in that trial; trial dropped if lick 0 is not launch-scorable -- no substitution). Hit rate: all
  engaged far-contra trials, hit = 1 / 0.
* Within-session position x = rank of the trial among the session's scored far-contra trials / (n - 1) in [0, 1]; THIRDS
  = `np.array_split` of that order into 3 (hit: same on the engaged far-contra trials). A session cell needs >= 24
  scored trials (>= 8 per third).
* Per session: **W = mean(late third) - mean(early third)** (deg; + = later trials more contralesional; the deficit is
  negative, so + = improvement), slope b = OLS of e_L on x (deg per session; secondary), session mean M.
* Overnight pairs: consecutive sessions of one animal with a CALENDAR gap of 1 day (primary; gap <= 2 days secondary), both
  August pre (control) or both in the post window. **O = mean(early third, day d+1) - mean(late third, day d)** (- =
  overnight loss); **S = M(d+1) - M(d)** (day step).
* WINDOW: post days 1-14 inclusive (acute + subacute for every animal; Priya's "acute / subacute window"). Secondary:
  the same per epoch (acute, subacute, chronic; chronic = late control where little is left to learn).
* Readouts: W_eL, O_eL (PRIMARY), W_hit, O_hit (PRIMARY, hit rate), S_eL, S_hit, b (secondary); TARGET-SPECIFICITY
  control: W at the OTHER five spouts (first-lick e_L, same thirds) and W_fc - W_o5 (a general warm-up raises both).
  Secondary readouts: first-lick e_E; mean e_L of the trial's first 3 scorable licks; non-first-in-block trials only.
* Predictions: LEARNING (within-session practice, partly lost overnight): W_eL > pre AND O_eL < pre in the window, with
  W_fc - W_o5 > 0 (target-specific); hit rate the same direction. SPONTANEOUS BIOLOGICAL RECOVERY: W and O as pre, S > 0
  (improvement arrives between days). WARM-UP decrement (alternative that mimics learning): W and O changed equally at
  other spouts (W_fc - W_o5 ~ 0). FATIGUE / satiety: W < 0 (handled by the engagement gate; the pre W is the control).
* Stats: per-animal table (animal x period: mean W, O, S, n sessions / pairs) printed and saved FIRST (rule 8). Then
  `analysis_kit.boot_delta(post-window values, pre values)` per animal dicts (seed 0; units = sessions for W / S / b,
  overnight PAIRS for O). **A change = 95 % interval excludes 0 AND >= 3 / 4 animals with the same sign.** Holm over the
  4 primary tests (W_eL, O_eL, W_hit, O_hit; `boot_p`). Learning signature declared only if W_eL AND O_eL are both
  changes in the predicted directions (W up, O down); "partial" if one of them. Mixed model (trial level, pre + window
  sessions): `e_L ~ x * post + first_in_block + (1 | animal) + (1 | session)` (statsmodels MixedLM, groups = animal,
  session as a variance component); the x:post term (deg per session, Wald 95 % CI) = window slope beyond pre; the same as
  a linear probability model for hit. Overnight reset in the mixed model: trial-level pairs model `e_L ~ day_second *
  post + (1 | animal) + (1 | pair)` on the early third of d+1 and the late third of d (secondary; the paired O above is
  primary).

## T2. Carry-over / generalisation of the re-aimed launch into the ITI -- PRIMARY
* ITI licks (as section 0) with a launch score; readout theta_L (launch; endpoint theta_E secondary).
* Reference = the animal's August-pre ITI licks with the same PREVIOUS spout and the same PHASE: mean theta_L
  (for a pre session, LEAVE-THAT-SESSION-OUT so pre has an honest level). dITI = theta_L - reference (deg; - =
  ipsilesional of the pre-stroke ITI pattern at the same place in the cycle after the same spout).
* Session cells: mean dITI per (previous spout group: FC = far contra, O5 = the other five; each of the six secondary) x
  phase; >= 10 licks per cell.
* **PRIMARY contrast C2 = [d(FC, retract) - d(FC, docked)] - [d(O5, retract) - d(O5, docked)]** (deg; + = early-ITI
  licks after a far-contra trial are more contralesional than the docked ones, beyond pre (built into d) and beyond
  other spouts). Secondary: d(FC, phase) and d(O5, phase) themselves; outbound split by next spout = previous spout or
  not (the incoming target is often the same spout: blocks of 4-8); 0.5 s since-stop bins 0-4 s (descriptive);
  bout-start ITI licks only (>= 0.5 s since the previous lick onset); previous trial a HIT only; theta_E.
  Cued comparator per session: far-contra first-lick e_L (T1 trials) -- the re-aim the carry-over would carry.
* Predictions: LEARNED + CARRY-OVER: C2 > 0 in subacute / chronic (once the cued launch is re-aimed), d(FC, retract)
  near the cued level, d(FC, docked) < 0 (default output). LEARNED, STRICTLY CONTEXT-BOUND (no carry-over): d(FC,
  retract) ~ d(FC, docked) < 0, C2 ~ 0 while cued has recovered. RESTORED CAPACITY: d ~ cued level ~ 0 in every phase,
  C2 ~ 0. ACUTE: no compensation yet -> C2 ~ 0 expected under every hypothesis.
* Stats: per-animal table first; boot_delta per post epoch vs pre (pre C2 from the leave-one-session-out references);
  change criterion as T1; Holm over C2 x 3 epochs. Secondary mixed model (post sessions, lick level, retract + docked):
  `dITI ~ FC * since_stop + epoch + (1 | animal) + (1 | session)`; FC:since_stop < 0 = carry-over decaying with time.

## T3. ITI aim relative to the previous trial's spout (Priya; capacity vs performance gap) -- PRIMARY
* DOCKED ITI licks (primary; all ITI phases secondary). Readout = ENDPOINT angle theta_E (where the tongue goes: the
  aim readout comparable with the cortical landing-referenced offset); theta_L secondary.
* phi_prev = midline angle (same sign) of the previous trial's commanded spout position (`spout_world.commanded_world`,
  the session's rig geometry, as within_lick_correction).
* Per session x previous spout: mean m_p (>= 10 licks). Reference r_p = the animal's August-pre mean at that previous spout
  (pre sessions leave-one-session-out).
  **D_off = mean over previous spouts of (m_p - r_p)** (deg; - = ITI licks land more ipsilesionally than pre after the
  same spout; needs >= 4 of 6 spouts); D_fc = m_fc - r_fc (after far-contra trials); GAIN g = OLS slope of m_p on phi_prev
  across spouts (deg / deg; 0 = ITI aim ignores where the spout was) -> change vs pre; descriptive raw relation m_p -
  phi_prev.
* Cued comparator per session: cued FIRST-lick endpoint error e_E averaged over the same spouts (cued mean) and at far
  contra. **GAP = D_off - cued mean** (same session; - = the ITI aim is more ipsilesional than the cued aim relative to
  each one's pre pattern = a capacity-vs-performance gap).
* Predictions: BEHAVIOURAL gap (performance < capacity): chronic D_off < 0 AND GAP < 0 (changes), cued ~ 0. ONLY A
  CORTICAL readout: chronic D_off and GAP not changes (behavioural ITI aim back to pre while the cortical ITI offset
  persists). Direction-specific compression (bias grows with how contralesional the last spout was): g < pre, D_fc < D_off.
* Stats: per-animal table first; boot_delta per epoch vs pre; change criterion; Holm over (D_off, GAP) x 3 epochs.
  Secondary: per-previous-spout D in canonical order; mixed model (post sessions, lick level) `theta_E - r_p ~ phi_prev
  + epoch + (1 | animal) + (1 | session)`.

## Confounds (fixed handling)
* Fatigue / satiety: engaged (canonical) trials only; ITI licks only after engaged trials; the pre within-session W and
  the pre overnight O are the practice / fatigue controls.
* Spout order / blocks: random block order (so early and late thirds sample block positions alike); `first_in_block` in
  the mixed model; non-first-in-block sensitivity.
* Warm-up decrement mimics learning (within-session gain + overnight loss): the other-five-spout W is the control
  (W_fc - W_o5).
* Regression to the mean: no session or animal is selected or split by its own early performance. W and O share the
  late third of day d (their noise is anti-correlated) but each MEAN is unbiased; no W-vs-O correlation is interpreted.
* ITI lick selection: the reference is matched on previous spout AND cycle phase (the retracting spout leaves FROM the
  previous target; the outbound one moves TO the next); bout-start sensitivity (continuation licks of the cued bout vs new
  bouts); docked as the primary T3 set (stationary spout, no target).
* Tracking: launch noise ~5-8 deg per lick (within_lick_correction); n per cell reported.

## Figures (Priya's defaults)
Session-level points = OPEN markers, SHAPE = animal (`orofacial_cohort.ANIMAL_MARKER`), COLOUR = epoch (`EPOCH_LINE`);
pooled = filled / thick with 95 % CI; day-course plots one panel per animal; canonical spout order; stars only in a
reserved top band; legends outside data; no many-thin-bar figures; + = contralesional. figdata pickle for every figure;
`--plot-only` redraws from figdata.

## Outcome (2026-10-30, run after this design was fixed; `learned_compensation.py run` then `summarise`)

Outputs: `.../imaging_encoding/learned_compensation/` (`trials_t1.csv.gz`, `iti_licks.csv.gz`, `iti_join_diag.csv`,
`spout_angles.csv`, `t1_sessions.csv`, `t1_pairs_gap1/2.csv`, `t1_per_animal_*.csv`, `t1_boot.csv`, `t1_mixed.csv`,
`t2_*.csv`, `t3_*.csv`, `figures/lc_fig1..6_*.png/svg`, `figdata/learned_compensation.pkl`, `run_log.txt`,
`summarise_log.txt`, `summarise_stdout.txt`, `timing/<sid>.pkl`). 107 sessions (31 August pre), 57,404 trials (51,284
engaged; first-lick launch score on 39,884), 78,962 ITI licks (98.8 % matched to the dock audit; own since-stop vs the
audit's: median 1 ms; 99.7 % follow an engaged trial). Window d1-14: 8 sessions per animal (PS94 hit 7); overnight
gap-1 pairs: window 4 per animal, pre 4-6.

### T1 learning signature -- NOT FOUND (improvement arrives BETWEEN sessions, not within them)
Per animal (session means, pre / window d1-14; deg, + = contralesional): W_eL PS92 -1.1 / -3.3, PS93 -3.4 / -6.4,
PS94 -1.7 / +0.7, PS95 +0.4 / -3.2; O_eL (gap-1 pairs) PS92 +2.2 / +5.5, PS93 +4.6 / +14.0, PS94 +1.5 / -3.2,
PS95 -0.3 / +11.0; W_hit PS92 -.06 / -.03, PS93 +.02 / -.18, PS94 -.06 / -.04, PS95 -.09 / -.34.
boot_delta window vs pre (Holm over the 4 primary): **W_eL -1.6 [-5.1, +2.1]** (3/4 negative, p_holm .65); **O_eL +4.8
[-3.0, +12.2]** (3/4 POSITIVE = overnight GAIN, not loss; .65); W_hit -0.10 [-0.26, +0.04] (.65); O_hit +0.14 [-0.04,
+0.35] (.65). Secondary: gap <= 2 d pairs O_eL +4.7 [+0.03, +9.2] (4/4), O_hit +0.23 [+0.06, +0.39] (4/4), day step
S_eL +3.3 [+0.7, +6.2] (4/4), S_hit +0.13 (4/4) -- all changes; W_hit subacute -0.17 [-0.31, -0.04] (4/4; hit rate
FALLS within subacute sessions); W at the other five spouts -1.6 [-3.6, +0.1] (4/4 negative, n.s.); W_fc - W_o5 -0.35
(n.s.); first-lick e_E, first-3-lick e_L, non-first-in-block: same (W n.s., negative). Session mean e_L -17.6 (4/4).
Mixed models: e_L slope x:post -2.5 [-5.2, +0.3] deg/session (x pre -2.1 [-3.9, -0.3]: a small within-session drift
toward ipsilesional also pre; first-in-block trials -7.1 deg); hit x:post -0.13 [-0.18, -0.08]; overnight trial-level
second:post **+5.8 [+2.9, +8.6]** (pairs model, overnight gain beyond pre).
Reading: within a session, far-contra launch error does not improve (if anything worsens, and engaged hit rate falls);
improvement is expressed between sessions (overnight / day steps), consistently across animals. This is the
pre-registered SPONTANEOUS-RECOVERY pattern (day steps, no within-session gain, no overnight loss). It does not exclude
offline (sleep-dependent) consolidation of learning, which also produces between-session gains; T1 cannot separate
those two. No online-learning-with-forgetting signature.

### T2 carry-over -- NOT FOUND (C2 goes the OTHER way in the subacute epoch)
Per animal d (deg vs pre ITI, same previous spout + phase), FC retract / FC docked / O5 retract / O5 docked / C2:
acute PS92 -27.7 / -34.4 / -6.4 / -12.6 / +0.6, PS93 -23.4 / -19.5 / -8.9 / -13.6 / -8.6, PS94 -22.7 / -24.9 / -10.9 /
-19.0 / -7.3, PS95 no FC retract cell (FC docked -28.5); chronic PS92 +5.9 / -0.7 / -1.5 / -1.6 / +6.6, PS93 -3.4 /
-5.2 / -6.9 / -10.1 / -1.5, PS94 -20.3 / -25.5 / -7.1 / -14.7 / -2.5, PS95 -7.5 / -10.8 / -5.3 / -10.4 / -1.9.
**C2: acute -5.0 [-10.0, +0.6] (2/3), subacute -5.9 [-10.7, -2.0] (4/4, change, p_holm < .01; NEGATIVE), chronic -1.0
[-6.2, +4.3]**. ITI licks after a far-contra trial carry the CUED deficit, not the re-aim: FC retract / docked / outbound
-24.6 / -27.1 / -23.6 acute (cued FC e_L -26.1), -12.5 / -11.6 / -11.2 subacute (cued -9.8), chronic -7.0 n.s. / -10.8
[-20.7, -2.6] (4/4) / -5.2 n.s. (cued FC first-lick e_L -3.7 n.s., 2/4). Other spouts: retract -8.3 / -5.0 / -5.1,
docked -14.0 / -8.0 / -8.4 (changes, 4/4 except subacute retract). FC_decay (retract - docked after FC) chronic +5.0
[+1.4, +9.2] (4/4) but the O5 decay is similar (+3.3), so it is not far-contra-specific. Mixed model (post, retract +
docked): since_stop -3.0 [-3.4, -2.6] deg/s (ITI licks drift ipsilesionally away from the pre pattern as the ITI goes
on, all spouts), FC:since_stop +1.4 [+0.7, +2.2] (WEAKER decay after far contra, i.e. opposite to carry-over), FC -5.4.
Variants: theta_E C2 subacute -3.5 (4/4), chronic +0.1; bout-start only and previous-hit only: same sign, chronic n.s.;
outbound by next spout same / different: same as outbound overall.
Reading: a general early-ITI advantage (licks right after any trial sit closer to the pre pattern than docked licks,
all spouts) and, after far-contra trials, ITI licks that follow the cued deficit acutely / subacutely. Chronically the
docked licks after far contra stay ipsilesional (-10.8, carried by PS94 -24.9 and PS95 -12.1; PS92 -1.0, PS93 -5.4).

### T3 ITI aim relative to the previous spout -- a BEHAVIOURAL ITI bias exists, but NO gap beyond the cued licks
Per animal (session means pre / acute / subacute / chronic): D_off PS92 -0.1 / -12.8 / -7.0 / -0.6, PS93 +0.3 / -9.6 /
-1.4 / -1.7, PS94 -0.2 / -18.8 / -10.5 / -11.0, PS95 +11.0 (ONE pre session) / none / -1.2 / -5.8; GAP PS92 -0.5 / -1.0 /
+1.8 / +1.3, PS93 -6.9 / -6.1 / -2.0 / -2.9, PS94 -0.4 / +2.2 / +0.6 / -0.1, PS95 +8.4 / none / +3.8 / -0.2.
**D_off acute -13.7 [-18.5, -9.5] (3/3), subacute -7.7 [-11.4, -3.4] (4/4), chronic -7.5 [-14.1, -1.0] (4/4; p_holm
.01)** -> docked ITI licks land ipsilesionally of their pre pattern after the same spout: the ITI persistence is NOT only
cortical. **GAP (= D_off - cued first-lick e_E over the same spouts) acute +1.0, subacute +0.9, chronic -0.7 [-6.1, +3.3]
(1/4) -- not changes**; the cued first-lick landing error over spouts is itself still -6.8 [-10.0, -3.5] chronic (4/4).
D_fc chronic -8.3 (4/4); post hoc at far contra GAP_fc = D_fc - cued far-contra first-lick e_E: +2.6 / +2.1 / -1.5
(n.s.; PS94 chronic D_fc -16.6 vs cued -20.2). Gain on the previous spout angle chronic -0.04 [-0.07, -0.00] (3/4;
pre ~0.23): slight compression; mixed model D ~ phi_prev -0.055 [-0.060, -0.050] deg/deg (larger ipsilesional bias after
more contralesional spouts). All-ITI theta_E: D_off chronic -4.1 (4/4) and GAP +2.2 [+0.7, +3.9] (4/4: all-phase ITI
licks are LESS deviated than cued first licks). Docked theta_L: same as theta_E.
Reading: behaviourally, docked ITI licks keep an ipsilesional bias that scales with where the last spout was, and it is
the same size as the cued FIRST-lick landing error at the same spouts, session by session. So no capacity-vs-performance
gap is visible behaviourally: the cued "recovery" is mostly in later licks / launch averages, not in a cued first-lick
aim that exceeds the ITI aim.

### Verdict on "learned, context-specific compensation" (behaviour)
Not supported by these three tests. T1: no online learning signature; gains are between sessions (recovery or offline
consolidation). T2: no carry-over of a far-contra re-aim into the ITI; C2 is negative subacutely. T3: the ITI bias is
behavioural (not only cortical) but matches the cued first-lick bias, so the behavioural data do not show cued licks
escaping a deficit that ITI licks keep. The cortical ITI-vs-cued dissociation (FINDINGS 0b) is therefore not mirrored by
a cued-vs-ITI dissociation in first-lick aim. PROVISIONAL (n = 4).

### Deviations
1. ITI time bins: the example "0-2 / 2-5 / > 5 s" bins were replaced BEFORE any number by the cycle phases (the ITI
   context spans only ~0-4 s after the stop; stated in section 0).
2. Pre references need >= 5 pre licks per (animal, previous spout, phase / bin) (implementation detail not written above).
3. The window d1-14 includes one or two sessions per animal classed chronic by the epoch boundaries (PS92 / PS93 day
   11); per-epoch rows are reported.
4. GAP_fc (far-contra-only gap) is POST HOC.
5. Stars use a local top-band helper (`sig_marks` in this worktree has no `star_top`).
Caveats: **PS95 has very few August-pre ITI licks** (one pre session with a T3 D_off cell, two with a C2 cell; acute one
session) -- rule-8 trap: its pre uncertainty never enters the CI and its T3 / T2 pre value is a single session
(+11.0 D_off). PS95 acute is one session. The within-session hit-rate decline post-stroke shows the engagement gate does
not remove all within-session state change. Launch angle noise ~5-8 deg per lick; cells are means of >= 10 licks /
>= 24 trials. T1 cannot separate offline consolidation from biological recovery.
