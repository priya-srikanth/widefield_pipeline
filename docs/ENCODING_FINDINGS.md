# Imaging encoding findings: target vs executed, per-spout coding, recovery

Status: **preliminary, living document.** Last updated **2026-10-21** (encoding window). Update this file whenever a
result below changes: edit the section, change its status tag, add the date to the change log at the bottom. The
reasoning and every number's history are in `DECISIONS.md` (entries 2026-10-19 ... 2026-10-20); the working state
(running jobs, commands, pitfalls) is in the newest `docs/status/STATUS_*_ENCODING_HANDOFF.md`.

Status tags: **STANDS** (survived the controls listed), **PROVISIONAL** (one control still pending), **SUSPENDED**
(contradicted by a better measurement on part of the data; waiting for the full check), **PENDING** (not yet run).

> **Cohort and rules.** VLS-stroke cohort PS92-PS95, all lesions LEFT (left hemisphere = ipsilesional; spouts named
> from the lesion: contra = the mouse's RIGHT). LP run `lp_r4_heatmap_20261013__clean_v36`; **August-only pre baseline**
> (June sessions only where stated, as a sensitivity check); engaged trials only. n = 4 animals: every claim needs the
> per-animal table and >= 3/4 animals (CLAUDE.md rule 8). Brain maps: image left = left hemisphere.
>
> **Open data issue (10-20):** the v36 timing fix changed PS92_0908 (chronic) and the June PS92_0606 / PS93_0606; their
> encoding results are stale until rerun (handoff to-do 7). Cohort conclusions below do not depend on them.

## 1. The question and the model

Does cortex track the **TARGET** (where the spout is) or the **EXECUTED** lick (where the tongue went), and how does that
change after the stroke? Model: Musall-style ridge encoding of the widefield signal (`wfield_local/combined_encoding.py`),
drift slower than 600 s removed inside the model, block-wise cross-validation.

**Names used below.**

| term | meaning |
|---|---|
| nokin base | cue, spout arrival, lick timing (lick + contact kernels), running; NO tongue / jaw / nose traces |
| category | a free lick- and cue-locked response per spout (any spout-specific pattern) |
| target | lick kernel x the spout's geometry (angle, angle^2, distance; or 3-D, see 2.4) |
| executed | lick kernel x each lick's own angle, angle^2, reach (cam4) or 3-D equivalent |
| ceiling | category - none: all spout-specific signal |
| unique target / executed | (target + executed) minus the model without that set: what each explains beyond the other |
| strict executed | (category + executed) - category: executed beyond ANY spout-specific pattern = lick-to-lick |
| trial-to-trial | variance around each spout's mean response (cue 0-1.5 s) |
| drop-p | per-spout unique signal: spout p constrained to the mean of the other five (no trial relabelled) |

All values are unique dR2 x1000 in the lick window, cortex, unless stated.

## 2. Target vs executed

### 2.1 Fair contest with cam4 kinematics, all four animals — STANDS for the post-stroke rise; pre-stroke reading SUSPENDED (2.4)

`imaging_encoding/movement_first_subset-nokin/` (119 sessions; `tables/relative_shares.csv`, `figures/spatial_contest_*`).
Pre-stroke unique target > unique executed in 4/4; unique executed rises subacutely (+0.012) and chronically (+0.010) in
4/4. Relative shares (unique / ceiling) move toward executed acutely / subacutely and back toward target chronically
in 3/4; PS94 stays executed-weighted.

### 2.2 Lick-matched control — STANDS

`movement_first_subset-nokin_matched/` (107 sessions; whole engaged trials dropped until every session of an animal has
the same usable licks). Pre -> chronic ceiling: PS92 .034 -> .107, PS93 .042 -> .096, PS95 .041 -> .071 (real),
PS94 .031 -> .025 (its unmatched rise was more data). Unique executed pre ~0 in all four (.004 / .003 / -.003 / -.004),
rising post-stroke. Chronic share of the ceiling, target / executed: PS92 .51 / .15, PS93 .43 / .12, PS95 .35 / .05,
PS94 .45 / .60.

### 2.3 Trial-to-trial contest (cam4) — STANDS as a cam4 result

`movement_first_subset-nokin_tr/tables/tr_contest_0-1.5s.csv` (107 sessions). Pre: neither set explains lick-to-lick
variance in 3/4 (PS92 the exception: MOs 10.2, SSp-m 13.3 executed). Acute: both rise a little (not executed-specific).
Chronic: only PS94 tracks the executed lick trial by trial (cortex 30.1, SSp-m 65.3). So no general post-stroke shift to
lick-by-lick executed tracking **with cam4 measurement** (its tracking noise biases this test against executed).
**First-lick-aligned version (10-21, O2 `_win` run, all 119; `movement_first_subset-nokin_win/tables/tr_contest_0-1.5s.csv`,
align = firstlick, -0.1 .. +1 s):** same picture. No executed tracking in >= 3 animals in any epoch; PS94 chronic stays
the exception (cortex executed 37.2, strict 34.8 x1e3); PS92 strict executed rises acute / subacute (cortex 22.5 / 35.3)
but not chronic. Aligning to the delayed far-contra first lick does not reveal a hidden executed code.

### 2.4 3-D executed sets, PS92 only — PROVISIONAL (one animal); this is why 2.1's pre-stroke reading is suspended

`movement_first_subset-nokin_exec-{cam4m,3d,3dxyz,xyz2}/`; figure
`movement_first_subset-nokin_exec-xyz2/figures/exec_sets_compared_PS92.png`. Same licks in every version.

| pre-stroke, cortex | unique target | unique executed |
|---|---|---|
| cam4 (angle, angle^2, reach) | 31.7 | 10.8 |
| 3-D, same form (angle out of the midline plane, angle^2, 3-D reach) | 26.8 | 12.7 |
| **xyz2: tongue-peak 3-D position vs spout 3-D position (coordinate-matched)** | **8.0** | **11.7** |

MOs (xyz2) 7.3 vs 17.6; SSp-m target ~ executed (left 29.7 vs 21.4, right 21.0 vs 20.9). Trial-to-trial (xyz2), pre: MOs
executed 12.6 vs target -1.7, MOp 11.2 vs -1.3, SSp-m 15.9 vs 10.0. Acute: executed > target (cortex 23.4 vs 10.6).
The 3-D sets are not redundant: 54 % of the tongue's 3-D position per lick is not predictable from the spout.
**Reading:** in PS92 the pre-stroke target advantage came mostly from cam4's description of the tongue (blind to
forward, compressed lateral angle); in 3-D, motor cortex carries the executed lick at least as much as the target.
`3dxyz` (linear 3-D executed vs angular target) is RETIRED — mismatched coordinates.
**Pending:** the `polar` version (both sides: lateral angle, its square, forward tilt, 3-D distance) on PS92; then the
cohort 3-D contest (xyz2 / polar + strict executed + combined cam1 / cam4 licks) when cam1 production lands — the
deciding analysis.

### 2.4b 3-D polar contest, PS92 + PS93 (47 sessions) — PROVISIONAL (two animals); the cohort 3-D answer so far

`movement_first_subset-nokin_exec-polar_licks-combined_bins/` (polar executed set; cam4 + cam1 lick timing; soft
calibrated landing-zone bins; 20261021 tables; 10-21 01:57). Cortex, dR2 x1e3, rule 8 (animals first):

| | PS92 pre / acute / sub / chronic | PS93 pre / acute / sub / chronic |
|---|---|---|
| unique target, lick | 5.2 / 14.4 / 29.6 / 29.6 | 27.5 / 8.3 / 14.2 / 51.4 |
| unique executed, lick | 6.1 / 8.2 / 20.6 / 8.1 | 6.3 / 4.9 / 9.7 / 20.2 |
| unique target, cue | 2.7 / 5.7 / 27.3 / 37.3 | 9.2 / 2.5 / 4.7 / 26.9 |
| unique executed, cue | 0.9 / 3.7 / 19.1 / 11.7 | 3.0 / 5.6 / 12.2 / 14.9 |
| binned: target bins / executed bins, lick | -0.1 / 8.9 / 10.0 / 10.5 vs -0.1 / -16.3 / 4.0 / 2.7 | 9.8 / -0.6 / 12.4 / 30.3 vs 5.0 / 3.9 / 10.3 / 9.2 |

Mixed models (`qc/encoding_mixed.csv`, 3d:* measures; index = time to contact): unique target recovers with log days
(cue z 7.4) and falls with the deficit (cue z -5.3; target bins z -7.3); executed weaker (cue time z 2.1, deficit
z -2.7). STRICT executed (beyond the spout category): present before the stroke in the lick window (12.4, z 5.4) and
unchanged after (step z ~1).
**Reading:** pre-stroke, target and executed are comparable (PS92) or target-led (PS93), consistent with 2.4 suspending
the cam4 pre-stroke target advantage. After the stroke the rise is TARGET: unique target grows over recovery in both
animals and both windows, beyond pre by chronic; executed rises less and the executed component beyond the spout
category does not change. **Lick-template test** (pre-stroke per-spout templates held fixed, first pass on 42 sessions,
20261020 tables; `qc/lick_templates_polar/`): target templates explain more unique post-stroke lick-window variance
than executed (soft-zone) templates from subacute on (cortex PS92 1.34 vs 0.80, chronic 0.95 vs 0.31; PS93 2.67 vs
-0.15, chronic 6.38 vs 0.10 -- check per session before quoting); acute ~tied; own-zone landing 0.70-0.76 pre vs
0.46-0.57 acute, so the test has power exactly there. **Rerun (all 47, 20261021 tables; `qc/lick_templates_polar_20261021/`;
QUOTE THIS ONE):** cortex u_tgt vs u_exec (% of lick-window variance) PS92 pre -0.03 / 0.28, acute 1.26 / 1.44,
subacute 1.34 / 0.80, chronic 0.95 / 0.31; PS93 pre 2.22 / 0.47, acute 0.36 / -0.12, subacute 2.72 / -0.15, chronic
5.44 / 0.09 -- PS93 chronic holds in EVERY session (target 2.4-8.2, executed -0.3-0.6). With the spout category in the
base (`_cat`) both shrink to <= 1 and executed is ~0 post-stroke: the target-template advantage is spout identity, and
landing-zone templates add nothing beyond it. Two animals; acute ~tied.

### 2.4c Destination vs landing (10b x 3-D landing zones), PS92 + PS93 — PROVISIONAL (two animals)

`scripts/best_match_destination.py` -> `qc/best_match_destination.csv`. Brain: the pre-stroke spout whose activity
pattern each post-stroke spout matches best (figure 10b, cue / working, frozen). Behaviour: where its licks landed (3-D
calibrated zones; cam4 zones FAILED the pre check and are not used). Pre self-match ~1.0 except PS93 far_mid (0.62).
Far contra: PS92 target-like in every session incl. acute (5/5, own-zone landing 16 %); PS93 acute mixed (1 target, 1
execution = far mid, 2 neither = close mid), target from subacute on. PS93 far_mid lands correctly (~0.7) but its
pattern matches pre-stroke FAR CONTRA by chronic (5/6) -- neither. All spouts: target-like 61 / 73 / 90 % (acute /
subacute / chronic), execution-like 4 % acute only, rest neither. **Reading:** post-stroke spout activity resembles the
pre-stroke TARGET pattern far more than pre-stroke execution; deviations are "neither" and fade with recovery.

### 2.5 Misdirected licks — STANDS (negative)

Template analysis of acute far-contra misses (`cohort/tables/misdirected_licks/`): every plan-like lean traced to touch,
reach, lick number or the previous lick; first attempts matched on lick number and reach ~0, same as pre-stroke misses.
No per-lick plan signature. "Target" here = spout location / touch / posture / context; it does not isolate intention.

- **Within-spout deviation check, cam4 (PROVISIONAL, 10-20).** With the spout fixed, the crossed (left - right) SSp-m
  response follows the trial's executed lateral deviation weakly pre-stroke (within / between 0.04-0.16, 4/4 same sign,
  t = -3.6 over 30 sessions; MOs ~0) and more after (subacute 0.11-0.25 in 4/4; chronic PS92 .34, PS93 / PS94 ~.12,
  PS95 0). Lower bounds (cam4 angle noise). `qc/within_spout_deviation/`, DECISIONS 2026-10-20.
  A BILATERAL component is the larger post-stroke effect (acute: more contra-directed licks -> larger response in
  both hemispheres), concentrated in PS94 and PS95 0817; read alongside the crossed part, cause under test
  (contact / reward / latency, spout side).
  LICK-ALIGNED (10-20, default now; acute far-contra licking starts >1.5 s after the cue on 35-38 % of PS92 / PS93
  trials): between maps unchanged; crossed within slope stronger (SSp-m pre t = -6.5, ratio .10-.20, MOs t = -3.2),
  roughly flat post-stroke except PS92 chronic .38; bilateral PS94 / PS95 acute directional effect survives, PS94
  flips positive subacute / chronic. `qc/within_spout_deviation_lick/`.
  Mixed model (animal random intercept + post step; post + post:log(day)): crossed within pre level significant
  (SSp-m -1.68, q .03; MOp q .003; MOs q .04); no post-stroke term survives FDR (bilateral course z ~2.3, q .08-.11).
  Repo map tests with animal as unit (nested bootstrap, cluster): between pre vs 0 = 28 % of brain; no post-vs-pre
  change in either map. `mixed_model.csv`, `figures/within_spout_maps_{nested,cluster}_*.png`.

- **Acute far-contra licking (behaviour, 10-20).** First lick > 1.5 s after the cue on 35-38 % (PS92 / PS93) vs
  17-18 % (PS94 / PS95) of licked trials; cam1's small licks barely change it (-> real delay); no spout contact on
  85-97 % of those trials in all four. PS92 / PS93 = slow to start, PS94 / PS95 = prompt but miss. `qc/first_lick_latency*`.

- **Direction vs toward-midline (eccentricity), within spout (10-20 night; POOLED VERSION NOT ROBUST, see the 10-21
  bullet below).** Crossed SSp-m DIRECTION
  code present pre in 4/4 (mixed-model pre level q .048) and persists. A TOWARD-MIDLINE (eccentricity) term changes
  after the stroke: crossed MOp / MOs / SSp-m step down at day 1 and recover (q .016-.053), and the map test (nested
  bootstrap, animal = unit) finds 20 % of brain changed chronically -- bilateral anterolateral frontal cortex comes to
  respond more to more LATERAL licks (pre: medial licks evoke more along the midline). First post-stroke map change
  surviving the animal-level test. PS94 / PS95's pre bilateral pattern = eccentricity; acute = contralesional direction.
  Split-half reliability ~0.5 -> executed shares bracketed x1-x2. `qc/within_spout_deviation_lick/`.

- **Within-spout per spout + contact control (PROVISIONAL, 10-21; cam4 lick-aligned, 107 sessions; folders
  `qc/within_spout_deviation_lick{,_contact,_contact_hits}/`, DECISIONS 2026-10-21 late morning).** (1) The POOLED
  chronic toward-midline reversal does not survive the contact covariate: animal-level cluster test 28 % of brain in
  base, nothing in `_contact` or `_contact_hits`; LMM chronic z (MOs bilateral) -3.4 / -2.9 / -1.5. (2) Per spout the
  lateral spouts disagree pre-stroke -- far spouts + (more activity for MEDIAL licks; far_ipsi pre z 2.9-3.8), close
  spouts - (more for LATERAL licks; close_contra pre z -2.2 to -3.3) -- so the pooled coefficient averages opposite
  signs. Robust to both controls (epoch models, descriptive): far_ipsi chronic loss (z -2.3 to -3.0, MOp / MOs / SSp-m
  bilateral) and close_contra acute flip (z +2.9 to +3.5); time-model terms weak (|z| < 2.6). (3) Commanded-axis
  across-spout slope rises post-stroke with the deficit (SSp-m bilateral z 9.7) but collapses hits-only (z 1.4): a
  miss / contact confound until the contact-adjusted between fit (code a00045b) is rerun. Executed-axis between: null.
  **Direct contact check** (`S/contact_dev_check.py`; per session x lateral spout, contacts per 10 deg toward the
  midline with reach / n_licks / latency fixed): the coupling is PRESENT PRE-STROKE and about the same chronically
  (pooled per animal pre -0.14 to -0.53, chronic -0.38 to -0.53; far spouts: outward licks get more contacts in 4/4 pre
  and chronic; hit fraction ~1 both), so contact cannot readily manufacture the pre -> chronic change -- the pooled
  cluster loss is mostly threshold fragility, NOT a demonstrated contact artifact. Acute far-contra trials touch on
  2-15 % only. **Reading:** no robust within-spout execution-like shift; the headline is the per-spout far_ipsi
  chronic change (contact-controlled); the pooled map is fragile because the spouts disagree in sign.
  **Follow-ups (10-21; DECISIONS late morning):** 3-D agrees with cam4 per spout; per-animal maps and first-lick
  angles put far_ipsi chronic at 2/4 (PS94 reversal, PS92 drop; PS93 opposite) and the close_contra acute flip at 2-3/4;
  far contra inconsistent. Within-spout angle terms do not predict held-out activity (CV gain ~0), so an eccentricity
  tuning curve cannot be tested. **No per-spout toward-midline effect meets rule 8; no robust within-spout
  execution-like signal post-stroke.** Behaviour: miss-then-correct trials common (far contra pre 25-56 %); PS94
  chronic far contra 80 % (first-lick contact 15 % vs 71 % pre) = corrective recovery.

- **Coding tracks behavioural recovery and overshoots (PROVISIONAL, 10-20 night; `scripts/encoding_mixed.py`).**
  Mixed models with animal random effects, sessions as observations. Calendar time: position coding drops at day 1
  (cue) and recovers with log(days), mostly via DISTANCE. Behaviour-indexed (far-contra contact deficit): position /
  angle coding falls with the deficit and sits ABOVE pre once behaviour has recovered (lick distance z -5.2 / +4.5;
  cue unique target z -8.0 / +8.7). Mixed models are now the primary readout; epochs (chronic starts day 11-25 by
  animal) descriptive. Cue-window effects partly = delayed licking -> `firstlick` window pending (O2).

- **Jaw-onset rewrite (10-21 07:57: cam1 for all PS93 + PS94 to 08-17) -> attempts + all LMMs rerun.** Unknown-class
  trials <= 6 %; far-contra deficit unchanged (acute attempt .81-.93, completion .01-.11; time to contact pre 545-648
  -> acute 3221-3467 -> chronic 471-692 ms). Target tracks the deficit more strongly: 3-D unique target post:deficit
  z -8.3 (was -5.3), cam4 z -8.2 (cue) / -6.3 (first lick); strict executed still null.

- **jaw_only = reaction, not attempt (10-21, Priya's blind review 0/24) -> attempts + LMMs rerun.** Acute far-contra
  attempt rate .79 / .80 / .64 / .32 (PS92-PS95): PS94 / PS95 also fail to INITIATE; completion .03-.15 in all four.
  Target still tracks the deficit (3-D cue z -6.8, cam4 -6.7 / first lick -5.1); position first lick now z -2.0
  (q .08); strict executed null. Old outputs `qc/superseded_attempts_jaw_20261021/`.

- **Decoders recover with behaviour, the code stays reorganised (PROVISIONAL, 10-20 late; `scripts/imaging_lmm.py`).**
  Frozen-decoder deficit ~0 at full behavioural recovery while refit-over-frozen reorganisation stays positive (z 3.7).
  All session-level time courses now via `wfield_local/mixed_models.py` (one definition).

## 3. Where and how spouts are coded

### 3.1 Per-spout unique maps — STANDS (PS92_0908 rerun done 10-20: PS92 chronic unchanged)

Drop-p: `movement_first_subset-droppos-/` (119 sessions; `figures/position_each_maps_*`, `position_share_maps_*`,
`tables/per_spout_dR2_by_animal.csv`). The one-vs-rest relabelling version (`_subset-shufpos-`) mislabels donor trials
and pulls the six spouts together; drop-p is the clean measure.
- Pre: the midline spouts are the most distinct (close mid 14-32, far mid 12-24 in 3/4); lateral spouts small (3-12).
- **Acute: far contra falls to ~0 in all four** (cortex 3 -> -3, 6 -> 0, 4 -> 2, 10 -> 0; contralesional SSp-m 11 -> 1,
  15 -> 2, 14 -> 9, 22 -> 2). Close mid falls in 3/4.
- **Chronic: gains concentrated in PS92 / PS93 at far mid** (12 -> 34, 24 -> 54; plus PS92 close contra 7 -> 35), with a
  posterior-medial share; PS95 back to pre levels, PS94 flat.

### 3.2 Signed per-spout maps and laterality — STANDS (descriptive)

`movement_first/figures/position_signed_{respdev,resp}_*.png`. In the measured activity every lateral spout is CROSSED
(SSp-m, pre, licking, left - right: far contra +10.8..+15.6, far ipsi -1.1..-13.2, 4/4). Splitting the model: for FAR
spouts the crossed part is in the movement terms (licks + tongue / jaw) and the position-only part is ~0; for CLOSE
spouts about half is position beyond movement (crossed, consistent with whisker / face touch range). The small
uncrossed far-spout signal in the drop-p maps is a residual of the movement-first ordering, not biology (shared with
lick direction; not separable here). Around licking the dominant per-spout pattern is gain on a shared network
(midline spouts > lateral), not topography.

## 4. Recovery

### 4.1 Chronic position-coding gain — STANDS

Beyond the pre-stroke practice trend: with June + August (69-day pre span) the pre-stroke slope is ~0 in all four;
chronic exceeds it in PS92 / PS93 / PS95 (ceiling +.047 / +.080 / +.058), not PS94. In signal units the spread of the
six spout deviations nearly doubles in PS92 / PS93 (4.1 -> 7.3, 3.9 -> 6.8), flat in PS94 / PS95.
`imaging_encoding/qc/position_coding_trend_recovery*.csv`.

### 4.2 It FOLLOWS behavioural recovery — PROVISIONAL (coarse sampling)

Day of 50 % of the acute -> chronic change: far-contra contact recovers first (PS92 day 9 vs neural 15, PS93 7 vs 15,
PS95 3 vs 4); PS94 partly recovers behaviourally with no cortical gain. Reading: cortex consolidates or reflects a
recovered behaviour more than it drives recovery. Sessions every 2-4 days. `qc/position_coding_half_recovery.csv`.

## 5. Caveats that apply across sections

- **Pre-cue slow band (STANDS as a size caveat; checked 10-20).** The >= 300 s component decodes position on its own
  (PS92 / PS93 ~0.3, PS94 / PS95 ~0.1) and scales with block length, which changes with epoch in PS92 / PS93. By-epoch
  re-split (108 curated sessions): removing it lowers pre-cue decode by a similar ~.03-.09 in every epoch, so the epoch
  pattern is not created by it; there is no consistent acute pre-cue LOSS (up in PS92 / PS95, down in PS93 / PS94);
  PS93's acute drop is the slow band (-.057 -> +.006) and its subacute drop halves; PS94's drop survives at ~2/3;
  chronic >= pre in all four. Report slow-band-removed pre-cue decode as a sensitivity column; lick windows unaffected.
  `qc/slowband_decode_all.csv`, `qc/slowband_caveat.png`.
- **Pre-cue window: contact-free by design, not necessarily lick-free (10-20 night).** The ENL restarts on any spout
  contact, so the final 2 s before the cue has no contacts (last contact median 2.2-2.5 s before the cue); but
  no-contact licks can occur in it and lick-evoked GCaMP6s activity from contacts just before it can spill in.
  Pending: whole-session lick detection (other window) and an early vs late ENL split. `qc/precue_contacts.csv`.
- **cam4 kinematics:** lateral angle faithful (r 0.95-0.97 with 3-D); "protrusion / reach" is 99 % the down axis, blind
  to forward. Every cam4 executed test is biased against executed by tracking noise and the missing axis.
- **Lick events:** cam4-only so far; cam1 adds 3-7 % small licks and incomplete licks rise post-stroke (PS92 far_R
  share .17 -> .86 acute). Combined cam1 + cam4 lick timing (optional lips kernel) is coded, not yet run.
- **Movement-first ordering** credits shared variance to movement; "after movement" = after licks, tongue, jaw, nose,
  running (cam4 kinematics, no video). Video is not used in the lick window (it re-describes the licks).
- **Shuffle dR2 of slow / event variables is inflated** (circular shift); lead with drop dR2.

## 6. Where to look

| question | folder under `<LP_analysis>/<run>/imaging_encoding/` | file |
|---|---|---|
| target vs executed, cohort (cam4) | `movement_first_subset-nokin_matched/` | `tables/relative_shares.csv` |
| trial-to-trial | `movement_first_subset-nokin_tr/` | `tables/tr_contest_0-1.5s.csv` |
| 3-D pilot (PS92) | `movement_first_subset-nokin_exec-xyz2/` | `figures/exec_sets_compared_PS92.png` |
| per-spout maps | `movement_first_subset-droppos-/` | `figures/position_each_maps_lick.png`, `position_share_maps_lick.png` |
| signed per-spout maps | `movement_first/` | `figures/position_signed_respdev_*.png` |
| recovery timing | `qc/` | `position_coding_half_recovery.csv` |
| slow band | `qc/` | `slowband_caveat.png`, `slowband_decode_all.csv` |
| within-spout deviation | `qc/within_spout_deviation/` | `within_spout_deviation.csv`, `figures/within_spout_*.png` |
| within-spout per spout / contact (10-21) | `qc/within_spout_deviation_lick{,_contact,_contact_hits}/` | `mixed_model.csv`, `map_significance_*.csv`; maps in `figures/within_spout/<folder>/` |
| attempt classes / deficit | `qc/` | `trial_attempts*.csv` |
| destination vs landing | `qc/` | `best_match_destination*.csv` |

## 7. Next (in order)

1. After the v36 rerun: PS92_0908 everywhere; polar on PS92; smoke-test combined licks + strict executed.
2. Cohort 3-D contest (xyz2 / polar, strict executed, combined licks) when cam1 production lands.
3. Binned contest (`COMBINED_ENCODING_BINS`) on PS92 polar, then in the cohort 3-D contest; within-spout deviation
   check (`scripts/within_spout_deviation.py`) cam4 cohort running 10-20, 3-D version with the cohort 3-D tables.
4. Pre-cue -> lick-window carry-over decode (cross-window, from the CD code).
5. Deferred: non-linear ceiling check.

## Change log

- **2026-10-21** — trial attempts + time-to-contact deficit; 3-D contest + template rerun (2.4b); destination vs 3-D
  landing (2.4c); jaw-onset rewrite rerun; within-spout per spout + contact control (pooled toward-midline not
  robust); one-load variants + local input cache (handoff STATUS_2026-10-21_ENCODING_HANDOFF).

- **2026-10-20 (evening)** — within-spout lick-aligned / bilateral / mixed model / map tests; 3-D landing zones
  calibrated (soft bins); lick-template test built; 3-D contest queued (handoff STATUS_2026-10-20_EVENING_ENCODING_HANDOFF).
- **2026-10-20** — slow band checked by epoch (size caveat; PS93 acute pre-cue drop withdrawn); binned contest and
  within-spout deviation check added (DECISIONS 2026-10-20).
- **2026-10-20** — created from DECISIONS 2026-10-19 ... 2026-10-20 (encoding window).
