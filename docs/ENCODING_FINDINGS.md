# Imaging encoding findings: target vs executed, per-spout coding, recovery

Status: **preliminary, living document.** Last updated **2026-10-23 afternoon** (encoding window). Update this file
whenever a result below changes: edit the section, change its status tag, add the date to the change log at the bottom.
The reasoning and every number's history are in `DECISIONS.md` (entries 2026-10-19 ... 2026-10-23); the working state
(running jobs, commands, pitfalls) is in the newest `docs/status/STATUS_*_ENCODING_HANDOFF.md` (now 10-22, §000f).

Status tags: **STANDS** (survived the controls listed), **PROVISIONAL** (one control still pending), **SUSPENDED**
(contradicted by a better measurement on part of the data; waiting for the full check), **PENDING** (not yet run),
**CORRECTED** (an earlier statement here was wrong; the corrected numbers follow).

> **Where things stand (10-23), with what each rests on.**
>
> | result | status | data it rests on / what is incomplete |
> |---|---|---|
> **10-24: attempts table REBUILT with complete cam1** (was cam1 on PS94 40 % / PS95 0 % of trials). Effect is small:
> gate flips 0 / 0 / 93 / 62 trials (0 / 0 / 0.6 / 0.4 %, all toward attempted); far-contra acute attempt rate PS94
> .441 -> .445, PS95 .325 -> .342 -> "PS94 / PS95 also fail to initiate" stands; gated results below are not expected
> to move; final-frame re-runs use the new table.
>
> | Acute far-contra drop is carried by LICK in the frozen pre-stroke encoder; target <= ~4 x1e3 (3.4) | PROVISIONAL, robust to credit order and to 5 running models | 4/4 animals, but PS95 = 1 acute session; PS94 / PS95 lick timing cam4-only in these runs (their cam1 landed 10-21..23, not yet used); running model v7 still being chosen; no test on data - model; not yet on the LMMs |
> | Post-stroke rise is TARGET, not executed (2.1, 2.4b, LMM 4.3.4) | STANDS for cam4; PROVISIONAL in 3-D | 3-D results so far PS92 + PS93 (10-21 snapshot; 3-D now exists for all 4 -> re-run due); the running 3-D 2x2 is on the snapshot; pre-stroke answer depends on the frame (2.4 vs 2.4b) |
> | Destination vs landing "target-like 61 / 73 / 90 %" (2.4c) | CORRECTED (inflated) | separating rows only: target-leaning, ~6 decisive sessions, 2 animals |
> | Target vs executed, decisive test = lick-level STRAYS vs on-target (2.6) | FIRST RUN: not decisive (far-contra reference fails validation 3/4) | census on the 10-21 snapshot (PS92 + PS93, trial licks); 3-D for all 4 now exists; per-lick ITI 3-D requested |
> | Within-spout lick direction (2.5) | PROVISIONAL; no robust post-stroke execution-like shift | cam4 (all 4) / 3-D (2, snapshot; 4 now available); split-half reliability ~0.5 |
> | Chronic coding gain beyond behavioural normalisation (4.1, 4.3.3) | STANDS (4.1) / PROVISIONAL (LMM) | PS94 shows no gain |
> | Frozen decoder back at full behavioural recovery (2, 4.3.5) | CORRECTED: NOT back (+0.08, only PS92 back); refit gain over frozen > 0 in 4/4 but not significant under the deck bootstrap | LMM vs deck resampling disagree (DECISIONS 2026-09-12) |
> | Running response is sublinear, forward-only from 1.5 mm/s = v7 default; a push term breaks post-stroke windows (3.4) | PROVISIONAL (v7 chosen 10-23) | push frames 2-5x more frequent post-stroke in 3/4 -> no push term |
>
> **Standing re-run rule (Priya 10-23) -- NOW DUE:** every lick-DIRECTION result (exec_within / exec_locked dR2, stray
> test, 3-D contests, within-spout direction, direction maps) must be RE-RUN on full 3-D. **Update 10-23 afternoon (pose
> window):** 3-D now covers ALL FOUR animals (124 sessions: PS92 35, PS93 31, PS94 30, PS95 28) and cam1 is 125 / 127
> curated sessions (PS95_0817 / PS95_0831 still on O2). Every 3-D result in this file was computed from the 10-21
> SNAPSHOT (`pose3d_kin_licks_snapshot_20261021`, PS92 + PS93 only), and the "combined" lick timing in the runs above
> predates PS94 / PS95 cam1 (cam4-only there). So the "2 animals" / "cam4 only" limits below are limits of OUR INPUTS,
> not of the tracking. **3-D COMPLETE 10-23 18:28** (cam1 127 / 127; 3-D for all 119 curated sessions incl. PS95_0817,
> PS95's only acute session); re-cut (`scripts/prep_3d_inputs.py`, 107 encoding sessions, all four animals) and the
> 4-animal re-runs started (polar contest first, `_3d4`). Until each re-run lands, the result above it is still the
> 2-animal / cam4 one. Per-lick 3-D for every lick including pre-cue and ITI licks exists since
> 10-23 (`session_poses/pose3d/kinematics_all/licks.csv.gz`, 880k licks, column `context`; ITI licks have no target).

> **Cohort and rules.** VLS-stroke cohort PS92-PS95, all lesions LEFT (left hemisphere = ipsilesional; spouts named
> from the lesion: contra = the mouse's RIGHT). LP run `lp_r4_heatmap_20261013__clean_v36`; **August-only pre baseline**
> (June sessions only where stated, as a sensitivity check); engaged trials only. n = 4 animals: every claim needs the
> per-animal table and >= 3/4 animals (CLAUDE.md rule 8). Brain maps: image left = left hemisphere.
>
> **Data issues (updated 10-23):** PS92_0908 (v36 timing fix) rerun done 10-20 (3.1: PS92 chronic unchanged); its
> FaceRhythm timing (cam4 dropped ~13 % of frames) RESOLVED 10-23 on fr-timing-v1 (FR model cortex R2 .506 -> .618;
> PS92_0812 unchanged). June PS92_0606 / PS93_0606 are only in the June sensitivity runs, which are not primary.

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

### 2.4b 3-D polar contest — 4 ANIMALS DONE 10-24 (107 sessions); the two-animal record follows below

**Four-animal re-run (`movement_first_subset-nokin_exec-polar_licks-combined_bins_3d4/`, 20261023 re-cut tables,
combined lick timing, soft calibrated bins; tables via `scripts/combined_encoding_figures.py --order ..._3d4`).**
Cortex, unique dR2 x1e3, August pre, per animal (PS92 / PS93 / PS94 / PS95), pre -> acute -> subacute -> chronic:

| | PS92 | PS93 | PS94 | PS95 |
|---|---|---|---|---|
| unique target, lick | 5.2 / 14.4 / 29.6 / 29.6 | 27.5 / 8.3 / 14.2 / 65.4 | 5.7 / 6.6 / 17.4 / 24.0 | 18.1 / 32.7 / 8.6 / 23.5 |
| unique executed, lick | 6.1 / 8.2 / 20.6 / 8.1 | 6.3 / 4.9 / 9.7 / 19.8 | 2.9 / 12.1 / 25.3 / 22.9 | -0.2 / -1.8 / 1.4 / 8.0 |
| unique target, cue | 2.7 / 5.7 / 27.3 / 37.3 | 9.2 / 2.5 / 4.7 / 35.0 | 7.5 / 7.7 / 15.7 / 36.1 | 18.8 / 14.1 / 13.8 / 35.1 |
| unique executed, cue | 0.9 / 3.7 / 19.1 / 11.7 | 3.0 / 5.6 / 12.2 / 15.5 | 0.9 / 9.7 / 15.6 / 28.7 | 0.3 / -3.0 / -3.2 / 8.3 |
| executed beyond spout category, lick | 15.1 / 21.4 / 30.2 / 13.3 | 9.3 / 13.4 / 11.8 / 19.2 | 6.4 / 10.5 / 21.7 / 17.7 | 1.4 / 9.4 / -10.0 / 5.9 |

Sessions per epoch: PS92 8 / 5 / 2 / 12, PS93 7 / 4 / 3 / 12, PS94 8 / 6 / 5 / 8, PS95 8 / 1 / 7 / 11. PS92 and
PS93 pre-to-subacute values reproduce the two-animal run exactly (same 3-D inputs); PS93 chronic now has 12 sessions.
- **PRIMARY (10-24, DECISIONS): the BINNED contest -- matched capacity, independent of polar vs xyz2.** Cortex, unique
  dR2 x1e3, spout identity (target bins) vs landing zone (soft executed bins, width 1 within-spout SD), pre -> acute ->
  subacute -> chronic. Lick: PS92 -0.1 / 8.9 / 10.0 / 10.5 vs -0.1 / -16.3 / 4.0 / 2.7; PS93 9.8 / -0.6 / 12.4 / 31.4
  vs 5.0 / 3.9 / 10.3 / 9.0; PS94 2.6 / 14.4 / 11.0 / 9.0 vs 4.0 / 5.4 / 12.0 / 6.3; PS95 0.1 / 2.7 / -0.5 / 9.2 vs
  -0.1 / -1.2 / 0.2 / 4.8. Cue: chronic 17.7 / 14.5 / 10.4 / 21.5 vs 4.7 / 9.4 / 7.7 / 3.9. **Chronic: spout identity >
  landing in 4/4, both windows** (PS94 closest). **Pre-stroke: no clear target lead** (PS92 / PS95 ~0, PS93 9.8 vs 5.0,
  PS94 2.6 vs 4.0). Acute mixed. Width sweep 0.5 / 2 queued (O2).
- (Polar, secondary / sensitivity) Pre-stroke target-led in 3/4 (PS93 27.5 vs 6.3, PS94 5.7 vs 2.9, PS95 18.1 vs -0.2
  lick; PS92 tied 5.2 vs 6.1) -- a CAPACITY effect of polar's target set, not seen at matched capacity.
- **By chronic, unique target is above pre in 4/4 in both windows** (cue 35-37 in all four, from 2.7-18.8).
- **The executed rise differs:** PS94's executed part rises as much as target (lick 2.9 -> 22.9, cue 0.9 -> 28.7);
  PS95's stays small; PS92 / PS93 rise less than target. So "the post-stroke rise is TARGET" holds in 3/4 (PS92,
  PS93, PS95), with PS94 rising in both.
- Acute is mixed (target up in PS92 / PS95, down in PS93, flat in PS94); PS95 acute is one session.
- **Polar vs xyz2 (coordinate frame), PS92 + PS93 on matched sessions (27 / 20), cortex, unique target - unique
  executed x1e3 (lick; cue):** PS92 polar -0.9 / 6.2 / 9.1 / 21.5 vs xyz2 -4.0 / -4.9 / -4.3 / 9.0 (cue 1.8 / 2.0 /
  8.2 / 25.6 vs -0.6 / -1.8 / -1.9 / 19.8); PS93 polar 21.2 / 3.5 / 4.5 / 31.1 vs xyz2 14.9 / 3.0 / -2.6 / 2.2 (cue
  6.1 / -3.1 / -7.5 / 12.1 vs 1.8 / -3.6 / -13.7 / -9.4). Executed parts agree across frames; TARGET is smaller in
  xyz2. So the target lead depends on the frame: PS93 pre target-led in both, PS92 chronic target-led in both; PS93
  chronic target-led in polar only. "The post-stroke rise is TARGET" is therefore frame-dependent -- the 4-animal xyz2
  run (`..._xyz2_licks-combined_bins_3d4`, launched 10-24) decides it for the cohort.
  **Why the frames differ:** there are only six spouts, so target regressors take six values and can span at most the
  5-D space of between-spout differences; polar's target set (lateral angle, its SQUARE, tilt, distance) covers more of
  it than linear xyz (it can express midline-vs-side, non-monotonic in ml) -- target gain / free-category gain, chronic,
  PS92 .83 vs .77, PS93 .88 vs .60. The tongue varies continuously lick to lick, so both frames capture executed alike.
  The frame-robust readout is executed BEYOND spout category (pre PS92 15.1 vs 16.2, PS93 9.3 vs 10.8).

#### Two-animal record (PS92 + PS93, 47 sessions, 10-21)

> **Status 10-23:** the 3-D 2x2 check (executed frame xyz2 / polar x lick timing combined / cam4) is RUNNING: xyz2 x
> combined done 10-23 09:32, xyz2 x cam4 40 / 47, polar x cam4 next; results not yet read. It runs on the 10-21
> PS92 + PS93 snapshot; 3-D for all four animals exists since 10-23, so the cohort version is a re-run on 4 animals.

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
**Ceiling (10-21, free per-spout kernels):** a NEITHER component -- spout-locked lick structure matching no pre-stroke
template -- appears post-stroke and persists: chronic 0.38 (PS92) / 0.30 (PS93) of the session's ceiling, > 0 in every
chronic session (pre ~0; PS93 acute 0.78). Target-like > execution-like in PS93; PS92's templates overlap.

### 2.4c Destination vs landing (10b x 3-D landing zones), PS92 + PS93 — PROVISIONAL (two animals)

`scripts/best_match_destination.py` -> `qc/best_match_destination.csv`. Brain: the pre-stroke spout whose activity
pattern each post-stroke spout matches best (figure 10b, cue / working, frozen). Behaviour: where its licks landed (3-D
calibrated zones; cam4 zones FAILED the pre check and are not used). Pre self-match ~1.0 except PS93 far_mid (0.62).
Far contra: PS92 target-like in every session incl. acute (5/5, own-zone landing 16 %); PS93 acute mixed (1 target, 1
execution = far mid, 2 neither = close mid), target from subacute on. PS93 far_mid lands correctly (~0.7) but its
pattern matches pre-stroke FAR CONTRA by chronic (5/6) -- neither. All spouts: target-like 61 / 73 / 90 % (acute /
subacute / chronic), execution-like 4 % acute only, rest neither. **Reading:** post-stroke spout activity resembles the
pre-stroke TARGET pattern far more than pre-stroke execution; deviations are "neither" and fade with recovery.
**CORRECTION (10-23) -- the percentages above are INFLATED; do not quote them.** In 150 / 192 post-stroke rows the
licks landed mostly at the AIMED spout, so a target match is equally an execution match, and every such row counts as
target-like. Only rows where landing departs from the aim can separate the two: 42 rows (PS92 + PS93), target 32 /
execution 2 / neither 8, and most are weak separations (own-zone landing 0.2-0.4, licks short toward the middle spout;
almost all chronic rows); strong ones (own-zone < 0.15): 4 target / 1 execution / 2 neither. The default arm is
CUE-aligned (spout present ~3 s before the cue; acute far-contra first licks often > 1.5 s after it), which favours
target. Re-analysis pending: dissociating rows only, graded by own-zone landing, lick-aligned arm
(`best_match_destination_lick_lick.csv`), a continuous target-vs-landing similarity score, a within-spout shuffle null,
per animal. Status: target-leaning in the separating rows, NOT established. Read with 2.4 (xyz2: executed >= target
pre-stroke in PS92) vs 2.4b (polar: comparable / target-led) -- the coordinate frame decides the pre-stroke answer; the
3-D 2x2 (xyz2 / polar x lick timing) is the check.

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
  **Commanded-axis across-spout slope (10-21 midday):** survives the contact-adjusted spout means (SSp-m bilateral
  deficit z 8.5 cam4 / 8.5 3-D; 3-D hits-only 4.8); the cam4 hits-only collapse = far contra missing from acute
  hits-only sessions. Target-defined (commanded), not execution; executed-axis between null.
  **Z-scored, per-position direction (10-21 afternoon):** each animal in its pre-stroke SD units; pre-stroke, far_ipsi
  (4/4) and close_contra (3/4) show more activity for licks deviating toward the CONTRALESIONAL side; close_contra flips
  acutely 3/4; far_ipsi chronic PS94 reversed, PS92 lost; reliability-corrected, PS95's chronic weakening was
  attenuation (scatter halves). Effect size ~0.2-0.4 SD per SD.

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

- **Frozen decoder vs refit after behavioural recovery (CORRECTED 10-23; was "decoders recover with behaviour, the code
  stays reorganised", 10-20 late; `scripts/imaging_lmm.py`).** The 10-20 line said the frozen-decoder deficit is ~0 at
  full behavioural recovery while refit-over-frozen reorganisation stays positive (z 3.7). **The current output (rerun
  10-21 14:36 after the attempt re-definitions, `qc/imaging_lmm_summary.csv`, post rows, y ~ deficit, cue / working)
  supports only the second half:**
  - frozen deficit F at deficit = 0: **+0.083 (z 2.2) matched / +0.082 (z 2.7) unmatched -- NOT ~0**; per animal
    (sessions from day 11 on, matched / unmatched) PS92 -.02 / .02, PS93 .09 / .06, PS94 .15 / .17, PS95 .09 / .08
    -> only PS92 is back at its pre-stroke frozen accuracy; the refit decoder itself is back (refit deficit -0.01, z -0.4);
  - reorganisation G = (refit - frozen) - pre gap at deficit = 0: +0.093 (z 3.9) matched / +0.092 (z 5.1) unmatched,
    > 0 in 4/4 animals from day 11 (.06-.17 matched, .05-.15 unmatched; last three sessions .08-.22); G does NOT track
    the behavioural deficit (slope z -1.4 / -0.4), so it is roughly flat over the post period; matched time model:
    ~0 at day 1 (z 0.7), rising with log(day) (z 2.2).
  **Status: PROVISIONAL and in tension with the 2026-09-12 withdrawal** (DECISIONS: "reorganisation persists into
  chronic" withdrawn because the deck's animals -> sessions -> BLOCKS bootstrap, Bonferroni over 9 contrasts, puts chronic
  G at +0.07-0.08 with intervals spanning 0; 83 % of the uncertainty between animals). The LMM has no block level and no
  correction across those contrasts, so per that entry's rule it does not overrule the deck. What is new: G is positive
  in each of the four animals individually (rule 8 met as a point-estimate pattern), and the frozen readout is not fully
  back at full behavioural recovery in 3/4. Read as "refit gains over frozen persist in every animal; significance
  depends on the resampling" -- not as established persistent reorganisation.
  All session-level time courses now via `wfield_local/mixed_models.py` (one definition).

- **CORRECTED 10-24 (afternoon) -- the Layer 2 drift-null comparison below is NOT like-for-like; the claim does not
  stand as written.** In the post fit EVERY epoch, pre included, gets its own kernel changes from shared kernels fitted
  over all epochs, and the PRE epoch's own change gain is as large as the post epochs': cortex, v7push (10-23) pre 31.8 /
  20.1 / 32.1 / 35.8 vs acute 25.8 / 40.7 / 14.5 / 45.4, chronic 29.1 / 13.2 / 10.8 / 16.9; final frame (`allsession_
  encoder_v7g3_t3d`) pre 34.8 / 23.0 / 32.1 / 33.7 vs acute 31.8 / 38.3 / 14.1 / 41.9, chronic 31.9 / 14.4 / 9.6 / 18.8.
  The null was fitted on pre sessions ONLY (shared kernels ~= pre kernels), so its pseudo-epoch gain is small by
  construction. Within the same fit, chronic change < the pre epoch's own change in 4/4 and acute > it in only 2/4
  (PS93, PS95). The gain measures each epoch's distance from an all-epoch compromise, not change from pre. Fix to run:
  pre kernels as the reference (deltas for post epochs only) with the null built the same way (pre-late deltas against
  pre-early in a fit that also contains the post sessions). Until then: no Layer 2 claim.
- **Layer 2 all-session encoder: the cortical RESPONSE to behaviour changes beyond pre-stroke drift, acutely AND
  chronically (FIRST RUN 10-23 night, PROVISIONAL; `wfield_local/allsession_encoder.py` at 2fc9fce, grouping v3, frozen
  push; `<E>/allsession_encoder_v7push/`).** Shared kernels + per-epoch kernel changes, scored on held-out sessions;
  gain = held-out R2 of (shared + changes) - (shared only), x1e3; null = same fit on pre sessions with the later half as
  a pseudo-epoch. Cortex, per animal (PS92 / PS93 / PS94 / PS95):
  - total change: acute **25.8 / 40.7 / 14.5 / 45.4** (PS95 n = 1), chronic **29.1 / 13.2 / 10.8 / 17.0** vs null
    pre_late **0.9 / 0.9 / 9.0 / 3.3** -> above null 4/4 in both epochs (PS94 chronic marginal: 10.8 vs 9.0);
  - per group (unique change gain), acute vs null: within-spout direction (exec_within) 5.2 / 5.0 / 2.7 / 1.5 vs
    <= 0.1; target 4.1 / 5.5 / 1.9 / 4.5 vs <= 0.3; lick 0.7 / 0.2 / 1.6 / 8.2 vs 0.6 / -0.5 / -0.3 / 0.5 (3/4;
    PS92 ~null). Chronic: exec_within 3.0 / 2.1 / 0.4 / 0.7, target 5.0 / 2.9 / 1.1 / 2.9, lick 3.2 / 0.7 / 1.4 / 3.0,
    all above null 4/4 (PS94 exec_within small). "other" (running / push / nose) is noisy (PS94 null 7.0);
  - contralesional SSp-m: acute 12.1 / 75.3 / -5.6 / -24.9 vs null 4.0 / 1.7 / 2.3 / 6.1 -> only 2/4 (PS94 / PS95
    below the shared fit); chronic 16.4 / 8.5 / 23.1 / 9.3 -> 4/4.
  **Reading:** on top of Layer 1 (behaviour changes carry the acute far-contra drop; small residual), the mapping from
  behaviour to cortex does change -- most in the target and within-spout direction responses -- and the change
  persists chronically after behavioural recovery. This is the encoding-side counterpart of the decoder's
  refit-over-frozen G > 0 in 4/4 above. Sizes are small (gains of 1-5e-3 R2 per group). Caveats: the null has 4 + 4
  sessions vs 1-12 per post epoch; kappa is chosen separately per fit (post 3-30, null 10-100); no block-level
  resampling yet; a summary bug (pandas read the fit label "null" as missing) hid the null rows on the first print --
  fixed in `summarise`.

### 2.6 Lick-level strays vs on-target licks — FIRST RUN 10-23: NOT DECISIVE for far contra (reference fails validation)

**First full run (10-23 night, `wfield_local/stray_licks.py` at 4bc7687, 104 / 107 sessions, 4 animals, all licks with
3-D; `<E>/stray_licks/`; PRELIMINARY).** Per session one regression of all frames on lick-event kernels (stray q->j,
non-contact near-miss q, ITI near j, contact|q; cue / arrival kernels; drift + running / push nuisance); kernels pooled
count-weighted per animal x epoch; the spout-specific part (minus the mean over spouts) of a stray q->j kernel is
correlated with the PRE references near|q (r_target: same aim) and near|j (r_exec: same landing).
- **Validation (pre, split-half near-miss kernels, own spout vs others): passes 16 / 18 spouts, but FAILS for
  far_contra in PS92 (r_own .04 vs others .22) and PS93 (-.62), and PS95 has no far_contra near-misses at all**; only
  PS94's far_contra reference is reliable (.50 vs -.04). Non-contact near-misses at far contra are rare pre-stroke (the
  animals touch when they get near). So r_target for far_contra strays is not trustworthy in 3 / 4 animals.
- **Pre-stroke strays mostly look like where the tongue LANDED** (r_exec > r_target in most stray types: PS92 / PS93 /
  PS95 weighted target_minus_exec -0.50 .. -0.67; PS94 -0.25). Caveat: pre strays and pre references come from the
  same sessions (partly circular). Reading: the spout-specific part of cortex's lick response follows the executed
  movement more than the aim, in the intact animal.
- **far_contra strays, pre -> acute (target_minus_exec):** -> far_mid PS92 -.39 -> -.76, PS93 +.21 (n = 20) -> +.29,
  PS94 +.93 -> +1.01, PS95 no target reference; -> close_mid PS92 -.72 -> -.55, PS93 -.97 -> -.52, PS94 -.11 -> .00.
  No consistent shift toward either answer; PS94 (the one valid reference) is target-like before AND after.
- **Contact vs non-contact lick at the same aimed spout** (`contact_vs_near.csv`, summed 0-0.5 s, cortex): pre mostly
  NEGATIVE (a touching lick evokes less than a non-touching one at the same spout; matches the negative contact kernel,
  3.4); acute / chronic mixed in sign and noisy (few non-contact references).
- **Verdict:** the stray test does not resolve A ("aims far contra, execution off") vs B ("re-aimed at far mid") for
  far contra, because the far-contra aim reference cannot be estimated from non-contact licks. Possible next step (not
  run): use contact|far_contra (on-target, touch-confounded) as the aim reference with the pre contact-minus-near
  difference from the other spouts subtracted, or rely on the relabel test (running).

- **Secondary rescue (Priya 10-23, option 3: try it, relabel stays primary) -- FAILS ITS CHECK, not used.** Aim and
  landing references rebuilt from contact licks: contact|q minus the touch response (contact - near) averaged over the
  OTHER spouts (`touch_corrected`, `stray_index_touchcorr.csv`, `stray_validation_touchcorr.csv`). Split-half
  reliable (own .23-.99 vs others <= .04: contact kernels are high-n), but where a near-miss reference exists the
  corrected reference does NOT resemble it (r_vs_near_ref mostly NEGATIVE, -0.3 .. -0.98; PS95 -0.83 .. -0.98; only
  close / far contra positive, .01-.54). So the equal-touch-response assumption fails (part of the negative r is
  mechanical: the leave-spout-out average contains the other spouts' near kernels, whose spout-specific part is
  anti-correlated with q's). Its far-contra table (e.g. PS95 "target-like" before and after, driven by very negative
  landing r) is an artefact of this and is NOT interpreted. **The stray test does not answer A vs B; relabel is the
  test.** Kept from 2.6: in the intact animal the spout-specific lick response follows the landing more than the aim.

**Original design notes (10-23, before the run):**

Why this test: on an on-target lick the TARGET and the spout-locked executed lick are the same variable (pre-stroke
per-spout contrast maps, 3.4: the spout part tracks the data, the target / exec_locked split is unstable, e.g. close_mid
+13 vs -13), so only licks that land away from the aimed spout can separate them (Priya 10-23: "don't we want to be
comparing LICK stray response vs on-target response"). The pre-stroke arm is a result in itself (target vs executed in
the intact animal).
- **Census (3-D, trial licks, August pre; `pose3d_kin_licks_snapshot_20261021`):** thousands of no-contact licks land
  nearer another spout's CALIBRATED centre (median of that spout's contact licks; the nominal spout angles are biased
  ~21 deg), pre-stroke included (PS92 1,129, PS93 6,679 pre). Close-spout centres overlap, so the error is treated as
  continuous (in within-spout SD units, plus direction toward the nearest other centre), not as zones.
- **Design (queued):** per-lick regressor sets (on-target at q; strays from q toward j) inside one model, so corrective
  licks overlapping in time are separated by the fit (raw stray averages are contaminated toward the target); compare
  stray kernels' maps with the pre on-target maps of q and j, controlling pre q-j similarity; pre AND post.
- **Incomplete data:** the census used the 10-21 snapshot (PS92 + PS93, trial licks); 3-D for all four animals exists
  since 10-23 -> run the test on 4 animals (PS95 acute still missing). Per-lick 3-D for pre-cue and spontaneous ITI
  licks landed 10-23 (`kinematics_all/licks.csv.gz`, `context` column, key `lick_uid`): ITI licks (PS92-95 20k / 25k /
  30k / 9k) have no target, so they serve as the execution-only reference (direction relative to the calibrated spout
  centres); pre-cue licks (spout parked) add strays with a defined target.

### 2.7 Relabel test: which spout's pre-stroke state the far-contra trials are in — FIRST RUN 10-24: HETEROGENEOUS

`wfield_local/frozen_relabel.py` (f27339a), `<E>/frozen_relabel_v7/` (107 / 107). Frozen pre-stroke model (v7 frame);
each spout q's trials re-predicted as every other spout j with the EXECUTED licks fixed (only the target + spout-locked
typical-movement parts move); cost = summed SSE increase over the 41 parcels / summed SST (x1e3; > 0 = true label fits
better). Pre sessions are predicted by leave-one-session-out fits. PRELIMINARY.
- **Validation passes fully:** pre, the true label beats every relabelling in 100 % of (q -> j) pairs in all four
  animals and all windows (all / cue / lick). The far spouts are now separated pre (far_contra vs far_mid 27 / 44 / 22 /
  420 for PS92-95, lick window), unlike the single-session smoke.
- **far_contra trials, lick window, margin vs far_mid (pre -> acute -> subacute -> chronic):** PS92 27 -> 90 -> 17 -> 12;
  PS93 44 -> 28 -> 30 -> 34; PS94 22 -> 5 -> **-28** -> -8; PS95 420 -> **-148** (n = 1) -> **-167** (n = 7) -> 209.
- **Best-fitting label per session for far_contra trials** (count of sessions), with the other spouts as the control
  for a general loss of spout structure:
  - **PS95 -> re-aimed at far mid (B), specific:** acute 1/1 and subacute 5/7 best = far_mid, while its other five
    spouts keep their true label (5-7/7); recovers chronic (far_contra 7/11).
  - **PS93 -> stays far contra (A):** acute 4/4 and chronic 10/12 best = far_contra (pre only 3/7).
  - **PS92 -> far-contra state lost, not toward far mid:** acute best = far_ipsi 4/5 and the far_mid margin GROWS (27 ->
    90); but PS92's acute frame is confused for other spouts too (close_ipsi 1/5, far_ipsi 1/5 true) -> not specific.
  - **PS94 -> nonspecific:** acute far_ipsi 3 / far_contra 2 / far_mid 1, subacute far_mid 2 / close_contra 2; other
    spouts equally mixed (close_ipsi -> close_mid 4/6, far_mid -> far_contra 3/6).
- **Reading:** no single answer across animals. One animal (PS95) shows the "re-aimed at far mid" pattern cleanly and
  transiently; one (PS93) keeps the far-contra state while executing badly ("aims far contra, execution off"); two lose
  spout-specific structure broadly. Combined with Layer 1 (behaviour carries the drop) and the stray test (lick responses
  follow the landing), the far-contra deficit is not one mechanism in all four. Caveats: PS95 acute is one session; a
  general shrink of all margins (Layer 2's changed mapping) can move best labels without re-aiming -- hence the
  per-spout control; executed-lick templates are spout-locked, so "state" = target + typical movement, not aim alone.

### 2.8 Failed attempts on the frozen pre-stroke decoder (video attempt classes) — FIRST RUN 10-24: "distance kept, side lost" on failed attempts; no pre-cue plan signature

`wfield_local/attempt_coding.py` (goal-question item 1; FINDINGS 7). LP source (rebuilt attempts table, complete cam1),
joint-LocaNMF features, the STORED DAQ-frozen pre-stroke decoder (1.0 s ENL models frozen 10-24 on the LP pool's
sessions; PS93 = 10 pre sessions, PS93_0813 has no video), default June + August pre. Classes per far-contra trial,
acute: hit (n 15 / 35 / 13 / 1, PS92-95), incomplete (257 / 196 / 251 / 39), no attempt while working (27 / 27 / 78 /
22), reaction (jaw only), stopped (after the video-attempt quit; PS95 none acute). Readout p_true = frozen-decoder
probability of the true spout (chance .167); within-session permutation contrasts. Outputs `E/attempt_coding/`
(`summary_*`, `contrasts_*`, `far_ring_share_contrasts.csv`).

- **Pre-stroke reference (LOSO):** far contra p_true ENL .22 / .27 / .45 / .30, cue .86 / .76 / .88 / .93, lick .91 / .86
  / .88 / .92.
- **Failed attempts lose the far-contra code** (acute incomplete, cue .24 / .08 / .38 / .33, lick .14 / .13 / .43 / .22),
  to the level of the no-task floor in PS92 / PS93 (vs stopped: cue .24 vs .23, .08 vs .10; lick .14 vs .10, .13 vs .18);
  PS94 keeps part (lick .43 vs stopped .17, p < .001). The few acute hits score higher than failed attempts in PS93 /
  PS94 (cue .40 vs .08, .69 vs .38).
- **But failed attempts are decoded as a FAR spout -- distance kept, side lost.** Share decoded as any far spout,
  incomplete vs stopped / no-attempt in the same sessions, LICK window: PS92 .58 vs .41 (p .06), PS93 .68 vs .34
  (p .001), PS94 .90 vs .49 (p < .001), PS95 .85 vs .18 vs no-attempt (1 session) -> 4/4 direction, 3/4 p < .01; CUE
  window 3/4 (PS93 .80 vs .48, PS94 .83 vs .65, PS95 .87 vs .55; PS92 none). Mostly decoded as far mid / far ipsi
  (PS93 far ipsi .41-.50). Same confusion as the 10-21 LP frozen decoders, now shown to belong to FAILED ATTEMPTS
  specifically, not to quitting.
- **No pre-cue plan signature:** in the ENL window incomplete ~ stopped in p_true (PS92 / PS93 / PS94 +.02 / +.01 /
  +.09) and far share (PS94 only, .72 vs .50, p < .001) -> nothing consistent before movement.
- **Not yet separable from execution.** The cue / lick windows contain the failed attempt's own tongue movement; post-stroke
  far-contra licks fall short and drift ipsi, and tongue protrusion is spatial (scales with spout distance; 10-22 audit),
  so "far, wrong side" may be the EXECUTED movement rather than a degraded target. Stopped / no-attempt trials have no
  movement, so the far-share contrast is movement-present vs absent. Deciding test = item 2 (binned contest on O2) and a
  movement-matched comparison (failed far-contra attempts vs pre-stroke / acute licks with the same 3-D landing).
  The per-spout projection (`own`) is unstable post-stroke (see module docstring) and is not used.

### 2.9 What failed attempts look like (trial-type maps) — FIRST RUN 10-24: same bilateral pattern; lower amplitude vs hits (3/4) but NOT vs touch-matched pre-stroke misses (2/4)

`wfield_local/attempt_maps.py` (goal-question item 4): 15r's rest-referenced maps split by video class, per animal,
session-weighted means (>= 5 trials per session x spout x class). `E/attempt_maps/` (`maps_*.npz`, `counts_*`,
`hemispheres_*.csv`, `figures/attempt_maps_<window>_<spout>[_minus_pre].png`). Amplitude below = mean over the brain
mask in each half, / the same animal's pre-stroke far-contra hits (ipsi | contra hemisphere).

- **Lick window (aligned to the trial's own first video lick), acute far contra:** failed attempts (incomplete) PS92
  .40 | .45, PS93 .38 | .42, PS94 .57 | .74, PS95 .90 | 1.25 -> weaker in 3/4, bilateral, SAME SPATIAL PATTERN (lateral
  SSp-m / MO / ALM, both hemispheres). The few acute HITS are full size or larger (PS92 1.77 | 2.09, n 10; PS93 1.27 |
  1.10, n 31; PS94 1.21 | 1.47, n 8). This splits the 15r LP pooled acute amplitude (0.70, 3.3): hits large, failed
  attempts small. Subacute failed attempts stay low in PS92 / PS93 (.25-.36), not in PS94 / PS95 (1.0-1.7).
- **Cue window:** failed attempts .29-.52 of pre in PS92-94 (PS95 .83 | 1.12); jaw-only REACTION trials are large in
  PS92 / PS93 (1.35 | 2.12, 1.11 | 1.13) -- read with the late-start caveat (acute PS92 / PS93 far-contra licks start
  > 1.5 s after the cue on 35-38 % of trials, so much of the cue window precedes the failed lick); stopped trials
  variable (PS93 ~0, PS92 .72 | 1.0, PS94 .75 | 1.21).
- **Touch / reward control (re-collected 10-24 with `pre_incomplete` = pre-stroke attempts that missed the spout, trial-
  weighted over sessions; far contra 7 / 27 / 30 / 20 trials, 4 / 8 / 10 / 8 sessions).** Pre-stroke misses are NOT
  uniformly smaller than hits (lick window, ipsi | contra: PS92 1.32 | 1.67, PS93 .41 | .49, PS94 1.35 | 1.67, PS95 .57 |
  .73). Acute failed attempts / pre-stroke misses: PS92 .30 | .27 (reference n 7), PS93 .92 | .85, PS94 .42 | .44, PS95
  1.58 | 1.72 -> smaller in 2/4 (PS94 well sampled; PS92 thin), unchanged in PS93 (its whole drop vs hits = missing
  contact), larger in PS95. Cue window the same pattern (.21 / .65 / .33 / 1.57 of pre misses). **So "failed attempts are
  weaker" does not survive the touch-matched reference at >= 3/4; the reduction vs hits is partly touch / reward / movement
  size.** Caveats: pre-stroke misses are a selected set (PS93's are its pre-existing leftward-reach misses), and few.
  Next: the frozen encoder's touch / lick split (item 3) on these classes. First-pass maps (no control) kept in
  `attempt_maps/v1_20261024/`.

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

### 3.3 Epoch change of the position activity maps, LP lick attempts — PROVISIONAL (10-21 night)

> **RE-RENDERED 10-24 on the rebuilt attempts table + complete cam1** (15r / 14m / acc / 5c / 5cr, all five arms,
> `G/LP_licks/epoch/`; LP trial counts re-run first; PS93 frozen models on its 10 LP-available pre sessions). The frozen
> decoders did not move: acute far contra lick .31 (10-21 .32), far ipsi .58 (.58), far mid .51 (.51); cue (working)
> far contra .29, far ipsi .45, far mid .40 (pre .88 / .85 / .82); ENL far contra .21 (pre .40). Map numbers below are
> still the 10-21 ones until re-read from the new figures; LP null calibration not yet re-run.

Family 15r (trial-average position maps, no model) re-gated on VIDEO lick attempts (LP_licks: cam4 + cam1 licks,
trial = hit / incomplete attempt within max_rt, lick-aligned = first LP lick onset, ENL lick-free window 1.0 s) instead
of spout contact. `labcams/grant_figures/LP_licks/epoch/` (corner-stamped); calibration `.../LP_licks/epoch/results/`.
Numbers: % of 2,022 bins flagged by the nested bootstrap (pre-normalised), rest reference; amplitude = acute / pre.
- **Acute far contra is now measurable around the lick.** Contact gating left 0 acute sessions above the 20-trial
  floor; LP gives 4 animals / 15 sessions / 778 trials. Post-lick activity is LOWER than pre (amplitude 0.70; 19.7 % of
  bins; restw 17.8 %, precue 17.3 %), against a June-vs-August drift of <= 1.4 % at that position. Every other position
  goes UP acutely (near ipsi 1.50, near contra 1.34, near mid 1.28). Per animal (rest ref, mean change / pre mean,
  ipsi | contra hemisphere): PS92 -55 | -49 %, PS93 -48 | -48 %, PS94 -38 | -27 %, **PS95 +2 | +36 %** -- 3/4,
  bilateral; PS95's acute cell is ONE session / 37 trials. The pooled outline looks ipsi-dominant only because PS95's
  contra increase cancels the others' contra drop: per animal the drop is bilateral.
- **The cue-window drop is unchanged by the gate** (far contra 60 % of bins, amplitude 0.48 under both; 4/4 animals,
  -40..-63 %, bilateral). So acutely the far-contra trial is about half as active from the cue on, and stays ~30 %
  below pre after the attempt.
- **Contact gating had inflated acute post-lick amplitude at the far spouts** (it kept only the licks that reached the
  spout): far mid 1.33 -> 1.03, far ipsi 1.40 -> 1.25 (9.5 -> 3.1 % of bins); subacute far contra 12.4 % -> 0.8 %
  (1.18 -> 1.01). Near spouts and all chronic cells barely move (chronic far contra 15.0 -> 16.4 %, amplitude 1.03).
- **Chronic:** far contra back to pre amplitude (1.03) in PS92-PS94 (per animal within +-7 %); PS95 +16 % contra.
  The chronic difference maps share one bilateral frontal increase / posterior-medial decrease at every position.
- **The acute far-contra drop is the EXECUTED MOVEMENT (10-22, LP-gated encoder, 4/4).** Movement-first encoder
  rerun on attempted trials with cam4 + cam1 lick timing (`movement_first_subset-drop_licks-combined_gate-attempted`,
  119 sessions; cam4 only for PS95 + PS94_1005, no cam1), full-model prediction split by group, evoked = 0.5-1.5 s
  after the cue minus -0.5-0 s (`scripts/lp_encoder_decompose.py` -> `qc/lp_encoder_decompose_by_animal.csv`).
  Acute minus pre, far contra, x1e-3 (data / movement = instructed / task / position part / residual):
  cortex PS92 -25.8 / -22.0 / -2.9 / -0.9 / +1.2; PS93 -13.0 / -12.9 / -1.6 / -2.2 / +2.2; PS94 -4.5 / -4.1 / -3.9 /
  -3.7 / +2.4; PS95 -2.6 / -7.8 / +4.3 / +4.8 / -0.4 (1 session). Contralesional SSp-m (left): -50 / -29 / -24 / -23
  in the data, -51 / -26 / -23 / -24 in the movement terms, |task| <= 3, |residual| <= 2.4, 4/4. The video-seen
  licks / tongue / jaw are smaller or absent, and the model's movement terms account for the whole drop; the
  position-specific part and the residual are small. Other positions go UP acutely, also mostly via movement terms.
  Chronic: data and movement return to or above pre together (PS92 / PS93 SSp-m +16, movement +12 / +17).
  CAVEATS: the partition is the ridge fit's -- task is penalised hardest (alpha 1e5 vs licks 1e4, orofacial 1), so
  variance shared between target and movement goes to movement (the movement-first convention, 2.x); it says the
  drop needs no target term, not that no target signal is lost (3.1's drop-p, far contra ~0 acutely, is that test).
  Cue-aligned over attempted trials, not LP-onset-aligned like the maps (PS95's map increase is not seen here).
  **CONTACT IS INSIDE "MOVEMENT" HERE (caught 10-22):** the encoder's `licks` block holds the spout-CONTACT kernel too,
  and acute far contra has few contacts, so part of this "movement" share may be missing touch / outcome activity,
  not weaker licking; the movement terms also carry lick DIRECTION (tongue x / y, jaw x, angle), which is target-
  correlated. Also: penalties are chosen per session by CV (task usually 1e4-1e5, lick 1-1e6), not the fixed values
  quoted above. Do not read as "movement" until the frozen encoder's contact / direction split
  (`wfield_local/frozen_encoder.py`, FGROUP) is in.
- ENL window: see the matched 1.0 s comparison below.

**Frozen pre-stroke decoders, per-position accuracy (10-22 ~01:10; `acc`, chance 0.167).** LP: `LP_licks/epoch/data/
epoch_acc_by_position_*`; matched contact 1.0 s ENL arm: `DAQ_ENL1s/epoch/`. Pre = per-animal held-out; post = frozen.
- **Lick window: contact gating OVERSTATED acute far-spout decoding.** Acute far contra 0.50 [0.26, 0.75] (contact; 3
  animals, 1-3 sessions each) -> **0.32 [0.17, 0.51] (LP; 4/4 animals, 16 sessions)**; per animal pre -> acute
  PS92 .92 -> .26, PS93 .85 -> .16, PS94 .90 -> .49, PS95 .94 -> .35. Far ipsi .72 -> .58, far mid .71 -> .51 (now
  significant). Near spouts unchanged (.73-.78 both). Pre far spouts .03-.05 lower under LP (non-contact attempts
  included). Chronic unchanged (far contra .75 / .76).
- **Cue window: the gate does not matter** (every cell within .05; acute far contra .32 / .29).
- **ENL: the lick definition does not matter at 1.0 s; the window length does.** Contact 1.0 s and LP 1.0 s agree within
  .03 in every cell. Both are lower than the 2 s contact window at EVERY position and epoch (pre near ipsi .75 -> .59,
  far contra .51 -> .39-.40): half the averaging window, less signal -- not lick contamination of the 2 s window, since
  removing LP-detected licks changes nothing at matched length. The acute pre-cue loss survives every definition:
  LP 1.0 s acute far contra .21 [.12, .29], 4/4 animals down (PS92 .29 -> .15, PS93 .33 -> .16, PS94 .54 -> .30,
  PS95 .40 -> .14; three at chance).
- **ENL maps (15r, acute - pre, far contra) DO move with the lick definition at matched length:** rest ref 17.2 % of
  bins, amplitude 1.44 (contact 2 s) -> 9.9 %, 1.67 (contact 1 s) -> **4.4 %, 1.45 (LP 1 s)**; restw 17.4 -> 9.1 ->
  3.1 %. So part of the acute far-contra pre-cue INCREASE in the contact-defined window is non-contact licking in the
  last second (LP removes those trials; >= 95 % kept, so power is similar). Other positions <= 8 % under all three.
  Extent is descriptive (see below), but the direction is consistent across both rest references.

**How to read the outlines (all map families).** Nested-bootstrap outlines (black) are bin-level family-wise
significant: they localise -- each outlined bin changed. Their EXTENT is a lower bound that depends on power (n = 4
animals), not a measured area; an unflagged bin is not evidence of no change, and "19 % vs 6 % of bins" is descriptive,
not a test of a difference. Cluster-permutation outlines support only "a change exists" (cluster-level control; false
clusters are large in area), not where it stops.

### 3.4 Frozen pre-stroke encoder (v4 -> v6 -> v7): the acute far-contra drop is a LICK change, not a target change — PROVISIONAL (10-22 afternoon; robustness updated 10-23)

> **FINAL FRAME 10-24 (`frozen_encoder_gate-attempted_v7g3_t3d`: 3-D tongue, rebuilt attempts gate, combined licks with
> complete cam1, grouping v3, 107 sessions).** The acute far-contra decomposition is unchanged: lick part, lick window,
> SSp-m_left -29.0 / -19.4 / -19.0 / -24.3 (v7pushfree -28.0 / -19.6 / -19.3 / -24.9), cue window -18.0 / -11.7 / -18.6 /
> -28.0; target <= 3.6 everywhere; contact +5..+9 (PS93 ~-1). `E/qc/lp_encoder_decompose_frozen_encoder_gate-attempted_
> v7g3_t3d_joint[_lick]_by_animal.csv`. Relabel, Layer 2 and basis maps in this frame are running.

> **Current version: v7** (STATS_VERSION 7, commit 73d2991, 10-23: forward-only running from 1.5 mm/s, no push term).
> Areas: `E/frozen_encoder_gate-attempted_v7fwd15/` (same model, run before the default changed); pixel maps + pixel /
> per-position unique R2: `_v7basis` RUNNING (started 10-23 16:36). v6 folders kept for comparison.
> Incomplete: PS95 1 acute session; PS94 / PS95 lick timing cam4-only in these runs (their cam1 not yet used); no statistical test on data - model; still
> epoch-based (not on the LMMs, 4.3.7); lick-direction parts (exec_within / exec_locked) to be re-run with full 3-D.

Model (`wfield_local/frozen_encoder.py`, STATS_VERSION 4; DECISIONS 2026-10-22 afternoon): one ridge model per animal
fitted on its August pre sessions and applied unchanged to every session (Allen-area targets). Every lick-related
movement trace is split into the animal's PRE-STROKE mean trace for that spout (cue-locked before the trial's first
post-cue lick, lick-locked after; pooled over its pre sessions) and the deviation from it. Groups: timing, target
(position blocks), exec_locked (the spout-locked part: executed but fully predictable from the spout), exec_within
(direction deviation within a spout), LICK (deviation of lick timing / tongue speed / jaw opening: vigour that is not
"which spout", and any post-stroke change at a spout), other (nose y, running), contact. Output
`E/frozen_encoder_gate-attempted_v4/`. Leakage (spout decoding, chance .17, templates leave the tested session out):
target / exec_locked .99-1.00; lick .38-.40 (v3 "directionless" .60-.69); with each session's OWN template lick
.04-.12 -> the residual .4 is day-to-day departure from the animal's typical licking at a spout, by design part of
"lick" (Priya chose the pooled template: "let's do pooled").

**Acute far contra, post-lick evoked response (0-1 s vs -0.3 to -0.1 s from the first post-cue lick), cortex, change
vs pre, dF/F x 1e3, per animal (PS92 5 / PS93 4 / PS94 6 / PS95 1 acute sessions):**
- data -14.1 / -9.3 / -1.0 / -0.1; frozen prediction -15.5 / -14.4 / -2.5 / -7.5;
- **LICK -20.0 to -20.9 / -12.1 to -14.0 / -13.5 to -14.0 / -19.9 to -20.4** (range over joint, target_first,
  exec_first, task_first, movement_first: robust to the credit order);
- **target** -1.6 to +1.3 / -1.7 to -1.1 / -0.6 to +0.8 / -0.1 to +0.7 (target_first gives it every bit shared with the
  spout-locked execution: still <= 1.7);
- contact +1.7 to +10.3 (opposes lick), other -0.7 to +4.6. **Contact kernel sign CHECKED 10-23 (v7): NEGATIVE in 4/4**
  (cortex sum over lags -21 / -47 / -42 / -46 x1e3 PS93 / PS92 / PS94 / PS95; strongest MOp / MOs; PS93 SSp-m the one
  positive cell): with lick timing in the model, a lick that touches is followed by LESS frontal activity than one that
  misses. So the + contact part = FEWER contacts x a negative kernel (acute far contra touches on 2-15 % of trials),
  not more activity per contact; lick and contact partly offset (PS92 cortex -22 / +9). A frozen part cannot show a
  per-event response change -- that would sit in the residual (mildly +; PS95 1 session strong). Direct test pending
  (model-free contact vs miss licks, pre vs post; or a per-epoch contact delta in the all-session model).
  PS93's unique contact is ~0 (lick window 1.2 vs 13-32 x1e3) although it misses most (0.74 of trial licks touch vs
  0.91-0.95): likely shared variance. **Cheap check (10-23, `scripts/contact_direction_predictability.py`, August pre
  3-D trial licks, session-fold CV AUC of contact):** direction alone .84 / .85 / .75 / .62 (PS92 / PS93 / PS94 /
  PS95), reach alone .77 / .85 / .68 / .83, both .86 / **.89** / .79 / .84 -- PS93's contacts are the MOST predictable
  from how the lick was executed (direction AND reach), and the order of predictability runs nearly inverse to unique
  contact (1.2 / 13.3 / 18.1 / 31.6 for PS93 / PS92 / PS94 / PS95; n = 4, PS94 / PS95 swapped). Reading: PS93's
  touch response is not absent but SHARED with the lick (reach -> lick group) and direction (exec_within) terms, so
  unique credit is ~0. **Joint-drop check DONE (grouping-v3 refit `_v7g3`, 10-24; pre, cortex, held-out): the
  contact / within-spout direction SHARED part = unique(both) - unique(contact) - unique(exec_within) is ~0 in PS93
  (+0.0 .. +0.2 x1e3 across cue / lick x frame / trial) and in the others (-4.7 .. +0.8)** -> PS93's missing contact
  credit is NOT shared with the direction group; that explanation is withdrawn. Remaining candidate (untested): shared
  with the LICK group (reach -> lick magnitude; reach alone predicts PS93 contact at AUC .85, the highest), or PS93's
  touch response is simply small. A `no_contact_lick` joint drop would test it;
- data - frozen prediction +1.3 to +7.5 (small, positive).

Reading: the pre-stroke mapping, fed the licking actually made, predicts the drop; the change is carried by licking
that departs from the animal's typical pre-stroke licking at far contra (fewer / weaker / differently timed licks), not
by the target (spout) response; cortex responds about as predicted for that licking (no extra per-lick loss).
Encoding is correlational: it shows the licking-cortex coupling is preserved, not which drives which. Area maps
(`E/frozen_encoder_gate-attempted_v4/figures/area_maps_<variant>_far_contra_<epoch>_<align>.png`): the drop and the
lick prediction share a bilateral, lateral-somatosensory / motor pattern; target blank; data - model mildly positive at
the contralesional lateral edge. Open: no test on data - model yet; PS95 1 acute session; PS94 data barely changes
(lick -14 offset by contact / other); **PS94 chronic (and partly subacute) sessions are not fit by the frozen model**
(all-frames R2 -0.4 to -0.6 vs per-session 0.50-0.62) because of the "other" group (without it +0.11 / +0.12) --
running or nose-y changed after the stroke; PS94 chronic frozen numbers are not to be read until resolved.

**v6 (10-22 night): RUNNING CAP resolves PS94; the headline is unchanged.** Cause: PS94 ran far harder after the stroke
(trial-frame |speed| p99 14 / 19 / 22 mm/s acute / subacute / chronic vs 9.4 pre; 7-9 % of chronic frames above its
pre running-speed p99) and the linear running kernel extrapolated. v6 clips running at each animal's pre p99 of
RUNNING frames (> 3 mm/s; PS92-95 14.8 / 17.3 / 13.2 / 16.8 mm/s). Joint frozen R2 cortex (all / cue / lick), v4 -> v6:
PS94 subacute -0.09 / -0.26 / -0.27 -> +0.28 / +0.02 / +0.00, chronic -0.44 / -0.12 / -0.11 -> +0.30 / +0.18 / +0.18;
PS92 / PS93 / PS95 within 0.01 in every epoch. Acute far-contra lick-window parts within 0.2 of v4 in all five
orderings (lick -20.9 / -12.7 / -14.0 / -20.0). (A first cap at the all-frame p99, v5, broke PS95 -- it runs on 1 % of
pre frames, cap 2.8 mm/s -- chronic cue-window R2 -0.65; superseded.) PS95 subacute / chronic cue / lick-window R2 stays
near 0 (as in v4; no_other is better there -- open, likely nose y). **Pixel maps** (per-animal 200-component basis,
`E/frozen_encoder_gate-attempted_v6basis/figures/pixel_maps_<variant>_far_contra_<epoch>_<align>.png`; 8 x 8 px grid
check for PS92, `..._v6grid/`): acute far contra, the data drop is bilateral, strongest over lateral SSp / MOp; the lick
part reproduces it in all four animals; spout / target / exec_locked faint; contact opposite in sign; residual small
and patchy. Basis and grid agree for PS92.

**Running model (10-23; areas; `_v6spl`, `_v6thr3`, `_v6thr1`; v7 candidates running).** The cortical response to
running is SUBLINEAR, close to on / off: a free piecewise-linear curve rises steeply to ~3-5 mm/s and is flat or
slightly falling above, in all four animals (the linear kernel keeps climbing, ~3x higher at 14 mm/s). Held-out pre R2
(all frames) linear -> curve +0.03 .. +0.10; but the curve's steep part lies below the 3 mm/s running threshold and
made later post-stroke cue / lick windows worse in PS92 / PS93 / PS95. Response only above 1 mm/s (forward hinges;
`_v6thr1`) is better than linear in every animal x epoch x window (PS94 chronic all-frames .30 -> .63, PS95 chronic
cue window .00 -> .15; held-out pre +.04 .. +.06); above 3 mm/s throws the 1-3 mm/s response away (pre no better than
linear). The linear v6 used SIGNED speed: backward movement was negative running, although cortex is HIGHER then.
QC (`scripts/qc_running_speed.py`, side-camera clips `scripts/qc_running_clips.py` -> `E/qc/running_clips/`): forward
speed is bimodal (trough ~1.5-3 mm/s); post-cue biphasic treadmill jolts (backward dip then a small forward bounce,
< ~1 mm/s forward) = upper-limb extension / posture change (Priya, clips); negative speed at bout ends = a paw pushed
forward (real, not belt backlash). Speeds < 1.5 mm/s sit within +-0.5 s of a cue / lick at 1.5-2.5x chance pre-stroke;
1.5-3 mm/s at chance pre in 3/4 (PS93 not), but lick-associated post-stroke (chronic .34-.51 vs chance .15-.31)
(`scripts/qc_running_lick_jitter.py`). -> v7 candidates: forward-only running from 1.5 mm/s (Priya), without / with a
separate PUSH variable (backward speed beyond 0.3 mm/s). **Headline unchanged in every running model** (acute
far-contra lick part within ~1 of v6; target <= 1).
**Side-by-side (10-23, `scripts/frozen_running_compare.py` -> `E/qc/frozen_running_compare_v6_to_v6thr1.csv`; cortex,
per animal; six models incl. v7 candidates: `..._v6_to_v7fwd15push.csv`):**
- fitted running response (x1e3, summed over lags) at 1.5 / 3 / 5 / 12 mm/s: linear 11-15 / 22-30 / 36-50 / 86-119 and
  -22 to -30 at -3 mm/s (backward = negative running); forward from 1 mm/s 10-16 / 41-64 / 75-119 / 51-80, i.e. peak
  near 5 mm/s then lower, in 4/4;
- unique dR2 of "other" (running + nose y), lick window, chronic, linear -> forward-from-1: PS92 246 -> 288, PS93 184 ->
  250, PS94 243 -> 476, PS95 -68 -> +82 (the PS95 post-stroke window problem was the running model);
- unique dR2 of lick, all frames: up to ~15 lower than linear (running no longer absorbs as much shared variance in
  the linear direction), same epoch pattern in every animal;
- acute far contra, lick window, contralesional SSp-m: lick part PS92 -27.4 to -28.2 / PS93 -19.4 to -19.8 / PS94 -19.1
  to -19.4 / PS95 -22.7 to -23.0; target +3.8 / -0.2 / +1.8 / +2.3 in all four running models.
**v7 chosen (10-23 evening).** Six models side by side (`_v7fwd15` = forward from 1.5 mm/s, `_v7fwd15push` = + push):
- forward from 1.5 vs from 1 mm/s: within .02 R2 in every animal x epoch x window (1 mm/s marginally higher); the
  lick group's unique dR2 is not lower with the 1-1.5 band in, so the band does not absorb lick variance; 1.5 kept
  (Priya: below it the treadmill moves with cue / lick jolts). Curve: 0 to 1.5 mm/s, peak 76-120 x1e3 near 5 mm/s,
  50-80 by 12 mm/s, 4/4.
- **push term: helps the whole-session fit (+3 to +34 x1e3 in most cells; pre PS95 +30, PS94 +18) but BREAKS post-stroke
  cue / lick windows** (chronic cue-window R2 PS95 .15 -> -.80, PS94 .41 -> .16, PS93 .52 -> .38) and inflates lick's
  unique dR2 there (PS95 chronic 26 -> 312). Cause (`scripts/qc_running_push_survey.py`, trial frames, backward speed > 0.3 mm/s): push
  frames are RARE pre-stroke (PS92 / PS93 / PS94 / PS95 0.6 / 0.7 / 3.9 / 1.8 %) and 2-5x more frequent by chronic in
  3/4 (2.5 / 3.8 / 5.3 / 5.9 %; PS94 1.4x), and harder (mean 0.54-0.60 -> 0.70-0.81 mm/s, p99 1.4-2.2 -> 2.2-2.5); a
  kernel fitted on rare pre-stroke pushes, applied to many post-stroke pushes, over-predicts. -> no push term in v7
  (option kept, `FROZEN_PUSH=1`). **Behavioural side-finding (PROVISIONAL):** backward / postural treadmill pushes
  become 2-5x more frequent after the stroke in PS92 / PS93 / PS95 (PS94 already high pre) -- bracing / posture change.
- acute far-contra decomposition identical in all six (contralesional SSp-m lick part -28 / -20 / -19 / -23 to -26,
  target <= 3.8).
**v7 basis maps (10-23 evening, `E/frozen_encoder_gate-attempted_v7basis/figures/`; PROVISIONAL, no statistics):**
acute far-contra pixel maps as v6 (bilateral lateral SSp / MOp drop reproduced by the lick part in 4/4; target /
exec_locked blank; contact opposite; residual mildly +). Pixel unique R2, pre held out (`r2_maps_pre.png`, all frames):
"other" largest (~0.55, posterior-medial), lick ~0.08 (frontal / MOs, anterolateral), target <= 0.04 (small bilateral
anterolateral spots). Acute - pre (`r2_maps_acute_minus_pre.png`): full R2 falls (posterior-medial, via "other"); unique
lick and exec_locked RISE in 4/4 (frontal / lateral: licking departs more from the pre template, so the same kernel
explains more variance -- not higher lick sensitivity); unique target slightly down (PS93 frontal) or mixed. Per
position, whole-spout unique R2 on own trials: pre = small lateral hot spots in both hemispheres, close_mid ~0; acute -
pre: far_ipsi down at the ipsi anterolateral spot (PS93, PS94, mean); **far_contra NOT a clean loss** (mixed by animal,
mean slightly +) -- the pre-stroke far-contra spout pattern still explains far-contra-trial activity about as well; the
drop is in amount (lick), not spout-specific shape. Low acute counts (PS95 1 session); unsigned (pair with
`pre_spout_contrast_*`); per-position direction readouts to be re-run with 4-animal 3-D.
**Pre-stroke per-spout maps (10-23, `scripts/frozen_spout_maps.py --contrast`; `_v6basis/figures/pre_spout_contrast_*`):**
data lateralised as expected in 4/4 (contra spouts -> left hemisphere, SSp-m L-R +12..+21; ipsi -> right); the SPOUT
part (target + spout-locked) tracks the data; the target vs exec_locked split is unstable (close_mid +13 vs -13) -- on
on-target licks they are one variable (-> 2.6). **Pixel / per-position unique R2 maps** (`scripts/frozen_r2_maps.py`:
full and per-group unique R2 per pixel, pre held out and epoch - pre; per position target-only and WHOLE-SPOUT = target +
spout-locked template swapped for the other spouts' pooled template): code + smoke test only (`_smoke-r2b`; whole-spout
~5x the target-only scale); full run on v7 RUNNING (`_v7basis`, 10-23 16:36).

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

### 4.3 What the time-course mixed models changed relative to the epoch tables (consolidated 10-23)

Session-level LMMs (`wfield_local/mixed_models.py`, one definition; `scripts/encoding_mixed.py`, `scripts/imaging_lmm.py`,
`scripts/within_spout_mixed.py`; models: TIME y ~ post + post:log(day), BEHAVIOUR y ~ post + post:deficit with deficit =
far-contra time-to-contact / contact loss relative to the animal's August pre, random intercept + step + slope per animal
with fallbacks; DECISIONS 2026-10-20 night / late night, 2026-10-21 night). What they did to the epoch-based results:

1. **Epochs became descriptive, LMMs primary (10-20).** The chronic epoch starts on day 11 (PS92 / PS93), 15 (PS95)
   and 25 (PS94), so an epoch table compares different post-stroke times across animals; the acute bin also mixes
   day 1 with days 2-5 while coding recovers over log time. Maps use each animal's actual chronic days.
2. **The acute drop is a step plus a log-time recovery, not a flat epoch.** Position coding drops at day 1 (cue step
   z -3.2) and recovers with log(days) (z 4.0), carried mostly by DISTANCE (recovery z 8.6); far-contra spout coding
   step z -2.6 / recovery z 5.5.
3. **The chronic gain (4.1) is a gain BEYOND behavioural normalisation, not just elapsed time.** Indexed by behaviour,
   coding falls with the far-contra deficit and sits ABOVE pre once behaviour is back at the pre level (cue ceiling
   +53 x1e3, z 4.6; lick distance z 4.5). This upgrades 4.1 and reframes 4.2 (cortex follows / overshoots behavioural
   recovery).
4. **Target vs executed over recovery (2.1 / 2.4b):** unique TARGET tracks the deficit strongly (3-D cue z -6.8 after
   the attempt re-definition; cam4 -6.7, first-lick window -5.1) and recovers with log days (3-D cue z 7.4); unique
   executed weaker (z 2.1 / -2.7); STRICT executed (beyond the spout category) present pre-stroke and unchanged after
   (step z ~1). The epoch reading "the post-stroke rise is target" holds as a time course.
5. **Frozen decoders / encoder: identity returns, the frozen code does not fully.** (CORRECTED 10-23, see the §2
   decoder bullet: the frozen-decoder deficit is NOT ~0 at full behavioural recovery -- +0.08, z 2.2-2.7, only PS92 back
   -- while refit-over-frozen G is +0.09, z 3.9-5.1, > 0 in 4/4 but not significant under the deck's block bootstrap,
   DECISIONS 2026-09-12; PROVISIONAL.) Frozen encoder EV (family 11)
   still below pre at full recovery (z -4.1) while far-contra own-position best match (10b) is back at pre (z -0.7).
   Epoch tables alone could not separate "not yet recovered" from "recovered but reorganised".
6. **Some epoch effects weakened or went away under the LMM + controls:** the within-spout toward-midline chronic
   change (LMM z -3.4) falls to -1.5 with the contact covariate (hits only) -- not robust; position coding in the
   first-lick window only z -2.0 (q .08) after the attempt re-definition; cue-window effects partly reflect delayed
   licking (hence the first-lick windows).
7. **NOT yet on the LMMs:** the frozen pre-stroke encoder decomposition (3.4, v4 / v6: acute far-contra drop carried
   by lick) is still EPOCH-based. Queued: its per-session parts (lick, spout, target, residual per spout x window) and
   the coming all-session shared + per-epoch model into the TIME and BEHAVIOUR models.

### 4.4 Postural / forelimb pushes collapse acutely with the deficit and overshoot after recovery — PROVISIONAL (10-23)

`scripts/push_behaviour.py` -> `qc/push_behaviour_sessions.csv`; LMMs `scripts/imaging_lmm.py --sources push` ->
`qc/imaging_lmm_summary_push.csv`. Push = backward treadmill movement > 0.3 mm/s (side-camera clips: a paw pushed
forward at bout ends; upper-limb extension / posture change right after the cue). Per animal (pre -> acute -> chronic):
cue-locked push (fraction of trials with a push in the 1 s after the cue) PS92 .13 -> .07 -> .43, PS93 .11 -> .02 -> .47,
PS94 .46 -> .10 -> .47, PS95 .25 -> .03 -> .52 (acute down 4/4; chronic above pre 3/4, PS94 ~equal); pushes per minute
1.3 -> 1.3 -> 4.3, 1.3 -> 0.6 -> 5.6, 7.2 -> 1.7 -> 7.9, 2.8 -> 0.4 -> 6.9. Forward running unchanged (no epoch / deficit
effect). LMMs: every push measure FALLS with the far-contra deficit (push rate z -11.5, push fraction z -10.3, far-contra
cue push z -5.0, cue push z -4.1) and sits ABOVE pre at full behavioural recovery (rate +2.7 / min z 7.9, size +0.18
mm/s z 10.1, cue push +0.22 z 2.6); time: day-1 step down (q ~.10), rise with log(day) (rate z 12.1, cue push z 11.3).
**Reading:** the cue-locked postural / forelimb adjustment is lost acutely in proportion to the orofacial deficit and
returns above pre -- the same shape as the cortical coding (4.3.3). Part of the same motor programme (postural set for
licking) and / or chronic compensation (bracing to reach). Open: engagement as a common cause (licked-trials-only /
engagement covariate), side-camera clips acute vs chronic, some LMMs fell back to random intercepts (4 animals).
Encoding consequence: push tracks the deficit, so it is modelled per session (`FROZEN_PUSH=free`, 3.4) and will get its
own kernel + per-epoch deltas in the all-session model; a frozen push kernel fails (3.4).
- **Free-push frozen run DONE (10-23 night, `_v7pushfree`; `qc/frozen_running_compare_v7fwd15_to_v7pushfree.csv`).**
  Re-fitting push per session (5-block CV ridge on the frozen residual) removes the frozen-push failure: PS95 chronic
  cue / lick R2 -0.80 / -0.75 (frozen push) -> 0.05 / 0.04 (free; no push 0.15 / 0.13); all-frames R2 >= no-push in
  15 / 16 animal x epoch cells. Not uniformly better in the windows: PS92 chronic cue / lick 0.21 / 0.08 vs 0.36 / 0.23
  without push (the CV push prediction costs in-window fit there). **The acute far-contra decomposition is unchanged
  by push treatment** (lick part SSp-m_left -28.0 / -19.6 / -19.3 / -24.9 free vs -28.2 / -19.7 / -19.4 / -25.8 none;
  target ~0; push / other part |x| < 3). Layer 2 (frozen push + per-epoch push deltas) launched 10-23 night.
- **The frozen pre-stroke FR basis does not miss the face at pushes (FaceRhythm window, 10-24; FACE_RHYTHM_FINDINGS 6b,
  `scripts/face_rhythm_push_check.py`, 107 sessions).** The face moves at push onsets, but reconstruction error there is
  LOWER than at random times in 27 / 32 animal x epoch x context cells and every chronic cell, despite 2-4x more pushes
  chronically. So no pre + post FR master; pushes stay a treadmill regressor here.

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
- **cam4 kinematics:** lateral angle faithful (r 0.95-0.97 with 3-D; confirmed in 4 / 4 animals by the pose window 10-23, OROFACIAL_FINDINGS §8, PS95 acute pending); "protrusion / reach" is 99 % the down axis, blind
  to forward. Every cam4 executed test is biased against executed by tracking noise and the missing axis.
- **Lick events:** the older cam4-subset runs (2.1-2.3, 3.1, 3.2) use cam4 licks; cam1 adds 3-7 % small licks and
  incomplete licks rise post-stroke (PS92 far_R share .17 -> .86 acute). Combined cam1 + cam4 lick timing is used in
  the LP-gated / frozen-encoder runs (3.3, 3.4) and 2.4b; where cam1 is missing (PS95, PS94_1005, other unfinished
  sessions) it falls back to cam4 only.
- **Lick DIRECTION inputs are incomplete (10-23):** our 3-D inputs are the 10-21 PS92 + PS93 snapshot (3-D for all four
  animals exists since 10-23 -- re-run due, box at top); PS94 / PS95 direction terms in the runs above are cam4
  2-D (lateral angle faithful, forward axis missing). Every direction result is to be re-run with full 3-D (box at top).
- **Running (10-23):** models before v6 extrapolated a linear running kernel (PS94 chronic fit failure); before v7 the
  kernel was signed (backward postural pushes counted as negative running). Results that depend on the "other" group in
  post-stroke windows (PS94 / PS95) should be read from v6 on and re-checked on v7.
- **Movement-first ordering** credits shared variance to movement; "after movement" = after licks, tongue, jaw, nose,
  running (cam4 kinematics, no video). Video is not used in the lick window (it re-describes the licks).
- **Shuffle dR2 of slow / event variables is inflated** (circular shift); lead with drop dR2.
- **Map significance tests (10-21 evening, STANDS).** Contours are now computed on the same pre-normalised scale as
  the pooled picture (`beta_maps.NORMALISE_TESTS_BY_PRE`). Calibration on pre-vs-pre splits: the nested bootstrap
  (primary) flags nothing on any pure null (figure 14 beta maps and all three 15r position windows); the cluster test
  is roughly calibrated (2 / 24 random splits) but its false clusters are LARGE on rest / restw maps -> read cluster
  contours as yes / no, not by area. June vs August pre (no stroke) shows up to ~21 % 'change' at a few positions:
  the August-only baseline is the right one. Acute far contra survives everything (58-93 %). Normalising mostly
  shrinks acute near-spout changes under the precue reference (lick near contra 36.7 -> 6.5 %; figure 14 far mid,
  cue, 20.0 -> 1.9 %). (An earlier line here said normalising removes changes "carried by the brighter animals":
  withdrawn 10-21 -- the per-animal check does not support it.) All map statistics are animal-weighted.
- **Lick-window imaging analyses use spout CONTACT unless tagged LP_licks (10-21).** Contact gating selects the
  licks that landed: acute far contra has 0 sessions with 20 contact trials, and acute far-spout post-lick amplitudes
  are inflated (3.3). The LP_licks family (video attempts, cam4 + cam1; outputs under `grant_figures/LP_licks/`) fixes
  both; pre-stroke the two definitions agree except PS93 far_ipsi / far_mid (27 % / 14 % non-contact attempts). LP
  'lick-free' ENL needs a 1.0 s window (>= 95 % of trials kept; 2 s keeps 48-73 % in PS93). LP null calibration
  (10-21 night, 312 pure-null tests): nested flags >= 1 bin in 1.6 % (mostly 1-7 bins), cluster max 27 % of area.
  cam1 was not finished when these ran (74/108 curated sessions; PS95 none) -- LP licks there are cam4 only. cam1 is
  now 125 / 127 (10-23); re-run the LP_licks families on it.

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
| destination vs landing | `qc/` | `best_match_destination*.csv` (lick arm `_lick_lick`; separating rows only) |
| frozen pre-stroke encoder (areas) | `frozen_encoder_gate-attempted_v6/` (v7 candidates `_v7fwd15`, `_v7fwd15push`) | `frozen_musall_r2_by_animal.csv`, `joint/sessions/*.json`, `figures/area_maps_*` |
| frozen encoder decomposition | `qc/` | `lp_encoder_decompose_frozen_encoder_gate-attempted_<tag>_<variant>[_lick]_by_animal.csv` |
| frozen encoder pixel maps | `frozen_encoder_gate-attempted_v6basis/` (`_v6grid` PS92 check) | `figures/pixel_maps_*`, `figures/pre_spout_{maps,contrast}_*` |
| running-model comparison | `qc/` | `frozen_running_compare_*.csv` (+ `_curves.csv`); QC `qc/running_clips/`, `_v6thr1/figures/qc_running_threshold*.png` |
| decoder / reorganisation LMM | `qc/` | `imaging_lmm_summary.csv` (source `decoder_*`); inputs `labcams/grant_figures/epoch/data/recovery_trajectory_*_cue_working.csv` |

## 7. Next (in order; 10-24) -- the goal question

**Goal (Priya 10-24): what does cortex "look like" post-stroke when the animal attempts a movement it cannot execute, or
executes differently?** Post-stroke trials per spout, by what the animal did (video attempt classes, `trial_attempts.csv`
rebuilt 10-24): hit (executed, maybe differently) / incomplete (tongue out, no contact = cannot execute) / no attempt while
working / reaction (jaw only) / stopped (quit = no-task floor); pre-stroke hits at the same spout = reference.

1. **Is the plan there when execution fails?** `wfield_local/attempt_coding.py` (NEW): stored frozen pre-stroke decoder +
   per-spout pre-stroke directions on each class, ENL (clean) / cue / lick windows; within-session contrasts incomplete vs
   stopped, no-attempt vs stopped. Redo of the August contact-defined coding directions (`miss_vs_stopped`), which could
   not split no-contact attempts from no attempt. Default (June + August) pre, matching the imaging families and the
   frozen decoders; August-only = sensitivity (needs its own DAQ-frozen set).
2. **When execution differs, does cortex follow target or movement?** Binned contest (O2, primary, w = 1 / 0.5 / 2);
   relabel test in the final frame; strays secondary.
3. **What is different / missing?** Frozen encoder final frame `_v7g3_t3d` (running) -> `lp_encoder_decompose` on it
   (audit gap 2), basis maps, Layer 2, relabel.
4. **What it looks like:** `wfield_local/attempt_maps.py` (NEW): 15r's rest-referenced maps split by class, per animal.
5. **Recovery + confounds:** LMMs after 1-4 land; FR groups (O2); push (4.4).

**Redo list (full audit `docs/status/OUTPUT_AUDIT_2026-10-24.md`):** LP_licks re-render (maps 15r / 14, frozen decoders
acc / accdelta / 5c / 5cr, calibration; `lp_trial_counts` first; PS93 LP-frozen sensitivity); `lp_encoder_decompose` on
the final frame + O2 main; O2 variants not in the batch (nokin_matched / tr / win, drop_win: submit or retire); within-spout
in the final frame (attempted gate + combined licks; cue-aligned + ENL variants; mixed model, per-position direction maps,
reliability chained after); destination vs landing (fix `CONTEST_3D`, deleted folder) and lick templates on 4 animals;
behaviour / target-vs-execution figures, `lick_contact_by_epoch`, recovery trend, trial-average R2 after the O2 collect.
Deferred: non-linear ceiling; pre-cue -> lick-window carry-over decode; decoder reorganisation contrast (2 / 4.3.5).

## Change log

- **2026-10-24 (3-D)** — 4-animal 3-D polar contest (2.4b): pre target-led 3/4; chronic target above pre 4/4; the
  post-stroke rise is target in 3/4 (PS94 executed rises as much).

- **2026-10-24 (relabel)** — relabel test full run (2.7): validation 100 %; heterogeneous -- PS95 re-aimed at far mid
  (specific, transient), PS93 stays far contra, PS92 / PS94 nonspecific loss of spout structure.

- **2026-10-24** — PS93 contact joint-drop check: no variance shared with within-spout direction (~0); the
  shared-with-direction explanation is withdrawn (3.4).

- **2026-10-23 (night, Layer 2)** — all-session encoder first run: per-epoch kernel changes beat the pre drift null in
  4/4 acute and chronic (cortex); target + within-spout direction responses change most; free-push frozen run (4.4).

- **2026-10-23 (night, last)** — stray-lick test first full run (2.6): validation fails for the far-contra aim reference
  in 3 / 4 animals; strays look like where they landed pre-stroke; no consistent acute shift -> not decisive. Relabel
  test readout = all areas jointly (smoke; full run going).

- **2026-10-23 (night, late)** — grouping v3 (contact its own category, out of "non-task"; nose -> other was v2);
  PS93 contact predictability check (contact most predictable from execution in PS93 -> shared variance); stray-lick
  test now has per-spout contact kernels (contact vs non-contact at the same aimed spout).

- **2026-10-23 (night)** — 4.4 added: pushes (incl. the cue-locked postural adjustment) collapse acutely with the
  far-contra deficit and overshoot pre after recovery (LMM deficit z -4 to -11.5; log-day z 5-12).

- **2026-10-23 (late evening)** — 3-D complete (119 sessions, 4 animals; re-cut 107 encoding sessions; 4-animal polar
  contest running); contact kernel sign checked (negative 4/4 -> the + contact part = fewer contacts); PS93 contact
  note; v7 basis maps (per-position far contra not a clean loss); free-push variant and push behaviour readout running.

- **2026-10-23 (evening)** — v7 chosen and made the default (forward-only running from 1.5 mm/s, no push term;
  STATS_VERSION 7); push term breaks post-stroke windows because pushes are 2-5x more frequent after the stroke
  (provisional behavioural finding); 3.4 / status box / next updated; `_v7basis` running.

- **2026-10-23 (afternoon)** — status box at the top (what each result rests on, incomplete data, the 3-D re-run
  rule); **CORRECTED** the decoder line (frozen deficit NOT ~0 at full behavioural recovery: +0.08, only PS92 back;
  refit gain G > 0 in 4/4 but not significant under the deck bootstrap, DECISIONS 2026-09-12); 2.6 strays section
  (census, design, pending); 2.4b 3-D 2x2 status; 3.4 retitled (v6 current, v7 pending) + running-model side-by-side,
  pre-stroke per-spout contrasts, pixel / per-position maps pending; caveats (lick timing, direction, running);
  where-to-look + next updated.

- **2026-10-23** — §3.4 running model (sublinear; forward-only from 1.5 mm/s; push term candidate; headline unchanged);
  2.4c destination % corrected; 4.3 what the LMMs changed vs epochs; pixel / per-position (target-only + whole-spout)
  unique R2 code; target and the spout-locked lick are one variable on on-target licks (strays decide).

- **2026-10-22 (night)** — §3.4: frozen v6 = running capped at the pre p99 of running frames; PS94 post-stroke fit
  restored, headline unchanged; v5 (all-frame p99) superseded; pixel maps (basis + PS92 grid).

- **2026-10-22 (afternoon / evening)** — §3.4 added: frozen encoder v4 (lick / spout-position groups by pre-stroke
  spout-locked templates), acute far-contra drop carried by LICK in all orderings, target <= 1.7; area maps; PS94
  chronic frozen fit fails via "other" (open). v3 frozen outputs (`frozen_encoder_gate-attempted/`) superseded.
- **2026-10-21 (night)** — 3.3 added: 15r position maps under LP lick attempts (acute far contra measurable: post-lick
  activity below pre in 3/4, bilateral; contact gating had inflated acute far-spout amplitudes); LP null calibration;
  how to read nested vs cluster outlines; "brighter animals" normalisation claim withdrawn.

- **2026-10-21** — trial attempts + time-to-contact deficit; 3-D contest + template rerun (2.4b); destination vs 3-D
  landing (2.4c); jaw-onset rewrite rerun; within-spout per spout + contact control (pooled toward-midline not
  robust); one-load variants + local input cache (handoff STATUS_2026-10-21_ENCODING_HANDOFF).

- **2026-10-21 (evening)** — map tests on the pre-normalised scale + calibration (caveats); LP_licks family
  built on a branch; ENL LP census running (handoff STATUS_2026-10-21_ENCODING_HANDOFF §1d).

- **2026-10-20 (evening)** — within-spout lick-aligned / bilateral / mixed model / map tests; 3-D landing zones
  calibrated (soft bins); lick-template test built; 3-D contest queued (handoff STATUS_2026-10-20_EVENING_ENCODING_HANDOFF).
- **2026-10-20** — slow band checked by epoch (size caveat; PS93 acute pre-cue drop withdrawn); binned contest and
  within-spout deviation check added (DECISIONS 2026-10-20).
- **2026-10-20** — created from DECISIONS 2026-10-19 ... 2026-10-20 (encoding window).
