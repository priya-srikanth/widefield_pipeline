# Imaging encoding findings: target vs executed, per-spout coding, recovery

> **Current conclusions: [FINDINGS.md](FINDINGS.md) section 4** (one-page, status-tagged; 10-25). This document is the detailed evidence record for the encoding model and the goal question.

Status: **preliminary, living document.** Last updated **2026-10-23 afternoon** (encoding window). Update this file
whenever a result below changes: edit the section, change its status tag, add the date to the change log at the bottom.
The reasoning and every number's history are in `DECISIONS.md` (entries 2026-10-19 ... 2026-10-23); the working state
(running jobs, commands, pitfalls) is in the newest `docs/status/STATUS_*_ENCODING_HANDOFF.md` (now 10-22, §000f).

Status tags: **STANDS** (survived the controls listed), **PROVISIONAL** (one control still pending), **SUSPENDED**
(contradicted by a better measurement on part of the data; waiting for the full check), **PENDING** (not yet run),
**CORRECTED** (an earlier statement here was wrong; the corrected numbers follow).

> **Where things stand (10-23; rows for 2.7-2.9, Layer 2 and section 8 added 10-25), with what each rests on.**
>
> | result | status | data it rests on / what is incomplete |
> |---|---|---|
> **10-24: attempts table REBUILT with complete cam1** (was cam1 on PS94 40 % / PS95 0 % of trials). Effect is small:
> gate flips 0 / 0 / 93 / 62 trials (0 / 0 / 0.6 / 0.4 %, all toward attempted); far-contra acute attempt rate PS94
> .441 -> .445, PS95 .325 -> .342 (denominator ALL far-contra trials); gated results below are not expected to move;
> final-frame re-runs use the new table.
> **Acute far-contra attempt rate, CURRENT (rebuilt table, attempted = hit | incomplete; 10-25 recount):** canonical
> (engaged) trials, pooled, PS92 / PS93 / PS94 / PS95 **.79 / .79 / .58 / .34** (n 336 / 292 / 450 / 120; session mean .79
> / .80 / .65 / .34 -- the ".79 / .80 / .64 / .32" in 2.5 is this measure on the 10-21 table); ALL trials .62 / .61 / .445
> / .342 (n 444 / 384 / 605 / 120); completion (hit / attempted) .06 / .15 / .04 / .02. The ".81-.93" in 2.5 counted
> jaw_only as an attempt (SUPERSEDED 10-21). **Mechanism (canonical trials):** PS94 = "try and fail" (252 incomplete vs 78
> no_try, 109 jaw_only); PS95 = mostly "fail to try" (40 incomplete vs 22 no_try + 57 jaw_only reactions); PS92 / PS93
> try and fail (250 / 195 incomplete vs 27 / 25 no_try) -- as FACE_RHYTHM_FINDINGS 5c.
>
> | Acute far-contra drop is carried by LICK in the frozen pre-stroke encoder; target <= ~4 x1e3 (3.4) | PROVISIONAL, robust to credit order and to 5 running models | 4/4 animals, but PS95 = 1 acute session; PS94 / PS95 lick timing cam4-only in these runs (their cam1 landed 10-21..23, not yet used); running model v7 still being chosen; no test on data - model; not yet on the LMMs |
> | Post-stroke rise is TARGET, not executed (2.1, 2.4b, LMM 4.3.4) | STANDS for cam4; PROVISIONAL in 3-D | 3-D results so far PS92 + PS93 (10-21 snapshot; 3-D now exists for all 4 -> re-run due); the running 3-D 2x2 is on the snapshot; pre-stroke answer depends on the frame (2.4 vs 2.4b) |
> | Destination vs landing "target-like 61 / 73 / 90 %" (2.4c) | CORRECTED (inflated) | separating rows only: target-leaning, ~6 decisive sessions, 2 animals |
> | Target vs executed, decisive test = lick-level STRAYS vs on-target (2.6) | FIRST RUN: not decisive (far-contra reference fails validation 3/4) | census on the 10-21 snapshot (PS92 + PS93, trial licks); 3-D for all 4 now exists; per-lick ITI 3-D requested |
> | Within-spout lick direction (2.5) | PROVISIONAL; no robust post-stroke execution-like shift | cam4 (all 4) / 3-D (2, snapshot; 4 now available); split-half reliability ~0.5 |
> | Chronic coding gain beyond behavioural normalisation (4.1, 4.3.3) | STANDS (4.1) / PROVISIONAL (LMM) | PS94 shows no gain |
> | Frozen decoder back at full behavioural recovery (2, 4.3.5) | CORRECTED: NOT back (+0.08, only PS92 back); refit gain over frozen > 0 in 4/4 but not significant under the deck bootstrap | LMM vs deck resampling disagree (DECISIONS 2026-09-12) |
> | Running response is sublinear, forward-only from 1.5 mm/s = v7 default; a push term breaks post-stroke windows (3.4) | PROVISIONAL (v7 chosen 10-23) | push frames 2-5x more frequent post-stroke in 3/4 -> no push term |
> | Relabel: far-contra trials sit in another spout's pre-stroke state (2.7, 8.3) | NOT ESTABLISHED at session level (heterogeneous; other spouts also lose their label; nothing survives Bonferroni) | 4 animals, final frame; per-trial run in progress |
> | Failed attempts on the frozen decoder: "distance kept, side lost"; no pre-cue plan signature (2.8) | FIRST RUN (10-24) | video attempt classes, rebuilt attempts table; DAQ-frozen pre decoder |
> | Failed-attempt maps: same bilateral pattern, lower amplitude than hits (3/4) but not than touch-matched pre misses (2/4) (2.9) | FIRST RUN (10-24) | >= 5 trials per session x spout x class |
> | Layer 2: the behaviour -> cortex mapping changes chronically beyond pre-stroke session variability (2.5 Layer 2, 8.5) | STANDS vs the random-split null (chronic P = 0, 4/4); PROVISIONAL as "reorganisation" | grows with time (early chronic ~null in PS93 / PS95); slow drift not separable without a trend model; no single group carries it |
> | Licks land / travel toward the wrong spout from launch; decoder follows landing on average, trial coupling breaks acutely (8.1, 8.2) | STANDS | landing + trajectory softmax, nested bootstrap, Bonferroni 36 cells; 4/4 |
>
> **Standing re-run rule (Priya 10-23) -- NOW DUE:** every lick-DIRECTION result (exec_within / exec_locked dR2, stray
> test, 3-D contests, within-spout direction, direction maps) must be RE-RUN on full 3-D. **Update 10-23 afternoon (pose
> window; counts SUPERSEDED by the 18:28 entry below):** 3-D then covered all four animals (124 sessions) and cam1 125 / 127
> curated sessions. Every 3-D result in this file was computed from the 10-21
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
> FaceRhythm timing (cam4 dropped 11.8 % of frames, FACE_RHYTHM_FINDINGS 1b) RESOLVED 10-23 on fr-timing-v1 (FR model cortex R2 .506 -> .618;
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

> **FINAL INPUTS 10-24 (O2 `enc_v36_full_20261024`, `bins_w1` / `bins_w05`; attempted gate on the rebuilt attempts
> table, combined licks with complete cam1, 20261023 3-D tables; `E/movement_first_subset-nokin_exec-polar_licks-
> combined_bins[_bw0.5]/`, 107 sessions; 7 big sessions re-run at 64 GB after OOM).** Cortex, unique dR2 x1e3, target
> bins vs executed bins, PS92 / PS93 / PS94 / PS95:
> - **chronic: target > executed 4/4 in both windows** -- cue 17.7 vs 4.7, 15.2 vs 9.2, 10.5 vs 7.3, 22.1 vs 3.8; lick
>   10.6 vs 2.7, 32.0 vs 9.1, 8.3 vs 5.4, 12.2 vs 6.5.
> - **acute (the failed / altered far-contra licks live here): target > executed 3/4 in the lick window** (14.6 vs -21.4,
>   17.8 vs 3.5, 9.0 vs -1.7; PS93 the other way, -0.9 vs 6.6); cue window PS94 / PS95 target, PS92 both negative, PS93
>   tied. Stronger than the reference-gate local run (PS92 lick 14.6 vs 8.9, PS95 9.0 vs 2.7).
> - subacute: MIXED -- target ahead 2/4 per window, different animals (cue PS94 9.8 vs 6.6, PS95 6.6 vs 0.1; PS93
>   executed ahead 2.4 vs 10.9; PS92 tied; lick PS92 10.7 vs 3.4, PS93 14.0 vs 10.9; PS94 tied; PS95 ~0); fewest
>   sessions (2 / 3 / 5 / 7), behaviour recovering.
> - pre: no target lead (both small).
> - **Softmax width sweep (0.5 / 1 / 2):** chronic target > executed 4/4 at 0.5 and 1 in both windows, 3/4 at 2 (PS94
>   ties: lick 14.1 vs 15.9, cue 18.4 vs 19.4); acute lick target ahead 4/4 at 0.5, 3/4 at 1, 2/4 at 2 (PS92, PS95). The
>   executed set gains as the bins widen (a wide soft bin smears each lick over neighbouring spouts, so it overlaps the
>   target categories more): the chronic lead holds across the sweep, the acute lead is width-dependent.
> - **Where the licks land (LICK data; `fig9*` on the goal page, `softmax_landing_change_stats.csv`):** within-animal
>   epoch - pre landing matrices, mean of animals, nested bootstrap (animals -> sessions), Bonferroni over 36 cells: far
>   contra's own bin -0.69 / -0.38 / -0.19 (acute / subacute / chronic), far mid +0.35 / +0.36 / +0.20, far ipsi +0.35
>   acute -- all significant, 4/4 same sign. Part of the far-contra licks still land like far-mid licks chronically.
> - What the acute licks are: failed far-contra licks land in the far-mid / far-ipsi executed bins (e.g. PS95_0817 soft
>   weights far mid .48, far ipsi .42, far contra .001) -- the dissociation the contest exploits. Caveat as 2.5:
>   "target" = spout location (touch, posture, context), not intention alone.

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
  PS94 2.6 vs 4.0). Acute mixed IN THIS LOCAL (reference-gate) RUN; SUPERSEDED by the FINAL-INPUTS box above (acute lick
  target > executed 3/4; width sweep 0.5 / 1 / 2 DONE: chronic lead holds, acute lead width-dependent).
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
(cue z 7.4) and falls with the deficit (cue z -5.3 on the 10-20 attempts; CURRENT -6.8, 2.5 jaw_only bullet; target bins z -7.3); executed weaker (cue time z 2.1, deficit
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
  85-97 % of those trials in all four. Among LICKED trials: PS92 / PS93 = slow to start, PS94 / PS95 = prompt but miss.
  Across all canonical trials (top box): PS92 / PS93 / PS94 try and fail; PS95 mostly fails to try. `qc/first_lick_latency*`.

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
  cue unique target z -8.0 / +8.7; 10-20 attempts -- current deficit z -6.8, jaw_only bullet below). Mixed models are now the primary readout; epochs (chronic starts day 11-25 by
  animal) descriptive. Cue-window effects partly = delayed licking -> `firstlick` window pending (O2).

- **Jaw-onset rewrite (10-21 07:57: cam1 for all PS93 + PS94 to 08-17) -> attempts + all LMMs rerun.** Unknown-class
  trials <= 6 %; far-contra deficit unchanged (acute attempt .81-.93 with jaw_only counted as an attempt -- SUPERSEDED,
  current rates in the top box; completion .01-.11; time to contact pre 545-648
  -> acute 3221-3467 -> chronic 471-692 ms). Target tracks the deficit more strongly: 3-D unique target post:deficit
  z -8.3 (was -5.3), cam4 z -8.2 (cue) / -6.3 (first lick); strict executed still null.

- **jaw_only = reaction, not attempt (10-21, Priya's blind review 0/24) -> attempts + LMMs rerun.** Acute far-contra
  attempt rate .79 / .80 / .64 / .32 (PS92-PS95; canonical trials, session mean, 10-21 table; current .79 / .80 / .65 /
  .34, top box): PS95 mostly fails to INITIATE (fail to try), PS94 tries and fails (incomplete >> no_try); completion
  .03-.15 in all four.
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
- **Layer 2 REDESIGNED (10-24, `allsession_encoder_v7g3_t3d_ref`, `change_ref` from the EARLY-pre reference, pre_late
  null in the same "combined" fit).** Cortex, x1e3, null / acute / subacute / chronic: PS92 10.3 / 50.5 / 59.5 / 101.9,
  PS93 33.9 / 32.9 / 23.9 / 49.8, PS94 28.7 / 52.3 / 61.0 / 61.7, PS95 21.6 / (1 session) / 33.4 / 100.1 -> chronic >
  null 4/4 (PS93 1.5x); SSp-m_left chronic 88.7 / 48.3 / 98.0 / 46.1 vs null 15.8 / 12.8 / 17.5 / 25.0. **No single
  group's unique change beats the null** (e.g. lick chronic 3.8 / 0.8 / 1.6 / 3.7 vs pre_late 3.4 / 1.8 / 2.9 / 7.5;
  target 5.2 / 4.0 / 0.9 / 3.1 vs 4.9 / 3.2 / 2.2 / 9.6): the change is shared across groups, not "motor" vs "sensory".
  **Session-count control** (chronic cut to 4 sessions = pre_late's; `matched_n_control.csv`): first-4 chronic 107.0 /
  38.7 / 59.0 / 4.0, last-4 122.6 / 59.6 / 62.1 / 138.3 vs null 9.8 / 33.4 / 28.7 / 22.1 -> not a session-count effect
  (PS92 / PS94 stay far above), but it GROWS WITH TIME (PS93 / PS95: early chronic ~null, late chronic >> null). Early
  chronic > null 2/4, late chronic 4/4. The null spans one week, chronic one to two months, so progressive
  reorganisation and slow drift are not separable here -> the deciding test is a trend model with a pre-stroke slope
  (as 4.3.3). PROVISIONAL.
  **All-August-pre reference (10-24 evening, Priya: "just use august sessions with loso"):** the "post" fit already does
  this (every August pre session is reference, held out LOSO): change_ref cortex acute 50.2 / 36.5 / 60.2 / (1 session),
  subacute 52.1 / 23.5 / 49.4 / 28.2, chronic 97.3 / 63.3 / 63.7 / 107.0; no single group carries it. It has no within-pre
  baseline by construction (pre change = 0) -> RANDOM-SPLIT null (`scripts/layer2_random_splits.py`: every choice of k
  August pre sessions as a pseudo-epoch, kappa fixed per animal) running; it controls session noise / count, not slow
  drift (the trend model's job). June stays out of encoder analyses (kinematics off, no FaceRhythm).
- **[WITHDRAWN 10-24 -- design flaw (gain = distance from an all-epoch compromise, null fitted on pre only), see the
  REDESIGNED bullets above and 8.5; kept as the record.] Layer 2 all-session encoder: the cortical RESPONSE to behaviour
  changes beyond pre-stroke drift, acutely AND chronically (FIRST RUN 10-23 night; `wfield_local/allsession_encoder.py` at 2fc9fce, grouping v3, frozen
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

> **TIMING BUG, RE-RUN NEEDED (found 10-25 ~23:50).** `stray_licks.py` places its lick events with the kinematics_all
> table's `daq_peak_s`, which is stored to 4 SIGNIFICANT DIGITS (0.1 s resolution after 100 s, 1 s after 1000 s: up to
> 0.5 s off in a 1 h session; e.g. two licks of one bout both read 121.3). The lick-event kernels below are therefore
> smeared by up to +-0.5 s, worse late in each session. Fix: take the exact time from the session's `licks_all.csv`
> row `lick_uid` (as `scripts/iti_licks.py` does; checked: context matches 100 %, clock match to the behaviour licks
> 1 ms). Read everything in this section as provisional until the re-run.

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

- **Regional / hemispheric readout (Priya 10-24, from the maps; `E/alm_roi_far_contra.csv`, `E/chronic_roi_hits.csv`;
  regions = Allen MOs / MOp / SSp-m per hemisphere on the eroded stat mask, ALM proxy = lateral half of MOs within its
  anterior half; ratios of region means, lick window; image left = ipsilesional).**
  - **Acute failed far-contra attempts: IPSILESIONAL ALM / tjM1 reduced more than contralesional.** / pre-stroke hits, left
    vs right: ALM proxy .33 vs .42, .41 vs .55, .46 vs 1.00, .73 vs 1.15; MOp .32 vs .33, .37 vs .42, .43 vs .79, .80 vs
    1.30; SSp-m .17 vs .21, .24 vs .35, .24 vs .52, .56 vs .87 -> left < right 4/4 in every region (PS92 small), cue window
    the same. Against touch-matched pre-stroke misses the ALM proxy keeps left < right 4/4 (PS92 .41 vs .42, PS93 1.57 vs
    1.89, PS94 .40 vs .60, PS95 1.14 vs 1.53); MOp does not (PS92 / PS93 reverse).
  - **Chronic hits: INCREASED tjM1 / SSp-m activation, larger contralesionally, all spouts.** / pre-stroke hits, far
    contra, left | right: SSp-m 1.54 | 1.65, 1.35 | 1.77, 1.32 | 1.94, 1.15 | 1.20; MOp 1.22 | 1.33, 1.21 | 1.32, 1.09 |
    1.46, 1.05 | 1.29 -> up 4/4 both hemispheres, right > left 4/4; median over the six spouts also > 1 for SSp-m and MOp
    4/4; ALM proxy inconsistent (PS95 < 1). Whole hemispheres ~1 (.87-1.15), so regional rather than a global gain change.
    Fits the chronic position-coding gain (4.1). Caveats: no statistics yet (ratios of session-weighted means), ALM / tjM1
    are proxies from Allen areas, amplitude across months also carries slow baseline drift (09-12 note).

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
  removing LP-detected licks changes nothing at matched length. The acute FAR-CONTRA pre-cue loss (this spout only;
  all spouts pooled show no consistent loss, section 5) survives every definition:
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
>
> **Across epochs (final frame, far contra, SSp-m, lick window, x1e3, left | right):** subacute the measured response
> recovers faster than the pre-stroke mapping predicts (residual +5..+15, 4/4); **chronic the response is LARGER than
> pre** (data 18.8 | 18.4, 16.0 | 19.5, 6.7 | 8.6, 2.4 | 3.9) while the frozen model predicts ~0 or a decrease (full
> -9.1 | -5.1, 4.9 | 6.8, -8.1 | -1.5, 0.3 | 0.6), so the increase is in the RESIDUAL (27.9 | 23.6, 11.0 | 12.7, 14.8 |
> 10.2, 2.2 | 3.3; 4/4 positive, PS95 small). MOs the same direction, weaker; cue window the same. Reading: the acute
> drop is behaviour through the pre-stroke mapping; the chronic increase is cortex responding MORE to the same behaviour
> (the response change the redesigned Layer 2 attributes to groups). Maps: `frozen_encoder_gate-attempted_v7g3_t3d_basis/
> figures/pixel_maps_joint_far_contra_{acute,subacute,chronic}_lick.png` (residual broad lateral / anterior, PS92
> strongest).
> **DRIFT CHECK (10-24, same frozen run, LOSO-scored pre sessions): NOT YET BEYOND DRIFT IN 3/4.** The residual already
> rises within the pre-stroke sessions (late half - early half, SSp-m far contra lick, left | right): 11.6 | 12.2, 11.6 |
> 5.4, -1.8 | -3.7, 12.1 | 7.0 (session SD 4-10). Chronic - pre 27.9 | 23.6, 11.0 | 12.7, 14.8 | 10.2, 2.2 | 3.3; chronic -
> late pre 22.1 | 17.5, 6.0 | 10.4, 15.7 | 12.1, -3.9 | -0.2 -> clearly beyond drift in PS92 and PS94, ~drift-sized in
> PS93, below in PS95. Something was already increasing the response during the pre-stroke weeks (practice / slow signal
> change), so the claim needs the chronic increase to exceed the extrapolated pre-stroke TREND. Tests: redesigned Layer 2
> (per-group change from early pre with the pre-late null in the same fit; lick vs contact = motor vs sensory), a
> residual-vs-day model with a pre-stroke slope (as 4.3.3), and the FR runs (does face movement absorb it). Not a
> finding until then. Also not far-contra-specific (chronic hits up at all six spouts, 2.9).
> **MAP SIGNIFICANCE of the residual (10-24, `scripts/frozen_residual_significance.py`; each epoch's per-session residual
> maps vs the PRE sessions' residual maps, repo map tests, pre-normalised; `frozen_encoder_gate-attempted_v7g3_t3d_basis/
> residual_significance.csv`, `figures/residual_significance_*`).** Nested bootstrap bins / 2,022 (cluster test):
> lick window -- drift calibration (late vs early pre) 22, acute 80, subacute 41, chronic 48; cue -- drift 39, acute 20
> (suppressed: rim), subacute 8, chronic 34. The cluster test finds NOTHING in any epoch; pure nulls flag ~1.6 % under
> nested (10-21 calibration). Chronic patches sit over CONTRALESIONAL anterolateral cortex (fits the 2.9 ROI result:
> chronic increase larger contralesionally); acute / subacute over contralesional lateral cortex. The drift panel itself
> is a broad positive residual over ipsilesional lateral cortex, resembling the chronic pattern. -> At map level the
> residual change is weak and not robust (nested ~1.5-2.5x the drift calibration, cluster nothing).
> **PER ANIMAL + TREND (10-24 evening).** Reference = ALL the August pre sessions (LOSO), no early / late split (Priya).
> Within-animal test (`--per-animal`; the repo's "musall" bin test = sessions as unit, since the nested bootstrap needs
> >= 2 animals), lick window, bins / 2,022: chronic PS92 522, PS93 110, PS94 135, PS95 113 (4/4); cue chronic 182 / 0 /
> 77 / 0; acute 0 in every animal; subacute only PS95 (114). CALIBRATED: 20 random 4-vs-4 splits of each animal's pre
> sessions flag 0 bins in 20/20 (`residual_significance_musall_null.csv`). Significant chronic regions are all
> CONTRALESIONAL (PS92 bilateral anterolateral, PS93 / PS94 right anterolateral, PS95 right posterolateral). TREND
> (`scripts/frozen_residual_trend.py`, `residual_trend.csv`): the pre-stroke rise (1-2 x1e3 / day over the 8 August days
> in PS92 / PS93 / PS95) did NOT continue -- chronic sits far BELOW the extrapolated line (z -1 .. -4); vs the flat pre mean
> chronic is above in 3/4 (SSp-m z 7-9 / 3-7 / 5 / PS95 0.5-1); within the chronic sessions the residual is flat or falling
> (no continuing drift). Reading: a raised, stable chronic level in 3/4 (not PS95), contralesional, significant within
> animal; the pooled pixel test stays weak because the location differs between animals. Pending: per-spout and
> all-spouts-pooled pooled maps, larger stat bins (16 / 32 px, calibrated), a region-level test (Allen ROIs fixed a priori).

> **Current version: v7** (STATS_VERSION 7, commit 73d2991, 10-23: forward-only running from 1.5 mm/s, no push term).
> Areas: `E/frozen_encoder_gate-attempted_v7fwd15/` (same model, run before the default changed); pixel maps + pixel /
> per-position unique R2: `_v7basis` (done 10-23). v6 folders kept for comparison. The FINAL FRAME
> (`_v7g3_t3d`, box below) uses combined licks with complete cam1 for all four animals.
> Incomplete: PS95 1 acute session; in the v7 runs of THIS box PS94 / PS95 lick timing was cam4-only; no statistical test on data - model; still
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
  frames are RARE pre-stroke (PS92 / PS93 / PS94 / PS95 0.6 / 0.7 / 3.9 / 1.8 %) and (descriptive, no test) 2-5x more frequent by chronic in
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
**10-25 re-run on the FINAL-FRAME O2 runs** (`scripts/position_coding_trend_recovery.py`, now defaulting to
`movement_first_subset-nokin_exec-polar_licks-combined_bins` + `movement_first_exec-polar_licks-combined`; 107 August+
sessions): the LEVEL change replicates, ceiling pre -> chronic PS92 .041 -> .108, PS93 .057 -> .108, PS95 .047 -> .076, PS94
.040 -> .037 (3/4 again). The PRACTICE-TREND test cannot be repeated there: the bundle has no June sessions, so the pre
span is 8 days and the extrapolated prediction intervals are +-0.07 .. +-0.45 (beyond_trend False everywhere: uninformative,
not a negative). The June + August result above is from the 10-19 cam4 runs (stale inputs); it stands only until a
final-frame run with June sessions exists. Status: level gain STANDS; beyond-trend PROVISIONAL.

### 4.2 It FOLLOWS behavioural recovery — PROVISIONAL (coarse sampling)
10-25 final-frame re-run: 50 % day, ceiling vs far-contra contact PS92 18 vs 9, PS93 15 vs 7, PS94 9 vs 11, PS95 no
ceiling change (spread 7 vs contact 3): cortex still later in PS92 / PS93. Numbers below are the 10-20 version.

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
  pattern is not created by it; there is no consistent acute pre-cue LOSS with ALL SPOUTS POOLED (up in PS92 / PS95,
  down in PS93 / PS94; far contra alone does fall 4/4, 3.3);
  PS93's acute drop is the slow band (-.057 -> +.006) and its subacute drop halves; PS94's drop survives at ~2/3;
  chronic >= pre in all four. Report slow-band-removed pre-cue decode as a sensitivity column; lick windows unaffected.
  `qc/slowband_decode_all.csv`, `qc/slowband_caveat.png`.
- **Pre-cue window: contact-free by design, not necessarily lick-free (10-20 night).** The ENL restarts on any spout
  contact, so the final 2 s before the cue has no contacts (last contact median 2.2-2.5 s before the cue); but
  no-contact licks can occur in it and lick-evoked GCaMP6s activity from contacts just before it can spill in.
  Pending: whole-session lick detection (other window) and an early vs late ENL split. `qc/precue_contacts.csv`.
- **cam4 kinematics:** lateral angle faithful (r 0.95-0.97 with 3-D; confirmed in 4 / 4 animals by the pose window 10-23, OROFACIAL_FINDINGS §8, PS95 acute pending); "protrusion / reach" is 99 % the down axis, blind
  to forward. Every cam4 executed test is biased against executed by tracking noise and the missing axis.
- **Lick events:** the older cam4-subset runs (2.1-2.3, 3.1, 3.2) use cam4 licks; the combined track adds 3-7 % licks,
  mostly licks cam4 lost to occlusion / merging (not small; POSE_MODELS combined-track licks (5)), and
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
  now 127 / 127 (10-23 18:28; 3-D 119 curated); re-run the LP_licks families on it.

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

## 8. Results of 10-24 night / 10-25 (goal question; to be folded into 2-4) — results page Figs renumbered 10-25

All on the goal-question results page https://claude.ai/artifact/Fnpn2KHknMwh3r13xADH7d (figure numbers as of 10-25,
in page order). August pre only; nested bootstrap = animals -> sessions (`analysis_kit.boot_delta`), Bonferroni where
cells are many; per-animal numbers on the page.

### 8.1 Licks: where they land and the whole path (LICK DATA) — STANDS
- **Landing (soft bins, Fig 3c)**: far-contra licks lose their own bin -0.69 / -0.38 / -0.19 (acute / subacute /
  chronic) to far mid +0.35 / +0.36 / +0.20 (all significant, 4/4).
- **Trajectory softmax (Fig 3d, `scripts/lick_path_softmax.py`)**: each lick's tracked 3-D path (-32..+32 ms around the
  peak) vs every spout's pre-stroke mean path (SD units, LOSO pre templates): far contra own -0.65 / -0.36 / -0.22;
  acute toward far IPSI +0.42 > far mid +0.26, then far mid +0.31 / +0.21. Caveat: acutely every spout's paths move
  toward the far-ipsi template (general forward overshoot, Fig 5), so part of the acute far-ipsi pull is reach, not side.
- **Paths (Fig 5 / 5b, camera bottom + front views)**: mis-landing far-contra licks are on the other spout's course from
  the first tracked point (wrong at launch, not corrected late).

### 8.2 Neural (frozen pre-stroke DECODER) vs executed — STANDS (average); trial coupling breaks acutely
- **Average (Fig 3f / 3g, `scripts/softmax_neural_vs_lick.py`)**: far-contra trials decoded as far contra -0.66 / -0.30
  / -0.22 (lick window; cue -0.59 / -0.27 / -0.17), as far mid +0.26 / +0.14 / +0.11 and far ipsi +0.19 acute, 4/4 —
  the decoder follows the landing. But acutely the neural diagonal drops for EVERY spout (far mid -0.32) while licks to
  those spouts land on target: part of the acute neural change is a general weakening of the code.
- **Trial by trial (Fig 3h / 3i, `scripts/trial_neural_vs_exec.py`)**: within session, neural (P far mid - P far contra)
  follows executed (far mid - far contra) PRE-STROKE in 4/4 (landing r .12-.33, trajectory .18-.41; shuffle p < .001):
  the decoder tracks execution in general. Subacute and chronic keep it (4/4). ACUTELY IT BREAKS (r ~ 0; PS92 .16 the
  only significant). Far-mid landings of far-contra trials: PS92 / PS95 decoded more far mid than far contra acutely;
  PS93 / PS94 stay far contra (as their rare pre-stroke far-mid landings did).
- **Executed best bin vs decoder best match, all trials (Fig 3j, `scripts/exec_vs_neural_match.py`)**: acute diagonal
  falls 4/4 (far mid -.41, far ipsi -.34, close mid -.24): licks that land on a spout are no longer read as that spout;
  mostly recovered chronically.
- **Off-target trials (Fig 3k / 3l)**: the decoder reads the TARGET far more than the executed bin (pre .82 vs .07;
  acute .52 vs .18); by target, close spouts .65-.95 target vs ~.05 executed at every epoch; FAR spouts split acutely
  (far contra .25 target vs .29 executed — the only target where executed wins acutely; far ipsi .21 / .20; far mid
  .38 / .30); far contra stays near even chronically (.45 / .33). The bars do not add to 1: the rest = a third spout.

### 8.3 Neural (frozen pre-stroke ENCODER, relabel test) vs target — PROVISIONAL; trial level DONE 10-25 (label-wide close-mid bias)
- `scripts/relabel_confusion.py` on `frozen_relabel_v7g3_t3d` (Fig 3m / 3n): best-fitting label per session (all
  parcels, lick window): far contra loses its label -.43 / -.45 / -.33 toward far mid +.29 / +.39 / +.18 and close
  contra, nothing survives Bonferroni; OTHER spouts lose their label too (close contra -.58 acute, close ipsi -.35 to
  -.37, far mid -.44 chronic 4/4): a general loss of each spout's pre-stroke pattern (cf. Layer 2), not far contra
  specifically re-aimed. PS93's spouts are poorly separated pre (far contra best in 2/7 pre sessions).
- Per-TRIAL relabel DONE 10-25 01:39 (`frozen_relabel_v7g3_t3d_trial`, 106 / 106; Fig 3o-3q): best label per trial,
  epoch - pre: far contra own label -.14 / -.17 / -.14 (4/4, Bonferroni), far ipsi -.23 / -.21 / -.17, and every spout's
  trials gain the CLOSE-MID label (close ipsi +.20 / +.17 / +.17, 4/4): post-stroke responses fit the most generic template
  best -- a label-wide bias, not far contra re-aimed (far mid +.11 subacute 3/4 only). Executed landing vs encoder label
  shows the same bias; acute far-contra landings exist in PS92 only. Off-target far-contra trials: target ~.2 pre -> .12-.15,
  executed ~.17-.23, another spout ~.65 -- the encoder's trial label is a weak target-vs-executed readout (the decoder,
  8.2, is the better one).

### 8.4 Frozen-encoder residual: pooled maps at calibrated bins, and by area — STANDS (pooled); CORRECTS 3.4's "pooled weak"
- **Bin size**: random pre half-splits — 8 px 0/20 false, 16 px 0/20 (lick) / 1/20 (cue), 32 px 2-4 % of 100 splits
  had any false bin (= nominal 5 %). 32 px primary pooled test, 16 px shown.
- **All 6 spouts pooled (Fig 9b, 32 px)**: lick window 19 / 35 / 30 of 128 bins significant (acute / subacute /
  chronic), EVERY one an increase (data > frozen model): bilateral lateral frontal (ALM / tongue-jaw motor) at every
  epoch, contralesional SSp subacute + chronic; cue 0 / 4 / 4. 16 px: 73 / 116 / 110 of 512. Far contra alone (Fig 9,
  16 px): 25 / 16 / 25 of 512. Across all 32 px tests 413 significant bins up, 2 down.
- **By area, pre-SD units (Fig 11b, `frozen_residual_trend.py` now MOs / MOp / SSp-n too)**: chronic change from pre
  mean: cortex +1.3*; MOs +0.7 (ns) ipsi / +1.4* contra; MOp +1.8* / +2.3*; SSp-m +2.1* / +2.5*; SSp-n +0.9 (ns) /
  +2.5*; contralesional > ipsilesional in every area; contralesional SSp-n jumps most acutely (+3.5*). Pre itself
  drifts up over ~10 days (Fig 11's per-animal trend model handles it).
- **Decomposition pooled (Fig 8b)**: chronic residual +0.32 (ipsi) / +0.43 (contra) of each animal's pre-stroke
  response, 4/4, while the predicted change is ~0.

### 8.5 Layer 2 random-split null — STANDS
- 20 random August-pre pseudo-epochs per animal (`scripts/layer2_random_splits.py`, now saves per split + resumes):
  CHRONIC beats every split in 4/4 animals in cortex, SSp-m left and right (P = 0.00); subacute cortex 4/4 (P <= .05);
  acute 3/4 (PS95 one session). Pooled in null-SD units (Fig 12d): chronic 8.2 / 8.4 / 6.0 null SDs. Pooled vs
  late-pre (Fig 12b): chronic cortex +54.8 [+22.0, +87.4], SSp-m +52.5 / +51.6. Per kernel group (Fig 13b): no group
  beyond the null chronically — the rewrite is spread across groups. Note: 5b's June -> August no-stroke drift group
  (imaging, 10-24) finds the frozen-DECODER chronic deficit within 2-month drift (3/4) but crossnobis distance beyond
  it (3/4); Layer 2's nulls are within-August only.

### 8.6 Pooled target-vs-executed contest — STANDS (chronic)
- Fig 1b / 2b: contest (target - executed) change from pre: chronic +9.0 [+3.6, +16.0] lick, +9.8 [+3.9, +16.2] cue,
  significant at every softmax width; acute lick +12.5 (ns; PS93 the other way).

### 8.7 FaceRhythm in the per-session encoder — PROVISIONAL (FR window owns the factors)
- O2 runs fr_A / B / S / D (master_pre_v1, rank 12; `scripts/fr_encoding_summary.py`, Fig 16-16c). COLLECTION BUG
  FIXED: the four modes had been collected into one folder (each overwrote the last); re-collected to
  `..._frg-{A,B,S,D}`; `combined_encoding.out_dir` now adds `_frg-<mode>`.
- FR adds R2 ~0.1-0.2 and takes about HALF of the unique spout-position R2 pre-stroke (cortex lick 52 -> 30 x1e3). The
  CHRONIC rise in position coding is mostly face-movement-related: lick +22 [+7, +38] -> +4 (ns) with FR; cue +39 ->
  +15 [+3, +27]. Acute SSp-m (ipsi) position loss survives FR (cue -44 -> -32, lick -31 -> -20). Modes A ~ B ~ S ~ D:
  FR side / distance add little beyond its lick amount. Chronic pre-cue face movement explains more cortex (+72).
- **Musall-style with FR (10-25, `scripts/fr_musall.py --mode A`, Fig 16d-16h)**: cortex, lick window, pre: FR alone
  313, unique 135 (x1e3); with FR in the model the unique R2 of uninstructed movement 241 -> 109, task 73 -> 43, position
  52 -> 30 (instructed ~10 either way): FR overlaps mostly with uninstructed movement. Chronic: task / position unique
  111 / 74 without FR vs 51 / 34 with. Lick-window FR unique rises acutely (135 -> 156; cue falls 132 -> 111). Area maps:
  pre FR unique broad, highest in contralesional lateral SSp; chronic full R2 and position / task unique rise most in
  contralesional lateral areas. Alone models are the main run's (FR is not in them); position has no alone model.
  **Pixel maps (Musall Fig. 4 style; Fig 16g-16h, from the stored 2x2-block SSE maps):** chronic position unique WITHOUT
  FR rises broadly (central + bilateral), WITH FR only a focal contralesional anterior-lateral rise survives (task unique
  the same); acutely FR alone / unique rise while uninstructed, task and position unique fall. Descriptive. NOTE: our
  "task-aligned" (alone - drop-one unique) is the overlap with EVERYTHING else in the model, not Musall's overlap with
  task only (R2(task + X) - R2(task)); schematic Fig 16d.
  **Tests (10-25; `scripts/fr_pixel_significance.py`, `scripts/fr_contra_orofacial.py`; R2 as a FRACTION from here on,
  Priya):** pixel test (nested bootstrap, Bonferroni over 514 / 131 bins; null half-splits flag >= 1 bin in 5 %): chronic
  position unique | FR 0 bins (lick and cue) vs 69 / 99 without FR; task unique | FR 13 / 8 up (contralesional lateral);
  uninstructed unique | FR down over ~190-300 central bins in every epoch. A priori areas (SSp-m, MOp, MOs, SSp-n x
  hemisphere, Bonferroni x8): chronic position unique | FR contralesional SSp-m cue +0.037 [0.010, 0.067] 4/4 p_bonf .002,
  lick +0.025 [0.002, 0.051] 3/4 n.s.; ipsilesional ~0; task unique | FR contralesional SSp-m cue +0.041 4/4 p_bonf .002;
  contra - ipsi SSp-m position +0.025 / +0.032, task +0.025 / +0.035 (CIs exclude 0). Uninstructed alone rises in SSp-m
  BOTH sides (lick +0.046 / +0.060, 4/4), falls in MOp / MOs / SSp-ul. Without FR position unique rises broadly (MOp / MOs
  / SSp-ul bilateral, cortex +0.022 / +0.039, 4/4). Per animal: carried by PS94 / PS95 (0.064 / 0.061 cue vs PS92 0.016,
  PS93 0.006); beyond each animal's late - early pre drift in PS94 / PS95, not PS93. Total variance per pixel ROSE
  chronically (not a denominator effect). PROVISIONAL.

### 8.19 Landing-split lick kernels, lagged FaceRhythm, self-fit on the lick-kernel design (10-26 ~04:30) -- PROVISIONAL

Handoff 000z to-dos B / C / D (self-fit half). Table `<E>/landing_lag_unique.csv` (script `S/landlag_unique.py`);
self-fit `frozen_encoder_gate-attempted_v7g3_t3d_frA_ls/session_vs_frozen/remap_boot.csv`. Unique = joint - no_<g>
(frozen, frame measure), epoch - pre, nested bootstrap, n up of 4.
- B, LANDING-split lick kernels (`..._frA_lsland`: six lick-kernel copies weighted by each lick's 3-D landing; licks
  without landing information -> lick_onset@unknown) vs TARGET-split (`..._frA_ls`). Pre: nearly the same split of
  variance (contra SSp-m target unique 0.0128 vs 0.0152 cue, lick kernels 0.0101 vs 0.0120): landing assignment takes
  slightly more of the target term. Post: in the landing frame the contralesional SSp-m TARGET unique RISES
  chronically (+0.010 cue / +0.011 lick, 4/4; CIs exclude 0) while the landing lick kernels LOSE unique variance
  (-0.015 / -0.016, 0/4 up); in the target-split frame target is n.s. chronically (+0.007 / +0.002) and the lick kernels
  fall less (-0.007 / -0.010). Executed spout-locked templates gain in both frames (acute +0.02-0.03, chronic
  +0.003-0.014). Reading: kernels that follow where the tongue LANDED do not absorb the chronic target term -- the
  target term carries something not explained by the landing (consistent with the target being remapped, 8.12-8.14).
- C, LAGGED FaceRhythm (`..._frA_frlag`, lags +-0.1..0.3 s). Pre: past-face columns ("face leads cortex") carry far more
  unique variance than future-face columns ("cortex leads face"): 0.02-0.05 vs 0.001-0.004 -- the expected calcium-lag
  bias, so only the change is read. Change: cortex-leads unique tiny and unchanged (small acute decreases in MOp / MOs
  contra and cortex, 0/4 up; chronic n.s.); face-leads unique RISES acutely in contralesional SSp-m only (+0.009 cue /
  +0.010 lick, 4/4) and otherwise falls (subacute / chronic decreases, mostly n.s.; ipsi SSp-m chronic lick -0.011, 0/4).
  The direction FR term falls chronically in SSp-m (fr_dir -0.008 / -0.015 cue, -0.013 / -0.024 lick). Reading: no
  sign of a shift toward cortex leading the face; the one acute increase is face -> contralesional SSp-m (sensory
  reafference of the changed face movement). Temporal precedence in this design cannot establish causal direction.
- D (self-fit half), session-specific weights on the lick-kernel design (remapping index = self-fit unique minus frozen
  unique, change from pre). Target: contra SSp-m chronic cue +0.005 [-0.002, 0.013] (was +0.014 without lick kernels),
  lick +0.010 [0.000, 0.019] (was +0.018); cortex +0.005 / +0.009 (was +0.016 / +0.021). Lick kernels: ~0 (+0.000-0.004,
  CIs include 0). Spout group (target + executed + lick kernels): cortex +0.027 cue / +0.039 lick (4/4), SSp-m +0.027-
  0.040 -- unchanged or larger than without lick kernels. Reading: the chronically remapped spout information is shared
  between target and spout-specific lick-locked regressors; adding either takes unique credit from the other, so the
  unique-variance contest cannot decide. The lick-kernel partial refit with block-permutation nulls (running, ~10 h)
  is the null-controlled version.

### 8.18 Per-epoch pooled pixel kernel fits: where the event kernels move (10-26 ~00:30) -- PROVISIONAL

`scripts/epoch_kernels.py` (000y item E; Priya 10-25: "what happened to our cross-epoch event kernel maps"), frame
`..._frA_ls_basis` (frozen + FaceRhythm A + per-spout lick kernels, per-animal 200-component pixel basis). Per animal,
sessions pooled into early pre / late pre (halves of the August pre by date; late pre = DRIFT reference) / acute /
subacute / chronic, refit with the FROZEN scaling and penalties (re-derived pre fit = saved fit exactly, 4/4); kernel maps
compared with the early-pre maps (pixel r, window-stacked: target 0.2-1.5 s, lick -0.1..+0.5 s); split-half
reliability per group = the ceiling. MATCHED session counts drop PS95 (one acute session) and leave PS92 at 2 per group,
so the UNMATCHED fit (all of a group's sessions; table mode = all) is read alongside.
- PER-SPOUT LICK KERNELS (reliable: split-half 0.8-0.95): r with early pre falls steadily -- late pre 0.82-0.92, acute
  0.71-0.76, subacute 0.63-0.73, chronic 0.50-0.69 (unmatched, 4/4 animals; matched 3/3 the same) -- chronic below both
  the drift reference and the group's own reliability. Largest at the MIDDLE spouts (close / far mid; PS92 / PS94
  chronic 0.25-0.43) and PS94 close ipsi (0.03); far ipsi stays near pre (0.66-0.92); far contra mixed (0.37-0.76).
  No far-contra-specific pattern.
- CUE kernel: changes sharply acutely (r 0.0-0.5 vs late pre 0.65-0.91) and partly returns; reliability low in PS93.
- TARGET kernels: too noisy to read (split-half 0.13-0.76; late pre vs early pre already 0.35-0.75).
- CAVEAT that decides the reading: chronic sessions are 3-7 weeks from early pre, late pre only days, so a steady
  divergence is what ordinary representational drift predicts (the 10-07 chronic_drift_vs_reorg result: the crossnobis
  distance still rising chronically was drift, not reorganisation). NEEDED: a drift-RATE test -- within-chronic early vs
  late kernel r per day against the pre-stroke rate per day -- before any "lick kernels remapped" claim. Maps at +0.1 s
  also show midline venous stripes post-stroke (vessel contamination, as in 8.13).
Figures `<run>/epoch_kernels/epoch_kernels_r.png`, `epoch_kernel_maps_<lick|target|cue>.png`; page Figs 1uc2 / 1uc3.

### 8.17 Self-initiated (ITI) licks: pre-lick ramp and target-free direction coding (10-26 ~00:15) -- PROVISIONAL

`scripts/iti_licks.py`, `<E>/iti_licks/` (102 sessions, 4 animals; Priya 10-25: "let's also do the inter-trial 3d licks,
pre-lick ramp timing"). ITI licks (whole-session 3-D table, context iti) have no trial, cue or target; the spout is still
in place. Lick times from each session's licks_all.csv via lick_uid (the kinematics table's daq_peak_s is rounded to
4 significant digits -- see 2.6); clock match to the behaviour licks 1 ms in every session. ROIs: ALM / tjM1 = the
anterolateral 40 % of MOs / MOp per hemisphere (Priya: "the anterolateral 1/3 to 1/2 of MOs and MOp").
- RAMP (bout-start ITI licks: no lick for 2 s, no cue / spout move within -2..+0.75 s; onset-aligned, baseline -1.5..-1.0
  s; 427-3177 per animal pre). Pre-stroke activity rises from ~-1 s, earliest in tjM1 (half-amplitude ~-0.35 s) and
  SSp-m, later in ALM / MOs (~-0.15 s). The half-amplitude time (t50) moves LATER post-stroke (tjM1 contra +0.25 s acute,
  4/4) -- but that is an AMPLITUDE artefact: the post-onset response roughly DOUBLES acutely (peak 0.1-0.4 s, in units of
  the animal's pre amplitude at the lick: tjM1 contra +1.43 [0.51, 2.41], SSp-m contra +1.52, MOp +0.85-0.94, ALM
  +0.55-0.73; 4/4) and stays elevated chronically (tjM1 / SSp-m / MOp +0.5-0.8, 4/4). Amplitude-free early ramp
  (-0.6..-0.2 s, same units): acute UNCHANGED (PS93 / PS94 up, PS92 / PS95 down); CHRONIC WEAKER in tjM1 (contra -0.40
  [-0.70, -0.08], ipsi -0.33), SSp-m (-0.46 / -0.35), MOp (-0.23 / -0.19), ALM (-0.15 / -0.12, 4/4 down); MOs and cortex
  CIs include 0. Per animal the chronic drop is PS92 / PS93 / PS94; PS95's tjM1 ramp does not fall.
  BOUT-LENGTH CONTROL (10-26): post-stroke ITI bouts are SHORTER, not longer (licks following the bout start within
  0.75 s: pre 0.42-1.72 per animal, acute 0.35-0.64), and ISOLATED licks (none following) show the same larger
  post-onset response (tjM1 contra +1.41 acute / +0.74 chronic, SSp-m contra +1.56 / +0.68, all 4/4), so the larger
  response is per lick, not a longer bout. Lick kinematics (reach / speed) of these licks not yet checked.
  Reading: after the stroke, self-initiated licks are preceded by less slow build-up in orofacial sensorimotor cortex
  and followed by a larger cortical response (consistent with the chronic positive frozen residual, 4.x / Fig 1 row 2).
  Caveats: pre-lick orofacial movement (jaw / whisking) is not controlled here (FaceRhythm could be regressed out); a
  cortex-wide component rides on every ROI (ROI minus cortex not yet done); calcium lag; ITI lick numbers fall acutely
  (PS95 65 acute bout starts). Trial FIRST licks (cue-triggered, secondary): t50 unchanged acutely, slightly later
  chronically (+0.02-0.05 s).
- DIRECTION WITHOUT A TRIAL (each lick's 3-D lateral direction decoded from area activity, ridge, 5 time-block folds;
  lags 0..0.2 s). Within session: ITI direction r 0.53 pre, unchanged in every epoch (acute -0.015 [-0.10, 0.11]); n-matched
  in-trial licks 0.71, unchanged. CORRECTED 10-26 (Priya: the spout is RETRACTED during the ITI): the first run's
  'within-spout' ITI measure grouped licks by the label of the trial WINDOW they fell in (the trial just finished for
  'tail' licks, the UPCOMING trial for 'lead' licks), which is not a spout in place. Re-run with direction RELATIVE to
  the PREVIOUS trial's spout (removes residual aiming toward the last spout location): 0.32 pre, unchanged acutely,
  +0.066 [0.014, 0.143] chronically. Information rises from ~-0.2 s and peaks ~+0.1 s after the lick peak in every epoch. FROZEN decoder
  (each animal's pre ITI licks; pre = leave-one-session-out): DROPS in every epoch, 4/4 -- direction -0.13 [-0.19, -0.07]
  acute, -0.12 subacute, -0.13 chronic; relative to the previous trial's spout -0.10 / -0.09 / -0.06 (chronic n.s.).
  FROZEN DECODER PER LAG + SESSION-POOLED (10-26, Priya: "KEY results"; page 3.8 / 3.9; one dot per session, nested
  bootstrap animals -> sessions): ITI lateral direction -0.138 [-0.191, -0.075] acute, -0.123 subacute, -0.135
  [-0.213, -0.067] chronic (normalised to each animal's pre: -23 % / -20 % / -22 %); IN-TRIAL licks (decoder trained on
  pre in-trial licks) -0.149 acute but -0.080 subacute and -0.063 [-0.131, -0.009] chronic (-20 % -> -11 % -> -9 %).
  => the pre-stroke mapping largely returns for cued, targeted licks but NOT for self-initiated ones. Pre-stroke ITI
  direction is decodable from ~-0.5 s before the lick peak (r ~0.25), post-stroke ~0.1 (may include the previous lick
  of a bout). Figures iti_frozen_timecourse.png / iti_frozen_summary.png; tests decode_frozen_session_tests.csv.
  Ramp session-pooled (page 3.11 / 3.12; ramp_session_tests.csv): post-onset response up acutely in every ROI, still up
  chronically in tjM1 / SSp-m (raw and normalised); early build-up down chronically in tjM1, SSp-m, ALM.
  Reading: cortex still carries the executed direction of self-initiated licks as strongly as before, but in a
  DIFFERENT mapping from the pre-stroke one -- the execution code without a target is intact but remapped, like the
  trial code (8.12-8.14). This is the target-free counterpart of the far-contra trial result: there the code on failed
  attempts is lost; here, with no target, direction stays decodable session by session.
  Caveat: in-trial within-spout r rises post-stroke (0.15 -> 0.25-0.26) with larger within-spout spread (r depends on the
  behavioural variance); the frozen drop is not explained by variance (more variance would raise r).

### 8.16 Per-spout lick kernels in the frozen model; target vs executed reconciled (10-25 night) -- PROVISIONAL

- Run `gate-attempted_v7g3_t3d_frA_ls` (frozen + FaceRhythm mode A + FROZEN_LICKSPOUT=1: the lick-onset s / d columns
  replaced by six per-spout copies of the recorded lick-onset columns, group "lickspout", executed side). Unique R2 =
  joint - no_<g>, frame, orofacial areas, August pre; epoch - pre nested bootstrap.
- Pre level (cue / lick): contra SSp-m target .015 / .019, lickspout .012 / .015, exec_locked ~0 / .004; ipsi SSp-m target
  .006 / .008, lickspout .013 / .014; MOp lickspout .009-.010 > target .003-.008. Versus the frA frame (no lickspout),
  target unique drops (contra SSp-m .020 -> .015 cue, .026 -> .019 lick): part of the "target" term was spout-specific
  lick-locked activity.
- Epoch - pre (frozen): target acute -.008 / -.014 (1/4), chronic +.007 / +.002 (2/4); lickspout acute -.007 / -.008, chronic
  -.007 / -.010 (0-1/4); exec_locked acute +.023 / +.024, chronic +.008 / +.007 (4/4 in every orofacial area).
- Reconciliation (FINDINGS 0 point 4): old refit "target > executed" was chronic + refit-only; frozen: neither target nor
  executed spout coding recovers; self-fit: target readable in a remapped pattern; partial refit: target AND executed
  remapped (executed at least as much). Pending: self-fit / partial refit on the lickspout design (chain), block nulls.

### 8.15 Trial-by-trial coupling by trial class; what acute far-contra trials look like; nulls (10-25 night) -- PROVISIONAL

- Coupling = within-session r between the frozen decoder's P(far mid) - P(far contra) and the trial's lick weight far mid
  - far contra (landing: 3-D endpoints; trajectory: lick paths; soft weights to each spout's pre landing centre), on
  far-contra target trials, August pre only (June had been in the landing panel; removed).
- NULLS: the per-session / per-cell test is now a BLOCK permutation (runs of consecutive far-contra trials permuted,
  internal order kept); the trial shuffle it replaced gave the same calls in every cell. Spout sequences: runs of ~6
  trials (10-90 % 4-8); one half-session circular shift agrees with the true labels 0.165 on average (chance 0.167) but
  0.12-0.19 per animal x epoch (PS92_1006 0.34) -> a single shift is a noisy null; the partial refit's nulls are being
  redone as 5-draw block permutations (scores averaged).
- Statistic = change ACROSS EPOCHS (Priya 10-25), not r vs 0. All trials: acute -0.25 [-0.39, -0.11] landing / -0.30
  [-0.42, -0.18] trajectory (nested bootstrap on session r); normalised to each animal's pre r: -1.04 / -1.01 (~100 %
  loss); subacute / chronic ~0. Per animal (Welch over sessions): acute < pre in PS93, PS94 (landing p .045 / .030) and
  PS92 trajectory (.004); PS95 has one acute session.
- BY CLASS (`scripts/coupling_by_class.py`): failed attempts acute -0.26 [-0.42, -0.15] (each animal p <= .025), subacute
  -0.29; hits acute -0.02 [-0.43, +0.44] (64 trials; PS92 +0.43, PS93 -0.14, PS94 -0.34, PS95 1 hit), subacute -0.10,
  chronic -0.07. -> the acute loss is mostly composition (92 % of acute far-contra trials are failed attempts).
- Not range restriction: within-session SD of the executed measure is larger acutely (PS92 landing .16 -> .35).
- Decoder posteriors on acute far-contra trials: confident (max P ~.8, entropy ~.3 of flat) but mostly another spout,
  animal-specific (PS93 far ipsi, PS95 close contra, PS92 spread, PS94 half far contra). Off-distribution confidence is
  possible; template-similarity check not run.
- Pre-cue (ENL) frozen-decoder P(true): pre hits .24 / .28 / .51 / .34 (chance .17); far contra lower on failed attempts
  than hits 3 / 4 acute, 4 / 4 chronic -- to test class-matched; and whether the pre-cue lean predicts the lick (coupling
  moved before movement onset; the closest "plan predicts execution" test this design allows).

### 8.14 Partial refit: target vs executed, each against its own null (10-25 night) -- PROVISIONAL

> **10-26: BLOCK-PERMUTATION NULLS DONE** (target_perm / exec_perm: same-spout runs permuted, 5 draws, scores averaged;
> 106 / 106 sessions). They reproduce the single-shift contrasts: chronic contralesional SSp-m target - null
> +0.036 [0.018, 0.058] cue / +0.034 [0.013, 0.055] lick, executed - null +0.055 [0.021, 0.087] / +0.064 [0.023, 0.104],
> cortex executed - null +0.038 / +0.045 (all 4/4); acute ipsilesional SSp-m target - null +0.039 [0.012, 0.069] /
> +0.041 (4/4); every other acute contrast's CI includes 0. Main figure now the line version (page 1yd),
> grouped bars = supplement (1ye). The numbers below are the single-shift ones.

- Design (`scripts/partial_refit.py`, frA frame, 107 sessions, page Figs 1yd / 1ye): per session refit ONLY one group set
  (target; executed = exec_locked + exec_within; + FR side / distance; combinations), everything else frozen, ridge with
  the frozen penalties, 5 contiguous trial-block CV (held-out predictions). Gain = R2 - R2[frozen], epoch - pre within
  animal, nested bootstrap. NULLS: target_shuf (target columns from trial positions circularly shifted by half the
  session) and exec_shuf (executed templates assigned by per-frame spout labels rolled by half the record; exec_angle is
  label-free and stays real -> conservative; added with `--add exec_shuf`).
- Raw gains (cortex cue, acute / chronic): target +0.021 / +0.037, executed +0.034 / +0.055 (4/4) -- but the target null
  gains +0.016 / +0.015 and the executed null up to +0.076 acutely in ipsilesional SSp-m. Read each set AGAINST its null.
- target - null: acute ~0 in contra SSp-m (+0.001) and cortex (+0.005), but +0.037 [0.012, 0.062] 4/4 in IPSILESIONAL
  SSp-m (cue; lick +0.038, 3/4); chronic +0.034 [0.015, 0.059] 4/4 contra SSp-m (lick +0.032, 4/4), cortex +0.022 3/4.
- executed - null: acute ~0 everywhere (SSp-m +0.003 to +0.015, CIs include 0); subacute contra SSp-m +0.020 4/4; chronic
  contra SSp-m +0.054 [0.021, 0.087] cue / +0.064 [0.022, 0.105] lick 4/4, cortex +0.038 / +0.045 4/4 (PS93 / PS95 small).
- executed - target (raw): + acutely in ipsi SSp-m (+0.038 / +0.046, 4/4) -- matched by the executed null, so it is
  flexibility, not execution-specific remapping.
- Reading: acutely a session-specific mapping helps whatever spout labels it is given (structured residual), except a
  target-specific gain ipsilesionally; chronically both target and executed movement are coded in a remapped,
  label-specific way in contralesional SSp-m, the executed part also cortex-wide. Open: (executed - null) - (target - null)
  test; the same on the lickspout frame (chain running; adds lickspout / exec_ls / target_ls sets, no lickspout null yet).

### 8.13 PETH vs trial-by-trial maps, Musall layouts, kernels, remapping time course, PS94 dominance (10-25 evening) -- PROVISIONAL

- PETH vs TRIAL-BY-TRIAL by Allen area (`scripts/frozen_peth_trial_maps.py`, frozen + FR, page Figs 1t / 1u; no-FR 1uf):
  pre-stroke task / spout unique live in the PETH, movement / FR unique in single trials (Musall's split). Post-stroke the
  change is in the PETH: full-model PETH R2 down 30 / 29 areas (acute / chronic, lick, 0 up), task-unique PETH down 16
  chronic; trial-by-trial maps barely move (chronic FR-unique +0.02 SSp-m both sides, movement-unique +0.058 contra SSp-m,
  4/4). Interpretation: the frozen model still tracks moment-to-moment movement coupling; the stroke changes the
  task-locked per-spout average. ALONE models (`--alone`, Musall et al. 2019 top-row layout; Figs 1ua / 1ua2): task /
  spout alone PETH ~0.6-0.8 pre; post full / task / spout PETH fall acutely (30 / 15 / 16), full / task stay down
  chronically; single-trial: acute RISE of lick alone (38 areas) and spout alone (26).
- EVENT-KERNEL beta maps (`scripts/frozen_kernel_maps.py`, frozen pixel fits, Figs 1ub / 1uc; no vessel mask -- Priya: too
  much): target kernels of all six spouts (canonical order) = deviations from the shared cue kernel; dominant pattern =
  distance / ease (close mid strongly positive bilateral lateral frontal / orofacial 0.3-1 s, far spouts negative) plus a
  lateralised component (close contra -> LEFT lateral cortex, close ipsi weak mirror). Ridge credit sharing; lick-onset /
  contact kernels look venous.
- REMAPPING maps + checks (`session_vs_frozen.py --maps`, `scripts/remap_checks.py`; Figs 1x / 1y / 1yb / 1yc): per animal
  >= 3/4 agree in all 41 areas (chronic full / target / spout). Time course: FULL model steps up at day 1 (+0.06, z 4);
  SPOUT / TARGET unique have NO step but rise smoothly with log day (z 3-6.4), equally in non-orofacial areas, larger in
  better-behaving sessions -> cortex-wide (stroke-related or drift: undecided); TASK unique rises in orofacial cortex
  (z 3.1) but not in non-orofacial areas (z 0.2) -- the orofacial-specific part.
- "CONTROL" WORDING (Priya): visual / retrosplenial / trunk SSp = non-orofacial COMPARISON set, not a null.
- PS94 DOMINANCE: pre target-unique R2 contra SSp-m (cue PETH / lick frame) PS92 .004 / .022, PS93 .024 / .020, PS94 .37 /
  .16, PS95 .029 / .004 -> mean target maps are largely PS94's (tests count animals). Per-spout target unique ~0 in PS93
  (close spouts ~1e-6..4e-4). R2 is already normalised per area / pixel, so not a data-scale issue; candidates = block-
  aligned state (positions in ~6-trial blocks; PS94 high running / pushing, lowest full R2), weaker movement regressors,
  or real tuning. CHECKED (`scripts/state_spout_check.py`, `qc/state_spout_check.csv` / `_tracking.csv`): post-cue
  PUSHING is strongly spout-locked in EVERY animal (systematic eta2 .13-.17 vs shift null <= .005; running weakly, PS93 .05)
  -- not specific to PS94 (.14), though PS94 pushes most (9 % of post-cue time vs 2-5 %); tracking not worse in PS94 (3-D
  first lick 97.7 %, single-view 23 %). Free push barely changes PS94's target unique (cue PETH .176 -> .163; others
  unchanged); FaceRhythm already halves it (.37 without FR -> .18). => not block-state, push or tracking; most likely
  stronger spout-specific responses in PS94 (or an unmeasured spout-locked factor). Mean target maps must be animal-
  normalised (the per-spout figures are). Per-spout target vs executed unique maps with animal-normalised means + per-animal panels / tests:
  `scripts/spout_unique_maps.py` (no-FR basis run; page Figs 1ud-1uh): animal-normalised (pre scales target .004-.036,
  executed .07-.18; PS94 highest). TARGET unique per spout changes little (chronic close ipsi up 16 bins; far spouts small
  decreases 2-7). EXECUTED unique changes ACUTELY and spout-specifically: far contra UP in ipsilesional lateral / frontal
  cortex (52 bins), close ipsi / far ipsi / far mid DOWN there (41 / 28 / 22); chronic close ipsi still down (32).
  Consistent with the acute far-contra change being carried by the executed (misdirected) movement.
- Allen-ROI versions of the residual pixel tests (`scripts/residual_roi.py`; Figs 10h / 10i / 10j): every significant area
  an increase, 4/4; cue chronic SSp-m both sides only; lick chronic SSp-m both + contra SSp-n / MOp / SSp-un / MOs + contra
  VISam / VISrl; far contra chronic lick SSp-m both + contra SSp-n / MOp / MOs / VISam.
- Free push on the current frame: see 8.12 (lick-window non-orofacial residual -40 %, cue unchanged).

### 8.12 Per-session self-fit vs frozen (remapping; resolves the scoring caveat) and free push (10-25 late) -- PROVISIONAL

- `scripts/session_vs_frozen.py` (frA frame; `frozen_encoder_gate-attempted_v7g3_t3d_frA/session_vs_frozen/`): each session
  refit on itself (frozen design, 5 trial blocks, penalties chosen in-session), scored with the frozen `musall_r2` windows.
  Pre gap self - frozen ~0.01. Change from pre: full model SSp-m +0.07-0.12 acute, +0.06-0.09 chronic (4/4); target unique
  self chronic +0.017-0.019 SSp-m / +0.010-0.013 cortex (3-4/4) vs frozen -0.006 / -0.008 cortex (0/4); spout unique self
  contra SSp-m +0.054 / +0.059 chronic 4/4; task unique self chronic +0.020-0.025 SSp-m. => the refit chronic target gain
  (8.6) reappears in the frozen windows when weights are refit: the frozen "not reproduced" is a frame effect. Target
  information persists chronically in a remapped pattern.
- Area maps (`session_vs_frozen.py --maps`, `session_vs_frozen/figures/remap_area_maps_<w>.png`, Bonferroni over 41
  areas): full model 27 / 36 areas acute, 40 / 40 chronic (cue / lick, all up). WHAT is remapped changes: acutely the
  movement / FR columns (broad, individually n.s.); chronically spout unique 35 / 38 areas, target 13 / 18, task 10 / 11,
  movement ~0 -- chronic remapping is of the spout / target code, cortex-wide (not only orofacial).
- Free push (`_v7pushfree` vs `_v7fwd15`, older no-FR base; specificity_control on both): cue residual unchanged; lick
  window chronic control residual halves (+1.03 -> +0.52 pre-SD), orofacial contra +2.12 -> +1.83, orofacial - control
  +0.69 (3/4) -> +0.92 (4/4). Pushes explain part of the cortex-wide lick-window residual; the orofacial residual stays.
- Free push on the CURRENT frame (`gate-attempted_v7g3_t3d_frA_pushfree`, 10-25 evening): same answer -- cue residual
  unchanged (orofacial contra chronic +1.42 -> +1.40, contrast +0.88 -> +0.93); lick window chronic non-orofacial
  +1.46 -> +0.83 pre-SD, orofacial contra +2.35 -> +1.95, contrast +0.53 -> +0.79. Pushes carry ~40 % of the lick-window
  residual in non-orofacial cortex and a smaller share in orofacial cortex; no headline claim depends on push.
  Allen-ROI (residual_roi): lick chronic significant areas 13 -> 7 with free push (posterior / visual drop out except
  contra VISam; SSp-m both, contra SSp-n / MOp / SSp-un / MOs stay), cue unchanged (3). Behaviour-matched residual
  still positive in SSp-m (lick +3.7 / +4.1 x1e3, 3-4/4; without push +5.5 / +5.6).

### 8.11 Is the chronic change specific to orofacial cortex? Does it track recovery? (10-25 afternoon) -- PROVISIONAL

**WORDING (Priya 10-25): the "control" areas are a non-orofacial COMPARISON set, not a null -- widefield movement,
arousal and stroke (diaschisis) effects reach visual / retrosplenial cortex too. They show where a change is concentrated;
a cortex-wide change is not thereby drift or artefact.** Frozen frame with FaceRhythm (`gate-attempted_v7g3_t3d_frA`; Layer 2 `allsession_encoder_v7g3_t3d_frA`). Orofacial = SSp-m,
SSp-n, MOp, MOs; CONTROL = visual (7 areas), retrosplenial (3), SSp trunk / lower limb; both hemispheres; group = mean over
the areas present in all of an animal's sessions. Residual = data - frozen prediction, evoked, mean of six spouts, in each
animal's pre-SD units per area. Scripts `scripts/specificity_control.py` (outputs
`imaging_encoding/specificity_control/<tag>/`), `scripts/remap_lmm.py` (`imaging_encoding/remap_lmm/<tag>/`).

- **LAYER 2 IS NOT OROFACIAL-SPECIFIC.** Chronic change beyond pre_late: orofacial +0.053 [0.033, 0.074], control +0.052
  [0.028, 0.075] (4/4 each), contrast +0.002 [-0.017, 0.018]; per-area map uniform, every area's CI excludes 0 (visual and
  retrosplenial included). Mixed model (time): control areas NO step at the stroke (-0.003) but a steady rise with log day
  (+0.016 / e-fold, z 6.1); orofacial - control has an ACUTE step only (+0.025, z 2.6) that decays (-0.007 / e-fold, z -3.0).
  Behaviour model: the change is LARGER in better-behaving (later) sessions (control -0.007 / unit ttc deficit, z -5.0).
  -> a cortex-wide change accumulating with time: a real stroke effect reaching all of cortex, or drift on a longer time
  scale than the 8-day August null -- undecided; the orofacial-SPECIFIC part is an acute, transient excess.
- **RESIDUAL: partly global, partly orofacial; a disruption that tracks the deficit plus a persistent part.** Chronic,
  pre-SD units: orofacial contra cue +1.42 [0.70, 2.36] / lick +2.35 [1.65, 3.07] (4/4); control cue +0.25 (n.s.) / lick
  +1.46 [0.35, 2.54] (3/4). Orofacial - control: cue +0.88 [-0.005, 2.16] 3/4, lick
  +0.53 n.s. Area map: cue residual = bilateral lateral frontal / orofacial cortex; lick residual also posterior (contra RSP
  / visual). Mixed model (ttc): orofacial residual steps up at day 1 (+2.7 to +3.3 SD, z 6.8-7.9) and DECLINES with log day
  (z -2.4 to -6.0); larger in sessions with a larger deficit (+0.16 to +0.36 SD / unit, z 2.6-5.5); still present at full
  behavioural recovery (contra cue +1.33, lick +2.31, z 5.0 / 8.3); orofacial - control at recovery, cue +0.73 (z 3.2),
  lick +0.39 (n.s.). Contact deficit: same signs.
- A global-gain correction (control-area scale removed) was tried and DROPPED (Priya 10-25: not needed; it assumes one
  multiplicative scale shared by all areas and was unstable, PS94); the orofacial - control contrast is the test.
- **DEFICIT SLOPE IS CARRIED BY FEW SESSIONS.** Figures `remap_lmm/<tag>/figures/remap_lmm_residual_<window>_behaviour_ttc.png`:
  the positive residual-vs-deficit slope rests on a handful of high-deficit acute sessions -- PS94 (deficit ~7, large
  residual) drives it; PS92's high-deficit sessions (~5) go the OTHER way (slightly negative). So "tracks the deficit" is
  NOT 4/4; "persists at recovery" (the fitted level at deficit 0, where most post sessions sit) is the robust part.
  Layer 2 vs days (`remap_lmm_layer2_time.png`): control areas rise steadily ~0.02 -> ~0.08 over 50 days in every animal;
  the orofacial excess over control is confined to the first ~5 days (PS93, PS94).
- **BEHAVIOUR-MATCHED RESIDUAL (analysis 4, `scripts/matched_residual.py`, `imaging_encoding/matched_residual/<tag>/`):
  NOT an extrapolation artefact.** Per animal x spout envelope = Mahalanobis distance on the first lick's 3-D endpoint,
  reach, peak speed, 3-D peak time and the trial's lick count <= the pre trials' leave-one-session-out 95th percentile.
  Share of 3-D-tracked trials inside: chronic .96 / .93 / .72 / .78 (PS92-95), acute .81 / .75 / .35 / .77; far contra acute
  .24-.46, chronic .85-.96. Residual (per-trial window means, x1e3 dF/F) on inside-envelope trials ~ all trials: chronic
  SSp-m lick +5.5 / +5.6 (ipsi / contra) 4/4, cue the same pattern; acute inside-envelope residual is if anything LARGER
  (PS94). The frozen model is not being asked to extrapolate chronically, and the residual does not depend on the trials
  it would have to extrapolate on.
- **No-FR frame repeat (`gate-attempted_v7g3_t3d`, Layer 2 `_ref`): same answers.** Layer 2 chronic orofacial +0.050 vs
  control +0.054, contrast -0.005 [-0.036, 0.019]; residual orofacial - control cue chronic +1.33 [0.17, 3.09] 4/4 (STRONGER
  without FR), lick +0.80 n.s.; at full recovery (mixed model, ttc) cue orofacial - control +1.09 pre-SD (z 4.0); matched
  residual holds (chronic SSp-m contra cue +4.9, lick +6.6 x1e3, 4/4).

### 8.10 FROZEN encoder WITH FaceRhythm; every post-stroke change claim redone in the frozen frame (10-25 day)
WHY: the FR results (8.7, 8.9) and the target-vs-executed contest (2.4b, 8.6) used the PER-SESSION REFIT encoder, which
re-learns every session's weights, so a post-stroke change there mixes changed movement with changed cortex; the frozen
encoder (one model per animal from its August pre sessions, pre held out leave-one-session-out) had no face movement. Code:
`frozen_encoder.py` FROZEN_FR=1 (+ COMBINED_ENCODING_FR=master_pre_v1, COMBINED_ENCODING_FR_GROUPS=master_v1:A; groups fr =
lick sum + other, fr_dir = side / distance; movement super-group; own penalty) and FROZEN_FR_SPLIT=1 (FR split into spout-
locked template + deviation like pose; groups fr_locked / fr / fr_dir; super-groups posinfo / noposinfo across pose + FR).
Runs: `M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/frozen_encoder_gate-attempted_v7g3_t3d_frA` (+ Layer 2 `allsession_encoder_v7g3_t3d_frA`, relabel
`frozen_relabel_v7g3_t3d_frA`, decomposition `qc/lp_encoder_decompose_frozen_encoder_gate-attempted_v7g3_t3d_frA_joint*`).
Scripts: frozen_fr_compare.py, frozen_fr_musall.py, frozen_claims.py, frozen_posinfo.py, frozen_pixel_unique_sig.py
(outputs `M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/frozen_fr_compare/`). Page section 0 + "Where things stand" (v53).
- FR raises pre-stroke frozen R2 (cortex cue 0.44 -> 0.60, lick 0.34 -> 0.53; LOSO areas-summed PS92 0.68 vs 0.64 no_fr).
- STANDS with FR: acute far-contra drop carried by movement (target ~0); chronic residual (SSp-m +9.5 / +10.6 cue, 4/4);
  Layer 2 chronic change beyond drift (cortex +0.054, SSp-m +0.059 / +0.061, 4/4); relabel pattern.
- REVISED: target vs executed (frozen target unique falls and stays down; executed spout-locked part up); contralesional
  position rise (not reproduced; SPOUT package contra - ipsi +0.039 / +0.048 with FR, 4/4); PETH loss does not recover
  chronically; recovery course 2/4 up (cortex at / after behaviour). Pixel maps (no FR, basis): position unique only
  decreases (few bins), spout unique up acutely (31 bins, lateral), full R2 down acute (377) / chronic central (69); per-
  animal z and per-animal session-level views agree; calibrated 4-6 % (raw). Per-animal session-level view: only the chronic medial full-R2 fall reaches >= 3 of 4 animals (20 bins at 16 px); the acute spout rise is 1-2 animals per bin. Page Figs 1l-1m (v54).
- FR unique in the frozen frame (per regressor): cortex directionless 0.008 vs direction 0.001; SSp-m 0.004 vs 0.003 --
  "almost all directionless" holds cortex-wide, NOT in SSp-m. Direction unique falls chronically in contra SSp-m (0.013 ->
  0.003 cue), but Layer 2 change_fr_dir <= pre_late drift: MIXED behaviour / neural.
- Movement with vs without lick-position info (quick, FR lick sum unsplit): pre SSp-m alone 0.46 / 0.46, unique 0.03 vs
  0.09; chronic noposinfo unique up in SSp-m (ipsi +0.050 [0.023, 0.074] 4/4), posinfo alone down contra (-0.065, 0/4).
- Refit FR maps with PER-ANIMAL z (`fr_pixel_significance.py --zscore`): counts kept or raised (chronic position no FR 99 ->
  188, task | FR 8 -> 28, position | FR 2 -> 13) but the z test is liberal (9 % / 8.6 % false-bin splits vs 5 %).

### 8.9 FaceRhythm: trial-to-trial vs PETH, direction vs directionless, total variance (10-25 night) — PROVISIONAL
R2 as a FRACTION from here on. FRAME: per-session refit encoder (FR factors frozen master_pre_v1; encoder weights per session).
- **Trial-to-trial vs PETH** (`scripts/fr_trial_peth.py`, O2 fr_A_tr, 107 sessions; Fig 16l-16m): pre cortex, FR unique 0.21-0.24
  and uninstructed 0.12-0.14 are TRIAL-TO-TRIAL (PETH ~0); task / position unique 0.07-0.10 are PETH (trial ~0). ACUTE spout loss is a
  PETH loss (position unique PETH cortex lick -0.050 [-0.131, -0.004], ipsilesional SSp-m -0.016 [-0.035, -0.004], 0/4 up) while
  trial-to-trial position rises slightly (cortex +0.013, 3/4); uninstructed unique trial-to-trial falls (cortex -0.07, 4/4). Chronic
  contralesional SSp-m rise splits (PETH +0.014-0.018 2/4, trial +0.020-0.022 3/4), n.s. with these windows (cue 0-1.5 s, first lick
  -0.1..1.0 s; the encoder's own windows give +0.037: window-sensitive).
- **Direction vs directionless** (`scripts/fr_direction.py`, modes B / S / D / A; Fig 16n-16o): FR gain ~all directionless (cortex
  0.13; direction +0.005; SSp-m +0.01 of 0.06). Direction factors' added fit in SSp-m FALLS acutely (cue -0.005 / -0.007, 0/4 up) and
  chronically (-0.010, 0/4); they take less of the position code post-stroke. The contralesional SSp-m position rise is present with
  DIRECTIONLESS FR (cue +0.028 [+0.002, +0.058], 4/4; lick +0.022 3/4 n.s.). 
- **Why the direction factors decouple** (`scripts/fr_direction_mechanism.py`, MODEL-FREE: raw FR factors, bundle 3-D tongue
  and cortex, engaged trials, cue 0-1.5 s; Fig 16p): (a) behaviour: side-index spout eta2 pre 0.72, acute -0.10 [-0.20, -0.01];
  tongue ml eta2 pre 0.86, acute -0.12 [-0.19, -0.04] 4/4, subacute -0.07, chronic back; within-spout SD of side index chronic
  -21 % [-38, -4] (3/4). (c) measurement: within-spout r(side, tongue ml) pre -0.23, acute +0.15 [0.01, 0.30] (weaker), sub /
  chronic back. (b) cortex: partial r(SSp-m, side | lick) ipsilesional pre -0.10 -> +0.12 acute, +0.13 subacute, +0.08 [0.02,
  0.15] chronic (4/4 toward 0 = coupling lost); contralesional ~0 pre, n.s. chronic; contra - ipsi pre 0.19, chronic -0.07
  [-0.11, -0.03] 4/4. Distance index: no change. READING: acute = face + measurement change; chronic = cortical decoupling
  (ipsilesional SSp-m) plus less within-spout side movement; not a measurement failure. |r| <= 0.2. The frozen-with-FR encoder
  is the model-based check (handoff 000t plan item 3).
- **Contralesional SSp-m position rise across FR groupings** (`scripts/fr_contra_orofacial.py --mode A|B|S|D`; non-A in
  `contra_orofacial_mode<X>/`): chronic position unique | FR, contralesional SSp-m, cue window: A +0.037 [0.010, 0.067]
  (p_bonf .002), B +0.028 [0.002, 0.058] (.16), S +0.026 [0.004, 0.053] (.05), D +0.038 [0.005, 0.073] (.06), all 4/4;
  lick window n.s. in every mode; contra - ipsi (lick) +0.031-0.035, CI excludes 0 in all four. The rise does not depend on
  how FR's lick direction / distance enters; only mode A survives Bonferroni x8.
- **Total variance** (`scripts/variance_maps.py`, data only; Fig 16p): every significant bin is an INCREASE (null 0-2 %); acute lateral
  orofacial both sides + posterior; chronic cortex x1.54 [1.37, 1.73] lick 4/4, VISp x1.83, RSPd x1.72 (global); R2 not raw variance
  is the measure.
- **Executed vs decoded by target** (`exec_vs_neural_match.py --target`; Fig 3m-3n): see FINDINGS 4.

### 8.8 Variance explained (Musall style) and R2 maps on the final frame — descriptive
- `frozen_musall_figures.py --tag gate-attempted_v7g3_t3d` (Fig 14 / 14b; MOs / MOp / SSp-m / SSp-n per hemisphere
  on the share): uninstructed movement dominates single-trial / frame R2, mostly task-independent; spout explains the
  trial-averaged response, almost all task-aligned. Pixel R2 (Fig 15-15e): frozen full R2 falls over most of cortex
  chronically. Colorbars added to r2 and trial-type maps (r2 redraw running).

## Change log

- **2026-10-25** — reviewer fixes: current acute far-contra attempt rates with denominators + mechanism (PS94 try and
  fail, PS95 mostly fail to try) in the top box; 2.4b local-run bullet marked superseded by the FINAL box; 10-23 Layer 2
  bullet WITHDRAWN + change-log entry; pre-cue scope (far contra vs all spouts) stated in 3.3 and 5; stale counts
  (3-D / cam1, v7basis, cam4-only, cam1 "small licks", FR 11.8 %, cue z) updated; status table rows for 2.7-2.9, Layer 2
  and section 8.

- **2026-10-25 (night)** — section 8 added: trajectory softmax; neural decoder vs landing / trajectory (average,
  trial-level coupling breaks acutely, executed-vs-decoder matrix, off-target by target); relabel confusion
  (session level; per-trial running); residual at calibrated 16 / 32 px bins (pooled now strong, all increases) and by
  area incl. MOs / MOp / SSp-n; Layer 2 random-split null; pooled versions of the per-animal figures; FR-in-encoder
  (collection bug fixed); Musall / R2 maps on the final frame. Repo-wide consolidation into docs/FINDINGS.md agreed
  with the FR (e3) and pose (5b) windows.

- **2026-10-24 (Layer 2)** — 10-23 Layer 2 result WITHDRAWN (gain measured distance from an all-epoch compromise);
  redesigned with an early-pre / all-August-pre reference (change_ref) and late-pre + random-split nulls: chronic
  change > null 4/4, no single group carries it, grows with time (2.5, 8.5).

- **2026-10-24 (3-D)** — 4-animal 3-D polar contest (2.4b): pre target-led 3/4 (later 10-24: a CAPACITY effect of
  polar's target set; the binned contest, now primary, shows no pre target lead); chronic target above pre 4/4; the
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
