# What in the FACE carries the pre-launch aim, when, and does face movement explain the decoder offset? Design written BEFORE any number (2026-10-31)

Follow-up to S26 (`PRELAUNCH_AIM_TRACKING_PREREG_20261031.md`, FINDINGS §4 item 24, page 14.18 S26). There, a frozen
across-spout launch-angle decoder (pooled August pre, window -0.50..-0.05 s from tongue emergence) showed no cortical
tracking of the stroke-deviated launch. A FaceRhythm-only decoder (5 master_v1 groups, no "other") read the pre-launch
angle far better (cued r 0.44-0.52) and tracked the deviated launch acutely (TI 0.49 [0.24, 0.80], 4/4). With the face
partialled, cortex was calibrated in no animal.

Five questions (Priya 10-31):
* **Q1 WHAT** in the face carries the pre-launch angle and its post-stroke deviation (jaw lateral steering, nose,
  whisker pad, tongue in the mouth)?
* **Q2 WHEN**: before jaw opening (plan-like) or only once the jaw moves (execution)?
* **Q3 RECOVERY**: does the face-decoded deviation recover with the launch aim? Is it a behavioural marker of the deficit?
* **Q4 WITHIN-SPOUT (pre-stroke)**: does pre-launch face activity read within-spout launch angle, where cortex failed
  (S25 / S25b)?
* **Q5 CONTROL**: does partialling FaceRhythm out of the cortical features change the post-stroke frozen-decoder
  OFFSET? Prediction stated now: **unchanged**, because the offset steps up at lick onset and the face effect is
  pre-launch.

Code (stage 2): `scripts/face_prelaunch_aim.py` (subcommands `extract` / `fit` / `offset` / `summarise`, `--plot-only`;
figdata pkl). Outputs only under `M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/
imaging_encoding/face_prelaunch_aim/` (`writeguard.assert_writable`).

**What was looked at before writing this:**
* S26's prereg, outcome and code (`prelaunch_aim_tracking.py`: `run_decoder`, `loso_multi`, `calib`, `track_licks`,
  `tracking_stats`, the face / face-residual decoders);
* `iti_licks.fr_unblanked`, `face_rhythm_outputs` (`combined_series`, `fr_signals`), `fr_groups.master_v1`;
* the master_pre_v1 proj file layout and `plan.json`, and the FR run's `params.json` (VQT parameters);
* the `vqt` package's filter construction (`vqt/helpers.py`, fr env);
* the v36 per-session `clean.npz` / `windows.npz` layout, the `session_poses/pose3d/*_frames.csv.gz` header;
* the jaw handoffs (10-15, 10-21);
* `lick_sspn_checks.check3`, the headers of `lick_iti_dock_compare.py`, `lick_matched_readout.py`, `likeforlike.py`,
  `lick_balance_matched_error.py`, `iti_lick_order.predictions`, `iti_where_decoder.area_basis`;
* FINDINGS "DOCKED-SPOUT ITI LICKS" and "ITI pre-lick window";
* BEHAVIOUR / DATA-AVAILABILITY counts only (section 0).

**Known before writing (stated so it is not mistaken for a fresh test):** S26's FR face decoder result above (cued gate
r, acute TI 0.49, 4/4). **No value of any NEW decoder, partial regression or offset in this design has been computed.**

## 0. Data availability (counted 10-31 on S26's 107 sessions / lick sets; no outcome values)
Lick sets are S26's, from its feature cache `prelaunch_aim_tracking/features/<label>.npz`:
* CUED: pre 1750 / 2751 / 3178 / 3611 licks (PS92 / 93 / 94 / 95);
* JAW-referenced (cued licks with a jaw onset): 1432 / 1891 / 2666 / 3275 pre;
* ITI docked: 1000 / 1637 / 1366 / **160** pre (PS95 acute **28 licks, 1 session**).

* **FaceRhythm sampling in the pre-launch window (-0.50..-0.05 s).** `combined_series` = 25 Hz trial windows
  (cue -2 .. +4 s of projected engaged trials), 5 Hz session bins (0.2-s means) everywhere else.
  * CUED: 99.6-100 % of licks have the whole window inside a 25 Hz trial window. JAW-referenced: 100 %.
  * **ITI docked: 0 %.** Every ITI pre-launch window is in the 5 Hz session bins. Its 0.05-s bins are linear
    interpolations between 0.2-s means, so ITI FR timing is resolved to 0.2 s at best.
  * FR projection: missing only on a few PS92_0908 licks (0.5 %, its lossy bins); every other session complete.
* **FR is ACAUSAL in time, and this bites in exactly this window.** The VQT filters are centred Hann windows of length
  Q(f)/f, with Q geometric from 4 (1 Hz) to 20 (60 Hz) over 36 log-spaced bins (`params.json`; `vqt.helpers`).
  * At the lick factors' frequencies (5.8-8.2 Hz) the window is 1.1-1.4 s long: support ±0.56-0.69 s, half-height
    half-width ±0.28-0.34 s.
  * Nose factors (1.3 Hz): ±1.7 s support. prep_jaw (2.9 Hz): ±1.05 s. whisk_run (14.7 Hz): ±0.39 s.
  * So **every FR lick-factor bin of the PRE window (and much of BASE, -0.80..-0.50) lies inside the filter support of
    the launch itself**. A tongue / jaw movement at 0 s leaks into FR values before 0 s.
  * **S26's face result can therefore be partly or wholly the launch smeared backward in time.** This is the main reason
    the design below adds timing-valid face measures and makes them primary.
* **Timing-valid face measures exist for every lick:**
  * **Point-displacement PCs** (`proj/<key>.npz` `points_pc`): frozen pre-stroke PCA (20 PCs) of the FR mesh point
    displacements, **25 Hz over the whole session** (dt 0.040 s), each 25 Hz value a 40-ms bin, no spectral smoothing.
    >= 10 samples in every PRE window of every lick, all three sets.
  * **cam4 cleaned pose at 250 Hz, whole session** (v36 `clean.npz` + `windows.npz` `ix_daq_s`):
    * jaw x / y, measured = fill 0 and finite. >= 80 % of PRE-window frames measured in 93-99 % of cued licks per
      animal, 98-100 % of ITI and jaw-referenced licks.
    * Exception: close_mid cued, median 0.96, 25th pct 0.72 (the chin under the spout).
    * Raw nose point: likelihood >= 0.9 in >= 80 % of frames for ~100 % of licks.
  * **3-D jaw / nose (cam1 + cam4)**: `session_poses/pose3d/<a>_<d>_frames.csv.gz` exists for all 107 sessions
    (jaw_ml / ap / dv mm, + ml = mouse right = contralesional). Per-lick coverage is counted in stage 2 (both cameras
    must be measured; close_center is masked).
* **Jaw onsets**: `qc/jaw_onsets.csv`, CUED only (S26's jaw set). **No jaw onsets for ITI licks.**
* **Tongue inside the mouth is NOT measurable**: cam4 / cam1 tongue = mouth-zero fill until the tip is visible. It can
  only show indirectly, through mouth-region mesh points / FR lick factors. That is stated wherever "tongue" is discussed.
* **Q5 inputs exist**:
  * `iti_licks/`: `lick_order_preds` cache, `offset_contrib/terms_licks_*`, `covref_covariates.csv.gz`,
    `itifilter_dock/` (matched_loclin, balmatch), `recmark_sessions.csv`;
  * the published headline values (docked ITI same-place offset +0.30 / +0.22 / +0.29 mm; docked matched-error
    +0.22 / +0.18 / +0.22).
* **Frozen store**: `C:\Users\SabatiniLab\frozen_models` (S26 printed it holding decoder / encoder / enl / lick /
  prelaunch / reembed / state kinds). Re-checked and printed before any fit.

## 1. Licks, angle, target (fixed; S26 section 1 unchanged)
* Sessions: S26's 107 (curated, PS92-95, August-only pre, cohort 2 never).
* Lick sets: CUED, JAW-referenced cued, ITI docked, exactly as cached by S26. No re-selection; meta read from the cache.
* Angle = theta_L (deg, + = contralesional = mouse right).
* Target cell: CUED = spout; ITI = previous trial's spout.
* Q5 uses the offset pipeline's own lick sets (section 7).
* Per family, a lick enters only if all its features are finite (rules in section 2). Each family's n is reported, plus
  a like-for-like intersection sensitivity (licks finite in every family).

## 2. Face signals and feature families (fixed)
All signals are sampled on S26's 10-ms FINE grid (-0.795..+0.295 s around the reference time, linear interpolation),
NaN across source gaps, and averaged into S26's 22 bins of 0.05 s (-0.80..+0.30).
* S26 sampled FR at bin CENTRES only. That is reproduced once from S26's cache (the S26 replication row) and changed to
  bin means for every new decoder. Bin means are the correct analogue of the 40-ms points / 4-ms pose bins.

Windows: **PRE -0.50..-0.05 (primary)**, **PRE10 -0.50..-0.10**, **POST +0.05..+0.30 (positive control)**.
* Each bin minus the BASE (-0.80..-0.50) mean (S26). This removes static images: posture, and in CUED the stationary
  spout at the target.
* No-baseline levels (PREnb) = sensitivity. It is **confounded in CUED by the visible spout at the target**, stated.

Reference times:
* tongue onset (CUED, ITI);
* jaw onset (JAW-referenced cued set; S26's `t` for that set = cue + jaw_onset_ms).

Families (each a separate decoder; features = bins x traces, flattened):

| family | traces | timing | role |
|---|---|---|---|
| **PTS** | point-displacement PCs 1-20 (frozen pre PCA, whole face mesh) | 40 ms | **timing-valid face, PRIMARY** |
| **JAW** | cam4 jaw lateral `-(jaw_x - jaw_rest_x)` (px, **+ = contralesional = image LEFT**), jaw opening `jaw_y - jaw_rest_y`, and both velocities (10-ms differences, bin-averaged) | 4 ms | **timing-valid jaw, PRIMARY** |
| FR-all | master_pre_v1 absolute expression, `fr_groups.master_v1` mode A: lick (sum), left, right, close, far + the six "other" factors singly (0 sniff_prelick, 1 nose_prelick_slow, 3 nose_steps, 6 run_onset_jaw, 9 whisk_run, 10 prep_jaw) = 11 traces | acausal (section 0) | S26 extension; LEAK-EXPOSED |
| FR groups | lick; left; right; side (left + right); close; far; other (6); each "other" factor alone | acausal | Q1 decomposition |
| NOSE | cam4 raw nose x / y (lk >= 0.9, else NaN) + velocities | 4 ms | Q1 decomposition |
| JAW3D | 3-D jaw ml / dv (mm, both cameras measured) + velocities | 4 ms | JAW sensitivity |
| FACE-ALL | PTS + JAW + FR-all | mixed | descriptive |

Finite rules:
* a bin is NaN if < 50 % of its fine samples are finite;
* a lick enters a family only if every bin it uses is finite, and its BASE bins too where BASE is subtracted.
* JAW at close / centre spouts: the chin is partly covered by the spout, and the jaw point lands on the visible edge
  (10-15 handoff). Base subtraction removes the static part. Any residual is a spout-geometry effect and is stated.

**"Beyond jaw" variants (Q1 key secondary):** PTS and FR-all features residualised on the JAW family's features of the
same window.
* OLS fitted on all PRE licks of the animal with both families finite; label-free, frozen with the decoder.
* PRE sessions are residualised with a regression fitted LEAVING THAT SESSION OUT (LOSO-consistent); post sessions with
  the all-pre fit.

**Mesh regions (OPTIONAL, decision needed, section 11):**
* PTS reconstructed per face region (nose / whisker pad L / whisker pad R / mouth-jaw). The 20-PC reconstruction is
  linear: region displacement = Σ_k pc_k · comp_k[region points].
* Regions must be drawn on the MASTER face and fixed BEFORE any region decode. Not run otherwise.

## 3. Decoder (fixed; S26 section 3 reused verbatim)
* Ridge on standardised features, alpha from 10^{-1..5} (13 values) by inner leave-one-session-out CV, pre scored
  leave-one-session-out, all-pre model applied to post (S26 `run_decoder` / `loso_multi`).
* Minimum counts as S26: >= 3 pre sessions and >= 150 pre licks per animal x arm x family; a session enters epoch
  tables with >= 10 licks; a cell with >= 3.
* **FROZEN** (rule 10): new kind **`face_aim`**, LOCAL store only.
  * `make_spec(animal, "face_aim", align=f"{arm}_{window}[_jawref]", source=f"{family}[_{variant}]", train_labels=<August
    pre>, n_features=...)`.
  * The payload holds the all-pre model + LOSO siblings + SHA-1 of (X, y, cell), with a loud warning on mismatch.
  * Residualising regressions are frozen in the same payload (kind `face_aim`, the variant in `source`).
* `local_dir()` is printed and must exist and already hold other kinds, else stop. The smoke test (PS93, 20
  permutations) uses a scratchpad store (`WIDEFIELD_FROZEN_MODEL_DIR`) and scratch outputs.
* Within-spout decoders (Q4) never score post sessions and are not frozen.

## 4. Gate, calibration, controls (computed and printed BEFORE any post-stroke statistic)
* **Gate** per arm x family x window:
  * pooled LOSO r vs **200 permutations of y within session**, whole label pipeline refit;
  * p = (1 + #{null >= obs}) / 201;
  * **pass = p < 0.05 in >= 3 / 4 animals** (S26).
  * Run for PTS, JAW, FR-all, FR groups, NOSE, JAW3D, beyond-jaw (PRE, PRE10, POST) and the pre-jaw window.
  * Single-bin decoders: gate r and calibration only, no permutations (S26).
* **Calibration** b_sp (session x target-cell mean slope, pre LOSO predictions), b_all, SD ratio, b_sp CI over pre
  sessions. b_sp <= 0 or CI including 0 -> "not calibrated", TI not estimable for that animal (S26).
* **Positive controls:**
  1. POST window: each family must pass the gate (the jaw and face follow the executed lick).
  2. JAW POST at far spouts: the known pre-stroke far-pair jaw steering (FAR_DIR > 0 in 4/4, 10-15 handoff) must
     appear as a positive jaw-lateral difference far_contra - far_ipsi in pre licks. Printed.
  3. Q4: within-spout POST.
  4. Q5: planted signal (section 7).
* **Reproduction check**: S26's `face:PRE` decoder rebuilt from S26's cache must reproduce S26's cued r and acute TI
  exactly (same frozen model -> frozen-hit). FR-all differs from it by design (6 "other" factors, bin means).
* A family / arm that fails the gate is DESCRIPTIVE ("gate failed"). A family / arm whose POST control fails: its PRE
  result is uninformative.

## 5. Q1 -- WHAT carries the pre-launch angle and its deviation (primary tracking)
Tracking index exactly as S26 section 5:
* d_dec* = (decoded - pre mean decoded at the target) / b_sp; d_exe = theta_L - pre mean at the target;
* TI = Δd_dec* / Δd_exe (Δ = epoch - August pre);
* `boot_delta_columns` (nested animals -> sessions, seed 0); H0 TI = 0 tested on the numerator (`boot_p` of Δd_dec*);
  H1 TI = 1 on Δ(d_dec* - d_exe);
* estimability: Δd_exe CI excludes 0 and |Δd_exe| >= 3 deg.
* Per-animal table first: n sessions / licks, d_dec*, d_exe, per-animal TI, calibrated.

**PRIMARY FAMILY A (Holm over 6; TI numerator p):**
| cell | arm | family | window | epoch |
|---|---|---|---|---|
| A1 | CUED | PTS | PRE | acute |
| A2 | CUED | PTS | PRE | subacute |
| A3 | CUED | JAW | PRE | acute |
| A4 | CUED | JAW | PRE | subacute |
| A5 | ITI | PTS | PRE | acute |
| A6 | ITI | PTS | PRE | subacute |

Why these 6:
* PTS and JAW are the timing-valid measures, so they are the only ones whose pre-launch value is not the launch smeared
  backward.
* Acute / subacute are where the execution change is clear (S26: chronic cued Δd_exe CI included 0).
* ITI PTS has no cue and 25 Hz face sampling.
* FR-all is NOT in the family. It is S26 extended, and S26 already saw its cued result.

A **tracking result** = numerator CI excludes 0 AND >= 3/4 animals same sign AND Holm p < 0.05 (S26). Gate-failed /
POST-failed cells are reported but cannot be a tracking result.

Secondary (descriptive, uncorrected, per-animal first):
* all families x {CUED, ITI} x {PRE, PRE10, POST, PREnb} x {acute, subacute, chronic};
* FR groups one by one, and **drop-one-group** from FR-all (gate r loss + TI);
* **beyond-jaw** PTS / FR-all (gate + TI);
* NOSE and JAW3D;
* far-contra only (Δ decoded vs Δ executed at far contra, calibration from the full set; S26 deviation 2);
* dose control (cell-level Δd_dec* on Δd_exe with intercept, S26 section 6);
* history regression (lick-level d_dec* on current d_exe and the previous lick's theta_L deviation). This matters for
  FR, whose ±0.7 s support can reach the previous lick at a 1-s gap.
* the like-for-like intersection lick set.

Reading Q1:
* "**Jaw lateral steering** carries it": JAW tracks (cell passes) AND beyond-jaw PTS / FR do not (gate fails or TI
  numerator CI includes 0).
* "**Something beyond the jaw**": beyond-jaw PTS passes the gate >= 3/4 AND tracks. The FR group / NOSE / region
  breakdown then says what, descriptively.
* "**Only FR tracks**" (PTS and JAW do not): consistent with **acausal leakage of the launch** into FR (section 0). It is
  not evidence of a pre-launch face signal.

## 6. Q2 -- WHEN: before jaw opening or only once the jaw moves
* **Pre-jaw window (CUED, JAW-referenced set)**: -0.50..-0.05 s from JAW onset, BASE -0.80..-0.50 s from jaw onset.
  Its own frozen decoders per family; gate + calibration + TI.
* The tongue-referenced decoder is also scored on the same licks (like-for-like).
* **PRIMARY FAMILY B (Holm over 4; TI numerator):** pre-jaw PTS acute (B1) / subacute (B2); pre-jaw JAW acute (B3) /
  subacute (B4).
  * Before jaw onset the jaw has not opened. JAW there = lateral posture / any lateral pre-shift.
* **Per-bin time course**: single-bin decoders per family (PTS, JAW, FR-all), 22 bins -0.80..+0.30, BASE bins as raw
  levels, other bins minus BASE (S26).
  * Tongue-referenced (CUED, ITI) and jaw-referenced (CUED). Each frozen; gate r, calibration, TI per bin.
  * FR bins whose centre lies within ±0.35 s of the reference (the half-height half-width at the lowest lick-factor
    frequency, 5.8 Hz) are flagged **leak-exposed**; within ±0.70 s they are flagged "inside filter support".
* **ITI**: no jaw onsets exist and no ITI jaw detector is built here. The ITI time course is tongue-referenced only.
  The mean cam4 jaw-opening trace (pre licks) is drawn under it, so the reader sees when the jaw moves.
* Reading Q2:
  * **PLAN-LIKE (PROVISIONAL)** = a timing-valid family (PTS or JAW) tracks in the pre-jaw window (a family-B cell
    passes), or its single bins ending <= -0.05 s before jaw onset show TI CI > 0 with >= 3/4 same sign and >= 3/4
    calibrated.
  * **EXECUTION** = tracking only in bins at / after jaw onset (or only in the last tongue-referenced PRE bin), none
    before.
  * **FR-only early tracking** is not a plan signature (leakage).

## 7. Q5 -- does partialling the face out of cortex change the frozen-decoder OFFSET?
Offsets (the published code paths, inputs unchanged except the decoder's cortical features):
* **ITI docked same-place offset**: `iti_lick_order` KEY-row lateral decoder (Allen split r8, lags 0..0.2 s from the
  lick peak, frame-SD z-scored, ridge on August pre, pre LOSO) -> `likeforlike` loclin reference
  (`lick_matched_readout --ref loclin`), docked filter (`WFIELD_ITI_FILTER=dock`, `lick_iti_dock_compare` headline a).
  Session means of resid.
* **CUED matched-error offset**: `lick_balance_matched_error` off_spout. These are cued licks whose landing lies inside
  the animal's pre 2.5-97.5 % range at that spout; pre within-spout line on landing; the per-lick read-out from the
  same frozen decoder.
* Secondary: ITI docked matched-error off_spout; cued same-place offset.

**Face covariates per lick:**
* FR-all's 11 traces, window means over [-0.2, 0) and [0, +0.2] s from the lick peak (the decoder's own window and the
  0.2 s before it: GCaMP6s lags) -> 22 regressors.
* Sensitivities: (i) the 5 groups only (check 3's set; 10 regressors); (ii) + PTS (20 PCs x 2 windows); (iii) + JAW.
* ITI FR here is 5 Hz (each 0.2-s window ~ one session bin), stated.

**Partialling (fixed):**
* **(A) PRIMARY, frozen pre regression, original decoder.**
  * Per animal x set, OLS of each cortical feature (the decoder's z-scored inputs) on the face covariates, fitted on
    PRE licks only (frozen, kind `face_aim`).
  * Applied as x_res = x - (f - f̄_pre)·B, so pre means are unchanged and only face-predicted deviations are removed.
  * Pre sessions use a regression fitted leaving that session out (LOSO), consistent with the decoder's pre scoring.
  * The ORIGINAL decoder weights score x_res, and the like-for-like reference is rebuilt from the residualised pre
    predictions.
  * This isolates "how much of the offset rides on face-predictable feature variance".
* **(B) Decoder refit on residualised pre features** (frozen, kind `face_aim`), same pipeline.
* **(C) PER-SESSION variant (Priya)**: B_s fitted within each session on its own licks (pre and post), applied as
  x - (f - f̄_pre,animal)·B_s; original decoder.
  * This uses each session's own face-cortex coupling. The regression is NOT frozen, by design. The decoder is.
* **Reproduction check**: "before" values recomputed through the same code must reproduce the published docked ITI
  (+0.30 / +0.22 / +0.29) and matched-error values to 0.01 mm. If not, the discrepancy is reported, and before / after
  are always compared within this script.
* **Planted-signal positive control**: add to the post cortical features a synthetic face-driven component x +=
  (f - f̄_pre)·B_rand, scaled so it shifts the offset by the observed acute offset. Partialling (A) must remove >= 90 %
  of it.

Statistics:
* per animal x epoch table first: offset before, after, retained = after / before, n sessions / licks;
* `boot_delta_columns` on [before, after] -> Δ_before, Δ_after (change vs pre) and the paired difference
  Δ_after - Δ_before;
* retained-fraction CI from the per-draw ratio.

**PRIMARY FAMILY D (Holm over 6; p of Δ_after - Δ_before):** ITI docked same-place offset x {acute, subacute, chronic}
(D1-D3); CUED matched-error offset x {acute, subacute, chronic} (D4-D6).

Reading Q5 (fixed):
* "**Offset survives face partialling**" = after-offset CI excludes 0 with the original sign, >= 3/4 animals same sign,
  AND retained-fraction CI lower bound >= 0.5.
* "**Offset is face-carried**" = after CI includes 0 AND retained upper bound < 0.5.
* Anything else = partial, stated with the table.
* Variants (B) and (C) are reported beside (A). Disagreement is reported, not resolved post hoc.

## 8. Q3 -- recovery, and Q4 -- within-spout
**Q3** (secondary; pre-specified tests, uncorrected, 3/4 rule). Per session, the face-decoded deviation at the same
target (d_dec*, raw d_dec) and the launch deviation d_exe (session means; PTS, JAW, FR-all; CUED and ITI).
1. Per-animal Pearson r across POST sessions of d_dec* with d_exe.
2. Within-animal MixedLM `d_dec* ~ w + b + log(day)`, groups = animal: w = within-animal-centred d_exe, b = animal mean
   (as `lick_sspn_checks._track`). The same with the established recovery axis `recmark_sessions.csv` (cued `aim`;
   ITI `iti_ang`).
3. TI by epoch (acute / subacute / chronic): a constant TI = the same time course.
4. Marker test: does d_dec* predict d_exe beyond log(day)? MixedLM `d_exe ~ d_dec*_w + d_dec*_b + log(day)`.

Reading:
* "**Same time course**" = r > 0 in >= 3/4 animals AND within-animal p < 0.05.
* Because d_dec* is decoded from the same licks' face, agreement shows the face tracks the launch, not that the face
  CAUSES the launch. It counts as a "pre-launch behavioural marker" only if the tracking is timing-valid (Q1 / Q2
  criteria met).

**Q4** (pre-stroke only).
* Within-spout decoder per family: y and features centred within session x target cell, same ridge / LOSO pipeline.
  Not frozen (never scores post).
* Statistic: pooled LOSO r; null = 200 permutations of y within cell.
* Also the within-cell partial r of the across-spout decoder (S26's partial_r).
* **Gate: p < 0.05 in >= 3/4 animals**, per family x window: PRE, PRE10, pre-jaw; POST = positive control.
* Strict sensitivity: Bonferroni over the 3 primary families (p < 0.0167 per animal).
* Reading:
  * "**Aim visible in the face before the tongue (pre-stroke)**" = a timing-valid family passes in PRE10 AND in the
    pre-jaw window.
  * PRE only = may be jaw opening.
  * FR only = possibly leak.
  * Cortex comparison: S25 / S25b (gate 0/4; 3/4 nominal with failed positive control).

## 9. Interpretation rules (fixed)
* n = 4: **NOT ESTABLISHED / PROVISIONAL** language only; per-animal tables before every pooled number.
* "**The face carries the deviated aim before the tongue emerges**" (PROVISIONAL at most) requires all of:
  1. a Family-A cell meets the tracking criterion;
  2. the same family passes its gate and POST control;
  3. the effect is present in PRE10;
  4. for CUED, a Family-B cell or the pre-jaw single bins meet the section-6 plan-like criterion.
* "**The face's pre-launch deviation is execution (jaw opening) or leakage**": tracking in PRE but not PRE10 / pre-jaw,
  or only in FR.
* "**Jaw steering vs something else**": section 5 reading.
* Q5: section 7 reading.
* Q3 / Q4: section 8 readings.
* GCaMP is irrelevant to Q1-Q4 (behaviour only). It matters for Q5 (cortical lag).

## 10. Statistics, figures, outputs
* Session loops via `analysis_kit.fan_sessions`, **<= 4 workers**, free RAM checked first; animals fitted in parallel
  (<= 4).
* Seeds fixed (SEED 0; per-cell seeds from animal / arm / family / window).
* Bootstraps via `analysis_kit` (`boot_delta`, `boot_delta_columns`, `boot_p`, `holm`, `boot_ci`).
* Figures (Priya's defaults):
  * sessions = OPEN markers, shape = animal (PS92 o, PS93 s, PS94 ^, PS95 D), colour = epoch
    (`orofacial_cohort.EPOCH_LINE`);
  * pooled = FILLED epoch-colour marker with 95 % CI;
  * no figure numbers in titles / images (panel letters only); no lines between epochs;
  * stars in the top band (`within_lick_correction.star_top` / `reserve_headroom`);
  * nothing overlapping data or neighbouring panels (gap columns, shared tick labels hidden);
  * time courses = faint per-animal lines + thick pooled.
* Primary figure:
  * **A** gate r vs null 95th per family x arm (PRE, POST);
  * **B** TI by epoch, CUED, per family (S26 cortex + S26 FR rows as references);
  * **C** same ITI;
  * **D** pre-jaw window TI;
  * **E** Q5 offset before vs after per epoch (ITI docked same-place, cued matched-error);
  * **F** Q4 within-spout gate.
* Secondary figure:
  * per-bin TI / r time courses (tongue- and jaw-referenced; leak-exposed FR bins shaded);
  * FR group / drop-one / beyond-jaw bars;
  * Q3 recovery scatter (d_dec* vs d_exe per post session, open markers);
  * Q5 variants (B), (C) and the sensitivities.
* Outputs:
  * `features/<label>.npz`, `extract_summary.csv`;
  * `fit/<animal>_{preds_<arm>.csv.gz, gate.csv, within.csv, log.txt}`;
  * `gate.csv`, `calibration.csv`, `frozen_status.csv`;
  * `per_animal.csv`, `primary_holm.csv` (families A, B, D);
  * `secondary_*.csv`, `perbin.csv`, `recovery.csv`, `within_spout.csv`;
  * `offset/{per_animal,pooled,variants,planted}.csv`;
  * `figures/`, `figdata/face_prelaunch_aim.pkl`, logs.

## 11. Decisions needed (defaults used unless Priya changes them before stage 2)
1. Family A includes ITI PTS (A5 / A6). Default yes; ITI has no cue and the 25 Hz face is timing-valid.
2. Mesh regions for PTS (section 2). Default: NOT run, unless regions are drawn on the master face first.
3. Q5 face covariates: FR-all (11 traces) is primary, as asked ("same factor set"). The 5-group set (check 3's) is a
   sensitivity.

## 12. Caveats known now
* n = 4. PS95 ITI is thin (160 pre licks; acute 28 licks in 1 session).
* **FR acausality** (section 0): FR pre-launch values can contain the launch. FR is never the basis of a timing claim.
* **ITI FR = 5 Hz session bins** (0 % of ITI pre-launch windows at 25 Hz).
* FR factors and PTS see the mouth region, so the tongue once protruded. Inside the mouth the tongue is invisible to
  every measure; a "tongue in the mouth" reading is indirect at best.
* cam4 view:
  * ipsilesional = image RIGHT (jaw lateral sign flipped so + = contralesional);
  * the chin is partly hidden at close / centre spouts (close_mid cued jaw coverage median 0.96, 25th pct 0.72);
  * the spout is visible near the mouth in CUED trials. BASE subtraction removes it while it is static; the
    no-baseline variant is confounded by it.
* Jaw onsets are CUED only. They are opening (vertical) onsets: lateral jaw drift before opening is not an "onset".
* The jaw leads the tongue by 52-120 ms, so the last 1-2 tongue-referenced PRE bins contain jaw opening (hence PRE10 and
  the pre-jaw window).
* The face can carry the previous lick (>= 1 s earlier; FR support up to ±0.7 s); history regression reported.
* Q3 agreement between d_dec* and d_exe is partly built in (same licks).
* Q5:
  * GCaMP6s lag; FR is a linear-covariate partial only (non-linear face-cortex coupling is not removed);
  * pre-only regression cannot remove a post-stroke CHANGE in coupling (hence variant C);
  * ITI FR 5 Hz.
* The calibration slope's uncertainty is not propagated into TI (reported per animal), as S26.

## Outcome
(empty -- filled after the run, gates first)

## Deviations
(none yet)
