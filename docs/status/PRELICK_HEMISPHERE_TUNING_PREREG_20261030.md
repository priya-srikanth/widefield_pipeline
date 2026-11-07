# Pre-launch hemispheric tuning to launch angle: design written BEFORE any post-stroke number (2026-10-30)

Question (Priya approved 10-30; RECOVERY_MECHANISMS_LIT_20261030.md sections 1a / 1b "discriminating analyses"; FINDINGS
4 item 21): recovery restores the LAUNCH aim (feed-forward command), not within-lick correction. **Which hemisphere's
PRE-LAUNCH cortical activity carries the launch angle, and does that change after the stroke?**
Code: `scripts/prelick_hemisphere_tuning.py` (`run` / `gate` / `summarise` / `plot`, `--plot-only`). Outputs only under
`M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/prelick_hemisphere_tuning/`
(`writeguard.assert_writable`). Only data looked at before this was written: AUGUST-PRE behavior of the 31 August-pre
imaging sessions (feasibility, below) and one pre session's imaging load (timing). No post-stroke number, no neural number.

## 0. Feasibility (August pre only)
* First cued post-cue licks with a scored launch angle (`licks_scored.csv.gz`, onset seen, theta_L finite): median 415 per
  session (153-658). Within-spout SD of theta_L 13.4 deg (8-18) -- launch noise is 5-8 deg per lick (within_lick prereg),
  so roughly half of the within-spout variance is real launch variation.
* Another lick onset (any context, `licks_all.csv` daq_on_s) in the 0.5 / 1.0 s before the first lick's tongue onset:
  6.2 % / 9.1 % of first licks (anticipatory pre-cue licks) -> excluded (below); the rest are single-lick windows.
* Tongue onset on the DAQ clock = `licks_all.daq_on_s` (cam4 onset through the alignment template; daq_on_s - cue_s
  equals on_ms within +-3 ms). Imaging frame interval 32 ms; one session loads in ~2 s from the local input cache.

## 1. Data (fixed)
* Sessions: `analysis_kit.curated_labels()` ∩ the imaging sessions of `lick_correction.imaging_labels()` ∩ sessions in
  `within_lick_correction/licks_scored.csv.gz`; animals PS92-PS95 ONLY (cohort2 PS118 / 120-122 never). Epochs
  `wfield_local.epochs.epoch_of`; **pre = AUGUST pre sessions only**.
* **PRIMARY lick set = cued FIRST post-cue licks** (`context == post_cue`, `lick_idx == 0`) of ENGAGED trials
  (`combined_encoding.load_behaviour(...).trials.engaged`, joined on trial_id), with onset_ok and a finite launch angle,
  reach and peak speed, and **no other lick onset (licks_all, any context) in the 1.0 s before tongue onset** (clean
  single-lick window covering baseline + pre-launch). SECONDARY sets: (a) cued LATER licks (lick_idx >= 1, engaged, no
  quiet requirement; previous lick's interval and launch angle as covariates -- GCaMP6s overlap is unavoidable there);
  (b) ITI first licks (`context == iti`, no lick onset in the preceding 2.0 s, no cue / strobe in [t - 2.0, t + 0.75] s,
  as iti_licks CUE_EXCL).
* **Launch angle** theta_L from `within_lick_correction` (`licks_scored.csv.gz`, joined on sid + lick_uid; midline angle
  of the first 12 ms of protrusion, deg, **+ = CONTRALESIONAL = mouse right**). **PRIMARY regressor = WITHIN-SPOUT launch
  angle** (theta_L minus the session x spout mean of the licks in the set; ITI: x previous-trial spout), so the spout /
  target is not what is read out. Equivalent: the OLS slope on theta_L with spout fixed effects. SECONDARY = TOTAL launch
  angle (same model WITHOUT spout fixed effects: target + launch).
* **Areas** (ROI v8 masks `iti_licks.roi_masks`, area pixel means of the SVTcorr reconstruction, slow drift removed as
  `lick_correction.extract`, dF/F x 1e3): ALM, tjM1, SSp-m, SSp-n, SSs, MOp_rest (= MOp pixels outside tjM1), MOs_rest
  (= MOs outside ALM), each hemisphere; **left = ipsilesional (image left), right = contralesional**. Controls RSPd,
  VISpm. Pools = mean of member areas: **POOL (primary) = ALM, tjM1, SSp-m, SSp-n, SSs**; MOTOR = ALM, tjM1; SENS =
  SSp-m, SSp-n, SSs; CTRL = RSPd, VISpm.
* **Time**: traces sampled at 10 ms around each lick's tongue onset, averaged into 0.05 s bins from -0.80 to +0.60 s.
  Windows (s from tongue onset): **BASE -0.80..-0.50**, **PRE (primary, pre-launch) -0.30..-0.05** (ends before onset),
  **POST (post-contact) +0.05..+0.30**. Response y = window mean MINUS BASE mean (baseline subtracted). GCaMP6s lags
  spiking ~0.1-0.2 s, so PRE fluorescence reflects activity up to ~0.3-0.5 s before onset; the jaw leads the tongue by
  52-120 ms, so PRE can contain jaw-movement-locked activity -> control window **PREJAW -0.30..-0.05 s from JAW onset**
  (first licks only; `imaging_encoding/qc/jaw_onsets.csv` jaw_onset_ms + cue_s, onset after the cue and before tongue
  onset, not onset_at_cue; baseline -0.80..-0.50 from jaw onset).

## 2. Readouts per session x hemisphere x feature x window x lick set (fixed)
* **Tuning slope** (OLS): y = b0 + **b_ang * angle / 10** + b1 reach + b2 peak speed / 100 + spout FE + b3 RT (first
  licks: tongue onset - cue) [+ previous-lick interval and angle, later licks]. **b_ang = dF/F x 1e3 per 10 deg, + = more
  activity for more CONTRALESIONAL (rightward) launches.** Partial r = corr of the residuals of y and angle on the other
  covariates. Cell needs >= 40 licks.
  Pre-stroke expectation (contraversive tuning): LEFT (ipsilesional) hemisphere b > 0, RIGHT (contralesional) b < 0.
* **Hemisphere difference** D = b_c - b_i (right minus left; pre expected < 0).
* **CV ridge decode** per hemisphere: within-spout launch angle from the 7 orofacial areas of that hemisphere (ALM, tjM1,
  SSp-m, SSp-n, SSs, MOp_rest, MOs_rest; window y, centered within spout, standardized on the training folds), ridge
  lambda = n features, 5 folds stratified by spout, 5 fold assignments averaged; r(prediction, angle). Null = 20
  within-spout permutations; readout **r_ex = r - mean null r**. Control: RSPd + VISpm of that hemisphere. Also c - i.
* Time course (descriptive, figure only): b_ang per 0.05 s bin (bin minus BASE).
* Sensitivity (secondary): S1 PREJAW window; S2 + jaw lateral covariate (-jaw3d_side_ipsi_mm, + = contralesional,
  trial-level, side spouts where available); S3 baseline as a COVARIATE instead of subtracted.

## 3. Primary tests (confirmatory; 3 tests x 3 epochs, Holm over 9 reported)
First cued licks, PRE window, within-spout angle, POOL: **P1 b_c** (contralesional hemisphere), **P2 b_i**
(ipsilesional), **P3 D = b_c - b_i**, each epoch vs August pre. Everything else (POST window, areas, MOTOR vs SENS,
decode, total angle, later / ITI, S1-S3, recovery link) is SECONDARY.

## 4. Hypotheses and predictions (fixed)
* **Contralesional take-over (routes A crossed corticostriatal, B right VLS iSPN via collicular competition, D right
  ALM / tjM1 -> bilateral brainstem premotor):** the RIGHT hemisphere gains IPSIVERSIVE (rightward) tuning BEFORE launch:
  P1 b_c INCREASES (less negative / positive) subacute-chronic; P3 D increases; right decode r_ex up. D specifically:
  the change sits in MOTOR (ALM / tjM1) at least as much as in SENS. A / B / D cannot be separated on the cortical side.
* **Ipsilesional bypass (E: left ALM PT -> left SC / brainstem):** the LEFT hemisphere gains weight: P2 b_i increases
  (more contraversive), D decreases, left decode up.
* **Feedback / reafference:** no PRE change; hemispheric tuning changes appear only in the POST window.
* **Restitution of the old map:** an acute disturbance at most, back to pre chronically (no chronic change in P1-P3).
* **Non-specific:** the same change in the CTRL pool (RSPd / VISpm) -> not an orofacial command signal.
* None: no change anywhere beyond the noise floor (then: not resolvable at area level with GCaMP6s).

## 5. Stability gate and noise floor (August pre; computed before any post statistic is summarized)
* Per animal x hemisphere, primary cell (first licks, PRE, within-spout, POOL): session slopes on ODD and on EVEN trials
  (trial_id parity); animal half-means b_odd, b_even (mean over pre sessions). The hemisphere PASSES if sign(b_odd) =
  sign(b_even) AND |b_odd + b_even| / 2 > |b_odd - b_even| (half-agreement better than the half-difference). An ANIMAL
  passes if BOTH hemispheres pass. **Post-stroke changes in P1-P3 are interpreted only if >= 3 / 4 animals pass**;
  otherwise they are reported as descriptive, "gate failed". The same gate is computed for the POST window (needed for the
  feedback contrast) and for the decode (both halves r_ex > 0, per hemisphere).
* Noise floor per animal x hemisphere: per-session estimation SD = SD over pre sessions of (b_odd - b_even) / 2, vs the
  between-session SD of the full-session b; reported. An epoch change smaller than the per-animal estimation SD is not
  read as a change even if the CI excludes 0 (stated next to each primary number).

## 6. Statistics (fixed)
Per-animal table (animal x epoch session means, n sessions) printed and saved FIRST (CLAUDE.md rule 8). Then
`analysis_kit.boot_delta` (paired nested animals -> sessions, seed 0) per post epoch vs AUGUST pre; levels `boot_ci`.
**A change = 95 % interval excludes 0 AND >= 3 / 4 animals with the same sign.** Holm over P1-P3 x 3 epochs (reported).
Recovery link (secondary): post sessions, `lick_correction.track` form: MixedLM aim ~ within-animal value + between +
log(day) + (1 | animal), aim = far-contra FIRST-lick launch error e_L (within_lick_correction `sessions.csv`, cued,
far_contra, order first, outcome any; deg, - = ipsilesional), for P1-P3 and the decode c - i; plus per-animal r.
Figures: session points = OPEN markers, shape = animal (`orofacial_cohort.ANIMAL_MARKER`), color = epoch
(`EPOCH_LINE`); pooled = filled marker / thick line with 95 % CI; hemisphere comparisons use one color per hemisphere
with epoch on the x axis; stars only in a reserved top band; no many-thin-bar figures; figdata saved for every figure.

## Outcome (2026-10-30, run after this design was fixed; `prelick_hemisphere_tuning.py run`, `gate`, `summarise`)

Outputs: `.../imaging_encoding/prelick_hemisphere_tuning/` (`sessions/<label>.npz`, `extract_summary.csv`,
`tuning_sessions.csv.gz`, `tuning_sessions_with_diff.csv.gz`, `decode_sessions.csv`, `timecourse_*`, `gate_*.csv`,
`gate.json`, `per_animal.csv`, `decode_per_animal.csv`, `boot.csv`, `decode_boot.csv`, `primary_holm.csv`, `track.csv`,
`figures/pht_fig1..6_*.png`, `figdata/prelick_hemisphere_tuning.pkl`, `run_log.txt`, `gate_log.txt`,
`summarise_log.txt`). 107 imaging sessions (31 August pre); first cued licks per session median 332 (147-559), later
3326, ITI 126, jaw-referenced 268.

### Gate (August pre; first licks, within-spout, POOL): FAILED
PRE window: **0 / 4 animals pass** (per-hemisphere pass only PS92 c, PS95 c). POST window: 1 / 4 (PS95). Decode gate
1 / 4 in both windows. Noise floor: per-session estimation SD of b 0.24-0.58 (PRE) vs between-session SD 0.31-0.44, i.e.
the session-to-session spread IS estimation noise; animal pre means -0.23..+0.13 (c) and -0.17..+0.04 (i); median
|partial r| 0.042 (PRE), 0.050 (POST). Pooled pre decode r_ex: PRE c +0.023 [-0.018, 0.059], i +0.010 [-0.026, 0.048];
POST c +0.041, i +0.051 (CIs include 0). **Pre-stroke, the pre-launch area activity carries no measurable within-spout
launch-angle tuning in either hemisphere** (the expected contraversive pattern b_i > 0, b_c < 0 is absent before
launch: D pre +0.04; it is present after contact: POST D -0.55, 4/4 animals negative, but noisy per session). So the
primary question is NOT TESTABLE at area level with GCaMP6s; all post-stroke numbers below are descriptive.

### Primary (descriptive, gate failed): per-animal session means, pre / acute / subacute / chronic (b, dF/F x1e3 per 10 deg)
| animal | b_c (right) PRE | b_i (left) PRE | D PRE | D POST | n sessions |
|---|---|---|---|---|---|
| PS92 | +.13 / -.23 / -.04 / -.18 | +.04 / -.21 / +.01 / -.26 | +.09 / -.03 / -.05 / +.08 | -.12 / -.29 / -.09 / -.55 | 8 / 5 / 2 / 12 |
| PS93 | -.12 / -.05 / +.16 / +.03 | -.17 / +.04 / +.13 / -.05 | +.05 / -.09 / +.04 / +.08 | -.42 / -.51 / -.61 / -.20 | 7 / 4 / 3 / 12 |
| PS94 | +.04 / -.51 / -.00 / -.16 | -.16 / -.12 / -.04 / -.28 | +.20 / -.38 / +.04 / +.12 | -.37 / -.65 / -.31 / -.14 | 8 / 6 / 5 / 8 |
| PS95 | -.23 / -.69 / -.11 / -.09 | -.06 / -.62 / -.06 / -.13 | -.17 / -.07 / -.05 / +.04 | -1.27 / -.38 / -.43 / -.13 | 8 / 1 / 7 / 11 |

boot_delta vs August pre (Holm over 9): P1 b_c acute -0.32 [-0.65, +0.03], subacute +0.05 [-0.21, +0.33], chronic -0.05
[-0.31, +0.19]; P2 b_i -0.14 / +0.10 / -0.09 (all CIs include 0); P3 D -0.18 [-0.50, +0.08] / -0.05 / +0.04 [-0.11,
+0.19]. **No change** (all p_holm >= 0.59). The chronic D estimate is +0.04, well under the per-animal estimation SD.
POST window: D -0.55 pre -> +0.09 / +0.19 / +0.29 change (CIs include 0; 1-3/4) -- a flattening trend after contact,
not a change. MOTOR vs SENS: no change in either window. Decode: no change except a subacute right-hemisphere PRE dip
(-0.05, 4/4; isolated). Jaw-referenced (PREJAW), baseline-as-covariate and jaw-lateral variants: no change.
Recovery link (post sessions, log(day) in the model): b_c +1.8 deg aim per unit [-0.7, 4.4], D +2.4 [-1.2, 6.0],
decode n.s.; per-animal r inconsistent.

### Changes that did meet the criterion (all descriptive, gate failed)
* CONTROL areas (RSPd / VISpm), first licks: chronic slope -0.29 (PRE) / -0.38 (POST) in BOTH hemispheres (4/4), and a
  subacute c - i +0.17 / +0.35 (4/4) -> non-orofacial, bilateral; treat as non-specific (state / drift), not a command.
* LATER cued licks, TOTAL angle (target + launch; PRE window contains the previous lick's post-contact response):
  pre D -0.87 (4/4 negative, i.e. contraversive), flattening chronically (D +0.35 [CI excl 0] subacute / chronic, 4/4;
  b_c +0.55 / +0.48). Present in SENS (+0.45 / +0.44, 4/4) at least as much as MOTOR (+0.24 / +0.22, 4/4), and partly in
  CTRL (c - i +0.10 / +0.16 / +0.15, 4/4). Within-spout on the same licks: no change. -> this is the already-reported
  post-lick contralesional-share rise / flatter contralesional tuning at the TARGET level (FINDINGS 0b), seen through the
  previous lick's response, not pre-launch launch tuning.

### Reading
None of the pre-registered hypotheses is favored: the pre-launch readout has no pre-stroke tuning to measure against
(gate 0/4), and no post-stroke change appears in either hemisphere, before or after contact, in orofacial areas.
Route A / B / D (right-hemisphere ipsiversive pre-launch tuning) is NOT supported but NOT excluded (power: the per-session
estimation SD ~0.3-0.6 dF/F x1e3 per 10 deg against animal-level effects of < 0.2). The only robust signals are
target-level and post-lick (later licks) or non-orofacial (RSPd / VISpm). The re-aimed launch command is therefore not
visible as within-spout area-level tuning in widefield GCaMP6s; testing the routes needs a perturbation (cohort 2) or a
higher-resolution readout.

### Deviations
1. The 1-s no-lick rule removed ~18 % of first licks in PS92 / PS94 (feasibility estimate pooled 9 %); per-session counts
   in `extract_summary.csv`.
2. PREJAW baseline is -0.80..-0.50 s from JAW onset (as written) -- for short reaction times this overlaps the cue.
3. Recovery-link MixedLM fits emitted convergence warnings for some metrics (estimates reported as is; secondary).
4. The decode uses within-spout centering of the features on all licks of the session (label-free; not fold-wise).
Caveats: n = 4; GCaMP6s lag; launch noise 5-8 deg / lick; area pools average out finer topography. PROVISIONAL.
