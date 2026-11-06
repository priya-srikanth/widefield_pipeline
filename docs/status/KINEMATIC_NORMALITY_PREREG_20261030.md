# Kinematic normality (restitution vs compensation) -- PRE-REGISTRATION, 2026-10-30

Analysis 2.2 of `docs/RECOVERY_MECHANISMS_LIT_20261030.md`. Written BEFORE any post-stroke number was computed. Only
August-pre data were opened, and only to measure feature coverage (section 3). Code: `scripts/kinematic_normality.py`.
Outputs: `M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/kinematic_normality/`.

## 1. Question

The framework is SRRR / Levin, Kleim & Wolf 2009. RECOVERY (restitution) means the pre-stroke movement pattern returns.
COMPENSATION means the task is solved again but with a different pattern. Hit rate is back by about day 9-12, but the
movement is not fully back (FINDINGS 0 item 4b; Fig 0.12).
(a) How normal is each post-stroke trial's movement, judged against the same animal's pre-stroke movements at the same
spout? What is its time course: half-recovery day and asymptote, per animal?
(b) Which EXISTING cortical readouts follow the FAST task-success course (hit rate)? Which follow the SLOW or incomplete
kinematic-normality course?
Per animal, and descriptive only (n = 4). Nothing is claimed about predicting across animals.

## 2. Data and unit

- Animals: PS92, PS93, PS94, PS95 only. Cohort 2 (PS118 / 120-122) is never included.
- Sessions: `analysis_kit.curated_labels()`. Epochs come from `wfield_local.epochs.epoch_of`, days from
  `epochs.days_since_stroke`. PRE = August-only pre sessions (MMDD >= 0801; 8 per animal). June sessions are dropped.
- Trial table: `<E>/qc/trial_attempts.csv` (`scripts/trial_attempts.py`). `canonical` = the engagement gate
  (`orofacial_cohort.engagement_keep`). `cls` = hit / incomplete / jaw_only / no_try / unknown.
- Per-lick 3-D table: `M:/MICROSCOPE/Priya/DeepLabCut/Widefield/session_poses/pose3d/kinematics_all/licks.csv.gz` with
  `context == post_cue`. Jaw per trial: `.../kinematics_all/jaw_trials.csv.gz`. Join key = label x trial_id. The join
  was checked on pre data: the 3-D `position` matches `pos` for 15,557 / 15,557 matched trials.
- Spouts: L = ipsilesional and R = contralesional (all lesions are LEFT VLS). They are reported in the canonical order
  close_ipsi, close_mid, close_contra, far_ipsi, far_mid, far_contra.
- UNIT, primary: one ENGAGED cued trial (`canonical`), described by its FIRST post-cue lick (the smallest post-cue
  `lick_idx` with a 3-D estimate) plus trial-level timing.
- TRIAL SET, primary: **HIT trials** (`cls == hit`). The SRRR definition of compensation is task success achieved with a
  different movement. So the normality of SUCCESSFUL trials is the restitution test, and it is not confounded with the
  hit rate itself.
- TRIAL SET, secondary: all ATTEMPTED trials (hit + incomplete). This set mixes failure with abnormality, so it is
  expected to track hit rate more closely.
- UNIT, secondary: the trial's lick SEQUENCE summary on hit trials (section 3b).

## 3. Features (fixed here)

Coverage on August-pre engaged trials:

| trial set | trials | 3-D first-lick features | first-lick latency | jaw lead | jaw lateral (lr_wide) |
|---|---|---|---|---|---|
| hits | 15,187 | 0.998-0.999 | 1.000 | 0.799 overall | 0.904 overall |
| attempted | 15,557 | 0.996-0.997 | 0.999 | 0.796 | 0.903 |

Coverage by spout on pre hits:

| spout | 3-D | jaw lead | jaw lateral |
|---|---|---|---|
| close_contra | 1.000 | 0.991 | 1.000 |
| close_ipsi | 1.000 | 0.990 | 1.000 |
| close_mid | 0.998 | 0.000 (by design: the chin is hidden) | 0.533 |
| far_contra | 0.999 | 0.965 | 0.999 |
| far_ipsi | 1.000 | 0.944 | 1.000 |
| far_mid | 0.999 | 0.923 | 0.899 |

RULE: a feature enters a spout's set only if its pooled pre coverage at that spout is >= 90 %. That gives three
feature sets.

### 3a. First-lick vector (primary)

The first eight features are used at every spout. Feature 9 is used only at the four side spouts.

| # | feature | definition and sign | transform |
|---|---|---|---|
| 1 | launch angle (deg) | asin(side_ml / reach_mm) of the first lick at the 3-D peak; + = contralesional (mouse right) | none |
| 2 | sagittal angle (deg) | `sagittal` = atan2(out, down), the forward tilt | none |
| 3 | protrusion | `reach_mm`, \|tip - E\| at the 3-D peak | none |
| 4 | peak tongue speed | `speed_peak` (mm/s) | log |
| 5 | landing lateral error | `lat_err_mm` at closest approach (+ = outward; centre spout + = mouse left) | none |
| 6 | closest approach to the spout | `miss_mm` | log(x + 0.05) |
| 7 | latency to first lick | `trial_attempts.cam4`, ms from the cue | log |
| 8 | jaw onset lead | first-lick latency minus `jaw_onset_ms` (+ = jaw before tongue). Not at close_mid. | none |
| 9 | jaw lateral steering | `lr_wide_mm` (+ = mouse left = ipsilesional). Side spouts only; not at close_mid or far_mid. | none |

- Feature counts: 7 at close_mid, 8 at far_mid, 9 at the four side spouts.
- A trial is scored only if every feature in its spout's set is present. Coverage of scored trials is reported by
  animal x epoch x spout. If post coverage drops more than 10 points below pre in some cell, that cell is flagged.
- The trial-level FAR_DIR (far_L - far_R) / 2 is a session contrast, not a per-trial feature. Its per-trial ingredient
  is feature 9.

### 3b. Sequence vector (secondary, hits)

All spouts use the same five features:

1. log latency to first contact (`trial_attempts.contact`, ms)
2. post-cue licks before the first contact lick (3-D table `contact`; 0 = the first lick hit)
3. log total post-cue licks in the trial (`n_licks`)
4. mean launch angle of the first <= 3 post-cue licks
5. mean protrusion of the first <= 3 post-cue licks

## 4. Score

These choices are made per animal x spout.

- Reference = that animal's August-pre trials at that spout. The cell must have >= 30 pre trials and >= 3 pre
  sessions, otherwise it is not scored.
- Features are z-scored on the reference mean and SD. Covariance = Ledoit-Wolf shrinkage
  (`sklearn.covariance.LedoitWolf`) on the z-scored reference. d2 = squared Mahalanobis distance.
- PRE trials are scored LEAVE-ONE-SESSION-OUT: the z-scoring and the covariance are refitted without the trial's
  session. This gives pre an honest reference level, so pre is NOT at 95 % by construction.
- POST trials are scored against all pre sessions.
- "Inside the pre 95 % contour" means d2 <= chi2.ppf(0.95, k), where k = the number of features in the cell. This is
  a FIXED threshold. The empirical pre (LOSO) fraction inside is the reference level it is compared against.
- Session values:
  - NORMALITY = the fraction of scored trials inside, pooled over spouts. Each trial is compared with its own spout,
    and each spout is weighted by its trial count.
  - Also: the median d2 and the far-contra-only fraction.
  - A spout contributes to a session only if it has >= 5 scored trials. Far-contra session values need >= 5 scored
    far-contra trials.
- CLASSIFIER (secondary): per animal x spout, L2 logistic regression (C = 1, class-balanced) on the same z-scored
  features, PRE vs POST.
  - Cross-validation is leave-one-SESSION-out over all of the animal's pre + post sessions. Each trial gets an
    out-of-fold P(post).
  - Session value = mean P(pre) = 1 - P(post), pooled over spouts like the main score.

## 5. Per-session series

Every series is per animal, and every value is per session.

| key | series | source | sign |
|---|---|---|---|
| hit | engaged hit rate, all spouts: hits / canonical trials | trial_attempts.csv | |
| hit_fc | engaged hit rate at far contra | trial_attempts.csv | |
| aim_fc | far-contra first-lick 3-D aim, change vs own pre (deg) | `<E>/iti_licks/aim_by_spout_sessions.csv` (`chg`, pos far_contra; `scripts/lick_cued_aim_by_spout.py`) | + = contralesional; - = ipsilesional deficit |
| norm | kinematic normality, all spouts, hits (fraction inside) | this analysis | |
| norm_fc | kinematic normality, far contra, hits | this analysis | |
| d2 | median d2, all spouts, hits | this analysis | |
| norm_att | normality on attempted trials | this analysis | |
| norm_seq | sequence normality on hits | this analysis | |
| norm_clf | classifier P(pre) on hits | this analysis | |
| C1 bal | PRIMARY recovery marker: cued first-lick orofacial balance (ALM + tjM1 + SSp-m + SSp-n + SSs angle-matched gain ratio, 0.1-0.5 s; page 14.17) | `<E>/iti_licks/recprim_sessions.csv`, panel `cued first lick -> aim`, column `marker` (`scripts/lick_recovery_primary.py`) | hemispheric sign: + = contralesional hemisphere larger |
| C2 iti_off | ITI decoder offset, same-place like-for-like, DOCKED-spout ITI (primary ITI set 10-29), all spouts | `<E>/iti_licks/itidock_main/matched_gauss_sessions.csv`, scope `all spouts`, cls `all`, set `iti`, column `resid` (mm) | + = read contralesional of the landing |
| C3 ret_hit | frozen-decoder far-contra retention on HITS, lick window | `<E>/attempt_coding/decoder_retention_sessions.csv`, window `lick`, cls2 `hit` (sessions with >= 5 far-contra hits) | 1 = pre |
| C4 tjm1 | first cued lick response AUC 0-1 s, contralesional tjM1 (`tjM1_right`) | `<E>/iti_licks/area_auc_sessions_split_cued.csv`, column `resp` (`scripts/iti_area_maps.py --split --cued`) | dF/F x1e3 x s |

None of the cortical readouts is refitted here.

## 6. Normalised scale and time constants

FRACTION OF ACUTE DEFICIT RECOVERED, per animal and series:

f = (x - acute) / (pre - acute)

- pre = the mean of the August-pre sessions. acute = the mean of the acute-epoch sessions; if there is none (C3 in
  PS95), the earliest post session is used and flagged.
- So pre = 1 and acute = 0.
- If \|pre - acute\| < 1 SD of that series' pre sessions, the series is flagged NO ACUTE DEFICIT in that animal. It is
  drawn, but it is not fitted and not used in the timing comparison (its fraction would be noise / noise).
- aim_fc already is a change vs pre (pre = 0).

Fit, per animal x series, on post-stroke sessions:

- Model: y(d) = a - (a - b) exp(-d / tau). Same form, bounds and starting values as `scripts/recovery_timing.exp_fit`:
  a, b in [-2, 3], tau in [0.3, 200], >= 4 sessions.
- Reported: half-recovery day t1/2 = tau ln 2, and the asymptote a (the fraction of the acute deficit recovered).
- 95 % CIs: resample post sessions with replacement, 1000 draws, seed 0.
- Per animal, the difference t1/2(series) - t1/2(hit) is reported with its CI from the SAME draws (paired resample).

## 7. Pre-registered comparison (cortex: success vs normality)

For each cortical series C and each animal, on post sessions with all needed values:

- Every series is residualised on log(day) (OLS with an intercept).
- r_hit = Pearson r(C_res, hit_res). r_norm = r(C_res, norm_res).
- Partial versions: r_hit|norm (C with hit, also controlling for norm) and r_norm|hit.
- Pooled = the mean of Fisher z over animals.
- Permutation p for each pooled r: the C residuals are permuted within animal, 5000 permutations, two-sided. Session
  autocorrelation makes this p optimistic; that caveat is stated with it.
- The difference r_hit - r_norm is descriptive. It gets a session-bootstrap CI (resampling sessions within animal,
  2000 draws) and an animal count of sign agreement.
- READ-OUT RULE: C "follows success" if r_hit|norm > r_norm|hit in >= 3/4 animals AND the pooled r_hit is permutation
  p < 0.05. C "follows normality" under the mirror condition. Anything else is "undetermined".
- Also per animal: t1/2(C) - t1/2(hit) and t1/2(C) - t1/2(norm). These are descriptive; the CIs are expected to be
  wide.
- Primary pair = all-spout hit vs all-spout norm. Secondary pair = hit_fc vs norm_fc (C3 is a far-contra readout, so
  the secondary pair is its natural partner).

## 8. Epoch statistics

- The per-animal table (animal x epoch session means) is printed FIRST (CLAUDE.md rule 8).
- Then `analysis_kit.boot_delta` (paired animals -> sessions, animal-weighted) per post epoch vs August pre, for:
  norm, norm_fc, d2, norm_att, norm_seq, norm_clf. Seed 0 per measure; draws in sorted-animal order.
- CHANGE = the 95 % CI excludes 0 AND >= 3/4 animals have the same-sign difference.
- PS95 acute = 1 session, so its inner draw is degenerate; this is stated.

## 9. Figures

- Fig 1: per animal (4 panels), day course on the common normalised scale (fraction of acute deficit; pre = 1,
  acute = 0). Hit rate, normality, far-contra normality and C1-C4, with the fitted exponentials. Legends sit outside
  the data area.
- Fig 2: normality by epoch. One dot per session, animal markers, and the animal-weighted mean. boot_delta stars go
  only in a reserved top band.
- Fig 3: per-animal t1/2 and asymptote per series (dot + CI, one row per series; no thin-bar grid).
- Fig 4: correlation summary, r_hit|norm vs r_norm|hit per cortical series, per animal + pooled.
- Every figure saves figdata. `--plot-only` redraws from figdata alone.

## 10. Known limits (stated in advance)

- n = 4. Session spacing is sparse early (days 1-5), which caps how well t1/2 can be resolved.
- Hit-trial normality excludes failed trials. Acutely at far contra there are few hits (completion .02-.15), so
  acute far-contra normality rests on few trials and on SELECTED trials (the hits that do happen may be the most
  normal ones). The all-trials attempted set is the check.
- The 3-D first-lick features carry tracking noise. The LOSO pre level absorbs the session-to-session noise floor
  (day-to-day camera pose).
- Representational drift and practice continue post-stroke. No time-matched no-stroke reference exists.
- The cortical series come from different trial sets and windows, and C3 starts late because it needs >= 5 hits.

## Outcome (run 2026-10-30, appended after the run; sections 1-10 above unchanged)

Run: `python scripts/kinematic_normality.py` (figures redrawn with `--plot-only`).
Outputs: `<E>/kinematic_normality/`. Tables: `sessions.csv`, `sessions_normalised.csv`, `coverage.csv`,
`per_animal_epoch.csv`, `epoch_tests.csv`, `norm_reference.csv`, `fits.csv`, `fit_diffs.csv`, `corr_animals.csv`,
`corr_pooled.csv`, `series_sources.csv`, `trials.csv.gz`. Figures: `figures/kn_fig1..4_*.png`. Figdata:
`figdata/kinematic_normality.pkl`.
Data: 108 sessions (8 August-pre per animal; post sessions acute 5 / 4 / 6 / 1, subacute 2 / 3 / 5 / 7, chronic
12 / 12 / 8 / 11 for PS92 / 93 / 94 / 95), 50,929 engaged trials, 46,466 hits.

### Coverage

- Scored hits: 95-97 % in every epoch.
- Far contra, acute, falls below the 10-point flag line:
  - PS92: 0.80 of 15 hits (pre 0.94)
  - PS94: 0.73 of 11 hits (pre 0.96)
  These two cells are FLAGGED; PS95 has 1 acute far-contra hit.
- Sequence set: >= 99 %.

### Normality by epoch (per animal first)

Fraction of hit trials inside the pre 95 % contour. Pre = leave-one-session-out.

| animal | pre | acute | subacute | chronic |
|---|---|---|---|---|
| PS92 | .894 | .719 | .670 | .867 |
| PS93 | .895 | .568 | .752 | .847 |
| PS94 | .889 | .254 | .573 | .618 |
| PS95 | .917 | .768 | .793 | .854 |

boot_delta vs August pre (* = change rule met):

| measure | acute | subacute | chronic |
|---|---|---|---|
| norm (hits) | -0.32 [-0.54, -0.16] 4/4 * | -0.20 [-0.28, -0.13] 4/4 * | **-0.10 [-0.22, -0.03] 4/4 *** (PS92 -.03, PS93 -.05, PS94 -.27, PS95 -.06) |
| norm_fc | -0.31 [-0.57, +0.01] (2 animals) | -0.14 [-0.26, -0.04] 4/4 * | -0.10 [-0.17, -0.04] 4/4 * |
| norm_att | -0.38 * | -0.21 * | -0.10 [-0.21, -0.03] 4/4 * |
| d2/k | +1.10 * | +0.66 * | +0.43 [+0.21, +0.78] 4/4 * |
| norm_seq | -0.31 * | -0.13 * | -0.07 [-0.19, +0.03] 2/4 (no change) |
| norm_clf | -0.51 * | -0.54 * | -0.49 * |
| hit rate (engaged, all spouts) | -0.30 * | -0.06 * | **+0.03 [+0.00, +0.06] 4/4** |
| hit_fc | -0.90 * | -0.21 * | +0.02 [-0.01, +0.05] |

So task success is back, or slightly above pre, chronically. The first-lick kinematics on SUCCESSFUL trials are still
measurably abnormal in all four animals. The chronic shortfall is small in PS92 / 93 / 95 (3-6 points) and large in
PS94 (27 points). The lick SEQUENCE summary is back to pre-normality in 2/4.

### Time constants (fraction-of-acute-deficit scale; t1/2 in days [95 % CI]; asymptote a)

| animal | hit rate | aim_fc | norm (hits) |
|---|---|---|---|
| PS92 | 4.9 [2.6, 8.4], a 1.11 | 5.4, a 0.91 | 8.8 [3.3, 51], a 0.99 [0.77, 2.55] |
| PS93 | 4.7 [2.4, 7.8], a 1.33 | 3.2, a 0.78 | 3.6 [2.1, 4.9], a 0.87 [0.79, 0.96] |
| PS94 | 4.1 [2.2, 6.1], a 1.02 | 3.1, a 0.55 | 2.7 [1.6, 4.1], a 0.56 [0.49, 0.64] |
| PS95 | 7.4 [0.5, 139], a 1.08 | 4.3, a 0.94 | 48 [9.5, 139], a 1.55 (UNRESOLVED) |

- PS95: the acute reference is 1 session with a small deficit (.77 vs .92).
- norm minus hit t1/2 (same draws): PS92 +3.9 [-3.3, +46], PS93 -1.2 [-3.3, +0.3], PS94 -1.4 [-3.9, +1.1]. No
  consistent order.

**KEY READING (differs from the prior expectation in section 1).** Kinematic normality is NOT a slower process. Where
it is resolved (PS93, PS94), it recovers on the same timescale as hit rate, or faster. It PLATEAUS SHORT of pre
(asymptote 0.87 and 0.56), while hit rate plateaus at or above pre. Read this as FAST but INCOMPLETE restitution plus
compensation; PS94 is the clear compensator.

Cortical series:

| series | t1/2 (days) | asymptote |
|---|---|---|
| C1 balance | 3.1 / 6.2 / 2.3 / 3.7 | 1.23 / 1.28 / **0.75** / 1.07 |
| C2 ITI offset | 0.4-4 | 0.17 / 0.38 / -0.11 / 0.53 |
| C3 frozen code on hits | 3.1 / 5.8 / 1.3 / 6.1 | **0.69 / 0.53 / 0.31 / 0.67** |
| C4 contra tjM1 | PS94 / PS95 only | 0.59 / 0.73 |

- C1: PS94's 0.75 is the one incomplete case, matching its normality and aim asymptotes (0.56 / 0.55).
- C2: CIs mostly unbounded; the offset persists and is barely recovered.
- C3: incomplete in all four.

### Which cortical readouts follow success vs normality (section 7 rule)

PRIMARY pair (all-spout hit rate vs all-spout normality; pooled partial r, permutation p):

| readout | r(hit \| norm) | r(norm \| hit) | animals hit / norm | verdict |
|---|---|---|---|---|
| C1 balance | +0.03 (p .85) | +0.24 (p .075) | 2 / 2 | undetermined. Raw r_norm +0.29 (p .023) vs r_hit +0.18 (p .15) leans to normality; fails the 3/4 rule. |
| C2 ITI offset | -0.12 (p .36) | -0.32 (p .013) | 2 / 2 | undetermined |
| C3 far-contra frozen code on hits | **+0.46 (p .001)** | +0.10 (p .51) | 3 / 1 (PS95 the exception) | **FOLLOWS SUCCESS** |
| C4 contra tjM1 | +0.16 (p .23) | -0.01 | 2 / 2 | undetermined |

SECONDARY pair (far contra):

| readout | r(hit \| norm) | r(norm \| hit) | animals hit / norm | verdict |
|---|---|---|---|---|
| C3 | +0.50 (p .001) | -0.05 | 3 / 1 | **FOLLOWS SUCCESS** |
| C2 | -0.36 (p .012) | -0.14 | 3 / 1 | follows SUCCESS (larger offset with lower far-contra hit rate) |
| C1 | | | 2 / 2 | undetermined |
| C4 | | | 3 / 1 | undetermined; p .37 |

The bootstrap CIs of (\|partial_hit\| - \|partial_norm\|) include 0 for every readout, e.g. C3 primary +0.40
[-0.23, +0.76].

Summary:
- The far-contra frozen code on hits moves session-to-session with task success, not with normality.
- Its LEVEL plateaus incompletely (0.3-0.7), like normality.
- No readout passes as "follows normality". The cued orofacial balance leans that way (raw r, PS94's asymptote),
  consistent with FINDINGS 0b, but it does not pass the rule.

### Deviations from the pre-registration

1. `d2` is reported as the median of d2 / k (k = features in the spout's set), not raw d2, so spouts with 7 / 8 / 9
   features pool on one scale.
2. The section 7 rule compared \|partial r\|. Cortical series were correlated in raw units; their signs differ by
   readout.
3. C4 is not fitted in PS92 or PS93:
   - PS92 is flagged no-acute-deficit (36.2 vs 35.6).
   - In PS93 the response keeps rising, so the normalised values fall below -2, outside the fit bounds, and the fits
     failed.
   The lick response GROWS after the stroke (FINDINGS 0b), so the fraction-of-deficit scale suits it poorly.
4. The norm_fc acute reference = the earliest post session in PS92 (day 7) and PS95 (day 2): no acute session had >= 5
   scored far-contra hits. Acute norm_fc boot_delta has 2 animals.
5. The classifier P(pre) is not usable as a restitution score:
   - Post sessions are separable at every epoch (chronic 0.26 vs pre LOSO 0.75, 4/4).
   - Pre sessions themselves are only 0.66-0.87, so it reads any session shift (camera pose, drift, practice).
   It is kept in the tables and Fig 2, not interpreted.

### Caveats

- n = 4.
- PS95 has 1 acute session, so its normality time constants are unresolved.
- Acute far-contra hit-trial normality rests on few, SELECTED trials. Attempted trials agree (norm_att chronic -0.10,
  4/4).
- Permutation p values ignore session autocorrelation, so they are optimistic.
- The chronic shortfall in PS92 / 93 / 95 (3-6 points) is near the pre session SD (0.03-0.04). The 4/4 chronic change
  is carried mainly by PS94.
- There is no time-matched no-stroke reference.
