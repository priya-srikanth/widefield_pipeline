# Pre-launch launch-angle readout from ALM + tjM1 PIXELS, pooled pre sessions: design written BEFORE any number (2026-10-30)

Add-on to `docs/status/PRELICK_HEMISPHERE_TUNING_PREREG_20261030.md` (read it first; its Outcome: area-pool gate FAILED,
0/4 animals PRE, per-session estimation SD 0.24-0.58 ~ between-session SD; POST D -0.55, 4/4). Priya 10-30: *why area
pools rather than an SVD / pixel readout inside the full Allen ALM / tjM1 areas? averaging pixels may cancel a spatial
pattern* -- pixel version approved. **Question: does a SPATIAL PATTERN of pre-launch activity inside ALM + tjM1 of one
hemisphere carry the within-spout launch angle (pre-stroke), and does that hemisphere's readout change after the stroke?**
Two changes against the area analysis, and only two: (1) features = pixels of ALM + tjM1 (not the area mean); (2) the
animal's August pre sessions are POOLED into one decoder (leave-one-SESSION-out), not scored session by session.

Code (stage 2): `scripts/prelick_svd_tuning.py` (`--plot-only`). Outputs only under
`M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/prelick_svd_tuning/`
(sibling of `prelick_hemisphere_tuning/`; `writeguard.assert_writable`). Looked at before writing this: the code of
`prelick_hemisphere_tuning.py`, `combined_encoding.load_imaging`, `iti_licks.roi_masks`, `frozen_models`; the Allen area
grid of three sessions (identical, see 1c) and the ROI pixel counts (geometry only). **No imaging value, no decode
number, no post-stroke number of this analysis has been computed.**

## 1. Data (fixed)
**a. Sessions, licks, angle, covariates: IDENTICAL to the area analysis, read from its cache.** The per-session files
`prelick_hemisphere_tuning/sessions/<label>.npz` (107 imaging sessions, 31 August pre; written by
`prelick_hemisphere_tuning.extract`) already hold the PRIMARY lick set -- cued FIRST post-cue licks of engaged trials,
onset_ok, finite theta_L / reach / peak speed / RT, no other lick onset in the 1.0 s before tongue onset -- with each
lick's tongue-onset time (`first_t`), trial_id, spout (`first_pos`), theta_L, reach, speed_peak, rt. Stage 2 reads the lick
set from there (no re-selection, so the trials are the same by construction) and only adds pixel features at those
times. Animals PS92-PS95 only (cohort 2 PS118 / 120-122 never); curated sessions; epochs `wfield_local.epochs.epoch_of`;
**pre = AUGUST pre only** (`prelick_hemisphere_tuning._aug`). Signs as there: theta_L + = contralesional (mouse right).

**b. Target y = within-spout, covariate-residualised launch angle**, per session: OLS of theta_L on spout fixed effects
+ reach + peak speed / 100 + RT (the area slope model's design `designs["within"]`), y = residual (deg). Label-side only,
fitted within the session (the target definition, not a model of the features). *Refinement:* the area analysis's
DECODE used the within-spout angle without covariates; its SLOPES used the covariates -- the residualised target is the
one asked for here, and is stated as a difference.

**c. Pixels.** ROI v8 masks (`iti_licks.roi_masks`, `WFIELD_ROI_VERSION` default 8, built exactly as
`prelick_hemisphere_tuning._masks`): per hemisphere the UNION of ALM and tjM1 (disjoint; left 6712 + 4354 = 11066
pixels, right 6701 + 4344 = 11045 on the 540 x 640 Allen grid, 19.4 um / px). *Code fact:* the masks are not mirrored
images of each other -- `roi_masks` builds each hemisphere from its own Allen labels by the same anterolateral-rank rule;
the counts differ by 0.2 %. The Allen area grid and brain mask are byte-identical across sessions (checked PS92_0806,
PS92_0812, PS93_0807), so ONE pixel set per animal x hemisphere serves every session (needed to pool sessions and to
apply a frozen decoder to post sessions). **2 x 2 spatial binning, fixed in advance** (mean of the ROI pixels inside each
2 x 2 block; 2833 left / 2828 right features, 39 um): it is far below the optical resolution of widefield and makes the
feature cache ~4x smaller. Switch to 4 x 4 only if memory forces it (decided on RAM, never on a result; stated if so).

**d. Signal and windows: IDENTICAL to the area analysis.** Source = `combined_encoding.load_imaging` (adopted hemo
variant SVTcorr temporal components V, U_atlas rows on the brain mask; `WFIELD_INPUT_CACHE`). Slow drift removed with
`ce._slow_basis(ft, ce.DRIFT_MIN_PERIOD_S)` projected out, x 1e3 (dF/F x 1e3), sampled with `iti_licks._sample` on the
10-ms FINE grid and averaged into the 0.05-s bins (`prelick_hemisphere_tuning._binned`, EDGES, `_bins_of`). Window means
**PRE -0.30..-0.05 s** (primary) and **POST +0.05..+0.30 s** (positive control) from tongue onset, each MINUS the
**BASE -0.80..-0.50 s** mean. *Implementation (exactly equivalent, stated):* every step (drift projection, interpolation,
bin and window means, baseline subtraction, 2 x 2 block mean) is linear in the pixel values, so they are applied to V
(T x K) and the per-lick window means are then projected through the binned U rows (pixel = V_win @ U_bin.T); the
full pixel movie is never formed. Per session cached: per lick y-inputs + (n x n_pix) PRE and POST features per
hemisphere (float32), plus the existing area pools (MOTOR = mean of ALM and tjM1 area traces, from the same npz `first_X`).

## 2. Decoder (fixed)
Per animal x hemisphere x window (x K):
1. **Feature centering**: within session x spout, on all licks of the set (label-free; as the area decode, its deviation 4).
2. **PCA** on the centered pixel features of the TRAINING sessions only (covariance PCA, pixels not z-scored), first K
   components; **K = 10 primary**, 5 and 20 sensitivity. PCA uses no label, so it is identical under any permutation of
   y: it is computed once per outer fold and reused for the inner CV and every permutation (equivalent to refitting it).
3. **Ridge** on the K PC scores standardized on the training sessions; alpha from the grid 10^{-1..5} (7 values) chosen
   by inner leave-one-session-out CV within the training sessions (maximize mean held-out per-session r); refit on all
   training sessions with that alpha. Intercept = training mean of y (~0 by construction).
4. **Pre-stroke score = leave-one-SESSION-out**: for each August pre session s, steps 2-3 on the other pre sessions,
   predict s; **r_s = corr(prediction, y) within s**. Animal score **r_LOSO = mean over its pre sessions of r_s**
   (per-session values also kept: they are the pre column of the epoch comparison). Pooled-concatenation r also reported.

**Frozen reference (rule 10, `docs/FROZEN_MODELS.md` section 7).** The decoder trained on ALL August pre sessions of
the animal, AND its LOSO siblings, go through `frozen_models.load_or_fit` in one payload (pixel index, binning, PCA mean +
components, PC scaler, alpha, ridge weights, LOSO models, SHA-1 of (X, y) with a loud warning on mismatch, as
`lick_enl_angle_decode.frozen_fits`). Spec: `make_spec(animal, "prelaunch_svd", align=f"tongue_{PRE|POST}",
source=f"pix_almtjm1_r8_{left|right}_bin2_K{K}", train_labels=<August pre labels>, n_features=n_pix)`. One model per
animal x hemisphere x window x K (K = 5, 10, 20) and the MOTOR-pool decoder (`source=f"motorpool_r8_{hemi}"`), since
each scores post sessions. **LOCAL store only, not published** (on this desktop `WIDEFIELD_FIGURES_WORKING=
C:/Users/SabatiniLab/cue_lick`; `local_dir()` checked and printed before fitting). The status line (`frozen-fit` /
`frozen-hit` / `SPEC-CHANGED`) is logged.

## 3. Gate and positive control (August pre; computed and printed before any post-stroke statistic)
* **Null**: 200 permutations of y within session x spout cells (pre sessions), whole label pipeline refit (ridge, inner
  alpha CV, LOSO; PCA unchanged by construction, step 2). p = (1 + #{null r_LOSO >= observed}) / 201; null 95th pct saved.
* **GATE (PRE window, K = 10)**: an animal passes if at least one hemisphere's r_LOSO beats its null at p < 0.05; **gate
  passes if >= 3 / 4 animals pass.** Then P1-P3 below are confirmatory; otherwise every post-stroke number is descriptive
  and labeled "gate failed". *Note, stated:* "at least one of two hemispheres" is not corrected across hemispheres (per
  animal false-pass ~0.10 under the null); the per-hemisphere p and the Bonferroni x 2 count are reported beside it.
* **POSITIVE CONTROL (POST window, K = 10, same rule)**: expected to pass (the area analysis saw a contraversive POST
  pattern 4/4). **If POST fails, the method itself is underpowered at these trial counts and a PRE failure says nothing
  about the presence of a pre-launch pattern** -- reported in those words. POST passing does not show a motor signal
  (contact location and reafference differ with launch angle); it only shows the pipeline can read angle-related signal.
* K = 5 / 20: r_LOSO with 100-permutation nulls, reported; the gate decision uses K = 10 only.
* Noise floor per animal x hemisphere: SD over pre sessions of r_s; an epoch change smaller than it is not read as a
  change even if the CI excludes 0 (stated next to each primary number).

## 4. Post-stroke readout and primary cells
* Per post session: **frozen r** = corr(frozen full-pre decoder prediction, y) on that session (features centered within
  session x spout as in pre). Pre column = the LOSO r_s (both out-of-session, so comparable).
* **Primary (PRE window, K = 10)**: **P1 r_c** = contralesional (RIGHT hemisphere, image right) frozen r; **P2 r_i** =
  ipsilesional (LEFT, image left); **P3 D = r_c - r_i** (per session). Each post epoch (acute / subacute / chronic, the
  repo's epochs) vs August pre via `analysis_kit.boot_delta` (paired nested animals -> sessions, seed 0); levels
  `boot_ci`. **A change = 95 % CI excludes 0 AND >= 3 / 4 animals with the same sign.** **Holm over the three cells
  (P1-P3) within each epoch** (the brief: "Holm over the three"); Holm over all 9 (3 cells x 3 epochs, the area
  prereg's family) reported beside it. Per-animal table (animal x epoch session means, n sessions) printed and saved
  FIRST (rule 8).
* **Secondary** (descriptive): (i) the same for POST and for K = 5 / 20; (ii) **per-session refit r**: within-session
  5-fold CV (folds stratified by spout, 5 fold assignments averaged; PCA + ridge fitted inside the folds, K = 10, alpha
  by 3-fold inner CV), every session incl. pre -> separates loss of the FROZEN FRAME (frozen r down, refit r unchanged =
  the code moved) from loss of the code (both down) or no change; (iii) **MOTOR area-pool comparison**: the identical
  pooled-pre LOSO + frozen pipeline on 1 feature per hemisphere (mean of the ALM and tjM1 area traces, i.e. the area
  analysis's MOTOR pool, no PCA), its own 200-permutation null -> pixels vs pool at equal pooling tells whether any gain
  over the area analysis comes from POOLING SESSIONS or from PIXELS; (iv) per-hemisphere weight maps (ridge weights
  back-projected through the PCA to pixels, full-pre decoder, PRE and POST).

## 5. Hypotheses and predictions (fixed; route logic copied from the area prereg section 4)
The decode r is unsigned (information about the angle, not its direction); weight maps are descriptive.
* **Contralesional take-over (A crossed corticostriatal, B right VLS iSPN via collicular competition, D right ALM / tjM1
  -> bilateral brainstem premotor):** the RIGHT hemisphere's pre-launch pattern carries MORE angle information after
  the stroke: P1 r_c rises subacute-chronic (frozen r if the old frame is reused; refit r if a new pattern forms --
  then frozen r may fall while refit r rises), P3 D rises. A / B / D are not separable on the cortical side; this
  readout is ALM + tjM1 only, so D (motor cortex) is the route it is closest to.
* **Ipsilesional bypass (E: left ALM PT -> left SC / brainstem):** P2 r_i rises, D falls.
* **Feedback / reafference:** no PRE change; changes only in POST.
* **Restitution of the old map:** an acute dip at most (frozen r), back to pre chronically; refit r tracks frozen r.
* **None:** no change beyond the noise floor (then: not resolvable at pixel level in widefield GCaMP6s either).
* Pre-stroke expectation (not a test): if a pre-launch pattern exists, the area analysis predicts it is contraversive
  and bilateral (both hemispheres r > 0); the weight maps show whether it is spatially opponent within an area (the
  cancellation Priya suspects) -- a map with mixed-sign lobes and a near-zero area sum.

## 6. Statistics, figures, outputs (fixed)
Statistics as sections 3-4; `analysis_kit.fan_sessions` (<= 4 workers, ~1.4 GB each, free RAM checked first) for the
per-session feature extraction; animals x hemispheres x windows fanned for the permutations. All seeds fixed
(`SEED = 0`, per-cell seeds from animal / hemisphere / window).
**Figures** (Priya's defaults): sessions = OPEN markers, shape = animal (PS92 o, PS93 s, PS94 ^, PS95 D), color =
epoch (`orofacial_cohort.EPOCH_LINE`); pooled = FILLED marker in the epoch color with 95 % CI; no figure numbers in
titles / images (panel letters only); no lines connecting epochs; stars in the top band (`star_top` + `reserve_headroom`,
via the `within_lick_correction` wrappers -- `sig_marks.star_top` is not in this worktree); legends / titles clear of
data; hemisphere comparisons orange = contralesional (right), purple = ipsilesional (left). Primary figure: **A** gate
(pre r_LOSO per animal x hemisphere x window vs the null 95th pct), **B** frozen r by epoch per hemisphere, PRE, **C**
same POST, **D** D = r_c - r_i by epoch, **E** pixel vs MOTOR area-pool decoder (pre r_LOSO and epoch levels), **F**
weight maps per animal x hemisphere on the Allen outline (ipsi = image LEFT). figdata (pkl) saved; `--plot-only`.
**Outputs**: `features/<label>.npz` (feature cache), `extract_summary.csv`, `loso_pre.csv`, `perm_null.csv`,
`gate.csv` / `gate.json`, `sessions_frozen.csv`, `sessions_refit.csv`, `per_animal.csv`, `boot.csv`,
`primary_holm.csv`, `weights/`, `figures/`, `figdata/prelick_svd_tuning.pkl`, logs.

## 7. Caveats (known now)
n = 4 animals (PROVISIONAL at best); GCaMP6s lag (~0.1-0.2 s), so PRE reflects activity up to ~0.3-0.5 s before onset;
the jaw leads the tongue by 52-120 ms, so PRE can carry jaw-movement-locked activity (no PREJAW variant here); launch
noise 5-8 deg per lick bounds any r; within-spout SD 13 deg; ~330 licks per session x 7-8 pre sessions per animal;
U_atlas registration jitter between sessions blurs a fine pattern pooled across sessions (pooling trades it for trial
count); per-session pixel offsets removed by centering but per-session gain differences are not; no control area
(RSPd / VISpm) in this add-on, so a non-specific post-stroke change cannot be excluded here.

## Outcome
(empty -- filled after the run, gate result first)

### Outcome (2026-10-30, run after this design was fixed and committed as 27bdef52; `prelick_svd_tuning.py extract / fit / refit / summarise`)
Outputs: `.../imaging_encoding/prelick_svd_tuning/` (`features/<label>.npz`, `extract_summary.csv`, `loso_pre.csv`,
`perm_null.csv`, `gate.csv`, `gate.json`, `frozen_status.csv`, `sessions_frozen.csv`, `sessions_refit.csv`,
`sessions_long.csv`, `per_animal.csv`, `noise_floor.csv`, `boot.csv`, `primary_holm.csv`, `weights/`,
`figures/pst_primary.png`, `pst_secondary.png`, `pst_weights_POST.png`, `figdata/prelick_svd_tuning.pkl`, `run_log.txt`,
`summarise_log.txt`). 107 sessions (31 August pre: PS92 8, PS93 7, PS94 8, PS95 8; 16 acute, 17 subacute, 43 chronic);
first cued licks per session median 331 (125-559). The V-space projection reproduces the area analysis's ALM PRE values
(max |diff| 6e-4 against a per-lick SD of 5-17). Frozen: 64 `prelaunch_svd` models `frozen-new` in the LOCAL store
`C:/Users/SabatiniLab/frozen_models` (existing, with decoder / encoder / enl / lick / reembed / state; not published); the
frozen-LOSO r equals the permutation-path r to 5e-9. Run time ~5 min (extract 0.9, fit 1.6, refit 1.1, summarize ~1).

#### Gate (August pre, pooled LOSO, K = 10): PASSED NOMINALLY (3 / 4); positive control (POST) FAILED (2 / 4)
| animal | PRE c (right) r_LOSO / p | PRE i (left) r_LOSO / p | POST c r / p | POST i r / p | pre sessions |
|---|---|---|---|---|---|
| PS92 | -0.033 / 0.84 | -0.076 / 0.96 | -0.010 / 0.61 | +0.016 / 0.30 | 8 |
| PS93 | +0.016 / 0.31 | **+0.054 / 0.015** | **+0.069 / 0.005** | **+0.093 / 0.005** | 7 |
| PS94 | +0.025 / 0.18 | **+0.050 / 0.020** | -0.014 / 0.71 | +0.038 / 0.075 | 8 |
| PS95 | +0.037 / 0.065 | **+0.045 / 0.035** | **+0.118 / 0.005** | **+0.125 / 0.005** | 8 |

Null 95th percentiles 0.035-0.054. **PRE: 3 / 4 animals pass, every time through the IPSILESIONAL (left) hemisphere**,
with r = 0.045-0.054 (about 0.25 % of the within-spout angle variance). It is fragile. With Bonferroni x 2 over
hemispheres 2 / 4 pass (PS95 p = 0.035 > 0.025). The sensitivity K values give 2 / 4 (K = 5: PS93, PS94) and 1 / 4
(K = 20: PS94). **POST positive control at K = 10: 2 / 4 (PS93, PS95, both hemispheres p = 0.005) -> FAILED.** It
reaches 3 / 4 at K = 5 and K = 20 (PS94 left added). As fixed above, the gate decision uses K = 10. **So the method is
underpowered at these trial counts: it does not reliably recover even the post-contact signal.** The nominal PRE pass
should therefore not be read as a demonstrated pre-launch pattern. Pooled pre levels: PRE c +0.011 [-0.024, 0.037],
i +0.017 [-0.046, 0.063]; POST c +0.040 [-0.014, 0.100], i +0.067 [0.021, 0.115]. Per-session noise floor (SD of the
held-out pre r): 0.025-0.074. Inner-CV alpha sits at a grid EDGE (0.1 or 1e5) in nearly every cell. In other words
the choice is between ~OLS and ~maximal shrinkage, as expected when the signal is near the noise.

#### Primary (PRE window, K = 10, frozen r; confirmatory by the letter of the gate, but see above): per-animal session means pre / acute / subacute / chronic
| animal | P1 r_c (right) | P2 r_i (left) | P3 D = r_c - r_i | n sessions |
|---|---|---|---|---|
| PS92 | -.033 / +.012 / +.014 / -.006 | -.076 / +.019 / -.021 / -.036 | +.043 / -.008 / +.035 / +.030 | 8 / 5 / 2 / 12 |
| PS93 | +.016 / +.063 / +.025 / +.034 | +.054 / +.100 / +.043 / +.016 | -.038 / -.037 / -.019 / +.017 | 7 / 4 / 3 / 12 |
| PS94 | +.025 / +.003 / +.033 / -.011 | +.050 / +.037 / +.029 / -.003 | -.025 / -.034 / +.005 / -.008 | 8 / 6 / 5 / 8 |
| PS95 | +.037 / -.083 / +.034 / +.005 | +.045 / -.105 / +.034 / -.001 | -.008 / +.022 / -.000 / +.007 | 8 / 1 / 7 / 11 |

boot_delta vs August pre: P1 acute -0.012 [-0.088, +0.057], subacute +0.015 [-0.015, +0.052], chronic -0.006 [-0.043,
+0.031]. P2 -0.005 [-0.104, +0.081] / +0.003 [-0.034, +0.044] / -0.024 [-0.064, +0.023]. P3 -0.007 [-0.049, +0.029] /
+0.012 [-0.015, +0.040] / +0.019 [-0.014, +0.049] (3 / 4 positive). **No change in any primary cell**: Holm within
epoch p >= 0.78, Holm over 9 p = 1.0, and none meets the CI + 3 / 4 criterion. NOT ESTABLISHED. Every point estimate
is smaller than the per-session noise floor. PS95 acute has a single session.

#### Pixels vs the MOTOR area pool (same pooled-pre LOSO pipeline, 1 feature)
Pre r_LOSO, animal means: PRE pixels c +0.011 / i +0.018 vs pool +0.012 / -0.013; POST pixels +0.041 / +0.069 vs pool
+0.004 / +0.007. Pool permutation passes: PRE 1 / 8 hemispheres (PS95 c, p = 0.050), POST 3 / 8 (PS93 c, PS95 c,
PS95 i). **Pooling sessions alone does not create a pre-launch readout from the area mean.** The pixels add a small one
in the ipsilesional hemisphere of three animals (PRE) and a clearer one after contact (POST, two animals robustly).
Spatial pattern therefore matters somewhat: averaging over ALM + tjM1 does cancel part of what pixels carry, most
visibly POST. The weight maps (F, normalized per hemisphere) show mixed-sign lobes within ALM / tjM1 rather than one
area-wide sign, which is consistent with cancellation. They are noisy and differ between animals (descriptive).

#### Secondary (descriptive, no correction)
* Refit r (within-session CV) is NEGATIVELY BIASED as a level. The MOTOR-pool refit sits at -0.07 to -0.11 in every
  epoch, and pixel refit at -0.04 to +0.02. The cause is concatenating fold predictions with a fold-wise intercept when
  there is almost no signal (the same `_cv_r` form as the area decode). Read refit only as a change: pixel PRE subacute
  c -0.038 [-0.077, -0.000] and i -0.035 [-0.058, -0.009], 4 / 4 each. Both are below the noise floor, and chronic is
  not changed (c -0.017, i -0.027 [-0.054, 0.000]). Not read as a change.
* MOTOR pool, frozen PRE: c chronic -0.027 [-0.054, -0.000] (4 / 4); D acute -0.045 [-0.085, -0.007] (3 / 4) and
  chronic -0.040 [-0.077, -0.007] (4 / 4). These start from a pre level indistinguishable from 0 (c +0.012, D +0.026,
  CIs include 0) and are below the per-session noise floor. They are counted against many uncorrected cells.
  PROVISIONAL at most; not a loss of a code that was never shown to exist pre-stroke.
* Pixels K = 5, POST, ipsilesional chronic -0.062 [-0.116, -0.012], 4 / 4 (the POST readout is the one with a pre
  level: +0.075). A post-contact decline in the left hemisphere is consistent with the area analysis's flattening POST
  D trend. One cell of many, NOT ESTABLISHED.

#### Reading
Replacing area means with ALM + tjM1 pixels and pooling the pre sessions gives, at best, a marginal pre-launch
readout of the within-spout launch angle. It reaches r ~0.05 in the ipsilesional (left) hemisphere of 3 / 4 animals
and does not survive a hemisphere correction or the K sensitivity. The pre-registered positive control (POST) fails
at K = 10. **So the honest summary is "not resolvable", now at pixel level as well as area level**, not "a pre-launch
pattern exists". No primary cell changes after the stroke. Routes A / B / D (right hemisphere gains) and E (left
gains) are both NOT supported, and power is low (noise floor 0.03-0.07 against effects < 0.03). Pixels beat the
area pool mostly AFTER contact, so spatial cancellation is real but does not uncover a pre-launch command. n = 4,
GCaMP6s; PROVISIONAL.

#### Deviations (2026-10-30, dated; the design above is unchanged)
1. Weight maps are drawn normalized per hemisphere (w / max |w|). The inner-CV alpha differs between hemispheres (0.1 vs
   1e5), so absolute weights are not comparable. This changes the figure only.
2. Refit r: run as pre-registered; its negative level bias was found after the run (see Secondary). No null was added.
3. The smoke test (PS92, 4 sessions) wrote its frozen models to a throwaway scratchpad store
   (`WIDEFIELD_FROZEN_MODEL_DIR`), never to the reference store; its features and outputs were in the scratchpad as well.
4. The secondary figure shows refit r for pixels only (the pool refit is biased; its numbers are in `boot.csv`).
