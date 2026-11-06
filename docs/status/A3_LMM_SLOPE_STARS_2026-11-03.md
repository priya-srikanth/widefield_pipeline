# A3 star rule: random-slope LMM on the 21 behaviour stars (2026-11-03)

Priya's rule (via sabatinilab-a8, 11-03): each post epoch is fitted against pre in its own model. The rows are
sessions: each animal's pre sessions plus its sessions in that epoch. The model is `y ~ post + (1 + post | animal)`,
fitted by REML. The fixed effect is tested with small-sample df (Satterthwaite or Kenward-Roger), and stars use Holm
within a family. A singular fit keeps its estimate and is flagged.

The animal-weighted `boot_delta` stays as a sensitivity column.

Files:
- `A3_LMM_SLOPE_STARS_BEHAVIOUR_2026-11-03.csv`: all 21 cells, with every column below.
- `A3_LMM_PER_ANIMAL_BEHAVIOUR_2026-11-03.csv`: the per-animal table (n pre, n post, means, each animal's own change).
- Script: `scripts/a3_lmm_slope_stars.py`.

## How it was fitted

R is not installed on this box, so lmerTest and pbkrtest are unavailable. The REML likelihood is written out directly:
- G is Cholesky-parameterised.
- The fit is L-BFGS-B from 5 starts, with the Cholesky diagonal bounded at ≥ 0.
- The point estimates and SEs match statsmodels MixedLM wherever it converged without a singular fit (acute fM:
  −0.4948 / 0.1520 here, −0.4948 / 0.1521 in statsmodels).

Satterthwaite df follows lmerTest's varpar route, `2·Var(b)² / (∇'A∇)`, with A = inverse Hessian of −REML. Both A
and ∇ are computed by central differences.

"Singular" uses lme4's `isSingular` rule: a diagonal element of the relative Cholesky factor below 1e-4.

## Result

**The rule as written does not work at n = 4.**

- **17 of 21 fits are singular.** The slope variance goes to 0, or the intercept–slope correlation goes to ±1.
- **Satterthwaite fails at those boundaries.** The Hessian is not positive definite in 5 cells, so df is undefined.
  In others the df is absurd: acute far-ipsi has t = −6.6 with df = 0.013, giving p = 0.94.

Kenward-Roger is not available here. For this design (one within-animal 2-level factor with a random slope), its df
reduces to the containment df, n_animals − 1 = 3.

The same random-intercept-only model is shown alongside, because it is close to flat session weighting (p ≈ 0 almost
everywhere).

Star key:
- `*` uncorrected p < 0.05.
- `**` Holm < 0.05 within the family (18 cells for 1b, 3 for 1f).
- `NA` Satterthwaite df undefined.

Columns: `old` = session-weighted bootstrap (before A3); `boot` = animal-weighted bootstrap; `Satt` = random-slope LMM
with Satterthwaite df; `df3` = random-slope LMM with df = 3.

| family | epoch | cell | old | boot | random-int β, p | slope β, t | Satt df, p, star | df=3 p, Holm, star | singular |
|---|---|---|---|---|---|---|---|---|---|
| 1b | acute | nI | | | −0.068, 4e-4 | −0.061, −1.5 | 1.95, 0.27 | 0.22, 1, | yes |
| 1b | acute | nM | * | * | −0.070, <1e-4 | −0.069, −4.0 | NA | 0.027, 0.44, * | no |
| 1b | acute | nC | ** | ** | −0.083, <1e-4 | −0.080, −3.8 | 2.0, 0.064 | 0.033, 0.49, * | yes |
| 1b | acute | fI | ** | ** | −0.220, <1e-4 | −0.221, −6.6 | 0.013, 0.94 | 0.007, 0.12, * | yes |
| 1b | acute | fM | ** | ** | −0.453, <1e-4 | −0.495, −3.3 | 2.9, 0.049, * | 0.047, 0.61, * | no |
| 1b | acute | fC | ** | ** | −0.879, <1e-4 | −0.878, −37 | 1.2, 0.008, * | 4e-5, 8e-4, ** | yes |
| 1b | subacute | nI | | | −0.010, 0.48 | −0.011, −0.8 | NA | 0.50, 1, | yes |
| 1b | subacute | nM | | | −0.007, 0.54 | −0.008, −0.6 | 1.6, 0.60 | 0.57, 1, | yes |
| 1b | subacute | nC | | | −0.023, 0.12 | −0.013, −0.5 | 3.1, 0.66 | 0.66, 1, | yes |
| 1b | subacute | fI | | | −0.006, 0.81 | −0.005, −0.2 | 2.6, 0.85 | 0.85, 1, | yes |
| 1b | subacute | fM | | | −0.048, 0.016 | −0.048, −1.4 | 2.6, 0.28 | 0.27, 1, | no |
| 1b | subacute | fC | ** | ** | −0.163, <1e-4 | −0.186, −3.6 | 1.9, 0.077 | 0.037, 0.52, * | yes |
| 1b | chronic | nI | * | * | +0.017, 0.028 | +0.018, 2.2 | NA | 0.11, 1, | yes |
| 1b | chronic | nM | * | * | +0.021, 0.001 | +0.021, 2.9 | 1.7, 0.12 | 0.061, 0.67, | yes |
| 1b | chronic | nC | ** | * | +0.021, 0.004 | +0.021, 2.6 | 2.5, 0.097 | 0.081, 0.81, | yes |
| 1b | chronic | fI | ** | ** | +0.085, <1e-4 | +0.085, 1.8 | 2.0, 0.21 | 0.16, 1, | yes |
| 1b | chronic | fM | ** | * | +0.049, <1e-4 | +0.050, 2.0 | 2.8, 0.14 | 0.13, 1, | yes |
| 1b | chronic | fC | * | * | +0.040, 0.002 | +0.042, 3.1 | NA | 0.055, 0.65, | yes |
| 1f | acute | engagement | ** | | −0.180, <1e-4 | −0.177, −4.4 | NA | 0.022, 0.065, * | yes |
| 1f | subacute | engagement | ** | ** | −0.136, <1e-4 | −0.114, −2.4 | 2.2, 0.13 | 0.097, 0.19, | yes |
| 1f | chronic | engagement | | | −0.011, 0.47 | −0.011, −0.7 | 2.5, 0.56 | 0.55, 0.55, | yes |

Under the slope model with df = 3:
- **Only acute far-contra hit rate keeps `**`.**
- Six cells keep `*`: acute nM, nC, fI, fM, subacute fC, and acute engagement.
- **Every chronic recovery star goes**, including chronic far-ipsi. There PS93 recovers by +0.22, against +0.03 to
  +0.06 in the other three animals.

The model is now a test across four animal changes. CLAUDE.md rule 8 warned about this: "anything not visible in at
least three animals individually will not survive". At n = 4 with a 2-sided t on 3 df, even a perfectly consistent
effect needs |t| > 3.18 for `*`. Holm across 18 cells then needs p < 0.0028, i.e. |t| > 9.1.

## Decision needed (Priya)

The pre-specified df method (Satterthwaite) is undefined, or far outside any sensible range, at the singular fits,
which are 17 of 21. It needs a pre-specified fallback before any star is final:

- **(a) df = n_animals − 1 = 3 for every cell** (containment, which equals KR for this design). It is defined whether
  or not the fit is singular. It is the column above; e3 recommends it.
- **(b)** install R + lmerTest/pbkrtest on this shared box and fit KR exactly. Expect the same df ≈ 3. This needs
  Priya's OK to install software.
- Also: should `**` stay as Holm across all 18 position × epoch cells in 1b, or be Holm within each epoch (6 cells)?
  The old bootstrap used Bonferroni across 18.

Imaging families: A3's draw code also feeds `epoch_figures.contrast_draws` (imaging scalar families) and
`matrix_bootstrap.delta_draws` (maps). A per-pixel LMM is not a practical star rule for maps. Which imaging families
the LMM rule should cover is also Priya's call. Not yet run.

## Pooled-epoch alternative (Priya asked, 11-03)

This is one model per cell, with all four epochs and session rows:

`y ~ C(epoch) + (1 | animal) + (1 | animal:epoch)`, fitted by REML in statsmodels.

The animal × epoch random intercept is ONE variance for how much each animal's epoch effect differs from the cohort's.
It asks the same question as the per-contrast slope (do the animals agree?). The difference is that the variance is
estimated from all three post epochs together rather than from 4 animals per contrast.

The tests use containment df = (4 − 1) × (4 − 1) = 9.

Results (`A3_LMM_POOLED_EPOCH_STARS_BEHAVIOUR_2026-11-03.csv`, script `scripts/a3_lmm_pooled_epoch_stars.py`):
- **18 of 21 fits are estimable without a singular fit.** The 3 singular ones are the engagement cells, where the
  animal × epoch variance is 0, so the model reduces to the random-intercept-only one.
- **Stars (Holm within family):**
  - all 6 acute positions `**`;
  - subacute far contra `**`;
  - acute and subacute engagement `**`;
  - no chronic cell has a star. The closest is chronic far ipsi, β = +0.085, p = 0.084.
- **Caveat:** the model assumes the between-animal spread of the effect is the same in every epoch. The acute spread
  is large (far mid: animals change by −0.17 to −0.84), so the chronic tests borrow that spread and are conservative.
  Some statsmodels fits report not converged, but their estimates match the random-intercept-only fit to within 0.01.

Recommendation (e3): use the pooled animal × epoch model for the stars. Keep the random-intercept-only model and the
animal-weighted bootstrap as sensitivity columns.

## Spread differs by epoch (Priya, 11-03)

Priya: the between-animal spread is NOT the same across epochs. So the pooled model's shared animal × epoch variance
is wrong. Each epoch's spread can only be estimated from its own 4 animals, which puts the df at 3 per epoch.

Two versions were fitted (`A3_LMM_HETEROGENEOUS_STARS_BEHAVIOUR_2026-11-03.csv`, script
`scripts/a3_lmm_heterogeneous_stars.py`):
- **(h) heterogeneous LMM:** `(1|animal)` plus a separate animal × epoch variance for each post epoch, t on df = 3.
  None of its fits is singular.
- **(t) per-animal t-test:** a one-sample t on the 4 per-animal changes (own post mean − own pre mean), df = 3.

Both give acute fC `**`. Most other acute cells and subacute fC get `*`. No chronic cell gets a star under (h). Under
(t), chronic nI / nM / fC get `*`, because all 4 animals change the same way by a similar amount.

With Holm within each epoch (6 cells) rather than across 18, acute fI and the engagement cells (h) reach `**`.

## Is the animal-weighted bootstrap calibrated at n = 4? (simulation, 11-03)

Priya noted that the animal bootstrap "seems better powered". We checked whether that is real power or a false-positive
rate above 5 % (`scripts/a3_bootstrap_n4_calibration.py`, 400 simulations per row).

Set-up:
- 4 animals with the acute session counts (pre 16 / 12 / 16 / 16, post 5 / 4 / 6 / 1).
- Each animal's true change ~ N(effect, tau); session noise sd sigma.
- A star means the 95 % interval excludes 0: the bootstrap is `analysis_kit.paired_animal_draws`, the A3 code, with
  1000 draws; the t-test is a one-sample t on the 4 per-animal changes, df 3.

| tau (animals differ) | sigma (session noise) | true effect | bootstrap star | t-test star |
|---|---|---|---|---|
| 0 | 0.05 | 0 | 0.06 | 0.06 |
| 0 | 0.02 | 0 | 0.06 | 0.06 |
| 0.05 | 0.05 | 0 | **0.16** | 0.07 |
| 0.05 | 0.02 | 0 | **0.19** | 0.07 |
| 0.02 | 0.05 | 0 | **0.10** | 0.04 |
| 0.05 | 0.05 | 0.05 | 0.47 | 0.21 |
| 0.05 | 0.05 | 0.10 | 0.91 | 0.58 |
| 0.05 | 0.02 | 0.05 | 0.62 | 0.27 |
| 0.05 | 0.02 | 0.10 | 0.96 | 0.72 |
| 0.02 | 0.05 | 0.05 | 0.71 | 0.41 |
| 0.02 | 0.05 | 0.10 | 0.99 | 0.92 |

When animals truly share the same change (tau = 0), the bootstrap is calibrated. This agrees with the 10-21 pre-vs-pre
calibration (ENCODING_FINDINGS), which by construction has tau = 0.

When animals truly differ (tau > 0, which is the post-stroke case), its false-positive rate is 10–19 % instead of 5 %.
That is about 2–4x the nominal rate. Its extra stars are bought with those false positives.

The cause is that resampling 4 animals treats the observed 4 as the whole population. The 10-21 map calibration is
not affected for null maps; post-stroke map contrasts inherit the same n = 4 limit.

## Priya's choice so far (11-03) and the imaging comparison

Decided so far:
- **The bootstrap stays, labelled for what it is:** a description of these 4 animals.
- **LMM versions of the key figures:** a mixed model with a separate between-animal spread per epoch; the mean is a
  filled round dot.

Previews are in `M:/MICROSCOPE/Priya/Widefield/labcams/grant_figures/lmm_preview_20261103/`, drawn by
`scripts/a3_lmm_preview_fig.py`.

Imaging comparison (`A3_LMM_IMAGING_STARS_2026-11-03.csv`, script `scripts/a3_lmm_imaging_stars.py`). The model was
run on every epoch bar family's stored per-session values (`grant_figures/epoch/data/*_sessions.csv`, the CURRENT,
pre-A3, pre-rebuild renders). Holm is across each figure's cells. Results against the stored mark:
- cells with any star: 854 → 316;
- cells with `**`: 581 → 27;
- 542 stars lost, 4 gained.
