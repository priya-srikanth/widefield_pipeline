# FR pixel-map null calibration: investigation (2026-11-05)

This follows up Section C's "looked odd" item (`SECTION_C_RENDER_20261105.md`). There, the `fr_pixel_significance` null
(50 random pre half-splits) gave a share of splits with >= 1 false bin of 0.12 for full_fr cue 16 px, 0.10 for
unique_position lick 32 px, and 0.08 for several others. **No figure or repo code was changed.** Code read at
`wf_render_daqlk @ a76352f3`. The render ran from `wf_rebuild`, and its `fr_pixel_significance.py` and `beta_maps.py` are
identical to that commit. Experiments ran on 2 workers. Their outputs and scripts are in
`C:/Users/SABATI~1/AppData/Local/Temp/claude/C--Users-SabatiniLab/f082ee0c-b33c-49a8-a537-dcee3bb7c84c/scratchpad/fr_calib/`:
`verify_full_fr_cue.csv`, `fresh_250.csv`, `chrono.csv`, `real_survival.csv`, `calibmod.py`, `real.py`.

## Verdict

- **The calibration is right on average. The 0.12 is mostly selection from noise across 28 cells.**
  - The render's null pooled over all maps is 4.6 % at 16 px and 4.1 % at 32 px.
  - In 250 fresh splits per map (3,500 map-splits per bin size), the rate is **4.9 % [4.2, 5.7]** at 16 px and
    **4.6 % [3.9, 5.3]** at 32 px.
  - The calibration loop is correct: re-running it reproduces the render's 6/50 and 4/50 exactly, bin for bin.
- **One map may be slightly liberal: full_fr cue.** Its 16 px rate is 25/300 = 8.3 % [5.5, 12.1] (render plus fresh
  splits). Across maps, rates range from 1.6 % to 7.6 %, which is not more spread than binomial noise (homogeneity
  chi2 p = 0.10 at 16 px and 0.09 at 32 px). unique_position lick 32 px fell back to 13/300 = 4.3 %.
- **The real issue the null exposed is not calibration. It is drift inside pre.**
  - The pre sessions are not exchangeable: an early (0806-0810) vs late (0810/0811-0814) pre split comes out
    significant in 10 of 14 maps, with 4/4 animals going the same way.
  - Random splits that happen to separate early from late sessions make most of the false bins, and the false bins
    take the sign of the drift.
  - The real epoch-minus-pre comparison is fully separated in time, so this drift can feed into the chronic contours.
    It matters most for **unique_position cue chronic**.

## 1. What the test is (from the code)

- **Script.** `scripts/fr_pixel_significance.py`. `test()` calls
  `beta_maps.hierarchical_bootstrap_significance(pre_by, post_by, factor=f, n_boot=2000, mask=block_mask())`.
- **Maps.** Per-session held-out R2 per 2x2 block, on the 135 x 160 block grid. Raw R2 is used, with no
  pre-normalisation.
- **Bins.**
  - Block means at factor 8 (16 full-res px) give 660 tested bins.
  - Block means at factor 16 (32 px) give 168 tested bins.
  - A bin is kept if at least half of it is inside the eroded stat mask.
- **Resampling.** 2,000 nested draws: animals with replacement (n = 4), then sessions with replacement inside each
  animal, separately for each side. Each draw's statistic is the mean over animals of (post mean - pre mean).
- **Interval.** The observed value +/- z x bootstrap SE. It is two-sided: a bin is significant if |obs| / SE > z.
- **Correction over bins.**
  - The correction is **not Bonferroni**, although the docstrings of `fr_pixel_significance.py`,
    `fr_pixel_maps_sig.py`, `variance_maps.py` and `significance_contour` say it is, and so do the figure suptitles.
  - `beta_maps.CORRECTION = "maxstat"`: z is the 95th percentile, over draws, of max over all tested bins of
    |boot - obs| / SE. That is a family-wise (Westfall-Young) threshold over all bins in the panel.
  - The measured z is 2.6-3.3. Bonferroni over 660 bins would need z = 3.6.
  - All bins are covered and the test is two-sided, so there is no coverage or sidedness mismatch.
- **Null.**
  - For split i, `rng = default_rng(1000 + i)`. Each animal's pre sessions are permuted and cut at `len // 2`.
  - Pre counts are PS92 8, PS93 7, PS94 8, PS95 8, so the halves are 4 v 4, except PS93 at 3 v 4.
  - The split is within animal, so every animal is in both halves and contributes its own difference. The split
    respects animals.
  - The **same 50 splits are reused for all 14 maps** (the seed does not depend on the quantity). The 28 cell rates are
    therefore correlated, and one bad split hits many cells: render split 21 hit 12 of the 28 cells.
- **variance_maps.**
  - It uses the identical `test()`, with seeds 2000 + i, log2 variance maps, and only 2 maps.
  - Its 0-0.02 is also within noise: P(<= 1/50 | p = 0.05) = 0.28.
- **The repo's shared map test.** `beta_maps.significance_contour(method="nested")` calls the same
  `hierarchical_bootstrap_significance` with the same maxstat correction. It adds two things:
  - each animal is divided by its pre RMS (`NORMALISE_TESTS_BY_PRE`);
  - rim-enriched results are suppressed.

  Switching to it would therefore change animal weighting and rim handling, not calibration.

## 2. Is 0.12 above 0.05?

| cell | k / n | 95 % CI (Clopper-Pearson) | P(X >= k \| p = .05) |
|---|---|---|---|
| render, full_fr cue 16 px | 6 / 50 | 0.045 - 0.243 | 0.038 |
| render, unique_position lick 32 px | 5 / 50 | 0.033 - 0.218 | 0.104 |
| render, pooled 14 maps, 16 px | 64 / 1400 | 0.035 - 0.058 | - |
| fresh 250, pooled 14 maps, 16 px | 172 / 3500 | 0.042 - 0.057 | - |
| fresh 250, pooled 14 maps, 32 px | 160 / 3500 | 0.039 - 0.053 | - |
| fresh 250, full_fr cue 16 px | 19 / 250 | 0.046 - 0.116 | 0.047 |
| render + fresh, full_fr cue 16 px | 25 / 300 | 0.055 - 0.121 | - |
| render + fresh, full_fr cue 32 px | 21 / 300 | 0.044 - 0.105 | - |
| render + fresh, unique_position lick 32 px | 13 / 300 | 0.023 - 0.073 | - |

The two flagged cells are the largest of 28 cells. If the 28 cells were independent, the chance of seeing at least 6/50
somewhere would be 0.66. The pattern across maps is consistent with noise around 5 %. Only full_fr cue stays slightly
high after 300 splits, at about 7-8 %.

Fresh-split rates per map (250 splits; 16 px / 32 px):
- alone_uninstructed: cue .060 / .056, lick .052 / .044
- full_fr: cue .076 / .068, lick .064 / .068
- unique_fr: cue .052 / .048, lick .016 / .016
- unique_position: cue .044 / .044, lick .036 / .032
- uniquefr_position: cue .048 / .052, lick .076 / .064
- uniquefr_task: cue .032 / .024, lick .048 / .052
- uniquefr_uninstructed: cue .032 / .024, lick .052 / .048

**Figure-level family rate.** In 37 % of splits at 16 px (36 % at 32 px), at least one of the 14 map panels has a
false bin. Each panel is calibrated, but the quantities x epochs grid is not corrected as a whole. Captions should say
"per panel".

## 3. What drives the excess: candidates checked

- **Sessions treated as independent: no.** The outer level is the animal. Each animal contributes one difference, and
  sessions only sharpen it.
- **Unbalanced animals across halves: no.** Every split is within animal. The only imbalance is PS93's 3 v 4.
- **Bonferroni not covering all bins, or one- vs two-sided: no.** maxstat covers every tested bin, and both the
  interval and the threshold are two-sided.
- **Too few animals or sessions: the effects cancel.**
  - With n = 4 the plug-in bootstrap SE is about 0.87x the unbiased SE, and the normal-type threshold has light tails
    compared with t on 3 df. Both make the test liberal.
  - Resampling sessions inside each animal adds variance on top of the between-animal spread. That makes it
    conservative.
  - Net, the test is close to nominal (4.6-4.9 %).
  - Inflating the SE by sqrt(4/3) as a "small-n correction" drops the rate to 1.7 %. That over-corrects, so it is not
    recommended.
  - An exact animal-level test is not possible: 4 animals give 16 sign flips, and the minimum two-sided p is 0.125.
- **Drift inside pre: yes. This is the structure behind the false bins.**
  - **Explicit early vs late split.** First half vs second half of each animal's pre sessions, in date order (pre is
    0806-0814 in all four animals). Significant bins at 16 px:

    | map | cue | lick |
    |---|---|---|
    | full_fr | 78 (late lower) | 0 |
    | alone_uninstructed | 59 | 5 |
    | uniquefr_task | 120 | 115 |
    | uniquefr_position | 160 | 138 |
    | unique_position | 239 | 0 |
    | unique_fr | 0 | 0 |
    | uniquefr_uninstructed | 1 | 0 |

  - **Per animal at the strongest bin (late - early), all four animals agree in sign:**

    | map | PS92 | PS93 | PS94 | PS95 |
    |---|---|---|---|---|
    | full_fr cue | -0.118 | -0.114 | -0.103 | -0.082 |
    | unique_position cue | +0.030 | +0.024 | +0.025 | +0.029 |
    | uniquefr_position lick | +0.035 | +0.024 | +0.024 | +0.031 |
    | uniquefr_task cue | +0.039 | +0.018 | +0.036 | +0.033 |

  - **The session traces show the trend directly.** In-mask mean unique_position cue (x1e3) rises across pre in
    every animal:
    - PS92 -3 -> 50
    - PS93 15 -> 38
    - PS94 13 -> 28
    - PS95 20 -> 79
  - **Random splits with more early/late imbalance give more false bins.** Splits were grouped into quartiles of
    |imbalance|, where imbalance is the mean date-rank difference between halves:

    | quartile of \|imbalance\| | false-bin rate, 16 px |
    |---|---|
    | Q1 (most balanced) | 4.0 % |
    | Q2 | 2.2 % |
    | Q3 | 3.7 % |
    | Q4 (most imbalanced) | 9.9 % |

    Spearman correlation between |imbalance| and the number of maps hit: 0.21 (p = 0.001).
  - **False bins take the drift's sign.** The sign of the false bin times the sign of the split's imbalance matches
    the drift's sign in 16/19 false splits (full_fr cue), 11/11 (unique_position cue), 17/19 (uniquefr_position lick)
    and 8/8 (uniquefr_task cue).
  - Map-level rate vs drift strength is only weakly ordered (Spearman 0.12 / 0.32, n.s.). Even so, the most liberal
    map, full_fr cue, is the one with the strongest down-drift.
- **Bug in the calibration loop: no.**
  - My copy of the test reproduces `hierarchical_bootstrap_significance` exactly: 100/100 split x bin-size cells agree.
  - Re-running seeds 1000-1049 gives the render's 0.12 / 0.08 for full_fr cue.
  - The only design flaw is the shared seed across maps, which correlates the 28 cell rates.

## 4. What a fix would change on the real maps

Rule 8: every significant bin in every real cell has **4/4 animals with the same sign** as the pooled difference. The
per-animal counts equal the pooled counts in all 84 cells (`real_survival.csv`).

### (a) Empirical threshold

The threshold is the 95th percentile of max(|obs| / SE) / z over the 250 null splits.
- Pooled over maps, this percentile is <= 1, so the threshold changes nothing.
- Per map, it is above 1 only for these maps:

  | map | factor (16 px / 32 px) |
  |---|---|
  | full_fr cue | 1.070 / 1.055 |
  | full_fr lick | 1.030 / 1.033 |
  | alone_uninstructed cue | 1.031 / 1.015 |
  | uniquefr_position lick | 1.040 / 1.032 |
  | uniquefr_uninstructed lick | 1.005 / 1.000 |
  | uniquefr_task lick | 1.000 / 1.002 |

Bins lost with the per-map threshold (16 px unless noted; all others unchanged):

| cell | before -> after |
|---|---|
| full_fr cue subacute | 50 -> 45 |
| full_fr cue chronic | 39 -> 32 |
| full_fr cue subacute, 32 px | 13 -> 11 |
| full_fr cue chronic, 32 px | 12 -> 10 |
| full_fr lick subacute | 43 -> 37 |
| full_fr lick chronic | 211 -> 195 |
| full_fr lick acute, 32 px | 3 -> 2 |
| alone_uninstructed cue acute | 68 -> 63 |
| alone_uninstructed cue chronic | 89 -> 79 |
| alone_uninstructed cue subacute | 1 -> 0 |
| uniquefr_uninstructed lick acute | 383 -> 380 |
| uniquefr_uninstructed lick subacute | 160 -> 155 |
| uniquefr_uninstructed lick chronic | 394 -> 393 |
| uniquefr_position lick acute | 13 -> 9 |
| uniquefr_position lick subacute | 1 -> 0 |
| uniquefr_task lick acute, 32 px | 15 -> 13 |

No conclusion changes.

### (b) Drift overlap

These are real significant bins whose late - early pre drift has the same sign as the epoch effect:

| cell | sig bins | same-sign drift (point estimate) | drift-significant, same sign | notes |
|---|---|---|---|---|
| unique_position cue chronic, 16 px | 232 | 232 | 90 | |
| unique_position cue chronic, 32 px | 72 | 72 | 39 | |
| unique_position lick chronic | 149 | 149 | 0 | the lick drift is not significant |
| uniquefr_position cue chronic, 16 px | 25 | 25 | 14 | |
| uniquefr_position cue chronic, 32 px | 7 | 7 | 6 | |
| uniquefr_task cue subacute | 11 | 11 | 4 | |
| uniquefr_task lick chronic | 14 | 14 | 1 | |
| full_fr | - | - | 0 | the epoch effects are up; the cue drift is down |
| uniquefr_uninstructed | - | - | 0 | no drift |
| unique_fr | - | - | 0 | no drift |
| alone_uninstructed acute effects | - | - | 0 | |

So the chronic position-unique rise (cue) continues a rise that is already under way across the 8 pre days, in all
4 animals. A pre-mean baseline cannot separate "stroke effect" from "trend continued". FINDINGS records the same
question for the frozen residual ("late - early pre drift"; 8.x, `frozen_residual_trend.py`).

## 5. Recommendation

1. **Report the test as is. Do not re-threshold.** Quote the measured rate: "per-panel null, 250 within-animal pre
   half-splits: 4.9 % (16 px) / 4.6 % (32 px); per map 1.6-7.6 %". If the strictest form is wanted, apply the per-map
   empirical threshold in 4(a). It removes about 5-15 % of bins in full_fr and alone_uninstructed cue and changes no
   claim.
2. **Fix the null itself (code change, needs Priya's OK):**
   - use `--n-null 250` (about 25 min on 2 workers for 14 maps);
   - seed by (quantity, window, i), or keep the seed shared but report the pooled rate;
   - print the pooled rate in the suptitle, not the per-window 50-split rate, which is +/- 6 points of noise.
3. **Fix captions and docstrings.** "Bonferroni over bins" -> "max-statistic (family-wise) over bins". This applies to
   `fr_pixel_significance`, `fr_pixel_maps_sig`, `variance_maps`, `frozen_pixel_*`, `spout_unique_maps`,
   `movement_null_plan_maps`, `fr_direction` and `significance_contour`. Add "per panel; not corrected across
   panels".
4. **Add a drift control.** The substantive fix is a late-vs-early pre panel (as for the residual maps), or an
   "epoch vs late pre" comparison, next to the chronic maps. Flag unique_position cue chronic as "continues a pre
   trend" until that is shown.
5. Switching to `beta_maps.significance_contour` is not needed for calibration: it is the same engine. It would add
   pre-RMS weighting and rim suppression, which is a separate choice.

## 6. Page figures affected

Goal/encoding page v312 (MASTER_MANIFEST):
- **Fig 21.2 / 21.3** (`fr_pixel_sig_cue` / `fr_pixel_sig_lick`): contours plus the calibration line in the suptitle.
- Fig 21.12 (`variance_maps_lick`): same test. Its own null is fine; only the caption wording changes.
- The un-numbered `fr_pixel_maps_sig_<w>_modeA` (same contours) is also affected.
- Families that import the same `test()` (Fig 1.1 `spout_unique_maps`, Fig 2.1 `frozen_unique_sig`, Figs 22.5 / 22.6
  `fr_direction_maps`, `movement_null_plan_maps`) share the "Bonferroni" wording. The drift question applies to them
  too, but was not measured here.
- FINDINGS 10-25 entry ("null half-splits flag >= 1 bin in 5 %"): it still stands.

Note: the 250-split run took 24 min on 2 workers, a little over the ~20 min budget. All experiment processes have
exited.
