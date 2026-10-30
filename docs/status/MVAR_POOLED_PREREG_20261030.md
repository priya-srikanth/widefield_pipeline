# Pooled sensory / motor MVAR: design and stability gate, written BEFORE any pooled fit (2026-10-30)

Priya approved (10-30, "let's try the smaller mvar") after the 18-region MVAR failed its gate (2 / 4 animals;
`MVAR_PREREG_20261030.md`, FINDINGS 4.15). Code: `scripts/mvar_pooled.py` (reuses the fitting, CV and gate code of
`scripts/mvar_regions.py` unchanged; only the nodes and the lag range differ). Outputs:
`M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/mvar_pooled/`.
Data looked at before this was written: the 18-region MVAR gate tables only (no pooled-node quantity).

## Why this reduction
The 18-region matrix was predictive (cross-region gain > 0 in 31/31 sessions) but its individual entries were not
reproducible in PS93 / PS95. The S18 question (does sensory -> motor directed influence weaken?) only needs group
averages, so the nodes are pooled BEFORE fitting (fewer, less noisy series; 36 instead of 324 coefficients per lag),
and the lag range is extended because the 18-region fits chose p = 3, the top of their range, everywhere.

## Data (fixed)
Same prepared sessions as the 18-region MVAR (`curbd_pilot.prep`, held-out encoder residuals, engaged rows, 3-frame
means ~10.4 Hz, runs = contiguous engaged stretches). Each of the 18 region series is z-scored over the session, then
averaged into 6 NODES: S_i, S_c (SSp-n, SSp-m, SSs, SSp-ul), M_i, M_c (tjM1, ALM, MOp_rest, MOs_rest), R_i, R_c (RSPd,
posterior CONTROL node); i = left = ipsilesional, c = right. Modes as before: raw (primary), gsr (each of the 18 regions
minus its regression on the mean of the other 17, BEFORE pooling), dec (deconvolved, tau 0.6 s, after pooling).

## Model, CV (as the 18-region pre-registration, except)
* p in {1, 2, 3, 4, 6, 8} (up to ~0.77 s); lagged samples need all 8 lags inside the run for every p (common sample
  set). lambda grid unchanged (10^-5 .. 10, 19 values).
* W = lag-summed coefficients (6 x 6, standardised units); 30 off-diagonal entries.

## Stability gate (per animal, AUGUST pre, raw; identical thresholds)
1. Prediction: cross-node gain > 0 in >= 75 % of pre sessions and >= 4/5 outer folds in the median session.
2. Halves: r(off-diagonal W, first vs second half) median >= 0.7, no session < 0.5.
3. Sessions: median pairwise r of off-diagonal W across pre sessions >= 0.7.
4. Ridge: r(W at CV lambda, W at lambda x10 and / 10) median >= 0.7 each.
5. Pair-specific halves (double-centred W) median >= 0.5.
**Interpret pre vs post only if >= 3 / 4 animals pass on raw.** Every number reported either way.
Reported, not gated: the split-half difference of each PRIMARY readout (below), as a noise floor to read the post
changes against.

## If passed (fixed now)
All curated sessions at the animal's FIXED (lambda = geometric median of its pre CV choices, p = their mode); per-session
CV choice = sensitivity. PRIMARY readouts, from W[target, source]:
* `S->M intra` = (W[M_i,S_i] + W[M_c,S_c]) / 2; `S->M inter` = (W[M_c,S_i] + W[M_i,S_c]) / 2; the same for M->S;
* `dir intra` = S->M intra - M->S intra; `dir inter` likewise (the S18 Granger analogue);
* per hemisphere pair: `dir Si<->Mi`, `dir Sc<->Mc`, `dir Si<->Mc`, `dir Sc<->Mi`.
SPECIFICITY control: `R->M` (mean of the 4 RSPd -> motor entries) and `dir R<->M`, same treatment; a sensory-specific
loss should exceed the RSPd one. Per-animal table first (rule 8), then `analysis_kit.boot_delta` per post epoch.
Interpretation rule: a change is reported as a change only if the CI excludes 0 AND >= 3 / 4 animals share its sign.
