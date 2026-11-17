# Region-level ridge MVAR: design and stability gate, written BEFORE any MVAR fit (2026-10-30)

Priya approved (10-30) after the CURBD pilot failed its gate (`CURBD_PILOT_PREREG_20261030.md`; FINDINGS 4.14).
Code: `scripts/mvar_regions.py` (imports `prep` / regions from `scripts/curbd_pilot.py`). Outputs:
`M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/mvar/`.
Only data looked at before this was written: the CURBD pilot's PS94 numbers (incl. its exploratory VAR(1)) and one prep
log line (PS92_0806: 381 engaged trials, 45,721 samples at 10.42 Hz).

## Data (fixed)
`curbd_pilot.prep` per session, variant "one": 18 region means (SSp-n, SSp-m, SSs, SSp-ul, tjM1, ALM, MOp_rest, MOs_rest,
RSPd x left = ipsilesional / right), ROI v8, HELD-OUT residuals of that session's encoder design (10-fold contiguous
trial blocks), engaged-trial rows, 3-frame means (~10.4 Hz). Runs = contiguous engaged-row stretches (mostly one trial
each); a lagged sample is used only if all its lags lie in the same run. Each region z-scored on the training samples.

## Model
x_t(a) = c_a + sum_{l=1..p} sum_b B_l[a, b] x_{t-l}(b) + e. Ridge on all coefficients except the intercept, one
penalty per model (shared across targets), parametrised per sample: alpha = lambda * n_train, lambda in
10^{-4 .. 1} (13 values, 0.33-decade steps); p in {1, 2, 3}.
**AR-only baseline**: the same with B_l diagonal (each region from its own past), its own CV-chosen (lambda, p).
**Readout matrix** W[a, b] = sum_l B_l[a, b] (lag-summed, standardized units; comparable across p). Off-diagonal = 306
directed region pairs.

## Cross-validation (no leakage)
* OUTER: 5 contiguous blocks of runs (whole runs, time order). Test block held out; inside the training set a nested
  INNER 4-fold contiguous-block CV picks (lambda, p), separately for the full and the AR-only model; refit on the outer
  training set; score the test block. Folds are cut at run boundaries and lagged samples never cross a run boundary,
  so no test sample's lags are in training and vice versa.
* Held-out one-step R2 = 1 - SSE / SS(x_t - training mean), pooled over regions and outer folds; also reported
  normalized by the variance of the CHANGE (x_t - x_{t-1}). **Cross-region gain = R2(full) - R2(AR-only)**.
* The session's reported matrix: (lambda, p) chosen by the same 5-fold blocked CV on the whole session, refit on all
  samples.

## Stability gate (per animal, AUGUST pre sessions only; primary = raw residuals, matrix W)
1. **Prediction.** Cross-region gain > 0 in >= 75 % of the animal's pre sessions AND in >= 4/5 outer folds in the
   median session. (Otherwise the cross terms add nothing held-out and the matrix is not interpreted.)
2. **Halves.** W fitted on the first vs second half of the session (cut at the run boundary nearest the middle; each
   half its own CV-chosen lambda, p): Pearson r of off-diagonal W >= 0.7 median over the pre sessions, no session < 0.5.
3. **Sessions.** Median pairwise r of the full-session off-diagonal W across the animal's pre sessions >= 0.7.
4. **Ridge.** r(W at the CV-chosen lambda, W at lambda x 10) and r(W at lambda, W at lambda / 10) (same p), each
   >= 0.7 median over pre sessions. (The CURBD exploratory VAR moved from weak to heavy ridge with r 0.07-0.21; this
   asks whether the CV-chosen point sits on a plateau.)
5. **Pair-specific.** Criterion 2 after removing source and target main effects (double-centered off-diagonal W,
   `curbd_pilot.dcenter`) >= 0.5 median over pre sessions; the same for criterion 3 reported, not gated.

An animal PASSES if 1-5 all pass. **The MVAR is interpreted further only if >= 3 of 4 animals pass.** Every number is
reported either way. Sensitivities reported, NOT gated: deconvolved traces (s = x + 0.6 s * dx/dt within run, GCaMP6s;
finite difference at the 10.4 Hz sample rate, no extra smoothing because the data are already 3-frame means) and
leave-self-out global-mode removal (each region minus its regression on the mean of the other 17, fit on the session),
both with the same gate metrics.

## If passed (fixed now)
Fit every curated session (pre = August only; acute / subacute / chronic from `epochs.epoch_of`). For the pre vs post
comparison the penalty and lag are FIXED per animal (lambda = geometric median of its pre sessions' CV choices, p = their
mode) so that a change in shrinkage cannot masquerade as a change in coupling; the per-session CV choice is a
sensitivity. Groups (S = SSp-n, SSp-m, SSs, SSp-ul; M = tjM1, ALM, MOp_rest, MOs_rest; i = left / ipsilesional, c =
right): Si->Mi, Sc->Mc, Si->Mc, Sc->Mi and the reverse M->S groups, each the mean of 16 signed W entries; derived: S->M
minus M->S per group, intra (mean of the two within) and inter (mean of the two across). Per-animal table (epoch means,
change vs pre) printed first; then `analysis_kit.boot_delta` (paired nested) per post epoch. Sensitivities: global-mode
removed, deconvolved, per-session CV lambda. Compared with S18 (`scripts/coupling_vigour_sm.py`: Granger S->M minus M->S
falls post-stroke, vanishes with the global mode removed).

## Outcome (appended after the fits, 2026-10-30)

**GATE FAILED on the primary (raw): 2 / 4 animals pass (PS92, PS94); >= 3 needed -> no pre vs post fits** (the
`fixed` command refuses). 31 August pre sessions (PS93_0813 has no encoder json).
* Criterion 1 (prediction) PASSES everywhere: cross-region gain > 0 in 31/31 sessions, 5/5 outer folds in every session.
  Held-out one-step R2 full vs AR-only, median per animal: PS92 .877 vs .855, PS93 .859 vs .843, PS94 .872 vs .849,
  PS95 .909 vs .901 -> gain .017 / .017 / .025 / .010 = 10.8 / 9.8 / 15.5 / 8.7 % of the one-step CHANGE variance.
* Halves median r .877 / .874 / .814 / **.640** (worst session .786 / **.281** (PS93_0814) / .537 / **.436**);
  sessions .786 / .841 / .727 / **.393**; ridge x10 / /10 >= .98 everywhere; pair-specific halves .905 / .793 / .874 /
  .698. PS93 fails ONLY on its single worst session; PS95 fails halves-median and sessions.
* Sensitivities (not gated): global-removed 1/4 pass (PS93); deconvolved 2/4 (PS92, PS94). Gain rises with
  global removal (median 10.7-14.2 % of change) and deconvolution (R2 lower, gain .037-.077 of level).
* CV choices: p = 3 (the TOP of the pre-stated range) in every session and mode; lambda 1e-5 to 1e-3 (near-OLS), at the
  lower grid edge in 2 raw sessions (PS94_0807, PS94_0809). So criterion 4 tested a flat near-OLS region and was
  easy to pass; EXPLORATORY (after the gate): the CV matrix vs a heavy ridge (lambda 0.1 = the CURBD comparator) r
  .25-.37, vs lambda 1 r .10-.13 -- the heavy-ridge matrix is a different object (closer to lagged covariance).
* Runtime: prep 27 sessions 389 s (4 workers); 93 fits (31 sessions x 3 modes) 275 s wall (5-23 s each); gate 3 s.
Tables `<mvar>/tables/` (gate_sessions, gate_session_pairs, gate_animals, explore_heavy_ridge_raw); figures 1-2.
