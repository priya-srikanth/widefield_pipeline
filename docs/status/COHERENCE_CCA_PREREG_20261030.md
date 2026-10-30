# Sensory-motor COHERENCE vs CCA (Guo 2021 / Heimbuch 2026 style): design + gate, written BEFORE any fit (2026-10-30)

Priya 10-30: "let's try coherence vs cca and if appropriate then dlag". Background: `docs/CROSS_AREA_METHODS_LIT_20261030.md`
(Guo et al. 2021 Cell Rep: LFP coherence FELL over recovery, cross-area CCA ROSE and predicted success; Heimbuch et al.
2026 bioRxiv: DLAG cross-area factors became more stereotyped with recovery, within-area factors did not). Code:
`scripts/sm_coherence_cca.py`. Outputs: `<iti_licks>/sm_coherence_cca/` (figdata/, figures/, tables/). Nothing about
these quantities has been looked at before this was written.

## Data (fixed)
* Events: CUED first licks (`lick_interhemispheric._set_arrays(label, meta)["cued"]`, the set used in 14.18 S10 / S17 /
  S18), -1..+1 s around the lick at 0.05 s (41 samples), area features `iti_licks.area_features` (dF/F, z-scored as the
  decoders). Secondary set: ITI docked licks (`interhemi_crossarea._sets(...)["iti_dock"]`), same treatment.
* Groups (as S18 / MVAR): S = SSp-n, SSp-m, SSs, SSp-ul; M = tjM1, ALM, MOp_rest, MOs_rest; each x hemisphere
  (left = ipsilesional). PRIMARY populations: S = 8 areas (both hemispheres), M = 8 areas. Hemisphere-pair variants
  (Si-Mi, Sc-Mc, Si-Mc, Sc-Mi; 4 x 4 areas) reported, not gated.
* Sessions: curated, pre = AUGUST only, epochs from `epochs.epoch_of`; >= 30 events per session (MIN_N).
* Two signal versions per event: **evoked** = the trace as is (Guo / Lemke: event-locked activity incl. the mean
  response); **trial-to-trial** = minus the session's mean trace for that spout (per-spout PSTH removed, i.e. what is
  shared beyond the average response; Heimbuch demeans by the trial average).

## Measures
**A. Coherence (the Lemke 2019 / Guo 2021 first measure).** S-group mean and M-group mean traces per event (mean of the
group's z-scored areas). Per event: linear detrend, Hann taper, FFT of the 41 samples (zero-padded to 64; 0.31 Hz grid,
true resolution ~0.5 Hz, Nyquist 10 Hz). Magnitude-squared coherence across events |<S M*>|^2 / (<|S|^2><|M|^2>), and
inter-trial phase coherence (ITPC) of each group. Bands: **low 0.5-1.5 Hz**, **delta 1.5-4 Hz** (the Ganguly LFO band;
GCaMP6s attenuates power above ~3 Hz, phase-based measures are used for that reason). Bias control: coherence computed on
a matched number of events (random subsample of N_SUB = 30 events, 50 draws, mean) so that event count cannot move it;
null = the same with the S and M events paired from DIFFERENT events (shuffle), reported as coherence minus null.
Deconvolved (tau 0.6 s, `lick_interhemispheric._deconv`) as sensitivity.

**B. CCA (Guo 2021).** Rows = time samples of the events concatenated (events x 41), S block (8) and M block (8),
each column z-scored on the training events. Top canonical correlation **CC1**, **cross-validated**: 2-fold by events
(odd / even trial order), weights fit on one half and the canonical correlation of the projections evaluated on the
other, averaged over both directions (in-sample CCA is biased upward with dimension; Guo used in-sample). Small ridge
(1e-3 x trace / dim) on each block covariance. Null = the same with M events shuffled against S events (trial pairing
broken; for the evoked version the null keeps the shared mean response, so CC1 - null measures trial-locked SINGLE-TRIAL
alignment beyond what the PSTH alone gives). Also CC2, and the number of canonical pairs whose held-out correlation
exceeds the 95th percentile of the null (20 shuffles).
**Stereotypy (Guo / Heimbuch).** For CC1 projections (S side and M side) and, as the WITHIN-AREA comparator, the top PC
of each block after projecting out its canonical direction(s): per event, Pearson r of the single-event trace vs the
leave-one-out mean of the other events at the same spout; session median. (No kinematic matching / time warping: lick
duration is ~0.1-0.2 s, far below the window; matched-kinematics version is a later sensitivity if needed.)

## Gate ("appropriate", per animal, AUGUST pre, primary = cued, trial-to-trial version for CCA, evoked for coherence)
1. **Signal**: CC1 held-out exceeds its shuffle null in >= 75 % of pre sessions; band coherence exceeds its null in
   >= 75 % of pre sessions (each band separately).
2. **Reliability of the readout**: split-half (odd / even events) SD of the session value smaller than the between-
   session SD over the animal's pre sessions (a session-level number must carry more than estimation noise; the pooled
   MVAR failed exactly this).
3. **CCA weight stability**: |cosine| between the CC1 weight vectors fitted on the two halves, median >= 0.7 over pre
   sessions (each block).
A measure is interpreted pre vs post only if it passes 1-2 (CCA also 3) in >= 3 / 4 animals. Every number reported
either way.

## If passed (fixed now)
Per-animal table first (rule 8), then `analysis_kit.boot_delta` vs August pre per post epoch, for: coherence (both bands),
ITPC, CC1 held-out minus null, stereotypy of CC1 vs within-block PC. Recovery tracking (Guo / Heimbuch claim): within
animal, partial r (| log day, post sessions) of each readout with cued hit rate and far-contra aim
(`interhemi_crossarea._recov_targets`), count of tests at p < 0.05 reported against the number expected.
Interpretation: a change is a change only if the CI excludes 0 AND >= 3 / 4 animals share its sign. The Guo pattern
would be: coherence down or flat, CCA / CC1 stereotypy up with recovery; the Heimbuch pattern: CC (cross-area)
stereotypy tracks performance, within-block PC stereotypy does not.

## DLAG decision (fixed now)
Go to DLAG only if CCA passes the gate in >= 3 / 4 animals (a cross-area subspace is reliably present at session level).
DLAG would add per-factor timescale and S-vs-M delay; the reference implementation is MATLAB (Gokcen et al.), so it needs
a port / separate env (no installs into the shared env) -- costed and brought to Priya before running.
