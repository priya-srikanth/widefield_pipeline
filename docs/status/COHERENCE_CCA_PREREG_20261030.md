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

## Outcome (appended after the fits, 2026-10-30)

**GATE: coherence / ITPC PASS (6 / 6 measures, >= 3 / 4 animals); CCA FAILS (1 / 4 animals) -> no CCA / stereotypy
pre vs post, and the DLAG criterion is NOT met.** Script `scripts/sm_coherence_cca.py`. 31 August pre sessions (both
families) 44 s wall (4 workers), gate 1 s; then 76 post sessions coherence family only 37 s; summarise 25 s; plots 13 s.
* Gate, per animal (PS92 / PS93 / PS94 / PS95), cued, S8M8, tau 0. Signal: every measure above its null in 100 % of pre
  sessions in every animal (CCA also above the null 95th percentile in 100 %; median 8 / 8 canonical pairs above null).
  Reliability = split-half SD / between-session SD (< 1 passes):
  coh low .79 / .50 / .46 / .50 (4/4); coh delta **1.15** / .71 / .86 / .42 (3/4); ITPC S low .61 / .54 / .68 / .54;
  ITPC S delta .93 / .42 / .19 / .66; ITPC M low .92 / .76 / .44 / .75; ITPC M delta .97 / .60 / .33 / .44 (all 4/4);
  **CCA CC1 - null 1.55 / 2.09 / 1.45 / .66 (1/4)**. Weight stability |cos| S .97 / .99 / .99 / .94, M .97 / .99 / .99 / .96
  (passes everywhere).
* Why CCA fails: held-out CC1 is at ceiling (pre mean .984 / .991 / .983 / .977, CC2 ~ .96; null ~ 0 on the ttt version),
  so the between-session SD is tiny (.006-.010) and smaller than the split-half SD (.007-.014): a session's CC1 is mostly
  estimation noise. The S and M CC1 projections are near-identical traces (example figure) -- the shared S/M subspace is
  dominated by a common, cortex-wide component at widefield resolution, not a specific sensory-motor channel.
  (Not gated, context: on the EVOKED version CC1 passes 4/4; on ITI docked licks ttt 2/4.)
* Pre vs post (cued, S8M8, coherence family; per-animal tables printed first, `tables/summary_sessions.csv.gz`):
  coherence minus shuffled-pair null, LOW band: subacute -.074 [-.122, -.022], chronic -.087 [-.138, -.032], 4/4 animals
  down both (PS92 / 93 / 94 / 95 chronic -.057 / -.140 / -.130 / -.021); acute -.021 (n.s.). DELTA band: no change
  (all CIs span 0). ITPC rises from subacute: S delta +.114 / +.165 (4/4), S low +.060 / +.057 (4/4), M low +.045 /
  +.044 (4/4), M delta +.085 (3/4) / +.095 (4/4; PS95 ~0). Acute: nothing.
  **Read with care:** raw evoked low-band coherence did NOT fall (pre .89-.94 -> chronic .92-.96); the drop in
  coherence-minus-null comes from the NULL rising (.36 -> .46-.54), i.e. more of the S-M coherence is the shared
  event-locked response, consistent with the ITPC rise. So the result is "post-stroke lick responses are more
  phase-locked in both groups, with no extra trial-specific S-M coherence", not a Guo-style coherence loss.
  ITI docked licks: no change in any coherence / ITPC measure (all CIs span 0, <= 2-3/4 same sign).
* Recovery tracking (within-animal partial r | log day, post sessions, targets far-contra aim and far-contra hit rate):
  5 / 24 tests p < 0.05 (1.2 expected), all cued: coh delta x aim +.27 (4/4 sign), coh low x hit -.25 (3/4), ITPC M low x
  hit +.25 (3/4), ITPC S delta x hit +.34 (4/4), ITPC S low x hit +.35 (3/4). Uncorrected, PROVISIONAL.
* Deviations (forced, decided before the gate was read): stereotypy is computed on EVOKED traces projected with the
  version's cross-fitted weights (on ttt data the leave-one-out spout mean is -x_i / (k-1), r = -1); ITPC null = random
  phases on the same subsamples (spec gave none); "exceeds its null" = value minus null mean > 0 (95th-percentile CCA
  version tabled, identical outcome); split-half SD = MVAR definition sqrt(mean d^2 / 2) (the SD of a half-session
  estimate, conservative); halves with < 30 events use N = min(30, n_half) per draw; recovery uses
  `interhemi_crossarea._recovery` (on `_recov_targets`: aim + far-contra hit rate). Example animal PS92 fixed in code
  before the fits (most pre sessions).
Tables `<iti_licks>/sm_coherence_cca/tables/` (sessions, gate_animals, gate_animals_all, gate_measures, tests_boot_delta,
recovery_partial_r); figures `figures/fig1_gate.png`, `fig2_change_vs_pre.png`, `fig3_example_pre_vs_chronic.png`
(figure data in `figdata/`).

## Addendum: ITPC controls (design written BEFORE running, 2026-10-30 late)
Priya 10-30: less interested in how reproducible the lick response is per se (behaviour shows that), but a regression
showing whether the ITPC rise is explained by behaviour / response size is useful. Cued first licks, S8M8, evoked.
1. **Event-level kinematic regression.** Per session, regress each event's group trace (S, M; all 41 samples) on the
   per-event movement covariates of `scripts/coupling_vigour_sm.py` (same set and coverage rules as 14.18 S17: 3-D reach,
   peak speed, lick duration, protrusion, lick count +-0.5 / +-1 s, FaceRhythm factors, jaw amplitude, landing error) plus
   spout dummies; keep spout means + residual (i.e. remove only the movement-explained part), recompute ITPC (same 30-event
   subsampling, 50 draws). Control: the same with covariate rows permuted within spout.
2. **Session-level adjustment.** Per session: neural SNR in each band (power of the event-mean trace / mean power of
   the event deviations, same taper / FFT) and behavioural variability (mean over covariates of the within-spout SD,
   each covariate z-scored on the animal's pre sessions). Within animal, ITPC_session ~ epoch + log SNR (and, separately,
   + behavioural variability); report the epoch effects with and without each covariate (per-animal table first, then
   boot_delta on the adjusted values = residual + epoch effect).
Read-out: the share of the pre -> chronic 1.5-4 Hz ITPC change that remains after (1), after SNR, and after behavioural
variability. Behavioural variability change itself is reported (for the behaviour reading).
