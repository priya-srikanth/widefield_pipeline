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

### Addendum outcome (2026-10-30)
Code: `sm_coherence_cca.py ctrl-run` (107 curated sessions, 4 workers, 54 s, frozen env for the FaceRhythm factors) /
`ctrl-summarise` / `plot` -> `figures/fig5_itpc_controls.png` (figdata `fig5_itpc_controls.pkl`). Tables `tables/ctrl_*`
(sessions, wide_sessions, coverage, behav_var_sessions, epoch_effects_animal, tests_boot_delta, share_remaining); logs
`ctrl_run.log`, `ctrl_summarise.log`. Values = ITPC minus random-phase null; the raw version reproduces the published
session values exactly (max |diff| 8e-17; same seeds, so raw / kin / perm share their subsamples).
* **Coverage**: movement design usable in 107 / 107 sessions (16 columns: 14 covariates + 2 missingness indicators);
  events matched to a 3-D lick 100 %; every covariate >= 90 % per animal x epoch (lowest speed_peak .90, jaw_amp .91).
  Covariates explain 21-42 % of the event-to-event variance of the S / M traces (permuted control 2-7 %).
* **Share of the pre -> chronic change remaining** (alt / raw; pooled = boot_delta points; per animal PS92 / 93 / 94 / 95):
  | | sensory 1.5-4 Hz (raw +.165 [.083, .228], 4/4) | motor 1.5-4 Hz (raw +.095 [.028, .147], 4/4) |
  |---|---|---|
  | kinematics regressed | **1.11** (+.183 [.099, .244], 4/4); 1.14 / 1.06 / 1.06 / 1.52 | **1.08** (+.102 [.044, .157], 4/4); 1.13 / 1.02 / .87 / (PS95 raw ~0) |
  | permuted control | 1.03; 1.05 / 1.02 / .99 / 1.17 | 1.03; 1.05 / 1.01 / .93 / - |
  | + log SNR | **.51** (+.085 [.028, .143], 4/4); .51 / .75 / .37 / .04 | **.26** (+.025 [-.040, .098], 2/4); .32 / .87 / -.35 / - |
  | + behavioural variability | **1.01** (+.166 [.097, .223], 4/4); 1.08 / 1.02 / .85 / 1.49 | **.92** (+.087 [.022, .154], 4/4); 1.09 / 1.06 / .37 / - |
  PS95's chronic change is ~0 in raw (S +.041, M +.001), so its ratios are not meaningful. Low band: kin .86 (S) / .96 (M),
  bvar .93 / .89, log SNR -.05 / .13 (the low-band rise is fully absorbed by SNR).
* **Behavioural variability does NOT fall consistently**: mean within-spout SD (pre-z units) chronic -.038 [-.091, +.025]
  (3/4 down; PS92 -.03, PS93 -.08, PS94 -.09, PS95 +.05); acute -.040 (PS95-driven, -.14); subacute ~0. Its within-animal
  slope on ITPC has no consistent sign.
* **Log SNR rises** (S 1.5-4 Hz chronic +.83 [.51, 1.10], M +.83 [.52, 1.12], both 4/4). Caveat: SNR (event-mean power /
  deviation power) and ITPC are both measures of how consistent the event response is relative to its trial-to-trial
  variability, so "explained by SNR" is close to a restatement, not a mechanism: it says the ITPC rise goes with a larger
  lick-locked response relative to trial noise, not with more variable / different movements. Sensitivity (not
  pre-specified): SNR on the same 30-event subsamples as the ITPC leaves even less (S chronic per animal .061 / .118 /
  -.026 / -.003, M .007 / .102 / -.131 / -.035).
* **Reading (PROVISIONAL, n = 4):** the ITPC rise is NOT explained by lick kinematics (regressing them out per event
  slightly increases it) and NOT by behavioural variability (which does not fall consistently); it goes together with a
  larger response SNR, which accounts for about half (sensory) to most (motor) of it.
* Deviations / choices: the movement design is [spout dummies + movement columns] without the 3-D lick angle (the
  addendum lists spout dummies only; S17's "old" design had the angle); eligibility = S17's rule with its base-design
  column count; permutation within spout (addendum) rather than S17's global permutation; behavioural variability uses,
  per animal, the covariates with >= 50 % coverage in every session (all 14 here), z-scored on the animal's pooled pre
  events; SNR on all events of the session (as written). `plot --plot-only` re-rendered figs 1-4 from their unchanged
  figdata.

## Addendum: GCaMP6s kinetics caveat (2026-10-30, after Priya's comment)
The deconvolved (tau 0.6) rows were computed by `run` but never summarised. Pre -> chronic, cued, S8M8, evoked, value
minus null, per animal PS92 / 93 / 94 / 95: ITPC S 1.5-4 Hz raw +.179 / +.220 / +.219 / +.041, deconvolved +.173 / +.221 /
+.229 / +.057; ITPC M 1.5-4 Hz raw +.136 / +.148 / +.094 / +.001, deconv +.128 / +.154 / +.099 / +.016; coherence low raw
-.057 / -.140 / -.130 / -.021, deconv -.067 / -.164 / -.145 / -.030. The agreement is EXPECTED and uninformative: a
linear time-invariant deconvolution scales and phase-shifts each frequency identically on every event, so ITPC and
coherence are unchanged by it except for window-edge effects. The indicator low-pass (tau 0.6 s: |H| ~.17 at 1.5 Hz,
~.07 at 4 Hz) and the 2-s window mean the 1.5-4 Hz phase is set by the shape / onset edge of the slow lick-evoked
transient relative to non-calcium noise. Reading fixed: the phase measures are a consistency / amplitude readout of the
evoked transient (already shown in time-domain amplitude, FINDINGS 0b), not oscillatory phase-locking.
