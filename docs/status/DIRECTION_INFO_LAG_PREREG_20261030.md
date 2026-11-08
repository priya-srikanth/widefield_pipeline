# Sensory vs motor timing of lick-DIRECTION information: design and gate, written BEFORE any fit (2026-10-30)

Priya approved (10-30, "sure") after the phase measures were demoted for GCaMP6s kinetics (FINDINGS 4.17 caveat). Option 4
of `docs/CROSS_AREA_METHODS_LIT_20261030.md` (Lemke 2024 "direction as information, not influence"): a time-domain
question that needs no oscillation and no lag model. Code: `scripts/sm_direction_lag.py`. Outputs:
`<iti_licks>/sm_direction_lag/`. Data looked at before this was written: event counts per session only (cued first
licks ~250-830 per session; ITI docked ~60-880).

## Question
Does information about WHERE the lick goes appear in sensory-area activity before or after it appears in motor-area
activity, and does that order change after the stroke? This is information timing, not influence: an earlier
information onset in one group says nothing about which group drives the other.

## Data (fixed)
Events, features and sessions as `sm_coherence_cca.py` (same loader `_arrays` / `ihc._sets`): cued first licks (primary)
and ITI docked licks (secondary); -1..+1 s at 0.05 s (41 bins, lick onset at 0); area features / the session's area SD.
Groups S8 = SSp-n, SSp-m, SSs, SSp-ul x both hemispheres; M8 = tjM1, ALM, MOp_rest, MOs_rest x both. Secondary pairings
SiMi, ScMc (4 vs 4, same hemisphere). Curated sessions, PS92-95, pre = AUGUST only. Cohort2 not included.
TARGETS: primary = executed 3-D first-lick angle (deg; includes the spout); secondary = WITHIN-SPOUT angle (angle minus
the session's mean angle at that spout: the executed deviation, which carries no target information).

## Decoder (fixed)
Per session, per time bin t, per group: ridge regression of the target on the group's 8 area values at bin t
(instantaneous, no history), features standardized on the training events, lambda = 1.0 x n_features (fixed, no
tuning), 5-fold CV with folds assigned within spout (every fold sees every spout). Score = Pearson r(CV prediction,
target) over events -> curve r_G(t). Equal event count across sessions: N = 200 events drawn without replacement, 10
draws, curves averaged (sessions with < 200 events use all and are flagged; < 60 events -> skipped). Null: target
permuted across events (within spout for the within-spout target is identical to global, so global), 20 permutations
on one draw, per-bin 95th percentile and the peak-of-null distribution.

## Readouts per session (fixed)
* baseline b_G = mean r_G over -1.0..-0.7 s; peak p_G = max r_G over -0.5..+1.0 s (time t_peak).
* **L50_G** = first time (from -0.7 s forward, before t_peak) the baseline-subtracted curve reaches 50 % of
  (p_G - b_G), linear interpolation between bins; NaN if p_G does not exceed the peak-of-null 95th percentile.
* **PRIMARY: dL50 = L50_M - L50_S** (s; + = sensory information first).
* Shape check (amplitude-invariant): **xlag** = lag maximizing the correlation of the baseline-subtracted, peak-normalized
  r_S and r_M over -0.5..+1.0 s, lags +-0.3 s (+-6 bins), parabolic sub-bin interpolation (+ = M lags S).
* Secondary: p_S, p_M (peak information), L50_S and L50_M separately.

## Stability gate (per animal, AUGUST pre, cued, S8M8, total angle)
1. Signal: p_S and p_M both above the peak-of-null 95th percentile in >= 75 % of pre sessions.
2. Reliability: split-half (odd / even events, all events of each half, no subsampling) SD of dL50,
   sqrt(mean d^2 / 2) over the animal's pre sessions, <= 0.05 s (one bin). The lags of interest are tens of ms (jaw leads
   the lick by 52-120 ms, FINDINGS ITI pre-lick re-run), so a noisier readout cannot resolve them.
**Interpret pre vs post only if >= 3 / 4 animals pass.** Every number reported either way; within-spout angle, xlag,
SiMi / ScMc and ITI docked reported, not gated.

## If passed (fixed now)
Per-animal table first (rule 8), then `ihc._tests` (boot_delta per post epoch vs August pre). A change is reported as a
change only if the CI excludes 0 AND >= 3 / 4 animals share its sign. Pre level of dL50 (who leads pre-stroke) reported
with its per-animal values and a session bootstrap.

## Known limits (stated now)
* GCaMP6s (decay ~0.6 s) and hemodynamic residuals delay and smear both groups' signals. Both groups share the
  indicator, so a RELATIVE lag is interpretable, but differences in response size / SNR between groups or epochs can move
  a half-max crossing; the post-stroke cued response is larger (FINDINGS 0b). xlag is the amplitude-invariant check, and
  a dL50 change that xlag does not reproduce is not reported as a timing change.
* Area-level features (8 per group), not pixels; a group's information can be limited by spatial averaging.
* Total angle includes target information available before the lick (cue / spout); the within-spout angle isolates the
  executed deviation (previously ~0 before the lick, FINDINGS ITI pre-lick re-run).

## Outcome (appended after the fits, 2026-10-30)
107 sessions (31 August pre), 43 s wall (4 workers). **Gate PASSED 3 / 4** (primary: cued, total angle, S8M8): signal in
100 % of pre sessions in every animal (peak r S .62-.77, M .58-.72; null 95th pct ~.10-.16); split-half SD of dL50
PS93 .008, PS94 .024, PS95 .012 s (pass), PS92 .125 s (fail; between-session SD .012, i.e. single noisy halves). Not
gated: within-spout angle passes in 0 / 4 (peak r ~.10-.15, barely above null; L50 unstable), SiMi 1 / 4, ScMc 0 / 4,
ITI docked total 1 / 4 -> those are not read.
* **Pre-stroke: motor information reaches half-max ~23 ms BEFORE sensory** (dL50 PS92-95 -.031 / -.017 / -.026 /
  -.016 s; xlag -.033 / -.020 / -.015 / -.021 s; 4/4 both readouts). Sub-bin (bins are 50 ms; interpolated).
* **Post-stroke the motor lead shrinks** (dL50 change acute +.029 [.016, .046], subacute +.011 [.001, .022], chronic +.014
  [.006, .021], 4/4 each). xlag (amplitude-invariant check) reproduces it only chronically (+.012 [.004, .021], 4/4;
  acute +.015 and subacute +.011, CI includes 0, 3/4) -> per the rule above, **only the chronic change is reported as a
  timing change**: motor lead ~23 ms pre -> ~9-10 ms chronic.
* What moves: sensory information arrives earlier (L50_S -.034 / -.031 / -.038 s, 4/4 every epoch); motor earlier only
  chronically (-.024, 4/4). Peak information rises subacute / chronic in both groups (chronic p_S +.066, p_M +.071, 4/4),
  i.e. similarly in both, so a group-specific amplitude difference is not the obvious driver; a sensory-specific
  steepening (larger sensory response, FINDINGS 0b) cannot be excluded.
* Reading limits: total angle carries the spout target (pre-lick r ~.2-.3 above null throughout = the pre-cue position
  code); the timing readout is the steep lick-onset rise on top of it. Calcium + hemodynamic timing, area level. Jaw
  opening leads tongue onset by 52-120 ms, so the motor-first half-max may partly be movement-locked activity around
  lick onset. Information timing, not influence.
Figures `figures/fig1_curves.png` (curves per epoch, both targets), `fig2_lags.png` (per-animal dL50 / xlag); tables
`tables/` (sessions, gate_animals[_all], tests_boot_delta); figdata `figdata/`; `plot --plot-only` re-renders.
