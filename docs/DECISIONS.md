## 2026-11-07 -- FaceRhythm factor matching, map statistics, per-session tests (Priya; a8 window)
- MATCHING RULE v2 (`face_rhythm_epoch_shape.MATCH_RULES`, `--match-rule v2`; v1 stays the code default until the page /
  deck switch): spatial score as before, but among columns within .05 of a factor's best spatial cosine the one with the
  best SPECTRAL cosine wins (Priya: "spatial cosine score, then only for factors with similar spatial cosine use
  frequency"). Hungarian re-tested and rejected again (re-creates the 10-25 PS95 lick_left mis-match). Product
  (spatial x spectral) matching rejected: circular for the spectral readouts.
- RECOVERED = spatial cos >= .7 AND spectral cos >= .5. Spectral floor from null fits (5th percentile .46). Spatial .8
  first, LOWERED to .7 by Priya: .8 dropped PS95 acute lick_right (same column as pre, spectral .94, spatial .78 = a
  changed, not a lost, factor); .5 and .7 pick the same real cells. Unrecovered cells leave the shape readouts and are a
  readout of their own (factor LOSS).
- MERGED UNITS: sniff_prelick + nose_prelick_slow = prelick_nose; run_onset_jaw + prep_jaw = slow_jaw (master spatial cos
  .99, unstable even in null fits). The fast jaw factors stay separate (merging the whole jaw cluster would hide their
  acute loss).
- MAP STATISTICS (Priya "ok for the stats as you propose"): CLUSTER MASS is the map test (mesh neighbours, |t| > 2,
  sign-separate), the MAX-STATISTIC the point-level one; shuffles WITHIN animal; every session-shuffle test gets an
  early-vs-late PRE DRIFT CONTROL (`face_rhythm.stats.session_shuffle_test`, `drift_split`). Spectra: keep the
  max-statistic (changes are narrow peak shifts; clusters along frequency find fewer). Maps drawn like the speed maps:
  every point coloured, rings red = up / blue = down; marked points never in the pale middle of the scale.
- PER-SESSION TESTS ARE THE MAIN SHAPE TESTS (Priya "ok"): every session's own fit (own8) matched to the master; maps,
  scalars (asymmetry, spatial / spectral angle to the animal's pre mean, side specificity) and factor loss as delta vs
  each animal's own pre; within-animal session shuffles, no refit (`scripts/face_rhythm_session_loading_perm.py`,
  `scripts/face_rhythm_session_shape_tests.py`). WHY: the pooled-fit refit null is inflated for factors that reorganise
  after the stroke (lick_left: pre-only pooled fits 4-7 deg, shuffled pools up to 37 deg, regardless of how many acute
  sessions they hold) -- a fragility check flags such factors (lick_left, lick_far_nose) as "pooled test
  inconclusive"; and a pooled fit is not the average of the animals (chronic lick_left: 3 / 4 animals' own fits +.27
  asymmetry, pooled fit -.12).
- EVERYTHING AS DELTA vs THE ANIMAL'S OWN PRE wherever possible (Priya); pooled-fit results are a descriptive second
  view. The 500-permutation O2 pooled null is now optional for subacute / chronic (acute complete).
- PS93 stays in the main analyses (n = 4, decided before seeing results); "without PS93" (= LOAO-PS93, retitled) as the
  sensitivity. PS93 had a pre-existing right facial deficit and a strong pre-stroke RIGHT lick bias (46 % right, 6 %
  left); its lick_right is often split over two lick_right-like components (pre margin .02), so its per-session maps
  flip. Split-aware merging NOT adopted (circular, can hide real splits); instead ambiguous sessions (margin < .05) are
  an exclusion sensitivity.
- FIGURES Priya likes for the page: cue- and lick-aligned speed maps (permutation versions), per-session lick_right /
  lick_left weighting, per-session other lick factors.

## 2026-10-28 (late night) -- balance definition for recovery markers (Priya)
- PRIMARY marker FIXED (Priya "proceed with the model including SSs"; "what's the harm in 0-0.5"): gain ratio 0.1..0.5 s
  after the first lick, pool ALM + tjM1 + SSp-m + SSp-n + SSs, both cued and ITI (one window: 0.1-0.5 best for both,
  0.70 / 0.42; every window from 0.1 s ~equal for ITI; the first 0.1 s carries nothing). scripts/lick_recovery_primary.py,
  page 14.16 (KEY); the all-lick raw 0-0.2 s balance figures (14.17-14.19) kept, marked as the earlier definition.
- Recovery-marker balance is measured on the FIRST POST-CUE LICK (cued) and the FIRST ITI LICK AFTER >= 2 s QUIET (ITI),
  not on all licks (Priya: "we'll use first iti lick after quiet and first post cue lick based windows").
- Cued window CONFIRMED on held-out animals (scripts/lick_balance_nested.py; window chosen on 3 animals, tested on the
  4th): first-post-cue-lick gain ratio 0.2..0.35 s, chosen in 4/4 folds, nested LOAO 0.69 (null p 0.005; days 0.23).
  ITI: no window survives the held-out selection on its own (timing families <= 0.11); 'any definition' 0.22 vs days
  0.17 (p 0.01). Area subsets: scripts/lick_balance_subsets.py (running at this note).
- Leave-one-EPOCH-out dropped as a criterion (Priya: tests extrapolation to an unseen aim range, 3 unequal folds);
  replaced by within-animal BLOCKED CV (4 contiguous blocks per animal, 1-session gap, animal intercepts;
  lick_recovery_markers.cv_blocked).

## 2026-10-28 (night) -- recovery markers, Shapley grouping, pre-cue reference (Priya)
- Pre-cue LEVEL is referenced to RESTW (Priya "3. yes"): position-weighted flat rest median, restdock05, engagement-gated,
  docked=False as the other restw consumers (scripts/lick_precue_restw.py; 0/107 fallbacks). Standalone figure 14.20,
  not a row of the AUC maps (a level, different units).
- Shapley grouping: SSs FOLDED INTO the orofacial group for the main 11.14 image (Priya; SSs unique share 0.000 4/4);
  6-group split kept as a supplement (WFIELD_BLOCK_VARIANT unset). The Allen atlas has no MO trunk area -> body = SSp-ul,
  -ll, -tr.
- Recovery-marker cross-validation is now a committed script (scripts/lick_recovery_markers.py; reproduces the earlier
  ad-hoc 14.16 numbers exactly): LOAO (with within-animal shuffle null), leave-one-EPOCH-out, LOSO animal unknown / known.
  Pre sessions scored leave-one-pre-session-out for the grey reference cloud (14.16, 14.17). Fair pre-fit weights =
  fit on the OTHER animals' pre licks (z-scored balances).
- Residual-ROI maps: per-area outlines + fading; atlas areas with no value (FRP, AUDd / AUDpo right, VISp left) blanked
  in the display (they were the 'edge fragments').
- Non-linear decoder versions added as collapsible supplements (12.1, 12.15, 12.21, 13.1, 13.6).


## 2026-10-28 -- Key figure and cued wording (Priya)
- KEY FIGURE replaced (Priya: "yes to your A and B; and use the newer behavior recovery metric"): Key figure 1 =
  scripts/key_figure.py (A behaviour = far-contra FIRST-lick aim; B/C frozen vs epoch-refit lateral decoders, + the old
  far-contra code on hits; D ITI like-for-like offset, any-spout and same-target). Key figure 2 = Fig 13.2 (lick order;
  Priya "another key figure"). The old key figure stays as Fig 0.6.
- CUED "read ahead of the failing movement" RETIRED (Priya "ok to reword"): the cued like-for-like offset is target
  composition (same-target same-place +0.01 n.s.; first cued lick +0.05 n.s.); page status rows, 12.x, 13.4 title and
  FINDINGS 4d0-V8 / 4d0-NIGHT re-worded. The ITI offset is the changed read-out.
- 'Where things stand' bottom line rewritten 10-28 (10-25 version: FINDINGS section 0).
- ALM / tjM1 encoder rows keep the STAT MASK (rim check: no difference, scripts/checks/rim_check_alm_tjm1.py).
- Like-for-like curves (12.2) Gaussian-smoothed with the offset's sigma, no display bins; SEM over ANIMALS.
