## 2026-11-17 -- Like-for-like curves: local-linear reference with a Gaussian kernel; strict = each animal's own pre curve (Priya; a8 window)

- **Curves** (decoded vs landing per epoch) are Gaussian-smoothed per animal x epoch (sigma 0.1 mm, held-out pre choice;
  drawn where the weight is >= 10 licks), mean +/- SEM over animals, dashed where only 3 animals. No binned dots: the bin
  means are the same data, and the binned display could rest an edge bin on one or two animals.
- **Reference** for the offset, the bar tests and the read-outs = the Gaussian-weighted LOCAL LINE (loclin). Priya:
  "pick loclin with gaussian kernel". A Gaussian-mean-reference row and a loclin row looked identical (the reference
  never changes the curves).
- **Two versions**: all post licks (edge line carried beyond the pre range) and STRICT like-for-like = post licks only
  where the SAME animal's pre curve of the SAME lick set exists, curves cut to that range. NOT the loclin `beyond` flag
  (0.5-99.5 % of all pre licks): it reaches past where a pre curve exists. Code: lick_matched_readout.py --inside-curves
  (inside_curves), scripts/lick_matched_panels.py. Bars stay on 0.25-mm bins (a test needs cells).
- One combined figure (cued all / cued first / ITI all / ITI first / ITI docked) replaces the separate curve figures that
  repeated the cued panel (goal page Fig 12.2 v317; deck slide 125).

## 2026-11-17 -- Lick source of the lick-free rest (RESTnl) and of the LP lick trains; every caption names its train (Priya)
- Priya 11-17: "ensure all figure captions / slide notes are clear about what data they used -- I don't know if we
  specify everywhere cam4 vs cam4 + cam1 LP licks". Audit and fixes: docs/status/LICK_SOURCE_AUDIT_20261117.md; the
  wording lives in `wfield_local/lick_provenance.py` (one phrase per train) and the analysis deck's provenance line
  now names the train on every slide.
- RESTnl (lick-free rest, `restnl_mask`; also `rest_frozen_decoder_nl`, `state_space_rest_baseline`) removes every DAQ
  spout-contact lick and every cam4 LP lick of `licks_all.csv`, each widened [0.5 s before, 2.0 s after]. It was
  defined 10-29 / 11-01 from cam4 `licks_all.csv` (`scripts/restw_strict_check.py` variant `strict`, then
  `restnl_mask`); cam1 was never added and no reason was recorded (the `strict_c1` variant with cam1 `events_all` was
  computed in that check but not adopted). Coverage is not the reason: cam4 `licks_all.csv` and cam1 `events_all.csv`
  are detected over the SAME trial windows (strobe - 0.5 s .. stop + 3 s, cut at the next trial; ~98-99 % of the
  video, `orofacial_outputs.all_licks_session`), and ITIs are ~1-1.5 s, so both cameras effectively cover the
  session. cam1-only (not cam4-matched, droplet-filtered) events add ~3-7 % licks per animal.
- DECIDED (Priya 11-17, later the same day): RESTnl moves to DAQ + cam4 + cam1 (cam1 `events_all`, droplets removed).
  The RESTnl products are re-run on branch `analysis/restnl-cam1-1117`; until those outputs are placed, every current
  RESTnl output is DAQ + cam4 and its caption says "rest licks excluded: DAQ + cam4". The wording is generated from
  `restnl_mask` (`lick_provenance.restnl_text`), so it reads "DAQ + cam4 + cam1" once the mask declares cam1.
- The LP lick train (`lp_licks.lp_lick_times`, WIDEFIELD_LICK_SOURCE=lp; also `combined_encoding` LICKS=combined and
  the frozen encoder) = cam4 `licks_all.csv` + cam1-only events of `cam1_licks/<a>_<d>_events.csv`, a POST-CUE file
  (cue .. trial stop, `cam1_licks.detect(span="post")`). So cam1 adds licks to the post-cue lick / no-lick call, the
  attempt classes, the first-lick time and the post-cue lick-onset kernel, but never to a pre-cue window: the 1 s ENL
  lick-free window under LP, and the encoder's pre-cue / ITI lick kernels, are in practice cam4 only. Captions say
  "pre-cue lick-free window: LP video, cam4 (cam1 is scored only from the cue to the trial stop)". Switching that
  input to `events_all` would change every LP output (cache tag `lp_licks.spec_token` follows the cam1 folder) and is
  not done here.
- Other trains, for the record: DAQ-source imaging runs use DAQ contact for the lick class, the lick alignment and the
  2 s ENL window; the camera trial classes (`trial_classes`) are DAQ contact + LP cam4 + cam1 (droplets removed); ITI
  licks, the per-lick kinematics / 3-D tables and FaceRhythm lick windows are cam4 only; quiet frames
  (QS_LICK_SOURCE=daq+lp) exclude DAQ + cam4 + cam1 `events_all`. PS93_0813 has no video (no LP licks, no camera
  classes, no RESTnl); every other curated cohort-1 session has cam1 events (11-17: 120 curated, 1 without).

## 2026-11-17 -- ITI response window 0..0.5 s; cued stays 0..1.0 s (Priya; a8 window)
- Priya 11-17: "for ITI let's use 0-0.5 ... we should probably just change ITI AUC and balance results to the 0-0.5s".
  Reverses the ITI half of the 10-28 (late) "one window for both" decision (0..1.0 s for ITI and cued; 0..0.5 s was the
  secondary "it matches the ITI kinetics"). Cued / cue-aligned results keep 0..1.0 s.
- Why: ITI bout-start responses are biphasic -- a short lick transient (peak ~0.2-0.3 s) then a dip below baseline at
  ~0.6-1.2 s in both hemispheres (clearest PS93 pre / acute / subacute, PS94 pre). Over 0..1.0 s the two cancel, the
  contra/ipsi AUC ratio is undefined in 2.8 % of ITI session x spout rows (docked 6.7 %; PS94 docked pre 17 / 47) --
  dropped non-randomly -- and the docked tjM1 spout x epoch model failed to converge. 0..0.5 s holds the lateralized
  transient: 0.3 % undefined (docked 0.8 %). Cued: 0.0 % undefined at 0..1.0 s. Diagnostic:
  docs/status/deck_swap_20261117/ITI_NEG_AUC_DIAGNOSTIC.md; traces .../iti_licks/neg_auc_diag_20261117/.
- Scope: every ITI response-AUC and contra/ipsi balance result (lateral ratio ITI rows, ITI AUC summaries / maps, ITI
  ramp companions, docked ITI arm). The primary recovery marker already uses 0.1..0.5 s for ITI (unchanged).
- Bounded index (c - i) / (|c| + |i|) considered (defined for every row; = (r - 1) / (r + 1) when both AUCs > 0) but in
  the dip sessions it would compare undershoots; window fix preferred. Kept as a robustness option.
- (2026-11-17, Priya "yes update 113 and 114 to match") Cued per-spout first-lick / cue response AUC (Fig 14.7, deck 113 /
  114; `lick_cued_ramp.spout_windows`) aligned to `iti_licks.CUED_RESP_WIN` (0..1.0 s; was 0..0.5 s, kept as secondary).

## 2026-11-16 -- June sessions against August-frozen references: drift sub-analyses only (Priya; a8 window)
- Priya 11-09 asked for the frozen pre-stroke analyses on the June sessions (August-frozen reference); scope settled
  11-16: **"just for drift sub-analyses"**. June is used as a longer (~2-month) pre-stroke drift floor for a readout,
  not as an extra baseline and not to re-run every frozen analysis. Done in FR (June floor, FR page supplement to Fig
  7.1) and in the jaw 3-D composition floor. Other arms (imaging position decoder, behavioral-state decoder, frozen
  encoder, no-lick decoder, CD trajectories) add June only where a drift sub-analysis is the question.
- FR per-trial readouts (a8, 11-16): prevalence vs amplitude-given-movement and per-trial timing are read on single
  trials and summarized per session; statistics stay at the session level (animal -> session bootstrap); conditioned
  quantities are tested only with a pre baseline in >= 3 animals; expression floored at 1 before log2.

## 2026-11-09 -- Aligned frozen decoder: the gate is alignment efficiency (Priya)
- GATE SWITCHED AFTER THE FIRST RUN (`scripts/aligned_decoder.py`). Written before the run: pre -> pre gap fraction
  (aligned - frozen) / (refit - frozen) >= .7 in >= 3/4. It cannot pass when nothing has moved: size-matched frozen and
  refit are nearly equal pre -> pre (G .04-.11), so the ratio divides by ~0 (values -3.7 .. +58). Now: pre -> pre
  alignment EFFICIENCY (aligned - 1/6) / (frozen - 1/6) >= .7 in >= 3/4 -- the 10-30 re-embedding prereg's G1, not a new
  criterion. Procrustes passes (LocaNMF .71 / .93 / .73 / .68; SVD .87 / .87 / .85 / .87). The old fraction stays in
  verdict.csv (`gate_pre_fraction`). Verdicts (Procrustes, k = 10): acute RE-EMBEDDED, subacute / chronic PARTIAL, both
  bases. Every other part of the verdict rule is unchanged.

## 2026-11-08 -- Refit floor, display floors, lick vs successful-lick classes (Priya)
- D7 REFIT FLOOR = 10 TRIALS ONLY. Every refit analysis marks a position untrainable only below 10 trials; the 1/18-of-
  session share rule (`grant_confusion.MIN_REFIT_SHARE`, added 09-09, never discussed with Priya) is RETIRED. The D6
  trial-mix control (matched pre subsamples, no floor) is shown beside the refit figures. Priya: 1/18, 1/20 and 10 are
  all arbitrary choices; 10 is kept for the refit fits only.
- PER-SESSION DISPLAY FLOORS REMOVED. The hidden per-session 5-trial floors (`epoch_accuracy._accuracy_of`,
  `epoch_kit._accuracy_at` and the other `< 5` sites) dropped sessions from the per-session tables; they are removed
  (branch fix/refit-floor-10-no-display-floor, together with the D7 change).
- LICK vs SUCCESSFUL-LICK CLASSES. Lick class = every camera-seen lick (contact + incomplete) on the LP source
  (`WIDEFIELD_LICK_SOURCE=lp`, LP lick onset, ENL 1.0 s, LP-trained frozen decoders). Successful-lick class = DAQ
  contact, supplemental only; never gate on it. PS93_0813 (no video) is in the successful-lick class only. Correction
  recorded at the 11-03 (night) camera-gate entry in the root DECISIONS.md.
- (2026-11-08) Section G G1-G3 (Priya 11-08): train camera WORKING vs NO_RESPONSE (unknown excluded), score post
  no_response rows as the no-lick set, working rows as the engaged comparison; DAQ-grouping supplement not run.
- (2026-11-09) Section G G4 (Priya 11-09): G4 trains working vs no_response, as G1-G3 (unknown never trained; no_response
  and unknown scored apart; stored id "neg-no_response"; DAQ path unchanged).

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
- MAP STATISTICS (Priya "ok for the stats as you propose"): CLUSTER MASS is the map test (mesh neighbors, |t| > 2,
  sign-separate), the MAX-STATISTIC the point-level one; shuffles WITHIN animal; every session-shuffle test gets an
  early-vs-late PRE DRIFT CONTROL (`face_rhythm.stats.session_shuffle_test`, `drift_split`). Spectra: keep the
  max-statistic (changes are narrow peak shifts; clusters along frequency find fewer). Maps drawn like the speed maps:
  every point colored, rings red = up / blue = down; marked points never in the pale middle of the scale.
- PER-SESSION TESTS ARE THE MAIN SHAPE TESTS (Priya "ok"): every session's own fit (own8) matched to the master; maps,
  scalars (asymmetry, spatial / spectral angle to the animal's pre mean, side specificity) and factor loss as delta vs
  each animal's own pre; within-animal session shuffles, no refit (`scripts/face_rhythm_session_loading_perm.py`,
  `scripts/face_rhythm_session_shape_tests.py`). WHY: the pooled-fit refit null is inflated for factors that reorganize
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
- (night) v2 is now the DEFAULT `--match-rule` (epoch_shape, drift_floor, cohort_null; v1 selectable; folders keyed on
  the rule name, so nothing moved).
- (night) EVERY MAP TEST AT 10,000 SHUFFLES, like the scalars (Priya "should the speed maps be 10k?"): at 2,000 a p
  near .05 carries ~+-.005 Monte-Carlo error and 4 / 14 / 17 clusters of the position / aligned / session-loading
  maps sat at p .03-.08. Re-run: position maps unchanged; aligned lick maps gain 2 small clusters; the cue-aligned
  drift control loses its far_center cluster (2 left, both at far_R, where the post-stroke change is a decrease);
  session loading: lick pair unchanged, two exploratory flips. Page / slide text reworded with Priya's OK.
- (night) RHYTHM-BAND MAPS (`scripts/face_rhythm_band_maps.py`): per-point spectra x a unit's master frequency
  profile (v2 units) = movement in that unit's rhythm, no refit; lick_right + lick_left share one band, read by
  spout. Cue 0..+4 s only (no spectra stored for the 0..2 s window). The rhythm GRADIENT is tested per session as
  band minus lick rhythm (face-wide mean log2), within-animal shuffles, Holm over 6 bands x 3 epochs.
- (night) WORDING (Priya asked what "the 7 Hz band falls more" means): the 'lick_ongoing_7Hz' profile is the UPPER
  part of the lick spectrum (6.5-15 Hz), so the result is reported as a downward FREQUENCY SHIFT of face movement,
  not as the loss of a separate fast movement; quantified directly by the spectrum's power-weighted center
  (3-15 Hz main, 4-12 Hz sensitivity; log-frequency mean because the bins are log-spaced).
- (night) AMERICAN SPELLING EVERYWHERE (Priya): prose in pages, deck, docs and code string literals uses American
  spellings (center, color, behavior, normalized, gray, labeled). Code identifiers, file names, paths and column names
  are NEVER renamed (outputs and caches are keyed on them). Files whose SOURCE feeds a cache key (map_collect
  code_digest / closure, trial_classes csig, result_cache code_version, combined_encoding._CACHE_CODE = full bytes of 7
  files, rest_frozen_decoder_nl._code_key) are left alone: even a docstring edit there invalidates a cache (section G
  lost ~1 h to one today). Convert those only when a render is about to rebuild that cache anyway.

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
  Pre sessions scored leave-one-pre-session-out for the gray reference cloud (14.16, 14.17). Fair pre-fit weights =
  fit on the OTHER animals' pre licks (z-scored balances).
- Residual-ROI maps: per-area outlines + fading; atlas areas with no value (FRP, AUDd / AUDpo right, VISp left) blanked
  in the display (they were the 'edge fragments').
- Non-linear decoder versions added as collapsible supplements (12.1, 12.15, 12.21, 13.1, 13.6).


## 2026-10-28 -- Key figure and cued wording (Priya)
- KEY FIGURE replaced (Priya: "yes to your A and B; and use the newer behavior recovery metric"): Key figure 1 =
  scripts/key_figure.py (A behavior = far-contra FIRST-lick aim; B/C frozen vs epoch-refit lateral decoders, + the old
  far-contra code on hits; D ITI like-for-like offset, any-spout and same-target). Key figure 2 = Fig 13.2 (lick order;
  Priya "another key figure"). The old key figure stays as Fig 0.6.
- CUED "read ahead of the failing movement" RETIRED (Priya "ok to reword"): the cued like-for-like offset is target
  composition (same-target same-place +0.01 n.s.; first cued lick +0.05 n.s.); page status rows, 12.x, 13.4 title and
  FINDINGS 4d0-V8 / 4d0-NIGHT re-worded. The ITI offset is the changed read-out.
- 'Where things stand' bottom line rewritten 10-28 (10-25 version: FINDINGS section 0).
- ALM / tjM1 encoder rows keep the STAT MASK (rim check: no difference, scripts/checks/rim_check_alm_tjm1.py).
- Like-for-like curves (12.2) Gaussian-smoothed with the offset's sigma, no display bins; SEM over ANIMALS.

## 2026-11-07 -- Where-decoder: UNSCALED features are primary; within-session z-scoring is supplementary at most (Priya)

Priya 11-07: "different gains per area would be meaningful, i don't think we want to normalize this way." The 10-26
within-session z-score check (`scripts/iti_zscore_check.py`: every area divided by its session frame-wise SD, pre and
post) removes per-area gain changes, which may be part of the stroke effect. Primary = the unscaled frozen where-decoder
(`where_decoder_tests.csv`); z-scored outputs (`zscore_check/*_z*`) are a sensitivity check only. Page figures drawn
from z-scored producers (Fig 10.2, 12.5-12.7, 13.3, 16.3-16.5) switch to unscaled producers in the replacement pass.
Where the two disagree (e.g. cued sagittal subacute -21 % unscaled vs -9 % z-scored) the gap is the part explained by
per-area gain. Fig 10.2 key statement rewritten the same day (FINDINGS / ENCODING_FINDINGS, unscaled numbers).

## 2026-11-18 -- cam1 tongue: detached, stationary labels masked BEFORE cleaning (Priya)

A cam1 tongue label sitting on a water drop hanging from the spout (PS93 0814 trial 540, four-camera review) became
seconds-long "licks" after cleaning bridged its wobble, and cam1-only events enter the LP lick train and the camera
classes (~23 trials incomplete -> no response, 11 acute). Priya: exclude points that are "relatively stationary and
'jumping' from the mouth", "before cleaning / interpolating". Rule: `cam1_licks.mask_detached` (raw confident runs,
gaps <= 24 ms joined; min out > 30 px AND spread < 6 px -> not visible), at both cam1 cleaning sites (`scripts/cam1_licks`,
`pose3d.cleaned_xy`). Evidence and rejected versions in the `cam1_licks` constants block and
`status/STATUS_2026-11-18_CAM23_RERUN_INVENTORY.md` section 0b. Applies from the cam2 / cam3 re-run (no partial
regeneration before it).
