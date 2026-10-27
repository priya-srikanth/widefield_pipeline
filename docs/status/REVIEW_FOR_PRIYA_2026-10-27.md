# For Priya to review (overnight 10-26 -> 10-27, encoding window sabatinilab-d4)

Live page: https://claude.ai/artifact/Fnpn2KHknMwh3r13xADH7d -- v121 (~04:00) has everything below: new Figs 3.33-3.38
(maps, drift, recovery link, far-contra table), 3.14 significance bars, and a new 'Where things stand' row on the r loss. Newest at the bottom of each
section. Everything here is PROVISIONAL until you have looked at it.

## Key results to look at
1. **Page v119: sagittal figures + new KEY row** ("Where things stand", 3.15-3.22). Lateral r is lost, sagittal r is
   not (3.15) -> the loss is direction-specific, not a measurement problem. KEY row now quotes z-scored features for
   both axes (unscaled in brackets); the figures themselves are still unscaled (3.22 links the z version).

2. **Spread-matched r** (`where_spread_matched[_z]_tests.csv`, FINDINGS 4d): the LATERAL r loss survives matching each
   session's lick distribution to pre (ITI -0.11 to -0.12, cued -0.12 -> -0.05, 4/4 every cell); sagittal ~unchanged.
   Retires the "narrower spread" caveat for the lateral loss (matching is partial in PS95 ITI: SD 0.79 of pre).

3. **As aim recovers, the ITI contralesional offset fades -- now 4/4 with the z-scored decoder**
   (`figures/iti_recovery_link_angle_z.png`, one panel per animal over post-stroke days: black = far-contra cued lick
   angle / pre, red = decoded ITI lateral offset). Time trend removed: -0.48 (PS92 -0.35, PS93 -0.44, PS94 -0.62, PS95
   -0.51; p 2e-4). Cued sagittal offset also tracks recovery (-0.57, 4/4): "read further" acutely -> "read closer" as
   aim returns. PS95 angle course: fast rise to ~0.95 by day 9, ~0.85 days 11-20, ~0.95 later (your "dip").

4. **Far-contra pattern test** (biology question about the chronic far-contra ipsi reading; FINDINGS 4d): the
   chronic far-contra pattern is the same pattern at lower gain in PS92 / PS94 / PS95 (cos 0.85-0.97, gain 0.57-0.84),
   degraded with partial drift toward close contra in PS93 -- but far ipsi loses gain as much (0.25-0.93), so the
   across-spout contrast shrinks broadly while only far contra is read ipsi. Not resolved; worth discussing.

5. **The ITI lateral r loss is not elapsed-time drift** (`figures/iti_where_drift.png`): single-pre-session decoders
   barely lose lateral r over the 8-day pre span; post sessions sit below that curve (ITI chronic -0.13, 4/4; cued
   lateral loss gone chronically; sagittal ~0).

6. **Haufe pattern maps of the lick-position decoder** (`figures/iti_decoder_patterns_lat.png`, per animal
   `iti_decoder_patterns_animals_lat.png`): pre-stroke lateral map = crossed push-pull in lateral orofacial S1/M1
   (more lesioned-side activity for contralesional licks). Preserved chronically in PS92 / PS93 / PS95 (similarity
   0.76-0.90); PS94's ITI map is genuinely different (0.19, reliability 0.99) -- the least-recovered animal remapped.
   PS93 / PS95-ITI pre maps are mostly one-signed (less clear push-pull) -- worth a look.

7. **Per-spout maps across epochs** (`figures/iti_spout_patterns.png`, your "map the decoder weights per position"):
   crossed maps per spout; FAR CONTRA's map is replaced acutely (sim 0.28-0.42) and returns chronically (0.61-0.97);
   the other spouts stay similar. Sagittal map is bilateral and preserved.
8. **3.14 now has per-lag significance bars**: ITI lateral below pre at every lag in all three bases and epochs.

9. **F0 check** (FINDINGS 4d): raw baseline fluorescence did not fall after the stroke (+11 % acute, +4 % chronic n.s.),
   so the larger ITI post-lick dF/F response (3.23-3.25) is not a baseline artefact; in dF units it is slightly larger.
   415 nm lick-aligned response still not done.

10. **Three-axis version (low priority, done)**: lateral / forward / down in one z-scored run reproduces everything --
    the cued "closer" of the KEY row is the FORWARD axis every epoch, and the DOWN axis carries "ahead acutely, closer
    chronically". Nothing new; numbers in FINDINGS 4d.
11. **Significance bars added** to the ITI-lick PETHs (3.23 / 3.24 / 3.26) and the orofacial-vs-control PETHs (3.31).

12. **RMSE = bias + scatter**: the larger post-stroke decoding error is a SHIFT of the predictions (ITI lateral |bias|
    +0.81 / +0.41 / +0.36 mm), not more scatter (error SD unchanged) -- consistent with the offset story; PS94 carries
    the chronic ITI bias.

## Questions for you
1. Recovery ranking: by far-contra cued lick ANGLE / pre, chronic PS92 1.01, PS93 0.77, PS94 0.13, PS95 0.89. You
   rank PS95 best (fast early recovery, dip, then improvement) -- is there a measure you want used for the
   between-animal ranking (e.g. area under the recovery curve, days to criterion)?
2. Should figures 3.15-3.22 be redrawn on z-scored features too (KEY row already uses them)? ~15 min chain.
3. Drift check: the August-only pre span is 8 days, so chronic gaps (~39 days) are extrapolated. June pre sessions
   would give 2-month pre -> pre gaps -- OK to use them for THIS drift reference only (August-only stays the baseline)?
4. 415 nm lick-aligned response (handoff to-do): CLAUDE.md says the haemodynamic arm is CLOSED ("do not spend analysis
   effort there"), so I did not run it. Do you still want it as a control for the larger post-lick response?
5. sabatinilab-e3 (PI deck) asked to be told when the decoder and activity-map figures are final: not told yet.

## Issues / caveats
1. The chronic cued far-contra "read ipsilesional" interpretation (scaled-down command vs changed strategy vs blurred
   pattern) is open; the "smaller decoded size" argument was WITHDRAWN (measured from the mouth on forward only).
2. Jaw per far-contra lick (proposed test of the "changed strategy" reading) is BLOCKED: the kinematics lick table has no
   per-lick jaw measure (jaw is per trial) -- needs the pose window (sabatinilab-5b).
3. Haufe maps use unscaled SVD features (z-scoring 200 components per session is not the same as per-area z-scoring);
   the decoder offsets / KEY row use z-scored areas. Say if you want a z-scored map version.
4. PS95 acute has a single session, so no acute map / pattern for PS95 (shown as n = 3 animals).

---

## AFTERNOON / EVENING RUNNING LIST (10-27, Priya away) -- results, decisions, questions as they come in

### Decisions recorded (DECISIONS.md 2026-10-27)
- Decoder r and decoder offsets: LATERAL primary; angle (+ polar elevation / reach, + x/y/z) supplementary.
- Behaviour (recovery-link aim, landing, recovery ranking): ANGLE, as change vs pre (deg), not a ratio.
- No lick-extension filter (all tracked licks; the 3-D lick table has no lip licks).
- Pixel maps: max-statistic (FWER) solid contour + BH-FDR & >= 3/4 dashed (FWER vs FDR discussed: FDR = localisation,
  map-level permutation = whole-map test, max-stat = per-pixel claims / decoder-page comparability).
- 3.15-3.22: redraw on z-scored features (running; headline changes will be flagged here).
- Recovery measure: cued angle offset from own pre landing per spout, weighted far contra 3 : far mid 2 : others 1,
  SIGNED (absolute adds each animal's pre noise floor, ~2x in PS93). Chronic - pre: PS92 -0.5, PS95 -2.4, PS93 -3.0,
  PS94 -5.7 deg -> ranking PS92 > PS95 > PS93 > PS94 (fraction of acute deficit left: 4 / 15 / 39 / 27 %).

### Results so far (all PROVISIONAL)
- Far-contra cued read-out vs landing (FINDINGS 4d0b): acutely the code follows the ipsilesional deviation but sits
  contralesional of an unchanged-decoder reading (beyond pull +0.21 / +0.04 / +0.68 / +0.05 mm); chronically read
  ipsilesional beyond the pull (-0.15 to -0.35, 4/4).
- Overcorrection test (iti_overcorrection[_angle].png): chronic ipsilesional read-out is specific to contralesional
  spouts (far contra -0.27 mm 4/4; close contra -0.11 3/4; others n.s.). Lick-to-lick slope at far contra drops in mm
  (-0.12, 4/4) but NOT in angle (-0.06, 2/4) -> in angle: shifted offset with intact coupling (fits a downstream
  correction); in mm: also weaker coupling (fits a degraded code). Not decisive.
- Polar decoder (iti_where_angle_reach.png): angle offset ITI +4.6 / +2.8 / +3.1 deg (4/4 every epoch), cued +2.7 ->
  resolved; elevation read steeper-down acutely only (+2.4 ITI, +1.3 cued, 4/4); cued reach read further acutely
  (+0.16) then shorter (-0.16 subacute and chronic, 4/4). The x/y/z "forward" discrepancy appears as reach, not elevation.
- Angle vs lateral recovery link (z-scored): angle -0.48 (4/4) vs lateral -0.40 (4/4) -> angle kept for behaviour.
- Why angle r is noisier: angle is measured from the apparatus (spout-projected) mouth origin -> error ~ offset/extension
  on short licks; atan2 amplifies lateral noise. Single-view frame share 18-36 %, flat across epochs and lick length.

- Z-SCORED REDRAW of 3.15 / 3.19-3.22 done (commit 8bd5f72; files *_z.png beside the originals). HEADLINES HOLD: no
  lateral decoder result or cued far-contra result changes sign or significance; same animal counts. Lateral r / pre
  ITI -0.25 / -0.22 / -0.23 (unscaled -0.22 / -0.19 / -0.20), cued -0.19 / -0.11 / -0.08; per animal within ~0.08.
  Smaller cells move: ITI 2-D acute becomes significant; ITI reach chronic, cued 2-D / angle subacute lose it (CIs touch
  0); per-spout cells 33 -> 44 of 108 significant (lost: cued close-contra lateral acute +0.20* -> +0.01; gained mostly
  acute cued sagittal and ipsi / mid lateral offsets; sign flips only in n.s. cells). -> Proposal: swap the page figures
  3.15 / 3.19-3.22 to the z-scored versions (unscaled kept as supplementary), captions note the changed cells.

- WHOLE-MAP PERMUTATION (per-session self-fit maps, unit-RMS = shape only; session_pattern_maplevel.csv): nearly EVERY
  map differs from pre (BH), chronic included -- the session-label permutation is powerful and also detects slow drift
  (pre sessions precede post), so read the DISTANCE d = 1 - r, not p. Cued per-spout d acute: far contra 0.67 (3/3) vs
  0.09-0.35 for the other five spouts; chronic far contra 0.17, inside the others' range (0.10-0.34). Lateral-position map
  d: ITI 0.39 / 0.35 / 0.33, cued 0.36 / 0.21 / 0.16 (ITI stays further from pre). Within-spout (executed-direction) maps
  d ~0.41-0.48 in both sets, every epoch. Re-running with a DRIFT FLOOR (early vs late pre sessions) to read d against;
  a bug (pandas `T.map` = method) had dropped the d / p subtitles from the change-map panels -- fixed.
- ITI scatter "SEM ovals look bigger" (Priya): they are 1-SD spread ellipses (never SEM; unchanged since 10-25). They grew
  on 10-26 22:45 when the y-axis switched from OUTWARD (forward) to SAGITTAL extension (forward + down), which spreads more
  (SD ~0.7-0.9 mm). The z-scored redraw did not change them. QUESTION: switch to SEM ellipses (mean +/- SEM) or keep SD?

- CONFOUND CHECKS on the larger lick response (iti_hemo_check.py; FINDINGS 4d0c): LATER increase survives everything
  (zero-phase filter, uncorrected 470, F0 per area, vessel pixels, pre-lick gap matching); 415 nm grows only ACUTELY
  (+0.17 of the pre response vs +0.76 corrected; mostly PS95 / PS94) and the acute increase is somewhat edge-rim weighted
  -> up to ~1/5 of the ACUTE increase may be haemodynamic / motion. PS94 runs during ~50 % of its subacute / chronic
  licks (9 % pre). Figures: iti_hemo_check_{variants,peth,edge_vessel,f0_areas,state_gap}.png.

- MIXED-EFFECTS MODELS (value ~ epoch + (1 | animal), sessions as observations; iti_mixed_models.py,
  mixed_models.csv): agree with the nested bootstrap on every headline, all p < 0.001 unless noted:
  ITI lateral r -0.26 / -0.19 / -0.22; cued lateral r -0.21 / -0.10 / -0.07; ITI lateral offset +0.32 / +0.22 / +0.24;
  cued lateral offset +0.25 / +0.06 (p 0.03; bootstrap n.s.) / -0.03 (n.s.); far-contra cued offset +0.35 / -0.09 (p 0.07;
  bootstrap significant) / -0.26; ITI response +0.83 / +0.42 / +0.37; aim (weighted signed angle) -14.5 / -5.5 / -2.9 deg.
  Mixed CIs are narrower than the bootstrap's (4 animals -> random-effect variance poorly estimated -> anti-conservative);
  the two disagree only on the two borderline subacute cells. Suggest: bootstrap stays on the figures, mixed-model p in
  captions / methods for a paper.

- PER-ANIMAL far-contra / close-contra maps (iti_spout_animal_panels.png): acute far-contra map is replaced in PS92 /
  PS93 / PS94 (r to own pre 0.37 / 0.28 / 0.52) by a similar layout in all three -- bilateral medial frontal / midline
  positive, lateral (incl. the lesioned-side lateral orofacial spot) weaker or negative; close contra kept (0.86-0.90).
  Chronic far contra back in PS92 / PS94 / PS95 (0.89-0.92), NOT PS93 (0.46; subacute 0.15). PS95 has no acute map (one
  session). Caveat: the lateral negatives sit near the window edge, and the acute response increase is somewhat
  edge-rim weighted (hemo check) -- check the acute far-contra layout against the edge rim before interpreting it.

### Running / queued
- Whole-map permutation + amplitude-normalised change maps (per-session self-fit maps, 107 sessions) -- drawing.
- Max-stat contour redraws of 3.50-3.53.
- Then: one page publish; mixed-effects models (sessions in animals) for headline results; per-animal per-spout panels;
  first-lick read-out; behaviour-only decoder (low). Tell the deck window the decoder figures are final AFTER the publish.

### Questions for Priya (accumulating)
- Page figures 3.15 / 3.19-3.22: swap to the z-scored versions (unscaled -> supplementary)?
- Scatter ellipses: 1 SD (now) or SEM?
- Whole-map test: also add a June no-stroke map reference (needs per-session SVD-basis features for June)?
