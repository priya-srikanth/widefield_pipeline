# Running list — overnight 2026-11-02 → 11-03 (Priya: "go ahead with all that you can without me overnight. keep handoff updated and a running list of results and any decisions you need from me")

Handoff: `docs/status/STATUS_2026-11-01_QUIET_EVENTS_HANDOFF.md` (top block "PI DECK STATE"). Deck:
`M:/MICROSCOPE/Priya/Widefield/presentations/PI_overview_2026-11-02.pptx` (edit by slide TITLE via backed-up run-once
scripts in `_source_PI_overview_2026-11-02/`; one deck writer at a time).

## PRIYA'S ANSWERS 11-03 (morning)

- 1 grant: 2-photon grant figure received (panels A-F); per-tile map 131 / 231 / 3,390 matches panel A (5c settled). Add
  one NES and one H2B mCherry review sheet (panel C analogue) -> deck helper.
- 2 fluoroscopy: cohort-4 stroke 11-01 (11-02 = post-stroke day 1 = "post-stroke1"). Mtimes confirm `20261029/` holds
  the 10-29 session saved as `20261028_*` (written 10-29 18:41-19:12, one day after `20261028/`). Source files NOT renamed;
  slides label them 10-29.
- 3, 4, 8: wording and the out-of-section corrections approved.
- 5a: ipsi / contra are relative to both the planned lesion side and the imaged hemisphere (imaging LEFT).
  5b: LB333 axon-GCaMP8s injected into left (ipsilesional) VM thalamus.
- 6: 4-8 Hz NOT adopted; reword the pre-cue event claim (main + exclusion tests). Asked whether the low band is significant.
- 7: revisit after the state decoder's running class uses the encoding-model speed limits (floor 1.5 mm/s, cap) ->
  helper running.
- 9: (a) ENGAGED trials in the deck; (b) 2-D and 3-D paths in CAMERA orientation (contra on the LEFT), sign labels
  + = contralesional; (c) convert slide 24's figure and its code to + = contralesional (make it the default) -> helper.

## DECISIONS NEEDED FROM PRIYA (newest first; 1-9 answered above)

1. Grant copy (path) — to check the 2-photon slides against the grant figures and add any missing panels.
2. Fluoroscopy: cohort-4 stroke date; meaning of "post-stroke1"; `20261029/` holds files named `20261028_*` that
   differ from `20261028/` (likely the 10-29 session mis-dated) — confirm.
3. Deck wording to check: Big picture first bullet ("still aimed ipsilesionally" instead of "launch stays ipsilesional");
   slide 6 "PS94 below the pooled curve" (read off the plot); small text on a few hidden supplements.
4. Imaging page v48: the helper also repointed the corrections list's "use the 10-15 values" to 11-02 (outside
   section 3) — keep or revert.
5. 2-photon hidden slides (108-110, 113): (a) the grant's 'ipsi / contra' lick-direction labels are named relative to the
   PLANNED lesion side (ipsi = rig L, contra = rig R; grant_cd_proj.py), not derived from the imaged hemisphere -- confirm;
   (b) LB333 axon-GCaMP slide has no construct / injection / target in any doc -- keep only with that context;
   (c) map: used the per-tile lick-direction map (FDR, deduplicated; 131 / 231 / 3,390); the map_sel_L_vs_R_GRANT
   versions are an older uncorrected test (172 / 249 of 4,235) and disagree -- OK?
6. 4-8 Hz laterality band as the primary quiet-event synchronisation measure? Recommendation: NO (keeps 57 % of the
   8-15 Hz effect, q .11, two animals, limb shifts equally, 415 nm asymmetric in the same band). Also: word "a pre-cue
   event marks a weaker code only after the stroke" as resting on the main + exclusion tests (two pre-stroke sub-tests
   are negative too).
7. STATE DECODER (new 11-03): under the production z-scored decoder PS94 loses its RUNNING class (.98 -> .54 / .49
   chronic), not licking (that was the unscaled decoder). PS94 runs ~4x more post-stroke, which pulls the plain
   session z-score. Adopt class-balanced session centring (new frozen spec id; restores PS94 running .97 / .96 / .91;
   others move <= .08 up) or keep the plain z-score and report it as a scaling sensitivity?
8. Deck slide 13 bullets changed tonight to match the production decoder ("running ~94 %"; "state loss is mostly PS94's
   running class") -- OK? Imaging page v49 also corrected the same claim outside section 3 (status row, corrections
   list) -- keep or revert.
9. Kinematics (+ = contralesional, page v18): (a) deck l_cue / l_phase pictures use prepost_alltrials (no engagement
   gate), page uses prepost (engaged) -- which should the deck show? (b) 2-D path figures are now mirrored left-right
   vs the camera view -- OK, or keep camera orientation and label the sign? (c) slide 24 caption still says "executed
   peak angle (+ = ipsi)" (orofacial per-spout figure, not part of the conversion) -- convert that figure too?

## PLAN TONIGHT (status)

- [x] 1. (DONE, pushed d74ab907, 225 tests passed) Push blocked by 21 ordering-test failures in scripts/cohort2_qc -> switched to analysis_kit.fan_sessions
      (sorted; outputs only gain determinism), test 144 passed; commit + push in progress.
- [x] 2. (DONE 11-03, commits 67acffc0 / 1bef3f98) Re-check "running alone ~97 %" after the state z-score fix (specificity slide + imaging page section 3).
- [x] 3. (DONE 11-03: imaging page v49 Fig 3.5; deck hidden slide 14) PS94 state-diagnostic figure (confusion grids frozen / z-scored / within-session refit) -> imaging page
      section 3 + hidden slide after "Acutely, position fails far more than behavioural state".
- [x] 4. (DONE: hidden slides 108-110 after the six-spout CD slide, 113 after ROICaT; add_twop_extra.py) Remaining 2-photon grant panels (2p-RAM/grant_figs: six-way decoding PS110, lick-direction selectivity, field
      maps) as hidden slides after the 2-photon main slides.
- [x] 5. (DONE 11-03: kinematics page v18, commits 5371c420 / d54a5246; deck pictures swapped) Kinematics page sign conversion (+ = contralesional) for section 2 angle / lateral rows, 3-D midline-angle
      figures (3.2-3.5, 3.7, 9.x), 2-D / 3-D paths (section 4); republish; swap the deck copies.
- [x] 6. (DONE 11-03, encoding page v312) Quiet events: T13 follow-up re-run on LP events; exploratory 4-8 Hz laterality (PROVISIONAL) -> encoding page
      section 23 (re-read live page first, only section 23 + its status row; tell sabatinilab-34 the version).

## RESULTS LOG (newest last)

- 11-02 ~23:50 State decoder per-session z-score fix done (commits abda8c9c, b12c533e): state retained .97 / .90 / .87
  / .87 (was .98 / .87 / .82 / .84), position unchanged; specificity ratio is an ACUTE claim (~8x), chronic losses
  similar (~.1; state part mostly PS94's licking class). Deck slide 13 retitled "Acutely, position fails far more than
  behavioural state"; imaging page v48 section 3 corrected.
- 11-03 ~00:30 2-photon grant extras added as hidden slides (deck 129 slides): six-way decoding PS110 09-25 (CD argmax
  decoder: pre-cue .72, cue .67, lick .95, chance .17, p = .005 every window; the grant TEXT quotes the multinomial
  decoder .81 / .74 / .98 -- both in the notes); lick-direction selective 362 / 3,752 (9.6 %, FDR .05; 131 ipsi, 231
  contra); per-tile map (no clear cluster, by eye); LB333 axon GCaMP 150 / 250 um. Pushed with this entry.
- 11-03 ~00:50 Priya: add the other window's jaw-only / incomplete-lick trial classes (scripts/jaw_only_incomplete_figs.py,
  DeepLabCut/Widefield/session_poses/cam1_licks/trial_classes/): deck slide 26 MAIN 'Acutely, far-spout trials end in
  incomplete or jaw-only moves' (far contra incomplete +.63 acute 4/4, +.16 subacute, ~0 chronic; jaw-only +.27 far
  contra, +.23 far mid, +.11 far ipsi acutely, gone chronically), hidden 27-29 (jaw / tongue traces by class, over-days
  model). add_trial_classes.py + fix_trial_classes.py. Not on a results page yet (the other window's figures).
- 11-03 ~01:30 Quiet events (agent): (a) T13 follow-up on LP-defined events UNCHANGED (acute + subacute after removing event
  frames -0.016 vs DAQ -0.016, p .001; per animal PS92 -.003 / PS93 -.001 / PS94 -.030 / PS95 -.036; neutral fill 3/4, was 4/4;
  caveat: two pre-stroke sub-tests also negative). (b) 4-8 Hz T20 laterality keeps 57 % of the 8-15 Hz ipsi > contra (+0.080, q .11;
  PS93 / PS95 only), 4-7 Hz 41 %; limb shifts equally; 415 nm 4-8 Hz power lower ipsilesionally -> not artefact-free; withdrawal
  stands; recommend NOT adopting 4-8 Hz as primary (decision 5). Encoding page v312 (Figs 23.23, 23.24; inline images, artifact
  at its 512-file limit). Commit 4452139a (+ running-list commit). No conclusion changed.
- 11-03 ~02:30 Running alone + PS94 state (agent): running-class recall pooled .98 / .92 / .83 / .89 after the z-fix
  (was .99 / .95 / .86 / .92), acute keeps ~94 %; boot_delta -.04 [-.12, .01] / -.12 [-.34, .02] / -.12 [-.37, .02],
  none excludes 0. CORRECTION of the 11-02 entry above: under the production z-scored decoder PS94 loses RUNNING
  (.98 / .84 / .54 / .49; chronic running read as quiet .29, licking .24), licking holds (.99 / 1.00 / .94 / .90);
  within-session refit .98-.99 every epoch (definitions fine). Likely cause: PS94 runs ~10 % -> ~40 % of segments and
  ~7 -> ~15 mm/s post-stroke, pulling the all-segment session mean; class-balanced centring restores .97 / .96 / .91
  (diagnostic only, decision 7). 13n / 13pos captions now measured (epoch_state._acute_line + test). Imaging page v49
  (Fig 3.5). Deck: hidden slide 14 "PS94's states stay separable; frozen decoder loses running"; slide 13 bullets fixed
  (decision 8).
- 11-03 ~02:30 Kinematics page v18: every lesion-signed figure + = contralesional (plot-level flip per animal; stats
  unchanged: cluster bars identical, bootstrap replayed to 7e-15). Deck pictures swapped on slides 33-36 (l_phase_all,
  p_paths, p_phase, p_3d_far_rows123) and backups 123-124 (l_cue excl / incl); sign text on those slides updated
  (far contra acute now -50 to -75 px, -30 to -50 deg lateral / angle; 3-D midline -30 to -40 deg acute). Script
  apply_overnight_specs.py; all eight changed slides checked by PowerPoint export. Deck 134 slides. Decision 9.
- 11-03 ~09:00 Deck audit helper DONE (deck 138 slides; docs/status/DECK_AUDIT_2026-11-03.md, DECK_TITLES_2026-11-03.csv;
  commit 1eec2abe): corrected results swapped, LP lick-source lines in 64 notes, mCherry sheets (113 NES PS110, 114 H2B
  PS104), all-engaged trial classes 31 / 32, VFSS dates, 2pRAM / VM-thalamus context, short titles. Then
  recrop_trial_classes_1103.py: sabatinilab-e3 re-rendered every trial-class figure from ONE all-engaged class table
  (07385b37) -> slides 27-32 re-cropped; slide 27 now incomplete far contra +.57 / +.17, jaw-only +.21 / +.19 / +.09;
  jaw-moved caveat removed. Open for Priya: slide 31 as THE main trial-class slide?; RESTW vs REST lead; PS110 mCherry
  count (559 / 3,804 vs refined 891 / 4,276); Big-picture quiet bullet qualifier; imaging page section 15 frozen at ~5x.
- 11-03 ~11:30 State decoder x running definition (decision 7; labcams/state_decoder_ps94_check/running_def/,
  DECISIONS 783c04b2): encoder v7 running regressor = forward hinge from 1.5 mm/s, cap at each animal's pre p99.
  The kinematic class rule (running > 1.5 mm/s) helps PS94 running (chronic .49 -> .78) but costs PS92 .05-.10 balanced
  accuracy; the cap shows PS94's lost running is faster than any pre-stroke running (20-67 % of its post segments).
  Class-balanced centring fixes PS94 best (.97 / .96 / .91) and worsens no animal by > .01; state retained .98 -> .94 /
  .95 / .94. Position still falls 6-15x further acutely under every arm. Recommendation: keep the class rule; adopt
  class-balanced centring as a new frozen spec (needs Priya).
