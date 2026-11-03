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
- 11-03 PM Priya: (3) RESTW leads the maps (deck already RESTW; imaging page to lead with RESTW after the figure-rules
  helper publishes); (6) oblique 3-D views relabelled not mirrored = OK (contra image-left matches the cameras); (8) 11-02
  mirrored path files DELETED (13 derived files, each with a *_contra_cam replacement and no deck reference: 6
  cohort_*trajectory*_contra.png, pose3d_paths_pooled{,_shared}_contra.png, 4 spout_error scatter_epoch_PS9x_contra.png,
  deck crop p_paths_shared_contra.png; list in DECISIONS 2026-11-03 camera-view entry). Asked to SEE: slide 27 vs 31
  (trial classes), slide 36 (path axis), PS110 strict vs refined mCherry overlays; explain class-balanced centring and the
  Big-picture quiet qualifier.
- 11-03 PM deck: apply_figure_rules_1103.py (spec 1: slides 24, 33, 57, 100; spec 2: 56, 58, 10, 19 -- session dots,
  epoch colours, stars no longer overlapping), trial_class_main_1103.py (Priya: slide 27 'Far contra: contact .95 -> .05'
  = THE main trial-class slide, the jaw-only / incomplete change slide now hidden at 29; Big-picture quiet bullet: "after
  the stroke it goes with a weaker pre-cue position code (a smaller version may be present at baseline)"). Priya: path
  axis + = contra increasing leftward OK. PAGES BLOCKED: this Claude Code session is now signed in to a different
  claude.ai account (psrikanth87@gmail.com); the four results pages are not visible to it, so the figure-rules page
  publishes (kinematics, FR, imaging incl. RESTW lead, encoding sec 23) wait for the owning account. Mapping: docs/status/
  FIGURE_RULES_AUDIT_2026-11-03.md. Running: pre-weighted (label-free) centring test for the state decoder.
- 11-03 late: e3 consolidation A1 + A2 on main (ef70a2a8): hit rate on each session's 3.5 s window (+0.016 mean, acute
  +0.027; quit later in 7/120 sessions); jaw-without-lick = trial_classes jaw_only share of ALL trials. Deck numbers that
  move after re-render: behaviour hit-rate slide, kinematics Fig 1.3 / 7.6 derived slides (24, 25, attempts slide). A3
  (animal-weighted change-vs-pre in epoch_figures / epoch_behaviour / matrix_bootstrap) coded, not on main; e3 will send
  the list of figures whose stars change. Priya: build bal_em min-mass fallback (test 2 / 5 / 10 %; default only if it
  clearly helps) -> bal_em helper. Mix-invariance test (1f33189e): no normalisation is both mix-invariant and right on
  PS94; keep bal_em.
- 11-03 late: decoder mix checks DONE (fa70b651; labcams/decoder_mix_checks/). SPOUT frozen decoder IS biased by the
  class mix of its per-session z-score set (engaged trials; far-contra share .16 pre -> .00-.03 acute): production
  UNDERSTATES the acute far-contra drop by ~.10 (cue/working: -.536 prod vs -.641 bal_em, 4/4 animals) and overstates
  near-position losses by ~.05-.07; overall accuracy moves <= .04. "Far contra falls most" stands and gets stronger.
  Fix proposed: bal_em centring on the engaged set as a new frozen spec (old models kept) -- AWAITING PRIYA (and the
  min-mass guard decision). LICK-LANDING decoder: clean (frame-SD over all frames, no lick-sample centring; artefact
  +.003 mm vs +.36 mm offset). Caveat: cued ALL-lick acute offset (+.17) loses ~1/3 when the 1.06-1.20x frame-SD rise
  is undone (CI then includes 0, PS95 flips); ITI and first-cued-lick offsets stand. Tell sabatinilab-34 after Priya.
  PS110 mCherry: the refined 891 / 4,276 (20.8 %) call file was built on the MOSAIC detection; the cited per-tile
  detection has only the strict 559 / 3,804 (14.7 %); refined per-tile never run -> asked Priya.
- 11-03 late: Priya: lick-landing decoder (the more important analysis) unaffected -- caveat sent to 34. Spout decoder:
  wants a DIFFERENT guard than the state decoder's (training mix ~1/6 there = no-op); asked whether adding jaw-only /
  incomplete (attempt) trials to the normalisation set helps -> helper running decoder_mix_checks/spout_guard/ (arms P,
  bal_em, G1 pre-geometry guard 2/5/10 %, G2 drop-absent, ATT engaged+attempt [z and bal_em], ALL). Adoption still
  Priya's call after that. PS110: Priya approved refined call on the PER-TILE detection -> helper running (2pRAM repo,
  new files only; reports consumers of the mosaic-based rev74c59ab file).
- 11-03 ~10:50 PI deck: e3 11-03 re-renders swapped in (presentations/_source_PI_overview_2026-11-02/recrop_e3_rerender_1103.py, backup_before_recrop_e3_rerender_1103.pptx): slides 24 (fig:k_metrics_4panel, fresh plain cohort_metrics_minus_pre_contra.png, byte-identical to the 07:55 _nolines render, so no visible change), 25 (fig:k_acc_pos, accuracy_contra 10:27; jaw-without-lick row changed by A2), 27-28 (tc_share_all*, 10:30), 29-32 (trial_classes figures, 09:30); new crops *_1103b.png, provenance updated. Numbers re-read from tc_share_all*.csv / trial_classes tests.csv + sessions.csv / prepost position_metrics.csv: far-contra acute no response .13 -> .12, jaw-only far contra / far mid +.21 / +.19 -> +.22 / +.18, CIs and per-animal values updated, unknown 4 -> 1 of 50,929. Left for decision: slide 29 notes still say jaw-only close mid +.04 ** (now 0: close-mid jaw not judged after A4) and far ipsi subacute +.03 * (now n.s., p .053); notes 27-32 define no_response as 'no jaw movement' (close mid is now no lick, jaw not judged). Slide 26 not touched. Slide 115 notes: PS110 09-25 sentence now gives the per-tile refined call of record 969 / 3,752 = 25.8 % (strict 559 / 3,804 beside it); 09-22 strict + refined 1470 / 4,081 = 36.0 % (fix_115_ps110_call_1103.py). PNG check of 24-32 OK.
- 11-03 late: PS110 09-25 per-tile refined call = call of record (Priya): 969 / 3,752 (25.8 %) after dedupe; strict 559 /
  3,804. 2pRAM c59b16a (routing refuses mismatched call / match files + test), c9a3c8d (register / DECISIONS numbers);
  all 09-25 outputs re-run per tile, 63 mosaic files moved to analysis/superseded/. "817" = mosaic call mapped onto per-
  tile cells. Deck: swap_ps110_pertile_1103.py -> slide 113 (gallery 984 / 823) + 119 (outline 969), 115 notes; PNG
  checked. fix_tc_notes_1103.py: slide 29 / 27-32 / 24 notes that no longer held after A4. NEW (per tile): mCh+ cells
  less lick-responsive (all bouts 26.0 vs 30.0 %, p .017; spontaneous 21.1 vs 26.4 %, p .0009; uncorrected Fisher);
  position contrasts unchanged. Spout decoder: G1 not adopted (Priya: selective pre-structure); camera-lick normalisation
  set proposed. A3 single-session rule: recommended option A (animal-weighted + LMM beside), awaiting Priya.
- 11-03 late: bal_em production state decoder DONE (9547a773 code, af638910 docs; spec ids PS92 96f8d3543d94 / PS93
  084308f266df / PS94 8bfbb26d78de / PS95 57866a0b31e3, published byte-verified; session_z models kept). Min-mass
  fallback built, OFF (does not cut the absent-state drop). 13s family re-rendered (session_z outputs ->
  _superseded_session_z_20261103/). Deck: slides 13-14 swapped (state_balem_swaps.json via apply_figure_rules_1103.py,
  note prefix now argv[2]); PNG checked. Page edits: docs/status/STATE_BALEM_PAGE_EDITS_2026-11-03.md (publish blocked).
  Open: 13pos hard-coded "carried mostly by one animal" in epoch_state.py is stale; 4 interim unpublished state models
  parked in C:/Users/SabatiniLab/frozen_models_interim_unpublished_20261103/ (delete? Priya). Slide 9: 5b's camera-
  working section-1 numbers (s1_refit_basis.csv split_camwork_*) -> new figure fig_decoding_sessions_camwork_1103.py
  + slide9_camwork_1103.py; ranges .44-.88 / .27-.57, acute post-cue drop 4/4; PNG checked.
- 11-03 late: slide 9 moved to the LAST-CAMERA-LICK cutoff (Priya primary; 5b 91664c86): split_campllick_*, ranges .43-.89 / .26-.56, directions + counts unchanged (fig_decoding_sessions_campllick_1103.py, slide9_campllick_1103.py; PNG checked). A3 rule under discussion between Priya and e3 (per-contrast slope df 3 / pooled epoch / heterogeneous-spread df 3); slide star text held. Spout bal_em (arm B, all alignments) build running; renders wait for A3 on main.
- 11-03 late: spout-position frozen decoder bal_em is PRODUCTION (99be6122 on main; 31 models published + SHA-verified; WIDEFIELD_SPOUT_SCALING=session_z restores the old ones). Verified: production scoring = diagnostic arm B (0/4164 cells differ); acute cue far contra -.536 -> -.644, close spouts +.05-.06 (close ipsi CI now includes 0); overall moves <= .03. NO re-renders yet (wait for A3 LMM rule on main); deck swap spec build_inputs/spout_balem_swaps.json (slides 10-13, 19; 89-93 check), page edits docs/status/SPOUT_BALEM_PAGE_EDITS_2026-11-03.md. Any nightly render from now scores with bal_em.
