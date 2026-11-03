# Running list — overnight 2026-11-02 → 11-03 (Priya: "go ahead with all that you can without me overnight. keep handoff updated and a running list of results and any decisions you need from me")

Handoff: `docs/status/STATUS_2026-11-01_QUIET_EVENTS_HANDOFF.md` (top block "PI DECK STATE"). Deck:
`M:/MICROSCOPE/Priya/Widefield/presentations/PI_overview_2026-11-02.pptx` (edit by slide TITLE via backed-up run-once
scripts in `_source_PI_overview_2026-11-02/`; one deck writer at a time).

## DECISIONS NEEDED FROM PRIYA (newest first)

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
6. 4-8 Hz laterality band as the primary quiet-event synchronisation measure? (exploratory run tonight, item D).

## PLAN TONIGHT (status)

- [x] 1. (DONE, pushed d74ab907, 225 tests passed) Push blocked by 21 ordering-test failures in scripts/cohort2_qc -> switched to analysis_kit.fan_sessions
      (sorted; outputs only gain determinism), test 144 passed; commit + push in progress.
- [ ] 2. (running) Re-check "running alone ~97 %" after the state z-score fix (specificity slide + imaging page section 3).
- [ ] 3. (running) PS94 state-diagnostic figure (confusion grids frozen / z-scored / within-session refit) -> imaging page
      section 3 + hidden slide after "Acutely, position fails far more than behavioural state".
- [x] 4. (DONE: hidden slides 108-110 after the six-spout CD slide, 113 after ROICaT; add_twop_extra.py) Remaining 2-photon grant panels (2p-RAM/grant_figs: six-way decoding PS110, lick-direction selectivity, field
      maps) as hidden slides after the 2-photon main slides.
- [ ] 5. (running) Kinematics page sign conversion (+ = contralesional) for section 2 angle / lateral rows, 3-D midline-angle
      figures (3.2-3.5, 3.7, 9.x), 2-D / 3-D paths (section 4); republish; swap the deck copies.
- [ ] 6. (running) Quiet events: T13 follow-up re-run on LP events; exploratory 4-8 Hz laterality (PROVISIONAL) -> encoding page
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
