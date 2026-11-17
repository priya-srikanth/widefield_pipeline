## 2026-11-04 code review decisions (Priya)

**(e) One DAQ lick detector: the BEHAVIOR profile everywhere (Priya 11-04, late).** That is `configs/defaults.yaml
lick_detection` (2.5 / 0.5 V, 40 ms min ILI). Imaging's 2.5 / 1.0 V + 0.10 s refractory literal is retired. Evidence:
docs/status/LICK_DETECTOR_COMPARE_2026-11-04.md, 16 sessions.
- The offset threshold changes 0 onsets.
- The refractory drops 0.58 % of licks, 94 % of them real camera licks at 10-11 Hz, and unevenly (PS95 1.4 %).
- 0 hits and 0 imaging RTs change.

Implementation is with sabatinilab-34 in the cohort-1 rebuild (sent 11-04).

Source: `docs/status/CODE_REVIEW_2026-11-04.md`, "Decisions that need you".

**(a) Epoch figures: LEVEL bars and CHANGE panels keep their different weighting, and the figures say so.** LEVEL bars
stay pooled flat over sessions (`analysis_kit.boot_ci`). CHANGE panels stay animal-weighted (`analysis_kit.boot_delta`).
This is CLAUDE.md rule 9 and is deliberate. The difference will be LABELED in each panel's subtitle and caption, so a
reader can see which weighting a number uses. Implementation waits for the cohort 1 rebuild to merge, because
`epoch_figures.py` is changing on the rebuild branches.

**(b) Encoder ENL scoring window follows the lick source: LP 1.0 s, DAQ 2.0 s.** Already decided 11-04 and implemented
on the rebuild branch (`rebuild/encoder-classes`; cohort 1 `combined_encoding` `enl_spec`). Until that merges, main
still hard-codes 2.0 s (`wfield_local/combined_encoding.py`, the `win["enl"]` line, `c - 2.0`).

**(c) Approved: delete the orphan modules and the ~26 unregistered superseded epoch PNGs.** The PNG list is kept in
`decoder_mix_checks/DELETED_EPOCH_PNGS_2026-11-04.csv`. These are derived outputs, not original data (ground rule 0).

**(d) Imaging vs behavior lick detector: under measurement.** The comparison is in
`docs/status/LICK_DETECTOR_COMPARE_2026-11-04.md` (pending). No change to either detector until it is read.

### Path fixes for DECISIONS.md (path text only)

Verified on main 2026-11-04: these script paths cited in DECISIONS.md moved to `scripts/rest_migration/archive/`:

block_order, cross_15k_15r, lick_arm_raw_counts, plot_drift_estimators, plot_rest_decode, precue_window_sweep,
rest_vs_restw, restw_smoke, rolling_detrend, settling_transient, state_decoder_by_time, stopped_tail_contamination,
trial_floor_sweep, why_no_rest, worktrunc_result_impact
(`scripts/rest_migration/<name>.py` -> `scripts/rest_migration/archive/<name>.py`).

Not on main: `scripts/lick_cued_aim_by_spout.py` (exists on `rebuild/cohort1-20261103`; leave as is),
`scripts/trial_class_all_engaged.py` (deleted; folded into `scripts/jaw_only_incomplete_figs.py`'s all-engaged table,
commit 0e657df2).

## (f) Priya 11-05: decisions after the re-analysis sweep
1. State discriminators (camera-class question 8): train working (contact + incomplete) vs no_response; STOPPED trials
   are scored as their own class, never trained on (stopped runs sit at session end, so training on them mixes time in
   session into the label). jaw_only and after_end stay scored-only.
2. Hit-rate period (question 3): the CAMERA working period (first cue -> last camera lick, minus stopped runs) is
   primary; the DAQ quit-tail rule is a labeled robustness row. Changes Fig 0.2 and the cohort_session_metrics readers.
3. Unjudged close spouts (close_center / close_mid): the side cameras (cam2 / cam3) judge the jaw there once the
   round-2 side model and whole-session side tracking exist; until then they stay a separate unjudged label.
4. Imaging-box lick-aligned products (lick_aligned_affine8v1, locanmf_lick_aligned_affine8v1/*.npz; made with the old
   0.10 s detector, no detector tag): kept as LABELED LEGACY. They feed only the preprocessing deck's per-session
   lick-aligned slides and retired LocaNMF lick pool / cards / contralateral modules; current analyses import only
   locanmf_lick_aligned's frame-timing helpers, not its outputs. Regenerate a product only when a current page / deck
   figure needs it.
5. Cohort 2 follows the DAQ behavior lick detector too (one detector everywhere).
6. recovery_timing keeps its animal-weighted level bootstrap, as an explicit labeled option in analysis_kit (no change
   to the 10-25 figures).
7. Per-animal trend-line panels in orofacial_cohort.fig_epochs go to a collapsible section, not the main page.
8. ITI pattern-stat scripts (iti_decoder_pattern_stats and the spout / area / session-map tables, 34's 98ca70ff): switch
   the local bootstrap p (2*min + 1/N) to the shared ak.boot_p (max(2*min, 1/N)), with ak.holm / ak.bh. Post-merge.
   Effect measured 11-05 (decoder_mix_checks/iti_star_shared_stats_1105/): stars only added, none lost; 22 / 4480 area
   rows; outlines +0.1-3 pp (largest cued far_contra acute 6.7 -> 9.6 %); session change maps 3.67-3.69 change in
   26 / 80 cells (up to +2.4 pp), captions rewritten after the re-render; maplevel 3.66 unchanged.

### (f) 1 / 2 and the 11-04 ENL slide: implemented 11-05 on branch fix/decisions-f1-f2-enl-20261105 (not merged)
- (f) 1 (a49795b3): `precue_engagement_states` under the camera gate trains WORKING (contact + incomplete, both DAQ arms)
  vs NO_RESPONSE; stopped / stopped_licking / jaw_only / after_end / unknown are scored per class only. New frozen id:
  align `<align>-disc-cls-work-v-noresp`. Section G's G4 camera-class discriminator (11-04) already met the decision
  (stopped / quit / stopped_licking are their own groups); pinned by a test, unchanged. OPEN: G4's negative class
  "none" also holds "unknown" rows; the precue discriminator keeps its late-window + SPATIAL_MISS design.
- (f) 2 (ae051fc0): Fig 0.2 = the camera working-period hit rate (run's trial_attempts_sessions.csv `frac_hit_<pos>`:
  contact / working-period trials); the DAQ quit-tail hit rate is `behaviour_hits_by_spout_daq_quittail.png`
  (ROBUSTNESS). The other cohort_session_metrics readers are left on DAQ: epoch_audit / epoch_behaviour timecourse
  (epoch boundaries stay pinned, Q11), engagement_by_epoch (a measure OF engagement). OPEN: epoch family 1b
  (`epoch_behaviour.fig_behaviour`) is the same quantity on DAQ engaged trials.
- ENL slide (8f4399d2): `enl_window.sliding_lags` (the decoders' search + guard strobe + 0.3 s, inside the span) in
  lick_enl_angle_decode ("slide" primary, "2s" robustness; new enl_angle / enl_spout ids `enl_w<W>_lfslide`),
  lick_enl_target_test, lick_enl_offset_explore (prelaunch_aim_tracking's B7 reference follows), revup_neural (pre-cue
  level), fr_enl_decode. Nothing re-run.
- Behavior-only figures on the camera period (802c3d54): `trial_classes.camera_quit` (A6) in disengagement_tests
  (DAQ quit = `DISENGAGEMENT_QUIT=daq` robustness row); revup_behaviour reads the run's attempts table.
