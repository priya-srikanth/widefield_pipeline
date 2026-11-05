## 2026-11-04 code review decisions (Priya)

**(e) One DAQ lick detector: the BEHAVIOUR profile everywhere (Priya 11-04, late).** That is `configs/defaults.yaml
lick_detection` (2.5 / 0.5 V, 40 ms min ILI). Imaging's 2.5 / 1.0 V + 0.10 s refractory literal is retired. Evidence:
docs/status/LICK_DETECTOR_COMPARE_2026-11-04.md, 16 sessions.
- The offset threshold changes 0 onsets.
- The refractory drops 0.58 % of licks, 94 % of them real camera licks at 10-11 Hz, and unevenly (PS95 1.4 %).
- 0 hits and 0 imaging RTs change.

Implementation is with sabatinilab-34 in the cohort-1 rebuild (sent 11-04).

Source: `docs/status/CODE_REVIEW_2026-11-04.md`, "Decisions that need you".

**(a) Epoch figures: LEVEL bars and CHANGE panels keep their different weighting, and the figures say so.** LEVEL bars
stay pooled flat over sessions (`analysis_kit.boot_ci`). CHANGE panels stay animal-weighted (`analysis_kit.boot_delta`).
This is CLAUDE.md rule 9 and is deliberate. The difference will be LABELLED in each panel's subtitle and caption, so a
reader can see which weighting a number uses. Implementation waits for the cohort 1 rebuild to merge, because
`epoch_figures.py` is changing on the rebuild branches.

**(b) Encoder ENL scoring window follows the lick source: LP 1.0 s, DAQ 2.0 s.** Already decided 11-04 and implemented
on the rebuild branch (`rebuild/encoder-classes`; cohort 1 `combined_encoding` `enl_spec`). Until that merges, main
still hard-codes 2.0 s (`wfield_local/combined_encoding.py`, the `win["enl"]` line, `c - 2.0`).

**(c) Approved: delete the orphan modules and the ~26 unregistered superseded epoch PNGs.** The PNG list is kept in
`decoder_mix_checks/DELETED_EPOCH_PNGS_2026-11-04.csv`. These are derived outputs, not original data (ground rule 0).

**(d) Imaging vs behaviour lick detector: under measurement.** The comparison is in
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
   primary; the DAQ quit-tail rule is a labelled robustness row. Changes Fig 0.2 and the cohort_session_metrics readers.
3. Unjudged close spouts (close_center / close_mid): the side cameras (cam2 / cam3) judge the jaw there once the
   round-2 side model and whole-session side tracking exist; until then they stay a separate unjudged label.
4. Imaging-box lick-aligned products (lick_aligned_affine8v1, locanmf_lick_aligned_affine8v1/*.npz; made with the old
   0.10 s detector, no detector tag): kept as LABELLED LEGACY. They feed only the preprocessing deck's per-session
   lick-aligned slides and retired LocaNMF lick pool / cards / contralateral modules; current analyses import only
   locanmf_lick_aligned's frame-timing helpers, not its outputs. Regenerate a product only when a current page / deck
   figure needs it.
5. Cohort 2 follows the DAQ behaviour lick detector too (one detector everywhere).
6. recovery_timing keeps its animal-weighted level bootstrap, as an explicit labelled option in analysis_kit (no change
   to the 10-25 figures).
7. Per-animal trend-line panels in orofacial_cohort.fig_epochs go to a collapsible section, not the main page.
