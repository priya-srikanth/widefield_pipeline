# Archived probes — how the analysis decisions were actually made

**These are evidence, not dead code.** Each module here answered ONE question, and the
answer is now baked into the pipeline as a constant, a default, or a method that looks
arbitrary until you know what the alternative did. `DECISIONS.md` records most of the
conclusions; this directory is the working that produced them.

Priya, 2026-09-21: *"anything that is a prior test that guided the current analysis can be
archived (as evidence of how we made these decisions), anything that is a test that was
cosmetic or did not end up contributing to the current analysis pipeline decisions can be
removed"*. **Nothing was removed.** Every module that was not still live was moved here, on
the grounds that the cost of keeping a 100-line probe is a directory entry and the cost of
deleting one that turns out to have mattered is a re-derivation nobody knows is needed.

## What stayed in `scripts/rest_migration/`

A module is LIVE if any of these holds, and the third one is the trap:

1. `wfield_local/` or `tests/` imports it;
2. a runbook or `CLAUDE.md` names it as something to run;
3. **it RENDERS a figure the deck registers** — even though nothing imports it.
4. it is an operational UTILITY someone runs by hand (`publish_cache`, the `regen_*` pair,
   `fix_mask_names`). Those are tools, not probes, and filing them under "evidence" would
   have made the cache-publishing step harder to find than it already is.

Criterion 3 caught a real mistake during this very pass: the first cut archived
`reference_family_figure`, which nothing imports and which draws `epoch_15k` — a figure the
deck places. Every test would have stayed green while the nightly lost the ability to
regenerate a published figure. **If you add a module here, check what it writes, not just
who calls it.**

## Cited in `DECISIONS.md` (20)

The conclusion is written up; this is the code that produced it.

| module | lines | cites | last touched | the question it answered |
|---|---:|---:|---|---|
| `timelocal_needed` | 237 | 3 |  | Is the TIME-LOCAL rest baseline capturing real drift, or fitting noise? |
| `plot_drift_estimators` | 202 | 2 |  | What each DRIFT ESTIMATOR does, end to end: trend, pre-subtraction residual, and final signal. |
| `precue_window_sweep` | 141 | 2 |  | Pre-cue window sweep ON THE PIPELINE WE ACTUALLY RUN: meegkit_hpfit + LocaNMF, curated set. |
| `restw_smoke` | 127 | 2 |  | Smoke-test `restw` on real sessions BEFORE any render depends on it. |
| `rolling_detrend` | 126 | 2 |  | ROLLING masked-median detrend -- the smooth version of `filter_acausality_test.detrend_masked`. |
| `stopped_tail_contamination` | 127 | 2 |  | Does the STOPPED TAIL distort the drift fit inside the WORKING period? |
| `worktrunc_result_impact` | 286 | 2 |  | THE DECIDING MEASUREMENT: does stopped-tail contamination move a WITHIN-WORKING RESULT? |
| `block_order` | 132 | 1 |  | Is BLOCK ORDER systematic across sessions? The precondition for reading the amplitude gradient. |
| `cross_15k_15r` | 182 | 1 |  | Do `15k`'s all-family regions sit UNDER `15r`'s significant blobs? The spatial cross-check. |
| `cutoff_scaling` | 125 | 1 |  | Does the polynomial's 50% cutoff really scale as duration/order? MEASURED, not assumed. |
| `find_stopped_sessions` | 100 | 1 |  | Rank POST-STROKE sessions by having a STOPPED CHUNK -- the animal quits well before the recording ends -- and by how far the raw fluorescence moves ac |
| `lick_arm_raw_counts` | 116 | 1 |  | RAW per-position trial counts entering the LICK-arm fit — is far-contra 0, or thin-but-present? |
| `onset_edge_bias` | 102 | 1 |  | How badly does each drift estimator handle the EARLY BLEACHING ONSET? |
| `plot_rest_decode` | 148 | 1 |  | PICTURE of the rest-position decode -- the diagnostics were all text-only until now. |
| `rest_vs_restw` | 175 | 1 |  | Does POSITION-WEIGHTING the rest baseline change any CONCLUSION, or only the third decimal? |
| `restw_reliability` | 176 | 1 |  | SPLIT-HALF RELIABILITY of `rest` vs `restw` -- is equal weighting normalising to thin data? |
| `settling_transient` | 135 | 1 |  | How long does the RECORDING-ONSET transient last, cohort-wide? Measure it, do not assume 30 s. |
| `state_decoder_by_time` | 183 | 1 |  | Does the frozen STATE decoder read BEHAVIOUR, or does it read WHEN IN THE SESSION? |
| `trial_floor_sweep` | 176 | 1 |  | WHAT WOULD `MIN_TRIALS_PER_CLASS = 10` BUY, AND WHAT WOULD IT COST? |
| `why_no_rest` | 136 | 1 |  | WHICH TERM starves a session's rest baseline? Decompose the mask, do not guess. |

## Not cited by name (11)

**Not the same as "did not matter."** `DECISIONS.md` records findings in prose and rarely
names the script that produced them, so absence here is weak evidence. Read the docstring:
most of these are plainly decision probes — whether the ±3 s treadmill buffer earns its
cost, whether the time-local rest reference differs from the flat one — whose answers are
in the pipeline's defaults today. They are archived on the same terms as the rest.

| module | lines | last touched | the question it answered |
|---|---:|---|---|
| `bout_durations` | 156 |  | REST BOUT DURATIONS per candidate definition -- the state decoder's window length depends on it. |
| `docked_rest_check` | 91 |  | Does the DOCKED rest term produce a usable mask, and how much rest does it cost? |
| `residual_working_vs_stopped` | 120 |  | How much SLOW structure survives the drift correction, WORKING period vs STOPPED period. |
| `rest_smoke` | 75 |  | Smoke-test the REST mask on real sessions, against the retired definition and the predictions. |
| `restw_bin_coverage` | 115 |  | Does `restw`'s PER-POSITION time-local baseline have enough frames per bin to be estimated? |
| `sweep_structure` | 138 |  | Does the task really run a PERMUTED SWEEP of all six positions, and can we bin rest by sweep? |
| `timelocal_smoke` | 92 |  | Does the TIME-LOCAL rest reference actually differ from the flat one, and by how much? |
| `treadmill_buffer_sweep` | 346 |  | Is the +-3 s treadmill buffer earning its cost, or is `speed < 1 mm/s` already doing the work? |
| `trial_anchors` | 106 |  | Where the trial's boundaries actually are, so the REST window can be anchored on a real event. |
| `variant_manifest_survey` | 64 |  | Survey the hemo-variant manifests the LocaNMF/ROI arm actually reads. |
| `what_locanmf_reads` | 59 |  | Print EXACTLY what the LocaNMF / Allen-ROI arm reads, resolved through the real config. |

