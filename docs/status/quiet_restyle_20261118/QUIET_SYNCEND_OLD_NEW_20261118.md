# Quiet-period events: analyzed time ends at the last sync pulse + bootstrap restyle, ONE old-vs-new list (a8, 2026-11-18)

**For Priya; supersedes QUIET_RESTYLE_K4_STAR_CHANGES_20261118.md.** Nothing on the deck or the Movement (Execution)
page changes until she has seen this. Decisions behind it (11-18): common k = 4; restyle conventions (bootstrap stars
primary, mixed-model dagger, change vs own pre, session markers; accepted); the analyzed time ends at the LAST SYNC PULSE.

## 1. The fix: engaged quiet ends at the last sync pulse
- Before: when no quit was detected, `engaged` ran to the end of imaging, so imaging after the task stopped counted as
  engaged quiet. Now engaged = before min(quit, last sync edge) (`QS_ENGAGED_END=lastsync`, `quiet_engaged_end.py`).
- In every session the sync heartbeat stops 0-25 s after the last cue (median 5.5 s): the task controller stops with
  the task. Post-task imaging tails (last camera frame - last sync), > 2 min: PS93_0806 12.6 (pre), PS95_0810 9.4 (pre),
  PS95_0819 8.4, PS92_0828 7.2, PS95_1001 5.8, PS95_0812 4.1 (pre), PS95_0827 3.6, PS93_0925 3.2, PS95_0928 3.2,
  PS93_0922 2.3, PS95_0809 2.2 (pre), PS92_1006 2.2; 14 more 1-2 min. A tail matters only when the quit is censored.
  Table: `<cohort1_quiet_spikes>/engaged_end_20261118.csv`.
- The last-sync end and the first rule (last cue + response window) differ by <= 21.9 s (PS92_0915; median 2 s); no
  session > 30 s.
- Run: `v2_quiet_lp_c1guard_syncend_20261118/` (whole chain re-extracted, code 871be8f9, runner
  `quiet_c1guard/run_chain_syncend.sh`). The intermediate `..._engend_20261118/` (last-cue rule) is superseded.

## 2. What the fix does to the numbers (per animal first)
- Face-regressed orofacial events / min (both hemispheres; slide 157), pre -> acute, old -> new: PS92 6.6 -> 32.0 becomes
  7.1 -> 36.2; PS93 10.2 -> 49.7 becomes 6.5 -> 50.9; PS94 4.9 -> 31.0 and PS95 3.8 -> 23.1 unchanged.
- PS93's pre falls because PS93_0806 loses its 12.6-min post-task tail (orofacial 22.2 / 31.5 -> 1.8 / 4.2 per min).
- PS92_0828 (the chronic outlier) keeps only its 2.3 task-period quiet minutes: still high (orofacial 68 / 82, limb 75 /
  84 per min), so it is an unusual task-period session rather than a tail artifact. It stays in the statistics.
- Base summary per animal and every mixed-model term old -> new: `COMPARE_k4_vs_syncend.md` (this folder). The acute and
  subacute rises grow a little (PS92, PS93); PS94 / PS95 barely move (no long censored tails in their key epochs).
- Event maps (slide 204), significant bins at -0.3 / 0 / +0.2 / +1.0 s: 60 / 56 / 542 / 85 -> 60 / 74 / 610 / 331; rim
  enrichment at -0.3 s 1.83x -> 1.63x. Orofacial decay: acute -0.0001 s (p .99) -> -0.0027 s (p .79), still no change.

## 3. Stars: what is on the deck / page now vs after (fix + bootstrap)
"Now" = the stars the current figures carry (mixed model, FDR; T13 = trial-pooled permutation p), on the pre-fix data.
"After" = bootstrap stars on the sync-end data. Last column: the star also flips between the bootstrap on the pre-fix data
and the bootstrap on the fixed data (the fix alone). Every cell (point, CI, per-animal changes, all four star columns):
`v2_quiet_lp_c1guard_syncend_20261118/restyled_20261118/star_changes_combined_vs_deck.csv`.

| family | cells | stars now (deck / page) | stars after | changed | changed by the fix alone |
|---|---|---|---|---|---|
| pooled | 24 | 14 | 14 | 2 | 2 |
| fr | 12 | 12 | 12 | 0 | 2 |
| fr_detail | 48 | 30 | 32 | 6 | 1 |
| hemo | 45 | 16 | 24 | 8 | 6 |
| stillness | 30 | 10 | 18 | 8 | 3 |
| t12 | 24 | 23 | 21 | 2 | 3 |
| decay | 12 | 4 | 4 | 0 | 0 |
| t13 | 64 | 37 | 20 | 23 | 7 |

| panel | epoch | now (mixed model / permutation) | q / p now | after (bootstrap, sync end) | change [95% CI] | animals in direction | from the fix alone? |
|---|---|---|---|---|---|---|---|
| pooled EITHER polarity limb ipsilesional | subacute | - | 0.19 | * | 2.46 [0.0198, 5.6] (difference) | 3/4 | yes |
| pooled EITHER polarity limb contralesional | chronic | * | 0.0481 | - | 8.13 [-0.175, 19.9] (difference) | 3/4 |  |
| fr_detail raw_both orofacial contralesional | chronic | * | 0.037 | - | 4.95 [-0.00522, 12] (difference) | 3/4 |  |
| fr_detail raw_both limb ipsilesional | subacute | - | 0.167 | * | 2.7 [0.222, 5.79] (difference) | 3/4 |  |
| fr_detail raw_both limb contralesional | subacute | - | 0.136 | * | 3.59 [0.287, 7.79] (difference) | 4/4 |  |
| fr_detail raw_both limb contralesional | chronic | * | 0.0426 | - | 8.38 [-0.091, 20.4] (difference) | 3/4 |  |
| fr_detail move_frac orofacial ipsilesional | subacute | - | 0.0564 | * | -0.162 [-0.272, -0.0542] (difference) | 4/4 |  |
| fr_detail r2 orofacial ipsilesional | acute | - | 0.718 | * | -0.0623 [-0.121, -0.00996] (difference) | 4/4 |  |
| hemo corr470 p0510 | subacute | - | 0.0574 | * | 0.284 [0.126, 0.464] (log10 fold) | 4/4 |  |
| hemo raw470 p14 | subacute | - | 0.467 | * | 0.219 [0.0561, 0.391] (log10 fold) | 4/4 | yes |
| hemo raw470 p0510 | subacute | - | 0.135 | * | 0.195 [0.0139, 0.362] (log10 fold) | 3/4 | yes |
| hemo raw470 rate | subacute | - | 0.0654 | * | 0.275 [0.16, 0.395] (log10 fold) | 4/4 |  |
| hemo raw470 eta | subacute | - | 0.436 | * | 0.415 [0.018, 0.994] (difference) | 4/4 | yes |
| hemo raw415 sd | subacute | - | 0.0909 | * | 0.0406 [0.00883, 0.0776] (log10 fold) | 4/4 |  |
| hemo raw415 eta | acute | - | 0.223 | * | 0.216 [0.0297, 0.476] (difference) | 4/4 |  |
| hemo raw415 eta | subacute | - | 0.522 | * | 0.17 [8.9e-05, 0.432] (difference) | 4/4 | yes |
| stillness rate bin 2-inf | acute | - | 0.0658 | * | 5.93 [0.301, 15] (difference) | 3/4 |  |
| stillness rate bin 2-inf | subacute | - | 0.113 | * | 3.1 [0.668, 5.67] (difference) | 4/4 | yes |
| stillness share bin 0-0.25 | acute | - |  | * | 0.141 [0.0493, 0.231] (difference) | 4/4 |  |
| stillness share bin 0-0.25 | chronic | - |  | * | -0.0686 [-0.133, -0.0211] (difference) | 4/4 |  |
| stillness share bin 0.25-0.5 | acute | - |  | * | 0.117 [0.0659, 0.169] (difference) | 4/4 |  |
| stillness share bin 0.5-1 | subacute | - |  | * | 0.0394 [0.0101, 0.0694] (difference) | 4/4 |  |
| stillness share bin 1-2 | acute | - |  | * | -0.151 [-0.227, -0.0767] (difference) | 4/4 |  |
| stillness share bin 2-inf | acute | - |  | * | -0.106 [-0.152, -0.0647] (difference) | 4/4 |  |
| t12 precue orofacial contralesional | chronic | * | 0.0263 | - | 5.07 [-0.38, 14.8] (difference) | 3/4 |  |
| t12 precue_notquiet orofacial | chronic | ** | 0.0041 | - | 5.38 [-0.000267, 15] (difference) | 3/4 | yes |
| t13 farcontra_fail | subacute | - | 0.0589 | * | 0.038 [0.00682, 0.0668] (level (vs 0)) | 4/4 |  |
| t13 p_true_enl_event | subacute | ** | 0.002 | - | -0.0183 [-0.0378, 0.000512] (level (vs 0)) | 4/4 |  |
| t13 allspout_miss close_ipsi | pre | * | 0.026 | - | 0.00626 [-0.00715, 0.0284] (level (vs 0)) | 2/4 |  |
| t13 allspout_miss close_ipsi | subacute | * | 0.034 | - | 0.0101 [-0.0029, 0.0374] (level (vs 0)) | 3/4 |  |
| t13 allspout_miss close_mid | pre | * | 0.023 | - | 0.00737 [0, 0.018] (level (vs 0)) | 3/4 |  |
| t13 allspout_miss close_mid | acute | ** | 0.002 | - | 0.0208 [-0.00741, 0.0546] (level (vs 0)) | 3/4 |  |
| t13 allspout_miss close_mid | chronic | *** | 0.000999 | - | 0.0148 [0, 0.0435] (level (vs 0)) | 3/4 |  |
| t13 allspout_miss close_contra | pre | * | 0.031 | - | 0.00725 [-0.00179, 0.023] (level (vs 0)) | 2/4 |  |
| t13 allspout_miss close_contra | chronic | * | 0.05 | - | 0.00999 [-0.000419, 0.0355] (level (vs 0)) | 3/4 |  |
| t13 allspout_miss far_mid | pre | * | 0.022 | - | 0.0163 [-0.00961, 0.055] (level (vs 0)) | 3/4 |  |
| t13 allspout_miss far_mid | subacute | *** | 0.000999 | - | 0.0648 [-0.0201, 0.13] (level (vs 0)) | 2/4 |  |
| t13 allspout_miss far_mid | chronic | ** | 0.00599 | - | 0.0197 [-0.0103, 0.0662] (level (vs 0)) | 3/4 |  |
| t13 allspout_miss far_contra | subacute | ** | 0.002 | - | 0.0872 [-0.00477, 0.177] (level (vs 0)) | 3/4 |  |
| t13 allspout_miss far_contra | chronic | ** | 0.003 | - | 0.0313 [-0.000277, 0.0954] (level (vs 0)) | 3/4 |  |
| t13 p_true_enl_event close_ipsi | pre | ** | 0.003 | - | -0.0303 [-0.0734, 0.0315] (level (vs 0)) | 3/4 |  |
| t13 p_true_enl_event close_ipsi | chronic | ** | 0.002 | - | -0.0319 [-0.0798, 0.00503] (level (vs 0)) | 4/4 |  |
| t13 p_true_enl_event close_mid | chronic | * | 0.021 | - | -0.0268 [-0.0832, 0.0134] (level (vs 0)) | 2/4 |  |
| t13 p_true_enl_event far_ipsi | subacute | *** | 0.000999 | - | -0.0592 [-0.0993, 0.00769] (level (vs 0)) | 3/4 | yes |
| t13 p_true_enl_event far_ipsi | chronic | *** | 0.000999 | - | -0.0111 [-0.0465, 0.0404] (level (vs 0)) | 3/4 |  |
| t13 p_true_enl_event far_mid | pre | - | 0.149 | * | -0.0253 [-0.0607, -9.69e-05] (level (vs 0)) | 4/4 | yes |
| t13 p_true_enl_event far_mid | subacute | - | 0.0899 | * | 0.0334 [0.000925, 0.0636] (level (vs 0)) | 3/4 |  |
| t13 p_true_enl_event far_contra | pre | ** | 0.00699 | - | 0.031 [-0.016, 0.063] (level (vs 0)) | 3/4 |  |
| t13 p_true_enl_event far_contra | acute | *** | 0.000999 | - | -0.0262 [-0.0526, 0.00468] (level (vs 0)) | 4/4 | yes |

## 4. Reading
- **The acute rise stands everywhere** (all acute orofacial event-rate cells starred, 4/4).
- **The chronic contralesional rise comes back with the fix:** orofacial both-polarity chronic (pooled), face-regressed
  orofacial contralesional chronic (+4.0 [0.25, 10.0] events / min, 3/4) and T12 ITI / quiet chronic are starred again by
  the bootstrap; on the pre-fix data the bootstrap had dropped them. The post-task tails had diluted chronic sessions.
- **Stillness:** the acute rise holds in every stillness bin; chronic rises in the two stiller-middle bins are starred
  again (bin 0.25-0.5 +10.6 [3.7, 18.7], 4/4).
- **415 nm:** the response at corrected-470 events stays above pre acutely and now also subacutely (+0.17, 4/4, small);
  415 0.5-1 Hz power no longer falls acutely after the fix. Uncorrected 470 rises subacutely (power, amplitude).
- **T13:** most by-spout permutation stars do not survive the session-level test (as in the earlier list); the decoder
  p_true excess on event trials stays starred pre / acute / chronic.
- Many fix-only flips are borderline (CI edge within 0.02 of 0), carried by 3/4 animals; read them as unstable.

## 5. Figures (restyled, sync-end data; not yet on the deck / page)
`v2_quiet_lp_c1guard_syncend_20261118/restyled_20261118/`: quiet_events_pooled_epochs_restyled,
quiet_events_face_regressed_restyled (q_fr, 1x4) and _2x2_restyled (deck slide 157 layout),
quiet_events_face_regressed_detail_restyled (q_fr_detail), quiet_hemo_check_restyled (q_hemo415),
quiet_stillness_restyled (q_stillness), t12_task_window_event_rates_restyled (q_task_rates),
quiet_eta_decay_fr_restyled (q_decay_fr), t13_precue_event_vs_outcome_restyled (q_task_outcome). Unrestyled outputs of
the same run for slides 155 / 156 / 204 (movie, example traces, event maps) and the page's other section-23 figures are in
the run folder as before. Script `scripts/cohort2_qc/quiet_restyle_figs.py` (QS_RESTYLE_BASE / QS_RESTYLE_FOL).

## 6. Applied (11-18, Priya: "yes you can swap the figures and numbers")
- Deck slides 155-159 + 204 swapped from this run: movie (155), example traces (156), restyled 2x2 face-regressed
  figure (157), restyled stillness panel + 470 / 415 rows with all five columns (158), restyled T13 decoder panel (159),
  event maps (204); notes rewritten with the sync-end numbers and the bootstrap / mixed-model star rule. Titles: 158
  "Holds at matched stillness; small at 415 nm", 159 "Pre-cue event: weaker code pre, acute and chronic"; 157 bullet
  x7.3 / x3.7 / x1.9. Script `docs/status/deck_swap_20261117/quiet_c1guard/deck_swap_syncend.py`, images
  `syncend_crops.py`; deck backup `Overview_2026-11-06.pptx.bak_pre_a8_quiet_syncend_20261118`.
- Note on slide 158: the 415 response at corrected-470 events rises acutely by +0.22 % dF/F [0.03, 0.48] (4/4, PS94
  +0.59) against +0.31 at corrected 470, so "small" holds for the 415 power / rate measures but less so for the event
  amplitude, which is carried mostly by PS94. Flagged to Priya.
- Movement page section 23: 3c (messaged with the run and this list).
