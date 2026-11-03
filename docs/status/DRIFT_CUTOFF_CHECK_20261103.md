# Drift-cutoff check (2026-11-03)

Measurement check: should within-session slow-drift removal be part of the SHARED corrected signal for the cohort 1
rebuild, and at what cutoff? Priya's constraint: block-level (position) information must NOT be removed; preference: in
the shared signal, cutoff >= 10 min, set from measurement. Framing update (Priya, same day): the goal is to REPLACE the
order-10 polynomial drift step with a block-protected slow cosine basis, not to stack a second removal on top of hpfit
(the 0.1 Hz high-pass used only for fitting T stays).

Signal: `hemo_meegkit_hpfit_lag_refitT_swapmask`, all 120 curated cohort 1 sessions. Worktree `wf_swapmask`, branch
`analysis/cohort1-swapmask`. Code `scripts/drift_check/`, `wfield_local/drift_dct.py`, new `dctblk<c>_lag` variants in
`hemo_variants`. Outputs `M:/MICROSCOPE/Priya/Widefield/labcams/grant_figures/drift_cutoff_check_20261103/`.
Nothing adopted changes; in-memory decoders only (WIDEFIELD_EPOCH_FROZEN=0); swapmask session cache.

## Progress log

- 2026-11-03: started. Method module `drift_dct` (DCT-II cosines >= cutoff + one indicator per position block, fitted on
  the hpfit out-of-trial mask, only the cosine part subtracted; 3 unit tests). Variants `dctblk{10,15,20,30}_lag` added
  to `hemo_variants` (replacement arm; same mask, swap / lag repair, 0.1 Hz high-pass for T). Note: the out-of-trial
  mask keeps only ~20-21% of frames (PS92_0606 0.203, PS92_0820 0.210).
- 2026-11-03: **the literal per-BLOCK protection is ill-posed** (one-session probe, PS92_0820 acute). With a free offset
  per ~1.5-2 min block, slow cosines are nearly collinear with the block indicators and are identified only from
  within-block slopes on the ITI frames; the fit extrapolates those into large slow terms. "Removal" then ADDS power:
  10-43 min band x5.2 (10 min) / x6.0 (20 min), block means r ~0 in the worst ROI, pre-cue decode .374 -> .148 / .173,
  cue .630 -> .354 / .379. Added a per-POSITION protection instead (one indicator per spout position = union of its
  blocks; 6 regressors) -- protects the between-position means, which is the information the constraint is about.
  Same session: pos10 / 15 / 20 / 30 remove 89 / 51 / 26 / 1% of 10-43 min power; pre-cue .374 -> .461 / .481 / .469
  / .387; cue .630 -> .720 / .695 / .667 / .621. Unprotected cosines (no labels used) give similar gains (cos10 .420 /
  .700, cos20 .453 / .675), so the gain is not created by using the labels. Decoder path checked: the stacked-feature
  shortcut reproduces `filter_acausality_test.decode` exactly (base .3745, blk20 arm .1728 both ways).
  Variants now `dctblk<c>_lag` (literal, for the record) and `dctpos<c>_lag` (position-protected).
- 2026-11-03: items 1-2 done, 120 / 120 sessions (`blocks.csv`, `sessions.csv`, `block_summary_animal_epoch.csv`,
  `spout_order.csv`, fig1 / fig2). Blocks: median 55-125 s; p95 79-330 s; longest 631 s (PS92_0929 chronic), only
  2 blocks > 10 min in the whole cohort (both PS92), PS92 3% of blocks > 5 min, PS93 0.6%, PS94 / PS95 none (max
  180 / 155 s). Sessions 74-167 min (median ~100-147). **Dominant alternation period is ~6 blocks, not 2:** median
  14.2 / 10.1 / 6.5 / 5.9 min (PS92 / 93 / 94 / 95), p95 22.2 / 17.3 / 10.6 / 8.4 min, PS92 max 40 min. Within a
  session positions are spread MORE evenly in time than chance (eta^2 of block time by position .005-.023 vs
  permutation null .05-.15): the scheduler cycles through all six. Across sessions there is NO fixed order: Kendall W
  .007-.058 (Friedman p .12-.96), time-only leave-one-session-out decoder .16-.18 (chance .167), modal first block
  20-30% of sessions. Added a CYCLE band (median block -> p95 peak period: PS92 1.5-22 min, PS93 1.2-17, PS94 1.0-10.6,
  PS95 0.9-8.4) next to the duration band (median -> 2 x p95: 1.5-8.7 / 1.2-5.7 / 1.0-2.9 / 0.9-2.6 min).
- 2026-11-03: `measure` launched on all 120 sessions (4 workers; arms pos10/15/20/30, blk10/20, cos10/20).
  PITFALL: first launch failed on every session (ImportError drift_dct) -- the locanmf env's editable install points
  at the MAIN checkout (`Github/widefield_pipeline`), and running a script file does not put the worktree on sys.path.
  Relaunched with `PYTHONPATH=<wf_swapmask>`; `_env_check` now refuses to run otherwise. The `blocks` pass ran on the
  main checkout's modules; the modules it uses are identical to the worktree's except code order in `daq_io` and
  plotting text in `orofacial_cohort` (checked by diff), and positions agree 100% with `trial_bounds` in all 120.
  Replacement arm (`dctpos10/15/20_lag`, compute only, 8 sessions: longest-block pre + acute session per animal)
  launched alongside with 2 workers.
- 2026-11-03: replacement arm done (`replacement_vs_stacked.csv`; 8 sessions x dctpos 10 / 15 / 20; nothing written
  to any hemo_* dir). **Replacement does NOT closely match stacked.** Variance-weighted R2 (all components)
  replacement vs stacked .88-.99 (median ~.97); it is often LOWER than hpfit-vs-stacked at 15-20 min (i.e. the
  replacement differs from stacked more than hpfit does). Brain-mean r .978-.999; median ROI r .976-.999; but the worst
  ROI drops to r .28 (PS92_0819, 20 min), .52 (same, 15 min), .59 (PS94_0823, 20 min). Slow (>= 5 min) brain-mean
  power replacement / stacked .76-1.37. Decoding replacement - stacked per session: pre-cue -.052 to +.031, cue -.040
  to +.029 (mixed sign, no consistent direction). Robust weighting is not the cause (all four OLS / robust
  combinations give R2 .87-.90 on PS92_0819). Diagnosis on PS92_0819 worst ROI (a low-variance ROI, sd .022): the
  difference sits at 10-20 min periods (65% of its power) and is largest in the first and last 20 min (+-0.6 SD) --
  the raw 470 / 415 channels carry bleaching with components faster than the cutoff; the order-10 polynomial fits
  those ramps (it is flexible at the record edges), a cosine basis >= cutoff cannot, and 415 / 470 bleach at different
  rates so `T` does not cancel it. Brain-mean difference variance is only 3.5% of the signal there.
- 2026-11-03 09:00: resume check after the account switch -- exactly one `measure` run alive (PID 11660, started
  08:34 with PYTHONPATH set), 32 / 120 sessions in `measure_sessions.csv` (checkpointed every 8; reruns skip done
  sessions); `blocks` and `replace` finished; nothing duplicated or relaunched.
- 2026-11-03 09:31: raised `measure` to 8 workers (Priya: spare CPU / RAM). Stopped exactly PID 11660 and its four
  spawn workers (25872 / 34104 / 29148 / 24316, created 09:29 by it) right after the 64-session checkpoint; relaunched
  as PID 11380 with --jobs 8 (cap in the script raised to 8, checkpoint every max(8, jobs)); 56 left. Plan: drop to 6
  if free RAM < 6 GB.
- 2026-11-03 09:40: interim (64 sessions): position-protected arms raise pre-cue decoding MORE than the label-free
  cosine arms (pos10 +.02 to +.07 per animal x epoch vs cos10 -.015 to +.035). Suspected LABEL LEAKAGE: the
  protection uses the session's own position labels, so the part of the slow drift that happens to align with the
  six positions in THIS session is turned into a position-constant offset, which a block-CV decoder can learn from
  training blocks and find again in the held-out blocks. Launched a permutation null (`leakcheck`, PID 35000, 4
  workers, 15 sessions = 2 pre + 2 acute per animal, PS95 has 1 acute): run codes permuted (counts kept) and used
  for BOTH the protection and the decoding target, 5 permutations; label-free arms must sit at chance there.
- 2026-11-03 10:10: **leakage CONFIRMED** (`leakcheck_permuted_labels.csv`, 15 sessions). Under permuted labels the
  label-free arms sit at chance (pre-cue base .163, cos10 .161, cos20 .160) but the position-protected arms do not:
  pos10 .189 (+.025), pos20 .174 (+.010); per animal pos10 pre-cue +.050 / +.023 / +.012 / +.014 (PS92 / 93 / 94 / 95),
  cue +.045 / +.013 / +.015 / +.010 -- in all four. With true labels the pos10 pre-cue gain is +.054 / +.040 / +.030
  / +.026, so in PS92 the whole gain and in the others about half of it is what the protection manufactures from the
  labels. **A position-protected (label-using) drift step must not go into the shared signal**: every position analysis
  run on it would be partly circular. The label-free cosine arms show no leakage and small true gains (pre-cue cos10
  +.015, cos20 +.011; cue +.008 / +.010 pooled).
- 2026-11-03 ~10:50: `measure` complete, 120 / 120, no failures (`measure_sessions.csv`, item3 / item4 tables, fig3,
  fig4, fig4b). Headlines: (3) after hpfit the brain-mean has 0.2-1.3% of variance at > 43 min and 0.4-2.7% at
  20-43 min in most animal x epoch cells, but PS92 subacute (2 sessions) 26% / 31% and PS94 chronic 8% / subacute 5%
  at > 43 min (medians; out-of-trial similar). The block band holds 5-23%, the cycle band 12-28%. (4) block-band
  retention is 1.000 for every label-free and position-protected arm in every cell -- by construction, since a DCT
  cutoff is a sharp edge and every animal's block band ends below 10 min (PS92 8.7). The informative band is the
  CYCLE band: cos10 keeps min .66 (PS92 pre) / .19 (PS92 subacute); cos20 keeps >= .96 except PS92 subacute (.73);
  PS93-95 1.000. Fitting on the 21% out-of-trial frames sometimes ADDS 10-43 min power (cos10 PS92 pre: 10-43 min
  power x1.13, brain-mean x1.43): the out-of-trial slow signal is not the same as the all-frame slow signal there.
  Decoding (pooled 120): cos10 pre-cue +.008, cue +.010; cos20 +.016 / +.010; pos arms larger but leak (above).
- Launched: extra label-free arms cos15 / 25 / 30 / 40 on all 120 (PID 10460, 8 workers,
  `measure_sessions_extra_arms.csv`); label-free replacement `dctcos20/25/30_lag` vs stacked on the same 8 sessions
  (PID 29688, 3 workers, compute only). New variants `dctcos<c>_lag` in hemo_variants.
- 2026-11-03 ~11:40: label-free replacement done (8 sessions x 20 / 25 / 30 min; `replacement_vs_stacked.csv`,
  family dctcos). **Replacing the polynomial is worse than stacking on it, and gets worse as the cutoff lengthens**:
  R2 vs stacked median .969 / .956 / .937 (min .87 / .81 / .79, PS92_0819), whereas hpfit itself is closer to the
  stacked signal (.97-.996); worst-ROI r down to .27 / .12 / .02 (PS92_0819); >= 5 min brain-mean power replacement /
  stacked up to 1.67 (PS93_0821, 25 min). Decoding replacement - stacked: pre-cue median ~0 (range -.066 to +.025),
  cue median -.006 / -.008 / -.019 (never above +.023). Same cause as before: cosines >= cutoff cannot follow the
  early / late bleaching ramps in the raw channels that the order-10 polynomial follows.
- 2026-11-03 ~13:40: extra label-free arms done (cos15 / 25 / 30 / 40, 120 / 120, no failures;
  `measure_sessions_extra_arms.csv`, merged into the item4 tables and fig4). Cycle-band retention min per animal x
  epoch: cos15 .39 (PS92 sub) / .87-.89 (PS92 pre / acute); cos20 .73 (PS92 sub), .956-.972 (PS92 pre / acute /
  chronic), 1.000 elsewhere; **cos25 / 30 / 40 1.000 in every cell**. Block-mean between-position variance ratio
  (median ROI, per cell) at cos25 .96-1.07 except PS92 subacute .60 (the two drift-dominated sessions: 26-31% of
  brain-mean variance > 20 min; decoding there RISES, pre-cue +.043, cue +.036). 10-43 min power removed (comp,
  median) at cos25: -.05 to .20 (median ~.05); at cos30 ~.01. Pooled decode change cos25 pre-cue +.015, cue +.007
  (positive in 16 / 16 cells for pre-cue). Fitting on out-of-trial frames often INCREASES the > 43 min all-frame power
  (removed_gt43 negative in most cells, all cutoffs): the out-of-trial slow signal is not the all-frame slow signal.

## Conclusion (2026-11-03)

- Spout order: no fixed order across sessions (Kendall W .007-.058, time-only LOSO decode at chance) -> no risk
  that a slow trend masquerades as position in a frozen cross-session decoder through block order.
- The literal block-protected design is ill-posed; position protection leaks labels; replacing the polynomial loses
  the bleaching-ramp fit. If anything is added, it is a LABEL-FREE cosine removal STACKED on hpfit.
- Recommended cutoff if added: **25 min** (shortest label-free cutoff with block AND cycle band retention >= .98 in
  every animal x epoch, block means intact apart from PS92's two drift sessions). Block-band retention alone would
  allow 10 min, but cos10 / 15 / 20 strip up to 34 / 13 / 4% of PS92's position-cycle band.
- Against adding it: the effect is small (pre-cue +.015, cue +.007 pooled; it removes ~5% of 10-43 min power),
  it does not touch the >= 5 min position-informative band from the 10-20 check (that band IS the position cycle,
  5-22 min, and stays a sensitivity column), and the out-of-trial fit is not a clean drift estimate.
