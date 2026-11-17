# 15f (imaging page Fig 12.1) on the rebuild: frozen pre-stroke REST decoder on RESTnl working-bracketed rest (2026-11-17)

Priya chose this re-run on 11-17. Script: `scripts/rest_migration/rest15f_restnl_1117.py`, which reuses `rest_frozen_decoder_nl`.
The page, FINDINGS and deck are NOT edited: conclusion changes go to Priya first.

## What changed against the page's 15f
- **Rest set.** restdock05 (docked, not running) minus every DAQ lick (the behavior DAQ detector) and every LP lick, widened
  -0.5 s / +2.0 s, with the min_rest_s floor re-applied. LP licks = cam4 licks_all plus cam1-only events (`lp_licks.lp_lick_times`).
  NB the map-baseline RESTnl (`restnl_mask`) removes cam4 only. cam1-only adds 4-7 % more LP licks per animal (119/120 sessions have cam1 events).
- **Brackets.** The cue before the period and the next trial must be at the same position, both inside the camera working
  period, AND both camera WORKING trials (`trial_classes.in_sel(cue_classes_for(...), "working")`, contact or incomplete).
  `_collect` gained the opt-in `period_cues=True` (the default output is unchanged and the pin tests pass).
- **Matching.** Duration and lick gap are matched to the central 80 % of the animal's pre periods. The gap is measured to the nearest DAQ-or-LP lick.
- **Frozen.** Every pre model (full + LOSO) goes through `frozen_models.load_or_fit` in the rebuild store, align
  `rest-restnlc1wk-b4-dur10-gap10-gate` (a new spec: nothing old is overwritten). A second run reloaded all four (`frozen-hit`)
  and reproduced the sessions table exactly.
- Env: profile cohort1_rebuild_20261103, block rest_frozen_decoder_nl (PRE_FROM unset); wfield_local from this worktree.
- Output: `<rebuild>/labcams/grant_figures/<primary tag>/epoch/rest15f_restnl_20261117/{allpre,augpre}/`.
- Timing: PS95 alone took 3.4 min to collect (30 sessions, 4 workers) and 46 s to fit. The full run took 13.6 min to collect 120 sessions
  (median 33 s each) and about 3-4 min per pre set to fit (animals in parallel).

## Primary pre set = June + August
Threshold = the script's fairness gate: >= 30 duration- and gap-matched pre TRAINING periods at every position per animal.
Below it the frozen model barely sees that class. August-only fails for PS93 (far contra: 8 matched, 10 working periods
before matching) and for PS92 (far mid: 21). Both animals are refused, which leaves 2 animals. June + August passes for all four (the minimum is PS92 far ipsi at 74).
So June + August is primary for this figure, and August-only (PS94 and PS95 only) is the sensitivity run. Counts tables:
`*/data/counts_animal_epoch_position_*`.

## Result (June + August pre, 200 block permutations)
Retained = above-chance fraction relative to the animal's own pre. Stars = boot_delta CI of the change vs own pre.

| | pre | acute | subacute | chronic |
|---|---|---|---|---|
| PS92 | 1 | 0.97 | 2.65 (1 session) | 0.98 |
| PS93 | 1 | 0.07 | 0.58 | 0.56 |
| PS94 | 1 | 0.37 | 0.41 | 0.20 |
| PS95 | 1 | 0.05 (1 session) | 0.73 | 0.62 |
| pooled flat / animal-weighted | 1 | 0.38 / 0.37 | 0.74 / 1.09 | 0.51 / 0.59 |
| change in above-chance fraction [95% CI] | | -0.122 [-0.194, -0.039] * | -0.028 [-0.119, +0.086] | -0.082 [-0.150, -0.016] * |
| sessions above null p<.05 | 29/35 | 4/14 | 9/16 | 24/43 |

- **PS92's pre baseline is weak once LP licks are removed:** acc 0.237 against a null of 0.167, and only 4/8 pre sessions are above null.
  Its retained values are ratios to a small number. (Old page: PS92 had the strongest rest decoding.)
- August-only (PS94 and PS95): acute 0.45, subacute 0.61, chronic 0.39. All three change CIs exclude 0.

Old page 15f for comparison (restdock05, DAQ contacts only, June + August, in-memory fits): acute 0.28, chronic 0.54. Per animal acute
0.61 / 0.28 / 0.09 / -0.09 and chronic 0.80 / 0.69 / 0.20 / 0.47.
