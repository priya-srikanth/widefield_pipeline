# 15f (imaging page Fig 12.1) on the rebuild: frozen pre-stroke REST decoder on RESTnl working-bracketed rest (2026-11-17)

Priya chose this re-run on 11-17. Script: `scripts/rest_migration/rest15f_restnl_1117.py`, which reuses `rest_frozen_decoder_nl`.
The page, FINDINGS and deck are NOT edited: conclusion changes go to Priya first.

## Priya's decisions (11-17, after the first run)
1. Lick-free rest = DAQ + cam4 LP licks only, matching the rebuild's `restnl_mask`. This is the PRIMARY run (`--lp cam4`).
   The first run, which also removed cam1-only events (`--lp cam4cam1`), is kept as a stricter SENSITIVITY run.
2. PS92 STAYS in the retained and pooled statistics. The caption carries a caveat that its lick-free pre-stroke rest accuracy is
   barely above chance.

## Definition
- **Rest set.** restdock05 (docked, not running) minus every DAQ lick (the behavior DAQ detector) and every cam4 LP lick
  (`licks_all.csv`), widened -0.5 s / +2.0 s, with the min_rest_s floor re-applied (`rest_frozen_decoder_nl.lickfree_rest`).
  Sensitivity run: also minus cam1-only events (`lp_licks.lp_lick_times`), which add 4-7 % more LP licks per animal.
- **Brackets.** The cue before the period and the next trial must be at the same position, both inside the camera working
  period, AND both camera WORKING trials (`trial_classes.in_sel(cue_classes_for(...), "working")`, contact or incomplete).
  `_collect` gained the opt-in `period_cues=True` (the default output is unchanged and the pin tests pass).
- **Matching.** Duration and lick gap are matched to the central 80 % of the animal's pre periods. The gap is measured to the nearest DAQ-or-LP lick.
- **Frozen.** Every pre model (full + LOSO) goes through `frozen_models.load_or_fit` in the rebuild store. New aligns:
  `rest-restnlwk-...` (cam4) and `rest-restnlc1wk-...` (cam4cam1); nothing old is overwritten. Re-runs reloaded every
  model (`frozen-hit`) and reproduced the sessions tables exactly.
  - cam4, June + August: PS92 `cd740abcfeeb`, PS93 `de1894f2546f`, PS94 `93195dd1afc7`, PS95 `35ec1bca2d25`
  - cam4, August-only: PS94 `155350ad046e`, PS95 `dc9fb769a6db`
  - cam4cam1, June + August: `c613fc71404b` / `fd9ca8f295a7` / `6e2fcb6839e8` / `54c4e3500718`
  - cam4cam1, August-only: `1c79a8880bed` / `72435b2cc60d`
- Env: profile cohort1_rebuild_20261103, block rest_frozen_decoder_nl (PRE_FROM unset); wfield_local from this worktree.
- Outputs: `<rebuild>/labcams/grant_figures/<primary tag>/epoch/`
  - `rest15f_restnl_cam4_20261117/{allpre,augpre}/` (primary)
  - `rest15f_restnl_20261117/{allpre,augpre}/` (cam4cam1)
- Timing: collection takes 14-17 min for 120 sessions on 4 workers (median 33 s per session). Fitting takes about 2-4 min per pre set.
- Cache note: `rest_decoder_nl_cache/restnlwk/` holds 30 orphan files under key `21cac4d79fcb`. A docstring edit made during the
  collect shifted `inspect.getsource` line numbers in workers that had already imported the module, so those workers computed a wrong key. Those
  sessions were recomputed under the correct key `3dcede1404f2`. The orphans are our own derived files and were not deleted (list them if a cleanup is wanted).

## Primary pre set = June + August (both LP sets)
Threshold = the fairness gate: >= 30 duration- and gap-matched pre TRAINING periods at every position per animal.
August-only fails in both runs. cam4: PS93 far contra 8 and PS92 far mid 20. cam4cam1: PS93 far contra 8 and PS92 far mid 21.
June + August passes for all four. Counts: `*/data/counts_animal_epoch_position_*`.

## Results (June + August pre; retained = above-chance fraction / own pre)
| animal | page (restdock05, DAQ) acute / sub / chr | cam4 | cam4 + cam1 |
|---|---|---|---|
| PS92 | 0.61 / 0.65 / 0.80 | 1.01 / 2.91 (only 0824 meets the 40-period minimum; 0826 has 33) / 1.34 | 0.97 / 2.65 / 0.98 |
| PS93 | 0.28 / 0.45 / 0.69 | -0.23 / 0.40 / 0.55 | 0.07 / 0.58 / 0.56 |
| PS94 | 0.09 / 0.13 / 0.19 | 0.39 / 0.51 / 0.27 | 0.37 / 0.41 / 0.20 |
| PS95 | -0.09 / 0.49 / 0.47 | 0.29 (the one recorded acute session, 0817) / 0.67 / 0.62 | 0.05 / 0.73 / 0.62 |
| pooled flat | 0.28 / 0.36 / 0.54 | 0.30 / 0.73 / 0.56 | 0.38 / 0.74 / 0.51 |
| change in above-chance fraction, acute | -0.222 [-0.313, -0.121] * | -0.118 [-0.186, -0.036] * | -0.122 [-0.194, -0.039] * |
| subacute | -0.170 [-0.269, -0.091] * | -0.031 [-0.116, +0.089] | -0.028 [-0.119, +0.086] |
| chronic | -0.139 [-0.246, -0.055] * | -0.071 [-0.140, -0.001] * | -0.082 [-0.150, -0.016] * |
| sessions above null (pre / acute / sub / chr) | 44/44, 6/16, 13/17, 37/43 | 30/38, 4/16, 7/16, 23/43 | 29/35, 4/14, 9/16, 24/43 |

- **Pre-stroke accuracy, cam4** (acc vs block-permutation null; pre sessions above null at p < .05):
  PS92 0.231 vs 0.167 (4/8), PS93 0.290 vs 0.171 (7/9), PS94 0.351 vs 0.167 (9/10), PS95 0.354 vs 0.166 (10/11).
  On the page (lick periods included): 0.383 / 0.402 / 0.470 / 0.356.
- August-only sensitivity (PS94 and PS95 only), cam4: acute 0.43, subacute 0.67, chronic 0.36. All three change CIs exclude 0.
- PS95 acute is the one recorded acute session (0817), and it moves 0.29 -> 0.05 when the cam1-only licks are also removed.

## 11-18: Fig 12.1 title redraw (c1all, allpre + augpre)
- The second suptitle line (the RESTnlc1 mask text, ~260 characters) was clipped at both edges. `plot_main` now wraps
  every title line at 185 characters (`suptitle_text`) and grows the figure by 0.18 in per extra line, so the panels keep
  their size. Redrawn with `--plot-only --lp c1all --pre all|aug` (no refit): `_sessions`, `_per_animal`, `_cohort`,
  `_confusion`, `_confusion_meta` CSVs byte-identical to the 11-17 run; the confusion PNGs are byte-identical and stayed
  in place. Same file names. The 11-17 main PNG + SVG of each pre set were moved to
  `<pre>/_superseded_rest15f_title_20261118/` (2 PNG, 2 SVG).
- `--plot-only` had been broken: `ef.confusion_row` writes its panel table over `data/<stem>_confusion.csv` (the merged
  counts it was drawn from). The plot-only path now re-merges the confusion counts from the per-animal parts first.
- cam4-only render (`rest15f_restnl_cam4_20261117`) checked with a scratch redraw: its third title line now wraps to two
  lines, nothing clipped, tables byte-identical; the published cam4 PNG on M: was NOT replaced.

## 11-18: Fig 12.1 panel C session markers (Priya; redraw only)
- Panel C now also draws one faded small open marker per SESSION (shape = animal, color = epoch, alpha 0.45, left of the
  CI with small jitter, own RNG so A / B jitter is unchanged): the session's above-chance fraction minus its animal's own
  pre mean (same units as the y axis). Pre sessions are drawn the same way (the no-stroke spread around 0). Animal means,
  the animal-weighted mean + 95% CI and the boot_delta stars (top band) are unchanged; no LMM dagger, no "k/4" labels.
  Legend: "animal mean (shape = animal)" and "session - own pre mean (faded; pre = no-stroke spread)".
- Redrawn with `--plot-only` for the floor30_pct5 renders: c1all allpre + augpre, cam4 allpre. All 84 CSV / JSON files in
  the three `data/` folders hash-identical before and after; the confusion PNGs byte-identical. Same file names. The
  previous main PNG + SVG of each were moved to `<pre>/_superseded_rest15f_panelC_20261118/` (3 PNG, 3 SVG).
- Imaging page Fig 12.1 images 1-3 swapped and the plot clause of the caption rewritten (page v64).
