# ITI recovery marker: spout-matched and docked ITI angles; far-contra within-spout raw balance (11-17, a8)

Cohort 1 (PS92-95), rebuild tag `hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_pre0801_sel-camera-cue_stop-endlick-cls-contact-incomplete-stopA2jexrt3500c70_daqlk`,
DAQ licks, August pre. PROVISIONAL. Code: `scripts/lick_iti_matched_markers.py`. It reuses `learned_compensation._iti_base`,
`_refs`, `t3_cells` and `PHASE_MAP` (`t3_cells` / `PHASE_MAP` / `D_OFF_MIN_SPOUTS` were factored out of `cmd_summarise`, logic
unchanged), plus `lick_recovery_primary.marker` and `fit_panels` (factored out of `compute`, logic unchanged).
`lick_balance_angle_lmm.compute` and `plot` are reused too.
Outputs: `M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/iti_licks/<folder>/<tag>/`
in four new folders. Each folder has a README that lists the runs and the env.
`bal_dock_itimatch_20261117`, `iti_matched_itimatch_20261117`, `balangle_all_itimatch_20261117`, `balangle_dock_itimatch_20261117`.

## 1. Definitions

All angles use the 3-D landing angle asin(side / reach) of the KEY ITI licks (`lick_order_preds` x `covref_covariates`), the
per-lick angle that `iti_ang` already uses. Sign: + = contralesional (the sign of the pre far-contra cued angle, as in
`lrm.sessions`). Every axis is in degrees relative to the animal's own pre.

| axis | definition |
|---|---|
| `iti_ang` (existing) | Session mean over ALL ITI licks, minus the animal's mean over its pre sessions. Reproduced exactly (max diff 2e-15 vs `recmark_sessions.csv`). |
| `iti_ang_matched` | T2 per-lick d = angle - the animal's pre mean for ITI licks after the SAME previous spout in the SAME ITI phase (retract / docked / outbound; `iti_dock_audit`). Pre sessions leave their own session out (`_refs`). Licks follow an engaged previous trial at one of the six spouts (`_iti_base`). Value = session mean of d, >= 10 licks. |
| `iti_ang_dock` | Docked ITI licks only (audit phase `docked`), unmatched: session mean minus the animal's pre mean (as `iti_ang`). |
| `iti_ang_dock_matched` | T3 D_off: docked licks after an engaged trial; per lick, angle - pre at the same previous spout. Per session: mean over previous-spout cells with >= 10 licks, only when >= 4 cells (`t3_cells`). It matches the learned-compensation T3 D_off (theta_E, own lick table) with r = 1.000 over 86 sessions. PS95 has few such sessions (pre 2, acute 0), so n = 62 post. The lick-weighted version (`_lw`, 76 sessions) is given as a check. |

Previous-trial engagement came from `learned_compensation/iti_licks.csv.gz` (rebuild), joined by DAQ onset. That table
stores onsets at 6 significant digits, so the join tolerance is 6 ms. At 2 ms only 50 % joined; at 6 ms 98.7 % join, and
the previous spout agrees for 99.6 %.

Predictors:
- **ITI balance (all ITI).** The primary marker: angle-matched gain ratio, oro + SSs, 0.1..0.5 s after the first ITI lick
  following >= 2 s quiet. Taken from `bal_full_20261117`, unchanged.
- **DOCKED ITI balance.** The same marker on docked first-after-quiet licks, through the `lick_balance_subsets`
  `WFIELD_ITI_FILTER=dock` hook (27,968 docked ITI licks; median 66-98 docked first-after-quiet licks per session against
  181-244 for all ITI).
- Recomputed: the session table only (`--sessions-only`, cached session files). No upstream rebuild step was re-run.

## 2. Comparison table: ITI axes x predictors (post-stroke sessions)

Per-animal r uses the STORED hemispheric sign: + = a larger contralesional response goes with more contralesional licks.
The figures show the implied direction (= -balance), so the r's there have the opposite sign.

Column key:
- **cvR2**: leave-one-ANIMAL-out cvR2. **days**: the same with log(day) alone, on the same sessions.
- **p**: within-animal shuffle, 2000 permutations.
- **blk**: within-animal blocked cvR2, for the marker and for days.
- **slope**: concurrent MixedLM with log(day); **next**: the next-session change model.

| predictor -> axis | n | r PS92 / 93 / 94 / 95 | cvR2 | days | p | blk (marker / days) | slope, p | next, p |
|---|---|---|---|---|---|---|---|---|
| ITI -> `iti_ang` (published) | 76 | .69 / .83 / .91 / .04 | **.437** | .167 | .0005 | .17 / .43 | 5.6, .003 | -0.7, .73 |
| ITI -> aim (published) | 76 | .67 / .87 / .86 / .33 | **.567** | .226 | .0005 | .38 / .75 | 12.5, <1e-4 | 3.7, .23 |
| ITI -> `iti_ang_matched` | 76 | .67 / .90 / .90 / .16 | **.511** | .187 | .0005 | .24 / .65 | 5.3, .001 | -0.2, .93 |
| ITI -> `iti_ang_dock` | 76 | .73 / .67 / .90 / .03 | **.366** | .027 | .0005 | .39 / .39 | 10.0, <1e-4 | -0.8, .77 |
| ITI -> `iti_ang_dock_matched` | 62 | .77 / .90 / .88 / -.19 | **.714** | -.250 | .0005 | .57 / .57 | 12.2, <1e-4 | -4.9, .09 |
| ITI -> `iti_ang_dock_matched_lw` | 76 | .69 / .83 / .88 / .07 | .528 | .044 | .0005 | .43 / .59 | 8.2, 1e-4 | 0.7, .79 |
| DOCKED ITI -> `iti_ang` | 74 | .49 / .35 / .78 / -.26 | .006 | .149 | .006 | -.12 / .44 | 1.1, .22 | -1.7, .03 |
| DOCKED ITI -> aim | 74 | .49 / .37 / .79 / .02 | .052 | .221 | .0005 | -.02 / .75 | 3.2, .007 | -1.1, .38 |
| DOCKED ITI -> `iti_ang_matched` | 74 | .47 / .38 / .73 / -.12 | -.004 | .168 | .001 | -.15 / .66 | 0.4, .57 | -1.0, .14 |
| DOCKED ITI -> `iti_ang_dock` | 74 | .53 / .20 / .84 / -.26 | -.119 | .035 | .018 | .13 / .40 | 3.2, .005 | -2.3, .04 |
| DOCKED ITI -> `iti_ang_dock_matched` | 61 | .60 / .33 / .58 / -.11 | -.084 | -.250 | .0005 | .17 / .57 | 3.8, .13 | 0.8, .72 |

Table: `iti_matched_itimatch_20261117/<tag>/itimatch_summary_table.csv`; all models: `itimatch_models.csv` / `itimatch_cv.csv`.
The days baseline for `iti_ang_dock_matched` (-0.25) is low because PS95 contributes only chronic sessions, so the
held-out-PS95 fold extrapolates. Read that row's cvR2 against the lick-weighted row (.53 vs .04) and the blocked CV
(.57 vs .57).

**Pre-rebuild docked vs rebuild docked** (`lick_recovery_primary`, docked ITI balance; cued rows unchanged):

| panel | pre-rebuild `itifilter_dock` (10-29, capped first-lick set) | rebuild `bal_dock_itimatch` (FULL set) |
|---|---|---|
| ITI -> `iti_ang` LOAO (days) | -0.109 (0.149), p .29 | 0.006 (0.149), p .006 |
| ITI -> aim LOAO (days) | -0.191 (0.221), p .31 | 0.052 (0.221), p .0005 |
| ITI -> aim concurrent slope, p | 6.6, .003 | 3.2, .007 |
| cued -> aim LOAO (days) | 0.696 (0.226) | 0.657 (0.226) |

The rebuild reproduces the 10-29 finding: on docked licks the ITI balance does not beat days since stroke in an unseen
animal. It is now just above the shuffle null rather than at it.

**Docked ITI balance level per animal** (stored sign, mean over sessions; pre / acute / subacute / chronic):

| | all-ITI marker | docked marker |
|---|---|---|
| PS92 | .01 / -.26 / .00 / -.05 | .01 / -.25 / .03 / -.08 |
| PS93 | .03 / -.17 / .12 / .14 | .03 / **+.20** / .16 / .29 |
| PS94 | -.01 / -.79 / -.27 / -.36 | .02 / -.98 / .02 / .01 |
| PS95 | -.02 / -.06 / -.14 / -.07 | -.01 / **+.14** / -.09 / -.08 |

The acute contralesional-smaller ITI shift is 4/4 on all ITI licks but only 2/4 on docked licks, which matches the
pre-rebuild "acute -0.11 ns vs -0.34".

Figures:
- `iti_matched_itimatch_20261117/<tag>/figures/iti_matched_markers_scatter.png`: rows = predictor, columns = axis; open
  markers, shape = animal, color = epoch, gray = pre.
- `iti_matched_itimatch_20261117/<tag>/figures/iti_matched_markers_cv.png`: LOAO cvR2 vs days with the shuffle null.
  A star means cvR2 > max(days, 0) and shuffle p < .05.
- `bal_dock_itimatch_20261117/<tag>/figures/lick_recovery_primary_full.png`: the docked primary figure.

Per animal x epoch, the axes themselves (deg) are in the `axes_try2` log. All four axes fall acutely in 4/4 and correlate
.90-.97 across post sessions.

## 3. Far contra only, within spout: raw balance vs lick angle

Unit: one session at the far-contra spout. Cued = first cued licks with target far contra. ITI = bout starts with the
previous spout far contra. Each session x spout cell needs >= 5 licks.
- **resp** = log2(contra / ipsi response AUC) over the orofacial pool, from `lateral_ratio_sessions.csv`: cued 0..1.0 s,
  ITI 0..0.5 s.
- **dec** = X_right - X_left over 0..0.2 s.
- Model: y ~ (angle - pre mean) * post + (1 | animal).
- "inside" = post sessions whose angle lies within the animal's own pre angle range. Acute far-contra licks fall 10-37 deg
  beyond that range, where the pre line is extrapolated.

| set, measure (orofacial) | pre slope /deg (p) | slope change post (p) | shift at same angle (p) | shift, no angle (p) | shift inside pre range (n post, p) |
|---|---|---|---|---|---|
| cued resp 0-1 s | +.006 (.22) | -.018 (1e-4) | **+.24 (<1e-4)** | +.39 (<1e-4) | **+.23** (18, <1e-4) |
| cued dec | -.40 (<1e-4) | +.13 (.10) | +1.73 (.001) | +5.26 (<1e-4) | +0.29 (18, .62) |
| ITI all, resp 0-0.5 s | -.013 (.03) | -.001 (.84) | +.08 (.09) | +.27 (<1e-4) | +.08 (23, .10) |
| ITI all, dec | -.29 (<1e-4) | +.09 (.11) | -1.08 (.02) | +1.70 (.001) | -0.63 (23, .22) |
| ITI docked, resp 0-0.5 s | -.026 (.001) | +.007 (.36) | +.11 (.12) | +.37 (<1e-4) | +.13 (19, .11) |
| ITI docked, dec | -.27 (<1e-4) | +.03 (.64) | **-1.82 (.001)** | +1.68 (.009) | **-1.42** (19, .013) |

Per animal (rule 8; r of balance with angle across sessions, pre / post; raw post - pre balance shift):

| | cued resp: r pre | r post | shift | cued dec: r pre | r post | docked ITI dec: r post | shift |
|---|---|---|---|---|---|---|---|
| PS92 | +.83 | -.95 | +.35 | -.91 | -.87 | -.86 | +1.86 |
| PS93 | -.62 | +.65 | +.28 | -.06 | -.17 | -.54 | -.33 |
| PS94 | -.39 | -.86 | +.67 | -.62 | -.90 | -.78 | +4.03 |
| PS95 | -.22 | -.49 | +.24 | -.62 | -.63 | -.70 | +.19 |

Each animal has only 7-8 pre far-contra sessions, spanning about 15 deg. PS95 has 3 docked far-contra pre sessions, so
its pre r is not computed.

Figures: `balangle_all_itimatch_20261117/<tag>/figures/farcontra_balance_angle.png` and
`balangle_dock_itimatch_20261117/<tag>/figures/farcontra_balance_angle.png`; the full session x spout LMM figures are
`lick_balance_angle_{dec,resp}.png` in the same folders.

**All spouts, session x spout** (Priya 11-17: "does the contra:ipsi balance still predict overall lick angle, just not
the post- vs pre-stroke angle delta?"). Values are pre slope / slope change post / shift at the same angle, orofacial pool:

| set | dec | resp |
|---|---|---|
| ITI all, resp 0..1 s (production) | -.31*** / +.06** / -1.79*** | -.023*** / +.003 / +.15** |
| ITI all, resp 0..0.5 s | -.31*** / +.06** / -1.79*** | -.018*** / +.003 / +.04 |
| ITI docked, resp 0..0.5 s | -.22*** / +.07*** / -1.35*** | -.017*** / +.000 / **+.16** ** |
| cued (0..1 s) | -.63*** / -.03 / -2.18*** | -.038*** / +.012*** / -.10*** |

Per-animal pre slopes are negative in 4/4 for every pair and both measures, on all ITI and on docked ITI.
Exceptions: the docked PS93 dec SSp-m slope is +.07, and the PS93 docked slopes are near 0.
Pairs: `balance_angle_lmm.csv` / `balance_angle_animals.csv` in the `balangle_*_itimatch_20261117` folders.

Caveat: across session x spout cells, the previous spout explains 57 % of the ITI angle variance pre-stroke (49 %
docked). Spout + animal x epoch explain 91-96 %. On all-ITI licks the spout is therefore a shared driver of angle and
balance, partly through the moving spout itself. The docked version, and the far-contra-only version above, are the
cleaner tests.

## 4. Reading

1. **Matching does not change the ITI marker result.** Against `iti_ang_matched` (same previous spout and phase, the T2
   reference), the all-ITI balance gives cvR2 .51 vs days .19 (p .0005), against .44 vs .17 unmatched. Against the docked
   ITI angle it gives .37 vs .03, and against docked + spout-matched (T3 D_off) .71 vs -.25 (.53 vs .04 lick-weighted).
   Per-animal r is .67-.90 in PS92-94 and near 0 in PS95 for every axis. So the ITI balance -> ITI angle link is not
   produced by which spouts preceded the licks or by the ITI phase of the BEHAVIORAL axis. Within-animal blocked CV stays
   below days (.24 vs .65), as before.
2. **Docking the BALANCE removes it, on the rebuild as before.** The docked ITI balance gives unseen-animal cvR2 .01 / .05
   (vs days .15 / .22) for ITI angle / aim, and -.12 to 0 for every matched or docked axis. Pre-rebuild: -.11 / -.19. Its
   acute shift holds in 2/4 animals (PS93 and PS95 reverse). The ITI-balance recovery link comes from licks while the
   spout moves, with 3x fewer docked licks per session; power was ruled out on 10-29 by subsampling.
3. **Far contra, cued (target fixed):** the raw response-ratio balance does not track angle before the stroke (pre slope
   ~0, per-animal pre r +.83 / -.62 / -.39 / -.22, narrow pre range). After the stroke it tracks angle in 3/4 (r -.95 /
   +.65 / -.86 / -.49). The post-stroke shift at the same angle survives: +.24 log2, p < 1e-4; +.23 inside the pre angle
   range. The raw shift is + in 4/4. The decoder-window difference tracks angle in pre and post (r -.91 / -.06 / -.62 /
   -.62 pre). Its same-angle shift (+1.7) disappears inside the pre range (+0.3, p .62), so for dec that shift rests on
   extrapolation.
4. **Far contra, ITI:** the 0..0.5 s resp balance tracks angle pre (slope -.013 all, -.026 docked). There is no
   same-angle shift (+.08 / +.11, ns), so the raw ITI shift (+.27 / +.37) is accounted for by the ipsilesional angle
   change. Docked dec shows a NEGATIVE same-angle shift (-1.8, p .001; -1.4 inside the range, p .013).
5. **Answer to Priya's question:** yes. Balance still predicts lick angle across spouts and sessions in pre and post
   (negative pre slopes 4/4, docked included, slope barely changed). On ITI licks, the post vs pre shift is mostly
   explained by the angle change (resp shift at the same angle ~0 on all ITI at 0..0.5 s). Two exceptions: the docked
   resp shift (+.16, p .002), and cued first licks, where a same-angle shift remains with the target fixed.
