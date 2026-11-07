# FINAL LP ARM render (11-02): plan, status, results

Priya 11-01/02: "ok for the full job". Every LP-arm imaging family rendered once in final form: LP licks + LP-inclusive
gate + August-only pre; RESTnl as the map reference plus the existing references as they are now; ENL decoders at 1.0
/ 1.5 (/ 2.0) s; DAQ-trained and LP-trained frozen decoders side by side. NEW outputs only (nothing overwritten, nothing
published, nothing committed). PROVISIONAL, n = 4 (PS92-95; cohort 2 not touched).

## 0. Tree

Worktree `C:/Users/SabatiniLab/Github/wf_lpfinal`, branch `analysis/lp-final`, created at `analysis/restnl-maps`
(00f1e2c2: RESTnl maps, main-based) + `git cherry-pick -n dea1de5f` (the gate option from `analysis/attempt-coding`;
auto-merged, the only overlaps were the cosmetic `config.date8` lines). Uncommitted; Priya commits.

Why this base: the two branches diverged at 2c6fcd89 (10-25); restnl-maps carries main (paths / cohort_out, sig_marks)
and the RESTnl code; the gate commit's files differ from main only by the date8 lines.

## 1. Code changes in this tree (beyond the two branches)

| file | change | default behavior |
|---|---|---|
| `position_reference_maps._raw_maps_code` | the WIDEFIELD_PRM_LEGACY_DATE8 path now drops gate hook lines (as `code_digest` does); without it the cherry-pick would have moved the pinned digest | unchanged; legacy digest still 81053b92244b (= the 10-25 `prm-*-3b1e0fc8b886` kinds) |
| `restnl_mask.cache_dir / _code` | under WIDEFIELD_GATE_SOURCE=lp the RESTnl baselines live in their own dir `.widefield_session_cache_restnl_glp<sha>` (the rest frames are engagement-gated, so they move with the gate; the build prunes superseded keys and must never touch the default ones); hook-marked lines are stripped from the module digest | `_code` 58b723824d63 and dir unchanged |
| `attempt_maps` | time-local REST variant: `signal_key` carries the gate tag under the switch (`tf-*` kinds are exempt from the gate tag, and this signal has the engaged-rest baseline subtracted INSIDE the features -> it would have reused DAQ-gated features); titles name lick source, gate and pre (`_run_text`) | key unchanged |
| `grant_confusion._collect_5c` + `_stored_epoch_decoder`, `grant_kit._pooled_bundle` (`basis_id`) | OPT-IN `WIDEFIELD_EPOCH_FROZEN=1`: the epoch acc / 5c decoders go through `frozen_models.load_or_fit` (kind `epoch_decoder`, payload full + LOSO = the same two fits as the in-memory path) | in-memory fit, unchanged |
| `epoch_figures.bar_row` | significance marks in the panel TOP BAND (`sig_marks.star_top` + `reserve_headroom`, one row per post epoch); no y ticks in the band | changes every bar figure rendered from this tree (Priya's 10-30 rule) |
| `epoch_figures.save_figdata / replot_figdata` | bar_row / contrast_panel inputs pickled to `data/<name>_figdata.pkl`; `--plot-only` redraws from them | new sidecar only |
| `scripts/map_drift_prepre.py` | title names the LP-inclusive gate under the switch | unchanged |
| `scripts/lp_final_render.py` (new) | the driver: `restnl`, `maps`, `decoders`, `compare`, `--plot-only` | - |

### FOUND: the LP-arm "frozen" decoders were LP-TRAINED all along
The epoch acc / 5c family (imaging page 2.1-2.4) never used `frozen_models`: `_collect_5c` fits the "frozen" decoder IN
MEMORY on the pooled pre-stroke bundle built under the CURRENT lick source. So every `LP_licks` acc figure to date
(2.4, the LP 1.0 s ENL comparison) was trained on LP-arm pre-stroke trials, and `WIDEFIELD_LP_FROZEN` never reached
this family (it applies to `locanmf_frozen_decoder`, section G). Also a rule-10 deviation (pre-stroke decoders refit
per render). Hence the opt-in store above: DAQ-TRAINED = fitted in a DAQ-source run at the same window + August pre
set, loaded by the LP run (which refuses to fit a DAQ spec); LP-TRAINED = `WIDEFIELD_LP_FROZEN=lp`, fitted in the LP
run. Own store `C:/Users/SabatiniLab/cue_lick/frozen_models_lpfinal` (WIDEFIELD_FROZEN_MODEL_DIR); the canonical stores
are untouched. Caveat: a DAQ-trained model scores LP-arm trials z-scored on the LP arm's engaged trials (training rows
used the DAQ arm's); pre-stroke the two engaged sets are near-identical.

## 2. Inventory: page figure -> entry point

| page | family | entry point (this tree) |
|---|---|---|
| 7.3 / 7.10 / 7.18 maps (+ 7.14 / 7.23 PRECUE, every ref) | 15r | `lp_final_render.py maps` (calls `epoch_maps._fig_15r_reference_maps`) |
| 7.19 per animal | 15rpa | same |
| 7.4 / 7.11 / 7.20 (+ 7.15 / 7.24) drift | 15rdrift | `map_drift_prepre.py --arm lp --refs rest precue restnl --out <LPfinal>/map_drift_prepre` (all-pre by construction) |
| 8.1 / 8.2 | attempt maps | `python -m wfield_local.attempt_maps --rest-variant restnl|rest --aligns lick cue precue --out ...` |
| 2.1 / 2.2 analogues | acc + accdelta, ENL arm | `lp_final_render.py decoders --window 1.0|1.5|2.0 --source daq|lp --trained daq|lp` |
| (comparison) | current LP arm decoders per animal | `lp_final_render.py compare --window 1.0` |

## 3. Env (set by the driver)
`WIDEFIELD_LICK_SOURCE=lp WIDEFIELD_GATE_SOURCE=lp WIDEFIELD_PRE_FROM=0801 WIDEFIELD_MAP_REFS="mean rest restw precue
raw restnl" WIDEFIELD_MAP_JOBS<=4 WIDEFIELD_PRM_LEGACY_DATE8=1` (+ ENL window / epoch frozen / LP frozen for decoders).
Gate-LP session-cache kinds carry `-glp<sha>`, so pixel-map (`prm`) and categorize caches are COLD; trial features
(`tf-*`) are shared.

## 4. Run order (shared box: <= 4 workers total, <= 2 heavy processes, free RAM > 25 GB; only own PIDs stopped)
1. `restnl` gate-LP baselines (3 workers) || DAQ-source decoder fits 1.5 s (1 process).
2. DAQ-source fits 1.0 / 2.0 s; then `maps` (4 workers).
3. LP decoders: DAQ-trained and LP-trained x 1.0 / 1.5 / 2.0 s; `compare 1.0`.
4. drift (4 workers, after maps: shares the glp prm caches, adds June); attempt maps restnl + rest.
5. Look at every figure; per-animal tables; this doc.

## 5. ENL trial counts (pre-check, census of all cues before the engagement gate; August pre)
Source: `enl_lp_census.csv` (10-21, the precue window rule applied to the LP train), curated sessions, current epochs.
Keep fraction (kept trials / all cues), far contra:

| animal | epoch | DAQ 2 s kept (n) | DAQ 2 s (frac) | LP 2.0 s | LP 1.5 s | LP 1.0 s |
|---|---|---|---|---|---|---|
| PS92 | pre | 502 | .99 | .86 (434) | .96 (486) | 1.00 (502) |
| PS92 | acute | 444 | 1.00 | .66 (291) | .86 (380) | .99 (441) |
| PS92 | subacute | 89 | 1.00 | **.53 (47)** | .79 (70) | .94 (84) |
| PS92 | chronic | 662 | 1.00 | .78 (514) | .93 (617) | .98 (652) |
| PS93 | pre | 583 | .94 | .71 (440) | .93 (577) | .99 (619) |
| PS93 | acute | 384 | 1.00 | .66 (253) | .82 (314) | .97 (373) |
| PS93 | subacute | 265 | 1.00 | .72 (191) | .87 (232) | .98 (262) |
| PS93 | chronic | 1002 | 1.00 | .87 (875) | .95 (956) | .99 (996) |
| PS94 | pre | 758 | .99 | .84 (643) | .93 (712) | .98 (743) |
| PS94 | acute | 605 | 1.00 | .83 (504) | .93 (564) | .99 (597) |
| PS94 | subacute | 500 | 1.00 | .70 (348) | .86 (430) | .96 (480) |
| PS94 | chronic | 800 | 1.00 | .74 (591) | .87 (694) | .96 (772) |
| PS95 | pre | 927 | 1.00 | .98 (905) | .99 (922) | 1.00 (927) |
| PS95 | acute | 120 | 1.00 | .91 (109) | .96 (115) | 1.00 (120) |
| PS95 | subacute | 703 | 1.00 | .93 (651) | .97 (682) | 1.00 (701) |
| PS95 | chronic | 1100 | 1.00 | .77 (845) | .88 (970) | .96 (1058) |

All positions: LP 1.5 s keeps >= .91 of every animal x epoch; LP 2.0 s .71-.98. Decision: 1.0 and 1.5 s are clean;
**2.0 s is run but flagged** -- enough trials in every cell (min 47, PS92 subacute far contra, 2 sessions), but the loss
is position- and epoch-dependent (acute far contra .66 in PS92 / PS93 vs ~.85 pre), i.e. a selection that tracks the
deficit. Post-gate counts per animal x epoch from the decoders' own trial sets are in section 7.

## 6. Status / log
(times EST, 11-02)
- 01:05 start; inventory, worktree, code changes, tests.
- 01:14 ruff clean on changed files (one pre-existing B023 in grant_confusion at HEAD); `tests/test_lp_gate_option.py` +
  `tests/test_epoch_figures.py`: 71 passed (4 min 55 s). Legacy digest check: 81053b92244b in both trees.
- 01:16 launched: gate-LP RESTnl baselines (3 workers) + DAQ-source 1.5 s decoder fit (cold features, ~13 min / animal).
- 01:33 queued (scratchpad `lpfinal/chain_maps.sh`, `chain_dec.sh`): maps (3 workers) -> drift -> attempt maps restnl /
  rest; decoders one process at a time (DAQ 1.0, DAQ 2.0, then LP x DAQ-/LP-trained x 1.0 / 1.5 / 2.0, compare 1.0).
- 01:34 gate-LP RESTnl baselines done (18.1 min): 120 sessions all ok; frame counts equal to the DAQ-gated ones in 91,
  differ in 16 sessions (= the 16 sessions whose gate status the switch moves, LP_GATE_OPTION table); 12 June + PS93_0813
  not in restw_check.
- 01:47 DAQ-source 1.5 s fit + figure (30.7 min, cold features); 1.0 s and 2.0 s DAQ fits 4 min each (warm). First
  figure looked at: stars in the top band OK; title wrapped at "DAQ-trained" -> wording changed to "trained on DAQ pre
  trials", left spine bounded to the data range (`--plot-only` redraw verified, 7 s).
- 02:03 maps precue done (28.8 min, cold glp caches). PS93_0813 (no video) is out of the LP arm, as before (107 sessions).
- 02:08 BUG caught in my own first LP DAQ-trained run: PS93 dropped ("none stored") because the DAQ model's training set
  includes PS93_0813 (pre-stroke for PS93, no video) and the LP bundle cannot pool it -> different spec id. Fix: under
  LP with DAQ training the spec names the DAQ arm's training set (every pre session of the animal). Stopped my chain
  (PIDs 17900, 22664, 3932 only), restarted from the LP steps at 02:09; 02:25 PS92-94 all frozen-hit.
- 02:30 LP 1.0 s DAQ-trained done (21 min); 02:52 LP 1.0 s LP-trained done (22 min; status "SPEC-CHANGED" is the
  sibling DAQ spec being reported, the LP-trained model is frozen under its own id, nothing superseded).
- 02:33 the corner stamp ran off the left edge of the bar figures -> wrapped to the canvas in `epoch_figures._save_png_svg`
  (NOT in `lp_licks.stamp`: that module's source keys the LP lick-times sidecars; an edit there was made and reverted).
- 02:58 maps done (84 min: precue 28.8, cue 30.5, lick 24.8). All 34 figures looked at (contact sheets).
- 03:30 drift done (32 min, 8 figures, all looked at). Attempt maps started.

## 7. Results (per animal first; far contra; "current" = the LP-arm versions on M: today)

Tables: `<G>/LPfinal/compare/lpfinal_vs_current_{maps,decoders}_farcontra.csv` (`scripts/lp_final_compare.py`).

### 7a. Maps (amplitude ratio epoch / pre, RMS of the mean map in the brain mask, within animal)
Current = `G/LP_licks/epoch_restnl` (11-01: LP licks, LP-only imaging gate, DAQ-only rest gate, Aug pre). Same
sessions; the only differences are the rest-baseline gate (RESTnl in 16 sessions, REST / RESTW time-local / flat gates)
and the S28 quit (not used by maps).

| window | ref | epoch | PS92 | PS93 | PS94 | PS95 | <1 | current PS92-95 | current <1 |
|---|---|---|---|---|---|---|---|---|---|
| cue (7.10) | RESTnl | acute | .56 | .54 | .60 | .43 | 4/4 | .56 .53 .60 .43 | 4/4 |
| cue | REST | acute | .58 | .51 | .50 | .43 | 4/4 | identical | 4/4 |
| cue | RESTnl | chronic | 1.02 | 1.21 | .71 | 1.20 | 1/4 | 1.02 1.20 .71 1.21 | 1/4 |
| lick (7.18 / 7.19) | RESTnl | acute | .50 | .55 | .82 | 1.07 | 3/4 | .50 .55 .82 1.07 | 3/4 |
| lick | REST | acute | .48 | .53 | .67 | 1.04 | 3/4 | identical | 3/4 |
| lick | RESTnl | chronic | 1.03 | 1.16 | .72 | 1.15 | 1/4 | identical | 1/4 |
| precue 1 s (7.3) | RESTnl | acute | 1.55 | .84 | .93 | 1.38 | >1: 2/4 | 1.55 .83 .94 1.38 | 2/4 |
| precue | RESTnl | subacute | .88 | .71 | .65 | 1.13 | | .88 .70 .61 1.13 | |
| precue | RESTnl | chronic | 1.68 | .91 | .98 | 1.55 | <1: 2/4 | 1.68 .92 1.02 1.56 | 1/4 |

Pooled (15r subtitle, acute / pre): cue RESTnl far contra .52 (current .52), lick .72 (.72), precue 1.60 (1.59).
Drift (June -> Aug, all-pre by construction; `drift_vs_stroke_panels.csv`): RESTnl cue .93 vs acute .52, lick 1.03 vs .72,
precue .78 vs 1.60; REST cue .94 / .49, lick 1.04 / .67; PRECUE cue 1.11 / .18, lick 1.24 / .56 -- every far-contra
panel within .015 of the current drift figures.
Attempt maps (8.1 / 8.2, lick, whole-brain mean x1e3, acute failed attempt - pre hits): PS92 -11.7, PS93 -10.1, PS94
-4.0, PS95 +1.8 (current -11.8 / -10.2 / -3.8 / +1.8); minus pre failed attempts -22.2 / +4.8 / -10.1 / +11.3 (current
-22.2 / +4.6 / -9.8 / +11.3). 3/4 below pre hits, 2/4 below pre failed attempts -- unchanged.

### 7b. ENL decoders (far-contra accuracy, chance .167; pre = LOSO within animal; trials = far-contra trials scored)

| arm | trained on | window | pre PS92-95 | acute PS92-95 | acute < pre | acute n far contra |
|---|---|---|---|---|---|---|
| CURRENT page (LP_licks, June + Aug pre, in-memory = LP-trained) | LP | 1.0 | .29 .33 .54 .40 | .15 .16 .29 .15 | 4/4 | 343 289 448 120 |
| LP + LP gate, Aug pre | DAQ | 1.0 | .34 .38 .51 .42 | .11 .14 .31 .17 | 4/4 | 344 289 448 120 |
| LP + LP gate, Aug pre | LP | 1.0 | .34 .38 .50 .42 | .12 .15 .28 .15 | 4/4 | same |
| LP + LP gate, Aug pre | DAQ | 1.5 | .44 .48 .57 .43 | .15 .14 .29 .22 | 4/4 | 281 230 415 115 |
| LP + LP gate, Aug pre | LP | 1.5 | .46 .49 .57 .44 | .14 .15 .27 .21 | 4/4 | same |
| LP + LP gate, Aug pre | DAQ | 2.0 (flagged) | .61 .56 .62 .44 | .16 .15 .35 .29 | 4/4 | 196 169 356 109 |
| LP + LP gate, Aug pre | LP | 2.0 (flagged) | .59 .54 .60 .45 | .13 .12 .30 .28 | 4/4 | same |
| DAQ contact, Aug pre (reference) | DAQ | 1.0 | .33 .41 .52 .42 | .10 .12 .34 .20 | 4/4 | 336 292 452 120 |
| DAQ contact, Aug pre (reference) | DAQ | 1.5 | .43 .46 .57 .44 | .13 .09 .35 .24 | 4/4 | |
| DAQ contact, Aug pre (reference) | DAQ | 2.0 | .58 .53 .60 .45 | .16 .09 .41 .34 | 4/4 | |

The `compare` row reproduces the page's 2.2 / LP 1.0 s numbers (.29 -> .15, .33 -> .16, .54 -> .30, .40 -> .14; .29 /
.15 here by rounding), so the comparison is like for like. Per animal x epoch trial counts (all positions and far
contra) for every run: `<arm>/epoch/data/lpfinal_enl_trials_acc_per_animal.csv`. Post-gate trials at 2.0 s vs 1.0 s,
acute far contra: PS92 196 / 344 (57 %), PS93 169 / 289 (58 %), PS94 356 / 448 (79 %), PS95 109 / 120 (91 %); pre
421 / 487, 433 / 609, 573 / 673, 769 / 786 -- the census's position- and epoch-dependent loss, confirmed.

Pooled (all trials, acute vs pre, marks ** = Bonferroni / 18): far contra .42 -> .20 ** (1.0 DAQ-trained), .42 -> .19
** (1.0 LP-trained), .48 -> .21 ** / .49 -> .20 ** (1.5), .54 -> .26 ** (2.0); current page LP 1.0 .40 -> .21 **. The
other positions keep the current pattern (near ipsi / far ipsi ** and near middle / near contra * acutely; far middle not
significant) at 1.0 and 1.5 s.

## 8. Conclusions: what changes, what does not

1. **No map headline changes.** The LP-inclusive gate moves the LP-arm maps only through the rest-baseline gates (16
   sessions' RESTnl baselines; REST / RESTW engaged rest frames), and every far-contra per-animal amplitude is within
   .04 of the current LP versions: cue loss 4/4 (pooled .52), lick loss 3/4 (PS95 not; .72), precue rise 2/4 (1.60),
   PS94 chronic RESTnl ~.71 (flat-baseline caveat stands). Only count that flips: precue RESTnl chronic < 1 in 2/4
   (PS94 .98 vs 1.02 before) -- noise-level, not a conclusion. Drift figures within .015. Attempt maps unchanged (3/4
   below pre hits, 2/4 below pre failed attempts).
2. **ENL decoders: the acute far-contra pre-cue loss survives every definition** -- LP licks + LP gate + August pre, 4/4
   animals at 1.0, 1.5 and 2.0 s, DAQ-trained and LP-trained. With August-only pre the pre-stroke far-contra accuracy is
   a little higher in PS92 / PS93 (.34 / .38 vs .29 / .33 all-pre) and the acute values a little lower, so the per-animal
   drop is slightly larger than on the page (e.g. PS92 .34 -> .11 vs .29 -> .15).
3. **DAQ-trained vs LP-trained makes no material difference** (every per-animal far-contra pre / acute cell within .05;
   the largest gap anywhere is PS92 subacute at 2.0 s, .38 vs .30 on 47 trials; pooled within .02). The training
   set's lick definition is not what the LP-arm decoders measure. NEW FACT for the page: the current LP-arm decoder
   figures (2.4, the LP 1.0 s ENL comparison) were LP-trained, not DAQ-trained as the frozen-model docs imply (section 1).
4. **Window length: 1.5 s is the better ENL window under LP licks** -- it keeps >= .79 of far-contra trials in every
   animal x epoch (census) and raises pre-stroke far-contra accuracy (.42 -> .48 pooled), with the same acute loss (4/4).
   2.0 s raises pre accuracy further (.54) but keeps only 57-58 % of PS92 / PS93 acute far-contra trials (a selection
   that tracks the deficit): report as a sensitivity, not as primary.
5. Nothing found that contradicts a current page statement; no headline result needs replacing. Page additions are
   NEW versions next to the existing figures (Priya's rule for tonight).

## 9. Outputs (all NEW; G = `M:/MICROSCOPE/Priya/Widefield/labcams/grant_figures`, E = `M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding`)

Every figure has figdata beside it (`data/`): maps `<name>_bundle.{npz,json}` + `_meta` / `_stats` / summary CSVs; bar
figures `<name>_figdata.pkl` + `.csv` / `_sessions.csv` / `_meta.csv`; attempt maps `maps_<align>_<animal>.npz` +
`counts_*` / `hemispheres_*`. Redraw: `python scripts/lp_final_render.py maps --plot-only`, `... decoders --source S
--trained T --window W --plot-only`, `map_drift_prepre.py --arm lp --out G/LPfinal/map_drift_prepre --plot-only` (env:
WIDEFIELD_GATE_SOURCE=lp), `python -m wfield_local.attempt_maps --rest-variant V --out <dir> --plot-only` (env: LP source,
LP gate, PRE_FROM=0801 -- the title reads them).

| page fig (current) | final version |
|---|---|
| 7.3 map_REST_precue_LP | `G/LPfinal/LP_licks_gateLP/epoch/epoch_15r_position_RESTNLref_precue_working.png` (+ `RESTref`, `RESTWref`, `RAWref`, `MEANref` twins) |
| 7.4 drift precue LP | `G/LPfinal/map_drift_prepre/LP_licks_gateLP/epoch_15rdrift_RESTNLref_precue_working.png` (+ `RESTref`) |
| 7.10 map cue LP | `.../epoch/epoch_15r_position_RESTNLref_cue_working.png` (+ REST / RESTW / RAW / MEAN) |
| 7.11 drift cue LP | `.../map_drift_prepre/LP_licks_gateLP/epoch_15rdrift_RESTNLref_cue_working.png` (+ `RESTref`) |
| 7.14 map PRECUE cue LP | `.../epoch/epoch_15r_position_PRECUEref_cue_working.png` |
| 7.15 drift PRECUE cue LP | `.../epoch_15rdrift_PRECUEref_cue_working.png` |
| 7.18 map lick LP | `.../epoch/epoch_15r_position_RESTNLref_lick_lick.png` (+ REST / RESTW / RAW / MEAN) |
| 7.19 lick LP by animal | `.../epoch/epoch_15rpa_position_RESTNLref_by_animal_lick_lick.png` (+ every ref; 15rpa also for precue / cue) |
| 7.20 drift lick LP | `.../epoch_15rdrift_RESTNLref_lick_lick.png` (+ `RESTref`) |
| 7.23 / 7.24 PRECUE lick LP | `.../epoch/epoch_15r_position_PRECUEref_lick_lick.png`, `.../epoch_15rdrift_PRECUEref_lick_lick.png` |
| 8.1 a_maps | `E/attempt_maps_LPfinal_restnl/figures/attempt_maps_lick_far_contra_restnl.png` (REST twin: `E/attempt_maps_LPfinal_rest/figures/attempt_maps_lick_far_contra.png`) |
| 8.2 a_maps_minus | same folders, `..._minus_pre_restnl.png` / `..._minus_pre.png` (+ cue / precue windows, far_mid / far_ipsi) |
| 2.2 analog (ENL 1.0 s) | `G/LPfinal/trained_daq/LP_licks_gateLP/epoch/epoch_acc[delta]_by_position_precue_working.png` (DAQ-trained); `G/LPfinal/trained_lp/LP_licks_gateLP/epoch/...` (LP-trained) |
| ENL 1.5 s | `G/LPfinal/trained_{daq,lp}/LP_licks_ENL1.5s_gateLP/epoch/...` |
| 2.1 analog (ENL 2.0 s, flagged) | `G/LPfinal/trained_{daq,lp}/LP_licks_ENL2s_gateLP/epoch/...` |
| DAQ-contact references (Aug pre) | `G/LPfinal/trained_daq/DAQ_ENL{1,1.5,2}s/epoch/...` |
| per-animal tables | `G/LPfinal/compare/lpfinal_vs_current_{maps,decoders}_farcontra.csv`; maps `.../epoch/data/lpfinal_amplitude_per_animal_*.csv`; current LP decoders `G/LPfinal/LP_licks/compare/current_LParm_enl1s_acc_per_animal.csv` |

Caches / stores written: `C:/Users/SabatiniLab/.widefield_session_cache_restnl_glp3e0db850/` (120 gate-LP RESTnl
baselines, 118 MB); session-cache entries tagged `-glp3e0db850` (prm / categorize / bm); `C:/Users/SabatiniLab/cue_lick/
frozen_models_lpfinal/` (24 `epoch_decoder` models: 4 animals x {DAQ, LP} x {1.0, 1.5, 2.0} s, 5.7 MB). Nothing
existing was overwritten, deleted or renamed.

## 10. Timings (shared box; <= 4 workers; free RAM >= 30 GB throughout)

| step | wall |
|---|---|
| gate-LP RESTnl baselines, 120 sessions, 3 workers | 18.1 min |
| DAQ decoders 1.5 s (cold features) / 1.0 s / 2.0 s (warm) | 30.7 / 4 / 4 min |
| maps 15r + 15rpa, 6 refs, 3 windows (cold glp caches, 3 workers) | 84 min (precue 28.8, cue 30.5, lick 24.8) |
| drift, 3 refs x 3 windows (June cold) | 32 min |
| attempt maps RESTnl / REST (3 windows x 4 animals) | 42 / 56 min |
| LP decoders 1.0 s DAQ- / LP-trained | 21 / 22 min |
| LP decoders 1.5 s DAQ- / LP-trained | 39 / 27 min |
| LP decoders 2.0 s DAQ- / LP-trained | 52 / 20 min |
| current-arm comparison (warm) | 1 min |
| tests (71 + 144) | 5 + 4 min |
| total wall, start of renders -> last figure | 01:16 -> 05:10 (3 h 54 min) |

## 11. Not done / open

- Imaging-page edits and publishing (Priya's / the caller's job; new versions to go next to the existing ones).
- The 14m (MEAN-ref beta maps) and 5c / 5cr (confusion) families of the old LP arm were NOT re-rendered (not page
  families); the lick- and cue-window decoders (2.3 / 2.4 analogues) were not re-rendered (task asked for the ENL ones).
- ENL maps at 1.5 / 2.0 s not rendered (maps stay on the LP 1.0 s window).
- Session markers on the bar figures are still the old style (filled dots colored by animal), not the results-page
  convention (open markers, shape = animal, color = epoch). A renderer change across every bar figure; not attempted.
- `WIDEFIELD_EPOCH_FROZEN` is opt-in; the default epoch render still fits its "frozen" decoders in memory (rule-10
  deviation, pre-existing). Decision for Priya: make the store the default.
- The epoch acc stars moved to the top band for every bar figure rendered from this tree (Priya's 10-30 rule); the
  canonical renders on main are unchanged until merged.
- Some accdelta titles wrap with an orphan word ("LP\nattempt"); cosmetic.
