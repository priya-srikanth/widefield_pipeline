# RESTnlc1 v3 re-run (11-17 evening)

**What changed.** `wfield_local/restnl_licks.py` v3 (3bf2dd41): cam1-only events with `med_out_px < -120` (static
far-from-mouth tracking labels) no longer count as licks (Priya's blind review, `CAM1_EXTRA_LICKS_CHECK_20261117.md`).
CACHE_TAG restnlc1v2 -> restnlc1v3, CHECK_DIR restw_check_c1v2 -> restw_check_c1v3. Every RESTnlc1 output from the
14:38-18:01 v2 queue run (a1-a9, b01-b13, c01, d1-d3; b14 + c02-c15 never completed) is superseded.

**Code (4d77936d).** No `restnlc1v2` literal is left: `rest15f_restnl_1117.py` c1all mask = `f"{CACHE_TAG}wk"`
(restnlc1v3wk), `restnl_cam1_compare.py` and `tests/test_restnl_cam1.py` read `restnl_licks.CACHE_TAG`. 49 restnl /
rest-frozen tests pass.

## Stale-read audit (what a re-run would have reused)

Method: a scan of every file on `rebuild_20261103/` (and the input cache / scratch) written 14:30-18:10 today, plus a
name scan for `restnlc1` / `_c1` anywhere on the rebuild root, read against each step's log header and script.

Keyed on the v3 tag / digest, so NOT moved (left in place, never read by v3):
- `session_cache_restnlc1v2/`, `session_cache_restnlc1v2_glp3ac97bde/` (a1 / a2; v3 builds `session_cache_restnlc1v3*`)
- `iti_licks/<HV>/restw_check_c1v2/` (b01; v3 = `restw_check_c1v3`)
- `figures_working/rest_decoder_nl_cache/restnlc1v2wk/` and `frozen_models/PS9x/rest_frozen_rest-restnlc1v2wk-*` (a8 / a9;
  v3 = `restnlc1v3wk`, a new frozen-store prefix)
- the v1 leftovers `session_cache_restnlc1/` (empty), `restw_check_c1/` and the earlier cam4cam1 (post-cue) set
  `restnlc1wk` (cache + frozen models + `rest15f_restnl_20261117/*restnlc1wk*`): other definitions, not read.

NOT keyed on the version (a re-run would skip / reuse them) -> MOVED ASIDE:
- per-session caches with the unversioned `_c1` / `_restnlc1` names: `precue_restnl_wr_c1` (b02), `precue_restnl_traces_wr_c1`
  (b03), `precue_enl_lp1s_c1` + `precue_enl_traces_lp1s_c1` (b04 / b05), `cue_to_lick/sessions_restnl_wr_c1` (b07),
  `disengagement_restnlc1/` (b11 / b12, sessions/ = cache), `interhemi_rest_w1_restnlc1` (b13), `attempt_maps_restnlc1`
  (a5, csig key carries no lick-set version), `state_space_recovery/*_restnlc1` (c01 skips existing `<label>.npz`);
- `frozen_models/PS9x/rest_frozen_rest-restnlc1-b4-*` (a6 / a7): the store matches on spec + pre-session set, not on the
  rest mask, so v3 a6 would have LOADED the v2 aug-pre models;
- `rest_decoder_nl_cache/restnlc1` (a6 / a7; digest-keyed incl. `restnl_licks.code_digest`, moved anyway for safety);
- unversioned outputs: LPfinal `..._refs-restnlc1` (a3, holds `lpfinal_maps_done.csv`, a done-list), map_drift (a4),
  `rest_frozen_decoder_restnl/<HV>/restnlc1_*pre` (a6 / a7), `rest15f_restnlc1all_20261117` (a8 / a9), and every
  `*restnlc1*` table / figure / figdata in iti_licks (b02-b10, b14 partial), revup, stopped_traces, cue_to_lick, LP_licks;
- lane D scratch (`<scratch>/rc1/h2h_c1`): `rest_sessions/*.pkl` checkpoints (d1 skips existing) and the d1-d3 outputs.

Note: `lick_cue_to_lick` names its tau-0.6 figdata with `Path.with_suffix`, so `..._tau0.6_restnlc1.pkl` lands as
`..._tau0.pkl` (shared with the RESTnl / RESTW runs, last writer wins). The four v2 `*_tau0.pkl` were moved with the rest;
the bug is pre-existing and not fixed here.

Queue markers: every a/b/c/d step's `.done/.failed/.skipped/.log/.check/.winpid` -> `<scratch>/rc1/logs/v2_superseded/`
(STOP -> `v2_superseded/STOP_v2_1801.txt`); lanes relaunched 18:11 (A C, B, D).

## Moves (all within Priya's rebuild root / our scratch; nothing deleted)

R = `M:/MICROSCOPE/Priya/Widefield/rebuild_20261103`, IE = `R/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding`,
HV = the `hv-meegkit..._daqlk` run tag (`pre0801` form; HVnp = without it).

| from (moved INTO `<parent>/_superseded_restnlc1v2_20261117/`) | files | MB |
|---|---:|---:|
| `R/IE/attempt_maps_restnlc1` | 24 | 226.5 |
| `R/IE/HV/disengagement_restnlc1` | 274 | 53.5 |
| `R/IE/iti_licks/LP_licks_HV/precue_enl_lp1s_c1` | 107 | 27.8 |
| `R/IE/iti_licks/LP_licks_HV/precue_enl_traces_lp1s_c1` | 107 | 8.3 |
| `R/IE/iti_licks/HV/cue_to_lick/sessions_restnl_wr_c1` | 107 | 1090.2 |
| `R/IE/iti_licks/HV/interhemi_rest_w1_restnlc1` | 107 | 26.3 |
| `R/IE/iti_licks/HV/precue_restnl_traces_wr_c1` | 107 | 8.5 |
| `R/IE/iti_licks/HV/precue_restnl_wr_c1` | 107 | 53.9 |
| `R/figures_working/rest_decoder_nl_cache/restnlc1` | 120 | 32.4 |
| `R/labcams/grant_figures/LPfinal/LP_licks_gateLP_HV_refs-restnlc1` | 93 | 285.6 |
| `R/labcams/grant_figures/HV/epoch/rest15f_restnlc1all_20261117` | 64 | 1.3 |
| `R/labcams/grant_figures/map_drift_prepre_restnlc1_20261117` | 22 | 67.9 |
| `R/labcams/grant_figures/rest_frozen_decoder_restnl/HVnp/restnlc1_allpre` | 13 | 0.6 |
| `R/labcams/grant_figures/rest_frozen_decoder_restnl/HVnp/restnlc1_augpre` | 13 | 0.5 |
| `R/state_space_recovery/chronic_maps_lick_svd_rest_restnlc1` | 0 | 0.0 |
| `R/state_space_recovery/chronic_maps_svd_rest_precue_restnlc1` | 0 | 0.0 |
| `R/state_space_recovery/chronic_maps_svd_rest_restnlc1` | 0 | 0.0 |
| `R/state_space_recovery/page_figs_restnlc1` | 0 | 0.0 |
| `R/state_space_recovery/rest_baseline_svd_restnlc1` | 12 | 0.0 |
| `R/state_space_recovery/shift_precue_lick_svd_restnlc1` | 1 | 0.0 |
| `R/state_space_recovery/shift_rawref_lick_svd_restnlc1` | 1 | 0.0 |
| `R/frozen_models/PS92/rest_frozen_rest-restnlc1-b4-dur10-gap10-gate_joint76d884_1ff5f9e84fa7` | 2 | 0.3 |
| `R/frozen_models/PS93/rest_frozen_rest-restnlc1-b4-dur10-gap10-gate_joint0141bb_bcb76877ca88` | 2 | 0.3 |
| `R/frozen_models/PS94/rest_frozen_rest-restnlc1-b4-dur10-gap10-gate_jointa5e70d_b1bba8472085` | 2 | 0.3 |
| `R/frozen_models/PS94/rest_frozen_rest-restnlc1-b4-dur10-gap10-gate_jointa5e70d_b41914498de4` | 2 | 0.2 |
| `R/frozen_models/PS95/rest_frozen_rest-restnlc1-b4-dur10-gap10-gate_jointa91701_926009262a98` | 2 | 0.3 |
| `R/frozen_models/PS95/rest_frozen_rest-restnlc1-b4-dur10-gap10-gate_jointa91701_aaae2cd5e576` | 2 | 0.3 |
| `<scratch rc1>/h2h_c1/rest_sessions` | 107 | 0.2 |

Loose v2 files (all named `*restnlc1*`, plus b08's `cue_to_lick/figdata/*_tau0.pkl`, see note), moved into a `_superseded_restnlc1v2_20261117/` beside them:

| folder | files | MB |
|---|---:|---:|
| `<scratch rc1>/h2h_c1` | 3 | 19.9 |
| `<scratch rc1>/h2h_c1/head_to_head` | 11 | 0.5 |
| `R/IE/iti_licks/LP_licks_HV` | 11 | 1.9 |
| `R/IE/iti_licks/LP_licks_HV/figdata` | 6 | 12.6 |
| `R/IE/iti_licks/LP_licks_HV/figures` | 8 | 3.7 |
| `R/IE/iti_licks/LP_licks_HV/figures/svg` | 8 | 4.6 |
| `R/IE/iti_licks/HV` | 15 | 8.1 |
| `R/IE/iti_licks/HV/cue_to_lick` | 28 | 126.5 |
| `R/IE/iti_licks/HV/cue_to_lick/figdata` | 23 | 2.5 |
| `R/IE/iti_licks/HV/cue_to_lick/figures` | 26 | 7.8 |
| `R/IE/iti_licks/HV/cue_to_lick/figures/svg` | 26 | 13.3 |
| `R/IE/iti_licks/HV/figdata` | 7 | 17.5 |
| `R/IE/iti_licks/HV/figures` | 26 | 11.9 |
| `R/IE/iti_licks/HV/figures/svg` | 26 | 20.9 |
| `R/IE/iti_licks/HV/revup` | 10 | 18.7 |
| `R/IE/iti_licks/HV/revup/figdata` | 1 | 13.9 |
| `R/IE/iti_licks/HV/stopped_traces` | 13 | 1.5 |
| `R/IE/iti_licks/HV/stopped_traces/figdata` | 19 | 4.6 |
| `R/IE/iti_licks/HV/stopped_traces/figures` | 20 | 11.8 |
| `R/IE/iti_licks/HV/stopped_traces/figures/svg` | 20 | 21.9 |

Total: 335 items, 1705 files, 2.21 GB. Full list (from, to, files, bytes): `docs/status/RESTNLC1_V3_RERUN_20261117_moves.tsv`.

## Queue outcome (v3, code 4d77936d / 259e95e9)

Launched 18:11. Lane A C: a1-a9 + c01-c15 all done (finished 21:53). Lane D: d1-d3 done (18:58). Lane B: b01-b18 done
(23:04); b19_cv_sm_plot (plot-only) held: another window's arhmm scoring jobs (8 x ~4 GB, not ours, not touched) took
free RAM to 1-2 GB at 22:51, the queue wrote STOP during b18 (nothing killed; b18 finished OK). A background waiter
re-runs `ONLY=b19_cv_sm_plot SECTIONS=B` once >= 12 GB is free (STOP -> `logs/STOP_lowram_b18_2251.txt`). No step failed.

## Old vs new

Tables (CSV, read-only scripts): `<G>/restnl_cam1_compare_20261117/frames_v3/`, `frames_v2/` (frames, `restnl_cam1_compare
frames`, CACHE_TAG patched to v2 for the v2 run), `v3/decoder_*_long.csv`, `v3/pairs_v3.csv`, `v3/flips_*.csv`
(scratch scripts `compare_v3.py`, `flips_v3.py`, `frames_ver.py`, run under the rebuild profile).

### 1. Rest frames removed by cam1 vs cam4-only RESTnl, per animal x epoch (pre = June + August)

The LP-gate and DAQ-gate restnl_mask caches carry the SAME key (the mask does not depend on the gate source), so the
two gates give identical frames; one table. "frames back" = lick-free rest frames v3 returns relative to v2.

| animal | epoch | sessions | removed v2 | removed v3 | max session v2 / v3 | frames back |
|---|---|---:|---:|---:|---:|---:|
| PS92 | pre | 11 | 2.4 % | 2.4 % | 4.7 / 4.7 % | 0 |
| PS92 | acute | 5 | 1.7 % | 1.7 % | 4.1 / 4.1 % | 18 |
| PS92 | subacute | 2 | 6.1 % | 4.4 % | 6.3 / 5.6 % | 59 |
| PS92 | chronic | 12 | 15.6 % | 12.8 % | 49.0 / 23.5 % | 935 |
| PS93 | pre | 10 | 13.1 % | 12.6 % | 28.0 / 28.0 % | 310 |
| PS93 | acute | 4 | 8.1 % | 7.7 % | 17.4 / 16.9 % | 117 |
| PS93 | subacute | 3 | 19.8 % | 19.7 % | 29.2 / 28.7 % | 20 |
| PS93 | chronic | 12 | 22.5 % | 21.8 % | 28.1 / 27.3 % | 368 |
| PS94 | pre | 11 | 6.5 % | 6.3 % | 11.8 / 11.8 % | 94 |
| PS94 | acute | 6 | 7.6 % | 6.4 % | 21.5 / 18.8 % | 749 |
| PS94 | subacute | 5 | 12.9 % | 9.9 % | 28.7 / 19.8 % | 682 |
| PS94 | chronic | 8 | 4.0 % | 3.9 % | 6.7 / 6.7 % | 55 |
| PS95 | pre | 11 | 3.6 % | 3.2 % | 10.4 / 8.4 % | 865 |
| PS95 | acute | 1 | 1.0 % | 0.7 % | 1.0 / 0.7 % | 71 |
| PS95 | subacute | 7 | 5.2 % | 4.0 % | 15.3 / 13.9 % | 1071 |
| PS95 | chronic | 11 | 10.6 % | 9.2 % | 18.3 / 18.3 % | 1513 |

(The v2 column reproduces the 11-17 afternoon table: PS92 2/2/6/16, PS93 13/8/20/23, PS94 7/8/13/4, PS95 4/1/5/11 %.)
The far-from-mouth guard mostly trims the worst sessions (PS92 chronic max 49 -> 24 %, PS94 subacute 29 -> 20 %); the
EPOCH DEPENDENCE stays (PS92 2.4 -> 12.8 %, PS93 ~13 -> 22 %), so RESTnlc1 still differs from RESTnl most in chronic
PS92 / PS93. No session changes wr status; every session has a cam1 events_all file.

### 2. Fig 12.1 rest frozen decoder (rest15f, c1all = RESTnlc1 rest), retained fraction per animal x epoch

All-pre (PRIMARY): cam4-only RESTnl / v2 / v3 (n sessions v3)

| animal | acute | subacute | chronic |
|---|---|---|---|
| PS92 | 1.01 / 0.99 / 0.99 (5) | 2.92 / 3.07 / 3.07 (1) | 1.34 / 1.48 / 1.40 (11) |
| PS93 | -0.23 / -0.09 / -0.09 (1) | 0.40 / 0.51 / 0.43 (3) | 0.55 / 0.52 / 0.52 (12) |
| PS94 | 0.39 / 0.38 / 0.41 (6) | 0.51 / 0.41 / 0.43 (5) | 0.27 / 0.12 / 0.14 (8) |
| PS95 | 0.29 / 0.10 / 0.13 (1) | 0.67 / 0.69 / 0.70 (7) | 0.63 / 0.53 / 0.55 (11) |

Cohort epoch change (delta above-chance fraction vs own pre, nested bootstrap 95 % CI, "*" = CI excludes 0):

| epoch | cam4 RESTnl | v2 | v3 |
|---|---|---|---|
| acute | -0.118 [-0.186, -0.036] * | -0.125 [-0.195, -0.039] * | -0.120 [-0.188, -0.037] * |
| subacute | -0.031 [-0.116, 0.089] | -0.030 [-0.121, 0.092] | -0.030 [-0.118, 0.092] |
| chronic | -0.071 [-0.140, -0.001] * | -0.084 [-0.167, 0.004] | -0.081 [-0.153, -0.005] * |

Aug-pre (PS94, PS95 only; PS92 / PS93 refused by the fairness gate):

| animal | acute | subacute | chronic |
|---|---|---|---|
| PS94 | 0.61 / 0.56 / 0.62 | 0.66 / 0.65 / 0.85 | 0.18 / 0.23 / 0.28 |
| PS95 | 0.21 / 0.20 / 0.19 | 0.67 / 0.54 / 0.56 | 0.43 / 0.41 / 0.40 |

| epoch | cam4 RESTnl | v2 | v3 |
|---|---|---|---|
| acute | -0.122 [-0.213, -0.007] * | -0.129 [-0.223, -0.014] * | -0.122 [-0.223, -0.003] * |
| subacute | -0.064 [-0.118, -0.004] * | -0.081 [-0.148, -0.006] * | -0.063 [-0.139, 0.017] |
| chronic | -0.128 [-0.175, -0.081] * | -0.125 [-0.185, -0.064] * | -0.118 [-0.177, -0.056] * |

12.1b (rest_frozen_decoder_nl, engaged brackets; cam4 exists for aug-pre only): all-pre v2 -> v3 acute -0.115* -> -0.113*,
subacute -0.026 -> -0.033 (ns both), chronic -0.081* -> -0.084*. Aug-pre cam4 / v2 / v3: acute -0.109* / -0.112* / -0.093*,
subacute -0.075* / -0.088* / -0.066*, chronic -0.090* / -0.121* / -0.114*. Per animal: `v3/decoder_per_animal_long.csv`.

### 3. Everything else (pairs of `*_restnl.csv` vs `*_restnlc1.csv`, v3 vs cam4 and vs v2)

No significant cell changes SIGN in any table (v3 vs cam4 or v2). CI-edge flips only, deltas nearly equal:
- pre-cue level changes (RESTW-window and ENL, all / far contra; 270 cells each): v3 vs cam4 gained 1-4 / lost 3-11
  stars per table (e.g. ENL pre-cue PL L/R subacute, VISa / VISpm R chronic lose their star, deltas within 0.1); v3 vs v2
  gained 0-3 / lost 0-1. No contralesional (right) ALM / tjM1 cell flips; two ipsilesional (left) cells gain a star:
  far-contra ENL tjM1_left acute (5.54 -> 5.62) and ENL ALM_left precue05 chronic (1.02 -> 1.05).
- cue-to-lick A / C / D: v3 vs cam4 lost 15 / 18 / 15 and gained 7 / 8 / 4 of ~1,200-2,000 cells; vs v2 mostly gains.
- interhemispheric tests (7,830 cells; no v2, b14 never finished under v2): v3 vs cam4 gained 106 / lost 176 CI-excludes-0
  flags, no sign flips; most in asym / lag_peak / gc_ic / gc_ci acute-subacute.
- LPfinal 15r / map drift stats: n-significant-bin counts per panel moved by up to 10-19 bins v3 vs v2 (no star rows).
- Fig 14.20 head-to-head (lane D): v2 -> v3 max |diff| 0.06, no p crosses 0.05, per-animal sign counts identical.
- State space (c lane) and disengagement: no cam4 twin with the `_restnl` naming / no v2 run; not compared here.

### Conclusions that change vs cam4-only RESTnl

1. Fig 12.1 aug-pre SUBACUTE drop loses its star under RESTnlc1 v3 (cam4 -0.064 [-0.118, -0.004] * -> v3 -0.063 [-0.139,
   0.017]); same point estimate, wider CI (2 animals). v2 still had the star.
2. Fig 12.1 all-pre CHRONIC drop keeps its star under v3 (-0.081 [-0.153, -0.005] *); v2 had lost it ([-0.167, 0.004]),
   so the v2 "chronic not significant" reading was driven by the far-from-mouth labels.
3. No sign change anywhere; all other changes are single cells at the CI edge (listed in `v3/flips_cells.csv`).
