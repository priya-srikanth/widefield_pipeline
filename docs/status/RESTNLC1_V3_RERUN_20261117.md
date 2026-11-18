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
