# Efficiency audit, 2026-11-06 (read-only; a8 helper)

Priya 11-06: "ensure code is also as efficient as possible so work isn't re-done / redundant and is appropriately
parallelized". Read-only pass over main; known open items from CODE_REVIEW_2026-11-04 / STATUS_2026-09-21_ENGINEERING
excluded (serial `channel_evoked_sign` / `rest_baseline_epoch_drift`, `_COST_HINT`, `session_signature` not hashing yaml,
parent-only BLAS pin). Timings measured on PS94_0820 where given. Nothing changed yet; every fix below is proposed.
Side effect of the audit: one timing call wrote a local `tf-lick` entry to `~/.widefield_session_cache` (derived, local).

## Ranked by wall-clock saved (all zero number change unless stated)

| # | where | what is redundant / serial | saving | fix |
|---|---|---|---|---|
| 1 | `locanmf_position_encoder.py:320,396,626,742` | plain `_trial_features` (not `trial_features_cached`), ~17 s/session x ~4 calls x 122 pooled sessions in EVERY date's encoder; pooled FEVE/EV figures redrawn per date (concurrent writes on backfill) | ~2 h / night (critical path) | use the cached call, build once, draw pooled figures from the last date only |
| 2 | `nightly_figs.py:657-669` | `pooled_frozen_loso` and `pooled_frozen_encoder` each `pool_sessions(source="roi")`; ROI features never cached (`locanmf_position_decoder.py:328`); ~19 s x 122 x 2 x 3 aligns, serial | 2-3.5 h | one pool per (animal, align) shared by both; fan the 12 units; later key ROI cache on `U_atlas` |
| 3 | `effective_kind` (`tf-` kinds) | `sel-` / `daqlk` / `csig-` parts in trial-feature cache kinds although `_trial_features` reads no trial selection; each stop-rule arm starts cold; cohort-wide `csig` invalidates all 122 sessions when one re-runs | ~2-3 CPU-h per arm / class-table touch | strip those parts from `tf-` kinds (pin equality test default vs stopA0); per-session `csig` (check class-table dependence first) |
| 4 | `run_mode.py:177-185` | `lickdet` part turns on with ANY tag part, incl. pooling-only `pre`, so a plain nightly (`WIDEFIELD_PRE_FROM=0801`) gets cache tag `daqlk` (0 entries locally: fully cold) and a new tag folder that makes `_perday_figs_incomplete` backfill all 45 dates | hours on the next plain nightly | drop `lickdet` from the CACHE tag (detector already covered by CACHE_VERSION 14); output-folder naming untouched |
| 5 | `nightly_figs.py:854-872` | post-stroke steps serial; `hemispheric_dynamics.collect` uncached serial (~17 min); `vessel_contrast` Frangi twice per channel | ~20-25 min | `_cli_many` for independent steps; fan + cache `session_dynamics`; Frangi once |
| 6 | `flatpool_vs_restw.py:81`, `chronic_stability.py:126` | uncached `session_raw_maps` (~13 s/session) where `session_raw_maps_cached` exists | ~20 min | call cached / `map_collect.collect` (confirm post_s 2.0) |
| 7 | `nightly_figs.py:1019` | nightly epoch render without `--parallel-arms` | 40-60 min (~18 GB RAM at 3 arms) | pass `--parallel-arms 3` (RAM permitting on the shared box) |
| 8 | `_load_daq_events` (~45 sites) | DAQ .h5 re-read every call (2.75 s warm / 9.6 s cold); `engagement.session_trials` opens it ~5x; `quit_prodrome`, `lick_bout_structure` rebuild `session_behavior` | 10-30 CPU-min / render | per-session memo keyed on h5 stat + detector + LP tag |
| 9 | `grant_figures._collect_7` | per-process `lru_cache`; units ordered by cost, so up to 8 workers rebuild one collection | not measured | disk cache, or group units by (align, variant) |
| 10 | `locanmf_cross_mouse`, `locanmf_rsa`, `enl_lick_control` | serial per-session loops on cold nights; `enl_lick_control` rebuilds `enl_decode`'s arms | cold nights | pre-warm the per-session cache with `fan_sessions` |

## Stale-hit risks (correctness; numbers move only if an entry is already stale)

- ROI-source caches miss Allen-registration changes: `session_signature` hashes `U.npy`, not `U_atlas.npy` / atlas / mask
  (`fixed_scale_maps.py:53,67`, `evoked_amplitude.py:83`, `spatial_reorganisation.py:69`, cd_trajectories ROI arms).
- Over-keying: `result_cache.code_version` hashes the whole module (`lp_licks.py:425`), `map_collect.code_digest`
  includes docstrings, so comment edits cause misses.

## Parallel use

- All 16 `fan_out` callers checked sort their results.
- `nightly_figs.cli()` (`:84`) does not pin BLAS for children (unlike `_cli_many`); fixing changes figure bytes
  (`test_render_is_machine_independent`), so NOT a pure efficiency fix.
- `iti_session_pattern_tests.py:342` `jobs=len(fams)` = 9 > cap 8.

## Proposed order

4, 1, 2, 6 (small, zero-change, several hours off a full nightly), then 3 and 5. Items 3 and 4 touch run tags /
cache kinds that sabatinilab-34's render uses: coordinate with 34 and land after its render (pinned worktree).
