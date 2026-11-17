# engaged_frame_mask camera-gate bug: which render steps it touches (read-only audit, 11-06 ~19:20)

Scope: every step in queue_daqlk7b, 8, 8b, 8c, 8d, 8e, 8e2, 8f, 8g, 8_s2, 9k, 9k2 (622 step lines). Code was read in
wf_render_v8 @ 5db29b07. `rest_engagement.py` and `position_reference_maps.py` are byte-identical in wf_render_daqlk
(a76352f3), v8d, trendfix, v8f, v9k and v9k2. `attempt_maps.py` differs only in v9k/v9k2, and no v9k step runs it.
Every step runs under profile cohort1_rebuild_20261103, which sets WIDEFIELD_TRIAL_GATE=camera, and no profile block
overrides it for these steps. The fix is ba5b63bf on fix/camera-rest-start-20261106. It touches only rest_engagement.py
and a test.

## What the bug did (from the logs)

In every affected run the time-local rest builder printed `!! time-local rest <label>: no engaged rest frames` for
exactly the 8 sessions. So `engaged & quiet` was EMPTY: no frames before the first cue were quiet. The builder then
returned None. The effect is that **those 8 sessions DROPPED OUT of the REST-referenced outputs**, rather than being
referenced to a baseline built from 200-800 frames. Two exceptions:
- gain_bio does not drop the sessions. It writes R5 / R5far / TASK sets that are empty or near-empty, and this is
  what led to the gb_stats assertion.
- PS93_0813 printed the same line once. That is expected: it has no camera working period (no video).

Epochs of the 8 sessions (configs/animals.yaml):
- pre (August): PS92_0813
- acute: PS92_0822, PS93_0818, PS94_0817, PS94_0820
- subacute: PS93_0822, PS93_0824, PS95_0818
- chronic: none

So a pre-only consumer would include only PS92_0813.

## Paths that reach engaged_frame_mask (whole-repo grep, followed to the entry points)

| Path | Reaches when | What changes |
|---|---|---|
| A. `position_reference_maps.session_rest_svt_timelocal` <- `session_raw_maps` <- `session_raw_maps_cached`/`_raw_maps_job` <- `prm.maps_by_epoch` <- `epoch_maps._fig_15r_reference_maps` / `_fig_15rpa_reference_by_animal` (epoch_grant_figures `15r`; scripts/restnl_maps.py; scripts/lp_final_render.py maps), plus scripts/map_drift_prepre.py | always (session_raw_maps always builds raw_rest) | only the **REST** (`_RESTref_`) reference column. MEAN / RESTW / PRECUE / RAW / RESTnl are unchanged. |
| B. `prm.session_raw_maps` called directly by scripts/chronic_stability.py `_session_map_amplitudes` | always | nothing: it reads only `reference_maps(parts, "precue")` |
| C. `attempt_maps.session_class_maps(rest_variant="rest")` -> `session_rest_svt_timelocal` | `--rest-variant rest` (default) | the rest-variant attempt maps; sessions with None are skipped |
| D. `cd_trajectories.rest_baseline` (rest and restw both AND with eng) <- `analyse_animal` (pre-stroke fit pass only) | `--reference rest` or `restw` only | not reached by any queued step (g5_cd_traj uses the default `contrast`) |
| E. `restnl_mask.measure_flat_vs_timelocal` <- `build_session(measure=True)` | `--measure` only | not reached (g2a/g2b run without `--measure`) |
| F. scripts/gain_bio/gain_bio.py `run()` (v8f/v9k/v9k2 only): `R5 = quiet & eng & ~bad`, `R5far`, `TASK &= eng` | always | the R5, R5far and TASK sets (ALL is unchanged) |
| (not queued) scripts/rest_migration/{flat_vs_timelocal, flatpool_vs_restw, rest_baseline_epoch_drift, reference_family_roi, meanref_balance, rest_null, subacute_daymatched}, map_test_calibration, restnl_position_coverage, cd_loo_sessions | n/a | no queued step runs them |

## Affected steps

| Queue | Step | Script + path | Status | Page / deck outputs | Do the 8 enter? |
|---|---|---|---|---|---|
| 7b | **g5_epoch** | `-m wfield_local.epoch_grant_figures --only ... 15r ...`, path A | **done** 11-06 01:59:57 (start 11-05 17:03, code 9aaa4cc6) | 15r / 15rpa = imaging-page position maps by reference. Only the RESTref panels are affected. The 15r family **crashed in the ENL (precue) and cue arms** with `TypeError tuple indices must be integers or slices, not str`, so RESTref was written only for the lick/lick arm: `epoch_15r_position_RESTref_lick_lick.png` and `epoch_15rpa_position_RESTref_by_animal_lick_lick.png`. That crash is separate and UNCERTAIN in cause; the lick arm ran with the same 8 missing. | YES. All 8 warned in the ENL, cue and lick arms, so they are missing from the REST column (pre and acute/subacute). Other families in this step are not affected. |
| 7b | g5_restnl_maps_contact | scripts/restnl_maps.py `--arm contact`, path A | **not started** (claim RESERVED_v7 only) | Imaging page 7.x RESTnl figures: unaffected (WIDEFIELD_MAP_REFS=restnl). Its `restnl_amplitude_per_animal_<align>_<variant>.csv` lists rest beside restnl/restw/raw, so the **REST rows of that CSV are affected**. | YES (August pre + all post) |
| 7b | g5_restnl_maps_lp | same, `--arm lp`, path A | **not started** (RESERVED_v7) | same as above (CSV REST rows only) | YES |
| 7b | g5_restnl_maps_enl1s | same, `--arm enl1s`, path A | **not started** (RESERVED_v7) | same as above (CSV REST rows only) | YES |
| 7b | **g5_lpfinal_maps** | scripts/lp_final_render.py maps, path A (refs = mean rest restw precue raw restnl) | **not started** (RESERVED_v7) | LP final-arm 15r / 15rpa **RESTref figures** + `lpfinal_amplitude_per_animal_*.csv` REST rows | YES (pre from 0801) |
| 7b | **g5_map_drift_contact** | scripts/map_drift_prepre.py `--refs rest precue restnl`, path A | **not started** (RESERVED_v7; dep g5_restnl_maps_contact) | `epoch_15rdrift_RESTref_*` figures + CSV | YES. PS92_0813 is in "late pre". The 4 acute sessions are in "acute - late pre". The chronic column has none of them. Block sets PRE_FROM null. |
| 7b | **g5_map_drift_lp** | same, `--arm lp` | **not started** (RESERVED_v7) | same as contact | YES |
| 7b | **g5_map_drift_enl1s** | same, `--arm enl1s` | **not started** (RESERVED_v7) | same as contact | YES |
| 7b / 8 / 8_s2 | g5_chronic_stability | `-m scripts.chronic_stability`, path B | **not started** (RESERVED_v8; v8 deps g5_epoch, g5_recovery_cue) | Calls engaged_frame_mask but uses only PRECUE maps, so **output unaffected** | YES (computed, then discarded) |
| 7b | **d_attempt_maps_rest** | `-m wfield_local.attempt_maps --rest-variant rest` (LP lick + LP gate), path C | **done** 11-06 09:22:24 (start 07:37, code a76352f3) | `attempt_maps/maps_<align>_<animal>.npz`, counts, `hemispheres_<align>.csv` | YES. All 8 were warned and skipped in the cue, lick and precue aligns. |
| 7b | **d_attempt_maps_plot** | `-m wfield_local.attempt_maps --plot-only` (default rest variant). Replots path-C products. | **done** 11-06 09:35 | Attempt-map figures (rest variant) | YES (downstream) |
| 8 / 8_s2 | d_fig_goal_summary_f3 | scripts/rebuild_figs/fig_goal_summary.py `--only fig3`. Reads `attempt_maps.out_dir()` (rest) `hemispheres_lick.csv`. Downstream of path C. | **held** (RESERVED_HOLD) | Goal-summary fig 3 (goal-question page) | YES (downstream) |
| 8 / 8_s2 | d_fig_goal_summary | aggregate marker of f1/f2/f3 | **held** (RESERVED_HOLD; a 09:28 log exists from an earlier v7b attempt) | marker only | n/a |
| 8g | **gb_sessions** | scripts/gain_bio/gain_bio.py `--jobs 2 --task-rows working`, path F | **done** 11-06 19:00 (start 17:54, code 110c4efd) | slide 102 (gain_bio) | YES. R5 / R5far / TASK are empty or near-empty for the 8. |
| 8g | gb_summarise | summarise_gain_bio.py (downstream of F) | **done** 11-06 19:04 | slide 102 inputs | YES (downstream) |
| 8g | gb_stats | gain_stats.py (downstream of F) | **failed** 11-06 19:05. `AssertionError: oro - post does not reproduce gain_bio's oro_post`; the fix commit attributes this to the empty R5. | slide 102 stats | YES (downstream) |
| 8g | gb_fig | fig_sessions.py (downstream of F) | **skipped** (dep gb_stats failed) | slide 102 figure | YES (downstream) |

Totals over the 17 rows:
- 14 reach engaged_frame_mask in their own computation or read an affected product. The other 3 are
  g5_chronic_stability (reaches it, output unaffected) and the 2 held goal-summary steps.
- Done: 5 (g5_epoch, d_attempt_maps_rest, d_attempt_maps_plot, gb_sessions, gb_summarise).
- Failed: 1 (gb_stats). Skipped: 1 (gb_fig).
- Not started: 8 (3 restnl_maps, lpfinal_maps, 3 map_drift, chronic_stability).
- Held: 2 (d_fig_goal_summary_f3, d_fig_goal_summary).
- Running now: 0.

Imminence: 7b lane G is live. g5_section_g holds a v7 claim and was waiting for the heavy lock at 18:57. The not-started
7b steps above are later in the same G section and would run on wf_render_daqlk a76352f3, which does not have the fix.
chronic_stability would run in v8 lane GBMD at 5db29b07, also without the fix, but its output is unaffected.

## Caches the fix will NOT invalidate (needed for a correct re-run)

1. **attempt_maps (rest variant).** `main` skips any `maps_<align>_<animal>.npz` that exists ("exists, skipped").
   The rest-variant npz files (and counts/hemisphere CSVs) for all 4 animals x 3 aligns must be moved or removed.
   Without that, d_attempt_maps_rest will not recompute and the 8 sessions stay missing.
2. **gain_bio.** `run()` returns the cached `<gb out_root>/sessions/<label>.npz` when `version == VERSION` (2). The 8
   sessions' npz files must be removed, or VERSION bumped; ba5b63bf does not touch gain_bio.
3. **map_collect (prm per-session maps):** recomputes on its own, because `_raw_maps_parts` includes
   `closure(rest_engagement, "engaged_frame_mask")` and the body changed.
4. **tf-* trial-feature cache with signal_key `svt:rank<K>:restlocal12`** (shared by prm REST and attempt_maps rest)
   is keyed by session signature + kind, not by code. Under the bug the builder returned None for the 8, so no
   restlocal entry should have been written for them in this camera run. UNCERTAIN: an entry written for these 8 under
   the same run-mode tag before the bug existed would be served after the fix. Worth one glob of the session cache for
   `<label>__tf-*restlocal12*` on the 8 labels.
5. **cd_trajectories `cdrest-<ref>-...` session_cache entries** are not code-keyed either. This is moot for now: no
   queued step uses `--reference rest|restw`.

## Checked and NOT affected

- **g2a_restnl_mask, g2b_restnl_mask_lp** (done 11-05): `restnl_mask` without `--measure`. It uses
  rest_by_position / lick-free mask, not engaged_frame_mask.
- **d_attempt_maps_restnl** (done 07:37): the restnl branch uses `restnl_mask.baseline_or_build`, not the time-local
  baseline.
- **g5_cd_traj** (not started) and **g7_cd_geometry** (reads cd results): `--reference` defaults to `contrast`, so
  `rest_baseline` is not called.
- **g5_epoch_1b_v8** (8/8_s2; v8-GBMD lane waiting for the heavy lock): `--only 1b` (behaviour), no maps.
- **g5_epoch families other than 15r** (1b 1c acc 5c 5cr 5r* 10e 10cs 12s 12b 13s 14m mat scal): beta_maps /
  epoch-arm collectors, no rest baseline.
- g5_section_g, g7_section_g_figs, g5_recovery_cue/precue, g6_grant, g5_chronic_drift, g5_rest_frozen (uses
  cue-indexed `engaged_by_cue`), g5_rest_frozen_nl (restnl_mask.lickfree_mask), g5_lpfinal_dec (decoders),
  g7_fig_frozen_refit_pooled, d_fig_goal_summary_f1/f2: no path to engaged_frame_mask.
- scripts/lick_precue_restw.py: uses `rest_frames_by_position` (cue-indexed engagement), so it is in the unaffected
  restw class.
- **Everything else in 7b/8/8b/8c/8d/8e/8e2/8f/8_s2/9k/9k2.** Entry scripts were scanned for references to
  position_reference_maps / epoch_maps / attempt_maps / cd_trajectories / chronic_stability / rest_engagement /
  restnl_mask / lp_final_render / restnl_maps / map_drift_prepre / epoch_grant_figures, and callers were grepped
  repo-wide for every function on the paths above. That includes the frozen/encoder, lick_*, iti_*, pose3d_*,
  orofacial_*, kinematics and lmm steps. No other caller exists. Helper modules imported by scripts (iti_licks,
  restnl, gb_common, figdata) do not call any of these functions.

## Notes

- The rest-referenced REST column is the only map quantity changed. Per the restnl_maps docstring, the imaging page
  uses RESTnl as primary and keeps REST as a supplement. Whether the lick/lick RESTref 15r figure from g5_epoch is
  currently on a page or deck: UNCERTAIN (no queue comment says so).
- The 15r TypeError in the ENL and cue arms of g5_epoch looks independent of this bug: the lick arm rendered with the
  same 8 sessions missing. It will likely recur in restnl_maps, lpfinal_maps and map_drift, which call the same
  renderers. UNCERTAIN.
