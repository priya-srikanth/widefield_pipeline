# Handoff — 2026-10-21: attempt-based deficit, target vs execution vs neither, within-spout per spout + contact, map statistics, LP_licks

**START HERE for the encoding model / target-vs-executed work (the ENCODING window).** **Evening state: §1d (results), §2 (running), §2c (LP_licks next steps).** Supersedes
`STATUS_2026-10-20_EVENING_ENCODING_HANDOFF.md` (still the reference for the 3-D contest set-up, §6 commands there).
The pose / 3-D / cam1 / jaw work is the OTHER window: `STATUS_2026-10-21_POSE_CAM1_JAW_HANDOFF.md`.
**Results: `docs/ENCODING_FINDINGS.md`** (status-tagged, updated 10-21). Reasoning + numbers: `DECISIONS.md` 2026-10-21
entries. Scratchpad (launchers, logs, 3-D tables): `C:\Users\SABATI~1\AppData\Local\Temp\claude\C--Users-SabatiniLab\cdd1e016-6bb3-4346-8600-f08116ccf61d\scratchpad\`
(`S`). Output root `M:\MICROSCOPE\Priya\Widefield\LP_analysis\lp_r4_heatmap_20261013__clean_v36\imaging_encoding\` (`E`).

## 0. Rules (Priya; unchanged + new today)

- Curated sessions only (`analysis_kit.curated_labels()`); August-only pre baseline primary. Lead with drop dR2;
  per-animal tables first (rule 8); do not dismiss bilateral changes; canonical spout order (`orofacial_cohort.POS_ORDER`).
- Mixed models primary, epochs descriptive. No circular behavioural index (lateral error descriptive only).
- Brain maps ipsi = image LEFT; cam1 face video ipsi = image RIGHT.
- Priya runs O2 / transfer node / process-stopping commands: give `Get-CimInstance ... | Select ProcessId, CommandLine`
  + `taskkill /PID <pid> /T /F`. **Kill the TOP shell of a queue** (walk ParentProcessId up to the bash under
  claude.exe) -- killing a sub-shell lets the queue start its next step (happened 10-21 08:07).
- Write only under MICROSCOPE/Priya; never delete there; copy superseded outputs aside. Stage files by name; commit and
  push; DECISIONS entries. Results figures in `E/figures/{mixed_models,behaviour,within_spout/<run>}`, tables in `E/qc`.
- Lick-zone analyses use 3-D calibrated zones (cam4 zones failed). 3-D jaw on hold.
- **Toward-midline:** the per-spout `mid:<pos>` test is the headline within-spout test, and the POOLED map is not
  robust to contact (memory note `toward-midline-headline`).

## 1. Done today (details ENCODING_FINDINGS + DECISIONS 2026-10-21)

- **Behavioural index:** `scripts/trial_attempts.py` (hit / incomplete / jaw_only / no_try / unknown; jaw from
  `qc/jaw_onsets.csv` cam4 OR cam1, re-onset rule for jaw deflected at the cue); deficit = **time to first far-contra
  contact** on attempted trials (`mixed_models.DEFICITS["ttc"]`, `--deficit` in encoding_mixed / imaging_lmm /
  within_spout_mixed). Deficit is execution, not initiation (acute attempt .81-.93, completion .01-.11).
- **jaw_onsets rewrite 07:57** (cam1 all PS93, PS94 to 08-17) -> watcher reran attempts + all LMMs 08:00-08:37.
  Unknown <= 6 %; target tracks the deficit more strongly (3-D z -8.3, cam4 -8.2 / firstlick -6.3); strict executed null.
- **Target vs execution vs neither:** 3-D contest (2.4b), lick-template rerun (QUOTE the 20261021 tables), destination
  vs 3-D landing zones (2.4c): post-stroke activity resembles the pre-stroke TARGET; execution-like 4 % acute only.
- **Within-spout per spout + contact (cam4, 107 sessions):** pooled toward-midline chronic reversal NOT robust
  (cluster 28 % base -> nothing with contact / hits-only); per-spout far_ipsi chronic loss and close_contra acute flip
  survive both controls; commanded-axis between = miss / contact confound pending the adjusted fit.
- **Speed:** runs were network-bound (gigabit link to the share saturated by the other window's video jobs; per-read
  latency the rest). `within_spout_deviation.py --variants contact,contact_hits,base` (one load, several fits) and
  `combined_encoding` local input cache `WFIELD_INPUT_CACHE=C:/Users/SabatiniLab/wfield_cache` (Priya OK'd; disposable
  INPUTS only; 5 s vs 80-150 s per session, verified identical; 41 GB at 09:52). **Any edit to a cached loader module
  (`combined_encoding`, `orofacial_cohort`, `orofacial_outputs`, `trial_windows`, `beta_maps`, `daq_trials`,
  `pose_kinematics_demo`) invalidates every entry** -- the next run re-reads the share once. 5 workers beat 2 (latency-bound).
- **Contact-adjusted across-spout fit** (a00045b): under `--contact`, spout means moved to the session's mean contact
  count with the within-spout contact coefficient. NOT yet in any output.

## 1b. KEY RESULTS + WHERE THE FIGURES ARE (10-21 ~11:15)

Root `E` = `M:\MICROSCOPE\Priya\Widefield\LP_analysis\lp_r4_heatmap_20261013__clean_v36\imaging_encoding\`. Numbers use
the CURRENT attempt definition (attempted = hit + incomplete; jaw_only = reaction).

1. **Behavioural deficit = execution in all four + initiation in PS94 / PS95.** Acute far contra: completion .03-.15;
   attempt rate .79 / .80 / .64 / .32 (PS92-PS95). Time to contact pre 344-619 ms -> acute 3114-3419 -> chronic
   239-631; back by subacute in PS94 / PS95, by chronic in PS92 / PS93.
   [attempt_classes_far.png](M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/figures/behaviour/attempt_classes_far.png),
   [lick_deficits.png](M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/figures/behaviour/lick_deficits.png)
2. **Spout coding tracks behavioural recovery and overshoots pre; the rise is TARGET.** Cortex deficit slopes: 3-D unique
   target z -6.8, cam4 -6.7 (cue) / -5.1 (first lick), position -2.7; strict executed null (-1.3).
   [encoding_mixed_behaviour_cortex.png](M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/figures/mixed_models/encoding_mixed_behaviour_cortex.png),
   [encoding_mixed_time_cortex.png](M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/figures/mixed_models/encoding_mixed_time_cortex.png),
   [imaging_lmm_trend_behaviour.png](M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/figures/mixed_models/imaging_lmm_trend_behaviour.png)
3. **3-D contest (PS92 / PS93):** pre-stroke target ~ executed (PS92) or target-led (PS93); post-stroke growth is target;
   executed beyond the spout category present pre and unchanged.
   [spatial_contest_nokin_cortex.png](M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/movement_first_subset-nokin_exec-polar_licks-combined_bins/figures/spatial_contest_nokin_cortex.png)
4. **Lick templates (PS92 / PS93, 3-D):** pre-stroke target templates explain more post-stroke variance than
   landing-zone (execution) templates from subacute on; PS93 chronic in every session; ~nothing beyond spout category.
   CEILING (free per-spout kernels): a NEITHER share appears post-stroke and persists -- chronic 0.38 / 0.30 of
   spout-locked lick structure (12/12, 6/6 sessions > 0; pre ~0); PS93 target-like > execution-like, PS92 overlap.
   [lick_templates.png](M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/figures/target_vs_execution/lick_templates.png)
5. **Destination vs 3-D landing (PS92 / PS93):** post-stroke spout activity matches its own pre-stroke spout (target) in
   most sessions; execution-like only acutely and rarely; PS93 far_mid comes to match pre-stroke far_contra (neither).
   [destination.png](M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/figures/target_vs_execution/destination.png)
6. **Frozen decoders / epoch families:** frozen EV and best-match accuracy stay below pre at full recovery; far-contra
   identity back (10-21 night; refreshed with the new deficit 10:23).
   [imaging_lmm_epoch-11_behaviour.png](M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/figures/mixed_models/imaging_lmm_epoch-11_behaviour.png),
   [imaging_lmm_epoch-10b_behaviour.png](M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/figures/mixed_models/imaging_lmm_epoch-10b_behaviour.png),
   [imaging_lmm_decoder_matched_behaviour_post.png](M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/figures/mixed_models/imaging_lmm_decoder_matched_behaviour_post.png)
7. **Within spout: no robust execution-like signal.** Pooled toward-midline chronic reversal fragile (cluster 28 % ->
   0 with contact control; spouts disagree in sign) but NOT a contact artifact (deviation-contact coupling same pre and
   chronic); per spout far_ipsi chronic 2/4, close_contra acute flip 2-3/4; 3-D agrees with cam4; angle terms do not
   predict held-out activity.
   [mixed_time_maps_mid.png (base)](M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/figures/within_spout/within_spout_deviation_lick/mixed_time_maps_mid.png),
   [within_spout_maps_cluster_mid.png (base)](M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/figures/within_spout/within_spout_deviation_lick/within_spout_maps_cluster_mid.png),
   [within_spout_maps_cluster_mid.png (contact)](M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/figures/within_spout/within_spout_deviation_lick_contact/within_spout_maps_cluster_mid.png),
   [spout_mid_maps_per_animal.png](M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/figures/within_spout/within_spout_deviation_lick_contact/spout_mid_maps_per_animal.png),
   [mixed_time_maps_mid.png (3-D, contact)](M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/figures/within_spout/within_spout_deviation_polar_lick_contact/mixed_time_maps_mid.png)
8. **Miss-then-correct trials are common** (far contra pre 25-56 %); PS94 chronic far contra 80 % (first-lick contact
   15 % vs 71 % pre) -- corrective recovery.
   [miss_then_correct.png](M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/figures/behaviour/miss_then_correct.png)
9. **Commanded-axis across-spout slope survives contact adjustment** (SSp-m bilateral deficit z 8.5 cam4 / 8.5 3-D;
   3-D hits-only 4.8); the cam4 hits-only collapse is a lost spout (far contra absent from acute hits-only sessions).
   Bilateral SSp-m responds less to contralesional commanded spouts while the deficit is large; executed axis null.
   [mixed_time_maps_betweencmd.png (contact)](M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/figures/within_spout/within_spout_deviation_lick_contact/mixed_time_maps_betweencmd.png)

Scripts for 7-8 and the four new summary figures: `scripts/contact_dev_check.py`, `scripts/spout_tuning_check.py`,
`scripts/spout_mid_maps.py`, `scripts/summary_figures_1021.py` (argument = the folder holding `spout_tuning_check.csv`).

## 1c. AFTERNOON (10-21 13:00-15:30) -- READ BEFORE CONTINUING (DECISIONS 10-21 afternoon entries)

10. **Template ceiling:** a NEITHER share (lick-locked structure matching no pre-stroke template) appears post-stroke
    and persists: chronic 0.38 (PS92) / 0.30 (PS93), > 0 in 12/12 and 6/6 chronic sessions, pre ~0. PS93 target-like >
    execution-like; PS92 templates overlap. [lick_templates.png](M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/figures/target_vs_execution/lick_templates.png) (third panel)
11. **Within-spout z-scored** (each animal in its pre-stroke SD units; families `*_z`) and REPORTED AS PER-POSITION
    DIRECTION (Priya: per-position, not toward-midline; + = licks toward the mouse's LEFT). Pre: far_ipsi negative 4/4,
    close_contra negative 3/4 (PS94 ~0) = more activity for licks deviating toward the contralesional side; close_contra
    flips acutely 3/4; far_ipsi chronic PS94 reversed, PS92 lost. Reliability-corrected (`scripts/spout_reliability.py`):
    PS95's chronic weakening was attenuation. [spout_mid_maps_per_animal_z.png](M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/figures/within_spout/within_spout_deviation_lick_contact/spout_mid_maps_per_animal_z.png)
    (NB the figure still labels values "toward midline": flip the sign at ipsi spouts to read direction.)
12. **Map statistics review** (Priya asked about the 15r POSITION ACTIVITY maps, ENL / cue / lick): pooled pictures are
    pre-normalised per animal since 09-12 but the significance contours are computed on RAW maps; 14z picture raw;
    within-spout vs-zero "cluster" call silently fails. Calibration on figure-14 beta maps: pure nulls flag NOTHING (both
    tests) -> no small-n correction; June vs August (drift) flags up to 14 % (nested) / 53 % (cluster, normalised) at
    contra positions; normalising removes acute far_center (23 % -> 0.1 %). 15r position-map calibration RUNNING.

## 1d. EVENING (10-21 15:30-18:00) -- normalised map tests MERGED, LP_licks family BUILT (branch), ENL census

13. **Map-test calibration, all three 15r windows** (precue/ENL, cue, lick x references raw / rest / restw / precue /
    mean; CSVs `labcams/grant_figures/epoch/results/map_test_calibration_position_<ref>_<align>_<variant>.csv`):
    - NESTED (primary) flags NOTHING on any pure null (August split, all-pre split), any window / reference, raw or
      normalised -> no small-sample correction.
    - CLUSTER (second line) occasionally fires on a null at ONE position and then covers 14-58 % of the brain (rest /
      restw close_center precue, far_R cue / lick). 24 random within-animal splits (rest close_center): fired on 2
      (8 %, consistent with 5 %), nested 0. **Read cluster contours as yes / no, never by area.**
    - June vs August drift up to ~21 % at a few positions -> supports the August-only baseline.
    - Acute far contra (cue): 58-93 % under every reference, both tests, raw and normalised. Lick window cannot test
      far contra (0 acute sessions reach 20 contact-lick trials) -> LP_licks below.
    - Normalising removes precue-referenced near-spout changes (18-28 % -> 0-9 %) and far_center (5-9 % -> 0-6 %):
      carried by the brighter animals.
14. **Normalised map tests MERGED to main** (`dc10705`, `babe04f`; DECISIONS 10-21 late afternoon). `beta_maps.pre_scale`
    = THE per-animal x position constant (shared with the pooled picture); nested / Musall on pre-scaled maps; cluster
    recomputes the constant inside every relabelling draw; `vs_zero_contour(scale_by=)`; labels "[pre-normalised]". 14z
    picture normalised; within-spout vs-zero uses nested (the "cluster" call used to raise); z pixel maps
    (`mixed_time_maps_*_z.png`); `spout_dir_maps_per_animal*.png` = per-position DIRECTION (+ = mouse's left), old
    toward-midline PNGs kept. `scripts/compare_map_tests.py` = raw vs normalised sig-bin table per figure cell.
    [spout_dir_maps_per_animal_z.png](M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/figures/within_spout/within_spout_deviation_lick_contact/spout_dir_maps_per_animal_z.png)
15. **LP_licks family (Priya: "a separate family - LP_licks"; lick-aligned AND lick-gated analyses) -- BUILT on branch
    `lick/lp-source` (worktree `S/wf_lp`, commits `09f163f`, `a6ac0fe`), NOT merged, full suite not yet run.**
    - Switch `WIDEFIELD_LICK_SOURCE=lp` (default daq = unchanged). `wfield_local/lp_licks.py`: LP lick = cam4 `licks_all`
      + cam1-only events (cam4 fallback where cam1 missing: PS95 pre); gate = attempt class hit / incomplete with the
      first LP lick inside max_rt; lick-aligned = LP ONSET (tongue out; Priya agreed -- peak is 28-56 ms later and
      kinematics-dependent; a -0.15 s version only as an optional sensitivity); trials matched to DAQ cues by TIME (all
      within 20 ms, median ~1 ms); `unknown` class = no-lick (1176/1181 are close_center with the jaw masked, no
      tongue lick); only trials with no attempts row dropped.
    - Wired at the shared DAQ lick loader (`plot_lick_aligned_averages._load_daq_events`, so the ~25 direct callers
      get LP too), `_trial_features`, `nolick_decoder.categorize`. `session_cache` tags every kind under LP;
      local sidecar of the LP tables (`.widefield_session_cache/lp_licks/`).
    - Frozen decoders: DAQ models REUSED (Priya: no new frozen set); an LP miss REFUSES to fit; PS93 far_ipsi /
      far_mid sensitivity via `WIDEFIELD_LP_FROZEN=lp` (own spec). Pre-stroke LP vs DAQ: LP attempt without contact
      <= 3 % everywhere except PS93 far_ipsi 27 % (onset -211 ms vs contact) / far_mid 14 %.
    - Outputs `labcams/grant_figures/LP_licks/...` + corner stamp; `writeguard` refuses LP-run share writes outside an
      `LP_licks` directory.
    - Check PS93_0819 (acute): far contra engaged 12 (DAQ) -> 59 (LP); others ~unchanged.
16. **ENL lick-free census** (Priya: redefine "lick-free" ENL with LP licks; maybe shorter window):
    `scripts/enl_lp_census.py` (branch) applies `precue_window_start` to DAQ vs LP trains at 2.0 / 1.5 / 1.0 / 0.5 s.
    PS93_0809 (pre, worst session): kept DAQ 91 % / LP 67 % at 2 s; LP 90 % at 1.5 s, 99 % at 1 s. Cohort run RUNNING.

## 2. RUNNING (updated 10-21 ~18:05)

| what | task / log | then |
|---|---|---|
| Rerender after the normalised tests (main): epoch 15r / 15e / 14m, then within-spout figures (8 folders), mixed models (z maps), `compare_map_tests.py` | `S/run_rerender_norm.sh`; logs `S/rerender_epoch.log`, `S/rerender_ws.log`, `S/rerender_mixed_<folder>.log`, `S/compare_map_tests.log`; started 16:32, first arm still running at 18:05 (CPU shared) | read `labcams/grant_figures/epoch/results/map_tests_raw_vs_normalised.csv`; links to new figures; ENCODING_FINDINGS |
| ENL LP census | DONE: `E/qc/enl_lp_census{,_summary}.csv` | window chosen: 1.0 s (DECISIONS late evening) |
| jaw_onsets watcher (`S/watch_jaw.sh`) | EXPIRED 18:13, no rewrite seen | restart only if the pose window still plans the rewrite |

## 2c. NEXT for LP_licks, in order

1. DONE (`4fadded`): ENL census -> Priya chose **1.0 s for LP, plus a matched DAQ 1.0 s arm** (LP 2 s keeps 48 % of
   PS93 pre far_L / far_center, 66 % of acute far_R; 1.0 s >= 95 % everywhere). `WIDEFIELD_ENL_WINDOW_S`; outputs
   `LP_licks/` and `DAQ_ENL1s/`; frozen precue specs carry the window -> run the DAQ_ENL1s arm FIRST (it freezes the
   1.0 s precue set; LP reuses it).
2. Full test suite in `S/wf_lp`; merge `lick/lp-source` to main (it touches `_trial_features`'s module, the shared lick
   loader and `session_cache`: wait until the normalised rerender is done -- spawned render workers import from main).
3. Per animal x position x epoch table: LP vs DAQ engaged / no-lick trial counts (rule 8) before any figure.
4. Null calibration of the LP lick maps (`map_test_calibration.py --family position` under the LP env).
5. First LP render: `WIDEFIELD_LICK_SOURCE=lp python -m wfield_local.epoch_grant_figures --only 14m 15r acc 5c` (lick,
   cuelick, ENLlick arms) -> `grant_figures/LP_licks/epoch/`; then grant_figures, CD trajectories (route their output
   with `lp_licks.output_dir` first -- writeguard will refuse otherwise).
6. PS93 far_ipsi / far_mid frozen sensitivity: `WIDEFIELD_LP_FROZEN=lp` on the lick arm; compare with the DAQ-frozen
   readout.
7. Optional `LP_licks_pre150` (window from 0.15 s before LP onset).
8. **REVISIT WHEN CAM1 IS FINISHED** (Priya 10-21: "not all cam1 analysis is done so we'll have to revisit"). cam1
   lick events exist for 74 / 108 curated sessions (August pre + post): PS92 all; PS93 all but pre 7 / 8 (0813 has no
   video); PS94 all but chronic 2 / 8; **PS95 none**. Those sessions' LP gate is cam4-only (misses the 3-7 % small
   licks only cam1 sees; incomplete licks rise post-stroke), so cross-animal LP comparisons carry that until cam1
   lands. Also cam4-only until then: `trial_attempts` `combined` latencies (PS95), the ENL census pre-cue train (cam1
   events are post-cue only anyway). The LP cache tag includes the cam1 folder (`306ba22`), so new cam1 files force a
   full LP recompute; rerun `scripts/trial_attempts.py` first (the gate reads it), then the LP renders.

## 2b. Normalised map tests (Priya 10-21 ~15:30: "sounds good proceed") -- DONE 1-4 and 7 (merged), 5 RUNNING, 6 open

1. `wfield_local/beta_maps.py`: add `NORMALISE_TESTS_BY_PRE = True` with the justification (DECISIONS 10-21 afternoon,
   map statistics entry); `pre_scale(pre_by_animal, mask)` (THE per-animal constant, brain_mask RMS of the mean pre
   map, 1.0 if degenerate) and use it inside `within_animal_pooled`; `scale_maps`; `significance_contour(normalise=None)`
   -> nested / musall on maps divided by `pre_scale(pre)`, CLUSTER via a new `cluster_permutation(normalise=True)` that
   recomputes each animal's constant inside `_stat(assign)` from that draw's pseudo-pre sessions; labels prefixed
   "[pre-normalised]"; `vs_zero_contour(scale_by=pre_by)`. A draft (fixed-constant cluster, to be revised) is in
   `S/patch_normtest.py`. Tests for `pre_scale` / scale invariance. Editing beta_maps invalidates the input cache.
2. `wfield_local/epoch_maps.py` `_fig_14z_beta_vs_zero`: normalise the PICTURE (each animal / its pre constant) and pass
   `scale_by` (pre maps of the same position).
3. `scripts/within_spout_deviation.py` ~line 642: vs-zero with method "cluster" -> use "nested" (label it) + `scale_by`.
4. `scripts/within_spout_mixed.py` `pixel_maps`: z versions (each animal's maps x its pre lat_sd / resp_sd, cortex,
   spouts "all"), `mixed_time_maps_<kind>_z.png`.
5. Rerender: `python -m wfield_local.epoch_grant_figures --only 15r 15e 14m` (~1-2 h; old versions archived in
   `labcams/grant_figures/epoch/archive_rawdF_tests_20261021/`); within-spout `--figures-only` for each folder +
   `within_spout_mixed.py --folder ...` (old test tables `map_significance_*_rawdF_20261021.csv`). Compare new vs archived
   contours per family / position; expect acute far_center to shrink. ENCODING_FINDINGS + DECISIONS + this handoff.
6. Optional: `scripts/rest_migration/rest_carries_position.py` (raw picture + raw vs-zero test).
7. Fix the "toward midline" wording in `spout_mid_maps.py` titles -> per-position direction.

## 2a. Superseded at ~09:55 (kept for the record)

| what | task / log | then |
|---|---|---|
| 3-D within-spout variants, PS92 / PS93 (47), contact / contact_hits / base + mixed models | `S/run_ws_polar_variants.sh`, logs `S/ws_polar_variants.log`, `S/ws_mixed_polar_*.log`; old pooled folder copied to `qc/superseded_within_spout_polar_pooled_20261021/` | report per-spout `mid:<pos>` (3-D), compare with cam4 |
| lick-template pass 2 with the free-kernel ceiling | `S/lick_templates_ceiling.log`, started 07:11 | report target_like / exec_like / neither_share (cortex, animal x epoch) |
| direct contact check (does contact count rise with toward-midline deviation, per spout, pre vs chronic?) | `S/contact_dev_check.py` -> `S/contact_dev_check.{csv,log}` | confirms or rules out the contact confound in the pooled reversal |
| jaw_onsets watcher | task b2d4hbh9m (4 h from ~06:40) | reruns attempts + LMMs on any further rewrite |

## 3. To-do (priority)

1. DONE 10:00-11:15: 3-D per spout, direct contact check, eccentricity / first-lick checks, per-animal maps, jaw_only
   reclassification + reruns (DECISIONS 10-21 late morning entries).
2. DONE 12:39: contact-adjusted across-spout rerun (DECISIONS 10-21 midday).
3. DONE 14:03: template ceiling (NEITHER share persists chronically; DECISIONS 10-21 afternoon).
4. Rerun after the pose window's jaw_onsets rewrite (watcher does it); then consider the new columns (jaw direction
   open vs up, pre-cue contrast, treadmill speed) for the reaction vs attempt split and the walking confound.
5. Behaviour follow-up: miss-then-correct (corrective recovery, PS94 chronic) as its own readout / class.
6. Shorten the LMM x-axis label (overlaps panels).
7. Destination test for PS94 / PS95 when their 3-D exists.
8. Whole-session-lick refits need an O2 re-bundle (Priya submits).
9. Old copies Priya may delete: stray empty `qc/figures/imaging_lmm_epoch`, old figure copies under `qc/figures/` and
   `qc/within_spout_deviation*/figures/`.
10. Carried over (10-20 evening §4): first-lick latency cam4 vs cam4 + cam1 vs DAQ; LocaNMF cross-check of within-spout
   slopes; pre-cue -> lick-window carry-over decode; 3-D jaw when revived.

## 4. Pitfalls (new today)

- `jaw_pass_qc` in v36 = "jaw moved", not a tracking-QC flag. cam1 jaw needs `jaw_onsets.csv` (old cam1 windows were on
  the frame-index clock). True/False strings vs 1/0 read as 0 % agreement.
- Colons in figure filenames create NTFS alternate data streams (`imaging_lmm.py` now sanitises).
- `combined_encoding_figures.py` on `drop_win` folders exits cleanly (no `alone:` models).
- The pooled within-spout slope weights spouts by their lateral spread (PS94_0806: pooled +0.00027 vs equal-weight
  -0.00024) -- read `equal` / per-spout rows.
- The cluster test needs EVERY animal to agree and is all-or-nothing at |t| 3.18 (df 3): "nothing survives" after a
  control can mean a modestly weaker map; read the mixed-model z alongside.
- Two pre-push test runs at once (both windows) take ~1 h on a busy network; a rejected push ("cannot lock ref") can
  still land if the other window's push carried the commits -- check `git branch -r --contains <sha>`.
- (evening) A script run from outside the repo (e.g. `/tmp/x.py`) imports the INSTALLED package = the MAIN checkout,
  not a worktree: set `PYTHONPATH=<worktree>`. The first LP check silently ran DAQ code this way.
- (evening) Bash heredocs in this Git Bash collapse `\\n` too, not only `\n`: patch files containing backslashes must
  go through the Edit tool (two more patch scripts stopped at their assert, nothing written).
- (evening) trial_attempts `unknown` is NOT "unclassifiable": it is close_center trials with no tongue lick and the jaw
  masked. Treat as no-lick for lick gates.
- (evening) Do not edit code in the main checkout while a render runs from it: process-pool workers spawn and import
  the CURRENT files. Code work goes in a worktree (`S/wf_normtest` merged; `S/wf_lp` open).
- (evening) Cluster-permutation false positives on rest / restw maps are large in AREA (session-wide offset): a cluster
  contour's extent is not evidence of a broad change.

## 5. Commands

```bash
cd /c/Users/SabatiniLab/Github/widefield_pipeline
export PYTHONPATH=. WFIELD_INPUT_CACHE=C:/Users/SabatiniLab/wfield_cache
# cam4 within-spout, all variants in one pass, then mixed models
python scripts/within_spout_deviation.py --align lick --jobs 5 --redo --variants contact,contact_hits,base
for d in within_spout_deviation_lick_contact within_spout_deviation_lick_contact_hits within_spout_deviation_lick; do
  python scripts/within_spout_mixed.py --folder $d; done
# 3-D: same with the env of S/run_ws_polar_variants.sh (20261021 tables) and $(cat S/labels_3d.txt)
# behaviour index / LMMs
python scripts/trial_attempts.py && python scripts/encoding_mixed.py && python scripts/imaging_lmm.py
WIDEFIELD_PRE_FROM=0801 python scripts/best_match_destination.py
python -m pytest -q tests/test_within_spout_deviation.py tests/test_input_cache.py tests/test_trial_attempts.py
```
- Editing ANY `_CACHE_CODE` module (even a comment) invalidates the whole local input cache (41 GB); keep doc-only
  changes to those files for a quiet moment.
- jaw_only trials are reactions (Priya's blind review); `trial_attempts.ATTEMPT_CLASSES` is the one definition.
- Python patch files written through a bash heredoc turn `\n` inside strings into real newlines (broke three edits
  today) -- use the Edit tool for any string containing `\n`.
- Report within-spout per-spout results as per-position DIRECTION, not toward-midline (Priya 10-21 afternoon).
