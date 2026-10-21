# Handoff — 2026-10-21: attempt-based deficit, target vs execution vs neither, within-spout per spout + contact

**START HERE for the encoding model / target-vs-executed work (the ENCODING window).** Supersedes
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
   Ceiling (target / execution / neither shares) still running.
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
9. **Commanded-axis across-spout slope:** rises with the deficit unadjusted (SSp-m z 9.7) but collapses hits-only;
   contact-adjusted rerun RUNNING.
   [mixed_time_maps_betweencmd.png (contact)](M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/figures/within_spout/within_spout_deviation_lick_contact/mixed_time_maps_betweencmd.png)

Scripts for 7-8 and the four new summary figures: `scripts/contact_dev_check.py`, `scripts/spout_tuning_check.py`,
`scripts/spout_mid_maps.py`, `scripts/summary_figures_1021.py` (argument = the folder holding `spout_tuning_check.csv`).

## 2. RUNNING (updated 10-21 ~11:15)

| what | task / log | then |
|---|---|---|
| contact-adjusted across-spout rerun, cam4 + 3-D contact / contact_hits + mixed models | `S/run_ws_contactadj.sh`, logs `S/ws_adj_*.log`; unadjusted copied to `qc/superseded_within_spout_contact_unadj_20261021/` | read `between_cmd` / `between` with contact control vs hits-only |
| lick-template pass 2 with free-kernel ceiling (since 07:11; ~12:30-14:00) | `S/lick_templates_ceiling.log` | target_like / exec_like / neither shares |
| jaw_onsets watcher (`S/watch_jaw.sh`, 8 h from 10:13) | waits for the pose window's rewrite (direction / cue contrast / treadmill columns) | reruns attempts + LMMs + six within-spout mixed models |

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
2. RUNNING: contact-adjusted across-spout rerun -> read `between_cmd` vs `between` (contact control vs hits-only).
3. Template ceiling shares when the run ends; add a shares panel to `summary_figures_1021.py`.
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
