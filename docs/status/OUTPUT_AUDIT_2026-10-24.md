# Output staleness audit -- widefield VLS-stroke (PS92-PS95, lesions LEFT) -- 2026-10-24

Written for: Priya and the next encoding session. EXTENDS `OUTPUT_INVENTORY_2026-10-24.md`, which covered only
`imaging_encoding/`; this sweep covers every output root and corrects that inventory where marked. Read-only sweep (nothing
changed); helper scripts / CSVs were in the session scratchpad (`audit_20261024/`), the family table is reproduced below.
**Update 10-24 (later):** UNSURE item 1 resolved -- the 1.0 s DAQ frozen pre-cue decoders were not stored for the joint basis;
`attempt_coding --freeze` stores them (DAQ path, `WIDEFIELD_ENL_WINDOW_S=1.0`) before any LP pre-cue run.
Roots swept: labcams (grant_figures incl. epoch / LP_licks / DAQ_ENL1s / epoch_latepre / histology, analysis_figures,
xday, chronic_stability, null_potent, frozen_decoder_latepre, frozen_models, joint_bases, decks), LP_analysis (v35, v36
cohort / sessions / imaging_encoding incl. qc / figures / run folders), session_poses derived tables (pose3d, cam1_licks,
attempt_review, whisker_me, legacy root files), behavior_summary, cue_lick, wfield_cache, local frozen store
(`C:/Users/SabatiniLab/frozen_models`), FaceRhythm factor sets, O2 encoding bundles.
No `attempt_coding` output folder exists yet (module only being smoke-tested: `ac_*` logs in the scratchpad).

## 1. Input-readiness timeline (verified against mtimes / DECISIONS / handoffs)

| date / time | change | verified / correction |
|---|---|---|
| 10-18 | LP run v36 (June px rescale) created; v35 SUPERSEDED (marked) | OK. |
| 10-18 late | AUGUST-ONLY pre baseline primary for orofacial + encoding; `*_allpre` = superseded sensitivity | OK. **Addition:** imaging epoch families were explicitly NOT switched ("imaging doesn't need to switch"); `epoch_latepre/` and `frozen_decoder_latepre/` are the August-only confirming sensitivity; the imaging default stays all curated pre. |
| 10-20 10:28-11:19 | v36 re-run of **ALL 133 sessions** on template timing (the 9 timing sessions incl. PS92_0908 first, in the morning); v36 cohort products rebuilt 11:33-12:02 | **Correction:** not only PS92_0908 -- every v36 session folder is dated 10-20 10:xx-11:xx. 3-D frames (PS92/PS93) re-cut 10-20 11:22-19:53, cam1 `events_all` 10-20 21:44 on. |
| 10-21 ~07:57 / 10:54 | jaw_onsets rewritten (template-timed cam1 + cam4); 10:54 nine columns appended (old columns identical) | OK. |
| 10-21 late morning | jaw_only = reaction, not attempt; attempts + deficit models re-run; table final 14:33 | OK (`superseded_attempts_cam1incomplete_20261021/` = the 14:33 files). |
| 10-21 afternoon | normalised map tests implemented (~16:30); DAQ 15r / 15e / 14m re-rendered 16:55-20:12; raw-test renders archived | Added change point: any map figure with significance contours made before 16:55 10-21 uses the old raw tests (affects `epoch_latepre`, 10-18 decks, cam4 within-spout maps before 20:18). |
| 10-21 evening | LP_licks lick source (cam4 + cam1 combined; gate = attempt class hit / incomplete); ENL window 1.0 s under LP + matched DAQ 1.0 s arm; render cache + race fix at night | OK. |
| cam1 events | at trial_attempts 10-21 14:33: PS92 28/28, PS93 26/26, **PS94 11/27, PS95 0/27** (August+); at the LP_licks render 10-21 22:50: PS94 26/27, **PS95 1/27**; COMPLETE 10-23 17:45 (125 event files) | **Correction:** the "PS94 partial (chronic 2/8)" note was already outdated at the LP render (PS94 missing only 1005); PS95 essentially cam4-only. `cam1_coverage_at.csv`. |
| 3-D | 10-21 snapshot PS92 + PS93; PS94 frames by 10-21 20:43 (1005 on 10-22), PS95 10-22 21:01 .. 10-23 17:27; kinematics / kinematics_all 17:30 / 17:41; derived tables to 18:23; chain end 18:28; encoding re-cut `*_20261023` 18:29-18:33 | OK ("complete 18:28" = chain end; the 3-D frames themselves were done 17:27). |
| 10-23 18:27 / 19:54 | jaw_onsets rebuilt on complete cam1; 3-D columns appended 19:54 | Added. |
| 10-23 evening | frozen STATS_VERSION 7 (forward-only running 1.5 mm/s, no push; 73d2991); grouping v2 (nose -> other) late; grouping v3 (contact own category) night | OK. |
| 10-24 06:29 | trial_attempts REBUILT on complete cam1 | OK. **Note DECISIONS 10-24:** flips 0 / 0 / 93 / 62 trials (all toward attempted); "gated runs done on the old table are not re-run for the gate alone; the final-frame runs use the new table". So old-gate outputs are STALE but replaced only through the final-frame / O2 re-runs. |
| 10-23 / 10-24 | FaceRhythm fr-timing-v1 (10-23); master_pre_v1 rank 12 decided 10-24: master.npz 05:49, proj 06:12, labels.json 06:30 (rank 10 kept as `*_r10`, 10-23 20:25); pilot_pre_PS92_PS93 set 10-22 (proj rewritten 10-23 07:21) | Added dates. |
| 10-24 | 3-D tongue option (9f45819); final frame tag `gate-attempted_v7g3_t3d` (first launch 05:54 exit 1, relaunched 06:36, writing since 07:13); polar primary -> **BINNED contest primary** (same day) | OK. |
| 10-24 06:36+ | O2 bundle `enc_v36_full_20261024`: 107 sessions, rebuilt attempts, combined licks, 20261023 3-D, 3-D tongue, RAW FR rank 12; inputs still being written (49/107 at audit time) | OK; `submit_all.sh` submits bins_w1 / w05 / w2, main, droppos, shufpos, nodir, drop, trmodels, musall_shuf, fr_A / B / S / D. |

## 2. Output families and verdicts

Ordered for the goal question ("what does cortex look like post-stroke when the animal attempts a movement it cannot
execute or executes differently"). P1 = attempt-gated / video-lick imaging readouts and their statistics; P2 = target vs
executed / direction; P3 = behavioural descriptors and side tables. `E` = `LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding`,
`G` = `labcams/grant_figures`.

### 2a. STALE-REDO

| pri | family | location | made | depends on | verdict | reason | re-run / owner step | status 10-25 (d4, checked against share mtimes) |
|---|---|---|---|---|---|---|---|---|
| P1 | **LP_licks maps** 15r / 15rpa / 14 / 14pa / 14z (cue_working, lick_lick, precue_working) | `G/LP_licks/epoch/` (+data, svg) | 10-21 22:50 - 10-22 01:10 | LP licks (cam4 + cam1) + LP gate = attempt class from the 10-21 14:33 table | STALE-REDO | cam1 PS95 1/27, PS94 26/27 at render; attempt classes from a table with cam1 PS94 11/27 / PS95 0/27 | planned LP re-render: `WIDEFIELD_LICK_SOURCE=lp python -m wfield_local.epoch_grant_figures --only 14m 15r acc 5c --parallel-arms 3` (LP cache key = cam1 folder digest + attempts mtime, so it recomputes) | DONE: re-rendered 10-24 10:24-10:25 (after the 06:29 attempts rebuild) |
| P1 | **LP_licks decoders** 5c / 5cr / acc / accdelta (cue_lick, cue_working, lick_lick, precue_lick, precue_working) | `G/LP_licks/epoch/` | same | DAQ-trained frozen decoders (unchanged) scored on LP-gated trials | STALE-REDO | trial selection = stale LP gate | same command (planned) | DONE: 10-24 09:48 (same render) |
| P1 | LP null calibration (`map_test_calibration_position_*`, `calibration_LP_vs_DAQ_summary.csv`) | `G/LP_licks/epoch/results/` | 10-21 23:00-23:23 | LP-gated maps | STALE-REDO | as above | `WIDEFIELD_LICK_SOURCE=lp python scripts/map_test_calibration.py --family position` per arm / ref (planned "calibration") | DONE: results/ 10-24 11:05-11:16 |
| P1 | LP vs DAQ trial counts (rule-8 table) | `E/qc/lp_trial_counts{,_summary}.csv` | 10-21 21:35 | LP gate | STALE-REDO | as above | **GAP**: `python scripts/lp_trial_counts.py --jobs 6` -- must precede the LP figures (handoff 10-21 step 3) | DONE: 10-24 08:48 |
| P1 | LP-gated encoder decompositions (acute far-contra drop carried by lick) | `E/qc/lp_encoder_decompose*.csv` | 10-22 07:21 - 10-23 22:10 | cam4-era / stale-gate encoder runs | STALE-REDO | source runs stale | **GAP**: `python scripts/lp_encoder_decompose.py --folder frozen_encoder_gate-attempted_v7g3_t3d/<variant> --align cue|lick` (+ O2 main after collect); not in `run_v7g3_t3d.sh` | DONE: final frame v7g3_t3d joint cue + lick, 10-25 01:47 |
| P1 | Frozen Layer 1 v7g3 / v7pushfree / v7fwd15 / v7basis | `E/frozen_encoder_gate-attempted_*` | 10-23 15:47 - 10-24 00:07 | 10-21 attempts gate, cam4 tongue, grouping v2/v3; v7fwd15 / v7basis also PS95_0817 / 0831 pre-cam1 | STALE-REDO | superseded frame + stale gate | RUNNING `_v7g3_t3d`; basis maps in final frame (planned) | DONE: final frame `_v7g3_t3d` (10-24 10:25); basis maps `_v7g3_t3d_basis` 10-24 14:11 |
| P1 | Relabel test | `E/frozen_relabel_v7` | 10-23 20:17 - 10-24 01:01 | v7fwd15 templates, stale gate | STALE-REDO | | planned (final frame) | DONE: `frozen_relabel_v7g3_t3d` 10-24 23:33, trial level `_trial` 10-25 01:39; no-attempt `_trial_refgate` RUNNING |
| P1 | Layer 2 all-session | `E/allsession_encoder_v7push` | 10-23 22:53-23:03 | v7pushfree frame, frozen push, stale gate | STALE-REDO | | planned (final frame) | DONE: `allsession_encoder_v7g3_t3d` 10-24 13:10 (+ `_ref` 20:54) |
| P1 | Session LMMs: imaging_lmm, encoding_mixed (ttc + `_contact`), epoch-family LMMs, push LMM | `E/qc/imaging_lmm*.csv`, `encoding_mixed*.csv` | 10-21 02:13 - 10-23 18:39 | deficit index from trial_attempts + cam4-era encoding measures | STALE-REDO | stale attempts (`_contact` 02:13 even predates the jaw_only reclass) | planned "LMMs after": `PYTHONPATH=. python scripts/imaging_lmm.py [--sources trend,decoder,behaviour,epoch | push]`, `scripts/encoding_mixed.py`, default + `--deficit contact` | PARTLY: imaging_lmm default + contact 10-25 01:40 (e3); encoding_mixed (10-21) and push LMM (10-23) still STALE (encoding_mixed = d4 to-do) |
| P1 | LMM figures | `E/figures/mixed_models/*.png` | 10-21 02:13 - 10-23 18:40 | LMMs above | STALE-REDO | **inventory had KEEP** | re-rendered by the LMM step | PARTLY: imaging_lmm figures 10-25 01:41; encoding_mixed / push figures stale |
| P1 | O2-era combined_encoding runs | `E/movement_first*` (main, droppos, shufpos, nodir, nokin, nokin_matched, nokin_tr, nokin_win, drop_win, trmodels, musall shuf; 10-21/22 combined + gate-attempted drop / full) | 10-19 - 10-22 | cam4 licks, reference gate, cam4 tongue (10-21/22: 27 PS95 sessions pre-cam1, stale gate) | STALE-REDO | | O2 bundle covers main / droppos / shufpos / nodir / drop / trmodels / musall_shuf / binned nokin:. **GAP**: nokin_matched (needs a new match.json), nokin_tr, nokin_win, drop_win are not in `submit_all.sh` -- submit or retire explicitly | DONE for the bundle runs (collected 10-24); nokin_matched / nokin_tr / nokin_win / drop_win: recommend RETIRE (d4 -> 5b, 10-25), not yet in DECISIONS |
| P2 | Destination vs landing (2.4c) + `destination.png` | `E/qc/best_match_destination*.csv`, `E/figures/target_vs_execution/destination.png` | 10-21 07:22 - 10-23 06:21 | landing zones from the 2-animal polar contest (10-21 snapshot) | STALE-REDO | 2 animals; **its source folder `..._polar_licks-combined_bins` was deleted 10-24** | **GAP**: edit `CONTEST_3D` in `scripts/best_match_destination.py` to `movement_first_subset-nokin_exec-polar_licks-combined_bins_3d4` (or the O2 binned run), then `WIDEFIELD_PRE_FROM=0801 PYTHONPATH=. python scripts/best_match_destination.py [--arm cue/working]` | STALE (not re-run) |
| P2 | Lick-template test (target vs landing vs "neither") + `lick_templates.png` | `E/qc/lick_templates_polar_20261021/`, `E/figures/target_vs_execution/lick_templates.png` | 10-21 02:30 - 14:05 | 10-21 snapshot (2 animals), combined licks then | STALE-REDO | 2 animals | **GAP**: polar 3-D env with the `*_20261023` tables, `PYTHONPATH=. python scripts/lick_template_test.py --jobs 1 --labels-file .../labels_3d_20261023.txt` | STALE (not re-run) |
| P2 | Within-spout cam4 **cue-aligned base** and **lick_enl** variants | `E/qc/within_spout_deviation/`, `E/qc/within_spout_deviation_lick_enl/` (+ figures/within_spout) | 10-20 14:19 - 10-21 20:56 | cam4 lateral angle | STALE-REDO | 3-D lateral angle is the definition (10-24); running ws3d4 is `--align lick --variants base,contact,contact_hits` only | **GAP**: re-run with the 3-D env for the cue-aligned and ENL variants, or retire them in DECISIONS | STALE (3-D polar ran lick-aligned only, 10-24 07:09-07:22) |
| P2 | Within-spout mixed models + per-position direction maps + reliability | `E/qc/within_spout_deviation*/mixed_model*.csv`, `spout_dir_maps_per_animal*.png`, `spout_reliability` | 10-20 23:17 - 10-21 20:56 | cam4 within-spout + deficit index | STALE-REDO | not chained after ws3d4 | **GAP**: `within_spout_mixed`, `spout_mid_maps.py`, `spout_reliability.py` on the ws3d4 outputs (after attempts-based deficit) | STALE (not re-run on the polar outputs) |
| P2 | `landing_zones.png` | `E/figures/behaviour/` | 10-21 02:03 | 2-animal contest soft weights | STALE-REDO | | **GAP**: `scripts/encoding_key_figures.py` pointed at `_3d4` / O2 bins | STALE |
| P3 | `attempt_classes_far.png`, `miss_then_correct.png` | `E/figures/behaviour/` | 10-21 14:05 | trial_attempts **before** the 14:33 write + scratch `spout_tuning_check.csv` | STALE-REDO | **inventory had KEEP** | **GAP**: `python scripts/summary_figures_1021.py <dir with spout_tuning_check.csv>` (input lives in the scratchpad) | STALE |
| P3 | `lick_deficits.png` + `qc/first_lick_latency_late_licked_trials.csv` | `E/figures/behaviour/`, `E/qc/` | 10-20 19:21 / 10-21 02:03 | first-lick latency at 10-20 (cam1 PS92/PS93 only) | STALE-REDO | **inventory had KEEP**; late-licked table has no producer in the repo | **GAP**: regenerate from current `qc/first_lick_latency.csv` (10-23 18:15), then `encoding_key_figures.py` | STALE |
| P3 | `precue_licking.png` | `E/figures/behaviour/` | 10-21 02:03 | DAQ pre-cue contacts + ENL dR2 of the cam4 O2 main run | STALE-REDO (low) | dR2 half superseded | `encoding_key_figures.py` after the O2 main collect | STALE (O2 main now collected: can re-run) |
| P3 | Lick miss rate by epoch (AR-HMM comparison) | `E/qc/lick_contact_by_epoch.csv` | 10-23 08:32 | `trial_attempts_sessions.csv` (10-21) + v36 `licks_all` (cam4 only) | STALE-REDO | **inventory had KEEP** | **GAP**: `python scripts/lick_contact_by_epoch.py` (seconds); consider adding cam1-only no-contact licks | STALE |
| P3 | Position-coding recovery trend | `E/qc/position_coding_trend_recovery*.csv` | 10-20 16:20 | cam4 O2 nokin + main runs, cam4 contact fraction | STALE-REDO | | **GAP**: `python scripts/position_coding_trend_recovery.py` after O2 main + nokin collect | DONE 10-25 ~09:40 on the O2 final-frame runs (ENCODING_FINDINGS 4.1-4.2; June-span trend test not repeatable) |
| P3 | Trial-average R2 | `E/qc/trialavg_r2_movement_first_subset-drop_licks-combined_gate-attempted.csv` | 10-22 08:39 | stale drop run | STALE-REDO (low) | | `python scripts/encoder_trialavg_r2.py --folder <O2 drop run>` | STALE (low) |
| P3 | Push LMM (behaviour part current) | `E/qc/imaging_lmm_push*.csv` | 10-23 18:39 | deficit index | STALE-REDO | `push_behaviour_sessions.csv` (DAQ treadmill) is CURRENT | `imaging_lmm.py --sources push` (planned LMM step) | STALE |

### 2b. SUPERSEDED

| family | location | made | replaced by |
|---|---|---|---|
| cam4 within-spout lick / contact / contact_hits | `E/qc/within_spout_deviation_lick*`, figures | 10-20/21 | ws3d4 (running, 3-D, 4 animals) |
| 2-animal xyz2 x combined; xyz2 x cam4 cell (CHECK) | `E/movement_first_subset-nokin_exec-xyz2_licks-combined_bins`, `_xyz2_bins` | 10-24 05:35-05:43 | xyz2 `_3d4` (running); cam4 cell low priority |
| Older duplicate LMM figures + early behaviour-figure copies | `E/qc/figures/` | 10-21 02:03-06:25 | `E/figures/mixed_models` (itself stale). **Inventory had KEEP.** |
| Frozen group leakage | `E/qc/frozen_group_leakage.csv` | 10-22 13:53 | design at STATS_VERSION 4 / grouping v1; groups changed since (v2, v3, v7 running). **Inventory had "current".** Optional: `scripts/frozen_group_leakage.py` in the t3d env |
| Whisker ME traces by attempt class | `session_poses/whisker_me` | 10-21 | FaceRhythm |
| v35 run; v36 `cohort/*_allpre`; `epoch/archive_rawdF_tests_20261021`; `qc/superseded_*`; session_poses legacy (lick_summary_*, lick_template_match*, movement_encoding_summary*, cam1_lick_aim*, jaw_lateral v34, pose3d/kinematics_setpoint, pose3d/figures maps pilot); FR rank-10 (`*_r10`) and pilot sets; ME pilot `movement_first_video-cam1-cam4`; FR pilot encoder runs | various | 10-02 - 10-23 | v36 / August-only / normalised tests / 3-D 4 animals / FR master r12 |
| `epoch_latepre` map figures (15r / 14) and the 10-18 `spout_position_analysis_summary*` decks (map slides) | `G/epoch_latepre`, `labcams/*.pptx` | 10-18 | contours drawn with the pre-10-21 raw tests; re-render only if quoted |
| Local polar + binned `_3d4` contest (CURRENT, see 2c) | `E/..._polar_licks-combined_bins_3d4` | | O2 `bins_w1` (attempted gate) becomes the primary binned contest |

### 2c. CURRENT

| family | location | made | why |
|---|---|---|---|
| trial_attempts (rebuilt) | `E/qc/trial_attempts*.csv` | 10-24 06:29 | jaw_onsets 10-23 19:54, cam1 complete, v36 |
| jaw_onsets | `E/qc/jaw_onsets.csv` | 10-23 18:27 / 19:54 | complete cam1 |
| first_lick_latency (+summary) | `E/qc/first_lick_latency*.csv` | 10-23 18:15 | pose_chain cohort_checks after the last cam1 files (17:45) |
| pose3d derived tables + figures | `session_poses/pose3d/{kinematics,kinematics_all,kinematics_combined,spout_error,phase,cue_traces,lick_traces,crosscheck,vs_cam4}`, `figures/summary_*` | 10-23 17:30-18:23 | all-animal 3-D |
| cam1 incomplete readout | `session_poses/cam1_licks/incomplete` | 10-23 17:46 | complete cam1 |
| 3-D encoding inputs | `wfield_cache/pose3d_inputs/*_20261023` | 10-23 18:29-18:33 | from final kinematics |
| Polar + binned 4-animal contest | `E/movement_first_subset-nokin_exec-polar_licks-combined_bins_3d4` | 10-23 18:50 - 10-24 03:35 | combined licks after cam1 complete, 20261023 tables; reference gate (O2 attempted-gate bins become primary) |
| Stray-lick test | `E/stray_licks` | 10-23 20:05-21:23 | kinematics_all 17:41; not gated |
| Contact vs direction | `E/qc/contact_direction_predictability.csv` | 10-23 19:20 | kinematics 17:30 |
| Push behaviour | `E/qc/push_behaviour_sessions.csv` | 10-23 18:39 | DAQ treadmill only |
| ENL LP census | `E/qc/enl_lp_census*.csv` | 10-21 18:08 | LP pre-cue train is cam4 only (cam1 events are post-cue), v36 retimed 10-20. **Inventory had REDO -- not needed.** |
| v36 cohort orofacial (prepost / jaw_lateral / w2 / heatmaps / misdirected / robustness / deck) | `LP_analysis/.../clean_v36/cohort` | 10-20 11:33 - 21:43 | v36 retimed 10-20; cam4 + reference gate by design |
| FaceRhythm master_pre_v1 rank 12 | `o2/fr_cam4_20261021/factors/master_pre_v1` | 10-24 05:49 - 06:30 (own8 still writing) | FR window owns |
| Attempt blind-review record | `session_poses/attempt_review` | 10-21 10:20 | historical QC; decision taken; 137 class flips in the rebuild do not touch it |

### 2d. NOT-AFFECTED (contact-only / imaging-only)

| family | location | made | note |
|---|---|---|---|
| DAQ epoch families 5c/5r/6/7/8/9/10/11/12/13/15d-x/23-26/1b-1g/cd_*/recovery | `G/epoch/` | 09-17 - 10-17 | contact gate; the lick-gated arms (lick_lick, cue_lick, precue_lick) are contact-gated -> the LP_licks versions are the better primary for the attempt question (acute far-contra lick cell is empty under DAQ) |
| DAQ maps 15r / 15rpa / 15e / 14 / 14pa / 14z (normalised tests) + calibration CSVs | `G/epoch/`, `G/epoch/results/` | 10-21 14:53 - 21:09 | made after the test change -> current |
| DAQ_ENL1s (matched contact 1.0 s ENL) | `G/DAQ_ENL1s/epoch/` | 10-22 00:12-00:59 | contact only (see UNSURE on its frozen 1.0 s set) |
| CD leave-one-out; histology; top-level grant_* (10-07 / 10-17); analysis_figures; channel_comparison; chronic_stability; null_potent; region_reference_contrast; enl_movement_multisession (DAQ 2 s ENL + ME); xday; joint_bases; labcams/frozen_models (state segments); local frozen store (10-18, August-only 2.0 s decoders/encoders = latepre set); behavior_summary (10-17); cue_lick (09-19..09-28); qc drift / slowband / position_blocks / precue_contacts | various | 06-01 - 10-21 | no cam1 / attempts / 3-D / encoder-version input. Caveat for ENL-based ones: DAQ 2 s "lick-free" windows contain non-contact licks (LP 1.0 s finding, DECISIONS 10-22) |

## 3. Gaps -- stale outputs NOT covered by the running / planned lists

1. `qc/lp_trial_counts*` -- LP vs DAQ per-animal counts; must be re-run before the LP re-render (not in the planned LP list).
2. `qc/lp_encoder_decompose*` -- the acute far-contra "carried by lick" decomposition; `run_v7g3_t3d.sh` does not run it
   and the O2 list has no decomposition step. Add it after t3d and after the O2 main collect.
3. O2 variants missing from `submit_all.sh`: nokin_matched (needs a new match.json), nokin_tr, nokin_win, drop_win.
4. Destination vs landing (2.4c): 2 animals, and `scripts/best_match_destination.py` `CONTEST_3D` points at a folder
   deleted 10-24 -> needs a one-line code change before it can re-run on `_3d4` / O2 bins.
5. Lick-template test (target / landing / "neither"): 2 animals; no 4-animal re-run planned.
6. Within-spout: cue-aligned base and ENL variants have no 3-D re-run; within_spout_mixed / spout_mid_maps (per-position
   DIRECTION maps -- the headline form per Priya 10-21) / spout_reliability are not chained after ws3d4.
7. `figures/behaviour/*` and `figures/target_vs_execution/*` (attempt classes, miss-then-correct, lick deficits,
   landing zones, destination, lick templates): all pre-final; inventory had them as KEEP.
8. `qc/lick_contact_by_epoch.csv` (stale attempts table, cam4-only lick level) and
   `qc/first_lick_latency_late_licked_trials.csv` (10-20 cam1): inventory had KEEP.
9. `qc/position_coding_trend_recovery*`, `qc/trialavg_r2_*`: need a re-run after the O2 collect; not listed.
10. Not stale but never done (10-21 LP plan): LP versions of CD trajectories and the non-epoch grant figures,
    PS93 far_ipsi / far_mid LP-frozen sensitivity (`WIDEFIELD_LP_FROZEN=lp`), optional `LP_licks_pre150`. The planned
    `attempt_coding` module covers the CD-on-attempt-classes part.
11. Inventory corrections: `enl_lp_census` is NOT a redo (cam4 pre-cue only); `frozen_group_leakage` and `qc/figures`
    are superseded, not current.

## 4. UNSURE

- **DAQ_ENL1s 1.0 s frozen precue decoders**: the local store (`C:/Users/SabatiniLab/frozen_models`) holds only the
  10-18 August-only 2.0 s set, and `labcams/frozen_models` only state segments. Where the 1.0 s set the LP ENL arm
  reuses was persisted (env `WIDEFIELD_FROZEN_MODEL_DIR` in the 10-22 run, or result_cache only) decides whether the LP
  ENL re-render can reuse it or refits it. Check the 10-22 run env / log before the LP re-render.
- `qc/enl_window_licks.csv` (10-20 22:05) and `qc/first_lick_latency_late_licked_trials.csv`: no producer in the repo
  (scratch scripts); the first looks DAQ-based (NOT-AFFECTED if so), the second is stale (cam1 10-20).
- `G/epoch` 15d / 15h / 15x map families (10-07): if their contours come from `beta_maps.significance_contour`, they
  carry the pre-10-21 raw tests (SUPERSEDED-definition); the 10-21 re-render covered 15r / 15e / 14m only.
- `qc/nose_onsets.csv` (10-21 07:46, cam1 incomplete): descriptive and unused (DECISIONS 10-21); stale only if quoted.
- Gate-only staleness: DECISIONS 10-24 says runs on the old attempts table are not re-run for the gate alone (flips
  0 / 0 / 93 / 62 trials). Everything in 2a that is stale ONLY through the gate (e.g. if one wanted to keep v7g3
  numbers) is formally stale but expected not to move; the LP_licks family is NOT in that category (PS95 cam4-only LP
  licks + gate).
