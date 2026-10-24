# Encoding-window output inventory (2026-10-24)

> **SUPERSEDED IN PART (10-24 later): the full-scope audit is `OUTPUT_AUDIT_2026-10-24.md`.** This inventory missed the
> LP_licks family (`grant_figures/LP_licks/`) and had several figures / qc tables as KEEP that are stale (audit §2a, §3.11).

Written for: Priya and the next encoding session. Scope: the encoding window's outputs in
`LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/` (run folders, `qc/`, `figures/`), the O2
encoding bundle `DeepLabCut/Widefield/o2/enc_v36_20261019/`, and the local input cache. Other windows' outputs
(pose, LP training, FaceRhythm bundles) are not listed. Regenerate: `S/inventory.py` then `S/classify_inventory.py`
(scratchpad scripts; copy into `scripts/` if this becomes routine).

## Data versions that decide whether an output is current

| axis | outdated | FULL (target) |
|---|---|---|
| lick timing | cam4 only (`licks=cam4`, all O2 runs 10-19/20) | `combined` cam4 + cam1, cam1 complete since 10-23 17:45 (PS95 last) |
| engagement gate | `reference`, or `attempted` from the 10-21 attempts table (cam1 on PS94 40 % / PS95 0 % of trials) | `attempted` from the REBUILT attempts table (10-24) |
| 3-D | 10-21 snapshot (PS92 + PS93) | 20261023 re-cut (4 animals, 107 encoding sessions) |
| target vs executed readout | cam4 angle sets; continuous polar / xyz2 | BINNED contest primary (DECISIONS 10-24) + executed beyond category; polar / xyz2 sensitivity |
| encoder tongue traces | cam4 px | 3-D (`COMBINED_ENCODING_TONGUE=3d`, 9f45819) |
| frozen encoder | stats v3-v6, grouping v1-v2 | stats v7, grouping v3 |
| face | none / FR pilot rank 10 | FaceRhythm master rank 12 (pending: proj + labels.json + grouping) |

**Full data = phase A** (combined licks + complete 3-D + 3-D tongue + rebuilt gate), **then phase B** (+ FR rank 12).

## Totals

| class | entries | GB | meaning |
|---|---|---|---|
| REDO | 58 | 41.2 | incomplete data; keep until its phase-A re-run lands, then delete |
| DELETE-NOW | 147 | 53.0 | far superseded (old model / incomplete data / smoke / duplicates) -- proposed, needs Priya's OK |
| DELETE-AFTER | 16 | 44.1 | superseded once the named re-run lands |
| CHECK | 3 | 2.9 | Priya decides |
| KEEP | 47 | 3.6 | current |

Local input cache `C:/Users/SabatiniLab/wfield_cache`: **304 GB** (imaging 129, behaviour 175). Disposable by
design (keys include code hash + inputs, so most entries belong to old code). Proposal: delete entries not modified
since 10-23 (before the current code) once no run is using them.

## REDO

| where | name | GB | last written | data version / reason | replaced by |
|---|---|---|---|---|---|
| qc | `best_match_destination.csv` | 0.00 | 10-21 07:22 | destination vs landing, 2 animals (10-21 snapshot) | 4 animals |
| qc | `best_match_destination_cue-working.csv` | 0.00 | 10-23 06:21 | destination vs landing, 2 animals (10-21 snapshot) | 4 animals |
| qc | `best_match_destination_lick_lick.csv` | 0.00 | 10-23 06:21 | destination vs landing, 2 animals (10-21 snapshot) | 4 animals |
| qc | `best_match_destination_lick_lick_summary.csv` | 0.00 | 10-23 06:21 | destination vs landing, 2 animals (10-21 snapshot) | 4 animals |
| qc | `best_match_destination_summary.csv` | 0.00 | 10-21 07:22 | destination vs landing, 2 animals (10-21 snapshot) | 4 animals |
| qc | `best_match_destination_summary_cue-working.csv` | 0.00 | 10-23 06:21 | destination vs landing, 2 animals (10-21 snapshot) | 4 animals |
| qc | `encoding_mixed.csv` | 0.00 | 10-21 14:35 | LMMs 10-21: stale attempts / deficit, cam4-era encoding measures | after attempts + re-runs |
| qc | `encoding_mixed_contact.csv` | 0.00 | 10-21 02:15 | LMMs 10-21: stale attempts / deficit, cam4-era encoding measures | after attempts + re-runs |
| qc | `encoding_mixed_summary.csv` | 0.00 | 10-21 14:35 | LMMs 10-21: stale attempts / deficit, cam4-era encoding measures | after attempts + re-runs |
| qc | `encoding_mixed_summary_contact.csv` | 0.00 | 10-21 02:15 | LMMs 10-21: stale attempts / deficit, cam4-era encoding measures | after attempts + re-runs |
| qc | `enl_lp_census.csv` | 0.01 | 10-21 18:08 | LP-lick counts 10-21 (PS94/95 cam1 incomplete, stale attempts) | after attempts |
| qc | `enl_lp_census_summary.csv` | 0.00 | 10-21 18:08 | LP-lick counts 10-21 (PS94/95 cam1 incomplete, stale attempts) | after attempts |
| qc | `imaging_lmm.csv` | 0.00 | 10-21 14:36 | LMMs 10-21: stale attempts / deficit, cam4-era encoding measures | after attempts + re-runs |
| qc | `imaging_lmm_contact.csv` | 0.00 | 10-21 02:13 | LMMs 10-21: stale attempts / deficit, cam4-era encoding measures | after attempts + re-runs |
| qc | `imaging_lmm_push.csv` | 0.00 | 10-23 18:39 | push LMM: behaviour is DAQ-only but the deficit index is stale | after attempts rebuild |
| qc | `imaging_lmm_summary.csv` | 0.00 | 10-21 14:36 | LMMs 10-21: stale attempts / deficit, cam4-era encoding measures | after attempts + re-runs |
| qc | `imaging_lmm_summary_contact.csv` | 0.00 | 10-21 02:13 | LMMs 10-21: stale attempts / deficit, cam4-era encoding measures | after attempts + re-runs |
| qc | `imaging_lmm_summary_push.csv` | 0.00 | 10-23 18:39 | push LMM: behaviour is DAQ-only but the deficit index is stale | after attempts rebuild |
| qc | `lick_templates_polar_20261021` | 0.00 | 10-21 14:03 | lick-template test, 2 animals | 4 animals |
| qc | `lp_encoder_decompose.csv` | 0.00 | 10-22 07:21 | decompositions of cam4-era / stale-gate runs | O2 bundle |
| qc | `lp_encoder_decompose_by_animal.csv` | 0.00 | 10-22 07:21 | decompositions of cam4-era / stale-gate runs | O2 bundle |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_v7fwd15_joint.csv` | 0.00 | 10-23 15:47 | decompositions of v7 runs (stale gate) | final frame |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_v7fwd15_joint_by_animal.csv` | 0.00 | 10-23 15:47 | decompositions of v7 runs (stale gate) | final frame |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_v7fwd15_joint_lick.csv` | 0.00 | 10-23 15:47 | decompositions of v7 runs (stale gate) | final frame |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_v7fwd15_joint_lick_by_animal.csv` | 0.00 | 10-23 15:47 | decompositions of v7 runs (stale gate) | final frame |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_v7pushfree_joint.csv` | 0.00 | 10-23 22:10 | decompositions of v7 runs (stale gate) | final frame |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_v7pushfree_joint_by_animal.csv` | 0.00 | 10-23 22:10 | decompositions of v7 runs (stale gate) | final frame |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_v7pushfree_joint_lick.csv` | 0.00 | 10-23 22:10 | decompositions of v7 runs (stale gate) | final frame |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_v7pushfree_joint_lick_by_animal.csv` | 0.00 | 10-23 22:10 | decompositions of v7 runs (stale gate) | final frame |
| qc | `lp_encoder_decompose_movement_first_subset-full_licks-combined_gate-attempted_lick.csv` | 0.00 | 10-22 16:58 | decompositions of cam4-era / stale-gate runs | O2 bundle |
| qc | `lp_encoder_decompose_movement_first_subset-full_licks-combined_gate-attempted_lick_by_animal.csv` | 0.00 | 10-22 16:58 | decompositions of cam4-era / stale-gate runs | O2 bundle |
| qc | `lp_trial_counts.csv` | 0.00 | 10-21 21:35 | LP-lick counts 10-21 (PS94/95 cam1 incomplete, stale attempts) | after attempts |
| qc | `lp_trial_counts_summary.csv` | 0.00 | 10-21 21:35 | LP-lick counts 10-21 (PS94/95 cam1 incomplete, stale attempts) | after attempts |
| qc | `position_coding_half_recovery.csv` | 0.00 | 10-20 16:20 | recovery trend from cam4-era runs | after O2 re-runs |
| qc | `position_coding_trend_recovery.csv` | 0.00 | 10-20 16:20 | recovery trend from cam4-era runs | after O2 re-runs |
| qc | `position_coding_trend_recovery_sessions.csv` | 0.00 | 10-20 16:20 | recovery trend from cam4-era runs | after O2 re-runs |
| qc | `push_behaviour_sessions.csv` | 0.00 | 10-23 18:39 | push LMM: behaviour is DAQ-only but the deficit index is stale | after attempts rebuild |
| qc | `trial_attempts.csv` | 0.01 | 10-21 14:33 | cam1 on PS94 40 % / PS95 0 % of trials | REBUILD running |
| qc | `trial_attempts_gate.csv` | 0.00 | 10-21 14:33 | cam1 on PS94 40 % / PS95 0 % of trials | REBUILD running |
| qc | `trial_attempts_sessions.csv` | 0.00 | 10-21 14:33 | cam1 on PS94 40 % / PS95 0 % of trials | REBUILD running |
| qc | `trialavg_r2_movement_first_subset-drop_licks-combined_gate-attempted.csv` | 0.00 | 10-22 08:39 | decompositions of cam4-era / stale-gate runs | O2 bundle |
| run | `allsession_encoder_v7push` | 0.22 | 10-23 23:03 | Layer 2, v7pushfree frame, stale gate | Layer 2 in final frame |
| run | `frozen_encoder_gate-attempted_v7g3` | 3.66 | 10-24 00:07 | current frame but STALE GATE (PS94/95) and cam4 tongue | v7g3_t3d (new gate) |
| run | `frozen_encoder_gate-attempted_v7pushfree` | 3.77 | 10-23 22:10 | current frame but STALE GATE (PS94/95) and cam4 tongue | v7g3_t3d (new gate) |
| run | `frozen_relabel_v7` | 0.06 | 10-24 01:01 | relabel, v7fwd15 frame, stale gate | relabel in final frame |
| run | `movement_first` | 7.15 | 10-22 12:30 | O2 10-19/20: CAM4 lick timing, reference gate, cam4 tongue | O2 bundle (combined licks + 3-D tongue + attempted gate) |
| run | `movement_first_subset-drop_licks-combined_gate-attempted` | 2.95 | 10-22 04:56 | combined licks but 27 PS95 sessions pre-final cam1, stale gate, cam4 tongue | O2 bundle |
| run | `movement_first_subset-drop_win` | 2.95 | 10-21 06:33 | O2 10-19/20: CAM4 lick timing, reference gate, cam4 tongue | O2 bundle (combined licks + 3-D tongue + attempted gate) |
| run | `movement_first_subset-droppos-` | 2.26 | 10-20 21:09 | O2 10-19/20: CAM4 lick timing, reference gate, cam4 tongue | O2 bundle (combined licks + 3-D tongue + attempted gate) |
| run | `movement_first_subset-full_licks-combined_gate-attempted` | 1.32 | 10-22 16:52 | combined licks but 27 PS95 sessions pre-final cam1, stale gate, cam4 tongue | O2 bundle |
| run | `movement_first_subset-nodir` | 1.67 | 10-20 20:52 | O2 10-19/20: CAM4 lick timing, reference gate, cam4 tongue | O2 bundle (combined licks + 3-D tongue + attempted gate) |
| run | `movement_first_subset-nokin` | 1.90 | 10-20 20:44 | O2 10-19/20: CAM4 lick timing, reference gate, cam4 tongue | O2 bundle (combined licks + 3-D tongue + attempted gate) |
| run | `movement_first_subset-nokin_matched` | 1.87 | 10-20 20:46 | O2 10-19/20: CAM4 lick timing, reference gate, cam4 tongue | O2 bundle (combined licks + 3-D tongue + attempted gate) |
| run | `movement_first_subset-nokin_tr` | 1.74 | 10-20 21:43 | O2 10-19/20: CAM4 lick timing, reference gate, cam4 tongue | O2 bundle (combined licks + 3-D tongue + attempted gate) |
| run | `movement_first_subset-nokin_win` | 2.84 | 10-21 06:25 | O2 10-19/20: CAM4 lick timing, reference gate, cam4 tongue | O2 bundle (combined licks + 3-D tongue + attempted gate) |
| run | `movement_first_subset-shufpos-` | 2.26 | 10-20 20:55 | O2 10-19/20: CAM4 lick timing, reference gate, cam4 tongue | O2 bundle (combined licks + 3-D tongue + attempted gate) |
| run | `movement_first_subset-trmodels` | 2.51 | 10-20 20:53 | O2 10-19/20: CAM4 lick timing, reference gate, cam4 tongue | O2 bundle (combined licks + 3-D tongue + attempted gate) |
| run | `musall_subset-shuf` | 2.09 | 10-20 20:53 | O2 10-19/20: CAM4 lick timing, reference gate, cam4 tongue | O2 bundle (combined licks + 3-D tongue + attempted gate) |

## DELETE-NOW

| where | name | GB | last written | data version / reason | replaced by |
|---|---|---|---|---|---|
| o2 | `inputs_superseded_pre_template_timing` | 0.65 | 10-19 07:36 | pre-template-timing inputs (3 sessions) | - |
| o2 | `out` | 20.13 | 10-21 00:19 | duplicates of collected results already in imaging_encoding/ | - |
| qc | `Thumbs.db` | 0.00 | 10-20 11:24 | marked superseded | - |
| qc | `exec_angle_spread.csv` | 0.00 | 10-19 08:06 | cam4 executed-angle spread (10-19), superseded by 3-D | - |
| qc | `exec_angle_spread.png` | 0.00 | 10-19 08:06 | cam4 executed-angle spread (10-19), superseded by 3-D | - |
| qc | `frozen_group_leakage_v3_SUPERSEDED.csv` | 0.00 | 10-22 13:14 | marked superseded | - |
| qc | `frozen_group_leakage_v4cuelocked_SUPERSEDED.csv` | 0.00 | 10-22 13:39 | marked superseded | - |
| qc | `lick_templates_polar` | 0.00 | 10-21 00:24 | first-pass lick templates (20261020 tables); rerun quoted | lick_templates_polar_20261021 |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_joint.csv` | 0.00 | 10-22 11:08 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_joint_by_animal.csv` | 0.00 | 10-22 11:08 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_joint_lick.csv` | 0.00 | 10-22 11:08 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_joint_lick_by_animal.csv` | 0.00 | 10-22 11:08 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_movement_first.csv` | 0.00 | 10-22 11:08 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_movement_first_by_animal.csv` | 0.00 | 10-22 11:08 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_movement_first_lick.csv` | 0.00 | 10-22 11:08 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_movement_first_lick_by_animal.csv` | 0.00 | 10-22 11:08 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_movement_only.csv` | 0.00 | 10-22 11:10 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_movement_only_by_animal.csv` | 0.00 | 10-22 11:10 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_movement_only_lick.csv` | 0.00 | 10-22 11:10 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_movement_only_lick_by_animal.csv` | 0.00 | 10-22 11:10 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_no_movement.csv` | 0.00 | 10-22 11:09 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_no_movement_by_animal.csv` | 0.00 | 10-22 11:09 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_no_movement_lick.csv` | 0.00 | 10-22 11:09 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_no_movement_lick_by_animal.csv` | 0.00 | 10-22 11:09 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_no_task.csv` | 0.00 | 10-22 11:10 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_no_task_by_animal.csv` | 0.00 | 10-22 11:10 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_no_task_lick.csv` | 0.00 | 10-22 11:10 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_no_task_lick_by_animal.csv` | 0.00 | 10-22 11:10 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_task_first.csv` | 0.00 | 10-22 11:09 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_task_first_by_animal.csv` | 0.00 | 10-22 11:09 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_task_first_lick.csv` | 0.00 | 10-22 11:09 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_task_first_lick_by_animal.csv` | 0.00 | 10-22 11:09 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_task_movement.csv` | 0.00 | 10-22 11:09 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_task_movement_by_animal.csv` | 0.00 | 10-22 11:09 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_task_movement_dir.csv` | 0.00 | 10-22 11:09 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_task_movement_dir_by_animal.csv` | 0.00 | 10-22 11:09 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_task_movement_dir_lick.csv` | 0.00 | 10-22 11:09 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_task_movement_dir_lick_by_animal.csv` | 0.00 | 10-22 11:09 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_task_movement_lick.csv` | 0.00 | 10-22 11:09 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_task_movement_lick_by_animal.csv` | 0.00 | 10-22 11:09 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_v4_exec_first.csv` | 0.00 | 10-22 16:55 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_v4_exec_first_by_animal.csv` | 0.00 | 10-22 16:55 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_v4_exec_first_lick.csv` | 0.00 | 10-22 16:55 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_v4_exec_first_lick_by_animal.csv` | 0.00 | 10-22 16:55 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_v4_joint.csv` | 0.00 | 10-22 16:55 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_v4_joint_by_animal.csv` | 0.00 | 10-22 16:55 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_v4_joint_lick.csv` | 0.00 | 10-22 16:55 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_v4_joint_lick_by_animal.csv` | 0.00 | 10-22 16:55 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_v4_movement_first.csv` | 0.00 | 10-22 16:55 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_v4_movement_first_by_animal.csv` | 0.00 | 10-22 16:55 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_v4_movement_first_lick.csv` | 0.00 | 10-22 16:55 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_v4_movement_first_lick_by_animal.csv` | 0.00 | 10-22 16:55 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_v4_target_first.csv` | 0.00 | 10-22 16:55 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_v4_target_first_by_animal.csv` | 0.00 | 10-22 16:55 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_v4_target_first_lick.csv` | 0.00 | 10-22 16:55 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_v4_target_first_lick_by_animal.csv` | 0.00 | 10-22 16:55 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_v4_task_first.csv` | 0.00 | 10-22 16:55 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_v4_task_first_by_animal.csv` | 0.00 | 10-22 16:55 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_v4_task_first_lick.csv` | 0.00 | 10-22 16:55 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_v4_task_first_lick_by_animal.csv` | 0.00 | 10-22 16:55 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_v5_exec_first.csv` | 0.00 | 10-22 20:25 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_v5_exec_first_by_animal.csv` | 0.00 | 10-22 20:25 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_v5_exec_first_lick.csv` | 0.00 | 10-22 20:25 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_v5_exec_first_lick_by_animal.csv` | 0.00 | 10-22 20:25 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_v5_joint.csv` | 0.00 | 10-22 20:25 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_v5_joint_by_animal.csv` | 0.00 | 10-22 20:25 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_v5_joint_lick.csv` | 0.00 | 10-22 20:25 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_v5_joint_lick_by_animal.csv` | 0.00 | 10-22 20:25 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_v5_movement_first.csv` | 0.00 | 10-22 20:25 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_v5_movement_first_by_animal.csv` | 0.00 | 10-22 20:25 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_v5_movement_first_lick.csv` | 0.00 | 10-22 20:25 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_v5_movement_first_lick_by_animal.csv` | 0.00 | 10-22 20:25 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_v5_target_first.csv` | 0.00 | 10-22 20:25 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_v5_target_first_by_animal.csv` | 0.00 | 10-22 20:25 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_v5_target_first_lick.csv` | 0.00 | 10-22 20:25 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_v5_target_first_lick_by_animal.csv` | 0.00 | 10-22 20:25 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_v5_task_first.csv` | 0.00 | 10-22 20:25 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_v5_task_first_by_animal.csv` | 0.00 | 10-22 20:25 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_v5_task_first_lick.csv` | 0.00 | 10-22 20:25 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_v5_task_first_lick_by_animal.csv` | 0.00 | 10-22 20:25 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_v6_exec_first.csv` | 0.00 | 10-22 22:54 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_v6_exec_first_by_animal.csv` | 0.00 | 10-22 22:54 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_v6_exec_first_lick.csv` | 0.00 | 10-22 22:54 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_v6_exec_first_lick_by_animal.csv` | 0.00 | 10-22 22:54 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_v6_joint.csv` | 0.00 | 10-22 22:54 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_v6_joint_by_animal.csv` | 0.00 | 10-22 22:54 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_v6_joint_lick.csv` | 0.00 | 10-22 22:54 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_v6_joint_lick_by_animal.csv` | 0.00 | 10-22 22:54 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_v6_movement_first.csv` | 0.00 | 10-22 22:55 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_v6_movement_first_by_animal.csv` | 0.00 | 10-22 22:55 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_v6_movement_first_lick.csv` | 0.00 | 10-22 22:55 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_v6_movement_first_lick_by_animal.csv` | 0.00 | 10-22 22:55 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_v6_target_first.csv` | 0.00 | 10-22 22:54 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_v6_target_first_by_animal.csv` | 0.00 | 10-22 22:54 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_v6_target_first_lick.csv` | 0.00 | 10-22 22:54 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_v6_target_first_lick_by_animal.csv` | 0.00 | 10-22 22:54 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_v6_task_first.csv` | 0.00 | 10-22 22:54 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_v6_task_first_by_animal.csv` | 0.00 | 10-22 22:54 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_v6_task_first_lick.csv` | 0.00 | 10-22 22:54 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_v6_task_first_lick_by_animal.csv` | 0.00 | 10-22 22:54 | decompositions of deleted frozen v3-v6 runs | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_v6spl_joint.csv` | 0.00 | 10-23 07:21 | decompositions of running-model candidates | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_v6spl_joint_by_animal.csv` | 0.00 | 10-23 07:21 | decompositions of running-model candidates | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_v6spl_joint_lick.csv` | 0.00 | 10-23 07:21 | decompositions of running-model candidates | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_v6spl_joint_lick_by_animal.csv` | 0.00 | 10-23 07:21 | decompositions of running-model candidates | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_v6thr1_joint.csv` | 0.00 | 10-23 10:10 | decompositions of running-model candidates | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_v6thr1_joint_by_animal.csv` | 0.00 | 10-23 10:10 | decompositions of running-model candidates | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_v6thr1_joint_lick.csv` | 0.00 | 10-23 10:10 | decompositions of running-model candidates | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_v6thr1_joint_lick_by_animal.csv` | 0.00 | 10-23 10:10 | decompositions of running-model candidates | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_v6thr3_joint.csv` | 0.00 | 10-23 10:10 | decompositions of running-model candidates | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_v6thr3_joint_by_animal.csv` | 0.00 | 10-23 10:10 | decompositions of running-model candidates | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_v6thr3_joint_lick.csv` | 0.00 | 10-23 10:10 | decompositions of running-model candidates | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_v6thr3_joint_lick_by_animal.csv` | 0.00 | 10-23 10:10 | decompositions of running-model candidates | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_v7fwd15push_joint.csv` | 0.00 | 10-23 15:47 | decomposition of frozen-push run | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_v7fwd15push_joint_by_animal.csv` | 0.00 | 10-23 15:47 | decomposition of frozen-push run | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_v7fwd15push_joint_lick.csv` | 0.00 | 10-23 15:47 | decomposition of frozen-push run | - |
| qc | `lp_encoder_decompose_frozen_encoder_gate-attempted_v7fwd15push_joint_lick_by_animal.csv` | 0.00 | 10-23 15:47 | decomposition of frozen-push run | - |
| qc | `superseded_attempts_jaw_20261021` | 0.02 | 10-21 10:11 | superseded 10-20/21 (already moved aside) | - |
| qc | `superseded_contact_index_20261020` | 0.00 | 10-20 23:11 | superseded 10-20/21 (already moved aside) | - |
| qc | `superseded_within_spout_contact_unadj_20261021` | 0.04 | 10-21 11:22 | superseded 10-20/21 (already moved aside) | - |
| qc | `superseded_within_spout_lick_pooled_20261021` | 0.40 | 10-20 23:18 | superseded 10-20/21 (already moved aside) | - |
| qc | `superseded_within_spout_polar_pooled_20261021` | 0.18 | 10-20 23:20 | superseded 10-20/21 (already moved aside) | - |
| run | `frozen_encoder_gate-attempted` | 1.81 | 10-22 13:00 | frozen v3 stats, 27 sessions pre-final cam1, stale gate | v7g3 |
| run | `frozen_encoder_gate-attempted_v4` | 3.62 | 10-22 19:02 | frozen v4-v6 (old running model / stats), 25-27 sessions pre-final cam1 | v7g3 |
| run | `frozen_encoder_gate-attempted_v5` | 3.62 | 10-22 21:56 | frozen v4-v6 (old running model / stats), 25-27 sessions pre-final cam1 | v7g3 |
| run | `frozen_encoder_gate-attempted_v5basis` | 0.61 | 10-22 21:56 | frozen v4-v6 (old running model / stats), 25-27 sessions pre-final cam1 | v7g3 |
| run | `frozen_encoder_gate-attempted_v5grid` | 0.28 | 10-22 21:56 | frozen v4-v6 (old running model / stats), 25-27 sessions pre-final cam1 | v7g3 |
| run | `frozen_encoder_gate-attempted_v6` | 3.62 | 10-22 22:56 | frozen v4-v6 (old running model / stats), 25-27 sessions pre-final cam1 | v7g3 |
| run | `frozen_encoder_gate-attempted_v6basis` | 0.62 | 10-23 07:08 | frozen v4-v6 (old running model / stats), 25-27 sessions pre-final cam1 | v7g3 |
| run | `frozen_encoder_gate-attempted_v6grid` | 0.28 | 10-22 23:50 | frozen v4-v6 (old running model / stats), 25-27 sessions pre-final cam1 | v7g3 |
| run | `frozen_encoder_gate-attempted_v6spl` | 3.59 | 10-23 07:20 | running-model candidates (v7 choice made; numbers in qc/frozen_running_compare_* + DECISIONS 10-23) | v7g3 |
| run | `frozen_encoder_gate-attempted_v6thr1` | 3.59 | 10-23 13:24 | running-model candidates (v7 choice made; numbers in qc/frozen_running_compare_* + DECISIONS 10-23) | v7g3 |
| run | `frozen_encoder_gate-attempted_v6thr3` | 3.59 | 10-23 10:10 | running-model candidates (v7 choice made; numbers in qc/frozen_running_compare_* + DECISIONS 10-23) | v7g3 |
| run | `frozen_encoder_gate-attempted_v7basis_g3` | 0.00 |  | empty / stopped at template pass | - |
| run | `frozen_encoder_gate-attempted_v7fwd15push` | 3.59 | 10-23 15:47 | frozen push (fails post-stroke windows); superseded by free push | v7pushfree |
| run | `frozen_encoder_gate-attempted_v7g3_t3d` | 0.00 |  | empty / stopped at template pass | - |
| run | `frozen_encoder_smoke-basis` | 0.01 | 10-22 19:04 | smoke tests (PS92, 2-4 sessions) | - |
| run | `frozen_encoder_smoke-grid` | 0.09 | 10-22 19:12 | smoke tests (PS92, 2-4 sessions) | - |
| run | `frozen_encoder_smoke-r2` | 0.20 | 10-23 06:18 | smoke tests (PS92, 2-4 sessions) | - |
| run | `frozen_encoder_smoke-r2b` | 0.21 | 10-23 13:31 | smoke tests (PS92, 2-4 sessions) | - |
| run | `frozen_encoder_smoke-v4` | 0.11 | 10-22 13:13 | smoke tests (PS92, 2-4 sessions) | - |
| run | `frozen_relabel_smoke` | 0.02 | 10-23 20:04 | smoke tests (PS92, 2-4 sessions) | - |
| run | `movement_first_subset-nokin_exec-3d` | 0.18 | 10-20 07:16 | PS92-only 3-D pilot (11 sessions, cam4 timing, 3dxyz retired); numbers kept in FINDINGS 2.4 | polar/xyz2 _3d4 |
| run | `movement_first_subset-nokin_exec-3dxyz` | 0.18 | 10-20 07:16 | PS92-only 3-D pilot (11 sessions, cam4 timing, 3dxyz retired); numbers kept in FINDINGS 2.4 | polar/xyz2 _3d4 |
| run | `movement_first_subset-nokin_exec-cam4m` | 0.18 | 10-20 06:05 | PS92-only 3-D pilot (11 sessions, cam4 timing, 3dxyz retired); numbers kept in FINDINGS 2.4 | polar/xyz2 _3d4 |
| run | `movement_first_subset-nokin_exec-polar_bins` | 0.47 | 10-23 18:14 | polar x cam4 timing, PS92 16 sessions (killed) | - |
| run | `movement_first_subset-nokin_exec-polar_licks-combined_bins` | 0.92 | 10-21 02:00 | 2-animal polar (10-21 snapshot, 7 pre-final cam1); _3d4 reproduces it | polar _3d4 |
| run | `movement_first_subset-nokin_exec-xyz2` | 0.18 | 10-20 07:27 | PS92-only 3-D pilot (11 sessions, cam4 timing, 3dxyz retired); numbers kept in FINDINGS 2.4 | polar/xyz2 _3d4 |

## DELETE-AFTER

| where | name | GB | last written | data version / reason | replaced by |
|---|---|---|---|---|---|
| o2 | `inputs` | 30.77 | 10-20 21:25 | O2 bundle inputs 10-20: cam4 licks, reference gate, cam4 tongue | new O2 bundle |
| qc | `superseded_2animal_20261021_within_spout_deviation_polar_lick` | 0.42 | 10-21 21:00 | 2-animal 3-D within-spout | ws 3-D 4 animals (running) |
| qc | `superseded_2animal_20261021_within_spout_deviation_polar_lick_contact` | 0.41 | 10-21 21:04 | 2-animal 3-D within-spout | ws 3-D 4 animals (running) |
| qc | `superseded_2animal_20261021_within_spout_deviation_polar_lick_contact_hits` | 0.40 | 10-21 21:08 | 2-animal 3-D within-spout | ws 3-D 4 animals (running) |
| qc | `superseded_attempts_cam1incomplete_20261021` | 0.01 | 10-21 14:33 | stale attempts tables (kept for the flip comparison) | attempts rebuild compared |
| qc | `within_spout_deviation` | 0.26 | 10-21 20:36 | within-spout direction, CAM4, 4 animals | ws 3-D 4 animals (running) |
| qc | `within_spout_deviation_lick` | 0.94 | 10-21 20:40 | within-spout direction, CAM4, 4 animals | ws 3-D 4 animals (running) |
| qc | `within_spout_deviation_lick_contact` | 0.93 | 10-21 20:46 | within-spout direction, CAM4, 4 animals | ws 3-D 4 animals (running) |
| qc | `within_spout_deviation_lick_contact_hits` | 0.92 | 10-21 20:54 | within-spout direction, CAM4, 4 animals | ws 3-D 4 animals (running) |
| qc | `within_spout_deviation_lick_enl` | 0.40 | 10-21 20:56 | within-spout direction, CAM4, 4 animals | ws 3-D 4 animals (running) |
| run | `frozen_encoder_gate-attempted_v7basis` | 2.75 | 10-23 20:02 | basis maps, grouping v2, cam4 tongue, PS95_0817/0831 pre-final cam1 | v7basis in final frame (t3d) |
| run | `frozen_encoder_gate-attempted_v7fwd15` | 3.59 | 10-23 15:47 | v7 cam4 tongue, 2 PS95 sessions pre-final cam1, stale gate; relabel templates come from its fit/ | v7g3_t3d + relabel rerun |
| run | `movement_first_fr-pilot_pre_PS92_PS93_subset-fr_licks-combined_gate-attempted_frA` | 0.92 | 10-23 07:36 | FaceRhythm pilot r10 / preview (2 animals or 1 session) | FR master r12 run |
| run | `movement_first_fr-preview_r10_PS92_subset-fr` | 0.02 | 10-22 08:21 | FaceRhythm pilot r10 / preview (2 animals or 1 session) | FR master r12 run |
| run | `movement_first_subset-nokin_exec-xyz2_licks-combined_bins` | 1.40 | 10-24 05:39 | 2-animal xyz2 (matched-session comparison done) | xyz2 _3d4 (running) |
| run | `movement_first_video-cam1-cam4_fr-pilot_pre_PS92_PS93_pts10_subset-fr-frvideo-frpts-frboth-video_licks-combined_gate-attempted` | 0.02 | 10-22 12:56 | FaceRhythm pilot r10 / preview (2 animals or 1 session) | FR master r12 run |

## CHECK

| where | name | GB | last written | data version / reason | replaced by |
|---|---|---|---|---|---|
| qc | `nose_onsets.csv` | 0.01 | 10-21 07:46 | nose onsets 10-21 (pose window's; cam1 incomplete then) | pose window |
| run | `movement_first_subset-nokin_exec-xyz2_bins` | 1.40 | 10-24 05:43 | xyz2 x cam4 timing, 2 animals (the 2x2 timing cell) | 4-animal cam4 cells (low priority) |
| run | `movement_first_video-cam1-cam4` | 1.47 | 10-19 05:10 | motion-energy video pilot, 23 sessions, 10-19, cam4 timing | FaceRhythm master r12 run |

## KEEP

| where | name | GB | last written | data version / reason | replaced by |
|---|---|---|---|---|---|
| figures | `behaviour` | 0.00 | 10-21 14:05 | small: figures / O2 bundle scaffolding | - |
| figures | `mixed_models` | 0.01 | 10-23 18:40 | small: figures / O2 bundle scaffolding | - |
| figures | `target_vs_execution` | 0.00 | 10-21 14:05 | small: figures / O2 bundle scaffolding | - |
| figures | `within_spout` | 0.10 | 10-21 21:09 | small: figures / O2 bundle scaffolding | - |
| o2 | `COMMANDS.md` | 0.00 | 10-20 21:25 | small: figures / O2 bundle scaffolding | - |
| o2 | `code` | 0.01 | 10-20 21:24 | small: figures / O2 bundle scaffolding | - |
| o2 | `enc_runner.py` | 0.00 | 10-20 21:25 | small: figures / O2 bundle scaffolding | - |
| o2 | `labels.txt` | 0.00 | 10-20 21:25 | small: figures / O2 bundle scaffolding | - |
| o2 | `logs` | 0.00 |  | small: figures / O2 bundle scaffolding | - |
| o2 | `match.json` | 0.00 | 10-19 12:31 | small: figures / O2 bundle scaffolding | - |
| o2 | `run_enc.sbatch` | 0.00 | 10-20 21:25 | small: figures / O2 bundle scaffolding | - |
| qc | `_cache` | 0.02 | 10-23 19:54 | imaging / behaviour only or small; not affected | - |
| qc | `contact_direction_predictability.csv` | 0.00 | 10-23 19:20 | current | - |
| qc | `drift_blocks_heatmap.png` | 0.00 | 10-18 19:34 | imaging / behaviour only or small; not affected | - |
| qc | `drift_cohort.png` | 0.00 | 10-18 19:34 | imaging / behaviour only or small; not affected | - |
| qc | `drift_examples.png` | 0.00 | 10-18 19:34 | imaging / behaviour only or small; not affected | - |
| qc | `drift_sessions.csv` | 0.00 | 10-18 19:34 | imaging / behaviour only or small; not affected | - |
| qc | `drift_traces.npz` | 0.00 | 10-18 19:34 | imaging / behaviour only or small; not affected | - |
| qc | `enl_window_licks.csv` | 0.00 | 10-20 22:05 | imaging / behaviour only or small; not affected | - |
| qc | `figures` | 0.01 | 10-21 06:25 | imaging / behaviour only or small; not affected | - |
| qc | `first_lick_latency.csv` | 0.00 | 10-23 18:15 | current | - |
| qc | `first_lick_latency_late_licked_trials.csv` | 0.00 | 10-20 19:21 | imaging / behaviour only or small; not affected | - |
| qc | `first_lick_latency_summary.csv` | 0.00 | 10-23 18:15 | current | - |
| qc | `frozen_group_leakage.csv` | 0.00 | 10-22 13:53 | current | - |
| qc | `frozen_running_compare_v6_to_v6thr1.csv` | 0.00 | 10-23 14:21 | evidence for the v7 running choice (small) | - |
| qc | `frozen_running_compare_v6_to_v6thr1_curves.csv` | 0.00 | 10-23 14:21 | evidence for the v7 running choice (small) | - |
| qc | `frozen_running_compare_v6_to_v7fwd15push.csv` | 0.00 | 10-23 15:48 | evidence for the v7 running choice (small) | - |
| qc | `frozen_running_compare_v6_to_v7fwd15push_curves.csv` | 0.00 | 10-23 15:48 | evidence for the v7 running choice (small) | - |
| qc | `frozen_running_compare_v7fwd15_to_v7pushfree.csv` | 0.00 | 10-23 22:11 | current | - |
| qc | `frozen_running_compare_v7fwd15_to_v7pushfree_curves.csv` | 0.00 | 10-23 22:11 | current | - |
| qc | `jaw_onsets.csv` | 0.01 | 10-23 19:54 | current | - |
| qc | `lick_contact_by_epoch.csv` | 0.00 | 10-23 08:32 | imaging / behaviour only or small; not affected | - |
| qc | `position_block_durations.png` | 0.00 | 10-19 06:13 | imaging / behaviour only or small; not affected | - |
| qc | `position_block_summary.csv` | 0.00 | 10-19 06:13 | imaging / behaviour only or small; not affected | - |
| qc | `position_blocks.csv` | 0.00 | 10-19 06:13 | imaging / behaviour only or small; not affected | - |
| qc | `precue_contacts.csv` | 0.00 | 10-20 21:55 | imaging / behaviour only or small; not affected | - |
| qc | `running_clips` | 0.06 | 10-23 13:24 | imaging / behaviour only or small; not affected | - |
| qc | `slowband_caveat.png` | 0.00 | 10-20 11:22 | imaging / behaviour only or small; not affected | - |
| qc | `slowband_decode.csv` | 0.00 | 10-19 06:42 | imaging / behaviour only or small; not affected | - |
| qc | `slowband_decode_all.csv` | 0.00 | 10-20 11:54 | imaging / behaviour only or small; not affected | - |
| qc | `svg` | 0.00 | 10-19 08:06 | imaging / behaviour only or small; not affected | - |
| qc | `within_spout_deviation_polar_lick` | 0.00 |  | RUNNING: within-spout 3-D, 4 animals | - |
| qc | `within_spout_deviation_polar_lick_contact` | 0.00 |  | RUNNING: within-spout 3-D, 4 animals | - |
| qc | `within_spout_deviation_polar_lick_contact_hits` | 0.00 |  | RUNNING: within-spout 3-D, 4 animals | - |
| run | `movement_first_subset-nokin_exec-polar_licks-combined_bins_3d4` | 3.18 | 10-24 03:35 | 4-animal polar + binned (PRIMARY) contest, complete cam1 + 3-D, reference gate | - |
| run | `movement_first_subset-nokin_exec-xyz2_licks-combined_bins_3d4` | 0.18 | 10-24 06:15 | RUNNING: xyz2 4 animals | - |
| run | `stray_licks` | 0.01 | 10-23 21:23 | stray test, kinematics_all 3-D, complete (not gated) | - |

## Phase-A re-run list (what replaces the REDO rows)

0. `scripts/trial_attempts.py` (running 10-24) -> per-animal gate / class flips vs the 10-21 table.
1. Frozen Layer 1 final frame `_v7g3_t3d` (grouping v3 + 3-D tongue + new gate) -> basis maps, Layer 2, relabel in that frame.
2. LMMs (`imaging_lmm.py`, `encoding_mixed.py`, push) on the new attempts / deficit.
3. O2 bundle (combined licks + 3-D tongue + attempted gate): main, droppos, shufpos, nodir, nokin, nokin_matched, win_nokin,
   win_drop, trmodels, musall_shuf; contests incl. binned width sweep 0.5 / 2 (`BINS_WIDTH`).
4. 4-animal: destination vs landing (2.4c), lick templates, LP-lick maps / counts, position-coding recovery trend.
5. Phase B: FR rank-12 encoder runs (after proj + labels.json + grouping from the FR window).

