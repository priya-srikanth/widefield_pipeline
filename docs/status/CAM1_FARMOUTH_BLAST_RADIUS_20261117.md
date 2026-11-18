# cam1 far-from-mouth events: blast radius outside RESTnlc1 (11-17, audit only)

**Question.** Priya's blind review (`CAM1_EXTRA_LICKS_CHECK_20261117.md`, last section) found that cam1-only events with
`med_out_px < -120` are static tracking labels, not tongue. She approved dropping them. The guard exists only in
`wfield_local/restnl_licks.py` (RESTnlc1 v3, being re-run now). Which other figures count these events as licks?

**Scope.** Curated cohort1, PS92-95, 120 sessions (`analysis_kit.curated_labels`). Epochs come from `epochs.epoch_of`; "pre" = August pre and
"pre_june" = the June pre sessions, reported separately. PS93_0813 has no video. Nothing was re-run, no pipeline code was edited, and nothing
was written outside the scratchpad except this file.

**Method.** Scratch scripts `blast.py` / `quiet_time.py` (scratchpad `blast_c1far/`):
- Each session's `events.csv` was copied, with the far cam1-only rows (`~cam4_matched & ~droplet & med_out_px < -120`) set to
  `droplet = True`. Every consumer drops droplet rows.
- `trial_classes._paths` was pointed at that copy (monkeypatch in-process). Then `session_trials`, `cue_classes_for` (the
  session-level working period, config defaults: end = last camera lick, stopped = both, excise) and
  `first_lick_latency.session_latency` + `lp_licks.first_lick_timing` were compared with and without the guard.
- `quiet_time.py`: seconds of whole-session time that ONLY far events mark as busy under the quiet-event rule.

## 1. Consumers

| consumer | file read | med_out guard | feeds |
|---|---|---|---|
| `wfield_local/restnl_licks.py` (`extend`, `cam1_only_onsets`) | events_all | **yes** (v3) | RESTnlc1 mask: `restnl_mask`, `rest_frozen_decoder_nl`, `rest15f_restnl_1117` c1all, `restw_strict_check` "strict" under cam4cam1, `restnl_cam1_compare`. Re-run in progress; not covered here |
| `wfield_local/trial_classes.py` `session_trials` | events.csv (post-cue; `~cam4_matched & ~in_3d & ~droplet`, plus the `jaw_only_trial` "seen" rows) | no | THE trial class (incomplete), `first_lick_ms`, `n_licks`, `n_nocontact`; through `cue_classes_for` the camera working period, stopped runs, `after_end`. Read by `trial_attempts.py`, `jaw_only_incomplete_figs.py`, `cam1_incomplete_licks.py`, gate_source / `attempt_coding` / kinematics gate (`working_period_trials`, `keep_working_period`) |
| `scripts/first_lick_latency.py` `session_latency` | events.csv (`~droplet`, on_ms >= 0) | no | `combined` first lick -> `trial_attempts.csv` -> `lp_licks.first_lick_timing` = the first LP lick of lick-aligned windows (LP_licks family, attempt_coding, encoder lick alignment) |
| `wfield_local/lp_licks.py` `_read_times` | events.csv (`~cam4_matched & ~droplet`) | no | LP lick train (`WIDEFIELD_LICK_SOURCE=lp`: `_trial_features`, `nolick_decoder`, ENL lick-free window, `plot_lick_aligned_averages`). Post-cue only; `edit_train` rewrites cue..cue+max_rt anyway |
| `wfield_local/combined_encoding.py` `_cam1_lick_events` | events.csv (`~cam4_matched & ~droplet`) | no | lick-timing kernel under `COMBINED_ENCODING_LICKS=combined` (frozen encoder, encoding page; `enl_licks` reuses it). Its pre-cue / ITI LP kernels come from cam4 `licks_all` only |
| `frozen_encoder.py`, `scripts/iti_licks.py` | via `combined_encoding.load_behaviour` | no | as above |
| **`scripts/cohort2_qc/lp_licks.py` `lick_times`** (despite the folder name, the COHORT1 quiet-event chain) | **events_all, whole session** (`~cam4_matched & ~droplet`) | **no** | `QS_LICK_SOURCE=daq+lp` busy frames (+-1.5 s) of `quiet_spikes_v2` and everything that imports it: quiet_eta_fr, quiet_fluct, quiet_lick_lfo, quiet_state_response, quiet_task_events(_followup), quiet_sync_4to8, quiet_event_movie, quiet_stillness; `gain_bio` (deck 112) uses the same exclusion |
| `scripts/revup_behaviour.py` | events_all (`~droplet`) | no | `n_cam1_pre` / "aborted" class (cam1-only, cue..first lick) and `move_ms` / wait-revup split (first cam1 event of any kind) |
| `scripts/restw_strict_check.py` `strict_c1` arm | events_all, every row (no cam1-only, droplet or med_out filter) | no | a diagnostic arm of the 10-29 check only (its `strict` arm uses `restnl_licks`, guarded) |
| `scripts/cam1_incomplete_licks.py` | events.csv cam1-only + `trial_classes` | no | per-spout cam1-only lick rates / incomplete share (kinematics page section 6 support, `fig_farcontra_attempts_sessions` "licks without contact") |
| `cam1_jaw_review.py`, `attempt_review.py`, `pose3d_*_sheet.py`, `pose_chain.py`, `scripts/cam1_licks.py` | review sheets / the producer | n/a | no figure |
| `pose3d_side.py` | none (its own side-camera droplet rule) | n/a | not a cam1-event consumer |
| `cohort2_qc/quiet_lp_sensitivity.py` and the cohort2 quiet runs | events_all | no | cohort2: out of scope here, but they use the same unguarded `lick_times` |

The 3-D kinematics (cam4 + cam1 triangulation) do not use cam1 2-D events; they reach them only through the camera working-period gate.

## 2. Events (cam1-only, non-droplet, med_out_px < -120)

| animal | epoch | sessions | post-cue (events.csv) far / all cam1-only | trials with a far post-cue event | events_all far: pre_cue / post_cue / iti | events_all far / all cam1-only |
|---|---|---:|---:|---:|---:|---:|
| PS92 | pre_june | 3 | 7 / 824 | 2 | 10 / 7 / 12 | 29 / 1562 |
| PS92 | pre | 8 | 9 / 3325 | 7 | 22 / 9 / 22 | 53 / 7807 |
| PS92 | acute | 5 | 3 / 1014 | 1 | 4 / 3 / 9 | 16 / 2369 |
| PS92 | subacute | 2 | 0 / 501 | 0 | 0 / 0 / 12 | 12 / 1243 |
| PS92 | chronic | 12 | 34 / 5247 | 23 | 475 / 33 / 342 | **850** / 11464 |
| PS93 | pre_june | 3 | 0 / 853 | 0 | 4 / 0 / 36 | 40 / 2154 |
| PS93 | pre | 8 | 0 / 3209 | 0 | 8 / 0 / 110 | 118 / 7002 |
| PS93 | acute | 4 | 0 / 1140 | 0 | 2 / 0 / 47 | 49 / 2683 |
| PS93 | subacute | 3 | 1 / 755 | 1 | 5 / 1 / 15 | 21 / 1917 |
| PS93 | chronic | 12 | 4 / 5207 | 4 | 11 / 4 / 228 | 243 / 11669 |
| PS94 | pre_june | 3 | 0 / 756 | 0 | 1 / 0 / 8 | 9 / 1955 |
| PS94 | pre | 8 | 1 / 1886 | 1 | 19 / 1 / 42 | 62 / 5104 |
| PS94 | acute | 6 | 30 / 1488 | 13 | 25 / 30 / 159 | 214 / 3643 |
| PS94 | subacute | 5 | 15 / 1659 | 11 | 52 / 15 / 210 | 277 / 4670 |
| PS94 | chronic | 8 | 10 / 2636 | 3 | 9 / 10 / 57 | 76 / 6858 |
| PS95 | pre_june | 3 | 0 / 722 | 0 | 1 / 0 / 49 | 50 / 1120 |
| PS95 | pre | 8 | 20 / 3038 | 10 | 43 / 20 / 152 | 215 / 4532 |
| PS95 | acute | 1 | 0 / 221 | 0 | 1 / 0 / 17 | 18 / 416 |
| PS95 | subacute | 7 | 10 / 2067 | 5 | 21 / 10 / 289 | 320 / 4295 |
| PS95 | chronic | 11 | 28 / 6335 | 17 | 112 / 28 / 317 | 457 / 22374 |
| **total** | | 120 | **172 / 42,883 (0.40 %)** | 98 | 825 / 171 / 2133 | **3129 / 104,837 (3.0 %)** |

- Post-cue far events are rare (0.4 % of post-cue cam1-only events). Most far events (95 %) are pre-cue or ITI, which only the
  events_all readers see.
- PS92 chronic is dominated by PS92_0929 (324 far events, label outside the crop).
- 169 / 172 post-cue far events are also not `in_3d`, so the `trial_classes` set sees nearly all of them.

## 3. Trial classes, working period, first LP lick (with vs without the guard)

| animal | epoch | trials | raw class changed | working trials (contact + incomplete in period) | working lost / gained | % working changed | stopped / stop-kind changes | first LP lick changed (attempted trials; -> none) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| PS92 | pre_june | 1124 | 0 | 1074 | 0 / 0 | 0 | 0 / 0 | 0 |
| PS92 | pre | 2920 | 0 | 2863 | 0 / 0 | 0 | 0 / 0 | 2 (0) |
| PS92 | acute | 2681 | 0 | 1798 | 0 / 0 | 0 | 0 / 0 | 0 |
| PS92 | subacute | 593 | 0 | 561 | 0 / 0 | 0 | 0 / 0 | 0 |
| PS92 | chronic | 3894 | 0 | 3890 | 0 / 0 | 0 | 0 / 0 | 6 (0) |
| PS93 | all epochs | 15,374 | 0 | 14,019 | 0 / 0 | 0 | 0 / 0 | 0 |
| PS94 | pre_june / pre | 6197 | 0 | 5769 | 0 / 0 | 0 | 0 / 0 | 0 |
| PS94 | acute | 3643 | 8 | 2205 | 8 / 0 | 0.36 | 5 / 8 | 10 (8) |
| PS94 | subacute | 3000 | 2 | 2515 | 2 / 0 | 0.08 | 3 / 5 | 4 (2) |
| PS94 | chronic | 4800 | 1 | 4502 | 1 / 0 | 0.02 | 0 / 42 | 2 (1) |
| PS95 | pre_june | 1394 | 0 | 1298 | 0 / 0 | 0 | 0 / 0 | 0 |
| PS95 | pre | 5556 | 0 | 4944 | 0 / 0 | 0 | 0 / 0 | 1 (0) |
| PS95 | acute | 720 | 0 | 543 | 0 / 0 | 0 | 0 / 0 | 0 |
| PS95 | subacute | 4275 | 0 | 2844 | 0 / 0 | 0 | 0 / 0 | 3 (0) |
| PS95 | chronic | 6600 | 4 | 5824 | 4 / 0 | 0.07 | 1 / 2 | 7 (4) |
| **total** | | **62,771** | **15** | **54,649** | **15 / 0** | **0.03** | 9 / 57 | **35 (15)** |

- **Raw class changes:** all 15 are incomplete -> jaw_only (11) or -> no_response (4).
- **Session-level classes:** 24 changes. The 15 working trials leave the working set; 9 non-working trials become `stopped`
  (no_response > stopped 8, jaw_only > stopped 1), because a removed "incomplete" no longer breaks a stop run.
- **Working-period end ("last camera lick"):** it moves in 0 of 120 sessions, and no `after_end` count changes.
- **Stop kind:** the other "stop-kind" changes are run / tail kind labels with the class unchanged (42 of them in one PS94 chronic
  session).
- **No cell above 1 %** at animal x epoch. The largest is PS94 acute, 0.36 %. The largest single session is PS94_0823: 6 / 408 working
  trials (1.5 %), 5 of them at close_R (5 / 81 trials at that spout, that session). Other affected sessions: PS94_0817 (far_R,
  far_L: 1 each), PS94_0831 (far_L 2), PS94_0910 (far_L 1), PS95_0924 (close_R 4).
- **First LP lick:** the 20 timing changes on trials that stay attempted move later by a median of 146 ms. These are the trials where
  a far label had preceded the real first lick.

## 4. Quiet-event exclusion (events_all, +-1.5 s; DAQ contacts, cam4 licks_all, other cam1-only events and running also exclude)

Seconds that ONLY far events mark as busy, as a % of the not-busy time. This is an approximation: 10 ms grid, whole session, no
swapped-pair, engagement or quit cut.

| animal | pre_june | pre | acute | subacute | chronic |
|---|---:|---:|---:|---:|---:|
| PS92 | 0.16 % | 0.17 % | 0.06 % | **1.48 %** | **2.37 %** |
| PS93 | 0.25 % | 0.80 % | 0.28 % | 0.26 % | **2.87 %** |
| PS94 | 0.10 % | 0.19 % | 0.84 % | **1.98 %** | 0.36 % |
| PS95 | 0.46 % | 0.69 % | 0.33 % | **1.15 %** | **1.52 %** |

- **Total:** 1302 s of 147,750 s (0.88 %).
- **Worst sessions:** PS92_0929 gives back 51 % of its small quiet time (64 s of 125 s), then PS93_0929 10 %, PS93_1002 9 %,
  PS93_0812 (pre) 8 %, PS94_0903 7 %.
- **Direction:** the contamination is largest in chronic / subacute, not acute. The headline acute-vs-pre quiet-event rise is not
  driven by it. Chronic levels and the chronic-vs-pre deltas are the cells that shift.
- **Cross-check:** under the RESTnl buffer (-0.5 / +2.0 s) the same events remove 0.73 % of time overall. This is the change
  RESTnlc1 v3 is now re-running.

## 5. Verdict per family

**(c) Needs a re-run.** Ordered by how much moves:
1. **Quiet-event family (QS_LICK_SOURCE=daq+lp, `v2_quiet_lp`).** It reads all 3129 far events (2133 ITI, 825 pre-cue), which
   remove about 1300 s of quiet time. Five animal x epoch cells are above 1 %, and PS92_0929 is badly hit (51 %).
   - Fix: add the same guard to `scripts/cohort2_qc/lp_licks.lick_times`. Reuse `restnl_licks.cam1_only_onsets` /
     `FAR_FROM_MOUTH_PX` (rule 9: one definition). Then re-run `quiet_spikes_v2` extraction and its readers.
   - Encoding page section 15 figures: f-q_fr, q_eta_maps_fr, q_examples, q_selection, q_stillness, q_task_outcome,
     q_task_rates, q_quit_map, q_spectrum, q_sdmaps, q_hemo415, q_breathing, q_hemi_amp, q_recovery_link, q_threshold,
     q_steinmetz, q_width, q_decay_fr, q_effort, q_fr_detail, q_lick_lfo, q_sync / q_sync_4to8, q_t13_followup_lp.
   - Deck (provenance sweep numbering): 112 (gain_bio, brain-wide gain), 156 (stillness), 157 (pre-cue event).
   - Same fix needed for the cohort2 quiet runs (not quantified here).
   - Headline expected to hold (acute changes are small); chronic numbers shift by about 1.5-3 % of quiet time.

**(b) Negligible.** Size stated; re-render only if the figure is being redone anyway:
2. **Camera trial classes and the working period** (`trial_classes`, `trial_attempts.csv`, camera gate everywhere).
   - 15 / 62,771 trials change class (incomplete -> jaw_only / no_response). 15 / 54,649 working trials are lost (0.03 %).
     The maximum is PS94 acute, 0.36 %. No end cut moves.
   - Affected figures: kinematics page section 6 / deck 35-43 (trial-class shares, jaw / tongue traces by class), Fig 6.1 far-contra
     class shares, farcontra_attempts_sessions, attempt-rate / completion, `attempt_coding` hits-vs-failed (deck 115-118, 141-146),
     and the camera working-period gate of the kinematics figures (deck 46-62).
   - The far-contra (far_R) acute change is 1 trial (PS94_0817). Within-session cells are below 2 % except PS94_0823 close_R,
     5 / 81 trials.
3. **First LP lick / lick-aligned windows** (LP_licks family; deck 13-16, 25-26). 35 trials' first LP lick changes, 15 of
   them because the trial stops being attempted. The LP lick train itself gains or loses 172 post-cue events of about 43k cam1-only
   events (0.4 %), and in cue..max_rt they are overwritten by `edit_train`.
4. **Frozen / combined encoder lick kernel** (`COMBINED_ENCODING_LICKS=combined`; deck 100-103, 148; encoding page frozen_* /
   peth_* / kernel_maps). The kernel has 172 extra post-cue onsets (0.4 % of cam1-only onsets, far less of all lick onsets);
   pre-cue / ITI kernels use cam4 only.
5. **Rev-up** (`revup_behaviour.py`, far-contra rev-up figures). At most the 98 trials with a far post-cue event, plus ITI events on
   trials whose first lick falls after the spout left. Far-contra acute is affected through PS94 acute (30 post-cue far
   events in 13 trials). Small, but the "aborted" class counts single openings, so check the PS94 acute far-contra cell if the
   figure is redone.
6. **cam1_incomplete_licks** (cam1-only lick rates per spout): the same 172 post-cue events, 0.4 %.

**(a) No change.**
- 3-D kinematics themselves (no cam1 2-D events).
- DAQ-lick families.
- `pose3d_side`.
- Review sheets.
- `restw_strict_check` `strict_c1` (a superseded diagnostic arm; it counts every events_all row, droplets included).
- RESTnlc1 is already guarded and being re-run.

## 6. Recommendation

- **Fix once, at the readers.** Add `med_out_px < FAR_FROM_MOUTH_PX` to the cam1-only filter of `trial_classes.session_trials`
  (both the `E` filter and the `seen` set), `first_lick_latency.session_latency`, `lp_licks._read_times`,
  `combined_encoding._cam1_lick_events`, `revup_behaviour`, `cam1_incomplete_licks` and `cohort2_qc/lp_licks.lick_times`.
  Import the constant from `restnl_licks`.
- **Caches.** `lp_licks` and `trial_classes` cache keys include their source code, so they recompute. Bump `session_cache`
  where needed.
- **Re-run order.** Re-run the quiet family first. Fold the rest into the next scheduled re-render of each family: the moves are
  below 0.5 % everywhere at animal x epoch.
- Not done here: no pipeline edit (audit only).

Scratch outputs: `sessions.csv`, `by_animal_epoch.csv`, `quiet_time.csv`, `quiet_by_animal_epoch.csv` in the 11-17 audit
scratchpad (`blast_c1far/`).
