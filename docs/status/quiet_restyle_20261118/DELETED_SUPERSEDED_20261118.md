# Superseded quiet-event outputs deleted (a8, 2026-11-18)

Priya 11-18: "ok to delete superceded folders". All DERIVED outputs under
rebuild_20261103/labcams/<T>/cohort1_quiet_spikes/, made by a8's runs; superseded by v2_quiet_lp_c1guard_syncend_20261118.
Checked first: no deck slide note cites them (all 208 slides); the Movement page v324 cites none (3c, grep of the
published source); no code default points at them (quiet_restyle_figs.py defaults moved to the sync-end run). The
deck-swap scripts and docs that name them are historical records. Kept: the sync-end run, fr_regressed_own/ (shared FR
series), engaged_end_20261118.csv, and the pre-rebuild labcams/cohort1_quiet_spikes/v2_quiet + v2_quiet_lp (page cites them).

| deleted | size | what it was |
|---|---|---|
| v2_quiet_lp_c1guard_k375_20261117/ | 1.6G | cam1-guard run, base summary k 3.75 + follow-ons (deck 11-18 morning) |
| v2_quiet_lp_c1guard_k4_20261118/ | 76M | base summary k 4 on pre-fix data + restyled_20261118 (pre-fix) |
| v2_quiet_lp_c1guard_engend_20261118/ | 1.6G | last-cue engaged-end rule (superseded by last sync) |
| v2_quiet_lp_20261117/ | 1.5G | pre-guard rebuild run (11-17 deck pass) |
| v2_quiet_lp_c1guard_20261117/ | 72M | cam1-guard session extraction + k 4.5 summary |
| engaged_end_20261118.wrong_trialstart.csv | 12K | first task-end table (used trial_start; cut the last trial) |

## Later 11-18 (Priya: "can delete the 13 old ones" [12 on recount], "if superceded and never cited, can delete")
- presentations/: 12 deck backups deleted (~1.9 GB): Overview_2026-11-06 .bak_pre_a2_s156ylim, _a8_c1guard, _a8_licksrc,
  _a8_pagerename, _a8_panels, _a8_quietnote, _a8_s126nl, _a8_s27c_rename, _a8_s27rest15f, _a8_swap_20261117,
  .pptx.bak_pre_a8_quiet_syncend, and PI_overview_2026-11-02_bak_before_s95_1105.pptx. Kept: .bak_pre_rebuild_20261116,
  .pptx.bak_pre_a8_ps94fix_20261118, .pptx.bak_pre_a8_s158title_20261118.
- rebuild cohort1_quiet_spikes/v2_quiet_lp/ (71 MB, 11-10 early rebuild v2 render): superseded by the sync-end run; cited
  by no deck note and not by the Movement page v324 (3c: the page's v2_quiet_lp refs are the PRE-REBUILD labcams path).
- Pending (Priya "ok"): pre-rebuild labcams/cohort1_quiet_spikes/v2_quiet + v2_quiet_lp (~8.5 GB) once the page figures
  that still cite them are re-run on the sync-end run.

## 11-18 late night: pre-rebuild quiet runs (Priya "3. ok", once uncited)
- Movement page v326 cites only v2_quiet_lp/task_events_followup + task_events (Fig 23.23; 3c grep); Spout page
  (1795029845-db7c) cites no cohort1_quiet_spikes path (3c); deck (208 slides) and FR page cite none (a8).
- DELETED: labcams/cohort1_quiet_spikes/v2_quiet/ (4.4 GB, DAQ-lick pre-rebuild run) and everything in
  labcams/cohort1_quiet_spikes/v2_quiet_lp/ (4.1 GB, pre-rebuild LP run) EXCEPT task_events/ + task_events_followup/
  (kept, ~0.75 GB, until 23.23 moves to the sync-end run) and README.txt. Superseded by
  rebuild .../cohort1_quiet_spikes/v2_quiet_lp_c1guard_syncend_20261118 (every section-23 figure re-rendered there).
- Kept: labcams/cohort1_quiet_spikes/{fr_regressed_own, fr_regressed, attempt_effort, sessions, rest_spikes_*} (not in scope).
