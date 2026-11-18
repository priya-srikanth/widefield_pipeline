# Running list 2026-11-17 (overnight): RESTnl with cam1 licks (RESTnlc1)

Record: `docs/status/RESTNL_CAM1_RERUN_20261117.md`. Branch `analysis/restnl-cam1-1117` (worktree `wf_restnlc1`), not pushed.
Queue: scratch `rc1/queue_rc1.sh` (scratchpad `C:/Users/SABATI~1/AppData/Local/Temp/claude/179e1046909c5be2/f082ee0c-b33c-49a8-a537-dcee3bb7c84c/scratchpad`),
logs `rc1/logs/<step>.log`, chain `rc1/logs/CHAIN.log`, PIDs `rc1/logs/PIDS.txt` (stop only those). `touch rc1/STOP` halts
before the next step.

## Decisions needed (Priya)
- Delete list (own derived, v1 definition, never read): `rebuild_20261103/session_cache_restnlc1/PS92_0814__*.npz` and
  `iti_licks/<T>/restw_check_c1/sessions/PS92_0814.npz` (written 14:3x by the v1 timing run before the events_all
  correction). Nothing deleted.

## Running (started 14:38)
- lane AC (map family, then state space): a1 baselines DAQ gate -> a2 LP gate -> a3 lp_final maps -> a4 drift LP -> a5 attempt
  maps -> a6/a7 rest decoder 12.1b (aug, all) -> a8/a9 Fig 12.1 15f c1all (all, aug) -> c01..c15 state space.
- lane B (area + interhemi, frozen env): b01 restw_check_c1v2 -> b02..b12 goal-page area scripts -> b13..b19 interhemi.
- Workers: lane AC <= 4, lane B 3 (<= 8 total with the 1-worker timing lane).

## Done
- 14:50 code v1 (lp_lick_times cam1 = post-cue only) -> CORRECTED 14:35 to v2 (cam1 events_all; aac676a8).
- Timing (one session): restnl_mask baseline 26 s (PS92_0814); restw_strict_check 1 session ~1 min; interhemi rest extract 1 session ~1 min.
- First measurement, PS92_0814: 1,314 cam1-only (events_all, non-droplet) onsets, 0 of them inside the cam4 RESTnl mask; median
  0.11 s from the nearest cam4 lick -> 2 rest samples removed, frames per position unchanged.

## Failed
- t2_restw_check_one (both runs): one-session summarise KeyError 'strict' (summary needs >1 session); the session npz was written. Not a problem for b01.
