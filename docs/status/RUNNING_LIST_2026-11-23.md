# Running list 2026-11-23 (four-camera re-analysis; shared, append-only, one section per agent)

## Imaging rebuild (four-camera sync60 root)

**Decisions needed:** none yet.

- 05:05 Root + profile: `M:/MICROSCOPE/Priya/Widefield/rebuild_20261123_fourcam`, profile `cohort1_rebuild_20261123_fourcam`
  (= the v36 rebuild's switches + `WIDEFIELD_FUSED_LICKS=..._sync60_gapfix60`, `WIDEFIELD_RESTNL_LICKS=fused`; own session
  cache / frozen store under the root). Code: worktree `Github/wf_fourcam_img`, branch `imaging/fourcam-rebuild-20261123`
  (off pilot 4097561e; 548b5f2c = profile + `{run}` seeds). Run tag is the SAME string as the v36 rebuild (the fused train
  enters cache keys via `session_licks.signature`, not the tag); the two roots never share a store.
- 05:09 Seeded (sha256): epoch_boundaries.json, sync60 `qc/jaw_onsets.csv` (cam3-on, 04:51). nose_onsets / precue_contacts /
  first_lick_latency_late / position_block_summary do not exist under sync60 (page-only readers; recorded missing).
- Resolver printout (`lp_stage/imaging_fourcam_20261123/tools/probe_paths.py`): RUN sync60, fused train, kin =
  kinematics_fused_postcue, RESTnl set fused -> `session_cache_restnlfv1`, GLM-HMM v1_floor020, classes live from the seeded
  jaw_onsets + sync60 sessions + cam1_licks_v37_sync60, attempts -> `<root>/LP_analysis/<sync60>/imaging_encoding/qc/<sel tag>/`.
- 05:12 Smoke: restnl_mask 1 session 1.8 min (cold cache). 05:13 started D5 RESTnl (DAQ gate, --jobs 4) and in-root
  trial_attempts (--jobs 3) -> camera_coverage -> RESTnl LP gate. Share link saturated (~120 MB/s receive): runtimes are I/O bound.
- Queue: `lp_stage/imaging_fourcam_20261123/{queue.py,steps.py,launch_lane.py}` (CREATE_NO_WINDOW, .done markers, RAM gate).
