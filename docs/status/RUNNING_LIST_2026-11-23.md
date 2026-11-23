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

### Imaging rebuild, 06:20
- Render code pinned: DETACHED worktree `Github/wf_fourcam_render` at e1a3c85d (branch `imaging/fourcam-rebuild-20261123`:
  rest15f / rest_frozen_decoder_nl `--lp fused`; run_quiet_chain env from `QS_PROFILE`; state space `WF_SS_PROFILE` /
  `WF_SS_RUN`; state_space_rest_decomp lick class follows the fused train).
- nose_onsets (sync60, 05:41) seeded into the root (sha256 in REBUILD_ROOT.json); not recomputed.
- Session-cache reuse (Priya 11-23 "reuse caches where valid"): the v36 rebuild's DAQ-arm trial-feature entries (tf-*, 2,678
  files, 3.7 GB) are COPIED into the new cache, renamed to the new class signature (csig 3d7bbe921e -> 241a3ea7f7). Valid:
  `_trial_features` reads DAQ cues / DAQ licks + imaging only (LP licks only under the LP source, whose `-lp...` kinds are
  not copied), no gate or class; the csig is in the kind only to guard the prune; the file signature (LocaNMF file
  mtime + size, params, CACHE_VERSION) is kept. Script `lp_stage/imaging_fourcam_20261123/tools/seed_tf_cache.py`, manifest
  in its logs. Every other kind (gated / class / LP) is rebuilt.
- In-root trial_attempts is slow (~3 h at 3 workers, link saturated). DAQ-arm steps wait only on RESTnl (DAQ gate) +
  camera_coverage; LP / attempt / quiet steps wait on the in-root trial_attempts (independent of the default-run pid 26880).
- Queue throttle: at most 3 steps of all lanes at once (`MAXRUN`; coordinator: keep the share load modest).
- Row I: ITI per-trial extracts (rebuild lineage) queued under the profile (labels = the 107 frame sessions); aim /
  first-lick / recprim tables PARKED until the O2 spont_frA refit is collected into this root (coordinator 06:1x).
