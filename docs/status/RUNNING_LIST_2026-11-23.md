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

## Behavior + encoding (BE agent)

**Decisions needed (Priya):**
1. **AR-HMM: refit or keep the re-inference?** Re-inferred with the saved v36 fits on VERSION 4 features: 15.5 % of 40-ms
   frames change state vs the v36 inference (median session 16.5 %, range 1.8-23.6 %), 8.5 % at the reviewed syllable-group
   level, even across epochs; group usage moves <= 0.6 points. J3 / E5 currently use the re-inferred states.
2. **GLM-HMM K-selection figure** (states page `model` family): `glm_hmm/v1_floor020` has only K = 5, no
   `select_k_pooled_all_v2*.csv`. Run a select-K pass, or keep the v1 figure?
3. **O2 jobs (G4 combined_encoding production; page-lineage spont_frA frame):** commands + smoke results below once ready.
   frozen_encoder has NO O2 port; the page-lineage frame is proposed as a local run in the four-camera root (asked 14:25).

**Status / log:**
- 05:00-06:53 class rebuild (launch.py): s1 (05:01-05:17) and trial_attempts (05:00-06:53) exited cleanly. vs
  `qc/*_precam3on.*`: 12 trials change `cls` / `cls_detail` / `working_raw`, all PS95_0820 (8 no_response -> jaw_only,
  4 jaw_only -> no_response); `cls_final` unchanged; no other session differs.
- Manuscript-lineage frozen encoder: NOT refit (coordinator / Priya 11-23: keep the 11-22 fourcam frame). Only a routing
  check ran (05:05); the empty folder `E/manuscript_cam3on_20261123/` is a delete candidate.
- Behavior block (05:15-14:18, sub-agent; code 20bc145f): C2 cohort tables (in place; old tree kept as
  `<sync60>/cohort_pre_cam3on_20261123/`): stars 103 -> 102 (resp_rate far_ipsi subacute lost). A1 within_session_decline
  (`Widefield/within_session_decline_fourcam_20261123`, 19 -> 18 CI-excluding-0 contrasts, 5 flips), jaw_only_incomplete
  (`session_poses/cam1_licks_v37_sync60/trial_classes/`), class share figs; disengagement analyse / stopped_trial_traces
  wait on imaging. D7 lp_trial_counts, misdirected counts, lick_provenance wording (opt-in). C3 nose_onsets,
  contact_direction_predictability, first_lick_latency, pose3d_spout_error + summary fig, lmm_curve_summaries
  (`labcams/grant_figures/lmm_hidden_fourcam_20261123`). J5 TIM (1.4 % of trials change class), J4 AR-HMM features v4 +
  re-infer (decision 1), J3 collect + states page figs (delta-vs-pre CI stars 98 -> 101), E5 FR cue-aligned 32 -> 30 stars,
  E3 FR validate / lick_epochs / readouts (`master_pre_v1_fourcam_20261123`; 16 -> 16 stars), E4 fr_enl_decode (no change).
- 14:26 G6 / G7 on the 11-22 frame: `lp_stage/reanalysis_20261123/g67_chain.py` (pid 26092) -> lick_correction extract +
  behavior + neural in `E/iti_licks_lickcorr_fourcam_20261123`, then WLC_LAUNCH=2v run + summarise in
  `E/within_lick_correction_fourcam_launch2v_neural_20261123`.
- 14:25 G4 bundle `rebuild_20261123_fourcam/o2/enc_fourcam_20261123`: one-session build (PS93_0814) first, for timing.
- 14:26-18:18 G6 / G7 done: see DECISIONS 11-23 "behavior + encoding arm". The behavior halves are byte-identical to the
  11-23 cam3-on runs; the neural tables are new.
- ~14:30-19:45 neural-tail helper (G8 / G9 / G10) was cut off by a rate limit without a report. Triage + star moves:
  `lp_stage/reanalysis_20261123/neuraltail_triage.md`; code committed 35678692 (opt-in OUT_X redirect). Its chain
  drivers 18552 (g8), 32700 (g10), 8560 (g9) keep starting steps. Stopping them was DENIED by the permission system;
  waiting on Priya (coordinator raised it). The running children are light (< 1.5 GB each).
- 19:50 Priya: manuscript first. PARKED (results pages): spont_frA local refit, GLM-HMM select-K, C4, G3, G5, the rest
  of G8-G10, E6, B2 / B3, K1a. SKIP: TIM engagement inputs (TIM left the paper in v26); FR figures (moved to four-camera
  by another helper). The O2 G4 bundle is HELD: one session built (`rebuild_20261123_fourcam/o2/enc_fourcam_20261123`);
  the full input build, env diff and smoke test are not done.
- 19:54-20:00 S16 treadmill rebuilt (DECISIONS). Delivered to `data_4cam_20261122/treadmill/*_fourcam_20261123.csv`.

## CC-B overnight rebuild (manuscript, six-position jaw rule + working period)

**Decisions (Priya 11-23).** Jaw call = ONE rule at all six spouts: four-camera 3-D jaw (cap + 3-D re-onset) OR an
in-mouth lick (`trial_classes.jaw_call(mode="3d_mouth")`); grooming false calls = noted limitation (option a). Every
six-position trial gate = camera working period (camera-tagged `trial_attempts`, `canonical` = working period), except
S4C epochs. Decisions still needed from Priya are listed at the top of
`ccB_caption_review_2026-11-23/SECTION_overnight_rebuild.md` as they come up.

Worktree `wf_ccb_overnight1123`, branch `ccb/overnight-rebuild-1123` (from 2d1d4db3). Logs:
`ccB_caption_review_2026-11-23/overnight/logs/`.

- 11-23 ~21:30 started step 1: production cap jaw table `qc/jaw_onsets_cap_ccB1123.csv` (full run, 126 sessions, 6 jobs, ~1 min / session single).
- 11-23 21:40 step 1 DONE: `qc/jaw_onsets_cap_ccB1123.csv` (126 sessions, 65,388 trials, has `jaw3d_reonset_ms`). Checks: every jaw column equals the lean review copy `jaw3d_qc/jaw_onsets_cap.csv` on all its trials; non-onset columns equal production `jaw_onsets.csv`. Step 2 launched (s1 snapshot -> `qc/jaw3dmouth_ccB1123/`, camera-gated attempts -> `qc/jaw3dmouth_ccB1123/sel-camera-...-stopA2jexrt3500c70_daqlk/`). Note: live Fig 3D stats reproduce exactly from `data_4cam_20261122/classes` (11-22 table).
