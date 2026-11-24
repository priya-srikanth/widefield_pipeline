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
- 11-23 22:20 step 2 DONE: `qc/jaw3dmouth_ccB1123/s1_trial_classes_cue_stop.csv.gz` + camera-tagged `qc/jaw3dmouth_ccB1123/sel-camera-...-stopA2jexrt3500c70_daqlk/trial_attempts{,_sessions,_gate}.csv` (gate = camera:sel-..., canonical = working period). Class change vs production = section 10.4 exactly (641 jaw-only -> no response, 713 no response -> jaw-only, 19 -> unknown, 2 unknown -> jaw-only). Attempts canonical: 918 trials leave, 814 join vs the reference gate. Compare: `overnight/classes_compare/`.
- 11-23 ~23:00 helpers DONE: S15 GLM-HMM refit (states/stars unchanged; l.178 59% -> 57%); S16 treadmill (no trial changes under its lick-span gate; DECISION: lick span vs working period, acute walking gains ** on the working period); S10A/E (wait carries 88% of the acute far-contra delay: 158 -> 403 ms, rev-up 32 -> 64; S10E only #12 raw rho -0.72 -> -0.76); S12 on working period (no star changes; S12F contrast 4/4 animals). Item 32 confirmed, not re-rendered.
- 11-23 ~23:00 Fig 3D DONE (3 variants, stats reproduce live exactly on the 11-22 table first). DECISION: live Fig 3D includes stopped trials (gate = to last lick only) although the caption says working period. New rule + working period: 12 distribution stars (live 13), far-mid chronic gains, near-mid subacute / far-ipsi subacute / near-contra... lose; new rule on the live gate: 13, same bars.
- 11-23 ~23:00 l.150 quit point: disengagement extract-beh on the new classes into `imaging_encoding/disengagement_jaw3dmouth_ccB1123/`; camera quit (fused licks) gives early-post KM -5.7/-3.1/+6.9/-2.5 min (was -15.1, 4/4, DAQ quit on v36): the claim does not hold as written; DAQ-quit row running.
- 11-23 ~23:45 step 3c DONE (l.150): DAQ quit on sync60 reproduces -15.2 min [-25.4, -5.6], -79 trials, 4/4; camera quit (fused) -1.1 min [-6.9, 6.3], +1 trial, but -87 responded trials 4/4. DECISION for Priya (text options in SECTION section 4).
- 11-23 ~23:50 step 4 DONE: 210-cell family with new hit-rate cells (11 session cells move <= 0.013) + S12 working-period cells: 118 stars, no flips. `manuscript_stats/c2_4cam_sync60_20261122/c2_scalar_family_FINAL_withS12_jaw3dmouth_wp_ccB1123_boot4000_cam3on.csv`.
- 11-23 ~23:55 step 5 DONE: `ccB_caption_review_2026-11-23/SECTION_overnight_rebuild.md` (decisions first), README addenda in `data_4cam_20261122/README_overnight_*_ccB1123.md`, renders in `ccB_caption_review_2026-11-23/overnight/`. Code: `ccb/overnight-rebuild-1123` d960e790 + helper branches (s15 6c63fb9d, s16 29ee6224, s10 215af1a7, s12 0828e118); none pushed. All CC-B jobs finished.
- 11-23 ~00:10 coordinator Q (why the jaw rule changes the working period): own effect = 11 of 57,601 trials (9 no-response -> unknown, 3-D jaw moving at the cue with no re-onset; 2 unknown -> jaw-only). No stop run or last lick moves. Gate switch (vs reference) reported separately: 918 out / 814 in. Added to SECTION section 1.
- 11-23 ~00:20 DECIDED (Priya via lead): Fig 3D = new rule + working period (render `overnight/FigureS11_trial_classes_c2_4cam_v28b_jaw3dmouth_new_wp_ccB1123.*`, caption in SECTION section 2); S16B = working period (acute walking **), S16 helper re-rendering the B traces. Recorded at the top of SECTION_overnight_rebuild.md.
- 11-23 ~00:40 S16 working-period render delivered: `overnight/S_treadmill_running_jaw3dmouth_wp_ccB1123.*`; README_overnight_S16 marks the working-period tables PRIMARY. CC-B overnight rebuild complete; hand-back sent.
- 11-23 ~00:50 S16 render detail: row B vs live figure gains acute walking ** and subacute push ** (the live figure lacked the subacute push star that the delivered table already had); S16 commit f2ee66da.

## CC-B inconsistency fixes (Priya 11-23 late: "fix all inconsistencies"; record `Widefield/run_records/2026-11-23/INCONSISTENCY_FIXES.md`)

- 11-23 23:25 task A (imaging root classes) IN PLACE. Findings: the pinned render code (e1a3c85d) has no `3d_mouth` mode, so
  every step computes classes live on the OLD jaw rule whatever table it reads; the imaging root had no in-root s1 snapshot
  (`working_trials.snapshot_path` is not rebuild-routed, and the rebuild fence refuses the production one). Fix: worktree
  `wf_fourcam_j3m`, branch `ccb/fourcam-j3m-render-1123` (merge e1a3c85d + ccb/jaw3d-mouthlick-1123, profile + jaw env, 57aa4800);
  in-root `qc/jaw3dmouth_ccB1123/` (cap jaw table + new-rule s1 copies); queue lanes J1-J4 (steps.py `*_j3m`, 22 steps incl.
  `s_classes_j3m` = in-root trial_attempts rebuild, old copied to `*_jawprod`). 18 pending class-dependent steps superseded
  (placeholder .done text on old names); lane H ended via a temporary skipped marker. Rule effect vs old table: attempted 0,
  working period 11 trials, run class 385, raw class 1,375 (jaw-only <-> no-response). DONE steps with old-rule jaw-only
  panels (not requeued, lead to decide): f4_restnl_maps_contact (15r supplement, 7.6 h); running f1_epoch (12s/12b stopped,
  15r supplement) and f4_restnl_maps_lp. s_classes_j3m waits on MAXRUN 3 (3 steps running).
- 11-24 00:30 A: `s_classes_j3m` done (57 min): in-root trial_attempts on the new rule match the manuscript table exactly (0 / 57,048
  differ; vs old: run class 385, raw class 1,375, canonical 11, attempted 0); old copies `*_jawprod.csv`. J1 running (nolick on
  j3m code). Lane J5 (lead 23:40: family-only re-draws, j3m eed09e0f `15rs` + `restnl_maps --supplement-only`) waiting on f1_epoch /
  f4_restnl_maps_lp. B: rev-up tables on the new rule + working period `imaging_encoding/revup_fourcam_jaw3dmouth_wp_ccB1123/`
  (branch ccb/revup-wp-1123; reproduces S10A overnight cells); rev-up neural chain running (launch_nt `revup_j3m`, pid 43912:
  precue_restw fixed -> revup_neural (+ restnl) -> revup_followup). Inventory of pages / decks / O2 in INCONSISTENCY_FIXES.md.
- 11-24 ~04:30 CC-B fixes: task C on `ccb/defaults-1123` 829c32bd (not merged). Pages behavior / states / kinematics /
  facerhythm delivered to `run_records/2026-11-23/page_updates/` (not published); PI deck `Overview_2026-11-24_fourcam.pptx`
  + CHANGELOG; O2 G4 bundle `enc_fourcam_20261124` ready except the smoke test (running); movement page mapped (assembly
  open); Fig 23.23 compare lane J6; rev-up neural finishing (`revup_j3m2`). Detail: INCONSISTENCY_FIXES.md progress log.
- 11-24 ~05:30 CC-B fixes: task B done (rev-up neural + followup on the working-period tables; RESTnl twin and the 3-D
  rewrite of followup part B left). Lane F: G4 ready except the smoke test (needs >= 40 GB free), frozen_encoder O2 port
  scoped. Lane E (movement 18.5 / 2.2 / 2.6 + 15.x basis-only build) still running. INCONSISTENCY_FIXES.md progress log.
