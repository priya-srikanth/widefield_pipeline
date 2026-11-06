# Status and handoff docs — what each one settled

**For the current handoff of each arm, go to [`../START_HERE.md`](../START_HERE.md) section 1.** This directory
holds 81 markdown status docs (2026-11-04) plus their CSV / TXT tables. The full list, newest first, is at the end
of this page ("All status docs"); it was generated from the file names and first lines.

The table just below covers only the first seventeen snapshots (08-09 to 09-21), written at the end of a working
day or before a context compaction. **They are the project's memory, and none of them has been deleted.** Several record
a number that was later withdrawn, or a plan that turned out wrong; those are kept unedited,
because knowing what was believed that week is how a stale figure in a document gets traced back
to the run that produced it.

Consolidated into this directory on 2026-09-21 — `docs/` was sixteen status files deep and the
topic docs were hard to see between them.

**Reading rule: newest first, and each one says what it supersedes.** The chain is real — most of
these open by naming the file they replace and what that file remains the record of. Start at the
bottom of this table and walk up only as far as you need.

| date | lines | what it settled | superseded by |
|---|---:|---|---|
| [08-14](STATUS_2026-08-14.md) | 219 | the drift-removal decision week. **Carries two withdrawn claims**, marked in the file: the +0.032 pre-cue sub-binning gain did not replicate (+0.009, 23/44) | 08-17 |
| [08-09 tasks](STATUS_2026-08-09_TASKS.md) | 88 | **the open-items list, retired 2026-09-22** — it had stopped being open. Kept as the record of what was outstanding in August; its header says where each live fact went | — |
| [08-17](STATUS_2026-08-17.md) | 299 | **the seventeen silent bugs found 15–17 Aug** and the guards put in for each. Those guards are still in force | 08-23 |
| [08-23](STATUS_2026-08-23.md) | 140 | the post-stroke readout's design and the nulls behind it | 08-26 |
| [08-26](STATUS_2026-08-26.md) | 198 | cohort state after the lesion series | 08-28 |
| [08-28 handoff](STATUS_2026-08-28_handoff.md) | 218 | five approved items, all since done — kept as the record of **what the plan got wrong** | — |
| [08-28 evening](STATUS_2026-08-28_evening.md) | 94 | what was running that night | — |
| [08-28 night](STATUS_2026-08-28_night.md) | 153 | the pooled epoch figures, the engagement gate, and the parallel render | — |
| [09-07](STATUS_2026-09-07.md) | 226 | both engagement gates **backdated**, and the cue-aligned clip pipeline built on the trial classes they define | — |
| [09-10](STATUS_2026-09-10.md) | 139 | state before a compaction; the science is in [`PRELIM_DATA_VLS_STROKE.md`](../PRELIM_DATA_VLS_STROKE.md) | — |
| [09-12](STATUS_2026-09-12.md) | 605 | the anatomical arm | — |
| [09-13](STATUS_2026-09-13.md) | 2,618 | **the REST baseline migration** — the largest of these, on branch `worktree-rest-baseline` | partly 09-16 |
| [09-16](STATUS_2026-09-16.md) | 522 | the engagement-gate audit and the `restdock05` migration. **Cited most often**; `CLAUDE.md` points here | 09-17, for everything after the rest-arm audit |
| [09-17](STATUS_2026-09-17.md) | 1,145 | the rest-arm audit and what followed it | — |
| [09-19](STATUS_2026-09-19.md) | 254 | **the 415 nm day** and the engagement decomposition | 09-20 |
| [09-20](STATUS_2026-09-20.md) | 127 | haemodynamics closed, engagement standing, loops parallelised | — |
| [09-21 engineering](STATUS_2026-09-21_ENGINEERING.md) | 258 | **the codebase-quality plan** — module splits, output-tree restructure, the archive pass. Sections 2.1–2.5 | — |

---

## Which document answers which question

These four kinds of document are easy to confuse, and looking in the wrong one is how a
settled question gets re-litigated:

| you want | read |
|---|---|
| **why** something is the way it is | [`DECISIONS.md`](../../DECISIONS.md) — the reasoning, usually because the alternative was tried and failed |
| **what the code is** and how the parts relate | [`docs/ARCHITECTURE.md`](../ARCHITECTURE.md) |
| **what to run**, tonight or for the first time | [`runbooks/`](../../runbooks) |
| **what was true on a given day** | here |

A status doc is a snapshot and goes out of date by design. `DECISIONS.md` is cumulative and does
not. **If the two disagree, `DECISIONS.md` wins** — and that disagreement is worth a line in
`DECISIONS.md` saying so.

---

## All status docs (generated 2026-11-04, newest first)

One line per markdown file in this directory: the date in its name and its first line. Regenerate rather than hand-edit; files with no date in the name come first.

| date | file | first line |
|---|---|---|
| undated | [OPEN_QUESTIONS_PRIYA.md](OPEN_QUESTIONS_PRIYA.md) | Waiting on Priya — running list |
| 11-04 | [SPOUT_POSITION_AUDIT_2026-11-04.md](SPOUT_POSITION_AUDIT_2026-11-04.md) | Spout position: ENL no-lick medians in every kinematics analysis? Audit + impact (2026-11-04, window e0) |
| 11-04 | [SD_RISE_2026-11-04.md](SD_RISE_2026-11-04.md) | Post-stroke SD rise of the decoder features: gain or added noise? (2026-11-04) |
| 11-04 | [REST_CENTRE_2026-11-04.md](REST_CENTRE_2026-11-04.md) | Should the frozen spout decoder's per-session frame come from REST? (2026-11-04) |
| 11-04 | [REFSPOUT_CENTRE_2026-11-04.md](REFSPOUT_CENTRE_2026-11-04.md) | Can the reference spouts anchor the frozen decoder's day offset? And what does an offset-free readout say? (2026-11-04) |
| 11-04 | [PAGE_EDITS_PENDING_2026-11-04.md](PAGE_EDITS_PENDING_2026-11-04.md) | Pending results-page edits from the 11-03 / 11-04 work |
| 11-04 | [INJECTED_EFFECT_2026-11-04.md](INJECTED_EFFECT_2026-11-04.md) | Which per-session frame recovers a KNOWN change? Injected-effect validation of the frozen spout decoder (2026-11-04) |
| 11-04 | [GAIN_BIO_2026-11-04.md](GAIN_BIO_2026-11-04.md) | Is the post-stroke SD rise a biological gain or a measurement change? (2026-11-04) |
| 11-04 | [G4_CAMCLASS_MIXSHIFT_2026-11-04.md](G4_CAMCLASS_MIXSHIFT_2026-11-04.md) | G4 by camera class + mix-shift test of the state-discriminator frame (2026-11-04) |
| 11-04 | [CUE70_CHECK_2026-11-04.md](CUE70_CHECK_2026-11-04.md) | Cue + 70 ms contact floor: is 70 ms right? (2026-11-04) |
| 11-04 | [CODE_REVIEW_2026-11-04.md](CODE_REVIEW_2026-11-04.md) | Code review of `widefield_pipeline` (main at fc09de78, 2026-11-04) |
| 11-04 | [CENTRE_MIXSHIFT_2026-11-04.md](CENTRE_MIXSHIFT_2026-11-04.md) | Which rows should set the scoring centre? Spout decoder and C projections (2026-11-04) |
| 11-03 | [STATUS_2026-11-03_TRIAL_GATES_S1_TABLE_HANDOFF.md](STATUS_2026-11-03_TRIAL_GATES_S1_TABLE_HANDOFF.md) | Handoff 2026-11-03 — camera trial gates, jaw-only / incomplete figures, imaging section 1 table (window 5b) |
| 11-03 | [STATUS_2026-11-03_KINEMATICS_PAGE_REPO_AUDIT_HANDOFF.md](STATUS_2026-11-03_KINEMATICS_PAGE_REPO_AUDIT_HANDOFF.md) | Handoff 2026-11-03 — kinematics page (behaviour split) + repo consolidation audit |
| 11-03 | [STATE_BALEM_PAGE_EDITS_2026-11-03.md](STATE_BALEM_PAGE_EDITS_2026-11-03.md) | Imaging page, section 3: edits for the bal_em production state decoder (2026-11-03) |
| 11-03 | [SPOUT_BALEM_PAGE_EDITS_2026-11-03.md](SPOUT_BALEM_PAGE_EDITS_2026-11-03.md) | Spout-position decoder on bal_em: results-page edits (2026-11-03) |
| 11-03 | [RUNNING_LIST_2026-11-03_AWAY.md](RUNNING_LIST_2026-11-03_AWAY.md) | Running list, 2026-11-03 (Priya away) -> overnight into 11-04, PI-deck window (sabatinilab-a8) |
| 11-03 | [REPO_CONSOLIDATION_PLAN_2026-11-03.md](REPO_CONSOLIDATION_PLAN_2026-11-03.md) | Repo consolidation plan, 2026-11-03 (DRAFT, for Priya's approval) |
| 11-03 | [REPO_AUDIT_VERIFY_2026-11-03.md](REPO_AUDIT_VERIFY_2026-11-03.md) | VERIFY: independent check of AUDIT.md (2026-11-03) |
| 11-03 | [REPO_AUDIT_2026-11-03.md](REPO_AUDIT_2026-11-03.md) | Redundancy and clarity audit: widefield_pipeline (main) and wf_attempt (branch analysis/attempt-coding) |
| 11-03 | [QUIT_TAIL_JAW_ONLY_2026-11-03.md](QUIT_TAIL_JAW_ONLY_2026-11-03.md) | Jaw-only trials after the session's last contact: same as the others? (11-03, e3) |
| 11-03 | [JAW_ONLY_CHECKS_2026-11-03.md](JAW_ONLY_CHECKS_2026-11-03.md) | Jaw-only trials in "working"? — checks, 2026-11-03 |
| 11-03 | [FIGURE_RULES_AUDIT_2026-11-03.md](FIGURE_RULES_AUDIT_2026-11-03.md) | Figure rules audit, 2026-11-03 |
| 11-03 | [DECK_AUDIT_2026-11-03.md](DECK_AUDIT_2026-11-03.md) | PI deck audit, 2026-11-03 |
| 11-03 | [A3_LMM_SLOPE_STARS_2026-11-03.md](A3_LMM_SLOPE_STARS_2026-11-03.md) | A3 star rule: random-slope LMM on the 21 behaviour stars (2026-11-03) |
| 11-02 | [RUNNING_LIST_2026-11-02_OVERNIGHT.md](RUNNING_LIST_2026-11-02_OVERNIGHT.md) | Running list — overnight 2026-11-02 → 11-03 (Priya: "go ahead with all that you can without me overnight. keep handoff updated and a running list o... |
| 11-01 | [STATUS_2026-11-01_QUIET_EVENTS_HANDOFF.md](STATUS_2026-11-01_QUIET_EVENTS_HANDOFF.md) | Handoff 2026-11-01: quiet-period cortical events after the VLS stroke (cohort1, PS92-95) + literature context |
| 11-01 | [RUNNING_LIST_2026-11-01_QUIET_EVENTS.md](RUNNING_LIST_2026-11-01_QUIET_EVENTS.md) | Running list 2026-11-01: quiet-period cortical events, hypothesis tests (cohort1 PS92-95) |
| 10-27 | [SVD_GAUSS_SUPERSEDE_QUEUE_2026-10-27.md](SVD_GAUSS_SUPERSEDE_QUEUE_2026-10-27.md) | Supersede queue: SVD-basis decoder + Gaussian same-place read-out as PRIMARY (Priya, 10-27 late) |
| 10-27 | [REVIEW_FOR_PRIYA_2026-10-27.md](REVIEW_FOR_PRIYA_2026-10-27.md) | For Priya to review (overnight 10-26 -> 10-27, encoding window sabatinilab-d4) |
| 10-26 | [STATUS_2026-10-26_ENCODING_HANDOFF.md](STATUS_2026-10-26_ENCODING_HANDOFF.md) | Encoding handoff — 2026-10-26 (updated ~22:45) — READ FIRST |
| 10-24 | [STATUS_2026-10-24_FACE_RHYTHM_HANDOFF.md](STATUS_2026-10-24_FACE_RHYTHM_HANDOFF.md) | FaceRhythm handoff (2026-10-24; afternoon update at the top of section 2) -- START HERE |
| 10-24 | [OUTPUT_INVENTORY_2026-10-24.md](OUTPUT_INVENTORY_2026-10-24.md) | Encoding-window output inventory (2026-10-24) |
| 10-24 | [OUTPUT_AUDIT_2026-10-24.md](OUTPUT_AUDIT_2026-10-24.md) | Output staleness audit -- widefield VLS-stroke (PS92-PS95, lesions LEFT) -- 2026-10-24 |
| 10-22 | [STATUS_2026-10-22_FACE_RHYTHM_HANDOFF.md](STATUS_2026-10-22_FACE_RHYTHM_HANDOFF.md) | > **SUPERSEDED for current state by [`STATUS_2026-10-24_FACE_RHYTHM_HANDOFF.md`](STATUS_2026-10-24_FACE_RHYTHM_HANDOFF.md)** (rank 12, names, own/... |
| 10-22 | [STATUS_2026-10-22_ENCODING_HANDOFF.md](STATUS_2026-10-22_ENCODING_HANDOFF.md) | Handoff — 2026-10-22: LP_licks results, frozen pre-stroke encoder, Musall-style readouts, FaceRhythm pilot |
| 10-21 | [STATUS_2026-10-21_POSE_CAM1_JAW_HANDOFF.md](STATUS_2026-10-21_POSE_CAM1_JAW_HANDOFF.md) | Handoff — 2026-10-21 (morning): pose / 3-D / cam1 — jaw calls, whole-session licks, per-session tables |
| 10-21 | [STATUS_2026-10-21_FACE_RHYTHM_O2_HANDOFF.md](STATUS_2026-10-21_FACE_RHYTHM_O2_HANDOFF.md) | FaceRhythm on O2 — handoff (2026-10-21, updated ~22:00) |
| 10-21 | [STATUS_2026-10-21_ENCODING_HANDOFF.md](STATUS_2026-10-21_ENCODING_HANDOFF.md) | Handoff — 2026-10-21: attempt-based deficit, target vs execution vs neither, within-spout per spout + contact, map statistics, LP_licks |
| 10-20 | [STATUS_2026-10-20_POSE3D_TIMING_HANDOFF.md](STATUS_2026-10-20_POSE3D_TIMING_HANDOFF.md) | Handoff — 2026-10-20 (midday): template timing, cam1 2-D licks, jaw anchor, full v36 + 3-D re-run in flight |
| 10-20 | [STATUS_2026-10-20_EVENING_ENCODING_HANDOFF.md](STATUS_2026-10-20_EVENING_ENCODING_HANDOFF.md) | Handoff — 2026-10-20 (evening): within-spout check, mixed model, 3-D contest queued, PS92_0908 rerun |
| 10-20 | [STATUS_2026-10-20_ENCODING_HANDOFF.md](STATUS_2026-10-20_ENCODING_HANDOFF.md) | Handoff — 2026-10-20: encoding model (3-D executed sets, drop-p per-spout maps, recovery timing, slow band) |
| 10-19 | [STATUS_2026-10-19_POSE3D_HANDOFF.md](STATUS_2026-10-19_POSE3D_HANDOFF.md) | Handoff — 2026-10-19 (evening): cam1 production on O2, the 3-D tongue / jaw pipeline, first 3-D readouts |
| 10-19 | [STATUS_2026-10-19_EVENING_ENCODING_HANDOFF.md](STATUS_2026-10-19_EVENING_ENCODING_HANDOFF.md) | Handoff — 2026-10-19 evening: encoding model results (fair target vs executed, lick-matched, trial-to-trial), misdirected licks, per-spout maps |
| 10-19 | [STATUS_2026-10-19_ENCODING_O2_HANDOFF.md](STATUS_2026-10-19_ENCODING_O2_HANDOFF.md) | Handoff — 2026-10-19: combined encoding model (target vs executed, drift, credit rules), two-band test, O2 path |
| 10-18 | [STATUS_2026-10-18_OROFACIAL_W2_ENCODING_HANDOFF.md](STATUS_2026-10-18_OROFACIAL_W2_ENCODING_HANDOFF.md) | Handoff — 2026-10-18: cam4 orofacial cohort, W2, baselines, imaging sensitivity, Musall-style encoding model |
| 10-18 | [STATUS_2026-10-18_EVENING_V36_BASELINE_ENCODING_HANDOFF.md](STATUS_2026-10-18_EVENING_V36_BASELINE_ENCODING_HANDOFF.md) | Handoff — 2026-10-18 evening: LP run v36, August-only baseline, encoding-model redesign, maps / ME in flight |
| 10-18 | [STATUS_2026-10-18_CAM1_ROUND2_3D_HANDOFF.md](STATUS_2026-10-18_CAM1_ROUND2_3D_HANDOFF.md) | Handoff — 2026-10-18: cam1 round-2 retrain, cam1 clean traces, the 3-D apparatus mouth, cam4 eps35 |
| 10-16 | [STATUS_2026-10-16_O2_PRODUCTION_OUTPUTS_HANDOFF.md](STATUS_2026-10-16_O2_PRODUCTION_OUTPUTS_HANDOFF.md) | Handoff — 2026-10-15 late / 10-16: LP r4 production on O2, organised orofacial outputs, split-session fixes |
| 10-15 | [STATUS_2026-10-15_JAW_CLEANING_MODEL_CHOICE_HANDOFF.md](STATUS_2026-10-15_JAW_CLEANING_MODEL_CHOICE_HANDOFF.md) | Handoff — 2026-10-15 (morning): jaw cleaning v35, jaw lateral readout across the cohort, LP r4 ctx-semi scored, jaw-source choice |
| 10-14 | [STATUS_2026-10-14_POSE_MODELS_HANDOFF.md](STATUS_2026-10-14_POSE_MODELS_HANDOFF.md) | Handoff — 2026-10-14 (late morning): cam4 pose-model selection (DLC / LP x round 3 / 4, temporal-context LP, EKS, human lick review), side-camera l... |
| 10-13 | [STATUS_2026-10-13_ENL_TRAJECTORY_TRACKING_R4_HANDOFF.md](STATUS_2026-10-13_ENL_TRAJECTORY_TRACKING_R4_HANDOFF.md) | Handoff — 2026-10-13: movement-removed ENL trajectory (118 sessions), null/potent closed, cam4 round 4 (DLC + LP, temporal-context LP), camera temp... |
| 10-07 | [STATUS_2026-10-07_MOVEMENT_SUBSPACES_HANDOFF.md](STATUS_2026-10-07_MOVEMENT_SUBSPACES_HANDOFF.md) | Handoff — 2026-10-07 (evening): movement-removed position coding (residual + ENL + null/potent subspaces), video regressors, multi-view LP pilot, c... |
| 10-07 | [STATUS_2026-10-07_CHRONIC_STABILITY_COHORT_COMPLETE.md](STATUS_2026-10-07_CHRONIC_STABILITY_COHORT_COMPLETE.md) | STATUS — 2026-10-07: Chronic neural stability → cohort imaging COMPLETE (PS92/PS93/PS94/PS95) |
| 10-06 | [STATUS_2026-10-06_FULLSESSION_MOVEMENT_HANDOFF.md](STATUS_2026-10-06_FULLSESSION_MOVEMENT_HANDOFF.md) | Handoff — 2026-10-06: first whole-session (O2) analyses, movement-model upgrades, contact classes, video regressors |
| 10-05 | [STATUS_2026-10-05_IMAGING_MOVEMENT_HANDOFF.md](STATUS_2026-10-05_IMAGING_MOVEMENT_HANDOFF.md) | Handoff — 2026-10-05: orofacial data in the widefield analyses (movement encoding, residual position coding, executed angle), O2 port |
| 10-05 | [STATUS_2026-10-05_CAM1_LP_EKS_O2_HANDOFF.md](STATUS_2026-10-05_CAM1_LP_EKS_O2_HANDOFF.md) | Handoff — 2026-10-05 (evening): cam1 DLC + LP, cam1 labelling round 2, 3-D spout frame, multi-camera EKS, first O2 run |
| 10-02 | [STATUS_2026-10-02_SESSIONS_REACH_O2_HANDOFF.md](STATUS_2026-10-02_SESSIONS_REACH_O2_HANDOFF.md) | Handoff — 2026-10-02: session-level kinematics (PS93 pre / acute / chronic), lick phase, reach measures, O2 |
| 10-01 | [STATUS_2026-10-01_KINEMATICS_HANDOFF.md](STATUS_2026-10-01_KINEMATICS_HANDOFF.md) | Handoff — 2026-10-01: clean traces, spout reference frame, tongue/jaw kinematics port, first QC on our data |
| 09-30 | [STATUS_2026-09-30_LP_OCCLUSION_ROUND4_HANDOFF.md](STATUS_2026-09-30_LP_OCCLUSION_ROUND4_HANDOFF.md) | > **2026-10-01:** kinematics work continues in `STATUS_2026-10-01_KINEMATICS_HANDOFF.md`; this doc remains the |
| 09-28 | [STATUS_2026-09-28_DLC_LIGHTNING_POSE_HANDOFF.md](STATUS_2026-09-28_DLC_LIGHTNING_POSE_HANDOFF.md) | Handoff — cam4 orofacial tracking: DLC round 3 in use; Lightning Pose (KL) training in WSL2; cam1 labelling under way |
| 09-28 | [STATUS_2026-09-28_CHRONIC_STABILITY.md](STATUS_2026-09-28_CHRONIC_STABILITY.md) | Is the chronic neural data stable? — per-session, per-animal, per-readout (2026-09-28) |
| 09-24 | [STATUS_2026-09-24_ENL_SENSORY_VS_PLAN.md](STATUS_2026-09-24_ENL_SENSORY_VS_PLAN.md) | ENL sensory-vs-plan — status and handoff (2026-09-24) |
| 09-24 | [STATUS_2026-09-24_CD_TRAJECTORIES.md](STATUS_2026-09-24_CD_TRAJECTORIES.md) | CD trajectories — status and handoff (2026-09-24) |
| 09-21 | [STATUS_2026-09-21_ENGINEERING.md](STATUS_2026-09-21_ENGINEERING.md) | HANDOFF 2026-09-21 — codebase quality: efficient, modular, readable, editable |
| 09-20 | [STATUS_2026-09-20.md](STATUS_2026-09-20.md) | STATUS 2026-09-20 — haemodynamics closed, engagement standing, loops parallelised |
| 09-19 | [STATUS_2026-09-19.md](STATUS_2026-09-19.md) | STATUS 2026-09-19 — the 415 nm day, and the engagement decomposition |
| 09-17 | [STATUS_2026-09-17.md](STATUS_2026-09-17.md) | STATUS 2026-09-17 — START HERE |
| 09-16 | [STATUS_2026-09-16.md](STATUS_2026-09-16.md) | STATUS 2026-09-16 / 17 — START HERE |
| 09-13 | [STATUS_2026-09-13.md](STATUS_2026-09-13.md) | Status 2026-09-13 — the REST baseline migration |
| 09-12 | [STATUS_2026-09-12.md](STATUS_2026-09-12.md) | Status, 2026-09-12 — the anatomical arm |
| 09-10 | [STATUS_2026-09-10.md](STATUS_2026-09-10.md) | Status, 2026-09-10 — what changed, what it showed, what is pending |
| 09-07 | [STATUS_2026-09-07.md](STATUS_2026-09-07.md) | Handoff, 2026-09-07 — the engagement gate, the camera clips, and what moved because of them |
| 08-28 | [STATUS_2026-08-28_night.md](STATUS_2026-08-28_night.md) | Handoff, 2026-08-28 night — epoch figures, the engagement gate, and the parallel render |
| 08-28 | [STATUS_2026-08-28_handoff.md](STATUS_2026-08-28_handoff.md) | Handoff, 2026-08-28 — five approved items, with the plumbing already traced |
| 08-28 | [STATUS_2026-08-28_evening.md](STATUS_2026-08-28_evening.md) | Handoff, 2026-08-28 evening — what is done, what is running, what is next |
| 08-26 | [STATUS_2026-08-26.md](STATUS_2026-08-26.md) | Status and handover — 2026-08-26 |
| 08-23 | [STATUS_2026-08-23.md](STATUS_2026-08-23.md) | Status and handoff — 2026-08-23 |
| 08-17 | [STATUS_2026-08-17.md](STATUS_2026-08-17.md) | Status and handoff — 2026-08-17 |
| 08-14 | [STATUS_2026-08-14.md](STATUS_2026-08-14.md) | Status and handoff — 2026-08-14 |
| 08-09 | [STATUS_2026-08-09_TASKS.md](STATUS_2026-08-09_TASKS.md) | Ongoing tasks & decisions (widefield pipeline) — RETIRED 2026-09-22 |
