# START HERE — the one entry point for a new lab member

Written 2026-11-04 from the code review of that day
([`status/CODE_REVIEW_2026-11-04.md`](status/CODE_REVIEW_2026-11-04.md), "Docs a new lab member would trip on").
This repo has many "start here" lines. This page puts them in one order. Section 1 (analysis branches) was
rewritten and checked against `main` on 2026-11-17; sections 2 and 3 were checked on 2026-11-04.

Read in this order:

1. **What we have found:** [`FINDINGS.md`](FINDINGS.md). One status-tagged line per claim, all arms. The per-arm
   FINDINGS docs (`ENCODING_FINDINGS.md`, `FACE_RHYTHM_FINDINGS.md`, `OROFACIAL_FINDINGS.md`) hold the detail.
2. **The rules:** [`../CLAUDE.md`](../CLAUDE.md), the ground rules at the top (never delete original data, configs are
   the single source of truth, fan per-session loops over cores, one definition per quantity, frozen models).
3. **Why things are the way they are:** [`../DECISIONS.md`](../DECISIONS.md). It is long; search it for the topic.
4. **Your analysis branch** (section 1: question, results page, findings, handoff, code).
5. **How the code is laid out:** [`ARCHITECTURE.md`](ARCHITECTURE.md), then [`../scripts/INDEX.md`](../scripts/INDEX.md)
   for the 415 scripts outside `archive/`.

Two cohorts exist. **Cohort 1** = PS92-PS95. **Cohort 2** = PS118, PS120, PS121, PS122, imaged from 2026-10-29. They
are analyzed separately and never pooled. Read [`NEW_COHORT_2026-10-29.md`](NEW_COHORT_2026-10-29.md) before running
anything: hand runs on cohort 2 need `WIDEFIELD_COHORT=cohort2`.

---

## 1. Analysis branches: where each one stands

The project runs as several analysis BRANCHES, each with its own question, results page, findings doc, handoff and
code. Find yours below. A "handoff" is the note a working session leaves for the next one (state, what is running,
to-dos, pitfalls); most handoffs grow dated blocks at the top, so **read the block named here first**. The full list
of status docs is in [`status/README.md`](status/README.md); CLAUDE.md "Current status / handoff" keeps older ones.
Results pages are claude.ai Artifacts on Priya's account (ask for access). Rewritten 2026-11-17: when you start,
move or retire a branch, edit its block here in the same commit.

**The two cortical results pages (renamed 2026-11-18, Priya):** **Spout (Target) Coding After Stroke**
https://claude.ai/artifact/PvraHiF82BFXur1hnNUCeV (was "Cortical Spout Coding" / "the imaging page": whether and where cortex carries the
TARGET spout: position decoders, frozen decoder, specificity, maps, coding directions, RSA, pre-cue code, rest) and
**Movement (Execution) Coding After Stroke** https://claude.ai/artifact/Fnpn2KHknMwh3r13xADH7d (was "Attempt Encoding Figures" /
"the goal page" / "the encoding page": what cortex carries about the EXECUTED movement: encoders, tongue / lick
read-out, FaceRhythm, state, quiet events). Each page links the other under its h1. Older docs use the old names;
same URLs. Quiet-period events live on the movement page (section 23); spout page section 15 is a pointer that keeps
the old figure anchors. Failed attempts appear on both (spout sections 5 / 8, movement sections 8 / 9, cross-linked).

Shared by every branch: trial classes and the working gate (branch D), the August-only pre baseline (CLAUDE.md
"Current status"), every readout as the change from the animal's OWN pre, significance stars from the nested
bootstrap (`analysis_kit.boot_delta`), and the block-permutation null for any label that follows the spout blocks
(branch J).

### A. Behavior (task performance, engagement, licking from the DAQ)
- **Question:** what does the stroke do to task performance, engagement and the quit tail?
- **Page:** behavior https://claude.ai/artifact/M1xhZBWSPS84aELTSFaMMK
- **Findings:** [`FINDINGS.md`](FINDINGS.md); [`WITHIN_SESSION_DECLINE_QUITTING.md`](WITHIN_SESSION_DECLINE_QUITTING.md);
  engagement in [`status/STATUS_2026-09-20.md`](status/STATUS_2026-09-20.md)
- **Handoff:** [`status/STATUS_2026-11-03_KINEMATICS_PAGE_REPO_AUDIT_HANDOFF.md`](status/STATUS_2026-11-03_KINEMATICS_PAGE_REPO_AUDIT_HANDOFF.md) (behavior / kinematics split)
- **Code:** `daq_trials.py`, `spout_behavior.py`, `behavior_events.py`, `trial_classes.py`

### B. Pose tracking and 3-D (DLC / Lightning Pose, cam1-cam4, side cameras)
- **Question:** clean tongue / jaw / nose positions in 2-D and 3-D, on the DAQ clock, for every other branch.
- **Register:** [`POSE_MODELS.md`](POSE_MODELS.md) (every model and how it was chosen; Part E = side cameras and
  four-camera 3-D)
- **Handoff:** [`status/STATUS_2026-11-03_TRIAL_GATES_S1_TABLE_HANDOFF.md`](status/STATUS_2026-11-03_TRIAL_GATES_S1_TABLE_HANDOFF.md)
  ("START HERE (window e0, 11-04 late)" block); before 11-03:
  [`status/STATUS_2026-10-21_POSE_CAM1_JAW_HANDOFF.md`](status/STATUS_2026-10-21_POSE_CAM1_JAW_HANDOFF.md)
- **cam2 / cam3 re-run (from 11-18):** what must be re-run with the four-camera 3-D and lick data, decisions first:
  [`status/STATUS_2026-11-18_CAM23_RERUN_INVENTORY.md`](status/STATUS_2026-11-18_CAM23_RERUN_INVENTORY.md)
- **Code:** `pose3d.py`, `pose3d_side.py`, `lp_licks.py`, `lp_multiview.py`, `lp_sidecam.py`, `orofacial_clean.py`;
  O2 runbook [`../runbooks/o2_pose_inference.md`](../runbooks/o2_pose_inference.md)

### C. Kinematics (tongue / jaw readouts from the tracking: direction, aim, attempts, incomplete licks)
- **Question:** how do individual licks change (direction, accuracy, timing, jaw steering) after the stroke?
- **Page:** kinematics https://claude.ai/artifact/DKcKRJKbhBoratxxRc6qZv
- **Findings:** [`OROFACIAL_FINDINGS.md`](OROFACIAL_FINDINGS.md)
- **Handoff:** [`status/STATUS_2026-11-03_KINEMATICS_PAGE_REPO_AUDIT_HANDOFF.md`](status/STATUS_2026-11-03_KINEMATICS_PAGE_REPO_AUDIT_HANDOFF.md) (section 00)
- **Code:** `orofacial_cohort.py` (the LP run `RUN`, spout order `POS_ORDER`), `orofacial_outputs.py`,
  `tongue_kinematics.py`, `jaw_kinematics.py`, `kinematics_chain.py`, `orofacial_stats.py`

### D. Trial classes and gates (contact / incomplete / jaw-only / no response; working; quit tail)
- **Question:** which trials count. One definition, used by every branch.
- **Handoff:** [`status/STATUS_2026-11-03_TRIAL_GATES_S1_TABLE_HANDOFF.md`](status/STATUS_2026-11-03_TRIAL_GATES_S1_TABLE_HANDOFF.md); DECISIONS 2026-11-03
- **Code:** `trial_classes.py` (`GATES`), `working_trials.py`; section 2 below says how to change it

### E. FaceRhythm (whole-face optic-flow factors)
- **Question:** unbiased whole-face movement factors (the master pre-stroke set `master_pre_v1`, rank 12) and how
  they change after the stroke.
- **Page:** FaceRhythm https://claude.ai/artifact/X8NDdQNHi3zMP77PBrZAC1
- **Findings:** [`FACE_RHYTHM_FINDINGS.md`](FACE_RHYTHM_FINDINGS.md); pipeline [`FACE_RHYTHM_PIPELINE.md`](FACE_RHYTHM_PIPELINE.md)
- **Handoff:** [`status/STATUS_2026-10-24_FACE_RHYTHM_HANDOFF.md`](status/STATUS_2026-10-24_FACE_RHYTHM_HANDOFF.md)
  (section 2: the newest `00-A8-...` item first)
- **Code:** `face_rhythm/` (factors, projection, O2), `face_rhythm_outputs.py`, `face_rhythm_readouts.py`, `fr_groups.py`

### F. Neural coding of spout position (decoders, maps, frozen pre-stroke decoders): the Spout (Target) Coding page
- **Question:** does cortex carry spout-position information before and after the cue, and how does that change?
- **Page:** Spout (Target) Coding After Stroke (the former imaging page) https://claude.ai/artifact/PvraHiF82BFXur1hnNUCeV (rebuild chain:
  [`results_pages/imaging/README.md`](results_pages/imaging/README.md))
- **Findings:** [`FINDINGS.md`](FINDINGS.md); [`WHERE_THE_CODE_MOVES.md`](WHERE_THE_CODE_MOVES.md);
  [`BEHAVIOURAL_STATE_CONTROL.md`](BEHAVIOURAL_STATE_CONTROL.md); [`FROZEN_MODELS.md`](FROZEN_MODELS.md);
  [`REST_ENGAGEMENT_AUDIT.md`](REST_ENGAGEMENT_AUDIT.md) (read before quoting any rest number)
- **Handoff:** [`status/STATUS_2026-11-03_TRIAL_GATES_S1_TABLE_HANDOFF.md`](status/STATUS_2026-11-03_TRIAL_GATES_S1_TABLE_HANDOFF.md)
  (imaging section 1). Older sub-arms: ENL sensory vs plan
  [`status/STATUS_2026-09-24_ENL_SENSORY_VS_PLAN.md`](status/STATUS_2026-09-24_ENL_SENSORY_VS_PLAN.md); CD trajectories
  [`status/STATUS_2026-09-24_CD_TRAJECTORIES.md`](status/STATUS_2026-09-24_CD_TRAJECTORIES.md); chronic stability
  [`status/STATUS_2026-10-07_CHRONIC_STABILITY_COHORT_COMPLETE.md`](status/STATUS_2026-10-07_CHRONIC_STABILITY_COHORT_COMPLETE.md);
  rest baseline [`status/STATUS_2026-09-16.md`](status/STATUS_2026-09-16.md); hemodynamics (CLOSED)
  [`status/STATUS_2026-09-20.md`](status/STATUS_2026-09-20.md)
- **Code:** `locanmf_position_decoder.py`, `locanmf_frozen_decoder.py`, `frozen_models.py`, `nolick_decoder.py`,
  `enl_decode.py`, `epoch_figures.py` / `epoch_grant_figures.py`, `attempt_coding.py`

### G. Encoding model (Musall-style ridge encoder: task, movement, FaceRhythm, contact)
- **Question:** how much of cortical activity is task vs movement, and does the stroke change that split?
- **Page:** Movement (Execution) Coding After Stroke (the former goal / encoding page) https://claude.ai/artifact/Fnpn2KHknMwh3r13xADH7d
- **Findings:** [`ENCODING_FINDINGS.md`](ENCODING_FINDINGS.md); [`ENCODER_CEILING.md`](ENCODER_CEILING.md)
- **Handoff:** [`status/STATUS_2026-10-26_ENCODING_HANDOFF.md`](status/STATUS_2026-10-26_ENCODING_HANDOFF.md)
  (the newest `00-COMPACT-...` block first)
- **Code:** `combined_encoding.py` (per session), `frozen_encoder.py` (frozen pre-stroke); O2 `scripts/o2_encoding.py`

### H. Quiet-period cortical events (and the PI deck)
- **Question:** why do anterolateral orofacial cortical events rise ~5x acutely while the face is still?
- **Page:** section 23 ("quiet-state") of the Movement (Execution) Coding page; Spout (Target) Coding section 15 is now
  only a pointer to it (old figure anchors kept)
- **Handoff:** [`status/STATUS_2026-11-01_QUIET_EVENTS_HANDOFF.md`](status/STATUS_2026-11-01_QUIET_EVENTS_HANDOFF.md) (the top `>` block first)
- **Code:** `quiet_periods.py`, `restnl_mask.py`

### I. State space / BCI-style recovery
- **Question:** is post-stroke cortical activity the pre-stroke activity re-mapped, re-weighted, or new?
- **Page:** state space https://claude.ai/artifact/9sGELLFEwADEt9k64fXTxe (source [`results_pages/state_space/`](results_pages/state_space/))
- **Plan / code map:** [`STATE_SPACE_RECOVERY_PLAN.md`](STATE_SPACE_RECOVERY_PLAN.md), [`STATE_SPACE_CODE.md`](STATE_SPACE_CODE.md),
  [`STATE_SPACE_RECOVERY_LITERATURE.md`](STATE_SPACE_RECOVERY_LITERATURE.md)
- **Handoff:** [`status/STATUS_2026-11-07_STATE_SPACE_HANDOFF.md`](status/STATUS_2026-11-07_STATE_SPACE_HANDOFF.md) (the newest "START HERE" block)

### J. Behavioral states, task-independent movement (TIM), AR-HMM syllables, block permutation (from 11-16)
- **Question:** separate instructed vs uninstructed and task-aligned vs task-independent movement (Yin et al. 2025);
  engagement states over ALL trials (GLM-HMM); movement syllables (AR-HMM, adapted from Hillary Tong's model, fit on
  O2); the block-permutation null for labels that follow the spout blocks.
- **Handoff:** `docs/status/STATUS_2026-11-16_ENCDESIGN_TIM_STATES_HANDOFF.md` on branch `analysis/tim-encdesign-1116`
  (design `docs/status/ENCODING_DESIGN_TIM_STATES_2026-11-16.md`; audit `docs/status/BLOCK_PERMUTATION_AUDIT_2026-11-16.md`)
- **Code (on that branch):** `task_independent_movement.py`, `glm_hmm.py`, `arhmm_features.py`, `arhmm_model.py`,
  `arhmm_states.py`, `scripts/o2_arhmm.py`; the block null `analysis_kit.block_permute` on branch
  `fix/block-perm-null-1116`
- **Results page:** https://claude.ai/artifact/ChbBM9K12WWRohx6nU6rCX (GLM-HMM engagement, treadmill push, AR-HMM
  syllables, TIM, frozen pre-stroke likelihood); findings: [`FINDINGS.md`](FINDINGS.md) section 9; one module for every
  state readout: `behavior_states.py` (+ `scripts/states_page_figs.py`), handoff section 3b.
- **Status:** not merged into main yet; every result is provisional.

### K. Anatomy and histology
- **Question:** lesion extent per animal; where on cortex the displaced position code moves.
- **Docs:** [`HISTOLOGY.md`](HISTOLOGY.md); [`WHERE_THE_CODE_MOVES.md`](WHERE_THE_CODE_MOVES.md) (with [`status/STATUS_2026-09-12.md`](status/STATUS_2026-09-12.md))
- **Code:** `histology_area.py`, `histology_map/`

### L. Cohort 2 (PS118, PS120-122; never pooled with cohort 1)
- **Docs:** [`NEW_COHORT_2026-10-29.md`](NEW_COHORT_2026-10-29.md), [`COHORT2_QC_2026-10-29.md`](COHORT2_QC_2026-10-29.md);
  hand runs need `WIDEFIELD_COHORT=cohort2`

### M. Code and engineering (refactors, consolidation)
- **Handoff:** [`status/CODE_REVIEW_2026-11-04.md`](status/CODE_REVIEW_2026-11-04.md) (verified findings, order of work);
  [`status/REPO_CONSOLIDATION_PLAN_2026-11-03.md`](status/REPO_CONSOLIDATION_PLAN_2026-11-03.md),
  [`status/REPO_AUDIT_2026-11-03.md`](status/REPO_AUDIT_2026-11-03.md),
  [`status/STATUS_2026-09-21_ENGINEERING.md`](status/STATUS_2026-09-21_ENGINEERING.md)

### N. Grant figures (code that composed figures for grant applications; not the pipeline)
- **Question:** none of its own. Re-plots and composes panels from outputs the other branches already wrote, for a
  specific grant. Nothing here recomputes a result or feeds the pipeline; a wrong number is fixed upstream and re-pulled.
- **Section:** [`../grants/README.md`](../grants/README.md), one folder per grant.
- **K08 resubmission, Nov 2026** (Research Strategy Figures 3-5: behavior and deficit, FaceRhythm, decoder and encoder):
  [`../grants/k08_2026/README.md`](../grants/k08_2026/README.md); provenance by final panel in
  [`../grants/k08_2026/PROVENANCE.md`](../grants/k08_2026/PROVENANCE.md); the stats file behind every star in
  [`../grants/k08_2026/FIGURE_SOURCES.md`](../grants/k08_2026/FIGURE_SOURCES.md)
- **Code:** `grants/k08_2026/` (`render_grant_panels.py`, `render_fig4_rows.py`, `make_fig1/2/3.py`, `build_fig.py`, ...),
  plus `grants/k08_2026/session_scripts/` (added 11-17): the last-mile scripts that relettered Figure 3, split and
  lightened Figure 4D, and swapped in the Figure 5D cued panel. They had never been copied out of the session that
  wrote them, so until 11-17 the submitted figures were not reproducible from the repo.
  Not to be confused with the pipeline's nightly summary set `wfield_local/grant_figures.py` / `epoch_grant_figures.py`
  (branch F), which these scripts read from.

---

## 2. To change analysis X

"Config key" means a block in `configs/defaults.yaml` unless it says otherwise. Many result-changing switches are
still **environment variables**, not yaml keys; the table names the important ones.
"Registry" is the list that says which figures exist and where they go.

| analysis | module(s) | config key / switch | registry | tests | re-render |
|---|---|---|---|---|---|
| **Position decoder (within session)** | `wfield_local/locanmf_position_decoder.py` (features, estimator `_pipe`, block CV) | `decode:` (aligns, `*_post_s`, `max_rt_s`, `cv`, `baseline`) | `grant_figures.JOBS` (keys `2`, `2b`, `4`-`5d`, ...) | `test_block_ids.py`, `test_feature_alignment.py`, `test_trial_features_cache.py` | `python -m wfield_local.grant_figures --only <key> -j <N>` |
| **Frozen decoder (pre-stroke reference)** | `wfield_local/frozen_models.py` (`load_or_fit`, `spec_id`), `wfield_local/locanmf_frozen_decoder.py` | store location derives from `configs/paths.yaml figures_working`; `WIDEFIELD_FROZEN_MODEL_DIR` overrides. Read [`FROZEN_MODELS.md`](FROZEN_MODELS.md) first | [`FROZEN_MODELS.md`](FROZEN_MODELS.md) and the store itself | `test_frozen_models.py`, `test_frozen_models_are_prestroke_only.py`, `test_frozen_store_move_is_loud.py`, `test_frozen_vs_refit.py` | `python -m wfield_local.epoch_grant_figures --only 5r` (frozen vs refit; also `5rm`, `5ro`, `5rmo`) |
| **Encoder** | `wfield_local/combined_encoding.py` (per session), `wfield_local/frozen_encoder.py` (frozen pre-stroke) | env switches `COMBINED_ENCODING_*` and `FROZEN_*` (no yaml block yet) | output folders under the LP run (`--run`, default `orofacial_cohort.RUN`) | `test_combined_encoding_options.py`, `test_frozen_encoder.py`, `test_encoder_terms.py` | `python -m wfield_local.combined_encoding --all --jobs 3`; `python -m wfield_local.frozen_encoder --all`; figures `scripts/combined_encoding_figures.py`; on O2 `scripts/o2_encoding.py` |
| **Trial classes / working gate** | `wfield_local/trial_classes.py` (one class per trial, `GATES`), `wfield_local/working_trials.py` (decoder rows) | env `WIDEFIELD_WORKING_TRIALS` = `in_working` (production: contact + incomplete camera licks) / `in_working_jaw` / `quit_tail` | `trial_classes.GATES`, `working_trials.WORKING_CLASSES` | `test_trial_classes.py`, `test_working_trials.py`, `test_stopped_trial_class.py` | snapshot `scripts/s1_trial_class_snapshot.py`; then re-render the families that use the gate |
| **Lick source (DAQ contact vs video licks)** | `wfield_local/lp_licks.py`; DAQ detector in `wfield_local/plot_lick_aligned_averages.py` (`_load_daq_events`) | env `WIDEFIELD_LICK_SOURCE` = `daq` (default) / `lp`; DAQ thresholds in `lick_detection:`. Bump `lp_licks.LP_VERSION` when the LP definition changes | `lp_licks.LICK_SOURCES`; per-figure lick source in `docs/results_pages/imaging/provenance/lick_source.tsv` | `test_lp_licks.py`, `test_lick_detection.py`, `test_lick_discriminator_is_one_definition.py` | run the analysis with `WIDEFIELD_LICK_SOURCE=lp`; outputs go to LP-tagged folders (`lp_licks.output_dir`) |
| **Epoch figure family** | `wfield_local/epoch_figures.py`, `wfield_local/epoch_grant_figures.py` (driver) | `epochs:` (boundary rules); `WIDEFIELD_MAP_JOBS` for map collection | `deck_registry.EPOCH_FIGURES` | `test_epoch_figures.py`, `test_epoch_registry.py`, `test_nightly_renders_epoch_figures.py` | `python -m wfield_local.epoch_grant_figures --only <key> [--parallel-arms 3]` (`--only` takes a list) |
| **Analysis deck** | `wfield_local/locanmf_analysis_deck.py`, `deck_registry.py`, `deck_text.py`, `deck_values.py`, `deck_layout.py` | `deck:` | `deck_registry` | `test_analysis_deck.py`, `test_deck_values.py`, `test_deck_audit.py` | `python -m wfield_local.locanmf_analysis_deck [--src <figure dir>]`; check structure with `scripts/deck_fingerprint.py`. The PI deck is built by the PI-deck window (see the quiet-events handoff) |
| **Results pages** | `docs/results_pages/imaging/` (the `p_img_*.py` chain edits the HTML in place) | page format rules in [`templates/README.md`](templates/README.md) | `docs/results_pages/imaging/imaging_fig_ids.csv` (stable figure ids) | none on main | the rebuild chain in [`results_pages/imaging/README.md`](results_pages/imaging/README.md); publish with the Artifact tool |

Before you change a number, read CLAUDE.md rule 9 and the review's "Proposed order of work": pin the outputs before
and after, and record any changed number in DECISIONS.md.

---

## 3. Which computer you are on

Three Windows boxes run this repo. `wfield_local/paths.py` (`PathResolver`) picks the mounts for the box it is on:
the `WIDEFIELD_MACHINE` env var if set, else a signature path (for this desktop, `C:/Users/SabatiniLab`). The
mounts per box are in `configs/paths.yaml`. **Take paths from `PathResolver`, never type a drive letter.**

| | **Analysis desktop** (this box; profile `analysis_desktop`) | **Analysis / behavior GPU box** (profile `analysis`) | **Imaging (PCO) box** (profile `imaging`) |
|---|---|---|---|
| python | `C:/Users/SabatiniLab/miniconda3/envs/locanmf/python.exe` | `C:/Users/sabatini/.conda/envs/locanmf/python.exe` | `C:\ProgramData\anaconda3\envs\wfield` (env `wfield`) |
| repo | `C:/Users/SabatiniLab/Github/widefield_pipeline` | `C:\Users\sabatini\GitHub\widefield_pipeline` | `C:\Github\widefield_pipeline` |
| MICROSCOPE share | `M:` | `M:` | `N:` |
| standby share | `N:` | — | `M:` |
| local data | — | `D:` (camera + behavior-log staging) | `E:` (raw imaging + DAQ) |
| local figure dir (`figures_working`) | `C:/Users/SabatiniLab/cue_lick` | `C:/Users/sabatini/source/cue_lick` | `E:/cue_lick` |
| GPU / torch | RTX 5060 (Blackwell): torch **cu128** is required (`locanmf`, `dlc`, `lp` envs here have `2.11.0+cu128`) | RTX 4060: torch cu124 (`2.6.0+cu124`, README setup) | not needed (numba + numpy < 2.1) |
| runbook | this page + the arm handoffs | [`runbooks/analysis_computer_nightly.md`](../runbooks/analysis_computer_nightly.md) | [`runbooks/imaging_computer_nightly.md`](../runbooks/imaging_computer_nightly.md) |

**On this desktop the drive letters can move on reboot.** On 2026-09-30 they swapped, and today M: = MICROSCOPE and
N: = standby. `configs/paths.yaml` lists both letters for every share root on this box (first one that exists wins),
so `PathResolver` still finds the share; a path typed by hand does not. This desktop is also shared with other jobs
(2pRAM): stop only processes you started, by their exact PID.

Other envs on this desktop: `dlc` (DeepLabCut, orofacial analysis), `lp` (Lightning Pose), `fr`, `eks` (see README
"Per-machine environments" and [`GPU_SETUP.md`](GPU_SETUP.md)). Never run `litpose predict` in WSL with the default
96-frame chunk: it crashed this machine twice; use 16.

When you commit from a git worktree, run scripts with `PYTHONPATH=$(pwd)`, or the editable install runs the main
checkout's code instead of yours ([`../scripts/README.md`](../scripts/README.md)).
