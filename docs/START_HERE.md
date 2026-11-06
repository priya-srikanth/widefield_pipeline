# START HERE — the one entry point for a new lab member

Written 2026-11-04 from the code review of that day
([`status/CODE_REVIEW_2026-11-04.md`](status/CODE_REVIEW_2026-11-04.md), "Docs a new lab member would trip on").
This repo has many "start here" lines. This page puts them in one order. Every path and name below was checked
against `main` on 2026-11-04.

Read in this order:

1. **What we have found:** [`FINDINGS.md`](FINDINGS.md). One status-tagged line per claim, all arms. The per-arm
   FINDINGS docs (`ENCODING_FINDINGS.md`, `FACE_RHYTHM_FINDINGS.md`, `OROFACIAL_FINDINGS.md`) hold the detail.
2. **The rules:** [`../CLAUDE.md`](../CLAUDE.md), the ground rules at the top (never delete original data, configs are
   the single source of truth, fan per-session loops over cores, one definition per quantity, frozen models).
3. **Why things are the way they are:** [`../DECISIONS.md`](../DECISIONS.md). It is long; search it for the topic.
4. **The current handoff for your arm** (table below).
5. **How the code is laid out:** [`ARCHITECTURE.md`](ARCHITECTURE.md), then [`../scripts/INDEX.md`](../scripts/INDEX.md)
   for the 415 scripts outside `archive/`.

Two cohorts exist. **Cohort 1** = PS92-PS95. **Cohort 2** = PS118, PS120, PS121, PS122, imaged from 2026-10-29. They
are analysed separately and never pooled. Read [`NEW_COHORT_2026-10-29.md`](NEW_COHORT_2026-10-29.md) before running
anything: hand runs on cohort 2 need `WIDEFIELD_COHORT=cohort2`.

---

## 1. One current handoff per arm

A "handoff" is the note a working session leaves for the next one: state, what is running, to-dos, pitfalls. Each
one names the older handoff it supersedes. The full list of status docs is in
[`status/README.md`](status/README.md); CLAUDE.md "Current status / handoff" keeps the older ones with a line each.

| arm | newest handoff on main | read with it |
|---|---|---|
| **Code (engineering, refactors)** | [`status/CODE_REVIEW_2026-11-04.md`](status/CODE_REVIEW_2026-11-04.md): verified findings, what to consolidate, the proposed order of work | [`status/REPO_CONSOLIDATION_PLAN_2026-11-03.md`](status/REPO_CONSOLIDATION_PLAN_2026-11-03.md), [`status/REPO_AUDIT_2026-11-03.md`](status/REPO_AUDIT_2026-11-03.md); older inventory [`status/STATUS_2026-09-21_ENGINEERING.md`](status/STATUS_2026-09-21_ENGINEERING.md) |
| **Trial gates, imaging section 1 table, pose / 3-D (incl. side cameras), spout position** | [`status/STATUS_2026-11-03_TRIAL_GATES_S1_TABLE_HANDOFF.md`](status/STATUS_2026-11-03_TRIAL_GATES_S1_TABLE_HANDOFF.md) (read its "START HERE (window e0, 11-04 late)" block first) | [`POSE_MODELS.md`](POSE_MODELS.md) (Part E = side cameras and four-camera 3-D); [`status/SPOUT_POSITION_AUDIT_2026-11-04.md`](status/SPOUT_POSITION_AUDIT_2026-11-04.md) |
| **Quiet-period cortical events, PI deck** | [`status/STATUS_2026-11-01_QUIET_EVENTS_HANDOFF.md`](status/STATUS_2026-11-01_QUIET_EVENTS_HANDOFF.md) (read its "11-04 (evening)" block first) | [`status/RUNNING_LIST_2026-11-03_AWAY.md`](status/RUNNING_LIST_2026-11-03_AWAY.md), [`status/PAGE_EDITS_PENDING_2026-11-04.md`](status/PAGE_EDITS_PENDING_2026-11-04.md) |
| **Kinematics page (behaviour), repo audit** | [`status/STATUS_2026-11-03_KINEMATICS_PAGE_REPO_AUDIT_HANDOFF.md`](status/STATUS_2026-11-03_KINEMATICS_PAGE_REPO_AUDIT_HANDOFF.md) | DECISIONS 2026-11-03 (trial classes) |
| **Encoding model** | [`status/STATUS_2026-10-26_ENCODING_HANDOFF.md`](status/STATUS_2026-10-26_ENCODING_HANDOFF.md) (section "00. START HERE") | [`ENCODING_FINDINGS.md`](ENCODING_FINDINGS.md). The cohort 1 rebuild (from 11-03) keeps its state on the `rebuild/*` branches (for example `rebuild/cohort1-20261103`), not yet on main. |
| **FaceRhythm** | [`status/STATUS_2026-10-24_FACE_RHYTHM_HANDOFF.md`](status/STATUS_2026-10-24_FACE_RHYTHM_HANDOFF.md) | [`FACE_RHYTHM_FINDINGS.md`](FACE_RHYTHM_FINDINGS.md), [`FACE_RHYTHM_PIPELINE.md`](FACE_RHYTHM_PIPELINE.md) |
| **Pose tracking / cam1 / jaw calls (before the 11-03 work)** | [`status/STATUS_2026-10-21_POSE_CAM1_JAW_HANDOFF.md`](status/STATUS_2026-10-21_POSE_CAM1_JAW_HANDOFF.md) | [`POSE_MODELS.md`](POSE_MODELS.md) |
| **Cohort 2 (PS118-122)** | [`NEW_COHORT_2026-10-29.md`](NEW_COHORT_2026-10-29.md) | [`COHORT2_QC_2026-10-29.md`](COHORT2_QC_2026-10-29.md) |
| **Chronic stability (cohort 1 imaging complete)** | [`status/STATUS_2026-10-07_CHRONIC_STABILITY_COHORT_COMPLETE.md`](status/STATUS_2026-10-07_CHRONIC_STABILITY_COHORT_COMPLETE.md) | DECISIONS 2026-10-07 |
| **ENL sensory vs plan** | [`status/STATUS_2026-09-24_ENL_SENSORY_VS_PLAN.md`](status/STATUS_2026-09-24_ENL_SENSORY_VS_PLAN.md) | |
| **CD trajectories** | [`status/STATUS_2026-09-24_CD_TRAJECTORIES.md`](status/STATUS_2026-09-24_CD_TRAJECTORIES.md) | |
| **Haemodynamics (closed) and engagement** | [`status/STATUS_2026-09-20.md`](status/STATUS_2026-09-20.md) | [`PREPROCESSING_DECISION.md`](PREPROCESSING_DECISION.md) |
| **Anatomical arm (where the code moves)** | [`status/STATUS_2026-09-12.md`](status/STATUS_2026-09-12.md) | [`WHERE_THE_CODE_MOVES.md`](WHERE_THE_CODE_MOVES.md) |
| **Rest baseline** | [`status/STATUS_2026-09-16.md`](status/STATUS_2026-09-16.md) | [`REST_ENGAGEMENT_AUDIT.md`](REST_ENGAGEMENT_AUDIT.md) (read before quoting any rest number) |

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

| | **Analysis desktop** (this box; profile `analysis_desktop`) | **Analysis / behaviour GPU box** (profile `analysis`) | **Imaging (PCO) box** (profile `imaging`) |
|---|---|---|---|
| python | `C:/Users/SabatiniLab/miniconda3/envs/locanmf/python.exe` | `C:/Users/sabatini/.conda/envs/locanmf/python.exe` | `C:\ProgramData\anaconda3\envs\wfield` (env `wfield`) |
| repo | `C:/Users/SabatiniLab/Github/widefield_pipeline` | `C:\Users\sabatini\GitHub\widefield_pipeline` | `C:\Github\widefield_pipeline` |
| MICROSCOPE share | `M:` | `M:` | `N:` |
| standby share | `N:` | — | `M:` |
| local data | — | `D:` (camera + behaviour-log staging) | `E:` (raw imaging + DAQ) |
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
