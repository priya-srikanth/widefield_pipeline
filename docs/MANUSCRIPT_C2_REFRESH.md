# Refreshing the VLS manuscript's six-position (PS92-95) numbers from a new 3D pose run

Written 2026-11-21 (Priya: the 4-camera / 3-network 3D kinematics will be run soon; every six-position number in the
manuscript must regenerate from code in this repo, and the manuscript only plots). Companion record:
DECISIONS 2026-11-21 "Six-position manuscript analyses: last paper-only computations ported; refresh runbook".
**Rewritten 2026-11-22** after the three blast-radius reviews (`docs/status/RERUN_PLAN_RECONCILED_2026-11-21.md`) and the
consumer rewiring (DECISIONS 2026-11-22 "Four-camera re-run: consumer rewiring before Phase B"): the 11-21 steps named
the wrong 3-D folder, had command bugs (`summarize`, `--kind`, `_NEW`), and let steps 5 / 6 / 8 read v36 / two-camera
inputs. The manuscript side (paper folder `stroke_orofacial_pipeline_outputs/vls_stroke_paper/`) keeps the figure scripts
and a provenance registry (`FIGURE_PROVENANCE_v18.md`, refresh map section 1) that maps each panel to the tables below.

## Ground rules for the refresh
- **New names only.** Writers that used to overwrite now REFUSE an existing output (`--force` to replace on purpose):
  `jaw_onsets`, `s1_trial_class_snapshot`, `trial_attempts`, `pose3d_kinematics`, `glm_hmm --pooled-all`, `cam1_licks`
  into the default cam1 folder. The `c2_*` scripts refuse as before. Skip-if-exists caches now check their inputs
  (iti_licks / lick_correction `lick_sig`, revup, within_lick_correction, learned_compensation, pose_chain, GLM-HMM).
- **Adoption = one commit.** The four-camera / v37 run is the new cohort default (Priya 11-21 Q2) through ONE labelled
  commit that sets `orofacial_cohort.RUN` (v37), `pose3d.ADOPTED_SOURCE` / `ADOPTED_TAG` (4cam, `_3net_20261121`),
  `lp_licks.CAM1_DEFAULT` (cam1_licks_v37) and `session_licks.FUSED_DEFAULT` (the Phase B fusion, `..._gapfix60`). Run
  Phase B on a checkout WITH that commit; without it, export the equivalent env for every step:
  `WIDEFIELD_LICK_INPUT_RUN=lp_r4_heatmap_20261013__clean_v37 WIDEFIELD_CAM1_LICKS_DIR=cam1_licks_v37
  WIDEFIELD_POSE3D_SOURCE=4cam WIDEFIELD_POSE3D_TAG=_3net_20261121
  WIDEFIELD_FUSED_LICKS=lp_side_r2b_prod_20261118_3net_20261121_gapfix60` -- but then writers that follow
  `orofacial_cohort.RUN` still write into the v36 tree, which is why the flip comes first.
- **If the whole-session (sync-span) camera re-run is adopted instead** (Q8; names carry `_sync`), change the four
  constants of the adoption commit to those names (v37_sync run, its 3-D tag, its cam1 folder, its fusion) before Phase B.
- **The lick set is the fused four-camera train** (Q1 = b) wherever a consumer was rewired (`session_licks`); the
  encoder stays cam4 unless `COMBINED_ENCODING_LICKS=fused`. Not rewired (still cam4 windows): see the DECISIONS entry.
- **Baseline.** The six-position pre-stroke reference is the August sessions (`config.early_pre` drops June); epoch
  boundaries come from `configs/animals.yaml` and the chronic-onset rule. Neither changes with the pose run.
- **Environment.** Every step: `MKL_THREADING_LAYER=SEQUENTIAL PYTHONPATH=<this checkout>` (check
  `python -c "import wfield_local; print(wfield_local.__file__)"` points into the checkout; a worktree otherwise imports the
  main checkout). Env `locanmf` unless noted `dlc`.

## Phase A (done 11-21)
v37 2-D chain; cam1 re-detection into `cam1_licks_v37`; four-camera 3-D `session_poses/pose3d_4cam_3net_20261121`
(`scripts.pose3d production --all --four-cam lp_side_r2b_prod_20261118`, dlc env); cam1 `in_3d` pass 2; fusion; Phase A
gate (`docs/status/RUNNING_LIST_2026-11-21.md`).

## Phase B, in order (kinematics first, then neural)
`<S>` = the 126 sessions of `camera_licks/lp_side_r2b_prod_20261118_3net_20261121_gapfix/views.csv` as ANIMAL:YYYYMMDD.
1. **Re-fuse at full precision with the 60 ms floor** (dlc env; the 11-21 fusion was written at 4 significant digits and
   is refused by `session_licks`): `python -m scripts.camera_licks session <S> --bundle lp_side_r2b_prod_20261118 --saved
   --gapfix --out-tag _gapfix60 --jobs 6` -> `camera_licks/lp_side_r2b_prod_20261118_3net_20261121_gapfix60/` (+
   `fused_manifest.json`). Check the reasons / views tables against `_gapfix`.
2. **Jaw calls (source-aware close spouts, Q5):** `python -m scripts.jaw_onsets --all --jobs 3` -> the v37 tree's
   `imaging_encoding/qc/jaw_onsets.csv` (new file; `jaw3d_source` = 4cam). Blind-check a sample of close_center jaw-only
   trials before quoting them (the 3-D onset floor is provisional).
3. **Trial classes:** `python scripts/trial_attempts.py --jobs 2`; `python scripts/s1_trial_class_snapshot.py --window
   cue_stop --jobs 8`; `python scripts/camera_coverage_tables.py` (the camera-gate coverage note).
4. **3-D kinematics on the fused train (Q1 = b):** `python -m scripts.pose3d_kinematics --licks fused --all --jobs 3` ->
   `pose3d_4cam_3net_20261121/kinematics_fused/` (licks.csv.gz with lick_key, jaw_trials.csv.gz, sessions.csv,
   by_animal.csv, frame_coverage.csv, licks_gated.csv.gz). Comparison / not-yet-rewired readers: `--licks cam4` ->
   `kinematics/`, `--licks all` -> `kinematics_all/` (cam4 v37 windows on the four-camera 3-D).
5. **Lick-phase curves (Fig 4G-H; trajectory supplement)** -- these read `kinematics/` (cam4 windows; not rewired):
   `python -m scripts.pose3d_phase --jobs 3`; `python scripts/pose3d_phase_mouth0.py --out <new>/curves_mouth0_first_peak.csv.gz
   --jobs N`; `python scripts/pose3d_phase_curves.py first side0 down0 dist mid0 --across sessions --curves <that file>
   --out-dir <new>`; `python scripts/pose3d_phase_from_peak.py --sig aligned --curves <that file> --out-dir <new>`;
   `python scripts/c2_phase_sessions_aligned.py <each _sessions[_suffix].csv> <curves> [--kind peak]`.
6. **Behavior tables:** `python scripts/revup_behaviour.py --jobs 6`; `python scripts/within_lick_correction.py run --jobs 3`
   then `python scripts/within_lick_correction.py summarise`; `python scripts/learned_compensation.py run` then
   `python scripts/learned_compensation.py summarise`; `python scripts/lick_correction.py --extract` (imaging) and
   `python scripts/lick_correction.py --behaviour-only`. Review the CSM measure first (gate 4; csmfix 11-21).
7. **GLM-HMM / engagement:** `GLMHMM_OUT=<v37 tree>/glm_hmm/v1 python -m wfield_local.glm_hmm --pooled-all --k 5 --design v2`
   (a refit: new folder); then `scripts/c2_engagement_tim/s0`-`s6` (REQUIRED: classes changed; 62 % jaw-only).
8. **Manuscript statistics, every input explicit** (`<E>` = `run_mode.encoding_out_root()` of the adopted run):
   `python scripts/c2_outcome_stats_aligned.py --classes <E>/qc/s1_trial_classes_cue_stop.csv.gz --out-dir <new>`;
   `python scripts/c2_outcome_weight_epoch_stats.py --classes <same> --out-dir <new>`;
   `python scripts/c2_outcome_distribution_test.py --classes <same> --out-dir <new>`;
   `python scripts/c2_scalar_family_aligned.py --aim <E>/iti_licks/aim_by_spout_sessions.csv --reach <new pooled table>
   --reach-sessions <new> --hit <new hit table> --stats <v37 cohort prepost stats> --out-dir <new>`;
   `python scripts/c2_recovery_mechanics_stats.py --enc <E> --family <new c2_scalar_family_aligned.csv> --out-dir <new>`;
   `python scripts/c2_manuscript_correlations.py --aim <E>/iti_licks/aim_by_spout_sessions.csv --revup
   <E>/iti_licks/revup/revup_track_sessions.csv --out-dir <new>`.
9. **Neural (after the kinematics review):** `python scripts/iti_licks.py --jobs 3` (+ `iti_zscore_check`, `iti_first_lick`,
   `lick_cued_aim_by_spout` for aim_by_spout), `python -m wfield_local.restnl_mask --jobs 4` (lick set `fused` by default:
   RESTnlf), the quiet chain (`scripts/cohort2_qc/run_quiet_chain.sh`, `QS_LICK_SOURCE=daq+lp`), FaceRhythm lick-rate
   readouts.
Still without a producer anywhere (reviews A-C): `c2_lick3d_session_summary_ungated.csv` (Fig 4B, 4E),
`aim_first_vs_all_*` (Fig 4D), `pose3d_change_pooled_contra_aligned.csv`.

## Then, on the manuscript side
Copy the new tables into `manuscript_code/data/` under NEW names, re-run the panel scripts listed in
`FIGURE_PROVENANCE_v18.md` (refresh map section 1; plot-only variants read the tables above:
`figures/c2_aimhit_plot.py`, `figures/c2_angle_latency_plot.py`, `figures/recovery_mechanics/c2_recovery_mechanics_r2.py`),
swap file names in `main.tex`, run `figures/check_provenance.py`, and re-check every in-text number listed in that
section (outcome fractions; rho -0.62 to -0.78 and "three of four animals"; ~0.9 s spout contact; 15 min / 75 trials
quit point; 2 mm path; five of six positions; Fig 4C / 4F legend numbers; recovery-mechanics paragraph and legend;
excursion paragraph; 62% jaw-only; Methods "Videography, pose estimation, and kinematics" -- two-view triangulation,
Lightning Pose version, decoding accuracy -- and the red pending marker in the aim Results paragraph). Affected, though
the 11-21 runbook listed them as not: Fig 3B hit rate (camera working period), the 15 min / 75-trial quit point, S19
treadmill rows, Fig 5B + S18 lick half, FaceRhythm lick-direction readouts.
