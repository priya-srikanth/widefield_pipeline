# Refreshing the VLS manuscript's six-position (PS92-95) numbers from a new 3D pose run

Written 2026-11-21 (Priya: the 4-camera / 3-network 3D kinematics will be run soon; every six-position number in the
manuscript must regenerate from code in this repo, and the manuscript only plots). Companion record:
DECISIONS 2026-11-21 "Six-position manuscript analyses: last paper-only computations ported; refresh runbook".
The manuscript side (paper folder `stroke_orofacial_pipeline_outputs/vls_stroke_paper/`) keeps the figure scripts and a
provenance registry (`FIGURE_PROVENANCE_v18.md`, refresh map section 1) that maps each panel to the tables below.

## Ground rules for the refresh
- **New names only.** Every step below refuses to overwrite. Write each regenerated table to a NEW run directory, run
  tag, or `--out-dir` / `--out-suffix`, compare with the current manuscript copy, then point the manuscript at it.
- **Adopt the run explicitly.** A new pose run gets a new run directory (e.g. `lp_<new>__clean_vNN`) and is adopted by
  `wfield_local/orofacial_cohort.py: RUN` (currently `lp_r4_heatmap_20261013__clean_v36`) in its own commit with a
  DECISIONS entry. Scripts that read the run through `run_mode` / `orofacial_cohort.RUN` follow it; scripts with
  literal `M:/.../clean_v36/...` defaults (the `c2_*` manuscript scripts) must be given the new paths with their
  `--enc` / `--aim` / `--revup` / `--classes` / `--curves` / `--licks` flags.
- **Baseline.** The six-position pre-stroke reference is the August sessions (`config.early_pre` drops June); epoch
  boundaries come from `configs/animals.yaml` and the chronic-onset rule. Neither changes with the pose run.
- **Environment.** The analysis desktop (`locanmf` env) is the reference; on the Mac run with `PYTHONPATH=<checkout>`
  and record the env in DECISIONS (pandas versions differ in float parsing; see DECISIONS 2026-11-20).

## Ordered steps
1. **3D pose.** `python -m scripts.pose3d production [ANIMAL:YYYYMMDD ...] --four-cam <side bundle>` (cam2 / cam3 via
   `pose3d_side.fuse_session`) into the new run's `session_poses/pose3d/`; check with `scripts/pose3d_check_sheet.py`,
   `pose3d_crosscheck.py`, `pose3d_side_check.py` before going on.
2. **Lick kinematics.** `python -m scripts.pose3d_kinematics` -> `pose3d/kinematics/licks.csv.gz`, `sessions.csv`;
   `python -m scripts.pose3d_phase` -> `pose3d/phase/curves.csv.gz`.
3. **Lick-phase curves (Fig 4G-H; trajectory supplement).**
   `python scripts/pose3d_phase_mouth0.py --out <new>/curves_mouth0_first_peak.csv.gz --jobs N`;
   `python scripts/pose3d_phase_curves.py first side0 down0 dist mid0 --across sessions --curves <that file> --out-suffix _NEW`;
   `python scripts/pose3d_phase_from_peak.py --sig aligned --curves <that file> --out-dir <new>`;
   `python scripts/c2_phase_sessions_aligned.py <each _sessions.csv> <curves> --kind peak`.
4. **Lick-level and trial tables.** `python scripts/iti_licks.py --jobs 3` (aim_by_spout / aim_first_vs_all sessions and
   tests); `python scripts/revup_behaviour.py --jobs 6` (first-lick delay cells, `revup/revup_track_sessions.csv`);
   `python -m scripts.jaw_onsets --jobs 3 --all` and the trial classes (`wfield_local/trial_classes.py` ->
   `qc/s1_trial_classes_cue_stop.csv.gz`).
5. **Trial outcomes and weights (Fig 3D, 3F).** `python scripts/c2_outcome_stats_aligned.py --out-dir <new>`;
   `python scripts/c2_outcome_weight_epoch_stats.py --out-dir <new>`; `python scripts/c2_outcome_distribution_test.py
   --out-dir <new>` (weights are pose-independent; rerun only to keep one folder).
6. **Scalar family (stars of Fig 3B, 4A-E and the scalar supplements).** `python scripts/c2_scalar_family_aligned.py
   --out-dir <new>` (single BH family; Priya 2026-11-19).
7. **Recovery mechanics (supplement "How lick aim recovers", six-position).** `python scripts/within_lick_correction.py
   run --jobs 3` then `summarize`; `python scripts/learned_compensation.py run` then `summarize`;
   `python scripts/lick_correction.py --extract` then `python scripts/lick_correction.py`; then
   `python scripts/c2_recovery_mechanics_stats.py --enc <new imaging_encoding> --family <new c2_scalar_family_aligned.csv>
   --out-dir <new>` (pooled family with the scalar cells; controls outside the family).
8. **Angle correlations (Fig 4F; angle-latency supplement).** `python scripts/c2_manuscript_correlations.py --aim
   <new iti_licks>/aim_by_spout_sessions.csv --revup <new iti_licks>/revup/revup_track_sessions.csv --out-dir <new>`
   (behavior hit rates come from `Behavior_logs/.../cohort_session_metrics.csv`, pose-independent).
9. **Engagement (only if trial classes or keypoint networks change).** `scripts/c2_engagement_tim/s0`-`s6` (GLM-HMM K5
   refit, TIM new windows, session tables, stats); then the manuscript's two-cohort combination
   (stroke_orofacial_pipeline `scripts/manuscript/engagement_movement/`).
Not affected by the pose run: Fig 3B (DAQ spout contact), FaceRhythm (optic flow), treadmill, histology, body weight.

## Then, on the manuscript side
Copy the new tables into `manuscript_code/data/` under NEW names, re-run the panel scripts listed in
`FIGURE_PROVENANCE_v18.md` (refresh map section 1; plot-only variants read the tables above:
`figures/c2_aimhit_plot.py`, `figures/c2_angle_latency_plot.py`, `figures/recovery_mechanics/c2_recovery_mechanics_r2.py`),
swap file names in `main.tex`, run `figures/check_provenance.py`, and re-check every in-text number listed in that
section (outcome fractions; rho -0.62 to -0.78 and "three of four animals"; ~0.9 s spout contact; 15 min / 75 trials
quit point; 2 mm path; five of six positions; Fig 4C / 4F legend numbers; recovery-mechanics paragraph and legend;
excursion paragraph; 62% jaw-only; Methods "Videography, pose estimation, and kinematics" -- two-view triangulation,
Lightning Pose version, decoding accuracy -- and the red pending marker in the aim Results paragraph).
