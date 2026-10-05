# Handoff — 2026-10-05: orofacial data in the widefield analyses (movement encoding, residual position coding, executed angle), O2 port

**START HERE.** Continues `STATUS_2026-10-02_SESSIONS_REACH_O2_HANDOFF.md` (session-level orofacial kinematics, lick
phase, direction shift, O2 bundle — still the reference for those) and `STATUS_2026-09-30_LP_OCCLUSION_ROUND4_HANDOFF.md`
(tracking models / labelling). Reasoning: `DECISIONS.md` entries 2026-10-02 (late) and 2026-10-05 (with addenda).
Design for everything imaging-side: `docs/DLC_IN_WIDEFIELD_ANALYSES.md`. Analysis desktop; **M: = MICROSCOPE, N: =
standby — paths from `PathResolver`**.

---

## 0. RESUME CHECKLIST

Nothing is running.

1. **Blocked on the next DLC / LP iteration** (labelling round 4: Priya's 131 cam4 frames, 0 done as of 10-02; cam1:
   student, 272 / 376 targets, ~4 folders left; 421 matched cam1 frames staged in
   `DeepLabCut/Widefield/_frame_staging_unassigned/` for the next cam1 batch). Then retrain DLC + LP
   (`lp_labels clean` for the LP export). **O2 is on hold until then (Priya)** — the bundle mechanism is ready
   (`wfield_local/o2_inference.py`; first bundle `o2/r3_PS93_20261002` uses round 3 and should be REGENERATED with the
   new model; `bundle --refresh` rewrites text only; never delete on MICROSCOPE).
2. **After whole-session predictions exist** (`o2_inference collect` → `session_poses/<a>_<d>_full/`), re-run, with
   NO changes to the pipelines (they are pre-registered now):
   * `scripts/movement_encoding_session.py` (movement encoder, variance partition),
   * `scripts/movement_position_angle.py` (residual position encoding + onset kernels by target vs angle bin),
   * `scripts/lick_template_match.py --pre <several pre :full> --post <post :full> --cued all` (the calibrated
     READOUT TEST is the inference; baseline pre_cue fixed in advance),
   * plus the orofacial summaries (`session_lick_summary`, `tongue_angle_phase`, `kinematics_checks`).
3. **Open decisions for Priya** (`docs/DLC_IN_WIDEFIELD_ANALYSES.md` §6): imaging-path DAQ lick detector (hard-coded
   1.0 V / 0.10 s vs config 0.5 V / 40 ms — DLC side keeps the config, decided); primary pose model (DLC / LP /
   consensus); which lick event (onset / peak / contact); engagement rule (jaw-only? tip-at-lips?); direction now on
   cam4 or wait for cam1 / 3-D.
4. Still pending from before: `python -m scripts.chronic_stability` once 0928 stage 2 has landed; PathResolver
   misdetects this desktop (local roots → `C:/Users/sabatini`; Priya's call).

## 1. What was done since 10-02

**Orofacial side** (details in the 10-02 handoff §0b): round-4 scan sessions converted (`session_poses from-scan` →
`<a>_<d>_r4`), PS93 five-session course; centered lick phase with tongue-in = lip level and a 20 % coverage rule for
direction; direction-shift figure (`lick_direction_shift_PS93.png`); per-lick deviation from the successful path
(`lick_reference`).

**Imaging side (new):**
| piece | what |
|---|---|
| `wfield_local/movement_encoding.py` | design matrix (event FIR kernels, optional per-event modulator; continuous signals binned to imaging frames, z-scored with TRAINING stats, lags), grouped ridge (penalty per group via column scaling, CV coordinate search), trial-block CV folds, CV R², variance partition by group or combined partitions, frozen models (`fit` / `predict`), `residual`, CROSS-FITTED `cv_residual`, `select_rows` |
| `wfield_local/movement_inputs.py` | `cam_frames_to_daq_s` (inverse of `dlc_frames.frame_of`), `pose_signals` (protrusion with tongue-in = lip level, speed, LR; jaw y / speed, unknown = NaN), `build_inputs` (cue kernel PER POSITION, contact / tongue onset / jaw / reward events, `tongue_onset_x_deviation` + `_x_angle` modulators for reach ≥ 60 px, continuous tongue / jaw / state) |
| `wfield_local/lick_templates.py` | Allen-region signals (common across sessions), lick patterns (pre_cue / pre_onset baseline), position templates, own-template accuracy, `compare` (descriptive executed-minus-target), **`angle_bin_templates` → `readout_angle` → `within_position_angle_test` (the calibrated inference)**, `paired_stats`, `perm_p`, `bh` |
| scripts | `movement_encoding_session.py` (`imaging`, `session_pieces` shared loaders), `movement_position_angle.py`, `lick_template_match.py`, `session_lick_summary.py` (+ direction / phase-mode figures) |
| config | `movement_encoding` (lags, groups, partitions task / movement / direction / state, `alpha_grid` up to 1e7, `min_reach_px`), `lick_template_match` (window, baseline pre_cue, tolerance, n_perm, n_angle_bins) |
| O2 | `o2_inference.py` + `o2_pose_runner.py` (standalone runner == local predictor + prior, tested), env `deeplabcut` (Priya's existing), user ps150 |

## 2. Findings (PS93, 60-trial pose windows = 11-13 % of each session; one animal — hypotheses)

* **Movement dominates** cue / lick-period dF/F: unique movement variance SSp 0.19-0.39, MOs / MOp 0.04-0.27 (CV);
  target (cue per position) ≈ 0 unique in encoding R² — position-specific variance is a tiny share; direction
  modulators ~0.01-0.02 pre, ~0 post.
* **Residual position encoding** (movement regressed out, cross-fitted; shared vs per-position cue kernel, label-shuffle
  null) follows the frozen decoder's course: pre +0.001-0.004 in every area → **acute ≈ 0 in MOs / MOp** → chronic
  recovers. Pre-stroke residual SSp traces separate far_R / far_L BEFORE the cue (spout-arrival signal).
* **Lick kernels:** per-target-position kernels beat per-angle-bin kernels pre-stroke; angle beyond position ≈ null.
* **Executed angle (Priya's far_R question):** a first template-matching result (acute far_R licks executed toward the
  mouse's left resembling pre-stroke licks of the same angle, +0.13) held with ONE baseline only and was inconsistent
  across positions → retracted to "lead". The calibrated lick-level readout test finds **no evidence** that cortex
  follows the executed angle within cued position (pooled r −0.05 acute, −0.10 chronic; the chronic one, if anything
  anti-aligned — unexplained). Re-test on whole sessions with the pre-registered pipeline.

## 3. Decisions (2026-10-02 late → 10-05)

* DAQ lick detection for all DLC / orofacial analyses = `configs lick_detection` (2.5 / 0.5 V, lockout 1-20 ms, min
  ILI 40 ms — stroke_orofacial's rule + our floor). Imaging path's detector: open.
* O2 waits for the next DLC / LP iteration.
* Movement regressors = events AND continuous; direction only as a PER-LICK modulator (no continuous angle);
  movement groups partitioned TOGETHER (they are redundant for stereotyped licks).
* Position information is tested on the movement RESIDUAL (cross-fitted), encoding with shared-vs-specific cue kernels
  + shuffle null; decoders remain the more sensitive complement.
* Template-matching inference = lick-level within-position readout test; executed-minus-target descriptive only;
  baseline pre_cue fixed in advance.

## 4. Pitfalls (new; earlier handoffs' lists still apply)

1. **Template-level permutation nulls are not calibratable** when licks share a template (30 % / 15 % false positives
   on random data) — exchange LICKS, not templates; always check a new null on structure-free synthetic data.
2. **Results that depend on an analysis choice** (baseline window) are not findings — fix choices in advance, report
   all variants run, correct across positions / classes.
3. **Penalty grid edges:** real data hit 1e4 then 1e6 (the target group) — check `alphas` against the grid ends.
4. **LocaNMF components differ between sessions** — cross-session comparisons go through Allen-region signals; area
   labels > 0 = LEFT hemisphere (= ipsilesional, all animals L-lesioned).
5. **Pose windows cover 11-13 % of a session** — fit only on `mask` frames (`select_rows` after building the design on
   the full timeline, so lagged columns are right at window edges).
6. **Fast bouts:** a −0.2 .. 0 s pre-onset window sits inside the previous lick.
7. **Imaging and behaviour code detect DAQ licks differently** — state the definition at every DAQ ↔ DLC join.
8. **Shell heredocs** with nested quotes / backslash-n broke edits several times — write edit scripts to a file, or use
   the Edit tool; `sed` turns `\n` into real newlines.
9. Pushes take ~2.5-3 min (pre-push hook runs the full ~3000-test suite) — not a failure.
10. Never delete on MICROSCOPE (bundles: `--refresh` or a new name); commit only your own files.
