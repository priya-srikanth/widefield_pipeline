# Four-camera licks for every figure (2026-11-22)

Priya, 11-22: "yes we should use 4 camera licks for all figures (manuscript and otherwise) - switch 4B/E to B".
Branch `pilot/fourcam-licks-everywhere` (worktree `Github/wf_fourcam_licks`), off `pilot/side4cam-wholesession`.

## Why
- cam4 misses ~6 % of post-cue licks that the other cameras see (12 % at far contra, 8 % far ipsi; 9.3 % of close-centre
  FIRST licks). Miss rates are similar across epochs (pre 5.5 %, acute 6.1 %, subacute 5.1 %, chronic 6.4 %).
- 6.5 % of cam4 lick windows are fragments that miss the 3-D peak (median window 24 ms vs 84 ms, reach 3.48 vs 4.55 mm),
  more often in pre first licks (15.8 %) than post (5.9-8.1 %), which inflates the post-stroke first-lick reach increase
  (`docs/status/CAM4_LONG_LICK_2026-11-22.md`). The fused lick's span covers the whole protrusion in 97.8 % of them.

## What changed in code
- `session_licks.kin_name()` / `kin_dir()` / `kin_sfx()` (env `WIDEFIELD_KIN_LICKS` = fused | cam4): THE post-cue 3-D
  kinematics folder. Default under a fused lick set = `kinematics_fused_postcue`; `WIDEFIELD_KIN_LICKS=cam4` = the cam4
  windows (`kinematics`), kept for comparison. Writers whose output depends on the lick set add `kin_sfx()` ('_fused') to
  their output folder, so a fused run never overwrites a cam4 build.
- `scripts/pose3d_kin_postcue.py` (new): the post-cue view of `kinematics_fused/` -- a drop-in for `kinematics/`:
  context == post_cue, `lick_idx` renumbered 0.. within the trial (`lick_idx_fused` keeps the original), `on_ms` /
  `off_ms` = the fused span (the window the 3-D measures were taken over), sessions.csv lick medians recomputed on the
  post-cue licks, by_animal (rule 8). No 3-D recompute (2.8 min).
- **3-D support** (coordinator 11-22): 6.0 % of fused post-cue licks are seen by ONE camera (cam1 only 4.5 %: median 0
  3-D frames, reach 1.85 mm, DAQ contact 0.3 %, late in the trial; cam4 only 1.5 %). PRIMARY folder: rows KEPT (counts
  unchanged) but kinematics blanked unless `n_views >= 2` (`kin_ok`); `kinematics_fused_postcue_allviews` keeps the
  single-view kinematics for comparison. In practice only 3,197 single-view licks had any 3-D measure (most have no 3-D
  frames), so the two folders differ little (table below).
- A first post-cue lick without 3-D support keeps lick_idx 0 with NaN kinematics: the second lick is NOT promoted to
  "first" (Q1 11-21). So `n_first` can be lower than on cam4 at a spout where the fused train finds an earlier lick the
  3-D cannot measure (e.g. PS93 0814 far_R 81 -> 71).
- `c2_lick3d_session_summary.py`: default folder = `kin_name()`; `n_first` / `n_all` = licks WITH 3-D kinematics (the
  recipe's n and the panels' n >= 20 cut); new `n_first_rows` / `n_all_rows` = every lick.
- `c2_scalar_family_aligned.py`: `--lick3d` / `--reach-sessions` / `--reach` defaults follow `kin_name()` / `kin_sfx()`.
- `pose3d_phase` (fused path reads the post-cue table with its spans; cache kind `phase_<kin_name>`; output `phase_fused/`),
  `pose3d_cue_traces` ('licked' set and first-lick alignment from the fused post-cue train; output `cue_traces_fused/`),
  `kinematics_pooled_figs` (`figures/pooled_fused/`), `pose3d_spout_error` / `pose3d_summary_fig` / `pose3d_june_reference`
  (`*_fused` outputs).

## Inventory: readers of the cam4 lick windows
| reader | role | status |
|---|---|---|
| `c2_lick3d_session_summary` (Fig 4B / 4E) | manuscript | rewired; re-run |
| `pose3d_phase_mouth0` -> `pose3d_phase_curves` / `pose3d_phase_from_peak` / `c2_phase_sessions_aligned` (Fig 4G-I) | manuscript | rewired; re-run into `phase/vls_manuscript_20261122_fused/` |
| `kinematics_pooled_figs` (pose3d_change_pooled_contra -> `c2_scalar_family_aligned` reach_all) | manuscript | rewired; re-run into `figures/pooled_fused/` |
| `c2_scalar_family_aligned` (`--lick3d`, `--reach`, `--reach-sessions` defaults) | manuscript | rewired (defaults) |
| `pose3d_phase` (phase/curves: trajectory / phase supplement, pooled phase figures) | manuscript supplement + pages | rewired; re-run into `phase_fused/` |
| `pose3d_cue_traces` (licked set, first-lick alignment) | page / supplement | rewired; re-run into `cue_traces_fused/` |
| `combined_encoding.lick_angle_3d` / `_exec_3d` / behaviour cache key | neural (encoder 3-D tongue regressors) | rewired (lick_key join); takes effect at the next encoder fit -- the 11-22 encoder fit used the cam4 table |
| `c2_engagement_tim/s4_session_tables` (3-D first licks) | S22 / pages | rewired |
| `pose3d_spout_error`, `pose3d_summary_fig`, `pose3d_june_reference`, `pose3d_check_sheet`, `contact_direction_predictability`, `state_space_sim`, `within_session_decline` | pages / other | rewired (not re-run here) |
| `pose3d_jaw_crosscheck`, `state_space_shift_movement` | jaw_trials only (lick-set independent; the folder is copied) | rewired (no effect) |
| `frozen_encoder` lick-spout posterior (`FROZEN_LICKSPOUT`, ITI aim chain) | neural | **stays cam4**: joins the cam4 licks.csv on (trial_id, lick_idx); needs a lick_key join -- open item |
| `pose3d_vs_cam4`, `cam4_long_lick_qc`, `pose3d_phaseA_gate`, `prep_3d_inputs` | QC / comparison | stay cam4 on purpose |
| `rebuild_figs/fig_pose_validation_sessions` | rebuild (pinned) | untouched |
| cam4 2-D lick kinematics (`orofacial_cohort` cohort tables: protrusion_px, angle_change_pooled) | 2-D cam4 readouts | not a 3-D table; no four-camera equivalent (by construction single-camera) |
| already on the fused train before today (11-22 rewiring): trial classes, lp_licks, first-lick latency, restnl, quiet chain, revup, GLM-HMM, FaceRhythm lick rate, iti_licks / lick_correction / within_lick_correction / learned_compensation (whole-session `kinematics_fused`) | | no change |

## Results
Filled in below as the re-runs finish.
