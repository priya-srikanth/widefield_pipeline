# Spout position: ENL no-lick medians in every kinematics analysis? Audit + impact (2026-11-04, window e0)

**DECIDED (Priya 11-04): fix with the next 3-D re-run, when cam2 / cam3 come in** (round-2 side model + whole-session
side inference). Until then the cohort-1 v36 / 3-D outputs stay as they are (the deviations below are <= 0.03 mm at the
session level and unbiased per trial). The fix list for that run: `orofacial_outputs.py:184` -> `SF.ruler_medians`;
`tongue_kinematics.trial_spout_reference` -> the session ENL medians (or a per-trial ENL median); the whole-window
scripts (section below, item 3); `prep_3d_inputs.spout_table` -> the commanded table (`scripts/pose3d_spout_commanded.py`);
re-run production `pose3d` with the per-frame spout fix (bddf02e7) and the four-camera pipeline (`pose3d_side`).

**Rule (Priya 11-04):** each session's spout position (per spout position) = the median spout keypoint over in-trial
NO-LICK ENL frames (`spout_frame.ruler_medians`: after the spout settles, before the cue, tongue not detected, spout
confident); in 3-D, the commanded positions (which those medians match). "Ensure this is true for all our kinematics
analyses."

## What uses what (read-only audit of wfield_local/ + scripts/, archive and tests skipped)

OK (ENL no-lick medians or commanded):
* 3-D: `pose3d.session_3d` poses cam1 / cam4 from `ruler_medians` (`pnp_group`), cleaning mouths from them, lick aim and
  kinematics against `spout_world.commanded_world`; `pose3d_side` (four cameras); `apparatus_mouth`; the pose3d check
  scripts. ENL medians triangulated with cam1's edge offset match the commanded tips to median 0.05 / p90 0.07 / max
  0.12 mm over all 126 production sessions (`summary_production.csv` fit_resid_mm_max).
* cam1 licks (`cam1_licks._cam1_frame`: ruler medians); encoder TARGET3D now `pose3d_spout_commanded.py` (main ce86f76d).

NOT the rule:
1. **v36 cam4 session spout frame** `orofacial_outputs.py:184`: `SF.from_medians(SF.position_medians(...))` over WHOLE
   trial windows (every confident frame, licks and spout moves included). Feeds every v36 table's mouth frame (tongue
   zero, spout_origin / spout_ap_axis): licks / trials / jaw / precue / ITI, heatmaps, phase portraits, jaw lateral, the
   encoder's tongue analogs.
2. **Per-trial spout reference** `tongue_kinematics.trial_spout_reference`: median spout cue .. trial stop (the licking
   period) -> licks.csv `spout_angle_deg`, `spout_dist_px`, `tip_to_spout_px`, `reach_frac`, `delta_angle_deg` -> cohort
   `angle_gain_far`, per-position spout angle / distance, encoder `cam4_angle_reach` target geometry.
3. ~17 scripts with the same whole-window pattern (QC / review / demo mostly; analyses: jaw_lateral,
   movement_encoding_session, session_lick_summary, cam1_lick_aim `from_medians_lr`), `whisker_motion` (per-trial
   whole-window median, no likelihood filter), `orofacial_heatmaps` `spout_ref` (cue .. stop histogram centre).
4. `scripts/prep_3d_inputs.py` `spout_table` still writes the all-frames 3-D spout table if re-run (the encoder no longer
   reads it: switch it to the commanded table).

## Impact (measured; scratchpad scripts `spout_frame_impact.py`, `trial_ref_impact.py`)

* (1) Session frame, whole-window vs ENL no-lick, 7 August sessions (PS92 0807 / 0818, PS93 0814 / 0821, PS94 0820,
  PS95 0817 / 0917): origin 0.25-0.97 px, ap axis 0.06-0.34 deg, per-position medians <= 1 px except PS95 0817 (2.3 /
  3.4 px at close_R / far_R). The stored v36 `spout_origin` equals the whole-window recompute exactly. At cam4 scale
  (~30 px / mm) that is <= 0.03 mm.
* (2) Per-trial reference vs the ENL session median, 5 sessions, 2865 trials: median 0.67-0.98 px, p90 1.6-3.9 px,
  max 4-27 px (single trials: spout pushed or mis-tracked while licking); spout distance from the mouth unbiased (median
  0.0 %), p90 1.1-3.2 % -> `reach_frac` / spout angle carry 1-3 % per-trial noise, no bias.

**Reading:** both violations are small and unbiased; the spout is parked almost the whole window, so lick frames barely
move the medians. No v36 regeneration is needed for the cohort-1 results. The code should still follow the rule
(future runs, cohort2, the outlier trials): `orofacial_outputs` -> `ruler_medians`, `trial_spout_reference` -> the
session ENL medians (or a per-trial ENL median), `prep_3d_inputs.spout_table` -> commanded. Priya decides when (it
changes the v36 params digest, so it belongs to the next regeneration, not a patch under existing outputs).
