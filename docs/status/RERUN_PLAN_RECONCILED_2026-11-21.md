# Four-camera re-run: reconciled plan after three independent blast-radius reviews (2026-11-21, window de)

Priya 11-21: "please be careful to review everything that will need to be rerun, and perhaps have more than 1 agent check
it". Three reviewers worked independently, read-only, without seeing each other's lists or this window's:

| reviewer | angle | report (C:/Users/SabatiniLab/lp_stage/blast_radius/) |
|---|---|---|
| A | forward: from every reader of the changed inputs to the outputs | `review_A_forward.md` (39-row table) |
| B | backward: from every manuscript panel / number / Methods item (main_v18, FIGURE_PROVENANCE_v18) | `review_B_backward.md` |
| C | recent commits on all refs since 11-14 + line-by-line audit of both plans against the code | `review_C_history.md` (53 v36-literal files listed) |

Plans reviewed: `STATUS_2026-11-18_CAM23_RERUN_INVENTORY.md` (sections 2-4), `MANUSCRIPT_C2_REFRESH.md`, the Mac prompt
`vls_stroke_paper/PROMPT_c2_4cam_3d_refresh.md`.

## 1. Decisions needed from Priya (none of these is a re-run; each changes code or adoption)

1. **Which licks define the 3-D kinematics?** (A gap 3, C gap 2.) `pose3d_kinematics` measures inside the cam4 v36 lick
   windows (`pose3d_kinematics.py:52,173,249`); the new four-camera 3-D licks (camera split + 40 ms floor) reach only
   `--licks combined`, which nothing reads. Options: (a) keep cam4 lick windows (re-pointed to v37) and only the
   coordinates change; (b) the fused four-camera train (D2 / D3) defines the lick set -- needs a new mode AND a fix of the
   `lick_uid` join (below). D2 / D3 already make the fused train the lick definition for classes / gates / windows, so
   (b) is the consistent choice; it is the bigger change.
2. **Adopt the new run?** The runbook says flip `orofacial_cohort.RUN`; the Mac prompt forbids it without Priya's OK.
   Without the flip, scripts that follow RUN stay on v36; with it, 53 files with literal v36 paths still do not follow
   (C). Proposal: no flip tonight; every step gets explicit v37 / tagged paths; the flip is one commit after review.
3. **3-D lick floor 40 vs 60-80 ms** (running list).

## 2. Agreed by all three (fix before the step that needs it)

- **The fused four-camera lick train has no consumer.** `lp_licks._inputs` (lp_licks.py:459-465), `trial_classes`,
  `restnl_mask`, `quiet_spikes_v2`, `combined_encoding`, `revup_behaviour`, `glm_hmm`, FaceRhythm read cam4 `licks_all` +
  cam1 `_events` directly. D2 / D3 require rewiring each (one shared loader; `lp_licks.LP_VERSION` bump; cache
  signatures must include the fused folder).
- **Order.** `pose3d --four-cam` already reads cam4 `licks_all` and cam1 events (lick-split cuts, drop-mask keep windows),
  and cam1's `in_3d` reads the 3-D. Correct order: **v37 2-D chain (spout fix, CLEANING_VERSION bump, `all-licks`) ->
  cam1 re-detection (versioned folder, `mask_detached`) -> four-camera 3-D -> cam1 `in_3d` again -> fusion (full
  session) -> jaw_onsets -> trial classes / s1 snapshot / gate / RESTnl -> kinematics (first) -> neural.** The inventory
  ran the 3-D before v37 and has no cam1 step; `jaw_onsets` is missing from L2.
- **The 3-D source / tag env vars must be set for every step** (`WIDEFIELD_POSE3D_SOURCE=4cam`,
  `WIDEFIELD_POSE3D_TAG=_3net_20261121`); ~20 scripts hard-code `session_poses/pose3d` and ignore them (incl.
  `within_lick_correction`, `lick_correction`, `c2_scalar_family_aligned.py:59`, `s4_session_tables.py:327`, paper-side
  `c2_phase_*`). Without the env, writers overwrite the two-camera `kinematics/` and `phase/`. The runbook names the wrong
  folder (`pose3d/`).
- **Overwrite / stale-reuse hazards** (the runbook's "every step refuses to overwrite" is false): `jaw_onsets` (no
  `--out`), `s1_trial_class_snapshot` (always the current run's folder), cam1 events (unversioned shared folder),
  `pose3d_kinematics`, `trial_attempts`, `lick_correction`, recovery-mechanics scripts, `within_lick_correction` /
  `learned_compensation` (write into `encoding_out_root(V36)`), `glm_hmm --pooled-all`; skip-if-exists extracts in
  `iti_licks`, `revup_behaviour`, `within_lick_correction`, `learned_compensation`, `pose_chain`; GLM-HMM sequence cache
  keyed on session labels only.
- **`lick_uid` row-index join breaks silently** when the lick table changes (`kinematics_all` vs `oc.RUN` `licks_all` in
  `iti_licks`, `lick_correction`, `frozen_encoder`, `stray_licks`): must become a key join before any lick set changes.
- **Runbook command bugs:** `summarize` vs code's `summarise` (`within_lick_correction` errors; `learned_compensation`
  exits 0 doing nothing); `c2_phase_sessions_aligned --kind` does not exist; `_NEW` suffix breaks its filename parsing;
  steps 5 / 6 / 8 pass only `--out-dir` (inputs default to v36 / two-camera); step 7 without `--redo` reuses old pickles;
  `pose3d_kinematics --licks all` (-> `kinematics_all`) never run.
- **Wrongly listed as unaffected:** Fig 3B hit rate (camera working period, `epoch_behaviour.py:13-19,252`; stars from the
  162-cell family), the 15 min / 75 trials quit (camera quit, `disengagement_tests.py`), S19 treadmill rows (lick
  working period + ITI lick buffer), Fig 5B + S18 lick half (cam4 licks), FaceRhythm lick-direction readouts. Step 9
  (engagement S20, "62 % jaw-only") is REQUIRED (classes change).
- **No producer anywhere** (all refs + manuscript_code searched): `c2_lick3d_session_summary_ungated.csv` (Fig 4B, 4E),
  `aim_first_vs_all_*` (Fig 4D), `pose3d_change_pooled_contra_aligned.csv`. `aim_by_spout_*` comes from
  `lick_cued_aim_by_spout.py` + `iti_zscore_check` / `iti_first_lick`, not `iti_licks.py`.
- **Paper-side computations not in the repo:** `trial_class_distribution.py`, `c2_angle_latency.py` (current S14C),
  `movement_combined.py` / `tim_combined_px.py` (S20 stats), `make_farcontra_aim_hit.py`; ~80 stale two-camera / v36
  copies in `manuscript_code/data`; the live `manuscript_code` lacks `recovery_mechanics`, `engagement_movement`,
  `task_structure`, `infarct_power` (only in `manuscript_code_v18.zip` / the Mac).
- Missing from inventory section 4: the whole manuscript layer (`c2_*`, `c2_engagement_tim`, `revup_behaviour`,
  `learned_compensation`, `pose3d_phase_*`, treadmill, FR lick-aligned), `prep_3d_inputs` / `exec_bins_calibration`
  snapshots (`frozen_encoder` defaults to a 10-23 two-camera calibration), `stray_licks`, goal-page rebuild_figs,
  imaging-page s1 table, `deck_keyfigs` / `deck_registry` literals, ~20 `iti_*` / `lick_*` / `revup_*` scripts, RESTnl
  caption text, `iti_dock_audit` (S16), `kinematics_pooled_figs --contra` (S15A), `orofacial_cohort` under v37 (S14A-B).
- Not affected (consensus): FaceRhythm factors (optic flow), weights, DAQ hit rates as such, histology, cohort 1 / VFSS /
  S23, `state_space_sim` method check, `nose_onsets` (feeds no class).

## 3. What runs tonight (no adoption, new names only) and where it stops

Phase A of the Mac prompt, in the corrected order, with explicit paths:
1. v37 2-D chain under a new RUN folder (D1 = a) -- new folder only.
2. cam1 re-detection into a versioned folder (never the shared `session_poses/cam1_licks`).
3. Four-camera 3-D, tag `_3net_20261121` (per-session side gate replaces the June skip).
4. cam1 `in_3d` + fusion (full session) into tagged folders.
5. Phase A gates: coverage / reprojection vs two-view v36; per-lick agreement on licks both detect (r, mean offset per
   animal); six-way spout decoding from the 3-D endpoint; submovement noise check (needs `within_lick_correction` to read
   the tagged 3-D and write to a new folder).
**Stops before Phase B** (the trial-class / kinematics / manuscript tables) until decisions 1-2 are made and the
consumer rewiring + `lick_uid` key join are reviewed.
