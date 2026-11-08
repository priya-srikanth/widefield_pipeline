# June pre-stroke sessions: what exists, what is filtered out, what was never run (11-07, a8)

Scope: cohort1 PS92-95, curated June = 0606-0608 (12 sessions). 0601-0605 are registered but excluded
(`animals.yaml cross_session_exclude`). The August-only cut is `cohorts.yaml baseline_pre_from: "0806"`, read by
`config.early_pre`. This inventory was read-only; the paths are evidence.

Why June was dropped, and the 3-D re-evaluation: `docs/POSE_MODELS.md` D13. The 3-D geometry holds. The June offsets are
behavior (an earlier training stage), not head placement or the camera frame.

| family | June status | evidence | to fill |
|---|---|---|---|
| Raw video cam1-4 | DONE (all registered June) | `Behavior_cameras/Widefield/202606xx/`; templates `alignment_templates/cam*/` (none for PS94/95 0601: cam4 drops) | -- |
| Behavior (DAQ trials, licks, events npz) | DONE | `behavior_summary/events/<PS>/202606xx.npz` | -- |
| Imaging SVD / LocaNMF | DONE | `labcams/202606xx/.../motion_corrected/` | -- |
| Decoders | in curated set; primary renders `--pre-from 0801` -> June only in "all" sensitivity + June->August drift | `june_august_drift.py` | -- |
| Combined / frozen encoding | PARTIAL: 10-19 O2 run fit June; 10-24 full run + frozen encoder exclude it (`frozen_encoder.py:1952,2101`) | `o2/enc_v36_20261019/labels.txt` (119) vs `enc_v36_full_20261024` (107) | O2 encoding run on the 12 June labels (inputs exist) |
| cam4 pose LP r4 v36 + licks + cohort tables | DONE-BUT-EXCLUDED (June only in `*_allpre`, SUPERSEDED) | `LP_analysis/lp_r4_heatmap_20261013__clean_v36/sessions/` | not fixable in 2-D (perspective); use 3-D or scale-free measures |
| cam1 pose + cam1 licks + jaw onsets | DONE for curated June (+ PS92 0602-05, PS93 0605); not PS94/95 0601/0603-0605 (reason unverified) | `o2/lp_cam1_ctx_prod_20261019/out/`, `session_poses/cam1_licks/` | only if non-curated June wanted |
| Trial classes / working-trial gate | NOT DONE (session list drops June: `s1_trial_class_snapshot.py:83`, `working_trials.py:41`) | `cam1_licks/trial_classes*/sessions.csv` 0 June rows | local CPU re-run with June (all inputs exist) |
| Two-camera 3-D, kinematics_all | DONE-BUT-EXCLUDED (`pose3d_kinematics.py:270` `date >= 20260801`; `lick_path_softmax`, `lick_trajectories_3d` `early_pre`) | `session_poses/pose3d/<PS>_202606xx_*`, `kinematics_all/sessions.csv` | consumer change: June-reference output, shift-invariant measures only (D13) |
| Side cams cam2/3 pose, 4-cam 3-D | NOT DONE for any session (August too); June = training labels + PS95 0606 pilot (PnP 1.1 px) | POSE_MODELS Part E | side round-2 model -> whole-session O2 -> confirm June side PnP |
| FaceRhythm | PARTIAL, stops before ROIs: frames, registration fields (similarity fallback, landmark err 4-16 px), O2 trial windows exist; no ROIs (`face_rhythm_warp_rois.py:93` skips June), no O2 output, no readouts | `face_rhythm/cam4/{frames,register/fields}`, `o2/fr_cam4_20261021/trials/` | QC June ROI warp (decision), O2 `--include-june` (12 tasks), project on frozen `master_pre_v1`; normalize for June 1.17-1.24x face magnification |
| Video motion energy | NOT DONE (`o2_video_me.py:89` drops June). NOT NEEDED: ME is not in any current primary result (only the opt-in `COMBINED_ENCODING_VIDEO` variant from the 23-session 10-18 pilot, the early-October ENL movement-removed lower bound, and the 10-29 rev-up jaw follow-up) | no June `video_motion_cam*.npz` | -- (only if one of those is revived) |
| ITI licks | DONE as a separate June reference (`iti_june_ref.py`) | `iti_licks/june_ref/sessions/` | -- |

Cheapest first:
1. 3-D June reference: no compute.
2. Trial classes for June: local CPU.
3. FR June ROI decision, then local warp.
4. FR O2 run with June: 12 tasks.
5. Optional: a June encoding O2 run, cam1 for non-curated June (motion energy not needed).
6. Side-camera chain: shared with August.

Unverified: the cam1 O2 gap for PS94/95 0601-0605; where the ENL movement-multisession outputs live (and whether they
include June); whether the 10-19 June encoding fits survive on disk.
