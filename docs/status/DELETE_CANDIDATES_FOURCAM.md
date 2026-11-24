# Superseded derived outputs: delete candidates (four-camera re-run)

Living list (Priya 11-23: "keep track of superseded outputs that we can delete"). Add each superseded output here when
it is replaced. Nothing is deleted until Priya OKs the batch (memory rule: derived outputs only; originals, raw data,
other people's folders and the frozen-model store are never deleted). Tick the box and date it when deleted.

Roots: `E` = `M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v37_sync60/imaging_encoding`,
`P3D` = `M:/MICROSCOPE/Priya/DeepLabCut/Widefield/session_poses/pose3d_4cam_3net_20261121_sync60`,
`S` = `M:/MICROSCOPE/Priya/Widefield/manuscript_stats/c2_4cam_sync60_20261122`,
`G` = `M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v37_sync60/glm_hmm`,
`PAPER` = `M:/MICROSCOPE/Priya/stroke_orofacial_pipeline_outputs/vls_stroke_paper`.
`RB` = `M:/MICROSCOPE/Priya/Widefield/rebuild_20261123_fourcam/LP_analysis/lp_r4_heatmap_20261013__clean_v37_sync60/imaging_encoding` (four-camera imaging root).

| | path | superseded by | since |
|---|---|---|---|
| [ ] | `E/frozen_encoder_gate-attempted_v7g3_t3d_frA_licks-fused_20261122/` | fourcam encoder refit (not truly four-camera) | 11-22 |
| [ ] | `E/iti_licks_licks-fused_20261122/` | `E/iti_licks/` (fourcam) | 11-22 |
| [ ] | `E/iti_licks_smoke_fourcam_20261122/`, `E/iti_licks_smoke2_fourcam_20261122/` | smoke tests | 11-22 |
| [ ] | `E/qc_check_flicks/` | byte-identical copies of `qc/` | 11-22 |
| [ ] | `E/within_lick_correction/`, `E/learned_compensation/` (cam4-era) | `*_fourcam*` folders | 11-22 |
| [ ] | `E/within_lick_correction_fourcam_launchdisp/`, `_launch2vdisp/` | launch Variant A (`_launch2v`) chosen | 11-23 |
| [ ] | `E/aim_kinematics_v36check/` | validation run, done | 11-22 |
| [ ] | `G/v1_floor020_withflicks/` | `G/v1_floor020` (cam1-only flicks out) | 11-22 |
| [ ] | `P3D/**/*_pass1_cam1kept` | cam1-only rows dropped (2a159eb5) | 11-22 |
| [ ] | `P3D/kinematics_fused/licks_lickidx_across_contexts_20261122.csv.gz`, `licks_with_cam1only_20261122.csv.gz` | renumbered / cam1-only-dropped table | 11-22 |
| [ ] | `P3D/kinematics_fused/c2_lick3d_session_summary_ungated.csv` | wrong population (all contexts); post-cue table is in `kinematics_fused_postcue/` | 11-22 |
| [ ] | `P3D/clips_20261122/*_main.mp4` | re-encode test copies (if originals open) | 11-22 |
| [ ] | `P3D/_superseded_cam3drop_20261123/` | cam3-on for PS95 0818 / 0820 (keep until Priya has checked the cam3-on results) | 11-23 |
| [ ] | `P3D/qc_cam3_drop_20261122/no_cam3/`, `control/`, `cam3_on/b_pnp4/` | QC done; `cam3_on/a_forced` became production (keep `validation.csv`, clips) | 11-23 |
| [ ] | `M:/MICROSCOPE/Priya/Widefield/scratch_encoder_smoke_20261122/` | smoke test | 11-22 |
| [ ] | `S/recovery_mechanics_launchdisp/`, `S/recovery_mechanics_launch2vdisp/` | Variant A chosen | 11-23 |
| [ ] | `S/*` tables at 2,000 draws where a `_boot4000` / `_boot4000_cam3on` twin exists | 4,000 draws + cam3-on | 11-23 |
| [ ] | `S/scalar_family_boot4000_cam3excl/` | cam3-on (exclusion test superseded) | 11-23 |
| [ ] | `PAPER/_to_delete/c1_aim_acc_boot5000_2026-11-23/` | 4,000-draw S8F tables | 11-23 |
| [ ] | `C:/Users/SabatiniLab/wfield_cache` entries keyed on superseded inputs | rebuilt on demand | -- |
| [ ] | `<sync60>/cohort_pre_cam3on_20261123/` | copy of `<sync60>/cohort` before the 11-23 re-run (keep until the stars are checked) | 11-23 |
| [ ] | `face_rhythm/cam4/validation/master_pre_v1/` (+ `lick_epochs/`) | `master_pre_v1_fourcam_20261123/` (once the page points at it) | 11-23 |
| [ ] | `face_rhythm/cam4/readouts/master_pre_v1/enl_decode/` | `enl_decode_fourcam_20261123/` | 11-23 |
| [ ] | `face_rhythm/cam4/readouts/master_pre_v1_fourcam_20261123/`, `factors/master_pre_v1/labels_suggested_pre_fourcam_20261123.json` | numerically equal to the originals (redundant) | 11-23 |
| [ ] | `Widefield/within_session_decline/` (11-07, v36) | `Widefield/within_session_decline_fourcam_20261123/` | 11-23 |
| [ ] | `session_poses/cam1_licks/trial_classes/` (11-03) | `session_poses/cam1_licks_v37_sync60/trial_classes/` | 11-23 |
| [ ] | `labcams/grant_figures/lmm_hidden_20261103/` | `lmm_hidden_fourcam_20261123/` | 11-23 |
| [ ] | `E/iti_licks_cam3on/lick_correction/` behavior tables | byte-identical in `E/iti_licks_lickcorr_fourcam_20261123/lick_correction/` (+ neural) | 11-23 |
| [ ] | `E/within_lick_correction_fourcam_launch2v_cam3on/` | byte-identical summaries in `..._launch2v_neural_20261123/` (+ neural) | 11-23 |
| [ ] | `lp_stage/reanalysis_20261123/mc_scratch_treadmill/` | scratch MC for c2_treadmill_extras (its npz output is superseded by c2_s13_treadmill_from_fr) | 11-23 |

Keep (never delete): `frozen_models/` stores; raw videos / scans; LP predictions; `qc_cam3_drop_20261122/clips/`.
| [ ] | `C:/Users/SabatiniLab/.widefield_session_cache_restnlfv1/` (Phase B RESTnl, bare env, 11-22 06:34) | `rebuild_20261123_fourcam/session_cache_restnlfv1` (built after the class rebuild and the cam1-only rule) once D5 finishes | 11-23 |
| [ ] | `PAPER/ccB_caption_review_2026-11-23/thirds/S8E_daycourse_thirdsboot_ccB1123.*`, `S12F_daycourse_firstangle_thirdsboot_ccB1123.*`, `thirds/data_twotone/`, `thirds/data_sixpos/` | `_v2` versions (all 8 animals, any gap, within vs session-to-session) | 11-23 |
| [ ] | `PAPER/ccB_caption_review_2026-11-23/twotone_rerun/v29dag/*_v29dag.*` (not `_v29dag2`) | `_v29dag2` († offset, corrected #6 reason) | 11-23 |
| [ ] | `E/qc/*_precam3on.*` | cam3-on class rebuild | 11-23 |
| [ ] | `G/v1/` (sync60 run) | `G/v1_floor020` default | 11-23 |
| [ ] | `E/manuscript_cam3on_20261123/` (empty folder) | never-started refit | 11-23 |
| [ ] | `PAPER/ccB_caption_review_2026-11-23/thirds/S8E_daycourse_thirdsboot_v2_all8_ccB1123.*`, `..._yshared_grade_ccB1123.*` | `..._yshared_all_ccB1123` (Priya: one scale) | 11-23 |
| [ ] | `PAPER/ccB_caption_review_2026-11-23/twotone_s8d/data/`, `twotone_s8d/run_log.txt` | `twotone_s8d/data_px/` (pixel main version) | 11-23 |
| [ ] | `PAPER/ccB_caption_review_2026-11-23/sixpos/S12D_miss_correction_mm_ccB1123.*` (v1, gray hits) | `S12D_miss_correction_mm_v2_ccB1123` | 11-23 |
| [ ] | `PAPER/ccB_caption_review_2026-11-23/twotone_s8d/S8D_miss_correction_px_ccB1123.*` (v1) | `S8D_miss_correction_px_v2_ccB1123` | 11-23 |
| [ ] | `PAPER/ccB_caption_review_2026-11-23/twotone_s8d/S8D_miss_correction_px_v2_ccB1123.*` | `..._px_v3_ccB1123` (2 x 2) | 11-23 |
| [ ] | `PAPER/ccB_caption_review_2026-11-23/twotone_s8d/S8D_miss_correction_px_v3_ccB1123.*` | `..._px_v4_ccB1123` | 11-23 |
| [ ] | `PAPER/ccB_caption_review_2026-11-23/twotone_s8d/S8D_miss_correction_px_v4_ccB1123.*` (left column missing hit stars) | `..._px_v5_ccB1123` | 11-23 |
| [ ] | `PAPER/ccB_caption_review_2026-11-23/bylicknum/FigS_kinematics_bylicknum_bylicknum_ccB1123.*`, `*_v2_*`, `bylicknum/c1/`, `c1_v2/`, `c1_bout10/`, `c1_v2_bout10/`, `c2/`, `c2_bout10/` | v3 figure + `c1_v3`, `c2_v2` tables | 11-23 |
| [ ] | `PAPER/ccB_caption_review_2026-11-23/bylicknum/FigS_kinematics_bylicknum_v3_ccB1123.*` | v4 (text moved to the caption) | 11-23 |
| [ ] | `PAPER/ccB_caption_review_2026-11-23/sixpos/S12D_miss_correction_mm_v2_ccB1123.*` | `..._mm_v3_ccB1123` (live S12 stats rule) | 11-23 |
| [ ] | `PAPER/ccB_caption_review_2026-11-23/fr_fourcam/FR_aligned_maps_fourcam_ccB1123.*`, `FR_aligned_maps_fourcam_w65_ccB1123.*`, `FR_fig5_col_v26b_terms_fourcam_ccB1123.*` | the `_v2` versions | 11-23 |
| [ ] | `RB/qc/sel-camera-cue_stop-endlick-cls-contact-incomplete-stopA2jexrt3500c70/trial_attempts{,_sessions,_gate}_jawprod.csv` (old jaw rule; copied aside by queue step `s_classes_j3m` before the in-place rebuild) | the same names without `_jawprod` (rule 3d_mouth, cap jaw table) | 11-23 |
| [ ] | `RB/attempt_coding/<run tag>_jawprod/` (f2_ac_lp on the old jaw rule; moved aside by `f2_ac_lp_j3m` when it starts) | `RB/attempt_coding/<run tag>/` (f2_ac_lp_j3m) | 11-23 |
