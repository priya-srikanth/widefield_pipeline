# Superseded derived outputs: delete candidates (four-camera re-run)

Living list (Priya 11-23: "keep track of superseded outputs that we can delete"). Add each superseded output here when
it is replaced. Nothing is deleted until Priya OKs the batch (memory rule: derived outputs only; originals, raw data,
other people's folders and the frozen-model store are never deleted). Tick the box and date it when deleted.

Roots: `E` = `M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v37_sync60/imaging_encoding`,
`P3D` = `M:/MICROSCOPE/Priya/DeepLabCut/Widefield/session_poses/pose3d_4cam_3net_20261121_sync60`,
`S` = `M:/MICROSCOPE/Priya/Widefield/manuscript_stats/c2_4cam_sync60_20261122`,
`G` = `M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v37_sync60/glm_hmm`,
`PAPER` = `M:/MICROSCOPE/Priya/stroke_orofacial_pipeline_outputs/vls_stroke_paper`.

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

Keep (never delete): `frozen_models/` stores; raw videos / scans; LP predictions; `qc_cam3_drop_20261122/clips/`.
