# S15B six-position ME and Results l.170 on four-camera inputs (CC-B, 2026-11-23)

Full write-up (numbers, old vs new, corrected text): `M:/MICROSCOPE/Priya/stroke_orofacial_pipeline_outputs/vls_stroke_paper/ccB_caption_review_2026-11-23/SECTION_s15_me_and_l170.md`.

- `scripts/c2_engagement_tim/s1_extract_frames.py`: `--licks fused|cam4` (default fused four-camera onsets for every
  lick exclusion), `--jaw-nose 3d|cam4` (default four-camera 3-D jaw / nose). The legacy switches reproduce frames_v3
  exactly. Cache: `manuscript_stats/c2_4cam_sync60_20261122/engagement_me4cam_20261123/frames_fused_jaw3d`.
- `s4_session_tables.py`: `S4_KP_JAWNOSE=2d`, `S4_LICKS=cam4` (comparison), `S4_GATE=working` (camera working period
  by owner trial). `s7_me_combined_tests.py`: S15B me_all tests at 4,000 draws.
- `wfield_local/face_rhythm_validation.py`: `FRV_DIRECTION=3d` (3-D lick angle, +-12 deg, n_views >= 2) and
  `FRV_TRIALS=working` (camera working period instead of the FR bundle's DAQ-engaged trials).
- Results: S15B has no star changes on the four-camera build; the working-period gate adds a subacute response star.
  For l.170, the contralesional licks per trial return in 3 of 4 animals (#11 at 0.21), and the per-lick deficit
  persists in all four.
