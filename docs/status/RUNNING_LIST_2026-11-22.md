# Running list 2026-11-22 (Phase B on the adopted whole-session run)

Supervisor + runners: `C:/Users/SabatiniLab/lp_stage/phaseB_20261122/` (`phaseb.py` DAG, `neural_chain.py`, `extra_chain.py`; logs/).
Manuscript inputs of record: `HANDOFF_to_CC_2026-11-22_manuscript_v26.md` + `PROMPT_CC_figure_fixes_2026-11-22.md` (Priya: follow both).

## Decisions needed (Priya)
- Fig 4B/4E 3-D lick set: cam4 windows on the four-camera 3-D (recommended; old recipe; r 0.98 reach / 0.89 speed vs manuscript) or fused post-cue (r 0.96 / 0.55, +12 % licks).
- Fig 3C frame (seeded t332 vs alt t64; nose marked?); Fig 5C palette; Fig 3B hit-rate re-render (`epoch_grant_figures --only 1b`) under the new classes.
- cam4 one-long-lick override in the fusion split (PS95 0821 far_L t259): measure prevalence?
- Encoder: plain v36-manuscript lineage + fresh v2 frozen store `<sync60 run>/frozen_models` used (agent plan); revup 5a (lick_recovery_markers family) not yet planned.

## Done (11-22)
- Whole-session re-run (cam4 / cam1 x2 / 3-D / fusion), 125 + 1 sessions, 0 missing; 3-D licks -0.6 % (= merged sub-60 ms pairs), medians unchanged.
- Adoption -> sync60 names + encoder fused default (65b58e35); class frame-coverage gate (136ab758); merges 9a7c26d0, 8df38484 (hand-made-table scripts, byte-identical repro), c3cd923c (Fig 3C strip), 7d8da84d (encoder plan fix).
- Fixes: pose3d_phase arity (788631f4); PS93 0805 no behavior table (4cf45727).
- Phase B done: refuse (gapfix60), jaw, kinematics cam4/all/fused, phase + manuscript phase tables, classes (892 / 57,601 changed; 763 close_center no_response -> jaw_only), attempts, coverage, cohort2d, outcomes x3, lick3d, wlc run/summarise, RESTnl.
- Comparison vs manuscript: `lp_stage/phaseB_20261122/compare/summary.md` (outcomes 3 star flips / 37; Fig 4I clusters: dist -1 +2, down0 +1).
- Figure fixes (new files): `vls_stroke_paper/figure_fixes_2026-11-22/` (Fig 4 staggered stars, 5C, 3C strip).
- Clips: `pose3d_4cam_3net_20261121_sync60/clips_20261122/` (docs/EXAMPLE_VIDEOS.md).

## Running
- neural_chain: frozen encoder frame (~5 h) -> ITI chain -> aim tables -> lick_correction.
- extra_chain: pose3d_cue_traces -> kinematics_pooled_figs.
- phaseb: glmhmm, lc_run; jaw blind check agent.

## Superseded / delete candidates (after review)
- `pose3d_4cam_3net_20261121_sync60/kinematics_fused/c2_lick3d_session_summary_ungated.csv` (all contexts incl. pre-cue / ITI; wrong population).
- `clips_20261122/*_main.mp4` (re-encode test copies) if Priya confirms the originals open.
