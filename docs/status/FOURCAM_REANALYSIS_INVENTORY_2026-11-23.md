# Four-camera re-analysis inventory: what is rebuilt on the current inputs (2026-11-23, read-only audit)

Scope: every item of `STATUS_2026-11-18_CAM23_RERUN_INVENTORY.md` §4 (A-J + decks), checked against the inputs of
record. Status comes from output mtimes vs inputs, run logs (`lp_stage/phaseB_20261122/logs`, `cam3on_20261123/logs`,
`state.json`) and the run each piece of code resolves. Nothing was run or changed.

**Inputs of record.** LP run `lp_r4_heatmap_20261013__clean_v37_sync60` (`orofacial_cohort.RUN` on this branch);
3-D `pose3d_4cam_3net_20261121_sync60` + `kinematics_fused_postcue`, with PS95 0818 / 0820 swapped to cam3-on
(11-23 01:03); classes `qc/s1_trial_classes_cue_stop.csv.gz` (14:15) + `qc/trial_attempts.csv` (14:32, 11-22: 0.20 mm,
tongue_at_lips -> jaw_only); cam1-only events are not licks (2a159eb5, 15:11); n_boot 4000; S12 launch `WLC_LAUNCH=2v`.

**Common env** (this box): `C:/Users/SabatiniLab/miniconda3/envs/locanmf/python.exe`, `MKL_THREADING_LAYER=SEQUENTIAL`,
`PYTHONPATH=<worktree>;<worktree>/scripts` (the editable install points at the main checkout), CREATE_NO_WINDOW for
supervised jobs. "Manuscript lineage" = the env of `neural_chain_fourcam.py` (no render profile, adopted hemo variant,
`COMBINED_ENCODING_LICKS=fourcam`, `WIDEFIELD_FROZEN_MODEL_DIR=<sync60>/frozen_models`). "Rebuild lineage" = render profile
`cohort1_rebuild_20261103` (swapmask hemo, LocaNMF projection, Wt centre, rebuild root).

## Status table

| # | item (inventory §4) | status | evidence / why | rebuild command (env) | runtime (measured) |
|---|---|---|---|---|---|
| B1 | `pose3d_4cam` production + checks | DONE-current | sync60 3-D tree; cam3-on swap 11-23 (`_superseded_cam3drop_20261123/MANIFEST.md`) | -- | -- |
| B2 | `docs/POSE_MODELS.md` Part E production record | NOT RUN | no sync60 / cam3-on entry in Part E | doc edit | -- |
| D1 | jaw_onsets (0.20 mm) | DONE-but-stale (2 sessions) | `qc/jaw_onsets.csv` 11-22 14:09 reads 3-D frames; PS95 0818 / 0820 frames replaced 11-23 01:03. The 11-23 reach entry covers licks / kinematics only, not the jaw calls | `python scripts/jaw_onsets.py --force` (or the two sessions) | 56 min (all) |
| D2 | trial_classes / s1 snapshot / trial_attempts | DONE-current (except PS95 0818 / 0820 close-spout jaw calls, via D1) | 14:15 / 14:32 11-22; byte-identical after the cam1-only source drop (b0b473dd) | `python scripts/s1_trial_class_snapshot.py --out ...`, `python scripts/trial_attempts.py --force` | 11 + 36 min |
| D3 | lp_licks / session_licks / gate_source | DONE-current | fused train `..._sync60_gapfix60` is `FUSED_DEFAULT`; cam1-only drop in `signature` | -- | -- |
| D4 | camera_coverage_tables | DONE-current | `qc/camera_coverage.csv` 15:04 | -- | 40 min |
| D5 | RESTnl mask / restnl_licks | DONE-but-stale | Phase B `restnl` 11-22 06:34-06:51, bare env, local cache `~/.widefield_session_cache_restnlfv1`; built BEFORE the class rebuild (working period) and BEFORE cam1-only events stopped being licks | rebuild inside the imaging rebuild (F); bare: `python -m wfield_local.restnl_mask --jobs 4` | 17.6 min |
| D6 | iti_licks extracts (first licks, ITI) | DONE-current (3-D values of PS95 0818 / 0820 pre-swap; lick list unchanged) | `imaging_encoding/iti_licks` 11-22 19:17-21:32 from the fourcam encoder | -- | ~55 min |
| D7 | lp_trial_counts, trial_class_raw_share_fig, misdirected_lick_counts, lick_provenance wording | NOT RUN | no sync60 outputs; caption wording still "cam4 + cam1" in deck text | each script, `--run` default = sync60 | minutes each |
| A1 | behavior page camera-class panels: within_session_decline, disengagement_tests, trial_class_raw_share_fig, jaw_only_incomplete_figs, stopped_trial_traces | NOT RUN | `within_session_decline` last 11-07 and its AIM is pinned to `rebuild_20261103/.../clean_v36`; disengagement writes under the encoding root (none under sync60) | per script; `within_session_decline` needs its AIM path re-pointed first | not measured |
| A2 | Fig 3B hit rate under the new classes (`epoch_grant_figures --only 1b`) | NOT RUN (decision open, RUNNING_LIST 11-22) | -- | `python -m wfield_local.epoch_grant_figures --only 1b` (rebuild lineage) | ~15-26 min/family |
| C1 | manuscript kinematics (Fig 4A-I, scalar family, phase, cue traces, pooled figs, protrusion vs reach, aim tables, S12, rev-up, correlations) | DONE-current | `_cam3on` folders 11-23 00:56-03:10, families at 4,000 (`cam3on_20261123/state.json`) | -- | driver ~4 h |
| C2 | 2-D cohort tables (`orofacial_cohort` cohort2d, prepost stats) | DONE-but-stale | `<sync60>/cohort` 11-22 07:06-07:51: before the 0.20 mm / tongue_at_lips classes and the cam1-only rule; stats at 2,000 draws (default now 4,000) | `python -m wfield_local.orofacial_cohort ...` (Phase B `cohort2d` step) | 45 min |
| C3 | page-only 3-D / lick-level scripts: lick_trajectories_3d, movement_position_angle, lick_path_softmax, contact_direction_predictability, lick_vigor_perspout, kinematic_normality, lmm_curve_summaries, cam1_lick_aim, face_prelaunch_aim, prelaunch_aim_tracking, tongue_angle_phase, first_lick_latency / iti_first_lick / iti_lick_rate page figs, jaw_lateral, nose_onsets (no `qc/nose_onsets.csv` under sync60), pose3d_summary_fig / pose3d_figures, cam1_incomplete_licks, rebuild_figs/* | NOT RUN | no sync60 outputs (3-D `figures/` holds only `pooled*`) | per script (default folder = `kin_name()`) | prereg times: face_prelaunch_aim ~2 h, prelaunch_aim_tracking ~40 min; rest minutes |
| C4 | lick-template test `lick_template_match.py` (Priya 10-07 MUST REVISIT) | NOT RUN | -- | `python scripts/lick_template_match.py` | not measured |
| E1 | FR factors / projections | N/A | raw video | -- | -- |
| E2 | FR peak by trial class (`face_rhythm_peak_trialclass`) | DONE-current | `readouts/master_pre_v1/peak_trialclass_20261122` 11-22 19:18 (after the 14:15 classes) | -- | -- |
| E3 | FR lick-window readouts: face_rhythm_readouts, face_rhythm_lick_epochs, face_rhythm_validate | NOT RUN; code still cam4 | last outputs 10-24..11-16; `face_rhythm_validation.lick_table` reads cam4 `licks_all.csv` of the input run (not the fused train) | re-point `lick_table` to `session_licks`, then re-run the three scripts | not measured |
| E4 | fr_enl_decode, enl_fr_removed_decode | NOT RUN | (`lp_licks.lp_lick_times` already follows the fused train) | per script | not measured |
| E5 | gated FR panels + FR Fig 5.6 (states collector) | NOT RUN | waits on J3 | via J3 | -- |
| F1 | spout-coding families on the camera gate (epoch_grant_figures / grant_figures, 15r maps, decoders, CDs, RSA, retained fraction) | NOT RUN; code pinned v36 | `rebuild_20261103/LP_analysis/` holds only `clean_v36`; `code_root` `wf_rebuild` = 11-05, `RUN = clean_v36`; a rebuild-profile run on this branch would look for `rebuild_20261103/LP_analysis/<sync60>/` (absent) | new rebuild root / re-seed (decision), then the queue in `REBUILD_CODE_20261103.md` under `render_profile cohort1_rebuild_...` | full epoch render ~2 h per arm set; whole 11-06 redraw est. ~16 h at 1-2 workers; original rebuild queue ran 11-04..11-10 |
| F2 | LP_licks family (`WIDEFIELD_LICK_SOURCE=lp`): lp_final_render maps / decoders, nolick_decoder, attempt_coding, attempt_maps, enl_window, precue_lickfree, ENL censuses | NOT RUN | v36 only | `render_profile --block lp_final_render`; `python scripts/lp_final_render.py maps|decoders`; `python -m wfield_local.attempt_coding --aligns precue cue lick` | lp_final maps ~40-81 min, attempt maps 42 min, RESTnl baselines 26 min/gate (11-17 table) |
| F3 | frozen spout decoders on the gated pre set (D6): locanmf_frozen_decoder, rest_frozen_decoder_nl | NOT RUN | gate changed -> new spec_id; rebuild store holds v36 specs only | fit under the new root's frozen store (rule 10: keep the old) | 12.1b ~25 min per pre set |
| F4 | RESTnl imaging arm: rest15f_restnl_1117, restnl_maps, restnl_position_coverage | NOT RUN | v36 only | after F1 seed; `rest15f`: collect 14-17 min + fit 2-4 min/pre set | ~1 h |
| F5 | misdirected / attempt families, deck keyfigs, `provenance/lick_source.tsv` | NOT RUN | v36 only; deck_keyfigs literals cite v36 on purpose | after F1-F4 | -- |
| G1 | frozen encoder, fourcam (`frozen_encoder_gate-attempted_v7g3_t3d_frA`) | DONE-current (by the 11-23 decision) | 11-22 15:31-19:17, fourcam licks, rebuilt classes. PS95 0818 / 0820 exec-3-D angle <= 0.3 deg stale; DECISIONS 11-23 keeps the fit. Manuscript lineage (no spont kernels, adopted hemo), not the movement page's rebuild lineage (`..._spont_frA`, swapmask) | -- | 226 min |
| G2 | frozen_encoder lick-spout posterior (`FROZEN_LICKSPOUT`) | DONE-current (open item resolved) | `lick_landing_weights` joins the fused post-cue kinematics by lick_key in fourcam mode (c1fb429f, 14:45, before the fit); the ITI chain ran with `FROZEN_LICKSPOUT=1`. The "stays cam4" row in FOURCAM_LICKS_EVERYWHERE is out of date (cam4 lick modes only) | -- | -- |
| G3 | movement-page encoder figures (`combined_encoding_figures`, frozen encoder figure scripts) | NOT RUN | no figures from the fourcam frame; page v326 is on rebuild v36 frames; n_boot default now 4000 (compare stars on re-render) | `python scripts/combined_encoding_figures.py --run lp_r4_heatmap_20261013__clean_v37_sync60 ...` after the lineage decision | not measured |
| G4 | combined_encoding production (O2) | NOT RUN | no sync60 bundle | `scripts/o2_encoding.py` bundle; diff env.json vs last production bundle; pin `COMBINED_ENCODING_LICKS=fourcam`; local smoke test first | O2 hours |
| G5 | tongue-from-cortex decoding: softmax_neural_vs_lick, trial_neural_vs_exec, exec_bins_calibration, exec_vs_neural_match, within_spout_deviation | NOT RUN | none under sync60 | per script, after G1 / G4 | not measured |
| G6 | lick_correction neural step | NOT RUN | `--extract` 11-22 20:15 + `--behaviour-only` tables done; `iti_licks/figdata/lick_correction_neural.pkl` absent ("run the script once without --plot-only") | `python scripts/lick_correction.py` (manuscript lineage, `FROZEN_LICKSPOUT=1`) | extract 4 min + behavior 9 min; neural not measured |
| G7 | within_lick_correction neural step (N1 / N2) | BLOCKED | `neural_session` builds its frame from `z["lick_uid"]` (`scripts/within_lick_correction.py:543`) but the sync60 lick_correction npz carries `lick_key` and no `lick_uid` -> KeyError on all 107 sessions (`cam3on_20261123/logs/wlc_run.log`). One-line fix: take `lick_uid` only when present (the join below already prefers `lick_key`) | after the fix: `WLC_LAUNCH=2v python scripts/within_lick_correction.py run` (WFIELD_ITI_OUT_W = the cam3on iti folder) | run 46 min (extract part) |
| G8 | lick-balance / lick-recovery neural (recmark, recprim, lick_enlv_*, lick_cued_ramp, lick_iti_*) | NOT RUN | `revup_fourcam(_cam3on)` = behavior only; DECISIONS 11-23: rebuild under `WLC_LAUNCH=2v` only if a neural figure uses the launch angle | per script after G6 / G7 | not measured |
| H | quiet events (`run_quiet_chain.sh`) | BLOCKED | runner hard-codes the rebuild env (`WIDEFIELD_REBUILD_ROOT=rebuild_20261103`, its frozen store and session cache); its preflight needs attempt_coding precue tables under the same key (F2), which exist only for v36. Last build `v2_quiet_lp_c1guard_syncend_20261118` (v36). `QS_LICK_SOURCE=daq+lp` itself follows the fused train via `lp_licks.lick_times` | after F2 under the same key: `scripts/cohort2_qc/run_quiet_chain.sh v2_quiet_fourcam_<YYYYMMDD>` from a clean checkout | 11-18 build: 13:58-23:36 (~10 h with gaps) |
| I | state space (enl / rest baselines, rest_decomp, shift_*, landing, sim, balance_angle, aligned_decoder, fr_coupling) | NOT RUN; code pinned v36 | `state_space_sim.REBUILD_RUN = clean_v36` (pinned 11-22 on purpose); `aligned_decoder` applies the rebuild profile; `state_space_rest_decomp` reads cam4 `licks_all` | after F1 (+ iti extracts in the rebuild lineage); re-point REBUILD_RUN; then the 15-step chain | ~1 h guessed (11-17 plan; none on record) |
| J1 | GLM-HMM refit | DONE-current, but default path stale | refit in `glm_hmm/v1_floor020` (16:26, after the rebuilt classes and the cam1-only drop). `glm_hmm.out_dir()` defaults to `glm_hmm/v1` = the 08:59 refit on the 0.10 classes with cam1-only flicks; every consumer without `GLMHMM_OUT` reads the stale fit | promote, or set `GLMHMM_OUT=<sync60>/glm_hmm/v1_floor020` for J3 / E5 / S22 | 88 min |
| J2 | S22 engagement (`c2_engagement_tim`, no TIM) | DONE-current | `engagement_floor020` 11-22 16:48-16:57 | -- | ~10 min |
| J3 | behavior_states collect + states_page_figs + GLM-HMM reels | NOT RUN | `behavior_states.out_dir()` = `<sync60>/arhmm/v1/poststroke_<tag>`: absent | `GLMHMM_OUT=... python -m wfield_local.behavior_states collect --jobs 8`; `python scripts/states_page_figs.py`; `... behavior_states reels` | not measured |
| J4 | AR-HMM features + re-infer / refit, syllable reels, arhmm_states | NOT RUN | no `<sync60>/arhmm/`; `arhmm_features.VERSION = 3` (pose3d jaw + 2-D tongue / nose) not bumped; `arhmm_states` lick enrichment reads cam4 `licks_all.csv`; fits live under the v36 run (`ARHMM_FITS` to re-infer) | bump VERSION; `python -m wfield_local.arhmm_features --all --jobs 4`; re-infer (`ARHMM_FITS=<v36>/arhmm/v1/fits`) or O2 refit `scripts/o2_arhmm.py` | not measured |
| J5 | TIM (`task_independent_movement`, TIM_RUN) | NOT RUN | `TIM_RUN` defaults to `oc.RUN` = sync60; no `<sync60>/task_independent_movement/v2` | `python -m wfield_local.task_independent_movement --all --jobs 4` | not measured |
| K1 | analysis deck (nightly), PI deck | NOT RUN | follow F / G / H | after F-H | deck build ~1.5 min |
| K2 | K08 grant figures | N/A | submitted; record changed numbers only | -- | -- |

**Counts** (42 rows): DONE-current 11 (B1, D2, D3, D4, D6, C1, E2, G1, G2, J1, J2; D2 with a 2-session caveat) ·
DONE-but-stale 3 (D1, D5, C2) · NOT RUN 24 (B2, D7, A1, A2, C3, C4, E3, E4, E5, F1-F5, G3-G6, G8, I, J3-J5, K1) ·
BLOCKED 2 (G7, H) · N/A 2 (E1, K2). Rows group several scripts, so the counts are per row.

## Run order (dependencies)

1. **Small fixes first (code, no compute):** `within_lick_correction.py:543` lick_uid guard (G7);
   `face_rhythm_validation.lick_table` -> `session_licks` (E3); `within_session_decline` AIM path (A1); re-point
   `state_space_sim.REBUILD_RUN` once the new rebuild run exists (I); bump `arhmm_features.VERSION` (J4). Fix the
   FROZEN_LICKSPOUT row in FOURCAM_LICKS_EVERYWHERE (G2).
2. **PS95 0818 / 0820 tail of the cam3 swap (if Priya wants it):** jaw_onsets -> s1 -> trial_attempts (~1.7 h all
   sessions), then check whether any class changed. If none changed, nothing downstream moves.
3. **Behavior / kinematics remainder (no imaging):** C2 cohort tables (45 min) -> C3 / C4 page scripts -> A1 / D7.
   J1 promotion -> J5 TIM -> J4 AR-HMM features + re-infer -> J3 collect + states figures + reels -> E5. E3 / E4.
4. **Manuscript-lineage neural tail (on the 11-22 fourcam encoder):** G6 lick_correction neural -> G7
   within_lick_correction neural -> G8 recovery neural (only if a neural figure needs it) -> G3 encoder figures (if
   the movement page moves to this lineage).
5. **Imaging rebuild (rebuild lineage):** new / re-seeded rebuild root on sync60 (`python -m wfield_local.rebuild_root
   init --profile <new profile>`) -> RESTnl baselines (D5, both gates) -> F1 epoch / grant families -> F2 LP family +
   attempt_coding precue/cue/lick -> F3 frozen decoders (new spec) -> F4 / F5. Then H (`run_quiet_chain.sh`, needs F2's
   precue tables under the same key) and I (state space). G4 (O2 combined_encoding) and the rebuild-lineage frozen
   encoder frames can run in parallel with F once the root exists. K1 decks last.

## Decisions needed (Priya)

1. **Imaging rebuild root:** a new root (e.g. `rebuild_<date>_fourcam`, new profile, `code_root` = a fresh worktree off
   the adopted branch) or a sync60 subtree in `rebuild_20261103`? Today nothing in the spout-coding, quiet or state-space
   arms has moved off v36. That is the largest block of compute left (~16 h redraw at 1-2 workers plus the frozen fits).
2. **Movement page lineage:** the fourcam frozen encoder is the manuscript lineage (`v7g3_t3d_frA`, no spontaneous lick
   kernels, adopted hemo). Page v326 is on the rebuild lineage (`v7g3_t3d_spont_frA`, swapmask, Wt). Re-fit the
   rebuild-lineage frames on fourcam licks (~4 h per frame), or move the page to the manuscript frame?
3. **D6 frozen decoders:** confirm a new spec per gate (old store kept as reference) for the imaging decoders.
4. **D5 AR-HMM:** re-infer with the saved fits on VERSION 4 features, or refit on O2? (The GLM-HMM was refit, not re-inferred.)
5. **GLM-HMM path:** promote `glm_hmm/v1_floor020` to the default `v1`, or keep `GLMHMM_OUT` in every consumer?
   Then `v1`, `v1_floor020_withflicks` and `v1_test_cam4licks_floor010` become delete candidates.
6. **PS95 0818 / 0820 jaw calls:** re-run jaw_onsets -> classes for the cam3-on frames, or record them as unchanged?
   This is not covered by the 11-23 reach entry, which addresses licks and kinematics only.
7. Open from RUNNING_LIST 11-22: Fig 3B hit-rate re-render (`--only 1b`), Fig 3C frame, Fig 5C palette, the cam4
   one-long-lick override, revup 5a.
8. Lick-template test (10-07 MUST REVISIT): run it in this pass?
