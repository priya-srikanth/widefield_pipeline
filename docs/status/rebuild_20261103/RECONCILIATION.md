# Reconciliation of inventories A and B: cohort 1 rebuild (2026-11-03)

Independent third check of `rebuild_inventory_A.csv` (outputs back to code) and `rebuild_inventory_B.csv` +
`rebuild_order_B.md` (code forward to outputs). Every disagreement was resolved by reading the code; file:line citations
are against the worktree named in brackets. Everything was read-only except this file and `MASTER_MANIFEST.csv`. No
render ran, no process was touched, nothing was committed.

Worktrees: [fd] = `wf_frozendef` (analysis/frozen-default 1553db6a), [sm] = `wf_swapmask` (analysis/cohort1-swapmask
3cd00814), [main] = `widefield_pipeline` (main **458f128b**, not 910a3ab9 as B states: five 11-03 commits since), [wa] =
`wf_attempt` (analysis/attempt-coding eedc1a1f).

Builder: `<scratchpad>/recon/build_master.py` (re-runnable; reads the two inventories plus `imgmap.json`, an
output-name grep of every script in [wa] / [fd] / [main] / [restnl] / the PI-deck source dir).

## 1. Counts

`MASTER_MANIFEST.csv`: 6,099 rows = 5,536 output rows (one per A row: every page figure, inline image, text block,
deck picture / table / video) + 548 script-level rows (one per B row) + 15 pipeline-step rows that neither inventory had.

| verdict | output rows (A) | script rows (B) | pipeline steps | total |
|---|---|---|---|---|
| REDO | 3,063 | 337 | 15 | 3,415 |
| NO | 724 | 206 | 0 | 930 |
| DECISION | 1,749 | 5 | 0 | 1,754 |

DECISION rows by item: D2 preprocessing decks 935; D1 per-session LocaNMF 783 (incl. B rows 13 / 542); D5 review page
18 (+3 lick-ROI script rows); D6 June decks 8; D4 imaging section 15 = 7.

Change dependencies on the output rows (column `changes`): change 1 on every imaging row; change 2 on 972; change 3 on
216; change 4 on 1,396 (plus the 4 trial-class rows below); change 5 on every imaging row. The 2 / 3 / 4 tags on deck
rows are set from A's family-level columns and titles (heuristic); the verifier should treat them as "check", not proof.

Mismatches found: **27** (section 2). 12 changed a firm verdict of one inventory (marked **changed**: M1-M8, M11,
M12, M20, M23); 6 resolved a shared or one-sided UNSURE (M9, M10, M13-M16); M24 adds a missing code step; the rest
confirmed one side. On output rows: 1,760 A rows
changed verdict (1,722 REDO -> DECISION, 27 UNSURE -> DECISION, 1 UNSURE -> REDO, 6 REDO -> NO, 4 NO -> REDO).

## 2. Mismatch list and resolutions

Join method: every script / module stem in A's `producing_script` (or, for A's 441 rows with no script, the script found
by grepping the output filename) was looked up in B; every B REDO script with a .png literal was looked up in A's image
list.

| # | item | A | B | resolution (code) |
|---|---|---|---|---|
| M1 | `hemispheric_intensity_*`, `vessel_contrast` (analysis deck, 6 rows) | REDO (family default) | NO | **changed -> NO.** Raw 415 / 470 mean-image ratios, no SVTcorr: [fd] `wfield_local/hemispheric_intensity.py:1-14`, `vessel_contrast.py:1-32`. `hemispheric_dynamics` stays REDO (`hemispheric_dynamics.py:11` uses SVTcorr; `--source roi` default `:251`). |
| M2 | `iti_lick_rate` (goal 10.6), `iti_recovery_measure` (16.6), `lick_aim_deficit` (16.10, review), `iti_lmm_figures` behaviour panel (16.9, PI slide 6) | NO | REDO ("imaging via imported script iti_licks") | **changed -> NO** (B over-propagated imports). Behaviour only: [wa] `scripts/iti_lick_rate.py:1-16,46`; aim = 3-D landing angle, `iti_recovery_measure.py:55-67`; `lick_aim_deficit.py:1-16`; `iti_lmm_figures.py:108-112` (`_behaviour` = aim only). Caveat: the aim rows are read from `where_preds_z.csv` (decoder prediction rows), which the rebuild rewrites -> re-run as a verify-identical check. |
| M3 | `fr_enl_decode` (FR page 6.1) | NO | REDO ("direct imaging consumer") | **changed -> NO.** FaceRhythm factors only, no SVTcorr / load_imaging: [fd] `scripts/fr_enl_decode.py:1-18`. Its engaged gate is in decision D8. |
| M4 | `rebuild_lick_maps`, `rest_migration/regen_lick_maps`, `regen_running_maps` | (preprocessing deck REDO) | CSV NO, addendum "REDO" | **changed -> REDO.** They run `preprocess._maps_commands` -> `framemap_event_maps`, which loads `config.svtcorr_in` ([fd] `wfield_local/framemap_event_maps.py:110-116`, `preprocess.py:512-600`). |
| M5 | per-session LocaNMF families labelled "joint-LocaNMF C" by A: `locanmf_rsa_*`, `locanmf_cross_mouse_*`, `locanmf_within_animal_consistency`, `locanmf_decoder_rolling_by_animal_*`, `locanmf_encoder_ev_*`, `precue_lickfree_*_locanmf` (46 rows) | REDO | UNSURE (rows 13 / 542) | **changed -> DECISION D1.** All use `source="locanmf"` = per-session C from `config.locanmf_dir`: [fd] `locanmf_rsa.py:56`, `locanmf_cross_mouse.py:63`, `locanmf_decoder_weights.py:63,225`, `locanmf_position_encoder.py:58,184`, `locanmf_position_decoder.py:57-60`; `nightly_figs.py:590-592` runs precue_lickfree in BOTH roi and locanmf. `locanmf_frozen_*_joint/_roi` stay REDO (joint basis / ROI). |
| M6 | `locanmf_position_session_*` (734 deck rows) + imaging Fig 11.2 | REDO / UNSURE | UNSURE | **changed -> DECISION D1** (same reason; Fig 11.2 `d_enl_move` reads per-session C per A). |
| M7 | preprocessing decks PS92-95, 935 map slides | REDO | no row (preprocess.py / preprocess_deck absent) | **changed -> DECISION D2**; B gap filled with a G2 pipeline-step row. |
| M8 | imaging section 15 (Figs 15.1-15.7) | REDO | REDO | **changed -> DECISION D4** (page says it is a frozen copy of encoding 23). |
| M9 | review page 10-27, 18 images; `lick_roi_candidates/check/foci` | UNSURE | UNSURE | **-> DECISION D5.** ROIs are rule-based (`iti_licks.roi_masks`, cited in [wa] `scripts/lick_roi_check.py:4-6`), so no ROI is re-selected. |
| M10 | `labcams/20260601/alignment_comparison_PS94_PS95*.pptx` (8) | UNSURE | none | **-> DECISION D6.** |
| M11 | PI 11-02 slides 27-30 `tc_trial_class_*` | NO | `jaw_only_incomplete_figs` NO | **changed -> REDO (change 4).** Rendered 11-02, before `classify` stopped filtering on `jaw_pass_qc` and the denominator became every engaged trial ([main] `scripts/jaw_only_incomplete_figs.py:64-69`; commits 5cae3880, 07385b37, 0e657df2 on 11-03). The same figures on the kinematics page (6.1-6.5) were already re-rendered 11-03: NO. Confirm with e3 (`_source_PI_overview_2026-11-02/recrop_trial_classes_1103.py` exists). |
| M12 | imaging page "Closed / corrected / pending" text | UNSURE | - | **changed -> REDO** (G8 text; quotes neural numbers and the correction name). |
| M13 | `cd_geometry_figure` (imaging 9.1) | REDO | UNSURE | **-> REDO**, results-fed: reads `cd_trajectories` dumps ([fd] `scripts/cd_geometry_figure.py:7-8,78`). Order G7. |
| M14 | `pose3d_cue_traces`, `pose3d_spout_error` (kinematics 3.2-3.5, 6.6, 7.1-7.4) | NO | UNSURE | **-> NO.** `session_traces` is a pose worker ([fd] `pose3d_cue_traces.py:55-61`; `pose3d_spout_error.py:89,139`). |
| M15 | `revup_followup` (goal supplements revup_jaw_traces / revup_late_engaged) | NO | UNSURE | **-> NO** (verify-identical). The "neural" gate is the revup_neural TRIAL SET read from `revup_neural_trials.csv` ([wa] `revup_followup.py:144-155`), a behaviour set that the rebuild rewrites. |
| M16 | `slowband_decode_check`, `iti_hemo_check`, `quiet_hemo_check`, `o2_video_me` | REDO (A) | row 16 UNSURE vs rows 387 / 217 / 76 REDO, 288 NO | **-> REDO, REDO, REDO, NO** (B internal inconsistency). |
| M17 | `exec_angle_spread_check` | NO (spurious name match) | REDO | REDO, share qc only; reads encoder dR2 json ([fd] `exec_angle_spread_check.py:68-72`). Not on any page. |
| M18 | `likeforlike`, `iti_lick_timing_audit` | REDO (co-listed) | NO | both right: library / audit; the A rows are REDO through the co-listed scripts. |
| M19 | helper vs B CSV: `rest_migration/channel_*`, `nvc_evoked`, `evoked_hrf_latency`, `rest_coupling` | - | helper UNSURE, CSV REDO | REDO (svtcorr_in + raw channels); deck-only (`epoch_16/20/21`) or not shown. |
| M20 | helper vs B CSV: `video_motion_session` | - | helper NO, CSV REDO | **changed -> NO**: imaging used only for frame times. |
| M21 | helper vs B CSV: `epileptiform_*`, `prestroke_quiet_videos`, `qc_running_survey`, `qc_running_push_survey` | - | helper UNSURE / CSV NO | NO (videos / behaviour; CSV already NO). |
| M22 | B's "main = 910a3ab9" | - | - | main is now 458f128b: 11-03 commits add the shared camera gates (aca69271), the s1 LocaNMF refit table, and `--plot-only` re-renders of quiet-events / FR / kinematics figures under the 11-03 figure rules. Merge 458f128b or later. |
| M23 | B section 1: "variant data DONE, do NOT rebuild" | - | - | **changed -> REDO (G1).** Change 1 now REPLACES the order-10 polynomial with block-protected DCT cosines (`DRIFT_CUTOFF_CHECK_20261103.md`, [sm] `hemo_variants.py:126-136` `dctblk<c>_lag`). That is a different SVTcorr, so a new variant dir must be built for all 120 sessions. `build_swapmask_cohort1.py:23-25` hard-codes `VARIANT="meegkit_hpfit_lag"`. All `swapmask_rerun_20261102` outputs become a comparison arm, not final. |
| M24 | change 4 in both inventories | not classified | not classified | **new pipeline step (G0).** The camera gate exists only in [main] `scripts/jaw_only_incomplete_figs.py:91-160` (GATES / session_classes / cue_classes / in_gate). No imaging family calls it: the working class is still `grant_kit._class_select` ([fd] `grant_kit.py:819-843`), `beta_maps._quit_mask`, `position_coding_directions._gate_all`, `nolick_decoder.categorize` (`nolick_decoder.py:255-264`), `cd_trajectories` (`:812`), `attempt_coding.classify` (`attempt_coding.py:76-88`). |
| M25 | `attempt_coding` pre set (imaging 5.1 / 5.2, goal 6.x / 8.x; A: "June+Aug assumed - check") | UNSURE pre | - | **June+Aug confirmed** ([fd] `attempt_coding.py:202` `config.phase_labels("pre")`), so change 2 applies; stored DAQ-frozen decoder on LP-gated classes ([fd] `attempt_coding.py:31-35`) -> D7. |
| M26 | B families with no page map (rows 5, 17, 18: "exact figure ids UNSURE", "per-slide map NOT finished", "figure-id-to-script map NOT finished") | - | UNSURE | closed by the join: every page / deck output is a manifest row. |
| M27 | A's 441 image rows with no producing script | blank | - | filled where an output-name grep found the writer (`script_source` = "reconciler grep"); still blank for deck-only crops and renamed copies (section 7). |

Behaviour NO rows re-checked against changes 2-4 specifically (both inventories classified by imaging):
- **Change 2:** behaviour epoch figures (imaging Fig 0.1, deck `epoch_1b/1c/1bdelta`, `grant_1/1b`) define pre as every
  pre-stroke day (`epoch_figures.epoch_of_day` [fd] `:581-582`), not `config.phase_labels`, so `WIDEFIELD_PRE_FROM`
  cannot move them. Kinematics / FR already use August-only (`orofacial_cohort.LATE_PRE_FROM` [fd] `:1487`). Goal 5.x
  (`lick_path_softmax`) uses the kinematics table's pre. All stay NO.
- **Change 3:** no behaviour figure uses a decoder. NO.
- **Change 4:** only M11 flips. The behaviour-pipeline engaged gate (Fig 0.1 / 0.2, deck `epoch_1f/1g`, `G1a`,
  kinematics / FR via `orofacial_cohort.engagement_keep`) is outside the gate the decision replaces
  (`gate_source.py` docstring [fd] lists them as "NOT wired"), so they stay NO pending D8. `disengagement_tests` (goal
  14.18 S28 behaviour panels) characterise the DAQ quit point itself, which is kept as the robustness row: NO.
  `enl_state_counts` numbers DO move (working / stopped counts): not on a page, re-run as the feasibility table.

## 3. What both inventories missed

1. **The final variant does not exist yet** (M23). Data must be rebuilt after the cutoff is set; B's "do not rebuild"
   is obsolete.
2. **Change 4 has no code path in any imaging family** (M24), and its cache keys are not safe: the gate hooks are marked
   `# [gate-source hook]` and `map_collect.code_digest` drops marked lines before hashing ([fd] `gate_source.py:55,82,
   190-192`). A camera-gate hook written the same way would NOT change the map-cache digest. Frozen specs deliberately do
   not carry the gate (gate_source docstring). So the camera default and the DAQ robustness row would share cache and
   output keys unless a gate tag (like `cache_tag()` / `_gateLP`) is added.
3. **Camera gate coverage holes**: PS93_0813 has no behaviour video, so `session_classes` returns None and "the caller
   must say what it does" ([main] `jaw_only_incomplete_figs.py:109-120`); PS92_0908 matches only 82.5 % of DAQ cues
   (commit aca69271 message). Decide per-session handling before any render.
4. **Preprocess map regeneration overwrites in place** (M4 / M7): maps are written into each session's
   `motion_corrected/{spout_trial_averages,lick_aligned,quiet,running_activity}_affine8v1` with no hemo tag
   ([fd] `configs/defaults.yaml:33`, `preprocess.py:529-536`). 15e and the preprocessing decks read them.
5. **Per-session LocaNMF families are larger than A thought** (M5): RSA, cross-mouse, within-animal consistency, rolling
   decoder, encoder EV and the locanmf half of precue_lickfree.
6. **Captions are code and state the old rules**: `deck_registry.py:227,384-408,1133` (working = DAQ engaged flag),
   `deck_text.py:535` and `locanmf_analysis_deck.py:582` ("order 10, 600 s" polynomial). Update before the deck rebuild.
7. **Stacked drift removal** (D9): with drift removed in the shared signal, the encoder / ITI / stray_licks cosine steps
   (B trap 14) would remove it twice.
8. **Behaviour figures that read regenerated tables** (M2, M15): numbers should not move, but their input files are
   rewritten; run them as verify-identical checks after the rebuild.
9. **Lost scripts are recoverable but only from temp scratchpads** (section 7): `fig_goal_summary.py` /
   `fig_goal_epochs.py` and `per_animal_fig.py`.
10. **PI deck trial-class slides 27-30 are stale under the 11-03 classify** (M11).

## 4. UNSURE resolutions

**A (28):** Fig 11.2 -> DECISION D1; imaging status text -> REDO; review page 18 -> DECISION D5; June decks 8 ->
DECISION D6.

**B (23 CSV rows):**

| B row | script | resolution |
|---|---|---|
| 13, 542 | nightly per-day A-C / RSA / cross-mouse / `--source locanmf` | DECISION D1 (ROI-source runs of the same families follow the variant: REDO) |
| 15 | wf_attempt-only family row | superseded by per-file rows B420-B541 |
| 16 | slowband / o2_video_me / iti_hemo_check / quiet_hemo_check | REDO / NO / REDO / REDO (M16) |
| 34 | cd_geometry_figure | REDO (M13) |
| 44 | coding_direction_audit | NO: no figure, not on any page; optional verifier |
| 131, 132 | enl_lick_rates, enl_lp_census | NO: behaviour counts / census (docstrings `:1-8`) |
| 136 | enl_state_counts | NO for pages; numbers move under change 4, re-run as feasibility table |
| 181 | frozen_default_compare | NO: diagnostic; use as post-rebuild verifier |
| 271 | lp_trial_counts | NO (no imaging; `lp_trial_counts.py:1-6`); trial-count columns of imaging section 2 tables are REDO via the text row |
| 277 | misdirected_lick_decode | REDO only if cited (O2 bundle imaging, `misdirected_lick_decode.py:5`); not on any page / deck |
| 298, 307 | pose3d_cue_traces, pose3d_spout_error | NO (M14) |
| 348 | plot_session_residual | NO: 09-13 diagnostic, not shown |
| 423, 522, 523 | curbd_pilot, mvar_pooled, mvar_regions | NO: pilots, no page / deck image |
| 436 | iti_dock_audit | NO: no figure; verify-identical |
| 511-513 | lick_roi_* | DECISION D5 |
| 532 | revup_followup | NO (M15) |

Extras in `rebuild_order_B.md` (helper disputes, family rows 5 / 17 / 18): M19-M21, M26.

## 5. Decisions for Priya

| id | decision | recommendation |
|---|---|---|
| D0 | Drift cutoff (blocking; being measured) | Pick from DRIFT_CUTOFF_CHECK; nothing in G1+ starts before it. |
| D1 | Per-session LocaNMF families: nightly A-C, RSA, cross-mouse, within-animal, rolling decoder, encoder EV, precue_lickfree (locanmf), deck per-session appendix decoders, imaging Fig 11.2 (783 rows) | **Project the corrected data onto the EXISTING per-session footprints** (fixed-footprint least squares; reproduces stored C at median r .9994, min .977 on PS93_0826 per the 11-02 log; refit is not reproducible, 95 vs 93 components). Needs new code: `config.locanmf_dir` consumers only switch by `WIDEFIELD_LOCANMF_VARIANT` ([fd] `config.py:161-164`). |
| D2 | Preprocessing decks PS92-95 (935 map slides) | **Regenerate the map slides.** The npz must be regenerated anyway for 15e; keep the 410 pre-correction slides (motion, bleach, 415/470 mean, vasculature). Write to new dirs, not in place (trap T19). Say on the slides that the decks now include October sessions. |
| D3 | Stale 10-24 analysis decks | **Regenerate wholesale** via `locanmf_analysis_deck` after G7, with the caption fixes (section 3 item 6). Priya named "imaging results ppt decks"; they are already behind the pages. Verdict kept REDO. |
| D4 | Imaging section 15 (frozen copy of encoding 23) | **Do not re-render**: replace with a pointer to encoding section 23 and mark it superseded. |
| D5 | Closed 10-27 review page | **Leave as archival** with a banner (adopted-variant figures, superseded); its ROI decisions are rule-based and stand. |
| D6 | June alignment decks (8) | **Leave**: methods history that predates the adopted variant. |
| D7 | Imaging 5.1 / 5.2, goal 6.x / 8.x: stored DAQ-frozen decoder on LP-gated attempt classes | **Apply the LP-trained rule** (consistency with the LP arm), August-only pre, DAQ-trained kept as cross-check row. They are REDO for change 1 either way. |
| D8 | Scope of change 4 beyond the imaging working class (Fig 0.1 / 0.2 engaged hit rate, deck `epoch_1f/1g`, `G1a`, kinematics / FR `engagement_keep`, `OROFACIAL_GATE` default "reference" [main] `orofacial_cohort.py:1612`) | **Keep these on their gate for this rebuild** (NO) and caption the gate; a hit rate over camera-working trials conditions on responding. The kinematics page is owned by window 34, which already imports the shared gate. |
| D9 | Family-level drift steps once the shared signal is drift-corrected (encoder >= 600 s cosine, ITI scripts, iti_basis_decoder, stray_licks) | **Remove the stacked family steps** (Priya: replace, do not stack); keep one as a sensitivity row. |
| D10 | How change 2 is applied | **Env var per decoder render block**, not a config default: `restnl_mask` refuses it, `cd_loo_sessions` / `epoch_sessions_long` require it, drift_decoder / 12.1b / Fig 4.3 need June. Tag output dirs. |
| D11 | Where the DAQ quit-tail robustness row is shown | Headline working arms only (imaging section 2, encoding headline); not every deck panel. |
| D12 | Camera-gate sessions without full video (PS93_0813 none; PS92_0908 82.5 %) | Exclude from working-class cells (no DAQ fallback, as the helper intends) and list them in captions. |

## 6. Dependency-ordered render plan

Rows in the manifest carry `order_group`. Steps inside a group are ordered.

- **G0 code** (branch, tests; no render before this): merge [fd] + [sm] (b334316a hemo_dir / output roots, 4813c3cf
  project_all, 3cd00814) + [main] >= 458f128b (quiet events, camera gates, cat_points) + [wa] goal-page scripts. Then:
  make `hemo_dir` follow the same switch as `config.svtcorr_path` (or always set the env var); `project_all` keyed on
  the basis's variant or `WIDEFIELD_JOINT_PROJECT_ALL=1`; wire the camera working gate with its own cache / output tag;
  per-session-footprint projection (D1); parameterise `build_swapmask_cohort1.py`; parameterise hard-coded-ADOPTED
  scripts (`slowband_decode_check`, `iti_hemo_check`, `quiet_hemo_check`, `lick_balance_imaging_control(+_primary)`,
  `lick_acad_midline_control`, `s1_refit_basis_table`, `encoding_drift_check`, `iti_mixed_models`,
  `wfield_local/postcue_window_test`); drop `WIDEFIELD_PRM_LEGACY_DATE8` ([fd] `lp_final_render.py:71`,
  `restnl_maps.py:43`); `lp_final_render --trained` default ([fd] `:249`) to lp or always pass it; `precue_engagement_states`
  into the frozen store; narrow the map digest; caption text; copy share-only / scratchpad scripts into the repo.
- **G1 data**: build `hemo_dctblk<c>_lag_refitT_swapmask` (name per D0) for 120 sessions.
- **G2 caches**: new `WIDEFIELD_SESSION_CACHE`; input cache after the hemo_dir merge; `restnl_mask` baselines (no
  PRE_FROM, each gate mode); preprocess activity-map npz (new dirs).
- **G3 frozen fits**: new `WIDEFIELD_FROZEN_MODEL_DIR`; DAQ arms, then `WIDEFIELD_LP_FROZEN=lp`; never `refreeze=`;
  frozen-encoder pre stats into a new `FROZEN_OUT_ROOT`.
- **G4 extracts**: new O2 bundle + O2 per-session encoder; `iti_licks.extract` and the skip-if-exists extracts
  (redirected); quiet events (`quiet_area_spikes` -> v2, main code, `QS_OUT_ROOT`, `QS_LICK_SOURCE=daq+lp`).
- **G5 families**: RESTnl / 15r / 15rpa / 14 maps + 15e; frozen epoch decoders (DAQ) with `--output`; LP arm
  (`lp_final_render --trained lp`); rest_frozen 12.1 / 12.1b (June kept); section G, nolick, cd, coding, attempt, drift;
  frozen encoder then `scripts/frozen_*`, combined_encoding + FR-in-encoder; quiet-events figures; goal-page lick chain in
  B's extract -> consumer order (B section "Extract -> consumer order", steps 1-15); deck-only families
  (epoch_16 / 20 / 21, channel maps).
- **G6 JSON producers**: grant 2 / 2b / 4, 3a / 5b, 3b / 5, section_g_figures, miss_vs_stopped, recovery_trajectory,
  cd_trajectories results, the PI-deck sidecars (`epoch_13`, `5r/5rgap`, `5rmo`) in the FINAL mode.
- **G7 JSON / sidecar-fed figures**: the renderers of G6; `cd_geometry_figure`; share-only `fig_*.py` (imaging 0.2,
  1.4, 3.4, 3.5, 4.4); `per_animal_fig.py` (3.3); `fig_goal_summary/epochs.py` (5.1 / 5.2, goal 8.1 / 8.2).
- **G8 pages / decks**: verifier side-by-side per family (per animal first); show Priya any direction / significance
  change before text is rewritten; then imaging page (from the LIVE version), encoding / goal page (live v312, incl.
  127 inline images and base64 23.23 / 23.24), FR / kinematics pages only where a row is REDO; analysis decks (D3);
  preprocessing decks (D2); PI deck with e3; re-crop deck crops; `publish_cache.py`; delete superseded caches from a
  kept list.

## 7. Lost or share-only scripts to recreate / bring into the repo

| script | output | where it is now | action |
|---|---|---|---|
| `fig_behaviour_hits.py`, `fig_decoding_sessions.py`, `fig_frozen_refit_pooled.py`, `fig_state_position_sessions.py`, `fig_state_ps94_diagnostic.py`, `fig_farcontra_attempts_sessions.py`, `fig_pose_validation_sessions.py` (+ `prep_figs.py`, `crop_*.py`, `recrop_trial_classes_1103.py`) | imaging 0.2, 1.4, 3.4, 3.5, 4.4; PI deck slides | share only: `M:/MICROSCOPE/Priya/Widefield/presentations/_source_PI_overview_2026-11-02/` (A wrote `labcams/presentations/`, the path is `Widefield/presentations/`) | copy into repo (G0) |
| `per_animal_fig.py` | imaging 3.3 | share `imaging_page_src/s_per_animal_regen_20261102/`; scratchpad `c1a9581a.../scratchpad/imaging/` | copy into repo |
| `fig_goal_summary.py`, `fig_goal_epochs.py` | imaging 5.1 / 5.2, goal 8.1 / 8.2 | only `AppData/Local/Temp/claude/C--Users-SabatiniLab/cdd1e016-6bb3-4346-8600-f08116ccf61d/scratchpad/` (temp; can be wiped) | copy into repo NOW |
| `d_enl_move` plotter | imaging 11.2 | lost; only the PNG in scratchpad `c67bec01...` | recreate from `enl_movement_multisession` outputs (D1 decides the data) |
| `master_mesh12` | PI slide 51 | "none on the share" | NO verdict (FR); nothing to rerun |
| goal-page copy / rename step | 5.1-5.8 (`fig9*`, `fig11`, `fig13*` from `lick_path_softmax` / `softmax_neural_vs_lick`), 14.18 S21-S28 (`rec_*`, `dis_*` from `within_lick_correction`, `kinematic_normality`, `kn_hits_deviation`, `lick_correction`, `face_prelaunch_aim`, `reembedding_test`, `disengagement_tests`) | no code maps writer filenames to page filenames (grep of all repos finds none) | write the copy map before G8 (5.x is NO; S21d, S23, S25-S26, S27b, S28c/f/g are REDO) |
| `s1_refit_basis_table.py` | imaging section 1 table | untracked edit in [wa]; [main] aca69271 has `s1_refit_locanmf_table` | commit / reconcile |
| kinematics / FR page sources | pages | share `kin_page_src`, `fr_page_src` | not in Github |
| `provenance/fig_sources.txt` "?" entries | imaging section 7 maps / drift, `cp_*`, `m_trend_beh_contact` | caption names a script, path unknown | resolve in G8 |

## 8. Traps (A + B + helper + reconciler, deduplicated)

Code / branch
- T1 Branch: frozen-default lacks hemo_dir / project_all / output roots (swapmask branch only); quiet-events and camera
  gate code are on main (>= 458f128b).
- T2 `combined_encoding.load_imaging` hard-codes `hemo_{ADOPTED}` on frozen-default, main and wf_attempt; even merged,
  `hemo_dir()` reads only the env var ([sm] `combined_encoding.py:394-403`). Affects encoder, frozen encoder,
  FR-in-encoder, quiet events, ITI and most wf_attempt scripts.
- T3 `project_all()` is False when the switch is made in defaults.yaml without the env var ([sm] `joint_locanmf.py:466-480`):
  in-fit pre sessions are then served stored adopted C.
- T4 ADOPTED, defaults.yaml and the env var must move together (B trap 19): spec ids hash the svtcorr_path mtime:size,
  not the data used, so a partial switch writes "variant" models fitted on adopted data.
- T5 wf_attempt cannot build or fully read the variant; `allen_brain_mask_native_grid.npy` depends on which branch last
  wrote `allen_aligned_affine8v1`.
- T6 Scripts with `hemo_{ADOPTED}` or `hemo_meegkit_hpfit/manifest.json` by literal name (list in G0); the comparison
  ones must ADD the new variant as the reference row.
- T7 Per-session LocaNMF C ignores the hemo variant (`config.py:161-164`): D1.
- T8 The final variant needs a rebuild (M23); `build_swapmask_cohort1.py` hard-codes its base variant.
- T9 Change 4 is unwired and its hooks would not change cache digests (section 3 item 2).
- T10 Camera gate has no verdict for PS93_0813 and partial match for PS92_0908 (D12).
- T11 Captions in deck_registry / deck_text / locanmf_analysis_deck state the old working rule and the polynomial.

Caches / stores
- T12 `session_cache.cached` prunes every other signature of label x kind; restnl_mask prunes its dir: new cache dirs.
- T13 Joint basis: `joint_locanmf.load()` returns the NEWEST basis; never build a basis under the variant (keep 76d884
  or pass the id).
- T14 `refreeze=` renames published adopted models (`.SUPERSEDED_*`): never pass it; expect SPEC-CHANGED lines.
- T15 LP arm before its DAQ models raises (`daq_trained_in_lp`); with `WIDEFIELD_LP_FROZEN=lp` it fits LP-trained.
- T16 `lp_final_render` forces `WIDEFIELD_PRM_LEGACY_DATE8=1` and its own `frozen_models_lpfinal` store; `--trained`
  default is daq ([fd] `:71`, `:249`); `restnl_maps` also forces LEGACY_DATE8 (`:43`). `find()` falls back to the M: store.
- T17 frozen_encoder is not in the store; never `--reuse-stats` into a dir holding adopted stats.
- T18 Skip-if-exists writers silently reuse adopted data (B trap 17 + helper list: iti_licks extract, cued_wide, enlv,
  firstlick, cue_to_lick, precue_restw / restnl, revup, stopped_traces, restw_check, lick_correction, prelick FEAT,
  prelaunch_aim FEAT, arrival_to_cue, disengagement img_*, quiet_area_spikes sessions, lmp_cache,
  frozen_encoder session_vs_frozen json, specificity_control --reuse, residual_roi, epoch_kernels stats,
  layer2_random_splits, enl_fr_removed_decode, enl_movement done-set, frozen_kernel_maps vessel cache,
  spout_unique_maps, map_test_calibration --cache, iti_licks basis / june_ref / zscore_check / session_patterns /
  decoder_pattern_stats / lick_order_preds / hemo_check). `WFIELD_ITI_OUT_W` exists only on wf_attempt and
  `lick_interhemispheric` hard-codes its OUT_W.
- T19 Output dirs ignore the run mode (no hemo / PRE_FROM tag): grant_figures/epoch, combined_encoding.out_dir,
  frozen_encoder, `cohort1_quiet_spikes` (hard-coded M: path), epoch_restnl, 13 writers without lp_licks routing or
  writeguard, and the preprocess map npz in each session dir. Always pass `--output` / OUT roots.
- T20 `--plot-only` / figdata / bundles replay stale data (no data signature): figdata.py, restnl_maps, lp_final_render,
  replot_map_figures, and main's new 11-03 `--plot-only` re-renders. Never plot-only until the data are regenerated.
- T21 JSON-fed grant / section G / miss_vs_stopped figures draw whatever mode last wrote the JSON: regenerate in the
  final mode right before rendering.
- T22 15e reads npz baked at preprocessing time: needs the G2 map regeneration, not a render.
- T23 O2 encoder bundles (`enc_v36_full_20261024/inputs/*.pkl`) hold adopted imaging and are never rebuilt: new bundle
  (affects o2_encoding, misdirected_lick_decode, fr_direction_mechanism, goal 1.6, 1.13-1.15, 21-22, imaging 13.4-13.6).
- T24 `summary_figures_1021` imports `wfield_local` from the widefield_pipeline checkout (main) via sys.path.
- T25 Re-run `publish_cache.py` after the re-render.

Pre baseline / training
- T26 PRE_FROM is refused by `restnl_mask` (`:404-405`) and `map_drift_prepre`, required by `cd_loo_sessions` and
  `epoch_sessions_long`, suspended by `epoch_audit.full_cohort`; drift_decoder, 12.1b and Fig 4.3 keep June by design.
- T27 Hard-coded August cut-offs already exist (`>= '0801'` in frozen_encoder, frozen_*, fr_*, encoder_trialavg_r2,
  contest_lines, disengagement_tests `:41`; `>= "0806"` in quiet v2 and `stray_licks.py:57`); quiet_area_spikes v1 uses
  June+Aug. Config families default to June+Aug.
- T28 Drift removal differs by family (encoder / ITI / stray_licks cosine >= 600 s; epoch decoders none; quiet events
  2-s running median): D9.

Outputs / pages
- T29 The live imaging page is ahead of every local copy (5 figures + 11-02 z-score captions): rebuild from the live
  page. The goal page live v312 has 127 images outside `<figure>` and base64-only 23.23 / 23.24.
- T30 The analysis deck reads the local `figures_working` dir, not the share.
- T31 Deck crops: PI slides use crops; re-crop after re-render. The PI deck is being edited 11-03 by e3 (backups and
  `apply_camera_view_1103.py`, `audit_1103.py` in the source dir): coordinate.
- T32 Imaging section 15 is a frozen copy (D4); the 10-27 review page is closed (D5).

## 9. Manifest columns

`mid`; `source` (A row / B row / reconciler); `destination`, `location`, `fig_or_slide`, `number`, `hidden`;
`output_file`, `output_path`; `producing_script` + `script_source` (A, B, or reconciler grep of the output name);
`command`; `changes` (1 hemo, 2 August-only pre, 3 LP-trained LP arm, 4 camera working, 5 cache / frozen store);
`change_notes`; `verdict` (REDO / NO / DECISION); `order_group` (G0-G8); `special_handling` (O2, share-only or lost
script, inline base64, hard-coded ADOPTED, load_imaging routing, per-session LocaNMF, re-crop, skip-if-exists cache,
verifier side-by-side, PI-deck owner); `resolution_note` (mismatch id + citation); `a_verdict`, `b_verdict` (original).
