# Amplitude outputs: two reference points (RESTnl + session mean) — read-only audit, 11-09

Decision (Priya 11-09): every amplitude output must exist with (1) RESTnl (`restnl` ref / `restnl.level`) and
(2) SESSION MEAN (dF/F as is, no further subtraction; map family ref name `raw`).
Code audited: `wf_render_v10` @ 90a5e25f. Queues + logs: `scratchpad/render_daqlk` (status as of 11-09 ~06:20).
Pages audited: imaging page (repo copy `wf_render_v10/docs/results_pages/imaging/imaging_page/imaging_results.html`),
goal page (`_goal_page/index.html`, v311 source) + FIGURE_PROVENANCE_20261106.csv, state-space page v28 (scratchpad copy
`c1a9581a-.../scratchpad/state_space_page/index.html`; origin/main still holds sections 1-8 only), PI deck 11-02 rows of
the CSV. Nothing was run, rendered, edited or committed.

Abbreviations: `G` = `M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/labcams/grant_figures`;
`TAG` = `hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_pre0801_sel-camera-cue_stop-endlick-cls-contact-incomplete-stopA2jexrt3500c70_daqlk`;
`IE` = `M:/.../rebuild_20261103/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding`; `D` = `IE/iti_licks/TAG`;
`SS` = `M:/MICROSCOPE/Priya/Widefield/state_space_recovery`. Queue steps are `step <name> ...` in `queue_daqlk*.sh`.

## 0. What "session mean" actually is in each signal (read first)

- **Area / ROI series (goal-page lick, ITI, cue-to-lick, rest-level families)**: ROI means of the hemo SVTcorr, then
  projected off `combined_encoding._slow_basis` (includes the constant, period >= 600 s), x1e3. Every series therefore
  has zero session mean, so "no subtraction" = SESSION MEAN exactly. RESTW / RESTnl levels are "rest minus session mean"
  on the same series. Exception: `disengagement_tests` `level0` uses the undetrended `pre_0` minus the first-30-min mean
  (neither reference).
- **Encoder families (combined / frozen)**: X and Y centered per session (slow basis incl. constant; combined also has an
  intercept). Kernels / betas / gains are invariant to any flat shift, so RESTnl and session mean give identical numbers.
- **SVT pixel maps (15r / 15rpa, drift, attempt maps, LPfinal, state-space SVD)**: `raw` = trial average of the
  `meegkit_hpfit_lag_refitT_swapmask` SVTcorr. Its drift removal is a robust polynomial detrend (order 10, 600 s) fitted
  on a mask that EXCLUDES strobe-0.25 s .. cue+4 s (`hemo_variants.py:99-117`, `filter_acausality_test.py:152`). So the
  zero of "raw" is a slow trend fitted through inter-trial frames, not strictly the frames_average session mean; the
  code comment in `position_reference_maps.reference_maps` ("RAW IS NOT UNREFERENCED ... implicitly referenced to its
  own temporal surround") says the same. **Decision needed (D0):** accept `raw` as the "session mean" twin for the maps
  (recommended: it is the dF/F as is), or caption it as "inter-trial-trend referenced". The state-space agent reports
  its per-trial extracts include the constant in the drift basis (session-mean zero); not re-verified here.

## 1. Gaps and the minimal step for each (for the plan)

Ordered by page importance. "No code" = existing CLI/env. All map runs: heavy lock, <= 4 workers, outputs land in new
`refs-<names>` tagged folders (run_mode tag from WIDEFIELD_MAP_REFS), nothing existing is overwritten.

### Imaging page

| # | Family (page figs) | Missing | Minimal step | Code? |
|---|---|---|---|---|
| I1 | 15r/15rpa maps, CONTACT arm, pre-cue + cue (7.1/7.1b, 7.7/7.7b, 7.8/7.8b) | SESSION MEAN (RAW). The g5_epoch 15r precue/working + cue/working arms FAILED (TypeError, fixed 4ecf72ed); only `epoch_15r*_RAWref_lick_lick` exists (G/TAG/epoch, 11-05) | re-queue `g5_epoch_15r_r` (queue_daqlk9s.sh:212, parked 11-08 13:55 as "redundant") with env `"base WIDEFIELD_MAP_REFS=raw"` and `--arm ENL cue` (add `lick` to make the lick twin on fixed code too): `python -m wfield_local.epoch_grant_figures --parallel-arms 2 --arm ENL cue lick --only 15r` | no |
| I2 | 15r ENL 1 s contact (7.5/7.5b) | RAW | same as I1 with env `WIDEFIELD_ENL_WINDOW_S=1.0 WIDEFIELD_MAP_REFS=raw`, `--arm ENL --only 15r`. Cleaner alternative for I1+I2+I3: add a `--refs` flag to `scripts/restnl_maps.py` (setup_env already takes `refs`; 3 lines) and run `restnl_maps.py --arm contact|enl1s --refs raw` so the twin uses exactly the RESTnl run's window names / captions | no (or 3-line) |
| I3 | 15r LP arm (7.3, 7.10, 7.18, 7.19 + b twins) | nothing if the page's LP figures are the gateLP set: LPfinal has all six refs incl. RAWref + RESTNLref for precue/cue/lick + 15rpa (G/LPfinal/LP_licks_gateLP_TAG_refs-mean+rest+restw+precue+raw+restnl/epoch, 11-07 12:20). If the page's RESTnl LP figures come from `restnl_maps --arm lp` (G/LP_licks_TAG_refs-restnl, DAQ gate), its RAW twin is missing -> `restnl_maps.py --arm lp --refs raw` (needs the I2 flag) | page source check only |
| I4 | Map drift June vs August (7.2/7.4/7.6/7.9/7.11/7.17/7.20 + b twins) | RAW for all three arms (folders are `refs-rest+precue+restnl`) | `scripts/map_drift_prepre.py --arm contact --refs raw`; `--arm enl1s --refs raw`; `--arm lp --refs raw` with the same env as `g5_map_drift_lp_june` (queue_daqlk10.sh:294, `WIDEFIELD_ATTEMPTS_SUPPLEMENT=.../trial_attempts_june/...`). Profile block `--block map_drift_prepre` | no |
| I5 | Attempt maps (8.1/8.1b, 8.2/8.2b) | (a) RAW: `attempt_maps` has `REST_VARIANTS = ("rest","restnl")` only. (b) The current-tag RESTnl FIGURES do not exist: `IE/attempt_maps_restnl/gateLP_TAG/` holds only maps/counts (rebuilt 11-08 22:00, csig-edce41f2a5); `d_attempt_maps_plot(_r)` ran `--plot-only` with the default `rest` variant | (b) no code: `python -m wfield_local.attempt_maps --plot-only --rest-variant restnl` (env `base WIDEFIELD_LICK_SOURCE=lp WIDEFIELD_GATE_SOURCE=lp`). (a) code: add `raw` to `REST_VARIANTS` = the restnl branch of `session_class_maps` (`attempt_maps.py:115-124`, plain `svt:rank<K>` features) without subtracting b_nl, out dir `attempt_maps_raw`, caption text at :356-366; then `attempt_maps --rest-variant raw --jobs 3` + `--plot-only --rest-variant raw`. Warm `tf-` features (same key as restnl) -> cheap | (a) yes ~10 lines |
| I6 | Fig 6.1 e_amp encoder gain `a` | none needed: patterns centered across positions per session (`grant_encoder.py:345-346`) -> invariant to a flat reference. Say so in the caption. (Note g6_grant_r4 FAILED; LP twin g6_grant_lp done 11-09 03:47) | — | no |

### Goal page

| # | Family (page figs) | Missing | Minimal step | Code? |
|---|---|---|---|---|
| G1 | lick_precue_restw (14.21 + S1; pre-cue level maps) | SESSION MEAN output (RESTW + RESTnl exist: D/precue_restw*, `_restnl` 11-09 01:27-01:29). The page already quotes a "no-baseline version (0 = session mean)" for the 14.21 D chronic dip, source not in the current tag | add `--rest-variant raw` (subtract 0; the npz already stores raw `p10`/`p05`) to `restnl.variant()` + suffix `_raw`; run `lick_precue_restw.py --jobs 3 --rest-variant raw` (env `frozen`, as l_precue_restnl_r) | yes, small; a shared `raw` choice in `scripts/restnl.py` serves G1-G5 |
| G2 | lick_precue_restw_traces (14.21 S2) | SESSION MEAN | `lick_precue_restw_traces.py --jobs 3 --rest-variant raw` after G1 | same change |
| G3 | revup_neural (14.21 S7): C_pre, L_level, pre-cue vs latency (levels) | SESSION MEAN (base + `_restnl` exist, 11-08/11-09) | cache stores `R - RESTW`; raw = load-time shift `+RESTW` (same pattern as restnl, revup_neural.py:123-133); `revup_neural.py --jobs 3 --rest-variant raw` | same change |
| G4 | lick_cue_to_lick (14.21 S12): `mean_rw`, `AUC_rw`, `lev_rw`, w03/w05 | SESSION MEAN (`_own` / `_res` / slope are within-trial and need nothing). Extract already stores raw Xc/Xl (`sessions/`, `sessions_restnl_wr/`) | `lick_cue_to_lick.py --jobs 3 --rest-variant raw` (rest vector = 0) | same change |
| G5 | lick_cue_to_lick_precue (S12 D) | none: `raw` column already output beside `rel` (both variants, 11-09 02:26) | page: show raw next to RESTnl | no |
| G6 | stopped_trial_traces (14.21 S11) | none: base `raw` is built in, drawn in the restw run (D/stopped_traces/figures, 11-08 16:57) + `_restnl` | page: show raw next to RESTnl | no |
| G7 | arrival_to_cue_traces (14.21 S13) | BOTH (nothing exists in this rebuild: `l_arrival_restnl` failed 11-07 02:07, RESERVED_HOLD; no restw/extract step queued) | queue `arrival_to_cue_traces.py --jobs 2` (restw extract, env `frozen`) then `--rest-variant restnl`; raw needs the G1 `raw` choice (shift +RESTW) | queue + same change |
| G8 | disengagement_tests P4 `level` (14.18 S28) | SESSION MEAN (`level`, `lev_rw` exist, 11-08 18:14). `level0` is neither (first-30-min mean, undetrended) | add a `lev_sm` row = `pre_b` window mean without subtraction (`disengagement_tests.py` ~:320/:1403); re-run `disengagement_tests.py all --jobs 3` (as l_disengage_r) | yes, small |
| G9 | lick_enl_offset_explore B4c (14.21 S4, contra-ipsi pre-cue level above RESTW) | RESTnl (base features = session mean already; `_restw` / `_smean` variants exist, enl_offx_* 11-09 03:48) | add `_restnl` variant reading `precue_restnl` sessions table (rollout plan G2); re-run `--history --offset --jobs 3` | yes (~40 lines) |
| G10 | Hemisphere-difference balances read 0..0.2 s from lick PEAK with no baseline: `lick_balance_angle_lmm` `dec` (14.13), `lick_balance_order` (14.15), `lick_balance_reference` (12.18), `lick_balance_regions` (12.20), `lick_balance_subsets` `raw_*` columns | RESTnl (session mean only; the difference X_R - X_L shifts by restnl_R - restnl_L) | per-session subtraction of `restnl.level(label, [R, L])` difference at load in each script (or one helper in the session-file loader), suffix `_restnl`; re-run the l_bal_* `_r2` steps. Several of these (12.18, 12.20) are offset-explained-variance / relative-to-animal's-pre-line readouts where the session constant only partly cancels -> also a decision (S3-3) | yes |
| G11 | iti `lick_gain_prediction` g (`e_gain`) | RESTnl (reads raw lags 0..0.2 s / frame SD = session mean) | subtract restnl level per area at load; re-run e_gain | yes, small |
| G12 | specificity_control / residual_roi / residual_trend / fig_resid* / lp_encoder_decompose / frozen_spout_maps | none for the two-reference rule: per-trial evoked windows on centered data (Section 3). `matched_residual` (2.3) has NO per-trial baseline: window mean of data - full on the centered signal = SESSION MEAN; RESTnl missing -> subtract the per-session RESTnl area level from the data part (residual unaffected if the model absorbs it: frozen predictions have no intercept, so it does enter) | matched_residual: decision whether worth adding (S3-4) | yes if wanted |

### State-space page (v28; from the state-space read)

| # | Family | Missing | Minimal step | Code? |
|---|---|---|---|---|
| S1 | Cortical change maps 9.1-9.3, 9.7 (RESTnl shown) | SESSION MEAN already RENDERED, not on page: `SS/page_figs/ss_cmap_{full,nocw,spout}_none_lick_svd.png`, `ss_cmap_{full,spout}_none_svd.png` (11-08) | page only | no |
| S2 | Pre-cue maps, lick-aligned trials (optional) | both | `WF_SS_BASIS=svd WF_SS_ALIGN=lick WF_SS_MAPBASE={rest,none} WF_SS_MAPWIN=precue python scripts/state_space_chronic_maps.py` (env guard not test-run) | no (verify) |
| S3 | 3.3-3.8 resp/pt size + noise size², 4.5/4.6 size row, 5.1/5.2 balance, 7.5-7.10 loop size | both are per-trial pre-window; RESTnl + session-mean versions need a reference option | `WF_SS_RESPBASE=lick|rest|none` in `state_space_sim.load()` (subtract `c_rest` via `state_space_chronic_maps._rest`, skip pre-window for rest/none); SVD basis only (RESTnl exists only in cohort-SVD coords); re-run trajectories / pertrial / noise / contact / balance_cells + page_figs for {cue,lick} x {rest,none} | yes |
| S4 | 10.1-10.12 change_parts / part_maps | RESTnl | same loader subtraction (or the `WF_SS_SHIFTBASE` pattern); re-run `state_space_change_parts.py`, `state_space_part_maps.py` (WF_SS_BASIS=svd WF_SS_ALIGN=lick) | yes |
| S5 | 10.17-10.20 shift_carried `_brest/_benl` | SESSION MEAN (`WF_SS_SHIFTBASE` accepts ""/rest/enl; "" = zero-window-mean, not session mean) | allow `none` (~line 112 `sh = shift + off_a`); `WF_SS_SHIFTBASE=none python scripts/state_space_shift_carried.py` (+ xgsr twin) | yes, 1-3 lines |
| S6 | 10.21-10.23, 10.25 shift_behavior / movement / recovery / spout_timing | SESSION MEAN (10.24 covered by 10.26) | make the `C._rest` subtraction conditional on the shared env; re-run the four scripts | yes, small |
| S7 | 9.10 why_cleaner | SESSION MEAN | low priority (needs 15r cache under raw ref) | yes |

### Deck (PI deck 11-02 rows of the provenance CSV)

Deck amplitude slides reuse the page families above (15r maps slides 22-24 / 172 -> I1-I3; lick_cued_ramp,
lateral_ratio, balance, resid16, quiet ETA -> Section 3; fig_sessions gb_fig slide 102 = gain_bio SDs, invariant).
No deck-only amplitude family found; `crop_pos_maps_rows` (slides 23-24) has no script on record (crop of 15r maps).

### Not amplitude-based (excluded; one line per family)

Decoders / accuracy / confusion (imaging 1-5, 10.1, 11-13; goal 8-13, 15.3-15.20 Haufe / patterns, session_levels Haufe
maps are covariance-based and offset-free); encoder R2 / dR2 / unique / Musall / PETH-vs-trial / Layer-2 change_ref /
rsplit / remap_checks / fr_trial_peth / variance_maps (size of fluctuations around each window mean); correlations /
coupling (coupling_by_class, exec_vs_neural, fr_direction_mechanism eta2, interhemispheric, xa, cv); distances
(crossnobis 10.1-10.2, june_august_drift distance, state-space positions / repertoire / drift 6.x-8.x); CD projections
9.x (coding geometry; NOTE: a flat session offset DOES enter `w.((sig-mu_pre)/sd_pre)` because mu/sd are frozen from
pre, so they are not offset-free — classed with decoding per the brief, flagged); gain_bio SD gains (variance, offset-free);
encoder kernels / betas / epoch_kernels / e_amp `a` (invariant, see Section 0); kinematics / behavior / within-session
decline pages (no dF/F).

## 2. Full table (amplitude-based families)

Columns: refs supported (CLI/env) | refs that EXIST now (current tag / corrected code) | gap | page shows which.

### Imaging page

| Figs | Producer | Supports | Exists (path, mtime) | Gap | Page shows |
|---|---|---|---|---|---|
| 7.1/7.7/7.8/7.16 + b (contact 15r/15rpa, precue 2 s, cue, lick) | `epoch_grant_figures --only 15r` (refs via `WIDEFIELD_MAP_REFS` = any of mean rest restw precue raw restnl); `restnl_maps.py --arm contact` (restnl only, Aug pre) | all six | RESTnl: `G/TAG_refs-restnl/epoch_restnl/epoch_15r(pa)_position_RESTNLref_{precue_working,cue_working,lick_lick}` (g5_restnl_maps_contact, 11-06 on 4ecf72ed, masks verified identical after the rest fix). RAW: only `G/TAG/epoch/epoch_15r(pa)_position_RAWref_lick_lick` (g5_epoch 11-05, code 9aaa4cc6; raw is rest-fix-independent); precue/cue 15r failed (TypeError) | RAW precue + cue (I1) | REST + RESTnl (b) + PRECUE; RAW only as numbers in captions (from `restnl_amplitude_per_animal_*.csv`, which hold restnl/rest/restw/raw per animal) |
| 7.5/7.5b (ENL 1 s contact) | as above, `WIDEFIELD_ENL_WINDOW_S=1.0` | all | RESTnl: `G/DAQ_ENL1s_TAG_refs-restnl/epoch_restnl/*precue_working*` (11-07 00:04 run) | RAW (I2) | REST + RESTnl |
| 7.3/7.10/7.18/7.19 + b (LP) | `lp_final_render.py maps` (refs = mean rest restw precue raw restnl); `restnl_maps.py --arm lp` | all | LPfinal gateLP: all 6 refs x precue/cue/lick, 15r + 15rpa + `lpfinal_amplitude_per_animal_*.csv` (11-07 12:20; LP-gate RESTnl baselines pre/post rest fix compared: 119/119 sessions identical). `G/LP_licks_TAG_refs-restnl/epoch_restnl` (restnl only, g5_restnl_maps_lp_r2 11-08) | none if page uses LPfinal (I3) | REST + RESTnl |
| 7.2/7.4/7.6/7.9/7.11/7.17/7.20 + b (drift) | `map_drift_prepre.py --refs ...` (any reference) | all | `G/map_drift_prepre/{TAG-nopre,LP_licks_...,DAQ_ENL1s_...}_refs-rest+precue+restnl/` (contact + ENL1s 11-07, LP 11-09 02:00) | RAW all arms (I4) | REST + RESTnl + PRECUE |
| 7.12-7.15, 7.21-7.24 (PRECUE) | same | — | exist | own per-trial pre-cue reference -> Section 3 | PRECUE |
| 8.1/8.2 + b (attempt maps, LP gate) | `attempt_maps --rest-variant rest|restnl` | rest, restnl | REST figures `IE/attempt_maps/gateLP_TAG/figures/*` (11-06 21:18); RESTnl data only `IE/attempt_maps_restnl/gateLP_TAG/maps_*.npz` (11-08 22:00), NO figures | RESTnl plot-only; RAW variant (I5) | REST + RESTnl (images are from an earlier render) |
| 6.1 e_amp | grant_encoder | — | g6_grant_lp 11-09 03:47; g6_grant_r4 FAILED | invariant (I6) | not stated |
| 15.x quiet events | quiet_spikes_v2*, quiet_fluct, quiet_eta_fr | — | v2_quiet_lp 11-06 | own event baseline -> Section 3 | states running-median baseline |

### Goal page

| Figs | Producer | Supports | Exists | Gap | Page shows |
|---|---|---|---|---|---|
| 14.21 main + S1 | lick_precue_restw | restw, restnl | both (D/precue_restw*, `_restnl` 11-09 01:27) | session mean (G1) | RESTnl primary, RESTW bracketed; a "no-baseline (0 = session mean)" number for the chronic dip |
| 14.21 S2 | lick_precue_restw_traces | restw, restnl | both 11-08 23:20 / 11-09 01:34 | session mean (G2) | RESTnl |
| 14.21 S7 | revup_neural | restw, restnl | both (11-08 23:31 / 11-09 00:32) | session mean (G3) | RESTW (rollout had it as (a); RESTnl now exists, page text not updated) |
| 14.21 S11 | stopped_trial_traces | restw, raw, restnl | all three (11-08 16:57) | none | RESTnl |
| 14.21 S12 | cue_to_lick extract / main / precue | restw, restnl (+ raw column in precue) | both (11-09 02:26) | session mean for `_rw` metrics (G4) | RESTnl (A, D), own pre-cue |
| 14.21 S13 | arrival_to_cue_traces | restw, restnl | NOTHING | both (G7) | RESTnl (old render) |
| 14.21 S4 | lick_enl_offset_explore B4 | base(=session mean), _restw, _smean | 11-09 03:48 | RESTnl (G9) | RESTW |
| 14.18 S28 | disengagement_tests | restnl (+ lev_rw) | 11-08 18:14 | session mean (G8) | RESTnl, RESTW sensitivity |
| 12.18, 12.20, 14.13 (dec), 14.15, balance_subsets raw_* | balance_reference / regions / angle_lmm / order / subsets | none (session mean) | 11-08 23:13 .. 11-09 04:41 | RESTnl (G10) | not stated |
| e_gain | lick_gain_prediction | none (session mean) | done | RESTnl (G11) | not stated |
| 2.3 matched_residual | matched_residual | none (session mean) | 11-08 06:09 | RESTnl (G12, decision) | not stated |
| 9.2-9.3 attempt maps | attempt_maps | as imaging 8.x | as imaging 8.x | as I5 | REST |
| 14.1-14.12, 14.14-14.20, 14.29-14.31, 18.x, 20.6-20.7, 2.2/2.6-2.8, 17.x, 23.x | per-trial-baseline families | — | — | Section 3 | windows stated in captions |

### State-space page

See Section 1 S1-S7. Exists with both: 9.4 (per animal RESTnl vs none), 9.8/9.9 (pre-cue vs RESTnl / session mean),
10.24 + 10.26 (B32 shift_rawref: raw levels of RESTnl, pre-cue, pre-lick, post-lick + task minus RESTnl; region means,
SVD, lick-aligned). 3.1/3.2 amp/shared/spread are within-session centered (identical under both); the uncentered
"shared shift" is reference-dependent and is session mean only. Change-map units differ by reference (each map divided
by the animal's pre response RMS under the same baseline). Page states the reference in sections 9-10; sections 3-5
state only the pre-window and never that the signal is session-mean zero.

## 3. Families with their own per-trial (or event) baseline — neither RESTnl nor session mean (decision needed)

For all of these the amplitude is post-window minus the SAME trial's pre-window, so RESTnl and session mean give
IDENTICAL numbers (the flat constant cancels). Recommendation: exempt them from the two-reference rule and state the
window in the caption ("per-trial baseline; reference-free"), unless Priya wants an absolute-level twin.

| Family (figs) | Reference | Script:line |
|---|---|---|
| ITI lick PETH / ramp / key resp_auc / controls (3.4, 14.x ITI, 14.27) | -1.5..-1.0 s before ITI bout-start onset; resp AUC base -1.0..-0.5 | iti_licks.py:239, :1508 |
| lick_iti_ramp_animals (14.2) | -1.5..-1.0 | via lick_cued_ramp BASE_L |
| iti_hemo_check (14.3, 14.35) | -1.5..-1.0, AUC 0..0.5 | iti_hemo_check.py:109 |
| lick_cued_ramp PETH / resp AUC / cue AUC (14.1-14.7, 14.25; deck 95-104) | lick-aligned -1.5..-1.0 (PETH), -1.0..-0.5 (resp AUC), cue -1.0..-0.5 | lick_cued_ramp.py:247, :352 |
| lick_vigor_perspout (14.8) | -1.0..-0.5 | :108 |
| lick_lateral_ratio (14.9-14.12; deck 125-187) | -1.0..-0.5 (lick, ITI, cue) | :39-41, :68 |
| balance primary / order_primary / explainer / recovery_primary / balimg(_primary) / balance_subsets ratio_* (14.13 resp, 14.14, 14.17, 14.18, 14.19; deck 57-130) | (0.1..0.5) - (-1.0..-0.5) around lick peak | lick_balance_subsets.py:148, :163-165; balimg :152 / primary :98 |
| iti_area_maps (14.20, 14.29-14.31) | ITI -1.5..-1.0; resp_wide -1.0..-0.5; cue BASE_C | :71, :151, :159-165 |
| cue_to_lick `_own` / `_res`, revup L_rise / slopes, disengagement `resp` | own pre-cue -0.5..0 (or (0..0.5)-(-0.5..0)) | lick_cue_to_lick.py:158 |
| lick_cue_to_lick_precue `shape` | mean(-2..-1.5) - mean(-1..-0.5) | — |
| Frozen-encoder evoked / residual: specificity_control, remap_lmm, residual_roi, residual_trend, fig_resid16 / spout / per_animal, frozen_spout_maps, lp_encoder_decompose (2.2, 2.6-2.8, 4.8, 17.x, 18.x, 20.6-20.7; deck 94-99, 138-141) | lick 0..1 s minus -0.3..-0.1 s around first lick; cue 0.5..1.5 s minus -0.5..0 s | frozen_encoder.py:1764 (EVOKED_WINDOWS); specificity_control.py:36, :55-58; lp_encoder_decompose.py:29-36 |
| fr_direction_mechanism internal (22.1) | cortex minus 0.5 s pre-cue mean (output is eta2 / correlations, so excluded anyway) | :43, :99 |
| Imaging PRECUE maps + drift (7.12-7.15, 7.21-7.24) | trial's own 1 s pre-cue mean (`precue` ref) | position_reference_maps |
| Quiet events (imaging 15.1-15.7; goal 23.x; deck 143-186) | event height above median of -1.0..-0.2 s on a 2-s running-median-detrended trace; ETA maps after 2-s moving-average high-pass | quiet_spikes_v2.py:58, :74-77 |
| quiet_state_response (23.16) | 0..0.5 minus -0.5..-0.1 | :53 |
| State-space sections 3-5 sizes / balance, 4.5-4.6, 7.5-7.10 loops | per-trial pre-window: cue 0-1 s minus -0.5..0; lick 0.1-0.5 minus -1.0..-0.6 (`state_space_sim.evoked`, BAL_BINS); loops = session mean minus animal's PRE-mean vector | state_space_sim.py |
| State-space 10.13-10.16 shift_carried default | zero window mean by construction (offset removed) — neither | state_space_shift_carried.py |
| disengagement `level0` | undetrended pre-cue minus the session's first-30-min mean — neither | disengagement_tests.py:545-557 |

Decisions to ask Priya:
- **S3-1**: exempt per-trial-baseline families (both references give identical numbers) and caption them as
  "per-trial baseline, reference-free"? Or also produce an absolute-level (session-mean / RESTnl) twin of the key ones
  (e.g. 14.1 PETH as raw dF/F level, which would show pre-lick level changes the baseline hides)?
- **S3-2**: state-space sections 3-5 — add RESTnl / session-mean response-size twins (needs WF_SS_RESPBASE code, SVD basis
  only) or keep pre-window only?
- **S3-3**: peak-referenced hemisphere-difference balances (G10) — are they "amplitudes" under the rule (they are levels
  on the session-mean signal) and should they get RESTnl twins?
- **S3-4**: matched_residual (2.3) and CD projections — add RESTnl twins or leave as is?
- **D0** (Section 0): accept map `raw` (inter-trial-trend-detrended SVT, no subtraction) as the "session mean" reference.
