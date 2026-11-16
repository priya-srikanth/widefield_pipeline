# Decision audit: does the 11-05 work (rebuild -> main merge, today's commits, the `_daqlk` render queue) go against a recorded decision?

Read-only audit, 11-05 evening. Nothing was edited, run, started or stopped.
- Code: main a8f1651d (worktree wf_merge_main).
- Queue: `scratchpad/render_daqlk/queue_daqlk3.sh`, which equals v2 plus the step claims.

The one already found is fixed in 76066e01: the encoding_mixed angle panels had switched to polar `3d:u_target` against
DECISIONS 2026-10-24 (later), which makes the binned contest primary. It is not repeated here.

Severity:
- **REVERTED-DECISION**: the decision text was found, and tonight's code or queue produces output against it. Here that
  means the decision was recorded but never implemented, so the render uses the old default.
- **POSSIBLE-REVERSION**: probably against a decision, but the scope of the decision is ambiguous, or the effect is
  indirect.
- **CONVENTION**: a standing figure convention.

## Findings, most severe first

### 1. REVERTED-DECISION: the state discriminators still train on DAQ engaged vs ALL DAQ no-lick; stopped trials are in the training set
- **Where:**
  - `wfield_local/precue_engagement_states.py:316-318`: `keep_dis = u_pre & late_nl & ~miss`, the DAQ no-lick arm.
  - `wfield_local/poststroke_compare.py:127-160`, `balanced_lick_sample`: pre DAQ engaged vs pre DAQ no-lick, used by
    `looks_like_which`, `fits_engaged_distribution` and `undetected_state_split`.
  - Queue steps that render them tonight: `g5_epoch` (12s / 12b / 13s), `g5_section_g`, `g6_grant`, `g7_section_g_figs`,
    `g7_fig_state_position`, `g7_fig_s_per_animal`.
- **Decision:** widefield_pipeline `docs/status/DECISIONS_PENDING_MERGE_2026-11-04.md` (f) 1, Priya 11-05: "State
  discriminators (camera-class question 8): train working (contact + incomplete) vs no_response; STOPPED trials are
  scored as their own class, never trained on ... jaw_only and after_end stay scored-only." It was relayed to 34
  (RUNNING_LIST_2026-11-03_AWAY, af61a261).
- **Evidence:**
  - No commit on main since 11-04 touches these sites.
  - CAMERA_CLASS_INVENTORY_20261104.md section 3a lists the current set as "trained on pre DAQ engaged vs pre DAQ
    no-lick. The no-lick side holds incomplete, jaw_only, no_response and stopped together". That is still the code.
- **Fix:** implement (f) 1 with new stored-model ids (class tag in the spec). Then re-render those G steps. Until then,
  mark every state-discriminator figure from tonight as the superseded training set, and do not place it.

### 2. REVERTED-DECISION: the between-trial lick ENL families and fr_enl_decode do not slide
- **Where:**
  - In the queue: `scripts/prelaunch_aim_tracking.py`, steps `e_pat_extract` / `fit` / `summarise`. Its pre-cue ENL
    reference (`:1064-1075`) is lick_enl_offset_explore B7, on a fixed window.
  - Not in any queue: `lick_enl_angle_decode`, `lick_enl_target_test`, `lick_enl_offset_explore`, the pre-cue part of
    `revup_neural`, and `fr_enl_decode`.
- **Decision:** RUNNING_LIST_2026-11-01.md, 11-04 ~20:00: "DECISION (Priya): the between-trial lick ENL family and
  fr_enl_decode also SLIDE, for consistency. Families: lick_enl_angle_decode, lick_enl_target_test,
  lick_enl_offset_explore, prelaunch_aim_tracking / revup_neural pre-cue parts, fr_enl_decode." Proposed guard: start
  >= strobe + 0.3 s. It is also queued item 1 in the encoding handoff, 00-COMPACT-25. It supersedes the ~17:40
  "keep them fixed" recommendation.
- **Evidence:**
  - None of these scripts imports `enl_window`.
  - `grep sliding / enl_window / precue_window_start` finds nothing in them.
  - `CUE_EXCL = (-2.0, 0.75)` is unchanged in iti_licks / prelaunch_aim_tracking.
- **Fix:**
  - Build the sliding version (new frozen ids, as the entry expects).
  - Treat tonight's e_pat pre-cue ENL numbers as the fixed-window version.
  - Add the lick ENL families to a later queue.

### 3. POSSIBLE-REVERSION: Fig 0.2 hit rate is still left on the DAQ engagement gate, citing D8; 11-05 (f) 2 reversed D8 for the hit rate
- **Where:**
  - RENDER_QUEUE_DAQLK_20261105.md "Not placed" 4: "fig_behaviour_hits: reads behaviour only, so it stays on its gate
    per D8".
  - `scripts/rebuild_figs/fig_behaviour_hits.py:11,16` reads `cohort_session_metrics.csv hit__<pos>`, the engaged-gated
    DAQ rate.
- **Decision:** DECISIONS_PENDING_MERGE (f) 2, Priya 11-05: "Hit-rate period: the CAMERA working period (first cue ->
  last camera lick, minus stopped runs) is primary; the DAQ quit-tail rule is a labelled robustness row. Changes Fig 0.2
  and the cohort_session_metrics readers." It supersedes RECONCILIATION.md D8 ("Keep these on their gate for this
  rebuild") for the hit rate.
- **Evidence:** no code on main computes the camera-working-period hit rate for `cohort_session_metrics` or
  `fig_behaviour_hits`. The plan doc still justifies leaving it out with D8.
- **Fix:**
  - Implement the camera working period for the hit-rate readers.
  - Place Fig 0.2 in a queue, with the DAQ quit tail as the labelled robustness row.
  - Update the plan doc's D8 line.

### 4. POSSIBLE-REVERSION: no queue renders the labelled robustness rows the decisions keep
Each of these is a decision that keeps a robustness row on the pages. None has a step in queue_daqlk3.sh, and nothing
else is scheduled. Only page and deck content is listed.

| row | decision (doc, date) | status |
|---|---|---|
| frozen no-spont (v7g3_t3d) downstream: decompose / relabel / Layer 2 / svf / partial / figures | RUNNING_LIST 11-04 ~20:55 (Priya via a8): "v7g3_t3d is never overwritten and stays as a labelled robustness row"; pages 2.1, 2.10 / 2.12, 20.3-20.7, 1.1 need "the frozen target columns before / after"; 00-COMPACT-28: "no-spont as a robustness row (PS95 late chronic shown)" | frames exist (`frozen_encoder_gate-attempted_v7g3_t3d{,_frA}` under `_daqlk`). The plan doc says "not asked for". |
| DAQ quit-tail working row | RECONCILIATION D11: "Headline working arms only (imaging section 2, encoding headline)"; DECISIONS 2026-11-03 (night) camera gate: "The DAQ quit-tail rule stays a labelled robustness row"; (f) 2 | no step with `WIDEFIELD_WORKING_TRIALS=quit_tail` / `WIDEFIELD_TRIAL_GATE=daq` |
| drift1500 encoder / ITI sensitivity arm | handoff 00-COMPACT-21 (Priya late 11-03): "keep 600 s primary, plus a 1500 s sensitivity arm for the spout contra - ipsi cells and the ITI pre-cue level readouts"; profile `arms.drift1500` | the plan doc says "not in this queue" |
| enl_fixed2s robustness row (encoder ENL) | RUNNING_LIST 11-04 ~17:15: "The old fixed 2 s ungated window is kept as the 'enl_fixed2s' robustness row" | computed inside the O2 runs; no figure step names it (not verified further) |

**Fix:** add a robustness lane, or record with Priya which of these rows stay on the pages.

### 5. POSSIBLE-REVERSION: encoding_key_figures fails or is stale under the rebuild
- **Where:** `scripts/encoding_key_figures.py`, queue step `b_encoding_key_figures`.
- **Problem 1, a stale caption against a decision.** `:58` hard-codes the suptitle "The pre-cue window is NOT
  lick-free: our lick tables start at the cue". That contradicts the 11-04 decisions: RUNNING_LIST ~17:15 / ~17:35 /
  ~17:40, "encoder enl = lick-free", sliding LP 1.0 s, LICKS_ALL=split with its own pre-cue lick kernel.
- **Problem 2, stale text.** `:83`: "(cam1 not yet for PS94 / PS95)".
- **Problem 3, a functional failure (not a decision).** `:41` reads `E / "movement_first"`, which does not exist under
  `_daqlk`: the main run is `movement_first_exec-polar_licks-combined`. `pd.concat([])` raises ValueError, which the
  `except FileNotFoundError` at `:146` does not catch. The step will therefore fail, and lick_deficits / landing_zones
  will not be drawn. 19d3cf71 gave encoding_mixed its `_run()` fallback; this script did not get one.
- **Fix:**
  - read the main run through the same `_run` rule;
  - rewrite the two suptitles from tonight's numbers (captions are replaced, not annotated);
  - see also C3 / C4 below.

### 6. POSSIBLE-REVERSION (wording): DECISIONS / the profile say 12.1b keeps a "June + August" pre set; the queued script runs August-only
- **Where:** queue step `g5_rest_frozen_nl`: `--block rest_frozen_decoder_nl` plus `--pre aug`.
- **Decisions:**
  - DECISIONS 2026-11-03 (late): "August-only pre everywhere in the rebuild, except june_august_drift and 12.1b".
  - The profile comment says: "12.1b (rest_frozen_decoder_nl) keep[s] their own June + August pre set".
- **Evidence:** the script's own docstring (`rest_frozen_decoder_nl.py:19-22`) says `--pre aug` is PRIMARY. It says
  `--pre all` only reproduces the pre-rebuild page, and "in the cohort-1 rebuild page 12.1 is August-only".
- **Reading:** the code and the queue are consistent with each other. The DECISIONS / profile wording is what
  disagrees.
- **Fix:** confirm with Priya which pre set 12.1b shows. Then make DECISIONS and the profile comment say it.

## Convention findings (CONVENTION)
Stars in the top band: Priya 10-30, rule in docs/templates/README.md, `sig_marks.star_top` + `reserve_headroom`. No top
spine: Priya 11-03. Canonical spout order: Priya 10-25, `orofacial_cohort.POS_ORDER`. All scripts below are run by
tonight's queue.

- **C1. Stars placed on the data through a shared helper.**
  - `wfield_local/sig_marks.py:13-24` `sig_star` puts the star at max(CI, 0, animal dots) + 1 pt.
  - `scripts/frozen_fr_compare.py:124` `bars` calls it, and does not hide the top spine.
  - Reached by: `partial_refit` (bars `:467`, `:473`), `session_vs_frozen` (`:263`), `specificity_control`
    (`:208-226`), `frozen_claims` (`:150`), and frozen_fr_compare itself.
  - Fix: make `sig_star` call `star_top`, and call `reserve_headroom(fig)` in the callers. One change covers all five.
- **C2. `scripts/rebuild_figs/fig_pooled.py:87-88`.** The star is `ax.text` at max(CI hi, animal values). No figure
  hides the top spine. Fix: `star_top` + `reserve_headroom` before each savefig (`:137`, 169, 199, 220, 255, 272, 293,
  339).
- **C3. `scripts/combined_encoding_figures.py`.**
  - `:381` `POS6 = (close_contra, close_mid, close_ipsi, far_contra, far_mid, far_ipsi)`: the row order of
    position_each_maps (droppos / shufpos), position_signed_* and position_share_maps. SECTION_C_RENDER already flagged
    this.
  - `:535`: the star is at the current ylim, with no headroom.
  - No figure hides the top spine.
  - Fix: `POS6 = oc.POS_ORDER`; `star_top` + `reserve_headroom`; `axes.spines.top=False` in `_plt` (`:84`).
- **C4. `scripts/frozen_spout_maps.py:25`.** `SPOUTS = (far_contra, close_contra, close_mid, far_mid, close_ipsi,
  far_ipsi)` is non-canonical. Fix: POS_ORDER.
- **C5. `scripts/encoding_key_figures.py:72`.** lick_deficits rows run far_contra, far_mid, close_contra. Fix:
  close_contra, far_mid, far_contra. Its panels also connect epochs with lines and colour by animal, against the
  results-page conventions (shape = animal, colour = epoch, no lines across epochs).
- **C6. No top-spine handling** (fix: `reserve_headroom` or `rcParams['axes.spines.top']=False`):
  - `wfield_local/mixed_models.py` `plot_epoch` (`:259`) and `plot_fits` (`:316`). These reach encoding_mixed (also
    noted 19:50 in the running list), remap_lmm and iti_lmm_figures.
  - `fig_layer2.py:52,92`, `fig_rsplit.py:45`, `fig_goal_epochs.py:54`, `fig_goal_summary.py:100,133,173`.
  - `exec_vs_neural_match.py:158,185`, `frozen_claims.py:171` (and its twinx), `misdirected_lick_decode.py:427,466`.
  - `within_lick_correction.py:876,911,954`.
- **C7. Uncertain.**
  - `misdirected_lick_decode.py:425-426` orders its panels far_contra -> far_mid -> far_ipsi. Is that a per-spout
    panel set?
  - Several imshow map / matrix panels keep a frame, so a top spine shows: fig_resid_per_animal, fig_softmax fig9 /
    fig9b, fig_softmax_stats, frozen_residual_significance, frozen_pixel_maps, epoch_kernels maps, session_vs_frozen and
    specificity_control maps, exec_vs_neural_match matrices. These need fixing only if the no-top-spine rule covers
    image panels.
- **Compliant:**
  - fig_frozen_refit_pooled, fig_state_position_sessions, fig_s_per_animal;
  - contest_lines and te_readout (`epoch_lines`);
  - residual_roi, spout_unique_maps, lick_matched_readout, learned_compensation, within_lick_correction (stars);
  - fig_softmax fig10 (e53f1cd2).
  - Ipsi = image LEFT on every brain map checked: no flips; suptitles say so.

## Checked and fine

**Merge 2730033a + 1ba23601, the 15 module files taken from the rebuild.** For each one I ran
`git log 6c00efbb --not 31ec5c1a`. Main's side touched them only through 26fd5b44, 48f2cc6d, 4a37cbc8 and cec38954.
b67e3dbd (Wt) is docs only. Checked on a8f1651d:
- **Jaw-only out of working (48f2cc6d).** `working_trials.WORKING_CLASSES`, `selection()`, the tag suffix
  `-cls-contact-incomplete`, and `_table(sel)` narrowing by `in_gate(cls, "camera_lick")` are all present.
  `trial_classes.GATES` / `run_selection` / `camera_select` carry the class part.
- **The B / C / D / G defaults (4a37cbc8)** are present:
  - `SECTION_D_SCORED` default working;
  - `SECTION_G_GROUPS` default camera;
  - `STATE_DISC_CENTRE` default working;
  - `STATE_DISC_DEFAULT = session_z` (B stays OFF, as DECISIONS 2026-11-04 B says).
- **Projection centre.** The default is `wt`, which matches the later 11-04 Wt decision (supersedes main's
  "engaged").
- **August-only pre (26fd5b44).** The profile env plus 3284cc25 restore it when no profile is active (decision 1
  resolved, Priya "ok").
- **grant_similarity.** The `_class_select(..., cls=bd.get("cls"))` port is equivalent to main's `work=`.
- **Line comparison.** Main-added lines that are missing from the merged files (scratch diff) are API-port differences
  (`work=` -> `cls=`, `wt.row_masks` -> live classes) or docstrings. I found no dropped rule or default.
- **Not checked.** I did not verify every ported hunk line by line beyond the default / env checks and the
  main-added-line comparison.

**Today's code commits:**
- **24d604cc `fr_trial_peth` now reads fr_A.** The model set is the same: the new fr_A `tr_models` and the old
  fr_A_tr `tr_models` are both `[full, fr:full, fr:alone, fr:drop:task / instructed / uninstructed / position]`
  (checked on PS92_0806 json). `fr_groups.run_folder` keeps the legacy fallback.
- **24d604cc `combined_encoding_figures` subset kind.** Stripping the `_exec-` suffix routes `_subset-trmodels_exec-...`
  and `_subset-shufpos-/droppos-_exec-...` to their intended suites. Orders without those subsets resolve the same as
  before.
- **24d604cc, other files.** The `--jobs` / `--bundle` changes are mechanical.
- **19d3cf71 encoding_mixed `_run`.** It reads suffixed runs only. The main run now carries LICKS_ALL=split and the
  polar exec, which is the 11-04 production design (option B). cam4 measures are absent, and that was resolved by the
  11-05 binned decision.
- **5a6b7986 `ttc_log`.** It is a non-default sensitivity; `DEFICIT` stays `ttc`.
- **76066e01.** The binned panels match DECISIONS 2026-10-24 (later).
- **5993d70e `--bin`.** The default stays 8 px (`MUSALL_DOWNSAMPLE = 8`). Non-default sizes are suffixed. This matches
  Priya's 11-05 8 px choice for the Figs 7 replacement.
- **b1ed55e7 / d39ee11c library moves.** Every removed script line reappears in the library (line comparison). No logic
  changed.
- **The rest:**
  - 5bd7faa2 (relabel output root) and a6edab62 (bundle_dir) fix paths only.
  - cc4855d8: per-session lookup, with the same hashes reported.
  - 7fdc0261 implements decision (f) 8.
  - e53f1cd2 rewrites the fig10 title from this render's numbers and drops the top spine.
  - dfd13345 changes date sorting only.
  - 411c8385 is the one-detector decision (f) 5.
  - 49f49e23 restricts the cross-day intensity to cohort 2 only, so it is not pooled with PS92-95.

**Render queue vs ENCODER_RENDER_COMMANDS and the decisions:**
- **Frames.**
  - Every A / B / Z step reads `gate-attempted_v7g3_t3d_spont*` (spont primary, 11-04 ~20:55).
  - Readers without `--tag` resolve through `frozen_encoder.design_tag`. Checked in frozen_fr_compare,
    frozen_fr_musall(_v2), remap_lmm / checks, movement_null_plan_maps, frozen_peth_trial_maps, frozen_kernel_maps,
    exec_vs_neural_match, frozen_claims, contest_lines, layer2_*, epoch_kernels, specificity_control, matched_residual,
    relabel_confusion, residual_roi, frozen_residual_trend and fig_pooled.
- **te.**
  - `FROZEN_DESIGN=te FROZEN_LICKS_ALL=none` (the env tokens are applied in order, so `none` overrides the
    re-exported `split`).
  - The te design has its own pre-cue / ITI lick families, as DECISIONS 2026-10-26 intends ("ITI LICKS INCLUDED").
  - te_readout defaults to the non-spont te tag.
  - The t_frame_te log env matches ENV-frA plus te.
- **lsland / frlag.**
  - The variant lists equal S1 `run_land_lag.sh` (10-25).
  - lsland is env frozen + LICKLAND; frlag is env frA + LAGS, without lickspout.
  - contest_lines reads only `no_target` / `no_lickspout` / lag variants, so `no_spont` is not needed.
- **PRE_FROM.**
  - The profile sets 0801 globally.
  - It is unset only in june_august_drift, rest_frozen_decoder_nl, map_drift_prepre and restnl_mask, as DECISIONS
    2026-11-03 (late) lists (see finding 6 for the 12.1b wording).
- **Gates.** The profile carries:
  - `WIDEFIELD_TRIAL_GATE=camera`, `WORKING_TRIALS=in_working`, `TRIAL_STOPPED=both`;
  - the Wt centres;
  - `LP_FROZEN=lp`, with `--trained lp` on lp_final decoders (D7).
  `WIDEFIELD_GATE_SOURCE=lp` on attempt_maps is inert under the camera gate: classes come from `trial_classes`, and no
  DAQ quit mask is built.
- **Contest measures.** fig_pooled, fig_goal_summary and fig_softmax use `nokin:u:tgtbin` / `u:exbin` (binned primary).
  encoding_mixed is fixed. No other queued script reads polar `u_target`.
- **Residual significance.** The queued frozen_residual_significance runs at 8 px, in canonical spout order. It is the
  agreed replacement for Figs 7 / 7c / 7d (Priya 11-05 ~18:20).
- **ENL window.** `WIDEFIELD_ENL_WINDOW_S=1.0` for attempt_coding `--freeze` and the frozen 1.0 s ENL models follow
  10-24 / 11-04 (b).

**Not covered:**
- the ITI / E-section scripts' internal choices beyond the windows;
- the combined_encoding_figures main suite's contest panels (they come from section C, already rendered);
- MASTER_MANIFEST row by row.

## Needed by the pages / decks, in no queue
Only page or deck content, from the plan doc's "not placed" list plus the findings above:
- the robustness rows of finding 4;
- Fig 0.2 on the camera working period (finding 3);
- the lick ENL families with the sliding window (finding 2);
- **rebuild_order_B consumer steps 3-15:** lick_cued_ramp, precue_restw / restnl / revup / cue_to_lick, enlv /
  enl_angle, the balance family on the KEY preds, decoder refits, lick_code, session-level maps, interhemi,
  lick_correction -> prelick_*, dock subsets;
- B11 panel, G4 figure, slide-95 raw + z pair, miss_vs_stopped, quiet events (G4), 15e preprocess maps;
- fig_decoding_sessions, fig_state_ps94_diagnostic, fig_enl_move;
- A14, the per-class encoder figure on the main encoding page (Priya 11-05). I did not find a producer step for it in
  the queue.
