# Figure provenance, cohort-1 rebuild (_daqlk), 2026-11-06

Priya 11-06: "ensure for all figures we are clear in analyses / captions / deck notes all of these variables -- what
window was analyzed, gated on LP vs DAQ licks, what trial period and how it was defined, what trial types and how
they're defined, etc."

The table is `FIGURE_PROVENANCE_20261106.csv`, beside this file. It has one row per item in the replacement-pass map:
page figures, inline images, LMM toggles, deck slides and NEW items.

**Updated 11-06 ~15:45: all six script groups are merged** (group A findings at the end). Counts: 45 CONFIRMED, 635 INFERRED,
152 UNKNOWN, 69 automatic caption conflicts; INFERRED rows turn CONFIRMED from the render logs when it finishes (section 7).

## 1. Method

1. **Figure list.** The figure list is `REPLACEMENT_PASS_MAP_2026-11-05.csv` (wf_render_v8f
   docs/status/reanalysis_20261105). It covers every live encoding / imaging / kinematics / behaviour figure and the
   PI deck PI_overview_2026-11-02 (187 slides). Rows inherit their source as follows:
   - **Deck slides** inherit from the page figure named in their FIGURE SOURCE note (`Encoding (goal) page Fig X`,
     `#f-id`).
   - **Hidden LMM companions** inherit from the main slide before them, or from the "main slide N" their note names.
   - **Inline images** inherit from their parent figure.
   - **LMM toggles** inherit from the figure with the same id.
   - **No script named in the map:** the script is read from the caption's "Script:" text. Failing that, it is found
     by grepping the code for the figure stem (`script_source` = manual).
2. **Producers.** Producers come from the render queues `queue_daqlk{7b,8,8b,8c,8d,8e,8e2,8f,8g}.sh` (scratchpad
   `render_daqlk`).
   - **Which queue owns a step name:** the queue holding its `RESERVED_v*` claim, or the queue named in its log.
   - **Step env:** read from the `# env` lines of `logs/<step>.log` when the log exists (`env_source` = log).
     Otherwise it is expanded from the step's env tokens with the queues' `apply_env` (noFR / frA / frozen /
     VAR=value / unset:VAR, plus `$PLAIN` / `$DE` / `$TEENV`), on top of the render profile and its `block_env`
     (`env_source` = queue tokens).
   - **Scripts with several steps:** matched to the figure by the robustness tokens in the figure stem and folder
     (plain, z, restnl, frA, cue / lick, ...). `step_match` records how: single / heuristic / all-variants / none.
3. **Definitions.** Code defaults per script come from docstrings, constants, argparse defaults and env reads in
   wf_render_v8f @ 110c4efd. They were extracted by six read-only helper passes (scratchpad `prov/agent_*.tsv`). On
   top of these, each row carries the run-level facts that the step env fixes:
   - LP lick source
   - a pre set the block unsets
   - plain basis
   - hemo variant
4. **Captions.** Captions are read from the live pages:
   - encoding: `wf_render_v8f/_goal_page/index.html` (11-06 13:26)
   - imaging: the live artifact, version 1793740803-1eef
   - kinematics / behaviour: `M:/.../kin_page_src`, `behaviour_page_src`

   The automated conflict checks are in section 6.
5. **Status.**
   - **CONFIRMED:** every mapped step is done, its env was read from the log, and the code gave window, licks, trials
     and stats.
   - **INFERRED:** the step is pending or running (env from queue tokens), or the figure is not a `_daqlk` render step
     (kept or behaviour-only; code defaults only).
   - **UNKNOWN:** no producing script, the script was not characterised, or a key field is unknown. The reason is in
     `status_note`.

Scripts: scratchpad `prov/parse_steps.py`, `prov/merge.py`. Nothing was run, rendered or edited outside these two files.

## 2. Fields (CSV columns)

| column | meaning |
|---|---|
| row, output, location | map row; page or deck; `Fig N.k` / `slide n (hidden)` |
| fig_id, png_stem | page anchor or deck shape name; image stem |
| inherits_from | the page figure, main slide or parent this row takes its provenance from |
| producing_script, script_source | script(s); where the script name came from (map / caption / manual) |
| render_steps, step_match, code_sha | `step [queue, status]`; how the step was matched; code sha from the log header |
| env_source, env_overrides | log or queue tokens; selection-relevant env that differs from the profile (e.g. `WIDEFIELD_LICK_SOURCE=lp`, `WIDEFIELD_AREA_BASIS=plain`, `WFIELD_LICK_DECODER`, `WFIELD_ITI_FILTER`, `FROZEN_*`, `QS_LICK_SOURCE=daq+lp`, `DISENGAGEMENT_QUIT=daq`, unset PRE_FROM) |
| run_tag | the output tag (section 3.1), or "not a _daqlk render step" |
| align_window | alignment and window in s |
| lick_source | DAQ contact detector / LP video / both, and WHERE it enters (classes, alignment, ENL gate, quiet exclusion) |
| trial_period | camera working period / DAQ quit-tail / behaviour-pipeline engaged / whole session, plus the defining code |
| trial_types | classes included (names from sections 3.2-3.3) |
| engagement_gate | the gate function in force |
| pre_baseline | August-only (0801 env / 0806 cohort cut) or June + August |
| sessions_animals | sessions and animals, plus min-trial rules |
| basis_targets, hemo_variant | Allen split / plain / pixels / LocaNMF / frozen frames; hemo variant |
| statistic | test and multiple-comparison correction (names from section 3.11) |
| code_refs, prov_notes | file:line of the definitions; caveats from the code reading |
| data_text | DRAFTED "Data:" paragraph for the caption (template in section 4) |
| caption_found, caption_conflicts | whether a caption was found; automated conflicts (section 6) |
| rerender, status, status_note | `rerender` from the map; CONFIRMED / INFERRED / UNKNOWN and the reason |

## 4. The standard "Data:" block (template)

> **Data:** *Window:* [aligned to cue / first lick / ENL], [start]-[end] s[; ENL = the [1.0|2.0] s before the cue
> free of [DAQ contact|video] licks, sliding to the latest lick-free gap]. *Licks:* trials classed from DAQ spout
> contact + video licks (cam4 LP + cam1); [windows / alignment use the [DAQ contact|LP video|cam4 3-D video] lick
> train]. *Trials:* [camera working period (to the session's last camera lick; stopped runs, A2 + excise, out) | DAQ
> quit-tail (robustness) | behaviour-pipeline engaged trials | every post-cue lick (no gate) | whole session]; classes
> [contact, incomplete (= attempted / working) | + jaw_only | no_response | stopped]. *Pre:* August pre-stroke
> sessions only (June excluded)[; June + August by design for this block]. *Sessions:* [n per animal x epoch],
> PS92-PS95. *Signal:* hemo meegkit_hpfit_lag_refitT_swapmask; [Allen areas, ALM / tjM1 split | plain Allen | pixels
> (8x grid) | LocaNMF (projected) | frozen encoder frames <tag>]. *Stats:* [test]; [max-statistic over N bins |
> Bonferroni over N | Holm across the figure's cells (LMM stars) | BH-FDR over N | uncorrected]. *Run:* _daqlk, step
> [step] (code [sha])[; overrides ...].

Deck notes carry the same block. The slide shows the main point; the block goes in the notes.

## 3. Canonical definitions (exact wording to use, and where the code defines each)

Code refs are to `wf_render_v8f` @ 110c4efd (the v8f/v8g render code). The v7b lane ran a76352f3, v8/v8b 5db29b07; the
definitions below are the same in all three unless a row's `code_sha` says otherwise.

### 3.1 The run tag

`hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_pre0801_sel-camera-cue_stop-endlick-cls-contact-incomplete-stopA2jexrt3500c70_daqlk`
(`wfield_local/run_mode.py` `_ALL`, L187; profile `configs/render_profiles/cohort1_rebuild_20261103.yaml`):

| part | meaning | defined in |
|---|---|---|
| `hv-meegkit_hpfit_lag_refitT_swapmask` | hemo variant: meegkit high-pass fit, 415 nm advanced 16 ms (lag), T refitted, swapped-frame mask | profile `WIDEFIELD_HEMO_VARIANT`; `config.hemo_tag` (config.py L263) |
| `lnmfproj` | LocaNMF components = projection of the corrected data onto the stored per-session / joint footprints (no refit) | profile `WIDEFIELD_LOCANMF_PROJECT=1`; `run_mode.lnmf_part` L57 |
| `pre0801` | August-only pre-stroke baseline (see 3.9) | profile `WIDEFIELD_PRE_FROM=0801`; `config.pre_from` L511 |
| `sel-camera-cue_stop` | camera trial classes, class window cue + 70 ms .. min(cue + 3.5 s, trial stop) | `WIDEFIELD_TRIAL_GATE=camera`, `WIDEFIELD_TRIAL_WINDOW=cue_stop`; `trial_classes.run_selection_tag` L1251 |
| `endlick` | camera working period ends at the session's last camera lick | `WIDEFIELD_TRIAL_END=lick`; `trial_classes.DEFAULT_END` L104 |
| `cls-contact-incomplete` | working = contact or incomplete (jaw-only is its own class) | `WIDEFIELD_WORKING_TRIALS=in_working`; `trial_classes.sel_classes` L397, `working_trials.WORKING_CLASSES` L51 |
| `stopA2j` | stopped rule A2 ("both"), jaw-only counts toward a stop run | `WIDEFIELD_TRIAL_STOPPED=both`; `trial_classes.stopped_rule_name` L1182 |
| `ex` | stopped trials touched by a lick are excised (`stopped_licking`, neither class) | `trial_selection.stop_licks: excise`; `trial_classes._lick_token` L1221 |
| `rt3500c70` | class window max_rt 3.5 s, post-cue floor 70 ms (lick peak) | `trial_classes._window_token` L1228 |
| `daqlk` | imaging DAQ licks from THE behaviour DAQ detector (0.5 V offset, 40 ms floor, no 0.10 s refractory) | `config.lick_detector` L713, `LICK_DETECTOR_TAG` L705; `run_mode.lickdet_part` L176 |
| `LP_licks_` prefix (some outputs) | the step ran with `WIDEFIELD_LICK_SOURCE=lp` | `lp_licks.output_tag` L275 |
| no `pre0801` | the block unsets `WIDEFIELD_PRE_FROM` (June + August pre): `june_august_drift`, `rest_frozen_decoder_nl`, `map_drift_prepre`, `restnl_mask` | profile `block_env` |

Frozen-encoder frames carry their own design tag, not a run-tag part: `frozen_encoder_gate-attempted_v7g3_t3d_spont[_frA|_basis|...]`
(profile `FROZEN_LICKS_ALL=split`). The robustness row without spontaneous lick kernels is `..._v7g3_t3d` (no `spont`).

### 3.2 Trial classes (one class per trial; first match wins)

`wfield_local/trial_classes.py` module docstring (L1-75), `CLASSES` L80, `session_trials` L224, `classify_session` L730.
Window = the RESPONSE WINDOW, cue + 70 ms .. min(cue + 3.5 s, trial stop): video licks by their PEAK, DAQ contacts by
their onset, the jaw by its onset. A lick whose peak falls in cue .. cue + 70 ms is neither pre- nor post-cue
(`POSTCUE_START_MS` L98, `EXCLUDED_CONTEXT` L100).

| class | wording to use | code |
|---|---|---|
| contact | "contact: the spout-contact sensor (DAQ) registered a lick in the response window" | docstring L6 |
| incomplete | "incomplete: a video lick (cam4 Lightning Pose, or cam1-only; water droplets removed) or cam1 tongue, but no spout contact" | docstring L7-8 |
| jaw_only | "jaw-only: no lick; the jaw moved after the cue (cam4 or cam1) -- a reaction, not an attempt; its own class, outside working" | docstring L9-10 |
| no_response | "no response: no lick, no contact, jaw judged still; at close_center / close_mid (jaw not judgeable) a trial with no lick, contact or tongue defaults to no response (`jaw_judged` = False)" | docstring L11-16, `JAW_UNJUDGED_POSITIONS` L92 |
| unknown | "unknown: no lick and the jaw could not be judged, or no video" | docstring L17-18 |
| stopped | "stopped: part of a sustained stop -- a run of >= 6 consecutive no-response / jaw-only trials before the last camera lick (rule A2: kept in working when the run has no reference-spout trial and the nearest reference-spout trials on both sides are contacts), plus every trial after the last camera lick (the quit tail); a run starting < 300 s after the first cue does not count" | L485-508 (rule text), `_classify_guarded` L950, `stop_min_start_s` L1152 |
| stopped_licking | "stopped trial touched by a lick: excised, in neither stopped nor working" | L513-523 |
| after_end | "after the session's last camera lick (working-gate trial past the end anchor): neither working nor stopped" | `AFTER_END_CLASS` L534 |
| early_run | "a qualifying stop run that started < 300 s into the session: neither class" | `EARLY_RUN_CLASS` L1061 |

### 3.3 Groupings: attempted, working, camera working period

| term | wording to use | code |
|---|---|---|
| attempted | "attempted = contact or incomplete (a lick was made, whether or not it reached the spout)" | `GATES["attempted"]` L81-87; encoder gate `OROFACIAL_GATE=attempted` (frames `gate-attempted`) |
| working (production) | "working trials = contact or incomplete, inside the camera working period (up to the session's last camera lick; stopped / stopped-licking / early-run / after-end / unknown trials excluded); jaw-only is reported as its own class" | `sel_classes("working")` L397, `in_sel` L410, `in_working_period` L1300 (`OUT_OF_WORKING_PERIOD` L1297) |
| miss while working | "miss while working = working minus contact (incomplete)" | `sel_classes("miss_working")` L397 |
| camera working period (trial selection) | "camera working period: from the session's first trial to its last camera lick (contact or incomplete), minus stopped runs" | `before_session_end` L422, `classify_session` L730 |
| camera working period (rest baselines) | "rest baseline taken inside the camera working period = cues from the session's first to its last camera lick, inclusive" | `camera_engaged_period` L660 (`rest_start: first_lick`, defaults.yaml L81) |
| camera quit (A6) | "camera quit = first cue of the quit tail (after the last camera lick) when the tail holds >= 6 cues and starts >= 300 s into the session; otherwise right-censored" | `camera_quit` L1329 |
| jaw-inclusive working (robustness) | "working incl. jaw-only (in_working_jaw)" | `WIDEFIELD_WORKING_TRIALS=in_working_jaw` |

### 3.4 DAQ quit-tail (the pre-rebuild rule; robustness rows and measures of engagement only)

Wording: "DAQ quit-tail: trials before the session's final, non-recovering collapse of the reference-spout (close_L /
close_center) response rate below 0.5 over a trailing 15 reference trials, backdated to the trial after the last
reference response; 'working' = DAQ licking trials + misses before the quit." Code:
`precue_engagement_states.engagement_gate` L212 (`REFERENCE` L81, `WINDOW` L85, `MIN_RATE` L86). Under the camera gate it
REFUSES (`trial_classes.refuse_daq_gate` L590) unless the caller opens `trial_classes.daq_quit_rule(reason)` (L555): a
measure OF engagement (quit point, S28 quit = `engagement.session_quit` L119, engagement decline, attempt rate), a
labelled DAQ robustness row (e.g. step `l_disengage_daq`, `DISENGAGEMENT_QUIT=daq`), or bookkeeping a camera caller overrides.
The behaviour pipeline's own engaged gate (`spout_behavior.reference_engagement` / `orofacial_cohort.engagement_keep`,
behaviour trial table `engaged` column; `behavior.engagement` in defaults.yaml L227-232) is a THIRD gate: behaviour and
kinematics pages that were not re-rendered still use it.

### 3.5 Lick sources

| term | wording to use | code |
|---|---|---|
| DAQ contact licks | "licks = DAQ spout-contact detector (behaviour detector: 2.5 V onset / 0.5 V offset, 40 ms floor, no refractory)" | `config.lick_detector` L713; `defaults.yaml lick_detection` L203 |
| LP video licks | "licks = video (Lightning Pose) licks: cam4 LP lick onsets + cam1-only events (water droplets removed); first lick timing falls back to the DAQ contact at close spouts when LP is > 50 ms late" | `lp_licks` docstring L1-40, `LP_DAQ_TOL_MS` L65 |
| LP lick source switch | `WIDEFIELD_LICK_SOURCE=lp`: lick-aligned windows start at the first LP lick, the lick / no-lick arms and the ENL lick-free gate use the LP train | `lp_licks.lick_source` L73, `session_lp` L471 |
| LP-inclusive gate | `WIDEFIELD_GATE_SOURCE=lp`: the (DAQ-rule) engagement gate counts DAQ contact OR an LP attempt as a response; no effect on classes | `gate_source.py` L1-40 |
| camera trial classes | the camera gate itself always uses BOTH: DAQ contact (class contact) and video licks / jaw (incomplete, jaw-only) | `trial_classes` (3.2) |
| quiet-event lick exclusion | quiet frames exclude +-1.5 s around licks: DAQ contact (`QS_LICK_SOURCE=daq`) or DAQ contact + LP video licks (`daq+lp`, output `v2_quiet_lp`) | `scripts/cohort2_qc/quiet_spikes_v2.py` L53-57, L117 |

So for every figure state BOTH where licks enter the trial SELECTION (camera classes = DAQ + video, unless the figure is
on the DAQ quit-tail) and where they enter the WINDOW / ALIGNMENT (DAQ train unless `WIDEFIELD_LICK_SOURCE=lp`).

### 3.6 ENL window (pre-cue, no licks)

Wording: "ENL window = the [1.0 | 2.0] s ending at the cue, free of [DAQ contact | video] licks; a trial with a lick in it
slides back to the latest lick-free gap after the spout strobe, and is dropped if none exists."
`lp_licks.enl_post_s` L185: `WIDEFIELD_ENL_WINDOW_S` if set; else 1.0 s under the LP lick source (`LP_ENL_WINDOW_S` L172);
else `decode.precue_post_s` = 2.0 s (defaults.yaml L109). Slide rule: `enl_window.precue_window_start` L56; time-based
callers `lp_licks.enl_window_starts_s` L238 with the lick train from `lp_licks.gate_licks_s` L221. The ENL window is
not a baseline: the pre-cue BASELINE subtracted from cue / lick windows is `decode.precue_baseline_s` = 1.0 s
(defaults.yaml L141), never used with `align: precue`.

### 3.7 Analysis windows (decoders / maps)

`configs/defaults.yaml decode` L105-142: `cue_post_s` 2.0 (cue .. cue + 2 s), `lick_post_s` 2.0 (first lick .. + 2 s;
the first lick of the run's lick source), `precue_post_s` 2.0 (ENL, see 3.6), `precue_baseline_s` 1.0,
`max_rt_s` 3.5 (defaults.yaml L177: a trial with no lick within 3.5 s of the cue is a no-lick trial). Per-alignment
resolver: `lp_licks.post_s_for` L208. Scripts that override these name it in their row.

### 3.8 Response window and hit rate

"Hit = a DAQ spout contact within the session's response window (3.5 s; `daq_trials.response_window_s`)" for the behaviour
pipeline. Under the camera gate the hit-rate numerator is camera class contact inside the camera working period
(`trial_classes.camera_period_trials` L1305).

### 3.9 Pre-stroke baseline

Wording: "pre = August pre-stroke sessions only (June excluded: the animals were still learning)". Two spellings exist and
select the same sessions: the run-mode env `WIDEFIELD_PRE_FROM=0801` (`config.pre_from` L511; names the `_pre0801` folders)
and the cohort cut `baseline_pre_from: "0806"` (configs/cohorts.yaml L43; `config.baseline_pre_from` L488,
`config.early_pre` L500), which are identical because 0805 is date-policy-excluded and 0806 is the first curated August
day. Blocks that keep June + August by design (profile `block_env`): `june_august_drift` (Fig 4.3 drift),
`rest_frozen_decoder_nl` (12.1b), `map_drift_prepre` (Figs 7.2 / 7.4 / 7.6: June vs August), `restnl_mask` (RESTnl
baseline build). Epoch boundaries always come from the full pre set (`epoch_audit.full_cohort` suspends PRE_FROM).

### 3.10 Area basis: Allen vs ALM / tjM1 split

Wording: "Allen areas with the ALM / tjM1 split (ALM = anterolateral 40 % of MOs, tjM1 = anterolateral 50 % of MOp, per
hemisphere; disjoint rests MOs_rest / MOp_rest; ROI version 8)" = PRIMARY; "plain Allen areas (whole MOs / MOp, no ALM /
tjM1)" = robustness. `wfield_local/area_basis.py` docstring + `SPLIT_PAIRS` L59, `ROI_SPEC` L49; ITI chain switch
`WFIELD_AREA_BASIS` (`scripts/iti_licks.py` L190-197; plain outputs carry `_plain`). Other targets: pixels (8x
down-sampled grid of the stat mask), LocaNMF components (`lnmfproj`), frozen encoder frames (frame tag).

### 3.11 Multiple comparisons

| method | wording | code |
|---|---|---|
| max-statistic | "family-wise corrected by the bootstrap max-statistic over N in-mask bins" | `beta_maps.CORRECTION = "maxstat"` L1187 (branch L1105, L1545); text `epoch_maps` L418 |
| Bonferroni | "Bonferroni over N tests (e.g. 6 positions x 3 epochs = 18)" | `epoch_maps` L425 (only when CORRECTION is changed), `poststroke_compare` L1442 (`bal_p_bonferroni_x8`), epoch_accuracy CI stars |
| Holm (LMM stars) | "* p < .05 uncorrected; ** Holm-adjusted p < .05 across the figure's cells" | `analysis_kit.lmm_stars` L371, `holm` L282; test `analysis_kit.epoch_change_lmm` L299 |
| BH-FDR | "Benjamini-Hochberg FDR q < .05 over N tests" | `analysis_kit.bh` L251, `orofacial_stats.bh_fdr` L26, `lick_templates.bh` L227 |
| LMM | "value ~ epoch + (1 | animal) + (1 | animal:epoch), separate animal x epoch variance per post epoch, REML; t on df = animals - 1" | `analysis_kit.epoch_change_lmm` L299 |

## 5. Counts (this pass)

| output | rows | CONFIRMED | INFERRED | UNKNOWN |
|---|---|---|---|---|
| PI deck 11-02 | 187 | 8 | 108 | 71 |
| behaviour page | 7 | 2 | 1 | 4 |
| encoding (goal) page | 439 | 28 | 369 | 42 |
| imaging page | 117 | 6 | 101 | 10 |
| kinematics page | 82 | 1 | 56 | 25 |
| **all** | 832 | 45 | 635 | 152 |

Rows with a mapped render step: 515. CONFIRMED rows: encoding Fig 0.3, encoding Fig 0.4, encoding Fig 0.9, encoding Fig 0.10, encoding Fig 0.11, encoding Fig 1.1, encoding Fig 1.7 [collapsible], encoding Fig 1.8 [collapsible], encoding Fig 1.9 [collapsible], encoding Fig 1.10 [collapsible], encoding Fig 3.11 [collapsible], encoding Fig 3.12 [collapsible], encoding Fig 4.6, encoding Fig 7.3 [collapsible], encoding Fig 9.2, encoding Fig 9.3, encoding Fig 12.16 [collapsible], encoding Fig 12.22 [collapsible], encoding Fig 13.6, encoding Fig 17.1, encoding Fig 17.2, encoding Fig 17.3, encoding Fig 17.4, encoding Fig 17.5, encoding inline image after Fig 12.22, encoding inline image after Fig 13.6, encoding inline image after Fig 13.6, encoding inline image after Fig 13.6, imaging Fig 0.2, imaging Fig 5.2, imaging Fig 8.1, imaging Fig 8.1b, imaging Fig 8.2, imaging Fig 8.2b, kinematics Fig 7.6, behaviour Fig 1.2, behaviour LMM toggle image, PI slide 4 (visible), PI slide 5 (hidden), PI slide 35 (visible), PI slide 89 (visible), PI slide 92 (hidden), PI slide 137 (visible), PI slide 184 (visible), PI slide 185 (hidden).

UNKNOWN reasons: missing (70); no producing script named (map / deck note) (65); script not characterised (17).

## 6. Caption conflicts

### 6a. Automated checks (live captions vs the _daqlk provenance; detail per row in `caption_conflicts`)

- **caption says Bonferroni; producer uses the max-statistic** (8): encoding Fig 3.9, encoding Fig 21.3, encoding Fig 21.12, imaging Fig 2.5, imaging Fig 2.7, imaging Fig 14.4, imaging Fig 14.5, imaging Fig 14.6
- **caption says June + August pre; the _daqlk run is August-only (pre0801)** (48): imaging Fig 1.1, imaging Fig 1.2, imaging Fig 2.1, imaging Fig 2.2, imaging Fig 2.3, imaging Fig 2.4, imaging Fig 2.4b, imaging Fig 2.5, imaging Fig 2.6, imaging Fig 2.7, imaging Fig 2.8, imaging Fig 3.4, imaging Fig 4.2, imaging Fig 4.4, imaging Fig 6.1, imaging Fig 6.2, imaging Fig 6.3, imaging Fig 7.1, imaging Fig 7.1b, imaging Fig 7.3, imaging Fig 7.3b, imaging Fig 7.5, imaging Fig 7.5b, imaging Fig 7.7, imaging Fig 7.7b, imaging Fig 7.8b, imaging Fig 7.10, imaging Fig 7.10b, imaging Fig 7.12, imaging Fig 7.14, imaging Fig 7.16, imaging Fig 7.16b, imaging Fig 7.18, imaging Fig 7.18b, imaging Fig 7.19b, imaging Fig 7.21, imaging Fig 7.23, imaging Fig 10.1, imaging Fig 12.1, imaging Fig 14.1, imaging Fig 14.2, imaging Fig 14.3, imaging Fig 14.4, imaging Fig 14.5, imaging Fig 14.6, imaging Fig 14.7, imaging Fig 14.8, imaging Fig 14.9
- **caption describes the DAQ quit-tail / engagement gate; the _daqlk run selects trials with the camera working period** (47): imaging Fig 2.1, imaging Fig 2.2, imaging Fig 2.3, imaging Fig 2.5, imaging Fig 2.6, imaging Fig 2.7, imaging Fig 2.8, imaging Fig 3.1, imaging Fig 4.1, imaging Fig 4.2, imaging Fig 6.1, imaging Fig 7.1, imaging Fig 7.1b, imaging Fig 7.2, imaging Fig 7.2b, imaging Fig 7.3, imaging Fig 7.3b, imaging Fig 7.4, imaging Fig 7.4b, imaging Fig 7.5, imaging Fig 7.5b, imaging Fig 7.6, imaging Fig 7.6b, imaging Fig 7.7, imaging Fig 7.7b, imaging Fig 7.8, imaging Fig 7.8b, imaging Fig 7.9, imaging Fig 7.9b, imaging Fig 7.10, imaging Fig 7.10b, imaging Fig 7.11, imaging Fig 7.11b, imaging Fig 7.12, imaging Fig 7.13, imaging Fig 7.14, imaging Fig 7.15, imaging Fig 10.1, imaging Fig 10.2, imaging Fig 13.1, imaging Fig 13.2, imaging Fig 14.1, imaging Fig 14.2, imaging Fig 14.4, imaging Fig 14.5, imaging Fig 14.7, imaging Fig 14.8
- **caption defines working as licking + miss-while-working (DAQ rule); rebuild working = camera contact | incomplete** (46): imaging Fig 2.1, imaging Fig 2.2, imaging Fig 2.3, imaging Fig 2.5, imaging Fig 2.6, imaging Fig 2.7, imaging Fig 2.8, imaging Fig 3.1, imaging Fig 4.1, imaging Fig 4.2, imaging Fig 7.1, imaging Fig 7.1b, imaging Fig 7.2, imaging Fig 7.2b, imaging Fig 7.3, imaging Fig 7.3b, imaging Fig 7.4, imaging Fig 7.4b, imaging Fig 7.5, imaging Fig 7.5b, imaging Fig 7.6, imaging Fig 7.6b, imaging Fig 7.7, imaging Fig 7.7b, imaging Fig 7.8, imaging Fig 7.8b, imaging Fig 7.9, imaging Fig 7.9b, imaging Fig 7.10, imaging Fig 7.10b, imaging Fig 7.11, imaging Fig 7.11b, imaging Fig 7.12, imaging Fig 7.13, imaging Fig 7.14, imaging Fig 7.15, imaging Fig 10.1, imaging Fig 10.2, imaging Fig 13.1, imaging Fig 13.2, imaging Fig 14.1, imaging Fig 14.2, imaging Fig 14.4, imaging Fig 14.5, imaging Fig 14.7, imaging Fig 14.8


These are expected: the live captions describe the renders made before the rebuild. The replacement pass must REWRITE
these captions, not annotate them.

### 6b. Findings from the code reading (each needs a caption check at replacement)

1. **ITI / ENL licks are video, not the DAQ detector.** Every ITI and "cued" lick set in the lick chain comes from cam4
   video: the 3-D kinematics table (`iti_licks.py` L226, L264-399) or `licks_all.csv` / `licks.csv`, with times on
   the DAQ clock. DAQ contact appears only as a contact / miss label.
   - "[licks: DAQ spout contact]" on these figures is wrong.
   - The `lick_iti_prelick` docstring wrongly says "DAQ lick PEAK (contact)".
2. **The decoder "cued" set is not gated.** The cued lick set (`iti_licks.extract` tr_X) is every post-cue 3-D lick,
   capped at 1500 per session, with no working-period gate.
   - ITI sets are whole-session.
   - Only first-lick / PETH sets use the attempted gate (`OROFACIAL_GATE=attempted`, which reads trial_attempts.csv
     with no canonical cut).
   - `iti_lick_rate` is the only ITI script that uses `camera_period_trials`.
   - Captions must not say "camera working trials" for these sets.
3. **Stale caches.**
   - `iti_licks` session extracts are skip-if-exists, keyed on hemo / lnmf / drift / pre / projall only, so not on
     the gate or the lick source.
   - `l_c2l_restw` takes first licks from the rev-up cache built under the OLD gate. The restnl variant rebuilds them.
   - The RESTW pre-cue level cache holds an older engaged gate in 86 / 107 sessions.
   - `lick_enl_target_test` Part A reads attempt_coding files from 10-29 (LP source), which the render does not
     regenerate.
   - `lick_offset_where` and `movement_null_plan_maps` read inputs no render step produces.
4. **"plain" steps that are not plain.** Under 7b, `e_plain_where` / `e_plain_order` / `e_plain_decomp` /
   `e_plain_gauss` / `e_loclin_plain` ran in the SPLIT basis (the F8 mis-name). 8c re-ran them as true plain under the
   same names (`_plain` outputs), and 8c `e_split_where` now writes the split `where_decoder.csv`. Check the output tag
   before captioning.
   - Separately, 7b `e_contrib_areas` uses a plain-Allen decoder while `AREA_BASIS` stays split, so its masks are
     UNKNOWN.
5. **Corrections are mostly absent in the lick chain.** Most stats are `boot_delta` nested bootstrap 95 % CIs
   (animals then sessions, 4000 draws), uncorrected. The `iti_area_maps` per-area outlines are uncorrected. The
   exceptions:
   - **BH-FDR:** `lick_balance_order(_primary)` (over lick orders); `lick_cued_ramp` and the per-lag strips (over
     lags); the LMM scripts (within model x term).
   - **Holm:** `learned_compensation`, `face_prelaunch_aim`, `face_offset_partial`.
   - **`iti_session_pattern_tests`:** the solid outline is a bootstrap max-statistic, but the docstring says BH.
   - **`lick_cued_aim_by_spout`** (not rendered): Bonferroni over 18 tests plus BH over 7 panels.
6. **Stale window and rule wording.**
   - `iti_area_maps` docstring says the response is 0-0.5 s; the code uses 0-1.0 s with a -1.0..-0.5 s baseline.
   - `lick_enl_offset_explore` (L637, L766) and `lick_enl_target_test` (L764) titles still say "no lick in the last
     2 s"; the primary rule now slides the window.
   - The `lick_precue_restw` docstring says strict_w50, but the restnl default is rule wr.
   - `lick_rim_overlay` describes the v4 rim but runs ROI v8; `lick_roi_check` says ROI_FRAC 0.4 (v1).
   - `misdirected_lick_decode` renders "templates" while its docstring describes logistic.
7. **Where June enters.** It never enters the pre baseline, but:
   - The LocaNMF basis footprints (`iti_basis_decoder`) were fit on 06-06..08-14.
   - These use June as a reference epoch: `iti_where_drift --june`, `iti_june_ref`, `lick_code_subspace`,
     `lick_code_noisematch` and `reembedding_test` (PRE_FROM suspended).
   - `fig_enl_move` filters by curated labels only, so June may be in it (UNKNOWN). It also draws lines across epochs.
8. **ENL movement figures.**
   - `enl_movement_multisession` / `fig_enl_move` (LP blocks) use a 1.0 s ENL window; engaged = an LP lick within
     max_rt.
   - `enl_decode` (not rendered) hard-codes 2.0 s and the DAQ quit-tail.
9. **Module-level overrides.** `face_offset_partial` sets `WFIELD_ITI_FILTER=dock` itself, so it is docked-only
   whatever the step env says.
10. **ITI spout grouping.** `iti_where_scatter` groups by the spout nearest the landing; the other scripts use the
    previous trial's spout.
11. **gain_bio (slide 102).** Step `gb_sessions` passes `--task-rows working` (contact | incomplete cues only), while
    the docstring and the `gb_common` default say every cue in the working period. Quiet frames exclude DAQ + LP licks
    (`QS_LICK_SOURCE=daq+lp`).
12. **Far-contra trial figures** (`fig_goal_epochs`, `fig_goal_summary` fig2, decoder_retention, coupling, precue).
    - The classes come from LP video, but the frozen decoder was trained on DAQ contact (`attempt_coding --freeze`
      refuses LP). Do not call it LP-trained.
    - `fig_goal_summary` fig1 / fig2 suptitles hold fixed 10-24 readings.
13. **`exec_vs_neural_match --source encoder`.** Its default relabel tag is `v7g3_t3d_trial`, but the render wrote
    `v7g3_t3d_spont_trial`, so it may read a stale relabel run.
14. **`recovery_timing`.** The hit-rate denominator is every far-contra trial_attempts row, with no `canonical` filter.
15. **`epoch_behaviour`.** 1b uses the camera working period; the 1c time course still uses DAQ engaged trials.
16. **Kinematics.**
    - `jaw_only_incomplete_figs` (camera period only with `--trials`), the cam4 cue traces and `pose3d_cue_traces` use
      the behaviour-pipeline engaged gate (`orofacial_cohort.engagement_keep`).
    - `pose3d_phase` / `spout_error` / `summary_fig` / `vs_cam4` / `exec_bins_calibration` use every post-cue cam4
      lick, with no gate.
    - Pre is August-only (`--pre-from 20260801` / 0806).
    - `orofacial_deck` / `orofacial_w2` keep an optional `--baseline all` (June + August).
17. **`fig_farcontra_attempts_sessions` (slide 35).** The incomplete / jaw-only panels come from the older
    cam1_incomplete_licks table, whose gate is unknown.
18. **`fig_state_position_sessions`.** The state sidecar is now bal_em (11-03), not the session_z image on the page.
    Sessions are paired by order, not matched.
19. **Imaging epoch figures are the DAQ-contact arm.** The profile's block_env would set LP licks for `epoch_maps` /
    `epoch_accuracy`, but no step passes `--block epoch_maps|epoch_accuracy`. The `g5_epoch` log has no
    `WIDEFIELD_LICK_SOURCE`, so the lick class, lick alignment and ENL gate (2.0 s) are DAQ contact. In the camera gate
    (`camera_select`), the "lick" class is DAQ contact; only working / stopped come from camera classes.
20. **Maps are max-statistic.** `beta_maps.CORRECTION='maxstat'` (nested bootstrap, 2000 draws, 8x8 bins). Many
    docstrings still say "Bonferroni over 3,237 bins", and `within_spout_deviation` calls Bonferroni "the repo
    default". The per-position accuracy Bonferroni is over positions x post epochs drawn (18 with chronic), but the
    docstrings say twelve.
21. **`rest_frozen_decoder` (`g5_rest_frozen`) runs with no flags.** That means 50 permutations and no refit /
    train / duration / lick-gap matching. Page 12.1 describes `--perm 200 --refit --match-*`, and the page CSV name
    (durmatched_gapmatched) is not what the default run produces. `rest_frozen_decoder_nl` runs `--pre aug`
    (August-only), even though the profile comment says June + August.
22. **Other "stopped" / "engaged" definitions.**
    - `stopped_trial_traces`: DAQ terminal collapse.
    - `quiet_spikes_v2`: "engaged" = before the DAQ `quit_point`.
    - `disengagement_tests`: camera quit (A6).
    - All three differ from camera A2 stopped.
23. **`lp_final_render`.** It resets `WIDEFIELD_FROZEN_MODEL_DIR` to `<figures_working>/frozen_models_lpfinal` and
    adds `GATE_SOURCE=lp`.
24. **Chronic drift.** `chronic_drift_vs_reorg` with August-only pre spans about 10 days of pre, not about 2 months.
    It and `june_august_drift` seed their bootstraps from Python `hash()`, so results are not reproducible run to run.
25. **Section G** keeps PS92 / PS93 0817 (small-lesion), while the epoch families exclude them.
26. **`imaging_lmm`** reads `*_sessions_long.csv`, but `epoch_sessions_long` is not in the render, so the input may be
    stale. `position_coding_trend_recovery` reads fixed older encoder runs.
27. **No render step:** `cd_loo_*`, `cd_migration`, `cd_overlap_null`, `slowband_*`, `seizure_events_video` and the
    `quiet_*` scripts other than `quiet_spikes_v2`. Their figures are older runs.
28. **Working gate.** `trial_classes.GATES["working"]` includes jaw_only; production narrows it through `in_sel`. A
    script that reads `GATES["working"]` directly is jaw-inclusive.

## 7. Gaps (what the replacement pass must still fill in)

1. **Script group A not characterised.** Its rows (frozen encoder / FR / O2 / residual / Musall, most of encoding
   page sections 0-4 and 17-22) are UNKNOWN with "script not characterised". To fill them, finish that pass (the
   group file and the prompt schema are in scratchpad `prov/`), then re-run `prov/merge.py`.
2. **After the render finishes.** Most `_daqlk` steps are still pending (about 289 pending and 8 running at 13:50).
   - Re-run `parse_steps.py` + `merge.py`. This reads the `# env` lines and the code sha of every finished step and
     moves its rows to CONFIRMED.
   - Copy the session counts per animal x epoch from each log body (e.g. `e_iti_main`: 107 sessions) into
     `sessions_animals`.
   - Resolve the failed / skipped steps: `e_iti_hemo` (failed), `e_iti_hemo_sum` (skipped), and the `.failed`
     markers on `a_decomp_noFR_cue`, `a_relabel_noFR` and `b_imaging_lmm`.
3. **Step matching.** About 360 rows have `step_match` = heuristic or all-variants. A person needs to pick the variant
   that made the page image: plain vs split, `_z`, restnl vs restw, frA vs noFR, cue vs lick.
4. **Out of the _daqlk render.**
   - Code defaults only (INFERRED with "no _daqlk render step"): kinematics camera figures (kept, rerender = no).
   - UNKNOWN, not covered:
     - the FaceRhythm, cohort 2, VFSS, 2pRAM and histology slides
     - title slides
     - the NEW items (A11, A14, A17, B1-B16), which have no figure yet
5. **Missing producers.** These scripts are not in any worktree (scratchpad-only):
   - `fig_decoding_sessions_campllick_1103`
   - `aim_first_vs_all`
   - `fig_state_ps94_balem`
   - `build_behaviour_page_1103`
   - `make_video_clips`
   - `update_4cam_video_1104`
   - `crop_pos_maps_rows`
   - `mcherry_refined`
   - `red_detection_gallery`
   - `roicat_track`
   - `lick_direction_cells` (2pRAM)
6. **Checks not yet automated.**
   - Caption window length vs code (e.g. "2 s" vs the 1.0 s LP ENL).
   - The "[licks: ...]" tag vs the video-lick ITI chain (finding 1).

   Both need the group A rows and then a manual read.

## Group A (frozen encoder / FR / O2 scripts) findings, 11-06 ~15:45

1. fig_resid16 / fig_resid_spout captions say "nested bootstrap, Bonferroni"; the nested map test is the max-statistic
   (beta_maps.CORRECTION = "maxstat"). Their inputs (residual_sig_<F>px.csv, resid<F>_*.npz) have no repo writer -> stale
   (Figs 7 / 7c / 7d were replaced by the 8 px residual-significance figures, 11-05).
2. Bonferroni is right ONLY for: the per-animal "musall" tests (fig_resid_per_animal, frozen_residual_significance
   --per-animal, per-animal panels of frozen_pixel_unique_sig / spout_unique_maps); Allen-area tests (residual_roi,
   session_vs_frozen maps, frozen_peth_trial_maps); fig_frozen_refit_pooled (x9), fig_softmax_stats (x36).
3. Pooled pixel maps = max-statistic (16 / 32 px bins, family-wise per panel): frozen_pixel_unique_sig,
   frozen_pixel_musall, spout_unique_maps, fr_pixel_significance, fr_pixel_maps_sig, variance_maps;
   frozen_residual_significance = 8 px bins + cluster test.
4. Bars / lines (frozen_fr_compare, frozen_fr_musall(_v2), frozen_claims, contest_lines, te_readout, partial_refit,
   matched_residual, specificity_control, simple_key_figures): star = "CI excludes 0", NO multiple-comparison correction;
   remap_lmm / remap_checks: BH-FDR on LMM terms.
5. Pre: fig_frozen_refit_pooled docstring says June + August (11 / animal); g5_epoch ran PRE_FROM=0801 -> August-only.
   The frozen-encoder family is August-only via config.early_pre (cohort-1 cut 0806) on top of PRE_FROM.
6. Licks: frozen-encoder figures = LP video lick kernels (cam4 + cam1-only), DAQ contact = contact kernel, pre-cue / ITI
   LP kernels (FROZEN_LICKS_ALL=split); lick alignment = LP first lick (DAQ contact fallback at close spouts); trials =
   attempted (contact | incomplete) inside the camera working period. Decoder figures (fig_frozen_refit_pooled,
   fig_s_per_animal, joint_xsession) = imaging DAQ detector + camera working classes; joint_xsession's ENCODER arm does
   not use the camera rows.
7. Windows: frozen Musall cue 0..1.5 s, lick -0.1..1.0 s; frozen_relabel / session_design cue 0..1.0 s; per-session
   encoder (FR, variance_maps) cue 0..1.0 s, "lick" = within 0.2 s of an LP lick peak; evoked residuals lick 0..1 s vs
   -0.3..-0.1 s, cue 0.5..1.5 s vs -0.5..0 s; allsession_encoder / Layer 2 = all frames.
8. Inputs: fr_pixel_maps_sig needs fr_musall's pixel npz (fr_musall has no render step); FR / per-session / binned-contest
   inputs come from run folders without the _gate-attempted suffix (trial gate UNKNOWN); fr_encoding_summary's "enl" may
   be the old fixed 2 s pre-cue window if those runs predate 11-04. Not in the render (no step): fig_resid16,
   fig_resid_spout, fig_softmax, fig_softmax_stats, fr_direction_mechanism, fr_encoding_summary, fr_musall,
   fr_trial_peth, grant_encoder, imaging_page/s_per_animal.
9. epoch_kernels' "early_pre" = first half of the AUGUST pre sessions (not June).
10. fig_s_per_animal's suptitle says "engaged trials"; its inputs are working-trial families.
11. CORRECTION to an earlier note: the far-contra / attempt-coding decoders in this render are LP-TRAINED (the LP step froze
    its own models; manifest lick_source: lp, e.g. PS92 decoder_precue_..._4503e20e2ae1). Caption: "frozen pre-stroke
    decoder trained on LP-classified pre trials".
