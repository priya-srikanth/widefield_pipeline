# scripts/ index (generated)

Generated 2026-11-04 from the code on `main` by a throw-away script kept outside the repo (not committed). Do not hand-edit the tables; regenerate instead. One line per script, taken from the first paragraph of its docstring. `scripts/archive/` and `scripts/rest_migration/archive/` are left out (finished probes; see their READMEs).

**How the kind is assigned.** This is a heuristic: read it as a hint, not a verdict. In this order, first match wins:

1. **superseded**: the docstring's first paragraph says superseded, retired, obsolete or deprecated.
2. **pipeline step**: the library (`wfield_local/`) imports or runs the script, or a runbook names it. Archiving or renaming one of these breaks production.
3. **diagnostic / check**: the name or the docstring's first paragraph says check, audit, QC, compare, census, probe or verify.
4. **figure producer**: the script saves figures (page, deck or exploratory; the tags say which are cited).
5. **analysis (tables / numbers)**: everything else. Usually prints a table or writes a CSV.

Tags: *imported or run by wfield_local*; *in a runbook*; *cited by a results page* (a path named under `docs/results_pages/`); *cited by deck_registry*.

**415 scripts.** figure producer: 164, pipeline step: 83, analysis (tables / numbers): 81, diagnostic / check: 87, superseded: 0.

No script on main currently says in its docstring that it is superseded, so that group is empty. Superseded OUTPUTS are tracked in the docs (docs/status, CUT_FIGURES) rather than in the scripts; a script tagged as a figure producer here may still make a figure that is no longer on a page.

Folders: top level, `cohort2_qc/`, `imaging_page/`, `rest_migration/`.


## scripts top level (297)


### figure producer (118)

| script | what it does | tags |
|---|---|---|
| [`a3_lmm_preview_fig.py`](a3_lmm_preview_fig.py) | PREVIEW of an LMM version of an epoch figure: per-session changes vs the animal's own pre (open markers, shape = animal, colour = epoch), LMM estimate (filled dot) + 95 % CI (t, df = n_animals - 1), a ... |  |
| [`cam1_incomplete_licks.py`](cam1_incomplete_licks.py) | Incomplete licks and jaw-without-lick trials with cam1 added (Priya 10-19/20; handoff to-do 4). |  |
| [`cd_loo_pooled.py`](cd_loo_pooled.py) | POOLED, within-animal normalised versions of the CD leave-one-out figures (`scripts/cd_loo_figures.py`; imaging page Figs 9c-e). | cited by a results page |
| [`contest_lines.py`](contest_lines.py) | Epoch-line figures for the 10-26 contest results that had only tables (ENCODING_FINDINGS 8.19; Priya 10-26: "which figure is this?"). |  |
| [`coupling_by_class.py`](coupling_by_class.py) | TRIAL-BY-TRIAL COUPLING SPLIT BY TRIAL CLASS: hits (lick reaches and touches the spout) vs failed attempts (incomplete licks, no contact). |  |
| [`cued_misdirected_licks.py`](cued_misdirected_licks.py) | MISDIRECTED far-contra licks: do far-right-trial licks that land LEFT look like left licks or right licks in cortex? (Priya 10-27: "PS94 has far R trials where the animal licks left. |  |
| [`decoder_retention_sessions.py`](decoder_retention_sessions.py) | FROZEN-DECODER RETENTION PER SESSION (Priya 10-25: session-level versions of the per-animal results; page Fig 7c). |  |
| [`early_predictors.py`](early_predictors.py) | DO EARLY CORTICAL FEATURES PREDICT THE CHRONIC OUTCOME PER ANIMAL? (Priya 10-25.) EXPLORATORY at n = 4. |  |
| [`encoding_key_figures.py`](encoding_key_figures.py) | Key-finding figures for the encoding / behaviour work (Priya 10-20: "are there other figures you could or should make to illustrate key and important points or findings"). |  |
| [`enl_fr_removed_decode.py`](enl_fr_removed_decode.py) | Is the CORTICAL pre-cue (ENL) position code face movement? Cortical ENL decoding after regressing out the FaceRhythm factors. |  |
| [`enl_movement_multisession.py`](enl_movement_multisession.py) | Pre-cue (ENL) position decoding after removing RUNNING + VIDEO movement, across sessions -- by residual AND by movement-null subspace. | cited by a results page |
| [`epoch_kernels.py`](epoch_kernels.py) | PER-EPOCH POOLED PIXEL KERNEL FITS (handoff 000y item E, approved 10-25; Priya 10-25 night: "what happened to our cross-epoch event kernel maps" -> "just queue locally"). |  |
| [`epoch_sessions_long.py`](epoch_sessions_long.py) | Per-SESSION long tables for the epoch scalar families + CD trajectories, for the mixed models (encoding window via Priya, 10-20 late: "change the collectors to also emit a per-session table ... | cited by a results page |
| [`exec3d_pilot_figure.py`](exec3d_pilot_figure.py) | Target vs executed with the cam4 and 3-D executed sets, side by side (Priya 10-20: "did you make 3d figures"). |  |
| [`face_rhythm_aligned_maps.py`](face_rhythm_aligned_maps.py) | Across-animal FACE MOVEMENT MAPS per spout position -- cue-, lick- and bout-onset-aligned -- with statistics on face PATCHES (Priya 10-24: "the whole face moves less acutely across spout positions, wi ... |  |
| [`face_rhythm_asym_aligned.py`](face_rhythm_asym_aligned.py) | Cue- and lick-aligned FACE ASYMMETRY per spout position, across epochs (Priya 10-24). |  |
| [`face_rhythm_bout_split.py`](face_rhythm_bout_split.py) | Why is close_R bout-aligned face asymmetry > 0 BEFORE bout onset? (Priya 10-24, open item (a) of the FR handoff.) |  |
| [`face_rhythm_cohort_null.py`](face_rhythm_cohort_null.py) | Score the PERMUTATION NULL for the all-animals-pooled shape readouts (Priya 10-25: "yes set up the null distribution" -- the jackknife over 4 animals cannot test the cohort fits; FR handoff 10-25 nigh ... |  |
| [`face_rhythm_drift_floor.py`](face_rhythm_drift_floor.py) | Is the post-stroke factor shape change bigger than pre-stroke DRIFT? (FR handoff 10-25 night: the cohort preA / preB floor interleaves days, so it measures fit noise only, 3-8 deg -- too lenient a flo ... |  |
| [`face_rhythm_epoch_shape.py`](face_rhythm_epoch_shape.py) | Do the factors change SHAPE after the stroke? The cleaner test on POOLED fits (Priya 10-24, FR handoff open item (c); FINDINGS 6c: per-session own fits are too noisy to read -- pre own-vs-master angle ... |  |
| [`face_rhythm_evr_epochs.py`](face_rhythm_evr_epochs.py) | Does the pre-stroke vocabulary fit post-stroke sessions worse? Epoch-pooled, each animal against its own pre-stroke sessions, with statistics across animals (Priya 10-24). |  |
| [`face_rhythm_match.py`](face_rhythm_match.py) | FaceRhythm factor MATCHING across independent fits ("morphing"; Priya 10-24, after the VLS-stroke methods paper): each session's OWN rank-R fit (`face_rhythm_factors.py fitsession` -> ``<set>/own/``) ... |  |
| [`face_rhythm_nontask_state.py`](face_rhythm_nontask_state.py) | Is the acute NON-TASK face-movement drop just "less active overall"? (Priya 10-24, FR handoff open item (c); FINDINGS 6d caveat.) |  |
| [`face_rhythm_overlay.py`](face_rhythm_overlay.py) | Every session's spout tips / rest nose / rest jaw / tongue reach mapped INTO the master face, for drawing the master ROI (Priya 10-21: "should we do it with the close spout locations projected so i ex ... |  |
| [`face_rhythm_peak.py`](face_rhythm_peak.py) | PEAK temporal-factor amplitude over recovery, relative to pre-stroke (Priya 10-24; the 2pRAM-cohort manuscript's FR figure: pre-stroke components refit to post sessions, temporal amplitude over days - ... |  |
| [`face_rhythm_position_maps.py`](face_rhythm_position_maps.py) | Face movement maps PER SPOUT POSITION, epoch vs pre (Priya 10-24: "show the spatial speed map per position to look at asymmetry"). |  |
| [`face_rhythm_session_shape.py`](face_rhythm_session_shape.py) | Lick-factor SHAPE per SESSION, with statistics (Priya 10-25: "so we need the per session version for stats"). |  |
| [`fr_contra_orofacial.py`](fr_contra_orofacial.py) | Is the chronic rise in task / position coding in contralesional orofacial sensorimotor cortex real once face movement is in the model? (Priya 10-25, on Figs 16g-16h: "with regressing out uninstructed ... |  |
| [`fr_direction.py`](fr_direction.py) | FaceRhythm lick DIRECTION vs DIRECTIONLESS vs other (Priya 10-25: "for fr runs - what happened to our lick direction vs lick directionless vs other analyses?"; "the FR direction vs directionless figur ... |  |
| [`fr_direction_mechanism.py`](fr_direction_mechanism.py) | WHY do FaceRhythm's direction factors explain less orofacial cortex after the stroke? (Priya 10-25: "can we test / delve into this ... it could be critical to interpretation".) |  |
| [`fr_encoding_summary.py`](fr_encoding_summary.py) | FaceRhythm in the per-session encoder (O2 batch enc_v36_full_20261024, runs fr_A / fr_B / fr_S / fr_D): does face movement add variance, and does it absorb the spout-position code? (handoff: "does Fac ... |  |
| [`fr_enl_decode.py`](fr_enl_decode.py) | Can spout position be decoded from FACE movement (FaceRhythm factors) in the lick-free pre-cue (ENL) window? |  |
| [`fr_musall.py`](fr_musall.py) | Musall-style variance figures and area maps for the encoder WITH FaceRhythm (Priya 10-25: "add the musall-style plots and maps for the fr-including runs"). |  |
| [`fr_pixel_maps_sig.py`](fr_pixel_maps_sig.py) | Musall-style FaceRhythm PIXEL maps WITH their significance on the same panels (Priya 10-25: "what happened with stats on the FR neural maps eg 16g" -- the tests were drawn only in the separate signifi ... |  |
| [`fr_pixel_significance.py`](fr_pixel_significance.py) | SIGNIFICANCE of the Musall-style pixel maps with FaceRhythm (Priya 10-25: "can we test significance on these maps?"). `--drift-control` = SUPPLEMENTAL late-pre vs early-pre per map (Priya 11-06; reads the epoch tests, no refit). |  |
| [`fr_trial_peth.py`](fr_trial_peth.py) | Trial-to-trial vs PETH variance WITH FaceRhythm (Priya 10-25: "what about PETH vs trial-by-trial variance?"). |  |
| [`frozen_area_maps.py`](frozen_area_maps.py) | AREA MAPS of the frozen-encoder decomposition (Priya 10-22: "are there maps that go along with this most recent analysis?" -> "go ahead with the area map and then pixel maps"). |  |
| [`frozen_claims.py`](frozen_claims.py) | POST-STROKE CHANGE CLAIMS REDONE IN THE FROZEN FRAME (Priya 10-25: "redo all post-stroke change claims with the frozen model"). |  |
| [`frozen_fr_musall.py`](frozen_fr_musall.py) | Musall-style UNIQUE R2 in the FROZEN frame, with vs without FaceRhythm (Priya 10-25: "make the frozen model the headline"). |  |
| [`frozen_kernel_maps.py`](frozen_kernel_maps.py) | Musall-style EVENT-KERNEL beta maps of the FROZEN pre-stroke encoder (Priya 10-25: "have we been looking at beta weights of event kernels" -> "show all 6 spout position kernels"; "lick onset and conta ... |  |
| [`frozen_musall_figures.py`](frozen_musall_figures.py) | Musall et al. 2019-style figures from the FROZEN pre-stroke encoder (Priya 10-22: "did we regenerate the musall style PETH vs trial-by-trial prediction, task-aligned and task-independent, and task-ali ... |  |
| [`frozen_peth_trial_maps.py`](frozen_peth_trial_maps.py) | Musall-style PETH vs TRIAL-BY-TRIAL maps in the FROZEN frame, at Allen-area resolution (Priya 10-25: "do we have PETH vs trial by trial maps (as in the Musall paper)"). Musall et al. 2019 Fig. |  |
| [`frozen_pixel_maps.py`](frozen_pixel_maps.py) | PIXEL MAPS of the frozen-encoder decomposition from a SPATIAL-target run (`frozen_encoder --target basis / grid`; Priya 10-22: "go ahead with the area map and then pixel maps"). |  |
| [`frozen_pixel_musall.py`](frozen_pixel_musall.py) | PIXEL Musall layouts of the FROZEN encoder (Priya 10-25; handoff 000y task C): the single-variable / single-group ALONE cvR2 maps (Musall et al. |  |
| [`frozen_pixel_unique_sig.py`](frozen_pixel_unique_sig.py) | FROZEN-frame unique R2 PIXEL maps WITH statistics (Priya 10-25: "make the frozen model the headline ... and make the pixel maps ... with stats"; per-animal z-scored version too). |  |
| [`frozen_posinfo.py`](frozen_posinfo.py) | Movement WITH vs WITHOUT lick-position information, across pose AND FaceRhythm, in the FROZEN frame (Priya 10-25: "can we group by lick position information vs no lick position information across FR a ... |  |
| [`frozen_residual_significance.py`](frozen_residual_significance.py) | SIGNIFICANCE of the frozen-encoder RESIDUAL maps per epoch (Priya 10-24: "can we measure significance in the residual maps in each epoch?"). |  |
| [`frozen_residual_trend.py`](frozen_residual_trend.py) | TREND MODEL for the frozen-encoder residual (Priya 10-24: "let's do random splits and trend model"). |  |
| [`frozen_spout_maps.py`](frozen_spout_maps.py) | PRE-STROKE PER-SPOUT pixel maps of the frozen-encoder decomposition from a basis run (Priya 10-23: "why is the far contra map not showing L hemisphere activation ..."). |  |
| [`histology_stroke_extent_figure.py`](histology_stroke_extent_figure.py) | (no docstring) target plate: (fallback plate for real data or None, interpolation allowed) |  |
| [`imaging_per_animal_tables.py`](imaging_per_animal_tables.py) | Per-animal tables for two imaging readouts the results page reported pooled only (Priya 10-25, imaging page "pending"): the pre-stroke position TEMPLATE (epoch family 11 / 11amp / 11c, cue window, eng ... |  |
| [`iti_area_maps.py`](iti_area_maps.py) | Whole-cortex view of the self-initiated-lick response (Priya 10-26: "should we also do some non-orofacial per-area activity maps to look at the whole cortex question? (supplemental figure with VIS are ... |  |
| [`iti_basis_decoder.py`](iti_basis_decoder.py) | The FROZEN lick-direction decoder of page 3.8 / 3.9 in two other feature bases (Priya 10-26: "would locaNMF basis be useful? ... |  |
| [`iti_decoded_decomposition.py`](iti_decoded_decomposition.py) | PER-SPOUT DECOMPOSITION of the decoded lateral shift (Priya 10-27: "if our effect was just a 'pull' the direction would be different for ipsi vs contra spouts" -> "try the per-spout decomposition"). |  |
| [`iti_decoder_contrib.py`](iti_decoder_contrib.py) | Which areas move the frozen decoder's lick position after the stroke? (Priya 10-26: "it's counter intuitive that cued licks are read as being closer ... when there is MORE tjM1 and ALM activity"). |  |
| [`iti_decoder_pattern_stats.py`](iti_decoder_pattern_stats.py) | Statistics for the Haufe decoder-pattern maps (3.33; Priya 10-27: "can 3.33 have stats?"). |  |
| [`iti_extension_dist.py`](iti_extension_dist.py) | Distribution of lick EXTENSION (sagittal extension out of the mouth, and 3-D reach) per animal x epoch, with the threshold used to define 'longer licks' for the angle-decoder r (Priya 10-27: "is longe ... |  |
| [`iti_first_lick.py`](iti_first_lick.py) | FIRST cued lick vs LATER licks: is the cortical read-out of the first lick (launched before any feedback) held further toward the target than later licks? (handoff 10-27: "cued group = all post-cue li ... |  |
| [`iti_lick_order.py`](iti_lick_order.py) | LICK ORDER and the lateral read-out, ITI and cued (Priya 10-27: "first-lick ITI, first-lick post-cue, subsequent bout licks ITI, subsequent post-cue -- or a correlation of lick order and offset?"). |  |
| [`iti_lick_rate.py`](iti_lick_rate.py) | Does INTER-TRIAL licking track recovery, and does it exceed pre-stroke levels? (Priya 10-26.) |  |
| [`iti_lmm_figures.py`](iti_lmm_figures.py) | PRIMARY figures for the ITI / lick read-out results: mixed models WITH TIME (Priya 10-27: mixed-model versions of the ITI / lick figures as the primary figures; the epoch-bar figures stay as supplemen ... |  |
| [`iti_overcorrection.py`](iti_overcorrection.py) | Is the chronic ipsilesional read-out of far-contra cued licks a downstream OVER-CORRECTION? (Priya 10-27: "run the overcorrection analyses you proposed"). |  |
| [`iti_pattern_area_stats.py`](iti_pattern_area_stats.py) | ALLEN-AREA statistics for the Haufe pattern maps (3.33 / 3.35 / ITI 3.35; Priya 10-27: "is per-pixel nested bootstrap too stringent? pixels correlate"). |  |
| [`iti_recovery_link.py`](iti_recovery_link.py) | Do the frozen where-decoder's post-stroke changes (r loss, beyond-pull offsets, MAE) track behavioural recovery? (Priya 10-26: "does PS94, who had the least complete recovery, show the most persistenc ... |  |
| [`iti_recovery_marker.py`](iti_recovery_marker.py) | Is the frozen pre-stroke lick-direction decoder a MARKER OF RECOVERY? (Priya 10-26: "perhaps this is a marker of recovery?" about the in-trial recovery of page 3.9.) |  |
| [`iti_session_cluster_tests.py`](iti_session_cluster_tests.py) | SESSION-LEVEL PERMUTATION + TFCE change maps for the per-spout Haufe maps (Priya 10-27: "3.52 / 3.53 -- can we do per session versions with better powered stats? do we have the shape data plotted or j ... |  |
| [`iti_session_pattern_tests.py`](iti_session_pattern_tests.py) | Map-level permutation tests + AMPLITUDE-NORMALISED change maps on the fully per-session Haufe maps (`iti_session_patterns.py`; Priya 10-27: "is there a way to measure significant DIFFERENCES between m ... |  |
| [`iti_spout_pattern_stats.py`](iti_spout_pattern_stats.py) | Statistics for the PER-SPOUT Haufe maps (3.35; Priya 10-27: "can we add stats to 3.35 too"). |  |
| [`iti_spout_patterns.py`](iti_spout_patterns.py) | PER-SPOUT activation maps across epochs (Priya 10-26: "map the decoder weights per position across epochs"). |  |
| [`iti_where_angle_reach.py`](iti_where_angle_reach.py) | DIRECTION vs EXTENT: the frozen where-decoder on lick ANGLE and 3-D REACH (Priya 10-27: "redo our lateral vs protrusion analysis with angle vs 3d protrusion"; angle = primary coordinate, with x / y / ... |  |
| [`iti_where_decoder.py`](iti_where_decoder.py) | WHERE did the lick go: the frozen decoder of page 3.8 / 3.9 for lateral AND outward position (Priya 10-26: "should we look at angle and distance together? ... |  |
| [`iti_where_drift.py`](iti_where_drift.py) | Is the frozen lick-position decoder's post-stroke r loss more than ELAPSED-TIME DRIFT? (handoff 00f to-do 3) |  |
| [`iti_where_mae.py`](iti_where_mae.py) | Decoding ERROR in mm per session (Priya 10-26: "let's add the MAE / RMSE - but this will be subject to pull to middle"). Companion to page 3.15 (r). |  |
| [`iti_where_per_spout.py`](iti_where_per_spout.py) | Per-spout decoding error and r (Priya 10-26: "should we look at r and lateral / distance error also per spout position? (ITI using spout position preceding the ITI)"). |  |
| [`iti_where_scatter.py`](iti_where_scatter.py) | Decoded vs ACTUAL lick landing in 2-D, per spout and epoch (Priya 10-26: "plot the actual 3d (or 2d) decoded vs true lick positions for given spout positions per epoch per position"). |  |
| [`kinematics_pooled_figs.py`](kinematics_pooled_figs.py) | POOLED, within-animal pre-normalised versions of the per-animal kinematics figures (Priya 10-25: "a pre-stroke- normalized within-animal normalized cross-animal session pooled version of fig 2 (and an ... |  |
| [`lick_gain_prediction.py`](lick_gain_prediction.py) | ATTRIBUTION (not a correction): how much of the like-for-like read-out shift could a pan-cortical change in the lick response SIZE produce through the frozen decoder? (Priya 10-27: "a pan-cortical act ... |  |
| [`lick_matched_readout.py`](lick_matched_readout.py) | LIKE-FOR-LIKE (landing-matched) lick read-out -- the PRIMARY offset measure from 10-27 (Priya: "let's do the like-for-like lick analysis"; replaces 'offset beyond the pull', whose answer depended on w ... |  |
| [`lick_path_softmax.py`](lick_path_softmax.py) | TRAJECTORY softmax confusion matrices, the path analogue of the landing matrices (Priya 10-24: "can you make the same softmax_trajectory_change confusion matrices as you did for landing?"). |  |
| [`lick_review_sheets.py`](lick_review_sheets.py) | Image sheets from the human lick review (Priya, 2026-10-14): (1) LP r4 FALSE POSITIVES (judged "not a lick", LP r4 detected) and (2) EKS FALSE NEGATIVES (judged real, LP r4 detected, EKS r4+ctx did no ... |  |
| [`lmm_change_fig.py`](lmm_change_fig.py) | LMM VERSION of a change-vs-pre figure (Priya 2026-11-03: hidden deck slides beside the bootstrap versions). |  |
| [`lmm_curve_summaries.py`](lmm_curve_summaries.py) | LMM versions of the CURVE figures (Priya 11-03): each session's curve reduced to the summaries PRE-SPECIFIED in DECISIONS 2026-11-03 ("Curve figures: mixed model on PRE-SPECIFIED per-session summaries ... |  |
| [`lmm_per_lick_fig.py`](lmm_per_lick_fig.py) | Slide layout of the FaceRhythm per-lick LMM figure (Priya 11-03 via a8: "bottom plot is way too spread out; top plot is too busy"): two rows of six factors grouped by preferred lick direction (row 1 r ... |  |
| [`map_drift_prepre.py`](map_drift_prepre.py) | How far do the position maps DRIFT with no stroke? Early pre (June) vs late pre (August), beside the stroke changes. | cited by a results page |
| [`matched_residual.py`](matched_residual.py) | BEHAVIOUR-MATCHED RESIDUAL (Priya 10-25, analysis 4 of the synthesis): is the chronic residual (cortex responding more than the frozen pre-stroke mapping predicts) a frozen-model EXTRAPOLATION artefac ... |  |
| [`misdirected_lick_decode.py`](misdirected_lick_decode.py) | Misdirected licks: does cortex look like the PLANNED far-contra lick or the lick the tongue EXECUTED? (Priya 2026-10-19: "in the acute far_contra ... |  |
| [`movement_position_angle.py`](movement_position_angle.py) | Position-specific encoding AFTER regressing out movement, and lick responses organised by EXECUTED tongue angle vs target position -- LocaNMF dF/F on the frames inside the session's pose windows. |  |
| [`null_potent_session.py`](null_potent_session.py) | Movement-null / movement-potent subspaces (Hasnain et al. 2025) per session, and where the spout-position code lives: ENL (pre-cue) and post-cue position decoding from activity reconstructed in each s ... |  |
| [`orofacial_cover_sensitivity.py`](orofacial_cover_sensitivity.py) | Sensitivity of the cohort change curves to the coverage rule (Priya 10-17: "how did you decide ... at least half of an epoch's sessions, and at least 3"). |  |
| [`partial_refit.py`](partial_refit.py) | PARTIAL REFIT: refit only the TARGET vs only the EXECUTED-POSITION terms per session, everything else frozen (Priya 10-25: "let's consider models with refitting only target vs executed position terms" ... |  |
| [`pose3d_close_dir_sheet.py`](pose3d_close_dir_sheet.py) | Video sheet for the 3-D jaw CLOSE_DIR readout (handoff 10-20 to-do 1a: look before interpreting). |  |
| [`pose3d_cue_traces.py`](pose3d_cue_traces.py) | Cue-aligned, trial-averaged 3-D tongue and jaw traces (handoff 10-20 to-do 1b; Priya 10-20 "yes"). |  |
| [`pose3d_figures.py`](pose3d_figures.py) | Pictures of the 3-D tongue (`wfield_local.pose3d`): trajectory maps, and a paired cam1 / cam4 / 3-D lick video. |  |
| [`pose3d_phase.py`](pose3d_phase.py) | Lick phase and trajectory plots in 3-D, a la the cam4 cohort (`orofacial_cohort._lick_phase`, `_phase_norm`, `_phase_fig`): Priya 10-19 "can we do phase and trajectory plots a la cam4?". |  |
| [`pose3d_spout_error.py`](pose3d_spout_error.py) | Error distance from the spout, in 3-D: where the tongue ends up relative to the TARGET (Priya 10-19: "error distance from spout is still an interesting (and separate) analysis"). |  |
| [`pose3d_summary_fig.py`](pose3d_summary_fig.py) | 3-D tongue / jaw readouts across sessions, per animal (Priya 10-19: "where are the 3-D kinematics plots"). |  |
| [`position_block_durations.py`](position_block_durations.py) | Position-block durations per session (Priya 2026-10-19: "in PS92 especially position blocks last longer, because the animal was so licky that it reset the ENL a lot"). | cited by a results page |
| [`precue_class_matched.py`](precue_class_matched.py) | CLASS-MATCHED PRE-CUE TEST (Priya 10-25: "then do class-matched pre cue"): is the frozen decoder's far-contra code weaker BEFORE the cue on trials that go on to be failed attempts than on hits? |  |
| [`recovery_timing.py`](recovery_timing.py) | RECOVERY TIMING, HITS ONLY (Priya 10-25: "how strong is the evidence that cortex recovers at / after behaviour" -> run the hits-only test; and "can we predict animals' recovery from cortical activity? ... |  |
| [`relabel_fc_by_class.py`](relabel_fc_by_class.py) | Far-contra TARGET trials in the frozen encoder relabel test, by ATTEMPT CLASS (Priya 10-25: "can we look at the relabel encoder model test for far contra trials on successful vs attempted vs no attemp ... |  |
| [`remap_lmm.py`](remap_lmm.py) | DOES THE CHRONIC REMAPPING TRACK RECOVERY? (Priya 10-25, analysis 2 of the synthesis.) Session mixed models (`wfield_local.mixed_models`, the shared definition: time y ~ post + post:lday, behaviour y ... |  |
| [`replot_epoch_from_sidecars.py`](replot_epoch_from_sidecars.py) | Redraw epoch bar / contrast figures from their stored sidecars, in the 11-03 session-dot style. |  |
| [`replot_map_figures.py`](replot_map_figures.py) | Redraw every map figure from its saved bundle -- presentation changes, no recomputation. |  |
| [`residual_lick_aligned.py`](residual_lick_aligned.py) | Descriptive companions to `movement_position_angle` (NOT part of its pre-registered tests): where the residual position activity sits in time, and what the lick kernels look like by target position vs ... |  |
| [`residual_roi.py`](residual_roi.py) | ALLEN-ROI version of the pooled residual pixel test (page Fig 10b; Priya 10-25: "should we do an allen ROI version of 10b"). |  |
| [`restructure_output_tree.py`](restructure_output_tree.py) | MOVE THE PUBLISHED OUTPUT TREE INTO ITS 2026-09-21 LAYOUT. Dry-run by default. |  |
| [`session_lick_summary.py`](session_lick_summary.py) | Per-session lick summary across sessions (scripts.session_poses clips), both models, by spout position: licks per trial, DAQ contact rate, protrusion, rise / retraction speed, peak angle, and DIRECTIO ... |  |
| [`slowband_caveat_figure.py`](slowband_caveat_figure.py) | Presentation figure for the pre-cue SLOW-BAND caveat (Priya 10-19 / 10-20: handoff to-do 6). | cited by a results page |
| [`specificity_control.py`](specificity_control.py) | SPECIFICITY CONTROL for the chronic residual and Layer 2 (Priya 10-25, analysis 1 of the synthesis: "is the chronic increase specific or global?"). |  |
| [`spout_mid_maps.py`](spout_mid_maps.py) | Per-animal per-position DIRECTION maps for far_ipsi and close_contra (contact-controlled, cam4 lick-aligned; Priya 10-21 item 4). |  |
| [`spout_reference_frame.py`](spout_reference_frame.py) | Test the spout-position reference frame: do the close->far lines for L, centre and R converge on the mouth? |  |
| [`spout_unique_maps.py`](spout_unique_maps.py) | PER-SPOUT UNIQUE-CONTRIBUTION pixel maps, TARGET vs EXECUTION (Priya 10-25: "we want the unique contribution spout target and execution maps"). |  |
| [`summary_figures_1021.py`](summary_figures_1021.py) | 10-21 summary figures for results that had tables only (Priya: "note where i can find figures"): figures/behaviour/attempt_classes_far.png attempt classes (canonical trials) + time to contact, far con ... |  |
| [`te_readout.py`](te_readout.py) | TARGET vs EXECUTION readout of the FROZEN_DESIGN=te run (DECISIONS 2026-10-26; Priya: "target kernel vs execution kernel"). |  |
| [`tongue_angle_phase.py`](tongue_angle_phase.py) | Tongue angle: (1) why `angle_max_signed_lick1` and the lick-1 angle at the peak can disagree in sign -- a visual example per disagreeing trial; (2) angle and protrusion over the LICK PHASE, per spout ... |  |
| [`trial_class_raw_share_fig.py`](trial_class_raw_share_fig.py) | RAW share of trials per class (jaw moves / no lick; licked / no spout contact), per session x spout, grouped by epoch. |  |
| [`variance_maps.py`](variance_maps.py) | TOTAL VARIANCE maps of the widefield signal, epoch vs pre (Priya 10-25: "is the increased variance post-stroke itself significant?" -> "plot variance maps pooled and pooled per-animal normalized for d ... |  |
| [`video_motion_session.py`](video_motion_session.py) | Video motion-energy components for one session and camera (`wfield_local/video_motion.py`), on the session's imaging-frame clock, for the movement encoding models (`scripts.movement_encoding_session - ... |  |

### pipeline step (66)

| script | what it does | tags |
|---|---|---|
| [`apparatus_mouth.py`](apparatus_mouth.py) | The commanded (apparatus) MOUTH in cam1 and cam4, per session, from the triangulated spout positions (`spout_world.apparatus_mouth`; Priya 2026-10-18 "build option 2"). | imported or run by wfield_local |
| [`attempt_review.py`](attempt_review.py) | Blind clip review of the ATTEMPTED gate (Priya 10-21: review the attempted / canonical_attempted gate before any default change; "we can also look at xy jaw traces and nose traces as part of judging a ... | imported or run by wfield_local |
| [`cam1_lick_aim.py`](cam1_lick_aim.py) | First cam1 behavioural readout (Priya 2026-10-18, PRELIMINARY: the cam1 model is not chosen yet): where does each lick point, relative to its spout, seen from below? cam1 sees left-right and up-down, ... | imported or run by wfield_local |
| [`cam4_timing_check.py`](cam4_timing_check.py) | How far does cam4's FRAME-INDEX time (``t_ms``, what v36 analyses use) sit from its CAMERA-TIMESTAMP time (``t_daq_ms``, from the sync pulses) inside trial windows? (Priya 10-20, after PS92 0908: drop ... | imported or run by wfield_local |
| [`cd_cross_animal_figure.py`](cd_cross_animal_figure.py) | CROSS-ANIMAL CD trajectories, as the CHANGE from each animal's own pre-stroke trajectory. | imported or run by wfield_local |
| [`cd_geometry_figure.py`](cd_geometry_figure.py) | The condition-independent mode across epochs: does it ROTATE, does it GROW, and what leaks. | imported or run by wfield_local; cited by a results page |
| [`cd_loo_figures.py`](cd_loo_figures.py) | CD figures on the AUGUST baseline with leave-one-out pre sessions (`scripts/cd_loo_sessions.py`; Priya 10-21: "please do the CD mean and trajectories"). | imported or run by wfield_local; cited by a results page; cited by deck_registry |
| [`cd_loo_sessions.py`](cd_loo_sessions.py) | CD window projections per session with the PRE sessions scored LEAVE-ONE-OUT (encoding window via Priya 10-20: "pre sessions: each needs its own value"; 10-21 to-do 4 of STATUS_2026-10-21_POSE_CAM1_JA ... | imported or run by wfield_local; cited by a results page; cited by deck_registry |
| [`cd_migration.py`](cd_migration.py) | WHERE DID A POSITION'S TRAJECTORY MOVE TO? The 6x6 similarity matrix and its post-stroke change. | imported or run by wfield_local; cited by a results page |
| [`chronic_stability.py`](chronic_stability.py) | Is the CHRONIC neural data stable, per animal, per session -- the same test behaviour gets? | imported or run by wfield_local |
| [`coding_direction_audit.py`](coding_direction_audit.py) | DOES ORTHOGONALISING THE CODING DIRECTIONS MOVE THEM TOWARD THE COVARIANCE-AWARE ANSWER? | imported or run by wfield_local |
| [`combined_encoding_figures.py`](combined_encoding_figures.py) | Figures for the combined task + movement encoding model (`wfield_local.combined_encoding`), Musall et al. 2019 Fig. | imported or run by wfield_local |
| [`compare_svg_renders.py`](compare_svg_renders.py) | COMPARE TWO SVG RENDER DIRECTORIES, ignoring what matplotlib makes non-reproducible. | in a runbook |
| [`deck_claim_audit.py`](deck_claim_audit.py) | LIST THE DECK'S QUANTITATIVE CLAIMS SO THEY CAN BE RE-MEASURED, and say which kind each is. | imported or run by wfield_local |
| [`deck_figure_coverage.py`](deck_figure_coverage.py) | WHICH RENDERED FIGURES DOES THE DECK NEVER PLACE? | imported or run by wfield_local; cited by deck_registry |
| [`deck_fingerprint.py`](deck_fingerprint.py) | PROVE A DECK REFACTOR CHANGED NOTHING — a semantic fingerprint of a built .pptx, and a stub figure tree to build one against. | in a runbook |
| [`encoding_drift_check.py`](encoding_drift_check.py) | Slow-drift diagnostic for the continuous-time combined encoding model (Priya 2026-10-18: "measure how common it is - and make some figures that show me the problem graphically"). | imported or run by wfield_local |
| [`encoding_mixed.py`](encoding_mixed.py) | Mixed-effects time course of the POSITION encoding measures (Priya 10-20: "what about mixed model for position / distance? are we using random intercept and random slope per animal"). | imported or run by wfield_local |
| [`engagement_axis_balance.py`](engagement_axis_balance.py) | HOW MUCH POSITION INFORMATION IS THE "ENGAGEMENT" AXIS CARRYING? | imported or run by wfield_local |
| [`engagement_by_epoch.py`](engagement_by_epoch.py) | `epoch_1f`: engagement by EPOCH, with no position axis -- because there is no position effect. | imported or run by wfield_local; cited by deck_registry |
| [`enl_decode_figure.py`](enl_decode_figure.py) | The ENL sensory-vs-plan result: does the pre-cue position code survive when the animal STOPS? | imported or run by wfield_local; cited by a results page |
| [`enl_lick_rates.py`](enl_lick_rates.py) | How much licking is actually near the ENL decoding window? Measure before changing anything. | imported or run by wfield_local |
| [`enl_lp_census.py`](enl_lp_census.py) | ENL lick-free census, DAQ contact vs LP licks (Priya 10-21: "re-evaluate our ENL gate using LP_licks, to see how we need to redefine 'lick-free' ENL where there are no LP licks detected (... | imported or run by wfield_local |
| [`enl_sparsity_figure.py`](enl_sparsity_figure.py) | The trial-count landscape for the ENL working-vs-stopped contrast -- what can and cannot be asked. | imported or run by wfield_local |
| [`enl_state_counts.py`](enl_state_counts.py) | How many WORKING and STOPPED trials actually exist, per animal x epoch x position? | imported or run by wfield_local |
| [`exec_bins_calibration.py`](exec_bins_calibration.py) | Calibration table for the CALIBRATED executed bins of the binned contest (combined_encoding BINS_CALIB_ENV). | imported or run by wfield_local |
| [`face_rhythm_factors.py`](face_rhythm_factors.py) | FaceRhythm frozen pre-stroke factor set: the command line. The code lives in `wfield_local/face_rhythm/factors/` (split from this script 10-25, every definition moved verbatim); stages, inputs and out ... | imported or run by wfield_local |
| [`face_rhythm_lick_epochs.py`](face_rhythm_lick_epochs.py) | FaceRhythm: is a post-stroke change in a lick factor FEWER licks or DIFFERENT licks? (10-22 FINDINGS 5 caveat: absolute expression scales with how much the animal licks.) Computation: `face_rhythm_val ... | imported or run by wfield_local |
| [`face_rhythm_mesh_qc.py`](face_rhythm_mesh_qc.py) | QC of the master mesh (12 px, 1282 points) warped to every session, BEFORE tracking: | imported or run by wfield_local |
| [`face_rhythm_prep.py`](face_rhythm_prep.py) | FaceRhythm prep (docs/FACE_RHYTHM_PLAN.md): per-session reference faces + LP landmarks, for registering every session onto one MASTER face per camera. | imported or run by wfield_local |
| [`face_rhythm_readouts.py`](face_rhythm_readouts.py) | FaceRhythm session readouts: tables + figures for a frozen factor set (computation in `wfield_local/face_rhythm_readouts.py`; pipeline: docs/FACE_RHYTHM_PIPELINE.md). | imported or run by wfield_local |
| [`face_rhythm_register.py`](face_rhythm_register.py) | Register every session's reference face onto ONE master face per camera (docs/FACE_RHYTHM_PLAN.md step 3; Priya 10-21: "morph across animals to fit a single 'master' mouse face mesh for each camera"; ... | imported or run by wfield_local |
| [`face_rhythm_validate.py`](face_rhythm_validate.py) | FaceRhythm factor identity vs independent behaviour: LP licks (executed direction, time in trial, bout position, spout position, lick frequency) and the treadmill (Priya 10-22; computation `wfield_loc ... | imported or run by wfield_local |
| [`face_rhythm_warp_rois.py`](face_rhythm_warp_rois.py) | Move FaceRhythm ROIs between sessions and the MASTER face through the registration fields (`scripts/face_rhythm_register.py`: ``register/fields/<s>.npy`` maps MASTER px -> SESSION px). | imported or run by wfield_local |
| [`frozen_group_leakage.py`](frozen_group_leakage.py) | How much SPOUT information does each frozen-encoder regressor group carry? (Priya 10-22: "ensure our variable groups appropriately include or exclude general directionless movement vs direction-specif ... | imported or run by wfield_local |
| [`frozen_r2_maps.py`](frozen_r2_maps.py) | PIXEL R^2 / UNIQUE R^2 MAPS of the frozen encoder from a SPATIAL-target run (`frozen_encoder --target basis / grid`; Priya 10-23: "do we have the pixel/SVD level unique r2 maps?"). | imported or run by wfield_local |
| [`frozen_running_compare.py`](frozen_running_compare.py) | Compare frozen-encoder RUNNING models (Priya 10-23: pick v7 among signed linear / spline / forward-only thresholds, with and without the backward "push" term). | imported or run by wfield_local |
| [`imaging_lmm.py`](imaging_lmm.py) | Session-level mixed models for the imaging / behaviour tables that already carry one value per session (Priya 2026-10-20: "take a close look at the current imaging analyses - decide where lmm approach ... | imported or run by wfield_local; cited by a results page |
| [`import_animal_weights.py`](import_animal_weights.py) | Import the per-animal body-weight sheet exports into `configs/animal_weights.csv`. | imported or run by wfield_local |
| [`import_histology_areas.py`](import_histology_areas.py) | Import ipsi/contra striatal (caudoputamen) areas from the PS92-95 QuPath project. | imported or run by wfield_local |
| [`iti_licks.py`](iti_licks.py) | SELF-INITIATED (ITI) 3-D LICKS: pre-lick ramp timing + target-free execution coding (Priya 10-25 night: "let's also do the inter-trial 3d licks, pre-lick ramp timing"; proposed 10-25 as analyses 3 and ... | imported or run by wfield_local |
| [`iti_recovery_measure.py`](iti_recovery_measure.py) | BEHAVIOURAL RECOVERY MEASURE for ranking animals (Priya 10-27: "angle offset from intended spout as measure of recovery - weighted for far contra then far mid, then all others"). | imported or run by wfield_local |
| [`jaw_lateral.py`](jaw_lateral.py) | Jaw x ceiling retune (handoff 10-14 to-do 9) and jaw lateral deviation on opening (to-do 10). | imported or run by wfield_local |
| [`jaw_only_incomplete_figs.py`](jaw_only_incomplete_figs.py) | JAW-WITHOUT-LICK and INCOMPLETE-LICK trials: how often per session, and what the jaw / tongue do on them (Priya 11-02: "kinematics Fig 6.1 is titled jaw without a lick but plots just jaw -- a figure s ... | imported or run by wfield_local |
| [`jaw_onsets.py`](jaw_onsets.py) | Per-trial JAW MOVEMENT ONSET, both cameras, on the DAQ clock (encoding window via Priya 10-20: "per-trial jaw movement onset (ms from cue, DAQ clock, camera that saw it), so jaw-only trials get an att ... | imported or run by wfield_local |
| [`june_august_drift.py`](june_august_drift.py) | A NO-STROKE drift comparison group: June pre -> August pre, for the representational-distance and frozen-vs-refit readouts (Priya 2026-10-24: "add in a june vs august pre comparison group (for effects ... | imported or run by wfield_local; cited by a results page; cited by deck_registry |
| [`lick_template_match.py`](lick_template_match.py) | Do post-stroke licks cued to one spout but EXECUTED in another direction look, in cortex, like pre-stroke licks to the same TARGET or like pre-stroke licks that went the same WAY? (`wfield_local/lick_ ... | imported or run by wfield_local |
| [`lp_encoder_decompose.py`](lp_encoder_decompose.py) | Is the acute far-contra post-lick activity drop the MOVEMENT or the TARGET? (Priya 10-21 night, LP_licks item 1: "yes do 1-3 as provisional"; ENCODING_FINDINGS 3.3.) | imported or run by wfield_local |
| [`movement_encoding_session.py`](movement_encoding_session.py) | Movement-regressor encoding model on REAL sessions: LocaNMF dF/F ~ cue per position + DAQ contacts + DLC tongue onsets + continuous tongue / jaw + per-lick direction modulators, fit on the imaging fra ... | imported or run by wfield_local |
| [`nocontact_census.py`](nocontact_census.py) | WHAT IS THE PRE-STROKE "NO-LICK" POPULATION, PER POSITION? Not what its label says. | imported or run by wfield_local |
| [`o2_encoding.py`](o2_encoding.py) | The combined encoding model (`wfield_local/combined_encoding.py`) on HMS O2 (Priya 2026-10-19: "yes build the o2 version"). | imported or run by wfield_local |
| [`o2_face_rhythm.py`](o2_face_rhythm.py) | FaceRhythm (face-rhythm 0.3.3) point tracking + VQT spectrograms on HMS O2, one task per session x cam4 video (docs/FACE_RHYTHM_PLAN.md step 5). | imported or run by wfield_local |
| [`pose3d.py`](pose3d.py) | Per-frame 3-D tongue / jaw / nose / spout (cam1 + cam4) and per-lick 3-D reach and aim (miss distance / lateral error against the target spout), per session (`wfield_local.pose3d`). | imported or run by wfield_local |
| [`pose3d_depth_prior_check.py`](pose3d_depth_prior_check.py) | Held-out test of DEPTH PRIORS for cam1-ONLY tongue points (PS92; Priya 10-19: small licks cam4 never sees). | imported or run by wfield_local |
| [`pose3d_jaw_anchor_check.py`](pose3d_jaw_anchor_check.py) | Held-out test of a REST-DEPTH ANCHOR for long one-camera JAW stretches (Priya 10-20: recover the centre-spout jaw). | imported or run by wfield_local |
| [`pose3d_jaw_depth_model_check.py`](pose3d_jaw_depth_model_check.py) | Held-out test of a DEPTH MODEL for one-camera jaw frames vs the rest-depth anchor (Priya 10-20: cam1 + anchor jaw sits too HIGH when the jaw is open; "we should also know when jaw is opening (if tongu ... | imported or run by wfield_local |
| [`pose_chain.py`](pose_chain.py) | The pose / 3-D / cam1 chain in ONE command, in dependency order (STATUS_2026-10-21_POSE_CAM1_JAW_HANDOFF §4; Priya 10-21: "use saved arrays, caches, parallelization ... | imported or run by wfield_local |
| [`pose_cue_traces.py`](pose_cue_traces.py) | Cue-aligned tongue / jaw x & y traces, trials overlaid: DLC vs Lightning Pose on IDENTICAL frames. | imported or run by wfield_local; in a runbook |
| [`pose_kinematics_demo.py`](pose_kinematics_demo.py) | First run of the ported kinematics on the PS93 0908 cue clip, DLC round 3 vs LP: clean -> spout frame -> tongue v7 kinematics -> jaw per-trial -> tongue-jaw mismatch. | imported or run by wfield_local |
| [`position_coding_trend_recovery.py`](position_coding_trend_recovery.py) | Is the chronic gain in cortical position coding beyond the pre-stroke practice trend, and does it lead or follow behavioural recovery? (Priya 2026-10-19: tests 1 and 4 of the "what does this reveal ab ... | imported or run by wfield_local; cited by a results page |
| [`s1_refit_locanmf_table.py`](s1_refit_locanmf_table.py) | Per-session WITHIN-SESSION refit accuracy on the joint-LocaNMF basis (imaging page section 1, Priya 11-03: "we also have locanmf based decoders"): the per-animal table of Figs 1.1 / 1.2 (family 5r), w ... | imported or run by wfield_local; cited by a results page |
| [`s1_trial_class_snapshot.py`](s1_trial_class_snapshot.py) | SNAPSHOT of the shared trial classes per DAQ cue for the imaging section 1 table (and any imaging decoder that runs on another branch): one row per DAQ cue of every curated imaging session (from 08-06 ... | imported or run by wfield_local |
| [`session_poses.py`](session_poses.py) | Session-level cam4 pose predictions on a TRIAL SUBSET: whole trials (position strobe -0.5 s .. | imported or run by wfield_local |
| [`trial_attempts.py`](trial_attempts.py) | Per-trial attempt classes, a movement-based engagement label and an attempt / completion / latency split of the behavioural deficit (Priya 10-20: "fraction of trials with lick as well as incomplete vs ... | imported or run by wfield_local |
| [`whisker_motion.py`](whisker_motion.py) | Whisker-pad motion energy per camera frame (`wfield_local.roi_motion`), and its cue-aligned per-trial traces. | imported or run by wfield_local |
| [`within_spout_mixed.py`](within_spout_mixed.py) | Mixed-effects readout of the within-spout deviation check (Priya 10-20: "let's try a mixed model - allowing time post stroke to be a variable"). | imported or run by wfield_local |

### analysis (tables / numbers) (50)

| script | what it does | tags |
|---|---|---|
| [`a3_bootstrap_n4_calibration.py`](a3_bootstrap_n4_calibration.py) | False-positive rate + power at n = 4 animals: animal-weighted bootstrap (analysis_kit.paired_animal_draws, the A3 code, 95% percentile interval excludes 0) vs one-sample t on the 4 animal changes (df ... |  |
| [`a3_lmm_heterogeneous_stars.py`](a3_lmm_heterogeneous_stars.py) | Heterogeneous spread: y ~ C(epoch) + (1/animal) + animal x epoch variance SEPARATE per post epoch (and per-epoch residual variance is NOT modelled -- statsmodels can't). |  |
| [`a3_lmm_pooled_epoch_stars.py`](a3_lmm_pooled_epoch_stars.py) | Pooled-epoch LMMs per cell (all four epochs in one model), session rows: M1 y ~ C(epoch) + (1 / animal) M2 y ~ C(epoch) + (1 / animal) + (1 / animal:epoch) one animal-x-epoch variance shared by the 3 ... |  |
| [`a3_lmm_slope_stars.py`](a3_lmm_slope_stars.py) | Priya's A3 rule (11-03, via a8): per contrast (one post epoch vs pre, session rows, that animal's pre + that epoch), y ~ post + (1 + post / animal), REML; fixed-effect t test with Satterthwaite df; Ho ... |  |
| [`axis_holdout_null.py`](axis_holdout_null.py) | Two-session holdout, and what 8/13 does to the null. |  |
| [`best_match.py`](best_match.py) | WHICH PRE-STROKE POSITION DOES POST-STROKE ACTIVITY LOOK LIKE? Per animal, window, class, session. |  |
| [`best_match_destination.py`](best_match_destination.py) | Does a post-stroke spout's ACTIVITY look like the pre-stroke pattern of its TARGET, of WHERE ITS LICKS LANDED, or of NEITHER? (Priya 10-21: "we want to know if the brain activity after stroke looks li ... |  |
| [`cam1_jaw_review.py`](cam1_jaw_review.py) | Blind review of the cam1 "jaw moved" call (encoding window via Priya 10-20 late: "blind-review the cam1 jaw-moved call on acute no_try / jaw_only trials (far contra, close center); the 4 px floor is c ... |  |
| [`cam1_licks.py`](cam1_licks.py) | cam1 2-D lick events on the production O2 output (`wfield_local.cam1_licks`; Priya 10-19) and a BLIND review sample. |  |
| [`cleaned_label_clips.py`](cleaned_label_clips.py) | Labelled video of the CLEANED tongue and jaw (orofacial_clean) on a few trials of a `session_poses` clip, to judge the cleaning -- and, with several ``--model``s, the models -- by eye. |  |
| [`eks_two_model.py`](eks_two_model.py) | Single-camera EKS over a TWO-MODEL ensemble (e.g. round-4 LP heatmap + round-4 LP temporal-context) for one clip. |  |
| [`enl_movement_removed_decode.py`](enl_movement_removed_decode.py) | Is spout position decodable from the PRE-CUE (ENL) window after movement is regressed out? |  |
| [`face_rhythm_factor_clips.py`](face_rhythm_factor_clips.py) | Video clips of each FaceRhythm factor at its strongest moments, for naming the components (fr env: decord, cv2). |  |
| [`first_lick_latency.py`](first_lick_latency.py) | First-lick latency three ways (Priya 10-20: "these median first lick times are based on keypoint tracking right? not spout contact"): per trial, time from the cue to the first cam4 LP cam4 lick onset ... |  |
| [`goal_page_reorder.py`](goal_page_reorder.py) | Reorder the goal page's figures into Key / Supporting / Supplementary tiers per section, give every figure a STABLE id, renumber, and rewrite every cross-reference. |  |
| [`iti_basis_frame_sd.py`](iti_basis_frame_sd.py) | FRAME-LEVEL spread of each basis component per session (Priya 10-27: "frame-level scaling for SVD -- yes, seems important esp. since we're analysing ITI"). |  |
| [`iti_farcontra_pattern.py`](iti_farcontra_pattern.py) | Is the chronic cued FAR-CONTRA cortical pattern a WEAKER COPY of its pre-stroke pattern, or has it DRIFTED toward another spout's pattern? (Priya 10-26: "how can we make sense of the chronic far-contr ... |  |
| [`iti_mixed_models.py`](iti_mixed_models.py) | MIXED-EFFECTS models for the headline ITI / lick read-out results (Priya 10-27: "mixed-effects model with sessions nested in animals ... for any headline result" -> "yes"). |  |
| [`iti_session_patterns.py`](iti_session_patterns.py) | FULLY PER-SESSION Haufe pattern maps (Priya 10-27: "can we not do per-session haufe maps"; map-level permutation and amplitude-normalised change maps; per-spout-position version of the position maps). |  |
| [`iti_where_3axis.py`](iti_where_3axis.py) | The where-decoder on all THREE axes in one run: lateral, forward (out_ap) and down (down_dv) (Priya 10-27: "run the 3-axis version but low priority"). |  |
| [`iti_where_beyond_pull.py`](iti_where_beyond_pull.py) | Decoded lick position BEYOND the decoder's pre-stroke pull toward the middle (Priya 10-26: "in 3.19 it's clearly not just pulling toward the center"). |  |
| [`iti_where_rmse_split.py`](iti_where_rmse_split.py) | RMSE of the frozen where-decoder split into BIAS and SCATTER (handoff 00f optional): per session, error e = decoded - actual; RMSE^2 = mean(e)^2 + SD(e)^2. |  |
| [`iti_where_signed_error.py`](iti_where_signed_error.py) | Signed lateral decoding error per lick, ITI vs in-trial (Priya 10-26: "the predicted lateral deviation (with directionality) per lick in ITI vs in-trial licks"). |  |
| [`iti_where_spread_matched.py`](iti_where_spread_matched.py) | Is the frozen decoder's r loss a narrower SPREAD of lick positions? (handoff 00f to-do; r is a correlation, so a session whose licks span a narrower range scores lower with unchanged cortex.) |  |
| [`layer2_matched_n.py`](layer2_matched_n.py) | Layer 2 combined fit with the chronic epoch cut to as many sessions as pre_late (session-count control, 10-24). |  |
| [`layer2_random_splits.py`](layer2_random_splits.py) | Layer 2 RANDOM-SPLIT null (Priya 10-24: "what if we do a random re-sampling of early and late pre-stroke sessions" -> "let's do random splits"). |  |
| [`lick_review_candidates.py`](lick_review_candidates.py) | Candidate licks for HUMAN review (Priya, 2026-10-14): blind sample of licks WITHOUT spout contact detected by any cam4 model, plus a few contact licks as hidden positive controls, each as a strip of m ... |  |
| [`lick_template_test.py`](lick_template_test.py) | Pre-stroke lick TEMPLATES: does a post-stroke lick evoke the pre-stroke response of the spout it was AIMED at, or of WHERE IT LANDED? (Priya 10-20: "should we have kernels that match pre-stroke per-sp ... |  |
| [`lp_eval_cpu.py`](lp_eval_cpu.py) | Evaluate the epoch-170 snapshot on CPU (GPU is busy training): labelled frames with per-frame error + split, then the three review clips via the opencv reader. |  |
| [`lp_trial_counts.py`](lp_trial_counts.py) | LP_licks vs DAQ: per-trial categories under both lick sources, per animal x position x epoch (rule 8 table before any LP figure; handoff 10-21 §2c step 3). |  |
| [`lp_unlabelled_clips.py`](lp_unlabelled_clips.py) | Unlabelled clips for Lightning Pose's semi-supervised losses (temporal, pose-PCA). |  |
| [`map_test_calibration.py`](map_test_calibration.py) | False-positive calibration of the map significance tests on PRE-vs-PRE comparisons (Priya 2026-10-21: "please carefully review the stats code"; "should we also do this on all pre vs august pre?"). |  |
| [`misdirected_lick_counts.py`](misdirected_lick_counts.py) | Feasibility counts for the misdirected-lick question (Priya 2026-10-19: "in the acute far_contra there are times when the animal clearly has a lick but it doesn't contact the spout (left deviated) -- ... |  |
| [`model_label_clips.py`](model_label_clips.py) | Labelled review clips of RAW pose predictions on a `session_poses` trial clip, one panel per model, the SAME frame in every panel (Priya 2026-10-18: labelled clips for the cam1 round-2 model). |  |
| [`nose_onsets.py`](nose_onsets.py) | Per-trial NOSE MOVEMENT call + onset, both cameras, on the DAQ clock -- a candidate "reacted" class kept SEPARATE from attempts (Priya 10-21: "sometimes nose could maybe be helpful? (they may also rea ... |  |
| [`o2_lp_train_bundle.py`](o2_lp_train_bundle.py) | Build an O2 TRAINING bundle for Lightning Pose models that do not fit this desktop's 8-GB card (Priya 2026-10-16: "if more ram would be meaningfully better for higher resolution training ... |  |
| [`o2_video_me.py`](o2_video_me.py) | Video MOTION-ENERGY components on HMS O2, for the encoding model's uninstructed group (Priya 2026-10-18: "sure can prep a pilot o2 ME"). |  |
| [`orofacial_robustness_table.py`](orofacial_robustness_table.py) | Robustness of every orofacial cohort cell to the baseline and to PS93 (Priya 2026-10-18). |  |
| [`page_stable_ids.py`](page_stable_ids.py) | Give every figure on a results page a PERMANENT id, so links survive renumbering (Priya 10-27). |  |
| [`pose3d_side_loo.py`](pose3d_side_loo.py) | Leave-one-camera-out offsets in the four-camera 3-D (`pose3d_side`): is a camera's keypoint a CONSTANT offset from the other cameras' 3-D point (a labelling-definition difference -> build it in, like ... |  |
| [`pose3d_spout_commanded.py`](pose3d_spout_commanded.py) | Per session x spout position, the COMMANDED 3-D spout tip (mouth-zero mm) in the format of `prep_3d_inputs`' ``pose3d_spout_positions_<tag>.csv`` -- the encoder's TARGET3D table (`combined_encoding._t ... |  |
| [`prelim_numbers_crossnobis.py`](prelim_numbers_crossnobis.py) | GAIN vs MOVE: the exact numbers behind the crossnobis table in `docs/PRELIM_DATA_VLS_STROKE.md`. |  |
| [`prelim_numbers_encoder_ceiling.py`](prelim_numbers_encoder_ceiling.py) | The exact numbers behind the encoder-ceiling tables in `docs/ENCODER_CEILING.md`. |  |
| [`prep_3d_inputs.py`](prep_3d_inputs.py) | Cut the encoding window's 3-D inputs from the pose window's CURRENT tables (moved from scratch 10-23, when 3-D became complete: 119 curated sessions, all four animals). |  |
| [`push_behaviour.py`](push_behaviour.py) | Backward / postural treadmill PUSHES as a behavioural readout, per session (Priya 10-23, option 3 of "let's do all 3": pushes become 2-5x more frequent after the stroke in 3/4 animals -- does that tra ... |  |
| [`rebuild_lick_maps.py`](rebuild_lick_maps.py) | Re-run ONLY the lick-dependent map steps for every registered session, with the ITI gate. |  |
| [`rerun_behaviour_figures.py`](rerun_behaviour_figures.py) | Regenerate EVERY per-session behaviour figure, for all registered dates. |  |
| [`side_review_clips.py`](side_review_clips.py) | Side-camera (cam2 + cam3) review clips for the shared side model: cut short trial windows in MODEL space, then render the model's predictions as one labelled clip, cam2 / cam3 side by side on the SAME ... |  |
| [`spout_reliability.py`](spout_reliability.py) | Per-spout reliability of the within-spout lick deviation, and DISATTENUATED per-spout slopes (Priya 10-21: "sure add the corrected version - but why not for far contra?"). |  |
| [`tongue_contact_recall.py`](tongue_contact_recall.py) | Recall of kept tongue licks against DAQ spout contacts (every contact is a lick) on the PS93 0908 cue clip, per model / cleaning cutoff. |  |

### diagnostic / check (63)

| script | what it does | tags |
|---|---|---|
| [`a3_lmm_imaging_stars.py`](a3_lmm_imaging_stars.py) | Heterogeneous-spread LMM (per post epoch vs pre: y ~ C(epoch) + (1/animal) + animal x epoch variance per epoch, t on df = n_animals - 1) on every epoch bar family's stored per-session values (<name>_s ... |  |
| [`cam1_clean_qc.py`](cam1_clean_qc.py) | QC figures for tuning cam1 clean traces (`docs/POSE_MODELS.md` D2; Priya 2026-10-18: "build tongue/jaw clean traces for cam1"). |  |
| [`cam1_model_compare.py`](cam1_model_compare.py) | Score cam1 pose models on the measures agreed with Priya 2026-10-18 (`docs/POSE_MODELS.md` Part C), on the cam1 trial clips (`scripts.session_poses clip --cam cam1` -> `session_poses/<animal>_<date>_c ... |  |
| [`camlog_label_audit.py`](camlog_label_audit.py) | Audit every session's ASSIGNED per-frame channel labels against the camera's own camlog LED record. |  |
| [`check_doc_links.py`](check_doc_links.py) | EVERY DOC CROSS-REFERENCE MUST RESOLVE. Run it after moving any document. |  |
| [`check_figure_layout.py`](check_figure_layout.py) | NO SIDECAR MAY SIT FLAT BESIDE A FIGURE. Run it against any figure output directory. |  |
| [`chronic_drift_vs_reorg.py`](chronic_drift_vs_reorg.py) | Drift vs reorganization, per animal. | cited by a results page |
| [`compare_map_tests.py`](compare_map_tests.py) | Old (raw-dF tests) vs new (pre-normalised tests) significant-bin counts, per figure x row x column (2026-10-21, `beta_maps.NORMALISE_TESTS_BY_PRE`; Priya: "archive the existing results so we don't los ... |  |
| [`component_exclusion_audit.py`](component_exclusion_audit.py) | Does the glue/bulb exclusion change the CD or the decoders? The measurements behind DECISIONS.md. |  |
| [`contact_dev_check.py`](contact_dev_check.py) | Direct check of the contact confound in the within-spout toward-midline slope (Priya 10-21). |  |
| [`contact_direction_predictability.py`](contact_direction_predictability.py) | Is spout CONTACT predictable from lick DIRECTION? (Priya 10-23, cheap check of why PS93's unique contact R2 is ~0 although PS93 misses most: if contact is predictable from the lick's direction error, ... |  |
| [`encoder_trialavg_r2.py`](encoder_trialavg_r2.py) | TRIAL-AVERAGED R^2 of an encoder run (Priya 10-22: "is this overall variance we're talking about or single trial?" -> "yes check"). |  |
| [`enl_video_spout_check.py`](enl_video_spout_check.py) | Does the video's pre-cue (ENL) position information come from the MOUSE or from the SPOUT? |  |
| [`exec_angle_spread_check.py`](exec_angle_spread_check.py) | Is the post-stroke rise in EXECUTED-angle encoding stronger tracking or just more lick-angle spread? (Priya 2026-10-19, on `position_dR2_maps_by_epoch.png`: the within-position executed-angle dR2 rise ... |  |
| [`exec_vs_neural_match.py`](exec_vs_neural_match.py) | EXECUTED best bin vs NEURAL best match, trial by trial (Priya 10-24: "can we make a decoder match vs executed best bin match confusion matrix?"). |  |
| [`face_rhythm_param_qc.py`](face_rhythm_param_qc.py) | Visual + factor QC of the FaceRhythm tracking-parameter test (`scripts/o2_face_rhythm.py paramtest`; 10-21). |  |
| [`face_rhythm_push_check.py`](face_rhythm_push_check.py) | Does the face do something at treadmill PUSHES that the pre-stroke FaceRhythm basis cannot describe? (Priya 10-24: "do we need a pre + post master fit, eg to capture the pushes we found in the encodin ... |  |
| [`figure_label_audit.py`](figure_label_audit.py) | DOES EVERY FIGURE THAT VARIES BY A PARAMETER SAY SO ON THE FIGURE? |  |
| [`frozen_fr_compare.py`](frozen_fr_compare.py) | FROZEN pre-stroke encoder WITH vs WITHOUT FaceRhythm (handoff 000t plan item 4, Priya 10-25: "should we be mixing frozen and refit encoder vs FR models" -> frozen-with-FR run). |  |
| [`imaging_baseline_compare.py`](imaging_baseline_compare.py) | Imaging August-only pre baseline vs the default (all curated pre): cell-by-cell comparison of every epoch-family data table both renders wrote (Priya 2026-10-18: "is the august-only baseline something ... |  |
| [`iti_check_figures.py`](iti_check_figures.py) | Figures for three checks that were table-only (Priya 10-27: "figure?"): rmse the where-decoder's error split into /bias/ and scatter (SD) per session (`iti_where_rmse_split.py`) f0 baseline fluorescen ... |  |
| [`iti_contact_check.py`](iti_contact_check.py) | Does spout CONTACT carry the frozen-decoder contrast of page 3.8 / 3.9 (ITI licks stay remapped, in-trial licks recover)? (handoff 10-26 B(i): "in-trial recovery may ride on spout contact".) |  |
| [`iti_decoder_patterns.py`](iti_decoder_patterns.py) | WHERE in cortex is lick position carried? Haufe activation patterns of the lick-position decoder, per epoch, as pixel maps (Priya 10-26: "compare per-session or per-epoch decoder weights by SVD map (a ... |  |
| [`iti_f0_check.py`](iti_f0_check.py) | Is the larger post-stroke ITI-lick dF/F response a smaller BASELINE (F0)? (handoff 00f to-do 2; pages 3.23-3.25) |  |
| [`iti_hemo_check.py`](iti_hemo_check.py) | Is the larger post-stroke ITI-lick response HAEMODYNAMIC, a FILTER effect, MOTION / EDGE / VESSEL, or an F0 artefact? |  |
| [`iti_june_ref.py`](iti_june_ref.py) | JUNE pre-stroke sessions as a NO-STROKE long-gap reference for the drift check (3.36; Priya 10-27: "can we try a version of 3.36 with the jun pre sessions too"). |  |
| [`iti_nmatch_check.py`](iti_nmatch_check.py) | Are ITI lick numbers a confound for the frozen-decoder result (page 3.8 / 3.9)? Counts per session by epoch; r vs n within epoch; frozen decoder (trained on all pre licks, pre LOSO) scored on test set ... |  |
| [`iti_spout_animal_panels.py`](iti_spout_animal_panels.py) | PER-ANIMAL per-spout Haufe maps (Priya 10-27: per-animal version of the per-spout maps; far contra + a comparison spout). |  |
| [`iti_zscore_check.py`](iti_zscore_check.py) | Within-session z-scored features for the frozen where-decoder (Priya 10-26: "z-scored within session is ok"; NOT a pre-lick baseline, because the inter-trial period carries position information). |  |
| [`kinematics_checks.py`](kinematics_checks.py) | Threshold-retune distributions + the three flagged checks (angle sign, jaw QC by position, velocity window), both models, on a cue clip or a session-level trial clip. |  |
| [`lick_aim_deficit.py`](lick_aim_deficit.py) | BEHAVIOURAL AIM DEFICIT per session, FIRST cued lick vs all licks, weighted vs far contra (Priya 10-27: switch the behaviour axis / mixed-model deficit to the first-lick weighted aim; "can I see a com ... |  |
| [`lick_contact_by_epoch.py`](lick_contact_by_epoch.py) | Per-lick MISS rate by spout side x epoch -- the PS92-95 check of the AR-HMM cohort's "licks and misses" result (github.com/htong1/Priya_ARHMM2026, PS46-50: contra miss-rate 25 % -> 40-90 % acute, ~45- ... |  |
| [`lick_floor_qc.py`](lick_floor_qc.py) | QC sheet for the tongue-detector amplitude floor (`orofacial_kinematics.tongue.detector.min_peak_y_abs`, 20 px, an OLD-RIG value marked retune). |  |
| [`lick_trajectories_3d.py`](lick_trajectories_3d.py) | 3-D LICK TRAJECTORIES by target spout and epoch (Priya 10-24: "can we compare trajectory instead of landing for the licks?"). LICK DATA only. |  |
| [`model_lick_compare.py`](model_lick_compare.py) | Compare pose models on the measures the analyses use: lick detection against DAQ spout contacts, and tongue angle at peak protrusion by spout position -- the same kinematics pipeline (`pose_kinematics ... |  |
| [`orofacial_svg_backfill.py`](orofacial_svg_backfill.py) | Re-draw every session's qc.png from its saved outputs so each also gets svg/qc.svg (Priya 10-17: PNG + SVG for all orofacial figures). |  |
| [`pose3d_check_sheet.py`](pose3d_check_sheet.py) | Video check sheet for three open 3-D questions (handoff 10-20 §4a / §7; Priya 10-20 "let's do 1 and 2"). |  |
| [`pose3d_crosscheck.py`](pose3d_crosscheck.py) | Do the cam4 (2-D) cohort lick measures agree with the 3-D tongue (`wfield_local.pose3d`)? Lick by lick, then pre -> post. |  |
| [`pose3d_jaw_crosscheck.py`](pose3d_jaw_crosscheck.py) | JAW cross-check, trial by trial: cam4 (2-D, v36 `cohort/tables/jaw_lateral/trials.csv`) vs 3-D (`pose3d_kinematics` jaw_trials). Priya 10-19: "is jaw 3d also good?" -- run the jaw cross check. |  |
| [`pose3d_jaw_gap_check.py`](pose3d_jaw_gap_check.py) | Held-out JAW fill error vs gap length (Priya 10-19: is 24 ms -- set for the fast tongue -- right for the slow jaw?). |  |
| [`pose3d_kinematics.py`](pose3d_kinematics.py) | The cam4 cohort's tongue / jaw readouts redone in 3-D (Priya 10-19: cross-check, then direction vs target, reach + speed, jaw opening), on the SAME lick events as the cam4 cohort. |  |
| [`pose3d_side_check.py`](pose3d_side_check.py) | Side cameras in the 3-D pose: per-session PnP of cam2 / cam3 and the cross-camera OFFSET check (`wfield_local.pose3d_side`). |  |
| [`pose3d_side_video.py`](pose3d_side_video.py) | Four-camera 3-D check clip: cam1 / cam4 / cam2 / cam3 raw views with their 2-D keypoints and the FUSED 3-D point reprojected, and the 3-D pose below (Priya 2026-11-04: "show me clips labelled in all 4 ... |  |
| [`pose3d_vs_cam4.py`](pose3d_vs_cam4.py) | Does 3-D add information over cam4 2-D? Two tests on the SAME licks (Priya 10-19: "run the 2d vs 3d on what we have"). |  |
| [`prelim_numbers_frozen_vs_refit.py`](prelim_numbers_frozen_vs_refit.py) | LOST or MISREAD: the exact numbers behind the 5r table in `docs/PRELIM_DATA_VLS_STROKE.md`. |  |
| [`qc_running_clips.py`](qc_running_clips.py) | Side-camera clips with the SIGNED treadmill trace under each frame (Priya 10-23: "the cue-locked speed increases could either be the treadmill 'rebound' ... or real walking ... |  |
| [`qc_running_lick_jitter.py`](qc_running_lick_jitter.py) | Check 1 (Priya 10-23): are low treadmill speeds lick / cue jitter? Per session, treadmill /speed/ on a 31.25 Hz grid over the engaged-trial span; per speed band, the fraction of samples within +/-0.5 ... |  |
| [`qc_running_push_survey.py`](qc_running_push_survey.py) | Backward-push survey per animal x epoch (trial frames): how often (fraction of frames with backward speed > PUSH_FLOOR) and how hard, vs the pre-stroke push cap. |  |
| [`qc_running_speed.py`](qc_running_speed.py) | QC (Priya 10-23): SIGNED treadmill speed vs cortex-mean dF/F per session -- forward > 1.5 mm/s and backward < -0.3 mm/s shaded, cues marked; whole session + 3-min zoom + signed speed histogram + corte ... |  |
| [`qc_running_survey.py`](qc_running_survey.py) | Running-speed survey per animal x epoch (trial frames, lag-0 /speed/) for choosing the frozen-encoder running cap. |  |
| [`relabel_confusion.py`](relabel_confusion.py) | ENCODING-MODEL confusion matrices from the relabel test (Priya 10-24: "compare neural encoding with real neural data to see if post-stroke neural data looks like a different spout"). |  |
| [`remap_checks.py`](remap_checks.py) | Checks on the per-session vs frozen REMAPPING maps (Priya 10-25: "this is striking, can we review" -> checks 1, 2): |  |
| [`session_vs_frozen.py`](session_vs_frozen.py) | PER-SESSION vs FROZEN "remapping" (Priya 10-25: "compare the per-session and frozen results too for remapping"). |  |
| [`side_clean_qc.py`](side_clean_qc.py) | Side-camera (cam2 / cam3) CLEANING QC: raw confident keypoints vs what `pose3d_side.side_clean` keeps, as a clip (Priya 2026-11-04: QC the side cleaning before adopting it, as for cam1). |  |
| [`side_label_offsets.py`](side_label_offsets.py) | Cross-camera keypoint DEFINITIONS from the student's own labels (Priya 2026-11-04: "plan to compare the student labels"): at the same DAQ moment labelled in a side camera AND in cam1 + cam4, where doe ... |  |
| [`slowband_decode_check.py`](slowband_decode_check.py) | Does the SLOW band of the adopted corrected signal carry decodable spout position? (Priya 2026-10-19.) | cited by a results page |
| [`softmax_neural_vs_lick.py`](softmax_neural_vs_lick.py) | NEURAL vs LICK softmax confusion matrices (Priya 10-24: "are you making the softmax neural confusion matrix? maybe bins 0.5 and 1"). |  |
| [`spout_tuning_check.py`](spout_tuning_check.py) | Follow-ups to the per-spout toward-midline results (Priya 10-21, items 2-3), cam4 lick-aligned, same trials / response windows / drift filter as scripts/within_spout_deviation.py. |  |
| [`state_spout_check.py`](state_spout_check.py) | Why is PS94's target-unique R2 ~10x the other animals'? (Priya 10-25.) Two checks, pre-stroke August sessions: |  |
| [`tongue_nocontact_qc.py`](tongue_nocontact_qc.py) | QC of kept tongue licks that never touched the spout (no DAQ contact): real incomplete licks or false detections? The DAQ cannot tell -- an incomplete lick makes no contact -- so this is judged on vid ... |  |
| [`tongue_qc_steps.py`](tongue_qc_steps.py) | Visual QC of every tongue stage on our data, before retuning a threshold: raw -> clean -> v7 pre-clean -> detectors -> gates -> kept licks, plus the video frame at every rejected peak. |  |
| [`trial_neural_vs_exec.py`](trial_neural_vs_exec.py) | TRIAL-BY-TRIAL: do far-contra licks that look like far mid come with cortex that looks like far mid? (Priya 10-24: "are the licks that look like far mid instead of far contra corresponding to cortical ... |  |
| [`within_spout_deviation.py`](within_spout_deviation.py) | Within-spout executed-deviation check (Priya 10-20, agreed): with the spout FIXED, does the cortical response shift with where the tongue actually went? |  |

## scripts/cohort2_qc (74)


### figure producer (33)

| script | what it does | tags |
|---|---|---|
| [`cohort2_qc/attempt_effort.py`](cohort2_qc/attempt_effort.py) | Do animals "try harder" with the body when they cannot lick? Running and limb cortex by trial outcome (cohort1, 11-01). |  |
| [`cohort2_qc/channel_drift.py`](cohort2_qc/channel_drift.py) | Where does cohort2's surviving slow drift come from -- 470 itself, or 415 bleaching leaked in through the hemo coefficient? DAQ-correct channels only (2026-10-30). |  |
| [`cohort2_qc/discharge_screen.py`](cohort2_qc/discharge_screen.py) | Epileptiform-discharge screen for the RiboL1 mice (PS121, PS122), all cohort2 sessions on MICROSCOPE. |  |
| [`cohort2_qc/epileptiform_gallery.py`](cohort2_qc/epileptiform_gallery.py) | Are the epileptiform_scan candidates PS122-like discharges or ordinary lick responses? (2026-10-31) |  |
| [`cohort2_qc/epileptiform_videos.py`](cohort2_qc/epileptiform_videos.py) | Combined video (470 map / cam4 face / traces) for the epileptiform candidates beyond PS122 (2026-10-31). |  |
| [`cohort2_qc/fast_events.py`](cohort2_qc/fast_events.py) | Large, fast, 470-only cortex-wide events in cohort2 (2026-10-31). |  |
| [`cohort2_qc/indicator_kinetics_spread.py`](cohort2_qc/indicator_kinetics_spread.py) | GCaMP6s vs jGCaMP8s with the SENSOR KINETICS equalised, and how far the responses SPREAD over cortex (2026-10-31). |  |
| [`cohort2_qc/native_maps_1029.py`](cohort2_qc/native_maps_1029.py) | Native-space (no Allen) cue / lick maps for the 10-29 cohort2 sessions (Priya 2026-10-30: "can we just see raw maps without allen alignment"). |  |
| [`cohort2_qc/processing_stages.py`](cohort2_qc/processing_stages.py) | Processing-stage cascade of the cortex-mean signal, cohort2 (the real meegkit_hpfit pipeline). |  |
| [`cohort2_qc/quiet_area_spikes.py`](cohort2_qc/quiet_area_spikes.py) | Spontaneous activity spikes in orofacial vs limb cortex during QUIET periods, cohort1 pre vs post stroke (2026-10-31). |  |
| [`cohort2_qc/quiet_breathing.py`](cohort2_qc/quiet_breathing.py) | T9 (breathing artefact): is the post-stroke rise of quiet-period orofacial 1-4 Hz power (peak ~2-3 Hz) breathing? (cohort1 PS92-95 only, 2026-11-01; handoff STATUS_2026-11-01_QUIET_EVENTS_HANDOFF sect ... |  |
| [`cohort2_qc/quiet_eta_fr.py`](cohort2_qc/quiet_eta_fr.py) | Movement-corrected event-triggered maps: face movement regressed out of the PIXEL movie (cohort1, 2026-11-01). | cited by a results page |
| [`cohort2_qc/quiet_eta_split.py`](cohort2_qc/quiet_eta_split.py) | Quiet-event maps split by face movement x height x polarity x state, and per-event ipsi / contra timing (cohort1, 11-01). |  |
| [`cohort2_qc/quiet_event_movie.py`](cohort2_qc/quiet_event_movie.py) | Side-by-side widefield dF/F movie of engaged QUIET periods, PRE vs ACUTE, same animal (cohort1, 2026-11-02). |  |
| [`cohort2_qc/quiet_event_width.py`](cohort2_qc/quiet_event_width.py) | Are quiet-period fluctuations FASTER after the stroke? Event width + trace autocorrelation (cohort1, 2026-11-01). |  |
| [`cohort2_qc/quiet_f0_noise.py`](cohort2_qc/quiet_f0_noise.py) | T22: per-area baseline fluorescence F0 (tonic proxy) + the noise floor under T20's fast band (cohort1, 2026-11-01). |  |
| [`cohort2_qc/quiet_fluct.py`](cohort2_qc/quiet_fluct.py) | What changes in quiet-period cortical activity after the stroke: spectrum, where (pixel map), and event shape (cohort1, 2026-11-01). | cited by a results page |
| [`cohort2_qc/quiet_hemi_amp_recovery.py`](cohort2_qc/quiet_hemi_amp_recovery.py) | Quiet-event tests that need no imaging re-load: T8 ipsi vs contra, T11 amplitude / epileptiform, T14-T15 recovery link (cohort1, 2026-11-01). |  |
| [`cohort2_qc/quiet_lick_lfo.py`](cohort2_qc/quiet_lick_lfo.py) | T16: lick-locked low-frequency (1-4 Hz) activity vs the quiet-period 1-4 Hz increase, same sessions (cohort1, 11-01). |  |
| [`cohort2_qc/quiet_recovery_link.py`](cohort2_qc/quiet_recovery_link.py) | Do quiet-period measures track behavioural recovery? (cohort1, 2026-11-01) |  |
| [`cohort2_qc/quiet_spikes_examples.py`](cohort2_qc/quiet_spikes_examples.py) | Example rest traces with the spike threshold drawn, pre vs acute per animal (cohort1, 2026-11-01). |  |
| [`cohort2_qc/quiet_spikes_fr.py`](cohort2_qc/quiet_spikes_fr.py) | Rest-period orofacial / limb cortex spikes AFTER regressing out face movement (FaceRhythm), pooled per epoch with statistics (cohort1, 2026-10-31). |  |
| [`cohort2_qc/quiet_spikes_v2.py`](cohort2_qc/quiet_spikes_v2.py) | Quiet-period cortical events v2: height detector, Priya's quiet rule, threshold chosen by sensitivity / specificity (cohort1, 2026-11-01). | cited by a results page |
| [`cohort2_qc/quiet_spikes_v2_examples.py`](cohort2_qc/quiet_spikes_v2_examples.py) | Example traces for the v2 quiet-event detector with the +/- thresholds drawn (cohort1, 2026-11-01). | cited by a results page |
| [`cohort2_qc/quiet_spikes_v2_fr.py`](cohort2_qc/quiet_spikes_v2_fr.py) | Quiet-period cortical events AFTER regressing out face movement, on the v2 definitions (cohort1, 2026-11-01). | cited by a results page |
| [`cohort2_qc/quiet_state_response.py`](cohort2_qc/quiet_state_response.py) | T21: is the acute ~2x larger lick-evoked cortical response a consequence of a SYNCHRONISED cortical state? (cohort1, 11-01) |  |
| [`cohort2_qc/quiet_stillness.py`](cohort2_qc/quiet_stillness.py) | T7: within engaged quiet, does the orofacial event rate rise with face STILLNESS? (cohort1, 2026-11-01) |  |
| [`cohort2_qc/quiet_sync_4to8.py`](cohort2_qc/quiet_sync_4to8.py) | T20b (EXPLORATORY, PROVISIONAL): T20-type synchronisation laterality with a fast band free of the 7-11 Hz artefact. (cohort1 PS92-95, 2026-11-02; quiet-events handoff NEXT item 3.) |  |
| [`cohort2_qc/quiet_sync_index.py`](cohort2_qc/quiet_sync_index.py) | T20: is quiet cortex more SYNCHRONISED after the stroke? (cohort1, 2026-11-01) |  |
| [`cohort2_qc/quiet_task_events.py`](cohort2_qc/quiet_task_events.py) | Quiet-event detector inside TASK windows (T12) and pre-cue events vs the trial's outcome / position code (T13) (cohort1 PS92-95, 2026-11-01; docs/status/STATUS_2026-11-01_QUIET_EVENTS_HANDOFF.md secti ... |  |
| [`cohort2_qc/quiet_task_events_followup.py`](cohort2_qc/quiet_task_events_followup.py) | T13 follow-up: is the weaker frozen-decoder position code on trials with a pre-cue orofacial event a CONTAMINATION of the decoder's features by the event transient, or a weaker PLAN? (cohort1 PS92-95, ... |  |
| [`cohort2_qc/spout_maps.py`](cohort2_qc/spout_maps.py) | Per-spout cue- and first-lick-aligned maps, cohort2, every session that has its DAQ .h5 on the share (2026-10-31). |  |
| [`cohort2_qc/stage_traces.py`](cohort2_qc/stage_traces.py) | Per-session brain-mean traces at each processing stage, cohort2 2026-10-29. |  |

### pipeline step (1)

| script | what it does | tags |
|---|---|---|
| [`cohort2_qc/swap_scan.py`](cohort2_qc/swap_scan.py) | Scan ANY session (cohort1 or cohort2) for frames that carry the other LED's image, from the SVD products only. | imported or run by wfield_local |

### analysis (tables / numbers) (18)

| script | what it does | tags |
|---|---|---|
| [`cohort2_qc/brainmask.py`](cohort2_qc/brainmask.py) | Brain mask for native-space cohort2 analysis (2026-10-30). |  |
| [`cohort2_qc/cohort1_impact.py`](cohort2_qc/cohort1_impact.py) | How much do the swapped frames + the 16 ms 415 lag change COHORT1 results? (Priya 2026-10-30: "how much is cohort 1 data affected"). |  |
| [`cohort2_qc/early_fade.py`](cohort2_qc/early_fade.py) | Early-fade / slow-bleach residual on cohort2 adopted SVTcorr (hemo_meegkit_hpfit), native SVD space. |  |
| [`cohort2_qc/epileptiform_scan.py`](cohort2_qc/epileptiform_scan.py) | Scan every recording (cohort1 + cohort2) for epileptiform-like discharges (2026-10-31). |  |
| [`cohort2_qc/iso_phase.py`](cohort2_qc/iso_phase.py) | Is the RiboL1-8s 415 BELOW its isosbestic point, i.e. does 415 fall when calcium rises? (2026-10-31) |  |
| [`cohort2_qc/iso_snr.py`](cohort2_qc/iso_snr.py) | Absolute 415/470 brain-ROI counts and shot-noise ΔF/F floor, cohort2 2026-10-29. |  |
| [`cohort2_qc/iso_vessel_split.py`](cohort2_qc/iso_vessel_split.py) | Calcium in the 8s 415? Split the 415 lick response into VESSEL and PARENCHYMA pixels (2026-10-31). |  |
| [`cohort2_qc/lp_licks.py`](cohort2_qc/lp_licks.py) | LP (video) lick and jaw-onset times on the DAQ clock, whole session, for the quiet-event chain (cohort1, 2026-11-01). |  |
| [`cohort2_qc/perpixel_snr.py`](cohort2_qc/perpixel_snr.py) | Per-pixel corrected-signal amplitude (fair across spatial scales) vs baseline F, cohort2. |  |
| [`cohort2_qc/quiet_modes.py`](cohort2_qc/quiet_modes.py) | Orofacial vs limb cortex alternation during QUIET periods, pre vs post stroke (cohort1, 2026-10-31). |  |
| [`cohort2_qc/quiet_quit_state.py`](cohort2_qc/quiet_quit_state.py) | Do the animals look ASLEEP after they quit? Quit-period quiet vs engaged quiet, cortex and face (cohort1, 2026-11-01). |  |
| [`cohort2_qc/slow_is_bleach.py`](cohort2_qc/slow_is_bleach.py) | Is the slow band that survives meegkit_hpfit the raw differential bleach, or real slow signal? |  |
| [`cohort2_qc/slow_spectrum.py`](cohort2_qc/slow_spectrum.py) | Where does the surviving slow variance live? Cumulative variance of the corrected cortex-mean trace removed by a cosine high-pass at a range of cutoff PERIODS, cohort2 adopted SVTcorr. |  |
| [`cohort2_qc/slowband_behaviour.py`](cohort2_qc/slowband_behaviour.py) | Is the slow band that survives meegkit_hpfit behaviour (real slow signal) or drift? cohort2 10-29, 2026-10-30. |  |
| [`cohort2_qc/swapmask_1029.py`](cohort2_qc/swapmask_1029.py) | Exclude the other-LED ("swapped") frames from the 10-29 cohort2 analysis (Priya 2026-10-30: "for our current non-cam-log analysis exploration, can we exclude the swapped frames from 10/29 analysis"). |  |
| [`cohort2_qc/swapmask_cohort1.py`](cohort2_qc/swapmask_cohort1.py) | Cohort1 sweep: swapped-frame mask + swap-masked hemo variant for every registered cohort1 session (Priya 2026-10-30: "let's set up the cohort 1 sweep"). |  |
| [`cohort2_qc/swapped_frames.py`](cohort2_qc/swapped_frames.py) | Channel-SWAPPED frames in the clean-pairs frame map, and what they do to the hemo coefficient. cohort2 10-29. |  |
| [`cohort2_qc/var_bands.py`](cohort2_qc/var_bands.py) | Absolute variance of the corrected cortex-mean trace by period band, cohort2 adopted SVTcorr. |  |

### diagnostic / check (22)

| script | what it does | tags |
|---|---|---|
| [`cohort2_qc/append_qc_slides.py`](cohort2_qc/append_qc_slides.py) | Append titled full-slide images to an existing .pptx (used to add the epileptiform-discharge screen and the 490-vs-470 LED panels to the cohort2 preprocessing deck after preprocess_deck.py builds it). |  |
| [`cohort2_qc/build_lag_variants_1029.py`](cohort2_qc/build_lag_variants_1029.py) | Build the lag-fix / fit-band hemo CANDIDATES for the 10-29 sessions, swap-masked (2026-10-30, §0g of the QC doc). Writes hemo_<variant>_refitT_swapmask/ per session; the adopted product is untouched. |  |
| [`cohort2_qc/cohort1_subset_headline.py`](cohort2_qc/cohort1_subset_headline.py) | Cohort1 epoch-subset check: do epoch-level results move with the swap-masked + 415-lag-fixed correction? (Priya 2026-10-30: "can we do this with a subset of sessions"). |  |
| [`cohort2_qc/compare_hemo_variants_1029.py`](cohort2_qc/compare_hemo_variants_1029.py) | Compare hemo variants on the 10-29 sessions (QC doc §0g): adopted, swap-masked, and the lag-fix / fit-band candidates. |  |
| [`cohort2_qc/fit_vs_output_demo.py`](cohort2_qc/fit_vs_output_demo.py) | Why the pipeline FITS on the 0.1 Hz high-pass (C) but OUTPUTS through the meegkit detrend (D). |  |
| [`cohort2_qc/indicator_compare.py`](cohort2_qc/indicator_compare.py) | jGCaMP8s (RiboL1: PS121, PS122) vs GCaMP6s (Thy1: PS118; cohort1 pre-stroke sessions as a same-rig reference) -- metrics that do NOT depend on raw dF/F amplitude (Priya 2026-10-30: "how to best analyz ... |  |
| [`cohort2_qc/indicator_figure.py`](cohort2_qc/indicator_figure.py) | Figure for the 8s-vs-6s comparison (QC doc §0j): first-lick traces, maps, spectra, spatial correlation. |  |
| [`cohort2_qc/iso_check.py`](cohort2_qc/iso_check.py) | Is 415 isosbestic for RiboL1-jGCaMP8s? And is the 415 still usable at the end of the session? cohort2 10-29. |  |
| [`cohort2_qc/led_490_vs_470.py`](cohort2_qc/led_490_vs_470.py) | 490 nm vs 470 nm functional LED for the RiboL1 mice (PS121, PS122), cohort2. |  |
| [`cohort2_qc/masked_vs_adopted_1029.py`](cohort2_qc/masked_vs_adopted_1029.py) | 10-29: the swap-masked hemo correction vs the adopted one -- does excluding the other-LED frames fix the RiboL1 correction, and what is left of the slow drift? (Priya 2026-10-30: "resume the 10/29 exp ... |  |
| [`cohort2_qc/prestroke_quiet_videos.py`](cohort2_qc/prestroke_quiet_videos.py) | Pre-stroke PS92 / PS93 videos over QUIET late-session windows, to compare with the post-stroke events (2026-10-31). |  |
| [`cohort2_qc/quiet_hemo_check.py`](cohort2_qc/quiet_hemo_check.py) | T10 (quiet-events handoff 11-01): are the post-stroke quiet-period orofacial events / 2-3 Hz power CALCIUM, or blood / light? The same measures on the corrected 470 (adopted), the UNCORRECTED 470 and ... |  |
| [`cohort2_qc/quiet_lp_sensitivity.py`](cohort2_qc/quiet_lp_sensitivity.py) | Sensitivity check: do the quiet-period results survive excluding NON-CONTACT (video-only) licks? (cohort1, 2026-11-01) |  |
| [`cohort2_qc/quiet_steinmetz_check.py`](cohort2_qc/quiet_steinmetz_check.py) | T19: do the quiet orofacial events meet the Steinmetz et al. 2017 interictal criteria? (cohort1, 2026-11-01) |  |
| [`cohort2_qc/raw470_vs_corrected.py`](cohort2_qc/raw470_vs_corrected.py) | Is the RiboL1-8s signal small because the 415 correction removes it, or already in raw 470? (2026-10-31) |  |
| [`cohort2_qc/running_sign_check.py`](cohort2_qc/running_sign_check.py) | Running-onset response in raw 470, raw 415 and corrected, pre vs post stroke (2026-10-31). |  |
| [`cohort2_qc/saturation_check.py`](cohort2_qc/saturation_check.py) | IMAGING BOX: are any pixels clipped at the camera ceiling (65535)? READ-ONLY on the raw .dat. |  |
| [`cohort2_qc/seizure_events_video.py`](cohort2_qc/seizure_events_video.py) | Video + traces of the fast 470-only events in PS122 10-29 (possible epileptiform discharges; QC doc §0o). |  |
| [`cohort2_qc/session_qc.py`](cohort2_qc/session_qc.py) | Per-session QC of the cohort2 sessions processed with the pre-SVD repair + clip mask (2026-10-31). |  |
| [`cohort2_qc/spout_compare_1029.py`](cohort2_qc/spout_compare_1029.py) | Per-spout coding, PS118 (Thy1-GCaMP6s) vs PS121 / PS122 (RiboL1-jGCaMP8s), with PS120 as the blood-flow-only floor (Priya 2026-10-30: "what about per-spout 118 vs 121-122"). |  |
| [`cohort2_qc/swap_raw_check.py`](cohort2_qc/swap_raw_check.py) | IMAGING BOX: look at the RAW frames and the camlog around the other-LED ("swapped") frames. READ-ONLY on raw data. |  |
| [`cohort2_qc/usability_qc.py`](cohort2_qc/usability_qc.py) | (no docstring) edge coverage: does brain reach within 10 px of the frame border? |  |

## scripts/imaging_page (1)


### figure producer (1)

| script | what it does | tags |
|---|---|---|
| [`imaging_page/s_per_animal.py`](imaging_page/s_per_animal.py) | Imaging page Fig 3.3 (`s_per_animal`): the pre-stroke position TEMPLATE and the frozen BEHAVIOURAL-STATE decoder, per animal across epochs -- redrawn under Priya's 11-03 categorical-x rules (no lines ... |  |

## scripts/rest_migration (43)


### figure producer (12)

| script | what it does | tags |
|---|---|---|
| [`rest_migration/channel_evoked_sign.py`](rest_migration/channel_evoked_sign.py) | WHICH WAY DOES 415 MOVE AFTER A CUE, AND WHEN? The sign of the haemodynamic term, measured. |  |
| [`rest_migration/channel_vessel_sign.py`](rest_migration/channel_vessel_sign.py) | IS THE CUE-EVOKED 415 RESPONSE VESSEL-SHAPED OR PARENCHYMA-SHAPED? The decisive test. |  |
| [`rest_migration/evoked_hrf_latency.py`](rest_migration/evoked_hrf_latency.py) | THE HAEMODYNAMIC DIP, TIMED DIRECTLY: does the evoked HRF slow after the stroke? |  |
| [`rest_migration/fix_mask_names.py`](rest_migration/fix_mask_names.py) | Rename the rebuilt masks to the TAGGED convention the rest of the pipeline constructs. |  |
| [`rest_migration/nvc_evoked.py`](rest_migration/nvc_evoked.py) | Cue-evoked RAW 415 and 470, at frame resolution -- task-evoked haemodynamics, and a test of whether 415 is isosbestic for THIS cohort. |  |
| [`rest_migration/plot_session_residual.py`](rest_migration/plot_session_residual.py) | Plot the whole-session signal BEFORE and AFTER drift correction, brain-masked spatial mean. |  |
| [`rest_migration/regen_running_maps.py`](rest_migration/regen_running_maps.py) | Rebuild the rest/running SVD activity maps for the preprocessing deck, on the REST baseline. |  |
| [`rest_migration/rest_coupling.py`](rest_migration/rest_coupling.py) | 415/470 coupling DURING REST, by hemisphere, with a LAG. The three extensions at once. |  |
| [`rest_migration/rest_null.py`](rest_migration/rest_null.py) | Does each animal change the same PART of cortex? The cross-position null, on the REST baseline. |  |
| [`rest_migration/rotation_stats.py`](rest_migration/rotation_stats.py) | 15h post-hoc statistics: the COHORT cosine and the FAMILY-WISE threshold. |  |
| [`rest_migration/subacute_daymatched.py`](rest_migration/subacute_daymatched.py) | Is the SUBACUTE between-animal non-replication biology, or is it the BIN? |  |
| [`rest_migration/substitute_code.py`](rest_migration/substitute_code.py) | Is the chronic rest code a REPLACEMENT for the pre-stroke one, or an ADDITION to it? |  |

### pipeline step (16)

| script | what it does | tags |
|---|---|---|
| [`rest_migration/channel_position_maps.py`](rest_migration/channel_position_maps.py) | 415 nm vs RAW 470 nm vs CORRECTED 470 nm, as spout-position maps. Pre-stroke sessions. | imported or run by wfield_local; cited by deck_registry |
| [`rest_migration/engagement_decomposition.py`](rest_migration/engagement_decomposition.py) | WHY does engagement fall after the stroke? Behaviour only -- no imaging. | imported or run by wfield_local |
| [`rest_migration/flatpool_vs_restw.py`](rest_migration/flatpool_vs_restw.py) | Does POSITION-WEIGHTING earn its place, with the TEMPORAL axis held fixed? | imported or run by wfield_local |
| [`rest_migration/lick_bout_structure.py`](rest_migration/lick_bout_structure.py) | IS THE WITHIN-SESSION DECLINE FATIGUE AFTER ALL? Three tests the median ILI cannot do. | imported or run by wfield_local |
| [`rest_migration/meanref_balance.py`](rest_migration/meanref_balance.py) | IS THE `MEANref` REFERENCE POSITION-WEIGHTED? No — and this measures what that costs. | imported or run by wfield_local |
| [`rest_migration/quit_point.py`](rest_migration/quit_point.py) | WHERE does the animal quit, and what has accumulated by then? The step, not the slope. | imported or run by wfield_local |
| [`rest_migration/quit_prodrome.py`](rest_migration/quit_prodrome.py) | LICK RATE ACROSS THE SESSION, and whether a quit has a PRODROME. | imported or run by wfield_local |
| [`rest_migration/reference_family_figure.py`](rest_migration/reference_family_figure.py) | The `epoch_15k` figure: the three reference families as region x position cohort maps. | imported or run by wfield_local |
| [`rest_migration/reference_family_roi.py`](rest_migration/reference_family_roi.py) | WHERE the position map changes, in Allen ROIs, under THREE references with DIFFERENT failure modes. | imported or run by wfield_local |
| [`rest_migration/rest_baseline_epoch_drift.py`](rest_migration/rest_baseline_epoch_drift.py) | Does the `restw` baseline on `restdock05` DRIFT ACROSS EPOCHS? The narrow question §0b leaves open. | imported or run by wfield_local |
| [`rest_migration/rest_frozen_decoder.py`](rest_migration/rest_frozen_decoder.py) | THE DOCKED FROZEN DECODER: does the PRE-STROKE rest code survive, or is it replaced? | imported or run by wfield_local; cited by a results page |
| [`rest_migration/rest_position_permutation.py`](rest_migration/rest_position_permutation.py) | Does REST carry POSITION information, tested against a null that contains the drift confound? | imported or run by wfield_local |
| [`rest_migration/rest_position_vs_drift.py`](rest_migration/rest_position_vs_drift.py) | Is rest's position dependence POSITION, or session DRIFT aliased through the block structure? | imported or run by wfield_local |
| [`rest_migration/rotation_maps.py`](rest_migration/rotation_maps.py) | WHERE the position code rotates -- Haufe decoder PATTERNS in the shared joint basis. | imported or run by wfield_local |
| [`rest_migration/shared_position_projection.py`](rest_migration/shared_position_projection.py) | `15s` — HOW MUCH OF A POSITION'S TASK MAP IS ALREADY PRESENT IN ITS OWN REST? | imported or run by wfield_local |
| [`rest_migration/transfer_arms.py`](rest_migration/transfer_arms.py) | The cross-epoch transfer matrix on ALL FOUR windows: ENL, post-cue, post-lick and rest. | imported or run by wfield_local |

### analysis (tables / numbers) (13)

| script | what it does | tags |
|---|---|---|
| [`rest_migration/cohort_summary.py`](rest_migration/cohort_summary.py) | What the FINAL rest definition actually produced, read off the written masks. |  |
| [`rest_migration/position_survival.py`](rest_migration/position_survival.py) | Do the behavioural exclusions remove ITIs EVENLY across positions, or do they track the deficit? |  |
| [`rest_migration/publish_cache.py`](rest_migration/publish_cache.py) | Publish this box's CURRENT-VERSION session cache to MICROSCOPE, so the nightly box inherits it. |  |
| [`rest_migration/recompute_masks.py`](rest_migration/recompute_masks.py) | Build the REST masks for every registered session, into `quiet_<tag>_<variant>/`. |  |
| [`rest_migration/regen_events.py`](rest_migration/regen_events.py) | Rebuild every session's canonical behavior-events npz on the REST definition (schema v3). |  |
| [`rest_migration/regen_lick_maps.py`](rest_migration/regen_lick_maps.py) | Rebuild the lick-aligned REST-normalised maps, so the preprocessing deck stops mixing baselines. |  |
| [`rest_migration/relaxed_buffer_percentile.py`](rest_migration/relaxed_buffer_percentile.py) | Can a LOWER PERCENTILE buy back what a RELAXED lick buffer gives away? |  |
| [`rest_migration/rest_block_boundary.py`](rest_migration/rest_block_boundary.py) | PERSISTENCE or ANTICIPATION? The rest periods at BLOCK BOUNDARIES separate them. |  |
| [`rest_migration/rest_carries_position.py`](rest_migration/rest_carries_position.py) | Does the REST baseline itself carry POSITION information? The control the reference assumes. |  |
| [`rest_migration/rest_position_decode.py`](rest_migration/rest_position_decode.py) | REST AS SIGNAL, not as baseline: can spout position be DECODED from the inter-trial interval? |  |
| [`rest_migration/restw_column_drops.py`](rest_migration/restw_column_drops.py) | Which sessions LOSE their RESTW column under the settled rest definition, and why -- by EPOCH. |  |
| [`rest_migration/settle.py`](rest_migration/settle.py) | How long a settle does the REST window need after the response window closes? |  |
| [`rest_migration/state_time_bins.py`](rest_migration/state_time_bins.py) | Can the state decoder be balanced across SESSION TIME? The feasibility count, before the fix. |  |

### diagnostic / check (2)

| script | what it does | tags |
|---|---|---|
| [`rest_migration/flat_vs_timelocal.py`](rest_migration/flat_vs_timelocal.py) | Does the TIME-LOCAL rest baseline change any CONCLUSION, or only the third decimal? |  |
| [`rest_migration/uniform_shift_check.py`](rest_migration/uniform_shift_check.py) | THE THING POSITION SPREAD CANNOT SEE: does relaxing the lick buffer shift the baseline UNIFORMLY? |  |
