# Block permutation audit (2026-11-16, window a2, branch analysis/tim-encdesign-1116)

**Decision (Priya 11-16):** "we should use block permutation everywhere it makes sense". Spout positions come in BLOCKS
of 4-8 consecutive trials (`wfield_local/block_ids.py`). A null that permutes spout labels (or anything that follows the
block: trial target, previous-trial target) trial by trial destroys the within-block correlation, so the null is easier
than the data and p values are optimistic. The same holds for circular time shifts short enough to stay inside the
same block, and for cross-validation folds that are not grouped by block.

Status: AUDIT ONLY. No analysis code changed yet. Nothing below says a published result is wrong; it says which
nulls must be re-run block-wise before their p values are quoted again. Magnitude is unmeasured (first step below).

Read-only sweep of the 154 shuffle / permutation / roll sites in `wfield_local/` and `scripts/` (archives excluded),
each classified by reading the code; the decoder null was re-checked by hand.

## The central item

`wfield_local/nolick_analysis.py:109 permutation_null` (through `evaluate_arm`) permutes `y_true` TRIAL-WISE against the
fixed cross-validated predictions. The decoders' CV is already block-grouped (`defaults.yaml decode.cv: block`); only
the label null is trial-wise. Callers: `enl_decode` (432 / 524 / 535), `nolick_decoder` (all arms), `locanmf_position_
decoder` 719, `locanmf_frozen_decoder` 1020, `poststroke_compare` 859 / 1642, `enl_lick_control` 147,
`scripts/enl_movement_removed_decode` 125, `null_potent_session` 298, `component_exclusion_audit` 224. Block-aware
versions already exist: `decode_ci.frozen_ci` (inline block -> label map) and `rest_frozen_decoder --null blockperm`.

## A. Trial-wise nulls over spout (or block-following) labels -> block permutation

| site | what is permuted | feeds | arm |
|---|---|---|---|
| nolick_analysis.py:127 `permutation_null` | spout labels vs fixed CV predictions | ENL / no-lick / frozen / post-stroke / LocaNMF decoder p values | decoders |
| locanmf_frozen_decoder.py:1000 | spout labels, shuffled-label refit (LOGO) | frozen-decoder entropy / max-prob baseline | decoders |
| combined_encoding.py:1658 `shuf:position` (+ shufpos-*) | engaged-trial spout labels | per-session encoder position dR2 (shuffle) | encoding |
| combined_encoding.py:1676 `_perm_within` (shufside / shufdist) | labels within same side / distance | side vs distance dR2 maps | encoding |
| combined_encoding.py:1690 / 1694 one-vs-rest | random trials get label p | per-spout dR2 maps | encoding |
| scripts/aligned_decoder.py:321 | y[cal] | aligned-decoder Procrustes null | state space |
| scripts/reembedding_test.py:334 (G3) | y within cal / score halves | re-embedding test | state space |
| scripts/lick_code_subspace.py:411, 446 | (side_ml, pos) jointly; ys | lick-code same / remapped floor | state space / lick |
| wfield_local/grant_similarity.py:315 | spout labels within session | grant pattern-similarity null | grant |
| scripts/fr_enl_decode.py:118; enl_fr_removed_decode.py:119; enl_movement_multisession.py:253; enl_video_spout_check.py:142 | spout labels vs fixed CV predictions (CV block-grouped) | ENL decode with face / movement / video removed (FR page 6.1) | FaceRhythm / kinematics |
| scripts/movement_position_angle.py:79 | cue position labels | movement vs position dR2 null | kinematics |
| scripts/sm_direction_lag.py:185 | executed angle, target | S vs M direction lag (preregistered) | lick direction |
| scripts/prelaunch_aim_tracking.py:583 | launch angle within session (across spouts) | aim-tracking decoder null | lick direction |
| scripts/lick_enl_angle_decode.py:301 (raw arm) | angle (partly = spout) | ENL angle decode | lick direction |
| scripts/component_exclusion_audit.py:175 | spout labels | component exclusion audit | decoders |
| scripts/cohort2_qc/spout_compare_1029.py:100 | positions | cohort2 QC spout specificity | cohort 2 |

## W. Within-spout trial-wise shuffles (lower priority)

Labels do not follow blocks, but same-spout trials are clustered in time (slow drift within a block). Block-aware form =
permute same-spout RUNS (`partial_refit._block_perm`): lick_enl_angle_decode 297 / 477, lick_enl_offset_explore 259,
prelaunch_aim_tracking 583 (cell), prelick_hemisphere_tuning 292, prelick_svd_tuning 327, lick_templates 211,
movement_position_angle 98, sm_coherence_cca 779 / 810.

## C. Circular time shifts

| site | shift | verdict |
|---|---|---|
| combined_encoding.py:1805 `shuf:<var>` (movement, licks) | >= SHIFT_MIN_S = 60 s | CANDIDATE: ~one block, so shifted movement can stay spout-aligned -> movement unique dR2 understated. Block-aware: move whole-trial segments with a permuted block order |
| frozen_encoder.py:476 landing-weight roll | 10-90 % of landed licks | probably fine |
| rest_position_decode 212, rest_position_permutation 240, rest_frozen_decoder 317 | roll k in 1..n-1 | conservative; primary rest null is blockperm |
| state_spout_check 101; partial_refit 276 / 291 | half-session roll | keeps blocks; fine |
| cohort2_qc/slowband_behaviour 96 | >= 600 s | fine |

## F. Position-decoder CV not grouped by block

cohort2_qc/spout_compare_1029.py:97 (StratifiedKFold, the COHORT2_QC decode_bal_acc); sm_direction_lag.py:59; prelaunch_
aim_tracking.py:690; within-spout: prelick_hemisphere_tuning 260, prelick_svd_tuning 473. Default-block but trial-wise
under `--cv random`: cohort1_subset_headline 63, locanmf_position_decoder 675.

## Split-half reliability ceilings (same issue, other direction)

Trial-wise split halves put block-mates in both halves -> optimistic ceilings: grant_figures, grant_geometry,
grant_similarity 553 / 605, null_potent 129, encoder_bins_test, state_space_orth_check, state_space_shift_spout_timing.

## B. Already block-aware (three duplicate helpers)

precue_significance.permute_block_labels and transfer_matrix.block_permute (identical: permute the block -> label map
over `block_ids` blocks), decode_ci.frozen_ci (inline), rest_frozen_decoder blockperm, shared_position_projection 356,
partial_refit._block_perm (permutes same-label RUNS, not `block_ids` blocks; long runs are not split, so it differs).

## D. Session / epoch / animal-level shuffles (not trial blocks)

Session labels within animal (pre vs epoch), event-pairing derangements, feature shuffles: fine with respect to trial
blocks. OPEN (raised by window 3c): association tests between two session-level series that both trend with days
(e.g. balance vs aim within animal) are not exchangeable under a plain within-animal session permutation; adjacent-day
autocorrelation makes that null optimistic too. Remedy: detrend both series (time trend removed, as several recovery-link
figures already do) or use a circular shift of one series over the animal's session order. Separate item from blocks.

## Plan (proposed, not started)

1. ONE helper `analysis_kit.block_permute(y, blocks, rng)` over `block_ids` (rule 9); the two existing module helpers
   become aliases (draws unchanged -- pin in tests); `partial_refit._block_perm` stays until its arm re-runs.
2. MEASURE first: `permutation_null` trial-wise vs block-wise on the ENL / no-lick decoder sessions (cheap: predictions
   are fixed, only the null changes) -> how many published p values cross 0.05. Report per arm before any re-render.
3. Then `evaluate_arm` gets `blocks=`; callers pass `block_ids`; re-render affected families by owner.
4. Encoding: `shuf:position` -> block permutation; movement `shuf:<var>` -> block-segment permutation (sensitivity; the
   primary drop dR2 has no null).

Owners notified 11-16: sabatinilab-89 (FaceRhythm), sabatinilab-3c (state space + pages).
