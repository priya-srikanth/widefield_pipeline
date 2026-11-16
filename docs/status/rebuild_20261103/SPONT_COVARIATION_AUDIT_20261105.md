# Spont co-variation audit: frozen variant lists and readers under v7g3_t3d_spont (11-05 evening)

Read-only. Code: main a8f1651d (the render worktree runs a76352f3; frozen_encoder is the same there for everything below).
Queue: `scratchpad/render_daqlk/queue_daqlk4.sh`. Frame lanes: `scratchpad/frozen_v7g3_spont/lane{1,2,3}*.sh`. The
variant folders on disk under `_daqlk` were listed per frame.

Priya asked for `no_spont` to be added to lsland, and: "anything else we need to adjust since lots of things co-vary?"

## A. How spont enters the frozen design

Source: `wfield_local/frozen_encoder.py`.

- **Group.**
  - `licks_pre` / `licks_iti` go into the frozen group **"other"** (`:166-169`, `:183-186`: `FGROUP[v] = "other"`, penalty
    group "lick").
  - Design comment `:158-160`: "licks_pre / licks_iti UNINSTRUCTED = the frozen group 'other' (nose, running, push: the
    encoder's uninstructed set), so `no_other` = the encoder's drop:uninstructed; `no_spont` drops the two together (=
    drop:licks_pre+licks_iti), novar_licks_pre / _iti one each".
  - Decision source: RUNNING_LIST 11-04 ~17:55 / ~20:55: "licks_pre (ENL) = UNINSTRUCTED; licks_iti = UNINSTRUCTED" and
    "It matches the combined encoder".
- **Derived variants (`:1206-1235`).** `_DROP` / `_ONLY` act on groups, `_SPONTDROP` / `_VDROP` on variable bases. So, under
  spont:

| variant | drops / keeps spont? | matches the intent? |
|---|---|---|
| `no_lick` | does NOT drop spont. "lick" = post-cue lick-onset deviation + tongue / jaw speed and opening deviations. Under FROZEN_LICKSPOUT the lick-onset kernels are group "lickspout", so `no_lick` holds only the tongue / jaw deviations. | Yes, by design (spont is uninstructed). Label "lick" everywhere = CUED lick vigour / timing only. |
| `no_other` / `other_only` | drops / keeps spont (nose, running, push + licks_pre / licks_iti) | yes (= encoder drop:uninstructed) |
| `no_movement` / `movement_only` / `task_first` / `movement_first` (SUPER movement / nontask include "other") | spont is MOVEMENT | yes (uninstructed movement); captions of "movement unique" now include spontaneous licks |
| `no_noposinfo` / `noposinfo_only` (frA_pos, frAsplit) | spont is in "noposinfo" (SUPER noposinfo = lick, other, fr) | consistent: the spont kernels carry no spout label |
| `target_first` / `exec_first` (`_ORDERED`, `_EXEC_SIDE` = exec_locked, exec_within, lickspout) | spont is on NEITHER side, so the joint ridge credits it | not stated in any design comment. Spont shares pre-cue variance with `pos_enl`, and the bracket does not order target vs spont. B12 (combined encoder) measured < .005 movement of pos_enl credit, so this is acceptable; say so in the bracket captions. |
| `no_spont` | both spont variables | in the DEFAULT variant set of every spont frame (`:1228-1235`) |
| `spont_only` | **does not exist** (`_ONLY` is group-based; no variable-set "only") | gap: Musall "alone" has no spontaneous-lick row (B6) |
| `lick_only` (Musall alone, "instructed (lick)") | excludes spont, and under LICKSPOUT also excludes the lick-onset kernels (group "lickspout") | label mismatch, pre-existing (B5) |

- **Saved PETH parts (`:1650-1660`, `:1840-1850`).** "other" already contains spont, and a separate `spont` part is ADDED as
  "the spontaneous licks' share of other". `spout`, `lick_all`, `pos` and `push` were already overlapping extras.
- **Readers of the parts.** Every reader found is name-based: `data` / `full` by index of `peth_parts`, or one column per
  part:
  - lp_encoder_decompose `:47-90`;
  - residual_roi `:50-55`, specificity_control `:52-58`, frozen_residual_trend `:45-50`;
  - frozen_area_maps `:38-45`, matched_residual `:49-50`.
  None sums all parts, so nothing double-counts `spont` + `other`. Keep it that way: summing groups must skip `spont` /
  `spout` / `lick_all` / `pos` / `push`.

## B. Required and recommended changes

### Add a variant to a frame (needs a re-render)
1. **`t_frame_lsland` (queue_daqlk4 `:323`): add `no_spont`. REQUIRED (Priya).**
   - The frame has not run yet (no `.done` / `.claim`), so this is just the arg list.
   - Its sister frame `..._spont_frA_ls` already has `no_spont` (lane3 GROUPV).
   - Without it, the ls-vs-lsland comparison cannot show how much the spont kernels take from the target / landing terms.
   - Also add `no_other`, which ls has. That makes the lsland list = ls's trimmed set + `no_spont no_other`.
   - Optional: `novar_licks_pre novar_licks_iti`.
2. **`t_frame_frlag` (`:325`): add `no_spont`. RECOMMENDED.**
   - Spont kernels and the FaceRhythm lick factor co-vary at pre-cue / ITI licks, and frlag has no spont-free comparison.
   - contest_lines does not read it, but without it the "face leads" uniques cannot be checked against the spont kernels.
   - Not yet run either.
3. **Pixmusall twins: add `no_spont no_other`. OPTIONAL; only if a Musall PETH / trial-by-trial claim about spontaneous
   licks is wanted.**
   - Twins: `..._spont_basis_pixmusall` (lane1 `V`) and `..._spont_frA_basis_pixmusall` (lane2 `V no_fr_all`). Both are
     rendered, cheap to redo with `--fit-from` / `--alphas-from`.
   - Today they hold `joint no_target no_exec_locked no_exec_within no_spout no_movement`, and frozen_claims reads only
     those. Their "movement" unique includes spont, which is consistent but must be captioned.
4. **`..._spont_frAsplit_basis` (lane3): add `no_spont no_other`. OPTIONAL.**
   - `b_pix_frAsplit` (frozen_pixel_unique_sig) silently skips the `other` / `spont` panels for this frame
     (`frozen_pixel_unique_sig.py:81` keeps only existing variant folders).
   - The noFR / frA basis frames already have them.
5. **`spont_only` (Musall "alone" of the spontaneous licks): code plus re-render, if wanted.**
   - Add a variable-set "only" next to `_SPONTDROP`: `{"spont_only": SPONT_VARS}`, keep = base in SPONT_VARS.
   - Re-render it into `..._spont_frA_ls_basis` (for `z_pixmusall_alone_*`) and `..._spont_frA` (for frozen_peth_trial_maps).
   - Today the "uninstructed (other)" alone row includes spont, with no separate row.
6. **No change needed:**
   - `..._spont`, `_basis`, `_frA`, `_frA_ls`, `_frA_ls_basis` already contain `no_spont`.
   - `_frA_ls*` also has `novar_licks_pre` / `novar_licks_iti` (on disk, checked).
   - `..._spont_frA_basis` already has `no_spont` / `no_other`.
   - `..._spont_frA_pos` needs nothing: spont sits in noposinfo by construction.

### Reader logic / queue args (no frame re-render)
1. **Self-fit remap of spont: add `no_spont` (and `no_other`) to session_vs_frozen. RECOMMENDED.**
   - `scripts/session_vs_frozen.py:37` `DEFAULT_VARIANTS = [joint, no_target, no_spout, no_task, no_movement, no_fr, no_fr_dir,
     no_lickspout]` has no `no_spont`, so the queue steps `a_svf_ls` (`:240`) and `b_svf_frA` (`:250`) compute no
     self-fit remap for spont. Both frames have the folder.
   - Fix: pass `--variants <default list> no_spont no_other` in those two steps (not run yet), and add `"no_spont": "spont",
     "no_other": "other"` to `remap_checks.py:28` QLAB.
   - Without it the remap's "movement" includes spont with no way to separate it.
2. **Stale captions and labels under spont** (rewrite; captions are replaced, not annotated):
   - `scripts/frozen_peth_trial_maps.py:32-34`: "uninstructed = 'other' (nose, running, other pose ...)". It now also holds
     the pre-cue / ITI licks.
   - `scripts/frozen_r2_maps.py:38` label `("lick", "no_lick")`: should read "cued lick vigour / timing (post-cue)", since
     spont is not in it.
   - `scripts/frozen_pixel_musall.py:43-44` "instructed (lick)" = `lick_only`. On `_frA_ls_basis` this excludes the
     lick-onset kernels (group lickspout), and spont is in "uninstructed (other)". The row is mislabelled for the ls frames
     (pre-existing, not spont-specific). Either add a lickspout-inclusive "only" or relabel it "tongue / jaw vigour
     deviations".
   - Every "movement unique" caption (frozen_fr_musall(_v2), frozen_claims, session_vs_frozen, frozen_pixel_unique_sig,
     remap_lmm): add "incl. spontaneous (pre-cue / ITI) licks".
   - Bracket captions (target_first / exec_first): "spontaneous licks are split jointly (in neither side)".
3. **Readers that would silently NaN or skip.** All of these skip a missing variant rather than fail:
   - `frozen_pixel_unique_sig.py:81` and `frozen_pixel_musall.py:117-118` (prints `missing`);
   - `frozen_r2_maps.py:43` (`OPTIONAL = {"no_spont"}`);
   - `frozen_musall_figures.py:45` (NaN for v7g3_t3d; not in the queue);
   - `remap_checks.py:35`.
   The only silent gaps under the current lists are the `other` / `spont` panels of `b_pix_frAsplit` (B4 above) and the
   spont self-fit (reader item 1). I found no wrong difference: every unique is `variant SSE - joint SSE`, read by name.
4. **No change needed in partial_refit.**
   - Spont stays frozen in every refit set (`scripts/partial_refit.py:51-56`: target / exec / lickspout sets).
   - The target-perm null permutes trial spouts only. The spont kernels carry no spout label, so the null is unaffected.
   - `:95-96` already captions the spont design.
5. **No change needed in spout_unique_maps / epoch_kernels.**
   - spout_unique_maps uses drop-one-spout on the TARGET (and spout-locked) columns; spont is untouched.
   - epoch_kernels refits the joint per epoch group with the frozen scaling (spont columns included) and maps target /
     lick@spout / cue kernels only. No spont map; add one only if wanted.

## C. te (FROZEN_DESIGN=te, FROZEN_LICKS_ALL=none): confirmed, no spont expected
- `frozen_encoder.py:1196-1197` refuses te with spont. te has its own lick families (`te_lick_parts` `:394-405`):
  - in-trial licks are target-labelled (`lick_onset@tgt:q`);
  - every lick, ITI included, is landing-labelled (execution).
- Readers:
  - `te_readout.py:26-28` uses `no_target / no_execution / no_contact / no_task / no_movement` and `target_only /
    execution_only`;
  - `t_svf_te` passes `joint no_target no_execution no_contact no_task no_movement`;
  - `partial_refit` uses TE_SETS / TE_NULLS (`:72-84`).
  None names `no_spont` or a spont part.
- Difference to note in captions: in te, ITI licks are EXECUTION; in v7g3_t3d_spont they are "other" (uninstructed).
