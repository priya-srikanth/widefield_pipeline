# State-space view of stroke recovery: plan (2026-11-07)

Branch `analysis/state-space-recovery-20261107`, worktree `wf_statespace`. Literature:
[STATE_SPACE_RECOVERY_LITERATURE.md](STATE_SPACE_RECOVERY_LITERATURE.md). Cohort 1 only (PS92-95), never pooled with
cohort 2.

## Question (Priya 11-07)

As in BCI perturbation work: after the VLS stroke, is cortical activity in state space the pre-stroke activity,
**scaled, rotated, shifted, reassociated**, or **new**? How does it move through acute and subacute to chronic?

The lesion is striatal, so the imaged cortex is intact (Priya 11-07). A region-specific gain change is therefore a
biological candidate (lost basal ganglia -> thalamus -> cortex drive), not an imaging artifact.

## What earlier analyses already say, in this vocabulary

| earlier result | where | reads as |
|---|---|---|
| post-stroke activity stays on the pre-stroke manifold: VAF .83-.88 vs ceilings .89-.91 (k = 14-16 PCs, joint basis) | `scripts/archive/axis_manifold.py`, DECISIONS ~l.4813 | within-manifold |
| position transfer matrix: symmetric "partial rotation", keeps ~.8 of the pre ceiling; blind to changes shared by all six positions | `epoch_15g` / `15h`, `transfer_matrix.py`, STATUS_2026-09-17 | some within-manifold remapping |
| condition-independent mode grows post-stroke (magnitude ratio > 1 in 12/12 cells, mean 1.41; overlap .61-.68); only PS94 chronic in the "reorganization corner" | `cd_trajectories.py`, `cd_overlap_null.py`, STATUS_2026-09-24_CD_TRAJECTORIES | uniform shift / gain of the shared mode (Sun 2022 analog) |
| lick-code subspace: no rotation beyond ~10-17° (detection limit); contralesional offset + gain loss | `scripts/lick_code_subspace.py`, `lick_code_noisematch.py`, FINDINGS ~l.671 | gain + offset, not rotation |
| CCA / Procrustes re-embedding: uninformative (failed its pre->pre gate, .29-.45) | `scripts/reembedding_test.py`, FINDINGS #20 | method failed, not a null |
| frozen decoder: far contra .90 -> .32 acutely; chronic gap within June->August drift | FINDINGS §1-3 | readout loss; chronic reorganization not established |

**Working picture to test:** within-manifold change, made of position-graded gain loss plus a growing shared
(condition-independent) shift, with little rotation. These came from five different analyses on different data
products, gates and bases. The new work puts them in ONE frame, on the current rebuild data, with one noise floor,
and adds what is missing:

- the gain-vs-rotation model ladder;
- reassociation;
- single-trial spread;
- intention vs execution;
- the trajectory over days.

## Data (current rebuild, `M:/MICROSCOPE/Priya/Widefield/rebuild_20261103/`)

- **A. Trial averages** = encoder per-position PETHs (`_daqlk` run, gate "attempted", 107 sessions):
  `.../imaging_encoding/hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_pre0801_sel-camera-cue_stop-endlick-cls-contact-incomplete-stopA2jexrt3500c70_daqlk/frozen_encoder_gate-attempted_v7g3_t3d_spont_frA/joint/sessions/<label>.npz`
  - `peth[0]` = data, shape (6 spouts x 59 areas x 187 t), -3..+3 s, cue-aligned;
  - `peth_lick[0]` = the same, first-lick-aligned.
  - Areas: Allen v8 split basis, both hemispheres, incl. ALM / tjM1.
- **B. Single trials:** `.../imaging_encoding/iti_licks/hv-meegkit_hpfit_lag_refitT_swapmask_lnmfproj_pre0801/sessions_roi8fr/<label>.npz`
  - `first_cue` = (trials x 71 lags, -1..2.5 s x 63 areas); `first_peth` = lick-aligned; labels `first_pos`, `epoch`.
  - First-lick trials only (no misses).
- **C. Cross-check basis:** joint LocaNMF (`joint_locanmf.load(animal)`, pinned per animal). Its fit includes June
  pre, so its pre is not August-only. Shared SVD via `joint_basis` if needed.

## Analyses, in order (all light; run locally)

1. **Pre manifold and noise floor.**
   - Per animal: PCA on the August pre sessions' condition x time matrices (areas as dimensions; k by
     cross-validation).
   - Score per session: the share of variance captured by the pre top-k, and principal angles to the pre subspace.
   - Floor: leave-one-pre-session-out, plus a trial-count-matched pre split.
   - Plot over days and by epoch: open markers per session (shape = animal, color = epoch).
2. **Model ladder.** Fit each session's X_post from the animal's pre template X_pre:
   1. identity;
   2. global gain;
   3. per-area gain;
   4. per-area gain + per-area offset (uniform shift);
   5. full Procrustes rotation + scale (in the pre top-k space).

   Report held-out variance (leave one spout out, and odd/even trials). Main readouts:
   - does the step from rung 4 to rung 5 add anything beyond the pre-vs-pre floor (rotation);
   - which areas carry the gains (ipsilesional ALM / tjM1 / MOp / SSp-mouth);
   - whether gains return toward 1 with behavior.

   Normalization: raw dF/F with pre-fixed scaling for gain questions; z-scored per area for geometry questions.
3. **Shared vs position-specific parts.**
   - Split X into the mean over spouts (shared trajectory) and the deviations (position code).
   - Ladder + angles on each part separately (covers the blind spot of 15g / 15h; links to the CD-trajectory shared
     mode).
4. **Reassociation / re-aiming.**
   - 6 x 6 cross-validated similarity: post spout i vs pre spout j, in the pre space.
   - Prediction: acute far contra matches pre far mid / far ipsi, returning to the diagonal with recovery.
   - Compare with kinematics Figs 4.12-4.19 (licks land like far mid).
5. **Single trials** (data B, projected into the pre space):
   - distance to the own-position template and to the others;
   - spread within and outside the pre space, at matched trial counts (exploration then consolidation);
   - intention vs execution: does a post far-contra trial sit at the pre far-contra template, or at the template of
     where its lick landed (3-D landing per trial)?
   - divergence time: pre-cue vs cue vs lick.
6. **Basis cross-check.** Repeat 1-4 in the joint LocaNMF basis. A claim stands only if it holds in both bases.

## Scope decisions (Priya 11-07)

- **Intention vs execution is already answered; do NOT redo it (item 5, third bullet).**
  - Frozen and self-fit encoder contest: FINDINGS synthesis 4. Acutely, activity follows the changed movement under
    the pre-stroke mapping. Chronically, target AND executed coding are remapped.
  - Landing / trajectory softmax decoding: ENCODING_FINDINGS 8.1-8.2. The decoder follows landing on average; trial
    coupling breaks acutely.
  - Misdirected licks: ENCODING_FINDINGS 2.5, no plan signature.
  - Failed attempts on the frozen decoder: 2.8, "distance kept, side lost"; no pre-cue plan signature.
  - CD trajectories: STATUS_2026-09-24.

  The state-space framing can cite these as the reassociation evidence. The single-trial work keeps only spread /
  consolidation and divergence time.
- **Working = contact + incomplete** (the encoder's "attempted" gate). Whether and when to split them is still open.
- **Nothing that is being re-rendered gets re-done here.** cd_trajectories, the transfer matrix and the encoder
  outputs are in the 11-06/07 render queues. Wait for the final render, then run on its outputs.

## First pass (11-07, PRELIMINARY: inputs may change with the render; not for pages)

`scripts/state_space_recovery.py` on the `_daqlk` encoder PETHs (107 sessions). Output:
`Widefield/state_space_recovery/state_space_sessions_{cue,lick}.csv`.

| reading (cue-aligned, all 4 animals) | pre floor | acute | chronic |
|---|---|---|---|
| variance inside the pre top-8 subspace | .98-.99 | .97-.98 | .97-.99 |
| overlap with the pre top-8 subspace (mean cos² of principal angles) | .83-.85 | .67-.76 | .74-.79 |

The ladder:

- Identity fit (L0) drops acutely (e.g. PS93 .67 -> .21) and stays below pre chronically in 3 / 4.
- Per-area gain + shift (L3) recovers part of it.
- Rotation (L4) adds nothing over L3 in any animal; the free map (L5) overfits.
- Global gain is below 1 acutely (cue .65-.88, 4 / 4) and above 1 chronically in PS92 / PS93 (1.3).
- The shared-trajectory norm grows (chronic 1.2-1.8 x).

Tentative reading: within-manifold, gain + shift, no rotation beyond what gain explains. This matches the earlier lick-code
and CD-trajectory results.

**Known flaw to fix before any claim:** the template averages 7-8 sessions while a test session is one, so norm ratios
and overlaps are biased even for pre (the pre "specific_norm_ratio" floor is 1.14-1.33). Next version: session-vs-session
comparisons, trial-count matched, with the pre-vs-pre distribution as the null.

## Rules

- Per-animal table first (CLAUDE.md rule 8); a claim needs >= 3 / 4 animals.
- Stats: `analysis_kit.boot_delta` + `epoch_change_lmm`.
- August-only pre; PS95 acute = 1 session.
- Hemodynamic blur: don't read sub-~200 ms timing.
- Composition: acute far-contra PETHs average attempted trials, mostly failed ones. Analysis 5 is the control.
