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
| post-stroke activity stays on the pre-stroke manifold: VAF .83-.88 vs ceilings .89-.91 (k = 14-16 PCs, joint basis). **08-23 only**: the then-registered sessions (acute days only, no subacute / chronic), pre incl. June, the old signal variant. Never re-run on the full set; no figure on any page or deck. | `scripts/archive/axis_manifold.py`, DECISIONS ~l.4813 | within-manifold, acute only |
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

## The path question (Priya 11-07): HOW the acute / subacute brain finds the new solution

Scaling vs rotation vs local search vs other, read from the sequence of sessions, per animal, against days:

- **P1. Decompose each session's change from pre** into four parts, then ask which part dominates in each epoch and
  whether the mix changes from acute to subacute:
  1. amplitude along the pre pattern;
  2. per-area gain profile;
  3. residual rotation within the pre space;
  4. variance outside the pre space.
- **P2. Session-to-session geometry.** Distance matrix between all sessions of an animal, on the amplitude-normalized
  pattern AND on raw amplitude; embed against days. The hypotheses predict different paths:
  - scaling: a monotone 1-D path;
  - rotation: a steady turn in one plane;
  - local search / exploration: a meandering subacute path, with long path length relative to net displacement and
    large day-to-day jumps, then settling.
  - Does chronic return to the pre cluster, or settle somewhere new, compared with the pre-to-pre spread?
- **P3. Variability as the search signature.** Day-to-day distance between consecutive sessions, and trial-to-trial
  spread (trial-count matched). Expected under exploration: both up in the subacute window, then contracting
  (Athalye 2017, Kim 2023).
- **P3b. Trial clouds over days (Priya 11-07: trial-level variability for the path question).** Project every working
  trial (contact + incomplete) of every session into ONE fixed per-animal pre space. Each day is then a cloud, and the
  sequence of clouds is the path. Per day, per spout and pooled:
  - the cloud's center (the path, with per-day uncertainty from the trials, which helps with only 1-7 sessions per
    epoch);
  - its spread (within-day dispersion vs the pre within-day dispersion, trial-count matched);
  - its shape: which directions the spread lies in. Is the spread along the direction the mean later moves (search
    aimed at the solution), or isotropic?
  - **Repertoire before mean** (Golub 2018 / Oby 2019 analog): do chronic-like patterns appear in the TAILS of
    subacute trial clouds before the session mean gets there? Score each trial's distance to the animal's chronic
    template vs its pre template, and track the share of "chronic-like" trials by day against the mean's position.
    Under local search the share rises before the mean moves; under smooth scaling the whole cloud slides together.
  - Split contact vs incomplete trials where counts allow (acute far contra): do incomplete trials sit further from
    pre, or in the same cloud?

  Data: the per-trial area extracts (`first_cue` / `first_peth`, 63 areas incl. ALM / tjM1; first-lick trials only)
  and the encoder `_trials.npz` `trial_parts` (attempted trials, window-averaged). Per-trial widefield noise is large:
  every spread number is relative to the pre within-day spread at matched counts.
- **P4. Lead / lag against behavior.** Does the neural path move before, with, or after daily far-contra hit rate and
  aim? Neural settling before behavior suggests the network finds the solution first; settling after suggests the
  network follows behavior.

Limits:
- Sparse days: acute has 4-6 sessions per animal and PS95 has 1; subacute has 2-7.
- Behavior changes on the same days, so the movement confound applies to every step. Run per-spout and on close
  spouts, where behavior is near normal, as the control.
- No-stroke imaging drift sets the floor: the pre-to-pre spread over the August span.

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

**CORRECTION (Priya 11-07: "activity is increased across all regions, how does gain go down?").** `gain_global` is a
regression slope = r x (amplitude ratio), NOT amplitude. Split, cue-aligned, per animal:

| | amplitude vs pre template | r with the pre pattern |
|---|---|---|
| pre (floor) | 1.06-1.09 | .90-.93 |
| acute | 1.04-1.49 | .58-.81 |
| subacute / chronic | 1.20-1.77 | .71-.91 |

So activity is as large or LARGER acutely; it is less like the pre pattern; the slope fell because r fell. Constant
per-area offsets (`level_shift`) are ~0, so the "shift" rung is negligible. The earlier claim "orientation shifts
acutely" is not established: the overlap drop shares the template-vs-single-session bias, and per-area gains alone
can reorient a PC subspace.

Tentative reading: within-manifold, gain + shift, no rotation beyond what gain explains. This matches the earlier lick-code
and CD-trajectory results.

**Known flaw to fix before any claim:** the template averages 7-8 sessions while a test session is one, so norm ratios
and overlaps are biased even for pre (the pre "specific_norm_ratio" floor is 1.14-1.33). Next version: session-vs-session
comparisons, trial-count matched, with the pre-vs-pre distribution as the null.

## Inputs are FINAL (34, 11-07)

The render rewrites neither the encoder PETHs (frA joint/sessions, written 11-05 02:14-04:37; no queued step runs
frozen_encoder on that tag) nor the per-trial extracts (`sessions_roi8fr`, step e_iti_main, done 11-05; downstream
steps done, not re-queued). Analyses here can run now.

## Method check: planted changes in pre-stroke sessions (11-07, `scripts/state_space_sim.py`)

- Each pre session in turn gets a known change.
- It is scored against every other pre session of its animal, single session vs single session, trial-matched.
- Unchanged sessions = the null (31 sessions; false-positive rate 6 % by construction).
- Detection = share of planted sessions outside the null's 95 % range.
- Tables: `Widefield/state_space_recovery/sim/sim_detection_n{0,15}.csv`; n0 = all trials, n15 = 15 per spout,
  acute-like.

| planted change | best readout | detected (all trials / 15 per spout) | confused with |
|---|---|---|---|
| rotation in the top-2 plane, 10 / 20 / 30° | rot_gain (ladder L4 - L3) | .48 / .84 / 1.00 ; .39 / .61 / .84 | none; r also drops |
| rotation in a random plane of the top-8, 20 / 30 / 45° | rot_gain | .32 / .42 / .74 ; .19 / .32 / .61 | - |
| per-area gain, random, log SD .15 / .30 | out_var | .94 / 1.00 | **looks "outside the pre space"** (out_resid .48 / .94) |
| ipsilesional hemisphere x .8 / x .6 | out_var | .26 / 1.00 | **looks "outside"** (out_resid .19 / .81); gain_sd misses it (.00 / .16) |
| global amplitude x 1.25 / x 1.5 | amp | .61 / .74 | inflates raw spread (.65 / .97) and shared norm; spread_norm is clean (.06) |
| shared (condition-independent) shift, .25 / .5 of its norm | shared_dist | .13 / .39 | weak: needs a better readout |
| new spout-coding dimension outside the pre space, 5 / 15 % of variance | out_var, out_resid | 1.00 / 1.00 | gain_sd also flags it (.87 / .97) |
| trial spread (local search), + .5 / + 1 pre trial SD | spread | .45 / .90 ; .19 / .71 | some leak into rot_gain (.48 at +1 SD) |

What this means for the analysis:

1. **A rotation can be claimed or excluded only for >= ~20° in the dominant plane** (>= ~30° at acute trial
   counts). Per-area gains do NOT read as rotation in the ladder (rot_gain stays near the null). The first pass's
   "no rotation beyond gain" therefore means "none >= ~20-30° in the main plane", not "none".
2. **"Outside the pre space" is NOT specific.** Region gains, especially a hemisphere-wide one, push variance
   outside the top-8 space as strongly as genuinely new dimensions do, even after a per-area gain correction
   fitted session vs session (noisy gains). A new-pattern claim needs the gain explanation excluded directly. Next:
   - a hemisphere-contrast gain readout (mean log gain ipsi - contra), because gain_sd misses a uniform hemisphere
     gain;
   - out_resid with gains fitted against the multi-session template rather than one noisy session.
3. **A condition-independent shift is barely detectable** with a norm-ratio or distance readout. Next: a per-time-bin
   shared-trajectory difference in units of the trial noise (Sun 2022 style).
4. **Trial spread:** use the amplitude-normalized version (spread_norm), because a global amplitude change inflates
   raw spread. It detects + 1 SD at .65 (all trials).
5. **Acute trial counts (15 per spout) cost about 1/4 to 1/3 of the power** for rotation and spread. Acute nulls
   are reported at matched counts.

## Single-trial run, all 107 sessions (11-07, EXPLORATORY, `scripts/state_space_recovery_trials.py`)

- Session vs each pre session, within spout, matched trial counts.
- First-lick trials (contact + incomplete).
- Path distances noise-corrected.
- Outputs: `Widefield/state_space_recovery/trials/` (sessions, pairs, drift, change_vs_pre, figure).
- Change vs pre = `boot_delta`, animal-weighted; "k/4" = animals above their own pre mean.

| readout | acute | subacute | chronic |
|---|---|---|---|
| pattern r with pre | -.20 (0/4 up) | -.08 (0/4) | -.10 (0/4) |
| amplitude ratio | +.12 (3/4) | +.31 (4/4) | +.38 (4/4) |
| rotation beyond gain (rot_gain) | **+.032 (4/4, CI > 0)** | +.002 | -.006 |
| variance outside the pre space, beyond per-area gains | +.011 (4/4) | +.009 (4/4) | +.006 (4/4) |
| shared (condition-independent) shift, noise units | +7.5 (3/4) | +16 (4/4) | +28 (4/4) |
| trial spread, raw / amplitude-normalized | +.43 (3/4) / +.05 | +.13 / -.38 (0/4) | +.13 (4/4) / -.46 (0/4) |
| balance log2 contra / ipsi, orofacial far ring (FINDINGS 0b measure) | +.21 (3/4; PS93 the other way) | +.21 (4/4) | +.26 (4/4) |
| position on the pre (0) -> chronic (1) line, alpha | -.13 (1/4) | +.59 (4/4) | +.97 (by construction) |
| share of chronic-like trials | +.02 | +.30 (4/4) | +.43 (4/4) |
| spread along pre -> chronic, relative | +1.0 | **-4.9 (0/4)** | +.2 |
| distance from the pre sessions (pre-pre median units) | +.65 (4/4) | +.62 (4/4) | +.87 (4/4) |
| day-to-day step | +.79 (4/4) | +.72 (4/4) | +.24 (3/4, CI spans 0) |

**Drift** (noise-corrected distance vs day gap):

- within pre: +.04 to +.18 per day, gaps <= 8 days;
- within chronic: +.00 to +.03 per day, over 25-39 days.

The chronic state is STABLE (no ongoing drift), but chronic vs pre still cannot be separated from time since pre with
these data (the earlier June -> August no-stroke drift problem).

**Corrections recorded:**

- `hemi_gain` (per-area regression slope, ipsi - contra) is a PATTERN measure, not response size. Its PS93 acute
  outlier (ipsi x2-9 on days 1-3) is real but one animal.
- The size-based balance reproduces FINDINGS 0b: contra / ipsi rises post-stroke (Priya 11-07).

**Tentative path, for discussion, not a claim:**

1. **Acute** = an excursion: the pattern decorrelates, some rotation beyond gain, a little activity outside the pre
   space, more trial spread, large day-to-day jumps. It does NOT point toward the eventual chronic state
   (alpha ~0 or negative).
2. **Subacute** = moving toward the new state: alpha .59, a third of trials already chronic-like. The trial spread is
   NOT aimed along the pre -> chronic line (aligned 0/4), which argues against a directed search.
3. **Chronic** = a stable NEW state, not a return to pre: larger (x1.4), pattern r still -.10, shared trajectory
   shifted, contra / ipsi balance up, day-to-day steps back near pre.

In BCI terms: not pure scaling. A disrupted acute state, then consolidation into a new mostly-within-manifold
solution with gain and a shared shift.

**Caveats before any of this is used:**

- Behavior differs on the same trials. Like-for-like executed landing is NOT yet matched (Priya 11-07); the per-trial
  extract has no trial id, so it must be joined via `iti_licks._first_lick_times`.
- The acute rot_gain may partly be trial spread (the sim showed leakage at +1 SD).
- Raw spread vs amplitude-normalized spread disagree whenever amplitude changes. Decide which noise model holds
  (does the signal grow, or signal and noise together?).
- alpha / chronic_like use a chronic template, so chronic values are circular. The "repertoire before mean" test
  (is the chronic-like share higher than a sliding cloud at that alpha would give?) is not done yet.
- First-lick trials only.
- PS95 acute = 1 session.

## Follow-ups 11-07 (exploratory): like-for-like landings, drift direction, repertoire before mean

**Landing join** (`scripts/state_space_landing.py` -> `Widefield/state_space_recovery/landing/first_lick_landing.csv`):

- Extract rows are paired with the 3-D first lick (licks.csv.gz, lick_idx 0) by spout + timing: 3-D peak - onset
  = 39 +- 18 ms. This avoids the 65 s/session behavior reload.
- Validated: 99 % exact trial agreement on PS94_0820; 92-98 % of rows matched in every epoch.
- Landing bin = nearest pre landing center per animal (Mahalanobis).
- Acute far-contra first licks land on target in 0-3 % (mostly the far-ipsi bin). Chronic: PS92 / PS95 .72-.74,
  PS93 .22, PS94 .08.

**Like-for-like** (`--condition on_target | landing`; outputs `trials_on_target/`, `trials_landing/`). Change vs pre,
k/4 animals up, * = boot_delta CI excludes 0:

| readout | target (all trials) | on-target trials only | grouped by landing |
|---|---|---|---|
| pattern r, acute / chronic | -.20* / -.10* | -.16* / -.11* | -.17* / -.12* |
| amplitude, chronic | +.38* | +.39* | +.32* |
| **rotation beyond gain, acute** | **+.032 (4/4*)** | **+.011 (2/4)** | **+.012 (3/4)** |
| outside the pre space beyond gain, acute / chronic | +.011* / +.006* | +.013* / +.006* | +.013* / +.006* |
| shared shift, chronic | +28* | +19* | +21* |
| raw trial spread, acute / chronic | +.43* / +.13* | +.38 / +.22* | +.44* / +.23* |
| **balance contra / ipsi, far ring, acute / chronic** | **+.21* / +.26*** | **+.33* / +.27*** | **+.02 / +.11*** |
| alpha subacute; chronic-like share subacute | .59* ; +.30* | .65* ; +.34* | .65* ; +.29* |
| distance from pre, chronic; day-to-day step, acute | +.87* ; +.79* | +.84* ; +.94* | +.89* ; +.73* |

Reading:

- Almost everything holds on like-for-like movements, so it is NOT the changed movement. This covers the
  decorrelated pattern, larger amplitude, the small out-of-space part, the shared shift, the path, and the
  day-to-day jumps.
- **Two exceptions:**
  - The acute "rotation beyond gain" does NOT survive. It was the changed movement / landing composition, so drop it.
  - ~~The hemispheric balance follows the TARGET, not the executed landing~~ **RETRACTED 11-07 (see "Balance at
    matched movement" below)**: the on-target / landing ring averages lack far contra acutely (too few trials), the
    spout with the most negative pre balance, so the +.33 / ~0 columns were composition artifacts.

**(2) Drift direction** (`scripts/state_space_path_tests.py` -> `path_tests/drift_direction.csv`). The pre-stroke
drift direction (slope over the 8 August days; split-half cosine .83 / .86 / .64 for PS92 / 93 / 94, .38 for PS95):

- Cosine with chronic - pre: +.59 / +.48 / +.63 / -.18.
  - 35 / 23 / 40 / 3 % of the chronic change lies along it.
  - At the pre rate, the whole chronic offset is only ~8-11 days of drift.
- Cosine with acute - pre: -.44 / -.36 / -.42 / +.28. The acute change points AGAINST the drift.

So in 3 / 4 animals a quarter to 40 % of the chronic "new state" lies along the direction cortex was already drifting
before the stroke. The drift then SLOWS (corrected 11-07 from "stops"): within-chronic slope +.026 / +.001 / +.006 /
+.020 per day vs pre +.182 / +.116 / +.098 / +.043 (PS92-95), i.e. flat in PS93 / PS94 only; PS92 / PS95 still drift
slowly, in line with FINDINGS 8 (crossnobis distance still rising at drift rate). Whether chronic is "where practice
would have gone anyway" cannot be decided without a no-stroke control. Cohort 2's pre period may say whether pre drift
saturates (cohort 2 is analyzed separately, never pooled).

**(3) Repertoire before mean** (`path_tests/repertoire_before_mean.csv`): observed share of chronic-like trials minus
the share for a pre cloud slid to the session's own position on the line, with matched spread:

- acute +.03 / +.01 / +.01 / -.01;
- subacute +.06 / +.01 / -.02 / .00;
- 10/16 acute and 8/17 subacute sessions > 0.

No sign that chronic-like patterns appear in the trial tails before the mean. The cloud SLIDES rather than searching
and selecting (no Golub / Oby-type repertoire signature at this resolution).

## Balance at matched movement (11-07, Priya: "use the angle / lateral offset to use all spouts")

Scripts: `state_space_balance_cells.py` (target x landing-bin cells, per-session target balance) and
`state_space_balance_angle.py` (every first lick; per animal x ring a PRE-only fit of each hemisphere's per-trial
orofacial response size = target intercept + shared slope x 3-D `side_ml`; post balance minus the fit's prediction for
the same trials). Outputs `landing/balance_cells*.csv`, `trial_sizes.csv`, `balance_angle_{sessions,coefs}.csv`.

| spout | by target, acute / sub / chronic | matched target + landing, acute / sub / chronic |
|---|---|---|
| close ipsi | -.30* / -.15 / -.14* | -.24* (0/4) / -.11 / -.11 (0/4) |
| close mid | +.03 / +.16* / +.17* | .00 / +.12 / +.14* (4/4) |
| close contra | +.19 / +.24 / +.25* | +.13 / +.20 / +.24* (4/4) |
| far ipsi | -.18* / +.12 / +.18 | -.39* (0/4) / -.05 / +.06 |
| far mid | +.11 / +.09* / +.20* | -.12 / -.04 / +.12* (4/4) |
| far contra | +.60* (4/4) / +.40* / +.40* | +.33 (2/4; PS94 / PS95 +.8, PS92 / PS93 < 0) / +.27* (4/4) / +.35* (4/4) |

Reading: subacute / chronic the balance moves contralesional at contra-side and middle targets at matched movement
(not the movement); ipsi-side targets do not share it (acutely they move the other way). The across-spout range of the
on-target balance (5 spouts, far contra excluded in every epoch) compresses: -.47 (0/3) / -.32 / -.27 (0/4), CIs < 0 --
consistent with FINDINGS 0b "post-stroke ~30 % flatter". About half the acute far-contra rise is the landing.
Same-landing / different-target cells: acute far-contra-aimed licks landing far ipsi sit between the pre far-contra and
far-ipsi values, as pre off-target licks did; neither target nor landing alone sets the balance. Caveat: cue-aligned
0-1 s, SSp-m / SSp-n / SSs; FINDINGS 0b's primary marker is first-lick-aligned 0.1-0.5 s with ALM / tjM1 -- not yet
checked in that window. PS94 acute far contra: only 37 % of trials inside the pre landing range.

## Alignment: cue vs first lick, and 2-D trajectories (11-07 evening)

Priya: the cue window (-0.5..2.0 s) misses late acute responses (acute far-contra first lick median 0.78 s, 90th pct
2.4 s). `state_space_sim.load` now takes `WF_SS_ALIGN` = cue (default, unchanged; pinned equal to the old binning) /
cue25 (-0.5..2.5 s) / lick (`first_peth`, -1.0..0.7 s from the first lick) / splice (cue -0.5..0.5 + lick -0.5..0.7 s).
Binned arrays cached per alignment at `state_space_recovery/cache/<align>/` (one share read for all). Outputs carry the
suffix: `trials_lick`, `trials_on_target_lick`, `trials_landing_lick`, `path_tests_lick`, `trials_splice`, ...

| readout, change vs pre | acute cue / lick | subacute cue / lick | chronic cue / lick |
|---|---|---|---|
| pattern r | -.20* / -.08* | -.08* / -.04* | -.10* / -.04* (0/4 back) |
| amplitude | +.12 / +.22* (4/4) | +.31* / +.30* | +.38* / +.30* |
| rotation beyond gain | +.032* / +.014* (gone like-for-like in both) | ~0 | ~0 |
| outside pre space, beyond gains | +.011* / +.006* | +.009* / +.005* | +.006* / +.003* |
| shared shift | +7.5* / +11.6* | +16* / +18* | +28* / +29* |
| alpha | -.13 / +.07 | +.59* / +.68* | ~1 |
| far-ring balance (lick: 0.1-0.5 s after the lick) | +.21* / +.31* | +.21* / +.29* | +.26* / +.29* |

Reading: much of the acute "re-patterning" (r) was latency; the acute amplitude increase is clearer lick-aligned. Splice
agrees with lick. Lick-aligned drift: pre +.177 / +.059 / +.096 / +.005 per day, chronic +.044 / -.002 / +.004 / +.015;
drift-direction cosine with chronic - pre +.41 / +.41 / +.37 / -.06 (share 14-17 %). Repertoire excess ~0 (11/16 acute,
7/17 subacute > 0).

2-D trajectories (`scripts/state_space_trajectories.py` -> `trajectories{tag}/`, `page_figs/ss_traj_*`):
- ACROSS DAYS, plane x = pre -> chronic, y = acute departure orthogonal to it (same units): 80-95 % of acute - pre is
  orthogonal to x in every animal and alignment. Lick: acute (x, y) (-.16, .56) / (.18, .62) / (-.18, .97) / (.49, 1.12),
  subacute (.65, .37) / (.42, .29) / (.80, .05) / (.90, .20). A DETOUR: out sideways acutely, back toward the axis while
  moving along it subacutely. Axes built from epoch templates (x circular for chronic, y for acute); subacute positions
  and trial clouds are not circular.
- WITHIN THE TRIAL, cohort top-2 PCs of the animals' pre templates (93-96 % of pre variance; mostly the shared response,
  so spout-specific rotations are not visible there): post-stroke loops larger and offset before the lick, clearest at
  close spouts.

## Decisions 11-07 (Priya) and the checks behind them

- **Trial spread is read RAW** (noise checks, `scripts/state_space_noise.py` -> `noise{,_lick}/`). Chronic, log2 vs own
  pre (cue / lick): response size^2 +.95 / +.84 (4/4), baseline trial noise +.25 / +.26 (4/4), response-window noise
  +.16 / +.09 (n.s.), noise beyond the pre mean-variance relation (var = a + b mean^2 per animal x area) +.17 / +.29;
  acute excess +.52 (4/4 cue), subacute +.40 (4/4 lick); PS92 the exception. Noise does not scale with the signal, so
  spread_norm (/ amp^2) over-corrects; its "spread falls" is retired.
- **Lick classes** (Priya: "contact + incomplete as primary, contact-only as robustness, incomplete-only as descriptive,
  and direct within-session contrast"). `state_space_recovery_trials.py --condition contact | incomplete` (contact flag
  = 3-D first lick in the landing join); contrast `scripts/state_space_contact.py` -> `contact{,_lick}/`.
  - Contact share of first licks: far contra .80 / .05 / .44 / .71 (pre / acute / subacute / chronic); other spouts
    .86-.95 every epoch.
  - Contact-only reproduces the primary result; acute amplitude larger (+.31 cue / +.40 lick, 4/4); acute rot_gain n.s.
  - Incomplete-only: no session-pair readouts (pre sessions have < 5 incomplete licks per spout in most spouts);
    template readouts: chronic alpha .64 / .72 vs ~.95, chronic-like share +.22 vs +.40.
  - Within session x spout (>= 5 of each; 75 pre / 28 acute / 32 subacute / 78 chronic, nearly all far spouts),
    incomplete - contact: size (lick) pre -.31 n.s., acute -.64*, subacute -.22, chronic -.31; r with pre template
    pre -.04*, acute -.09*; position on the pre -> chronic line chronic -.30 lick / -.51* cue. The chronic shift is
    carried mostly by contact licks.
- **Close vs far in the 2-D loops** (Priya: "unexpected that the change is so much more dramatic for close than far"):
  full-space lick response size (`trajectories*/resp_size.csv`) grows at every spout chronically, log2 +.41-.56 close
  (4/4) and +.27-.47 far (3/4; PS94 far ~no change). The close > far look is mostly the top-2 projection: at close spouts
  post-stroke trials gain a PC2 transient 0.1-0.3 s after the lick that pre close trials lacked and far trials had.
- **Error on the 2-D plots**: days plane = epoch means with 95 % bootstrap ellipses (animals -> sessions); loops =
  continuous 1-SE band (bootstrap covariance at every bin, interpolated, shapely union); loops vs time with 95 % bands.
- Cue vs lick traces by epoch already exist (goal page 14.7-14.22, FINDINGS 0b): not rebuilt here.

## Results page (draft, 11-07)

https://claude.ai/artifact/9sGELLFEwADEt9k64fXTxe (v5: cue + first-lick alignments, 2-D trajectories with error, noise checks, contact / incomplete, zoom viewer). Figures: `scripts/state_space_page_figs.py` ->
`Widefield/state_space_recovery/page_figs/`; page source backed up at `Widefield/state_space_recovery/page/index.html`.
Per-trial alpha for the trial-cloud figure: `state_space_path_tests.py --trial-alpha` -> `path_tests/trial_alpha.csv`.

## Rules

- Per-animal table first (CLAUDE.md rule 8); a claim needs >= 3 / 4 animals.
- Stats: `analysis_kit.boot_delta` + `epoch_change_lmm`.
- August-only pre; PS95 acute = 1 session.
- Hemodynamic blur: don't read sub-~200 ms timing.
- Composition: acute far-contra PETHs average attempted trials, mostly failed ones. Analysis 5 is the control.
