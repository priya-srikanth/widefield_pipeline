# Re-embedding test (lit item 2.3): pre-registration, 2026-10-30

**Written before any post-stroke number of this analysis was computed.** Question (RECOVERY_MECHANISMS_LIT_20261030
item 2.3 / section 2 item 3): is the subacute / chronic cortical spout code the OLD pre-stroke code RE-EMBEDDED in a
different spatial pattern of area activity (Gallego et al. 2020 Nat Neurosci; Degenhart et al. 2020 Nat Biomed Eng;
Makin & Krakauer 2023 eLife), or a NEW code? Context: FINDINGS 0 items 3-4 / 4b(d) -- the frozen pre-stroke readouts lose
ground chronically while per-session refits read the target, i.e. the code is "remapped"; this test asks what kind of
remapping. Script: `scripts/reembedding_test.py`. Outputs only under
`M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding/reembedding/`.

## 1. Data (fixed)

- **Sessions**: the ITI / cued-lick extraction's August+ sessions (`iti_licks.labels()`, 107 = the frozen encoder
  run), intersected with `analysis_kit.curated_labels()`; PS92-PS95 only (cohort 2 never). Epoch from the session file
  (`wfield_local.epochs`): pre (AUGUST-ONLY), acute, subacute, chronic. **June reference** (`iti_licks/june_ref/`, 6/6-6/8,
  ~60 days before the August pre, no lesion in between, never trained on) is scored exactly like a post epoch = the
  no-stroke drift reference at a lag comparable to chronic (June includes learning -> conservative).
- **Licks (PRIMARY)**: the first cued lick of each engaged trial (`first_peth` / `first_cue` rows of the session file,
  onset-aligned; times from `iti_licks/cued_first_lick_times.csv`, rebuilt with `iti_licks._first_lick_times` when
  absent). Kinematics (3-D `side_ml`, `reach_mm`) from the nearest cam4 onset within 30 ms (`lick_vigor_perspout._kin`
  rule).
- **Features**: the 45 split-basis Allen-area features (`iti_licks.area_features`: both hemispheres, ALM / tjM1 +
  MOs_rest / MOp_rest), each divided by its session frame SD (`iti_licks.area_sd`); NO per-trial baseline (decoder
  convention). Centered per session by the mean over the calibration-fold samples (section 3).
- **Target (PRIMARY)**: spout position, 6 classes, canonical order close_ipsi, close_mid, close_contra, far_ipsi,
  far_mid, far_contra. Metric = balanced accuracy over the spouts present (>= 3 licks) in the scored half; reported as
  above chance (minus 1 / n spouts present). **Secondary target**: executed lateral landing `side_ml` (continuous; ridge,
  alpha 10 fixed; Pearson r).
- **Behavior matching (PRIMARY)**: a post lick is kept only if its side_ml AND its reach lie inside the animal's PRE
  2.5-97.5 percentile for that spout (the `lick_balance_matched_error` / `lick_code_subspace.add_match` rule, extended
  to reach). Pre licks: all with kinematics. Licks without kinematics drop from the matched set. **Sensitivity**: all
  licks, no matching.

## 2. Latents and decoder (fixed)

- **k = 10 PCs** (fixed now; not tuned) of the area activity: samples = every lag of the lick-aligned window -0.5..+0.75 s
  of every lick (single trials).
- **Decoder input**: the 10 latents averaged in 4 lick-aligned bins: [-0.2, 0), [0, 0.15), [0.15, 0.3), [0.3, 0.5] s
  (40 features) -> StandardScaler + multinomial logistic regression, C = 0.1 fixed (no tuning). **Secondary family
  "cue"**: same latents (same PCA), 3 cue-aligned bins 0-0.5, 0.5-1.0, 1.0-1.5 s.
- **Pre-stroke reference, per animal (FROZEN, rule 10)**: PCA on the pooled pre sessions (each centered on its own mean),
  the decoders trained on all pre licks, and the reference latent trajectories (below), plus one leave-one-session-out
  sibling per pre session -- one payload via `frozen_models.load_or_fit` (kind `reembed`, align `first_lick`, source
  `areas_split_r8_k10`, train labels = the animal's August pre sessions). Never refit per render.

## 3. The three decoders on each session (2-fold, blocked)

Trials are split into 2 folds by interleaved contiguous blocks of 10 trials (time order). For each fold f as the SCORED
half, the other half is the CALIBRATION half; every session-specific fit uses calibration trials only; the session value
= mean of the 2 folds. Pre sessions are scored against their leave-one-session-out reference (pre->pre).

- **(i) FROZEN**: calibration-centered features projected on the PRE PCs, frozen pre decoder.
- **(ii) ALIGNED** (unsupervised, label-free): the session's OWN 10 PCs (calibration half), then a linear map of its
  latents onto the pre latents fitted by **CCA (Gallego 2020)** on time-locked TRIAL-AVERAGED latent trajectories POOLED
  OVER SPOUTS (no target labels): the lick-aligned -0.5..+0.75 s average and the cue-aligned -0.5..+2.0 s average of the
  calibration half, concatenated (77 time samples x 10), against the pre reference trajectories (mean of the pre
  sessions' averages). Aligned latents = own latents x B A^-1 (canonical weights); then the SAME frozen decoder.
  **Secondary alignment**: orthogonal Procrustes on the same trajectories (fewer free parameters).
- **(iii) REFIT**: the same architecture (own PCs, same bins, same C) trained on the calibration half's labels.

## 4. Quantities, null, interpretation (fixed)

Per session: (i), (ii), (iii); **A = (ii) - (i)** (gap closed by alignment), **R = (iii) - (ii)** (gap not closed),
**G = (iii) - (i) = A + R**.
- **NULL = pre->pre**: the same quantities on held-out pre sessions (LOSO reference). Drift reference: the June epoch
  (lag ~60 d), plus per-session A and R against the day gap to the nearest pre session (descriptive).
- **Change vs pre**: `analysis_kit.boot_delta` (paired animals -> sessions, animal-weighted) for each epoch (acute,
  subacute, chronic, june) on (i), (ii), (iii), A, R, G. A change = 95 % interval excludes 0 AND >= 3 / 4 animals same sign.
- **Re-embedding index** = dA / dG (the epoch-minus-pre changes; point = ratio of the boot_delta point estimates; interval
  = ratio of the PAIRED draws of A and G, same seed so the animal / session draw is identical). Per animal: the ratio of
  that animal's epoch-minus-pre means. Also reported: the plain ratio (ii - i) / (iii - i) of animal-epoch means for each
  epoch including pre->pre.
- **Interpretation rule (fixed now)**: an epoch is interpreted only if dG is a change (the frozen decoder lost ground
  against the refit beyond pre->pre). Then: **index >= 0.67 and dR not a change -> the OLD code RE-EMBEDDED**;
  **index <= 0.33 and dA not a change -> a NEW code**; otherwise PARTIAL, reported as the fraction of the frozen->refit gap
  closed by alignment. The per-animal indices must fall on the same side in >= 3 / 4 animals. If dG is not a change: "no
  extra frozen->refit gap to explain" (the index is not interpreted).
- **Spatial readout**: the aligned decoder's effective AREA weights (own PCs x alignment map x decoder weights; RMS over
  classes and bins, normalized to shares summing to 1) per session; pre = the held-out pre session's own aligned readout.
  Group shares: contralesional (right) vs ipsilesional (left); orofacial (ALM, tjM1, SSp-m, SSp-n, SSs) vs other;
  boot_delta per epoch. Map: ipsilesional = image LEFT.

## 5. Stability gate (checked BEFORE interpreting any post epoch)

- **G1 alignment efficiency (pre->pre)**: per animal, mean (ii) above chance / mean (i) above chance >= 0.7 in >= 3 / 4
  animals. Fails -> the alignment loses the code even without a stroke; result reported as UNINFORMATIVE.
- **G2 correspondence-shuffled alignment**: CCA fitted after permuting the time samples of the session's trajectory
  (breaks the time-locked correspondence) must NOT close the gap: pooled post-epoch index_tshuf < 0.25 and A_real > A_tshuf.
- **G3 label-permuted control**: spout labels permuted within each half (calibration and scored): (i), (ii), (iii) above
  chance within +-0.03 pooled (the alignment cannot manufacture decodability).
- **G4 split-half reliability**: Pearson r over sessions of A from fold 0 vs fold 1 (and of the index ingredients). r <
  0.3 -> only epoch / animal means interpreted, no session time courses.
- **C5 synthetic re-embedding (code check)**: held-out pre sessions with the area features rotated by a random 45 x 45
  orthogonal matrix: (i) must fall, (ii) must be unchanged (the CCA-on-own-PCs pipeline is invariant to any orthogonal
  re-mixing of the areas by construction; this checks the code, not the data).
- Per-animal table printed BEFORE any bootstrap (rule 8).

## 6. Known limits (stated in advance)

- Two-fold halves: the refit (iii) trains on ~half a session, the frozen decoder on ~all pre licks, so (iii) - (i) is
  negative or small pre->pre; the CHANGE vs pre->pre is the readout, never the raw gap.
- CCA on 77 smooth time samples: the trajectory average pooled over spouts constrains mostly the condition-independent
  dimensions; whether it also pins the spout dimensions is exactly what G1 measures.
- Matching uses the spout label to define the pre range (behavior selection, identical for all three decoders); the
  alignment itself never sees labels.
- n = 4 animals; June includes learning; chronic lag (> 60 d for some sessions) exceeds June's.

## Outcome (run 2026-10-30, `scripts/reembedding_test.py`, log `reembedding/run_20261030.log`)

119 curated sessions loaded (107 August+ incl. 31 pre; 12 June). Frozen references `frozen-new` (local store
`C:/Users/SabatiniLab/frozen_models`, NOT published): PS92 `reembed_first_lick_areas_split_r8_k10_ff071fa7adc2` (8 pre),
PS93 `..._406d7b72db20` (7), PS94 `..._9118ed846678` (8), PS95 `..._b9644e98b25e` (8).

### Gate: FAILED on G1 -> by the pre-registered rule the re-embedding test is UNINFORMATIVE

| gate | result | pass |
|---|---|---|
| G1 alignment efficiency pre->pre, (ii)/(i) above chance | PS92 0.29, PS93 0.44, PS94 0.45, PS95 0.29 (need >= 0.7 in 3/4) | **NO** |
| G2 time-shuffled CCA vs time-locked, post A | -0.455 vs -0.356 (real better, but neither closes anything) | weak |
| G3 label-permuted above chance (i / ii / iii) | -0.008 / -0.005 / +0.001 | yes |
| G4 split-half r of A (116 sessions) | 0.56 | yes |
| C5 random area rotation, pre (code check) | (i) 0.598 -> 0.027; (ii) 0.218 -> 0.218 exactly | yes |

The label-free CCA alignment (spout-pooled time-locked trajectories, k = 10) loses 55-71 % of the frozen decoder's
above-chance accuracy on HELD-OUT PRE sessions where nothing changed (pre (i) 0.50-0.66 vs (ii) 0.17-0.30). The
spout-pooled averages pin the condition-independent dimensions but not the spout dimensions, which is the limitation
stated in section 6. So the alignment cannot re-embed even an unchanged code, and an alignment that fails to close a
post-stroke gap says nothing about re-embedding vs a new code. The orthogonal-Procrustes variant behaves the same (index
CI spans 0).

### Per-animal (matched first licks, spout, lick window; above-chance balanced accuracy)

| animal | pre (i / ii / iii) | acute | subacute | chronic | June |
|---|---|---|---|---|---|
| PS92 | .603 / .172 / .448 | .477 / .105 / .371 | .485 / .081 / .446 | .544 / .089 / .546 | .467 / .002 / .381 |
| PS93 | .502 / .220 / .448 | .304 / .085 / .304 | .365 / .118 / .448 | .368 / .053 / .582 | .454 / -.009 / .486 |
| PS94 | .662 / .300 / .568 | .365 / -.005 / .226 | .351 / .042 / .441 | .323 / .102 / .484 | .547 / .203 / .539 |
| PS95 | .615 / .180 / .586 | .431 / .138 / .319 | .533 / .107 / .500 | .598 / .163 / .590 | .469 / .225 / .502 |

Change vs pre->pre (boot_delta, matched; per animal PS92 / PS93 / PS94 / PS95):
- (i) frozen: acute -0.201 [-0.269, -0.140] 0/4 up; subacute -0.162 [-0.266, -0.079]; chronic -0.137 [-0.273, -0.030]
  (-.059 / -.135 / -.339 / -.017); June -0.111 [-0.162, -0.063] (4/4 down).
- (iii) refit: acute -0.207 [-0.313, -0.105]; subacute -0.053 [-0.119, 0.015]; chronic +0.039 [-0.050, 0.122]
  (+.098 / +.134 / -.083 / +.005); June -0.035 [-0.084, 0.020].
- G = (iii) - (i): acute -0.006 n.s.; subacute +0.109 [0.026, 0.183] 3/4; **chronic +0.176 [0.069, 0.268] 4/4**
  (+.157 / +.269 / +.256 / +.022); **June +0.076 [0.046, 0.106] 4/4**.
- A = (ii) - (i): acute +0.064 [-0.007, 0.127]; subacute +0.031 [-0.012, 0.077]; chronic +0.021 [-0.047, 0.110]
  (-.025 / -.033 / +.141 / .000); June -0.002 n.s.
- R = (iii) - (ii): subacute +0.078 [0.004, 0.152] 3/4; **chronic +0.155 [0.046, 0.269] 4/4**; June +0.078 n.s.
- Index dA / dG (paired draws): acute undefined (dG ~ 0); subacute 0.29 [-0.19, 0.86] (per animal .23 / .26 / .29 /
  -2.8); chronic 0.12 [-0.33, 0.58] (-.16 / -.12 / .55 / .00); June -0.02 [-1.8, 2.1].
- All licks (sensitivity): chronic G +0.198 [0.105, 0.276] 4/4, A +0.001 n.s., R +0.197 [0.092, 0.286] 4/4, index
  0.01 [-0.19, 0.23]; acute (more sessions unmatched) A +0.108 [0.046, 0.159] 4/4, index 0.63 [0.26, 1.09] -- the
  matched primary does not reproduce it (acute dG not a change).
- Cue family (index): chronic 0.03 [-0.59, 0.55]. side_ml family: indices meaningless (dG near 0, CIs span +-10).

### Descriptive reading, NOT a gated conclusion

Chronically the frozen pre decoder falls further behind a same-session refit than it does between pre sessions (dG
+0.18, 4/4; PS95 small). The label-free alignment closes none of that growth (dA ~ 0; the growth sits in R, 4/4), so
the point estimate of the index (0.12) is in the "new code" zone. **But G1 failed**: this alignment cannot re-embed
even the unchanged pre code, so "the alignment does not recover it" is expected whatever the truth, and the index is
not interpreted. Also, **June (no stroke, ~60 d) shows dG +0.076 (4/4)**, so part of the chronic frozen->refit gap
growth is drift-sized (chronic 0.18 vs June 0.08: about half beyond drift; June includes learning).

Spatial read-out (aligned decoder's effective area weights; uninterpretable for the same reason, reported for
completeness): contralesional share pre 0.52 -> chronic 0.48 (change -0.038 [-0.071, -0.007], 4/4 down); orofacial
share unchanged chronically (+0.009 n.s.); acute ipsilesional orofacial share -0.043 [-0.075, -0.013] 4/4.

### Deviations from the pre-registration

- None in design. Additions: the plain ratio (ii - i) / (iii - i) per animal-epoch is in the tables but unusable
  (denominator near 0 or negative pre->pre, as foreseen in section 6). Figures: decoders (lick / cue / side), index,
  areas, gate.
- The frozen references were frozen on the analysis desktop and NOT published to MICROSCOPE (FROZEN_MODELS §5: publish
  from one box; left for Priya to decide).

### What would make the test informative (not run; would need a new pre-registration)

An alignment with spout-resolving but target-free correspondence. Options: kinematics-binned trajectories (landing
angle bins; a different label, and close to the target), alignment on ITI-lick or ENLv trajectories binned by
executed direction, or Degenhart-style anchors (areas whose loadings stay stable). Any of these must first pass G1 on
pre->pre.
