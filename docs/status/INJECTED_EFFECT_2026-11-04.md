# Which per-session frame recovers a KNOWN change? Injected-effect validation of the frozen spout decoder (2026-11-04)

Cohort 1 (PS92-95). Read-only diagnostic, approved by Priya after the literature review
(`docs/LITERATURE_CROSS_SESSION_NORMALISATION_2026-11-04.md` §4.1). No production default, model, deck or page was
changed. Every number uses the stored 11-03-late production models (frozen-hit, expected ids). The only fits are in
memory: the R5 LOSO refits and the Iref per-session offsets. Nothing was stored or published.
Scripts, CSVs and logs are in `M:/MICROSCOPE/Priya/Widefield/labcams/decoder_mix_checks/injected_effect_1104/`.
Pools, frame engine and section-D reproduction are reused from `CENTRE_MIXSHIFT_2026-11-04.md`, the rest frame from
`REST_CENTRE_2026-11-04.md`, and Rref / Iref from `REFSPOUT_CENTRE_2026-11-04.md`.

## 1. Question

Every frame we compared so far was judged by stability or by agreement with other frames, never against a known
change. The review found that nobody in the field checks a frame against a known change either. Here the truth is
known by construction:

- Take held-out August pre sessions (LOSO models, as production).
- Impose the animal's acute trial mix.
- Inject a known effect into the features.
- Ask which frame reports the injected change, and which absorbs a change that is not neural.

## 2. Design

**Sessions.** 31 held-out August pre sessions with camera classes (PS92 8, PS93 7, PS94 8, PS95 8), each scored with
its stored LOSO model on its camera working rows, cue and precue.

**Arms.**

- **nat:** natural rows, no resampling. Every condition is run. E0 here is the pre cost (and the reproduction).
- **acute:** the animal's pooled ACUTE working-row mix, resampled with replacement within (spout, class) cells.
  - Both the spout shares and the per-spout incomplete share are set to the acute values; the total is the session's
    own working count.
  - 30 seeds. The same resample per seed is used for every condition, so effects are paired.
- **ctrl:** the natural mix resampled (E0 only); this is the noise floor.

**Injections.** They apply to every trial row, working and non-working. Parameters come from the natural session, in
its pool-z coordinates:

- m6 = the true-label balanced working centre;
- μ_q = spout q's working-row mean;
- sdB = between-session SD of the training sessions' six-spout centres.

| tag | effect | what is injected |
|---|---|---|
| E0 | none (mix only) | control |
| E1_f | spout-specific loss | far-contra rows += f (m6 − μ_fc), f = .25 / .5 / .75; the far-contra mean moves a fraction f toward the six-spout mean; within-class scatter untouched |
| E2_g | global gain | x → m6 + g (x − m6), g = .6 / .8 / 1.25 (signal and trial scatter both scaled) |
| E2s_g | signal-only global loss (added) | x += (g − 1)(μ_y − m6), g = .6 / .8; every spout's mean shrinks, scatter kept |
| E3u_k / E3r_k | uniform offset on every row AND the rest frames | u = all-features direction (an F0 / brightness error is a near-uniform dF/F offset); r = sdB ⊙ N(0,1), fixed per session; ‖d‖ = k‖sdB‖, k = 1 / 2 |
| E4u / E4r | offset + spout-specific loss | E1 f = .5 + E3 k = 1 |
| E5 | rest state only | rest centre += the animal's MEASURED acute R5 shift, rest SD × 1.3; trials untouched (E5c centre only, E5s SD only) |

**Injection sizes** (cue, session means, RMS over features in units of the working-row SD):

| animal | ‖m6 − μ_fc‖ | offset u, k = 1 | offset r, k = 1 | measured rest shift |
|---|---|---|---|---|
| PS92 | .27 | .72 | .23 | .13 |
| PS93 | .22 | .39 | .17 | .08 |
| PS94 | .38 | .47 | .18 | .09 |
| PS95 | .29 | .42 | .16 | .12 |

**Frames.** Each centre is recomputed from the resampled, injected rows:

- Wt (primary), W (bal_em), N, Cto, Z (re-z on the DAQ-lick rows);
- Rref (Rref_w_a);
- R5 (rest frame, LOSO model refitted in memory in the R5 frame);
- IrefN / IrefWt (the class-balanced intercept-only refits, Iref_N_bal / Iref_Wt_bal).

**Truth.** The ORACLE frame O is the natural session's Wt frame (m6, sdWt), computed before any mix or injection. It
adds the injected offset to its centre for E3 / E4 and ignores rest changes. The truth is O's accuracy on the
injected rows.

- O treats E1, E2 and E2s as real effects and E3 and E5 as nuisance.
- For E2 that is a choice: a gain change could be neural or hemodynamic, and the oracle counts it as real.

**Readouts.** All are on the scored rows (the acute composition in the acute arm; frame and oracle see identical
rows):

- **bias** = frame − O, per seed, then the session mean;
- **effect part** = bias(Ek) − bias(E0), paired per seed;
- **recovery** = the frame's change (Ek − E0) / the oracle's change.

Per animal first (rule 8). Pooled = `analysis_kit.boot_delta` of session values against 0 (animal-weighted).
Frame-only (natural-weight) versions are in `tables/P_bias_frameonly_pooled.csv` and agree in sign.

### Reproduction (passed before any injected number was read)

- All 8 units are frozen-hit with the expected ids. Per-session accuracy and confusion are exact against
  `bcdg_1104/checkDG`.
- W predictions on the natural rows equal production row for row, and O equals Wt at E0.
- At nat E0, all 62 session × align cells match the earlier tables to 1e-16 for every frame:
  - W / Wt / N / Cto / Z against `centre_mixshift_1104 D_sessions_frames.csv`;
  - Rref / IrefN / IrefWt against `refspout_centre_1104`;
  - R5 against `rest_centre_1104`.

### The achieved acute mix (cue; summed over each animal's held-out pre sessions)

| animal | far-contra share (nat → acute) | far-contra incomplete share (nat → target → achieved) | pre far-contra incomplete trials | lost |
|---|---|---|---|---|
| PS92 | .171 → .154 | .014 → .946 → .434 | 7 | .47 |
| PS93 | .167 → .161 | .028 → .852 → .610 | 17 | .24 |
| PS94 | .166 → .122 | .025 → .952 → .847 | 17 | .12 |
| PS95 | .166 → .075 | .022 → .976 → .712 | 17 | .24 |

The acute far-contra rows are therefore a heavy upsampling of 7-17 pre incomplete trials. **The acute arm is noisy at
far contra (per-animal effects of ±.1-.2). The nat arm is the clean test of effect recovery, and the acute arm tests
mix + effect together.**

## 3. Results (cue unless stated)

### 3a. Pre cost (nat E0; frame − O, where O = natural Wt) and mix-only bias (acute E0)

| frame | pre cost, far contra | pre cost, overall | mix-only, far contra (per animal 92 / 93 / 94 / 95) | mix-only, overall |
|---|---|---|---|---|
| Wt | 0 | 0 | +.034 [−.06, +.15] (+.14 / −.00 / +.07 / −.08) | −.018 [−.045, +.003] |
| W | +.001 | −.004 | −.018 (+.13 / −.11 / +.04 / −.14) | −.022 [−.043, −.002] |
| N | −.008 | −.016 [−.034, −.002] | −.001 (−.01 / +.02 / +.08 / −.10) | −.015 |
| Cto | −.001 | −.000 | −.027 (+.13 / −.12 / +.05 / −.16) | −.004 [−.024, +.015] |
| Rref | −.013 | −.009 | −.052 (+.11 / −.13 / +.05 / −.24) | −.010 |
| R5 | −.020 | −.030 [−.052, −.010] | +.021 | −.024 [−.052, −.003] |
| IrefN | −.015 | +.000 | −.060 | −.034 [−.053, −.017] |
| IrefWt | −.009 | −.001 | −.077 | −.044 [−.068, −.021] |
| Z | +.002 | −.001 | **+.091 [+.003, +.22]** | −.019 |

- **The mix alone moves far contra by ±.1-.2 per animal in every session-centred frame**, with the sign differing by
  animal. Pooled, nothing but Z is clear of 0. Z overstates far contra under the mix (+.09), as found on 11-03.
- **Overall**, the mix lowers accuracy by about .02 in W, Wt, Z and R5, and by .03-.04 in the Iref arms. Cto is the
  most mix-stable (−.004), as in CENTRE_MIXSHIFT.

### 3b. The main table: frame × effect, effect part of the bias (pooled; * = CI excludes 0)

**Far contra.**

| effect | truth: oracle change (nat / acute) | Wt | W | N | Cto | Rref | R5 | IrefN | IrefWt | Z |
|---|---|---|---|---|---|---|---|---|---|---|
| E1 f = .25, nat | −.090 / −.224 | +.022* | +.019* | −.014 | +.025* | −.009 | −.014 | +.044* | +.044* | +.027* |
| **E1 f = .5, nat** | −.274 / −.295 | **+.089*** | +.061* | −.001 | +.090* | −.012 | −.010 | +.156* | +.151* | +.091* |
| E1 f = .75, nat | −.516 / −.399 | +.133* | +.062* | +.003 | +.142* | −.000 | −.017 | +.277* | +.273* | +.142* |
| E1 f = .5, acute | | +.016 | +.040 | −.017 | +.074 | +.043 | −.031 | +.125* | +.150* | −.015 |
| E2 g = .6, nat | −.026 / −.137 | +.026* | +.026* | −.018 | +.026* | −.158* | −.047* | +.025* | +.026* | +.026* |
| E2 g = .6, acute | | +.137* | +.137* | +.088 | +.137* | −.008 | −.008 | +.142* | +.137* | +.137* |
| E2s g = .6, nat | −.182 / −.262 | +.008 | −.010 | −.013 | +.012 | **−.185*** | −.023 | +.040* | +.038* | +.011* |
| E3u k = 1, nat | 0 | 0 | 0 | +.046 | 0 | 0 | 0 | 0 | 0 | 0 |
| E3u k = 2, nat | 0 | 0 | 0 | +.063* | 0 | 0 | 0 | 0 | 0 | 0 |
| E3r k = 1, nat | 0 | 0 | 0 | −.073* | 0 | 0 | 0 | 0 | 0 | 0 |
| E3r k = 2, nat | 0 | 0 | 0 | −.189* | 0 | 0 | 0 | 0 | 0 | 0 |
| E4u, nat | −.274 | +.089* | +.061* | **+.112*** | +.090* | −.012 | −.010 | +.156* | +.151* | +.091* |
| E4r, nat | −.274 | +.089* | +.061* | **−.105*** | +.090* | −.012 | −.010 | +.156* | +.151* | +.091* |
| E5, nat | 0 | 0 | 0 | 0 | 0 | 0 | −.029 | 0 | 0 | 0 |
| **E5, acute** | 0 | 0 | 0 | 0 | 0 | 0 | **−.189*** | 0 | 0 | 0 |

How to read the signs:

- A positive effect part on a LOSS condition means the frame UNDERSTATES the loss.
- A negative one means it OVERSTATES the loss.
- On a nuisance condition (E3 / E5) the right answer is 0.

**Overall** (nat unless stated; the oracle changes for E1 f = .5 / E2 g = .6 / E2s g = .6 are −.04 / −.04 / −.18):

| effect | Wt | W | N | Cto | Rref | R5 | IrefN | Z |
|---|---|---|---|---|---|---|---|---|
| E1 f = .5 | +.008* | +.005* | −.000 | +.007* | −.002 | −.002 | +.010* | +.008* |
| E2 g = .6 | +.035* | +.035* | −.014 | +.035* | −.077* | −.045* | +.036* | +.035* |
| E2s g = .6 | +.010* | −.004 | −.001 | +.012* | −.058* | −.017* | +.023* | +.013* |
| E3r k = 1 | 0 | 0 | −.071* | 0 | 0 | 0 | 0 | 0 |
| E3u k = 2 | 0 | 0 | −.121* | 0 | 0 | 0 | 0 | 0 |
| E5 (acute) | 0 | 0 | 0 | 0 | 0 | −.035* | 0 | 0 |

Total biases with intervals for every cell: `tables/P_bias_pooled.csv`; effect parts: `tables/P_effectpart_pooled.csv`.

### 3c. Per animal: recovery of the spout-specific loss (E1 f = .5, nat; frame change / oracle change at far contra)

| animal | oracle change | Wt | W | N | Cto | Rref | R5 | IrefN | IrefWt | Z |
|---|---|---|---|---|---|---|---|---|---|---|
| PS92 | −.277 | .65 | .81 | 1.02 | .66 | 1.04 | 1.15 | .43 | .44 | .68 |
| PS93 | −.228 | .79 | .85 | .98 | .76 | 1.22 | 1.00 | .56 | .61 | .72 |
| PS94 | −.317 | .62 | .71 | .89 | .62 | .95 | .88 | .37 | .36 | .62 |
| PS95 | −.275 | .67 | .78 | 1.13 | .66 | 1.01 | 1.13 | .40 | .43 | .67 |

The other five spouts are unchanged in the oracle. Under Wt / Cto / Z they lose −.009, and under Iref −.02 (spill-over).

**Precue, E1 f = .5, nat** (oracle change −.17):

| frame | recovery |
|---|---|
| Wt | .82-.88 |
| W | .79-.92 |
| N | .95-1.03 |
| Rref | .87-1.01 |
| R5 | .92-1.00 |
| Iref | .60-.85 |

**Per animal, nuisance conditions (nat; effect part at far contra / overall):**

| animal | N, E3u k = 1 | N, E3r k = 1 | N, E3r k = 2 | R5, E5 nat | R5, E5 acute |
|---|---|---|---|---|---|
| PS92 | +.097 / −.100 | −.209 / −.138 | −.428 / −.332 | +.002 / −.009 | −.000 / −.011 |
| PS93 | +.079 / −.035 | −.036 / −.057 | −.132 / −.161 | +.001 / −.007 | −.117 / −.028 |
| PS94 | −.019 / −.013 | +.002 / −.020 | −.036 / −.118 | −.050 / −.032 | −.446 / −.085 |
| PS95 | +.025 / −.008 | −.050 / −.069 | −.161 / −.187 | −.067 / −.024 | −.194 / −.018 |

**Per animal, E2 gain g = .6 (far-contra effect part):**

| animal | Wt / W / Cto / Z, nat | Rref, nat | Wt / W / Cto / Z, acute |
|---|---|---|---|
| PS92 | +.023 | −.181 | +.008 |
| PS93 | +.006 | −.229 | +.119 |
| PS94 | +.039 | −.117 | +.220 |
| PS95 | +.035 | −.105 | +.201 |

### 3d. Precue (pooled effect part, nat, far contra)

| effect | Wt | W | N | Rref | R5 | IrefN |
|---|---|---|---|---|---|---|
| E1 f = .5 | +.027* | +.023* | +.004 | +.008 | +.007 | +.046* |
| E2 g = .6 | +.036* | +.036* | −.000 | −.098* | +.011 | +.041* |
| E3u k = 1 | 0 | 0 | +.123 | 0 | 0 | 0 |
| E3r k = 1 | 0 | 0 | −.077* | 0 | 0 | 0 |
| E5 | 0 | 0 | 0 | 0 | −.061 | 0 |

At precue E5 acute is −.040 at far contra and −.040* overall.

## 4. Reading

**Which frame recovers the spout-specific loss (E1 / E4)?**

- **Recovered in full, in all four animals:** N (recovery .89-1.13), Rref (.95-1.22) and R5 (.88-1.15). None of them
  re-centres on the far-contra rows.
- **Wt, Cto and Z UNDERSTATE it by about a third**, in every animal (recovery .62-.79; precue .82-.88).
  - Mechanism: part comes from the centre, which holds 1/6 of the far-contra shift. The larger part comes from the
    scale: shrinking the far-contra class reduces the session SD, which re-expands every row.
  - They also spill a small loss onto the other five spouts (−.009).
- **W understates it less (.71-.85)**, because its EM down-weights far contra.
- **Iref understates it most (.36-.61):** the label-fitted far-contra offset re-thresholds a real loss away.
- **Consequence for REFSPOUT.** Iref's smaller real-data deficit (−.46 against Wt's −.64) is what a GENUINE
  spout-specific loss would also produce. It is not evidence that part of the deficit is a day offset.
- **E4 behaves as E1** for every session-centred frame: they absorb the offset exactly and leave the E1 part as
  above. N does not: the same offset made N's far-contra loss look smaller (u: +.11) or larger (r: −.11).

**Which frame absorbs the nuisance (E3 / E5)?**

- **Offsets (E3).** Every session-centred frame absorbs them exactly (Wt, W, Cto, Z, Rref, R5, Iref: 0.000), by
  construction. N absorbs nothing.
- **Rest change (E5).** Every trial-based frame is untouched. R5 is the only frame that moves: acute far contra
  −.19 [CI excludes 0], −.45 in PS94 and −.19 in PS95; overall −.035. The rest frame turns a pure rest-state change
  into a far-contra "deficit", which is what REST_CENTRE suspected.

**Global gain (E2), and the review's predictions.**

- **Prediction: "Wt / Z understate a loss shared by all spouts."** TRUE for a proportional gain (E2, signal and
  scatter scaled together):
  - Wt, W, Cto, Z and IrefWt are exactly blind to it (change 0.000);
  - the oracle change is small on natural rows (−.03 far contra, −.04 overall) but large at the acute mix
    (−.14 far contra), where far-contra rows sit near the boundary;
  - so they understate by +.14 there.
- **The same prediction is FALSE for a signal-only global loss (E2s,** every class mean shrunk, scatter unchanged):
  Wt recovers .91-1.07 at far contra and .92-.97 at the other spouts.
- **So a z-scored frame hides a global gain change, not a global loss of pattern.**
- **Prediction: "N can only overstate a loss."** TRUE for overall accuracy: every offset lowered N, by −.04 to −.21.
  FALSE at a single spout: a uniform (F0-like) offset RAISED N's far-contra accuracy (+.05 to +.06; per animal up to
  +.10, and +.30 at precue PS92). So N can understate a far-contra loss.
- **Prediction: "W can go either way."** Under the mix, W's far-contra bias is +.13 / −.11 / +.04 / −.14 by animal.
  Under E1 it understates, by less than Wt. Partly confirmed.
- **New finding: Rref OVERSTATES every global change.** Rref anchors on the close spouts, so a global change at those
  spouts is pushed onto every other spout:
  - E2 g = .6: −.16 at far contra and −.08 overall, nat;
  - E2s: recovery 1.9-2.1, i.e. it doubles a global loss at far contra;
  - this happens in all four animals.

**Which frame suits which claim?**

| frame | spout-specific deficit | global change | nuisance (offset) | rest change | pre cost |
|---|---|---|---|---|---|
| Wt (primary) | understates by ~1/3 (conservative) | blind to a proportional gain; tracks a pattern loss | immune | immune | 0 |
| W | understates by ~1/4; mix bias both ways | as Wt | immune | immune | ~0 |
| N | unbiased | partly | biased either way at a spout; always lower overall | immune | −.016 overall |
| Rref | unbiased IF the close spouts are unchanged | overstates ~2x | immune | immune | −.009 |
| R5 | unbiased | partly | immune | converts it into a deficit (−.19) | −.03 overall |
| Iref | understates by ~1/2 to 2/3 | blind (IrefWt) | immune | immune | ~0 |
| Cto / Z | as Wt; Z also mix-biased at far contra (+.09) | as Wt | immune | immune | ~0 |

**Is Wt a sound primary for the far-contra claim? Yes, as a CONSERVATIVE frame.**

- No nuisance in this test can manufacture a far-contra loss under Wt: offsets and rest changes give exactly 0, and
  the mix-only bias is centred near 0 (+.03 pooled, ±.1 per animal).
- A genuine spout-specific loss is reported at ~2/3 of its size.
- So a far-contra deficit under Wt is real, and it is at least as large as Wt says, to the extent the effect is
  spout-specific and linear.
- What Wt can miss is a proportional global gain change. That is a separate claim and needs a frame that does not
  z-score (N), with its offset caveat.

**What the bracket should look like.**

- Wt (primary, conservative) beside N (no session centre, unbiased for a spout-specific loss).
- State the directions this test measured:
  - Wt can only understate a spout-specific loss (by ~1/3);
  - N's per-spout error is offset-driven and can go either way, while its overall error can only lower accuracy.
- A deficit present under both is robust to every injected confound here.
- Report Rref and R5 as supplementary, each with its measured failure: Rref doubles global changes, and R5 turns a
  rest change into a deficit.
- Do not use Iref as a bound on the size of the deficit. It understates a real spout-specific loss the most.
- In real data N reads a SMALLER acute far-contra deficit than Wt (−.60 against −.64), although Wt understates a
  spout-specific loss in this test. Two readings fit:
  - a nuisance offset in N happens to raise far contra (as E3u did);
  - the real change is not purely a linear far-contra shrink.
- Either way the deficit is large under both, and Wt's figure is the safer one.

Nothing here changes production; Priya decides.

## 5. Limits

- **The injected effects are linear and synthetic.** E1 shrinks a mean along one axis; real post-stroke change may
  rotate patterns, add variance, or differ between contact and incomplete trials. Recovery ratios are for these
  effect shapes only and should not be extrapolated to the real −.64 deficit.
- **n = 4 animals and 7-8 pre sessions each.**
- **Pre incomplete trials at far contra are 7-17 per animal.** Only PS93 has many pre incomplete trials (252), and
  those are at far ipsi and far mid. The acute mix is therefore an upsampling at far contra; the target was not
  reached (lost .12-.47). Acute-arm far-contra numbers are noisy (seed and animal spread ±.1-.2), so the nat arm
  carries the recovery reading.
- **The oracle is the natural Wt frame.** It counts a gain change as real by definition. Whether a gain change after
  stroke is neural or optical cannot be settled here.
- **E5 uses the measured acute R5 centre shift (raw) and a fixed SD ×1.3.** PS95 has one acute session.
- **Offsets are set at 1-2× the pre between-session SD of the six-spout centres.** Real post-stroke offsets may be
  larger (REST_CENTRE measured rest-centre shifts of up to 1.2 log-odds at far contra).
- **R5 uses an in-memory refit.** The others use the stored models.

## 6. Files (`decoder_mix_checks/injected_effect_1104/`)

**Scripts:**

- `injected.py`: reproduction, injections, all frames and the oracle; arms nat / ctrl / acute; 3 workers.
- `summarise.py`: sections R, M, 0-5.
- `compact.py`: the tables above.

**Logs:**

- `injected_log.txt`;
- `summarise_log.txt`;
- `compact_log.txt`;
- `test_PS95_cue_2seeds_log.txt` (timing test; its outputs were deleted and rerun).

**Per unit (`units/`):** `<an>_<al>_cells.csv.gz` (k / n per arm × seed × condition × frame × spout × class),
`_mix.csv`, `_params.csv`, `_repro.json`.

**Tables (`tables/`):**

- `R_readouts_seed.csv.gz`;
- `B_bias_sessions.csv`, `B_effectpart_sessions.csv`, `B_bias_frameonly_sessions.csv`;
- `C_change_sessions.csv`, `C_recovery.csv`;
- `O_oracle_sessions.csv`;
- `P_bias_pooled.csv`, `P_effectpart_pooled.csv`, `P_bias_frameonly_pooled.csv`;
- `M_achieved_mix.csv`, `M_injection_sizes.csv`.
