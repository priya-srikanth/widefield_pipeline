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

## 7. Follow-up: two scale variants of Wt (WtF, WtR), injected grid and real sessions (11-04, Priya OK)

Same harness, grid, seeds, oracle and readouts as §2. Wt, N and Z are kept as references, and their cells are
identical to the first run (1.18 M cells, |Δk| = 0). The real-session Wt / N / Z equal `centre_mixshift_1104`
`D_sessions_frames.csv` to 1e-16 (214 sessions × aligns). Diagnostic only; nothing stored or published.

### The new frames

Both keep the Wt centre (the DAQ-label balanced mean of the session's working rows) and change only the scale.

| tag | scale |
|---|---|
| **WtF** | FIXED: the animal's pooled pre-stroke SD, the training frame's own (mean over the model's training sessions of their working-row SD, raw units; the N frame's scale). No per-session SD |
| **WtR** | ROBUST per session: the within-spout residual SD, sqrt(mean_q var_q), equal weight over spouts with ≥ 5 rows. Calibrated per feature by the pooled pre ratio (working SD / residual SD over the training sessions), so it matches the training scale at pre. A shrunk spout mean cannot lower it |
| WtR0 | the same residual SD without the calibration (check only) |

### 7a. Pre cost and mix-only bias (cue; per animal 92 / 93 / 94 / 95, then pooled [CI])

**Pre cost** (nat E0, frame − O):

| frame | far contra, per animal | far contra, pooled | overall, pooled |
|---|---|---|---|
| Wt | 0 | 0 | 0 |
| WtF | −.007 / −.021 / −.000 / +.003 | −.006 [−.019, +.004] | −.001 [−.009, +.005] |
| WtR | −.006 / +.003 / −.003 / +.006 | +.000 [−.008, +.007] | +.002 [−.001, +.005] |
| WtR0 | | +.012 [+.003, +.021] | +.003 [+.000, +.006] |
| N | | −.008 | −.016 [−.034, −.002] |

At precue every new frame is within ±.006.

**Mix-only bias** (acute E0):

| frame | far contra, per animal | far contra, pooled | overall, pooled |
|---|---|---|---|
| Wt | +.143 / −.002 / +.069 / −.076 | +.034 [−.064, +.148] | −.018 [−.045, +.003] |
| WtF | +.110 / +.035 / +.106 / +.027 | **+.069 [+.004, +.160]** | −.005 [−.024, +.013] |
| WtR | +.227 / +.044 / +.105 / +.026 | **+.101 [+.022, +.228]** | +.001 [−.022, +.023] |
| N | −.005 / +.023 / +.081 / −.102 | −.001 [−.093, +.084] | −.015 |
| Z | +.227 / −.002 / +.079 / +.060 | +.091 [+.003, +.220] | −.019 |

- **Both new frames remove Wt's small overall mix drift** (−.018 → −.005 / +.001).
- **But at far contra they read high under the acute mix in all four animals.** Pooled WtF is +.07 and WtR +.10,
  both with intervals above 0, where Wt's interval spans 0.
- At precue the new frames are within ±.03, with no interval clear of 0.

### 7b. Recovery of the spout-specific loss (E1 / E4; frame change / oracle change at far contra)

**Cue, nat arm, per animal:**

| frame | E1 f = .5: PS92 / PS93 / PS94 / PS95 | E1 f = .25 / .5 / .75, animal mean | close-spout change at f = .5 (truth 0) |
|---|---|---|---|
| Wt | .65 / .79 / .62 / .67 | .75 / .68 / .74 | −.009 |
| WtF | .69 / .77 / .58 / .69 | .80 / .68 / .73 | −.007 |
| WtR | .68 / .79 / .69 / .71 | .82 / .71 / .76 | −.008 |
| WtR0 | | .83 / .68 / .73 | −.010 |
| N | 1.02 / .98 / .89 / 1.13 | 1.15 / 1.00 / .99 | 0 |
| Z | | .70 / .67 / .73 | −.011 |

- **Neither scale variant fixes the understatement:** WtF .68 and WtR .71 against Wt .68.
- E4u / E4r equal E1 f = .5 for every Wt variant (the offset is absorbed). N gives .59 / 1.38: the offset pushes it
  either way.
- **Precue, nat** (E1 f = .5): Wt .85, WtF .83, WtR .85, N .97.
- **Acute arm** (cue E1 f = .5; noisy, §2): Wt .95, WtF 1.13, WtR 1.20, N 1.06. Per animal WtR is 1.96 / 1.07 /
  1.22 / .92. The over-recovery rides on the mix-only far-contra inflation above.

**Correction to §4.** The mechanism bullet under "Which frame recovers the spout-specific loss" says the larger part
of Wt's understatement comes from the session SD. That is wrong.

- A fixed scale (WtF) and a scale the class means cannot move (WtR) give the same recovery as Wt (.68-.71). The scale
  is not the cause.
- The understatement comes from the CENTRE. Shrinking far contra moves the six-spout centre away from far contra by
  f/6 of the far-contra offset. Re-centring therefore pushes every row toward far contra:
  - far contra regains part of its distance;
  - the other spouts pick up the −.01 spill-over.
- On a frozen decoder near its boundary, that 1/6 shift is worth about a third of the accuracy change.
- Only frames whose centre does not include the far-contra rows recover E1 fully: N, Rref and R5.

### 7c. Effect part under E2 / E2s / E3 / E5 (pooled; * = CI excludes 0)

**Cue:**

| effect | Wt | WtF | WtR | N | Z |
|---|---|---|---|---|---|
| E2 gain g = .6, nat, far contra | +.026* | **+.008** | +.026* | −.018 | +.026* |
| E2 g = .6, nat, overall | +.035* | **+.002** | +.035* | −.014 | +.035* |
| E2 g = .6, acute, far contra | +.137* | **+.013** | +.137* | +.088 | +.137* |
| E2 g = .8, acute, far contra | +.070* | +.059 | +.070* | +.052 | +.070* |
| E2 g = 1.25, nat, far contra | −.004 | −.000 | −.004 | +.001 | −.004 |
| E2s g = .6, nat, far contra | +.008 | +.009 | **+.000** | −.013 | +.011* |
| E2s g = .6, nat, overall | +.010* | +.002 | **−.000** | −.001 | +.013* |
| E2s g = .6, acute, far contra | −.008 | −.065* | −.082 | −.007 | −.032 |
| E3u / E3r, k = 1, 2 | 0 | 0 | 0 | +.05 to +.06 / −.07 to −.19 | 0 |
| E5 / E5c / E5s | 0 | 0 | 0 | 0 | 0 |

**Precue:** E2 g = .6 nat far contra Wt +.036*, WtF +.001, WtR +.036*. E2s is within ±.01 everywhere. E3 / E5 are 0
for every Wt variant.

- **WtF is the only frame that sees a proportional gain change and is still immune to offsets and rest changes.** Its
  E2 effect part is ~0 at every level (acute +.013 against +.137 for Wt).
- **WtR behaves exactly like Wt under gain** (scatter scaled, so its residual SD scales too). It is the best frame for
  a signal-only global loss (E2s nat +.000).
- **Under the acute mix both new frames overstate E2s** (−.07 / −.08).

### 7d. Real post-stroke sessions (section-D working rows; pre = held-out LOSO)

**Cue, far contra, per animal × epoch (session means):**

| animal | epoch | Wt | WtF | WtR | N | Z |
|---|---|---|---|---|---|---|
| PS92 | pre | .867 | .860 | .861 | .858 | .870 |
| PS92 | acute | .156 | .195 | .150 | .175 | .182 |
| PS92 | subacute | .504 | **.737** | .517 | .420 | .482 |
| PS92 | chronic | .717 | .721 | .738 | .314 | .725 |
| PS93 | pre | .746 | .725 | .749 | .726 | .743 |
| PS93 | acute | .160 | .185 | .175 | .238 | .152 |
| PS93 | chronic | .640 | .665 | .665 | .608 | .643 |
| PS94 | pre | .919 | .918 | .916 | .919 | .921 |
| PS94 | acute | .359 | **.574** | .388 | .329 | .542 |
| PS94 | chronic | .675 | .660 | .631 | .335 | .689 |
| PS95 | pre | .926 | .929 | .932 | .922 | .930 |
| PS95 | acute | .244 | **.366** | .195 | .293 | .268 |
| PS95 | chronic | .847 | .827 | .816 | .690 | .854 |

**Headlines, pooled.** Level = mean of animal means; Δ = frame − Wt (`boot_delta` of session differences); deficit =
epoch − held-out pre (`boot_delta`).

| quantity | Wt | WtF | WtR | N | Z |
|---|---|---|---|---|---|
| cue far contra, acute | .230 | **.330** (Δ +.100 [+.026, +.183]) | .227 (Δ −.003 [−.034, +.026]) | .259 (+.029) | .286 (+.057) |
| deficit | −.635 [−.710, −.558] | −.528 [−.645, −.398] | −.638 [−.736, −.538] | −.597 [−.675, −.518] | −.580 |
| cue far contra, subacute | .602 | .657 (+.055 [−.021, +.205]) | .605 (+.003) | .505 (−.097) | .605 |
| cue far contra, chronic | .720 | .718 (−.001) | .712 (−.007) | .487 (−.233) | .728 |
| deficit, chronic | −.145 [−.221, −.080] | −.140 | −.152 | −.369 | −.138 |
| cue overall, acute | .575 | .637 (+.062 [+.012, +.123]) | .603 (+.028 [+.005, +.053]) | .619 (+.044) | .594 (+.019) |
| cue overall, chronic | .802 | .797 | .800 | .670 | .804 |
| cue close spouts, acute | .686 | .759 (+.074 [+.016, +.142]) | .719 (+.033 [+.005, +.064]) | .735 | .672 |
| precue far contra, acute | .198 | .185 (−.013 [−.043, +.010]) | .198 (+.001) | .251 (+.053) | .209 |
| deficit | −.321 | −.328 | −.319 | −.271 | −.305 |
| precue far contra, chronic | .381 | .397 (+.016 [+.005, +.030]) | .385 (+.004) | .488 (+.107) | .385 |
| precue overall, acute | .355 | .364 (+.009) | .356 (+.001) | .345 (−.010) | .352 |

**Per-animal cue acute far-contra deficit** (Wt / WtF / WtR / N):

| animal | Wt | WtF | WtR | N |
|---|---|---|---|---|
| PS92 | −.712 | −.666 | −.711 | −.683 |
| PS93 | −.586 | −.540 | −.574 | −.488 |
| PS94 | −.560 | −.345 | −.528 | −.590 |
| PS95 | −.682 | −.563 | −.737 | −.629 |

### 7e. Reading

**WtR.**
- On real data it is a drop-in for Wt: acute far contra .227 against .230, chronic .712 against .720, precue
  identical. Overall rises slightly (+.03 acute).
- On the injected grid it buys nothing for a spout-specific loss (.71 against .68). It is the best frame for a
  signal-only global loss.
- It is mix-biased at far contra (+.10), where Wt is not clearly biased.
- **No case for switching to it.**

**WtF.**
- It is the one frame that reports a proportional gain change correctly while staying immune to offsets and rest
  changes.
- On real data it changes the headline: acute far contra .230 → .330 (+.10). That is carried by PS94 (+.21) and PS95
  (+.12), with PS92 +.04 and PS93 +.03. PS92 subacute also rises (+.23).
- The likely cause is the acute rise in the trial-row SD (REST_CENTRE §5d: ×1.41 PS94, ×1.11-1.14 elsewhere):
  - Wt divides that rise out; WtF keeps it, as signal.
  - If the rise is a real gain increase, WtF is right. If it is added trial-to-trial variance (more incomplete,
    variable trials), WtF reads it as signal and inflates the margins.
  - The injected grid cannot tell these apart, and WtF's +.07 mix-only far-contra bias points the same way.
- **The deficit stays large under WtF** (−.53 [−.65, −.40]; −.35 to −.67 per animal).

**For the bracket.**
- Wt stays the primary.
- WtF is worth reporting beside N as the gain-sensitive frame, with its offset immunity stated. It is the
  "Wt-centred but not z-scored" reading.
- The far-contra deficit holds in every frame (−.53 to −.64 cue acute).
- How much of the acute gap between frames is a gain change (WtF) and how much is a day offset (N) is the open
  question.

Nothing here changes production; Priya decides.

### 7f. Limits

- These are the same as §5.
- WtR's calibration ratio is fitted on the training sessions, so at pre it matches by construction. WtR0 shows that
  the uncalibrated version costs little: +.012 far contra pre.
- WtF uses the pre-stroke SD of the model's own training sessions. A real post-stroke change in SD therefore enters
  WtF in full, whatever its cause.

### 7g. Files (`decoder_mix_checks/injected_effect_1104/`)

- **Scripts:** `injected2.py` (frames, grid, real sessions; 3 workers) and `summarise2.py`.
- **Logs:** `injected2_log.txt`, `summarise2_log.txt`, and `test2_PS95_cue_1seed_log.txt` (timing test; outputs
  deleted and rerun).
- **Per unit (`units2/`):** `<an>_<al>_cells.csv.gz`, `_real.csv.gz`, `_repro.json`.
- **Tables (`tables2/`):**
  - `P_pooled.csv` (pre cost, mix-only, recovery, effect parts, with CIs);
  - `B_bias_sessions.csv`, `B_effectpart_sessions.csv`, `C_change_sessions.csv`;
  - `REAL_sessions.csv`, `REAL_pooled.csv` (levels, Δ vs Wt and deficits per epoch, with CIs).
