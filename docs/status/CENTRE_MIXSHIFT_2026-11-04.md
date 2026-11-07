# Which rows should set the scoring center? Spout decoder and C projections (2026-11-04)

Cohort 1 (PS92-95). Read-only checks: no production default, model, deck or page changed, and no model was fitted or
published (every decoder number uses the stored 11-03-late production models, frozen-hit).
Scripts, CSVs and logs are in `M:/MICROSCOPE/Priya/Widefield/labcams/decoder_mix_checks/centre_mixshift_1104/`.
The design and code follow the G4 mix-shift test (`G4_CAMCLASS_MIXSHIFT_2026-11-04.md`).

## 1. Question

Production scores the frozen spout decoder in a per-session bal_em frame. The center comes from the session's WORKING
rows (contact + incomplete camera licks). After the stroke the incomplete share rises sharply, most at far contra.
Two ways that could move the center, and with it every trial of the session:

- (i) the EM confuses spout priors (for example, far-contra incomplete trials read as far mid);
- (ii) incomplete trials have a different pattern at the same spout, which a label-shift correction does not handle.

The same question applies to the C projections (far-contra CD score, pattern correlation, engagement axis), whose
production frame is the session z.

## 2. Design

### Frames

Every frame is applied to the same model. Decoder: the stored model (full model for post sessions; the
leave-that-session-out model for held-out pre sessions). Projections: production's in-memory working-row FrozenFrame,
the one `_proj_arrays` uses under the working center. Only the center differs. The scored rows are always the session's
working rows (section D's 11-04 default).

| tag | center rows | center statistic |
|---|---|---|
| W | working (contact + incomplete) | label-free bal_em: production decoder frame; projection "working center" |
| C | contact only | label-free bal_em, same model and prior |
| Wt / Ct | working / contact | TRUE-LABEL balanced: equal-weight mean of per-spout means (spout from the DAQ strobe); a spout with < 5 rows is dropped and the weights renormalised |
| Wo / Co | working / contact | bal_em + pre-stroke spout offsets (11-03 `prebal_off` / Option 2): center = mean − sd · Σ π̂(c) δ(c) |
| Wto / Cto | working / contact | the same with the TRUE spout mix of the center rows instead of π̂ (no EM) |
| Z | DAQ-lick rows | production session z: projection production frame; the old decoder frame |
| Zc | contact rows | plain mean / SD (supplementary) |
| N | none (pooled pre) | one fixed frame from pre-stroke training sessions only: mean of their training centers and SDs in raw units |

Notes on the frames:

- **δ(c)**, used by the offset arms, is the pre-stroke offset of spout c's mean from the training center, in
  training-frame units, averaged over the model's training sessions. It is taken from working rows for Wo / Wto and
  contact rows for Co / Cto. These arms ASSUME the pre-stroke spout offsets still hold post-stroke.
- **N** uses the LOSO training sessions for a held-out pre session. For projections it uses all pre sessions. A common
  shift cancels in the pole-normalized projections, so only the scale matters there.
- **Wfix** (mix-shift only) is a null: the session's natural W center, never recomputed. Its "bias" is the noise floor
  from resampling the scored rows.

### Test 1: direct

Every scored session is scored under every frame, cue and precue:

- post sessions with the full model;
- held-out pre sessions with their LOSO model.

The readouts are:

- **Decoder:** accuracy per animal × epoch × spout, and per camera class.
- **Projections:** cd, pat and eax exactly as `bcdg_1104/verify_C.py`. Each is given per class as well.
- **Center mass:** per session and frame, the center rows' true spout counts, the EM π̂ and the spouts used.

### Test 2: mix-shift

Each held-out August pre session's working rows are resampled with replacement within (spout, class) cells to the
animal's own acute (or subacute) incomplete share at each spout. Spout totals stay the session's own; non-working rows
are kept once. ctrl = the natural mix resampled the same way. 50 seeds.

The held-out session's center is recomputed from the resampled rows; Z re-z-scores on the resampled DAQ-lick rows. N
is fixed by construction and is not run. References and training sessions are untouched.

Bias = shift − ctrl of FRAME-ONLY rates: class × spout rates reweighted to the session's natural composition, so a
change of mix moves them only through the center.

### Checks

Every check passed before any new frame was read.

- **Section D reproduced exactly.** All 8 models are frozen-hit with the expected ids. Per-session accuracies and
  confusions equal `bcdg_1104/checkDG/D_work_*.json`. The scored rows and predictions are identical, and the W frame
  differs by 0.
- **bcdg tables reproduced.** Max |W − table| = 1e-16 over 96 cells per align. Cue far contra acute .184, overall
  .570; precue far contra .135, overall .356.
- **Raw pools.** They rebuild the production z frame bitwise; this was checked against the G4 pools.
- **Projections reproduced.** Z and W reproduce `bcdg_1104/checkC` (z, spout_work) to 2e-16. W equals
  `grant_kit._frozen_frame`.

## 3. Results: spout decoder

### 3a. Direct, per animal, cue (session means; pre = held-out LOSO)

**Far-contra recall.**

| animal | epoch | W (prod) | C | Wt | Ct | Co | Cto | Z | N |
|---|---|---|---|---|---|---|---|---|---|
| PS92 | pre | .865 | .867 | .867 | .865 | .865 | .869 | .870 | .858 |
| PS92 | acute | .095 | .092 | .156 | .150 | .068 | .072 | .182 | .175 |
| PS93 | pre | .753 | .759 | .746 | .746 | .749 | .739 | .743 | .726 |
| PS93 | acute | .166 | .174 | .160 | .156 | .163 | .103 | .152 | .238 |
| PS94 | pre | .915 | .917 | .919 | .921 | .917 | .919 | .921 | .919 |
| PS94 | acute | .282 | .424 | .359 | .525 | .324 | .314 | .542 | .329 |
| PS95 | pre | .929 | .930 | .926 | .925 | .930 | .925 | .930 | .922 |
| PS95 | acute | .195 | .244 | .244 | .244 | .220 | .220 | .268 | .293 |

**Overall accuracy.**

| animal | epoch | W (prod) | C | Wt | Ct | Co | Cto | Z | N |
|---|---|---|---|---|---|---|---|---|---|
| PS92 | acute | .572 | .595 | .590 | .595 | .593 | .609 | .597 | .585 |
| PS93 | acute | .379 | .384 | .380 | .382 | .379 | .401 | .382 | .392 |
| PS94 | acute | .561 | .625 | .580 | .636 | .618 | .652 | .623 | .666 |
| PS95 | acute | .769 | .776 | .749 | .772 | .809 | .817 | .772 | .833 |

Pre between-session SD of W: overall .10 / .07 (PS92 / PS93), far contra .12 / .07 / .08 / .05.

### 3b. Direct, animal-weighted, cue (level; frame − W with `boot_delta` 95%)

| quantity | W | C | Wt | Co | Cto | Z | N |
|---|---|---|---|---|---|---|---|
| acute far contra | .184 | .234 (+.049 [−.002, .115]) | .230 (+.045 [.009, .078]) | .194 (+.009 [−.023, .044]) | .177 (−.007 [−.047, .035]) | .286 (+.102) | .259 (+.074 [.037, .106]) |
| acute overall | .570 | .595 (+.025 [.005, .052]) | .575 (+.004) | .600 (+.030 [.008, .056]) | .620 (+.049 [.027, .080]) | .594 (+.024) | .619 (+.049 [.010, .098]) |
| acute contact / incomplete | .662 / .205 | .687 / .235 | .658 / .249 | .700 / .209 | .725 / .212 | .665 / .312 | .705 / .285 |
| subacute far contra | .566 | .575 | .602 | .505 (−.060) | .594 (+.029) | .605 | .505 |
| chronic far contra | .656 | .657 | .720 (+.064) | .589 (−.067) | .719 (+.063 [.038, .091]) | .728 | .487 (−.169) |
| chronic overall | .767 | .767 | .802 | .747 | .803 (+.036 [.024, .048]) | .804 | .670 (−.097) |

**Precue, acute.**

| quantity | W | C | Wt | Co | Cto | Z | N |
|---|---|---|---|---|---|---|---|
| far contra | .135 | .158 | .198 | .117 | .120 (−.014 [−.043, .008]) | .209 | .251 |
| overall | .356 | .355 | .355 | .362 | .367 (+.011) | .352 | .345 |

C vs W is NOT within ~.02 everywhere. Acute overall differs by +.025 [.005, .052], and far contra by +.049; the far
contra gap is carried by PS94 (+.14) and PS95 (+.05). So the center matters.

### 3c. The EM's spout priors (cue, acute, session means)

Columns are W π̂ (with the true share of the working center rows) and C π̂ (with the true share of the contact rows).

| animal | W π̂ far contra (true) | W π̂ far mid (true) | C π̂ far contra (true) |
|---|---|---|---|
| PS92 | .005 (.154) | .298 (.148) | .012 (.010) |
| PS93 | .167 (.160) | .134 (.164) | .194 (.031) |
| PS94 | .067 (.122) | .058 (.140) | .048 (.008) |
| PS95 | .004 (.075) | .041 (.097) | .000 (.002) |

- **Mechanism (i) is real.** Under W the EM strips most of the far-contra mass, even though far contra holds 8-15% of
  the working rows. In PS92 that mass goes to far mid (π̂ .30 vs .15 true), which is exactly the feared reading. In
  PS94 / PS95 it is spread out. PS93 is unaffected.
- **Its cost.** The true-label center (Wt) beats W at far contra by +.045 [.009, .078] acute. It also does so in
  chronic (+.064), where the incomplete share is back to normal. The EM depresses far contra whatever the mix.
- **Thin contact cells.** Contact rows at acute far contra are absent or thin: 10 / 16 acute cue sessions have a spout
  with < 5 contact rows, and in 13 an EM mass is < .05. C and Ct are then 5-spout centers. That inflates far contra,
  as the coordinator predicted (Ct .269 acute). The pre-offset fill removes most of it (Cto .177, Co .194).

### 3d. Mix-shift bias (cue, acute target; frame-only; per animal [seed 2.5, 97.5])

| animal (pre incomplete) | class | W | C | Wt | Co | Cto | Z | Wfix (null) |
|---|---|---|---|---|---|---|---|---|
| PS92 (16; not interpretable) | overall | −.034 [−.06, .00] | −.007 | −.026 | −.006 | −.003 | −.013 | −.001 |
| PS93 (252; primary) | overall | −.001 [−.02, .02] | −.002 | −.011 | −.001 | .000 | −.012 | +.001 |
| PS94 (4; not interpretable) | overall | −.025 [−.04, −.01] | −.004 | −.014 | −.001 | −.002 | −.020 | .000 |
| PS95 (37; thin) | overall | −.039 [−.07, −.02] | −.013 | −.046 [−.07, −.03] | −.007 | −.007 | −.029 | −.002 |
| PS93 | contact far contra | −.019 [−.11, .07] | −.004 | +.019 | −.007 | −.005 | +.029 | −.006 |
| PS95 | contact far mid | −.058 [−.13, .00] | −.015 | −.031 | −.027 | −.026 | −.014 | −.006 |

**Pooled** (`boot_delta` of session biases vs 0), cue acute:

| class | W | C | Wt | Co | Cto | Z | Wfix |
|---|---|---|---|---|---|---|---|
| overall | −.025 [−.041, −.008] | −.006 [−.013, −.002] | −.024 [−.042, −.011] | −.004 [−.008, .000] | −.003 [−.006, .000] | −.018 [−.028, −.010] | .000 |
| contact far contra | −.029 [−.051, −.013] | +.001 | −.002 | −.004 | −.006 | +.025 [.015, .036] | −.004 |

- **Subacute and precue.** At the subacute mix every frame is within .002 overall. At precue (acute mix) W drifts
  −.008 overall, against −.002 to −.004 for C / Co / Cto.
- **What the mix actually shifted.** PS93's 252 pre incomplete trials sit at far ipsi (159) and far mid (88). Its
  natural incomplete share there (.27 / .14) already matches acute (.35 / .14). The acute mix therefore moves PS93
  almost only at far contra, where every animal has just 7-17 pre incomplete trials. Those trials were upsampled to
  a 47-83% share against an 85-98% target (lost mass .12-.47). So at far contra PS93 is no better sampled than the
  others.
- **The direction holds in every animal.** W loses accuracy as incomplete rows enter its center; PS93 is the
  smallest case.
- **Mechanism.** Wt (true labels, working rows) drifts as much as W. The drift is mechanism (ii), the different
  pattern of incomplete trials, not the EM's priors.

## 4. Results: C projections

### 4a. Direct, per animal, acute (cue; Z = current production)

| animal | CD far contra W / C / Co / Cto / **Z** | eax W / C / Co / Cto / **Z** | pat far contra Cto / **Z** |
|---|---|---|---|
| PS92 | −.175 / −.290 / −.443 / −.380 / **−.190** | 1.010 / .899 / .909 / .879 / **.852** | −.209 / **−.104** |
| PS93 | −.344 / −.420 / −.425 / −.451 / **−.325** | .791 / .697 / .723 / .749 / **.757** | −.382 / **−.234** |
| PS94 | −.178 / −.397 / −.287 / −.323 / **−.069** | 1.017 / .823 / .812 / .793 / **.783** | −.193 / **−.005** |
| PS95 | .175 / .177 / .154 / .163 / **.378** | .914 / .801 / .769 / .775 / **.813** | .035 / **.158** |

Held-out pre sessions move by .00-.01 in every frame (pre SD of Z: cd .19-.21).

**Animal-weighted, acute, frame − Z.**

| readout | Z | W | C | Co | Cto | N |
|---|---|---|---|---|---|---|
| CD far contra | −.051 | −.079 [−.167, .005] | −.181 [−.295, −.092] | −.199 [−.259, −.130] | −.197 [−.244, −.148] | +.086 |
| pattern far contra | −.046 | −.078 | −.084 | −.130 | −.141 [−.174, −.112] | +.043 |
| eax, cue | .801 | +.132 [.057, .206] | +.004 | +.002 | −.002 [−.026, .021] | +.086 |
| eax, precue | .906 | +.324 [−.02, .88] | +.257 | +.033 | +.016 [−.007, .052] | +.016 |

Chronic: Cto ≈ Z for every readout (cd −.002, pat −.003, eax −.003 / −.009).

### 4b. Mix-shift bias (acute target; frame-only; pooled `boot_delta`)

Per animal: `tables/P3_bias_per_animal.csv`. Pattern correlation shrinks with fewer rows even when the center is fixed
(Wfix −.20), so its frame effect is bias − Wfix.

| readout | W | C | Co | Cto | Z | Wfix |
|---|---|---|---|---|---|---|
| eax, cue | +.073 [.045, .103] | +.006 | −.003 | −.003 [−.013, .007] | +.002 | .000 |
| eax, precue | +.065 [−.012, .140] | −.003 | −.013 | −.009 | −.012 | −.008 |
| CD far contra, cue | +.006 [−.078, .082] | +.023 | +.029 | +.025 [−.018, .085] | +.147 [.071, .241] | .000 |
| pattern far contra, cue (− Wfix) | −.004 | +.007 | −.017 | −.019 | +.036 | 0 |

- **eax.** W drifts in all four animals (cue +.05 to +.10, including PS93 +.06). C, Co, Cto and Z do not drift.
- **CD far contra.** Z (production) drifts upward in all four animals (+.08 to +.28; per-animal intervals span 0). The
  reason is that the shift removes far-contra rows from the DAQ-lick set, the same spout-mix artifact bal_em was
  adopted for on 11-03. W is near 0 pooled but heterogeneous (PS93 +.08, PS92 −.05). Contact-centered frames stay
  within +.03.

## 5. Reading

**Spout decoder.**

- **Production W fails the stability test.** Moving the pre mix toward acute costs −.025 overall and −.03 at contact
  far contra. Most of this is mechanism (ii): the true-label working center drifts the same way.
- **W also has mechanism (i) in real post data.** Its EM strips far-contra mass (PS92 into far mid). That alone
  depresses far contra by ~.045 acute and ~.06 chronic.
- **The contact-row centers are stable** (C, Co, Cto; |bias| ≤ .013 in every animal, Cto ≤ .007).
- **Plain C cannot be used post-stroke.** Acute far contra has almost no contact rows, so C is a 5-spout center that
  inflates far contra.
- **Co is out.** Its EM-weighted offsets drift away from the true-label arms in subacute / chronic (far contra −.06 /
  −.07).
- **Cto is the stable center.** It uses contact rows, the true spout mix and pre-stroke offsets for thin spouts. It is
  mix-stable, needs no EM, and agrees with Wt / Z wherever the mix is normal (chronic, pre).

**Headline under Cto** (vs production W):

| quantity | W | Cto |
|---|---|---|
| cue far contra, acute | .184 | .177 (−.007 [−.047, .035]); essentially unchanged |
| cue overall, acute | .570 | .620 (+.049 [.027, .080]) |
| cue far contra, chronic | .656 | .719 (+.063) |
| cue overall, chronic | .767 | .803 (+.036) |
| precue far contra, acute | .135 | .120 (−.014) |
| precue overall, acute | .356 | .367 (+.011) |

- So the far-contra loss is not an artifact of the center: it stands at ~.18.
- W understates recovery elsewhere: near spouts acute, and chronic far contra.

**C projections.**

- **Engagement axis.** The current z frame is stable, and so are the contact centers. The working center is not
  (+.07 drift; the +.13 acute rise under W is mostly this artifact, as G4 found for the state discriminator). Under
  Cto the headline barely moves: cue .801 → .799, precue .906 → .923.
- **Far-contra CD.** The current z frame is NOT stable (+.15 drift). Under the stable Cto center the acute far-contra
  CD drops from −.051 to −.248 (−.197 [−.244, −.148]), lower in all four animals. That is consistent with the 11-03
  engaged-row bal_em build (−.284).
- **Pattern correlation, far contra.** It falls from −.046 to −.187 under Cto. All non-z frames agree on a lower
  value.

**Non-circular reference (N).**

- **What it shows.** No post-stroke data set the frame, and N gives higher acute far contra than W (decoder .259,
  CD +.035). The frames that post-stroke data define therefore make the acute far-contra numbers lower, not higher.
- **Why it is not a usable frame.** N falls apart with day-to-day drift: chronic overall .670 vs .803, chronic pattern
  .36 vs .61.

## 6. Should production change? (not changed here; Priya decides)

- **Spout decoder.** Consider replacing the working-row bal_em center with Cto. Its center = mean of the session's
  contact rows − sd · Σ (true spout share) × (pooled pre-stroke contact offset from the training center).
  - It uses the scored session's DAQ-strobe spout labels in the center, which bal_em was built to avoid. The spout is
    known from the strobe, not decoded.
  - It assumes the pre-stroke spout offsets hold post-stroke.
  - Effect: far contra acute unchanged (.18), overall acute +.05, chronic far contra +.06.
- **Projections.**
  - Keep z for the engagement axis (stable, unchanged).
  - For CD and the pattern families, z is mix-unstable. Cto would serve every projection family from one frame:
    eax ≈ z, CD far contra −.05 → −.25, pattern −.05 → −.19.
  - Do NOT adopt `WIDEFIELD_PROJECTION_CENTRE=working` (W): it is unstable for eax.

## 7. Limits

- n = 4 animals.
- **Far-contra incomplete trials are rare in every pre session (7-17 per animal).** The mix-shift at far contra
  therefore rests on heavily upsampled trials, and on PS93 no better than on the others. The target mix was not
  reached (lost mass .12-.47), so the biases are lower bounds.
- The shift test checks one property: stability when only the trial mix changes. It cannot test a genuine post-lesion
  change in the spout offsets, which is the assumption under Cto / Co.
- Precue EM is weak (C precue eax 1.16), which is one reason the EM arms are not recommended there.
- **Contact-centered frames depend on camera classes.** PS93_0813 and the June sessions have none and stay in the
  production frame. Contact rows come from the A4 snapshot.
- N for projections includes the held-out pre session in its own frame (1/n). For the decoder N is LOSO.
- The decoder pooled intervals are `boot_delta` over sessions within animals. PS92 and PS94 have few pre incomplete
  trials in total.

## 8. Files (`decoder_mix_checks/centre_mixshift_1104/`)

- **Scripts:**
  - `build_raw.py`: raw pools (`raw/`, 450 MB, derived, safe to delete);
  - `cmcommon.py`: frames (Engine) and resampling;
  - `decoder_centre.py`, `proj_centre.py`: tests 1 + 2;
  - `summarise_decoder.py`, `summarise_shift.py`, `summarise_proj.py`;
  - `run_chain.sh`.
- **Logs:** `summarise_decoder_log.txt` (R = reproduction; 1-3), `summarise_shift_log.txt`, `summarise_proj_log.txt`,
  `*_centre_log.txt`.
- **Tables (`tables/`):**
  - `D_sessions_frames.csv`, `D1_*`, `D1b_*_class_spout.csv`, `D2_pooled_frame_minus_W.csv`, `D3_frames_mass.csv`
    (π̂ and flags);
  - `M_achieved_mix.csv`, `B_decoder_bias_{per_animal,pooled,seed}`;
  - `P_direct_all.csv`, `P1_*`, `P2_pooled.csv`, `P3_bias_*`.
