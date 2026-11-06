# G4 by camera class + mix-shift test of the state-discriminator frame (2026-11-04)

Cohort 1 (PS92-95). Read-only checks: no production default, model, deck or page changed.
Scripts, CSVs and logs are in `M:/MICROSCOPE/Priya/Widefield/labcams/decoder_mix_checks/g4_camclass_mixshift_1104/`.
The code ran on the 11-04 working tree (HEAD 4d7c484c plus the section-G edits that were then uncommitted). It
reproduces production exactly: see section 2.

## 1. Question

Section G's G4 readouts are `poststroke_compare.looks_like_which` (lw) and `fits_engaged_distribution` (fe). Each
reports the share of post-stroke DAQ-no-lick trials that look "engaged-like" (cue, acute, animal means):

- z frame: .195
- spout frame centred on contact + incomplete: .323
- spout frame centred on contact + incomplete + jaw_only: .502

This note asks two things.

- (1) Which camera class of no-lick trial carries that share? Genuine attempts (incomplete lick) or jaw_only / none?
- (2) Which frame stays stable when only the trial mix changes, as it does after the stroke?

Priya's 11-04 design change is folded in:

- a CAMERA-CLASS discriminator, the primary reading;
- a 4th frame, z on the WORKING rows;
- the mix-shift test run for both discriminators under all four frames.

## 2. Design

### Frames

Each frame is used for the training sessions and for the scored sessions.

| tag | frame | centre rows | centre statistic |
|---|---|---|---|
| z | production `session_z` | the session's DAQ-lick rows | mean / SD |
| zw | new (11-04) | the session's working rows (contact + incomplete, either DAQ side) | mean / SD |
| sp | `spout_frame`, `in_working` (11-04 production working) | contact + incomplete | label-free bal_em under the frozen spout model |
| spj | `spout_frame`, `in_working_jaw` (11-03 afternoon models) | contact + incomplete + jaw_only | label-free bal_em |

### Discriminators

- **daq (production).** Pre DAQ-lick vs pre DAQ-no-lick, position-balanced: `balanced_lick_sample`, `lick_pipe`.
- **cam (primary, new).** Pre WORKING (contact | incomplete) vs NONE (no response), from either DAQ side.
  - Position-balanced with the same algorithm and the same rng use.
  - The none class includes close-mid trials whose jaw was not judged; that is section G's 'none'.
  - jaw_only and after_end are not in training. They are scored as their own classes.
  - It is built in the check script (`g4common.cam_balanced_sample`) from the production helpers.
    `poststroke_compare` is not edited.

### Classes

The classes are `working_trials.g_group` on the A4 snapshot, applied to the DAQ-no-lick rows: incomplete, jaw_only,
none, after_end (after the last camera lick), plus rare contact. The DAQ-lick rows are the control.

- PS93_0813 and the June sessions have no camera classes, so they leave every split. In the z frame PS93_0813 stays
  in training, and its rows show up as `unk`.

### Task 1

The unit is one post session, with keep = the cue pool's `preserved_positions`, as in section G.

- lw: fresh `RandomState(0)`, then one full-pre fit.
- fe: fe's own sequence. One `RandomState(0)` runs across the leave-one-pre-session-out fits, then the full fit.
- Every scored row is kept with its DAQ side, class and spout.
- Pre reference per class: the held-out pre rows (each pre session scored by the fit that left it out). They are
  taken at the keep set of the animal's first acute session.

### Task 2

The unit is one held-out August pre session (LOSO, fe's fit sequence). Keep = the animal's modal acute keep.

1. **Resample.** Each held-out session's trials are resampled with replacement to a target class mix:
   - DAQ-lick, plus the no-lick classes incomplete / jaw_only / none / after_end / contact;
   - the target is the animal's own acute (or subacute) post sessions;
   - spout totals stay those of the session itself.
2. **Variants.**
   - per_spout: the epoch's class mix at each spout.
   - pooled: the epoch's mix over all spouts, applied at every spout.
3. **Recompute and score.** The held-out session's frame is recomputed from the resampled rows (z: lick rows;
   zw / sp / spj: working rows), then scored. Training sessions are untouched.
4. **Arms.**
   - ctrl = the natural mix resampled the same way (the resampling null).
   - shift = the target mix.
5. **Bias.**
   - bias = shift − ctrl of each class's rate, reweighted by its natural spout composition at the kept spouts.
   - NL_overall = what the frame adds to the raw no-lick share beyond the change in class composition.
6. **Seeds and intervals.** 50 seeds (0-49). Per-animal intervals are the 2.5 / 97.5 percentiles over seeds. The
   pooled values are animal-weighted (`analysis_kit.boot_delta` of session biases against 0).

### Checks

- Our replicated DAQ discriminator equals production `looks_like_which` / `fits_engaged_distribution`. Over 92 lw and
  86 fe post-session values per frame, the largest difference is 1e-16.
- The pooled endpoints reproduce: lw z / sp / spj .195 / .323 / .502, fe .170 / .322 / .498.
- The frame recomputed on the natural rows equals the production scored frame. sp / spj / zw differ by 0; z differs
  by at most 5e-6, which is float rounding of the re-z.

## 3. Task 1: G4 by camera class (cue, lw)

Values are the share engaged-like (cam: working-like). Pre = held-out pre rows; n = trials. Values from fewer than 10
trials are marked *.

### 3a. Per animal: camera-class discriminator (primary)

| animal | epoch | L (DAQ-lick) z / sp / spj | NL all z / zw / sp / spj | incomplete z / sp / spj (n) | jaw_only z / sp / spj (n) | none z / sp / spj (n) |
|---|---|---|---|---|---|---|
| PS92 | pre | .81 / .80 / .80 | .29 / .31 / .31 / .31 | .38 / .38 / .31 (16) | .30 / .30 / .40 (10) | .23 / .27 / .27 (26) |
| PS92 | acute | .76 / .82 / .86 | .06 / .15 / .18 / .28 | .05 / .16 / .26 (93) | .05 / .13 / .24 (218) | .06 / .20 / .31 (376) |
| PS93 | pre | .78 / .80 / .80 | .33 / .47 / .46 / .46 | .40 / .43 / .42 (252) | .05 / .50 / .50 (22) | .08 / .44 / .44 (25) |
| PS93 | acute | .78 / .83 / .87 | .36 / .41 / .43 / .53 | .34 / .41 / .49 (117) | .30 / .36 / .46 (225) | .38 / .46 / .57 (372) |
| PS94 | pre | .91 / .90 / .90 | .13 / .13 / .13 / .12 | .00 / .00 / .00 (4*) | .09 / .07 / .07 (54) | .11 / .11 / .11 (64) |
| PS94 | acute | .94 / .96 / .97 | .09 / .26 / .25 / .48 | .17 / .41 / .63 (147) | .06 / .20 / .42 (315) | .09 / .26 / .49 (456) |
| PS95 | pre | .95 / .95 / .94 | .06 / .06 / .06 / .08 | .14 / .16 / .16 (37) | .04 / .05 / .08 (141) | .05 / .05 / .08 (149) |
| PS95 | acute | .97 / .97 / .98 | .21 / .26 / .26 / .47 | .38 / .38 / .75 (8*) | .25 / .25 / .42 (12) | .12 / .24 / .41 (17) |

### 3b. Per animal: DAQ-class discriminator (production), z / sp / spj

| animal | epoch | NL all | incomplete | jaw_only | none | after_end |
|---|---|---|---|---|---|---|
| PS92 | pre | .16 / .14 / .14 | .38 / .31 / .31 | .00 / .00 / .00 | .08 / .08 / .08 | .20* |
| PS92 | acute | .24 / .45 / .59 | .26 / .45 / .57 | .17 / .37 / .53 | .26 / .50 / .66 | .24 / .45 / .53 |
| PS93 | pre | .19 / .24 / .24 | .23 / .26 / .26 | .09 / .09 / .09 | .08 / .08 / .08 | .25 / .28 / .34 |
| PS93 | acute | .32 / .34 / .46 | .33 / .33 / .45 | .31 / .30 / .43 | .32 / .36 / .47 | .35 / .39 / .51 |
| PS94 | pre | .14 / .15 / .13 | .25* | .11 / .11 / .07 | .09 / .09 / .09 | .17 / .18 / .18 |
| PS94 | acute | .12 / .33 / .55 | .27 / .50 / .71 | .10 / .27 / .48 | .11 / .33 / .56 | .05 / .24 / .48 |
| PS95 | pre | .06 / .07 / .09 | .27 / .30 / .22 | .06 / .06 / .09 | .03 / .04 / .08 | .03 |
| PS95 | acute | .11 / .18 / .45 | .25* / .38* / .75* | .17 / .25 / .50 | .00 / .06 / .29 | none |

### 3c. Share of the engaged-like calls among acute no-lick trials, by class (DAQ discriminator, z frame) vs share of trials

| | PS92 | PS93 | PS94 | PS95 (n=38) |
|---|---|---|---|---|
| incomplete | .13 (of .12 trials) | .15 (.15) | .31 (.14) | .50 (.21) |
| jaw_only | .20 (.27) | .27 (.28) | .24 (.31) | .50 (.32) |
| none | .52 (.47) | .47 (.47) | .41 (.45) | .00 (.45) |
| after_end | .15 (.15) | .11 (.10) | .04 (.10) | none |

The same split under sp and spj moves by .05 or less per cell, and the camera-class discriminator gives the same
picture (tables/compact_cue_cam_lw.csv).

### 3d. Animal-weighted pooled values and paired contrasts

Cue, lw. Pooled values are the mean over animals of cells with n ≥ 10. Contrasts are `boot_delta` of session shares.

| quantity | z | zw | sp | spj |
|---|---|---|---|---|
| cam: acute NL all (pre) | .179 (.202) | .270 (.241) | .279 (.241) | .441 (.243) |
| cam: acute incomplete / jaw / none | .189 / .166 / .163 | .308 / .229 / .276 | .327 / .235 / .287 | .459 / .383 / .446 |
| cam: NL all, post − pre | −.03 [−.20, .12] | +.05 [−.13, .20] | +.06 [−.11, .19] | +.21 [−.02, .41] |
| cam: incomplete − none, acute | +.10 [−.04, .23] | +.06 [−.07, .19] | +.07 [−.07, .20] | +.11 [−.09, .29] |
| daq: acute NL all (pre) | .197 (.137) | .311 (.155) | .326 (.148) | .512 (.150) |
| daq: acute incomplete / jaw / none | .286 / .187 / .175 | .421 / .266 / .307 | .427 / .296 / .313 | .579 / .483 / .496 |
| daq: NL all, post − pre | +.07 [.02, .14] | +.18 [.08, .29] | +.19 [.10, .30] | +.38 [.26, .48] |
| daq: incomplete − none, acute | +.12 [−.01, .25] | +.12 [−.02, .26] | +.12 [−.04, .28] | +.14 [−.06, .35] |

### 3e. Precue

The precue discriminators barely separate the two classes pre-stroke. Pre DAQ-lick .73 vs no-lick .43 (cam) or .36
(daq). Acute no-lick trials sit at .57-.64 in every class and every frame, and no class stands out. Full tables are in
`summarise_split_log.txt`.

### 3f. Reading of task 1

**The post-stroke engaged-like share is NOT carried by the incomplete-lick trials.**

- In every frame the engaged-like calls fall on the classes roughly in proportion to their trial counts (3c). About
  half land on 'none' trials.
- Incomplete beats none only in PS94; PS95 points the same way but its cells are thin. PS92 and PS93 show no
  difference. That is one or two animals, so by rule 8 it is not a cohort result. The pooled incomplete − none
  intervals all span 0.

**The frame moves every class by about the same amount** (daq, z → sp: incomplete +.14, jaw +.11, none +.14; z → spj:
+.29 / +.30 / +.32).

- The rise from .195 to .323 to .502 is a uniform offset. It is not a shift toward attempts.
- zw ≈ sp everywhere. What moves the frame is which rows set it (working rows vs DAQ-lick rows), not plain z vs bal_em.

**Under the camera-class discriminator in z, acute no-lick trials are no more working-like than pre-stroke no-lick
trials** (.18 vs .20; post − pre −.03 [−.20, .12]). That holds for every class.

## 4. Task 2: mix-shift bias (shift − ctrl), cue

### 4a. Achieved mix

The mix that was actually reached falls short of the target.

- Pre no-lick trials are thin: 2.6% (PS92) to 13% (PS95) of a pre session.
- So the target was reached only by upsampling: 15-78% of drawn trials come from upsampled cells, at up to 100x.
- 7-35% of the target mass sat at spouts where the class was absent. That mass was spread over the classes present.

Typical achieved acute per_spout mix (DAQ-lick share): PS92 .81, PS93 .72, PS94 .61, PS95 .65, against natural
.87-.97. The real post-stroke mix is more extreme, so these biases are LOWER bounds.

The incomplete cells of PS92 and PS94 rest on 1-2 pre trials per session, so their bias is not interpretable (for
example PS94 incomplete +.5 with truth 0/0). They are left out of the reading below.

### 4b. Per animal, camera-class discriminator (primary), acute, per_spout

Values are bias [seed 2.5, 97.5].

| animal | class | z | zw | sp | spj |
|---|---|---|---|---|---|
| PS92 | L | −.005 [−.02, .02] | +.010 | +.034 [.01, .06] | +.047 [.03, .07] |
| PS92 | none | +.03 [−.30, .31] | +.09 | +.13 [−.08, .33] | +.20 [.00, .40] |
| PS92 | NL_overall | −.01 | +.05 | +.10 | +.13 |
| PS93 | L | −.008 | +.011 | +.005 | +.017 [.00, .04] |
| PS93 | incomplete | −.01 [−.08, .06] | +.02 | +.02 | +.04 [−.03, .11] |
| PS93 | none | −.004 | −.002 | −.002 | +.04 |
| PS93 | NL_overall | −.001 | +.007 | +.005 | +.03 |
| PS94 | L | −.026 [−.04, −.01] | −.001 | +.009 | +.038 [.02, .05] |
| PS94 | jaw_only | −.03 [−.12, .05] | +.05 | +.07 | +.17 [.05, .26] |
| PS94 | none | −.02 [−.08, .03] | +.02 | +.03 | +.24 [.20, .28] |
| PS94 | NL_overall | −.014 [−.03, .01] | +.04 [.00, .07] | +.07 [.04, .10] | +.17 [.13, .21] |
| PS95 | L | +.004 | +.018 [.01, .03] | +.013 [.00, .02] | +.042 [.03, .05] |
| PS95 | incomplete | +.05 [−.05, .17] | +.17 [.04, .35] | +.16 [.05, .29] | +.43 [.24, .65] |
| PS95 | jaw_only | +.005 [−.14, .18] | +.15 [.02, .29] | +.14 [.01, .29] | +.40 [.28, .50] |
| PS95 | none | +.005 [−.05, .08] | +.05 | +.02 | +.20 [.09, .29] |
| PS95 | NL_overall | +.001 | +.13 [.03, .24] | +.12 [.02, .21] | +.35 [.22, .40] |

### 4c. Animal-weighted pooled bias, cue, acute

`boot_delta`; n = 4 animals (3 for after_end).

| disc / variant | class | z | zw | sp | spj |
|---|---|---|---|---|---|
| cam / per_spout | L | −.009 [−.02, .00] | +.010 [.00, .02] | +.015 [.00, .03] | +.036 [.02, .05] |
| | jaw_only | +.002 [−.02, .03] | +.077 [.01, .16] | +.073 [.01, .15] | +.185 [.04, .36] |
| | none | +.003 [−.02, .03] | +.039 [.00, .09] | +.046 [.00, .12] | +.171 [.07, .28] |
| | NL_overall | −.005 [−.02, .01] | +.056 [.01, .11] | +.071 [.02, .13] | +.168 [.06, .30] |
| cam / pooled | NL_overall | −.006 [−.02, .00] | +.053 [.02, .09] | +.061 [.02, .11] | +.145 [.05, .25] |
| daq / per_spout | L | −.010 [−.02, .00] | +.012 [.00, .02] | +.016 [.00, .03] | +.032 [.02, .05] |
| | jaw_only | −.010 [−.03, .00] | +.031 [−.01, .07] | +.033 [.00, .07] | +.177 [.08, .27] |
| | none | −.008 [−.02, .01] | +.118 [.02, .29] | +.133 [.03, .31] | +.245 [.11, .42] |
| | NL_overall | −.005 [−.02, .01] | +.072 [.02, .15] | +.074 [.03, .15] | +.168 [.10, .24] |
| daq / pooled | NL_overall | −.001 [−.01, .01] | +.037 [.01, .08] | +.031 [.00, .07] | +.077 [.03, .12] |

Per animal, the largest |bias| in z is .05 or less in every animal × class × variant cell with enough pre trials:

| disc / variant | z (each animal) | spj (each animal) |
|---|---|---|
| cam / per_spout | ≤ .05 | .04-.50 |
| daq / per_spout | ≤ .07 | .14-.44 |

At subacute mixes, which are closer to pre, z / zw / sp all stay within about .02 pooled, and spj still adds +.05
(NL_overall). Precue gives the same ordering with smaller sizes: pooled NL_overall, cam per_spout acute: z −.004,
zw +.02, sp +.02, spj +.06.

### 4d. Reading of task 2

**z is mix-stable.** For both discriminators, in all four animals, both variants and both epochs, the rates stay at
their natural-mix values when only the mix changes. This holds even though per_spout also shifts the spout make-up of
the lick rows.

**Every frame centred on working rows is not stable.**

- When the mix moves toward post-stroke, more no-lick working rows enter the centre (incomplete for zw / sp, plus
  jaw_only for spj). The centre then moves toward the no-lick pattern, so every class reads more engaged-like.
- zw and sp behave the same. The fault is the centre rows, not bal_em.
- spj is worst, about 2-3x sp, because jaw_only trials are common post-stroke.

**This artefact explains a large part of task 1's frame gap.** The pre trials could reach only part of the post
mix, yet the bias still accounts for:

- about half of the sp − z gap: NL_overall +.07 per_spout of the +.13 gap;
- about half of the spj − z gap: +.17 of the +.31 gap.

The real post mix is more extreme, so the true share is likely larger.

## 5. Which frame to trust post-stroke, and the G4 headline under it

**Trust z (production `session_z`)**, read through the camera-class discriminator.

- z is the only frame that does not move when the trial mix changes.
- The spout frames' extra engaged-like share is mostly the mix moving the centre, not a property of the trials.
- The 11-03 jaw-inclusive centre (.502) should not be used for G4.
- Keeping z is the current production default, so nothing changes in production.

G4 headline in z, cue, acute (per animal in 3a; animal-weighted):

- **Camera-class discriminator.**
  - Acute no-lick trials are working-like at .18. Pre-stroke no-lick trials score .20, and DAQ-lick trials .86 in
    both epochs.
  - By class: incomplete .19, jaw_only .17, none .16, after_end .18. Post − pre is about 0 for each class.
  - Acute no-lick trials look like pre-stroke non-working trials, whatever their camera class. Post-stroke
    incomplete licks do not look like successful trials (pre held-out incomplete .30 → acute .19).
- **DAQ discriminator (production).**
  - .197 vs pre .137 (+.07 [.02, .14]).
  - By class: incomplete .29 (pre .29, no change); jaw_only .19 (pre .07); none .18 (pre .07).
  - The small rise is in the jaw_only / none trials, not the attempts. It is not consistent across animals:
    PS92 and PS93 rise, PS94 and PS95 do not.

Precue: the readout separates the states too weakly pre-stroke to say anything by class.

## 6. Limits

- **n = 4 animals.** PS95 acute has only 38 no-lick trials; PS92 has 58 held-out pre no-lick trials, and PS94 has
  4 pre incomplete trials at its acute keep.
- **Thin pre classes.** Pre incomplete cells are thin for PS92 and PS94, so their mix-shift biases are not
  interpretable. PS93 is the only animal with many pre incomplete trials.
- **The target mix was not fully reached** (heavy upsampling, 7-35% lost mass). Biases are lower bounds.
- **The mix-shift test checks ONE property: stability under a change of composition.** It cannot test the other job
  the spout frame was meant to do: absorb a global post-lesion shift in the signal. z being mix-stable does not prove
  that z has no post-stroke drift. Its DAQ-lick control is unchanged, though: L post − pre −.015 [−.04, .005] for daq
  and −.002 for cam.
- **The camera-class discriminator's positives are mostly contact trials.** As a result it scores held-out pre
  incomplete trials at only .30 (PS93 .40). It reads "contact-like vs none", and a weaker incomplete signal would be
  hard for it to show.
- **'none' includes close-mid trials whose jaw was not judged.** These are counted in `n_unjudged` in the split CSVs.
- **The zw and sp centres use camera-class membership (working rows).** They are label-free with respect to the
  discriminator's no-lick class, but not with respect to the camera classes. This applies to the cam discriminator.
- **Missing camera classes.** PS93_0813 and the June sessions have no camera classes and are absent from every split.
  In z, PS93_0813 stays in the DAQ discriminator's training, as in production.
- **Keep sets.** Task 2 uses one keep per animal (modal acute), not the per-session keep.

## 7. Files

All files are under `M:/MICROSCOPE/Priya/Widefield/labcams/decoder_mix_checks/g4_camclass_mixshift_1104/`.

- **Scripts:**
  - `build_pools.py`: pools cached in `pools/` (derived; safe to delete);
  - `g4common.py`: frames, the cam discriminator and the helpers;
  - `g4_split.py`: task 1;
  - `mixshift.py`: task 2;
  - `summarise_split.py`, `summarise_mixshift.py`, `compact_tables.py`.
- **Outputs:** `split/*.csv.gz` (+ `_prod.csv`), `mixshift/*.csv.gz`, `mixshift/mix_*.csv`, `mixshift/natcheck_*.csv`.
- **Tables:**
  - `tables/R_reproduction.csv`;
  - `T1_cells_animal_epoch_class.csv`, `T1_cells_session_class.csv`;
  - `T1_n_animal_epoch_class_spout.csv` (n per animal × epoch × class × spout; codes 0-5 = close_center, close_L,
    close_R, far_center, far_L, far_R);
  - `C_contrasts.csv`, `compact_*_lw.csv`;
  - `B_bias_per_animal.csv`, `B_bias_pooled.csv`, `B_overall_total_shift.csv`;
  - `M_*.csv`.
- **Logs:** `summarise_split_log.txt`, `summarise_mixshift_log.txt`, `compact_tables_log.txt`, `g4_split_log.txt`,
  `mixshift_log.txt`.
