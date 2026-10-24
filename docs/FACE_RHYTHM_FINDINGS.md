# FaceRhythm findings (cam4)

Status: **preliminary, living document.** Last updated **2026-10-24 morning** (FaceRhythm window). Update a section
when its result changes (edit, re-tag, add to the change log). Reasoning and decisions: `DECISIONS.md` (2026-10-21
evening, 2026-10-22, 2026-10-23); how everything runs: `docs/FACE_RHYTHM_PIPELINE.md`; working state: the newest
`docs/status/STATUS_*_FACE_RHYTHM_*` handoff (10-24: `STATUS_2026-10-24_FACE_RHYTHM_HANDOFF.md`).

Status tags: **STANDS** (survived the listed checks on the data named), **PROVISIONAL** (a control or the four-animal
set still pending), **PILOT** (two animals, PS92 + PS93, the pilot factor set `pilot_pre_PS92_PS93` -- to be redone on
`master_pre_v1`), **PENDING** (not yet run / not yet summarised), **CORRECTED** (an earlier statement here was wrong;
corrected numbers follow).

> **Results page (figures + one-line takeaways, in this order): https://claude.ai/artifact/X8NDdQNHi3zMP77PBrZAC1**
>
> **Story order (Priya 10-24; DECISIONS 2026-10-24 "FaceRhythm framing").**
> 1. Acute deficit that confirms what we know: acute drop in RIGHT (contralesional) lick factor amplitude > LEFT (5d).
> 2. PERSISTENT contralesional lick deficit in lick_right expression, by several methods: per lick (5b-master), peak
>    amplitude per spout side / position (5d), raw point movement (5e).
> 3. Contra / ipsi face asymmetry: contralesional licks made with less contralesional face movement (5e, 5f, maps).
> 4. Drop in the pre-stroke factors' EVR after the stroke (6-master, normalised ratio).
> 5. Survey of every other factor post-stroke: per-trial peak temporal amplitude (5d, `face_rhythm_peak.py`), + WHEN
>    each behaviour is expressed relative to the cue (timing; PENDING).
> Secondary: factor angles / shape (6c, + the pooled per-epoch fits, PENDING); task vs non-task (6d).

> **Where things stand (10-24 morning), with what each rests on.**
>
> | result | status | data it rests on / what is incomplete |
> |---|---|---|
> | cam4 tracking clean in every animal x epoch (1) | STANDS | all 108 cam4 tasks, 4 animals; cam1 / cam2 / cam3 not run (await Priya's landmark clicks) |
> | Bin times now TRUE (per-frame timestamps); PS92_0908 / PS92_0812 v1 had wrong stamps (1b) | STANDS | all 108 cam4 tasks (`timing`); pilot projections of the two tasks redone on fr-timing-v1; FR-model R2 PS92_0908 .506 -> .618 (encoding window) |
> | Interior mesh folds 0-0.15 % (1) | CORRECTED | 0-0.34 % on all 108 (PS95 highest, 0.1-0.34 %); still negligible |
> | How many factors (2) | DECIDED: **rank 12** (Priya 10-24; supersedes rank 10) | rank 10 had no left-lick factor; rank 12 has a clean left / right lick pair (6 / 8 starts, both samples); EVR .803, stability .912, leave-one-animal-out .82-.90, split half .911 (lick pair .96 / .99; f6 .36, f10 .76) |
> | Pre-stroke noise floor: leaving a day out costs <= 0.01 EVR (3) | STANDS, 4 animals | master rank 12, 31 pre days: held-out EVR .768-.820, cost <= .010 (median ~.004) |
> | Factor identities (4a) | STANDS (names 10-24) | master rank 12 named from gallery + validation on 31 pre sessions of 4 animals (lick-triggered, running onsets, rate r); pilot section 4 kept for the record |
> | Contralesional licking factor (f9) down after stroke, still down chronically (5) | PILOT (2 animals) | PS92 + PS93; re-checked on fr-timing-v1 projections 10-23 (unchanged); master projections of PS94 / PS95 PENDING |
> | Chronic contralesional (right) deficit is PER LICK, not fewer licks (5b) | **STANDS, 4 / 4 animals** (10-24) | master rank 12: lick_right per right lick chronic .57-.79 of pre in all four; right-lick count back to .97-1.41 in 3 / 4 (PS94 .52) |
> | "Licks and misses": acute far-contra completion .03-.15, recovers fully by chronic; per-lick contra misses mildly up chronically in 3 / 4 (5c) | STANDS acutely / PROVISIONAL chronic | behaviour only (LP v36 + DAQ contact), 4 / 4 animals; close + far contra pooled per lick; LP post-stroke detection caveat |
> | Pre-stroke vocabulary fits post-stroke movement worse acutely (EVR gap ~2x), PS93 back by subacute, PS92 slightly above chronically (6) | PILOT (2 animals) | PS92 + PS93; own fits with 1 start; re-checked on fr-timing-v1 10-23 (one PS92 chronic gap .042 -> .040, otherwise unchanged) |
> | FR in the encoder (encoding window's FR run A: grouped pilot factors, 50 PS92 + PS93 sessions) | PENDING summary | run done (incl. PS92_0908 / 0812 reruns on fr-timing-v1); summary by the encoding window; results belong in ENCODING_FINDINGS |
> | AR-HMM not needed for imaging (DECISIONS 10-23) | decision, not a result | FR + LP + contact / angle regressors already cover specific behaviours and hit vs miss |

> **Cohort and rules.** Consistency threshold = >= 3 of 4 animals (Priya 10-24). PS93 has a PRE-EXISTING right orofacial
> deficit and may be an outlier for that reason (its pre-stroke lick_right is already atypical); where it is the one
> exception it is named. PS92-PS95, all lesions LEFT; on cam4 the ipsilesional face is IMAGE RIGHT; spouts / licks
> named by the animal's side (right = contralesional). August-only pre baseline; engaged trials. n = 4 animals at most;
> every claim needs the per-animal table (CLAUDE.md rule 8). Expression = ABSOLUTE (weights x FR normaliser D) unless
> stated; a.u. on each animal's own scale, so compare within an animal across epochs.

## 1. Tracking quality (STANDS, 108 / 108 tasks, 10-23)

Production tracking (`midlk12`, 12 px master mesh) is clean in every animal and epoch: whole videos covered, 0 % NaN
bins, 0 violations, folds inside the face in 0-0.34 % of sampled frames (`o2_face_rhythm.py qc`, all 108; **CORRECTED**
10-23 from "0-0.15 %", which was the 74-task figure before PS95: PS95 0.1-0.34 %, the rest <= 0.15 %). Rim folds (28-65 %)
are near-flat sliver triangles at the mesh edge and do not affect the factors. Median point displacement (px) pre ->
acute -> subacute -> chronic: PS92 1.33 / 1.11 / 1.16 / 1.18; PS93 1.19 / **0.80** / 1.21 / 1.36; PS94 2.06 / 1.33 /
2.35 / 2.63 (PS94 moves ~2x more or has a larger face in px; absorbed by the per-animal scale); PS95 1.65 / **0.96** /
1.49 / 2.06. Acute displacement drops below pre in PS93 and PS95 (and mildly PS92) -- the face moves less in the first
days -- then rises above pre chronically in PS94 / PS95.

### 1b. Bin timing: true times from per-frame timestamps (STANDS, 10-23)

The runner's LINEAR frame -> DAQ stamps were wrong on 2 of 108 cam4 tasks (`o2_face_rhythm.py timing`,
`<bundle>/timing_summary.csv`): PS92_0908 dropped 11.8 % of frames (57 % in its last tenth) -> session / points stamps
off by a median 3.2 s, up to 841 s; PS92_0812 v1's video ran ~39 min past its ~10 min of DAQ recording -> bins after
636 s stamped at DAQ times with no recording. All others: <= 20 ms. Now true times everywhere + lossy bins NaN
(PS92_0908: 5 % of session, 6 % of trial bins). Effect downstream (encoding window, FR run A rerun on the corrected
pilot projections): PS92_0908 FR-model cortex R2 **0.506 -> 0.618** (ENL 0.545 -> 0.663, cue 0.466 -> 0.573, first lick
0.413 -> 0.527; FR's unique contribution +0.26) -- the bad stamps had been diluting FR; PS92_0812 unchanged (its
mistimed bins lay outside the imaged frames).

## 2. How many factors (rank 12 chosen for the master, 10-24; was 10 on 10-23)

**MASTER (master_pre_v1, all four animals, 10-23; fit sample: FACE_RHYTHM_PIPELINE "Time windows and the fit sample"
-- 120 trials / animal = 20 per position, pre-cue rows (spout arrival U cue -4..-2 s), between rows half running /
half rest, true bin times; 27.4 k rows; 8 starts per rank, fit-only):**

| rank | EVR | starts at the best solution | stability (median) | per-factor worst match |
|---|---|---|---|---|
| 8 | 0.775 | 2 / 8 (two solutions, err .4742 / .4764) | 0.83 | 0.31-0.98 (one unstable factor, #4: 0.31) |
| 10 | 0.792 | **7 / 8** (err .4565) | **0.999** | 0.69-0.99 (only the one outlier start) |
| 12 | 0.803 | 3 / 8 | 0.91 | two weaker factors (.80 / .82 median): the extra factors split left and centre-right licking |

**Rank 10 chosen (Priya 10-23 evening)**: the only well-determined rank (rank 8 is not; rank 12 splits lick factors
less reproducibly for +0.011 EVR). Master EVR is a little lower than the pilot's at the same rank (more varied sample).
Next at rank 10: split-half (is 20 / position enough), leave-one-animal-out, leave-one-day-out.

**O2 rank-10 fit (10-23 ~20:25, equal-days sample): NO dedicated LEFT-lick factor (OPEN, Priya surprised).** EVR .792,
5 / 8 starts at the best solution (.4564; 3 at .4571), stability .998, leave-one-animal-out similarity .957-.984,
split half .990. L/R index (L-R)/(L+R) of mean 0-2 s expression, pre sessions, far / close: m5 lick_right -.35 to -.91
in all four animals (PS93 weakest), but the most left-leaning factor (m7, pilot lick_left_slow, spatial match .56) only
+.16 to +.45; the shared lick factors m3 / m4 / m10 lean +.1 to +.3. NOT the sample: exactly 20 trials per position per
animal (300 rows in 0-3 s per position per animal), and left trials are reconstructed as well as right (0-2 s rel err
within ~.03 across positions, same movement energy D). So rank 10 codes left licking as the DEFAULT lick and right
licking as default + m5. **Rank 12 (archived sweep, pooled sample, rows re-mapped to positions) HAS the left factor**:
r12 f8, 6.5 Hz, L/R +.52 to +.91 in all four animals (far and close), present in 6 / 7 other starts, mirror of r12 f11
= m5 (right, -.64 to -.85; PS93 weak). Its unstable factors are slow ones (f7 5.1 Hz 2 / 7 starts, f12 1.1 Hz 3 / 7).
Rank 8 already had a left-leaning factor (+.4 to +.66, not PS93).

**O2 refits at 11 / 12 (10-24) -> RANK 12 CHOSEN (Priya 10-24; supersedes rank 10).** Both have the left lick (6.5 Hz,
L/R +.40 to +.92 in all four animals, PS93 weakest; = the sweep's r12 f8, match .99 / 1.00 across two samples); found
in 6 / 8 starts at 12, only the best 2 / 8 at 11. Rank 12: EVR .803, stability .912, leave-one-animal-out factor
similarity .82-.90 (PS93 lowest); keeps every rank-10 factor (spatial >= .95, m7 .89), adds the left lick, splits m1
(4.1 / 1.3 Hz). Less reproducible: 5.8 Hz m7-like (4 / 8 starts), whisk 14.7 Hz (4 / 8, median .97), 2.9 Hz jaw (5 / 8).

**PILOT (2 animals; superseded for the rank choice by the master sweep).** Fit-only sweep on the PS92 + PS93 pre-stroke
fit rows (3 starts each): EVR / stability across starts -- rank 6 0.767 /
0.84; 8 0.792 / 0.96; 10 0.807 / 0.87; 12 0.819 / **0.997**; 14 0.827 / 0.89; 16 0.835 / 0.92. EVR rises slowly and
evenly (~+0.01 per 2 factors past rank 8; no elbow), so "enough" is a judgement, not a knee: ranks 8-12 capture what the
data offer cheaply, and rank 12 is the most reproducible. Stability is not monotonic (rank 12 more reproducible than
10 or 14 on this tensor). The pilot stays at rank 10 (Priya 10-22); rank 12 is the candidate to compare on the master.
"Stability" = similarity of the other random starts' factors to the best start's (matched, spatial x spectral cosine),
one number per rank from only 3 starts here (two comparisons each), so noisy. At rank 10 the three starts reach the same
error (0.4394-0.4401) with differently arranged factors; at rank 12 all three find the same solution.

**Rank 12 vs rank 10 (structure on the fit rows, not yet lick-validated).** No extra licking factor: the four
lick factors (f0, f4, f5, f9) plus f1, f3, f7 recur unchanged (spatial / spectral / timecourse 0.92-1.00). The two extra
factors go to SLOW movement: f8 (~2.3 Hz) splits by frequency into ~4.6 Hz and ~1.3 Hz components on the same face
region (not lick-locked; slightly running-linked); f6 sharpens to ~3 Hz; a new ~1.3 Hz component overlaps f6 / f2. Likely
why rank 12 is reproducible -- to confirm with per-factor stability on the master (from 10-22 `fit` saves every
start's factors, `stability_per_factor.csv`; default 8 starts). The cohort rank is chosen on
master_pre_v1 (four animals). Figure: `<bundle>/factors/pilot_pre_PS92_PS93/report/rank_sweep.png`.

## 3. The pre-stroke noise floor (STANDS for PS92 + PS93)

**MASTER rank 12, all four animals (10-24, `loso/rank12_<day>.json`, 31 pre days):** a pre-stroke day scored on the fit
that left it out explains .768-.820 (PS92 .795-.820, PS93 .768-.805, PS94 .784-.817, PS95 .796-.812); in fit .778-.822.
Leaving a day out costs at most .010 EVR (median ~.004; PS93_0814, the largest left-out day, the most) -> the master
is not fitted to any one day, and ~.80 is the pre-stroke level a post session is read against. Factor similarity of
the day-out refits to the full fit .87-1.00: the rank-12 start-to-start level (stability .91), not day-specific factors.

Leaving one pre-stroke DAY out of the fit costs <= 0.01 EVR on that day's session rows: PS92 held-out median 0.798
(0.788-0.815) vs in-fit 0.805, gap median 0.006 (max 0.009); PS93 0.805 (0.773-0.813) vs 0.804, gap 0.001 (max 0.004);
held-out factors match the full fit (similarity 0.84-1.00). Moving between ANIMALS costs ~0.10 EVR (fit one, project
the other: held-out error 0.52-0.54 vs 0.43-0.45 in fit) -- animals differ far more than days, so the master set needs
all four animals. Pre-stroke sessions are always scored on their held-out basis (DECISIONS 10-22 #2).

## 4a. What the MASTER factors are (master_pre_v1, rank 12; names 10-24, Priya + validation)

Names in `master_pre_v1/labels.json`; evidence = gallery (`report/factor_gallery_abs.png`) + validation
(`face_rhythm/cam4/validation/master_pre_v1/`: lick-triggered by direction / bout number / spout / lick frequency,
running onsets, rate correlations over 31 pre sessions). Grouping `fr_groups.master_v1`.

| # | Hz | name | evidence |
|---|---|---|---|
| 0 | 4.1 | sniff_prelick | nose, vertical; peaks ~0.4 s before the lick, 1st licks, far spouts; new vs pilot |
| 1 | 1.3 | nose_prelick_slow | nose, vertical; early in trial, before licking, far spouts higher |
| 2 | 6.5 | lick_ongoing_7Hz | peaks after the lick on 4th+ licks, sharply tuned to ~7 Hz lick frequency, all spouts |
| 3 | 1.3 | nose_steps | nose, lateral; falls during licking, rises late in trial and at running onset |
| 4 | 6.5 | lick_right | right spouts only; lick-rate r right .53 / left -.03 |
| 5 | 6.5 | lick_left | left spouts only; r left .52 / right -.03 (absent at rank 10) |
| 6 | 1.0 | run_onset_jaw | peaks just before running onset; split half .36 (least reproducible) |
| 7 | 8.2 | lick_fast_close | 2nd-3rd licks, >= 8 Hz, close spouts (Priya's name) |
| 8 | 5.8 | lick_far_nose | nose region, locked to 1st-3rd licks at FAR spouts (L ~ R); new vs pilot |
| 9 | 14.7 | whisk_run | running (speed r .55, .68 lick-free), also rises at each lick |
| 10 | 2.9 | prep_jaw | peaks ~0.4 s before the lick, 1st lick highest; also at running onset; split half .76 |
| 11 | 5.8 | lick_slow_late | peaks ~0.5 s into licking, 4th+ licks, ~5 Hz (pilot "bout_init" renamed: it peaks DURING licking) |

Lick group (2, 4, 5, 7, 8, 11) summed over 0-2 s is within +-10 % across the six spouts in PS92 / PS94 / PS95; PS93
far_L .79, close .13-.17 above (its licking, not the grouping). 7 (close) and 8 (far) carry spout DISTANCE.

## 4. What the pilot factors are (PILOT; names Priya's, checked against licks + treadmill)

**Master rank 10 vs this pilot (10-23, structure only; master factors not yet named):** matched one to one, mean
similarity 0.825. Close: f2 run_onset_jaw .99 / 1.00 (spatial / spectral), f6 prep_jaw .98 / 1.00, f8 nose_steps
.98 / .97, f9 lick_right .96 / .93 (6.5 Hz both), f3 nose_prelick .95 / 1.00, f1 bout_init .92 / .88, f4 .93 / .77,
f5 .89 / .79 (peak 7.3 -> 8.2 Hz), f7 whisk_run .88 / .93 (peak **11.7 -> 14.7 Hz**). Weak: **f0 lick_left_slow
-> spatial .51** (spectrum .88) -- left licking is mapped differently with four animals (or split with the fast early
factor). Names never carry over by number: the master factors get their own gallery / clips / validation.

Rank 10 on PS92 + PS93. Names from the clips (8 examples per factor, PS92_0806) refined by the identity check over all
15 pre-stroke sessions (~49 k LP licks, 1351 running onsets; `scripts/face_rhythm_validate.py`;
`<dlc>/face_rhythm/cam4/validation/pilot_pre_PS92_PS93/`):

| factor | Priya (clips) | licks / treadmill (15 pre sessions) |
|---|---|---|
| f0 | left licks, later | LEFT-selective (left-rate r 0.33, right 0.03); through the bout; strongest at SLOW rates (~5 Hz); close-L > far-L |
| f1 | right / central licks, later | BOUT INITIATION: peaks at the 1st-3rd licks, ~0.25 s before early licks; also at running onset; weak lick-rate link |
| f2 | sniff / whisk / jaw | sharp peak at RUNNING onset; rises with faster licking |
| f3 | vertical nose, around a (late) lick attempt | ~0.4 s BEFORE early licks and the 1st licks of a bout; rises after running onset |
| f4 | close-R / central licks, early (faster?) | CENTRE / RIGHT (r 0.50 / 0.32), ONGOING licks (4th+), SLOWER (peak 6 Hz); close-R > close-centre |
| f5 | left licks, early (faster?) | EARLY + FAST (7-8 Hz), builds over the first licks; left > centre > right; far-L highest |
| f6 | weak; jaw, preparatory? | PREPARATORY: ~0.35 s before early / first licks, all directions |
| f7 | nose / sniff / whisk while walking | strongest RUNNING factor (speed r 0.46 in trials, 0.62 outside licks) + a lick-initiation transient |
| f8 | nose wiggle with steps | running-linked (0.29), sustained after onset; no lick-direction link |
| f9 | right licking, early (faster) | RIGHT-selective (r 0.48, left -0.07), early AND late, ongoing licks, ~7 Hz; far-R > close-R |

Groups: lick INITIATION / preparation (f6, f3, f1, + f7's transient), ONGOING licking by side and tempo (f0 left slow,
f4 centre-right slow, f9 right, f5 fast early), MOVEMENT / whisking (f7, f2, f8). Caveat: absolute expression includes
the frame's overall movement, so every factor correlates positively with speed outside licks (0.03-0.62) -- read
speed relatively. Re-rendered 10-23 on the fr-timing-v1 projections (PS92_0812 v1 is one of the 15 pre sessions):
unchanged except speed outside licks (f7 0.58 -> 0.62).

**Names (Priya 10-22, `pilot_pre_PS92_PS93/labels.json`; this fit only -- never carried to another fit by number):**
f0 `lick_left_slow`, f1 `bout_init`, f2 `run_onset_jaw`, f3 `nose_prelick`, f4 `lick_centreR_ongoing`, f5
`lick_fast_early_Lbias` (left-leaning but the least side-specific lick factor: per-lick expression at right licks
0.47-0.72 of left vs f0 0.24-0.52), f6 `prep_jaw`, f7 `whisk_run`, f8 `nose_steps`, f9 `lick_right`. Loader / encoder
columns are `fr_<name>`. The encoding direction split keeps f5 in "left" for now (Priya).

## 5b-master. Per lick, by epoch, on the rank-12 master (STANDS, four animals, 10-24)

`scripts/face_rhythm_lick_epochs.py --set master_pre_v1` -> `face_rhythm/cam4/validation/master_pre_v1/lick_epochs/`.
Ratios to each animal's pre-stroke mean (right = contralesional):

| | PS92 | PS93 | PS94 | PS95 |
|---|---|---|---|---|
| right licks / trial, acute | .09 | .36 | .06 | .13 |
| right licks / trial, chronic | 1.01 | .97 | .52 | 1.41 |
| lick_right per right lick, acute | .60 | .73 | .51 | .28 |
| **lick_right per right lick, chronic** | **.57** | **.77** | **.79** | **.62** |
| lick_left per left lick, chronic | .67 | .97 | 1.23 | .59 |

The pilot's result (5b) replicates on four animals and on the master basis: the contralesional lick COUNT recovers (3 /
4; PS94 half), the contralesional lick factor PER LICK stays at .57-.79 chronically in all four -> the lasting deficit is
in how contralesional licks are made. Ipsilesional per-lick change is smaller and mixed in sign.

**Statistics (10-24 evening; `face_rhythm_lick_epochs.py --stats-only`, `lick_epochs/per_lick_stats.png`,
data/per_lick_stats.csv; log2 vs own pre, `boot_delta`; ** = also Holm over the 12 per-lick factors):** lick_right per
right lick acute -1.06 [-1.64, -.51]**, subacute -.44 [-.75, -.10]*, chronic -.57 [-.79, -.34]**, 4 / 4 below pre at
every epoch; right licks per trial acute -3.1*, subacute -1.1*, chronic -.14 [-.75, +.35] n.s. (count recovers);
lick_left per left lick n.s. at every epoch. STANDS with statistics.

## 5. Changes after the stroke (PILOT, two animals; read per animal)

Response-window (0..+4 s) absolute expression, per animal x epoch, relative to that animal's pre-stroke mean (session
means first; n sessions pre / acute / subacute / chronic: PS92 8 / 5 / 2 / 12, PS93 7 / 4 / 3 / 9):

| factor | PS92 acute / sub / chronic | PS93 acute / sub / chronic | reading |
|---|---|---|---|
| **f9 right (contralesional) licking** | **0.45 / 0.82 / 0.65** | **0.33 / 0.48 / 0.46** | down in BOTH animals, still down chronically |
| f4 centre-right ongoing licking | 0.58 / 0.90 / 1.21 | 0.50 / 1.26 / 2.00 | down acutely in both, above pre chronically |
| f3 nose before licks | 0.67 / 0.95 / 0.83 | 0.39 / 0.81 / 0.79 | down acutely in both |
| f6 preparatory | 1.40 / 1.36 / 1.32 | 1.40 / 1.31 / 1.26 | UP in every post epoch, both animals |
| f0 left (ipsilesional) licking | 0.75 / 0.81 / 0.84 | 0.92 / 0.90 / 0.64 | mildly down |
| f5 fast early licking | 0.36 / 0.70 / 0.75 | 0.76 / 0.98 / 1.47 | down acutely; animals diverge later |
| f1 bout initiation | 1.39 / 1.60 / 1.43 | 0.99 / 1.18 / 1.12 | up in PS92 only |

Re-checked 10-23 on the fr-timing-v1 projections (PS92_0908, PS92_0812 v1): unchanged to two decimals.

Reading: contralesional (right) licking expression falls and stays low in both animals, while preparatory movement
before licks rises in both -- consistent with more preparation / attempts and fewer executed contralesional licks.
CAVEATS: two animals; absolute expression scales with how much the animal licks, so part of a drop is FEWER licks,
not a different movement per lick -- next: per-lick expression (lick-triggered, as in section 4) by epoch, and the
master set's four animals. Figures: `report/factor_epochs_abs.png` (cue-aligned, pre max = 1).

### 5b. Fewer licks or different licks? (PILOT, two animals)

`scripts/face_rhythm_lick_epochs.py` (`<dlc>/face_rhythm/cam4/validation/pilot_pre_PS92_PS93/lick_epochs/`): per
session and executed lick DIRECTION, response-window licks per engaged trial and each factor's expression AT the lick
(0..+0.2 s after onset); per animal x epoch, relative to that animal's pre. f9 (right / contralesional licking):

| | right licks / trial (acute / sub / chronic) | f9 per right lick | f9 total |
|---|---|---|---|
| PS92 | **0.09** / 0.40 / **1.01** | 0.59 / 0.62 / **0.60** | 0.46 / 0.84 / 0.66 |
| PS93 | **0.36** / 0.80 / **0.97** | 0.36 / 0.49 / **0.40** | 0.33 / 0.48 / 0.46 |

Reading: ACUTELY both -- far fewer contralesional licks (PS92 457 right licks in 2091 trials) and each one expresses f9
less. CHRONICALLY the number of right licks is back at pre in both animals but f9 per right lick stays at 0.4-0.6:
the persistent deficit is in HOW contralesional licks are made, not how many. Other per-lick changes: f6 (preparatory)
per lick UP 1.2-1.8 in every post epoch, both animals and all directions (more preparation per lick); f1 (bout
initiation) per lick up in PS92 (acute ~3x); f4 per centre lick down acutely, above pre chronically; f0 per left lick
down chronically (PS92 0.79, PS93 0.58) -- part of the per-lick drop is not side-specific, but contralesional f9 drops
most. CAVEATS: direction is the executed angle (+-12 deg): smaller post-stroke right licks may be classed centre;
absolute expression includes the frame's whole-face movement scale.

### 5c. "Licks and misses" on PS92-95 (behaviour; the AR-HMM cohort's result, 10-23)

The AR-HMM work on PS46-50 (github.com/htong1/Priya_ARHMM2026) found the contralesional deficit splits into "can't
lick" vs "licks and misses" (contra miss-rate 25 % -> 40-90 % acute, ~45-50 % plateau by day 14; ipsi flat) -- a
post-hoc contact split, not the model itself. Same check here (`scripts/lick_contact_by_epoch.py`; August pre):

| | far-contra attempt rate (pre / acute / sub / chronic) | far-contra trial completion | per-LICK no-contact, contra spouts | ipsi spouts |
|---|---|---|---|---|
| PS92 | .97 / .79 / .93 / 1.00 | .98 / **.05** / .71 / .99 | .08 / .24 / .19 / **.14** | .08 / .06 / .19 / .09 |
| PS93 | .99 / .80 / .97 / 1.00 | .97 / **.15** / .72 / .99 | .19 / .24 / .20 / .15 | .35 / .27 / .28 / .26 |
| PS94 | .97 / .64 / .94 / .99 | .97 / **.04** / .86 / .95 | .09 / .33 / .27 / **.18** | .05 / .06 / .04 / .03 |
| PS95 | .96 / .32 / .87 / .99 | .98 / **.03** / .93 / .99 | .09 / .14 / .15 / **.13** | .10 / .09 / .10 / .10 |

Reading: REPLICATES acutely -- far-contra trials are attempted (64-80 %, PS95 32 %) but almost never completed
(.03-.15), close contra and ipsi spared; "try and fail" (PS92 / 93 / 94) vs "fail to try" (PS95) as in the AR-HMM
cohort. CHRONICALLY trial completion is back at pre (.95-.99, unlike the AR-HMM cohort's plateau), but per-lick misses
on contra spouts stay mildly elevated in 3 / 4 (PS92 .14 vs .08, PS94 .18 vs .09, PS95 .13 vs .09; PS93 not) while
ipsi is flat -- with 5b (f9 per right lick 0.4-0.6 chronically): animals get the water again, with less accurate and
differently made contralesional licks. Per-lick numbers pool close + far contra (far alone will be larger); LP lick
detection post-stroke is the caveat (cam1 rescues hidden tongues, `trial_attempts`).

## 6. Does the pre-stroke vocabulary still fit after the stroke? (PILOT, two animals)

EVR GAP = a session's own rank-10 fit (`fitsession`, 1 start) minus the frozen pre-stroke factors on the same session
rows; larger = the session's movement fits the pre-stroke factors less well. Pre sessions are scored on their
left-out-day basis, so the pre gaps are the noise floor (`face_rhythm_readouts.py --set pilot_pre_PS92_PS93`,
`<dlc>/face_rhythm/cam4/readouts/pilot_pre_PS92_PS93/`, `recovery_by_day.png`, `data/session_table.csv`):

| | pre (range) | acute d1-5 | subacute d5-9 | chronic d11-50 |
|---|---|---|---|---|
| PS92 | 0.028-0.044 (median 0.030) | 0.046-0.080, **5 / 5 above pre max** | 0.037-0.043, 0 / 2 | 0.036-0.052, **7 / 12 above**, rising late (d36-50 0.042-0.052) |
| PS93 | 0.026-0.043 (median 0.031) | 0.048-0.070, **4 / 4 above** | 0.033-0.047, 1 / 3 | 0.029-0.039, 0 / 9 (back at pre) |

Reading: acutely BOTH animals move in ways the pre-stroke factors capture less well (gap ~2x pre); PS93 returns to the
pre range by the subacute epoch, PS92 stays slightly above it chronically. Trial reconstruction error moves the same way
(acute 0.467 / 0.468 vs pre 0.444 / 0.443). Movement asymmetry ((contra - ipsi) / (contra + ipsi), response window):
PS93 acute -0.10 vs pre -0.03 (less contralesional-side movement), PS92 no clear change. The readouts' automatic
`labels_suggested.json` names are heuristics -- ignore them; names are Priya's (section 4). Their `lick_rate_corr` is SHARE-based
(normalised weights, whole 5 Hz session incl. between trials, raw lick counts): slow factors lose share during licking
and come out negative (f2 -0.24) although their ABSOLUTE expression rises a little (validation + the encoding window's
check: f2 tonically elevated during licking, not lick-locked, lick-triggered response 0.12-0.20 SD vs f4 0.62-1.39).
Re-checked 10-23 on the fr-timing-v1 projections of PS92_0908 / PS92_0812 v1: the table above is unchanged except one
PS92 chronic gap (.042 -> .040).
Encoding grouping (Priya-approved, `wfield_local/fr_groups.py`, pilot set only): lick = f0 + f4 + f5 + f9 (left f0 + f5,
centre f4, right f9); other = f1 f2 f3 f6 f7 f8 (includes the lick-initiation factors).

## 5d. Peak factor amplitude over recovery; contralesional trials recruit the IPSILESIONAL lick pattern (STANDS, 4 / 4, 10-24)

`scripts/face_rhythm_peak.py [--split side]` (`readouts/master_pre_v1/peak/`: peak_by_day[_side].png, data/peak_*.csv):
per session and frozen factor, the PEAK of the trial-averaged absolute expression over cue 0..+4 s, / the animal's pre
mean (the 2pRAM manuscript's FR panel: pre-stroke components refit to post sessions, amplitude over recovery).

| peak / pre | PS92 | PS93 | PS94 | PS95 |
|---|---|---|---|---|
| lick_right, all trials: acute / chronic | .45 / .65 | .46 / .81 | .39 / .61 | .30 / .77 |
| lick_right, CONTRA-spout trials: acute / chronic | .41 / **.57** | .43 / **.73** | .42 / **.76** | .33 / **.74** |
| lick_left, CONTRA-spout trials: acute / chronic | .78 / **2.30** | 1.21 / **1.20** | 2.30 / **3.28** | 1.57 / **1.54** |
| lick_left, ipsi-spout trials: chronic | .69 | 1.10 | .96 | .78 |

* The contralesional lick factor falls acutely and stays reduced chronically in all four (with 5b-master's per-lick
  .57-.79: the same deficit by two measures).
* On CONTRALESIONAL-spout trials the IPSILESIONAL lick factor rises above pre in all four (1.2-3.3x chronically), not on
  ipsilesional-spout trials: the animals reach the contralesional spouts with more of the ipsilesional lick pattern --
  a substitution / compensation in how the pre-stroke repertoire is MIXED, not a new movement (push check 6b, EVR gap
  back to pre in 2 / 4) and not a reshaped factor (6c: shape changes inconsistent across animals).
* Survey statistics (10-24 evening, `peak_cohort_by_epoch*.png`, ** = Holm over the 12 factors of a trial group x
  epoch): all trials acute lick_right -1.37**, lick_ongoing_7Hz -1.22**, lick_fast_close -1.01**, nose_prelick_slow
  -.78**, sniff_prelick -.68**; chronic lick_right -.51**, and ABOVE pre lick_fast_close +.72**, lick_ongoing_7Hz
  +.50**, lick_far_nose +.42**, whisk_run +.42**, lick_slow_late +.39**, sniff_prelick +.26**.
* Shared lick factors fall acutely in all four (lick_ongoing .38-.54, lick_fast_close .29-.85) and recover;
  lick_fast_close overshoots chronically (1.2-2.1).
Caveat: a trial-average peak also falls with fewer licks per trial (acute contralesional licks are rare); the chronic
values are where lick counts are back (5b-master).

**5d statistics (cohort, `peak/data/peak_epoch_stats_side.csv`, figure `peak/peak_side_headline.png`):** change of
peak / own pre vs pre, `analysis_kit.boot_delta` (paired animals -> sessions, animal-weighted, 95 % interval):
contra-spout lick_right acute -.60 [-.67, -.52], subacute -.39 [-.46, -.31], chronic -.30 [-.40, -.21], 4 / 4 below
pre at every epoch; contra-spout lick_left subacute +.94 [.34, 1.54], chronic +1.08 [.35, 1.87], 4 / 4 above pre;
ipsi-spout lick_left n.s. at every epoch, ipsi-spout lick_right only acute (-.45). Cohort views: peak_cohort_by_day /
by_epoch (+ _side).

**5e. The same substitution in raw face movement, per lick (STANDS, 4 / 4, 10-24).** `scripts/face_rhythm_asym_aligned.py`
(`readouts/master_pre_v1/asym_aligned/`): per 25 Hz frame, mean point speed on the contra- vs ipsilesional half of the
face (reconstructed from the frozen point PCs; independent of the factors), LICK-aligned (-0.1..+0.2 s, LP licks in
engaged trials, PER LICK so lick count drops out), per spout position. Pre-stroke a lick at the far contralesional
spout moves the contralesional face more (far_R asym +.26 / .00 / +.22 / +.27 for PS92 / 93 / 94 / 95); acutely
+.05 / -.11 / -.05 / -.09, chronically +.14 / -.03 / -.05 / +.16 -- below pre in all four: contralesional licks are made
with a more ipsilesional face movement. Cue-aligned per-position speed maps (`position_maps/`, response window) agree:
the contralesional face loses more movement than the ipsilesional mainly on contralesional-spout trials (far_R contra -
ipsi log2 change acute -.17 to -.62 in 4 / 4; chronic PS92 -.25, PS94 -.53, PS95 -.10, PS93 recovered).

**5e speed maps with statistics (10-24 late; Priya).** `face_rhythm_position_maps.py` (`position_maps/
position_maps_response_mean.png`, data/position_maps_response_{points,halves}.csv): the MOVEMENT SPEED of each face
point (RMS px / s; not reaction time) in the response window (cue 0..+4 s) at each spout, per session log2 vs the
animal's pre; per point `boot_delta_columns` + >= 3 / 4 agree + BH FDR < .05 (rings); halves contra - ipsi with Holm
over 6 positions. Acute: whole face lower at most spouts (4 / 4 at far_L, far_center, close_R, far_R; far_R -.47).
Contra - ipsi at the R spouts: close_R -.27** / -.16** / -.19** (acute / subacute / chronic), far_R -.47** / -.29** /
-.22*; also close_L acute -.14**. Chronic: the face moves MORE than pre at the close and centre spouts, mostly the
ipsilesional side (close_R +.22, close_center +.23, close_L +.21; marked points ipsilesional, all "up"). Rows of every
per-spout figure now in the lab order close ipsi / centre / contra, far ipsi / centre / contra
(`face_rhythm.common.POSITIONS_DISPLAY`).

**5e statistics (10-24 evening; `asym_aligned/asym_aligned_change.png`, data/asym_aligned_stats.csv):** per session
the window asymmetry minus the animal's pre mean, `boot_delta`, Holm over the 6 positions of an alignment x epoch. The
asymmetry FALLS at both contralesional spouts, every alignment, every epoch, ** after Holm (bout-aligned chronic: far_R
* only, close_R n.s.): lick-aligned far_R acute -.24 [-.33, -.15], subacute -.16, chronic -.14 [-.23, -.05]; close_R
acute -.12, subacute -.08, chronic -.09 [-.14, -.04]; cue-aligned far_R -.15 / -.12 / -.10, close_R -.11 / -.07 / -.08
(all 4 / 4 lower). Smaller marks elsewhere (close_center cue-aligned chronic -.06**, far_center bout acute -.05**,
close_L lick-aligned chronic +.05**), not the claim. STANDS with statistics.

**5f per-point maps (10-24 late; Priya: no large square bins).** `face_rhythm_aligned_maps.py` now tests every mesh
point (`--unit point`, default; `--unit patch` = the earlier bins): `analysis_kit.boot_delta_columns` (the same paired
draw for all 1,282 points), marked when the interval excludes 0, >= 3 / 4 animals agree AND Benjamini-Hochberg FDR < .05
over the face (agreement threshold >= 3 of 4 animals since 10-24 late, Priya). Lick-aligned far_R: the contralesional lower face (jaw / cheek) moves less and the ipsilesional face
more per lick at every epoch (marked regions); close_R: the ipsilesional cheek gains (subacute / chronic). Marked points
summed over positions (>= 3 / 4): cue 5,282 / 1,994 / 3,013, lick 1,873 / 751 / 1,499, bout 907 / 992 / 0 (acute /
subacute / chronic). Halves statistics now draw from their own seeded stream (unchanged point estimates; interval ends move <=
.014; no mark changed).

**5f. Face-surface movement per trial vs per lick; the contralesional asymmetry is per lick (STANDS, 4 / 4, 10-24; first bullet CORRECTED 10-24).**
`scripts/face_rhythm_aligned_maps.py` (`readouts/master_pre_v1/aligned_maps/`: aligned_maps_{cue,lick,bout}.png with
significant face patches outlined, halves_{cue,lick,bout}.png, data/patch_stats.csv, halves_stats.csv). Per mesh
point, mean speed in a window (cue 0..+2 s; every lick -0.1..+0.2 s; bout onset 0..+0.5 s) per spout position, log2 vs
the animal's pre; patches = 4 columns per face half (mirrored) x 7 rows (48 with >= 10 points); `boot_delta` per
patch, marked when the 95 % interval excludes 0 AND 4 / 4 animals share the sign (a consistency map; many tests).

* Whole-face MESH speed falls acutely PER TRIAL (cue-aligned: far_R -1.15 log2, far_center -.74, 4 / 4) but not in the
  -0.1..+0.2 s window around each lick (lick-aligned whole face ~0 at every position). **CORRECTED 10-24 (Priya): this
  does NOT mean "fewer licks, not smaller licks".** The DLC / LP kinematics say acute licks ARE different: slower (ILI
  +14 to +26 ms at every position), shorter protrusion / reach at the far spouts, later first lick (+488 ms far contra)
  (OROFACIAL_FINDINGS 1, 8c). The face mesh tracks the cheek / jaw / nose surface, not the tongue, and a short per-lick
  speed window is insensitive to tongue extent and to lick rate. So the acute per-trial drop = fewer, later AND slower /
  shorter licks; the per-lick face-surface speed is simply not where the per-lick deficit shows. The contra / ipsi
  SPLIT below is unaffected (it compares halves of the same window).
* Contra-half minus ipsi-half change, per lick: far_R acute -.68 [-.97, -.38], chronic -.38 [-.66, -.13]; close_R acute
  -.31 [-.43, -.18], chronic -.25 [-.42, -.11]; 4 / 4 at every epoch; n.s. at the L and centre spouts. Bout onsets:
  far_R significant at every epoch. Licks to the contralesional spouts are made with relatively less contralesional face
  movement -- the raw-movement counterpart of 5d / 5e.
* close_R asymmetry > 0 BEFORE bout onset (Priya 10-24) -- ANSWERED, with statistics added 10-24 evening (pre-stroke
  LEVEL of the pre-onset asymmetry, nested bootstrap `boot_ci`, data/bout_split_pre_window_stats.csv): close_R carry
  +.15 [.10, .20]*, first-late +.12 [.10, .15]*, first +.06 [.02, .10]*, but lick-free lead-in +.03 [-.01, +.06] n.s. --
  so the "sustained offset" (3) is NOT significant, and the close_L mirror is a trend only (every close_L interval
  includes 0). Original reading, three parts (`scripts/face_rhythm_bout_split.py`,
  `asym_aligned/bout_split/`: bout_split.png, bout_split_pre_window.png, bout_split_windows.csv; pre-onset window
  -0.5..-0.1 s, per animal pre): (1) CARRY-OVER from earlier licks: bouts with a lick < 0.5 s before onset +.10 to +.20
  (4 / 4); (2) PREPARATION after the cue: first bouts starting >= 0.6 s after the cue (window entirely post-cue, pre-lick)
  +.10 to +.15; (3) a smaller SUSTAINED close_R offset with no lick in the preceding 0.5 s (+.02 to +.06; flat from -1 s,
  so partly before the cue). Mirror image at close_L (carry / first_late -.12 to -.21): the pre-onset asymmetry points
  toward the TARGET side, i.e. directed movement, not a close_R artefact. Same in every epoch; since 5f's statistics are
  changes vs each animal's own pre, a stable offset cancels there.

**5g. WHEN each factor is expressed after the cue: the contralesional lick factor is re-timed LATER (STANDS, 4 / 4,
10-24 evening).** Priya: "should we also look at when each behaviour is expressed with respect to cue and whether that
changes?" `scripts/face_rhythm_peak.py` (`peak/timing_by_epoch_{com,t_half}[_side,_position].png`,
data/timing_epoch_stats*.csv): per session the trial average's CENTRE OF MASS above its pre-cue level over cue 0..+4 s
(and the half-rise time), change vs pre in seconds by `boot_delta`; reference row = median first-lick latency of the
same trials.

* Statistics (10-24 evening): * = 95 % interval excludes 0; ** = also Holm over the 12 factors of a trial group x
  epoch. All trials: lick_right subacute +.31 s**, chronic +.35 s**; lick_fast_close acute +.41 s**; nothing else
  survives Holm (prep_jaw subacute +.44 s *).
* lick_right on CONTRALESIONAL-spout trials: subacute +.40 s [.19, .62], chronic +.41 s [.23, .59], 4 / 4 later; per
  position chronic close_R +.51 [.22, .75], far_R +.27 [.12, .43], 4 / 4. Over the same trials the FIRST LICK comes
  EARLIER chronically (contra -.06 s [-.11, -.01]; far_R -.04, close_R -.04; 0 / 4 later). So the shift is not reaction
  time: the contralesional lick movement is concentrated later in the response window (later in the bout), not delayed
  at its start. Acutely far_R lick_right is later too (+.80 s) but there the first lick is also late (+.48 s, 4 / 4).
* lick_left (ipsilesional factor) on contralesional-spout trials: acute far_R +1.26 s (4 / 4) -- the substitute
  movement appears late in the trial acutely; not consistent later.
* lick_fast_close all trials: acute +.41 s [.18, .73], 4 / 4; back by subacute. Acute first lick +.08 s.
* Other factors: no change consistent across animals (nose / sniff factors have wide pre ranges).
Caveat: a centre of mass of a trial average mixes timing within a trial with WHICH trials have the movement; a per-lick /
per-bout timing (lick number within the bout at which lick_right is expressed) would separate the two.

Figures (10-24 evening): animals `config.animal_color()`; epochs the orofacial / 3-D kinematics line palette
(pre black, acute pink, subacute gold, chronic teal; `face_rhythm.common.epoch_colors`, Priya: easier than the greys);
* / ** marks as defined in DECISIONS 2026-10-24 "FaceRhythm figures: epoch colours and significance marks".

## 6-master. Does the pre-stroke vocabulary still fit? Four animals, rank 12 (10-24)

`scripts/face_rhythm_readouts.py --set master_pre_v1` (`face_rhythm/cam4/readouts/master_pre_v1/`: session_table,
epoch_means, evr_own_vs_frozen.png, recovery_by_day.png). Own fits = 8 random starts (`own8/`; vs 3 starts: session EVR
up in 19 / 108 sessions by <= .0013, gap unchanged to 3 decimals). Pre sessions scored on their held-out day basis.

| EVR gap (own - frozen) | PS92 | PS93 | PS94 | PS95 |
|---|---|---|---|---|
| pre | .035 | .043 | .031 | .030 |
| acute | **.051** | **.077** | **.058** | **.060** (1 session) |
| chronic | **.047** | .042 | **.044** | .033 |

Acute rise in 4 / 4 (reconstruction error too, +.016 to +.033); chronically back to pre in PS93 / PS95, still up in
PS92 / PS94 (reconstruction error: only PS94, +.013). Replicates the pilot (6: PS93 back, PS92 elevated) on four
animals. STANDS (per animal; no pooled test at n = 4).

**Normalised and pooled by epoch (Priya 10-24; `scripts/face_rhythm_evr_epochs.py` -> `readouts/master_pre_v1/
evr_epochs.png`, data/evr_epoch_stats.csv; per-day panel `evr_own_vs_frozen.png` bottom row).** Own-fit EVR itself dips
acutely (.822-.831 vs .833-.840 pre: less movement against a fixed tracking-noise floor, less stereotyped movement),
so the RATIO EVR frozen / EVR own, / each animal's pre mean (pre = 1), divides that out. Per animal, sessions pooled
within the epoch; change vs pre by `boot_delta` (paired animals -> sessions, 95 % interval):

| ratio / pre | PS92 | PS93 | PS94 | PS95 | change [95 % CI] | worse |
|---|---|---|---|---|---|---|
| acute | .979 | .954 | .965 | .963 | **-.035 [-.045, -.025]** | 4 / 4 |
| subacute | .997 | .991 | .989 | .992 | -.008 [-.014, -.003] | 4 / 4 |
| chronic | .986 | 1.001 | .983 | .997 | -.008 [-.016, -.0003] | 3 / 4 |

EVR gap (- own pre): acute +.027 [.019, .035] 4 / 4, chronic +.007 [.0003, .013] 3 / 4; frozen reconstruction error /
pre: acute +7.5 % [5.9, 9.4] 4 / 4, chronic n.s. -> the pre-stroke vocabulary fits clearly worse ACUTELY in all four;
chronically a small residual carried by PS92 / PS94 (interval only just excludes 0), PS93 fully back.

## 6c. Factor "morphing": which factors change shape, per animal (10-24; NOT consistent across animals -- secondary)

**6c-pooled (10-24 late; the cleaner test Priya asked for).** `scripts/face_rhythm_epoch_shape.py` on
`own_epoch/` (O2 `fitepoch`: one fit per animal x epoch on the epoch's sessions pooled, 11,000 rows each; two pre fits
on disjoint alternating halves of the pre days = the FLOOR) -> `readouts/master_pre_v1/epoch_shape/`
(epoch_shape_angles.png, epoch_shape_asymmetry.png, data/epoch_shape.csv). Pooling makes the fits stable: the pre-half
floor is 4-8 deg spatial for the lick factors (per-session own fits: 10-50 deg).
* The LICK factors change shape beyond the floor in 4 / 4 animals at every epoch (spatial angle to BOTH pre halves >
  floor): lick_right 11-46 deg vs floor 4-8, lick_left 12-60 vs 7-8, lick_ongoing_7Hz 15-42 vs 5-11; spectrally
  lick_right 4 / 4 / 3 and lick_left 4 / 3 / 3 (acute / subacute / chronic). Weak factors do not (run_onset_jaw 1 / 0 /
  0, prep_jaw 2 / 2 / 2).
* DIRECTION (loading asymmetry, floor = |preA - preB| <= .05): lick_right (pre +.54 to +.87, contralesional face)
  moves OFF the contralesional face -- chronic -.13 / -.08 / -.13 in PS92 / PS94 / PS95, acute -.14 to -.34 -- and
  lick_left (pre -.57 to -.76, ipsilesional) becomes LESS ipsilesional -- chronic +.27 / +.29 / +.28. PS93 goes the
  other way on both (lick_right +.05 to +.23, lick_left -.09 to -.12). The INDEX moves toward symmetric in 3 / 4.
  Priya 10-24: the contra - ipsi index is too coarse to read -- it collapses the loading map to one number, a factor is
  fit across ALL spouts (so it is not per position; position-resolved evidence is the raw movement, 5e / 5f: R spouts
  only), and a shift toward symmetric can mean (1) less contralesional recruitment, (2) licks aimed differently (more
  central), or (3) a different mix of licks in the post fit. Separated (10-24 late) with loading DIFFERENCE MAPS
  (`epoch_shape_loading_<factor>.png`: post minus pre halves per point, vs preA - preB) and SPOUT TEMPLATES
  (`epoch_shape_templates.png`: r between the loading and each spout's pre-stroke spout-specific movement pattern):
  - WHERE: in PS92 / PS94 / PS95 lick_right's loading leaves the contralesional face near the midline and concentrates
    at the mouth / chin rim and the ipsilesional lateral cheek, at every epoch, far above the pre-half difference
    (which is nearly blank). PS93 differs (acute: gains contralesional, loses chin).
  - WHICH SPOUT: the post-stroke lick_right still looks most like the R spouts and least like the L spouts (the same
    ranking as pre), but LESS spout-specific -- correlations shrink toward 0 (PS95 far_R .82 -> .53 / .55 / .59, far_L
    -.69 -> -.24 / -.33 / -.46; PS94 far_R .58 -> .39 / .45 / .45). It does NOT turn into another spout's pattern, with
    one exception: PS92 subacute (close_center .09 -> .50, far_R .71 -> .14; back by chronic). lick_left the same way
    (PS95 subacute loses its pattern entirely).
  - Reading: not a re-aimed lick (2) in general; the lick factors become less side-specific -- consistent with less
    contralesional recruitment (1) or a blend of more lick types in the post fit (3), which shape alone cannot separate.
    The position-resolved, fit-free evidence for (1) is the per-lick raw movement (5e / 5f): at the R spouts each lick
    moves the contralesional lower face less and the ipsilesional face more.
* Caveats: one value per animal x epoch, so descriptive (k / 4, no interval). A post epoch's fit also sees a different
  MIX of behaviour (fewer contralesional licks acutely), which can reshape a factor without any single movement
  changing; the alternating-day floor excludes slow drift across days (an early-vs-late split of the pre days would be
  the stricter floor: 8 more O2 tasks). PS95 acute = one session, half the rows.
Status: PROVISIONAL (descriptive; strong per-animal consistency for the lick factors).

`scripts/face_rhythm_match.py` (`readouts/master_pre_v1/matching/`: match_by_day.png, match_example.png,
asymmetry_two_ways.png, data/match_*.csv). Each session's own fit Hungarian-matched to the master (spatial x spectral
cosine); angle = arccos. The baseline is like for like: each post session's own-vs-master angle against the RANGE of
the animal's pre sessions' own-vs-master angles (not the pre-pre floor, which mixes two fits' noise). Chronic sessions
beyond the pre range:

| | PS92 | PS93 | PS94 | PS95 |
|---|---|---|---|---|
| lick_right spatial angle | **12 / 12** | 0 / 12 | **8 / 8** | 5 / 11 |
| lick_right loading asymmetry | 4 / 12 | 2 / 12 | 3 / 8 | **11 / 11** |
| lick_left spatial angle | 2 / 12 | **11 / 12** | **8 / 8** | 3 / 11 |
| lick_left spectral angle | 1 / 12 | 10 / 12 | **8 / 8** | 5 / 11 |
| lick_left loading asymmetry | 1 / 12 | **10 / 12** | **8 / 8** | 4 / 11 |

* Both lick factors change shape chronically, in DIFFERENT animals (no clean left / right dissociation; an earlier
  reading against the pre-pre floor suggested one -- CORRECTED 10-24). PS94 changes on both sides.
* Direction (own-fit loading asymmetry, (contra - ipsi) / sum of |loading|, `face_rhythm_matching.loading_asymmetry`):
  lick_right moves OFF the contralesional face (pre +.50 to +.76 -> acute +.37 to +.63, chronic +.47 to +.64; strongest
  PS95, all 11 chronic sessions below its pre range); lick_left becomes LESS ipsilesional (pre -.55 to -.66 -> chronic
  -.31 to -.50; PS93, PS94).
* Unreadable: lick_far_nose (pre-pre floor at chance), PS93's nose / sniff factors (at chance even pre: not in PS93's
  own fits), PS93 lick_right (pre range 24-52 deg). Acute angles partly reflect few contralesional licks to fit.
* Split / merge flags: pre .18, acute .27, chronic .22.
Priya 10-24: "the cosine similarity doesn't show clear consistent changes post stroke" -- agreed: per-session own
fits are noisy (pre angles 10-50 deg) and changes differ by animal. LEAD with frozen-basis amplitudes (5b-master, 5d)
and the EVR gap; report shape as heterogeneous. A cleaner shape test if wanted: one own fit per animal x epoch
(pooled sessions). Amplitude per lick (5b-master) is the 4 / 4 result; shape changes are animal-specific (the 2pRAM manuscript's
"heterogeneous reorganization", here on one basis across animals).

## 6d. Task vs non-task, asymmetry, severity (10-24)

* Task (lick factors, response window) vs non-task (other factors, between trials), vs pre: acute task .66-.80, non-task
  .58-.83 -- non-task is NOT preserved acutely (drops as much or more in PS93 / PS94); both >= pre chronically. The
  2pRAM manuscript's "relative preservation of non-task movements" does not replicate here. Caveat checked 10-24
  (`scripts/face_rhythm_nontask_state.py`, `readouts/master_pre_v1/nontask_state/`: between-trial bins classed by the
  DAQ behaviour events into running / rest / other): NOT explained by state occupancy -- running time does not fall
  consistently acutely (PS92 .05 -> .11, PS94 .07 -> .11, PS93 ~flat, PS95 .06 -> .005), and non-task expression falls
  WITHIN state: rest acute -.99 log2 [-1.24, -.80] 4 / 4, "other" (85-99 % of between-trial time) -.37 [-.63, -.17]
  4 / 4, running -.28 (3 animals, n.s.); back to pre chronically. (The events carry no grooming bouts for these
  sessions, so grooming is inside "other".) STANDS: non-task face movement is reduced acutely, not preserved.
* Face movement asymmetry (tracked points, response window, (contra - ipsi) / sum): acutely less contralesional movement
  in PS93 / PS94 / PS95 (-.08 to -.11), PS92 ~0; chronically only PS94 (-.07). Agrees with the loading asymmetry.
* Severity vs lesion burden (histology, ipsi / contra CP area: PS93 45 %, PS92 47 %, PS94 68 %, PS95 82 %): no ordering
  -- largest acute task drop in the mildest (PS95), chronic EVR-gap elevation in PS92 (severe) and PS94 (intermediate),
  not PS93 (severe). Descriptive, n = 4.

## 6b. Treadmill pushes are inside the pre-stroke face repertoire (STANDS, four animals, 10-24)

Question (Priya 10-24): do we need a pre + post master to capture the encoder's postural / forelimb PUSHES
(ENCODING_FINDINGS 4.4: 2-5x more frequent after the stroke)? `scripts/face_rhythm_push_check.py` (all 107 sessions;
`face_rhythm/cam4/readouts/master_pre_v1/push_check/`): around every push onset (backward treadmill speed > 0.3 mm/s,
>= 2 samples), the frozen master's reconstruction error (pre sessions on their held-out day basis) and the face's
movement energy D, against random times of the same context (in-trial cue -1..+4 s / between trials), >= 2 s from any
push; (0..1 s) - (-1.5..-0.5 s), push minus random.

* The face MOVES at pushes: D rises +1.3 to +5.5 above random times in every animal x epoch x context.
* The frozen pre-stroke basis describes that movement BETTER than average: reconstruction-error excess is NEGATIVE in
  27 / 32 animal x epoch x context cells, every chronic cell included (chronic -.005 to -.041); the five positive cells
  are small (<= +.007; PS93 in-trial pre / subacute, PS94 between acute, PS95 acute = 1 session).
* Pushes are 2-4x more frequent chronically (between trials PS92 .87 -> 2.63 / min, PS93 .69 -> 3.01, PS94 3.9 -> 4.4,
  PS95 .98 -> 2.19), yet no animal's error rises at them.

-> No pre + post master for pushes (Priya's question answered by the check, DECISIONS 10-24): the face movement that
accompanies a push is already in the pre-stroke repertoire; pushes stay a treadmill regressor for the encoder.

## Where to look

* Pilot factor set: `<dlc>/../o2/fr_cam4_20261021/factors/pilot_pre_PS92_PS93/` (`report/` gallery, epochs, rank
  sweep; `labels.json`; `proj/`, `own/`, `loso/`); validation `<dlc>/face_rhythm/cam4/validation/pilot_pre_PS92_PS93/`
  (+ `lick_epochs/`); readouts `<dlc>/face_rhythm/cam4/readouts/pilot_pre_PS92_PS93/`.
* Master set (in progress): `.../factors/master_pre_v1/` (`plan.json`, `rank*.json` incl. per-factor stability,
  `logs/run_master_local.py`).
* QC / timing: `<dlc>/face_rhythm/cam4/o2_fr_cam4_20261021_qc.png`, `.../fr_cam4_20261021/timing_summary.csv`.
* Licks and misses: `<run>/imaging_encoding/qc/lick_contact_by_epoch.csv` + `trial_attempts*.csv`.

## Next

1. Master rank 12 -> Priya chooses the rank -> refit with leave-one-animal-out -> split-half -> leave-one-day-out ->
   freeze -> project 108 tasks -> fitsession -> report + clips + validation -> Priya names the master factors.
2. Redo sections 5, 5b, 6 on the master (all four animals): f9 / contralesional licking, per lick, EVR gap.
3. Encoding: FR run A summary (encoding window); then the master set into the encoder.
4. Other cameras: Priya's landmark clicks -> registration -> ROIs -> O2 -> multi-view factor set (`plan --views`).

## Change log

* 2026-10-24 (later): four-animal readouts on own8 (6-master, 6c, 6d); push check (6b).

* 2026-10-24: rank 12 chosen (left / right lick pair); master named (4a); leave-one-day-out noise floor on four animals
  (3); per-lick by epoch on four animals (5b-master: contralesional per-lick deficit 4 / 4); new code: factor matching /
  angles (`face_rhythm_match.py`), push-aligned check, task vs non-task + severity panels (readouts).

* 2026-10-23 evening: master rank sweep complete (12 added); rank 10 chosen.
* 2026-10-23 afternoon: findings pass (Priya via the encoding window): status table; section 1 -> 108 / 108 and
  interior folds CORRECTED to 0-0.34 %; section 2 master sweep (ranks 8, 10); master-vs-pilot rank-10 match (4);
  sections 5 / 6 re-checked on fr-timing-v1 (unchanged); FR run A marked pending summary; where to look / next.
* 2026-10-23: section 5c (licks and misses, AR-HMM cohort cross-check).
* 2026-10-23: section 1b (true bin times; PS92_0908 encoding R2 0.506 -> 0.618 after the fix).
* 2026-10-22 pm: pilot labels.json written (Priya's names).
* 2026-10-22 pm: QC 95 / 108 (PS94 complete, PS95 pre-subacute).
* 2026-10-22 pm: section 5b (per-lick by epoch: f9 deficit persists per lick, counts recover).
* 2026-10-22 pm: section 6 (EVR gap vs the pre noise floor, pilot readouts).
* 2026-10-22 pm: rank sweep complete (14, 16); rank 12 vs 10 structure; per-factor stability added.
* 2026-10-22: created (tracking QC, rank sweep start, noise floor, pilot factor identities, first epoch changes).
