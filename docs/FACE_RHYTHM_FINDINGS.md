# FaceRhythm findings (cam4)

Status: **preliminary, living document.** Last updated **2026-10-22** (FaceRhythm window). Update a section when its
result changes (edit, re-tag, add to the change log). Reasoning and decisions: `DECISIONS.md` (2026-10-21 evening,
2026-10-22); how everything runs: `docs/FACE_RHYTHM_PIPELINE.md`; working state: the newest
`docs/status/STATUS_*_FACE_RHYTHM_*` handoff.

Status tags: **STANDS**, **PROVISIONAL** (a control or the four-animal set still pending), **PILOT** (two animals,
PS92 + PS93, the pilot factor set -- to be redone on `master_pre_v1`), **PENDING**.

> **Cohort and rules.** PS92-PS95, all lesions LEFT; on cam4 the ipsilesional face is IMAGE RIGHT; spouts / licks
> named by the animal's side (right = contralesional). August-only pre baseline; engaged trials. n = 4 animals at most;
> every claim needs the per-animal table (CLAUDE.md rule 8). Expression = ABSOLUTE (weights x FR normaliser D) unless
> stated; a.u. on each animal's own scale, so compare within an animal across epochs.

## 1. Tracking quality (STANDS, 95 / 108 tasks)

Production tracking (`midlk12`, 12 px master mesh) is clean in every animal and epoch: whole videos covered, 0 % NaN
bins, 0 violations, folds inside the face in 0-0.15 % of sampled frames (`o2_face_rhythm.py qc`). Rim folds (28-65 %)
are near-flat sliver triangles at the mesh edge and do not affect the factors. Median point displacement (px) pre ->
acute -> subacute -> chronic: PS92 1.33 / 1.11 / 1.16 / 1.18; PS93 1.19 / **0.80** / 1.21 / 1.36; PS94 2.06 / 1.33 /
2.35 / 2.63 (PS94 moves ~2x more or has a larger face in px; absorbed by the per-animal scale). PS95 (pre / acute / subacute,
chronic still copying): 1.65 / **0.96** / 1.41, interior folds 0.1-0.3 % (highest of the four, still negligible). Acute
displacement drops below pre in PS93 and PS95 (and mildly PS92) -- the face moves less in the first days.

### 1b. Bin timing: true times from per-frame timestamps (STANDS, 10-23)

The runner's LINEAR frame -> DAQ stamps were wrong on 2 of 108 cam4 tasks (`o2_face_rhythm.py timing`,
`<bundle>/timing_summary.csv`): PS92_0908 dropped 11.8 % of frames (57 % in its last tenth) -> session / points stamps
off by a median 3.2 s, up to 841 s; PS92_0812 v1's video ran ~39 min past its ~10 min of DAQ recording -> bins after
636 s stamped at DAQ times with no recording. All others: <= 20 ms. Now true times everywhere + lossy bins NaN
(PS92_0908: 5 % of session, 6 % of trial bins). Effect downstream (encoding window, FR run A rerun on the corrected
pilot projections): PS92_0908 FR-model cortex R2 **0.506 -> 0.618** (ENL 0.545 -> 0.663, cue 0.466 -> 0.573, first lick
0.413 -> 0.527; FR's unique contribution +0.26) -- the bad stamps had been diluting FR; PS92_0812 unchanged (its
mistimed bins lay outside the imaged frames).

## 2. How many factors (PROVISIONAL, pilot tensor)

Fit-only sweep on the PS92 + PS93 pre-stroke fit rows (3 starts each): EVR / stability across starts -- rank 6 0.767 /
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

Leaving one pre-stroke DAY out of the fit costs <= 0.01 EVR on that day's session rows: PS92 held-out median 0.798
(0.788-0.815) vs in-fit 0.805, gap median 0.006 (max 0.009); PS93 0.805 (0.773-0.813) vs 0.804, gap 0.001 (max 0.004);
held-out factors match the full fit (similarity 0.84-1.00). Moving between ANIMALS costs ~0.10 EVR (fit one, project
the other: held-out error 0.52-0.54 vs 0.43-0.45 in fit) -- animals differ far more than days, so the master set needs
all four animals. Pre-stroke sessions are always scored on their held-out basis (DECISIONS 10-22 #2).

## 4. What the pilot factors are (PILOT; names Priya's, checked against licks + treadmill)

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
| f7 | nose / sniff / whisk while walking | strongest RUNNING factor (speed r 0.46 in trials, 0.58 outside licks) + a lick-initiation transient |
| f8 | nose wiggle with steps | running-linked (0.29), sustained after onset; no lick-direction link |
| f9 | right licking, early (faster) | RIGHT-selective (r 0.48, left -0.07), early AND late, ongoing licks, ~7 Hz; far-R > close-R |

Groups: lick INITIATION / preparation (f6, f3, f1, + f7's transient), ONGOING licking by side and tempo (f0 left slow,
f4 centre-right slow, f9 right, f5 fast early), MOVEMENT / whisking (f7, f2, f8). Caveat: absolute expression includes
the frame's overall movement, so every factor correlates positively with speed outside licks (0.05-0.58) -- read
speed relatively.

**Names (Priya 10-22, `pilot_pre_PS92_PS93/labels.json`; this fit only -- never carried to another fit by number):**
f0 `lick_left_slow`, f1 `bout_init`, f2 `run_onset_jaw`, f3 `nose_prelick`, f4 `lick_centreR_ongoing`, f5
`lick_fast_early_Lbias` (left-leaning but the least side-specific lick factor: per-lick expression at right licks
0.47-0.72 of left vs f0 0.24-0.52), f6 `prep_jaw`, f7 `whisk_run`, f8 `nose_steps`, f9 `lick_right`. Loader / encoder
columns are `fr_<name>`. The encoding direction split keeps f5 in "left" for now (Priya).

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
Encoding grouping (Priya-approved, `wfield_local/fr_groups.py`, pilot set only): lick = f0 + f4 + f5 + f9 (left f0 + f5,
centre f4, right f9); other = f1 f2 f3 f6 f7 f8 (includes the lick-initiation factors).

## Change log

* 2026-10-23: section 5c (licks and misses, AR-HMM cohort cross-check).
* 2026-10-23: section 1b (true bin times; PS92_0908 encoding R2 0.506 -> 0.618 after the fix).
* 2026-10-22 pm: pilot labels.json written (Priya's names).
* 2026-10-22 pm: QC 95 / 108 (PS94 complete, PS95 pre-subacute).
* 2026-10-22 pm: section 5b (per-lick by epoch: f9 deficit persists per lick, counts recover).
* 2026-10-22 pm: section 6 (EVR gap vs the pre noise floor, pilot readouts).
* 2026-10-22 pm: rank sweep complete (14, 16); rank 12 vs 10 structure; per-factor stability added.
* 2026-10-22: created (tracking QC, rank sweep start, noise floor, pilot factor identities, first epoch changes).
