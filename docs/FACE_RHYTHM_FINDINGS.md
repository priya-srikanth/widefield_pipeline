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

## 1. Tracking quality (STANDS, 74 / 108 tasks)

Production tracking (`midlk12`, 12 px master mesh) is clean in every animal and epoch: whole videos covered, 0 % NaN
bins, 0 violations, folds inside the face in 0-0.15 % of sampled frames (`o2_face_rhythm.py qc`). Rim folds (28-65 %)
are near-flat sliver triangles at the mesh edge and do not affect the factors. Median point displacement (px) pre ->
acute -> subacute -> chronic: PS92 1.33 / 1.11 / 1.16 / 1.18; PS93 1.19 / **0.80** / 1.21 / 1.36; PS94 2.06 / 1.33 /
2.35 / 2.63 (PS94 moves ~2x more or has a larger face in px; absorbed by the per-animal scale).

## 2. How many factors (PROVISIONAL, pilot tensor, sweep running)

Fit-only sweep on the PS92 + PS93 pre-stroke fit rows (3 starts each): EVR / stability across starts -- rank 6 0.767 /
0.84; 8 0.792 / 0.96; 10 0.807 / 0.87; 12 0.819 / **0.997**; 14, 16 pending. EVR rises slowly (no sharp elbow);
stability is not monotonic (rank 12 more reproducible than rank 10 on this tensor). The cohort rank is chosen on
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
speed relatively. No labels.json written yet.

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

## Change log

* 2026-10-22: created (tracking QC, rank sweep start, noise floor, pilot factor identities, first epoch changes).
