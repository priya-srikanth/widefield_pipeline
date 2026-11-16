# Stop-rule census, round 2, 2026-11-04 (read-only)

Round 1: `STOP_RULE_GUARD_CENSUS_20261104.md`.

Priya's calls on round 1:
- close_center's unjudged no_response counts as stop evidence.
- Every trial after the last camera lick is the quit and is **stopped**, with no guard. This applies in every arm.
- The block-level guard is dropped.

Everything here uses round 1's per-trial raw classes. Those came from `TC.session_trials(label, "cue_stop")` in cue order, under profile `cohort1_rebuild_20261103` (tag ends `stopRun6j`). There are 107 cohort-1 sessions: pre 31, acute 16, subacute 17, chronic 43. The pre sessions are August-only, and PS93_0813 is absent. No repo code was edited. Nothing was written to M:. One process was used.

## Summary

**A2 (both-sides) is the only arm that leaves pre untouched and still restores acute far-spout failures.**
- Pre pooled hit rate is unchanged (Δ 0.000); 0 pre trials are restored.
- 222 acute far-spout trials come back (about 14 per acute session).
- Acute far-spout hit: .431 → .405. Acute pooled deficit: −.265 → −.289.

**The longer-run arms are the next most specific, and they give a smooth curve.** A3 > 8 / > 10 / > 12 restore 200 / 295 / 372 acute far-spout trials. They cost pre only −.002 / −.002 / −.004 pooled hit (33 / 33 / 78 pre trials).

**The other arms move pre more:**
- A1 trial-level guard: −.011 with trimming, −.005 without.
- A4 (end cut only): −.023. A4 also takes acute close-spout hit rates down to about .78, which is plainly disengagement being scored as failure.

**In every arm the working set and the rest period are unchanged.** Across all 8 arms, 0 contact / incomplete trials are reclassified (checked on every trial). Every arm differs only in which no_response / jaw_only trials before the last lick are stopped.

### Arm comparison

Hit rates are the cohort mean of the four animals. The deficit is each animal's hit rate minus its own August pre, then averaged. "Restored" = trials stopped under A0 that are back in the working period under the arm.

| arm | stopped trials | mid-session stop runs (no ref) | **pre pooled hit** (Δ vs A0) | pre trials restored (per session) | **acute far-spout trials restored** (per session) | acute far-spout hit | acute far mid hit | acute deficit, far spouts | acute deficit, pooled | subacute deficit, pooled | chronic deficit, pooled |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A0 current (run ≥ 6 + end) | 5925 | 237 (79) | .956 (0) | 0 | 0 | .431 | .470 | −.500 | −.265 | −.059 | +.018 |
| A1 guard, trimmed | 4394 | 131 (0) | .946 (−.011) | 208 (6.7) | 511 (31.9) | .378 | .420 | −.537 | −.309 | −.086 | +.018 |
| A1 guard, no trim | 5189 | 158 (0) | .952 (−.005) | 91 (2.9) | 319 (19.9) | .394 | .435 | −.530 | −.296 | −.067 | +.017 |
| **A2 both sides** | 5611 | 200 (42) | **.956 (0.000)** | **0 (0)** | 222 (13.9) | .405 | .452 | −.527 | −.289 | −.063 | +.017 |
| A3 run > 8 | 5408 | 160 (33) | .954 (−.002) | 33 (1.1) | 200 (12.5) | .407 | .447 | −.523 | −.287 | −.072 | +.017 |
| A3 run > 10 | 5175 | 135 (20) | .954 (−.002) | 33 (1.1) | 295 (18.4) | .397 | .431 | −.533 | −.299 | −.077 | +.015 |
| A3 run > 12 | 4922 | 113 (12) | .952 (−.004) | 78 (2.5) | 372 (23.3) | .390 | .426 | −.537 | −.306 | −.080 | +.016 |
| A4 end cut only | 1313 | 0 | .933 (−.023) | 465 (15.0) | 1191 (74.4) | .324 | .345 | −.581 | −.383 | −.143 | +.020 |

The per-session rates are the comparable specificity numbers. The acute set has 16 sessions and pre has 31, so the raw counts are not directly comparable.

**Reading.** Every arm that restores acute far-spout failures also lowers the acute close-spout hit rates a little, except A2 and A1-no-trim, which leave close ipsi / close mid at .933 / .915. Those trials are disengagement.
- **A2:** the specificity follows from its construction. All 6 pre no-reference runs lack a reference-spout contact on at least one side, so A2 stops all of them. 16 of 45 acute no-reference runs are stopped for the same reason; the other 29 are bracketed by contacts and stay in.
- **A3 curve:** acute far-spout trials restored rise 200 → 295 → 372 as the threshold goes > 8 → > 10 → > 12. Pre stays almost flat until > 12 (33 → 33 → 78). Acute close-mid hit falls .910 → .899 → .890, so the extra trials at > 10 and > 12 include some disengagement too.
- **A1 trimmed:** the costliest guard on pre (−.011, 6.7 restored per session).
- **A4:** confirms the mid-session stop rule is needed. Without it, pre loses 2.3 points of hit and acute close ipsi falls to .782.
- **Far contra** is at floor in acute under every arm (.040-.054). The arms differ at **far mid** (.470 → .420-.452) and **far ipsi** (.739 → .666-.709).
- **Chronic** is unchanged under every arm (pooled deficit +.015 to +.020).

## Tables

### R1. Hit rate per spout, cohort mean (canonical order)

Pre is given as its own block.

| epoch | arm | close ipsi | close mid | close contra | far ipsi | far mid | far contra | pooled |
|---|---|---|---|---|---|---|---|---|
| **pre** | A0 | .980 | .979 | .981 | .898 | .937 | .958 | .956 |
| pre | A1 trim | .978 | .979 | .971 | .885 | .922 | .938 | .946 |
| pre | A1 no trim | .980 | .979 | .977 | .890 | .930 | .950 | .952 |
| pre | A2 | .980 | .979 | .981 | .898 | .937 | .958 | .956 |
| pre | A3 > 8 | .979 | .978 | .979 | .896 | .935 | .957 | .954 |
| pre | A3 > 10 | .979 | .978 | .979 | .896 | .935 | .957 | .954 |
| pre | A3 > 12 | .978 | .975 | .979 | .894 | .933 | .951 | .952 |
| pre | A4 | .962 | .958 | .964 | .874 | .911 | .930 | .933 |
| **acute** | A0 | .933 | .915 | .916 | .739 | .470 | .054 | .691 |
| acute | A1 trim | .930 | .910 | .853 | .666 | .420 | .048 | .637 |
| acute | A1 no trim | .933 | .915 | .886 | .691 | .435 | .051 | .655 |
| acute | A2 | .933 | .915 | .911 | .707 | .452 | .052 | .667 |
| acute | A3 > 8 | .929 | .910 | .900 | .709 | .447 | .052 | .667 |
| acute | A3 > 10 | .924 | .899 | .887 | .696 | .431 | .051 | .655 |
| acute | A3 > 12 | .923 | .890 | .881 | .682 | .426 | .051 | .647 |
| acute | A4 | .782 | .771 | .777 | .585 | .345 | .040 | .551 |
| **subacute** | A0 | .944 | .954 | .948 | .880 | .903 | .748 | .897 |
| subacute | A1 trim | .937 | .946 | .907 | .841 | .830 | .698 | .859 |
| subacute | A1 no trim | .944 | .954 | .938 | .868 | .876 | .724 | .884 |
| subacute | A2 | .944 | .954 | .941 | .875 | .896 | .741 | .893 |
| subacute | A3 > 8 | .928 | .945 | .943 | .867 | .871 | .738 | .883 |
| subacute | A3 > 10 | .927 | .938 | .935 | .866 | .863 | .735 | .878 |
| subacute | A3 > 12 | .922 | .932 | .928 | .865 | .853 | .729 | .873 |
| subacute | A4 | .839 | .845 | .850 | .788 | .765 | .649 | .790 |
| **chronic** | A0 | .978 | .980 | .984 | .957 | .981 | .967 | .975 |
| chronic | A1 trim | .976 | .980 | .970 | .941 | .964 | .954 | .964 |
| chronic | A1 no trim | .978 | .980 | .975 | .949 | .971 | .959 | .969 |
| chronic | A2 | .978 | .980 | .982 | .955 | .981 | .965 | .973 |
| chronic | A3 > 8 | .974 | .980 | .979 | .953 | .977 | .965 | .971 |
| chronic | A3 > 10 | .972 | .980 | .977 | .951 | .974 | .964 | .970 |
| chronic | A3 > 12 | .972 | .979 | .975 | .948 | .972 | .962 | .968 |
| chronic | A4 | .961 | .967 | .958 | .936 | .955 | .942 | .953 |

### R2. Deficit vs pre, cohort mean

| epoch | arm | close ipsi | close mid | close contra | far ipsi | far mid | far contra | pooled |
|---|---|---|---|---|---|---|---|---|
| acute | A0 | −.047 | −.065 | −.065 | −.159 | −.467 | −.904 | −.265 |
| acute | A1 trim | −.049 | −.069 | −.118 | −.219 | −.502 | −.890 | −.309 |
| acute | A1 no trim | −.047 | −.065 | −.091 | −.199 | −.495 | −.899 | −.296 |
| acute | A2 | −.047 | −.065 | −.070 | −.191 | −.485 | −.906 | −.289 |
| acute | A3 > 8 | −.050 | −.068 | −.079 | −.188 | −.488 | −.905 | −.287 |
| acute | A3 > 10 | −.055 | −.079 | −.092 | −.200 | −.504 | −.906 | −.299 |
| acute | A3 > 12 | −.055 | −.085 | −.097 | −.212 | −.507 | −.900 | −.306 |
| acute | A4 | −.181 | −.187 | −.187 | −.289 | −.566 | −.890 | −.383 |
| subacute | A0 | −.036 | −.026 | −.032 | −.018 | −.034 | −.210 | −.059 |
| subacute | A1 trim | −.042 | −.032 | −.064 | −.044 | −.092 | −.240 | −.086 |
| subacute | A1 no trim | −.036 | −.026 | −.039 | −.022 | −.053 | −.226 | −.067 |
| subacute | A2 | −.036 | −.026 | −.040 | −.023 | −.041 | −.217 | −.063 |
| subacute | A3 > 8 | −.051 | −.032 | −.036 | −.029 | −.063 | −.219 | −.072 |
| subacute | A3 > 10 | −.052 | −.040 | −.044 | −.030 | −.072 | −.222 | −.077 |
| subacute | A3 > 12 | −.056 | −.043 | −.051 | −.030 | −.080 | −.222 | −.080 |
| subacute | A4 | −.123 | −.112 | −.113 | −.086 | −.146 | −.281 | −.143 |
| chronic | A0 | −.003 | .001 | .003 | .059 | .044 | .009 | .018 |
| chronic | A1 trim | −.002 | .001 | −.001 | .057 | .042 | .016 | .018 |
| chronic | A1 no trim | −.003 | .001 | −.002 | .059 | .041 | .009 | .017 |
| chronic | A2 | −.003 | .001 | .001 | .057 | .044 | .007 | .017 |
| chronic | A3 > 8 | −.005 | .002 | −.000 | .056 | .042 | .008 | .017 |
| chronic | A3 > 10 | −.006 | .002 | −.002 | .054 | .039 | .007 | .015 |
| chronic | A3 > 12 | −.007 | .004 | −.004 | .054 | .039 | .011 | .016 |
| chronic | A4 | −.001 | .009 | −.005 | .062 | .044 | .012 | .020 |

### R3. Stopped trials per spout, cohort totals

These include the after-last-lick quit.

| epoch | arm | close ipsi | close mid | close contra | far ipsi | far mid | far contra | all |
|---|---|---|---|---|---|---|---|---|
| pre | A0 | 120 | 130 | 111 | 164 | 155 | 139 | 819 |
| pre | A1 trim | 115 | 127 | 81 | 116 | 100 | 72 | 611 |
| pre | A1 no trim | 120 | 130 | 99 | 136 | 130 | 113 | 728 |
| pre | A2 | 120 | 130 | 111 | 164 | 155 | 139 | 819 |
| pre | A3 > 8 | 114 | 124 | 106 | 158 | 148 | 136 | 786 |
| pre | A3 > 10 | 114 | 124 | 106 | 158 | 148 | 136 | 786 |
| pre | A3 > 12 | 113 | 114 | 105 | 151 | 143 | 115 | 741 |
| pre | A4 | 61 | 54 | 56 | 79 | 62 | 42 | 354 |
| acute | A0 | 357 | 346 | 327 | 466 | 476 | 478 | 2450 |
| acute | A1 trim | 351 | 336 | 207 | 309 | 314 | 286 | 1803 |
| acute | A1 no trim | 357 | 346 | 269 | 368 | 366 | 367 | 2073 |
| acute | A2 | 357 | 346 | 319 | 397 | 408 | 393 | 2220 |
| acute | A3 > 8 | 348 | 337 | 301 | 399 | 410 | 411 | 2206 |
| acute | A3 > 10 | 342 | 321 | 280 | 375 | 363 | 387 | 2068 |
| acute | A3 > 12 | 341 | 306 | 269 | 348 | 349 | 351 | 1964 |
| acute | A4 | 59 | 62 | 50 | 76 | 85 | 68 | 400 |
| subacute | A0 | 291 | 276 | 282 | 271 | 351 | 296 | 1767 |
| subacute | A1 trim | 278 | 265 | 190 | 188 | 236 | 177 | 1334 |
| subacute | A2 | 291 | 276 | 265 | 258 | 337 | 279 | 1706 |
| subacute | A3 > 10 | 257 | 246 | 251 | 236 | 298 | 264 | 1552 |
| subacute | A4 | 49 | 24 | 30 | 31 | 36 | 18 | 188 |
| chronic | A0 | 137 | 105 | 141 | 170 | 161 | 175 | 889 |
| chronic | A1 trim | 130 | 101 | 88 | 106 | 95 | 126 | 646 |
| chronic | A2 | 137 | 105 | 136 | 162 | 159 | 167 | 866 |
| chronic | A3 > 10 | 114 | 102 | 116 | 143 | 129 | 165 | 769 |
| chronic | A4 | 70 | 52 | 36 | 85 | 55 | 73 | 371 |

All arms × epochs, including the ones left out here, are in `stop_rule_guard_r2_cohort.csv`.

### R4. Mid-session stop runs, total (no reference-spout trial)

| arm | pre | acute | subacute | chronic |
|---|---|---|---|---|
| A0 | 23 (6) | 104 (45) | 75 (15) | 35 (13) |
| A1 trim | 13 (0) | 52 (0) | 49 (0) | 17 (0) |
| A1 no trim | 17 (0) | 59 (0) | 60 (0) | 22 (0) |
| A2 | 23 (6) | 75 (16) | 68 (8) | 34 (12) |
| A3 > 8 | 18 (4) | 67 (17) | 52 (6) | 23 (6) |
| A3 > 10 | 18 (4) | 52 (8) | 46 (4) | 19 (4) |
| A3 > 12 | 14 (3) | 43 (3) | 41 (2) | 15 (4) |
| A4 | 0 | 0 | 0 | 0 |

Under A0, mid-session runs have median 12 trials (IQR 7-24, max 129). The after-last-lick quit has median 1 trial and is 0 in 41 % of sessions (max 123). The round-1 count of 277 runs included runs inside the tail; these 237 are mid-session runs only.

### R5. Pooled hit per animal × epoch

| animal | epoch | A0 | A1 trim | A1 no trim | A2 | A3 > 8 | A3 > 10 | A3 > 12 | A4 |
|---|---|---|---|---|---|---|---|---|---|
| PS92 | pre | .980 | .975 | .980 | .980 | .980 | .980 | .980 | .975 |
| PS92 | acute | .734 | .687 | .712 | .723 | .720 | .710 | .706 | .570 |
| PS92 | subacute | .909 | .897 | .897 | .909 | .897 | .897 | .897 | .897 |
| PS92 | chronic | .991 | .991 | .991 | .991 | .991 | .991 | .991 | .991 |
| PS93 | pre | .918 | .914 | .914 | .918 | .916 | .916 | .914 | .914 |
| PS93 | acute | .707 | .674 | .694 | .705 | .699 | .680 | .675 | .522 |
| PS93 | subacute | .876 | .840 | .868 | .876 | .872 | .866 | .859 | .785 |
| PS93 | chronic | .985 | .983 | .984 | .985 | .983 | .983 | .983 | .983 |
| PS94 | pre | .976 | .972 | .974 | .976 | .974 | .974 | .974 | .950 |
| PS94 | acute | .629 | .551 | .579 | .596 | .596 | .584 | .570 | .476 |
| PS94 | subacute | .916 | .878 | .909 | .912 | .906 | .900 | .888 | .826 |
| PS94 | chronic | .963 | .946 | .956 | .958 | .960 | .960 | .958 | .936 |
| PS95 | pre | .951 | .922 | .938 | .951 | .947 | .947 | .941 | .895 |
| PS95 | acute | .693 | .635 | .635 | .646 | .654 | .646 | .635 | .635 |
| PS95 | subacute | .888 | .821 | .863 | .874 | .857 | .849 | .846 | .652 |
| PS95 | chronic | .960 | .937 | .945 | .960 | .951 | .945 | .940 | .904 |

- PS95 acute is one session (PS95_0817).
- Under A2 the acute change is carried mainly by PS94 (−.033) and PS95 (−.047). PS92 moves −.011 and PS93 −.002. Per CLAUDE.md rule 8, the A2 shift is not visible in three animals individually.
- Per animal × epoch × spout, with a pre-hit column and the deficit: `stop_rule_guard_r2_by_animal_epoch_spout.csv`.

### R6. Working set and rest period

| arm | contact / incomplete trials reclassified | working set (contact + incomplete in the working period) | rest period (first to last camera lick) |
|---|---|---|---|
| all 8 arms | **0** | unchanged | unchanged |

This holds by construction. A run is made only of no_response / jaw_only trials, and the quit starts after the last camera lick. No arm can therefore touch a lick trial or move the first or last lick. It was also checked trial by trial on the table train, and on the DAQ cue train in round 1.

One exception is on the cue train only: PS92_0908 has an uncovered video tail. There the production code applies no end cut (`classify_session` `tail_uncovered`), while this census, working on the table train, stops that session's after-last-lick trials.

## Method

Script: `C:/Users/SabatiniLab/AppData/Local/Temp/claude/C--Users-SabatiniLab/f082ee0c-b33c-49a8-a537-dcee3bb7c84c/scratchpad/srg/r2.py`. It reads round 1's `trials.csv` (raw classes, positions and cue times per trial).

Common to every arm:
- **Quit:** every trial after the last camera lick (contact or incomplete) is `stopped`.
- **Mid-session runs:** maximal runs of consecutive no_response / jaw_only trials before the last lick. Unknown trials are skipped; a lick ends a run.
- **300 s start rule:** a stopped run that starts less than 300 s after the first cue becomes early_run, which stays in the denominator. This is the same as `classify_session` (`wfield_local/trial_classes.py:632`, `stop_min_start_s` :817).
- **Reference spouts:** `REFERENCE_SPOUTS` (:88) = close_L + close_center = close ipsi + close mid. All four lesions are LEFT.

The arms:
- **A0:** a run of ≥ 6 is stopped whole. This is the current rule; its stopped set is round 1's stopped + after_end (5,925).
- **A1 trimmed:** a run is stopped from its first reference-spout trial onward, if that part is ≥ 6. A run with no reference trial is kept. A1 no-trim: a run of ≥ 6 with any reference trial is stopped whole.
- **A2:** a run of ≥ 6 that contains a reference trial is stopped whole. A no-reference run of ≥ 6 is kept only if the nearest reference-spout trial before it AND the nearest one after it are both contacts; otherwise it is stopped whole.
- **A3:** a run of ≥ 9 / ≥ 11 / ≥ 13 trials (> 8 / > 10 / > 12) is stopped from its first trial. There is no reference guard.
- **A4:** no mid-session stops; only the quit.

Hit rate = contact / (contact + incomplete + jaw_only + no_response + early_run), pooled over trials within animal × epoch (× spout). The cohort figure is the mean of the animals. Deficit = the animal's epoch hit minus its own pre hit, then averaged.

## Uncertain

- **A2's "both sides".** I read it as the *nearest* reference-spout trial on each side being a contact. "Any reference contact before and after" would almost always hold, so it would be close to A1-no-trim's restored set plus the no-reference runs.
- **A2 rests on few runs.** Only 29 acute no-reference runs are kept, and 2 of the 4 animals (PS94, PS95) carry most of the acute change.
- **A3 counts jaw_only toward the run, as asked.** It has no reference guard, so above > 10 it starts restoring some close-spout trials in acute (close mid .915 → .899 → .890).
