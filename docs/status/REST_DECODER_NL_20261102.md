# Rest decoder (imaging page Fig 12.1) on LICK-FREE rest: STOPPED at the frame-count step (11-02)

Priya approved this run on 11-01: "rest frames used as DATA: rest decoder 12.1 re-run on lick-free rest first".
Worktree `C:/Users/SabatiniLab/Github/wf_restdec`, branch `analysis/rest-decoder-nl` (off `analysis/restnl-maps`
0e592a20). Nothing is committed. No page was edited, nothing was published, and nothing on M: was overwritten,
deleted or renamed. **No decoder was fitted and no frozen model was written**: the run stopped at the counts step
because of the fairness check below (task step 2: "too few frames, stop and report").

## 1. How 12.1 builds its rest data (unchanged)

`scripts/rest_migration/rest_frozen_decoder.py --perm 200 --refit --match-train --match-duration --match-lickgap`
(the `nightly_figs` call; it rendered on 10-07; the caption's "50 permutations" is wrong, because p = 1/201 in the CSV).

- Mask: the restdock05 `quiet_sample.npy` (docked spout, not running, DAQ contact licks excluded [0.5, 1.0] s,
  min_rest_s).
- Periods: a period is one run of that mask that lies between two ENGAGED working trials at the same position.
  Features are the joint-LocaNMF signal averaged in 4 equal bins per period.
- Decoder: frozen logistic regression (C = 0.1), fitted on all pre sessions. Each pre session is scored by a
  leave-one-session-out model.
- Matching: both train and test are cut to the central 80 % of the PRE duration and lick-gap distributions. The lick
  gap counts DAQ licks only.
- Pre set: the page used June + August pre (44 sessions).
- Not frozen: the models were refitted in memory on every render, so the script does not follow CLAUDE.md rule 10.

## 2. What was built (uncommitted)

| file | change |
|---|---|
| `scripts/rest_migration/rest_frozen_decoder.py` | `_collect` gets two opt-in keywords. `rest_mask=None` takes a callable that replaces the rest mask and adds lick onsets for the gap measure. `period_frames=False` appends the frame count of each period. No other line of the module changed. |
| `scripts/rest_migration/rest_frozen_decoder_nl.py` (new) | Driver: `--mask restnl\|rest`, `--pre aug\|all`, stages collect / counts / fit / plot, `--plot-only`, `--compare`, `--allow-thin`. <br>The RESTnl mask is `restnl_mask.lickfree_mask` applied to the decoder's own mask file: rest minus DAQ licks (the mask's own detector) and cam4 LP `licks_all.csv` licks, each widened [0.5 s before, 2.0 s after], with the min_rest_s floor re-applied. <br>Under RESTnl the lick gap is measured to DAQ and cam4 licks together. <br>Pre-stroke models (full + LOSO) go through `frozen_models.load_or_fit` (kind `rest_frozen`, align `rest-<mask>-b4-dur10-gap10-gate`). <br>Statistics: per-animal table first, then `analysis_kit.boot_delta` (nested, paired). Stars use `sig_marks.star_top` + `reserve_headroom`. Session markers are open, shape = animal, color = epoch (EPOCH_LINE). Titles name the lick source and the baseline. <br>New fairness gate `MIN_TRAIN_PER_POS = 30`: an animal is not fitted unless every position has at least 30 matched pre training periods. |
| `tests/test_rest_frozen_decoder_nl.py` (new) | Default-path proof (section 3). |

Cache: the collected features are in `C:/Users/SabatiniLab/cue_lick/rest_decoder_nl_cache/{rest,restnl}/<label>__<key>.npz`.
The key hashes the source of `_collect`, `_frame_samples` and `lickfree_rest`, plus the RESTnl parameters.

## 3. The default output stays byte-identical

`tests/test_rest_frozen_decoder_nl.py` passes 4 of 4, and 71 tests in the existing rest-decoder suites also pass
(`test_rest_frozen_decoder*`, `test_substitute_code`, `test_rest_engagement`). The new test checks three things:

1. Every function of the module other than `_collect` has source identical to commit 0e592a20. This covers `main`,
   `_plot`, the nulls, the fits and the confusion code.
2. The new keywords default to None / False.
3. On PS92_0814 and PS94_0821, the new `_collect` with default arguments returns SHA-256-identical arrays (dtype,
   shape and bytes) to the 0e592a20 source, which the test loads from git history. Adding `period_frames=True` leaves
   the first 6 elements unchanged.

No cache key or frozen-model spec of the default path changes: the default path does not use either.

Mask check: on PS92_0814, the decoder's RESTnl period frames per position, [269, 147, 64, 178, 59, 483], equal the map
baseline's `nfr_pos` (RESTNL_MAPS_20261101 section 3). The mask is therefore the same RESTnl.

## 4. Frame / period counts (before any fit)

Files: `G/rest_frozen_decoder_restnl/{rest,restnl}_augpre/data/rest_frozen_<mask>_augpre_durgap_counts_by_session.csv`
(session x position, every session including June), `G/rest_frozen_decoder_restnl/counts_summary/counts_session_summary.csv`
and `scorable_after_matching.txt`. Here `G = M:/MICROSCOPE/Priya/Widefield/labcams/grant_figures`. Collection covered
120 sessions with 0 failures. **PS93_0813 has no `licks_all.csv` and so no RESTnl periods.**

Periods are decoder-valid rest periods: bracketed by same-position engaged working trials, with at least 4 frames,
before matching. Position order: close ipsi, close mid, close contra, far ipsi, far mid, far contra (lesion left, so
L = ipsi).

**August pre, summed over sessions: REST -> RESTnl periods (frames)**

| animal | close ipsi | close mid | close contra | far ipsi | far mid | far contra | total kept |
|---|---|---|---|---|---|---|---|
| PS92 (8 sess) | 206 -> 78 | 308 -> 75 | 286 -> 86 | 270 -> **41** | 269 -> 100 | 370 -> 130 | 1709 -> 510 (30 %) |
| PS93 (8 -> 7) | 569 -> 152 | 503 -> 101 | 571 -> 179 | 570 -> 88 | 400 -> 45 | 542 -> **14** | 3155 -> 579 (18 %) |
| PS94 (8) | 373 -> 136 | 424 -> 149 | 378 -> 142 | 504 -> 97 | 444 -> 224 | 465 -> 183 | 2588 -> 931 (36 %) |
| PS95 (8) | 621 -> 574 | 605 -> 553 | 615 -> 567 | 639 -> 551 | 636 -> 580 | 656 -> 540 | 3772 -> 3365 (89 %) |

Frames, August pre (sum over 6 positions), REST -> RESTnl: PS92 63.1k -> 16.2k, PS93 119.4k -> 18.4k, PS94 96.6k ->
30.7k, PS95 147.5k -> 128.8k. PS93 far contra = 425 RESTnl frames. (CSV rig names: L = ipsi, center = mid, R = contra.)

**Median periods per session, REST -> RESTnl**

| animal | pre (Aug) | acute | subacute | chronic |
|---|---|---|---|---|
| PS92 | 223 -> 54 | 297 -> 102 | 207 -> 56 | 242 -> 94 |
| PS93 | 402 -> 55 | 328 -> 88 | 353 -> 127 | 306 -> 116 |
| PS94 | 313 -> 120 | 328 -> 198 | 199 -> 98 | 258 -> 118 |
| PS95 | 481 -> 431 | 596 -> 513 | 250 -> 212 | 394 -> 215 |

**After the page's duration + lick-gap matching: the frozen model's PRE training pool, per position (canonical order)**

| animal | REST, Aug pre | RESTnl, Aug pre | RESTnl, June + Aug pre |
|---|---|---|---|
| PS92 | 150 179 215 164 135 244 (1087) | 59 53 59 **25** 66 88 (350) | 118 124 128 82 142 151 (745) |
| PS93 | 436 309 459 266 182 348 (2000) | 120 67 129 64 33 **9** (422) | 212 185 272 141 116 111 (1037) |
| PS94 | 274 228 260 389 228 257 (1636) | 106 113 105 66 161 117 (668) | 171 194 191 136 255 192 (1139) |
| PS95 | 386 410 384 405 475 376 (2436) | 356 382 334 342 452 329 (2195) | 455 487 422 443 558 428 (2793) |

Scored periods per pre session after matching, RESTnl Aug pre: PS92 median 45 (6 of 7 sessions reach the minimum of
40), PS93 median 43 (6 of 7), PS94 median 84 (7 of 8), PS95 median 282 (8 of 8). Every post session stays scorable.

## 5. Why it stopped: the August-only RESTnl decode is not fair for PS92 and PS93

- **PS93 far contra.** Lick-free rest at far contra almost disappears before the stroke: 14 periods and 425 frames
  (about 28 s) over 7 August sessions. After matching, 9 periods remain to train the frozen model, against 348 under
  REST.
  - A model trained this way rarely predicts far contra.
  - The pre test sets contain about 1 far-contra period per session, so balanced accuracy is averaged almost without
    that class.
  - The post test sets do contain far contra (PS93 chronic RESTnl: 118 periods).
  - The retained fraction would therefore be biased DOWN at exactly the stroke-relevant position, for one animal, by
    design rather than by the lesion.
- **PS92.** The pool is 350 periods with only 25 at far ipsi. Pre sessions
  are scored on about 45 periods each, roughly 7 per class, so the pre baseline that every retained fraction divides
  by is noisy.
- PS94 is adequate: the minimum is 66 per position and it passes the gate. PS95 is barely changed.
- With n = 4 and 2 animals compromised, a pooled retained trajectory, or the 4/4 "rest > null" headline, could not be
  read as a lick-free result. The new `MIN_TRAIN_PER_POS = 30` gate encodes this check. Under it, PS92 (far ipsi 25) and PS93 (far contra 9)
  are not fitted without `--allow-thin`.

**A finding in its own right (counts only, no decoding):** before the stroke, PS93's docked-spout "rest" at far
contra is 97 % within [-0.5, +2.0] s of a video or DAQ lick (14 of 542 periods survive). PS92 keeps 30 % of its rest
periods and PS94 36 %. The page's 12.1 pre-stroke rest code for PS92 / PS93 is therefore built mostly on periods next
to licks that DAQ contact does not see. This is the confound the 12.1 script docstring warned about (Priya 09-16,
"PS92 is VERY licky"). The page's lick-gap matching cannot remove it, because the gap was measured to DAQ licks only.

## 6. Comparison with the current 12.1 / conclusions that change

**Not computed, because no fit was run.** Nothing in the page's 12.1 numbers is tested or changed by this run.
- The page's numbers: retained 0.28 acute, 0.36 subacute, 0.54 chronic; 4 / 4 rest > null.
- What the counts do say: the page's PS92 / PS93 pre rest periods are mostly within the video-lick buffer. "Rest
  carries position" is not yet shown to hold for lick-free rest in those two animals.

## 7. Options for Priya (decision needed)

1. **RESTnl with June + August pre** as the sensitivity run. Every animal passes the gate (minimum 82 per position;
   PS93 far contra 111). The cost is breaking the August-only primary rule, and June adds a learning-stage baseline.
   Command: `--mask restnl --pre all`, plus `--mask rest --pre all` to validate the driver against the page CSV.
2. **August pre RESTnl on PS94 + PS95 only** (they pass), reported per animal, with no pooled claim.
3. **Score only positions with at least 30 training periods in every epoch** (symmetric across epochs). For PS93 this
   drops far contra, which is the position of interest.
4. **A shorter post-lick buffer** (for example [0.5, 1.0] s, as the page's REST uses for DAQ licks), applied to cam4
   licks. It recovers periods but is a different RESTnl from the map baseline's.
5. `--allow-thin` (fit anyway), reported with the bias stated. Not recommended.

The driver supports options 1, 2 (`--allow-thin` off just skips PS92 / PS93) and 5 with no code change. Each fit is
expected to take a few minutes from the cache (200 permutations x 3 nulls + refit per session). Under option 1, the
frozen models are written to the local store `C:/Users/SabatiniLab/frozen_models/<animal>/rest_frozen_*` (new kind;
publishing is a separate act).

## 8. Timings

- Collection: RESTnl 14.5 min and REST 14.2 min (120 sessions each, 4 workers, about 10-14 s per session, basis
  served from MICROSCOPE).
- Tests: about 1 min each run.
- Dry matching count: about 1 min.
- Free RAM: at least 36 GB at the start. Only the processes started here were run, and all have exited.
