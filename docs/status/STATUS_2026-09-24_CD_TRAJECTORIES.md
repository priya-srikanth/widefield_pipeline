# CD trajectories — status and handoff (2026-09-24)

**START HERE** for `wfield_local/cd_trajectories.py`. Everything below is measured or implemented;
where it is neither, it says so.

Priya, 2026-09-24: *"In my 2pRAM_pipeline I've been doing a lot of CD analyses and plotting the CD
for different trial types/epochs… build a CD for each lick direction and plot all [trials] for each
spout position trial type aligned to lick? same for ENL etc"* — and on how it is done there,
*"I think we're getting scalar CD but then plotting the projections onto the CD over time."*

---

## What it is

`position_coding_directions` already fits per-position directions on the per-animal frozen joint
basis and reports a SCALAR per trial per window. This module reuses its `direction`, `poles` and
`orthogonalise` unchanged and adds the missing step: projecting the signal at every frame onto one
fixed direction, so a trial becomes a time course.

**A trajectory needs a TIME-INVARIANT direction.** `position_coding_directions` fits in
(component × time sub-bin) space — 90 × 4 for an ENL window — and such a vector cannot be applied to
a single frame. So this module fits on the window MEAN (`bins=1`) and lets the time structure live
in the projection. Nothing existing is replaced; this is a fifth object beside its four methods.

---

## THE HEADLINE FINDING: the position differences in the raw figures were an ARTIFACT

Priya: *"are the downward deflections in close locations artifact?"* — **yes, largely.**

The directions are one-vs-rest, `w_P = mean(P) − mean(not-P)`, so across the six they nearly cancel:
**the six unit vectors sum to a vector of length 0.289**, where six aligned ones would give 6.0. A
signal common to every trial therefore CANNOT load positively on all six — the geometry forces it
positive on some and negative on others. The lick response is exactly such a signal, and it is large.

Decomposing each trace into a SHARED part (the grand mean — the same signal for every position,
differing only in which direction it is projected onto) and a position-specific residual:

**PS95 pre-cue, extreme over t ∈ [0.2, 1.5] s:**

| position | observed | SHARED | POS-SPEC |
|---|---|---|---|
| close_center | **+14.08** | **+14.32** | **−0.24** |
| close_R | +6.11 | +5.01 | +1.10 |
| close_L | +4.05 | +2.61 | +1.44 |
| far_L | **−4.71** | **−4.96** | +0.25 |
| far_center | **−6.43** | **−5.56** | −0.87 |
| far_R | −3.64 | −3.45 | −0.19 |

**The position-specific component is ~1–2 at every position.** The spectacular +14 and the −6 dips
are the shared term. Priya: *"i'm confused why the orthogonalized versions still seem grossly
similar across positions"* — **that similarity is the signal; the drama was the artifact.**

---

## FOUR CONSTRUCTION BUGS, all found by checking against the definition rather than by eye

The anchor check is what caught two of them and it now prints on every figure: **pre-stroke success
must average 1.00 over the fitting window**, because that is what the poles define.

1. **Components were not standardised.** `pcd.direction(method="dom")` is `mean(Xp) − mean(Xn)`
   normalised as a WHOLE VECTOR. Measured on PS95's 95-component basis: per-component sd spans
   **107×** and the **top five components hold 82% of variance**, so the direction was set by about
   five components on amplitude alone. A 2p CD is built on dF/F of comparable scale — Priya: *"this
   should be similar to using a single-cell population for CD"*. Now z-scored in a **frozen
   pre-stroke frame** (per session would erase real between-session differences; per epoch would
   normalise away the post-stroke change being measured).

2. **The smoothing window was erasing the shape.** A 2 s boxcar turns a step into a 2 s ramp, so
   every lick-aligned panel rose from 0 and was still climbing at +2 s — the boxcar's impulse
   response, not the animal's. It came from requiring t=0 to equal the scalar score exactly, which
   was a conceptual error: **the poles constrain the AVERAGE over the fitting window, not every
   instant.**

3. **Median → mean.** The poles define 1 as a MEAN. PS92's per-trial projection is strongly skewed:
   mean of pre-stroke success is 1.000 at every position, exactly, while the median is
   2.13/1.48/1.07/−0.03/−0.63/1.48 — **three of six read below zero for pre-stroke success alone.**

4. **The "trailing" window was LEADING.** `np.convolve(..., "full")[i]` is already the window ending
   at `i`; taking `full[n-1:]` returns the window ending at `i+n-1`, shifting the whole trace one
   window early so t=0 showed the POST-cue response.

**AND THE CACHE HID FIX 4.** The re-run returned identical values to twelve significant figures:
`courses_cache_kind` hashed the INPUTS and none of them move when the code between them does.
`COURSE_VERSION` now exists for exactly this — **bump it whenever `cd_courses` or `smooth` changes
what they compute.**

---

## MEASURED PARAMETERS — none of these are inherited defaults any more

### Fitting window: 2 s (Priya: *"should we try a tighter pre-cue window, eg the 1s before cue?"*)

d′ = (p1 − p0) / pooled per-trial sd, which is what a coding direction is for:

| width | PS92 d′ (n) | PS95 d′ (n) |
|---|---|---|
| 0.5 s | 0.255 (3911) | 0.336 (5974) |
| 1.0 s | 0.255 (3911) | 0.337 (5974) |
| **2.0 s** | **0.353 (3829)** | **0.357 (5958)** |
| 3.0 s | 0.411 (**458**) | 0.412 (5081) |

**Tighter is WORSE**, monotonically, in both animals — 1 s costs 28% of d′ in PS92. Wider scores
better and is REFUSED: PS92 keeps only 458 of 3911 trials at 3 s (the spout arrives ~3 s before the
cue, so the lick-free gate has no slack), and a 3 s window swallows the spout-arrival transient.

### Display smoothing: 0.2 s centred (Priya: *"should we try narrower than 0.4s centered?"*)

Peak amplitude @ latency vs NO smoothing, PS95 lick-aligned:

| width | close_center | far_center | worst-case cost |
|---|---|---|---|
| 0.00 | 2.61 @ +1.99 | 2.72 @ +0.38 | — |
| **0.20** | 2.59 @ +1.99 | 2.61 @ +0.32 | **−4%** |
| 0.40 | 2.58 @ +1.99 | 2.50 @ +0.38 | −8% |
| 1.60 | 2.38 @ +1.99 | 1.32 @ +0.70 | −51%, latency shifts |

The cost is NOT uniform: slow close-position peaks (+1.4 to +2.0 s) lose <1% even at 0.4 s, while
the FAST far-position transients (+0.32 to +0.51 s) lose 4–8%. Below 0.2 s there is no return —
0.10 and 0.00 sit within 1% everywhere, which is the 1–2 s haemodynamic kernel.

### Span: −3 s to +4 s. ### Aggregation: MEAN, band = 95% CI OF THE MEAN, not trial spread.

---

## ORTHOGONALISATION, AND WHY K=1 WAS NOT ENOUGH

`--orth` projects the condition-independent subspace out of every direction, fitted on **pre-stroke**
sessions and applied to every epoch (a per-epoch mode would subtract away the post-stroke change
being measured — rule 10). Now the **top-K** components of the grand-mean trajectory, K chosen by
variance explained (`CIM_VAR = 0.90`, `CIM_KMAX = 8`). PS95 pre-cue picks **K = 2**.

K=1 was insufficient on two counts, both measured:

* the shared response is a TIME COURSE occupying several dimensions — after K=1, PS95's shared
  column still ran −0.47 to +1.83 against a position-specific signal of +1.0 to +1.9;
* **a subspace rotates where a single vector appears not to**: PS95's K=1 cosines were
  0.88/0.95/0.88 across post-stroke epochs while the K=2 subspace overlap is **0.68/0.61/0.66**.

### THE OVERLAP METRIC MUST BE READ AGAINST CHANCE, AND BRIEFLY WAS NOT

Two unrelated K-dim subspaces of an n-dim space overlap at **K/n**, not 0 — here 2/95 = **0.021**.
So 0.61 is **~30× chance**. Both readings are true and they answer different questions:

* **biologically** — the global mode is strongly CONSERVED across epochs; 0.61 is not "a third
  rotated away", and must NOT be reported as a deficit;
* **for the ARTIFACT correction** — overlap is a POWER fraction, so the amplitude remaining is
  √(1−0.61) = **0.62**. Against a shared amplitude of 2.6–14.3 that leaves a residual of **1.6–8.9**
  in the trace, i.e. **~1× to 6× the position-specific signal, worst at close_center.**

**Only the second licenses discounting a post-stroke panel.** Per position, residual =
0.62 × |SHARED|, against the position-specific signal that survives orthogonalisation.
**THE TABLE BELOW IS OPTIMISTIC BY A FACTOR THAT IS NOW MEASURED AND IS NOT UNIFORM.** It assumes
the shared amplitude is unchanged post-stroke; `cim_scale` measures it GROWING in 12 of 12 cells.
Leak = `sqrt(1 − overlap) × scale` runs **0.68 to 1.46, mean 0.89**, against the 0.62 assumed here —
so between 10% and 130% worse, not a flat 35%. **PS94 chronic exceeds 1.0 in all three alignments**
(1.40–1.46): more unremoved shared amplitude than the whole pre-stroke shared response, against a
position-specific signal of ~1–2. That panel is not readable at any scaling.

| position | \|SHARED\| | residual | signal | ratio |
|---|---|---|---|---|
| close_L | 2.61 | 1.62 | 1.70 | **0.95×** |
| close_R | 5.01 | 3.11 | 1.92 | 1.6× |
| far_R | 3.45 | 2.14 | 1.17 | 1.8× |
| far_center | 5.56 | 3.45 | 1.57 | 2.2× |
| far_L | 4.96 | 3.08 | 1.00 | 3.1× |
| **close_center** | **14.32** | **8.88** | **0.99** | **9.0×** |

**close_L, close_R and far_R are interpretable post-stroke; close_center is NOT.** It is pathological
on both counts at once — the largest shared component and one of the smallest signals.

### THE COSINE CANNOT SEE AMPLITUDE, AND THE TABLE ABOVE IS AN UPPER BOUND UNTIL IT IS PAIRED

Priya, 2026-09-24: *"the cosine will not read out amplitude changes though, right"* — right. Subspace
overlap is SCALE-INVARIANT, so a response that keeps its orientation and halves in size scores 1.00.
For a lesion study that is a blind spot on the most likely effect.

`cim_scale` now reports the magnitude beside it (Frobenius norm of the grand-mean deviation, as a
ratio to pre). **MEASURED ACROSS ALL FOUR ANIMALS AND ALL THREE ALIGNMENTS** (2026-09-24 evening;
the PS95-only figure that used to stand here was 1.00 / 1.51 / 1.35 / 1.32 and it generalises):

| magnitude / pre | acute | subacute | chronic | overlap range |
|---|---|---|---|---|
| PS92 | 1.10 | **1.61** | 1.28 | 0.60–0.63 |
| PS93 | 1.32 | 1.49 | **1.75** | 0.57–0.72 |
| PS94 | 1.17 | 1.21 | **1.74** | **0.33**–0.64 |
| PS95 | **1.51** | 1.35 | 1.32 | 0.61–0.68 |

**Above 1.0 in 12 of 12 cells, mean 1.41.** The TIME COURSE differs per animal — PS93 and PS94 climb
to chronic, PS92 peaks subacute, PS95 acute — so only the SIGN is claimed, and that is unanimous
(rule 8: present in all four individually).

**CUE IS NOT AN INDEPENDENT REPLICATION OF PRE-CUE.** `session_arms` centres BOTH on the cue
(`at = c0`); the align token only chooses which window the DIRECTION is fitted on, and these two
quantities come from the grand mean alone. So the two agree by construction and differ only in the
trial set (pre-cue drops trials with no lick-free window; PS93 shows the difference is real but
small, K=2 vs 3). **LICK is the independent alignment** and gives 12/12 above 1.0, mean 1.43.
**Effective n is 12 cells, not 24.**

**PS94 CHRONIC IS THE ONLY CELL IN THE REORGANISATION CORNER** — lowest overlap (0.332–0.338) AND
largest amplitude (1.72–1.79), consistent across all three alignments. Overlap down with scale up is
the only combination that means the code went somewhere else rather than got stronger or weaker.

Two consequences, and the first is a finding in its own right:

* **The condition-independent mode GROWS post-stroke.** A cosine-only analysis would have reported
  "overlap 0.61, mostly conserved" and missed this completely. It is exactly the dissociation the
  pairing exists to expose.
* **Every residual figure above is ~35% too optimistic.** The unremoved amplitude is
  `sqrt(1 − overlap) × ||shared|| × scale`: acute 0.57 × 1.51 = **0.85**, subacute 0.62 × 1.35 =
  **0.84**, chronic 0.58 × 1.32 = **0.77**, against the 0.62 the table assumes. close_center's 9×
  is closer to **12×**.

The two DISSOCIATE, and the dissociation is the point:

| overlap | scale | reading |
|---|---|---|
| ~1 | ~1 | nothing moved |
| ~1 | <1 | same geometry, weaker drive |
| <1 | ~1 | **REORGANISATION** — the code went somewhere else |
| <1 | <1 | mixed, and neither number alone would say so |

A decoder score conflates all four, since every one lowers accuracy. **This is the main thing the
subspace view adds over "can position still be read out".** It also repairs the residual estimate:
the unremoved amplitude is `sqrt(1 − overlap) × ||shared||`, and the table above assumes the second
factor is unchanged post-stroke. **`cim_scale` is computed but has NOT yet been run on real data.**

This is why PS93's dips survive orthogonalisation post-stroke while vanishing pre-stroke:

| PS93 precue, min of success over [0,4] | pre | acute | subacute | chronic |
|---|---|---|---|---|
| close_center plain | −3.64 | −5.56 | −7.14 | −6.85 |
| close_center **orth** | **+0.22** | −1.82 | −5.13 | **−6.45** |

The correction is total at pre and almost nothing by chronic — degrading in proportion to distance
from the epoch it was fitted on.

---

## THE close_center FIX: SCALE BY THE ORIGINAL GAP, NOT THE COLLAPSED ONE

Priya: *"the residual after orthogonalizing is so small that the small denominator blows everything
up"* — confirmed, and it is TWO problems compounding. PS95 pre-cue, K=2:

| position | gap plain | \|w·CIM\| | gap orth | surviving |
|---|---|---|---|---|
| far_R | 1.761 | 0.75 | 1.156 | 65.6% |
| close_L | 1.565 | 0.47 | 1.383 | 88.4% |
| close_R | 1.291 | 0.47 | 1.141 | 88.4% |
| far_center | 1.072 | 0.44 | 0.964 | 89.9% |
| far_L | 0.932 | 0.37 | 0.867 | 93.0% |
| **close_center** | **0.737** (smallest) | **0.80** (largest) | **0.448** | **60.7%** |

close_center is worst on BOTH counts at once — the weakest separation to begin with AND the most of
it inside the shared subspace — so its recomputed denominator is **3.1× smaller** than close_L's and
everything divided by it is inflated 3.1×. **far_R is the control that proves it is the gap and not
the alignment**: nearly the same overlap (0.75) and loss (66%), but it started largest and lands
fine.

**THE CAUSE WAS RECOMPUTING THE POLES AFTER ROTATING**, which rescales whatever residual survives
back up to 1 and manufactures the inflation. Now scaled by the UNROTATED gap, so the axis means
"fraction of the ORIGINAL position-P signature" and a barely-surviving direction draws SMALL, which
is the truth. `surviving` is recorded per position and printed on each panel, red below 70%.

**Effect on real data** (PS95 pre-cue, peak over [0.2, 1.5] s):

| position | survives | peak BEFORE (renormalised) | peak AFTER |
|---|---|---|---|
| close_center | 60.7% | **+14.08** | **2.32** |
| close_L | 88.4% | +4.05 | 2.01 |
| close_R | 88.4% | +6.11 | 1.75 |
| far_center | 89.9% | −6.43 | 2.10 |
| far_R | 65.6% | −3.64 | 1.77 |
| far_L | 93.0% | −4.71 | **0.29** |

close_center falls into line with everything else, and far_L's genuinely small excursion (0.29,
despite surviving 93% intact) becomes visible where the old scaling hid it.

## WHAT IS BUILT

| | state |
|---|---|
| `--layout epochs` | 6 panels, one per position, 4 epochs overlaid, **shared y** |
| `--layout cross` | 4 epochs × 6 CDs, each overlaying all 6 positions' trials; own-position heavy. **sharey per row** |
| `--gate {lick, lick_or_working}` | which trials the direction is fitted on AND the trace is built from, pooled at TRIAL level. Priya: *"traces should be the same trials its trained on"* |
| `--reference {contrast, rest, restw}` | what P is contrasted against. **`rest`/`restw` are UNVERIFIED — never completed a run** |
| `--orth {off,on}` | top-K condition-independent subspace projected out |
| parallelism | fanned over **animals** via `ak.fan_sessions`; inner loops serial on purpose (one shared basis per animal) |
| caches | `cdarms-` / `cdfit-` / `cdcourse-` / `cdgm-` / `cdrest-`. 140 s cold → **3 s warm** |
| tests | `tests/test_cd_trajectories.py`, **21 passing** |

---

## NOT DONE / NOT VERIFIED — read this before trusting anything on the share

0. **SUPERSEDED IN PART — read "STATE AT 2026-09-24 EVENING" below first.** Items 3 (persistence),
   5 (the null, now built but not run) and the overlap figure are done; item 2's re-render is done for
   `contrast`/`lick`/`orth on`.
1. **`rest` / `restw` HAVE NEVER COMPLETED A RUN.** The code is written and unit-tested against
   synthetic data; the one real-data attempt died before producing a result. **Verify before using.**
2. **Every figure currently on the share is STALE** —
   `N:\MICROSCOPE\Priya\Claude outputs\enl_cd_20260924\` predates the reference flag, top-K, the
   chance level and the shared y-axis. A clean full re-render is the first thing to do.
3. **No overlap figure.** The numbers are in every figure's title; Priya asked for a dedicated one
   and it does not exist yet.
4. **The miss-while-working-on-cue-and-lick-CDs figure** Priya asked to keep separate from these is
   not built.
5. **The overlap has no matched null.** See below — this is the most important open item.
6. **Figure data is not persisted**, so every re-plot recomputes. Priya asked for this explicitly.
7. **`cim_scale` has been run only on PS95 pre-cue.** The 1.51/1.35/1.32 growth is one animal, one
   alignment. It rewrites the residual arithmetic, so it should be measured across animals before
   being leaned on.

---

## OPEN: the null for subspace overlap, and a possible general method

Priya: *"so we want to compare post-stroke vs pre-stroke cosine to pre-stroke vs pre-stroke LOSO
cosine?"* — **yes, and it is not built.**

Two subspaces estimated from DIFFERENT SESSIONS will not fully overlap even when nothing changed, so
0.61 has no scale without a matched null. The construction to copy is
`scripts/rest_migration/rest_baseline_epoch_drift.py`, which answers a structurally identical
question about the restw baseline: **split PRE into two groups, one holding `n_E` sessions**, and
read the observed value against that. Match on session count; a true leave-ONE-out null is too noisy
and mismatched in n.

One thing that makes our case CLEANER than theirs: their cosine is biased positive because `d`
carries `−mean_pre` and `e_ref` carries the same vector with the same sign. Our subspaces are
estimated from DISJOINT session sets, so that shared-term bias does not arise.

**And the larger idea** (Priya: *"this would be a new way for us to think about representational
geometry that might be really good"*): the same machinery applied to the **POSITION subspace** — the
span of the six CDs — rather than the shared one asks whether the geometry of the position code
itself rotates after stroke. That is distinct from what the project measures now: the decoder asks
"can position still be read out", RSA asks "are the distances preserved", subspace overlap asks "is
it the SAME subspace" — and a code can stay equally decodable while living somewhere else in state
space. To be trustworthy it needs: the chance level always shown; **principal angles, not just their
mean** (which dimensions rotate is the finding); and the matched null above.

---

## STATE AT COMPACTION (2026-09-24)

**`eed1731` and `a11a136` are PUSHED** (`origin/main`, verified in sync). The evening's work is
NOT yet committed.

### DO THESE IN ORDER

1. **Verify `restw`.** It has never completed a run. `python -m wfield_local.cd_trajectories
   --animal PS95 --align precue --reference restw --orth on --layout epochs`. Both previous attempts
   died because the module was edited mid-run — see the process note below.
2. **Re-render everything.** All 24 figures on the share predate the reference flag, top-K, the
   chance level, `cim_scale`, the original-gap scaling and the shared y-axis. One command, parallel
   over animals:
   `--align precue cue lick --layout epochs cross --gate lick lick_or_working --reference contrast
   restw --orth on --out "N:/MICROSCOPE/Priya/Claude outputs/enl_cd_20260924"`
3. **Persist the figure data** (Priya, 2026-09-24: *"don't re-compute things multiple times, store
   caches where able, and store the numbers needed for each figure so we can re-plot without having
   to recompute"*). `analyse_animal` returns everything a figure needs; dump it per
   (animal, align, gate, reference, orth) and let the layouts read from disk. **NOT STARTED.**
4. **The overlap figure.** The numbers are in every title; Priya asked for a dedicated one. With (3)
   done it is a small script over the dumps, like `enl_decode_figure` over `enl_decode`'s JSON.
5. **The subspace-overlap null** — see below. Without it, 0.61 has no scale beyond K/n.
6. **miss-while-working projected onto the cue and lick CDs**, which Priya asked to keep OUT of
   these figures (*"separately (to not clutter these figures)"*).

## STATE AT 2026-09-24 EVENING — what changed since the morning handoff

### DONE

1. **Figure data PERSISTS** (was item 3, Priya's explicit ask). `save_result` / `load_result` /
   `result_tag` on top of `wfield_local/results_store.py` — the repo's existing store, not a new one
   (rule 9). `--replot` redraws from `<out>/results/` and computes nothing. `RESULT_GUARD` **refuses**
   a dump made under different constants rather than redrawing it under today's title, which is the
   `COURSE_VERSION` lesson applied to the dump: a saved result is a cache too.
2. **All 24 figures + 12 dumps re-rendered** for `contrast`/`lick`/`--orth on`, all animals, all
   three alignments. 0 problems.
3. **`cim_scale` measured across animals** — see above. This was the open item the morning handoff
   flagged as resting on one animal.
4. **The overlap figure exists**: `scripts/cd_geometry_figure.py`, three rows — overlap against its
   K/n chance BAND, magnitude, and `sqrt(1−overlap) × scale` with a dashed line marking what the
   residual table assumed. Reads the dumps, computes nothing, so it cannot drift from the panels.
5. **The trial-matched pre-to-pre null is built**: `wfield_local/cd_overlap_null.py` (+ 10 tests).
   See DECISIONS.md for the construction, why it is TRIALS not sessions, and the two bugs found.
6. **`wfield_local/component_exclusion.py`** — which components sit under the painted glue or in a
   bulb, and `cd_trajectories --mask-occluded` to drop them. See DECISIONS.md: about a quarter of
   every basis, no preferential loading by the CD, and dropping them costs no more than dropping a
   random quarter.

### BUGS FIXED (both were live and both were silent)

* **The two passes disagreed about which trials the grand mean is over** while writing ONE cache key.
  Pass 1 averaged every class in the gate, pass 2 `success` only, so under `--gate lick_or_working`
  whichever session ran first won the key: a two-class pre-stroke mean against one-class post-stroke
  ones, i.e. a trial-composition change arriving as an amplitude change **in `cim_scale`, which
  exists to measure amplitude changes**. Identical under `--gate lick`, which is why the default path
  was correct and this stayed invisible. The accumulation is now ONE function, `grand_means`, and the
  cache token is `cdgm2-` because the old entries cannot be told apart from correct ones by key.
* **Every figure title named no reference at all.** `f"{{'contrast': …}}[res.get('reference')]"` —
  doubled braces make the dict a literal and the subscript plain text, so the title printed the
  source of the lookup instead of its answer. Now `REFERENCE_LABEL`.

### STILL OPEN, in order

1. **Run the matched null on real data** and report `observed − null` per cell. Until then the 0.59
   overlap and the 1.41 scale have no scale beyond K/n — *"the right comparison [is] the amplitude
   and cosine similarity of pre to pre vs pre to post stroke"* (Priya).
2. **Fold subsampling for the trial match** (Priya: *"can we do subsampling of each session for
   trial matching"*). K disjoint trial-fold means per session instead of one; see DECISIONS.md for
   why this also gives the split-half debias for free. PS93 acute/subacute need it (6.9%/10.6%
   mismatch on whole sessions).
3. **`restw` is STILL unverified** — it has never completed a run on real data.
4. **DONE — outputs are on the server** (Priya, 2026-09-24: *"outputs should go to the widefield
   directory on the server when appropriate"*) at
   `N:/MICROSCOPE/Priya/Widefield/labcams/analysis_figures/cd_trajectories`: 26 current PNGs plus the
   geometry figure, dumps in `results/`, and the **62 superseded pre-fix figures MOVED to `retired/`
   rather than deleted** (rule 1 — never delete on MICROSCOPE; `retired/` is already a recognised
   layout dir). `cd_trajectories.default_out()` now resolves `cue_analysis_out` and
   `cd_overlap_null` shares it, so there is ONE answer to where this arm writes.
   `check_figure_layout` reports 0 strays. Two things that constrain the location and are easy to
   trip over: `nightly_figs._publish_figs` globs `*.json` at the TOP level of that root as publish
   inputs (hence the `results/` subdir), and the root has NOT adopted `figure_layout`. The old
   `Claude outputs/enl_cd_20260924` copy is still in place and can be removed once Priya confirms.
5. **`locanmf_decoder_weights` maps weights to anatomy through the RAW mask** (lines 227, 473) with
   no glue or bulb subtraction. This is the one place the exclusion is a genuine defect rather than a
   scope difference — see DECISIONS.md.
6. **miss-while-working projected onto the cue and lick CDs**, which Priya asked to keep OUT of these
   figures.
7. **The gate comparison** (`lick` vs `lick_or_working`) has never been rendered since the `cdgm2-`
   fix, and that fix only changes anything for `lick_or_working`.
8. **A SELECTIVITY GATE on components** is the most promising open idea and it is not built. See
   DECISIONS.md: it is the 2p-faithful answer to equal-variance weighting, and it would subsume the
   glue question by dropping components for WHAT THEY CARRY rather than where they sit.
9. **`--method lr` as a third arm.** It cuts the occluded components' weight share to 15-20% in three
   of four animals. Never rendered.
10. **DONE — the decoder mask test says no.** Median change −0.9% over 15 (animal, epoch) cells,
   improving in 5 of them; see DECISIONS.md. No published decoder number needs revisiting.
11. **Post-stroke CD panels are mask-sensitive** (DECISIONS.md, 2026-09-24): pre-stroke is robust
   (median r 0.885-0.910) but individual post-stroke cells move and one reverses sign, and the cells
   that move are the low-`surviving` ones. Any post-stroke claim needs the `--mask-occluded` arm
   beside it; only PS94 and PS95 pre-cue have been run.

## STATE AT 2026-09-25 — the pooled/derived layer, and one retraction

### DONE SINCE THE 09-24 ENTRY

1. **The trial-matched pre-to-pre null RAN**, first time on real data, and both readings survive it:
   observed overlap is below its ceiling in 12/12 cells (ceiling 0.79-0.97, not 1.0) and the
   magnitude is above its ceiling in 11/12. **FOLD SUBSAMPLING was needed and changed a conclusion**:
   whole-session matching left PS93 acute at +18% and four cells with an interval collapsed to a
   point, and it put PS95 acute's scale ceiling at 1.42 against an observed 1.53 -- nearly explaining
   the effect. Trial-matched to +0% it is 1.04, so the growth is clear. The coarse match was
   flattering the NULL, not the observation.
2. **`scripts/cd_cross_animal_figure.py`** — the pooled delta from each animal's own pre-stroke
   trajectory, both layouts, all three windows.
3. **`scripts/cd_migration.py`** — where a trajectory moved to. See DECISIONS.md, including the claim
   it retracted.
4. **Per-session diagonal traces are persisted**, so `boot_delta` runs its real nested draw. THREE
   CELLS MOVED FROM EXCLUDING ZERO TO SPANNING IT; the four-animal interval had been too narrow.
5. **Wired into the nightly** (`--skip-cd`) in dependency order, and three new families registered in
   the deck with their own caveats.
6. **Colours reconciled** to `epoch_figures.EPOCH_GREY` and `spout_behavior.position_style`; line
   weights and alphas centralised.

### THE RETRACTION, because it is the most important thing on this page

**`far_R -> far_center` is ONE animal of four**, and that animal's acute epoch is a single session. It
was stated as a cohort finding before the per-animal check was run. Rule 8 is now enforced
structurally in `cd_migration` rather than by a docstring caveat -- the caveat was already there.

### OPEN, in order

1. **Add the session-weighted column and the disagreement flag** (decided in DECISIONS.md, not built).
2. **far_R's diagonal collapse, per animal** — the other half of the migration claim, still only read
   off the pooled matrix.
3. **`restw` REMAINS UNVERIFIED.** Carried from 09-24 and still never run on real data.
4. The `lick_or_working` gate comparison, never rendered since the `cdgm2-` fix.
5. **miss-while-working on the cue and lick CDs**, still not built.

## PROCESS NOTE — two background jobs were lost to this

**Do not edit `cd_trajectories.py` while a render is running.** Workers re-import the module per
task, so a mid-patch file kills them with exit 255 and no traceback. It happened twice on
2026-09-24.
