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
0.62 × |SHARED|, against the position-specific signal that survives orthogonalisation:

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
ratio to pre). The two DISSOCIATE, and the dissociation is the point:

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

## PROCESS NOTE — two background jobs were lost to this

**Do not edit `cd_trajectories.py` while a render is running.** Workers re-import the module per
task, so a mid-patch file kills them with exit 255 and no traceback. It happened twice on
2026-09-24.
