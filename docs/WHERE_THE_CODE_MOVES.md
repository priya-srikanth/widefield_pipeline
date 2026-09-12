# Where does the displaced spout-position code move? — the anatomical question

Started 2026-09-12. Priya: *"I want to start trying to answer **where** the displaced spout position
codes move post-stroke."*

Everything in the deck so far answers **where in representational space** — far-contralateral's best
match goes to far-middle (0.50 of sessions) and far-ipsilateral (0.25); the row-centred crossnobis
gives the direction; the encoder says far-contra keeps its own structure while losing the template.
Nothing answers **where in cortex**. This document is that work.

---

## 0. Does the pre-stroke basis invalidate the question? No — and not for the reason first given

Priya asked the right question before letting any of this proceed: the joint LocaNMF basis is fitted
on PRE-STROKE sessions and post-stroke days are projected onto it, so could it be discarding exactly
the post-stroke structure we are looking for?

**The measured answer:**

| | |
|---|---|
| session SVD rank (`SVTcorr`) | **100** |
| joint basis dimensions | **100** |
| subspace overlap (pre-stroke) | **0.996–0.997** |

**The basis is not a reduction, it is very nearly a rotation.** One hundred dimensions spanning
99.7% of a hundred-dimensional dataset discards almost nothing. The dimensionality reduction people
worry about happened earlier — in the **rank-100 SVD in preprocessing** — which is upstream of every
analysis here and applies identically to pre- and post-stroke sessions. Whatever it discarded is
gone from all of them equally and cannot bias a pre-versus-post comparison.

**A small real drift does exist.** The misalignment roughly doubles after the lesion, 0.29–0.37%
pre-stroke against 0.48–0.80% acutely. That is worth reporting as drift outside the pre-stroke
subspace; it is not evidence of relocation.

### The control that failed, and why it is worth knowing about

The plan was to decode position from the residual — the component of activity orthogonal to the
basis. **It cannot work at this rank**, and it produced a convincing wrong answer first:

> "The residual" decoded position at **0.93 balanced accuracy pre-stroke** (chance 0.167), from 0.3%
> of the variance. That is not a discovery. Feeding 95 near-null directions into a pipeline whose
> `StandardScaler` z-scores every feature independently resurrects them all to unit variance and
> reconstructs a re-mix of the session's entire temporal space.

Two things gave it away: the residual eigenvalue spectrum is **isotropic** (0.029, 0.028, 0.028,
0.028, 0.028, 0.027 — a noise floor, not a coding subspace), and restoring the correct
singular-value scaling changed the decode **not at all**, because the scaler undoes it.

Recorded in `wfield_local/basis_residual.py`, which is kept for that reason rather than deleted.
**Testing what the preprocessing SVD discards needs the pre-SVD movie and is a separate undertaking.**

---

## 1. Why the existing families cannot answer "where"

**Best-match destination is anatomically blind.** It collapses all 380 features into one correlation
per position pair and takes an `argmax`. It names a representational destination, not a location.
Restricting it to hemisphere or Allen module recovers only as much anatomy as the grouping has bins
— two numbers, or eight. A coarse proxy, not a location.

**The mean pattern carries anatomy but only if it is not collapsed**, and per component the basis
gives ~2–4 components per Allen area (64 distinct areas across 95 components; 32 areas ignoring the
hemisphere sign), which is too thin to localise.

**Decoder weights are the wrong tool, and not for the reason first assumed.** The ROI constraint
does not apply — the joint basis makes component *j* the same footprint in every session by
construction, which is what `joint_xsession` exists for, so weights ARE comparable across sessions.
The problem is that multinomial-logistic weights over 95 correlated components are unstable: a
weight can flip sign with no change in the information represented, because a correlated neighbour
absorbs it. Any anatomical claim from raw decoder weights would need the Haufe transform
(`A = Σ_x W`) to convert discriminative filters into interpretable activation patterns first.

---

## 2. What does work

### Pixel-space position maps (Priya's suggestion, and the direct answer)

`framemap_event_maps` already writes, per session, a `*_spout_positions_1s_pre_post_delta_maps.npz`
holding 18 arrays — 6 positions × {pre-cue, post-cue, delta} — as **540 × 640 Allen-aligned pixel
maps**. **123 of these exist on the share.** Aggregating them by epoch is the remaining work.

Full spatial resolution, a common anatomical frame, and no component-space intermediary. "Where"
becomes a picture.

### Coding directions rendered back to cortex

`position_coding_directions.direction()` is a **difference of means** — `mean(P) − mean(not-P)`,
unit-normalised in (component × bin) space. Two properties matter:

* it is **contrastive**, so it isolates position coding from global activity — a delta map lights up
  when the animal simply moves more, a coding direction cannot;
* it is a **mean difference, not a fitted discriminative weight**, so none of the logistic-weight
  instability above applies.

The module currently only ever *projects* onto it (`project()` returns a pole-normalised scalar per
trial), so the anatomy in `w` is computed and then discarded. Rendering `w` back through the LocaNMF
footprints gives the cortical pattern that distinguishes one position from the rest; doing it
pre- and post-stroke and differencing is the anatomical shift, isolated to position coding.

### The two are complementary, not redundant

| | frame | resolution | isolates position coding? |
|---|---|---|---|
| post−pre position maps | Allen pixels | full | no — any activity change shows |
| coding direction → cortex | LocaNMF footprints | ~95 blobs | **yes** — contrastive by construction |

---

## 3. Statistics for the maps

Priya: *"is there a way to show stats or the variation, of within-animal pre- and post-stroke
sessions? (a graphical version of post-minus-pre mean ± SEM)?"*

**Unit of variation: sessions WITHIN animal.** ~11 pre-stroke and 4–6 acute sessions per animal
makes per-pixel variance estimable within animal. Pooling four animals for a per-pixel SEM would
rest on n=4, which is far worse — better to draw each animal with its own session-level variation
and let the four panels show the replication.

**Display: mean map with a reliability contour, not a masked map.** Masking destroys magnitude,
which is the thing being read. Per animal × position: mean post−pre delta in colour, contour where
the effect is reliable, effect-size map beside it, with
`SE = sqrt(var_post/n_post + var_pre/n_pre)` and `t = delta/SE` per pixel.

**The threshold is a real test, not an arbitrary cut.** 540 × 640 = 345,600 pixels, so an
uncorrected per-pixel threshold is meaningless. A **cluster-based permutation test** is the field
standard and is feasible here: shuffle the pre/post session labels within animal, recompute the *t*
map, record the largest cluster mass, repeat. With 11 pre and 5 acute sessions there are
C(16,5) = 4,368 distinct relabellings — ample for p<0.05 with no parametric assumption. The contour
then means "this cluster is larger than 95% of clusters obtainable by relabelling sessions".

**And the per-position delta is confounded by global change** — it lights up wherever activity
changed at all, including from the animal simply moving more. The position maps are already
post-cue-minus-pre-cue so common changes largely cancel, but the **contrast between positions**
(far-contra delta *minus* far-middle delta) is the version that isolates position coding. Both get
rendered.


---

# Status at 2026-09-12 — what is built, what is known-broken, what is pending

## Built and pushed

| | commit | state |
|---|---|---|
| `wfield_local/beta_maps.py` — Haufe-transformed L2-on-SVT cortical maps | `946c413` | working |
| epoch figure 14 + `ef.map_grid` + deck placement with full method notes | `84bbcba` | working |
| CCF outlines, per-panel trial n, class weighting, floor, permutation contours | uncommitted | working |

## KNOWN BUG — every arm is fitted on LICKING trials only

`session_maps` takes `X, y, g` from `trial_features_cached`; those are the ENGAGED trials. The
no-lick trials that make the `working` class uniform come back as `Xn, yn` and are **discarded**.
So `variant` currently selects only whether class weighting is applied, not which trials are used,
and the two arms labelled `working` are lick-trial maps.

**How it surfaced:** figure 14's pre-cue panel refused the far-contralateral ACUTE cell. Chasing
that gave far-contra trials per session of 0–5, 4–12, 0–8 and 1 for the four animals, against
66–119 at the best position — which is the deficit itself, not a bug in the floor.

**Consequence:** the "pre-cue and post-cue need no balancing" argument does not apply to what is
being fitted, and far-contra acute is thin in *every* arm. **Fix before quoting any number from
this family.**

## What the figures currently show (subject to the bug above)

Post-cue, map amplitude relative to each position's own pre-stroke value:

| near ipsi | near middle | near contra | far ipsi | **far middle** | **far CONTRA** |
|---|---|---|---|---|---|
| 0.67 | 1.53 | 1.04 | 1.09 | **0.48** | **0.47** |

The two positions losing more than half their map amplitude acutely are far-middle and
far-contralateral — the same pair every other analysis implicates — both recovering by subacute
(0.83 / 0.91). **This converges with the encoder**, whose fitted amplitude factor goes 0.749
pre-stroke → 0.286 acute → 0.745 chronic, and with the encoder ceiling's displaced-vs-degraded
split. Three methods sharing little machinery agreeing is the strongest thing here.

**Priya's reading was right and mine was wrong:** the faint acute maps are biological, not a
measurement failure. Low split-half *r* at far-contra acute (0.53) and its 0.47 amplitude are ONE
observation, not two — a split-half correlation of a near-absent signal is low *because* the signal
is near-absent. Far-middle falls to 0.48 amplitude while **keeping** r = 0.86, which is what shows
the two are separable.

## Pending

1. **Fix the engaged-only bug** (above). Everything else waits on it.
2. **Per-animal panels** — asked for, not built. More important than the pooled view given n=4.
3. **Permutation contours produced nothing visible** on the pre-cue render. With 4 animals the
   between-animal SE is coarse and the test is probably conservative; report that rather than
   loosening the threshold until something appears.
4. **Vessel structure** faintly visible in POST mean maps — rule out residual haemodynamic
   artefact before this is a headline.
5. **Pixel-space post−pre position maps** from the 123 existing `*_spout_positions_1s_pre_post_delta_maps.npz`
   — never built. Complementary to the decoder maps: raw activity change at full resolution, with
   no decoder and no basis in the path.
6. **Coding directions rendered through the footprints** — `position_coding_directions.direction()`
   is a difference of means and already computes the right object, then collapses it to a scalar
   via `project()`. The anatomy in `w` is discarded.

## To solidify what is here

* Four animals is the ceiling on the permutation test. The per-animal panels are the honest way to
  show replication.
* The chronic column rests on 2–3 animals; PS94 has no chronic epoch.
* A pattern map says where the signal is, **not** which pixels are necessary for decoding. That
  second claim needs the filter map, which the module can produce (`filter_map=True`) and which
  correlates with the pattern at only r = 0.245.


---

## A structural limit of one-vs-rest, found 2026-09-12

Priya: *"in acute there may be less ss-ul/ll activity in far-center trials, which makes the near ipsi
acute trial map look as though there is a relative **increase** in ss-ul/ll activity compared to
pre-stroke."*

**Correct, and it is structural.** The Haufe pattern is `cov(pixel, decoder output)` on trial-mean-
centred data, so the reference is the average over all six positions in that session. If one
position loses drive the reference falls, and **every other position's map gains an apparent
increase it did not earn**. The six maps are not independent.

### What this invalidates

The acute amplitude *increases* — near-middle 1.53 post-cue, 2.82 post-lick, 3.30 pre-cue, far-ipsi
1.09 — **cannot be read as those positions gaining anything.** They are consistent with being the
shadow of far-middle (0.48) and far-contralateral (0.47) losing amplitude: the same observation
counted twice, with a sign flip.

### What survives

**The falls.** The mechanism works *against* a position appearing to fall — its own loss raises its
reference less than it lowers everyone else's — so far-contra and far-middle dropping below half
their pre-stroke amplitude is not manufactured by the contrast. That, and the convergence with the
encoder's fitted amplitude (0.749 → 0.286 → 0.745), is the part worth keeping.

### The fix, and it is already on disk

A **per-position reference** makes the six maps independent, and `framemap_event_maps` already
computes one: per session and per position, `post-cue mean − pre-cue mean` in Allen pixels, with 123
`*_spout_positions_1s_pre_post_delta_maps.npz` on the share. The reference is **within trial**, so
far-contra's loss cannot leak into near-ipsi's map.

**This moves the pixel-map arm from last to first in the build order.** It is not a complementary
view any more — it is the control that decides whether four of the six rows in figure 14 mean
anything.

### A related check worth running

Compare **one-vs-rest** against **one-vs-quiet** maps for the same position. If they look alike, the
position code and the task-evoked response are spatially confounded and "where the position code is"
is weaker than it appears. If they differ, that is evidence the decoder reads position rather than
licking.
