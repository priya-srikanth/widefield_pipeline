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
