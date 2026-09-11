# The encoder ceiling — what `frozen EV` is failing against

Branch `encoder/ceiling`, built 2026-09-10, **not merged**: held back so tonight's re-render runs on
a stable `main`. Merge after that render completes.

Priya: "should we do a similar per-session refit with the encoder analysis as we did with the
decoder?" Yes — and more than for the decoder, because the frozen encoder's score had no ceiling.

---

## The problem

`frozen EV` is an R², and acutely it is **−0.388** post-cue. That says "worse than predicting the
mean" and nothing about what was achievable in that session. `EV after rescale` frees the AMPLITUDE
only — and `_enc_terms` warns in its own docstring that a large rescaling gain means amplitude only
when the rescaled value is HIGH, because a code that is simply gone also recovers a lot under
rescaling. That ambiguity forced the withdrawal of the "half amplitude, half shape" reading on
2026-09-10.

> **WHAT "EXPLAINED VARIANCE" IS THE VARIANCE OF, because the name invites the wrong reading.**
> Priya, 2026-09-10: "if the encoder is literally an average per position, we're asking how much of
> all the trials the AVERAGE (binned) per-position activity explains?" The first half is right and
> the second half is not. The encoder IS the per-position average of the binned component activity.
> But the score is computed MEAN against MEAN: `M` and `P` are both 6 x features matrices of
> per-position mean patterns, centred across positions, and the denominator is `sum M^2` — the
> BETWEEN-POSITION variance of the measured means. Trial-to-trial variance never enters it. So the
> number answers "how much of the measured position-to-position pattern does the template's
> position-to-position pattern reproduce", not "how much of the trial variance is explained"; a
> trial-level R2 would be far lower, because single-trial noise is large and is excluded here by
> construction. It also means the CEILING's shortfall from 1.0 is entirely sampling noise in the
> half-session means — which is why halving the trials lowers it, and why the size-matched arm was
> needed at all.

## Construction

A refit encoder must be **cross-validated or it is 1.0 by construction** — ridge on a one-hot
position design reduces to the per-position mean, so predicting a session's own means from
themselves is an identity. Splitting the trials makes it a prediction. Three arms, all scoring the
SAME half-session means, 8 random splits, both orderings averaged:

| arm | reference | holds constant |
|---|---|---|
| **ceiling** | the other half of the same session | — |
| **frozen (matched)** | an equally sized draw from the pre-stroke pool | training-set size |
| **frozen (all pre)** | the whole pre-stroke pool | nothing; this is the encoder as actually used |

**Read `ceiling − frozen (matched)`.** Only that pair holds size constant, so only it isolates "the
template came from other sessions". And read it against its own PRE value, not against zero.

## Why the matched arm was required

Without it the pre-cue window was uninterpretable — frozen EV **0.330** against a ceiling of
**0.090**, a frozen arm beating its own ceiling, which is impossible for a real ceiling and is a
fact about the construction rather than the data. The unmatched frozen arm scores against a
reference pooled over ~10 sessions while the ceiling's reference is half a session.

Matching removes the asymmetry, and the inversion goes with it **in all three windows**: ceiling ≥
matched at pre-stroke everywhere.

## Results — SHAPE ONLY (after rescale)

All three arms are scored after rescale, so amplitude is out of every number. Raw scores at these
amplitudes are dominated by the encoder not being allowed to rescale: with PERFECT shape and only a
scale mismatch, `R² = 1 − (1−a)²/a²`, which is exactly 0.000 at a = 0.5 and **−2.13** at the
a = 0.361 observed acutely. A raw ceiling-minus-matched gap therefore reports amplitude and calls it
template mismatch, because the ceiling's two halves have matched amplitude by construction while the
matched frozen arm does not.

### Post-cue (new boundaries; chronic = 12 sessions, 3 animals)

| epoch | ceiling | matched | gap | Δ vs pre | template captures | amplitude a |
|---|---|---|---|---|---|---|
| pre | 0.703 | 0.426 | 0.277 | — | 61% | 0.745 |
| **acute** | 0.571 | 0.098 | 0.473 | **+0.196** | **17%** | **0.285** |
| subacute | 0.643 | 0.222 | 0.421 | +0.144 | 35% | 0.517 |
| chronic | 0.778 | 0.319 | 0.459 | **+0.183** | 41% | 0.741 |

**AMPLITUDE RECOVERS; SHAPE DOES NOT.** The fitted scale goes 0.285 → 0.517 → **0.741** against a
pre-stroke 0.745 — back to baseline. The shape mismatch goes +0.196 → +0.144 → **+0.183** — flat.
An earlier version of this document reported the mismatch as monotonically recovering (+0.666,
+0.309, +0.181); those were RAW scores and what was recovering in them was the amplitude.

**THE PRE-STROKE GAP IS 0.277 AND IS NOT AN EFFECT** — the cost of a template coming from other
sessions at equal training-set size, with no lesion involved. Same phenomenon the matched frozen
DECODER arm exposed, where matching flipped its pre-stroke gap from −0.073 to +0.090.

### Post-lick

| epoch | ceiling | matched | gap | Δ vs pre | captures | a |
|---|---|---|---|---|---|---|
| pre | 0.749 | 0.467 | 0.282 | — | 62% | 0.775 |
| acute | 0.608 | 0.231 | 0.377 | +0.095 | 38% | 0.561 |
| subacute | 0.678 | 0.245 | 0.433 | +0.151 | 36% | 0.559 |
| chronic | 0.817 | 0.376 | 0.441 | +0.158 | 46% | 0.785 |

Milder acutely than post-cue, and the same non-recovery of shape with amplitude back at baseline.

### Pre-cue — self-consistent now, but weak throughout

| epoch | ceiling | matched | gap | Δ vs pre | captures | a |
|---|---|---|---|---|---|---|
| pre | 0.363 | 0.168 | 0.194 | — | 46% | 0.437 |
| acute | 0.309 | 0.059 | 0.250 | +0.056 | 19% | 0.222 |
| subacute | 0.278 | 0.067 | 0.211 | +0.017 | 24% | 0.223 |
| chronic | 0.335 | 0.063 | 0.271 | +0.077 | 19% | 0.245 |

The inversion is gone — ceiling exceeds matched at every epoch — but the ceiling is 0.363 at its best
against 0.70–0.75 for the other two windows, and the amplitude factor never recovers (0.245 chronic
against 0.437 pre-stroke). Contrasts here are small differences between small numbers; treat as
suggestive only.

## Per position — and this is the finding

Post-cue, shape (after rescale). `ceiling` = what that position's own trials can predict; `match` =
what the pre-stroke template gets; `frac` = the second over the first.

| position | pre ceil / frac | acute ceil / frac | subacute frac | chronic frac | Δ ceiling acute |
|---|---|---|---|---|---|
| near ipsi | 0.741 / 0.67 | 0.585 / 0.22 | 0.41 | 0.33 | −0.156 |
| near middle | 0.453 / 0.26 | 0.438 / 0.22 | 0.15 | 0.05 | −0.015 |
| near contra | 0.726 / 0.62 | 0.627 / 0.28 | 0.46 | 0.53 | −0.099 |
| **far ipsi** | 0.629 / 0.54 | 0.336 / 0.19 | 0.30 | 0.34 | **−0.293** |
| **far middle** | 0.717 / 0.63 | 0.468 / 0.43 | 0.33 | 0.52 | **−0.249** |
| **far contra** | 0.613 / 0.57 | **0.613** / **−0.11** | 0.26 | 0.26 | **0.000** |

**FAR-CONTRA LOSES NO STRUCTURE AND LOSES ITS TEMPLATE ENTIRELY.** Ceiling 0.613 acutely, identical
to its 0.613 pre-stroke — its own trials predict each other exactly as well as before — while the
pre-stroke template captures nothing. The positions that lose CEILING are the flanking ones: far-ipsi
−0.293 and far-middle −0.249.

(Per-position values use the session-global scale factor rather than a per-position one, which is
why far-contra's acute match can be negative where the pooled after-rescale score cannot.)

**This converges with the frozen-vs-refit DECODER arm**, which found refitting recovers 34% of
far-contra's acute deficit and only 9% / 11% at far-ipsi / far-middle. Two analyses, opposite
directions of fit, same dissociation: far-contra's code is DISPLACED, its neighbours' codes are
DEGRADED. The pooled "~28% of structure lost" figure averages those two different things and should
not be quoted without the split.

Figures: `epoch_11cpos_encoder_ceiling_by_position_*` (the ceiling) and
`epoch_11cfrac_encoder_captured_by_position_*` (the fraction). The fraction is dropped rather than
drawn where the ceiling is below 0.10, since a ratio to a near-zero denominator reads as a result.

## Still to do on this branch

1. **Deck placement** — no `_EPOCH` entry yet, deliberately: placing it before the numbers are
   reviewed would put an unreviewed figure in front of readers.
2. **Intervals on the derived gap.** The bars carry hierarchical-bootstrap intervals; the
   `ceiling − matched` difference is computed per epoch from the pooled values and has none.
3. Rebase before merging — `main` moved on after this branch was cut.
