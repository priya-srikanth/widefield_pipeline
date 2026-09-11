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

## Results

### Post-cue, lick + miss-while-working (new epoch boundaries; chronic = 12 sessions, 3 animals)

| epoch | ceiling | frozen (matched) | frozen (all pre) | ceiling − matched | Δ vs pre |
|---|---|---|---|---|---|
| pre | 0.660 | 0.343 | 0.557 | 0.317 | — |
| **acute** | **0.478** | **−0.505** | −0.388 | **0.983** | **+0.666** |
| subacute | 0.584 | −0.042 | 0.176 | 0.626 | +0.309 |
| chronic | 0.753 | 0.255 | 0.400 | 0.498 | +0.181 |

**THE ACUTE SESSION CAN PREDICT ITSELF.** Ceiling 0.478 against a matched frozen arm of −0.505. The
post-stroke pattern is not noise: it carries real position structure the pre-stroke template cannot
reach.

**THE PRE-STROKE GAP IS LARGE AND IS NOT AN EFFECT.** 0.317 at baseline, with no lesion involved —
the cost of a template coming from other sessions at equal training-set size. Same phenomenon the
matched frozen DECODER arm exposed, where matching flipped the pre-stroke gap from −0.073 to +0.090.

**LESION-ATTRIBUTABLE TEMPLATE MISMATCH RECOVERS MONOTONICALLY AND DOES NOT REACH ZERO:**
+0.666 acute → +0.309 subacute → +0.181 chronic.

The ceiling itself falls 0.660 → 0.478 acutely, so roughly 28% of the available structure is
genuinely lost. Both things happen; the mismatch dominates. That is the encoder-side statement of
the same result the frozen-vs-refit decoder arm gives, reached from a different quantity — and it
**restores a weaker form of the withdrawn claim on a defensible footing**: the acute failure is not
"the tuning is gone", because the tuning is measurably there. It is "the pre-stroke tuning no longer
describes it".

### Post-lick

| epoch | ceiling | matched | ceiling − matched | Δ vs pre |
|---|---|---|---|---|
| pre | 0.719 | 0.395 | 0.324 | — |
| acute | 0.502 | −0.190 | 0.692 | +0.368 |
| subacute | 0.628 | 0.013 | 0.615 | +0.291 |
| chronic | 0.798 | 0.323 | 0.475 | +0.151 |

Same shape as post-cue, smaller acutely.

### Pre-cue — now self-consistent, still nearly signal-free

| epoch | ceiling | matched | ceiling − matched | Δ vs pre |
|---|---|---|---|---|
| pre | 0.117 | −0.143 | 0.261 | — |
| acute | 0.019 | −0.508 | 0.527 | +0.266 |
| subacute | −0.045 | −0.527 | 0.482 | +0.221 |
| chronic | 0.105 | −0.437 | 0.542 | +0.281 |

The inversion is fixed, but **the ceiling is 0.117 pre-stroke and hovers near zero afterwards**, so
the gap is a difference between two near-zero quantities. Consistent is not the same as informative.
Note also that unlike the other two windows the pre-cue mismatch does NOT recover — chronic (+0.281)
is as high as acute (+0.266). Do not build on that until the arm has signal to divide.

## Still to do on this branch

1. **Per-position ceiling.** `_enc_half_scores` already returns the per-position dict and nothing
   plots it. Far-contra is where the ceiling matters most and it is currently pooled away.
2. **Deck placement** — no `_EPOCH` entry yet, deliberately: placing it before the numbers are
   reviewed would put an unreviewed figure in front of readers.
3. **Intervals on the derived gap.** The bars carry hierarchical-bootstrap intervals; the
   `ceiling − matched` difference is computed per epoch from the pooled values and has none.
4. Rebase before merging — `main` moved on after this branch was cut.
