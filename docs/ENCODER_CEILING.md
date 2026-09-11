# The encoder ceiling — what `frozen EV` is failing against

Branch `encoder/ceiling`, built 2026-09-10, **not merged**: it is held back so tonight's re-render
runs on a stable `main`. Merge after that render completes.

Priya: "should we do a similar per-session refit with the encoder analysis as we did with the
decoder?" Yes — and more than for the decoder, because the frozen encoder's score had no ceiling.

---

## The problem it solves

`frozen EV` is an R², and acutely it is **−0.388** post-cue. That says "worse than predicting the
mean" and says nothing about what was achievable in that session. The existing companion,
`EV after rescale`, frees the AMPLITUDE only — and `_enc_terms` warns in its own docstring that a
large rescaling gain means amplitude only when the rescaled value is HIGH, because a code that is
simply gone also recovers a lot under rescaling (the best scale collapses toward zero, and
predicting nothing beats predicting an unrelated pattern).

That ambiguity is exactly what forced the withdrawal of the "half amplitude, half shape" reading on
2026-09-10. A ceiling resolves it.

## Construction

A refit encoder must be **cross-validated or it is 1.0 by construction** — ridge on a one-hot
position design reduces to the per-position mean, so predicting a session's own means from
themselves is an identity. Splitting the trials is what makes it a prediction:

    ceiling = _enc_terms(mean of half A, mean of half B)

both orderings, 8 random splits, averaged. Which makes this the split-half family scored in the
ENCODER's units rather than as a correlation — deliberately, because a correlation is gain-blind and
`frozen EV` is not, so the two could not otherwise be read against each other.

## What it shows — post-cue, lick + miss-while-working

On the NEW epoch boundaries (chronic = PS92/PS93/PS95, 12 sessions).

| epoch | ceiling | frozen EV | EV after rescale | template loss | Δ loss vs pre |
|---|---|---|---|---|---|
| pre | 0.660 | 0.557 | 0.580 | 0.102 | — |
| **acute** | **0.475** | **−0.388** | 0.134 | **0.862** | **+0.760** |
| subacute | 0.550 | 0.176 | 0.313 | 0.373 | +0.271 |
| chronic | 0.752 | 0.400 | 0.421 | 0.352 | +0.250 |

**THE ACUTE SESSION CAN PREDICT ITSELF.** Ceiling 0.475 against a frozen EV of −0.388. So the
post-stroke pattern is not noise: it carries real position structure that the pre-stroke template
cannot reach.

Read as a fraction of the structure available in that session (EV after rescale ÷ ceiling, using the
rescaled value so amplitude is not charged twice):

| epoch | captured by the pre-stroke template |
|---|---|
| pre | 88% |
| **acute** | **28%** |
| subacute | 57% |
| chronic | 56% |

**BOTH THINGS HAPPEN, AND THE MISMATCH DOMINATES.** The ceiling itself falls 0.660 → 0.475 acutely,
so about 28% of the available structure is genuinely lost. Of what survives, the pre-stroke template
captures 28% where it captured 88% before. That is the encoder-side statement of the same result the
frozen-vs-refit decoder arm gives — partly lost, mostly mismatched — arrived at from a different
quantity.

It also **restores a weaker version of the withdrawn claim**, on a defensible footing: the acute
encoder failure is not "the tuning is gone", because the tuning is measurably there. It is "the
pre-stroke tuning no longer describes it".

## The pre-cue arm must NOT be read this way

| epoch | ceiling | frozen EV |
|---|---|---|
| pre | **0.090** | **0.330** |

The frozen arm BEATS its own ceiling, which is impossible for a real ceiling and is diagnostic of
the construction rather than of the data. Both sides of the ceiling are half-session means while the
frozen arm scores against a reference pooled over ~10 sessions, so on a weak signal the ceiling is
noise-dominated. **It is a ceiling for the cue and lick windows and a lower bound at best for ENL.**

The same training-set-size bracketing the matched frozen DECODER arm exposed, where matching flipped
the pre-stroke gap from −0.073 to +0.090.

## Still to do on this branch

1. **A size-matched frozen arm**, scoring against a reference built from a half-session-sized
   subsample of pre-stroke trials. That is what would make the pre-cue arm interpretable, and it is
   the direct analogue of `5rm`.
2. **Layout polish** — the three x-tick labels crowd, and the title wraps at the current width.
3. **Per-position ceiling.** `_enc_ceiling` already returns the per-position dict; nothing plots it.
   Far-contra is the position where the ceiling matters most and it is currently pooled away.
4. **Deck placement** — no `_EPOCH` entry yet, deliberately: placing it before the numbers are
   reviewed would put an unreviewed figure in front of readers.
