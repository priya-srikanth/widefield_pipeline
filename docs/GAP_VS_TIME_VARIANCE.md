# How much of the frozen-to-refit gap is just time since lesion?

**Status: provisional.** The number below is from the REST arm, pooled over the whole
cortical field. The grant's claim is about the CUE- and LICK-aligned task decoders, per
LocaNMF component. Rerun before quoting a figure (see "How to redo this properly").

Written 2026-09-30, for the K08 resubmission (Aim 1 ranking model).

## The question

Aim 1 ranks nodes by how closely their frozen-to-refit gap tracks recovery, using one LMM
per aim with a node x gap interaction. Behaviour and the gap both change over the weeks
after stroke, so a raw gap-to-behaviour correlation is confounded by their shared time
course. The fix is to enter the gap as a deviation from each animal's own trajectory rather
than raw.

That fix has a failure mode of its own, and it is the one Priya raised: **if the gap is
nearly a deterministic function of time, detrending removes almost all its variance and the
gap coefficient is then estimated from noise.** Over-adjustment looks exactly like a null
result and is not one. So the question is which regime we are in.

There is also a defence that does not depend on this at all, and it is the stronger one:
**the ranking is a within-session contrast.** Every node in an aim is recorded in the same
animals on the same days, so a shared monotone time trend is common to all of them and
cannot make node A appear to track behaviour better than node B. Time confounds the absolute
claim "gap predicts behaviour"; it does not confound "node A tracks recovery more closely
than node B", which is what the aim actually claims.

## What was measured

Source: `grant_figures/epoch/data/epoch_15f_rest_frozen_restdock05_durmatched_gapmatched_sessions.csv`
(written by `scripts/rest_migration/rest_frozen_decoder.py`; variant `restdock05`,
duration-matched and lick-gap-matched). 112 sessions, PS92-PS95, 11 pre and 17 post each.

Two gap columns, both from that script:

- `gap_refit_minus_frozen` = `refit_acc` - `acc`. Frozen = decoder fit on pooled pre-stroke
  sessions, applied unchanged, balanced accuracy, scored only on classes the frozen model
  saw. Refit = decoder fit to this session, group-cross-validated, scored on the same
  periods. Positive means the session's own readout finds position information the frozen
  readout cannot: information present but displaced.
- `gap_refit_minus_matched` = `refit_acc` - `matched_frozen_acc`, where the frozen model is
  given the SAME number of training periods as the refit arm, drawn as whole pre-stroke
  blocks with a label-seeded RNG. This removes the training-set-size handicap and is the
  column to trust.

Method: per animal, post-stroke sessions only, OLS of gap on day, on log(day), and on
day + day^2. R2 = 1 - var(resid)/var(y), plus adjusted R2, slope and residual SD.
**No pooling across animals and no random effects** -- the question is how much of ONE
animal's own gap trajectory its own day count predicts.

Day axis caveat: day 1 was taken as each animal's earliest non-`pre` session, not a recorded
lesion date. The true dates are PS94/PS95 lesioned 0816 and PS92/PS93 0817
(`epoch_grant_figures`, `_sessions`/`_day`). A constant offset is absorbed by the intercept
and cannot change the linear R2; it slightly affects the log fit only.

## Result

Linear fit, post-stroke sessions, n = 17 per animal:

| animal | raw gap R2 | adj R2 | slope/day | matched gap R2 | adj R2 | slope/day |
|--------|-----------|--------|-----------|----------------|--------|-----------|
| PS92 | 0.029 | -0.036 | +0.00081 | 0.044 | -0.020 | +0.00126 |
| PS93 | 0.003 | -0.063 | +0.00029 | 0.185 | 0.131 | +0.00312 |
| PS94 | 0.002 | -0.065 | +0.00033 | 0.002 | -0.065 | -0.00032 |
| PS95 | 0.144 | 0.086 | +0.00187 | 0.254 | 0.205 | +0.00216 |

Mean R2: raw 0.044 linear / 0.041 log / 0.082 quadratic; matched 0.121 / 0.103 / 0.138.

Residual SD after detrending 0.054-0.111, against raw gap SD 0.060-0.107: detrending removes
essentially none of the spread.

**Read adjusted R2, not R2.** With n = 17 and one predictor a pure-noise regressor earns
R2 ~ 1/16 ~ 0.06 on average. Three of four animals sit at or below that on the raw gap and
go negative when adjusted, meaning day explains no more than a random regressor would. Only
PS95 (raw) and PS93 + PS95 (matched) show anything.

The slopes are small in absolute terms too: +0.0003 to +0.0031 per day, so 0.01-0.13 of
accuracy across the full 43-day window, against session-to-session scatter of 0.06-0.11.
The noise is larger than the trend.

## What this licenses, and what it does not

Licensed: detrending the gap against each animal's own trajectory costs almost no variance
in this dataset, so over-adjustment is not the live threat. The same result says the shared
monotone trend was weak to begin with, so the confound it would have created is also small.
Both failure modes are minor here.

Not licensed: anything about the cue- or lick-aligned task decoder, and anything per-node.
A single LocaNMF component can trend with time far more strongly than the pooled field does.
Four animals is thin for a general claim.

## How to redo this properly

`scripts/gap_time_variance.py` does the same computation on the TASK arm, using
`grant_figures._collect_5c(align, variant, "paired_matched")` and
`epoch_grant_figures._overall_of(record, "gap")` so the gap agrees exactly with the 5rmo
panel rather than being recomputed a second way:

    python scripts/gap_time_variance.py --align cue  --variant working --unmatched
    python scripts/gap_time_variance.py --align lick --variant lick    --unmatched

Needs the LocaNMF caches and the conda env (scipy); run it on the imaging computer, not in a
bare checkout.

Still pooled over the field. For the per-node number the ranking model needs, restrict the
design matrix to one component's columns before `_pipe().fit` inside `_collect_5c`, rebuild
the paired records per component, and run `_fits` on each. That is the version whose result
should go in the grant.

## Where this shows up in the K08

`K08_Consolidated_Aims_and_Research_Strategy_v1.docx`, Approach / Shared analysis framework:
the ranking is described as a within-session contrast that time cannot confound; the gap is
entered as a deviation from each animal's own trajectory; sham animals are the time-only
null; epoch-stratified refitting is the robustness check. One clause states that in
preliminary data days since lesion accounts for little of the within-animal session-to-session
variance in the gap. **That clause rests on the provisional number above.** If the per-node
task-arm rerun disagrees, change the clause.
