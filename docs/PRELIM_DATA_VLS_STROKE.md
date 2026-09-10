# Preliminary data: what VLS stroke does to cortical position coding

**Status: DRAFT for a career-development grant. Numbers are POINT ESTIMATES READ OFF THE POOLED
EPOCH FIGURES, not pulled from the underlying JSON. Pull exact values and intervals before
submission.** Source run: `nightly_figs 20260907`, completed 2026-09-08 15:13, 0 failed steps,
96/96 grant units. Figures under `labcams/grant_figures/epoch/`.

Written 2026-09-09 at Priya's request, so the reasoning behind the paragraph is recoverable rather
than living in a chat log.

---

## The paragraph

> Widefield calcium imaging of dorsal cortex during a six-position mobile-spout licking task,
> decomposed into localised components by LocaNMF, shows that unilateral left ventrolateral striatal
> (VLS) stroke degrades the cortical representation of reach target by collapsing its spatial
> specificity rather than abolishing it. An L2-regularised multinomial logistic-regression decoder
> (C = 0.5) trained on z-scored component time courses and frozen on each animal's pre-stroke
> sessions falls from 0.89 to 0.52 accuracy acutely (chance 0.167; N = 4 animals, 7,354 acute vs
> 21,017 pre-stroke trials), with a spatial gradient that tracks the behavioural deficit: -0.24 at
> the near-ipsilateral spout versus -0.58 at the far-contralateral spout. Critically, the resulting
> errors are structured rather than random -- far-contralateral trials are preferentially
> misclassified as OTHER FAR TARGETS (far-ipsilateral, far-middle) rather than as near targets, so
> the acute deficit is a loss of discriminability within the far-target subspace rather than a
> uniform degradation. Three measures indicate that the affected code is PARTLY relocated rather than
> simply lost (a within-session refit recovers a third of the far-contralateral deficit and under
> an eighth of the far-ipsilateral one -- see the frozen-vs-refit section below before quoting
> "relocated" unqualified).
> Within-session split-half pattern reliability is preserved at the most impaired position (+0.12 at
> far-contralateral, interval excluding zero), so the post-stroke representation remains as
> internally repeatable as the pre-stroke one even as the frozen decoder fails on it; a similarity
> drop that reliability cannot explain is a moved code, not a noisier one. Cross-validated crossnobis
> distance between each position's post-stroke pattern and its own pre-stroke pattern -- normalised
> so 1.0 equals the separation between two DIFFERENT pre-stroke positions -- rises acutely to +1.02
> at far-contralateral, and the fraction of sessions in which that position's pattern still
> best-matches its own pre-stroke pattern falls by 0.92. A ridge encoder mapping position to
> component activity loses 0.94 of its pre-stroke explained variance acutely, roughly half
> attributable to response gain and half to pattern shape, indicating that the change is not simple
> amplitude scaling. The deficit is also non-uniform across the trial: pre-cue (ENL) decoding falls
> far less than post-cue decoding and does so evenly across positions (-0.11 to -0.26 versus -0.24 to
> -0.58), and on post-stroke trials with no detected lick the pre-cue code survives above chance
> while the post-cue code does not. Decoding recovers substantially by the subacute period (0.77) and
> approaches baseline chronically (0.87), yet the geometric displacement persists. Together these
> data support a model in which VLS stroke spares target selection while disrupting the
> transformation from selected target to executed movement, and in which behavioural recovery
> proceeds on a reorganised rather than a restored cortical code.

## HOW the code changed -- the mechanistic claim and its evidence

Measured post-cue, lick + miss-while-working, 2026-09-09. **Exact values, not heatmap readings.**

### Crossnobis own-position diagonal, RAW vs ROW-CENTRED

Row-centring subtracts each row's own mean. It matters because
`d(post P, pre Q) = |mu_postP|^2 - 2 mu_postP . mu_preQ + |mu_preQ|^2` has a first term depending
ONLY on P: a change in the magnitude of P's post-stroke response shifts its distance to every
pre-stroke position equally. Raw therefore mixes "moved" with "got bigger/smaller"; row-centred
isolates the within-row contrast, which is where a substitution lives.

| position | ROW-CENTRED acute-pre | ROW-CENTRED chronic-pre | RAW acute-pre | RAW chronic-pre |
|---|---|---|---|---|
| near ipsi   | +0.463 | **-0.032** | +0.211 | **+0.862** |
| near middle | +0.188 | +0.107 | +0.008 | +0.400 |
| near contra | +0.455 | **-0.063** | +0.245 | **+0.776** |
| far ipsi    | +0.421 | +0.061 | +0.304 | +0.220 |
| far middle  | +0.272 | **+0.010** | +0.319 | **+0.845** |
| **far contra** | **+0.665** | **+0.177** | **+1.017** | +0.264 |

Three readings, in order of confidence:

1. **ACUTELY THE PATTERN GENUINELY MOVES, most at the impaired position.** Row-centred far-contra
   +0.665 against +0.188 to +0.463 elsewhere; raw far-contra +1.017, i.e. as far from its own
   baseline as two DIFFERENT pre-stroke positions are from each other.
2. **CHRONICALLY THE PATTERN RESOLVES EVERYWHERE EXCEPT THE IMPAIRED POSITION.** Row-centred
   returns to ~0 at near-ipsi (-0.032), near-contra (-0.063) and far-middle (+0.010), but far-contra
   remains +0.177 (95% interval excludes zero on `epoch_8rcdiagdelta_...`).
3. **THE LARGE CHRONIC RAW VALUES ARE AMPLITUDE, NOT REORGANISATION.** Raw is most elevated
   chronically at SPARED positions -- near-ipsi +0.862, far-middle +0.845, near-contra +0.776 --
   exactly where row-centred is ~0. A global gain change moves every row; only row-centring
   separates it. This is also why the deck's old claim that crossnobis is "immune to uniform
   amplitude change" was wrong and has been corrected.

### GAIN vs MOVE: the two are separated in TIME, not mixed

Measured 2026-09-09 from `_matrices_crossnobis` / `_matrices_crossnobis_rowcentred`, post-cue,
lick + miss-while-working, pooled as a mean over sessions.

Row-centring is an EXACT decomposition, not an approximation. Because
`rowcentred[i,j] = raw[i,j] - mean_j raw[i,:]`, the own-position distance splits into two terms
that sum back to it:

    raw_diag[i]  =  rowmean[i]         +  rc_diag[i]
                    ^GAIN               ^MOVE
                    position-NONspecific  position-SPECIFIC
                    (shifts the whole row) (which column it moved toward)

Verified numerically: max |raw_diag - rowmean - rc_diag| = 5.6e-16 in every epoch.

Change from pre-stroke in each term:

| position | acute ΔGAIN | acute ΔMOVE | subacute ΔGAIN | subacute ΔMOVE | chronic ΔGAIN | chronic ΔMOVE |
|---|---|---|---|---|---|---|
| near ipsi   | **-0.252** | +0.463 | +0.540 | +0.056 | **+0.894** | **-0.032** |
| near middle | -0.180 | +0.188 | +0.239 | +0.127 | +0.293 | +0.107 |
| near contra | **-0.211** | +0.455 | +0.334 | +0.098 | **+0.839** | **-0.063** |
| far ipsi    | -0.117 | +0.421 | +0.112 | +0.163 | +0.159 | +0.061 |
| far middle  | +0.047 | +0.272 | +0.309 | +0.099 | **+0.835** | **+0.010** |
| **far contra** | +0.352 | **+0.665** | +0.117 | **+0.296** | +0.087 | **+0.177** |

**ACUTE IS A MOVE, NOT A GAIN CHANGE.** ΔMOVE is positive at all six positions (+0.19 to +0.67)
while ΔGAIN is NEGATIVE at four of six. Whatever the acute lesion does, it is not turning the
response volume down uniformly -- if anything the non-specific term shrinks.

**CHRONIC IS A GAIN CHANGE, NOT A MOVE.** The sign flips: ΔGAIN reaches +0.89 / +0.84 / +0.84 at
near-ipsi, near-contra and far-middle while ΔMOVE at those same positions is -0.03 / -0.06 / +0.01.
This is the whole reason the RAW chronic diagonal looks alarming at SPARED positions -- it is
reading a global amplitude change as though it were reorganisation.

**SUBACUTE IS THE CROSSOVER**, with both terms present and neither dominant.

**FAR-CONTRA IS THE EXCEPTION IN BOTH DIRECTIONS.** It is the only position whose ΔGAIN stays small
throughout (+0.35 / +0.12 / +0.09) and the only one whose ΔMOVE never returns to zero
(+0.665 -> +0.296 -> +0.177). The impaired target's deficit is position-specific at every epoch;
the spared positions' chronic change is not position-specific at all.

Equivalently, in terms of the OWN-POSITION ADVANTAGE (`rc_diag`, negative = closer to its own
pre-stroke pattern than to the average pre-stroke pattern), far-contra runs
pre **-0.831** -> acute **-0.166** -> subacute **-0.535** -> chronic **-0.654**: acutely the
advantage is all but abolished, and it never fully returns.

### Direction of the acute move

`epoch_8rc_matrices_crossnobis_rowcentred_cue_working`, acute-minus-pre panel. Exact far-contra row,
acute minus pre (negative = moved TOWARD that pre-stroke position):

| toward | nI | nM | nC | fI | fM | fC (own) |
|---|---|---|---|---|---|---|
| acute    | **-0.410** | -0.062 | +0.350 | **-0.466** | -0.076 | **+0.665** |
| subacute | -0.052 | -0.039 | +0.088 | -0.215 | -0.078 | +0.296 |
| chronic  | -0.157 | -0.040 | -0.038 | +0.052 | +0.006 | +0.177 |

The two columns it moves toward acutely are both IPSILESIONAL-side targets (far-ipsi -0.466,
near-ipsi -0.410); it moves AWAY from near-contra (+0.350) and from its own pre-stroke pattern
(+0.665). By chronic only the near-ipsi pull survives (-0.157). Consistent with the decoder
confusions, which send far-contra's lost recall to far-ipsi and far-middle
(`epoch_5c_frozen_confusion_cue_working`, acute-pre panel), and with best-match fraction falling
0.94 (`epoch_10bdelta_best_match_by_position_cue_working`).

### What this does NOT establish

The off-diagonal values above are exact point estimates, but they carry NO per-cell interval. The
claim "toward ipsilateral" is supported by three measures pointing the same way (row-centred
off-diagonal, decoder confusions, best-match fraction) and by no significance test on any
off-diagonal cell. A per-cell interval on the row-centred off-diagonal would settle it.

Epoch coverage for the tables above is NOT balanced across animals: acute = PS94 6 / PS92 5 /
PS93 4 / PS95 1; subacute = PS95 10 / PS93 7 / PS94 5 / PS92 2; **chronic = PS92 4, one animal**.
Pooling is a mean over sessions, so subacute leans on PS95 and chronic is a single-animal claim.

## LOST or MISREAD? The frozen-vs-refit arm (figures 5r), 2026-09-09

> **SESSION-SET CAVEAT, added 2026-09-09 late.** Every table below, and the crossnobis tables above,
> were computed on the 0606-0907 session set. `PS92_0908` (day 22, CHRONIC) and `PS93_0908` (day 22,
> SUBACUTE) were registered by the poller and analysed afterwards, and they are now in the pooled
> bundle. The ACUTE numbers are unaffected -- neither session is acute -- but subacute and chronic
> have shifted in the third decimal (post-cue far-contra recovery subacute +0.130 -> +0.136, pooled
> subacute refit 0.764 -> 0.767). Regenerate with `scripts/prelim_numbers_frozen_vs_refit.py` and
> `scripts/prelim_numbers_crossnobis.py` before quoting a subacute or chronic value, and re-render
> the epoch figures so the deck agrees with the text.


The "relocated rather than lost" claim above rested on three INDIRECT measures -- split-half
reliability, crossnobis displacement, best-match fraction. `epoch_5rgap_frozen_vs_refit_*` tests it
directly: refit a decoder WITHIN each session on the SAME trials, same estimator, same block
grouping, and ask whether the position becomes decodable again. Post-cue, lick + miss-while-working.

    gap ~ 0, both arms low  ->  the code is degraded; no model recovers it
    gap > 0                 ->  the code is present and DISPLACED; only the frozen readout fails

**THE PRE PANEL IS NOT ZERO AND IS NOT AN EFFECT.** The frozen arm trains on ten pre-stroke sessions
and the refit arm on one, so refitting COSTS 0.073 accuracy at baseline (frozen 0.886 vs refit
0.813). Every number below is the gap at that epoch MINUS the pre gap.

| epoch | frozen | refit | gap | gap - pre gap |
|---|---|---|---|---|
| pre | 0.886 | 0.813 | -0.073 | -- |
| acute | 0.525 | 0.552 | **+0.027** | **+0.100** |
| subacute | 0.772 | 0.764 | -0.008 | +0.065 |
| chronic | 0.846 | 0.841 | -0.005 | +0.068 |

The sign of the gap FLIPS acutely: post-stroke, refitting stops costing accuracy and starts buying
it. Per position, acute (gap minus pre gap), against each position's own frozen deficit:

| position | frozen deficit (acute - pre) | recovered by refitting | fraction recovered |
|---|---|---|---|
| near ipsi   | -0.234 | +0.027 | 12% |
| near middle | -0.273 | +0.132 | 48% |
| near contra | -0.313 | +0.168 | 54% |
| far ipsi    | -0.373 | +0.034 | **9%** |
| far middle  | -0.408 | +0.044 | **11%** |
| **far contra** | **-0.573** | **+0.196** | 34% |

**THIS QUALIFIES THE "MOVED NOT LOST" CLAIM RATHER THAN CONFIRMING IT.** Three readings:

1. **The displaced component is REAL and largest in absolute terms at the impaired position**
   (+0.196 at far-contra, the largest of the six), which is where crossnobis displacement is also
   largest. Two independent methods agree on where the code moved.
2. **It is a MINORITY of the deficit.** A third of far-contra's acute drop is recovered by
   refitting; two thirds is not. The paragraph's "relocated rather than lost" overstates it --
   the defensible claim is "partly relocated, mostly lost, and the relocated part is
   position-specific."
3. **The far ipsilateral and far middle positions lose information OUTRIGHT** -- 9% and 11%
   recovered, the two smallest fractions. Their deficits are not a readout problem at all. So the
   acute lesion does two different things at once, and which one dominates depends on the position.

The gap remains positive subacutely (+0.065) and chronically (+0.068) even as frozen accuracy
returns to 0.846, so a residual readout mismatch outlives the behavioural recovery -- consistent
with the chronic row-centred far-contra displacement of +0.177.

### The cell that is NOT drawn, and why

Acute far-contralateral is GATED OUT of the lick-aligned arm. That arm conditions on a detected
lick, and acutely far-contra is the spout the animal does not lick: every acute session holds it at
0.0-4.3% of trials against a pre-stroke 16.5%. A within-session refit will not predict a class at
4% prior in a six-way problem, so the cell read -0.42 -- the mouse not licking, presented as the
code being gone, in the direction that would have flattered the "lost" reading. Gated by
`grant_figures.MIN_REFIT_SHARE` (a third of uniform); the gate fires on 78 of 37,562 trials, 0.21%,
all of them that one cell, and never in the cue or pre-cue arms. A count-only floor did NOT catch it
and made it worse -- see the constant's own note.

## THE POSITIONS STOP BEING DIFFERENT FROM EACH OTHER (split-half off-diagonal), 2026-09-10

Priya, reading the deck: "after stroke, split-half similarity suggests significantly more similarity
between R/L/center spout position brain activity compared to pre-stroke, when these were more
different than each other." That reading is correct, and it is the most direct statement of the
effect in the whole figure set.

`_split_half_matrix` splits each position's trials into halves WITHIN one session. The diagonal is
that position's own reliability -- corr(half A at P, half B at P) -- and the OFF-diagonal is how
similar two DIFFERENT positions look, measured on independent halves so no cell is a mean correlated
with itself. Post-cue, lick + miss-while-working:

| epoch | own-position (diagonal) | between-position (off-diagonal) | separation |
|---|---|---|---|
| pre | 0.791 | **-0.169** | 0.959 |
| **acute** | 0.724 | **+0.091** | **0.632** |
| subacute | 0.785 | -0.133 | 0.918 |
| chronic | 0.858 | -0.180 | 1.038 |

**EACH POSITION STAYS ABOUT AS REPEATABLE AS BEFORE WHILE THE POSITIONS STOP BEING DISTINGUISHABLE
FROM EACH OTHER.** The diagonal moves 0.791 -> 0.724, a 0.067 drop. The off-diagonal moves -0.169 ->
+0.091, a swing of +0.260 -- four times larger, and in the direction that says different targets now
evoke the same pattern. A code that had gone NOISY would show the opposite: diagonal collapsing,
off-diagonal unchanged.

The far-contralateral row is where it happens. Its correlation with the other far positions, acute
minus pre: far-ipsi -0.24 -> **+0.58**, far-middle +0.15 -> **+0.63**. The impaired target's pattern
does not become noise; it becomes the OTHER far positions' pattern.

**FOUR MEASURES, ONE EVENT -- and "independent" needs qualifying.** The split-half off-diagonal
(patterns merge), the frozen decoder's confusions (far-contra misread as far-ipsi/far-middle), the
row-centred crossnobis (far-contra moves toward ipsilesional targets) and the best-match fraction
(its nearest pre-stroke neighbour stops being itself) are four DIFFERENT quantities and they agree.
They are analytically independent -- different metric, different reference frame -- but NOT
statistically independent: all four are computed from the same per-trial joint-LocaNMF features, the
same engagement gate and the same trial sets, so a fault in those would move all four together. The
convergence rules out four different analysis choices, not one bad feature matrix.

It reverses by subacute: off-diagonal back to -0.133, and to -0.180 chronically, slightly BELOW its
own pre-stroke value.

### Split-half off-diagonal vs row-centred crossnobis: related, not redundant

Priya, 2026-09-10: "are these essentially the same?" No -- they agree on the acute event and
disagree in ways that identify what each one measures.

| epoch | Spearman rho, 30 off-diagonal cells | mean delta, split-half | mean delta, row-centred crossnobis |
|---|---|---|---|
| acute | **+0.711** | **+0.260** | **+0.082** |
| subacute | +0.472 | +0.036 | +0.028 |
| chronic | -0.115 | -0.012 | +0.009 |

Three differences, and each one is the reason to keep both:

1. **REFERENCE FRAME.** Split-half lives entirely inside ONE session: both halves come from the same
   session, so it asks "are these positions distinguishable right now". The crossnobis matrices are
   POST x PRE, so every cell carries the pre-stroke geometry and asks "which position did this one
   move TOWARD". A code that scrambled into a brand-new configuration would show up in the first and
   not the second.
2. **WHAT ROW-CENTRING REMOVES.** Row-centring subtracts each row's mean, which removes anything
   that shifted the whole row -- including the amplitude term. Split-half has no such term to
   remove.

   **CORRECTION, 2026-09-10.** An earlier version of this section said the +0.260 vs +0.082
   difference meant "roughly two thirds of the merging is a COMMON shift and one third is the
   differential substitution". That was wrong and should not be quoted. The two numbers are a
   CORRELATION change and a NORMALISED DISTANCE change: different units, no reason for their
   magnitudes to be commensurate, and no ratio to take between them. Priya caught it by asking the
   right question -- split-half is a Pearson correlation, so it is already fully blind to gain,
   global or per position, and cannot contain an amplitude component for row-centring to be
   removing. The defensible comparison between the two measures is their RANK agreement, not their
   magnitudes.

   **AND ROW-CENTRED CROSSNOBIS IS NOT FULLY GAIN-BLIND EITHER.** Writing the row out,

       d(P,Q) - mean_Q d(P,.) = -2 mu_postP . (mu_preQ - mean mu_pre) + (|mu_preQ|^2 - mean |mu_pre|^2)

   the first term scales LINEARLY with |mu_postP| and the second does not depend on P at all. So
   row-centring removes gain from the row's OFFSET but leaves it multiplying the row's SHAPE: double
   P's response and the whole row-centred profile doubles. Row-centred values are therefore
   comparable in SIGN and in RANK across epochs but not in MAGNITUDE when amplitude has changed --
   and the encoder says it changed a lot (fitted gain a: 0.94 pre to 0.36 acute, post-cue). Dividing
   each row by its own SD across columns would make it scale-free as well; that is not currently
   done.
3. **METRIC.** Pearson correlation is gain-blind by construction; crossnobis is a noise-normalised
   distance and is gain-SENSITIVE, which is the whole reason the row-centred family had to be built.

They dissociate exactly where that matters. Far-contra against NEAR-contra, acute minus pre:
split-half **+0.152** (slightly more alike) but row-centred crossnobis **-0.350** (moved AWAY).
Same trials, opposite sign, because one is absolute similarity within the session and the other is
similarity relative to that row's own mean.

Where they DO agree they agree completely: the far-contra row's five off-diagonal columns come out in
IDENTICAL rank order under both measures (rho = +1.000) -- far-ipsi > near-ipsi > far-middle >
near-middle > near-contra. That is the substitution claim, made twice from different arithmetic.



### Why a gain change cannot explain the similarity drop

`_corr_matrix` uses `np.corrcoef`, i.e. Pearson, which centres and scales each pattern vector. A
uniform gain change on all components leaves *r* exactly unchanged -- so a drop in mean-pattern
similarity is a change in the SHAPE of the pattern across components, never its size. The inverse is
the one to watch: a LINEAR DECODER is sensitive to gain (fixed hyperplane, fixed intercepts), so
amplitude can break decoding while correlation holds. Anyone reasoning "the components still decode,
so the pattern must be intact" has it backwards.

Nor is it attenuation. Split-half reliability falls only 0.791 -> 0.724 acutely, so the most
attenuation can account for is a factor sqrt(0.724 / 0.791) = 0.957: it would take the mean-pattern
similarity from 0.741 to 0.709, not to the observed **0.350**. That is also what the disattenuated
third panel of `grant_7b_reliability_*` shows cell by cell.

## Where each number comes from

| claim | figure | value |
|---|---|---|
| post-cue decoder Δ, acute | `epoch_accdelta_by_position_cue_working` | nI −0.24, nM −0.28, nC −0.31, fI −0.38, fM −0.41, **fC −0.58** |
| post-cue decoder Δ, subacute | same | nI −0.03, nM −0.15, nC −0.03, fI −0.13, fM −0.12, fC −0.22 |
| post-cue decoder Δ, chronic | same | nI +0.07, nM −0.01, nC +0.03, fI −0.06, fM −0.02, fC −0.13 |
| pre-cue decoder Δ, acute | `epoch_accdelta_by_position_precue_working` | nI −0.23, nM −0.11, nC −0.23, fI −0.16, fM −0.13, fC −0.26 |
| encoder Δ (variance, gain) | `epoch_11delta_encoder_gain_shape_cue_working` | acute −0.94 / −0.45; subacute −0.34 / −0.26; chronic −0.05 / −0.03 |
| within-session reliability Δ | `epoch_7diagdelta_matrices_splithalf_cue_working` | acute nI −0.12, nM −0.33, nC −0.07, fI +0.01, fM −0.03, **fC +0.12** |
| crossnobis displacement | `epoch_8diagdelta_matrices_crossnobis_cue_working` | acute fC **+1.02**; chronic nI +0.98, nC +0.86, fM +1.01 |
| best-match DESTINATION, acute fC | `epoch_10c_matrices_best_match_destination_cue_working` | fM 0.50, fI 0.25, nI 0.12, nC 0.06, itself 0.06 |
| best-match fraction fC, pre -> acute | `epoch_10cdiagdelta_...` | 0.98 -> 0.06 (a drop of **0.92**, not 0.94 -- see below) |
| recoverable, MATCHED training sets | `epoch_5rmgapdelta_frozen_vs_refit_cue_working` | pre gap **+0.090**; acute fC **+0.158**, nC +0.128, nI +0.093 |
| pre-stroke frozen decoder accuracy | run log, `frozen decoder [.]` | cue .87/.78/.93/.92; precue .47/.46/.66/.45; lick .92/.88/.95/.93 |
| no-lick dissociation | run log, `=>` verdicts | PS95 pre-cue 0.36 vs post-cue 0.09 |
| refit-minus-frozen gap | `epoch_5rgapdelta_frozen_vs_refit_cue_working` | pre -0.073; acute +0.027, subacute -0.008, chronic -0.005 |
| refit recovery by position, acute | same | nI +0.03, nM +0.13, nC +0.17, fI +0.03, fM +0.04, **fC +0.20** |

Epoch n: pre 44 sessions, acute 16, subacute 23, **chronic 3**.

## Sign conventions that are easy to get wrong

* **`_matrices_crossnobis` returns RAW distances** normalised to pre-stroke units, so on the epoch
  figures **larger = further from baseline = more changed**.
* **`_mats_crossnobis` (figure 8d) NEGATES them** (`sign=-1`) so that "larger diagonal = more
  preserved", matching the correlation figures. The two are one letter apart in the name and carry
  opposite signs. Check which one a figure used before describing its direction.
* The matrices are **post x pre cross-matrices**, so the diagonal is position *i* post against
  position *i* pre -- not a within-session RDM.

## Caveats that must survive into any submitted version

1. **The chronic epoch is ONE ANIMAL** (PS92, 3 sessions). The most quotable claim -- geometry stays
   displaced while behaviour recovers -- is the least supported. Either restrict it to subacute or
   state n in the text.
2. **Pre-cue is not "spared".** Its baseline is only ~0.45-0.66, so −0.26 at far-contra leaves it
   near chance (0.167). The defensible claim is about the *shape* of the loss (flat across positions
   vs graded), not its absence.
3. **Reliability preservation is not uniform.** It holds at far-contra (+0.12) but near-middle
   drops −0.33 acutely. The "moved not lost" argument is strongest exactly at the impaired position,
   which is convenient -- and therefore worth stating precisely rather than generalising.
4. **The no-lick dissociation is basis-dependent.** It holds under both the Allen-ROI and joint
   LocaNMF bases for PS92 and PS95; PS93 and PS94 flip to "no clear dissociation" depending on basis
   and RT cut. The run flags `BASES DISAGREE`. Quote it as a per-animal result, not a cohort one.
5. **Several intervals cross zero** (subacute near-ipsi and near-contra decoder). The paragraph reads
   as more uniform than the intervals support.
6. **"Relocated rather than lost" is now QUALIFIED by the refit arm.** A within-session refit
   recovers about a third of the acute far-contralateral deficit and under an eighth of the
   far-ipsilateral and far-middle ones. The displacement is real, position-specific and agrees with
   the crossnobis geometry on WHERE, but most of the acute deficit is information the population no
   longer carries linearly. Do not quote "relocated" without it.
7. **Seven placed figures were stale at deck build** (`section_g_smalllesion_*`,
   `poststroke_G7d_smalllesion_*`, `coding_rtdrift`; 17-21 days old, predating both engagement-gate
   changes). None are cited above, but do not cite them until re-run.

## Engagement gate

All post-stroke numbers above use the reference-restricted, backdated gate
(`precue_engagement_states.engagement_gate`): judged only at close_L / close_center, requiring a
non-recovering collapse, backdated to the start of the run of misses that trips it. This matters
because the previous position-blind gate discarded far-position motor failures as "disengagement" --
i.e. deleted the effect as the confound. See `docs/STATUS_2026-09-07.md` and
`tests/test_one_engagement_gate.py`.

---

# A manipulation hypothesis these data motivate

Priya, 2026-09-09: "what might be a reasonable hypothesis to test how we can manipulate
post-stroke recovery? eg DREADD- or optogenetic manipulation of activity in contralateral striatum
vs ipsi/contra orofacial motor cortex in subacute post-stroke?" Recorded as a FRAMING to argue
with, not a recommendation -- the design choices are the lab's.

## The question the data poses

The impaired target's pattern moves TOWARD ipsilateral target representations
(`epoch_8rc_matrices_crossnobis_rowcentred_cue_working`, acute-minus-pre). Two readings make
OPPOSITE predictions, which is what makes it worth an experiment:

* **MALADAPTIVE CAPTURE.** The intact contralesional hemisphere captures the impaired target's
  code and the drift IMPEDES recovery. Rodent analogue of the interhemispheric-rivalry rationale
  behind contralesional low-frequency rTMS in human stroke.
* **COMPENSATORY.** The drift IS the recovery -- the intact circuit takes the target over.

## What the data already says, and how weakly

PS92 -- the only animal to reach chronic -- recovered to 378/378 hits while retaining a residual
far-contra row-centred displacement of +0.177. A displaced-but-functional code is what the
COMPENSATORY account predicts. This is n = 1 and cannot separate "the displacement is the
mechanism" from "the displacement is a harmless scar".

## Predictions by target, subacute window

| target | if MALADAPTIVE | if COMPENSATORY |
|---|---|---|
| contralesional (R) orofacial M1, inhibit | far-contra displacement falls, hit rate rises | displacement rises, recovery stalls |
| perilesional (L) orofacial M1, excite | contra-specific drive restored; displacement falls | little effect, or gain-only change |
| contralesional (R) VLS, inhibit | as R-M1 but slower onset | recovery blocked if the striatal route carries it |

## Two design points the preliminary data adds

1. **USE THE IMAGING READOUT AS THE DEPENDENT VARIABLE, NOT HIT RATE.** Behaviour saturates: PS92
   hit 378/378 while its representation was still displaced. The row-centred far-contra
   displacement is unsaturated and position-specific, so it can detect an effect hit rate cannot.
2. **PREDICT A DISSOCIATION, NOT A GLOBAL EFFECT.** The deficit is post-cue and execution-side;
   pre-cue selection is comparatively preserved (LOSO 0.510 vs 0.873 post-cue) and the plan
   survives on failed trials. A manipulation of motor-output circuits should move the POST-cue code
   and leave the PRE-cue code alone. One that moved both would argue for arousal or engagement
   rather than the transformation -- and the engagement gate and no-lick arm already exist to catch
   exactly that confound.

## Timing

Row-centred far-contra displacement runs +0.665 acute -> +0.30 subacute -> +0.177 chronic, so the
trajectory is set during subacute. That supports subacute as the intervention window.
