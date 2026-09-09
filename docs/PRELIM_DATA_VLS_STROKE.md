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
> uniform degradation. Three measures indicate that the affected code is relocated rather than lost.
> Within-session split-half pattern reliability is preserved at the most impaired position (+0.12 at
> far-contralateral, interval excluding zero), so the post-stroke representation remains as
> internally repeatable as the pre-stroke one even as the frozen decoder fails on it; a similarity
> drop that reliability cannot explain is a moved code, not a noisier one. Cross-validated crossnobis
> distance between each position's post-stroke pattern and its own pre-stroke pattern -- normalised
> so 1.0 equals the separation between two DIFFERENT pre-stroke positions -- rises acutely to +1.02
> at far-contralateral, and the fraction of sessions in which that position's pattern still
> best-matches its own pre-stroke pattern falls by 0.94. A ridge encoder mapping position to
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

### Direction of the acute move

`epoch_8rc_matrices_crossnobis_rowcentred_cue_working`, acute-minus-pre panel: the far-contra row
rises against its own pre-stroke column and FALLS against the ipsilateral columns -- the impaired
target's pattern moves toward ipsilateral target representations. Consistent with the decoder
confusions, which send far-contra's lost recall to far-ipsi and far-middle
(`epoch_5c_frozen_confusion_cue_working`, acute-pre panel), and with best-match fraction falling
0.94 (`epoch_10bdelta_best_match_by_position_cue_working`).

### What this does NOT establish

Direction is read off the acute-minus-pre heatmap, not from a per-cell interval. The claim
"toward ipsilateral" is supported by two independent measures pointing the same way, but neither
carries a significance test on the OFF-diagonal cells. A per-cell interval on the row-centred
off-diagonal would settle it.

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
| pre-stroke frozen decoder accuracy | run log, `frozen decoder [.]` | cue .87/.78/.93/.92; precue .47/.46/.66/.45; lick .92/.88/.95/.93 |
| no-lick dissociation | run log, `=>` verdicts | PS95 pre-cue 0.36 vs post-cue 0.09 |

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
6. **Seven placed figures were stale at deck build** (`section_g_smalllesion_*`,
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
