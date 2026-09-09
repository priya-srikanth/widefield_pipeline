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

The "collapse of the far-target subspace" claim is the one that says something biological, so it is
worth separating from the measures that only say "changed":

* `epoch_5c_frozen_confusion_cue_working`, acute-minus-pre panel. The diagonal loss is deepest at
  far-contra (~-0.55 recall), and the recall it loses reappears as INCREASED off-diagonal recall in
  the far-ipsi and far-middle COLUMNS of that row. Errors go to other FAR targets, not to near ones
  and not uniformly.
* `epoch_10bdelta_best_match_by_position_cue_working`: the fraction of sessions where far-contra's
  post-stroke pattern still best-matches its own pre-stroke pattern falls by 0.94 acutely.
* Pooled frozen-decoder accuracy: pre 0.89 (n=21,017 trials) -> acute 0.52 (7,354) -> subacute 0.77
  (10,691) -> chronic 0.87 (1,067).

What this does NOT establish: which position far-contra moved TOWARD in the representational space,
as opposed to which label the decoder assigns. `_matrices_crossnobis(row_centre=True)` is the tool
for that question -- row-centring removes the amplitude term that shifts a whole row together, and is
the difference between "did this position move" and "which position did it move toward". That
analysis has not been run per-epoch; the substitution claim above rests on decoder confusions and
best-match fraction, both of which are label-level.

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
