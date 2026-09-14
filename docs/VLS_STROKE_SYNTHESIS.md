# What VLS stroke does to the cortical representation of lick target — synthesis

**2026-09-13.** Every claim below names the figure that supports it. Figure files live in
`<labcams>/grant_figures/epoch/` (the `epoch_*` families, deck section I) and
`<labcams>/grant_figures/` (the `grant_*` families, deck section H); each has a `.csv` value sidecar
beside it holding the numbers quoted here, and a `_stats.csv` for the map families. Numbers were
pulled from those sidecars, not from prose — a caption on this deck was found today quoting
0.86 → 0.70 where its own sidecar said 0.97 → 0.84.

---

## The paragraph

**A ventrolateral striatal lesion selectively degrades the cortical representation of lick TARGET
while leaving the cortical representation of behavioural STATE substantially intact, and the loss is
anatomically specific, graded by target eccentricity, and largely reversible.** Behaviourally, the
lesion produces a position-graded deficit that recovers over ~2 weeks
(`epoch_1c_behaviour_timecourse.png`, `epoch_2_behaviour_by_position.png`,
`epoch_3_behaviour_delta.png`). A decoder frozen on pre-stroke cortex loses **half** of its
above-chance spout-position accuracy acutely — 0.863 → **0.428** as a fraction of pre-stroke
above-chance performance — and recovers to 0.698 subacutely and 0.789 chronically
(`epoch_13n_state_vs_position_cue.csv`; the full per-epoch decoding families are `epoch_4*` and
`epoch_5r*`). **The same undamaged cortex, the same window, the same LocaNMF basis, the same frozen
estimator, with only the LABEL changed, loses one fifth as much**: a three-way behavioural-state
decoder (rest / running / licking) retains 0.902 of its pre-stroke above-chance performance acutely
against position's 0.496, and running alone loses 3%
(`epoch_13_state_decoder_cue.png`, `epoch_13pos_state_decoder_by_class_cue.png`,
`epoch_13n_state_vs_position_cue.png`). Because the lesion is striatal and no cortex is damaged,
this rules out the generic explanations at once — window clouding, haemodynamic drift, arousal,
basis drift — since every one of them would degrade the state readout too. An encoder says the same
thing in components rather than in accuracy: the fitted amplitude factor *a* goes **0.943 pre →
0.361 acute → 0.677 subacute → 0.943 chronic** (`epoch_11amp_encoder_amplitude_cue_working.csv`),
i.e. acute cortex retains ~38% of its pre-stroke drive and fully recovers. At the level of cortical
maps the loss is not uniform across targets: referenced to the time-local rest baseline, acute map
amplitude relative to each position's own pre-stroke value is **1.40 / 1.15 / 1.21** for the three
NEAR positions and **0.94 / 0.64 / 0.44** for the three FAR ones, with **far-contralateral the only
position whose change is both large and spatially specific — 1,211 of 2,022 tested bins**
(`epoch_15r_position_RESTref_cue_working.png` + `_stats.csv`). Far-contralateral and far-middle,
the two positions the behaviour implicates, are the two the maps implicate. All four animals make
the acute far-contralateral change in the same spatial shape, against a null that re-pairs animals
ACROSS positions and so preserves anatomy, warp and mask while destroying only position-specific
agreement (`docs/WHERE_THE_CODE_MOVES.md`; a mandatory null here, because pre-stroke maps already
correlate between animals at r ≈ 0.88).

---

## What supports what

| claim | figure | sidecar / number |
|---|---|---|
| position-graded behavioural deficit, recovering | `epoch_1c_behaviour_timecourse.png`, `epoch_2_*`, `epoch_3_*` | deck I1–I3 |
| frozen position decoding collapses acutely, recovers | `epoch_13n_state_vs_position_cue.png`; `epoch_4*`, `epoch_5r*` | 0.863 → 0.428 → 0.698 → 0.789 |
| state decoding is largely preserved | `epoch_13_state_decoder_cue.png` | 0.982 → 0.918 → 0.886 → 0.952 |
| …but DOES fall, at every epoch | `epoch_13delta_state_decoder_cue.csv` | −0.063 / −0.095 / −0.029, all excluding zero after Bonferroni |
| position falls ~5× further than state | `epoch_13n_state_vs_position_cue.csv` | retained 0.496 vs 0.902 |
| per-class: rest is the class that moves | `epoch_13pos_state_decoder_by_class_cue.png` | rest 0.97 → 0.84; running 0.99 → 0.96 |
| encoder amplitude collapses and recovers | `epoch_11amp_encoder_amplitude_cue_working.png` | 0.943 → 0.361 → 0.677 → 0.943 |
| map loss is graded by eccentricity | `epoch_15r_position_RESTref_cue_working_stats.csv` | near 1.40/1.15/1.21, far 0.94/0.64/**0.44** |
| far-contra is the only spatially specific change | same | **1,211 / 2,022** bins, not suppressed |
| the same shape in all four animals | `docs/WHERE_THE_CODE_MOVES.md` | cross-position null |

---

## What is NOT established, and must travel with the paragraph

1. **THE CROSS-POSITION NULL'S REST COLUMN IS FROM A SUPERSEDED BASELINE.** The far-contra numbers
   (+0.812 observed vs +0.071 null, p = 0.001) were computed before the rest reference became
   time-local on 2026-09-13. MEAN (+0.819) and PRECUE (+0.856) are unaffected, so the finding has two
   independent references behind it, but the REST column needs re-deriving.
2. **THE AMPLITUDE GRADIENT IS A DESCRIPTION, NOT A RESULT.** Nothing has tested the near-vs-far
   contrast itself. Its precondition is now clear — block order is shuffled between sessions, so
   pooled drift cannot manufacture it (`scripts/rest_migration/block_order.py`: every position's
   mean normalised trial index 0.50 ± 0.006, mean pairwise Spearman +0.011 over 4,186 pairs) — but
   the test is per-animal amplitude ratios with the nested bootstrap on near-minus-far, unrun.
3. **REST MAY ITSELF CARRY POSITION INFORMATION.** An earlier control was read as ruling this out; it
   does not (see `scripts/rest_migration/rest_position_permutation.py` for why the magnitude-ratio
   argument fails). A circular-shift permutation that keeps the block-time structure in the NULL
   gives observed/null **1.32, 5/6 sessions** on a first pass. If that holds on the full cohort the
   rest subtrahend is not position-neutral and leaves a contaminant in every rest-referenced map.
   **This is the most important open item in this document.**
4. **THE MEAN REFERENCE COUPLES THE POSITIONS** and its near-position "increases" are partly that
   artefact — Near Middle reads 2.56 acute under MEAN
   (`epoch_15r_position_MEANref_cue_working_stats.csv`) against 1.15 under REST. Read figure 14 and
   any MEAN-referenced panel as *which positions FALL*, never *which positions change*.
5. **THE CHRONIC COLUMN OF THE CROSS-POSITION NULL IS UNINTERPRETABLE** under every reference —
   observed ≈ null ≈ +0.50, every p > 0.2 — and estimating it from 4.3× more frames did not fix it.
6. **"NO LICK DETECTED" IS NOT "NO TONGUE PROTRUSION."** The state decoder's acute errors point
   specifically at licking (rest → licking 0.143, 9.5× pre, against rest → running 0.027;
   `epoch_13c_state_confusion_cue.csv`), consistent with incomplete protrusions being filed as REST.
   DLC tongue tracking is the measurement that would settle it, and the same sensor defines the map
   rest baseline.
7. **PS94 HAS NO CHRONIC SESSIONS** and several panels are therefore N = 3 at chronic; the sidecars
   carry `n_animals` per cell.


---

## 2026-09-13 — REST probably carries position because the animal is STILL ON TASK, and the fix is a second reference, not a better baseline

Priya: *"is there a better 'quiet' we can use?? or should we do quiet per-position? It's possible
there is motor planning ongoing between trials."*

### First: the earlier control did NOT rule this out, and the new one points the other way

`rest_position_vs_drift` reported DRIFT 0.00282 against POSITION 0.00288, ratio 1.02, and that was
read as "drift aliased onto blocks, not position coding". **That reading is wrong**, for two
independent reasons recorded in `scripts/rest_migration/rest_position_permutation.py`: the two
contrasts are not matched on TIME SEPARATION (drift spans half a session, position spans ~zero,
because positions are interleaved throughout each half), and a ratio of magnitudes is not a test —
if both are noise-dominated the ratio is ~1 whatever the truth.

The proper test puts the confound INSIDE the null: circular-shift the position labels over
time-ordered rest periods, so blocks stay blocks and drift stays drift and only their
correspondence breaks. First pass, 6 pre-stroke sessions, 100 permutations each:
**observed/null = 1.32, observed > null in 5/6**. Rest carries position structure that the
block-time structure does not explain.

### Why that is EXPECTED, and continuous with a result already in the deck

**THE SPOUT IS PHYSICALLY AT THAT POSITION FOR THE WHOLE INTERVAL.** `trial_start` precedes the
position strobe by a median 0.925 s — the travel time — so a REST period, which runs from
`cue + response_window + 0.5 s` to the NEXT `trial_start`, sits entirely while the spout is still at
the JUST-COMPLETED trial's position. The animal has a spatial target in front of it throughout.

And the deck already reports the same phenomenon one step later in the trial: **pre-cue position
decoding is LOSO 0.510 against post-cue 0.873, chance 0.167** — position is well above chance
BEFORE the cue. Rest carrying position is that same signal extended earlier in the interval, not a
new or anomalous finding.

**THE SAME INTERPRETIVE LIMIT APPLIES, UNCHANGED.** CLAUDE.md records that the pre-cue signal is
deliberately NOT called a "maintained motor plan": the spout arrives ~3 s before the cue, so a
sustained SENSORY response and a held INTENTION are temporally coextensive and this design cannot
separate them. Exactly the same is true between trials. "Motor planning ongoing between trials" is
one of at least two readings, and the others — sustained sensory drive from a visible/whisker-
detectable spout, or a postural set toward it — are not excluded by anything measured here.

### So: NOT a better baseline. A SECOND reference, and the difference between them is the measurement

**A "better quiet" that removed position would be removing signal**, which is the F12 objection
appearing in a new place: the pre-cue reference was demoted precisely because subtracting a window
that carries genuine anticipatory position information measures the cue-evoked INCREMENT rather than
the position map. A per-position rest baseline does the same thing one interval earlier.

The useful construction is therefore BOTH, kept side by side:

    REST (common)         subtract the session's time-local rest baseline, IDENTICAL for all six
                          positions. Keeps any sustained between-trial position component IN the
                          map. This is what is on the deck now.
    REST (per position)   subtract each position's OWN time-local rest baseline. Removes the
                          between-trial component; what remains is the cue-evoked increment over
                          that position's own resting state.

**THE DIFFERENCE BETWEEN THE TWO MAPS IS THE BETWEEN-TRIAL POSITION SIGNAL** — i.e. the quantity
Priya's question is about, measured rather than assumed. That is a better outcome than either
reference alone, and it is why this is an addition rather than a replacement.

**BOTH STAY UNCOUPLED**, which is the property that matters against MEAN. Per-position rest gives six
different subtrahends, so the six maps are no longer on a common scale — but no position's loss can
raise another position's reference, which is the specific failure the MEAN reference has.

### What this does NOT threaten

The far-contralateral acute result does not rest on the REST reference: it replicates under MEAN
(+0.819 vs null +0.018) and PRECUE (+0.856 vs +0.083), which are constructed differently and share
no subtrahend with it. A between-trial position component would have to be far-contra-specific AND
animal-consistent to produce it, and the cross-position null is built to be blind to anything that
is not position-specific.

### TO DO

1. Finish the permutation on the full pre-stroke cohort (running; 6 sessions is a first pass).
2. Implement the per-position rest reference in `position_reference_maps` — mechanically small now
   that the baseline is time-local and applied to the SVT: build it from that position's rest
   periods instead of all of them.
3. Render both and difference them; that difference figure is the between-trial position signal.
4. Ask whether the between-trial component CHANGES post-stroke. If the position code degrades
   acutely, a between-trial position signal that degrades with it is evidence the two are the same
   representation; one that does not is evidence they are separable.
