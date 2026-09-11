# The behavioural-state control — does *everything* degrade after the lesion, or only the target?

Built 2026-09-11. Priya: *"I'm more looking for evidence that not **all** decoding/encoding degrades
post stroke, with running as an example."*

Figures `epoch_13*` (deck section I). Code: `wfield_local/locomotor_state.py`,
`locomotor_features.py`, `locomotor_decoder.py`. Tests: `tests/test_locomotor_state.py`.

---

## The argument, and why it works

**The lesion is ventrolateral STRIATAL. No cortex is damaged anywhere in the field of view.** I had
this wrong at first and proposed splitting by hemisphere to find "undamaged" cortex to compare
against; Priya: *"but cortex isn't lesioned at all, so we should be able to use whole cortex right?"*
Correct, and it makes the control **stronger** rather than weaker.

The same cortex, the same cranial window, the same LocaNMF basis, the same estimator, the same
frozen-model logic. Only the label changes — spout target, or behavioural state. If the target
readout collapses and the state readout does not, then the deficit is specific, and every generic
explanation a reader reaches for first fails at once:

| alternative explanation | why this control kills it |
|---|---|
| the window clouded over | would degrade the state readout too |
| haemodynamic / F0 drift post-surgery | same |
| the animal is differently aroused | same, and more so — state IS arousal-adjacent |
| the LocaNMF basis drifted | same, and it is one basis for both readouts |
| "you made a lesion and everything got worse" | this is the claim being tested |

## The result

Frozen pre-stroke decoders, post-cue window, same sessions:

| epoch | n | state: balanced acc (chance 1/3) | quiet | running | licking |
|---|---|---|---|---|---|
| pre | 43 | 0.945 | 0.86 | 0.98 | 0.99 |
| **acute** | 16 | **0.875** | 0.70 | **0.95** | 0.97 |
| subacute | 18 | 0.888 | 0.82 | 0.94 | 0.90 |
| chronic | 12 | 0.909 | 0.80 | **0.97** | 0.96 |

Against the frozen POSITION decoder on the same sessions: 0.89 → 0.52 → 0.75 → 0.83 (chance 1/6).

Normalised as the fraction of above-chance performance retained — the only way to compare a 6-way
problem at chance 0.167 with a 3-way at 0.333:

* **position: 0.87 → 0.42 acutely. It loses 51% of what it had.**
* **state: 0.92 → 0.81. It loses 11%.**
* **running alone: 0.98 → 0.95. It loses 3%.**

**Running is the clean example, exactly as Priya predicted: 0.98 / 0.95 / 0.94 / 0.97, flat at every
epoch.**

### One class moves, and it is probably real

**Quiet falls 0.86 → 0.70 acutely.** Do not sweep this into the pooled number. Quiet goes from 3.4%
of a pre-stroke session to 15.1% acutely, so a post-stroke animal sitting still plausibly *is* in a
different state from a pre-stroke one sitting still. That is a finding about immobility, not a
failure of the control — and it is the reason the per-class panel exists rather than only the pooled
bar.

---

## Four measurements, each of which changed the design

Every one of these was found by measuring rather than by assuming, and each would have produced a
wrong or uninterpretable figure if skipped.

### 1. The trial is not the unit

Classifying whole trials by locomotor state gives **1,795 running and 3,151 quiet trials across 107
sessions** — about 17 running trials per session, spread over six positions. That decodes nothing.
Tiling the bouts themselves gives **33,060 running and 41,549 quiet one-second segments**. The
segment is the unit; the analysis does not exist at the trial level.

### 2. The window length is set by QUIET, not chosen

| | n periods | median | p75 | p95 | max |
|---|---|---|---|---|---|
| running bouts | 6,924 | 3.73 s | 5.82 | 11.08 | 2441 |
| quiet periods | 14,017 | **1.10 s** | 1.60 | 12.48 | 370 |

| window | running bouts usable | quiet periods usable |
|---|---|---|
| 2.0 s (every other family here) | 100% | **17%** |
| **1.0 s (chosen)** | 100% | **58%** |
| 0.5 s | 100% | 100% |

A 2 s window discards 83% of quiet. One second at four 0.25 s bins gives 95 × 4 = **380 columns —
the same width as the 2 s × 0.5 s trial arms**, so the two are comparable in size. A model still
cannot be transferred between them: bin *k* means a different thing.

### 3. Licking needed the opposite fix

Lick bouts have a **median of 0.37 s**. Tiling strictly inside them keeps 22% of 101,018 bouts and
biases the class toward sustained licking — the same failure a 2 s window inflicts on quiet.

Licking is an **EVENT with a defined onset**, unlike the two sustained states. Its window is anchored
at bout onset and allowed to run past the bout's end. Quiet must **not** be: a window past its end
sits in the movement or licking the period was buffered away from, which is the one thing the class
must not contain. Per session this took licking from 72–297 segments to 413–969.

**This asymmetry is a stated limit, not a hidden one.** Licking windows are locked to a behavioural
*transition* while running and quiet are sampled from inside sustained *states*, so a decoder could
separate them partly on transient-versus-sustained rather than on which behaviour it is. The control
answers "does cortex still distinguish behavioural state at all", which is what the argument needs;
it is not a clean three-way contrast of matched epochs. It is in the figure subtitle.

### 4. One session is an artefact

**PS92 8/12's longest "running bout" is 2,441 s** — 41 minutes, 29% of the session — against a cohort
maximum of 54 s everywhere else. That is the `concat_split_session` discontinuity
(`docs/EXPERIMENT_ERRORS.md`) read as sustained locomotion. Uncapped it would supply ~7% of the
entire running class from one artefact. Excluded by name in `locomotor_state.EXCLUDE_SESSIONS`.

---

## Mutual exclusivity

Priya's rule, 2026-09-11: *"mutually exclusive groups, so running only if not also licking, licking
only if not also running, and quiet can be during ITI only"*.

The rule had to be stated because the three names do not describe one partition on their own.
Licking is a task EVENT and running is a locomotor STATE, so a trial can be both and most licking
trials are; and `quiet` as `behavior_events` defines it is already *slow treadmill AND
not-near-lick/reward, buffered*, so no licking window can ever be quiet.

    LICKING   a lick bout onset, and NO running bout overlapping the window
    RUNNING   inside a running bout, and NO lick in the window
    QUIET     CONTAINED in a behavior_events quiet period (containment, not overlap — quiet
              asserts an absence, so a window half outside one is not quiet)
    dropped   a lick AND running together: counted and reported, never assigned by tie-break

`_class_select` is the single implementation. There were four hand-written copies of the trial-class
rule, and every one of them included the engaged (licking) rows unconditionally — correct for `lick`
and `working`, and catastrophic for a class defined as "the animal had quit". That is what
`tests/test_locomotor_state.py` pins.

## Why segments are capped, and why the cap is not enough

The distribution is skewed and the two classes are skewed differently:

| | top 1% of periods supply | top 5% supply |
|---|---|---|
| running | 12.7% of segments | 23.7% |
| **quiet** | **36.6%** | **70.0%** |

Untiled and uncapped, five percent of quiet periods — the long inactive stretches, up to 370 s —
would supply seventy percent of the class. Those are not ITIs. `MAX_SEGMENTS_PER_PERIOD = 8` bounds
any single period's vote.

**The cap does not make segments independent.** Only clustering the resampling by PERIOD does that,
the same rule the position families apply to trials inside a scheduler block. `period_id` is
returned alongside for exactly that reason, and is made unique across the three classes so a caller
cannot merge a running bout with a same-numbered quiet period.

---

## Two confounds checked before anything was plotted

**Is it reading session TIME rather than cortex?** Quiet periods could cluster late, when the animal
is sated, and cortical signals drift across a session. Tested directly: a classifier on segment time
alone gives AUROC **0.165–0.752** — near chance — and restricting to the time range where both
classes overlap leaves the cortical AUROC unchanged (0.996–1.000). It is cortex.

**Is the measure at ceiling?** Yes, within session — and this is why the figure is the FROZEN arm.
Binary running-vs-quiet decodes at AUROC 0.99–1.00 within session and the three-way problem at
macro-AUROC 0.98–1.00. **A ceiling cannot demonstrate preservation**: a reader sees "the task was too
easy to fail" and is right. The position claim rests on a frozen pre-stroke model failing on
post-stroke data, so the control has to be the same object, carrying the same cross-session
generalisation burden. The within-session number is the ceiling line, not the result.

**And the running survives the lesion**, which had to be true for any of this to be possible:

| epoch | sessions | running (% of session) | quiet % | bouts/session |
|---|---|---|---|---|
| pre | 61 | 3.1 (IQR 1.5–5.9) | 3.4 | 46 |
| acute | 16 | **5.1** (3.0–9.6) | 15.1 | 61 |
| subacute | 18 | **6.3** (3.3–11.2) | 9.2 | 63 |
| chronic | 12 | 2.0 (0.7–2.4) | 0.7 | 21 |

The animals run **more** acutely, not less — so the state arm is not rescued by having more data at
baseline than afterwards, and there is more data exactly where the position code collapses. Chronic
is the thin end and its panel states its n.

---

## Scoring

**BALANCED accuracy against chance 1/3, never raw.** The class balance moves with epoch — quiet is
3.4% of a pre-stroke session, 15.1% acutely, 0.7% chronically — and raw accuracy under a base rate
that moves that much is not comparable across the epochs the figure exists to compare.

**AUROC is computed over the classes actually PRESENT.** Chronic sessions at 0.7% quiet often hold
two classes; passing three to `roc_auc_score` raises rather than degrading gracefully.

**`MIN_PER_CLASS = 15`.** A session holding three quiet segments can still produce a balanced
accuracy, and it would be noise with a number on it.

---

## Known limits

1. **The two bars of `epoch_13n` are not the same estimator.** Position is the trial-weighted pooled
   accuracy read off the 5c confusion counts; state is the mean over sessions of a balanced
   accuracy. The contrast (51% vs 11%) is far larger than that difference can account for, but the
   two columns are not interchangeable numbers and the subtitle says so.
2. **No intervals on the retention bars.** Both are pooled point estimates.
3. **The licking/state asymmetry** described above.
4. **One alignment only.** These segments are not trials and have no cue to align to, so there is no
   pre-cue / post-cue / post-lick split to make; rendering the same figure under three arm labels
   would imply three analyses where there is one.
5. **`by_animal_day` projects every session onto the joint basis** (~10 min for the cohort) and is
   `lru_cache`d in-process only. A second process pays it again.
