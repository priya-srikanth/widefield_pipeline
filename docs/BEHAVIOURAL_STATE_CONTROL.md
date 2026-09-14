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

Against the frozen POSITION decoder on the same sessions, scored the SAME way (balanced
accuracy, the mean of the six row recalls): 0.886 → 0.523 → 0.749 → 0.833 (chance 1/6).

Normalised as the fraction of above-chance performance retained — the only way to compare a 6-way
problem at chance 0.167 with a 3-way at 0.333:

* **position: 0.863 → 0.428 acutely. It loses 50% of what it had.**
* **state: 0.917 → 0.812. It loses 11%.**
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
at that onset and allowed to run past the event's end. Quiet must **not** be: a window past its end
sits in the movement or licking the period was buffered away from, which is the one thing the class
must not contain. Per session this took licking from 72–297 segments to 413–969.

**SUPERSEDED 2026-09-12 — the anchor is now the trial's FIRST LICK AFTER THE CUE**, not the onset of
a free-running lick bout (Priya: *"for the state decoder - let's do the same post-lick window we use
for the other decoders throughout analysis"*). The two anchors ask different questions. Bout onset
asks *is the animal licking*, and answers from whatever the ILI rule grouped; the post-cue first lick
is the anchor **every position decoder in the deck already uses**, so the licking class and the
position trials become the same event observed twice rather than two events sharing a word. Cue and
reward are simultaneous in this task, so the first lick after the cue is also the first lick after
reward. A trial whose lick never arrives inside the 3.5 s response window contributes nothing — a
miss has no lick to anchor on.

**The window stays 1 s, and that is a decision rather than an inherited default.** The position
decoder uses 2 s; matching it here would have made window DURATION a cue the decoder could separate
the classes on, because a longer window is a smoother binned feature whatever the behaviour. All
three classes are therefore duration-matched at 1 s and only the ANCHOR is shared with the position
arm. `locomotor_state.LICK_POSTCUE_S` is pinned equal to `SEGMENT_S` by a test for exactly this
reason. `lick_mode="bout"` keeps the old behaviour available for comparison; `"postcue"` is the
default the figures use, and it refuses to run without cue samples rather than falling back — a
figure captioned "post-cue" built from free-running bouts would be undetectable downstream.

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

1. ~~The two bars of `epoch_13n` are not the same estimator.~~ **FIXED** rather than disclosed:
   both are now balanced accuracy (mean of the six row recalls for position,
   `balanced_accuracy_score` for state). It matters because the post-stroke position sets are
   skewed by construction — PS93's are 49% far_center — so a trial-weighted accuracy is pulled
   toward whichever positions the animal still attempts.
2. **No intervals on the retention bars.** Both are pooled point estimates.
3. **The licking/state asymmetry** described above.
4. **One alignment only.** These segments are not trials and have no cue to align to, so there is no
   pre-cue / post-cue / post-lick split to make; rendering the same figure under three arm labels
   would imply three analyses where there is one.
5. **`by_animal_day` projects every session onto the joint basis** (~10 min for the cohort) and is
   `lru_cache`d in-process only. A second process pays it again.


---

## A regression this analysis caught, in the position pipeline

Promoting `stopped` to a trial class meant routing the selection rule through one helper
(`_class_select`). Routing the **pre-stroke branch** of `_collect_5c` through it too silently added
the miss-while-working trials to a panel that had always been licking-only:

| | before | after the mistake | restored |
|---|---|---|---|
| frozen position, pre-stroke n | 21,017 | 22,076 | 21,017 |
| its accuracy | 0.886 | 0.859 | 0.886 |

That moves **every pre-stroke position number in the deck**. `_collect_7` already stated the rule a
pre panel follows — a pre-stroke animal is not missing, so `working` adds nothing at pre except a
different KIND of trial, and the reference then differs from itself — and the `_collect_5c` copy of
that rule was not written down anywhere, so it was easy to "tidy" away.

**It surfaced only because a brand-new figure's pre bar disagreed with a number printed on an older
one.** That is luck, not a guard. `stopped` is now the only class that may take unengaged rows at
pre, and `test_pre_panel_is_licking_only_for_lick_and_working` pins it.

Every figure rendered from `_collect_5c` on the `working` / `lick` arms between the two commits
carries the inflated pre panel — `acc`, `5c`, `5cr`, `5r`, `5rm`. The matrix and scalar families go
through `_collect_7`, which was not touched, and are unaffected.


---

# 2026-09-12 — the quiet class was REDEFINED; every number above is provisional

The class this document calls `quiet` is now `rest`, and its definition changed: it is bounded by the
TRIAL (`cue + response_window + 0.5 s` to the next `trial_start`) rather than by an 8 s post-reward
buffer. The old buffer made the class track the animal's performance — 4.4% of frames pre-stroke
against 17.1% acutely — so **every state-decoder number in this document was computed on a class
whose size moved with the lesion.** They must be re-measured. See `docs/REST_BASELINE_MIGRATION.md`.

**SECTION 2 IS THE PART TO RE-READ, and its conclusion survives.** The window is 1 s because the
quiet distribution said so, and on the rebuilt cohort it still does:

| | retired (this doc) | rest |
|---|---|---|
| median period | 1.10 s | **1.63 s** |
| a 1 s window fits | 58% | **84.6%** |
| a 2 s window fits | 17% | **19.4%** |
| periods | 14,017 | **26,010** |

So 2 s remains rejected for the same reason, and 1 s is now much better supported. **The window does
not change, which is what makes the re-measured numbers comparable to the ones above.**

Two further changes affect this analysis and are recorded here so they are not discovered later:

* **Segments now tile from the END of a period**, not the start (Priya: "avoid leftover incomplete
  licks"). A rest period opens the moment the lick buffer expires, so residual licking sits at its
  start. Licking stays ONSET-anchored — it is an event, and tiling backwards from the end of a
  0.37 s median bout would sample the silence after it.
* **`MAX_SEGMENTS_PER_PERIOD` is now nearly inactive.** It existed because "the top 1% of QUIET
  periods supply 36.6% of all segments"; the trial anchor caps a period at the inter-trial interval,
  so measured capping is 0.000–0.088 of periods. More periods contributing one or two segments each
  is more independent units for a bootstrap clustered by period, not fewer.


---

## 2026-09-13 — the state decoder can read TIME, and per-session balancing cannot fix it

**THE CONFOUND.** `locomotor_state` splits a session into licking / running / rest. Those classes are
not distributed alike over session time — licking is cue-locked, rest fills the ITIs, running drifts
— while cortex itself drifts: `rest_position_vs_drift` measured the same position's rest early-vs-late
at **RMS 0.00282**, with no behavioural difference at all. A three-way decoder can therefore separate
the classes partly on WHEN the window sat. **The time-local baseline that fixes this for the maps does
not reach here**: rest is a CLASS in this analysis, not a subtrahend, so there is nothing to subtract
it from.

**THE COMPOSITION IS MEASURED** (`scripts/rest_migration/state_time_bins.py`, 91 sessions, 1 skipped
— PS92_0812, the crash+concat session — 5 equal bins of each session's imaging span, segments counted
exactly as `locomotor_features` builds them):

| epoch | licking bin1 → bin5 | rest bin1 → bin5 | running bin1 → bin5 |
|---|---|---|---|
| pre | 4,105 → 3,413 | 2,198 → 3,893 | 2,305 → 4,097 (U-shaped) |
| acute | **1,131 → 441** | 732 → 1,858 | 294 → 1,778 |
| subacute | **1,802 → 649** | 318 → 1,494 | 821 → 2,593 |
| chronic | 1,212 → 1,163 | 187 → 903 | 289 → 441 |

**THE SHIFT IS STEEPER POST-STROKE THAN PRE.** Pre-stroke licking is nearly flat across the session
and the largest class share per bin is 0.36–0.51; acutely and subacutely licking falls by 60–64% from
first bin to last while rest and running rise, and chronically the largest share reaches **0.79**.
So the amount of information carried by "when in the session" is itself different pre and post —
which is exactly the axis `BEHAVIOURAL_STATE_CONTROL.md` reads for preservation. **This is a stated
limit on that control, not a refutation of it**: the decoder is FROZEN on pre-stroke data, so it
cannot fit a post-stroke-specific time structure; what it can do is carry a pre-stroke time signal
into a post-stroke session where the classes sit at different times.

**BALANCING IS POSSIBLE POOLED AND NOT PER SESSION.** Priya, 2026-09-13: *"I don't know that it's
worth doing the time regression or balancing across time bins, unless it's pretty simple to implement
and doable (ie there is enough of each class across all time bins)."* Retained fraction under
`3 x min-over-classes` per bin:

| epoch | pooled | per session |
|---|---|---|
| pre | 0.63 | **0.35** |
| acute | 0.48 | **0.33** |
| subacute | 0.59 | **0.31** |
| chronic | 0.36 | **0.25** |

and **30 of 91 sessions have at least one EMPTY class-bin** (11/43 pre, 5/16 acute, 9/18 subacute,
5/14 chronic), where an empty cell zeroes that bin for all three classes. The worst single session
would retain **0.02** of its segments. PS94_0606 has **20 running segments in the whole session**.

**DECISION: the simple (unbalanced) state decoder stands, and the composition table above is
reported as its limit.** Per-session balancing would cost 65–75% of the data to remove a confound
whose size has not been shown to matter, and would silently drop a third of the sessions' bins.

**THE CHEAP TEST THAT WOULD SIZE IT, and has not been run:** score the frozen decoder separately
within each session-time bin. If accuracy is flat across bins, time is not carrying it. That is one
pass over the existing features with no refitting, and it is the thing to do before either balancing
or dismissing this.


---

## 2026-09-13 — the state decoder does NOT read session time: the confound is real and inert

The composition confound recorded above is real -- the three classes are not spread alike over
session time and the imbalance is steeper post-stroke than pre. What it did not establish is whether
the decoder USES it. Priya, 2026-09-13: *"sure do the frozen decoder scoring within time bin (but we
have to re run the frozen decoder first right, on the new 'REST' definition?)"* -- yes, and the run
does both at once (`scripts/rest_migration/state_decoder_by_time.py`, 91 sessions, PS92 8/12
excluded, frozen on all pre-stroke segments with the pre column leave-one-session-out).

**Score the frozen decoder separately within each fifth of the session, same model, same labels,
only the subset moves:**

| epoch | overall | bin 1 | bin 2 | bin 3 | bin 4 | bin 5 | **spread** | n |
|---|---|---|---|---|---|---|---|---|
| pre | 0.982 | 0.977 | 0.983 | 0.984 | 0.983 | 0.976 | **0.008** | 43 |
| acute | 0.918 | 0.918 | 0.937 | 0.926 | 0.913 | 0.893 | **0.044** | 16 |
| subacute | 0.886 | 0.848 | 0.907 | 0.899 | 0.919 | 0.899 | **0.071** | 18 |
| chronic | 0.952 | 0.948 | 0.925 | 0.931 | 0.951 | 0.967 | **0.042** | 14 |

**THE PROFILE IS FLAT, AND IT IS FLAT WHERE IT MATTERS MOST.** Pre-stroke spread is 0.008 against a
pre-to-subacute drop of 0.096 -- an order of magnitude smaller than the effect it could have
explained. In every epoch the per-bin values BRACKET the overall value, so no single fifth is
carrying the score.

**THE SPREAD DOES NOT TRACK THE COMPOSITION SHIFT**, which is the decisive part. Acute has the
steepest class-composition drift of any epoch (licking falls 1,131 -> 441 across the session) and
the second SMALLEST spread, 0.044, with a non-monotone profile that peaks in bin 2. If the decoder
were reading time, the epoch whose classes separate most strongly in time would be the one whose
accuracy varied most across bins. It is not.

**CONCLUSION: the confound exists in the data and is inert in the estimator.** The specificity
control stands as written, and per-session balancing -- which would have cost 65-75% of the segments
and emptied a class-bin in 30 of 91 sessions -- is correctly not done. This is now the reason,
rather than the absence of a measurement.

**A SECOND RESULT, unplanned: the existing `epoch_13*` figures are already on the REST definition.**
This run refits from the events npz on disk, and its overall numbers reproduce the 2026-09-12
sidecar to four decimals (pre 0.9818, acute 0.9185, subacute 0.8864, chronic 0.9524). The
straddle-the-events-rebuild worry recorded in STATUS_2026-09-13 does NOT apply to the state
decoder's cue arm -- verified by re-computation rather than inferred from file timestamps, which is
the only way that question can actually be settled.

**WHAT THIS DOES NOT SHOW.** Within a fifth of a session there is still a time axis, and a decoder
reading slow drift would read it there too. What is ruled out is the version that threatened the
control: that post-stroke accuracy is held up by the classes having migrated to more separable parts
of the session.


---

## 2026-09-13 — YES, the state decoder falls too. "Preserved" is a RATIO, never "unchanged"

Priya, 2026-09-13: *"does the new decoder show a change in accuracy post-stroke?"* It does, at every
epoch, and the interval excludes zero even after Bonferroni (`epoch_13delta`, change from pre-stroke
in balanced accuracy):

| epoch | change | 95% CI | Bonferroni-corrected CI |
|---|---|---|---|
| acute | **−0.063** | [−0.121, −0.017] | [−0.130, −0.014] |
| subacute | **−0.095** | [−0.207, −0.019] | [−0.221, −0.016] |
| chronic | **−0.029** | [−0.045, −0.012] | [−0.049, −0.010] |

**THE CONTROL HAS NEVER CLAIMED THE STATE READOUT IS UNCHANGED, and a reader who takes
"preserved" to mean "flat" will misread every panel in section I.** The claim is a RATIO: as a
fraction of pre-stroke above-chance performance retained (`epoch_13n`),

| | pre | acute | retained |
|---|---|---|---|
| POSITION, 6-way | 0.863 | **0.428** | **0.496 — loses 50%** |
| STATE, 3-way | 0.973 | **0.878** | **0.902 — loses 10%** |
| RUNNING alone (recall) | 0.990 | 0.956 | loses 3% |

Both fall. Position falls **five times further**, on the same window, the same basis, the same
estimator and the same frozen-model discipline, with only the LABEL changed. That is the whole
argument and it does not need the state readout to be flat -- it needs the two to be measured the
same way, which they are.

**SUBACUTE IS THE STATE DECODER'S WORST EPOCH, not acute** (−0.095 against −0.063), while position's
worst is acute by a wide margin. The two do not even fall in the same temporal pattern, which is
further against a common cause.

### The headline numbers moved for STATE and did not move for POSITION

The previously documented triple was *position 0.86 → 0.43 (loses 50%), behavioural state 0.92 →
0.81 (loses 11%), running alone 0.98 → 0.95 (loses 3%)*. On the REST definition:

* **POSITION is unchanged to three decimals** (0.863 → 0.428). Expected, and a useful check: the
  position decoder never reads the rest mask (`_build_signal` loads LocaNMF `C` or raw `U`/`SVT`),
  so the migration could not have touched it. It didn't.
* **STATE moved** (0.92 → **0.973** pre, 0.81 → **0.878** acute). Expected in the other direction:
  rest is a CLASS in that decoder, so redefining it redefines a third of the problem. The retained
  fraction barely moved (0.89 → 0.90), which is why the conclusion is unaffected.

**Both directions are confirmations, not coincidences.** A migration that changed the position
number would have meant the rest mask was leaking into an analysis that must not see it; one that
left the state number alone would have meant the class redefinition had no effect on a decoder built
from it. Neither happened.


---

## 2026-09-13 — the state decoder's post-stroke errors point at LICKING specifically, and DLC can test why

Priya, reading `epoch_13c`: *"quiet is sometimes looking like 'licking' in acute post stroke and
running like licking > quiet in subacute stroke ... the DLC analysis will help see if 'quiet' in
those times include incomplete licks"*.

**THE MEASUREMENT** (row = true class, cell = fraction predicted; cue arm, frozen pre-stroke model):

| epoch | quiet→licking | running→licking | licking→quiet |
|---|---|---|---|
| pre | 0.015 | 0.006 | 0.013 |
| **acute** | **0.143  (9.5x pre)** | 0.019 | 0.032 |
| **subacute** | 0.062 | **0.101  (17x pre)** | **0.137  (10.5x pre)** |
| chronic | 0.066 | 0.015 | 0.016 |

Both readings hold. Acutely, quiet's errors go to LICKING; subacutely, running's errors go to
licking (0.101) MORE than to quiet (0.058), and licking↔quiet becomes bidirectional.

**THE ERRORS ARE SPECIFIC, NOT DIFFUSE, AND THAT IS THE ARGUMENT.** If the acute rise were simply
"post-stroke cortex is less distinguishable", quiet's errors would spread across both other classes.
They do not: quiet→licking is 0.143 while quiet→running is 0.027, a 5:1 split toward one class.
A generic degradation does not pick a direction. Something is making quiet windows look
specifically LICKING-like.

**PRIYA'S HYPOTHESIS, and it would make this a LABELLING error rather than a decoder error.** REST
is defined as *between trials, not running, and NOT LICKING* — where "not licking" means **no event
on the DAQ lick sensor**, which requires tongue-to-spout contact. An INCOMPLETE tongue protrusion
that never reaches the spout produces no sensor event and is therefore filed as REST. Cortex would
be doing licking-like things inside a window labelled quiet, and the decoder calling it *licking*
would be RIGHT while the label was wrong.

**AND THE PREDICTED TIMING IS THE DEFICIT ITSELF.** Failed and incomplete protrusions are what a
ventrolateral-striatal lesion with an orofacial deficit should produce, and should produce MOST
acutely — which is exactly where quiet→licking peaks. The hypothesis predicts its own time course.

**THE DLC TEST, stated so it can fail.** Score tongue protrusions from the behaviour video
independently of the lick sensor, then ask, within REST-labelled windows:

* Do windows the decoder calls LICKING contain more DLC protrusions than windows it calls quiet?
* Is that excess larger ACUTELY than pre-stroke, and does it track the quiet→licking rate by
  epoch and by animal?
* Confirmed → the REST class is contaminated by sensor-invisible licking and the confusion is a
  labelling artefact with a biological cause. Refuted (protrusion rates equal) → the acute
  quiet↔licking confusion is a genuine change in resting cortical state, which is a different and
  also interesting result.

**THE KNOCK-ON NOBODY HAS CHECKED: the same sensor defines the REST BASELINE for the maps.**
`quiet_periods` builds the rest mask from the same DAQ lick events, so if incomplete licks fall
inside REST then the MAP subtrahend contains licking-related activity too — most acutely, which is
the epoch every headline map contrast is read at. This does not obviously threaten the far-contra
result (it would add a position-independent term, and the cross-position null is built to be blind
to exactly that), but it has not been tested and should be once DLC protrusion times exist.

**RELATED CAVEAT ALREADY ON RECORD**, now with a quantitative prediction attached: section G's G6
caption has said since 2026-08-18 that *"'no lick detected' is not 'no tongue protrusion' — DLC
replaces this inference with a measurement"*. This is the same caveat surfacing in a second analysis.


---

## 2026-09-13 — the STATE DECODER redo is not a re-render, and it moves published numbers

Priya: *"we'll also need to redo the state decoder"*. Yes, and it is worth separating from the map
redo because the two are different kinds of work.

**FOR THE MAPS, REST IS A SUBTRAHEND.** Changing its window changes what is subtracted.

**FOR THE STATE DECODER, REST IS A CLASS.** Changing its window changes which SEGMENTS EXIST, how
many there are, and what the other two classes are contrasted against. Nothing downstream is a
re-plot.

**WHAT MOVES, and all of it is currently published:**

* the class balance and segment counts (`_state_class_share`, now measured rather than asserted)
* **the retention ratio in the headline triple** — `behavioural state 0.97 → 0.88 (loses 10%)`,
  against position's 0.86 → 0.43. In `CLAUDE.md`, `BEHAVIOURAL_STATE_CONTROL.md` and the synthesis.
* the per-class recalls (rest 0.97 → 0.84 acutely) and the `{{SELF:}}` tokens quoting them
* **the confusion result the DLC hypothesis rests on** — rest→licking 0.143 acute, 9.5× pre, the
  5:1 split toward licking over running. If that survives the docked window the incomplete-lick
  hypothesis stands; if it does not, the hypothesis goes with it.
* `epoch_13*` (6 figures), `epoch_12b*`, and deck section I's state panels

**THE DIRECTION IS PREDICTABLE AND WORTH STATING IN ADVANCE**, so the redo is a test rather than a
reveal: the docked window is SHORTER and excludes the retraction, so REST segments will be fewer and
cleaner. Fewer segments widen the intervals; a cleaner class should if anything RAISE rest's recall.
**If rest's recall instead falls, the retraction was contributing to it** — which would matter,
because the rest class is the one the specificity control reports as moving.

**ONE THING THAT WILL NOT MOVE:** the 1 s segment window. It was re-derived on the trial-anchored
REST definition (2 s fits 19.4% of bouts, 1 s fits 84.6%) and the docked window is shorter still, so
if anything it is more firmly 1 s. Re-measure rather than assume — the bout-duration distribution
changes with the window.
