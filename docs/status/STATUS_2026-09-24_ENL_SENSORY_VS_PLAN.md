# ENL sensory-vs-plan — status and handoff (2026-09-24)

**START HERE** if you are picking up the "how much of the pre-cue signal is sensory" analysis.
Everything below is either measured or implemented; where something is neither, it says so.

Priya, 2026-09-24: *"id like to add an analysis that looks at ENL responses on working vs stopped
trials — in theory, sensory information should be present in both, so we can tease out how much ENL
signal is from that alone vs motor plan."*

---

## The idea, and the one thing that makes it work

The spout is physically in position during the ENL of **every** trial. A trial the animal is still
working is one where a plan is also being formed; a trial from the terminal quit period is one where
(pre-stroke) it is not. So the contrast uses the animal's own behaviour as the switch: **remove the
intention, leave the stimulus.**

That matters because CLAUDE.md records this as a standing, unresolved limit:

> Deliberately NOT called a "maintained motor plan": the spout arrives ~3 s before the cue, so a
> sustained sensory response and a held intention are temporally coextensive and **this design
> cannot separate them.**

Timing cannot separate them. Behaviour can. **Position decodable from STOPPED ENL is the first
direct evidence in this project that the pre-cue code is not only a held plan.**

---

## The classes

| class | definition | gate |
|---|---|---|
| `success` | detected lick within `decode.max_rt_s` | `is_engaged` (per trial) |
| `miss_working` | no lick in the response window, before the terminal collapse | `engagement_gate` (per session) |
| `stopped` | no lick in the response window, inside the terminal collapse | `engagement_gate` |

**The default contrast is `miss_working` vs `stopped`, not `success` vs `stopped`** — both are
non-hits, so outcome, reward and movement are matched and only engagement differs.

**This is not the forbidden gate.** `POSTSTROKE_ENGAGEMENT_FILTERING = False` rejects the ROLLING
per-trial reference-rate gate. `engagement_gate` requires a NON-RECOVERING FINAL collapse, which
`grant_confusion` states is "a much weaker claim than adjudicating individual trials, and it is the
same construct the miss-vs-stopped split rests on throughout". Nothing here filters another analysis.

---

## MEASURED: the trial-count landscape

`scripts/enl_state_counts.py`, 104 curated sessions, cached to CSV.
Figure: `scripts/enl_sparsity_figure.py` → `enl_stopped_sparsity.png`.

**STOPPED trials per position, pre-stroke:**

| animal | close_L | close_R | close_ctr | far_L | far_R | far_ctr |
|---|---|---|---|---|---|---|
| PS92 | 5 | 0 | 0 | 3 | 2 | 0 |
| PS93 | 10 | 12 | 7 | 16 | 10 | 12 |
| PS94 | 59 | 40 | 59 | 60 | 55 | 54 |
| PS95 | 74 | 74 | 98 | 90 | 77 | 96 |

Animals clearing ≥10 stopped per position: **pre 2** (PS94, PS95) · **acute 3** (PS92/93/94, PS95 has
none) · **subacute 3** (PS93/94/95) · **chronic 2** (PS94, PS95).

### The bind this creates

**The epoch where the inference is VALID has the least data; the epochs with data are the ones where
the inference is NOT underwritten.** Pre-stroke `stopped` is genuinely the sated state
(`precue_engagement_states`: "pre-stroke this is essentially the sated/disengaged state"). Post-stroke
it may be a late motor collapse with an intact plan — `grant_confusion`: *"nothing in the spout data
proves the terminal run is satiety rather than a late motor collapse."*

Not a coincidence: a well-trained animal does not quit, a lesioned one does. The thing that creates
the data is the thing that makes it ambiguous.

**DECISION (Priya, 2026-09-24): pool positions pre-stroke.** Pooled, PS93/94/95 all clear — three
animals, rule 8's bar. Pooling means not reporting per-position effect sizes; the position LABELS
still drive the decoder, so the sensory question is untouched.

---

## THREE DESIGN ERRORS CAUGHT BEFORE THEY TOUCHED DATA

Each was found by a smoke test or by reading the source, not by inspection of results.

### 1. The two classes cannot be time-matched — they are disjoint by construction

`engagement_gate` marks ONE terminal run, so `miss_working` exists only BEFORE the onset and
`stopped` only AFTER. Working always precedes stopped, in every session.

The first draft copied `precue_engagement_states`' control (draw both classes from the same late
window) and **silently left ZERO working trials**. `enl_states.adjacent_window` now takes working
trials from a same-length window ENDING at the onset, and `time_gap` reports the residual.

**Consequence:** a working−stopped amplitude difference is ALWAYS partly a late-session difference.
This is why the decode, not the subtraction, carries the claim.

### 2. `late_rewarded` was being scored as `working`

A lick after `max_rt` but inside the response window is a **hit** — movement happened, reward
arrived. `enl_states.states_for` scored it `working`, putting a rewarded executed trial in the class
whose whole purpose is to be outcome-matched to `stopped`. `nolick_decoder` already separates it;
`enl_decode.arms_for_session` uses that split and excludes it.

### 3. Block groups restart per session

Pooling them unchanged lets `GroupKFold` put block 3 of session A and block 3 of session B in one
fold — not trial leakage but **session** leakage, and the session is the level this must generalise
across. `enl_decode.pool_arms` offsets them (verified: 16 distinct blocks from two 8-block sessions).

### And one wrong statistic

`enl_decode.ratio` first measured both arms against **uniform chance** — the comparison
`nolick_analysis` exists to retire (it ships the old flag as `above_uniform_chance_DEPRECATED`, "the
claim this module exists to retire"). It also returned **100** for an arm at chance, because
`0.167 − 1/6` is a tiny positive rather than zero. Now: both arms measured above **their own
permutation null**, guarded on `above_null_balanced`.

Measured while fixing it: under LABEL SKEW alone the BALANCED null stays at ~1/6 (macro-recall is
robust to imbalance); the 0.211-vs-0.167 figure in `nolick_analysis` is the RAW null. The balanced
null moves when the PREDICTOR is biased.

---

## READOUT 4 — the ratio, rebuilt so it CAN be computed

Readout 2 (`ratio`) fits a **separate** decoder on each arm and divides. On PS95 pre-stroke neither
fit cleared its own null, so it returned `undefined`. That was the correct answer for that statistic
and the wrong answer to the question: the within-arm nulls say a six-way fit needs more than 495
trials, not that stopped trials lack position information — and the transfer, reading those same 495
trials, shows they carry it.

**Readout 4 (`shared_decoder`) trains ONCE on `success` and scores every arm with the same fitted
model.** Training set, hyperparameters and feature space are identical across arms, so the only thing
differing between numerator and denominator is *which trials are being read*.

**Blocks are held out across ALL arms at once, not just the training one.** A stopped trial and a
success trial from the same block share a spout position and sit seconds apart, so scoring a stopped
trial with a fold that trained on success trials from its own block would report memorisation. Every
trial in every arm is predicted by exactly the fold in which its block was held out — which also
makes the `success` number a genuine cross-validated score rather than a training fit, and therefore
comparable to the others. `coverage` reports the fraction of an arm that fell in a held-out block, so
an arm scored on a subset is visible rather than silent.

Two ratios, and **the outcome-matched one is primary**:

| ratio | denominator | what it is |
|---|---|---|
| `clean_ratio_vs_working` | `miss_working` | both arms are no-lick trials — outcome, reward and movement matched, **only engagement differs**. This is the sensory-vs-plan fraction. |
| `clean_ratio_vs_success` | held-out `success` | the ceiling: what this basis and window give when position information is present AND a plan formed AND reward followed. A scale reference, not a contrast. |

Both numerator and denominator are still measured **above their own permutation null**, and the
`above_null_balanced` guard still applies to the denominator.

**Verified on synthetic data:** signal in `stopped` → clean ratio **0.971** (vs working), coverage
1.00; noise in `stopped` → **0.022**, flagged *"stopped arm is NOT above its own null"*. The leakage
test in `tests/test_enl_decode.py` makes the label a function of the BLOCK with block-identity
features, so any fold that trained on a trial's own block would score it perfectly — all three arms
are required to sit below 0.45.

### What readout 4 still does not license

A ratio of above-null balanced accuracies is **not** a ratio of information, so "X% of the ENL signal
is sensory" stays unsupportable. What it supports is: *"the position code read by one success-trained
decoder survives on stopped trials at X× its strength on outcome-matched working trials, both above
their own nulls."* And attention rides with engagement, so a reduced-but-present code may be
attenuated sensory rather than sensory-minus-plan.

---

## RESULT — all four animals, pre-stroke, STRICT LICK-FREE GATE (2026-09-24)

`python -m wfield_local.enl_decode --epoch pre --transfer --out <json>`
`python -m scripts.enl_decode_figure --json <json>` -> **`enl_decode_pre.png`**
11 sessions each, positions pooled, per-animal frozen joint basis, readout 4.

| animal | `success` | `miss_working` | `stopped` | miss/succ | stop/miss |
|---|---|---|---|---|---|
| PS92 | 0.565 | 0.227 *at null* | 0.111 *at null* | — | — |
| PS93 | 0.513 | **0.302** | 0.089 *at null* | 0.392 [0.18, 0.77] | — |
| PS94 | 0.712 | **0.314** | **0.251** | 0.269 [0.07, 0.44] | 0.584 [0.20, 2.02] |
| PS95 | 0.485 | **0.328** | **0.304** | 0.505 [0.33, 0.73] | 0.856 [0.50, 1.41] |

Nulls ~0.166 except PS92, restricted by `common_positions` to 3/6 (null 0.236). Brackets are paired
block-bootstrap 95% CIs. Bold = above its own permutation null.

### THE LADDER IS NOT EVENLY SPACED — that is the result

Arm-vs-arm tests. Nothing before the bootstrap did this: every earlier p was an arm against its OWN
null, which says whether a code is present and nothing about whether two arms differ.

* **`success` > `miss_working` in 4/4** (all p < 0.0001, every CI excludes zero). The large,
  reliable drop is losing EXECUTION AND REWARD.
* **`miss_working` > `stopped` in only 1/4** — and that one is PS93, whose stopped arm is at null
  with n=40. In both animals with a usable stopped arm the two are **not distinguishable**
  (PS94 p=0.275, PS95 p=0.521), and both stop/miss CIs include **1.0**.

Removing the plan-and-execution component costs a lot. Removing ENGAGEMENT on top of that costs
nothing this design can resolve. The outcome-matched rung is the one where the arms are hardest to
tell apart — which is a stronger form of the sensory answer than a point estimate would be.

**"Not distinguishable" is NOT "equal."** PS94's stop/miss CI spans 0.203 to 2.017. The "could not
test" vs "tested and found nothing" distinction applies here as everywhere else.

**Withdrawn:** a stop/miss spread of 0.42 vs 0.80 was first called "a factor of two apart". The CIs
overlap heavily; that spread is not resolvable and the claim is retracted.

### `success` IS THE TRAINING ARM — every ratio against it is a LOWER BOUND

Block hold-out makes `success` a genuine cross-validated score, but the decoder is FIT TO
SUCCESS-TRIAL STATISTICS and the other arms are scored out of distribution. Any shift between arms —
activity level, noise, hemodynamics — costs accuracy on its own, so **miss/succ and stop/succ are
biased downward**.

That is the argument for **stop/miss as the primary readout**: numerator and denominator are BOTH
out of distribution, so the domain-shift cost largely cancels. It also explains why stop/succ (0.157,
0.432) sits far below stop/miss (0.584, 0.856) in the same animal, and **that gap is not biology**.

---

## THE LICK-FREE GATE, AND THE NUMBER NOBODY HAD MEASURED

`enl_decode` reaches features through `nolick_decoder.session_features`, which **built a fixed
`[cue-2s, cue]` window and applied no lick gate at all** — a second definition of "the pre-cue
window" against `locanmf_position_decoder._trial_features`, which slides to a clean gap and DROPS a
trial with none (rule 9). Priya, 2026-09-24: *"we should keep the consistent strict no lick in ENL
gate for ENL analyses."* Aligned, along with a second divergence found in the same function:
`categorize` used the superseded position-change-only block rule rather than the audited
`block_ids`, and those ids are the `GroupKFold` groups readout 4's hold-out rests on.

**`scripts/enl_lick_rates.py`, 104 curated sessions:**

| | range | what happens to it |
|---|---|---|
| lick INSIDE the window | 0.1 – 8.3% (worst session 26.4%) | window SLIDES to a clean gap |
| no clean window anywhere | 0.2 – 2.1% pre, ≤0.34% post | trial is DROPPED — the only rate that costs data |
| **lick in the second BEFORE** | **13.7 – 90.9%** | **nothing — unguarded** |

The in-window figure replicates the measurement that justified the 2026-08-17 gate (PS93_0809 is
76.3% clean, the "PS93 8/9 falls to 76%" on record). The gate was correctly sized.

**The lead window is a different universe.** The window is `[cue-2s, cue]` and the spout arrives
~3 s before the cue, so the second before the window IS the first second after spout arrival. This
is spout-arrival licking, and in three of four animals it is on the MAJORITY of trials.

**It is a confound, not noise: the lick is DIRECTED AT THE SPOUT, so it carries position.** The
contaminating signal is correlated with the label being decoded. An HRF peaking 1–2 s later lands
inside the window, so a window that is lick-free by construction can still contain a
position-informative MOTOR signal. If the ENL code were largely that, "the pre-cue code is not only
a held plan" would be true for the wrong reason.

### Effect of the gate on the result: it HELD

| | before | after |
|---|---|---|
| PS92 `miss_working` | 0.280 ABOVE | **0.227 at null** (n=69 scored; 85% lead rate) |
| PS93 `miss_working` | 0.337 ABOVE | 0.302 ABOVE |
| PS94 `stopped` | 0.239 (p=0.0015) | **0.251 (p=0.0005)** |
| PS95 `stopped` | 0.295 | **0.304** |

`success` > `miss_working` stayed 4/4; `miss_working` > `stopped` stayed 1/4. Numbers moved slightly
UP where they moved, which is the direction flagged in advance: the gate removes contaminated
windows and the block fix gives a finer hold-out. Only PS92's `miss_working` was lost, and at 69
scored trials it was never solid. The miss/succ rung is therefore **n=3**, still at rule 8's bar.

---

## THE CONTAMINATION CONTROL — `wfield_local/enl_lick_control.py`

Priya, 2026-09-24: *"the pre-ENL-lick vs no-pre-ENL-lick can be on the hit and working trials. The
success vs miss while working ENL decoding is a SEPARATE question."*

So the split is **WITHIN** each arm, and this module deliberately does not re-report the cross-arm
ladder — mixing them would let a cross-arm difference read as a contamination effect.

    train    `success` trials with NO lead lick, blocks held out across every group
    score    success_clean (held out) · success_lead · miss_working_clean · miss_working_lead
    compare  clean vs lead, WITHIN each arm
    reverse  `transfer_lead_to_clean` — does a LICK-TRAINED decoder read LICK-FREE trials?

Training on clean trials forces the decoder onto information available without a preceding lick.

| outcome | reading |
|---|---|
| clean ≈ lead | the lick is not what the decoder reads. Control passes. |
| lead >> clean | the lick ADDS position information — confound real and sized. |
| **clean at null** | the code is present only when a lick preceded it. **Worst case.** |

`stopped` is NOT split: at 6/40/326/495 trials, halving it leaves nothing to test either half with.

**Power, from the measured lead rates.** PS95 (13.7% lead) is the animal that can answer this —
~86% clean. PS92 (85%) has ~19 clean `miss_working` trials and will report "could not test".
`success` (3.8k–6k) splits comfortably everywhere; `miss_working` is reportable in PS95, marginal
elsewhere.

`test_clean_at_null_is_stated_outright` builds the bad answer and requires the verdict to say so in
words. A control that cannot report its own failure is not a control.

## What is built

| file | state |
|---|---|
| `scripts/enl_state_counts.py` | **done**, cached, fanned over cores. Counts the BEHAVIOUR table — see the caveat below |
| `scripts/enl_sparsity_figure.py` | **done**. Drawn from that cache, so it OVERSTATES every cell |
| `scripts/enl_lick_rates.py` | **done and run**, 104 sessions — the in-window / lead / no-clean rates |
| `scripts/enl_decode_figure.py` | **done and run** → `enl_decode_pre.png` (3 panels, bootstrap CIs) |
| `wfield_local/enl_states.py` | classes + `adjacent_window` + `time_gap` + witness stamping. **Driver NOT wired** (raises a clear message) |
| `wfield_local/enl_decode.py` | **done and run** — four readouts, strict gate, frozen joint basis, paired bootstrap |
| `wfield_local/enl_lick_control.py` | **built and unit-tested, NOT YET RUN ON DATA** |
| `wfield_local/nolick_decoder.py` | strict pre-cue gate + audited `block_ids` + `lead_lick`/`shifted` flags |
| `wfield_local/position_reference_maps.py` | extended with `miss_working` / `stopped` variants, reusing `_quit_mask` |

**Verified on synthetic data.** `enl_decode`: signal → stopped 0.637 vs null 0.166; **noise → 0.152
vs null 0.171, p=0.72, flagged**. Readout 4: signal → clean ratio 0.971; **noise → 0.022, flagged**.
`enl_lick_control`: a clean/lead difference is detected when present and the clean-at-null case is
stated in words. The negative controls are the ones that matter.

### THE FEASIBILITY TABLE OVERSTATES THE DATA

| animal | behaviour table | actual decode |
|---|---|---|
| PS92 | 10 | **6** |
| PS93 | 67 | **40** |
| PS95 | ~509 | 495 |

`enl_state_counts` gates on the behaviour table; `enl_decode` gates on `sess_eng` in the IMAGING
universe, and coverage exclusions remove more. **Which dominates has NOT been verified.** If it is a
gate-boundary difference rather than coverage, that is a third instance of the rule 9 problem fixed
above and should be closed the same way. Until then `enl_stopped_sparsity.png` overstates every cell.

Consequence: pooling positions was justified by PS93/94/95 all clearing the bar, and **PS93 does
not**. `stopped` is **n=2**; the miss/succ rung is **n=3**.

### Not built
- the ENL maps themselves (`raw` and `rest` references, per epoch, n annotated)
- the `enl_states` amplitude-contrast driver
- the `precue_engagement_states` witness — **must run before interpreting any post-stroke cell**
- `enl_lick_control` across all four epochs (built; Priya asked for pre + acute + subacute + chronic)

---

## Design decisions to preserve

1. **`align = "precue"` always.** The ENL is the pre-cue period; a cue-aligned window contains the
   response this contrast exists to exclude. `precue_engagement_states.run_animal` is precue for the
   same reason.
2. **References are `raw` and `rest`, never `precue` or `mean`.** `defaults.yaml` states the pre-cue
   baseline "MUST NEVER BE USED WITH `align: precue`" — it is self-referential there. `mean` couples
   the six positions.
3. **The decode carries the claim; the subtraction is descriptive.** A global state change moves
   amplitude and CANNOT produce above-chance position decoding.
4. **Nothing is dropped for sparsity — it is marked.** "Could not test" and "tested and found
   nothing" are different facts.
5. **One definition per quantity.** `arms_for_session` re-splits `nolick_decoder`'s existing
   `undetected` category by its existing `sess_eng`; `enl_states.states_for` is behaviour-table-only
   (for counting) and must never build features.

## What a positive result will NOT license

- **Not a percentage.** Accuracy is not linear in information. "The position code survives at X% of
  its working-trial accuracy, both above null" is supportable; "40% of ENL is sensory" is not.
- **Attention rides with engagement.** A stopped animal may attend the spout less, so a
  reduced-but-present code could be attenuated sensory rather than sensory-minus-plan.
- **Pre-stroke only, and thinly.** Expect PS94/PS95 solid, **PS93 marginal** (~67 stopped pooled).
  If PS93 comes back underpowered that is n=2, and rule 8 says it will not survive the paired test —
  the honest output is then the per-animal table plus "not established at n=2".

## Post-stroke

Computed and reported, **never** used to support "no plan was formed". `enl_states.witness_verdict`
reads whether `precue_engagement_states`' discriminator (trained on the PRE-stroke lick/no-lick
contrast) separates post-stroke working from stopped, and `stamp_caveat` puts the licensed
interpretation on every row. Currently `"not run"` → reports as DESCRIPTIVE ONLY rather than
defaulting to the favourable reading. **Run that witness before interpreting any post-stroke cell.**
