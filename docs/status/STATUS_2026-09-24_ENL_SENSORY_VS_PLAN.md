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

## What is built

| file | state |
|---|---|
| `scripts/enl_state_counts.py` | **done**, cached, fanned over cores |
| `scripts/enl_sparsity_figure.py` | **done**, renders from the cache |
| `wfield_local/enl_states.py` | classes + `adjacent_window` + `time_gap` + witness stamping. **Driver NOT wired** (raises a clear message) |
| `wfield_local/enl_decode.py` | all three readouts + `pool_arms` + `arms_for_session`. **Session loop NOT wired** |
| `wfield_local/position_reference_maps.py` | extended with `miss_working` / `stopped` variants, reusing `_quit_mask` |

**Verified on synthetic data** (`enl_decode`): signal → stopped 0.637 vs null 0.166 (above), ratio
0.858, transfer 0.703. **Noise → stopped 0.152 vs null 0.171, p=0.72, not above; ratio flagged.**
The negative control is the one that matters.

### Not built
- the session-loading loop for `enl_decode.analyse` and `enl_states`
- the ENL maps themselves (`raw` and `rest` references, per epoch, n annotated)

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
