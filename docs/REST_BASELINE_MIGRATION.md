# The REST baseline — retiring "quiet"

**Status: the masks are built.** All 92 sessions have a `quiet_<tag>_rest/` mask, anchored on
`trial_start`, and `segmentation.rest.variant` is `rest`. The figures have NOT yet been re-rendered;
the retired QUIETref set has been moved to `grant_figures/epoch/retired_QUIETref/`.

---

## Why the old definition had to go

`configs/defaults.yaml segmentation.quiet` excluded **8 s after every reward**. Priya, 2026-09-12:
*"there should no be an 8s post-reward buffer. that was a carryover from the stroke orofacial
pipeline with a different task. the response window in our task is only 3.5s"*, and then the
framing that settles what the category actually is: *"I think the goal is to have non-running ITI
frames essentially"*, *"it SHOULDN'T be post reward, it should be post-cue"*.

She is right about the provenance. `stroke_orofacial_pipeline` has no `reward_buffer` of its own,
but its task is built end to end on an **8 s post-tone window** (`window_around_tones: [1.0, 8.0]`,
`end_ms: 8000`, "the (0, 8000) ms post-tone window"). The 8 was that task's event timescale,
imported as a number rather than as a reason. Ours is a **3.5 s response window**.

### Two measurements that make this more than a tidy-up

**1. The old category is epoch-dependent, and the deficit is what moves it.** Quiet as a fraction of
corrected frames, over the 92 sessions that carry a mask:

| epoch | n | mean | median | min |
|---|---|---|---|---|
| pre | 44 | 0.044 | 0.029 | 0.001 |
| **acute** | 16 | **0.171** | **0.151** | 0.055 |
| subacute | 18 | 0.103 | 0.093 | 0.002 |
| **chronic** | 14 | **0.031** | **0.007** | 0.004 |

The QUIET subtrahend is therefore estimated from a different fraction of the session, in a different
behavioural context, at every epoch — and at chronic from a **median 0.7% of frames**.

*(The first explanation offered for this — "post-stroke animals miss more, so earn fewer rewards, so
less of the session is buffered out" — was challenged by Priya on the grounds that
`reward_mode: auto_after_delay` delivers on nearly every trial and holds only after >6 consecutive
misses. She then noted misses do still reduce the total. Which term actually drives the ratio is
being measured per term rather than argued. ANSWER: it is the LICK term, which excludes 73.7% of
samples pre-stroke and 82.5% at chronic; reward is near-universal at a median 0.977 per trial.)*

**2. The QUIET reference's chronic agreement is an artefact, and the null proves it.** Mean pairwise
between-animal r of each animal's own `chronic − pre` map, against a null that re-pairs animals
across *different positions* (preserving shared anatomy, warp, rim and glue, destroying only
position-specific agreement):

| chronic − pre, QUIET | observed | **null mean** | p |
|---|---|---|---|
| Far Ipsi | +0.717 | **+0.504** | 0.213 |
| Far Middle | +0.693 | **+0.488** | 0.223 |
| Near Middle | +0.601 | **+0.498** | 0.399 |

The null is as high as the observed. Every one of those values is **position-independent shared
offset** — exactly what a baseline estimated from 0.7% of frames, biased the same way in every
animal, would subtract. **Nothing in QUIET's chronic column is interpretable under the old
definition.**

By contrast the acute far-contralateral result is real under every reference, with nulls near zero:
MEAN +0.819 (null +0.018, p=0.001), QUIET +0.769 (+0.040, p=0.001), PRECUE +0.856 (+0.083,
p<0.0005). *That* result does not depend on this migration.

---

## The new category, and its name

**"Quiet" is retired as a term**, because it meant two different things during the transition and
the old masks keep the old meaning on disk. Priya, on the first proposed name: *"it's not JUST ITI
though, because we're also excluding running"* — the category is **between trials AND not running
AND not licking**.

| | old | new |
|---|---|---|
| config block | `segmentation.quiet` | `segmentation.rest` |
| mask variant / directory | `quiet_<tag>` | `quiet_<tag>_rest` |
| the old definition's name | (unnamed, the default) | **`reward8`** — named for what was wrong with it |
| map reference key | `quiet` | `rest` |
| figure suffix | `_QUIETref_` | `_RESTref_` |
| state-decoder class | `quiet` | `rest` |

`reward8` names the old definition by its defect, so a figure or a note referring to it says which
baseline it used instead of silently meaning "whatever quiet was that week".

### What is retired vs what is deleted

**Nothing is deleted.** The `quiet_<tag>/` directories stay exactly where they are:

* they are the provenance of every figure produced before this migration, and
* Rule 1 makes `{mc}/` read-only anyway.

"Retired" means **no code reads them once `segmentation.rest.variant` is set to `rest`**. This is
the same rule `docs/PREPROCESSING_DECISION.md` already applies to the hemodynamic variants — the
original is never overwritten, every alternative gets its own directory beside it, and a manifest
records which definition produced it. Flipping back is one config line, which is the point: this
change moves a lot of results, and "did it move because of this?" has to stay answerable.

`quiet_periods.quiet_frame_path` resolves the variant in one place, and it does **NOT fall back
across definitions** — `fallback=False` is the default. An earlier version defaulted to True on the
reasoning that a partial cohort should "degrade per session rather than per figure"; that is
backwards, because falling back means a pooled map averages two different definitions of its own
subtrahend with nothing on the figure saying so. A session without the selected variant loses its
rest column instead, and `quiet_variant_used` reports what each session actually got.

---

## THE RE-RENDER LIST — everything in the blast radius

Priya, 2026-09-12: *"note that we will need to re-render anything in the quiet blast radius"*, and
*"we're going to have to re-do the state decoder as well as the quiet normalization for the maps"*.
Verified by reading each consumer rather than by grep alone.

### Depends on the mask — must be recomputed and re-rendered

| what | how it depends | outputs |
|---|---|---|
| **REST reference maps** | `position_reference_maps.session_quiet_svt` — the subtrahend itself | `15r`, `15rpa` `_RESTref_` |
| **State decoder** | `rest` is one of the three classes, **and the 1 s segment length was DERIVED from quiet's 1.10 s median bout** ("a 2 s window discards 83% of quiet") | `epoch_12b*`, `epoch_13*`, and the 0.92 → 0.81 / 0.98 → 0.95 headline in `BEHAVIOURAL_STATE_CONTROL.md` |
| **Position encoder** | `_quiet_baseline` — a time-local quiet median per component, explicitly "the stable cross-session reference for the pre/post-stroke residual" | the **0.749 → 0.286 → 0.745** convergence result |
| **Deck sections A–C** | `locanmf_cue_lick_analysis` z-scores every LocaNMF trace by quiet mean/SD (`_quiet_zscore`) | all within-day decode/encode figures |
| lick-aligned normalised maps | `locanmf_lick_aligned`, `plot_lick_aligned_averages`, `framemap_event_maps --quiet-frame`, `roi_activity` | those maps |
| preprocessing deck | `plot_running_activity_maps` quiet / running / running−quiet | preprocessing deck |

**The state decoder's window length is a DERIVED quantity, not a constant.** It is 1 s *because*
quiet's median bout was 1.10 s, so the derivation had to be re-run. **MEASURED, and it does not
change: 2 s fits 19.4% of the new rest bouts against the 17% that rejected it originally, while 1 s
fits 84.6% against 58%.** The window stays 1 s — which is what keeps the re-measured state-decoder
numbers comparable to the recorded ones. See the cohort result at the end of this document.

### Does NOT depend on the mask — untouched by this migration

**The position decoder never reads quiet.** `locanmf_position_decoder._build_signal` loads
footprint-scaled LocaNMF `C`, or raw `U`/`SVT`, with no quiet z-score anywhere in that path.
Therefore these stand unchanged:

* every headline decoding number, the frozen decoder, LOSO, pre-cue and post-cue accuracies;
* **figure 14's beta maps**, and the **MEAN** and **PRECUE** references;
* the acute far-contralateral result, which replicates under MEAN and PRECUE independently of the
  rest baseline.

---

## Order of work

1. **Measure** the candidate definitions and, critically, *which term* drives the epoch-dependence —
   treadmill, lick, reward or trial window. (`scratchpad/quiet_variants.py`: variants `reward8`,
   `reward4`, `noreward`, cue-anchored, full-trial ITI; plus the strobe→cue lead, which also decides
   whether a 2 s pre-cue baseline reaches back past the spout movement.) **Not yet complete.**
2. Fix the definition in `segmentation.rest`, with the measurement recorded beside each parameter.
3. Rename `quiet` → `rest` across config, code, reference keys, figure suffixes and deck text.
4. Recompute the masks into `quiet_<tag>_rest/` with a manifest. Nothing overwritten.
5. Flip `segmentation.rest.variant` to `rest`.
6. Re-render the blast radius above, then re-read every number into `DECISIONS.md` and
   `BEHAVIOURAL_STATE_CONTROL.md`.

**Until step 5, this branch changes nothing** — the resolver defaults to the original masks.


---

## RESULT — the cohort on the final definition (2026-09-12)

All **92 sessions** rebuilt, **`trial_start` anchor on every one**, none falling back to the strobe
or cue-only path. The cohort is not mixed.

### Rest fraction of corrected frames

| epoch | n | mean | median | min | max |
|---|---|---|---|---|---|
| pre | 44 | 0.082 | 0.065 | 0.011 | 0.185 |
| acute | 16 | 0.091 | 0.076 | 0.055 | 0.174 |
| subacute | 18 | 0.064 | 0.057 | 0.008 | 0.145 |
| chronic | 14 | 0.042 | 0.030 | 0.017 | 0.096 |

**The acute artefact is gone.** Ratio to pre: **acute 1.11** (retired definition: 3.89), subacute
0.78, chronic 0.51.

**And the chronic baseline is estimated from 4.3x more frames** — median 3.0% against 0.7%. That is
the specific quantity that made the retired definition's chronic column uninterpretable, where
between-animal agreement was position-INDEPENDENT shared offset (observed r = +0.494 against a
cross-position null of +0.497).

**CHRONIC IS STILL 0.51 OF PRE, and that is not the reward buffer.** It is the LICK term, which
excludes 73.7% of samples pre-stroke and 82.5% at chronic. Real behaviour, not a definition
artefact, and it survives under every candidate definition tested. The chronic column needs its own
caveat regardless of the baseline.

### Rest bout duration, and the state decoder's window

| epoch | n bouts | median | p75 | p95 | >=1 s | >=2 s |
|---|---|---|---|---|---|---|
| pre | 12,934 | 1.73 | 2.05 | 2.50 | 0.866 | 0.277 |
| acute | 5,992 | 1.63 | 1.86 | 2.15 | 0.904 | 0.108 |
| subacute | 4,447 | 1.57 | 1.83 | 2.24 | 0.808 | 0.130 |
| chronic | 2,637 | 1.34 | 1.73 | 2.11 | 0.681 | 0.088 |
| **ALL** | **26,010** | **1.63** | 1.92 | 2.37 | **0.846** | **0.194** |

**THE 1 s WINDOW STANDS, AND 2 s IS STILL REJECTED.** `BEHAVIOURAL_STATE_CONTROL.md` chose 1 s
because a 2 s window fitted only 17% of quiet periods while 1 s fitted 58%. On the new definition
2 s fits **19.4%** — barely moved — while 1 s fits **84.6%**. So the original rejection of 2 s is
unchanged and the 1 s choice is now much better supported, on nearly double the periods (26,010
against 14,017), which also means more independent units for a bootstrap clustered by period.

**A CORRECTION TO AN EARLIER ESTIMATE IN THIS DOCUMENT'S WORKING NOTES.** A subsample suggested 2 s
would become viable for ~45% of trials at a 0.5 s settle. That figure came from the raw INTER-TRIAL
GEOMETRY, before the treadmill and lick exclusions were applied; the bouts that actually survive
those exclusions are shorter, and the real number is 19.4%. The geometry sets an upper bound, not
the answer.

### What this does not fix

The lick term's epoch-dependence, above. And the chronic sample is still the thinnest: 2,637 bouts
across 14 sessions, median 1.34 s, with only 68% admitting a 1 s segment against 87% pre-stroke.


---

## 2026-09-13 — a position-graded amplitude change WITHOUT a position-specific pattern

Priya, on the REST null: *"so there's a position-independent amplitude increase in near positions
and amplitude decrease in far positions? seems like that could reflect real biology."*

**The two results are compatible, and the distinction that makes them so is that the null tests
SHAPE, NOT AMPLITUDE.** It correlates each animal's `epoch - pre` MAP against the others'. A cell
where observed ~= null means the four animals' change maps do not resemble each other beyond what
shared anatomy, warp and optics already supply -- it says nothing about whether the amplitude moved,
and nothing about whether it moved consistently.

So on the REST baseline, acute:

| position | amplitude (acute/pre) | position-specific shape? | nested bootstrap |
|---|---|---|---|
| Near Ipsi | 1.48 | no, p = 0.356 | 413 / 2,022 bins significant |
| Near Middle | 1.21 | no, p = 0.382 | significant |
| Near Contra | 1.27 | no, p = 0.714 | significant |
| Far Ipsi | 1.01 | no, p = 0.294 | — |
| Far Middle | 0.69 | marginal, p = 0.094 | significant |
| **Far Contra** | **0.48** | **yes, p = 0.001** | significant |

The reading those support together is **a broad gain change graded by spout position -- near up, far
down, monotone -- with a spatially SPECIFIC far-contralateral collapse on top of it**. The gradient
is the striking part: 1.48 at near-ipsilateral falling monotonically to 0.48 at far-contralateral,
across six positions that were analysed independently.

### WHY THIS IS NOT YET ESTABLISHED, and what would establish it

**1. NOTHING HERE HAS TESTED THE AMPLITUDE GRADIENT ITSELF.** The ratios above are pooled
within-animal values printed on the figure. The bootstrap tests each position against pre
SEPARATELY; the null tests map SHAPE. Neither asks whether the near-to-far ORDERING is consistent
across animals, which is the actual claim. **The test that would: per-animal amplitude ratios for all
six positions, then the nested animals->sessions bootstrap on the near-minus-far contrast, or simply
whether all four animals show the same monotone ordering.** That is cheap -- the maps are already
loaded by `position_reference_maps.maps_by_epoch` -- and it has not been run.

**2. A GLOBAL GAIN CHANGE AND A MOVING BASELINE ARE HARD TO SEPARATE HERE.** These maps are
`activity - rest`, and the REST baseline is not epoch-invariant: chronic retains only 0.51 of the
pre-stroke rest fraction, and the lick term alone excludes 73.7% of samples pre-stroke against 82.5%
at chronic. If the rest baseline itself shifts, `activity - rest` shifts with it, and a
POSITION-INDEPENDENT amplitude change is exactly the signature that confound would produce. That the
change is GRADED by position argues against a pure baseline artefact -- a baseline shift should
affect all six positions equally -- but the gradient has not been tested against that alternative.

**3. THE CHRONIC COLUMN SHOWS SHARED OFFSETS ARE REAL IN THIS DATA.** Under REST, chronic observed
r runs +0.43 to +0.72 against nulls of +0.50 to +0.51, every p > 0.2 -- position-independent shared
offset, unchanged from the retired baseline despite estimating chronic from 4.3x more frames. So
"the animals move together for reasons unrelated to position" is demonstrably a thing that happens
in this cohort, and it is the null hypothesis the amplitude gradient has to beat.

**4. n = 4.**

### What does NOT depend on any of this

The far-contralateral acute result. It is position-specific under every reference tested
(MEAN +0.819 / REST +0.812 / PRECUE +0.856, nulls 0.018-0.083, p <= 0.001), it survived the entire
baseline migration essentially unchanged from the retired definition's +0.769, and the position
decoder never reads the rest mask at all.


---

## 2026-09-13 — the REST baseline and the block structure: an alarm, and what it actually was

Priya asked the question the rest reference had never been asked: *"can we test if the REST activity
shows significant difference between trials of different positions... do rest periods between all 6
positions look similar to or different from each other?"*

The reference makes two claims. The first -- that the subtrahend is IDENTICAL for all six positions
-- is true by construction: it is one session mean, so it cannot couple them. The second is
load-bearing and had never been tested: **that it carries no position information, so subtracting it
removes nothing real.**

### The alarm

Rest periods labelled by position ONLY where the preceding and following trial share one (so the
label is unambiguous; positions run in ~6-trial blocks, so this is the common case). Each position's
rest map minus the animal's own across-position mean, nested animals->sessions bootstrap vs zero,
eroded stat mask, max-statistic threshold -- the same machinery every map figure uses:

| position | bins significant / 2,022 | edge enrichment |
|---|---|---|
| Near Ipsi | 15 | 0.00 |
| Near Middle | 436 | 0.79 |
| **Near Contra** | **1,042** | 0.39 |
| Far Ipsi | 210 | 1.24 |
| **Far Middle** | **973** | 0.33 |
| Far Contra | 215 | 0.26 |

6/6 positions significant, low edge enrichment everywhere, between/within-position RMS ratio 1.45.
Read at face value: the REST reference is subtracting a position-weighted mixture, and the
position-graded amplitude change (1.48 near -> 0.48 far) might be manufactured by it.

### What it actually was

**POSITIONS ARE PRESENTED IN ~6-TRIAL BLOCKS, so position is confounded with TIME-WITHIN-SESSION.**
Slow drift -- photobleaching, arousal, the rest baseline's own documented drift -- makes rest during
block k differ from block j, and blocks carry position labels. Splitting each position's rest frames
at the session midpoint separates the two in the same units, within animal:

    DRIFT     same position, EARLY half vs LATE half        RMS 0.00282   n = 224
    POSITION  different positions, MATCHED halves           RMS 0.00288   n = 1,048
    POSITION / DRIFT                                        **1.02**

**Rest differs between positions by essentially exactly as much as the same position's rest differs
from itself across a session.** It is drift aliased onto the block structure, not position coding.

### What follows, and what does not

* **The REST reference is NOT conceptually broken.** The first test alone could not have shown that;
  reported without the drift control it would have read as a fatal objection to the reference this
  deck just adopted as primary.
* **The amplitude gradient is not explained away by position-dependent rest.** It could still be
  touched by drift if block ORDER is systematic across sessions -- worth checking before leaning on
  the gradient, and not yet checked.
* **A REAL DEFECT IS EXPOSED, and the fix already existed in this repo.** The map reference
  subtracts ONE SESSION MEAN, which is flat and cannot remove drift. `locanmf_position_encoder.
  _quiet_baseline` has always used a TIME-LOCAL baseline -- "bin the session into nbins, take the
  median of quiet frames per bin, interpolate to every frame -> tracks slow drift". The map
  reference and the encoder were computing the same quantity two different ways, and the encoder's
  is the correct one. `position_reference_maps.session_rest_svt_timelocal` now implements it on the
  map side (12 bins, median per bin, interpolated; measured to carry SD 0.047 of across-time
  structure that the session mean is blind to by construction).
* **STILL TO DO: wire it into the subtrahend.** `session_raw_maps` averages over a position's trials
  and subtracts one map; using a time-local baseline means evaluating it at THOSE TRIALS' times,
  which needs per-trial frame indices that function does not currently carry. The baseline exists
  and is verified; the substitution is not yet made, so **every REST-referenced figure on the share
  still uses the flat session mean.**

### A methodological note worth keeping

The first version of this control printed "no significant position dependence" while testing ZERO
positions -- it intersected each animal's positions across all its sessions, one thin session
emptied the intersection, and the loop never ran. Priya caught it on the implausibility of the
numbers ("how could so few animals share the same positions? there should be dozens of each per
session"). The verdict now refuses to print a negative when nothing was tested. A control that
reports "passed" without executing is worse than one that fails.


---

## 2026-09-13 — the behaviour log DOES carry spout dock timing, and REST currently starts ~0.65 s too early

Priya: *"the behavior log should have the spout dock arrival and leave timing."* It does, and the
limitation recorded in the correction above -- "the pipeline cannot see the retraction" -- is
**withdrawn**. `events.csv` carries `dock_start` and `dock`, one of each per trial, alongside
`trial_start`, `position`, `cue` and `reward`.

### The measured trial cycle

Medians relative to the cue, over six sessions spanning 5/19 to 9/08 (91-599 trials each):

| event | median s after cue | what it is |
|---|---|---|
| `cue` | 0 | |
| `reward` | 0.13 | |
| `dock_start` | **3.68 – 3.88** | the spout BEGINS retracting |
| `trial_stop_ttl` | +0.01 after dock_start | |
| `dock` | **4.61 – 4.80** | the spout is AWAY (≈0.92 s of travel) |
| `trial_start` | 5.88 – 6.11 | next trial opens |
| `position` | 6.77 – 7.00 | the spout ARRIVES at the next target (≈0.90 s of travel) |

**NO TARGET IS PRESENT FOR ~2.15 s**, from `dock` to the next `position`, and that figure is stable
to ±0.05 s across sessions (the 5/19 session, an early one, runs 2.58 s).

### What this says about the REST window we are using

REST currently runs `cue + response_window + 0.5 s` → next `trial_start`, i.e. **4.0 s → ~6.0 s**.
Against the measured cycle:

* **IT STARTS ~0.65 s TOO EARLY.** `dock` is at ~4.65 s, so the first third of every REST window
  contains the spout PHYSICALLY RETRACTING -- a moving object the animal can see and may track.
  The 0.5 s settle was derived as a margin after reward; it happens to land mid-retraction.
* **IT ENDS ~0.9 s EARLY**, at `trial_start`, while the spout does not reach the next target until
  `position`. That is the right call for a different reason -- see below -- but the interval is
  available and is currently unused.

**THE PRINCIPLED WINDOW IS `dock` → next `trial_start`**: ~1.35 s in which there is no target AND no
spout movement. Shorter than the current 2.0 s, and clean rather than nearly clean. The interval
`trial_start` → `position` (~0.9 s) is a THIRD thing again: no target yet, but the spout is moving
toward a position the animal can often predict from the block -- which makes it the natural window
for an anticipation analysis rather than for a baseline.

### Why this matters beyond tidiness

`rest_position_permutation` found rest carries position information across the cohort
(observed/null **1.429**, above null in **41/44** sessions). A window containing spout retraction
offers an obvious mundane explanation for part of that: the retraction STARTS at the position the
spout was at, so early-REST frames contain movement whose trajectory differs by position.
**Re-running the permutation on a `dock`-anchored window is the control that separates "position
information with no target present" from "the tail of the retraction".** Until that is run, the
1.429 stands as measured but its interpretation does not.

### TO DO

1. Add a `dock`-anchored REST variant and re-run `rest_position_permutation` on it. If the ratio
   survives, the position signal is genuinely target-free.
2. The device clock needs mapping to DAQ samples; `spout_behavior.load_gui_licks` already extracts
   the `sync` heartbeat for exactly this, so the machinery exists.
3. `segmentation.rest` gains the anchor as a config option rather than a literal, alongside the
   measurements above, per the pattern every other rest parameter follows.


---

## 2026-09-13 — spout travel time is POSITION-DEPENDENT and reproducible to 0.04 s: reconstructable, and a mechanism

Priya: *"the spout takes the same amount of time to move from position to dock for each position
across sessions (zaber speed is always equal) - so we should be able to reconstruct."*

**CONSTANT WITHIN A POSITION ACROSS SESSIONS, DIFFERENT BETWEEN POSITIONS** -- exactly as Priya
meant it: the Zaber speed is fixed, so travel time is set by distance, and distance is a property of
the position. Measured over **110 sessions** (June onward, all four animals), `dock_start` -> `dock`
per position:

| pos_idx | median travel (s) | sd across sessions |
|---|---|---|
| 0 | **0.775** | 0.038 |
| 1 | 0.975 | 0.038 |
| 2 | 0.976 | 0.037 |
| 3 | **0.649** | 0.037 |
| 4 | 0.949 | 0.039 |
| 5 | 0.951 | 0.037 |

The spread ACROSS positions is 0.33 s; the sd ACROSS SESSIONS within a position is 0.038 s. So the
between-position difference is nearly **nine times** the between-session variability, and the
per-position constants reproduce across animals to ~0.01-0.03 s. `trials.csv` also carries
`pos_dist_mm_before_trial` / `pos_dist_mm_after_trial`, so the distance is recorded directly.

### Consequence 1 — the reconstruction works, PER POSITION and not as one constant

A session whose GUI/DAQ clocks cannot be aligned (PS93 6/6) can have its dock times rebuilt from DAQ
anchors plus the per-position constant above, instead of dropping out. Using ONE constant would
inject a position-dependent error of up to 0.33 s into the window start -- which is precisely the
kind of error that manufactures a position effect in an analysis asking whether rest carries
position. **The reconstruction must be per position or it is worse than dropping the session.**

### Consequence 2 — this is a REAL mechanism for position information in a loose rest window

The retraction takes between 0.65 s and 0.98 s DEPENDING ON WHERE THE SPOUT IS COMING FROM. The old
rest window opens ~0.65 s before `dock`, so the amount of retraction it contains is itself
POSITION-SPECIFIC: a window starting at a fixed offset from the cue captures nearly all of position
3's short retraction and only part of position 1's long one. That is a concrete, mundane route from
"rest window" to "position information" with no neural interpretation at all.

**Which is exactly why the docked window had to be tried, and why the result surviving it matters.**
On the strict `dock` -> next `trial_start` window the permutation still gives observed/null 1.338
(4/5 sessions, first pass) against 1.429 on the loose one. The mechanism above is real and is now
excluded.

### TO DO

1. Add the per-position travel constants as the FALLBACK path in `docked_periods`, used only when
   `_sync_affine` refuses, and label sessions reconstructed that way so a result can be re-run
   without them.
2. **Redo every REST analysis on the docked definition once the investigation closes** (Priya:
   *"we're gonna have to redo all the 'rest' analyses with the new version - after we finish our
   investigation of it"*). That is the rest baseline for the maps, the state decoder's REST class,
   `behavior_events` schema v4, and every figure downstream of them.
