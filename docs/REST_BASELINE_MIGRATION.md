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
