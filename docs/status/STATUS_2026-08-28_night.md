# Handoff, 2026-08-28 night — epoch figures, the engagement gate, and the parallel render

## What we are doing and why

Priya asked for **pooled cross-animal grant figures stratified by RECOVERY EPOCH** — pre / acute /
subacute — instead of a linear time axis, sized to be read at a quarter page. Four animals with
different lesion dates and different cadences are incomparable at every x position on a calendar
axis; epochs make them comparable. Everything below either builds those, or fixes something they
exposed.

`wfield_local/epoch_grant_figures.py` renders them into `<labcams>/grant_figures/epoch`;
`wfield_local/epoch_figures.py` holds the renderers and the statistics;
`wfield_local/epochs.py` holds the boundaries. **None of it recomputes anything** — every
population comes from `grant_figures`' existing collectors, and a test forbids this module from
fitting a model or pooling sessions itself.

## The figures, and what each is built from

| figure | source | pooling |
|---|---|---|
| 1b behaviour by position | `_position_metrics` | session-weighted; trials sum |
| 1c time course + epoch boundaries | `_position_metrics` | one dot per animal per day |
| per-position decoding accuracy | `_collect_5c` | trials sum (raw counts add) |
| 5c/5d frozen confusion + deltas | `_collect_5c` | trials sum |
| 6 / 7 / 8 matrices + diagonals | `_matrices_{pattern,splithalf,crossnobis}` | **mean over sessions** |
| 8g geometry per position | `_rdm_rows` | mean over sessions |
| 9 delta trajectory | `_delta_cis(full=True)` | mean over sessions |
| 10 / 10b best match | `_matrices_pattern` | mean over sessions |
| 11 encoder variance + gain | `_enc_tables` | mean over sessions |

**The two pooling rules are not interchangeable.** Confusion collectors return raw counts, so
pooling is a SUM and every trial counts once. The `_matrices_*` and scalar collectors return
values already reduced per session, so pooling is a MEAN and the session is the unit — which is
the weighting Priya asked for, and why the session dots matter.

## Statistics

Priya chose **session-level clustered by animal**, then asked for blocks nested within session.

* **Trial-level families** resample **animals → sessions → blocks**. The animal draw is shared
  between the two epochs so the contrast stays paired.
* **Per-session scalar families** resample **animals → sessions**, with NO block level, because
  their collectors already reduced the trials away. Every subtitle states which was used.
* **Marks are two-level**: `*` the 95% interval excludes zero, `**` it survives Bonferroni across
  every comparison on that figure. Both, because with four animals the corrected interval is wide
  enough that reporting only it would blank every real effect, and reporting only the uncorrected
  one would call twelve comparisons one.
* Both come from **one** set of draws, so the corrected interval necessarily contains the
  uncorrected one.
* Every bar carries its **own** CI from that same resampling, and every bar family has a companion
  interval panel showing the effect size and what the correction costs.

## Discoveries worth carrying forward

**The render had never been reproducible.** Every bootstrap seed was
`abs(hash((animal, align, variant))) % 2**31`, and Python salts string hashing per process — three
consecutive interpreters gave 1125027485, 2138950357, 223190567 for the same tuple. Point estimates
held; every confidence interval moved on every rerun. Now blake2b of the labels.

**Where the render's time actually goes**: 94.7% of 5.79 h is six bootstrap families (7b, 8d, 6d,
7d, 8b, 8e), each writing five files at 10–37 min. Collection is ~20 s. That overturned my earlier
claim that the unit of parallelism had to be the figure FAMILY — it is the FIGURE.

**The bootstrap cache works**: one 7b unit, 18 min 38 s cold → **21 s warm**, keyed on the input
BYTES rather than a session name.

**The behaviour engagement gate was circular.** `flag_engagement` judged the trailing response rate
over ALL positions, so post-stroke it called the animal disengaged precisely because it could not
reach the far spouts — and dropped those trials from the hit rate meant to measure the deficit. It
excluded 380 of PS94_0817's 643 trials; the reference-judged gate excludes 0.

**A figure's height must be a RESULT, not a budget.** `fig_h` was the constant 2.60 and the axes
took what the text left over, so the figure with the most chrome got the least plot -- 8g ran a
0.68in plot under 1.92in of text, 26% of the canvas. Height is now `top + bar_axes_height(fig_w) +
bottom`. The same fault existed rotated ninety degrees: narrowing the canvas handed the reclaimed
width to the LEGEND, leaving a 0.57in axes. Reclaiming space says nothing about who receives it.

**A cache that memoises failures makes one bad run permanent.** Every bootstrap producer signals
failure by returning None through a broad `except`, and `_boot_cached` pickled it. A swallowed
NameError poisoned twelve keys, and the CORRECTED code then read them back and produced nothing --
silently, both times. Falsy results are no longer persisted.

**Three fault classes the layout checker cannot see**, all found by driving:
1. A figure sized per panel count renders type 25% smaller when the deck places it at a fixed
   width — no overlap anywhere.
2. Text that runs OFF the canvas collides with nothing. `_overlaps` compares artists to each
   other; `off_canvas` now reports what falls outside, in inches over.
3. A rule drawn through tick labels is a Line2D over text, which no text-vs-text check covers.

## Decisions

**Engagement (behaviour only).** `reference_engagement` = the imaging gate's sustained-collapse arm
UNIONed with a terminal-tail arm, both counting REFERENCE trials (close_L, close_center) only.
Measured: old 7181 excluded (17.9%), imaging-only 4043 (10.1%), union 4274 (10.6%). The union costs
231 trials over the imaging gate and recovers sated tails it was too slow to see (PS95_0811:
collapse 21, tail 66).

**The imaging gate is NOT changed.** Adopting the union there would move **111 trials, 0.81% of the
working class**, with 20 of 26 post-stroke sessions unchanged — against re-deriving every published
imaging number. Behaviour and imaging now differ on ~0.5% of trials, documented rather than silent.

**Deltas are measured against PRE-STROKE**, not against the previous epoch. `subacute − acute`
answers "did it recover from its worst point"; `subacute − pre` answers "has it returned to
baseline", which is what a recovery figure is asked.

**Positions are named by anatomy**, derived from `stroke_laterality`: nI/nM/nC, fI/fM/fC. A mixed
cohort RAISES rather than mislabelling — a pooled figure keeping one label set would average ipsi
with contra under one name.

## Running right now

| | |
|---|---|
| other window | `poststroke_section_g --animals PS92 PS93`, and `spout_behavior 20260828 --cohort` |
| next | the FINAL epoch render, then ONE deck build |

**Do not build the deck while `poststroke_section_g` is writing.** A deck built mid-write places
partial section G figures and reports **0 missing**, because a partially-written figure is present.

## Pending

1. **Final epoch render** to `<labcams>/grant_figures/epoch` — everything: 1b, 1c, accuracy, 5c/5d,
   matrices 6/7/8 with their diagonal bars, and scalars 8g/9/10/10b/11, each with its CI companion.
2. **One deck build** once section G and the cohort run finish. Section I is already wired
   (`locanmf_analysis_deck.py`), with the speaker notes written as grant figure legends.
3. **Today's PS92/PS93 sessions** are not in the figures: 0828 is not registered in
   `sessions.yaml`, and registration is imaging-keyed — no `20260828` directories under labcams
   yet. Registering an imaging-less session would put it in `phase_labels("post")` where the
   imaging collectors would try to pool a session with no `motion_corrected`; those loops catch per
   ANIMAL, so the likely result is PS92 and PS93 vanishing from the imaging epoch figures. Wait for
   the upload. Their behaviour has already been re-scored under the new gate.
4. ~~**Bootstrap cache** for `_rdm_ci`~~ **DONE.** Cached per ANIMAL, which is what makes it
   sound: the loop runs draws-outer/days-inner so an animal's days share each draw's reference
   resample, and the per-DAY split I had feared is exactly the one that breaks it. `_asymmetry_ci`
   (8e) is still uncached and has the same shape, so it is the same change whenever it is wanted.
5. ~~**Age-based prune**~~ **DROPPED, measured.** 61 entries, under 0.05 MB in total, against a
   2.4 GB session cache. It would have been machinery for a problem that does not exist.

## Method notes

- **Verify a layout fix by DRIVING the function**, with the REAL titles and labels. Every layout
  fault this week was found that way and none by reading the diff.
- **Both checks, always**: `_overlaps` for artists against each other, `off_canvas` for artists
  against the edge. A figure needs both clean.
- **A `None` return is an ABSENCE.** `print(f"wrote {fn(...)}")` renders it as "wrote None", which
  scans as success — that is how the per-position accuracy figure came back empty three times.
- **`--only` guards must list every key.** The arm loop skipped itself for `--only scal` and
  produced nothing, exit 0, no error.
- **Heredocs through Git Bash halve `\n` escapes.** Four syntax errors this session. Write patch
  scripts to a file, or use `chr(10)`.
- **The shared checkout**: stage explicit paths, never `git add -A`. When origin has moved and the
  other window has dirty files, push through a temporary detached worktree rather than stashing
  their work.
