# HANDOFF 2026-09-21 — codebase quality: efficient, modular, readable, editable

**The analysis threads are at a natural stopping point.** Read
[`STATUS_2026-09-20.md`](STATUS_2026-09-20.md) for where the science landed and
`DECISIONS.md` from **"WHERE THE TWO THREADS LANDED"** for the reasoning. **This document is about
the CODE**, at Priya's direction (2026-09-20): *"i want to focus on making the codebase efficient
and modular, as well as readable and editable, as a claude engineer would do"*.

---

## 0. READ THIS FIRST — WHAT NOT TO BREAK

This repo's documentation culture is its best asset and is **not** clutter to tidy away.
`DECISIONS.md` (15.7k lines) and the long module docstrings record *why* a thing is the way it is,
usually because the alternative was tried and failed. **Refactors must carry the reasoning
forward, not delete it.** Several comments name a specific bug that a "simplification" would
reintroduce.

Three rules that are load-bearing and must survive any restructuring:

- **Ground rule 0/1 (CLAUDE.md):** never delete original data; only ever write inside
  `MICROSCOPE/Priya/…`; never delete on the server.
- **Ground rule 6:** per-session loops fan out over cores via `wfield_local/parallel.py`.
- **Ground rule 8:** print the per-animal table before the bootstrap.

**Behaviour-preserving is the bar.** Every refactor below can be verified the same way the
parallelisation was: run before, run after, diff the per-session output. That method caught a real
CI shift that a passing test suite would have missed.

---

## 1. THE MEASURED PROBLEM

Counts taken 2026-09-20, not estimated:

| | |
|---|---|
| `wfield_local/locanmf_analysis_deck.py` | **5,468 lines**; `build_analysis_deck()` alone spans 1550→5440 |
| `wfield_local/epoch_grant_figures.py` | **3,942 lines** |
| `scripts/rest_migration/` | **74 modules, 19,208 lines** |
| modules defining their own `_boot*` helper | **9** (two of them define two each) |
| modules repeating the `phase_labels("pre")` filter | **70** |
| modules loading licks via `_load_daq_events` directly | **37** |
| per-session loops NOT fanned out | **8 of 11** in `rest_migration` |
| epoch figures on disk but unregistered in the deck | **25 of 390** |

**THE REPO ALREADY KNOWS DUPLICATION IS ITS FAILURE MODE.** `rest_by_position`'s docstring: *"The
last time one quantity had two implementations in this repo -- the flat map baseline against the
encoder's time-local one -- they disagreed for months and the map side was the wrong one."* And
`parallel.py` exists precisely because *"`grant_figures` grew a process pool and the stages around
it did not"*. The nine `_boot` copies are the same pattern, unresolved.

---

## 2. THE WORK, IN DEPENDENCY ORDER

### 2.1 Extract a shared analysis toolkit (highest value, lowest risk)

Create `scripts/rest_migration/_common.py` (or better, `wfield_local/analysis_kit.py` if the
pipeline should use it too) holding the four things every module re-implements:

1. **`boot_ci(by_animal, rng)` and `boot_delta(pairs, rng)`** — the nested animals→sessions
   bootstrap and its paired difference-of-differences form. **Nine copies today.** They are not
   all identical, which is the danger: `rest_coupling` returns the mean of the FLAT pool as its
   point estimate while some others return the mean of animal means. **Pick one, document which,
   and state it in the docstring** — the distinction changed a reported number today (pooled
   4496/4338 against paired −1270).
2. **`curated_sessions(animals=None, epochs=True)`** — the `want`-set filter repeated in 70
   modules.
3. **`session_behavior(s)`** — cue samples, lick samples, DAQ rate, trials, engagement gate. The
   four-import incantation in `quit_prodrome`, `lick_bout_structure`, `evoked_hrf_latency` and
   `channel_position_maps` is character-identical.
4. **`fan_sessions(labels, worker, jobs)`** — `fan_out` plus **sorted collection**, so the
   completion-order trap is impossible to re-introduce rather than merely documented.

### 2.2 Finish the parallelisation (mechanical, measured payoff)

8 of 11 `rest_migration` loops are still serial. Converted: `quit_prodrome` 36→6 min,
`rest_coupling` 85→9m43s (×8.7), `evoked_hrf_latency` 90→~12 min, all byte-identical. **Use 2.1's
`fan_sessions` rather than hand-wiring each**, and remember the two traps recorded in ground rule
6: sorted collection, and options travel in the ITEM because spawn does not inherit globals.

### 2.3 Make the deck registry importable

`_EPOCH` is a **local** inside `build_analysis_deck()`. Nothing can import it, so
`scripts/deck_figure_coverage.py` must regex-scrape the source — and that misses patterns built by
f-string, which `DECISIONS` already records as reporting live figures as orphans.

**Move `_EPOCH` (and its siblings) to module scope.** Then a test can assert every figure
`epoch_grant_figures` renders is registered. The code's own comment says the previous scheme left
*33 of 75 figures unreferenced*, and 25 are unregistered right now — the deck's completeness check
"reports figures it EXPECTS and is silent about ones it was never told about".

### 2.4 Split the two giant modules

`build_analysis_deck()` is ~3,900 lines in one function. The figure registries are data and should
be `deck_registry.py`; the slide-construction helpers are machinery and should be `deck_layout.py`;
what remains is orchestration. Do this **after** 2.3, since moving the registries out is most of
it.

### 2.5 Standardise intermediates so re-runs are cheap

`rest_coupling --from-csv` re-derives every table from its CSV in seconds. **Nothing else has
it**, and the absence cost two full re-runs today: ILI was not written to CSV so the per-animal
check needed a whole pass, and `channel_position_maps`' change-from-pre contrasts are console-only
and cannot be re-derived at all.

**Rule to adopt: any number that appears in a summary table must be reconstructible from a written
CSV.** Also unify the output directory — `channel_position_maps` writes to `channel_comparison/`
while everything else writes to `grant_figures/epoch/`.

---

## 3. SUGGESTED SEQUENCE

1. `_common.py` with the four helpers + tests that pin the bootstrap convention (**2.1**)
2. Migrate the three already-parallel modules onto it, verify byte-identical output
3. Convert the remaining 8 loops (**2.2**)
4. Registry to module scope + a coverage test (**2.3**)
5. `--from-csv` everywhere + unify output dirs (**2.5**)
6. Split the deck module (**2.4**)

---

## 4. HOW TO VERIFY A REFACTOR HERE

Tests exist (`tests/`, including `test_analysis_deck.py`, `test_deck_*`) but they do not cover the
analysis scripts. **The method that actually worked today:**

```
run module          > before.txt
<refactor>
run module          > after.txt
diff the per-session lines, ignoring order
```

That caught a CI moving from [−22.9, −13.7] to [−23.1, −13.6] while the point estimate stayed
exact — a real reproducibility bug no assertion in the suite would have flagged. **Numbers are the
test.**

---

## 5. ANALYSIS ITEMS STILL OPEN (do not start these; they are context)

- **Restate the far-contra amplitude gradient under `restw`, not `rest`** — a numbers-in-text job
  against a figure that already exists (`epoch_15r_position_RESTWref_*`).
- **DLC tongue tracking** — the blocker for "did not try" vs "tried and missed", and the only
  thing that would turn the within-bout deceleration argument into a control.
- **530 nm reflectance** — DEFERRED hardware. Do not spend analysis effort on 470/415 coupling.
- **25 unregistered epoch figures** — mostly `_erodedgate`/`_mf075` sensitivity variants of the
  rotation and reference-family threads. I did not caption them because I have not read those
  analyses closely enough to do it honestly; someone who has should.
