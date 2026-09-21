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
| modules defining their own `_boot*` helper | ~~**9**~~ → **0** (2026-09-21, see 2.1) |
| modules repeating the `phase_labels("pre")` filter | ~~**70**~~ → **62** (8 migrated) |
| modules loading licks via `_load_daq_events` directly | ~~**37**~~ → **~35** |
| per-session loops NOT fanned out | **7 of 11** in `rest_migration` |
| epoch figures on disk but unregistered in the deck | ~~**25 of 390**~~ → **33 of 402** (recounted 2026-09-21; see below) |

**THE REPO ALREADY KNOWS DUPLICATION IS ITS FAILURE MODE.** `rest_by_position`'s docstring: *"The
last time one quantity had two implementations in this repo -- the flat map baseline against the
encoder's time-local one -- they disagreed for months and the map side was the wrong one."* And
`parallel.py` exists precisely because *"`grant_figures` grew a process pool and the stages around
it did not"*. The nine `_boot` copies are the same pattern, unresolved.

---

## 2. THE WORK, IN DEPENDENCY ORDER

### 2.1 Extract a shared analysis toolkit — **DONE 2026-09-21**

`wfield_local/analysis_kit.py` + `tests/test_analysis_kit.py`; eight modules migrated, 393 lines
deleted against 179 added. `DECISIONS.md` → *"THE SHARED ANALYSIS TOOLKIT, AND THE THIRD ORDERING
BUG IT TURNED UP"* has the full account. What landed:

1. **`boot_ci(by_animal, rng)` / `boot_delta(post, pre, rng)` / `boot_delta_pairs(pairs, rng)`** —
   nine copies collapsed to one. **The warning below was WRONG and is kept for the record:** all
   seven CI copies took the flat-pool mean, `rest_coupling` included. The real differences were an
   empty-pool guard missing from two of them, and one copy returning the animal count as a fourth
   element (now `Interval.n_animals`). ~~`rest_coupling` returns the mean of the FLAT pool as its
   point estimate while some others return the mean of animal means.~~ The two conventions that DO
   differ are `boot_ci` (flat pool, for LEVELS) and `boot_delta` (animal-weighted, for CHANGES);
   both docstrings now say which and why, because that distinction is what retracted the
   4496/4338 number.
2. **`curated_sessions` / `curated_labels`** — the 70-module filter, calling `config.pooled_labels`
   rather than re-deriving it. **Defaults to `load_sessions` order, NOT sorted**, because that
   list is unsorted and the pools are iterated into a seeded RNG.
3. **`session_behavior(lab, gate=…, horizon_min=…)`** plus `daq_rate` and `lick_samples` — the
   character-identical block from `quit_prodrome` and `lick_bout_structure`, including both trial
   floors. It also removes two scripts importing a third script's private `_daq_rate`.
4. **`fan_sessions(items, worker, jobs=…)`** — `fan_out` plus sorted collection, so the
   completion-order trap is now unrepresentable rather than merely documented.

**A THIRD ORDERING BUG FELL OUT OF VERIFYING IT.** `rest_coupling --from-csv` was nondeterministic
run to run — `for lab in {set comprehension}` iterating in `PYTHONHASHSEED` order into a seeded
RNG. Fixed with `sorted()`. The general rule is now CLAUDE.md ground rule 9: **any container that
feeds a bootstrap pool must have a defined order.**

**STILL OPEN FROM 2.1:** the `_load_daq_events(…, "lick_analog", 2.5, 1.0, (0.001, 0.020), 0.10)`
incantation is still written out in ~35 places, most of them in `wfield_local/` core modules that
want the whole returned dict rather than just `lick_samples`. `analysis_kit.lick_samples` exists;
sweeping the call sites is a separate change, and several of them are on the nightly path.

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

### 2.4 Split the two giant modules — **DONE 2026-09-21 (the deck; `grant_figures` still open)**

    locanmf_analysis_deck  5,484 -> 2,134   orchestration, gates, provenance, captions
    deck_registry                  2,163   EPOCH_FIGURES, GRANT_FIGURES, ALIGNS, BASES, legends
    deck_text                      1,187   TRIALS_* / S_* / M_* -- the slide prose
    deck_layout                      243   SlideCanvas
    build_analysis_deck    3,892 -> 1,679

Verified two ways: 87 module constants hashed exactly before/after, and a **531-slide** deck
fingerprinted shape-for-shape (`scripts/deck_fingerprint.py`, new). The 531 matters — a local build
reaches only 475 slides because `figures_working` here has three PNGs, so the tool builds a stub
tree from the last nightly's manifest and sections A–G are exercised too. The fingerprint was
mutation-tested first. `DECISIONS.md` → *"SECTION 2.4 FINISHED"* has the account, including the
five tests and two audit scripts that went half-blind and the numbers that measured it.

**`grant_figures.py` 6,599 → 6,040**, with `grant_kit.py` (656) taking the twenty shared helpers —
chosen by call graph, not by eye: they call nothing outside themselves and 45 functions call into
them. Verified by a full render both sides: 96/96 units, **90 of 90 PNGs byte-identical**.

**THE REMAINING FAMILY SPLIT IS MEASURED AND WAITING**, so it needs no survey:

| | functions | lines |
|---|---:|---:|
| shared by >1 family (already in `grant_kit` or candidates for it) | 68 | 1,894 |
| private to ONE family | 47 | 3,335 |
| largest single family (`fig_encoder_gain_shape`) | — | 438 |

So it is ~6 family modules over `grant_kit`. **Do it while the bootstrap cache is warm:** the cold
baseline render took ~2 h, the verification re-render took minutes. `epoch_grant_figures.py`
(3,942) is the same shape and the same argument.

Two method notes for whoever does it. **SVG byte-comparison is not a verification method** —
matplotlib writes a `<dc:date>` and random element ids, so all 90 SVGs "differ" on every run; use
`scripts/compare_svg_renders.py`. And **module state does not survive a module boundary**:
`_ONLY_WINDOW`/`_ONLY_VARIANT` had to become `grant_kit.set_only()`, because a global assigned in
one module and read in another is two variables and the failure is silent.

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

1. ~~`_common.py` with the four helpers + tests that pin the bootstrap convention (**2.1**)~~
   **DONE** — `wfield_local/analysis_kit.py`
2. ~~Migrate the already-parallel modules onto it, verify byte-identical output~~ **DONE** — all
   four, plus `quit_point`, `nvc_evoked`, `engagement_decomposition`, `channel_position_maps`
3. ~~Convert the remaining 7 loops (**2.2**)~~ **FOUR DONE 2026-09-21** — `quit_point`,
   `engagement_decomposition`, `nvc_evoked`, `channel_position_maps`; **all four verified** against
   a `git worktree` at HEAD -- the first three diff-identical, `channel_position_maps`
   order-insensitively identical with all four written artefacts byte-identical. Use `analysis_kit.input_order`, not the default
   alphabetical collection, or the CIs move. Remaining serial loops are the small diagnostics
   (`channel_evoked_sign`, `channel_vessel_sign`, `cutoff_scaling`, `onset_edge_bias`,
   `restw_reliability`, …) — none on the nightly path
4. ~~Registry to module scope + a coverage test (**2.3**)~~ **DONE 2026-09-21** —
   `EPOCH_FIGURES` (109 entries) is importable; `build_analysis_deck` 3,892 → 2,116 lines;
   `tests/test_epoch_registry.py` asserts every `name=` the renderer emits has an entry
5. **2.5 PARTLY DONE 2026-09-21.** `--from-csv` added to `engagement_decomposition`
   (4 m 50 s → **10.7 s**) and `channel_position_maps` (3 m 32 s → **23.6 s**), both verified to
   reproduce their full run's tables exactly; `analysis_kit.read_rows` is the shared reader.
   `quit_point`, `nvc_evoked`, `quit_prodrome` and `lick_bout_structure` still lack one and each
   needs a SECOND artefact written first (quit-aligned hits, pooled curves, nested per-session
   records) — a `--from-csv` rebuilding only the table half would break the rule it serves.
   ~~The output-dir unification is NOT done and needs a decision~~ **DONE 2026-09-21**:
   the two POOLED figures go to `grant_figures/epoch` and ARE registered (109 → 111 entries); the
   sixteen per-session maps stay in `channel_comparison` as diagnostics. Nothing on the share was
   moved or deleted. Verified byte-identical to HEAD via `--from-csv`
6. ~~Split the deck module (**2.4**)~~ **DONE 2026-09-21** — four modules, verified by a
   531-slide fingerprint; `grant_figures.py` is the remaining giant

**ADDED AND DONE 2026-09-21, at Priya's direction:**

7. **The output tree.** `grant_figures/epoch` was 1,747 files flat, 54% of them not figures.
   `wfield_local/figure_layout.py` now defines one layout (`<name>.png`, `svg/`, `data/`) and
   `scripts/restructure_output_tree.py` moved 1,671 files to match — nothing deleted, idempotent,
   dry-run by default. The nightly mirror was renamed from
   `labcams/locanmf_lick_pooled/cue_analysis` to `labcams/analysis_figures`: it is the live mirror
   of every analysis figure and it had been sitting inside a directory named after an abandoned
   June 2026 pooling experiment.
8. **`scripts/rest_migration` 74 → 43 live + 31 archived**, nothing deleted, with a README indexing
   what each probe asked. Three wrong cuts first — see `DECISIONS.md`; the one to remember is that
   *imported* is not the same as *live*, because `reference_family_figure` is imported by nothing
   and renders a figure the deck places.
9. **The `channel_position_maps` output decision** (§2.5's open item) — taken: pooled figures to
   `grant_figures/epoch` and registered, per-session diagnostics stay put.

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

**TWO REFINEMENTS FROM DOING IT (2026-09-21).**

- **Run the BEFORE twice.** `rest_coupling --from-csv` disagreed with itself between two runs of
  identical code, which made the first before/after diff meaningless and hid a real bug for an
  hour. A baseline you have not shown to be reproducible is not a baseline. `git worktree add
  <tmp> HEAD` gives you the old code to run beside the new one without stashing.
- **For a pure function, PIN it instead.** `tests/test_analysis_kit.py` checks in literal copies of
  the pre-extraction sources and asserts EXACT equality of the draws — milliseconds instead of
  ninety minutes, and stronger, because it covers inputs the real data does not contain. Use
  `==`, never a tolerance: the CI shift above is well inside any sane `rtol`. **Those frozen
  copies must never be tidied to match the extracted version.**

---

## 5. ANALYSIS ITEMS STILL OPEN (do not start these; they are context)

- **Restate the far-contra amplitude gradient under `restw`, not `rest`** — a numbers-in-text job
  against a figure that already exists (`epoch_15r_position_RESTWref_*`).
- **DLC tongue tracking** — the blocker for "did not try" vs "tried and missed", and the only
  thing that would turn the within-bout deceleration argument into a control.
- **530 nm reflectance** — DEFERRED hardware. Do not spend analysis effort on 470/415 coupling.
- **33 unregistered epoch figures** (recounted 2026-09-21: 402 files, 109 registered, and **0
  dead registry entries** — nothing registered has stopped rendering). The renderer emits nothing
  the registry has never heard of — all 39 of its `name=` templates are covered, and
  `tests/test_epoch_registry.py` now enforces that — so these 33 arrive by other paths:

  | group | n | what it is |
  |---|---|---|
  | `_erodedgate` / `_mf075` | 18 | sensitivity variants of the `15h` rotation and `15k` reference-family threads |
  | `_QC_*` / `_mask_*` | 8 | ad-hoc QC images dropped in this folder; **not deck material** |
  | `epoch_15r_position_RAWref_*` | 3 | a reference-family variant |
  | `epoch_1d*` / `epoch_1e_engagement*` | 3 | probably superseded by the `1b` family |
  | `epoch_23_quit_prodrome.png` | 1 | the UNGATED version; the gated `_gated_h90` one IS registered |
  | `epoch_24_first_last_quartile_gated_h90.png` | 1 | superseded by the quintile version |

  Only the first group needs captions written by someone who has read those analyses. The QC
  images should move out of the epoch directory rather than be registered.
