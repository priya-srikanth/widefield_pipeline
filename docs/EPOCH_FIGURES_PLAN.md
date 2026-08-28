# Pooled cross-animal EPOCH figures — scope, reuse map, and what is left

Priya, 2026-08-28. Pooled across all four animals, three panels (pre / acute / subacute), sized to be
read at **a quarter page or smaller**. Constraint she set: *"I do NOT want this to have to re-create
the wheel."* Audited — nothing needs recomputing.

## Done

| | |
|---|---|
| `43bede0` | `wfield_local/epochs.py` — epoch assignment, 11 tests |
| `ace8c18` | `wfield_local/epoch_figures.py` — per-epoch pooling, coverage, behaviour weighting |
| `f248ab9` | per-session dot overlay (4 colours, transparency, deterministic spread) |

**Epochs are DAYS SINCE EACH ANIMAL'S OWN LESION, not session index.** Read as session index the
specification is impossible: PS94 has 8 post-stroke sessions and no ninth, so "subacute 9+" would be
empty and PS94 would silently contribute nothing to every subacute panel. Lesion dates differ
(PS94/PS95 0816, PS92/PS93 0817).

**Session counts every panel must state** — acute PS92 5 / PS93 4 / PS94 6 / **PS95 1**; subacute
**PS92 2** / PS93 3 / PS94 2 / PS95 7. Behaviour is weighted BY SESSION (Priya's call), so PS94
dominates acute and PS95 subacute. The dot overlay makes that visible rather than asserted.

## Reuse map — every population already exists

| source | gives | serves |
|---|---|---|
| `_collect_5c(align, variant)` | `{animal: (pre LOSO record, {day: record})}`, trial level | 4, 5c, 5d, per-position accuracy |
| `_collect_7(align, variant, min_trials)` | per-day post trials as trials, pre sessions separate | 6, 6b, 6d, 7, 7b, 7d, 9 |
| `_pooled_bundle(animal, align)` | memoised joint-basis load | 6, 6b, 7, 8 |
| `_position_metrics(animal, mmdd)` | per-position `(hit, lo, hi, n)` | 1b |
| `locanmf_frozen_decoder_loso_roi_{align}.json` | `confusion[label]`, per-session raw counts | 4, 5c, 5d |
| `section_g.json` | per session, `arms.all` / `arms.lickonly`, pattern similarity | 6 family |

`variant="working"` **is** "lick + miss while working" — already implemented, already 5c's default.
The lick window admits only `variant="lick"` and raises rather than returning the wrong population,
since a no-lick trial has no lick to align to.

## The figures requested

Windows: **ENL, cue** (both `variant="working"`), **lick** (`variant="lick"` by necessity).

| # | function | stem | per-session? | note |
|---|---|---|---|---|
| 1b | `fig_behaviour` | `grant_1_behaviour_by_position` | via `_position_metrics` | session-weighted |
| — | per-position decoding accuracy | from 5c diagonals | yes | |
| 4 | `fig_confusion_prestroke` | `grant_4_confusion_prestroke_*` | yes | LOSO matrix |
| 5c / 5d | `fig_confusion_per_session` / `fig_confusion_delta` | `grant_5c_*`, `grant_5d_*` | yes | frozen + delta |
| 6 / 6b / 6d | `fig_pattern_similarity{,_per_session}`, `fig_pattern_delta` | `grant_6*` | yes | |
| 7 / 7b / 7d | `fig_splithalf_matrix`, `fig_reliability_verdict`, `fig_splithalf_delta` | `grant_7*` | yes | |
| 8 / 8b | `fig_crossnobis_cross`, `fig_crossnobis_geometry` | `grant_8*` | yes | **no numbers in boxes** |
| 8g | `fig_geometry_by_position` | `grant_8g_geometry_by_position_{align}_{v}` | yes | builds per-session inline |
| 9 | `fig_delta_trajectory` | `grant_9_delta_trajectory_{align}_{v}` | yes, via `_collect_7` | time axis -> 3 epoch bins |
| 10 | `fig_best_match` | `grant_10_best_match_{align}_{v}` | yes | builds per-session inline |
| 10b | `fig_best_match_by_session` | `grant_10b_best_match_by_session_{align}_{v}` | yes | builds per-session inline |
| 11 | `fig_encoder_gain_shape` | `grant_11_encoder_gain_shape_{align}_{v}` | yes | builds per-session inline |

## What is actually left

1. ~~**Extract per-session collectors for 8g, 10, 10b, 11.**~~ **NOT NEEDED — this item was wrong.**
   Audited 2026-08-28: all four already route through collectors, which are simply not named
   `_collect_*`, so a grep for that prefix missed them. Every one bottoms out in `_collect_7` and
   is keyed by `"PRE"` or day, which is exactly what an epoch is defined on:

   | figure | collector | returns |
   |---|---|---|
   | 8b, 8g | `_rdm_rows`, `_rdm_ci` | `{animal: {"PRE"\|day: (row r, whole-RDM r, n)}}` |
   | 10 | `_match_tables` → `_matrices_pattern` | `{animal: (pre 6x6, post 6x6, {day: (acc, rank)})}` |
   | 10b | `_matrices_pattern` | `{animal: {"PRE": M, day: M}}` |
   | 11 | `_enc_tables`, `_enc_ci` | `{animal: {"PRE"\|day: (raw, a, gain, per-position)}}` |
   | 7, 7d | `_matrices_splithalf` | `{animal: {"PRE": M, day: M}}` |
   | 8, 8d | `_matrices_crossnobis` | `{animal: {"PRE": D, day: D}}` |

   Note what these return: **reduced matrices, not trials.** So pooling within an epoch is a mean
   over sessions, not a re-pooling of trials — which is exactly the session weighting Priya asked
   for. Only the 4/5c/5d family returns trial-level records, and those stay summable.

2. **Write the renderers.** Each is short given the scaffolding. `confusion_row` is done and driven.
3. **Drive every one through `_overlaps` at 6.2in.** Not optional — see below.
4. ~~**8/8b without in-box numbers**~~ — `annotate=False` is already the default on `confusion_row`.
5. **Deck placement** — a new section, quarter-page figures placed 4-up.

## What driving `confusion_row` actually found (2026-08-28)

Three faults, and **none of them was a collision** — `_overlaps` reported the figure clean at every
step until the last one.

* **The canvas was sized per panel**: 6.2in for three, 8.27in for four. The deck places both in the
  same quarter-page column, so the four-panel delta row rendered its ticks at **6.0pt** beside the
  three-panel row's 8.0pt. A 25% type difference between two figures side by side, invisible to any
  overlap check. Fixed by making the canvas ALWAYS `QUARTER_IN`: a point size written in the module
  is now the point size the reader gets, and a fourth panel costs panel width, which is the honest
  price of a quarter page.
* **The axes were numbered 0–5, not named.** `labels` was a local initialised to `None` and
  immediately overwritten with digits, with no way for a caller to pass the position names.
* **The comment said "ABSOLUTE MARGINS" over `subplots_adjust` fractions** — `left=0.085` is 0.53in
  on one canvas and 0.70in on another. Now inch targets divided by the canvas.

One thing I got wrong on the way and the measurement caught: laying the short labels flat instead of
rotating them looks like a saving and is not. Rotated, a label's horizontal extent is the type
HEIGHT (~0.11in at 8pt); `cC` flat is ~0.13in. Flat labels crowded a row that rotation had already
passed clean. Rotation stays, with the numbers written next to it.

Final state, measured: 3-panel and 4-panel variants both 6.20in wide, ticks at 8.0pt, titles at
8.5pt, **0 overlaps**, pinned by `test_nothing_collides_at_a_quarter_page` and
`test_the_canvas_is_the_placed_size_whatever_the_panel_count`.

## Open

- **Chronic epoch** — one entry per animal in `EPOCH_SPEC` plus its name in `EPOCHS`; nothing else
  needs to know.
- **`grant_figures._day` vs `epochs.days_since_stroke`** agree on every post-stroke session (all
  August, a 31-day month) and differ by one on June dates (month*31 accumulates a day per short
  month). Harmless — every pre-stroke day is negative and both say `pre` — but pinned, so a
  September session divides them loudly instead of sliding an epoch boundary under a published
  figure.
