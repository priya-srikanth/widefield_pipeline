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

1. **Extract per-session collectors for 8g, 10, 10b, 11.** They build their per-day data INLINE
   rather than through a `_collect_*`. `epoch_figures` must not re-derive it — that would put a
   second definition of the same population in the codebase, which is how the frozen-decoder
   contamination survived eight days. `_collect_5c`, `_collect_7` and `_pooled_bundle` were all
   extracted for exactly this reason; these four are the same job.
2. **Write the renderers.** Each is short given the scaffolding.
3. **Drive every one through `_overlaps` at 6.2in.** This is the step that must not be skipped:
   larger fonts on a smaller canvas is precisely where labels collide, and three layout faults this
   month were found only by driving the function rather than reading the diff.
4. **8/8b without in-box numbers** — a `annotate=False` flag on the crossnobis renderers rather
   than a forked copy.
5. **Deck placement** — a new section, quarter-page figures placed 4-up.

## Open

- **Chronic epoch** — one entry per animal in `EPOCH_SPEC` plus its name in `EPOCHS`; nothing else
  needs to know.
- **`grant_figures._day` vs `epochs.days_since_stroke`** agree on every post-stroke session (all
  August, a 31-day month) and differ by one on June dates (month*31 accumulates a day per short
  month). Harmless — every pre-stroke day is negative and both say `pre` — but pinned, so a
  September session divides them loudly instead of sliding an epoch boundary under a published
  figure.
