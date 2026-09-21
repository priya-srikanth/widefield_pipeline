# Handoff, 2026-08-28 — five approved items, with the plumbing already traced

Priya approved all five. **ALL FIVE ARE DONE** -- `837a8e3`, `73b651d`, `b93fbc0`, `c63ff54`,
`f3601d6`. What follows is kept as the record of what each was and what the plan got wrong.

Two things this document got WRONG, corrected in place below so nobody re-derives them:

* **#1 said to bump `CACHE_VERSION` 11 -> 12. Do not.** `with_rt=False` returns a tuple byte-identical
  to what `_trial_features` returned before the flag existed, so every warm entry on disk is still
  correct. The key omits the flag when false and carries it when true, which is unambiguous (two specs
  differing by the presence of a NAMED key cannot collide) and preserves ~1 MB per session-alignment
  at ~168 s each to rebuild. A bump would additionally have discarded every UNRELATED cached kind --
  RSA, spatial reorganisation, the engagement tables -- for a change none of them can see.
  `CACHE_VERSION` is for when the COMPUTE CODE moves; this was a new key, not a moved one.
* **#2's blocker was already clear.** `_pooled` built `phase_labels("pre") + phase_labels("post")`
  filtered by animal, which is what `config.pooled_labels` returns. It now CALLS that function rather
  than agreeing with it, which is the point: agreement by coincidence is the failure mode.

---

## 1. Early vs late-rewarded confusion classes -- DONE (`837a8e3`)

Landed as `position_coding_directions.figure_rt_split` -> `coding_rtsplit_<window>_<animal>.png`,
deck section **G9e**, one animal per slide. D2's respwin arm is unplaced (figures still written, so
it is reversible). `rt_drift` was relabelled **G9d** -- it and the per-session matrices both called
themselves G9c, so a spoken reference picked out two different slides.

Two design points worth keeping:

* `poststroke_lick_early` / `_late` are in a new **`CONFUSION_SUBCLASSES`** mapping, deliberately NOT
  in `CONFUSION_CLASSES`. That tuple's whole invariant is that summing a subset gives a population;
  a fourth and fifth sibling would have made "all trials" count every lick trial twice, silently,
  and the result would still have looked like a confusion matrix.
* The figure uses **absolute `subplots_adjust` margins, not `tight_layout`** -- `imshow` fixes an
  aspect, so tight_layout warns once per panel and fills the nightly log, and a negotiated layout is
  not reproducible (`_delta_grid` was clean at 10 post-stroke days and overlapping at 12 for exactly
  that reason). Verified by driving the function through `_overlaps`: 0, including the empty-late-arm
  case.

The original plan follows, since the plumbing description is still the record of how RT gets out.

### (as planned)

**Why.** `decode.max_rt_s = 3.5 s` is the task's response window, so "engaged" everywhere outside
`nolick_decoder` already merges a 0.2 s lick with a 3.0 s lick. Post-stroke the mass shifts into the
late bin, and that is precisely the distinction the study is about: position coding preserved on LATE
trials is *plan intact, execution slow*; degraded on late trials is a different result. Today no
figure can tell them apart.

**Why it is nearly free.** `position_coding_directions._class_confusions` already stores RAW COUNTS
scored by ONE frozen model, so any population is a sum. `early + late` must reconstruct today's
`poststroke_lick` exactly — that is the acceptance test, and it means nothing already published moves.

**THE BLOCKER, traced.** The split needs per-trial reaction time, and RT is computed inside
`_trial_features` (`locanmf_position_decoder.py:405`, `rt = first - cue_f`) but **never returned**.
`with_indices` returns trial indices only. Do NOT reconstruct RT from those indices in the caller —
the module's own comment records that rebuilding a trial filter elsewhere is how bugs 15, 16 and 17
happened, and one such mask came out 633 long against 575 kept trials.

**The change, in order:**

1. `_trial_features(..., with_rt=False)` → append `rt[idx_eng]` to the returned tuple, beside the
   existing `with_indices` extras.
2. `feature_cache_kind` — add `with_rt` to the spec. It changes the RETURN SHAPE, so omitting it
   would serve a 6-tuple to a caller expecting 8. Bump `CACHE_VERSION` 11 → 12.
3. `precue_engagement_states.features_with_indices` — pass it through and hang the RTs off the
   closure the way `.indices` and `.variance_captured` already are.
4. `_class_confusions` — split `poststroke_lick` at **2.0 s**, the same boundary
   `nolick_decoder._args` uses, so "late" means one thing in both places. NOT the median RT:
   that is session-relative and cannot be compared across days or against `late_rewarded`.
5. `CONFUSION_CLASSES` gains `poststroke_lick_early` / `poststroke_lick_late`. Keep
   `poststroke_lick` as their sum so existing consumers (`grant_figures` 5b) keep working unchanged.
6. Deck: a G-section slide showing early vs late side by side, and **then** delete the four D2
   `respwin` slides — they set the cut TO the response window, which empties the late arm and leaves
   "engaged = early + late", i.e. exactly what every other figure already shows.

**Acceptance:** `early + late == poststroke_lick`, element-wise, on every animal and window.

---

## 2. `poststroke_compare` -- stop refitting what is already stored -- DONE (`73b651d`)

`locanmf_frozen_decoder.frozen_decoder_models()` now owns the spec and the fit; both modules call it,
so the models are identical BY CONSTRUCTION rather than by six recipes agreeing.
`poststroke_compare.frozen(d)` loads once per pool and memoises on the pool dict.

THE TEST THAT MATTERS IS EQUIVALENCE, not that the call happens:
`tests/test_poststroke_compare_uses_the_stored_model.py` asserts the frozen path predicts IDENTICALLY
to the old inline fit and that `models["loso"]` reproduces `cross_val_predict(LeaveOneGroupOut)` fold
for fold. Keyed by LABEL, not pooled index -- the index depends on the order the caller assembled its
pool and the label does not.

The two sites that are NOT the same model still fit locally and are pinned: `decode_matched`'s
lick-only arm (class-filtered, 4-way for PS94/PS95, so a different chance level) and
`_within_accuracy`. A helper that quietly served `full` to those would be worse than the duplication
it removed.

### (as planned)

Priya: *"let's not refit independently if we are replicating the exact same thing."*

**Genuinely identical to `frozen_models` `models["full"]` / `models["loso"]`** — same `_pipe()`, same
`pool_sessions` conventions, all pre-stroke engaged trials, no class filter:

| site | what it re-derives |
|---|---|
| `poststroke_compare.py:448` `crossed_confusion` | `models["full"]` |
| `:452` | `models["loso"]` |
| `:472` | `models["loso"][kept[gsess]]` literally |
| `:734` `impaired_nolick_readout` | `models["full"]` |
| `:147/:149` `decode_matched`, **all-trials arm only** | `full` / `loso` |

The redundancy is large: `poststroke_section_g` mutates only `dd["post_i"] = {session}` (`:92`,
`:105`, `:122`), so `crossed_confusion` refits the identical pre-stroke model once per post-session ×
arm × alignment.

**NOT replaceable, do not touch:** `decode_matched`'s lick-only arm (class-filtered to preserved
positions — 4-way for PS94/PS95, so a different chance level), and `_within_accuracy:799` (a
within-session ceiling).

**THE BLOCKER:** the pools differ. `poststroke_compare._pooled` uses `phase_labels("pre")+("post")`
while `pooled_frozen_loso(source="roi")` used the curated all-phases list. `config.pooled_labels`
(commit `641118f`) fixed the nightly side; verify `_pooled` now agrees, because for ROI `_align_many`
intersects region×bin columns across the pool, so a different pool gives a different `n_features` →
a different `spec_id` → a permanent cache miss rather than a hit.

---

## 3. Modules still on the 2.0 s engaged cut -- DONE (`b93fbc0`)

**THERE WERE FOUR, NOT THREE.** `postcue_window_test.py:40` was missed by the by-hand survey below
and found on the first run of `tests/test_engaged_cut_comes_from_config.py`, which walks the AST
instead of grepping. It is also where the literal did the most damage: that module sweeps the
post-cue WINDOW over [2.0, 2.5, 3.0, 3.5] s, so the longer windows were scored on trials selected by
the shortest one -- the window under test and the trials it was tested on disagreed by up to 1.5 s.

All four now read `decode.max_rt_s` AND announce the cut at run time, because fixing only the code
would have moved the disagreement into the docs: every number recorded for these modules was measured
at 2.0 s. `nolick_decoder` / `nolick_analysis` are exempt and the guard checks the exemption still
carries its reason.

### (as planned)

`decoder_c_sweep.py:57`, `encoder_bins_test.py:123`, `filter_acausality_test.py:205` hardcode
`max_rt=2.0`; `decode.max_rt_s` moved to 3.5 on 2026-08-21. They are internally-consistent PARAMETER
SWEEPS, so they are not wrong — but they would disagree with the headline if ever quoted against it.

Priya: fix where needed, *"or at least ensure the slide label and notes are clear about the
difference"*. Preferred: read the config like everything else. If a sweep genuinely needs the old
boundary for comparability with its own earlier runs, say so in the docstring the way
`nolick_decoder._args` does — that docstring is the model to copy, because it explains why 2.0 s is
correct THERE rather than merely recording that it is used.

---

## 4. Three copies of the lick-vs-no-lick discriminator -- DONE (`c63ff54`)

Collapsed to `balanced_lick_sample` + `lick_pipe`. **The coincidence was already broken**, which is
the argument for having done it: `looks_like_which` and `undetected_state_split` each start a fresh
`RandomState(seed)` and draw the same sample, but `fits_engaged_distribution.balanced_fit` shares one
generator across its leave-one-out loop, so its full-pool fit is a DIFFERENT sample from a function
that reads as though it were the same. The rng is passed in rather than created, so every call site
keeps its exact draw and the refactor moved no number -- asserted draw for draw against the inline
loop as it was written.

Not frozen via `frozen_models` after all: the sample depends on generator state, so a stored artifact
would be keyed on something the spec cannot see. `lick_pipe`'s docstring carries the
`kind="lick_discriminator"` warning instead.

### (as planned)

`poststroke_compare.py:207` (`looks_like_which`), `:309` (`fits_engaged_distribution.balanced_fit`),
`:673` (`undetected_state_split`) each build the same model: a bare
`make_pipeline(StandardScaler(), LogisticRegression(3000, C=0.5))` on pre-stroke ENGAGED vs pre-stroke
NO-LICK, position-balanced, labels `{0,1}`.

This is **not** the position decoder — different label space entirely — so it must not load
`kind="decoder"`. Give it `kind="lick_discriminator"` in `frozen_models`, which already keys on
animal/align/source/basis/train_labels. Three copies agreeing by coincidence is exactly what the
store exists to replace with an identity.

---

## 5. Figures written nightly and never shown -- DONE (`f3601d6`)

Fourteen placed, measured by instrumenting `add_picture` on a real build. 496 -> 513 slides.

**The list below was partly wrong.** `coding_cosslope_*` and `coding_pairsplit_*` are NOT unplaced --
G9b shows the orthogonalised variant and omits the plain one deliberately, because both cohort
diagnostics were measured on the orthogonalised directions. And the real gap in section G was one the
list missed: `section_g_grid_withcontrol_lickonly` (the all-trials arm was placed, its sibling was
not) plus the whole nine-figure `section_g_smalllesion_*` family.

### (as planned)

Include the informative ones: `section_g_grid_{all,lickonly,withcontrol_lickonly}`,
`section_g_smalllesion_*` (5), `joint_basis_health_{cue,lick}`, `locanmf_rsa_hemisphere_{rdms,summary}`
at the CURRENT tag, `coding_cosslope_*` and `coding_pairsplit_*` (6).

`joint_basis_health_cue` is the one with an argument attached: the deck already shows the PRE-CUE
variant and says the post-cue figure "is a different measurement and is not shown here" — so adding it
needs that note updated, not just the slide.

---

## Method notes that cost time to learn

- **Never trust a layout-fault count from a log.** `_save` prints `bad[:6]`, so every logged count is
  a floor. The other window's 5c "6" was really 40; my 204 on 2026-08-26 was a floor too. Use
  `_overlaps(fig)` directly.
- **Verify a layout fix by DRIVING the function**, not by re-reading the diff. On 2026-08-28 a comment
  asserting `hspace 0.60` shipped over a `gridspec_kw` that never got it, and a `fig_asymmetry` fix
  landed on `fig_delta_trajectory` because both call `plt.subplots(len(ANIMALS), 2, ...)`.
- **Deck figure coverage can only be measured by instrumenting `add_picture` on a real build.**
  Grepping stems reports everything present; converting f-strings to regexes reports everything
  covered (`f"{stem}.png"` is a wildcard). Both were tried and both were wrong.
- **A figure's legibility is `fontsize × (placed width / figure width)`.** Making a figure SMALLER in
  inches makes its type LARGER on the slide. That is why `coding_cross_` went 22.7in → 10.8in.
