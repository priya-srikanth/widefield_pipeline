# Stored frozen decoders become the DEFAULT for the epoch acc family and rest decoder 12.1 (11-02)

Priya 11-02: *"we should cache frozen decoders as we always do"* (CLAUDE.md rule 10; `docs/FROZEN_MODELS.md` §7).
Worktree `C:/Users/SabatiniLab/Github/wf_frozendef`, branch `analysis/frozen-default`. It was created at
`analysis/rest-decoder-nl` (7c53214a), and `analysis/lp-final` (5822cbad) was merged in with `git merge` (merge commit
d966e80b, no conflicts; both descend from `analysis/restnl-maps`). Everything below the merge is uncommitted. No page
was edited, nothing was published, and nothing on M: was overwritten, deleted or renamed.

## 1. Code

| file | change | opt-out |
|---|---|---|
| `wfield_local/grant_confusion.py` | `_stored_epoch_decoder` (lp-final's opt-in) is now ON by default (`epoch_frozen_on()`). `_collect_5c` mode `frozen` gets its full and LOSO pre-stroke models from `frozen_models.load_or_fit` (kind `epoch_decoder`). | `WIDEFIELD_EPOCH_FROZEN=0` (also false / no / off): in-memory fit as before; logged per animal |
| `scripts/rest_migration/rest_frozen_decoder.py` | `main` takes its pre-stroke models (full model plus one LOSO model per pre session) from `frozen_rest_payload` -> `load_or_fit` (kind `rest_frozen`). For the page flags the spec `align` is `rest-rest-b4-dur10-gap10-gate`, which is `rest_frozen_decoder_nl`'s spec, so both scripts share one stored model per animal. Other flag sets get their own ids (`durall` / `gapall` / `nogate` / `-min<P>x<C>` / variant). The new sidecar `<stem>_frozen.csv` names the model id and status per animal. `_collect` and `_frame_samples` are untouched, so the nl cache key is unchanged. | `WIDEFIELD_REST_FROZEN=0` |
| `scripts/lp_final_render.py` | `compare` (the "current, in-memory" LP table) now sets `WIDEFIELD_EPOCH_FROZEN=0` explicitly | - |
| `tests/test_frozen_default.py` (new, 19 tests) | Pinned behaviour: (1) each default reaches `load_or_fit` and each opt-out does not; (2) the stored payload equals the in-memory fit bit for bit, for epoch full + LOSO and for rest full + LOSO under three matching settings; (3) the page spec equals the nl spec; (4) `_collect_5c` / `main` ask the store; (5) `_collect` / `_frame_samples` sources are the same as at 7c53214a. | |
| `tests/test_rest_frozen_decoder_nl.py` | The "only `_collect` changed" pin now also exempts `main` (documented). Every other function is still source-identical to 0e592a20. | |
| `scripts/frozen_default_compare.py` (new) | Exact cell-by-cell comparison of two renders' data CSVs (epoch `data/epoch_acc*.csv`; 12.1 sessions table plus per-animal / cohort retained) | |
| `docs/FROZEN_MODELS.md` §7 | Two rows added (`epoch_decoder`, `rest_frozen`), plus the opt-outs and what is still in memory | |

**Still fitted in memory (not in scope, listed so nobody assumes otherwise):**
- the frozen column of `_collect_5c` modes `refit` / `paired` / `paired_matched` (5r / frozen-vs-refit families);
- `grant_figures` 5b confusions (`grant_confusion.py` ~l.344).

Under the LP arm these would now disagree with the acc family on which pre-stroke model is "frozen" (paired = LP-trained
in memory, acc = DAQ-trained stored). **Decision for Priya.**

## 2. Store

Canonical local store (rule 10): `local_dir()` = `C:/Users/SabatiniLab/frozen_models`; it exists (checked; no
`paths.yaml` / config edit was made). No separate store was needed or used. The server store (`labcams/frozen_models`) is
read-only and untouched. Nothing was published.

- New: **28 `epoch_decoder` models** (12 MB). 4 animals x DAQ-trained {cue, lick, precue 2.0 s, precue 1.0 s}, plus
  LP-trained (`lick_source=lp`, WIDEFIELD_LP_FROZEN=lp) {cue, lick, precue 1.0 s}.
- The 12.1 run used the 4 `rest_frozen_rest-rest-b4-dur10-gap10-gate_*` models that `rest_frozen_decoder_nl --mask rest
  --pre all` froze earlier today (all 4 `frozen-hit`). No new rest model was written.
- Store total: 2067 models.

**Log note: `SPEC-CHANGED` is a false alarm here.** `siblings()` groups models by kind / align / source and ignores
`post_s` and `lick_source`. So:
- the first 1.0 s precue fit reported the 2.0 s model as a changed sibling (the message reads "11 sessions -> 11",
  nothing added or removed);
- the first LP-trained fit reported the DAQ-trained model the same way.

Each model was frozen under its own id and nothing was superseded. lp-final saw the same thing. A fix would add
`post_s` / `lick_source` to the sibling prefix in `frozen_models.siblings`; that is a core module, so it was not
changed here.

## 3. Verification: stored equals in-memory wherever the training set is the same

All renders are NEW folders under `G/frozen_default_20261102/` (G = `M:/MICROSCOPE/Priya/Widefield/labcams/grant_figures`).
Comparison tables are in `G/frozen_default_20261102/compare/<tag>_{summary,cells}.csv`. Cells are compared exactly.

| comparison | training set | result |
|---|---|---|
| DAQ 2 s, 5 arms: stored (first run, `frozen-new`) vs in-memory (`=0`) | same | **IDENTICAL**, 25 / 25 CSVs byte-identical |
| DAQ 2 s: stored first run vs re-run (`frozen-hit`, loaded from joblib) | same | **IDENTICAL** |
| DAQ ENL 1.0 s, 2 arms: stored vs in-memory; stored vs re-run (hit) | same | **IDENTICAL** / **IDENTICAL** |
| LP arm, 5 arms: stored LP-trained (`WIDEFIELD_LP_FROZEN=lp`) vs in-memory (= LP-trained) | same | **IDENTICAL** (25 CSVs) |
| 12.1: stored (4 x `frozen-hit` on the nl script's models) vs in-memory (`WIDEFIELD_REST_FROZEN=0`) | same | **IDENTICAL**: sessions, cohort and confusion CSVs byte-identical |
| **LP arm, new default (stored DAQ-trained) vs in-memory (LP-trained)** | **different** | **DIFFERENT, expected**: section 5 |

So the store changes nothing when it serves the same training set; within this box the fit is bitwise reproducible
and a joblib round-trip is exact. The only numeric change the new default makes is the LP arm's training set.

## 4. New renders vs the CURRENT page renders: drift that predates this change

These renders are older than today's code. The in-memory path reproduces every one of these differences, so none
of them is caused by the store.

| family (current render) | in-memory now vs current | what moves |
|---|---|---|
| DAQ 2 s, `G/epoch` (10-07; Figs 2.1, 2.3) | 937 cells differ, bar values at most 0.024 (cue_lick acute far contra .016); session / trial counts (meta) identical | far contra pooled: ENL .507/.250/.340/.411 -> .507/.251/.340/.410; cue .896/.321/.630/.739 -> .896/.322/.631/.740; lick .952/.500/.768/.751 -> .952/.500/.770/.750. One mark: ENL working, subacute far mid ** -> *. Per animal (session means, far contra) all within .021 (PS92 lick subacute .875 -> .896). |
| DAQ ENL 1.0 s, `G/DAQ_ENL1s` (10-22; Fig 2.2) | 210 cells, at most .009; no mark changes | far contra pooled .388/.231/.297/.319 -> .388/.231/.296/.319; per animal within .004 |
| LP arm, `G/LP_licks` (10-24; Fig 2.4) | 999 cells, at most .038 at a single session; no mark changes | far contra pooled within .002; per animal within .007 |
| 12.1, `G/epoch/data/epoch_15f_rest_frozen_restdock05_durmatched_gapmatched_*` (10-07) | 162 of 3840 session cells (23 sessions), at most .035; period counts identical | cohort retained acute .279 -> .281, subacute .359 -> .360, chronic .537 -> .537 (**page 0.28 / 0.54 unchanged**). Per animal: PS92 acute .605 -> .612, PS93 subacute .453 -> .457; every other cell within .001. |

12.1: the differing columns include the within-session refit and the nulls, which never touch the store. The imaging
inputs of the affected sessions (SVTcorr, U_atlas, rest mask) all predate 10-07. So the source is behavioural-input or
code change after the render (block ids / features of 23 sessions). It was not traced further. The DAQ epoch drift is
of the same kind and also appears in the in-memory path. To reproduce the page exactly, re-render; the frozen default
has nothing to do with it.

## 5. The one real change: the LP arm now LOADS the DAQ-trained decoder

Before 11-02 the LP arm's in-memory fit trained on LP-arm pre trials. Under the default it loads the stored DAQ-trained
model for the same window and pre set (20 / 20 `frozen-hit`). Fig 2.4's caption already says "the frozen decoder is
the stored DAQ-contact model", which is true only from now on. Where the training set matches (LP-trained stored), the
result is identical (section 3).

Far contra, LP arm (`G/LP_licks` settings: LP licks, DAQ gate, June + August pre, ENL 1.0 s). Values are in-memory
LP-trained -> stored DAQ-trained. Per animal = mean over sessions (from `_sessions.csv`); pooled = the bar (trial-pooled,
mark).

| arm | animal / pooled | pre | acute | subacute | chronic |
|---|---|---|---|---|---|
| **lick / lick (Fig 2.4)** | PS92 | .922 -> .913 | **.268 -> .145** | .549 -> .498 | .791 -> .720 |
| | PS93 | .853 -> .858 | .158 -> .169 | .460 -> .455 | **.747 -> .658** |
| | PS94 | .898 -> .890 | .466 -> .436 | .639 -> .669 | .550 -> .541 |
| | PS95 | .940 -> .926 | .275 -> .225 | .814 -> .804 | .908 -> .889 |
| | pooled | .903 -> .897 | **.314** -> **.260** | .662** -> .662** | **.758** -> **.710** |
| ENL working (precue) | PS92 | .291 -> .248 | .151 -> .115 | .132 -> .125 | .264 -> .218 |
| | PS93 | .327 -> .331 | .161 -> .168 | .230 -> .259 | .349 -> .382 |
| | PS94 | .543 -> .546 | .296 -> .329 | .316 -> .373 | .289 -> .337 |
| | PS95 | .402 -> .395 | .150 -> .150 | .278 -> .288 | .346 -> .338 |
| | pooled | .400 -> .392 | .206** -> .208** | .279** -> .306* | .316 -> .327 |
| cue working | PS92 | .876 -> .879 | .256 -> .256 | .587 -> .587 | .730 -> .757 |
| | PS93 | .806 -> .795 | .135 -> .133 | .431 -> .409 | .685 -> .684 |
| | PS94 | .895 -> .900 | .376 -> .333 | .661 -> .647 | .602 -> .589 |
| | PS95 | .943 -> .937 | .283 -> .333 | .761 -> .727 | .885 -> .889 |
| | pooled | .883 -> .880 | .283** -> .272** | .652** -> .627** | .733** -> .736** |

ENL lick and cue lick track their working twins (`compare/lp_farcontra_per_animal_session_mean.csv`). Every
animal x epoch x arm, plus pooled with marks, is in `compare/lp_farcontra_*`.

Reading:
- Fig 2.4's acute far-contra drop gets deeper under the DAQ-trained decoder: pooled .31 -> .26; PS92 .27 -> .15.
  Acute < pre stays in 4 / 4 animals, and pre is unchanged within .015.
- The caption's "~0.3" becomes ~0.26.
- Lick-window chronic far contra is lower: .758 -> .710 pooled, mostly PS92 / PS93.

Significance marks that change (DAQ- vs LP-trained, all positions, LP arm):
- lick / lick: acute far ipsi ** -> *; subacute far ipsi ** -> none; subacute near contra * -> **; subacute and
  chronic far mid none -> *; chronic far ipsi * -> none.
- ENL working / lick: acute far ipsi ** -> *; subacute far contra ** -> *; chronic far ipsi * -> none; subacute far
  mid none -> *; ENL working acute near mid * -> none.
- cue working: chronic far ipsi * -> none. cue lick: acute near ipsi * -> **.

At n = 4 these marks are fragile, and the far-contra acute ** holds in every arm. This matches lp-final's finding that
DAQ- vs LP-training changes the ENL cells by <= .05; the lick window moves more (PS92 acute .12).

## 6. Status / runs (11-02, EST; shared box, <= 4 BLAS threads in total, free RAM >= 35 GB, only own PIDs)

| run | wall |
|---|---|
| A: DAQ 2 s, 5 arms, stored (fits 12 models) | 23.4 min |
| B: DAQ 1.0 s, 2 arms, stored (fits 4) | 9.2 min |
| C: LP, 5 arms, stored DAQ-trained (20 hits) | 20.4 min |
| A2 / B2: DAQ re-runs (all hits) | 16.3 / 6.3 min |
| D: LP in-memory (`=0`) | 13.1 min |
| E: LP stored LP-trained (fits 12) | 18.4 min |
| F / G: DAQ 2 s / 1.0 s in-memory (`=0`) | 12.5 / 5.5 min |
| R1 / R2: 12.1 stored / in-memory (`--perm 200 --refit --match-train --match-duration --match-lickgap`, serial collect) | 21.9 / 18.9 min |
| tests: new + nl (23) / related suites (332) | 2.7 / 5.8 min |

Tests: 315 of 332 pass. The 17 failures are all `test_parallel_results_are_ordered` on the committed
`scripts/cohort2_qc/*` scripts. Those scripts are untouched by this branch, so the failures pre-date it. ruff is clean
on every changed file (the pyproject B023 per-file ignore for grant_confusion still applies).

The wf_restdec RESTnl fit (PID 34152) and its outputs were not touched.

## 7. Open / for Priya

1. The LP arm's acc family is now DAQ-trained by default (section 5). Fig 2.4 should be re-rendered and its numbers
   updated (acute far contra ~.26), or the old render kept with "LP-trained in memory" stated. Decide which.
2. The paired / refit / 5b frozen columns are still in memory (section 1); in the LP arm they are LP-trained.
3. `siblings()` false SPEC-CHANGED across `post_s` / `lick_source` (section 2).
4. The pre-existing drift of the 10-07 / 10-22 / 10-24 renders (section 4) is small, and no headline changes. A
   re-render under the new default would replace them.
5. Commit: Priya.
