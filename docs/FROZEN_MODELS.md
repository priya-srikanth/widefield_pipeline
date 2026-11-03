# FROZEN MODELS — where they live, how they are identified, and the two ways that has gone wrong

**Read this before touching `wfield_local/frozen_models.py`, before adding a machine profile, and
before concluding that two runs used "the same model".**

Companion reading: `DECISIONS.md` → *"THE FROZEN MODEL STORE MOVED ON 2026-09-18 AND NOTHING SAID
SO"*, and the module docstrings in `frozen_models.py` / `publish_basis.py`.

---

## 1. WHY THEY EXIST

Priya, 2026-09-21: *"the point was to avoid recomputing each night and also to ensure they were the
same models used across days and analyses"*. Both halves matter, and the second is the one that
cannot be recovered after the fact:

- **A frozen model is a REFERENCE, not a cache.** A decoder keyed on a pre-stroke training set
  cannot be regenerated once that set has grown. Every post-stroke number is scored against it, so
  losing it loses the frame those numbers were computed in.
- The cache saving is real but secondary. `--loso` refits ~48 models otherwise.

---

## 2. WHERE THEY LIVE

`find()` looks **local first, then the server**, because the local one is not read over SMB.

| | resolved from | note |
|---|---|---|
| local | `figures_working` root's parent + `frozen_models` | per machine, **derived** |
| override | `WIDEFIELD_FROZEN_MODEL_DIR` | absolute; wins over the derivation |
| server | `labcams/frozen_models` on MICROSCOPE | **read-only** from analysis code |

Publishing is a separate, byte-verified act: `python -m wfield_local.publish_basis --what frozen`
(add `--dry-run` first). **It COPIES; it does not move.** Verification is by SHA-256 read back from
the destination, and it refuses to overwrite a file whose digest differs under an unchanged
directory name.

### The current state (2026-09-21)

| box (profile) | local store | contents |
|---|---|---|
| behavior box (`analysis`) | `C:/Users/sabatini/source/frozen_models` | **46** — the canonical set; the server copy was made from it |
| analysis desktop (`analysis_desktop`) | `C:/Users/SabatiniLab/frozen_models` | **absent** — this box reads the server |
| …its 2026-08-28 fits, orphaned | `C:/wf_local/frozen_models` | **48** — keep, see §4 |
| MICROSCOPE | `labcams/frozen_models` | **48**, 12 per animal — complete |

Every box now resolves every spec. **Do not publish from a second box** — see §5.

---

## 3. HOW A MODEL IS IDENTIFIED

`spec_id = sha1(json.dumps(spec, sort_keys=True))` over `make_spec`'s dict: animal, kind, align,
source, basis_id, post_s, zscore, alpha, n_features, the **sorted** training labels, and a
`_session_sig` per label (stat signature of that session's `U_atlas` and `SVTcorr`).

Verified 2026-09-21: the same `spec_id` under `PYTHONHASHSEED` 0 / 1 / 12345, and identical whether
the training labels arrive as a list, that list reversed, or a set. `sort_keys=True` and the
`sorted(train_labels)` in `make_spec` are what make that true — **drop either and a pooled fit
whose labels arrived in a different order would miss the store, refit, and publish a second
reference for the same data.** `tests/test_parallel_results_are_ordered.py` pins both.

The signature matters because labels alone are not enough: on 2026-08-14 the switch to the
meegkit_hpfit `SVTcorr` changed the underlying data while every label stayed identical.

---

## 4. FAILURE ONE — THE STORE MOVED, SILENTLY (2026-09-18 → 2026-09-21)

`local_dir()` is derived from `figures_working`. Commit **889c5e0**, *"Give the analysis desktop
its own machine profile, instead of a masquerade"*, set that root for the desktop for the first
time. Before it, `joint_locanmf._basis_dir` fell through to its `C:/wf_local` literal, and the 48
models fitted on 2026-08-28 went there. After it, `local_dir()` pointed at a directory that did not
exist.

`find()` returned None. `load_or_fit` did exactly what it is designed to do on a miss: refit and
store. **Nothing in the output changed for two weeks.**

The recompute was the visible cost. The one that mattered:

> **With the store invisible, `siblings()` had nothing to compare against, so `SPEC-CHANGED` could
> not fire.** That status is the only thing separating "the pre-stroke training set grew" from "the
> reference silently moved". A changed training set would have been reported as a first run.

Nothing had in fact changed — PS92's 11 training labels and all 11 input signatures were identical
to August — but that is luck, not the safety net.

**The fix and the rule.** `frozen_models.warn_if_store_moved()` now reports, once per process, when
the local root is **absent**, and names the legacy directory and the publish command if models are
sitting there. `tests/test_frozen_store_move_is_loud.py` pins it, including that `load_or_fit`
actually calls it.

> **An EMPTY store and an ABSENT store are different facts and must not produce the same silence.**
> Empty = "nothing frozen yet", the normal first run. Absent = "you are not looking where the
> models are". Deriving the path from config is still right — a drive-letter literal is only
> correct on the machine it was written on — but **a derived path can MOVE**, and the consumer has
> to be able to notice.

### The PS92 "gap" that is not a gap

The behavior box has 46, not 48: it is missing `decoder_cue_joint76d884` and
`encoder_cue_joint76d884` for PS92 — **exactly the two models that were already on MICROSCOPE.**
`load_or_fit` consults local then server, and on a hit it returns `frozen-hit` *without writing a
local copy*. Those two were served from the share and correctly not refitted. 46 + 2 = 48.

---

## 5. FAILURE TWO — `spec_id` DOES NOT PROMISE THE SAME FITTED BYTES

The two boxes independently fitted the same 46 specs. **46 of 48 `model.joblib` files differ, and
not only in pickle metadata — the coefficients differ.** Worst pair,
`PS95/decoder_precue_roi_d57cc33c6485`:

| | |
|---|---|
| cosine between coefficient matrices | 0.9999723784 |
| max abs difference | 1.42e-02 = **0.54% of the largest coefficient** |
| `n_iter_` | **589 here, 620 there** |

`lbfgs` stops at `tol=1e-4`; a different BLAS build reaches that tolerance along a different path.

- **Within a machine the fit IS bitwise reproducible**, including across BLAS thread counts
  (pinned in `tests/test_parallel_results_are_ordered.py`).
- **Across machines it is reproducible only to the optimiser's tolerance.**

### Does it change any result? YES — SLIGHTLY. Two proxy tests said no and both were wrong.

Run `pooled_frozen_loso` for PS95 (roi, cue) against each store in turn. Same `spec_id`
(`5e19afa87323`), `frozen-hit` from both, so this is one spec served from two places:

| quantity | sessions differing | max | mean | level |
|---|---|---|---|---|
| `per_session` (the FROZEN model's score) | **7 of 25** | **0.002976** | 0.00056 | 0.9036 |
| `within_session` (the same-day refit ceiling) | 0 of 25 | 0.000000 | — | 0.8408 |

Up to **0.3 percentage points** on a per-session frozen accuracy — roughly one or two trials in
six hundred. The within-day ceiling is untouched, as it must be: it is refitted each run and never
comes from the store.

**Against the effects this study reports** (near-spout hit rate falling 0.972 → 0.457 across
session quintiles) that is immaterial. **Against a quoted three-decimal accuracy, or a marginal
comparison, it is not** — those are not reproducible across stores, and numbers computed on the
desktop between 28 Aug and 18 Sept will not re-derive exactly now that it reads the server.

### The two tests that got this wrong, and why

| test | said | why it was wrong |
|---|---|---|
| 20,000 random Gaussian probes | 99.275% agreement | probes land nowhere near where ROI activity lives; **overstates** the disagreement |
| raw per-session feature matrices, 3 pairs, 5,417 trials | 100.000% agreement | accuracy came out ~0.17 on six classes, i.e. **chance** — those columns are not the aligned pooled feature space `_aligned` builds, so it exercised a function the pipeline never runs. **Understates** it |
| `pooled_frozen_loso` end to end | 7/25 sessions differ | the deployed quantity, at its real ~0.90 accuracy |

**A MODEL COMPARISON IS ONLY AS GOOD AS THE INPUT DISTRIBUTION IT IS RUN ON — and "the accuracy
came out near chance" was the tell that should have stopped the second test being reported at
all.** It was flagged as a caveat and then under-weighted, and the conclusion drawn from it ("no
result changes") was stated before the end-to-end check came back. Run the real entry point.

### Consequences

1. **Publish from ONE box.** A second box's upload would report `MISMATCH` — and it would be
   **right**, not spurious. (An earlier note in this repo called it spurious pickle noise. Wrong.)
2. **Neither local store is redundant.** The desktop's orphaned 48 is the only record of what that
   box's results between 28 Aug and 18 Sept were scored against, and the server copy does not
   reproduce those models exactly. The behavior box's 46 is the ORIGINAL the server copy was made
   from, and `find()` prefers local to avoid the SMB read. **Delete neither.** ~16 MB each.

---

## 5b. WHICH MODELS ARE THE PUBLISHED RESULTS ACTUALLY STANDING ON? (checked 2026-09-21)

The question §5 provokes and does not answer. Checked rather than assumed, because a first pass
at it got the answer wrong.

**THE CURRENT FROZEN RESULTS ARE `cue_analysis/analysis_json/locanmf/locanmf_frozen_*_loso_roi_
{cue,lick,precue}.json`, written 2026-09-20, and ALL 24 (4 animals x 6 files) REPORT
`frozen-hit`.** Not one was refitted. All 24 distinct `frozen_model_id`s are present on MICROSCOPE
today, and all 24 are also in the analysis desktop's orphaned store.

Which store served them? On 2026-09-20 the desktop's `local_dir()` had been dead for two days and
MICROSCOPE held only two models, so the only store that could have supplied 24 hits is **the
behavior box's** — the same set published on 2026-09-21. So:

- **The published numbers came from the canonical models, consistently, and a re-run on any box
  now reproduces them**, because those exact ids resolve from the server.
- **The 0.3 pp divergence measured in §5 is between the behavior box's models and the DESKTOP's
  orphaned copies — and the desktop's copies never produced a published number.** The divergence
  is real and it is off to the side of everything that has been reported.

### A trap for the next reader: there are two files with the same name

`cue_analysis/locanmf_frozen_decoder_loso_roi_cue.json` (the parent directory) is dated
**2026-08-12**, predates the freeze mechanism entirely, and carries `frozen_model_id: null`.
The live one is a directory deeper, in `analysis_json/locanmf/`. **Opening the shallower path
gives six-week-old numbers that look current**, and only the `frozen_model_id` field distinguishes
them. Nothing has been deleted; check the path and check for a non-null id.

---

## 6. IF YOU ARE ADDING A MACHINE OR CHANGING `paths.yaml`

1. `python -c "from wfield_local import frozen_models as m; print(m.local_dir(), m.local_dir().exists())"`
2. If it prints `False` and that box has ever fitted models, **they are somewhere else.** Find them
   before letting a nightly refit over the top.
3. Prefer resolving through the published server set over re-fitting locally: one store is the
   entire point.

---

**2026-11-02: the state models gained `scaling="session_z"` in their spec (per-session z-score was specified but
never applied); old unscaled state models keep their ids as the record. DECISIONS 2026-11-02 (late).**

## 7. EVERY PRE-STROKE DECODER IS FROZEN — the rule for this and future cohorts (Priya, 2026-10-15)

*"Establish in the code for future cohorts that we will make a frozen pre-stroke decoder that is cached / stored
and reused."* Any decoder that scores post-stroke data against a pre-stroke reference goes through
`frozen_models.load_or_fit` -- never an in-memory refit per render:

| decoder | where | spec `kind` | since |
|---|---|---|---|
| position (cue / lick / precue; roi and joint bases) | `locanmf_position_decoder`, `pooled_frozen_*` | `decoder` / `encoder` | 2026-08/09 |
| behavioural STATE (quiet / running / licking) | `locomotor_decoder.frozen_state_models` | `state` | **2026-10-15** |
| epoch acc / 5c position decoder (imaging page 2.x; `_collect_5c` mode `frozen`) | `grant_confusion._stored_epoch_decoder` | `epoch_decoder` | opt-in 11-02 (lp-final), **default 11-02** (`frozen-default`) |
| docked REST position decoder (imaging page 12.1) | `rest_frozen_decoder.frozen_rest_payload` (+ `rest_frozen_decoder_nl.frozen_payload`, same spec for the page flags) | `rest_frozen` | **default 11-02** |
| frozen column of frozen-vs-refit (`_collect_5c` mode `paired`: epoch 5r / 5ro, `recovery_trajectory`, `prelim_numbers_frozen_vs_refit`) | `grant_confusion._stored_epoch_decoder` (the acc family's model, same spec) | `epoch_decoder` | **default 11-02** (`FROZEN_ALL_20261102`) |
| 5b confusions (`grant_figures --only 5b`, recompute path) | `grant_confusion._stored_epoch_decoder` (same spec) | `epoch_decoder` | **default 11-02** (`FROZEN_ALL_20261102`) |
| no-lick reference (`nightly_figs` -> `nolick_reference*.json`) | `nolick_decoder._stored_nolick_reference` (full + LOSO); opt-out `WIDEFIELD_NOLICK_FROZEN=0` | `nolick_reference` | **default 11-02 pass 2** |
| attempt-coding per-spout axes (imaging page 5.1 / 5.2 inputs; LP only) | `attempt_coding.stored_axes`; opt-out `WIDEFIELD_ATTEMPT_FROZEN=0` | `attempt_axes` | **default 11-02 pass 2** |
| June -> August drift decoder (imaging page fig 4.3's unmatched frozen column) | `scripts/june_august_drift.drift_payload` (one spec per month); opt-out `WIDEFIELD_DRIFT_FROZEN=0` | `drift_decoder` | **default 11-02 pass 2** |
| coding directions + per-class confusion decoder (`coding_direction.json` -> grant 3a / 5b, `miss_vs_stopped`) | `position_coding_directions.stored_reference` / `stored_confusion_decoder`; opt-out `WIDEFIELD_CODING_FROZEN=0` | `coding_directions` / `coding_decoder` | **default 11-02 pass 2** |
| CD trajectories reference (z-score frame, CIM, directions; imaging page section 9) | `cd_trajectories.stored_cd_reference`; opt-out `WIDEFIELD_CD_FROZEN=0` | `cd_directions` | **default 11-02 pass 2** |
| ITI / cued-lick ridge read-outs (12 scripts; encoding page section 3; LP only) | `frozen_readouts.pre_loso_ridge` (one helper; spec hashes the training data); opt-out `WIDEFIELD_ITI_FROZEN=0` | `iti_readout` | **default 11-02 pass 2** |

**LP arm, 11-03 (Priya): the DEFAULT is the LP-TRAINED model again** (spec `lick_source=lp`, trained on the LP arm's own
pre trials); the DAQ-trained model is the opt-in cross-check, `WIDEFIELD_LP_FROZEN=daq` (`frozen_models.lp_trained` /
`daq_trained_in_lp`). Section G's stored references (`g_*`) and the frozen encoder are NOT converted (pass-2 sections
P2.8 / P2.6), and no kind's spec sees a correction applied downstream of SVTcorr except `iti_readout` (P2.10).

The last two were refitted in memory on every render until 11-02 (Priya 11-02: *"we should cache frozen decoders as we
always do"*). Opt-outs, for reproducing an old render only: `WIDEFIELD_EPOCH_FROZEN=0`, `WIDEFIELD_REST_FROZEN=0`.
Under `WIDEFIELD_LICK_SOURCE=lp` the epoch family now LOADS the DAQ-trained model (it refuses to fit one); before 11-02
its in-memory fit was LP-trained. Verification and per-animal effects: `docs/status/FROZEN_DEFAULT_20261102.md`.
Since the later 11-02 commit (Priya: *"yes cache all frozen decoders"*) the paired frozen column (5r / 5ro) and 5b use
the same stored model, opt-out the same `WIDEFIELD_EPOCH_FROZEN=0`; under the LP arm all of them are DAQ-trained by
default and LP-trained with `WIDEFIELD_LP_FROZEN=lp`. `siblings()` now also groups by `post_s` and `lick_source`, so a
1.0 s / 2.0 s or LP / DAQ pair no longer reports a false SPEC-CHANGED (no id or stored model changed). Still in memory,
with the reason for each, and the inventory of every other pre-stroke reference fitted per run (coding directions,
nolick reference, section G discriminators, CD trajectories, ITI read-outs, the frozen encoder, ...):
`docs/status/FROZEN_ALL_20261102.md`.

The state decoder was refitted in memory on every render until 2026-10-15, so when PS92 0812 re-entered its
pre-stroke set (after the split-session fixes, `docs/EXPERIMENT_ERRORS.md`) every post-stroke state score moved
with it and nothing said so. Now its identity is animal + `state` + joint-basis id + feature count + the sorted
pre-stroke labels and their data signatures; the payload is the full model plus one leave-one-session-out model
per pre-stroke session (the PRE column). A grown pre-stroke set reports `SPEC-CHANGED` and freezes a new id.

**For a new decoder:** build the spec with `make_spec` (one animal, sorted PRE labels only -- post-stroke sessions
never enter an identity), put every model the analysis needs (full + LOSO) in one payload, and read
`load_or_fit`'s status in the log. **First freezes follow §5:** fit and publish from ONE box (the behaviour box
holds the canonical set); a second box's fit is a different reference to the optimiser's tolerance.

**State: the first freeze (2026-10-15).** Fitted on the analysis desktop (local store `C:/Users/SabatiniLab/frozen_models`,
created by this run) and published to `labcams/frozen_models` (4 models, `PS9x/state_segments_joint<basis>_<id>`, 11
pre-stroke sessions each). By Priya's decision **the analysis desktop is the one box that publishes STATE models**; the
behaviour box keeps publishing position models. On this desktop always set `WIDEFIELD_FIGURES_WORKING=C:/Users/
SabatiniLab/cue_lick` (its profile otherwise resolves local roots to `C:/Users/sabatini`, and `local_dir()` derives
from that root). It also has 48 orphaned position models in `C:/wf_local/frozen_models` (§4) -- untouched, not published.

---

## 8. FREEZE_VERSION 2 — THE TRAINING-DATA FINGERPRINT (cohort-1 rebuild, 2026-11-03; Priya: yes)

**Why.** §3's identity hashed the training LABELS and each session's `U_atlas` / `SVTcorr` stat signature, never the
data actually fitted. Two failures followed: (1) a changed TRIAL set under unchanged labels loaded the old model --
the PS92 section G models frozen 11-02 21:23 no longer reproduce with an identical spec, because PS92's pre trial
features changed afterwards; (2) a partial hemo switch (rebuild trap T4) gave "variant" spec ids to models fitted on
adopted data, because the signature is a file stat, not the bytes read.

**What every spec now carries** (`frozen_models.make_spec`, required argument `train_data`):

| field | value |
|---|---|
| `freeze_version` | 2 |
| `train_fp` | `data_fingerprint(train_data)`: sha1 of the training arrays (dtype, shape, bytes; sequences in order, dicts by sorted key) |
| `selection` | the run's trial-selection tag, `trial_classes.run_selection()` (`sel-daq-quittail` for the DAQ quit-tail rule, `sel-camera-<window>` for the camera gate), or an explicit `selection=` |
| `hemo_variant` | `config.hemo_variant()` (the one hemo switch) |

`train_data` per kind: decoder / epoch_decoder / coding_decoder / drift_decoder / nolick_reference = the pre-stroke
engaged rows (X, y, session) the model is fitted on; encoder = the same rows; attempt_axes = the video-hit reference
rows; coding_directions = engaged rows + positions (+ pre no-lick rows when the engagement axis uses them);
cd_directions = the pass-1 window means, labels, pre grand means, baselines and dropped components; state = every pre
session's (X, y, groups); rest_frozen = `rest_frozen_decoder.rest_train_data` (one builder for both rest scripts);
section G = the animal's whole pre pool (engaged + no-lick); iti_readout = its existing `data_sig`.

**Consequences.**
- EVERY id changes by design. The rebuild freezes into a FRESH store (`WIDEFIELD_FROZEN_MODEL_DIR`, rebuild order G3).
  `_write` REFUSES to put a v2 model into a store that already holds v1 models for that animal (the pre-rebuild
  `C:/Users/SabatiniLab/frozen_models`, the M: store), so the generations never mix and `refreeze` can never retire
  a v1 model in favour of a v2 one.
- A changed pool now mints a new id (reported as SPEC-CHANGED against its siblings) instead of loading a model trained
  on other data. Expect SPEC-CHANGED lines whenever a gate, a trial table or a hemo variant changes.
- **LP arm, DAQ-trained cross-check** (`WIDEFIELD_LP_FROZEN=daq`): the LP run cannot rebuild the DAQ run's arrays, so its
  spec carries `train_fp = "from-daq-run"` and `load_or_fit` finds the DAQ model by every other field
  (`find_daq_trained`, `base_id`); two DAQ models with different fingerprints under one identity raise (ambiguous).
- The ITI / lick-landing read-outs (`frozen_readouts`): only the SPEC changed. `ridge_fit` / `fit_all` / `data_sig` are
  pinned by source hash and the stored payload is pinned equal to the in-memory fit
  (`tests/test_frozen_fingerprint.py`); their per-session FRAME-level SD scaling (scripts' feature building, no centring
  over lick samples; DECISIONS fa70b651) is untouched.
- Tests use temp stores only.
