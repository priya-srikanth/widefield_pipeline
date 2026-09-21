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

## 6. IF YOU ARE ADDING A MACHINE OR CHANGING `paths.yaml`

1. `python -c "from wfield_local import frozen_models as m; print(m.local_dir(), m.local_dir().exists())"`
2. If it prints `False` and that box has ever fitted models, **they are somewhere else.** Find them
   before letting a nightly refit over the top.
3. Prefer resolving through the published server set over re-fitting locally: one store is the
   entire point.
