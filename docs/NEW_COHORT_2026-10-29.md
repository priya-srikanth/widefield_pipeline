# NEW COHORT from 2026-10-29: cohort2 = PS118, PS120, PS121, PS122 — read before running anything

**For both the IMAGING (PCO) computer and the ANALYSIS / behaviour computer, after `git pull`.**

Imaging of a second cohort started 2026-10-29. It is analysed **fresh and separately**: it is never pooled
with PS92-95 (now called **cohort1**). The repo enforces that. This page says how, what each computer runs,
and what is still open.

## The new mice

| animal | genotype | indicator | notes |
|---|---|---|---|
| PS118 | Thy1-GCaMP6s GP4.3 (DOB 2026-08-03, M) | 470 nm (+415) | 10-29: small ROI, cam1 position A |
| PS120 | Thy1-GCaMP6s GP4.3 (DOB 2026-08-03, M) | 470 nm (+415) | 10-29: small ROI, cam1 position A |
| PS121 | VGLUT1-IRES-Cre x LSL-RiboL1-jGCaMP8s (DOB 2026-01-29, F) | 470 nm (+415) | 10-29: small ROI, cam1 position A; the 10-09 pilot PS121 was RiboL1-GCaMP8s |
| PS122 | VGLUT1-IRES-Cre x LSL-RiboL1-jGCaMP8s (DOB 2026-01-29, F) | 470 nm (+415) | larger ROI (kept for everyone from here on), cam1 position B |

PS118 / PS120 are **Thy1-GCaMP6s GP4.3**; PS121 / PS122 are **VGLUT1-IRES-Cre x LSL-RiboL1-jGCaMP8s**
(soma-localised jGCaMP8s); recorded as `genotype` in `configs/cohort2/animals.yaml`. Two indicators in one
cohort means every pooled result must also be shown split by `genotype`. Same task as cohort1 (one moving
spout, 6 positions, 4 cameras); a VLS stroke comes later (`stroke_date` null until then).

## How the separation works

* **`configs/cohorts.yaml`** lists the cohorts: their animals, their config files, their date range, and
  where their outputs go. cohort2's own files are **`configs/cohort2/animals.yaml`** and
  **`configs/cohort2/sessions.yaml`**. cohort1 keeps the original `configs/animals.yaml` +
  `configs/sessions.yaml`, unchanged.
* **One cohort is active at a time.** Every animal list, session list, curated / pre-stroke set, frozen
  decoder, joint basis, epoch file and deck sees ONLY the active cohort.
  * The **nightly commands choose it from the date**. 20261029 onward is cohort2, earlier is cohort1. They
    print `[cohort] active cohort = ...` at the top of the log, so check that line.
  * Otherwise **`WIDEFIELD_COHORT`** chooses it, and the **default is cohort1**. Every existing PS92-95
    analysis, script and page therefore runs exactly as before.
* **Cohort-level outputs for cohort2 go in a `cohort2/` subfolder.** That covers `figures_working`
  (local), `labcams/analysis_figures`, `Behavior_logs/Widefield/behavior_summary`, `LP_analysis`, and the
  labcams top level (deck, `epoch_boundaries.json`, `frozen_models/`, `joint_bases/`, `grant_figures/`), so
  `labcams/cohort2/spout_position_analysis_summary.pptx` etc. cohort1's files are never overwritten.
  Per-session data stays where it always was (`labcams/<date>/<animal>_<date>_*`, DAQ, cameras, alignment
  templates, xday), because it is keyed by animal.
* Discovery now accepts any `PSnnn` id (it only matched `PS9x` before, so PS118-122 would have been
  silently skipped). It processes only the active cohort's animals and prints a SKIPPED line for anything
  else: the 10-08 / 10-09 pilots, or the other cohort.

## What each computer does

**Imaging (PCO) computer, as before:**

```powershell
git pull
conda activate wfield
python -m wfield_local.nightly 20261029      # -> "[cohort] active cohort = cohort2"
```

Motion correction, SVD, the hemo variant, the push to MICROSCOPE, maps and photobleach all run.
**Cross-registration is SKIPPED** for every cohort2 mouse until it has a reference (below). The log
says `[xreg] PSxxx: no reference_date chosen yet ... SKIPPED` and the night carries on; that is expected.

**Analysis / behaviour computer, as before:**

```powershell
git pull
conda activate locanmf
python -m wfield_local.nightly 20261029      # camera + behaviour, then figs; cohort2 picked from the date
python -m wfield_local.await_locanmf 20261029   # registers cohort2 sessions into configs/cohort2/sessions.yaml
```

Camera upload, dropped-frame QC, alignment templates, behaviour events and per-session behaviour figures
work now. **LocaNMF and everything downstream need the Allen-aligned atlas from cross-registration, so they
wait for the reference step below.** `await_locanmf` will keep reporting the sessions as not ready until then.

**To analyse cohort2 by hand** (any script, any window), set the cohort first:

```powershell
$env:WIDEFIELD_COHORT = "cohort2"     # bash: export WIDEFIELD_COHORT=cohort2
```

A process loads its session list when it starts, so set this BEFORE launching. To work on PS92-95 again,
unset it or set `cohort1`.

## To do before cohort2 can be analysed across sessions

1. ~~Genotype, sex, DOB~~ entered 10-29 (`configs/cohort2/animals.yaml`).
2. **Reference session per mouse.** Pick one session per animal on the **large, fixed ROI**. For PS118 /
   PS120 / PS121 that means NOT 10-29 (see below). Click `dorsal_cortex_landmarks_<v>.json` on it, then set
   `reference_date` (MMDD) and `reference_landmarks` (`<v>`) in `configs/cohort2/animals.yaml`. Then re-run
   `python -m wfield_local.preprocess <dates>` on the imaging box. Motion and SVD are skipped as already
   done, and cross-registration runs.
3. **Variable imaging ROI (Priya 2026-10-29: "will need to be considered when aligning across sessions").**
   PS118 / PS120 / PS121 on 10-29 used the old, smaller ROI, and PS122 plus every later session use the
   larger one. `cross_day_align` currently RESIZES (stretches) a frame of a different size onto the
   reference. That is right for a binning change and **wrong for a different ROI**, which is a crop of a
   different region. It now prints a WARNING when it does this. Before any 10-29 small-ROI session is
   registered:
   * read each session's ROI (size and offset on the sensor) from the raw / camera settings;
   * change `_register_to_reference_native` and the warp in `run_reference_native` (`wfield_local/cross_day_align.py`) to
     EMBED the frame at its ROI offset in the reference frame, not resize it, and mask the area outside it;
   * check each registration QC image (`xday/<animal>_<mmdd>`) before pooling.

   If that is not worth it for one day, exclude 10-29 for those three animals from pooled sets
   (`exclude: ["1029"]` in `configs/cohort2/animals.yaml`), keeping their per-session figures.
4. **cam1 (bottom camera) moved between [PS118, PS120, PS121] and [PS122] on 10-29**, and a new calibration
   video is coming. Do not use a calibration taken after the change for the three earlier sessions. See
   `docs/EXPERIMENT_ERRORS.md`.
5. **This cohort's noisy / training days** go in `date_policy.cross_session_exclude` in
   `configs/cohort2/animals.yaml`. cohort1's June exclusions deliberately do not carry over.
6. **Before 2027-01-01: the year.** Sessions are keyed by MMDD, and cohort2's year is taken from its
   first session (2026). A January 2027 session would sort and compare before October 2026. Phase,
   curation and stroke cutoffs are all string comparisons on MMDD. Fix this before registering any 2027
   session, for example by keying cohort2 by YYYYMMDD and making the date helpers width-aware, or by
   starting a new cohort block at the new year.

## Things that are NOT cohort-aware yet (check before relying on them for cohort2)

* `scripts/` one-off analyses that hardcode `PS92`-`PS95` or cohort1 dates (e.g. `scripts/deck_claim_audit.py`,
  many results-page scripts). They are cohort1 records. Copy and adapt them rather than pointing them at
  cohort2.
* `configs/defaults.yaml` `preprocess.reference_date: "0606"` and `xall_min_date: "0605"` are cohort1 values.
  cohort2 uses each animal's own `reference_date`, and its MMDD dates (10-29 onward) pass `xall_min_date`.
* Pose models (DLC / LP for cam1 and cam4) were trained on cohort1. Check them on cohort2 video before
  trusting the tracking, especially cam1 after the move.
* Frozen pre-stroke decoders, joint bases and epoch boundaries are built per cohort (in the cohort's
  folders). cohort2 has none until it has enough pre-stroke sessions.

## Changed in the code (2026-10-29)

* `configs/cohorts.yaml` (new), `configs/cohort2/{animals,sessions}.yaml` (new).
* `wfield_local/paths.py`: active cohort (`WIDEFIELD_COHORT`), cohort-scoped output roots, a new logical
  root `cohort_out`.
* `wfield_local/config.py`: animals / sessions / date policy come from the active cohort, plus `cohort()`,
  `cohort_year()`, `cohort_of()` and `use_cohort_for_dates()`. An animal with no sessions yet is allowed.
* `wfield_local/preprocess.py`: `PS\d+` ids, an active-cohort filter, a per-animal `reference_date`
  (skips cross-registration when it is not set), and the date's real year (was hardcoded 2026).
* The `nightly`, `camera_nightly`, `behavior_events`, `spout_behavior` and `await_locanmf` entry points
  choose the cohort from the date. `await_locanmf` registers into the active cohort's sessions file.
  `nightly_figs` logs the cohort.
* Cohort-level output paths (deck, epoch file, frozen models, joint bases, grant figures, results folders)
  now use `root("cohort_out")`, which is identical to `labcams` for cohort1.
* `label[:4]` as "the animal" (wrong for `PS118_1029`, giving `PS11`) is replaced by `label.split("_")[0]`
  in ~130 places.
* `cross_day_align`: a WARNING when a frame is resized onto the reference.
* `tests/test_cohorts.py` pins all of the above.
