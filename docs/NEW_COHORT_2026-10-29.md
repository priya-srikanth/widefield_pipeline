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
says `[xreg] PSxxx: no master alignment session (reference_date) chosen yet -> cross-registration SKIPPED`
and the night carries on; that is expected.

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
2. **Master alignment session (the cross-registration reference).** Choose ONE date for the whole cohort,
   on the large, fixed ROI (not 10-29). Each mouse's own session of that day becomes its reference.
   * See every session with its frame size:
     `python -m wfield_local.alignment_reference --cohort cohort2 --sessions`
   * Set `reference_date: "<MMDD>"` in **`configs/cohorts.yaml`** under cohort2. `reference_landmarks` is
     already `v1`. A different date for one mouse goes in that mouse's `reference_date` in
     `configs/cohort2/animals.yaml`.
   * Click the landmarks. `python -m wfield_local.alignment_reference --cohort cohort2` prints the GUI command
     per mouse (`python -m wfield_local.allen_register "<.dat>"`). Run it on the imaging box, where the raw .dat
     is. The GUI reads the frame size from the file name, so any ROI works. Type `v1` in the label box and
     Save.
   * Re-run `python -m wfield_local.preprocess <dates>` on the imaging box. It copies the landmark file to
     MICROSCOPE, builds the reference session's Allen dir (`apply_allen_transform`, which was a manual step for
     cohort1), and registers every other session. Motion and SVD are skipped as already done.
     `alignment_reference` then shows `landmarks on MICROSCOPE: yes` and `reference Allen dir: yes`.
3. **Variable imaging ROI: handled (2026-10-29).** `cross_day_align` used to stretch a frame of a different
   size onto the reference. Every size mismatch here is a different ROI, not binning: cohort1 06-01 to 06-03,
   and cohort2 10-29 PS118 / PS120 / PS121. Such a frame is now PLACED at its offset at native scale, using a
   masked phase correlation plus the usual rotation refine, and the old stretch is kept only where it matches
   better.
   * Measured on cohort1: PS94 / PS95 06-03 NCC 0.94 -> 0.99 and 06-01 0.27-0.40 -> 0.48-0.64. PS92 06-02 /
     06-03 register poorly either way and keep the stretch.
   * The pixels a smaller ROI did not see are ZERO in its warped U and means, and are recorded as
     `roi_coverage_atlas.npy` in its Allen dir (plus `*_roi_coverage_in_ccf.npy` in `xday/`). Anything that
     pools such a session pixel-wise or by area must mask by that file.
   * Check each registration QC image (`xday/<animal>_<mmdd>`) before pooling.
4. **cam1 (bottom camera) moved between [PS118, PS120, PS121] and [PS122] on 10-29**, and a new calibration
   video is coming. Do not use a calibration taken after the change for the three earlier sessions. See
   `docs/EXPERIMENT_ERRORS.md`.
5. **This cohort's noisy / training days** go in `date_policy.cross_session_exclude` in
   `configs/cohort2/animals.yaml`. cohort1's June exclusions deliberately do not carry over.
6. **The year: handled (2026-10-29).** Session keys stay MMDD, and the year comes from the cohort's `first`
   date. For cohort2, 1029-1231 is 2026 and January onward is 2027 (`config.date8`). Phase, curation, stroke
   cutoffs, session order and every `YYYYMMDD` path built from a label use it.
   * `load_sessions` checks each registered session against the date in its own folder path, so a cohort that
     outgrows one year fails loudly.
   * Remaining caveat: analysis code that sorts labels as plain strings would put January before October.
     Sort with `config.date_key`.

## Open question (Priya 2026-10-29): PS121 / PS122 415 "dimmed a TON" — raise 415 LED next time, or usable as is?

**Answer: usable as is; do NOT raise the 415 LED.** The perceived dimming is a *within-session fade
(bleaching)*, not a low light level. Raising 415 power would not meaningfully help the correction and
would make the fade worse.

Measured on 2026-10-29 (sampled raw frames, brain-ROI mean; PCO ~0.46 e-/count; offset = darkest-corner
mean = electronic dark level; shot-noise ΔF/F = 100/√(photoelectrons)):

| session | 470 func | 415 iso − offset | 415 / offset | shot-noise ΔF/F (1 px,1 frame) | 415 start→end |
|---|---|---|---|---|---|
| PS118 Thy1 | 25493 | 23962 | 18× | 0.95% | −24% |
| PS120 Thy1 (dim) | 16348 | 14769 | 11× | 1.21% | −43% |
| PS121 RiboL1 | 32922 | **30598** | **22×** | **0.84%** | **−34%** |
| PS122 RiboL1 | 29904 | **30690** | **24×** | **0.84%** | +4% |

Reasoning:
* **415 is not photon-starved in the RiboL1 mice — it is the brightest of the cohort** (~30,600 counts
  over offset, 22–24× the camera floor, ~49% of 16-bit range → no saturation). Its single-pixel
  shot-noise ΔF/F floor (~0.84%) is the *best* of the four. More 415 power buys negligible SNR.
* **What actually varies is bleaching, and it is differential between the two channels** — exactly what
  a scalar isosbestic regression handles worst. PS121: 415 fades −34% (470 rises +18%). PS122: 470 fades
  −41% while 415 is flat (+4%). A single 415→470 coefficient cannot match both baseline and dynamics when
  the two channels drift apart.
* **The lever for this is software, not the LED.** It is detrended in `hemo_variants` (adopted
  `meegkit_hpfit`), and the candidate `meegkit_2band` (slow_band_s=300) adds a second 415→470 coefficient
  for the slow band specifically to absorb this differential drift. Decision gated on inspecting
  PS121/PS122 `hemo_*/SVTcorr.npy` after this run (to-do #4 below).
* If you want to reduce bleaching *at acquisition*, the knob is **total light dose** (session length, LED
  duty / exposure), not 415 amplitude — and if anything 415 has enough headroom that it could be turned
  *down* without hurting SNR. Only raise 415 if a future animal's 415 drops toward the camera floor
  (say < ~5× offset) or the SVTcorr shows injected 415 shot noise; neither is the case here.

(Verify once hemo finishes: a clean `SVTcorr` with no 415-shaped residual confirms "usable as is".)

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
* Later on 10-29 (Priya):
  * `cross_day_align._register_different_roi`: a different camera ROI is placed at its offset, with coverage masks.
  * `config.date8` / `date_key`: years come from the cohort's `first` date, and are checked against each session's path.
  * Hardcoded `"2026" + mmdd` replaced in ~25 sites.
  * A cohort master alignment session (`cohorts.yaml reference_date`).
  * Preprocess builds the reference's Allen dir and copies its landmarks to MICROSCOPE.
  * New `wfield_local/alignment_reference.py`.
  * New tests in `tests/test_cross_day_roi.py`, plus more cases in `tests/test_cohorts.py`.
