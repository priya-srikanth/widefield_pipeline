# Cohort2 handoff — 2026-11-08 (imaging-box window, pre-compaction)

Working-state handoff for the cohort2 nightly + the in-progress Allen-alignment task. Pipeline facts that
are already durable live in `docs/NEW_COHORT_2026-10-29.md`, `docs/490_VS_470_EXCITATION.md`,
`docs/COHORT2_QC_2026-10-29.md`; this is the transient "where we are / what's next".

## Pipeline state (dates)
- **Processed + archived + E:-cleaned:** 10-29, 10-30, 11-02, 11-03, 11-04, 11-05.
- **Processed, STILL on E: (not archived):** 11-06 (PS118, PS121 — both 470).
- SVD/hemo for every date is on N: `…/labcams/<date>/<session>/motion_corrected/wfield_local_results/`.
  Raw+bin on M: standby; DAQ h5 on N: `…/DAQ_recorder_output/<date>/`.
- Archive/clean flow: `python -m wfield_local.archive_day archive --date <d> --hash` then
  `clean --date <d> --hash --execute`. **Rule (Priya 11-06): git fetch→merge→push, NEVER rebase.**

## Decks (moved 2026-11-06)
- New location: **`N:/MICROSCOPE/Priya/Widefield/presentations/pipeline/cohort2/preprocessing/`**
  (root `decks_out`; was `labcams/cohort2/`). Split by indicator: `_PS118-PS120` (Thy1, 10 sessions),
  `_PS121-PS122` (RiboL1, 9 sessions, 40 slides, 4 embedded cam4/slow-mo videos).
- Rebuild wrapper: `scripts/cohort2_qc/build_cohort2_deck.sh` (desktop-enhanced). On the imaging box the
  cam4/`seizure_events_video`/`spout_maps` steps need data not here; the cheap path is: regenerate
  `discharge_screen.py` + `led_490_vs_470.py` + `led_490_vs_470_dprime.py`, run `preprocess_deck`
  (default → new decks_out), then `append_qc_slides.py` with the seizure-folder clips
  (`labcams/cohort2/_tmp_qc_figures_20261031/seizure/`).

## Key findings (so a reader has them)
- **PS122 epileptiform:** frank calcium discharges 10-29 and 11-04 (recurrent, intermittent; 470-only,
  bilateral, stereotyped). 10-30 vascular-only; 11-03 a few sub-threshold. **PS121 clean every date.**
- **490 vs 470:** RiboL1 at-best-matches its best 470 day (no clean gain; 490 always the last session →
  day/practice confound). GCaMP6s (PS118) neutral. No excess overdrive LED droop (490 fade < 415 fade).
  "d′" renamed → **single-trial response SNR** (it is mean/across-trial-SD = effect size, not an SDT d′).
- **PS120** behaves GCaMP-negative/WT (blood-flow null), genotype to re-check.

## IN PROGRESS: Allen affines for all 4 animals (the cross-reg reference = gate for LocaNMF/maps)
Reference session per animal = its latest large-ROI (509×480) session:
- **PS118 → 11-06**, **PS121 → 11-06** — raw on **E:**, ready. GUI launched for PS118 11-06.
- **PS120 → 11-03**, **PS122 → 11-03** — raw **archived to M:** (208 GB each), nothing on E:.

**GUI:** `python -m wfield_local.allen_register "<…_2_509_480_uint16.dat>"`. It saves
`dorsal_cortex_landmarks_<label>.json` **next to the .dat** (`sess_dir = dirname(dat_path)`).
Pipeline reads it from `<session>/raw_widefield_data/dorsal_cortex_landmarks_<version>.json`
(preprocess.py:359), version label **v1** (`reference_landmarks: v1` already in config).

**Read-from-M: plan (avoids the 208 GB restore):** the `.dat` can be READ from M: (only the averaged
frames are read), but the GUI must NOT write landmarks onto M: (immutable standby). Fix = add a
`--save-dir` arg to `allen_register` so it reads the M: `.dat` but writes the JSON to an E: session dir.
Then for these archived sessions, build the reference Allen dir by running `apply_allen_transform`
directly on the N: SVD outputs with `--landmarks <that json>` (preprocess's reference flow discovers from
E: raw, which is gone — so do the alignment step explicitly).

**GUI usage (how to draw):** pick the 470 channel (clearer); scrub time/avg sliders to a clean mean image;
set transform = **affine**; select each landmark on the "place landmark" radio then click its spot —
medial OB_center/OB_left/OB_right/RSP_base + lateral MOp_left/MOp_right/SS_left/SS_right (all 8 for affine);
tick **overlay** to morph the Allen atlas onto the image and nudge points until outlines match cortex;
type **v1** in the label box; **Save**. `Load .dat` switches to the next session (PS121 11-06) in the same window.

## Next steps
1. Add `allen_register --save-dir`; draw landmarks for all 4 (PS118/PS121 from E:, PS120/PS122 reading .dat
   from M:, saving to E:).
2. Set `reference_date` per animal (PS118 1106, PS120 1103, PS121 1106, PS122 1103) in
   `configs/cohort2/animals.yaml` (or cohorts.yaml master), `reference_landmarks: v1`.
3. Build each reference's Allen dir (preprocess for the E: ones; `apply_allen_transform` on N: for the
   archived ones), then register every other session → unblocks LocaNMF/maps.
4. Archive 11-06 once alignment doesn't need its raw on E:.
