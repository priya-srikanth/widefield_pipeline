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

## Decks — PER-ANIMAL now (Priya 2026-11-08, was indicator-split)
- Location: **`N:/MICROSCOPE/Priya/Widefield/presentations/pipeline/cohort2/preprocessing/`**
  (root `decks_out`). **Four per-animal decks** `cross-session_preprocessing_PS{118,120,122,121}.pptx`
  (cohort1 style), built with `WIDEFIELD_COHORT=cohort2 python -m wfield_local.preprocess_deck
  --max-sessions 1` (forces one bucket per animal; old `_PS118-PS120`/`_PS121-PS122` auto-pruned).
- Deck changes committed 2026-11-08 (`preprocess_deck.py`): (1) `spout_native` (CCF-free native-pixel
  maps) REMOVED — superseded by the Allen-overlay cue/lick maps now that every session is warped;
  (2) `photobleach` is now ONE trailing "Photobleaching" block at the END (all animals), via
  `TRAILING_TYPES`, matching cohort1 (was per-animal-interleaved on multi-animal decks).
- **quiet-vs-running maps now on all 19 sessions**: the figure needs the `behavior_events` npz, which
  only existed for 10-29. Generated for every cohort2 date (`behavior_events <date>` → cohort2-scoped
  `behavior_summary/cohort2/events/<animal>/<date>.npz`). NB ad-hoc map scripts must set
  `WIDEFIELD_COHORT=cohort2` or the resolver returns the UN-scoped `behavior_out` and the lookup misses.
- **490 labeling** works off the folder tag `PS<n>_490_<date>` (`led_nm`): PS118 11-05, PS121/PS122 11-04
  show "(490 nm)" on every slide (photobleach title reads "415/490").
- QC appends (`append_qc_slides.py`, MANUAL post-build — a `preprocess_deck` rebuild DROPS them):
  490 comparison (`E:/labcams_data/cohort2_490_vs_470{,_dprime}.png`) → PS118/PS121/PS122 decks;
  discharge screen (`cohort2_discharge_screen.png`) → PS121/PS122 decks. cam4/slow-mo seizure videos
  are analysis-box assets, not on the imaging box.

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

## DONE 2026-11-08 (alignment)
- `allen_register --save-dir` added (read .dat from M:, save landmarks to E:); landmarks drawn for all 4
  (affine, 8 pts): PS118 1106, PS121 1106 (E:), PS120 1103, PS122 1103 (read .dat from M:).
- `configs/cohort2/animals.yaml` reference_date set per animal (1106/1103/1106/1103), reference_landmarks v1.
- Reference Allen dirs built (`apply_allen_transform` on N:); all 19 sessions registered + warped to their
  reference CCF (`cross_day_align` reference-native, warp_u=True) from N: — NCC 0.88-1.0, small-ROI 10-29
  placed at offset + roi_coverage masked. Per-session `allen_aligned_affine8v1/` (U_atlas, atlas, masks) on N:.
  Cross-day QC `labcams/xday/<animal>_xall/<animal>_cross_day_alignment_qc.png` (2 slides/deck already).
- Maps (cue/lick + Allen overlay) built per session from N: (`scripts`-style `_maps_commands`); deck shows them.

## STILL TO DO: glue mask (fiber-glue occlusion) — AFTER the CCF settles
`paint_exclusion.py` (`python -m wfield_local.paint_exclusion`): hand-paint, PER ANIMAL, the fiber-glue
occlusion no threshold can find (brightness here is dominated by window geometry). `beta_maps.brain_mask()`
applies it and **every map / LocaNMF / decoder consumes it** — additive to `EXCLUDE_REGIONS` (MOB etc.).
**It is painted in CCF/atlas space** (loads `frames_average_atlas.npy` + CCF outlines, 540x640). THEREFORE it
is tied to the current alignment: **refine the CCF → re-paint the glue** (small nudge usually fine; material
change = redo). So **settle the CCF first, then paint the glue** (it's the last masking step). cohort1 painted
2026-09-12. Controls in the painter docstring (drag=paint, 0=all-animals mask, 1-4=per-animal, s=save).

## If refining the CCF
Re-draw that animal's landmarks (same GUI, label v1), re-run `apply_allen_transform` for its reference, then
re-run the `cross_day_align` warp (warp_u=True) + the maps for its sessions, then re-paint its glue mask.
