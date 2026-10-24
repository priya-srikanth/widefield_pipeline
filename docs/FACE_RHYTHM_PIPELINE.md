# FaceRhythm pipeline (cam4; cam1 / cam2 / cam3 prepped) — stages, files, run order

**New here? Start with [`FACE_RHYTHM_GUIDE.md`](FACE_RHYTHM_GUIDE.md)** (concepts, the analysis catalogue `python -m wfield_local.face_rhythm list`, adding an analysis). The reference for HOW the FaceRhythm (FR) analysis works and how to run it. WHY each choice was made:
[`FACE_RHYTHM_PLAN.md`](FACE_RHYTHM_PLAN.md) + `DECISIONS.md` (2026-10-21 evening entries). Current status / running jobs:
the newest `docs/status/STATUS_*_FACE_RHYTHM_*` handoff. Paper: Hakim et al., bioRxiv 10.1101/2025.09.10.675423 v4.

## Stages

| # | stage | code | runs on | output |
|---|---|---|---|---|
| 1 | session frames, registration to the master face (PS95_20260812, pre-stroke) | `scripts/face_rhythm_prep.py`, `face_rhythm_register.py` | desktop | `<dlc>/face_rhythm/cam4/{frames,register}/` |
| 2 | master ROI (face + mask polygons; Priya) | `notebooks/face_rhythm_draw_roi.ipynb` (JupyterLab, fr env) | desktop | `rois/PS95_20260812/ROIs.h5` |
| 3 | mesh at 12 px warped to every session | `scripts/face_rhythm_warp_rois.py to-sessions`; QC `face_rhythm_mesh_qc.py` | desktop | `rois/<a>_<d>/ROIs.h5`, `rois_qc_12px_*/` |
| 4 | tracking + spectrograms (one task = session x video) | `scripts/o2_face_rhythm.py bundle / collect / qc` (runner `fr_runner.py`) | O2 array; collect + qc desktop | `<bundle>/out/<key>_{trials,session,points}.npz`; `qc`: mesh folds (INTERIOR vs RIM triangles), frozen points, displacement by animal x epoch -> `<dlc>/face_rhythm/cam4/o2_<name>_qc.png` |
| 5 | master PRE-STROKE factor set (frozen) | `scripts/face_rhythm_factors.py plan / build / fit / report / select` | plan, report, select: desktop; build, fit: O2 | `<bundle>/factors/<set>/master.npz` |
| 6 | every session on the frozen set + its own fit | `face_rhythm_factors.py project / fitsession` | O2 arrays | `<set>/proj/<key>.npz`, `<set>/own/<key>.npz` |
| 7 | session readouts | `scripts/face_rhythm_readouts.py` (computation: `wfield_local/face_rhythm_readouts.py`) | desktop | `<dlc>/face_rhythm/cam4/readouts/<set>/` (figures; CSVs in `data/`) |
| 8 | FR in the widefield encoding model | `wfield_local/face_rhythm_outputs.py` (loader), `combined_encoding.py --fr <set>`, `o2_encoding.py bundle --fr <set>` | O2 / desktop | `LP_analysis/<run>/imaging_encoding/<order>[_video-..]_fr-<set>/` |

`<dlc>` = `M:\MICROSCOPE\Priya\DeepLabCut\Widefield`; `<bundle>` = `<dlc>\o2\fr_cam4_20261021` (share) and
`/n/scratch/users/p/ps150/o2/fr_cam4_20261021` (O2). Parameter choices (tracking `midlk12`, VQT, notches): stage-4
`PARAMS` in `o2_face_rhythm.py`, test record in DECISIONS.md.

## Other cameras (cam1 / cam2 / cam3; prepped 10-22, Priya: "prep for all 3 cameras")

Same chain per camera, root `<dlc>/face_rhythm/<cam>/`, one O2 bundle per camera. What differs is in ONE place,
`wfield_local/face_rhythm_cams.py` (view, frame size, landmarks, rest-frame exclusions, which side of the face is seen).

| view | what it adds over cam4 | ROI / mask care |
|---|---|---|
| cam2 = animal's LEFT side (IPSILESIONAL; all lesions left), 450 x 600 | eye, ear, side of nose / mouth / chin, whisker pad from the side | face polygon = HEAD only; mask off body, forepaws, treadmill and the spout rod (crosses just below the mouth in the rest face) |
| cam3 = animal's RIGHT side (CONTRALESIONAL; PS93's deficit side), 450 x 600 | as cam2, other side | as cam2 |
| cam1 = from below, 600 x 600 | nose tip, lips, chin, whiskers both sides (ipsi = image RIGHT) | the CENTRE spout rod hangs through the middle of the face |

1. **Rest faces**: `python -m scripts.face_rhythm_prep frames --cam <cam>` (locanmf). Rest moments come from cam4's LP
   (tongue absent, -1.5 .. -0.5 s), mapped to the camera's frames through ITS alignment template
   (`face_rhythm_cams.frame_daq_seconds` / `frames_at`); cam1 skips centre-spout trials. No LP landmarks on these views.
2. **Templates, then clicks**: `<fr> scripts/face_rhythm_register.py --cam <cam> --stage templates` picks each animal's
   template + the master and writes `register_inputs/click_<s>.png`; Priya clicks the camera's landmarks in
   `notebooks/face_rhythm_click_landmarks.ipynb` -> `register_inputs/landmarks.json`.
3. **Registration**: `<fr> scripts/face_rhythm_register.py --cam <cam>`: template -> master from the clicked landmarks
   (similarity), session -> template from an image-only ECC fit (`coarse_field`), DeepFlow on top kept only where it
   raises the whole-image correlation. QC: `register/qc_ncc.csv` (correlation with the master before / after) + overlays.
4. **ROI**: `notebooks/face_rhythm_draw_roi.ipynb` with `CAM = "<cam>"`, SESSION = the master; then
   `<fr> scripts/face_rhythm_warp_rois.py to-sessions --cam <cam>` and mesh QC.
5. **O2**: `python scripts/o2_face_rhythm.py bundle --name fr_<cam>_<date> --user ps150 --cam <cam>`: cam4's trials and
   DAQ grid on this camera's clock (`other_cam_grid`; checked on PS92 0818: same 439 trials and coverage, cam2 ~14
   frames behind cam4), same runner and notches (every camera has the 12-frame keyframe). A one-session pilot first.
6. **Downstream**: factor sets per camera (`face_rhythm_factors.py --bundle fr_<cam>_<date>`); the loader finds any
   camera's keys; encoding takes another camera's set as `--fr fr_<cam>_<date>/<set>` (folder tag `<cam>-<set>`).
   NOT done yet: readouts' midline / regions are cam4's (side views have no midline: compare cam2 vs cam3 instead);
   combining several cameras' sets in one encoding model.
7. **Multi-view factor set** (10-23, Priya chose this over a colleague's 2 x 2 mosaic video): ONE factor set over
   several cameras' points, `plan --bundle fr_cam4_<date> --set <set> --views cam1=fr_cam1_<d> cam2=fr_cam2_<d>
   cam3=fr_cam3_<d>` (the bundles side by side under `o2/`; `FR_OUT_<CAM>` overrides an out dir), then the usual
   build / fit / loso / select / project / fitsession / report. Code: `wfield_local/face_rhythm_multiview.py` (a lazy
   `ViewStack` stands in for the spectrogram memmap, `face_rhythm_factors.spec_of`). Rows: trial rows match by index
   (same cam4 DAQ grid; cue times checked to 20 ms), session 5 Hz bins by nearest DAQ time (<= 0.1 s, else NaN ->
   dropped). Balance: per animal, each view divided by RMS x sqrt(its elements / mean) -> equal energy per view
   (`view_scales.json`, in master.npz). Absolute expression: D = mean of the views' own FR normalisers (provisional).
   master.npz gains `view_cams`, `view_P`; the gallery draws the reference view and lists each factor's weight share
   per view. Why not a mosaic: independent camera clocks + frame drops (a mosaic needs every video resampled to one
   clock), ~3.7x cam4's pixels per run, and ONE FR normaliser across all views. Tasks lacking any view are left out
   (listed by `plan`). Needs the other cameras' O2 runs first (steps 2-5). Tests: `tests/test_face_rhythm_multiview.py`.

## Files

* **out/<key>_trials.npz** `spec` (trial, 151, 2, P, F) float16: engaged trials, 25 Hz bins cue -2 .. +4 s on the DAQ
  clock (`t_ms`, `cue_daq_s`, `position`, `trial_id`); **_session.npz** `spec` (bin, 2, P, F) at 5 Hz + `t_daq_s`;
  **_points.npz** `disp` (bin, P, 2) 25 Hz displacement from home + `viol_frac`. P = 1282 mesh points (same face location
  in every session), F = 36 freqs 1-60 Hz. Uncompressed: `face_rhythm_factors.npz_memmap` reads them without copies.
* **factors/<set>/plan.json** (desktop `plan`): every task + epoch; master fit rows (per animal: 60 engaged pre-stroke
  trials balanced over positions + between-trial bins, equal per animal); each task's own-fit rows. Seeded.
* **master.npz** (`select`, FROZEN): `spatial` (2, P, R), `spectral` (F, R), `freqs`, `home`, `scales_json` (one scalar
  per animal), `spec_hash`; plus the dense-POSITION basis `points_components` (K, 2P) / `points_explained` (PCA of the
  25 Hz displacements, each session centred on its own mean, equal pre-stroke bins per animal). Never refitted once used (rule 10); `select` refuses a different spec.
* **proj/<key>.npz**: `trials` (trial, 151, R), `session` (bin, R) + `session_t_daq_s`, per-bin FR normaliser
  `trials_D` / `session_D` (absolute expression = weights x D), per-bin `trials_err` /
  `session_err` (relative reconstruction error), per-point movement power `power_between` (2, P, F),
  `power_precue` / `power_response` (position, 2, P, F) + `power_positions`, `power_n_bins`; `points_pc` (25 Hz bins, K)
  + `points_t_daq_s` (dense position on the frozen PCA).
* **own/<key>.npz**: the session's own rank-R fit (`A_own`, `B_own` in the master's component order), `evr_own`,
  `evr_frozen` (same rows), `cos_spatial` / `cos_spectral` (matched cosine per component).

## Time windows and the fit sample (master_pre_v1, decided 10-23 with Priya)

All times are DAQ seconds; every bin carries its TRUE time from the camera's per-frame timestamps
(`face_rhythm_timing`, `o2_face_rhythm.py timing` -> `<bundle>/timing/<key>.npz`), never the runner's linear
frame -> DAQ stamps (841 s off on PS92_0908). LOSSY bins (> 2 % of frames missing within +-2 s) and bins outside the
camera's DAQ segment are excluded everywhere (NaN in projections; never drawn for a fit).

**Windows around each ENGAGED trial** (the cohort tables' gate; disengaged trials are not on the grid):

| window | span | resolution | source | used for |
|---|---|---|---|---|
| TRIAL grid | cue -2 .. +4 s | 25 Hz (151 bins) | `<key>_trials.npz` (computed on O2) | trial rows of the fit (every 5th-bin block -> 200 ms rows); projections; readouts (pre-cue -2..0, response 0..4) |
| PRE-CUE rows | [position strobe -1, strobe +2] UNION [cue -4, cue -2), cut at cue -2 s | 5 Hz | session spectrogram | fit rows drawn WITH their trial: spout arrival (the strobe; its lead to the cue varies, 2.4 s to > 10 s) + anticipation / ENL licks; each bin once, never inside the 25 Hz grid |
| BUSY (no between rows) | from min(strobe -1, cue -4, cue -3) to cue +6 s | -- | -- | keeps trial tails, arrival and anticipation out of the between pool |
| BETWEEN | everything else (2 s trimmed at each recording end) | 5 Hz | session spectrogram | between rows of the fit, labelled from the CANONICAL per-session events on the server (`behavior_events`, `behavior_summary/events/<animal>/<date>.npz`; no DAQ read): RUN = inside a running bout (> 3 mm/s, gaps < 0.3 s merged, >= 2 s), STILL = inside a REST period (the lab's quiet definition: spout docked between trials, not running, not near licks / rewards), other = not drawn |

Projections cover WHOLE sessions (25 Hz in trial windows, 5 Hz everywhere else); only the FIT uses a sample.

**Per animal** (pre-stroke, August baseline sessions: PS92 9, PS93 7, PS94 8, PS95 8):
* 120 fit trials = **20 per spout position** x 6, drawn at random from the animal's pooled pre sessions (no per-session
  quota; trials with a lossy grid bin excluded) -> 120 x 30 = 3600 trial rows (200 ms; `--trial-ds 5`).
* their pre-cue rows, then subsampled to the SAME count per animal (the smallest animal's): the raw count follows each
  animal's strobe -> cue lead (drawn 10-23: PS92 2425, PS93 2055, PS94 1567, PS95 1246 -> 1246 each).
* between rows = 0.556 x the trial rows (~2000), **half RUN, half STILL (REST)** (`--run-share 0.5`; if one pool is
  short the other fills; "other" bins are not used).
* `plan` prints each animal's drawn counts; the plan stores them (`fit.<animal>.n_between`).
* Drawn 10-23 (master_pre_v1): per animal 120 trials (3600 rows) + 1246 pre-cue + 2002 between rows = ~6850; run /
  still 1001 / 1001 except PS92 794 / 1208 (it runs little between trials; rest fills). 27.4 k rows, 10.1 GB.

**Across animals**: the SAME sample size and composition per animal (trials, positions, run / still), so no animal
dominates by having more sessions; then each animal is divided by ONE scalar (RMS of its own fit rows, `scales.json`)
so faces of different size / tracked area carry equal energy (a per-point scale would amplify still points). The same
scalar is applied to all that animal's sessions, pre and post, so post-stroke amplitude changes stay visible.

**Scoring rows** (`session_rows`, for `fitsession` / `loso`): per session, 60 engaged trials (10 per position or all it
has) + between rows in the same proportion and run / still split; seeded per session.

**Row spacing vs time resolution** (Priya 10-23: "280 ms is too long with respect to the timing of a lick"): a fit
row is the MEAN of `trial_ds` consecutive 25 Hz bins (not a subsample). FR's VQT (params.json: Q 4 -> 20 geometric
over 1-60 Hz, Gaussian windows sigma = Q / (4 f)) already smooths each frequency over: 1 Hz sigma 1.0 s; 5 Hz 0.38 s;
7 Hz 0.31 s (FWHM 0.72 s, ~5 lick cycles); 10 Hz 0.25 s; 30 Hz 0.13 s. A 200 ms boxcar (sigma 58 ms) widens the
lick-band smoothing by ~2 % (30 Hz: ~10 %); individual licks are never resolved by the spectrogram -- it measures
rhythmic power over ~0.7 s. Rows stay at 200 ms (`--trial-ds 5`); not coarser. Lick-level timing comes from LP licks
against the 25 Hz projections, not from the fit rows.
**Is the sample enough?** `face_rhythm_factors.py splithalf --rank R`: two disjoint halves of the fit trials (10 per
position per animal each) fitted separately and compared (per factor); run by the local driver after the rank choice.

Plan: `face_rhythm_factors.py plan --bundle fr_cam4_20261021 --set master_pre_v1 --trials-per-animal 120 --trial-ds 5
--between-frac 0.556 --run-share 0.5 --arrival --inits 8 --ranks 8 10 12` (local run, Priya: "local, trimmed").
The PILOT (`pilot_pre_PS92_PS93`, 10-22) predates this: 30 trials / animal at 80 ms rows, 25 % uniform between bins,
no pre-cue rows, linear stamps (its two affected tasks re-projected with true times, 10-23).

## Conventions

* **Figures** (10-22, Priya): every FR figure goes through `wfield_local/face_rhythm_figures.py` -- readable fonts
  (`style()`), constrained layout with legends outside the axes (`subplots`, `legend_outside`: no title / legend
  overlap), SHORT titles, and `save(fig, path, methods)`, which writes the figure's METHODS (data, selection, units,
  normalisation, caveats) as a section of the output folder's `README.md`. Tables in `data/` (`figure_layout`).
  Factor identity vs behaviour: `scripts/face_rhythm_validate.py` (licks by executed direction x time in trial, by lick
  number in bout, by spout position, vs lick frequency; running onsets; correlations).
  Fewer licks or different licks by epoch: `scripts/face_rhythm_lick_epochs.py` (licks / trial by direction; each
  factor's expression AT a lick, relative to pre; total = licks x per lick for the factor's preferred direction).

* **cam4 ipsilesional = IMAGE RIGHT** (all lesions left; brain maps are the other way). Readouts split the mesh at its
  midline: x = 342, the master's LP nose (`face_rhythm_readouts.MASTER_MIDLINE_X`; `--midline-x` overrides).
* **Scaling:** one scalar per animal (RMS of its pre-stroke fit rows), applied to every session of that animal -- removes
  face size, keeps post-stroke amplitude changes.
* **EVR** is uncentred, 1 - |S - S^|^2 / |S|^2 (the paper's definition).
* **FR spectrograms are normalised per frame** by the mean power over the whole face (VQT `normalization_factor`
  0.95): the pattern across points and frequencies is kept, overall amplitude over time is not (found in the 10-21
  dry run). Factors are FITTED on the normalised data (the paper's choice: shape, not a few huge events). For HOW MUCH
  a component is expressed, `project` recovers each bin's normaliser D from the saved spectrogram (exact formula;
  0.01 % median error per 25 Hz bin) and the absolute expression is weights x D (`trials_D` / `session_D`; readout
  columns `abs_*` next to the share `amp_*`; loader `absolute=True`; `report --absolute`). Per-point movement: SPEED
  from the 25 Hz points (absolute) and SHARE from the spectrogram (`--measure`).
* **Epochs** `wfield_local.epochs.epoch_of`; pre = August only (June not in the bundle); statistics through
  `orofacial_cohort.epoch_tables` (per-animal means, paired animal-weighted `boot_delta`); per-animal table first (rule 8).
* **Engaged trials only** in `trials`; the whole session (incl. disengaged trials, between trials) stays in `session` /
  `points`.

## Run order (after the stage-4 array)

```bash
# transfer node: outputs -> share (desktop collect) and the factors folder share -> scratch
rsync -ah <scratch bundle>/out/ <share bundle>/out/
rsync -ah <share bundle>/factors/ <scratch bundle>/factors/
# desktop
python scripts/o2_face_rhythm.py collect --name fr_cam4_20261021
python scripts/o2_face_rhythm.py qc --name fr_cam4_20261021        # post-stroke tracking as clean as pre?
# O2: master fit
sbatch <scratch bundle>/factors/master_pre_v1/run_build.sbatch
sbatch --dependency=afterok:<build id> <scratch bundle>/factors/master_pre_v1/run_fit.sbatch     # ranks 6-16
# transfer: rank*.json / rank*.npz / scales.json back; desktop: look, choose R, freeze
python scripts/face_rhythm_factors.py report && python scripts/face_rhythm_factors.py select --rank R
# transfer: master.npz -> scratch; O2:
sbatch .../run_project.sbatch ; sbatch .../run_fitsession.sbatch
# transfer: proj/ own/ back; desktop
python scripts/face_rhythm_readouts.py
python scripts/o2_encoding.py bundle --name enc_fr_<date> --user ps150 --fr master_pre_v1     # then COMMANDS.md
```
Factor CLIPS for naming (fr env): `python scripts/face_rhythm_factor_clips.py --fdir <set> [--key K] [--comps ...]` ->
`<dlc>/face_rhythm/cam4/factor_clips/<set>/<key>_f<k>.mp4`: each factor's strongest moments (absolute expression), the
tracked mesh coloured by the direction the factor moves each point (blue lateral, red vertical; Rich's convention),
size = weight x expression, with an expression strip and time from cue.

Name the components: the readouts write `<set>/labels_suggested.json` (rules in `face_rhythm_readouts.describe_components`:
peak frequency, face region around the master's LP nose, lick-rate correlation, far-L / far-R selectivity, rim share)
and `components.csv`; after checking them against the gallery, copy to `<set>/labels.json` (used by the loader and the
readouts; names only, no effect on numbers).

## What each readout answers (2pRAM-cohort manuscript Fig 4 -> here, across animals)

| question | readout |
|---|---|
| do pre-stroke components (licking, sniffing, whisking) lose amplitude after stroke and recover? | `amp_<comp>_<precue/response/between>` by epoch |
| is post-stroke behaviour still described by the pre-stroke repertoire? | `recon_err_*`, `evr_gap` (own - frozen) |
| do the components themselves change shape ("morphing")? | `cos_spatial_*`, `cos_spectral_*` |
| where on the face does movement change, and is it lateralised? | `movement_maps_<window>.png`, `asym_<window>` |
| are non-task movements preserved? | `between` vs `response` amplitudes |
| does FR explain cortex beyond / instead of motion energy; does position coding survive FR? | encoding `fr:*`, `frvideo:*` models |
| does dense face POSITION add to FR (the paper's points + FR)? | encoding `--fr-points K`: `frpts:*`, `frboth:raw:full` |

Encoding uses ABSOLUTE FR expression by default (Priya 10-21; a share drops whenever any other movement grows);
`--fr-share` = the paper's normalised regressors (folder `..._fr-<set>share`). Check the absolute regressors' tails on
the first real session (log-transform if a few outliers dominate after z-scoring).

## Tests

`tests/test_face_rhythm_factors.py` (synthetic bundle: planted factors recovered, freeze guard, project / fitsession /
movement power), `tests/test_face_rhythm_readouts.py` (windows, contra / ipsi sign, split-session join),
`tests/test_face_rhythm_outputs.py` (loader), `tests/test_face_rhythm_tracking_qc.py` (fold counting, interior vs rim), `tests/test_combined_encoding_options.py` (`_fr-<set>` folders).
