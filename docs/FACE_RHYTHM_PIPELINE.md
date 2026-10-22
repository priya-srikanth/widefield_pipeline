# FaceRhythm pipeline (cam4) — stages, files, run order

The reference for HOW the FaceRhythm (FR) analysis works and how to run it. WHY each choice was made:
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

## Conventions

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
