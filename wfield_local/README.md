# Local wfield Processing Helpers

This folder is the `wfield_local` package of the **`widefield_pipeline`** repo (carved out of
`Widefield_DAQ_recorder` — the recorder GUI — in 2026-08). It holds BOTH the imaging-box
**preprocessing** helpers (motion / SVD / hemo / Allen / cue-lick maps, §1–16 below) AND the
behavior-GPU **analysis** pipeline (LocaNMF spout-position decode / encode / RSA, §17–23). Large imaging
outputs stay outside git.

Day-to-day the individual steps below are driven by **config-driven entry points** (single source
of truth = `configs/*.yaml`, resolved by `config.py` + `paths.py`) — see **Running the pipeline**
next. The repo is `pip install -e .`, so `python -m wfield_local.*` works with no PYTHONPATH.
Preprocessing runs in the `wfield` conda env (imaging box); analysis runs in the `locanmf` env
(behavior-GPU box).

```powershell
conda activate wfield          # imaging box (preprocessing); the analysis box uses `locanmf`
```

## Running the pipeline (config-driven — current)

The §1–23 commands are the underlying steps; day-to-day they are orchestrated by:

- **Preprocessing (imaging box):**
  `python -m wfield_local.preprocess <DATE> ... [--only PS9x ...] [--dry-run]`
  — auto-discovers each date's raw sessions on `E:` (no per-date hard-coding), then per session runs
  motion(fixed) → SVD → cross-register to the animal's 6/6 reference → push LocaNMF inputs to
  MICROSCOPE, then the cue/lick/quiet **activity maps** (§4/5/8/9/12), the all-days cross-day QC
  overlay (`xall`, §14), and photobleach QC. Replaces the retired per-date `_nightly_*`/`_mc_svd_*`/
  `_maps_*`/`_photobleach_*` drivers.
- **Cross-session preprocessing deck:** `python -m wfield_local.preprocess_deck` — rebuilds the single
  `labcams/cross-session_preprocessing_<animal>.pptx` in place (grouped animal → figure type → date; split
  into per-animal files to bound size). Replaces the retired `PS92_94_95_affine8v1.pptx` builder.
- **Analysis (behavior-GPU box):**
  `python -m wfield_local.nightly_figs <DATE> ... [--only PS9x ...] [--from <DATE spec>]` — orchestrates
  the LocaNMF decode/encode/RSA (see **Analysis** §17–23 below) and builds the decoder summary deck.

Both CLIs share ONE date/animal knob grammar (`config.expand_dates` / `config.normalize_animals`):
a `<DATE>` token is `MMDD` or `YYYYMMDD`; give a space/comma list (`0806 0807`), an inclusive range
(`0806-0808`), or `all` (preprocess → every date-dir on `E:`; analysis → every registered session).
`--only` takes animals (`--only PS94 PS95`) or `all`; omitting it means all. Preprocess processes
each date; analysis builds per-day figures for each `<DATE>` (default: the latest registered session)
and the cross-session comparison spans `--from` (default: the curated set).

Register each new session in `configs/sessions.yaml`; per-animal metadata + date policy in
`configs/animals.yaml`; params in `configs/defaults.yaml`. Per-machine nightly runbooks: `runbooks/`.
Repo setup + architecture: root `README.md`; project rules + roadmap: `CLAUDE.md`.

## Preprocessing (imaging box)

The imaging-box preprocessing (env `wfield`) turns each raw labcams `.dat` into SVD + Allen-aligned
outputs and cue/lick activity maps. Conceptually:

- Motion-correct the labcams `.dat` (optionally QC it, §13).
- Run local SVD + dual-color hemodynamic correction.
- Apply Allen/wfield landmark alignment.
- Generate cue-aligned spout-position maps (relabeled sessions: §12) and lick-aligned post-event maps.
- Optional alignment diagnostics + comparison decks; align the same animal across days (§14).

> Decomposition stops at SVD + hemo + atlas here; PMD/LocaNMF are not run locally — those are the
> analysis half (§17–23). See the "Decomposition note" below and `DECISIONS.md`.

Large outputs stay in the recording folder, usually under:

```text
E:\labcams_data\YYYYMMDD\SESSION\motion_corrected
```

Steps 1–16 are the individual preprocessing commands.

### 1. Motion Correction

Example:

```powershell
python .\wfield_local\run_wfield_motion.py `
  "E:\labcams_data\20260601\PS95_20260601_153653\raw_widefield_data\pco_edge_run000_00000000_2_540_640_uint16.dat" `
  --output-dir "E:\labcams_data\20260601\PS95_20260601_153653\motion_corrected" `
  --mode 2d
```

Important adjustable parameters:

- `--mode 2d`: lean rigid XY correction. This is the default practical choice.
- `--mode ecc`: slower, less lean option that can estimate a richer transform. Try on a subset first.
- `--output-dir`: where corrected `.bin` and shift summaries are written.

The raw `.dat` file is not overwritten. (The script's flag is `--output`.)

#### TTL-based LED relabeling (trial-gated recordings)

For trial-gated recordings, run motion correction with `--daq-h5` so the DAT is
relabeled to DAQ-confirmed 415/470 pairs BEFORE motion correction and SVD. The
saved `.dat` channel split is only a frame-parity interpretation, which can drift
from the true LED across trials; relabeling from the DAQ `pco_exposure` /
`led415_ttl` / `led470_ttl` channels makes channel 0 = 415, channel 1 = 470
deterministically (and drops any dark inter-trial frames).

```powershell
python .\wfield_local\run_wfield_motion.py `
  "E:\labcams_data\...\raw_widefield_data\pco_edge_run000_00000000_2_H_W_uint16.dat" `
  --daq-h5 "E:\DAQ_recorder_output\<session>.h5" `
  --relabel-mode acquire-enable `
  --output "E:\labcams_data\...\motion_corrected"
```

This writes a `*_daq_led_cleanpairs_*.dat` (plus a frame map) into the output
folder and motion-corrects that. Use `--relabel-mode rescue` for older
continuously-saved (LED-gated) sessions that contain dark inter-trial frames.
The relabel can also be run standalone via
`python -m wfield_local.trim_illuminated_labcams <dat> <daq.h5> --output-dir <dir> --mode acquire-enable`.

### 2. SVD And Hemodynamic Correction

Example:

```powershell
python .\wfield_local\run_wfield_local.py `
  "E:\labcams_data\20260601\PS95_20260601_153653\motion_corrected\motioncorrect_2_540_640_uint16.bin" `
  --output-dir "E:\labcams_data\20260601\PS95_20260601_153653\motion_corrected\wfield_local_results" `
  --functional-channel 1 `
  --n-components 100
```

Important adjustable parameters:

- `--functional-channel 1`: for this rig, channel 0 is 415 nm and channel 1 is 470 nm, so 470 is the functional channel. (PS92 6/2 rescue was recorrected with `--functional-channel 0` after a swap.)
- `--n-components`: SVD component count. `100` has been used for PS94/PS95.
- Chunk/memory parameters in the script can be adjusted if a computer runs out of RAM.
- `--freq-highpass` / `--freq-lowpass`: hemo-correction filter cutoffs (defaults 0.1 / 14 Hz). The default 0.1 Hz highpass already removes the slow 415 nm LED drift, so it does not leak into ΔF/F.
- `--detrend-order N`: optional per-component polynomial detrend (per channel) applied before the β regression. Use with a lowered `--freq-highpass` to strip slow LED drift while keeping slow neural signal. Default off; the running pipeline behavior is unchanged.

> The activity maps are `U @ SVTcorr` averaged over an event window, in **fractional ΔF/F** relative to the **session-mean image** (`divide_by_average=True`), hemo-corrected and high-pass filtered. The cue figure's `post-pre` delta is the baselined evoked map; the lick maps are post-only (vs session mean), so treat them as activity relative to session mean unless a pre-lick baseline is added.

Outputs include:

- `U.npy`
- `SVT.npy`
- `SVTcorr.npy`
- `frames_average.npy`
- `rcoeffs.npy`
- `T.npy`

### 3. Allen Alignment

After making or revising `dorsal_cortex_landmarks.json` in the wfield/NeuroCAAS GUI, apply it locally:

```powershell
python .\wfield_local\apply_allen_transform.py `
  "E:\labcams_data\20260601\PS95_20260601_153653\motion_corrected\wfield_local_results" `
  --landmarks "E:\labcams_data\20260601\PS95_20260601_153653\raw_widefield_data\dorsal_cortex_landmarks.json" `
  --output "E:\labcams_data\20260601\PS95_20260601_153653\motion_corrected\wfield_local_results\allen_aligned_v6"
```

Use a versioned output folder, such as `allen_aligned_v6`, whenever you are comparing landmark attempts.

Important note:

- `wfield` stores a transform that maps reference/atlas landmark coordinates to clicked image coordinates.
- Image warping uses that transform through `skimage.warp`, which treats the transform as an output-to-input inverse map.
- The helper diagnostic scripts account for this when plotting where clicked points land after warping.

### 4. Cue-Aligned Spout-Position Averages

By default, DAQ events are aligned to imaging frames using the DAQ-recorded
`pco_exposure` rising edges. This is more robust than wall-clock timestamps:
the cue sample is mapped to the nearest PCO exposure pulse index, then divided
by two because each 415/470 raw frame pair becomes one hemodynamic-corrected
timepoint in `SVTcorr.npy`. Passing `--camlog` is still useful because the
summary JSON records frame-count QC against the labcams camlog.

Example:

```powershell
python .\wfield_local\plot_spout_trial_averages.py `
  --label PS95_v6 `
  --daq-h5 "E:\DAQ_recorder_output\PS95_baseline_20260601_153627.h5" `
  --wfield-results "E:\labcams_data\20260601\PS95_20260601_153653\motion_corrected\wfield_local_results" `
  --allen-dir "E:\labcams_data\20260601\PS95_20260601_153653\motion_corrected\wfield_local_results\allen_aligned_v6" `
  --camlog "E:\labcams_data\20260601\PS95_20260601_153653\pco_edge_run000_00000000.camlog" `
  --output "E:\labcams_data\20260601\PS95_20260601_153653\motion_corrected\spout_trial_averages_allen_v6" `
  --frame-align pco `
  --pre-s 1.0 `
  --post-s 1.0 `
  --fs 31.23
```

Adjustable parameters:

- `--pre-s`: seconds before cue.
- `--post-s`: seconds after cue.
- `--fs`: hemodynamic-corrected paired-frame sampling rate. For PS94/PS95 this was `31.23`.
- `--frame-align pco`: use DAQ `pco_exposure` pulse order. Use `camlog` only for legacy wall-clock reproduction.
- `--activity-percentile`: display scaling percentile for pre/post panels.

Spout position is assigned using the most recent `spout_strobe` before each cue:

```text
code = spout_bit0 + 2*spout_bit1 + 4*spout_bit2
```

### 5. Shared-Scale Spout Figures

The original cue-aligned plot uses one scale for pre/post and a separate scale for post-minus-pre. To make pre, post, and delta visually comparable, regenerate a shared-scale figure:

```powershell
python .\wfield_local\plot_spout_trial_averages_shared_scale.py `
  --label PS95_v6 `
  --trial-maps "E:\labcams_data\20260601\PS95_20260601_153653\motion_corrected\spout_trial_averages_allen_v6\PS95_v6_spout_positions_1s_pre_post_delta_maps.npz" `
  --allen-dir "E:\labcams_data\20260601\PS95_20260601_153653\motion_corrected\wfield_local_results\allen_aligned_v6" `
  --summary "E:\labcams_data\20260601\PS95_20260601_153653\motion_corrected\spout_trial_averages_allen_v6\PS95_v6_spout_positions_1s_pre_post_delta_summary.json" `
  --output "E:\labcams_data\20260601\PS95_20260601_153653\motion_corrected\spout_trial_averages_allen_v6"
```

### 6. Alignment Diagnostics

Plot clicked landmark points before and after the transform:

```powershell
python .\wfield_local\plot_alignment_before_after.py `
  --label PS95 `
  --json-dir "E:\labcams_data\20260601\PS95_20260601_153653\raw_widefield_data" `
  --results "E:\labcams_data\20260601\PS95_20260601_153653\motion_corrected\wfield_local_results" `
  --output "E:\labcams_data\20260601\PS95_20260601_153653\motion_corrected\alignment_before_after" `
  --current-version v6
```

Plot mean 470 nm with Allen outlines and landmark points:

```powershell
python .\wfield_local\plot_alignment_landmark_overlays.py `
  --label PS95 `
  --json-dir "E:\labcams_data\20260601\PS95_20260601_153653\raw_widefield_data" `
  --results "E:\labcams_data\20260601\PS95_20260601_153653\motion_corrected\wfield_local_results" `
  --output "E:\labcams_data\20260601\PS95_20260601_153653\motion_corrected\alignment_landmark_overlays" `
  --current-version v6
```

Plot the fixed Allen/wfield reference landmarks:

```powershell
python .\wfield_local\plot_allen_reference_landmarks.py `
  --landmarks "E:\labcams_data\20260601\PS95_20260601_153653\raw_widefield_data\dorsal_cortex_landmarks.json" `
  --output "E:\labcams_data\20260601\allen_wfield_reference_landmark_targets.png"
```

Plot a color-coded Allen ROI label map:

```powershell
python .\wfield_local\plot_allen_roi_labels.py `
  --allen-dir "E:\labcams_data\20260601\PS95_20260601_153653\motion_corrected\wfield_local_results\allen_aligned_v6" `
  --output "E:\labcams_data\20260601\allen_wfield_roi_labels_ps95_v6.png" `
  --title "Allen/wfield ROI labels from PS95 v6 atlas"
```

### 7. Alignment Comparison PowerPoint

Build a comparison deck from the generated PNGs:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File `
  "C:\Github\Widefield_DAQ_recorder\wfield_local\build_alignment_comparison_ppt.ps1" `
  -OutputPath "E:\labcams_data\20260601\alignment_comparison_PS94_PS95_shared_scale.pptx"
```

This uses local PowerPoint COM automation. PowerPoint must be installed on the machine.

### 8. Post-Lick Averages

Post-lick averages use analog `lick_analog` falling threshold crossings. No pre-lick baseline is used by default, because lick bouts can make pre-lick windows hard to interpret.

Like cue-aligned maps, lick events default to `--frame-align pco`, so lick
sample indices are mapped through DAQ-recorded PCO exposure pulses rather than
labcams wall-clock timestamps. The optional `--camlog` path adds frame-count QC
to the output summary.

Example:

```powershell
python .\wfield_local\plot_lick_aligned_averages.py `
  --label PS95_v6 `
  --daq-h5 "E:\DAQ_recorder_output\PS95_baseline_20260601_153627.h5" `
  --wfield-results "E:\labcams_data\20260601\PS95_20260601_153653\motion_corrected\wfield_local_results" `
  --allen-dir "E:\labcams_data\20260601\PS95_20260601_153653\motion_corrected\wfield_local_results\allen_aligned_v6" `
  --camlog "E:\labcams_data\20260601\PS95_20260601_153653\pco_edge_run000_00000000.camlog" `
  --output "E:\labcams_data\20260601\PS95_20260601_153653\motion_corrected\lick_aligned_v6" `
  --frame-align pco `
  --lick-thresh-upper-v 2.5 `
  --lick-thresh-lower-v 1.0 `
  --refractory-s 0.10 `
  --post-s 0.150 `
  --fs 31.23
```

Adjustable parameters:

- `--lick-thresh-upper-v`: lick onset threshold. For PS95, licks dropped from ~5.5 V to ~0 V, so `2.5` V is a reasonable default.
- `--lick-thresh-lower-v`: lick offset threshold for hysteresis. `1.0` V has been used for PS94/PS95.
- `--refractory-s`: minimum separation between lick events. `0.10` s avoids counting one lick as many threshold crossings.
- `--post-s`: post-lick window duration. `0.150` s was requested for PS95.
- `--fs`: paired-frame hemodynamic-corrected sampling rate.

Outputs:

- `*_lick_aligned_150ms_post_by_spout.png`
- `*_lick_aligned_150ms_post_by_spout_maps.npz`
- `*_lick_aligned_150ms_post_by_spout_summary.json`

The lick detector is in `lick_detection.py`. It ports the double-threshold
hysteresis + lockout logic from the stroke/orofacial pipeline:

- onset: voltage crosses below `thresh_upper`
- offset: voltage crosses above `thresh_lower`
- cleanup: drop onset events inside a post-offset lockout window
- optional refractory: collapse dense lick bouts for imaging-triggered averages

### 9. Cue vs Lick Spout-Position Comparisons

Compare cue-aligned and lick-aligned maps for the same aligned session:

```powershell
python .\wfield_local\plot_lick_vs_cue_spout_maps.py `
  --label PS95_v6 `
  --cue-maps "E:\labcams_data\20260601\PS95_20260601_153653\motion_corrected\spout_trial_averages_allen_v6\PS95_v6_spout_positions_1s_pre_post_delta_maps.npz" `
  --lick-maps "E:\labcams_data\20260601\PS95_20260601_153653\motion_corrected\lick_aligned_v6\PS95_v6_lick_aligned_150ms_post_by_spout_maps.npz" `
  --allen-dir "E:\labcams_data\20260601\PS95_20260601_153653\motion_corrected\wfield_local_results\allen_aligned_v6" `
  --cue-summary "E:\labcams_data\20260601\PS95_20260601_153653\motion_corrected\spout_trial_averages_allen_v6\PS95_v6_spout_positions_1s_pre_post_delta_summary.json" `
  --lick-summary "E:\labcams_data\20260601\PS95_20260601_153653\motion_corrected\lick_aligned_v6\PS95_v6_lick_aligned_150ms_post_by_spout_summary.json" `
  --output "E:\labcams_data\20260601\PS95_20260601_153653\motion_corrected\lick_aligned_v6"
```

The figure columns are:

- cue-aligned post map
- lick-aligned post map
- `lick - cue`

By default the labels are `1 s post-cue` and `150 ms post-lick`; these are not identical windows, so interpret the third column as a descriptive contrast rather than a pure event-type subtraction.

### 10. Sync-Pulse Timebase Alignment

`frame_sync.py` ports the sync-pulse alignment algorithm from the
stroke/orofacial pipeline. Use it when you need to align DAQ/WaveSurfer-like
sync channels to camera timestamp CSVs without relying only on wall-clock
timestamps.

Main entry point:

```python
from wfield_local.frame_sync import make_alignment_template

template = make_alignment_template(
    ws_signals={
        "Sync_signal": sync_trace,
        "sample_rate_input": 5000.0,
    },
    cam_csv=cam_timestamp_dataframe,
    params={
        "csv_idx_sync": 0,
        "csv_idx_time": 1,
        "key_sync": "Sync_signal",
        "fps_cam": 62.46,
        "window": 20,
        "p": 0.1,
        "min_matched_edges": 5,
        "edge_threshold": 0.5,
    },
)
```

Core outputs:

- `sig_camIdx__idx_ws`: for each DAQ/WaveSurfer sample, estimated camera frame index
- `sig_wsIdx__idx_cam`: for each camera frame, estimated DAQ/WaveSurfer sample index
- affine fit parameters and matched edge diagnostics

Save templates with:

```python
import numpy as np
np.savez_compressed("alignment_template.npz", **template)
```

### 11. Treadmill Velocity And Running Bout QC

`treadmill.py` ports the treadmill pieces from the stroke/orofacial pipeline:

```text
raw voltage -> calibrated speed in mm/s -> Gaussian smoothing -> running bout mask
```

Calibration uses:

```text
speed_mm_s = (voltage - offset_v) * (1 / volt_sec_per_rot) * mm_per_rot
```

Run a QC plot from a DAQ recorder file:

```powershell
python .\wfield_local\plot_treadmill_running_qc.py `
  --label PS95 `
  --daq-h5 "E:\DAQ_recorder_output\PS95_baseline_20260601_153627.h5" `
  --output "E:\labcams_data\20260601\PS95_20260601_153653\motion_corrected\treadmill_qc" `
  --channel treadmill `
  --smoothing-sigma-s 0.15 `
  --thresh-speed 5.0 `
  --max-gap-duration 0.3 `
  --min-duration 2.0
```

Important parameters:

- `--offset-v`: voltage zero offset. Default is the legacy cohort value `1.2587643276652853`.
- `--volt-sec-per-rot`: encoder calibration constant. Default is the legacy cohort value `0.382`.
- `--mm-per-rot`: wheel circumference. Default is the legacy cohort value `29.25`.
- `--smoothing-sigma-s`: Gaussian smoothing sigma in seconds. Default is `0.15`.
- `--thresh-speed`: speed threshold in mm/s for running. Default is `5.0`.
- `--max-gap-duration`: fill below-threshold gaps shorter than this duration. Default is `0.3`.
- `--min-duration`: discard bouts shorter than this duration. Default is `2.0`.

Default legacy cohort constants:

```text
OFFSET_IN_VOLTS = 1.2587643276652853
VOLT_SEC_PER_ROT = 0.382
MM_PER_ROT = 29.25
SMOOTHING_SIGMA_SEC = 0.15
THRESH_SPEED_MM_PER_S = 5.0
MAX_GAP_DURATION_SEC = 0.3
MIN_DURATION_SEC = 2.0
```

Outputs:

- `*_treadmill_running_bout_overview.png`
- `*_treadmill_running_not_running_examples.png`
- `*_treadmill_running_bouts.npz`
- `*_treadmill_running_bout_summary.json`

After QC looks sensible, generate **quiet-vs-running** SVD activity maps. These consume the CANONICAL
behavior events (`wfield_local.behavior_events`, the shared licks/reward/running/quiet identity) rather
than re-detecting movement, so the map is consistent with the behavior figures. The preprocessing
orchestrator (`preprocess`) runs this automatically after the lick maps (it first emits the events, then
the maps), and the deck shows it before the QC/photobleach slides. Standalone:

```powershell
python -m wfield_local.plot_running_activity_maps `
  --label PS92_0806_affine8v1 `
  --events "M:\...\Behavior_logs\Widefield\behavior_summary\events\PS92\20260806.npz" `
  --wfield-results "...\motion_corrected\wfield_local_results" `
  --allen-dir "...\motion_corrected\wfield_local_results\allen_aligned_affine8v1" `
  --daq-h5 "M:\...\DAQ_recorder_output\20260806\PS92_20260806_124733.h5" `
  --frame-map "...\motion_corrected\*_cleanpairs_frame_map.npz" `
  --cleanpairs-summary "...\motion_corrected\*_cleanpairs_summary.json" `
  --output "...\motion_corrected\running_activity_affine8v1"
```

It maps corrected imaging frames to DAQ samples through the cleanpairs `frame_map` + `pco_exposure` pulse
train (same mapping the cue/lick maps use), classifies each frame as quiet/running from the event bouts,
and averages `SVTcorr` frames per state → quiet, running, and running−quiet maps (RdBu, Allen overlay).

### 12. Relabeled (cleanpairs) cue/lick maps

For trial-gated recordings relabeled with `--relabel-mode rescue`, the corrected
movie is a non-contiguous subset of kept 415/470 pairs, so the stock plotters'
`raw//2` event→frame mapping is wrong. `framemap_event_maps.py` maps each event to
the nearest kept corrected frame via the cleanpairs frame map + DAQ `pco_exposure`
pulses (`chosen_exposure_offset` read from the `*_cleanpairs_summary.json`), and
writes the **same filenames/npz keys** as the stock plotters, so the downstream
`plot_spout_position_contrasts`, `plot_lick_position_contrasts`, and
`plot_lick_vs_cue_spout_maps` steps run unchanged.

```powershell
python -m wfield_local.framemap_event_maps --what cue `
  --daq-h5 <session.h5> --wfield-results <...\wfield_local_results> `
  --allen-dir <...\allen_aligned_*> --frame-map <...\*_cleanpairs_frame_map.npz> `
  --cleanpairs-summary <...\*_cleanpairs_summary.json> --output <...\spout_trial_averages_*> --label <LABEL>
# --what lick  (add --post-s 0.15) for the post-lick maps
```

Full-FOV (non-relabeled) sessions still use the stock `plot_spout_trial_averages` /
`plot_lick_aligned_averages` with `--frame-align pco`.

### 13. Motion-Correction QC

`qc_motion_correction.py` reads the saved per-frame shifts and the pre/post movies
and emits one QC figure (shift traces + magnitude histogram, mean-image sharpness
raw vs corrected, corrected temporal-std residual-motion image) plus a pass/warn
JSON. Verdict comes from the shift distribution; the sharpness ratio only downgrades
when there was real motion to remove (sub-pixel sessions are not penalized for warp
interpolation softening).

```powershell
python -m wfield_local.qc_motion_correction `
  --motion-dir "E:\labcams_data\...\motion_corrected" --label <LABEL> `
  --output "E:\labcams_data\...\motion_corrected\motion_qc"
```

### 14. Cross-Day Alignment (within animal)

Register each day's motion-corrected **mean 470 nm vasculature** to one reference
session so all days share the reference/CCF frame (`cross_day_align.py`). The
reference is CCF-aligned by its landmarks; other days are landmark-initialized then
refined by intensity-based ECC affine (SIFT+RANSAC fallback), with a greedy
keep-best on masked NCC so refinement never worsens the init. Outputs a red/green
vessel-overlay QC + NCC table + per-session transforms; `"warp_u": true` also warps
each day's `U` into the common frame.

```powershell
python -m wfield_local.cross_day_align config.json
```

Config (one per animal): `{"animal","func_channel","reference","output","warp_u",
"sessions":{"<id>":{"results":"...wfield_local_results","landmarks":"...v1.json",
"func_channel":<optional override>}, ...}}`. Keep the same ROI/zoom across days
(full-FOV↔ROI pairs register poorly). Across **animals**, vasculature is not shared —
use CCF/landmarks + Allen-area (ROI) or LocaNMF-component comparison instead.

### 15. ROI-based (Allen-area) activity extraction

`roi_activity.py` is a lightweight CPU baseline alongside LocaNMF: it averages the
Allen-aligned `U` over each atlas region to get a region x time trace
(`U_bar @ SVTcorr`, no pixel reconstruction), and optionally aligns to cue/lick
events by spout position. Runs in the wfield CPU env (numpy + h5py; no torch/GPU).
The wfield atlas is already lateralized (`MOp_left` / `MOp_right`), so you get one
trace per area per hemisphere — useful for stroke laterality.

```powershell
# region traces only
python -m wfield_local.roi_activity --allen-dir <...\allen_aligned_affine8v1> `
  --label PS94_0603 --output <...\roi_activity_affine8v1>
# + cue/lick per-region responses by spout position (regime B: pass --frame-map + --cleanpairs-summary)
python -m wfield_local.roi_activity --allen-dir <...> --label PS94_0603 --output <...> `
  --daq-h5 <session.h5> --what both `
  --frame-map <...\*_cleanpairs_frame_map.npz> --cleanpairs-summary <...\*_cleanpairs_summary.json>
```

Outputs: `*_roi_traces.npy` (R x T) + meta, `*_{cue,lick}_roi_by_position.npz`
(regions x positions: post + delta) + named-region heatmaps (cue and lick use the
same region names/order), `*_allen_reference_labeled.png` (all regions colored +
labeled at their centroids — the key for reading the heatmaps), and a
`*_roi_overview.png` that shows the locations + traces of `--overview-regions`
(default `MOp_left,MOp_right,MOs_left,MOs_right`). This is
the simple "one signal per area" baseline; `run_locanmf.py` gives the denoised,
region-anchored, multi-component version. Use ROI traces as a fast cross-check and
for quick per-area trial stats; use LocaNMF components for cross-animal claims.

### 16. Quiet-period detection (baseline selection)

`quiet_periods.py` builds a per-sample and per-corrected-frame "quiet" mask (animal
not running and not licking, not in a peri-reward window) for behavior-controlled
baseline (F0) selection — useful because trial-triggered acquisition records no true
inter-trial rest. Ported from the stroke pipeline's `find_quiet_bouts`
(`quiet = slow-treadmill AND not-near-lick AND not-near-reward`, buffered) and adapted
for ONE spout. Reuses the ported `treadmill` + `lick_detection` helpers; CPU env.

```powershell
python -m wfield_local.quiet_periods --daq-h5 <session.h5> --label PS94_0603 `
  --output <...\quiet_affine8v1> `
  --frame-map <...\*_cleanpairs_frame_map.npz> --cleanpairs-summary <...\*_cleanpairs_summary.json>
```
Outputs `*_quiet_sample.npy` (DAQ-rate bool), `*_quiet_frame.npy` (per corrected
frame), a summary, and a QC plot (speed + lick + quiet shading). Intersect
`quiet_frame` with the pre-cue ENL window, or pool it, to define F0.

**Quiet-normalized lick maps:** pass `*_quiet_frame.npy` as `--quiet-frame` to
`plot_lick_aligned_averages.py`, `framemap_event_maps.py` (`--what lick`), or
`roi_activity.py`. Each then also emits a quiet-normalized output
(`*_quietnorm*`) = the post-lick map/per-region value **minus the mean quiet-period
map** (lick activity relative to the not-running/not-licking baseline), alongside the
raw version. The shared helper is `quiet_periods.quiet_baseline_svt` (mean SVT over
quiet frames).

- **Grooming is OFF by default.** The stroke pipeline detects grooming via *bilateral*
  two-spout conjunction, which doesn't apply here. Single-spout "long-touch" is the
  only proxy, but a TRUE long lick at close spouts also looks long, so it is
  unreliable — enable only experimentally with `--grooming`.
- **TUNE LATER:** running/quiet speed, min durations, and lick/reward/treadmill time
  buffers are stroke-pipeline starting points (e.g. the 8 s reward buffer is generous
  for short ENL windows). Revisit per rig/task, ideally validated against
  DLC/FaceRhythm movement once available (the future movement regressor).

### NeuroCAAS Compatibility Launcher

If `wfield ncaas` crashes during upload with `QProgressBar.setValue` receiving a `numpy.float64`, launch through:

```powershell
python .\wfield_local\wfield_ncaas_fixed.py "E:\labcams_data\20260601\PS94_20260601_141614"
```

This launcher also supports AWS session tokens in the local credentials file.

### End-of-day archival + cleanup (`archive_day.py`)

Reusable daily off-load of a recording day from the local E: drive:

- **Raw camera movies → M: standby** (cold, immutable originals)
- **Motion-corrected `.bin` → M: standby** under `<date>\<animal>\motion_corrected\`
  (huge; kept OFF MICROSCOPE to save space — LocaNMF doesn't need it)
- **SVD + Allen transform + maps + QC + DAQ → N: MICROSCOPE**
- Once copies are **size-verified**, the copied E: files plus the reproducible
  E:-only intermediates (cleanpairs `*.dat`, any `*_concat` raw) can be deleted
  from E: to reclaim space.

Nothing is hardcoded — pass `--date YYYYMMDD` and it walks
`E:\labcams_data\<date>`, mirrors the tree to M:/N:, and re-verifies each
destination before any deletion.

```powershell
python -m wfield_local.archive_day archive --date 20260604   # copy E -> M/N (LocaNMF inputs first)
python -m wfield_local.archive_day verify  --date 20260604   # confirm every E file is copied
python -m wfield_local.archive_day clean   --date 20260604   # DRY-RUN: show what would be deleted
python -m wfield_local.archive_day clean   --date 20260604 --execute   # actually delete from E
```

`clean` deletes an E: file only after re-confirming its destination size matches,
and removes a reproducible intermediate only once its regeneration sources (the
session's raw on M:, a DAQ h5 for the date on N:) are confirmed archived. Drive
roots default to this rig's mounts; override with `--m-raw`/`--n-lab`/etc.
(One-off per-date archive/cleanup drivers were superseded by this and removed.)

## Decomposition note: SVD vs LocaNMF

This local pipeline stops at **SVD + hemodynamic correction + Allen alignment**. It
does **not** run **PMD** denoising or **LocaNMF** (localized semi-NMF), which the
wfield/NeuroCAAS protocol adds after SVD to produce anatomically-localized,
cross-session/animal-reproducible components. For evoked maps and within-animal work,
SVD + atlas is adequate; for cross-animal / functional-subnetwork analysis, add
LocaNMF (runs on the existing low-rank `U`/`SVTcorr` + the `allen_area_atlas`). It is
GPU-oriented and not installed in the `wfield` (imaging-box) env. In `widefield_pipeline`,
LocaNMF + the decode/encode/RSA analysis run on the **behavior-GPU box** (`locanmf` env) — see the
**Analysis** section (§17–23) below, `runbooks/analysis_computer_nightly.md`, and `DECISIONS.md`.

## Analysis (behavior-GPU box)

The preprocessing above stops at SVD + atlas. The **analysis** half runs GPU LocaNMF and the
spout-position models on the behavior-GPU box (env `locanmf`). `nightly_figs` runs steps 17–22 for
the per-day date(s) + the cross-session span and builds the refined analysis deck at the `labcams` top
level (`spout_position_analysis_summary.pptx`, curated animal→type→date via `locanmf_analysis_deck.py`;
the old `spout_position_decoder_summary.pptx` / `spout_position_decoder_xsession_6on.pptx` builders are
retired, kept as `LEGACY_*` and no longer updated); run any step standalone with
the same `configs/` source of truth. In the commands below `<OUT>` is the figure/deck dir (default
`C:/Users/sabatini/source/cue_lick`), `<DATE>` is `MMDD`, `<SPAN>` is a comma `MMDD` list, and the
cross-session `<TAG>` is `<first>-<last>` of the span.

### Camera & behavior nightly (analysis box, no imaging needed)

`python -m wfield_local.camera_nightly <YYYYMMDD>` runs the camera/behavior side (it needs no imaging
output, so it runs first, while the imaging box is still preprocessing): upload `D:` → MICROSCOPE
(camera videos/CSVs + behavior logs, size-verified, never deletes `D:`) → **dropped-frame QC**
(`dropframe_qc`, gaps in the monotonic frame_id) → **camera↔DAQ alignment templates** (`camera_sync`,
GPIO sync train ↔ DAQ `sync` line) → **spout behavior figures** (`spout_behavior`). `--skip-copy` /
`--skip-dropframe` / `--skip-align` / `--skip-behavior` subset it; `nightly` calls it before `nightly_figs`.

`python -m wfield_local.spout_behavior <YYYYMMDD> [--cohort] [--from curated]` — from the task-controller's
scored `trials.csv` + `events.csv` (1 cue + 6 spout positions: close/far × L/center/R), per session it writes
(1) an **accuracy** figure (2×3 spatial hit-rate grid, engagement timeline, per-position bars, latency) and
(2) a **lick-microstructure** figure (peri-cue raster + PSTH, ILI distribution, lick bouts, per-position lick
rate / licks-per-trial / anticipatory pre-cue licks, and a GUI-vs-DAQ-pipeline lick-count comparison), under
`behavior_summary/sessions/<animal>/<date>/`. `--cohort` adds a cross-animal cohort figure and a per-animal
`cohort/by_animal/<animal>_across_sessions.png` tracking every per-position metric over days (color=ring,
marker=side). Accuracy is **engaged-gated**: reward auto-holds after a miss run, so a sated animal's late
misses are disengagement, not spatial inaccuracy — a terminal sated-tail + rolling-collapse gate
(`configs/defaults.yaml behavior.*`) excludes them, raw shown alongside. The DAQ comparison applies the
`lick_detection.min_ili_ms` 40 ms physiological floor (see below).

### 17. LocaNMF decomposition

Atlas-anchored components (r²=0.95, loc=80, maxrank=20) from `SVTcorr` + the Allen-aligned `U`. One
session, or a manifest of many:

```powershell
python -m wfield_local.run_locanmf --allen-dir <...>/allen_aligned_affine8v1 --output <...>/locanmf --label PS95_0807
python -m wfield_local.batch_locanmf --manifest <manifest.json>   # JSON list of {allen_dir,label,output[,svt]}
```

### 18. Position decoder

Multinomial logistic regression of spout position from individual components (first-lick / cue /
pre-cue windows, block-CV by ~6-trial position blocks, no per-trial baseline; chance 0.167). SSp
dominates, MO secondary:

```powershell
python -m wfield_local.locanmf_position_decoder --date <DATE> --align lick   --post-s 2.0 --output <OUT>
python -m wfield_local.locanmf_position_decoder --date <DATE> --align cue    --post-s 2.0 --output <OUT>
python -m wfield_local.locanmf_position_decoder --date <DATE> --align precue --post-s 1.0 --output <OUT>
```

### 19. Decoder-weight & dynamics figures

Rolling cue, first-lick temporal dynamics, laterality, and top-component maps (`nightly_figs` inlines
these as function calls; standalone entry point):

```powershell
python -m wfield_local.locanmf_decoder_weights --output <OUT> --weights-day <DATE>
```

### 20. Position encoder

Ridge position→activity + fraction-explainable-variance (FEVE) vs a noise ceiling, pooled over the
cross-session span:

```powershell
python -m wfield_local.locanmf_position_encoder --date <DATE> --pool-dates <SPAN> --output <OUT>
```

### 21. Cross-mouse consistency

Within-/across-animal per-position consistency over the whole span (run once):

```powershell
python -m wfield_local.locanmf_cross_mouse --output <OUT> --dates <SPAN> --tag <TAG>
```

### 22. RSA

RDM / **crossnobis** (noise-unbiased) representational similarity over the whole span (run once):

```powershell
python -m wfield_local.locanmf_rsa --output <OUT> --dates <SPAN> --tag <TAG>
```

### 23. Frozen decoder (post-stroke plan)

Fit + persist the pre-stroke decoders, or the cross-day transfer demo:

```powershell
python -m wfield_local.locanmf_frozen_decoder --save --output <OUT>       # persist baseline decoders
python -m wfield_local.locanmf_frozen_decoder --transfer --output <OUT>   # cross-day transfer figure
```

Subset knobs: `nightly_figs --only PS93` scopes every decode/encode/RSA subprocess via the
`WIDEFIELD_ONLY_ANIMALS` env var + the in-process figs; any single module can be scoped directly by
prefixing `WIDEFIELD_ONLY_ANIMALS=PS93` / `WIDEFIELD_ONLY_DATES=0807`. Full decisions/findings:
`LOCANMF_LICK_CUE_ANALYSIS.md`, `runbooks/analysis_computer_nightly.md`, `DECISIONS.md`.
