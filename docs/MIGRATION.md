# Migration: pipeline carved out of `Widefield_DAQ_recorder` → `widefield_pipeline`

Date: 2026-08-08. The widefield preprocessing + analysis code (`wfield_local/` package + the `_*_run.py`
/ `_nightly_*` / `_maps_*` / `_xall_*` drivers + `_xday_*.json` state + the pipeline docs) was split out of
`Widefield_DAQ_recorder` into this repo. `Widefield_DAQ_recorder` keeps ONLY the DAQ recorder GUI + camera
acquisition (`daq_recorder/`, `labcams/`, `labcams_ps/`, `arduino/`, `tools/`). Zero cross-imports.

## Analysis / behavior GPU box (this computer) — DONE 2026-08-08
- Cloned `widefield_pipeline` to `C:\Users\sabatini\GitHub\widefield_pipeline`.
- `pip install -e .` into the `locanmf` env → `wfield_local` resolves to the new repo (no PYTHONPATH hack).
- Going forward, all analysis work happens in `widefield_pipeline`; the old `Widefield_DAQ_recorder` clone is
  redundant here (GUI is not used on this box) and can be removed once references are fully migrated.

## Imaging (PCO) computer — TODO (prompt to paste there)

> [MIGRATION] The widefield preprocessing + analysis code has been split out of `Widefield_DAQ_recorder`
> into a new repo, **widefield_pipeline** (https://github.com/priya-srikanth/widefield_pipeline).
> `Widefield_DAQ_recorder` now keeps ONLY the DAQ recorder GUI + camera acquisition, which this machine
> still uses for recording. Please migrate this machine's PREPROCESSING to the new repo:
> 1. Clone it next to the recorder: `cd C:\Github && git clone https://github.com/priya-srikanth/widefield_pipeline.git`
> 2. Install into the wfield env (editable, so `python -m wfield_local.*` works with no PYTHONPATH):
>    `C:\ProgramData\anaconda3\envs\wfield\python.exe -m pip install -e C:\Github\widefield_pipeline`
> 3. Run the nightly PREPROCESSING from the new clone from now on: `cd C:\Github\widefield_pipeline`, then
>    `_nightly_<DATE>.py`, `_mc_svd_*`, `_maps_*`, `_xall_refresh.py`, etc. (they and the `_xday_*.json`
>    state moved with the package). Runbook: `runbooks/imaging_computer_nightly.md`.
> 4. Keep using `Widefield_DAQ_recorder` for the DAQ recorder GUI + labcams acquisition (unchanged).
> 5. Dry-run: confirm a small preprocessing step runs from the new clone (e.g. `python -m wfield_local.run_wfield_motion --help`).
> 6. Tell Priya once verified — then the duplicated pipeline code gets removed from `Widefield_DAQ_recorder`.
>
> Notes: standby path is `M:\collaborations\Priya\Widefield\labcams` (NOT `M:\Widefield`). The
> imaging→analysis contract is unchanged: still write `motion_corrected/`, `wfield_local_results/` (incl.
> `allen_aligned_affine8v1/` + the `*cleanpairs_frame_map.npz`), and DAQ h5 to MICROSCOPE for the GPU box.

## Final step (only after BOTH machines confirm migration)
1. In `Widefield_DAQ_recorder`: `git rm` the pipeline half (`wfield_local/`, the pipeline `_*` drivers/state,
   the pipeline `*.md` docs), leaving the recorder GUI. Commit + push. (Recorder history retains it.)
2. Remove the redundant local `Widefield_DAQ_recorder` clone from the GPU box.
3. Update any lingering path references (cron prompts, memory) from `Widefield_DAQ_recorder` → `widefield_pipeline`.
