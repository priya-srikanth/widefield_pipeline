# Nightly runbook — IMAGING (PCO microscope) computer

Mounts here: **`N:` = MICROSCOPE** (`\\research.files.med.harvard.edu\Neurobio`), **`E:` = local
raw/DAQ**, **`M:` = standby** (`\\standby...\sabatini`, under `collaborations\Priya\Widefield\labcams`).
Env: `C:\ProgramData\anaconda3\envs\wfield`. Full detail (params, paths, hard rules, cleanup):
[`../NIGHTLY_PIPELINE.md`](../NIGHTLY_PIPELINE.md); underlying per-step commands: `../wfield_local/README.md`.

## The whole night

```powershell
conda activate wfield
git -C C:\Github\widefield_pipeline pull
python -m wfield_local.nightly <YYYYMMDD>         # DAQ upload -> preprocess -> deck -> archive COPY (+verify)
```

`nightly` auto-detects this is the imaging box and chains **four stages, in this order**:

1. **`archive_day upload-daq`** — push the day's DAQ `.h5` to `N:` FIRST, so the analysis/GPU box can start
   its behavior pipeline (behavior events + spout figures) while this box does the heavy SVD/LocaNMF work.
   *(skip: `--skip-daq-upload`)*
2. **`preprocess`** — the heavy stage. Auto-discovers the date's raw sessions on `E:`, then **per session in
   animal order**:
   a. **motion correction** (sign-fixed, 2D) → `motioncorrect_*.bin` *(auto-skips if the bin already exists)*
   b. **SVD** (functional channel 470) → `wfield_local_results/SVTcorr.npy` *(auto-skips if it exists)*
   c. **cross-register** to that animal's 6/6 reference → emits `allen_aligned_affine8v1`
   d. **push LocaNMF inputs to `N:` first** (results dir + `*cleanpairs_frame_map.npz` + summary; NOT the
      `.bin`) — so the GPU box gets the decode inputs as early as possible.

   then, **per date** (after that date's sessions):
   e. **activity maps** — the cue/lick spout-position maps + contrasts, quiet-period frame, **canonical
      behavior events**, and the **quiet-vs-running SVD activity maps** *(skip: `--skip-maps`)*
   f. **photobleach QC** over the date's sessions *(skip: `--skip-photobleach`)*

   then, **once after all dates**:
   g. **xall** — the all-days cross-day QC overlay for every animal processed *(skip: `--skip-xall`)*
   h. **cross-day raw ROI intensity** trend *(skip: `--skip-crossday-intensity`)*

   *(`--skip-preprocess` skips a–d — motion/SVD/xreg/push — and re-runs only the downstream steps e–h on a
   date whose processed outputs are already on `N:`.)*
3. **`preprocess_deck`** — rebuild `cross-session_preprocessing_<animal>.pptx` in place (includes the
   quiet-vs-running slide before QC). *(skip: `--skip-deck`)*
4. **`archive_day archive` + `verify`** — raw + `.bin` → `M:` standby, outputs → `N:`, size-verified.
   It **never deletes `E:`**. *(skip: `--skip-archive`)*

Then, **after checking in that N:/M: copies are byte-verified**, reclaim E: space (MANUAL — never automatic):

```powershell
python -m wfield_local.archive_day clean --date <YYYYMMDD>              # DRY-RUN
python -m wfield_local.archive_day clean --date <YYYYMMDD> --execute    # actually delete from E:
```

## Running only part of the pipeline

**Whole-stage skips on the top-level `nightly`:** `--skip-daq-upload` (stage 1), `--skip-deck` (stage 3),
`--skip-archive` (stage 4), and `--dry-run` (plan only, no writes). `nightly` always runs `preprocess`
(stage 2) — it forwards `--only` / `--dry-run` but **not** `preprocess`'s per-step skips. For finer control,
run the sub-command standalone:

```powershell
python -m wfield_local.preprocess <DATE>      # stage 2 alone; per-step skips below
#   --skip-preprocess  (skip motion/SVD/xreg/push; re-run only downstream maps/xall/etc.)
#   --skip-maps  --skip-photobleach  --skip-xall  --skip-crossday-intensity   --dry-run
python -m wfield_local.preprocess_deck                                  # stage 3 only
python -m wfield_local.archive_day upload-daq --date <DATE>             # stage 1 only
python -m wfield_local.archive_day archive    --date <DATE>            # stage 4 (copy) …
python -m wfield_local.archive_day verify     --date <DATE>            # … then verify
```

## Selecting animals and dates

- **Animals — `--only`:** `--only PS94` (or `--only PS94 PS95`, or `all` = no filter). Scopes the
  **`preprocess`** stage (discovery + motion/SVD/maps/photobleach). The `upload-daq` and `archive`/`verify`
  stages operate on the **whole date** (they take no `--only`), so a subset run still uploads/archives every
  animal's data for that date.
- **Dates (shared grammar, resolved against the raw acquisition dirs on `E:`):** `MMDD` **or** `YYYYMMDD`; a
  **range** `0806-0808` (intersected with the available dates, so month gaps/boundaries are respected); a
  comma/space **list** `0806,0807` / `0806 0807`; or `all`. Same grammar as `nightly_figs` on the analysis box.

Examples:

```powershell
python -m wfield_local.nightly 20260808                       # full night, all animals
python -m wfield_local.nightly 0806-0808 --only PS94          # a 3-day range, preprocess just PS94
python -m wfield_local.nightly 20260808 --skip-archive        # preprocess + deck, no standby/N: archive yet
python -m wfield_local.preprocess 20260808 --skip-preprocess --skip-photobleach   # only re-run the maps
python -m wfield_local.nightly 20260808 --dry-run             # print every planned sub-command, no writes
```

## Notes

- `preprocess` auto-discovers the date's raw sessions on `E:` and, per session in animal order,
  runs sign-fixed 2D motion → SVD → cross-register to that animal's 6/6 (emitting
  `allen_aligned_affine8v1`) → **prioritized push of the LocaNMF inputs to `N:` first**, then the
  cue/lick/quiet maps, the all-days cross-day QC overlay (`xall`), the cross-day raw ROI intensity
  trend, and photobleach QC.
- **Hard rules:** sign-fixed motion; allen dir named exactly `allen_aligned_affine8v1`; the GPU push
  must include `motion_corrected/*cleanpairs_frame_map.npz`+summary (not just `wfield_local_results/`);
  never delete from E: until byte-verified, never from N:/non-Priya folders. Params live in
  `configs/defaults.yaml preprocess`; the 6/6 reference + per-animal landmark version in
  `configs/animals.yaml`.
- **M: flaky?** The drive letter often fails to resolve (`net use` error 67). Copy+verify via the UNC
  path in the Bash tool (`cp -f` then `cmp -s`); that is the fallback when `archive_day`'s drive-letter
  path breaks.
- After committing any `configs/sessions.yaml` additions or code changes, push via the rig procedure
  (`export CONDA_PREFIX=...wfield`; add/commit; fetch; rebase; push — never force-push).
