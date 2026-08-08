# Nightly runbook — IMAGING (PCO microscope) computer

Mounts here: **`N:` = MICROSCOPE** (`\\research.files.med.harvard.edu\Neurobio`), **`E:` = local
raw/DAQ**, **`M:` = standby** (`\\standby...\sabatini`, under `collaborations\Priya\Widefield\labcams`).
Env: `C:\ProgramData\anaconda3\envs\wfield`. Full detail (params, paths, hard rules, cleanup):
[`../NIGHTLY_PIPELINE.md`](../NIGHTLY_PIPELINE.md); underlying per-step commands: `../wfield_local/README.md`.

## The whole night

```powershell
conda activate wfield
git -C C:\Github\widefield_pipeline pull
python -m wfield_local.preprocess <YYYYMMDD>      # discover -> motion/SVD/xreg/push -> maps -> xall -> photobleach
python -m wfield_local.preprocess_deck            # rebuild PS92-95_cross_sessions_aligned.pptx in place
python -m wfield_local.archive_day archive --date <YYYYMMDD>   # raw + .bin -> M:, everything else -> N:
```

Then, **after checking in that N:/M: copies are byte-verified**, reclaim E: space:

```powershell
python -m wfield_local.archive_day clean --date <YYYYMMDD>              # DRY-RUN
python -m wfield_local.archive_day clean --date <YYYYMMDD> --execute    # actually delete from E:
```

## Notes

- `preprocess` auto-discovers the date's raw sessions on `E:` and, per session in animal order,
  runs sign-fixed 2D motion → SVD → cross-register to that animal's 6/6 (emitting
  `allen_aligned_affine8v1`) → **prioritized push of the LocaNMF inputs to `N:` first**, then the
  cue/lick/quiet maps, the all-days cross-day QC overlay (`xall`), and photobleach QC. `--dry-run`
  to preview; `--only PS94 PS95` to subset; ranges / `all` accepted (`preprocess 0806-0808`).
- Still manual (not folded): `_crossday_intensity.py` (cross-day raw ROI intensity; `preprocess_deck`
  embeds its PNG if present).
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
