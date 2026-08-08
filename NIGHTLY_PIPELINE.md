# Nightly widefield pipeline (imaging box)

Run after each recording day, once imaging is done and the rig is free. `<DATE>` = labcams
folder name, `YYYYMMDD`. Sessions are `PSxx_<DATE>_<hhmmss>`. Everything below is driven by
the **config-driven orchestrator** (`configs/*.yaml` = single source of truth); the underlying
per-step commands and params are documented in `wfield_local/README.md` §1–16.

## The whole night, three commands

```powershell
conda activate wfield
git -C C:\Github\widefield_pipeline pull        # get the latest configs/code from the other box
python -m wfield_local.preprocess <DATE>        # discover -> motion/SVD/xreg/push -> maps -> xall -> photobleach
python -m wfield_local.preprocess_deck          # rebuild the single cross-sessions_aligned.pptx in place
python -m wfield_local.archive_day archive --date <DATE>   # raw+.bin -> M: standby, everything else -> N:
```

- `preprocess <DATE>` auto-discovers that date's raw sessions on `E:` (no session-dir / DAQ /
  dims hard-coding), then **per session, in animal order**: sign-fixed 2D motion → SVD (k=100,
  functional ch1, fs 31.23, hp 0.1, lp 14) → cross-register to that animal's `reference_date`
  (6/6) emitting `allen_aligned_affine8v1` → **push the LocaNMF inputs to `N:` first** (so the
  GPU box can start LocaNMF on early sessions while later ones are still correcting). Then it
  runs the cue/lick/quiet **activity maps**, the per-animal **all-days cross-day QC overlay**
  (`xall`), and **photobleach** QC. Flags: `--dry-run` prints the plan; `--only PS94 PS95`
  subsets animals; multiple dates / ranges / `all` accepted (`preprocess 0806-0808`).
- `preprocess_deck` rebuilds the one canonical deck `labcams/PS92-95_cross_sessions_aligned.pptx`
  in place (grouped animal → figure type → date; per-animal split to bound size). ~30 s, pure
  figure assembly.
- `archive_day archive` copies raw + motion-corrected `.bin` → **M: standby** and all other
  outputs → **N: MICROSCOPE** (LocaNMF inputs first), re-verifying sizes. See E: cleanup below.

Still a **separate one-off** (not yet folded into `preprocess`): `_crossday_intensity.py`
(brain-ROI median raw counts per animal across days from each session's `frames_average`;
`preprocess_deck` embeds its `crossday_raw_intensity*.png` output if present). CAVEAT: LED is
manually titrated day-to-day, so an intensity trend may reflect LED, not bleaching.

## Hard rules

- **Sign-fixed** motion correction (`run_wfield_motion` → `motion_correct_fixed.py`), `--mode 2d`.
- Cross-register each session to that animal's **6/6** session; the emitted dir MUST be named
  `allen_aligned_affine8v1` (GPU/LocaNMF, maps, and deck all expect that name). Target NCC ~0.99.
  No new per-session landmark placement. Per-animal 6/6 landmark version: PS92/PS93 v2, PS94/PS95 v1
  (`configs/animals.yaml reference_landmarks`).
- The GPU needs ALL of: `wfield_local_results/SVTcorr.npy`, `.../allen_aligned_affine8v1/`, the
  `motion_corrected/*cleanpairs_frame_map.npz` + `*_cleanpairs_summary.json` (the frame_map lives
  in `motion_corrected/`, NOT `wfield_local_results/` — easy to miss), and the DAQ `.h5`. The
  prioritized push in `preprocess` handles all four.
- **Never delete from E:** until copies are byte-verified. Never delete from `N:` (MICROSCOPE) or
  any non-Priya folder without explicit per-time permission.
- After any motion redo, the GPU must **re-run LocaNMF** on the corrected inputs.

## Paths

- Raw + DAQ on E: `E:\labcams_data\<DATE>\<session>\raw_widefield_data\...`,
  `E:\DAQ_recorder_output\PSxx_<DATE>_*.h5` (DAQ files sit loose, not per-session).
- MICROSCOPE (analysis): `N:\MICROSCOPE\Priya\Widefield\labcams\<DATE>\<session>\...`
  (`N:` = `\\research.files.med.harvard.edu\Neurobio`).
- Standby (huge files): `M:\collaborations\Priya\Widefield\labcams\<DATE>\<session>\...`
  (`M:` = `\\standby.files.med.harvard.edu\hms\neurobio\sabatini`; note it is under
  `collaborations\Priya\`). The `M:` drive letter often fails to resolve in the shell
  (per-session mapping / `net use` error 67) — copy+verify via the UNC path in the Bash tool
  (`cp -f` then `cmp -s`), which works through the MSYS layer. `archive_day.py`'s drive-letter
  path can break when M: is flaky; the manual UNC `cp`/`cmp` loop is the fallback.
- Deck: `N:\MICROSCOPE\Priya\Widefield\labcams\PS92-95_cross_sessions_aligned.pptx`.
- Snapshots / camlogs are mirrored to N: by `archive_day`.
- wfield python: `C:\ProgramData\anaconda3\envs\wfield\python.exe`. The repo is `pip install -e .`,
  so `python -m wfield_local.*` needs no PYTHONPATH.

## E: cleanup (only after explicit check-in)

`python -m wfield_local.archive_day clean --date <DATE>` (dry-run; add `--execute`). It re-verifies
the M:/N: copy (byte size) before deleting and only removes confirmed-copied files + reproducible
intermediates (cleanpairs `*.dat`, `*_concat` raw). For a single re-corrected session, delete
session-scoped after byte-verifying its `.bin` on M: + outputs on N:.

## Dead strobe bit / behavior-log position recovery

If spout cue/lick maps show <6 positions, a DAQ strobe bit may be dead (see 8/5–8/6). `preprocess`
auto-recovers TRUE positions: its maps step passes `--behavior-trials <trials.csv>` whenever a
recovered CSV is discoverable (an explicit `behavior_trials` in `sessions.yaml`, or
`N:\...\Behavior_logs\Widefield\<sess>\trials.csv`). See `STROBE_BIT1_RECOVERY.md` for the incident
detail and the cam1 video-recovery fallback (PS93 8/5).

See also: `TASKS.md` (decisions), `MOTION_CORRECTION_SIGN_BUG.md`, `wfield_local/README.md` (steps).
