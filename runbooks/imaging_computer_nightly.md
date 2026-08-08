# Nightly runbook — IMAGING (PCO microscope) computer

The **canonical prompt Priya pastes each night** is below verbatim. Detailed step-by-step reference (params,
paths, hard rules): [`../NIGHTLY_PIPELINE.md`](../NIGHTLY_PIPELINE.md). Mounts on this machine: **`N:` =
MICROSCOPE** (`\\research.files.med.harvard.edu\Neurobio`), **`E:` = local raw/DAQ**, **`M:` = standby**
(`\\standby...\sabatini`, under `Widefield\labcams`). wfield env: `C:\ProgramData\anaconda3\envs\wfield`.

## Canonical prompt

> Today's sessions are done. Please copy the widefield DAQ recorder outputs to MICROSCOPE/Priya/Widefield.
> Please do motion correction, SVD, cross-session registration to 20260606 and use that session's Allen CCF
> alignment, and then copy non-movie (ie not the motion-corrected .bin movie) outputs, camlogs, and snapshots
> to MICROSCOPE. Prioritize, as they become available, transferring the outputs that will be needed for the
> GPU machine to run locaNMF. While that is running, please analyze photobleaching and motion-correction QC
> and add to our powerpoint "\\research.files.med.harvard.edu\Neurobio\MICROSCOPE\Priya\Widefield\labcams\PS92_94_95_affine8v1.pptx".
> I'd also like to evaluate for cross-day photobleaching - compare the raw fluorescence intensity across days
> and see if there's a trend. Please add that analysis to the powerpoint. Once done with session processing,
> analyze and produce cue- and lick- aligned activity (2s cue-evoked activity +- 2s pre-cue subtraction and
> 150 ms lick-aligned activity +- quiet period subtraction) maps to the powerpoint
> "...\labcams\PS92_94_95_affine8v1.pptx" and cross-session aligned powerpoint as well as xday folders.
> Copy the raw imaging .dat files and the motion corrected .bin files to the standby Priya drive
> 'M:\Widefield\labcams'. Don't delete anything on the E drive until we check in and make sure everything was
> copied appropriately. Once I confirm with you that the imaging files are copied to standby, please delete
> the raw and motion-corrected movies from the local computer.

## Notes / hard rules
- **Sign-fixed** motion correction (`run_wfield_motion` → `motion_correct_fixed.py`), `--mode 2d`.
- Cross-register each session to that animal's **6/6** session; emit the allen dir named exactly
  **`allen_aligned_affine8v1`** (the GPU/LocaNMF expects that name). Target NCC ~0.99.
- **Prioritized GPU push**: `SVTcorr.npy` + `allen_aligned_affine8v1/` + the
  `motion_corrected/*cleanpairs_frame_map.npz`+summary (the GPU needs the frame_map, not just
  `wfield_local_results/`) + the DAQ h5 — pushed FIRST so LocaNMF can start on early sessions.
- Fast path since 8/7: `_nightly_<DATE>.py` (template `_nightly_0807.py`) chains motion→SVD→register→push
  one session at a time; then `_maps_<DATE>_run.py`, `_photobleach_<DATE>.py`, `_crossday_intensity.py`,
  `_xall_refresh.py`, deck update, standby transfer.
- **Never delete from E:** until byte-verified; never delete from N: / non-Priya folders.

## Review — clarity / omissions / flags (2026-08-08)
Suggested edits to the canonical prompt so nothing is left implicit or wrong:
1. **⚠️ Standby path likely wrong.** The prompt says `M:\Widefield\labcams`, but `NIGHTLY_PIPELINE.md`
   states the standby is `M:\collaborations\Priya\Widefield\labcams` (`M:` = `\\standby...\sabatini`) and
   explicitly "NOT `M:\Widefield`". **Verify the correct standby path** and fix whichever is stale.
2. **Say "sign-fixed" motion correction** — it's a hard rule (`motion_correct_fixed.py`); the prompt
   just says "motion correction," which is ambiguous given the historical sign bug.
3. **"that animal's 6/6"** — registration reference is per-ANIMAL (each mouse to ITS own 20260606), not a
   single shared session. Minor wording.
4. **Name the frame_map in the GPU push** — the LocaNMF inputs the GPU needs include the
   `motion_corrected/*cleanpairs_frame_map.npz`+summary, which live in `motion_corrected/` (NOT
   `wfield_local_results/`) and are easy to miss. Worth stating explicitly.
5. Deck filename `PS92_94_95_affine8v1.pptx` omits **PS93** — cosmetic (legacy name), but confirm the
   deck actually includes PS93.

