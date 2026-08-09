# Ongoing tasks & decisions (widefield pipeline)

## Standing nightly pipeline
The per-night runbook lives in **`runbooks/imaging_computer_nightly.md`** (source of truth); it is now driven by the
config-driven orchestrator — `python -m wfield_local.preprocess <DATE>` (discover → fixed motion + SVD
→ cross-register each session to that animal's 6/6 + 6/6 Allen CCF, emit `allen_aligned_affine8v1`, no
new landmarks → LocaNMF inputs → N: first → cue/lick maps → xall → photobleach), then
`preprocess_deck`, then `archive_day` (raw+bin → M:). Don't delete E: until checked in.

Deck: `N:\MICROSCOPE\Priya\Widefield\labcams\cross-session_preprocessing_<animal>.pptx`.

## 6/6 reference landmarks for cross-register-all (use v2 where present)
When doing the final cross-register-all-to-6/6, the per-animal 6/6 reference Allen CCF uses
these landmark JSONs (in each 6/6 session's `raw_widefield_data\`, on N:):
- **PS92 -> `dorsal_cortex_landmarks_v2.json`** (re-done 2026-06-09)
- **PS93 -> `dorsal_cortex_landmarks_v2.json`** (re-done 2026-06-09)
- PS94 -> `dorsal_cortex_landmarks_v1.json`
- PS95 -> `dorsal_cortex_landmarks_v1.json`
Also recompute PS92 & PS93's 6/6 reference `allen_aligned_affine8v1` with their v2 landmarks
so the reference session itself is consistent with what the other days are aligned to.

## Redo scope: 6/5 onward only (6/4 and earlier NOT redone)
Decision (2026-06-09): the sign-bug drift on 6/4 and earlier was negligible (<1.1 px), so
those sessions are **not** re-motion-corrected. Redo covers 6/5-6/8. The cross-sessions_aligned
deck likewise starts at 6/5.

## Motion-correction sign-bug remediation (DONE)
wfield 0.4.2 doubled drift (sign error); fixed in `wfield_local/motion_correct_fixed.py`
(see `docs/archive/MOTION_CORRECTION_SIGN_BUG.md`, now the standard motion path via `run_wfield_motion`).
All 6/5-6/8 sessions were re-corrected with the fix and cross-registered to their 2026-06-06
reference (Allen-aligned via 6/6), and the deck + cross-day QC refreshed. (The resumable redo
driver + its status file were one-offs, retired once the batch completed.)

## Servers
- M: standby = raw `.dat` + corrected `.bin` (huge files), `M:\Widefield\labcams\<date>\<session>\`.
- N: MICROSCOPE = analyzed (SVD, CCF-aligned, maps, QC, DAQ, deck); NOT the `.bin`.

## Re-run LocaNMF after a motion redo (GPU lane)
Any session whose motion correction is re-done with the sign-fixed code gets a NEW
`SVTcorr.npy` + `allen_aligned_affine8v1/` (its SVD and CCF-aligned U change). **Any
LocaNMF result computed on the pre-fix inputs is stale and must be re-run** against the
corrected inputs now on N:. This applied to every re-corrected 6/5-6/8 session.
GPU: re-run LocaNMF for a session after its `wfield_local_results` mtime updates on N:.

## Allen-dir naming (GPU/LocaNMF)
Cross-session-to-6/6 emits the CCF allen dir as **`allen_aligned_affine8v1`** (the
standard name the GPU/LocaNMF, maps, and deck all expect) -- it CONTAINS the 6/6-CCF
alignment. Do not use a custom name (e.g. xday6) on N: or the GPU won't find it.

## Dead strobe bit1 (2026-08-05/06)
Widefield DAQ strobe bit1 dead 8/5-8/6 -> close_R+far_center lost from DAQ; recovered from behavior trials.csv via `--behavior-trials` (see STROBE_BIT1_RECOVERY.md). Hardware fix pending for 8/7+.

## Post-stroke science prerequisites (plan in DECISIONS.md Part V)
The forward work the baseline pipeline is building toward:
- **Per-trial behavioral-state table** — spout-contact + DAQ lick → hit/miss/**failed-attempt**, latency,
  executed position. Keep failed-attempt trials (the deficit); state is **movement-gated, not lick-gated**.
- **Frozen pre-stroke decoder + baseline noise floor** — package the fixed pre-stroke `A` (refit `C`) or
  Allen-ROI model + each animal's multi-baseline-day distribution, so a post-stroke session is diffable.
- **Movement regressors** — DLC / FaceRhythm, **time-synced** to widefield+DAQ (the camera↔DAQ alignment
  templates exist), to separate "cortex codes position differently" from "the movement just changed".
- **Cross-day vasculature registration** (`cross_day_align.py`) across the baseline set.

## Quiet-period thresholds (tune per rig)
Running/quiet speed, min durations, and lick/reward/treadmill buffers are stroke-pipeline defaults; tune per
rig/task and validate against DLC/FaceRhythm movement once available (see `DECISIONS.md` "Quiet-period
baseline"). Grooming detection stays OFF (single-spout long-touch is unreliable).
