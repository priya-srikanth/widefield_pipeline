# Nightly runbook — ANALYSIS (behavior DLC GPU) computer

The **canonical prompt Priya pastes each night** is below verbatim. Date-parametrized executable detail:
[`../LOCANMF_NIGHTLY_PIPELINE.md`](../LOCANMF_NIGHTLY_PIPELINE.md). Env:
`C:/Users/sabatini/.conda/envs/locanmf/python.exe`; repo is `pip install -e .` (no PYTHONPATH hack).
MICROSCOPE mounts as **`M:`** here (`net use M: \\research.files.med.harvard.edu\Neurobio`).

## Canonical prompt

> Today's sessions are being motion-corrected and undergoing SVD and Allen alignment on the imaging computer,
> then will be uploaded to MICROSCOPE. Please check for the outputs needed for locaNMF every 30 minutes,
> starting at 1 am, then do locaNMF on all sessions. Then run rolling temporal window cue-aligned and
> first-lick-aligned spout position decoding analysis and pre-cue decoding analysis on all sessions. Please
> also perform encoding analysis including plotting fraction of explained variance (both raw number and
> normalized to 1, as we have done before). Compare encoding and decoding performance across sessions for all
> animals, starting from 6/6. Include all outputs in the powerpoint
> "\\research.files.med.harvard.edu\Neurobio\MICROSCOPE\Priya\Widefield\labcams\locanmf_lick_pooled\cue_analysis\spout_position_decoder_summary.pptx".
> Compare the neural encoding and decoding between mice and across sessions as possible to evaluate for
> systematic differences between how cortical representation of movement and motor planning differs across mice.
> Use all sessions available.
>
> While waiting for the locaNMF to be able to run - Today's camera sessions are at D:\camera. Please check the
> csv's and report the number of dropped frames per camera (first column is frame number and second column is
> timestamp) - ideally all rows are 1 frame and 4 ms apart. You previously wrote a script to do this. If there
> are dropped frames please summarize this in a document or spreadsheet that is saved in the same folder as
> these files. Please then copy all folders & files in the camera folder to
> \\research.files.med.harvard.edu\Neurobio\MICROSCOPE\Priya\Behavior_Cameras\widefield. Today's behavior
> sessions are at D:\behavior_logs - please copy all sessions to
> \\research.files.med.harvard.edu\Neurobio\MICROSCOPE\Priya\Behavior_logs\Widefield. Don't delete anything on
> the D drive until we check in and make sure everything was copied appropriately. Once I confirm with you that
> the camera and behavior log files are copied to MICROSCOPE, please delete these files from the computer.

## Notes / how it maps to code
- Poll → LocaNMF (r2=0.95 loc=80 maxrank=20; manifest written in Python, not PowerShell — BOM breaks json).
- Analyses + deck in one command: `python -m wfield_local.nightly_figs <MMDD>` (per-day decode lick/cue 2 s +
  pre-cue 1 s, rolling, encoder + FEVE raw & normalized, cross-mouse / within-animal / RSA incl. crossnobis
  over ALL registered sessions, tag `0601-<MMDD>`). Then update `locanmf_decoder_ppt.py` refs, `build_ppt`,
  copy to `cue_analysis/`, commit + push (rig procedure: export CONDA_PREFIX; add/commit; fetch; rebase; push).
- Dropped-frame QC: `dropframe_check_all.py "D:\camera"` (col0 frame id, col1 ns timestamp, nominal 4 ms).
  Camera → `Behavior_Cameras\Widefield\<DATE>`, behavior → `Behavior_logs\Widefield\` (the `Widefield\` SUBDIR).
- Positions: DAQ strobe bits; dead-bit (Aug-2026) auto-repairs from behavior log (`classify_cues_with_backup`);
  empty-log sessions (PS93 8/5) use a `behavior_trials` recovered CSV. Validate by SSp >> chance 0.167.
- Engagement: the DAQ cue/strobe stream = the REWARDED subset (reward held after ~6 misses) → an engagement
  filter. Keep it; unrewarded trials belong to the future post-stroke failed-attempt analysis (movement-gated).

## Review — clarity / omissions / flags (2026-08-08)
1. **"starting from 6/6" is now stale.** The pipeline was changed to use ALL available sessions (6/1 onward;
   the prompt's own last line already says "Use all sessions available"). Replace "starting from 6/6" with
   "across all available sessions" so the two lines don't contradict.
2. **⚠️ Cross-day analyses must be INCREMENTAL — do not re-analyze old data each night.** Right now the
   cross-mouse / within-animal / RSA (+ crossnobis) steps recompute `per_session` recall/EV and the 6×6 RDMs
   for EVERY session from scratch on every run (the slow pole — the hemisphere-RSA/crossnobis recompute took
   ~20+ min). **Cache each session's per-session outputs** (recall/EV vectors, RDM, crossnobis dissimilarities,
   features) keyed by session + a hash of (LocaNMF output mtime, params); on each nightly run compute only the
   NEW session(s) and load cached results for the rest. Invalidate a session's cache only when its LocaNMF
   output or params change (e.g. the PS93 8/5 recovered-positions rerun). See README "Roadmap".
3. Add the **git commit + push of code + deck** explicitly (the deck lives on MICROSCOPE, but the ppt-builder
   ref updates + config/session additions must be committed via the rig procedure).
4. Camera destination should be `Behavior_Cameras\Widefield` (capital W); Windows is case-insensitive so the
   lowercase `widefield` in the prompt still works, but match the canonical casing.

