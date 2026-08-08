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
> animals, using the curated good sessions (6/6-6/8 and 8/6 onward; the code enforces this). Include all outputs in the powerpoint
> "\\research.files.med.harvard.edu\Neurobio\MICROSCOPE\Priya\Widefield\labcams\locanmf_lick_pooled\cue_analysis\spout_position_decoder_summary.pptx".
> Compare the neural encoding and decoding between mice and across sessions as possible to evaluate for
> systematic differences between how cortical representation of movement and motor planning differs across mice.
> Use all available curated good sessions (see above).
>
> While waiting for the locaNMF to be able to run - Today's camera sessions are at D:\camera. Please check the
> csv's and report the number of dropped frames per camera (first column is frame number and second column is
> timestamp) - ideally all rows are 1 frame and 4 ms apart. You previously wrote a script to do this. If there
> are dropped frames please summarize this in a document or spreadsheet that is saved in the same folder as
> these files. Please then copy all folders & files in the camera folder to
> \\research.files.med.harvard.edu\Neurobio\MICROSCOPE\Priya\Behavior_Cameras\Widefield. Today's behavior
> sessions are at D:\behavior_logs - please copy all sessions to
> \\research.files.med.harvard.edu\Neurobio\MICROSCOPE\Priya\Behavior_logs\Widefield. Don't delete anything on
> the D drive until we check in and make sure everything was copied appropriately. Once I confirm with you that
> the camera and behavior log files are copied to MICROSCOPE, please delete these files from the computer.

## Notes / how it maps to code
- Poll → LocaNMF (r2=0.95 loc=80 maxrank=20; manifest written in Python, not PowerShell — BOM breaks json).
- Analyses + deck in one command: `python -m wfield_local.nightly_figs <MMDD>` (per-day decode lick/cue 2 s +
  pre-cue 1 s, rolling, encoder + FEVE raw & normalized, cross-mouse / within-animal / RSA incl. crossnobis
  over the CURATED sessions 6/6-6/8 + 8/6 onward, tag `0606-<MMDD>`). Then update `locanmf_decoder_ppt.py` refs, `build_ppt`,
  copy to `cue_analysis/`, commit + push (rig procedure: export CONDA_PREFIX; add/commit; fetch; rebase; push).
- Dropped-frame QC: `dropframe_check_all.py "D:\camera"` (col0 frame id, col1 ns timestamp, nominal 4 ms).
  Camera → `Behavior_Cameras\Widefield\<DATE>`, behavior → `Behavior_logs\Widefield\` (the `Widefield\` SUBDIR).
- Cross-day analyses are CACHED (`session_cache.py`) — only new/changed sessions recompute; a LocaNMF re-run
  auto-invalidates that session. Force a full recompute with `WIDEFIELD_NO_CACHE=1` if you suspect staleness.
- Positions: DAQ strobe bits; dead-bit (Aug-2026) auto-repairs from behavior log (`classify_cues_with_backup`);
  empty-log sessions (PS93 8/5) use a `behavior_trials` recovered CSV. Validate by SSp >> chance 0.167.
- Engagement: the DAQ cue/strobe stream = the REWARDED subset (reward held after ~6 misses) → an engagement
  filter. Keep it; unrewarded trials belong to the future post-stroke failed-attempt analysis (movement-gated).

## Review — applied 2026-08-08
Folded INTO the canonical prompt: cross-session wording → the CURATED good sessions (6/6-6/8 + 8/6 onward,
enforced by `nightly_figs` reading `configs/animals.yaml date_policy.cross_session_exclude`); camera dest
cased `Behavior_Cameras\Widefield`. DONE separately: incremental cross-day caching (`session_cache.py` — see
the Notes). Still implicit (covered in the Notes, not the pasted prompt): the git commit + push of the deck
ref updates + any `configs/sessions.yaml` additions via the rig procedure.

