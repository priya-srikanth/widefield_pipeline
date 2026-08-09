# Nightly runbook — ANALYSIS (behavior DLC GPU) computer

MICROSCOPE mounts as **`M:`** here (`net use M: \\research.files.med.harvard.edu\Neurobio`); local
staging on **`D:`** (`D:\camera`, `D:\behavior_logs`). Env:
`C:/Users/sabatini/.conda/envs/locanmf/python.exe`; repo is `pip install -e .` (no PYTHONPATH).
Date-parametrized detail: [`../LOCANMF_NIGHTLY_PIPELINE.md`](../LOCANMF_NIGHTLY_PIPELINE.md); analysis
steps + standalone commands: `../wfield_local/README.md` §17–23.

## The whole night

Once the imaging box has pushed a session's LocaNMF inputs to MICROSCOPE (`SVTcorr.npy`, the
`allen_aligned_affine8v1/` dir, the `*cleanpairs_frame_map.npz`+summary, DAQ h5), run LocaNMF, then:

```powershell
conda activate locanmf
git -C C:\Users\sabatini\GitHub\widefield_pipeline pull
python -m wfield_local.nightly_figs <MMDD>       # per-day decode/encode + cross-session RSA + deck
```

`nightly_figs <MMDD>` runs the per-day decode (lick/cue 2 s, pre-cue 1 s), decoder-weight & dynamics
figures, encoder (+ FEVE raw & normalized), and the cross-mouse / RSA (incl. crossnobis) comparison
over the CURATED sessions (6/6–6/8 + 8/6 onward, enforced from `configs/animals.yaml date_policy`),
then builds `locanmf_lick_pooled/cue_analysis/spout_position_decoder_summary.pptx`. Subset with
`--only PS93`; ranges / `all` / `--from <span>` accepted (grammar shared with `preprocess`).

## Camera + behavior-log offload (while LocaNMF is unavailable)

```powershell
python -m wfield_local.dropframe_qc 20260807               # MICROSCOPE behavior_cameras/<date>
python -m wfield_local.dropframe_qc 20260807 --root D:/camera   # pre-upload staging (flat <PSxx>/ dirs)
```

Writes `dropped_frames_summary_<DATE>.csv` (one row per cam: rows/id_span/dropped/gaps + timestamp-delta
stats) + a `.txt` table next to the data. CSV cols 0/1/2 = frame_id / timestamp_ns (~4.003 ms apart, ~250
fps) / GPIO (bit0 = Arduino sync); a drop is a gap in the monotonic frame_id.

Then copy (don't delete D: until byte-verified + checked in): `D:\camera\*` →
`M:\MICROSCOPE\Priya\Behavior_Cameras\Widefield\<DATE>\`, `D:\behavior_logs\*` →
`M:\MICROSCOPE\Priya\Behavior_logs\Widefield\` (note the `Widefield\` SUBDIR in both).

Once the camera CSVs + DAQ `.h5` are on MICROSCOPE, build the **camera↔DAQ alignment templates** (one
per cam per date, for post-stroke multi-angle DLC / behavior↔imaging alignment):

```powershell
python -m wfield_local.camera_sync <DATE>            # all animals; --only PS94 to subset
```

Each cam's GPIO sync train (bit0) is matched to the DAQ `sync` line (bit0, 5000 Hz) via the bounded-window
ITI matcher; a compact `<cam>_<recording>_daq_alignment.npz` is written next to the CSV, mapping camera
TIME→DAQ time (affine, drop-proof). It logs `matched/edges`, `resid_ms_rms` (~1-2 ms good), and
`frame_drops`; a `QUALITY CHECK FAILED` flag means the residual is off (investigate before trusting it).

## Notes

- LocaNMF: r2=0.95, loc=80, maxrank=20; write the manifest in Python, not PowerShell (a UTF-8 BOM
  breaks the JSON).
- Cross-day analyses are **cached** (`session_cache.py`) — only new/changed sessions recompute; a
  LocaNMF re-run auto-invalidates that session. Force a full recompute with `WIDEFIELD_NO_CACHE=1`.
- Positions come from DAQ strobe bits; a dead bit (Aug-2026) auto-repairs from the behavior log
  (`classify_cues_with_backup`); an empty-log session (PS93 8/5) uses a `behavior_trials` recovered
  CSV set in `sessions.yaml`. Sanity check: SSp decode >> chance 0.167.
- The DAQ cue/strobe stream = the REWARDED subset (reward held after ~6 misses) → an engagement
  filter. Keep it; unrewarded trials belong to the future post-stroke failed-attempt analysis.
- After the run, commit + push deck/config changes via the rig procedure (`export CONDA_PREFIX=...
  locanmf`; add/commit; fetch; rebase; push — never force-push).
