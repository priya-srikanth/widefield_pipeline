# Nightly runbook — ANALYSIS (behavior DLC GPU) computer

MICROSCOPE mounts as **`M:`** here (`net use M: \\research.files.med.harvard.edu\Neurobio`); local
staging on **`D:`** (`D:\camera`, `D:\behavior_logs`). Env:
`C:/Users/sabatini/.conda/envs/locanmf/python.exe`; repo is `pip install -e .` (no PYTHONPATH).
Date-parametrized detail: [`../LOCANMF_NIGHTLY_PIPELINE.md`](../LOCANMF_NIGHTLY_PIPELINE.md); analysis
steps + standalone commands: `../wfield_local/README.md` §17–23.

## The whole night

```powershell
conda activate locanmf
git -C C:\Users\sabatini\GitHub\widefield_pipeline pull
python -m wfield_local.nightly <YYYYMMDD>        # camera archive + behavior FIRST, then LocaNMF figs
```

`nightly` auto-detects this is the analysis box and runs **two stages, in this order**. The camera stage
comes first because it needs no imaging output — it runs while the imaging box is still preprocessing; the
figs stage then waits for the imaging box's LocaNMF push.

### Stage 1 — `camera_nightly <date>` (per date). Steps, in order:

0. **Upload `D:` → MICROSCOPE** (size-verified, idempotent): `D:\camera\<DATE>\<PSxx>\*` →
   `Behavior_Cameras\Widefield\<DATE>\<PSxx>\`, `D:\behavior_logs\<PSxx>_<DATE>_*` →
   `Behavior_logs\Widefield\<session>\` (note the `Widefield\` SUBDIR). **`D:` is never deleted** — a copy
   failure stops the run before the later steps; delete `D:` manually only after byte-verify + check-in.
1. **Dropped-frame QC** → `dropped_frames_summary_<DATE>.{csv,txt}` next to the data (per cam:
   rows/id_span/dropped/gaps + timestamp-delta stats; CSV cols 0/1/2 = frame_id / timestamp_ns, ~4.003 ms
   apart @ ~250 fps / GPIO, bit0 = Arduino sync; a drop = a gap in the monotonic frame_id).
2. **Camera↔DAQ alignment templates** — each cam's GPIO sync train (bit0) matched to the DAQ `sync` line
   (bit0, 5000 Hz) via the bounded-window ITI matcher, written to
   `...\Behavior_Cameras\Widefield\alignment_templates\<cam>\<PSxx>\<YYYYMMDD>.npz` (maps camera TIME→DAQ
   time, affine, drop-proof; for post-stroke multi-angle DLC / behavior↔imaging alignment). Logs
   `matched/edges`, `resid_ms_rms` (~1-2 ms good), `frame_drops`, and a `QUALITY CHECK FAILED` flag if off.
3. **Canonical behavior events** (`wfield_local.behavior_events`) → `behavior_summary\events\<animal>\
   <date>.npz`: shared licks / reward / running-bouts / quiet-periods on the DAQ clock, so behavior AND
   imaging load the SAME event identity instead of re-detecting. (The imaging box regenerates the same file
   during its maps step; whichever runs first wins, the other reuses it.)
4. **Spout behavior figures** (`wfield_local.spout_behavior`) — reads the task-controller's already-scored
   `trials.csv` and writes, under `Behavior_logs\Widefield\behavior_summary\sessions\<animal>\<date>\`, a
   per-session accuracy PNG + per-position CSV **and** a lick-microstructure PNG (peri-cue raster/PSTH, ILI
   distribution, lick bouts, per-position lick rate / licks-per-trial / anticipatory licks, and a
   GUI-vs-DAQ-pipeline lick-count comparison), plus a refreshed cross-session cohort figure and a
   `cohort\by_animal\` across-session figure per animal. Accuracy is **engaged-gated** (terminal sated-tail +
   rolling-collapse gate; raw all-trial rate shown alongside). One cue + 6 spout positions (close/far ×
   L/center/R; latency + licks from `events.csv`; DAQ licks use the 40 ms physiological floor). Aborted runs
   (< `min_session_trials`) are auto-skipped.

### Stage 2 — `nightly_figs <date>` (waits for the imaging box's LocaNMF push). Steps, in order:

1. **Per-day position decode** — one run per alignment (lick / cue / pre-cue, each 2 s from
   `configs/defaults.yaml decode.*_post_s`), `--per-session`.
2. **Decoder-weight & dynamics figures** (in-process): rolling-cue, temporal-dynamics, rolling-laterality,
   and per-session top-components.
3. **Encoder** (`locanmf_position_encoder`) — per-position EV + FEVE (raw & normalized-to-1.0), pooled over
   the cross-session set.
4. **Cross-mouse + RSA** (incl. crossnobis) — once, over the whole `--from` set.
5. **Per-animal rolling decoder** across the curated sessions (Section A of the deck).
6. **Build the analysis deck** `spout_position_analysis_summary.pptx` at the `labcams` top level
   (`locanmf_analysis_deck.py`, curated animal→type→date). LocaNMF itself (r2 0.95 / loc 80 / maxrank 20)
   must already have run on the pushed inputs before this stage.

## Running only part of the pipeline

**Whole-stage skips on the top-level `nightly`:**

| flag | effect |
|------|--------|
| `--skip-camera` | skip Stage 1 entirely (upload + QC + align + events + behavior) |
| `--skip-figs`   | skip Stage 2 entirely (LocaNMF decode/encode/RSA + deck) |
| `--dry-run`     | print the plan for both stages; write nothing |

`nightly` forwards only `--only` / `--dry-run` to the sub-stages — **not** per-step skips. For finer control,
run the sub-command standalone:

```powershell
python -m wfield_local.camera_nightly <DATE>    # Stage 1 alone; per-step skips below
#   --skip-copy  --skip-dropframe  --skip-align  --skip-events  --skip-behavior   --dry-run
python -m wfield_local.spout_behavior <DATE> [--cohort] [--from curated]   # step 4 only
python -m wfield_local.behavior_events <YYYYMMDD> [--only PS94]            # step 3 only
python -m wfield_local.nightly_figs <DATE> [--from <span>]                # Stage 2 alone
```

## Selecting animals and dates

- **Animals — `--only`:** `--only PS93` (or `--only PS93 PS94`, or `all` = no filter). Forwarded to every
  sub-stage; for the figs subprocesses it is passed through as `WIDEFIELD_ONLY_ANIMALS`. Note the upload in
  Stage 1 step 0 is scoped too, but a whole date's cameras still copy per animal folder.
- **Dates (shared grammar, resolved against the camera dirs):** `MMDD` **or** `YYYYMMDD`; a **range**
  `0806-0808` (intersected with the available dates, so month gaps/boundaries are respected); a comma/space
  **list** `0806,0807` / `0806 0807`; or `all`. Same grammar in `preprocess` on the imaging box.
- **`--from <span>` (analysis only):** the cross-session comparison span for `nightly_figs` (Stage 2 steps
  3–6). Default = the CURATED set from `configs/animals.yaml date_policy` (6/6–6/8 + 8/6 onward, auto-including
  future dates; excludes noisy early June + the wonky 8/5). The per-day `dates` arg and `--from` are
  independent: `dates` picks which day(s) get per-session figures, `--from` picks the cross-session pool.

Examples:

```powershell
python -m wfield_local.nightly 20260808                       # both stages, all animals, latest logic
python -m wfield_local.nightly 0806-0808 --only PS94          # a 3-day range, one animal
python -m wfield_local.nightly 20260808 --skip-camera         # just the LocaNMF figs + deck
python -m wfield_local.nightly 20260808 --skip-figs           # just camera upload/QC/align/events/behavior
python -m wfield_local.nightly 20260808 --from 0606-0807      # per-day 8/8; cross-session pool 6/6→8/7
```

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
