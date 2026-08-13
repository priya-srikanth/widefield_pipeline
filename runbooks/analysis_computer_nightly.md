# Nightly runbook — ANALYSIS (behavior DLC GPU) computer

MICROSCOPE mounts as **`M:`** here (`net use M: \\research.files.med.harvard.edu\Neurobio`); local
staging on **`D:`** (`D:\camera`, `D:\behavior_logs`). Env:
`C:/Users/sabatini/.conda/envs/locanmf/python.exe`; repo is `pip install -e .` (no PYTHONPATH).
Underlying analysis steps + standalone commands: [`../wfield_local/README.md`](../wfield_local/README.md)
§17–23. Design/findings: `../DECISIONS.md`.

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
4. **Spout behavior figures** (`wfield_local.spout_behavior`) — trials come from the **DAQ recorder `.h5`**
   (`daq_trials.py`; the GUI `trials.csv` mislabels `pos_idx` on ~15% of trials — see
   [`../docs/GUI_TRIALS_LOGGING.md`](../docs/GUI_TRIALS_LOGGING.md)), with the log as fallback when the
   strobe stream is degraded. Writes, under `Behavior_logs\Widefield\behavior_summary\sessions\<animal>\<date>\`, a
   per-session accuracy PNG + per-position CSV **and** a lick-microstructure PNG (peri-cue raster/PSTH, ILI
   distribution, lick bouts, per-position lick rate / licks-per-trial / anticipatory licks, and a
   GUI-vs-DAQ-pipeline lick-count comparison), plus a refreshed cross-session cohort figure and a
   `cohort\by_animal\` across-session figure per animal. The accuracy PNG also carries the by-position
   lick metrics (licks/trial, lick rate, anticipatory) so a session reads against the cross-day trend.
   Accuracy **and the per-position lick metrics** are **engaged-gated** (terminal sated-tail +
   rolling-collapse gate; raw all-trial rate shown alongside). One cue + 6 spout positions (close/far ×
   L/center/R; latency + licks from the DAQ, or `events.csv` on the log fallback; DAQ licks use
   the 40 ms physiological floor; hit/miss uses the session's real response window, 3500 ms). Aborted runs
   (< `min_session_trials`) are auto-skipped.

### Stage 2 — `nightly_figs <date>` — **auto-gated on LocaNMF being done.** Steps, in order:

> **Gate:** `nightly` runs Stage 2 only if the date's sessions are **registered in `configs/sessions.yaml`**
> — which happens in the manual "Before the figs" step *after* the GPU LocaNMF run. A freshly-recorded night
> is not registered yet, so Stage 2 **auto-defers** (does the camera/behavior work now, holds the figs) and
> prints the command to run once LocaNMF lands. Force it with `--figs`; hard-skip with `--skip-figs`.


1. **Per-day position decode** — one run per alignment (lick / cue / pre-cue, each 2 s from
   `configs/defaults.yaml decode.*_post_s`), `--per-session`. Also **backfills** any curated date whose
   per-day figures are missing on disk, so a night that failed does not leave a blank column forever.
2. **Decoder-weight & dynamics figures** (in-process): rolling-cue, temporal-dynamics, rolling-laterality,
   and per-session top-components.
3. **Encoder** (`locanmf_position_encoder`) — per-position EV + FEVE (raw & normalized-to-1.0), pooled over
   the cross-session set.
4. **Cross-mouse + RSA** (incl. crossnobis) — once, over the whole `--from` set.
5. **Pre-cue lick-free control** (`precue_lickfree`), in BOTH bases (roi, locanmf) — decode/encode on a
   searched 2 s window containing no licks, between the position strobe and the cue. Deck Section C.
6. **Per-animal rolling decoder** across the curated sessions (Section A of the deck).
7. **Frozen cross-day decoder + encoder** (`locanmf_frozen_decoder`, Allen-ROI, leave-one-session-out),
   for BOTH the post-cue and pre-cue alignments. ~30–40 min; `--skip-frozen` skips it and leaves those
   deck slides blank.
8. **Cross-session decode/encode in the joint-LocaNMF basis** (`joint_xsession`) — the same LOSO design
   in the second, independent basis, plus the basis-health (variance-captured) diagnostic. Deck Section
   D alongside the ROI version. Requires a joint basis built for the animal
   (`wfield_local.joint_locanmf`); a missing one is reported and skipped, **never silently refitted** —
   a refit over a grown session set is a different reference frame. Also gated by `--skip-frozen`.
9. **Build the analysis deck** `spout_position_analysis_summary.pptx` at the `labcams` top level
   (`locanmf_analysis_deck.py`; A–C within-day per animal→type→date, D cross-session per basis→
   alignment→animal, E–F cohort summaries). LocaNMF itself (r2 0.95 / loc 80 / maxrank 20) must already
   have run on the pushed inputs before this stage.
10. **Publish the component PNGs** to MICROSCOPE (`cue_analysis_out`) so the individual figures persist
    on the server beside the deck. Incremental; never deletes.

A step that fails is logged and the run continues, but `nightly_figs` **exits non-zero** if any did — an
all-steps-failed run used to exit 0 with an empty deck, which is indistinguishable from success.

## Before the figs: LocaNMF + session registration

Stage 2 assumes LocaNMF has run and the sessions are registered. Once the imaging box's push lands on
MICROSCOPE, either **automate the whole wait** or do it by hand.

**Automated (recommended) — poll + run, standalone from PowerShell:**

```powershell
python -m wfield_local.await_locanmf 20260809          # every 30 min: detect inputs -> LocaNMF -> register -> push -> figs
python -m wfield_local.await_locanmf 20260809 --once --dry-run   # one detection pass, no writes
#   --animals PS93 PS94   --interval-min 15   --no-push   --no-figs   --no-locanmf
```

It checks MICROSCOPE for each mouse's `SVTcorr.npy` + `allen_aligned_affine8v1/U_atlas.npy`; when a mouse is
ready it runs `batch_locanmf`, registers it in `configs/sessions.yaml` (regime B if a `*cleanpairs_frame_map.npz`
is present, else A), commits + pushes, and refreshes `nightly_figs`. It exits once all four mice are registered.
It needs the GPU (for LocaNMF) and it auto-commits `sessions.yaml` — use `--no-push`/`--dry-run` to hold back.

**By hand** — the same four steps the poller automates:

1. **Detect inputs.** For each mouse, check for
   `…/labcams/<YYYYMMDD>/PS9*_<YYYYMMDD>*/motion_corrected/wfield_local_results/allen_aligned_affine8v1/
   U_atlas.npy` + sibling `../../SVTcorr.npy`. If none ready, wait (the cron re-fires in ~30 min). Inputs
   sometimes land under a temp `allen_aligned_*` name first; use `allen_aligned_affine8v1` for consistency.
2. **Run LocaNMF.** Write `~/source/locanmf_batch_<MMDD>.json` (one `{label, allen_dir, output}` per ready
   session; `output=…/motion_corrected/locanmf_affine8v1_final`) **in Python, not PowerShell** (a UTF-8 BOM
   breaks the JSON), then:
   `python -u -m wfield_local.batch_locanmf --manifest <m> --r2 0.95 --loc 80 --maxrank 20`.
   Sanity: ~90–180 components/session.
3. **Register the sessions** in `configs/sessions.yaml` (keyed `animal → "MMDD"`, **dates QUOTED**);
   `config.load_sessions()` supplies the runtime `SESSIONS` — the old hardcoded list is retired, do NOT re-add
   it. **Regime:** `*cleanpairs_frame_map.npz` present in `motion_corrected/` → regime `"B"` (`fmdir=None`);
   absent → `"A"`. Validate the regime by **sensible decoding** (SSp ≫ chance 0.167), NOT by RT — if decoding
   collapses to chance, the regime is wrong, try the other (the 6/5 bug: A gave chance, B fixed it).
4. **Spout-position source + backup.** Positions come from the DAQ spout-strobe bits; a dead bit (Aug-2026
   `spout_bit1`) is auto-repaired by `classify_cues_with_backup` (wired into `_trial_features`, so
   decoder/encoder/cross-mouse/RSA all inherit it): if the DAQ shows <6 positions AND the task controller's
   `trials.csv` (`pos_idx`) aligns to the DAQ's good positions at ≥0.9 by an integer trial-offset, it
   substitutes the behavior-log positions; otherwise DAQ is left untouched (good sessions never altered). An
   empty-log session (PS93 8/5) uses a `behavior_trials` recovered CSV set in `sessions.yaml`. Full incident
   detail: [`../STROBE_BIT1_RECOVERY.md`](../STROBE_BIT1_RECOVERY.md).

Then run Stage 2 (`nightly_figs`, above). It commits nothing — after the run, push deck/config changes via
the rig procedure (see Notes).

## Running only part of the pipeline

**Whole-stage skips on the top-level `nightly`:**

| flag | effect |
|------|--------|
| `--skip-camera` | skip Stage 1 entirely (upload + QC + align + events + behavior) |
| `--skip-figs`   | hard-skip Stage 2 (LocaNMF decode/encode/RSA + deck) |
| `--figs`        | force Stage 2 even if the date isn't registered (overrides the LocaNMF-ready gate; e.g. refresh the curated deck) |
| `--await-locanmf` | after Stage 1, hand off to the `await_locanmf` poller — blocks ~30-min loop until the LocaNMF inputs land, then auto-runs LocaNMF + register + figs (one-command overnight) |
| `--dry-run`     | print the plan for both stages; write nothing |

By default Stage 2 **defers automatically** until the date is registered in `configs/sessions.yaml` (see the
gate note above), so on a fresh night you normally just run `nightly <DATE>` and it does the camera/behavior
work, then run `nightly_figs <MMDD>` (or `nightly <DATE> --figs`) after LocaNMF + registration.

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
