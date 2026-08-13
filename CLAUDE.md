# CLAUDE.md

Project-level instructions for Claude Code. Read in full at the start of every session in this repo.

`widefield_pipeline` = widefield calcium-imaging **preprocessing + LocaNMF spout-position decode/encode
analysis** for the Sabatini-lab VLS-stroke study (mice PS92/PS93/PS94/PS95; PS93 has a right orofacial
deficit). Carved out of `Widefield_DAQ_recorder` on 2026-08-08 (that repo keeps only the DAQ recorder GUI +
camera acquisition). See `README.md` for setup, `docs/archive/MIGRATION.md` for the split record.

---

## Ground rules (non-negotiable)

1. **Never modify/delete source data.** Raw imaging, DAQ `.h5`, the imaging computer's preprocessing outputs
   on MICROSCOPE (`motion_corrected/`, `wfield_local_results/`, `SVTcorr.npy`, `U_atlas.npy`,
   `*cleanpairs_frame_map.npz`), behavior logs, and the cam1 recovered-position CSVs are READ-ONLY inputs.
   Only ever write inside `MICROSCOPE/Priya/…`; never another person's folder; never delete on MICROSCOPE.
   Never delete local staging (`D:`/`E:`) until byte-verified and the user confirms.
2. **Never modify the `Widefield_DAQ_recorder` repo** from here — it is the separate recorder GUI. Zero
   cross-imports; the two talk only through files on MICROSCOPE.
3. **`configs/*.yaml` is the single source of truth.** Register sessions in `configs/sessions.yaml`, animals/
   colors/date-policy in `configs/animals.yaml`, params in `configs/defaults.yaml`, mounts in
   `configs/paths.yaml`. The hardcoded `SESSIONS`/`ANIMAL_COLOR`/`L`/`D` are RETIRED — do not re-add them.
   MMDD dates are QUOTED strings (`0606` unquoted parses as octal 390).
4. **Rig commit procedure:** `export CONDA_PREFIX=C:/Users/sabatini/.conda/envs/locanmf`; `git add -A` →
   commit → `git fetch origin` → `git rebase origin/main` → `git push`. NEVER force-push; re-fetch/rebase if
   rejected. Both machines push `main`, so always fetch/rebase first.
5. **Cross-day caching:** per-session results are memoized (`wfield_local/session_cache.py`). **Bump
   `CACHE_VERSION` whenever you change a cached function's logic** (mtimes don't see code changes).
6. **Per-machine envs differ by design** (README "Per-machine environments"): imaging box = `wfield` env,
   numba + numpy<2.1; this box = `locanmf` env, numpy 2.2.6, no numba. Deps are lower-bounds-only so
   `pip install -e .` never force-upgrades a working stack.

## Architecture (two machines)

One nightly command, dispatched by machine: **`python -m wfield_local.nightly <YYYYMMDD>`**.
- **Imaging (PCO) computer** — acquisition (recorder GUI, separate repo) + preprocessing (motion/SVD/Allen,
  cue/lick maps) from THIS repo → uploads to MICROSCOPE. Mounts: `N:`=MICROSCOPE, `E:`=local, `M:`=standby
  (`M:\collaborations\Priya\Widefield\labcams`). `nightly` = **`archive_day upload-daq`** (push the DAQ `.h5`
  to N: FIRST so the analysis box's behavior pipeline can start in parallel) → `preprocess` →
  `preprocess_deck` → `archive_day archive` (COPY; never deletes E:). Runbook: `runbooks/imaging_computer_nightly.md`.
- **Analysis / behavior GPU box (this)** — LocaNMF + decode/encode/RSA + decks. `M:`=MICROSCOPE. `nightly` =
  `camera_nightly` (upload D: cameras+behavior-logs → MICROSCOPE, never deletes D:; dropped-frame QC;
  camera↔DAQ alignment templates) FIRST, THEN `nightly_figs` (LocaNMF; waits on the imaging push). Runbook:
  `runbooks/analysis_computer_nightly.md`. Never auto-deletes local staging (E:/D:) — cleanup is manual.

## Key decisions (see DECISIONS.md for the full set)

- Decode = multinomial logistic regression on individual LocaNMF components, first-lick 2 s, NO per-trial
  baseline, block-CV (GroupKFold by ~6-trial position blocks), chance 0.167. SSp dominates, MO secondary.
- Pre-cue no-lick decode above chance = PRE-CUE POSITION INFORMATION, present before the cue and not lick-driven (the key pre-stroke readout). Deliberately NOT called a "maintained motor plan": the spout arrives ~3 s before the cue, so a sustained sensory response and a held intention are temporally coextensive and this design cannot separate them. It does not need to -- a pre-cue position signal that changes post-stroke is the readout either way (Priya, 2026-08-13).
- Positions from DAQ strobe bits; dead bit1 (Aug-2026) auto-repairs from the behavior log
  (`classify_cues_with_backup`), or a `behavior_trials` recovered CSV when the log is empty (PS93 8/5, cam1).
- **BEHAVIOR trials also come from the DAQ** (`daq_trials.py`), not the GUI log — the GUI mislabels
  `pos_idx` on ~15% of trials (docs/GUI_TRIALS_LOGGING.md). Both halves of the pipeline now resolve
  positions from the same strobe codes with the same time-pairing rule.
- ~~The DAQ cue/strobe stream tracks the REWARDED subset~~ **CORRECTED 2026-08-09:** it does not — DAQ cue
  count equals the log's scored-trial count exactly in every session, and includes unrewarded trials. There
  is no reward-based subsetting; the disengaged tail is removed by `flag_engagement` (a REPORTING gate on
  the full trial table), and unrewarded trials stay available for the post-stroke failed-attempt analysis.
- Cross-session comparisons use the CURATED set: 6/6-6/8 + 8/6 onward (exclude noisy early June 6/1-6/5 and
  the wonky 8/5), from `configs/animals.yaml date_policy`. Crossnobis is the noise-unbiased RDM metric.

## Restructure roadmap — mirror `../stroke_orofacial_pipeline`

Target = that repo's mature config-driven layout. **DONE:** `configs/{animals,sessions,paths,defaults}.yaml`
(animals now carries full cohort metadata: sex/DOB/genotype/stroke_laterality/reference_landmarks);
config loader (`wfield_local/config.py` ≈ their `config_loader.py`+`animals.py`); `tests/` (22);
**PathResolver** (`wfield_local/paths.py`, machine-aware analysis=M: / imaging=N:/E:) + `sessions.yaml`
migrated to root-relative resolving on both boxes (was roadmap #2); **config-driven preprocessing
orchestrator** `wfield_local/preprocess.py` (was roadmap #4 — retires the per-date `_nightly_*`/`_mc_svd_*`/
`_maps_*`/`_photobleach_*` drivers; photobleach moved to `wfield_local/photobleach.py`); installable
packaging; `README`; `runbooks/`; incremental per-session caching; `docs/archive/MIGRATION.md`; this `CLAUDE.md`.

**NEXT (priority order):**
1. **Consume `defaults.yaml` in code.** LocaNMF params (r2/loc/maxrank), decode windows/CV/max_rt, sync
   params are still hardcoded in module `_args()`/constants — wire them to `config.defaults()` so params live
   in ONE place (preprocess params already consumed by `preprocess.py`). Add `session_overrides.yaml`
   (per-session param overrides layered on defaults), mirroring their defaults+overrides pattern.
2. **`.githooks/` DONE** — `pre-commit` (ruff on staged `.py`) + `pre-push` (pytest); enable per clone
   with `bash scripts/setup-hooks.sh`. Quality gates ONLY (no branch protection — both machines commit
   direct-to-main per rule 4); interpreter resolved from `CONDA_PREFIX` to dodge the App Store shim.
3. **Legacy one-off fold-in DONE** — `_crossday_intensity.py`→`wfield_local.crossday_intensity` (run once
   after `xall` in `preprocess`; `preprocess_deck` consumes its PNG). Earlier:
   `_nightly_*`/`_mc_svd_*`/`_maps_*`/`_photobleach_*`→`preprocess`; `_xall_refresh`→`preprocess.refresh_xall`;
   deck→`preprocess_deck`; standby transfer→`archive_day`. `_build_xsession_deck`/`_redo_motion_all` retired;
   `_qc_from_standby` moved to `scripts/`. No root `_*_run.py` drivers remain.
4. **Optional `src/` layout** — move `wfield_local/` under `src/` and/or split into submodules like their
   `src/pkg/{alignment,figures,stats,…}`. KEEP the `wfield_local` import name (both machines + docs depend
   on `python -m wfield_local.*`).
5. **Write-guard DONE** — `wfield_local/writeguard.py` `assert_writable(path)` refuses writes/deletes that
   land on the MICROSCOPE/standby shares outside the Priya subtree (location-based, zero false-positive;
   wired at `preprocess` push-rmtree + `preprocess_deck` stale-delete). Call it before any new
   MICROSCOPE write/delete site (`archive_day` adoption is a reasonable follow-up). STILL OPEN: **exclusions
   with dotted-tag scoping** (their `Exclusion.applies`) for per-analysis-context date/animal exclusions.

**Nightly-pipeline extensions (mirror stroke_orofacial where noted):**
- **Temporal alignment templates — DONE** (`wfield_local/camera_sync.py`, one per cam per date). Each
  Blackfly's Bonsai CSV is (frame_id, timestamp_ns, GPIO); GPIO bit0 = the Arduino sync heartbeat, which also
  lands on DAQ digital `sync` (bit0 of `packed_samples`, 5000 Hz). `camera_sync` matches the two rising-edge
  trains with the **bounded-window** ITI matcher in `frame_sync.align_edge_sequences` (faithful port of
  stroke_orofacial; NB the prior widefield copy was a degraded O(N²) all-pairs form — fixed to O(N·window),
  ~4 s vs ~9 min at 12875 edges) and writes a COMPACT `.npz` mapping camera TIME→DAQ time (affine, drop-proof:
  built on absolute timestamps, so an ITI-dropped frame just removes an anchor). Improvements over orofacial:
  a residual `quality_ok` gate + reported edge-count delta + B6 `frame_drops`, and a dropped-frame test suite.
  The PCO imaging cam needs no template (on the DAQ clock via `pco_exposure`); its gap is dropped-frame
  count-reconciliation in `trim_illuminated_labcams.py` (LED-parity, not ITI). Wired into the camera nightly
  orchestrator `wfield_local/camera_nightly.py` (`python -m wfield_local.camera_nightly <DATE>` = upload
  `D:`→MICROSCOPE [size-verified, never deletes `D:`; like `archive_day` for the imaging box] → dropped-frame
  QC → alignment templates). `--skip-copy`/`--skip-dropframe`/`--skip-align`/`--dry-run`.
- **Camera dropped-frame QC — DONE** (`wfield_local/dropframe_qc.py`, folds in the local
  `dropframe_check_all.py`): per Blackfly cam CSV, flags gaps in the monotonic frame_id + long timestamp
  deltas; writes `dropped_frames_summary_<DATE>.{csv,txt}`. Byte-verified against the existing 20260807 CSV.
  STILL OPEN — the **sync-train** cross-check: match each cam's GPIO train + the behavior-log event/ITI train
  against the DAQ sync line to localize dropped DAQ samples / frames (all derive from the one Arduino source).
  That is the same DAQ↔GPIO alignment as the Blackfly templates above (raw sync line, not the REWARDED cue
  subset — see the June count-mismatch resolution).
- **Nightly behavior-session figures from spout data — DONE** (`wfield_local/spout_behavior.py`, a la
  stroke_orofacial `spout_behavior`; 1 cue + 6 spout positions here vs their 2 cues + L/R).
  **Trials come from the DAQ recorder `.h5`** (`wfield_local/daq_trials.py`), NOT the task-controller
  log: the GUI's `trials.csv` mislabels `pos_idx` on ~15% of trials (every position-change trial — its
  a trial_start collides with the id already on the open row, so it never closes and gets overwritten
  with the NEXT trial's position; fixed upstream in mobile_spout_behavior bb16533, but every session
  recorded before that deploys is affected). The IMAGING analysis was NOT affected (its offset aligner
  absorbs the shift; see the doc). Position comes from the
  `spout_strobe`+`spout_bit0/1/2` code the firmware emits after the move and before the cue; hit/miss/
  latency are scored from DAQ licks over the session's REAL response window (**3500 ms**, read per
  session from `gui_config.json timing.response_window` — `defaults.yaml`'s 2.0 s was never the task's
  window and is now only a fallback). The log stays the FALLBACK (gated by `daq_trials.quality`: the
  8/5–8/6 dead-`spout_bit1` sessions collapse to 4 positions and fall back) and still supplies the
  free-reward designation. **Never DAQ-only.** Full write-up:
  [`docs/GUI_TRIALS_LOGGING.md`](docs/GUI_TRIALS_LOGGING.md).
  Per-session figure — row 1 = task performance (2x3 spatial hit-rate grid,
  per-position accuracy bars w/ Wilson CI + raw overlay, engagement-over-session timeline, first-lick
  latency by position); row 2 = the by-position lick metrics (licks/trial, within-trial
  lick rate, anticipatory licks) — **the same metric families the across-session per-animal figure
  tracks**, so one session reads directly against the cross-day trend — plus per-position metrics CSV,
  and a curated cross-session cohort figure (per-animal per-position accuracy,
  learning curve, close-vs-far distance effect). **Engagement gate** (`flag_engagement`): reward is
  auto-held after a miss run, so a sated animal's late misses are DISENGAGEMENT, not spatial inaccuracy —
  a terminal sated-tail detector (>= `tail_min_misses` trailing non-responses) UNION a rolling
  response-rate collapse gate separates them; per-position accuracy is engaged-gated by default with the
  raw all-trial rate shown alongside. Params in `configs/defaults.yaml behavior.*`; figures ->
  `behavior_out` root (`Behavior_logs/Widefield/behavior_summary/`), per-session under
  `sessions/<animal>/<date>/`. Wired as camera-nightly step 3 (`--skip-behavior`); standalone
  `python -m wfield_local.spout_behavior <DATE> [--cohort] [--from curated]`. Also emits: a
  **lick-microstructure** figure per session (peri-cue raster + PSTH, ILI distribution, lick bouts,
  within-trial lick rate + licks/trial + anticipatory (pre-cue ENL-reset) licks per position) with a
  **GUI-vs-DAQ-pipeline** lick comparison panel. The **per-position** lick aggregates are
  engagement-gated exactly like per-position hit rate (a sated tail would otherwise read as a spatial
  effect); session-level scalars and the raster/PSTH stay over the whole recording; and **per-animal across-session** figures
  (`cohort/by_animal/<animal>_across_sessions.png`) tracking every per-position metric over days
  (color=ring, marker/linestyle=side so all 6 positions are distinguishable). Aborted runs
  (< `min_session_trials`) are auto-skipped.
- **Lick-detection physiological floor (pipeline-wide).** `configs/defaults.yaml lick_detection.min_ili_ms`
  (40 ms) is a mandatory min-inter-lick-interval applied in `detect_licks` as `max(min_ili, refractory)`.
  These mice lick 5-7 Hz (peak ~9-11 Hz); 40 ms (25 Hz) removes only 0-3 physiologically-impossible
  sub-40 ms doubles/session (a single contact split by a 1-sample voltage glitch — the true onset is kept;
  QC: `behavior_summary/qc/`), and is a **no-op for imaging** (its 0.1 s bout-collapse refractory subsumes
  the floor, so lick-aligned maps are unchanged). Wired into every lick consumer (behavior + imaging) via
  config, so lick identification stays consistent across the pipeline.

- **Canonical DAQ behavior events — DONE** (`wfield_local/behavior_events.py`): one `.npz` per session
  (`behavior_summary/events/<animal>/<date>.npz`) of licks/reward/running-bouts/quiet-periods on the DAQ
  clock, so behavior AND imaging load the SAME identity instead of re-detecting. `get_or_compute()` is the
  consumer entry; params in `configs/defaults.yaml segmentation` (+ `lick_detection`). Wired: analysis
  nightly (camera_nightly step 3) produces them; imaging `preprocess` maps step consumes them for the
  **quiet-vs-running SVD activity maps** (`plot_running_activity_maps.py` → quiet / running / running−quiet,
  in the deck before QC/photobleach). Running threshold 3 mm/s (sedentary mice; see memory). To unblock the
  analysis box, the imaging nightly pushes the DAQ `.h5` to N: up front (`archive_day upload-daq`).

**Then the science:** post-stroke prerequisites — per-trial behavioral-state table (spout-contact + DAQ lick
→ hit/miss/failed, latency, executed position), and packaging the frozen pre-stroke model + baseline
noise-floor — followed by the post-stroke intention-readout (frozen decoder) and representational-similarity
(crossnobis / encoder-residual) analyses. Design in `DECISIONS.md`.

## Hemodynamic-correction VARIANTS — naming rule (2026-08-13)

The pipeline's `SVTcorr.npy` removes slow drift with a **zero-phase (acausal) 0.1 Hz Butterworth**, and
the filtered blue channel is what becomes the output. Measured over all 36 curated sessions this
**inflates PRE-CUE decoding by ~0.21** while leaving post-cue unchanged or better, because the filter
smears each post-cue response backwards in time (DECISIONS.md; `wfield_local/filter_acausality_test.py`
reproduces it). `wfield_local/hemo_variants.py` builds alternatives.

**Nothing overwrites the originals.** `SVTcorr.npy` / `T.npy` / `rcoeffs.npy` stay put — they ARE the
`zerophase` variant. Every alternative goes in its own subdirectory beside them:

```
<session>/motion_corrected/wfield_local_results/
    SVTcorr.npy  T.npy  rcoeffs.npy      <- ORIGINAL pipeline output, never touched
    hemo_<variant>[_refitT]/             <- one DIRECTORY per variant, never a bare file
        SVTcorr.npy  T.npy  rcoeffs.npy  manifest.json
```

- `hemo_<variant>` — the drift-removal variant (`fitonly`, `causal`, `taskdetrend`, `strobedetrend`,
  `meegkit`; `python -m wfield_local.hemo_variants --list`).
- `_refitT` suffix — the hemodynamic coefficients were **refitted on the drift-removed traces**.
  Without it, the saved `T` was reused. That distinction matters: reusing `T` is correct for a
  controlled COMPARISON (it isolates the filter as the only changed variable) and wrong for a PRODUCT
  (it applies a high-pass-derived transform to detrended data). `refit_T` is validated against the
  saved coefficients — fed the same high-passed traces it reproduces them to 1.5e-6.
- `manifest.json` records variant, drift method, mask spec + surviving fraction, refit flag, params.

**Consequence for readers:** a bare `SVTcorr.npy` is the original by construction, and any result built
on a variant must carry the variant string so two of them can never be silently mixed.

## Reference
- `configs/` — source of truth. `wfield_local/config.py` — loader. `wfield_local/nightly_figs.py` — orchestrator.
- `runbooks/` — the per-machine nightly prompts (Priya's canonical prompts + notes).
- Env: `C:/Users/sabatini/.conda/envs/locanmf/python.exe`; repo is `pip install -e .` (no PYTHONPATH).
- Deck: the CURRENT analysis deck is `locanmf_analysis_deck.py` → auto-built by `nightly_figs` into
  `<labcams>/spout_position_analysis_summary.pptx`. Sections: **A–C within-day** (decode, encode,
  pre-cue lick-free control), grouped animal→type→date; **D cross-session/frozen**, grouped
  basis→alignment→animal and run in TWO bases (Allen-ROI and the shared joint-LocaNMF basis, see
  `joint_xsession.py`); **E–F** cohort summary + RSA. The curated date set is DERIVED
  (`config.curated_dates()` = registered minus `cross_session_exclude`) — the static
  `date_policy.cross_session` list was deleted 2026-08-13 because it lagged five nights and silently
  shrank hand-run decks;
  analysis PNGs land in `…\labcams\locanmf_lick_pooled\cue_analysis\`. The superseded
  `locanmf_decoder_ppt.py` / `locanmf_xsession_deck.py` builders were DELETED on 2026-08-13 (707 lines,
  reachable only via a `--ppt` flag the nightly never passed; recover from git history if ever needed).
  Their spec lives on in `docs/archive/LOCANMF_XSESSION_DECK_SPEC.md`, marked retired.
