# Handoff — 2026-10-20 (evening): within-spout check, mixed model, 3-D contest queued, PS92_0908 rerun

**START HERE for the encoding model / target-vs-executed work (the ENCODING window).** Supersedes
`STATUS_2026-10-20_ENCODING_HANDOFF.md` (morning; still the reference for the drop-p / polar / slow-band decisions).
The pose / 3-D / cam1 work is the OTHER window: `STATUS_2026-10-20_POSE3D_TIMING_HANDOFF.md`.
**Results live in `docs/ENCODING_FINDINGS.md`** (status-tagged; update it when a result changes). Reasoning + numbers:
`DECISIONS.md` 2026-10-20 entries. Scratchpad (launchers, logs, 3-D input tables):
`C:\Users\SABATI~1\AppData\Local\Temp\claude\C--Users-SabatiniLab\cdd1e016-6bb3-4346-8600-f08116ccf61d\scratchpad\`
(`S` below). Output root: `M:\MICROSCOPE\Priya\Widefield\LP_analysis\lp_r4_heatmap_20261013__clean_v36\imaging_encoding\`
(`E` below).

## 0. Rules (Priya; unchanged)

- **Curated sessions only** (`analysis_kit.curated_labels()`): no early June, no 0805, no PS92 / PS93 0817 ("stroke
  too mild"); June 0606-0608 sensitivity only. August-only pre baseline is primary. Check new lists against it.
- Lesions LEFT; contra = mouse's RIGHT; brain maps image left = left (ipsilesional) hemisphere; cam1 video ipsi = image
  right. Lateral angle sign: + = toward the mouse's LEFT (ipsilesional).
- Lead with drop dR2; per-animal tables before pooled (rule 8); **do not dismiss bilateral (both-hemisphere) changes**.
- Priya runs O2 / transfer / process-stopping commands (give `Get-CimInstance` + `taskkill /PID <pid> /T /F`). Write only
  under `MICROSCOPE/Priya`; never delete originals (superseded outputs are COPIED aside). Stage files by name. Commit +
  push; DECISIONS entries.

## 1. Findings this afternoon (details: ENCODING_FINDINGS + DECISIONS 2026-10-20)

| finding | status |
|---|---|
| Within-spout deviation check (`scripts/within_spout_deviation.py`), LICK-aligned: with the spout fixed, the CROSSED (left - right) SSp-m response follows the trial's executed lateral deviation; ~10-20 % of the across-spout slope pre-stroke (SSp-m t = -6.5 over 31 pre sessions; MOs t = -3.2) | STANDS (cam4 lower bound) |
| Mixed model (`scripts/within_spout_mixed.py`; animal random intercept + post step; post + post:log(day)): crossed within PRE level significant with animal as unit (SSp-m q .03, MOp q .003, MOs q .04); NO post-stroke term survives FDR | STANDS |
| BILATERAL within-spout component: PS94 / PS95 pre = medial licks evoke more (eccentricity), acute = contralesional-directed licks evoke more at every spout side (not contact / reward / latency / lick count); PS94 flips positive subacute / chronic; PS92 / PS93 acute ~0 (incl. their first acute sessions) | PROVISIONAL, animal-specific |
| Repo map tests (beta_maps `significance_contour` nested bootstrap = primary, cluster permutation = second line; `vs_zero_contour`): between-spout crossed map vs 0 = 28 % of brain pre; NO post-vs-pre change in within or between maps at n = 4 | STANDS |
| Cue alignment drops delayed trials: acute far-contra first lick (cam4 keypoint onset) median 1.2 / 1.0 s in PS92 / PS93, 35-38 % of trials after 1.5 s | STANDS -> lick-align |
| Slow band (morning): rescales pre-cue decode .03-.09 in every epoch; PS93's acute pre-cue drop = slow band | DONE |
| 3-D landing geometry: tongue peak lands ~2 mm deeper than the spout table and laterally compressed (~0.4x close, ~0.6x far) -> raw nearest-spout bins fail (PS92 pre agreement 0.46); per-animal PRE-stroke calibrated landing zones fix it (pre 0.70-0.77, acute 0.50-0.56 with contra licks landing in the ipsi-neighbouring zone, PS92 chronic 0.84) | STANDS (descriptive) |

## 2. Decisions this afternoon (all in DECISIONS)

- **Lick alignment is the default** for the within-spout check (first lick + 0-1 s, pre-cue baseline, latency covariate).
- **Bilateral and crossed components are both read** (Priya). Per-animal maps; animal-as-unit tests.
- **Map significance = the repo's beta_maps tests** (nested bootstrap primary, cluster second); per-pixel FDR is
  supplementary only. Primary statistics at AREA level (mixed model).
- **Direction + toward-midline model** (`dirmid` in within_spout_deviation): separates "follows direction" from
  "prefers medial licks"; **split-half lick-angle reliability** brackets the executed share (uncorrected = lower bound,
  / reliability = upper bound).
- **Binned contest uses SOFT executed bins on pre-stroke-calibrated 3-D landing zones** (`COMBINED_ENCODING_BINS=1` +
  `COMBINED_ENCODING_BINS_CALIB=<table>`; `scripts/exec_bins_calibration.py`, leave-one-session-out for pre sessions).
  Hard nearest-zone confusion kept only as a behavioural readout (`res["bins"]`).
- **3-D contest = polar (coordinate-matched angles) + soft bins + COMBINED cam4 + cam1-only lick timing**
  (`COMBINED_ENCODING_LICKS=combined`), nokin model set, all PS92 + PS93 sessions with full 3-D.
- **Pre-stroke lick TEMPLATE test** (`scripts/lick_template_test.py`, Priya "yes"): per-spout lick kernels learned
  pre-stroke, held fixed; post-stroke activity predicted by aimed-spout (target) vs landing-zone (executed, soft)
  templates; CV R2 in the lick window, one gain per hypothesis.
- **PS92_0908 rerun** in all 9 encoding folders (+ June PS92 / PS93 0606 in movement_first + subset-nokin) after the
  template-timing fix; old outputs copied to `sessions/_superseded_pre_template_timing/`.

## 3. RUNNING / QUEUED at handoff (10-20 ~19:00)

| job | where / log | then |
|---|---|---|
| PS92_0908 rerun lane A: movement_first (done 18:55), nokin, nokin_matched (`MATCH=<DLC>/o2/enc_v36_20261019/match.json`), nokin_tr (`SUFFIX=_tr`), June 0606 x2 folders | `S\rerun_0908_laneA.sh` / `.log`; writes `S\laneA.done` | ~12 min per fit |
| PS92_0908 rerun lane B: nodir (done 18:53), trmodels, musall shuf, shufpos-, then droppos- (waits for laneA.done) | `S\rerun_0908_laneB.sh` / `.log` ("== B done") | then re-render all affected figures + per-spout table (§6) |
| within-spout cohort --redo (lick-aligned + dirmid + split-half), 1 worker | `S\within_spout_dirmid.log` (results written at the END) | read `E\qc\within_spout_deviation_lick\dirmid.csv` + report; maps `figures\*_{dir,mid}*` |
| lick-template smoke (PS92 0809 / 0811 / 0814 / 0819) | background task; `E\qc\lick_templates_polar\` | if sane: full run on `S\labels_3d.txt` (§6) |
| **QUEUED 3-D contest** (`S\queue_3d_contest.sh`, log `S\queue_3d_contest.log`): waits for (1) both rerun lanes, (2) the other window's PS93 0814 / 0821 / 0908 / 0911 3-D re-run (kinematics + cam1 events newer than `S\chain_start_epoch.txt`) + 15 min; then `S\prep_3d.py` rebuilds snapshot / spout table / calibration / labels (tag 20261021), smoke-fits the FIRST session alone (aborts if no output), then all sessions, 2 workers | box | out `E\movement_first_subset-nokin_exec-polar_licks-combined_bins\`; ~46 sessions x ~12 min / 2 ≈ 5 h |
| other window: PS93 0814 / 0821 / 0908 / 0911 3-D re-run (were clip-only / missing; full cam1 production has landed), 4 workers | other window | the queue waits for it |
| cam1 LP production on O2 (56830841) | O2 | PS94 / PS95 3-D -> cohort-wide contest |

## 4. To-do (priority)

1. DONE 10-20 20:58 (PS92 chronic per-spout unchanged; PS92_0908 lick-window fits improved, unique executed 0.007 -> 0.026). Was: re-render (`combined_encoding_figures.py --order` each folder; §6) and refresh the per-spout
   table; check whether PS92 chronic per-spout numbers moved; DECISIONS line.
2. Read the within-spout dirmid / reliability results; mixed model on dir / mid (extend `within_spout_mixed.py` to read
   `dirmid.csv`); repo map tests on dir / mid maps (already wired in `figures_repo_sig`).
3. Lick-template test: smoke -> full run (3-D env, `--labels-file S\labels_3d.txt`), then after the PS93 re-run with the
   20261021 tables; mixed model on gain / unique shares.
4. 3-D contest readout when the queue finishes: `combined_encoding_figures.py --order
   movement_first_subset-nokin_exec-polar_licks-combined_bins` (contest + trial-to-trial + binned `u:tgtbin` /
   `u:exbin` / `u:exbin|category`), `scripts/exec3d_pilot_figure.py` (now lists polar; PS92 figure path still
   `..._exec-xyz2/figures/`).
5. Within-spout check in 3-D: `COMBINED_ENCODING_EXEC=polar` + tables (+ CALIB) -> `within_spout_deviation.py --align
   lick` (folder `within_spout_deviation_polar_lick`).
6. First-lick latency cam4 vs cam4 + cam1 vs DAQ contact (cam1 events now final for PS92 / PS93).
7. 3-D JAW: ON HOLD (Priya 10-20: 3-D jaw is unreliable at the close-centre spout -- cam4 occlusion). When revived:
   3-D jaw / tongue traces in the main + kin models, a per-lick 3-D jaw-lateral executed term, jaw deviation in the
   within-spout check.
8. LocaNMF cross-check of the within-spout slopes. Later: pre-cue -> lick-window carry-over decode; tongue traces to
   3-D; non-linear ceiling (deferred).

## 5. Pitfalls (new this afternoon)

- **Windows Python cannot open `/c/...` paths**: env vars / arguments passed to python must be `C:/...` (git-bash `$S`
  paths only for bash itself).
- **Heredocs mangle `\n` in Python strings** (happened again): edit Python with the Edit tool.
- **`_event_cols(amp=...)` CENTRES the modulator** (movement_encoding): amplitude-modulated blocks are already
  deviation-coded -- do not subtract the mean-weighted shared kernel again.
- **Cue-aligned windows silently drop delayed trials** post-stroke (no lick in the window) -> lick-align.
- **Cohort-mean maps hide animal-specific components** (bilateral: PS94 opposite sign); per-pixel FDR favours broad
  shifts and misses focal effects -> per-animal maps + beta_maps tests + area-level mixed model.
- **3-D spout table position != tongue landing point** (rod's lower edge; lateral compression) -> calibrate landing
  zones per animal from pre-stroke licks; never nearest raw spout.
- **3-D for a session can be clip-only** (pose3d `clips` mode before production lands): check lick counts / join
  coverage per session (`S\prep_3d.py` prints frac3d), don't call it "low coverage".
- **Editing a running bash script is only safe after the current line** (bash reads by offset) -- append / change later
  lines only, or restart it.
- **Fits are ~10-13 min each on this box** (one session, one folder); the main movement_first set is the slowest.
- Allen area names end in `_left` / `_right`.

## 6. Commands

```bash
E=M:/MICROSCOPE/Priya/Widefield/LP_analysis/lp_r4_heatmap_20261013__clean_v36/imaging_encoding
# re-render after the PS92_0908 rerun
for o in movement_first movement_first_subset-nokin movement_first_subset-nokin_matched movement_first_subset-nokin_tr \
         movement_first_subset-nodir movement_first_subset-trmodels musall_subset-shuf movement_first_subset-shufpos- \
         movement_first_subset-droppos-; do python scripts/combined_encoding_figures.py --order $o; done
# within-spout (lick-aligned) + mixed model
PYTHONPATH=. python scripts/within_spout_deviation.py --align lick --figures-only
PYTHONPATH=. python scripts/within_spout_mixed.py
# 3-D environment (tables in S; tag 20261020 now, 20261021 after the PS93 re-run)
W=C:/Users/SABATI~1/AppData/Local/Temp/claude/C--Users-SabatiniLab/cdd1e016-6bb3-4346-8600-f08116ccf61d/scratchpad
export COMBINED_ENCODING_EXEC=polar COMBINED_ENCODING_LICKS=combined \
  COMBINED_ENCODING_EXEC3D_TABLE=$W/pose3d_kin_licks_snapshot_20261020.csv.gz \
  COMBINED_ENCODING_TARGET3D_TABLE=$W/pose3d_spout_positions_PS92_PS93_20261020.csv \
  COMBINED_ENCODING_BINS_CALIB=$W/exec_bins_calib_20261020.csv
PYTHONPATH=. python scripts/lick_template_test.py --jobs 1 --labels-file $W/labels_3d.txt
PYTHONPATH=. python scripts/within_spout_deviation.py --align lick --jobs 1 $(cat $W/labels_3d.txt)
python -m pytest -q tests/test_combined_encoding_options.py tests/test_within_spout_deviation.py   # 21 + 5
```
