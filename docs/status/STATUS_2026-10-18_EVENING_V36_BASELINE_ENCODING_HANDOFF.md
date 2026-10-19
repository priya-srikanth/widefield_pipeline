# Handoff — 2026-10-18 evening: LP run v36, August-only baseline, encoding-model redesign, maps / ME in flight

**START HERE.** Supersedes `STATUS_2026-10-18_OROFACIAL_W2_ENCODING_HANDOFF.md` (still the reference for the cohort
pipeline, the W2 design, the pre-v47 log fix and CURBD / ARHMM discussion). Reasoning and numbers: `DECISIONS.md`, the
five 2026-10-18 entries (evening + late). Orofacial results: `docs/OROFACIAL_FINDINGS.md` (banner: being re-checked on
v36 against the August-only baseline).

## 0. NIGHT UPDATE (10-18 ~21:00) -- read first; details in DECISIONS "2026-10-18 (night)"

**Running (detached; scratchpad `C:\Users\SABATI~1\AppData\Local\Temp\claude\C--Users-SabatiniLab\cdd1e016-6bb3-4346-8600-f08116ccf61d\scratchpad\`):**
| job | log | then |
|---|---|---|
| FULL v36 encoding run, final model (c0b2c45: drift removal + target-on-lick + distance), 119 sessions, `--jobs 6`, PID 24016, started 20:49, ~4.5 min / session -> ~06:00-07:00 | `encoding_v36_full.log` | `python scripts/combined_encoding_figures.py` (August-only default); per-animal readout `enc_peek2.py`; maps `ps92_maps.py <ANIMAL>` |
| VIDEO pilot (23 sessions, cam1 + cam4) -- queued: `queue_video_pilot.ps1` waits for PID 24016 | `video_pilot_queue.log`, `encoding_video_pilot.log` | outputs `imaging_encoding/movement_first_video-cam1-cam4/`; compare position / task / target / executed unique WITH vs WITHOUT video, ENL vs lick window |

Stop command (Priya runs it; matches by command line):
`$p=@(Get-CimInstance Win32_Process -Filter "name='python.exe'" | ? { $_.CommandLine -match 'wfield_local.combined_encoding --all' }); $k=@(Get-CimInstance Win32_Process -Filter "name='python.exe'" | ? { $p.ProcessId -contains $_.ParentProcessId }); $w=@(Get-CimInstance Win32_Process -Filter "name='powershell.exe'" | ? { $_.CommandLine -match 'queue_video_pilot' }); ($w+$p+$k) | % { Stop-Process -Id $_.ProcessId -Force }`

**Done since the evening handoff:** August-primary regeneration + both orofacial decks (110 slides; speaker notes on
every slide BUILT AT DECK TIME from run.json / defaults.yaml / module docstrings + constants / stats tables;
`oc.stats_methods()`); O2 ME pilot collected (46 / 46, QC clean, unmasked by the 10-06 design); encoding: post-fit crash
fixed, smoke mode (`COMBINED_ENCODING_SMOKE=1`), video variable (`--video`), target-vs-executed panel, `lickboth`,
collinearity fixed, slow-drift removal; `scripts/encoding_drift_check.py` (figures in `imaging_encoding/qc/`); resting-
jaw per-animal check (chronic shift NOT consistent: +7 / +9 / +2.6 / -3 px).

**New to-dos (priority):**
1. When the run finishes: cohort target-vs-executed (`target_lick` vs `executed`, unique from `lickboth`), per-animal
   first (rule 8), maps; then the figures script and an encoding deck.
2. Video pilot readout (03:00+ start): with vs without video; line the ENL result up against the ENL movement-removed decode.
3. Decide (Priya): adopt the TWO-BAND hemodynamic correction? (prototype: PS92 0824 slow share 87 -> 21 %, fast band
   identical). Would need the variant built for all sessions + decode numbers checked unchanged.
4. Check PS92's subacute imaging dip without / with PS92_0824 (its early-session correction residual).
5. Earlier list (§4 below) still stands from item 2 on.

**New pitfalls:**
- **The adopted preprocessing keeps slow drift (> ~20 min) by design** -- fine for trial contrasts, fatal for any
  continuous-time model with contiguous-block CV (negative R2). Remove slow drift INSIDE such a model (FWL projection),
  never by re-adding a 0.1 Hz filter.
- **The early-session hemodynamic-correction residual** (within-session bleaching, one coefficient per pixel fitted on the
  0.1 Hz band) can be several SD in the corrected signal while the raw channels look normal.
- **Variance explained cannot answer "is there position information beyond movement"** -- video absorbs the shared
  variance; use the residual DECODE.
- **Long single-session tests: use the smoke mode first** (a post-fit bug cost a 40-min run).
- Bash heredocs on this box mangle `\n` / `\\` in Python string literals: write patch scripts with the Write tool.
- Stopping processes is denied to Claude here: hand Priya the command-line-matched stop command (wrap in `@()`).

## 1. Rules now in force (do not undo)

1. **LP run `lp_r4_heatmap_20261013__clean_v36` replaces v35.** June 0602-0608 cam4 pose x 0.8853 before cleaning (no-lick
   ENL spout ruler; Aug+ bit-identical to v35, verified on cleaned arrays, windows, lick tables). v35 carries
   `SUPERSEDED.md`; its decks are renamed `SUPERSEDED_v35_*`. `clean.npz` / manifest record `px_scale`.
2. **AUGUST-ONLY pre baseline is PRIMARY** for every orofacial analysis and the encoding model (Priya). Code default
   `--baseline late`; primary outputs in the PLAIN folder names; `--baseline all` -> `*_allpre` (each folder has
   `SUPERSEDED.md`). All-pre may be quoted only for scale-free measures, as a sensitivity check. Anything measured from
   the mouth origin (jaw-from-mouth, nose, spout) is August-only.
3. **Imaging keeps its default baseline** (all curated pre). The August-only imaging run confirms it (§2c).
4. **Windows / gates:** heatmaps cue .. trial stop; W2 cue + 70 ms .. stop; both cut each trial at spout-retraction onset
   (> 30 px and staying off; bites only on PS92 / PS93 0606 and 5 trials of PS94 0818). The 15 px whole-trial spout gate
   was RETIRED (it dropped licking / water-drop trials). W2 spout control = no-lick ENL frames (cue - 2 s .. cue, tongue
   not detected). Engaged trials only, everywhere.
5. **Encoding model** (`wfield_local/combined_encoding.py`, DECISIONS 10-18 late): own penalty per reduced model, drop
   dR2, imaging-map masks, shared cue + arrival + deviation-coded position (one variable), lick -0.5..+2 s, no reward,
   target-vs-executed variants with / without tongue x / y. The first all-session run is INVALID.

## 2. Findings today

**a. June camera geometry.** Spout spacing 12.9 % larger in June in all four animals (Aug+ 87.44 +- 0.69 px over 107
sessions). After rescaling, June still differs in PERSPECTIVE: head-only ruler (nose to jaw rest) 1.17-1.24x vs spout
plane 1.13x; no 2-D transform fits (spout similarity residual already ~1-2 px). Hence rule 2.

**b. What the baseline choice changes (v36):** angles ~nothing (far contra acute +29.0 vs +29.8 deg); protrusion px same
size (7.4 vs 7.5 px median |change|); contact fraction inflated by all-pre (0.11 vs 0.02; real June -> August learning).
Spout control now flat (floor 0.36 all-pre ~ 0.41 August-only; post 0.24-0.34). Jaw rest-relative floor 0.37-0.39 under
both baselines (genuine session-to-session rest jitter); jaw-from-mouth far-contra floor 0.41 all-pre vs 0.25
August-only (June head placement). Nose heatmap changes do not exceed its floor (0.72-0.82) -> posture rests on W2
(August: pre 5.6 vs post 7.4-9.2 px), still "candidate".

**c. Imaging, August-only vs default:** no headline change. Frozen-decoder position accuracy acute far contra 0.32 ->
0.25 (same pattern, slightly deeper); state decoder acute 0.915 -> 0.886 (specificity holds); encoder frozen vs ceiling
unchanged (0.100 -> 0.104); crossnobis distances 5-10 % smaller, all still significant. Epoch families: 703 cells
significant under both, 36 lost, 122 gained (`labcams/grant_figures/epoch_latepre/baseline_compare.csv`,
`scripts/imaging_baseline_compare.py`). MAP families being rendered now (§3).

**d. Spout "movement".** The spout moves only after the DAQ trial stop (except the 0606 early retraction). Within-window
offsets are licking push (2-5 px), tongue occlusion, and WATER DROPS on an untouched spout shifting the tip label (frame
gallery in the session scratchpad) -- not grooming. Engagement gating removes most late sated-tail drop trials (PS95 0924
24 -> 5) but not PS94 0818 (drops form while working).

**e. Encoding-model problems** (first run): alone < unique for instructed / jaw (penalty borrowing); running collinear
with instructed (min subspace angle 0.10-0.13); raw brain mask; 2-s cue kernel cut-off. All fixed (DECISIONS).

**f. Lick-phase in-mouth incl / excl figures were identical by construction** (0.04 % in-mouth samples in phase 0.1-0.9);
the included version was dropped. Their significance bars differed only through near-threshold clusters.

## 3. RUNNING at handoff (scratchpad logs: `C:\Users\SABATI~1\AppData\Local\Temp\claude\C--Users-SabatiniLab\cdd1e016-6bb3-4346-8600-f08116ccf61d\scratchpad\`)

| job | log | then |
|---|---|---|
| August-primary regeneration: jaw x2, cohort, cohort --all-trials, robustness, decks (primary + `_allpre_SENSITIVITY`) | `aug_primary.log` | check `!!`; decks in `<v36>/cohort/deck/` |
| Encoding-model timed test, PS92_0811 (final design) | task output `bj2hrln56` | check alone >= unique, cue / position split, target vs executed shares; set `--jobs` from the timing; then `python -m wfield_local.combined_encoding --all --jobs N --run lp_r4_heatmap_20261013__clean_v36` (NOT --skip-done: the v35 copies in the v36 folder are invalid -- delete-free: the run overwrites) |
| Imaging map families 14m / 15e / 15r, August-only, -> `labcams/grant_figures/epoch_latepre/` | `imaging_latepre_maps.log` | compare with `grant_figures/epoch/` figure by figure (areas of the acute far-contra drop, the near > far ipsi > far mid > far contra ordering, contours vs rim rule) |
| O2 video ME pilot `me_pilot_20261018` (46 tasks, cam1 + cam4, 23 sessions) | O2 `logs/`; Priya rsyncs `out/` | `python scripts/o2_video_me.py collect --name me_pilot_20261018`; check components / coverage; add video LAST (orthogonalised) to the encoding model; pilot comparison with vs without video |
| O2 ViT-B (56700781), cam1 LP r2 (56709413) | O2 | other window owns cam1; ViT-B: score when `DONE` |

NOTE: the v36 `imaging_encoding/movement_first/sessions/` folder holds 60 outputs COPIED from v35 (identical inputs,
`PROVENANCE_v35_copy.json`) -- produced by the OLD model, so invalid; the full re-run overwrites them.

## 4. To-dos (priority order)

1. Encoding: validate the test -> full v36 re-run -> `python scripts/combined_encoding_figures.py` (August-only default;
   `--baseline all` for sensitivity) -> add the target-vs-executed panel (shares per epoch, with / without tongue x / y,
   exec_angle unique) and per-position one-vs-rest / response maps / side-vs-distance maps to the figures script; decide
   deck home (own encoding deck suggested). Then `--order task_first` sensitivity.
2. Rewrite `docs/OROFACIAL_FINDINGS.md` numbers from v36, August-only primary (`cohort/tables/{prepost,w2,jaw_lateral}`,
   `robustness/robust_cells.md`), leave-PS93-out column; remove the "both baselines" selection rule.
3. Heatmap contours: tighten to "every animal agrees AND |delta| > 95th pct of the keypoint's floor map" (offered;
   Priya not yet answered) -- figure-only re-render.
4. Per-animal check of the chronic resting-jaw shift (jaw-from-mouth W2 chronic 15.5 vs pre 8.3 px, August).
5. Imaging maps comparison (§3) -> DECISIONS; `chronic_drift_vs_reorg` needs the lag-matched rework (within-August vs
   within-chronic pairs at 1-8-day gaps), not a baseline swap.
6. ME pilot -> decision on all-session ME (O2).
7. Older: jaw-only example gallery (to-do 7), kinematic suggestions (to-do 9), MVAR -> CURBD pilot on encoding residuals,
   ARHMM decision, production pose model (ctx-semi 640 vs LP r4) after ViT-B.

## 5. Pitfalls (new today)

- **June is not just scaled:** perspective differs (head vs spout plane). Never pool June pre for absolute geometry; the
  August-only baseline is the rule, not a sensitivity.
- **A pre-vs-pre leave-one-out delta averages to EXACTLY zero** -- use the later-minus-earlier half split as the floor.
- **"Every animal agrees in sign" is not a test** (~1 bin in 8 by chance with 4 animals) -- contours appear in the floor
  column too.
- **Spout tip label moves with water drops and tongue occlusion**; it is not the spout moving. Do not gate trials on it.
- **Reduced models must not borrow the full model's penalties**; and with movement-first, shared variance goes to the
  earlier group -- always show the drop dR2 beside the shuffle dR2.
- **Do not restart long single-session tests for every small code change** -- batch the changes; a run_session with the
  full model set takes many minutes (timing pending).
- **Encoding module reads the LP run via `--run` / `COMBINED_ENCODING_RUN`**, not `orofacial_cohort.RUN`.
- **O2 sbatch needs `#!/bin/bash -l` + `conda activate`** (plain shell + `source activate` fell back to the system Python).
- **Share listings can hang while many jobs read M:** -- keep listing commands narrow or background them.
- **`orofacial_robustness_table.py` reads `prepost_allpre/`** (it needs the June rows).
- Older lists still apply (10-18 morning handoff §5).

## 6. Commands

```bash
# dlc env: /c/Users/SabatiniLab/miniconda3/envs/dlc/python.exe ; locanmf env: .../envs/locanmf/python.exe
python -m wfield_local.orofacial_outputs process --bundle lp_r4_prod_20261015 --jobs 10     # -> run v36
python -m wfield_local.orofacial_jaw_lateral --jobs 6 [--all-trials] [--baseline all]
python -m wfield_local.orofacial_cohort --jobs 8 [--all-trials] [--baseline all]
python -m wfield_local.orofacial_w2 --jobs 6 [--baseline all]
python -m wfield_local.orofacial_heatmaps --jobs 6 [--baseline all]
python scripts/orofacial_robustness_table.py
python -m wfield_local.orofacial_deck [--baseline all]                                      # locanmf env
python -m wfield_local.combined_encoding --all --jobs N --run lp_r4_heatmap_20261013__clean_v36   # locanmf env
python scripts/combined_encoding_figures.py --run lp_r4_heatmap_20261013__clean_v36 [--baseline all]
WIDEFIELD_PRE_FROM=0801 python -m wfield_local.epoch_grant_figures --only <families> --output <labcams>/grant_figures/epoch_latepre
python scripts/imaging_baseline_compare.py
PYTHONPATH=. python scripts/o2_video_me.py collect --name me_pilot_20261018
```
