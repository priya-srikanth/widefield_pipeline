# Handoff — 2026-10-18: cam4 orofacial cohort, W2, baselines, imaging sensitivity, Musall-style encoding model

**START HERE.** This supersedes `STATUS_2026-10-16_O2_PRODUCTION_OUTPUTS_HANDOFF.md`, which is still the reference
for O2 production and for the to-do list items 1-11 it carries. Reasoning and numbers are in `DECISIONS.md`
(entries from 2026-10-17 and 10-18). Orofacial results are in **`docs/OROFACIAL_FINDINGS.md`**.

## 00. UPDATE 2026-10-18 evening (read first)

- **Run v36 replaces v35** (June camera-scale correction from the no-lick ENL spout ruler; Aug+ bit-identical). v35 is
  marked SUPERSEDED on the share.
- **The AUGUST-ONLY pre baseline is PRIMARY** for every orofacial analysis and the encoding model (default
  `--baseline late`; primary = plain folder names; June-including = `*_allpre` + `SUPERSEDED.md`, sensitivity only, and
  only for scale-free measures). June cannot be rescaled further: head features magnify 1.17-1.24x vs the spout plane
  1.13x (perspective). Imaging has NOT been switched yet (needs the frozen-model / nightly / deck step; DECISIONS).
- **Encoding model redesigned** (`combined_encoding.py`): own ridge penalty per reduced model, order-free drop dR2,
  imaging-map masks, shared cue + arrival kernels with deviation-coded position (one variable; per-position one-vs-rest
  and side / distance permutations; absolute + deviation response maps), lick kernel -0.5..+2 s, no reward, variants
  for target vs executed angle (with and without tongue x/y). Old v35 encoding outputs are invalid (borrowed penalties).
- **New figures:** heatmaps (incl. no-lick ENL spout / nose, jaw from the mouth), jaw paths + phase portraits, delta
  and per-animal trajectories; W2 spout control on no-lick ENL frames; retraction cut at trial end.
- **O2:** video motion-energy pilot (`scripts/o2_video_me.py`, bundle `me_pilot_20261018`, 46 tasks) submitted by Priya.

## 0. RUNNING when this was written (check these first)

All runs are on the analysis desktop. Logs are in the session scratchpad
`C:\Users\SABATI~1\AppData\Local\Temp\claude\C--Users-SabatiniLab\cdd1e016-...\scratchpad\`. Find the processes
with `Get-CimInstance Win32_Process -Filter "Name='python.exe'"` and match on the command line.

| what | command | log | outputs |
|---|---|---|---|
| **Imaging August-only sensitivity** | `WIDEFIELD_PRE_FROM=0801 epoch_grant_figures --only acc 5c 12b 13s scal mat --output <labcams>/grant_figures/epoch_latepre`, then `locanmf_frozen_decoder --loso --align {cue,lick,precue} --output <labcams>/frozen_decoder_latepre` (script `imaging_latepre.sh`) | `imaging_latepre.log` | `labcams/grant_figures/epoch_latepre/`, `labcams/frozen_decoder_latepre/` |
| **Combined encoding model, all sessions** | `python -m wfield_local.combined_encoding --all --jobs 2 --skip-done` (locanmf env) | `encoding_all.log` | `LP_analysis/<run>/imaging_encoding/movement_first/sessions/<label>.{npz,json}`, `_regions.csv` |
| **ViT-B DINOv2 640 retrain on O2** | job **56700781** (array task 1 of `lp_train_20261016/train.sbatch`), after pinning `transformers==5.17.0` in the O2 `lp` env | O2 `logs/56700781_1.err`; progress in `models/vitb_dinov2_640_b8/train_status.json` | rsync `models/vitb_dinov2_640_b8/` to the share when `DONE` exists, then score it (§4) |

Expected `[frozen] !! ... SPEC-CHANGED` lines in the imaging log are correct. The August-only training set gets
**new** frozen-model ids, and the canonical models are untouched. Never pass `refreeze=`.

## 1. What we did and why

### A. cam4 orofacial cohort (Lightning Pose r4, cleaning v35): COMPLETE data

- **O2 inference:** all **136 sessions** are processed (137 videos; PS92_0812 has 2 videos).
  - 133 ok.
  - PS94 / PS95 0601 are skipped (dropped frames).
  - PS93 0805 fails: it is the excluded "wonky 8/5" with an empty GUI log.
  - **119 curated sessions** enter the cohort analyses.
  - PS92_0812 (crash + concat) uses both videos correctly (24 + 201 trials). The ~55 log trials from the 41-minute
    gap are not in it, and that is acceptable.
- **Pipeline:** `orofacial_outputs process` → `orofacial_jaw_lateral` → `orofacial_cohort` → `orofacial_w2` →
  `orofacial_deck`.
  - All modules run in the dlc env, except the deck (locanmf env, python-pptx).
  - Each takes `--baseline late` and `--all-trials` (the jaw module takes `--all-trials` only).
- **Engagement gate (10-17):**
  - Every orofacial analysis uses ENGAGED trials only, via the behaviour pipeline's reference-position gate.
  - The ungated control goes to `*_alltrials`.
  - Behaviour-table trial ids are matched to ours **directly** when the positions agree; otherwise **by DAQ cue
    time**; otherwise **by trial order per recording segment**, using the shared aligner
    `behavior_position.best_trial_offset`.
- **Two baselines (10-18):**
  - The June → August pre period drifts as much as many stroke effects (learning / framing).
  - Everything is therefore rendered under **all-pre** and **August-only pre** (`--baseline late`,
    `late_pre_from: "0801"`, into `*_latepre` folders, with a second deck).
  - Findings report only effects significant by both tests under **both** baselines.
- **PS93:** kept. Leave-PS93-out is reported:
  - `scripts/orofacial_robustness_table.py` → `cohort/tables/robustness/`;
  - 70 of 85 cells that are robust to both baselines also hold without PS93.
- **Figure set:** see `orofacial_deck.SECTIONS`, about 85 slides.
  - Lick figures are {lick-peak time, per-lick onset→offset phase} × {first, all} × {tongue out, in-mouth
    included}.
  - Each comes as a change-vs-pre version and a RAW version; the raw versions have the spout-angle line.
  - Phase is shown and tested at 0.1-0.9 only (edge artefacts).
  - Also: cue-aligned figures (Family 8.2), raw and delta.
  - Cohort by-position epochs and day-bin timecourses.
  - Within-animal x-y paths and phase portraits, with significance markers for the x / y / both coordinates.
  - Ipsi / contra labels throughout.
- **Statistics:**
  - Headline: session Mann-Whitney with BH-FDR.
  - Robustness: the animal-weighted bootstrap.
  - Traces: within-animal cluster permutation, 50 % bars.
  - Coverage rule: 25 % of sessions and at least 3 (`orofacial_cohort` config).

### B. W2 spatial-distribution readout (`wfield_local/orofacial_w2.py`, 10-18)

- Wasserstein-2 between each session's keypoint cloud and the animal's pooled pre cloud.
  - Exact optimal transport via scipy `linear_sum_assignment`; 800 points × 3 seeded draws.
  - Pre sessions are scored leave-one-out, which gives the noise floor.
- Keypoints, ALL in the mouth frame (Priya 10-18): tongue (out frames), jaw, nose (behaviour: head / snout posture),
  spout (the noise / drift control: fixed apparatus).
- **Results:**
  - Tongue: +61 px at far contra acute; chronic +6-11 px under both baselines.
  - Spout control: chronic +1-3 px (bootstrap-only) -> ~2-3 px drift floor.
  - Nose: +3-5 px acute, +3-4 px chronic (August baseline) -> candidate postural change, not drift.
  - Raw image framing drifts 25-60 px over months; image-pixel rows were dropped.
- Outputs: `cohort/{tables,figures}/w2{,_latepre}/`.

### C. Behaviour-pipeline fix: pre-v47 GUI-log position shift (10-17)

- `spout_behavior._gui_shift_corrected` measures the shift per session against the DAQ. With no DAQ, logs dated
  before 08-10 (first v47 session) get the documented shift.
- 70 sessions were corrected (May, 06-01, 08-05, 08-06), and every behaviour, grant 1 / 1b and epoch 1b / 1c output
  was re-run.
- The imaging position repair was already immune; verified that both aligners agree to ≥ 99.8 %.
- Recorded in DECISIONS and `docs/GUI_TRIALS_LOGGING.md`.

### D. Imaging August-only sensitivity (10-18, RUNNING)

- `config.pre_from()` / env **`WIDEFIELD_PRE_FROM`**:
  - opt-in;
  - early pre sessions become `'excluded'`;
  - suspended inside `epoch_audit.full_cohort` (epoch boundaries must not move);
  - tested in `tests/test_pre_from.py`.
- Readouts being re-rendered:
  - position retained fraction (acc / 5c / 13s);
  - state control (12b);
  - encoder (scal);
  - crossnobis (mat);
  - frozen decoder LOSO.
- **Not a straight swap:** `chronic_drift_vs_reorg` uses June-August pairs as its drift floor. It needs a
  lag-matched version: within-August pairs vs within-chronic pairs at 1-8-day gaps.
- **Left out:** grant 3a / 3b / 5 / 11 (shared JSONs would mix the baselines).

### E. Combined task + movement encoding model, Musall 2019 Fig. 4-5 (`wfield_local/combined_encoding.py`, 10-18)

- **Data:**
  - pixel level: the adopted hemo-corrected SVD (`hemo_meegkit_hpfit/SVTcorr`, 100 components, 31.25 Hz, full rate);
  - R² per pixel on the shared Allen grid via `U_atlas`, computed exactly from the component-space residual
    covariance;
  - maps stored at 2×2-pixel blocks.
- **Variables:**
  - task: `pos_enl` (per-position spout-arrival kernel + spout-present state, strobe → cue), `pos_cue`
    (per-position cue kernel), reward;
  - instructed: licks (shared lick-onset + contact kernels), `exec_angle` (lick kernel modulated by the
    within-position executed angle), tongue analog, jaw analog;
  - uninstructed: nose, running.
  - Position specificity lives ONLY in the task group.
- **Orthogonalisation:** **movement-first** is primary (shared variance → movement, so position's ΔR² is
  conservative); `--order task_first` is the sensitivity version.
- **Measures:**
  - single-variable cvR² ("alone");
  - shuffle ΔR² (circular shift; position = label permutation);
  - Fig. 5 task-aligned vs task-independent;
  - PETH partition (full model fitted once, split by group), cue-aligned, per position and area;
  - time-resolved held-out SSE;
  - collinearity (cumulative subspace angles).
  - Windows: all / enl (pre-cue 2 s) / cue / lick, each **window-offset-free** (bug found on the first test).
- **First sessions:** full cvR² 0.44 (Musall 41 %).
  - Uninstructed ΔR² ≈ 0.2-0.28 (running dominates), instructed ≈ 0.16-0.24, task ≈ 0.
  - Position-specific ΔR² is about 0.06 cue / 0.07 lick pre, and about 0.03 / 0.04 on PS92 acute 0820.
  - Pre-cue (ENL) position ΔR² is small (≈ 0.002).
- **Still to build:** `scripts/combined_encoding_figures.py`:
  - Musall-style Fig. 4 bars and maps (alone vs ΔR², per variable / group, per epoch, per window);
  - time-resolved ΔR²;
  - Fig. 5 task-aligned / independent bars and PETH partitions per position;
  - position ΔR² maps by epoch;
  - executed vs intended (exec_angle vs pos_cue ΔR²);
  - cohort averaging (within animal first, then across; all-pre and August-only).
  - Then the `task_first` sensitivity run, and the factorised distance × side position variable.

## 2. Decisions needing Priya (open)

- `grant_kit._sessions()` is not curated: 06-01..05 and 08-05 enter the pre pools of grant 1 / 1b and epoch 1b.
  Curate it?
- Production pose model:
  - ctx-semi 640 (O2) scored ≥ LP r4: same share of real licks, higher precision, much steadier jaw.
  - The recommendation was NOT to switch now (that means re-running everything); revisit with the ViT-B score.
- Superseded figure files on the share: the clean-up PowerShell is in an earlier message and
  `OPEN_QUESTIONS_PRIYA.md`.

## 3. Discussed, not started

- **MVAR, then a CURBD pilot (Priya wants this, after the encoding model):**
  - Units: joint-basis LocaNMF components grouped by Allen region, NOT pixels. CURBD needs one unit per model unit;
    pixels are ~200k strongly correlated units that do not partition into regions. SVD components are global, not
    regional.
  - Fit on encoding-model **residuals**, so shared task / movement drive is not read as inter-region current.
  - A stability check across seeds and session halves comes before any pre vs post comparison.
  - Caveat: the lesion is subcortical and striatum is unobserved.
- **ARHMM (`github.com/htong1/Priya_ARHMM2026`, private; cloned to the scratchpad):**
  - The stroke_orofacial cohort (PS46-50): frozen pre-stroke AR-HMM via jax-moseq, K = 24, nlags 3, 12.5 Hz, on
    FaceRhythm TCA latents + DLC tongue / jaw + treadmill; post-stroke projected.
  - Result: the contra deficit splits into "can't lick" vs "licks and misses".
  - For us, one option is a keypoint-only version on LP pose (tongue / jaw / nose, engaged windows). FaceRhythm
    for cam4 is a pending to-do; without it the model sees no whisking / face.
  - Possible uses:
    1. a behavioural segmentation of licking modes and non-lick behaviours across recovery;
    2. syllable indicators as an uninstructed (or instructed) regressor group in the encoding model;
    3. syllable-triggered maps.
  - Risk: syllables built from tongue / jaw pose are collinear with the instructed variables.
- **Video motion energy** exists for only 3 whole sessions. Computing it for all sessions would add Musall's
  strongest uninstructed variable (video-heavy job; `scripts.video_motion_session`).
- To-dos 7-11 from the 10-16 handoff:
  - (7) jaw-only / minimal-tongue example gallery;
  - (8) drift control: now partly answered by W2 and the August-only baseline;
  - (9) suggested kinematic analyses;
  - (10) curation;
  - (11) W2: DONE.

## 4. Commands

```bash
# dlc env:    /c/Users/SabatiniLab/miniconda3/envs/dlc/python.exe
# locanmf env: /c/Users/SabatiniLab/miniconda3/envs/locanmf/python.exe
python -m wfield_local.orofacial_outputs process --bundle lp_r4_prod_20261015 --jobs 4
python -m wfield_local.orofacial_jaw_lateral [--all-trials]
python -m wfield_local.orofacial_cohort --jobs 6 [--baseline late] [--all-trials]
python -m wfield_local.orofacial_w2 --jobs 6 [--baseline late]
python -m wfield_local.orofacial_deck [--baseline late]                     # locanmf env
python scripts/orofacial_robustness_table.py                                # 4 variants
python scripts/orofacial_cover_sensitivity.py
python -m wfield_local.combined_encoding --all --jobs 2 [--order task_first] [--skip-done]   # locanmf env
python -m wfield_local.locanmf_analysis_deck --src M:/MICROSCOPE/Priya/Widefield/labcams/analysis_figures
python -m wfield_local.deck_split                                             # after the analysis deck
PYTHONPATH=. python -m scripts.model_lick_compare PS93:<d> --model "NAME=<csv>" ... \
    --review C:/Users/SabatiniLab/lp_stage/lick_review/review_scored.csv --tag <tag>
```

## 5. Pitfalls (new; older lists still apply)

- **Another session's uncommitted DLC edits are in the working tree**:
  - files: `wfield_local/dlc_*.py`, `lp_multiview.py`, `scripts/{lp_unlabelled_clips,o2_lp_train_bundle,session_poses}.py`,
    `tests/test_dlc_train.py`, `docs/status/OPEN_QUESTIONS_PRIYA.md`;
  - content: the manifest move to `dlc_frames.manifest_path`;
  - **stage files by name; never `git add -A`**;
  - `git rebase --autostash` keeps them intact.
- **Analysis deck on this box needs `--src <labcams>/analysis_figures`**. `figures_working` lags the share; the
  completeness guard refused 857 missing figures.
- **Spawned workers re-import modules.** Runtime options must travel by env var and be re-applied at import:
  `OROFACIAL_BASELINE`, `OROFACIAL_ALL_TRIALS`. `tests/test_parallel_results_are_ordered.py` enforces this, and also
  that `fan_out` results are sorted.
- **Window-restricted R² needs a free window offset** (combined_encoding). Otherwise "alone" R² goes very negative
  and ΔR² can exceed the full R².
- **W2 must be in the mouth frame.** Image pixels measure camera framing drift (25-60 px), not behaviour. The spout is
  the control; the nose is behaviour.
- **Do not reuse Python's `hash()` for seeds** (it changes with PYTHONHASHSEED); use `zlib.crc32`.
- **The ViT backbone needs `transformers==5.17.0`** with Lightning Pose 2.4.2 (README, O2 runbook).
- **Heredocs with nested quotes broke repeatedly.** Write edit scripts with the Write tool, then run them.
- **The PDF reader needs Poppler** (`pdftoppm`), which is not installed. Paper text was extracted with pypdf
  installed into the scratchpad only.
