# FaceRhythm guide — start here

FaceRhythm (FR; Hakim et al., bioRxiv 10.1101/2025.09.10.675423) turns face video into a small set of movement
**factors**: each factor is a pattern of where on the face things move (spatial loading) and at what frequencies
(spectral loading), with a time course saying how strongly it is expressed in every 25 Hz frame. This guide is for
someone who wants to answer a question with the cam4 FR data of the VLS-stroke cohort (PS92–PS95). Other documents:

| need | document |
|---|---|
| the results so far, with status tags | [`FACE_RHYTHM_FINDINGS.md`](FACE_RHYTHM_FINDINGS.md); figures + takeaways: https://claude.ai/artifact/X8NDdQNHi3zMP77PBrZAC1 |
| how the tracking and the factor fits run (O2, files, run order) | [`FACE_RHYTHM_PIPELINE.md`](FACE_RHYTHM_PIPELINE.md) |
| why a choice was made | `DECISIONS.md` (2026-10-21 onward, "FaceRhythm") and [`FACE_RHYTHM_PLAN.md`](FACE_RHYTHM_PLAN.md) |
| what is running / next | the newest `status/STATUS_*_FACE_RHYTHM_*` handoff |

## 1. The ideas you need

* **One frozen factor set for everyone.** The master set `master_pre_v1` (rank 12) is fit on PRE-STROKE sessions of all
  four animals, on one shared face mesh, and then frozen. Every session, pre or post, is projected onto it
  (`<set>/proj/<key>.npz`), so a change after the stroke is a change in how much of each pre-stroke movement the animal
  makes. Factor names and numbers: FINDINGS 4a (`labels.json`); numbering starts at 0.
* **Own fits** (`own8/`, and per animal x epoch `own_epoch/`) are fresh fits of the same rank to a session's (or epoch's)
  data. They answer "does the vocabulary itself change?": EVR of the frozen set vs the own fit, and the angle between
  matched factors.
* **Absolute vs share.** FR normalises each frame by its total movement. *Absolute* expression = factor weight x the
  frame's normaliser D (how much of that movement there is) — the default in every readout. *Share* = the weight
  alone (how movement is distributed).
* **Change vs each animal's own pre.** Animals differ in scale, so every number is relative to that animal's
  pre-stroke sessions: log2 for amplitudes (0 = pre, −1 = halved, +1 = doubled), ratio or difference for EVR.
* **Sides.** All lesions are LEFT. The animal's RIGHT spouts (close_R, far_R) and the *lick_right* factor are
  CONTRALESIONAL. On the cam4 image the contralesional half of the face is IMAGE LEFT (x ≤ the master's nose midline).
* **Statistics.** n = 4 animals. Every figure first shows each animal (rule 8: a result not visible in ≥ 3 animals will
  not survive), then the paired animals → sessions bootstrap of the change (`analysis_kit.boot_delta`). Marks:
  `*` = the 95 % interval excludes 0; `**` = also survives Holm over everything that figure tests at once (12 factors or
  6 spout positions). Levels (is something above 0?) use `boot_ci`.
* **Colours.** Animals: `config.animal_color()` (the repo palette). Epochs: pre black, acute pink, subacute gold, chronic
  teal (the orofacial / 3-D kinematics palette).

## 2. Answering a question

```bash
export PYTHONPATH=C:/Users/SabatiniLab/Github/widefield_pipeline   # the fr env does not install the repo
conda activate fr
python -m wfield_local.face_rhythm list                 # every analysis: the question it answers, runtime, outputs
python -m wfield_local.face_rhythm show peak            # one analysis in full, with its options
python -m wfield_local.face_rhythm run peak --split side
python -m wfield_local.face_rhythm run peak --split side --plot-only    # redraw from the saved cache in seconds
```

Figures land in `M:/MICROSCOPE/Priya/DeepLabCut/Widefield/face_rhythm/cam4/readouts/master_pre_v1/<analysis>/`, each
folder with a `README.md` of figure captions (methods) and the tables behind the figures in `data/`.

| you want to know | analysis | FINDINGS |
|---|---|---|
| how each factor's amplitude changes after the stroke (all trials / by spout side / by spout) | `peak` | 5d |
| when after the cue each factor is expressed, and whether that moved | `peak` (timing figures) | 5g |
| fewer licks or different licks? | `per-lick` | 5b-master |
| does the contralesional half of the face move less (cue / lick / bout aligned)? | `asym` | 5e |
| where on the face movement changes | `maps`, `position-maps` | 5f |
| does the pre-stroke vocabulary still fit? | `readouts` then `evr` | 6-master |
| do the factors change shape? | `match` (and the pooled `own_epoch/` fits) | 6c |
| non-task movement, by behavioural state | `nontask` | 6d |
| treadmill pushes | `push` (locanmf env) | 6b |

## 3. Adding an analysis

Write a script in `scripts/face_rhythm_<name>.py` and build it from the shared pieces in `wfield_local/face_rhythm/`
rather than copying code out of another script:

| module | gives you |
|---|---|
| `common` | `EPOCHS`, `POSITIONS` (one order; `POSITIONS_CONTRA_FIRST` for display), `SIDE`, `sessions(fdir)` (the projected sessions, sorted), `session_keys`, `position_groups`, `label`, `day`, `factor_names`, `readouts_dir`, `animal_colors()`, `epoch_colors()` |
| `face` | `master_points` (point components + contralesional mask), `side_speed` (contra / ipsi speed per frame from the frozen point PCs), `aligned_mean`, `window_weights`, `asymmetry` |
| `licks` | `lick_table` (LP licks with bouts, time from cue), `bout_classes`, `first_lick_latency` |
| `stats` | `log2_vs_pre`, `epoch_change` (per-animal epoch means + `boot_delta`, with the bootstrap p), `add_holm`, `mark`, `tick_label` |

Conventions every FR script follows:
1. Loop over sessions with `analysis_kit.fan_sessions` (parallel, results sorted) and a module-level worker.
2. Save what the figures need (`data/<name>_cache.*`) and offer `--plot-only`, so a figure change does not recompute.
3. Print the per-animal table before any bootstrap; one value per SESSION (a split video, PS92_0812, counts once).
4. `figure_layout.sidecar_for(out, ...)` for tables, `face_rhythm_figures.save(fig, path, caption)` for figures (the
   caption is the methods text, written to the folder's README).
5. `writeguard.assert_writable(out)` before writing on the share.
6. Add the analysis to `CATALOGUE` in `wfield_local/face_rhythm/__main__.py`, and its result to FINDINGS.

## 4. Pitfalls

* Pre-stroke sessions in `proj/` are scored on the basis fit WITHOUT their own day (leave-one-day-out, `proj_loso/`)
  for EVR; otherwise their fit is in-sample and looks better than any post session.
* A trial-averaged amplitude falls when fewer trials contain the movement (acutely, few contralesional licks); use the
  per-lick readout to separate "fewer" from "different".
* The face mesh tracks the face surface, not the tongue: per-lick face speed cannot see tongue extent or lick rate (the
  DLC / LP kinematics can; OROFACIAL_FINDINGS).
* Factor numbers differ between fits (rank 10 vs 12, own fits): always go through names (`labels.json`) and matching.
* `fr_groups` and `face_rhythm_outputs` are used by the encoding model, `face_rhythm_timing` / `face_rhythm_multiview`
  are copied next to the O2 factor script: change their interfaces only with those users in mind.
* The factor stages live in the package `wfield_local/face_rhythm/factors/` (config, files, cp, sample, fit, project,
  validate, o2, report, cli; split from the single script 10-25, map in its docstring); `scripts/face_rhythm_factors.py`
  is the launcher. O2 has no repo, so a set carries its own code: `plan` (or, for an existing set without
  re-planning, `python -c "from pathlib import Path; from wfield_local.face_rhythm.factors import deploy_code;
  deploy_code(Path(r'<share>/<bundle>/factors/<set>'))"`) writes `fr_factors.py` + `fr_factors_lib/` + the sidecar
  modules into the set folder; then copy the set's code to scratch before submitting jobs. The launcher prefers a
  `fr_factors_lib/` beside it, so a set always runs the code deployed into it (sets deployed before 10-25 hold the old
  single-file `fr_factors.py`, which still runs).
