# FaceRhythm handoff (2026-10-24 morning) -- START HERE

Supersedes `STATUS_2026-10-22_FACE_RHYTHM_HANDOFF.md` for current state (that file keeps the 10-22 / 10-23 history:
pilot, timing fix, master planning, rank-10 fit). Findings: `docs/FACE_RHYTHM_FINDINGS.md` (status table at the top).
Decisions: `DECISIONS.md` 2026-10-24 (rank 12; names + grouping master_v1; close / far parts). How it runs:
`docs/FACE_RHYTHM_PIPELINE.md`.

Rules still in force: don't run heavy things locally without checking the box (shared; d4's encoder runs); commit only
your own files, staged by name; never delete data; write only under MICROSCOPE/Priya. Document as you go.

## 1. State

**Master = `master_pre_v1`, RANK 12, frozen 10-24 05:49, spec `f4f84d67773df9bc`**
(`M:/MICROSCOPE/Priya/DeepLabCut/Widefield/o2/fr_cam4_20261021/factors/master_pre_v1/`). Fit on O2 (job 57395128_12):
4 animals x 120 trials (20 / position, even over pre days), pre-cue rows, run / rest between rows, true bin times;
EVR .803, stability .912, leave-one-animal-out factor similarity .82-.90, split half .911.

| on the share | state |
|---|---|
| `master.npz`, `rank12.*`, `labels.json` | done; names in FINDINGS 4a |
| `proj/` 108 | done (full frozen basis for every session; `basis` full_in_sample 32 / full 76) -> **ENCODER** |
| `loso/rank12_*` 31, `proj_loso/` 32 | done (pre sessions on their left-out-day basis; FR scoring only) |
| `splithalf/rank12.json` | done (.911; lick pair .96 / .99; f6 run_onset_jaw .36, f10 prep_jaw .76) |
| `own/` (per-session fits, EVR gap, morphing) | **15 / 108 copied** -- `frf_own` (57438149) was still running; RSYNC AGAIN |
| `report/` | rank-12 gallery (abs + share, with names), rank sweep, stability, loso tables |
| rank-10 outputs | record kept: `master_r10.npz`, `rank10.*`, `own_r10/`, `report_r10/`, validation `master_pre_v1_r10` |

Local clips (one pre session per animal, 12 factors) `face_rhythm/cam4/factor_clips/master_pre_v1/`; validation
`face_rhythm/cam4/validation/master_pre_v1/` (+ `lick_epochs/`).

**Encoder (d4)**: has proj/ + labels + grouping `fr_groups.master_v1` (lick 2, 4, 5, 7, 8, 11; left 5; right 4; close 7;
far 8; other 0, 1, 3, 6, 9, 10; modes A / B / S / D; side = S - B, distance = D - B). Bundling into
`enc_v36_full_20261024` (107 sessions; raw 12 factors, grouping applied at submit).

## 2. Next, in order

1. **Rsync `own/`** once `frf_own` is done (transfer node; RESET the variables -- the encoding bundle reuses B / S):
   `B=/n/scratch/users/p/ps150/o2/fr_cam4_20261021; S=/n/files/Neurobio/MICROSCOPE/Priya/DeepLabCut/Widefield/o2/fr_cam4_20261021`
   `rsync -ah --exclude fit_tensor.npy --exclude 'proj_r10' --exclude 'proj_loso_r10' $B/factors/master_pre_v1/ $S/factors/master_pre_v1/`
   -> expect `own/` 108. (The excludes keep the rank-10 projections deleted from the share on 10-24 from coming back.)
2. Run on all four animals (fr env; `export PYTHONPATH=<repo>`; push check needs the locanmf env for behaviour):
   * `python scripts/face_rhythm_readouts.py --set master_pre_v1` -> movement maps, amplitudes, EVR gap, recovery by
     day, task vs non-task, severity (`face_rhythm/cam4/readouts/master_pre_v1/`)
   * `python scripts/face_rhythm_match.py --set master_pre_v1` -> factor angles vs master, pre floor, chance (`matching/`)
   * `python scripts/face_rhythm_push_check.py --set master_pre_v1 --jobs 4` -> recon error at treadmill pushes
     (`push_check/`); decides whether a pre + post fit is needed (see 3). Test on 2 PS95 sessions was still loading
     behaviour at handoff time -- check `scratchpad` output / run one session first.
3. Redo FINDINGS 5 / 6 on four animals; add the matching, push, task / non-task and severity results.
4. Other cameras (cam1 / cam2 / cam3): await Priya's landmark clicks -> register -> ROIs -> O2 -> multi-view set.

## 2b. Clean-up done 10-24 (Priya OK)

Deleted from the share (all DERIVED, rebuildable; `writeguard.assert_deletable(derived=True, approved=True)`), ~14.7 GB:
`master_pre_v1/fit_tensor.npy` (STALE: the earlier sweep sample's tensor, 27,392 rows, no longer matching fit_index's
27,448; the live tensor is on O2 scratch, `build` rebuilds it), `master_pre_v1/proj_r10/` + `proj_loso_r10/` (rank-10
projections; `master_r10.npz` kept), `face_rhythm/cam4/factor_clips/master_pre_v1_r10/` (rank-12 clips complete: 48 =
12 factors x 4 animals), `factors/preview_r10_PS92/` + `face_rhythm/cam4/preview_r10_PS92/` (single-session preview,
superseded). Kept: `register_v1..v4_20261021/` (1.7 GB, comparison sets) and `param_test/` (1.1 GB) pending Priya; O2
scratch left to purge itself. JupyterLab (ROI notebook) stopped -- restart for the cam1-3 landmark clicks. Other
windows asked to prune their stale git worktrees.

## 2c. TO DO: make the FaceRhythm code modular and easy to use (Priya 10-24)

"Clean up the code to make it modular and easy to use, as a Claude engineer would organise it for a systems
neuroscientist to use." Not mid-analysis (after the four-animal readouts), and refactors must carry the reasoning in
the comments forward (CLAUDE.md engineering note). Starting points:
* `scripts/face_rhythm_factors.py` (~1,480 lines) mixes the CP fit, planning / sampling, O2 sbatch generation,
  projection, LOSO / split half, and the report. Split into `wfield_local/face_rhythm/` modules (fit, sample, project,
  validate, report, o2) with a thin CLI; keep the O2 single-file copy working (`plan` copies the sidecars today).
* One entry point per question a user asks ("fit a factor set", "project sessions", "name / validate", "readouts",
  "match fits"), each with a short docstring stating inputs, outputs and where they land; one config block (set name,
  bundle, rank, windows) instead of constants spread over modules.
* A worked notebook / README walk-through: from FR outputs to a named factor set to the encoder regressors, with the
  pitfalls (in-sample pre, absolute vs share, factor numbers shift between fits).
* Tests per module (the existing ones in tests/test_face_rhythm_*.py are the base).

## 3. Open questions

* **Pre + post master?** (Priya 10-24, for the encoder's treadmill "pushes"). Not yet: the pre-only basis makes the EVR
  gap / reconstruction error the readout of new movement. Decide from `face_rhythm_push_check.py`: if the frozen basis's
  error rises at pushes chronically (not pre, not at random times), fit pre + post (or a post-only fit matched to the
  master with `face_rhythm_match.py`; unmatched factors = the new movements). If not, pushes stay a treadmill regressor.
* Rank-12 weak factors: 6 run_onset_jaw (split half .36) and 10 prep_jaw (.76) -- grouped in "other", never read singly.
* PS93: weakest side selectivity in both lick factors; lick sum not position-balanced (far_L .79, close ~1.15).

## 4. Code added 10-23 / 10-24 (all committed)

* `wfield_local/face_rhythm_matching.py` + `scripts/face_rhythm_match.py` (+ tests): Hungarian matching of independent
  fits (on product / spatial / spectral / sum; spatial as vectors or magnitudes), angles in degrees, split / merge
  flags, chance (point / frequency permutation), pre-vs-pre floor.
* `scripts/face_rhythm_push_check.py`: frozen-basis error + face energy around treadmill push onsets vs random times.
* `face_rhythm_readouts`: `lick_split`, `task_nontask`, `severity_table` (histology burden) + two figures.
* `face_rhythm_factors.py`: `match` handles unequal ranks; gallery names on the timecourse panel.
* `fr_groups.master_v1` with close / far parts and modes S / D.

## 5. Pitfalls met this session

* The `fr` env lacks the repo install: `export PYTHONPATH=C:/Users/SabatiniLab/Github/widefield_pipeline`.
* `select` refuses an existing `master.npz`; `project` / `fitsession` skip existing files -> move old outputs aside
  before re-selecting a rank (done for rank 10 as `*_r10`). Plan `loso_jobs` carries the rank (switched to 12 on O2;
  backup `logs/plan_loso_rank10.json`).
* O2 shell variables B / S collide between the FR and encoding bundles.
* Gallery / validation number factors from 0; FINDINGS 4a uses the same numbering.
* Multi-line bash heredocs with quotes break in this shell: write patch files instead.
