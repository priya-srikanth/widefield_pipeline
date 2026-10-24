# FaceRhythm handoff (2026-10-24; afternoon update at the top of section 2) -- START HERE

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
| `own/` (3 starts) + `own8/` (8 starts, O2 rerun 10-24) | done, 108 each; readouts / matching read `own8/` automatically (`face_rhythm_readouts.own_sub`) |
| `report/` | rank-12 gallery (abs + share, with names), rank sweep, stability, loso tables |
| rank-10 outputs | record kept: `master_r10.npz`, `rank10.*`, `own_r10/`, `report_r10/`, validation `master_pre_v1_r10` |

Local clips (one pre session per animal, 12 factors) `face_rhythm/cam4/factor_clips/master_pre_v1/`; validation
`face_rhythm/cam4/validation/master_pre_v1/` (+ `lick_epochs/`).

**Encoder (d4)**: has proj/ + labels + grouping `fr_groups.master_v1` (lick 2, 4, 5, 7, 8, 11; left 5; right 4; close 7;
far 8; other 0, 1, 3, 6, 9, 10; modes A / B / S / D; side = S - B, distance = D - B). Bundling into
`enc_v36_full_20261024` (107 sessions; raw 12 factors, grouping applied at submit).

## 1b. Results and figure index (10-24 afternoon; FINDINGS 5b-master, 5d, 5e, 6-master, 6b-6d)

All under `M:/MICROSCOPE/Priya/DeepLabCut/Widefield/face_rhythm/cam4/readouts/master_pre_v1/` unless noted.

| question (Priya) | result | figure(s) / script |
|---|---|---|
| contralesional deficit per lick? | lick_right per right lick .57-.79 of pre chronically, 4 / 4 | `validation/master_pre_v1/lick_epochs/` (`face_rhythm_lick_epochs.py`) |
| peak factor amplitude over recovery (manuscript FR panel) | contra-spout lick_right chronic -.30 [-.40, -.21] 4/4; contra-spout lick_left +1.08 [.35, 1.87] 4/4 (ratio scale); per position: far_R loses most contralesional, close_R gains most ipsilesional | `peak/peak_by_day*.png`, `peak_cohort_by_{day,epoch}*.png`, `peak_side_headline.png`, `peak_position_headline.png` -- LOG2 scale since 10-24 pm (`face_rhythm_peak.py --split none/side/position`) |
| KEY: contralesional licks made with an ipsilesional face movement | factors (above) + raw point movement, lick-aligned, far_R asym pre +.22-.27 -> chronic below pre 4/4 | `asym_aligned/asym_aligned_traces.png` (cue / lick / bout-onset columns), `asym_aligned_by_animal.png` (`face_rhythm_asym_aligned.py`) |
| does the pre vocabulary fit post? (EVR, normalised) | frozen / own ratio vs own pre: acute -.035 [-.045, -.025] 4/4, chronic -.008 (3/4, borderline) | `evr_own_vs_frozen.png` (bottom row pre = 1, +-1 SEM), `evr_epochs.png` (`face_rhythm_evr_epochs.py`) |
| why does own EVR drop too? | less movement vs fixed tracking noise + less stereotyped movement (partial: r with between-trial speed .48-.66 in PS94 / PS95 only); the ratio divides it out | -- |
| factor angles / morphing | NOT consistent across animals (Priya: "no clear consistent changes") -> secondary; lead with amplitudes | `matching/match_by_day.png`, `match_example.png` (`face_rhythm_match.py`) |
| asymmetry two ways (points vs own-fit loadings) | agree acutely; loadings: lick_right moves off the contralesional face | `matching/asymmetry_two_ways.png` |
| per-position face speed maps | whole face moves less acutely at every position; contralesional half more, strongest far_R; chronic overshoot, ipsi half up on R spouts | `position_maps/position_maps_response_{mean,PS9x}.png` (`face_rhythm_position_maps.py`; cue 0..+4 s) |
| lick- / bout-aligned maps + patch statistics | DONE: whole face down per TRIAL acutely, not per LICK; per lick the contralesional half loses more at close_R / far_R, 4/4, every epoch (FINDINGS 5f) | `aligned_maps/aligned_maps_{cue,lick,bout}.png`, `halves_*.png` (`face_rhythm_aligned_maps.py`, needs the cache from `face_rhythm_asym_aligned.py`) |
| pre + post master for pushes? | no: frozen error LOWER at pushes, 27 / 32 cells | `push_check/` (`face_rhythm_push_check.py`); DECISIONS 10-24 |
| task vs non-task, severity | non-task NOT preserved acutely; no ordering by lesion size (n = 4) | `task_vs_nontask.png`, `severity.png` |

## 2. Next, in order

0. **10-24 afternoon: items 1-3 below are DONE**, plus aligned maps / statistics (FINDINGS 5f) and the repo palettes
   (animals `config.animal_color()`, epochs `epoch_figures.EPOCH_GREY`; all FR figures re-rendering in the background
   at handoff, push check last, ~1.5 h). OPEN with Priya: (a) close_R bout-aligned asymmetry > 0 before onset -- split
   first bout of trial vs later and >= 0.5 s lick-free lead-in (offered, not run); (b) lead framing: contralesional
   licks keep their rate but are made with less contralesional / more ipsilesional face movement (5d / 5e / 5f) + the
   acute EVR drop; (c) optional: one own fit per animal x epoch for a cleaner shape test (6c); non-task drop vs
   running / per factor (6d caveat).
1. (done) **Rsync `own/`** once `frf_own` is done (transfer node; RESET the variables -- the encoding bundle reuses B / S):
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
* SEPARATE COMPUTE FROM PLOT (Priya 10-24: "can figures re-render without redoing calculations?"): every heavy script
  saves what its figures need and takes ``--plot-only``. Done: `face_rhythm_push_check.py` (data/push_check_traces.npz).
  To do: `face_rhythm_readouts.py` (~30 min recompute), `face_rhythm_asym_aligned.py` (traces; maps already cached),
  `face_rhythm_aligned_maps.py` (patch / halves statistics + mean maps), `face_rhythm_match.py` (chance shuffles).

## 3. Open questions

* **Pre + post master?** (Priya 10-24, for the encoder's treadmill "pushes"). Not yet: the pre-only basis makes the EVR
  gap / reconstruction error the readout of new movement. Decide from `face_rhythm_push_check.py`: if the frozen basis's
  error rises at pushes chronically (not pre, not at random times), fit pre + post (or a post-only fit matched to the
  master with `face_rhythm_match.py`; unmatched factors = the new movements). If not, pushes stay a treadmill regressor.
* Rank-12 weak factors: 6 run_onset_jaw (split half .36) and 10 prep_jaw (.76) -- grouped in "other", never read singly.
* PS93: weakest side selectivity in both lick factors; lick sum not position-balanced (far_L .79, close ~1.15).

## 4. Code added 10-23 / 10-24 (all committed)

* 10-24 afternoon: `scripts/face_rhythm_peak.py` (peak amplitude, cohort views, boot_delta, log2, side / position
  splits, headline figures), `face_rhythm_evr_epochs.py` (normalised EVR ratio by epoch), `face_rhythm_position_maps.py`
  (per-position speed maps), `face_rhythm_asym_aligned.py` (cue / lick / bout-onset aligned contra vs ipsi speed + cached
  per-point maps), `face_rhythm_aligned_maps.py` (aligned maps + patch / halves statistics), readouts EVR ratio row,
  matching loading asymmetry, `fitsession --own-dir`.

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
