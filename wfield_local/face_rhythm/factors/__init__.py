"""FaceRhythm factor set fitted on PRE-STROKE sessions of all four animals, frozen, and every session projected onto it
(Priya 2026-10-21: one "master pre-stroke" factor set across animals; DECISIONS.md 10-21 evening). Pipeline overview,
inputs / outputs of every stage and run order: docs/FACE_RHYTHM_PIPELINE.md.

    desktop (locanmf)  python scripts/face_rhythm_factors.py plan   --bundle fr_cam4_20261021 --set master_pre_v1
    O2 (fr)            python <bundle>/factors/<set>/fr_factors.py build                    one job: the fit tensor
    O2 (fr)            python <bundle>/factors/<set>/fr_factors.py fit --rank R             array over ranks
    O2 (fr)            python <bundle>/factors/<set>/fr_factors.py loso --task j            array over (rank, pre day)
    PILOT (10-22)      plan --set pilot_pre_PS92_PS93 --animals PS92 PS93 --with-outputs --role pilot
    desktop            python scripts/face_rhythm_factors.py report --set ... [--face img]  rank sweep + factor gallery
    desktop            python scripts/face_rhythm_factors.py select --set ... --rank R      FREEZE (refuses to refreeze)
    O2 (fr)            python <bundle>/factors/<set>/fr_factors.py project --task i         array over tasks
    O2 (fr)            python <bundle>/factors/<set>/fr_factors.py fitsession --task i      array over tasks
    then the session readouts: scripts/face_rhythm_readouts.py (wfield_local/face_rhythm_readouts.py)

Inputs (`o2_face_rhythm.py`, `<bundle>/out/<key>_{trials,session}.npz`, float16, UNCOMPRESSED -> memory-mapped here):
trials spec (trial, 151 bins at 25 Hz cue -2..+4 s, xy, point, freq), session spec (5 Hz bin, xy, point, freq). Points
are the master mesh warped to each session: point i is the same face location in every session and animal.
The O2 commands need only numpy (+ scipy for the Hungarian match); everything that needs the repo runs in `plan`.

plan       every task's epoch (`wfield_local.epochs.epoch_of`). ROWS: engaged trials balanced over positions + between-trial
           5 Hz bins (outside cue -3 .. +6 s of engaged trials, not the first / last 2 s) = ``between_frac`` x the trial
           rows. MASTER rows: per animal ``trials_per_animal`` trials drawn across its pre-stroke (August) sessions --
           EQUAL rows per animal. SESSION rows (for `fitsession`): ``session_trials`` per task, seeded per task.
build      the master fit tensor (rows, 2 x points, freqs) float32, trial windows at 12.5 Hz (pairs of 25 Hz bins; time
           is not shared across sessions, so the rate only sets the row count). Each ANIMAL is divided by ONE scalar,
           the RMS of its own fit rows (stored; applied to all its sessions, pre and post): face size differs between
           animals (mesh QC: PS93's face is smaller); one scalar removes that without flattening which points move
           most (a per-point scale would amplify the still forehead) and keeps post-stroke amplitude changes visible.
fit        non-negative CP (HALS, numpy, chunked) of rank R: ``inits`` random starts (best kept; similarity between
           starts = stability, Williams et al. 2018, also PER FACTOR: each best-start component's matched spatial x
           spectral cosine in every other start; all starts' A / B saved) and LEAVE-ONE-ANIMAL-OUT (fit on three animals, project the fourth
           with spatial + spectral fixed; held-out vs in-fit error, factor similarity to the full fit).
loso       LEAVE ONE PRE-STROKE DAY OUT (job = (rank, day) from plan ``loso_jobs``): refit without the day's fit rows,
           project the day's SESSION rows (the rows `fitsession` scores every session on) held out vs with the full
           fit. The held-out pre EVR is the noise floor post-stroke frozen EVRs / EVR gaps are read against (report:
           ``loso.csv``, ``loso_by_rank.csv``, rank_sweep panel 4). With only two animals, leave-one-animal-out fits ONE
           animal and projects the other: reported, weak.
pilot      ``plan --animals ... --with-outputs --role pilot``: a set on the animals / tasks available now, stamped
           ``role = pilot`` in plan.json and master.npz (the file name is the same; the SET is never the cohort master,
           and plan refuses a restricted set without --role pilot).
select     freezes rank R as ``master.npz`` (spatial (2, P, R), spectral (F, R), animal scales, spec hash of plan + rank +
           init). A frozen set is a REFERENCE (rule 10, docs/FROZEN_MODELS.md): never refitted once used.
project    per task, spatial + spectral fixed: non-negative temporal weights per bin for the trial windows (25 Hz) and
           the whole session (5 Hz), each bin's relative reconstruction error, and per point and window (pre-cue /
           response by position, between trials) the SPECTRAL SHARE (FR normalises each frame by its whole-face mean
           power: distribution, not amount) and the absolute movement SPEED from the 25 Hz points -> ``proj/<key>.npz``.
           Plus each bin's FR NORMALISER D (`normaliser`): weights x D = the component's ABSOLUTE expression (the
           projection is linear, so un-normalising a bin scales its weights by exactly D). Recovered from the saved
           float16 spectrogram to 0.01 % median / 0.1 % max per 25 Hz bin (0.14 % / 1.5 % 99th on the 5 Hz
           block means; 10-21 test on the param-test points, D varying with CV 0.5).
           Window means come from the 5 Hz session spectrogram: a 5 Hz bin is the mean of five 25 Hz bins, so means
           over windows aligned to 0.2 s are exact, and one pass over the session file covers every trial.
points     (inside build / select / project) a frozen PCA basis of the 25 Hz point DISPLACEMENTS (`points` output,
           2P dims, each session centred on its own mean), fitted on the same pre-stroke animals (equal bins per
           animal): the dense POSITION family (the paper's best widefield regressors were points + FR, 0.38; M1
           follows position below ~0.5 Hz, the spectral envelope above). ``proj/<key>.npz`` gets ``points_pc``
           (bins, K) + ``points_t_daq_s``.
fitsession per task, the session's OWN rank-R fit on its plan rows, the frozen set's fit on the same rows, and the own
           factors matched to the frozen ones (Hungarian on spatial x spectral cosine) -> ``own/<key>.npz``: the
           explained-variance gap (own - frozen) and factor "morphing" over recovery (the 2pRAM manuscript's Fig 4
           analyses iii and the factor-angle analysis, here against one cross-animal pre-stroke basis).

Modules (split from the single script 10-25; every definition moved verbatim, so the comments above carry the
reasoning; `scripts/face_rhythm_factors.py` is the launcher, and `o2.deploy_code` copies launcher + package to O2):
    config   Constants of the factor stages (one place; read by every stage and by `plan`).
    files    Reading the runner's npz files without copies, the set / output folders, plan + frozen-set loading, and
             the sidecar modules O2 imports from the set folder.
    cp       The maths: non-negative CP by HALS (numpy, chunked), projection with spatial + spectral fixed, the FR
             normaliser, pooled EVR, and factor matching / similarity.
    sample   Choosing the fit and session ROWS: engaged trials balanced over positions, between-trial 5 Hz bins, even
             draws over days / groups, and reading the chosen rows off the memory-mapped spectrograms.
    fit      O2 stages `build` (the fit tensor + the frozen point-PCA basis), `fit` (rank sweep, random starts, leave-
             one-animal-out) and the desktop stage `select` (freeze one rank as master.npz).
    project  O2 stage `project`: every session on the frozen set (temporal weights, normaliser D, per-window spectral
             share and point speed, point PCs) -> proj/<key>.npz.
    validate O2 stages that test the set: `splithalf`, `loso` (leave one pre-stroke day out: the noise floor),
             `fitsession` (each session's own fit vs the frozen set) and `fitepoch` (pooled own fit per animal x
             epoch, + the two pre-day halves).
    o2       Desktop stage `plan`: every task's epoch, the fit / session rows, plan.json, the O2 sbatch files, and the
             copy of the code the O2 jobs run (fr_factors.py + this package as fr_factors_lib/ + the sidecar modules).
    report   Desktop stage `report`: rank sweep, stability, leave-one-out and the factor gallery.
    cli      The command line (`python scripts/face_rhythm_factors.py <stage>` on the desktop, `python
             <set>/fr_factors.py <stage>` on O2).
"""
from __future__ import annotations

from .config import (
    EPS, TRIAL_DS, CHUNK, PRECUE_S, RESP_S, BUSY_S, EDGE_BINS, ARRIVAL_S, PRECUE_EXT_S, POINTS_CHUNK,
    SIDECAR_MODULES, EPOCH_GROUPS,
)
from .files import npz_memmap, npz_small, rows_of, set_dir, out_dir, view_out_dir, plan_views, load_plan, load_master
from .cp import ncp_hals, normaliser, project_rows, pooled_evr, match, similarity, match_sets
from .sample import (
    session_candidates, trial_strobes, between_states, even_draw, draw_rows, gather_rows, spec_of, session_group,
)
from .fit import cmd_build, build_points_pca, cmd_fit, spec_hash, cmd_select
from .project import task_basis, window_labels, ClassMeans, window_power, window_speed, points_stream, cmd_project
from .validate import cmd_splithalf, cmd_loso, cmd_fitsession, epoch_fit_tasks, cmd_fitepoch, cohort_fit_tasks, cmd_fitcohort
from .o2 import LIB, cmd_plan, deploy_code
from .report import direction_colours, position_style, cmd_report
from .cli import main

__all__ = [
    "EPS", "TRIAL_DS", "CHUNK", "PRECUE_S", "RESP_S", "BUSY_S", "EDGE_BINS", "ARRIVAL_S", "PRECUE_EXT_S",
    "POINTS_CHUNK", "SIDECAR_MODULES", "EPOCH_GROUPS", "npz_memmap", "npz_small", "rows_of", "set_dir", "out_dir",
    "view_out_dir", "plan_views", "load_plan", "load_master", "ncp_hals", "normaliser", "project_rows",
    "pooled_evr", "match", "similarity", "match_sets", "session_candidates", "trial_strobes", "between_states",
    "even_draw", "draw_rows", "gather_rows", "spec_of", "session_group", "cmd_build", "build_points_pca", "cmd_fit",
    "spec_hash", "cmd_select", "task_basis", "window_labels", "ClassMeans", "window_power", "window_speed",
    "points_stream", "cmd_project", "cmd_splithalf", "cmd_loso", "cmd_fitsession", "epoch_fit_tasks",
    "cmd_fitepoch", "cohort_fit_tasks", "cmd_fitcohort", "LIB", "cmd_plan", "deploy_code", "direction_colours", "position_style", "cmd_report", "main",
]
