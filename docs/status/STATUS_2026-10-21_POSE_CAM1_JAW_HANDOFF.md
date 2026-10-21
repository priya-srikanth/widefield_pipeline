# Handoff — 2026-10-21 (morning): pose / 3-D / cam1 — jaw calls, whole-session licks, per-session tables

**START HERE for pose tracking / 3-D / cam1 / jaw calls.** Supersedes the status line of
`STATUS_2026-10-20_POSE3D_TIMING_HANDOFF.md`, which stays the detailed record (§4a = every finding 10-20 → 10-21 with
numbers; §6 commands; §8 pitfalls 1-21). Method register: `docs/POSE_MODELS.md`. Analysis desktop (shared box); M: =
MICROSCOPE. Write only inside `MICROSCOPE/Priya/...`; never delete original data; never kill python.exe by name; never
WSL LP predict with the 96-frame DALI chunk (use 16). Commit only my own files, staged by name (shared clone with the
ENCODING window, which owns `scripts/trial_attempts.py`, `wfield_local/mixed_models.py`, `scripts/imaging_lmm.py`).

## 0. Resume checklist

1. **Background chain `bfvyguza2` (started 06:17)** processes the 18 new cam1 sessions (PS93 0922 / 0925 / 0929 / 1002 /
   1006, PS94 0606-0817): pose3d production (DONE 06:29; gates: only PS92 0605 fails) -> kinematics cam4 (DONE) ->
   kinematics combined (running at hand-off) -> cam1_licks run (new) -> incomplete readout -> cue + first-lick traces ->
   summary_fig, jaw_crosscheck, phase, spout_error -> (collect DONE 06:34) -> jaw_onsets (all). If the session is gone,
   re-run the remaining steps with §4 commands (all safe to re-run). Log: the task output; collect log in the session
   scratchpad `collect_new.log`.
2. **cam1 on M:** 79 of 137 videos (PS92 36, PS93 31, PS94 12, PS95 0), ALL collected to template-time
   `session_poses/<a>_<d>_cam1_full/`. Priya says O2 has ~84 done -> re-run the rsync (§3), then
   `o2_inference collect --name lp_cam1_ctx_prod_20261019 --missing-only` and the §4 chain for the new sessions.
3. **jaw_onsets.csv** is watched by the encoding window (rewrite -> they re-run trial_attempts + mixed models). Rewrite
   it after every new cam1 batch (`python -m scripts.jaw_onsets --jobs 3`).
4. Git: local main was ahead 3 at hand-off (my jaw-onset commit + the encoding window's); push with fetch/rebase.
   Pushes often collide with the other window ("failed to push" then `git status -sb` level = it landed; else retry).
   Pre-push runs the full test suite (~7 min); `tests/test_no_writer_bypasses_figure_layout.py` forbids building
   `*_sessions*.csv` paths by hand -- use `figure_layout.sidecar_for`.

## 1. Done since the 10-20 handoff (details: 10-20 doc §4a)

* **Jaw anchor OFF** (`pose3d.ANCHOR_PARTS = ()`): open one-camera jaws misplaced 0.5-1.4 mm; a depth model (held out
  0.25-0.31 mm) still extrapolated at close_center. Centre-spout 3-D jaw ~0.81-0.86 coverage, close_center opening
  biased low -- keep it out of opening comparisons. FAR_DIR / CLOSE_DIR unaffected.
* **3-D / cam1 figures and analyses** (all on template time): cue- and FIRST-LICK-aligned 3-D traces
  (`scripts/pose3d_cue_traces.py --align cue|first_lick`, incl. aim angles `mid` / `elev`), spout-error epoch figures
  (lateral + = ipsi, zoomed, SEM bars; `reach_epoch_*`), CLOSE_DIR sheet, check sheet, far_center diagonal = real rod
  contact. Key results: acute far_R = a LATE full-size first lick then no train; aim shifts ipsi at every spout;
  chronic close-spout licking is earlier / more cue-locked (practice in PS92 / PS93; licks-per-trial up in all four).
* **Whole-session licks** (`orofacial_outputs all-licks` -> per v36 session `licks_all.csv` + coverage; cam1
  `<s>_events_all.csv`): context pre_cue / post_cue / iti by onset (DAQ .h5 strobe / trial_end), absolute DAQ time,
  post-cue rows identical to licks.csv.
* **Per-session long tables** for the epoch families (`scripts/epoch_sessions_long.py`, August baseline;
  `<labcams>/grant_figures/epoch/data/<stem>_sessions_long.csv`, 40 files) + CD (`cd_window_projection_sessions_long.csv`,
  all-pre fit, pre in-sample). Validated against the deck files.
* **cam1 windows re-collected on template time** (all 44 old ones were frame-index time) + `collect --sessions /
  --missing-only`.
* **Jaw onsets** (`jaw_kinematics.jaw_onsets`, `scripts/jaw_onsets.py` -> `<v36>/imaging_encoding/qc/jaw_onsets.csv`):
  the "jaw moved" rule + onset + re-onset for jaws deflected at the cue; **cam1 floor 15 px** (cam4 4 px); BOTH cameras
  masked at close_center. cam1 vs cam4 "moved" agree 95.7 %; onsets median -0.7 ms (IQR ±8).
* **Blind reviews of the cam1 jaw call** (Priya): frames (`jaw_review/scored.csv`, 58 % with the old 4 px call;
  close_center at chance) and clips (`jaw_review_video/scored.csv`: < 12 px mostly still, >= 16 px nearly always moved,
  12-16 a coin flip) -> the 15 px floor. Priya: frame review is weak ground truth for subtle moves; clips better.
* **Attempted gate**: `orofacial_cohort.engagement_keep(..., gate="attempted"|"canonical_attempted")` (+ env
  OROFACIAL_GATE), reads trial_attempts.csv; default "reference" unchanged.
* `OROFACIAL_FINDINGS.md`: far-contra protrusion row carries the per-animal caveat.

## 2. Decisions and why

| decision | why |
|---|---|
| no one-camera jaw depth (anchor off) | open jaws misplaced; the depth model extrapolates where both cameras never see an open jaw |
| cam1 jaw floor 15 px, cam4 4 px | two blind reviews: 4 px is jitter at cam1's larger scale; >= 16 px reliably real |
| jaw unknown at close_center (both cameras) | cam4 chin hidden; cam1 jaw point jumps under the rod (review at chance) |
| re-onset for at-cue deflections | Priya: "a deflected jaw at the cue doesn't mean no reaction" |
| licks_all from the 3 v36 tables, no re-detection | the windows already hold 98.6 % of the video with one detector |
| long tables reproduce collector rng order | 8g / 11c are random; validated post rows == deck values |
| 10 / 10b pre per-session (not argmax of the averaged matrix) | the averaged-matrix argmax is biased high |
| nose / whisker reactions = a SEPARATE class (proposed, not built) | a reaction is not an attempted lick |

## 3. O2 (Priya runs)

```
B=/n/scratch/users/p/ps150/o2/lp_cam1_ctx_prod_20261019
ls $B/out/*/*_done.json | wc -l
squeue -u ps150 -j 56830841 -t PD -o "%.20i %.14P %.25b %.20r" | head
# transfer node:
rsync -ah --partial --exclude 'tmp*' $B/out/ /n/files/Neurobio/MICROSCOPE/Priya/DeepLabCut/Widefield/o2/lp_cam1_ctx_prod_20261019/out/
```
Array 56830841: pending tasks ask `gres/gpu:1`, wait on Priority (fairshare). 10 non-curated tasks cancelled.
`gpu_requeue` accepts 10 h jobs but its estimated start was ~1 day out. Local cam1 LP run (reverse order from task 137
= PS95 1005; WSL, sequence_length 16, ~10-12 h / video) DEFERRED (box shared) -- if started, `scancel 56830841_<idx>`.

## 4. Commands for new cam1 sessions (repo root; locanmf env unless marked dlc)

```
python -m wfield_local.o2_inference collect --name lp_cam1_ctx_prod_20261019 --missing-only
python -m scripts.pose3d production A:D ... --jobs 3                       # dlc; merges into summary_production.csv
python -m scripts.pose3d_kinematics ; python -m scripts.pose3d_kinematics --licks combined   # SERIAL (12 GB each)
python -m scripts.cam1_licks run A:D ... --jobs 3                          # events + events_all
python -m scripts.cam1_incomplete_licks
python -m scripts.pose3d_cue_traces --align cue --jobs 3 ; ... --align first_lick
python -m scripts.pose3d_summary_fig / pose3d_jaw_crosscheck / pose3d_phase / pose3d_spout_error
python -m scripts.jaw_onsets --jobs 3                                      # watched by the encoding window
python -m wfield_local.orofacial_outputs all-licks                         # only if v36 changes
WIDEFIELD_PRE_FROM=0801 python -m scripts.epoch_sessions_long ; python -m scripts.epoch_sessions_long --families cd
```
Memory: the other window can hold ~25 GB; check free RAM first; kinematics serial; pose3d 3 workers.

## 5. To-do

1. Copy back the remaining cam1 outputs (§3) -> collect -> §4 chain (first PS94 3-D / cam1; PS95 none yet) ->
   per-animal CLOSE_DIR, incomplete licks, cue / lick traces, aim angles for all four animals (rule 8).
2. Optional: nose-movement call (same rule as the jaw, both cameras) as a separate "reacted" class; whiskers need a
   motion-energy extraction. Priya to decide.
3. Optional: 3-D lick-aligned figures by epoch for PS94 once its 3-D lands; re-check practice-vs-stroke with PS94.
4. CD long table is in-sample for pre (fits used all pre incl. June); a LOO CD refit would be a separate costly job.
5. Encoding window owes nothing back to me; they re-run on `jaw_onsets.csv` rewrites.

## 6. Pitfalls (new 10-21; older: 10-20 doc §8)

22. **`collect` rewrote every finished session** (~5 min each) -- use `--missing-only` / `--sessions`.
23. **Old `cam1_full` folders were frame-index time** (no `t_daq_ms`): anything reading them must check the column.
24. **`licks_precue.csv` / `licks_iti.csv` time on_ms / off_ms from their SEGMENT start**; only `t_ms` is from the cue
    (licks_all converts).
25. **`df.align` is a pandas method** -- use `df["align"]` for a column named align.
26. **Importing `scripts/pose3d_jaw_anchor_check.py` RUNS it** (module-level body); copy constants instead.
27. **Unquoted shell heredocs eat backslashes** (`\\r` -> `\r`) -- write patches as scratchpad .py files.
28. **CD dumps refuse to load under WIDEFIELD_PRE_FROM** (cohort guard) -- run CD-only exports without it.
29. **Push collisions with the other window**: re-fetch; "failed to push" may still have landed (check `git status`).
