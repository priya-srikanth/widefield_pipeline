# Handoff — 2026-10-21 (morning): pose / 3-D / cam1 — jaw calls, whole-session licks, per-session tables

**START HERE for pose tracking / 3-D / cam1 / jaw calls.** Supersedes the status line of
`STATUS_2026-10-20_POSE3D_TIMING_HANDOFF.md`, which stays the detailed record (§4a = every finding 10-20 → 10-21 with
numbers; §6 commands; §8 pitfalls 1-21). Method register: `docs/POSE_MODELS.md`. Analysis desktop (shared box); M: =
MICROSCOPE. Write only inside `MICROSCOPE/Priya/...`; never delete original data; never kill python.exe by name; never
WSL LP predict with the 96-frame DALI chunk (use 16). Commit only my own files, staged by name (shared clone with the
ENCODING window, which owns `scripts/trial_attempts.py`, `wfield_local/mixed_models.py`, `scripts/imaging_lmm.py`).

## 000. LATE-MORNING UPDATE (10-21 ~10:35) -- read first, then §00

**DONE 10:54:** the watched `jaw_onsets.csv` was rewritten (62,771 rows; every existing column verified identical to the morning file; 9 appended columns, 26 total) -> the encoding window's watcher re-runs. Previously RUNNING: background `jaw_onsets` full rebuild (started 10:15, ~40 min) with the APPENDED columns
below -> writes `$TMP/jaw_onsets_v2.csv`, checks every EXISTING column against the morning snapshot
(scratchpad `regress/jaw_onsets.csv`) and ONLY THEN copies it over the WATCHED
`<v36>/imaging_encoding/qc/jaw_onsets.csv` (the encoding window's watcher then re-runs trial_attempts + mixed models).
If the session died: re-run `python -m scripts.jaw_onsets --jobs 3` (cache invalidated by the code change -> cold) and
verify the old columns are unchanged before telling the encoding window.

**Attempt review DONE (24 / 24 answers, page TbQ4pud711MBrT6ygAH5FX, db `attempt_review`, key
`session_poses/attempt_review/key.csv`): 0 / 24 judged attempts.** jaw_only (11) = jaw response without opening 6,
other movement 4 (2 walking), still 1; no_try just under the floor: 4 / 6 jaw responses; no_try inside the old gate:
4 / 6 still; tags walking 5, spout_leaving 2. RECOMMENDATION sent to the encoding window (their file): attempted = hit +
incomplete only; jaw_only -> a separate "reacted (jaw)" class. They started a jaw-only rerun (fine: existing
jaw_onsets columns unchanged).
**jaw_onsets APPENDED columns** (code committed with this hand-off): `cam4_open_px` / `cam1_open_px` (largest opening,
+y both views), `cam4_up_px` / `cam1_up_px` (jaw up / clench), `jaw_dir` (open / up, onset camera), `cam4_contrast` /
`cam1_contrast` (mean |jaw speed| cue .. min(stop, 1.5 s) / -1 .. 0 s; ~1 = no cue response), `tread_pre_mm_s` /
`tread_post_mm_s` (DAQ treadmill). Reviewed jaw_only trials: contrast 0.90-1.04, opening 5-12 px (lick openings ~60-90
px); a001 walking at 8 mm/s, a014 2.1 mm/s pre-cue.
**cam4 ENCODER KEYFRAME artefact:** every video frame with index % 12 == 0 (from the video start) has ~33 % more
frame-difference energy (20.8 Hz + 41.7 / 62.5 Hz harmonics; a static background corner peaks at 12.01 frames, 190x).
Fix = interpolate over those frames (20.8 Hz power down 4-8x): `scripts/attempt_review.py` (KEYFRAME_EVERY),
documented in `wfield_local/roi_motion.py`. Told the FaceRhythm window (interpolate point displacements over those
frames before the VQT, or expect a 20.8 Hz band). The published review clips predate this fix (5-frame median only).

**FaceRhythm registration FINAL (10-21 ~11:30; supersedes the smoothing paragraph below where they differ):**
* EYES: Priya clicked eye centres on the master + 3 templates (page https://claude.ai/artifact/Duo8fgvKosCnXvMyFDjqho,
  db `eye_landmarks`) -> `face_rhythm/cam4/register_inputs/eye_landmarks.json` (outside register/ so a re-run keeps
  it). They join nose + jaw in the template -> master similarity fit (equal weight; "eyes once" was worse: PS94 eye
  7 -> 12.5 px). Raw eye zones were tried and DROPPED (`RAW_EYES = ()`): DeepFlow folded around the eyes (bunching median
  0 -> 24). Fixes: eyes now inside every session's face contour; PS94_0823's image-left cheek contour follows the face.
* FOLD REPAIR: where the warp's Jacobian determinant < 0.6 (grown 25 px, feathered), the displacement is replaced by a
  20 px smoothing of it (not in the mouth zone): worst bunching 82 -> 12.
* LOCAL-QUALITY FALLBACK: where the similarity-only warp beats the full warp by > 0.05 windowed NCC (41 px), the
  similarity field is used there (feathered; mouth kept full).
* Result (final run): eye 5-7 px, nose 1.6-4.9 px, jaw 3.3-5.7 px EXCEPT PS92 jaw ~7.6 px (was 2.8 before the eyes;
  OPEN: probably PS92's template -> master step; ~0.2 mm), bunched points median ~6 / max ~26, mostly in the raw mouth
  zone. Field sets kept for comparison: register_v1_postmaster / v2_unsmoothed / v3_threezone / v4_foldrepair_20261021.
* Per-session ROIs REWRITTEN from these fields (b69f1jwnu at hand-off; check `register/rois_check.png` + 107 files) ->
  Priya rsyncs rois/ to O2 after looking at the sheet. QC page https://claude.ai/artifact/Ffm8u7RseBs5NPvmPa5JRY
  (republish `scratchpad/face_registration_qc.html` after regenerating if the session is still alive).

**FaceRhythm registration SMOOTHING (10-21 ~11:00, after Priya saw wiggly contours / bunched points):** the DeepFlow
field now has THREE zones (`face_rhythm_register.register`): MOUTH (within MOUTH_R 80 px of the template's rest jaw,
ramp SHELL_PX) = raw field (Priya: keep the mouth edge); face INTERIOR (inside Priya's master face polygon, carried into
each animal template through its template->master field) = INNER_SMOOTH_PX 8 px Gaussian; OUTER SHELL (50 px band inside
the contour, ramp) and outside = SMOOTH_PX 25 px. Results vs unsmoothed: face landmark error UNCHANGED (jaw 2.8-4.3, nose
1.6-5.2 px), mouth 0.0 px change, bunched points (< 4 px neighbour of 2,287) median 45 -> 0, worst 305 -> 82 (PS93: its
template->master field had fold seams along the left pad / under the snout, inherited by every PS93 session), spacing p5
4.9 -> 6.5 px (`register/qc_spacing.csv`, `register/qc_bunching.png`). Uniform 10-20 px smoothing was rejected (cost PS92
/ PS93 jaw 3-4 -> 6-7 px); a radius-around-the-nose blend did not follow the face. Earlier field sets kept:
`register_v1_postmaster_20261021/` (PS93 master), `register_v2_unsmoothed_20261021/`. The per-session ROIs were REWRITTEN
from the 3-zone fields (each rewrite moves the old file aside as ROIs_<time>.h5) -> rsync to O2 only this version.

**FaceRhythm ROIs WRITTEN:** Priya saved the master ROI on PS95_20260812 (10:21; 2,287 points; earlier versions kept as
ROIs_<time>.h5). `face_rhythm_warp_rois.py to-sessions` -> `face_rhythm/cam4/rois/<a>_<d>/ROIs.h5` for all 107
non-June sessions (106 + master; PS92_0812's two videos share one), same 2,287-point mesh, none outside the image;
pre-existing files moved aside (PS94_20260810 pilot, PS93_20260821 draft). NEXT (Priya): rsync rois/ to O2 per the
FaceRhythm window's COMMANDS.md. Registration QC page: https://claude.ai/artifact/Ffm8u7RseBs5NPvmPa5JRY (cross-animal
sheet `register/qc_cross_animal.png`: snout / outline aligned, fur texture differs, a green / magenta band at the lower
lip although the jaw landmark lands within 3-4 px -- check that region; per-animal `qc_overlay_<a>.png`;
`rois_check.png`). PROPOSED: per-session facial ASYMMETRY index from the fields (PS93: right lower face higher, jaw
deviates left -- Priya).

**CD:** `scripts/cd_loo_sessions.py` (+ traces, rerun 5 min with caches) and `scripts/cd_loo_figures.py` ->
`grant_figures/epoch/cd_loo/cd_loo_{window,traj}_{precue,cue,lick}.png`. Encoding window: use `epoch:cd_loo`.

## 00. MIDDAY UPDATE (10-21 ~10:00) -- §000 above supersedes where they differ; §0-§6 below are the morning state

**Pipeline efficiency (Priya: "saved arrays, caches, parallelization ... less redundant"). All verified to reproduce
the morning outputs EXACTLY** (snapshots in the session scratchpad `regress/`; jaw_onsets 62,771 rows, kinematics
cam4 + combined, phase 1.68 M rows, spout error, cue + first-lick traces: max diff 0, 0 NaN mismatches):
* `wfield_local/pose_windows.py`: binary `windows.npz` sidecar next to each cam1 `windows_index.csv` / `windows_LP.csv`
  (built once, rebuilt when a CSV is newer; LP float32 = model precision, < 3e-5 px from the CSV parse; index exact).
  `o2_inference.full_to_windows` builds it on collect; `python -m wfield_local.pose_windows` builds all (78 done).
  Readers switched: jaw_onsets, nose_onsets, attempt_review, whisker_motion, cam1_jaw_review (NOT session_poses /
  jaw_lateral / cleaned_label_clips / pose_cue_traces: they read old DLC folders; a sidecar there is 200 MB for a count).
* `wfield_local/result_cache.py`: per-session result cache keyed on input files (size + mtime) + a hash of the code
  (`code_version`), pickles under `<out>/_cache/<step>/`. Used by jaw_onsets, pose3d_kinematics (now a parallel
  per-session worker, ~1 GB each instead of 12 GB serial), pose3d_phase + pose3d_spout_error (one exact `get_indexer`
  per position instead of a reindex per lick), pose3d_cue_traces (key includes the behaviour table + engagement gate).
  Cached reruns: kinematics 16 s (was 6 + 22 min), phase 74 s (was 17 min). `WFIELD_RESULT_CACHE=0` disables reads.
  OPEN: the cached jaw_onsets rerun still took 13 min on the loaded box -- not yet diagnosed.
* `scripts/pose_chain.py`: the whole §4 chain in order (collect + sidecars -> pose3d production -> kinematics x2 ->
  cam1_licks + incomplete -> cue / first-lick traces -> spout_error -> summary_fig + jaw_crosscheck -> phase ->
  jaw_onsets LAST), one command after an O2 copy-back; logs `session_poses/pose_chain_logs/`; `--from`, `--only`,
  `--dry-run`. Its dry run found PS93_20260918 (cam1 copied back after the 06:17 production) -> picked up next run.
* `--out` on jaw_onsets / kinematics / phase / spout_error / cue_traces for checks (jaw_onsets.csv is WATCHED).

**Attempt / reaction review** (`scripts/attempt_review.py`; page https://claude.ai/artifact/TbQ4pud711MBrT6ygAH5FX,
db collection `attempt_review`, key `session_poses/attempt_review/key.csv`): 24 post-stroke trials, 4 strata (gate
adds / removes, jaw_only / no_try near the floor), clips -300 ms .. trial stop + 200 ms, cam1 + cam4, traces: jaw x/y
both cameras, cam4 nose, tongue likelihood, cam4 WHISKER-PAD motion energy (5-frame median: the cam4 videos carry a
one-frame ME spike every ~15 frames = keyframes), DAQ TREADMILL speed (1 mm/s guide). Answers: attempted / jaw
response not opening / other movement / no movement / unsure + tags walking, spout_leaving. First pass: only 5 answers
saved (page did not show failed saves -- fixed: counter reads back from the store, red banner on failure); none of the
5 was an attempt (2 near-floor jaw_only judged non-attempts). Priya's notes: WALKING (below the running threshold)
jiggles the jaw without a cue response; some jaw responses when the SPOUT LEAVES (spout departs 60-80 ms after the
trial stop, measured; detection ends at stop, so these are not counted). PROPOSED (not built): jaw_onsets columns for a
cue CONTRAST (post vs the trial's own pre-cue movement) and pre / post treadmill speed; jaw deflection DIRECTION
(jaw-up responses currently count as moved). Decide after Priya's re-review.

**Nose call** (`scripts/nose_onsets.py`, qc/nose_onsets.csv): NOT used for any class (Priya) -- not specific.
**Whisker-pad ME** (`wfield_local/roi_motion.py`, `scripts/whisker_motion.py`, pilot PS92 0818 / PS93 0922 / PS94
0810 / PS95 0820): class averages = licking (hits) only, flat on jaw_only / no_try -> not run cohort-wide; FaceRhythm
is the route to whisking.

**CD leave-one-out** (`cd_trajectories.analyse_animal(exclude_fit=, project_only=)`, `scripts/cd_loo_sessions.py`,
WIDEFIELD_PRE_FROM=0801): `grant_figures/epoch/data/cd_window_projection_loo_sessions_long.csv` (family `cd_loo` --
NOT `cd`: imaging_lmm globs *_sessions_long.csv and would merge it with the in-sample file) + `data/cd_loo_traces.npz`
(per-session diagonal time courses). Figures `grant_figures/epoch/cd_loo/cd_loo_{window,traj}_{precue,cue,lick}.png`
(`scripts/cd_loo_figures.py`). Held-out pre means 0.66-0.87 (pre-cue), i.e. in-sample inflation removed. Encoding
window: use `epoch:cd_loo` (full models incl. the pre step); old `epoch:cd` stays post-only.

**FaceRhythm** (plan docs/FACE_RHYTHM_PLAN.md; O2 bundle by the FaceRhythm window: scripts/o2_face_rhythm.py,
fr_cam4_20261021, 108 tasks, June excluded):
* `scripts/face_rhythm_prep.py frames` -> `<dlc>/face_rhythm/cam4/frames/<s>.npz` (median rest face + LP landmarks), 119.
* `scripts/face_rhythm_register.py` (fr env): templates = medoid of each animal's AUGUST PRE sessions
  (`face_rhythm/cam4/pre_august_sessions.json`), MASTER = PS95_20260812 (PS93 excluded: right orofacial deficit;
  FaceRhythm window + Priya 10-21). Coarse = landmark similarity (face weighted x3), fine = DeepFlow on the
  similarity-warped image, CHAINED by us (fr's remappingIdx_init applied the init twice). Face error PS92-PS95 1.5-6 px
  in every epoch (PS93 post no worse than its pre: the non-rigid step absorbs the lesion asymmetry); June = similarity
  fallback (camera change) -> excluded. The first run (post-stroke templates, master PS93_20260821) is kept in
  `register_v1_postmaster_20261021/`.
* `scripts/face_rhythm_overlay.py` -> `register/master_overlay.npz` (every session's tips / nose / jaw / reach in master
  px). BLOCKED at hand-off: the FaceRhythm notebook holds the file open (0-byte write) -> restart its kernel, rerun.
* `scripts/face_rhythm_warp_rois.py`: `carry` (Priya's polygons PS93_20260821 -> PS95_20260812 done: 79 / 46 vertices,
  2,207 points), `to-sessions` (NOT RUN: after Priya re-checks on PS95_20260812; moves existing ROIs.h5 aside, skips June).
* Proposed: a per-session facial ASYMMETRY index from the registration field (lower-face L vs R displacement from the
  animal's own pre template), beside LP rest-jaw `jaw_ml`.

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
**10-22 UPDATE:** fairshare tanked by the FR array (`short`, 24 x 8 CPUs) -> no GPU starts since 10-21 afternoon
(gpu_quad estimate 10-25). Pending tasks 112-137 MOVED: `scancel -t PENDING 56830841`, then
`sbatch -p gpu_requeue --gres=gpu:1 --array=112-137%12 $B/run_pose.sbatch` = **array 57249401** (estimate ~10-23 06:30;
preemptible -> a preempted video restarts). `-p gpu` multi-partition gives "Invalid qos"; `-p gpu` alone estimated
10-24. 100 / 137 videos done on 10-22 (98 curated sessions through pose_chain). Suggested `ArrayTaskThrottle=6` on FR.
**10-23:** 124 / 137 done (all curated except 112 running, 119 + 127). 119 (PS95_0817) and 127 (PS95_0831) FAILED in
seconds on **compute-gc-17-242** (L40S x4, MIXED): "no GPU visible to torch" -> resubmit with
`--exclude=compute-gc-17-242` (EXCLUDE THIS NODE in future cam1 / LP submissions). gpu_requeue preempts (112 lost 28 min
at 34 % once); A100 there ran 3.6 it/s (~84 min / video). gpu_quad test-only 10-23 still later (10-25) than gpu_requeue.
**pose_chain now defaults to --jobs 6 and collects in parallel** (856bcfc; 10-session batch was ~66 min, collect 19 min serial).
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
