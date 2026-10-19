# Handoff — 2026-10-19 (evening): cam1 production on O2, the 3-D tongue / jaw pipeline, first 3-D readouts

**START HERE for pose tracking / 3-D.** Supersedes `STATUS_2026-10-18_CAM1_ROUND2_3D_HANDOFF.md` (still the reference for
cam1 round 2, model scoring, cleaning). Method detail and every number: `docs/POSE_MODELS.md` **D10-D12** (3-D), D8-D9 (cam1
cleaning / lateral readout). Decisions: `DECISIONS.md` 2026-10-19 (evening). Acquisition fact found today:
`docs/EXPERIMENT_ERRORS.md` "Post-stroke — mouth set point placed closer (slack jaw)". Analysis desktop; M: = MICROSCOPE.

## 0. Resume checklist (in order)

1. **O2 cam1 production** (array **56830841**, bundle `o2/lp_cam1_ctx_prod_20261019`, 137 cam1 videos, LP ctx-semi): check
   progress (§5), have Priya copy finished outputs back (§5), then run the LOCAL chain (§6). As of 10-19 evening: 20 videos
   back (all PS92: 0602-0608, 0805-0814 incl. split day 0812, acute 0818-0820). O2 fairshare is depleted, so it is slow.
2. **Background jobs that were running locally at hand-off** (§3): read their outputs; if the session is gone, re-run them
   with the commands given there.
3. **Not yet run**: the jaw cross-check (`scripts/pose3d_jaw_crosscheck.py`, after the chain), the depth-prior test
   (`scripts/pose3d_depth_prior_test.py`), and — whatever the jaw gap test says — set the jaw's fill limit (§7).
4. When all four animals are in: cohort tables (per-animal first, then `analysis_kit.boot_delta`), §7 item 6.

## 1. What was done (10-19)

* **cam1 model + production**: LP ctx-semi chosen (most faint licks after cleaning, no emergence lag); O2 production bundle
  built and running (`wfield_local/o2_inference.py --cam cam1`; collect writes `session_poses/<a>_<d>_cam1_full/`).
* **3-D pipeline** (`wfield_local/pose3d.py`, `scripts/pose3d.py`): per session, cut both cameras' O2 output into trial
  windows on the DAQ clock (`o2_inference.session_windows`) → June cam4 x0.8853 about the principal point → spout medians per
  position → **PnP camera poses on the commanded spouts** (09-11 intrinsics only; spout edge offsets) → clean each camera with
  its own rules (MEASURED points only) → pair frames exactly on (trial, t) → triangulate (aniposelib, undistort + DLT) →
  **fill one-camera frames** on the seen camera's ray (depth from both-camera neighbours <= 24 ms, Akima) → **plausibility
  gates** → `session_poses/pose3d/<a>_<d>_frames.csv.gz` (per frame: tongue / jaw / nose / spout ml, ap, dv mm; reprojection;
  source 1 both / 2 cam1 / 3 cam4).
* **Readouts** on top: `pose3d_kinematics` (cam4 lick events or `--licks combined`: 3-D peak, direction vs target, reach,
  speed, jaw opening + FAR_DIR), `pose3d_spout_error` (closest approach, endpoint vs spout, endpoint vs own pre endpoint),
  `pose3d_crosscheck` (cam4 2-D vs 3-D, lick by lick), `pose3d_jaw_crosscheck` (written, not run), `pose3d_vs_cam4` (does
  3-D add information: decoding + effect sizes; running), figures: `pose3d_figures` (maps; 2 x 2 paired video with tongue +
  jaw), `pose3d_summary_fig` (per session over days), `pose3d_phase` (peak-aligned distance, phase portrait, aim vs phase,
  trajectories, speed / velocity traces).
* **Repo tooling**: pre-push hook skips the test suite when only `.md` files change (Priya); suite ~7 min otherwise.

## 2. Decisions and why (full list: DECISIONS.md 2026-10-19 evening)

| decision | why |
|---|---|
| cam1 production = LP ctx-semi | most faint licks after cleaning; no emergence lag (POSE_MODELS Part B) |
| cameras posed per session by PnP on the commanded spouts | 09-11 extrinsics: every part 6-10 px apart between cameras (sideways error + drift); PnP ~1 px, stage mm directly, absorbs the moved June cam4 |
| 09-11 intrinsics kept | lens properties; validated by nose / jaw agreeing to ~1 px between cameras (not used for posing) |
| spout edge offsets: cam4 = superior edge = mouth zero, cam1 = 1.24 mm below | the mouth zero was set with the UPPER spout tip (Priya); rod diameter measured 1.23-1.31 mm |
| mouth = the spout mouth zero only | jaw and nose can change post-stroke (Priya); the set point marks the mouth that day |
| no v36 re-run for the June scaling centre | the centre adds a constant that cancels in every 2-D measure; only 3-D needs the principal point |
| one-camera fill: Akima, <= 24 ms, ray-miss gate 3 mm | held-out error 0.13-0.23 mm; Akima beats linear / PCHIP on long gaps; a 1 mm gate dropped half the real fills |
| gates: > 8 mm out, > 2 mm behind the mouth, spikes > 1.5 mm; jaw > 5 mm | paw-in-face frames (both models call them tongue at 0.9+) triangulate 7-36 mm out; reprojection does not catch them |
| readouts: distances primary (3-D mm) + 3-D angle error + midline-plane angle | distances are origin-free; angles vs the spout direction are consistent across the set-point move (spouts commanded from it) |
| peak = 3-D frame farthest from the mouth zero | cam4's peak is the max of its DOWN axis |
| lick events: cam4's (comparable) + combined 3-D track | combined adds 3-7 % real small licks (47 % touched the spout) |

## 3. Running / pending at hand-off

| what | how it was started | output | if lost, re-run |
|---|---|---|---|
| O2 array **56830841** (cam1 production) | Priya, 10-19 ~07:40; pending tasks opened to any GPU, 16 h (`scontrol update ... Gres=gpu:1 TimeLimit=16:00:00`); V100 nodes excluded | scratch `o2/lp_cam1_ctx_prod_20261019/out/` | §5 |
| LOCAL chain (3-D for all copied sessions with the 3 mm fill gate → kinematics (both lick sets) → spout error → summary fig → phase / speed figs → `pose3d_vs_cam4`) | background task `b4npluehh` | `session_poses/pose3d/` | §6 (all of it) |
| LOCAL jaw gap test, then the jaw + tongue paired videos (0807, 0818) | background `bvrlb6zlo` | stdout; `pose3d/figures/lick_paired_*_cam4view.mp4` | `python -m scripts.pose3d_jaw_gap_test` (dlc env); `python -m scripts.pose3d_figures video PS92:20260807 PS92:20260818 --view cam4` |

## 4. Results so far (preliminary: PS92 = 9 August pre + 3 acute; PS93 = trial clips only, ~10 % of licks pair)

* **cam4 vs 3-D**: cam4 lateral r 0.95-0.97 with 3-D ml (faithful → the cam4 ipsi-aim + jaw-steering findings stand); cam4
  'out' / protrusion = 99 % the DOWN axis (r ~0 with forward) → its acute 'shorter protrusion' = less downward extension.
  3-D has 8-9 % more truly measured tongue frames than cam4 (cam1 covers cam4's occlusions, often at contact).
* **Acute (PS92)**: aim shifted ipsi from lick ONSET through the whole lick at every position (+5..+13 deg close / centre,
  ~+30 deg far_R; the far_R mean path crosses the midline); far_R closest approach 3.2 vs ~1 mm, lateral -3.0 mm; slower
  protrusion (far_R radial ~100 vs ~160 mm/s; peak speed below the pre range at every far spout); jaw opens less, FAR_DIR
  0.26 -> 0.02 mm. PS93 (clips) agrees in direction.
* **Pre-stroke geometry**: licks end 1.4-2.6 mm BELOW the upper spout tip (the drop); closest approach 0.65-1.1 mm; licks run
  steeper than the line to the spout (3-D angle error 25-40 deg pre) → read angles as change from pre.
* **June** sits off the August band on several measures (residual geometry or early training) → August-only baseline.

## 5. O2: check jobs and copy outputs (Priya runs these)

Login node (`ssh ps150@o2.hms.harvard.edu`):
```
B=/n/scratch/users/p/ps150/o2/lp_cam1_ctx_prod_20261019
ls $B/out/*/*_done.json 2>/dev/null | wc -l                      # finished videos (of 137)
squeue -u ps150 -j 56830841 -h -t R | wc -l                       # running
squeue -u ps150 -j 56830841 -h -t PD -r | wc -l                   # pending
squeue -u ps150 -j 56830841 -t PD -o "%.18i %r" | head -3         # why pending (Priority / Resources = GPUs / fairshare)
squeue -u ps150 -j 56830841 -t R -o "%.14i %.10M %.20N %b"        # where running
sacct -j 56830841 --state=FAILED,TIMEOUT,OUT_OF_MEMORY -X -o JobID,NodeList,State,Elapsed   # send to Claude if not empty
grep -h "frames in" $B/logs/56830841_*.out                        # per-video timings (L40S ~180 fps; others 87-146)
```
Copy back (transfer node `ssh ps150@transfer.rc.hms.harvard.edu`; partial copies are fine, re-run any time):
```
rsync -ah --partial --exclude 'tmp*' /n/scratch/users/p/ps150/o2/lp_cam1_ctx_prod_20261019/out/  /n/files/Neurobio/MICROSCOPE/Priya/DeepLabCut/Widefield/o2/lp_cam1_ctx_prod_20261019/out/
```
A video is complete when its `_done.json` exists (written last); every local step uses only those. If tasks fail on a new
node type: `scontrol update JobId=56830841 ExcNodeList=<old list>,<new node>` (pending tasks only).

## 6. Local chain after each copy-back (analysis desktop, repo root)

```
# locanmf env (C:/Users/SabatiniLab/miniconda3/envs/locanmf/python.exe); dlc env for anything importing aniposelib
python -m wfield_local.o2_inference collect --name lp_cam1_ctx_prod_20261019                     # locanmf; frame-count check
python -m scripts.pose3d production --jobs 4                                                     # dlc; ALL sessions in both bundles
python -m scripts.pose3d clips PS93:20260814 PS93:20260821 PS93:20260908 --jobs 3                 # dlc; until PS93 production lands
python -m scripts.pose3d_kinematics                     ;  python -m scripts.pose3d_kinematics --licks combined   # locanmf
python -m scripts.pose3d_spout_error                    ;  python -m scripts.pose3d_crosscheck
python -m scripts.pose3d_jaw_crosscheck                 ;  python -m scripts.pose3d_vs_cam4
python -m scripts.pose3d_summary_fig                    ;  python -m scripts.pose3d_phase
python -m scripts.pose3d_figures maps PS92:20260807 PS92:20260818 PS92:20260607                   # dlc
python -m scripts.pose3d_figures video PS92:20260807 PS92:20260818 --view cam4                    # dlc
```
Outputs under `M:/MICROSCOPE/Priya/DeepLabCut/Widefield/session_poses/pose3d/` (frames, `summary_*.csv`, `kinematics/`,
`kinematics_combined/`, `spout_error/`, `crosscheck/`, `vs_cam4/`, `phase/`, `figures/`). Comparison snapshots kept:
`nofill_20261019/`, `noedge_20261019/`. The duplicate `kinematics_setpoint/` can be deleted by Priya.

## 7. To-do

1. **Jaw fill limit**: read the gap test; if the error stays near the floor at 48-100 ms, give the jaw its own limit
   (`session_3d(fill_gap_ms=...)` is shared today -- add a per-part limit) and re-run.
2. **Jaw cross-check** vs cam4 (script ready); then document in POSE_MODELS D10-D12.
3. **Depth prior** for cam1-only small licks: run `scripts.pose3d_depth_prior_test` (dlc env, ~20 min); adopt (a) closest
   point to the mouth zero or (b) the learned prior only if the held-out error is small AND (b) does not bias acute licks
   toward pre; flag prior-based points with their own source code.
4. **2-D vs 3-D** (`pose3d_vs_cam4`): read decoding + effect sizes; decide what is headline 3-D vs supporting.
5. Combined-track licks: re-run kinematics / figures on them once the depth prior is settled; consider making them default.
6. **Cohort**: when PS93 / PS94 / PS95 production lands, curated sessions only, August pre baseline, per-animal tables first
   (CLAUDE.md rule 8), then `analysis_kit.boot_delta`; same epochs as the cam4 tables.
7. Labelling (only if needed): paw-in-face frames as hard negatives (tongue blank) for both cameras.
8. June 0602-0604 (cam4 moved) and PS92 0605 (cam4 template failure) are outside the curated set; leave unless needed.

## 8. Pitfalls (new today)

1. **Heredocs with Python code in Git Bash break on quotes / backslashes** (`"\\n"` became real newlines; `'` inside
   `<<'EOF'` blocks). Write patches to a scratchpad `.py` and run it.
2. **Scratchpad file names can shadow packages**: a scratch `coverage.py` was imported by numba (`import coverage`) and
   crashed aniposelib. Never name scratch files after modules.
3. **`scripts.pose3d production <subset>` OVERWRITES `summary_production.csv` with only that subset**; downstream
   `pose3d_kinematics` reads the fit check from it. Re-run the full production set after a subset test.
4. **Never read `*_frames.csv.gz` while a chain is rewriting them** (truncated gzip → EOFError). Wait for the chain.
5. **Do not silence step output** (`>/dev/null`): a failing step (kinematics skipping every session after the fit check
   broke) looked like success. Grep for `-> ` / `rror` / `skipped` instead.
6. **Changing the PnP object points needs the fit check changed too** (`_identity_world` triangulates with the offsets), or
   every session reads > 0.2 mm and is skipped.
7. **aniposelib is in the dlc env only**; locanmf for the rest. Run file scripts with `PYTHONPATH=.` from the repo root.
8. **`config.stroke_date` returns MMDD** (`'0817'`); use `config._to8` before date maths.
9. **Paw-in-face frames** are confidently labelled as tongue by both models; the gates catch the far ones only.
10. **Post-stroke set point ~0.5 mm closer** (slack jaw): set-point-relative absolute values shift by that much post-stroke.
11. **PS93 is on trial clips** (only ~10 % of licks pair with the cam4 production windows): treat its numbers as weak until
    PS93 production lands.
12. **Push race with the other Claude window**: a push can fail silently between rebase and push; check `git status -sb`
    shows `## main...origin/main` with nothing ahead, re-fetch / rebase / push if not. Docs-only pushes skip tests.
13. `scripts/apparatus_mouth.py` still uses the 09-11 EXTRINSICS (method "calib"); the 3-D pipeline does not.
