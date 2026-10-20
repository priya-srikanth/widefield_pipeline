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
3. ~~Jaw 200 ms re-run, jaw cross-check, depth-prior check~~ DONE late 10-19 (results §4).
4. ~~cam1 lick review~~ DONE, 2 rounds (pages https://claude.ai/artifact/EnDtPi95nsfyyT4eGQDbCV and
   https://claude.ai/artifact/8ayHFgP9hshafCna2d2Aq5; scored: `session_poses/cam1_licks/review{,2}/scored.csv`). Every false
   cam1-only event was a WATER DROPLET -> droplet flag adopted (§2, §4). New O2 sessions: re-run `cam1_licks run` after 3-D.
5. When all four animals are in: cohort tables (per-animal first, then `analysis_kit.boot_delta`), §7 item 6.

## 1. What was done (10-19)

* **cam1 model + production**: LP ctx-semi chosen (most faint licks after cleaning, no emergence lag); O2 production bundle
  built and running (`wfield_local/o2_inference.py --cam cam1`; collect writes `session_poses/<a>_<d>_cam1_full/`).
* **3-D pipeline** (`wfield_local/pose3d.py`, `scripts/pose3d.py`): per session, cut both cameras' O2 output into trial
  windows on the DAQ clock (`o2_inference.session_windows`) → June cam4 x0.8853 about the principal point → spout medians per
  position → **PnP camera poses on the commanded spouts** (09-11 intrinsics only; spout edge offsets) → clean each camera with
  its own rules (MEASURED points only) → pair frames exactly on (trial, t) → triangulate (aniposelib, undistort + DLT) →
  **fill one-camera frames** on the seen camera's ray (depth from both-camera neighbours <= 24 ms tongue / 200 ms jaw, Akima) → **plausibility
  gates** → `session_poses/pose3d/<a>_<d>_frames.csv.gz` (per frame: tongue / jaw / nose / spout ml, ap, dv mm; reprojection;
  source 1 both / 2 cam1 / 3 cam4).
* **Readouts** on top: `pose3d_kinematics` (cam4 lick events or `--licks combined`: 3-D peak, direction vs target, reach,
  speed, jaw opening + FAR_DIR), `pose3d_spout_error` (closest approach, endpoint vs spout, endpoint vs own pre endpoint),
  `pose3d_crosscheck` (cam4 2-D vs 3-D, lick by lick), `pose3d_jaw_crosscheck` (written, not run), `pose3d_vs_cam4` (does
  3-D add information: decoding + effect sizes; running), figures: `pose3d_figures` (maps; 2 x 2 paired video with tongue +
  jaw), `pose3d_summary_fig` (per session over days), `pose3d_phase` (peak-aligned distance, phase portrait, aim vs phase,
  trajectories in 3 planes, **oblique 3-D tongue + jaw trajectories per spout** `trajectories3d_<a>.png`, speed / velocity
  traces). Checks: `pose3d_jaw_gap_check`, `pose3d_depth_prior_check`, `pose3d_jaw_crosscheck` (all run, §4).
* **cam1 2-D lick detector** (late 10-19, Priya: count the incomplete licks cam4 / 3-D miss): `wfield_local/cam1_licks.py`,
  `scripts/cam1_licks.py run|review`. Events = runs of cleaned cam1 tongue frames (cue .. stop, >= 3 frames); size in cam1 px
  / mm (scale from the far_L / far_R spout spacing; a size scale only -- reach toward the camera is foreshortened);
  ``size_class`` lips (peak never > 20 px below the cam1 mouth) / out; flags ``cam4_matched``, ``in_3d``, ``contact`` (DAQ),
  ``jaw_only_trial`` (inside a cam4 "jaw movement without a lick" trial's 70 ms .. quiet-window span). Output
  `session_poses/cam1_licks/` (`<s>_events.csv`, `sessions.csv`, `run_summary.csv`, `review/`).
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
| one-camera fill: Akima, tongue <= 24 ms, ray-miss gate 3 mm | held-out error 0.13-0.23 mm; Akima beats linear / PCHIP on long gaps; a 1 mm gate dropped half the real fills |
| jaw fill limit 200 ms | jaw moves slowly: held-out error at 200 ms median 0.04-0.07 mm (p90 0.20-0.34); 24 ms covered only 1-3 % of one-camera jaw frames, 200 ms 13-23 % |
| gates: > 8 mm out, > 2 mm behind the mouth, spikes > 1.5 mm; jaw > 5 mm | paw-in-face frames (both models call them tongue at 0.9+) triangulate 7-36 mm out; reprojection does not catch them |
| readouts: distances primary (3-D mm) + 3-D angle error + midline-plane angle | distances are origin-free; angles vs the spout direction are consistent across the set-point move (spouts commanded from it) |
| peak = 3-D frame farthest from the mouth zero | cam4's peak is the max of its DOWN axis |
| lick events: cam4's (comparable) + combined 3-D track | combined adds 3-7 % licks -- NOT small ones (median reach 3.7-4.1 vs 4.5 mm, 45-50 % touched the spout): mostly licks cam4 lost to occlusion / merging |
| no depth prior for one-camera tongue points | held-out: (a) closest point to the mouth zero 0.6 mm error on licks < 1 mm, 1.3-3.8 mm beyond; (b) learned from cam1 2-D 1.5-1.7 mm on licks < 1 mm (0.5 mm on long ones, already covered by the neighbour fill) |
| incomplete licks counted from cam1 2-D (`cam1_licks`), not 3-D | counting / sizing needs no depth; tongue tips seen only by cam1 cannot be placed in 3-D (row above) |
| cam1-only events flagged as WATER DROPLETS and dropped: spread < 3 px, range < 12 px, median point away from the mouth opening (out > -35 px) | Priya's 2-round blind review: all false events were droplets; the rule catches 16 / 17, loses 0 / 70 real. Static points at the mouth opening are real tongue tips at the lips; static points cam4 also calls a lick are the tongue held on the spout, so only cam1-ONLY events are flagged |
| recent misses kept as a covariate (`miss_prev3`), not a gate | droplets follow uncollected water (Priya): pre 3 % of trials after a hit, 12 % after a miss, 15 % after 3; acute flat ~1 %; the shape rule alone already works |
| cameras PAIRED ON CAMERA-TIMESTAMP TIME (`t_daq_ms`, nearest within 2 ms), not frame-index `t_ms` (10-20) | Priya: "align using the sync pulse"; frame-index time slides with every dropped frame -- PS92 0908 (frame-drop cliff at min ~104) paired wrong partners (tongue reprojection 8.5 px, 36k frames); now 3.5 px, 153k frames, all 4 session quarters normal. Also fixed PS92 0606, PS93 0605 / 0606 (6.4-9.0 -> 3.1-3.7 px) -- the "June outliers" were drops. Drop-free sessions unchanged (0807: 231615 -> 231601 frames). Output `t_ms` stays cam4's (v36 lick windows) + `t_daq_ms` |
| session gates `pose3d.failed_sessions`: spout fit > 0.2 mm, median tongue reprojection > 6 px | catches timing / pose failures automatically (normal 2.7-4.0 px); now only PS92 0605 (spout fit) fails |
| ALL video timing from the alignment templates' per-frame camera times (10-20, Priya: "all timings should use the alignment templates") -- `o2_inference.build_windows` -> `trial_frame_spans_ts`: cue frame = frame nearest the cue in template time, windows by time, `t_ms` = template time from the cue frame snapped to 4 ms (drops leave gaps) | the template's frame-index affine drifted 4 ms per dropped frame: `scripts/cam4_timing_check.py` over 134 cam4 sessions -- 125 within one frame; 9 off: PS92 0605 / 0606 / 0908, PS93 0605 / 0606, PS94 0603 / 0605 / 0805, PS95 0605 (cue 26 ms - 1.3 s; PS94 0805 = no drops but frames ~1.7 % slow). Clean session (PS92 0807): 100 % same cue frames; PS93 0605: cam1 vs cam4 now within 4 ms (was up to 0.6 s -- Priya saw it in the round-3 review). v36 re-run (`--force`) for the 9 |
| jaw: long one-camera stretches placed with a REST-DEPTH ANCHOR (`pose3d.anchor_single_view`, src 4 cam1 / 5 cam4) | centre spouts: the rod is end-on in cam1's view (cam1 keeps ~46-49 % of jaw frames there), close_center also hides the chin in cam4; held-out on side spouts: ml error 0.03-0.05 mm, per-trial lr_wide r 0.95-0.99, FAR_DIR within 0.02 mm; ap from cam4 alone is 0.15-0.39 mm off -> do not use anchored frames for the 3-D opening's backward part. far_center jaw coverage 62 -> ~100 % |
| lick phase: onset -> offset per lick (primary), peak-to-peak secondary | same as cam4's per-lick normalised time (`orofacial_cohort.U_NORM`); keeps isolated licks (acute far contra: 38-56 % have a neighbour < 250 ms); peak-to-peak only for stroke_orofacial comparison |

## 3. Running / pending at hand-off

| what | how it was started | output | if lost, re-run |
|---|---|---|---|
| O2 array **56830841** (cam1 production) | Priya, 10-19 ~07:40; pending tasks opened to any GPU, 16 h (`scontrol update ... Gres=gpu:1 TimeLimit=16:00:00`); V100 nodes excluded | scratch `o2/lp_cam1_ctx_prod_20261019/out/` | §5 |
| ~~LOCAL chain incl. jaw 200 ms re-run, jaw cross-check, `pose3d_vs_cam4`, depth-prior check~~ DONE (§4) | — | `session_poses/pose3d/` | §6 |
| ~~cam1 lick detector on all 19 PS92 sessions + review sample~~ DONE | — | `session_poses/cam1_licks/` | `python -m scripts.cam1_licks run` then `review` (locanmf) |
| Priya's blind review of cam1 lick candidates | page in §0 item 4 | db collection `cam1_lick_review` | — |
| ~~LOCAL jaw gap test + jaw / tongue videos~~ DONE: result in §4; videos `pose3d/figures/lick_paired_PS92_2026{0807,0818}_cam4view.mp4` (tongue: green 2-D, red / orange 3-D; jaw: white 2-D, blue / cyan 3-D) | — | — | `python -m scripts.pose3d_jaw_gap_check`; `python -m scripts.pose3d_figures video ... --view cam4` (dlc env) |

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
* **Jaw one-camera fill (gap check, PS92 0807 / 0818)**: held-out error median 0.03-0.05 mm at 24 ms, 0.04-0.07 mm at 200 ms
  (p90 0.10 -> 0.20-0.34); one-camera jaw frames are mostly LONG stretches (only 13-23 % in gaps <= 200 ms; e.g. the rod
  over the chin in cam4 for whole centre-spout trials) -> 200 ms adopted; the long stretches stay single-camera (an anchor to
  the trial's resting jaw would be the next idea, untested).
* **Coverage vs cam4 alone (0807 / 0818, response window)**: 3-D has 8-9 % more MEASURED tongue frames than cam4 (12-16 % of
  3-D frames are ones cam4 did not measure); cam4's own 2-D gap fill counts more frames (it interpolates up to 56 ms).
* **cam1 vs cam4 axes**: cam4 alone = ml + dv (no forward); cam1 alone (looking up ~45 deg) = ml + a mix of forward and down
  that it cannot separate; only both cameras separate all three.
* **Does 3-D add information? (`pose3d_vs_cam4`, PS92, 9 August pre, leave-one-session-out balanced accuracy)**: target spout
  from the lick endpoint, 6 positions 0.53 cam4 (0.52 + angle) vs **0.80 3-D**; close vs far L 0.83 vs 0.94, centre 0.59 vs
  **0.87**, R 0.89 vs 0.96 (depth is what cam4 lacks). Acute effect sizes (|z| vs pre day-to-day spread): lateral / angle agree
  in pattern (far_R, close_L), 3-D larger at far_R (~30 vs ~21); 3-D-only miss and forward reach also move (far_R 30.8 / 12.6).
* **Jaw cross-check (cam4 v36 `jaw_lateral` vs 3-D, 14 sessions, per trial)**: lateral r 0.66-0.88, opening r 0.63-0.92
  (cam4 opening tracks 3-D DOWN r 0.58-0.90 more than BACK 0.22-0.74); **session FAR_DIR r = 0.99** (pre 9-15 px / 0.16-0.29
  mm -> acute -0.7..0.9 px / -0.06..0.02 mm; PS93 0.20 -> -0.03 -> 0.02 mm). ~33-38 px/mm lateral. June / 0805 are not in the
  v36 jaw table (none matched).
* **Depth prior (held-out, cam4 hidden; PS92 8 pre LOSO + 3 acute)**: error by distance from the mouth zero, (a) closest point
  on the ray / (b) KNN on cam1 2-D: 0-1 mm 0.60-0.69 / 1.49-1.74; 1-2 mm 1.33-1.38 / 0.83-0.99; 2-3 mm 2.1-2.2 / 0.55; > 4 mm
  3.8 / 0.6-0.7. Neither usable for small licks (the neighbour fill is 0.13-0.23 mm) -> not adopted.
* **cam1 2-D licks (PS92, 16 pre incl. June / 3 acute; REVIEWED, droplets removed)**: blind review 2 rounds (87 events):
  pre cam1-only 23 / 24 real; acute cam1-only away from jaw-only trials 10 / 10 real (8 at the lips); cam1 events on acute
  jaw-only trials 0 / 5 real, pre 5 / 10; controls 10 / 10 contact real, 9 / 9 null none; every false one a water droplet.
  After the droplet flag: events 63.1k / 9.0k; cam1-only 7.5 / 3.9 % (at the lips 3.3 / 3.3 %); cam1-only without contact
  4.6 / 3.4 %. **cam4 jaw-only trials with a cam1 tongue: pre 45 / 241 (19 %), acute 9 / 259 (<= 3.5 %, round-2 acute
  0 / 5 real -> an upper bound)** -> pre "jaw without lick" was partly a tongue cam4 missed; acute ones are tongue-absent, so
  the acute jaw-tongue dissociation is slightly STRONGER after correction. Droplets: 1.7 % of all cam1 events, most at
  close_center (pre 14 % of trials), after misses (pre 3 % after a hit -> 15 % after 3 misses). Bigger review crops mattered:
  4 / 12 round-1 'none / unsure' were real at 1.6x size with the detected point shown.
* **No-contact licks from the combined track**: +12 % in both epochs (pre 12954 -> 14510, acute 1036 -> 1157).
* **10-20: 44 sessions (PS92 all 35 to 10-06; PS93 June + August pre).** PS93 repeats PS92 on 2-D vs 3-D (target spout 0.42
  cam4 vs 0.68 3-D; acute far_R lateral |z| 27 vs 12). Jaw cross-check CHRONIC (PS92 from 0828): opening r drops because
  every trial opens fully (3-D SD 0.3-0.4 mm vs 0.5-1.5) -- range restriction; LATERAL per-trial r drops to 0.1-0.6 with
  unchanged variance in both measures (FAR_DIR still agrees, r 0.95 / 34 sessions) -> chronic trial-level jaw lateral is
  UNCONFIRMED; suspect head posture in cam4's image-px measure (untested). cam1 licks: PS92 chronic cam1-only 11.7 % (lips
  7 %) vs pre 5.7 %; PS93 pre 44 % no-contact / 27 % lips (check videos). Frame-index vs camera time on a NORMAL session
  (0807): median 1.6 ms, p99 28, max 46 ms -> cue-relative v36 timings carry per-session drop error (to quantify).
  QUANTIFIED + FIXED 10-20 (decision row "ALL video timing from the alignment templates"). Chronic jaw lateral RESOLVED:
  on far_L / far_R (the trials FAR_DIR uses) cam4 vs 3-D per-trial r is 0.80 chronic vs 0.90 pre; the drop is the
  close / centre spouts cam4's jaw readout already excludes. PS93 low contact (62-69 % Aug) is real (cam4 v36 70 %).

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
python -m scripts.cam1_licks run --jobs 3                                                        # locanmf; after pose3d (in_3d flag)
```
Outputs under `M:/MICROSCOPE/Priya/DeepLabCut/Widefield/session_poses/pose3d/` (frames, `summary_*.csv`, `kinematics/`,
`kinematics_combined/`, `spout_error/`, `crosscheck/`, `vs_cam4/`, `phase/`, `figures/`). Comparison snapshots kept:
`nofill_20261019/`, `noedge_20261019/`. The duplicate `kinematics_setpoint/` can be deleted by Priya.

## 7. To-do

1. ~~Jaw fill limit re-run~~ done. Optional: anchor long single-camera jaw stretches to the trial's resting jaw.
2. ~~Jaw cross-check~~ done (§4).
3. ~~Depth prior~~ done: not adopted (§2).
4. **Incomplete-lick and jaw-without-lick readouts with cam1**: per position x epoch, cam4 no-contact + cam1-only (droplets
   removed) and the corrected jaw-only rate; add to the cohort tables when the other animals land. ``lips`` size class means
   "at the lips" only for cam1-ONLY events (cam4-matched "lips" events were all full sideways licks).
5. **3-D phase plots to match cam4**: show onset->offset phase 0.1-0.9 (`orofacial_cohort.U_SHOW`) and add a FIRST post-cue
   lick set beside "all licks". 2-D vs 3-D: 3-D is headline for depth / reach / miss and close-vs-far; cam4 lateral stands.
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
14. **Never name a script `*_test.py` / `test_*.py` outside `tests/`**: pytest collects it, imports it, and a script that does
    work at import breaks collection -- the pre-push hook then blocks the push (10-19: the depth-prior / jaw-gap scripts).
