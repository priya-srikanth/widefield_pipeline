# FaceRhythm on O2 — handoff (2026-10-21, updated late afternoon)

Plan: [`docs/FACE_RHYTHM_PLAN.md`](../FACE_RHYTHM_PLAN.md) (steps 1-7; status section at its end). This doc covers step 4
(master ROI, drawing tool), step 5 (the O2 bundle) and the **tracking-parameter test** that now gates the full run.
Registration (step 3) and the ROI warp are the pose/3-D window's (`scripts/face_rhythm_register.py`,
`scripts/face_rhythm_warp_rois.py`, `STATUS_2026-10-21_POSE_CAM1_JAW_HANDOFF.md`).

## TL;DR — where it stands

* **Parameters: recommend `mids12`** (mesh rigidity 0.03 / 24 neighbours, relaxation 0.005, paper's LK 40 px / 3 levels /
  (3,3,0.03), threshold 300, **point spacing 12 px**). Priya has not decided yet: **round 3 is queued on O2** (array
  57142222_[1-3]: `midlk12`, `midlk12lr`, `midlk12i3` = mids12's mesh with oldrun's optical flow 50 px / 5 levels;
  relaxation 0.005 / 0.0025; LK criteria (3,2,0.0003) / (3,3,0.03)). Score it, then decide.
* **Production is NOT running.** The one full-session pilot (task 87, PS95_20260812, paper params) was killed
  OUT_OF_MEMORY (fixed since, see pitfalls) and is obsolete (paper params). Its partial `work/PS95_20260812_cam4/`
  (~30 GB) is still on scratch: Priya's call to remove; a new task 87 refuses to start until it is moved aside.
* After the parameter decision: set PARAMS["tracking"] (+ spacing) in `o2_face_rhythm.py`, **re-seed the master ROI at
  the chosen spacing** (same polygons, face AND mask) and **re-warp** to sessions (pose window), rebuild the bundle,
  rsync, **pilot one session** (checks speed + the memory fix), then the array.

## Where things are

| what | where |
|---|---|
| bundle / collect / param test | `scripts/o2_face_rhythm.py` (`bundle`, `collect`, `paramtest`, `paramscore`) |
| QC clips + factor decomposition | `scripts/face_rhythm_param_qc.py` (`clips [--color plain] [--sets ...]`, `decompose [--sets ...]`; fr env) |
| ROI drawing notebook | `notebooks/face_rhythm_draw_roi.ipynb` (fr env, **JupyterLab** -- VS Code's widget renderer fails) |
| bundle on the share | `M:\MICROSCOPE\Priya\DeepLabCut\Widefield\o2\fr_cam4_20261021\` (108 tasks; June out; PS92_0812 = 2 tasks) |
| bundle on O2 | `/n/scratch/users/p/ps150/o2/fr_cam4_20261021/` (rsync from the share before submitting) |
| master face / ROI | **PS95_20260812** (August pre-stroke; `face_rhythm/cam4/register/templates.json`); `rois/PS95_20260812/ROIs.h5` = Priya's final polygons, 2287 points at 9 px; earlier versions kept as `ROIs_<time>.h5` |
| param test outputs | `<bundle>/out_test/PS95_20260812_cam4_<set>.npz` (full-rate points, frames 31199-61199, 13 trials) |
| param test scores / figures / clips | `<dlc>/face_rhythm/cam4/param_test/` (`param_test_fr_cam4_20261021.{csv,png}`, `decomp_*`, `clips/`) |
| O2 env | `fr` in `/home/ps150/.conda/envs/fr`: face-rhythm 0.3.3, torch 2.14.1+cpu |

## The runner (one task = session x cam4 video)

* FR's own per-frame tracking step on frame ranges, **10-min segments** restarting at home (10 s warm-up, 520-frame VQT
  margin), **4 segment workers x 2 threads** (spawn pool), shared float16 memmaps in `work/<key>/`, `progress.json`
  resume. Parallel output bit-identical to single-process.
* **Integer frame formatting** (87c88bf): `(r+g+b)//3 * mask` -> CLAHE in numpy = FR's float32 mean exactly; identical
  tracked points, 26.9 -> 33.3 fps (+24 %). Checked against FR's on the first frames at setup (falls back if different).
* **Keyframe notches** 20.83 / 41.67 / 62.5 Hz (Q 30, zero-phase) on the 250 Hz points before the VQT and the 25 Hz
  points (see pitfalls).
* **Memory fix** (5c71000): flush + `MADV_DONTNEED` the work memmaps after every 10 000-frame VQT chunk.
* VQT Fs 250, 1-60 Hz, 36 bins, Q 4 -> 20, 25 Hz; outputs (float16): `trials` (trial, time, xy, point, freq) on the DAQ
  grid cue -2 .. +4 s from v36 `windows.npz`, engaged trials only (`orofacial_cohort.engagement_keep`, 90 % kept);
  `session` 5 Hz; `points` 25 Hz disp + `viol_frac`; summary json. ~24-32 GB / session at 2287 points, about half at 12 px.
* sbatch in the bundle: `short`, `-c 8`, `--mem 36G`, `-t 6 h`, `%24` -- **re-derive after the pilot** (speeds below).

## Parameter test (PS95_20260812, 30 000 frames, 13 trials; `paramtest` / `paramscore`)

Measures: folds = Delaunay triangles of the home points flipping orientation (mesh crossing over); jitter = 30-60 Hz SD;
nose vs LP 2-20 Hz (r, gain; split lick bout / rest); **licking explained by the whole mesh** = cross-validated jaw-y R2
(50 velocity PCs, ridge, 5 blocks) and tongue-visible AUC (no FR point sits on the jaw: the mask stops above it).
Decision rule agreed: no / few folds, jaw R2 and nose gain near the best, jitter <= paper, then cheapest.

| set | rig / nb, relax, LK | pts | fps | folds/frame (frames) | jitter | nose rest gain | nose lick r / gain | jaw R2 | tongue AUC |
|---|---|---|---|---|---|---|---|---|---|
| paper | 0.1/24, 0.01, 40/3 | 2287 | 18 | 0.09 (0.7 %) | 0.134 | 0.90 | 0.63 / 0.76 | 0.62 | 0.92 |
| mid | 0.03/24, 0.005, 40/3 | 2287 | 18 | 0.59 (3 %) | 0.134 | 0.92 | 0.58 / 0.63 | 0.66 | 0.92 |
| **mids12** | mid, 12 px | 1282 | 19* | 0.21 (2 %) | 0.134 | 0.94 | 0.59 / 0.66 | 0.66 | 0.92 |
| soft | 0.01/24, 0.003, 40/3 | 2287 | 18 | 1.83 (15 %) | 0.134 | 0.95 | 0.54 / 0.52 | 0.66 | 0.92 |
| softs12 | soft, 12 px | 1282 | 25 | 0.83 (12 %) | 0.134 | 0.97 | 0.55 / 0.54 | 0.67 | 0.91 |
| oldmesh | 0.002/90, 0.002, 40/3 | 2287 | 18 | 5.27 (40 %) | 0.135 | 0.96 | 0.42 / 0.41 | 0.60 | 0.93 |
| oldrun | 0.002/90, 0.002, 50/5 | 2287 | 13 | 2.34 (18 %) | 0.117 | 1.00 | 0.73 / 0.61 | 0.69 | 0.93 |
| frdefault | 0.025/8, 0.0015, 20/2 | 2287 | 30 | 9.56 (46 %) | 0.161 | 0.93 | 0.76 / 0.56 | 0.57 | 0.90 |

\*shared a node; ~25 fps alone, ~31 with the integer formatting. fps = O2, 2 CPUs, before the formatting speed-up.

Read-out: frdefault folds (Priya saw it buckle) -> out. Paper is intact but too stiff for small / local motion (Priya:
nose rows stay rectangular; rest gain 0.90) though its stiffness *helps* large fast licks (neighbours outvote bad LK on
the low-texture nose). Oldrun's fidelity comes from its **optical flow** (50 px, 5 levels), not its mesh (oldmesh folds
and is worst in licks) -> round 3 = mids12 mesh + oldrun LK. 12 px costs nothing (mids12 = mid on every measure).
**Factors are the same for every set** (rank-10 NN-CP, 66-71 % variance): 4 Hz nose (sniff), 1-2 Hz slow nose/face,
8.2 Hz pad whisking, 5.8 Hz shared licking, **two 6.5 Hz lick-direction components** (far_L vs far_R trials, selectivity
+0.61 / -0.73, identical for paper / mids12 / softs12), 15 Hz mask-rim noise. #4 (8.2 Hz) rises over the window --
position vs time not separable in 2 min.

**After round 3 lands** (copy back `out_test/`): `python scripts/o2_face_rhythm.py paramscore --name fr_cam4_20261021`
(locanmf env; ~15 min; new figure = one row per set: nose lick / nose rest / jaw from the whole mesh) ->
`fr env: python scripts/face_rhythm_param_qc.py clips --color plain --sets paper mids12 midlk12 oldrun` and
`decompose --sets <winner>`; check L/R lick selectivity as above.

## O2 commands (pattern)

```bash
# transfer node: share -> scratch, and results back
rsync -ah /n/files/Neurobio/MICROSCOPE/Priya/DeepLabCut/Widefield/o2/fr_cam4_20261021/ /n/scratch/users/p/ps150/o2/fr_cam4_20261021/
rsync -ah /n/files/Neurobio/MICROSCOPE/Priya/DeepLabCut/Widefield/face_rhythm/cam4/rois/ /n/scratch/users/p/ps150/o2/fr_cam4_20261021/rois/
rsync -ah /n/scratch/users/p/ps150/o2/fr_cam4_20261021/out_test/ /n/files/Neurobio/MICROSCOPE/Priya/DeepLabCut/Widefield/o2/fr_cam4_20261021/out_test/
# login node
sbatch /n/scratch/users/p/ps150/o2/fr_cam4_20261021/run_param_test.sbatch        # or run_fr.sbatch --array=<i>
tail -n 1 /n/scratch/users/p/ps150/o2/fr_cam4_20261021/logs/pt_*.out
```
Task index of a session: `awk -F'\t' '$1=="<key>" {print NR-1}' tasks.tsv` (PS95_20260812_cam4 = 87, PS94_20260810_cam4 = 58).

## Pitfalls met today

* **OUT_OF_MEMORY on the first full session** (task 87, 57115611_87): 41.9 GB of 36 after 2 of 12 segments, worker peaks
  11.8 / 20.1 GB. Dirty pages written through the shared memmaps (trials array 29 GB) pile up on /n/scratch's slow
  write-back and count against the cgroup. Fixed (flush + MADV_DONTNEED per chunk); **unverified on O2** -> the pilot.
* **O2 speed**: paper at 2287 points = 17.5 fps per worker -> 2.5 h per 10-min segment, ~7 h per session (> 6 h limit).
  CPUs saturated (TotalCPU ~2x elapsed). Desktop profile per frame: LK 54 %, FR's torch formatting 20 %, CLAHE 20 %.
* **Slurm MaxRSS counts the video file cache** (param tests: 8.4 GB "MaxRSS" vs 2.5 GB Python peak).
* **Clip colouring by |displacement from home| misleads**: loose meshes sit offset (saturated colour) while a tight one
  flickers near 0 -> looks like jitter / motion. At rest all sets have the same 2-20 Hz movement (~0.6 px). Use
  `--color plain` to judge motion by position.
* **cam4 encoder keyframe line**: FMP4 (FFmpeg MPEG-4, GOP 12, ~13 Mbit/s); keyframe = video frame index % 12 == 0.
  Points carry it as a sawtooth over the group (positions integrate the keyframe's apparent shift, then relax), so
  interpolating keyframe frames -- right for motion energy (pose window) -- does NOT clean points (20.83 / 41.67 / 62.5 Hz
  at 1.59 / 1.52 / 1.48x raw, 1.58 / 1.56 / 1.52x interpolated, 0.30 / 0.05 / 0.02x notched). Future recordings: raise
  the Bonsai writer's bitrate (recorder repo; not from here).
* **Master must be pre-stroke**: the first registration picked post-stroke templates (PS93_20260821, acute, the
  right-orofacial-deficit animal) by medoid score alone; rebuilt from August pre templates (PS92_0811, PS93_0809,
  PS94_0812, PS95_0812; master PS95_0812).
* **Eyes are in the ROI** (Priya's earlier PS46 analysis had strong weights there).
* **O2 /home quota**: CUDA torch + a 15 GB pip cache filled /home; CPU torch first, `PIP_CACHE_DIR` / `TMPDIR` on
  scratch (COMMANDS.md). `quota-v2`, not `quota`.
* **FR's ROI GUI** (Plotly freehand) fails in VS Code and can't click vertices -> the notebook uses matplotlib
  `PolygonSelector` in JupyterLab: click vertices, drag to edit, 60 px border for off-image vertices, `EDIT = True`
  reopens a saved ROI; points = face AND mask; save moves the old file aside.
* **The 08/06 sessions' trial ids** differ from the behaviour tables (dead `spout_bit1` -> GUI log, 1-based):
  `engagement_keep` matches by order. **Split sessions** (`ix_video`): one task per video.
* **Committing shared files**: a whole-file `git add CLAUDE.md` swept another window's hunk into 6aff451 -- diff first.
* A JupyterLab server for the notebook may still be running (port 8899, fr env).
