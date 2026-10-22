# FaceRhythm on O2 — handoff (2026-10-21, updated ~22:00)

Plan: [`docs/FACE_RHYTHM_PLAN.md`](../FACE_RHYTHM_PLAN.md) (steps 1-7; status section at its end). This doc covers step 4
(master ROI, drawing tool), step 5 (the O2 bundle) and the **tracking-parameter test** that now gates the full run.
Registration (step 3) and the ROI warp are the pose/3-D window's (`scripts/face_rhythm_register.py`,
`scripts/face_rhythm_warp_rois.py`, `STATUS_2026-10-21_POSE_CAM1_JAW_HANDOFF.md`).

## CURRENT STATE (10-21, ~22:00) -- read this first

**HOW everything runs: [`docs/FACE_RHYTHM_PIPELINE.md`](../FACE_RHYTHM_PIPELINE.md)** (stages, files, run order, conventions,
tests). WHY: `docs/FACE_RHYTHM_PLAN.md` + DECISIONS.md 10-21 evening. This section = status + what to do next.

**1. O2 production: array 57168778** (4 workers x 4 threads, 72 GB, 9 h, `%24`). At ~21:00: 3 tasks done (PS92_0806,
PS92_0812 v1 / v2), 1-24 running at 29-34 fps / worker (~82 min per round of 4 segments: 9-12 segments ~4.1 h, 13-16
~5.5 h), violations 0.000 everywhere, worker peaks 6-11.6 GB, finished tasks MaxRSS 30-32 GB (sacct `.batch` line).
Expect the whole array ~20 h (fair share may slow the tail). Tasks 21-23 resumed segments from the cancelled 8 x 2 array.
Check / resubmit: "PRODUCTION RUNNING" below. Status commands: `sstat` needs the RAW job id
(`sacct -j 57168778 -X -n -s RUNNING -o JobIDRaw` -> `sstat -j <raw>.batch`); finished: `sacct ... | grep batch`.
**Faster defaults for FUTURE runs are committed** (8 workers x 4 threads, 5-min segments, 12 GB / worker + 24,
`--max-running`): ~2 h per session instead of ~4; the running array keeps its own params.json.

**2. Downstream code -- ready, tested on synthetic data AND dry-run on real outputs (PS92_0806 / 0812):**
`face_rhythm_factors.py` (plan done for `master_pre_v1`, regenerated after every change; build / fit / select /
project / fitsession / report), `face_rhythm_readouts.py` (+ `wfield_local/face_rhythm_readouts.py`), loader
`wfield_local/face_rhythm_outputs.py`, encoding `--fr` / `--fr-points`, `o2_face_rhythm.py qc`, factor clips
`face_rhythm_factor_clips.py`. ~17 FR tests. Things learned from REAL data tonight (all in the code + pipeline doc):
* **FR normalises every frame** by the whole-face mean power (VQT normalization_factor 0.95) -> factor weights are
  SHARES. **Absolute expression = weights x D** (per-bin normaliser recovered EXACTLY from the saved float16 spectrogram,
  0.01 % median error; `trials_D` / `session_D`; readout `abs_*` next to `amp_*`; loader `absolute=True`). Lead with
  absolute; the share version produced two artefacts in the preview (a "pre-cue" peak, dips during licking).
* Per-point movement: absolute SPEED from the 25 Hz points (default) and the spectral SHARE.
* Tracking QC: folds are ~all in ~10 RIM triangles (headplate / mask edge); INTERIOR folds 0.03 % of frames.
* **Grooming**: the session-wide top of several factors (0, 4, 5) is grooming (Priya's clip calls). The `large_movement`
  flag (whole-face movement above the licking range, outside response windows) catches big bouts but also vigorous
  spontaneous licks and misses small grooming; LP licks (LP reports licks during grooming) and regional speed do not
  separate them; no pose model tracks paws. **Decision: no grooming label for now** (Priya: paw motion not worth it now;
  paw-region motion energy on cam4 is the route if needed). Handled by: clips and trial readouts in engaged-trial windows
  (clips default `--window trial` = cue -2..+4 s), between-trial amplitudes = MEDIANS with large movement excluded,
  large movement reported separately (`large_frac`).
* Asymmetry midline = the master's LP nose x 342 (cam4 ipsilesional = IMAGE RIGHT).

**3. Rank-10 PREVIEW (scratch, PS92 only, 3 pre sessions; not the master set)** -- EVR 0.83 (uncentred), starts agree
0.97, leave-one-session-out 0.82-0.84. Components (absolute gallery `preview_r10/report/factor_gallery_abs.png`):
#0 left lick (image-right lower face, 5.8 Hz; also centre trials -- PS92's centre licks lean LEFT, +7-11 deg, measured
from LP), #4 right lick (image-left, 6.5 Hz; far_R > close_R > centre > left ~0), #9 early fast lick (7.3 Hz, both
sides, ~0.7 s), #2 pad lick (5.8 Hz), #7 late slow lick (5.1 Hz, 2-3 s) -- licking splits by RHYTHM: LP lick rate
7.1 -> 6.6 -> 6.25 -> 6.1 Hz over the trial (so post-stroke slowing would shift expression between components; read
early vs late windows + LP lick rate, not "lick factor down"); #5 cue-evoked 2.3 Hz nose / centre face; #8 broad
10-17 Hz on pads + above the pads (NOT the rim), coupled to #5 (r 0.5) -> probably fast whisking (check: f8 clip,
whisker-pad motion energy); #3 1 Hz whole face, peak 0.7 s after cue; #1, #6 slow lower edge / nose tip; whisking at
~8 Hz did not separate at rank 10 in one animal. Clips: `preview_r10/clips/PS92_20260806_cam4_f<k>_trial.mp4`
(all 10, rendering at handoff time), `_response` (0, 4), `_large` (0), older `_f<k>.mp4` = events from anywhere
(include grooming). Scratch root: C:/Users/SABATI~1/AppData/Local/Temp/claude/C--Users-SabatiniLab/
c1a9581a-546a-45cc-b071-de0ab459161e/scratchpad/ (preview_r10/, dryrun_set/, master_face.png).

**Provisional PREVIEW names** (Priya 10-21; `preview_r10/labels.json`): 0 lick_left, 1 lick_probe (lower central probing / exploratory licks, Priya), 2 lick_center (centre-ish licking; top clip events all close_L, whose licks are only +5-8 deg, Priya),
3 lick_onset_nose (nose orienting down at lick initiation, Priya), 4 lick_right, 5 nose_cue, 6 nose_up_prep (preparatory upward nose movement before an apparent attempt -- a lick or jaw movement only; Priya; cf. jaw_only attempts in trial_attempts.py), 7 lick_slow_center (slower centre-ish licks, 5.1 Hz, late in the trial; Priya), 8 whisk_fast (unconfirmed), 9 lick_early. The master
set gets its own names after its gallery + clips (components may merge / split with four animals).

**4. Next, in order:** (a) Priya reviews the 10 trial-window clips -> provisional names; (b) when the array is done:
rsync out/ + factors/, `collect`, `qc` (post-stroke tracking as clean as pre?), then `run_build` -> `run_fit` (ranks
6-16) on O2 -> `report` + `report --absolute` -> choose rank (Priya) -> `select` -> `run_project` + `run_fitsession`
-> `face_rhythm_readouts.py` (writes labels_suggested.json; Priya confirms -> labels.json) -> clips per factor on the
master set; (c) encoding: new bundle `o2_encoding.py bundle --fr master_pre_v1 [--fr-points 10]`, `check` one session
(first real test of the fr path), then the fr run (FR vs ME vs both; does position survive FR). Open checks: factor 8 =
whisking?; whether to use absolute or share FR in the encoding model (paper: normalised).

## UPDATE 10-21 evening — parameters chosen: `midlk12` (9245e51)

* Round 3 scored (11 sets): **midlk12** = mids12 mesh (0.03 / 24, relax 0.005) + oldrun LK (50 px / 5 levels / (3,2,0.0003)),
  12 px, 1282 points: folds 0.015 / frame (fewest of all; paper 0.089), jitter 0.120, nose rest gain 0.97, nose lick
  r / gain 0.65 / 0.69, jaw R2 0.69, tongue AUC 0.938, ~19 fps per 2-thread worker. midlk12lr: rest gain 1.00 but lick
  gain 0.57; midlk12i3: no gain over 2 iterations. Decomposition unchanged (71 %; L/R 6.5 Hz lick components +0.57 /
  -0.63). Plain clips `param_test/clips/*_plain_paper-mids12-midlk12-midlk12i3-midlk12lr-oldrun.mp4`.
* Variance explained vs the 2pRAM cohort's run (`FaceRhythm/fr_run_20250520_latent_bundle`, PS48 0301 / 0309, same
  centred measure, 2-min slices): fresh rank-10 fit 73-84 % (ours 71 %); its whole-session factors on those slices
  66-76 %. Uncentred (1 - SSE/SS) theirs 81-90 %. bioRxiv blocked fetches (429): the paper's own number not checked.
* Done: PARAMS["tracking"] = midlk12 (asserted equal to the tested set); master `rois/PS95_20260812/ROIs.h5` re-seeded
  at 12 px (old file moved aside), `face_rhythm_warp_rois.py to-sessions` -> 106 sessions, all 108 tasks 1282 points;
  bundle rebuilt: 8 workers x 2 threads (`-c 16`), `--mem 56G`, `-t 7 h` (est. 2.2-4.5 h; estimates now use the O2
  rate in whole rounds of segments).
* **Next: pilot task 87 (PS95_20260812)** after moving the old paper-params `work/PS95_20260812_cam4` aside; check
  speed, MaxRSS (memory fix) and outputs; then the array.

## TL;DR — where it stood before the parameter choice

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

## Mesh QC of the warped 12 px master (10-21 evening, after the array was submitted)

`scripts/face_rhythm_mesh_qc.py` -> `<dlc>/face_rhythm/cam4/rois_qc_12px_20261021/` (`mesh_qc.csv`, `mesh_qc_<animal>.png`:
points on each session's own median face). All 107 ROI files: 1282 points, 0 outside the image; flipped Delaunay
triangles 0 (PS92 / PS94 / PS95) or 0-2 (PS93, at the mouth); local area vs master 5th-95th pct ~0.5-1.3 (PS93 / PS92
compressed at the mouth: a smaller face than PS95's). From ~09/10 every animal's mesh sits ~40 px from the master
(camera / rig shift; registration follows the face, the spout ring is now visible at the image top). Visually: post-
stroke meshes cover the same face as pre (PS93 + PS95 sheets checked). No session needed re-drawing.

## PRODUCTION RUNNING (10-21 night): array 57168778 (`--array=1-108%24`, `--mem 72G`, `-t 9 h`, 4 workers x 4 threads)

What happened first: pilot 57149639_87 + array 57154014 ran 8 workers x 2 threads at 56 GB. Speed exactly as tested (19
fps / worker, 137.5 min / segment), but at the end of round 1 the job sat at the 56 GB cgroup limit and the kernel killed
3 of the pilot's 8 workers ("terminate called without an active exception"; 12 torch.jit warnings in the .err = 1 parent +
8 workers + 3 replacements). `mp.Pool` silently replaces a killed worker and waits forever for its segment -> hang to the
time limit. Task 6 (PS92_0812 v1, 4 segments -> 4 workers x 4 threads) completed in 1.3 h at **34 fps / worker**,
MaxRSS 31.9 GB: 4 x 4 gives about the same throughput as 8 x 2 (136 vs 152 fps) at half the memory, and is faster for
<= 12 segments. Fixed (70d4c3b): ProcessPoolExecutor (a dead worker -> finished segments recorded, exit "segment(s) lost
... resubmit to resume"), workers 4 x 4 threads, sbatch memory floor 72 GB. Both old jobs cancelled; the new array resumes
the pilot's saved segment 1; task 6 = done already.
Expect ~3.8 h per task (9-12 segments), ~5.1 h (13-16), ~18-20 h for the array. Check:
`sacct -j 57168778 -X --format=JobID%16,State,Elapsed,MaxRSS | grep -v COMPLETED`, `grep -l "segment(s) lost" logs/57168778_*.out`;
resubmit failures with the same sbatch line and `--array=<list>`. Then transfer-node rsync of `out/` to the share and
`python scripts/o2_face_rhythm.py collect --name fr_cam4_20261021`. NOTE: the runner has no lock -- never run two copies of
one task (check `squeue` before any resubmit).

## Factorization code ready (10-21 night): `scripts/face_rhythm_factors.py` (7755d4b)

Master pre-stroke factor set across animals (DECISIONS 10-21 evening). `plan` DONE for set `master_pre_v1`
(`<bundle>/factors/master_pre_v1/`: plan.json, fr_factors.py, run_build / run_fit / run_project.sbatch): per animal 60
engaged pre-stroke trials (10 per position, drawn across its 7-9 August pre sessions, seed 0) + 1125 between-trial 5 Hz
bins; ~22 500 rows, fit tensor ~8.3 GB float32; 108 tasks to project. Synthetic end-to-end test (planted rank 3, four
animals at different scales): factors recovered at 0.999, leave-one-animal-out + projection + refreeze guard work.
WHEN THE ARRAY IS DONE (O2, after rsyncing the factors/ folder from the share):
`sbatch <b>/factors/master_pre_v1/run_build.sbatch` -> `sbatch --dependency=afterok:<id> .../run_fit.sbatch` (ranks
6-16, 5 inits + 4 held-out-animal fits each, ~3 h per rank est.) -> copy rank*.json/npz back -> pick rank -> desktop
`python scripts/face_rhythm_factors.py select --rank R` (freezes master.npz; needs scales.json + the rank files on the
share) -> O2 `run_project.sbatch` -> proj/<key>.npz (trial + session time courses, per-bin reconstruction error).
Still to write: `report` (factor gallery, rank sweep, residual by epoch), the loader `wfield_local/face_rhythm_outputs.py`,
the encoding-model `fr` family (spec for the encoding window).

## FR -> encoding model: code ready (10-21 night, 01aff84; Priya: "go ahead with the encoding work")

* Loader `wfield_local/face_rhythm_outputs.py`: proj/<key>.npz (split videos joined) -> one series per factor (25 Hz
  trial windows, 5 Hz elsewhere) -> imaging frames by linear interpolation (NaN across > 0.5 s gaps), blanked outside
  strobe .. trial end like video ME; refuses projections from two different frozen sets. Tests
  `tests/test_face_rhythm_outputs.py`. Component names from `<set>/labels.json` if written ({"0": "sniff", ...}).
* `combined_encoding.py`: FR family = VIDEO's placement (FR_ENV `COMBINED_ENCODING_FR` / `--fr <set>`): fr:full (orth,
  added last), fr:alone, fr:raw:full, fr:drop:<task|instructed|uninstructed|position>; with `--video` too:
  frvideo:raw:full, frvideo:alone -> FR beyond ME = frvideo:raw:full - video:raw:full; ME beyond FR = frvideo:raw:full -
  fr:raw:full. Own penalty group "fr". Folder `<order>[_video-..]_fr-<set>/`. The target model Priya named (task + behaviour
  events + LP + treadmill + FR, no ME) = fr:raw:full.
* `o2_encoding.py`: `bundle --fr <set>` (needs a NEW bundle name: existing inputs are not rebuilt), sbatch export FR=,
  COMMANDS.md adds the run `fr_<set>` (VIDEO=cam1,cam4 + FR together). `check` takes --fr: run it on one session first --
  it compares bundled vs share inputs and is the first end-to-end test of the fr path on real data (NOT yet run: no
  projections exist).
* `face_rhythm_factors.py report [--face <img>]`: rank sweep, factor gallery (spatial on the face, spectrum, pre-stroke
  trial averages by position), reconstruction error by epoch (+ CSV); tested on the synthetic set.

## Readouts + docs (10-21 late night)

* `face_rhythm_factors.py` restructured (shared `draw_rows` / `gather_rows`; `project` now also writes per-point movement
  power by window; NEW `fitsession` = each session's own rank-R fit vs the frozen basis on the same rows + matched
  cosines). `plan` for `master_pre_v1` regenerated (adds per-task rows; nothing had been built from the old plan).
* `wfield_local/face_rhythm_readouts.py` + `scripts/face_rhythm_readouts.py`: the 2pRAM manuscript's Fig 4 analyses
  across animals -- amplitudes by window / epoch, reconstruction error, EVR gap + morphing, movement maps + contra / ipsi
  asymmetry (cam4 ipsi = image right) -- statistics via `orofacial_cohort.epoch_tables(metrics=...)` (now takes a
  metric list; default unchanged).
* **`docs/FACE_RHYTHM_PIPELINE.md`** = the reference for stages / files / run order / conventions / tests.
  `FACERHYTHM_PROPOSAL.md` marked historical. Tests: 8 FR tests (synthetic), all pass.
