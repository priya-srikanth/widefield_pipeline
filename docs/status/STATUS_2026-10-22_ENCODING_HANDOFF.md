# Handoff — 2026-10-22: LP_licks results, frozen pre-stroke encoder, Musall-style readouts, FaceRhythm pilot

**START HERE for the encoding model (the ENCODING window, session `sabatinilab-d4`).** Supersedes
`STATUS_2026-10-21_ENCODING_HANDOFF.md` (still the reference for §0 rules, §4 pitfalls, §5 commands and the 10-21 work).
Other windows: pose / 3-D / cam1 = `sabatinilab-5b` (`STATUS_2026-10-21_POSE_CAM1_JAW_HANDOFF.md`); FaceRhythm =
`sabatinilab-e3` (`STATUS_2026-10-21_FACE_RHYTHM_O2_HANDOFF.md`). **Results: `docs/ENCODING_FINDINGS.md` §3.3** (updated
10-22). Reasoning: `DECISIONS.md` 2026-10-21 (night) and 2026-10-22 entries. `S` = scratchpad
`C:\Users\SABATI~1\AppData\Local\Temp\claude\C--Users-SabatiniLab\cdd1e016-6bb3-4346-8600-f08116ccf61d\scratchpad\`;
`E` = `M:\MICROSCOPE\Priya\Widefield\LP_analysis\lp_r4_heatmap_20261013__clean_v36\imaging_encoding\`;
`G` = `M:\MICROSCOPE\Priya\Widefield\labcams\grant_figures\`.

## 000. UPDATE 10-22 ~09:40 — read first (supersedes §2 "RUNNING")

**NOTHING of the encoding chain is running now.** What happened this morning:
1. **Duplicate runs found and stopped.** Four copies of `S/run_chain_1022.sh` (08:44 / 08:45 / 08:51 / 08:52) + the old
   `S/run_lp_encoder_lick.sh` were live (leftovers of the earlier pause / relaunches: killing a step's python lets its
   wrapper start the NEXT step). Three were on step 2 writing the SAME lick-aligned folder concurrently (~17 GB), then
   on step 3. All stopped (Priya ran the `taskkill /T` for the bash trees). **Integrity checked:** the lick-aligned folder
   `E/movement_first_subset-full_licks-combined_gate-attempted/` = 35 sessions, all json + npz (with `peth_lick`) +
   regions load, no torn writes (22 PS92 combined, 13 PS94_1005 / PS95 cam4). 3-D 2x2 folders were never created.
   Check script: `S/check_lick_folder.py <folder>`.
2. **The real frozen run (08:51) HUNG** in the pre-statistics phase: workers finished (~650 CPU-s each) but the main
   process never collected (one worker in an "Executive" kernel wait = stalled I/O on M:). Priya stopped it (and the
   post-frozen queue `S/run_after_frozen.sh`). `fit/pre_stats.pkl` + `fit/*.npz` on the share are from the 08:32 run =
   BEFORE the protrusion move (STATS_VERSION 3) -> stale; the restart recomputes them (no `--reuse-stats`).
   `<variant>/sessions/` hold 6 sessions from that stale run; they get overwritten.
3. **Efficiency audit (Priya: "are things parallelized and using cache/sidecars appropriately?"):**
   - Input cache works (hits: imaging 5 s + behaviour 5 s per session). One frozen `session_design` = **27 s** (the
     uncached remainder is `raw_blocks`, 14 s). A per-session design sidecar was REJECTED: X is 170k x 507 (0.35-0.7 GB
     / session, 35-70 GB for 107) to save ~10 min / run.
   - **Fits were serial in the main process** (4 animals x (8 variants + 18 drop-spout fits)) with the pool idle ->
     now one `fit_animal_job` per animal via `fan_sessions` (branch `frozen-parallel-fits` in worktree `S/wf_frozen`,
     test `test_fit_animal_job_equals_direct_fits`). Merge to main after the tests pass.
   - **Logs looked empty because stdout was redirected = block-buffered**: relaunch with `PYTHONUNBUFFERED=1` so
     `fan_out`'s per-item progress lines appear (a hang is then visible in minutes).
   - Relaunch plan: frozen at `--jobs 6` (44 GB free at 09:10); decompositions 4 at a time; then chain step 2
     (lick-aligned resume, `--skip-done`, 35 / 119 done) and step 3 (3-D 2x2, 2 workers) SIDE BY SIDE.
   - Expected: frozen ~30-40 min after relaunch (stats ~4 min, fits parallel, predict ~15-20 min); post-frozen queue
     (Musall figures, leakage, per-session maps every epoch) ~+30 min; step 2 ~3-4 h; step 3 ~2-3 h.
4. **Input cache backed up to the server** (Priya: "ensure caches are backed up to the server"):
   `C:\Users\SabatiniLab\wfield_cache` (106 GB, 720 entries) -> `M:\MICROSCOPE\Priya\Widefield\input_cache_backup\wfield_cache\`
   by `robocopy /E /XO` (copy only, never deletes; `*.tmp*` partials excluded); log `S/cache_backup.log`. Re-run the same
   command to refresh. To restore on a fresh box, copy back and set `WFIELD_INPUT_CACHE`; entries are keyed by code hash
   + env + source mtimes, so stale ones are simply never hit.
5. **FR pilot projections arriving** (e3): `fr_cam4_20261021/factors/pilot_pre_PS92_PS93/proj/` has PS92_20260818 and
   PS92_20260820 (acute, scored on the full pre basis). The watcher waits for PS92_0812 / PS92_0820 / PS93_0812.
6. Other windows asked to free RAM: 5b had nothing running; e3 stopped its output-check script + 2 watchers (pilot fit,
   pytest and fr JupyterLab kept).

### 000b. UPDATE 10-22 ~12:30

- **Relaunched 09:32** (`S/run_chain_1022b.sh`, task log `S/chain_1022b.log`): frozen done 11:07 (107 sessions, 0
  failures; slow share stretched it to 1.5 h), decompositions 11:10 (`E/qc/lp_encoder_decompose_frozen_encoder_gate-attempted_<v>[_lick]_by_animal.csv`),
  steps 2 + 3 running side by side since 11:10. Post-frozen queue `S/run_after_frozen_b.sh` (free-RAM gate lowered to
  8 GB) running: Musall figures, leakage, per-session maps.
- **Frozen LOSO R^2 (areas summed), pre:** joint .54 / .56 / .54 / .39 (PS92 / 93 / 94 / 95); no_movement .09 / .09 /
  .06 / .14; movement_only .50 / .52 / .50 / .34.
- **BUG TO CHECK before quoting task_first:** task_first should equal joint on all frames (same span) but breaks on some
  PS95 post sessions (PS95_0818 all-frames -1.06 vs joint .36; 0825 .02 vs .44; 0928 .26 vs .31; 0910 .40 vs .43) --
  likely the task-on-movement residualisation estimated on pre amplifying out-of-range post movement. Read joint /
  movement_first first; no task_first number or task_first drop-spout until fixed.
- **Why PS95's R^2 is lower (Priya asked):** not cam1 (cam4 licks for all: full .39 vs .56-.58); the gap is entirely
  "uninstructed" (alone .20 vs .39-.47; task alone is HIGHEST in PS95, .14). PS95 RUNS ~4x less (running regressor
  variance 0.30 of the cohort mean; pose energy normal) and its cortical variance is ~25 % lower. PROVISIONAL until the
  running-vs-still check (`S/run_vs_still_r2.py` -> `S/run_vs_still_r2.csv`; queued for >= 10 GB free). Consequence:
  compare animals by their own pre -> post change, not by shares / percentages.
- **Z-scoring (Priya asked): no.** Regressors are already scaled (pooled pre SD, frozen; per-session z, per-session);
  per-area R^2 is scale-free; per-session z-scoring of the targets would erase the post-stroke amplitude changes being
  measured. Only the "areas summed" summary / penalty choice weight areas by variance (optional: also report the
  equal-weight mean of per-area R^2).
- **FR pilot: LOCAL, not O2** (Priya: O2 fairshare ~4e-5; 51 sessions, FR / video / FR+video / FR+points models only = a
  few hours here). (1) one-session check PS92_0812 queued (`S/run_fr_pilot_one.sh`: FR pilot_pre_PS92_PS93, 10 point
  PCs, video cam1+cam4, ONLY fr,frvideo,frpts,frboth,video; waits for the running-vs-still check + 15 GB) -> REVIEW;
  (2) the 51-session run at 4 workers once the FR window's last PS93 projections land (42 / 51 at 11:49); if RAM is
  short ask Priya to pause step 3 (3-D 2x2; resumes with --skip-done); (3) O2 kept for the master_pre_v1 rerun.
- RAM: the FR window's projection jobs held ~30 GB at 12:00 (3 GB free) -- my queued jobs wait on free-RAM gates.
- Cache backup STOPPED (Priya: "prioritize the analyses over the cache backup"); it was loading the share (listing 17 s
  while it ran, 0.8 s after). Rerun after the runs, adding `C:\Users\SabatiniLab\.widefield_session_cache` and
  `C:\Users\SabatiniLab\frozen_models` (copy only) under `M:\MICROSCOPE\Priya\Widefield\input_cache_backup\`.
- Code: per-animal parallel fits merged to main (worktree branch `frozen-parallel-fits`; the running chain uses the
  worktree, which is at the same code).

**New pitfalls (add to §4):**
- **Stopping a bash wrapper does not stop its `( ... )` subshells** -- they carry on to the next step. Kill the whole
  tree: `taskkill /F /T /PID <wrapper>` (Priya runs kill commands); then list `python.exe` + `bash.exe` command lines to
  confirm. Never relaunch a chain without first confirming no older copy of it is alive.
- **Redirected python stdout is block-buffered**: an empty log is not evidence of a hang or of progress. Set
  `PYTHONUNBUFFERED=1` in chain scripts; diagnose a hang by per-worker CPU deltas (`Get-Process ... CPU` 20 s apart).
- `fan_out` has no per-task timeout: a worker stuck on an M: read blocks the whole run silently.

## 0. Rules added 10-21 night / 10-22 (all of the 10-21 §0 still holds)

- **LP_licks results are PROVISIONAL until cam1 finishes** (74 / 108 curated sessions have cam1; PS95 none, PS94_1005
  none -> cam4-only LP licks there). Rerun `scripts/trial_attempts.py` first when cam1 lands.
- **Outline reading rule (ENCODING_FINDINGS 3.3):** nested-bootstrap outlines localise (bin-level FWER); their extent is
  a power-dependent LOWER BOUND, not an area, and "x % vs y % of bins" is descriptive. Cluster outlines = yes / no only.
- **Per-session encoder "movement" includes spout CONTACT and lick DIRECTION** (both inside its `licks` / `instructed`
  terms); do not read a per-session movement share as "movement" -- use the frozen encoder's FGROUP split.
- **Directionless movement = timing + vigour only** (lick onsets, tongue speed, jaw opening / speed, nose y, running);
  tongue PROTRUSION is spatial (scales with spout distance) and sits with direction (10-22 audit).
- **FR carries position** (lick factors are left / right / centre): never headline "does position survive FR";
  bracket it (FR first vs position first) and use the within-spout test.
- Memory on this box: the FR window runs `face_rhythm_factors.py` locally (3-6 GB, growing); keep my runs at 4 workers
  and check free RAM (`python S/freegb.py`) before adding jobs. Pausing a run = stop the bash wrapper FIRST, then kill
  the python tree by command line (parent + `ParentProcessId` children), else the wrapper starts its next step.

## 1. Results since the 10-21 handoff (details ENCODING_FINDINGS 3.3, DECISIONS)

1. **LP map render + calibration** (`G/LP_licks/epoch/`, `.../results/calibration_LP_vs_DAQ_summary.csv`): acute far
   contra drawable (4 an / 15 sess; contact gating had 0); post-lick activity BELOW pre (amplitude 0.70; 19.7 % of bins,
   drift <= 1.4 %), 3/4 bilateral (PS95 1 session opposite). Contact gating had INFLATED acute far-spout amplitudes.
   Pure-null false positives: nested 1.6 % of 312 tests (mostly 1-7 bins), cluster max 27 % area.
2. **LP frozen decoders** (`G/LP_licks/epoch/epoch_{acc,accdelta,5c,5cr}_*`): acute far contra 0.50 -> **0.32** (4/4),
   far ipsi .72 -> .58, far mid .71 -> .51; cue window unchanged. Confusion: acute far contra mis-decoded as FAR IPSI /
   far mid (distance kept, side lost).
3. **Matched contact 1.0 s ENL arm** (`G/DAQ_ENL1s/epoch/`): decoding = LP 1.0 s within .03 (lick definition irrelevant at
   1 s; the 2 s -> 1 s drop is window length). ENL far-contra map change 9.9 % (contact 1 s) -> 4.4 % (LP 1 s).
4. **LP-gated per-session encoder** (`E/movement_first_subset-drop_licks-combined_gate-attempted/`, 119 sessions;
   `scripts/lp_encoder_decompose.py` -> `E/qc/lp_encoder_decompose_by_animal.csv`): acute far-contra drop carried by the
   per-session "instructed" terms 4/4 -- BUT those include contact + direction (§0); pending the frozen split.
5. **Encoder variant review** (all variants, what each tested / showed): in the 10-22 conversation; main open
   contradiction = cam4 post-stroke executed rise vs 3-D polar "rise is target". Being tested by the 3-D 2x2 (§2).
6. **FR plumbing test** (`E/movement_first_fr-preview_r10_PS92_subset-fr/`, PS92_0812, preview set, IN-SAMPLE): path
   works; FR adds a lot (all frames .56 -> .66-.68; lick .36 -> .53; FR alone beats the full model in the ENL window,
   .65 vs .41); position adds ~.01-.02 beyond FR (expected -- FR carries side).
7. Checks: per-session "alone" models DO re-tune penalties (on all frames); jaw-alone lick-window R2 < 0 = fast jaw traces
   overfitting slow GCaMP, not a bug. Task-only frame R2 ~0.10 matches Musall (single-trial variance is movement-driven).

## 2. RUNNING / QUEUED (10-22 ~midday)

| what | how | output / then |
|---|---|---|
| **chain** `S/run_chain_1022.sh` (bash task in this session) | (1) `frozen_encoder --all --jobs 4` from worktree `S/wf_frozen` (detached at main) with OROFACIAL_GATE=attempted, LICKS=combined, ONLY=full, SUFFIX=_gate-attempted (input-cache env), then `lp_encoder_decompose.py --folder frozen_encoder_gate-attempted/<v> --align {cue,lick}` for 8 variants; (2) RESUME the lick-aligned per-session run (`--skip-done`, 21 / 119 done before the pause) + its cam4 pass; (3) 3-D 2x2 (xyz2 x {combined, cam4}, polar x cam4; 47 PS92 + PS93 sessions; 20261021 tables; soft bins; reference gate; `--only nokin:`) | (1) `E/frozen_encoder_gate-attempted/<variant>/sessions`, `dropspout/`, `frozen_musall_r2*.csv`, `fit/pre_stats.pkl`, log `S/frozen_encoder.log`; (2) `E/movement_first_subset-full_licks-combined_gate-attempted/` (`peth_lick`); (3) `E/movement_first_subset-nokin_exec-{xyz2_licks-combined,xyz2,polar}_bins/`, logs `S/3d_2x2_*.log` |
| **post-frozen** `S/run_after_frozen.sh` | waits for "frozen decomps done" + >= 20 GB free; `frozen_musall_figures.py`; `frozen_group_leakage.py --jobs 2`; `combined_encoding_figures.py --order movement_first` (Musall Fig. 4 bars + maps for EVERY epoch now) | `E/frozen_encoder_gate-attempted/figures/musall_*.png`; `E/qc/frozen_group_leakage.csv`; `E/movement_first/figures/fig4_*_{pre,acute,subacute,chronic}_*.png` |
| FR pilot watcher (python loop in this session) | polls `fr_cam4_20261021/factors/pilot_pre_PS92_PS93/proj/` for PS92_0812 / 0820, PS93_0812 every 10 min (24 h) | then §3 step FR |
| push of 2e7ebfe ... b530065 (frozen drop-spout, Musall readouts, figures, leakage) | retry loop | verify by FILE CONTENT on origin/main (§4) |

## 3. NEXT (in order)

1. **Read the frozen run** (per animal first): FGROUP shares of the acute far-contra post-lick drop (cue- and lick-
   aligned), data - full (does cortex respond LESS than the pre-stroke mapping predicts?), no_movement -> task_movement ->
   task_movement_dir -> joint increments, ordered bounds; frozen drop-one-spout far contra under joint / task_first /
   movement_first (`dropspout/frozen_dropspout_by_animal.csv`); Musall readouts (`frozen_musall_quantities_by_animal.csv`).
   Leakage table: directionless groups should decode the spout near chance (1/6). Write into ENCODING_FINDINGS 3.3.
2. **Link figures for Priya** (she asked: "both r2 graphs and maps"): frozen `musall_*` + per-session `fig4_maps_<epoch>_*`.
3. **Frozen MAPS** (Priya: "i guess frozen maps?" + "also re-rendered per-session maps"): add a grid target to
   `frozen_encoder` (Rd on the 8x stat grid, 2,022 bins, instead of 42 areas) -> per-bin Musall quantities + far-contra
   decomposition maps; run after the chain with <= 3 workers (memory).
4. **FaceRhythm pilot** (`pilot_pre_PS92_PS93`, PS92 + PS93, 51 sessions; pre sessions on a leave-one-day-out basis,
   post on the full fit): one local session; then `o2_encoding.py bundle --name enc_fr_pilot_<date> --fr
   pilot_pre_PS92_PS93 --fr-points 10 --cams cam1,cam4` + `check`; Priya submits. Readouts: FR vs ME vs both; position
   bracketed (FR first / position first) + within-spout FR; merged-lick FR (preview names: lick factors 0,1,2,4,7,9;
   3 = lick_onset_nose separate) once pilot factors are matched to preview names by spatial-spectral cosine (Priya
   confirms) + spout-decode leakage of the merged trace. Two-animal pilot; rerun on `master_pre_v1`.
5. **3-D 2x2** (when the chain reaches it): which factor flips PS92 acute (coordinates vs lick set).
6. Lick-aligned per-session decomposition (`lp_encoder_decompose.py --folder movement_first_subset-full_licks-combined_gate-attempted --align lick`).
7. LP_licks leftovers: PS93 far_ipsi / far_mid frozen sensitivity (`WIDEFIELD_LP_FROZEN=lp`); drop-one-spout under the LP
   gate (per-session, ~64 GB / session: subset only); full rerun when cam1 finishes.
8. Carried over: 10-21 §3 items 4-11.

## 4. Pitfalls (new 10-21 night / 10-22)

- **Pushes race the other windows.** The pre-push suite takes ~8-10 min; another window's push in that time -> "cannot
  lock ref". Their push often CARRIES my commits (rebased) -- check `git show origin/main:<path> | grep <new symbol>`.
- **Missing test dependency:** the histology tests need `shapely` (declared in pyproject); installed 10-22 (2.1.2) -- a
  fresh env fails the pre-push suite without it.
- **Rule-9 guard** (`tests/test_parallel_results_are_ordered.py`) requires an explicit sort within 14 lines of every
  `fan_out` / `fan_sessions` call -- a helper that returns raw `fan_out` results fails the push.
- **`combined_encoding.out_dir` does NOT encode the gate** (OROFACIAL_GATE): tag runs with `COMBINED_ENCODING_SUFFIX`.
  The input-cache key includes every `COMBINED_ENCODING_*` / `OROFACIAL_*` env var, so match the env to reuse the cache.
- **`COMBINED_ENCODING_LICKS=combined` raises for sessions without cam1** (PS95 all, PS94_1005): run those with LICKS
  unset and `COMBINED_ENCODING_SUFFIX=_licks-combined<rest>` to land in the same folder (json `lick_set` says which).
  `frozen_encoder` falls back automatically.
- **Editing `combined_encoding.py` invalidates the input cache** (cold reloads ~100 s / session). Added 10-22:
  `peth_lick` (lick-aligned PETH) -- the edit made the next runs cold.
- **Bash heredocs mangle `\n` AND `$`-escapes**: patch Python / shell with the Write / Edit tools (happened 3x today).
- **PowerShell `-File` scripts are blocked** (execution policy): use `python S/freegb.py` for free RAM.
- `lp_encoder_decompose.py` names outputs by folder when `--folder` is not the default (variants no longer overwrite).
- A frozen-encoder layout change (FGROUP) makes `fit/pre_stats.pkl` stale: bump `STATS_VERSION`, rerun without
  `--reuse-stats`.

## 5. New code (all on main; tests in `tests/test_frozen_encoder.py`, 7)

- `wfield_local/frozen_encoder.py` -- per-animal pre-stroke ridge in Allen-area space; pooled-pre scaling; FGROUP (task /
  movement / direction / contact); variants joint, movement_first, task_first, no_movement, task_movement,
  task_movement_dir, movement_only, no_task; frozen drop-one-spout (`<base>|drop:<spout>`); Musall readouts (frame /
  single-trial / trial-averaged R2, cue- and lick-aligned); `--reuse-stats`, `--summarise-drop`.
- `scripts/lp_encoder_decompose.py` (`--align cue|lick`, `--folder`), `scripts/encoder_trialavg_r2.py`,
  `scripts/frozen_musall_figures.py`, `scripts/frozen_group_leakage.py`; `combined_encoding` saves `peth_lick`;
  `combined_encoding_figures.py` renders Fig. 4 for every epoch.
