# Handoff — 2026-10-22: LP_licks results, frozen pre-stroke encoder, Musall-style readouts, FaceRhythm pilot

**START HERE for the encoding model (the ENCODING window, session `sabatinilab-d4`).** Supersedes
`STATUS_2026-10-21_ENCODING_HANDOFF.md` (still the reference for §0 rules, §4 pitfalls, §5 commands and the 10-21 work).
Other windows: pose / 3-D / cam1 = `sabatinilab-5b` (`STATUS_2026-10-21_POSE_CAM1_JAW_HANDOFF.md`); FaceRhythm =
`sabatinilab-e3` (`STATUS_2026-10-21_FACE_RHYTHM_O2_HANDOFF.md`). **Results: `docs/ENCODING_FINDINGS.md` §3.3** (updated
10-22). Reasoning: `DECISIONS.md` 2026-10-21 (night) and 2026-10-22 entries. `S` = scratchpad
`C:\Users\SABATI~1\AppData\Local\Temp\claude\C--Users-SabatiniLab\cdd1e016-6bb3-4346-8600-f08116ccf61d\scratchpad\`;
`E` = `M:\MICROSCOPE\Priya\Widefield\LP_analysis\lp_r4_heatmap_20261013__clean_v36\imaging_encoding\`;
`G` = `M:\MICROSCOPE\Priya\Widefield\labcams\grant_figures\`.

## 000g. UPDATE 10-23 ~15:00 (read with 000f)

- **3-D NOW COVERS ALL FOUR ANIMALS** (pose window sabatinilab-5b, 10-23): 124 sessions (PS92 35, PS93 31, PS94 30,
  PS95 28); cam1 125 / 127 curated (PS95_0817 / PS95_0831 still on O2; they'll message when landed). jaw_onsets.csv,
  kinematics/licks.csv.gz, cue traces, spout error, phase all include PS94 / PS95. **Our 3-D inputs are the 10-21
  snapshots** (`S/pose3d_kin_licks_snapshot_20261021.csv.gz`, `S/pose3d_spout_positions_PS92_PS93_20261021.csv`,
  `S/exec_bins_calib_20261021.csv`, `S/labels_3d.txt` = PS92 + PS93) -> **the standing re-run rule is DUE**: re-cut 3-D
  tables (incl. spout positions + bin calibration for PS94 / PS95), re-run all lick-direction results on 4 animals, and
  refresh combined lick timing (PS94 / PS95 cam1) in the frozen encoder / LP_licks families. The running 3-D 2x2 is on
  the snapshot (2 animals): let xyz2 x cam4 finish, then replace polar x cam4 with the 4-animal re-run.
- Per-frame 3-D spans nearly the whole session (`<dlc>/session_poses/pose3d/<s>_frames.csv.gz`, cam1 windows from the
  previous trial stop); per-lick ITI 3-D: asked the pose window for `pose3d_kinematics --licks all` (Priya approved
  spontaneous licks for the stray test) -- they'll send path + trial/ITI column.
- FINDINGS updated (status box, decoder correction, 2.6 strays); DECISIONS 2026-10-23 (afternoon) correction entry;
  `scripts/frozen_running_compare.py` (41650de). FaceRhythm + pose windows asked to do the same findings pass.

## 000f. STATE 10-23 ~14:05 — READ FIRST (then 000e / 000d; decisions DECISIONS 2026-10-23; results FINDINGS 3.4)

- **RUNNING:**
  - **v7 candidates** (started 13:36, ETA ~15:45): `S/run_v7cand.sh` from `S/wf_v5` (detached at 215f0ff), Allen areas,
    2 workers each: `E/frozen_encoder_gate-attempted_v7fwd15` (FROZEN_RUNNING=spline_thr, FROZEN_RUN_KNOT0=1.5 =
    forward-only running from 1.5 mm/s) and `_v7fwd15push` (+ FROZEN_PUSH=1). Logs `S/frozen_v7<k>.log`,
    `S/v7_chain.log`; joint decompositions after.
  - **3-D 2x2** (`S/run_3d_resume.sh`, `S/wf_v4`, 2 workers): xyz2 x combined DONE 09:32; xyz2 x cam4 38/47 at 14:00;
    then polar x cam4 (47). No failures.
- **WHEN v7 CANDIDATES LAND:** compare with `_v6` (signed linear), `_v6spl`, `_v6thr3`, `_v6thr1` (all done): per animal
  x epoch frozen R2 (all / cue / lick, pre held out), running curve (`fe.running_curve(..., mode="spline_thr")` with
  `FROZEN_RUN_KNOT0` set), unique dR2 of push and of lick with push in / out, the acute far-contra decomposition.
  Pick v7 -> make it the DEFAULT in code (STATS_VERSION 7: default FROZEN_RUNNING=spline_thr + knot0 1.5 [+ push]),
  test, commit; then the **basis re-run** (`--target basis`, all animals; it now fits drop variants + task_only +
  drop-spout and saves resid / data moments incl. whole-spout swaps) -> `scripts/frozen_r2_maps.py --tag <basis tag>`
  and `scripts/frozen_spout_maps.py --tag <basis tag> [--contrast]`; ~1.5-2 h + maps.
- **THEN (queue, 000d / 000e):** 2b lick-level strays vs on-target (pre + post); 3 all-session shared + per-epoch model;
  4 FR inside frozen; 6 session-level competition LMM; 7 frozen parts onto the TIME / BEHAVIOUR LMMs; no_running /
  no_nose variants if PS95 windows still need it.
- **10-23 afternoon results** (FINDINGS 3.4 running paragraph): running sublinear (~on / off), forward-only from 1-1.5
  mm/s beats linear everywhere, signed linear treated backward pushes as negative running; side-camera clips: bout-end
  negative speed = paw push, post-cue dip = upper-limb extension (real) -> push variable; jitter check supports 1.5
  mm/s; headline unchanged in every running model. Whole-spout per-position code 9ac1e60 (smoke `_smoke-r2b`: ~5x the
  target-only scale). QC / clip scripts now in the repo: `scripts/qc_running_{speed,clips,lick_jitter,survey}.py`.
- **Worktrees:** `S/wf_v5` = v7 candidates (busy until ~15:45); `S/wf_v4` = 3-D 2x2 (busy ~1 day); `S/wf_frozen` idle.

## 000e. STATE 10-23 ~07:00 (superseded by 000f for state)

- **RUNNING** (worktrees in `S/`; never edit them while a run uses them):
  - `_v6spl` = sublinear running test (`FROZEN_RUNNING=spline`, commit 5045ef5), Allen areas, 3 workers, from
    `S/wf_v5` (detached at 5045ef5); `S/run_v6spl.sh`, progress `S/v6spl_chain.log` / `S/frozen_v6spl.log`; then
    joint decompositions. READ: per-animal running curve (`running_curve` on fit/<an>_joint.npz betas + templates
    knots), held-out pre R2 vs v6, post R2 per animal vs v6 (PS94 the test case), headline unchanged?
  - 3-D 2x2 RESUMED 10-23 05:05 (`S/run_3d_resume.sh`, from `S/wf_v4`, 2 workers, `--skip-done`; progress
    `S/3d_resume_1023.log`, cells `S/3d_2x2_<exec>_<licks>.log`): xyz2 x combined 25 remaining, then xyz2 x cam4 and
    polar x cam4 (41 each) -- ~1 day at 2 workers; can go to 3-4 when memory frees (workers reach ~27 GB).
- **10-23 morning findings / decisions** (details FINDINGS 2.4c correction, 4.3; queue in 000d):
  - Destination-vs-landing % were inflated (target == landing rows counted as target); separating rows only:
    cue 32 / 2 / 8, LICK-aligned 33 / 2 / 2 (target / execution / neither); strong (own-zone < 0.15) 5 / 0 / 1 lick-
    aligned -- target-leaning, rests on ~6 acute far-contra sessions, 2 animals (lick arm:
    `qc/best_match_destination_lick_lick.csv`; cue copy `_cue-working`).
  - Target and the spout-locked lick are the SAME variable on on-target licks: pre-stroke per-spout contrast maps
    (`scripts/frozen_spout_maps.py --contrast`; `E/..._v6basis/figures/pre_spout_contrast_*`) show the data
    lateralised as expected in 4/4 (contra spouts -> left hemisphere, SSp-m L-R +12..+21; ipsi -> right), the SPOUT
    part (target + spout-locked) tracking it, and the target / exec_locked split unstable (close_mid +13 vs -13).
    -> per-position maps: add the whole-spout version; target vs executed only via STRAYS (queue 2b) and the bracket.
  - Lick-direction dR2 exists only as exec_within unique dR2 (whole-cortex ~1 % pre; PS94 post negative in windows,
    possibly out-of-range direction deviations); per-lick version = queue 2b.
  - Stray census (good 3-D, August): thousands of no-contact licks nearer another calibrated centre, pre included
    (PS92 1129, PS93 6679 pre); close-spout centres overlap -> continuous error, not zones. Spontaneous ITI licks =
    execution-only reference once whole-session 3-D exists.
  - LMM with 2 animals: animal FIXED (a random-effect variance cannot be estimated from 2 levels); per animal first.
  - PS95 post-stroke window R2 ~0 with "other" in: running vs nose NOT separable with the current variants (they drop
    together) -- needs no_running / no_nose variants (cheap, saved statistics).
- **Stale in 000d below:** "3-D 2x2 still paused" (resumed, above); "PS95 ... nose y?" (see the line above).

## 000d. UPDATE 10-22 NIGHT (then 000c)

- **Frozen encoder is now v6 (STATS_VERSION 6, commit 34c803d)**: running clipped at each animal's pre-stroke p99 of
  RUNNING frames (> 3 mm/s, `segmentation.running.thresh_speed_mm_s`); caps PS92-95 14.8 / 17.3 / 13.2 / 16.8 mm/s.
  Use `E/frozen_encoder_gate-attempted_v6/` (areas), `_v6basis/` (pixel maps, all animals), `_v6grid/` (PS92 check).
  v5 folders (all-frame p99 cap: broke PS95) carry SUPERSEDED.txt; v4 kept for comparison.
- **PS94 "OPEN" below is RESOLVED** by the cap (chronic joint all-frames R2 -0.44 -> +0.30; FINDINGS §3.4 v6 para).
  Headline unchanged (acute far-contra parts within 0.2 of v4 in all orderings).
- **Pixel maps done** (basis + grid, joint / target_first / exec_first); they match the area maps.
- **PS92_0908 FR timing: RESOLVED 10-23 ~07:35** (FaceRhythm window: cam4 dropped ~13 % of frames, the old linear
  frame->DAQ fit was off by median 3.2 s; corrected projections carry `timing = fr-timing-v1`, lossy bins NaN). PS92_0812
  and PS92_0908 rerun in FR run A (old outputs `_superseded_linear_fr_timing/`; note `RESOLVED_PS92_0908_fr_timing.txt`):
  PS92_0908 fr:full cortex R2 0.506 -> 0.618 (FR alone +0.26); PS92_0812 unchanged (mistimed bins were after the DAQ
  recording). Both sessions back IN. Any FR set projected before 10-23 without the timing field: check the same way.
- **FR run A done** (50/50 PS92 + PS93 sessions; `S/fr_runA.log`) -- summarise how much grouped FR adds; the cam4
  pass exited instantly with an empty log (check it is "all done", not a skip by error).
- **Still open:** PS95 subacute / chronic cue / lick-window R2 ~0 (no_other better: nose y?); contact kernel sign; test
  on data - model; PS95 1 acute session. **3-D 2x2 still paused** at 16/41 (`run_3d_2x2.sh`, `--skip-done`, 2 workers).
- **Next:** FR inside frozen v6 (PS92 + PS93; FR traces through the spl / dev split, own penalty group, env flag);
  FR run A summary; resume 3-D 2x2; cache backup when idle.
- **10-23 queue (Priya-approved, reordered 10-23 "yes reorder"):** (1) sublinear running test `_v6spl` (running);
  (2) basis re-run with pixel unique R2 + per-position (target-only AND whole-spout = target + spout-locked template
  swapped for the other spouts' mean at prediction) + direction maps (`scripts/frozen_r2_maps.py`; code fc6d441) on
  the winning running model; (2b) **LICK-LEVEL STRAYS vs ON-TARGET, within spout -- the decisive target-vs-executed
  test** (Priya: "don't we want to be comparing LICK stray response vs on-target response"; target and the spout-locked
  lick are the same variable on on-target licks, only strays separate them): per lick, continuous 3-D error from the
  aimed spout's CALIBRATED centre (pre contact licks; nominal angles are biased ~21 deg) in within-spout SD units +
  direction toward the nearest other centre + contact; per-lick regressor sets (on-target at q; strays from q toward j)
  so overlapping corrective licks are separated by the model (raw stray averages are contaminated toward target);
  compare stray kernels' maps with pre on-target maps of q and j, controlling pre q-j similarity = competition LMM at
  LICK level, pre AND post. **The PRE-STROKE arm is a result in itself** (Priya: "pre-stroke variation can also be useful
  to us (allow us to look at executed vs target pre-stroke also)"): stroke-independent target vs executed -- settles the
  pre-stroke disagreement between xyz2 (executed >= target, PS92) and polar (comparable / target-led) by strays rather
  than by coordinate choice; pre sessions scored leave-one-session-out; lick-level check of the within-spout crossed
  SSp-m direction code (pre 4/4); PS93's high pre stray rate (21 % vs PS92 5 %) may reflect its known right orofacial
  deficit -> per animal first. Census 10-23 (good 3-D, August pre, PS92 / PS93): no-contact licks nearer another spout's
  centre PS92 1129 / 1054 / 665 / 2645, PS93 6679 / 1783 / 2320 / 5429 (pre / acute / sub / chronic); far contra acute
  564 / 537 -- thousands, vs ~6 sessions in the session-level test. Close-spout centres overlap (3-10 deg apart vs
  within-spout SD 6-9 deg: 5-13k CONTACT licks sit nearer another centre) -> use the continuous error, categorical zones
  only for well-separated spouts. Spontaneous (ITI) licks = an EXECUTION-ONLY reference (no target) once whole-session
  3-D exists (ask the pose window); pre-cue licks are still targeted (spout already in place).
  (3) **ALL-SESSION model: shared kernels + per-epoch deltas** (Priya: "i like the shared kernels with per-epoch
  changes idea!"): per animal, kernel = shared + acute / subacute / chronic delta (ridge-shrunk to 0), pre-stroke
  templates kept as the reference frame, epoch weighting, held-out-session scoring for every session, DRIFT NULL =
  early- vs late-pre pseudo-epoch; readout = per group, does an epoch delta improve held-out prediction (tests
  "coupling preserved") + delta maps; built on the Gram statistics (one stats pass over post sessions); areas then basis;
  (4) FR inside frozen; (5) [folded into 2b]; (6) SESSION-level competition LMM for destination vs landing
  (secondary to 2b; per (session, aimed, candidate) pattern correlation ~ target + 3-D landing weight + pre-stroke
  spout similarity, x log days / deficit; animal FIXED with 2 animals); (7) frozen v6 per-session parts onto the
  TIME / BEHAVIOUR LMMs (FINDINGS 4.3 item 7).
  **Target-vs-executed headline under verification (Priya 10-23):** FINDINGS 2.4c destination % were inflated (see the
  correction there); lick-aligned arm running; 2x2 decides the coordinate frame.
- **RE-ADDRESS WHEN 3-D TRACKING COVERS ALL SESSIONS (Priya 10-23).** 3-D tables exist for PS92 + PS93 only
  (20261021 snapshots); PS94 / PS95 direction is cam4 2-D, PS95 licks cam4 (cam1 unfinished). Everything that
  depends on lick DIRECTION must be re-run then and compared: exec_within / exec_locked groups and their unique dR2
  (incl. PS94's negative post-stroke exec_within dR2 -- possible out-of-range extrapolation of direction deviations),
  the strayed-lick direction test (planned 10-23: per-lick SSE gain from direction vs a within-spout shuffle of
  direction deviations, gated on tracking quality, matched on vigour / contact, per animal), the 3-D target-vs-executed
  contests and 2x2, within-spout per-position direction, and the direction pixel maps. LP results stay provisional
  until cam1 finishes.

## 000c. UPDATE 10-22 EVENING (supersedes 000 / 000b where they differ)

**Results: ENCODING_FINDINGS §3.4** (frozen v4). Reasoning: DECISIONS 2026-10-22 (afternoon / evening).

**State of the frozen encoder (`wfield_local/frozen_encoder.py`, STATS_VERSION 4, main 8cf72f4+):**
- **v4 groups by construction** (Priya: "should we have a 'lick' group and a 'spout position information' group?" ->
  build it): each lick-related movement trace = the animal's PRE-STROKE mean trace for that spout (cue-locked before
  the first post-cue lick, lick-locked after; POOLED over its pre sessions -- Priya "let's do pooled") + the deviation.
  Groups timing / target / exec_locked / exec_within / lick / other / contact; super-groups task / spout / movement.
  Variants: joint, target_first / exec_first (target vs executed bracket), task_first / movement_first, no_<g>,
  <g>_only, task_only. Fit dir has `templates.pkl` (pass 1, behaviour only) + `pre_stats.pkl` (stats_version 4).
- **Ordered-projection fix**: thresholded pseudo-inverse, PROJ_RCOND = 1e-3 (1e-6 ridge and then 1e-8 both blew up on
  PS95 post sessions: task_first -1.06, target_first -163). After the fix every ordered variant matches joint within
  0.003 on all frames, every session.
- **Run done**: `E/frozen_encoder_gate-attempted_v4/` (107 sessions, re-scored 16:55 with the fix), decompositions
  `E/qc/lp_encoder_decompose_frozen_encoder_gate-attempted_v4_<variant>[_lick]_by_animal.csv`, Musall figures
  `.../figures/musall_*.png` (new groups, STACKED task-aligned figure per Priya), **area maps**
  `.../figures/area_maps_{joint,target_first,exec_first}_far_contra_{acute,subacute,chronic}_{lick,cue}.png`
  (`scripts/frozen_area_maps.py`). **v3 folder `frozen_encoder_gate-attempted/` = SUPERSEDED** (kept).
  Leakage tables: `E/qc/frozen_group_leakage.csv` (v4, lick-aligned pooled templates, held out),
  `..._v3_SUPERSEDED.csv`, `..._v4cuelocked_SUPERSEDED.csv`. Smoke folders `frozen_encoder_smoke-*` = plumbing only.
- **Headline (provisional, §3.4)**: acute far-contra post-lick drop is carried by LICK in all five orderings (PS92
  -20 to -21, PS93 -12 to -14, PS94 -13.5 to -14, PS95 -20, dF/F x 1e3); target <= 1.7 even with target_first; data -
  prediction small and positive. = less licking-related activity because of changed licking, not less activity per
  lick (correlational: coupling preserved, direction not determined).
- **OPEN, PS94**: every PS94 chronic session fails under the frozen model (all-frames R2 -0.41 to -0.62; per-session
  fits 0.50-0.62); dropping the "other" group (nose y, running) restores +0.11 / +0.12 -> a running or nose-y change
  after the stroke. A running-speed check (pre vs subacute vs chronic, PS94 vs PS92) was running at handoff time
  (scratch `S/` task). Do not read PS94 chronic frozen numbers until resolved. Also open: contact kernel sign (contact
  part opposes lick), no statistical test on data - prediction, PS95 has 1 acute session in this window.

**Pixel maps (Priya: "go ahead with the area map and then pixel maps"; chose the literature route):**
- `--target basis`: per-animal SHARED SVD basis on the Allen-aligned stat mask (all sessions share the 540 x 640 grid,
  129,770 stat pixels; each pre session contributes Ub Ue sqrt(w)); top components to 95 % of pooled pre variance
  (<= 200) + the cortex mean as targets. `--target grid`: 8 x 8 px block means (~2,000 blocks) = the transparent check.
  Spatial runs store EVOKED responses per part x spout x target (no full PETHs; lazy projections, no frames x targets
  arrays), R2 / Musall on the cortex mean, no drop-spout. Renderer `scripts/frozen_pixel_maps.py` (written, untested
  at handoff). Per animal not cohort basis (Priya asked; per animal fits each animal best, maps are averaged in pixel
  space anyway; a cohort-basis variance check is a cheap optional diagnostic).
- **NEXT**: the 3-session smoke test (`--tag smoke-basis` / `smoke-grid`, PS92_0806 / 0812 / 0820) was running at
  handoff; then full runs `--target basis --tag gate-attempted_v4basis` (all animals) and `--target grid --tag
  gate-attempted_v4grid` (PS92 only, the check), then `frozen_pixel_maps.py` for each; compare grid vs basis and both
  vs the area maps (bins averaged within areas should match).

**FaceRhythm in the encoder (Priya approved each step):**
- `wfield_local/fr_groups.py` + an opt-in hook in `face_rhythm_outputs.fr_signals` (env `COMBINED_ENCODING_FR_GROUPS=
  pilot_v1:A|B`, default off; keyed to set `pilot_pre_PS92_PS93`, refuses others). pilot_v1: lick amplitude = f0 + f4 +
  f5 + f9 (lick-locked in every check); direction left f0 + f5, centre f4, right f9; other f1 f2 f3 f6 f7 f8 (f2 tonic
  with licking but not lick-locked -- the FR window's share-based r -0.24 vs absolute +0.04-.24 is a units difference;
  f1 / f3 / f6 = lick INITIATION, so "other" is not "non-lick"). labels.json now written by the FR window (named
  columns; harmless: the hook re-keys by position).
- Directionless FR does NOT exist as a factor subset: FR lick sum decodes the spout .15 / .15 (PS92) but .41 / .26
  (PS93; SD-balancing does not help -- PS93 licks different amounts at different spouts), and the six "other" factors
  together .31-.57 (each alone near chance). So run B (lick + other) was DROPPED; the clean route = FR inside frozen v4
  (spout-locked + deviation split of the FR traces), PS92 + PS93 on the pilot set -- NOT BUILT YET (next).
- **FR run A** (grouped FR lick + sides + other, ONLY=fr, 50 pilot sessions, 2 workers, cam4 fallback pass) RUNNING
  overnight from `S/run_evening_1022.sh` (log `S/evening_1022.log`, `S/fr_runA*.log`) -> folder
  `movement_first_fr-..._subset-fr_licks-combined_gate-attempted_frA`. Gives how much FR adds and position beyond all
  of FR (lower bound).
- O2 not used (fairshare ~4e-5); O2 kept for the master_pre_v1 rerun (all animals).

**Other state:** step 2 (lick-aligned per-session run) COMPLETE, 119 sessions; its lick-aligned decomposition ran
(`E/qc/lp_encoder_decompose_movement_first_subset-full_licks-combined_gate-attempted_lick_by_animal.csv`, not yet
read). 3-D 2x2 PAUSED (its two workers took 27 GB each next to FR run A; resume with --skip-done, 2 workers, alone).
PS95 R2 is lower because PS95 runs ~1 % of task frames vs 4-7 % (running-vs-still check, `S/run_vs_still_r2.csv`):
on still frames it matches the others -> compare animals by their own pre -> post change. No z-scoring of targets
(per-area R2 is scale-free; per-session z would erase the amplitude changes being measured). Cache backup paused
(analysis first); rerun `robocopy /E /XO` of `C:\Users\SabatiniLab\wfield_cache`, `.widefield_session_cache`,
`frozen_models` -> `M:\MICROSCOPE\Priya\Widefield\input_cache_backup\` when the box is idle.

**New pitfalls (10-22 afternoon):** `DataFrame.align` is a pandas METHOD (`Q.align == "cue"` was always False: every
Musall figure silently skipped) -- use column indexing; a model part named "pos" collides with the spout column
(rename to pos_part, twice now); bash heredocs still mangle `\n` (again in the figure title); each long-running worker
pool on this box can reach 25-30 GB per worker (2x2, FR with points + video): two such pools do not fit -- run them one
at a time and check `freegb.py` + per-process private bytes, not working set.

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
